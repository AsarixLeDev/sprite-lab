"""Trusted acquisition contracts and the separately certifiable local adapter."""

from __future__ import annotations

import ast
import glob
import hashlib
import importlib.metadata
import io
import os
import platform
import re
import ssl
import stat
import sys
import sysconfig
import time
import unicodedata
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog import SHA256_PATTERN, HarvestSource
from spritelab.utils.safe_fs import AnchoredDirectory, open_anchored_directory, require_confined_path

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
MIME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
MAX_HARDENED_MODULE_BYTES = 4 << 20
MAX_RUNTIME_FILE_BYTES = 2 << 30
MAX_LOADED_MODULES = 4096


@dataclass(frozen=True)
class HarvestLimits:
    """Server-owned hard limits; browsers cannot increase them."""

    max_files: int = 5000
    max_file_bytes: int = 16 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024
    max_response_bytes: int = 512 * 1024 * 1024
    max_depth: int = 8
    max_duration_seconds: float = 1800.0
    max_events: int = 1000
    max_event_bytes: int = 4096
    max_redirects: int = 5
    max_archive_members: int = 10000
    max_archive_uncompressed_bytes: int = 1024 * 1024 * 1024
    allowed_artifact_mime_types: tuple[str, ...] = (
        "image/gif",
        "image/png",
        "image/webp",
    )
    allowed_response_mime_types: tuple[str, ...] = (
        "application/octet-stream",
        "application/zip",
        "image/gif",
        "image/png",
        "image/webp",
    )

    def __post_init__(self) -> None:
        integer_values = (
            self.max_files,
            self.max_file_bytes,
            self.max_total_bytes,
            self.max_response_bytes,
            self.max_depth,
            self.max_events,
            self.max_event_bytes,
            self.max_redirects,
            self.max_archive_members,
            self.max_archive_uncompressed_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in integer_values):
            raise ValueError("Harvest limits must be positive integers.")
        if not 0 < self.max_duration_seconds <= 86400:
            raise ValueError("Harvest duration limit is invalid.")
        if self.max_total_bytes > self.max_archive_uncompressed_bytes:
            raise ValueError("Harvest total output bytes cannot exceed the archive expansion limit.")
        for mime_type in (*self.allowed_artifact_mime_types, *self.allowed_response_mime_types):
            if MIME_PATTERN.fullmatch(mime_type) is None:
                raise ValueError("Harvest MIME allowlist contains an invalid value.")
        if len(set(self.allowed_artifact_mime_types)) != len(self.allowed_artifact_mime_types):
            raise ValueError("Harvest artifact MIME allowlist must be unique.")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.limits.v1",
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_depth": self.max_depth,
            "max_duration_seconds": self.max_duration_seconds,
            "max_events": self.max_events,
            "max_event_bytes": self.max_event_bytes,
            "max_redirects": self.max_redirects,
            "max_archive_members": self.max_archive_members,
            "max_archive_uncompressed_bytes": self.max_archive_uncompressed_bytes,
            "allowed_artifact_mime_types": list(self.allowed_artifact_mime_types),
            "allowed_response_mime_types": list(self.allowed_response_mime_types),
        }


@dataclass(frozen=True)
class CertifiedBackendCapabilities:
    """Immutable certification supplied separately from backend construction."""

    backend_id: str
    backend_version: str
    downloader_id: str
    downloader_version: str
    code_identity_sha256: str
    dataset_import_callback_id: str
    dataset_import_callback_code_identity_sha256: str
    dataset_import_callback_runtime_identity_sha256: str
    enforces_http_success: bool
    enforces_https_direct_url: bool
    resolves_and_blocks_private_networks: bool
    validates_every_redirect: bool
    enforces_response_mime_allowlist: bool
    enforces_expected_response_hash: bool
    enforces_per_file_hashes: bool
    enforces_file_count_and_byte_limits: bool
    enforces_depth_and_name_policy: bool
    enforces_archive_limits: bool
    enforces_duration_and_cancellation: bool
    enforces_bounded_evidence_fetch: bool
    enforces_quarantine_hash_probe: bool
    enforces_probe_no_decode_extract_import: bool
    enforces_deterministic_evidence_verification: bool
    enforces_transactional_catalog_promotion: bool
    enforces_direct_static_image_derivation: bool
    enforces_retained_anchored_state: bool
    enforces_whole_operation_deadline: bool
    enforces_durable_import_control: bool
    enforces_same_pack_license_and_zero_cost: bool
    enforces_technical_usability_and_pixel_uniqueness: bool
    enforces_non_self_attested_production_bindings: bool

    def __post_init__(self) -> None:
        for value in (self.backend_id, self.downloader_id, self.dataset_import_callback_id):
            if IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise ValueError("Harvest backend/downloader/callback identifier is invalid.")
        for value in (self.backend_version, self.downloader_version):
            if not value.strip() or len(value) > 100:
                raise ValueError("Harvest backend/downloader version is invalid.")
        for value in (
            self.code_identity_sha256,
            self.dataset_import_callback_code_identity_sha256,
            self.dataset_import_callback_runtime_identity_sha256,
        ):
            if SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError("Harvest backend and callback identities must be lowercase SHA-256.")
        gates = self.to_dict()["enforced_gates"]
        if not all(gates.values()):
            raise ValueError("Harvest backend certification must affirm every required safety gate.")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.backend-capabilities.v4",
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "downloader_id": self.downloader_id,
            "downloader_version": self.downloader_version,
            "code_identity_sha256": self.code_identity_sha256,
            "dataset_import_callback_id": self.dataset_import_callback_id,
            "dataset_import_callback_code_identity_sha256": self.dataset_import_callback_code_identity_sha256,
            "dataset_import_callback_runtime_identity_sha256": self.dataset_import_callback_runtime_identity_sha256,
            "enforced_gates": {
                "http_success": self.enforces_http_success,
                "https_direct_url": self.enforces_https_direct_url,
                "private_network_block": self.resolves_and_blocks_private_networks,
                "every_redirect": self.validates_every_redirect,
                "response_mime": self.enforces_response_mime_allowlist,
                "expected_response_hash": self.enforces_expected_response_hash,
                "per_file_hashes": self.enforces_per_file_hashes,
                "file_count_and_bytes": self.enforces_file_count_and_byte_limits,
                "depth_and_name_policy": self.enforces_depth_and_name_policy,
                "archive_limits": self.enforces_archive_limits,
                "duration_and_cancellation": self.enforces_duration_and_cancellation,
                "bounded_evidence_fetch": self.enforces_bounded_evidence_fetch,
                "quarantine_hash_probe": self.enforces_quarantine_hash_probe,
                "probe_no_decode_extract_import": self.enforces_probe_no_decode_extract_import,
                "deterministic_evidence_verification": self.enforces_deterministic_evidence_verification,
                "transactional_catalog_promotion": self.enforces_transactional_catalog_promotion,
                "direct_static_image_derivation": self.enforces_direct_static_image_derivation,
                "retained_anchored_state": self.enforces_retained_anchored_state,
                "whole_operation_deadline": self.enforces_whole_operation_deadline,
                "durable_import_control": self.enforces_durable_import_control,
                "same_pack_license_and_zero_cost": self.enforces_same_pack_license_and_zero_cost,
                "technical_usability_and_pixel_uniqueness": (self.enforces_technical_usability_and_pixel_uniqueness),
                "non_self_attested_production_bindings": (self.enforces_non_self_attested_production_bindings),
            },
        }


@dataclass(frozen=True)
class AcquiredFile:
    relative_path: str
    byte_count: int
    sha256: str
    mime_type: str
    usable: bool = True
    quarantine_reason: str | None = None
    taxonomy: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.relative_path or len(self.relative_path) > 500:
            raise ValueError("Acquired file path is invalid.")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("Acquired file byte count is invalid.")
        if SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("Acquired file hash is invalid.")
        if MIME_PATTERN.fullmatch(self.mime_type) is None:
            raise ValueError("Acquired file MIME type is invalid.")
        if self.usable and self.quarantine_reason is not None:
            raise ValueError("A usable acquired file cannot have a quarantine reason.")
        if not self.usable and not self.quarantine_reason:
            raise ValueError("A quarantined acquired file requires a controlled reason.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "expected_sha256": self.sha256,
            "actual_sha256": self.sha256,
            "mime_type": self.mime_type,
            "usable": self.usable,
            "quarantine_reason": self.quarantine_reason,
            "taxonomy": list(self.taxonomy),
        }


@dataclass(frozen=True)
class AcquisitionReceipt:
    final_url: str
    redirect_chain: tuple[str, ...]
    http_status: int
    response_mime_type: str
    expected_response_sha256: str
    actual_response_sha256: str
    response_bytes: int
    elapsed_seconds: float
    archive_members: int
    archive_uncompressed_bytes: int
    backend_capability_identity: str
    files: tuple[AcquiredFile, ...]
    snapshot_residue: Mapping[str, Any] | None = None
    direct_image_derivation: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AcquisitionResult:
    receipt: AcquisitionReceipt


ProgressCallback = Callable[[str, int, int | None], None]
CancelProbe = Callable[[], bool]


class AcquisitionBackend(Protocol):
    def acquire(
        self,
        source: HarvestSource,
        destination: Path,
        limits: HarvestLimits,
        *,
        cancel_requested: CancelProbe,
        progress: ProgressCallback,
    ) -> AcquisitionResult: ...


BackendFactory = Callable[[], AcquisitionBackend]
_HARDENED_BACKEND_SNAPSHOT_SEAL = object()


@dataclass(frozen=True)
class HardenedBackendIdentitySnapshot:
    """One exact identity capture reused throughout a single validation."""

    module_sha256: dict[str, str]
    runtime_dependencies: dict[str, dict[str, Any]]
    callback_binding: dict[str, str]
    code_identity_sha256: str
    _seal: object | None = field(default=None, init=False, compare=False, repr=False)


def hardened_backend_identity_snapshot() -> HardenedBackendIdentitySnapshot:
    """Capture every certified identity once without weakening the hash boundary."""

    from spritelab.product_features.conditioned_v5.identity import (
        conditioned_callback_runtime_inventory,
        conditioned_code_inventory,
    )

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="harvest-identity") as executor:
        module_future = executor.submit(_read_hardened_backend_modules)
        conditioned_future = executor.submit(conditioned_code_inventory)
        python_future = executor.submit(_python_runtime_identity)
        openssl_future = executor.submit(_openssl_runtime_identity)
        module_payloads = module_future.result()
        code_inventory = conditioned_future.result()
        python_runtime = python_future.result()
        openssl_runtime = openssl_future.result()

    module_records = [
        {
            "module": module_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }
        for module_name, payload in module_payloads
    ]
    module_sha256 = {str(record["module"]): str(record["sha256"]) for record in module_records}

    callback_binding = _conditioned_callback_binding_from_inventory(
        code_inventory,
        conditioned_callback_runtime_inventory(code_inventory),
    )
    runtime_dependencies = _hardened_backend_runtime_dependencies(
        conditioned_runtime_dependencies=code_inventory.get("runtime_dependencies"),
        python_runtime=python_runtime,
        openssl_runtime=openssl_runtime,
    )
    code_identity_sha256 = _identity(
        {
            "schema_version": "spritelab.harvest.hardened-backend-code.v4",
            "modules": module_records,
            "runtime_dependencies": runtime_dependencies,
            "dataset_import_callback": callback_binding,
        }
    )
    snapshot = HardenedBackendIdentitySnapshot(
        module_sha256=module_sha256,
        runtime_dependencies=runtime_dependencies,
        callback_binding=callback_binding,
        code_identity_sha256=code_identity_sha256,
    )
    object.__setattr__(snapshot, "_seal", _HARDENED_BACKEND_SNAPSHOT_SEAL)
    return snapshot


def identity_snapshot_matches_capabilities(
    snapshot: HardenedBackendIdentitySnapshot | None,
    capabilities: CertifiedBackendCapabilities,
) -> bool:
    """Verify an opaque snapshot captured by this process for these capabilities."""

    return bool(
        snapshot is not None
        and snapshot._seal is _HARDENED_BACKEND_SNAPSHOT_SEAL
        and capabilities.code_identity_sha256 == snapshot.code_identity_sha256
        and all(getattr(capabilities, key) == value for key, value in snapshot.callback_binding.items())
    )


def hardened_backend_code_identity() -> str:
    """Bind certification input to the exact modules enforcing acquisition.

    This is an implementation identity, not a certification decision. A
    separately supplied ``CertifiedBackendCapabilities`` must still affirm the
    gates and carry this exact digest before the adapter will run.
    """

    return hardened_backend_identity_snapshot().code_identity_sha256


def hardened_backend_module_hashes() -> dict[str, str]:
    """Return exact current hashes for every module in the audited boundary."""

    return {
        module_name: hashlib.sha256(payload).hexdigest() for module_name, payload in _read_hardened_backend_modules()
    }


def hardened_backend_runtime_dependencies() -> dict[str, dict[str, Any]]:
    """Return compact exact identities for the runtime trust boundary.

    Version strings are descriptive only. Every owned file participating in
    image, array, or YAML decisions is descriptor-rehashed by the conditioned
    runtime's public inventory primitive, including unrecorded executable
    supplements. Python's executable, full stdlib and native-extension trees,
    TLS modules, loaded interpreter libraries, and loaded OpenSSL libraries are
    independently descriptor-rehashed as well. No absolute path is returned to
    a certificate or browser response.
    """

    return _hardened_backend_runtime_dependencies()


def _hardened_backend_runtime_dependencies(
    *,
    conditioned_runtime_dependencies: Any = None,
    python_runtime: dict[str, Any] | None = None,
    openssl_runtime: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    inventories = conditioned_runtime_dependencies if isinstance(conditioned_runtime_dependencies, Mapping) else {}
    distributions = {
        name: _compact_distribution_identity(name, inventory=inventories.get(name))
        for name in ("NumPy", "Pillow", "PyYAML")
    }
    python_runtime = _python_runtime_identity() if python_runtime is None else python_runtime
    openssl_runtime = _openssl_runtime_identity() if openssl_runtime is None else openssl_runtime
    return {
        **distributions,
        "OpenSSL": openssl_runtime,
        "Python": python_runtime,
    }


def _compact_distribution_identity(
    distribution_name: str,
    *,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute one exact wheel inventory and retain only bounded evidence."""

    from spritelab.product_features.conditioned_v5.identity import (
        ConditionedCodeIdentityError,
        installed_distribution_inventory,
    )

    before_path, before_metadata = _installed_distribution_root(distribution_name)
    try:
        inventory = installed_distribution_inventory(distribution_name) if inventory is None else inventory
    except ConditionedCodeIdentityError as exc:
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} is unsafe.") from exc
    after_path, after_metadata = _installed_distribution_root(distribution_name)
    if before_path != after_path or not _same_runtime_directory(before_metadata, after_metadata):
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} changed while inventorying.")
    expected_keys = {
        "schema_version",
        "distribution",
        "version",
        "record_relative_path",
        "record_sha256",
        "record_declared_paths",
        "record_file_count",
        "owned_roots",
        "files",
        "file_count",
        "unrecorded_file_count",
        "total_bytes",
        "paths_exposed",
        "inventory_sha256",
    }
    if not isinstance(inventory, Mapping) or set(inventory) != expected_keys:
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} returned invalid evidence.")
    canonical_name = inventory.get("distribution")
    version = inventory.get("version")
    identity = inventory.get("inventory_sha256")
    record_file_count = inventory.get("record_file_count")
    unrecorded_file_count = inventory.get("unrecorded_file_count")
    owned_roots = inventory.get("owned_roots")
    files = inventory.get("files")
    file_count = inventory.get("file_count")
    total_bytes = inventory.get("total_bytes")
    if (
        inventory.get("schema_version") != "spritelab.runtime.installed-distribution-inventory.v2"
        or inventory.get("paths_exposed") is not False
        or not isinstance(inventory.get("record_relative_path"), str)
        or not inventory.get("record_relative_path")
        or not isinstance(inventory.get("record_sha256"), str)
        or SHA256_PATTERN.fullmatch(str(inventory.get("record_sha256"))) is None
        or not isinstance(canonical_name, str)
        or not 0 < len(canonical_name) <= 200
        or _normalized_distribution_name(canonical_name) != _normalized_distribution_name(distribution_name)
        or not isinstance(version, str)
        or not version
        or len(version) > 200
        or not isinstance(identity, str)
        or SHA256_PATTERN.fullmatch(identity) is None
        or type(record_file_count) is not int
        or not 0 < record_file_count <= 100_000
        or type(unrecorded_file_count) is not int
        or not 0 <= unrecorded_file_count <= 100_000
        or not isinstance(owned_roots, list)
        or not 0 < len(owned_roots) <= 100_000
        or not isinstance(files, Mapping)
        or type(file_count) is not int
        or not 0 < file_count <= 100_000
        or len(files) != file_count
        or record_file_count + unrecorded_file_count != file_count
        or type(total_bytes) is not int
        or not 0 < total_bytes <= 8 * 1024**3
    ):
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} returned invalid evidence.")
    declared_paths = inventory.get("record_declared_paths")
    if (
        not isinstance(declared_paths, list)
        or len(declared_paths) != record_file_count
        or any(not isinstance(value, str) or not value for value in declared_paths)
        or len(set(declared_paths)) != record_file_count
        or not set(declared_paths) <= set(files)
        or any(
            not isinstance(root, Mapping)
            or set(root) != {"relative_path", "kind"}
            or not isinstance(root.get("relative_path"), str)
            or not root.get("relative_path")
            or root.get("kind") not in {"directory", "file"}
            for root in owned_roots
        )
    ):
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} returned invalid ownership evidence.")
    computed_bytes = 0
    for locator, binding in files.items():
        if (
            not isinstance(locator, str)
            or not locator
            or not isinstance(binding, Mapping)
            or set(binding) != {"sha256", "byte_count"}
            or not isinstance(binding.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(str(binding.get("sha256"))) is None
            or type(binding.get("byte_count")) is not int
            or not 0 <= int(binding["byte_count"]) <= MAX_RUNTIME_FILE_BYTES
        ):
            raise ValueError(f"Harvest runtime dependency {distribution_name!r} returned invalid file evidence.")
        computed_bytes += int(binding["byte_count"])
    if computed_bytes != total_bytes:
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} returned inconsistent byte evidence.")
    inventory_payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if _identity(inventory_payload) != identity:
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} returned unbound evidence.")
    compact = {
        "schema_version": "spritelab.harvest.runtime-distribution.v2",
        "inventory_schema_version": inventory["schema_version"],
        "distribution": canonical_name,
        "version": version,
        "inventory_sha256": identity,
        "record_file_count": record_file_count,
        "owned_root_count": len(owned_roots),
        "file_count": file_count,
        "unrecorded_file_count": unrecorded_file_count,
        "total_bytes": total_bytes,
        "installation_root_identity_sha256": _runtime_directory_identity(before_metadata),
        "paths_exposed": False,
    }
    return {**compact, "runtime_identity_sha256": _identity(compact)}


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _installed_distribution_root(distribution_name: str) -> tuple[Path, os.stat_result]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        raw_root = Path(distribution.locate_file(""))
        root = raw_root.resolve(strict=True)
    except (importlib.metadata.PackageNotFoundError, OSError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} is unavailable.") from exc
    if Path(os.path.abspath(raw_root)) != root:
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} crosses a link seam.")
    try:
        with AnchoredDirectory(root, root) as anchor:
            metadata = anchor.directory_metadata()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} has an unsafe root.") from exc
    if not stat.S_ISDIR(metadata.st_mode) or _metadata_is_link_or_reparse(metadata):
        raise ValueError(f"Harvest runtime dependency {distribution_name!r} has an unsafe root.")
    return root, metadata


def _runtime_directory_identity(metadata: os.stat_result) -> str:
    return _identity(
        {
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "mode": int(stat.S_IFMT(metadata.st_mode)),
            "mtime_ns": int(metadata.st_mtime_ns),
        }
    )


def _same_runtime_directory(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
        left.st_mtime_ns,
        _metadata_is_link_or_reparse(left),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
        right.st_mtime_ns,
        _metadata_is_link_or_reparse(right),
    )


def _python_runtime_identity() -> dict[str, Any]:
    import _ssl

    _prime_python_runtime_inventory_dependencies()
    stdlib_path = getattr(ssl, "__file__", None)
    native_path = getattr(_ssl, "__file__", None)
    if not isinstance(stdlib_path, str) or not isinstance(native_path, str):
        raise ValueError("Harvest could not identify the active Python TLS runtime.")
    try:
        executable_path = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Harvest could not identify the active Python executable.") from exc
    interpreter_libraries = _python_interpreter_library_inventory()
    stdlib_inventory, native_inventory = _python_stdlib_inventories()
    runtime = {
        "schema_version": "spritelab.harvest.python-runtime.v1",
        "version": f"{platform.python_implementation()} {platform.python_version()}",
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version_hex": sys.hexversion,
        "executable": _runtime_file_identity(executable_path, label="Python executable"),
        "ssl_stdlib": _runtime_file_identity(Path(stdlib_path), label="Python ssl stdlib module"),
        "ssl_native": _runtime_file_identity(Path(native_path), label="Python native ssl module"),
        "stdlib_inventory": stdlib_inventory,
        "native_inventory": native_inventory,
        "interpreter_libraries": interpreter_libraries,
        "paths_exposed": False,
    }
    return {**runtime, "runtime_identity_sha256": _identity(runtime)}


def _prime_python_runtime_inventory_dependencies() -> None:
    """Load platform enumerator modules before snapshotting stdlib bytecode."""

    if os.name == "nt":
        __import__("ctypes.wintypes")
    elif sys.platform == "darwin":
        __import__("ctypes")


def _python_stdlib_inventories() -> tuple[dict[str, Any], dict[str, Any]]:
    """Inventory the complete importable stdlib and native extension trees."""

    stdlib_values = [sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")]
    stdlib_roots: list[Path] = []
    for value in stdlib_values:
        if not value:
            raise ValueError("Python standard-library root is unavailable.")
        root = _resolved_runtime_root(Path(value), label="Python standard-library root")
        if root not in stdlib_roots:
            stdlib_roots.append(root)
    stdlib_components = [
        _runtime_tree_inventory(
            root,
            label=f"stdlib-{index}",
            excluded_top_level=frozenset({"site-packages", "dist-packages"}),
        )
        for index, root in enumerate(stdlib_roots)
    ]
    stdlib_base = {
        "schema_version": "spritelab.harvest.python-stdlib-inventory.v1",
        "components": stdlib_components,
        "component_count": len(stdlib_components),
        "file_count": sum(int(item["file_count"]) for item in stdlib_components),
        "total_bytes": sum(int(item["total_bytes"]) for item in stdlib_components),
        "paths_exposed": False,
    }
    stdlib_inventory = {**stdlib_base, "inventory_sha256": _identity(stdlib_base)}

    destination = sysconfig.get_config_var("DESTSHARED")
    if not isinstance(destination, str) or not destination:
        import _ssl

        native_module = getattr(_ssl, "__file__", None)
        if not isinstance(native_module, str) or not native_module:
            raise ValueError("Python native-extension root is unavailable.")
        destination = os.fspath(Path(native_module).parent)
    native_root = _resolved_runtime_root(Path(destination), label="Python native-extension root")
    native_inventory = _runtime_tree_inventory(
        native_root,
        label="native-extensions",
        excluded_top_level=frozenset(),
    )
    return stdlib_inventory, native_inventory


def _resolved_runtime_root(path: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"The {label} is unavailable.") from exc
    if Path(os.path.abspath(path)) != resolved:
        raise ValueError(f"The {label} crosses a link or reparse seam.")
    return resolved


def _runtime_tree_inventory(
    root: Path,
    *,
    label: str,
    excluded_top_level: frozenset[str],
) -> dict[str, Any]:
    """Descriptor-rehash every regular file below one immutable runtime root."""

    records: list[dict[str, Any]] = []
    collisions: set[str] = set()
    total_bytes = 0
    maximum_files = 100_000
    maximum_total_bytes = 8 * 1024**3
    try:
        with AnchoredDirectory(root, root) as root_anchor:
            root_device = root_anchor.directory_metadata().st_dev

            def visit(directory: AnchoredDirectory, prefix: tuple[str, ...]) -> None:
                nonlocal total_bytes
                directory_before = directory.directory_metadata()
                if len(prefix) > 64:
                    raise ValueError(f"The {label} exceeds its path-depth bound.")
                for name in directory.names():
                    if not prefix and name.casefold() in excluded_top_level:
                        continue
                    metadata = directory.lstat(name)
                    if _metadata_is_link_or_reparse(metadata) or metadata.st_dev != root_device:
                        raise ValueError(f"The {label} crosses a link, reparse, or device seam.")
                    relative_parts = (*prefix, name)
                    relative = Path(*relative_parts).as_posix()
                    collision = unicodedata.normalize("NFC", relative).casefold()
                    if collision in collisions:
                        raise ValueError(f"The {label} contains a portable path collision.")
                    collisions.add(collision)
                    if stat.S_ISDIR(metadata.st_mode):
                        with directory.open_directory_immovable(name) as child:
                            visit(child, relative_parts)
                        continue
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_nlink != 1
                        or not 0 <= metadata.st_size <= MAX_RUNTIME_FILE_BYTES
                    ):
                        raise ValueError(f"The {label} contains an unsafe runtime file.")
                    digest, byte_count = _hash_anchored_runtime_file(directory, name, metadata, label=label)
                    total_bytes += byte_count
                    if len(records) >= maximum_files or total_bytes > maximum_total_bytes:
                        raise ValueError(f"The {label} exceeds its inventory bounds.")
                    records.append({"relative_path": relative, "sha256": digest, "byte_count": byte_count})
                directory_after = directory.directory_metadata()
                visible_after = directory.directory.lstat()
                if not _same_runtime_directory(directory_before, directory_after) or not _same_runtime_directory(
                    directory_before,
                    visible_after,
                ):
                    raise ValueError(f"The {label} directory changed while it was inventoried.")

            visit(root_anchor, ())
            root_metadata = root_anchor.directory_metadata()
    except OSError as exc:
        raise ValueError(f"The {label} could not be inventoried safely.") from exc
    ordered = sorted(records, key=lambda item: str(item["relative_path"]))
    if not ordered:
        raise ValueError(f"The {label} inventory is empty.")
    inventory_payload = {
        "files": ordered,
        "root_identity_sha256": _runtime_directory_identity(root_metadata),
    }
    compact = {
        "schema_version": "spritelab.harvest.python-runtime-tree.v1",
        "label": label,
        "inventory_sha256": _identity(inventory_payload),
        "root_identity_sha256": inventory_payload["root_identity_sha256"],
        "file_count": len(ordered),
        "total_bytes": total_bytes,
        "paths_exposed": False,
    }
    return {**compact, "runtime_identity_sha256": _identity(compact)}


def _hash_anchored_runtime_file(
    parent: AnchoredDirectory,
    name: str,
    before: os.stat_result,
    *,
    label: str,
) -> tuple[str, int]:
    descriptor = parent.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        if not _same_runtime_file(before, opened):
            raise ValueError(f"The {label} changed while opening a runtime file.")
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = parent.lstat(name)
    if (
        byte_count != before.st_size
        or not _same_runtime_file(before, opened_after)
        or not _same_runtime_file(before, path_after)
    ):
        raise ValueError(f"The {label} changed while hashing a runtime file.")
    return digest.hexdigest(), byte_count


def _python_interpreter_library_inventory() -> dict[str, Any]:
    if os.name == "nt":
        paths = _windows_loaded_module_paths()
    elif sys.platform.startswith("linux"):
        paths = _linux_loaded_module_paths()
    elif sys.platform == "darwin":
        paths = _darwin_loaded_module_paths()
    else:
        raise ValueError("Loaded Python library inventory is unsupported on this platform.")
    selected: set[Path] = set()
    for path in paths:
        folded = path.name.casefold()
        if not (
            re.fullmatch(r"python\d+(?:\d+)?(?:_d)?\.dll", folded)
            or re.fullmatch(r"libpython\d+(?:\.\d+)*(?:[a-z]*)?\.dylib", folded)
            or re.fullmatch(r"libpython\d+(?:\.\d+)*(?:[a-z]*)?\.so(?:\.\d+)*", folded)
        ):
            continue
        selected.add(_resolved_runtime_root(path, label="loaded Python runtime library"))
    if os.name == "nt" and not selected:
        raise ValueError("The loaded Python runtime DLL could not be identified.")
    if len(selected) > 8:
        raise ValueError("Loaded Python runtime library inventory is unbounded.")
    libraries = [
        _runtime_file_identity(path, label="loaded Python runtime library")
        for path in sorted(selected, key=lambda value: os.path.normcase(os.fspath(value)))
    ]
    compact = {
        "schema_version": "spritelab.harvest.python-interpreter-libraries.v1",
        "inventory_sha256": _identity(libraries),
        "library_count": len(libraries),
        "total_bytes": sum(int(item["byte_count"]) for item in libraries),
        "paths_exposed": False,
    }
    return {**compact, "runtime_identity_sha256": _identity(compact)}


def _openssl_runtime_identity() -> dict[str, Any]:
    before = _loaded_openssl_library_paths()
    summaries = [
        {"role": role, **_runtime_file_identity(path, label=f"loaded OpenSSL {role} library")} for role, path in before
    ]
    after = _loaded_openssl_library_paths()
    if before != after:
        raise ValueError("Loaded OpenSSL libraries changed while they were inventoried.")
    roles = {str(item["role"]) for item in summaries}
    if not {"crypto", "ssl"} <= roles:
        raise ValueError("The loaded OpenSSL ssl and crypto libraries could not both be identified.")
    libraries = sorted(summaries, key=lambda item: (str(item["role"]), str(item["sha256"])))
    runtime = {
        "schema_version": "spritelab.harvest.openssl-runtime.v1",
        "version": ssl.OPENSSL_VERSION,
        "libraries": libraries,
        "library_count": len(libraries),
        "total_bytes": sum(int(item["byte_count"]) for item in libraries),
        "paths_exposed": False,
    }
    return {**runtime, "runtime_identity_sha256": _identity(runtime)}


def _runtime_file_identity(path: Path, *, label: str) -> dict[str, Any]:
    """Hash one runtime file through a stable no-follow descriptor."""

    try:
        raw = Path(path)
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"The {label} is unavailable.") from exc
    if Path(os.path.abspath(raw)) != resolved:
        raise ValueError(f"The {label} crosses a link or reparse seam.")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with AnchoredDirectory(resolved.parent, resolved.parent) as parent:
            before = parent.lstat(resolved.name)
            if (
                not stat.S_ISREG(before.st_mode)
                or _metadata_is_link_or_reparse(before)
                or before.st_nlink != 1
                or not 0 < before.st_size <= MAX_RUNTIME_FILE_BYTES
            ):
                raise ValueError(f"The {label} is not a safe single-link regular file.")
            descriptor = parent.open_file(resolved.name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            try:
                opened = os.fstat(descriptor)
                if not _same_runtime_file(before, opened):
                    raise ValueError(f"The {label} changed while opening.")
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
                opened_after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            path_after = parent.lstat(resolved.name)
            if (
                byte_count != before.st_size
                or not _same_runtime_file(before, opened_after)
                or not _same_runtime_file(before, path_after)
            ):
                raise ValueError(f"The {label} changed while hashing.")
    except OSError as exc:
        raise ValueError(f"The {label} could not be read safely.") from exc
    metadata = {
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mtime_ns": int(before.st_mtime_ns),
        "size": int(before.st_size),
    }
    return {
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
        "metadata_sha256": _identity(metadata),
    }


def _same_runtime_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
        _metadata_is_link_or_reparse(left),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
        _metadata_is_link_or_reparse(right),
    )


def _loaded_openssl_library_paths() -> tuple[tuple[str, Path], ...]:
    if os.name == "nt":
        paths = _windows_loaded_module_paths()
    elif sys.platform.startswith("linux"):
        paths = _linux_loaded_module_paths()
    elif sys.platform == "darwin":
        paths = _darwin_loaded_module_paths()
    else:
        raise ValueError("Loaded-library inventory is unsupported on this platform.")
    selected: set[tuple[str, Path]] = set()
    for path in paths:
        role = _openssl_library_role(path.name)
        if role is None:
            continue
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("A loaded OpenSSL library is unavailable.") from exc
        if Path(os.path.abspath(path)) != resolved:
            raise ValueError("A loaded OpenSSL library crosses a link or reparse seam.")
        selected.add((role, resolved))
    if not selected or len(selected) > 16:
        raise ValueError("Loaded OpenSSL library inventory is unavailable or unbounded.")
    return tuple(sorted(selected, key=lambda item: (item[0], os.path.normcase(os.fspath(item[1])))))


def _openssl_library_role(name: str) -> str | None:
    folded = name.casefold()
    shared_library = folded.endswith((".dll", ".dylib")) or re.search(r"\.so(?:\.\d+)*$", folded) is not None
    if folded.startswith("_ssl") or not shared_library:
        return None
    if re.search(r"(?:^|lib)crypto(?:[-_.]|$)", folded):
        return "crypto"
    if re.search(r"(?:^|lib)ssl(?:[-_.]|$)", folded):
        return "ssl"
    return None


def _windows_loaded_module_paths() -> tuple[Path, ...]:
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()
    modules = (wintypes.HMODULE * MAX_LOADED_MODULES)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModulesEx(
        process,
        modules,
        ctypes.sizeof(modules),
        ctypes.byref(needed),
        0x03,
    ):
        raise ValueError("Loaded Windows module inventory is unavailable.")
    count = needed.value // ctypes.sizeof(wintypes.HMODULE)
    if count <= 0 or count > MAX_LOADED_MODULES:
        raise ValueError("Loaded Windows module inventory is unbounded.")
    paths: list[Path] = []
    for module in modules[:count]:
        buffer = ctypes.create_unicode_buffer(32768)
        length = psapi.GetModuleFileNameExW(process, module, buffer, len(buffer))
        if not length or length >= len(buffer) - 1:
            raise ValueError("A loaded Windows module path is unavailable.")
        paths.append(Path(buffer.value))
    return tuple(paths)


def _linux_loaded_module_paths() -> tuple[Path, ...]:
    try:
        with open("/proc/self/maps", "rb") as handle:
            payload = handle.read(16 * 1024 * 1024 + 1)
    except OSError as exc:
        raise ValueError("Loaded Linux module inventory is unavailable.") from exc
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("Loaded Linux module inventory is unbounded.")
    paths: set[Path] = set()
    for raw_line in payload.splitlines():
        fields = raw_line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith(b"/"):
            continue
        if fields[5].endswith(b" (deleted)"):
            raise ValueError("A loaded Linux module was deleted after loading.")
        paths.add(Path(os.fsdecode(fields[5])))
        if len(paths) > MAX_LOADED_MODULES:
            raise ValueError("Loaded Linux module inventory is unbounded.")
    return tuple(sorted(paths, key=lambda value: os.fspath(value)))


def _darwin_loaded_module_paths() -> tuple[Path, ...]:
    import ctypes

    process = ctypes.CDLL(None)
    image_count = process._dyld_image_count
    image_count.argtypes = []
    image_count.restype = ctypes.c_uint32
    image_name = process._dyld_get_image_name
    image_name.argtypes = [ctypes.c_uint32]
    image_name.restype = ctypes.c_char_p
    count = int(image_count())
    if count <= 0 or count > MAX_LOADED_MODULES:
        raise ValueError("Loaded macOS module inventory is unbounded.")
    paths: list[Path] = []
    for index in range(count):
        value = image_name(index)
        if value is None:
            raise ValueError("A loaded macOS module path is unavailable.")
        paths.append(Path(os.fsdecode(value)))
    return tuple(paths)


def conditioned_dataset_import_callback_binding() -> dict[str, str]:
    """Return the runtime-selected callback's exact code and worker binding."""

    # Keep this import lazy: conditioned intake imports this contracts module.
    # The identity module itself has no dependency on Harvest and inventories
    # the full callback closure, runtime distributions, executable, isolated
    # worker policy, and dependency-root identities.
    from spritelab.product_features.conditioned_v5.identity import (
        conditioned_callback_runtime_inventory,
        conditioned_code_inventory,
    )

    code_inventory = conditioned_code_inventory()
    runtime_inventory = conditioned_callback_runtime_inventory(code_inventory)
    return _conditioned_callback_binding_from_inventory(code_inventory, runtime_inventory)


def _conditioned_callback_binding_from_inventory(
    code_inventory: Mapping[str, Any],
    runtime_inventory: Mapping[str, Any],
) -> dict[str, str]:
    code_identity = code_inventory.get("inventory_sha256")
    runtime_identity = runtime_inventory.get("runtime_identity_sha256")
    if (
        not isinstance(code_identity, str)
        or SHA256_PATTERN.fullmatch(code_identity) is None
        or not isinstance(runtime_identity, str)
        or SHA256_PATTERN.fullmatch(runtime_identity) is None
    ):
        raise ValueError("The conditioned Dataset import callback identity is invalid.")
    return {
        "dataset_import_callback_id": "dataset.conditioned-intake",
        "dataset_import_callback_code_identity_sha256": code_identity,
        "dataset_import_callback_runtime_identity_sha256": runtime_identity,
    }


def _hardened_backend_module_paths() -> dict[str, Path]:
    """Return explicit roots for the audited production import closure."""

    implementation_path = Path(os.path.abspath(__file__))
    spritelab_root = implementation_path.parents[2]
    roots = (
        "spritelab.dataset_maker.exporter",
        "spritelab.dataset_maker.importer",
        "spritelab.dataset_maker.model",
        "spritelab.dataset_maker.qa",
        "spritelab.dataset_maker.training_manifest",
        "spritelab.dataset_maker.training_manifest_qa",
        "spritelab.harvest.archive",
        "spritelab.harvest.download",
        "spritelab.harvest.extract",
        "spritelab.harvest.sources",
        "spritelab.product_core",
        "spritelab.product_core.contracts",
        "spritelab.product_core.events",
        "spritelab.product_core.plugins",
        "spritelab.product_features.conditioned_v5",
        "spritelab.product_features.conditioned_v5.audit_runner",
        "spritelab.product_features.conditioned_v5.identity",
        "spritelab.product_features.conditioned_v5.intake",
        "spritelab.product_features.conditioned_v5.plugin",
        "spritelab.product_features.conditioned_v5.service",
        "spritelab.product_features.conditioned_v5.web",
        "spritelab.product_features.dataset.evidence",
        "spritelab.product_features.dataset.intake",
        "spritelab.product_features.dataset.managed",
        "spritelab.product_features.dataset.packs",
        "spritelab.product_features.dataset.semantics",
        "spritelab.product_features.dataset.sidecar",
        "spritelab.product_features.harvest",
        "spritelab.product_features.harvest.catalog",
        "spritelab.product_features.harvest.catalog_verifier",
        "spritelab.product_features.harvest.catalog_writer",
        "spritelab.product_features.harvest.certification",
        "spritelab.product_features.harvest.evidence_fetch",
        "spritelab.product_features.harvest.onboarding",
        "spritelab.product_features.harvest.service",
        "spritelab.product_features.harvest.storage",
        "spritelab.product_features.harvest.trusted_backend",
        "spritelab.product_features.harvest.web",
        "spritelab.product_features.training.activation",
        "spritelab.product_runtime",
        "spritelab.product_web.app",
        "spritelab.training.campaign",
        "spritelab.utils.safe_fs",
    )
    return {name: _first_party_module_path(name, spritelab_root) for name in roots}


def _read_hardened_backend_modules() -> tuple[tuple[str, bytes], ...]:
    implementation_path = Path(os.path.abspath(__file__))
    spritelab_root = implementation_path.parents[2]
    pending = dict(_hardened_backend_module_paths())
    payloads: dict[str, bytes] = {}
    while pending:
        module_name = min(pending)
        path = pending.pop(module_name)
        if module_name in payloads:
            continue
        payload = _read_hardened_module(module_name, path, spritelab_root=spritelab_root)
        payloads[module_name] = payload
        for dependency in _direct_first_party_imports(module_name, path, payload, spritelab_root):
            if dependency not in payloads and dependency not in pending:
                pending[dependency] = _first_party_module_path(dependency, spritelab_root)
    return tuple(sorted(payloads.items()))


def _first_party_module_path(module_name: str, spritelab_root: Path) -> Path:
    if module_name == "spritelab":
        candidate = spritelab_root / "__init__.py"
    elif module_name.startswith("spritelab."):
        relative = module_name.removeprefix("spritelab.").split(".")
        module_path = spritelab_root.joinpath(*relative)
        candidate = module_path.with_suffix(".py")
        if not candidate.is_file():
            candidate = module_path / "__init__.py"
    else:
        raise ValueError(f"Harvest implementation import is outside spritelab: {module_name}")
    if not candidate.is_file():
        raise ValueError(f"Harvest implementation module cannot be resolved: {module_name}")
    return candidate


def _direct_first_party_imports(
    module_name: str,
    path: Path,
    payload: bytes,
    spritelab_root: Path,
) -> tuple[str, ...]:
    try:
        tree = ast.parse(payload, filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError(f"Harvest implementation module cannot be parsed: {module_name}") from exc
    package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    dependencies: set[str] = set()

    def add_if_local(candidate: str) -> None:
        if candidate == "spritelab" or candidate.startswith("spritelab."):
            try:
                _first_party_module_path(candidate, spritelab_root)
            except ValueError:
                return
            dependencies.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_if_local(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".")
                keep = len(package_parts) - (node.level - 1)
                if keep <= 0:
                    continue
                base = ".".join((*package_parts[:keep], *((node.module or "").split("."))))
            else:
                base = node.module or ""
            add_if_local(base)
            for alias in node.names:
                add_if_local(f"{base}.{alias.name}" if base else alias.name)
    return tuple(sorted(dependencies))


def _read_hardened_module(
    module_name: str,
    path: Path,
    *,
    spritelab_root: Path,
) -> bytes:
    """Read one implementation module through a stable no-follow descriptor."""

    try:
        path.relative_to(spritelab_root)
    except ValueError:
        anchor_root = path.parent
    else:
        anchor_root = spritelab_root
    with open_anchored_directory(path.parent, anchor_root) as parent:
        before = parent.lstat(path.name)
        _validate_hardened_module_metadata(module_name, before)
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        descriptor = parent.open_file(path.name, flags)
        try:
            opened = os.fstat(descriptor)
            _validate_hardened_module_metadata(module_name, opened)
            if not _same_hardened_module(before, opened):
                raise ValueError(f"Harvest implementation module changed while opening: {module_name}")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read(MAX_HARDENED_MODULE_BYTES + 1)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)

        path_after = parent.lstat(path.name)
        _validate_hardened_module_metadata(module_name, opened_after)
        _validate_hardened_module_metadata(module_name, path_after)
        if (
            len(payload) != before.st_size
            or not _same_hardened_module(before, opened_after)
            or not _same_hardened_module(before, path_after)
        ):
            raise ValueError(f"Harvest implementation module changed while reading: {module_name}")
        return payload


def _validate_hardened_module_metadata(module_name: str, metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _metadata_is_link_or_reparse(metadata)
        or metadata.st_nlink != 1
        or not 1 <= metadata.st_size <= MAX_HARDENED_MODULE_BYTES
    ):
        raise ValueError(f"Harvest implementation module is unsafe: {module_name}")


def _same_hardened_module(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
        _metadata_is_link_or_reparse(left),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
        _metadata_is_link_or_reparse(right),
    )


class HardenedArchiveAcquisitionBackend:
    """Download one bound ZIP and publish only validated PNG artifacts."""

    requires_destination_parent_anchor = True

    def __init__(
        self,
        capabilities: CertifiedBackendCapabilities,
        *,
        downloader: Any | None = None,
        _identity_snapshot: HardenedBackendIdentitySnapshot | None = None,
    ) -> None:
        if not identity_snapshot_matches_capabilities(_identity_snapshot, capabilities):
            if capabilities.code_identity_sha256 != hardened_backend_code_identity():
                raise ValueError("Harvest backend capability is not bound to the exact implementation modules.")
            callback_binding = conditioned_dataset_import_callback_binding()
            if any(getattr(capabilities, key) != value for key, value in callback_binding.items()):
                raise ValueError(
                    "Harvest backend capability is not bound to the exact Dataset import callback runtime."
                )
        from spritelab.harvest.download import download_file_with_receipt

        if downloader is None:
            downloader = download_file_with_receipt
        self.capabilities = capabilities
        self._downloader = downloader
        self._downloader_accepts_anchor = downloader is download_file_with_receipt

    def acquire(
        self,
        source: HarvestSource,
        destination: Path,
        limits: HarvestLimits,
        *,
        cancel_requested: CancelProbe,
        progress: ProgressCallback,
        destination_parent_anchor: AnchoredDirectory | None = None,
    ) -> AcquisitionResult:
        from spritelab.harvest.archive import ArchiveSnapshot, archive_member_summary, extract_archive
        from spritelab.harvest.download import ReceiptDownloadResult
        from spritelab.harvest.extract import discover_png_candidates, filter_candidate_basic
        from spritelab.harvest.sources import SourceRecord

        started = time.monotonic()
        deadline = started + limits.max_duration_seconds
        _check_backend_abort(cancel_requested, deadline)
        destination = Path(os.path.abspath(os.path.expanduser(os.fspath(destination))))
        run_directory = destination.parent
        if destination_parent_anchor is None:
            with AnchoredDirectory(run_directory, run_directory) as trusted_run_parent:
                return self.acquire(
                    source,
                    destination,
                    limits,
                    cancel_requested=cancel_requested,
                    progress=progress,
                    destination_parent_anchor=trusted_run_parent,
                )
        destination_parent_anchor.verify()
        if destination_parent_anchor.directory != run_directory or Path(destination.name).name != destination.name:
            raise ValueError("Harvest artifact destination does not belong to its supplied run anchor.")
        destination_metadata = destination_parent_anchor.lstat(destination.name)
        if _metadata_is_link_or_reparse(destination_metadata) or not stat.S_ISDIR(destination_metadata.st_mode):
            raise ValueError("Harvest artifact destination must be an unlinked directory.")
        with destination_parent_anchor.open_directory(destination.name) as destination_anchor:
            if destination_anchor.names():
                raise ValueError("Harvest artifact destination must be empty.")
        downloads = require_confined_path(run_directory / "downloads", run_directory)
        with nullcontext(destination_parent_anchor) as run_anchor:
            if run_anchor.lexists(downloads.name):
                raise FileExistsError("Harvest download boundary already exists; reuse is forbidden.")
            run_anchor.mkdir(downloads.name)
            with run_anchor.open_directory(downloads.name) as downloads_anchor:
                raw_response = require_confined_path(downloads / "response.zip", downloads)
                progress("downloading", 0, None)
                download_arguments: dict[str, Any] = {
                    "allowed_hosts": source.normalized_download_hosts,
                    "overwrite": False,
                    "timeout_seconds": limits.max_duration_seconds,
                    "max_duration_seconds": limits.max_duration_seconds,
                    "allowed_content_types": limits.allowed_response_mime_types,
                    "max_bytes": limits.max_response_bytes,
                    "expected_sha256": source.expected_response_sha256,
                    "max_redirects": limits.max_redirects,
                    "require_https": True,
                    "cancel_requested": cancel_requested,
                    "progress": lambda current, total: progress("downloading", current, total),
                }
                if self._downloader_accepts_anchor:
                    download_arguments["destination_anchor"] = downloads_anchor
                result = self._downloader(
                    source.acquisition_reference,
                    raw_response,
                    **download_arguments,
                )
                if not isinstance(result, ReceiptDownloadResult):
                    raise ValueError("Harvest downloader returned no receipt-bound result.")
                if Path(result.path).absolute() != raw_response:
                    raise ValueError("Harvest downloader published outside the fixed raw-response boundary.")
                _check_backend_abort(cancel_requested, deadline)
                if result.receipt.response_mime_type not in limits.allowed_response_mime_types:
                    raise ValueError("Harvest raw response MIME type is not allowed.")
                direct_image_derivation: dict[str, Any] | None = None
                archive_snapshot: ArchiveSnapshot | None = None
                if result.receipt.response_mime_type in {"image/gif", "image/png", "image/webp"}:
                    with destination_parent_anchor.open_directory(destination.name) as destination_anchor:
                        direct_image_derivation = _publish_direct_static_image(
                            downloads_anchor,
                            raw_response.name,
                            destination_anchor,
                            response_mime_type=result.receipt.response_mime_type,
                            expected_sha256=source.expected_response_sha256,
                            expected_bytes=result.receipt.response_bytes,
                            max_file_bytes=limits.max_file_bytes,
                            cancel_requested=cancel_requested,
                            deadline=deadline,
                        )
                    selected_images = (str(direct_image_derivation["output_relative_path"]),)
                    selected_image_set = set(selected_images)
                    summary = {"total_archive_members": 0, "total_uncompressed_bytes": 0}
                else:
                    archive_snapshot = ArchiveSnapshot.open(
                        raw_response,
                        max_archive_bytes=limits.max_response_bytes,
                        expected_sha256=source.expected_response_sha256,
                        cancel_requested=cancel_requested,
                        deadline_monotonic=deadline,
                        source_anchor=downloads_anchor,
                    )

        if archive_snapshot is not None:
            with archive_snapshot:
                if (
                    archive_snapshot.byte_count != result.receipt.response_bytes
                    or archive_snapshot.sha256 != result.receipt.response_sha256
                    or archive_snapshot.sha256 != source.expected_response_sha256
                ):
                    raise ValueError("Harvest raw response does not match its downloader receipt.")
                summary = archive_member_summary(
                    archive_snapshot,
                    include_member_globs=("*.png",),
                    require_selected_png_magic=True,
                    max_members=limits.max_archive_members,
                    max_member_bytes=limits.max_file_bytes,
                    max_total_bytes=limits.max_archive_uncompressed_bytes,
                    max_archive_bytes=limits.max_response_bytes,
                    cancel_requested=cancel_requested,
                    deadline_monotonic=deadline,
                )
                selected_images = tuple(summary["selected_image_members"])
                selected_image_set = set(selected_images)
                ignored_non_png_members = tuple(summary["ignored_non_png_members"])
                _check_backend_abort(cancel_requested, deadline)
                progress("extracting", 0, len(selected_images))
                extract_archive(
                    archive_snapshot,
                    destination,
                    overwrite=False,
                    include_member_globs=("*.png",),
                    exclude_member_globs=tuple(glob.escape(name) for name in ignored_non_png_members),
                    max_members=limits.max_archive_members,
                    max_member_bytes=limits.max_file_bytes,
                    max_total_bytes=limits.max_archive_uncompressed_bytes,
                    max_archive_bytes=limits.max_response_bytes,
                    expected_sha256=source.expected_response_sha256,
                    cancel_requested=cancel_requested,
                    progress=lambda current, total: progress("extracting", current, total),
                    deadline_monotonic=deadline,
                    destination_parent_anchor=destination_parent_anchor,
                )
        _check_backend_abort(cancel_requested, deadline)
        progress("extracting", len(selected_images), len(selected_images))

        snapshot_residue: dict[str, Any] | None = None
        if archive_snapshot is not None and archive_snapshot.snapshot_residue_path is not None:
            residue_name = archive_snapshot.snapshot_residue_path.name
            residue_metadata, residue_digest = _anchored_artifact_stat_and_hash(
                destination_parent_anchor,
                downloads.name,
                residue_name,
                cancel_requested=cancel_requested,
                deadline=deadline,
            )
            if (
                residue_metadata.st_size != archive_snapshot.byte_count
                or residue_digest != archive_snapshot.sha256
                or stat.S_IMODE(residue_metadata.st_mode) != 0o400
            ):
                raise ValueError("Harvest archive snapshot residue changed before receipt binding.")
            snapshot_residue = {
                "kind": "retained_archive_snapshot_evidence",
                "relative_path": f"{downloads.name}/{residue_name}",
                "byte_count": residue_metadata.st_size,
                "sha256": residue_digest,
                "mode": "0400",
            }

        compatibility_source = SourceRecord(
            source_id=source.source_id,
            source_name=source.title,
            source_type="direct_zip_url",
            source_url=source.source_page,
            local_root_path=str(destination),
            author=source.creator,
        )
        with destination_parent_anchor.open_directory(destination.name) as destination_anchor:
            candidates = [
                filter_candidate_basic(candidate, allow_non_32=False)
                for candidate in discover_png_candidates(
                    destination,
                    compatibility_source,
                    recursive=True,
                    include_hidden=True,
                    root_anchor=destination_anchor,
                )
            ]
        if len(candidates) != len(selected_images):
            raise ValueError("Harvest extracted PNG set does not match the validated archive selection.")
        receipts: list[AcquiredFile] = []
        seen_pixel_sha256: set[str] = set()
        total_bytes = 0
        for index, candidate in enumerate(candidates, start=1):
            _check_backend_abort(cancel_requested, deadline)
            relative = candidate.relative_path
            if relative not in selected_image_set:
                raise ValueError("Harvest extracted PNG set differs from the validated archive selection.")
            if len(Path(relative).parts) > limits.max_depth:
                raise ValueError("Harvest artifact exceeds the configured path depth.")
            metadata, final_digest = _anchored_artifact_stat_and_hash(
                destination_parent_anchor,
                destination.name,
                relative,
                cancel_requested=cancel_requested,
                deadline=deadline,
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _metadata_is_link_or_reparse(metadata)
                or metadata.st_nlink != 1
                or metadata.st_size > limits.max_file_bytes
                or final_digest != candidate.image_sha256
            ):
                raise ValueError("Harvest artifact is linked, special, or oversized.")
            total_bytes += metadata.st_size
            if total_bytes > limits.max_total_bytes or len(receipts) >= limits.max_files:
                raise ValueError("Harvest artifact count or byte limit was exceeded.")
            usable = candidate.status == "candidate"
            quarantine_reason = None if usable else _candidate_quarantine_reason(candidate)
            if usable:
                if not candidate.pixel_sha256 or SHA256_PATTERN.fullmatch(candidate.pixel_sha256) is None:
                    raise ValueError("Harvest candidate has no exact decoded-pixel identity.")
                if candidate.pixel_sha256 in seen_pixel_sha256:
                    usable = False
                    quarantine_reason = "duplicate_exact_pixels"
                else:
                    seen_pixel_sha256.add(candidate.pixel_sha256)
            receipts.append(
                AcquiredFile(
                    relative_path=relative,
                    byte_count=metadata.st_size,
                    sha256=candidate.image_sha256,
                    mime_type="image/png",
                    usable=usable,
                    quarantine_reason=quarantine_reason,
                    taxonomy=source.taxonomy_hints,
                )
            )
            progress("validating", index, len(candidates))
        elapsed = time.monotonic() - started
        if elapsed > limits.max_duration_seconds:
            raise ValueError("Harvest acquisition exceeded the configured duration limit.")
        return AcquisitionResult(
            AcquisitionReceipt(
                final_url=result.receipt.final_url,
                redirect_chain=result.receipt.redirect_chain,
                http_status=result.receipt.http_status,
                response_mime_type=result.receipt.response_mime_type,
                expected_response_sha256=source.expected_response_sha256,
                actual_response_sha256=result.receipt.response_sha256,
                response_bytes=result.receipt.response_bytes,
                elapsed_seconds=elapsed,
                archive_members=int(summary["total_archive_members"]),
                archive_uncompressed_bytes=int(summary["total_uncompressed_bytes"]),
                backend_capability_identity=self.capabilities.identity,
                files=tuple(receipts),
                snapshot_residue=snapshot_residue,
                direct_image_derivation=direct_image_derivation,
            )
        )


def _publish_direct_static_image(
    source_anchor: AnchoredDirectory,
    source_name: str,
    destination_anchor: AnchoredDirectory,
    *,
    response_mime_type: str,
    expected_sha256: str,
    expected_bytes: int,
    max_file_bytes: int,
    cancel_requested: CancelProbe,
    deadline: float,
) -> dict[str, Any]:
    """Decode one MIME-matched static image and publish one fresh PNG artifact."""

    from PIL import Image, UnidentifiedImageError

    _check_backend_abort(cancel_requested, deadline)
    before = source_anchor.lstat(source_name)
    if (
        _metadata_is_link_or_reparse(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected_bytes
        or not 0 < before.st_size <= max_file_bytes
    ):
        raise ValueError("Harvest direct-image response is linked, empty, or oversized.")
    descriptor = source_anchor.open_file(source_name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise ValueError("Harvest direct-image response changed while opening.")
        payload = bytearray()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                _check_backend_abort(cancel_requested, deadline)
                chunk = handle.read(min(1 << 20, max_file_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > max_file_bytes:
                    raise ValueError("Harvest direct-image response exceeded its file limit.")
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = source_anchor.lstat(source_name)
    if _file_identity(before) != _file_identity(opened_after) or _file_identity(before) != _file_identity(path_after):
        raise ValueError("Harvest direct-image response changed while reading.")
    raw = bytes(payload)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected_bytes or raw_sha256 != expected_sha256:
        raise ValueError("Harvest direct-image bytes do not match their bound response identity.")

    expected_format = {
        "image/gif": "GIF",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }.get(response_mime_type)
    if expected_format is None:
        raise ValueError("Harvest direct-image MIME type is unsupported.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if image.format != expected_format:
                raise ValueError("Harvest direct-image MIME type does not match its decoded magic.")
            frame_count = getattr(image, "n_frames", 1)
            if type(frame_count) is not int or frame_count != 1 or bool(getattr(image, "is_animated", False)):
                raise ValueError("Harvest direct image is animated or multi-frame.")
            if width <= 0 or height <= 0 or width * height > 16_777_216:
                raise ValueError("Harvest direct-image dimensions exceed the bounded pixel limit.")
            image.load()
            rgba = image.convert("RGBA")
            rgba_bytes = rgba.tobytes()
            decoded_rgba_sha256 = hashlib.sha256(rgba_bytes).hexdigest()
            if expected_format == "PNG":
                output = raw
                derived = False
            else:
                encoded = io.BytesIO()
                rgba.save(encoded, format="PNG", optimize=False, compress_level=9)
                output = encoded.getvalue()
                derived = True
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Harvest direct-image decoding failed safely.") from exc
    if not output or len(output) > max_file_bytes:
        raise ValueError("Harvest direct-image PNG artifact exceeded its file limit.")
    _check_backend_abort(cancel_requested, deadline)
    output_name = "direct-image.png"
    _write_exclusive_anchored_bytes(destination_anchor, output_name, output)
    output_metadata, output_sha256 = _hash_direct_anchored_file(
        destination_anchor,
        output_name,
        cancel_requested=cancel_requested,
        deadline=deadline,
    )
    if output_metadata.st_size != len(output) or output_sha256 != hashlib.sha256(output).hexdigest():
        raise ValueError("Harvest direct-image artifact changed during publication.")
    return {
        "schema_version": "spritelab.harvest.direct-image-derivation.v1",
        "kind": "direct_static_image",
        "source_format": expected_format,
        "source_mime_type": response_mime_type,
        "raw_byte_count": len(raw),
        "raw_sha256": raw_sha256,
        "frame_count": 1,
        "width": width,
        "height": height,
        "decoded_rgba_sha256": decoded_rgba_sha256,
        "output_relative_path": output_name,
        "output_mime_type": "image/png",
        "output_byte_count": len(output),
        "output_sha256": output_sha256,
        "recipe_identity": "spritelab.harvest.direct-static-image-to-png.v1",
        "derived": derived,
        "source_bytes_modified": False,
    }


def _write_exclusive_anchored_bytes(anchor: AnchoredDirectory, name: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = anchor.open_file(name, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_size != 0:
            raise ValueError("Harvest direct-image artifact descriptor is unsafe.")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Harvest direct-image artifact write was incomplete.")
            offset += written
        os.fsync(descriptor)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = anchor.lstat(name)
    if opened_after.st_size != len(payload) or _file_identity(opened_after) != _file_identity(path_after):
        raise ValueError("Harvest direct-image artifact path changed during publication.")


def _hash_direct_anchored_file(
    anchor: AnchoredDirectory,
    name: str,
    *,
    cancel_requested: CancelProbe,
    deadline: float,
) -> tuple[os.stat_result, str]:
    _check_backend_abort(cancel_requested, deadline)
    before = anchor.lstat(name)
    if _metadata_is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Harvest direct-image artifact is linked or special.")
    descriptor = anchor.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise ValueError("Harvest direct-image artifact changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                _check_backend_abort(cancel_requested, deadline)
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = anchor.lstat(name)
    if _file_identity(before) != _file_identity(opened_after) or _file_identity(before) != _file_identity(path_after):
        raise ValueError("Harvest direct-image artifact changed while hashing.")
    return before, digest.hexdigest()


def _candidate_quarantine_reason(candidate: Any) -> str:
    reasons = " ".join(str(value).casefold() for value in candidate.rejection_reasons)
    if candidate.extraction_disposition == "reject_resource_fork":
        return "metadata_resource_fork"
    if "animated" in reasons or "multi-frame" in reasons or "apng" in reasons:
        return "animated_png_unsupported"
    if "fully transparent" in reasons:
        return "fully_transparent"
    if "constant rgba" in reasons:
        return "constant_rgba_image"
    if "expected 32x32" in reasons or "too small" in reasons or "too large" in reasons:
        return "not_exact_32x32"
    return "technical_png_validation_failed"


def _anchored_artifact_stat_and_hash(
    run_parent: AnchoredDirectory,
    destination_name: str,
    relative_path: str,
    *,
    cancel_requested: CancelProbe,
    deadline: float,
) -> tuple[os.stat_result, str]:
    _check_backend_abort(cancel_requested, deadline)
    parts = Path(relative_path).parts
    if not parts:
        raise ValueError("Harvest artifact has no relative path.")
    with ExitStack() as stack:
        parent = stack.enter_context(run_parent.open_directory(destination_name))
        for part in parts[:-1]:
            parent = stack.enter_context(parent.open_directory(part))
        name = parts[-1]
        before = parent.lstat(name)
        if _metadata_is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("Harvest artifact is linked or special.")
        descriptor = parent.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        digest = hashlib.sha256()
        try:
            opened = os.fstat(descriptor)
            if _file_identity(before) != _file_identity(opened):
                raise ValueError("Harvest artifact changed while opening.")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while True:
                    _check_backend_abort(cancel_requested, deadline)
                    chunk = handle.read(1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
            _check_backend_abort(cancel_requested, deadline)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = parent.lstat(name)
        if _file_identity(before) != _file_identity(opened_after) or _file_identity(before) != _file_identity(
            path_after
        ):
            raise ValueError("Harvest artifact changed while hashing.")
        return before, digest.hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _check_backend_abort(cancel_requested: CancelProbe, deadline: float) -> None:
    if cancel_requested():
        raise RuntimeError("Harvest acquisition was cancelled.")
    if time.monotonic() > deadline:
        raise RuntimeError("Harvest acquisition exceeded its duration limit.")


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


@dataclass(frozen=True)
class DatasetImportRequest:
    run_id: str
    artifacts_directory: Path
    handoff: Mapping[str, Any]
    artifact_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class DatasetImportResult:
    dataset_reference: str
    accepted_count: int
    quarantined_count: int

    def __post_init__(self) -> None:
        if IDENTIFIER_PATTERN.fullmatch(self.dataset_reference) is None:
            raise ValueError("Dataset import reference must be an opaque identifier.")
        if self.accepted_count < 0 or self.quarantined_count < 0:
            raise ValueError("Dataset import counts cannot be negative.")


class DatasetImportCancelled(RuntimeError):
    """The trusted Dataset callback observed the supplied cancellation probe."""


class DatasetImportDeadlineExceeded(RuntimeError):
    """The trusted Dataset callback exhausted the supplied operation deadline."""


class DatasetImportCallback(Protocol):
    callback_id: str
    code_identity_sha256: str
    runtime_identity_sha256: str
    supports_operation_control: bool

    def import_harvest(
        self,
        request: DatasetImportRequest,
        *,
        idempotency_key: str,
        deadline_monotonic: float,
        cancel_requested: CancelProbe,
    ) -> DatasetImportResult: ...


def validate_callback_identity(callback: DatasetImportCallback) -> None:
    if IDENTIFIER_PATTERN.fullmatch(callback.callback_id) is None:
        raise ValueError("Dataset import callback identifier is invalid.")
    if SHA256_PATTERN.fullmatch(callback.code_identity_sha256) is None:
        raise ValueError("Dataset import callback code identity is invalid.")
    if SHA256_PATTERN.fullmatch(callback.runtime_identity_sha256) is None:
        raise ValueError("Dataset import callback runtime identity is invalid.")
    if getattr(callback, "supports_operation_control", None) is not True:
        raise ValueError("Dataset import callback lacks deadline and cancellation control.")


def _identity(value: Any) -> str:
    payload = strict_json_dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AcquiredFile",
    "AcquisitionBackend",
    "AcquisitionReceipt",
    "AcquisitionResult",
    "BackendFactory",
    "CancelProbe",
    "CertifiedBackendCapabilities",
    "DatasetImportCallback",
    "DatasetImportCancelled",
    "DatasetImportDeadlineExceeded",
    "DatasetImportRequest",
    "DatasetImportResult",
    "HardenedArchiveAcquisitionBackend",
    "HardenedBackendIdentitySnapshot",
    "HarvestLimits",
    "ProgressCallback",
    "conditioned_dataset_import_callback_binding",
    "hardened_backend_code_identity",
    "hardened_backend_identity_snapshot",
    "hardened_backend_module_hashes",
    "hardened_backend_runtime_dependencies",
    "identity_snapshot_matches_capabilities",
    "validate_callback_identity",
]
