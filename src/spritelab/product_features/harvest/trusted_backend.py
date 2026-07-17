"""Trusted acquisition contracts and the separately certifiable local adapter."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import os
import platform
import re
import ssl
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog import SHA256_PATTERN, HarvestSource
from spritelab.utils.safe_fs import AnchoredDirectory, open_anchored_directory, require_confined_path

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
MIME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
MAX_HARDENED_MODULE_BYTES = 4 << 20


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
            "schema_version": "spritelab.harvest.backend-capabilities.v3",
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


def hardened_backend_code_identity() -> str:
    """Bind certification input to the exact modules enforcing acquisition.

    This is an implementation identity, not a certification decision. A
    separately supplied ``CertifiedBackendCapabilities`` must still affirm the
    gates and carry this exact digest before the adapter will run.
    """

    modules = [
        {
            "module": module_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }
        for module_name, payload in _read_hardened_backend_modules()
    ]
    return _identity(
        {
            "schema_version": "spritelab.harvest.hardened-backend-code.v3",
            "modules": modules,
            "runtime_dependencies": hardened_backend_runtime_dependencies(),
            "dataset_import_callback": conditioned_dataset_import_callback_binding(),
        }
    )


def hardened_backend_module_hashes() -> dict[str, str]:
    """Return exact current hashes for every module in the audited boundary."""

    return {
        module_name: hashlib.sha256(payload).hexdigest() for module_name, payload in _read_hardened_backend_modules()
    }


def hardened_backend_runtime_dependencies() -> dict[str, dict[str, str]]:
    """Return runtime versions that materially participate in trust checks."""

    return {
        "NumPy": {"version": importlib.metadata.version("numpy")},
        "OpenSSL": {"version": ssl.OPENSSL_VERSION},
        "Pillow": {"version": importlib.metadata.version("Pillow")},
        "PyYAML": {"version": importlib.metadata.version("PyYAML")},
        "Python": {"version": f"{platform.python_implementation()} {platform.python_version()}"},
    }


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
    ) -> None:
        if capabilities.code_identity_sha256 != hardened_backend_code_identity():
            raise ValueError("Harvest backend capability is not bound to the exact implementation modules.")
        callback_binding = conditioned_dataset_import_callback_binding()
        if any(getattr(capabilities, key) != value for key, value in callback_binding.items()):
            raise ValueError("Harvest backend capability is not bound to the exact Dataset import callback runtime.")
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
        from spritelab.harvest.extract import discover_png_candidates
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
                raw_archive = require_confined_path(downloads / "response.zip", downloads)
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
                    raw_archive,
                    **download_arguments,
                )
                if not isinstance(result, ReceiptDownloadResult):
                    raise ValueError("Harvest downloader returned no receipt-bound result.")
                if Path(result.path).absolute() != raw_archive:
                    raise ValueError("Harvest downloader published outside the fixed raw-response boundary.")
                _check_backend_abort(cancel_requested, deadline)
                if result.receipt.response_mime_type not in limits.allowed_response_mime_types:
                    raise ValueError("Harvest raw response MIME type is not allowed.")
                archive_snapshot = ArchiveSnapshot.open(
                    raw_archive,
                    max_archive_bytes=limits.max_response_bytes,
                    expected_sha256=source.expected_response_sha256,
                    cancel_requested=cancel_requested,
                    deadline_monotonic=deadline,
                    source_anchor=downloads_anchor,
                )

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
                max_members=limits.max_archive_members,
                max_member_bytes=limits.max_file_bytes,
                max_total_bytes=limits.max_archive_uncompressed_bytes,
                max_archive_bytes=limits.max_response_bytes,
                cancel_requested=cancel_requested,
                deadline_monotonic=deadline,
            )
            selected_images = tuple(summary["selected_image_members"])
            selected_image_set = set(selected_images)
            _check_backend_abort(cancel_requested, deadline)
            progress("extracting", 0, len(selected_images))
            extract_archive(
                archive_snapshot,
                destination,
                overwrite=False,
                include_member_globs=("*.png",),
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
        if archive_snapshot.snapshot_residue_path is not None:
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
            candidates = discover_png_candidates(
                destination,
                compatibility_source,
                recursive=True,
                include_hidden=True,
                root_anchor=destination_anchor,
            )
        if len(candidates) != len(selected_images):
            raise ValueError("Harvest extracted PNG set does not match the validated archive selection.")
        receipts: list[AcquiredFile] = []
        total_bytes = 0
        for index, candidate in enumerate(candidates, start=1):
            _check_backend_abort(cancel_requested, deadline)
            relative = candidate.relative_path
            if candidate.status != "candidate" or relative not in selected_image_set:
                raise ValueError("Harvest archive contains a PNG that failed strict image validation.")
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
            receipts.append(
                AcquiredFile(
                    relative_path=relative,
                    byte_count=metadata.st_size,
                    sha256=candidate.image_sha256,
                    mime_type="image/png",
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
            )
        )


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


class DatasetImportCallback(Protocol):
    callback_id: str
    code_identity_sha256: str
    runtime_identity_sha256: str

    def import_harvest(
        self,
        request: DatasetImportRequest,
        *,
        idempotency_key: str,
    ) -> DatasetImportResult: ...


def validate_callback_identity(callback: DatasetImportCallback) -> None:
    if IDENTIFIER_PATTERN.fullmatch(callback.callback_id) is None:
        raise ValueError("Dataset import callback identifier is invalid.")
    if SHA256_PATTERN.fullmatch(callback.code_identity_sha256) is None:
        raise ValueError("Dataset import callback code identity is invalid.")
    if SHA256_PATTERN.fullmatch(callback.runtime_identity_sha256) is None:
        raise ValueError("Dataset import callback runtime identity is invalid.")


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
    "DatasetImportRequest",
    "DatasetImportResult",
    "HardenedArchiveAcquisitionBackend",
    "HarvestLimits",
    "ProgressCallback",
    "conditioned_dataset_import_callback_binding",
    "hardened_backend_code_identity",
    "hardened_backend_module_hashes",
    "hardened_backend_runtime_dependencies",
    "validate_callback_identity",
]
