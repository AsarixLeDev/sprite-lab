"""Trusted acquisition contracts and the separately certifiable local adapter."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog import SHA256_PATTERN, HarvestSource
from spritelab.utils.safe_fs import require_confined_path

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
MIME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")


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

    def __post_init__(self) -> None:
        for value in (self.backend_id, self.downloader_id):
            if IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise ValueError("Harvest backend/downloader identifier is invalid.")
        for value in (self.backend_version, self.downloader_version):
            if not value.strip() or len(value) > 100:
                raise ValueError("Harvest backend/downloader version is invalid.")
        if SHA256_PATTERN.fullmatch(self.code_identity_sha256) is None:
            raise ValueError("Harvest backend code identity must be lowercase SHA-256.")
        gates = self.to_dict()["enforced_gates"]
        if not all(gates.values()):
            raise ValueError("Harvest backend certification must affirm every required safety gate.")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.backend-capabilities.v1",
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "downloader_id": self.downloader_id,
            "downloader_version": self.downloader_version,
            "code_identity_sha256": self.code_identity_sha256,
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

    modules: list[dict[str, Any]] = []
    for module_name, path in sorted(_hardened_backend_module_paths().items()):
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size > 4 << 20
        ):
            raise ValueError(f"Harvest implementation module is unsafe: {module_name}")
        modules.append(
            {
                "module": module_name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "byte_count": metadata.st_size,
            }
        )
    return _identity(
        {
            "schema_version": "spritelab.harvest.hardened-backend-code.v1",
            "modules": modules,
        }
    )


def hardened_backend_module_hashes() -> dict[str, str]:
    """Return exact current hashes for every module in the audited boundary."""

    result: dict[str, str] = {}
    for module_name, path in sorted(_hardened_backend_module_paths().items()):
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size > 4 << 20
        ):
            raise ValueError(f"Harvest implementation module is unsafe: {module_name}")
        result[module_name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _hardened_backend_module_paths() -> dict[str, Path]:
    spritelab_root = Path(__file__).resolve().parents[2]
    feature_root = Path(__file__).parent
    return {
        "spritelab.harvest.archive": spritelab_root / "harvest" / "archive.py",
        "spritelab.harvest.download": spritelab_root / "harvest" / "download.py",
        "spritelab.harvest.extract": spritelab_root / "harvest" / "extract.py",
        "spritelab.product_features.harvest": feature_root / "__init__.py",
        "spritelab.product_features.harvest.catalog": feature_root / "catalog.py",
        "spritelab.product_features.harvest.certification": feature_root / "certification.py",
        "spritelab.product_features.harvest.service": feature_root / "service.py",
        "spritelab.product_features.harvest.trusted_backend": Path(__file__),
    }


class HardenedArchiveAcquisitionBackend:
    """Download one bound ZIP and publish only validated PNG artifacts."""

    def __init__(
        self,
        capabilities: CertifiedBackendCapabilities,
        *,
        downloader: Any | None = None,
    ) -> None:
        if capabilities.code_identity_sha256 != hardened_backend_code_identity():
            raise ValueError("Harvest backend capability is not bound to the exact implementation modules.")
        if downloader is None:
            from spritelab.harvest.download import download_file_with_receipt

            downloader = download_file_with_receipt
        self.capabilities = capabilities
        self._downloader = downloader

    def acquire(
        self,
        source: HarvestSource,
        destination: Path,
        limits: HarvestLimits,
        *,
        cancel_requested: CancelProbe,
        progress: ProgressCallback,
    ) -> AcquisitionResult:
        from spritelab.harvest.archive import archive_member_summary, extract_archive
        from spritelab.harvest.download import ReceiptDownloadResult, compute_sha256
        from spritelab.harvest.extract import discover_png_candidates
        from spritelab.harvest.sources import SourceRecord

        started = time.monotonic()
        deadline = started + limits.max_duration_seconds
        _check_backend_abort(cancel_requested, deadline)
        destination = _validated_empty_destination(destination)
        run_directory = destination.parent
        downloads = require_confined_path(run_directory / "downloads", run_directory)
        if os.path.lexists(downloads):
            raise FileExistsError("Harvest download boundary already exists; reuse is forbidden.")
        downloads.mkdir()
        downloads_metadata = downloads.lstat()
        if _metadata_is_link_or_reparse(downloads_metadata) or not stat.S_ISDIR(downloads_metadata.st_mode):
            raise ValueError("Harvest download boundary is unsafe.")
        raw_archive = require_confined_path(downloads / "response.zip", downloads)
        progress("downloading", 0, None)
        result = self._downloader(
            source.acquisition_reference,
            raw_archive,
            allowed_hosts=source.normalized_download_hosts,
            overwrite=False,
            timeout_seconds=limits.max_duration_seconds,
            max_duration_seconds=limits.max_duration_seconds,
            allowed_content_types=limits.allowed_response_mime_types,
            max_bytes=limits.max_response_bytes,
            expected_sha256=source.expected_response_sha256,
            max_redirects=limits.max_redirects,
            require_https=True,
            cancel_requested=cancel_requested,
            progress=lambda current, total: progress("downloading", current, total),
        )
        if not isinstance(result, ReceiptDownloadResult):
            raise ValueError("Harvest downloader returned no receipt-bound result.")
        if Path(result.path).absolute() != raw_archive:
            raise ValueError("Harvest downloader published outside the fixed raw-response boundary.")
        _check_backend_abort(cancel_requested, deadline)
        raw_metadata = raw_archive.lstat()
        if (
            not stat.S_ISREG(raw_metadata.st_mode)
            or _metadata_is_link_or_reparse(raw_metadata)
            or raw_metadata.st_nlink != 1
            or raw_metadata.st_size != result.receipt.response_bytes
            or compute_sha256(raw_archive, max_bytes=limits.max_response_bytes) != result.receipt.response_sha256
            or result.receipt.response_sha256 != source.expected_response_sha256
        ):
            raise ValueError("Harvest raw response does not match its downloader receipt.")
        if result.receipt.response_mime_type not in limits.allowed_response_mime_types:
            raise ValueError("Harvest raw response MIME type is not allowed.")

        summary = archive_member_summary(
            raw_archive,
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
            raw_archive,
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
        )
        _check_backend_abort(cancel_requested, deadline)
        progress("extracting", len(selected_images), len(selected_images))

        compatibility_source = SourceRecord(
            source_id=source.source_id,
            source_name=source.title,
            source_type="direct_zip_url",
            source_url=source.source_page,
            local_root_path=str(destination),
            author=source.creator,
        )
        candidates = discover_png_candidates(
            destination,
            compatibility_source,
            recursive=True,
            include_hidden=True,
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
            metadata = candidate.extracted_path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _metadata_is_link_or_reparse(metadata)
                or metadata.st_nlink != 1
                or metadata.st_size > limits.max_file_bytes
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
            )
        )


def _validated_empty_destination(path: Path) -> Path:
    destination = Path(path).absolute()
    destination = require_confined_path(destination, destination.parent)
    if not os.path.lexists(destination):
        raise FileNotFoundError("Harvest artifact destination must be created by the run owner.")
    metadata = destination.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or _metadata_is_link_or_reparse(metadata)
        or destination.is_mount()
        or any(destination.iterdir())
    ):
        raise ValueError("Harvest artifact destination must be an empty, unlinked directory.")
    parent_metadata = destination.parent.lstat()
    if _metadata_is_link_or_reparse(parent_metadata) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("Harvest run directory is unsafe.")
    return destination


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
    "hardened_backend_code_identity",
    "hardened_backend_module_hashes",
    "validate_callback_identity",
]
