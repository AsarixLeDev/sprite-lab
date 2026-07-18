"""Passive loader for independently authored Harvest capability evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog import SHA256_PATTERN, SOURCE_ID_PATTERN
from spritelab.product_features.harvest.trusted_backend import (
    CertifiedBackendCapabilities,
    HardenedBackendIdentitySnapshot,
    hardened_backend_identity_snapshot,
)
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation

BACKEND_CAPABILITY_CERTIFICATE_SCHEMA = "spritelab.harvest.backend-capability-certificate.v5"
BACKEND_AUDIT_REPORT_SCHEMA = "spritelab.harvest.backend-audit-report.v5"
BACKEND_CAPABILITIES_RELATIVE_PATH = Path("artifacts") / "harvest" / "backend_capabilities.json"
BACKEND_AUDIT_REPORT_RELATIVE_PATH = Path("artifacts") / "harvest" / "backend_audit_report.json"
MAX_CAPABILITY_EVIDENCE_BYTES = 1 << 20
MAX_CERTIFICATE_VALIDITY = timedelta(days=90)

_CERTIFICATE_KEYS = frozenset(
    {
        "schema_version",
        "auditor_id",
        "issued_at",
        "expires_at",
        "audit_report_relative_path",
        "audit_report_sha256",
        "module_sha256",
        "runtime_dependencies",
        "capabilities",
        "certificate_identity",
    }
)
_AUDIT_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "outcome",
        "auditor_id",
        "audited_at",
        "implementation_identity_sha256",
        "module_sha256",
        "runtime_dependencies",
        "gate_results",
        "report_identity",
    }
)
_CAPABILITY_KEYS = frozenset(field.name for field in fields(CertifiedBackendCapabilities))
_CAPABILITY_TEXT_KEYS = frozenset(
    {
        "backend_id",
        "backend_version",
        "code_identity_sha256",
        "dataset_import_callback_id",
        "dataset_import_callback_code_identity_sha256",
        "dataset_import_callback_runtime_identity_sha256",
        "downloader_id",
        "downloader_version",
    }
)
_CAPABILITY_BOOL_KEYS = _CAPABILITY_KEYS - _CAPABILITY_TEXT_KEYS
_CAPABILITY_GATE_FIELDS = {
    "enforces_http_success": "http_success",
    "enforces_https_direct_url": "https_direct_url",
    "resolves_and_blocks_private_networks": "private_network_block",
    "validates_every_redirect": "every_redirect",
    "enforces_response_mime_allowlist": "response_mime",
    "enforces_expected_response_hash": "expected_response_hash",
    "enforces_per_file_hashes": "per_file_hashes",
    "enforces_file_count_and_byte_limits": "file_count_and_bytes",
    "enforces_depth_and_name_policy": "depth_and_name_policy",
    "enforces_archive_limits": "archive_limits",
    "enforces_duration_and_cancellation": "duration_and_cancellation",
    "enforces_bounded_evidence_fetch": "bounded_evidence_fetch",
    "enforces_quarantine_hash_probe": "quarantine_hash_probe",
    "enforces_probe_no_decode_extract_import": "probe_no_decode_extract_import",
    "enforces_deterministic_evidence_verification": "deterministic_evidence_verification",
    "enforces_transactional_catalog_promotion": "transactional_catalog_promotion",
    "enforces_direct_static_image_derivation": "direct_static_image_derivation",
    "enforces_retained_anchored_state": "retained_anchored_state",
    "enforces_whole_operation_deadline": "whole_operation_deadline",
    "enforces_durable_import_control": "durable_import_control",
    "enforces_same_pack_license_and_zero_cost": "same_pack_license_and_zero_cost",
    "enforces_technical_usability_and_pixel_uniqueness": "technical_usability_and_pixel_uniqueness",
    "enforces_non_self_attested_production_bindings": "non_self_attested_production_bindings",
}
if frozenset(_CAPABILITY_GATE_FIELDS) != _CAPABILITY_BOOL_KEYS:
    raise RuntimeError("Harvest certified capability fields and audit gates have drifted.")
REQUIRED_BACKEND_AUDIT_GATES = frozenset(_CAPABILITY_GATE_FIELDS.values())


class BackendCapabilityCertificateError(ValueError):
    """Independent capability evidence is malformed, stale, or unsafe."""


@dataclass(frozen=True)
class BackendCapabilityEvidence:
    """Validated independent certificate/report identities for durable binding."""

    capabilities: CertifiedBackendCapabilities
    auditor_id: str
    audited_at: str
    issued_at: str
    expires_at: str
    audit_report_sha256: str
    audit_report_identity: str
    certificate_identity: str
    implementation_identity_sha256: str
    _validation_snapshot: HardenedBackendIdentitySnapshot | None = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.backend-capability-evidence.v4",
            "auditor_id": self.auditor_id,
            "audited_at": self.audited_at,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "audit_report_sha256": self.audit_report_sha256,
            "audit_report_identity": self.audit_report_identity,
            "certificate_identity": self.certificate_identity,
            "implementation_identity_sha256": self.implementation_identity_sha256,
            "backend_capability_identity": self.capabilities.identity,
        }


def load_backend_capability_certificate(
    project_root: str | Path,
    *,
    now: datetime | None = None,
) -> CertifiedBackendCapabilities | None:
    """Return independently certified capabilities only while all evidence matches.

    This passive loader performs no mutation and no network access. Absence is
    not an error. An existing unsafe, non-PASS, expired, or code-stale
    certificate fails closed and never constructs capabilities.
    """

    evidence = load_backend_capability_evidence(project_root, now=now)
    return evidence.capabilities if evidence is not None else None


def load_backend_capability_evidence(
    project_root: str | Path,
    *,
    now: datetime | None = None,
) -> BackendCapabilityEvidence | None:
    """Load the capability plus exact independent report/certificate binding."""

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    try:
        certificate_bytes, report_bytes = _read_capability_evidence_bytes(root)
        if certificate_bytes is None:
            return None
        if report_bytes is None:
            raise BackendCapabilityCertificateError("Harvest backend audit report is missing.")
        certificate = _exact_mapping(_strict_json(certificate_bytes), _CERTIFICATE_KEYS, "certificate")
        report = _exact_mapping(_strict_json(report_bytes), _AUDIT_REPORT_KEYS, "audit report")
        return _validate_certificate(
            certificate,
            report,
            report_bytes,
            current_time=now or datetime.now(timezone.utc),
        )
    except (OSError, UnicodeError, UnsafeFilesystemOperation, ValueError) as exc:
        if isinstance(exc, BackendCapabilityCertificateError):
            raise
        raise BackendCapabilityCertificateError("Harvest backend capability evidence is unsafe or invalid.") from exc


def _validate_certificate(
    certificate: dict[str, Any],
    report: dict[str, Any],
    report_bytes: bytes,
    *,
    current_time: datetime,
) -> BackendCapabilityEvidence:
    if certificate["schema_version"] != BACKEND_CAPABILITY_CERTIFICATE_SCHEMA:
        raise BackendCapabilityCertificateError("Harvest backend capability certificate schema is unsupported.")
    if report["schema_version"] != BACKEND_AUDIT_REPORT_SCHEMA or report["outcome"] != "PASS":
        raise BackendCapabilityCertificateError("Harvest backend audit did not record PASS.")
    gate_results = _exact_mapping(report["gate_results"], REQUIRED_BACKEND_AUDIT_GATES, "audit gates")
    if any(result != "PASS" for result in gate_results.values()):
        raise BackendCapabilityCertificateError("Harvest backend audit gates did not each record PASS.")
    auditor_id = certificate["auditor_id"]
    if (
        not isinstance(auditor_id, str)
        or SOURCE_ID_PATTERN.fullmatch(auditor_id) is None
        or report["auditor_id"] != auditor_id
    ):
        raise BackendCapabilityCertificateError("Harvest backend auditor identity is invalid or changed.")
    if certificate["audit_report_relative_path"] != BACKEND_AUDIT_REPORT_RELATIVE_PATH.as_posix():
        raise BackendCapabilityCertificateError("Harvest backend audit report path is not the fixed repository path.")
    if (
        not isinstance(certificate["audit_report_sha256"], str)
        or certificate["audit_report_sha256"] != hashlib.sha256(report_bytes).hexdigest()
    ):
        raise BackendCapabilityCertificateError("Harvest backend audit report hash changed.")

    issued = _parse_utc(certificate["issued_at"], "certificate issued_at")
    expires = _parse_utc(certificate["expires_at"], "certificate expires_at")
    audited = _parse_utc(report["audited_at"], "audit report audited_at")
    current_time = current_time.astimezone(timezone.utc)
    if audited > issued or issued > current_time or expires <= issued or expires <= current_time:
        raise BackendCapabilityCertificateError("Harvest backend capability certificate is future-dated or expired.")
    if expires - issued > MAX_CERTIFICATE_VALIDITY:
        raise BackendCapabilityCertificateError("Harvest backend capability certificate validity is too long.")

    snapshot = hardened_backend_identity_snapshot()
    current_modules = snapshot.module_sha256
    expected_module_names = set(current_modules)
    certificate_modules = _module_hashes(
        certificate["module_sha256"],
        expected_names=expected_module_names,
    )
    report_modules = _module_hashes(
        report["module_sha256"],
        expected_names=expected_module_names,
    )
    if certificate_modules != current_modules or report_modules != current_modules:
        raise BackendCapabilityCertificateError("Harvest backend implementation changed after the PASS audit.")
    current_dependencies = snapshot.runtime_dependencies
    if (
        certificate["runtime_dependencies"] != current_dependencies
        or report["runtime_dependencies"] != current_dependencies
    ):
        raise BackendCapabilityCertificateError("Harvest backend runtime dependencies changed after the PASS audit.")
    current_identity = snapshot.code_identity_sha256
    if report["implementation_identity_sha256"] != current_identity:
        raise BackendCapabilityCertificateError("Harvest backend aggregate code identity changed after audit.")

    expected_report_identity = _identity({key: value for key, value in report.items() if key != "report_identity"})
    if report["report_identity"] != expected_report_identity:
        raise BackendCapabilityCertificateError("Harvest backend audit report identity is invalid.")
    expected_certificate_identity = _identity(
        {key: value for key, value in certificate.items() if key != "certificate_identity"}
    )
    if certificate["certificate_identity"] != expected_certificate_identity:
        raise BackendCapabilityCertificateError("Harvest backend capability certificate identity is invalid.")

    capability_record = _exact_mapping(certificate["capabilities"], _CAPABILITY_KEYS, "capabilities")
    if any(capability_record[key] is not True for key in _CAPABILITY_BOOL_KEYS):
        raise BackendCapabilityCertificateError("Harvest backend capability gates were not all independently affirmed.")
    for key in _CAPABILITY_TEXT_KEYS:
        if not isinstance(capability_record[key], str):
            raise BackendCapabilityCertificateError("Harvest backend capability field types are invalid.")
    if capability_record["code_identity_sha256"] != current_identity:
        raise BackendCapabilityCertificateError("Harvest backend capability is not bound to current code.")
    current_callback_binding = snapshot.callback_binding
    if any(capability_record[key] != value for key, value in current_callback_binding.items()):
        raise BackendCapabilityCertificateError(
            "Harvest backend capability is not bound to the current Dataset import callback runtime."
        )
    capabilities = CertifiedBackendCapabilities(**capability_record)
    evidence = BackendCapabilityEvidence(
        capabilities=capabilities,
        auditor_id=auditor_id,
        audited_at=report["audited_at"],
        issued_at=certificate["issued_at"],
        expires_at=certificate["expires_at"],
        audit_report_sha256=certificate["audit_report_sha256"],
        audit_report_identity=report["report_identity"],
        certificate_identity=certificate["certificate_identity"],
        implementation_identity_sha256=current_identity,
    )
    object.__setattr__(evidence, "_validation_snapshot", snapshot)
    return evidence


def evidence_has_current_validation_snapshot(
    evidence: BackendCapabilityEvidence,
    capabilities: CertifiedBackendCapabilities,
) -> bool:
    """Return whether evidence came from one complete loader validation."""

    snapshot = evidence._validation_snapshot
    return bool(
        snapshot is not None
        and evidence.capabilities == capabilities
        and evidence.implementation_identity_sha256 == snapshot.code_identity_sha256
        and capabilities.code_identity_sha256 == snapshot.code_identity_sha256
        and all(getattr(capabilities, key) == value for key, value in snapshot.callback_binding.items())
    )


def _module_hashes(value: Any, *, expected_names: set[str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise BackendCapabilityCertificateError("Harvest backend module hashes must be an object.")
    result = dict(value)
    if set(result) != expected_names or any(
        not isinstance(name, str) or not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None
        for name, digest in result.items()
    ):
        raise BackendCapabilityCertificateError("Harvest backend module hash set is incomplete or invalid.")
    return result


def _read_capability_evidence_bytes(root: Path) -> tuple[bytes | None, bytes | None]:
    certificate_parts = BACKEND_CAPABILITIES_RELATIVE_PATH.parts
    report_parts = BACKEND_AUDIT_REPORT_RELATIVE_PATH.parts
    if certificate_parts[:-1] != report_parts[:-1]:
        raise BackendCapabilityCertificateError("Harvest capability evidence parents are inconsistent.")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in certificate_parts[:-1]:
            if not anchor.lexists(part):
                return None, None
            anchor = stack.enter_context(anchor.open_directory(part))
        certificate = _read_single_link_file(anchor, certificate_parts[-1])
        if certificate is None:
            return None, None
        return certificate, _read_single_link_file(anchor, report_parts[-1])


def _read_single_link_file(anchor: AnchoredDirectory, name: str) -> bytes | None:
    if not anchor.lexists(name):
        return None
    before = anchor.lstat(name)
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_link_or_reparse(before)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= MAX_CAPABILITY_EVIDENCE_BYTES
    ):
        raise BackendCapabilityCertificateError("Harvest capability evidence must be a bounded single-link file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = anchor.open_file(name, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_file(before, opened):
            raise BackendCapabilityCertificateError("Harvest capability evidence changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(MAX_CAPABILITY_EVIDENCE_BYTES + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = anchor.lstat(name)
    if (
        len(payload) != before.st_size
        or len(payload) > MAX_CAPABILITY_EVIDENCE_BYTES
        or not _same_file(before, opened_after)
        or not _same_file(before, path_after)
    ):
        raise BackendCapabilityCertificateError("Harvest capability evidence changed while reading.")
    return payload


def _strict_json(payload: bytes) -> Any:
    def reject_constant(token: str) -> None:
        raise BackendCapabilityCertificateError(f"Non-standard JSON constant {token!r} is forbidden.")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BackendCapabilityCertificateError(f"Duplicate JSON key {key!r} is forbidden.")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BackendCapabilityCertificateError("Harvest capability evidence is not strict UTF-8 JSON.") from exc


def _exact_mapping(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise BackendCapabilityCertificateError(f"Harvest backend {label} fields do not match the schema.")
    return dict(value)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise BackendCapabilityCertificateError(f"Harvest backend {label} must be a timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BackendCapabilityCertificateError(f"Harvest backend {label} is invalid.") from exc
    if parsed.tzinfo is None:
        raise BackendCapabilityCertificateError(f"Harvest backend {label} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _identity(value: Any) -> str:
    encoded = strict_json_dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


__all__ = [
    "BACKEND_AUDIT_REPORT_RELATIVE_PATH",
    "BACKEND_AUDIT_REPORT_SCHEMA",
    "BACKEND_CAPABILITIES_RELATIVE_PATH",
    "BACKEND_CAPABILITY_CERTIFICATE_SCHEMA",
    "REQUIRED_BACKEND_AUDIT_GATES",
    "BackendCapabilityCertificateError",
    "BackendCapabilityEvidence",
    "evidence_has_current_validation_snapshot",
    "load_backend_capability_certificate",
    "load_backend_capability_evidence",
]
