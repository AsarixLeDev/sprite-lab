"""Passive loader for independently authored Harvest capability evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog import SHA256_PATTERN, SOURCE_ID_PATTERN
from spritelab.product_features.harvest.trusted_backend import (
    CertifiedBackendCapabilities,
    hardened_backend_code_identity,
    hardened_backend_module_hashes,
)
from spritelab.utils.safe_fs import UnsafeFilesystemOperation, require_confined_path

BACKEND_CAPABILITY_CERTIFICATE_SCHEMA = "spritelab.harvest.backend-capability-certificate.v1"
BACKEND_AUDIT_REPORT_SCHEMA = "spritelab.harvest.backend-audit-report.v1"
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
        "report_identity",
    }
)
_CAPABILITY_KEYS = frozenset(field.name for field in fields(CertifiedBackendCapabilities))
_CAPABILITY_TEXT_KEYS = frozenset(
    {"backend_id", "backend_version", "code_identity_sha256", "downloader_id", "downloader_version"}
)
_CAPABILITY_BOOL_KEYS = _CAPABILITY_KEYS - _CAPABILITY_TEXT_KEYS


class BackendCapabilityCertificateError(ValueError):
    """Independent capability evidence is malformed, stale, or unsafe."""


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

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    try:
        certificate_path = require_confined_path(root / BACKEND_CAPABILITIES_RELATIVE_PATH, root)
        if not _path_exists_safely(certificate_path, root):
            return None
        report_path = require_confined_path(root / BACKEND_AUDIT_REPORT_RELATIVE_PATH, root)
        if not _path_exists_safely(report_path, root):
            raise BackendCapabilityCertificateError("Harvest backend audit report is missing.")
        certificate_bytes = _read_single_link_file(certificate_path)
        report_bytes = _read_single_link_file(report_path)
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
) -> CertifiedBackendCapabilities:
    if certificate["schema_version"] != BACKEND_CAPABILITY_CERTIFICATE_SCHEMA:
        raise BackendCapabilityCertificateError("Harvest backend capability certificate schema is unsupported.")
    if report["schema_version"] != BACKEND_AUDIT_REPORT_SCHEMA or report["outcome"] != "PASS":
        raise BackendCapabilityCertificateError("Harvest backend audit did not record PASS.")
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

    current_modules = hardened_backend_module_hashes()
    certificate_modules = _module_hashes(certificate["module_sha256"])
    report_modules = _module_hashes(report["module_sha256"])
    if certificate_modules != current_modules or report_modules != current_modules:
        raise BackendCapabilityCertificateError("Harvest backend implementation changed after the PASS audit.")
    current_identity = hardened_backend_code_identity()
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
    return CertifiedBackendCapabilities(**capability_record)


def _module_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise BackendCapabilityCertificateError("Harvest backend module hashes must be an object.")
    result = dict(value)
    expected_names = set(hardened_backend_module_hashes())
    if set(result) != expected_names or any(
        not isinstance(name, str) or not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None
        for name, digest in result.items()
    ):
        raise BackendCapabilityCertificateError("Harvest backend module hash set is incomplete or invalid.")
    return result


def _path_exists_safely(path: Path, root: Path) -> bool:
    current = root
    relative_parts = path.relative_to(root).parts
    for index, part in enumerate(relative_parts):
        current = current / part
        if not os.path.lexists(current):
            return False
        metadata = current.lstat()
        if _is_link_or_reparse(metadata):
            raise BackendCapabilityCertificateError("Harvest capability evidence path crosses a link.")
        is_target = index == len(relative_parts) - 1
        if not is_target and (not stat.S_ISDIR(metadata.st_mode) or current.is_mount()):
            raise BackendCapabilityCertificateError("Harvest capability evidence path crosses an unsafe seam.")
    return True


def _read_single_link_file(path: Path) -> bytes:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_link_or_reparse(before)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= MAX_CAPABILITY_EVIDENCE_BYTES
    ):
        raise BackendCapabilityCertificateError("Harvest capability evidence must be a bounded single-link file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_file(before, opened):
            raise BackendCapabilityCertificateError("Harvest capability evidence changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(MAX_CAPABILITY_EVIDENCE_BYTES + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = path.lstat()
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
    "BackendCapabilityCertificateError",
    "load_backend_capability_certificate",
]
