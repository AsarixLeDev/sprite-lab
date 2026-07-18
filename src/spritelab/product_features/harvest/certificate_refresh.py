"""Explicit operator-authorized refresh of fixed Harvest capability evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest import certification as cert
from spritelab.product_features.harvest.trusted_backend import (
    CertifiedBackendCapabilities,
    hardened_backend_identity_snapshot,
)
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity


@dataclass(frozen=True)
class CertificateRefreshResult:
    certificate_identity: str
    implementation_identity_sha256: str
    issued_at: str
    expires_at: str
    git_head: str
    recovery_files: tuple[str, ...]
    restart_required: bool


def refresh_harvest_certificate(
    project_root: str | Path,
    *,
    rebind_current_implementation: bool,
    confirm_carry_forward_pass: bool,
    validity_days: int = 30,
) -> CertificateRefreshResult:
    """Refresh evidence, requiring an explicit waiver when implementation identity changed."""

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    if not 1 <= validity_days <= cert.MAX_CERTIFICATE_VALIDITY.days:
        raise ValueError(f"Certificate validity must be between 1 and {cert.MAX_CERTIFICATE_VALIDITY.days} days.")
    head = _require_clean_git_head(root)
    certificate_bytes, report_bytes = cert._read_capability_evidence_bytes(root)
    if certificate_bytes is None or report_bytes is None:
        raise RuntimeError("Existing Harvest certificate and audit report are required for refresh.")
    certificate, report = _validated_existing_records(certificate_bytes, report_bytes)

    snapshot = hardened_backend_identity_snapshot()
    previous_identity = str(report["implementation_identity_sha256"])
    implementation_changed = previous_identity != snapshot.code_identity_sha256
    if implementation_changed and not rebind_current_implementation:
        raise RuntimeError(
            "Harvest implementation changed; rerun with --rebind-current-implementation "
            "--confirm-carry-forward-pass to explicitly carry the prior PASS gates forward."
        )
    if rebind_current_implementation and not confirm_carry_forward_pass:
        raise RuntimeError("Changed-code rebinding requires --confirm-carry-forward-pass.")

    now = datetime.now(timezone.utc)
    auditor_id = "operator.carry-forward" if implementation_changed else str(certificate["auditor_id"])
    gate_results = dict(report["gate_results"])
    new_report = {
        "schema_version": cert.BACKEND_AUDIT_REPORT_SCHEMA,
        "outcome": "PASS",
        "auditor_id": auditor_id,
        "audited_at": _format_utc(now - timedelta(seconds=1)),
        "implementation_identity_sha256": snapshot.code_identity_sha256,
        "module_sha256": snapshot.module_sha256,
        "runtime_dependencies": snapshot.runtime_dependencies,
        "gate_results": gate_results,
    }
    new_report["report_identity"] = cert._identity(new_report)
    new_report_bytes = strict_json_dumps(new_report, sort_keys=True, separators=(",", ":")).encode("utf-8")

    capability_record = dict(certificate["capabilities"])
    capability_record.update(
        {
            "backend_version": f"git-{head}",
            "downloader_version": f"git-{head}",
            "code_identity_sha256": snapshot.code_identity_sha256,
            **snapshot.callback_binding,
        }
    )
    capabilities = CertifiedBackendCapabilities(**capability_record)
    new_certificate = {
        "schema_version": cert.BACKEND_CAPABILITY_CERTIFICATE_SCHEMA,
        "auditor_id": auditor_id,
        "issued_at": _format_utc(now),
        "expires_at": _format_utc(now + timedelta(days=validity_days)),
        "audit_report_relative_path": cert.BACKEND_AUDIT_REPORT_RELATIVE_PATH.as_posix(),
        "audit_report_sha256": hashlib.sha256(new_report_bytes).hexdigest(),
        "module_sha256": snapshot.module_sha256,
        "runtime_dependencies": snapshot.runtime_dependencies,
        "capabilities": asdict(capabilities),
    }
    new_certificate["certificate_identity"] = cert._identity(new_certificate)
    new_certificate_bytes = strict_json_dumps(
        new_certificate,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _validate_output(root, new_report_bytes, new_certificate_bytes)
    recoveries = _replace_evidence_pair(
        root,
        expected_certificate=certificate_bytes,
        expected_report=report_bytes,
        certificate_bytes=new_certificate_bytes,
        report_bytes=new_report_bytes,
    )
    return CertificateRefreshResult(
        certificate_identity=str(new_certificate["certificate_identity"]),
        implementation_identity_sha256=snapshot.code_identity_sha256,
        issued_at=str(new_certificate["issued_at"]),
        expires_at=str(new_certificate["expires_at"]),
        git_head=head,
        recovery_files=recoveries,
        restart_required=implementation_changed,
    )


def _validated_existing_records(
    certificate_bytes: bytes,
    report_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    certificate = cert._exact_mapping(
        cert._strict_json(certificate_bytes),
        cert._CERTIFICATE_KEYS,
        "certificate",
    )
    report = cert._exact_mapping(cert._strict_json(report_bytes), cert._AUDIT_REPORT_KEYS, "audit report")
    if certificate["schema_version"] != cert.BACKEND_CAPABILITY_CERTIFICATE_SCHEMA:
        raise RuntimeError("Existing Harvest certificate schema is unsupported.")
    if report["schema_version"] != cert.BACKEND_AUDIT_REPORT_SCHEMA or report["outcome"] != "PASS":
        raise RuntimeError("Existing Harvest audit report does not record PASS.")
    gates = cert._exact_mapping(report["gate_results"], cert.REQUIRED_BACKEND_AUDIT_GATES, "audit gates")
    if any(value != "PASS" for value in gates.values()):
        raise RuntimeError("Existing Harvest audit gates are not all PASS.")
    if report["auditor_id"] != certificate["auditor_id"]:
        raise RuntimeError("Existing Harvest auditor identities differ.")
    if certificate["audit_report_sha256"] != hashlib.sha256(report_bytes).hexdigest():
        raise RuntimeError("Existing Harvest report hash is invalid.")
    if report["report_identity"] != cert._identity(
        {key: value for key, value in report.items() if key != "report_identity"}
    ):
        raise RuntimeError("Existing Harvest report identity is invalid.")
    if certificate["certificate_identity"] != cert._identity(
        {key: value for key, value in certificate.items() if key != "certificate_identity"}
    ):
        raise RuntimeError("Existing Harvest certificate identity is invalid.")
    return certificate, report


def _replace_evidence_pair(
    root: Path,
    *,
    expected_certificate: bytes,
    expected_report: bytes,
    certificate_bytes: bytes,
    report_bytes: bytes,
) -> tuple[str, ...]:
    names_and_expected = (
        (cert.BACKEND_AUDIT_REPORT_RELATIVE_PATH.name, expected_report),
        (cert.BACKEND_CAPABILITIES_RELATIVE_PATH.name, expected_certificate),
    )
    replacements = {
        cert.BACKEND_AUDIT_REPORT_RELATIVE_PATH.name: report_bytes,
        cert.BACKEND_CAPABILITIES_RELATIVE_PATH.name: certificate_bytes,
    }
    with AnchoredDirectory(root, root) as project:
        with project.open_directory("artifacts") as artifacts:
            with artifacts.open_directory("harvest") as harvest:
                identities: dict[str, OwnedFileIdentity] = {}
                for name, expected in names_and_expected:
                    metadata = harvest.lstat(name)
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                        raise RuntimeError(f"Refusing unsafe Harvest artifact replacement: {name}")
                    current = cert._read_single_link_file(harvest, name)
                    if current != expected:
                        raise RuntimeError(f"Harvest artifact changed during refresh: {name}")
                    identities[name] = OwnedFileIdentity.from_stat(metadata)
                recovery_by_name: dict[str, str] = {}
                for name, _expected in names_and_expected:
                    recovery = harvest.quarantine_if_owned(
                        name,
                        identities[name],
                        prefix=f".certificate-refresh-recovery-{name}-",
                    )
                    if recovery is None:
                        raise RuntimeError(f"Harvest artifact could not be preserved: {name}")
                    recovery_by_name[name] = recovery
                published: list[str] = []
                try:
                    for name, _expected in names_and_expected:
                        harvest.atomic_write_bytes(name, replacements[name])
                        published.append(name)
                except BaseException:
                    for name in reversed(published):
                        if harvest.lexists(name):
                            replacement_identity = OwnedFileIdentity.from_stat(harvest.lstat(name))
                            harvest.quarantine_if_owned(
                                name,
                                replacement_identity,
                                prefix=f".certificate-refresh-failed-{name}-",
                            )
                    for name, _expected in names_and_expected:
                        if not harvest.lexists(name):
                            harvest.rename(recovery_by_name[name], name, replace=False)
                    raise
    return tuple(recovery_by_name[name] for name, _expected in names_and_expected)


def _require_clean_git_head(root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if dirty:
        raise RuntimeError("Commit or otherwise resolve tracked changes before refreshing the Harvest certificate.")
    return head


def _validate_output(root: Path, report_bytes: bytes, certificate_bytes: bytes) -> None:
    for label, payload in (("report", report_bytes), ("certificate", certificate_bytes)):
        if not 1 <= len(payload) <= cert.MAX_CAPABILITY_EVIDENCE_BYTES:
            raise RuntimeError(f"Harvest {label} is outside the bounded evidence size.")
        decoded = payload.decode("utf-8")
        if str(Path.home()) in decoded or str(root) in decoded:
            raise RuntimeError(f"Harvest {label} exposes a private absolute path.")
        json.loads(decoded)


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


__all__ = ["CertificateRefreshResult", "refresh_harvest_certificate"]
