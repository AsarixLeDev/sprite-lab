from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from spritelab.product_core import (
    AUDIT_HASH_MANIFEST_SCHEMA,
    AUDIT_REPORT_SCHEMA,
    LABELING_BOUND_FILES,
    ProjectContext,
    compute_labeling_audit_identity,
)


def copy_labeling_identity_root(source_root: Path, target_root: Path) -> Path:
    for relative in LABELING_BOUND_FILES:
        source = source_root / relative
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target_root


def write_labeling_audit(
    root: Path,
    artifact_dir: Path,
    *,
    verdict: str = "PASS",
    scopes: tuple[str, ...] = (),
    bound_commit: str = "a" * 40,
    legacy: bool = False,
    report_updates: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    identity = compute_labeling_audit_identity(root)
    report: dict[str, Any] = {
        "schema_version": "legacy.synthetic-audit.v1" if legacy else AUDIT_REPORT_SCHEMA,
        "subsystem": "labeling",
        "audit_kind": "conservative_labeling_health",
        "verdict": verdict,
        "bound_commit": bound_commit,
        "auditor_identity": "independent-test-auditor",
        "created_at_utc": "2026-07-13T12:00:00Z",
        "evidence_role": "independent_certification",
        "authorized_scopes": list(scopes),
    }
    if not legacy:
        report.update(
            {
                "bound_code_identity_sha256": identity.code_identity_sha256,
                "bound_contract_identity_sha256": identity.contract_identity_sha256,
                "bound_data_identity_sha256": identity.data_identity_sha256,
                "bound_component_identities": dict(identity.component_identities),
            }
        )
    if report_updates:
        report.update(report_updates)
    report_path = artifact_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_path = artifact_dir / "artifact_hashes.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": AUDIT_HASH_MANIFEST_SCHEMA,
                "artifacts": [{"path": str(report_path.resolve()), "sha256": digest}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path, manifest_path


def audit_context(
    root: Path,
    report_path: Path,
    manifest_path: Path,
    *,
    stage_source_commit: str = "a" * 40,
    extra_config: dict[str, Any] | None = None,
) -> ProjectContext:
    config: dict[str, Any] = {
        "labeling": {
            "audit_report": str(report_path),
            "audit_hashes": str(manifest_path),
            "audit_stage_source_commit": stage_source_commit,
        }
    }
    if extra_config:
        config.update(extra_config)
    return ProjectContext(root, config)
