"""Independent audit projection with explicit applicability and freshness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spritelab.product_core.audit_evidence import (
    ApplicabilityStatus,
    AuditVerdict,
    verify_labeling_audit,
)
from spritelab.product_features.dataset.certification import labeling_downstream_consequences
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import AuditStatus, ProjectState, StageState
from spritelab.v3.status import verify_memorization_audit_applicability

_AUDIT_SOURCES = (
    ("semantic-labeling", "semantic-labeling", "labeling", "audit_report", "BLOCKS_LABELING_AND_FREEZE"),
    (
        "training-infrastructure",
        "training-infrastructure-audit",
        "training",
        "audit_report",
        "BLOCKS_TRAINING",
    ),
    ("memorization", "memorization-review", "evaluation", "memorization_audit", "BLOCKS_PROMOTION"),
)


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _failed_gates(report: dict[str, Any] | None, stage: StageState) -> list[str]:
    failed: set[str] = set()
    stage_gates = stage.metrics.get("failed_gates", [])
    if isinstance(stage_gates, list):
        failed.update(str(value) for value in stage_gates)
    if not report:
        return sorted(failed, key=_gate_sort_key)
    gates = report.get("gates")
    if isinstance(gates, dict):
        failed.update(str(key) for key, value in gates.items() if str(value).upper() == "FAIL" or value is False)
    verdicts = report.get("gate_verdicts")
    if isinstance(verdicts, list):
        for item in verdicts:
            if isinstance(item, dict) and str(item.get("verdict", "")).upper() == "FAIL":
                failed.add(str(item.get("gate", item.get("name", "unknown"))))
    rerun = report.get("rerun")
    rerun_gates = rerun.get("gates") if isinstance(rerun, dict) else None
    if isinstance(rerun_gates, dict):
        failed.update(str(key) for key, value in rerun_gates.items() if value is False and key != "passed")
    return sorted(failed, key=_gate_sort_key)


def _gate_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _bound_commit(report: dict[str, Any] | None) -> str | None:
    if not report:
        return None
    for key in ("bound_commit", "audited_commit", "source_commit", "commit", "head"):
        value = report.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _freshness(stage: StageState, report: dict[str, Any] | None) -> tuple[str, str, bool]:
    if report is None:
        return "NOT_APPLICABLE", "MISSING", False
    if stage.audit == AuditStatus.STALE:
        return "NOT_APPLICABLE", "STALE", False
    if stage.audit == AuditStatus.NOT_AUDITED:
        return "UNKNOWN", "UNBOUND", False
    return "APPLICABLE", "FRESH", True


def collect_audits(config: ProjectConfig, state: ProjectState) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for subsystem, stage_key, section, key, blocking_consequence in _AUDIT_SOURCES:
        try:
            stage = state.stage(stage_key)
        except KeyError:
            continue
        try:
            report_path = config.path_for(section, key)
        except KeyError:
            report_path = None
        report = _read_json(report_path)
        if subsystem == "semantic-labeling":
            try:
                manifest_path = config.path_for("labeling", "audit_hashes")
            except KeyError:
                manifest_path = None
            verification = verify_labeling_audit(
                config.root,
                report_path,
                manifest_path,
                authoritative_stage_source_commit=stage.source_commit or state.source_commit,
            )
            applicable = verification.applicability_status in {
                ApplicabilityStatus.APPLICABLE,
                ApplicabilityStatus.LEGACY_APPLICABLE,
            }
            audit_verdict = verification.report_verdict.value if verification.report_verdict else "NOT_AUDITED"
            audits.append(
                {
                    "subsystem": subsystem,
                    "verdict": verification.display_verdict,
                    "audit_verdict": audit_verdict,
                    "bound_commit": verification.bound_commit,
                    "current_commit": state.source_commit,
                    "current_relevant_code_identity": (
                        verification.current_identity.code_identity_sha256 if verification.current_identity else None
                    ),
                    "applicability": verification.applicability_status.value,
                    "freshness": "FRESH" if applicable else verification.applicability_status.value,
                    "applicable": applicable,
                    "current_certification": verification.is_current_pass,
                    "staleness_reasons": list(verification.reasons),
                    "verified_artifact_status": verification.artifact_status.value,
                    "authorized_scopes": list(verification.authorized_scopes),
                    "downstream_consequences": list(labeling_downstream_consequences(verification)),
                    "failed_gates": _failed_gates(report, stage),
                    "report": str(report_path) if report_path else None,
                    "report_exists": bool(report_path and report_path.is_file()),
                    "artifact_hash_manifest": str(manifest_path) if manifest_path else None,
                    "authorization_consequence": (
                        "ELIGIBLE_FOR_DECLARED_SCOPE"
                        if verification.is_current_pass
                        else "NO_CURRENT_CERTIFICATION"
                        if verification.applicability_status is ApplicabilityStatus.STALE
                        else blocking_consequence
                        if verification.report_verdict in {AuditVerdict.FAIL, AuditVerdict.INCONCLUSIVE}
                        else "DEPENDENT_AUTHORIZATION_UNAVAILABLE"
                    ),
                }
            )
            continue
        if subsystem == "memorization":
            verification = verify_memorization_audit_applicability(config.root, report)
            applicable = verification.applicable
            if applicable and verification.status == AuditStatus.PASS:
                consequence = "ELIGIBLE_FOR_DEPENDENT_AUTHORIZATION"
            elif verification.status in {AuditStatus.STALE, AuditStatus.NOT_COMPARABLE}:
                consequence = "NO_CURRENT_CERTIFICATION"
            elif verification.status in {AuditStatus.FAIL, AuditStatus.INCONCLUSIVE}:
                consequence = blocking_consequence
            else:
                consequence = "DEPENDENT_AUTHORIZATION_UNAVAILABLE"
            audits.append(
                {
                    "subsystem": subsystem,
                    "verdict": verification.status.value,
                    "audit_verdict": verification.report_verdict or "NOT_AUDITED",
                    "bound_commit": _bound_commit(report),
                    "current_commit": state.source_commit,
                    "current_relevant_code_identity": (
                        verification.current_identity.get("code_identity_sha256")
                        if verification.current_identity
                        else None
                    ),
                    "applicability": "APPLICABLE" if applicable else verification.status.value,
                    "freshness": "FRESH" if applicable else verification.status.value,
                    "applicable": applicable,
                    "current_certification": applicable and verification.status == AuditStatus.PASS,
                    "staleness_reasons": list(verification.reasons),
                    "failed_gates": _failed_gates(report, stage),
                    "report": str(report_path) if report_path else None,
                    "report_exists": bool(report_path and report_path.is_file()),
                    "authorization_consequence": consequence,
                }
            )
            continue
        applicability, freshness, applicable = _freshness(stage, report)
        if applicable and stage.audit == AuditStatus.PASS:
            consequence = "ELIGIBLE_FOR_DEPENDENT_AUTHORIZATION"
        elif freshness == "STALE":
            consequence = "NO_CURRENT_CERTIFICATION"
        elif stage.audit in {AuditStatus.FAIL, AuditStatus.INCONCLUSIVE}:
            consequence = blocking_consequence
        else:
            consequence = "DEPENDENT_AUTHORIZATION_UNAVAILABLE"
        audits.append(
            {
                "subsystem": subsystem,
                "verdict": stage.audit.value,
                "bound_commit": _bound_commit(report),
                "current_commit": state.source_commit,
                "applicability": applicability,
                "freshness": freshness,
                "applicable": applicable,
                "current_certification": applicable and stage.audit == AuditStatus.PASS,
                "failed_gates": _failed_gates(report, stage),
                "report": str(report_path) if report_path else None,
                "report_exists": bool(report_path and report_path.is_file()),
                "authorization_consequence": consequence,
            }
        )
    return audits
