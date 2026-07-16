"""Identity-bound labeling capability adapter and plain-language projection."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.product_core.audit_evidence import (
    BULK_UNREVIEWED_EXACT_OBJECT_LABELS,
    CALIBRATION_READINESS,
    CONDITIONED_VIEW_CANDIDATES,
    CONSERVATIVE_PROPOSAL_GENERATION,
    HUMAN_TRUTH,
    PRODUCTION_CONDITIONED_DATASET_FREEZE,
    STRONG_SUPERVISION,
    ApplicabilityStatus,
    AuditVerdict,
    AuditVerification,
    verify_labeling_audit,
)
from spritelab.product_core.backend_contracts import (
    ActionAuthorization,
    BackendCapabilitySnapshot,
    CapabilityState,
)
from spritelab.product_core.contracts import ProjectContext

_SCOPE_ACTIONS = {
    CONSERVATIVE_PROPOSAL_GENERATION: "labeling.propose",
    CONDITIONED_VIEW_CANDIDATES: "labeling.conditioned",
    PRODUCTION_CONDITIONED_DATASET_FREEZE: "dataset.conditioned_freeze",
    HUMAN_TRUTH: "labeling.human_truth",
    CALIBRATION_READINESS: "labeling.calibration",
    STRONG_SUPERVISION: "labeling.strong_supervision",
    BULK_UNREVIEWED_EXACT_OBJECT_LABELS: "labeling.bulk_exact_object",
}


@dataclass(frozen=True)
class LabelingProductProjection:
    automatic_descriptions_status: str
    automatic_descriptions_message: str
    exact_object_labels_status: str
    exact_object_labels_message: str
    calibration_status: str
    dataset_freeze_status: str
    authorized_scopes: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Return consequences without hashes, commits, or audit jargon."""

        return {
            "automatic_descriptions": {
                "status": self.automatic_descriptions_status,
                "message": self.automatic_descriptions_message,
            },
            "exact_object_labels": {
                "status": self.exact_object_labels_status,
                "message": self.exact_object_labels_message,
            },
            "calibration_readiness": self.calibration_status,
            "dataset_freeze_authorization": self.dataset_freeze_status,
            "authorized_scopes": list(self.authorized_scopes),
        }


@dataclass(frozen=True)
class VerifiedLabelingCapabilityAdapter:
    """Read-only adapter; every probe recomputes evidence applicability."""

    persisted_capability: Mapping[str, Any] | None = None

    def probe_labeling_capability(self, context: ProjectContext) -> BackendCapabilitySnapshot:
        return labeling_capability(context, persisted_capability=self.persisted_capability)


def labeling_audit_verification(context: ProjectContext) -> AuditVerification:
    report_path, manifest_path = _configured_evidence_paths(context)
    return verify_labeling_audit(
        context.project_root,
        report_path,
        manifest_path,
        authoritative_stage_source_commit=_authoritative_source_commit(context),
    )


def labeling_capability(
    context: ProjectContext,
    *,
    persisted_capability: Mapping[str, Any] | None = None,
    mandatory_blockers: tuple[str, ...] = (),
) -> BackendCapabilitySnapshot:
    """Rehydrate fail-closed; persisted READY and audit booleans are never trusted."""

    verification = labeling_audit_verification(context)
    blocker_codes = list(mandatory_blockers)
    if persisted_capability and "audit_passed" in persisted_capability:
        blocker_codes.append("persisted_boolean_audit_ignored")
    if verification.reasons:
        blocker_codes.extend(verification.reasons)

    evidence = verification.evidence
    applicable = verification.applicability_status in {
        ApplicabilityStatus.APPLICABLE,
        ApplicabilityStatus.LEGACY_APPLICABLE,
    }
    pass_current = bool(evidence and evidence.verdict is AuditVerdict.PASS and applicable)
    scopes = verification.authorized_scopes if pass_current else ()
    actions = tuple(_SCOPE_ACTIONS[scope] for scope in scopes if scope in _SCOPE_ACTIONS)
    if verification.applicability_status is ApplicabilityStatus.STALE:
        certification_state = CapabilityState.STALE
        production_state = CapabilityState.STALE
    elif pass_current:
        certification_state = CapabilityState.READY
        production_state = (
            CapabilityState.READY
            if CONSERVATIVE_PROPOSAL_GENERATION in scopes and not mandatory_blockers
            else CapabilityState.BLOCKED
        )
    elif verification.report_verdict is AuditVerdict.FAIL and applicable:
        certification_state = CapabilityState.BLOCKED
        production_state = CapabilityState.BLOCKED
    else:
        certification_state = CapabilityState.CERTIFICATION_PENDING
        production_state = CapabilityState.BLOCKED
    return BackendCapabilitySnapshot(
        backend_id="labeling",
        technical_state=CapabilityState.READY,
        independent_certification_state=certification_state,
        production_state=production_state,
        normal_actions=actions,
        authorized_scopes=scopes,
        blocker_codes=tuple(dict.fromkeys(blocker_codes)),
        audit_evidence=evidence,
    )


def authorize_labeling_scope(context: ProjectContext, scope: str) -> ActionAuthorization:
    """Recompute applicability at action submission time."""

    action = _SCOPE_ACTIONS.get(scope)
    if action is None:
        return ActionAuthorization(False, "unknown_labeling_authorization_scope")
    capability = labeling_capability(context)
    if scope not in capability.authorized_scopes:
        return ActionAuthorization(False, f"scope_not_authorized:{scope}")
    return capability.authorize(action)


def project_labeling_status(verification: AuditVerification) -> LabelingProductProjection:
    scopes = verification.authorized_scopes
    applicable = verification.applicability_status in {
        ApplicabilityStatus.APPLICABLE,
        ApplicabilityStatus.LEGACY_APPLICABLE,
    }
    if verification.report_verdict is AuditVerdict.FAIL and applicable:
        automatic_status = "UNAVAILABLE"
        automatic_message = "Not available — the latest reliability check found problems."
    elif verification.is_current_pass and CONSERVATIVE_PROPOSAL_GENERATION in scopes:
        automatic_status = "READY"
        automatic_message = "Available for broad suggestions."
    else:
        automatic_status = "CERTIFICATION_PENDING"
        automatic_message = "Waiting for a reliability check."

    exact_authorized = (
        verification.is_current_pass and HUMAN_TRUTH in scopes and BULK_UNREVIEWED_EXACT_OBJECT_LABELS in scopes
    )
    return LabelingProductProjection(
        automatic_descriptions_status=automatic_status,
        automatic_descriptions_message=automatic_message,
        exact_object_labels_status="READY" if exact_authorized else "UNAVAILABLE",
        exact_object_labels_message=(
            "Available within the independently verified scope."
            if exact_authorized
            else "Not available — insufficient reviewed truth."
        ),
        calibration_status=(
            "READY" if verification.is_current_pass and CALIBRATION_READINESS in scopes else "NOT_READY"
        ),
        dataset_freeze_status=(
            "AUTHORIZED"
            if verification.is_current_pass and PRODUCTION_CONDITIONED_DATASET_FREEZE in scopes
            else "NOT_AUTHORIZED"
        ),
        authorized_scopes=scopes,
    )


def labeling_downstream_consequences(verification: AuditVerification) -> tuple[str, ...]:
    scopes = verification.authorized_scopes
    consequences = [
        "BROAD_PROPOSALS_AUTHORIZED" if CONSERVATIVE_PROPOSAL_GENERATION in scopes else "BROAD_PROPOSALS_BLOCKED",
        "HUMAN_TRUTH_AUTHORIZED" if HUMAN_TRUTH in scopes else "HUMAN_TRUTH_NOT_AUTHORIZED",
        "CALIBRATION_READY" if CALIBRATION_READINESS in scopes else "CALIBRATION_NOT_AUTHORIZED",
        "CONDITIONED_VIEW_AUTHORIZED" if CONDITIONED_VIEW_CANDIDATES in scopes else "CONDITIONED_VIEW_NOT_AUTHORIZED",
        "PRODUCTION_FREEZE_AUTHORIZED"
        if PRODUCTION_CONDITIONED_DATASET_FREEZE in scopes
        else "PRODUCTION_FREEZE_NOT_AUTHORIZED",
    ]
    return tuple(consequences)


def _configured_evidence_paths(context: ProjectContext) -> tuple[Path | None, Path | None]:
    labeling = context.config.get("labeling", {}) if isinstance(context.config, Mapping) else {}
    if not isinstance(labeling, Mapping):
        return None, None
    return (
        _resolve(context.project_root, labeling.get("audit_report")),
        _resolve(context.project_root, labeling.get("audit_hashes")),
    )


def _resolve(root: Path, raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _authoritative_source_commit(context: ProjectContext) -> str | None:
    labeling = context.config.get("labeling", {}) if isinstance(context.config, Mapping) else {}
    if isinstance(labeling, Mapping):
        configured = labeling.get("audit_stage_source_commit")
        if isinstance(configured, str) and configured:
            return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=context.project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None
