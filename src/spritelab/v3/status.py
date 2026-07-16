"""Derive authoritative v3 project state from manifests and audit artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.evaluation.audit_identity import (
    MEMORIZATION_AUDIT_BOUND_FILES as MEMORIZATION_AUDIT_BOUND_FILES,
)
from spritelab.evaluation.audit_identity import (
    MEMORIZATION_AUDIT_CODE_IDENTITY_VERSION,
    MEMORIZATION_AUDIT_SUBSYSTEM,
    MemorizationAuditIdentityError,
    memorization_audit_code_identity,
    recorded_identity_errors,
)
from spritelab.product_core.audit_evidence import (
    CALIBRATION_READINESS,
    CONDITIONED_VIEW_CANDIDATES,
    PRODUCTION_CONDITIONED_DATASET_FREEZE,
    ApplicabilityStatus,
    AuditVerdict,
    AuditVerification,
)
from spritelab.product_core.contracts import ProjectContext
from spritelab.product_features.dataset.certification import labeling_audit_verification
from spritelab.product_features.evaluation.checkpoints import discover_checkpoint_candidates
from spritelab.training.campaign import CampaignValidationError, training_code_identity_source_paths
from spritelab.v3.config import ProjectConfig, configured_training_identities
from spritelab.v3.model import AuditStatus, Evidence, ProjectState, StageState, StageStatus


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path | None, source_commit: str | None = None) -> list[Evidence]:
    if path is None or not path.is_file():
        return []
    return [Evidence(path=str(path), sha256=_sha256(path), source_commit=source_commit)]


def _source_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _training_audit_status(config: ProjectConfig, report: dict[str, Any] | None) -> AuditStatus:
    if report is None:
        return AuditStatus.NOT_AUDITED
    hashes_path = config.path_for("training", "audit_hashes")
    hashes = _read_json(hashes_path)
    if not hashes or not isinstance(hashes.get("files"), list):
        return AuditStatus.STALE
    audited_paths: list[str] = []
    for item in hashes["files"]:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256_before"):
            return AuditStatus.STALE
        relative = Path(str(item["path"])).as_posix()
        audited_paths.append(relative)
        target = config.root / relative
        if not target.is_file() or _sha256(target) != item["sha256_before"]:
            return AuditStatus.STALE
    if len(audited_paths) != len(set(audited_paths)):
        return AuditStatus.STALE
    try:
        required_paths = {
            path.relative_to(config.root.resolve()).as_posix()
            for path in training_code_identity_source_paths(config.root)
        }
    except (OSError, ValueError, CampaignValidationError):
        return AuditStatus.STALE
    if not required_paths.issubset(set(audited_paths)):
        return AuditStatus.STALE
    gates = report.get("gates", {})
    verdicts = [str(value).upper() for value in gates.values()] if isinstance(gates, dict) else []
    if "FAIL" in verdicts:
        return AuditStatus.FAIL
    return AuditStatus.PASS if verdicts and all(value == "PASS" for value in verdicts) else AuditStatus.INCONCLUSIVE


@dataclass(frozen=True)
class MemorizationAuditApplicability:
    """One shared, fail-closed applicability result for every status surface."""

    status: AuditStatus
    reasons: tuple[str, ...]
    report_verdict: str | None
    current_identity: Mapping[str, Any] | None

    @property
    def applicable(self) -> bool:
        return self.status in {AuditStatus.PASS, AuditStatus.FAIL, AuditStatus.INCONCLUSIVE}

    @property
    def identity_current(self) -> bool:
        return self.current_identity is not None and "code_identity_changed" not in self.reasons


def verify_memorization_audit_applicability(
    root: Path,
    report: Mapping[str, Any] | None,
) -> MemorizationAuditApplicability:
    """Recompute v3 identity and classify legacy, malformed, and cross-subsystem evidence."""

    if report is None:
        return MemorizationAuditApplicability(AuditStatus.NOT_AUDITED, ("audit_report_missing",), None, None)
    recorded = report.get("code_identity")
    if (
        not isinstance(recorded, Mapping)
        or recorded.get("contract_version") != MEMORIZATION_AUDIT_CODE_IDENTITY_VERSION
    ):
        return MemorizationAuditApplicability(
            AuditStatus.STALE,
            ("legacy_or_incomplete_v3_code_identity",),
            str(report.get("overall_verdict") or "").upper() or None,
            None,
        )
    if report.get("subsystem") != MEMORIZATION_AUDIT_SUBSYSTEM:
        return MemorizationAuditApplicability(
            AuditStatus.NOT_COMPARABLE,
            ("audit_subsystem_mismatch",),
            str(report.get("overall_verdict") or "").upper() or None,
            None,
        )
    malformed = recorded_identity_errors(recorded)
    if malformed:
        return MemorizationAuditApplicability(
            AuditStatus.STALE,
            malformed,
            str(report.get("overall_verdict") or "").upper() or None,
            None,
        )
    try:
        current = memorization_audit_code_identity(root)
    except MemorizationAuditIdentityError as exc:
        return MemorizationAuditApplicability(
            AuditStatus.NOT_COMPARABLE,
            (f"current_code_identity_unavailable:{exc}",),
            str(report.get("overall_verdict") or "").upper() or None,
            None,
        )
    if dict(recorded) != current:
        return MemorizationAuditApplicability(
            AuditStatus.STALE,
            ("code_identity_changed",),
            str(report.get("overall_verdict") or "").upper() or None,
            current,
        )
    verdict = str(report.get("overall_verdict") or "").upper()
    if verdict not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        return MemorizationAuditApplicability(
            AuditStatus.INCONCLUSIVE,
            ("audit_verdict_missing_or_invalid",),
            verdict or None,
            current,
        )
    return MemorizationAuditApplicability(AuditStatus(verdict), (), verdict, current)


def _memorization_audit_status(config: ProjectConfig, report: dict[str, Any] | None) -> AuditStatus:
    return verify_memorization_audit_applicability(config.root, report).status


def _count_disagreements(campaign: dict[str, Any]) -> tuple[int | None, int | None]:
    checks = campaign.get("health_checks")
    if not isinstance(checks, list) or not checks or not isinstance(checks[-1], dict):
        return None, None
    comparisons = checks[-1].get("critical_field_comparisons")
    rate = checks[-1].get("critical_field_disagreement_rate")
    if not isinstance(comparisons, int) or not isinstance(rate, (int, float)):
        return None, comparisons if isinstance(comparisons, int) else None
    return round(comparisons * float(rate)), comparisons


def _labeling_audit_status(verification: AuditVerification) -> AuditStatus:
    if verification.applicability_status is ApplicabilityStatus.STALE:
        return AuditStatus.STALE
    if verification.applicability_status is ApplicabilityStatus.NOT_COMPARABLE:
        return AuditStatus.NOT_COMPARABLE
    if verification.report_verdict is AuditVerdict.PASS:
        return AuditStatus.PASS
    if verification.report_verdict is AuditVerdict.FAIL:
        return AuditStatus.FAIL
    if verification.report_verdict is AuditVerdict.INCONCLUSIVE:
        return AuditStatus.INCONCLUSIVE
    return AuditStatus.NOT_AUDITED


def build_project_state(config: ProjectConfig) -> ProjectState:
    raw_path = config.path_for("dataset", "raw_provenance_report")
    extraction_path = config.path_for("dataset", "extraction_report")
    suitability_path = config.path_for("dataset", "suitability_report")
    view_path = config.path_for("dataset", "view_manifest")
    freeze_path = config.path_for("dataset", "freeze_manifest")
    label_path = config.path_for("labeling", "campaign_report")
    label_audit_path = config.path_for("labeling", "audit_report")
    training_path = config.path_for("training", "audit_report")
    training_hashes_path = config.path_for("training", "audit_hashes")
    mem_path = config.path_for("evaluation", "memorization_audit")

    raw = _read_json(raw_path)
    extraction = _read_json(extraction_path)
    suitability = _read_json(suitability_path)
    view = _read_json(view_path)
    freeze = _read_json(freeze_path)
    labels = _read_json(label_path)
    label_audit = _read_json(label_audit_path)
    training = _read_json(training_path)
    mem = _read_json(mem_path)
    source_commit = _source_commit(config.root)
    labeling_verification = labeling_audit_verification(
        ProjectContext(config.root, config.values, config.path, config.runs_dir)
    )
    labeling_audit_status = _labeling_audit_status(labeling_verification)
    stages: list[StageState] = []

    raw_passed = bool(raw and raw.get("source_gate_passed") is True and raw.get("remaining_unresolved_sources") == 0)
    raw_metrics = {
        key: raw[key]
        for key in (
            "sources_verified",
            "sources_excluded",
            "remaining_unresolved_sources",
            "missing_downloads_still_requiring_external_retrieval",
            "newly_eligible_record_count",
        )
        if raw and key in raw
    }
    stages.append(
        StageState(
            key="raw-source-provenance",
            title="Raw-source provenance",
            status=StageStatus.COMPLETE if raw_passed else StageStatus.BLOCKED,
            explanation=(
                "Every source binding has a verified/recovered or explicit terminal disposition."
                if raw_passed
                else "The source provenance gate has not produced a complete authoritative resolution."
            ),
            blockers=[] if raw_passed else ["Raw-source provenance gate is incomplete or missing."],
            warnings=(
                ["At least one source still requires manual external retrieval."]
                if raw and raw.get("missing_downloads_still_requiring_external_retrieval")
                else []
            ),
            evidence=_evidence(raw_path),
            source_commit=source_commit,
            next_action="Continue to extraction using only eligible records."
            if raw_passed
            else "Complete provenance remediation.",
            next_command="python -m spritelab v3 dataset build",
            metrics=raw_metrics,
        )
    )

    deterministic = bool(extraction and extraction.get("determinism", {}).get("byte_identical_rgba_outputs") is True)
    stages.append(
        StageState(
            key="extraction",
            title="Extraction",
            status=StageStatus.COMPLETE if deterministic else StageStatus.BLOCKED,
            explanation=(
                "Deterministic extraction evidence exists; ambiguous inputs remain explicitly excluded."
                if deterministic
                else "Deterministic extraction evidence is missing or incomplete."
            ),
            blockers=[] if deterministic else ["Extraction determinism is not established."],
            evidence=_evidence(extraction_path),
            source_commit=source_commit,
            next_action="Evaluate extracted operations for suitability."
            if deterministic
            else "Run extraction remediation.",
            next_command="python -m spritelab v3 dataset build",
            metrics=extraction.get("remaining_ambiguity", {}) if extraction else {},
        )
    )

    suitability_ready = bool(
        suitability and isinstance(suitability.get("unique_extraction_suitability_status_counts"), dict)
    )
    stages.append(
        StageState(
            key="suitability",
            title="Suitability",
            status=StageStatus.COMPLETE if suitability_ready else StageStatus.BLOCKED,
            explanation=(
                "Suitability dispositions are recorded for unique extraction operations."
                if suitability_ready
                else "No authoritative suitability summary is available."
            ),
            blockers=[] if suitability_ready else ["Suitability dispositions are unavailable."],
            evidence=_evidence(suitability_path),
            source_commit=source_commit,
            next_action="Proceed to conservative semantic labeling."
            if suitability_ready
            else "Run suitability evaluation.",
            next_command="python -m spritelab v3 dataset build",
            metrics=suitability or {},
        )
    )

    label_stopped = bool(labels and labels.get("campaign_status") == "stopped_health_gate")
    disagreements, comparisons = _count_disagreements(labels or {})
    label_metrics = {
        key: labels[key]
        for key in ("pass_a_completed", "pass_b_completed", "pass_a_record_abstention_count")
        if labels and key in labels
    }
    if comparisons is not None:
        label_metrics.update({"critical_disagreements": disagreements, "critical_comparisons": comparisons})
    if labeling_audit_status == AuditStatus.STALE:
        labeling_stage_status = StageStatus.STALE
    elif labeling_audit_status == AuditStatus.FAIL:
        labeling_stage_status = StageStatus.FAILED
    elif labeling_audit_status == AuditStatus.PASS:
        labeling_stage_status = StageStatus.READY
    elif label_stopped:
        labeling_stage_status = StageStatus.NEEDS_REVIEW
    else:
        labeling_stage_status = StageStatus.BLOCKED
    labeling_blockers = []
    if labeling_audit_status != AuditStatus.PASS:
        labeling_blockers.append(f"Independent labeling audit is {labeling_audit_status.value}.")
    if label_stopped:
        labeling_blockers.append("Blind-label health gate failed; Pass B is not authorized.")
    stages.append(
        StageState(
            key="semantic-labeling",
            title="Semantic labeling",
            status=labeling_stage_status,
            explanation=(
                "The current independent reliability check authorizes only its declared labeling scopes."
                if labeling_audit_status == AuditStatus.PASS
                else "The latest independent labeling check found problems."
                if labeling_audit_status == AuditStatus.FAIL
                else "The labeling reliability evidence is stale for the current code and contracts."
                if labeling_audit_status == AuditStatus.STALE
                else "No comparable independent labeling reliability evidence is available."
            ),
            blockers=labeling_blockers,
            warnings=["Existing labels are proposals, not human ground truth."] if labels else [],
            evidence=_evidence(label_path) + _evidence(label_audit_path),
            source_commit=source_commit,
            next_action="Review disagreement causes before authorizing any resume.",
            next_command="python -m spritelab v3 review",
            resume_available=bool(labels and labels.get("resume"))
            and bool(label_audit and label_audit.get("resume_authorized")),
            audit=labeling_audit_status,
            production_authorized=labeling_verification.is_current_pass,
            metrics={
                **label_metrics,
                "applicability": labeling_verification.applicability_status.value,
                "applicability_reasons": list(labeling_verification.reasons),
                "authorized_scopes": list(labeling_verification.authorized_scopes),
                "artifact_status": labeling_verification.artifact_status.value,
            },
        )
    )

    calibrated = bool(
        labels
        and labels.get("labels_are_calibrated_truth") is True
        and labeling_verification.authorizes(CALIBRATION_READINESS)
    )
    stages.append(
        StageState(
            key="semantic-calibration",
            title="Semantic calibration",
            status=StageStatus.COMPLETE if calibrated else StageStatus.BLOCKED,
            explanation="Semantic labels are calibrated."
            if calibrated
            else "Label proposals have not been calibrated as truth.",
            blockers=[] if calibrated else ["Semantic labeling health and calibration requirements are not satisfied."],
            evidence=_evidence(label_path),
            source_commit=source_commit,
            next_action="Resolve and independently validate labeling disagreements.",
            next_command="python -m spritelab v3 review",
        )
    )

    view_complete = bool(view and view.get("candidate_dataset_created") is True and view.get("status") == "complete")
    view_is_conditioned = bool(
        view
        and (
            view.get("requires_semantic_labels") is True
            or "condition" in str(view.get("view", view.get("view_kind", ""))).casefold()
        )
    )
    if view_complete and view_is_conditioned and not labeling_verification.authorizes(CONDITIONED_VIEW_CANDIDATES):
        view_complete = False
    stages.append(
        StageState(
            key="dataset-v5-view-construction",
            title="Dataset-v5 view construction",
            status=StageStatus.COMPLETE if view_complete else StageStatus.BLOCKED,
            explanation="A candidate view manifest is complete."
            if view_complete
            else "The configured Dataset-v5 view is explicitly blocked.",
            blockers=[]
            if view_complete
            else ["Dataset-v5 view construction cannot pass the current label/calibration gates."],
            evidence=_evidence(view_path),
            source_commit=source_commit,
            next_action="Clear upstream labeling and calibration gates.",
            next_command="python -m spritelab v3 dataset build",
            metrics={"view": view.get("view"), "candidate_dataset_created": view.get("candidate_dataset_created")}
            if view
            else {},
        )
    )

    freeze_complete = bool(
        freeze and freeze.get("production_authorized") is True and freeze.get("status") == "complete"
    )
    freeze_is_conditioned = bool(
        freeze
        and (
            freeze.get("requires_semantic_labels") is True
            or "condition" in str(freeze.get("dataset_kind", freeze.get("view", ""))).casefold()
        )
    )
    if (
        freeze_complete
        and freeze_is_conditioned
        and not labeling_verification.authorizes(PRODUCTION_CONDITIONED_DATASET_FREEZE)
    ):
        freeze_complete = False
    stages.append(
        StageState(
            key="dataset-freeze",
            title="Dataset-v5 production freeze",
            status=StageStatus.COMPLETE if freeze_complete else StageStatus.BLOCKED,
            explanation="An authoritative production freeze is configured and authorized."
            if freeze_complete
            else "No authorized production Dataset-v5 freeze is configured.",
            blockers=[] if freeze_complete else ["Dataset-v5 is not production-frozen."],
            evidence=_evidence(freeze_path),
            source_commit=source_commit,
            next_action="Clear every dataset gate and obtain explicit freeze authorization.",
            next_command="python -m spritelab v3 status",
            production_authorized=freeze_complete,
        )
    )

    training_audit = _training_audit_status(config, training)
    training_failed = training_audit == AuditStatus.FAIL
    training_stale = training_audit == AuditStatus.STALE
    failed_gates = sorted(key for key, value in (training or {}).get("gates", {}).items() if value == "FAIL")
    stages.append(
        StageState(
            key="training-infrastructure-audit",
            title="Training infrastructure audit",
            status=StageStatus.STALE
            if training_stale
            else (StageStatus.FAILED if training_failed else StageStatus.INCONCLUSIVE),
            explanation=(
                "The latest independent audit is stale because an audited identity changed."
                if training_stale
                else "The latest applicable independent audit reports failed safety gates."
                if training_failed
                else "No conclusive applicable training-infrastructure audit is available."
            ),
            blockers=[f"Independent training audit is {training_audit.value}."],
            evidence=_evidence(training_path) + _evidence(training_hashes_path),
            source_commit=source_commit,
            next_action="Remediate failed gates and commission a new independent audit.",
            next_command="python -m spritelab v3 explain training-audit",
            audit=training_audit,
            metrics={"failed_gates": failed_gates, "training_runs": (training or {}).get("training_runs")},
        )
    )

    training_blockers = []
    if not freeze_complete:
        training_blockers.append("Dataset-v5 is not frozen.")
    if training_audit != AuditStatus.PASS:
        training_blockers.append(f"Independent training-infrastructure audit: {training_audit.value}.")
    if not config.values["execution"]["allow_training"]:
        training_blockers.append("Project execution policy does not authorize training.")
    stages.append(
        StageState(
            key="training-campaign",
            title="Training campaign",
            status=StageStatus.BLOCKED if training_blockers else StageStatus.READY,
            explanation="Training is blocked by authoritative prerequisites."
            if training_blockers
            else "Training prerequisites are satisfied; confirmation is still required.",
            blockers=training_blockers,
            evidence=_evidence(freeze_path) + _evidence(training_path),
            source_commit=source_commit,
            next_action="Resolve every dataset, audit, and execution-policy blocker.",
            next_command="python -m spritelab v3 status",
            production_authorized=not training_blockers,
        )
    )

    checkpoint = config.path_for("evaluation", "checkpoint")
    benchmark = config.path_for("evaluation", "benchmark")
    training_dataset_identity, training_view_identity = configured_training_identities(config.values)
    bound_checkpoint = None
    eval_blockers = []
    if checkpoint is None or not checkpoint.is_file():
        eval_blockers.append("Evaluation checkpoint is missing.")
    if benchmark is None or not benchmark.exists():
        eval_blockers.append("Evaluation benchmark is missing.")
    if training_dataset_identity is None:
        eval_blockers.append("Active training dataset identity is missing or malformed.")
    if training_view_identity is None:
        eval_blockers.append("Active training view identity is missing or malformed.")
    if checkpoint is not None and checkpoint.is_file() and training_dataset_identity and training_view_identity:
        try:
            catalog = discover_checkpoint_candidates(
                config.runs_dir,
                project_root=config.root,
                active_dataset_identity=training_dataset_identity,
                active_view_identity=training_view_identity,
            )
            configured_checkpoint = checkpoint.resolve()
            bound_checkpoint = next(
                (
                    candidate
                    for candidate in catalog.eligible
                    if candidate.path is not None and candidate.path.resolve() == configured_checkpoint
                ),
                None,
            )
        except OSError:
            bound_checkpoint = None
        if bound_checkpoint is None:
            eval_blockers.append(
                "Configured checkpoint is not an eligible checkpoint bound to the active training dataset and view."
            )
    if not config.values["execution"]["allow_generation"]:
        eval_blockers.append("Project execution policy does not authorize generation.")
    stages.append(
        StageState(
            key="evaluation-generation",
            title="Evaluation generation",
            status=StageStatus.BLOCKED if eval_blockers else StageStatus.READY,
            explanation="Evaluation generation prerequisites are not satisfied."
            if eval_blockers
            else "Evaluation generation is ready for explicit confirmation.",
            blockers=eval_blockers,
            evidence=_evidence(checkpoint) + _evidence(benchmark),
            source_commit=source_commit,
            next_action="Configure an immutable checkpoint and benchmark identity.",
            next_command="python -m spritelab v3 status",
            production_authorized=not eval_blockers,
            metrics={
                "training_dataset_identity": training_dataset_identity,
                "training_view_identity": training_view_identity,
                "identity_binding_complete": bound_checkpoint is not None,
                "bound_checkpoint_id": bound_checkpoint.checkpoint_id if bound_checkpoint else None,
            },
        )
    )
    stages.append(
        StageState(
            key="evaluation-metrics",
            title="Evaluation metrics",
            status=StageStatus.NOT_STARTED if eval_blockers else StageStatus.READY,
            explanation="No comparable evaluation output is available yet."
            if eval_blockers
            else "Metrics can run after generation completes.",
            blockers=["Evaluation generation has not completed."] if eval_blockers else [],
            source_commit=source_commit,
            next_action="Complete an authorized evaluation generation run.",
            next_command="python -m spritelab v3 eval",
        )
    )

    mem_verification = verify_memorization_audit_applicability(config.root, mem)
    mem_audit = _memorization_audit_status(config, mem)
    mem_verdict_failed = mem_audit == AuditStatus.FAIL
    mem_findings = [
        {"id": item.get("id"), "severity": item.get("severity"), "summary": item.get("summary")}
        for item in (mem or {}).get("findings", [])
        if isinstance(item, dict)
    ]
    stages.append(
        StageState(
            key="memorization-review",
            title="Memorization review",
            status=StageStatus.STALE
            if mem_audit == AuditStatus.STALE
            else (StageStatus.FAILED if mem_verdict_failed else StageStatus.INCONCLUSIVE),
            explanation=(
                "The latest memorization audit is stale for the current evaluation code."
                if mem_audit == AuditStatus.STALE
                else "The latest applicable independent audit found fail-closed review-integrity defects."
                if mem_verdict_failed
                else "No conclusive applicable memorization audit is available."
            ),
            blockers=[f"Independent memorization audit: {mem_audit.value}."],
            evidence=_evidence(mem_path, str(mem.get("commit")) if mem else None),
            source_commit=str(mem.get("commit")) if mem else source_commit,
            next_action="Remediate review integrity and commission a new independent audit.",
            next_command="python -m spritelab v3 explain memorization",
            audit=mem_audit,
            metrics={
                "findings": mem_findings,
                "applicability_reasons": list(mem_verification.reasons),
                "code_identity_current": mem_verification.identity_current,
            },
        )
    )

    promotion_authorized = bool(
        mem
        and mem_audit == AuditStatus.PASS
        and mem.get("authorization", {}).get("checkpoint_promotion") is True
        and config.values["execution"]["allow_promotion"]
        and not eval_blockers
    )
    promotion_blockers: list[str] = []
    if eval_blockers:
        promotion_blockers.extend(f"Evaluation prerequisite: {blocker}" for blocker in eval_blockers)
    if mem_audit != AuditStatus.PASS:
        promotion_blockers.append(f"Independent memorization audit: {mem_audit.value}.")
    elif not mem or mem.get("authorization", {}).get("checkpoint_promotion") is not True:
        promotion_blockers.append("Applicable memorization evidence does not authorize checkpoint promotion.")
    if not config.values["execution"]["allow_promotion"]:
        promotion_blockers.append("Project execution policy does not authorize checkpoint promotion.")
    stages.append(
        StageState(
            key="promotion-decision",
            title="Promotion decision",
            status=StageStatus.READY if promotion_authorized else StageStatus.BLOCKED,
            explanation="Promotion may be evaluated."
            if promotion_authorized
            else "Checkpoint promotion remains fail-closed.",
            blockers=promotion_blockers,
            evidence=_evidence(checkpoint) + _evidence(benchmark) + _evidence(mem_path),
            source_commit=source_commit,
            next_action="Resolve evaluation, memorization, and authorization gates.",
            next_command="python -m spritelab v3 status",
            production_authorized=promotion_authorized,
            metrics={
                "training_dataset_identity": training_dataset_identity,
                "training_view_identity": training_view_identity,
                "identity_binding_complete": bound_checkpoint is not None,
                "bound_checkpoint_id": bound_checkpoint.checkpoint_id if bound_checkpoint else None,
            },
        )
    )

    warnings = [] if config.path else ["No spritelab.yaml is active; defaults are for inspection only."]
    return ProjectState(
        project_name=config.name,
        project_root=config.root,
        config_path=config.path,
        source_commit=source_commit,
        stages=stages,
        warnings=warnings,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
