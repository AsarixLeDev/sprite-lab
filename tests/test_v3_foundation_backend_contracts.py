"""Cross-subsystem invariants for the v3 product/backend convergence boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from spritelab.product_core import (
    BackendAdapterSet,
    BackendCapabilitySnapshot,
    CalibrationReadinessSnapshot,
    CapabilityState,
    LabelingDatasetEligibility,
    MemorizationReviewItem,
    ProductBackendGateway,
    ProductResult,
    ProductStatus,
    ProjectContext,
    PromotionStatusSnapshot,
    ReviewEvidenceState,
    TrainingPlanSnapshot,
)

LABELING_ACTIONS = ("labeling.conditioned", "dataset.conditioned_freeze")
TRAINING_ACTIONS = ("training.launch",)
EVALUATION_ACTIONS = ("checkpoint.promotion",)


def _ready(backend_id: str, actions: tuple[str, ...]) -> BackendCapabilitySnapshot:
    return BackendCapabilitySnapshot(
        backend_id=backend_id,
        technical_state=CapabilityState.READY,
        independent_certification_state=CapabilityState.READY,
        production_state=CapabilityState.READY,
        normal_actions=actions,
    )


@dataclass
class _CapabilityProvider:
    capability: BackendCapabilitySnapshot
    probes: int = 0

    def probe_labeling_capability(self, _context: ProjectContext) -> BackendCapabilitySnapshot:
        self.probes += 1
        return self.capability

    def probe_training_capability(self, _context: ProjectContext) -> BackendCapabilitySnapshot:
        self.probes += 1
        return self.capability

    def probe_evaluation_capability(self, _context: ProjectContext) -> BackendCapabilitySnapshot:
        self.probes += 1
        return self.capability


@dataclass
class _TrainingPlanProvider:
    plan: TrainingPlanSnapshot
    calls: int = 0

    def training_plan(self, _context: ProjectContext) -> TrainingPlanSnapshot:
        self.calls += 1
        return self.plan


@dataclass
class _TrainingExecutionProvider:
    calls: int = 0

    def execute_training(self, _plan: TrainingPlanSnapshot, _context: ProjectContext) -> ProductResult:
        self.calls += 1
        return ProductResult(ProductStatus.COMPLETE, "Synthetic execution seam invoked.")


@dataclass
class _PromotionStatusProvider:
    status: PromotionStatusSnapshot
    calls: int = 0

    def promotion_status(self, _context: ProjectContext) -> PromotionStatusSnapshot:
        self.calls += 1
        return self.status


class _NeverCalledProvider:
    def __init__(self) -> None:
        self.calls = 0

    def labeling_review_queue(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("review queue must not be read during capability probing")

    def calibration_readiness(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("calibration must not be read during capability probing")

    def training_plan(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("training plan must not be built during capability probing")

    def execute_training(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("training must not launch during capability probing")

    def memorization_review_queue(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("memorization queue must not be read during capability probing")

    def promotion_status(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("promotion status must not be read during capability probing")


def test_remediation_pass_cannot_become_independent_audit_pass() -> None:
    capability = BackendCapabilitySnapshot.remediated_pending_certification("labeling", normal_actions=LABELING_ACTIONS)
    assert capability.technically_ready is True
    assert capability.independent_certification_state is CapabilityState.CERTIFICATION_PENDING
    assert capability.production_authorized is False
    assert capability.authorize("labeling.conditioned").authorized is False
    with pytest.raises(ValueError, match="independent certification"):
        BackendCapabilitySnapshot(
            backend_id="invalid",
            technical_state=CapabilityState.READY,
            independent_certification_state=CapabilityState.CERTIFICATION_PENDING,
            production_state=CapabilityState.READY,
        )


def test_model_model_agreement_is_not_displayed_as_human_truth() -> None:
    readiness = CalibrationReadinessSnapshot(
        ready=False,
        reviewed_human_truth_count=0,
        model_model_agreement_count=41,
        independent_certification_state=CapabilityState.CERTIFICATION_PENDING,
    )
    payload = readiness.to_public_dict()
    assert payload["reviewed_human_truth_count"] == 0
    assert payload["model_model_agreement_count"] == 41
    assert payload["model_model_agreement_is_human_truth"] is False
    with pytest.raises(ValueError, match="reviewed human truth"):
        CalibrationReadinessSnapshot(
            ready=True,
            reviewed_human_truth_count=0,
            model_model_agreement_count=100,
            independent_certification_state=CapabilityState.READY,
        )


def test_semantic_abstention_preserves_image_only_dataset_eligibility() -> None:
    eligibility = LabelingDatasetEligibility(image_only_eligible=True, semantic_abstained=True)
    assert eligibility.eligible_for("image_only") is True
    assert eligibility.eligible_for("conditioned") is False


@pytest.mark.parametrize(
    "audit_state",
    [CapabilityState.BLOCKED, CapabilityState.CERTIFICATION_PENDING],
)
def test_failed_or_pending_labeling_audit_blocks_conditioned_freeze(audit_state: CapabilityState) -> None:
    capability = BackendCapabilitySnapshot(
        backend_id="labeling",
        technical_state=CapabilityState.READY,
        independent_certification_state=audit_state,
        production_state=CapabilityState.BLOCKED,
        normal_actions=LABELING_ACTIONS,
    )
    assert capability.authorize("dataset.conditioned_freeze").authorized is False


def test_pending_training_audit_prevents_training_launch(tmp_path: Path) -> None:
    capability = _CapabilityProvider(
        BackendCapabilitySnapshot.remediated_pending_certification("training", normal_actions=TRAINING_ACTIONS)
    )
    plan = _TrainingPlanProvider(TrainingPlanSnapshot("synthetic-plan"))
    execution = _TrainingExecutionProvider()
    gateway = ProductBackendGateway(
        BackendAdapterSet(training_capability=capability, training_plan=plan, training_execution=execution)
    )
    result = gateway.request_training_launch(ProjectContext(tmp_path), confirmed=True)
    assert result.authorized is False
    assert result.executed is False
    assert plan.calls == 0
    assert execution.calls == 0


def test_product_confirmation_cannot_bypass_training_execution_gates(tmp_path: Path) -> None:
    capability = _CapabilityProvider(_ready("training", TRAINING_ACTIONS))
    plan = _TrainingPlanProvider(TrainingPlanSnapshot("synthetic-plan", blocker_codes=("resume_identity_mismatch",)))
    execution = _TrainingExecutionProvider()
    gateway = ProductBackendGateway(
        BackendAdapterSet(training_capability=capability, training_plan=plan, training_execution=execution)
    )
    result = gateway.request_training_launch(ProjectContext(tmp_path), confirmed=True)
    assert result.reason == "backend_plan_blocked"
    assert result.executed is False
    assert execution.calls == 0


def test_unsafe_resume_is_absent_from_normal_product_capabilities(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe resume"):
        _ready("training", ("training.unsafe-resume",))
    capability = _CapabilityProvider(_ready("training", TRAINING_ACTIONS))
    plan = _TrainingPlanProvider(TrainingPlanSnapshot("synthetic-plan", unsafe_resume_requested=True))
    execution = _TrainingExecutionProvider()
    gateway = ProductBackendGateway(
        BackendAdapterSet(training_capability=capability, training_plan=plan, training_execution=execution)
    )
    result = gateway.request_training_launch(ProjectContext(tmp_path), confirmed=True)
    assert result.reason == "unsafe_resume_not_available"
    assert execution.calls == 0


def test_pending_memorization_audit_prevents_promotion_authorization(tmp_path: Path) -> None:
    capability = _CapabilityProvider(
        BackendCapabilitySnapshot.remediated_pending_certification("evaluation", normal_actions=EVALUATION_ACTIONS)
    )
    promotion = _PromotionStatusProvider(PromotionStatusSnapshot(ReviewEvidenceState.COMPLETE, True, False, True))
    gateway = ProductBackendGateway(BackendAdapterSet(evaluation_capability=capability, promotion_status=promotion))
    result = gateway.request_promotion_authorization(ProjectContext(tmp_path))
    assert result.authorized is False
    assert result.executed is False
    assert promotion.calls == 0


def test_unsigned_review_events_remain_non_authoritative_through_adapter_contract() -> None:
    item = MemorizationReviewItem(
        item_id="pair-1",
        authoritative=False,
        evidence_state=ReviewEvidenceState.COMPLETE,
        hard_evidence=False,
        backend_reviewable=True,
    )
    assert item.reviewable is False
    assert item.authoritative is False


def test_hard_memorization_evidence_remains_unreviewable() -> None:
    item = MemorizationReviewItem(
        item_id="pair-hard",
        authoritative=True,
        evidence_state=ReviewEvidenceState.COMPLETE,
        hard_evidence=True,
        backend_reviewable=True,
    )
    assert item.reviewable is False


@pytest.mark.parametrize("state", [ReviewEvidenceState.INCOMPLETE, ReviewEvidenceState.NOT_COMPARABLE])
def test_missing_or_malformed_review_evidence_does_not_become_pass(state: ReviewEvidenceState) -> None:
    status = PromotionStatusSnapshot(
        evidence_state=state,
        review_log_authoritative=False,
        hard_evidence_blocked=False,
        backend_decision_authorized=True,
    )
    assert status.authoritative_authorization is False
    assert status.to_public_dict()["authoritative_authorization"] is False


def test_stale_audits_do_not_authorize_downstream_actions() -> None:
    for backend_id, action in (
        ("labeling", "dataset.conditioned_freeze"),
        ("training", "training.launch"),
        ("evaluation", "checkpoint.promotion"),
    ):
        capability = BackendCapabilitySnapshot(
            backend_id=backend_id,
            technical_state=CapabilityState.READY,
            independent_certification_state=CapabilityState.STALE,
            production_state=CapabilityState.STALE,
            normal_actions=(action,),
        )
        assert capability.authorize(action).authorized is False
        assert capability.authorize(action).reason == "certification_stale"


def test_technical_readiness_and_production_authorization_are_separate() -> None:
    capability = BackendCapabilitySnapshot.remediated_pending_certification("training")
    payload = capability.to_public_dict()
    assert payload["technically_ready"] is True
    assert payload["production_authorized"] is False


def test_backend_status_serialization_contains_no_secrets(tmp_path: Path) -> None:
    secret = "super-secret-token"
    capability = _CapabilityProvider(
        BackendCapabilitySnapshot(
            backend_id="labeling",
            technical_state=CapabilityState.READY,
            independent_certification_state=CapabilityState.CERTIFICATION_PENDING,
            production_state=CapabilityState.BLOCKED,
            blocker_codes=(f"api_key={secret}",),
            internal_details={"authorization": f"Bearer {secret}"},
        )
    )
    payload = BackendAdapterSet(labeling_capability=capability).public_status(ProjectContext(tmp_path))
    serialized = json.dumps(payload)
    assert secret not in serialized
    assert "api_key" not in serialized
    assert "authorization" not in serialized


def test_capability_probing_starts_no_provider_trainer_generator_or_cloud_job(tmp_path: Path) -> None:
    labeling = _CapabilityProvider(
        BackendCapabilitySnapshot.remediated_pending_certification("labeling", normal_actions=LABELING_ACTIONS)
    )
    training = _CapabilityProvider(_ready("training", TRAINING_ACTIONS))
    evaluation = _CapabilityProvider(_ready("evaluation", EVALUATION_ACTIONS))
    never = _NeverCalledProvider()
    adapters = BackendAdapterSet(
        labeling_capability=labeling,
        labeling_review_queue=never,
        calibration_readiness=never,
        training_capability=training,
        training_plan=never,
        training_execution=never,
        evaluation_capability=evaluation,
        memorization_review_queue=never,
        promotion_status=never,
    )
    payload = adapters.public_status(ProjectContext(tmp_path))
    assert set(payload["capabilities"]) == {"labeling", "training", "evaluation"}
    assert (labeling.probes, training.probes, evaluation.probes) == (1, 1, 1)
    assert never.calls == 0
