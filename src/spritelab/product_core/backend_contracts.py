"""Neutral, dependency-injected contracts between the v3 product and backend packages.

The product owns presentation and coarse authorization composition only. Labeling,
training, and evaluation backends remain authoritative for their validation,
evidence, review, and execution decisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from spritelab.product_core.audit_evidence import (
    ApplicabilityStatus,
    AuditVerdict,
    VerifiedAuditEvidence,
)
from spritelab.product_core.contracts import ProductResult, ProjectContext

BACKEND_STATUS_SCHEMA = "spritelab.product.backend-status.v1"


class CapabilityState(str, Enum):
    """Technical, certification, and production states exposed by backend adapters."""

    READY = "READY"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    CERTIFICATION_PENDING = "CERTIFICATION_PENDING"
    STALE = "STALE"


class ReviewEvidenceState(str, Enum):
    """Controlled evidence states suitable for neutral product presentation."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True)
class ActionAuthorization:
    authorized: bool
    reason: str


@dataclass(frozen=True)
class BackendCapabilitySnapshot:
    """Read-only capability state supplied by an authoritative backend adapter.

    ``technical_state`` describes implementation availability. Independent audit
    state and production state are deliberately separate so a remediation result
    cannot silently become production authorization.
    """

    backend_id: str
    technical_state: CapabilityState
    independent_certification_state: CapabilityState
    production_state: CapabilityState
    normal_actions: tuple[str, ...] = ()
    authorized_scopes: tuple[str, ...] = ()
    blocker_codes: tuple[str, ...] = ()
    audit_evidence: VerifiedAuditEvidence | None = field(default=None, repr=False)
    internal_details: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id cannot be empty")
        if len(set(self.normal_actions)) != len(self.normal_actions):
            raise ValueError("normal_actions cannot contain duplicates")
        if any(not action.strip() for action in self.normal_actions):
            raise ValueError("normal_actions cannot contain empty values")
        if any("unsafe_resume" in action.lower().replace("-", "_") for action in self.normal_actions):
            raise ValueError("unsafe resume cannot be exposed through normal product capabilities")
        if len(set(self.authorized_scopes)) != len(self.authorized_scopes):
            raise ValueError("authorized_scopes cannot contain duplicates")
        if self.audit_evidence is not None and self.audit_evidence.subsystem != self.backend_id:
            raise ValueError("audit evidence for one subsystem cannot satisfy another subsystem gate")
        if self.backend_id == "labeling" and self.independent_certification_state is CapabilityState.READY:
            if self.audit_evidence is None:
                raise ValueError("labeling readiness requires verified audit evidence")
            if self.audit_evidence.verdict is not AuditVerdict.PASS:
                raise ValueError("labeling readiness requires an independent audit PASS")
            if self.audit_evidence.applicability_status not in {
                ApplicabilityStatus.APPLICABLE,
                ApplicabilityStatus.LEGACY_APPLICABLE,
            }:
                raise ValueError("labeling readiness requires applicable audit evidence")
            if set(self.authorized_scopes) - set(self.audit_evidence.authorized_scopes):
                raise ValueError("capability scopes must be authorized by the verified audit")
        if self.production_state is CapabilityState.READY and (
            self.technical_state is not CapabilityState.READY
            or self.independent_certification_state is not CapabilityState.READY
        ):
            raise ValueError("production readiness requires technical readiness and independent certification")

    @classmethod
    def remediated_pending_certification(
        cls,
        backend_id: str,
        *,
        normal_actions: tuple[str, ...] = (),
        blocker_codes: tuple[str, ...] = (),
    ) -> BackendCapabilitySnapshot:
        return cls(
            backend_id=backend_id,
            technical_state=CapabilityState.READY,
            independent_certification_state=CapabilityState.CERTIFICATION_PENDING,
            production_state=CapabilityState.BLOCKED,
            normal_actions=normal_actions,
            blocker_codes=blocker_codes,
        )

    @property
    def technically_ready(self) -> bool:
        return self.technical_state is CapabilityState.READY

    @property
    def production_authorized(self) -> bool:
        return (
            self.technical_state is CapabilityState.READY
            and self.independent_certification_state is CapabilityState.READY
            and self.production_state is CapabilityState.READY
        )

    def authorize(self, action: str) -> ActionAuthorization:
        if action not in self.normal_actions:
            return ActionAuthorization(False, "action_not_exposed")
        if self.independent_certification_state is CapabilityState.CERTIFICATION_PENDING:
            return ActionAuthorization(False, "certification_pending")
        if self.independent_certification_state is CapabilityState.STALE:
            return ActionAuthorization(False, "certification_stale")
        if self.backend_id == "labeling" and self.audit_evidence is None:
            return ActionAuthorization(False, "verified_audit_evidence_missing")
        if not self.production_authorized:
            return ActionAuthorization(False, f"production_{self.production_state.value.lower()}")
        return ActionAuthorization(True, "authorized_by_backend_capability")

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize only an allowlisted, secret-free product view."""

        return {
            "backend_id": self.backend_id,
            "technical_state": self.technical_state.value,
            "independent_certification_state": self.independent_certification_state.value,
            "production_state": self.production_state.value,
            "technically_ready": self.technically_ready,
            "production_authorized": self.production_authorized,
            "normal_actions": list(self.normal_actions),
            "authorized_scopes": list(self.authorized_scopes),
            "verified_audit_evidence": bool(self.audit_evidence),
            "blocker_count": len(self.blocker_codes),
        }

    def to_internal_dict(self) -> dict[str, Any]:
        """Serialize technical evidence only for explicit developer surfaces."""

        value = self.to_public_dict()
        value["blocker_codes"] = list(self.blocker_codes)
        value["audit_evidence"] = self.audit_evidence.to_internal_dict() if self.audit_evidence else None
        return value


@dataclass(frozen=True)
class LabelingReviewItem:
    item_id: str
    exception_case: bool
    summary: str = ""


@dataclass(frozen=True)
class LabelingReviewQueueSnapshot:
    items: tuple[LabelingReviewItem, ...]

    def __post_init__(self) -> None:
        if any(not item.exception_case for item in self.items):
            raise ValueError("normal labeling review queues may contain exception cases only")


@dataclass(frozen=True)
class CalibrationReadinessSnapshot:
    ready: bool
    reviewed_human_truth_count: int
    model_model_agreement_count: int
    independent_certification_state: CapabilityState
    blocker_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reviewed_human_truth_count < 0 or self.model_model_agreement_count < 0:
            raise ValueError("calibration counts cannot be negative")
        if self.ready and self.reviewed_human_truth_count == 0:
            raise ValueError("calibration cannot be ready without reviewed human truth")
        if self.ready and self.independent_certification_state is not CapabilityState.READY:
            raise ValueError("calibration readiness requires current independent certification")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reviewed_human_truth_count": self.reviewed_human_truth_count,
            "model_model_agreement_count": self.model_model_agreement_count,
            "model_model_agreement_is_human_truth": False,
            "independent_certification_state": self.independent_certification_state.value,
            "blocker_count": len(self.blocker_codes),
        }


@dataclass(frozen=True)
class LabelingDatasetEligibility:
    image_only_eligible: bool
    semantic_abstained: bool

    def eligible_for(self, dataset_kind: str) -> bool:
        if dataset_kind == "image_only":
            return self.image_only_eligible
        if dataset_kind == "conditioned":
            return not self.semantic_abstained
        return False


@dataclass(frozen=True)
class TrainingPlanSnapshot:
    plan_id: str
    blocker_codes: tuple[str, ...] = ()
    unsafe_resume_requested: bool = False

    @property
    def executable_from_normal_product(self) -> bool:
        return bool(self.plan_id.strip()) and not self.blocker_codes and not self.unsafe_resume_requested


@dataclass(frozen=True)
class MemorizationReviewItem:
    item_id: str
    authoritative: bool
    evidence_state: ReviewEvidenceState
    hard_evidence: bool
    backend_reviewable: bool

    @property
    def reviewable(self) -> bool:
        return (
            self.authoritative
            and self.evidence_state is ReviewEvidenceState.COMPLETE
            and not self.hard_evidence
            and self.backend_reviewable
        )


@dataclass(frozen=True)
class MemorizationReviewQueueSnapshot:
    items: tuple[MemorizationReviewItem, ...]


@dataclass(frozen=True)
class PromotionStatusSnapshot:
    evidence_state: ReviewEvidenceState
    review_log_authoritative: bool
    hard_evidence_blocked: bool
    backend_decision_authorized: bool
    reason_codes: tuple[str, ...] = ()

    @property
    def authoritative_authorization(self) -> bool:
        return (
            self.evidence_state is ReviewEvidenceState.COMPLETE
            and self.review_log_authoritative
            and not self.hard_evidence_blocked
            and self.backend_decision_authorized
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "evidence_state": self.evidence_state.value,
            "review_log_authoritative": self.review_log_authoritative,
            "hard_evidence_blocked": self.hard_evidence_blocked,
            "backend_decision_authorized": self.backend_decision_authorized,
            "authoritative_authorization": self.authoritative_authorization,
            "reason_count": len(self.reason_codes),
        }


@runtime_checkable
class LabelingBackendCapability(Protocol):
    def probe_labeling_capability(self, context: ProjectContext) -> BackendCapabilitySnapshot: ...


@runtime_checkable
class LabelingReviewQueueProvider(Protocol):
    def labeling_review_queue(
        self, context: ProjectContext, *, limit: int | None = None
    ) -> LabelingReviewQueueSnapshot: ...


@runtime_checkable
class CalibrationReadinessProvider(Protocol):
    def calibration_readiness(self, context: ProjectContext) -> CalibrationReadinessSnapshot: ...


@runtime_checkable
class TrainingBackendCapability(Protocol):
    def probe_training_capability(self, context: ProjectContext) -> BackendCapabilitySnapshot: ...


@runtime_checkable
class TrainingPlanProvider(Protocol):
    def training_plan(self, context: ProjectContext) -> TrainingPlanSnapshot: ...


@runtime_checkable
class TrainingExecutionProvider(Protocol):
    def execute_training(self, plan: TrainingPlanSnapshot, context: ProjectContext) -> ProductResult: ...


@runtime_checkable
class EvaluationBackendCapability(Protocol):
    def probe_evaluation_capability(self, context: ProjectContext) -> BackendCapabilitySnapshot: ...


@runtime_checkable
class MemorizationReviewQueueProvider(Protocol):
    def memorization_review_queue(
        self, context: ProjectContext, *, limit: int | None = None
    ) -> MemorizationReviewQueueSnapshot: ...


@runtime_checkable
class PromotionStatusProvider(Protocol):
    def promotion_status(self, context: ProjectContext) -> PromotionStatusSnapshot: ...


@dataclass(frozen=True)
class BackendAdapterSet:
    """Injected adapters; constructing or probing this set performs no execution."""

    labeling_capability: LabelingBackendCapability | None = None
    labeling_review_queue: LabelingReviewQueueProvider | None = None
    calibration_readiness: CalibrationReadinessProvider | None = None
    training_capability: TrainingBackendCapability | None = None
    training_plan: TrainingPlanProvider | None = None
    training_execution: TrainingExecutionProvider | None = None
    evaluation_capability: EvaluationBackendCapability | None = None
    memorization_review_queue: MemorizationReviewQueueProvider | None = None
    promotion_status: PromotionStatusProvider | None = None

    def probe_capabilities(self, context: ProjectContext) -> dict[str, BackendCapabilitySnapshot]:
        return {
            "labeling": self._probe_labeling(context),
            "training": self._probe_training(context),
            "evaluation": self._probe_evaluation(context),
        }

    def public_status(self, context: ProjectContext) -> dict[str, Any]:
        capabilities = self.probe_capabilities(context)
        return {
            "schema_version": BACKEND_STATUS_SCHEMA,
            "capabilities": {name: value.to_public_dict() for name, value in capabilities.items()},
        }

    def _probe_labeling(self, context: ProjectContext) -> BackendCapabilitySnapshot:
        if self.labeling_capability is None:
            return _unavailable_capability("labeling")
        return self.labeling_capability.probe_labeling_capability(context)

    def _probe_training(self, context: ProjectContext) -> BackendCapabilitySnapshot:
        if self.training_capability is None:
            return _unavailable_capability("training")
        return self.training_capability.probe_training_capability(context)

    def _probe_evaluation(self, context: ProjectContext) -> BackendCapabilitySnapshot:
        if self.evaluation_capability is None:
            return _unavailable_capability("evaluation")
        return self.evaluation_capability.probe_evaluation_capability(context)


@dataclass(frozen=True)
class BackendActionResult:
    authorized: bool
    executed: bool
    reason: str
    backend_result: ProductResult | None = None


@dataclass(frozen=True)
class ProductBackendGateway:
    """Fail-closed composition around injected backend authorities.

    The gateway does not reproduce campaign, review-signature, or promotion
    validation. It respects backend capability/plan/status decisions and never
    supplies an unsafe-resume path.
    """

    adapters: BackendAdapterSet

    def request_training_launch(self, context: ProjectContext, *, confirmed: bool) -> BackendActionResult:
        capability = self.adapters._probe_training(context)
        authorization = capability.authorize("training.launch")
        if not authorization.authorized:
            return BackendActionResult(False, False, authorization.reason)
        if not confirmed:
            return BackendActionResult(False, False, "confirmation_required")
        if self.adapters.training_plan is None:
            return BackendActionResult(False, False, "training_plan_unavailable")
        plan = self.adapters.training_plan.training_plan(context)
        if plan.unsafe_resume_requested:
            return BackendActionResult(False, False, "unsafe_resume_not_available")
        if not plan.executable_from_normal_product:
            return BackendActionResult(False, False, "backend_plan_blocked")
        if self.adapters.training_execution is None:
            return BackendActionResult(False, False, "training_execution_unavailable")
        result = self.adapters.training_execution.execute_training(plan, context)
        return BackendActionResult(True, True, "backend_execution_invoked", result)

    def request_promotion_authorization(self, context: ProjectContext) -> BackendActionResult:
        capability = self.adapters._probe_evaluation(context)
        authorization = capability.authorize("checkpoint.promotion")
        if not authorization.authorized:
            return BackendActionResult(False, False, authorization.reason)
        if self.adapters.promotion_status is None:
            return BackendActionResult(False, False, "promotion_status_unavailable")
        status = self.adapters.promotion_status.promotion_status(context)
        if not status.authoritative_authorization:
            return BackendActionResult(False, False, "backend_promotion_not_authorized")
        return BackendActionResult(True, False, "read_only_authorization")


def _unavailable_capability(backend_id: str) -> BackendCapabilitySnapshot:
    return BackendCapabilitySnapshot(
        backend_id=backend_id,
        technical_state=CapabilityState.UNAVAILABLE,
        independent_certification_state=CapabilityState.UNAVAILABLE,
        production_state=CapabilityState.UNAVAILABLE,
    )
