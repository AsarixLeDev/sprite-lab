"""Typed, JSON-safe product models for evaluation and exploratory generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class CheckpointAvailability(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"
    FOREIGN = "FOREIGN"
    UNSAFE_RESUME = "UNSAFE_RESUME"
    STALE_DATASET = "STALE_DATASET"
    STALE_VIEW = "STALE_VIEW"
    UNVERIFIED = "UNVERIFIED"
    MISSING = "MISSING"


@dataclass(frozen=True)
class CheckpointCandidate:
    """A checkpoint derived from one durable, verified training-run state."""

    checkpoint_id: str
    run_id: str
    friendly_run_name: str
    date: str | None
    training_profile: str
    completion_state: str
    dataset_identity: str | None
    dataset_identity_summary: str
    view_identity: str | None
    view_identity_summary: str
    checkpoint_step: int | None
    weights: str
    verification_state: str
    availability: CheckpointAvailability
    checkpoint_sha256: str | None = None
    unavailable_reasons: tuple[str, ...] = ()
    path: Path | None = field(default=None, repr=False, compare=False)
    run_directory: Path | None = field(default=None, repr=False, compare=False)

    @property
    def eligible(self) -> bool:
        return self.availability is CheckpointAvailability.ELIGIBLE

    def to_dict(self, *, technical_details: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "friendly_run_name": self.friendly_run_name,
            "date": self.date,
            "training_profile": self.training_profile,
            "completion_state": self.completion_state,
            "dataset_identity_summary": self.dataset_identity_summary,
            "view_identity_summary": self.view_identity_summary,
            "checkpoint_step": self.checkpoint_step,
            "weights": self.weights,
            "verification_state": self.verification_state,
            "availability": self.availability.value,
            "eligible": self.eligible,
            "unavailable_reasons": list(self.unavailable_reasons),
        }
        if technical_details:
            value.update(
                {
                    "dataset_identity": self.dataset_identity,
                    "view_identity": self.view_identity,
                    "checkpoint_path": str(self.path) if self.path else None,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "run_directory": str(self.run_directory) if self.run_directory else None,
                }
            )
        return value


@dataclass(frozen=True)
class CheckpointCatalog:
    eligible: tuple[CheckpointCandidate, ...]
    unavailable: tuple[CheckpointCandidate, ...]
    default_checkpoint_id: str | None

    def find(self, checkpoint_id: str | None, *, weights: str | None = None) -> CheckpointCandidate | None:
        requested = checkpoint_id or self.default_checkpoint_id
        if requested is None:
            return None
        direct = next((item for item in self.eligible if item.checkpoint_id == requested), None)
        if direct is None:
            return None
        normalized_weights = weights.lower() if weights else direct.weights
        if direct.weights == normalized_weights:
            return direct
        return next(
            (
                item
                for item in self.eligible
                if item.run_id == direct.run_id
                and item.checkpoint_step == direct.checkpoint_step
                and item.weights == normalized_weights
            ),
            None,
        )

    def to_dict(self, *, include_unavailable: bool = False, technical_details: bool = False) -> dict[str, Any]:
        value = {
            "label": "baseline — latest complete checkpoint",
            "default_checkpoint_id": self.default_checkpoint_id,
            "eligible": [item.to_dict(technical_details=technical_details) for item in self.eligible],
        }
        if include_unavailable:
            value["unavailable"] = [item.to_dict(technical_details=technical_details) for item in self.unavailable]
        return value


EVALUATION_STAGE_TITLES: tuple[tuple[str, str], ...] = (
    ("checkpoint_validation", "Checkpoint validation"),
    ("benchmark_validation", "Benchmark validation"),
    ("generation", "Generation"),
    ("structural_metrics", "Structural metrics"),
    ("conditional_metrics", "Conditional metrics"),
    ("diversity", "Diversity"),
    ("palette_analysis", "Palette analysis"),
    ("memorization_detector", "Memorization detector"),
    ("review_completeness", "Review completeness"),
    ("promotion_decision_report", "Promotion decision report"),
)


@dataclass
class EvaluationStage:
    key: str
    title: str
    status: str = "NOT_STARTED"
    message: str = ""
    current: int = 0
    total: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_evaluation_stages() -> list[EvaluationStage]:
    return [EvaluationStage(key, title) for key, title in EVALUATION_STAGE_TITLES]


@dataclass(frozen=True)
class ReviewFeatureLink:
    """Read-only link through the shared review feature; no review format is introduced here."""

    feature: str = "review"
    action_id: str = "review.open"
    queue_id: str = "memorization"
    href: str = "/review?queue=memorization"
    label: str = "Review candidates"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
