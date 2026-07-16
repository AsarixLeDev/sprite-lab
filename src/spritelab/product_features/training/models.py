"""Typed product-facing training models; backend configuration stays authoritative."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from spritelab.remote_compute import ComputeEstimate


class TrainingProfile(str, Enum):
    RECOMMENDED = "recommended"
    FAST_PREVIEW = "fast_preview"
    QUALITY = "quality"
    CUSTOM = "custom"


@dataclass(frozen=True)
class TrainingGate:
    gate_id: str
    passed: bool
    message: str
    resolution: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedTrainingPlan:
    profile: TrainingProfile
    model_label: str
    dataset_count: int | None
    dataset_ready: bool
    backend_id: str
    campaign: dict[str, Any] | None
    gates: tuple[TrainingGate, ...]
    estimate: ComputeEstimate
    resume_report: dict[str, Any] | None = None
    advanced_collapsed: bool = True

    @property
    def blockers(self) -> tuple[TrainingGate, ...]:
        return tuple(gate for gate in self.gates if not gate.passed)

    @property
    def ready(self) -> bool:
        return self.campaign is not None and not self.blockers

    def to_dict(self, *, include_campaign: bool = False) -> dict[str, Any]:
        result = {
            "profile": self.profile.value,
            "model_label": self.model_label,
            "dataset": {
                "images": self.dataset_count,
                "status": "Ready" if self.dataset_ready else "Blocked",
            },
            "compute": self.backend_id,
            "estimate": self.estimate.to_dict(),
            "ready": self.ready,
            "advanced_collapsed": self.advanced_collapsed,
            "gates": [asdict(gate) for gate in self.gates],
            "blockers": [asdict(gate) for gate in self.blockers],
            "campaign_identity": self.campaign.get("campaign_identity") if self.campaign else None,
            "seeds": list(self.campaign.get("seeds", ())) if self.campaign else [],
            "checkpoint_schedule": self.campaign.get("checkpoint_schedule", {}) if self.campaign else {},
            "resume": self.resume_report,
        }
        if include_campaign:
            result["campaign"] = self.campaign
        return result
