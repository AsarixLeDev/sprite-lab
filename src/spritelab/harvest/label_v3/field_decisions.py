"""Auto-Labeling v3: field-level decision contracts.

Each field decision is an independent unit that may be accepted, abstained,
quarantined, rejected, or marked as unknown/novel/ambiguous. No decision is
collapsed into a generic unknown token.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "field_decision_v3.1"

FieldState = Literal[
    "accepted",
    "abstained",
    "quarantined",
    "rejected",
    "unknown",
    "novel",
    "ambiguous",
    "unlabeled",
    "not_applicable",
]

DecisionReason = Literal[
    "strong_evidence_consensus",
    "hierarchy_fallback",
    "insufficient_evidence",
    "conflicting_evidence",
    "open_set_unknown",
    "novel_class",
    "ambiguous_identity",
    "impossible_combination",
    "provenance_failure",
    "image_integrity_failure",
    "missing_required_field",
    "calibration_insufficient",
    "human_correction",
    "human_flag",
    "legacy_override",
    "policy_rejection",
    "field_not_applicable",
]


@dataclass(frozen=True)
class FieldDecision:
    """One per-field decision with evidence, state, calibration, and lineage."""

    schema_version: str = SCHEMA_VERSION
    sprite_id: str = ""
    field_name: str = ""
    accepted_value: Any = None
    accepted_values: tuple[Any, ...] = ()
    hierarchy_node: str = ""
    candidates: tuple[str, ...] = ()
    n_best_alternatives: tuple[tuple[str, float], ...] = ()
    state: FieldState = "unlabeled"
    calibrated_estimate: float | None = None
    confidence_interval: tuple[float, float] | None = None
    calibration_support: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    excluded_evidence_refs: tuple[str, ...] = ()
    exclusion_reasons: tuple[str, ...] = ()
    contradiction_codes: tuple[str, ...] = ()
    open_set_state: Literal["in_distribution", "open_set", "novel", "unknown"] = "unknown"
    decision_reason: DecisionReason = "insufficient_evidence"
    policy_hash: str = ""
    config_hash: str = ""
    calibration_artifact_hash: str = ""


def field_decision_to_json(decision: FieldDecision) -> dict[str, Any]:
    ci = decision.confidence_interval
    return {
        "schema_version": decision.schema_version,
        "sprite_id": decision.sprite_id,
        "field": decision.field_name,
        "accepted_value": decision.accepted_value,
        "accepted_values": list(decision.accepted_values),
        "hierarchy_node": decision.hierarchy_node,
        "candidates": list(decision.candidates),
        "n_best_alternatives": [[str(k), float(v)] for k, v in decision.n_best_alternatives],
        "state": decision.state,
        "calibrated_estimate": decision.calibrated_estimate,
        "confidence_interval": [ci[0], ci[1]] if ci is not None else None,
        "calibration_support": dict(decision.calibration_support),
        "evidence_refs": list(decision.evidence_refs),
        "excluded_evidence_refs": list(decision.excluded_evidence_refs),
        "exclusion_reasons": list(decision.exclusion_reasons),
        "contradiction_codes": list(decision.contradiction_codes),
        "open_set_state": decision.open_set_state,
        "decision_reason": decision.decision_reason,
        "policy_hash": decision.policy_hash,
        "config_hash": decision.config_hash,
        "calibration_artifact_hash": decision.calibration_artifact_hash,
    }


def field_decision_from_json(data: Mapping[str, Any]) -> FieldDecision:
    ci = data.get("confidence_interval")
    n_best = data.get("n_best_alternatives") or ()
    return FieldDecision(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        sprite_id=str(data.get("sprite_id", "")),
        field_name=str(data.get("field", "")),
        accepted_value=data.get("accepted_value"),
        accepted_values=tuple(data.get("accepted_values") or ()),
        hierarchy_node=str(data.get("hierarchy_node", "")),
        candidates=tuple(str(v) for v in data.get("candidates") or ()),
        n_best_alternatives=tuple(
            (str(pair[0]), float(pair[1])) for pair in n_best if isinstance(pair, (list, tuple)) and len(pair) == 2
        ),
        state=str(data.get("state", "unlabeled")),
        calibrated_estimate=float(data["calibrated_estimate"]) if data.get("calibrated_estimate") is not None else None,
        confidence_interval=(float(ci[0]), float(ci[1])) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
        calibration_support=dict(data.get("calibration_support") or {}),
        evidence_refs=tuple(str(v) for v in data.get("evidence_refs") or ()),
        excluded_evidence_refs=tuple(str(v) for v in data.get("excluded_evidence_refs") or ()),
        exclusion_reasons=tuple(str(v) for v in data.get("exclusion_reasons") or ()),
        contradiction_codes=tuple(str(v) for v in data.get("contradiction_codes") or ()),
        open_set_state=str(data.get("open_set_state", "unknown")),
        decision_reason=str(data.get("decision_reason", "insufficient_evidence")),
        policy_hash=str(data.get("policy_hash", "")),
        config_hash=str(data.get("config_hash", "")),
        calibration_artifact_hash=str(data.get("calibration_artifact_hash", "")),
    )


# Tag-specific decision: each accepted tag carries its own provenance.
@dataclass(frozen=True)
class TagDecision:
    tag: str
    state: FieldState = "unlabeled"
    evidence_refs: tuple[str, ...] = ()
    calibrated_estimate: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "state": self.state,
            "evidence_refs": list(self.evidence_refs),
            "calibrated_estimate": self.calibrated_estimate,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TagDecision:
        return cls(
            tag=str(data.get("tag", "")),
            state=str(data.get("state", "unlabeled")),
            evidence_refs=tuple(str(v) for v in data.get("evidence_refs") or ()),
            calibrated_estimate=float(data["calibrated_estimate"])
            if data.get("calibrated_estimate") is not None
            else None,
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass(frozen=True)
class AcceptedTagSet:
    """A provenance-aware set of accepted tags."""

    schema_version: str = SCHEMA_VERSION
    decisions: tuple[TagDecision, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_tags(self) -> tuple[str, ...]:
        return tuple(d.tag for d in self.decisions if d.state == "accepted")

    @property
    def all_tags(self) -> tuple[str, ...]:
        return tuple(d.tag for d in self.decisions)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decisions": [d.to_json() for d in self.decisions],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> AcceptedTagSet:
        raw_decisions = data.get("decisions") or ()
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            decisions=tuple(
                TagDecision.from_json(d) if isinstance(d, Mapping) else TagDecision(tag=str(d)) for d in raw_decisions
            ),
            provenance=dict(data.get("provenance") or {}),
        )
