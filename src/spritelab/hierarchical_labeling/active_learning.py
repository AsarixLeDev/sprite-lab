"""Deterministic active-learning review rounds with duplicate controls."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    content_identity,
    require_probability,
    require_text,
)

ACTIVE_LEARNING_ROUND_SCHEMA = "spritelab.labeling.active-learning-round.v1"


@dataclass(frozen=True, eq=False)
class ActiveLearningCandidate(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.active-learning-candidate.v1"
    IDENTITY_FIELDS = ("record_identity", "image_identity", "cluster_identity")

    record_identity: str
    image_identity: str
    cluster_identity: str
    duplicate_cluster_identity: str
    near_duplicate_cluster_identity: str | None
    expected_information_gain: float
    cluster_representativeness: float
    novelty: float
    high_confidence_disagreement: float
    visual_metadata_conflict: bool
    low_calibrated_margin: float
    rare_class: bool
    affected_cluster_size: int
    taxonomy_gap: bool
    source_drift: float
    provider_disagreement: float
    already_reviewed: bool
    legally_eligible: bool
    technically_usable: bool

    def __post_init__(self) -> None:
        for name in ("record_identity", "image_identity", "cluster_identity", "duplicate_cluster_identity"):
            require_text(getattr(self, name), name.replace("_", " "))
        if self.near_duplicate_cluster_identity is not None:
            require_text(self.near_duplicate_cluster_identity, "near duplicate cluster identity")
        for name in (
            "expected_information_gain",
            "cluster_representativeness",
            "novelty",
            "high_confidence_disagreement",
            "low_calibrated_margin",
            "source_drift",
            "provider_disagreement",
        ):
            require_probability(getattr(self, name), name.replace("_", " "))
        if type(self.affected_cluster_size) is not int or self.affected_cluster_size < 1:
            raise ContractValidationError("affected cluster size must be positive")
        self.validate_record()


@dataclass(frozen=True)
class ActiveLearningPolicy:
    review_budget: int = 100
    seed: int = 20260715
    maximum_per_near_duplicate_cluster: int = 2
    minimum_marginal_gain: float = 0.05
    weights: tuple[tuple[str, float], ...] = (
        ("expected_information_gain", 1.0),
        ("cluster_representativeness", 0.7),
        ("novelty", 0.5),
        ("high_confidence_disagreement", 0.9),
        ("visual_metadata_conflict", 0.8),
        ("low_calibrated_margin", 0.85),
        ("rare_class", 0.6),
        ("affected_cluster_size", 0.45),
        ("taxonomy_gap", 0.95),
        ("source_drift", 0.7),
        ("provider_disagreement", 0.75),
    )

    def __post_init__(self) -> None:
        if type(self.review_budget) is not int or self.review_budget < 1:
            raise ContractValidationError("active-learning review budget must be positive")
        if type(self.seed) is not int:
            raise ContractValidationError("active-learning seed must be an integer")
        if type(self.maximum_per_near_duplicate_cluster) is not int or self.maximum_per_near_duplicate_cluster < 1:
            raise ContractValidationError("active-learning near-duplicate limit must be positive")
        require_probability(self.minimum_marginal_gain, "minimum marginal gain")
        names = [name for name, _weight in self.weights]
        if len(names) != len(set(names)) or any(
            not math.isfinite(weight) or weight < 0 for _name, weight in self.weights
        ):
            raise ContractValidationError("active-learning weights must be unique finite non-negative values")

    @property
    def identity(self) -> str:
        return content_identity(
            "spritelab-active-learning-policy-v1",
            {
                "review_budget": self.review_budget,
                "seed": self.seed,
                "maximum_per_near_duplicate_cluster": self.maximum_per_near_duplicate_cluster,
                "minimum_marginal_gain": self.minimum_marginal_gain,
                "weights": self.weights,
            },
        )


def _candidate_components(candidate: ActiveLearningCandidate, weights: Mapping[str, float]) -> dict[str, float]:
    cluster_scale = min(1.0, math.log1p(candidate.affected_cluster_size) / math.log1p(1000))
    return {
        "expected_information_gain": weights["expected_information_gain"] * candidate.expected_information_gain,
        "cluster_representativeness": weights["cluster_representativeness"] * candidate.cluster_representativeness,
        "novelty": weights["novelty"] * candidate.novelty,
        "high_confidence_disagreement": weights["high_confidence_disagreement"]
        * candidate.high_confidence_disagreement,
        "visual_metadata_conflict": weights["visual_metadata_conflict"] * float(candidate.visual_metadata_conflict),
        "low_calibrated_margin": weights["low_calibrated_margin"] * candidate.low_calibrated_margin,
        "rare_class": weights["rare_class"] * float(candidate.rare_class),
        "affected_cluster_size": weights["affected_cluster_size"] * cluster_scale,
        "taxonomy_gap": weights["taxonomy_gap"] * float(candidate.taxonomy_gap),
        "source_drift": weights["source_drift"] * candidate.source_drift,
        "provider_disagreement": weights["provider_disagreement"] * candidate.provider_disagreement,
    }


def _tie(record_identity: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{record_identity}".encode()).hexdigest(), 16)


def stopping_reason(
    *,
    coverage_target_reached: bool,
    precision_target_maintained: bool,
    marginal_gain: float,
    reviews_used: int,
    review_budget: int,
    taxonomy_work_required: bool,
    minimum_marginal_gain: float,
) -> str | None:
    if coverage_target_reached:
        return "coverage_target_reached"
    if not precision_target_maintained:
        return "precision_target_cannot_be_maintained"
    if taxonomy_work_required:
        return "new_taxonomy_work_required"
    if reviews_used >= review_budget:
        return "review_budget_exhausted"
    if marginal_gain < minimum_marginal_gain:
        return "marginal_gain_below_threshold"
    return None


def generate_review_round(
    candidates: Sequence[ActiveLearningCandidate],
    *,
    dataset_identity: str,
    reference_set_identity: str,
    embedding_identity: str,
    calibration_identity: str,
    round_number: int,
    policy: ActiveLearningPolicy | None = None,
) -> dict[str, Any]:
    selected_policy = policy or ActiveLearningPolicy()
    for name, value in (
        ("dataset identity", dataset_identity),
        ("reference-set identity", reference_set_identity),
        ("embedding identity", embedding_identity),
        ("calibration identity", calibration_identity),
    ):
        require_text(value, name)
    if type(round_number) is not int or round_number < 1:
        raise ContractValidationError("active-learning round number must be positive")
    identities = [candidate.record_identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise ContractValidationError("active-learning candidate identities cannot repeat")
    eligible = [
        candidate
        for candidate in candidates
        if not candidate.already_reviewed and candidate.legally_eligible and candidate.technically_usable
    ]
    weights = dict(selected_policy.weights)
    scored = []
    for candidate in eligible:
        components = _candidate_components(candidate, weights)
        scored.append(
            (sum(components.values()), _tie(candidate.record_identity, selected_policy.seed), candidate, components)
        )
    scored.sort(key=lambda item: (-item[0], -item[1], item[2].record_identity))
    selected: list[dict[str, Any]] = []
    duplicate_clusters: set[str] = set()
    near_counts: Counter[str] = Counter()
    represented_clusters: set[str] = set()
    for score, _tie_value, candidate, components in scored:
        if len(selected) >= selected_policy.review_budget:
            break
        if candidate.duplicate_cluster_identity in duplicate_clusters:
            continue
        if (
            candidate.near_duplicate_cluster_identity
            and near_counts[candidate.near_duplicate_cluster_identity]
            >= selected_policy.maximum_per_near_duplicate_cluster
        ):
            continue
        diversity_bonus = 0.2 if candidate.cluster_identity not in represented_clusters else 0.0
        selected.append(
            {
                "record_identity": candidate.record_identity,
                "image_identity": candidate.image_identity,
                "cluster_identity": candidate.cluster_identity,
                "priority_score": round(score + diversity_bonus, 8),
                "component_scores": {**components, "new_cluster_bonus": diversity_bonus},
            }
        )
        duplicate_clusters.add(candidate.duplicate_cluster_identity)
        represented_clusters.add(candidate.cluster_identity)
        if candidate.near_duplicate_cluster_identity:
            near_counts[candidate.near_duplicate_cluster_identity] += 1
    payload = {
        "schema_version": ACTIVE_LEARNING_ROUND_SCHEMA,
        "round_number": round_number,
        "dataset_identity": dataset_identity,
        "reference_set_identity": reference_set_identity,
        "embedding_identity": embedding_identity,
        "calibration_identity": calibration_identity,
        "selection_policy_identity": selected_policy.identity,
        "selection_seed": selected_policy.seed,
        "review_budget": selected_policy.review_budget,
        "eligible_count": len(eligible),
        "selected_record_identities": [item["record_identity"] for item in selected],
        "selected": selected,
        "excluded": {
            "already_reviewed": sum(candidate.already_reviewed for candidate in candidates),
            "legally_ineligible": sum(not candidate.legally_eligible for candidate in candidates),
            "technically_unusable": sum(not candidate.technically_usable for candidate in candidates),
            "duplicate_or_near_duplicate": max(0, len(eligible) - len(selected)),
        },
    }
    payload["round_identity"] = content_identity(ACTIVE_LEARNING_ROUND_SCHEMA, payload)
    return payload
