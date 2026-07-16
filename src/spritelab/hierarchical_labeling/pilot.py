"""Deterministic, non-authorizing planner for a representative 5,000-image pilot."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    content_identity,
    require_probability,
    require_text,
)

PILOT_PLAN_SCHEMA = "spritelab.labeling.pilot-5000-plan.v1"


@dataclass(frozen=True, eq=False)
class PilotCandidate(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.pilot-candidate.v1"
    IDENTITY_FIELDS = ("record_identity", "image_identity", "cluster_identity")

    record_identity: str
    image_identity: str
    source_identity: str
    pack_identity: str
    cluster_identity: str
    style_identity: str
    image_size_bucket: str
    sheet_derived: bool
    technical_bucket: str
    duplicate_cluster_identity: str
    semantic_difficulty: float
    legally_eligible: bool
    technically_usable: bool

    def __post_init__(self) -> None:
        for name in (
            "record_identity",
            "image_identity",
            "source_identity",
            "pack_identity",
            "cluster_identity",
            "style_identity",
            "image_size_bucket",
            "technical_bucket",
            "duplicate_cluster_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        require_probability(self.semantic_difficulty, "semantic difficulty")
        self.validate_record()


def plan_pilot(
    candidates: Sequence[PilotCandidate],
    *,
    dataset_identity: str,
    profile: str = "balanced",
    target_size: int = 5000,
    seed: int = 20260715,
    reference_cohort_size: int = 400,
    maximum_hosted_calls: int = 0,
    measured_local_records_per_hour: float | None = None,
) -> dict[str, Any]:
    require_text(dataset_identity, "pilot dataset identity")
    if profile not in {"fast_local", "balanced", "high_quality"}:
        raise ContractValidationError("pilot profile is not controlled")
    if type(target_size) is not int or not 1 <= target_size <= 5000:
        raise ContractValidationError("pilot target size must be from one through 5,000")
    if type(reference_cohort_size) is not int or not 300 <= reference_cohort_size <= 500:
        raise ContractValidationError("pilot human reference cohort must contain 300 to 500 records")
    if type(maximum_hosted_calls) is not int or maximum_hosted_calls < 0:
        raise ContractValidationError("pilot maximum hosted calls must be non-negative")
    if measured_local_records_per_hour is not None and (
        not math.isfinite(measured_local_records_per_hour) or measured_local_records_per_hour <= 0
    ):
        raise ContractValidationError("measured pilot throughput must be finite and positive")
    identities = [candidate.record_identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise ContractValidationError("pilot candidate identities cannot repeat")
    eligible = [candidate for candidate in candidates if candidate.legally_eligible and candidate.technically_usable]
    selected: list[PilotCandidate] = []
    duplicate_clusters: set[str] = set()
    dimensions = ("source_identity", "pack_identity", "cluster_identity", "style_identity", "image_size_bucket")
    counts = {name: Counter() for name in dimensions}
    while len(selected) < min(target_size, len(eligible)):
        choices = [
            candidate for candidate in eligible if candidate.duplicate_cluster_identity not in duplicate_clusters
        ]
        choices = [candidate for candidate in choices if candidate not in selected]
        if not choices:
            break

        def score(candidate: PilotCandidate) -> tuple[float, int]:
            diversity = sum(1.0 / (1.0 + counts[name][getattr(candidate, name)]) for name in dimensions)
            technical = 1.0 / (1.0 + counts.setdefault("technical_bucket", Counter())[candidate.technical_bucket])
            sheet = 0.35 if candidate.sheet_derived else 0.0
            difficulty = 0.6 * candidate.semantic_difficulty
            tie = int(hashlib.sha256(f"{seed}:{candidate.record_identity}".encode()).hexdigest(), 16)
            return diversity + technical + sheet + difficulty, tie

        winner = max(choices, key=score)
        selected.append(winner)
        duplicate_clusters.add(winner.duplicate_cluster_identity)
        for name in dimensions:
            counts[name][getattr(winner, name)] += 1
        counts.setdefault("technical_bucket", Counter())[winner.technical_bucket] += 1
    records = [candidate.record_identity for candidate in selected]
    local_calls_per_record = 2 if profile == "fast_local" else 3
    embedding_dimensions = 64
    embedding_bytes = len(selected) * embedding_dimensions * 4
    render_bytes_estimate = len(selected) * 7 * 32 * 32 * 4
    hours = len(selected) / measured_local_records_per_hour if measured_local_records_per_hour else None
    payload: dict[str, Any] = {
        "schema_version": PILOT_PLAN_SCHEMA,
        "dataset_identity": dataset_identity,
        "selection_seed": seed,
        "requested_records": target_size,
        "selected_records": len(selected),
        "selected_record_identities": records,
        "representativeness": {
            name: len({getattr(candidate, name) for candidate in selected})
            for name in (*dimensions, "technical_bucket", "duplicate_cluster_identity")
        },
        "provider_profile": profile,
        "expected_local_calls": len(selected) * local_calls_per_record,
        "maximum_hosted_calls": maximum_hosted_calls,
        "embedding_work": {"records": len(selected), "dimensions": embedding_dimensions, "device": "cpu"},
        "human_reference_cohort_size": min(reference_cohort_size, len(selected)),
        "active_review_rounds": 2,
        "storage_estimate_bytes": embedding_bytes + render_bytes_estimate,
        "time_estimate_hours": hours if hours is not None else "unknown_without_measured_throughput",
        "stages": [
            "technical preprocessing",
            "initial 300-500 reference cohort",
            "human review",
            "embedding/index build",
            "automatic labeling",
            "calibration",
            "held-out evaluation",
            "error-cluster review",
            "second active-learning round",
            "final precision/coverage report",
        ],
        "production_authorization": False,
        "pilot_runs_automatically": False,
        "human_labels_created": 0,
    }
    payload["plan_identity"] = content_identity(PILOT_PLAN_SCHEMA, payload)
    return payload
