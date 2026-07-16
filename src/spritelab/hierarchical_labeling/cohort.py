"""Deterministic, diversity-aware human reference cohort selection."""

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
    require_sha256,
    require_text,
)

COHORT_SCHEMA_VERSION = "spritelab.labeling.reference-cohort-manifest.v2"


@dataclass(frozen=True, eq=False)
class CohortMembership(StrictRecord):
    """Verified membership in one immutable, leakage-safe cohort partition."""

    SCHEMA_VERSION = "spritelab.labeling.cohort-membership.v1"
    IDENTITY_FIELDS = ("cohort_identity", "record_identity", "image_identity", "partition")

    cohort_identity: str
    record_identity: str
    image_identity: str
    partition: str
    source_identity: str
    cluster_identity: str
    duplicate_cluster_identity: str
    near_duplicate_cluster_identity: str | None
    leakage_group_identity: str

    def __post_init__(self) -> None:
        for name in (
            "record_identity",
            "image_identity",
            "source_identity",
            "cluster_identity",
            "duplicate_cluster_identity",
            "leakage_group_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        require_sha256(self.cohort_identity, "cohort identity")
        require_sha256(self.leakage_group_identity, "leakage group identity")
        if self.partition not in {"reference", "calibration", "holdout"}:
            raise ContractValidationError("cohort membership partition is not controlled")
        if self.near_duplicate_cluster_identity is not None:
            require_text(self.near_duplicate_cluster_identity, "near duplicate cluster identity")
        self.validate_record()


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("cohort_identity", None)
    return payload


def _validated_membership_rows(manifest: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    if manifest.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise ContractValidationError("cohort manifest schema version is invalid")
    cohort_identity = manifest.get("cohort_identity")
    require_text(cohort_identity, "cohort identity")
    if cohort_identity != content_identity(COHORT_SCHEMA_VERSION, _manifest_payload(manifest)):
        raise ContractValidationError("cohort manifest identity does not match its content")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != {"reference", "calibration", "holdout"}:
        raise ContractValidationError("cohort manifest partitions are not exact")
    rows: dict[str, tuple[str, Mapping[str, Any]]] = {}
    group_partitions: dict[tuple[str, str], str] = {}
    group_fields = (
        "image_identity",
        "source_identity",
        "cluster_identity",
        "duplicate_cluster_identity",
        "near_duplicate_cluster_identity",
        "leakage_group_identity",
    )
    for partition in ("reference", "calibration", "holdout"):
        values = partitions[partition]
        if not isinstance(values, list):
            raise ContractValidationError("cohort manifest partition rows must be arrays")
        for row in values:
            if not isinstance(row, Mapping):
                raise ContractValidationError("cohort manifest rows must be objects")
            required = {
                "record_identity",
                "image_identity",
                "cluster_identity",
                "duplicate_cluster_identity",
                "near_duplicate_cluster_identity",
                "source_identity",
                "style_identity",
                "size_bucket",
                "leakage_group_identity",
                "selection_factors",
            }
            if set(row) != required:
                raise ContractValidationError("cohort manifest row does not match the exact schema")
            record_identity = require_text(row["record_identity"], "cohort record identity")
            if record_identity in rows:
                raise ContractValidationError("cohort record identities cannot repeat across partitions")
            rows[record_identity] = (partition, row)
            for field in group_fields:
                value = row[field]
                if value is None and field == "near_duplicate_cluster_identity":
                    continue
                require_text(value, field.replace("_", " "))
                key = (field, value)
                previous = group_partitions.setdefault(key, partition)
                if previous != partition:
                    raise ContractValidationError(
                        f"cohort {field.replace('_', ' ')} leaks across {previous} and {partition} partitions"
                    )
    if manifest.get("selected_size") != len(rows):
        raise ContractValidationError("cohort selected size does not match membership rows")
    expected_disjointness = {
        "record_identity": True,
        "image_identity": True,
        "source_identity": True,
        "cluster_identity": True,
        "duplicate_cluster_identity": True,
        "near_duplicate_cluster_identity": True,
        "leakage_group_identity": True,
    }
    if manifest.get("partition_group_disjointness") != expected_disjointness:
        raise ContractValidationError("cohort manifest disjointness evidence is incomplete")
    if manifest.get("partition_identities_disjoint") is not True:
        raise ContractValidationError("cohort manifest must attest partition identity disjointness")
    return rows


def cohort_membership(manifest: Mapping[str, Any], record_identity: str) -> CohortMembership:
    """Verify the full manifest before projecting a record membership."""

    rows = _validated_membership_rows(manifest)
    try:
        partition, row = rows[record_identity]
    except KeyError as exc:
        raise ContractValidationError("record is not a member of the verified cohort") from exc
    return CohortMembership(
        str(manifest["cohort_identity"]),
        record_identity,
        str(row["image_identity"]),
        partition,
        str(row["source_identity"]),
        str(row["cluster_identity"]),
        str(row["duplicate_cluster_identity"]),
        None if row["near_duplicate_cluster_identity"] is None else str(row["near_duplicate_cluster_identity"]),
        str(row["leakage_group_identity"]),
    )


@dataclass(frozen=True, eq=False)
class CohortCandidate(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.cohort-candidate.v1"
    IDENTITY_FIELDS = ("record_identity", "image_identity", "cluster_identity")

    record_identity: str
    image_identity: str
    cluster_identity: str
    duplicate_cluster_identity: str
    near_duplicate_cluster_identity: str | None
    source_identity: str
    style_identity: str
    size_bucket: str
    cluster_size: int
    is_cluster_medoid: bool
    novelty: float
    visual_uncertainty: float
    metadata_conflict: bool
    taxonomy_confusion: float
    sheet_derived: bool
    animation_frame: bool
    rare_category_candidate: bool
    legally_eligible: bool
    technically_usable: bool

    def __post_init__(self) -> None:
        for name in (
            "record_identity",
            "image_identity",
            "cluster_identity",
            "duplicate_cluster_identity",
            "source_identity",
            "style_identity",
            "size_bucket",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        if self.near_duplicate_cluster_identity is not None:
            require_text(self.near_duplicate_cluster_identity, "near duplicate cluster identity")
        if type(self.cluster_size) is not int or self.cluster_size < 1:
            raise ContractValidationError("cohort candidate cluster size must be positive")
        for name in ("novelty", "visual_uncertainty", "taxonomy_confusion"):
            require_probability(getattr(self, name), name)
        self.validate_record()


@dataclass(frozen=True)
class CohortSelectionPolicy:
    target_size: int = 400
    seed: int = 20260715
    reference_fraction: float = 0.6
    calibration_fraction: float = 0.2
    holdout_fraction: float = 0.2
    maximum_per_near_duplicate_cluster: int = 2
    weights: tuple[tuple[str, float], ...] = (
        ("cluster_medoid", 1.0),
        ("large_cluster", 0.55),
        ("rare_cluster", 0.7),
        ("novelty", 0.85),
        ("visual_uncertainty", 0.9),
        ("metadata_conflict", 0.8),
        ("taxonomy_confusion", 0.75),
        ("source_diversity", 0.6),
        ("style_diversity", 0.5),
        ("size_diversity", 0.35),
        ("sheet_derived", 0.3),
        ("animation_frame", 0.3),
        ("rare_category", 0.5),
    )

    def __post_init__(self) -> None:
        if type(self.target_size) is not int or not 1 <= self.target_size <= 50_000:
            raise ContractValidationError("reference cohort target size must be from 1 through 50,000")
        if type(self.seed) is not int:
            raise ContractValidationError("cohort seed must be an integer")
        fractions = (self.reference_fraction, self.calibration_fraction, self.holdout_fraction)
        if any(not math.isfinite(value) or value < 0 for value in fractions) or not math.isclose(sum(fractions), 1.0):
            raise ContractValidationError("cohort partition fractions must be finite, non-negative, and sum to 1")
        names = [name for name, _weight in self.weights]
        if len(names) != len(set(names)):
            raise ContractValidationError("cohort selection weights cannot repeat")
        if any(not math.isfinite(weight) or weight < 0 for _name, weight in self.weights):
            raise ContractValidationError("cohort selection weights must be finite and non-negative")
        if type(self.maximum_per_near_duplicate_cluster) is not int or self.maximum_per_near_duplicate_cluster < 1:
            raise ContractValidationError("near-duplicate cohort limit must be positive")

    @property
    def identity(self) -> str:
        return content_identity(
            "spritelab-reference-cohort-selection-policy-v1",
            {
                "target_size": self.target_size,
                "seed": self.seed,
                "reference_fraction": self.reference_fraction,
                "calibration_fraction": self.calibration_fraction,
                "holdout_fraction": self.holdout_fraction,
                "maximum_per_near_duplicate_cluster": self.maximum_per_near_duplicate_cluster,
                "weights": self.weights,
            },
        )


def _tie_breaker(record_identity: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{record_identity}".encode()).hexdigest(), 16)


def _base_score(candidate: CohortCandidate, weights: Mapping[str, float], maximum_cluster: int) -> float:
    return (
        weights["cluster_medoid"] * float(candidate.is_cluster_medoid)
        + weights["large_cluster"] * math.log1p(candidate.cluster_size) / math.log1p(maximum_cluster)
        + weights["rare_cluster"] / math.sqrt(candidate.cluster_size)
        + weights["novelty"] * candidate.novelty
        + weights["visual_uncertainty"] * candidate.visual_uncertainty
        + weights["metadata_conflict"] * float(candidate.metadata_conflict)
        + weights["taxonomy_confusion"] * candidate.taxonomy_confusion
        + weights["sheet_derived"] * float(candidate.sheet_derived)
        + weights["animation_frame"] * float(candidate.animation_frame)
        + weights["rare_category"] * float(candidate.rare_category_candidate)
    )


def _leakage_components(candidates: Sequence[CohortCandidate], seed: int) -> tuple[tuple[CohortCandidate, ...], ...]:
    """Return connected components for every hard leakage relationship."""

    parent = {candidate.record_identity: candidate.record_identity for candidate in candidates}

    def find(record_identity: str) -> str:
        while parent[record_identity] != record_identity:
            parent[record_identity] = parent[parent[record_identity]]
            record_identity = parent[record_identity]
        return record_identity

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_by_key: dict[tuple[str, str], str] = {}
    for candidate in candidates:
        values = (
            ("image", candidate.image_identity),
            ("source", candidate.source_identity),
            ("visual_cluster", candidate.cluster_identity),
            ("duplicate", candidate.duplicate_cluster_identity),
        )
        if candidate.near_duplicate_cluster_identity is not None:
            values = (*values, ("near_duplicate", candidate.near_duplicate_cluster_identity))
        for key in values:
            previous = first_by_key.setdefault(key, candidate.record_identity)
            union(previous, candidate.record_identity)
    grouped: dict[str, list[CohortCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(find(candidate.record_identity), []).append(candidate)
    return tuple(
        tuple(sorted(values, key=lambda item: item.record_identity))
        for _root, values in sorted(
            grouped.items(),
            key=lambda item: (
                -len(item[1]),
                min(_tie_breaker(value.record_identity, seed) for value in item[1]),
            ),
        )
    )


def _partition_components(
    components: Sequence[tuple[CohortCandidate, ...]], policy: CohortSelectionPolicy
) -> dict[str, list[tuple[CohortCandidate, ...]]]:
    names = ("reference", "calibration", "holdout")
    fractions = {
        "reference": policy.reference_fraction,
        "calibration": policy.calibration_fraction,
        "holdout": policy.holdout_fraction,
    }
    required = [name for name in names if fractions[name] > 0]
    if components and len(components) < len(required):
        raise ContractValidationError(
            "cohort leakage components cannot populate every configured partition without leakage"
        )
    total = sum(len(component) for component in components)
    target = {name: total * fractions[name] for name in names}
    assigned: dict[str, list[tuple[CohortCandidate, ...]]] = {name: [] for name in names}
    counts = dict.fromkeys(names, 0)
    for index, component in enumerate(components):
        remaining = len(components) - index
        empty_required = [name for name in required if not assigned[name]]
        choices = empty_required if remaining == len(empty_required) else required
        winner = max(
            choices,
            key=lambda name: (
                target[name] - counts[name],
                fractions[name],
                -names.index(name),
            ),
        )
        assigned[winner].append(component)
        counts[winner] += len(component)
    return assigned


def select_reference_cohort(
    candidates: Sequence[CohortCandidate],
    *,
    dataset_identity: str,
    embedding_identity: str,
    clustering_identity: str,
    policy: CohortSelectionPolicy | None = None,
) -> dict[str, Any]:
    """Select identities only; this function never creates a human label."""

    selected_policy = policy or CohortSelectionPolicy()
    for name, value in (
        ("dataset identity", dataset_identity),
        ("embedding identity", embedding_identity),
        ("clustering identity", clustering_identity),
    ):
        require_text(value, name)
    identities = [candidate.record_identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise ContractValidationError("cohort candidate record identities cannot repeat")
    eligible = [candidate for candidate in candidates if candidate.legally_eligible and candidate.technically_usable]
    maximum_cluster = max((candidate.cluster_size for candidate in eligible), default=1)
    weights = dict(selected_policy.weights)
    target = min(selected_policy.target_size, len(eligible))
    chosen: list[CohortCandidate] = []
    chosen_records: set[str] = set()
    duplicate_clusters: set[str] = set()
    near_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    while len(chosen) < target:
        scored: list[tuple[float, int, CohortCandidate]] = []
        for candidate in eligible:
            if (
                candidate.record_identity in chosen_records
                or candidate.duplicate_cluster_identity in duplicate_clusters
            ):
                continue
            if (
                candidate.near_duplicate_cluster_identity
                and near_counts[candidate.near_duplicate_cluster_identity]
                >= selected_policy.maximum_per_near_duplicate_cluster
            ):
                continue
            score = _base_score(candidate, weights, maximum_cluster)
            score += weights["source_diversity"] / (1.0 + source_counts[candidate.source_identity])
            score += weights["style_diversity"] / (1.0 + style_counts[candidate.style_identity])
            score += weights["size_diversity"] / (1.0 + size_counts[candidate.size_bucket])
            score += 0.4 / (1.0 + cluster_counts[candidate.cluster_identity])
            scored.append((score, _tie_breaker(candidate.record_identity, selected_policy.seed), candidate))
        if not scored:
            break
        _score, _tie, winner = max(scored, key=lambda item: (item[0], item[1]))
        chosen.append(winner)
        chosen_records.add(winner.record_identity)
        duplicate_clusters.add(winner.duplicate_cluster_identity)
        if winner.near_duplicate_cluster_identity:
            near_counts[winner.near_duplicate_cluster_identity] += 1
        source_counts[winner.source_identity] += 1
        style_counts[winner.style_identity] += 1
        size_counts[winner.size_bucket] += 1
        cluster_counts[winner.cluster_identity] += 1
    ordered = sorted(
        chosen, key=lambda item: (_tie_breaker(item.record_identity, selected_policy.seed), item.record_identity)
    )
    count = len(ordered)
    components = _leakage_components(ordered, selected_policy.seed)
    grouped_partitions = _partition_components(components, selected_policy)
    partitions = {
        name: [candidate for component in values for candidate in component]
        for name, values in grouped_partitions.items()
    }
    group_identity_by_record: dict[str, str] = {}
    for component in components:
        group_identity = content_identity(
            "spritelab.labeling.cohort-leakage-component.v1",
            [candidate.record_identity for candidate in component],
        )
        for candidate in component:
            group_identity_by_record[candidate.record_identity] = group_identity

    def row(candidate: CohortCandidate) -> dict[str, Any]:
        return {
            "record_identity": candidate.record_identity,
            "image_identity": candidate.image_identity,
            "cluster_identity": candidate.cluster_identity,
            "duplicate_cluster_identity": candidate.duplicate_cluster_identity,
            "near_duplicate_cluster_identity": candidate.near_duplicate_cluster_identity,
            "source_identity": candidate.source_identity,
            "style_identity": candidate.style_identity,
            "size_bucket": candidate.size_bucket,
            "leakage_group_identity": group_identity_by_record[candidate.record_identity],
            "selection_factors": {
                "cluster_medoid": candidate.is_cluster_medoid,
                "cluster_size": candidate.cluster_size,
                "novelty": candidate.novelty,
                "visual_uncertainty": candidate.visual_uncertainty,
                "metadata_conflict": candidate.metadata_conflict,
                "taxonomy_confusion": candidate.taxonomy_confusion,
                "sheet_derived": candidate.sheet_derived,
                "animation_frame": candidate.animation_frame,
                "rare_category_candidate": candidate.rare_category_candidate,
            },
        }

    partition_rows = {name: [row(candidate) for candidate in values] for name, values in partitions.items()}

    def is_disjoint(field: str) -> bool:
        owner: dict[str, str] = {}
        for partition, values in partition_rows.items():
            for value in values:
                identity = value[field]
                if identity is None:
                    continue
                previous = owner.setdefault(str(identity), partition)
                if previous != partition:
                    return False
        return True

    disjointness = {
        field: is_disjoint(field)
        for field in (
            "record_identity",
            "image_identity",
            "source_identity",
            "cluster_identity",
            "duplicate_cluster_identity",
            "near_duplicate_cluster_identity",
            "leakage_group_identity",
        )
    }

    payload = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "dataset_identity": dataset_identity,
        "embedding_identity": embedding_identity,
        "clustering_identity": clustering_identity,
        "selection_policy_identity": selected_policy.identity,
        "selection_seed": selected_policy.seed,
        "requested_size": selected_policy.target_size,
        "selected_size": count,
        "eligible_candidate_count": len(eligible),
        "excluded_exact_or_near_duplicates": max(0, len(eligible) - count),
        "partitions": partition_rows,
        "partition_group_disjointness": disjointness,
        "partition_identities_disjoint": all(disjointness.values()),
        "human_labels_created": 0,
    }
    payload["cohort_identity"] = content_identity(COHORT_SCHEMA_VERSION, payload)
    _validated_membership_rows(payload)
    return payload
