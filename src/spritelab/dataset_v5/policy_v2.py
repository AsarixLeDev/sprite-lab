"""Dataset-v5 policy previews that separate membership, splitting, and sampling.

This module deliberately does not mutate or replace the immutable v5 builder.  It
adapts its provenance-complete records into policy previews whose manifests make
membership, evaluation selection, and training weights independently auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.dataset_v5.builder import (
    PARTITIONS,
    _deduplicate,
    _discover,
    _hash_json,
    _load_explicit,
    _load_v4,
    _normalized_alpha_key,
    _public,
    _stable_key,
    _write_json,
    _write_jsonl,
    distribution_report,
    source_file_sha256,
)

POLICY_BUILDER_VERSION = "dataset_v5_policy_v2.2"
CURRENT_POLICY_SCHEMA_VERSION = "dataset_v5_policy_v2.current.v1"
LEGACY_POLICY_SCHEMA_VERSION = "dataset_v5_policy_v2.legacy.v1"
POLICY_NAMES = ("strict_hard_cap", "soft_balanced", "core_plus_weighted_sampling")
QUALITY_TIERS = frozenset(("strict", "standard", "unreviewed"))
SAFE_QUALITY_MULTIPLIERS = {"strict": 1.0, "standard": 0.8, "unreviewed": 0.0}
LEGACY_QUALITY_MULTIPLIERS = {"strict": 1.0, "standard": 0.8, "unreviewed": 0.2}


@dataclass(frozen=True)
class WeightingPolicy:
    mode: str = "inverse_frequency_weighting"
    temperature: float = 0.5
    soft_target_share: float = 0.15
    minimum_weight: float = 0.05
    maximum_weight: float = 10.0
    pack_exponent: float = 0.5
    artist_exponent: float = 0.5
    source_family_exponent: float = 0.25
    canonical_object_exponent: float = 0.25
    geometry_family_exponent: float = 1.0
    quality_multipliers: dict[str, float] = field(default_factory=lambda: dict(SAFE_QUALITY_MULTIPLIERS))
    quality_multipliers_explicit: bool = False
    diagnostic_only_allow_positive_unreviewed: bool = False
    per_epoch_quota: int | None = None

    def __post_init__(self) -> None:
        tiers = set(self.quality_multipliers)
        missing = sorted(QUALITY_TIERS - tiers)
        unknown = sorted(tiers - QUALITY_TIERS)
        if missing:
            raise ValueError(f"missing required quality multiplier tiers: {missing}")
        if unknown:
            raise ValueError(f"unknown quality multiplier tiers: {unknown}")
        for tier, value in self.quality_multipliers.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"quality multiplier for {tier!r} must be finite")
            if value < 0:
                raise ValueError(f"quality multiplier for {tier!r} must be nonnegative")
        if self.quality_multipliers["unreviewed"] > 0 and not self.diagnostic_only_allow_positive_unreviewed:
            raise ValueError("unreviewed > 0 requires diagnostic_only_allow_positive_unreviewed=true")

    def resolved(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_policy_hash(self) -> str:
        return _hash_json(self.resolved())

    @property
    def promotion_forbidden(self) -> bool:
        return self.diagnostic_only_allow_positive_unreviewed


@dataclass(frozen=True)
class PolicyV2Config:
    policy_schema_version: str = CURRENT_POLICY_SCHEMA_VERSION
    dataset_name: str = "sprite_lab_multisource_v5_policy_v2_preview"
    policy_name: str = "core_plus_weighted_sampling"
    seed: int = 20260711
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    source_ood_packs: tuple[str, ...] = ("oga_potion_dcss",)
    held_out_artists: tuple[str, ...] = ()
    open_set_objects: tuple[str, ...] = ()
    evaluation_pack_cap: float = 0.15
    evaluation_artist_cap: float = 0.15
    strict_membership_pack_cap: float = 0.15
    strict_membership_artist_cap: float = 0.15
    evaluation_max_records: int = 160
    weighting: WeightingPolicy = field(default_factory=WeightingPolicy)
    production_policy: bool = False

    def __post_init__(self) -> None:
        if self.policy_schema_version not in {CURRENT_POLICY_SCHEMA_VERSION, LEGACY_POLICY_SCHEMA_VERSION}:
            raise ValueError(f"unsupported policy_schema_version {self.policy_schema_version!r}")
        if self.production_policy and self.weighting.diagnostic_only_allow_positive_unreviewed:
            raise ValueError("production-policy configs cannot use the diagnostic unreviewed-weight override")

    @classmethod
    def from_json(cls, path: str | Path, *, legacy: bool = False) -> PolicyV2Config:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        weighting_raw = raw.pop("weighting", {})
        if "quality_tier_multipliers" in weighting_raw:
            raise ValueError("quality_tier_multipliers is obsolete; use quality_multipliers")
        explicit = "quality_multipliers" in weighting_raw
        if legacy:
            declared = raw.get("policy_schema_version")
            if declared not in (None, LEGACY_POLICY_SCHEMA_VERSION):
                raise ValueError("explicit legacy loading requires the legacy policy schema")
            raw["policy_schema_version"] = LEGACY_POLICY_SCHEMA_VERSION
            weighting_raw.setdefault("quality_multipliers", dict(LEGACY_QUALITY_MULTIPLIERS))
            weighting_raw.setdefault("diagnostic_only_allow_positive_unreviewed", True)
        else:
            if not explicit:
                raise ValueError("current policy requires explicit weighting.quality_multipliers")
            if raw.get("policy_schema_version") != CURRENT_POLICY_SCHEMA_VERSION:
                raise ValueError(f"current policy requires policy_schema_version={CURRENT_POLICY_SCHEMA_VERSION!r}")
        weighting_raw["quality_multipliers_explicit"] = explicit
        weighting = WeightingPolicy(**weighting_raw)
        for key in ("source_ood_packs", "held_out_artists", "open_set_objects"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(weighting=weighting, **raw)

    def canonical(self) -> dict[str, Any]:
        return asdict(self)

    def validate_for_build(self) -> None:
        if (
            self.policy_schema_version == CURRENT_POLICY_SCHEMA_VERSION
            and not self.weighting.quality_multipliers_explicit
        ):
            raise ValueError("current production/candidate policy must explicitly resolve quality_multipliers")

    def resolved_policy(self) -> dict[str, Any]:
        return {
            "policy_schema_version": self.policy_schema_version,
            "policy_name": self.policy_name,
            "production_policy": self.production_policy,
            "weighting": self.weighting.resolved(),
        }

    def resolved_policy_hash(self) -> str:
        return _hash_json(self.resolved_policy())


def _normal_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def source_sheet_identity(record: dict[str, Any]) -> tuple[str, str]:
    """Return a split key and evidence level without conflating a directory with a sheet."""

    pack = str(record.get("source_pack") or record.get("source_id") or "unknown")
    explicit = str(record.get("actual_sheet_id") or record.get("sheet_id") or "")
    if explicit:
        return f"{pack}:declared:{explicit}", "declared_sheet_id"
    mapping = str(record.get("declarative_sheet_mapping") or "")
    if mapping:
        return f"{pack}:mapping:{mapping}", "declarative_mapping"

    archive = _normal_path(record.get("archive_member"))
    source_image = _normal_path(record.get("source_image"))
    source_hash = str(record.get("source_image_hash") or record.get("downloaded_file_hash") or "")
    coordinates = record.get("cell_coordinates")
    sliced = "/sliced/" in f"/{source_image.lower()}" or bool(coordinates)
    if archive and source_hash and sliced:
        return f"{pack}:archive:{archive}:{source_hash}", "archive_member_and_source_hash"
    if source_hash and coordinates:
        return f"{pack}:hash:{source_hash}", "source_hash_and_cell_coordinates"

    # A standalone image is its own source image, even when hundreds share a directory.
    if source_image:
        return f"{pack}:image:{source_image}", "standalone_source_image"
    if archive:
        return f"{pack}:archive-member:{archive}", "archive_member"

    broad = str(record.get("source_sheet") or f"{pack}:unknown")
    return f"{pack}:fallback:{broad}", "broad_directory_fallback"


def build_policy_groups(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build immutable leakage groups with evidence-aware source-sheet identity."""

    relations: list[dict[str, Any]] = []
    hard_edges: list[tuple[str, str]] = []
    sheet_evidence: Counter[str] = Counter()

    def add(kind: str, members: list[str], key: str, hard: bool = True, evidence: str | None = None) -> None:
        values = sorted(set(members))
        if len(values) < 2:
            return
        relation_id = "rel_" + hashlib.sha256(f"policy-v2:{kind}:{key}".encode()).hexdigest()[:20]
        row: dict[str, Any] = {
            "relation_id": relation_id,
            "kind": kind,
            "key": key,
            "members": values,
            "hard_split_constraint": hard,
        }
        if evidence:
            row["grouping_evidence"] = evidence
        relations.append(row)
        if hard:
            hard_edges.extend((values[0], value) for value in values[1:])

    for field_name, kind, hard in (
        ("exported_rgba_hash", "exact_exported_rgba", True),
        ("alpha_mask_hash", "exact_alpha_mask_recolor", True),
        ("animation_group", "animation_siblings", True),
        ("known_variant_family", "known_variant_family", True),
        ("source_pack", "same_source_pack", False),
        ("sub_artist", "same_artist_or_subartist", False),
    ):
        buckets: dict[str, list[str]] = defaultdict(list)
        for row in records:
            value = str(row.get(field_name) or "")
            if value:
                buckets[value].append(row["sprite_id"])
        for key, members in sorted(buckets.items()):
            add(kind, members, key, hard)

    sheets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in records:
        key, evidence = source_sheet_identity(row)
        row["policy_source_sheet_id"] = key
        row["policy_source_sheet_evidence"] = evidence
        sheet_evidence[evidence] += 1
        sheets[(key, evidence)].append(row["sprite_id"])
    for (key, evidence), members in sorted(sheets.items()):
        add("source_sheet_siblings", members, key, evidence=evidence)

    declared: dict[str, list[str]] = defaultdict(list)
    translated: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in records:
        for value in row.get("declared_variant_ids", []):
            declared[str(value)].append(row["sprite_id"])
        # Translation-normalized silhouettes are only variant evidence inside the
        # same pack and approved object class.  Global silhouette grouping turns
        # unrelated icons with common primitive masks into one giant component.
        translated[
            (str(row.get("source_pack") or ""), str(row.get("object_name") or ""), _normalized_alpha_key(row["_alpha"]))
        ].append(row["sprite_id"])
    for key, members in sorted(declared.items()):
        add("declared_variant_family", members, key)
    for key, members in sorted(translated.items()):
        add("translation_padding_variant", members, ":".join(key))

    parent = {row["sprite_id"]: row["sprite_id"] for row in records}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in hard_edges:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    components: dict[str, list[str]] = defaultdict(list)
    for sprite_id in sorted(parent):
        components[find(sprite_id)].append(sprite_id)
    groups: list[dict[str, Any]] = []
    for members in sorted(components.values(), key=lambda values: values[0]):
        group_id = "split_" + hashlib.sha256("|".join(members).encode()).hexdigest()[:20]
        groups.append({"split_group_id": group_id, "members": members})
    audit = {
        "evidence_record_counts": dict(sorted(sheet_evidence.items())),
        "sheet_group_count": sum(len(value) > 1 for value in sheets.values()),
        "fallback_record_count": sheet_evidence["broad_directory_fallback"],
        "largest_split_group": max((len(row["members"]) for row in groups), default=0),
        "split_group_count": len(groups),
    }
    return sorted(relations, key=lambda row: row["relation_id"]), groups, audit


def geometry_family_map(records: list[dict[str, Any]], relations: list[dict[str, Any]]) -> dict[str, str]:
    """Map each record to one transitive geometry/recolor family."""

    parent = {row["sprite_id"]: row["sprite_id"] for row in records}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for relation in relations:
        if relation.get("kind") not in {
            "exact_alpha_mask_recolor",
            "translation_padding_variant",
            "declared_variant_family",
            "known_variant_family",
        }:
            continue
        members = [value for value in relation["members"] if value in parent]
        for member in members[1:]:
            a, b = find(members[0]), find(member)
            if a != b:
                parent[max(a, b)] = min(a, b)
    families: dict[str, list[str]] = defaultdict(list)
    for sprite_id in sorted(parent):
        families[find(sprite_id)].append(sprite_id)
    return {
        sprite_id: "geometry_" + hashlib.sha256("|".join(members).encode()).hexdigest()[:20]
        for members in families.values()
        for sprite_id in members
    }


def _frequency_factor(count: int, total: int, exponent: float, policy: WeightingPolicy) -> float:
    if exponent <= 0 or count <= 0 or total <= 0:
        return 1.0
    share = count / total
    if policy.mode == "soft_cap":
        return min(1.0, (policy.soft_target_share / share) ** exponent) if share > policy.soft_target_share else 1.0
    power = policy.temperature if policy.mode == "temperature_sampling" else 1.0
    return count ** (-exponent * power)


def compute_sampling_weights(
    records: list[dict[str, Any]],
    families: dict[str, str],
    policy: WeightingPolicy,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute deterministic bounded weights; recolors divide a family's influence."""

    if policy.minimum_weight <= 0 or policy.maximum_weight < policy.minimum_weight:
        raise ValueError("sampling weight bounds must satisfy 0 < minimum <= maximum")
    total = len(records)
    fields: tuple[tuple[str, float, Callable[[dict[str, Any]], str]], ...] = (
        ("pack", policy.pack_exponent, lambda row: str(row.get("source_pack") or "unknown")),
        ("artist", policy.artist_exponent, lambda row: str(row.get("sub_artist") or "unknown")),
        ("source_family", policy.source_family_exponent, lambda row: str(row.get("source_family") or "unknown")),
        (
            "canonical_object",
            policy.canonical_object_exponent,
            lambda row: str(row.get("object_name") or "unknown") if row.get("is_supervised") else "unlabeled",
        ),
    )
    counts = {name: Counter(key(row) for row in records) for name, _exponent, key in fields}
    family_counts = Counter(families[row["sprite_id"]] for row in records)
    raw: dict[str, float] = {}
    factor_rows: dict[str, dict[str, float]] = {}
    for row in sorted(records, key=lambda item: item["sprite_id"]):
        factors: dict[str, float] = {}
        value = 1.0
        for name, exponent, key in fields:
            factor = _frequency_factor(counts[name][key(row)], total, exponent, policy)
            factors[name] = factor
            value *= factor
        family_size = family_counts[families[row["sprite_id"]]]
        geometry = family_size ** (-policy.geometry_family_exponent)
        factors["geometry_family"] = geometry
        value *= geometry
        tier = "strict" if row.get("strict_quality") else "standard" if row.get("is_supervised") else "unreviewed"
        quality = _label_quality_sampling_multiplier(row, policy, fallback_tier=tier)
        factors["quality_tier"] = quality
        raw[row["sprite_id"]] = value * quality
        factor_rows[row["sprite_id"]] = factors

    positive = [value for value in raw.values() if value > 0]
    scale = len(positive) / sum(positive) if positive else 1.0
    weights = {
        sprite_id: min(policy.maximum_weight, max(policy.minimum_weight, value * scale)) if value > 0 else 0.0
        for sprite_id, value in raw.items()
    }
    # Rake pack and artist marginals to the configured soft ceiling. This keeps
    # intersecting factors (for example, a rare pack from a common artist) from
    # undoing one another. It changes sampling probabilities, never membership.
    record_by_id = {row["sprite_id"]: row for row in records}
    for _iteration in range(100):
        changed = False
        for field_name in ("source_pack", "sub_artist"):
            grouped: dict[str, list[str]] = defaultdict(list)
            for sprite_id in weights:
                grouped[str(record_by_id[sprite_id].get(field_name) or "unknown")].append(sprite_id)
            if len(grouped) < 2 or len(grouped) * policy.soft_target_share < 1.0 - 1e-12:
                continue
            total_weight = sum(weights.values())
            sprite_ids = max(grouped.values(), key=lambda values: sum(weights[sprite_id] for sprite_id in values))
            group_weight = sum(weights[sprite_id] for sprite_id in sprite_ids)
            if group_weight / total_weight <= policy.soft_target_share + 1e-9:
                continue
            desired = policy.soft_target_share * (total_weight - group_weight) / (1.0 - policy.soft_target_share)
            factor = desired / group_weight
            for sprite_id in sprite_ids:
                weights[sprite_id] = max(policy.minimum_weight, weights[sprite_id] * factor)
            adjusted_group = sum(weights[sprite_id] for sprite_id in sprite_ids)
            sprite_id_set = set(sprite_ids)
            complement = [sprite_id for sprite_id in weights if sprite_id not in sprite_id_set]
            complement_weight = sum(weights[sprite_id] for sprite_id in complement)
            if complement_weight and adjusted_group / (adjusted_group + complement_weight) > policy.soft_target_share:
                required_complement = adjusted_group * (1.0 - policy.soft_target_share) / policy.soft_target_share
                boost = required_complement / complement_weight
                for sprite_id in complement:
                    weights[sprite_id] = min(policy.maximum_weight, weights[sprite_id] * boost)
            changed = True
        if not changed:
            break
    weights = {
        sprite_id: round(min(policy.maximum_weight, max(policy.minimum_weight, value)), 12) if value > 0 else 0.0
        for sprite_id, value in weights.items()
    }
    effective_total = sum(weights.values())
    squared_total = sum(value * value for value in weights.values())
    kish_effective_sample_size = effective_total**2 / squared_total if squared_total else 0.0

    def effective(field_name: str) -> list[dict[str, Any]]:
        grouped: dict[str, float] = defaultdict(float)
        for row in records:
            grouped[str(row.get(field_name) or "unknown")] += weights.get(row["sprite_id"], 0.0)
        return [
            {
                "value": key,
                "effective_count": round(value, 9),
                "share": value / effective_total if effective_total else 0.0,
            }
            for key, value in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
        ]

    family_influence: dict[str, float] = defaultdict(float)
    for row in records:
        family_influence[families[row["sprite_id"]]] += weights.get(row["sprite_id"], 0.0)
    report = {
        "schema_version": "sampling_weights_v2.1",
        "policy": asdict(policy),
        "record_count": len(weights),
        "minimum": min(weights.values(), default=0.0),
        "maximum": max(weights.values(), default=0.0),
        "mean": sum(weights.values()) / len(weights) if weights else 0.0,
        "effective_sample_size": kish_effective_sample_size,
        "effective_weight_mass": effective_total,
        "effective_pack_distribution": effective("source_pack"),
        "effective_artist_distribution": effective("sub_artist"),
        "geometry_family_influence": [
            {"family": key, "effective_count": round(value, 9), "members": family_counts[key]}
            for key, value in sorted(family_influence.items(), key=lambda item: (-item[1], item[0]))
        ],
        "factor_digest": _hash_json(factor_rows),
        "weight_digest": _hash_json(weights),
    }
    return weights, report


def _label_quality_sampling_multiplier(row: dict[str, Any], policy: WeightingPolicy, *, fallback_tier: str) -> float:
    """Bound sampling contribution by calibrated record risk when available."""

    quality = row.get("label_quality") if isinstance(row.get("label_quality"), dict) else {}
    score = quality.get("record_uncertainty_1_20")
    if score is None:
        return float(policy.quality_multipliers[fallback_tier])
    value = max(1, min(20, int(score)))
    if value <= 4:
        multiplier = 1.0
    elif value <= 8:
        multiplier = 0.75
    elif value <= 12:
        multiplier = 0.35
    elif value <= 16:
        multiplier = 0.15
    else:
        multiplier = 0.05
    # Unresolved contradictions may not receive strong-sampling treatment.
    if int(quality.get("unresolved_conflict_count", 0) or 0) > 0:
        multiplier = min(multiplier, 0.35)
    return multiplier


def _hard_cap_subset(
    records: list[dict[str, Any]], pack_cap: float, artist_cap: float, seed: int, maximum: int | None = None
) -> list[dict[str, Any]]:
    """Construct the largest deterministic evaluation subset satisfying hard caps."""

    maximum = len(records) if maximum is None else min(maximum, len(records))
    pack_frequency = Counter(str(row.get("source_pack") or "unknown") for row in records)
    artist_frequency = Counter(str(row.get("sub_artist") or "unknown") for row in records)
    ordered = sorted(
        records,
        key=lambda row: (
            pack_frequency[str(row.get("source_pack") or "unknown")]
            + artist_frequency[str(row.get("sub_artist") or "unknown")],
            _stable_key(seed, f"evaluation:{row['sprite_id']}"),
        ),
    )
    for target in range(maximum, 0, -1):
        pack_limit = max(1, math.floor(pack_cap * target + 1e-12))
        artist_limit = max(1, math.floor(artist_cap * target + 1e-12))
        pack_counts: Counter[str] = Counter()
        artist_counts: Counter[str] = Counter()
        selected: list[dict[str, Any]] = []
        for row in ordered:
            pack = str(row.get("source_pack") or "unknown")
            artist = str(row.get("sub_artist") or "unknown")
            if pack_counts[pack] >= pack_limit or artist_counts[artist] >= artist_limit:
                continue
            selected.append(row)
            pack_counts[pack] += 1
            artist_counts[artist] += 1
            if len(selected) == target:
                return sorted(selected, key=lambda item: item["sprite_id"])
    return []


def _legacy_hard_cap_subset(
    records: list[dict[str, Any]], pack_cap: float, artist_cap: float, seed: int
) -> list[dict[str, Any]]:
    """Reproduce the destructive simultaneous-cap policy for failure comparison."""

    selected = sorted(records, key=lambda row: _stable_key(seed, f"evaluation:{row['sprite_id']}"))
    changed = True
    while selected and changed:
        changed = False
        for field_name, cap in (("source_pack", pack_cap), ("sub_artist", artist_cap)):
            counts = Counter(str(row.get(field_name) or "unknown") for row in selected)
            dominant, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            if len(counts) > 1 and count / len(selected) > cap + 1e-12:
                victim = max(
                    (row for row in selected if str(row.get(field_name) or "unknown") == dominant),
                    key=lambda row: _stable_key(seed, f"hard-cap:{field_name}:{row['sprite_id']}"),
                )
                selected.remove(victim)
                changed = True
    return sorted(selected, key=lambda row: row["sprite_id"])


def assign_policy_splits(
    records: list[dict[str, Any]], groups: list[dict[str, Any]], config: PolicyV2Config
) -> dict[str, str]:
    """Allocate whole leakage groups against size targets and guarantee regular test coverage."""

    record_by_id = {row["sprite_id"]: row for row in records}
    available_groups = [
        [record_by_id[sprite_id] for sprite_id in group["members"] if sprite_id in record_by_id] for group in groups
    ]
    available_groups = [members for members in available_groups if members]
    assignments: dict[str, str] = {}
    normal: list[list[dict[str, Any]]] = []
    held_packs = set(config.source_ood_packs)
    held_artists = set(config.held_out_artists)
    open_objects = set(config.open_set_objects)
    for members in available_groups:
        if {row["source_pack"] for row in members} & held_packs or {
            row["sub_artist"] for row in members
        } & held_artists:
            partition = "source_ood_test"
        elif open_objects and {row["object_name"] for row in members} <= open_objects:
            partition = "open_set_test"
        else:
            normal.append(members)
            continue
        for row in members:
            assignments[row["sprite_id"]] = partition

    total = sum(len(members) for members in normal)
    targets = {
        "validation": max(1, round(total * config.validation_fraction)),
        "test": max(1, round(total * config.test_fraction)),
    }
    counts = Counter()
    ordered = sorted(
        normal, key=lambda members: _stable_key(config.seed, "|".join(row["sprite_id"] for row in members))
    )
    remaining = list(ordered)
    # Seed regular test with the rarest available artists. This prevents a
    # source-heavy random draw from making evaluation balancing mathematically
    # impossible while still moving only complete leakage groups.
    object_counts = Counter(row["object_name"] for row in records)
    artist_counts = Counter(row["sub_artist"] for row in records if row["sub_artist"] not in held_artists)
    for artist, _count in sorted(artist_counts.items(), key=lambda item: (item[1], item[0])):
        seeded = 0
        while seeded < 2:
            candidates = [
                members
                for members in remaining
                if any(row["sub_artist"] == artist for row in members)
                and all(
                    object_counts[row["object_name"]] > sum(x["object_name"] == row["object_name"] for x in members)
                    for row in members
                )
            ]
            if not candidates:
                break
            candidate = min(
                candidates, key=lambda members: (len(members), _stable_key(config.seed, members[0]["sprite_id"]))
            )
            remaining.remove(candidate)
            for row in candidate:
                assignments[row["sprite_id"]] = "test"
            added = sum(row["sub_artist"] == artist for row in candidate)
            seeded += added
            counts["test"] += len(candidate)

    # Fill the two evaluation partitions, choosing a whole group that best fits each remaining target.
    for partition in ("test", "validation"):
        while remaining and counts[partition] < targets[partition]:
            need = targets[partition] - counts[partition]
            candidate = min(
                remaining,
                key=lambda members: (
                    abs(len(members) - need),
                    _stable_key(config.seed, f"{partition}:{members[0]['sprite_id']}"),
                ),
            )
            remaining.remove(candidate)
            for row in candidate:
                assignments[row["sprite_id"]] = partition
            counts[partition] += len(candidate)
    for members in remaining:
        for row in members:
            assignments[row["sprite_id"]] = "train"

    # Supervised classes in regular eval must be represented in train; move whole groups if necessary.
    train_objects = {row["object_name"] for row in records if assignments[row["sprite_id"]] == "train"}
    for members in normal:
        partition = assignments[members[0]["sprite_id"]]
        if partition in {"validation", "test"} and any(row["object_name"] not in train_objects for row in members):
            for row in members:
                assignments[row["sprite_id"]] = "train"
                train_objects.add(row["object_name"])

    # Repair an emptied regular partition with a train group whose classes remain represented.
    for partition in ("test", "validation"):
        if any(value == partition for value in assignments.values()):
            continue
        train_counts = Counter(row["object_name"] for row in records if assignments[row["sprite_id"]] == "train")
        candidates = [
            members
            for members in normal
            if assignments[members[0]["sprite_id"]] == "train"
            and all(
                train_counts[row["object_name"]] > sum(x["object_name"] == row["object_name"] for x in members)
                for row in members
            )
        ]
        if not candidates:
            raise ValueError(f"unable to construct non-empty {partition} without creating an eval-only class")
        chosen = min(candidates, key=lambda members: (len(members), _stable_key(config.seed, members[0]["sprite_id"])))
        for row in chosen:
            assignments[row["sprite_id"]] = partition
    return assignments


def exclusion_summary(exclusions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand reason occurrences while retaining one primary reason per excluded record."""

    by_record: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in exclusions:
        reasons = row.get("details", {}).get("all_reasons") if isinstance(row.get("details"), dict) else None
        values = (
            [str(value) for value in reasons] if isinstance(reasons, list) and reasons else [str(row["reason_code"])]
        )
        for reason in values:
            item = (str(row.get("stage") or "unknown"), reason)
            if item not in by_record[str(row["sprite_id"])]:
                by_record[str(row["sprite_id"])].append(item)
    occurrences: list[dict[str, Any]] = []
    primary: Counter[str] = Counter()
    secondary: Counter[str] = Counter()
    for sprite_id, reasons in sorted(by_record.items()):
        for index, (stage, reason) in enumerate(reasons):
            occurrences.append(
                {
                    "sprite_id": sprite_id,
                    "stage": stage,
                    "reason_code": reason,
                    "reason_role": "primary" if index == 0 else "secondary",
                }
            )
            (primary if index == 0 else secondary)[reason] += 1
    report = {
        "unique_excluded_records": len(by_record),
        "reason_occurrences": len(occurrences),
        "primary_exclusion_reasons": dict(sorted(primary.items())),
        "secondary_reason_occurrences": dict(sorted(secondary.items())),
    }
    return occurrences, report


def _write_variant(
    output: Path,
    name: str,
    records: list[dict[str, Any]],
    assignments: dict[str, str],
    weights: dict[str, float] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: item["sprite_id"]):
        item = {
            "sprite_id": row["sprite_id"],
            "split": assignments.get(row["sprite_id"]),
            "blob_path": f"../../blobs/{row['exported_rgba_hash']}.rgba",
            "supervision": "supervised" if row.get("is_supervised") else "unlabeled",
        }
        if weights is not None and row["sprite_id"] in weights:
            item["sampling_weight"] = weights[row["sprite_id"]]
        rows.append(item)
    target = output / "variants" / name
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target / "dataset_manifest.jsonl", rows)


def _leakage_report(
    assignments: dict[str, str], relations: list[dict[str, Any]], config: PolicyV2Config, records: list[dict[str, Any]]
) -> dict[str, Any]:
    crossings: Counter[str] = Counter()
    for relation in relations:
        if relation.get("hard_split_constraint"):
            splits = {assignments[member] for member in relation["members"] if member in assignments}
            if len(splits) > 1:
                crossings[str(relation["kind"])] += 1
    held = set(config.source_ood_packs)
    held_leaks = sum(
        (row["source_pack"] in held) != (assignments[row["sprite_id"]] == "source_ood_test")
        for row in records
        if not config.held_out_artists
    )
    return {
        "exact_rgba_crossings": crossings["exact_exported_rgba"],
        "alpha_recolor_crossings": crossings["exact_alpha_mask_recolor"] + crossings["translation_padding_variant"],
        "declared_variant_crossings": crossings["declared_variant_family"] + crossings["known_variant_family"],
        "source_sheet_crossings": crossings["source_sheet_siblings"],
        "held_out_pack_leakage": held_leaks,
        "all_hard_relation_crossings": sum(crossings.values()),
    }


def build_policy_preview(
    *,
    v4_dir: str | Path,
    harvest_root: str | Path,
    output_dir: str | Path,
    config: PolicyV2Config,
    explicit_manifests: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Build a non-production policy preview without changing any frozen input."""

    config.validate_for_build()
    if config.policy_name not in POLICY_NAMES:
        raise ValueError(f"unknown policy {config.policy_name!r}; expected one of {POLICY_NAMES}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preview: {output}")
    output.mkdir(parents=True)
    v4 = Path(v4_dir).resolve()
    harvest = Path(harvest_root).resolve()
    config_hash = _hash_json(config.canonical())
    resolved_policy = config.resolved_policy()
    resolved_policy_hash = config.resolved_policy_hash()
    discovered = _discover(v4, harvest, tuple(Path(path).resolve() for path in explicit_manifests))
    records, exclusions = _load_v4(v4, harvest, config_hash)
    records.extend(_load_explicit(explicit_manifests, config_hash, exclusions))
    deduplicated, duplicates, dedupe_report = _deduplicate(records)
    exclusions.extend(duplicates)

    # Membership is provenance/approval policy, never the balancing policy.
    supervised_core = [
        row
        for row in deduplicated
        if row.get("is_supervised")
        and not (
            row.get("label_provenance", {}).get("adapter") == "label_v3"
            and not row.get("label_provenance", {}).get("record", {}).get("approved")
        )
    ]
    unlabeled_pool = [row for row in deduplicated if row.get("unlabeled_pool") and not row.get("is_supervised")]
    relations, groups, sheet_audit = build_policy_groups(supervised_core)
    families = geometry_family_map(supervised_core, relations)
    core_assignments = assign_policy_splits(supervised_core, groups, config)
    core_train = [row for row in supervised_core if core_assignments[row["sprite_id"]] == "train"]
    core_train_weights, core_weight_report = compute_sampling_weights(core_train, families, config.weighting)
    core_weights = {row["sprite_id"]: core_train_weights.get(row["sprite_id"], 1.0) for row in supervised_core}
    weights, weight_report = core_weights, core_weight_report
    assignments = core_assignments

    membership = supervised_core
    membership_overflow: list[dict[str, Any]] = []
    if config.policy_name == "strict_hard_cap":
        membership = _legacy_hard_cap_subset(
            supervised_core,
            config.strict_membership_pack_cap,
            config.strict_membership_artist_cap,
            config.seed,
        )
        member_ids = {row["sprite_id"] for row in membership}
        membership_overflow = [
            {"sprite_id": row["sprite_id"], "reason_code": "strict_hard_cap_overflow", "stage": "policy_comparison"}
            for row in supervised_core
            if row["sprite_id"] not in member_ids
        ]
        # Reassign after legacy destructive membership so comparison splits remain meaningful.
        assignments = assign_policy_splits(membership, groups, config)
        membership_train = [row for row in membership if assignments[row["sprite_id"]] == "train"]
        train_weights, weight_report = compute_sampling_weights(membership_train, families, config.weighting)
        weights = {row["sprite_id"]: train_weights.get(row["sprite_id"], 1.0) for row in membership}
    exclusions.extend(membership_overflow)

    def train_effective_distribution(field_name: str) -> list[dict[str, Any]]:
        grouped: dict[str, float] = defaultdict(float)
        for row in membership:
            if assignments[row["sprite_id"]] == "train":
                grouped[str(row.get(field_name) or "unknown")] += weights[row["sprite_id"]]
        total_weight = sum(grouped.values())
        return [
            {"value": key, "effective_count": round(value, 9), "share": value / total_weight if total_weight else 0.0}
            for key, value in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
        ]

    weight_report["effective_train_pack_distribution"] = train_effective_distribution("source_pack")
    weight_report["effective_train_artist_distribution"] = train_effective_distribution("sub_artist")
    weight_report["effective_train_epoch_mass"] = round(
        sum(weights[row["sprite_id"]] for row in membership if assignments[row["sprite_id"]] == "train"), 9
    )
    weight_report["resolved_policy"] = resolved_policy
    weight_report["resolved_policy_hash"] = resolved_policy_hash

    evaluation_candidates = [
        row for row in supervised_core if core_assignments.get(row["sprite_id"]) in {"validation", "test"}
    ]
    evaluation_balanced = _hard_cap_subset(
        evaluation_candidates,
        config.evaluation_pack_cap,
        config.evaluation_artist_cap,
        config.seed,
        config.evaluation_max_records,
    )

    blobs = output / "blobs"
    blobs.mkdir()
    for row in deduplicated:
        (blobs / f"{row['exported_rgba_hash']}.rgba").write_bytes(
            np.ascontiguousarray(row["_rgba"], dtype=np.uint8).tobytes()
        )
    dataset_rows: list[dict[str, Any]] = []
    for row in membership:
        public = _public(row)
        public.update(
            split=assignments[row["sprite_id"]],
            blob_path=f"blobs/{row['exported_rgba_hash']}.rgba",
            geometry_family_id=families[row["sprite_id"]],
            sampling_weight=weights[row["sprite_id"]],
            sampling_weight_schema="sampling_weights_v2.1",
            resolved_policy_hash=resolved_policy_hash,
        )
        dataset_rows.append(public)
    _write_jsonl(output / "dataset_manifest.jsonl", sorted(dataset_rows, key=lambda row: row["sprite_id"]))
    _write_jsonl(output / "group_manifest.jsonl", groups + relations)
    occurrences, exclusion_report = exclusion_summary(exclusions)
    _write_jsonl(output / "excluded_manifest.jsonl", occurrences)
    _write_json(output / "exclusion_summary.json", exclusion_report)
    _write_json(
        output / "split_manifest.json",
        {
            "partitions": {
                partition: sorted(sprite_id for sprite_id, value in assignments.items() if value == partition)
                for partition in PARTITIONS
            },
            "seed": config.seed,
            "source_ood_packs": list(config.source_ood_packs),
            "held_out_artists": list(config.held_out_artists),
            "open_set_objects": list(config.open_set_objects),
            "fractions": {
                "train": config.train_fraction,
                "validation": config.validation_fraction,
                "test": config.test_fraction,
            },
            "resolved_policy": resolved_policy,
            "resolved_policy_hash": resolved_policy_hash,
        },
    )
    _write_variant(output, "supervised_core", supervised_core, core_assignments)
    _write_variant(output, "train_balanced", supervised_core, core_assignments, core_weights)
    _write_variant(output, "evaluation_balanced", evaluation_balanced, core_assignments)
    _write_variant(
        output, "strict_quality", [row for row in supervised_core if row.get("strict_quality")], core_assignments
    )
    _write_variant(output, "unlabeled_pool", unlabeled_pool, core_assignments)

    # Reuse the established deterministic NPZ adapter through a temporary legacy export, then retain only loader artifacts.
    from spritelab.dataset_v5.builder import _write_loader_adapter

    _write_loader_adapter(output, membership, assignments)
    training_rows = [
        json.loads(line)
        for line in (output / "training_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in training_rows:
        row["sampling_weight"] = weights[row["sprite_id"]]
        row["sampling_weight_schema"] = "sampling_weights_v2.1"
        row["resolved_policy_hash"] = resolved_policy_hash
    _write_jsonl(output / "training_manifest.jsonl", training_rows)

    leakage = _leakage_report(assignments, relations, config, membership)
    summary = {
        "dataset_name": config.dataset_name,
        "preview": True,
        "production_frozen": False,
        "builder_version": POLICY_BUILDER_VERSION,
        "policy_name": config.policy_name,
        "config_hash": config_hash,
        "resolved_policy": resolved_policy,
        "resolved_policy_hash": resolved_policy_hash,
        "input_records": len(records),
        "deduplicated_records": len(deduplicated),
        "total_membership": len(membership),
        "supervised_core_count": len(supervised_core),
        "unlabeled_pool_count": len(unlabeled_pool),
        "strict_quality_count": sum(bool(row.get("strict_quality")) for row in supervised_core),
        "evaluation_balanced_count": len(evaluation_balanced),
        "split_counts": dict(sorted(Counter(assignments.values()).items())),
        "distribution": distribution_report(membership),
        "effective_sampling": weight_report,
        "sheet_grouping": sheet_audit,
        "deduplication": dedupe_report,
        "exclusions": exclusion_report,
        "overflow_count": len(membership_overflow),
        "leakage": leakage,
        "source_manifest_hashes": discovered["source_manifest_hashes"],
    }
    _write_json(output / "dataset_summary.json", summary)
    _write_json(output / "weighting_report.json", weight_report)
    _write_json(output / "sheet_grouping_audit.json", sheet_audit)
    _write_json(output / "leakage_report.json", leakage)
    _write_json(output / "policy_config.json", config.canonical())
    hashes = {
        path.name: source_file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "PREVIEW.json"
    }
    _write_json(
        output / "PREVIEW.json",
        {
            "production_frozen": False,
            "promotion_forbidden": True,
            "builder_version": POLICY_BUILDER_VERSION,
            "config_hash": config_hash,
            "resolved_policy": resolved_policy,
            "resolved_policy_hash": resolved_policy_hash,
            "artifact_hashes": hashes,
        },
    )
    (output / "README.md").write_text(
        f"# {config.dataset_name}\n\nPolicy preview `{config.policy_name}`. This is not a frozen or promoted production v5.\n",
        encoding="utf-8",
        newline="\n",
    )
    verification = verify_policy_preview(output)
    if not verification["ok"]:
        raise ValueError(f"policy preview verification failed: {verification['errors']}")
    return {"ok": True, "summary": summary, "verification": verification}


def verify_policy_preview(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    errors: list[str] = []
    rows = [
        json.loads(line)
        for line in (output / "dataset_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
    if summary.get("production_frozen") is not False or (output / "FREEZE.json").exists():
        errors.append("policy preview must not be production-frozen")
    if not any(row.get("split") == "test" for row in rows):
        errors.append("regular test partition is empty")
    train_objects = {str(row.get("object_name") or "") for row in rows if row.get("split") == "train"}
    regular_eval_objects = {
        str(row.get("object_name") or "") for row in rows if row.get("split") in {"validation", "test"}
    }
    missing_train_objects = sorted(regular_eval_objects - train_objects)
    if missing_train_objects:
        errors.append(f"regular evaluation-only supervised classes: {missing_train_objects}")
    config = json.loads((output / "policy_config.json").read_text(encoding="utf-8"))
    open_objects = set(config.get("open_set_objects", []))
    invalid_open = sorted(
        row["sprite_id"]
        for row in rows
        if row.get("split") == "open_set_test" and row.get("object_name") not in open_objects
    )
    if invalid_open:
        errors.append(f"undeclared open-set records: {invalid_open}")
    for row in rows:
        weight = float(row.get("sampling_weight", 0.0))
        policy = summary["effective_sampling"]["policy"]
        if weight != 0.0 and not policy["minimum_weight"] <= weight <= policy["maximum_weight"]:
            errors.append(f"out-of-bounds sampling weight for {row['sprite_id']}")
        if row.get("label_provenance", {}).get("adapter") == "label_v3" and not row.get("label_provenance", {}).get(
            "record", {}
        ).get("approved"):
            errors.append(f"unreviewed label-v3 supervision for {row['sprite_id']}")
    leakage = summary.get("leakage", {})
    if any(int(value) for value in leakage.values()):
        errors.append("leakage detected")
    from spritelab.dataset_v5.builder import _verify_loader_contract

    loader = _verify_loader_contract(output)
    errors.extend(loader["errors"])
    return {"ok": not errors, "errors": errors, "leakage": leakage, "training_loader_contract": loader}


def compare_policy_previews(paths: list[str | Path]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        summary = json.loads((path / "dataset_summary.json").read_text(encoding="utf-8"))
        comparisons.append(
            {
                "policy": summary["policy_name"],
                "path": str(path),
                "total_membership": summary["total_membership"],
                "split_counts": summary["split_counts"],
                "pack_distribution": summary["distribution"]["pack"],
                "artist_distribution": summary["distribution"]["artist"],
                "effective_pack_distribution": summary["effective_sampling"]["effective_pack_distribution"],
                "effective_artist_distribution": summary["effective_sampling"]["effective_artist_distribution"],
                "class_coverage": len(summary["distribution"]["object"]),
                "geometry_family_influence": summary["effective_sampling"]["geometry_family_influence"],
                "overflow_count": summary["overflow_count"],
                "unique_excluded_records": summary["exclusions"]["unique_excluded_records"],
                "reason_occurrences": summary["exclusions"]["reason_occurrences"],
                "leakage": summary["leakage"],
            }
        )
    return {"ok": True, "recommended_policy": "core_plus_weighted_sampling", "previews": comparisons}
