"""Field-safe label propagation policies for Labeling v4."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

FIELD_PROPAGATION_POLICY_VERSION = "field_propagation_v4.1"

EXACT_RGBA_DUPLICATE = "exact_rgba_duplicate"
RECOLOR_VARIANT = "recolor_variant"
GEOMETRY_FAMILY = "geometry_family"
MATERIAL_VARIANT = "material_variant"

_RELATION_ALIASES = {
    "exact_duplicate": EXACT_RGBA_DUPLICATE,
    "exact_rgba": EXACT_RGBA_DUPLICATE,
    "recolor": RECOLOR_VARIANT,
    "alpha_mask_recolor": RECOLOR_VARIANT,
    "geometry": GEOMETRY_FAMILY,
    "material_recolor": MATERIAL_VARIANT,
}

IDENTITY_FIELDS = frozenset({"canonical_object", "category", "domain", "role"})
SHAPE_FIELDS = frozenset({"shape", "silhouette", "aspect", "orientation", "structure", "edge_profile", "parts"})
COLOR_FIELDS = frozenset(
    {
        "color",
        "palette_colors",
        "primary_color",
        "primary_colors",
        "secondary_colors",
        "outline_color",
        "outline_colors",
        "shadow_colors",
        "highlight_colors",
        "filename_color_hints",
    }
)
MATERIAL_FIELDS = frozenset({"material", "explicit_material", "visual_material_cue", "material_visual_cues"})
DESCRIPTION_FIELDS = frozenset({"description", "canonical_description", "enriched_description"})
SEMANTIC_FIELDS = frozenset(
    set(IDENTITY_FIELDS)
    | set(SHAPE_FIELDS)
    | set(COLOR_FIELDS)
    | set(MATERIAL_FIELDS)
    | {"surface_alias", "size_hint", "condition", "style", "style_modifiers"}
    | set(DESCRIPTION_FIELDS)
)

_COLOR_WORDS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "cyan",
        "gray",
        "grey",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "teal",
        "violet",
        "white",
        "yellow",
        "golden",
        "silver",
        "dark",
        "light",
    }
)
_MATERIAL_WORDS = frozenset(
    {
        "bronze",
        "chainmail",
        "cloth",
        "copper",
        "crystal",
        "fabric",
        "glass",
        "gold",
        "iron",
        "leather",
        "metal",
        "platemail",
        "silver",
        "steel",
        "stone",
        "wood",
        "wooden",
    }
)


@dataclass(frozen=True)
class PropagationPolicy:
    version: str = FIELD_PROPAGATION_POLICY_VERSION
    recolor_risk_penalty: int = 2
    geometry_risk_penalty: int = 3
    material_variant_risk_penalty: int = 3
    conservative_uncalibrated_score: int = 17

    def __post_init__(self) -> None:
        for name in ("recolor_risk_penalty", "geometry_risk_penalty", "material_variant_risk_penalty"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 1 <= int(self.conservative_uncalibrated_score) <= 20:
            raise ValueError("conservative_uncalibrated_score must be in [1, 20]")


DEFAULT_PROPAGATION_POLICY = PropagationPolicy()


@dataclass(frozen=True)
class PropagationResult:
    fields: Mapping[str, Any]
    field_quality: Mapping[str, Mapping[str, Any]]
    propagated_fields: tuple[str, ...]
    blocked_fields: Mapping[str, str]
    propagated_from: str
    propagation_relation: str
    field_propagation_policy_version: str
    propagation_risk_penalty: int
    description_regenerated: bool
    audit: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": copy.deepcopy(dict(self.fields)),
            "field_quality": copy.deepcopy(dict(self.field_quality)),
            "propagated_fields": list(self.propagated_fields),
            "blocked_fields": dict(self.blocked_fields),
            "propagated_from": self.propagated_from,
            "propagation_relation": self.propagation_relation,
            "field_propagation_policy_version": self.field_propagation_policy_version,
            "propagation_risk_penalty": self.propagation_risk_penalty,
            "description_regenerated": self.description_regenerated,
            "audit": copy.deepcopy(dict(self.audit)),
        }


def normalize_relation(relation: str) -> str:
    value = str(relation).strip().lower()
    value = _RELATION_ALIASES.get(value, value)
    if value not in {EXACT_RGBA_DUPLICATE, RECOLOR_VARIANT, GEOMETRY_FAMILY, MATERIAL_VARIANT}:
        raise ValueError(f"unsupported propagation relation: {relation}")
    return value


def _tokens(value: Any) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(value or "").lower()) if token}


def is_color_neutral_alias(value: Any) -> bool:
    return not bool(_tokens(value) & _COLOR_WORDS)


def is_material_neutral_alias(value: Any) -> bool:
    return not bool(_tokens(value) & _MATERIAL_WORDS)


def _value(entry: Any) -> Any:
    return entry.get("value") if isinstance(entry, Mapping) and "value" in entry else entry


def _is_eligible_source_entry(entry: Any) -> bool:
    """Scalars are final facts; structured entries need an accepted/reviewed marker."""

    if not isinstance(entry, Mapping):
        return entry is not None
    state = str(entry.get("review_state") or entry.get("state") or "").lower()
    return bool(entry.get("human_reviewed")) or state in {"accepted", "reviewed", "strong", "usable_weak"}


def _allowed_fields(
    relation: str,
    source_fields: Mapping[str, Any],
    *,
    declared_relationship: bool,
    strong_source_evidence: bool,
    calibrated_family_policy: bool,
) -> tuple[set[str], dict[str, str]]:
    eligible = {
        name for name, value in source_fields.items() if name in SEMANTIC_FIELDS and _is_eligible_source_entry(value)
    }
    blocked: dict[str, str] = {}
    if relation == EXACT_RGBA_DUPLICATE:
        allowed = set(eligible)
    elif relation == RECOLOR_VARIANT:
        allowed = eligible & (set(IDENTITY_FIELDS) | set(SHAPE_FIELDS) | {"surface_alias"})
    elif relation == MATERIAL_VARIANT:
        allowed = eligible & (set(IDENTITY_FIELDS) | set(SHAPE_FIELDS) | {"surface_alias"})
    else:
        authorized = declared_relationship or strong_source_evidence or calibrated_family_policy
        if not authorized:
            return set(), dict.fromkeys(
                sorted(eligible), "geometry_identity_requires_independent_relationship_evidence"
            )
        allowed = eligible & (set(IDENTITY_FIELDS) | set(SHAPE_FIELDS) | {"surface_alias"})

    # Descriptions are always rebuilt from target facts, never copied.
    for name in eligible & set(DESCRIPTION_FIELDS):
        allowed.discard(name)
        blocked[name] = "description_must_be_regenerated"

    if relation != EXACT_RGBA_DUPLICATE:
        for name in eligible & set(COLOR_FIELDS):
            blocked[name] = "target_variant_color_must_be_measured"
        for name in eligible & set(MATERIAL_FIELDS):
            blocked[name] = "material_not_safe_across_variant_relation"

    alias = _value(source_fields.get("surface_alias"))
    if (
        "surface_alias" in allowed
        and relation in {RECOLOR_VARIANT, GEOMETRY_FAMILY}
        and not is_color_neutral_alias(alias)
    ):
        allowed.remove("surface_alias")
        blocked["surface_alias"] = "color_qualified_alias"
    if "surface_alias" in allowed and relation == MATERIAL_VARIANT and not is_material_neutral_alias(alias):
        allowed.remove("surface_alias")
        blocked["surface_alias"] = "material_qualified_alias"

    for name in sorted(eligible - allowed - set(blocked)):
        blocked[name] = "field_not_allowed_for_relation"
    return allowed, blocked


def _band(score: int | None) -> str:
    if score is None:
        return "not_scorable"
    if score <= 4:
        return "strong"
    if score <= 8:
        return "usable_weak"
    if score <= 12:
        return "auxiliary_only"
    if score <= 16:
        return "excluded_from_primary_supervision"
    return "abstain_or_quarantine"


def _penalize_quality(
    source: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
    penalty: int,
    relation: str,
    source_id: str,
    policy: PropagationPolicy,
) -> dict[str, Any]:
    source = source or {}
    target = target or {}
    candidates = [
        value for value in (source.get("uncertainty_1_20"), target.get("uncertainty_1_20")) if value is not None
    ]
    calibration_state = str(source.get("calibration_state") or source.get("uncertainty_state") or "")
    if candidates:
        score: int | None = min(20, max(int(value) for value in candidates) + penalty)
    elif relation == EXACT_RGBA_DUPLICATE:
        score = None
    else:
        score = policy.conservative_uncalibrated_score
        calibration_state = "uncalibrated"

    risk_candidates = [
        value for value in (source.get("risk_upper_95"), target.get("risk_upper_95")) if value is not None
    ]
    if risk_candidates:
        risk_upper = min(1.0, max(float(value) for value in risk_candidates) + penalty / 20.0)
    elif score is not None:
        risk_upper = min(1.0, score / 20.0)
    else:
        risk_upper = None
    result = copy.deepcopy(dict(source or target))
    result.update(
        {
            "uncertainty_1_20": score,
            "uncertainty_band": _band(score),
            "risk_upper_95": risk_upper,
            "calibration_state": calibration_state or ("calibrated" if score is not None else "not_scorable"),
            "propagated_from": source_id,
            "propagation_relation": relation,
            "field_propagation_policy_version": policy.version,
            "propagation_risk_penalty": penalty,
        }
    )
    return result


def _first(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


def description_from_target_facts(fields: Mapping[str, Any]) -> str:
    """Build a literal description without importing source-variant wording."""

    obj = _first(_value(fields.get("surface_alias"))) or _first(_value(fields.get("canonical_object")))
    color = _first(_value(fields.get("primary_colors"))) or _first(_value(fields.get("primary_color")))
    material = _first(_value(fields.get("explicit_material")))
    structure = _first(_value(fields.get("structure"))) or _first(_value(fields.get("shape")))
    modifiers = [value.replace("_", " ") for value in (color, material, structure) if value]
    noun = obj.replace("_", " ") or "item"
    text = " ".join([*modifiers, noun]).strip()
    article = "An" if text[:1].lower() in "aeiou" else "A"
    return f"{article} {text}."


def propagate_fields(
    source_fields: Mapping[str, Any],
    target_fields: Mapping[str, Any],
    relation: str,
    *,
    source_id: str,
    source_quality: Mapping[str, Mapping[str, Any]] | None = None,
    target_quality: Mapping[str, Mapping[str, Any]] | None = None,
    approved_fields: Iterable[str] | None = None,
    declared_relationship: bool = False,
    strong_source_evidence: bool = False,
    calibrated_family_policy: bool = False,
    policy: PropagationPolicy = DEFAULT_PROPAGATION_POLICY,
    description_generator: Callable[[Mapping[str, Any]], str] = description_from_target_facts,
) -> PropagationResult:
    """Propagate only relation-safe fields and retain a complete policy trace.

    ``source_fields`` and ``target_fields`` are copied.  Structured source
    fields must be marked accepted/reviewed; direct scalar values are treated as
    final facts.  ``approved_fields`` can narrow either form further.
    """

    normalized = normalize_relation(relation)
    source_copy = copy.deepcopy(dict(source_fields))
    target_copy = copy.deepcopy(dict(target_fields))
    allowed, blocked = _allowed_fields(
        normalized,
        source_copy,
        declared_relationship=declared_relationship,
        strong_source_evidence=strong_source_evidence,
        calibrated_family_policy=calibrated_family_policy,
    )
    if approved_fields is not None:
        approved = {str(value) for value in approved_fields}
        for name in sorted(allowed - approved):
            blocked[name] = "field_not_approved_for_propagation"
        allowed &= approved

    penalty = 0
    if normalized == RECOLOR_VARIANT:
        penalty = policy.recolor_risk_penalty
    elif normalized == GEOMETRY_FAMILY:
        penalty = policy.geometry_risk_penalty
    elif normalized == MATERIAL_VARIANT:
        penalty = policy.material_variant_risk_penalty

    source_quality = source_quality or {}
    target_quality = target_quality or {}
    output_quality: dict[str, Mapping[str, Any]] = copy.deepcopy(dict(target_quality))
    for name in sorted(allowed):
        target_copy[name] = copy.deepcopy(source_copy[name])
        output_quality[name] = _penalize_quality(
            source_quality.get(name), target_quality.get(name), penalty, normalized, source_id, policy
        )

    regenerated = bool(allowed)
    if regenerated:
        target_copy["description"] = description_generator(target_copy)
        propagated_scores = [
            value.get("uncertainty_1_20")
            for name, value in output_quality.items()
            if name in allowed and value.get("uncertainty_1_20") is not None
        ]
        description_score = (
            min(20, max(propagated_scores) + (0 if normalized == EXACT_RGBA_DUPLICATE else 1))
            if propagated_scores
            else (None if normalized == EXACT_RGBA_DUPLICATE else policy.conservative_uncalibrated_score)
        )
        output_quality["description"] = {
            "uncertainty_1_20": description_score,
            "uncertainty_band": _band(description_score),
            "risk_upper_95": description_score / 20.0 if description_score is not None else None,
            "calibration_state": "borrowed_calibration" if description_score is not None else "not_scorable",
            "propagated_from": source_id,
            "propagation_relation": normalized,
            "field_propagation_policy_version": policy.version,
            "propagation_risk_penalty": penalty,
            "description_regenerated": True,
        }

    return PropagationResult(
        fields=target_copy,
        field_quality=output_quality,
        propagated_fields=tuple(sorted(allowed)),
        blocked_fields=dict(sorted(blocked.items())),
        propagated_from=source_id,
        propagation_relation=normalized,
        field_propagation_policy_version=policy.version,
        propagation_risk_penalty=penalty,
        description_regenerated=regenerated,
        audit={
            "declared_relationship": declared_relationship,
            "strong_source_evidence": strong_source_evidence,
            "calibrated_family_policy": calibrated_family_policy,
        },
    )


def uncertainty_from_risk_upper(risk_upper_95: float) -> int:
    """Public helper matching the v4 1-20 uncertainty contract."""

    return max(1, min(20, math.ceil(20.0 * float(risk_upper_95))))
