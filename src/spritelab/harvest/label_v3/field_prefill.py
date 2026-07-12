"""Uncalibrated, field-level GUI proposals for Auto-Labeling v3.

Prefills rank normalized evidence for human review.  They are deliberately
separate from :class:`FieldDecision`: confidence is a deterministic ranking
score, never a calibrated probability and never authority to auto-accept.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from typing import Any

from spritelab.harvest.label_v3.evidence import EvidenceItem

SCHEMA_VERSION = "field_prefill_v3.2"

DOMAIN_VALUES = {
    "inventory_icon",
    "item_icon",
    "entity_sprite",
    "character_sprite",
    "effect_icon",
    "environment_tile",
    "environment_prop",
    "ui_icon",
    "unknown",
}
CATEGORY_VALUES = {
    "weapon",
    "armor",
    "tool",
    "material",
    "food",
    "plant",
    "gem",
    "effect",
    "entity",
    "environment",
    "ui",
    "unknown",
}
MATERIAL_VALUES = {"metal", "wood", "stone", "crystal", "glass", "cloth", "leather", "organic", "unknown"}
ROLE_VALUES = {"item", "resource", "equipment", "weapon", "consumable", "crafting_material", "decoration", "unknown"}
STYLE_VALUES = {"pixel_art", "outlined", "isometric", "front_facing", "top_down", "high_contrast"}
COLOR_ALIASES = {
    "teal-green": "teal",
    "greenish-blue": "teal",
    "cyan": "cyan",
    "grey": "gray",
    "violet": "purple",
    "magenta": "pink",
    "golden": "yellow",
}
COLOR_VALUES = {
    "red",
    "orange",
    "yellow",
    "green",
    "teal",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
    "tan",
    "beige",
    "white",
    "gray",
    "black",
    "transparent",
}
GEM_ALIASES = {
    "agate",
    "amethyst",
    "jade",
    "diamond",
    "ruby",
    "sapphire",
    "emerald",
    "topaz",
    "garnet",
    "opal",
    "quartz",
    "pearl",
    "turquoise",
}
ALIAS_SPELLINGS = {"amethist": "amethyst"}
GENERIC_FILENAME_TOKENS = {"tile", "sprite", "asset", "image", "img", "icon", "item", "object", "frame"}


@dataclass(frozen=True)
class PrefillAlternative:
    value: Any
    score: float
    sources: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"value": self.value, "score": round(float(self.score), 4), "sources": list(self.sources)}


@dataclass(frozen=True)
class FieldPrefill:
    schema_version: str = SCHEMA_VERSION
    sprite_id: str = ""
    field_name: str = ""
    value: Any = None
    normalized_value: Any = None
    alternatives: tuple[PrefillAlternative, ...] = ()
    confidence: float = 0.0
    confidence_kind: str = "prefill_ranking_score"
    evidence_refs: tuple[str, ...] = ()
    supporting_dependency_groups: tuple[str, ...] = ()
    conflicting_dependency_groups: tuple[str, ...] = ()
    normalization_actions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    open_set_state: str = "unknown"
    reason: str = "insufficient_evidence"
    score_components: dict[str, float] = field(default_factory=dict)
    raw_candidates: tuple[dict[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sprite_id": self.sprite_id,
            "field": self.field_name,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "alternatives": [v.to_json() for v in self.alternatives],
            "confidence": round(float(self.confidence), 4),
            "confidence_kind": self.confidence_kind,
            "evidence_refs": list(self.evidence_refs),
            "supporting_dependency_groups": list(self.supporting_dependency_groups),
            "conflicting_dependency_groups": list(self.conflicting_dependency_groups),
            "normalization_actions": list(self.normalization_actions),
            "warnings": list(self.warnings),
            "open_set_state": self.open_set_state,
            "reason": self.reason,
            "score_components": {k: round(float(v), 4) for k, v in self.score_components.items()},
            "raw_candidates": [dict(v) for v in self.raw_candidates],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> FieldPrefill:
        alternatives = tuple(
            PrefillAlternative(
                value=v.get("value"), score=float(v.get("score", 0.0)), sources=tuple(v.get("sources") or ())
            )
            for v in data.get("alternatives") or ()
            if isinstance(v, Mapping)
        )
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            sprite_id=str(data.get("sprite_id", "")),
            field_name=str(data.get("field", "")),
            value=data.get("value"),
            normalized_value=data.get("normalized_value", data.get("value")),
            alternatives=alternatives,
            confidence=float(data.get("confidence", 0.0)),
            confidence_kind=str(data.get("confidence_kind", "prefill_ranking_score")),
            evidence_refs=tuple(str(v) for v in data.get("evidence_refs") or ()),
            supporting_dependency_groups=tuple(str(v) for v in data.get("supporting_dependency_groups") or ()),
            conflicting_dependency_groups=tuple(str(v) for v in data.get("conflicting_dependency_groups") or ()),
            normalization_actions=tuple(str(v) for v in data.get("normalization_actions") or ()),
            warnings=tuple(str(v) for v in data.get("warnings") or ()),
            open_set_state=str(data.get("open_set_state", "unknown")),
            reason=str(data.get("reason", "insufficient_evidence")),
            score_components={str(k): float(v) for k, v in (data.get("score_components") or {}).items()},
            raw_candidates=tuple(dict(v) for v in data.get("raw_candidates") or () if isinstance(v, Mapping)),
        )


def filename_semantics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Parse source-facing filename semantics without changing Labeling v2."""
    raw_path = str(record.get("relative_path") or record.get("path") or record.get("filename") or "")
    stem = PurePath(raw_path.replace("\\", "/")).stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
    semantic = [
        t for t in tokens if not t.isdigit() and t not in GENERIC_FILENAME_TOKENS and not re.fullmatch(r"[a-z]?\d+", t)
    ]
    if len(semantic) != 1:
        return {"original_token": semantic[0] if len(semantic) == 1 else "", "normalized_alias": "", "generic": True}
    original = semantic[0]
    alias = ALIAS_SPELLINGS.get(original, original)
    result: dict[str, Any] = {"original_token": original, "normalized_alias": alias, "generic": False}
    if alias in GEM_ALIASES:
        result.update({"surface_alias": alias, "canonical_object": "gem", "category": "gem"})
    elif alias == "crystal":
        result.update({"surface_alias": "crystal", "canonical_object": "crystal", "category": "gem"})
    else:
        auto = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), Mapping) else {}
        safe = auto.get("label_v2_safe_prefill") if isinstance(auto.get("label_v2_safe_prefill"), Mapping) else {}
        reason = str(safe.get("confidence_reason", "")).lower()
        recognized = "recognized" in reason and "not recognized" not in reason and "unknown" not in reason
        known_object = str(safe.get("object_name", "")).strip().lower().replace(" ", "_")
        if recognized and known_object:
            result.update(
                {
                    "surface_alias": alias,
                    "canonical_object": known_object,
                    "category": _category_for_object(known_object, str(safe.get("category", ""))),
                    "generic": False,
                }
            )
        else:
            result["generic"] = True
    return result


def _category_for_object(object_name: str, raw_category: str = "") -> str:
    raw = raw_category.strip().lower().replace(" ", "_")
    if object_name in GEM_ALIASES or "crystal" in object_name:
        return "gem"
    if any(token in object_name for token in ("sword", "dagger", "axe", "bow", "arrow", "spear", "hammer")):
        return "weapon"
    if any(token in object_name for token in ("armor", "helmet", "shield", "boots", "chestplate")):
        return "armor"
    if any(token in object_name for token in ("key", "pickaxe", "shovel", "scissor", "hoe", "rod", "tool")):
        return "tool"
    if any(
        token in object_name
        for token in ("apple", "bread", "meat", "fish", "food", "fruit", "vegetable", "cheese", "potion")
    ):
        return "food"
    if any(token in object_name for token in ("flower", "plant", "mushroom", "herb", "tree")):
        return "plant"
    if raw in CATEGORY_VALUES:
        return raw
    return "material" if raw == "item" else "unknown"


def normalize_shape(raw: Any) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    text = " ".join(str(v) for v in raw) if isinstance(raw, (list, tuple)) else str(raw or "")
    low = text.lower().replace("_", " ")
    rules = (
        ("round", ("round", "orb", "circular")),
        ("oval", ("oval", "egg", "ovoid")),
        ("diamond", ("diamond", "rhomb")),
        ("triangular", ("triang", "conical", "cone")),
        ("pointed", ("pointed", "sharp tip", "protrusion")),
        ("faceted", ("facet",)),
        ("clustered", ("cluster", "pile")),
        ("irregular", ("irregular", "jagged")),
        ("elongated", ("elongated", "long narrow")),
        ("symmetrical", ("symmetr",)),
        ("rounded_base", ("rounded base",)),
    )
    tags = tuple(name for name, needles in rules if any(n in low for n in needles))
    if "round" in tags and "rounded base" in low and not re.search(r"\b(round|roundish|circular|orb)\b", low):
        tags = tuple(tag for tag in tags if tag != "round")
    actions = ("shape_prose_to_controlled_tags",) if text and ("," in text or " " in text) else ()
    return tags, text.strip(), actions


def normalize_colors(raw: Any) -> tuple[str, ...]:
    values = raw if isinstance(raw, (list, tuple)) else re.split(r"[,/]|\band\b", str(raw or ""))
    result: list[str] = []
    for value in values:
        low = str(value).lower().strip().replace("_", " ")
        low = COLOR_ALIASES.get(low, low)
        for color in COLOR_VALUES:
            if re.search(rf"\b{re.escape(color)}\b", low) and color not in result:
                result.append(color)
    return tuple(result)


def color_roles_from_evidence(evidence: Sequence[EvidenceItem], fallback_colors: Any = ()) -> dict[str, Any]:
    """Preserve visual color roles while retaining a legacy flat color list."""
    roles: dict[str, list[str]] = {
        "primary_colors": [],
        "secondary_colors": [],
        "highlight_colors": [],
        "shadow_colors": [],
    }
    outline_color = ""
    for item in evidence:
        proposed = item.proposed_value if isinstance(item.proposed_value, Mapping) else {}
        stage = proposed.get("stage_output") if isinstance(proposed.get("stage_output"), Mapping) else proposed
        for name in roles:
            for color in normalize_colors(stage.get(name, ())):
                if color not in roles[name]:
                    roles[name].append(color)
        candidate_outline = normalize_colors(stage.get("outline_color", ""))
        if candidate_outline and not outline_color:
            outline_color = candidate_outline[0]
    flattened: list[str] = []
    for name in ("primary_colors", "secondary_colors", "highlight_colors", "shadow_colors"):
        for color in roles[name]:
            if color not in flattened:
                flattened.append(color)
    if outline_color and outline_color not in flattened:
        flattened.append(outline_color)
    for color in normalize_colors(fallback_colors):
        if color not in flattened:
            flattened.append(color)
    if not roles["primary_colors"] and flattened:
        roles["primary_colors"] = [color for color in flattened if color != outline_color][:1]
    return {**roles, "outline_color": outline_color, "colors": flattened}


def _source_component(item: EvidenceItem) -> str:
    if item.evidence_family == "filename":
        return "filename"
    if item.evidence_family == "source_profile":
        return "source_profile"
    if item.evidence_family == "pack_consistency":
        return "pack_prior"
    if item.producer_stage == "vlm_stage_a_blind_descriptor":
        return "blind_vlm"
    if item.producer_stage == "vlm_stage_c_constrained_classification":
        return "vlm_classification"
    if item.producer_stage in {"vlm_stage_d_open_set_verify", "vlm_stage_e_consistency"}:
        return "verification"
    if item.evidence_family in {"deterministic_visual", "color_palette"}:
        return "pixel_analysis"
    return item.evidence_family


COMPONENT_WEIGHTS = {
    "filename": 0.96,
    "source_profile": 0.84,
    "pack_prior": 0.80,
    "blind_vlm": 0.48,
    "vlm_classification": 0.76,
    "verification": 0.72,
    "pixel_analysis": 0.68,
    "declarative_sheet_mapping": 0.98,
    "hierarchy_compatibility": 0.62,
    "legacy_v2_derived": 0.985,
}


def build_prefills(
    sprite_id: str, record: Mapping[str, Any], evidence: Sequence[EvidenceItem]
) -> tuple[dict[str, FieldPrefill], tuple[str, ...], dict[str, Any]]:
    """Normalize and rank raw evidence into field prefills."""
    filename = filename_semantics(record)
    raw_by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    morphology: list[str] = []
    styles: set[str] = {"pixel_art"}

    def add(field: str, value: Any, source: str, ref: str = "", score: float | None = None, group: str = "") -> None:
        if value is None or value == "" or value == []:
            return
        raw_by_field[field].append(
            {
                "value": value,
                "source": source,
                "evidence_ref": ref,
                "raw_score": score,
                "dependency_group": group or source,
            }
        )

    if not filename.get("generic"):
        for field_name in ("surface_alias", "canonical_object", "category"):
            add(field_name, filename.get(field_name), "filename", group="filename")

    # Imported v2 safe-prefill data is preserved as a named legacy evidence
    # dependency.  It is never written back to v2 and never auto-accepted by
    # v3, but it prevents the new GUI from discarding a previously grounded
    # sheet/mapping result while raw v3 evidence is being rebuilt.
    auto = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), Mapping) else {}
    legacy = auto.get("label_v2_safe_prefill") if isinstance(auto.get("label_v2_safe_prefill"), Mapping) else {}
    try:
        legacy_confidence = float(legacy.get("confidence", 0.0))
    except (TypeError, ValueError):
        legacy_confidence = 0.0
    if legacy_confidence >= 0.7 and legacy.get("object_name"):
        legacy_object = str(legacy["object_name"]).strip().lower().replace(" ", "_")
        add("canonical_object", legacy_object, "legacy_v2_derived", group="legacy_v2_prefill")
        add(
            "category",
            _category_for_object(legacy_object, str(legacy.get("category", ""))),
            "legacy_v2_derived",
            group="legacy_v2_prefill",
        )

    profile_is_gem = False
    pack_context = record.get("_v3_pack_context") if isinstance(record.get("_v3_pack_context"), Mapping) else {}
    for item in evidence:
        proposed = item.proposed_value if isinstance(item.proposed_value, Mapping) else {}
        source = _source_component(item)
        group = item.dependency_group or item.evidence_family
        if item.evidence_family == "source_profile" and (
            proposed.get("profile_name") == "cc0_gem" or str(item.pack_id).lower().find("gem") >= 0
        ):
            profile_is_gem = True
        for field_name in (
            "domain",
            "category",
            "canonical_object",
            "surface_alias",
            "color",
            "material",
            "shape",
            "role",
        ):
            if item.target_fields and field_name not in item.target_fields:
                continue
            value = proposed.get(field_name)
            if value is None and field_name == "canonical_object":
                value = proposed.get("object_name")
            add(field_name, value, source, item.evidence_id, item.raw_score, group)
        stage = proposed.get("stage_output") if isinstance(proposed.get("stage_output"), Mapping) else {}
        if proposed.get("dominant_colors"):
            add("color", proposed["dominant_colors"], source, item.evidence_id, item.raw_score, group)
        if proposed.get("shape_hints"):
            add("shape", proposed["shape_hints"], source, item.evidence_id, item.raw_score, group)
        if proposed.get("aspect_hint"):
            add("shape", proposed["aspect_hint"], source, item.evidence_id, item.raw_score, group)
        if stage.get("colors"):
            add("color", stage["colors"], source, item.evidence_id, item.raw_score, group)
        if stage.get("shape_features"):
            morphology.append(str(stage["shape_features"]))
        if stage.get("raw_morphology"):
            morphology.append(str(stage["raw_morphology"]))
        for style in stage.get("style_attributes") or ():
            if str(style) in STYLE_VALUES:
                styles.add(str(style))
        description = str(stage.get("literal_description", "")).lower()
        if "outline" in description:
            styles.add("outlined")

    # A verified homogeneous gem profile supplies only broad, field-scoped context.
    pack_is_gem = float((pack_context.get("category_distribution") or {}).get("gem", 0.0)) >= 0.7
    if profile_is_gem or pack_is_gem:
        add("domain", "inventory_icon", "pack_prior", group="pack_context")
        add("category", "gem", "pack_prior", group="pack_context")
        add("canonical_object", "gem", "pack_prior", group="pack_context")
    elif pack_context:
        distribution = pack_context.get("category_distribution") or {}
        if distribution:
            dominant_category, ratio = sorted(distribution.items(), key=lambda pair: (-float(pair[1]), str(pair[0])))[0]
            if float(ratio) >= 0.7 and dominant_category in CATEGORY_VALUES:
                add("category", dominant_category, "pack_prior", group="pack_context")
                add("domain", "inventory_icon", "pack_prior", group="pack_context")
                add("canonical_object", dominant_category, "pack_prior", group="pack_context")
        vocabulary = list(pack_context.get("pack_candidate_vocabulary") or ())
        if len(vocabulary) == 1 and float(pack_context.get("pack_prior_strength", 0.0)) >= 0.4:
            add("canonical_object", vocabulary[0], "pack_prior", group="pack_context")

    shape_strings = [c["value"] for c in raw_by_field.get("shape", ())] + morphology
    shape_tags, raw_morphology, shape_actions = normalize_shape(shape_strings)
    if "round" in shape_tags and set(shape_tags) & {"oval", "triangular", "diamond", "clustered"}:
        shape_tags = tuple(tag for tag in shape_tags if tag != "round")
        shape_actions = (*shape_actions, "drop_broader_round_shape")
    for tag in shape_tags:
        add("shape", tag, "attribute_decomposition", group="shape_normalizer")

    normalized_categories = {
        value
        for candidate in raw_by_field.get("category", ())
        for value in _normalize_candidate(
            "category", candidate["value"], profile_is_gem or pack_is_gem, filename, shape_tags
        )
        if value != "unknown"
    }
    if normalized_categories:
        add("domain", "inventory_icon", "hierarchy_compatibility", group="semantic_hierarchy")
        if normalized_categories & {"weapon"}:
            add("role", "weapon", "hierarchy_compatibility", group="semantic_hierarchy")
        elif normalized_categories & {"armor", "tool"}:
            add("role", "equipment", "hierarchy_compatibility", group="semantic_hierarchy")
        elif normalized_categories & {"food"}:
            add("role", "consumable", "hierarchy_compatibility", group="semantic_hierarchy")
        elif normalized_categories & {"gem", "material"}:
            add("role", "resource", "hierarchy_compatibility", group="semantic_hierarchy")

    prefills: dict[str, FieldPrefill] = {}
    for field_name in ("domain", "category", "canonical_object", "surface_alias", "color", "material", "shape", "role"):
        prefills[field_name] = _rank_field(
            sprite_id,
            field_name,
            raw_by_field.get(field_name, ()),
            profile_is_gem or pack_is_gem,
            filename,
            shape_tags,
            shape_actions,
        )

    if profile_is_gem or pack_is_gem or normalized_categories & {"gem", "material"}:
        # These are review alternatives, not a hidden role decision.  The
        # calibrated role field remains abstained unless independent evidence
        # supports it.
        role_prefill = prefills["role"]
        alternatives = (
            PrefillAlternative("crafting_material", 0.56, ("gem_role_alternative",)),
            PrefillAlternative("item", 0.45, ("gem_role_alternative",)),
        )
        prefills["role"] = replace(
            role_prefill,
            value=role_prefill.value or "resource",
            normalized_value=role_prefill.normalized_value or "resource",
            alternatives=tuple(a for a in alternatives if a.value != (role_prefill.value or "resource")),
            warnings=tuple(sorted({*role_prefill.warnings, "role_prefill_not_auto_accepted"})),
            reason="gem_role_alternatives_for_review",
        )

    tags: list[str] = []
    for field_name in ("canonical_object", "surface_alias", "color", "material", "shape", "role", "domain"):
        prefill = prefills[field_name]
        if prefill.confidence < 0.55 or prefill.value in (None, "", "unknown"):
            continue
        values = prefill.value if isinstance(prefill.value, (list, tuple)) else (prefill.value,)
        for value in values:
            tag = str(value)
            if tag not in tags:
                tags.append(tag)
    metadata = {
        "raw_morphology": raw_morphology,
        "style_tags": sorted(styles),
        "filename": filename,
        "pack_homogeneity_score": float(pack_context.get("pack_homogeneity_score", 1.0 if profile_is_gem else 0.0)),
        "pack_candidate_vocabulary": list(
            pack_context.get("pack_candidate_vocabulary") or (["gem", "crystal"] if profile_is_gem else [])
        ),
        "pack_prior_strength": float(pack_context.get("pack_prior_strength", 0.8 if profile_is_gem else 0.0)),
        "pack_outlier_score": float(pack_context.get("pack_outlier_score", 0.0)),
        "color_roles": color_roles_from_evidence(evidence, prefills["color"].value),
    }
    return prefills, tuple(tags), metadata


def _rank_field(
    sprite_id: str,
    field_name: str,
    raw: Sequence[dict[str, Any]],
    profile_is_gem: bool,
    filename: Mapping[str, Any],
    shape_tags: Sequence[str],
    shape_actions: Sequence[str],
) -> FieldPrefill:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    refs: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    actions: set[str] = set()
    warnings: set[str] = set()

    for candidate in raw:
        source = str(candidate["source"])
        for value in _normalize_candidate(field_name, candidate["value"], profile_is_gem, filename, shape_tags):
            if not value or value == "unknown":
                continue
            key = str(value)
            weight = COMPONENT_WEIGHTS.get(source, 0.45)
            previous = scores[key].get(source, 0.0)
            scores[key][source] = max(previous, weight)
            if candidate.get("evidence_ref"):
                refs[key].add(str(candidate["evidence_ref"]))
            groups[key].add(str(candidate.get("dependency_group") or source))
            if str(candidate["value"]) != key:
                actions.add(f"{candidate['value']}->{key}")

    # Shape is explicitly multi-label; retain all independently normalized tags.
    if field_name == "shape" and shape_tags:
        values = list(dict.fromkeys(shape_tags))
        confidence = min(0.94, 0.58 + 0.08 * len(values))
        return FieldPrefill(
            sprite_id=sprite_id,
            field_name=field_name,
            value=values,
            normalized_value=values,
            confidence=confidence,
            evidence_refs=tuple(sorted({r for s in refs.values() for r in s})),
            supporting_dependency_groups=tuple(sorted({g for s in groups.values() for g in s})),
            normalization_actions=tuple(sorted(set(shape_actions) | actions)),
            reason="normalized_morphology",
            score_components={"attribute_decomposition": confidence},
            raw_candidates=tuple(dict(v) for v in raw),
        )

    if field_name == "color" and scores:
        values = list(scores)
        if "green" in values and "cyan" in values and "teal" not in values:
            values.insert(0, "teal")
        elif "teal" in values:
            values.remove("teal")
            values.insert(0, "teal")
        component_scores = {
            component: max(v.get(component, 0.0) for v in scores.values())
            for component in {c for v in scores.values() for c in v}
        }
        confidence = 1.0
        for component_score in component_scores.values():
            confidence *= 1.0 - component_score
        confidence = 1.0 - confidence
        return FieldPrefill(
            sprite_id=sprite_id,
            field_name=field_name,
            value=values,
            normalized_value=values,
            confidence=confidence,
            evidence_refs=tuple(sorted({r for s in refs.values() for r in s})),
            supporting_dependency_groups=tuple(sorted({g for s in groups.values() for g in s})),
            normalization_actions=tuple(sorted(actions)),
            reason="normalized_multi_color",
            score_components=component_scores,
            raw_candidates=tuple(dict(v) for v in raw),
        )

    ranked: list[tuple[str, float]] = []
    for value, components in scores.items():
        # Noisy-or rewards independent agreement without double counting a source.
        score = 1.0
        for component_score in components.values():
            score *= 1.0 - component_score
        score = 1.0 - score
        if field_name == "material" and set(components) <= {"blind_vlm", "verification"}:
            score *= 0.35
            warnings.add("material_cue_not_fact")
        ranked.append((value, round(score, 6)))
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    if not ranked:
        return FieldPrefill(
            sprite_id=sprite_id,
            field_name=field_name,
            warnings=tuple(sorted(warnings)),
            raw_candidates=tuple(dict(v) for v in raw),
        )
    top, confidence = ranked[0]
    supporting = groups[top]
    conflicts = sorted({g for value, _ in ranked[1:] for g in groups[value] if g not in supporting})
    alternatives = tuple(PrefillAlternative(value=v, score=s, sources=tuple(sorted(scores[v]))) for v, s in ranked[1:4])
    if field_name == "material" and confidence < 0.55:
        cue = PrefillAlternative(value=top, score=confidence, sources=tuple(sorted(scores[top])))
        return FieldPrefill(
            sprite_id=sprite_id,
            field_name=field_name,
            alternatives=(cue, *alternatives[:2]),
            confidence=confidence,
            evidence_refs=tuple(sorted(refs[top])),
            supporting_dependency_groups=tuple(sorted(supporting)),
            conflicting_dependency_groups=tuple(conflicts),
            normalization_actions=tuple(sorted(actions)),
            warnings=tuple(sorted(warnings)),
            reason="material_cue_below_prefill_threshold",
            score_components=dict(sorted(scores[top].items())),
            raw_candidates=tuple(dict(v) for v in raw),
        )
    reason = "independent_evidence_agreement" if len(scores[top]) > 1 else f"{next(iter(scores[top]))}_support"
    return FieldPrefill(
        sprite_id=sprite_id,
        field_name=field_name,
        value=top,
        normalized_value=top,
        alternatives=alternatives,
        confidence=confidence,
        evidence_refs=tuple(sorted(refs[top])),
        supporting_dependency_groups=tuple(sorted(supporting)),
        conflicting_dependency_groups=tuple(conflicts),
        normalization_actions=tuple(sorted(actions)),
        warnings=tuple(sorted(warnings)),
        open_set_state="known",
        reason=reason,
        score_components=dict(sorted(scores[top].items())),
        raw_candidates=tuple(dict(v) for v in raw),
    )


def _normalize_candidate(
    field_name: str, raw: Any, profile_is_gem: bool, filename: Mapping[str, Any], shape_tags: Sequence[str]
) -> tuple[str, ...]:
    values = raw if isinstance(raw, (list, tuple)) else (raw,)
    result: list[str] = []
    for item in values:
        value = re.sub(r"\s+", "_", str(item).strip().lower().replace("-", " "))
        if not value:
            continue
        if field_name == "domain":
            mapping = {
                "game_asset": "item_icon",
                "material": "item_icon",
                "gem": "inventory_icon",
                "weapon": "item_icon",
                "armor": "item_icon",
                "key": "inventory_icon",
                "food": "inventory_icon",
                "tool": "inventory_icon",
                "potion": "inventory_icon",
                "jewelry": "inventory_icon",
                "mushroom": "inventory_icon",
                "plant": "inventory_icon",
            }
            value = mapping.get(value, value)
            if value in STYLE_VALUES or value not in DOMAIN_VALUES:
                continue
        elif field_name == "category":
            mapping = {"mineral": "material", "object": "unknown", "item": "unknown", "egg": "unknown"}
            value = mapping.get(value, value)
            if value not in CATEGORY_VALUES:
                continue
        elif field_name == "canonical_object":
            value = ALIAS_SPELLINGS.get(value, value)
            if value in GEM_ALIASES:
                value = "gem"
            if value == "crystal" and "clustered" in shape_tags:
                value = "crystal_cluster"
            if value in {"pink_orb", "orb", "egg", "egg_shaped_gem"} and profile_is_gem:
                value = "gem"
            if "crystal_cluster" in value or ("cluster" in value and "crystal" in value):
                value = "crystal_cluster"
            if value in {"object", "item", "material"}:
                continue
        elif field_name == "surface_alias":
            value = ALIAS_SPELLINGS.get(value, value)
            if value in GENERIC_FILENAME_TOKENS or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
                continue
        elif field_name == "color":
            for color in normalize_colors(item):
                if color not in result:
                    result.append(color)
            continue
        elif field_name == "material":
            mapping = {"crystalline": "crystal", "rock": "stone", "mineral": "stone"}
            value = mapping.get(value, value)
            if value not in MATERIAL_VALUES:
                continue
            # Shiny/faceted visual language never establishes glass/crystal alone.
            if value in {"glass", "crystal", "stone"} and not profile_is_gem:
                continue
        elif field_name == "shape":
            normalized, _, _ = normalize_shape(item)
            for tag in normalized:
                if tag not in result:
                    result.append(tag)
            continue
        elif field_name == "role":
            mapping = {"collectible": "item", "decorative": "decoration"}
            value = mapping.get(value, value)
            if value not in ROLE_VALUES:
                continue
        if value not in result:
            result.append(value)
    return tuple(result)


def prefill_from_legacy_decision(sprite_id: str, field_name: str, decision: Mapping[str, Any]) -> FieldPrefill:
    """Backward-compatible adapter for v3.1 records without prefills."""
    value = decision.get("accepted_value")
    candidates = list(decision.get("candidates") or ())
    if value is None and candidates:
        value = candidates[0]
    alternatives = tuple(
        PrefillAlternative(v, float(score), ("legacy_v3_adapter",))
        for v, score in decision.get("n_best_alternatives") or ()
        if v != value
    )
    return FieldPrefill(
        sprite_id=sprite_id,
        field_name=field_name,
        value=value,
        normalized_value=value,
        alternatives=alternatives,
        confidence=0.0,
        warnings=("legacy_v3_uncalibrated_score_unavailable",),
        reason="legacy_v3_adapter",
        raw_candidates=tuple({"value": v, "source": "legacy_v3"} for v in candidates),
    )
