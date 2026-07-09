"""Safe filename-first fusion for label v2."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from spritelab.harvest.label_candidates import object_is_generic
from spritelab.harvest.label_schema import LabelSuggestion, SafeFusedLabel
from spritelab.harvest.label_taxonomy import (
    normalize_category,
    normalize_object_name,
    normalize_tag,
    normalize_tags,
    object_name_token_f1,
    tag_overlap,
)
from spritelab.harvest.sheet_specializations import is_rpg_496_profile
from spritelab.harvest.source_profiles import SourceProfile, is_exact_filename_trusted, is_prefix_family_trusted
from spritelab.harvest.visual_facts import VisualFacts


@dataclass(frozen=True)
class FusionThresholds:
    trusted_filename_threshold: float = 0.85
    auto_vlm_threshold: float = 0.8
    agreement_token_f1: float = 0.8
    review_trusted_filename_conflicts: bool = False
    filename_confidence_threshold: float = 0.65


_GENERIC_OBJECTS = {
    "",
    "unknown",
    "unknown_object",
    "unidentified",
    "unidentified_object",
    "ambiguous",
    "ambiguous_object",
    "ambiguous_shape",
    "generic_object",
    "shape",
    "generic_shape",
    "gem",
    "tool",
    "item",
    "object",
    "thing",
    "sprite",
    "icon",
    "tile",
    "grid",
    "grid_cell",
    "cell",
}

_MALFORMED_OBJECTS = {
    "sho",
    "armour",
    "ambiguou",
    "ambiguou_object",
    "ambiguou_shape",
}

_PREFIX_FAMILY_LOW_INFORMATION_OBJECTS = {
    "gold",
    "material",
    "resource",
    "loot",
    "treasure",
    "accessory",
    "jewelry",
    "item_icon",
    "equipment",
}

_PREFIX_FAMILY_SAFE_OBJECTS = {
    "antidote",
    "armor",
    "axe",
    "banana",
    "boots",
    "bottle",
    "bow",
    "breastplate",
    "carrot",
    "cheese",
    "cherry",
    "chestplate",
    "clock",
    "clover",
    "dagger",
    "grape",
    "helmet",
    "helm",
    "hat",
    "headgear",
    "key",
    "leather_armor",
    "lemon",
    "meat",
    "medal",
    "necklace",
    "orange",
    "pendant",
    "pepper",
    "pineapple",
    "poison",
    "potion",
    "radish",
    "ring",
    "shield",
    "shoes",
    "sword",
    "watermelon",
}


@dataclass(frozen=True)
class _CandidateMatch:
    object_name: str
    kind: str
    score: float = 1.0


_KNOWN_HALLUCINATION_OBJECTS = {
    "gold_bar",
    "gold_coin",
    "coin",
    "coin_stack",
    "orb",
    "stone_bottle",
    "potion",
    "potion_bottle",
    "red_potion_bottle",
    "rolled_parchment",
    "mossy_rock",
}
_CURRENCY_METAL_TAGS = {"coin", "currency", "gold", "metal", "treasure", "bar", "ingot"}
_SAFE_VLM_VISUAL_TAGS = {
    "black",
    "white",
    "gray",
    "dark_gray",
    "light_gray",
    "red",
    "dark_red",
    "orange",
    "yellow",
    "gold",
    "brown",
    "tan",
    "green",
    "dark_green",
    "lime",
    "blue",
    "dark_blue",
    "cyan",
    "teal",
    "purple",
    "pink",
    "magenta",
    "cream",
    "beige",
    "roundish",
    "tall",
    "wide",
    "thin",
    "small_content",
    "full_canvas",
    "rectangular",
    "square",
    "oval",
}


def fuse_label_v2(
    filename: LabelSuggestion | None,
    vlm: LabelSuggestion | None,
    visual_facts: VisualFacts | None,
    *,
    profile: SourceProfile,
    thresholds: FusionThresholds = FusionThresholds(),  # noqa: B008
) -> SafeFusedLabel:
    """Fuse filename and VLM suggestions without letting VLM override trusted metadata."""

    filename = filename if filename and filename.object_name else None
    vlm = vlm if vlm and _has_signal(vlm) else None
    flags: list[str] = []
    conflict_reasons: list[str] = []
    provenance: dict[str, str | list[str]] = {}
    exact_filename_trusted = is_exact_filename_trusted(profile)
    prefix_family_trusted = is_prefix_family_trusted(profile)

    if exact_filename_trusted:
        flags.append("filename_trusted")
    elif prefix_family_trusted:
        flags.append("prefix_family_trusted")
        flags.append("filename_family_not_exact")
    if filename is None or filename.confidence < thresholds.filename_confidence_threshold:
        flags.append("filename_weak")
    if visual_facts is not None and "small_content" in visual_facts.shape_hints:
        flags.append("small_content")
    if filename is not None and _low_information_filename(filename):
        flags.append("low_information_filename")
    if filename is not None and _malformed_filename_object(filename, profile):
        flags.append("malformed_filename_object")
    candidates = _candidate_objects(filename, vlm)
    if candidates:
        flags.append("candidate_object_list")
    specialization_flags = _rpg_496_specialization_flags(filename)
    flags.extend(specialization_flags)

    vlm_degenerate = _is_vlm_degenerate(vlm)
    if vlm_degenerate:
        flags.append("vlm_degenerate")
    candidate_match = _candidate_match(vlm, candidates)
    alternative_candidate_match = _alternative_candidate_match(vlm, candidates)
    known_hallucination = _is_known_hallucination(vlm)
    if (
        known_hallucination
        and prefix_family_trusted
        and candidate_match is not None
        and candidate_match.kind == "primary"
    ):
        known_hallucination = False
    if known_hallucination:
        flags.append("vlm_known_hallucination")
    candidate_conflict = False
    if candidates and vlm is not None:
        vlm_primary_generic = object_is_generic(vlm.object_name) or vlm.object_name in _GENERIC_OBJECTS
        generic_alternative_candidate = (
            prefix_family_trusted
            and vlm.source_consistency != "contradicted"
            and candidate_match is not None
            and candidate_match.kind == "alternative"
            and vlm_primary_generic
        )
        if vlm_primary_generic:
            flags.append("vlm_generic_with_candidates")
            if not generic_alternative_candidate:
                candidate_conflict = True
        if candidate_match is not None:
            if alternative_candidate_match is not None:
                flags.append("vlm_alternative_candidate_match")
            if candidate_match.kind == "alternative":
                if not generic_alternative_candidate and not _vlm_primary_is_compatible_with_candidate_family(
                    vlm, candidates, profile
                ):
                    flags.append("vlm_outside_candidate_family")
                    candidate_conflict = True
            else:
                flags.append("vlm_candidate_match")
        elif vlm.object_name:
            if _prefix_family_specialization_can_auto(filename, vlm, profile):
                flags.append("vlm_supports_rpg_496_specialization")
            else:
                flags.append("vlm_outside_candidate_family")
                candidate_conflict = True
    if candidate_conflict:
        flags.append("needs_review_candidate_conflict")
        conflict_reasons.append(
            f"candidate_family: vlm={vlm.object_name if vlm else ''}, candidates={','.join(candidates[:8])}"
        )

    agreement = _agreement_score(filename, vlm)
    if filename is not None and vlm is not None:
        if agreement >= thresholds.agreement_token_f1:
            flags.append("vlm_agrees_with_filename")
        else:
            flags.append("vlm_conflicts_with_filename")
            conflict_reasons.append(f"object_name: filename={filename.object_name}, vlm={vlm.object_name}")
        if filename.category != "unknown" and vlm.category != "unknown" and filename.category != vlm.category:
            conflict_reasons.append(f"category: filename={filename.category}, vlm={vlm.category}")

    food_currency_hallucination = _food_currency_hallucination(filename, vlm, profile)
    if food_currency_hallucination:
        flags.append("vlm_food_as_currency_hallucination")
        if "vlm_known_hallucination" not in flags:
            flags.append("vlm_known_hallucination")

    if vlm is not None and vlm.source_consistency == "contradicted":
        flags.append("vlm_contradicts_source")
        if known_hallucination or food_currency_hallucination:
            if "vlm_known_hallucination" not in flags:
                flags.append("vlm_known_hallucination")
        else:
            flags.append("review_source_visual_mismatch")
            conflict_reasons.extend(vlm.evidence_against_source or (f"source visual mismatch: {vlm.object_name}",))

    if _prefix_family_specialization_can_auto(filename, vlm, profile):
        safe = _safe_from_filename(filename, vlm, visual_facts, profile, conflict=False)
        provenance = {
            "object_name": "filename_rules_v2_rpg_496_specialization",
            "category": "filename_rules_v2_rpg_496_specialization",
            "tags": ["filename_rules_v2", "rpg_496_specialization", "deterministic_visual_facts", "vlm_descriptor"],
            "description": "vlm_descriptor" if vlm and vlm.short_description else "filename_rules_v2",
        }
        return _result(
            safe,
            filename,
            vlm,
            "auto_rpg_496_specialized",
            False,
            [*flags, "auto_rpg_496_specialized"],
            conflict_reasons,
            provenance,
            0.1,
        )

    if _trusted_filename_wins(filename, profile, thresholds):
        safe = _safe_from_filename(filename, vlm, visual_facts, profile, conflict=bool(conflict_reasons))
        provenance = _filename_provenance(vlm, visual_facts)
        if vlm is None or agreement >= thresholds.agreement_token_f1:
            bucket = "auto_filename_trusted"
            flags.append("auto_filename_trusted")
            needs_review = False
            review_priority = 0.05
        else:
            bucket = "auto_filename_with_vlm_conflict"
            flags.append("auto_filename_trusted")
            severe = _severe_conflict(filename, vlm, profile) and not food_currency_hallucination
            needs_review = bool(thresholds.review_trusted_filename_conflicts and severe)
            review_priority = 0.35 if food_currency_hallucination else 0.5 if severe else 0.25
        return _result(safe, filename, vlm, bucket, needs_review, flags, conflict_reasons, provenance, review_priority)

    if (
        filename is not None
        and vlm is not None
        and agreement >= thresholds.agreement_token_f1
        and not vlm_degenerate
        and not candidate_conflict
    ):
        if prefix_family_trusted:
            if _prefix_family_agreement_can_auto(
                filename,
                vlm,
                profile,
                candidates=candidates,
                candidate_match=candidate_match,
                candidate_conflict=candidate_conflict,
            ):
                safe = _merged_label(filename, vlm, visual_facts, preferred=filename)
                provenance = {
                    "object_name": "filename_rules_v2_prefix_family+vlm_descriptor",
                    "category": "filename_rules_v2_prefix_family",
                    "tags": ["filename_rules_v2", "deterministic_visual_facts", "vlm_descriptor"],
                    "description": "vlm_descriptor" if vlm.short_description else "filename_rules_v2",
                }
                return _result(
                    safe,
                    filename,
                    vlm,
                    "auto_prefix_family_trusted",
                    False,
                    [*flags, "auto_prefix_family_trusted"],
                    conflict_reasons,
                    provenance,
                    0.12,
                )
        else:
            safe = _merged_label(
                filename, vlm, visual_facts, preferred=filename if filename.confidence >= vlm.confidence else vlm
            )
            provenance = {
                "object_name": "filename_rules_v2+vlm_descriptor",
                "category": "filename_rules_v2+vlm_descriptor",
                "tags": ["filename_rules_v2", "deterministic_visual_facts", "vlm_descriptor"],
                "description": "vlm_descriptor" if vlm.short_description else "filename_rules_v2",
            }
            return _result(
                safe,
                filename,
                vlm,
                "fused_automatically",
                False,
                [*flags, "vlm_agrees_with_filename"],
                conflict_reasons,
                provenance,
                0.05,
            )

    candidate_winner = _vlm_candidate_can_win(
        filename,
        vlm,
        thresholds,
        profile=profile,
        vlm_degenerate=vlm_degenerate,
        known_hallucination=known_hallucination,
        candidates=candidates,
        candidate_match=candidate_match,
        candidate_conflict=candidate_conflict,
    )
    if candidate_winner is not None:
        safe = _with_visual_facts(candidate_winner, visual_facts, source="vlm_candidate_ranked")
        provenance = {
            "object_name": "vlm_descriptor+source_candidates",
            "category": "vlm_descriptor+source_candidates",
            "tags": ["vlm_descriptor", "source_candidates", "deterministic_visual_facts"],
            "description": "vlm_descriptor",
        }
        return _result(
            safe,
            filename,
            vlm,
            "auto_vlm_candidate_ranked",
            False,
            [*flags, "auto_vlm_candidate_ranked"],
            conflict_reasons,
            provenance,
            0.15,
        )

    if candidate_conflict:
        safe = _fallback_prefill(filename, vlm, visual_facts, profile=profile)
        provenance = {
            "object_name": safe.source or "fallback",
            "category": safe.source or "fallback",
            "tags": [safe.source or "fallback", "deterministic_visual_facts"],
            "description": "vlm_descriptor" if vlm and vlm.short_description else safe.source or "fallback",
        }
        return _result(
            safe,
            filename,
            vlm,
            "needs_review_candidate_conflict",
            True,
            flags,
            conflict_reasons,
            provenance,
            0.85,
        )

    if _vlm_can_win(
        filename,
        vlm,
        thresholds,
        profile=profile,
        vlm_degenerate=vlm_degenerate,
        known_hallucination=known_hallucination,
    ):
        safe = _with_visual_facts(vlm, visual_facts, source="vlm_descriptor")
        provenance = {
            "object_name": "vlm_descriptor",
            "category": "vlm_descriptor",
            "tags": ["vlm_descriptor", "deterministic_visual_facts"],
            "description": "vlm_descriptor",
        }
        return _result(
            safe,
            filename,
            vlm,
            "auto_vlm_when_filename_weak",
            False,
            [*flags, "auto_vlm_when_filename_weak"],
            conflict_reasons,
            provenance,
            0.2,
        )

    safe = _fallback_prefill(filename, vlm, visual_facts, profile=profile)
    if filename is None or filename.confidence < 0.65:
        flags.append("filename_weak")
    if conflict_reasons:
        flags.append("needs_review_conflict")
    provenance = {
        "object_name": safe.source or "fallback",
        "category": safe.source or "fallback",
        "tags": [safe.source or "fallback", "deterministic_visual_facts"],
        "description": "vlm_descriptor" if vlm and vlm.short_description else safe.source or "fallback",
    }
    priority = 0.9 if conflict_reasons or vlm_degenerate or known_hallucination else 0.7
    return _result(safe, filename, vlm, "needs_review", True, flags, conflict_reasons, provenance, priority)


def _result(
    safe: LabelSuggestion,
    filename: LabelSuggestion | None,
    vlm: LabelSuggestion | None,
    bucket: str,
    needs_review: bool,
    flags: Iterable[str],
    conflict_reasons: Iterable[str],
    provenance: dict[str, str | list[str]],
    review_priority: float,
) -> SafeFusedLabel:
    return SafeFusedLabel(
        safe_prefill=safe,
        filename_suggestion=filename,
        vlm_suggestion=vlm,
        fused_suggestion=safe,
        bucket=bucket,
        needs_review=needs_review,
        flags=tuple(flags),
        conflict_reasons=tuple(conflict_reasons),
        provenance=provenance,
        review_priority=review_priority,
    )


def _trusted_filename_wins(
    filename: LabelSuggestion | None, profile: SourceProfile, thresholds: FusionThresholds
) -> bool:
    if filename is None:
        return False
    return (
        is_exact_filename_trusted(profile)
        and filename.confidence >= thresholds.trusted_filename_threshold
        and not _low_information_filename(filename)
    )


def _low_information_filename(filename: LabelSuggestion) -> bool:
    return (
        object_is_generic(filename.object_name)
        or normalize_object_name(filename.object_name) in _GENERIC_OBJECTS
        or filename.confidence <= 0.25
    )


def _malformed_filename_object(filename: LabelSuggestion, profile: SourceProfile) -> bool:
    object_name = normalize_object_name(filename.object_name)
    if object_name in _MALFORMED_OBJECTS:
        return True
    if is_prefix_family_trusted(profile) and object_name in {"elm"}:
        return True
    return False


def _prefix_family_agreement_can_auto(
    filename: LabelSuggestion,
    vlm: LabelSuggestion,
    profile: SourceProfile,
    *,
    candidates: tuple[str, ...],
    candidate_match: _CandidateMatch | None,
    candidate_conflict: bool,
) -> bool:
    if not is_prefix_family_trusted(profile):
        return False
    if candidate_conflict:
        return False
    if vlm.source_consistency == "contradicted":
        return False
    for suggestion in (filename, vlm):
        object_name = normalize_object_name(suggestion.object_name)
        if not object_name or object_name != suggestion.object_name:
            return False
        if object_is_generic(object_name) or object_name in _GENERIC_OBJECTS:
            return False
        if object_name in _MALFORMED_OBJECTS or object_name in _PREFIX_FAMILY_LOW_INFORMATION_OBJECTS:
            return False
        if object_name == "elm":
            return False
    if candidates:
        if candidate_match is not None and candidate_match.kind == "alternative":
            return False
        return filename.object_name in candidates or vlm.object_name in candidates or candidate_match is not None
    return _is_known_safe_prefix_family_object(filename.object_name, profile)


def _prefix_family_specialization_can_auto(
    filename: LabelSuggestion | None,
    vlm: LabelSuggestion | None,
    profile: SourceProfile,
) -> bool:
    if filename is None or vlm is None:
        return False
    if not is_prefix_family_trusted(profile) or not is_rpg_496_profile(profile):
        return False
    flags = set(_rpg_496_specialization_flags(filename))
    if not flags:
        return False
    if vlm.source_consistency == "contradicted" or _is_vlm_degenerate(vlm):
        return False
    if _malformed_filename_object(filename, profile) or _low_information_filename(filename):
        return False
    return _vlm_supports_rpg_496_specialization(filename, vlm, flags)


def _rpg_496_specialization_flags(filename: LabelSuggestion | None) -> tuple[str, ...]:
    if filename is None:
        return ()
    prefix = "rpg_496_specialization:"
    return normalize_tags(
        str(value).removeprefix(prefix) for value in filename.evidence if str(value).startswith(prefix)
    )


def _vlm_supports_rpg_496_specialization(filename: LabelSuggestion, vlm: LabelSuggestion, flags: set[str]) -> bool:
    filename_object = normalize_object_name(filename.object_name)
    vlm_object = normalize_object_name(vlm.object_name)
    if not filename_object:
        return False
    if filename_object == vlm_object or object_name_token_f1(filename_object, vlm_object) >= 0.8:
        return True
    alternatives = set(normalize_tags(vlm.alternative_object_names))
    if filename_object in alternatives:
        return True

    terms = _suggestion_terms(vlm)
    prefix = _filename_evidence_value(filename, "rpg_prefix")
    family = _filename_evidence_value(filename, "filename_object")

    if "rpg_496_vlm_alternative_promoted" in flags:
        return True
    if "rpg_496_color_potion_promoted" in flags:
        return bool({"potion", "liquid", "bottle", "vial", "flask", "cork", "glass"} & terms) or vlm_object in {
            "potion",
            "bottle",
            "vial",
            "flask",
        }
    if "rpg_496_shield_shape_override" in flags:
        return bool(
            {"shield", "shield_shape", "shield_like", "bordered", "wood_grain", "metallic"} & terms
        ) or vlm_object in {"wood", "metal", "shield"}
    if "rpg_496_arrow_variant_promoted" in flags:
        return prefix == "s" and (
            family == "bow"
            or vlm_object in {"bow", "arrow"}
            or bool({"bow", "arrow", "weapon_shape", "curved"} & terms)
        )
    if "rpg_496_filename_variant_promoted" in flags:
        if prefix in {"s", "w"}:
            return True
        if family and (family == vlm_object or family in terms):
            return True
        if family == "cannon" and filename_object.endswith("_cannon"):
            if "fire" in filename_object and bool({"fire", "flame", "blast", "orange"} & terms):
                return True
            if "yellow" in filename_object and bool({"yellow", "gold"} & terms):
                return True
            if "nature" in filename_object and bool({"green", "lime", "olive", "nature", "leaf", "plant"} & terms):
                return True
        if filename_object == "fire_feather" and bool({"fire", "flame", "orange"} & terms):
            return True
        if (
            filename_object == "yellow_ink_bucket"
            and bool({"yellow"} & terms)
            and bool({"bucket", "container", "container_like"} & terms)
        ):
            return True
        object_parts = set(filename_object.split("_"))
        return bool(object_parts & terms)
    if "rpg_496_material_category_override" in flags:
        return filename_object == vlm_object or family == vlm_object
    return False


def _suggestion_terms(suggestion: LabelSuggestion) -> set[str]:
    values = [
        suggestion.object_name,
        suggestion.category,
        suggestion.short_description,
        *suggestion.tags,
        *suggestion.alternative_object_names,
        *suggestion.evidence,
        *suggestion.evidence_for_source,
        *suggestion.evidence_against_source,
        *suggestion.materials,
        *suggestion.dominant_colors,
    ]
    terms: set[str] = set()
    for value in values:
        normalized = normalize_tag(str(value))
        if not normalized:
            continue
        terms.add(normalized)
        terms.update(part for part in normalized.split("_") if part)
    return terms


def _filename_evidence_value(filename: LabelSuggestion, key: str) -> str:
    prefix = f"{key}:"
    for value in filename.evidence:
        text = str(value)
        if text.startswith(prefix):
            return normalize_tag(text[len(prefix) :])
    return ""


def _is_known_safe_prefix_family_object(object_name: str, profile: SourceProfile) -> bool:
    if not is_prefix_family_trusted(profile):
        return False
    normalized = normalize_object_name(object_name)
    return normalized in _PREFIX_FAMILY_SAFE_OBJECTS


def _has_signal(suggestion: LabelSuggestion) -> bool:
    return (
        suggestion.category != "unknown"
        or bool(suggestion.object_name)
        or bool(suggestion.tags)
        or bool(suggestion.short_description)
        or bool(suggestion.warnings)
    )


def _agreement_score(filename: LabelSuggestion | None, vlm: LabelSuggestion | None) -> float:
    if filename is None or vlm is None:
        return 0.0
    if filename.object_name and vlm.object_name:
        score = object_name_token_f1(filename.object_name, vlm.object_name)
        if score > 0:
            return score
    if not filename.tags and not vlm.tags:
        return 0.0
    return tag_overlap(filename.tags, vlm.tags)


def _safe_from_filename(
    filename: LabelSuggestion,
    vlm: LabelSuggestion | None,
    visual_facts: VisualFacts | None,
    profile: SourceProfile,
    *,
    conflict: bool,
) -> LabelSuggestion:
    category = _profile_category(filename.category, profile)
    tags = list(filename.tags)
    if visual_facts is not None:
        tags.extend(visual_facts.dominant_colors)
        tags.extend(visual_facts.shape_hints)
    if vlm is not None and not conflict:
        tags.extend(tag for tag in vlm.tags if tag in _SAFE_VLM_VISUAL_TAGS)
    description = filename.short_description
    if vlm is not None and vlm.short_description and not conflict:
        description = vlm.short_description
    return LabelSuggestion(
        category=category,
        object_name=filename.object_name,
        tags=normalize_tags(tags),
        short_description=description or _filename_description(filename.object_name, category),
        confidence=filename.confidence,
        confidence_reason=filename.confidence_reason,
        source="filename_rules_v2",
        materials=filename.materials,
        mood=filename.mood,
        dominant_colors=visual_facts.dominant_colors if visual_facts is not None else filename.dominant_colors,
        warnings=filename.warnings,
        evidence=(
            *filename.evidence,
            *((f"vlm_note:{vlm.object_name}",) if conflict and vlm and vlm.object_name else ()),
        ),
        source_consistency=filename.source_consistency,
        candidate_object_names=filename.candidate_object_names,
    )


def _profile_category(category: str, profile: SourceProfile) -> str:
    if profile.name == "cc0_tool":
        return "tool"
    if profile.name == "cc0_gem":
        return "material"
    if profile.name == "cc0_food":
        return "item_icon"
    if profile.name == "cc0_potion":
        return "item_icon"
    return normalize_category(category)


def _merged_label(
    filename: LabelSuggestion,
    vlm: LabelSuggestion,
    visual_facts: VisualFacts | None,
    *,
    preferred: LabelSuggestion,
) -> LabelSuggestion:
    other = vlm if preferred is filename else filename
    tags = [preferred.object_name, *preferred.tags, *other.tags]
    if visual_facts is not None:
        tags.extend(visual_facts.dominant_colors)
        tags.extend(visual_facts.shape_hints)
    return LabelSuggestion(
        category=preferred.category if preferred.category != "unknown" else other.category,
        object_name=preferred.object_name or other.object_name,
        tags=normalize_tags(tags),
        short_description=vlm.short_description or filename.short_description,
        confidence=max(filename.confidence, vlm.confidence),
        confidence_reason="filename and VLM descriptor agree",
        source="label_fusion_v2",
        materials=normalize_tags((*filename.materials, *vlm.materials)),
        mood=normalize_tags((*filename.mood, *vlm.mood)),
        dominant_colors=visual_facts.dominant_colors
        if visual_facts is not None
        else normalize_tags((*filename.dominant_colors, *vlm.dominant_colors)),
        warnings=(*filename.warnings, *vlm.warnings),
        evidence=(*filename.evidence, *vlm.evidence),
        source_consistency=vlm.source_consistency,
        alternative_object_names=vlm.alternative_object_names,
        evidence_for_source=vlm.evidence_for_source,
        evidence_against_source=vlm.evidence_against_source,
        candidate_object_names=vlm.candidate_object_names or filename.candidate_object_names,
    )


def _with_visual_facts(
    suggestion: LabelSuggestion, visual_facts: VisualFacts | None, *, source: str
) -> LabelSuggestion:
    tags = list(suggestion.tags)
    if visual_facts is not None:
        tags.extend(visual_facts.dominant_colors)
        tags.extend(visual_facts.shape_hints)
    return LabelSuggestion(
        category=suggestion.category,
        object_name=suggestion.object_name,
        tags=normalize_tags(tags),
        short_description=suggestion.short_description,
        confidence=suggestion.confidence,
        confidence_reason=suggestion.confidence_reason,
        source=source,
        materials=suggestion.materials,
        mood=suggestion.mood,
        dominant_colors=visual_facts.dominant_colors if visual_facts is not None else suggestion.dominant_colors,
        warnings=suggestion.warnings,
        evidence=suggestion.evidence,
        source_consistency=suggestion.source_consistency,
        alternative_object_names=suggestion.alternative_object_names,
        evidence_for_source=suggestion.evidence_for_source,
        evidence_against_source=suggestion.evidence_against_source,
        candidate_object_names=suggestion.candidate_object_names,
    )


def _fallback_prefill(
    filename: LabelSuggestion | None,
    vlm: LabelSuggestion | None,
    visual_facts: VisualFacts | None,
    *,
    profile: SourceProfile | None = None,
) -> LabelSuggestion:
    if filename is not None and profile is not None and _malformed_filename_object(filename, profile):
        return _unknown_review_label(visual_facts)
    if filename is not None and filename.object_name:
        return _with_visual_facts(filename, visual_facts, source="filename_rules_v2")
    if vlm is not None and not _is_vlm_degenerate(vlm):
        return _with_visual_facts(vlm, visual_facts, source="vlm_descriptor")
    return _unknown_review_label(visual_facts)


def _unknown_review_label(visual_facts: VisualFacts | None) -> LabelSuggestion:
    return LabelSuggestion(
        category="unknown",
        object_name="",
        tags=visual_facts.dominant_colors if visual_facts is not None else (),
        short_description="",
        confidence=0.0,
        source="label_fusion_v2",
        dominant_colors=visual_facts.dominant_colors if visual_facts is not None else (),
    )


def _filename_provenance(vlm: LabelSuggestion | None, visual_facts: VisualFacts | None) -> dict[str, str | list[str]]:
    tag_sources = ["filename_rules_v2"]
    if visual_facts is not None:
        tag_sources.append("deterministic_visual_facts")
    if vlm is not None:
        tag_sources.append("vlm_descriptor")
    return {
        "object_name": "filename_rules_v2",
        "category": "filename_rules_v2",
        "tags": tag_sources,
        "description": "vlm_descriptor" if vlm is not None and vlm.short_description else "filename_rules_v2",
    }


def _vlm_can_win(
    filename: LabelSuggestion | None,
    vlm: LabelSuggestion | None,
    thresholds: FusionThresholds,
    *,
    profile: SourceProfile,
    vlm_degenerate: bool,
    known_hallucination: bool,
) -> bool:
    if vlm is None or vlm_degenerate or known_hallucination:
        return False
    if object_is_generic(vlm.object_name) or vlm.object_name in _GENERIC_OBJECTS:
        return False
    candidates = _candidate_objects(filename, vlm)
    if candidates and vlm.object_name not in candidates:
        return False
    if is_exact_filename_trusted(profile) and (filename is None or _low_information_filename(filename)):
        return False
    if (
        is_prefix_family_trusted(profile)
        and filename is not None
        and filename.confidence >= thresholds.filename_confidence_threshold
    ):
        return False
    filename_confidence = filename.confidence if filename is not None else 0.0
    return (
        filename_confidence < thresholds.filename_confidence_threshold
        and vlm.confidence >= thresholds.auto_vlm_threshold
        and vlm.object_name not in _GENERIC_OBJECTS
    )


def _vlm_candidate_can_win(
    filename: LabelSuggestion | None,
    vlm: LabelSuggestion | None,
    thresholds: FusionThresholds,
    *,
    profile: SourceProfile,
    vlm_degenerate: bool,
    known_hallucination: bool,
    candidates: tuple[str, ...],
    candidate_match: _CandidateMatch | None,
    candidate_conflict: bool,
) -> LabelSuggestion | None:
    if vlm is None or not candidates or vlm_degenerate or known_hallucination:
        return None
    if vlm.source_consistency == "contradicted":
        return None
    if candidate_match is None or candidate_conflict:
        return None
    vlm_primary_generic = object_is_generic(vlm.object_name) or vlm.object_name in _GENERIC_OBJECTS
    if (
        vlm_primary_generic
        and candidate_match.kind != "primary"
        and not (is_prefix_family_trusted(profile) and candidate_match.kind == "alternative")
    ):
        return None
    if _profile_category(vlm.category, profile) == "unknown":
        return None
    filename_confidence = filename.confidence if filename is not None else 0.0
    filename_weak = filename is None or filename_confidence < 0.65 or _low_information_filename(filename)
    if not filename_weak and not is_prefix_family_trusted(profile):
        return None
    if vlm.confidence < min(thresholds.auto_vlm_threshold, thresholds.filename_confidence_threshold):
        return None
    if candidate_match.kind == "primary":
        return vlm
    if candidate_match.kind == "alternative":
        if vlm_primary_generic and not is_prefix_family_trusted(profile):
            return None
        if not vlm_primary_generic and not _vlm_primary_is_compatible_with_candidate_family(vlm, candidates, profile):
            return None
        return replace(
            vlm,
            object_name=candidate_match.object_name,
            confidence=min(vlm.confidence, 0.75),
            confidence_reason=(vlm.confidence_reason or "VLM alternative matched source candidates"),
        )
    if candidate_match.kind == "token_overlap":
        return replace(
            vlm,
            object_name=candidate_match.object_name,
            confidence=min(vlm.confidence, 0.8),
            confidence_reason=(vlm.confidence_reason or "VLM object strongly overlapped source candidates"),
        )
    return None


def _candidate_match(vlm: LabelSuggestion | None, candidates: tuple[str, ...]) -> _CandidateMatch | None:
    if vlm is None or not candidates:
        return None
    normalized_candidates = normalize_tags(candidates)
    object_name = normalize_object_name(vlm.object_name)
    if object_name in normalized_candidates:
        return _CandidateMatch(object_name, "primary")
    for alternative in vlm.alternative_object_names:
        candidate = normalize_object_name(alternative)
        if candidate in normalized_candidates:
            return _CandidateMatch(candidate, "alternative")
    best_name = ""
    best_score = 0.0
    for candidate in normalized_candidates:
        score = object_name_token_f1(object_name, candidate)
        if score > best_score:
            best_name = candidate
            best_score = score
    if best_name and best_score >= 0.8:
        return _CandidateMatch(best_name, "token_overlap", best_score)
    return None


def _alternative_candidate_match(vlm: LabelSuggestion | None, candidates: tuple[str, ...]) -> _CandidateMatch | None:
    if vlm is None or not candidates:
        return None
    normalized_candidates = normalize_tags(candidates)
    for alternative in vlm.alternative_object_names:
        candidate = normalize_object_name(alternative)
        if candidate in normalized_candidates:
            return _CandidateMatch(candidate, "alternative")
    return None


def _vlm_primary_is_compatible_with_candidate_family(
    vlm: LabelSuggestion,
    candidates: tuple[str, ...],
    profile: SourceProfile,
) -> bool:
    object_name = normalize_object_name(vlm.object_name)
    if object_name in normalize_tags(candidates):
        return True
    if _is_known_safe_prefix_family_object(object_name, profile):
        return True
    return any(object_name_token_f1(object_name, candidate) >= 0.5 for candidate in candidates)


def _is_vlm_degenerate(vlm: LabelSuggestion | None) -> bool:
    if vlm is None:
        return False
    text = " ".join([vlm.object_name, vlm.short_description, *vlm.tags, *vlm.warnings]).lower()
    if "degenerate" in text or "checkerboard" in text or "grid pattern" in text:
        return True
    if (vlm.object_name in _GENERIC_OBJECTS or object_is_generic(vlm.object_name)) and vlm.confidence >= 0.8:
        return True
    return False


def _is_known_hallucination(vlm: LabelSuggestion | None) -> bool:
    if vlm is None:
        return False
    return vlm.object_name in _KNOWN_HALLUCINATION_OBJECTS


def _candidate_objects(filename: LabelSuggestion | None, vlm: LabelSuggestion | None) -> tuple[str, ...]:
    values: list[str] = []
    if filename is not None:
        values.extend(filename.candidate_object_names)
    if vlm is not None:
        values.extend(vlm.candidate_object_names)
    return normalize_tags(values)


def _food_currency_hallucination(
    filename: LabelSuggestion | None, vlm: LabelSuggestion | None, profile: SourceProfile
) -> bool:
    if filename is None or vlm is None:
        return False
    filename_food = profile.domain == "food" or "food" in filename.tags
    if not filename_food:
        return False
    if vlm.object_name in _KNOWN_HALLUCINATION_OBJECTS:
        return True
    vlm_tags = set(vlm.tags)
    return bool(vlm_tags & _CURRENCY_METAL_TAGS) and not bool(vlm_tags & {"food", "fruit", "dairy", "vegetable"})


def _severe_conflict(filename: LabelSuggestion, vlm: LabelSuggestion | None, profile: SourceProfile) -> bool:
    if vlm is None:
        return False
    if _food_currency_hallucination(filename, vlm, profile):
        return False
    if filename.category != "unknown" and vlm.category != "unknown" and filename.category != vlm.category:
        return True
    return object_name_token_f1(filename.object_name, vlm.object_name) == 0.0


def _filename_description(object_name: str, category: str) -> str:
    if not object_name:
        return ""
    return f"A 32x32 pixel-art {object_name.replace('_', ' ')} {category.replace('_', ' ')}."
