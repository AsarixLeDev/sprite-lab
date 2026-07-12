"""Source-specific object candidates for weak sheet-cell filenames."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spritelab.harvest.config_loader import load_hallucination_denylist_config
from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.label_taxonomy import normalize_object_name, normalize_tag, normalize_tags
from spritelab.harvest.sheet_specializations import is_rpg_496_profile
from spritelab.harvest.source_profiles import SourceProfile

if TYPE_CHECKING:
    from spritelab.harvest.visual_facts import VisualFacts


GEM_7SOUL1_CELL_OBJECTS: dict[str, str] = {
    "gem_7soul1_r000_c000": "round_gem",
    "gem_7soul1_r000_c001": "triangle_gem",
    "gem_7soul1_r000_c002": "diamond_gem",
    "gem_7soul1_r000_c003": "oval_gem",
    "gem_7soul1_r001_c000": "mixed_gem",
    "gem_7soul1_r001_c001": "ruby_gem",
    "gem_7soul1_r001_c002": "sapphire_gem",
    "gem_7soul1_r001_c003": "dark_blue_gem",
    "gem_7soul1_r002_c000": "red_gem",
    "gem_7soul1_r002_c001": "gray_gem",
}

TOOL_OCAL_CELL_OBJECTS: dict[str, str] = {
    "tool_ocal_r000_c001": "compass",
    "tool_ocal_r000_c002": "compass",
    "tool_ocal_r000_c004": "compass_geometric",
    "tool_ocal_r000_c005": "compass_geometric",
    "tool_ocal_r001_c000": "ruler",
    "tool_ocal_r001_c001": "ruler_triangle",
    "tool_ocal_r001_c003": "meter",
    "tool_ocal_r001_c004": "meter",
    "tool_ocal_r002_c001": "tool_case",
    "tool_ocal_r002_c002": "tool_case",
    "tool_ocal_r002_c003": "secateur",
}

FOOD_OCAL_CELL_OBJECTS: dict[str, str] = {
    "food_ocal_r000_c003": "apple",
    "food_ocal_r001_c008": "broccoli",
    "food_ocal_r003_c008": "camembert",
    "food_ocal_r005_c000": "sushi",
    "food_ocal_r005_c005": "ice_cream",
    "food_ocal_r005_c006": "donut",
    "food_ocal_r005_c007": "cookie",
    "food_ocal_r006_c003": "ice_cream_cup",
    "food_ocal_r006_c006": "orange_juice",
    "food_ocal_r008_c002": "sandwich",
}

GEM_CANDIDATES: tuple[str, ...] = (
    "round_gem",
    "triangle_gem",
    "diamond_gem",
    "oval_gem",
    "mixed_gem",
    "ruby_gem",
    "sapphire_gem",
    "dark_blue_gem",
    "red_gem",
    "gray_gem",
)

TOOL_CANDIDATES: tuple[str, ...] = (
    "compass",
    "compass_geometric",
    "ruler",
    "ruler_triangle",
    "meter",
    "tool_case",
    "secateur",
    "scissors",
    "tape_measure",
    "wiresnips",
    "wiresnips_blue",
    "wiresnips_yellow",
    "hammer",
    "wrench",
    "saw",
    "screwdriver",
    "pliers",
)

FOOD_CANDIDATES: tuple[str, ...] = (
    "apple",
    "banana",
    "blackberry",
    "blueberry",
    "raspberry",
    "coconut",
    "plum",
    "orange",
    "kiwi",
    "corn",
    "daikon",
    "broccoli",
    "cherry_tomatoes",
    "butter",
    "cheese_wedge",
    "cheese_wheel",
    "camembert",
    "milk_carton",
    "orange_juice",
    "ketchup",
    "ham",
    "steak",
    "hot_dog",
    "burrito",
    "sushi",
    "sashimi",
    "noodle_macaroni",
    "sandwich",
    "ice_cream",
    "ice_cream_cup",
    "ice_cream_sandwich",
    "donut",
    "cookie",
)

RPG_496_CANDIDATES: dict[str, tuple[str, ...]] = {
    "armor": (
        "armor",
        "chestplate",
        "golden_chestplate",
        "breastplate",
        "cuirass",
        "armor_piece",
        "leather_armor",
        "leather_chestplate",
        "mail_armor",
        "plate_armor",
    ),
    "clothing": ("clothing", "robe", "shirt", "tunic", "garment", "cloak"),
    "shoes": ("shoes", "boots", "boot", "footwear"),
    "medal": ("medal", "badge", "star_medal", "cross_medal", "award"),
    "necklace": ("necklace", "amulet", "pendant", "medallion", "charm"),
    "ring": ("ring", "metal_ring", "jewelry_ring", "signet_ring"),
    "helmet": ("helmet", "helm", "headgear"),
    "hat": ("hat", "wizard_hat", "cap", "hood", "headgear"),
    "bones": ("bones", "chest_bones", "bone", "skull", "rib", "fossil"),
    "bone": ("bone", "bones", "chest_bones", "skull", "rib", "fossil"),
    "gold": (
        "gold",
        "gold_coin",
        "coin",
        "metal_coin",
        "gold_ingot",
        "gold_nugget",
        "golden_mace",
        "golden_sword",
        "gold_spear",
        "currency",
    ),
    "metal": ("metal", "metal_coin", "metal_shield", "shield", "metal_ore", "ore", "stone", "ingot", "metal_chunk"),
    "wood": ("wood", "wooden_shield", "shield", "wood_plank", "log", "crafting_material"),
    "fabric": ("fabric", "cloth", "textile", "crafting_material"),
    "agate": ("agate", "agate_gem", "gem"),
    "amethyst": ("amethyst", "amethyst_gem", "gem"),
    "crystal": ("crystal", "gem", "raw_gem"),
    "diamond": ("diamond", "diamond_gem", "gem"),
    "ruby": ("ruby", "ruby_gem", "gem"),
    "sapphire": ("sapphire", "sapphire_gem", "gem"),
    "ore": ("ore", "metal_ore", "metal_chunk", "ingot"),
    "ingot": ("ingot", "metal_ingot", "bronze_bar", "gold_ingot"),
    "sword": ("sword", "short_sword", "long_sword", "bleeding_sword", "fire_sword", "golden_sword", "blade"),
    "axe": ("axe", "battle_axe", "hatchet"),
    "bow": (
        "bow",
        "longbow",
        "shortbow",
        "gray_bow",
        "dark_bow",
        "air_bow",
        "metal_decorated_bow",
        "electric_arrow",
        "ice_arrow",
        "silver_arrow",
        "explosive_arrow",
        "poison_arrow",
        "two_arrows",
        "multiple_arrows",
    ),
    "dagger": ("dagger", "electric_dagger", "knife", "blade"),
    "spear": ("spear", "gold_spear", "golden_spear", "lance", "polearm"),
    "mace": ("mace", "golden_mace", "club", "flail"),
    "staff": ("staff", "magic_staff", "wand"),
    "fist": ("fist", "metal_fist", "knuckle_weapon"),
    "throw": ("throw", "throwing_star", "shuriken"),
    "shield": ("shield", "wooden_shield", "metal_shield", "buckler", "round_shield"),
    "potion": (
        "potion",
        "red_potion",
        "blue_potion",
        "green_potion",
        "yellow_potion",
        "white_potion",
        "orange_potion",
        "bottle",
        "vial",
        "flask",
    ),
    "bottle": ("bottle", "medicine_bottle", "orange_bottle", "potion", "vial", "flask"),
    "vial": (
        "vial",
        "pink_vial",
        "red_vial",
        "white_vial",
        "yellow_vial",
        "purple_vial",
        "antidote_vial",
        "potion",
        "bottle",
        "flask",
    ),
    "flask": ("flask", "potion", "bottle", "vial"),
    "red": ("red_potion", "red_vial", "potion", "vial", "bottle"),
    "blue": ("blue_potion", "potion", "vial", "bottle"),
    "green": ("green_potion", "potion", "vial", "bottle"),
    "yellow": ("yellow_potion", "yellow_vial", "potion", "vial", "bottle"),
    "pink": ("pink_vial", "pink_potion", "potion", "vial", "bottle"),
    "white": ("white_vial", "white_potion", "potion", "vial", "bottle"),
    "purple": ("purple_vial", "purple_potion", "potion", "vial", "bottle"),
    "medicine": ("medicine", "medicine_bottle", "bottle", "vial", "flask"),
    "antidote": ("antidote", "antidote_vial", "vial", "potion", "bottle"),
    "food": ("food", "meat", "bread", "cheese", "fruit"),
    "meat": ("meat", "raw_meat", "food"),
    "raw_meat": ("raw_meat", "meat", "food"),
    "fish": ("fish", "fish_skewer", "food"),
    "raw_fish": ("raw_fish", "raw_fish_skewer", "fish_skewer", "fish", "food"),
    "pie": ("pie", "pie_slice", "food"),
    "bread": ("bread", "food"),
    "cheese": ("cheese", "food"),
    "banana": ("banana", "fruit", "food"),
    "carrot": ("carrot", "vegetable", "food"),
    "cherry": ("cherry", "fruit", "food"),
    "grape": ("grape", "fruit", "food"),
    "lemon": ("lemon", "fruit", "food"),
    "orange": ("orange", "fruit", "food"),
    "pepper": ("pepper", "vegetable", "food"),
    "pineapple": ("pineapple", "fruit", "food"),
    "radish": ("radish", "vegetable", "food"),
    "watermelon": ("watermelon", "watermelon_slice", "fruit", "food"),
    "cannon": ("cannon", "fire_cannon", "yellow_cannon", "nature_cannon"),
    "feather": ("feather", "fire_feather"),
    "ink": ("ink", "yellow_ink_bucket", "ink_bucket"),
    "fire": ("fire", "fire_spiral", "fireball", "flame", "spell", "effect_icon"),
    "ice": ("ice", "frost", "spell", "effect_icon"),
    "lightning": ("lightning", "spark", "spell", "effect_icon"),
    "poison": ("poison", "poison_cloud", "status_effect", "effect_icon"),
    "heal": ("heal", "healing", "spell", "effect_icon"),
    "water": ("water", "water_wave", "splash", "spell", "effect_icon"),
    "buff": ("buff", "speed_buff", "strength_buff", "status_effect", "effect_icon"),
    "light": ("light", "light_effect", "flash", "glow", "effect_icon"),
    "holy": ("holy", "pink_heart", "heart", "effect_icon"),
    "magic": ("magic", "spell", "orb", "scroll", "effect_icon"),
    "orb": ("orb", "magic_orb", "spell"),
    "scroll": ("scroll", "spell_scroll", "magic"),
}

_FALLBACK_GENERIC_OBJECT_NAMES = frozenset(
    {
        "",
        "gem",
        "tool",
        "item",
        "object",
        "icon",
        "thing",
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
    }
)

_FALLBACK_MALFORMED_CANDIDATES = frozenset({"sho", "armour", "elm", "ambiguou", "ambiguou_object", "ambiguou_shape"})


def _config_denylist_values() -> tuple[frozenset[str], frozenset[str]]:
    fallback = {
        "vlm_hallucination_objects": [],
        "malformed_objects": sorted(_FALLBACK_MALFORMED_CANDIDATES),
        "generic_objects": sorted(_FALLBACK_GENERIC_OBJECT_NAMES),
    }
    config = load_hallucination_denylist_config(fallback)
    if set(config) != {"schema_version", "vlm_hallucination_objects", "malformed_objects", "generic_objects"}:
        raise ValueError("invalid label-v2 hallucination_denylist config: unknown or missing top-level keys")
    generic = config.get("generic_objects")
    malformed = config.get("malformed_objects")
    if not isinstance(generic, list) or not isinstance(malformed, list):
        raise ValueError("invalid label-v2 hallucination_denylist config: generic and malformed lists required")
    if not all(isinstance(value, str) for value in [*generic, *malformed]):
        raise ValueError("invalid label-v2 hallucination_denylist config: values must be strings")
    return (
        frozenset(str(value).strip() for value in generic) & _FALLBACK_GENERIC_OBJECT_NAMES,
        frozenset(str(value).strip() for value in malformed) & _FALLBACK_MALFORMED_CANDIDATES,
    )


GENERIC_OBJECT_NAMES, _MALFORMED_CANDIDATES = _config_denylist_values()
_RPG_PREFIX_TOKENS = frozenset({"a", "ac", "c", "e", "i", "p", "s", "w"})


@dataclass(frozen=True)
class Rpg496ObjectSpecialization:
    object_name: str = ""
    category: str = ""
    candidate_object_names: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


_RPG_496_MATERIAL_OBJECTS = frozenset(
    {
        "agate",
        "agate_gem",
        "amethyst",
        "amethyst_gem",
        "bone",
        "bones",
        "bronze_bar",
        "bronze_coin",
        "crystal",
        "diamond",
        "diamond_gem",
        "fabric",
        "gem",
        "ingot",
        "metal",
        "metal_coin",
        "metal_ingot",
        "metal_ore",
        "ore",
        "ruby",
        "ruby_gem",
        "sapphire",
        "sapphire_gem",
        "wood",
    }
)
_RPG_496_CATEGORY_MATERIAL_OBJECTS = frozenset(
    {
        "agate",
        "agate_gem",
        "amethyst",
        "amethyst_gem",
        "crystal",
        "diamond",
        "diamond_gem",
        "fabric",
        "gem",
        "ingot",
        "metal",
        "metal_coin",
        "metal_ingot",
        "metal_ore",
        "ore",
        "ruby",
        "ruby_gem",
        "sapphire",
        "sapphire_gem",
        "wood",
    }
)
_RPG_496_COLORS = frozenset({"red", "blue", "green", "yellow", "pink", "white", "orange", "purple"})
_RPG_496_CONTAINER_WORDS = frozenset({"potion", "liquid", "bottle", "vial", "flask", "cork", "glass", "transparent"})
_RPG_496_VIAL_WORDS = frozenset({"vial", "diagonal"})
_RPG_496_SHIELD_WORDS = frozenset({"shield", "shield_shape", "shield_like", "bordered", "wood_grain", "wooden"})
_RPG_496_CHEST_WORDS = frozenset(
    {
        "chestplate",
        "breastplate",
        "cuirass",
        "chest_shape",
        "torso_shape",
        "torso_covering",
        "shoulder_coverage",
        "shoulder_guards",
        "shoulder_details",
    }
)
_RPG_496_LEATHER_WORDS = frozenset({"leather", "leather_armor", "hide"})
_RPG_496_GOLD_WORDS = frozenset({"gold", "golden", "yellow"})
_RPG_496_TUNIC_WORDS = frozenset({"tunic", "shirt", "v_neck", "short_sleeve", "fabric_like", "garment"})
_RPG_496_WIZARD_HAT_WORDS = frozenset({"wizard", "wizard_hat", "conical", "wide_brim", "pointed_hat"})

_RPG_496_FILENAME_VARIANTS: dict[str, tuple[str, str, str]] = {
    "i_c_fish": ("fish_skewer", "item_icon", "rpg_496_filename_variant_promoted"),
    "i_c_raw_fish": ("raw_fish_skewer", "item_icon", "rpg_496_filename_variant_promoted"),
    "i_c_pie": ("pie_slice", "item_icon", "rpg_496_filename_variant_promoted"),
    "i_c_watermelon": ("watermelon_slice", "item_icon", "rpg_496_filename_variant_promoted"),
    "i_c_raw_meat": ("raw_meat", "item_icon", "rpg_496_filename_variant_promoted"),
    "i_agate": ("agate_gem", "material", "rpg_496_filename_variant_promoted"),
    "i_amethyst": ("amethyst_gem", "material", "rpg_496_filename_variant_promoted"),
    "i_antidote": ("antidote_vial", "item_icon", "rpg_496_filename_variant_promoted"),
    "e_metal01": ("metal_coin", "material", "rpg_496_filename_variant_promoted"),
    "s_bow02": ("ice_arrow", "effect_icon", "rpg_496_arrow_variant_promoted"),
    "s_bow03": ("electric_arrow", "effect_icon", "rpg_496_arrow_variant_promoted"),
    "s_bow05": ("silver_arrow", "effect_icon", "rpg_496_arrow_variant_promoted"),
    "s_bow08": ("two_arrows", "effect_icon", "rpg_496_arrow_variant_promoted"),
    "s_bow11": ("multiple_arrows", "effect_icon", "rpg_496_arrow_variant_promoted"),
    "s_bow12": ("poison_arrow", "effect_icon", "rpg_496_arrow_variant_promoted"),
    "s_dagger02": ("electric_dagger", "effect_icon", "rpg_496_filename_variant_promoted"),
    "s_sword07": ("bleeding_sword", "effect_icon", "rpg_496_filename_variant_promoted"),
    "s_fire04": ("fire_spiral", "effect_icon", "rpg_496_filename_variant_promoted"),
    "s_water03": ("water_wave", "effect_icon", "rpg_496_filename_variant_promoted"),
    "w_throw004": ("throwing_star", "weapon", "rpg_496_filename_variant_promoted"),
}


def exact_sheet_object_for_record(record: Mapping[str, Any], profile: SourceProfile | None = None) -> str:
    """Return a deterministic object for known sheet-cell sprite ids."""

    maps = _maps_for_profile(profile, record)
    for key in _record_keys(record):
        for mapping in maps:
            value = _match_suffix(mapping, key)
            if value:
                return value
    return ""


def candidate_objects_for_record(
    record: Mapping[str, Any],
    profile: SourceProfile,
    filename_suggestion: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return source/profile candidate object names for weak sheet cells."""

    if is_rpg_496_profile(profile):
        return _rpg_496_candidates(record, filename_suggestion)

    exact = exact_sheet_object_for_record(record, profile)
    candidates = _profile_candidates(profile, record)
    if exact:
        return _dedupe((exact, *candidates))
    if not candidates:
        return ()
    if _filename_is_generic_or_weak(filename_suggestion):
        return candidates
    return ()


def object_is_generic(object_name: str) -> bool:
    return normalize_object_name(object_name) in GENERIC_OBJECT_NAMES


def specialize_496_rpg_object(
    record: Mapping[str, Any],
    profile: SourceProfile,
    filename_suggestion: LabelSuggestion,
    *,
    parsed_tokens: Sequence[str] = (),
    candidate_object_names: Sequence[str] = (),
    vlm: LabelSuggestion | None = None,
    visual_facts: VisualFacts | None = None,
) -> Rpg496ObjectSpecialization:
    """Return a conservative 496-RPG specific object/category refinement."""

    if not is_rpg_496_profile(profile):
        return Rpg496ObjectSpecialization()

    prefix, subtype, semantic = _rpg_496_parts(record, parsed_tokens)
    family = normalize_object_name(filename_suggestion.object_name) or (semantic[0] if semantic else "")
    filename_key = _rpg_496_filename_key(record)
    evidence_terms = _rpg_496_evidence_terms(vlm, visual_facts)
    source_terms = {prefix, subtype, family, *semantic}
    candidates = tuple(candidate_object_names) or _rpg_496_candidates(record, {"object_name": family})

    def build(
        object_name: str = "",
        category: str = "",
        *flags: str,
        extra_candidates: Sequence[str] = (),
    ) -> Rpg496ObjectSpecialization:
        normalized_object = normalize_object_name(object_name)
        normalized_category = category
        all_candidates = _rpg_496_specialized_candidates(
            normalized_object,
            family,
            prefix=prefix,
            existing=(*extra_candidates, *candidates),
        )
        return Rpg496ObjectSpecialization(
            object_name=normalized_object,
            category=normalized_category,
            candidate_object_names=all_candidates,
            flags=normalize_tags(flags),
        )

    if prefix == "e" and family in {"metal", "wood"} and _has_any(evidence_terms, _RPG_496_SHIELD_WORDS):
        if family == "wood" or _has_any(evidence_terms, {"wood", "wooden", "wood_grain", "grain", "striped"}):
            return build(
                "wooden_shield", "armor", "rpg_496_shield_shape_override", extra_candidates=("shield", "wooden_shield")
            )
        return build("shield", "armor", "rpg_496_shield_shape_override", extra_candidates=("shield", "metal_shield"))

    if prefix == "p":
        color = _rpg_496_potion_color(semantic, family, evidence_terms, visual_facts)
        if color:
            shape = _rpg_496_container_shape(evidence_terms, color)
            object_name = f"{color}_{shape}"
            flag = "rpg_496_color_potion_promoted" if shape == "potion" else "rpg_496_filename_variant_promoted"
            return build(object_name, "item_icon", flag, extra_candidates=_rpg_496_container_candidates(color))
        if family == "medicine" or "medicine" in semantic:
            shape = (
                "vial"
                if _has_any(evidence_terms, _RPG_496_VIAL_WORDS) and not _has_any(evidence_terms, {"bottle", "tall"})
                else "bottle"
            )
            return build(
                f"medicine_{shape}",
                "item_icon",
                "rpg_496_filename_variant_promoted",
                extra_candidates=("medicine_bottle", "bottle", "vial"),
            )
        if family == "antidote" or "antidote" in semantic:
            return build(
                "antidote_vial",
                "item_icon",
                "rpg_496_filename_variant_promoted",
                extra_candidates=("antidote_vial", "vial", "potion"),
            )

    if prefix in {"a", "c"}:
        if family in {"armor", "leather_armor"} and _has_any(evidence_terms, _RPG_496_CHEST_WORDS):
            if family == "leather_armor" or _has_any(evidence_terms, _RPG_496_LEATHER_WORDS | {"leather_chestplate"}):
                return build(
                    "leather_chestplate",
                    "armor",
                    "rpg_496_vlm_alternative_promoted",
                    extra_candidates=("leather_chestplate", "chestplate", "breastplate"),
                )
            if _rpg_496_gold_armor_evidence(evidence_terms | source_terms, visual_facts):
                return build(
                    "golden_chestplate",
                    "armor",
                    "rpg_496_vlm_alternative_promoted",
                    extra_candidates=("golden_chestplate", "chestplate", "breastplate"),
                )
            return build(
                "chestplate",
                "armor",
                "rpg_496_vlm_alternative_promoted",
                extra_candidates=("chestplate", "breastplate", "cuirass"),
            )
        if family == "clothing" and _has_any(evidence_terms, _RPG_496_TUNIC_WORDS):
            return build(
                "tunic", "armor", "rpg_496_vlm_alternative_promoted", extra_candidates=("tunic", "shirt", "garment")
            )
        if family == "hat" and _rpg_496_wizard_hat_evidence(vlm, evidence_terms):
            return build(
                "wizard_hat",
                "armor",
                "rpg_496_vlm_alternative_promoted",
                extra_candidates=("wizard_hat", "hat", "headgear"),
            )

    if (
        prefix == "ac"
        and family == "ring"
        and _has_any(evidence_terms, {"metal", "metallic", "silver", "gray", "gold", "golden"})
    ):
        return build(
            "metal_ring",
            "item_icon",
            "rpg_496_vlm_alternative_promoted",
            extra_candidates=("metal_ring", "ring", "jewelry_ring"),
        )

    if family in {"bone", "bones"} and _has_any(evidence_terms, {"chest", "rib", "ribs", "ribcage", "chest_bones"}):
        return build(
            "chest_bones",
            "material",
            "rpg_496_vlm_alternative_promoted",
            extra_candidates=("chest_bones", "bones", "bone", "rib"),
        )

    if (
        prefix in {"i", "e"}
        and family in {"metal", "gold"}
        and _has_any(evidence_terms, {"coin", "currency", "round_coin", "metal_coin"})
    ):
        object_name = "metal_coin" if family == "metal" else "gold_coin"
        return build(
            object_name, "material", "rpg_496_vlm_alternative_promoted", extra_candidates=(object_name, "coin", family)
        )

    if filename_key in _RPG_496_FILENAME_VARIANTS:
        object_name, category, flag = _RPG_496_FILENAME_VARIANTS[filename_key]
        return build(object_name, category, flag)

    if prefix == "i" and subtype == "c":
        if family == "mushroom":
            return build("", "item_icon", "rpg_496_material_category_override")
        if family == "raw_meat" or ("raw" in semantic and "meat" in semantic):
            return build(
                "raw_meat", "item_icon", "rpg_496_filename_variant_promoted", extra_candidates=("raw_meat", "meat")
            )
        if family == "pie" and _has_any(evidence_terms, {"slice", "wedge"}):
            return build(
                "pie_slice", "item_icon", "rpg_496_vlm_alternative_promoted", extra_candidates=("pie_slice", "pie")
            )
        if family == "watermelon" and _has_any(evidence_terms, {"slice", "wedge", "red", "green"}):
            return build(
                "watermelon_slice",
                "item_icon",
                "rpg_496_vlm_alternative_promoted",
                extra_candidates=("watermelon_slice", "watermelon"),
            )
        if family == "fish" and _has_any(evidence_terms, {"skewer", "stick", "spear", "fish_skewer"}):
            return build(
                "fish_skewer", "item_icon", "rpg_496_vlm_alternative_promoted", extra_candidates=("fish_skewer", "fish")
            )
        if family == "raw_fish" and _has_any(evidence_terms, {"skewer", "stick", "spear", "raw_fish_skewer"}):
            return build(
                "raw_fish_skewer",
                "item_icon",
                "rpg_496_vlm_alternative_promoted",
                extra_candidates=("raw_fish_skewer", "raw_fish"),
            )

    if prefix == "i":
        item_object = _rpg_496_item_object(family, evidence_terms, filename_key)
        if item_object:
            return build(item_object, "item_icon", "rpg_496_filename_variant_promoted")

    if prefix == "s":
        effect_object = _rpg_496_effect_object(family, evidence_terms, filename_key)
        if effect_object:
            return build(
                effect_object,
                "effect_icon",
                "rpg_496_arrow_variant_promoted" if "arrow" in effect_object else "rpg_496_filename_variant_promoted",
            )

    if prefix == "w":
        weapon_object = _rpg_496_weapon_object(semantic, family, evidence_terms, filename_key, visual_facts)
        if weapon_object:
            return build(weapon_object, "weapon", "rpg_496_filename_variant_promoted")

    if family in _RPG_496_CATEGORY_MATERIAL_OBJECTS and prefix in {"e", "i"} and subtype != "c":
        return build("", "material", "rpg_496_material_category_override")

    return Rpg496ObjectSpecialization(candidate_object_names=tuple(candidates))


def _profile_candidates(profile: SourceProfile, record: Mapping[str, Any]) -> tuple[str, ...]:
    text = _record_text(record)
    if profile.name == "cc0_gem" or "gem_7soul1" in text:
        return GEM_CANDIDATES
    if profile.name == "cc0_tool" or "tool_ocal" in text:
        return TOOL_CANDIDATES
    if profile.name == "cc0_food" or "food_ocal" in text:
        return FOOD_CANDIDATES
    return ()


def _rpg_496_candidates(record: Mapping[str, Any], filename_suggestion: Mapping[str, Any] | None) -> tuple[str, ...]:
    family = ""
    if isinstance(filename_suggestion, Mapping):
        family = normalize_object_name(str(filename_suggestion.get("object_name", "")))
    if not family:
        family = _rpg_496_family_from_record(record)
    prefix, subtype, semantic = _rpg_496_parts(record, ())
    scoped = _rpg_496_scoped_candidates(prefix, subtype, semantic, family)
    candidates = scoped or RPG_496_CANDIDATES.get(family, ())
    return _dedupe((family, *candidates) if family else candidates)


def _rpg_496_parts(record: Mapping[str, Any], parsed_tokens: Sequence[str]) -> tuple[str, str, tuple[str, ...]]:
    tokens = tuple(normalize_tag(str(token)) for token in parsed_tokens if normalize_tag(str(token)))
    if not tokens:
        tokens = _rpg_496_tokens(record)
    if not tokens:
        return "", "", ()
    prefix_index = -1
    for index, token in enumerate(tokens):
        if token in _RPG_PREFIX_TOKENS:
            prefix_index = index
    if prefix_index < 0:
        semantic = tuple(normalize_object_name(token) for token in tokens if token)
        return "", "", tuple(token for token in semantic if token)
    prefix = tokens[prefix_index]
    start = prefix_index + 1
    subtype = ""
    if prefix == "i" and start < len(tokens) and len(tokens[start]) == 1:
        subtype = tokens[start]
        start += 1
    semantic = tuple(normalize_object_name(token) for token in tokens[start:] if normalize_object_name(token))
    return prefix, subtype, semantic


def _rpg_496_scoped_candidates(prefix: str, subtype: str, semantic: Sequence[str], family: str) -> tuple[str, ...]:
    if prefix == "p":
        color = next(
            (token for token in semantic if token in _RPG_496_COLORS), family if family in _RPG_496_COLORS else ""
        )
        if color:
            return _rpg_496_container_candidates(color)
        if family == "medicine" or "medicine" in semantic:
            return ("medicine_bottle", "medicine", "bottle", "vial", "flask")
        if family == "antidote" or "antidote" in semantic:
            return ("antidote_vial", "antidote", "vial", "potion", "bottle")
    if prefix == "s" and family == "bow":
        return (
            "ice_arrow",
            "electric_arrow",
            "silver_arrow",
            "explosive_arrow",
            "poison_arrow",
            "two_arrows",
            "multiple_arrows",
            "bow",
        )
    if prefix == "i" and subtype == "c":
        if family == "fish":
            return ("fish_skewer", "fish", "food")
        if family == "raw_fish":
            return ("raw_fish_skewer", "raw_fish", "fish_skewer", "fish")
        if family == "pie":
            return ("pie_slice", "pie", "food")
        if family == "watermelon":
            return ("watermelon_slice", "watermelon", "fruit", "food")
        if family in {"meat", "raw_meat"}:
            return ("raw_meat", "meat", "food")
    if prefix == "w" and "gold" in semantic:
        if "mace" in semantic:
            return ("golden_mace", "mace", "weapon")
        if "sword" in semantic:
            return ("golden_sword", "sword", "weapon")
        if "spear" in semantic:
            return ("gold_spear", "golden_spear", "spear", "weapon")
    return ()


def _rpg_496_container_candidates(color: str) -> tuple[str, ...]:
    if color == "pink":
        return ("pink_vial", "pink_potion", "vial", "potion", "bottle", "flask")
    if color == "white":
        return ("white_vial", "white_potion", "vial", "potion", "bottle", "flask")
    if color == "orange":
        return ("orange_potion", "orange_bottle", "potion", "bottle", "vial", "flask")
    if color == "purple":
        return ("purple_vial", "purple_potion", "vial", "potion", "bottle", "flask")
    if color == "yellow":
        return ("yellow_potion", "yellow_vial", "potion", "vial", "bottle", "flask")
    return (f"{color}_potion", f"{color}_vial", "potion", "vial", "bottle", "flask")


def _rpg_496_family_from_record(record: Mapping[str, Any]) -> str:
    tokens = _rpg_496_tokens(record)
    if not tokens:
        return ""
    prefix_index = -1
    for index, token in enumerate(tokens):
        if token in _RPG_PREFIX_TOKENS:
            prefix_index = index
    if prefix_index >= 0:
        prefix = tokens[prefix_index]
        start = prefix_index + 1
        if prefix == "i" and start < len(tokens) and len(tokens[start]) == 1:
            start += 1
        semantic = tokens[start:] or tokens[prefix_index + 1 :]
    else:
        semantic = tokens
        prefix = ""
    if not semantic:
        return ""
    family = normalize_object_name("_".join(semantic))
    if prefix == "c" and family in {"elm", "helm"}:
        return "helmet"
    return family


def _rpg_496_tokens(record: Mapping[str, Any]) -> tuple[str, ...]:
    for field in ("filename", "relative_path", "final_png_path", "sprite_id"):
        raw = str(record.get(field, "")).strip()
        if not raw:
            continue
        stem = Path(raw).stem
        normalized = normalize_tag(stem)
        tokens: list[str] = []
        for token in normalized.split("_"):
            stripped = re.sub(r"(\D)\d+$", r"\1", token)
            stripped = normalize_tag(stripped)
            if stripped and not stripped.isdigit() and stripped not in {"oga", "496", "rpg", "icons", "32fix"}:
                tokens.append(stripped)
        if tokens:
            return tuple(tokens)
    return ()


def _rpg_496_filename_key(record: Mapping[str, Any]) -> str:
    for field in ("filename", "relative_path", "final_png_path", "sprite_id"):
        raw = str(record.get(field, "")).strip()
        if raw:
            return normalize_tag(Path(raw).stem)
    return ""


def _rpg_496_evidence_terms(vlm: LabelSuggestion | None, visual_facts: VisualFacts | None) -> set[str]:
    values: list[str] = []
    if vlm is not None:
        values.extend(
            [
                vlm.object_name,
                vlm.category,
                vlm.short_description,
                *vlm.tags,
                *vlm.alternative_object_names,
                *vlm.evidence,
                *vlm.evidence_for_source,
                *vlm.evidence_against_source,
                *vlm.dominant_colors,
                *vlm.materials,
            ]
        )
    if visual_facts is not None:
        values.extend([*visual_facts.dominant_colors, *visual_facts.shape_hints, visual_facts.aspect_hint])
    return _terms_from_values(values)


def _terms_from_values(values: Sequence[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        normalized = normalize_tag(str(value))
        if not normalized:
            continue
        terms.add(normalized)
        terms.update(part for part in normalized.split("_") if part)
    return terms


def _has_any(terms: set[str], expected: set[str] | frozenset[str]) -> bool:
    return bool(terms & set(expected))


def _rpg_496_potion_color(
    semantic: Sequence[str],
    family: str,
    terms: set[str],
    visual_facts: VisualFacts | None,
) -> str:
    filename_color = next(
        (token for token in semantic if token in _RPG_496_COLORS), family if family in _RPG_496_COLORS else ""
    )
    visual_colors = tuple(
        color
        for color in (visual_facts.dominant_colors if visual_facts is not None else ())
        if color in _RPG_496_COLORS
    )
    if filename_color == "pink" and "purple" in visual_colors and "pink" not in visual_colors:
        return "purple"
    if filename_color == "pink" and "purple" in terms and "pink" not in terms:
        return "purple"
    if visual_colors:
        primary = visual_colors[0]
        if (
            primary != filename_color
            and primary in {"purple", "yellow", "green", "blue", "red", "orange"}
            and filename_color in {"pink", "white", "orange"}
        ):
            return primary
    return filename_color


def _rpg_496_container_shape(terms: set[str], color: str) -> str:
    compact_vial = _has_any(terms, {"diagonal", "small", "small_content"}) or ("compact" in terms and "vial" in terms)
    narrow_vial = "narrow" in terms and not _has_any(terms, {"tall", "vertical"})
    if compact_vial or narrow_vial:
        return "vial"
    if color in {"pink", "purple"} and "potion" not in terms:
        return "vial"
    if "tall" in terms or "vertical" in terms:
        return "potion"
    if _has_any(terms, _RPG_496_VIAL_WORDS) and not _has_any(terms, {"potion", "bottle", "tall", "vertical"}):
        return "vial"
    if color == "white" and _has_any(terms, {"compact", "diagonal", "small", "vial"}):
        return "vial"
    if (
        color == "orange"
        and _has_any(terms, {"bottle", "drink", "beverage"})
        and not _has_any(terms, {"magic", "magical"})
    ):
        return "bottle"
    if _has_any(terms, _RPG_496_CONTAINER_WORDS) or color in _RPG_496_COLORS:
        return "potion"
    return "potion"


def _rpg_496_gold_armor_evidence(terms: set[str], visual_facts: VisualFacts | None) -> bool:
    dominant = tuple(visual_facts.dominant_colors if visual_facts is not None else ())
    if dominant and dominant[0] in {"yellow", "gold"}:
        return True
    if "golden_chestplate" in terms or "metallic_gold" in terms:
        return True
    if "golden" in terms:
        return not bool(terms & {"red", "dark_red", "orange"})
    if "yellow" in terms and not bool(terms & {"red", "dark_red", "orange"}):
        return True
    if "gold" in terms and "gold" in dominant and dominant.index("gold") <= 1:
        return True
    return False


def _rpg_496_wizard_hat_evidence(vlm: LabelSuggestion | None, terms: set[str]) -> bool:
    if vlm is None:
        return False
    names = {normalize_object_name(vlm.object_name), *normalize_tags(vlm.alternative_object_names)}
    if "wizard_hat" not in names:
        return False
    descriptive_terms = _terms_from_values(
        [
            vlm.short_description,
            *vlm.tags,
            *vlm.evidence,
            *vlm.evidence_for_source,
            *vlm.dominant_colors,
            *vlm.materials,
        ]
    )
    return _has_any(descriptive_terms, {"wizard", "magic", "magical", "witch", "pointed", "pointed_hat"})


def _rpg_496_item_object(family: str, terms: set[str], filename_key: str) -> str:
    if family == "cannon":
        if filename_key == "i_cannon02" and _has_any(terms, {"orange", "blast", "fire", "flame"}):
            return "fire_cannon"
        if filename_key == "i_cannon04" and _has_any(terms, {"yellow", "gold"}):
            return "yellow_cannon"
        if filename_key == "i_cannon05" and _has_any(terms, {"green", "lime", "nature", "leaf", "plant", "olive"}):
            return "nature_cannon"
    if family == "feather" and _has_any(terms, {"fire", "flame", "orange"}):
        return "fire_feather"
    if (
        family == "ink"
        and _has_any(terms, {"yellow"})
        and _has_any(terms, {"bucket", "container", "container_like", "vial"})
    ):
        return "yellow_ink_bucket"
    return ""


def _rpg_496_effect_object(family: str, terms: set[str], filename_key: str) -> str:
    if filename_key in _RPG_496_FILENAME_VARIANTS:
        object_name, category, _ = _RPG_496_FILENAME_VARIANTS[filename_key]
        if category == "effect_icon":
            return object_name
    arrow_evidence = family == "bow" and _has_any(
        terms, {"arrow", "arrows", "shaft", "pointed", "fletching", "arrow_nock", "nock"}
    )
    explosive_evidence = _has_any(terms, {"explosive", "explosion", "blast", "fire", "flame", "detonation", "burst"})
    if (
        family == "bow"
        and explosive_evidence
        and _has_any(terms, {"bow", "arrow", "weapon_shape", "projectile", "curved"})
    ):
        return "explosive_arrow"
    if arrow_evidence:
        if _has_any(terms, {"electric", "lightning", "spark"}):
            return "electric_arrow"
        if _has_any(terms, {"ice", "frost", "blue", "light_blue"}):
            return "ice_arrow"
        if explosive_evidence:
            return "explosive_arrow"
        if _has_any(terms, {"poison", "green", "toxic"}):
            return "poison_arrow"
        if _has_any(terms, {"silver", "gray", "grey", "metal"}):
            return "silver_arrow"
        if _has_any(terms, {"multiple", "many", "bundle"}):
            return "multiple_arrows"
        if _has_any(terms, {"two", "dual", "pair"}):
            return "two_arrows"
    if family == "dagger" and _has_any(terms, {"electric", "lightning", "spark"}):
        return "electric_dagger"
    if family == "sword":
        if _has_any(terms, {"blood", "bleeding", "red"}):
            return "bleeding_sword"
        if _has_any(terms, {"fire", "flame"}):
            return "fire_sword"
    if family == "fire":
        if _has_any(terms, {"spiral", "swirl", "curl"}):
            return "fire_spiral"
        if _has_any(terms, {"ball", "roundish", "orb"}):
            return "fireball"
    if family == "water" and _has_any(terms, {"wave", "splash"}):
        return "water_wave"
    if family == "buff":
        if filename_key == "s_buff04" or _has_any(terms, {"speed", "haste", "fast", "movement", "swiftness"}):
            return "speed_buff"
        if _has_any(terms, {"strength", "power", "strong"}):
            return "strength_buff"
    if family == "light" and _has_any(terms, {"light", "glow", "burst", "radial", "flash", "effect"}):
        return "light_effect"
    if family == "holy" and _has_any(terms, {"heart", "pink_heart"}):
        return "pink_heart"
    return ""


def _rpg_496_weapon_object(
    semantic: Sequence[str],
    family: str,
    terms: set[str],
    filename_key: str,
    visual_facts: VisualFacts | None,
) -> str:
    if filename_key in _RPG_496_FILENAME_VARIANTS:
        object_name, category, _ = _RPG_496_FILENAME_VARIANTS[filename_key]
        if category == "weapon":
            return object_name
    if "gold" in semantic:
        if "mace" in semantic:
            return "golden_mace"
        if "sword" in semantic:
            return "golden_sword"
        if "spear" in semantic:
            return "gold_spear"
    if family == "fist" and _has_any(terms, {"metal", "metallic", "gray", "light_gray", "dark_gray", "silver"}):
        return "metal_fist"
    if family == "throw" and _has_any(terms, {"star", "star_shaped", "throwing", "pointed"}):
        return "throwing_star"
    if family == "bow":
        dominant = tuple(visual_facts.dominant_colors if visual_facts is not None else ())
        if _has_any(terms, {"metal_decorated", "metallic", "metal"}) and _has_any(terms, {"decorated", "ornate"}):
            return "metal_decorated_bow"
        if _has_any(terms, {"air", "wind"}) or (dominant[:1] == ("white",) and "light_blue" in dominant):
            return "air_bow"
        if dominant[:1] == ("dark_gray",):
            return "dark_bow"
        if (
            dominant[:1]
            and dominant[0] in {"light_gray", "gray"}
            and "light_brown" not in dominant
            and "brown" not in dominant
        ):
            return "gray_bow"
    return ""


def _rpg_496_specialized_candidates(
    object_name: str,
    family: str,
    *,
    prefix: str,
    existing: Sequence[str],
) -> tuple[str, ...]:
    related: tuple[str, ...] = ()
    if object_name:
        related = RPG_496_CANDIDATES.get(object_name, ())
    if not related and object_name:
        parts = tuple(part for part in object_name.split("_") if part)
        if "potion" in parts or "vial" in parts or "bottle" in parts:
            related = tuple(part for part in (object_name, "potion", "vial", "bottle", "flask") if part)
        elif "arrow" in object_name or object_name.endswith("_arrows"):
            related = (object_name, "arrow", "bow")
        elif object_name.endswith("_shield"):
            related = (object_name, "shield")
        elif object_name.endswith("_chestplate"):
            related = (object_name, "chestplate", "breastplate")
        elif object_name.endswith("_sword"):
            related = (object_name, "sword", "blade")
        elif object_name.endswith("_dagger"):
            related = (object_name, "dagger", "blade")
    scoped = _rpg_496_scoped_candidates(prefix, "", (), family)
    return _dedupe((object_name, *related, *scoped, *existing, family))


def _maps_for_profile(profile: SourceProfile | None, record: Mapping[str, Any]) -> tuple[Mapping[str, str], ...]:
    text = _record_text(record)
    maps: list[Mapping[str, str]] = []
    if profile is None or profile.name == "cc0_gem" or "gem_7soul1" in text:
        maps.append(GEM_7SOUL1_CELL_OBJECTS)
    if profile is None or profile.name == "cc0_tool" or "tool_ocal" in text:
        maps.append(TOOL_OCAL_CELL_OBJECTS)
    if profile is None or profile.name == "cc0_food" or "food_ocal" in text:
        maps.append(FOOD_OCAL_CELL_OBJECTS)
    return tuple(maps)


def _match_suffix(mapping: Mapping[str, str], key: str) -> str:
    for suffix, object_name in mapping.items():
        if key == suffix or key.endswith(f"_{suffix}"):
            return object_name
    return ""


def _record_keys(record: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for field in ("sprite_id", "filename", "relative_path", "final_png_path"):
        raw = str(record.get(field, "")).strip()
        if not raw:
            continue
        values.append(normalize_tag(raw))
        stem = Path(raw).stem
        if stem and stem != raw:
            values.append(normalize_tag(stem))
    return _dedupe(values)


def _record_text(record: Mapping[str, Any]) -> str:
    return "_".join(
        normalize_tag(str(record.get(field, "")))
        for field in ("source_id", "source_name", "sprite_id", "filename", "relative_path", "final_png_path")
    )


def _filename_is_generic_or_weak(filename_suggestion: Mapping[str, Any] | None) -> bool:
    if not isinstance(filename_suggestion, Mapping):
        return True
    object_name = normalize_object_name(str(filename_suggestion.get("object_name", "")))
    try:
        confidence = float(filename_suggestion.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return object_name in GENERIC_OBJECT_NAMES or confidence < 0.85


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_object_name(value)
        if normalized and normalized not in _MALFORMED_CANDIDATES and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)
