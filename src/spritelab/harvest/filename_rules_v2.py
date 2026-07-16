"""Source-aware filename metadata rules for label v2."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from spritelab.harvest.label_candidates import exact_sheet_object_for_record
from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.label_taxonomy import (
    canonicalize_object_name,
    normalize_category,
    normalize_object_name,
    normalize_tag,
    normalize_tags,
)
from spritelab.harvest.sheet_specializations import is_rpg_496_profile
from spritelab.harvest.source_profiles import SourceProfile, detect_source_profile


@dataclass(frozen=True)
class FilenameRuleResult:
    suggestion: LabelSuggestion
    profile: SourceProfile
    parsed_tokens: tuple[str, ...]
    raw_tokens: tuple[str, ...]
    confidence: float
    confidence_reason: str


@dataclass(frozen=True)
class _LexiconEntry:
    category: str
    tags: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    confidence: float = 0.95


_GENERIC_TOKENS = {
    "",
    "32",
    "32x32",
    "32fix",
    "496",
    "cc0",
    "fixed",
    "fix",
    "food",
    "icon",
    "icons",
    "kenney",
    "ocal",
    "oga",
    "opengameart",
    "pack",
    "png",
    "rpg",
    "sprite",
    "sprites",
    "tile",
    "tiles",
}

_GRID_TOKEN_RE = re.compile(r"^[rc]\d+$")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SOURCE_AUTHOR_TOKENS = {"arlantr", "arlan", "tr", "rcorre", "buch", "dcss", "bizmasterstudios", "kotnaszynce", "melle"}
_SOURCE_AUTHOR_COMPOUNDS = {
    "arlan_tr",
    "arlantr",
    "key_rcorre",
    "tool_dcss",
    "potion_bizmasterstudios",
    "potion_buch",
    "potion_kotnaszynce",
    "potion_melle",
    "potion_rcorre",
}
_COLOR_ATTRIBUTE_TOKENS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "violet",
    "pink",
    "white",
    "black",
    "gray",
    "grey",
    "brown",
    "gold",
    "golden",
    "silver",
    "bronze",
}

_FRUITS = {
    "apple",
    "banana",
    "blackberry",
    "blueberry",
    "cherry",
    "coconut",
    "grape",
    "kiwi",
    "lemon",
    "lime",
    "olive",
    "orange",
    "pear",
    "pineapple",
    "plum",
    "raspberry",
    "strawberry",
    "watermelon",
    "watermelon_slice",
}
_VEGETABLES = {
    "artichoke",
    "asparagus",
    "avocado",
    "beans",
    "broccoli",
    "cabbage",
    "carrot",
    "corn",
    "daikon",
    "eggplant",
    "mushroom",
    "pepper",
    "radish",
    "cherry_tomatoes",
    "tomato",
    "tomatoes_cherry",
}
_MEATS = {"bacon", "chicken", "chicken_drumstick", "fish", "ham", "hamburger", "meat", "steak"}
_DAIRY = {"butter", "camembert", "cheese", "cheese_wedge", "cheese_wheel", "milk", "milk_carton"}
_DESSERTS = {"cake", "candy", "cookie", "donut", "honey", "ice_cream", "ice_cream_cup", "ice_cream_sandwich"}
_DRINKS = {
    "juice",
    "orange_juice",
    "smoothie",
    "soda",
    "soda_bottle",
    "soda_can",
    "soda_glass",
    "strawberry_smoothie",
}
_GRAINS = {"baguette", "bread", "noodle", "noodle_macaroni", "pretzel", "rice", "sandwich", "sandwich_grilled"}
_PREPARED = {"burrito", "hamburger", "hot_dog", "pizza_slice", "sandwich", "sandwich_grilled", "sashimi", "sushi"}

_FOOD_OBJECTS = {
    "apple",
    "banana",
    "artichoke",
    "asparagus",
    "avocado",
    "bacon",
    "baguette",
    "beans",
    "blackberry",
    "blueberries",
    "blueberry",
    "bread",
    "broccoli",
    "burrito",
    "butter",
    "cabbage",
    "cake",
    "camembert",
    "candy",
    "carrot",
    "cheese",
    "cheese_wedge",
    "cheese_wheel",
    "cherries",
    "cherry",
    "cherry_tomato",
    "cherry_tomatoes",
    "chicken",
    "chicken_drumstick",
    "coconut",
    "cookie",
    "corn",
    "daikon",
    "donut",
    "egg",
    "eggplant",
    "fish",
    "grapes",
    "grape",
    "berry",
    "fruit",
    "fantasy_fruit",
    "ham",
    "hamburger",
    "honey",
    "hot_dog",
    "ice_cream",
    "ice_cream_cup",
    "ice_cream_sandwich",
    "juice",
    "orange_juice",
    "ketchup",
    "kiwi",
    "lemon",
    "lime",
    "meat",
    "milk",
    "milk_carton",
    "mushroom",
    "noodle",
    "noodle_macaroni",
    "olive",
    "orange",
    "pear",
    "pepper",
    "pineapple",
    "pizza_slice",
    "plum",
    "pretzel",
    "radish",
    "raspberries",
    "raspberry",
    "rice",
    "sandwich",
    "sandwich_grilled",
    "smoothie",
    "soda",
    "soda_bottle",
    "soda_can",
    "soda_glass",
    "sashimi",
    "strawberry",
    "strawberry_smoothie",
    "steak",
    "sushi",
    "tomato",
    "tomatoes_cherry",
    "watermelon",
    "watermelon_slice",
}

_TOOL_OBJECTS = {
    "compass",
    "compass_geometric",
    "ruler",
    "ruler_triangle",
    "scissors",
    "tape_measure",
    "meter",
    "case",
    "tool_case",
    "toolbox",
    "secateur",
    "wiresnips",
    "wiresnips_blue",
    "wiresnips_yellow",
    "hammer",
    "axe",
    "shovel",
    "pickaxe",
    "wrench",
    "saw",
    "screwdriver",
    "pliers",
    "shears",
    "knife",
    "torch",
    "rope",
    "clock",
    "telescope",
    "net",
    "throwing_net",
    "hoe",
    "bucket",
}

_JEWELRY_OBJECTS = {
    "ring",
    "necklace",
    "amulet",
    "bracelet",
    "crown",
    "jewel",
    "gem",
}

_KEY_OBJECTS = {
    "key",
    "golden_key",
    "silver_key",
    "rusty_key",
}

_GEM_OBJECTS = {
    "gem",
    "crystal",
    "agate",
    "jade",
    "opal",
    "amethyst",
    "ruby",
    "sapphire",
    "emerald",
    "diamond",
    "round_gem",
    "triangle_gem",
    "diamond_gem",
    "oval_gem",
    "mixed_gem",
    "blue_crystal",
    "red_crystal",
    "gray_crystal",
    "dark_blue_gem",
    "red_gem",
    "gray_gem",
    "ruby_gem",
    "sapphire_gem",
    "amethyst_gem",
    "emerald_gem",
}

_POTION_CONTAINER_OBJECTS = {
    "potion",
    "vial",
    "bottle",
    "flask",
    "jar",
    "soda_bottle",
    "soda_can",
    "soda_glass",
    "juice",
    "smoothie",
    "milk_carton",
    "mug",
    "cup",
    "glass",
}

_RPG_MAIN_TOKENS = {
    "antidote",
    "axe",
    "banana",
    "bottle",
    "bow",
    "carrot",
    "cheese",
    "cherry",
    "clock",
    "clover",
    "dagger",
    "grape",
    "grapes",
    "key",
    "lemon",
    "meat",
    "orange",
    "pepper",
    "pineapple",
    "poison",
    "radish",
    "shield",
    "sword",
    "watermelon",
}

_WEAPON_OBJECTS = {"axe", "bow", "dagger", "sword", "staff", "knife"}
_ARMOR_OBJECTS = {
    "armor",
    "shield",
    "helmet",
    "helm",
    "hat",
    "headgear",
    "shoes",
    "boots",
    "clothing",
    "chestplate",
    "breastplate",
    "leather_armor",
}
_EFFECT_OBJECTS = {"poison", "burn", "fire", "freeze", "ice", "heal", "sleep", "spell"}
_MATERIAL_OBJECTS = {
    "metal",
    "gold",
    "wood",
    "coal",
    "bone",
    "bones",
    "fabric",
    "crystal",
    "gem",
    "agate",
    "jade",
    "opal",
    "ruby",
    "sapphire",
    "amethyst",
    "diamond",
    "ore",
    "ingot",
    "bronze",
    "bronze_bar",
    "bronze_coin",
    "metal_coin",
}
_RPG_496_PREFIX_MATERIAL_OBJECTS = {
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


def suggest_from_filename_v2(record: Mapping[str, Any]) -> FilenameRuleResult:
    """Suggest a safe label from source-aware filename/path rules."""

    profile = detect_source_profile(record)
    raw_name = _record_filename(record)
    raw_tokens = _raw_tokens(raw_name)
    parsed_tokens = _semantic_tokens(raw_tokens)
    exact_object = exact_sheet_object_for_record(record, profile)

    if exact_object:
        suggestion = _suggestion_for_object(
            exact_object,
            profile,
            0.96,
            f"known sheet-cell map for source profile {profile.name}",
            alias_used=False,
        )
    elif is_rpg_496_profile(profile):
        suggestion, confidence, reason = _suggest_rpg_icon(parsed_tokens, profile)
    else:
        object_name, confidence, reason, alias_used = _choose_object(parsed_tokens, profile)
        suggestion = _suggestion_for_object(object_name, profile, confidence, reason, alias_used=alias_used)
    suggestion = _apply_filename_attributes(suggestion, parsed_tokens)

    return FilenameRuleResult(
        suggestion=suggestion,
        profile=profile,
        parsed_tokens=parsed_tokens,
        raw_tokens=raw_tokens,
        confidence=suggestion.confidence,
        confidence_reason=suggestion.confidence_reason,
    )


def filename_rule_result_to_json(result: FilenameRuleResult) -> dict[str, Any]:
    from spritelab.harvest.label_schema import label_suggestion_to_json
    from spritelab.harvest.source_profiles import source_profile_to_json

    return {
        "suggestion": label_suggestion_to_json(result.suggestion),
        "profile": source_profile_to_json(result.profile),
        "parsed_tokens": list(result.parsed_tokens),
        "raw_tokens": list(result.raw_tokens),
        "confidence": result.confidence,
        "confidence_reason": result.confidence_reason,
    }


def _record_filename(record: Mapping[str, Any]) -> str:
    for key in ("relative_path", "final_png_path", "filename", "sprite_id"):
        value = str(record.get(key, "")).strip()
        if value:
            return Path(value).name
    return ""


def _raw_tokens(value: str | Path) -> tuple[str, ...]:
    stem = Path(str(value)).stem
    stem = _CAMEL_BOUNDARY_RE.sub("_", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem)
    tokens: list[str] = []
    for raw in stem.split("_"):
        token = normalize_tag(raw)
        if _GRID_TOKEN_RE.fullmatch(token):
            tokens.append(token)
            continue
        token = re.sub(r"(\D)\d+$", r"\1", token)
        token = normalize_tag(token)
        if token:
            tokens.append(token)
    return tuple(tokens)


def _semantic_tokens(raw_tokens: tuple[str, ...]) -> tuple[str, ...]:
    tokens = []
    for token in raw_tokens:
        if token in _GENERIC_TOKENS or token.isdigit() or _GRID_TOKEN_RE.fullmatch(token):
            continue
        tokens.append(token)
    return tuple(tokens)


def _choose_object(tokens: tuple[str, ...], profile: SourceProfile) -> tuple[str, float, str, bool]:
    potion_fallback = _potion_family_object(tokens, profile)
    if potion_fallback:
        object_name, reason = potion_fallback
        return object_name, 0.92, reason, object_name not in tokens
    if profile.name == "mushroom" and "mushroom" in tokens:
        return "mushroom", 0.95, "known mushroom source profile with mushroom object token", False
    if not tokens:
        return "", 0.25, "grid cell or generic sheet name with no object clue", False
    if _contains_source_author_artifact(tokens):
        object_name = canonicalize_object_name("_".join(tokens))
        return object_name, 0.25, f"filename token {object_name!r} looks like a source/author artifact", False

    special = _special_food_container(tokens)
    if special:
        object_name, alias_used = special
        confidence = _known_confidence(object_name, profile, alias_used=alias_used)
        return object_name, confidence, _confidence_reason(object_name, profile, alias_used), alias_used

    canonical_tokens = tuple(canonicalize_object_name(token) for token in tokens if token)
    joined = canonicalize_object_name("_".join(canonical_tokens))
    alias = _alias_object(joined)
    if alias:
        confidence = _known_confidence(alias, profile, alias_used=alias != joined)
        return alias, confidence, _confidence_reason(alias, profile, alias != joined), alias != joined
    if _is_known_object(joined, profile):
        confidence = _known_confidence(joined, profile, alias_used=False)
        return joined, confidence, _confidence_reason(joined, profile, False), False

    for size in range(len(canonical_tokens), 0, -1):
        for start in range(0, len(canonical_tokens) - size + 1):
            candidate = canonicalize_object_name("_".join(canonical_tokens[start : start + size]))
            alias = _alias_object(candidate)
            if alias:
                confidence = _known_confidence(alias, profile, alias_used=alias != candidate)
                return alias, confidence, _confidence_reason(alias, profile, alias != candidate), alias != candidate
            if _is_known_object(candidate, profile):
                confidence = _known_confidence(candidate, profile, alias_used=False)
                return candidate, confidence, _confidence_reason(candidate, profile, False), False

    if profile.name == "cc0_gem" and len(canonical_tokens) >= 2 and canonical_tokens[-1] == "gem":
        object_name = canonicalize_object_name("_".join(canonical_tokens))
        return object_name, 0.85, f"gem-profile alias/synonym object {object_name!r}", True

    object_name = joined
    category_known = bool(profile.expected_category_bias and profile.expected_category_bias != ("unknown",))
    confidence = 0.75 if category_known and profile.trusted_filename else 0.55
    reason = (
        f"recognized source category bias for {profile.name}; object token {object_name!r} is not in the lexicon"
        if category_known and profile.trusted_filename
        else f"object token {object_name!r} inferred from filename but not recognized"
    )
    return object_name, confidence, reason, False


def _special_food_container(tokens: tuple[str, ...]) -> tuple[str, bool] | None:
    normalized = tuple(canonicalize_object_name(token) for token in tokens)
    if len(normalized) >= 2 and normalized[0] == "juice" and normalized[1] in _FRUITS:
        return f"{normalized[1]}_juice", True
    if len(normalized) >= 3 and normalized[0] == "soda" and normalized[1] in {"can", "bottle", "glass"}:
        flavor = normalized[2]
        if flavor:
            return f"soda_{normalized[1]}_{flavor}", True
    if len(normalized) >= 2 and normalized[-1] == "smoothie":
        return canonicalize_object_name("_".join(normalized)), False
    return None


def _potion_family_object(tokens: tuple[str, ...], profile: SourceProfile) -> tuple[str, str] | None:
    if profile.name != "cc0_potion":
        return None
    normalized = tuple(canonicalize_object_name(token) for token in tokens if token)
    object_tokens = tuple(
        token
        for token in normalized
        if token
        and token not in _SOURCE_AUTHOR_TOKENS
        and token not in _COLOR_ATTRIBUTE_TOKENS
        and token not in _GENERIC_TOKENS
        and token not in {"r", "c"}
    )
    for token in object_tokens:
        if token in {"potion", "vial", "bottle", "flask"}:
            return token, f"known potion-only source profile with clear object token {token!r}"
    if not object_tokens or all(_contains_source_author_artifact((token,)) for token in object_tokens):
        return "potion", "known potion-only source profile fallback for coordinate/source-author filename"
    if "potion" in normalized:
        return "potion", "known potion-only source profile with potion token"
    return None


def _alias_object(name: str) -> str:
    aliases = {
        "blueberries": "blueberry",
        "blackberries": "blackberry",
        "raspberries": "raspberry",
        "cherries": "cherry",
        "grapes": "grape",
        "tomatoes": "tomato",
        "tomato_cherry": "cherry_tomatoes",
        "tomatoes_cherry": "cherry_tomatoes",
        "cherry_tomato": "cherry_tomatoes",
        "cherry_tomatoes": "cherry_tomatoes",
        "juice_orange": "orange_juice",
        "scissor": "scissors",
        "wire_snips": "wiresnips",
        "wiresnip": "wiresnips",
        "wiresnip_blue": "wiresnips_blue",
        "wiresnip_yellow": "wiresnips_yellow",
        "case": "tool_case",
        "triangle_ruler": "ruler_triangle",
        "triangular_ruler": "ruler_triangle",
        "wire_snip": "wiresnips",
        "secateurs": "secateur",
        "amethist": "amethyst",
        "amethist_gem": "amethyst_gem",
        "saphire_gem": "sapphire_gem",
        "saphire": "sapphire",
        "ovale_gem": "oval_gem",
        "ovale": "oval",
    }
    if name in aliases:
        return aliases[name]
    if name.endswith("_gem"):
        prefix = name[: -len("_gem")]
        if prefix in {"ruby", "sapphire", "emerald", "diamond"}:
            return name
    return ""


def _is_known_object(name: str, profile: SourceProfile) -> bool:
    if name in _FOOD_OBJECTS or name in {canonicalize_object_name(value) for value in _FOOD_OBJECTS}:
        return True
    if name in _TOOL_OBJECTS or name in _GEM_OBJECTS or name in _POTION_CONTAINER_OBJECTS:
        return True
    if name in _JEWELRY_OBJECTS or name in _KEY_OBJECTS:
        return True
    if is_rpg_496_profile(profile) and name in _RPG_MAIN_TOKENS:
        return True
    return False


def _known_confidence(object_name: str, profile: SourceProfile, *, alias_used: bool) -> float:
    if alias_used:
        return 0.85
    if not profile.trusted_filename:
        return 0.75
    if "_" in object_name:
        return 0.9
    return 0.95


def _confidence_reason(object_name: str, profile: SourceProfile, alias_used: bool) -> str:
    if alias_used:
        return f"recognized alias/synonym object {object_name!r} in source profile {profile.name}"
    return f"recognized exact object {object_name!r} in source profile {profile.name}"


def _suggestion_for_object(
    object_name: str,
    profile: SourceProfile,
    confidence: float,
    confidence_reason: str,
    *,
    alias_used: bool,
) -> LabelSuggestion:
    category = _category_for_object(object_name, profile)
    tags = _tags_for_object(object_name, profile)
    if object_name and object_name not in tags:
        tags = (object_name, *tags)
    description = _description(object_name, category)
    return LabelSuggestion(
        category=category,
        object_name=object_name,
        tags=tags,
        short_description=description,
        confidence=confidence,
        confidence_reason=confidence_reason,
        source="filename_rules_v2",
        evidence=(f"profile:{profile.name}", f"filename_object:{object_name}")
        if object_name
        else (f"profile:{profile.name}",),
    )


def _category_for_object(object_name: str, profile: SourceProfile) -> str:
    if not object_name:
        return "unknown"
    if profile.name == "cc0_tool" or object_name in _TOOL_OBJECTS:
        return "tool"
    if profile.name == "cc0_gem" or object_name in _GEM_OBJECTS:
        return "material"
    if profile.name in {"cc0_jewelry", "cc0_key"} or object_name in _JEWELRY_OBJECTS or object_name in _KEY_OBJECTS:
        return "item_icon"
    if profile.name == "mushroom":
        return "plant"
    if profile.name == "cc0_potion":
        return "item_icon"
    if object_name in _WEAPON_OBJECTS:
        return "weapon"
    if object_name in _ARMOR_OBJECTS:
        return "armor"
    if object_name in _EFFECT_OBJECTS:
        return "effect_icon"
    if object_name in _MATERIAL_OBJECTS:
        return "material"
    if profile.expected_category_bias and profile.expected_category_bias[0] != "unknown":
        return normalize_category(profile.expected_category_bias[0])
    return "item_icon"


def _tags_for_object(object_name: str, profile: SourceProfile) -> tuple[str, ...]:
    tags: list[str] = []
    base = normalize_object_name(object_name)
    if base:
        tags.append(base)
    if _is_food_object(base):
        tags.extend(["food", "consumable"])
        if base in _FRUITS:
            tags.extend(["fruit", "plant_based"])
        if base in {"orange", "lemon", "lime", "orange_juice"}:
            tags.append("citrus")
        if base == "lemon":
            tags.append("sour")
        if base == "kiwi":
            tags.append("green")
        if base in _VEGETABLES:
            tags.extend(["vegetable", "plant_based"])
        if base in _MEATS:
            tags.extend(["meat", "animal_product", "savory"])
        if base in _DAIRY:
            tags.extend(["dairy", "animal_product"])
        if base == "butter":
            tags.extend(["fat", "ingredient"])
        if base.startswith("cheese"):
            tags.append("cheese")
        if base == "cheese_wedge":
            tags.append("wedge")
        if base == "cheese_wheel":
            tags.append("wheel")
        if base in _DESSERTS:
            tags.extend(["dessert", "sweet"])
        if base in _DRINKS or base in {"juice", "smoothie", "milk_carton"}:
            tags.extend(["drink", "beverage"])
        if base in _GRAINS:
            tags.append("grain")
        if base in {"noodle", "noodle_macaroni"}:
            tags.append("pasta")
        if base in _PREPARED:
            tags.append("prepared_food")
        if base == "sushi":
            tags.append("sushi")
        if base == "sashimi":
            tags.append("fish")
        if base in {"milk_carton", "soda_can", "soda_bottle", "soda_glass"} or "soda_can" in base:
            tags.append("container")
            if base == "milk_carton":
                tags.extend(["milk", "carton"])
            if "soda_can" in base:
                tags.extend(["soda_can", "soda", "drink", "beverage", "container"])
                flavor = base.removeprefix("soda_can_")
                if flavor and flavor != base:
                    tags.append(flavor)
    elif base in _TOOL_OBJECTS or profile.name == "cc0_tool":
        tags.extend(["tool", "utility"])
        if base in {"compass", "compass_geometric"}:
            tags.extend(["navigation", "measuring_tool", "drafting"])
        if base in {"ruler", "ruler_triangle"}:
            tags.extend(["measuring_tool", "drafting", "geometry"])
        if base == "ruler_triangle":
            tags.append("triangle")
        if base in {"tape_measure"}:
            tags.extend(["measuring_tool", "construction"])
        if base == "meter":
            tags.append("measuring_tool")
        if base in {"scissors"}:
            tags.extend(["cutting_tool", "metal", "handle"])
        if base in {"wiresnips", "wiresnips_blue", "wiresnips_yellow"}:
            tags.extend(["cutting_tool", "wire", "electrical"])
        if base in {"toolbox", "tool_case"}:
            tags.extend(["storage", "container", "tools"])
        if base in {"ruler", "ruler_triangle", "tape_measure", "meter"}:
            tags.extend(["measuring_tool", "geometry", "drafting"])
        if base in {"scissors", "secateur", "wiresnips", "saw", "knife", "pliers", "shears"}:
            tags.append("cutting_tool")
        if base in {"hammer", "axe", "shovel", "pickaxe", "wrench", "screwdriver"}:
            tags.append("construction")
        if base in {"torch", "rope", "clock", "telescope", "net", "throwing_net", "bucket"}:
            tags.append("utility")
        if base == "secateur":
            tags.extend(["shears", "gardening", "plant_tool"])
    elif base in _JEWELRY_OBJECTS or profile.name == "cc0_jewelry":
        tags.extend(["jewelry", "accessory"])
        if base in {"ring", "necklace", "amulet", "bracelet", "crown"}:
            tags.append("wearable")
        if base in {"jewel", "gem"}:
            tags.extend(["gem", "treasure"])
    elif base in _KEY_OBJECTS or profile.name == "cc0_key":
        tags.extend(["key", "utility", "unlock"])
        if base.startswith("golden"):
            tags.extend(["gold", "metal"])
        if base.startswith("silver"):
            tags.extend(["silver", "metal"])
        if base.startswith("rusty"):
            tags.extend(["rusty", "metal"])
    elif base in _GEM_OBJECTS or profile.name == "cc0_gem":
        tags.extend(["gem", "gemstone", "mineral", "treasure", "crafting_material"])
        if "crystal" in base:
            tags.extend(["crystal", "raw_gem"])
        else:
            tags.extend(["polished_gem", "faceted", "precious_stone"])
        for color in ("blue", "red", "gray", "dark_blue"):
            if base.startswith(f"{color}_"):
                tags.append(color)
        for shape in ("round", "triangle", "diamond", "oval", "mixed"):
            if base.startswith(f"{shape}_"):
                tags.append(shape)
    elif base in _POTION_CONTAINER_OBJECTS or profile.name == "cc0_potion":
        tags.extend(["potion", "liquid", "container", "consumable"])
        if base in {"bottle", "vial", "flask", "jar"}:
            tags.append(base)
    elif base in _ARMOR_OBJECTS:
        tags.extend(["armor", "wearable", "defense"])
        if base in {"helmet", "helm", "hat", "headgear"}:
            tags.append("headgear")
        if base in {"shoes", "boots"}:
            tags.append("footwear")
        if base == "shield":
            tags.append("shield")
    else:
        if profile.domain and profile.domain != "unknown":
            tags.append(profile.domain)
    return normalize_tags(tags)


def _is_food_object(object_name: str) -> bool:
    return object_name in {canonicalize_object_name(value) for value in _FOOD_OBJECTS} or object_name.startswith(
        "soda_can_"
    )


def _contains_source_author_artifact(tokens: tuple[str, ...]) -> bool:
    normalized = tuple(canonicalize_object_name(token) for token in tokens if token)
    joined = canonicalize_object_name("_".join(normalized))
    if joined in _SOURCE_AUTHOR_COMPOUNDS:
        return True
    return any(token in _SOURCE_AUTHOR_TOKENS for token in normalized)


def _apply_filename_attributes(suggestion: LabelSuggestion, tokens: tuple[str, ...]) -> LabelSuggestion:
    colors = [
        "purple" if token == "violet" else "gray" if token == "grey" else "gold" if token == "golden" else token
        for token in (canonicalize_object_name(value) for value in tokens)
        if token in _COLOR_ATTRIBUTE_TOKENS
    ]
    if not colors:
        return suggestion
    return replace(
        suggestion,
        tags=normalize_tags((*suggestion.tags, *colors)),
        dominant_colors=normalize_tags((*suggestion.dominant_colors, *colors)),
        evidence=(*suggestion.evidence, *(f"filename_attribute:{color}" for color in colors)),
    )


def _description(object_name: str, category: str) -> str:
    if not object_name:
        return "A 32x32 pixel-art sprite."
    return f"A 32x32 pixel-art {object_name.replace('_', ' ')} {category.replace('_', ' ')}."


def _suggest_rpg_icon(tokens: tuple[str, ...], profile: SourceProfile) -> tuple[LabelSuggestion, float, str]:
    if not tokens:
        suggestion = _suggestion_for_object(
            "", profile, 0.25, "RPG icon filename has no semantic token", alias_used=False
        )
        return suggestion, suggestion.confidence, suggestion.confidence_reason

    prefix = tokens[0]
    subtype = tokens[1] if len(tokens) > 1 and len(tokens[1]) == 1 else ""
    start = 1 + (1 if subtype else 0)
    semantic = tokens[start:] or tokens[1:] or tokens
    object_name = _rpg_object_from_tokens(semantic)
    if prefix == "c" and object_name in {"elm", "helm"}:
        object_name = "helmet"

    category = {
        "w": "weapon",
        "s": "effect_icon",
        "p": "item_icon",
        "ac": "item_icon",
        "a": "armor",
        "c": "armor",
        "e": "material",
        "i": "item_icon",
    }.get(prefix, "item_icon")
    if prefix == "i" and subtype == "c":
        category = "item_icon"
    elif prefix in {"i", "e"} and object_name in _RPG_496_PREFIX_MATERIAL_OBJECTS:
        category = "material"
    if prefix == "a" and object_name in {"shield"}:
        category = "armor"
    if prefix == "w" and object_name in _TOOL_OBJECTS - _WEAPON_OBJECTS:
        category = "tool"

    tags = list(_tags_for_object(object_name, profile))
    if prefix == "w":
        tags.extend(["weapon"])
    if prefix in {"a", "c"}:
        tags.extend(["armor", "wearable"])
    if prefix == "c":
        tags.append("headgear")
    if prefix == "ac":
        tags.extend(["accessory", "jewelry"])
    if prefix == "i" and subtype == "c":
        tags.extend(["food", "consumable"])
    if prefix == "p":
        tags.extend(["potion", "liquid"])
    if prefix == "s":
        tags.extend(["effect", "status_effect"])
    if prefix == "e":
        tags.extend(["material", "crafting_material"])

    confidence = 0.95 if object_name in _RPG_MAIN_TOKENS or object_name in _WEAPON_OBJECTS | _EFFECT_OBJECTS else 0.75
    reason = f"recognized RPG icon prefix {prefix!r}"
    if object_name:
        reason += f" and object token {object_name!r}"
    suggestion = LabelSuggestion(
        category=category,
        object_name=object_name,
        tags=normalize_tags(tags),
        short_description=_description(object_name, category),
        confidence=confidence,
        confidence_reason=reason,
        source="filename_rules_v2",
        evidence=(f"profile:{profile.name}", f"rpg_prefix:{prefix}", f"filename_object:{object_name}"),
    )
    return suggestion, confidence, reason


def _rpg_object_from_tokens(tokens: tuple[str, ...]) -> str:
    canonical_tokens = tuple(canonicalize_object_name(token) for token in tokens if token)
    joined = canonicalize_object_name("_".join(canonical_tokens))
    alias = _alias_object(joined)
    if alias:
        return alias
    if joined in _MATERIAL_OBJECTS:
        return joined
    if joined in _RPG_MAIN_TOKENS or _is_known_object(
        joined, detect_source_profile({"source_id": "oga_496_rpg_icons"})
    ):
        return joined
    for token in canonical_tokens:
        if (
            token in _RPG_MAIN_TOKENS
            or token in _WEAPON_OBJECTS
            or token in _EFFECT_OBJECTS
            or token in _MATERIAL_OBJECTS
        ):
            return token
    return joined
