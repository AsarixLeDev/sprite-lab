"""Central label taxonomy and normalization helpers for harvest labels."""

from __future__ import annotations

import re
from collections.abc import Iterable

CATEGORY_VALUES = (
    "unknown",
    "item_icon",
    "block",
    "plant",
    "ui_icon",
    "entity",
    "character",
    "weapon",
    "tool",
    "armor",
    "material",
    "effect_icon",
    "environment_prop",
)

FOOD_TAGS = frozenset(
    {
        "food",
        "consumable",
        "fruit",
        "vegetable",
        "meat",
        "dairy",
        "dessert",
        "drink",
        "beverage",
        "grain",
        "pasta",
        "sushi",
        "prepared_food",
        "ingredient",
        "sweet",
        "savory",
        "tropical",
        "plant_based",
        "animal_product",
    }
)
TOOL_TAGS = frozenset(
    {
        "tool",
        "utility",
        "measuring_tool",
        "cutting_tool",
        "crafting_tool",
        "navigation",
        "construction",
        "metal",
        "wood",
        "handle",
    }
)
GEM_MATERIAL_TAGS = frozenset(
    {
        "gem",
        "gemstone",
        "mineral",
        "treasure",
        "crafting_material",
        "crystal",
        "polished_gem",
        "raw_gem",
        "faceted",
        "precious_stone",
    }
)
POTION_CONTAINER_TAGS = frozenset(
    {"potion", "vial", "bottle", "flask", "jar", "liquid", "container", "drink", "beverage"}
)
WEAPON_TAGS = frozenset({"weapon", "melee", "ranged", "blade", "metal", "wood", "handle"})
ARMOR_TAGS = frozenset({"armor", "shield", "helmet", "clothing", "defense", "wearable"})
UI_EFFECT_TAGS = frozenset({"ui", "icon", "effect", "status_effect", "spell", "buff", "debuff"})
ENVIRONMENT_TILE_TAGS = frozenset({"environment", "tile", "terrain", "prop", "wall", "floor"})

_TOKEN_SEPARATORS_RE = re.compile(r"[\s/\\\-.:;]+")
_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_TYPO_REPLACEMENTS = {
    "amethist": "amethyst",
    "saphire": "sapphire",
    "ovale": "oval",
    "armour": "armor",
    "ambiguou": "ambiguous",
    "watermellon": "watermelon",
}

_PLURAL_REPLACEMENTS = {
    "cherries": "cherry",
    "blueberries": "blueberry",
    "blackberries": "blackberry",
    "raspberries": "raspberry",
    "strawberries": "strawberry",
    "grapes": "grape",
    "apples": "apple",
    "bananas": "banana",
    "carrots": "carrot",
    "lemons": "lemon",
    "oranges": "orange",
    "peppers": "pepper",
    "radishes": "radish",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "heroes": "hero",
    "echoes": "echo",
    "mangoes": "mango",
    "berries": "berry",
    "keys": "key",
    "potions": "potion",
    "vials": "vial",
    "bottles": "bottle",
    "gems": "gem",
    "crystals": "crystal",
    "coins": "coin",
    "shoes": "shoes",
    "boots": "boots",
    "scissors": "scissors",
    "wiresnips": "wiresnips",
    "secateurs": "secateur",
    "pliers": "pliers",
    "shears": "shears",
}

_COMPOUND_REPLACEMENTS = {
    "case": "tool_case",
    "scissor": "scissors",
    "wire_snip": "wiresnips",
    "wiresnip": "wiresnips",
    "wire_snip_blue": "wiresnips_blue",
    "wiresnip_blue": "wiresnips_blue",
    "wire_snip_yellow": "wiresnips_yellow",
    "wiresnip_yellow": "wiresnips_yellow",
    "tomato_cherry": "cherry_tomatoes",
    "tomato_cherries": "cherry_tomatoes",
    "cherry_tomato": "cherry_tomatoes",
    "cherry_tomatoes": "cherry_tomatoes",
    "juice_orange": "orange_juice",
    "unidentified": "unknown",
    "unidentified_object": "unknown",
    "unknown_object": "unknown",
}


def normalize_category(value: str) -> str:
    """Normalize a category token and reject values outside the taxonomy."""

    category = normalize_tag(value)
    return category if category in CATEGORY_VALUES else "unknown"


def normalize_object_name(value: str) -> str:
    """Normalize an object name to canonical snake_case."""

    return canonicalize_object_name(value)


def normalize_tag(value: str) -> str:
    """Normalize a free-form token to lowercase snake_case."""

    text = _CAMEL_BOUNDARY_RE.sub("_", str(value).strip())
    text = text.replace("&", " and ")
    text = _TOKEN_SEPARATORS_RE.sub("_", text.lower())
    text = _SAFE_TOKEN_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return ""
    parts = [_TYPO_REPLACEMENTS.get(part, part) for part in text.split("_") if part]
    parts = [_PLURAL_REPLACEMENTS.get(part, part) for part in parts]
    return "_".join(part for part in parts if part)


def normalize_tags(tags: Iterable[str]) -> tuple[str, ...]:
    """Normalize, dedupe, and keep first-seen tag order."""

    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        normalized = normalize_tag(str(tag))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def split_object_tokens(name: str) -> tuple[str, ...]:
    """Return canonical object-name tokens."""

    normalized = normalize_tag(name)
    if not normalized:
        return ()
    tokens = []
    for token in normalized.split("_"):
        token = singularize_basic(_TYPO_REPLACEMENTS.get(token, token))
        if token:
            tokens.append(token)
    return tuple(tokens)


def singularize_basic(token: str) -> str:
    """Handle the limited plural forms common in sprite filenames."""

    normalized = normalize_tag(token)
    if normalized in _PLURAL_REPLACEMENTS:
        return _PLURAL_REPLACEMENTS[normalized]
    if normalized.endswith(("ous", "us", "ss")):
        return normalized
    return normalized


def canonicalize_object_name(name: str) -> str:
    """Normalize typo variants and basic plurals in a compound object name."""

    joined = "_".join(split_object_tokens(name))
    return _COMPOUND_REPLACEMENTS.get(joined, joined)


def object_name_token_f1(a: str, b: str) -> float:
    """Token-level F1 between two normalized object names."""

    a_tokens = set(split_object_tokens(a))
    b_tokens = set(split_object_tokens(b))
    return _set_f1(a_tokens, b_tokens)


def tag_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    """F1-style overlap score between two tag sets."""

    a_tags = set(normalize_tags(a))
    b_tags = set(normalize_tags(b))
    return _set_f1(a_tags, b_tags)


def _set_f1(a_values: set[str], b_values: set[str]) -> float:
    if not a_values and not b_values:
        return 1.0
    if not a_values or not b_values:
        return 0.0
    overlap = len(a_values & b_values)
    if overlap == 0:
        return 0.0
    precision = overlap / len(b_values)
    recall = overlap / len(a_values)
    return 2 * precision * recall / (precision + recall)
