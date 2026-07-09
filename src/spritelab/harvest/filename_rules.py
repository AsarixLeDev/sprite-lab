"""Deterministic filename-derived metadata suggestions for harvested sprites."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.dataset_maker.model import normalize_category, normalize_sprite_id, normalize_tag


@dataclass(frozen=True)
class FilenameMetadataSuggestion:
    category: str = "unknown"
    object_name: str = ""
    tags: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    short_description: str = ""
    confidence: float = 0.0
    confidence_reason: str = ""
    source: str = "filename_rules"

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_category(self.category))
        object.__setattr__(self, "object_name", normalize_tag(self.object_name))
        object.__setattr__(self, "tags", _dedupe_normalized(self.tags))
        object.__setattr__(self, "materials", _dedupe_normalized(self.materials))
        object.__setattr__(self, "mood", _dedupe_normalized(self.mood))
        object.__setattr__(self, "short_description", str(self.short_description).strip())
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "confidence_reason", str(self.confidence_reason).strip())
        object.__setattr__(self, "source", "filename_rules")


@dataclass(frozen=True)
class _ObjectRule:
    category: str
    tags: tuple[str, ...]
    materials: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    confidence: float = 0.95


_CATEGORY_CODES = {
    "ac": "item_icon",
    "i": "item_icon",
    "s": "effect_icon",
    "w": "weapon",
    "a": "armor",
    "b": "block",
    "p": "plant",
    "u": "ui_icon",
    "e": "entity",
    "m": "material",
}

_SUBTYPE_TAGS = {
    ("i", "c"): ("consumable",),
    ("i", "m"): ("material",),
    ("i", "q"): ("quest_item",),
    ("s", "p"): ("status_effect",),
}

_CATEGORY_CODE_TAGS = {
    "ac": ("accessory", "jewelry"),
}

_STRUCTURAL_PREFIX_CODES = {
    "ac",
    "c",
}

_GENERIC_TOKENS = {
    "oga",
    "opengameart",
    "rpg",
    "icons",
    "icon",
    "sprite",
    "sprites",
    "fix",
    "32fix",
    "32x32",
    "png",
}

_OBJECT_RULES: dict[str, _ObjectRule] = {
    "apple": _ObjectRule("item_icon", ("apple", "fruit", "food", "consumable"), confidence=0.96),
    "banana": _ObjectRule("item_icon", ("banana", "fruit", "food", "consumable"), confidence=0.98),
    "bread": _ObjectRule("item_icon", ("bread", "food", "consumable"), confidence=0.96),
    "fish": _ObjectRule("item_icon", ("fish", "food", "consumable"), confidence=0.94),
    "meat": _ObjectRule("item_icon", ("meat", "food", "consumable"), confidence=0.94),
    "potion": _ObjectRule(
        "item_icon", ("potion", "vial", "liquid", "consumable", "magic"), ("glass",), confidence=0.94
    ),
    "vial": _ObjectRule("item_icon", ("vial", "liquid", "consumable"), ("glass",), confidence=0.92),
    "bottle": _ObjectRule("item_icon", ("bottle", "liquid", "consumable"), ("glass",), confidence=0.9),
    "coin": _ObjectRule("item_icon", ("coin", "currency", "treasure"), ("metal",), confidence=0.94),
    "medal": _ObjectRule("item_icon", ("medal", "accessory", "jewelry", "award"), ("metal",), confidence=0.95),
    "necklace": _ObjectRule("item_icon", ("necklace", "accessory", "jewelry"), ("metal",), confidence=0.95),
    "ring": _ObjectRule("item_icon", ("ring", "accessory", "jewelry"), ("metal",), confidence=0.95),
    "amulet": _ObjectRule(
        "item_icon", ("amulet", "accessory", "jewelry", "magic"), ("metal",), ("mystical",), confidence=0.93
    ),
    "crystal": _ObjectRule("item_icon", ("crystal", "gem", "magic"), ("crystal",), ("mystical",), confidence=0.92),
    "gem": _ObjectRule("item_icon", ("gem", "crystal", "treasure"), ("crystal",), confidence=0.92),
    "key": _ObjectRule("item_icon", ("key", "lock", "quest_item"), ("metal",), confidence=0.92),
    "scroll": _ObjectRule("item_icon", ("scroll", "paper", "magic"), ("paper",), confidence=0.9),
    "mushroom": _ObjectRule("plant", ("mushroom", "fungus", "organic"), confidence=0.9),
    "flower": _ObjectRule("plant", ("flower", "plant", "organic"), confidence=0.9),
    "leaf": _ObjectRule("plant", ("leaf", "plant", "organic"), confidence=0.9),
    "herb": _ObjectRule("plant", ("herb", "plant", "organic"), confidence=0.9),
    "poison": _ObjectRule("effect_icon", ("poison", "status_effect", "debuff", "magic"), confidence=0.95),
    "burn": _ObjectRule("effect_icon", ("burn", "status_effect", "debuff", "fire"), mood=("fiery",), confidence=0.9),
    "fire": _ObjectRule("effect_icon", ("fire", "magic", "element"), mood=("fiery",), confidence=0.88),
    "freeze": _ObjectRule("effect_icon", ("freeze", "status_effect", "debuff", "ice"), mood=("cold",), confidence=0.9),
    "ice": _ObjectRule("effect_icon", ("ice", "magic", "element"), mood=("cold",), confidence=0.88),
    "heal": _ObjectRule("effect_icon", ("heal", "status_effect", "buff", "magic"), confidence=0.9),
    "sleep": _ObjectRule("effect_icon", ("sleep", "status_effect", "debuff"), confidence=0.9),
    "axe": _ObjectRule("weapon", ("axe", "weapon", "tool", "melee"), ("metal", "wood"), confidence=0.95),
    "sword": _ObjectRule("weapon", ("sword", "weapon", "melee"), ("metal",), confidence=0.95),
    "dagger": _ObjectRule("weapon", ("dagger", "weapon", "melee"), ("metal",), confidence=0.95),
    "bow": _ObjectRule("weapon", ("bow", "weapon", "ranged"), ("wood",), confidence=0.95),
    "staff": _ObjectRule("weapon", ("staff", "weapon", "magic", "melee"), ("wood",), ("mystical",), confidence=0.92),
    "hammer": _ObjectRule("weapon", ("hammer", "weapon", "tool", "melee"), ("metal", "wood"), confidence=0.93),
    "pickaxe": _ObjectRule("tool", ("pickaxe", "tool", "mining"), ("metal", "wood"), confidence=0.93),
    "shield": _ObjectRule("armor", ("shield", "armor", "defense"), ("metal", "wood"), confidence=0.92),
    "helmet": _ObjectRule("armor", ("helmet", "armor", "headgear"), ("metal",), confidence=0.92),
    "hat": _ObjectRule("armor", ("hat", "clothing", "headgear"), confidence=0.9),
}

_CATEGORY_DESCRIPTORS = {
    "item_icon": "item icon",
    "effect_icon": "status/effect icon",
    "weapon": "weapon icon",
    "tool": "tool icon",
    "armor": "armor icon",
    "plant": "plant icon",
    "block": "block sprite",
    "ui_icon": "UI icon",
    "entity": "entity sprite",
    "material": "material icon",
}


def parse_filename_metadata(
    sprite_id: str,
    *,
    filename: str | Path | None = None,
) -> FilenameMetadataSuggestion:
    """Suggest metadata from filename/sprite_id codes and object tokens."""

    raw_name = Path(filename).name if filename is not None else sprite_id
    tokens = _tokens(raw_name)
    if not tokens and sprite_id:
        tokens = _tokens(sprite_id)

    category_code, subtype_code, semantic_tokens = _split_codes(tokens)
    if not semantic_tokens and filename is not None:
        category_code, subtype_code, semantic_tokens = _split_codes(_tokens(sprite_id))

    object_name = _object_name(semantic_tokens)
    rule = _OBJECT_RULES.get(object_name)
    category = _category_for(category_code, rule, object_name)
    tags = list(rule.tags if rule else ())
    materials = list(rule.materials if rule else ())
    mood = list(rule.mood if rule else ())

    if category_code and subtype_code:
        tags.extend(_SUBTYPE_TAGS.get((category_code, subtype_code), ()))
    if category_code:
        tags.extend(_CATEGORY_CODE_TAGS.get(category_code, ()))
    if object_name and object_name not in tags:
        tags.insert(0, object_name)
    if category in {"weapon", "tool", "armor"} and category not in tags:
        tags.append(category)
    if category == "effect_icon" and "status_effect" not in tags and object_name:
        tags.append("status_effect")

    confidence = _confidence(category_code, object_name, rule)
    confidence_reason = _confidence_reason(category_code, object_name, rule)
    description = _description(object_name, category)
    return FilenameMetadataSuggestion(
        category=category,
        object_name=object_name,
        tags=tuple(tags),
        materials=tuple(materials),
        mood=tuple(mood),
        short_description=description,
        confidence=confidence,
        confidence_reason=confidence_reason,
    )


def filename_suggestion_to_dict(suggestion: FilenameMetadataSuggestion) -> dict[str, Any]:
    return {
        "category": suggestion.category,
        "object_name": suggestion.object_name,
        "tags": list(suggestion.tags),
        "materials": list(suggestion.materials),
        "mood": list(suggestion.mood),
        "short_description": suggestion.short_description,
        "confidence": suggestion.confidence,
        "confidence_reason": suggestion.confidence_reason,
        "source": suggestion.source,
    }


def metadata_suggestions_differ(
    filename_suggestion: FilenameMetadataSuggestion,
    qwen_suggestion: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return human-readable reasons when filename rules and Qwen disagree."""

    if not qwen_suggestion:
        return ("missing_qwen_suggestion",)

    reasons: list[str] = []
    qwen_category = normalize_category(str(qwen_suggestion.get("category", "unknown")))
    qwen_object = normalize_tag(str(qwen_suggestion.get("object_name", "")))
    qwen_tags = set(_dedupe_normalized(_as_sequence(qwen_suggestion.get("tags"))))

    if filename_suggestion.category != "unknown" and qwen_category != filename_suggestion.category:
        reasons.append(f"category: filename={filename_suggestion.category}, qwen={qwen_category or 'unknown'}")
    if filename_suggestion.object_name and qwen_object and qwen_object != filename_suggestion.object_name:
        reasons.append(f"object_name: filename={filename_suggestion.object_name}, qwen={qwen_object}")
    if filename_suggestion.object_name and not qwen_object:
        reasons.append(f"object_name: filename={filename_suggestion.object_name}, qwen=missing")

    filename_key_tags = {filename_suggestion.object_name, *filename_suggestion.tags}
    filename_key_tags.discard("")
    if filename_key_tags and qwen_tags and not (filename_key_tags & qwen_tags):
        reasons.append("tags: no overlap between filename rules and qwen")

    return tuple(reasons)


def _split_codes(tokens: list[str]) -> tuple[str, str, list[str]]:
    code_index = -1
    for index, token in enumerate(tokens):
        if token in _CATEGORY_CODES:
            code_index = index
    if code_index < 0 and len(tokens) > 1 and tokens[0] in _STRUCTURAL_PREFIX_CODES:
        code_index = 0

    category_code = tokens[code_index] if code_index >= 0 else ""
    subtype_code = ""
    if code_index >= 0:
        tail = tokens[code_index + 1 :]
        if tail and len(tail[0]) == 1 and tail[0].isalpha():
            subtype_code = tail[0]
            tail = tail[1:]
    else:
        tail = tokens

    semantic = [token for token in tail if token and token not in _GENERIC_TOKENS and not token.isdigit()]
    return category_code, subtype_code, semantic


def _tokens(value: str | Path) -> list[str]:
    stem = Path(str(value)).stem
    stem = re.sub(r"([a-z])([A-Z])", r"\1_\2", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem)
    result: list[str] = []
    for token in stem.split("_"):
        token = token.strip().lower()
        if not token:
            continue
        token = re.sub(r"\d+$", "", token)
        if token and not token.isdigit():
            result.append(token)
    return result


def _object_name(tokens: list[str]) -> str:
    for token in tokens:
        if token in _OBJECT_RULES:
            return token
    normalized = normalize_sprite_id("_".join(token for token in tokens if token not in _GENERIC_TOKENS))
    return normalized


def _category_for(category_code: str, rule: _ObjectRule | None, object_name: str) -> str:
    code_category = _CATEGORY_CODES.get(category_code, "unknown")
    if code_category != "unknown":
        return code_category
    if rule is not None:
        return rule.category
    if object_name:
        return "item_icon"
    return "unknown"


def _confidence(category_code: str, object_name: str, rule: _ObjectRule | None) -> float:
    if rule is not None and category_code:
        return rule.confidence
    if rule is not None:
        return min(rule.confidence, 0.88)
    if category_code and object_name:
        return 0.75
    if object_name:
        return 0.55
    return 0.0


def _confidence_reason(category_code: str, object_name: str, rule: _ObjectRule | None) -> str:
    if rule is not None and category_code:
        return f"recognized filename code {category_code!r} and object token {object_name!r}"
    if rule is not None:
        return f"recognized object token {object_name!r}"
    if category_code and object_name:
        return f"recognized filename code {category_code!r}; object token {object_name!r} is not in the rule table"
    if object_name:
        return f"object token {object_name!r} inferred from filename but not recognized"
    return "no usable object token found in filename"


def _description(object_name: str, category: str) -> str:
    descriptor = _CATEGORY_DESCRIPTORS.get(category, "sprite")
    if not object_name:
        return f"A 32x32 pixel-art {descriptor}."
    return f"A 32x32 pixel-art {object_name.replace('_', ' ')} {descriptor}."


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _dedupe_normalized(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in _as_sequence(values):
        normalized = normalize_tag(str(value))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)
