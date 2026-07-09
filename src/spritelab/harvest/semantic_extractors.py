"""Reusable source-family semantic extractor vocabularies.

These dictionaries decompose object names and tags into semantic components
(base object, colors, materials, effects, state, function, shape, parts).
They are deliberately generic: adding a new source pack should mean adding a
few *generic* tokens here, never a per-pack ``filename -> exact object`` map.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from spritelab.harvest.label_taxonomy import normalize_tag, split_object_tokens

# ---------------------------------------------------------------------------
# Token vocabularies (pack-independent)
# ---------------------------------------------------------------------------

# token -> color names it implies (first entry is the primary color)
COLOR_TOKENS: dict[str, tuple[str, ...]] = {
    "red": ("red",),
    "blue": ("blue",),
    "green": ("green",),
    "yellow": ("yellow",),
    "orange": ("orange",),
    "purple": ("purple",),
    "violet": ("purple",),
    "pink": ("pink",),
    "white": ("white",),
    "black": ("black",),
    "gray": ("gray",),
    "grey": ("gray",),
    "brown": ("brown",),
    "cyan": ("cyan",),
    "teal": ("teal",),
    "gold": ("gold", "yellow"),
    "golden": ("gold", "yellow"),
    "silver": ("silver", "gray"),
    "bronze": ("bronze", "brown"),
    "crimson": ("red",),
    "azure": ("blue",),
    "emerald": ("green",),
    "ruby": ("red",),
    "sapphire": ("blue",),
    "amethyst": ("purple",),
    "jade": ("green",),
    "topaz": ("yellow",),
    "obsidian": ("black",),
    "pearl": ("white",),
    "dark": ("dark",),
    "light_gray": ("gray",),
}

# token -> materials it implies
MATERIAL_TOKENS: dict[str, tuple[str, ...]] = {
    "gold": ("metal", "gold"),
    "golden": ("metal", "gold"),
    "silver": ("metal", "silver"),
    "bronze": ("metal", "bronze"),
    "iron": ("metal", "iron"),
    "steel": ("metal", "steel"),
    "metal": ("metal",),
    "copper": ("metal", "copper"),
    "wood": ("wood",),
    "wooden": ("wood",),
    "leather": ("leather",),
    "stone": ("stone",),
    "rock": ("stone",),
    "bone": ("bone",),
    "glass": ("glass",),
    "crystal": ("crystal",),
    "cloth": ("fabric",),
    "fabric": ("fabric",),
    "fur": ("fur",),
    "paper": ("paper",),
    "obsidian": ("crystal", "mineral"),
    "ruby": ("crystal", "mineral"),
    "sapphire": ("crystal", "mineral"),
    "amethyst": ("crystal", "mineral"),
    "emerald": ("crystal", "mineral"),
    "diamond": ("crystal", "mineral"),
    "opal": ("crystal", "mineral"),
    "agate": ("crystal", "mineral"),
    "jade": ("crystal", "mineral"),
    "topaz": ("crystal", "mineral"),
    "garnet": ("crystal", "mineral"),
    "quartz": ("crystal", "mineral"),
    "pearl": ("mineral",),
}

# token -> effects it implies
EFFECT_TOKENS: dict[str, tuple[str, ...]] = {
    "fire": ("fire",),
    "flame": ("fire",),
    "burning": ("fire", "burning"),
    "fireball": ("fire",),
    "ice": ("ice",),
    "frost": ("ice",),
    "frozen": ("ice", "frozen"),
    "electric": ("electric",),
    "thunder": ("electric",),
    "lightning": ("electric",),
    "charged": ("electric", "charged"),
    "shock": ("electric",),
    "poison": ("poison",),
    "toxic": ("poison",),
    "venom": ("poison",),
    "holy": ("holy",),
    "blessed": ("holy", "blessed"),
    "shadow": ("shadow",),
    "dark": ("shadow",),
    "cursed": ("cursed",),
    "bleeding": ("bleeding",),
    "blood": ("bleeding",),
    "magic": ("magic",),
    "arcane": ("magic",),
    "calming": ("calming",),
    "soothing": ("calming",),
    "glowing": ("glowing",),
    "glow": ("glowing",),
    "wind": ("wind",),
    "air": ("wind",),
    "water": ("water",),
    "wave": ("water",),
    "earth": ("earth",),
    "nature": ("nature",),
    "speed": ("speed",),
    "moonlit": ("glowing", "moonlit"),
}

# token -> shape it implies (used for names like square_gem)
SHAPE_TOKENS: dict[str, tuple[str, ...]] = {
    "square": ("square",),
    "round": ("round",),
    "oval": ("oval",),
    "star": ("star",),
    "spiral": ("spiral",),
    "twisted": ("twisted",),
    "long": ("long",),
    "curved": ("curved",),
}

# token -> object state it implies
STATE_TOKENS: dict[str, tuple[str, ...]] = {
    "raw": ("raw",),
    "cooked": ("cooked",),
    "polished": ("polished",),
    "broken": ("broken",),
    "cracked": ("cracked",),
    "rusty": ("rusty",),
    "sliced": ("sliced",),
    "slice": ("sliced",),
    "dried": ("dried",),
    "solid": ("solid",),
}

# effect -> extra visual/mood hints usable by caption or creative layers
EFFECT_MOOD_HINTS: dict[str, tuple[str, ...]] = {
    "calming": ("soft", "soothing"),
    "holy": ("radiant",),
    "cursed": ("ominous",),
    "shadow": ("ominous",),
}


# ---------------------------------------------------------------------------
# Base-object families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticFamily:
    """Default semantic attributes contributed by a family of base objects."""

    name: str
    base_objects: tuple[str, ...]
    function: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    shapes: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    expects_color: bool = False


SEMANTIC_FAMILIES: tuple[SemanticFamily, ...] = (
    SemanticFamily(
        name="gem",
        base_objects=(
            "gem",
            "crystal",
            "diamond",
            "ruby",
            "sapphire",
            "amethyst",
            "emerald",
            "opal",
            "agate",
            "jade",
            "topaz",
            "garnet",
            "quartz",
            "pearl",
            "obsidian",
        ),
        function=("crafting_material",),
        materials=("crystal", "mineral"),
        expects_color=True,
    ),
    SemanticFamily(
        name="container",
        base_objects=("potion", "vial", "bottle", "flask", "jar", "test_tube"),
        function=("consumable", "magic_item"),
        materials=("glass", "liquid"),
        parts=("cork",),
        expects_color=True,
    ),
    SemanticFamily(
        name="food",
        base_objects=(
            "apple",
            "banana",
            "bread",
            "cheese",
            "cherry",
            "grape",
            "lemon",
            "orange",
            "pear",
            "mulberry",
            "strawberry",
            "watermelon",
            "pineapple",
            "radish",
            "carrot",
            "pepper",
            "tomato",
            "potato",
            "nut",
            "meat",
            "fish",
            "pie",
            "mushroom",
            "skewer",
            "egg",
        ),
        function=("food", "consumable"),
    ),
    SemanticFamily(
        name="weapon",
        base_objects=(
            "sword",
            "dagger",
            "axe",
            "bow",
            "arrow",
            "mace",
            "spear",
            "staff",
            "wand",
            "gun",
            "cannon",
            "knife",
            "club",
            "whip",
            "scythe",
            "crossbow",
            "shuriken",
            "fist",
            "blade",
            "throwing_star",
        ),
        function=("weapon",),
    ),
    SemanticFamily(
        name="armor",
        base_objects=(
            "chestplate",
            "breastplate",
            "cuirass",
            "helmet",
            "helm",
            "shield",
            "boots",
            "shoes",
            "gloves",
            "gauntlets",
            "tunic",
            "hat",
            "cloak",
            "belt",
        ),
        function=("protection",),
    ),
    SemanticFamily(
        name="tool",
        base_objects=(
            "hammer",
            "pickaxe",
            "shovel",
            "compass",
            "ruler",
            "scissors",
            "telescope",
            "clock",
            "mirror",
            "map",
            "torch",
            "rope",
            "needle",
            "brush",
        ),
        function=("tool",),
    ),
    SemanticFamily(
        name="jewelry",
        base_objects=(
            "ring",
            "necklace",
            "amulet",
            "medal",
            "medallion",
            "pendant",
            "earring",
            "bracelet",
            "crown",
        ),
        function=("accessory",),
        state=("ornamental",),
    ),
    SemanticFamily(
        name="key",
        base_objects=("key",),
        function=("unlocking",),
        materials=("metal",),
    ),
    SemanticFamily(
        name="currency_material",
        base_objects=("coin", "bar", "ingot", "ore", "shard", "coal"),
        function=("crafting_material",),
    ),
    SemanticFamily(
        name="organic_part",
        base_objects=(
            "bone",
            "feather",
            "wing",
            "fang",
            "claw",
            "tail",
            "paw",
            "shell",
            "tentacle",
            "leg",
            "beak",
            "eye",
            "heart",
            "sinew",
            "spores",
            "root",
            "resin",
            "leaf",
            "clover",
            "ash",
            "scale",
            "fur",
        ),
        function=("crafting_material",),
        materials=("organic",),
    ),
    SemanticFamily(
        name="readable",
        base_objects=("book", "scroll", "letter", "note"),
        function=("readable",),
        materials=("paper",),
    ),
    SemanticFamily(
        name="prop",
        base_objects=("chest", "rock", "lantern", "candle", "barrel", "crate", "wood", "fabric", "bucket", "ball"),
        function=("prop",),
    ),
)

# Per-base-object attribute extras where a family default is too coarse.
BASE_OBJECT_EXTRAS: dict[str, dict[str, tuple[str, ...]]] = {
    "chestplate": {"shapes": ("torso_shaped",), "parts": ("shoulder_plates",)},
    "breastplate": {"shapes": ("torso_shaped",)},
    "cuirass": {"shapes": ("torso_shaped",)},
    "helmet": {"shapes": ("head_shaped",)},
    "boots": {"parts": ("sole",)},
    "shoes": {"parts": ("sole",)},
    "shield": {"shapes": ("rounded",)},
    "arrow": {"function": ("projectile",), "parts": ("shaft", "arrowhead")},
    "bow": {"function": ("ranged",), "parts": ("string", "limbs")},
    "gun": {"function": ("ranged",)},
    "cannon": {"function": ("ranged",)},
    "throwing_star": {"function": ("projectile",)},
    "shuriken": {"function": ("projectile",)},
    "sword": {"function": ("melee",), "parts": ("blade", "hilt")},
    "dagger": {"function": ("melee",), "parts": ("blade", "hilt")},
    "axe": {"function": ("melee",), "parts": ("blade", "handle")},
    "mace": {"function": ("melee",), "parts": ("head", "handle")},
    "spear": {"function": ("melee",), "parts": ("shaft", "point")},
    "hammer": {"parts": ("head", "handle")},
    "pickaxe": {"parts": ("head", "handle")},
    "compass": {"function": ("navigation",)},
    "map": {"function": ("navigation",)},
    "telescope": {"function": ("navigation",)},
    "ruler": {"function": ("measuring",)},
    "scissors": {"function": ("cutting",)},
    "gem": {"state": ("polished",)},
}

# Compound object names whose identity is the compound itself; the value is
# the base object to use (usually the compound, sometimes the head noun).
COMPOUND_BASE_OVERRIDES: dict[str, str] = {
    "throwing_star": "throwing_star",
    "fish_skewer": "skewer",
    "raw_fish_skewer": "skewer",
    "water_wave": "water_wave",
    "test_tube": "test_tube",
    "chest_bones": "bone",
    "pie_slice": "pie",
    "watermelon_slice": "watermelon",
    "medicine_bottle": "bottle",
    "medicine_vial": "vial",
    "ink_bucket": "bucket",
}

# Effect-icon names whose identity *is* the effect (496 S_/B_ style icons).
EFFECT_ICON_BASES: frozenset[str] = frozenset(
    {
        "fire",
        "ice",
        "holy",
        "shadow",
        "thunder",
        "wind",
        "water",
        "earth",
        "poison",
        "magic",
        "buff",
        "physic",
        "light_effect",
        "fireball",
        "fire_spiral",
        "water_wave",
        "throw",
        "speed_buff",
    }
)

_FAMILY_BY_BASE: dict[str, SemanticFamily] = {
    base: family for family in SEMANTIC_FAMILIES for base in family.base_objects
}


def family_for_base_object(base_object: str) -> SemanticFamily | None:
    """Return the semantic family owning ``base_object``, if any."""

    return _FAMILY_BY_BASE.get(normalize_tag(base_object))


def known_base_object(token: str) -> str:
    """Return the canonical base object for a token, or '' if unknown."""

    normalized = normalize_tag(token)
    if normalized in _FAMILY_BY_BASE:
        return normalized
    # light plural tolerance beyond the shared taxonomy replacements
    if normalized.endswith("s") and normalized[:-1] in _FAMILY_BY_BASE:
        return normalized[:-1]
    return ""


def extract_base_object(object_name: str, *, category: str = "") -> tuple[str, tuple[str, ...]]:
    """Extract the visual-identity base object from an object name.

    Returns ``(base_object, warnings)``. Conservative policy:

    1. exact compound overrides;
    2. right-to-left first known base-object token;
    3. effect-icon identities become their own base object;
    4. fallback to the full name (with a warning when compound).
    """

    tokens = split_object_tokens(object_name)
    if not tokens:
        return "", ("empty_object_name",)
    joined = "_".join(tokens)

    if joined in COMPOUND_BASE_OVERRIDES:
        return COMPOUND_BASE_OVERRIDES[joined], ()

    for token in reversed(tokens):
        base = known_base_object(token)
        if base:
            return base, ()

    if joined in EFFECT_ICON_BASES or normalize_tag(category) == "effect_icon":
        return joined, ()

    warnings = ("base_object_fallback_full_name",) if len(tokens) > 1 else ()
    return joined, warnings


@dataclass(frozen=True)
class ExtractedAttributes:
    """Raw semantic attribute lists extracted from tokens and family defaults."""

    colors: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    shapes: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    function: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())


def extract_attribute_tokens(tokens: Iterable[str]) -> ExtractedAttributes:
    """Map raw name/tag tokens onto color/material/effect/state vocabularies."""

    colors: list[str] = []
    materials: list[str] = []
    shapes: list[str] = []
    effects: list[str] = []
    state: list[str] = []
    mood: list[str] = []
    for raw in tokens:
        token = normalize_tag(raw)
        if not token:
            continue
        for value in color_values(token):
            colors.append(value)
        for value in MATERIAL_TOKENS.get(token, ()):
            materials.append(value)
        for value in SHAPE_TOKENS.get(token, ()):
            shapes.append(value)
        for value in EFFECT_TOKENS.get(token, ()):
            effects.append(value)
        for value in STATE_TOKENS.get(token, ()):
            state.append(value)
    for effect in effects:
        for hint in EFFECT_MOOD_HINTS.get(effect, ()):
            mood.append(hint)
    return ExtractedAttributes(
        colors=_dedupe(colors),
        materials=_dedupe(materials),
        shapes=_dedupe(shapes),
        effects=_dedupe(effects),
        state=_dedupe(state),
        mood=_dedupe(mood),
    )


def color_values(token: str) -> tuple[str, ...]:
    """Color names implied by a token, tolerating light_/dark_ prefixes."""

    normalized = normalize_tag(token)
    if normalized in COLOR_TOKENS:
        return COLOR_TOKENS[normalized]
    for prefix in ("light_", "dark_", "pale_", "deep_"):
        if normalized.startswith(prefix) and normalized[len(prefix) :] in COLOR_TOKENS:
            return COLOR_TOKENS[normalized[len(prefix) :]]
    return ()


def family_attributes(base_object: str) -> ExtractedAttributes:
    """Default attributes contributed by the base object's family + extras."""

    base = normalize_tag(base_object)
    family = family_for_base_object(base)
    function: list[str] = []
    materials: list[str] = []
    shapes: list[str] = []
    parts: list[str] = []
    state: list[str] = []
    if family is not None:
        function.extend(family.function)
        materials.extend(family.materials)
        shapes.extend(family.shapes)
        parts.extend(family.parts)
        state.extend(family.state)
    elif base in EFFECT_ICON_BASES:
        function.append("status_effect")
    extras = BASE_OBJECT_EXTRAS.get(base, {})
    function.extend(extras.get("function", ()))
    materials.extend(extras.get("materials", ()))
    shapes.extend(extras.get("shapes", ()))
    parts.extend(extras.get("parts", ()))
    state.extend(extras.get("state", ()))
    return ExtractedAttributes(
        materials=_dedupe(materials),
        shapes=_dedupe(shapes),
        function=_dedupe(function),
        parts=_dedupe(parts),
        state=_dedupe(state),
    )


def family_expects_color(base_object: str) -> bool:
    family = family_for_base_object(base_object)
    return bool(family is not None and family.expects_color)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = normalize_tag(str(value))
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return tuple(result)
