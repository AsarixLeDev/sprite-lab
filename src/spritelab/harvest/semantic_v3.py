"""Semantic-v3 compositional metadata layer on top of label-v2 predictions.

Converts label-v2 prediction records into compositional semantic records
(base object + attributes + grounded captions) without ever changing the
primary ``safe_prefill.category`` / ``safe_prefill.object_name`` labels.
Deterministic, offline: no VLM/LLM calls, no network, no GPU.

See ``docs/semantic_labeling_v3_architecture.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.harvest.label_taxonomy import normalize_object_name, normalize_tag, normalize_tags, split_object_tokens
from spritelab.harvest.semantic_extractors import (
    color_values,
    extract_attribute_tokens,
    extract_base_object,
    family_attributes,
    family_expects_color,
    family_for_base_object,
    known_base_object,
)

SCHEMA_VERSION = "semantic_v3.0"

DEFAULT_NEGATIVE_TAGS: tuple[str, ...] = ("photorealistic", "large_scene", "text", "watermark")

# Captions longer than this are a bug in caption generation.
MAX_CAPTION_LENGTH = 220

# Categories whose sprites are item-style icons (get rpg_icon style + fantasy mood).
_ICON_CATEGORIES = frozenset({"item_icon", "weapon", "armor", "material", "effect_icon", "tool"})

_CATEGORY_FUNCTION = {
    "weapon": "weapon",
    "armor": "protection",
    "material": "crafting_material",
    "effect_icon": "status_effect",
    "tool": "tool",
}

_CATEGORY_WORD = {
    "item_icon": "item",
    "weapon": "weapon",
    "armor": "armor",
    "material": "material",
    "effect_icon": "effect",
    "tool": "tool",
    "plant": "plant",
    "block": "block",
    "ui_icon": "ui",
    "entity": "entity",
    "character": "character",
    "environment_prop": "prop",
}

# Functions that make no sense on an effect/status icon even when the base
# object's family implies them (e.g. heart -> crafting_material).
_EFFECT_ICON_FUNCTION_BLOCKLIST = frozenset({"food", "consumable", "crafting_material", "magic_item"})

_SHAPE_HINT_MAP = {
    "roundish": "round",
    "round": "round",
    "square": "square",
    "tall": "tall",
    "wide": "wide",
    "thin": "thin",
}


@dataclass(frozen=True)
class SemanticAttributes:
    colors: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    shapes: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    function: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    style: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "colors",
            "materials",
            "shapes",
            "effects",
            "state",
            "function",
            "mood",
            "style",
            "parts",
            "environment",
        ):
            object.__setattr__(self, name, normalize_tags(getattr(self, name)))

    @property
    def non_empty_group_count(self) -> int:
        return sum(
            1
            for name in (
                "colors",
                "materials",
                "shapes",
                "effects",
                "state",
                "function",
                "mood",
                "parts",
                "environment",
            )
            if getattr(self, name)
        )


@dataclass(frozen=True)
class SemanticV3Record:
    schema_version: str
    category: str
    object_name: str
    base_object: str
    open_name: str
    attributes: SemanticAttributes
    aliases: tuple[str, ...] = ()
    captions: tuple[str, ...] = ()
    prompt_phrases: tuple[str, ...] = ()
    negative_tags: tuple[str, ...] = DEFAULT_NEGATIVE_TAGS
    source_evidence: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


def semantic_attributes_to_json(attributes: SemanticAttributes) -> dict[str, Any]:
    return {
        "colors": list(attributes.colors),
        "materials": list(attributes.materials),
        "shapes": list(attributes.shapes),
        "effects": list(attributes.effects),
        "state": list(attributes.state),
        "function": list(attributes.function),
        "mood": list(attributes.mood),
        "style": list(attributes.style),
        "parts": list(attributes.parts),
        "environment": list(attributes.environment),
    }


def semantic_attributes_from_json(data: Mapping[str, Any] | None) -> SemanticAttributes:
    if not isinstance(data, Mapping):
        return SemanticAttributes()
    return SemanticAttributes(
        colors=_as_str_tuple(data.get("colors")),
        materials=_as_str_tuple(data.get("materials")),
        shapes=_as_str_tuple(data.get("shapes")),
        effects=_as_str_tuple(data.get("effects")),
        state=_as_str_tuple(data.get("state")),
        function=_as_str_tuple(data.get("function")),
        mood=_as_str_tuple(data.get("mood")),
        style=_as_str_tuple(data.get("style")),
        parts=_as_str_tuple(data.get("parts")),
        environment=_as_str_tuple(data.get("environment")),
    )


def semantic_v3_to_json(record: SemanticV3Record) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "category": record.category,
        "object_name": record.object_name,
        "base_object": record.base_object,
        "open_name": record.open_name,
        "attributes": semantic_attributes_to_json(record.attributes),
        "aliases": list(record.aliases),
        "captions": list(record.captions),
        "prompt_phrases": list(record.prompt_phrases),
        "negative_tags": list(record.negative_tags),
        "source_evidence": dict(record.source_evidence or {}),
        "warnings": list(record.warnings),
    }


def semantic_v3_from_json(data: Mapping[str, Any] | None) -> SemanticV3Record | None:
    if not isinstance(data, Mapping):
        return None
    return SemanticV3Record(
        schema_version=str(data.get("schema_version", "")),
        category=str(data.get("category", "")),
        object_name=str(data.get("object_name", "")),
        base_object=str(data.get("base_object", "")),
        open_name=str(data.get("open_name", "")),
        attributes=semantic_attributes_from_json(data.get("attributes")),
        aliases=_as_str_tuple(data.get("aliases")),
        captions=_as_str_tuple(data.get("captions")),
        prompt_phrases=_as_str_tuple(data.get("prompt_phrases")),
        negative_tags=_as_str_tuple(data.get("negative_tags")) or DEFAULT_NEGATIVE_TAGS,
        source_evidence=dict(data.get("source_evidence") or {}),
        warnings=_as_str_tuple(data.get("warnings")),
    )


# ---------------------------------------------------------------------------
# Conversion: label-v2 prediction -> semantic-v3 record
# ---------------------------------------------------------------------------


def build_semantic_v3_record(prediction: Mapping[str, Any], *, max_captions: int = 8) -> SemanticV3Record:
    """Build a semantic-v3 record from one label-v2 prediction record.

    ``safe_prefill.category`` and ``safe_prefill.object_name`` are copied
    verbatim; everything else is inferred deterministically from existing
    prediction fields.
    """

    safe = _mapping(prediction.get("safe_prefill"))
    visual_facts = _mapping(prediction.get("visual_facts"))
    profile = _mapping(prediction.get("source_profile"))

    category = normalize_tag(str(safe.get("category", "unknown"))) or "unknown"
    object_name = normalize_object_name(str(safe.get("object_name", "")))
    name_tokens = split_object_tokens(object_name)
    tags = normalize_tags(_as_str_tuple(safe.get("tags")))

    warnings: list[str] = []
    base_object, base_warnings = extract_base_object(object_name, category=category)
    warnings.extend(base_warnings)

    from_name = extract_attribute_tokens(name_tokens)
    from_tags = extract_attribute_tokens(tags)
    from_family = family_attributes(base_object)

    dominant_colors = normalize_tags(
        _as_str_tuple(visual_facts.get("dominant_colors")) or _as_str_tuple(safe.get("dominant_colors"))
    )
    dominant_color_names = extract_attribute_tokens(dominant_colors).colors

    colors = _dedupe_cap((*from_name.colors, *from_tags.colors, *dominant_color_names), cap=5)
    materials = _dedupe_cap(
        (
            *normalize_tags(_as_str_tuple(safe.get("materials"))),
            *from_name.materials,
            *from_tags.materials,
            *from_family.materials,
        ),
        cap=5,
    )
    shape_hints = tuple(
        _SHAPE_HINT_MAP[hint]
        for hint in normalize_tags(_as_str_tuple(visual_facts.get("shape_hints")))
        if hint in _SHAPE_HINT_MAP
    )
    shapes = _dedupe_cap((*from_name.shapes, *from_family.shapes, *shape_hints), cap=4)
    effects = _dedupe_cap((*from_name.effects, *from_tags.effects), cap=4)
    state = _dedupe_cap((*from_name.state, *from_tags.state, *from_family.state), cap=3)

    function = [*from_family.function]
    category_function = _CATEGORY_FUNCTION.get(category)
    if category_function:
        function.append(category_function)
    if category == "effect_icon":
        function = [value for value in function if value not in _EFFECT_ICON_FUNCTION_BLOCKLIST]
    function_tuple = _dedupe_cap(function, cap=4)

    is_icon = category in _ICON_CATEGORIES or str(profile.get("domain", "")) == "rpg_icons"
    mood = _dedupe_cap(
        (
            *normalize_tags(_as_str_tuple(safe.get("mood"))),
            *from_name.mood,
            *from_tags.mood,
            *(("fantasy",) if is_icon else ()),
        ),
        cap=4,
    )
    style = ("32x32", "pixel_art", "rpg_icon") if is_icon else ("32x32", "pixel_art")

    attributes = SemanticAttributes(
        colors=colors,
        materials=materials,
        shapes=shapes,
        effects=effects,
        state=state,
        function=function_tuple,
        mood=mood,
        style=style,
        parts=from_family.parts,
    )

    if not object_name:
        warnings.append("missing_object_name")
    if attributes.non_empty_group_count == 0:
        warnings.append("no_attributes_extracted")
    if family_expects_color(base_object) and not colors:
        warnings.append("no_color_information")

    aliases = _extract_aliases(prediction, object_name=object_name, base_object=base_object, name_tokens=name_tokens)

    record = SemanticV3Record(
        schema_version=SCHEMA_VERSION,
        category=category,
        object_name=object_name,
        base_object=base_object,
        open_name=_open_name(object_name),
        attributes=attributes,
        aliases=aliases,
        negative_tags=DEFAULT_NEGATIVE_TAGS,
        source_evidence={
            "object_name_source": "safe_prefill",
            "profile": str(profile.get("name", "")),
            "bucket": str(prediction.get("bucket") or _mapping(prediction.get("label_quality")).get("bucket") or ""),
            "name_tokens": list(name_tokens),
            "tags": list(tags),
            "dominant_colors": list(dominant_colors),
            "shape_hints": list(_as_str_tuple(visual_facts.get("shape_hints"))),
        },
        warnings=tuple(warnings),
    )
    captions = build_captions(record, max_captions=max_captions)
    prompt_phrases = build_prompt_phrases(record)
    return SemanticV3Record(
        schema_version=record.schema_version,
        category=record.category,
        object_name=record.object_name,
        base_object=record.base_object,
        open_name=record.open_name,
        attributes=record.attributes,
        aliases=record.aliases,
        captions=captions,
        prompt_phrases=prompt_phrases,
        negative_tags=record.negative_tags,
        source_evidence=record.source_evidence,
        warnings=record.warnings,
    )


def attach_semantic_v3(prediction: Mapping[str, Any], *, max_captions: int = 8) -> dict[str, Any]:
    """Return a copy of the prediction record with a ``semantic_v3`` key added."""

    record = build_semantic_v3_record(prediction, max_captions=max_captions)
    output = dict(prediction)
    output["semantic_v3"] = semantic_v3_to_json(record)
    return output


def convert_label_v2_predictions(
    predictions: Sequence[Mapping[str, Any]], *, max_captions: int = 8
) -> list[dict[str, Any]]:
    return [attach_semantic_v3(prediction, max_captions=max_captions) for prediction in predictions]


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------


def build_captions(record: SemanticV3Record, *, max_captions: int = 8) -> tuple[str, ...]:
    """Generate grounded caption variants from a semantic record.

    Every word is traceable to an extracted attribute or the fixed style
    vocabulary — no invented details, no lore.
    """

    attributes = record.attributes
    base_noun = _open_name(record.base_object)
    base_tokens = set(record.base_object.split("_"))
    open_name = record.open_name or base_noun
    color = _caption_color(attributes.colors)
    material = _first_solid_material(attributes.materials, base_tokens=base_tokens)
    effect = attributes.effects[0] if attributes.effects else ""
    if effect in base_tokens:
        effect = ""
    state = attributes.state[0] if attributes.state else ""
    category_word = _CATEGORY_WORD.get(record.category, "")
    is_icon = "rpg_icon" in attributes.style

    captions: list[str] = []

    # 1. minimal
    captions.append(open_name)

    # 2. decomposed: adjectives + base + material clause
    adjectives = _join_words([effect, color, state])
    decomposed = f"{adjectives} {base_noun}".strip()
    if material:
        decomposed = f"{decomposed} made of {_open_name(material)}"
    if decomposed != open_name:
        captions.append(decomposed)

    # 3. style-aware
    style_color = "" if _color_redundant(color, open_name) else color
    if is_icon:
        style_parts = ["32x32 pixel art"]
        if "fantasy" in attributes.mood:
            style_parts.append("fantasy RPG")
        style_parts.append(_join_words([style_color, open_name]))
        style_parts.append("icon")
        captions.append(_join_words(style_parts))
    else:
        captions.append(_join_words(["32x32 pixel art", style_color, open_name]))

    # 4. prompt-like
    prompt_parts = ["centered 32x32 pixel art", _join_words([color, _open_name(material), base_noun])]
    if "black" in attributes.colors:
        prompt_parts.append(", black outline")
    prompt_parts.append(", transparent background")
    captions.append("".join(part if part.startswith(",") else f" {part}" for part in prompt_parts).strip())

    # 5. attribute dropout variants
    captions.append(base_noun)
    if color:
        captions.append(_join_words([color, base_noun]))
    if category_word:
        captions.append(f"{category_word} icon" if is_icon else category_word)
    if effect and effect != color:
        captions.append(_join_words([effect, base_noun]))

    cleaned: list[str] = []
    seen: set[str] = set()
    for caption in captions:
        text = " ".join(str(caption).split())
        key = text.lower()
        if not text or key in seen or len(text) > MAX_CAPTION_LENGTH:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max(1, int(max_captions)):
            break
    return tuple(cleaned)


def build_prompt_phrases(record: SemanticV3Record) -> tuple[str, ...]:
    attributes = record.attributes
    open_name = record.open_name or _open_name(record.base_object)
    color = _caption_color(attributes.colors)
    category_word = _CATEGORY_WORD.get(record.category, "item")
    phrases = [f"32x32 pixel art {open_name}".strip()]
    second = _join_words(
        [
            color,
            "fantasy" if "fantasy" in attributes.mood else "",
            _open_name(record.base_object),
            category_word,
            "icon",
        ]
    )
    if second and second.lower() != phrases[0].lower():
        phrases.append(second)
    return tuple(" ".join(phrase.split()) for phrase in phrases if phrase)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def _extract_aliases(
    prediction: Mapping[str, Any],
    *,
    object_name: str,
    base_object: str,
    name_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    """Grounded aliases: VLM alternatives vetted by the candidate list, plus
    same-family name tokens (e.g. ``ruby`` for ``ruby_gem``)."""

    candidates = {normalize_object_name(value) for value in _as_str_tuple(prediction.get("candidate_object_names"))}
    vlm = _mapping(prediction.get("vlm_descriptor") or prediction.get("vlm_suggestion"))
    aliases: list[str] = []
    for value in _as_str_tuple(vlm.get("alternative_object_names")):
        alias = normalize_object_name(value)
        if alias and alias != object_name and alias in candidates:
            aliases.append(alias)
    for token in name_tokens:
        if token != base_object and token != object_name and known_base_object(token):
            aliases.append(token)
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        open_alias = _open_name(alias)
        if open_alias and open_alias not in seen:
            seen.add(open_alias)
            result.append(open_alias)
        if len(result) >= 4:
            break
    return tuple(result)


# ---------------------------------------------------------------------------
# Summary / report
# ---------------------------------------------------------------------------


def summarize_semantic_v3_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    with_semantic = 0
    caption_count = 0
    attribute_group_count = 0
    base_objects: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    materials: Counter[str] = Counter()
    effects: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    family_with_colors: Counter[str] = Counter()
    family_with_function: Counter[str] = Counter()
    base_object_present = 0
    base_object_fallbacks = 0

    for record in records:
        semantic = _mapping(record.get("semantic_v3"))
        if not semantic:
            continue
        with_semantic += 1
        parsed = semantic_v3_from_json(semantic)
        if parsed is None:
            continue
        caption_count += len(parsed.captions)
        attribute_group_count += parsed.attributes.non_empty_group_count
        if parsed.base_object:
            base_object_present += 1
            base_objects[parsed.base_object] += 1
        if "base_object_fallback_full_name" in parsed.warnings:
            base_object_fallbacks += 1
        for value in parsed.attributes.colors:
            colors[value] += 1
        for value in parsed.attributes.materials:
            materials[value] += 1
        for value in parsed.attributes.effects:
            effects[value] += 1
        for warning in parsed.warnings:
            warning_counts[warning] += 1
        family = family_for_base_object(parsed.base_object)
        family_name = family.name if family is not None else "(none)"
        family_totals[family_name] += 1
        if parsed.attributes.colors:
            family_with_colors[family_name] += 1
        if parsed.attributes.function:
            family_with_function[family_name] += 1

    return {
        "records": total,
        "records_with_semantic_v3": with_semantic,
        "average_captions": caption_count / with_semantic if with_semantic else 0.0,
        "average_attribute_groups": attribute_group_count / with_semantic if with_semantic else 0.0,
        "base_object_coverage": base_object_present / with_semantic if with_semantic else 0.0,
        "base_object_fallbacks": base_object_fallbacks,
        "top_base_objects": dict(base_objects.most_common(25)),
        "top_colors": dict(colors.most_common(15)),
        "top_materials": dict(materials.most_common(15)),
        "top_effects": dict(effects.most_common(15)),
        "warnings": dict(warning_counts.most_common()),
        "family_coverage": {
            name: {
                "records": int(family_totals[name]),
                "with_colors": int(family_with_colors.get(name, 0)),
                "with_function": int(family_with_function.get(name, 0)),
            }
            for name in sorted(family_totals)
        },
    }


def format_semantic_v3_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Semantic v3 Report",
        "",
        f"Records: {int(summary.get('records', 0))}",
        f"Records with semantic_v3: {int(summary.get('records_with_semantic_v3', 0))}",
        f"Average captions: {float(summary.get('average_captions', 0.0)):.2f}",
        f"Average attribute groups: {float(summary.get('average_attribute_groups', 0.0)):.2f}",
        f"Base object coverage: {float(summary.get('base_object_coverage', 0.0)):.3f}",
        f"Base object fallbacks: {int(summary.get('base_object_fallbacks', 0))}",
        "",
        "## Top base objects",
    ]
    for name, count in dict(summary.get("top_base_objects") or {}).items():
        lines.append(f"- {name}: {count}")
    for title, key in (
        ("Top colors", "top_colors"),
        ("Top materials", "top_materials"),
        ("Top effects", "top_effects"),
    ):
        lines.extend(["", f"## {title}"])
        for name, count in dict(summary.get(key) or {}).items():
            lines.append(f"- {name}: {count}")
    lines.extend(["", "## Family coverage"])
    for name, coverage in dict(summary.get("family_coverage") or {}).items():
        coverage = dict(coverage or {})
        lines.append(
            f"- {name}: records={int(coverage.get('records', 0))}"
            f" with_colors={int(coverage.get('with_colors', 0))}"
            f" with_function={int(coverage.get('with_function', 0))}"
        )
    lines.extend(["", "## Warnings"])
    warnings = dict(summary.get("warnings") or {})
    if warnings:
        for name, count in warnings.items():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Materials that read as substances in "made of X" clauses; skip abstract ones.
_NON_SOLID_MATERIALS = frozenset({"liquid", "organic"})

# Neutral colors that are usually outline/shading, not the object's identity.
_NEUTRAL_COLORS = frozenset({"black", "white", "gray", "dark"})


def _first_solid_material(materials: Sequence[str], *, base_tokens: set[str] = frozenset()) -> str:
    for material in materials:
        if material in _NON_SOLID_MATERIALS or material in base_tokens:
            continue
        return material
    return ""


def _caption_color(colors: Sequence[str]) -> str:
    """Lead caption color: prefer the object's identity color over outline neutrals."""

    for value in colors:
        if value not in _NEUTRAL_COLORS:
            return value
    return colors[0] if colors else ""


def _color_redundant(color: str, open_name: str) -> bool:
    """True when the open name already carries the color (golden -> gold)."""

    if not color:
        return True
    return any(color in color_values(token) for token in open_name.split())


def _open_name(value: str) -> str:
    return str(value).replace("_", " ").strip()


def _join_words(values: Sequence[str]) -> str:
    return " ".join(word for word in (str(value).strip() for value in values) if word)


def _dedupe_cap(values: Sequence[str], *, cap: int) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = normalize_tag(str(value))
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= cap:
            break
    return tuple(result)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
