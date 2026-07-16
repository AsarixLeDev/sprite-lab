"""Small grammar for decomposing creative concept names into semantics.

A future generator must translate names like ``charged_sinew`` or
``calming_spores`` into grounded visual semantics even though those exact
objects never appear in any dataset. This module parses
``modifier* + substance/base_object`` names into the same
:class:`~spritelab.harvest.semantic_v3.SemanticAttributes` vocabulary the
dataset uses, so prompt-side and dataset-side semantics stay aligned.

Deliberately tiny: dictionaries plus one parse function. No model calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from spritelab.harvest.label_taxonomy import split_object_tokens
from spritelab.harvest.semantic_extractors import (
    EFFECT_MOOD_HINTS,
    color_values,
    extract_attribute_tokens,
    extract_base_object,
    family_attributes,
    known_base_object,
)
from spritelab.harvest.semantic_v3 import SemanticAttributes

# modifier -> visual translation tokens (colors / effects / mood / shapes)
VISUAL_TRANSLATION: dict[str, dict[str, tuple[str, ...]]] = {
    "charged": {"effects": ("electric", "charged"), "colors": ("blue",), "mood": ("energetic",)},
    "calming": {"effects": ("calming",), "colors": ("green",), "mood": ("soft", "soothing"), "shapes": ("rounded",)},
    "cursed": {"effects": ("cursed",), "colors": ("purple",), "mood": ("ominous",)},
    "blessed": {"effects": ("holy", "blessed"), "colors": ("gold",), "mood": ("radiant",)},
    "glowing": {"effects": ("glowing",), "mood": ("radiant",)},
    "frozen": {"effects": ("ice", "frozen"), "colors": ("blue", "white")},
    "burning": {"effects": ("fire", "burning"), "colors": ("orange", "red")},
    "moonlit": {"effects": ("glowing", "moonlit"), "colors": ("blue", "white"), "mood": ("calm",)},
    "bitter": {"mood": ("harsh",), "colors": ("brown",)},
    "mossy": {"colors": ("green",), "materials": ("moss",), "state": ("weathered",)},
}

# substance -> intrinsic visual properties
SUBSTANCE_PROPERTIES: dict[str, dict[str, tuple[str, ...]]] = {
    "sinew": {"materials": ("organic",), "shapes": ("fibrous", "twisted"), "colors": ("red",)},
    "spores": {"materials": ("organic", "powder"), "shapes": ("small_dots", "particle_cloud")},
    "resin": {"materials": ("organic",), "state": ("glossy",), "colors": ("amber",)},
    "root": {"materials": ("organic", "wood"), "shapes": ("twisted",), "colors": ("brown",)},
    "bone": {"materials": ("bone",), "colors": ("white",)},
    "crystal": {"materials": ("crystal", "mineral"), "shapes": ("faceted",)},
    "ash": {"materials": ("powder",), "colors": ("gray",)},
    "venom": {"materials": ("liquid",), "effects": ("poison",), "colors": ("green",)},
    "dreamroot": {"materials": ("organic", "wood"), "shapes": ("twisted",), "mood": ("mystical",)},
}


@dataclass(frozen=True)
class CreativeConcept:
    """Decomposition of a creative concept name into semantic fields."""

    name: str
    base_object: str
    modifiers: tuple[str, ...]
    attributes: SemanticAttributes
    recognized: bool


def parse_creative_concept(name: str) -> CreativeConcept:
    """Decompose a concept name (``charged_sinew``) into visual semantics.

    The last token that is a known substance or base object carries the
    visual identity; the remaining tokens are treated as modifiers and mapped
    through the shared attribute vocabularies plus :data:`VISUAL_TRANSLATION`.
    """

    tokens = split_object_tokens(name)
    if not tokens:
        return CreativeConcept(name="", base_object="", modifiers=(), attributes=SemanticAttributes(), recognized=False)

    base = ""
    base_index = -1
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token in SUBSTANCE_PROPERTIES or known_base_object(token):
            base = token
            base_index = index
            break
    if not base:
        base, _ = extract_base_object("_".join(tokens))
        base_index = len(tokens) - 1
    modifiers = tuple(token for index, token in enumerate(tokens) if index != base_index)

    colors: list[str] = []
    materials: list[str] = []
    shapes: list[str] = []
    effects: list[str] = []
    state: list[str] = []
    mood: list[str] = []
    recognized_any = base in SUBSTANCE_PROPERTIES or bool(known_base_object(base))

    for group, values in SUBSTANCE_PROPERTIES.get(base, {}).items():
        _extend(group, values, colors, materials, shapes, effects, state, mood)
    family = family_attributes(base)
    materials.extend(family.materials)
    shapes.extend(family.shapes)
    state.extend(family.state)

    for modifier in modifiers:
        translation = VISUAL_TRANSLATION.get(modifier)
        if translation is not None:
            recognized_any = True
            for group, values in translation.items():
                _extend(group, values, colors, materials, shapes, effects, state, mood)
            continue
        extracted = extract_attribute_tokens((modifier,))
        if extracted.colors or extracted.materials or extracted.effects or extracted.state or extracted.shapes:
            recognized_any = True
        colors.extend(extracted.colors)
        colors.extend(color_values(modifier))
        materials.extend(extracted.materials)
        shapes.extend(extracted.shapes)
        effects.extend(extracted.effects)
        state.extend(extracted.state)

    for effect in effects:
        mood.extend(EFFECT_MOOD_HINTS.get(effect, ()))

    attributes = SemanticAttributes(
        colors=tuple(colors),
        materials=tuple(materials),
        shapes=tuple(shapes),
        effects=tuple(effects),
        state=tuple(state),
        function=family_attributes(base).function,
        mood=tuple(mood),
        style=("32x32", "pixel_art"),
    )
    return CreativeConcept(
        name="_".join(tokens),
        base_object=base,
        modifiers=modifiers,
        attributes=attributes,
        recognized=recognized_any,
    )


def _extend(
    group: str,
    values: tuple[str, ...],
    colors: list[str],
    materials: list[str],
    shapes: list[str],
    effects: list[str],
    state: list[str],
    mood: list[str],
) -> None:
    target = {
        "colors": colors,
        "materials": materials,
        "shapes": shapes,
        "effects": effects,
        "state": state,
        "mood": mood,
    }.get(group)
    if target is not None:
        target.extend(values)
