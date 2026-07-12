"""Fact-grounded description generation and claim validation for v4."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

DESCRIPTION_POLICY_VERSION = "description_facts_v1.2"

_SPECULATIVE_TERMS = frozenset(
    {
        "ancient",
        "enchanted",
        "legendary",
        "magical",
        "healing",
        "poisonous",
        "cursed",
        "valuable",
        "rare",
    }
)
_MATERIAL_TERMS = frozenset(
    {
        "iron",
        "copper",
        "bronze",
        "steel",
        "gold",
        "silver",
        "leather",
        "cloth",
        "chainmail",
        "plate metal",
        "wood",
        "glass",
        "stone",
    }
)
_OBJECT_NOUN_TERMS = frozenset(
    {
        "agate",
        "amethyst",
        "apple",
        "armor",
        "arrow",
        "axe",
        "bottle",
        "bow",
        "bread",
        "buckler",
        "cap",
        "chest",
        "crystal",
        "dagger",
        "diamond",
        "eggplant",
        "gem",
        "helmet",
        "jacket",
        "jewel",
        "key",
        "mace",
        "matchstick",
        "pants",
        "pen",
        "pencil",
        "potion",
        "ring",
        "rod",
        "scroll",
        "shield",
        "shirt",
        "spear",
        "stick",
        "sword",
        "weapon",
    }
)


def generate_description(facts: Mapping[str, Any]) -> str:
    """Generate from accepted/proposed target facts, never from a source caption."""

    facts = normalized_description_facts(facts)
    alias = str(facts.get("surface_alias") or "").strip()
    object_name = str(facts.get("canonical_object") or "").replace("_", " ").strip()
    category = str(facts.get("category") or "").replace("_", " ").strip()
    subject_name = alias or object_name or (category if category and category != "unknown" else "item")
    subject = subject_name if subject_name.endswith(" icon") else f"{subject_name} icon"
    explicit_material = str(facts.get("explicit_material") or "").replace("_", " ").strip()
    colors = _colors(facts)
    shape = facts.get("shape") if isinstance(facts.get("shape"), Mapping) else {}
    silhouettes = _list(shape.get("silhouette"))
    aspects = _list(shape.get("aspect"))
    orientations = _list(shape.get("orientation"))
    visual_forms = _list(facts.get("visual_form"))

    modifiers: list[str] = []
    elongated = any(
        token in " ".join([*visual_forms, *silhouettes, *aspects]).replace("_", " ").lower()
        for token in ("elongated", "rod", "cylinder", "stick")
    )
    if elongated and "elongated" not in subject.lower():
        modifiers.append("elongated")
    if colors and not explicit_material and not _contains_any(subject, colors):
        modifiers.append(colors[0].replace("_", "-"))
    if explicit_material and not _contains_any(subject, [explicit_material]):
        modifiers.append(explicit_material if object_name or alias else explicit_material + "-colored")
    simple_silhouettes = [
        value.replace("_", " ") for value in silhouettes if value in {"round", "oval", "square", "diamond", "compact"}
    ]
    shape_text = " ".join([*silhouettes, *aspects, *visual_forms]).replace("_", " ").lower()
    if object_name == "agate" and "oval" in shape_text and "oval" not in simple_silhouettes:
        simple_silhouettes.insert(0, "oval")
    if not elongated and simple_silhouettes:
        modifiers.append(simple_silhouettes[0])
    phrase = " ".join([*modifiers, subject]).strip()
    details: list[str] = []
    colors_mapping = facts.get("colors") if isinstance(facts.get("colors"), Mapping) else {}
    outline = _list(colors_mapping.get("outline_colors", colors_mapping.get("outline")))
    if outline:
        details.append("a dark outline" if outline[0] in {"black", "dark_gray", "dark_brown"} else "a visible outline")
    if "small" in _list(facts.get("size_hint")) and "small" not in modifiers:
        modifiers.insert(0, "small")
    if "small" in subject.lower():
        subject = re.sub(r"\bsmall\s+", "", subject, flags=re.IGNORECASE)
    if "small" in modifiers and "flattened" in shape_text:
        details = ["a rounded, slightly flattened silhouette"]
    phrase = " ".join([*modifiers, subject]).strip()
    if (
        object_name != "agate"
        and (object_name or alias)
        and orientations
        and orientations[0] in {"diagonal", "horizontal", "vertical"}
    ):
        details.append(orientations[0].replace("_", " ") + " orientation")
    suffix = " with " + _join_words(details) if details else ""
    article = "An" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "A"
    return _sentence(f"{article} {phrase}{suffix}")


def validate_description(description: str, facts: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Reject unsupported semantic claims while allowing ordinary glue words."""

    facts = normalized_description_facts(facts)
    lowered = str(description).lower().replace("_", " ")
    unsupported: list[str] = []
    for term in sorted(_SPECULATIVE_TERMS):
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            unsupported.append(term)
    explicit = str(facts.get("explicit_material") or "").replace("_", " ").lower()
    aliases = {
        explicit,
        str(facts.get("surface_alias") or "").replace("_", " ").lower(),
    }
    for material in sorted(_MATERIAL_TERMS):
        if re.search(rf"\b{re.escape(material)}\b", lowered) and not any(material in value for value in aliases):
            unsupported.append(f"unsupported_material:{material}")
    palette = set(_colors(facts))
    filename_hints = {str(value).replace("_", " ").lower() for value in facts.get("filename_color_hints") or ()}
    known_colors = palette | filename_hints
    for color in ("red", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "black", "white", "gray"):
        supported_families = {part for value in known_colors for part in value.replace("-", "_").split("_")}
        if re.search(rf"\b{color}\b", lowered) and color not in supported_families:
            unsupported.append(f"unsupported_color:{color}")
    allowed_objects = _allowed_object_claims(facts)
    candidate_objects = set(_OBJECT_NOUN_TERMS)
    for key in ("object_alternatives", "alternative_object_terms", "object_claim_candidates"):
        candidate_objects.update(value.replace("_", " ").lower() for value in _list(facts.get(key)))
    for object_term in sorted(candidate_objects, key=len, reverse=True):
        readable = object_term.replace("_", " ")
        if re.search(rf"\b{re.escape(readable)}\b", lowered) and not any(
            readable == allowed or readable in allowed or allowed in readable for allowed in allowed_objects
        ):
            unsupported.append(f"unsupported_object:{readable.replace(' ', '_')}")
    return not unsupported, tuple(unsupported)


def choose_or_regenerate_description(
    candidates: Sequence[str],
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    facts = normalized_description_facts(facts)
    candidate_rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        valid, unsupported = validate_description(candidate, facts)
        if not valid:
            candidate_rejected.append({"value": candidate, "unsupported_claims": list(unsupported)})
    description = generate_description(facts)
    valid, unsupported = validate_description(description, facts)
    final_rejected = [] if valid else [{"value": description, "unsupported_claims": list(unsupported)}]
    return {
        "description": description if valid else None,
        "rejected_description": description if not valid else None,
        "source": "regenerated_from_target_facts",
        "claims_rejected": final_rejected,
        "candidate_claims_rejected": candidate_rejected,
        "policy_version": DESCRIPTION_POLICY_VERSION,
    }


def normalized_description_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sole fact representation shared by generation and validation."""

    result = dict(facts)
    colors = dict(facts.get("colors") or {}) if isinstance(facts.get("colors"), Mapping) else {}
    for name in (
        "palette_colors",
        "primary_colors",
        "secondary_colors",
        "outline_colors",
        "shadow_colors",
        "highlight_colors",
    ):
        colors[name] = list(dict.fromkeys(_list(colors.get(name))))
    result["colors"] = colors
    # Alternatives are provenance only and can never become allowed nouns.
    result["object_alternatives"] = list(_list(facts.get("object_alternatives")))
    result["object_claim_candidates"] = list(_list(facts.get("object_claim_candidates")))
    return result


def _allowed_object_claims(facts: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("canonical_object", "surface_alias", "category", "role"):
        value = str(facts.get(key) or "").replace("_", " ").strip().lower()
        if value and value != "unknown":
            result.add(value)
    for key in ("visual_form", "allowed_object_claims"):
        result.update(value.replace("_", " ").lower() for value in _list(facts.get(key)))
    return result


def _colors(facts: Mapping[str, Any]) -> list[str]:
    colors = facts.get("colors") if isinstance(facts.get("colors"), Mapping) else {}
    for key in ("primary_colors", "primary", "palette_colors"):
        values = _list(colors.get(key))
        if values:
            return [value.replace(" ", "_") for value in values]
    return []


def _list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return [str(item).strip() for item in values if str(item).strip() and str(item).strip() != "unknown"]


def _contains_any(text: str, values: Sequence[str]) -> bool:
    lowered = text.lower().replace("_", " ")
    return any(value.lower().replace("_", " ") in lowered for value in values)


def _parts_phrase(parts: Sequence[str]) -> str:
    readable = [part.replace("_", " ") for part in parts]
    if len(readable) == 1:
        return "a visible " + readable[0]
    return "visible " + _join_words(readable)


def _join_words(values: Sequence[str]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + " and " + cleaned[-1]


def _sentence(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text[:1].upper() + text[1:].rstrip(".") + "."
