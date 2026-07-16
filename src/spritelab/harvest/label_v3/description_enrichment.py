"""Derived, non-evidentiary training-description helpers for Auto-Labeling v3."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DescriptionArtifact:
    canonical_description: str = ""
    enriched_description: str = ""
    facts_used: tuple[str, ...] = ()
    facts_omitted: tuple[str, ...] = ()
    unsupported_claims_detected: tuple[str, ...] = ()
    valid: bool = True
    dependency_group: str = "derived_description"

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_description": self.canonical_description,
            "enriched_description": self.enriched_description,
            "facts_used": list(self.facts_used),
            "facts_omitted": list(self.facts_omitted),
            "unsupported_claims_detected": list(self.unsupported_claims_detected),
            "valid": self.valid,
            "dependency_group": self.dependency_group,
        }


def accepted_description_facts(fields: Mapping[str, Any], *, include_uncertain: bool = False) -> dict[str, Any]:
    """Read only accepted values (or explicitly opted-in uncertain values)."""
    facts: dict[str, Any] = {}
    for name, value in fields.items():
        if isinstance(value, Mapping):
            state, raw = str(value.get("state", "")), value.get("value", value.get("accepted_value"))
            if state != "accepted" and not include_uncertain:
                continue
        else:
            raw = value
        if isinstance(raw, str) and raw.strip() and raw.strip().lower() not in {"unknown", "none_of_the_above"}:
            facts[name] = raw.strip()
        elif isinstance(raw, (list, tuple)) and raw:
            facts[name] = [str(v) for v in raw if str(v).strip()]
    return facts


def canonical_description_from_facts(facts: Mapping[str, Any]) -> str:
    """Deterministically express accepted structured facts in one sentence."""
    object_name = facts.get("canonical_object") or facts.get("category") or facts.get("domain")
    if not object_name:
        return ""
    object_name = str(object_name).replace("_", " ")
    modifiers: list[str] = []
    # Prefer the visual reading order users expect: primary color, shape,
    # grounded material, object.  Flat ``color`` remains the compatibility
    # fallback for older records.
    for name in ("primary_colors", "color", "shape", "material"):
        value = facts.get(name)
        if isinstance(value, (list, tuple)):
            selected = list(value)
            if name in {"color", "primary_colors"}:
                selected = [v for v in selected if str(v) not in {"black", "white", "transparent"}][:1] or selected[:1]
            else:
                selected = selected[:2]
            modifiers.extend(str(v).replace("_", " ") for v in selected)
        elif value:
            modifiers.append(str(value).replace("_", " "))
    modifiers = list(dict.fromkeys(modifiers))
    prefix = " ".join(modifiers)
    article = "An" if (prefix or object_name).lower().startswith(tuple("aeiou")) else "A"
    description = f"{article} {prefix + ' ' if prefix else ''}{object_name}"
    highlights = facts.get("highlight_colors") or ()
    if isinstance(highlights, str):
        highlights = (highlights,)
    if highlights:
        description += f" with {str(highlights[0]).replace('_', ' ')} highlights"
    outline = facts.get("outline_color")
    if outline:
        description += f" and a {str(outline).replace('_', ' ')} outline"
    return description + "."


def validate_enriched_description(text: str, facts: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Validate normalized semantic claims while allowing safe paraphrases."""
    candidate = " ".join(str(text).strip().split())
    if not candidate or candidate.count(".") + candidate.count("!") + candidate.count("?") > 1:
        return False, ("not_one_sentence",)
    allowed = set(
        "a an the with and of is has made from in on by small large form object icon sprite "
        "rendered featuring several multiple dark light body base top edges surface appearance".split()
    )
    for value in facts.values():
        values = value if isinstance(value, (list, tuple)) else (value,)
        for one in values:
            allowed.update(re.findall(r"[a-z]+", str(one).lower().replace("_", " ")))
    synonym_groups = (
        {"round", "rounded", "circular"},
        {"outline", "outlined"},
        {"facet", "faceted"},
        {"cluster", "clustered", "crystals", "crystal"},
        {"pixel", "pixelated", "pixelart"},
        {"highlight", "highlighted", "highlights"},
        {"shade", "shaded", "shading"},
        {"oval", "ovoid"},
    )
    for group in synonym_groups:
        if allowed & group:
            allowed.update(group)
    unsupported = tuple(word for word in re.findall(r"[a-z]+", candidate.lower()) if word not in allowed)
    return not unsupported, tuple(sorted(set(unsupported)))


def enrich_description(
    fields: Mapping[str, Any],
    *,
    literal_description: str = "",
    generator: Callable[[Mapping[str, str], str], str] | None = None,
    include_uncertain: bool = False,
) -> DescriptionArtifact:
    facts = accepted_description_facts(fields, include_uncertain=include_uncertain)
    canonical = canonical_description_from_facts(facts)
    if generator is None:
        return DescriptionArtifact(
            canonical_description=canonical, enriched_description=canonical, facts_used=tuple(sorted(facts))
        )
    generated = generator(facts, literal_description)
    valid, unsupported = validate_enriched_description(generated, facts)
    return DescriptionArtifact(
        canonical_description=canonical,
        enriched_description=" ".join(str(generated).split()) if valid else canonical,
        facts_used=tuple(sorted(facts)),
        facts_omitted=tuple(() if literal_description else ("literal_description",)),
        unsupported_claims_detected=unsupported,
        valid=valid,
    )
