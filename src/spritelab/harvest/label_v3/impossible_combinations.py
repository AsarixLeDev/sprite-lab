"""Auto-Labeling v3: impossible-combination validation rules.

Declarative rules that reject label *combinations* that cannot be true
simultaneously (e.g. a "weapon" category with a food canonical object).

Design contract (important):

- Every rule is *cross-field*. A rule fires only when **all** of its field
  predicates are positively satisfied at once. A single field value on its own
  can never be an "impossible combination".
- A predicate whose field is empty/unknown does **not** match, so a rule can
  never fire on a record with missing fields. This guarantees that clean
  records (where only one field is known, or fields are abstained) are never
  flagged — a hard requirement of the v3 spec (no false hard-rejects).
- Set-valued fields (e.g. ``tags``) match if **any** member is in the trigger
  set.

This replaces an earlier single-field implementation that flagged every valid
``food``/``weapon``/``armor`` sprite as an impossible combination.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Value families used by the cross-field rules.
# ---------------------------------------------------------------------------

WEAPON_CATEGORIES: frozenset[str] = frozenset({"weapon", "bladed_weapon", "ranged_weapon", "blunt_weapon"})
ARMOR_CATEGORIES: frozenset[str] = frozenset({"armor", "head_armor", "body_armor", "shield"})
FOOD_CATEGORIES: frozenset[str] = frozenset({"food", "fruit", "vegetable", "meat"})
PLANT_CATEGORIES: frozenset[str] = frozenset({"plant", "flower", "mushroom", "herb", "tree"})
MATERIAL_CATEGORIES: frozenset[str] = frozenset({"material", "crafting_material", "gem", "mineral"})

WEAPON_OBJECTS: frozenset[str] = frozenset(
    {"sword", "dagger", "knife", "blade", "axe", "hatchet", "bow", "arrow", "hammer", "mace", "spear", "lance", "pike"}
)
FOOD_OBJECTS: frozenset[str] = frozenset(
    {"apple", "bread", "cheese", "meat", "fish", "carrot", "banana", "berry", "cake", "egg", "corn", "steak"}
)

# Materials that are structurally incompatible with certain categories.
METALLIC_MATERIALS: frozenset[str] = frozenset({"metal"})
LIQUID_MATERIALS: frozenset[str] = frozenset({"liquid"})


@dataclass(frozen=True)
class ImpossibleCombinationRule:
    """A cross-field impossibility.

    ``when_all`` maps a field name to the set of values that satisfy that
    field's predicate. The rule fires only when **every** predicate is
    satisfied simultaneously; a predicate over a missing/empty field never
    matches, so partially-known records cannot be flagged.
    """

    rule_id: str
    description: str
    when_all: Mapping[str, frozenset[str]]
    severity: str = "fatal"

    def check(self, field_values: Mapping[str, str | tuple[str, ...]]) -> str | None:
        """Return a violation description if the rule fires, else ``None``."""
        for field_name, trigger_values in self.when_all.items():
            value = field_values.get(field_name)
            if value is None or value == "" or value == ():
                # Unknown field -> predicate cannot be confirmed -> no violation.
                return None
            if isinstance(value, str):
                if value not in trigger_values:
                    return None
            elif isinstance(value, (tuple, list, set, frozenset)):
                if not any(v in trigger_values for v in value):
                    return None
            else:
                return None
        # Every predicate matched.
        return f"{self.rule_id}: {self.description}"


IMPOSSIBLE_COMBINATIONS: tuple[ImpossibleCombinationRule, ...] = (
    ImpossibleCombinationRule(
        rule_id="IC001",
        description="Weapon category cannot have a food canonical object",
        when_all={"category": WEAPON_CATEGORIES, "canonical_object": FOOD_OBJECTS},
    ),
    ImpossibleCombinationRule(
        rule_id="IC002",
        description="Food category cannot be made of metal",
        when_all={"category": FOOD_CATEGORIES, "material": METALLIC_MATERIALS},
    ),
    ImpossibleCombinationRule(
        rule_id="IC003",
        description="Food category cannot have a weapon canonical object",
        when_all={"category": FOOD_CATEGORIES, "canonical_object": WEAPON_OBJECTS},
    ),
    ImpossibleCombinationRule(
        rule_id="IC004",
        description="Armor category cannot have a food canonical object",
        when_all={"category": ARMOR_CATEGORIES, "canonical_object": FOOD_OBJECTS},
    ),
    ImpossibleCombinationRule(
        rule_id="IC005",
        description="Weapon category cannot be a liquid",
        when_all={"category": WEAPON_CATEGORIES, "material": LIQUID_MATERIALS},
    ),
    ImpossibleCombinationRule(
        rule_id="IC006",
        description="Plant category cannot be made of metal",
        when_all={"category": PLANT_CATEGORIES, "material": METALLIC_MATERIALS},
    ),
    ImpossibleCombinationRule(
        rule_id="IC007",
        description="Raw-material category cannot have a weapon canonical object",
        when_all={"category": MATERIAL_CATEGORIES, "canonical_object": WEAPON_OBJECTS},
    ),
)


def validate_impossible_combinations(
    category: str = "",
    canonical_object: str = "",
    material: str = "",
    shape: str = "",
    tags: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(violation_codes, violation_descriptions)``.

    A record with only one known field can never produce a violation: every
    rule requires at least two concrete, conflicting fields.
    """

    field_values: dict[str, str | tuple[str, ...]] = {
        "category": category,
        "canonical_object": canonical_object,
        "material": material,
        "shape": shape,
        "tags": tags,
    }

    codes: list[str] = []
    descriptions: list[str] = []
    for rule in IMPOSSIBLE_COMBINATIONS:
        violation = rule.check(field_values)
        if violation is not None:
            codes.append(rule.rule_id)
            descriptions.append(violation)

    return tuple(codes), tuple(descriptions)


def impossible_combinations_hash() -> str:
    import hashlib

    data = ";".join(
        f"{r.rule_id}:{sorted((k, sorted(v)) for k, v in r.when_all.items())}"
        for r in sorted(IMPOSSIBLE_COMBINATIONS, key=lambda x: x.rule_id)
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]
