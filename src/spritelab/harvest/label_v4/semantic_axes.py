"""Strict semantic axes for risk-aware Labeling v4.

The controlled axes in this module intentionally describe different things:
``domain`` is presentation context, ``category`` is a broad semantic class,
and ``role`` is function.  Object identity and surface aliases remain open-set.
Visual material appearance is kept separate from explicit material evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

SEMANTIC_AXES_VERSION = "semantic_axes_v4.2"

DOMAIN_VALUES: tuple[str, ...] = (
    "inventory_icon",
    "equipment_icon",
    "resource_icon",
    "food_icon",
    "plant_icon",
    "spell_icon",
    "unknown",
)

CATEGORY_VALUES: tuple[str, ...] = (
    "weapon",
    "armor",
    "tool",
    "key",
    "gem",
    "material",
    "plant",
    "food",
    "potion",
    "jewelry",
    "clothing",
    "container",
    "spell",
    "misc_item",
    "unknown",
)

ROLE_VALUES: tuple[str, ...] = (
    "weapon",
    "defensive_equipment",
    "wearable_equipment",
    "crafting_material",
    "resource",
    "consumable",
    "container",
    "quest_item",
    "decorative_item",
    "tool",
    "unknown",
)

# Style is deliberately not a domain.  The set is extensible through a style
# vocabulary version without changing the domain contract.
STYLE_VALUES: tuple[str, ...] = (
    "pixel_art",
    "outlined",
    "isometric",
    "high_contrast",
    "ornate",
    "minimal",
    "fantasy",
)

SILHOUETTE_VALUES: tuple[str, ...] = (
    "round",
    "oval",
    "square",
    "rectangular",
    "triangular",
    "diamond",
    "elongated",
    "compact",
    "irregular",
    "multipart",
    "unknown",
)
ASPECT_VALUES: tuple[str, ...] = ("tall", "wide", "square", "compact", "elongated", "unknown")
ORIENTATION_VALUES: tuple[str, ...] = (
    "front_facing",
    "side_facing",
    "top_down",
    "diagonal",
    "horizontal",
    "vertical",
    "left_facing",
    "right_facing",
    "unknown",
)
STRUCTURE_VALUES: tuple[str, ...] = (
    "solid",
    "hollow",
    "ring_shaped",
    "rimmed",
    "bossed",
    "clustered",
    "layered",
    "articulated",
    "container_like",
    "blade_like",
    "multipart",
    "unknown",
)
EDGE_PROFILE_VALUES: tuple[str, ...] = (
    "smooth",
    "jagged",
    "serrated",
    "pointed",
    "rounded",
    "beveled",
    "pixelated",
    "unknown",
)

COLOR_ROLE_FIELDS: tuple[str, ...] = (
    "primary_colors",
    "secondary_colors",
    "outline_colors",
    "shadow_colors",
    "highlight_colors",
)

_COLOR_ROLE_ALIASES = {
    "primary": "primary_colors",
    "secondary": "secondary_colors",
    "outline": "outline_colors",
    "shadow": "shadow_colors",
    "highlight": "highlight_colors",
}

# These are lexical color families, not a replacement for measured palette
# membership.  They are used only to map phrases such as ``darker brown along
# edges`` onto a compatible color that is already present in the deterministic
# palette.
_COLOR_FAMILIES: tuple[str, ...] = (
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "orange",
    "yellow",
    "green",
    "teal",
    "cyan",
    "blue",
    "purple",
    "violet",
    "pink",
    "brown",
    "tan",
)

# Friendly aliases for callers that prefer the shorter names.
DOMAINS = DOMAIN_VALUES
CATEGORIES = CATEGORY_VALUES
ROLES = ROLE_VALUES


class AxisValidationError(ValueError):
    """Raised when a controlled value is placed on the wrong semantic axis."""


def normalize_semantic_term(value: Any) -> str:
    """Return a stable snake-case token without deciding its semantic axis."""

    text = str(value or "").strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _normalize_many(values: Iterable[Any] | Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        values = (values,)
    result: list[str] = []
    for value in values:
        token = normalize_semantic_term(value)
        if token and token not in result:
            result.append(token)
    return tuple(result)


def validate_axis_value(axis: str, value: Any) -> str:
    """Normalize and validate one controlled axis value.

    Open-set canonical objects and aliases are intentionally not accepted by
    this helper; callers should use :func:`normalize_semantic_term` for them.
    """

    normalized = normalize_semantic_term(value)
    vocabularies = {
        "domain": DOMAIN_VALUES,
        "category": CATEGORY_VALUES,
        "role": ROLE_VALUES,
        "style": STYLE_VALUES,
        "silhouette": SILHOUETTE_VALUES,
        "aspect": ASPECT_VALUES,
        "orientation": ORIENTATION_VALUES,
        "structure": STRUCTURE_VALUES,
        "edge_profile": EDGE_PROFILE_VALUES,
    }
    if axis not in vocabularies:
        raise AxisValidationError(f"unknown controlled semantic axis: {axis}")
    if normalized not in vocabularies[axis]:
        raise AxisValidationError(f"invalid {axis} value: {value!r}")
    return normalized


def is_valid_domain(value: Any) -> bool:
    return normalize_semantic_term(value) in DOMAIN_VALUES


def is_valid_category(value: Any) -> bool:
    return normalize_semantic_term(value) in CATEGORY_VALUES


def is_valid_role(value: Any) -> bool:
    return normalize_semantic_term(value) in ROLE_VALUES


@dataclass(frozen=True)
class MaterialEvidence:
    """Material facts and appearance cues without promotion between them."""

    explicit_material: str | None = None
    visual_material_cue: tuple[str, ...] = ()
    explicit_support: tuple[str, ...] = ()
    visual_support: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        explicit = normalize_semantic_term(self.explicit_material) or None
        object.__setattr__(self, "explicit_material", explicit)
        object.__setattr__(self, "visual_material_cue", _normalize_many(self.visual_material_cue))
        object.__setattr__(self, "explicit_support", _normalize_many(self.explicit_support))
        visual_support = (self.visual_support,) if isinstance(self.visual_support, str) else self.visual_support
        object.__setattr__(self, "visual_support", tuple(str(v) for v in visual_support if str(v).strip()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "explicit_material": self.explicit_material,
            "visual_material_cue": list(self.visual_material_cue),
            "explicit_support": list(self.explicit_support),
            "visual_support": list(self.visual_support),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialEvidence:
        return cls(
            explicit_material=data.get("explicit_material"),
            visual_material_cue=tuple(data.get("visual_material_cue") or ()),
            explicit_support=tuple(data.get("explicit_support") or ()),
            visual_support=tuple(data.get("visual_support") or ()),
        )


@dataclass(frozen=True)
class ShapeAttributes:
    """Multi-axis geometry; ``parts`` remains open-set and visibly grounded."""

    silhouette: tuple[str, ...] = ()
    aspect: tuple[str, ...] = ()
    orientation: tuple[str, ...] = ()
    structure: tuple[str, ...] = ()
    edge_profile: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for axis in ("silhouette", "aspect", "orientation", "structure", "edge_profile"):
            values = _normalize_many(getattr(self, axis))
            for value in values:
                validate_axis_value(axis, value)
            object.__setattr__(self, axis, values)
        object.__setattr__(self, "parts", _normalize_many(self.parts))

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "silhouette": list(self.silhouette),
            "aspect": list(self.aspect),
            "orientation": list(self.orientation),
            "structure": list(self.structure),
            "edge_profile": list(self.edge_profile),
            "parts": list(self.parts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShapeAttributes:
        return cls(**{name: _as_tuple(data.get(name)) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ColorAttributes:
    """Deterministic palette membership plus semantic color roles."""

    palette_colors: tuple[str, ...] = ()
    primary_colors: tuple[str, ...] = ()
    secondary_colors: tuple[str, ...] = ()
    outline_colors: tuple[str, ...] = ()
    shadow_colors: tuple[str, ...] = ()
    highlight_colors: tuple[str, ...] = ()
    filename_color_hints: tuple[str, ...] = ()
    provenance: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "palette_colors",
            "primary_colors",
            "secondary_colors",
            "outline_colors",
            "shadow_colors",
            "highlight_colors",
            "filename_color_hints",
        ):
            object.__setattr__(self, name, _normalize_many(getattr(self, name)))
        object.__setattr__(
            self,
            "provenance",
            {str(key): _normalize_many(value) for key, value in self.provenance.items()},
        )

    def role_membership_conflicts(self) -> tuple[str, ...]:
        """Return role colors absent from the deterministic palette.

        The values are surfaced, not silently deleted.  An empty palette means
        membership is unavailable, so no contradiction can be asserted.
        """

        if not self.palette_colors:
            return ()
        palette = set(self.palette_colors)
        result: list[str] = []
        for role in (
            "primary_colors",
            "secondary_colors",
            "outline_colors",
            "shadow_colors",
            "highlight_colors",
        ):
            for color in getattr(self, role):
                if color not in palette:
                    result.append(f"{role}:{color}:not_in_palette")
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(getattr(self, name))
            for name in (
                "palette_colors",
                "primary_colors",
                "secondary_colors",
                "outline_colors",
                "shadow_colors",
                "highlight_colors",
                "filename_color_hints",
            )
        } | {"provenance": {key: list(value) for key, value in self.provenance.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ColorAttributes:
        names = (
            "palette_colors",
            "primary_colors",
            "secondary_colors",
            "outline_colors",
            "shadow_colors",
            "highlight_colors",
            "filename_color_hints",
        )
        return cls(
            **{name: _as_tuple(data.get(name)) for name in names},
            provenance={str(k): _as_tuple(v) for k, v in (data.get("provenance") or {}).items()},
        )


@dataclass(frozen=True)
class ColorRoleNormalization:
    """Auditable palette-constrained normalization of visual color phrases.

    ``raw_visual_color_roles`` preserves provider wording. ``color_roles``
    contains only deterministic palette members (or a compatible measured
    shade such as ``dark_blue`` for the phrase ``blue``). Anything else is
    retained as a conflict and never enters the structured conditioning
    vocabulary.
    """

    raw_visual_color_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    color_roles: ColorAttributes = field(default_factory=ColorAttributes)
    conflicts: tuple[dict[str, Any], ...] = ()
    role_evidence: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_visual_color_roles": {role: list(values) for role, values in self.raw_visual_color_roles.items()},
            "color_roles": self.color_roles.to_dict(),
            "conflicts": [dict(conflict) for conflict in self.conflicts],
            "role_evidence": {role: [dict(item) for item in items] for role, items in self.role_evidence.items()},
        }


def normalize_visual_color_roles(
    raw_roles: Mapping[str, Any] | None,
    palette_colors: Iterable[Any] | Any,
) -> ColorRoleNormalization:
    """Map free-form VLM color roles onto measured palette members only.

    The raw phrase remains available even when no compatible palette member
    exists.  An absent deterministic palette is treated conservatively: raw
    observations are retained, but no structured role color is emitted.
    """

    palette = _normalize_many(palette_colors)
    raw_by_role: dict[str, tuple[str, ...]] = {}
    normalized_by_role: dict[str, list[str]] = {name: [] for name in COLOR_ROLE_FIELDS}
    conflicts: list[dict[str, Any]] = []
    role_evidence: dict[str, list[dict[str, Any]]] = {name: [] for name in COLOR_ROLE_FIELDS}
    for raw_name, raw_values in dict(raw_roles or {}).items():
        role = _COLOR_ROLE_ALIASES.get(str(raw_name), str(raw_name))
        if role not in COLOR_ROLE_FIELDS:
            continue
        values = _raw_strings(raw_values)
        raw_by_role[role] = values
        for raw_value in values:
            matches = _compatible_palette_colors(raw_value, palette)
            compatible = _select_role_color(role, matches)
            if compatible is None:
                conflicts.append(
                    {
                        "field": "color",
                        "role": role,
                        "code": "color_role_outside_palette",
                        "raw_visual_color": raw_value,
                        "deterministic_palette": list(palette),
                    }
                )
                continue
            ambiguous = len(matches) > 1 and " or " in raw_value.lower()
            role_evidence[role].append(
                {
                    "raw": raw_value,
                    "palette_candidates": list(matches),
                    "selected": compatible,
                    "confidence": 0.65 if ambiguous else 0.9,
                    "policy": "darkest"
                    if role in {"outline_colors", "shadow_colors"}
                    else "lightest"
                    if role == "highlight_colors"
                    else "foreground_compatible",
                }
            )
            if ambiguous and role not in {"outline_colors", "shadow_colors", "highlight_colors"}:
                conflicts.append(
                    {
                        "field": "color",
                        "role": role,
                        "code": "ambiguous_color_disjunction",
                        "raw_visual_color": raw_value,
                        "alternatives": list(matches),
                    }
                )
                continue
            if compatible not in normalized_by_role[role]:
                normalized_by_role[role].append(compatible)

    colors = ColorAttributes(
        palette_colors=palette,
        primary_colors=normalized_by_role["primary_colors"],
        secondary_colors=normalized_by_role["secondary_colors"],
        outline_colors=normalized_by_role["outline_colors"],
        shadow_colors=normalized_by_role["shadow_colors"],
        highlight_colors=normalized_by_role["highlight_colors"],
        provenance=dict.fromkeys(raw_by_role, ("vlm_visual_palette_normalized",)),
    )
    return ColorRoleNormalization(
        raw_visual_color_roles=raw_by_role,
        color_roles=colors,
        conflicts=tuple(conflicts),
        role_evidence={role: tuple(items) for role, items in role_evidence.items() if items},
    )


def _raw_strings(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _compatible_palette_colors(raw_value: str, palette: tuple[str, ...]) -> tuple[str, ...]:
    token = normalize_semantic_term(raw_value)
    if token in palette:
        return (token,)
    if not token or not palette:
        return ()

    candidates: list[tuple[int, str]] = []
    for family in _COLOR_FAMILIES:
        position = token.find(family)
        if position >= 0:
            normalized_family = "gray" if family == "grey" else "purple" if family == "violet" else family
            candidates.append((position, normalized_family))
    result: list[str] = []
    for _position, family in sorted(candidates):
        if family in palette:
            result.append(family)
        result.extend(
            palette_value
            for palette_value in palette
            if family in palette_value.split("_")
            or palette_value.startswith(family + "_")
            or palette_value.endswith("_" + family)
        )
    return tuple(dict.fromkeys(result))


def _select_role_color(role: str, matches: tuple[str, ...]) -> str | None:
    if not matches:
        return None
    darkness = {
        "black": 0,
        "dark_gray": 1,
        "dark_brown": 2,
        "dark_blue": 2,
        "dark_purple": 2,
        "brown": 4,
        "purple": 5,
        "blue": 5,
        "pink": 7,
        "yellow": 8,
        "white": 10,
    }
    if role in {"outline_colors", "shadow_colors"}:
        return min(matches, key=lambda value: (darkness.get(value, 5), value))
    if role == "highlight_colors":
        return max(matches, key=lambda value: (darkness.get(value, 5), value))
    return matches[0]


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)
