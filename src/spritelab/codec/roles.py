"""Stable semantic color-role constants."""

from __future__ import annotations

from enum import IntEnum


class ColorRole(IntEnum):
    """Semantic role IDs used by optional 32x32 role maps."""

    TRANSPARENT = 0
    OUTLINE = 1
    DEEP_SHADOW = 2
    SHADOW = 3
    MIDTONE = 4
    LIGHT = 5
    HIGHLIGHT = 6
    ACCENT = 7
    EMISSIVE = 8
    TEXTURE_DETAIL = 9
    UNKNOWN = 255


ROLE_TRANSPARENT = int(ColorRole.TRANSPARENT)
ROLE_OUTLINE = int(ColorRole.OUTLINE)
ROLE_DEEP_SHADOW = int(ColorRole.DEEP_SHADOW)
ROLE_SHADOW = int(ColorRole.SHADOW)
ROLE_MIDTONE = int(ColorRole.MIDTONE)
ROLE_LIGHT = int(ColorRole.LIGHT)
ROLE_HIGHLIGHT = int(ColorRole.HIGHLIGHT)
ROLE_ACCENT = int(ColorRole.ACCENT)
ROLE_EMISSIVE = int(ColorRole.EMISSIVE)
ROLE_TEXTURE_DETAIL = int(ColorRole.TEXTURE_DETAIL)
ROLE_UNKNOWN = int(ColorRole.UNKNOWN)

ROLE_NAMES: dict[int, str] = {
    ROLE_TRANSPARENT: "transparent",
    ROLE_OUTLINE: "outline",
    ROLE_DEEP_SHADOW: "deep_shadow",
    ROLE_SHADOW: "shadow",
    ROLE_MIDTONE: "midtone",
    ROLE_LIGHT: "light",
    ROLE_HIGHLIGHT: "highlight",
    ROLE_ACCENT: "accent",
    ROLE_EMISSIVE: "emissive",
    ROLE_TEXTURE_DETAIL: "texture_detail",
    ROLE_UNKNOWN: "unknown",
}

ROLE_PREVIEW_COLORS: dict[int, tuple[int, int, int, int]] = {
    ROLE_TRANSPARENT: (0, 0, 0, 0),
    ROLE_OUTLINE: (20, 20, 28, 255),
    ROLE_DEEP_SHADOW: (55, 40, 80, 255),
    ROLE_SHADOW: (85, 70, 130, 255),
    ROLE_MIDTONE: (130, 130, 150, 255),
    ROLE_LIGHT: (190, 190, 210, 255),
    ROLE_HIGHLIGHT: (255, 255, 255, 255),
    ROLE_ACCENT: (255, 80, 160, 255),
    ROLE_EMISSIVE: (80, 240, 255, 255),
    ROLE_TEXTURE_DETAIL: (240, 180, 60, 255),
    ROLE_UNKNOWN: (255, 0, 255, 255),
}


def role_name(role_id: int) -> str:
    """Return a stable display name for a role ID."""

    return ROLE_NAMES.get(role_id, ROLE_NAMES[int(ColorRole.UNKNOWN)])
