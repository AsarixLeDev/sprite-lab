"""Small RGB color helpers used by codec heuristics."""

from __future__ import annotations

import colorsys


def srgb_luminance(rgb: tuple[int, int, int]) -> float:
    """Return simple sRGB luminance normalized to 0..1."""

    red, green, blue = (channel / 255.0 for channel in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def rgb_saturation(rgb: tuple[int, int, int]) -> float:
    """Return RGB saturation normalized to 0..1.

    This uses the lightweight estimate ``(max - min) / max`` and treats black
    as zero saturation.
    """

    red, green, blue = rgb
    highest = max(red, green, blue)
    if highest == 0:
        return 0.0
    lowest = min(red, green, blue)
    return (highest - lowest) / highest


def rgb_hue_degrees(rgb: tuple[int, int, int]) -> float:
    """Return HSV hue in degrees for deterministic color sorting."""

    red, green, blue = (channel / 255.0 for channel in rgb)
    hue, _saturation, _value = colorsys.rgb_to_hsv(red, green, blue)
    return hue * 360.0
