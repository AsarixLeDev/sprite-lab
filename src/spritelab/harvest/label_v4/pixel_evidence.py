"""Deterministic image identity, palette membership, and weak geometry facts."""

from __future__ import annotations

import colorsys
import hashlib
import struct
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.label_v4.cache import exact_image_content_hash

PIXEL_EVIDENCE_VERSION = "pixel_evidence_v1.0"


def exact_rgba_content_hash(path: str | Path) -> str:
    """Hash decoded dimensions and exact RGBA bytes for image-dependent keys."""

    return exact_image_content_hash(path)


def alpha_mask_content_hash(path: str | Path) -> str:
    """Hash dimensions plus the binary visible-pixel mask for family audits."""

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        alpha = bytes(1 if value > 0 else 0 for value in _flattened_pixels(rgba.getchannel("A")))
        payload = struct.pack(">II", rgba.width, rgba.height) + alpha
    return hashlib.sha256(payload).hexdigest()


def analyze_pixels(path: str | Path) -> dict[str, Any]:
    """Extract deterministic palette membership and weak geometric evidence."""

    image_path = Path(path)
    with Image.open(image_path) as source:
        rgba = source.convert("RGBA")
        width, height = rgba.size
        pixels = _flattened_pixels(rgba)

    visible = [pixel for pixel in pixels if pixel[3] > 0]
    opaque = [pixel for pixel in pixels if pixel[3] >= 128]
    exact_counts = Counter((r, g, b, a) for r, g, b, a in visible)
    rgb_counts = Counter((r, g, b) for r, g, b, _a in visible)
    palette_rgba = [
        {"rgba": list(color), "hex": _hex(color[:3], color[3]), "count": count}
        for color, count in sorted(exact_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    named_counts: Counter[str] = Counter()
    for rgb, count in rgb_counts.items():
        named_counts[color_name(rgb)] += count
    palette_colors = [name for name, _count in named_counts.most_common()]

    coordinates = [(index % width, index // width) for index, pixel in enumerate(pixels) if pixel[3] >= 128]
    if coordinates:
        xs = [x for x, _ in coordinates]
        ys = [y for _, y in coordinates]
        bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
        content_width = bbox[2] - bbox[0]
        content_height = bbox[3] - bbox[1]
    else:
        bbox = [0, 0, 0, 0]
        content_width = 0
        content_height = 0

    aspect = _aspect(content_width, content_height)
    orientation = _orientation(coordinates, bbox)
    silhouette = _silhouette(coordinates, bbox)
    quality_signals: list[str] = []
    if not visible:
        quality_signals.append("blank_image")
    if visible and len(opaque) / len(visible) < 0.5:
        quality_signals.append("mostly_translucent")
    if content_width <= 2 or content_height <= 2:
        quality_signals.append("tiny_fragment")

    return {
        "schema_version": PIXEL_EVIDENCE_VERSION,
        "image_path": str(image_path),
        "image_hash": exact_rgba_content_hash(image_path),
        "alpha_mask_hash": alpha_mask_content_hash(image_path),
        "width": width,
        "height": height,
        "visible_pixel_count": len(visible),
        "opaque_pixel_count": len(opaque),
        "foreground_fraction": len(visible) / max(1, width * height),
        "bbox": bbox,
        "content_width": content_width,
        "content_height": content_height,
        # Exact palette membership is authoritative.  Named colors are a
        # deterministic convenience projection and retain their RGBA source.
        "palette_rgba": palette_rgba,
        "palette_colors": palette_colors,
        "dominant_color_counts": dict(named_counts.most_common()),
        "shape": {
            "silhouette": [silhouette] if silhouette else [],
            "aspect": [aspect] if aspect else [],
            "orientation": [orientation] if orientation else [],
            "structure": [],
            "edge_profile": [],
            "parts": [],
        },
        "quality_signals": quality_signals,
    }


def color_name(rgb: tuple[int, int, int] | list[int]) -> str:
    """Deterministically map RGB to a small audit-friendly color vocabulary."""

    r, g, b = (max(0, min(255, int(value))) for value in rgb)
    maximum = max(r, g, b)
    minimum = min(r, g, b)
    if maximum <= 28:
        return "black"
    if minimum >= 232:
        return "white"
    saturation = 0.0 if maximum == 0 else (maximum - minimum) / maximum
    value = maximum / 255.0
    if saturation < 0.12:
        if value < 0.28:
            return "dark_gray"
        if value > 0.78:
            return "light_gray"
        return "gray"

    hue, saturation_hsv, value_hsv = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    degrees = hue * 360.0
    if value_hsv < 0.25:
        prefix = "dark_"
    elif value_hsv > 0.82 and saturation_hsv < 0.55:
        prefix = "light_"
    else:
        prefix = ""
    if degrees < 15 or degrees >= 345:
        base = "red"
    elif degrees < 42:
        base = "orange"
    elif degrees < 68:
        base = "yellow"
    elif degrees < 155:
        base = "green"
    elif degrees < 190:
        base = "teal"
    elif degrees < 255:
        base = "blue"
    elif degrees < 285:
        base = "purple"
    elif degrees < 330:
        base = "pink"
    else:
        base = "red"
    if base == "orange" and value_hsv < 0.55:
        return "brown"
    return prefix + base


def palette_membership_supports(color: str, pixel_evidence: Mapping[str, Any]) -> bool:
    return str(color).strip().lower() in {str(value).lower() for value in pixel_evidence.get("palette_colors") or ()}


def _aspect(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return ""
    ratio = width / height
    if ratio >= 1.35:
        return "wide"
    if ratio <= 0.74:
        return "tall"
    return "compact"


def _orientation(coordinates: list[tuple[int, int]], bbox: list[int]) -> str:
    if not coordinates:
        return ""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width >= height * 1.35:
        return "horizontal"
    if height >= width * 1.35:
        return "vertical"
    return "front_facing_or_unknown"


def _silhouette(coordinates: list[tuple[int, int]], bbox: list[int]) -> str:
    if not coordinates:
        return ""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = max(1, width * height)
    fill = len(coordinates) / area
    ratio = width / max(1, height)
    if 0.82 <= ratio <= 1.22 and 0.58 <= fill <= 0.90:
        return "round_or_compact"
    if ratio >= 1.6:
        return "elongated_horizontal"
    if ratio <= 0.625:
        return "elongated_vertical"
    if fill < 0.35:
        return "open_or_fragmented"
    return "irregular_compact"


def _hex(rgb: tuple[int, int, int], alpha: int) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}{alpha:02x}"


def _flattened_pixels(image: Image.Image) -> list[Any]:
    modern = getattr(image, "get_flattened_data", None)
    return list(modern() if modern is not None else image.getdata())
