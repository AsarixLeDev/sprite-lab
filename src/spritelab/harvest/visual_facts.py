"""Deterministic visual facts extracted from 32x32 sprite PNGs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.codec.color_names import color_name


@dataclass(frozen=True)
class VisualFacts:
    content_bbox: tuple[int, int, int, int] | None
    content_width: int
    content_height: int
    opaque_pixel_count: int
    alpha_hard: bool
    palette_size: int
    dominant_colors: tuple[str, ...]
    aspect_hint: str
    shape_hints: tuple[str, ...]


def extract_visual_facts_from_png(path: Path) -> VisualFacts:
    """Extract deterministic facts from a sprite PNG."""

    rgba = _load_rgba(path)
    pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = pixels[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    opaque_count = int(xs.size)
    alpha_values = {int(value) for value in np.unique(alpha)}
    alpha_hard = alpha_values <= {0, 255}

    if opaque_count == 0:
        return VisualFacts(
            content_bbox=None,
            content_width=0,
            content_height=0,
            opaque_pixel_count=0,
            alpha_hard=alpha_hard,
            palette_size=0,
            dominant_colors=(),
            aspect_hint="empty",
            shape_hints=("empty",),
        )

    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    bbox = (left, top, right, bottom)
    content_width = right - left
    content_height = bottom - top
    opaque_rgb = pixels[alpha > 0, :3]
    palette_size = int(np.unique(opaque_rgb, axis=0).shape[0])
    shape_hints = shape_hints_from_alpha(path)
    aspect_hint = _aspect_hint(content_width, content_height, opaque_count, rgba.size)
    return VisualFacts(
        content_bbox=bbox,
        content_width=content_width,
        content_height=content_height,
        opaque_pixel_count=opaque_count,
        alpha_hard=alpha_hard,
        palette_size=palette_size,
        dominant_colors=dominant_color_names_from_rgba(path),
        aspect_hint=aspect_hint,
        shape_hints=shape_hints,
    )


def dominant_color_names_from_rgba(path: Path, top_k: int = 4) -> tuple[str, ...]:
    """Return deterministic dominant opaque color names from a PNG."""

    rgba = _load_rgba(path)
    pixels = np.asarray(rgba, dtype=np.uint8)
    opaque_rgb = pixels[pixels[:, :, 3] > 0, :3]
    if opaque_rgb.size == 0:
        return ()
    colors, counts = np.unique(opaque_rgb, axis=0, return_counts=True)
    name_counts: dict[str, int] = {}
    for rgb, count in zip(colors, counts, strict=False):
        name = color_name(rgb)
        name_counts[name] = name_counts.get(name, 0) + int(count)
    ranked = sorted(name_counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(name_counts.values())
    result = [
        name for index, (name, count) in enumerate(ranked[: max(1, int(top_k))]) if index == 0 or count / total >= 0.08
    ]
    return tuple(result)


def shape_hints_from_alpha(path: Path) -> tuple[str, ...]:
    """Return rough deterministic shape hints from alpha coverage."""

    rgba = _load_rgba(path)
    alpha = np.asarray(rgba, dtype=np.uint8)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    if xs.size == 0:
        return ("empty",)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    opaque = int(xs.size)
    bbox_area = max(1, width * height)
    density = opaque / bbox_area
    hints: list[str] = []

    ratio = width / max(1, height)
    if width <= 12 and height <= 12:
        hints.append("small_content")
    if width >= rgba.width - 2 and height >= rgba.height - 2:
        hints.append("full_canvas")
    if min(width, height) <= 3 or density < 0.22:
        hints.append("thin")
    if ratio >= 1.55:
        hints.append("wide")
    elif ratio <= 0.65:
        hints.append("tall")
    elif 0.75 <= ratio <= 1.33 and density >= 0.45:
        hints.append("roundish")
    if not hints:
        hints.append("compact")
    return tuple(dict.fromkeys(hints))


def visual_facts_to_json(facts: VisualFacts | None) -> dict[str, Any] | None:
    if facts is None:
        return None
    return {
        "content_bbox": list(facts.content_bbox) if facts.content_bbox is not None else None,
        "content_width": facts.content_width,
        "content_height": facts.content_height,
        "opaque_pixel_count": facts.opaque_pixel_count,
        "alpha_hard": facts.alpha_hard,
        "palette_size": facts.palette_size,
        "dominant_colors": list(facts.dominant_colors),
        "aspect_hint": facts.aspect_hint,
        "shape_hints": list(facts.shape_hints),
    }


def visual_facts_from_json(data: dict[str, Any] | None) -> VisualFacts | None:
    if not isinstance(data, dict):
        return None
    bbox = data.get("content_bbox")
    return VisualFacts(
        content_bbox=tuple(int(value) for value in bbox) if isinstance(bbox, list | tuple) and len(bbox) == 4 else None,
        content_width=int(data.get("content_width") or 0),
        content_height=int(data.get("content_height") or 0),
        opaque_pixel_count=int(data.get("opaque_pixel_count") or 0),
        alpha_hard=bool(data.get("alpha_hard", True)),
        palette_size=int(data.get("palette_size") or 0),
        dominant_colors=tuple(str(value) for value in data.get("dominant_colors") or ()),
        aspect_hint=str(data.get("aspect_hint") or ""),
        shape_hints=tuple(str(value) for value in data.get("shape_hints") or ()),
    )


def _aspect_hint(width: int, height: int, opaque_count: int, canvas_size: tuple[int, int]) -> str:
    if width <= 0 or height <= 0 or opaque_count <= 0:
        return "empty"
    ratio = width / max(1, height)
    if min(width, height) <= 3:
        return "thin"
    if width <= 12 and height <= 12:
        return "small_content"
    if width >= canvas_size[0] - 2 and height >= canvas_size[1] - 2:
        return "full_canvas"
    if ratio >= 1.55:
        return "wide"
    if ratio <= 0.65:
        return "tall"
    return "roundish"


def _load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA")
