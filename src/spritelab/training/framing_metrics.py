"""Shared deterministic sprite framing metrics."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SPRITE_SIZE = 32
CONNECTIVITY = "8-neighbor"


def compute_alpha_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Compute deterministic alpha/silhouette metrics from a 32x32 boolean mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"alpha mask must be 32x32, got {mask.shape}")

    opaque = int(np.count_nonzero(mask))
    metrics: dict[str, Any] = {
        "opaque_pixels": opaque,
        "alpha_coverage": opaque / float(SPRITE_SIZE * SPRITE_SIZE),
        "bounding_box": None,
        "bbox_width": 0,
        "bbox_height": 0,
        "bbox_area": 0,
        "bbox_fill_ratio": 0.0,
        "center_of_mass_x": None,
        "center_of_mass_y": None,
        "center_offset_from_image_center": None,
        "touches_border": False,
    }
    if opaque == 0:
        return metrics

    ys, xs = np.nonzero(mask)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    bbox_area = width * height
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    image_center = (SPRITE_SIZE - 1) / 2.0
    metrics.update(
        {
            "bounding_box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            "bbox_width": int(width),
            "bbox_height": int(height),
            "bbox_area": int(bbox_area),
            "bbox_fill_ratio": opaque / float(bbox_area) if bbox_area else 0.0,
            "center_of_mass_x": center_x,
            "center_of_mass_y": center_y,
            "center_offset_from_image_center": float(math.hypot(center_x - image_center, center_y - image_center)),
            "touches_border": bool(
                np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
            ),
        }
    )
    return metrics


def compute_connected_components(mask: np.ndarray) -> dict[str, Any]:
    """Compute 8-neighbor connected-component metrics from a 32x32 mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"alpha mask must be 32x32, got {mask.shape}")

    visited = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    for y in range(SPRITE_SIZE):
        for x in range(SPRITE_SIZE):
            if not mask[y, x] or visited[y, x]:
                continue
            sizes.append(_flood_component(mask, visited, y, x))

    opaque = int(np.count_nonzero(mask))
    largest = max(sizes) if sizes else 0
    largest_ratio = largest / float(opaque) if opaque else 0.0
    return {
        "connected_components": len(sizes),
        "largest_component_pixels": int(largest),
        "largest_component_ratio": float(largest_ratio),
        "small_component_count": int(sum(1 for size in sizes if size <= 2)),
        "single_pixel_islands": int(sum(1 for size in sizes if size == 1)),
        "fragmentation_score": float(1.0 - largest_ratio) if opaque and len(sizes) > 1 else 0.0,
    }


def compute_color_metrics(image_or_rgba: Image.Image | np.ndarray | None) -> dict[str, Any]:
    """Compute visible-color distribution metrics from PIL or RGBA array input."""

    if image_or_rgba is None:
        return {
            "visible_color_count": 0,
            "dominant_color_ratio": 0.0,
            "palette_entropy": 0.0,
            "rare_color_count": 0,
            "transparent_index_used": False,
        }

    rgba = image_to_rgba_array(image_or_rgba)
    visible = rgba[..., 3] > 0
    if not bool(np.any(visible)):
        return {
            "visible_color_count": 0,
            "dominant_color_ratio": 0.0,
            "palette_entropy": 0.0,
            "rare_color_count": 0,
            "transparent_index_used": transparent_index_used(image_or_rgba),
        }

    colors, counts = np.unique(rgba[..., :3][visible], axis=0, return_counts=True)
    total = int(counts.sum())
    probs = counts.astype(np.float64) / float(total)
    entropy = float(-np.sum(probs * np.log2(probs))) if total else 0.0
    return {
        "visible_color_count": int(colors.shape[0]),
        "dominant_color_ratio": float(int(counts.max()) / float(total)) if total else 0.0,
        "palette_entropy": entropy,
        "rare_color_count": int(np.count_nonzero(counts <= 2)),
        "transparent_index_used": transparent_index_used(image_or_rgba),
    }


def compute_edge_metrics(mask: np.ndarray, image_or_rgba: Image.Image | np.ndarray | None) -> dict[str, Any]:
    """Compute simple alpha-boundary and visible RGB edge metrics."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"alpha mask must be 32x32, got {mask.shape}")

    opaque = int(np.count_nonzero(mask))
    padded = np.pad(mask, 1, constant_values=False)
    neighbors_full = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    edge_pixels = int(np.count_nonzero(mask & ~neighbors_full))
    return {
        "alpha_edge_pixels": edge_pixels,
        "alpha_edge_density": float(edge_pixels / float(opaque)) if opaque else 0.0,
        "rgb_edge_density_visible": rgb_edge_density_visible(image_or_rgba),
    }


def compute_sprite_framing_metrics(image_or_rgba: Image.Image | np.ndarray | None) -> dict[str, Any]:
    """Compute combined source/generated structural metrics for a 32x32 sprite."""

    mask = alpha_mask_from_image(image_or_rgba)
    metrics: dict[str, Any] = {}
    metrics.update(compute_alpha_metrics(mask))
    metrics.update(compute_connected_components(mask))
    metrics.update(compute_color_metrics(image_or_rgba))
    metrics.update(compute_edge_metrics(mask, image_or_rgba))
    return jsonable(metrics)


def alpha_mask_from_image(image_or_rgba: Image.Image | np.ndarray | None) -> np.ndarray:
    if image_or_rgba is None:
        return np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    return image_to_rgba_array(image_or_rgba)[..., 3] > 0


def image_to_rgba_array(image_or_rgba: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image_or_rgba, Image.Image):
        return np.asarray(image_or_rgba.convert("RGBA"), dtype=np.uint8)

    arr = np.asarray(image_or_rgba)
    if arr.shape == (4, SPRITE_SIZE, SPRITE_SIZE):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape != (SPRITE_SIZE, SPRITE_SIZE, 4):
        raise ValueError(f"RGBA image must have shape [32, 32, 4] or [4, 32, 32], got {arr.shape}")
    value = arr.astype(np.float32, copy=False)
    if arr.dtype.kind in "f" and value.size and float(np.nanmax(value)) <= 1.0:
        value = value * 255.0
    return np.rint(np.clip(value, 0.0, 255.0)).astype(np.uint8)


def transparent_index_used(image_or_rgba: Image.Image | np.ndarray) -> bool:
    if isinstance(image_or_rgba, Image.Image) and image_or_rgba.mode == "P":
        transparency = image_or_rgba.info.get("transparency")
        index = np.asarray(image_or_rgba, dtype=np.uint8)
        if isinstance(transparency, int):
            return bool(np.any(index == int(transparency)))
        if isinstance(transparency, (bytes, bytearray)):
            alpha = np.frombuffer(transparency, dtype=np.uint8)
            if alpha.size:
                transparent = np.nonzero(alpha == 0)[0]
                return bool(transparent.size and np.any(np.isin(index, transparent)))
    return bool(np.any(image_to_rgba_array(image_or_rgba)[..., 3] == 0))


def rgb_edge_density_visible(image_or_rgba: Image.Image | np.ndarray | None) -> float:
    if image_or_rgba is None:
        return 0.0
    rgba = image_to_rgba_array(image_or_rgba)
    rgb = rgba[..., :3]
    visible = rgba[..., 3] > 0
    horizontal = visible[:, :-1] & visible[:, 1:]
    vertical = visible[:-1, :] & visible[1:, :]
    pair_count = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    if pair_count == 0:
        return 0.0
    h_changed = np.any(rgb[:, :-1, :] != rgb[:, 1:, :], axis=2) & horizontal
    v_changed = np.any(rgb[:-1, :, :] != rgb[1:, :, :], axis=2) & vertical
    changed = int(np.count_nonzero(h_changed) + np.count_nonzero(v_changed))
    return float(changed / float(pair_count))


def checkerboard_rgba(image: Image.Image, *, tile: int = 4) -> Image.Image:
    rgba = image.convert("RGBA")
    board = Image.new("RGBA", rgba.size, (208, 208, 208, 255))
    dark = Image.new("RGBA", (tile, tile), (164, 164, 164, 255))
    for y in range(0, rgba.height, tile):
        for x in range(0, rgba.width, tile):
            if ((x // tile) + (y // tile)) % 2:
                board.alpha_composite(dark, (x, y))
    board.alpha_composite(rgba)
    return board


def rgba_array_to_image(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(image_to_rgba_array(rgba), mode="RGBA")


def load_png(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image.copy()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _flood_component(mask: np.ndarray, visited: np.ndarray, start_y: int, start_x: int) -> int:
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
    visited[start_y, start_x] = True
    size = 0
    while queue:
        y, x = queue.popleft()
        size += 1
        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= SPRITE_SIZE or xx < 0 or xx >= SPRITE_SIZE:
                continue
            if mask[yy, xx] and not visited[yy, xx]:
                visited[yy, xx] = True
                queue.append((yy, xx))
    return size
