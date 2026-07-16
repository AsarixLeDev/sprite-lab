"""Deterministic, CPU-only image and batch metrics for generation benchmark v1."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SIZE = 32
REQUIRED_METADATA = (
    "sample_id",
    "prompt_id",
    "prompt",
    "checkpoint",
    "seed",
    "noise_seed",
    "steps",
    "cfg_scale",
    "model_output_finite",
)


def rgba_hash(rgba: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()).hexdigest()


def alpha_hash(rgba: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgba[..., 3] > 0).tobytes()).hexdigest()


def normalized_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((SIZE, SIZE), dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return out
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    out[: crop.shape[0], : crop.shape[1]] = crop
    return out


def normalized_alpha_hash(rgba: np.ndarray) -> str:
    return hashlib.sha256(normalized_mask(rgba[..., 3] > 0).tobytes()).hexdigest()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for y, x in zip(*np.nonzero(mask), strict=True):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue = deque([(int(y), int(x))])
        component: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        result.append(component)
    return result


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    interior = padded[1:-1, 1:-1]
    surrounded = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return interior & ~surrounded


def _rgb_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a.astype(np.float32) - b.astype(np.float32)) ** 2, axis=-1)) / math.sqrt(3 * 255**2)


def _nearest_palette_distance(colors: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(colors) or not len(target):
        return np.ones(len(colors), dtype=np.float32)
    delta = colors[:, None, :].astype(np.float32) - target[None, :, :].astype(np.float32)
    return np.sqrt(np.sum(delta**2, axis=2)).min(axis=1) / math.sqrt(3 * 255**2)


def _target_palette(metadata: Mapping[str, Any]) -> np.ndarray | None:
    raw = metadata.get("target_palette") or metadata.get("palette_condition")
    if not isinstance(raw, list) or not raw:
        return None
    try:
        arr = np.asarray(raw, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    return arr[:, :3] if arr.ndim == 2 and arr.shape[1] >= 3 else None


def score_rgba(rgba: np.ndarray, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score one decoded RGBA array. Callers handle corrupt/non-PNG files."""
    metadata = metadata or {}
    rgba = np.asarray(rgba)
    dimensions_ok = rgba.shape == (SIZE, SIZE, 4)
    numeric = np.issubdtype(rgba.dtype, np.number)
    finite = bool(np.isfinite(rgba).all()) if numeric else False
    range_ok = bool(rgba.min() >= 0 and rgba.max() <= 255) if finite and rgba.size else False
    missing_metadata = [key for key in REQUIRED_METADATA if metadata.get(key) in (None, "")]
    hard = {
        "valid_png_rgba": bool(dimensions_ok and numeric),
        "exact_dimensions": bool(dimensions_ok),
        "alpha_range_valid": bool(range_ok),
        "fully_transparent": True,
        "fully_opaque_rectangle": False,
        "corrupt_output": False,
        "model_output_finite": bool(metadata.get("model_output_finite", finite)),
        "unexpected_interpolation": False,
        "wrong_export_scaling": False,
        "missing_generation_metadata": bool(missing_metadata),
        "missing_metadata_fields": missing_metadata,
    }
    if not dimensions_ok or not finite:
        hard["pass"] = False
        return {"hard_validity": hard, "pixel_art": {}}
    arr = np.clip(rgba, 0, 255).astype(np.uint8)
    alpha = arr[..., 3]
    mask = alpha > 0
    visible = arr[..., :3][mask]
    hard["fully_transparent"] = not bool(mask.any())
    hard["fully_opaque_rectangle"] = bool(np.all(alpha == 255))
    semi = (alpha > 0) & (alpha < 255)
    hard["unexpected_interpolation"] = _looks_interpolated(arr)
    hard["wrong_export_scaling"] = _looks_block_scaled(arr)
    hard["pass"] = not any(
        (
            hard["fully_transparent"],
            hard["fully_opaque_rectangle"],
            hard["corrupt_output"],
            not hard["model_output_finite"],
            hard["unexpected_interpolation"],
            hard["wrong_export_scaling"],
            hard["missing_generation_metadata"],
        )
    )

    colors, counts = (
        np.unique(visible, axis=0, return_counts=True) if len(visible) else (np.empty((0, 3)), np.array([]))
    )
    concentration = float(np.max(counts) / np.sum(counts)) if len(counts) else 0.0
    components = _components(mask)
    sizes = sorted((len(c) for c in components), reverse=True)
    edge = _edge_mask(mask)
    transition: list[float] = []
    for axis in (0, 1):
        left = np.take(arr[..., :3], range(SIZE - 1), axis=axis)
        right = np.take(arr[..., :3], range(1, SIZE), axis=axis)
        lm = np.take(mask, range(SIZE - 1), axis=axis)
        rm = np.take(mask, range(1, SIZE), axis=axis)
        use = lm & rm
        if use.any():
            transition.extend(_rgb_distance(left, right)[use].tolist())
    boundary_alpha = semi & edge
    ys, xs = np.nonzero(mask)
    bbox_area = 0
    empty_padding = 1.0
    if len(xs):
        bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        empty_padding = 1.0 - bbox_area / float(SIZE * SIZE)
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    high_freq = _high_frequency_noise(arr[..., :3], mask)
    target = _target_palette(metadata)
    adherence = None
    if target is not None:
        adherence = (
            float(np.mean(_nearest_palette_distance(visible, target) <= (12.0 / 255.0))) if len(visible) else 0.0
        )
    shadow_components = _disconnected_shadow_components(arr, components)
    compactness = None
    if mask.any() and edge.any():
        compactness = float(4.0 * math.pi * mask.sum() / max(1.0, float(edge.sum()) ** 2))
    pixel = {
        "unique_palette_size": len(colors),
        "palette_concentration": concentration,
        "semi_transparent_pixel_ratio": float(semi.mean()),
        "antialiased_edge_ratio": float(boundary_alpha.sum() / max(1, edge.sum())),
        "local_color_transition_sharpness": float(np.mean(transition)) if transition else 0.0,
        "isolated_pixel_noise": int(sum(size == 1 for size in sizes)),
        "small_component_count": int(sum(size <= 3 for size in sizes)),
        "silhouette_occupancy": float(mask.mean()),
        "border_clipping": bool(np.any(mask & border)),
        "border_pixel_ratio": float(np.sum(mask & border) / max(1, mask.sum())),
        "empty_padding": empty_padding,
        "foreground_fragmentation": 0.0 if not sizes else float(1.0 - sizes[0] / max(1, sum(sizes))),
        "connected_component_count": len(components),
        "disconnected_shadow_components": shadow_components,
        "palette_adherence": adherence,
        "alpha_mask_compactness": compactness,
        "high_frequency_pixel_noise": high_freq,
        "rgba_sha256": rgba_hash(arr),
        "alpha_sha256": alpha_hash(arr),
        "normalized_alpha_sha256": normalized_alpha_hash(arr),
        "bbox_area": bbox_area,
    }
    return {"hard_validity": hard, "pixel_art": pixel}


def score_image(path: Path, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            is_png = image.format == "PNG"
            size_ok = image.size == (SIZE, SIZE)
            arr = np.asarray(image.convert("RGBA"))
    except Exception as exc:
        return {
            "hard_validity": {
                "valid_png_rgba": False,
                "exact_dimensions": False,
                "alpha_range_valid": False,
                "fully_transparent": False,
                "fully_opaque_rectangle": False,
                "corrupt_output": True,
                "model_output_finite": bool((metadata or {}).get("model_output_finite", True)),
                "unexpected_interpolation": False,
                "wrong_export_scaling": False,
                "missing_generation_metadata": True,
                "missing_metadata_fields": list(REQUIRED_METADATA),
                "error": str(exc),
                "pass": False,
            },
            "pixel_art": {},
        }
    result = score_rgba(arr, metadata)
    # Indexed PNG with a transparency chunk is a lossless RGBA export too; evaluate
    # decoded RGBA semantics rather than rejecting the repository's canonical P mode.
    result["hard_validity"]["valid_png_rgba"] = bool(is_png and arr.shape[-1] == 4)
    result["hard_validity"]["exact_dimensions"] = size_ok
    if not is_png or not size_ok:
        result["hard_validity"]["pass"] = False
    return result


def _looks_block_scaled(arr: np.ndarray) -> bool:
    # A native sprite can coincidentally contain blocks; require exact repetition across every block.
    for factor in (4, 2):
        reduced = arr[::factor, ::factor]
        rebuilt = np.repeat(np.repeat(reduced, factor, axis=0), factor, axis=1)
        if np.array_equal(arr, rebuilt) and len(np.unique(reduced.reshape(-1, 4), axis=0)) > 2:
            return True
    return False


def _looks_interpolated(arr: np.ndarray) -> bool:
    alpha = arr[..., 3]
    semi_ratio = float(np.mean((alpha > 0) & (alpha < 255)))
    if semi_ratio > 0.08:
        return True
    # Broad color ramps plus many singleton colors are a conservative bilinear-resize signal.
    visible = arr[..., :3][alpha > 0]
    if len(visible) < 8:
        return False
    _colors, counts = np.unique(visible, axis=0, return_counts=True)
    return len(counts) > 96 and float(np.mean(counts == 1)) > 0.75


def _high_frequency_noise(rgb: np.ndarray, mask: np.ndarray) -> float:
    values: list[np.ndarray] = []
    for axis in (0, 1):
        a = np.take(rgb, range(SIZE - 1), axis=axis)
        b = np.take(rgb, range(1, SIZE), axis=axis)
        ma = np.take(mask, range(SIZE - 1), axis=axis)
        mb = np.take(mask, range(1, SIZE), axis=axis)
        use = ma & mb
        if use.any():
            values.append((_rgb_distance(a, b)[use] > 0.35).astype(np.float32))
    return float(np.mean(np.concatenate(values))) if values else 0.0


def _disconnected_shadow_components(arr: np.ndarray, components: Sequence[Sequence[tuple[int, int]]]) -> int:
    count = 0
    for component in components:
        colors = np.asarray([arr[y, x, :3] for y, x in component], dtype=np.float32)
        luminance = colors @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        if len(component) <= 16 and float(np.mean(luminance)) < 55.0:
            count += 1
    return count


def image_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if not union else float(np.logical_and(a, b).sum() / union)


def perceptual_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Translation-tolerant 16x16 luminance/alpha descriptor, intentionally not a learned embedding.
    ia = Image.fromarray(a, "RGBA").resize((16, 16), Image.Resampling.BOX)
    ib = Image.fromarray(b, "RGBA").resize((16, 16), Image.Resampling.BOX)
    return image_distance(np.asarray(ia), np.asarray(ib))


def batch_metrics(items: Sequence[Mapping[str, Any]], arrays: Sequence[np.ndarray]) -> dict[str, Any]:
    n = len(arrays)
    if not n:
        return {"sample_count": 0}
    exact = [rgba_hash(a) for a in arrays]
    alpha = [alpha_hash(a) for a in arrays]
    normalized = [normalized_alpha_hash(a) for a in arrays]
    pairwise: list[float] = []
    near_pairs = 0
    geometry_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = perceptual_distance(arrays[i], arrays[j])
            pairwise.append(d)
            near_pairs += d <= 0.035
            geometry_pairs += (
                mask_iou(normalized_mask(arrays[i][..., 3] > 0), normalized_mask(arrays[j][..., 3] > 0)) >= 0.96
            )
    pairs = max(1, len(pairwise))
    palette_sets = [set(map(tuple, a[..., :3][a[..., 3] > 0].tolist())) for a in arrays]
    palette_jaccard: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            union = palette_sets[i] | palette_sets[j]
            palette_jaccard.append(len(palette_sets[i] & palette_sets[j]) / max(1, len(union)))
    by_prompt: dict[str, list[int]] = defaultdict(list)
    by_seed: dict[int, list[int]] = defaultdict(list)
    by_condition: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_prompt[str(item.get("prompt_id") or item.get("prompt") or "")].append(index)
        by_seed[int(item.get("noise_seed") or item.get("seed") or index)].append(index)
        by_condition[str(item.get("category") or "unknown")].append(index)
    return {
        "sample_count": n,
        "exact_duplicate_rate": 1.0 - len(set(exact)) / n,
        "alpha_mask_duplicate_rate": 1.0 - len(set(alpha)) / n,
        "perceptual_near_duplicate_rate": near_pairs / pairs,
        "geometry_recolor_duplicate_rate": geometry_pairs / pairs,
        "unique_silhouettes": len(set(normalized)),
        "palette_diversity": 1.0 - float(np.mean(palette_jaccard)) if palette_jaccard else 0.0,
        "pairwise_distance": _distribution(pairwise),
        "seed_sensitivity": _group_distance(by_prompt, arrays),
        "prompt_sensitivity": _group_distance(by_seed, arrays),
        "repeated_template_rate": 1.0 - len(set(normalized)) / n,
        "per_condition_mode_collapse": {
            key: 1.0 - len({normalized[i] for i in indexes}) / len(indexes)
            for key, indexes in sorted(by_condition.items())
            if indexes
        },
        "palette_consistency_mean_jaccard": float(np.mean(palette_jaccard)) if palette_jaccard else None,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "mean": None, "p90": None, "max": None}
    arr = np.asarray(values)
    return {
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _group_distance(groups: Mapping[Any, Sequence[int]], arrays: Sequence[np.ndarray]) -> dict[str, Any]:
    values: list[float] = []
    scorable = 0
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        scorable += 1
        for pos, i in enumerate(indexes):
            values.extend(perceptual_distance(arrays[i], arrays[j]) for j in indexes[pos + 1 :])
    return {"scorable_groups": scorable, **_distribution(values)}


def duplicate_groups(items: Sequence[Mapping[str, Any]], key: str) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in items:
        value = str(item.get("metrics", {}).get("pixel_art", {}).get(key, ""))
        if value:
            grouped[value].append(str(item.get("sample_id", "")))
    return [ids for ids in grouped.values() if len(ids) > 1]
