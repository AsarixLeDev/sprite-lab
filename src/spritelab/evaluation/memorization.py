"""Deterministic training-neighbor evidence with minimum-evidence controls.

The detector reports evidence; it does not decide checkpoint promotion.  Policy
values live in :data:`DETECTOR_POLICY` so every threshold is serialized and
bound to the same SHA-256 in reports.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.evaluation.metrics import (
    alpha_hash,
    image_distance,
    mask_iou,
    normalized_alpha_hash,
    normalized_mask,
    perceptual_distance,
    rgba_hash,
)

DETECTOR_POLICY_VERSION = "memorization_detector_v2"
COMPARISON_METHOD = "deterministic_rgba_alpha_occupancy_v2"
DETECTOR_POLICY: dict[str, Any] = {
    "detector_policy_version": DETECTOR_POLICY_VERSION,
    "comparison_method": COMPARISON_METHOD,
    "thresholds": {
        # Both sides must satisfy these floors before near-pixel evidence or
        # nontrivial exact-RGBA evidence is possible.
        "minimum_foreground_pixels": 16,
        "minimum_foreground_occupancy": 0.015625,
        # A mask at or below either bound is near blank (32x32 reference).
        "near_blank_threshold": {
            "maximum_foreground_pixels": 4,
            "maximum_foreground_occupancy": 0.00390625,
        },
        # Small shapes with little canvas support and a highly symmetric axis
        # are generic sparse collisions, even when their masks match exactly.
        "generic_sparse_collision_thresholds": {
            "maximum_foreground_pixels": 32,
            "maximum_foreground_occupancy": 0.03125,
            "minimum_axis_symmetry": 0.9,
        },
        "near_pixel": {
            "maximum_union_rgba_distance": 0.025,
            "minimum_alpha_iou": 0.9,
            "minimum_compared_foreground_pixels": 16,
        },
        "legacy_diagnostic_thresholds": {
            "full_canvas_pixel_distance": 0.025,
            "geometry_iou": 0.98,
            "perceptual_distance": 0.08,
        },
    },
    "diagnostic_semantics": {
        "alpha_bbox": "[left, top, right_exclusive, bottom_exclusive], or null when blank",
        "alpha_centroid": "[x, y] pixel-coordinate mean, or null when blank",
        "foreground": "decoded alpha > 0",
        "connected_components": "four-connected binary-alpha components",
        "unique_visible_rgba": "unique decoded RGBA values at alpha > 0",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


COMPARISON_PARAMETERS: dict[str, Any] = {
    "thresholds": DETECTOR_POLICY["thresholds"],
    "diagnostic_semantics": DETECTOR_POLICY["diagnostic_semantics"],
}
COMPARISON_PARAMETERS_SHA256 = hashlib.sha256(_canonical_json(COMPARISON_PARAMETERS).encode("utf-8")).hexdigest()
DETECTOR_POLICY_SHA256 = hashlib.sha256(_canonical_json(DETECTOR_POLICY).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainingImage:
    sprite_id: str
    dataset: str
    npz_file: str
    npz_row: int
    rgba: np.ndarray


def detector_policy_record() -> dict[str, Any]:
    """Return a JSON-safe copy of the canonical policy and its hashes."""
    return {
        **json.loads(_canonical_json(DETECTOR_POLICY)),
        "comparison_parameters": json.loads(_canonical_json(COMPARISON_PARAMETERS)),
        "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
        "detector_policy_sha256": DETECTOR_POLICY_SHA256,
    }


def load_training_images(manifest_paths: list[Path], *, limit: int = 0) -> list[TrainingImage]:
    """Reconstruct unique exported RGBA training targets without changing datasets."""
    result: list[TrainingImage] = []
    seen: set[tuple[str, str, int]] = set()
    npz_cache: dict[Path, Any] = {}
    for manifest_path in manifest_paths:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = _dataset_dir(row, manifest_path)
            npz_file = str(row.get("npz_file") or "")
            npz_row = int(row.get("npz_row", -1))
            identity = (str(dataset.resolve()), npz_file, npz_row)
            if not npz_file or npz_row < 0 or identity in seen:
                continue
            seen.add(identity)
            path = dataset / npz_file
            if path not in npz_cache:
                npz_cache[path] = np.load(path, mmap_mode="r")
            rgba = reconstruct_rgba(npz_cache[path], npz_row)
            result.append(
                TrainingImage(
                    sprite_id=str(row.get("sprite_id") or row.get("source_sprite_id") or f"row_{npz_row}"),
                    dataset=str(dataset),
                    npz_file=npz_file,
                    npz_row=npz_row,
                    rgba=rgba,
                )
            )
            if limit and len(result) >= limit:
                return result
    return result


def _dataset_dir(row: dict[str, Any], manifest_path: Path) -> Path:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    raw = source.get("dataset_dir")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    return manifest_path.parent


def reconstruct_rgba(npz: Any, row: int) -> np.ndarray:
    alpha = np.asarray(npz["alpha"][row])
    index = np.asarray(npz["index_map"][row])
    palette = np.asarray(npz["palette"][row])
    mask = np.asarray(npz["palette_mask"][row]) if "palette_mask" in npz.files else np.ones(len(palette), bool)
    palette = palette[mask]
    safe = np.clip(index, 0, max(0, len(palette) - 1))
    rgb = palette[safe] if len(palette) else np.zeros((*alpha.shape, 3), dtype=np.uint8)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(alpha > 0, 255, 0).astype(np.uint8)
    rgba[rgba[..., 3] == 0, :3] = 0
    return rgba


def image_diagnostics(rgba: np.ndarray) -> dict[str, Any]:
    """Return deterministic minimum-evidence diagnostics for one decoded image."""
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError("memorization detector requires an HxWx4 RGBA array")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise ValueError("memorization detector requires finite numeric RGBA values")
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    height, width = arr.shape[:2]
    alpha = arr[..., 3] > 0
    count = int(np.count_nonzero(alpha))
    occupancy = count / float(max(1, width * height))
    ys, xs = np.nonzero(alpha)
    bbox = None if not count else [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    bbox_width = 0 if bbox is None else bbox[2] - bbox[0]
    bbox_height = 0 if bbox is None else bbox[3] - bbox[1]
    near = DETECTOR_POLICY["thresholds"]["near_blank_threshold"]
    visible = arr[alpha]
    unique_visible = len(np.unique(visible, axis=0)) if count else 0
    centroid = None if not count else [float(np.mean(xs)), float(np.mean(ys))]
    return {
        "width": int(width),
        "height": int(height),
        "foreground_pixel_count": count,
        "foreground_occupancy": occupancy,
        "alpha_bbox": bbox,
        "alpha_bbox_width": bbox_width,
        "alpha_bbox_height": bbox_height,
        "connected_component_count": _component_count(alpha),
        "blank_alpha": count == 0,
        "near_blank_alpha": count <= int(near["maximum_foreground_pixels"])
        or occupancy <= float(near["maximum_foreground_occupancy"]),
        "unique_visible_rgba_count": unique_visible,
        "alpha_centroid": centroid,
        "horizontal_symmetry": _symmetry(alpha, axis=1),
        "vertical_symmetry": _symmetry(alpha, axis=0),
    }


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for y, x in zip(*np.nonzero(mask), strict=True):
        if seen[y, x]:
            continue
        count += 1
        seen[y, x] = True
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        while queue:
            cy, cx = queue.popleft()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
    return count


def _symmetry(mask: np.ndarray, *, axis: int) -> float:
    if not mask.any():
        return 1.0
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return float(np.mean(crop == np.flip(crop, axis=axis)))


def _sufficient(diag: dict[str, Any]) -> bool:
    thresholds = DETECTOR_POLICY["thresholds"]
    return bool(
        diag["foreground_pixel_count"] >= thresholds["minimum_foreground_pixels"]
        and diag["foreground_occupancy"] >= thresholds["minimum_foreground_occupancy"]
        and not diag["near_blank_alpha"]
    )


def _generic_sparse(diag: dict[str, Any]) -> bool:
    thresholds = DETECTOR_POLICY["thresholds"]["generic_sparse_collision_thresholds"]
    symmetric = max(float(diag["horizontal_symmetry"]), float(diag["vertical_symmetry"])) >= float(
        thresholds["minimum_axis_symmetry"]
    )
    return bool(
        diag["foreground_pixel_count"] <= thresholds["maximum_foreground_pixels"]
        and diag["foreground_occupancy"] <= thresholds["maximum_foreground_occupancy"]
        and symmetric
    )


def _union_rgba_distance(a: np.ndarray, b: np.ndarray) -> tuple[float | None, int]:
    mask_a = a[..., 3] > 0
    mask_b = b[..., 3] > 0
    union = mask_a | mask_b
    count = int(np.count_nonzero(union))
    if not count:
        return None, 0
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
    return float(np.mean(delta[union])), count


def _evidence_attributes(evidence_class: str, low_evidence_reason: str | None = None) -> dict[str, Any]:
    hard = evidence_class == "exact_rgba_nontrivial"
    review = evidence_class in {
        "exact_alpha_review_required",
        "translation_alpha_review_required",
        "near_pixel_review_required",
    }
    warning = evidence_class in {
        "exact_rgba_low_evidence_collision",
        "generic_sparse_collision",
        "blank_collision",
    }
    strength = "hard" if hard else "review_required" if review else "low_evidence" if warning else "none"
    return {
        "evidence_class": evidence_class,
        "evidence_strength": strength,
        "requires_human_review": review,
        "machine_hard_block_candidate": hard,
        "warning_only": warning,
        "low_evidence_reason": low_evidence_reason,
        "suspicious": hard or review or warning,
    }


def _classify(
    *,
    exact_rgba: bool,
    exact_alpha: bool,
    translated: bool,
    union_distance: float | None,
    compared_pixels: int,
    alpha_iou: float,
    generated_diag: dict[str, Any],
    training_diag: dict[str, Any],
) -> dict[str, Any]:
    both_blank = generated_diag["blank_alpha"] and training_diag["blank_alpha"]
    any_blank = generated_diag["blank_alpha"] or training_diag["blank_alpha"]
    generic = _generic_sparse(generated_diag) or _generic_sparse(training_diag)
    insufficient = not _sufficient(generated_diag) or not _sufficient(training_diag)
    near_blank = generated_diag["near_blank_alpha"] or training_diag["near_blank_alpha"]
    low_reason = "blank_alpha" if any_blank else "near_blank_alpha" if near_blank else "generic_sparse_mask"

    if exact_rgba:
        if both_blank or insufficient or generic:
            return _evidence_attributes("exact_rgba_low_evidence_collision", low_reason)
        return _evidence_attributes("exact_rgba_nontrivial")
    if both_blank:
        return _evidence_attributes("blank_collision", "blank_alpha_with_transparent_rgb_difference")
    if any_blank:
        return _evidence_attributes("no_material_match")
    if exact_alpha:
        if insufficient or generic:
            return _evidence_attributes("generic_sparse_collision", low_reason)
        return _evidence_attributes("exact_alpha_review_required")
    if translated:
        if insufficient or generic:
            return _evidence_attributes("generic_sparse_collision", low_reason)
        return _evidence_attributes("translation_alpha_review_required")

    near = DETECTOR_POLICY["thresholds"]["near_pixel"]
    near_pixel = bool(
        not insufficient
        and not generic
        and union_distance is not None
        and compared_pixels >= near["minimum_compared_foreground_pixels"]
        and union_distance <= near["maximum_union_rgba_distance"]
        and alpha_iou >= near["minimum_alpha_iou"]
    )
    if near_pixel:
        return _evidence_attributes("near_pixel_review_required")
    return _evidence_attributes("no_material_match")


def retrieve_neighbors(
    generated: np.ndarray,
    training: list[TrainingImage],
    *,
    top_k: int = 3,
    detector_policy_version: str = DETECTOR_POLICY_VERSION,
) -> list[dict[str, Any]]:
    """Rank deterministic evidence rows, failing closed for unknown policies."""
    if detector_policy_version != DETECTOR_POLICY_VERSION:
        raise ValueError(f"unsupported detector policy version: {detector_policy_version!r}")
    generated = np.clip(np.asarray(generated), 0, 255).astype(np.uint8)
    generated_diag = image_diagnostics(generated)
    gen_rgba_hash = rgba_hash(generated)
    gen_alpha_hash = alpha_hash(generated)
    gen_norm_hash = normalized_alpha_hash(generated)
    gen_norm_mask = normalized_mask(generated[..., 3] > 0)
    rows: list[dict[str, Any]] = []
    class_rank = {
        "exact_rgba_nontrivial": 0,
        "exact_alpha_review_required": 1,
        "translation_alpha_review_required": 2,
        "near_pixel_review_required": 3,
        "exact_rgba_low_evidence_collision": 4,
        "generic_sparse_collision": 5,
        "blank_collision": 6,
        "no_material_match": 7,
    }
    for target in training:
        target_rgba = np.clip(np.asarray(target.rgba), 0, 255).astype(np.uint8)
        training_diag = image_diagnostics(target_rgba)
        exact_rgba = gen_rgba_hash == rgba_hash(target_rgba)
        exact_alpha = gen_alpha_hash == alpha_hash(target_rgba)
        translated = gen_norm_hash == normalized_alpha_hash(target_rgba) and not exact_alpha
        pixel = image_distance(generated, target_rgba)
        perceptual = perceptual_distance(generated, target_rgba)
        geometry = mask_iou(gen_norm_mask, normalized_mask(target_rgba[..., 3] > 0))
        alpha_iou = mask_iou(generated[..., 3] > 0, target_rgba[..., 3] > 0)
        union_distance, compared_pixels = _union_rgba_distance(generated, target_rgba)
        evidence = _classify(
            exact_rgba=exact_rgba,
            exact_alpha=exact_alpha,
            translated=translated,
            union_distance=union_distance,
            compared_pixels=compared_pixels,
            alpha_iou=alpha_iou,
            generated_diag=generated_diag,
            training_diag=training_diag,
        )
        rank = class_rank[evidence["evidence_class"]] + (union_distance if union_distance is not None else pixel)
        rows.append(
            {
                "sprite_id": target.sprite_id,
                "dataset": target.dataset,
                "npz_file": target.npz_file,
                "npz_row": target.npz_row,
                "detector_policy_version": DETECTOR_POLICY_VERSION,
                "comparison_method": COMPARISON_METHOD,
                "comparison_parameters": COMPARISON_PARAMETERS,
                "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
                "detector_policy_sha256": DETECTOR_POLICY_SHA256,
                "generated_diagnostics": generated_diag,
                "training_diagnostics": training_diag,
                "exact_rgba": exact_rgba,
                "exact_alpha": exact_alpha,
                "rgb_values_differ": bool(exact_alpha and not exact_rgba),
                "translated_duplicate": translated,
                "pixel_distance": pixel,
                "union_rgba_distance": union_distance,
                "compared_foreground_pixel_count": compared_pixels,
                "alpha_iou": alpha_iou,
                "perceptual_distance": perceptual,
                "geometry_iou": geometry,
                "optional_embedding_similarity": None,
                **evidence,
                "rank": rank,
            }
        )
    rows.sort(key=lambda item: (float(item["rank"]), str(item["sprite_id"])))
    return rows[:top_k]


def suspicious_kind(neighbor: dict[str, Any] | None) -> str | None:
    """Derive legacy relation labels without collapsing new evidence strengths."""
    if not neighbor or not neighbor.get("suspicious"):
        return None
    evidence_class = str(neighbor.get("evidence_class") or "")
    legacy = {
        "exact_rgba_nontrivial": "exact_rgba",
        "exact_alpha_review_required": "exact_alpha",
        "translation_alpha_review_required": "translated_duplicate",
        "near_pixel_review_required": "near_duplicate_pixel",
    }
    return legacy.get(evidence_class, evidence_class or None)
