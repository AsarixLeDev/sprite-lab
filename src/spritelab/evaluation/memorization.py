"""Training-neighbor retrieval with exact, translated, pixel, and geometry evidence."""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class TrainingImage:
    sprite_id: str
    dataset: str
    npz_file: str
    npz_row: int
    rgba: np.ndarray


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


def retrieve_neighbors(generated: np.ndarray, training: list[TrainingImage], *, top_k: int = 3) -> list[dict[str, Any]]:
    """Rank neighbors; individual evidence remains separate to avoid style==memorization."""
    gen_rgba_hash = rgba_hash(generated)
    gen_alpha_hash = alpha_hash(generated)
    gen_norm_hash = normalized_alpha_hash(generated)
    gen_norm_mask = normalized_mask(generated[..., 3] > 0)
    rows: list[dict[str, Any]] = []
    for target in training:
        exact_rgba = gen_rgba_hash == rgba_hash(target.rgba)
        exact_alpha = gen_alpha_hash == alpha_hash(target.rgba)
        translated = gen_norm_hash == normalized_alpha_hash(target.rgba) and not exact_alpha
        pixel = image_distance(generated, target.rgba)
        perceptual = perceptual_distance(generated, target.rgba)
        geometry = mask_iou(gen_norm_mask, normalized_mask(target.rgba[..., 3] > 0))
        suspicious = bool(
            exact_rgba or exact_alpha or translated or pixel <= 0.025 or (geometry >= 0.98 and perceptual <= 0.08)
        )
        # Exact evidence dominates; otherwise rank by combined pixel/geometry evidence.
        rank = 0.0 if exact_rgba else 0.1 if exact_alpha else 0.2 if translated else pixel + (1.0 - geometry) * 0.25
        rows.append(
            {
                "sprite_id": target.sprite_id,
                "dataset": target.dataset,
                "npz_file": target.npz_file,
                "npz_row": target.npz_row,
                "exact_rgba": exact_rgba,
                "exact_alpha": exact_alpha,
                "translated_duplicate": translated,
                "pixel_distance": pixel,
                "perceptual_distance": perceptual,
                "geometry_iou": geometry,
                "optional_embedding_similarity": None,
                "suspicious": suspicious,
                "rank": rank,
            }
        )
    rows.sort(key=lambda item: (float(item["rank"]), str(item["sprite_id"])))
    return rows[:top_k]


def suspicious_kind(neighbor: dict[str, Any] | None) -> str | None:
    if not neighbor or not neighbor.get("suspicious"):
        return None
    if neighbor.get("exact_rgba"):
        return "exact_rgba"
    if neighbor.get("exact_alpha"):
        return "exact_alpha"
    if neighbor.get("translated_duplicate"):
        return "translated_duplicate"
    if float(neighbor.get("pixel_distance", 1.0)) <= 0.025:
        return "near_duplicate_pixel"
    return "geometry_and_perceptual"
