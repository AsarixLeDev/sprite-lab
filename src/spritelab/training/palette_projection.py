"""Decode-time palette projection for generated sprite folders."""

from __future__ import annotations

import json
import math
import shutil
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.training.framing_metrics import checkerboard_rgba
from spritelab.training.generated_canonicalizer import (
    SPRITE_SIZE,
    GeneratedSprite,
    build_generation_contact_sheet,
    reconstruct_indexed_rgba,
)

SCHEMA_VERSION = "palette_projection_v1.0"
PROJECTION_METHODS = ("deterministic_kmeans",)
IMAGE_PATH_KEYS = ("raw_rgba", "hard_rgba", "indexed_png")


@dataclass(frozen=True)
class PaletteProjectionConfig:
    generated: Path
    out: Path
    target_colors: int = 16
    min_pixel_share: float = 0.01
    alpha_threshold: float = 0.5
    method: str = "deterministic_kmeans"
    max_contact_sheet_samples: int = 64


@dataclass(frozen=True)
class PaletteProjectionResult:
    rgba: np.ndarray
    metrics: dict[str, Any]


def project_generated_palette(config: PaletteProjectionConfig) -> dict[str, Any]:
    """Project each generated sample to a cleaner per-image palette."""

    _validate_projection_options(
        target_colors=config.target_colors,
        min_pixel_share=config.min_pixel_share,
        alpha_threshold=config.alpha_threshold,
        method=config.method,
    )
    generated_dir = Path(config.generated)
    out_dir = Path(config.out)
    if not generated_dir.is_dir():
        raise FileNotFoundError(f"generated directory does not exist: {generated_dir}")

    manifest_path = generated_dir / "generated_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"generated_manifest.jsonl is missing: {manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    source_records = _read_manifest(manifest_path)
    projected_records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for record in source_records:
        projected_record, sample = _project_record(generated_dir, out_dir, record, config)
        projected_records.append(projected_record)
        samples.append(sample)

    _write_manifest(out_dir / "generated_manifest.jsonl", projected_records)
    _copy_ancillary_metadata(generated_dir, out_dir)

    contact_sheets = _write_projection_contact_sheets(
        generated_dir,
        out_dir,
        samples,
        max_samples=config.max_contact_sheet_samples,
    )
    generation_contact_sheet = build_generation_contact_sheet(
        out_dir,
        projected_records,
        out_dir / "generation_contact_sheet.png",
        include_raw=True,
        max_items=config.max_contact_sheet_samples,
    )
    if generation_contact_sheet is not None:
        contact_sheets["generation_after"] = generation_contact_sheet

    report = _aggregate_report(
        config=config,
        generated_dir=generated_dir,
        out_dir=out_dir,
        samples=samples,
        records=projected_records,
        contact_sheets=contact_sheets,
    )
    _write_projection_reports(out_dir, report, samples)
    _write_generation_report(out_dir, projected_records, report, generation_contact_sheet)
    return report


def project_rgba_array(
    rgba: np.ndarray,
    *,
    target_colors: int = 16,
    min_pixel_share: float = 0.01,
    alpha_threshold: float = 0.5,
    method: str = "deterministic_kmeans",
) -> PaletteProjectionResult:
    """Project visible RGB pixels in an RGBA array while preserving alpha exactly."""

    _validate_projection_options(
        target_colors=target_colors,
        min_pixel_share=min_pixel_share,
        alpha_threshold=alpha_threshold,
        method=method,
    )
    source = _rgba_to_uint8(rgba)
    projected = source.copy()
    visible = _visible_mask(source, alpha_threshold=alpha_threshold)
    metrics_before = _color_distribution_metrics(source, visible, min_pixel_share=min_pixel_share)

    cluster_count = 0
    tiny_cluster_merge_count = 0
    if bool(np.any(visible)):
        projected_pixels, cluster_count, tiny_cluster_merge_count = _project_rgb_pixels(
            source[..., :3][visible],
            target_colors=target_colors,
            min_pixel_share=min_pixel_share,
        )
        projected[..., :3][visible] = projected_pixels

    metrics_after = _color_distribution_metrics(projected, visible, min_pixel_share=min_pixel_share)
    diff = np.abs(projected[..., :3].astype(np.float32) - source[..., :3].astype(np.float32)) / 255.0
    if bool(np.any(visible)):
        visible_diff = diff[visible]
        rgb_mae = float(np.mean(visible_diff))
        rgb_rmse = float(math.sqrt(float(np.mean(np.square(visible_diff)))))
        max_rgb_error = float(np.max(visible_diff))
    else:
        rgb_mae = 0.0
        rgb_rmse = 0.0
        max_rgb_error = 0.0

    alpha_changed = int(np.count_nonzero(projected[..., 3] != source[..., 3]))
    metrics = {
        "visible_color_count_before": metrics_before["visible_color_count"],
        "visible_color_count_after": metrics_after["visible_color_count"],
        "rare_color_count_before": metrics_before["rare_color_count"],
        "rare_color_count_after": metrics_after["rare_color_count"],
        "rare_color_rate_before": metrics_before["rare_color_rate"],
        "rare_color_rate_after": metrics_after["rare_color_rate"],
        "rgb_mae_visible": rgb_mae,
        "rgb_rmse_visible": rgb_rmse,
        "max_rgb_error_visible": max_rgb_error,
        "alpha_changed_pixels": alpha_changed,
        "cluster_count": int(cluster_count),
        "tiny_cluster_merge_count": int(tiny_cluster_merge_count),
        "destructiveness": _destructiveness_label(rgb_mae),
    }
    return PaletteProjectionResult(rgba=_restore_input_range(projected, rgba), metrics=metrics)


def project_generated_sprite_record(
    sprite: GeneratedSprite,
    out_dir: Path,
    record: Mapping[str, Any],
    *,
    target_colors: int = 16,
    min_pixel_share: float = 0.01,
    alpha_threshold: float = 0.5,
    method: str = "deterministic_kmeans",
) -> dict[str, Any]:
    """Project the exported/canonical RGB image for one generated manifest row.

    Raw and hard RGBA artifacts are left untouched. The projected image becomes
    the manifest's ``indexed_png`` so existing review and faithfulness tools use
    it without changing their image-selection semantics.
    """

    _validate_projection_options(
        target_colors=target_colors,
        min_pixel_share=min_pixel_share,
        alpha_threshold=alpha_threshold,
        method=method,
    )
    out_root = Path(out_dir)
    sample_id = _safe_sample_id(str(record.get("sample_id") or "sample"))
    before_rgba = reconstruct_indexed_rgba(index_map=sprite.index_map, palette=sprite.palette)
    projection = project_rgba_array(
        before_rgba,
        target_colors=target_colors,
        min_pixel_share=min_pixel_share,
        alpha_threshold=alpha_threshold,
        method=method,
    )
    projected_rel = Path("projected") / f"{sample_id}.png"
    _save_rgba(projection.rgba, out_root / projected_rel)

    paths = dict(record.get("paths") if isinstance(record.get("paths"), Mapping) else {})
    before_rel = str(paths.get("indexed_png") or "")
    if before_rel:
        paths["pre_projection_indexed_png"] = before_rel
    paths["projected_png"] = projected_rel.as_posix()
    paths["indexed_png"] = projected_rel.as_posix()

    projected_record = dict(record)
    projected_record["paths"] = paths
    projected_record["canonical_max_colors_before_projection"] = int(record.get("max_colors") or 0)
    projected_record["max_colors"] = int(target_colors)
    projected_record["visible_color_count"] = int(projection.metrics["visible_color_count_after"])
    projected_record = _with_projection_manifest_fields(
        projected_record,
        metrics=projection.metrics,
        source_generated_dir=out_root,
        method=method,
        target_colors=target_colors,
        min_pixel_share=min_pixel_share,
        alpha_threshold=alpha_threshold,
        metric_source="indexed_png",
    )
    return projected_record


def write_runtime_projection_report(
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    target_colors: int = 16,
    min_pixel_share: float = 0.01,
    alpha_threshold: float = 0.5,
    method: str = "deterministic_kmeans",
    max_contact_sheet_samples: int = 64,
) -> dict[str, Any] | None:
    """Write palette-projection reports for samples projected during decoding."""

    projected_records = [record for record in records if bool(record.get("palette_projection_applied"))]
    if not projected_records:
        return None
    out_root = Path(out_dir)
    samples = [_runtime_projection_sample(record) for record in projected_records]
    contact_sheets: dict[str, Path] = {}
    before_after = out_root / "contact_sheet_projected.png"
    if _build_runtime_projection_contact_sheet(out_root, samples, before_after, max_samples=max_contact_sheet_samples):
        contact_sheets["before_after"] = before_after
    before = out_root / "palette_projection_before_contact_sheet.png"
    after = out_root / "palette_projection_after_contact_sheet.png"
    if _build_projection_contact_sheet(
        out_root, samples, before, path_key="before_path", max_samples=max_contact_sheet_samples
    ):
        contact_sheets["before"] = before
    if _build_projection_contact_sheet(
        out_root, samples, after, path_key="after_path", max_samples=max_contact_sheet_samples
    ):
        contact_sheets["after"] = after

    report = _aggregate_report(
        config=PaletteProjectionConfig(
            generated=out_root,
            out=out_root,
            target_colors=target_colors,
            min_pixel_share=min_pixel_share,
            alpha_threshold=alpha_threshold,
            method=method,
            max_contact_sheet_samples=max_contact_sheet_samples,
        ),
        generated_dir=out_root,
        out_dir=out_root,
        samples=samples,
        records=list(projected_records),
        contact_sheets=contact_sheets,
    )
    _write_projection_reports(out_root, report, samples)
    return report


def _project_record(
    generated_dir: Path,
    out_dir: Path,
    record: Mapping[str, Any],
    config: PaletteProjectionConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = _safe_sample_id(str(record.get("sample_id") or "sample"))
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    projected_paths: dict[str, str] = {}
    image_metrics: dict[str, dict[str, Any]] = {}

    primary_key = _primary_image_key(paths)
    primary_before_rel = str(paths.get(primary_key) or "") if primary_key else ""
    primary_after_rel = primary_before_rel
    alpha_opaque_counts: dict[str, int] = {}
    alpha_threshold_byte = _alpha_threshold_byte(config.alpha_threshold)

    for key in IMAGE_PATH_KEYS:
        rel_value = paths.get(key)
        if not rel_value:
            continue
        source_path = _resolve_generated_path(generated_dir, str(rel_value))
        if not source_path.is_file():
            continue
        image = _load_rgba(source_path)
        source_array = np.asarray(image, dtype=np.uint8)
        projection = project_rgba_array(
            source_array,
            target_colors=config.target_colors,
            min_pixel_share=config.min_pixel_share,
            alpha_threshold=config.alpha_threshold,
            method=config.method,
        )
        out_rel = Path(str(rel_value).replace("\\", "/"))
        out_path = out_dir / out_rel
        _save_rgba(projection.rgba, out_path)
        projected_paths[key] = out_rel.as_posix()
        image_metrics[key] = projection.metrics
        # Projection never touches alpha, so the source array's alpha channel already
        # reflects the projected output; no need to re-read the file we just wrote.
        alpha_opaque_counts[key] = int(np.count_nonzero(source_array[..., 3] >= alpha_threshold_byte))
        if key == primary_key:
            primary_after_rel = out_rel.as_posix()

    if not projected_paths:
        raise FileNotFoundError(f"{sample_id}: no generated PNG artifacts were found to project")

    if primary_key is None or primary_key not in image_metrics:
        primary_key = next(key for key in IMAGE_PATH_KEYS if key in image_metrics)
        primary_before_rel = str(paths.get(primary_key) or "")
        primary_after_rel = projected_paths[primary_key]

    primary_metrics = dict(image_metrics[primary_key])
    projected_record = dict(record)
    projected_record["paths"] = {**dict(paths), **projected_paths}
    projected_record["max_colors"] = int(config.target_colors)
    projected_record["visible_color_count"] = int(primary_metrics["visible_color_count_after"])
    alpha_count_key = next(
        (key for key in ("hard_rgba", "indexed_png", "raw_rgba") if key in alpha_opaque_counts), None
    )
    projected_record["alpha_opaque_count"] = alpha_opaque_counts.get(alpha_count_key, 0) if alpha_count_key else 0
    projected_record = _with_projection_manifest_fields(
        projected_record,
        metrics=primary_metrics,
        source_generated_dir=generated_dir,
        method=config.method,
        target_colors=config.target_colors,
        min_pixel_share=config.min_pixel_share,
        alpha_threshold=config.alpha_threshold,
        metric_source=primary_key,
    )

    sample = {
        "sample_id": sample_id,
        "prompt_id": str(record.get("prompt_id") or ""),
        "prompt": str(record.get("prompt") or ""),
        "category": str(record.get("category") or ""),
        "metric_source": primary_key,
        "source_paths": {key: str(paths.get(key) or "") for key in IMAGE_PATH_KEYS if paths.get(key)},
        "projected_paths": projected_paths,
        "before_path": primary_before_rel,
        "after_path": primary_after_rel,
        **primary_metrics,
        "image_metrics": image_metrics,
    }
    return projected_record, sample


def _with_projection_manifest_fields(
    record: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    source_generated_dir: Path,
    method: str,
    target_colors: int,
    min_pixel_share: float,
    alpha_threshold: float,
    metric_source: str,
) -> dict[str, Any]:
    output = dict(record)
    output.update(
        {
            "palette_projection_applied": True,
            "palette_projection_method": str(method),
            "palette_projection_target_colors": int(target_colors),
            "palette_projection_min_pixel_share": float(min_pixel_share),
            "visible_color_count_before_projection": int(metrics.get("visible_color_count_before") or 0),
            "visible_color_count_after_projection": int(metrics.get("visible_color_count_after") or 0),
            "rare_color_count_before_projection": int(metrics.get("rare_color_count_before") or 0),
            "rare_color_count_after_projection": int(metrics.get("rare_color_count_after") or 0),
            "rare_color_rate_before_projection": float(metrics.get("rare_color_rate_before") or 0.0),
            "rare_color_rate_after_projection": float(metrics.get("rare_color_rate_after") or 0.0),
            "rgb_mae_visible_projection": float(metrics.get("rgb_mae_visible") or 0.0),
            "rgb_rmse_visible_projection": float(metrics.get("rgb_rmse_visible") or 0.0),
            "max_rgb_error_visible_projection": float(metrics.get("max_rgb_error_visible") or 0.0),
            "alpha_changed_pixels_projection": int(metrics.get("alpha_changed_pixels") or 0),
            "cluster_count_projection": int(metrics.get("cluster_count") or 0),
            "tiny_cluster_merge_count_projection": int(metrics.get("tiny_cluster_merge_count") or 0),
            "projection_destructiveness": str(metrics.get("destructiveness") or "safe"),
            "project_palette": True,
            "project_palette_target_colors": int(target_colors),
            "project_palette_min_pixel_share": float(min_pixel_share),
            "project_palette_method": str(method),
        }
    )
    output["palette_projection"] = {
        "schema_version": SCHEMA_VERSION,
        "source_generated_dir": str(source_generated_dir),
        "method": str(method),
        "target_colors": int(target_colors),
        "min_pixel_share": float(min_pixel_share),
        "alpha_threshold": float(alpha_threshold),
        "metric_source": str(metric_source),
        "rgb_mae_visible": float(metrics.get("rgb_mae_visible") or 0.0),
        "rgb_rmse_visible": float(metrics.get("rgb_rmse_visible") or 0.0),
        "max_rgb_error_visible": float(metrics.get("max_rgb_error_visible") or 0.0),
        "destructiveness": str(metrics.get("destructiveness") or "safe"),
        "metrics": dict(metrics),
    }
    return output


def _runtime_projection_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    nested = record.get("palette_projection") if isinstance(record.get("palette_projection"), Mapping) else {}
    metrics = nested.get("metrics") if isinstance(nested.get("metrics"), Mapping) else {}
    before_rel = str(paths.get("pre_projection_indexed_png") or "")
    after_rel = str(paths.get("projected_png") or paths.get("indexed_png") or "")
    sample = {
        "sample_id": str(record.get("sample_id") or ""),
        "prompt_id": str(record.get("prompt_id") or ""),
        "prompt": str(record.get("prompt") or ""),
        "category": str(record.get("category") or ""),
        "metric_source": str(nested.get("metric_source") or "indexed_png"),
        "source_paths": {
            key: str(paths.get(key) or "")
            for key in ("raw_rgba", "hard_rgba", "pre_projection_indexed_png")
            if paths.get(key)
        },
        "projected_paths": {"indexed_png": after_rel, "projected_png": after_rel},
        "before_path": before_rel,
        "after_path": after_rel,
        "image_metrics": {"indexed_png": dict(metrics)},
    }
    for key in (
        "visible_color_count_before",
        "visible_color_count_after",
        "rare_color_count_before",
        "rare_color_count_after",
        "rare_color_rate_before",
        "rare_color_rate_after",
        "rgb_mae_visible",
        "rgb_rmse_visible",
        "max_rgb_error_visible",
        "alpha_changed_pixels",
        "cluster_count",
        "tiny_cluster_merge_count",
        "destructiveness",
    ):
        if key in metrics:
            sample[key] = metrics[key]
    return sample


def _project_rgb_pixels(
    rgb_pixels: np.ndarray,
    *,
    target_colors: int,
    min_pixel_share: float,
) -> tuple[np.ndarray, int, int]:
    pixels = np.asarray(rgb_pixels, dtype=np.uint8)
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError(f"rgb_pixels must have shape [N, 3], got {pixels.shape}")
    if pixels.shape[0] == 0:
        return pixels.copy(), 0, 0

    unique, inverse, counts = np.unique(pixels, axis=0, return_inverse=True, return_counts=True)
    labels, centers = _deterministic_kmeans_unique(unique, counts, target_colors=target_colors)
    labels, centers, merge_count = _merge_tiny_clusters(
        unique,
        counts,
        labels,
        centers,
        min_pixel_share=min_pixel_share,
    )
    centers_uint8 = np.rint(np.clip(centers, 0.0, 255.0)).astype(np.uint8)
    return centers_uint8[labels[inverse]], int(centers_uint8.shape[0]), int(merge_count)


def _deterministic_kmeans_unique(
    unique: np.ndarray,
    counts: np.ndarray,
    *,
    target_colors: int,
) -> tuple[np.ndarray, np.ndarray]:
    color_count = int(unique.shape[0])
    k = min(int(target_colors), color_count)
    if color_count <= int(target_colors):
        return np.arange(color_count, dtype=np.int64), unique.astype(np.float64)

    centers = _initial_centers(unique.astype(np.float64), counts, k)
    labels = np.full((color_count,), -1, dtype=np.int64)
    values = unique.astype(np.float64)
    weights = counts.astype(np.float64)

    for _ in range(50):
        distances = np.sum(np.square(values[:, None, :] - centers[None, :, :]), axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int64)
        next_centers = centers.copy()
        for cluster_index in range(k):
            mask = next_labels == cluster_index
            if bool(np.any(mask)):
                next_centers[cluster_index] = np.average(values[mask], axis=0, weights=weights[mask])
        if np.array_equal(next_labels, labels) and np.allclose(next_centers, centers):
            labels = next_labels
            centers = next_centers
            break
        labels = next_labels
        centers = next_centers

    return labels, centers


def _initial_centers(unique: np.ndarray, counts: np.ndarray, k: int) -> np.ndarray:
    first = min(
        range(unique.shape[0]),
        key=lambda index: (-int(counts[index]), int(unique[index, 0]), int(unique[index, 1]), int(unique[index, 2])),
    )
    selected = [first]
    while len(selected) < int(k):
        selected_values = unique[np.asarray(selected, dtype=np.int64)]
        distances = np.sum(np.square(unique[:, None, :] - selected_values[None, :, :]), axis=2)
        min_distances = np.min(distances, axis=1)
        for index in selected:
            min_distances[index] = -1.0
        next_index = max(
            range(unique.shape[0]),
            key=lambda index: (
                float(min_distances[index]),
                int(counts[index]),
                -int(unique[index, 0]),
                -int(unique[index, 1]),
                -int(unique[index, 2]),
            ),
        )
        selected.append(int(next_index))
    return unique[np.asarray(selected, dtype=np.int64)].astype(np.float64)


def _merge_tiny_clusters(
    unique: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    *,
    min_pixel_share: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if float(min_pixel_share) <= 0.0:
        compacted = _compact_clusters(unique, counts, labels)
        return compacted, _centers_for_labels(unique, counts, compacted), 0

    total = int(counts.sum())
    if total <= 0:
        return _compact_clusters(unique, counts, labels), _centers_for_labels(unique, counts, labels), 0

    merged = labels.astype(np.int64, copy=True)
    merge_count = 0
    cluster_counts = _cluster_counts(merged, counts)
    tiny_ids = [
        cluster_id
        for cluster_id, count in sorted(cluster_counts.items(), key=lambda item: (item[1], item[0]))
        if count / float(total) < float(min_pixel_share)
    ]
    for cluster_id in tiny_ids:
        cluster_counts = _cluster_counts(merged, counts)
        if cluster_id not in cluster_counts or len(cluster_counts) <= 1:
            continue
        count = cluster_counts[cluster_id]
        candidates = [
            other_id
            for other_id, other_count in cluster_counts.items()
            if other_id != cluster_id and other_count / float(total) >= float(min_pixel_share)
        ]
        if not candidates:
            candidates = [
                other_id
                for other_id, other_count in cluster_counts.items()
                if other_id != cluster_id and other_count > count
            ]
        if not candidates:
            candidates = [other_id for other_id in cluster_counts if other_id != cluster_id]
        if not candidates:
            continue
        source_center = _center_for_cluster(unique, counts, merged, cluster_id, centers)
        target_id = min(
            candidates,
            key=lambda other_id: (
                float(
                    np.sum(np.square(source_center - _center_for_cluster(unique, counts, merged, other_id, centers)))
                ),
                -int(cluster_counts[other_id]),
                int(other_id),
            ),
        )
        merged[merged == cluster_id] = int(target_id)
        merge_count += 1

    compacted = _compact_clusters(unique, counts, merged)
    return compacted, _centers_for_labels(unique, counts, compacted), merge_count


def _cluster_counts(labels: np.ndarray, counts: np.ndarray) -> dict[int, int]:
    output: dict[int, int] = {}
    for label, count in zip(labels, counts, strict=False):
        output[int(label)] = output.get(int(label), 0) + int(count)
    return output


def _center_for_cluster(
    unique: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    centers: np.ndarray,
) -> np.ndarray:
    mask = labels == int(cluster_id)
    if bool(np.any(mask)):
        return np.average(unique[mask].astype(np.float64), axis=0, weights=counts[mask].astype(np.float64))
    if 0 <= int(cluster_id) < centers.shape[0]:
        return centers[int(cluster_id)].astype(np.float64)
    return np.zeros((3,), dtype=np.float64)


def _compact_clusters(unique: np.ndarray, counts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centers = _centers_for_labels(unique, counts, labels)
    old_ids = sorted({int(label) for label in labels})
    old_centers = {old_id: _center_for_cluster(unique, counts, labels, old_id, centers) for old_id in old_ids}
    ordered = sorted(
        old_ids,
        key=lambda old_id: (
            -sum(int(count) for label, count in zip(labels, counts, strict=False) if int(label) == old_id),
            round(float(old_centers[old_id][0])),
            round(float(old_centers[old_id][1])),
            round(float(old_centers[old_id][2])),
            old_id,
        ),
    )
    remap = {old_id: new_id for new_id, old_id in enumerate(ordered)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int64)


def _centers_for_labels(unique: np.ndarray, counts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centers: list[np.ndarray] = []
    for cluster_id in sorted({int(label) for label in labels}):
        mask = labels == cluster_id
        centers.append(np.average(unique[mask].astype(np.float64), axis=0, weights=counts[mask].astype(np.float64)))
    return np.stack(centers, axis=0) if centers else np.zeros((0, 3), dtype=np.float64)


def _color_distribution_metrics(
    rgba: np.ndarray,
    visible: np.ndarray,
    *,
    min_pixel_share: float,
) -> dict[str, Any]:
    if not bool(np.any(visible)):
        return {
            "visible_color_count": 0,
            "rare_color_count": 0,
            "rare_color_rate": 0.0,
        }
    colors, counts = np.unique(np.asarray(rgba)[..., :3][visible], axis=0, return_counts=True)
    total = int(counts.sum())
    rare = counts.astype(np.float64) / float(total) < float(min_pixel_share)
    rare_pixels = int(np.sum(counts[rare])) if bool(np.any(rare)) else 0
    return {
        "visible_color_count": int(colors.shape[0]),
        "rare_color_count": int(np.count_nonzero(rare)),
        "rare_color_rate": float(rare_pixels / float(total)) if total else 0.0,
    }


def _aggregate_report(
    *,
    config: PaletteProjectionConfig,
    generated_dir: Path,
    out_dir: Path,
    samples: list[Mapping[str, Any]],
    records: list[Mapping[str, Any]],
    contact_sheets: Mapping[str, Path],
) -> dict[str, Any]:
    labels = Counter(str(sample.get("destructiveness") or "") for sample in samples)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_dir": str(generated_dir),
        "out_dir": str(out_dir),
        "sample_count": len(samples),
        "method": config.method,
        "target_colors": int(config.target_colors),
        "min_pixel_share": float(config.min_pixel_share),
        "alpha_threshold": float(config.alpha_threshold),
        "contact_sheets": {key: str(path) for key, path in sorted(contact_sheets.items())},
        "median_visible_color_count_before": _median(samples, "visible_color_count_before"),
        "median_visible_color_count_after": _median(samples, "visible_color_count_after"),
        "mean_visible_color_count_before": _mean(samples, "visible_color_count_before"),
        "mean_visible_color_count_after": _mean(samples, "visible_color_count_after"),
        "rare_color_rate_before": _mean(samples, "rare_color_rate_before"),
        "rare_color_rate_after": _mean(samples, "rare_color_rate_after"),
        "mean_rgb_mae_visible": _mean(samples, "rgb_mae_visible"),
        "median_rgb_mae_visible": _median(samples, "rgb_mae_visible"),
        "max_rgb_mae_visible": _max(samples, "rgb_mae_visible"),
        "max_rgb_error_visible": _max(samples, "max_rgb_error_visible"),
        "safe_count": int(labels.get("safe", 0)),
        "moderate_count": int(labels.get("moderate", 0)),
        "destructive_count": int(labels.get("destructive", 0)),
        "destructive_rate": float(labels.get("destructive", 0) / max(len(samples), 1)),
        "alpha_changed_pixels": int(sum(int(sample.get("alpha_changed_pixels") or 0) for sample in samples)),
        "tiny_cluster_merge_count": int(sum(int(sample.get("tiny_cluster_merge_count") or 0) for sample in samples)),
        "max_visible_color_count_after": max(
            [int(sample.get("visible_color_count_after") or 0) for sample in samples],
            default=0,
        ),
        "samples_path": "palette_projection_samples.jsonl",
        "projected_manifest": "generated_manifest.jsonl",
        "generation_report": "generation_report.json",
        "samples": [_jsonable(sample) for sample in samples],
        "record_count": len(records),
    }
    return report


def format_palette_projection_markdown(report: Mapping[str, Any]) -> str:
    samples = [sample for sample in report.get("samples", []) if isinstance(sample, Mapping)]
    destructive = [sample for sample in samples if sample.get("destructiveness") == "destructive"]
    improved = [
        sample
        for sample in samples
        if float(sample.get("rare_color_rate_after") or 0.0) < float(sample.get("rare_color_rate_before") or 0.0)
        or int(sample.get("rare_color_count_after") or 0) < int(sample.get("rare_color_count_before") or 0)
    ]
    lines = [
        "# Palette Projection Report",
        "",
        f"Generated: `{report.get('generated_dir', '')}`",
        f"Projected: `{report.get('out_dir', '')}`",
        f"Samples: {int(report.get('sample_count') or 0)}",
        "",
        "## Summary",
        "",
        f"- Method: `{report.get('method', '')}`",
        f"- Target colors: {int(report.get('target_colors') or 0)}",
        f"- Median visible colors: {_fmt(report.get('median_visible_color_count_before'))} -> {_fmt(report.get('median_visible_color_count_after'))}",
        f"- Mean visible colors: {_fmt(report.get('mean_visible_color_count_before'))} -> {_fmt(report.get('mean_visible_color_count_after'))}",
        f"- Rare-color rate: {_fmt(report.get('rare_color_rate_before'))} -> {_fmt(report.get('rare_color_rate_after'))}",
        f"- RGB MAE visible mean: {_fmt(report.get('mean_rgb_mae_visible'))}",
        f"- Safe / moderate / destructive: {int(report.get('safe_count') or 0)} / {int(report.get('moderate_count') or 0)} / {int(report.get('destructive_count') or 0)}",
        "",
        "## Worst RGB MAE",
        "",
        "| Sample | Before colors | After colors | RGB MAE | Label | Prompt |",
        "|---|---:|---:|---:|---|---|",
    ]
    for sample in sorted(samples, key=lambda item: float(item.get("rgb_mae_visible") or 0.0), reverse=True)[:20]:
        lines.append(_sample_row(sample, "rgb_mae_visible"))

    lines.extend(
        [
            "",
            "## Worst Max RGB Error",
            "",
            "| Sample | Before colors | After colors | Max RGB error | Label | Prompt |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for sample in sorted(samples, key=lambda item: float(item.get("max_rgb_error_visible") or 0.0), reverse=True)[:20]:
        lines.append(_sample_row(sample, "max_rgb_error_visible"))

    lines.extend(
        [
            "",
            "## Rare Colors Improved",
            "",
            "| Sample | Rare count | Rare rate | RGB MAE | Label | Prompt |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for sample in sorted(
        improved,
        key=lambda item: (
            float(item.get("rare_color_rate_before") or 0.0) - float(item.get("rare_color_rate_after") or 0.0),
            int(item.get("rare_color_count_before") or 0) - int(item.get("rare_color_count_after") or 0),
        ),
        reverse=True,
    )[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_escape(str(sample.get('sample_id') or ''))}`",
                    f"{int(sample.get('rare_color_count_before') or 0)} -> {int(sample.get('rare_color_count_after') or 0)}",
                    f"{_fmt(sample.get('rare_color_rate_before'))} -> {_fmt(sample.get('rare_color_rate_after'))}",
                    _fmt(sample.get("rgb_mae_visible")),
                    _md_escape(str(sample.get("destructiveness") or "")),
                    _md_escape(str(sample.get("prompt") or ""))[:120],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Destructive Projections",
            "",
            "| Sample | RGB MAE | Max RGB error | Before colors | After colors | Prompt |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for sample in sorted(destructive, key=lambda item: float(item.get("rgb_mae_visible") or 0.0), reverse=True)[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_escape(str(sample.get('sample_id') or ''))}`",
                    _fmt(sample.get("rgb_mae_visible")),
                    _fmt(sample.get("max_rgb_error_visible")),
                    str(int(sample.get("visible_color_count_before") or 0)),
                    str(int(sample.get("visible_color_count_after") or 0)),
                    _md_escape(str(sample.get("prompt") or ""))[:120],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _sample_row(sample: Mapping[str, Any], metric_key: str) -> str:
    return (
        "| "
        + " | ".join(
            [
                f"`{_md_escape(str(sample.get('sample_id') or ''))}`",
                str(int(sample.get("visible_color_count_before") or 0)),
                str(int(sample.get("visible_color_count_after") or 0)),
                _fmt(sample.get(metric_key)),
                _md_escape(str(sample.get("destructiveness") or "")),
                _md_escape(str(sample.get("prompt") or ""))[:120],
            ]
        )
        + " |"
    )


def _write_projection_reports(out_dir: Path, report: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> None:
    (out_dir / "palette_projection_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "palette_projection_report.md").write_text(format_palette_projection_markdown(report), encoding="utf-8")
    (out_dir / "palette_projection_samples.jsonl").write_text(
        "".join(json.dumps(_jsonable(sample), sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )


def _write_generation_report(
    out_dir: Path,
    records: Sequence[Mapping[str, Any]],
    projection_report: Mapping[str, Any],
    contact_sheet: Path | None,
) -> None:
    warnings = sum(len(record.get("warnings") or []) for record in records)
    fully_transparent = sum(1 for record in records if int(record.get("alpha_opaque_count") or 0) == 0)
    report = {
        "sample_count": len(records),
        "warnings": int(warnings),
        "fully_transparent_count": int(fully_transparent),
        "max_visible_color_count": int(projection_report.get("max_visible_color_count_after") or 0),
        "contact_sheet": None if contact_sheet is None else contact_sheet.name,
        "manifest": "generated_manifest.jsonl",
        "config": {
            "palette_projection": {
                "schema_version": SCHEMA_VERSION,
                "source_generated_dir": projection_report.get("generated_dir"),
                "method": projection_report.get("method"),
                "target_colors": projection_report.get("target_colors"),
                "min_pixel_share": projection_report.get("min_pixel_share"),
                "alpha_threshold": projection_report.get("alpha_threshold"),
            }
        },
    }
    (out_dir / "generation_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Generated Sprite Report",
        "",
        f"Samples: {len(records)}",
        f"Warnings: {warnings}",
        f"Fully transparent: {fully_transparent}",
        f"Max visible colors: {report['max_visible_color_count']}",
        "",
        "Projected with decode-time palette projection.",
        "",
    ]
    (out_dir / "generation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_projection_contact_sheets(
    generated_dir: Path,
    out_dir: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    max_samples: int,
) -> dict[str, Path]:
    sheets: dict[str, Path] = {}
    before = out_dir / "palette_projection_before_contact_sheet.png"
    after = out_dir / "palette_projection_after_contact_sheet.png"
    if _build_projection_contact_sheet(generated_dir, samples, before, path_key="before_path", max_samples=max_samples):
        sheets["before"] = before
    if _build_projection_contact_sheet(out_dir, samples, after, path_key="after_path", max_samples=max_samples):
        sheets["after"] = after
    return sheets


def _build_projection_contact_sheet(
    root: Path,
    samples: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    path_key: str,
    max_samples: int,
    scale: int = 4,
) -> bool:
    images: list[Image.Image] = []
    for sample in samples[: max(0, int(max_samples))]:
        rel = str(sample.get(path_key) or "")
        if not rel:
            continue
        path = _resolve_generated_path(root, rel)
        if not path.is_file():
            continue
        try:
            image = Image.open(path).convert("RGBA")
            image.load()
        except Exception:
            continue
        if image.size == (SPRITE_SIZE, SPRITE_SIZE):
            images.append(image)
    if not images:
        return False

    columns = min(8, max(1, len(images)))
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGBA",
        (columns * cell + (columns + 1) * padding, rows * cell + (rows + 1) * padding),
        (38, 38, 42, 255),
    )
    for index, image in enumerate(images):
        col = index % columns
        row = index // columns
        left = padding + col * (cell + padding)
        top = padding + row * (cell + padding)
        sheet.alpha_composite(checkerboard_rgba(image).resize((cell, cell), Image.Resampling.NEAREST), (left, top))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def _build_runtime_projection_contact_sheet(
    root: Path,
    samples: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    max_samples: int,
    scale: int = 4,
) -> bool:
    pairs: list[tuple[Image.Image, Image.Image]] = []
    for sample in samples[: max(0, int(max_samples))]:
        before_rel = str(sample.get("before_path") or "")
        after_rel = str(sample.get("after_path") or "")
        if not before_rel or not after_rel:
            continue
        before_path = _resolve_generated_path(root, before_rel)
        after_path = _resolve_generated_path(root, after_rel)
        if not before_path.is_file() or not after_path.is_file():
            continue
        try:
            before = Image.open(before_path).convert("RGBA")
            before.load()
            after = Image.open(after_path).convert("RGBA")
            after.load()
        except Exception:
            continue
        if before.size == (SPRITE_SIZE, SPRITE_SIZE) and after.size == (SPRITE_SIZE, SPRITE_SIZE):
            pairs.append((before, after))
    if not pairs:
        return False

    columns = min(6, max(1, len(pairs)))
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    tile_w = cell * 2 + padding
    tile_h = cell
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new(
        "RGBA",
        (columns * tile_w + (columns + 1) * padding, rows * tile_h + (rows + 1) * padding),
        (38, 38, 42, 255),
    )
    for index, (before, after) in enumerate(pairs):
        col = index % columns
        row = index // columns
        left = padding + col * (tile_w + padding)
        top = padding + row * (tile_h + padding)
        sheet.alpha_composite(checkerboard_rgba(before).resize((cell, cell), Image.Resampling.NEAREST), (left, top))
        sheet.alpha_composite(
            checkerboard_rgba(after).resize((cell, cell), Image.Resampling.NEAREST),
            (left + cell + padding, top),
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return True


def _copy_ancillary_metadata(generated_dir: Path, out_dir: Path) -> None:
    skip_names = {
        "generated_manifest.jsonl",
        "generation_report.json",
        "generation_report.md",
        "generation_contact_sheet.png",
        "generated_review_report.json",
        "generated_review_report.md",
        "generated_qa_report.json",
        "generated_qa_report.md",
        "prompt_faithfulness_report.json",
        "prompt_faithfulness_report.md",
        "palette_projection_report.json",
        "palette_projection_report.md",
        "palette_projection_samples.jsonl",
        "palette_projection_before_contact_sheet.png",
        "palette_projection_after_contact_sheet.png",
    }
    for path in generated_dir.iterdir():
        if path.name in skip_names or path.name in IMAGE_PATH_KEYS:
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        target = out_dir / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _primary_image_key(paths: Mapping[str, Any]) -> str | None:
    for key in ("indexed_png", "hard_rgba", "raw_rgba"):
        if paths.get(key):
            return key
    return None


def _validate_projection_options(
    *,
    target_colors: int,
    min_pixel_share: float,
    alpha_threshold: float,
    method: str,
) -> None:
    if method not in PROJECTION_METHODS:
        raise ValueError(f"unsupported projection method {method!r}; expected one of {', '.join(PROJECTION_METHODS)}")
    if int(target_colors) < 1:
        raise ValueError("target_colors must be at least 1")
    if int(target_colors) > 256:
        raise ValueError("target_colors must be at most 256")
    if float(min_pixel_share) < 0.0 or float(min_pixel_share) >= 1.0:
        raise ValueError("min_pixel_share must be in [0, 1)")
    if not 0.0 <= float(alpha_threshold) <= 1.0:
        raise ValueError("alpha_threshold must be in 0..1")


def _rgba_to_uint8(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"rgba must have shape [H, W, 4], got {arr.shape}")
    value = arr.astype(np.float32, copy=False)
    if arr.dtype.kind in "f" and value.size and float(np.nanmax(value)) <= 1.0:
        value = value * 255.0
    return np.rint(np.clip(value, 0.0, 255.0)).astype(np.uint8)


def _restore_input_range(value: np.ndarray, original: np.ndarray) -> np.ndarray:
    original_arr = np.asarray(original)
    if original_arr.dtype.kind in "f":
        original_float = original_arr.astype(np.float32, copy=False)
        if original_float.size and float(np.nanmax(original_float)) <= 1.0:
            return (value.astype(np.float32) / 255.0).astype(np.float32, copy=False)
    return value.astype(np.uint8, copy=False)


def _visible_mask(rgba: np.ndarray, *, alpha_threshold: float) -> np.ndarray:
    return np.asarray(rgba, dtype=np.uint8)[..., 3] >= _alpha_threshold_byte(alpha_threshold)


def _alpha_threshold_byte(alpha_threshold: float) -> int:
    return math.ceil(float(alpha_threshold) * 255.0)


def _destructiveness_label(rgb_mae_visible: float) -> str:
    if float(rgb_mae_visible) <= 0.03:
        return "safe"
    if float(rgb_mae_visible) <= 0.06:
        return "moderate"
    return "destructive"


def _load_rgba(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image.convert("RGBA")


def _save_rgba(rgba: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgba_to_uint8(rgba), mode="RGBA").save(path)


def _resolve_generated_path(generated_dir: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return generated_dir / Path(str(value).replace("\\", "/"))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        records.append(value)
    return records


def _write_manifest(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonable(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _safe_sample_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value).strip())
    return cleaned or "sample"


def _mean(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _numeric_values(samples, key)
    return float(statistics.fmean(values)) if values else None


def _median(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _numeric_values(samples, key)
    return float(statistics.median(values)) if values else None


def _max(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _numeric_values(samples, key)
    return float(max(values)) if values else None


def _numeric_values(samples: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "NA" if value is None else str(value)


def _md_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Project generated samples to deterministic per-image palettes.")
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-colors", type=int, default=16)
    parser.add_argument("--min-pixel-share", type=float, default=0.01)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--method", choices=PROJECTION_METHODS, default="deterministic_kmeans")
    parsed = parser.parse_args(argv)
    report = project_generated_palette(PaletteProjectionConfig(**vars(parsed)))
    print(f"Projected samples: {report['sample_count']}")
    print(
        f"Median visible colors: {report['median_visible_color_count_before']} -> {report['median_visible_color_count_after']}"
    )
    print(f"Mean RGB MAE visible: {report['mean_rgb_mae_visible']}")
    print(f"Outputs written to {parsed.out}")
