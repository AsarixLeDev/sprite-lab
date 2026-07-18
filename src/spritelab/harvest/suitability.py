"""Deterministic, semantics-independent suitability audit for sprite PNGs."""

from __future__ import annotations

import hashlib
import io
import json
import math
import warnings
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

SCHEMA_VERSION = "sprite_suitability_v1"
CONFIG_VERSION = "sprite_suitability_thresholds_v1"
PROFILE_NAMES = (
    "single_object_32px",
    "single_object_source_resolution",
    "environment_tile",
    "effect_icon",
    "character_sprite",
)
SUPPORTED_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})


@dataclass(frozen=True)
class SuitabilityProfile:
    name: str
    target_width: int | None = 32
    target_height: int | None = 32
    min_dimension: int = 4
    max_dimension: int = 1024
    min_foreground_pixels: int = 8
    min_occupancy: float = 0.025
    max_occupancy: float = 0.80
    min_bbox_dimension: int = 4
    max_padding_ratio: float = 0.82
    max_semitransparent_ratio: float = 0.015
    reject_semitransparent_ratio: float = 0.35
    max_palette_colors: int = 48
    reject_palette_colors: int = 192
    max_smooth_gradient_ratio: float = 0.18
    reject_smooth_gradient_ratio: float = 0.52
    max_significant_components: int = 3
    reject_unrelated_components: int = 4
    significant_component_min_pixels: int = 4
    significant_component_min_ratio: float = 0.025
    speckle_max_pixels: int = 2
    max_speckle_ratio: float = 0.04
    detached_shadow_max_ratio: float = 0.22
    detached_shadow_max_gap: int = 3
    sheet_aspect_ratio: float = 2.6
    reject_sheet_regions: int = 4
    alpha_threshold: int = 8
    background_color_tolerance: int = 8


def _profiles() -> dict[str, SuitabilityProfile]:
    base = SuitabilityProfile(name="single_object_32px")
    return {
        base.name: base,
        "single_object_source_resolution": replace(
            base, name="single_object_source_resolution", target_width=None, target_height=None, max_dimension=512
        ),
        "environment_tile": replace(
            base,
            name="environment_tile",
            max_occupancy=1.0,
            min_occupancy=0.0,
            max_padding_ratio=1.0,
            max_significant_components=12,
            reject_unrelated_components=99,
            reject_sheet_regions=99,
        ),
        "effect_icon": replace(
            base,
            name="effect_icon",
            max_semitransparent_ratio=0.12,
            reject_semitransparent_ratio=0.45,
            max_palette_colors=80,
            reject_palette_colors=256,
            max_significant_components=7,
        ),
        "character_sprite": replace(
            base,
            name="character_sprite",
            max_occupancy=0.94,
            max_significant_components=5,
            reject_unrelated_components=7,
            sheet_aspect_ratio=3.2,
        ),
    }


@dataclass(frozen=True)
class SuitabilityConfig:
    version: str = CONFIG_VERSION
    profile: SuitabilityProfile = field(default_factory=lambda: _profiles()["single_object_32px"])

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "profile": asdict(self.profile)}

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SuitabilityInput:
    sprite_id: str
    image_path: Path
    resize_history: tuple[str, ...] = ()
    source_run: str = ""
    image_bytes: bytes | None = None


@dataclass
class SuitabilityResult:
    schema_version: str
    sprite_id: str
    status: Literal["accept", "quarantine", "reject"]
    score: float
    reason_codes: list[str]
    metrics: dict[str, Any]
    warnings: list[str]
    profile: str
    config_hash: str
    image_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DuplicateGroup:
    group_id: str
    kind: str
    member_sprite_ids: tuple[str, ...]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "kind": self.kind,
            "member_sprite_ids": list(self.member_sprite_ids),
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class AuditOutput:
    results: tuple[SuitabilityResult, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    summary: Mapping[str, Any]


def load_config(profile: str = "single_object_32px", config_path: Path | None = None) -> SuitabilityConfig:
    profiles = _profiles()
    if profile not in profiles:
        raise ValueError(f"unknown suitability profile {profile!r}; choose from {', '.join(PROFILE_NAMES)}")
    selected = profiles[profile]
    version = CONFIG_VERSION
    if config_path is not None:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("suitability config must be a JSON object")
        version = str(raw.get("version") or CONFIG_VERSION)
        overrides = raw.get("profile", raw.get("thresholds", {}))
        if not isinstance(overrides, dict):
            raise ValueError("suitability config profile/thresholds must be an object")
        valid = set(asdict(selected))
        unknown = sorted(set(overrides) - valid - {"name"})
        if unknown:
            raise ValueError(f"unknown suitability thresholds: {', '.join(unknown)}")
        selected = SuitabilityProfile(**(asdict(selected) | {k: v for k, v in overrides.items() if k != "name"}))
    return SuitabilityConfig(version=version, profile=selected)


def audit_sprite(item: SuitabilityInput, config: SuitabilityConfig) -> SuitabilityResult:
    hard: list[str] = []
    soft: list[str] = []
    notices: list[str] = []
    metrics: dict[str, Any] = {
        "source_run": item.source_run,
        "resize_history": list(item.resize_history),
        "resize_policy": "native_32_required" if config.profile.target_width == 32 else "preserve_source_resolution",
    }
    path = Path(item.image_path)
    if item.image_bytes is None:
        if not path.is_file():
            return _terminal_result(item.sprite_id, config, "FILE_MISSING", metrics)
        try:
            metrics["file_size_bytes"] = path.stat().st_size
        except OSError:
            return _terminal_result(item.sprite_id, config, "FILE_UNREADABLE", metrics)
        image_source: Path | io.BytesIO = path
    else:
        if not isinstance(item.image_bytes, bytes):
            return _terminal_result(item.sprite_id, config, "FILE_UNREADABLE", metrics)
        metrics["file_size_bytes"] = len(item.image_bytes)
        image_source = io.BytesIO(item.image_bytes)
    if metrics["file_size_bytes"] == 0:
        return _terminal_result(item.sprite_id, config, "FILE_EMPTY", metrics)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Image.open(image_source) as opened:
                opened.load()
                original_mode = opened.mode
                original_format = opened.format or ""
                rgba_image = opened.convert("RGBA")
            for warning in caught:
                notices.append(f"decode warning: {warning.message}")
    except Image.DecompressionBombError:
        return _terminal_result(item.sprite_id, config, "DECOMPRESSION_BOMB", metrics)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        metrics["decode_error"] = str(exc)
        return _terminal_result(item.sprite_id, config, "IMAGE_DECODE_FAILED", metrics)

    rgba = np.asarray(rgba_image, dtype=np.uint8)
    height, width = rgba.shape[:2]
    metrics.update(
        {"native_width": width, "native_height": height, "source_mode": original_mode, "format": original_format}
    )
    image_hash = _decoded_hash(rgba)
    if original_mode not in SUPPORTED_MODES:
        hard.append("UNSUPPORTED_COLOR_MODE")
    p = config.profile
    if width < p.min_dimension or height < p.min_dimension or width > p.max_dimension or height > p.max_dimension:
        hard.append("INVALID_DIMENSIONS")
    elif p.target_width is not None and (width, height) != (p.target_width, p.target_height):
        if width > p.target_width or height > (p.target_height or p.target_width):
            soft.append("NON_TARGET_DIMENSIONS")
        else:
            soft.append("SOURCE_RESOLUTION_TOO_SMALL")

    alpha = rgba[..., 3]
    visible = alpha >= p.alpha_threshold
    visible_count = int(np.count_nonzero(visible))
    semi = (alpha > 0) & (alpha < 255)
    semi_count = int(np.count_nonzero(semi))
    semi_ratio = semi_count / max(1, visible_count)
    metrics.update(
        {
            "visible_pixel_count": visible_count,
            "semi_transparent_pixel_count": semi_count,
            "semi_transparent_ratio": _round(semi_ratio),
            "alpha_value_count": int(np.unique(alpha).size),
        }
    )
    if visible_count == 0:
        hard.append("FULLY_TRANSPARENT")
        return _finish(item, config, metrics, image_hash, hard, soft, notices)
    if visible_count < p.min_foreground_pixels:
        hard.append("EFFECTIVELY_EMPTY")
    if semi_ratio > p.reject_semitransparent_ratio:
        hard.append("INVALID_PARTIAL_ALPHA")
    elif semi_ratio > p.max_semitransparent_ratio:
        soft.append("EXCESSIVE_PARTIAL_ALPHA")

    checker = _checkerboard_evidence(rgba)
    metrics.update(checker)
    if checker["checkerboard_score"] >= 0.90:
        hard.append("BAKED_CHECKERBOARD")

    opaque = bool(np.all(alpha == 255))
    bg = _background_evidence(rgba, tolerance=p.background_color_tolerance)
    inferred_foreground = np.asarray(bg.pop("_foreground_mask"), dtype=bool)
    metrics.update(bg)
    if (
        opaque
        and p.name != "environment_tile"
        and bg["corner_consistency"] >= 0.75
        and bg["background_area_ratio"] >= 0.30
    ):
        hard.append("OPAQUE_RECTANGULAR_BACKGROUND")
        foreground = inferred_foreground
    else:
        foreground = visible
        if opaque and bg["corner_consistency"] < 0.75:
            soft.append("NONTRANSPARENT_CORNER_BACKGROUND")

    rgb_visible = rgba[..., :3][foreground]
    unique_rgba = int(np.unique(rgba[visible], axis=0).shape[0])
    unique_rgb = int(np.unique(rgb_visible, axis=0).shape[0]) if rgb_visible.size else 0
    palette_compactness = _palette_compactness(rgb_visible)
    gradient_ratio = _smooth_gradient_ratio(rgba[..., :3], foreground)
    antialias_ratio = _antialias_edge_ratio(rgba, foreground)
    metrics.update(
        {
            "unique_rgba_colors": unique_rgba,
            "unique_rgb_colors": unique_rgb,
            "palette_compactness": _round(palette_compactness),
            "smooth_gradient_ratio": _round(gradient_ratio),
            "antialiased_edge_ratio": _round(antialias_ratio),
        }
    )
    if unique_rgb > p.reject_palette_colors and (
        gradient_ratio > p.reject_smooth_gradient_ratio or palette_compactness < 0.25
    ):
        hard.append("PHOTOGRAPHIC_OR_PAINTED")
    else:
        if unique_rgb > p.max_palette_colors:
            soft.append("LARGE_PALETTE")
        if gradient_ratio > p.max_smooth_gradient_ratio:
            soft.append("SMOOTH_GRADIENT_EVIDENCE")
        if antialias_ratio > 0.20:
            soft.append("ANTIALIASED_EDGE_EVIDENCE")
    if gradient_ratio > p.reject_smooth_gradient_ratio and unique_rgb > max(64, p.max_palette_colors):
        hard.append("SEVERE_INTERPOLATION")
    elif gradient_ratio > p.max_smooth_gradient_ratio and (semi_ratio > 0.02 or unique_rgb > p.max_palette_colors):
        soft.append("ACCIDENTAL_INTERPOLATION")

    scale_factor, scale_consistency = _nearest_scale_consistency(rgba)
    metrics["nearest_neighbor_scale_factor"] = scale_factor
    metrics["nearest_neighbor_scale_consistency"] = _round(scale_consistency)

    geometry = _geometry_metrics(foreground, p)
    metrics.update(geometry)
    occupancy = float(geometry["foreground_occupancy"])
    if occupancy < p.min_occupancy:
        soft.append("EXCESSIVE_EMPTY_SPACE")
    if float(geometry["padding_ratio"]) > p.max_padding_ratio:
        soft.append("EXCESSIVE_PADDING")
    if occupancy > p.max_occupancy:
        soft.append("OVERSIZED_FOREGROUND")
    if min(int(geometry["bbox_width"]), int(geometry["bbox_height"])) < p.min_bbox_dimension:
        soft.append("THIN_OR_UNREADABLE_SILHOUETTE")
    edges = int(geometry["border_edges_touched"])
    at_target_size = p.target_width is None or (width, height) == (p.target_width, p.target_height)
    if edges >= 3 and occupancy >= 0.65 and at_target_size:
        hard.append("SEVERELY_CLIPPED_FOREGROUND")
    elif edges:
        soft.append("FOREGROUND_TOUCHES_BORDER")
    if float(geometry["speckle_pixel_ratio"]) > p.max_speckle_ratio:
        soft.append("ISOLATED_SPECKLES")

    significant = int(geometry["significant_component_count"])
    detached_shadow = bool(geometry["detached_shadow_likely"])
    if detached_shadow:
        notices.append("small low detached component retained as possible shadow/highlight")
    if significant >= p.reject_unrelated_components and float(geometry["largest_component_ratio"]) < 0.62:
        hard.append("MULTIPLE_UNRELATED_OBJECTS")
    elif significant > p.max_significant_components and not detached_shadow:
        soft.append("MULTIPART_COMPOSITION")

    layout = _layout_evidence(foreground, p)
    metrics.update(layout)
    if int(layout["strip_region_count"]) >= p.reject_sheet_regions and float(layout["strip_confidence"]) >= 0.80:
        hard.append("SPRITE_SHEET_OR_ANIMATION_STRIP")
    elif int(layout["strip_region_count"]) >= 2 and float(layout["strip_confidence"]) >= 0.55:
        soft.append("POSSIBLE_SPRITE_STRIP")
    if float(layout["frame_score"]) >= 0.78:
        soft.append("POSSIBLE_UI_FRAME")
    if float(layout["panel_score"]) >= 0.78:
        soft.append("POSSIBLE_UI_PANEL_OR_COLLECTION")
    if float(layout["text_like_score"]) >= 0.72:
        soft.append("POSSIBLE_TEXT_OR_LABEL")

    halo_ratio = _alpha_halo_ratio(rgba, foreground)
    metrics["alpha_halo_ratio"] = _round(halo_ratio)
    if halo_ratio > 0.25:
        soft.append("ALPHA_FRINGE_OR_HALO")
    matte = _matte_metrics(rgba)
    metrics.update(matte)
    if semi_count and matte["transparent_rgb_nonzero_ratio"] > 0.10 and matte["transparent_rgb_color_count"] > 4:
        soft.append("INCONSISTENT_MATTE_COLORS")

    metrics.update(_duplicate_fingerprints(rgba, foreground))
    return _finish(item, config, metrics, image_hash, hard, soft, notices)


def audit_inputs(items: Sequence[SuitabilityInput], config: SuitabilityConfig) -> AuditOutput:
    ordered = sorted(items, key=lambda item: (item.sprite_id, item.image_path.as_posix()))
    results = [audit_sprite(item, config) for item in ordered]
    groups = _build_duplicate_groups(results)
    memberships: dict[str, list[dict[str, str]]] = {}
    for group in groups:
        for sprite_id in group.member_sprite_ids:
            memberships.setdefault(sprite_id, []).append({"group_id": group.group_id, "kind": group.kind})
    for result in results:
        result.metrics["duplicate_groups"] = sorted(
            memberships.get(result.sprite_id, []), key=lambda value: (value["kind"], value["group_id"])
        )
    status_counts = Counter(result.status for result in results)
    reason_counts = Counter(code for result in results for code in result.reason_codes)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_version": config.version,
        "config_hash": config.config_hash,
        "profile": config.profile.name,
        "total": len(results),
        "status_counts": {name: status_counts.get(name, 0) for name in ("accept", "quarantine", "reject")},
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "duplicate_group_counts": dict(sorted(Counter(group.kind for group in groups).items())),
    }
    return AuditOutput(tuple(results), tuple(groups), summary)


def write_audit_output(output: AuditOutput, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "suitability_results.jsonl", [result.to_dict() for result in output.results])
    _write_json(out_dir / "summary.json", output.summary)
    _write_json(out_dir / "duplicate_groups.json", {"groups": [group.to_dict() for group in output.duplicate_groups]})
    for status, filename in (("accept", "accepted.txt"), ("quarantine", "quarantined.txt"), ("reject", "rejected.txt")):
        values = [result.sprite_id for result in output.results if result.status == status]
        (out_dir / filename).write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    (out_dir / "report.md").write_text(_render_markdown(output), encoding="utf-8")


def _finish(
    item: SuitabilityInput,
    config: SuitabilityConfig,
    metrics: dict[str, Any],
    image_hash: str,
    hard: Sequence[str],
    soft: Sequence[str],
    notices: Sequence[str],
) -> SuitabilityResult:
    hard_codes = sorted(set(hard))
    soft_codes = sorted(set(soft) - set(hard_codes))
    reason_codes = [*hard_codes, *soft_codes]
    status: Literal["accept", "quarantine", "reject"] = (
        "reject" if hard_codes else "quarantine" if soft_codes else "accept"
    )
    score = max(0.0, 100.0 - 30.0 * len(hard_codes) - 8.0 * len(soft_codes))
    metrics["hard_reject_reason_codes"] = hard_codes
    return SuitabilityResult(
        schema_version=SCHEMA_VERSION,
        sprite_id=item.sprite_id,
        status=status,
        score=_round(score),
        reason_codes=reason_codes,
        metrics=_jsonable(metrics),
        warnings=sorted({str(value) for value in notices}),
        profile=config.profile.name,
        config_hash=config.config_hash,
        image_hash=image_hash,
    )


def _terminal_result(
    sprite_id: str, config: SuitabilityConfig, code: str, metrics: dict[str, Any]
) -> SuitabilityResult:
    item = SuitabilityInput(sprite_id=sprite_id, image_path=Path("."))
    return _finish(item, config, metrics, "", [code], [], [])


def _decoded_hash(rgba: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(rgba.shape, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(rgba).tobytes())
    return digest.hexdigest()


def _background_evidence(rgba: np.ndarray, tolerance: int) -> dict[str, Any]:
    rgb = rgba[..., :3].astype(np.int16)
    corners = np.asarray([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    colors, counts = np.unique(corners, axis=0, return_counts=True)
    background = colors[int(np.argmax(counts))]
    corner_consistency = int(np.max(counts)) / 4.0
    distance = np.max(np.abs(rgb - background), axis=2)
    background_mask = distance <= tolerance
    area = float(np.mean(background_mask))
    return {
        "corner_consistency": _round(corner_consistency),
        "inferred_background_rgb": [int(value) for value in background],
        "background_area_ratio": _round(area),
        "_foreground_mask": ~background_mask,
    }


def _checkerboard_evidence(rgba: np.ndarray) -> dict[str, Any]:
    rgb = rgba[..., :3]
    height, width = rgb.shape[:2]
    _, inverse, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    if len(order) < 2:
        return {"checkerboard_score": 0.0, "checkerboard_block_size": 0}
    a, b = int(order[0]), int(order[1])
    coverage = float((counts[a] + counts[b]) / (height * width))
    labels = inverse.reshape(height, width)
    best_score, best_block = 0.0, 0
    for block in (1, 2, 4, 8, 16):
        if block > min(width, height) // 2:
            continue
        yy, xx = np.indices((height, width))
        pattern = ((xx // block + yy // block) % 2).astype(bool)
        match1 = np.mean(np.where(pattern, labels == a, labels == b))
        match2 = np.mean(np.where(pattern, labels == b, labels == a))
        score = float(max(match1, match2) * coverage)
        if score > best_score:
            best_score, best_block = score, block
    return {"checkerboard_score": _round(best_score), "checkerboard_block_size": best_block}


def _palette_compactness(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    _, counts = np.unique(rgb, axis=0, return_counts=True)
    keep = min(16, len(counts))
    return float(np.sort(counts)[-keep:].sum() / counts.sum())


def _smooth_gradient_ratio(rgb: np.ndarray, mask: np.ndarray) -> float:
    rgb16 = rgb.astype(np.int16)
    scores: list[np.ndarray] = []
    for axis in (0, 1):
        diff = np.abs(np.diff(rgb16, axis=axis)).max(axis=2)
        pair_mask = (mask[:-1, :] & mask[1:, :]) if axis == 0 else (mask[:, :-1] & mask[:, 1:])
        valid = diff[pair_mask]
        if valid.size:
            scores.append((valid > 0) & (valid <= 12))
    if not scores:
        return 0.0
    joined = np.concatenate(scores)
    return float(np.mean(joined))


def _antialias_edge_ratio(rgba: np.ndarray, foreground: np.ndarray) -> float:
    alpha = rgba[..., 3]
    boundary = _boundary_mask(foreground)
    if not np.any(boundary):
        return 0.0
    semi = (alpha > 0) & (alpha < 255)
    return float(np.count_nonzero(semi & boundary) / max(1, np.count_nonzero(boundary)))


def _alpha_halo_ratio(rgba: np.ndarray, foreground: np.ndarray) -> float:
    semi = (rgba[..., 3] > 0) & (rgba[..., 3] < 255)
    if not np.any(semi):
        return 0.0
    boundary = _boundary_mask(foreground)
    return float(np.count_nonzero(semi & boundary) / max(1, np.count_nonzero(semi)))


def _matte_metrics(rgba: np.ndarray) -> dict[str, Any]:
    transparent_rgb = rgba[..., :3][rgba[..., 3] == 0]
    if transparent_rgb.size == 0:
        return {"transparent_rgb_nonzero_ratio": 0.0, "transparent_rgb_color_count": 0}
    nonzero = np.any(transparent_rgb != 0, axis=1)
    return {
        "transparent_rgb_nonzero_ratio": _round(float(np.mean(nonzero))),
        "transparent_rgb_color_count": int(np.unique(transparent_rgb, axis=0).shape[0]),
    }


def _nearest_scale_consistency(rgba: np.ndarray) -> tuple[int, float]:
    height, width = rgba.shape[:2]
    best_factor, best = 1, 0.0
    for factor in (2, 3, 4, 5, 6, 8, 16):
        if width % factor or height % factor:
            continue
        blocks = rgba.reshape(height // factor, factor, width // factor, factor, 4)
        first = blocks[:, :1, :, :1, :]
        score = float(np.mean(blocks == first))
        if score > best:
            best_factor, best = factor, score
    return (best_factor, best) if best >= 0.985 else (1, best)


def _geometry_metrics(mask: np.ndarray, profile: SuitabilityProfile) -> dict[str, Any]:
    height, width = mask.shape
    count = int(np.count_nonzero(mask))
    ys, xs = np.nonzero(mask)
    if not count:
        return {
            "foreground_occupancy": 0.0,
            "tight_bbox": None,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_fill_ratio": 0.0,
            "padding_ratio": 1.0,
            "border_edges_touched": 0,
            "border_contact_ratio": 0.0,
            "connected_component_count": 0,
            "significant_component_count": 0,
            "component_sizes": [],
            "largest_component_ratio": 0.0,
            "speckle_count": 0,
            "speckle_pixel_ratio": 0.0,
            "detached_shadow_likely": False,
        }
    left, right, top, bottom = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    bbox_area = (right - left) * (bottom - top)
    components = _components(mask)
    sizes = [len(points) for points in components]
    threshold = max(
        profile.significant_component_min_pixels, math.ceil(count * profile.significant_component_min_ratio)
    )
    significant = [points for points in components if len(points) >= threshold]
    speckles = [size for size in sizes if size <= profile.speckle_max_pixels]
    edges = [bool(mask[0].any()), bool(mask[-1].any()), bool(mask[:, 0].any()), bool(mask[:, -1].any())]
    border_count = int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
    detached = _detached_shadow_likely(significant, count, profile)
    return {
        "foreground_occupancy": _round(count / (width * height)),
        "tight_bbox": [left, top, right, bottom],
        "bbox_width": right - left,
        "bbox_height": bottom - top,
        "bbox_fill_ratio": _round(count / max(1, bbox_area)),
        "padding_ratio": _round(1.0 - bbox_area / (width * height)),
        "border_edges_touched": sum(edges),
        "border_contact_ratio": _round(border_count / count),
        "connected_component_count": len(components),
        "significant_component_count": len(significant),
        "component_sizes": sorted(sizes, reverse=True)[:32],
        "largest_component_ratio": _round(max(sizes) / count),
        "speckle_count": len(speckles),
        "speckle_pixel_ratio": _round(sum(speckles) / count),
        "detached_shadow_likely": detached,
    }


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for y, x in zip(*np.nonzero(mask), strict=False):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        points: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            points.append((cy, cx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        result.append(points)
    return result


def _detached_shadow_likely(
    components: Sequence[Sequence[tuple[int, int]]], total: int, profile: SuitabilityProfile
) -> bool:
    if len(components) != 2:
        return False
    ordered = sorted(components, key=len, reverse=True)
    small = ordered[1]
    if len(small) / total > profile.detached_shadow_max_ratio:
        return False
    main_bottom = max(y for y, _ in ordered[0])
    small_top = min(y for y, _ in small)
    return 0 <= small_top - main_bottom <= profile.detached_shadow_max_gap


def _layout_evidence(mask: np.ndarray, profile: SuitabilityProfile) -> dict[str, Any]:
    height, width = mask.shape
    x_regions = _projection_regions(mask.any(axis=0))
    y_regions = _projection_regions(mask.any(axis=1))
    aspect = max(width / max(1, height), height / max(1, width))
    strip_regions = max(len(x_regions), len(y_regions))
    regularity = _region_regularity(x_regions if len(x_regions) >= len(y_regions) else y_regions)
    strip_confidence = min(
        1.0, (0.35 if aspect >= profile.sheet_aspect_ratio else 0.0) + 0.35 * regularity + 0.1 * strip_regions
    )
    border = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    border_possible = 2 * width + 2 * height
    inner_empty = 1.0 - float(mask[2:-2, 2:-2].mean()) if width > 4 and height > 4 else 0.0
    frame_score = min(1.0, border / max(1, border_possible) * 0.75 + inner_empty * 0.35)
    row_transitions = np.count_nonzero(mask[:, 1:] != mask[:, :-1], axis=1)
    col_transitions = np.count_nonzero(mask[1:, :] != mask[:-1, :], axis=0)
    text_like = float(np.mean(row_transitions >= 6) * 0.5 + np.mean(col_transitions >= 6) * 0.5)
    panel_score = min(1.0, frame_score * 0.55 + min(1.0, strip_regions / 4) * 0.45)
    return {
        "x_occupied_region_count": len(x_regions),
        "y_occupied_region_count": len(y_regions),
        "strip_region_count": strip_regions,
        "strip_confidence": _round(strip_confidence),
        "frame_score": _round(frame_score),
        "panel_score": _round(panel_score),
        "text_like_score": _round(text_like),
    }


def _projection_regions(values: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*values.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append((start, index))
            start = None
    return regions


def _region_regularity(regions: Sequence[tuple[int, int]]) -> float:
    if len(regions) < 2:
        return 0.0
    widths = np.asarray([end - start for start, end in regions], dtype=np.float64)
    gaps = np.asarray([regions[i + 1][0] - regions[i][1] for i in range(len(regions) - 1)], dtype=np.float64)
    width_score = 1.0 - min(1.0, float(widths.std() / max(1.0, widths.mean())))
    gap_score = 1.0 - min(1.0, float(gaps.std() / max(1.0, gaps.mean()))) if gaps.size else 0.0
    return (width_score + gap_score) / 2.0


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = padded[1:-1, 1:-1]
    surrounded = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return interior & ~surrounded


def _duplicate_fingerprints(rgba: np.ndarray, foreground: np.ndarray) -> dict[str, Any]:
    alpha = foreground.astype(np.uint8) * 255
    ys, xs = np.nonzero(foreground)
    if xs.size:
        crop_alpha = alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        crop_rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    else:
        crop_alpha, crop_rgba = alpha, rgba
    return {
        "fingerprints": {
            "rgba": _array_hash(rgba),
            "alpha_mask": _array_hash(alpha),
            "canonical_alpha": _array_hash(crop_alpha),
            "canonical_rgba": _array_hash(crop_rgba),
            "canonical_alpha_flip_x": _array_hash(np.fliplr(crop_alpha)),
            "canonical_alpha_flip_y": _array_hash(np.flipud(crop_alpha)),
            "canonical_alpha_flip_xy": _array_hash(np.flipud(np.fliplr(crop_alpha))),
            "perceptual": _difference_hash(rgba),
        }
    }


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _difference_hash(rgba: np.ndarray) -> str:
    image = Image.fromarray(rgba, "RGBA")
    matte = Image.new("RGBA", image.size, (0, 0, 0, 255))
    matte.alpha_composite(image)
    gray = np.asarray(matte.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (gray[:, :-1] > gray[:, 1:]).flatten()
    return f"{sum(int(bit) << (63 - i) for i, bit in enumerate(bits)):016x}"


def _build_duplicate_groups(results: Sequence[SuitabilityResult]) -> list[DuplicateGroup]:
    kinds = {
        "exact_rgba": "rgba",
        "exact_alpha_mask": "alpha_mask",
        "recolor_geometry": "canonical_alpha",
        "padded_or_translated": "canonical_rgba",
    }
    groups: list[DuplicateGroup] = []
    for kind, key in kinds.items():
        buckets: dict[str, list[SuitabilityResult]] = {}
        for result in results:
            fingerprint = dict(result.metrics.get("fingerprints") or {}).get(key)
            if fingerprint:
                buckets.setdefault(str(fingerprint), []).append(result)
        for fingerprint, members in sorted(buckets.items()):
            ids = tuple(sorted(result.sprite_id for result in members))
            if len(ids) < 2:
                continue
            if kind == "recolor_geometry" and len({result.image_hash for result in members}) == 1:
                continue
            if kind == "exact_alpha_mask" and len({result.image_hash for result in members}) == 1:
                continue
            group_id = f"{kind}__{hashlib.sha256((kind + ':' + fingerprint).encode()).hexdigest()[:16]}"
            groups.append(DuplicateGroup(group_id, kind, ids, "keep_group_together_in_dataset_split"))
    # Flip-like geometry groups use a canonical minimum of normal/x/y alpha hashes.
    flip_buckets: dict[str, list[SuitabilityResult]] = {}
    for result in results:
        fp = dict(result.metrics.get("fingerprints") or {})
        values = [
            fp.get("canonical_alpha"),
            fp.get("canonical_alpha_flip_x"),
            fp.get("canonical_alpha_flip_y"),
            fp.get("canonical_alpha_flip_xy"),
        ]
        fingerprint_values = [str(value) for value in values if value]
        if fingerprint_values:
            flip_buckets.setdefault(min(fingerprint_values), []).append(result)
    for fingerprint, members in sorted(flip_buckets.items()):
        ids = tuple(sorted(result.sprite_id for result in members))
        if (
            len(ids) >= 2
            and len({result.metrics.get("fingerprints", {}).get("canonical_alpha") for result in members}) > 1
        ):
            group_id = f"trivial_flip__{hashlib.sha256(fingerprint.encode()).hexdigest()[:16]}"
            groups.append(DuplicateGroup(group_id, "trivial_flip", ids, "keep_group_together_in_dataset_split"))
    # Conservative near-identical variants: same perceptual hash only.
    near: dict[str, list[SuitabilityResult]] = {}
    for result in results:
        fp = dict(result.metrics.get("fingerprints") or {}).get("perceptual")
        if fp:
            near.setdefault(str(fp), []).append(result)
    for fingerprint, members in sorted(near.items()):
        ids = tuple(sorted(result.sprite_id for result in members))
        if len(ids) >= 2 and len({result.image_hash for result in members}) > 1:
            group_id = f"near_identical__{hashlib.sha256(fingerprint.encode()).hexdigest()[:16]}"
            groups.append(DuplicateGroup(group_id, "near_identical", ids, "review_and_keep_group_together"))
    unique = {(group.kind, group.member_sprite_ids): group for group in groups}
    return sorted(unique.values(), key=lambda group: (group.kind, group.member_sprite_ids, group.group_id))


def _render_markdown(output: AuditOutput) -> str:
    summary = output.summary
    status = summary["status_counts"]
    lines = [
        "# Sprite Suitability Audit",
        "",
        f"- Schema: `{summary['schema_version']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Config hash: `{summary['config_hash']}`",
        f"- Total: {summary['total']}",
        f"- Accept: {status['accept']}",
        f"- Quarantine: {status['quarantine']}",
        f"- Reject: {status['reject']}",
        "",
        "## Reason codes",
        "",
        "| Code | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {code} | {count} |" for code, count in summary["reason_code_counts"].items())
    if not summary["reason_code_counts"]:
        lines.append("| None | 0 |")
    lines.extend(["", "## Duplicate groups", "", "| Kind | Group | Members | Recommendation |", "|---|---|---:|---|"])
    for group in output.duplicate_groups:
        lines.append(f"| {group.kind} | `{group.group_id}` | {len(group.member_sprite_ids)} | {group.recommendation} |")
    if not output.duplicate_groups:
        lines.append("| None |  | 0 |  |")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical_json(value) + "\n" for value in values), encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _round(value: float) -> float:
    return round(float(value), 6)
