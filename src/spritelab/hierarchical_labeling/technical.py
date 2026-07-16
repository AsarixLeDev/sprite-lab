"""Consolidated deterministic visual evidence extraction."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.harvest.label_v4.pixel_evidence import (
    PIXEL_EVIDENCE_VERSION,
    alpha_mask_content_hash,
    analyze_pixels,
    exact_rgba_content_hash,
)
from spritelab.hierarchical_labeling.contracts import (
    EvidenceStrength,
    FeatureValue,
    TechnicalVisualEvidence,
    ValidityState,
)
from spritelab.hierarchical_labeling.json_utils import content_identity, require_text

TECHNICAL_EVIDENCE_VERSION = "spritelab-deterministic-visual-evidence-v1"


def _flatten(image: Image.Image) -> list[Any]:
    modern = getattr(image, "get_flattened_data", None)
    return list(modern() if modern is not None else image.getdata())


def _component_areas(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
    for y, x in np.argwhere(mask):
        y_value, x_value = int(y), int(x)
        if visited[y_value, x_value]:
            continue
        queue: deque[tuple[int, int]] = deque([(y_value, x_value)])
        visited[y_value, x_value] = True
        area = 0
        while queue:
            current_y, current_x = queue.popleft()
            area += 1
            for next_y, next_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and bool(mask[next_y, next_x])
                    and not bool(visited[next_y, next_x])
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        areas.append(area)
    return sorted(areas, reverse=True)


def _symmetry(mask: np.ndarray) -> dict[str, float | None]:
    if not mask.any():
        return {"horizontal": None, "vertical": None}
    vertical = float(np.mean(mask == np.fliplr(mask)))
    horizontal = float(np.mean(mask == np.flipud(mask)))
    return {"horizontal": round(horizontal, 6), "vertical": round(vertical, 6)}


def _edge_density(mask: np.ndarray) -> float:
    if mask.size <= 1:
        return 0.0
    vertical = np.count_nonzero(mask[1:, :] != mask[:-1, :])
    horizontal = np.count_nonzero(mask[:, 1:] != mask[:, :-1])
    possible = mask[1:, :].size + mask[:, 1:].size
    return round(float((vertical + horizontal) / max(1, possible)), 6)


def _detail_density(rgba: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    rgb = rgba[:, :, :3].astype(np.int16)
    horizontal_visible = mask[:, 1:] & mask[:, :-1]
    vertical_visible = mask[1:, :] & mask[:-1, :]
    horizontal_changes = np.any(rgb[:, 1:] != rgb[:, :-1], axis=2) & horizontal_visible
    vertical_changes = np.any(rgb[1:, :] != rgb[:-1, :], axis=2) & vertical_visible
    possible = np.count_nonzero(horizontal_visible) + np.count_nonzero(vertical_visible)
    changes = np.count_nonzero(horizontal_changes) + np.count_nonzero(vertical_changes)
    return round(float(changes / max(1, possible)), 6)


def _entropy(counts: Mapping[Any, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    result = -sum((count / total) * math.log2(count / total) for count in counts.values() if count)
    return round(float(result), 6)


def _sheet_grid_status(width: int, height: int, component_areas: list[int], frame_count: int) -> dict[str, Any]:
    repeated_components = False
    if len(component_areas) >= 4:
        median = float(np.median(np.asarray(component_areas, dtype=float)))
        repeated_components = median > 0 and sum(abs(area - median) / median <= 0.2 for area in component_areas) >= 4
    likely_sheet = frame_count == 1 and width >= 64 and height >= 32 and repeated_components
    return {
        "state": "likely_sheet_or_grid" if likely_sheet else "not_detected",
        "repeated_component_sizes": repeated_components,
        "dimension_multiple_8": width % 8 == 0 and height % 8 == 0,
    }


def _feature(
    name: str,
    value: Any,
    image_identity: str,
    *,
    strength: EvidenceStrength,
    validity: ValidityState | None = None,
    confidence: float | None = None,
    method: str = "spritelab_consolidated_pixel_analysis",
) -> FeatureValue:
    return FeatureValue(
        name=name,
        value=value,
        method=method,
        method_version=TECHNICAL_EVIDENCE_VERSION,
        validity=validity
        or (ValidityState.VALID if strength == EvidenceStrength.STRONG_DETERMINISTIC else ValidityState.HEURISTIC),
        strength=strength,
        source_image_identity=image_identity,
        confidence=confidence,
    )


def extract_technical_evidence(
    image_path: str | Path,
    *,
    record_identity: str,
    duplicate_cluster_identity: str | None = None,
    near_duplicate_cluster_identity: str | None = None,
) -> TechnicalVisualEvidence:
    """Extract exact facts and explicitly marked heuristics from one image.

    Exact RGBA/alpha/palette facts reuse Label v4's canonical identity and
    pixel-analysis helpers. Duplicate identity is that same exact RGBA content
    identity, avoiding a second incompatible duplicate system.
    """

    path = Path(image_path)
    require_text(record_identity, "record identity")
    pixels = analyze_pixels(path)
    image_identity = exact_rgba_content_hash(path)
    with Image.open(path) as opened:
        frame_count = int(getattr(opened, "n_frames", 1))
        opened.seek(0)
        rgba_image = opened.convert("RGBA")
        rgba = np.asarray(rgba_image, dtype=np.uint8)
        palette_counts = Counter(tuple(int(value) for value in pixel) for pixel in _flatten(rgba_image))
    alpha = rgba[:, :, 3]
    visible_mask = alpha > 0
    opaque_mask = alpha >= 128
    component_areas = _component_areas(opaque_mask)
    bbox = list(pixels["bbox"])
    bbox_area = max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    object_minimum = max(2, int(0.01 * max(1, opaque_mask.size)))
    object_count = sum(area >= object_minimum for area in component_areas)
    exact_duplicate = duplicate_cluster_identity or f"rgba:{image_identity}"
    dominant_colors = [
        {"rgba": list(row["rgba"]), "count": int(row["count"]), "hex": str(row["hex"])}
        for row in pixels["palette_rgba"][:8]
    ]
    strong = EvidenceStrength.STRONG_DETERMINISTIC
    heuristic = EvidenceStrength.HEURISTIC_TECHNICAL
    features = (
        _feature("image_width", int(pixels["width"]), image_identity, strength=strong, confidence=1.0),
        _feature("image_height", int(pixels["height"]), image_identity, strength=strong, confidence=1.0),
        _feature("frame_count", frame_count, image_identity, strength=strong, confidence=1.0),
        _feature(
            "alpha_coverage",
            round(float(np.count_nonzero(visible_mask) / max(1, visible_mask.size)), 6),
            image_identity,
            strength=strong,
            confidence=1.0,
        ),
        _feature("opaque_bounding_box", bbox, image_identity, strength=strong, confidence=1.0),
        _feature(
            "opaque_area_ratio",
            round(float(np.count_nonzero(opaque_mask) / max(1, bbox_area)), 6),
            image_identity,
            strength=strong,
            confidence=1.0,
        ),
        _feature("connected_component_count", len(component_areas), image_identity, strength=strong, confidence=1.0),
        _feature("dominant_colors", dominant_colors, image_identity, strength=strong, confidence=1.0),
        _feature("palette_size", len(palette_counts), image_identity, strength=strong, confidence=1.0),
        _feature("color_entropy", _entropy(palette_counts), image_identity, strength=strong, confidence=1.0),
        _feature(
            "alpha_silhouette_hash", alpha_mask_content_hash(path), image_identity, strength=strong, confidence=1.0
        ),
        _feature("symmetry_estimates", _symmetry(opaque_mask), image_identity, strength=heuristic, confidence=0.55),
        _feature(
            "orientation_estimate",
            list(pixels.get("shape", {}).get("orientation", ())),
            image_identity,
            strength=heuristic,
            confidence=0.5,
        ),
        _feature("edge_density", _edge_density(opaque_mask), image_identity, strength=heuristic, confidence=0.6),
        _feature(
            "detail_density", _detail_density(rgba, opaque_mask), image_identity, strength=heuristic, confidence=0.55
        ),
        _feature("object_count_estimate", object_count, image_identity, strength=heuristic, confidence=0.45),
        _feature("empty_blank_status", not bool(visible_mask.any()), image_identity, strength=strong, confidence=1.0),
        _feature(
            "sheet_grid_status",
            _sheet_grid_status(rgba.shape[1], rgba.shape[0], component_areas, frame_count),
            image_identity,
            strength=heuristic,
            confidence=0.4,
        ),
        _feature("animation_status", frame_count > 1, image_identity, strength=strong, confidence=1.0),
        _feature("duplicate_cluster_identity", exact_duplicate, image_identity, strength=strong, confidence=1.0),
        _feature(
            "near_duplicate_cluster_identity",
            near_duplicate_cluster_identity,
            image_identity,
            strength=heuristic,
            validity=ValidityState.HEURISTIC if near_duplicate_cluster_identity else ValidityState.NOT_APPLICABLE,
            confidence=0.7 if near_duplicate_cluster_identity else None,
        ),
    )
    extraction_identity = content_identity(
        TECHNICAL_EVIDENCE_VERSION,
        {
            "image_identity": image_identity,
            "pixel_evidence_version": PIXEL_EVIDENCE_VERSION,
            "duplicate_cluster_identity": exact_duplicate,
            "near_duplicate_cluster_identity": near_duplicate_cluster_identity,
        },
    )
    return TechnicalVisualEvidence(record_identity, image_identity, features, extraction_identity)


def technical_supervision(evidence: TechnicalVisualEvidence) -> dict[str, Any]:
    """Export only strong deterministic facts; heuristics remain evidence."""

    return {
        feature.name: feature.value
        for feature in evidence.features
        if feature.strength == EvidenceStrength.STRONG_DETERMINISTIC and feature.validity == ValidityState.VALID
    }
