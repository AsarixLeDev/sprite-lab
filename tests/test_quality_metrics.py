from __future__ import annotations

import numpy as np
import pytest

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.data.quality_report import (
    EMPTY_SPRITE,
    HAS_ALPHA_HOLES,
    LOW_CONTRAST,
    MANY_COMPONENTS,
    MANY_SINGLE_PIXELS,
    OFF_CENTER,
    TOUCHES_EDGE,
    compute_sprite_quality_metrics,
)


def test_centered_square_metrics() -> None:
    bundle = _bundle_from_mask(_square_mask(12, 12, 8, 8), sprite_id="centered")

    metrics = compute_sprite_quality_metrics(bundle, bundle_dir="bundles/centered")

    assert metrics.opaque_pixel_count == 64
    assert metrics.opaque_pixel_ratio == pytest.approx(64 / 1024)
    assert metrics.bbox_x == 12
    assert metrics.bbox_y == 12
    assert metrics.bbox_width == 8
    assert metrics.bbox_height == 8
    assert metrics.bbox_area == 64
    assert metrics.bbox_fill_ratio == pytest.approx(1.0)
    assert metrics.center_x == pytest.approx(15.5)
    assert metrics.center_y == pytest.approx(15.5)
    assert metrics.center_distance == pytest.approx(0.0)
    assert metrics.connected_component_count == 1
    assert metrics.single_pixel_component_count == 0
    assert metrics.alpha_hole_count == 0


def test_empty_sprite_metrics() -> None:
    bundle = _bundle_from_mask(np.zeros((32, 32), dtype=np.uint8), sprite_id="empty")

    metrics = compute_sprite_quality_metrics(bundle, bundle_dir="bundles/empty")

    assert metrics.opaque_pixel_count == 0
    assert metrics.bbox_x is None
    assert metrics.center_x is None
    assert metrics.connected_component_count == 0
    assert metrics.contrast_score is None
    assert EMPTY_SPRITE in metrics.issue_codes


def test_single_pixel_off_center_touching_edge_metrics() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0, 31] = 1
    bundle = _bundle_from_mask(mask, sprite_id="single")

    metrics = compute_sprite_quality_metrics(bundle, bundle_dir="bundles/single")

    assert metrics.opaque_pixel_count == 1
    assert metrics.bbox_x == 31
    assert metrics.bbox_y == 0
    assert metrics.connected_component_count == 1
    assert metrics.single_pixel_component_count == 1
    assert metrics.edge_touching_opaque_count == 1
    assert TOUCHES_EDGE in metrics.issue_codes
    assert OFF_CENTER in metrics.issue_codes


def test_alpha_hole_metrics() -> None:
    mask = _square_mask(8, 8, 16, 16)
    mask[15:17, 15:17] = 0
    bundle = _bundle_from_mask(mask, sprite_id="hole")

    metrics = compute_sprite_quality_metrics(bundle, bundle_dir="bundles/hole")

    assert metrics.alpha_hole_count == 1
    assert HAS_ALPHA_HOLES in metrics.issue_codes


def test_low_contrast_palette_metrics() -> None:
    mask = _square_mask(12, 12, 8, 8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[mask == 1] = 1
    index_map[12:16, 12:16] = 2
    bundle = SpriteBundle(
        alpha=mask,
        palette=np.array([[0, 0, 0], [100, 100, 100], [105, 105, 105]], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id="low_contrast"),
    )

    metrics = compute_sprite_quality_metrics(bundle, bundle_dir="bundles/low_contrast")

    assert metrics.contrast_score is not None
    assert metrics.contrast_score < 0.12
    assert LOW_CONTRAST in metrics.issue_codes


def test_multi_component_single_pixel_metrics() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    for y, x in [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (14, 14), (16, 16)]:
        mask[y, x] = 1
    bundle = _bundle_from_mask(mask, sprite_id="speckles")

    metrics = compute_sprite_quality_metrics(bundle, bundle_dir="bundles/speckles")

    assert metrics.connected_component_count == 8
    assert metrics.single_pixel_component_count == 8
    assert metrics.single_pixel_component_ratio == pytest.approx(1.0)
    assert MANY_COMPONENTS in metrics.issue_codes
    assert MANY_SINGLE_PIXELS in metrics.issue_codes


def _square_mask(x: int, y: int, width: int, height: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[y : y + height, x : x + width] = 1
    return mask


def _bundle_from_mask(mask: np.ndarray, *, sprite_id: str) -> SpriteBundle:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[mask == 1] = 1
    return SpriteBundle(
        alpha=mask,
        palette=np.array([[0, 0, 0], [40, 40, 40]], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id),
    )
