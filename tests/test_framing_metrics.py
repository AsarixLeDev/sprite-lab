from __future__ import annotations

import numpy as np
import pytest

from spritelab.training.framing_metrics import (
    compute_alpha_metrics,
    compute_color_metrics,
    compute_connected_components,
    compute_sprite_framing_metrics,
)


def _rgba(mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[mask, :3] = color
    rgba[mask, 3] = 255
    return rgba


def test_empty_sprite_alpha_coverage_is_zero() -> None:
    metrics = compute_alpha_metrics(np.zeros((32, 32), dtype=bool))
    assert metrics["opaque_pixels"] == 0
    assert metrics["alpha_coverage"] == 0.0


def test_full_sprite_alpha_coverage_is_one() -> None:
    metrics = compute_alpha_metrics(np.ones((32, 32), dtype=bool))
    assert metrics["opaque_pixels"] == 1024
    assert metrics["alpha_coverage"] == 1.0


def test_centered_square_does_not_touch_border() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 12:20] = True
    metrics = compute_alpha_metrics(mask)
    assert metrics["touches_border"] is False


def test_border_square_touches_border() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:4, 10:14] = True
    metrics = compute_alpha_metrics(mask)
    assert metrics["touches_border"] is True


def test_bounding_box_is_correct() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:9, 7:11] = True
    metrics = compute_alpha_metrics(mask)
    assert metrics["bounding_box"] == {"x_min": 7, "y_min": 4, "x_max": 10, "y_max": 8}
    assert metrics["bbox_width"] == 4
    assert metrics["bbox_height"] == 5


def test_center_of_mass_is_correct() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[15:17, 15:17] = True
    metrics = compute_alpha_metrics(mask)
    assert metrics["center_of_mass_x"] == pytest.approx(15.5)
    assert metrics["center_of_mass_y"] == pytest.approx(15.5)
    assert metrics["center_offset_from_image_center"] == pytest.approx(0.0)


def test_connected_component_count_is_correct() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[2:4, 2:4] = True
    mask[20:22, 20:22] = True
    metrics = compute_connected_components(mask)
    assert metrics["connected_components"] == 2


def test_fragmented_sprite_has_multiple_components() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    for y, x in [(2, 2), (2, 12), (12, 2), (12, 12)]:
        mask[y, x] = True
    metrics = compute_sprite_framing_metrics(_rgba(mask))
    assert metrics["connected_components"] == 4
    assert metrics["fragmentation_score"] > 0.0


def test_visible_color_count_is_correct() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[1, 1] = [255, 0, 0, 255]
    rgba[1, 2] = [0, 255, 0, 255]
    rgba[1, 3] = [0, 0, 255, 255]
    metrics = compute_color_metrics(rgba)
    assert metrics["visible_color_count"] == 3


def test_palette_entropy_and_dominant_ratio_are_sane() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[0, 0:3] = [255, 0, 0, 255]
    rgba[0, 3] = [0, 255, 0, 255]
    metrics = compute_color_metrics(rgba)
    assert metrics["visible_color_count"] == 2
    assert metrics["dominant_color_ratio"] == pytest.approx(0.75)
    assert 0.0 < metrics["palette_entropy"] < 1.0
