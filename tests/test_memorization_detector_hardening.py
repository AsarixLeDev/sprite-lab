from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest

from spritelab.evaluation.memorization import (
    COMPARISON_PARAMETERS,
    COMPARISON_PARAMETERS_SHA256,
    DETECTOR_POLICY_VERSION,
    TrainingImage,
    detector_policy_record,
    image_diagnostics,
    retrieve_neighbors,
)
from spritelab.evaluation.suite import DEFAULT_GATES, evaluate_gates


def _blank(*, transparent_noise: bool = False) -> np.ndarray:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    if transparent_noise:
        y, x = np.indices((32, 32))
        rgba[..., 0] = (x * 17 + y * 3) % 256
        rgba[..., 1] = (x * 5 + y * 11) % 256
        rgba[..., 2] = (x * 13 + y * 7) % 256
    return rgba


def _mask_sprite(mask: np.ndarray, color: tuple[int, int, int] = (210, 70, 35)) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, :3] = color
    rgba[mask, 3] = 255
    return rgba


def _rect(
    y0: int = 8,
    y1: int = 22,
    x0: int = 10,
    x1: int = 20,
    color: tuple[int, int, int] = (210, 70, 35),
) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return _mask_sprite(mask, color)


def _neighbor(generated: np.ndarray, training: np.ndarray) -> dict[str, Any]:
    target = TrainingImage("train", "dataset", "train.npz", 0, training)
    return retrieve_neighbors(generated, [target], top_k=1)[0]


def _case_images() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    one_a = np.zeros((32, 32), dtype=bool)
    one_a[15, 15] = True
    one_b = np.zeros((32, 32), dtype=bool)
    one_b[16, 17] = True
    two = np.zeros((32, 32), dtype=bool)
    two[15:17, 15:17] = True

    sword = np.zeros((32, 32), dtype=bool)
    sword[9:22, 15] = True
    sword[19, 12:19] = True
    sword[22:25, 15] = True
    gem = np.zeros((32, 32), dtype=bool)
    gem[14, 15:17] = True
    gem[15:17, 14:18] = True
    gem[17, 15:17] = True

    exact = _rect()
    recolor = _rect(color=(20, 180, 230))
    translated = _rect(x0=13, x1=23)
    one_pixel_edit = exact.copy()
    one_pixel_edit[8, 10] = 0
    large_rgb = _rect(color=(0, 255, 255))

    vertical = _rect(y0=6, y1=26, x0=13, x1=17)
    horizontal = _rect(y0=13, y1=17, x0=6, x1=26)
    small_a = _rect(y0=10, y1=15, x0=10, x1=14)
    small_b = _rect(y0=10, y1=14, x0=10, x1=15)

    material_a = np.zeros((32, 32), dtype=bool)
    material_a[6:12, 5:11] = True
    material_a[19:25, 20:26] = True
    material_b = np.zeros((32, 32), dtype=bool)
    material_b[5:11, 20:26] = True
    material_b[20:26, 5:11] = True

    return {
        "blank versus blank": (_blank(), _blank()),
        "blank versus transparent RGB noise": (_blank(), _blank(transparent_noise=True)),
        "blank versus 1-pixel foreground": (_blank(), _mask_sprite(one_a)),
        "blank versus 2x2 foreground": (_blank(), _mask_sprite(two)),
        "near-blank versus near-blank with different placement": (_mask_sprite(one_a), _mask_sprite(one_b)),
        "nontrivial exact RGBA": (exact, exact.copy()),
        "nontrivial exact alpha with different RGB": (exact, recolor),
        "same silhouette with palette remap": (exact, recolor),
        "translated nontrivial alpha": (exact, translated),
        "simple sword-like generic silhouette": (_mask_sprite(sword), _mask_sprite(sword, (40, 130, 220))),
        "tiny symmetric gem-like silhouette": (_mask_sprite(gem), _mask_sprite(gem)),
        "one-pixel edit on a nontrivial sprite": (one_pixel_edit, exact),
        "large RGB difference with identical alpha": (exact, large_rgb),
        "small full-canvas distance caused mostly by transparency": (small_a, small_b),
        "two materially different sprites with similar occupancy": (
            _mask_sprite(material_a),
            _mask_sprite(material_b, (30, 180, 90)),
        ),
        "different equal-occupancy orthogonal sprites": (vertical, horizontal),
    }


EXPECTED_CLASSES = {
    "blank versus blank": "exact_rgba_low_evidence_collision",
    "blank versus transparent RGB noise": "blank_collision",
    "blank versus 1-pixel foreground": "no_material_match",
    "blank versus 2x2 foreground": "no_material_match",
    "near-blank versus near-blank with different placement": "generic_sparse_collision",
    "nontrivial exact RGBA": "exact_rgba_nontrivial",
    "nontrivial exact alpha with different RGB": "exact_alpha_review_required",
    "same silhouette with palette remap": "exact_alpha_review_required",
    "translated nontrivial alpha": "translation_alpha_review_required",
    "simple sword-like generic silhouette": "generic_sparse_collision",
    "tiny symmetric gem-like silhouette": "exact_rgba_low_evidence_collision",
    "one-pixel edit on a nontrivial sprite": "near_pixel_review_required",
    "large RGB difference with identical alpha": "exact_alpha_review_required",
    "small full-canvas distance caused mostly by transparency": "no_material_match",
    "two materially different sprites with similar occupancy": "no_material_match",
    "different equal-occupancy orthogonal sprites": "no_material_match",
}


@pytest.mark.parametrize("name", EXPECTED_CLASSES)
def test_synthetic_regression_matrix(name: str) -> None:
    generated, training = _case_images()[name]
    row = _neighbor(generated, training)
    assert row["evidence_class"] == EXPECTED_CLASSES[name]
    assert row["detector_policy_version"] == DETECTOR_POLICY_VERSION
    assert row["comparison_parameters_sha256"] == COMPARISON_PARAMETERS_SHA256
    assert row["generated_diagnostics"] == image_diagnostics(generated)
    assert row["training_diagnostics"] == image_diagnostics(training)


def test_required_strength_and_action_semantics() -> None:
    cases = _case_images()
    hard = _neighbor(*cases["nontrivial exact RGBA"])
    review = _neighbor(*cases["nontrivial exact alpha with different RGB"])
    low = _neighbor(*cases["blank versus blank"])
    no_match = _neighbor(*cases["blank versus 2x2 foreground"])

    assert hard["machine_hard_block_candidate"] is True
    assert hard["requires_human_review"] is False
    assert review["requires_human_review"] is True
    assert review["machine_hard_block_candidate"] is False
    assert review["rgb_values_differ"] is True
    assert low["warning_only"] is True
    assert low["low_evidence_reason"] == "blank_alpha"
    assert no_match["suspicious"] is False
    assert no_match["union_rgba_distance"] is not None
    assert no_match["compared_foreground_pixel_count"] == 4


def test_near_pixel_requires_occupancy_aware_foreground_evidence() -> None:
    cases = _case_images()
    blank_two = _neighbor(*cases["blank versus 2x2 foreground"])
    transparent_dominated = _neighbor(*cases["small full-canvas distance caused mostly by transparency"])
    edited = _neighbor(*cases["one-pixel edit on a nontrivial sprite"])

    assert (
        blank_two["pixel_distance"]
        <= COMPARISON_PARAMETERS["thresholds"]["legacy_diagnostic_thresholds"]["full_canvas_pixel_distance"]
    )
    assert blank_two["evidence_class"] == "no_material_match"
    assert transparent_dominated["pixel_distance"] <= 0.025
    assert transparent_dominated["union_rgba_distance"] > 0.025
    assert transparent_dominated["evidence_class"] == "no_material_match"
    assert edited["union_rgba_distance"] <= 0.025
    assert edited["alpha_iou"] >= 0.9
    assert edited["evidence_class"] == "near_pixel_review_required"


def test_diagnostics_are_complete_and_deterministic() -> None:
    image = _rect()
    first = image_diagnostics(image)
    second = image_diagnostics(image.copy())
    required = {
        "width",
        "height",
        "foreground_pixel_count",
        "foreground_occupancy",
        "alpha_bbox",
        "alpha_bbox_width",
        "alpha_bbox_height",
        "connected_component_count",
        "blank_alpha",
        "near_blank_alpha",
        "unique_visible_rgba_count",
        "alpha_centroid",
        "horizontal_symmetry",
        "vertical_symmetry",
    }
    assert required <= first.keys()
    assert first == second
    assert first["foreground_pixel_count"] == 140
    assert first["alpha_bbox"] == [10, 8, 20, 22]
    assert first["connected_component_count"] == 1


def test_policy_hash_binds_all_serialized_thresholds() -> None:
    canonical = json.dumps(COMPARISON_PARAMETERS, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == COMPARISON_PARAMETERS_SHA256
    policy = detector_policy_record()
    thresholds = policy["thresholds"]
    assert thresholds["minimum_foreground_pixels"] == 16
    assert thresholds["minimum_foreground_occupancy"] == 0.015625
    assert thresholds["near_blank_threshold"]["maximum_foreground_pixels"] == 4
    assert "generic_sparse_collision_thresholds" in thresholds


def _summary(*, hard: int = 0, review: int = 0, low: int = 0, version: str = DETECTOR_POLICY_VERSION) -> dict:
    no_match = int(hard + review + low == 0)
    class_counts = {
        key: value
        for key, value in {
            "exact_rgba_nontrivial": hard,
            "exact_alpha_review_required": review,
            "generic_sparse_collision": low,
            "no_material_match": no_match,
        }.items()
        if value
    }
    return {
        "sample_count": max(1, hard + review + low),
        "hard_validity": {"malformed_count": 0},
        "pixel_art": {"semi_transparent_ratio_mean": 0.0, "palette_size_mean": 8.0},
        "diversity": {"exact_duplicate_rate": 0.0, "repeated_template_rate": 0.0},
        "conditional": {"represented_rate": 1.0},
        "memorization": {
            "detector_policy_version": version,
            "comparison_method": "deterministic_rgba_alpha_occupancy_v2",
            "comparison_parameters": COMPARISON_PARAMETERS,
            "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
            "hard_evidence_count": hard,
            "review_required_count": review,
            "warning_count": low,
            "low_evidence_collision_count": low,
            "unresolved_candidate_count": hard + review,
            "evidence_class_counts": class_counts,
        },
    }


def test_machine_gates_distinguish_hard_review_and_low_evidence() -> None:
    low = evaluate_gates(_summary(low=1), DEFAULT_GATES)
    review = evaluate_gates(_summary(review=1), DEFAULT_GATES)
    hard = evaluate_gates(_summary(hard=1), DEFAULT_GATES)

    assert low["pass"] is True
    assert low["manual_review_required"] is False
    assert low["low_evidence_collision_count"] == 1
    assert review["pass"] is False
    assert review["manual_review_required"] is True
    assert review["checks"]["memorization_reviews_resolved"] is False
    assert hard["pass"] is False
    assert hard["manual_review_required"] is False
    assert hard["checks"]["memorization_hard_evidence"] is False


def test_unsupported_policy_versions_fail_closed() -> None:
    image = _rect()
    target = TrainingImage("train", "dataset", "train.npz", 0, image)
    with pytest.raises(ValueError, match="unsupported detector policy"):
        retrieve_neighbors(image, [target], detector_policy_version="memorization_detector_v999")

    gate = evaluate_gates(_summary(version="memorization_detector_v999"), DEFAULT_GATES)
    assert gate["pass"] is False
    assert gate["checks"]["detector_policy_supported"] is False
