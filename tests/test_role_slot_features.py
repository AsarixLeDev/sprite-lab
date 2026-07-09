from __future__ import annotations

import pytest

from spritelab.codec.role_inference import compute_palette_slot_role_features
from test_role_helpers import make_role_demo_bundle


def test_slot_feature_extraction_counts_frequency_and_edges() -> None:
    bundle = make_role_demo_bundle()

    features = compute_palette_slot_role_features(bundle.palette, bundle.index_map, bundle.alpha)

    assert set(features) == {1, 2, 3, 4, 5}
    assert features[1].pixel_count == 60
    assert features[2].pixel_count == 154
    assert features[1].frequency == pytest.approx(60 / 256)
    assert features[1].edge_contact_ratio > 0.85
    assert features[2].edge_contact_ratio < 0.15


def test_slot_feature_spatial_and_local_values_are_sane() -> None:
    bundle = make_role_demo_bundle()

    features = compute_palette_slot_role_features(bundle.palette, bundle.index_map, bundle.alpha)
    features[1]
    fill = features[2]
    detail = features[5]

    assert fill.mean_x == pytest.approx(16.0, abs=1.0)
    assert fill.mean_y == pytest.approx(16.0, abs=1.0)
    assert fill.min_x == 9
    assert fill.max_x == 22
    assert fill.local_same_color_ratio > detail.local_same_color_ratio
    assert detail.local_same_color_ratio == pytest.approx(0.0)


def test_slot_feature_color_values_are_deterministic() -> None:
    bundle = make_role_demo_bundle()

    first = compute_palette_slot_role_features(bundle.palette, bundle.index_map, bundle.alpha)
    second = compute_palette_slot_role_features(bundle.palette, bundle.index_map, bundle.alpha)

    assert first[3].oklab_l == second[3].oklab_l
    assert first[3].chroma == second[3].chroma
    assert 0.0 <= first[3].hue_degrees < 360.0
    assert first[4].is_high_chroma
