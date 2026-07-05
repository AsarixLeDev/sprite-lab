from __future__ import annotations

import numpy as np

from spritelab.codec.role_inference import infer_palette_slot_roles_v2
from spritelab.codec.roles import (
    ROLE_ACCENT,
    ROLE_EMISSIVE,
    ROLE_HIGHLIGHT,
    ROLE_LIGHT,
    ROLE_MIDTONE,
    ROLE_OUTLINE,
    ROLE_SHADOW,
    ROLE_TEXTURE_DETAIL,
    ROLE_TRANSPARENT,
)

from test_role_helpers import make_role_demo_bundle, make_single_color_bundle


def test_obvious_dark_edge_color_becomes_outline() -> None:
    bundle = make_role_demo_bundle()

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)

    assert 0 not in result.slot_roles
    assert result.slot_roles[1] == ROLE_OUTLINE
    assert np.all(result.role_map[bundle.alpha == 0] == ROLE_TRANSPARENT)


def test_main_fill_is_not_classified_as_accent() -> None:
    bundle = make_role_demo_bundle()

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)

    assert result.slot_roles[2] in {ROLE_MIDTONE, ROLE_SHADOW, ROLE_LIGHT}
    assert result.slot_roles[2] != ROLE_ACCENT


def test_bright_and_saturated_rare_colors_get_special_roles() -> None:
    bundle = make_role_demo_bundle()

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)

    assert result.slot_roles[4] in {ROLE_HIGHLIGHT, ROLE_ACCENT, ROLE_EMISSIVE}
    assert result.slot_roles[5] in {ROLE_HIGHLIGHT, ROLE_ACCENT, ROLE_EMISSIVE, ROLE_TEXTURE_DETAIL}


def test_all_visible_slots_receive_roles_confidence_and_scores() -> None:
    bundle = make_role_demo_bundle()

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)

    assert set(result.slot_roles) == {1, 2, 3, 4, 5}
    assert set(result.confidence) == {1, 2, 3, 4, 5}
    assert set(result.debug_scores) == {1, 2, 3, 4, 5}
    assert all(0.0 <= confidence <= 1.0 for confidence in result.confidence.values())
    assert all(result.debug_scores[slot] for slot in result.debug_scores)


def test_tiny_palette_cases_do_not_crash() -> None:
    bundle = make_single_color_bundle()

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)

    assert result.slot_roles[1] in {ROLE_MIDTONE, ROLE_OUTLINE}
    assert result.role_map.shape == (32, 32)
