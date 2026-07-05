from __future__ import annotations

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.role_inference import (
    build_role_map_from_slot_roles,
    infer_palette_slot_roles_v2,
    validate_role_map,
)
from spritelab.codec.roles import ROLE_MIDTONE, ROLE_OUTLINE, ROLE_SHADOW, ROLE_TRANSPARENT, ROLE_UNKNOWN


def test_build_role_map_from_slot_roles() -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[10, 10] = 1
    alpha[10, 11] = 1
    index_map[10, 10] = 1
    index_map[10, 11] = 2

    role_map = build_role_map_from_slot_roles(index_map, alpha, {1: ROLE_OUTLINE})

    assert role_map.shape == (32, 32)
    assert role_map.dtype == np.uint8
    assert role_map[0, 0] == ROLE_TRANSPARENT
    assert role_map[10, 10] == ROLE_OUTLINE
    assert role_map[10, 11] == ROLE_UNKNOWN


def test_validate_role_map_catches_shape_and_alpha_errors() -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    role_map = np.zeros((32, 32), dtype=np.uint8)
    role_map[0, 0] = ROLE_MIDTONE

    assert validate_role_map(role_map[:8, :8], alpha) == ["role_map shape must be exactly 32x32."]
    assert "transparent pixels should have ROLE_TRANSPARENT." in validate_role_map(role_map, alpha)

    alpha[1, 1] = 1
    role_map[1, 1] = ROLE_TRANSPARENT
    errors = validate_role_map(role_map, alpha)
    assert "opaque pixels should not have ROLE_TRANSPARENT." in errors


def test_outline_refinement_marks_dark_edge_pixels() -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[8:24, 8:24] = 1
    index_map[8:24, 8:24] = 2
    index_map[8, 8:24] = 1
    index_map[23, 8:24] = 1
    index_map[8:24, 8] = 1
    index_map[8:24, 23] = 1
    index_map[9, 9] = 3
    index_map[15, 15] = 3
    palette = np.array(
        [
            [0, 0, 0],
            [8, 8, 12],
            [130, 80, 160],
            [45, 38, 58],
        ],
        dtype=np.uint8,
    )
    bundle = SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id="refine", palette_size=3),
    )

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)

    assert result.role_map[9, 9] in {ROLE_OUTLINE, ROLE_SHADOW, ROLE_MIDTONE}
    assert result.role_map[15, 15] != ROLE_TRANSPARENT
