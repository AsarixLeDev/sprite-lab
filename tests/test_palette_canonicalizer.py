from __future__ import annotations

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.canonical_palette import (
    canonicalize_bundle_palette,
    compute_palette_slot_stats,
)
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.codec.validate import assert_valid_bundle


def make_scrambled_square_bundle(include_unused: bool = False) -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)

    alpha[8:24, 8:24] = 1
    index_map[8:24, 8:24] = 3
    index_map[8, 8:24] = 2
    index_map[23, 8:24] = 2
    index_map[8:24, 8] = 2
    index_map[8:24, 23] = 2
    index_map[13:19, 13:19] = 1

    palette_rows = [
        [0, 0, 0],  # slot 0 dummy transparent
        [255, 238, 64],  # slot 1 bright highlight
        [12, 10, 18],  # slot 2 dark outline
        [116, 72, 152],  # slot 3 midtone
    ]
    if include_unused:
        palette_rows.append([0, 255, 128])

    return SpriteBundle(
        alpha=alpha,
        palette=np.array(palette_rows, dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id="scrambled_square", palette_size=len(palette_rows) - 1),
    )


def test_slot_zero_remains_fixed() -> None:
    bundle = make_scrambled_square_bundle()

    result = canonicalize_bundle_palette(bundle)

    np.testing.assert_array_equal(result.bundle.palette[0], bundle.palette[0])
    assert result.old_to_new[0] == 0
    assert result.new_to_old[0] == 0
    assert np.all(result.bundle.index_map[result.bundle.alpha == 0] == 0)


def test_dark_edge_color_moves_before_highlight() -> None:
    bundle = make_scrambled_square_bundle()

    result = canonicalize_bundle_palette(bundle)

    assert result.old_to_new[2] == 1
    assert result.old_to_new[1] > result.old_to_new[3]


def test_reconstructed_image_is_identical_after_canonicalization() -> None:
    bundle = make_scrambled_square_bundle()

    before = reconstruct_rgba(bundle)
    result = canonicalize_bundle_palette(bundle)
    after = reconstruct_rgba(result.bundle)

    np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_index_map_remapping_is_correct() -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[10:14, 10:14] = 1
    index_map[10:14, 10:14] = 1
    index_map[10, 10:14] = 2
    index_map[13, 10:14] = 2
    index_map[10:14, 10] = 2
    index_map[10:14, 13] = 2
    index_map[11:13, 11:13] = 3

    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array(
            [
                [0, 0, 0],
                [255, 0, 0],
                [0, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        ),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id="remap"),
    )

    result = canonicalize_bundle_palette(bundle)

    assert result.old_to_new[2] == 1
    assert result.bundle.index_map[10, 10] == 1
    assert tuple(int(value) for value in result.bundle.palette[1]) == (0, 0, 0)


def test_unused_slots_move_to_the_end() -> None:
    bundle = make_scrambled_square_bundle(include_unused=True)

    result = canonicalize_bundle_palette(bundle)

    assert 4 in result.old_to_new
    assert max(result.old_to_new.values()) == result.old_to_new[4]
    assert result.new_to_old[result.old_to_new[4]] == 4
    assert_valid_bundle(result.bundle)


def test_stats_are_sane() -> None:
    bundle = make_scrambled_square_bundle()

    stats = {stat.slot: stat for stat in compute_palette_slot_stats(bundle)}
    outline = stats[2]
    center = stats[3]

    assert outline.edge_contact_ratio > center.edge_contact_ratio
    assert outline.count > 0
    assert all(0.0 <= stat.luminance <= 1.0 for stat in stats.values())
    assert all(0.0 <= stat.frequency <= 1.0 for stat in stats.values())
    assert all(stat.role_hint for stat in stats.values())
