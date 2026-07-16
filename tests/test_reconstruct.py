from __future__ import annotations

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.reconstruct import reconstruct_rgba


def make_reconstruct_bundle() -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)

    alpha[1, 1] = 1
    index_map[1, 1] = 2

    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [100, 150, 200],
        ],
        dtype=np.uint8,
    )

    return SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id="reconstruct"),
    )


def test_reconstruct_outputs_rgba_image() -> None:
    image = reconstruct_rgba(make_reconstruct_bundle())

    assert image.mode == "RGBA"
    assert image.size == (32, 32)


def test_reconstruct_transparent_pixels_become_alpha_zero() -> None:
    image = reconstruct_rgba(make_reconstruct_bundle())

    assert image.getpixel((0, 0)) == (0, 0, 0, 0)


def test_reconstruct_opaque_pixels_use_palette_rgb() -> None:
    image = reconstruct_rgba(make_reconstruct_bundle())

    assert image.getpixel((1, 1)) == (100, 150, 200, 255)
