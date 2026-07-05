from __future__ import annotations

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata


def make_role_demo_bundle() -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)

    alpha[8:24, 8:24] = 1
    index_map[8:24, 8:24] = 2

    index_map[8, 8:24] = 1
    index_map[23, 8:24] = 1
    index_map[8:24, 8] = 1
    index_map[8:24, 23] = 1

    index_map[9:15, 9:15] = 3
    index_map[11:13, 18:20] = 4
    index_map[17, 12] = 5
    index_map[19, 20] = 5

    palette = np.array(
        [
            [0, 0, 0],
            [12, 10, 18],
            [116, 72, 152],
            [70, 42, 105],
            [250, 220, 70],
            [230, 170, 55],
        ],
        dtype=np.uint8,
    )
    return SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id="role_demo", category="item_icon", palette_size=5),
    )


def make_single_color_bundle(color: tuple[int, int, int] = (120, 80, 160)) -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[10:22, 10:22] = 1
    index_map[10:22, 10:22] = 1
    return SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], list(color)], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id="single", palette_size=1),
    )
