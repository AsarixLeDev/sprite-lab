"""Reconstruct RGBA images from sprite bundles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SPRITE_HEIGHT, SPRITE_WIDTH, SpriteBundle
from spritelab.codec.validate import assert_valid_bundle


def reconstruct_rgba(bundle: SpriteBundle) -> Image.Image:
    """Reconstruct a 32x32 Pillow RGBA image from a sprite bundle."""

    assert_valid_bundle(bundle)

    pixels = np.zeros((SPRITE_HEIGHT, SPRITE_WIDTH, 4), dtype=np.uint8)

    for y in range(SPRITE_HEIGHT):
        for x in range(SPRITE_WIDTH):
            if int(bundle.alpha[y, x]) == 0:
                pixels[y, x] = (0, 0, 0, 0)
            else:
                slot = int(bundle.index_map[y, x])
                red, green, blue = bundle.palette[slot]
                pixels[y, x] = (int(red), int(green), int(blue), 255)

    return Image.fromarray(pixels, mode="RGBA")


def save_reconstructed_png(bundle: SpriteBundle, path: str | Path) -> None:
    """Save the reconstructed 32x32 RGBA sprite as a PNG."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruct_rgba(bundle).save(output_path)
