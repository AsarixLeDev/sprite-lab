"""Build palette index maps from clean RGBA sprites."""

from __future__ import annotations

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SPRITE_SIZE
from spritelab.utils.image import assert_exact_size, ensure_rgba


def build_index_map_from_palette(
    image: Image.Image,
    alpha: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """Build a 32x32 index map from a clean image and exact palette.

    Transparent pixels become 0. Opaque pixels must match one of the visible
    palette rows, slots 1..K-1, or a ValueError is raised.
    """

    _assert_binary_alpha(alpha)
    _assert_palette_shape(palette)
    assert_exact_size(image)

    rgba = ensure_rgba(image)
    pixels = np.asarray(rgba, dtype=np.uint8)
    rgb_to_slot = _visible_palette_lookup(palette)
    index_map = np.zeros(SPRITE_SIZE, dtype=_index_dtype_for_palette(palette))

    for y in range(SPRITE_SIZE[0]):
        for x in range(SPRITE_SIZE[1]):
            if int(alpha[y, x]) == 0:
                continue

            rgb = tuple(int(channel) for channel in pixels[y, x, :3])
            slot = rgb_to_slot.get(rgb)
            if slot is None:
                raise ValueError(f"opaque pixel color {rgb} at x={x}, y={y} is not in palette.")
            index_map[y, x] = slot

    return index_map


def _visible_palette_lookup(palette: np.ndarray) -> dict[tuple[int, int, int], int]:
    lookup: dict[tuple[int, int, int], int] = {}
    for slot in range(1, int(palette.shape[0])):
        rgb = tuple(int(channel) for channel in palette[slot])
        if rgb in lookup:
            raise ValueError(f"palette contains duplicate visible RGB color {rgb}.")
        lookup[rgb] = slot
    return lookup


def _index_dtype_for_palette(palette: np.ndarray) -> np.dtype[np.integer]:
    if palette.shape[0] <= 256:
        return np.dtype(np.uint8)
    return np.dtype(np.int16)


def _assert_binary_alpha(alpha: np.ndarray) -> None:
    if not isinstance(alpha, np.ndarray):
        raise ValueError("alpha must be a numpy array.")

    if alpha.shape != SPRITE_SIZE:
        raise ValueError("alpha shape must be exactly 32x32.")

    if not np.all(np.isin(alpha, [0, 1])):
        raise ValueError("alpha values must be only 0 or 1.")


def _assert_palette_shape(palette: np.ndarray) -> None:
    if not isinstance(palette, np.ndarray):
        raise ValueError("palette must be a numpy array.")

    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("palette shape must be Kx3 RGB.")

    if palette.shape[0] < 2:
        raise ValueError("palette must contain dummy transparent row plus at least one visible color.")
