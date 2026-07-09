"""Palette constants for sprite bundles."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from spritelab.codec.bundle import SPRITE_SIZE
from spritelab.utils.image import assert_exact_size, ensure_rgba

TRANSPARENT_INDEX: Final[int] = 0
DEFAULT_TRANSPARENT_RGB: Final[tuple[int, int, int]] = (0, 0, 0)
MIN_PALETTE_ROWS: Final[int] = 2


def visible_palette_size(palette: NDArray[np.generic]) -> int:
    """Return the count of usable visible colors, excluding the dummy slot."""

    if palette.ndim != 2:
        return 0
    return max(0, int(palette.shape[0]) - 1)


def extract_exact_palette(
    image: Image.Image,
    alpha: np.ndarray,
    *,
    max_visible_colors: int = 32,
    sort_colors: bool = True,
) -> np.ndarray:
    """Extract exact RGB colors from opaque pixels.

    The returned palette includes the dummy transparent row at slot 0. Visible
    colors occupy slots 1..K-1. This strict helper does not quantize; it raises
    when the visible color count exceeds ``max_visible_colors``.
    """

    if max_visible_colors < 1:
        raise ValueError("max_visible_colors must be at least 1.")

    _assert_binary_alpha(alpha)
    assert_exact_size(image)

    rgba = ensure_rgba(image)
    pixels = np.asarray(rgba, dtype=np.uint8)
    visible_colors = _collect_visible_colors(pixels[:, :, :3], alpha, sort_colors=sort_colors)

    if not visible_colors:
        raise ValueError("Image contains no opaque pixels; cannot create a valid SpriteBundle.")

    if len(visible_colors) > max_visible_colors:
        raise ValueError(
            "Image contains "
            f"{len(visible_colors)} visible colors, above max_visible_colors={max_visible_colors}. "
            "Quantization is not implemented in this strict encoder."
        )

    rows = [DEFAULT_TRANSPARENT_RGB, *visible_colors]
    return np.array(rows, dtype=np.uint8)


def _collect_visible_colors(
    rgb_pixels: np.ndarray,
    alpha: np.ndarray,
    *,
    sort_colors: bool,
) -> list[tuple[int, int, int]]:
    seen: set[tuple[int, int, int]] = set()
    colors: list[tuple[int, int, int]] = []

    for y in range(SPRITE_SIZE[0]):
        for x in range(SPRITE_SIZE[1]):
            if int(alpha[y, x]) == 0:
                continue

            color = tuple(int(channel) for channel in rgb_pixels[y, x])
            if color not in seen:
                seen.add(color)
                colors.append(color)

    if sort_colors:
        colors.sort()

    return colors


def _assert_binary_alpha(alpha: np.ndarray) -> None:
    if not isinstance(alpha, np.ndarray):
        raise ValueError("alpha must be a numpy array.")

    if alpha.shape != SPRITE_SIZE:
        raise ValueError("alpha shape must be exactly 32x32.")

    if not np.all(np.isin(alpha, [0, 1])):
        raise ValueError("alpha values must be only 0 or 1.")
