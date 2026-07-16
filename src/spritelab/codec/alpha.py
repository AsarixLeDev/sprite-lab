"""Hard alpha extraction for clean 32x32 RGBA sprites."""

from __future__ import annotations

import numpy as np
from PIL import Image

from spritelab.utils.image import assert_exact_size, ensure_rgba


def extract_hard_alpha(image: Image.Image, threshold: int = 128) -> np.ndarray:
    """Convert an image alpha channel into a 32x32 binary uint8 mask.

    Pixels with alpha greater than or equal to ``threshold`` become opaque
    ``1``. Pixels below the threshold become transparent ``0``.
    """

    _validate_alpha_threshold(threshold)
    assert_exact_size(image)

    rgba = ensure_rgba(image)
    pixels = np.asarray(rgba, dtype=np.uint8)
    return (pixels[:, :, 3] >= threshold).astype(np.uint8)


def _validate_alpha_threshold(threshold: int) -> None:
    if threshold < 0 or threshold > 255:
        raise ValueError(f"alpha threshold must be in 0..255, got {threshold}.")
