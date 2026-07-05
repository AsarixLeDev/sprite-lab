from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritelab.codec.alpha import extract_hard_alpha


def test_extract_hard_alpha_threshold_behavior() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 0))
    image.putpixel((1, 0), (255, 0, 0, 127))
    image.putpixel((2, 0), (255, 0, 0, 128))
    image.putpixel((3, 0), (255, 0, 0, 255))

    alpha = extract_hard_alpha(image, threshold=128)

    assert alpha[0, 0] == 0
    assert alpha[0, 1] == 0
    assert alpha[0, 2] == 1
    assert alpha[0, 3] == 1
    assert alpha.shape == (32, 32)
    assert alpha.dtype == np.uint8


def test_extract_hard_alpha_converts_non_rgba_image() -> None:
    image = Image.new("RGB", (32, 32), (10, 20, 30))

    alpha = extract_hard_alpha(image)

    assert np.all(alpha == 1)


def test_extract_hard_alpha_rejects_wrong_size() -> None:
    image = Image.new("RGBA", (31, 32), (0, 0, 0, 0))

    with pytest.raises(ValueError, match="expected image size"):
        extract_hard_alpha(image)
