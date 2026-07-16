from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritelab.codec.alpha import extract_hard_alpha
from spritelab.codec.palette import extract_exact_palette


def test_extract_exact_palette_ignores_transparent_pixels_and_sorts() -> None:
    image = Image.new("RGBA", (32, 32), (200, 100, 50, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((1, 0), (0, 0, 255, 255))
    image.putpixel((2, 0), (0, 255, 0, 255))

    alpha = extract_hard_alpha(image)
    palette = extract_exact_palette(image, alpha)

    assert palette.dtype == np.uint8
    np.testing.assert_array_equal(palette[0], np.array([0, 0, 0], dtype=np.uint8))
    assert [tuple(int(channel) for channel in row) for row in palette[1:]] == [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
    ]


def test_extract_exact_palette_preserves_first_seen_order_when_requested() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((1, 0), (0, 0, 255, 255))

    alpha = extract_hard_alpha(image)
    palette = extract_exact_palette(image, alpha, sort_colors=False)

    assert [tuple(int(channel) for channel in row) for row in palette[1:]] == [
        (255, 0, 0),
        (0, 0, 255),
    ]


def test_extract_exact_palette_rejects_too_many_visible_colors() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for i in range(33):
        image.putpixel((i % 32, i // 32), (i, 255 - i, (i * 7) % 256, 255))

    alpha = extract_hard_alpha(image)

    with pytest.raises(ValueError, match="above max_visible_colors=32"):
        extract_exact_palette(image, alpha, max_visible_colors=32)


def test_extract_exact_palette_rejects_empty_visible_palette() -> None:
    image = Image.new("RGBA", (32, 32), (123, 45, 67, 0))
    alpha = extract_hard_alpha(image)

    with pytest.raises(ValueError, match="no opaque pixels"):
        extract_exact_palette(image, alpha)
