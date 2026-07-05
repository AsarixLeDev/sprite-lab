from __future__ import annotations

import numpy as np

from spritelab.codec.oklab import oklab_array_to_rgb_u8, rgb_u8_array_to_oklab


def test_oklab_array_conversion_preserves_shape_and_dtype() -> None:
    rgb = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 0, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )

    oklab = rgb_u8_array_to_oklab(rgb)
    roundtrip = oklab_array_to_rgb_u8(oklab)

    assert oklab.shape == rgb.shape
    assert roundtrip.shape == rgb.shape
    assert roundtrip.dtype == np.uint8


def test_oklab_roundtrip_basic_colors_are_close() -> None:
    rgb = np.array(
        [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 96, 32],
        ],
        dtype=np.uint8,
    )

    roundtrip = oklab_array_to_rgb_u8(rgb_u8_array_to_oklab(rgb))

    assert np.max(np.abs(roundtrip.astype(np.int16) - rgb.astype(np.int16))) <= 2
    assert np.array_equal(roundtrip[0], np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(roundtrip[1], np.array([255, 255, 255], dtype=np.uint8))


def test_oklab_to_rgb_clips_to_uint8_range() -> None:
    odd_oklab = np.array([[2.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], dtype=np.float64)

    rgb = oklab_array_to_rgb_u8(odd_oklab)

    assert rgb.dtype == np.uint8
    assert int(rgb.min()) >= 0
    assert int(rgb.max()) <= 255
