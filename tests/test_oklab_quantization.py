from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spritelab.codec.quantize import QuantizationOptions, quantize_rgba_image_to_palette_indices


def test_quantizing_over_color_image_respects_palette_and_index_conventions() -> None:
    image = _over_color_image()

    result = quantize_rgba_image_to_palette_indices(
        image,
        options=QuantizationOptions(target_visible_colors=16, canonicalize_palette=False),
    )

    assert result.palette.shape[0] <= 17
    assert np.array_equal(result.palette[0], np.array([0, 0, 0], dtype=np.uint8))
    assert result.palette.dtype == np.uint8
    assert result.index_map.shape == (32, 32)
    assert result.alpha.shape == (32, 32)
    assert np.all(result.index_map[result.alpha == 0] == 0)
    assert np.all(result.index_map[result.alpha == 1] > 0)
    assert int(result.index_map.max()) < result.palette.shape[0]
    assert result.mean_oklab_error >= 0.0
    assert result.max_oklab_error >= 0.0
    assert result.original_visible_color_count > result.quantized_visible_color_count


def test_quantization_is_deterministic_with_same_options() -> None:
    image = _over_color_image()
    options = QuantizationOptions(target_visible_colors=8, seed=77, canonicalize_palette=False)

    first = quantize_rgba_image_to_palette_indices(image, options=options)
    second = quantize_rgba_image_to_palette_indices(image, options=options)

    np.testing.assert_array_equal(first.palette, second.palette)
    np.testing.assert_array_equal(first.index_map, second.index_map)
    np.testing.assert_array_equal(first.alpha, second.alpha)
    assert first.mean_oklab_error == second.mean_oklab_error


def test_preserve_exact_if_under_limit_keeps_exact_colors() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            image.putpixel((x, y), (10, 20, 30, 255) if x < 16 else (200, 210, 220, 255))

    result = quantize_rgba_image_to_palette_indices(
        image,
        options=QuantizationOptions(target_visible_colors=16, preserve_exact_if_under_limit=True),
    )

    assert result.original_visible_color_count == 2
    assert result.quantized_visible_color_count == 2
    assert result.mean_oklab_error == 0.0
    assert result.max_oklab_error == 0.0
    assert {tuple(row) for row in result.palette[1:]} == {(10, 20, 30), (200, 210, 220)}


def test_empty_sprite_raises_useful_error() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))

    with pytest.raises(ValueError, match="empty sprite"):
        quantize_rgba_image_to_palette_indices(image, options=QuantizationOptions())


def test_near_identical_colors_can_collapse_to_one_quantized_color() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(8, 24):
        for x in range(8, 24):
            image.putpixel((x, y), (120, 80, 200, 255) if x < 16 else (121, 81, 201, 255))

    result = quantize_rgba_image_to_palette_indices(
        image,
        options=QuantizationOptions(target_visible_colors=1, preserve_exact_if_under_limit=False),
    )

    assert result.original_visible_color_count == 2
    assert result.quantized_visible_color_count == 1
    assert result.palette.shape == (2, 3)


def _over_color_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(6, 26):
        for x in range(6, 26):
            red = 40 + x * 5
            green = 30 + y * 4
            blue = 80 + ((x + y) % 20) * 5
            image.putpixel((x, y), (red % 256, green % 256, blue % 256, 255))
    return image
