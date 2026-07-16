from __future__ import annotations

import pytest
from PIL import Image

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.encode import encode_rgba_image_to_bundle
from spritelab.codec.quantize import QuantizationOptions, encode_rgba_image_to_quantized_bundle
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.codec.validate import assert_valid_bundle


def test_quantized_encoder_returns_valid_bundle_with_metadata() -> None:
    image = _over_color_image()
    metadata = SpriteMetadata(id="over_color", category="item_icon")

    bundle = encode_rgba_image_to_quantized_bundle(
        image,
        metadata,
        options=QuantizationOptions(target_visible_colors=16, canonicalize_palette=True),
    )

    assert isinstance(bundle, SpriteBundle)
    assert_valid_bundle(bundle)
    assert bundle.palette.shape[0] <= 17
    assert bundle.metadata.palette_size == bundle.palette.shape[0] - 1
    assert bundle.metadata.extra["quantized"] is True
    assert bundle.metadata.extra["original_visible_color_count"] > 16
    assert bundle.metadata.extra["quantized_visible_color_count"] <= 16
    assert bundle.metadata.extra["mean_oklab_error"] >= 0.0

    reconstructed = reconstruct_rgba(bundle)
    assert reconstructed.mode == "RGBA"
    assert reconstructed.size == (32, 32)


def test_quantized_encoder_without_canonicalization_is_valid() -> None:
    image = _over_color_image()

    bundle = encode_rgba_image_to_quantized_bundle(
        image,
        SpriteMetadata(id="raw_quantized"),
        options=QuantizationOptions(target_visible_colors=8, canonicalize_palette=False),
    )

    assert_valid_bundle(bundle)
    assert bundle.palette.shape[0] <= 9
    assert bundle.metadata.extra["quantization"]["target_visible_colors"] == 8


def test_strict_encoder_rejects_over_color_image_but_quantized_encoder_accepts() -> None:
    image = _over_color_image()

    with pytest.raises(ValueError, match="above max_visible_colors"):
        encode_rgba_image_to_bundle(
            image,
            SpriteMetadata(id="strict"),
            max_visible_colors=16,
        )

    bundle = encode_rgba_image_to_quantized_bundle(
        image,
        SpriteMetadata(id="quantized"),
        options=QuantizationOptions(target_visible_colors=16),
    )

    assert_valid_bundle(bundle)


def _over_color_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(5, 27):
        for x in range(5, 27):
            red = 32 + x * 6
            green = 24 + y * 5
            blue = 60 + ((x * 3 + y * 2) % 40)
            image.putpixel((x, y), (red % 256, green % 256, blue % 256, 255))
    return image
