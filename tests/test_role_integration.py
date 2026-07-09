from __future__ import annotations

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SpriteMetadata
from spritelab.codec.canonical_palette import canonicalize_bundle_palette
from spritelab.codec.encode import encode_rgba_image_to_bundle
from spritelab.codec.quantize import QuantizationOptions, encode_rgba_image_to_quantized_bundle
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.codec.role_inference import apply_role_inference_to_bundle, validate_role_map
from spritelab.codec.validate import assert_valid_bundle
from test_role_helpers import make_role_demo_bundle


def test_strict_encoder_generates_inferred_role_map() -> None:
    image = _clean_role_image()

    bundle = encode_rgba_image_to_bundle(
        image,
        SpriteMetadata(id="strict_roles"),
        canonicalize_palette=False,
        generate_role_map=True,
    )

    assert bundle.role_map is not None
    assert validate_role_map(bundle.role_map, bundle.alpha) == []
    assert bundle.metadata.extra["role_inference"]["version"] == "v2_heuristic"


def test_quantized_encoder_generates_inferred_role_map() -> None:
    image = _over_color_role_image()

    bundle = encode_rgba_image_to_quantized_bundle(
        image,
        SpriteMetadata(id="quant_roles"),
        options=QuantizationOptions(target_visible_colors=8, canonicalize_palette=False),
    )

    assert bundle.role_map is not None
    assert validate_role_map(bundle.role_map, bundle.alpha) == []
    assert bundle.metadata.extra["role_inference"]["version"] == "v2_heuristic"


def test_canonicalization_preserves_decoded_rgba_and_valid_role_map() -> None:
    bundle = apply_role_inference_to_bundle(make_role_demo_bundle())
    before = reconstruct_rgba(bundle)

    result = canonicalize_bundle_palette(bundle)
    after = reconstruct_rgba(result.bundle)

    np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    assert_valid_bundle(result.bundle)
    assert result.bundle.role_map is not None
    assert validate_role_map(result.bundle.role_map, result.bundle.alpha) == []


def test_apply_role_inference_does_not_mutate_input_bundle() -> None:
    bundle = make_role_demo_bundle()
    original_alpha = bundle.alpha.copy()
    original_index = bundle.index_map.copy()
    original_palette = bundle.palette.copy()

    inferred = apply_role_inference_to_bundle(bundle)

    assert inferred is not bundle
    assert inferred.role_map is not None
    assert bundle.role_map is None
    np.testing.assert_array_equal(bundle.alpha, original_alpha)
    np.testing.assert_array_equal(bundle.index_map, original_index)
    np.testing.assert_array_equal(bundle.palette, original_palette)


def _clean_role_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(8, 24):
        for x in range(8, 24):
            color = (120, 80, 160, 255)
            if x in (8, 23) or y in (8, 23):
                color = (10, 10, 16, 255)
            elif y < 13:
                color = (70, 42, 105, 255)
            image.putpixel((x, y), color)
    image.putpixel((18, 11), (250, 220, 70, 255))
    return image


def _over_color_role_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(5, 27):
        for x in range(5, 27):
            edge = x in (5, 26) or y in (5, 26)
            if edge:
                color = (12 + x % 8, 10 + y % 8, 18 + (x + y) % 8, 255)
            else:
                color = ((90 + x * 5) % 255, (50 + y * 6) % 255, (110 + x + y) % 255, 255)
            image.putpixel((x, y), color)
    return image
