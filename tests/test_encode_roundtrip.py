from __future__ import annotations

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SpriteMetadata
from spritelab.codec.encode import encode_rgba_image_to_bundle
from spritelab.codec.reconstruct import reconstruct_rgba


def make_roundtrip_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (99, 88, 77, 0))
    image.putpixel((4, 4), (12, 10, 18, 255))
    image.putpixel((5, 4), (116, 72, 152, 255))
    image.putpixel((6, 4), (255, 238, 64, 255))
    return image


def normalized_expected(image: Image.Image) -> Image.Image:
    expected = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for y in range(image.height):
        for x in range(image.width):
            red, green, blue, alpha = image.getpixel((x, y))
            if alpha == 255:
                expected.putpixel((x, y), (red, green, blue, 255))
    return expected


def test_encode_reconstruct_roundtrip_matches_normalized_source() -> None:
    image = make_roundtrip_image()
    bundle = encode_rgba_image_to_bundle(
        image,
        SpriteMetadata(id="roundtrip"),
        canonicalize_palette=False,
    )

    reconstructed = reconstruct_rgba(bundle)

    assert reconstructed.tobytes() == normalized_expected(image).tobytes()


def test_canonicalization_does_not_change_reconstructed_pixels() -> None:
    image = make_roundtrip_image()
    raw_bundle = encode_rgba_image_to_bundle(
        image,
        SpriteMetadata(id="roundtrip_raw"),
        canonicalize_palette=False,
    )
    canonical_bundle = encode_rgba_image_to_bundle(
        image,
        SpriteMetadata(id="roundtrip_canonical"),
        canonicalize_palette=True,
    )

    np.testing.assert_array_equal(
        np.asarray(reconstruct_rgba(raw_bundle)),
        np.asarray(reconstruct_rgba(canonical_bundle)),
    )
