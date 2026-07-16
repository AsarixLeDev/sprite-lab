from __future__ import annotations

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.encode import encode_rgba_image_to_bundle
from spritelab.codec.validate import assert_valid_bundle


def make_clean_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (9, 8, 7, 0))
    image.putpixel((0, 0), (10, 10, 10, 255))
    image.putpixel((1, 0), (220, 200, 40, 255))
    image.putpixel((2, 0), (80, 40, 120, 255))
    return image


def test_encode_rgba_image_returns_valid_sprite_bundle() -> None:
    metadata = SpriteMetadata(id="encoded", category="test")

    bundle = encode_rgba_image_to_bundle(make_clean_image(), metadata, canonicalize_palette=False)

    assert isinstance(bundle, SpriteBundle)
    assert_valid_bundle(bundle)
    assert bundle.alpha.shape == (32, 32)
    assert bundle.palette.shape == (4, 3)
    assert bundle.index_map.shape == (32, 32)
    assert bundle.role_map is not None
    assert bundle.metadata.palette_size == 3
    assert bundle.metadata.id == "encoded"


def test_encode_rgba_image_index_conventions() -> None:
    bundle = encode_rgba_image_to_bundle(
        make_clean_image(),
        SpriteMetadata(id="index_conventions"),
        canonicalize_palette=False,
    )

    assert np.all(bundle.index_map[bundle.alpha == 0] == 0)
    assert np.all(bundle.index_map[bundle.alpha == 1] != 0)


def test_encode_rgba_image_valid_with_and_without_canonicalization() -> None:
    image = make_clean_image()
    metadata = SpriteMetadata(id="canonical_flag")

    canonical = encode_rgba_image_to_bundle(image, metadata, canonicalize_palette=True)
    raw = encode_rgba_image_to_bundle(image, metadata, canonicalize_palette=False)

    assert_valid_bundle(canonical)
    assert_valid_bundle(raw)
    assert canonical.metadata.palette_size == raw.metadata.palette_size == 3
    assert canonical.metadata.extra["palette_canonicalized"] is True
    assert "palette_canonicalized" not in raw.metadata.extra


def test_encode_rgba_image_can_omit_role_map() -> None:
    bundle = encode_rgba_image_to_bundle(
        make_clean_image(),
        SpriteMetadata(id="no_roles"),
        canonicalize_palette=False,
        generate_role_map=False,
    )

    assert bundle.role_map is None
    assert_valid_bundle(bundle)
