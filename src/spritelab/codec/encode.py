"""Strict clean 32x32 RGBA to SpriteBundle encoder."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.codec.alpha import extract_hard_alpha
from spritelab.codec.bundle import SPRITE_HEIGHT, SPRITE_WIDTH, SpriteBundle, SpriteMetadata
from spritelab.codec.canonical_palette import canonicalize_bundle_palette
from spritelab.codec.index_map import build_index_map_from_palette
from spritelab.codec.palette import extract_exact_palette, visible_palette_size
from spritelab.codec.role_inference import apply_role_inference_to_bundle
from spritelab.codec.validate import assert_valid_bundle
from spritelab.utils.image import assert_exact_size, ensure_rgba


def encode_rgba_image_to_bundle(
    image: Image.Image,
    metadata: SpriteMetadata,
    *,
    alpha_threshold: int = 128,
    max_visible_colors: int = 32,
    canonicalize_palette: bool = True,
    generate_role_map: bool = True,
) -> SpriteBundle:
    """Encode a clean 32x32 RGBA image into a validated SpriteBundle.

    The encoder is strict: it does not resize and does not quantize. Alpha is
    hardened with ``alpha_threshold`` before exact visible RGB colors are
    extracted.
    """

    assert_exact_size(image)
    rgba = ensure_rgba(image)

    alpha = extract_hard_alpha(rgba, threshold=alpha_threshold)
    palette = extract_exact_palette(
        rgba,
        alpha,
        max_visible_colors=max_visible_colors,
        sort_colors=True,
    )
    index_map = build_index_map_from_palette(rgba, alpha, palette)
    prepared_metadata = _copy_metadata_with_size_and_palette(metadata, visible_palette_size(palette))

    bundle = SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=None,
        metadata=prepared_metadata,
    )
    assert_valid_bundle(bundle)

    if generate_role_map:
        bundle = apply_role_inference_to_bundle(bundle)

    if canonicalize_palette:
        bundle = canonicalize_bundle_palette(bundle).bundle
        bundle = _copy_bundle_with_palette_size(bundle)
        assert_valid_bundle(bundle)

    return bundle


def encode_png_to_bundle(
    image_path: str | Path,
    metadata: SpriteMetadata | None = None,
    *,
    alpha_threshold: int = 128,
    max_visible_colors: int = 32,
    canonicalize_palette: bool = True,
    generate_role_map: bool = True,
) -> SpriteBundle:
    """Load a PNG and encode it into a SpriteBundle."""

    path = Path(image_path)
    if metadata is None:
        metadata = SpriteMetadata(id=path.stem, width=SPRITE_WIDTH, height=SPRITE_HEIGHT, source=str(path))

    with Image.open(path) as image:
        rgba = ensure_rgba(image).copy()

    return encode_rgba_image_to_bundle(
        rgba,
        metadata,
        alpha_threshold=alpha_threshold,
        max_visible_colors=max_visible_colors,
        canonicalize_palette=canonicalize_palette,
        generate_role_map=generate_role_map,
    )

def _copy_metadata_with_size_and_palette(metadata: SpriteMetadata, palette_size: int) -> SpriteMetadata:
    metadata_data = copy.deepcopy(metadata.to_dict())
    metadata_data["width"] = SPRITE_WIDTH
    metadata_data["height"] = SPRITE_HEIGHT
    metadata_data["palette_size"] = palette_size
    metadata_data["extra"] = _encoder_extra(metadata_data.get("extra"))
    return SpriteMetadata.from_dict(metadata_data)


def _copy_bundle_with_palette_size(bundle: SpriteBundle) -> SpriteBundle:
    metadata = _copy_metadata_with_size_and_palette(
        bundle.metadata,
        visible_palette_size(np.asarray(bundle.palette)),
    )
    return SpriteBundle(
        alpha=np.asarray(bundle.alpha).copy(),
        palette=np.asarray(bundle.palette).copy(),
        index_map=np.asarray(bundle.index_map).copy(),
        role_map=None if bundle.role_map is None else np.asarray(bundle.role_map).copy(),
        metadata=metadata,
    )


def _encoder_extra(extra: object) -> dict[str, Any]:
    copied = dict(extra or {})
    copied["encoded_from_rgba"] = True
    copied["encoder"] = "strict_rgba_32x32_v1"
    copied.setdefault("quantized", False)
    return copied
