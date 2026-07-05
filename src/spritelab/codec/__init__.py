"""Sprite bundle codec helpers."""

from spritelab.codec.bundle import (
    BUNDLE_SCHEMA_VERSION,
    CODEC_VERSION,
    INDEX_MASK,
    INDEX_PAD,
    INDEX_TRANSPARENT,
    MAX_TRAINING_PALETTE_SLOTS,
    SpriteBundle,
    SpriteMetadata,
)
from spritelab.codec.canonical_palette import (
    CanonicalizationResult,
    PaletteSlotStats,
    canonical_palette_order,
    canonicalize_bundle_palette,
    compute_palette_slot_stats,
    remap_index_map,
)
from spritelab.codec.alpha import extract_hard_alpha
from spritelab.codec.encode import encode_png_to_bundle, encode_rgba_image_to_bundle
from spritelab.codec.index_map import build_index_map_from_palette
from spritelab.codec.io import load_bundle, save_bundle
from spritelab.codec.palette import extract_exact_palette
from spritelab.codec.preview import make_preview, save_preview
from spritelab.codec.quantize import (
    QuantizationOptions,
    QuantizationResult,
    encode_png_to_quantized_bundle,
    encode_rgba_image_to_quantized_bundle,
    fit_oklab_kmeans,
    quantize_rgba_image_to_palette_indices,
)
from spritelab.codec.role_inference import (
    PaletteSlotRoleFeatures,
    RoleInferenceOptions,
    RoleInferenceResult,
    apply_role_inference_to_bundle,
    build_role_map_from_slot_roles,
    compute_palette_slot_role_features,
    describe_role_inference,
    infer_palette_slot_roles_v2,
    role_map_to_preview_image,
    save_role_map_preview,
    validate_role_map,
)
from spritelab.codec.reconstruct import reconstruct_rgba, save_reconstructed_png
from spritelab.codec.validate import assert_valid_bundle, validate_bundle

__all__ = [
    "CanonicalizationResult",
    "BUNDLE_SCHEMA_VERSION",
    "CODEC_VERSION",
    "INDEX_MASK",
    "INDEX_PAD",
    "INDEX_TRANSPARENT",
    "MAX_TRAINING_PALETTE_SLOTS",
    "PaletteSlotStats",
    "PaletteSlotRoleFeatures",
    "QuantizationOptions",
    "QuantizationResult",
    "RoleInferenceOptions",
    "RoleInferenceResult",
    "SpriteBundle",
    "SpriteMetadata",
    "apply_role_inference_to_bundle",
    "assert_valid_bundle",
    "build_role_map_from_slot_roles",
    "canonical_palette_order",
    "canonicalize_bundle_palette",
    "build_index_map_from_palette",
    "compute_palette_slot_stats",
    "compute_palette_slot_role_features",
    "describe_role_inference",
    "encode_png_to_bundle",
    "encode_png_to_quantized_bundle",
    "encode_rgba_image_to_bundle",
    "encode_rgba_image_to_quantized_bundle",
    "extract_exact_palette",
    "extract_hard_alpha",
    "fit_oklab_kmeans",
    "infer_palette_slot_roles_v2",
    "load_bundle",
    "make_preview",
    "quantize_rgba_image_to_palette_indices",
    "reconstruct_rgba",
    "remap_index_map",
    "role_map_to_preview_image",
    "save_bundle",
    "save_preview",
    "save_reconstructed_png",
    "save_role_map_preview",
    "validate_bundle",
    "validate_role_map",
]
