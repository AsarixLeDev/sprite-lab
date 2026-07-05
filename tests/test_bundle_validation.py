from __future__ import annotations

import numpy as np

from spritelab.codec.bundle import (
    BUNDLE_SCHEMA_VERSION,
    CODEC_VERSION,
    INDEX_MASK,
    INDEX_PAD,
    MAX_TRAINING_PALETTE_SLOTS,
    SpriteBundle,
    SpriteMetadata,
)
from spritelab.codec.roles import ROLE_MIDTONE, ROLE_TRANSPARENT
from spritelab.codec.validate import validate_bundle


def make_valid_bundle() -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    role_map = np.zeros((32, 32), dtype=np.uint8)

    alpha[5, 5] = 1
    index_map[5, 5] = 1
    role_map[5, 5] = ROLE_MIDTONE

    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
        ],
        dtype=np.uint8,
    )

    metadata = SpriteMetadata(id="validation")
    return SpriteBundle(alpha, palette, index_map, role_map, metadata)


def assert_has_error(bundle: SpriteBundle, text: str) -> None:
    errors = validate_bundle(bundle)
    assert any(text in error for error in errors), errors


def test_validation_catches_wrong_alpha_shape() -> None:
    bundle = make_valid_bundle()
    bundle.alpha = np.zeros((31, 32), dtype=np.uint8)

    assert_has_error(bundle, "alpha shape")


def test_validation_catches_non_binary_alpha() -> None:
    bundle = make_valid_bundle()
    bundle.alpha[0, 0] = 2

    assert_has_error(bundle, "alpha values")


def test_validation_catches_index_outside_palette_range() -> None:
    bundle = make_valid_bundle()
    bundle.index_map[5, 5] = 2

    assert_has_error(bundle, "outside the palette range")


def test_validation_catches_transparent_pixel_with_nonzero_index() -> None:
    bundle = make_valid_bundle()
    bundle.index_map[0, 0] = 1

    assert_has_error(bundle, "transparent alpha pixels")


def test_validation_catches_opaque_pixel_with_zero_index() -> None:
    bundle = make_valid_bundle()
    bundle.index_map[5, 5] = 0

    assert_has_error(bundle, "opaque alpha pixels")


def test_validation_catches_invalid_role_map_shape() -> None:
    bundle = make_valid_bundle()
    bundle.role_map = np.zeros((32, 31), dtype=np.uint8)

    assert_has_error(bundle, "role_map shape")


def test_valid_bundle_with_default_metadata_passes() -> None:
    assert validate_bundle(make_valid_bundle()) == []


def test_metadata_id_cannot_be_empty() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.id = ""

    assert_has_error(bundle, "metadata id")


def test_metadata_id_cannot_contain_spaces() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.id = "bad id"

    assert_has_error(bundle, "metadata id")


def test_metadata_id_cannot_contain_forward_slash() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.id = "bad/id"

    assert_has_error(bundle, "metadata id")


def test_metadata_id_cannot_contain_backslash() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.id = "bad\\id"

    assert_has_error(bundle, "metadata id")


def test_metadata_id_cannot_contain_path_traversal_like_value() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.id = "../evil"

    assert_has_error(bundle, "metadata id")


def test_metadata_palette_size_must_match_visible_palette_rows() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.palette_size = 2

    assert_has_error(bundle, "metadata.palette_size")


def test_old_metadata_dict_without_version_fields_loads_with_defaults() -> None:
    metadata = SpriteMetadata.from_dict({"id": "old_metadata"})

    assert metadata.bundle_schema_version == BUNDLE_SCHEMA_VERSION
    assert metadata.codec_version == CODEC_VERSION

    bundle = make_valid_bundle()
    bundle.metadata = metadata
    assert validate_bundle(bundle) == []


def test_unsupported_bundle_schema_version_is_rejected() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.bundle_schema_version = "0.9"

    assert_has_error(bundle, "unsupported bundle schema version")


def test_empty_codec_version_is_rejected() -> None:
    bundle = make_valid_bundle()
    bundle.metadata.codec_version = ""

    assert_has_error(bundle, "codec_version")


def test_palette_zero_must_be_dummy_transparent_black() -> None:
    bundle = make_valid_bundle()
    bundle.palette[0] = [1, 2, 3]

    assert_has_error(bundle, "palette[0]")


def test_duplicate_visible_palette_rows_are_rejected() -> None:
    bundle = make_valid_bundle()
    bundle.palette = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.uint8,
    )

    assert_has_error(bundle, "visible palette rows")


def test_visible_black_after_dummy_slot_is_allowed_when_unique() -> None:
    bundle = make_valid_bundle()
    bundle.palette = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert validate_bundle(bundle) == []


def test_too_many_visible_palette_rows_are_rejected() -> None:
    bundle = make_valid_bundle()
    visible_rows = [
        [index % 256, index // 256 + 1, 0]
        for index in range(MAX_TRAINING_PALETTE_SLOTS + 1)
    ]
    bundle.palette = np.array([[0, 0, 0], *visible_rows], dtype=np.uint8)

    assert_has_error(bundle, "too many visible rows")


def test_index_mask_token_is_rejected_in_normal_index_map() -> None:
    bundle = make_valid_bundle()
    bundle.index_map[0, 0] = INDEX_MASK

    assert_has_error(bundle, "reserved training token")


def test_index_pad_token_is_rejected_in_normal_index_map() -> None:
    bundle = make_valid_bundle()
    bundle.index_map[0, 0] = INDEX_PAD

    assert_has_error(bundle, "reserved training token")


def test_non_integer_role_map_dtype_is_rejected() -> None:
    bundle = make_valid_bundle()
    bundle.role_map = bundle.role_map.astype(np.float32)

    assert_has_error(bundle, "role_map dtype")


def test_unknown_role_id_is_rejected() -> None:
    bundle = make_valid_bundle()
    bundle.role_map[5, 5] = 123

    assert_has_error(bundle, "unknown role IDs")


def test_transparent_alpha_pixels_must_have_transparent_role() -> None:
    bundle = make_valid_bundle()
    bundle.role_map[0, 0] = ROLE_MIDTONE

    assert_has_error(bundle, "transparent alpha pixels")


def test_opaque_alpha_pixels_must_not_have_transparent_role() -> None:
    bundle = make_valid_bundle()
    bundle.role_map[5, 5] = ROLE_TRANSPARENT

    assert_has_error(bundle, "opaque alpha pixels")


def test_valid_role_map_with_known_roles_passes() -> None:
    bundle = make_valid_bundle()

    assert validate_bundle(bundle) == []
