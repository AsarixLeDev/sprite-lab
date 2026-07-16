"""Validation for sprite bundles."""

from __future__ import annotations

import json
import re

import numpy as np

from spritelab.codec.bundle import (
    BUNDLE_SCHEMA_VERSION,
    INDEX_MASK,
    INDEX_PAD,
    MAX_TRAINING_PALETTE_SLOTS,
    SPRITE_HEIGHT,
    SPRITE_SIZE,
    SPRITE_WIDTH,
    SpriteBundle,
)
from spritelab.codec.palette import MIN_PALETTE_ROWS
from spritelab.codec.roles import ROLE_NAMES, ROLE_TRANSPARENT

_METADATA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def validate_bundle(bundle: SpriteBundle) -> list[str]:
    """Return human-readable validation errors for a sprite bundle.

    Validation never mutates or repairs the bundle. Callers can decide whether
    to reject the data, report the errors, or build a corrected bundle.
    """

    errors: list[str] = []

    alpha_ok = _validate_alpha(bundle.alpha, errors)
    palette_len = _validate_palette(bundle.palette, errors)
    index_ok = _validate_index_map(bundle.index_map, palette_len, errors)
    _validate_role_map(bundle.role_map, bundle.alpha if alpha_ok else None, errors)
    _validate_metadata(bundle, palette_len, errors)

    if alpha_ok and index_ok:
        transparent_bad = (bundle.alpha == 0) & (bundle.index_map != 0)
        if bool(np.any(transparent_bad)):
            errors.append("transparent alpha pixels must have index_map value 0.")

        opaque_bad = (bundle.alpha == 1) & (bundle.index_map == 0)
        if bool(np.any(opaque_bad)):
            errors.append("opaque alpha pixels must use palette slots 1..K-1, not 0.")

    return errors


def assert_valid_bundle(bundle: SpriteBundle) -> None:
    """Raise ValueError if a sprite bundle is invalid."""

    errors = validate_bundle(bundle)
    if errors:
        message = "Invalid SpriteBundle:\n- " + "\n- ".join(errors)
        raise ValueError(message)


def _validate_alpha(alpha: object, errors: list[str]) -> bool:
    if not isinstance(alpha, np.ndarray):
        errors.append("alpha must be a numpy array.")
        return False

    if alpha.shape != SPRITE_SIZE:
        errors.append("alpha shape must be exactly 32x32.")
        return False

    if not np.all(np.isin(alpha, [0, 1])):
        errors.append("alpha values must be only 0 or 1.")
        return False

    return True


def _validate_palette(palette: object, errors: list[str]) -> int | None:
    if not isinstance(palette, np.ndarray):
        errors.append("palette must be a numpy array.")
        return None

    if palette.ndim != 2 or palette.shape[1] != 3:
        errors.append("palette shape must be Kx3 RGB.")
        return None

    if palette.shape[0] < MIN_PALETTE_ROWS:
        errors.append("palette must have at least 2 rows: dummy transparent + one visible color.")

    if not _is_safely_uint8_convertible(palette):
        errors.append("palette dtype and values must be uint8-compatible RGB values in 0..255.")
        return int(palette.shape[0])

    visible_row_count = int(palette.shape[0] - 1)
    if visible_row_count > MAX_TRAINING_PALETTE_SLOTS:
        errors.append("palette has too many visible rows for the SpriteBundle token contract.")

    palette_values = np.asarray(palette)
    if palette_values.shape[0] >= 1 and not np.array_equal(palette_values[0], np.array([0, 0, 0])):
        errors.append("palette[0] must be the dummy transparent RGB slot [0, 0, 0].")

    visible_rows = palette_values[1:]
    if visible_rows.shape[0] > 1:
        visible_tuples = [tuple(int(channel) for channel in row) for row in visible_rows]
        if len(set(visible_tuples)) != len(visible_tuples):
            errors.append("visible palette rows must be unique.")

    return int(palette.shape[0])


def _validate_index_map(index_map: object, palette_len: int | None, errors: list[str]) -> bool:
    if not isinstance(index_map, np.ndarray):
        errors.append("index_map must be a numpy array.")
        return False

    if index_map.shape != SPRITE_SIZE:
        errors.append("index_map shape must be exactly 32x32.")
        return False

    if not np.issubdtype(index_map.dtype, np.integer):
        errors.append("index_map dtype must be an integer type.")
        return False

    if index_map.size and int(np.min(index_map)) < 0:
        errors.append("index_map contains negative values.")

    if index_map.size and bool(np.any(np.isin(index_map, [INDEX_MASK, INDEX_PAD]))):
        errors.append("index_map contains reserved training token values.")

    if palette_len is not None and index_map.size and int(np.max(index_map)) >= palette_len:
        errors.append("index_map contains values outside the palette range.")

    return True


def _validate_role_map(role_map: object, alpha: object | None, errors: list[str]) -> None:
    if role_map is None:
        return

    if not isinstance(role_map, np.ndarray):
        errors.append("role_map must be a numpy array when present.")
        return

    if role_map.shape != SPRITE_SIZE:
        errors.append("role_map shape must be exactly 32x32 when present.")
        return

    if not np.issubdtype(role_map.dtype, np.integer):
        errors.append("role_map dtype must be an integer type.")
        return

    role_values = [int(value) for value in np.unique(role_map)]
    if any(value not in ROLE_NAMES for value in role_values):
        errors.append("role_map contains unknown role IDs.")

    if isinstance(alpha, np.ndarray) and alpha.shape == SPRITE_SIZE:
        transparent_bad = (alpha == 0) & (role_map != ROLE_TRANSPARENT)
        if bool(np.any(transparent_bad)):
            errors.append("transparent alpha pixels must have role_map value ROLE_TRANSPARENT.")

        opaque_bad = (alpha == 1) & (role_map == ROLE_TRANSPARENT)
        if bool(np.any(opaque_bad)):
            errors.append("opaque alpha pixels must not have role_map value ROLE_TRANSPARENT.")


def _validate_metadata(bundle: SpriteBundle, palette_len: int | None, errors: list[str]) -> None:
    metadata = bundle.metadata

    metadata_id = getattr(metadata, "id", None)
    if not isinstance(metadata_id, str) or not metadata_id.strip() or not _METADATA_ID_RE.fullmatch(metadata_id):
        errors.append("metadata id must be a non-empty filesystem-safe identifier.")

    if getattr(metadata, "width", None) != SPRITE_WIDTH or getattr(metadata, "height", None) != SPRITE_HEIGHT:
        errors.append("metadata width and height must both be 32.")

    palette_size = getattr(metadata, "palette_size", None)
    if palette_size is not None:
        palette_size_ok = isinstance(palette_size, int) and not isinstance(palette_size, bool)
        expected_visible_rows = palette_len - 1 if palette_len is not None else None
        if not palette_size_ok or palette_size != expected_visible_rows:
            errors.append("metadata.palette_size must match the number of visible palette rows.")

    if getattr(metadata, "bundle_schema_version", None) != BUNDLE_SCHEMA_VERSION:
        errors.append("unsupported bundle schema version.")

    codec_version = getattr(metadata, "codec_version", None)
    if not isinstance(codec_version, str) or not codec_version.strip():
        errors.append("metadata.codec_version must be a non-empty string.")

    try:
        json.dumps(metadata.to_dict())
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"metadata must be JSON-serializable: {exc}")


def _is_safely_uint8_convertible(array: np.ndarray) -> bool:
    if array.dtype == np.uint8:
        return True

    if not np.issubdtype(array.dtype, np.integer):
        return False

    if array.size == 0:
        return True

    return int(np.min(array)) >= 0 and int(np.max(array)) <= 255
