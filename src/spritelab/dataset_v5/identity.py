"""Opaque, content-bound identities for the raw Dataset-v5 rebuild.

This module deliberately has no dependency on filenames used by prepared
datasets, label proposals, or source descriptions.  A record binding may use
the immutable original archive member path, but that value is hashed and is
never exposed as an identifier.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

RECORD_ID_VERSION = "raw_record_identity_v1"
BLOB_ID_VERSION = "decoded_rgba_v1"
GEOMETRY_ID_VERSION = "geometry_family_v1"

_OPAQUE_RECORD_RE = re.compile(r"^rec_[0-9a-f]{64}$")
_OPAQUE_GEOMETRY_RE = re.compile(r"^geo_[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON representation used in identities."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_rgba_bytes(rgba: np.ndarray) -> bytes:
    """Encode decoded RGBA pixels with unambiguous dimensions and versioning."""

    value = np.ascontiguousarray(rgba, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 4:
        raise ValueError(f"expected RGBA array [height,width,4], got {value.shape}")
    height, width = value.shape[:2]
    return BLOB_ID_VERSION.encode("ascii") + b"\0" + struct.pack(">II", width, height) + value.tobytes()


def decoded_rgba_sha256(rgba: np.ndarray) -> str:
    """Hash decoded pixels, not a PNG encoder's byte representation."""

    return hashlib.sha256(canonical_rgba_bytes(rgba)).hexdigest()


@dataclass(frozen=True)
class RecordBinding:
    """Immutable source and extraction facts that bind one logical record."""

    source_archive_sha256: str
    archive_member_path: str
    extraction_operation: str
    crop_coordinates: tuple[int, int, int, int] | None
    decoded_rgba_sha256: str
    padding_operation: Mapping[str, Any] | None = None
    identity_version: str = RECORD_ID_VERSION

    def canonical(self) -> dict[str, Any]:
        archive_hash = _require_sha256(self.source_archive_sha256, "source_archive_sha256")
        decoded_hash = _require_sha256(self.decoded_rgba_sha256, "decoded_rgba_sha256")
        member = self.archive_member_path.replace("\\", "/")
        if not member or member.startswith("/") or "\0" in member:
            raise ValueError("archive_member_path must be a non-empty relative member path")
        if self.crop_coordinates is not None:
            _validate_crop(self.crop_coordinates)
        return {
            "archive_member_path": member,
            "crop_coordinates": list(self.crop_coordinates) if self.crop_coordinates is not None else None,
            "decoded_rgba_sha256": decoded_hash,
            "extraction_operation": str(self.extraction_operation),
            "identity_version": self.identity_version,
            "padding_operation": _canonical_mapping(self.padding_operation),
            "source_archive_sha256": archive_hash,
        }


def make_record_id(binding: RecordBinding) -> str:
    """Create a semantic-free record identifier from an immutable binding."""

    digest = hashlib.sha256(canonical_json_bytes(binding.canonical())).hexdigest()
    return f"rec_{digest}"


def make_geometry_family_id(alpha: np.ndarray) -> str:
    """Hash translation-normalized foreground geometry.

    Geometry identity intentionally ignores RGB values and transparent padding.
    Empty masks retain their original dimensions so unrelated blank sheets do
    not all become one useful geometry family.
    """

    mask = np.ascontiguousarray(np.asarray(alpha) > 0, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D alpha mask, got {mask.shape}")
    ys, xs = np.nonzero(mask)
    if len(xs):
        tight = mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        payload = {
            "height": int(tight.shape[0]),
            "mask_hex": np.packbits(tight, axis=None).tobytes().hex(),
            "visible_pixels": int(tight.sum()),
            "width": int(tight.shape[1]),
        }
    else:
        payload = {"blank_height": int(mask.shape[0]), "blank_width": int(mask.shape[1]), "visible_pixels": 0}
    digest = hashlib.sha256(GEOMETRY_ID_VERSION.encode("ascii") + b"\0" + canonical_json_bytes(payload)).hexdigest()
    return f"geo_{digest}"


def is_opaque_record_id(value: str) -> bool:
    return bool(_OPAQUE_RECORD_RE.fullmatch(value))


def is_opaque_geometry_id(value: str) -> bool:
    return bool(_OPAQUE_GEOMETRY_RE.fullmatch(value))


def assert_opaque_id(value: str, *, kind: str = "record") -> None:
    predicate = is_opaque_record_id if kind == "record" else is_opaque_geometry_id
    if not predicate(value):
        raise ValueError(f"non-opaque {kind} identifier: {value!r}")


def relation_membership_key(record_ids: Sequence[str]) -> str:
    """Create a deterministic, semantic-free relation membership digest."""

    values = sorted(set(record_ids))
    for value in values:
        assert_opaque_id(value)
    return hashlib.sha256(b"relation_membership_v1\0" + b"\0".join(v.encode("ascii") for v in values)).hexdigest()


def _validate_crop(crop: tuple[int, int, int, int]) -> None:
    if len(crop) != 4 or any(not isinstance(value, int) for value in crop):
        raise ValueError("crop_coordinates must contain four integers")
    left, top, right, bottom = crop
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f"invalid crop_coordinates: {crop}")


def _require_sha256(value: str, field: str) -> str:
    normalized = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return normalized


def _canonical_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
