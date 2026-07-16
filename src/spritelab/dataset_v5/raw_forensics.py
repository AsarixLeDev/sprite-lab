"""Read-only forensic inventory of the original Dataset-v5 downloads.

This module is deliberately more permissive than :mod:`raw_inventory`.  It
records incomplete and unsafe evidence instead of aborting at the first bad
source row.  Permissive collection does *not* make the evidence eligible for
dataset membership: every gap is retained as a blocking issue and the
fail-closed decision is emitted with each record.

Only paths explicitly rooted below ``source_root`` are read.  No source is
renamed, extracted to disk, rewritten, or assigned provenance from filesystem
timestamps.  Original names are provenance evidence and must never be passed
to a blind semantic prompt.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import warnings
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, UnidentifiedImageError

from spritelab.dataset_v5.identity import canonical_json_bytes, decoded_rgba_sha256
from spritelab.harvest.archive import appledouble_detection_basis, is_appledouble_record
from spritelab.harvest.sources import TRAINING_ALLOWED_LICENSES, normalize_license_name

RAW_FORENSIC_INVENTORY_SCHEMA_VERSION = "sprite_lab_raw_forensic_inventory_v1"
RAW_FORENSIC_ARCHIVE_SCHEMA_VERSION = "sprite_lab_raw_forensic_archive_hashes_v1"
RAW_FORENSIC_REPORT_VERSION = "sprite_lab_raw_forensic_report_v1"
RAW_EXTRACTION_OPERATION_SCHEMA_VERSION = "sprite_lab_raw_extraction_operation_v1"
RAW_EXTRACTION_OPERATIONS = (
    "direct_decode",
    "frame_select",
    "crop",
    "sheet_cell",
    "center_pad",
    "crop_then_pad",
    "reject_resource_fork",
    "reject_unreadable",
    "exclude_ambiguous",
    "exclude_ambiguous_coordinate",
)
RAW_OUTPUT_OPERATIONS = frozenset(RAW_EXTRACTION_OPERATIONS[:6])
RAW_TERMINAL_OPERATIONS = frozenset(RAW_EXTRACTION_OPERATIONS[6:])
SUPPORTED_RAW_IMAGE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tga", ".tif", ".tiff", ".webp"})
_PROVENANCE_GAP_ISSUES = frozenset(
    {
        "missing_acquisition_url",
        "missing_creator_or_publisher",
        "missing_historical_archive_sha256",
        "missing_license",
        "missing_license_url",
        "missing_original_filename",
        "missing_source_id",
        "unknown_license",
    }
)


class ExtractionOperationError(ValueError):
    """Raised when an explicit raw extraction operation is invalid or changed."""


@dataclass(frozen=True, order=True)
class CandidateSheetCoordinate:
    row: int
    column: int

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (self.row, self.column)):
            raise ExtractionOperationError("candidate sheet coordinates must be non-negative integers")

    def canonical(self) -> dict[str, int]:
        return {"column": self.column, "row": self.row}


@dataclass(frozen=True)
class RawExtractionOperation:
    """Complete versioned identity for one output or terminal exclusion.

    Every field is required at construction.  ``None`` therefore means
    explicitly not applicable; it never means "use a default".
    """

    operation_version: str
    operation: str
    source_archive_sha256: str
    archive_member_path: str
    source_member_sha256: str | None
    frame_index: int | None
    crop_rectangle: tuple[int, int, int, int] | None
    sheet_row: int | None
    sheet_column: int | None
    cell_width: int | None
    cell_height: int | None
    padding_dimensions: tuple[int, int, int, int] | None
    interpolation_policy: str
    decoded_rgba_sha256: str | None
    terminal_reason: str | None
    candidate_coordinates: tuple[CandidateSheetCoordinate, ...]

    def __post_init__(self) -> None:
        if self.operation_version != RAW_EXTRACTION_OPERATION_SCHEMA_VERSION:
            raise ExtractionOperationError(f"unsupported operation version: {self.operation_version}")
        if self.operation not in RAW_EXTRACTION_OPERATIONS:
            raise ExtractionOperationError(f"unsupported extraction operation: {self.operation}")
        if not _SHA256_RE.fullmatch(self.source_archive_sha256):
            raise ExtractionOperationError("source_archive_sha256 must be a lowercase SHA-256")
        if not self.archive_member_path or "\x00" in self.archive_member_path:
            raise ExtractionOperationError("archive_member_path must be explicitly recorded")
        if self.source_member_sha256 is not None and not _SHA256_RE.fullmatch(self.source_member_sha256):
            raise ExtractionOperationError("source_member_sha256 must be null or a lowercase SHA-256")
        if self.interpolation_policy != "none":
            raise ExtractionOperationError("interpolation, resizing, and resampling are forbidden")
        _validate_optional_index(self.frame_index, "frame_index")
        _validate_rectangle(self.crop_rectangle)
        _validate_sheet_fields(self.sheet_row, self.sheet_column, self.cell_width, self.cell_height)
        _validate_padding(self.padding_dimensions)
        coordinates = tuple(sorted(set(self.candidate_coordinates)))
        if coordinates != self.candidate_coordinates:
            raise ExtractionOperationError("candidate_coordinates must be sorted and unique")
        self._validate_operation_fields()

    def _validate_operation_fields(self) -> None:
        sheet_bound = self.sheet_row is not None
        has_padding = self.padding_dimensions is not None
        has_crop = self.crop_rectangle is not None
        if self.operation in RAW_OUTPUT_OPERATIONS:
            if self.decoded_rgba_sha256 is None or not _SHA256_RE.fullmatch(self.decoded_rgba_sha256):
                raise ExtractionOperationError("output operations require decoded_rgba_sha256")
            if self.terminal_reason is not None or self.candidate_coordinates:
                raise ExtractionOperationError("output operations cannot record a terminal exclusion")
        else:
            if self.decoded_rgba_sha256 is not None:
                raise ExtractionOperationError("terminal operations cannot bind output pixels")
            if not self.terminal_reason:
                raise ExtractionOperationError("terminal operations require terminal_reason")
            if any((self.frame_index is not None, has_crop, sheet_bound, has_padding)):
                raise ExtractionOperationError("terminal operations cannot apply pixel transformations")

        if self.operation == "direct_decode" and any(
            (self.frame_index is not None, has_crop, sheet_bound, has_padding)
        ):
            raise ExtractionOperationError("direct_decode requires every transformation field to be null")
        if self.operation == "frame_select":
            if self.frame_index is None or any((has_crop, sheet_bound, has_padding)):
                raise ExtractionOperationError("frame_select requires only an explicit frame_index")
        elif self.operation == "crop":
            if not has_crop or any((self.frame_index is not None, sheet_bound, has_padding)):
                raise ExtractionOperationError("crop requires only an explicit crop_rectangle")
        elif self.operation == "sheet_cell":
            if not has_crop or not sheet_bound or any((self.frame_index is not None, has_padding)):
                raise ExtractionOperationError("sheet_cell requires crop and complete sheet coordinates")
        elif self.operation == "center_pad":
            if not has_padding or any((self.frame_index is not None, has_crop, sheet_bound)):
                raise ExtractionOperationError("center_pad requires only explicit padding_dimensions")
            _validate_center_padding(self.padding_dimensions)
        elif self.operation == "crop_then_pad":
            if not has_crop or not has_padding or self.frame_index is not None:
                raise ExtractionOperationError("crop_then_pad requires explicit crop and padding")
            _validate_center_padding(self.padding_dimensions)
        elif self.operation == "exclude_ambiguous_coordinate":
            if len(self.candidate_coordinates) < 2:
                raise ExtractionOperationError("ambiguous coordinate exclusions require every candidate coordinate")
        elif self.operation in RAW_TERMINAL_OPERATIONS and self.candidate_coordinates:
            raise ExtractionOperationError("candidate coordinates are only valid for coordinate exclusions")

        if sheet_bound and self.crop_rectangle is not None:
            left, top, right, bottom = self.crop_rectangle
            if right - left != self.cell_width or bottom - top != self.cell_height:
                raise ExtractionOperationError("sheet cell dimensions must match the crop rectangle")

    def identity_payload(self) -> dict[str, Any]:
        padding = self.padding_dimensions
        return {
            "archive_member_path": self.archive_member_path,
            "candidate_coordinates": [coordinate.canonical() for coordinate in self.candidate_coordinates],
            "cell_dimensions": (
                {"height": self.cell_height, "width": self.cell_width} if self.cell_width is not None else None
            ),
            "crop_rectangle": list(self.crop_rectangle) if self.crop_rectangle is not None else None,
            "decoded_rgba_sha256": self.decoded_rgba_sha256,
            "frame_index": self.frame_index,
            "interpolation_policy": self.interpolation_policy,
            "operation": self.operation,
            "operation_version": self.operation_version,
            "padding_dimensions": (
                {"bottom": padding[3], "left": padding[0], "right": padding[2], "top": padding[1]}
                if padding is not None
                else None
            ),
            "sheet_coordinate": (
                {"column": self.sheet_column, "row": self.sheet_row} if self.sheet_row is not None else None
            ),
            "source_archive_sha256": self.source_archive_sha256,
            "source_member_sha256": self.source_member_sha256,
            "terminal_reason": self.terminal_reason,
        }

    @property
    def operation_id(self) -> str:
        return "xop_" + _sha256_bytes(canonical_json_bytes(self.identity_payload()))

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "operation_id": self.operation_id}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RawExtractionOperation:
        required = {
            "archive_member_path",
            "candidate_coordinates",
            "cell_dimensions",
            "crop_rectangle",
            "decoded_rgba_sha256",
            "frame_index",
            "interpolation_policy",
            "operation",
            "operation_id",
            "operation_version",
            "padding_dimensions",
            "sheet_coordinate",
            "source_archive_sha256",
            "source_member_sha256",
            "terminal_reason",
        }
        if set(value) != required:
            raise ExtractionOperationError(
                f"operation manifest fields differ: missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        cell = value["cell_dimensions"]
        sheet = value["sheet_coordinate"]
        padding = value["padding_dimensions"]
        coordinates_value = value["candidate_coordinates"]
        if not isinstance(coordinates_value, list):
            raise ExtractionOperationError("candidate_coordinates must be a list")
        operation = cls(
            operation_version=str(value["operation_version"]),
            operation=str(value["operation"]),
            source_archive_sha256=str(value["source_archive_sha256"]),
            archive_member_path=str(value["archive_member_path"]),
            source_member_sha256=_optional_text(value["source_member_sha256"]),
            frame_index=_optional_int(value["frame_index"]),
            crop_rectangle=_rectangle_from_json(value["crop_rectangle"]),
            sheet_row=_mapping_int(sheet, "row"),
            sheet_column=_mapping_int(sheet, "column"),
            cell_width=_mapping_int(cell, "width"),
            cell_height=_mapping_int(cell, "height"),
            padding_dimensions=_padding_from_json(padding),
            interpolation_policy=str(value["interpolation_policy"]),
            decoded_rgba_sha256=_optional_text(value["decoded_rgba_sha256"]),
            terminal_reason=_optional_text(value["terminal_reason"]),
            candidate_coordinates=tuple(sorted(_candidate_coordinate_from_json(item) for item in coordinates_value)),
        )
        if value["operation_id"] != operation.operation_id:
            raise ExtractionOperationError(f"extraction-operation identity mismatch: {value['operation_id']}")
        return operation


@dataclass(frozen=True)
class PayloadDisposition:
    payload_classification: str
    terminal_operation: str | None
    terminal_reason: str | None
    frame_count: int | None
    image_mode: str | None
    image_format: str | None
    resource_fork_detection_basis: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "image_format": self.image_format,
            "image_mode": self.image_mode,
            "payload_classification": self.payload_classification,
            "resource_fork_detection_basis": list(self.resource_fork_detection_basis),
            "terminal_operation": self.terminal_operation,
            "terminal_reason": self.terminal_reason,
        }


def extraction_operation_json_schema() -> dict[str, Any]:
    """Return the frozen JSON Schema for explicit extraction operations."""

    nullable_sha = {"anyOf": [{"type": "null"}, {"pattern": "^[0-9a-f]{64}$", "type": "string"}]}
    nullable_nonnegative = {"anyOf": [{"type": "null"}, {"minimum": 0, "type": "integer"}]}
    coordinate = {
        "additionalProperties": False,
        "properties": {"column": {"minimum": 0, "type": "integer"}, "row": {"minimum": 0, "type": "integer"}},
        "required": ["column", "row"],
        "type": "object",
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": RAW_EXTRACTION_OPERATION_SCHEMA_VERSION,
        "additionalProperties": False,
        "properties": {
            "archive_member_path": {"minLength": 1, "type": "string"},
            "candidate_coordinates": {"items": coordinate, "type": "array", "uniqueItems": True},
            "cell_dimensions": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "additionalProperties": False,
                        "properties": {
                            "height": {"minimum": 1, "type": "integer"},
                            "width": {"minimum": 1, "type": "integer"},
                        },
                        "required": ["height", "width"],
                        "type": "object",
                    },
                ]
            },
            "crop_rectangle": {
                "anyOf": [
                    {"type": "null"},
                    {"items": {"minimum": 0, "type": "integer"}, "maxItems": 4, "minItems": 4, "type": "array"},
                ]
            },
            "decoded_rgba_sha256": nullable_sha,
            "frame_index": nullable_nonnegative,
            "interpolation_policy": {"const": "none"},
            "operation": {"enum": list(RAW_EXTRACTION_OPERATIONS)},
            "operation_id": {"pattern": "^xop_[0-9a-f]{64}$", "type": "string"},
            "operation_version": {"const": RAW_EXTRACTION_OPERATION_SCHEMA_VERSION},
            "padding_dimensions": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "additionalProperties": False,
                        "properties": {
                            name: {"minimum": 0, "type": "integer"} for name in ("bottom", "left", "right", "top")
                        },
                        "required": ["bottom", "left", "right", "top"],
                        "type": "object",
                    },
                ]
            },
            "sheet_coordinate": {"anyOf": [{"type": "null"}, coordinate]},
            "source_archive_sha256": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
            "source_member_sha256": nullable_sha,
            "terminal_reason": {"anyOf": [{"type": "null"}, {"minLength": 1, "type": "string"}]},
        },
        "required": [
            "archive_member_path",
            "candidate_coordinates",
            "cell_dimensions",
            "crop_rectangle",
            "decoded_rgba_sha256",
            "frame_index",
            "interpolation_policy",
            "operation",
            "operation_id",
            "operation_version",
            "padding_dimensions",
            "sheet_coordinate",
            "source_archive_sha256",
            "source_member_sha256",
            "terminal_reason",
        ],
        "title": "Sprite Lab explicit raw extraction operation",
        "type": "object",
    }


def classify_image_payload(
    member_path: str,
    payload: bytes | None,
    *,
    expected_size: int | None,
    member_bytes_complete: bool,
) -> PayloadDisposition:
    """Classify a payload without repair, frame choice, or resource-fork parsing."""

    prefix = payload[:4] if payload is not None else b""
    basis = appledouble_detection_basis(member_path, prefix)
    if basis:
        return PayloadDisposition(
            "appledouble_resource_fork",
            "reject_resource_fork",
            "metadata_resource_fork_not_sprite",
            None,
            None,
            None,
            basis,
        )
    if payload is None or not member_bytes_complete or (expected_size is not None and len(payload) != expected_size):
        return PayloadDisposition(
            "truncated_archive_member",
            "reject_unreadable",
            "truncated_archive_member",
            None,
            None,
            None,
            (),
        )
    suffix = PurePosixPath(member_path).suffix.casefold()
    if suffix not in _IMAGE_SUFFIXES:
        return PayloadDisposition("non_image_payload", "reject_unreadable", "non_image_payload", None, None, None, ())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                frame_count = int(getattr(image, "n_frames", 1))
                mode = str(image.mode)
                image_format = str(image.format or "unknown")
                if frame_count != 1:
                    return PayloadDisposition(
                        "multi_frame_image",
                        "exclude_ambiguous",
                        "missing_explicit_frame_index",
                        frame_count,
                        mode,
                        image_format,
                        (),
                    )
                if mode not in SUPPORTED_RAW_IMAGE_MODES:
                    return PayloadDisposition(
                        "unsupported_image_mode",
                        "reject_unreadable",
                        "unsupported_image_mode",
                        frame_count,
                        mode,
                        image_format,
                        (),
                    )
                image.load()
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return PayloadDisposition(
            "corrupt_image", "reject_unreadable", "image_decompression_bomb", None, None, None, ()
        )
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return PayloadDisposition("corrupt_image", "reject_unreadable", "corrupt_image", None, None, None, ())
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
        return PayloadDisposition(
            "corrupt_image", "reject_unreadable", "invalid_decoded_rgba_geometry", 1, mode, image_format, ()
        )
    return PayloadDisposition("decodable_image", None, None, 1, mode, image_format, ())


def execute_extraction_operation(payload: bytes, operation: RawExtractionOperation) -> np.ndarray:
    """Reproduce one output operation exactly and verify its frozen RGBA hash."""

    if operation.operation not in RAW_OUTPUT_OPERATIONS:
        raise ExtractionOperationError(f"terminal extraction operation has no output: {operation.operation}")
    if is_appledouble_record(operation.archive_member_path, payload[:4]):
        raise ExtractionOperationError("resource-fork metadata must never be decoded as an image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                frame_count = int(getattr(image, "n_frames", 1))
                if operation.operation == "frame_select":
                    assert operation.frame_index is not None
                    if operation.frame_index >= frame_count:
                        raise ExtractionOperationError(
                            f"frame_index {operation.frame_index} is outside frame_count {frame_count}"
                        )
                    image.seek(operation.frame_index)
                elif frame_count != 1:
                    raise ExtractionOperationError("multi-frame image requires an explicit frame_select operation")
                if image.mode not in SUPPORTED_RAW_IMAGE_MODES:
                    raise ExtractionOperationError(f"unsupported image mode: {image.mode}")
                image.load()
                result = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    except ExtractionOperationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ExtractionOperationError("image decompression bomb") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ExtractionOperationError("image payload is unreadable; silent repair is forbidden") from exc

    if operation.crop_rectangle is not None:
        left, top, right, bottom = operation.crop_rectangle
        height, width = result.shape[:2]
        if right > width or bottom > height:
            raise ExtractionOperationError(
                f"crop rectangle is outside decoded bounds: {operation.crop_rectangle} for {width}x{height}"
            )
        result = result[top:bottom, left:right].copy()
    if operation.padding_dimensions is not None:
        left, top, right, bottom = operation.padding_dimensions
        height, width = result.shape[:2]
        padded = np.zeros((height + top + bottom, width + left + right, 4), dtype=np.uint8)
        padded[top : top + height, left : left + width] = result
        result = padded
    result = np.ascontiguousarray(result, dtype=np.uint8)
    observed = decoded_rgba_sha256(result)
    if observed != operation.decoded_rgba_sha256:
        raise ExtractionOperationError(
            f"decoded RGBA hash mismatch for {operation.operation_id}: "
            f"expected {operation.decoded_rgba_sha256}, observed {observed}"
        )
    return result


def verify_extraction_operation_manifest(rows: Iterable[Mapping[str, Any]]) -> tuple[RawExtractionOperation, ...]:
    """Verify schema, identities, ordering independence, and duplicate IDs."""

    operations = tuple(RawExtractionOperation.from_mapping(row) for row in rows)
    operation_ids = [operation.operation_id for operation in operations]
    if len(operation_ids) != len(set(operation_ids)):
        raise ExtractionOperationError("duplicate extraction-operation identity")
    return tuple(sorted(operations, key=lambda operation: operation.operation_id))


def operation_manifest_bytes(operations: Iterable[RawExtractionOperation]) -> bytes:
    """Return deterministic canonical JSONL bytes sorted by operation identity."""

    materialized = tuple(sorted(operations, key=lambda operation: operation.operation_id))
    if len(materialized) != len({operation.operation_id for operation in materialized}):
        raise ExtractionOperationError("duplicate extraction-operation identity")
    return b"".join(canonical_json_bytes(operation.to_dict()) + b"\n" for operation in materialized)


def make_extraction_relation_id(relation_type: str, operation_ids: Sequence[str]) -> str:
    """Make a path- and timestamp-independent relation identity."""

    members = sorted(set(operation_ids))
    if not relation_type or len(members) < 2 or any(not re.fullmatch(r"xop_[0-9a-f]{64}", item) for item in members):
        raise ExtractionOperationError("relation identity requires a type and at least two operation IDs")
    return "xrel_" + _sha256_bytes(
        canonical_json_bytes(
            {
                "member_operation_ids": members,
                "relation_type": relation_type,
                "relation_version": "sprite_lab_extraction_relation_v1",
            }
        )
    )


def _validate_optional_index(value: int | None, name: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ExtractionOperationError(f"{name} must be null or a non-negative integer")


def _validate_rectangle(value: tuple[int, int, int, int] | None) -> None:
    if value is None:
        return
    if len(value) != 4 or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ExtractionOperationError("crop_rectangle must be null or exactly four integers")
    left, top, right, bottom = value
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ExtractionOperationError(f"invalid crop_rectangle: {value}")


def _validate_sheet_fields(row: int | None, column: int | None, width: int | None, height: int | None) -> None:
    values = (row, column, width, height)
    if all(value is None for value in values):
        return
    if any(value is None or not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ExtractionOperationError("sheet row, column, and cell dimensions must be all null or all integers")
    assert row is not None and column is not None and width is not None and height is not None
    if row < 0 or column < 0 or width <= 0 or height <= 0:
        raise ExtractionOperationError("sheet coordinates must be non-negative and cell dimensions positive")


def _validate_padding(value: tuple[int, int, int, int] | None) -> None:
    if value is None:
        return
    if len(value) != 4 or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise ExtractionOperationError("padding_dimensions must be null or four non-negative integers")
    if not any(value):
        raise ExtractionOperationError("zero padding must be represented by explicit null")


def _validate_center_padding(value: tuple[int, int, int, int] | None) -> None:
    assert value is not None
    left, top, right, bottom = value
    if right not in {left, left + 1} or bottom not in {top, top + 1}:
        raise ExtractionOperationError("center padding must place any odd pixel on the right or bottom")


def _rectangle_from_json(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ExtractionOperationError("crop_rectangle must be null or a four-item list")
    return tuple(value)  # type: ignore[return-value]


def _padding_from_json(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"bottom", "left", "right", "top"}:
        raise ExtractionOperationError("padding_dimensions has invalid fields")
    return tuple(_required_mapping_int(value, name) for name in ("left", "top", "right", "bottom"))


def _mapping_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    expected = {"row", "column"} if name in {"row", "column"} else {"width", "height"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ExtractionOperationError(f"{name} object has invalid fields")
    return _required_mapping_int(value, name)


def _required_mapping_int(value: Any, name: str) -> int:
    if not isinstance(value, Mapping):
        raise ExtractionOperationError(f"{name} must be read from an object")
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ExtractionOperationError(f"{name} must be an integer")
    return item


def _candidate_coordinate_from_json(value: Any) -> CandidateSheetCoordinate:
    if not isinstance(value, Mapping) or set(value) != {"column", "row"}:
        raise ExtractionOperationError("candidate coordinate has invalid fields")
    return CandidateSheetCoordinate(_required_mapping_int(value, "row"), _required_mapping_int(value, "column"))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExtractionOperationError("optional integer field has invalid type")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExtractionOperationError("optional text field has invalid type")
    return value


@dataclass(frozen=True)
class RawForensicInventory:
    """Deterministic evidence collected from one explicit source root."""

    records: tuple[Mapping[str, Any], ...]
    artifacts: tuple[Mapping[str, Any], ...]
    unresolved_source_bindings: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    schema_version: str = RAW_FORENSIC_INVENTORY_SCHEMA_VERSION

    def inventory_jsonl_bytes(self) -> bytes:
        """Return canonical JSONL bytes for every forensic record."""

        return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in self.records)

    def archive_hashes_bytes(self) -> bytes:
        """Return the canonical archive/path binding document."""

        document = {
            "artifacts": [dict(row) for row in self.artifacts],
            "schema_version": RAW_FORENSIC_ARCHIVE_SCHEMA_VERSION,
            "summary": dict(self.summary),
            "unresolved_source_bindings": [dict(row) for row in self.unresolved_source_bindings],
        }
        return canonical_json_bytes(document) + b"\n"

    def report_text(self) -> str:
        """Render a deterministic, human-readable fail-closed report."""

        summary = self.summary
        issue_counts = summary.get("blocking_issue_counts", {})
        lines = [
            "# Raw source forensic inventory",
            "",
            f"Schema: `{RAW_FORENSIC_REPORT_VERSION}`",
            "",
            "This is a read-only inventory. A current hash first observed during this audit is not treated as a "
            "historically recorded hash. Missing provenance and license evidence remains blocking.",
            "",
            "## Counts",
            "",
        ]
        ordered_counts = (
            "source_manifest_count",
            "source_binding_count",
            "resolved_source_binding_count",
            "unresolved_source_binding_count",
            "historical_hash_present_source_binding_count",
            "historical_hash_match_source_binding_count",
            "unknown_license_source_binding_count",
            "missing_provenance_source_binding_count",
            "physical_path_count",
            "unique_artifact_count",
            "zip_archive_count",
            "standalone_image_artifact_count",
            "archive_file_member_count",
            "archive_image_member_count",
            "decoded_image_record_count",
            "unique_decoded_rgba_count",
            "accepted_record_count",
            "quarantined_record_count",
            "rejected_record_count",
            "acquisition_orphan_artifact_count",
            "appledouble_rejected_count",
            "unsafe_member_count",
            "zip_crc_failure_count",
        )
        for key in ordered_counts:
            lines.append(f"- {key}: {int(summary.get(key, 0))}")
        lines.extend(["", "## Blocking issues", ""])
        if issue_counts:
            for issue, count in sorted(issue_counts.items()):
                lines.append(f"- `{issue}`: {int(count)}")
        else:
            lines.append("- None observed.")
        lines.extend(
            [
                "",
                "## Gate status",
                "",
                f"- raw_source_gate_passed: `{str(bool(summary.get('raw_source_gate_passed'))).lower()}`",
                "- Unknown or incomplete evidence was recorded; it was not guessed, repaired, or promoted.",
                "- No modification times were read as provenance.",
                "",
            ]
        )
        return "\n".join(lines)


@dataclass
class _ManifestBinding:
    manifest_path: str
    manifest_sha256: str
    line_number: int
    source_row: dict[str, Any]
    source_row_sha256: str
    issues: list[str]
    historical_hash: str | None
    historical_hash_raw: str
    artifact_sha256: str | None = None
    resolution_method: str = "unresolved"


@dataclass(frozen=True)
class _PhysicalArtifact:
    sha256: str
    size_bytes: int
    paths: tuple[str, ...]
    canonical_path: Path


def audit_raw_source_inventory(source_root: str | Path) -> RawForensicInventory:
    """Inventory all original physical downloads below ``source_root``.

    The discovery roots are intentionally fixed and versioned: harvest source
    manifests/download caches, the acquisition-diversity download cache, and
    the original Itemicon PNG.  Manifest-declared local archive files below the
    same root are also included.  Failures become records and issue counts;
    they are not silently repaired.
    """

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw source root does not exist: {root}")

    manifest_paths = sorted((root / "harvest_runs").glob("*/sources.jsonl"), key=_path_sort_key)
    bindings = _read_manifest_bindings(root, manifest_paths)
    physical_paths = _discover_physical_paths(root, bindings)
    artifacts = _group_physical_artifacts(root, physical_paths)
    path_to_hash = {path: artifact.sha256 for artifact in artifacts for path in artifact.paths}
    artifacts_by_hash = {artifact.sha256: artifact for artifact in artifacts}
    _resolve_bindings(root, bindings, artifacts, path_to_hash)

    binding_rows_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved: list[dict[str, Any]] = []
    for binding in bindings:
        row = _binding_to_dict(binding)
        if binding.artifact_sha256 is None:
            unresolved.append(row)
        else:
            binding_rows_by_hash[binding.artifact_sha256].append(row)

    records: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for digest in sorted(artifacts_by_hash):
        artifact = artifacts_by_hash[digest]
        source_bindings = sorted(binding_rows_by_hash.get(digest, []), key=_binding_sort_key)
        forensic_rows, artifact_row = _inspect_artifact(root, artifact, source_bindings)
        records.extend(forensic_rows)
        artifact_rows.append(artifact_row)

    for row in sorted(unresolved, key=_binding_sort_key):
        records.append(_unresolved_record(row))

    records = sorted(records, key=_record_sort_key)
    artifact_rows = sorted(artifact_rows, key=lambda row: str(row["current_observed_archive_sha256"]))
    unresolved = sorted(unresolved, key=_binding_sort_key)
    summary = _summarize(
        records=records,
        artifacts=artifact_rows,
        bindings=bindings,
        manifest_count=len(manifest_paths),
        physical_path_count=len(physical_paths),
    )
    return RawForensicInventory(
        records=tuple(records),
        artifacts=tuple(artifact_rows),
        unresolved_source_bindings=tuple(unresolved),
        summary=summary,
    )


def write_raw_forensic_inventory(
    inventory: RawForensicInventory,
    evidence_root: str | Path,
) -> dict[str, Path]:
    """Write the three Phase-1 artifacts without overwriting any file."""

    root = Path(evidence_root)
    targets = {
        "raw_source_inventory": root / "raw_source_inventory.jsonl",
        "source_archive_hashes": root / "source_archive_hashes.json",
        "raw_source_inventory_report": root / "raw_source_inventory_report.md",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in sorted(existing, key=_path_sort_key))
        raise FileExistsError(f"refusing to overwrite raw forensic artifact(s): {joined}")
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "raw_source_inventory": inventory.inventory_jsonl_bytes(),
        "source_archive_hashes": inventory.archive_hashes_bytes(),
        "raw_source_inventory_report": inventory.report_text().encode("utf-8"),
    }
    created: list[Path] = []
    try:
        for key in ("raw_source_inventory", "source_archive_hashes", "raw_source_inventory_report"):
            path = targets[key]
            with path.open("xb") as handle:
                handle.write(payloads[key])
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return targets


def _read_manifest_bindings(root: Path, manifests: Iterable[Path]) -> list[_ManifestBinding]:
    bindings: list[_ManifestBinding] = []
    for manifest in manifests:
        manifest_rel = _relative_path(root, manifest)
        manifest_hash = _file_sha256(manifest)
        with manifest.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    value = {"_unparsed_source_row_sha256": _sha256_bytes(raw_line.encode("utf-8"))}
                    issues = ["invalid_source_manifest_json"]
                else:
                    issues = []
                    if not isinstance(value, dict):
                        value = {"_non_object_source_row": value}
                        issues.append("source_manifest_row_not_object")
                row = _json_safe(value)
                issues.extend(_source_row_issues(row))
                historical_raw = str(row.get("download_sha256") or row.get("sha256") or "").strip().lower()
                historical_hash = historical_raw if _SHA256_RE.fullmatch(historical_raw) else None
                if historical_raw and historical_hash is None:
                    issues.append("invalid_historical_archive_sha256")
                bindings.append(
                    _ManifestBinding(
                        manifest_path=manifest_rel,
                        manifest_sha256=manifest_hash,
                        line_number=line_number,
                        source_row=row,
                        source_row_sha256=_sha256_bytes(canonical_json_bytes(row)),
                        issues=sorted(set(issues)),
                        historical_hash=historical_hash,
                        historical_hash_raw=historical_raw,
                    )
                )
    return bindings


def _source_row_issues(row: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if not _text(row.get("source_id")):
        issues.append("missing_source_id")
    if not _text(row.get("source_url")) and not _text(row.get("download_url")):
        issues.append("missing_acquisition_url")
    if not _text(row.get("author") or row.get("creator") or row.get("publisher")):
        issues.append("missing_creator_or_publisher")
    if not _text(row.get("original_filename")):
        issues.append("missing_original_filename")
    if not _text(row.get("download_sha256") or row.get("sha256")):
        issues.append("missing_historical_archive_sha256")
    license_record = row.get("license")
    if not isinstance(license_record, Mapping):
        issues.append("missing_license")
    else:
        license_name = normalize_license_name(_text(license_record.get("license")))
        if license_name == "unknown":
            issues.append("unknown_license")
        elif license_name not in TRAINING_ALLOWED_LICENSES:
            issues.append("license_not_training_allowed")
        if not _text(license_record.get("license_url")):
            issues.append("missing_license_url")
        if not bool(license_record.get("user_confirmed", False)):
            issues.append("license_not_user_confirmed")
    return issues


def _discover_physical_paths(root: Path, bindings: Iterable[_ManifestBinding]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    scan_roots = (
        root / "harvest_runs",
        root / "experiments" / "acquisition_diversity_wave_v1" / "downloads",
    )
    harvest_root = scan_roots[0]
    if harvest_root.is_dir():
        for downloads in sorted(harvest_root.glob("*/downloads"), key=_path_sort_key):
            if downloads.is_dir():
                paths.update(_safe_files_below(root, downloads))
    acquisition = scan_roots[1]
    if acquisition.is_dir():
        paths.update(_safe_files_below(root, acquisition))

    itemicon = root / "data_sources" / "cc_by_itemiconpack32" / "itemiconpack32.png"
    if itemicon.is_file() and _is_below(root, itemicon):
        paths.add(itemicon.resolve())

    for binding in bindings:
        local = _manifest_local_path(root, binding.source_row)
        if local is not None and local.is_file():
            paths.add(local)
    return tuple(sorted(paths, key=_path_sort_key))


def _safe_files_below(root: Path, directory: Path) -> set[Path]:
    files: set[Path] = set()
    for candidate in directory.rglob("*"):
        if candidate.is_file():
            resolved = candidate.resolve()
            if _is_below(root, resolved):
                files.add(resolved)
    return files


def _group_physical_artifacts(root: Path, paths: Iterable[Path]) -> tuple[_PhysicalArtifact, ...]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    sizes: dict[str, int] = {}
    for path in paths:
        digest = _file_sha256(path)
        size = path.stat().st_size
        if digest in sizes and sizes[digest] != size:
            raise RuntimeError(f"SHA-256 size collision while inventorying {path}")
        sizes[digest] = size
        grouped[digest].append(path)
    artifacts: list[_PhysicalArtifact] = []
    for digest, group_paths in grouped.items():
        ordered_paths = sorted(group_paths, key=_path_sort_key)
        artifacts.append(
            _PhysicalArtifact(
                sha256=digest,
                size_bytes=sizes[digest],
                paths=tuple(_relative_path(root, path) for path in ordered_paths),
                canonical_path=ordered_paths[0],
            )
        )
    return tuple(sorted(artifacts, key=lambda item: item.sha256))


def _resolve_bindings(
    root: Path,
    bindings: Iterable[_ManifestBinding],
    artifacts: Iterable[_PhysicalArtifact],
    path_to_hash: Mapping[str, str],
) -> None:
    artifacts_by_hash = {artifact.sha256: artifact for artifact in artifacts}
    hashes_by_basename: dict[str, set[str]] = defaultdict(set)
    for artifact in artifacts:
        for path in artifact.paths:
            hashes_by_basename[PurePosixPath(path).name.casefold()].add(artifact.sha256)

    for binding in bindings:
        local = _manifest_local_path(root, binding.source_row)
        local_hash: str | None = None
        if local is not None and local.is_file():
            local_rel = _relative_path(root, local)
            local_hash = path_to_hash.get(local_rel)

        if binding.historical_hash is not None and binding.historical_hash in artifacts_by_hash:
            binding.artifact_sha256 = binding.historical_hash
            binding.resolution_method = "historical_sha256_match"
            if local_hash is not None and local_hash != binding.historical_hash:
                binding.issues.append("declared_path_hash_mismatch")
            binding.issues = sorted(set(binding.issues))
            continue

        if local_hash is not None:
            binding.artifact_sha256 = local_hash
            binding.resolution_method = "manifest_local_archive_path"
            if binding.historical_hash is not None and local_hash != binding.historical_hash:
                binding.issues.append("changed_archive_hash")
            binding.issues = sorted(set(binding.issues))
            continue

        candidate_names = _candidate_basenames(binding.source_row)
        run_name = PurePosixPath(binding.manifest_path).parent.name
        run_prefix = f"harvest_runs/{run_name}/downloads/"
        run_paths = {path: digest for path, digest in path_to_hash.items() if path.startswith(run_prefix)}
        run_named_hashes = {
            digest
            for path, digest in run_paths.items()
            if PurePosixPath(path).name.casefold() in {name.casefold() for name in candidate_names}
        }
        if len(run_named_hashes) == 1:
            binding.artifact_sha256 = next(iter(run_named_hashes))
            binding.resolution_method = "run_download_original_basename_match"
            if binding.historical_hash is not None and binding.artifact_sha256 != binding.historical_hash:
                binding.issues.append("changed_archive_hash")
            binding.issues = sorted(set(binding.issues))
            continue
        run_hashes = set(run_paths.values())
        if len(run_hashes) == 1:
            binding.artifact_sha256 = next(iter(run_hashes))
            binding.resolution_method = "unique_run_download_artifact"
            if binding.historical_hash is not None and binding.artifact_sha256 != binding.historical_hash:
                binding.issues.append("changed_archive_hash")
            binding.issues = sorted(set(binding.issues))
            continue
        if len(run_hashes) > 1 and not run_named_hashes:
            binding.issues.append("ambiguous_run_download_binding")

        candidate_hashes = {
            digest for name in candidate_names for digest in hashes_by_basename.get(name.casefold(), set())
        }
        if len(candidate_hashes) == 1:
            binding.artifact_sha256 = next(iter(candidate_hashes))
            binding.resolution_method = "unique_original_basename_match"
            if binding.historical_hash is not None and binding.artifact_sha256 != binding.historical_hash:
                binding.issues.append("changed_archive_hash")
        elif len(candidate_hashes) > 1:
            binding.issues.append("ambiguous_original_download_binding")
        elif binding.historical_hash is not None:
            binding.issues.append("missing_historical_archive_bytes")
        else:
            binding.issues.append("missing_original_download")
        binding.issues = sorted(set(binding.issues))


def _binding_to_dict(binding: _ManifestBinding) -> dict[str, Any]:
    row = binding.source_row
    license_record = row.get("license") if isinstance(row.get("license"), Mapping) else {}
    license_name = normalize_license_name(_text(license_record.get("license"))) if license_record else "unknown"
    creator = _text(row.get("author") or row.get("creator") or row.get("publisher"))
    source_url = _text(row.get("source_url"))
    download_url = _text(row.get("download_url"))
    return {
        "acquisition_run": PurePosixPath(binding.manifest_path).parent.name,
        "creator_or_publisher": creator or None,
        "current_observed_archive_sha256": binding.artifact_sha256,
        "distribution_platform": _distribution_platform(source_url, download_url),
        "download_url": download_url or None,
        "historically_recorded_archive_sha256": binding.historical_hash,
        "historically_recorded_archive_sha256_raw": binding.historical_hash_raw or None,
        "license": _json_safe(dict(license_record)),
        "license_normalized": license_name,
        "manifest_line_number": binding.line_number,
        "manifest_path": binding.manifest_path,
        "manifest_sha256": binding.manifest_sha256,
        "original_archive_filename": _text(row.get("original_filename")) or None,
        "pack": _text(row.get("source_name")) or None,
        "provenance_issues": list(binding.issues),
        "provenance_status": "complete" if not binding.issues else "blocked",
        "resolution_method": binding.resolution_method,
        "source_id": _text(row.get("source_id")) or None,
        "source_record": _json_safe(row),
        "source_row_sha256": binding.source_row_sha256,
        "source_type": _text(row.get("source_type")) or None,
        "source_url": source_url or None,
    }


def _inspect_artifact(
    root: Path,
    artifact: _PhysicalArtifact,
    bindings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del root  # All persisted paths were made root-relative during discovery.
    common_issues = _artifact_provenance_issues(artifact, bindings)
    historical_hashes = sorted(
        {
            str(row["historically_recorded_archive_sha256"])
            for row in bindings
            if row.get("historically_recorded_archive_sha256")
        }
    )
    archive_status = _archive_hash_status(artifact.sha256, historical_hashes)
    if zipfile.is_zipfile(artifact.canonical_path):
        return _inspect_zip(artifact, bindings, common_issues, historical_hashes, archive_status)
    return _inspect_standalone(artifact, bindings, common_issues, historical_hashes, archive_status)


def _inspect_zip(
    artifact: _PhysicalArtifact,
    bindings: list[dict[str, Any]],
    common_issues: list[str],
    historical_hashes: list[str],
    archive_status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    archive_issues: list[str] = []
    member_file_count = 0
    image_member_count = 0
    unsafe_count = 0
    casefold_collision_count = 0
    crc_ok = True
    try:
        with zipfile.ZipFile(artifact.canonical_path) as archive:
            infos = list(archive.infolist())
            names = [info.filename for info in infos if not info.is_dir()]
            folded = Counter(name.replace("\\", "/").casefold() for name in names)
            collision_names = {name for name, count in folded.items() if count > 1}
            casefold_collision_count = len(collision_names)
            try:
                bad_crc_member = archive.testzip()
            except (RuntimeError, zipfile.BadZipFile, OSError):
                bad_crc_member = "<archive-read-failure>"
            if bad_crc_member is not None:
                crc_ok = False
                archive_issues.append("zip_crc_failure")

            ordered = sorted(enumerate(infos), key=lambda pair: (pair[1].filename, pair[0]))
            for member_index, info in ordered:
                if info.is_dir():
                    continue
                member_file_count += 1
                member_path = info.filename.replace("\\", "/")
                if PurePosixPath(member_path).suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                image_member_count += 1
                content_issues: list[str] = []
                unsafe = _unsafe_member_path(member_path)
                if unsafe:
                    content_issues.append("unsafe_archive_member_path")
                    unsafe_count += 1
                if member_path.casefold() in collision_names:
                    content_issues.append("archive_member_casefold_collision")
                if info.flag_bits & 0x1:
                    content_issues.append("encrypted_archive_member")
                if _zipinfo_is_symlink(info):
                    content_issues.append("archive_symlink_member")
                payload: bytes | None = None
                try:
                    payload = archive.read(info)
                except (RuntimeError, zipfile.BadZipFile, OSError):
                    content_issues.append("unreadable_archive_member_bytes")
                appledouble = is_appledouble_record(member_path, payload[:4] if payload is not None else b"")
                if appledouble:
                    content_issues.append("appledouble_resource_fork")
                    # Retain the legacy issue for compatibility while the
                    # payload_classification field distinguishes metadata
                    # from a genuinely corrupt image.
                    content_issues.append("unreadable_image_payload")
                if not crc_ok and info.filename == bad_crc_member:
                    content_issues.append("member_crc_failure")
                if payload is not None:
                    byte_hash = _sha256_bytes(payload)
                else:
                    byte_hash = None
                decoded = None if appledouble or unsafe else _decode_image(payload)
                if decoded is not None and decoded.get("error"):
                    content_issues.append(str(decoded["error"]))
                records.append(
                    _image_record(
                        artifact=artifact,
                        bindings=bindings,
                        common_issues=common_issues,
                        archive_issues=archive_issues,
                        historical_hashes=historical_hashes,
                        archive_status=archive_status,
                        record_type="archive_member_image",
                        member_path=member_path,
                        member_index=member_index,
                        original_filename=PurePosixPath(member_path).name,
                        original_byte_sha256=byte_hash,
                        original_byte_size=len(payload) if payload is not None else int(info.file_size),
                        decoded=decoded,
                        content_issues=content_issues,
                        member_crc32=f"{info.CRC:08x}",
                    )
                )
    except (zipfile.BadZipFile, OSError, RuntimeError):
        archive_issues.append("unreadable_zip_archive")

    artifact_row = _artifact_row(
        artifact=artifact,
        bindings=bindings,
        common_issues=common_issues,
        historical_hashes=historical_hashes,
        archive_status=archive_status,
        artifact_kind="zip_archive",
        archive_issues=archive_issues,
        archive_member_count=member_file_count,
        archive_image_member_count=image_member_count,
        archive_crc_ok=crc_ok,
        unsafe_member_count=unsafe_count,
        casefold_collision_count=casefold_collision_count,
    )
    return records, artifact_row


def _inspect_standalone(
    artifact: _PhysicalArtifact,
    bindings: list[dict[str, Any]],
    common_issues: list[str],
    historical_hashes: list[str],
    archive_status: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = artifact.canonical_path.suffix.lower()
    archive_issues: list[str] = []
    records: list[dict[str, Any]] = []
    kind = "standalone_image" if suffix in _IMAGE_SUFFIXES else "unsupported_download_artifact"
    if kind == "standalone_image":
        payload = artifact.canonical_path.read_bytes()
        decoded = _decode_image(payload)
        content_issues = [str(decoded["error"])] if decoded is not None and decoded.get("error") else []
        records.append(
            _image_record(
                artifact=artifact,
                bindings=bindings,
                common_issues=common_issues,
                archive_issues=archive_issues,
                historical_hashes=historical_hashes,
                archive_status=archive_status,
                record_type="standalone_image",
                member_path=None,
                member_index=None,
                original_filename=artifact.canonical_path.name,
                original_byte_sha256=artifact.sha256,
                original_byte_size=artifact.size_bytes,
                decoded=decoded,
                content_issues=content_issues,
                member_crc32=None,
            )
        )
    else:
        archive_issues.append("unsupported_download_artifact")
        records.append(
            _nonimage_artifact_record(
                artifact,
                bindings,
                common_issues,
                historical_hashes,
                archive_status,
            )
        )
    artifact_row = _artifact_row(
        artifact=artifact,
        bindings=bindings,
        common_issues=common_issues,
        historical_hashes=historical_hashes,
        archive_status=archive_status,
        artifact_kind=kind,
        archive_issues=archive_issues,
        archive_member_count=0,
        archive_image_member_count=1 if kind == "standalone_image" else 0,
        archive_crc_ok=None,
        unsafe_member_count=0,
        casefold_collision_count=0,
    )
    return records, artifact_row


def _image_record(
    *,
    artifact: _PhysicalArtifact,
    bindings: list[dict[str, Any]],
    common_issues: list[str],
    archive_issues: list[str],
    historical_hashes: list[str],
    archive_status: str,
    record_type: str,
    member_path: str | None,
    member_index: int | None,
    original_filename: str,
    original_byte_sha256: str | None,
    original_byte_size: int,
    decoded: dict[str, Any] | None,
    content_issues: list[str],
    member_crc32: str | None,
) -> dict[str, Any]:
    decoded_ok = decoded is not None and not decoded.get("error")
    width = int(decoded["width"]) if decoded_ok else None
    height = int(decoded["height"]) if decoded_ok else None
    decoded_hash = str(decoded["decoded_rgba_sha256"]) if decoded_ok else None
    all_content_issues = sorted(set(archive_issues + content_issues))
    all_issues = sorted(set(common_issues + all_content_issues))
    payload_classification, terminal_disposition, terminal_reason = _payload_terminal_fields(
        decoded_ok=decoded_ok,
        content_issues=all_content_issues,
        record_type=record_type,
    )
    fatal_content = bool(all_content_issues) or not decoded_ok
    decision = "reject" if fatal_content else ("quarantine" if common_issues else "accept")
    metadata = _binding_metadata(bindings)
    identity = {
        "archive_sha256": artifact.sha256,
        "member_index": member_index,
        "member_path": member_path,
        "original_byte_sha256": original_byte_sha256,
    }
    return {
        "acquisition_runs": metadata["acquisition_runs"],
        "archive_hash_status": archive_status,
        "archive_member_path": member_path,
        "audit_issues": all_issues,
        "creator_or_publishers": metadata["creators"],
        "crop_coordinates": [0, 0, width, height] if decoded_ok else None,
        "decoded_image_sha256": decoded_hash,
        "decoded_rgba_hash_schema": "decoded_rgba_v1",
        "distribution_platforms": metadata["platforms"],
        "download_urls": metadata["download_urls"],
        "extraction_operation": "identity_decode_to_rgba",
        "forensic_record_id": "fr_" + _sha256_bytes(canonical_json_bytes(identity)),
        "frame_count": decoded.get("frame_count") if decoded is not None else None,
        "height": height,
        "image_format": decoded.get("format") if decoded is not None else None,
        "image_mode": decoded.get("mode") if decoded is not None else None,
        "inclusion_decision": decision,
        "interpolation_policy": "none",
        "licenses": metadata["licenses"],
        "member_crc32": member_crc32,
        "member_index": member_index,
        "original_archive_path": artifact.paths[0],
        "original_archive_paths": list(artifact.paths),
        "original_archive_sha256": artifact.sha256,
        "original_archive_size_bytes": artifact.size_bytes,
        "original_byte_sha256": original_byte_sha256,
        "original_byte_size": original_byte_size,
        "original_filename": original_filename,
        "output_decoded_rgba_sha256": decoded_hash,
        "packs": metadata["packs"],
        "padding_operation": "none",
        "payload_classification": payload_classification,
        "provenance_issues": common_issues,
        "provenance_status": "complete" if not common_issues else "blocked",
        "record_type": record_type,
        "resource_fork_detection_basis": (
            list(appledouble_detection_basis(member_path or original_filename))
            if "appledouble_resource_fork" in all_content_issues
            else []
        ),
        "schema_version": RAW_FORENSIC_INVENTORY_SCHEMA_VERSION,
        "source_bindings": bindings,
        "source_urls": metadata["source_urls"],
        "historically_recorded_archive_sha256s": historical_hashes,
        "terminal_disposition": terminal_disposition,
        "terminal_reason": terminal_reason,
        "width": width,
    }


def _payload_terminal_fields(
    *, decoded_ok: bool, content_issues: Sequence[str], record_type: str
) -> tuple[str, str | None, str | None]:
    issues = set(content_issues)
    if "appledouble_resource_fork" in issues:
        return "appledouble_resource_fork", "reject_resource_fork", "metadata_resource_fork_not_sprite"
    if "multi_frame_image_requires_explicit_operation" in issues:
        return "multi_frame_image", "exclude_ambiguous", "missing_explicit_frame_index"
    if issues.intersection({"unreadable_archive_member_bytes", "member_crc_failure", "zip_crc_failure"}):
        return "truncated_archive_member", "reject_unreadable", "truncated_archive_member"
    if "unsupported_image_mode" in issues:
        return "unsupported_image_mode", "reject_unreadable", "unsupported_image_mode"
    if "unreadable_image_payload" in issues or "invalid_decoded_rgba_geometry" in issues:
        return "corrupt_image", "reject_unreadable", "corrupt_image"
    if decoded_ok:
        return "decodable_image", None, None
    if record_type not in {"archive_member_image", "standalone_image"}:
        return "non_image_payload", "reject_unreadable", "non_image_payload"
    return "corrupt_image", "reject_unreadable", "corrupt_image"


def _nonimage_artifact_record(
    artifact: _PhysicalArtifact,
    bindings: list[dict[str, Any]],
    common_issues: list[str],
    historical_hashes: list[str],
    archive_status: str,
) -> dict[str, Any]:
    issues = sorted({*common_issues, "unsupported_download_artifact"})
    identity = {"archive_sha256": artifact.sha256, "record_type": "unsupported_download_artifact"}
    return {
        "archive_hash_status": archive_status,
        "archive_member_path": None,
        "audit_issues": issues,
        "decoded_image_sha256": None,
        "forensic_record_id": "fr_" + _sha256_bytes(canonical_json_bytes(identity)),
        "historically_recorded_archive_sha256s": historical_hashes,
        "inclusion_decision": "reject",
        "original_archive_path": artifact.paths[0],
        "original_archive_paths": list(artifact.paths),
        "original_archive_sha256": artifact.sha256,
        "original_archive_size_bytes": artifact.size_bytes,
        "provenance_issues": common_issues,
        "provenance_status": "complete" if not common_issues else "blocked",
        "record_type": "unsupported_download_artifact",
        "schema_version": RAW_FORENSIC_INVENTORY_SCHEMA_VERSION,
        "source_bindings": bindings,
    }


def _unresolved_record(binding: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "manifest_path": binding["manifest_path"],
        "source_row_sha256": binding["source_row_sha256"],
    }
    return {
        "archive_member_path": None,
        "audit_issues": list(binding["provenance_issues"]),
        "decoded_image_sha256": None,
        "forensic_record_id": "fr_" + _sha256_bytes(canonical_json_bytes(identity)),
        "inclusion_decision": "quarantine",
        "original_archive_path": None,
        "original_archive_sha256": None,
        "provenance_issues": list(binding["provenance_issues"]),
        "provenance_status": "blocked",
        "record_type": "unresolved_source_binding",
        "schema_version": RAW_FORENSIC_INVENTORY_SCHEMA_VERSION,
        "source_bindings": [binding],
    }


def _artifact_row(
    *,
    artifact: _PhysicalArtifact,
    bindings: list[dict[str, Any]],
    common_issues: list[str],
    historical_hashes: list[str],
    archive_status: str,
    artifact_kind: str,
    archive_issues: list[str],
    archive_member_count: int,
    archive_image_member_count: int,
    archive_crc_ok: bool | None,
    unsafe_member_count: int,
    casefold_collision_count: int,
) -> dict[str, Any]:
    return {
        "archive_casefold_collision_count": casefold_collision_count,
        "archive_crc_ok": archive_crc_ok,
        "archive_file_member_count": archive_member_count,
        "archive_image_member_count": archive_image_member_count,
        "archive_issues": sorted(set(archive_issues)),
        "archive_sha256_status": archive_status,
        "artifact_kind": artifact_kind,
        "current_observed_archive_sha256": artifact.sha256,
        "historically_recorded_archive_sha256s": historical_hashes,
        "original_archive_paths": list(artifact.paths),
        "original_archive_size_bytes": artifact.size_bytes,
        "provenance_issues": common_issues,
        "schema_version": RAW_FORENSIC_ARCHIVE_SCHEMA_VERSION,
        "source_bindings": bindings,
        "unsafe_archive_member_count": unsafe_member_count,
    }


def _artifact_provenance_issues(artifact: _PhysicalArtifact, bindings: list[dict[str, Any]]) -> list[str]:
    issues = {str(issue) for row in bindings for issue in row.get("provenance_issues", [])}
    paths = set(artifact.paths)
    acquisition_prefix = "experiments/acquisition_diversity_wave_v1/downloads/"
    itemicon = "data_sources/cc_by_itemiconpack32/itemiconpack32.png"
    if not bindings:
        if any(path.startswith(acquisition_prefix) for path in paths):
            issues.add("acquisition_orphan_artifact")
        elif itemicon in paths:
            issues.add("incomplete_itemicon_provenance")
        else:
            issues.add("unbound_download_artifact")
    if itemicon in paths:
        issues.add("incomplete_itemicon_provenance")
    return sorted(issues)


def _binding_metadata(bindings: list[dict[str, Any]]) -> dict[str, list[Any]]:
    def unique(key: str) -> list[Any]:
        serialized: dict[bytes, Any] = {}
        for row in bindings:
            value = row.get(key)
            if value in (None, "", {}, []):
                continue
            serialized[canonical_json_bytes(value)] = value
        return [serialized[key] for key in sorted(serialized)]

    return {
        "acquisition_runs": unique("acquisition_run"),
        "creators": unique("creator_or_publisher"),
        "download_urls": unique("download_url"),
        "licenses": unique("license"),
        "packs": unique("pack"),
        "platforms": unique("distribution_platform"),
        "source_urls": unique("source_url"),
    }


def _decode_image(payload: bytes | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                frame_count = int(getattr(image, "n_frames", 1))
                mode = str(image.mode)
                image_format = str(image.format or "unknown")
                if frame_count != 1:
                    return {
                        "error": "multi_frame_image_requires_explicit_operation",
                        "format": image_format,
                        "frame_count": frame_count,
                        "mode": mode,
                    }
                if mode not in SUPPORTED_RAW_IMAGE_MODES:
                    return {
                        "error": "unsupported_image_mode",
                        "format": image_format,
                        "frame_count": frame_count,
                        "mode": mode,
                    }
                image.load()
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return {"error": "image_decompression_bomb"}
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return {"error": "unreadable_image_payload"}
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
        return {"error": "invalid_decoded_rgba_geometry"}
    height, width = rgba.shape[:2]
    return {
        "decoded_rgba_sha256": decoded_rgba_sha256(rgba),
        "format": image_format,
        "frame_count": 1,
        "height": int(height),
        "mode": mode,
        "width": int(width),
    }


def _summarize(
    *,
    records: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    bindings: list[_ManifestBinding],
    manifest_count: int,
    physical_path_count: int,
) -> dict[str, Any]:
    image_records = [row for row in records if row["record_type"] in {"archive_member_image", "standalone_image"}]
    decisions = Counter(str(row["inclusion_decision"]) for row in records)
    issue_counts = Counter(str(issue) for row in records for issue in row.get("audit_issues", []))
    unique_rgba = {row["decoded_image_sha256"] for row in image_records if row.get("decoded_image_sha256")}
    zip_artifacts = [row for row in artifacts if row["artifact_kind"] == "zip_archive"]
    standalone = [row for row in artifacts if row["artifact_kind"] == "standalone_image"]
    resolved = sum(binding.artifact_sha256 is not None for binding in bindings)
    historical_present = sum(binding.historical_hash is not None for binding in bindings)
    historical_matches = sum(
        binding.historical_hash is not None and binding.artifact_sha256 == binding.historical_hash
        for binding in bindings
    )
    unknown_license_bindings = sum("unknown_license" in binding.issues for binding in bindings)
    missing_provenance_bindings = sum(bool(_PROVENANCE_GAP_ISSUES.intersection(binding.issues)) for binding in bindings)
    appledouble = sum("appledouble_resource_fork" in row.get("audit_issues", []) for row in records)
    unsafe = sum(int(row.get("unsafe_archive_member_count", 0)) for row in artifacts)
    crc_failures = sum(row.get("archive_crc_ok") is False for row in zip_artifacts)
    orphan = sum("acquisition_orphan_artifact" in row.get("provenance_issues", []) for row in artifacts)
    gate_passed = not issue_counts and all(row.get("inclusion_decision") == "accept" for row in records)
    return {
        "accepted_record_count": decisions["accept"],
        "acquisition_orphan_artifact_count": orphan,
        "appledouble_rejected_count": appledouble,
        "archive_file_member_count": sum(int(row["archive_file_member_count"]) for row in zip_artifacts),
        "archive_image_member_count": sum(int(row["archive_image_member_count"]) for row in zip_artifacts),
        "blocking_issue_counts": dict(sorted(issue_counts.items())),
        "decoded_image_record_count": sum(bool(row.get("decoded_image_sha256")) for row in image_records),
        "historical_hash_match_source_binding_count": historical_matches,
        "historical_hash_present_source_binding_count": historical_present,
        "missing_provenance_source_binding_count": missing_provenance_bindings,
        "physical_path_count": physical_path_count,
        "quarantined_record_count": decisions["quarantine"],
        "raw_source_gate_passed": gate_passed,
        "rejected_record_count": decisions["reject"],
        "resolved_source_binding_count": resolved,
        "schema_version": RAW_FORENSIC_INVENTORY_SCHEMA_VERSION,
        "source_binding_count": len(bindings),
        "source_manifest_count": manifest_count,
        "standalone_image_artifact_count": len(standalone),
        "unique_artifact_count": len(artifacts),
        "unique_decoded_rgba_count": len(unique_rgba),
        "unknown_license_source_binding_count": unknown_license_bindings,
        "unsafe_member_count": unsafe,
        "unresolved_source_binding_count": len(bindings) - resolved,
        "zip_archive_count": len(zip_artifacts),
        "zip_crc_failure_count": crc_failures,
    }


def _archive_hash_status(current_hash: str, historical_hashes: list[str]) -> str:
    if not historical_hashes:
        return "first_observed_current_hash"
    if current_hash in historical_hashes:
        return "matches_historically_recorded_hash"
    return "changed_from_historically_recorded_hash"


def _manifest_local_path(root: Path, row: Mapping[str, Any]) -> Path | None:
    raw = _text(row.get("local_archive_path"))
    if not raw:
        return None
    normalized = raw.replace("\\", "/")
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = root.joinpath(*PurePosixPath(normalized).parts)
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    return resolved if _is_below(root, resolved) else None


def _candidate_basenames(row: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    original = _text(row.get("original_filename"))
    if original:
        names.add(PurePosixPath(original.replace("\\", "/")).name)
    for key in ("download_url", "source_url"):
        url = _text(row.get(key))
        if url:
            basename = PurePosixPath(unquote(urlparse(url).path)).name
            if basename:
                names.add(basename)
    return names


def _distribution_platform(source_url: str, download_url: str) -> str | None:
    for url in (source_url, download_url):
        host = (urlparse(url).hostname or "").lower()
        if host:
            return host
    return None


def _unsafe_member_path(member_path: str) -> bool:
    normalized = member_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    return (
        not normalized
        or normalized.startswith("/")
        or "\x00" in normalized
        or any(part in {"", ".", ".."} for part in path.parts)
        or bool(path.parts and re.fullmatch(r"[A-Za-z]:", path.parts[0]))
    )


def _is_appledouble(member_path: str) -> bool:
    path = PurePosixPath(member_path)
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _zipinfo_is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return (unix_mode & 0o170000) == 0o120000


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _is_below(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _path_sort_key(path: Path) -> str:
    return path.as_posix()


def _record_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("original_archive_sha256") or "~"),
        str(row.get("archive_member_path") or ""),
        int(row.get("member_index") or 0),
        str(row.get("forensic_record_id") or ""),
    )


def _binding_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("manifest_path") or ""),
        int(row.get("manifest_line_number") or 0),
        str(row.get("source_row_sha256") or ""),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


__all__ = [
    "RAW_EXTRACTION_OPERATIONS",
    "RAW_EXTRACTION_OPERATION_SCHEMA_VERSION",
    "RAW_FORENSIC_ARCHIVE_SCHEMA_VERSION",
    "RAW_FORENSIC_INVENTORY_SCHEMA_VERSION",
    "RAW_FORENSIC_REPORT_VERSION",
    "RAW_OUTPUT_OPERATIONS",
    "RAW_TERMINAL_OPERATIONS",
    "CandidateSheetCoordinate",
    "ExtractionOperationError",
    "PayloadDisposition",
    "RawExtractionOperation",
    "RawForensicInventory",
    "audit_raw_source_inventory",
    "classify_image_payload",
    "execute_extraction_operation",
    "extraction_operation_json_schema",
    "make_extraction_relation_id",
    "operation_manifest_bytes",
    "verify_extraction_operation_manifest",
    "write_raw_forensic_inventory",
]
