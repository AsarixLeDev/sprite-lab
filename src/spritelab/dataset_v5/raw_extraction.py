"""Deterministic, no-interpolation extraction from verified raw sources."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from spritelab.dataset_v5.identity import (
    RecordBinding,
    canonical_json_bytes,
    canonical_rgba_bytes,
    decoded_rgba_sha256,
    make_geometry_family_id,
    make_record_id,
)
from spritelab.dataset_v5.raw_inventory import RawSourceRecord, file_sha256
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, remove_confined_tree

RAW_EXTRACTION_SCHEMA_VERSION = "sprite_lab_raw_extraction_v1"
RAW_BLOB_SCHEMA_VERSION = "sprite_lab_canonical_rgba_blob_v1"
RAW_BUILD_SCHEMA_VERSION = "sprite_lab_raw_extraction_build_v1"
DECODING_POLICY = "pillow_single_frame_convert_rgba_v1"
INTERPOLATION_POLICY = "none"
_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tga", ".tif", ".tiff", ".webp"})


class RawExtractionError(RuntimeError):
    """Base error for fail-closed deterministic extraction."""


class UnsafeArchiveError(RawExtractionError):
    """Raised for duplicate, unsafe, encrypted, or link-like ZIP members."""


class RebuildMismatchError(RawExtractionError):
    """Raised when two complete extraction builds differ at any byte."""


@dataclass(frozen=True)
class TransparentPadding:
    """Explicit transparent padding amounts in output pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int
    fill_rgba: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        amounts = (self.left, self.top, self.right, self.bottom)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in amounts):
            raise ValueError("padding amounts must be non-negative integers")
        if not any(amounts):
            raise ValueError("a zero-size padding operation must be omitted")
        if self.fill_rgba != (0, 0, 0, 0):
            raise ValueError("raw rebuild padding must use deterministic transparent black")

    def canonical(self) -> dict[str, Any]:
        return {
            "bottom": self.bottom,
            "fill_rgba": list(self.fill_rgba),
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "version": "explicit_transparent_padding_v1",
        }


@dataclass(frozen=True)
class ExtractionTransform:
    """The complete and only permitted pixel transformation ledger."""

    crop_coordinates: tuple[int, int, int, int] | None
    padding: TransparentPadding | None
    decoding_policy: str = DECODING_POLICY
    interpolation_policy: str = INTERPOLATION_POLICY

    def __post_init__(self) -> None:
        if self.decoding_policy != DECODING_POLICY:
            raise ValueError(f"unsupported decoding policy: {self.decoding_policy}")
        if self.interpolation_policy != INTERPOLATION_POLICY:
            raise ValueError("interpolation, resize, and resampling are forbidden")
        if self.crop_coordinates is not None:
            crop = self.crop_coordinates
            if len(crop) != 4 or any(not isinstance(value, int) or isinstance(value, bool) for value in crop):
                raise ValueError("crop_coordinates must contain exactly four integers")
            left, top, right, bottom = crop
            if left < 0 or top < 0 or right <= left or bottom <= top:
                raise ValueError(f"invalid crop coordinates: {crop}")

    @classmethod
    def whole_image(cls) -> ExtractionTransform:
        """Record an explicit no-crop, no-padding operation."""

        return cls(crop_coordinates=None, padding=None)

    @property
    def operation(self) -> str:
        if self.crop_coordinates is not None and self.padding is not None:
            return "decode_rgba_crop_then_pad"
        if self.crop_coordinates is not None:
            return "decode_rgba_crop"
        if self.padding is not None:
            return "decode_rgba_pad"
        return "decode_rgba_whole_image"

    def canonical(self) -> dict[str, Any]:
        return {
            "crop_coordinates": list(self.crop_coordinates) if self.crop_coordinates is not None else None,
            "decoding_policy": self.decoding_policy,
            "interpolation_policy": self.interpolation_policy,
            "operation": self.operation,
            "padding": self.padding.canonical() if self.padding is not None else None,
        }


@dataclass(frozen=True)
class RawExtractionSpec:
    """Bind a verified source/member to an explicit transform."""

    source: RawSourceRecord
    archive_member_path: str | None
    transform: ExtractionTransform


def list_source_image_members(source: RawSourceRecord) -> tuple[str, ...]:
    """List supported image members after fully validating ZIP structure."""

    _verify_archive_binding(source)
    if zipfile.is_zipfile(source.resolved_archive_path):
        with zipfile.ZipFile(source.resolved_archive_path) as archive:
            members = _validated_zip_members(archive)
            images = sorted(name for name in members if Path(name).suffix.casefold() in _IMAGE_SUFFIXES)
        if not images:
            raise RawExtractionError(f"ZIP contains no supported image members: {source.archive_path}")
        return tuple(images)
    _reject_corrupt_declared_zip(source)
    name = source.original_filename or source.resolved_archive_path.name
    return (name,)


def build_raw_extraction(
    specs: Iterable[RawExtractionSpec],
    output_root: str | Path,
    *,
    publish_direct_fresh: bool = False,
) -> dict[str, Any]:
    """Create a fresh content-addressed extraction artifact transactionally."""

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"raw extraction output already exists: {destination}")
    materialized = tuple(specs)
    if not materialized:
        raise RawExtractionError("cannot build an empty raw extraction")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if publish_direct_fresh:
        with AnchoredDirectory(destination.parent, destination.parent) as parent_anchor:
            owned = parent_anchor.mkdir(destination.name, exist_ok=False)
            try:
                with AnchoredDirectory(destination, destination) as destination_anchor:
                    if not owned.matches(destination_anchor.directory_metadata()):
                        raise RawExtractionError("fresh raw extraction directory changed while it was being opened")
                    records, blobs_by_id = _materialize_raw_extraction(materialized, destination, destination_anchor)
                    result = _raw_build_result(destination, records, blobs_by_id)
            except Exception as exc:
                residue = parent_anchor.quarantine_if_owned(
                    destination.name,
                    owned,
                    prefix=".raw-extraction-failed-",
                )
                if residue is None:
                    raise RawExtractionError("fresh raw extraction identity changed during rollback") from exc
                raise
        return result

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        with AnchoredDirectory(staging, staging) as staging_anchor:
            records, blobs_by_id = _materialize_raw_extraction(materialized, staging, staging_anchor)
        staging.replace(destination)
    except Exception:
        remove_confined_tree(staging, destination.parent, missing_ok=True)
        raise
    return _raw_build_result(destination, records, blobs_by_id)


def _materialize_raw_extraction(
    materialized: tuple[RawExtractionSpec, ...],
    staging: Path,
    staging_anchor: AnchoredDirectory,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    owned_blobs = staging_anchor.mkdir("blobs", exist_ok=False)
    blobs = staging / "blobs"
    with AnchoredDirectory(blobs, blobs) as blobs_anchor:
        if not owned_blobs.matches(blobs_anchor.directory_metadata()):
            raise RawExtractionError("raw extraction blob directory changed while it was being opened")
        records: list[dict[str, Any]] = []
        blobs_by_id: dict[str, dict[str, Any]] = {}
        record_ids: set[str] = set()
        for spec in sorted(materialized, key=_spec_sort_key):
            row, blob_row, blob_bytes = _extract_one(spec)
            record_id = str(row["record_id"])
            if record_id in record_ids:
                raise RawExtractionError(f"duplicate content-bound record identity: {record_id}")
            record_ids.add(record_id)
            records.append(row)

            blob_id = str(blob_row["blob_id"])
            blob_path = staging / str(blob_row["blob_path"])
            previous = blobs_by_id.get(blob_id)
            if previous is None:
                _write_new_bytes(blob_path, blob_bytes)
                if file_sha256(blob_path) != blob_id:
                    raise RawExtractionError(f"content-addressed blob verification failed: {blob_id}")
                blobs_by_id[blob_id] = blob_row
            elif previous != blob_row or blob_path.read_bytes() != blob_bytes:
                raise RawExtractionError(f"blob identity collision: {blob_id}")

        records.sort(key=lambda row: str(row["record_id"]))
        blob_rows = sorted(blobs_by_id.values(), key=lambda row: str(row["blob_id"]))
        _write_new_jsonl(staging / "extraction_manifest.jsonl", records)
        _write_new_jsonl(staging / "blob_manifest.jsonl", blob_rows)
        artifact_hashes = directory_file_hashes(staging)
        _write_new_json(
            staging / "build_manifest.json",
            {
                "artifact_hashes": artifact_hashes,
                "blob_count": len(blob_rows),
                "record_count": len(records),
                "schema_version": RAW_BUILD_SCHEMA_VERSION,
            },
        )
    return records, blobs_by_id


def _raw_build_result(
    destination: Path,
    records: list[dict[str, Any]],
    blobs_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_hashes": directory_file_hashes(destination),
        "blob_count": len(blobs_by_id),
        "output_root": str(destination),
        "record_count": len(records),
        "schema_version": RAW_BUILD_SCHEMA_VERSION,
    }


def build_twice_and_verify(
    specs: Iterable[RawExtractionSpec], first_root: str | Path, second_root: str | Path
) -> dict[str, Any]:
    """Run two complete fresh builds and require identical relative bytes."""

    materialized = tuple(specs)
    build_raw_extraction(materialized, first_root)
    build_raw_extraction(materialized, second_root)
    return verify_builds_byte_identical(first_root, second_root)


def verify_builds_byte_identical(first_root: str | Path, second_root: str | Path) -> dict[str, Any]:
    """Compare complete directory membership and every file SHA-256."""

    left = directory_file_hashes(first_root)
    right = directory_file_hashes(second_root)
    if left != right:
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        changed = sorted(path for path in set(left) & set(right) if left[path] != right[path])
        raise RebuildMismatchError(
            f"raw extraction rebuilds differ: missing_left={missing_left}, "
            f"missing_right={missing_right}, changed={changed}"
        )
    return {
        "byte_identical": True,
        "file_count": len(left),
        "file_hashes": left,
        "schema_version": "sprite_lab_two_build_verification_v1",
    }


def directory_file_hashes(root: str | Path) -> dict[str, str]:
    """Return sorted hashes for every regular file below a build root."""

    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {
        item.relative_to(path).as_posix(): file_sha256(item)
        for item in sorted((value for value in path.rglob("*") if value.is_file()), key=lambda p: p.as_posix())
    }


def _extract_one(spec: RawExtractionSpec) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    _verify_archive_binding(spec.source)
    payload, member_path = _read_original_bytes(spec)
    member_sha256 = hashlib.sha256(payload).hexdigest()
    source_rgba, source_mode, source_format = _decode_rgba(payload, member_path=member_path)
    source_height, source_width = source_rgba.shape[:2]
    source_decoded_hash = decoded_rgba_sha256(source_rgba)
    output = _apply_transform(source_rgba, spec.transform)
    output_height, output_width = output.shape[:2]
    output_hash = decoded_rgba_sha256(output)
    padding = spec.transform.padding.canonical() if spec.transform.padding is not None else None
    binding = RecordBinding(
        source_archive_sha256=spec.source.archive_sha256,
        archive_member_path=member_path,
        extraction_operation=f"{RAW_EXTRACTION_SCHEMA_VERSION}:{spec.transform.operation}",
        crop_coordinates=spec.transform.crop_coordinates,
        decoded_rgba_sha256=output_hash,
        padding_operation=padding,
    )
    record_id = make_record_id(binding)
    blob_bytes = canonical_rgba_bytes(output)
    blob_path = f"blobs/{output_hash}.rgba"
    row = {
        "acquisition_run": spec.source.acquisition_run,
        "archive_member_path": member_path,
        "blob_id": output_hash,
        "blob_path": blob_path,
        "crop_coordinates": (
            list(spec.transform.crop_coordinates) if spec.transform.crop_coordinates is not None else None
        ),
        "decoded_image_sha256": source_decoded_hash,
        "decoding_policy": spec.transform.decoding_policy,
        "geometry_family_id": make_geometry_family_id(output[:, :, 3]),
        "image_mode": source_mode,
        "image_format": source_format,
        "interpolation_policy": spec.transform.interpolation_policy,
        "license": dict(spec.source.license),
        "original_byte_sha256": member_sha256,
        "original_filename": Path(member_path).name,
        "output_decoded_rgba_sha256": output_hash,
        "output_height": output_height,
        "output_width": output_width,
        "padding_operation": padding,
        "provenance_status": spec.source.provenance_status,
        "record_id": record_id,
        "schema_version": RAW_EXTRACTION_SCHEMA_VERSION,
        "source_archive_path": spec.source.archive_path,
        "source_archive_sha256": spec.source.archive_sha256,
        "source_creator": spec.source.creator_or_publisher,
        "source_height": source_height,
        "source_manifest_path": spec.source.manifest_path,
        "source_manifest_sha256": spec.source.manifest_sha256,
        "source_pack": spec.source.source_name,
        "source_row_sha256": spec.source.source_row_sha256,
        "source_type": spec.source.source_type,
        "source_url": spec.source.source_url or spec.source.download_url,
        "source_width": source_width,
        "tight_foreground_bbox": _tight_foreground_bbox(output),
        "transformation": spec.transform.canonical(),
    }
    blob_row = {
        "blob_file_sha256": output_hash,
        "blob_id": output_hash,
        "blob_path": blob_path,
        "byte_length": len(blob_bytes),
        "encoding": "identity.canonical_rgba_bytes/decoded_rgba_v1",
        "height": output_height,
        "schema_version": RAW_BLOB_SCHEMA_VERSION,
        "width": output_width,
    }
    return row, blob_row, blob_bytes


def _read_original_bytes(spec: RawExtractionSpec) -> tuple[bytes, str]:
    path = spec.source.resolved_archive_path
    if zipfile.is_zipfile(path):
        if not spec.archive_member_path:
            raise RawExtractionError(f"ZIP extraction requires an explicit archive member: {spec.source.source_id}")
        with zipfile.ZipFile(path) as archive:
            members = _validated_zip_members(archive)
            requested = _canonical_member_name(spec.archive_member_path)
            info = members.get(requested)
            if info is None:
                raise RawExtractionError(f"archive member missing: {spec.archive_member_path}")
            try:
                return archive.read(info), requested
            except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError, OSError) as exc:
                raise RawExtractionError(f"cannot verify archive member {requested}: {exc}") from exc
    _reject_corrupt_declared_zip(spec.source)
    if spec.archive_member_path is not None:
        raise RawExtractionError("direct-file extraction must use archive_member_path=None")
    member_path = spec.source.original_filename or path.name
    return path.read_bytes(), _canonical_member_name(member_path.replace("\\", "/"))


def _validated_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    seen_names: set[str] = set()
    casefolded: dict[str, str] = {}
    for info in archive.infolist():
        canonical = _canonical_member_name(info.filename)
        if info.flag_bits & 0x1:
            raise UnsafeArchiveError(f"encrypted ZIP member is unsupported: {info.filename}")
        unix_mode = info.external_attr >> 16
        if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
            raise UnsafeArchiveError(f"ZIP symlink is forbidden: {info.filename}")
        if canonical in seen_names:
            raise UnsafeArchiveError(f"duplicate ZIP member: {canonical}")
        folded = canonical.casefold()
        previous = casefolded.get(folded)
        if previous is not None:
            raise UnsafeArchiveError(f"case-colliding ZIP members are ambiguous: {previous!r}, {canonical!r}")
        seen_names.add(canonical)
        casefolded[folded] = canonical
        if info.is_dir():
            continue
        members[canonical] = info
    return members


def _canonical_member_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise UnsafeArchiveError(f"unsafe ZIP member path: {value!r}")
    if value.startswith("/") or _is_drive_path(value):
        raise UnsafeArchiveError(f"absolute ZIP member path is forbidden: {value!r}")
    without_trailing = value[:-1] if value.endswith("/") else value
    raw_parts = without_trailing.split("/")
    if not without_trailing or any(part in {"", ".", ".."} for part in raw_parts):
        raise UnsafeArchiveError(f"unsafe ZIP member path: {value!r}")
    path = PurePosixPath(without_trailing)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise UnsafeArchiveError(f"unsafe ZIP member path: {value!r}")
    return path.as_posix()


def _is_drive_path(value: str) -> bool:
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"


def _decode_rgba(payload: bytes, *, member_path: str) -> tuple[np.ndarray, str, str]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if int(getattr(image, "n_frames", 1)) != 1:
                raise RawExtractionError(f"multi-frame image requires an explicit frame operation: {member_path}")
            source_mode = str(image.mode)
            source_format = str(image.format or "unknown")
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RawExtractionError(f"cannot deterministically decode {member_path}: {exc}") from exc
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
        raise RawExtractionError(f"invalid decoded RGBA geometry for {member_path}: {rgba.shape}")
    return rgba, source_mode, source_format


def _apply_transform(source: np.ndarray, transform: ExtractionTransform) -> np.ndarray:
    result = source
    if transform.crop_coordinates is not None:
        left, top, right, bottom = transform.crop_coordinates
        height, width = source.shape[:2]
        if right > width or bottom > height:
            raise RawExtractionError(
                f"crop is outside decoded image bounds: crop={transform.crop_coordinates}, size={width}x{height}"
            )
        result = source[top:bottom, left:right].copy()
    else:
        result = source.copy()
    if transform.padding is not None:
        padding = transform.padding
        height, width = result.shape[:2]
        padded = np.zeros(
            (height + padding.top + padding.bottom, width + padding.left + padding.right, 4), dtype=np.uint8
        )
        padded[padding.top : padding.top + height, padding.left : padding.left + width] = result
        result = padded
    return np.ascontiguousarray(result, dtype=np.uint8)


def _tight_foreground_bbox(rgba: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(rgba[:, :, 3] > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _verify_archive_binding(source: RawSourceRecord) -> None:
    path = source.resolved_archive_path
    if not path.is_file():
        raise RawExtractionError(f"verified source disappeared: {path}")
    observed = file_sha256(path)
    if observed != source.archive_sha256:
        raise RawExtractionError(
            f"verified source changed after inventory: expected {source.archive_sha256}, observed {observed}"
        )


def _reject_corrupt_declared_zip(source: RawSourceRecord) -> None:
    row = source.source_record
    declared = any(
        "zip" in str(value).casefold()
        for value in (source.source_type, row.get("download_kind", ""), source.original_filename)
    )
    if declared:
        raise RawExtractionError(f"source declares ZIP content but is not a valid ZIP: {source.archive_path}")


def _spec_sort_key(spec: RawExtractionSpec) -> tuple[str, str, bytes]:
    return (
        spec.source.archive_sha256,
        spec.archive_member_path or spec.source.original_filename,
        canonical_json_bytes(spec.transform.canonical()),
    )


def _write_new_bytes(path: Path, value: bytes) -> None:
    with AnchoredDirectory(path.parent, path.parent) as parent_anchor:
        descriptor = parent_anchor.open_file(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
        )
        identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            if not identity.matches(parent_anchor.lstat(path.name)) or os.fstat(descriptor).st_nlink != 1:
                raise RawExtractionError("raw extraction output changed while it was being written")
        finally:
            os.close(descriptor)


def _write_new_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(canonical_json_bytes(dict(row)).decode("utf-8") + "\n" for row in rows)
    _write_new_text(path, text)


def _write_new_json(path: Path, value: Any) -> None:
    _write_new_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _write_new_text(path: Path, value: str) -> None:
    _write_new_bytes(path, value.encode("utf-8"))
