"""Fail-closed loading of append-only acquisition provenance repairs."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPAIR_SCHEMA_VERSION = "spritelab_provenance_repair_v1"
RGBA_MARKER = b"spritelab-exported-rgba-v1\0"
ALPHA_MARKER = b"spritelab-alpha-mask-v1\0"
STATUS_BY_METHOD = {
    "original_file_recovered": "original_download_verified",
    "exact_source_reacquired": "reacquired_exact_source_verified",
    "upstream_changed": "reacquired_equivalent",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_json(value: Any) -> str:
    """Return the canonical on-disk representation for a repair artifact."""

    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def load_provenance_repairs(
    paths: Iterable[str | Path], *, workspace_root: Path
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Load and verify explicitly selected repair artifacts.

    The downloaded bytes and every declared source-to-derived correspondence are
    checked before a hash can be supplied to the pool builder.
    """

    index: dict[tuple[str, str], dict[str, Any]] = {}
    artifact_hashes: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        repair = json.loads(path.read_text(encoding="utf-8"))
        _validate_repair(repair, workspace_root=workspace_root.resolve())
        relative = _display_path(path, workspace_root.resolve())
        artifact_hashes[relative] = file_sha256(path)
        source_run = _required_text(repair, "source_run")
        mapping_by_id = {
            entry["sprite_id"]: entry for entry in repair["verification_evidence"]["archive_member_mapping"]
        }
        for sprite_id in repair["affected_sprite_ids"]:
            key = (source_run, sprite_id)
            if key in index:
                raise ValueError(f"multiple provenance repairs target {source_run}/{sprite_id}")
            mapping = mapping_by_id[sprite_id]
            width = mapping["source_dimensions"]["width"]
            height = mapping["source_dimensions"]["height"]
            crop = mapping["crop_box"]
            index[key] = {
                "archive_member": mapping["archive_member"],
                "cell_coordinates": f"crop_box_{'_'.join(str(value) for value in crop)}",
                "download_timestamp": repair.get("download_timestamp", ""),
                "download_url": repair.get("recorded_download_url", ""),
                "downloaded_file_size": repair["download_size"],
                "downloaded_filename": repair["downloaded_filename"],
                "downloaded_file_hash": repair["download_sha256"],
                "native_dimensions": mapping["source_dimensions"],
                "provenance_status": repair["new_provenance_status"],
                "provenance_recovery_method": repair["recovery_method"],
                "provenance_repair_artifact": relative,
                "provenance_repair_tool_config_hash": repair["tool_config_hash"],
                "resize_policy": (
                    f"exact_rgba_crop_{width}x{height}_box_{'_'.join(str(value) for value in crop)}"
                    f"_to_{crop[2] - crop[0]}x{crop[3] - crop[1]}"
                ),
                "source_image": mapping["archive_member"],
                "source_sheet": mapping["archive_member"],
            }
    return index, dict(sorted(artifact_hashes.items()))


def apply_provenance_repair(row: dict[str, Any], repair_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    repair = repair_index.get((str(row.get("source_run") or ""), str(row.get("sprite_id") or "")))
    if repair is None:
        return row
    if row.get("downloaded_file_hash") and row["downloaded_file_hash"] != repair["downloaded_file_hash"]:
        raise ValueError(f"repair conflicts with existing download hash for {row['sprite_id']}")
    return {**row, **repair}


def _validate_repair(repair: dict[str, Any], *, workspace_root: Path) -> None:
    if repair.get("repair_schema_version") != REPAIR_SCHEMA_VERSION:
        raise ValueError("unsupported provenance repair schema")
    method = _required_text(repair, "recovery_method")
    expected_status = STATUS_BY_METHOD.get(method)
    if expected_status is None or repair.get("new_provenance_status") != expected_status:
        raise ValueError("recovery method and new provenance status disagree")
    if repair.get("old_provenance_status") != "blocked_provenance":
        raise ValueError("repair must be append-only from blocked_provenance")
    for field in (
        "source_id",
        "source_run",
        "recorded_source_url",
        "downloaded_filename",
        "timestamp",
    ):
        _required_text(repair, field)
    _required_sha256(repair, "tool_config_hash")
    digest = _required_sha256(repair, "download_sha256")
    if repair.get("download_hash_scope") != "downloaded_file_bytes":
        raise ValueError("download hash scope must be downloaded_file_bytes")
    size = repair.get("download_size")
    if not isinstance(size, int) or size < 0:
        raise ValueError("download_size must be a non-negative integer")
    affected = repair.get("affected_sprite_ids")
    if not isinstance(affected, list) or not affected or affected != sorted(set(affected)):
        raise ValueError("affected_sprite_ids must be a non-empty sorted unique list")

    evidence = repair.get("verification_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("verification_evidence must be an object")
    correspondence = evidence.get("archive_member_mapping")
    if not isinstance(correspondence, list):
        raise ValueError("verification evidence must include archive_member_mapping")
    raw_mapped_ids = [entry.get("sprite_id") for entry in correspondence if isinstance(entry, dict)]
    if not all(isinstance(sprite_id, str) for sprite_id in raw_mapped_ids):
        raise ValueError("archive member mapping sprite IDs must be strings")
    mapped_ids = [sprite_id for sprite_id in raw_mapped_ids if isinstance(sprite_id, str)]
    if sorted(mapped_ids) != affected:
        raise ValueError("archive member mapping must cover every affected sprite exactly once")
    sprite_hashes = {
        entry.get(field)
        for entry in correspondence
        for field in ("derived_image_sha256", "exported_rgba_sha256", "alpha_mask_sha256")
    }
    if digest in sprite_hashes:
        raise ValueError("exported-sprite hash cannot be used as a download hash")

    local_path = _workspace_path(_required_text(repair, "local_download_path"), workspace_root)
    if local_path.name != repair["downloaded_filename"]:
        raise ValueError("downloaded filename does not match recovered file")
    if local_path.stat().st_size != size or file_sha256(local_path) != digest:
        raise ValueError("recovered download size or SHA-256 mismatch")

    _verify_correspondence(local_path, correspondence, workspace_root, digest)

    if method == "upstream_changed":
        historical = _required_sha256(repair, "historical_download_sha256")
        if historical == digest:
            raise ValueError("upstream_changed requires distinct historical and reacquired hashes")


def _verify_correspondence(
    download: Path, correspondence: list[dict[str, Any]], workspace_root: Path, download_digest: str
) -> None:
    if not zipfile.is_zipfile(download):
        raise ValueError("provenance repair currently requires a ZIP source with member mapping")
    with zipfile.ZipFile(download) as archive:
        members = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        for entry in correspondence:
            member = _required_text(entry, "archive_member")
            if member not in members:
                raise ValueError(f"archive member missing: {member}")
            payload = archive.read(member)
            if hashlib.sha256(payload).hexdigest() != _required_sha256(entry, "source_image_sha256"):
                raise ValueError(f"archive member hash mismatch: {member}")
            derived = _workspace_path(_required_text(entry, "derived_image_path"), workspace_root)
            derived_file_hash = file_sha256(derived)
            if derived_file_hash != _required_sha256(entry, "derived_image_sha256"):
                raise ValueError(f"derived image hash mismatch: {derived}")
            with (
                archive.open(member) as source_handle,
                Image.open(source_handle) as source,
                Image.open(derived) as target,
            ):
                source_rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
                target_rgba = np.asarray(target.convert("RGBA"), dtype=np.uint8)
            expected_dimensions = entry.get("source_dimensions")
            if expected_dimensions != {"height": int(source_rgba.shape[0]), "width": int(source_rgba.shape[1])}:
                raise ValueError(f"source dimensions mismatch: {member}")
            crop = entry.get("crop_box")
            if not isinstance(crop, list) or len(crop) != 4 or not all(isinstance(value, int) for value in crop):
                raise ValueError(f"invalid crop box: {member}")
            left, top, right, bottom = crop
            reproduced = source_rgba[top:bottom, left:right]
            if reproduced.shape != target_rgba.shape or not np.array_equal(reproduced, target_rgba):
                raise ValueError(f"source member does not reproduce derived image: {member}")
            rgba_hash = _canonical_rgba_sha256(target_rgba)
            alpha_hash = _alpha_mask_sha256(target_rgba[..., 3])
            if rgba_hash != _required_sha256(entry, "exported_rgba_sha256"):
                raise ValueError(f"exported RGBA hash mismatch: {member}")
            if alpha_hash != _required_sha256(entry, "alpha_mask_sha256"):
                raise ValueError(f"alpha mask hash mismatch: {member}")
            forbidden = {derived_file_hash, rgba_hash, alpha_hash}
            if download_digest in forbidden:
                raise ValueError("exported-sprite hash cannot be used as a download hash")


def _canonical_rgba_sha256(rgba: np.ndarray) -> str:
    value = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width = value.shape[:2]
    return hashlib.sha256(RGBA_MARKER + struct.pack(">II", width, height) + value.tobytes()).hexdigest()


def _alpha_mask_sha256(alpha: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(alpha) > 0, dtype=np.uint8)
    height, width = value.shape
    return hashlib.sha256(ALPHA_MARKER + struct.pack(">II", width, height) + value.tobytes()).hexdigest()


def _workspace_path(value: str, workspace_root: Path) -> Path:
    path = Path(value.replace("\\", "/"))
    resolved = path.resolve() if path.is_absolute() else (workspace_root / path).resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(f"repair path escapes workspace: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _required_text(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"missing or invalid {field}")
    return result


def _required_sha256(value: dict[str, Any], field: str) -> str:
    result = _required_text(value, field).lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"invalid SHA-256 in {field}")
    return result


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()
