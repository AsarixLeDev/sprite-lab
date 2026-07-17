"""Independent, read-only audits for conditioned Dataset-v5 candidates.

This module deliberately does not import the conditioned builder.  Pixel
descriptors and retained-pair duplicate decisions are recomputed from the
published Phase-7 arrays with an independently inventoried implementation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import struct
import unicodedata
import zipfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
from PIL import Image

from spritelab.product_features.conditioned_v5.identity import (
    TRUSTED_AUDITOR_IDS,
    trusted_auditor_inventory,
)
from spritelab.training.campaign import stable_hash
from spritelab.utils.portable_paths import is_portable_relative_path
from spritelab.utils.safe_fs import AnchoredDirectory, open_anchored_directory

LABEL_AUDIT_SCHEMA: Final = "spritelab.audit.conditioned-labels.v1"
DATASET_VALIDATION_SCHEMA: Final = "spritelab.audit.conditioned-dataset.v1"
LOCAL_PIXEL_VISION_ALGORITHM: Final = "local_pixel_vision_v1"
NEAR_DUPLICATE_ALGORITHM: Final = "conditioned_near_duplicate_v2"

LABEL_AUDIT_GATES: Final = frozenset(
    {
        "deterministic_source_grounding",
        "no_human_truth_claim",
        "taxonomy_contract",
        "semantic_coverage",
        "provenance_and_license_binding",
        "duplicate_family_split_integrity",
        "local_pixel_descriptor_recomputation",
    }
)
DATASET_VALIDATION_GATES: Final = frozenset(
    {
        "phase7_arrays",
        "exact_32x32",
        "manifest_npz_parity",
        "training_loader_all_splits",
        "vocabulary_and_benchmark",
        "count_range",
        "publication_filesystem_safety",
        "portable_paths",
        "provenance_hashes",
        "near_duplicate_retained_pair_recomputation",
    }
)

_LOCAL_PIXEL_VISION_CONFIG: Final = {
    "schema_version": "spritelab.dataset.local-pixel-vision-config.v1",
    "alpha_threshold": 255,
    "canvas_size": [32, 32],
    "dominant_palette": [
        ["black", [24, 24, 24]],
        ["gray", [128, 128, 128]],
        ["white", [232, 232, 232]],
        ["red", [210, 52, 52]],
        ["orange", [224, 126, 42]],
        ["yellow", [222, 204, 56]],
        ["green", [66, 170, 82]],
        ["cyan", [56, 184, 190]],
        ["blue", [58, 104, 206]],
        ["purple", [142, 74, 190]],
        ["pink", [220, 104, 164]],
        ["brown", [126, 82, 48]],
    ],
    "scale_max_bbox_dimension": {"tiny": 8, "small": 16, "medium": 24, "large": 30},
    "symmetry_mismatch_basis_points": {"high": 1000, "moderate": 2500},
    "edge_density_basis_points": {"low": 2500, "medium": 5000},
}
LOCAL_PIXEL_VISION_CONFIG_IDENTITY: Final = stable_hash(_LOCAL_PIXEL_VISION_CONFIG)

_NEAR_DUPLICATE_CONFIG: Final = {
    "schema_version": "spritelab.dataset.conditioned-near-duplicate-config.v1",
    "same_taxonomy_category": True,
    "max_perceptual_hamming": 2,
    "max_bbox_dimension_delta": 1,
    "max_bbox_center_delta_half_pixels": 2,
    "max_alpha_xor_pixels": 12,
}
NEAR_DUPLICATE_CONFIG_IDENTITY: Final = stable_hash(_NEAR_DUPLICATE_CONFIG)

_TAXONOMY: Final = frozenset(
    {
        "character",
        "creature",
        "weapon",
        "tool",
        "armor",
        "potion",
        "food",
        "plant",
        "terrain",
        "building",
        "furniture",
        "vehicle",
        "effect",
        "icon",
        "interface",
    }
)
_ALLOWED_LICENSES: Final = frozenset({"cc0-1.0", "public-domain"})
_HIGH_IMPACT: Final = frozenset({"character", "creature", "weapon", "armor", "vehicle", "effect"})
_GENERIC_OBJECTS: Final = frozenset({"sprite", "pixel_art_sprite", "item", "object", "asset", "unknown"})
_LABEL_CONTRACT_KEYS: Final = frozenset(
    {
        "schema_version",
        "category",
        "object_name",
        "tags",
        "short_description",
        "confidence",
        "confidence_reason",
        "captions",
        "prompt_phrases",
        "negative_tags",
        "disagreement",
        "audit_state",
        "human_truth_claim",
    }
)
_SEMANTIC_KEYS: Final = frozenset(
    {
        "schema_version",
        "category",
        "object_name",
        "base_object",
        "open_name",
        "attributes",
        "aliases",
        "captions",
        "prompt_phrases",
        "negative_tags",
        "source_evidence",
        "warnings",
    }
)
_ATTRIBUTE_KEYS: Final = frozenset(
    {"colors", "materials", "shapes", "effects", "state", "function", "mood", "style", "parts", "environment"}
)
_LABEL_EVIDENCE_KEYS: Final = frozenset(
    {
        "evidence_type",
        "inference_method",
        "human_verified",
        "human_truth_claim",
        "claim_scope",
        "source_relative_path",
        "source_path_sha256",
        "tokens",
        "taxonomy_category",
        "source_id",
        "source_pack",
        "source_group",
        "source_sha256",
        "source_byte_count",
        "license_id",
        "creator",
        "duplicate_family_id",
        "local_pixel_vision",
        "local_pixel_vision_algorithm",
        "local_pixel_vision_config",
        "local_pixel_vision_config_identity",
        "implementation_code_inventory_sha256",
        "semantic_category_from_pixels",
    }
)
_DERIVED_SHEET_FRAME_KEYS: Final = frozenset(
    {
        "schema_version",
        "dataset_item_id",
        "parent_source_relative_path",
        "parent_source_raw_sha256",
        "parent_source_decoded_rgba_sha256",
        "crop_rectangle",
        "frame_index",
        "recipe_version",
        "recipe_identity",
        "decoded_rgba_sha256",
        "width",
        "height",
        "source_provenance_identity",
        "source_group_identity",
        "semantic_relative_path",
        "output_relative_path",
        "encoded_output_sha256",
        "encoded_output_byte_count",
        "derivation_identity",
        "record_identity",
        "source_derived_not_augmentation",
    }
)
_DERIVED_SHEET_RECIPE: Final = {
    "schema_version": "spritelab.dataset.conditioned-derived-sheet-recipe.v1",
    "input_encoding": "decoded-rgba8",
    "crop_semantics": "left-top-inclusive-right-bottom-exclusive",
    "output_encoding": "png-rgba8-filter-none-zlib-level-9",
    "resize_or_resample": False,
    "augmentation": False,
}
_DERIVED_SHEET_RECIPE_IDENTITY: Final = stable_hash(_DERIVED_SHEET_RECIPE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^harvest-[a-z0-9][a-z0-9-]{5,80}$")
_DATASET_REFERENCE = re.compile(r"^dataset\.[0-9a-f]{24}$")
_WORK_NAME = re.compile(r"^intake-[0-9a-f]{32}$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_MAX_RECEIPT_BYTES: Final = 128 * 1024 * 1024
_MAX_DERIVED_PARENT_PIXELS: Final = 16_777_216
_MAX_FILE_BYTES: Final = 512 * 1024 * 1024
_MAX_TOTAL_BYTES: Final = 1024 * 1024 * 1024
_INVENTORY_SCHEMA: Final = "spritelab.dataset.freeze.inventory.v1"
_DEFAULT_NEGATIVE_TAGS: Final = ("photorealistic", "large_scene", "text", "watermark")
_ROLE_NAMES: Final = {
    "0": "transparent",
    "1": "outline",
    "2": "deep_shadow",
    "3": "shadow",
    "4": "midtone",
    "5": "light",
    "6": "highlight",
    "7": "accent",
    "8": "emissive",
    "9": "texture_detail",
    "255": "unknown",
}
_REQUIRED_PHASE7_FILES: Final = frozenset(
    {
        "benchmark_manifest.json",
        "conditioned_records.jsonl",
        "conditioning_vocabulary.json",
        "coverage_report.json",
        "dataset_config.json",
        "dataset_qa_report.json",
        "dataset_report.md",
        "duplicate_report.json",
        "label_audit_subjects.json",
        "loader_check.json",
        "manifest_test.jsonl",
        "manifest_train.jsonl",
        "manifest_val.jsonl",
        "provenance_manifest.json",
        "rejected.jsonl",
        "split_assignments.jsonl",
        "split_integrity_report.json",
        "test.npz",
        "train.npz",
        "training_manifest.jsonl",
        "training_manifest_qa_report.json",
        "val.npz",
        "view_manifest.json",
        "vocab.json",
    }
)

ProgressCallback = Callable[[str, int, int, str], None]
CancellationCallback = Callable[[], bool]


class IndependentAuditError(ValueError):
    """A candidate failed one independently recomputed audit contract."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


class IndependentAuditCancelled(IndependentAuditError):
    """The durable audit cancellation flag was observed."""

    def __init__(self) -> None:
        super().__init__("audit_cancelled", "Independent audit cancelled; no PASS evidence was attached.")


def run_independent_audit(
    kind: str,
    job_root: os.PathLike[str] | str,
    candidate: Mapping[str, Any],
    *,
    project_root: os.PathLike[str] | str,
    progress: ProgressCallback,
    cancelled: CancellationCallback,
) -> dict[str, Any]:
    """Run one independently inventoried audit without modifying candidate bytes."""

    if kind not in {"label_audit", "dataset_validation"}:
        raise IndependentAuditError("audit_kind", "The independent audit kind is unsupported.")
    _check_cancelled(cancelled)
    inventory = _candidate_inventory(candidate)
    progress("inventory", 0, len(inventory), "Verifying every exact candidate payload file.")
    payloads = _read_and_verify_payloads(job_root, inventory, progress=progress, cancelled=cancelled)
    _check_cancelled(cancelled)
    progress("dataset_load", 0, 3, "Loading every production split through the Phase-7 array contract.")
    expected_count = candidate.get("image_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise IndependentAuditError("audit_count", "The candidate image count is invalid.")
    dataset = _load_dataset(
        payloads,
        expected_total=expected_count,
        progress=progress,
        cancelled=cancelled,
    )
    _verify_common_contracts(candidate, payloads, dataset)
    _verify_parent_bound_derivations(
        project_root,
        job_root,
        candidate,
        dataset,
        progress=progress,
        cancelled=cancelled,
    )
    if kind == "label_audit":
        metrics = _run_label_audit(candidate, payloads, dataset, progress=progress, cancelled=cancelled)
        gates = LABEL_AUDIT_GATES
        schema = LABEL_AUDIT_SCHEMA
    else:
        metrics = _run_dataset_validation(candidate, payloads, dataset, progress=progress, cancelled=cancelled)
        gates = DATASET_VALIDATION_GATES
        schema = DATASET_VALIDATION_SCHEMA
    _check_cancelled(cancelled)
    progress("report", 1, 1, "All independently recomputed gates passed; binding the exact report.")
    return _report(kind, schema, gates, candidate, metrics)


def _candidate_inventory(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = candidate.get("payload_inventory")
    if not isinstance(raw, Mapping) or not raw:
        raise IndependentAuditError("audit_inventory", "The candidate payload inventory is unavailable.")
    result: dict[str, dict[str, Any]] = {}
    collision_keys: set[str] = set()
    total = 0
    for raw_name, raw_record in sorted(raw.items()):
        name = str(raw_name)
        if not _portable_relative_path(name) or PurePosixPath(name).parts != (name,):
            raise IndependentAuditError("audit_inventory", "The candidate payload inventory is not flat and portable.")
        collision = unicodedata.normalize("NFC", name).casefold()
        if collision in collision_keys:
            raise IndependentAuditError("audit_inventory", "The candidate payload inventory contains a path collision.")
        collision_keys.add(collision)
        if not isinstance(raw_record, Mapping):
            raise IndependentAuditError("audit_inventory", "A candidate payload identity is invalid.")
        digest = str(raw_record.get("sha256") or "")
        size = raw_record.get("byte_count")
        if not _SHA256.fullmatch(digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise IndependentAuditError("audit_inventory", "A candidate payload identity is invalid.")
        if size > _MAX_FILE_BYTES:
            raise IndependentAuditError("audit_inventory", "A candidate payload exceeds the independent audit bound.")
        total += size
        result[name] = {"sha256": digest, "byte_count": size}
    if total > _MAX_TOTAL_BYTES or set(result) != _REQUIRED_PHASE7_FILES:
        raise IndependentAuditError(
            "audit_inventory", "The Phase-7 payload file set is not exact or exceeds its audit bound."
        )
    inventory_payload = {
        "schema_version": _INVENTORY_SCHEMA,
        "files": result,
        "file_count": len(result),
        "total_bytes": total,
    }
    if stable_hash(inventory_payload) != candidate.get("payload_inventory_sha256"):
        raise IndependentAuditError("audit_inventory", "The candidate payload inventory identity is invalid.")
    return result


def _read_and_verify_payloads(
    job_root: os.PathLike[str] | str,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    progress: ProgressCallback,
    cancelled: CancellationCallback,
) -> dict[str, bytes]:
    root = os.fspath(job_root)
    phase7 = os.path.join(root, "candidate", "phase7")
    payloads: dict[str, bytes] = {}
    with open_anchored_directory(phase7, root) as anchor:
        if set(anchor.names()) != set(inventory):
            raise IndependentAuditError("audit_inventory", "Candidate payload entries differ from the bound inventory.")
        for index, name in enumerate(sorted(inventory), start=1):
            _check_cancelled(cancelled)
            expected = inventory[name]
            payload = _read_bound_file(anchor, name, int(expected["byte_count"]))
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise IndependentAuditError("audit_inventory", "A candidate payload file changed during audit.")
            payloads[name] = payload
            progress("inventory", index, len(inventory), f"Verified candidate file {index} of {len(inventory)}.")
    return payloads


def _read_bound_file(anchor: AnchoredDirectory, name: str, expected_size: int) -> bytes:
    before = anchor.lstat(name)
    reparse = int(getattr(before, "st_file_attributes", 0)) & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or reparse
        or before.st_nlink != 1
        or before.st_size != expected_size
    ):
        raise IndependentAuditError("audit_payload_unsafe", "A candidate payload entry is not a safe regular file.")
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    try:
        opened = os.fstat(descriptor)
        if not _same_file(before, opened):
            raise IndependentAuditError("audit_payload_changed", "A candidate payload changed while being opened.")
        chunks: list[bytes] = []
        remaining = expected_size + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = anchor.lstat(name)
    payload = b"".join(chunks)
    if len(payload) != expected_size or not _same_file(before, after_open) or not _same_file(before, after):
        raise IndependentAuditError("audit_payload_changed", "A candidate payload changed while being read.")
    return payload


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_nlink == right.st_nlink == 1
    )


def _load_dataset(
    payloads: Mapping[str, bytes],
    *,
    expected_total: int,
    progress: ProgressCallback,
    cancelled: CancellationCallback,
) -> dict[str, Any]:
    for name, payload in payloads.items():
        if name.endswith(".json"):
            _assert_portable(_json_value(payload))
        elif name.endswith(".jsonl"):
            _assert_portable(_jsonl(payload))
    records = _jsonl(payloads["conditioned_records.jsonl"])
    record_by_id = _unique_rows(records, "sprite_id", "conditioned records")
    manifests: dict[str, list[dict[str, Any]]] = {}
    sprites: dict[str, dict[str, Any]] = {}
    required_arrays = {"alpha", "index_map", "role_map", "palette", "palette_mask", "category_id", "sprite_id"}
    for split_index, split in enumerate(("train", "val", "test"), start=1):
        _check_cancelled(cancelled)
        manifest = _jsonl(payloads[f"manifest_{split}.jsonl"])
        manifest_by_id = _unique_rows(manifest, "sprite_id", f"{split} manifest")
        manifest_index_by_id = {str(row["sprite_id"]): index for index, row in enumerate(manifest)}
        _validate_npz_archive(payloads[f"{split}.npz"], expected_total=expected_total)
        with np.load(io.BytesIO(payloads[f"{split}.npz"]), allow_pickle=False) as arrays:
            if set(arrays.files) != required_arrays:
                raise IndependentAuditError("audit_arrays", "A Phase-7 split has unexpected array fields.")
            sprite_ids = [str(value) for value in np.asarray(arrays["sprite_id"])]
            alpha = np.asarray(arrays["alpha"])
            index_map = np.asarray(arrays["index_map"])
            role_map = np.asarray(arrays["role_map"])
            palette = np.asarray(arrays["palette"])
            palette_mask = np.asarray(arrays["palette_mask"])
            category_id = np.asarray(arrays["category_id"])
            count = len(sprite_ids)
            if (
                alpha.dtype != np.dtype(np.uint8)
                or index_map.dtype != np.dtype(np.int16)
                or role_map.dtype != np.dtype(np.uint8)
                or palette.dtype != np.dtype(np.uint8)
                or palette_mask.dtype != np.dtype(np.bool_)
                or category_id.dtype != np.dtype(np.int64)
                or np.asarray(arrays["sprite_id"]).dtype.kind != "U"
                or np.asarray(arrays["sprite_id"]).dtype.itemsize > 512
                or alpha.shape != (count, 32, 32)
                or index_map.shape != (count, 32, 32)
                or role_map.shape != (count, 32, 32)
                or palette.shape != (count, 33, 3)
                or palette_mask.shape != (count, 33)
                or category_id.shape != (count,)
                or any(
                    not np.asarray(arrays[key]).flags.c_contiguous
                    for key in ("alpha", "index_map", "role_map", "palette", "palette_mask", "category_id", "sprite_id")
                )
                or len(set(sprite_ids)) != count
                or set(sprite_ids) != set(manifest_by_id)
            ):
                raise IndependentAuditError("audit_arrays", "A Phase-7 split violates shape or manifest parity.")
            if set(sprite_ids) & set(sprites):
                raise IndependentAuditError("audit_arrays", "A sprite identity appears in more than one split.")
            for row_index, sprite_id in enumerate(sprite_ids):
                if sprite_id not in record_by_id:
                    raise IndependentAuditError("audit_arrays", "A Phase-7 array lacks a conditioned record.")
                alpha_row = np.asarray(alpha[row_index], dtype=np.uint8)
                index_row = np.asarray(index_map[row_index], dtype=np.int64)
                palette_row = np.asarray(palette[row_index], dtype=np.uint8)
                mask_row = np.asarray(palette_mask[row_index], dtype=bool)
                if (
                    not np.all(np.isin(alpha_row, (0, 1)))
                    or np.any(index_row < 0)
                    or np.any(index_row >= palette_row.shape[0])
                    or np.any((alpha_row == 0) & (index_row != 0))
                    or np.any((alpha_row == 1) & (index_row < 1))
                    or np.any((alpha_row == 0) & (np.asarray(role_map[row_index]) != 0))
                    or np.any((alpha_row == 1) & (np.asarray(role_map[row_index]) == 0))
                    or not np.all(np.isin(np.asarray(role_map[row_index]), (*range(10), 255)))
                    or not mask_row[0]
                    or not np.all(palette_row[0] == 0)
                    or np.any(mask_row[1:] < mask_row[:-1])
                    or np.any(~mask_row[index_row[alpha_row == 1]])
                ):
                    raise IndependentAuditError("audit_arrays", "A Phase-7 sprite violates palette/index invariants.")
                rgba = np.empty((32, 32, 4), dtype=np.uint8)
                rgba[:, :, :3] = palette_row[index_row]
                rgba[:, :, 3] = alpha_row * 255
                sprites[sprite_id] = {
                    "sprite_id": sprite_id,
                    "split": split,
                    "rgba": rgba,
                    "manifest": manifest_by_id[sprite_id],
                    "record": record_by_id[sprite_id],
                    "category_id": int(category_id[row_index]),
                    "npz_row": row_index,
                    "manifest_row": manifest_index_by_id[sprite_id],
                }
        manifests[split] = manifest
        progress("dataset_load", split_index, 3, f"Loaded and checked {split} arrays and manifest.")
    if set(sprites) != set(record_by_id):
        raise IndependentAuditError("audit_arrays", "Conditioned records and Phase-7 arrays differ.")
    return {
        "records": records,
        "record_by_id": record_by_id,
        "manifests": manifests,
        "sprites": sprites,
        "split_assignments": _jsonl(payloads["split_assignments.jsonl"]),
    }


def _validate_npz_archive(payload: bytes, *, expected_total: int) -> None:
    header_allowance = 64 * 1024
    expected_members = {
        "alpha.npy": expected_total * 32 * 32 + header_allowance,
        "index_map.npy": expected_total * 32 * 32 * 2 + header_allowance,
        "role_map.npy": expected_total * 32 * 32 + header_allowance,
        "palette.npy": expected_total * 33 * 3 + header_allowance,
        "palette_mask.npy": expected_total * 33 + header_allowance,
        "category_id.npy": expected_total * 8 + header_allowance,
        "sprite_id.npy": expected_total * 512 + header_allowance,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            entries = archive.infolist()
            if len(entries) != len(expected_members) or {entry.filename for entry in entries} != set(expected_members):
                raise IndependentAuditError("audit_npz_archive", "A Phase-7 NPZ has unexpected or duplicate members.")
            expanded_total = 0
            for entry in entries:
                if (
                    entry.is_dir()
                    or entry.flag_bits & 0x1
                    or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or entry.file_size <= 0
                    or entry.file_size > expected_members[entry.filename]
                    or entry.compress_size <= 0
                    or entry.compress_size > len(payload)
                ):
                    raise IndependentAuditError("audit_npz_archive", "A Phase-7 NPZ member exceeds safe bounds.")
                expanded_total += entry.file_size
            if expanded_total > sum(expected_members.values()) or archive.testzip() is not None:
                raise IndependentAuditError("audit_npz_archive", "A Phase-7 NPZ failed bounded integrity checks.")
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        if isinstance(exc, IndependentAuditError):
            raise
        raise IndependentAuditError("audit_npz_archive", "A Phase-7 NPZ is invalid.") from exc


def _verify_common_contracts(
    candidate: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    dataset: Mapping[str, Any],
) -> None:
    sprites = dataset["sprites"]
    source_bindings = _candidate_source_bindings(candidate)
    if len(sprites) != candidate.get("image_count"):
        raise IndependentAuditError("audit_count", "The candidate count differs from the loaded production arrays.")
    assignments = _unique_rows(dataset["split_assignments"], "sprite_id", "split assignments")
    if set(assignments) != set(sprites):
        raise IndependentAuditError("audit_split", "Split assignments do not cover every candidate sprite.")
    family_splits: dict[str, set[str]] = defaultdict(set)
    source_group_splits: dict[str, set[str]] = defaultdict(set)
    decoded_hashes: dict[str, str] = {}
    for sprite_id, sprite in sorted(sprites.items()):
        record = sprite["record"]
        manifest = sprite["manifest"]
        assignment = assignments[sprite_id]
        split = str(sprite["split"])
        if str(record.get("split") or "") != split or str(manifest.get("split") or "") != split:
            raise IndependentAuditError("audit_split", "A sprite split binding is inconsistent.")
        if str(assignment.get("split") or "") != split:
            raise IndependentAuditError("audit_split", "A split assignment differs from its production array.")
        family = str(record.get("duplicate_family_id") or "")
        source_group = str(record.get("source_group") or "")
        source_binding = source_bindings.get(str(record.get("source_id") or ""))
        if not family or not source_group:
            raise IndependentAuditError(
                "audit_provenance", "A sprite lacks duplicate-family or source-group provenance."
            )
        if source_binding is None or (
            record.get("source_pack") != source_binding.get("title")
            or record.get("creator") != source_binding.get("creator")
            or record.get("license_id") != source_binding.get("license_id")
        ):
            raise IndependentAuditError("audit_provenance", "A sprite differs from its exact candidate source binding.")
        if assignment.get("duplicate_family_id") != family or assignment.get("source_group") != source_group:
            raise IndependentAuditError("audit_split", "A split assignment changed its provenance group.")
        decoded_sha256 = hashlib.sha256(np.asarray(sprite["rgba"], dtype=np.uint8).tobytes()).hexdigest()
        if decoded_sha256 in decoded_hashes:
            raise IndependentAuditError(
                "audit_exact_duplicate",
                "Two retained sprites decode to identical RGBA pixels across the candidate.",
            )
        decoded_hashes[decoded_sha256] = sprite_id
        record_visual = record.get("local_pixel_vision")
        manifest_evidence = manifest.get("label_evidence")
        if (
            not isinstance(record_visual, Mapping)
            or record_visual.get("decoded_rgba_sha256") != decoded_sha256
            or not isinstance(manifest_evidence, Mapping)
            or dict(manifest_evidence.get("local_pixel_vision") or {}).get("decoded_rgba_sha256") != decoded_sha256
        ):
            raise IndependentAuditError("audit_pixel_binding", "Decoded RGBA does not match its record and manifest.")
        _verify_source_derivation(
            record,
            manifest,
            rgba=np.asarray(sprite["rgba"], dtype=np.uint8),
            source_binding=source_binding,
        )
        family_splits[family].add(split)
        source_group_splits[source_group].add(split)
    if any(len(values) != 1 for values in family_splits.values()) or any(
        len(values) != 1 for values in source_group_splits.values()
    ):
        raise IndependentAuditError("audit_split", "A duplicate family or source group crosses dataset splits.")
    split_report = _json_mapping(payloads["split_integrity_report.json"])
    if split_report.get("ok") is not True or split_report.get("cross_split_duplicate_families") != []:
        raise IndependentAuditError("audit_split", "The bound split-integrity report is not a clean PASS.")


def _verify_parent_bound_derivations(
    project_root: os.PathLike[str] | str,
    job_root: os.PathLike[str] | str,
    candidate: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    progress: ProgressCallback,
    cancelled: CancellationCallback,
) -> None:
    derived: dict[str, list[tuple[str, Mapping[str, Any], np.ndarray]]] = defaultdict(list)
    for sprite_id, sprite in sorted(dataset["sprites"].items()):
        record = sprite["record"]
        if record.get("source_derivation") is None:
            continue
        derived[str(record.get("source_id") or "")].append(
            (sprite_id, record, np.asarray(sprite["rgba"], dtype=np.uint8))
        )
    if not derived:
        return

    project = Path(os.path.abspath(os.fspath(project_root)))
    job = Path(os.path.abspath(os.fspath(job_root)))
    expected_jobs_root = project / "runs" / "v3" / "conditioned-dataset-v5"
    if job.parent != expected_jobs_root:
        raise IndependentAuditError("audit_parent_binding", "The audit job root is outside its managed namespace.")
    receipts_root = project / "datasets" / "conditioned_intake_receipts"
    bindings = _candidate_source_bindings(candidate)
    completed = 0
    total = sum(len(rows) for rows in derived.values())
    with open_anchored_directory(receipts_root, project) as receipts_anchor:
        for source_id, rows in sorted(derived.items()):
            _check_cancelled(cancelled)
            binding = bindings.get(source_id)
            if binding is None:
                raise IndependentAuditError("audit_parent_binding", "A derived source lacks its managed input binding.")
            reference = str(binding.get("dataset_reference") or "")
            if _DATASET_REFERENCE.fullmatch(reference) is None:
                raise IndependentAuditError(
                    "audit_parent_binding", "A derived source has an invalid managed Dataset reference."
                )
            receipt_name = f"{reference}.json"
            receipt_metadata = receipts_anchor.lstat(receipt_name)
            if not 0 < receipt_metadata.st_size <= _MAX_RECEIPT_BYTES:
                raise IndependentAuditError("audit_parent_binding", "A managed intake receipt exceeds audit bounds.")
            receipt_value = _json_value(_read_bound_file(receipts_anchor, receipt_name, receipt_metadata.st_size))
            if not isinstance(receipt_value, Mapping):
                raise IndependentAuditError("audit_parent_binding", "A managed intake receipt is malformed.")
            receipt = dict(receipt_value)
            receipt_payload = dict(receipt)
            receipt_identity = str(receipt_payload.pop("receipt_identity", ""))
            managed = receipt.get("managed")
            harvest = receipt.get("harvest")
            if (
                receipt.get("schema_version") != "spritelab.dataset.conditioned-import-receipt.v2"
                or receipt.get("dataset_reference") != reference
                or receipt_identity != binding.get("managed_intake_receipt_identity")
                or stable_hash(receipt_payload) != receipt_identity
                or not isinstance(managed, Mapping)
                or not isinstance(harvest, Mapping)
                or harvest.get("run_id") != binding.get("harvest_run_id")
                or receipt.get("source") != binding.get("source_document")
                or receipt.get("license") != binding.get("license_document")
            ):
                raise IndependentAuditError(
                    "audit_parent_binding", "A managed intake receipt differs from its candidate binding."
                )
            managed_value = dict(managed)
            source_inventory = managed_value.get("source_inventory")
            derived_inventory = managed_value.get("derived_inventory")
            manifest = managed_value.get("derived_sheet_manifest")
            if (
                not isinstance(source_inventory, Mapping)
                or not isinstance(derived_inventory, Mapping)
                or not isinstance(manifest, Mapping)
                or stable_hash(dict(source_inventory)) != binding.get("managed_source_inventory_sha256")
                or managed_value.get("source_inventory_sha256") != binding.get("managed_source_inventory_sha256")
                or stable_hash(dict(derived_inventory)) != binding.get("managed_derived_inventory_sha256")
                or managed_value.get("derived_inventory_sha256") != binding.get("managed_derived_inventory_sha256")
                or manifest.get("manifest_identity") != binding.get("derived_sheet_manifest_identity")
            ):
                raise IndependentAuditError(
                    "audit_parent_binding", "Managed parent or derived inventories differ from the candidate."
                )
            manifest_payload = dict(manifest)
            manifest_identity = str(manifest_payload.pop("manifest_identity", ""))
            manifest_records = manifest.get("records")
            if stable_hash(manifest_payload) != manifest_identity or not isinstance(manifest_records, list):
                raise IndependentAuditError("audit_parent_binding", "A derived-frame manifest is malformed.")
            receipt_records: dict[str, dict[str, Any]] = {}
            for raw in manifest_records:
                if not isinstance(raw, Mapping):
                    raise IndependentAuditError("audit_parent_binding", "A derived-frame manifest row is malformed.")
                row = dict(raw)
                row_identity = str(row.get("record_identity") or "")
                if not _SHA256.fullmatch(row_identity) or row_identity in receipt_records:
                    raise IndependentAuditError("audit_parent_binding", "A derived-frame manifest row is ambiguous.")
                receipt_records[row_identity] = row

            work_relative = str(managed_value.get("work_relative_path") or "")
            work_parts = PurePosixPath(work_relative).parts
            source_relative = str(managed_value.get("source_relative_path") or "")
            derived_relative = str(managed_value.get("derived_root_relative_path") or "")
            if (
                not _portable_relative_path(work_relative)
                or len(work_parts) != 3
                or work_parts[:2] != ("datasets", "conditioned_intake_work")
                or _WORK_NAME.fullmatch(work_parts[2]) is None
                or source_relative != f"{work_relative}/source"
                or derived_relative != f"{work_relative}/derived_sprites"
            ):
                raise IndependentAuditError("audit_parent_binding", "A managed parent root is outside its namespace.")
            source_root = project.joinpath(*PurePosixPath(source_relative).parts)
            derived_root = project.joinpath(*PurePosixPath(derived_relative).parts)
            source_files = source_inventory.get("files")
            derived_files = derived_inventory.get("files")
            artifact_manifest = receipt.get("artifact_manifest")
            artifact_rows = artifact_manifest.get("files") if isinstance(artifact_manifest, Mapping) else None
            if (
                not isinstance(source_files, Mapping)
                or not isinstance(derived_files, Mapping)
                or not isinstance(artifact_rows, list)
            ):
                raise IndependentAuditError("audit_parent_binding", "Managed parent inventory evidence is unavailable.")
            artifact_files = {
                str(raw.get("relative_path") or ""): raw for raw in artifact_rows if isinstance(raw, Mapping)
            }
            parent_cache: dict[str, np.ndarray] = {}
            with (
                open_anchored_directory(source_root, project) as source_anchor,
                open_anchored_directory(derived_root, project) as derived_anchor,
            ):
                for _sprite_id, record, child_rgba in rows:
                    _check_cancelled(cancelled)
                    derivation = record.get("source_derivation")
                    if not isinstance(derivation, Mapping):
                        raise IndependentAuditError("audit_parent_binding", "Derived-source evidence is unavailable.")
                    value = dict(derivation)
                    row_identity = str(value.get("record_identity") or "")
                    if receipt_records.get(row_identity) != value:
                        raise IndependentAuditError(
                            "audit_parent_binding", "Candidate derivation differs from its receipt-bound record."
                        )
                    parent_relative = str(value.get("parent_source_relative_path") or "")
                    parent_binding = source_files.get(parent_relative)
                    artifact_binding = artifact_files.get(parent_relative)
                    if (
                        not _portable_relative_path(parent_relative)
                        or not isinstance(parent_binding, Mapping)
                        or set(parent_binding) != {"sha256", "byte_count"}
                        or not isinstance(artifact_binding, Mapping)
                        or parent_binding.get("sha256") != value.get("parent_source_raw_sha256")
                        or artifact_binding.get("actual_sha256") != value.get("parent_source_raw_sha256")
                        or artifact_binding.get("byte_count") != parent_binding.get("byte_count")
                    ):
                        raise IndependentAuditError(
                            "audit_parent_binding", "A derived parent differs from its raw Harvest binding."
                        )
                    if parent_relative not in parent_cache:
                        byte_count = parent_binding.get("byte_count")
                        if (
                            isinstance(byte_count, bool)
                            or not isinstance(byte_count, int)
                            or not 0 < byte_count <= _MAX_FILE_BYTES
                        ):
                            raise IndependentAuditError(
                                "audit_parent_binding", "A derived parent exceeds independent audit bounds."
                            )
                        parent_content = _read_relative_bound_file(source_anchor, parent_relative, byte_count)
                        if hashlib.sha256(parent_content).hexdigest() != value.get("parent_source_raw_sha256"):
                            raise IndependentAuditError(
                                "audit_parent_binding", "A receipt-bound raw parent changed before audit."
                            )
                        parent_cache[parent_relative] = _decode_parent_png(parent_content)
                    parent_rgba = parent_cache[parent_relative]
                    if _decoded_rgba_identity(parent_rgba) != value.get("parent_source_decoded_rgba_sha256"):
                        raise IndependentAuditError(
                            "audit_parent_binding", "A receipt-bound parent changed its decoded pixels."
                        )
                    crop = value.get("crop_rectangle")
                    if not isinstance(crop, list) or len(crop) != 4:
                        raise IndependentAuditError("audit_parent_binding", "A derived parent crop is malformed.")
                    left, top, right, bottom = crop
                    if not 0 <= left < right <= parent_rgba.shape[1] or not 0 <= top < bottom <= parent_rgba.shape[0]:
                        raise IndependentAuditError(
                            "audit_parent_binding", "A derived parent crop exceeds its decoded source."
                        )
                    reconstructed = np.asarray(parent_rgba[top:bottom, left:right], dtype=np.uint8)
                    if reconstructed.shape != child_rgba.shape or not np.array_equal(reconstructed, child_rgba):
                        raise IndependentAuditError(
                            "audit_parent_binding", "A candidate sprite differs from its receipt-bound parent crop."
                        )
                    output_relative = str(value.get("output_relative_path") or "")
                    derived_binding = derived_files.get(output_relative)
                    derived_size = value.get("encoded_output_byte_count")
                    if (
                        not _portable_relative_path(output_relative)
                        or not isinstance(derived_binding, Mapping)
                        or derived_binding.get("sha256") != value.get("encoded_output_sha256")
                        or derived_binding.get("byte_count") != derived_size
                        or isinstance(derived_size, bool)
                        or not isinstance(derived_size, int)
                        or not 0 < derived_size <= _MAX_FILE_BYTES
                    ):
                        raise IndependentAuditError(
                            "audit_parent_binding", "A derived output differs from its managed inventory."
                        )
                    derived_content = _read_relative_bound_file(
                        derived_anchor,
                        output_relative,
                        derived_size,
                    )
                    canonical_content = _canonical_rgba_png(child_rgba)
                    if derived_content != canonical_content:
                        raise IndependentAuditError(
                            "audit_parent_binding", "A managed derived output differs from its exact parent crop."
                        )
                    completed += 1
                    progress(
                        "parent_rehash",
                        completed,
                        total,
                        f"Rehashed derived parent crop {completed} of {total}.",
                    )


def _read_relative_bound_file(anchor: AnchoredDirectory, relative: str, expected_size: int) -> bytes:
    parts = PurePosixPath(relative).parts
    if not parts:
        raise IndependentAuditError("audit_parent_binding", "A managed parent path is empty.")
    with ExitStack() as stack:
        current = anchor
        for name in parts[:-1]:
            current = stack.enter_context(current.open_directory_immovable(name))
        return _read_bound_file(current, parts[-1], expected_size)


def _decode_parent_png(content: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(content)) as opened:
            width, height = opened.size
            if (
                opened.format != "PNG"
                or getattr(opened, "n_frames", 1) != 1
                or width < 1
                or height < 1
                or width * height > _MAX_DERIVED_PARENT_PIXELS
            ):
                raise IndependentAuditError(
                    "audit_parent_binding", "A receipt-bound parent is not one bounded PNG image."
                )
            opened.load()
            return np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    except IndependentAuditError:
        raise
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise IndependentAuditError("audit_parent_binding", "A receipt-bound parent PNG is invalid.") from exc


def _run_label_audit(
    candidate: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    dataset: Mapping[str, Any],
    *,
    progress: ProgressCallback,
    cancelled: CancellationCallback,
) -> dict[str, Any]:
    subjects = _audit_subjects(candidate, payloads)
    sprites = dataset["sprites"]
    expected_bindings = subjects.get("visual_descriptor_bindings")
    if not isinstance(expected_bindings, list):
        raise IndependentAuditError("audit_subjects", "Visual descriptor subjects are unavailable.")
    expected_by_id = _unique_rows(expected_bindings, "sprite_id", "visual descriptor subjects")
    if set(expected_by_id) != set(sprites):
        raise IndependentAuditError("audit_subjects", "Visual descriptor subjects do not cover every sprite.")
    recomputed: list[dict[str, Any]] = []
    for index, (sprite_id, sprite) in enumerate(sorted(sprites.items()), start=1):
        _check_cancelled(cancelled)
        descriptor = _local_pixel_descriptor(sprite["rgba"])
        record = sprite["record"]
        manifest = sprite["manifest"]
        binding = {
            "sprite_id": sprite_id,
            "descriptor_identity": descriptor["descriptor_identity"],
            "decoded_rgba_sha256": descriptor["decoded_rgba_sha256"],
        }
        if binding != expected_by_id[sprite_id]:
            raise IndependentAuditError("audit_visual_descriptor", "A local pixel descriptor failed recomputation.")
        if record.get("local_pixel_vision") != descriptor:
            raise IndependentAuditError("audit_visual_descriptor", "A conditioned record changed its pixel facts.")
        for source in (record.get("label_evidence"), manifest.get("label_evidence")):
            if not isinstance(source, Mapping) or source.get("local_pixel_vision") != descriptor:
                raise IndependentAuditError("audit_visual_descriptor", "Label evidence lacks recomputed pixel facts.")
            if (
                source.get("local_pixel_vision_algorithm") != LOCAL_PIXEL_VISION_ALGORITHM
                or source.get("local_pixel_vision_config_identity") != LOCAL_PIXEL_VISION_CONFIG_IDENTITY
                or source.get("semantic_category_from_pixels") is not False
                or source.get("human_truth_claim") is not False
            ):
                raise IndependentAuditError("audit_visual_descriptor", "Label evidence overstates local pixel facts.")
        _verify_label_record(record, manifest)
        recomputed.append(binding)
        progress("label_recomputation", index, len(sprites), f"Recomputed label evidence {index} of {len(sprites)}.")
    recomputed_subjects = _recompute_audit_subjects(dataset["records"], recomputed)
    if recomputed_subjects != subjects:
        raise IndependentAuditError("audit_subjects", "Stratified and mandatory label subjects failed recomputation.")
    return {
        "audited_record_ids": subjects["required_label_audit_ids"],
        "stratified_sample_ids": subjects["stratified_sample_ids"],
        "low_confidence_ids": subjects["low_confidence_ids"],
        "disagreement_ids": subjects["disagreement_ids"],
        "high_impact_ids": subjects["high_impact_ids"],
        "generic_label_ids": subjects["generic_label_ids"],
        "distributions": subjects["distributions"],
        "quality_rates_basis_points": subjects["quality_rates_basis_points"],
        "recomputed_visual_descriptor_bindings": recomputed,
        "local_pixel_vision_config_identity": LOCAL_PIXEL_VISION_CONFIG_IDENTITY,
    }


def _run_dataset_validation(
    candidate: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    dataset: Mapping[str, Any],
    *,
    progress: ProgressCallback,
    cancelled: CancellationCallback,
) -> dict[str, Any]:
    sprites = dataset["sprites"]
    records = dataset["records"]
    coverage = _json_mapping(payloads["coverage_report.json"])
    benchmark = _json_mapping(payloads["benchmark_manifest.json"])
    vocabulary = _json_mapping(payloads["conditioning_vocabulary.json"])
    provenance = _json_mapping(payloads["provenance_manifest.json"])
    loader = _json_mapping(payloads["loader_check.json"])
    dataset_qa = _json_mapping(payloads["dataset_qa_report.json"])
    training_qa = _json_mapping(payloads["training_manifest_qa_report.json"])
    count_policy = candidate.get("count_policy")
    if not isinstance(count_policy, Mapping):
        raise IndependentAuditError("audit_count", "The candidate count policy is unavailable.")
    minimum, maximum = count_policy.get("minimum"), count_policy.get("maximum")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or not minimum <= len(sprites) <= maximum
    ):
        raise IndependentAuditError("audit_count", "The loaded candidate count is outside its bound policy.")
    categories = Counter(str(row.get("category") or "") for row in records)
    sources = Counter(str(row.get("source_id") or "") for row in records)
    splits = Counter(str(sprite["split"]) for sprite in sprites.values())
    if (
        dict(sorted(categories.items())) != candidate.get("category_counts")
        or dict(sorted(sources.items())) != candidate.get("source_counts")
        or dict(sorted(splits.items())) != candidate.get("split_counts")
        or coverage.get("image_count") != len(sprites)
        or coverage.get("category_counts") != candidate.get("category_counts")
        or coverage.get("source_counts") != candidate.get("source_counts")
        or coverage.get("split_counts") != candidate.get("split_counts")
    ):
        raise IndependentAuditError("audit_distribution", "Candidate distributions failed independent recomputation.")
    if (
        not isinstance(vocabulary.get("category_to_id"), Mapping)
        or not isinstance(vocabulary.get("object_to_id"), Mapping)
        or vocabulary.get("human_truth_claim") is not False
        or benchmark.get("category_counts") != candidate.get("benchmark_category_counts")
        or benchmark.get("source_group_disjoint_from_training") is not True
        or benchmark.get("duplicate_family_disjoint_from_training") is not True
    ):
        raise IndependentAuditError("audit_vocabulary", "Vocabulary or benchmark bindings failed validation.")
    category_to_id = vocabulary["category_to_id"]
    for sprite in sprites.values():
        category = str(sprite["record"].get("category") or "")
        if category_to_id.get(category) != sprite["category_id"]:
            raise IndependentAuditError("audit_vocabulary", "A category array differs from its vocabulary binding.")
    if (
        provenance.get("all_source_files_rehashed") is not True
        or provenance.get("paths_are_portable") is not True
        or sorted(provenance.get("license_policy") or []) != sorted(_ALLOWED_LICENSES)
        or loader.get("ok") is not True
        or loader.get("checked_all_splits") is not True
        or loader.get("split_counts") != candidate.get("split_counts")
        or dataset_qa.get("errors") not in ([], None)
        or training_qa.get("errors") not in ([], None)
    ):
        raise IndependentAuditError("audit_structural", "Bound structural QA and provenance reports are not clean.")
    training_rows = _jsonl(payloads["training_manifest.jsonl"])
    _validate_training_manifest(training_rows, sprites)
    progress("near_duplicate_recomputation", 0, len(sprites), "Recomputing retained near-duplicate pairs.")
    near_gate, pair_count = _recompute_retained_near_gate(sprites, progress=progress, cancelled=cancelled)
    duplicate_report = _json_mapping(payloads["duplicate_report.json"])
    if (
        duplicate_report.get("near_duplicate_algorithm") != NEAR_DUPLICATE_ALGORITHM
        or duplicate_report.get("near_duplicate_config") != _NEAR_DUPLICATE_CONFIG
        or duplicate_report.get("near_duplicate_config_identity") != NEAR_DUPLICATE_CONFIG_IDENTITY
        or duplicate_report.get("retained_near_duplicate_gate") != near_gate
        or candidate.get("near_duplicate_retained_gate") != near_gate
        or near_gate.get("ok") is not True
    ):
        raise IndependentAuditError("audit_near_duplicate", "Retained near-duplicate evidence failed recomputation.")
    expected_pairs = sum(count * (count - 1) // 2 for count in categories.values())
    if pair_count != expected_pairs:
        raise IndependentAuditError(
            "audit_near_duplicate", "The retained pair audit did not cover every category pair."
        )
    return {
        "split_counts": candidate["split_counts"],
        "category_counts": candidate["category_counts"],
        "source_counts": candidate["source_counts"],
        "benchmark_category_counts": candidate["benchmark_category_counts"],
        "payload_inventory_sha256": candidate["payload_inventory_sha256"],
        "verified_file_count": len(candidate["payload_inventory"]),
        "near_duplicate_recomputation": {
            "algorithm_id": NEAR_DUPLICATE_ALGORITHM,
            "config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
            "retained_count": len(sprites),
            "checked_same_category_pairs": pair_count,
            "violation_count": int(near_gate["violation_count"]),
            "gate_identity": near_gate["gate_identity"],
        },
    }


def _validate_training_manifest(rows: Sequence[Mapping[str, Any]], sprites: Mapping[str, Mapping[str, Any]]) -> None:
    required_keys = {
        "schema_version",
        "sprite_id",
        "split",
        "npz_file",
        "npz_row",
        "category",
        "object_name",
        "label_confidence_tier",
        "base_object",
        "caption",
        "caption_type",
        "caption_source",
        "conditioning",
        "dropout_mask",
        "negative_tags",
        "source",
        "audit",
    }
    optional_keys = {"label_quality"}
    source_keys = {
        "dataset_dir",
        "manifest_file",
        "manifest_row",
        "source_id",
        "source_pack",
        "artist",
        "suitability_status",
        "inference_path",
        "propagation_relation",
    }
    audit_keys = {
        "label_v2_bucket",
        "label_confidence_tier",
        "semantic_schema_version",
        "caption_policy",
        "variant_index",
        "seed",
    }
    conditioning_keys = {
        "semantic_v3",
        "kept_attributes",
        "dropped_attributes",
        "dropout_policy",
        "dropout_ops",
    }
    variants: dict[str, set[int]] = defaultdict(set)
    if len(rows) != len(sprites) * 2:
        raise IndependentAuditError("audit_training_manifest", "Training rows do not provide two variants per sprite.")
    for row in rows:
        if set(row) - optional_keys != required_keys or not set(row).issubset(required_keys | optional_keys):
            raise IndependentAuditError("audit_training_manifest", "A training row has unknown or missing fields.")
        sprite_id = str(row.get("sprite_id") or "")
        sprite = sprites.get(sprite_id)
        source = row.get("source")
        audit = row.get("audit")
        conditioning = row.get("conditioning")
        if (
            sprite is None
            or not isinstance(source, Mapping)
            or set(source) != source_keys
            or not isinstance(audit, Mapping)
            or set(audit) != audit_keys
            or not isinstance(conditioning, Mapping)
            or set(conditioning) - {"semantic_v4"} != conditioning_keys
        ):
            raise IndependentAuditError("audit_training_manifest", "A training row binding is invalid.")
        record = sprite["record"]
        contract = record.get("label_contract")
        if not isinstance(contract, Mapping):
            raise IndependentAuditError("audit_training_manifest", "A training row lacks its label contract.")
        variant_index = audit.get("variant_index")
        if type(variant_index) is not int or variant_index not in {0, 1}:
            raise IndependentAuditError("audit_training_manifest", "A training row variant index is invalid.")
        variants[sprite_id].add(variant_index)
        caption = str(row.get("caption") or "")
        object_text = str(record.get("object_name") or "").replace("_", " ").casefold()
        if (
            row.get("schema_version") != "training_manifest_v1.0"
            or row.get("split") != sprite["split"]
            or row.get("npz_file") != f"{sprite['split']}.npz"
            or row.get("npz_row") != sprite["npz_row"]
            or row.get("category") != record.get("category")
            or row.get("object_name") != record.get("object_name")
            or row.get("label_confidence_tier") != contract.get("confidence")
            or not caption
            or object_text not in caption.casefold()
            or row.get("caption_type") not in {"object", "style_aware", "attribute", "minimal"}
            or not isinstance(row.get("caption_source"), str)
            or row.get("negative_tags") != contract.get("negative_tags")
            or not isinstance(row.get("dropout_mask"), Mapping)
            or source.get("dataset_dir") != "."
            or source.get("manifest_file") != f"manifest_{sprite['split']}.jsonl"
            or source.get("manifest_row") != sprite["manifest_row"]
            or source.get("source_id") != record.get("source_id")
            or source.get("source_pack") != record.get("source_pack")
            or source.get("artist") != record.get("creator")
            or source.get("inference_path") != ""
            or audit.get("label_v2_bucket") != "filename_grounded"
            or audit.get("label_confidence_tier") != contract.get("confidence")
            or audit.get("semantic_schema_version") != "semantic_v3.0"
            or audit.get("caption_policy") != "mixed"
            or audit.get("seed") != 731001
            or conditioning.get("dropout_policy") != "balanced"
            or not isinstance(conditioning.get("dropout_ops"), list)
        ):
            raise IndependentAuditError("audit_training_manifest", "A training row failed exact source/split binding.")
    if set(variants) != set(sprites) or any(values != {0, 1} for values in variants.values()):
        raise IndependentAuditError("audit_training_manifest", "Training variants do not exactly cover every sprite.")


def _audit_subjects(candidate: Mapping[str, Any], payloads: Mapping[str, bytes]) -> dict[str, Any]:
    subjects = candidate.get("label_audit_subjects")
    if not isinstance(subjects, Mapping):
        raise IndependentAuditError("audit_subjects", "The candidate audit subjects are unavailable.")
    value = dict(subjects)
    if _json_mapping(payloads["label_audit_subjects.json"]) != value:
        raise IndependentAuditError("audit_subjects", "Candidate audit subject bytes changed.")
    base = dict(value)
    identity = str(base.pop("subjects_identity", ""))
    if stable_hash(base) != identity or identity != candidate.get("label_audit_subjects_identity"):
        raise IndependentAuditError("audit_subjects", "Candidate audit subject identity is invalid.")
    if (
        value.get("local_pixel_vision_algorithm") != LOCAL_PIXEL_VISION_ALGORITHM
        or value.get("local_pixel_vision_config_identity") != LOCAL_PIXEL_VISION_CONFIG_IDENTITY
        or value.get("all_visual_descriptors_recompute_required") is not True
    ):
        raise IndependentAuditError("audit_subjects", "Candidate pixel-audit coverage is incomplete.")
    return value


def _candidate_source_bindings(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_bindings = candidate.get("input_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise IndependentAuditError("audit_provenance", "Candidate source bindings are unavailable.")
    bindings: dict[str, dict[str, Any]] = {}
    for raw in raw_bindings:
        if not isinstance(raw, Mapping):
            raise IndependentAuditError("audit_provenance", "A candidate source binding is malformed.")
        value = dict(raw)
        source_id = str(value.get("source_id") or "")
        source_document = value.get("source_document")
        license_document = value.get("license_document")
        if (
            not source_id
            or source_id in bindings
            or _RUN_ID.fullmatch(str(value.get("harvest_run_id") or "")) is None
            or not isinstance(source_document, Mapping)
            or not isinstance(license_document, Mapping)
            or source_document.get("source_id") != source_id
            or source_document.get("title") != value.get("title")
            or source_document.get("creator") != value.get("creator")
            or dict(license_document) != value.get("license_evidence")
            or str(license_document.get("identifier") or "").casefold() != value.get("license_id")
        ):
            raise IndependentAuditError("audit_provenance", "A candidate source binding is inconsistent.")
        bindings[source_id] = value
    return bindings


def _verify_label_record(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    category = str(record.get("category") or "")
    object_name = str(record.get("object_name") or "")
    source_relative = str(record.get("source_relative_path") or "")
    license_id = str(record.get("license_id") or "")
    source_sha = str(record.get("source_sha256") or "")
    if category not in _TAXONOMY or not object_name or license_id not in _ALLOWED_LICENSES:
        raise IndependentAuditError("audit_label", "A conditioned label violates taxonomy or license policy.")
    if object_name.casefold() in _GENERIC_OBJECTS:
        raise IndependentAuditError("audit_label", "A generic object label was retained in the candidate.")
    if not _SHA256.fullmatch(source_sha) or not _portable_relative_path(source_relative):
        raise IndependentAuditError("audit_provenance", "A conditioned label has invalid portable provenance.")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise IndependentAuditError("audit_provenance", "A manifest row lacks source provenance.")
    label_evidence = record.get("label_evidence")
    if not isinstance(label_evidence, Mapping) or set(label_evidence) != _LABEL_EVIDENCE_KEYS:
        raise IndependentAuditError("audit_label_evidence", "Label evidence has unknown or missing fields.")
    expected = {
        "source_id": record.get("source_id"),
        "source_pack": record.get("source_pack"),
        "source_group": record.get("source_group"),
        "relative_path": source_relative,
        "sha256": source_sha,
        "byte_count": record.get("source_byte_count"),
        "license_id": license_id,
        "creator": record.get("creator"),
    }
    manifest_source_path = str(manifest.get("source_path") or "")
    if (
        dict(provenance) != expected
        or manifest.get("human_truth_claim") is not False
        or not _portable_relative_path(manifest_source_path)
        or not manifest_source_path.endswith(f"/{source_relative}")
    ):
        raise IndependentAuditError("audit_provenance", "Manifest provenance differs from its conditioned record.")
    if (
        manifest.get("label_evidence") != label_evidence
        or label_evidence.get("source_id") != record.get("source_id")
        or label_evidence.get("source_pack") != record.get("source_pack")
        or label_evidence.get("source_group") != record.get("source_group")
        or label_evidence.get("source_sha256") != source_sha
        or label_evidence.get("source_byte_count") != record.get("source_byte_count")
        or label_evidence.get("license_id") != license_id
        or label_evidence.get("creator") != record.get("creator")
        or label_evidence.get("source_relative_path") != source_relative
        or label_evidence.get("taxonomy_category") != category
        or label_evidence.get("duplicate_family_id") != record.get("duplicate_family_id")
        or label_evidence.get("human_verified") is not False
        or label_evidence.get("claim_scope") != "source_grounded_non_human_proposal"
        or label_evidence.get("evidence_type") != "source_grounding_with_deterministic_local_pixel_facts"
        or label_evidence.get("inference_method") != "conditioned_filename_taxonomy_v1+local_pixel_vision_v1"
        or label_evidence.get("source_path_sha256") != hashlib.sha256(source_relative.encode("utf-8")).hexdigest()
        or not _strict_string_list(label_evidence.get("tokens"), allow_empty=False)
    ):
        raise IndependentAuditError("audit_label_evidence", "Label evidence differs from its exact source binding.")
    contract = record.get("label_contract")
    semantic = record.get("semantic_v3")
    if not isinstance(contract, Mapping) or set(contract) != _LABEL_CONTRACT_KEYS:
        raise IndependentAuditError("audit_label_schema", "A conditioned label contract has unknown or missing fields.")
    if not isinstance(semantic, Mapping) or set(semantic) != _SEMANTIC_KEYS:
        raise IndependentAuditError("audit_semantic_schema", "A semantic-v3 label has unknown or missing fields.")
    attributes = semantic.get("attributes")
    if not isinstance(attributes, Mapping) or set(attributes) != _ATTRIBUTE_KEYS:
        raise IndependentAuditError("audit_semantic_schema", "Semantic-v3 attributes have unknown or missing fields.")
    list_fields = ("tags", "captions", "prompt_phrases", "negative_tags")
    if any(not _strict_string_list(contract.get(key), allow_empty=False) for key in list_fields):
        raise IndependentAuditError("audit_label_schema", "A conditioned label list is empty or invalid.")
    if any(not _strict_string_list(attributes.get(key), allow_empty=True) for key in _ATTRIBUTE_KEYS):
        raise IndependentAuditError("audit_semantic_schema", "A semantic-v3 attribute list is invalid.")
    if any(
        not _strict_string_list(semantic.get(key), allow_empty=key in {"aliases", "warnings"})
        for key in ("aliases", "captions", "prompt_phrases", "negative_tags", "warnings")
    ):
        raise IndependentAuditError("audit_semantic_schema", "A semantic-v3 list is invalid.")
    if (
        contract.get("schema_version") != "spritelab.dataset.conditioned-label-contract.v1"
        or semantic.get("schema_version") != "semantic_v3.0"
        or contract.get("category") != category
        or semantic.get("category") != category
        or contract.get("object_name") != object_name
        or semantic.get("object_name") != object_name
        or manifest.get("category") != category
        or contract.get("captions") != semantic.get("captions")
        or contract.get("prompt_phrases") != semantic.get("prompt_phrases")
        or contract.get("negative_tags") != semantic.get("negative_tags")
        or contract.get("confidence") != "source_grounded_low"
        or contract.get("confidence_reason") != "filename_path_category_with_verified_local_pixel_descriptors"
        or contract.get("disagreement") is not False
        or contract.get("audit_state") != "SOURCE_GROUNDED_REQUIRES_INDEPENDENT_AUDIT"
        or contract.get("human_truth_claim") is not False
        or category not in contract.get("tags", [])
        or "non_human_filename_grounded" not in semantic.get("warnings", [])
        or semantic.get("source_evidence") != record.get("label_evidence")
    ):
        raise IndependentAuditError("audit_label_contract", "A conditioned label is inconsistent or uncertain.")
    object_text = object_name.replace("_", " ").casefold()
    short_description = str(contract.get("short_description") or "").strip()
    if (
        not short_description
        or object_text not in short_description.casefold()
        or not any(object_text in str(value).casefold() for value in contract["captions"])
        or len(contract["captions"]) > 8
        or any(len(str(value)) > 500 for key in list_fields for value in contract[key])
    ):
        raise IndependentAuditError("audit_label_usefulness", "A conditioned label lacks useful bounded text.")
    if _has_true_human_truth_claim(record) or _has_true_human_truth_claim(manifest):
        raise IndependentAuditError("audit_human_truth", "A non-human label is presented as human truth.")


def _verify_source_derivation(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    rgba: np.ndarray,
    source_binding: Mapping[str, Any],
) -> None:
    derivation = record.get("source_derivation")
    if derivation is None:
        if manifest.get("source_derivation") is not None:
            raise IndependentAuditError("audit_provenance", "Direct-source derivation evidence is inconsistent.")
        return
    if not isinstance(derivation, Mapping) or set(derivation) != _DERIVED_SHEET_FRAME_KEYS:
        raise IndependentAuditError("audit_provenance", "Derived-source evidence is malformed.")
    value = dict(derivation)
    if manifest.get("source_derivation") != value:
        raise IndependentAuditError("audit_provenance", "Derived-source record and manifest evidence differ.")
    crop = value.get("crop_rectangle")
    frame_index = value.get("frame_index")
    width = value.get("width")
    height = value.get("height")
    item_id = value.get("dataset_item_id")
    parent = str(value.get("parent_source_relative_path") or "")
    semantic = str(value.get("semantic_relative_path") or "")
    output = str(value.get("output_relative_path") or "")
    record_identity = str(value.get("record_identity") or "")
    record_payload = dict(value)
    record_payload.pop("record_identity", None)
    pixels = np.asarray(rgba, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2:] != (4,):
        raise IndependentAuditError("audit_provenance", "Derived-source pixels have an invalid shape.")
    decoded_sha256 = _decoded_rgba_identity(pixels)
    source_document = source_binding.get("source_document")
    license_document = source_binding.get("license_document")
    if not isinstance(source_document, Mapping) or not isinstance(license_document, Mapping):
        raise IndependentAuditError("audit_provenance", "Derived-source parent bindings are unavailable.")
    expected_group_identity = stable_hash(
        {
            "schema_version": "spritelab.dataset.conditioned-derived-source-group.v1",
            "run_id": source_binding.get("harvest_run_id"),
            "source_id": source_binding.get("source_id"),
            "parent_source_relative_path": parent,
            "parent_source_raw_sha256": value.get("parent_source_raw_sha256"),
        }
    )
    expected_provenance_identity = stable_hash(
        {
            "schema_version": "spritelab.dataset.conditioned-derived-source-provenance.v1",
            "run_id": source_binding.get("harvest_run_id"),
            "source": dict(source_document),
            "license": dict(license_document),
            "parent_source_relative_path": parent,
            "parent_source_raw_sha256": value.get("parent_source_raw_sha256"),
        }
    )
    canonical_png = _canonical_rgba_png(pixels)
    if (
        value.get("schema_version") != "spritelab.dataset.conditioned-derived-sheet-frame.v1"
        or value.get("recipe_version") != "spritelab.dataset.conditioned-derived-sheet-recipe.v1"
        or value.get("recipe_identity") != _DERIVED_SHEET_RECIPE_IDENTITY
        or not isinstance(item_id, str)
        or not item_id
        or not isinstance(crop, list)
        or len(crop) != 4
        or any(isinstance(part, bool) or not isinstance(part, int) for part in crop)
        or not 0 <= crop[0] < crop[2]
        or not 0 <= crop[1] < crop[3]
        or max(crop) > 16_777_216
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
        or isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width != crop[2] - crop[0]
        or height != crop[3] - crop[1]
        or width != pixels.shape[1]
        or height != pixels.shape[0]
        or width * height > 16_777_216
        or not _portable_relative_path(parent)
        or not _portable_relative_path(semantic)
        or not _portable_relative_path(output)
        or value.get("source_derived_not_augmentation") is not True
        or any(
            _SHA256.fullmatch(str(value.get(name) or "")) is None
            for name in (
                "parent_source_raw_sha256",
                "parent_source_decoded_rgba_sha256",
                "decoded_rgba_sha256",
                "source_provenance_identity",
                "source_group_identity",
                "encoded_output_sha256",
                "derivation_identity",
                "record_identity",
            )
        )
        or stable_hash(record_payload) != record_identity
        or semantic != f"{parent}#frame{frame_index:04d}"
        or semantic != record.get("source_relative_path")
        or value.get("encoded_output_sha256") != record.get("source_sha256")
        or value.get("encoded_output_byte_count") != record.get("source_byte_count")
        or value.get("encoded_output_sha256") != hashlib.sha256(canonical_png).hexdigest()
        or value.get("encoded_output_byte_count") != len(canonical_png)
        or value.get("decoded_rgba_sha256") != decoded_sha256
        or value.get("source_group_identity") != expected_group_identity
        or value.get("source_provenance_identity") != expected_provenance_identity
        or value.get("source_group_identity") != record.get("source_group")
        or output != f"frames/{value.get('derivation_identity')}.png"
    ):
        raise IndependentAuditError("audit_provenance", "Derived-source evidence is inconsistent.")
    derivation_payload = {
        "schema_version": "spritelab.dataset.conditioned-derived-sheet-derivation.v1",
        "dataset_item_id": value.get("dataset_item_id"),
        "parent_source_relative_path": value.get("parent_source_relative_path"),
        "parent_source_raw_sha256": value.get("parent_source_raw_sha256"),
        "parent_source_decoded_rgba_sha256": value.get("parent_source_decoded_rgba_sha256"),
        "crop_rectangle": crop,
        "frame_index": frame_index,
        "recipe_identity": value.get("recipe_identity"),
        "decoded_rgba_sha256": value.get("decoded_rgba_sha256"),
        "source_provenance_identity": value.get("source_provenance_identity"),
        "source_group_identity": value.get("source_group_identity"),
    }
    if stable_hash(derivation_payload) != value.get("derivation_identity"):
        raise IndependentAuditError("audit_provenance", "Derived-source derivation identity is inconsistent.")


def _canonical_rgba_png(rgba: np.ndarray) -> bytes:
    pixels = np.asarray(rgba, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2:] != (4,) or pixels.shape[0] < 1 or pixels.shape[1] < 1:
        raise IndependentAuditError("audit_provenance", "Derived-source pixels cannot be canonically encoded.")
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    payload = pixels.tobytes()
    stride = width * 4
    scanlines = b"".join(b"\x00" + payload[offset : offset + stride] for offset in range(0, len(payload), stride))

    def chunk(kind: bytes, content: bytes) -> bytes:
        checksum = zlib.crc32(kind + content) & 0xFFFFFFFF
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )


def _decoded_rgba_identity(rgba: np.ndarray) -> str:
    """Independently reproduce the Dataset-v5 decoded-pixel identity."""

    pixels = np.ascontiguousarray(rgba, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2:] != (4,) or pixels.shape[0] < 1 or pixels.shape[1] < 1:
        raise IndependentAuditError("audit_provenance", "Derived-source pixels have an invalid shape.")
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    canonical = b"decoded_rgba_v1\0" + struct.pack(">II", width, height) + pixels.tobytes()
    return hashlib.sha256(canonical).hexdigest()


def _recompute_audit_subjects(
    records: Sequence[Mapping[str, Any]], visual_bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_category: dict[str, list[str]] = defaultdict(list)
    low_confidence: list[str] = []
    disagreements: list[str] = []
    high_impact: list[str] = []
    generic: list[str] = []
    source_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    confidence_reason_counts: Counter[str] = Counter()
    disagreement_counts: Counter[str] = Counter()
    generic_counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda value: str(value.get("sprite_id") or "")):
        sprite_id = str(record.get("sprite_id") or "")
        category = str(record.get("category") or "")
        object_name = str(record.get("object_name") or "").casefold()
        evidence = record.get("label_evidence")
        contract = record.get("label_contract")
        if not isinstance(evidence, Mapping) or not isinstance(contract, Mapping):
            raise IndependentAuditError("audit_label", "A conditioned record lacks source-grounded evidence.")
        confidence = str(contract.get("confidence") or "unknown")
        confidence_reason = str(contract.get("confidence_reason") or "unknown")
        disagreed = bool(contract.get("disagreement") or evidence.get("disagreement"))
        is_generic = object_name in _GENERIC_OBJECTS
        by_category[category].append(sprite_id)
        source_counts[str(evidence.get("source_id") or "unknown")] += 1
        confidence_counts[confidence] += 1
        confidence_reason_counts[confidence_reason] += 1
        disagreement_counts["disagreement" if disagreed else "no_disagreement"] += 1
        generic_counts["generic" if is_generic else "specific"] += 1
        if confidence in {"source_grounded_low", "low", "unknown"}:
            low_confidence.append(sprite_id)
        if disagreed:
            disagreements.append(sprite_id)
        if category in _HIGH_IMPACT:
            high_impact.append(sprite_id)
        if is_generic:
            generic.append(sprite_id)
    stratified = sorted(
        sprite_id for category in sorted(by_category) for sprite_id in sorted(by_category[category])[:10]
    )
    required = sorted({*stratified, *low_confidence, *disagreements, *high_impact, *generic})
    base = {
        "schema_version": "spritelab.audit.conditioned-subjects.v1",
        "stratified_sample_ids": stratified,
        "low_confidence_ids": sorted(low_confidence),
        "disagreement_ids": sorted(disagreements),
        "high_impact_ids": sorted(high_impact),
        "generic_label_ids": sorted(generic),
        "required_label_audit_ids": required,
        "visual_descriptor_bindings": [dict(value) for value in visual_bindings],
        "local_pixel_vision_algorithm": LOCAL_PIXEL_VISION_ALGORITHM,
        "local_pixel_vision_config_identity": LOCAL_PIXEL_VISION_CONFIG_IDENTITY,
        "distributions": {
            "category": {key: len(by_category[key]) for key in sorted(by_category)},
            "source": dict(sorted(source_counts.items())),
            "confidence": dict(sorted(confidence_counts.items())),
            "confidence_reason": dict(sorted(confidence_reason_counts.items())),
            "disagreement": dict(sorted(disagreement_counts.items())),
            "generic_label": dict(sorted(generic_counts.items())),
        },
        "all_low_confidence_required": True,
        "all_disagreements_required": True,
        "all_high_impact_required": True,
        "all_generic_labels_required": True,
        "all_visual_descriptors_recompute_required": True,
        "quality_rates_basis_points": {
            "unknown_category": (
                sum(1 for record in records if str(record.get("category") or "") == "unknown") * 10_000
                + max(1, len(records)) // 2
            )
            // max(1, len(records)),
            "generic_object": (len(generic) * 10_000 + max(1, len(records)) // 2) // max(1, len(records)),
            "disagreement": (len(disagreements) * 10_000 + max(1, len(records)) // 2) // max(1, len(records)),
            "useful_label": ((len(records) - len(generic)) * 10_000 + max(1, len(records)) // 2)
            // max(1, len(records)),
        },
        "human_truth_claim": False,
    }
    return {**base, "subjects_identity": stable_hash(base)}


def _local_pixel_descriptor(rgba: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(rgba, dtype=np.uint8)
    if pixels.shape != (32, 32, 4):
        raise IndependentAuditError("audit_visual_shape", "Local pixel recomputation requires exact 32x32 RGBA.")
    mask = pixels[:, :, 3] == 255
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise IndependentAuditError("audit_visual_blank", "A candidate sprite is fully transparent.")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bbox_width, bbox_height = right - left, bottom - top
    foreground_pixels = int(mask.sum())
    occupancy_bp = (foreground_pixels * 10_000 + 512) // 1024
    palette = _LOCAL_PIXEL_VISION_CONFIG["dominant_palette"]
    prototypes = np.asarray([entry[1] for entry in palette], dtype=np.int32)
    visible_rgb = pixels[mask, :3].astype(np.int32)
    distances = ((visible_rgb[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    assignments = np.argmin(distances, axis=1)
    color_counts = np.bincount(assignments, minlength=len(palette))
    dominant_index = int(np.argmax(color_counts))
    dominant_color = str(palette[dominant_index][0])
    dominant_share = (int(color_counts[dominant_index]) * 10_000 + foreground_pixels // 2) // foreground_pixels
    bbox_max = max(bbox_width, bbox_height)
    thresholds = _LOCAL_PIXEL_VISION_CONFIG["scale_max_bbox_dimension"]
    if bbox_max <= int(thresholds["tiny"]):
        scale = "tiny"
    elif bbox_max <= int(thresholds["small"]):
        scale = "small"
    elif bbox_max <= int(thresholds["medium"]):
        scale = "medium"
    elif bbox_max <= int(thresholds["large"]):
        scale = "large"
    else:
        scale = "full_canvas"
    horizontal_offset = left + right - 32
    vertical_offset = top + bottom - 32
    horizontal_position = (
        "horizontally_centered"
        if abs(horizontal_offset) <= 2
        else "left_weighted"
        if horizontal_offset < 0
        else "right_weighted"
    )
    vertical_position = (
        "vertically_centered"
        if abs(vertical_offset) <= 2
        else "top_weighted"
        if vertical_offset < 0
        else "bottom_weighted"
    )
    symmetry_pixels = int(np.count_nonzero(mask != np.fliplr(mask)))
    symmetry_bp = (symmetry_pixels * 10_000 + 512) // 1024
    symmetry_thresholds = _LOCAL_PIXEL_VISION_CONFIG["symmetry_mismatch_basis_points"]
    symmetry = (
        "high_horizontal_symmetry"
        if symmetry_bp <= int(symmetry_thresholds["high"])
        else "moderate_horizontal_symmetry"
        if symmetry_bp <= int(symmetry_thresholds["moderate"])
        else "asymmetric_silhouette"
    )
    padded = np.pad(mask, 1, constant_values=False)
    interior = padded[1:-1, 1:-1]
    surrounded = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary_pixels = int(np.count_nonzero(interior & ~surrounded))
    edge_bp = (boundary_pixels * 10_000 + foreground_pixels // 2) // foreground_pixels
    edge_thresholds = _LOCAL_PIXEL_VISION_CONFIG["edge_density_basis_points"]
    edge = (
        "low_edge_density"
        if edge_bp <= int(edge_thresholds["low"])
        else "medium_edge_density"
        if edge_bp <= int(edge_thresholds["medium"])
        else "high_edge_density"
    )
    occupancy = (
        "sparse_occupancy"
        if occupancy_bp < 1500
        else "balanced_occupancy"
        if occupancy_bp < 6000
        else "dense_occupancy"
    )
    tags = [
        f"dominant_{dominant_color}",
        f"{scale}_silhouette",
        horizontal_position,
        vertical_position,
        symmetry,
        edge,
        occupancy,
    ]
    metrics = {
        "alpha_bbox": [left, top, right, bottom],
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "foreground_pixels": foreground_pixels,
        "alpha_occupancy_basis_points": occupancy_bp,
        "dominant_coarse_color": dominant_color,
        "dominant_color_share_basis_points": dominant_share,
        "silhouette_scale": scale,
        "horizontal_offset_half_pixels": horizontal_offset,
        "vertical_offset_half_pixels": vertical_offset,
        "horizontal_position": horizontal_position,
        "vertical_position": vertical_position,
        "horizontal_symmetry_mismatch_pixels": symmetry_pixels,
        "horizontal_symmetry_mismatch_basis_points": symmetry_bp,
        "horizontal_symmetry": symmetry,
        "boundary_pixels": boundary_pixels,
        "edge_density_basis_points": edge_bp,
        "edge_density": edge,
        "occupancy": occupancy,
    }
    payload = {
        "schema_version": "spritelab.dataset.local-pixel-vision.v1",
        "algorithm_id": LOCAL_PIXEL_VISION_ALGORITHM,
        "config_identity": LOCAL_PIXEL_VISION_CONFIG_IDENTITY,
        "decoded_rgba_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "metrics": metrics,
        "visual_tags": tags,
        "semantic_category_inferred": False,
        "provider_contacted": False,
        "model_weights_loaded": False,
    }
    return {**payload, "descriptor_identity": stable_hash(payload)}


def _recompute_retained_near_gate(
    sprites: Mapping[str, Mapping[str, Any]],
    *,
    progress: ProgressCallback,
    cancelled: CancellationCallback,
) -> tuple[dict[str, Any], int]:
    prepared: list[dict[str, Any]] = []
    for sprite_id, sprite in sorted(sprites.items()):
        rgba = np.asarray(sprite["rgba"], dtype=np.uint8)
        descriptor = _local_pixel_descriptor(rgba)
        alpha = rgba[:, :, 3] == 255
        prepared.append(
            {
                "sprite_id": sprite_id,
                "category": str(sprite["record"].get("category") or ""),
                "alpha_bitmap": np.packbits(alpha).tobytes(),
                "bbox": tuple(int(value) for value in descriptor["metrics"]["alpha_bbox"]),
                "perceptual_hash": _perceptual_hash(rgba),
            }
        )
    violations: list[dict[str, Any]] = []
    pair_count = 0
    processed_left = 0
    for index, left in enumerate(prepared):
        _check_cancelled(cancelled)
        for right in prepared[index + 1 :]:
            if left["category"] != right["category"]:
                continue
            pair_count += 1
            metrics = _near_metrics(left, right)
            if metrics["is_near_duplicate"] is True:
                violations.append(
                    {
                        "left_record_key": left["sprite_id"],
                        "right_record_key": right["sprite_id"],
                        "metric_evidence": metrics,
                    }
                )
        processed_left += 1
        progress(
            "near_duplicate_recomputation",
            processed_left,
            len(prepared),
            f"Recomputed retained pairs for {processed_left} of {len(prepared)} sprites.",
        )
    payload = {
        "algorithm_id": NEAR_DUPLICATE_ALGORITHM,
        "config": _NEAR_DUPLICATE_CONFIG,
        "config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
        "retained_count": len(prepared),
        "violation_count": len(violations),
        "violations": violations,
        "ok": not violations,
    }
    return {**payload, "gate_identity": stable_hash(payload)}, pair_count


def _near_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_bbox = left["bbox"]
    right_bbox = right["bbox"]
    width_delta = abs((left_bbox[2] - left_bbox[0]) - (right_bbox[2] - right_bbox[0]))
    height_delta = abs((left_bbox[3] - left_bbox[1]) - (right_bbox[3] - right_bbox[1]))
    center_x_delta = abs((left_bbox[0] + left_bbox[2]) - (right_bbox[0] + right_bbox[2]))
    center_y_delta = abs((left_bbox[1] + left_bbox[3]) - (right_bbox[1] + right_bbox[3]))
    alpha_xor = sum(
        byte.bit_count()
        for byte in bytes(a ^ b for a, b in zip(left["alpha_bitmap"], right["alpha_bitmap"], strict=True))
    )
    perceptual_hamming = (int(left["perceptual_hash"]) ^ int(right["perceptual_hash"])).bit_count()
    same_category = left["category"] == right["category"]
    is_near = (
        same_category
        and perceptual_hamming <= int(_NEAR_DUPLICATE_CONFIG["max_perceptual_hamming"])
        and width_delta <= int(_NEAR_DUPLICATE_CONFIG["max_bbox_dimension_delta"])
        and height_delta <= int(_NEAR_DUPLICATE_CONFIG["max_bbox_dimension_delta"])
        and center_x_delta <= int(_NEAR_DUPLICATE_CONFIG["max_bbox_center_delta_half_pixels"])
        and center_y_delta <= int(_NEAR_DUPLICATE_CONFIG["max_bbox_center_delta_half_pixels"])
        and alpha_xor <= int(_NEAR_DUPLICATE_CONFIG["max_alpha_xor_pixels"])
    )
    return {
        "algorithm_id": NEAR_DUPLICATE_ALGORITHM,
        "config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
        "same_taxonomy_category": same_category,
        "perceptual_hamming": perceptual_hamming,
        "bbox_width_delta": width_delta,
        "bbox_height_delta": height_delta,
        "bbox_center_x_delta_half_pixels": center_x_delta,
        "bbox_center_y_delta_half_pixels": center_y_delta,
        "alpha_xor_pixels": alpha_xor,
        "is_near_duplicate": is_near,
    }


def _perceptual_hash(rgba: np.ndarray) -> int:
    image = (
        Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").convert("L").resize((9, 8), Image.Resampling.BOX)
    )
    values = np.asarray(image, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _report(
    kind: str,
    schema: str,
    gates: frozenset[str],
    candidate: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    implementation = trusted_auditor_inventory(kind)
    report = {
        "schema_version": schema,
        "verdict": "PASS",
        "independent": True,
        "generated_by_conditioned_workflow": False,
        "auditor": {
            "auditor_id": TRUSTED_AUDITOR_IDS[kind],
            "code_identity_sha256": implementation["inventory_sha256"],
            "implementation_inventory": implementation,
        },
        "bindings": {
            "candidate_identity": candidate["candidate_identity"],
            "payload_inventory_sha256": candidate["payload_inventory_sha256"],
            "image_count": candidate["image_count"],
            "production_code_identity": candidate["production_code_identity"],
            "label_audit_subjects_identity": candidate["label_audit_subjects_identity"],
        },
        "subject_files": candidate["payload_inventory"],
        "checks": dict.fromkeys(sorted(gates), "PASS"),
        "audit_subjects": candidate["label_audit_subjects"],
        "metrics": dict(metrics),
    }
    return {**report, "audit_run_identity": stable_hash(report)}


def _json_value(payload: bytes) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, IndependentAuditError) as exc:
        if isinstance(exc, IndependentAuditError):
            raise
        raise IndependentAuditError("audit_json", "A candidate JSON document is invalid.") from exc


def _json_mapping(payload: bytes) -> dict[str, Any]:
    value = _json_value(payload)
    if not isinstance(value, dict):
        raise IndependentAuditError("audit_json", "A candidate JSON document is not an object.")
    return value


def _jsonl(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise IndependentAuditError("audit_jsonl", "A candidate JSONL document is invalid.") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, IndependentAuditError) as exc:
            if isinstance(exc, IndependentAuditError):
                raise
            raise IndependentAuditError("audit_jsonl", "A candidate JSONL row is invalid.") from exc
        if not isinstance(value, dict):
            raise IndependentAuditError("audit_jsonl", "A candidate JSONL row is not an object.")
        rows.append(value)
    return rows


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentAuditError("audit_json_duplicate_key", "A candidate JSON object repeats a key.")
        result[key] = value
    return result


def _unique_rows(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity or identity in result:
            raise IndependentAuditError("audit_row_identity", f"The {label} contain an invalid duplicate identity.")
        result[identity] = dict(row)
    return result


def _strict_string_list(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item.strip() and item == item.strip() for item in value)
    )


def _assert_portable(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_portable(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str) and _is_private_path(value):
        raise IndependentAuditError("audit_private_path", "A candidate artifact contains an absolute private path.")


def _is_private_path(value: str) -> bool:
    text = value.strip()
    return bool(text.startswith(("/", "\\", "file:")) or _WINDOWS_DRIVE.match(text) or "\\" in text or "\x00" in text)


def _portable_relative_path(value: str) -> bool:
    return is_portable_relative_path(value)


def _has_true_human_truth_claim(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("human_truth_claim") is True:
            return True
        return any(_has_true_human_truth_claim(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_true_human_truth_claim(item) for item in value)
    return False


def _check_cancelled(cancelled: CancellationCallback) -> None:
    if cancelled():
        raise IndependentAuditCancelled()


__all__ = [
    "IndependentAuditCancelled",
    "IndependentAuditError",
    "run_independent_audit",
]
