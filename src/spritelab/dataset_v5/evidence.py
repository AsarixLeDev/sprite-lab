"""Deterministic evidence for a Dataset-v5 rebuild blocked before labeling.

This module intentionally performs no provider calls and creates no dataset
root.  It turns the forensic raw-source inventory into auditable, immutable
evidence while preserving source-facing names only in the provenance
manifest.  A caller may provide the read-only source root to re-decode the
original bytes; otherwise pixel-derived claims are explicitly marked as
unverified inventory observations.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from spritelab.dataset_v5.blind import (
    BLIND_REQUEST_SCHEMA_VERSION,
    SOL_ADJUDICATION_PROMPT_VERSION,
    SOL_CONSISTENCY_PROMPT_VERSION,
    deterministic_pixel_facts,
    deterministic_png_bytes,
)
from spritelab.dataset_v5.canary import unavailable_canary_report
from spritelab.dataset_v5.identity import (
    RecordBinding,
    canonical_json_bytes,
    canonical_rgba_bytes,
    decoded_rgba_sha256,
    make_geometry_family_id,
    make_record_id,
    relation_membership_key,
)
from spritelab.dataset_v5.raw_freeze import FreezeGateEvidence
from spritelab.dataset_v5.sol import SOL_PROVIDER_SCHEMA_VERSION
from spritelab.harvest.suitability import SuitabilityInput, audit_inputs, load_config
from spritelab.utils.safe_fs import remove_confined_tree

EVIDENCE_SCHEMA_VERSION = "sprite_lab_v5_blocked_evidence_v1"
SOL_UNAVAILABLE = "SOL_MODEL_UNAVAILABLE"
VIEW_NAMES = (
    "v5_debug",
    "v5_architecture",
    "v5_scale_check",
    "v5_eval_balanced",
    "v5_source_ood",
    "v5_open_set",
    "v5_unlabeled",
)

_MACHINE_FILES = (
    "extraction_manifest.jsonl",
    "blob_manifest.jsonl",
    "provenance_manifest.jsonl",
    "suitability_manifest.jsonl",
    "relation_manifest.jsonl",
    "rebuild_report.json",
    "rebuild_report.md",
    "sol_canary_report.json",
    "label_health_report.json",
    "label_drift_report.json",
    "source_conflict_report.json",
    "freeze_manifest.json",
    "frozen_hash_verification.json",
)
_MACHINE_DIRECTORIES = (
    "raw_audit_blobs",
    "candidate_view_manifests",
    "contact_sheets",
    "batch_audit_reports",
)
_SHA256_CHARS = frozenset("0123456789abcdef")


class EvidenceCompileError(RuntimeError):
    """Raised when blocked evidence cannot be produced without guessing."""


@dataclass(frozen=True)
class _DecodedRecord:
    extraction: Mapping[str, Any]
    provenance: Mapping[str, Any]
    rgba: np.ndarray | None
    blob_bytes: bytes | None


@dataclass(frozen=True)
class _BuildPass:
    extraction_rows: tuple[Mapping[str, Any], ...]
    blob_rows: tuple[Mapping[str, Any], ...]
    provenance_rows: tuple[Mapping[str, Any], ...]
    suitability_rows: tuple[Mapping[str, Any], ...]
    relation_rows: tuple[Mapping[str, Any], ...]
    blobs: Mapping[str, bytes]
    decoded_count: int
    decode_failure_count: int
    source_verified_count: int

    def comparison_value(self) -> Mapping[str, Any]:
        """Return every deterministic product used by the evidence build."""

        return {
            "blob_rows": list(self.blob_rows),
            "blobs": {key: hashlib.sha256(value).hexdigest() for key, value in sorted(self.blobs.items())},
            "decode_failure_count": self.decode_failure_count,
            "decoded_count": self.decoded_count,
            "extraction_rows": list(self.extraction_rows),
            "provenance_rows": list(self.provenance_rows),
            "relation_rows": list(self.relation_rows),
            "source_verified_count": self.source_verified_count,
            "suitability_rows": list(self.suitability_rows),
        }


def compile_blocked_evidence(
    experiment_root: str | Path,
    *,
    source_root: str | Path | None = None,
    historical_reproduction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile fresh provider-free evidence and refuse every overwrite.

    ``experiment_root`` must already contain ``raw_source_inventory.jsonl``
    and ``source_archive_hashes.json``.  When ``source_root`` is supplied,
    every resolvable original is hash-checked and decoded twice.  Regardless
    of that read-only audit's result, this compiler leaves the raw source gate
    closed because it is used specifically for a blocked rebuild run.
    """

    root = Path(experiment_root)
    inventory_path = root / "raw_source_inventory.jsonl"
    archive_hash_path = root / "source_archive_hashes.json"
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    if not archive_hash_path.is_file():
        raise FileNotFoundError(archive_hash_path)
    _require_targets_absent(root)

    inventory_rows = _read_jsonl(inventory_path)
    if not inventory_rows:
        raise EvidenceCompileError("raw source inventory is empty")
    archive_document = _read_json_object(archive_hash_path)
    archive_hashes = _archive_hash_bindings(archive_document)
    reproduction = _canonical_mapping(historical_reproduction or {})

    resolved_source_root = Path(source_root).resolve() if source_root is not None else None
    if resolved_source_root is not None and not resolved_source_root.is_dir():
        raise FileNotFoundError(resolved_source_root)

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.evidence-", dir=root.parent))
    try:
        with tempfile.TemporaryDirectory(prefix="spritelab-evidence-pass-a-") as first_temp:
            first = _build_pass(
                inventory_rows,
                archive_hashes,
                source_root=resolved_source_root,
                temporary_root=Path(first_temp),
            )
        with tempfile.TemporaryDirectory(prefix="spritelab-evidence-pass-b-") as second_temp:
            second = _build_pass(
                inventory_rows,
                archive_hashes,
                source_root=resolved_source_root,
                temporary_root=Path(second_temp),
            )
        _verify_two_passes(first, second)
        _write_staging_artifacts(
            staging,
            inventory_rows=inventory_rows,
            build=first,
            source_root_supplied=resolved_source_root is not None,
            historical_reproduction=reproduction,
        )
        _copy_staging_exclusively(staging, root)
    finally:
        remove_confined_tree(staging, root.parent, missing_ok=True)

    report = _read_json_object(root / "rebuild_report.json")
    return {
        "artifact_count": len(_MACHINE_FILES) + len(_MACHINE_DIRECTORIES),
        "candidate_dataset_created": False,
        "decoded_record_count": report["raw_audit"]["decoded_record_count"],
        "experiment_root": str(root),
        "production_frozen": False,
        "provider_calls": 0,
        "raw_source_gate_passed": False,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "sol_status": SOL_UNAVAILABLE,
        "training_authorized": False,
    }


def _build_pass(
    rows: Sequence[Mapping[str, Any]],
    archive_hashes: Mapping[str, str],
    *,
    source_root: Path | None,
    temporary_root: Path,
) -> _BuildPass:
    resolver = _OriginalResolver(source_root, archive_hashes) if source_root is not None else None
    decoded: list[_DecodedRecord] = []
    for row in rows:
        decoded.append(_decode_inventory_row(row, resolver))

    extraction_rows = sorted((dict(item.extraction) for item in decoded), key=_record_sort_key)
    provenance_rows = sorted((dict(item.provenance) for item in decoded), key=_provenance_sort_key)
    materialized = [item for item in decoded if item.rgba is not None]
    blobs: dict[str, bytes] = {}
    blob_dimensions: dict[str, tuple[int, int]] = {}
    for item in materialized:
        assert item.blob_bytes is not None
        blob_id = str(item.extraction["blob_id"])
        previous = blobs.setdefault(blob_id, item.blob_bytes)
        if previous != item.blob_bytes:
            raise EvidenceCompileError(f"decoded blob identity collision: {blob_id}")
        dimensions = (int(item.extraction["width"]), int(item.extraction["height"]))
        previous_dimensions = blob_dimensions.setdefault(blob_id, dimensions)
        if previous_dimensions != dimensions:
            raise EvidenceCompileError(f"decoded blob dimensions conflict: {blob_id}")

    blob_rows: list[dict[str, Any]] = []
    for blob_id, value in sorted(blobs.items()):
        width, height = blob_dimensions[blob_id]
        blob_rows.append(
            {
                "blob_file_sha256": hashlib.sha256(value).hexdigest(),
                "blob_id": blob_id,
                "blob_path": f"raw_audit_blobs/{blob_id}.rgba",
                "byte_length": len(value),
                "encoding": "identity.canonical_rgba_bytes/decoded_rgba_v1",
                "height": height,
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "width": width,
            }
        )

    suitability_rows = _suitability_audit(materialized, temporary_root)
    relation_rows = _relation_manifest(extraction_rows, suitability_rows, provenance_rows)
    return _BuildPass(
        extraction_rows=tuple(extraction_rows),
        blob_rows=tuple(blob_rows),
        provenance_rows=tuple(provenance_rows),
        suitability_rows=tuple(suitability_rows),
        relation_rows=tuple(relation_rows),
        blobs=blobs,
        decoded_count=len(materialized),
        decode_failure_count=sum(row.get("decode_status") == "not_decodable" for row in extraction_rows),
        source_verified_count=sum(row.get("source_bytes_verified") is True for row in extraction_rows),
    )


class _OriginalResolver:
    """Read and verify immutable source bytes without exposing names downstream."""

    def __init__(self, source_root: Path, archive_hashes: Mapping[str, str]) -> None:
        self.source_root = source_root
        self.archive_hashes = archive_hashes
        self._verified_paths: dict[str, Path] = {}
        self._zip_payloads: dict[str, Mapping[str, bytes]] = {}

    def read(self, row: Mapping[str, Any]) -> bytes:
        display_path = _required_text(row, "original_archive_path")
        expected_hash = _required_sha256(row.get("original_archive_sha256"), "original_archive_sha256")
        declared_hash = self.archive_hashes.get(_canonical_display_path(display_path))
        if declared_hash is None:
            raise EvidenceCompileError(f"source archive hash manifest has no binding for {display_path!r}")
        if declared_hash != expected_hash:
            raise EvidenceCompileError(f"inventory/archive hash manifest conflict for {display_path!r}")
        source_path = self._verified_path(display_path, expected_hash)
        record_type = str(row.get("record_type") or "")
        member = row.get("archive_member_path")
        if record_type == "archive_member_image" or member is not None:
            member_name = _safe_member_name(_required_text(row, "archive_member_path"))
            payloads = self._zip_members(display_path, source_path)
            try:
                payload = payloads[member_name]
            except KeyError as exc:
                raise EvidenceCompileError(f"archive member missing: {member_name}") from exc
        else:
            payload = source_path.read_bytes()
        expected_byte_hash = _required_sha256(row.get("original_byte_sha256"), "original_byte_sha256")
        observed_byte_hash = hashlib.sha256(payload).hexdigest()
        if observed_byte_hash != expected_byte_hash:
            raise EvidenceCompileError(
                f"original byte hash mismatch: expected {expected_byte_hash}, observed {observed_byte_hash}"
            )
        expected_size = row.get("original_byte_size")
        if isinstance(expected_size, int) and len(payload) != expected_size:
            raise EvidenceCompileError(
                f"original byte size mismatch: expected {expected_size}, observed {len(payload)}"
            )
        return payload

    def _verified_path(self, display_path: str, expected_hash: str) -> Path:
        key = _canonical_display_path(display_path)
        cached = self._verified_paths.get(key)
        if cached is not None:
            return cached
        # Rebuilding through validated POSIX parts works for a forensic
        # inventory produced on any host and rejects parent traversal.
        parts = PurePosixPath(_canonical_display_path(display_path)).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise EvidenceCompileError(f"unsafe source path: {display_path!r}")
        path = self.source_root.joinpath(*parts).resolve()
        try:
            path.relative_to(self.source_root)
        except ValueError as exc:
            raise EvidenceCompileError(f"source path leaves source root: {display_path!r}") from exc
        if not path.is_file():
            raise EvidenceCompileError(f"original source missing: {display_path!r}")
        observed = _file_sha256(path)
        if observed != expected_hash:
            raise EvidenceCompileError(
                f"source archive changed: expected {expected_hash}, observed {observed}, path={display_path!r}"
            )
        self._verified_paths[key] = path
        return path

    def _zip_members(self, display_path: str, source_path: Path) -> Mapping[str, bytes]:
        key = _canonical_display_path(display_path)
        cached = self._zip_payloads.get(key)
        if cached is not None:
            return cached
        try:
            with zipfile.ZipFile(source_path) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise EvidenceCompileError(f"ZIP CRC check failed for {bad!r}")
                payloads: dict[str, bytes] = {}
                folded: dict[str, str] = {}
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    name = _safe_member_name(info.filename)
                    if info.flag_bits & 0x1:
                        raise EvidenceCompileError(f"encrypted ZIP member is unsupported: {name}")
                    previous = folded.setdefault(name.casefold(), name)
                    if previous != name or name in payloads:
                        raise EvidenceCompileError(f"ambiguous ZIP member name: {name!r}")
                    payloads[name] = archive.read(info)
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
            raise EvidenceCompileError(f"cannot verify ZIP archive {display_path!r}: {exc}") from exc
        self._zip_payloads[key] = payloads
        return payloads


def _decode_inventory_row(row: Mapping[str, Any], resolver: _OriginalResolver | None) -> _DecodedRecord:
    decision = str(row.get("inclusion_decision") or "quarantine")
    if decision not in {"accept", "quarantine", "reject"}:
        raise EvidenceCompileError(f"invalid forensic inclusion decision: {decision!r}")
    expected_decoded = _optional_sha256(
        row.get("output_decoded_rgba_sha256") or row.get("decoded_image_sha256"),
        "output_decoded_rgba_sha256",
    )
    source_hash = _optional_sha256(row.get("original_archive_sha256"), "original_archive_sha256")
    can_resolve = source_hash is not None and bool(row.get("original_archive_path"))

    rgba: np.ndarray | None = None
    blob_bytes: bytes | None = None
    decode_error: str | None = None
    source_verified = False
    if resolver is not None and can_resolve:
        payload = resolver.read(row)
        source_verified = True
        try:
            rgba = _decode_rgba(payload)
        except EvidenceCompileError as exc:
            decode_error = str(exc)
            if expected_decoded is not None:
                raise
        if rgba is not None:
            rgba = _apply_recorded_transform(rgba, row)
            observed_decoded = decoded_rgba_sha256(rgba)
            if expected_decoded is not None and observed_decoded != expected_decoded:
                raise EvidenceCompileError(
                    f"decoded RGBA hash mismatch: expected {expected_decoded}, observed {observed_decoded}"
                )
            _verify_dimensions(row, rgba)
            expected_decoded = observed_decoded
            blob_bytes = canonical_rgba_bytes(rgba)

    record_id: str | None = None
    geometry_id: str | None = None
    pixel_facts: Mapping[str, Any] | None = None
    width = row.get("width")
    height = row.get("height")
    if expected_decoded is not None and source_hash is not None:
        binding = RecordBinding(
            source_archive_sha256=source_hash,
            archive_member_path=_binding_member_path(row),
            extraction_operation=f"forensic_inventory_v1:{row.get('extraction_operation') or 'unresolved'}",
            crop_coordinates=_crop_tuple(row.get("crop_coordinates")),
            decoded_rgba_sha256=expected_decoded,
            padding_operation=_padding_binding(row.get("padding_operation")),
        )
        record_id = make_record_id(binding)
    if rgba is not None:
        height, width = rgba.shape[:2]
        geometry_id = make_geometry_family_id(rgba[:, :, 3])
        pixel_facts = deterministic_pixel_facts(rgba)

    extraction = {
        "blob_id": expected_decoded,
        "crop_coordinates": _json_crop(row.get("crop_coordinates")),
        "decode_error": decode_error,
        "decode_status": (
            "verified_from_original"
            if rgba is not None
            else "inventory_observation_unverified"
            if expected_decoded is not None
            else "not_decodable"
        ),
        "deterministic_pixel_facts": pixel_facts,
        "forensic_inclusion_decision": decision,
        "geometry_family_id": geometry_id,
        "height": int(height) if isinstance(height, int) else None,
        "interpolation_policy": "none",
        "padding_operation": _nonsemantic_padding(row.get("padding_operation")),
        "record_id": record_id,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_bytes_verified": source_verified,
        "width": int(width) if isinstance(width, int) else None,
    }
    provenance = {
        "archive_member_path": row.get("archive_member_path"),
        "audit_issues": _sorted_strings(row.get("audit_issues")),
        "creator_or_publishers": _sorted_strings(row.get("creator_or_publishers")),
        "declared_sheet_ids": _declared_sheet_ids(row),
        "licenses": row.get("licenses") if isinstance(row.get("licenses"), list) else [],
        "original_archive_path": row.get("original_archive_path"),
        "original_archive_sha256": source_hash,
        "original_byte_sha256": row.get("original_byte_sha256"),
        "original_filename": row.get("original_filename"),
        "packs": _sorted_strings(row.get("packs")),
        "provenance_issues": _sorted_strings(row.get("provenance_issues")),
        "provenance_status": row.get("provenance_status"),
        "record_id": record_id,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_bindings": row.get("source_bindings") if isinstance(row.get("source_bindings"), list) else [],
        "source_urls": _sorted_strings(row.get("source_urls")),
    }
    return _DecodedRecord(extraction=extraction, provenance=provenance, rgba=rgba, blob_bytes=blob_bytes)


def _suitability_audit(records: Sequence[_DecodedRecord], temporary_root: Path) -> list[dict[str, Any]]:
    if not records:
        return []
    image_root = temporary_root / "opaque_images"
    image_root.mkdir(parents=True, exist_ok=False)
    inputs: list[SuitabilityInput] = []
    decisions: dict[str, str] = {}
    for item in sorted(records, key=lambda value: str(value.extraction["record_id"])):
        record_id = str(item.extraction["record_id"])
        if not record_id.startswith("rec_"):
            raise EvidenceCompileError("a decoded record lacks an opaque canonical identity")
        assert item.rgba is not None
        image_path = image_root / f"{record_id}.png"
        image_path.write_bytes(deterministic_png_bytes(item.rgba))
        inputs.append(SuitabilityInput(sprite_id=record_id, image_path=image_path))
        decisions[record_id] = str(item.extraction["forensic_inclusion_decision"])
    output = audit_inputs(inputs, load_config("single_object_source_resolution"))
    return [
        {
            "audit_status": result.status,
            "forensic_inclusion_decision": decisions[result.sprite_id],
            "metrics": result.metrics,
            "reason_codes": result.reason_codes,
            "record_id": result.sprite_id,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "score": result.score,
            "suitability_config_hash": result.config_hash,
            "suitability_profile": result.profile,
            "warnings": result.warnings,
        }
        for result in output.results
    ]


def _relation_manifest(
    extraction_rows: Sequence[Mapping[str, Any]],
    suitability_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in extraction_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            continue
        for kind, field in (("exact_rgba", "blob_id"), ("geometry_family", "geometry_family_id")):
            value = row.get(field)
            if isinstance(value, str):
                groups[(kind, value)].append(record_id)
    output: list[dict[str, Any]] = []
    for (kind, _value), member_ids in sorted(groups.items()):
        members = sorted(set(member_ids))
        if kind == "exact_rgba" and len(members) < 2:
            continue
        membership = relation_membership_key(members)
        output.append(_relation_row(kind, members, membership=membership, hard=True))

    duplicate_groups: dict[tuple[str, tuple[str, ...]], None] = {}
    kind_map = {
        "exact_alpha_mask": "alpha_recolor",
        "exact_rgba": "exact_rgba",
        "near_identical": "near_identical",
        "padded_or_translated": "translation",
        "recolor_geometry": "alpha_recolor",
        "trivial_flip": "known_flip",
    }
    for row in suitability_rows:
        record_id = row.get("record_id")
        metrics = row.get("metrics")
        if not isinstance(record_id, str) or not isinstance(metrics, Mapping):
            continue
        memberships = metrics.get("duplicate_groups")
        if not isinstance(memberships, list):
            continue
        for membership_row in memberships:
            if not isinstance(membership_row, Mapping):
                continue
            source_kind = str(membership_row.get("kind") or "")
            relation_type = kind_map.get(source_kind)
            group_id = membership_row.get("group_id")
            if relation_type is not None and isinstance(group_id, str):
                groups[(f"suitability:{source_kind}", group_id)].append(record_id)
    for (source_key, _group_id), member_ids in sorted(groups.items()):
        if not source_key.startswith("suitability:"):
            continue
        source_kind = source_key.partition(":")[2]
        relation_type = kind_map[source_kind]
        members = tuple(sorted(set(member_ids)))
        if len(members) < 2:
            continue
        duplicate_groups[(relation_type, members)] = None
    for relation_type, members in sorted(duplicate_groups):
        hard = relation_type != "near_identical"
        output.append(
            _relation_row(
                relation_type,
                members,
                membership=relation_membership_key(members),
                hard=hard,
                evidence="suitability_pixel_fingerprint",
            )
        )

    source_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in provenance_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str):
            continue
        for relation_type, field in (
            ("pack", "packs"),
            ("creator_lineage", "creator_or_publishers"),
            ("sheet", "declared_sheet_ids"),
        ):
            values = row.get(field)
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value:
                    source_groups[(relation_type, value)].append(record_id)
    for (relation_type, _source_value), member_ids in sorted(source_groups.items()):
        members = tuple(sorted(set(member_ids)))
        if len(members) < 2:
            continue
        output.append(
            _relation_row(
                relation_type,
                members,
                membership=relation_membership_key(members),
                hard=True,
                evidence="provenance_only_post_blind_not_used",
                post_blind_not_used=True,
            )
        )

    unique = {(str(row["relation_type"]), tuple(row["member_record_ids"])): row for row in output}
    return sorted(unique.values(), key=lambda row: (str(row["relation_type"]), str(row["relation_id"])))


def _relation_row(
    relation_type: str,
    members: Sequence[str],
    *,
    membership: str,
    hard: bool,
    evidence: str = "deterministic_pixel_identity",
    post_blind_not_used: bool = False,
) -> dict[str, Any]:
    member_list = sorted(set(members))
    relation_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "member_record_ids": member_list,
                "relation_type": relation_type,
                "version": "raw_relation_v1",
            }
        )
    ).hexdigest()
    return {
        "declared_variant_status": "unknown_not_inferred",
        "evidence": evidence,
        "hard_relation": hard,
        "hard_split_constraint": hard,
        "member_record_ids": member_list,
        "members": member_list,
        "membership_digest": membership,
        "post_blind_not_used": post_blind_not_used,
        "relation_id": f"rel_{relation_digest}",
        "relation_type": relation_type,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
    }


def _write_staging_artifacts(
    staging: Path,
    *,
    inventory_rows: Sequence[Mapping[str, Any]],
    build: _BuildPass,
    source_root_supplied: bool,
    historical_reproduction: Mapping[str, Any],
) -> None:
    for name in _MACHINE_DIRECTORIES:
        (staging / name).mkdir(parents=True, exist_ok=False)
    _write_jsonl(staging / "extraction_manifest.jsonl", build.extraction_rows)
    _write_jsonl(staging / "blob_manifest.jsonl", build.blob_rows)
    _write_jsonl(staging / "provenance_manifest.jsonl", build.provenance_rows)
    _write_jsonl(staging / "suitability_manifest.jsonl", build.suitability_rows)
    _write_jsonl(staging / "relation_manifest.jsonl", build.relation_rows)
    for blob_id, value in sorted(build.blobs.items()):
        _write_bytes(staging / "raw_audit_blobs" / f"{blob_id}.rgba", value)

    decision_counts = Counter(str(row.get("inclusion_decision") or "quarantine") for row in inventory_rows)
    geometry_ids = {
        str(row["geometry_family_id"])
        for row in build.extraction_rows
        if isinstance(row.get("geometry_family_id"), str)
    }
    blob_counts = Counter(str(row["blob_id"]) for row in build.extraction_rows if isinstance(row.get("blob_id"), str))
    suitability_counts = Counter(str(row.get("audit_status")) for row in build.suitability_rows)
    relation_counts = Counter(str(row.get("relation_type")) for row in build.relation_rows)
    hard_variant_relation_types = {"alpha_recolor", "translation", "known_flip"}
    report = {
        "candidate_dataset_created": False,
        "candidate_only": None,
        "external_read_only_audit_evidence": {
            "authoritative_for_new_rebuild": False,
            "evidence": historical_reproduction,
            "status": "supplied_historical_reproduction_not_recomputed_by_this_compiler",
        },
        "freeze": {
            "all_automated_gates_passed": False,
            "frozen_dataset_created": False,
            "production_frozen": False,
            "promotion_forbidden": True,
            "training_authorized": False,
        },
        "forensic_inventory": {
            "accept_count": decision_counts.get("accept", 0),
            "quarantine_count": decision_counts.get("quarantine", 0),
            "record_count": len(inventory_rows),
            "reject_count": decision_counts.get("reject", 0),
        },
        "provider": {
            "provider_calls": 0,
            "sol_status": SOL_UNAVAILABLE,
        },
        "raw_audit": {
            "authoritative_dataset_rebuild": False,
            "decode_failure_count": build.decode_failure_count,
            "decoded_record_count": build.decoded_count,
            "duplicate_blob_record_count": sum(count - 1 for count in blob_counts.values() if count > 1),
            "materialized_blob_count": len(build.blobs),
            "mode": "verified_original_decode" if source_root_supplied else "inventory_observation_only",
            "source_bytes_verified_count": build.source_verified_count,
            "suitability_status_counts": {
                status: suitability_counts.get(status, 0) for status in ("accept", "quarantine", "reject")
            },
            "two_complete_audit_rebuilds_byte_identical": True,
            "unique_geometry_count": len(geometry_ids),
            "hard_deterministic_variant_relation_group_count": sum(
                count
                for relation_type, count in relation_counts.items()
                if relation_type in hard_variant_relation_types
            ),
            "hard_deterministic_variant_relation_type_counts": {
                relation_type: relation_counts.get(relation_type, 0)
                for relation_type in sorted(hard_variant_relation_types)
            },
            "near_identical_similarity_group_count": relation_counts.get("near_identical", 0),
        },
        "relations": {
            "declared_variants": "unknown_not_inferred",
            "relation_group_count": len(build.relation_rows),
            "relation_type_counts": dict(sorted(relation_counts.items())),
        },
        "raw_source_gate_passed": False,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "blocked",
    }
    _write_json(staging / "rebuild_report.json", report)
    _write_text(staging / "rebuild_report.md", _render_report_markdown(report))
    canary_report = unavailable_canary_report(_configured_sol_identity())
    canary_report.update(
        {
            "canary_completed": False,
            "labels_abstained": 0,
            "labels_accepted": 0,
            "labels_conflicted": 0,
            "labels_excluded": 0,
            "records_not_labeled": len(inventory_rows),
            "required_canary_record_count": 20,
        }
    )
    _write_json(staging / "sol_canary_report.json", canary_report)
    _write_json(
        staging / "label_health_report.json",
        {
            "audit_completed": False,
            "batch_count": 0,
            "labels_audited": 0,
            "provider_calls": 0,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": SOL_UNAVAILABLE,
        },
    )
    _write_json(
        staging / "label_drift_report.json",
        {
            "baseline_available": False,
            "drift_checked": False,
            "provider_calls": 0,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": SOL_UNAVAILABLE,
        },
    )
    _write_json(
        staging / "source_conflict_report.json",
        {
            "blind_labels_frozen": False,
            "conflict_count": None,
            "historical_synthetic_fixture_taint_cases": {
                "count": 4,
                "status": "historical_forensic_finding_not_recomputed_by_this_compiler",
            },
            "raw_inventory_issue_counts": _inventory_issue_counts(inventory_rows),
            "reconciliation_performed": False,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "blocked_until_blind_labeling_completes",
        },
    )
    freeze_gates = FreezeGateEvidence(two_rebuilds_byte_identical=True)
    _write_json(
        staging / "freeze_manifest.json",
        {
            "candidate_dataset_created": False,
            "candidate_only": None,
            "failed_gates": freeze_gates.failures(),
            "gate_evidence": freeze_gates.canonical(),
            "production_frozen": False,
            "promotion_forbidden": True,
            "raw_source_gate_passed": False,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "sol_status": SOL_UNAVAILABLE,
            "training_authorized": False,
        },
    )
    _write_json(
        staging / "frozen_hash_verification.json",
        {
            "frozen_dataset_created": False,
            "hash_verification_performed": False,
            "production_frozen": False,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "freeze_blocked",
            "training_authorized": False,
        },
    )
    for view in VIEW_NAMES:
        _write_json(
            staging / "candidate_view_manifests" / f"{view}.json",
            {
                "candidate_dataset_created": False,
                "member_record_ids": [],
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "status": "blocked",
                "view": view,
            },
        )
    _write_json(
        staging / "candidate_view_manifests" / "index.json",
        {
            "candidate_dataset_created": False,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "blocked",
            "views": list(VIEW_NAMES),
        },
    )
    _write_json(
        staging / "contact_sheets" / "index.json",
        {
            "full_labeled_contact_sheets_created": False,
            "groupings": [
                "category",
                "source_pack",
                "creator_lineage",
                "uncertainty",
                "conflict_status",
                "quarantine_reason",
            ],
            "labeled_sheet_count": 0,
            "provider_calls": 0,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": SOL_UNAVAILABLE,
        },
    )
    _write_text(
        staging / "contact_sheets" / "README.md",
        "# Contact sheets\n\nNo labeled contact sheets were created because GPT-5.6 Sol was unavailable.\n",
    )
    _write_json(
        staging / "batch_audit_reports" / "index.json",
        {
            "audit_batch_count": 0,
            "provider_calls": 0,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": SOL_UNAVAILABLE,
        },
    )


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    inventory = report["forensic_inventory"]
    audit = report["raw_audit"]
    external = report["external_read_only_audit_evidence"]
    evidence_json = json.dumps(external["evidence"], indent=2, sort_keys=True, ensure_ascii=False)
    return (
        "# Dataset-v5 raw rebuild evidence\n\n"
        "Status: **blocked**. `SOL_MODEL_UNAVAILABLE`; provider calls: **0**.\n\n"
        "No candidate or frozen dataset was created. Production freeze and training remain forbidden.\n\n"
        "## Forensic inventory decisions\n\n"
        f"- Accept: {inventory['accept_count']}\n"
        f"- Quarantine: {inventory['quarantine_count']}\n"
        f"- Reject: {inventory['reject_count']}\n\n"
        "## Non-authoritative raw audit\n\n"
        f"- Mode: `{audit['mode']}`\n"
        f"- Decoded records: {audit['decoded_record_count']}\n"
        f"- Unique pixel geometry families: {audit['unique_geometry_count']}\n"
        f"- Content suitability decisions: {json.dumps(audit['suitability_status_counts'], sort_keys=True)}\n"
        "- Hard deterministic variant relation groups: "
        f"{audit['hard_deterministic_variant_relation_group_count']}\n"
        f"- Near-identical similarity groups (not declared variants): {audit['near_identical_similarity_group_count']}\n"
        "- Two complete audit passes byte-identical: true\n"
        "- Raw source gate passed: false\n\n"
        "This is read-only audit evidence, not an authoritative new candidate rebuild.\n\n"
        "## External/read-only historical reproduction evidence\n\n"
        f"Status: `{external['status']}`. It is not authoritative for the new rebuild.\n\n"
        "```json\n"
        f"{evidence_json}\n"
        "```\n"
    )


def _verify_two_passes(first: _BuildPass, second: _BuildPass) -> None:
    left = canonical_json_bytes(first.comparison_value())
    right = canonical_json_bytes(second.comparison_value())
    if left != right:
        raise EvidenceCompileError("two complete raw audit rebuild passes were not byte-identical")


def _configured_sol_identity() -> dict[str, Any]:
    """Return secret-free configured identity plus explicit unavailable attestations."""

    backend = os.environ.get("SPRITELAB_SOL_BACKEND", "").strip()
    model = os.environ.get("SPRITELAB_SOL_MODEL", "").strip()
    base_url = os.environ.get("SPRITELAB_SOL_BASE_URL", "").strip()
    endpoint_identity: str | None = None
    if base_url:
        try:
            parsed = urllib.parse.urlsplit(base_url)
            if parsed.scheme and parsed.netloc and not parsed.username and not parsed.password:
                path = parsed.path.rstrip("/") or "/"
                endpoint_identity = urllib.parse.urlunsplit(
                    (parsed.scheme.lower(), parsed.netloc.lower(), path, "", "")
                )
        except ValueError:
            endpoint_identity = None
    return {
        "api_key_configured": bool(os.environ.get("SPRITELAB_SOL_API_KEY", "").strip()),
        "backend": backend or None,
        "blind_request_schema_version": BLIND_REQUEST_SCHEMA_VERSION,
        "configured_model_identifier": model or None,
        "endpoint_identity": endpoint_identity,
        "model_version": None,
        "provider": None,
        "provider_identity_status": "unavailable_not_attested",
        "provider_request_schema_version": None,
        "provider_schema_version": SOL_PROVIDER_SCHEMA_VERSION,
        "prompt_versions": {
            "adjudication": SOL_ADJUDICATION_PROMPT_VERSION,
            "consistency": SOL_CONSISTENCY_PROMPT_VERSION,
        },
    }


def _decode_rgba(payload: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if int(getattr(image, "n_frames", 1)) != 1:
                raise EvidenceCompileError("multi-frame source has no recorded frame extraction operation")
            image.load()
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    except UnidentifiedImageError as exc:
        raise EvidenceCompileError("unidentified_image") from exc
    except OSError as exc:
        raise EvidenceCompileError("image_io_error") from exc
    except ValueError as exc:
        raise EvidenceCompileError("invalid_image_value") from exc
    if rgba.ndim != 3 or rgba.shape[2] != 4 or min(rgba.shape[:2]) <= 0:
        raise EvidenceCompileError(f"invalid decoded image geometry: {rgba.shape}")
    return np.ascontiguousarray(rgba, dtype=np.uint8)


def _apply_recorded_transform(rgba: np.ndarray, row: Mapping[str, Any]) -> np.ndarray:
    operation = str(row.get("extraction_operation") or "")
    crop = _crop_tuple(row.get("crop_coordinates"))
    if operation not in {"", "identity_decode_to_rgba"} and "crop" not in operation:
        raise EvidenceCompileError(f"unrecorded or unsupported extraction operation: {operation!r}")
    result = rgba
    if crop is not None:
        left, top, right, bottom = crop
        height, width = rgba.shape[:2]
        if right > width or bottom > height:
            raise EvidenceCompileError(f"recorded crop leaves image bounds: {crop}, image={width}x{height}")
        result = rgba[top:bottom, left:right]
    padding = row.get("padding_operation")
    if padding not in (None, "", "none"):
        if not isinstance(padding, Mapping):
            raise EvidenceCompileError(f"unrecorded padding operation: {padding!r}")
        values = [padding.get(name) for name in ("left", "top", "right", "bottom")]
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise EvidenceCompileError(f"invalid recorded padding operation: {padding!r}")
        fill = padding.get("fill_rgba", [0, 0, 0, 0])
        if list(fill) != [0, 0, 0, 0]:
            raise EvidenceCompileError("only deterministic transparent-black padding is supported")
        left, top, right, bottom = values
        height, width = result.shape[:2]
        padded = np.zeros((height + top + bottom, width + left + right, 4), dtype=np.uint8)
        padded[top : top + height, left : left + width] = result
        result = padded
    return np.ascontiguousarray(result, dtype=np.uint8)


def _verify_dimensions(row: Mapping[str, Any], rgba: np.ndarray) -> None:
    height, width = rgba.shape[:2]
    for field, observed in (("width", width), ("height", height)):
        expected = row.get(field)
        if expected is not None and (not isinstance(expected, int) or expected != observed):
            raise EvidenceCompileError(f"decoded {field} mismatch: expected {expected!r}, observed {observed}")


def _archive_hash_bindings(document: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    archives = document.get("archives")
    if isinstance(archives, Mapping):
        for path, digest in archives.items():
            _add_archive_binding(result, str(path), _required_sha256(digest, "archive hash"))
    artifacts = document.get("artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise EvidenceCompileError("source archive hash artifact must be an object")
            digest = _optional_sha256(
                artifact.get("current_observed_archive_sha256") or artifact.get("archive_sha256"),
                "current_observed_archive_sha256",
            )
            if digest is None:
                continue
            paths = artifact.get("original_archive_paths")
            if not isinstance(paths, list):
                one = artifact.get("original_archive_path")
                paths = [one] if isinstance(one, str) else []
            for path in paths:
                if isinstance(path, str):
                    _add_archive_binding(result, path, digest)
    if not result:
        raise EvidenceCompileError("source archive hashes document contains no path/hash bindings")
    return dict(sorted(result.items()))


def _add_archive_binding(result: dict[str, str], path: str, digest: str) -> None:
    key = _canonical_display_path(path)
    previous = result.setdefault(key, digest)
    if previous != digest:
        raise EvidenceCompileError(f"archive path is bound to multiple hashes: {path!r}")


def _binding_member_path(row: Mapping[str, Any]) -> str:
    member = row.get("archive_member_path")
    if isinstance(member, str) and member:
        return _safe_member_name(member)
    source = _required_text(row, "original_archive_path")
    return f"direct/{hashlib.sha256(_canonical_display_path(source).encode('utf-8')).hexdigest()}"


def _crop_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise EvidenceCompileError(f"invalid crop coordinates: {value!r}")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise EvidenceCompileError(f"invalid crop coordinates: {value!r}")
    crop = tuple(value)
    left, top, right, bottom = crop
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise EvidenceCompileError(f"invalid crop coordinates: {value!r}")
    return crop


def _json_crop(value: Any) -> list[int] | None:
    crop = _crop_tuple(value)
    return list(crop) if crop is not None else None


def _padding_binding(value: Any) -> Mapping[str, Any] | None:
    if value in (None, "", "none"):
        return None
    if not isinstance(value, Mapping):
        raise EvidenceCompileError(f"invalid padding operation: {value!r}")
    return _canonical_mapping(value)


def _nonsemantic_padding(value: Any) -> Mapping[str, Any] | None:
    return _padding_binding(value)


def _safe_member_name(value: str) -> str:
    if not value or "\x00" in value or "\\" in value or value.startswith("/"):
        raise EvidenceCompileError(f"unsafe archive member path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EvidenceCompileError(f"unsafe archive member path: {value!r}")
    return path.as_posix()


def _canonical_display_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not normalized or normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        raise EvidenceCompileError(f"source path must be relative: {value!r}")
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise EvidenceCompileError(f"unsafe source path: {value!r}")
    return PurePosixPath(*parts).as_posix()


def _require_targets_absent(root: Path) -> None:
    existing = [name for name in (*_MACHINE_FILES, *_MACHINE_DIRECTORIES) if (root / name).exists()]
    if existing:
        raise FileExistsError(f"blocked evidence output already exists: {', '.join(sorted(existing))}")


def _copy_staging_exclusively(staging: Path, destination: Path) -> None:
    created: list[Path] = []
    try:
        for name in (*_MACHINE_FILES, *_MACHINE_DIRECTORIES):
            source = staging / name
            target = destination / name
            if source.is_dir():
                target.mkdir(exist_ok=False)
                created.append(target)
                for child in sorted((path for path in source.rglob("*") if path.is_file()), key=lambda p: p.as_posix()):
                    relative = child.relative_to(source)
                    output = target / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("xb") as handle:
                        handle.write(child.read_bytes())
            else:
                with target.open("xb") as handle:
                    handle.write(source.read_bytes())
                created.append(target)
    except Exception:
        for path in reversed(created):
            if path.is_dir():
                remove_confined_tree(path, destination, missing_ok=True)
            else:
                path.unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise EvidenceCompileError(f"blank JSONL row at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceCompileError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EvidenceCompileError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceCompileError(f"invalid JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceCompileError(f"JSON document must be an object: {path}")
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(canonical_json_bytes(dict(row)).decode("utf-8") + "\n" for row in rows))


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def _record_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    record = row.get("record_id")
    return (record is None, str(record or ""), str(row.get("blob_id") or ""))


def _provenance_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        row.get("record_id") is None,
        str(row.get("record_id") or ""),
        str(row.get("original_archive_path") or ""),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise EvidenceCompileError(f"missing required text field {field}")
    return value


def _required_sha256(value: Any, field: str) -> str:
    result = _optional_sha256(value, field)
    if result is None:
        raise EvidenceCompileError(f"missing {field}")
    return result


def _optional_sha256(value: Any, field: str) -> str | None:
    if value in (None, ""):
        return None
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in _SHA256_CHARS for character in normalized):
        raise EvidenceCompileError(f"{field} is not a SHA-256 digest: {value!r}")
    return normalized


def _sorted_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if isinstance(item, str)})


def _inventory_issue_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    affected_records = 0
    license_affected_records = 0
    explicit_conflict_records = 0
    for row in rows:
        issues = set(_sorted_strings(row.get("provenance_issues"))) | set(_sorted_strings(row.get("audit_issues")))
        if issues:
            affected_records += 1
        issue_counts.update(issues)
        if any("license" in issue.casefold() for issue in issues):
            license_affected_records += 1
        if any(token in issue.casefold() for issue in issues for token in ("conflict", "ambiguous")):
            explicit_conflict_records += 1
    return {
        "affected_record_count": affected_records,
        "explicit_conflict_or_ambiguity_record_count": explicit_conflict_records,
        "issue_code_counts": dict(sorted(issue_counts.items())),
        "license_issue_record_count": license_affected_records,
    }


def _declared_sheet_ids(row: Mapping[str, Any]) -> list[str]:
    """Return only explicit sheet declarations; never infer from names/pixels."""

    output: set[str] = set()
    for field in ("sheet_id", "sheet_membership", "source_sheet"):
        value = row.get(field)
        if isinstance(value, str) and value:
            output.add(value)
        elif isinstance(value, list):
            output.update(item for item in value if isinstance(item, str) and item)
    return sorted(output)


def _canonical_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise EvidenceCompileError(f"historical reproduction evidence is not JSON-serializable: {exc}") from exc
