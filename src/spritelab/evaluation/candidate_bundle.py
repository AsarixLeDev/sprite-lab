"""Canonical production writer and strict loader for memorization evidence v2."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from spritelab.evaluation.memorization import (
    COMPARISON_METHOD,
    COMPARISON_PARAMETERS_SHA256,
    DETECTOR_POLICY,
    DETECTOR_POLICY_SHA256,
    DETECTOR_POLICY_VERSION,
    EvidenceClass,
    MemorizationMachineStatus,
    detector_policy_record,
    evaluate_memorization_outcome,
    parse_evidence_class,
    recompute_memorization_status,
    reconstruct_rgba,
    resolve_training_context_identities,
    training_record_context_identities,
)
from spritelab.evaluation.strict_json import strict_json_loads

CANDIDATE_SCHEMA_VERSION = "sprite_lab_memorization_candidate_evidence_v2"
CANDIDATE_CONTRACT_VERSION = "sprite_lab_memorization_candidate_bundle_contract_v2.2"
REVIEW_LOG_CONTRACT_VERSION = "sprite_lab_review_log_contract_v2"

_BOUND_FILES = (
    ("checkpoint_path", "checkpoint_sha256", "checkpoint"),
    ("benchmark_manifest_path", "benchmark_manifest_sha256", "benchmark manifest"),
    ("machine_report_path", "machine_report_sha256", "machine scoring report"),
    ("generated_report_path", "generated_report_sha256", "generated report"),
    ("generated_manifest_path", "generated_manifest_sha256", "generated manifest"),
    ("training_manifest_path", "training_manifest_sha256", "training manifest"),
    ("detector_policy_artifact_path", "detector_policy_artifact_sha256", "detector policy"),
)
_PAIR_BINDING_FIELDS = (
    "training_dataset_identity",
    "training_view_identity",
    "checkpoint_path",
    "checkpoint_sha256",
    "benchmark_manifest_path",
    "benchmark_manifest_sha256",
    "generated_manifest_path",
    "generated_manifest_sha256",
    "training_manifest_path",
    "training_manifest_sha256",
    "detector_policy_version",
    "detector_policy_sha256",
    "comparison_method",
    "comparison_parameters_sha256",
)
_PAIR_REQUIRED_FIELDS = frozenset(
    {
        "pair_id",
        "generated_sample_id",
        "prompt_id",
        "seed",
        "generated_png_path",
        "generated_png_sha256",
        "generated_decoded_rgba_sha256",
        "training_dataset_identity",
        "training_view_identity",
        "training_source_sprite_id",
        "training_row_or_index",
        "training_image_path",
        "training_source_blob_path",
        "training_source_blob_sha256",
        "training_decoded_rgba_sha256",
        "training_manifest_sha256",
        "evidence_class",
        "evidence_metrics",
        "evidence_diagnostics",
        "candidate_bundle_identity_inputs",
        *_PAIR_BINDING_FIELDS,
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_array_sha256(rgba: np.ndarray) -> str:
    array = np.clip(np.asarray(rgba), 0, 255).astype(np.uint8)
    height, width = array.shape[:2]
    dimensions = width.to_bytes(8, "big") + height.to_bytes(8, "big")
    return hashlib.sha256(dimensions + array.tobytes()).hexdigest()


def decoded_rgba_sha256(path: Path) -> str:
    with Image.open(path) as image:
        image.load()
        return _decoded_array_sha256(np.asarray(image.convert("RGBA")))


def candidate_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    """Hash the complete ordered bundle, excluding only its self-hash."""
    payload = dict(bundle)
    payload.pop("candidate_evidence_sha256", None)
    return canonical_sha256(payload)


def pair_evidence_sha256(pair: Mapping[str, Any]) -> str:
    return canonical_sha256(pair)


def _path_identity(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").casefold()


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and _path_identity(Path(value)) == _path_identity(expected)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_object(path: Path, label: str, reasons: list[str]) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        reasons.append(f"{label} is missing or malformed: {error}")
        return {}
    if not isinstance(value, dict):
        reasons.append(f"{label} must be a JSON object")
        return {}
    return value


def _read_manifest(path: Path, label: str, reasons: list[str]) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        try:
            value = strict_json_loads(text)
        except json.JSONDecodeError:
            return [strict_json_loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for key in ("records", "samples", "images", "rows"):
                if isinstance(value.get(key), list):
                    return [row for row in value[key] if isinstance(row, dict)]
            return [value]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        reasons.append(f"{label} is missing or malformed: {error}")
    return []


def _record_value(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _training_record_context_identity(record: Mapping[str, Any], field: str) -> Any:
    dataset_identity, view_identity = training_record_context_identities(record)
    return dataset_identity if field == "dataset_identity" else view_identity


def _generated_record_path(record: Mapping[str, Any], manifest: Path) -> Path | None:
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    raw = _record_value(record, "image_path", "generated_png_path", "png_path", "image")
    if raw in (None, ""):
        raw = next((paths.get(key) for key in ("indexed_png", "hard_rgba", "raw_rgba") if paths.get(key)), None)
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else manifest.parent / candidate


def _training_record_source(record: Mapping[str, Any], manifest: Path) -> Path | None:
    direct = _record_value(record, "image_path", "training_image_path", "source_image_path", "blob_path")
    if isinstance(direct, str) and direct:
        candidate = Path(direct)
        return candidate if candidate.is_absolute() else manifest.parent / candidate
    npz_file = record.get("npz_file")
    if not isinstance(npz_file, str) or not npz_file:
        return None
    source = record.get("source") if isinstance(record.get("source"), Mapping) else {}
    root = Path(str(source.get("dataset_dir"))) if source.get("dataset_dir") else manifest.parent
    return root / npz_file


@dataclass(frozen=True)
class CandidateBundleValidation:
    """Controlled strict-loader result shared by product and promotion paths."""

    path: Path
    bundle: dict[str, Any]
    pairs: tuple[dict[str, Any], ...]
    machine_report: dict[str, Any]
    reasons: tuple[str, ...]
    state: str

    @property
    def valid(self) -> bool:
        return self.state == "complete" and not self.reasons

    @property
    def pair_order(self) -> tuple[str, ...]:
        return tuple(str(pair.get("pair_id")) for pair in self.pairs)


def write_candidate_bundle(
    output_path: Path,
    *,
    pairs: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    benchmark_manifest: Path,
    machine_report: Path,
    generated_report: Path,
    generated_manifest: Path,
    training_manifest: Path,
    detector_policy_artifact: Path,
    training_dataset_identity: str,
    training_view_identity: str,
    generated_images_root: Path,
    training_images_root: Path,
) -> dict[str, Any]:
    """Write one canonical bound-v2 bundle and verify it through the strict loader."""
    bindings: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "contract_version": CANDIDATE_CONTRACT_VERSION,
        "generated_images_root": str(generated_images_root.resolve()),
        "training_images_root": str(training_images_root.resolve()),
        "training_dataset_identity": training_dataset_identity,
        "training_view_identity": training_view_identity,
        "detector_policy_version": DETECTOR_POLICY_VERSION,
        "detector_policy_sha256": DETECTOR_POLICY_SHA256,
        "comparison_method": COMPARISON_METHOD,
        "comparison_parameters_sha256": COMPARISON_PARAMETERS_SHA256,
        "review_log_contract": {
            "schema_version": REVIEW_LOG_CONTRACT_VERSION,
            "absence_allowed_before_first_signed_review": True,
            "absence_allowed_when_no_review_candidates": True,
        },
    }
    for path, prefix in (
        (checkpoint, "checkpoint"),
        (benchmark_manifest, "benchmark_manifest"),
        (machine_report, "machine_report"),
        (generated_report, "generated_report"),
        (generated_manifest, "generated_manifest"),
        (training_manifest, "training_manifest"),
        (detector_policy_artifact, "detector_policy_artifact"),
    ):
        bindings[f"{prefix}_path"] = str(path.resolve())
        bindings[f"{prefix}_sha256"] = file_sha256(path)
    identity_inputs = {
        field: bindings[field]
        for field in (
            "training_dataset_identity",
            "training_view_identity",
            "checkpoint_sha256",
            "benchmark_manifest_sha256",
            "machine_report_sha256",
            "generated_manifest_sha256",
            "training_manifest_sha256",
            "detector_policy_sha256",
            "comparison_parameters_sha256",
        )
    }
    bound_pairs: list[dict[str, Any]] = []
    for raw_pair in sorted(pairs, key=lambda item: str(item.get("pair_id") or "")):
        pair = dict(raw_pair)
        for field in _PAIR_BINDING_FIELDS:
            pair[field] = bindings[field]
        pair["candidate_bundle_identity_inputs"] = dict(identity_inputs)
        bound_pairs.append(pair)
    bundle = {
        **bindings,
        "candidate_count": len(bound_pairs),
        "candidate_order": [pair.get("pair_id") for pair in bound_pairs],
        "pairs": bound_pairs,
    }
    bundle["candidate_evidence_sha256"] = candidate_bundle_sha256(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".candidate",
        dir=output_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        validation = load_candidate_bundle(
            temporary,
            expected_context={
                "training_dataset_identity": training_dataset_identity,
                "training_view_identity": training_view_identity,
            },
        )
        if not validation.valid:
            raise ValueError("refusing invalid candidate bundle: " + "; ".join(validation.reasons))
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return bundle


def load_candidate_bundle(
    path: Path,
    *,
    expected_context: Mapping[str, Any] | None = None,
) -> CandidateBundleValidation:
    """Strictly validate schema, identities, sources, pairs, and machine candidate set."""
    reasons: list[str] = []
    bundle = _read_object(path, "candidate evidence bundle", reasons)
    if bundle.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        reasons.append("candidate evidence has an unsupported schema")
    if bundle.get("contract_version") != CANDIDATE_CONTRACT_VERSION:
        reasons.append("candidate evidence has an unsupported contract version")
    if bundle.get("candidate_evidence_sha256") != candidate_bundle_sha256(bundle):
        reasons.append("candidate evidence bundle SHA-256 mismatch")

    actual_paths: dict[str, Path] = {}
    for path_field, hash_field, label in _BOUND_FILES:
        raw = bundle.get(path_field)
        if not isinstance(raw, str) or not raw:
            reasons.append(f"{label} path identity is missing")
            continue
        actual = Path(raw)
        actual_paths[path_field] = actual
        try:
            actual_hash = file_sha256(actual)
        except OSError as error:
            reasons.append(f"{label} cannot be read: {error}")
            continue
        if bundle.get(hash_field) != actual_hash:
            reasons.append(f"{label} SHA-256 mismatch")
    for field, value in dict(expected_context or {}).items():
        if field.endswith("_path") and value not in (None, ""):
            if not _same_path(bundle.get(field), Path(str(value))):
                reasons.append(f"expected {field} identity mismatch")
        elif value is not None and bundle.get(field) != value:
            reasons.append(f"expected {field} identity mismatch")

    policy_path = actual_paths.get("detector_policy_artifact_path")
    if policy_path is not None:
        policy = _read_object(policy_path, "detector policy artifact", reasons)
        if policy != detector_policy_record():
            reasons.append("detector policy artifact content is unsupported or changed")
        core = {key: policy.get(key) for key in DETECTOR_POLICY}
        if canonical_sha256(core) != policy.get("detector_policy_sha256"):
            reasons.append("detector policy artifact hash does not match its content")
    for field, canonical in (
        ("detector_policy_version", DETECTOR_POLICY_VERSION),
        ("detector_policy_sha256", DETECTOR_POLICY_SHA256),
        ("comparison_method", COMPARISON_METHOD),
        ("comparison_parameters_sha256", COMPARISON_PARAMETERS_SHA256),
    ):
        if bundle.get(field) != canonical:
            reasons.append(f"candidate bundle {field} mismatch")
    for field in ("training_dataset_identity", "training_view_identity"):
        value = bundle.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            reasons.append(f"candidate bundle {field} is missing or invalid")

    raw_pairs = bundle.get("pairs")
    pairs = [dict(pair) for pair in raw_pairs if isinstance(pair, dict)] if isinstance(raw_pairs, list) else []
    if not isinstance(raw_pairs, list) or len(pairs) != len(raw_pairs):
        reasons.append("candidate evidence pairs must be a list of objects")
    pair_order = cast(list[str], [pair.get("pair_id") for pair in pairs])
    if bundle.get("candidate_order") != pair_order:
        reasons.append("candidate ordering contract mismatch")
    if bundle.get("candidate_count") != len(pairs):
        reasons.append("candidate count mismatch")
    if len(pair_order) != len(set(pair_order)):
        reasons.append("candidate pair IDs must be duplicate-free")
    if all(isinstance(pair_id, str) for pair_id in pair_order) and pair_order != sorted(pair_order):
        reasons.append("candidate pair IDs are not in canonical ordering")

    generated_manifest = actual_paths.get("generated_manifest_path")
    training_manifest = actual_paths.get("training_manifest_path")
    generated_rows = _read_manifest(generated_manifest, "generated manifest", reasons) if generated_manifest else []
    training_rows = _read_manifest(training_manifest, "training manifest", reasons) if training_manifest else []
    if training_manifest is not None:
        try:
            resolved_dataset, resolved_view = resolve_training_context_identities(
                dataset_identities=(
                    _training_record_context_identity(row, "dataset_identity") for row in training_rows
                ),
                view_identities=(_training_record_context_identity(row, "view_identity") for row in training_rows),
                manifest_sha256=file_sha256(training_manifest),
                explicit_dataset_identity=(expected_context or {}).get("training_dataset_identity"),
                explicit_view_identity=(expected_context or {}).get("training_view_identity"),
            )
            if bundle.get("training_dataset_identity") != resolved_dataset:
                reasons.append("candidate bundle training dataset identity disagrees with authoritative context")
            if bundle.get("training_view_identity") != resolved_view:
                reasons.append("candidate bundle training view identity disagrees with authoritative context")
        except (OSError, ValueError) as error:
            reasons.append(f"training manifest identity cannot be resolved: {error}")
    expected_inputs = {
        field: bundle.get(field)
        for field in (
            "training_dataset_identity",
            "training_view_identity",
            "checkpoint_sha256",
            "benchmark_manifest_sha256",
            "machine_report_sha256",
            "generated_manifest_sha256",
            "training_manifest_sha256",
            "detector_policy_sha256",
            "comparison_parameters_sha256",
        )
    }
    roots: dict[str, Path] = {}
    for field, label in (
        ("generated_images_root", "generated images root"),
        ("training_images_root", "training images root"),
    ):
        raw_root = bundle.get(field)
        if not isinstance(raw_root, str) or not Path(raw_root).is_dir():
            reasons.append(f"{label} is missing or invalid")
        else:
            roots[field] = Path(raw_root).resolve()
    for index, pair in enumerate(pairs):
        pair_id = pair.get("pair_id")
        missing = sorted(_PAIR_REQUIRED_FIELDS - pair.keys())
        if missing:
            reasons.append(f"candidate pair {index} missing fields: {', '.join(missing)}")
        if not isinstance(pair_id, str) or not pair_id:
            reasons.append(f"candidate pair {index} has invalid pair_id")
        try:
            evidence_class = parse_evidence_class(pair.get("evidence_class"))
        except ValueError as error:
            reasons.append(f"candidate pair {index} {error}")
            evidence_class = None
        if pair.get("exact_rgba") is True and evidence_class not in {
            EvidenceClass.EXACT_RGBA_NONTRIVIAL,
            EvidenceClass.EXACT_RGBA_LOW_EVIDENCE_COLLISION,
        }:
            reasons.append(f"candidate pair {index} silently downgrades an exact-RGBA match")
        if not isinstance(pair.get("evidence_metrics"), dict) or not isinstance(pair.get("evidence_diagnostics"), dict):
            reasons.append(f"candidate pair {index} has malformed evidence diagnostics")
        for field in _PAIR_BINDING_FIELDS:
            if pair.get(field) != bundle.get(field):
                reasons.append(f"candidate pair {pair_id} {field} binding mismatch")
        if pair.get("candidate_bundle_identity_inputs") != expected_inputs:
            reasons.append(f"candidate pair {pair_id} candidate-bundle identity mismatch")
        if pair.get("training_dataset_identity") != bundle.get("training_dataset_identity"):
            reasons.append(f"candidate pair {pair_id} training dataset identity mismatch")
        if pair.get("training_manifest_sha256") != bundle.get("training_manifest_sha256"):
            reasons.append(f"candidate pair {pair_id} training manifest identity mismatch")
        for field in (
            "generated_png_sha256",
            "generated_decoded_rgba_sha256",
            "training_source_blob_sha256",
            "training_decoded_rgba_sha256",
        ):
            if not _valid_sha256(pair.get(field)):
                reasons.append(f"candidate pair {pair_id} has invalid {field}")
        for root_field, pair_field, label in (
            ("generated_images_root", "generated_png_path", "generated image"),
            ("training_images_root", "training_image_path", "training image"),
        ):
            root = roots.get(root_field)
            if root is not None:
                try:
                    Path(str(pair.get(pair_field))).resolve().relative_to(root)
                except ValueError:
                    reasons.append(f"{label} is outside the bound image root for pair {pair_id}")
        _validate_generated_source(pair, generated_manifest, generated_rows, reasons)
        _validate_training_source(pair, training_manifest, training_rows, reasons)

    machine_path = actual_paths.get("machine_report_path")
    machine = _read_object(machine_path, "machine scoring report", reasons) if machine_path else {}
    summary = machine.get("summary") if isinstance(machine.get("summary"), dict) else {}
    memo = summary.get("memorization") if isinstance(summary.get("memorization"), dict) else {}
    if summary.get("sample_count") != len(pairs):
        reasons.append("machine report sample_count disagrees with candidate evidence")
    machine_ids = memo.get("candidate_pair_ids")
    if not isinstance(machine_ids, list):
        reasons.append("machine report candidate_pair_ids is missing")
    else:
        if len(machine_ids) != len(set(machine_ids)):
            reasons.append("machine report candidate_pair_ids contains duplicates")
        if machine_ids != pair_order:
            reasons.append("machine report candidate set or ordering mismatch")
    classes = [pair.get("evidence_class") for pair in pairs]
    machine_outcome = evaluate_memorization_outcome(memo, expected_total=len(pairs))
    if machine_outcome.status in {MemorizationMachineStatus.INCOMPLETE, MemorizationMachineStatus.NOT_COMPARABLE}:
        reasons.extend(f"machine report: {reason}" for reason in machine_outcome.reasons)
    promotion = machine.get("promotion") if isinstance(machine.get("promotion"), dict) else {}
    reported_status = promotion.get("memorization_machine_status", memo.get("machine_status"))
    recomputed = recompute_memorization_status(classes).value
    if reported_status != recomputed or machine_outcome.status.value != recomputed:
        reasons.append("reported and recomputed memorization status mismatch")
    if memo.get("candidate_count") != len(pairs):
        reasons.append("machine report candidate_count disagrees with candidate evidence")
    if memo.get("evidence_class_counts") != dict(Counter(classes)):
        reasons.append("machine report evidence_class_counts disagrees with candidate evidence")
    if memo.get("detector_policy_sha256") != bundle.get("detector_policy_sha256"):
        reasons.append("machine report detector policy mismatch")
    if memo.get("comparison_parameters_sha256") != bundle.get("comparison_parameters_sha256"):
        reasons.append("machine report comparison parameters mismatch")

    unique_reasons = tuple(sorted(set(reasons)))
    state = "complete" if not unique_reasons else ("incomplete" if not bundle or not pairs else "not_comparable")
    return CandidateBundleValidation(path.resolve(), bundle, tuple(pairs), machine, unique_reasons, state)


def _validate_generated_source(
    pair: Mapping[str, Any], manifest: Path | None, records: Sequence[Mapping[str, Any]], reasons: list[str]
) -> None:
    pair_id = str(pair.get("pair_id"))
    if manifest is None:
        return
    matches = [
        row
        for row in records
        if _record_value(row, "sample_id", "generated_sample_id", "id") == pair.get("generated_sample_id")
    ]
    if len(matches) != 1:
        reasons.append(f"generated manifest identity mismatch for pair {pair_id}")
        return
    record = matches[0]
    record_path = _generated_record_path(record, manifest)
    pair_path = Path(str(pair.get("generated_png_path")))
    if record_path is None or _path_identity(record_path) != _path_identity(pair_path):
        reasons.append(f"generated image path mismatch for pair {pair_id}")
    try:
        blob_hash = file_sha256(pair_path)
        decoded_hash = decoded_rgba_sha256(pair_path)
    except (OSError, ValueError) as error:
        reasons.append(f"generated image cannot be validated for pair {pair_id}: {error}")
        return
    if pair.get("generated_png_sha256") != blob_hash or pair.get("generated_decoded_rgba_sha256") != decoded_hash:
        reasons.append(f"generated image hash mismatch for pair {pair_id}")
    recorded_blob = _record_value(record, "png_sha256", "generated_png_sha256", "blob_sha256")
    recorded_decoded = _record_value(record, "decoded_rgba_sha256", "generated_decoded_rgba_sha256")
    if recorded_blob is not None and recorded_blob != blob_hash:
        reasons.append(f"generated manifest hash mismatch for pair {pair_id}")
    if recorded_decoded is not None and recorded_decoded != decoded_hash:
        reasons.append(f"generated manifest decoded hash mismatch for pair {pair_id}")


def _validate_training_source(
    pair: Mapping[str, Any], manifest: Path | None, records: Sequence[Mapping[str, Any]], reasons: list[str]
) -> None:
    pair_id = str(pair.get("pair_id"))
    if manifest is None:
        return
    matches = [
        row
        for row in records
        if _record_value(row, "source_sprite_id", "training_source_sprite_id", "sprite_id")
        == pair.get("training_source_sprite_id")
        and _record_value(row, "row_or_index", "training_row_or_index", "row", "npz_row")
        == pair.get("training_row_or_index")
    ]
    if len(matches) != 1:
        reasons.append(f"training manifest source identity mismatch for pair {pair_id}")
        return
    record = matches[0]
    source_path = _training_record_source(record, manifest)
    pair_source = Path(str(pair.get("training_source_blob_path")))
    if source_path is None or _path_identity(source_path) != _path_identity(pair_source):
        reasons.append(f"training source blob path mismatch for pair {pair_id}")
    try:
        source_hash = file_sha256(pair_source)
        image_hash = decoded_rgba_sha256(Path(str(pair.get("training_image_path"))))
    except (OSError, ValueError) as error:
        reasons.append(f"training image cannot be validated for pair {pair_id}: {error}")
        return
    if pair.get("training_source_blob_sha256") != source_hash:
        reasons.append(f"training source blob hash mismatch for pair {pair_id}")
    if pair.get("training_decoded_rgba_sha256") != image_hash:
        reasons.append(f"training decoded-RGBA hash mismatch for pair {pair_id}")
    recorded_blob = _record_value(record, "source_blob_sha256", "training_source_blob_sha256", "blob_sha256")
    recorded_decoded = _record_value(record, "decoded_rgba_sha256", "training_decoded_rgba_sha256")
    if recorded_blob is not None and recorded_blob != source_hash:
        reasons.append(f"training manifest blob hash mismatch for pair {pair_id}")
    if recorded_decoded is not None and recorded_decoded != image_hash:
        reasons.append(f"training manifest decoded hash mismatch for pair {pair_id}")
    if isinstance(record.get("npz_file"), str):
        try:
            archive = np.load(pair_source, mmap_mode="r")
            try:
                reconstructed = reconstruct_rgba(archive, int(pair.get("training_row_or_index")))
            finally:
                archive.close()
            if _decoded_array_sha256(reconstructed) != image_hash:
                reasons.append(f"training reconstructed RGBA mismatch for pair {pair_id}")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            reasons.append(f"training source cannot be reconstructed for pair {pair_id}: {error}")
