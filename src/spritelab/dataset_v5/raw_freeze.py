"""Fresh candidate writing and immutable freeze verification for raw Dataset-v5."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any

from spritelab.dataset_v5.raw_relations import assert_no_hard_relation_crossing
from spritelab.dataset_v5.raw_views import SUPERVISION_CLASSES, VIEW_NAMES

CANDIDATE_DATASET_SCHEMA_VERSION = "sprite_lab_raw_candidate_dataset_v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "sprite_lab_recursive_artifact_manifest_v1"
FREEZE_MANIFEST_SCHEMA_VERSION = "sprite_lab_raw_immutable_freeze_v1"
FREEZE_GATE_SCHEMA_VERSION = "sprite_lab_raw_freeze_gates_v1"

ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
FREEZE_MANIFEST_NAME = "freeze_manifest.json"

CANDIDATE_FLAGS = {
    "candidate_only": True,
    "production_frozen": False,
    "promotion_forbidden": True,
    "training_authorized": False,
}
FROZEN_FLAGS = {
    "candidate_only": False,
    "production_frozen": True,
    "promotion_forbidden": False,
    "training_authorized": False,
}


class RawFreezeError(RuntimeError):
    """Base exception for candidate and immutable-freeze failures."""


class FreezeGateError(RawFreezeError):
    """Raised before output creation when any required freeze gate fails."""

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = tuple(failures)
        super().__init__("freeze blocked: " + "; ".join(self.failures))


class ArtifactVerificationError(RawFreezeError):
    """Raised for missing, extra, malformed, or tampered artifacts."""


@dataclass(frozen=True)
class FreezeGateEvidence:
    """Positive, fail-closed evidence required for a production freeze."""

    raw_source_hashes_verified: bool = False
    two_rebuilds_byte_identical: bool = False
    filename_leakage_free: bool = False
    silent_semantic_renaming_free: bool = False
    schema_valid: bool = False
    critical_conflicts_resolved: bool = False
    source_bindings_valid: bool = False
    licenses_valid: bool = False
    provenance_complete: bool = False
    relation_closure_valid: bool = False
    splits_leakage_free: bool = False
    masks_valid: bool = False
    supervision_classes_valid: bool = False
    audit_batches_passed: bool = False
    full_pre_freeze_audit_passed: bool = False
    artifact_hashes_recorded: bool = False
    sol_audit_completed: bool = False

    @classmethod
    def passing(cls) -> FreezeGateEvidence:
        """Construct all-positive evidence for a caller that has verified it."""

        return cls(**dict.fromkeys((field.name for field in fields(cls)), True))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FreezeGateEvidence:
        """Parse canonical flags plus common zero-count audit evidence."""

        aliases: dict[str, tuple[str, bool]] = {
            "two_rebuilds_byte_identical": ("rebuilds_byte_identical", False),
            "filename_leakage_free": ("filename_leakage_count", True),
            "silent_semantic_renaming_free": ("silent_semantic_renaming_count", True),
            "schema_valid": ("schema_invalidity_count", True),
            "critical_conflicts_resolved": ("unresolved_critical_conflict_count", True),
            "source_bindings_valid": ("invalid_source_binding_count", True),
            "licenses_valid": ("invalid_license_count", True),
            "provenance_complete": ("missing_provenance_count", True),
            "relation_closure_valid": ("invalid_relation_closure_count", True),
            "splits_leakage_free": ("hard_relation_leakage_count", True),
            "masks_valid": ("invalid_mask_count", True),
            "supervision_classes_valid": ("invalid_supervision_class_count", True),
            "audit_batches_passed": ("failed_audit_batch_count", True),
            "full_pre_freeze_audit_passed": ("full_audit_passed", False),
        }
        parsed: dict[str, bool] = {}
        for field in fields(cls):
            name = field.name
            direct = value.get(name)
            if direct is not None:
                if not isinstance(direct, bool):
                    raise FreezeGateError([f"gate {name} must be boolean"])
                parsed[name] = direct
                continue
            alias = aliases.get(name)
            if alias is None or alias[0] not in value:
                parsed[name] = False
                continue
            alias_name, zero_means_pass = alias
            alias_value = value[alias_name]
            if zero_means_pass:
                if not isinstance(alias_value, int) or isinstance(alias_value, bool) or alias_value < 0:
                    raise FreezeGateError([f"gate evidence {alias_name} must be a non-negative integer"])
                parsed[name] = alias_value == 0
            else:
                if not isinstance(alias_value, bool):
                    raise FreezeGateError([f"gate evidence {alias_name} must be boolean"])
                parsed[name] = alias_value
        return cls(**parsed)

    def failures(self) -> list[str]:
        return [field.name for field in fields(self) if getattr(self, field.name) is not True]

    def canonical(self) -> dict[str, Any]:
        return {
            "gates": asdict(self),
            "passed": not self.failures(),
            "schema_version": FREEZE_GATE_SCHEMA_VERSION,
        }


def write_candidate_dataset(
    output_root: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    relation_manifest: Mapping[str, Any],
    view_bundle: Mapping[str, Any],
    gate_evidence: FreezeGateEvidence | Mapping[str, Any],
    blob_source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a candidate only after every automated gate is positive."""

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"candidate dataset output already exists: {destination}")
    evidence = (
        gate_evidence
        if isinstance(gate_evidence, FreezeGateEvidence)
        else FreezeGateEvidence.from_mapping(gate_evidence)
    )
    if failures := evidence.failures():
        raise FreezeGateError(failures)
    views = view_bundle.get("views")
    if not isinstance(views, Mapping) or set(views) != set(VIEW_NAMES):
        raise RawFreezeError("view bundle must contain exactly the seven Dataset-v5 views")
    split_by_record = view_bundle.get("split_by_record")
    if not isinstance(split_by_record, Mapping):
        raise RawFreezeError("view bundle is missing split_by_record")
    assert_no_hard_relation_crossing(relation_manifest, split_by_record, require_all=True)

    source_index = _record_index(records)
    membership: set[str] = set()
    for view_name in VIEW_NAMES:
        view = views[view_name]
        if not isinstance(view, Mapping) or not isinstance(view.get("records"), list):
            raise RawFreezeError(f"invalid view manifest: {view_name}")
        for entry in view["records"]:
            if not isinstance(entry, Mapping):
                raise RawFreezeError(f"invalid view record in {view_name}")
            record_id = str(entry.get("record_id") or "")
            if record_id not in source_index:
                raise RawFreezeError(f"view {view_name} references unknown record {record_id!r}")
            decision = _record_decision(source_index[record_id])
            if decision != "accept":
                raise FreezeGateError([f"{view_name} includes {decision} record {record_id}"])
            membership.add(record_id)
    if not membership:
        raise RawFreezeError("cannot create an empty candidate dataset")
    included = [source_index[record_id] for record_id in sorted(membership)]
    record_failures = _record_freeze_failures(included)
    if record_failures:
        raise FreezeGateError(record_failures)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        blob_count = _copy_bound_blobs(included, blob_source_root, staging)
        _write_jsonl(staging / "record_manifest.jsonl", included)
        _write_json(staging / "relation_manifest.json", dict(relation_manifest))
        view_root = staging / "candidate_view_manifests"
        view_root.mkdir()
        for view_name in VIEW_NAMES:
            _write_json(view_root / f"{view_name}.json", views[view_name])
        view_index = {key: value for key, value in view_bundle.items() if key != "views"}
        view_index["view_files"] = {name: f"candidate_view_manifests/{name}.json" for name in VIEW_NAMES}
        _write_json(staging / "view_index.json", view_index)
        _write_json(
            staging / DATASET_MANIFEST_NAME,
            {
                **CANDIDATE_FLAGS,
                "blob_count": blob_count,
                "candidate_gates": evidence.canonical(),
                "record_count": len(included),
                "relation_universe_record_count": int(relation_manifest.get("record_count") or 0),
                "schema_version": CANDIDATE_DATASET_SCHEMA_VERSION,
                "view_names": list(VIEW_NAMES),
            },
        )
        _write_artifact_manifest(staging, status=CANDIDATE_FLAGS)
        verification = verify_candidate_dataset(staging)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verification


def verify_candidate_dataset(candidate_root: str | Path) -> dict[str, Any]:
    """Verify exact candidate membership, recursive hashes, and status flags."""

    root = Path(candidate_root)
    artifact = verify_artifact_manifest(root)
    dataset = _read_json(root / DATASET_MANIFEST_NAME)
    _require_status(dataset, CANDIDATE_FLAGS, artifact_name=DATASET_MANIFEST_NAME)
    gate_value = dataset.get("candidate_gates")
    if not isinstance(gate_value, Mapping) or not isinstance(gate_value.get("gates"), Mapping):
        raise ArtifactVerificationError("candidate dataset has no canonical gate evidence")
    evidence = FreezeGateEvidence.from_mapping(gate_value["gates"])
    if evidence.failures() or gate_value.get("passed") is not True:
        raise ArtifactVerificationError("candidate dataset records failing gate evidence")
    records = _read_jsonl(root / "record_manifest.jsonl")
    if dataset.get("record_count") != len(records):
        raise ArtifactVerificationError("candidate dataset record_count mismatch")
    record_ids = {str(row.get("record_id") or "") for row in records}
    if len(record_ids) != len(records):
        raise ArtifactVerificationError("candidate record manifest has missing or duplicate record IDs")
    blob_count = _verify_blob_bindings(root, records)
    if dataset.get("blob_count") != blob_count:
        raise ArtifactVerificationError("candidate dataset blob_count mismatch")

    view_index = _read_json(root / "view_index.json")
    view_files = view_index.get("view_files")
    if not isinstance(view_files, Mapping) or set(view_files) != set(VIEW_NAMES):
        raise ArtifactVerificationError("candidate view index is incomplete")
    view_membership: set[str] = set()
    for view_name in VIEW_NAMES:
        relative = _safe_relative_path(str(view_files[view_name]))
        view = _read_json(root / relative)
        _require_status(view, CANDIDATE_FLAGS, artifact_name=relative.as_posix())
        if view.get("view_name") != view_name or not isinstance(view.get("records"), list):
            raise ArtifactVerificationError(f"invalid candidate view manifest: {view_name}")
        if view.get("record_count") != len(view["records"]):
            raise ArtifactVerificationError(f"candidate view record_count mismatch: {view_name}")
        view_membership.update(str(row.get("record_id") or "") for row in view["records"] if isinstance(row, Mapping))
    if view_membership != record_ids:
        raise ArtifactVerificationError("candidate record manifest does not exactly match view membership")

    relation_manifest = _read_json(root / "relation_manifest.json")
    split_by_record = view_index.get("split_by_record")
    if not isinstance(split_by_record, Mapping):
        raise ArtifactVerificationError("candidate view index has no split_by_record")
    try:
        assert_no_hard_relation_crossing(relation_manifest, split_by_record, require_all=True)
    except ValueError as exc:
        raise ArtifactVerificationError(f"candidate hard-relation split verification failed: {exc}") from exc
    record_failures = _record_freeze_failures(records)
    # Candidate records may legitimately be awaiting global audits, but may not
    # contain a direct, already-known critical safety violation.
    if record_failures:
        raise ArtifactVerificationError("candidate record safety failure: " + "; ".join(record_failures))
    return {
        **CANDIDATE_FLAGS,
        "artifact_count": artifact["artifact_count"],
        "artifact_root_sha256": artifact["root_sha256"],
        "ok": True,
        "record_count": len(records),
        "schema_version": "sprite_lab_raw_candidate_verification_v1",
    }


def freeze_candidate_dataset(
    candidate_root: str | Path,
    frozen_root: str | Path,
    gate_evidence: FreezeGateEvidence | Mapping[str, Any],
) -> dict[str, Any]:
    """Create a separate production freeze only after every gate passes."""

    source = Path(candidate_root)
    destination = Path(frozen_root)
    if destination.exists():
        raise FileExistsError(f"frozen dataset output already exists: {destination}")
    verify_candidate_dataset(source)
    evidence = (
        gate_evidence
        if isinstance(gate_evidence, FreezeGateEvidence)
        else FreezeGateEvidence.from_mapping(gate_evidence)
    )
    failures = evidence.failures()
    records = _read_jsonl(source / "record_manifest.jsonl")
    failures.extend(_record_freeze_failures(records))
    if failures:
        raise FreezeGateError(sorted(set(failures)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        _copy_candidate_payload(source, staging)
        dataset = _read_json(source / DATASET_MANIFEST_NAME)
        dataset.update(FROZEN_FLAGS)
        _write_json(staging / DATASET_MANIFEST_NAME, dataset)
        _promote_view_status(staging)
        payload_artifacts = _artifact_entries(
            staging,
            excluded={ARTIFACT_MANIFEST_NAME, FREEZE_MANIFEST_NAME},
        )
        _write_json(
            staging / FREEZE_MANIFEST_NAME,
            {
                **FROZEN_FLAGS,
                "freeze_gates": evidence.canonical(),
                "payload_artifacts": payload_artifacts,
                "payload_root_sha256": _artifact_root_hash(payload_artifacts),
                "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
            },
        )
        _write_artifact_manifest(staging, status=FROZEN_FLAGS)
        verification = verify_frozen_rebuild(staging)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verification


def verify_frozen_rebuild(dataset_path: str | Path, rebuilt_path: str | Path | None = None) -> dict[str, Any]:
    """Verify one frozen tree, and optionally compare a second full rebuild.

    The one-argument form is the public CLI verification API.  Supplying
    ``rebuilt_path`` additionally requires exact recursive file membership and
    byte hashes across two independently created frozen trees.
    """

    first = _verify_one_frozen(dataset_path)
    if rebuilt_path is None:
        return first
    second = _verify_one_frozen(rebuilt_path)
    left = _all_file_hashes(Path(dataset_path))
    right = _all_file_hashes(Path(rebuilt_path))
    if left != right:
        raise ArtifactVerificationError(_inventory_difference_message(left, right, "frozen rebuilds differ"))
    return {
        "artifact_count": len(left),
        "byte_identical": True,
        "first_artifact_root_sha256": first["artifact_root_sha256"],
        "ok": True,
        "production_frozen": True,
        "schema_version": "sprite_lab_raw_frozen_rebuild_verification_v1",
        "second_artifact_root_sha256": second["artifact_root_sha256"],
        "training_authorized": False,
    }


def verify_deterministic_rebuild(first_root: str | Path, second_root: str | Path) -> dict[str, Any]:
    """Compare two candidate or frozen artifact trees byte-for-byte."""

    verify_artifact_manifest(first_root)
    verify_artifact_manifest(second_root)
    left = _all_file_hashes(Path(first_root))
    right = _all_file_hashes(Path(second_root))
    if left != right:
        raise ArtifactVerificationError(_inventory_difference_message(left, right, "dataset rebuilds differ"))
    return {
        "artifact_count": len(left),
        "byte_identical": True,
        "file_hashes": left,
        "ok": True,
        "schema_version": "sprite_lab_raw_deterministic_rebuild_v1",
    }


def verify_artifact_manifest(root: str | Path) -> dict[str, Any]:
    """Reject missing, extra, symlinked, malformed, or byte-changed files."""

    path = Path(root)
    if not path.is_dir():
        raise ArtifactVerificationError(f"dataset root is missing: {path}")
    manifest = _read_json(path / ARTIFACT_MANIFEST_NAME)
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ArtifactVerificationError("unsupported recursive artifact manifest schema")
    expected = manifest.get("artifacts")
    if not isinstance(expected, Mapping):
        raise ArtifactVerificationError("recursive artifact manifest has no artifacts mapping")
    actual = _artifact_entries(path, excluded={ARTIFACT_MANIFEST_NAME})
    if dict(expected) != actual:
        expected_hashes = {
            str(name): str(value.get("sha256") or "") for name, value in expected.items() if isinstance(value, Mapping)
        }
        actual_hashes = {name: str(value["sha256"]) for name, value in actual.items()}
        raise ArtifactVerificationError(
            _inventory_difference_message(expected_hashes, actual_hashes, "artifact verification failed")
        )
    root_hash = _artifact_root_hash(actual)
    if manifest.get("root_sha256") != root_hash:
        raise ArtifactVerificationError("recursive artifact root hash mismatch")
    return {
        "artifact_count": len(actual),
        "ok": True,
        "root_sha256": root_hash,
        "schema_version": "sprite_lab_recursive_artifact_verification_v1",
    }


def _verify_one_frozen(dataset_path: str | Path) -> dict[str, Any]:
    root = Path(dataset_path)
    artifact = verify_artifact_manifest(root)
    dataset = _read_json(root / DATASET_MANIFEST_NAME)
    _require_status(dataset, FROZEN_FLAGS, artifact_name=DATASET_MANIFEST_NAME)
    freeze = _read_json(root / FREEZE_MANIFEST_NAME)
    if freeze.get("schema_version") != FREEZE_MANIFEST_SCHEMA_VERSION:
        raise ArtifactVerificationError("unsupported freeze manifest schema")
    _require_status(freeze, FROZEN_FLAGS, artifact_name=FREEZE_MANIFEST_NAME)
    gate_value = freeze.get("freeze_gates")
    if not isinstance(gate_value, Mapping) or not isinstance(gate_value.get("gates"), Mapping):
        raise ArtifactVerificationError("freeze manifest has no canonical gate evidence")
    evidence = FreezeGateEvidence.from_mapping(gate_value["gates"])
    if evidence.failures() or gate_value.get("passed") is not True:
        raise ArtifactVerificationError("frozen dataset records failing gate evidence")
    expected_payload = freeze.get("payload_artifacts")
    if not isinstance(expected_payload, Mapping):
        raise ArtifactVerificationError("freeze manifest has no payload artifact inventory")
    actual_payload = _artifact_entries(root, excluded={ARTIFACT_MANIFEST_NAME, FREEZE_MANIFEST_NAME})
    if dict(expected_payload) != actual_payload:
        raise ArtifactVerificationError("frozen payload artifact inventory mismatch")
    if freeze.get("payload_root_sha256") != _artifact_root_hash(actual_payload):
        raise ArtifactVerificationError("frozen payload root hash mismatch")
    records = _read_jsonl(root / "record_manifest.jsonl")
    blob_count = _verify_blob_bindings(root, records)
    if dataset.get("blob_count") != blob_count:
        raise ArtifactVerificationError("frozen dataset blob_count mismatch")
    failures = _record_freeze_failures(records)
    if failures:
        raise ArtifactVerificationError("frozen record safety failure: " + "; ".join(failures))
    return {
        **FROZEN_FLAGS,
        "artifact_count": artifact["artifact_count"],
        "artifact_root_sha256": artifact["root_sha256"],
        "ok": True,
        "record_count": len(records),
        "schema_version": "sprite_lab_raw_frozen_verification_v1",
    }


def _record_freeze_failures(records: Sequence[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in records:
        record_id = str(row.get("record_id") or "<missing-record-id>")
        decision = _record_decision(row)
        if decision != "accept":
            failures.append(f"{record_id}: included suitability decision is {decision}")
        for field in ("critical_conflict", "unresolved_critical_conflict", "critical_field_conflict"):
            if row.get(field) is True:
                failures.append(f"{record_id}: unresolved critical conflict ({field})")
        for field in ("unresolved_critical_conflicts", "critical_field_conflicts"):
            value = row.get(field)
            if (isinstance(value, int) and value > 0) or (
                isinstance(value, Sequence) and not isinstance(value, str) and len(value) > 0
            ):
                failures.append(f"{record_id}: unresolved critical conflict ({field})")
        for field in ("filename_leakage", "filename_leakage_detected", "blind_payload_filename_leakage"):
            if row.get(field) is True:
                failures.append(f"{record_id}: filename leakage ({field})")
        if isinstance(row.get("filename_leakage_count"), int) and int(row["filename_leakage_count"]) > 0:
            failures.append(f"{record_id}: filename leakage count is nonzero")
        if row.get("source_binding_valid") is False:
            failures.append(f"{record_id}: invalid source binding")
        provenance = row.get("provenance_status")
        if provenance is not None:
            normalized = str(provenance).strip().casefold()
            if not normalized or any(token in normalized for token in ("missing", "unknown", "unverified")):
                failures.append(f"{record_id}: invalid provenance status")
        if "license" in row and not _valid_license(row.get("license")):
            failures.append(f"{record_id}: invalid or unknown license")
        supervision = row.get("supervision_class")
        if supervision is not None and supervision not in SUPERVISION_CLASSES:
            failures.append(f"{record_id}: invalid supervision class")
        failures.extend(_mask_failures(record_id, row.get("field_masks")))
        audit_status = row.get("audit_status")
        if audit_status is not None and str(audit_status).casefold() not in {"pass", "passed", "ok", "accepted"}:
            failures.append(f"{record_id}: audit status is not passing")
    return sorted(set(failures))


def _mask_failures(record_id: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{record_id}: field_masks is not an object"]
    failures = []
    non_target_states = {"missing", "abstained", "unknown", "oov", "not_applicable", "conflicted"}
    for field_name, mask in value.items():
        if isinstance(mask, bool):
            continue
        if not isinstance(mask, Mapping):
            failures.append(f"{record_id}: invalid mask for {field_name}")
            continue
        if mask.get("valid") is False:
            failures.append(f"{record_id}: invalid mask for {field_name}")
        supervision = mask.get("supervision_class")
        if supervision is not None and supervision not in SUPERVISION_CLASSES:
            failures.append(f"{record_id}: invalid mask supervision class for {field_name}")
        state = str(mask.get("state") or mask.get("target_state") or "").casefold()
        included = mask.get("included") is True or mask.get("target_present") is True
        if included and state in non_target_states:
            failures.append(f"{record_id}: non-target state included by mask for {field_name}")
    return failures


def _valid_license(value: Any) -> bool:
    if isinstance(value, str):
        name = value
    elif isinstance(value, Mapping):
        name = str(value.get("spdx_id") or value.get("license") or value.get("name") or "")
    else:
        return False
    normalized = name.strip().casefold()
    return bool(normalized) and normalized not in {"unknown", "unverified", "none", "n/a"}


def _record_decision(row: Mapping[str, Any]) -> str:
    for field in ("suitability_status", "suitability_decision"):
        if row.get(field) is not None:
            value = str(row[field]).strip().casefold()
            return value or "unknown"
    nested = row.get("source_suitability")
    if isinstance(nested, Mapping):
        value = str(nested.get("status", nested.get("decision")) or "").strip().casefold()
        return value or "missing"
    return "missing"


def _copy_candidate_payload(source: Path, staging: Path) -> None:
    for item in sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()):
        relative = item.relative_to(source)
        if item.is_symlink():
            raise ArtifactVerificationError(f"symlinked dataset artifact is forbidden: {relative.as_posix()}")
        if item.is_dir():
            (staging / relative).mkdir(parents=True, exist_ok=True)
        elif relative.as_posix() not in {ARTIFACT_MANIFEST_NAME, DATASET_MANIFEST_NAME}:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)


def _copy_bound_blobs(records: Sequence[Mapping[str, Any]], blob_source_root: str | Path | None, staging: Path) -> int:
    bindings: dict[str, str] = {}
    for row in records:
        blob_path_value = row.get("blob_path")
        if blob_path_value is None:
            continue
        relative = _safe_relative_path(str(blob_path_value))
        if not relative.parts or relative.parts[0] != "blobs":
            raise RawFreezeError(f"candidate blob_path must be below blobs/: {blob_path_value!r}")
        blob_id = str(row.get("blob_id") or "")
        previous = bindings.get(relative.as_posix())
        if previous is not None and previous != blob_id:
            raise RawFreezeError(f"blob path collision at {relative.as_posix()}")
        bindings[relative.as_posix()] = blob_id
    if bindings and blob_source_root is None:
        raise RawFreezeError("blob_source_root is required for records with blob_path bindings")
    if blob_source_root is None:
        return 0
    source_root = Path(blob_source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"blob source root is missing: {source_root}")
    for relative_value, blob_id in sorted(bindings.items()):
        relative = _safe_relative_path(relative_value)
        source = source_root / relative
        if not source.is_file() or source.is_symlink():
            raise ArtifactVerificationError(f"bound source blob is missing or unsafe: {relative_value}")
        actual = _file_sha256(source)
        if actual != blob_id:
            raise ArtifactVerificationError(
                f"content-addressed source blob mismatch for {relative_value}: expected {blob_id}, got {actual}"
            )
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return len(bindings)


def _verify_blob_bindings(root: Path, records: Sequence[Mapping[str, Any]]) -> int:
    expected: dict[str, str] = {}
    for row in records:
        blob_path_value = row.get("blob_path")
        if blob_path_value is None:
            continue
        relative = _safe_relative_path(str(blob_path_value))
        if not relative.parts or relative.parts[0] != "blobs":
            raise ArtifactVerificationError(f"candidate blob_path must be below blobs/: {blob_path_value!r}")
        blob_id = str(row.get("blob_id") or "")
        if not re_full_sha256(blob_id):
            raise ArtifactVerificationError(f"invalid blob_id for {row.get('record_id')}")
        previous = expected.get(relative.as_posix())
        if previous is not None and previous != blob_id:
            raise ArtifactVerificationError(f"blob path collision at {relative.as_posix()}")
        expected[relative.as_posix()] = blob_id
    actual_paths = (
        {
            item.relative_to(root).as_posix()
            for item in (root / "blobs").rglob("*")
            if item.is_file() and not item.is_symlink()
        }
        if (root / "blobs").is_dir()
        else set()
    )
    if actual_paths != set(expected):
        raise ArtifactVerificationError(
            f"content-addressed blob inventory mismatch: missing={sorted(set(expected) - actual_paths)}, "
            f"extra={sorted(actual_paths - set(expected))}"
        )
    for relative, blob_id in expected.items():
        if _file_sha256(root / _safe_relative_path(relative)) != blob_id:
            raise ArtifactVerificationError(f"content-addressed blob mismatch: {relative}")
    return len(expected)


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _promote_view_status(root: Path) -> None:
    index_path = root / "view_index.json"
    index = _read_json(index_path)
    index.update(FROZEN_FLAGS)
    _write_json(index_path, index)
    for view_name in VIEW_NAMES:
        path = root / "candidate_view_manifests" / f"{view_name}.json"
        view = _read_json(path)
        view.update(FROZEN_FLAGS)
        _write_json(path, view)


def _write_artifact_manifest(root: Path, *, status: Mapping[str, bool]) -> None:
    artifacts = _artifact_entries(root, excluded={ARTIFACT_MANIFEST_NAME})
    _write_json(
        root / ARTIFACT_MANIFEST_NAME,
        {
            **dict(status),
            "artifacts": artifacts,
            "root_sha256": _artifact_root_hash(artifacts),
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        },
    )


def _artifact_entries(root: Path, *, excluded: set[str]) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        raise ArtifactVerificationError(f"artifact root is missing: {root}")
    entries: dict[str, dict[str, Any]] = {}
    for item in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise ArtifactVerificationError(f"symlinked dataset artifact is forbidden: {relative}")
        if not item.is_file() or relative in excluded:
            continue
        entries[relative] = {"sha256": _file_sha256(item), "size_bytes": item.stat().st_size}
    return entries


def _artifact_root_hash(entries: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(entries))).hexdigest()


def _all_file_hashes(root: Path) -> dict[str, str]:
    return {
        item.relative_to(root).as_posix(): _file_sha256(item)
        for item in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
        if item.is_file() and not item.is_symlink()
    }


def _inventory_difference_message(left: Mapping[str, str], right: Mapping[str, str], prefix: str) -> str:
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    changed = sorted(name for name in set(left) & set(right) if left[name] != right[name])
    return f"{prefix}: missing={missing}, extra={extra}, changed={changed}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_index(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in records:
        row = dict(source)
        record_id = str(row.get("record_id") or "")
        if not record_id or record_id in result:
            raise RawFreezeError(f"missing or duplicate record_id: {record_id!r}")
        result[record_id] = row
    return result


def _require_status(value: Mapping[str, Any], expected: Mapping[str, bool], *, artifact_name: str) -> None:
    actual = {name: value.get(name) for name in expected}
    if actual != dict(expected):
        raise ArtifactVerificationError(f"invalid freeze status flags in {artifact_name}: {actual}")


def _safe_relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise ArtifactVerificationError(f"unsafe artifact path: {value!r}")
    return Path(*posix.parts)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RawFreezeError(f"artifact is not canonical JSON: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in sorted(rows, key=lambda value: str(value.get("record_id") or ""))
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactVerificationError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactVerificationError(f"cannot read JSONL artifact {path}: {exc}") from exc
    result = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactVerificationError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ArtifactVerificationError(f"JSONL row must be an object at {path}:{line_number}")
        result.append(value)
    return result
