"""Transactional Harvest-to-Dataset intake for conditioned Dataset-v5.

The callback is intentionally narrower than the conditioned builder.  It copies
an already verified Harvest artifact set into unique repository-managed work,
runs the ordinary Dataset intake contract there, and publishes only an atomic
opaque receipt after every byte and managed document has been revalidated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from spritelab.product_core import ProductStatus, ProjectContext, strict_json_dumps, strict_json_loads
from spritelab.product_features.dataset.intake import DatasetIntakeService, discover_source_packs
from spritelab.product_features.dataset.managed import validate_managed_dataset_output
from spritelab.product_features.dataset.sidecar import save_grouping, save_pack_metadata, sidecar_path
from spritelab.product_features.harvest.storage import scan_artifacts
from spritelab.product_features.harvest.trusted_backend import (
    AcquiredFile,
    DatasetImportRequest,
    DatasetImportResult,
    HarvestLimits,
)
from spritelab.training.campaign import stable_hash
from spritelab.utils.safe_fs import (
    UnsafeFilesystemOperation,
    atomic_write_bytes,
    atomic_write_text,
    require_confined_path,
)
from spritelab.v3.run_state import lock_file

HANDOFF_SCHEMA = "spritelab.harvest.dataset-handoff.v2"
ARTIFACT_MANIFEST_SCHEMA = "spritelab.harvest.artifact-manifest.v1"
HARVEST_IMPORT_RECEIPT_SCHEMA = "spritelab.harvest.dataset-import-receipt.v1"
MANAGED_IMPORT_RECEIPT_SCHEMA = "spritelab.dataset.conditioned-import-receipt.v1"
MANAGED_IMPORT_INVENTORY_SCHEMA = "spritelab.dataset.conditioned-import-inventory.v1"

ALLOWED_LICENSES = frozenset({"cc0-1.0", "public-domain"})
REFERENCE_PATTERN = re.compile(r"^dataset\.[0-9a-f]{24}$")
RUN_ID_PATTERN = re.compile(r"^harvest-[a-z0-9][a-z0-9-]{5,80}$")
KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_MAX_FILES = 5_000
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_DEPTH = 8
_MAX_JSON_BYTES = 128 * 1024 * 1024


class ConditionedIntakeError(ValueError):
    """A managed import cannot establish the required immutable bindings."""


class ConditionedDatasetImportAdapter:
    """Harvest callback that publishes an opaque, DatasetIntake-backed receipt."""

    callback_id = "dataset.conditioned-intake"
    code_identity_sha256 = stable_hash(
        {
            "adapter": callback_id,
            "contract": MANAGED_IMPORT_RECEIPT_SCHEMA,
            "revision": 1,
            "publication": "atomic-receipt-after-managed-validation",
        }
    )

    def import_harvest(
        self,
        request: DatasetImportRequest,
        *,
        idempotency_key: str,
    ) -> DatasetImportResult:
        if not isinstance(request, DatasetImportRequest):
            raise ConditionedIntakeError("Harvest supplied an invalid Dataset import request.")
        if KEY_PATTERN.fullmatch(str(idempotency_key)) is None:
            raise ConditionedIntakeError("Dataset import requires a valid idempotency key.")

        verified = _verify_harvest_request(request)
        project_root = verified["project_root"]
        request_identity = stable_hash(
            {
                "schema_version": "spritelab.dataset.conditioned-import-request.v1",
                "callback_id": self.callback_id,
                "callback_code_identity_sha256": self.code_identity_sha256,
                "run_id": request.run_id,
                "idempotency_key": idempotency_key,
                "handoff_identity": verified["handoff_identity"],
                "artifact_manifest_identity": verified["artifact_manifest_identity"],
            }
        )
        reference = f"dataset.{request_identity[:24]}"
        datasets_root = _ensure_directory(project_root / "datasets", project_root)
        receipts_root = _ensure_directory(datasets_root / "conditioned_intake_receipts", datasets_root)
        work_root = _ensure_directory(datasets_root / "conditioned_intake_work", datasets_root)
        receipt_path = require_confined_path(receipts_root / f"{reference}.json", receipts_root)
        lock_path = require_confined_path(receipts_root / f".{reference}.lock", receipts_root)

        with lock_file(lock_path, timeout=600.0):
            if os.path.lexists(receipt_path):
                receipt = _load_managed_receipt(project_root, reference, require_harvest_receipt=False)
                if receipt["request_identity"] != request_identity:
                    raise ConditionedIntakeError("The Dataset import reference is bound to another request.")
                return DatasetImportResult(
                    reference,
                    int(receipt["accepted_count"]),
                    int(receipt["quarantined_count"]),
                )

            work = require_confined_path(work_root / f"intake-{uuid.uuid4().hex}", work_root)
            work.mkdir()
            source_root = require_confined_path(work / "source", work)
            output_root = require_confined_path(work / "managed", work)
            source_root.mkdir()
            stage = "copy"
            try:
                _copy_verified_artifacts(
                    verified["artifacts_root"],
                    source_root,
                    verified["artifact_manifest"],
                )
                # A second source scan proves that the callback did not change or
                # replace raw Harvest bytes while publishing its independent copy.
                _verify_artifact_tree(
                    verified["artifacts_root"],
                    verified["artifact_manifest"],
                )
                copied_manifest = _verify_artifact_tree(source_root, verified["artifact_manifest"])
                if copied_manifest != verified["artifact_manifest"]:
                    raise ConditionedIntakeError("The managed source copy does not reproduce the Harvest manifest.")

                stage = "source_evidence"
                context = ProjectContext(
                    project_root=project_root,
                    config={},
                    runs_directory=project_root / "runs" / "v3",
                )
                save_grouping(project_root, source_root, ["."])
                _root, png_paths, packs = discover_source_packs(source_root, context=context)
                if len(packs) != 1 or packs[0].relative_root != ".":
                    raise ConditionedIntakeError("The copied Harvest source did not resolve to one bound source pack.")
                source = verified["source"]
                license_record = verified["license"]
                sidecar = save_pack_metadata(
                    project_root,
                    source_root,
                    packs[0],
                    {
                        "creator_or_rights_holder": source["creator"],
                        "pack_title": source["title"],
                        "source_type": "other_downloaded",
                        "source_page_url": source["source_page"],
                        "original_work_declaration": False,
                        "license_identifier": _sidecar_license(str(license_record["identifier"])),
                        "license_url": license_record["evidence_url"],
                        "license_evidence_file": None,
                        "attribution_text": str(license_record.get("attribution_text") or "") or None,
                        "permission_confirmed": False,
                        "notes": f"Verified Harvest handoff {request.run_id}.",
                    },
                    covered_byte_hashes=[_file_sha256(path) for path in png_paths],
                )

                stage = "dataset_intake"
                result = DatasetIntakeService().build(source_root, output_root=output_root, context=context)
                if result.status in {ProductStatus.BLOCKED, ProductStatus.FAILED, ProductStatus.UNAVAILABLE}:
                    raise ConditionedIntakeError("Dataset intake could not publish a training-eligible managed output.")
                validate_managed_dataset_output(output_root, context=context, require_datasets_root=True)
                accepted_paths = _accepted_source_paths(output_root, source_root, verified["artifact_manifest"])
                if not accepted_paths:
                    raise ConditionedIntakeError("Dataset intake accepted no source images.")

                stage = "publication_validation"
                source_inventory = _inventory(source_root)
                output_inventory = _inventory(output_root)
                sidecar_file = sidecar_path(project_root, packs[0].pack_id)
                sidecar_identity = _stable_file_identity(sidecar_file, sidecar_file.parent.lstat().st_dev)
                grouping_file = _grouping_file_for(project_root, source_root)
                grouping_identity = _stable_file_identity(grouping_file, grouping_file.parent.lstat().st_dev)
                artifact_count = int(verified["artifact_manifest"]["artifact_count"])
                receipt_without_identity = {
                    "schema_version": MANAGED_IMPORT_RECEIPT_SCHEMA,
                    "dataset_reference": reference,
                    "request_identity": request_identity,
                    "callback_id": self.callback_id,
                    "callback_code_identity_sha256": self.code_identity_sha256,
                    "harvest": {
                        "run_id": request.run_id,
                        "handoff_identity": verified["handoff_identity"],
                        "request_handoff_identity": verified["request_handoff_identity"],
                        "artifact_manifest_identity": verified["artifact_manifest_identity"],
                        "artifact_manifest_file_sha256": verified["artifact_manifest_file_sha256"],
                        "artifact_set_identity": verified["artifact_manifest"]["artifact_set_identity"],
                        "provenance_identity": verified["handoff"]["provenance_identity"],
                        "source_evidence_binding_identity": verified["handoff"]["source_evidence_binding_identity"],
                    },
                    "handoff_document": verified["handoff"],
                    "artifact_manifest": verified["artifact_manifest"],
                    "source": source,
                    "license": license_record,
                    "managed": {
                        "work_relative_path": work.relative_to(project_root).as_posix(),
                        "source_relative_path": source_root.relative_to(project_root).as_posix(),
                        "output_relative_path": output_root.relative_to(project_root).as_posix(),
                        "source_inventory": source_inventory,
                        "source_inventory_sha256": stable_hash(source_inventory),
                        "output_inventory": output_inventory,
                        "output_inventory_sha256": stable_hash(output_inventory),
                        "intake_result_identity": stable_hash(result.to_dict()),
                        "accepted_relative_paths": accepted_paths,
                        "sidecar_relative_path": sidecar_file.relative_to(project_root).as_posix(),
                        "sidecar_identity": sidecar_identity,
                        "sidecar_record_identity": stable_hash(sidecar),
                        "grouping_relative_path": grouping_file.relative_to(project_root).as_posix(),
                        "grouping_identity": grouping_identity,
                    },
                    "accepted_count": len(accepted_paths),
                    "quarantined_count": artifact_count - len(accepted_paths),
                    "raw_harvest_mutated": False,
                    "atomic_publication": "receipt_pointer_after_validation",
                    "portable_relative_paths": True,
                    "paths_exposed": False,
                    "created_at": _now(),
                }
                receipt = {
                    **receipt_without_identity,
                    "receipt_identity": stable_hash(receipt_without_identity),
                }
                atomic_write_text(
                    receipt_path,
                    strict_json_dumps(receipt, indent=2, sort_keys=True) + "\n",
                )
                # Load through the same boundary used by the conditioned builder.
                published = _load_managed_receipt(project_root, reference, require_harvest_receipt=False)
                return DatasetImportResult(
                    reference,
                    int(published["accepted_count"]),
                    int(published["quarantined_count"]),
                )
            except BaseException:
                failure = {
                    "schema_version": "spritelab.dataset.conditioned-import-failure.v1",
                    "request_identity": request_identity,
                    "stage": stage,
                    "published": False,
                    "retained_for_safe_inspection": True,
                    "paths_exposed": False,
                    "created_at": _now(),
                }
                try:
                    atomic_write_text(
                        work / "failure.json", strict_json_dumps(failure, indent=2, sort_keys=True) + "\n"
                    )
                except OSError:
                    pass
                raise


def managed_intake_inventory(project_root: str | Path) -> list[dict[str, Any]]:
    """Passively inventory only atomically published managed intake receipts."""

    project = Path(project_root).resolve()
    root = project / "datasets" / "conditioned_intake_receipts"
    if not _safe_directory(root):
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(root.glob("dataset.*.json"), key=lambda value: value.name):
        reference = path.name.removesuffix(".json")
        if REFERENCE_PATTERN.fullmatch(reference) is None:
            continue
        try:
            receipt = _read_json(path, root)
        except ConditionedIntakeError:
            continue
        harvest = receipt.get("harvest") if isinstance(receipt.get("harvest"), Mapping) else {}
        source = receipt.get("source") if isinstance(receipt.get("source"), Mapping) else {}
        results.append(
            {
                "dataset_reference": reference,
                "harvest_run_id": str(harvest.get("run_id") or ""),
                "source_id": str(source.get("source_id") or ""),
                "source_title": str(source.get("title") or ""),
                "accepted_count": _integer(receipt.get("accepted_count"), 0),
                "quarantined_count": _integer(receipt.get("quarantined_count"), 0),
                "status": "COMPLETE",
                "paths_exposed": False,
            }
        )
    return results


def load_managed_intake(project_root: str | Path, dataset_reference: str) -> dict[str, Any]:
    """Revalidate a published intake, its managed bytes, and original handoff."""

    project = Path(project_root).resolve()
    receipt = _load_managed_receipt(project, dataset_reference, require_harvest_receipt=True)
    managed = receipt["managed"]
    source_root = _project_relative_path(project, managed["source_relative_path"], expected="directory")
    output_root = _project_relative_path(project, managed["output_relative_path"], expected="directory")
    context = ProjectContext(project_root=project, config={}, runs_directory=project / "runs" / "v3")
    validate_managed_dataset_output(output_root, context=context, require_datasets_root=True)
    if _inventory(source_root) != managed["source_inventory"]:
        raise ConditionedIntakeError("Managed intake source bytes changed after publication.")
    if _inventory(output_root) != managed["output_inventory"]:
        raise ConditionedIntakeError("Managed Dataset intake output changed after publication.")
    if stable_hash(managed["source_inventory"]) != managed["source_inventory_sha256"]:
        raise ConditionedIntakeError("Managed intake source inventory identity is inconsistent.")
    if stable_hash(managed["output_inventory"]) != managed["output_inventory_sha256"]:
        raise ConditionedIntakeError("Managed intake output inventory identity is inconsistent.")

    sidecar_file = _project_relative_path(project, managed["sidecar_relative_path"], expected="file")
    grouping_file = _project_relative_path(project, managed["grouping_relative_path"], expected="file")
    if _stable_file_identity(sidecar_file, sidecar_file.parent.lstat().st_dev) != managed["sidecar_identity"]:
        raise ConditionedIntakeError("Managed intake provenance sidecar changed after publication.")
    if _stable_file_identity(grouping_file, grouping_file.parent.lstat().st_dev) != managed["grouping_identity"]:
        raise ConditionedIntakeError("Managed intake pack grouping changed after publication.")

    manifest = receipt["artifact_manifest"]
    if _verify_artifact_tree(source_root, manifest) != manifest:
        raise ConditionedIntakeError("Managed source files no longer reproduce the original artifact manifest.")
    accepted = _accepted_source_paths(output_root, source_root, manifest)
    if accepted != managed["accepted_relative_paths"]:
        raise ConditionedIntakeError("Managed intake dispositions changed after publication.")

    harvest = receipt["harvest"]
    run_id = str(harvest["run_id"])
    run_root = _project_relative_path(project, f"harvest_runs/{run_id}", expected="directory")
    handoff = _read_json(run_root / "handoff.json", run_root)
    if handoff != receipt["handoff_document"] or stable_hash(handoff) != harvest["handoff_identity"]:
        raise ConditionedIntakeError("The original Harvest handoff changed after Dataset import.")
    _validate_handoff_document(run_id, handoff, manifest)
    disk_manifest = _read_json(run_root / "artifact_manifest.json", run_root)
    if disk_manifest != manifest or stable_hash(disk_manifest) != harvest["artifact_manifest_identity"]:
        raise ConditionedIntakeError("The original Harvest artifact manifest changed after Dataset import.")

    harvest_receipt = _read_json(run_root / "dataset_import_receipt.json", run_root)
    if (
        harvest_receipt.get("schema_version") != HARVEST_IMPORT_RECEIPT_SCHEMA
        or harvest_receipt.get("run_id") != run_id
        or harvest_receipt.get("dataset_reference") != dataset_reference
        or harvest_receipt.get("callback_id") != receipt["callback_id"]
        or harvest_receipt.get("callback_code_identity_sha256") != receipt["callback_code_identity_sha256"]
        or harvest_receipt.get("artifact_manifest_identity") != harvest["artifact_manifest_identity"]
        or harvest_receipt.get("accepted_count") != receipt["accepted_count"]
        or harvest_receipt.get("quarantined_count") != receipt["quarantined_count"]
        or harvest_receipt.get("paths_exposed") is not False
    ):
        raise ConditionedIntakeError("The Harvest Dataset import receipt is missing or inconsistent.")
    harvest_receipt_identity = stable_hash(harvest_receipt)

    source = receipt["source"]
    license_record = receipt["license"]
    return {
        "dataset_reference": dataset_reference,
        "run_id": run_id,
        "run_root": run_root,
        "artifacts_root": source_root,
        "managed_output_root": output_root,
        "handoff": handoff,
        "handoff_identity": harvest["handoff_identity"],
        "harvest_import_receipt_identity": harvest_receipt_identity,
        "managed_intake_receipt_identity": receipt["receipt_identity"],
        "managed_output_inventory_sha256": managed["output_inventory_sha256"],
        "managed_source_inventory_sha256": managed["source_inventory_sha256"],
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": harvest["artifact_manifest_file_sha256"],
        "source_id": source["source_id"],
        "source_title": source["title"],
        "creator": source["creator"],
        "license_id": str(license_record["identifier"]).casefold(),
        "license_evidence": license_record,
        "artifact_count": int(manifest["artifact_count"]),
        "accepted_relative_paths": accepted,
    }


def _verify_harvest_request(request: DatasetImportRequest) -> dict[str, Any]:
    if RUN_ID_PATTERN.fullmatch(str(request.run_id)) is None:
        raise ConditionedIntakeError("Harvest Dataset import run identity is invalid.")
    artifacts = Path(os.path.abspath(request.artifacts_directory))
    run_root = artifacts.parent
    harvest_root = run_root.parent
    project_root = harvest_root.parent
    if artifacts.name != "artifacts" or run_root.name != request.run_id or harvest_root.name != "harvest_runs":
        raise ConditionedIntakeError("Harvest artifacts are outside the expected managed run boundary.")
    try:
        require_confined_path(artifacts, project_root)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedIntakeError("Harvest artifacts are outside the project boundary.") from exc
    for directory in (project_root, harvest_root, run_root, artifacts):
        if not _safe_directory(directory):
            raise ConditionedIntakeError("Harvest Dataset import crosses an unsafe directory boundary.")

    projected = dict(request.handoff)
    request_handoff_identity = stable_hash(projected)
    projected.pop("dataset_import_available", None)
    disk_handoff = _read_json(run_root / "handoff.json", run_root)
    if projected != disk_handoff:
        raise ConditionedIntakeError("Harvest handoff projection disagrees with its durable document.")
    manifest = dict(request.artifact_manifest)
    disk_manifest_path = run_root / "artifact_manifest.json"
    disk_manifest_bytes = _read_regular_bytes(disk_manifest_path, run_root, _MAX_JSON_BYTES)
    try:
        disk_manifest_value = strict_json_loads(disk_manifest_bytes)
    except ValueError as exc:
        raise ConditionedIntakeError("Harvest artifact manifest is invalid.") from exc
    if not isinstance(disk_manifest_value, Mapping) or manifest != dict(disk_manifest_value):
        raise ConditionedIntakeError("Harvest artifact manifest projection disagrees with its durable document.")
    _validate_handoff_document(request.run_id, disk_handoff, manifest)
    if _verify_artifact_tree(artifacts, manifest) != manifest:
        raise ConditionedIntakeError("Harvest artifact bytes do not reproduce their exact manifest.")
    return {
        "project_root": project_root,
        "run_root": run_root,
        "artifacts_root": artifacts,
        "handoff": disk_handoff,
        "handoff_identity": stable_hash(disk_handoff),
        "request_handoff_identity": request_handoff_identity,
        "artifact_manifest": manifest,
        "artifact_manifest_identity": stable_hash(manifest),
        "artifact_manifest_file_sha256": hashlib.sha256(disk_manifest_bytes).hexdigest(),
        "source": dict(disk_handoff["source"]),
        "license": dict(disk_handoff["license"]),
    }


def _validate_handoff_document(run_id: str, handoff: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if (
        handoff.get("schema_version") != HANDOFF_SCHEMA
        or handoff.get("run_id") != run_id
        or handoff.get("handoff_ready") is not True
        or handoff.get("portable_relative_paths") is not True
        or handoff.get("paths_exposed") is not False
        or manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA
    ):
        raise ConditionedIntakeError("Harvest handoff is incomplete or uses an unsupported contract.")
    managed = handoff.get("managed_reference")
    if not isinstance(managed, Mapping) or managed != {"kind": "harvest_run", "run_id": run_id}:
        raise ConditionedIntakeError("Harvest handoff is not bound to its managed run.")
    source = handoff.get("source")
    license_record = handoff.get("license")
    if not isinstance(source, Mapping) or not isinstance(license_record, Mapping):
        raise ConditionedIntakeError("Harvest source or license evidence is missing.")
    source_license = source.get("license")
    if (
        source.get("source_id") != handoff.get("source_id")
        or source_license != license_record
        or str(license_record.get("identifier") or "").casefold() not in ALLOWED_LICENSES
        or license_record.get("permissive_policy") is not True
        or not str(source.get("title") or "").strip()
        or not str(source.get("creator") or "").strip()
        or not str(source.get("source_page") or "").strip()
        or not str(license_record.get("evidence_url") or "").strip()
    ):
        raise ConditionedIntakeError("Harvest provenance or permissive-license evidence is incomplete.")
    identity_fields = (
        "provenance_identity",
        "source_evidence_binding_identity",
        "backend_capability_identity",
        "limits_identity",
        "acquisition_receipt_identity",
        "artifact_manifest_identity",
        "artifact_set_identity",
    )
    if any(SHA256_PATTERN.fullmatch(str(handoff.get(key) or "")) is None for key in identity_fields):
        raise ConditionedIntakeError("Harvest handoff identity bindings are invalid.")
    expected_provenance = stable_hash(
        {"source": dict(source), "acquisition_receipt_identity": handoff["acquisition_receipt_identity"]}
    )
    evidence = source.get("evidence_binding")
    if (
        handoff["provenance_identity"] != expected_provenance
        or not isinstance(evidence, Mapping)
        or evidence.get("binding_identity") != handoff["source_evidence_binding_identity"]
        or stable_hash(dict(manifest)) != handoff["artifact_manifest_identity"]
        or manifest.get("artifact_set_identity") != handoff["artifact_set_identity"]
        or manifest.get("files") != handoff.get("files")
        or manifest.get("artifact_count") != handoff.get("artifact_count")
        or manifest.get("total_bytes") != handoff.get("total_bytes")
    ):
        raise ConditionedIntakeError("Harvest handoff provenance or artifact bindings are inconsistent.")


def _verify_artifact_tree(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) > _MAX_FILES:
        raise ConditionedIntakeError("Harvest artifact manifest is missing or oversized.")
    expected: list[AcquiredFile] = []
    for raw in files:
        if not isinstance(raw, Mapping):
            raise ConditionedIntakeError("Harvest artifact manifest contains an invalid row.")
        expected_sha = str(raw.get("actual_sha256") or raw.get("sha256") or "")
        if raw.get("expected_sha256") not in {None, "", expected_sha}:
            raise ConditionedIntakeError("Harvest expected and actual artifact identities disagree.")
        try:
            expected.append(
                AcquiredFile(
                    relative_path=_canonical_relative(str(raw.get("relative_path") or "")),
                    byte_count=int(raw.get("byte_count")),
                    sha256=expected_sha,
                    mime_type=str(raw.get("mime_type") or ""),
                    usable=raw.get("usable") is True,
                    quarantine_reason=(
                        str(raw.get("quarantine_reason")) if raw.get("quarantine_reason") is not None else None
                    ),
                    taxonomy=tuple(str(value) for value in raw.get("taxonomy") or ()),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ConditionedIntakeError("Harvest artifact manifest contains an invalid file identity.") from exc
    limits = HarvestLimits(
        max_files=_MAX_FILES,
        max_file_bytes=_MAX_FILE_BYTES,
        max_total_bytes=_MAX_TOTAL_BYTES,
        max_response_bytes=_MAX_TOTAL_BYTES,
        max_depth=_MAX_DEPTH,
        max_archive_uncompressed_bytes=1024 * 1024 * 1024,
    )
    try:
        return scan_artifacts(root, limits, expected_files=tuple(expected))
    except (OSError, ValueError) as exc:
        raise ConditionedIntakeError("Managed artifact bytes failed complete per-file re-verification.") from exc


def _copy_verified_artifacts(source: Path, destination: Path, manifest: Mapping[str, Any]) -> None:
    for raw in manifest["files"]:
        relative = _canonical_relative(str(raw["relative_path"]))
        target = require_confined_path(destination.joinpath(*PurePosixPath(relative).parts), destination)
        content = _read_regular_bytes(
            require_confined_path(source.joinpath(*PurePosixPath(relative).parts), source),
            source,
            _MAX_FILE_BYTES,
        )
        expected = str(raw.get("actual_sha256") or raw.get("sha256") or "")
        if len(content) != int(raw["byte_count"]) or hashlib.sha256(content).hexdigest() != expected:
            raise ConditionedIntakeError("A Harvest artifact changed while it was copied.")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(target, content)
        copied = _read_regular_bytes(target, destination, _MAX_FILE_BYTES)
        if copied != content:
            raise ConditionedIntakeError("A managed source copy changed during atomic publication.")


def _accepted_source_paths(output: Path, source: Path, manifest: Mapping[str, Any]) -> list[str]:
    rows = _read_jsonl(output / "items.jsonl", output)
    known = {
        str(row["relative_path"]): str(row.get("actual_sha256") or row.get("sha256") or "")
        for row in manifest["files"]
        if isinstance(row, Mapping)
    }
    accepted: list[str] = []
    for item in rows:
        if item.get("current_disposition") != "accepted":
            continue
        relative = _canonical_relative(str(item.get("relative_path") or ""))
        if relative not in known or item.get("byte_sha256") != known[relative]:
            raise ConditionedIntakeError("Dataset intake accepted an image outside the original artifact manifest.")
        expected_path = source.joinpath(*PurePosixPath(relative).parts).resolve(strict=True)
        try:
            item_path = Path(str(item.get("source_path") or "")).resolve(strict=True)
        except OSError as exc:
            raise ConditionedIntakeError("Dataset intake source binding is unavailable.") from exc
        if item_path != expected_path:
            raise ConditionedIntakeError("Dataset intake source binding changed after preprocessing.")
        accepted.append(relative)
    if len(accepted) != len(set(accepted)):
        raise ConditionedIntakeError("Dataset intake accepted duplicate source identities.")
    return sorted(accepted)


def _load_managed_receipt(
    project_root: Path,
    dataset_reference: str,
    *,
    require_harvest_receipt: bool,
) -> dict[str, Any]:
    del require_harvest_receipt  # The caller performs the Harvest-receipt check after managed validation.
    if REFERENCE_PATTERN.fullmatch(str(dataset_reference)) is None:
        raise ConditionedIntakeError("Managed Dataset reference is invalid.")
    receipts_root = project_root / "datasets" / "conditioned_intake_receipts"
    if not _safe_directory(receipts_root):
        raise ConditionedIntakeError("Managed Dataset receipt storage is unavailable.")
    receipt = _read_json(receipts_root / f"{dataset_reference}.json", receipts_root)
    identity = receipt.get("receipt_identity")
    payload = dict(receipt)
    payload.pop("receipt_identity", None)
    if (
        receipt.get("schema_version") != MANAGED_IMPORT_RECEIPT_SCHEMA
        or receipt.get("dataset_reference") != dataset_reference
        or receipt.get("paths_exposed") is not False
        or receipt.get("portable_relative_paths") is not True
        or not SHA256_PATTERN.fullmatch(str(identity or ""))
        or stable_hash(payload) != identity
        or not isinstance(receipt.get("managed"), Mapping)
        or not isinstance(receipt.get("harvest"), Mapping)
        or not isinstance(receipt.get("handoff_document"), Mapping)
        or not isinstance(receipt.get("artifact_manifest"), Mapping)
        or not isinstance(receipt.get("source"), Mapping)
        or not isinstance(receipt.get("license"), Mapping)
    ):
        raise ConditionedIntakeError("Managed Dataset import receipt is malformed or inconsistent.")
    return receipt


def _project_relative_path(project: Path, value: Any, *, expected: str) -> Path:
    relative = _canonical_relative(str(value or ""))
    try:
        path = require_confined_path(project.joinpath(*PurePosixPath(relative).parts), project)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedIntakeError("Managed Dataset receipt contains an unsafe relative path.") from exc
    if expected == "directory" and not _safe_directory(path):
        raise ConditionedIntakeError("Managed Dataset directory is unavailable or unsafe.")
    if expected == "file":
        _read_regular_bytes(path, project, _MAX_JSON_BYTES)
    return path


def _grouping_file_for(project: Path, source: Path) -> Path:
    from spritelab.product_features.dataset.sidecar import grouping_path

    return grouping_path(project, source)


def _sidecar_license(identifier: str) -> str:
    normalized = identifier.casefold()
    if normalized == "cc0-1.0":
        return "cc0"
    if normalized == "public-domain":
        return "public_domain"
    raise ConditionedIntakeError("The Harvest license is outside the conditioned intake policy.")


def _inventory(root: Path) -> dict[str, Any]:
    if not _safe_directory(root):
        raise ConditionedIntakeError("Managed inventory root is unavailable or unsafe.")
    device = root.lstat().st_dev
    files: dict[str, dict[str, Any]] = {}
    collision_keys: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted(directory_names):
            child = require_confined_path(parent / name, root)
            if not _safe_directory(child) or child.lstat().st_dev != device:
                raise ConditionedIntakeError("Managed inventory crosses a link, mount, or filesystem boundary.")
        for name in sorted(file_names):
            child = require_confined_path(parent / name, root)
            relative = _canonical_relative(child.relative_to(root).as_posix())
            collision = unicodedata.normalize("NFC", relative).casefold()
            if collision in collision_keys:
                raise ConditionedIntakeError("Managed inventory contains a case or Unicode path collision.")
            collision_keys.add(collision)
            files[relative] = _stable_file_identity(child, device)
    normalized = dict(sorted(files.items()))
    return {
        "schema_version": MANAGED_IMPORT_INVENTORY_SCHEMA,
        "files": normalized,
        "file_count": len(normalized),
        "total_bytes": sum(int(value["byte_count"]) for value in normalized.values()),
    }


def _stable_file_identity(path: Path, root_device: int) -> dict[str, Any]:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_link_or_reparse(path)
        or before.st_nlink != 1
        or before.st_dev != root_device
    ):
        raise ConditionedIntakeError("Managed inventory entries must be owned single-link regular files.")
    content = _read_regular_bytes(path, path.parent, max(_MAX_JSON_BYTES, before.st_size))
    after = path.lstat()
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ConditionedIntakeError("A managed file changed while its identity was computed.")
    return {"sha256": hashlib.sha256(content).hexdigest(), "byte_count": before.st_size}


def _read_regular_bytes(path: Path, root: Path, max_bytes: int) -> bytes:
    try:
        target = require_confined_path(path, root)
        before = target.lstat()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedIntakeError("A required managed file is unavailable or unsafe.") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _is_link_or_reparse(target)
        or before.st_nlink != 1
        or before.st_size > max_bytes
    ):
        raise ConditionedIntakeError("A required managed file is not a bounded single-link regular file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_nlink != 1
        ):
            raise ConditionedIntakeError("A managed file changed while it was opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if len(content) > max_bytes:
        raise ConditionedIntakeError("A required managed file exceeds its byte limit.")
    after = target.lstat()
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or _is_link_or_reparse(target)
    ):
        raise ConditionedIntakeError("A managed file changed while it was read.")
    return content


def _read_json(path: Path, root: Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(_read_regular_bytes(path, root, _MAX_JSON_BYTES))
    except ValueError as exc:
        raise ConditionedIntakeError("A required managed JSON document is invalid.") from exc
    if not isinstance(value, Mapping):
        raise ConditionedIntakeError("A required managed JSON document must be an object.")
    return dict(value)


def _read_jsonl(path: Path, root: Path) -> list[dict[str, Any]]:
    content = _read_regular_bytes(path, root, _MAX_JSON_BYTES)
    rows: list[dict[str, Any]] = []
    try:
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("JSONL row is not an object")
            rows.append(dict(value))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ConditionedIntakeError("A required managed JSONL document is invalid.") from exc
    return rows


def _ensure_directory(path: Path, parent: Path) -> Path:
    try:
        target = require_confined_path(path, parent)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedIntakeError("Managed Dataset storage is outside the project boundary.") from exc
    if os.path.lexists(target):
        if not _safe_directory(target):
            raise ConditionedIntakeError("Managed Dataset storage is linked, mounted, or not a directory.")
    else:
        if not _safe_directory(parent):
            raise ConditionedIntakeError("Managed Dataset storage parent is unsafe.")
        target.mkdir()
    return target


def _safe_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and not _metadata_is_link_or_reparse(metadata) and not path.is_mount()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        return _metadata_is_link_or_reparse(path.lstat())
    except OSError:
        return False


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _canonical_relative(value: str) -> str:
    if not value or value != unicodedata.normalize("NFC", value) or "\\" in value or "\x00" in value:
        raise ConditionedIntakeError("A managed relative path is not canonical or portable.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != value
    ):
        raise ConditionedIntakeError("A managed relative path is not canonical or portable.")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path, path.parent, _MAX_FILE_BYTES)).hexdigest()


def _integer(value: Any, default: int) -> int:
    return value if type(value) is int else default


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "MANAGED_IMPORT_RECEIPT_SCHEMA",
    "ConditionedDatasetImportAdapter",
    "ConditionedIntakeError",
    "load_managed_intake",
    "managed_intake_inventory",
]
