from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pytest
from PIL import Image

import spritelab.product_features.conditioned_v5.audit_runner as audit_runner
from spritelab.training.campaign import stable_hash


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _inventory(root: Path) -> dict[str, Any]:
    files = {
        path.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    files = dict(sorted(files.items()))
    return {
        "schema_version": "spritelab.dataset.conditioned-import-inventory.v1",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["byte_count"]) for item in files.values()),
    }


def _canonical_document(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_receipt(fixture: dict[str, Any]) -> None:
    receipt = fixture["receipt"]
    payload = dict(receipt)
    payload.pop("receipt_identity", None)
    receipt["receipt_identity"] = stable_hash(payload)
    fixture["binding"]["managed_intake_receipt_identity"] = receipt["receipt_identity"]
    fixture["receipt_path"].write_bytes(_canonical_document(receipt))


def _rebind_manifest(fixture: dict[str, Any]) -> None:
    receipt = fixture["receipt"]
    managed = receipt["managed"]
    manifest = managed["derived_sheet_manifest"]
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_identity", None)
    manifest["manifest_identity"] = stable_hash(manifest_payload)
    manifest_bytes = _canonical_document(manifest)
    fixture["manifest_path"].write_bytes(manifest_bytes)
    inventory = managed["derived_inventory"]
    inventory["files"]["manifest.json"] = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "byte_count": len(manifest_bytes),
    }
    inventory["files"] = dict(sorted(inventory["files"].items()))
    inventory["file_count"] = len(inventory["files"])
    inventory["total_bytes"] = sum(int(item["byte_count"]) for item in inventory["files"].values())
    managed["derived_sheet_manifest_identity"] = manifest["manifest_identity"]
    managed["derived_inventory_sha256"] = stable_hash(inventory)
    fixture["binding"]["derived_sheet_manifest_identity"] = manifest["manifest_identity"]
    fixture["binding"]["managed_derived_inventory_sha256"] = managed["derived_inventory_sha256"]
    _write_receipt(fixture)


def _rebind_output_inventory(fixture: dict[str, Any]) -> None:
    managed = fixture["receipt"]["managed"]
    inventory = _inventory(fixture["result_path"].parent)
    managed["output_inventory"] = inventory
    managed["output_inventory_sha256"] = stable_hash(inventory)
    fixture["binding"]["managed_output_inventory_sha256"] = managed["output_inventory_sha256"]
    _write_receipt(fixture)


def _write_confinement(work: Path, *, strategy: str | None = None) -> dict[str, Any]:
    strategy = audit_runner.write_confinement_strategy() if strategy is None else strategy
    evidence = {
        "schema_version": "spritelab.write-confinement-evidence.v3",
        "strategy": strategy,
        "platform": "linux" if strategy == audit_runner._LINUX_LANDLOCK_STRATEGY else "windows",
        "kernel_abi": 3 if strategy == audit_runner._LINUX_LANDLOCK_STRATEGY else 0,
        "root_identity_sha256": audit_runner.DirectoryIdentity.from_stat(work.stat()).identity_sha256,
        "handled_access_fs": 1 if strategy == audit_runner._LINUX_LANDLOCK_STRATEGY else 0,
        "allowed_access_fs": 1 if strategy == audit_runner._LINUX_LANDLOCK_STRATEGY else 0,
        "no_new_privileges": strategy == audit_runner._LINUX_LANDLOCK_STRATEGY,
        "restricted_token": False,
        "integrity_level_rid": 0,
        "mandatory_no_write_up": strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY,
        "workspace_integrity_level_rid": 0,
        "startup_integrity_level_rid": 4096 if strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY else 0,
        "bootstrap_lowered_before_worker_import": strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY,
        "new_thread_integrity_level_rid": 0,
        "raise_to_low_denied": strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY,
        "medium_probe_write_denied": strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY,
        "low_world_probe_write_denied": strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY,
        "untrusted_world_outside_guaranteed": False,
        "job_kill_on_close": strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY,
        "job_active_process_limit": 1 if strategy == audit_runner._WINDOWS_UNTRUSTED_STRATEGY else 0,
        "paths_exposed": False,
    }
    return evidence


def _fixture(tmp_path: Path) -> dict[str, Any]:
    project = tmp_path / "project"
    work_name = f"intake-{'1' * 32}"
    work_relative = f"datasets/conditioned_intake_work/{work_name}"
    work = project.joinpath(*PurePosixPath(work_relative).parts)
    source_root = work / "source"
    output_root = work / "datasets" / "managed"
    metadata_root = work / "datasets" / "source_metadata"
    derived_root = work / "derived_sprites"
    frames_root = derived_root / "frames"
    for root in (source_root, output_root, metadata_root, frames_root):
        root.mkdir(parents=True, exist_ok=True)

    source_id = "source.receipt-strictness"
    run_id = "harvest-receipt-strictness"
    license_document = {
        "identifier": "cc0-1.0",
        "evidence_url": "https://example.test/license",
        "attribution_text": "Fixture creator",
        "permissive_policy": True,
    }
    source_document = {
        "source_id": source_id,
        "title": "Receipt strictness fixture",
        "creator": "Fixture creator",
        "source_page": "https://example.test/source",
        "license": license_document,
    }

    parent_path = source_root / "weapons" / "iron_swords.png"
    parent_path.parent.mkdir(parents=True)
    parent_pixels = np.zeros((32, 64, 4), dtype=np.uint8)
    parent_pixels[:, :32] = (220, 48, 48, 255)
    parent_pixels[:, 32:] = (48, 96, 220, 255)
    Image.fromarray(parent_pixels, mode="RGBA").save(parent_path, format="PNG")
    parent_content = parent_path.read_bytes()
    parent_sha256 = hashlib.sha256(parent_content).hexdigest()
    parent_decoded_sha256 = audit_runner._decoded_rgba_identity(parent_pixels)
    parent_relative = "weapons/iron_swords.png"
    group_identity = stable_hash(
        {
            "schema_version": "spritelab.dataset.conditioned-derived-source-group.v1",
            "run_id": run_id,
            "source_id": source_id,
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_sha256,
        }
    )
    provenance_identity = stable_hash(
        {
            "schema_version": "spritelab.dataset.conditioned-derived-source-provenance.v1",
            "run_id": run_id,
            "source": source_document,
            "license": license_document,
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_sha256,
        }
    )
    records: list[dict[str, Any]] = []
    child_pixels: list[np.ndarray] = []
    for frame_index, left in enumerate((0, 32)):
        pixels = np.asarray(parent_pixels[:, left : left + 32], dtype=np.uint8).copy()
        child_pixels.append(pixels)
        decoded_sha256 = audit_runner._decoded_rgba_identity(pixels)
        derivation_payload = {
            "schema_version": "spritelab.dataset.conditioned-derived-sheet-derivation.v1",
            "dataset_item_id": f"item-frame-{frame_index}",
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_sha256,
            "parent_source_decoded_rgba_sha256": parent_decoded_sha256,
            "crop_rectangle": [left, 0, left + 32, 32],
            "frame_index": frame_index,
            "recipe_identity": audit_runner._DERIVED_SHEET_RECIPE_IDENTITY,
            "decoded_rgba_sha256": decoded_sha256,
            "source_provenance_identity": provenance_identity,
            "source_group_identity": group_identity,
        }
        derivation_identity = stable_hash(derivation_payload)
        frame_content = audit_runner._canonical_rgba_png(pixels)
        output_relative = f"frames/{derivation_identity}.png"
        record_payload = {
            "schema_version": "spritelab.dataset.conditioned-derived-sheet-frame.v1",
            "dataset_item_id": f"item-frame-{frame_index}",
            "parent_source_relative_path": parent_relative,
            "parent_source_raw_sha256": parent_sha256,
            "parent_source_decoded_rgba_sha256": parent_decoded_sha256,
            "crop_rectangle": [left, 0, left + 32, 32],
            "frame_index": frame_index,
            "recipe_version": audit_runner._DERIVED_SHEET_RECIPE["schema_version"],
            "recipe_identity": audit_runner._DERIVED_SHEET_RECIPE_IDENTITY,
            "decoded_rgba_sha256": decoded_sha256,
            "width": 32,
            "height": 32,
            "source_provenance_identity": provenance_identity,
            "source_group_identity": group_identity,
            "semantic_relative_path": f"{parent_relative}#frame{frame_index:04d}",
            "output_relative_path": output_relative,
            "encoded_output_sha256": hashlib.sha256(frame_content).hexdigest(),
            "encoded_output_byte_count": len(frame_content),
            "derivation_identity": derivation_identity,
            "source_derived_not_augmentation": True,
        }
        records.append({**record_payload, "record_identity": stable_hash(record_payload)})
        frames_root.joinpath(PurePosixPath(output_relative).name).write_bytes(frame_content)
    records.sort(key=lambda row: row["semantic_relative_path"])
    manifest_payload = {
        "schema_version": "spritelab.dataset.conditioned-derived-sheet-manifest.v1",
        "recipe": audit_runner._DERIVED_SHEET_RECIPE,
        "recipe_identity": audit_runner._DERIVED_SHEET_RECIPE_IDENTITY,
        "records": records,
        "record_count": len(records),
        "total_bytes": sum(int(row["encoded_output_byte_count"]) for row in records),
        "portable_relative_paths": True,
        "raw_source_mutated": False,
        "source_derived_not_augmentation": True,
        "paths_exposed": False,
    }
    manifest = {**manifest_payload, "manifest_identity": stable_hash(manifest_payload)}
    manifest_path = derived_root / "manifest.json"
    manifest_path.write_bytes(_canonical_document(manifest))

    output_path = output_root / "items.jsonl"
    output_path.write_bytes(b"{}\n")
    result = {
        "schema_version": "spritelab.product.result.v1",
        "status": "COMPLETE",
        "data": {"processed": 2},
    }
    result_path = output_root / "result.json"
    result_path.write_bytes(_canonical_document(result))
    sidecar_path = metadata_root / "source.sidecar.json"
    grouping_path = metadata_root / "source.grouping.json"
    sidecar_path.write_bytes(b"{}\n")
    grouping_path.write_bytes(b"{}\n")

    artifact_row = {
        "relative_path": parent_relative,
        "byte_count": len(parent_content),
        "expected_sha256": parent_sha256,
        "actual_sha256": parent_sha256,
        "mime_type": "image/png",
        "usable": True,
        "quarantine_reason": None,
        "taxonomy": ["weapon"],
    }
    artifact_identity_payload = [
        {
            "relative_path": parent_relative,
            "byte_count": len(parent_content),
            "sha256": parent_sha256,
            "mime_type": "image/png",
            "usable": True,
            "quarantine_reason": None,
            "taxonomy": ["weapon"],
        }
    ]
    artifact_manifest = {
        "schema_version": "spritelab.harvest.artifact-manifest.v1",
        "artifact_count": 1,
        "usable_count": 1,
        "quarantined_count": 0,
        "total_bytes": len(parent_content),
        "max_depth_observed": 2,
        "artifact_set_identity": stable_hash(artifact_identity_payload),
        "taxonomy_counts": {"weapon": 1},
        "files": [artifact_row],
        "paths_are_relative": True,
        "absolute_paths_exposed": False,
    }
    handoff = {
        "schema_version": "spritelab.harvest.dataset-handoff.v2",
        "run_id": run_id,
        "source": source_document,
        "license": license_document,
        "provenance_identity": _digest("handoff-provenance"),
        "source_evidence_binding_identity": _digest("source-evidence"),
    }
    worker_runtime = {"schema_version": "test.conditioned-worker-runtime.v1", "paths_exposed": False}
    code_payload = {
        "schema_version": "spritelab.dataset.conditioned-code-inventory.v3",
        "files": {
            "spritelab/product_features/conditioned_v5/intake.py": {
                "sha256": _digest("intake-code"),
                "byte_count": 42,
            }
        },
        "file_count": 1,
        "total_bytes": 42,
        "runtime_dependencies": {},
        "worker_runtime": worker_runtime,
    }
    code_inventory = {**code_payload, "inventory_sha256": stable_hash(code_payload)}
    source_inventory = _inventory(source_root)
    output_inventory = _inventory(output_root)
    derived_inventory = _inventory(derived_root)
    harvest = {
        "run_id": run_id,
        "handoff_identity": stable_hash(handoff),
        "request_handoff_identity": stable_hash(handoff),
        "artifact_manifest_identity": stable_hash(artifact_manifest),
        "artifact_manifest_file_sha256": _digest("artifact-manifest-file"),
        "artifact_set_identity": artifact_manifest["artifact_set_identity"],
        "provenance_identity": handoff["provenance_identity"],
        "source_evidence_binding_identity": handoff["source_evidence_binding_identity"],
        "trusted_catalog_identity": _digest("trusted-catalog"),
        "source_catalog_identity": _digest("source-catalog"),
        "backend_capability_identity": _digest("backend-capability"),
        "backend_capability_evidence_identity": _digest("backend-evidence"),
        "backend_certificate_identity": _digest("backend-certificate"),
        "backend_audit_report_sha256": _digest("backend-report-bytes"),
        "backend_audit_report_identity": _digest("backend-report"),
        "backend_capability_issued_at": "2026-07-17T00:00:00Z",
        "backend_capability_expires_at": "2026-07-24T00:00:00Z",
        "authorization_receipt_identity": _digest("authorization"),
        "acquisition_receipt_identity": _digest("acquisition"),
        "request_document_identity": _digest("request-document"),
    }
    managed = {
        "work_relative_path": work_relative,
        "source_relative_path": f"{work_relative}/source",
        "output_relative_path": f"{work_relative}/datasets/managed",
        "derived_root_relative_path": f"{work_relative}/derived_sprites",
        "source_inventory": source_inventory,
        "source_inventory_sha256": stable_hash(source_inventory),
        "output_inventory": output_inventory,
        "output_inventory_sha256": stable_hash(output_inventory),
        "derived_inventory": derived_inventory,
        "derived_inventory_sha256": stable_hash(derived_inventory),
        "derived_sheet_manifest": manifest,
        "derived_sheet_manifest_identity": manifest["manifest_identity"],
        "intake_result_identity": stable_hash(result),
        "accepted_relative_paths": [],
        "covered_source_relative_paths": [parent_relative],
        "write_confinement": _write_confinement(work),
        "worker_runtime": worker_runtime,
        "sidecar_relative_path": f"{work_relative}/datasets/source_metadata/{sidecar_path.name}",
        "sidecar_identity": {
            "sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
            "byte_count": sidecar_path.stat().st_size,
        },
        "sidecar_record_identity": stable_hash({}),
        "grouping_relative_path": f"{work_relative}/datasets/source_metadata/{grouping_path.name}",
        "grouping_identity": {
            "sha256": hashlib.sha256(grouping_path.read_bytes()).hexdigest(),
            "byte_count": grouping_path.stat().st_size,
        },
    }
    request_identity = _digest("managed-request")
    reference = f"dataset.{request_identity[:24]}"
    receipt_payload = {
        "schema_version": "spritelab.dataset.conditioned-import-receipt.v2",
        "dataset_reference": reference,
        "request_identity": request_identity,
        "callback_id": "dataset.conditioned-intake",
        "callback_code_identity_sha256": code_inventory["inventory_sha256"],
        "callback_code_inventory": code_inventory,
        "operation_control": {
            "schema_version": "spritelab.dataset.conditioned-operation-control.v1",
            "deadline_monotonic": 200.0,
            "started_monotonic": 100.0,
            "initial_budget_seconds": 100.0,
            "cancellation_probe_bound": True,
            "paths_exposed": False,
        },
        "harvest": harvest,
        "handoff_document": handoff,
        "artifact_manifest": artifact_manifest,
        "source": source_document,
        "license": license_document,
        "managed": managed,
        "accepted_count": 2,
        "quarantined_count": 0,
        "raw_harvest_mutated": False,
        "atomic_publication": "receipt_pointer_after_validation",
        "portable_relative_paths": True,
        "paths_exposed": False,
        "created_at": "2026-07-17T00:00:00Z",
    }
    receipt = {**receipt_payload, "receipt_identity": stable_hash(receipt_payload)}
    binding = {
        "dataset_reference": reference,
        "harvest_run_id": run_id,
        "handoff_identity": harvest["handoff_identity"],
        "harvest_import_receipt_identity": _digest("harvest-import-receipt"),
        "managed_intake_receipt_identity": receipt["receipt_identity"],
        "managed_source_inventory_sha256": managed["source_inventory_sha256"],
        "managed_output_inventory_sha256": managed["output_inventory_sha256"],
        "managed_derived_inventory_sha256": managed["derived_inventory_sha256"],
        "derived_sheet_manifest_identity": manifest["manifest_identity"],
        "trusted_catalog_identity": harvest["trusted_catalog_identity"],
        "source_catalog_identity": harvest["source_catalog_identity"],
        "backend_capability_identity": harvest["backend_capability_identity"],
        "backend_capability_evidence_identity": harvest["backend_capability_evidence_identity"],
        "backend_certificate_identity": harvest["backend_certificate_identity"],
        "backend_audit_report_sha256": harvest["backend_audit_report_sha256"],
        "backend_audit_report_identity": harvest["backend_audit_report_identity"],
        "backend_capability_issued_at": harvest["backend_capability_issued_at"],
        "backend_capability_expires_at": harvest["backend_capability_expires_at"],
        "authorization_receipt_identity": harvest["authorization_receipt_identity"],
        "acquisition_receipt_identity": harvest["acquisition_receipt_identity"],
        "artifact_manifest_sha256": harvest["artifact_manifest_file_sha256"],
        "artifact_set_identity": harvest["artifact_set_identity"],
        "source_id": source_id,
        "title": source_document["title"],
        "creator": source_document["creator"],
        "license_id": license_document["identifier"],
        "license_evidence": license_document,
        "source_document": source_document,
        "license_document": license_document,
    }
    receipts_root = project / "datasets" / "conditioned_intake_receipts"
    receipts_root.mkdir(parents=True)
    receipt_path = receipts_root / f"{reference}.json"
    receipt_path.write_bytes(_canonical_document(receipt))
    job_root = project / "runs" / "v3" / "conditioned-dataset-v5" / "conditioned-test"
    job_root.mkdir(parents=True)
    candidate = {"input_bindings": [binding]}
    dataset = {
        "sprites": {
            "sprite-1": {
                "record": {
                    "source_id": source_id,
                    "source_pack": source_document["title"],
                    "creator": source_document["creator"],
                    "license_id": license_document["identifier"],
                    "source_relative_path": records[0]["semantic_relative_path"],
                    "source_sha256": records[0]["encoded_output_sha256"],
                    "source_byte_count": records[0]["encoded_output_byte_count"],
                    "source_group": records[0]["source_group_identity"],
                    "source_derivation": records[0],
                },
                "rgba": child_pixels[0],
            }
        }
    }
    return {
        "project": project,
        "work": work,
        "job_root": job_root,
        "candidate": candidate,
        "dataset": dataset,
        "binding": binding,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "manifest_path": manifest_path,
        "result_path": result_path,
        "sidecar_path": sidecar_path,
        "grouping_path": grouping_path,
        "second_frame_path": frames_root / PurePosixPath(records[1]["output_relative_path"]).name,
    }


def _audit(fixture: dict[str, Any]) -> None:
    audit_runner._verify_parent_bound_derivations(
        fixture["project"],
        fixture["job_root"],
        fixture["candidate"],
        fixture["dataset"],
        progress=lambda *_args: None,
        cancelled=lambda: False,
    )


def test_strict_receipt_accepts_valid_contract_and_rejects_non_candidate_frame_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _audit(fixture)
    fixture["second_frame_path"].write_bytes(fixture["second_frame_path"].read_bytes() + b"drift")
    with pytest.raises(audit_runner.IndependentAuditError):
        _audit(fixture)


def test_request_handoff_identity_is_recomputed_from_exact_document(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["receipt"]["harvest"]["request_handoff_identity"] = _digest("forged-request-handoff")
    _write_receipt(fixture)
    with pytest.raises(audit_runner.IndependentAuditError) as captured:
        _audit(fixture)
    assert captured.value.code == "audit_receipt_contract"


def test_persisted_result_json_is_bound_after_inventory_rehash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    forged_result = {
        "schema_version": "spritelab.product.result.v1",
        "status": "COMPLETE",
        "data": {"processed": 999},
    }
    fixture["result_path"].write_bytes(_canonical_document(forged_result))
    _rebind_output_inventory(fixture)
    with pytest.raises(audit_runner.IndependentAuditError, match="intake result identity"):
        _audit(fixture)


@pytest.mark.parametrize("forgery", ("root", "strategy"))
def test_write_confinement_is_bound_to_reopened_work_root(tmp_path: Path, forgery: str) -> None:
    fixture = _fixture(tmp_path)
    confinement = fixture["receipt"]["managed"]["write_confinement"]
    if forgery == "root":
        confinement["root_identity_sha256"] = _digest("different-work-root")
    else:
        current = audit_runner.write_confinement_strategy()
        other = (
            audit_runner._WINDOWS_UNTRUSTED_STRATEGY
            if current == audit_runner._LINUX_LANDLOCK_STRATEGY
            else audit_runner._LINUX_LANDLOCK_STRATEGY
        )
        fixture["receipt"]["managed"]["write_confinement"] = _write_confinement(fixture["work"], strategy=other)
    _write_receipt(fixture)
    with pytest.raises(audit_runner.IndependentAuditError, match="exact work root"):
        _audit(fixture)


def test_sidecar_record_identity_is_bound_after_file_identity_rehash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    content = _canonical_document({"forged": True})
    fixture["sidecar_path"].write_bytes(content)
    fixture["receipt"]["managed"]["sidecar_identity"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    _write_receipt(fixture)
    with pytest.raises(audit_runner.IndependentAuditError, match="sidecar differs from its record identity"):
        _audit(fixture)


def test_grouping_file_bytes_are_rehashed_from_exact_metadata_root(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["grouping_path"].write_bytes(b"[]\n")
    with pytest.raises(audit_runner.IndependentAuditError, match="metadata file"):
        _audit(fixture)


@pytest.mark.parametrize("scope", ("top", "managed", "harvest"))
def test_rehashed_underspecified_receipt_is_rejected(tmp_path: Path, scope: str) -> None:
    fixture = _fixture(tmp_path)
    receipt = fixture["receipt"]
    if scope == "top":
        receipt.pop("operation_control")
    elif scope == "managed":
        receipt["managed"].pop("output_inventory")
    else:
        receipt["harvest"].pop("request_document_identity")
    _write_receipt(fixture)
    with pytest.raises(audit_runner.IndependentAuditError) as captured:
        _audit(fixture)
    assert captured.value.code == "audit_receipt_contract"
    assert str(fixture["project"]) not in captured.value.public_message


def test_on_disk_manifest_bytes_are_verified_exactly(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    content = fixture["manifest_path"].read_bytes()
    changed = content.replace(b"false", b"falsE", 1)
    assert changed != content and len(changed) == len(content)
    fixture["manifest_path"].write_bytes(changed)
    with pytest.raises(audit_runner.IndependentAuditError, match="on-disk derived manifest"):
        _audit(fixture)


def test_every_non_candidate_manifest_row_is_validated(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    second = fixture["receipt"]["managed"]["derived_sheet_manifest"]["records"][1]
    second["width"] = 31
    record_payload = dict(second)
    record_payload.pop("record_identity")
    second["record_identity"] = stable_hash(record_payload)
    _rebind_manifest(fixture)
    with pytest.raises(audit_runner.IndependentAuditError) as captured:
        _audit(fixture)
    assert captured.value.code == "audit_receipt_contract"


def test_derived_inventory_must_cover_manifest_and_all_frames_exactly(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    managed = fixture["receipt"]["managed"]
    second_output = managed["derived_sheet_manifest"]["records"][1]["output_relative_path"]
    managed["derived_inventory"]["files"].pop(second_output)
    managed["derived_inventory"]["file_count"] -= 1
    managed["derived_inventory"]["total_bytes"] = sum(
        int(item["byte_count"]) for item in managed["derived_inventory"]["files"].values()
    )
    managed["derived_inventory_sha256"] = stable_hash(managed["derived_inventory"])
    fixture["binding"]["managed_derived_inventory_sha256"] = managed["derived_inventory_sha256"]
    _write_receipt(fixture)
    with pytest.raises(audit_runner.IndependentAuditError, match="exact manifest and frames"):
        _audit(fixture)
