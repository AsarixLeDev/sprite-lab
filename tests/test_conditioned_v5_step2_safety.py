from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from PIL import Image

import spritelab.product_features.conditioned_v5.intake as intake_module
from spritelab.product_features.conditioned_v5 import ConditionedDatasetImportAdapter
from spritelab.product_features.harvest.storage import scan_artifacts
from spritelab.product_features.harvest.trusted_backend import DatasetImportRequest, HarvestLimits
from spritelab.training.campaign import stable_hash
from spritelab.utils.safe_fs import OwnedFileIdentity


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sprite_png() -> bytes:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(32):
        for x in range(32):
            if abs(x - 16) + abs(y - 16) <= 11:
                shade = ((x // 3 + y // 3) % 3) * 24
                image.putpixel((x, y), (180 + shade, 64 + shade, 32, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _synthetic_code_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    content = Path(intake_module.__file__).read_bytes()
    worker_runtime = {
        "schema_version": "test.conditioned-worker-runtime.v1",
        "paths_exposed": False,
    }
    payload = {
        "schema_version": "spritelab.dataset.conditioned-code-inventory.v3",
        "files": {
            "spritelab/product_features/conditioned_v5/intake.py": {
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        },
        "file_count": 1,
        "total_bytes": len(content),
        "runtime_dependencies": {},
        "worker_runtime": worker_runtime,
    }
    return {**payload, "inventory_sha256": stable_hash(payload)}, worker_runtime


def _synthetic_adapter(
    project: Path,
    code_inventory: dict[str, Any],
) -> ConditionedDatasetImportAdapter:
    adapter = object.__new__(ConditionedDatasetImportAdapter)
    adapter.project_root = project
    adapter.code_inventory = code_inventory
    adapter.code_identity_sha256 = code_inventory["inventory_sha256"]
    adapter.runtime_inventory = {}
    adapter.runtime_identity_sha256 = "0" * 64
    adapter._catalog_loader = lambda _root: ()
    adapter._capability_evidence_loader = lambda _root: None
    return adapter


def _write_confinement_evidence(
    strategy: str,
    workspace_identity: Any,
) -> dict[str, Any]:
    windows = strategy == intake_module.WINDOWS_PARENT_ANCHORS_STRATEGY
    return {
        "schema_version": "spritelab.write-confinement-evidence.v3",
        "strategy": strategy,
        "platform": "windows" if windows else "linux",
        "kernel_abi": 0 if windows else 3,
        "root_identity_sha256": workspace_identity.identity_sha256,
        "handled_access_fs": 0 if windows else 1,
        "allowed_access_fs": 0 if windows else 1,
        "no_new_privileges": not windows,
        "restricted_token": False,
        "integrity_level_rid": 0,
        "mandatory_no_write_up": windows,
        "workspace_integrity_level_rid": 0,
        "startup_integrity_level_rid": 4096 if windows else 0,
        "bootstrap_lowered_before_worker_import": windows,
        "new_thread_integrity_level_rid": 0,
        "raise_to_low_denied": windows,
        "medium_probe_write_denied": windows,
        "low_world_probe_write_denied": windows,
        "untrusted_world_outside_guaranteed": False,
        "job_kill_on_close": windows,
        "job_active_process_limit": 1 if windows else 0,
        "paths_exposed": False,
    }


def _synthetic_harvest_fixture(
    project: Path,
) -> tuple[DatasetImportRequest, dict[str, Any], Path, bytes]:
    run_id = "harvest-step2-safety"
    artifacts = project / "harvest_runs" / run_id / "artifacts"
    artifacts.mkdir(parents=True)
    raw_path = artifacts / "hero.png"
    raw_bytes = _sprite_png()
    raw_path.write_bytes(raw_bytes)
    manifest = scan_artifacts(
        artifacts,
        HarvestLimits(max_files=4, max_total_bytes=1024 * 1024),
    )
    license_document = {
        "identifier": "cc0-1.0",
        "evidence_url": "https://license.example.test/cc0",
        "attribution_text": "Synthetic fixture creator",
        "permissive_policy": True,
    }
    source_document = {
        "source_id": "source.step2-safety",
        "title": "Step-2 safety fixture",
        "creator": "Synthetic fixture creator",
        "source_page": "https://source.example.test/step2-safety",
        "license": license_document,
    }
    handoff = {
        "run_id": run_id,
        "source": source_document,
        "license": license_document,
        "provenance_identity": _digest("provenance"),
        "source_evidence_binding_identity": _digest("source-evidence"),
    }
    verification = {
        "artifacts_root": artifacts,
        "handoff": handoff,
        "handoff_identity": stable_hash(handoff),
        "request_handoff_identity": _digest("request-handoff"),
        "artifact_manifest": manifest,
        "artifact_manifest_identity": stable_hash(manifest),
        "artifact_manifest_file_sha256": _digest("artifact-manifest-file"),
        "trusted_catalog_identity": _digest("trusted-catalog"),
        "source_catalog_identity": _digest("source-catalog"),
        "backend_capability_identity": _digest("backend-capability"),
        "backend_capability_evidence_identity": _digest("backend-evidence"),
        "backend_certificate_identity": _digest("backend-certificate"),
        "backend_audit_report_sha256": _digest("backend-audit-bytes"),
        "backend_audit_report_identity": _digest("backend-audit-document"),
        "backend_capability_issued_at": "2026-07-17T00:00:00Z",
        "backend_capability_expires_at": "2026-07-24T00:00:00Z",
        "authorization_receipt_identity": _digest("authorization"),
        "acquisition_receipt_identity": _digest("acquisition"),
        "request_document_identity": _digest("request"),
        "source": source_document,
        "license": license_document,
    }
    return DatasetImportRequest(run_id, artifacts, handoff, manifest), verification, raw_path, raw_bytes


def _install_synthetic_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    code_inventory: dict[str, Any],
    worker_runtime: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    strategy = (
        intake_module.WINDOWS_PARENT_ANCHORS_STRATEGY if os.name == "nt" else intake_module.LINUX_LANDLOCK_STRATEGY
    )
    monkeypatch.setattr(intake_module, "write_confinement_strategy", lambda: strategy)
    monkeypatch.setattr(intake_module, "conditioned_code_inventory", lambda: code_inventory)
    monkeypatch.setattr(intake_module, "controlled_worker_runtime", lambda: worker_runtime)
    monkeypatch.setattr(intake_module, "_verify_harvest_request", lambda *_args, **_kwargs: verification)

    def run_in_process(
        work: Path,
        *,
        strategy: str,
        workspace_identity: Any,
        request_payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        result = intake_module._run_legacy_intake_in_process(
            work=work,
            source_root=work / "source",
            output_root=work / "datasets" / "managed",
            source=request_payload["source"],
            license_record=request_payload["license"],
            artifact_sha256=request_payload["artifact_sha256"],
            run_id=request_payload["run_id"],
        )
        return {
            "schema_version": intake_module._LEGACY_RESPONSE_SCHEMA,
            "ok": True,
            "result": result,
            "write_confinement": _write_confinement_evidence(strategy, workspace_identity),
            "paths_exposed": False,
        }

    monkeypatch.setattr(intake_module, "_run_legacy_intake_child", run_in_process)


def test_created_workspace_identity_substitution_fails_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"preserve")
    code_inventory, worker_runtime = _synthetic_code_inventory()
    adapter = _synthetic_adapter(project, code_inventory)
    verification = {
        "handoff_identity": _digest("handoff"),
        "artifact_manifest_identity": _digest("manifest"),
    }
    strategy = (
        intake_module.WINDOWS_PARENT_ANCHORS_STRATEGY if os.name == "nt" else intake_module.LINUX_LANDLOCK_STRATEGY
    )
    monkeypatch.setattr(intake_module, "_verify_harvest_request", lambda *_args, **_kwargs: verification)
    monkeypatch.setattr(intake_module, "write_confinement_strategy", lambda: strategy)
    monkeypatch.setattr(intake_module, "conditioned_code_inventory", lambda: code_inventory)
    monkeypatch.setattr(intake_module, "controlled_worker_runtime", lambda: worker_runtime)

    original_mkdir_unique = intake_module.AnchoredDirectory.mkdir_unique

    def substitute_created_identity(
        anchor: intake_module.AnchoredDirectory,
        prefix: str,
    ) -> tuple[str, OwnedFileIdentity]:
        name, identity = original_mkdir_unique(anchor, prefix)
        if prefix != "intake-":
            return name, identity
        return name, OwnedFileIdentity(identity.device, identity.inode + 1, identity.file_type)

    monkeypatch.setattr(intake_module.AnchoredDirectory, "mkdir_unique", substitute_created_identity)
    worker_calls: list[Path] = []

    def unexpected_worker(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        worker_calls.append(project)
        raise AssertionError("worker ran after creation identity substitution")

    monkeypatch.setattr(intake_module, "_run_legacy_intake_boundary", unexpected_worker)
    request = DatasetImportRequest("harvest-identity-substitution", project / "unused", {}, {})
    with pytest.raises(intake_module.ConditionedIntakeError, match="workspace changed"):
        adapter.import_harvest(request, idempotency_key="step2-created-identity")

    assert worker_calls == []
    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.parametrize(
    ("tamper_revalidation_call", "receipt_is_committed"),
    ((1, False), (2, True)),
    ids=("pre-commit", "post-receipt"),
)
def test_full_transaction_revalidation_rejects_managed_tamper_without_touching_harvest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_revalidation_call: int,
    receipt_is_committed: bool,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    request, verification, raw_path, raw_bytes = _synthetic_harvest_fixture(project)
    code_inventory, worker_runtime = _synthetic_code_inventory()
    adapter = _synthetic_adapter(project, code_inventory)
    _install_synthetic_runtime(
        monkeypatch,
        code_inventory=code_inventory,
        worker_runtime=worker_runtime,
        verification=verification,
    )

    original_revalidate = intake_module._revalidate_managed_transaction
    calls = 0
    observed_receipt_state: bool | None = None

    def tamper_at_selected_boundary(
        project_root: Path,
        receipt: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal calls, observed_receipt_state
        calls += 1
        if calls == tamper_revalidation_call:
            receipt_path = project / "datasets" / "conditioned_intake_receipts" / f"{receipt['dataset_reference']}.json"
            observed_receipt_state = receipt_path.is_file()
            relative = PurePosixPath(receipt["managed"]["source_relative_path"])
            managed_source = project.joinpath(*relative.parts) / "hero.png"
            managed_source.write_bytes(managed_source.read_bytes() + b"managed-copy-tamper")
        try:
            return original_revalidate(project_root, receipt, **kwargs)
        except intake_module.ConditionedIntakeError as exc:
            if calls < tamper_revalidation_call:
                raise AssertionError(
                    f"untampered transaction failed before injection: {exc!r}; cause={exc.__cause__!r}"
                ) from exc
            raise

    monkeypatch.setattr(intake_module, "_revalidate_managed_transaction", tamper_at_selected_boundary)
    with pytest.raises(intake_module.ConditionedIntakeError) as caught:
        adapter.import_harvest(request, idempotency_key=f"step2-tamper-{tamper_revalidation_call}")

    receipts = list((project / "datasets" / "conditioned_intake_receipts").glob("dataset.*.json"))
    assert calls == tamper_revalidation_call
    assert observed_receipt_state is receipt_is_committed
    assert len(receipts) == int(receipt_is_committed)
    assert str(caught.value) == "Managed intake source bytes changed after publication."
    assert os.fspath(project) not in str(caught.value)
    assert raw_path.read_bytes() == raw_bytes
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == hashlib.sha256(raw_bytes).hexdigest()
