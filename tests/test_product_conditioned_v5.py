from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from spritelab.product_core import ProjectContext
from spritelab.product_features.conditioned_v5 import (
    CandidatePolicy,
    ConditionedDatasetImportAdapter,
    ConditionedDatasetService,
)
from spritelab.product_features.conditioned_v5.intake import load_managed_intake
from spritelab.product_features.conditioned_v5.plugin import create_plugin
from spritelab.product_features.conditioned_v5.service import (
    DATASET_VALIDATION_GATES,
    DATASET_VALIDATION_SCHEMA,
    HANDOFF_SCHEMA,
    LABEL_AUDIT_GATES,
    LABEL_AUDIT_SCHEMA,
)
from spritelab.product_features.conditioned_v5.web import create_router
from spritelab.product_features.harvest.storage import scan_artifacts
from spritelab.product_features.harvest.trusted_backend import (
    DatasetImportRequest,
    HarvestLimits,
)
from spritelab.product_web.app import create_app
from spritelab.training.campaign import stable_hash


def _png(path: Path, color: tuple[int, int, int, int], marker: int) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(32):
        for x in range(32):
            if abs(x - 16) + abs(y - 16) <= 11:
                shade = ((x // 3 + y // 3 + marker) % 3) * 36
                image.putpixel(
                    (x, y),
                    (
                        (color[0] + shade) % 255,
                        (color[1] + shade // 2) % 255,
                        (color[2] + shade // 3) % 255,
                        255,
                    ),
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _handoff(project: Path, run_id: str, source_id: str, offset: int) -> dict[str, Any]:
    run = project / "harvest_runs" / run_id
    artifacts = run / "artifacts"
    names = (
        "weapons/iron_sword.png",
        "tools/copper_pickaxe.png",
        "armor/steel_helmet.png",
        "potions/blue_elixir.png",
    )
    for index, name in enumerate(names, start=offset):
        _png(artifacts / name, ((index * 29) % 255, (index * 53) % 255, (index * 71) % 255, 255), index)
    manifest = scan_artifacts(artifacts, HarvestLimits(max_files=32, max_total_bytes=8 * 1024 * 1024))
    manifest_path = run / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    license_value = {
        "identifier": "cc0-1.0",
        "evidence_url": "https://example.test/license",
        "evidence_url_sha256": "1" * 64,
        "evidence_text": "CC0 public-domain dedication.",
        "attribution_text": f"Creator {source_id}",
        "permissive_policy": True,
    }
    source = {
        "schema_version": "spritelab.harvest.source.v2",
        "source_id": source_id,
        "title": f"Source {source_id}",
        "creator": f"Creator {source_id}",
        "source_page": f"https://example.test/{source_id}",
        "license": license_value,
        "evidence_binding": {"binding_identity": "2" * 64},
    }
    acquisition_identity = "3" * 64
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "run_id": run_id,
        "source_id": source_id,
        "managed_reference": {"kind": "harvest_run", "run_id": run_id},
        "source": source,
        "provenance_identity": stable_hash({"source": source, "acquisition_receipt_identity": acquisition_identity}),
        "source_evidence_binding_identity": "2" * 64,
        "backend_capability_identity": "4" * 64,
        "limits_identity": "5" * 64,
        "acquisition_receipt_identity": acquisition_identity,
        "artifact_manifest_identity": stable_hash(manifest),
        "artifact_set_identity": manifest["artifact_set_identity"],
        "artifact_count": manifest["artifact_count"],
        "usable_count": manifest["usable_count"],
        "quarantined_count": manifest["quarantined_count"],
        "total_bytes": manifest["total_bytes"],
        "taxonomy_counts": manifest["taxonomy_counts"],
        "files": manifest["files"],
        "license": license_value,
        "handoff_ready": True,
        "portable_relative_paths": True,
        "paths_exposed": False,
    }
    (run / "handoff.json").write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return handoff


def _service(project: Path) -> ConditionedDatasetService:
    def campaign_builder(*_args: Any, **_kwargs: Any) -> Any:
        portable = {
            "campaign_id": "conditioned_test",
            "seeds": [731001, 731002, 731003],
            "training": {"max_optimizer_steps": 5000},
            "identities": {"dataset_freeze_hash": _kwargs["activation_manifest_sha256"]},
            "executable": True,
            "launch_authorized": True,
        }
        return SimpleNamespace(portable_campaign=portable, validation={"launch_ready": True})

    return ConditionedDatasetService(
        project,
        campaign_builder=campaign_builder,
        policy=CandidatePolicy(min_images=4, target_images=8, max_images=10, max_source_files=32),
    )


def _import_handoff(project: Path, run_id: str, handoff: dict[str, Any]) -> str:
    run = project / "harvest_runs" / run_id
    manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
    work_root = project / "datasets" / "conditioned_intake_work"
    prior_work_count = len(list(work_root.iterdir())) if work_root.is_dir() else 0
    before = {
        path.relative_to(run / "artifacts").as_posix(): path.read_bytes()
        for path in sorted((run / "artifacts").rglob("*"))
        if path.is_file()
    }
    adapter = ConditionedDatasetImportAdapter()
    result = adapter.import_harvest(
        DatasetImportRequest(run_id, run / "artifacts", handoff, manifest),
        idempotency_key=f"dataset-import-{run_id}",
    )
    repeated = adapter.import_harvest(
        DatasetImportRequest(run_id, run / "artifacts", handoff, manifest),
        idempotency_key=f"dataset-import-{run_id}",
    )
    assert repeated == result
    assert len(list(work_root.iterdir())) == prior_work_count + 1
    receipt = {
        "schema_version": "spritelab.harvest.dataset-import-receipt.v1",
        "run_id": run_id,
        "idempotency_key": f"dataset-import-{run_id}",
        "callback_id": adapter.callback_id,
        "callback_code_identity_sha256": adapter.code_identity_sha256,
        "dataset_reference": result.dataset_reference,
        "accepted_count": result.accepted_count,
        "quarantined_count": result.quarantined_count,
        "artifact_manifest_identity": stable_hash(manifest),
        "paths_exposed": False,
        "created_at": "2026-07-17T00:00:00Z",
    }
    (run / "dataset_import_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    after = {
        path.relative_to(run / "artifacts").as_posix(): path.read_bytes()
        for path in sorted((run / "artifacts").rglob("*"))
        if path.is_file()
    }
    assert after == before
    loaded = load_managed_intake(project, result.dataset_reference)
    assert loaded["dataset_reference"] == result.dataset_reference
    assert loaded["accepted_relative_paths"]
    return result.dataset_reference


def _wait(service: ConditionedDatasetService, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = service.job(job_id)
        if job["status"] != "RUNNING":
            return job
        time.sleep(0.02)
    raise AssertionError("conditioned build did not finish")


def _evidence(kind: str, candidate: dict[str, Any]) -> dict[str, Any]:
    label = kind == "label_audit"
    return {
        "schema_version": LABEL_AUDIT_SCHEMA if label else DATASET_VALIDATION_SCHEMA,
        "verdict": "PASS",
        "independent": True,
        "generated_by_conditioned_workflow": False,
        "auditor": {"auditor_id": f"independent.{kind}", "code_identity_sha256": "a" * 64},
        "audit_run_identity": ("b" if label else "c") * 64,
        "bindings": {
            "candidate_identity": candidate["candidate_identity"],
            "payload_inventory_sha256": candidate["payload_inventory_sha256"],
            "image_count": candidate["image_count"],
        },
        "subject_files": candidate["payload_inventory"],
        "checks": dict.fromkeys(LABEL_AUDIT_GATES if label else DATASET_VALIDATION_GATES, "PASS"),
    }


def test_preview_build_evidence_and_publication_are_bound_and_portable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    handoffs = {
        "harvest-source-one": _handoff(project, "harvest-source-one", "source.one", 1),
        "harvest-source-two": _handoff(project, "harvest-source-two", "source.two", 20),
    }
    references = [_import_handoff(project, run_id, handoff) for run_id, handoff in handoffs.items()]
    service = _service(project)

    preview = service.preview(references)
    assert preview["ready_to_build"] is True
    assert preview["selected_images"] == 8
    assert preview["source_counts"] == {"source.one": 4, "source.two": 4}
    assert preview["labels_are_human_truth"] is False

    started, created = service.start_build(
        references, idempotency_key="conditioned-build-test-0001", explicit_action=True
    )
    assert created is True
    job = _wait(service, started["job_id"])
    assert job["status"] == "NEEDS_REVIEW", job["message"]
    assert job["candidate"]["image_count"] == 8
    root = project / "runs" / "v3" / "conditioned-dataset-v5" / started["job_id"]
    candidate = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert candidate["dataset_references"] == sorted(references)
    assert len(candidate["managed_intake_receipt_identities"]) == 2
    phase7 = root / "candidate" / "phase7"
    assert all((phase7 / f"{split}.npz").is_file() for split in ("train", "val", "test"))
    assert str(project) not in (phase7 / "training_manifest.jsonl").read_text(encoding="utf-8")
    assert json.loads((phase7 / "split_integrity_report.json").read_text(encoding="utf-8"))["ok"] is True

    job = service.attach_evidence(started["job_id"], kind="label_audit", document=_evidence("label_audit", candidate))
    job = service.attach_evidence(
        started["job_id"], kind="dataset_validation", document=_evidence("dataset_validation", candidate)
    )
    published = service.publish(
        started["job_id"],
        candidate_identity=candidate["candidate_identity"],
        label_audit_sha256=job["evidence"]["label_audit"]["sha256"],
        dataset_validation_sha256=job["evidence"]["dataset_validation"]["sha256"],
        authorization_id="freeze-authorization-test-0001",
        explicit_action=True,
        authorize_one_time_freeze=True,
    )
    assert published["status"] == "COMPLETE"
    publication = published["publication"]
    activation = project / publication["activation_manifest"]
    activation_value = json.loads(activation.read_text(encoding="utf-8"))
    assert activation_value["schema_version"] == "spritelab.dataset.freeze.conditioned.v5"
    assert activation_value["publication_inventory"]["file_count"] > 10
    assert set(next(iter(activation_value["publication_inventory"]["files"].values()))) == {
        "sha256",
        "byte_count",
    }
    assert publication["campaign_seeds"] == [731001, 731002, 731003]
    assert publication["campaign_steps"] == 5000
    assert publication["configuration_activated"] is False
    assert not (project / "spritelab.yaml").exists()


def test_plugin_and_api_reject_browser_paths_and_expose_navigation(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    project = tmp_path / "project"
    project.mkdir()
    handoff = _handoff(project, "harvest-source-one", "source.one", 1)
    reference = _import_handoff(project, "harvest-source-one", handoff)
    service = _service(project)
    plugin = create_plugin(service_factory=lambda _context: service)
    assert plugin.navigation[0].path == "/dataset-v5"
    assert plugin.api_prefixes == ("/dataset-v5/api",)

    runs = project / "runs-shell"
    runs.mkdir()
    shell = TestClient(create_app(ProjectContext(project, runs_directory=runs), plugins=(plugin,)))
    page = shell.get("/dataset-v5")
    assert page.status_code == 200
    assert 'meta name="spritelab-csrf"' in page.text
    no_csrf = shell.post("/dataset-v5/api/preview", json={"dataset_references": [reference]})
    assert no_csrf.status_code == 403
    assert no_csrf.json()["error_code"] == "csrf_validation_failed"

    app = FastAPI()
    app.include_router(create_router(ProjectContext(project), service=service))
    client = TestClient(app)
    assert client.get("/dataset-v5").status_code == 200
    rejected = client.post(
        "/dataset-v5/api/jobs",
        json={
            "dataset_references": [reference],
            "output_path": "C:/private",
            "idempotency_key": "conditioned-build-test-0002",
            "explicit_action": True,
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["error_code"] == "invalid_conditioned_v5_payload"
