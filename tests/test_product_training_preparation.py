from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from spritelab.product_core import ProjectContext
from spritelab.product_features.training.models import TrainingProfile
from spritelab.product_features.training.plans import TrainingPlanResolver
from spritelab.product_features.training.preparation import (
    TrainingPreparationError,
    prepare_active_dataset,
)
from spritelab.product_features.training.web import create_router
from spritelab.remote_compute import FakeComputeBackend
from spritelab.training.campaign import validate_campaign
from spritelab.training.cli.experiment_cmds import _prepare_manifest
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig


def _project(tmp_path: Path) -> tuple[ProjectContext, Path]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "spritelab.yaml").write_text(
        yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
        encoding="utf-8",
    )
    source = project / "accepted.png"
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(8, 24):
        for x in range(10, 22):
            image.putpixel((x, y), (220, 80, 40, 255))
    image.save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    active = project / "datasets" / "active"
    active.mkdir(parents=True)
    (active / "result.json").write_text("{}\n", encoding="utf-8")
    (active / "review_queue.json").write_text('{"items": []}\n', encoding="utf-8")
    item = {
        "item_id": "sword-1",
        "relative_path": "accepted.png",
        "source_path": str(source),
        "byte_sha256": digest,
        "current_disposition": "accepted",
        "semantic": {
            "object_name": "sword",
            "short_description": "orange pixel sword",
            "debug_path": str(source),
        },
    }
    (active / "items.jsonl").write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")
    loaded = ProjectConfig.load(project)
    runs = loaded.runs_dir
    runs.mkdir(parents=True, exist_ok=True)
    return ProjectContext(project, loaded.values, loaded.path, runs), source


def _publication(project: Path) -> Path:
    publications = list((project / ".spritelab" / "training-preparation").glob("dataset-*"))
    assert len(publications) == 1
    return publications[0]


def test_preparation_builds_qa_valid_campaign_and_experiment_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _source = _project(tmp_path)

    result = prepare_active_dataset(
        context,
        authorize_freeze=True,
        authorize_training=True,
    )

    assert result["image_count"] == 1
    assert result["training_authorized"] is True
    assert result["remaining_gate"] == "independent_training_infrastructure_audit"
    assert result["paths_exposed"] is False
    output = _publication(context.project_root)
    assert not json.loads((output / "dataset_qa_report.json").read_text(encoding="utf-8"))["errors"]
    assert not json.loads((output / "training_manifest_qa_report.json").read_text(encoding="utf-8"))["errors"]
    assert (output / "training_manifest.jsonl").is_file()
    assert (output / "conditioning_vocabulary.json").is_file()
    assert (output / "publication_manifest.json").is_file()
    training_rows = [
        json.loads(line) for line in (output / "training_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["object_name"] for row in training_rows} == {"sprite"}
    assert all("sword" not in row["caption"] and "orange" not in row["caption"] for row in training_rows)
    text_artifacts = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.suffix in {".json", ".jsonl", ".md"}
    )
    assert str(context.project_root) not in text_artifacts
    assert "accepted.png" not in text_artifacts

    persisted = (output / "campaign.json").read_text(encoding="utf-8")
    assert str(context.project_root) not in persisted
    config = ProjectConfig.load(context.project_root)
    assert config.values["execution"]["allow_dataset_production_freeze"] is True
    assert config.values["execution"]["allow_training"] is True
    assert not Path(config.values["training"]["campaign_config"]).is_absolute()

    fresh_context = ProjectContext(config.root, config.values, config.path, config.runs_dir)
    resolved = TrainingPlanResolver().resolve(
        fresh_context,
        TrainingProfile.RECOMMENDED,
        FakeComputeBackend(),
        probe_backend=False,
    )
    assert resolved.campaign is not None
    validation = validate_campaign(resolved.campaign)
    assert validation["errors"] == []
    assert validation["blockers"] == []
    run = resolved.campaign["expected_runs"][0]
    assert run["resolved_config"]["runtime"]["device"] == "auto"
    resolved_path = Path(run["resolved_config_path"])
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(run["resolved_config"], sort_keys=True), encoding="utf-8")
    monkeypatch.chdir(context.project_root)
    manifest = _prepare_manifest(resolved_path, write=False)
    assert manifest["name"].startswith("recommended_")
    assert manifest["dataset_manifest_hash"] == resolved.campaign["identities"]["split_manifest_hash"]


def test_preparation_reverifies_source_hash_before_reuse(tmp_path: Path) -> None:
    context, source = _project(tmp_path)
    prepare_active_dataset(context, authorize_freeze=True, authorize_training=False)
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_freeze=True, authorize_training=False)

    assert caught.value.code == "accepted_source_changed"
    assert source.name not in str(caught.value)


def test_preparation_requires_a_stable_accepted_item_identity(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    items = context.project_root / "datasets" / "active" / "items.jsonl"
    row = json.loads(items.read_text(encoding="utf-8"))
    row.pop("item_id")
    items.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_freeze=True, authorize_training=False)

    assert caught.value.code == "accepted_item_identity_missing"
    assert not (context.project_root / ".spritelab" / "training-preparation").exists()


def test_preparation_refuses_a_modified_content_addressed_publication(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    prepare_active_dataset(context, authorize_freeze=True, authorize_training=False)
    output = _publication(context.project_root)
    manifest_before = (output / "publication_manifest.json").read_bytes()
    train = output / "train.npz"
    train.write_bytes(train.read_bytes() + b"modified")

    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_freeze=True, authorize_training=False)

    assert caught.value.code == "training_publication_identity_mismatch"
    assert (output / "publication_manifest.json").read_bytes() == manifest_before


def test_preparation_failure_leaves_config_and_outside_sentinel_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _source = _project(tmp_path)
    config_before = context.config_path.read_bytes()  # type: ignore[union-attr]
    sentinel = tmp_path / "outside-sentinel.txt"
    sentinel.write_bytes(b"outside user data")

    def fail_export(*args, **kwargs):
        del args, kwargs
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(
        "spritelab.product_features.training.preparation.export_dataset_from_imported_sprites",
        fail_export,
    )
    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_freeze=True, authorize_training=True)

    assert caught.value.code == "training_preparation_failed"
    assert context.config_path.read_bytes() == config_before  # type: ignore[union-attr]
    assert sentinel.read_bytes() == b"outside user data"
    preparation_root = context.project_root / ".spritelab" / "training-preparation"
    assert list(preparation_root.iterdir()) == []


def test_preparation_endpoint_refuses_a_concurrent_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _source = _project(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def fake_prepare(*args, progress, **kwargs):
        del args, kwargs
        progress(1, 2, "Synthetic worker running.")
        started.set()
        assert release.wait(5)
        return {"image_count": 1, "paths_exposed": False}

    monkeypatch.setattr(
        "spritelab.product_features.training.preparation.prepare_active_dataset",
        fake_prepare,
    )
    app = FastAPI()
    app.include_router(create_router(context, service=object()))  # type: ignore[arg-type]
    client = TestClient(app)
    first = client.post(
        "/training/api/preparation",
        json={"authorize_freeze": True, "authorize_training": False},
    )
    assert first.status_code == 202
    assert started.wait(2)
    second = client.post(
        "/training/api/preparation",
        json={"authorize_freeze": True, "authorize_training": False},
    )
    assert second.status_code == 409
    assert second.json()["status"] == "running"
    release.set()
    for _ in range(100):
        state = client.get("/training/api/preparation").json()
        if state["status"] == "complete":
            break
        time.sleep(0.01)
    assert state["status"] == "complete"


def test_preparation_endpoint_redacts_controlled_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, source = _project(tmp_path)

    def fail(*args, **kwargs):
        del args, kwargs
        raise TrainingPreparationError("accepted_source_missing", "An accepted source image is missing.")

    monkeypatch.setattr(
        "spritelab.product_features.training.preparation.prepare_active_dataset",
        fail,
    )
    app = FastAPI()
    app.include_router(create_router(context, service=object()))  # type: ignore[arg-type]
    client = TestClient(app)
    assert (
        client.post(
            "/training/api/preparation",
            json={"authorize_freeze": True, "authorize_training": False},
        ).status_code
        == 202
    )
    for _ in range(100):
        state = client.get("/training/api/preparation").json()
        if state["status"] == "failed":
            break
        time.sleep(0.01)
    serialized = json.dumps(state)
    assert state["error"]["code"] == "accepted_source_missing"
    assert str(source) not in serialized
    assert source.name not in serialized


def test_training_page_exposes_preparation_authorizations(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)

    class PageService:
        def latest_run_id(self):
            return None

    app = FastAPI()
    app.include_router(create_router(context, service=PageService()))  # type: ignore[arg-type]
    page = TestClient(app).get("/training")
    assert page.status_code == 200
    assert 'id="authorize-freeze"' in page.text
    assert 'id="authorize-training"' in page.text
    assert 'id="preparation-progress"' in page.text
    assert "independent training-infrastructure audit" in page.text
