from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from spritelab.product_core import ProjectContext
from spritelab.product_features.training.preparation import (
    CONDITIONED_TRAINING_CONTRACT_SCHEMA,
    TrainingPreparationError,
    prepare_active_dataset,
)
from spritelab.product_features.training.preparation_jobs import (
    MAX_PREPARATION_EVENTS,
    PreparationJobRepository,
    PreparationJobStateError,
)
from spritelab.product_features.training.web import create_router
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
    publications = list((project / ".spritelab" / "training-preparation").glob("baseline-*"))
    assert len(publications) == 1
    return publications[0]


def _make_training_unsupported(context: ProjectContext, source: Path) -> bytes:
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
    image.putpixel((10, 10), (220, 80, 40, 128))
    image.save(source)
    content = source.read_bytes()
    items = context.project_root / "datasets" / "active" / "items.jsonl"
    item = json.loads(items.read_text(encoding="utf-8"))
    item["byte_sha256"] = hashlib.sha256(content).hexdigest()
    items.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")
    return content


def test_preparation_builds_immutable_nonproduction_baseline_without_config_activation(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    config_before = context.config_path.read_bytes()  # type: ignore[union-attr]

    result = prepare_active_dataset(
        context,
        authorize_baseline=True,
    )

    assert result["image_count"] == 1
    assert result["artifact_kind"] == "image_only_baseline"
    assert result["immutable"] is True
    assert result["production_authorized"] is False
    assert result["training_authorized"] is False
    assert result["activated"] is False
    assert result["required_contract"] == CONDITIONED_TRAINING_CONTRACT_SCHEMA
    assert result["remaining_gate"] == "audited_conditioned_dataset_v5_freeze_and_campaign"
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

    baseline = json.loads((output / "baseline_manifest.json").read_text(encoding="utf-8"))
    assert baseline["artifact_kind"] == "image_only_baseline"
    assert baseline["immutable"] is True
    assert baseline["production_authorized"] is False
    assert baseline["training_eligible"] is False
    assert baseline["activation_forbidden"] is True
    persisted = (output / "baseline_campaign.json").read_text(encoding="utf-8")
    assert str(context.project_root) not in persisted
    campaign = json.loads(persisted)["product_profiles"]["image_only_baseline"]["campaign"]
    assert campaign["executable"] is False
    assert campaign["launch_authorized"] is False
    assert type(campaign["executable"]) is bool
    assert type(campaign["launch_authorized"]) is bool
    assert "dataset_freeze_hash" not in campaign["identities"]
    assert len(campaign["seeds"]) == 3

    assert context.config_path.read_bytes() == config_before  # type: ignore[union-attr]
    config = ProjectConfig.load(context.project_root)
    assert config.values["dataset"]["view_manifest"] == DEFAULT_CONFIG["dataset"]["view_manifest"]
    assert config.values["dataset"]["freeze_manifest"] == DEFAULT_CONFIG["dataset"]["freeze_manifest"]
    assert config.values["training"]["dataset_freeze"] == DEFAULT_CONFIG["training"]["dataset_freeze"]
    assert config.values["training"]["campaign_config"] == DEFAULT_CONFIG["training"]["campaign_config"]
    assert config.values["execution"]["allow_dataset_production_freeze"] is False
    assert config.values["execution"]["allow_training"] is False


def test_preparation_reverifies_source_hash_before_reuse(tmp_path: Path) -> None:
    context, source = _project(tmp_path)
    prepare_active_dataset(context, authorize_baseline=True)
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_baseline=True)

    assert caught.value.code == "accepted_source_changed"
    assert source.name not in str(caught.value)


def test_preparation_identifies_an_accepted_image_the_encoder_does_not_support(tmp_path: Path) -> None:
    context, source = _project(tmp_path)
    _make_training_unsupported(context, source)

    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_baseline=True)

    error = caught.value
    public = error.to_public_dict()
    assert error.code == "canonical_encoding_failed"
    assert public["item_id"] == "sword-1"
    assert any("soft alpha values" in reason for reason in public["reasons"])
    assert public["next_action"] == (
        "Remove this image from the currently built dataset, then retry baseline preparation."
    )
    assert str(source) not in json.dumps(public)
    assert source.name not in json.dumps(public)


def test_preparation_requires_a_stable_accepted_item_identity(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    items = context.project_root / "datasets" / "active" / "items.jsonl"
    row = json.loads(items.read_text(encoding="utf-8"))
    row.pop("item_id")
    items.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_baseline=True)

    assert caught.value.code == "accepted_item_identity_missing"
    assert not (context.project_root / ".spritelab" / "training-preparation").exists()


def test_preparation_refuses_a_modified_content_addressed_publication(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    prepare_active_dataset(context, authorize_baseline=True)
    output = _publication(context.project_root)
    manifest_before = (output / "publication_manifest.json").read_bytes()
    train = output / "train.npz"
    train.write_bytes(train.read_bytes() + b"modified")

    with pytest.raises(TrainingPreparationError) as caught:
        prepare_active_dataset(context, authorize_baseline=True)

    assert caught.value.code == "training_publication_identity_mismatch"
    assert (output / "publication_manifest.json").read_bytes() == manifest_before


def test_preparation_reuse_rejects_hardlinked_publication_artifact(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    prepare_active_dataset(context, authorize_baseline=True)
    output = _publication(context.project_root)
    train = output / "train.npz"
    outside = tmp_path / "outside-train.npz"
    outside.write_bytes(train.read_bytes())
    linked = tmp_path / "linked-train.npz"
    try:
        os.link(outside, linked)
    except OSError:
        pytest.skip("hard links are unavailable in this test session")
    os.replace(linked, train)

    with pytest.raises(TrainingPreparationError, match="incomplete"):
        prepare_active_dataset(context, authorize_baseline=True)
    assert outside.read_bytes() == train.read_bytes()


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
        prepare_active_dataset(context, authorize_baseline=True)

    assert caught.value.code == "training_preparation_failed"
    assert context.config_path.read_bytes() == config_before  # type: ignore[union-attr]
    assert sentinel.read_bytes() == b"outside user data"
    preparation_root = context.project_root / ".spritelab" / "training-preparation"
    assert list(preparation_root.glob(".staging-*")) == []
    residues = list(preparation_root.glob(".spritelab-retired-tree-*"))
    assert len(residues) == 1
    assert set(preparation_root.iterdir()) == set(residues)
    assert list(residues[0].iterdir()) == []


def test_preparation_endpoint_rejects_any_training_or_freeze_authorization(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    app = FastAPI()
    app.include_router(create_router(context, service=object()))  # type: ignore[arg-type]
    client = TestClient(app)

    for legacy_authorization in ({"authorize_training": True}, {"authorize_freeze": True}):
        response = client.post(
            "/training/api/preparation",
            json={"authorize_baseline": True, **legacy_authorization},
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == "baseline_cannot_authorize_training"
    assert client.get("/training/api/preparation").json()["status"] == "not_started"
    assert not (context.project_root / ".spritelab" / "training-preparation").exists()


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
        json={"authorize_baseline": True},
    )
    assert first.status_code == 202
    assert started.wait(2)
    second = client.post(
        "/training/api/preparation",
        json={"authorize_baseline": True},
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
            json={"authorize_baseline": True},
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


def test_preparation_failure_shows_the_verified_image_reason_and_dataset_remediation(tmp_path: Path) -> None:
    context, source = _project(tmp_path)
    expected_image = _make_training_unsupported(context, source)
    app = FastAPI()
    app.include_router(create_router(context, service=object()))  # type: ignore[arg-type]
    client = TestClient(app)

    assert client.post("/training/api/preparation", json={"authorize_baseline": True}).status_code == 202
    for _ in range(100):
        state = client.get("/training/api/preparation").json()
        if state["status"] == "failed":
            break
        time.sleep(0.01)

    error = state["error"]
    assert state["status"] == "failed"
    assert error["code"] == "canonical_encoding_failed"
    assert error["item_id"] == "sword-1"
    assert any("soft alpha values" in reason for reason in error["reasons"])
    assert error["next_action"].startswith("Remove this image from the currently built dataset")
    assert error["review_url"] == "/dataset/review"
    assert error["image_url"].startswith("/training/api/preparation/error-image?item_id=")
    serialized = json.dumps(state)
    assert str(source) not in serialized
    assert source.name not in serialized

    preview = client.get(error["image_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert preview.headers["cache-control"] == "no-store"
    assert preview.content == expected_image

    source.write_bytes(expected_image + b"changed after failure")
    changed = client.get(error["image_url"])
    assert changed.status_code == 409
    assert changed.json()["error_code"] == "accepted_source_changed"
    assert str(source) not in changed.text


def test_preparation_error_image_is_limited_to_the_current_failed_item(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    app = FastAPI()
    app.include_router(create_router(context, service=object()))  # type: ignore[arg-type]
    client = TestClient(app)

    response = client.get("/training/api/preparation/error-image?item_id=sword-1")

    assert response.status_code == 404
    assert response.json()["error_code"] == "preparation_error_image_unavailable"


def test_training_page_exposes_preparation_authorizations(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)

    class PageService:
        def latest_run_id(self):
            return None

    app = FastAPI()
    app.include_router(create_router(context, service=PageService()))  # type: ignore[arg-type]
    page = TestClient(app).get("/training")
    assert page.status_code == 200
    assert 'id="authorize-baseline"' in page.text
    assert 'id="authorize-freeze"' not in page.text
    assert 'id="authorize-training"' not in page.text
    assert 'id="preparation-progress"' in page.text
    assert 'id="preparation-error-card"' in page.text
    assert 'id="preparation-error-image"' in page.text
    assert "Why preparation stopped" in page.text
    assert "Review the currently built dataset" in page.text
    assert "This never unlocks training" in page.text
    assert "conditioned Dataset-v5 freeze" in page.text
    root = Path(__file__).resolve().parents[1]
    javascript = (root / "src/spritelab/product_features/training/static/training.js").read_text(encoding="utf-8")
    stylesheet = (root / "src/spritelab/product_features/training/static/training.css").read_text(encoding="utf-8")
    assert "renderPreparationError(data.error)" in javascript
    assert "image.src=error.image_url" in javascript
    assert "error.reasons" in javascript
    assert ".preparation-error-card[hidden]{display:none}" in stylesheet


def test_preparation_job_state_survives_router_recreation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _source = _project(tmp_path)

    def fake_prepare(*args, progress, **kwargs):
        del args, kwargs
        progress(1, 1, "Synthetic durable preparation complete.")
        return {"image_count": 1, "paths_exposed": False}

    monkeypatch.setattr(
        "spritelab.product_features.training.preparation.prepare_active_dataset",
        fake_prepare,
    )
    first_app = FastAPI()
    first_app.include_router(create_router(context, service=object()))  # type: ignore[arg-type]
    first_client = TestClient(first_app)
    assert first_client.post("/training/api/preparation", json={"authorize_baseline": True}).status_code == 202
    for _ in range(100):
        completed = first_client.get("/training/api/preparation").json()
        if completed["status"] == "complete":
            break
        time.sleep(0.01)
    assert completed["status"] == "complete"
    assert completed["result_identity"]

    recreated_app = FastAPI()
    recreated_app.include_router(create_router(context, service=object()))  # type: ignore[arg-type]
    reconstructed = TestClient(recreated_app).get("/training/api/preparation").json()
    assert reconstructed == completed
    assert "worker_owner" not in reconstructed
    assert "worker_pid" not in reconstructed


def test_preparation_job_recovers_dead_worker_and_stale_owned_lock(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    repository = PreparationJobRepository(context)
    state = repository.begin(
        {
            "input_identity": "input",
            "source_identity": "source",
            "config_identity": "config",
            "code_identity": "code",
        }
    )
    retained = repository.load()
    retained["worker_pid"] = 999_999_999
    repository.state_path.write_text(json.dumps(retained), encoding="utf-8")
    repository.lock_path.write_text("pid=999999999\n", encoding="utf-8")

    recreated = PreparationJobRepository(context)
    reconstructed = recreated.reconstruct()

    assert reconstructed["job_id"] == state["job_id"]
    assert reconstructed["status"] == "interrupted"
    assert reconstructed["error"]["code"] == "training_preparation_interrupted"
    assert recreated.stale_lock_path.read_text(encoding="utf-8") == "pid=999999999\n"
    assert not recreated.lock_path.exists()


def test_preparation_event_history_is_capped_and_rejects_hardlink_append_target(tmp_path: Path) -> None:
    context, _source = _project(tmp_path)
    repository = PreparationJobRepository(context)
    state = repository.begin(
        {
            "input_identity": "input",
            "source_identity": "source",
            "config_identity": "config",
            "code_identity": "code",
        }
    )
    job_id = str(state["job_id"])
    owner = str(state["worker_owner"])
    for index in range(MAX_PREPARATION_EVENTS + 5):
        repository.progress(job_id, owner, index, MAX_PREPARATION_EVENTS + 5, f"event {index}")
    assert len(repository.events_path.read_text(encoding="utf-8").splitlines()) == MAX_PREPARATION_EVENTS

    retained_events = repository.events_path.with_suffix(".retained")
    repository.events_path.replace(retained_events)
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(b"outside user data")
    try:
        os.link(outside, repository.events_path)
    except OSError:
        pytest.skip("hard links are unavailable in this test session")
    with pytest.raises(PreparationJobStateError, match="unsafe"):
        repository.progress(job_id, owner, 1, 1, "must not append")
    assert outside.read_bytes() == b"outside user data"
