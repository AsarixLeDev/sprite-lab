from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from spritelab import __main__ as package_cli
from spritelab.dev_features.cli import DeveloperCommandEnvironment
from spritelab.dev_features.cli import main as dev_main
from spritelab.evaluation.memorization import EvidenceClass
from spritelab.product_core import ProductEvent, ProductStatus, ProjectContext
from spritelab.product_features.dataset.intake import build_dataset
from spritelab.product_features.dataset.plugin import create_plugin as create_dataset_plugin
from spritelab.product_features.dataset.review import DatasetReviewStore, ReviewDecisionError
from spritelab.product_features.dataset.web import discover_review_queues
from spritelab.product_features.evaluation import (
    CheckpointAvailability,
    GeneratedAsset,
    GenerationRequest,
    PlaygroundService,
    build_dashboard,
    discover_checkpoint_candidates,
    memorization_display,
)
from spritelab.product_features.evaluation.service import EvaluationRequest, EvaluationService
from spritelab.product_features.providers import (
    DeterministicMockVisionProvider,
    ImageInput,
    PrivacyClass,
    PrivacyPolicy,
    ProviderSettings,
    VisionProviderHub,
    VisionProviderRegistry,
)
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingProfile
from spritelab.product_features.training.previews import PreviewConfiguration, PreviewScheduler
from spritelab.product_features.training.service import TrainingService
from spritelab.product_runtime import _product_status, build_product_runtime
from spritelab.product_ux.acceptance import USER_JOURNEYS
from spritelab.product_web import cli as web_cli
from spritelab.product_web.app import create_app
from spritelab.remote_compute import (
    ArtifactReference,
    ComputeEstimate,
    ComputeJobRequest,
    ComputeStatus,
    FakeComputeBackend,
    ResumeRequest,
    RunPodComputeBackend,
    RunPodSettings,
    SSHComputeBackend,
    SSHSettings,
)
from spritelab.remote_compute.ssh import RemoteResult
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, ProjectState, StageState, StageStatus
from training_launch_test_utils import compute_request


@dataclass
class SyntheticProject:
    root: Path
    input_root: Path
    output_root: Path
    legal_output: Path
    context: ProjectContext
    client: TestClient
    dataset_result: Any


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sprite(path: Path, *, opaque: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    background = (20, 40, 80, 255) if opaque else (0, 0, 0, 0)
    image = Image.new("RGBA", (32, 32), background)
    for y in range(9, 23):
        for x in range(9, 23):
            image.putpixel((x, y), (240, 190, 40, 255))
    image.save(path)
    return path


def _provider_label(*, abstained: bool = False) -> dict[str, Any]:
    return {
        "state": "abstained" if abstained else "labeled",
        "domain": None if abstained else "fantasy",
        "category": None if abstained else "weapon",
        "canonical_object": None if abstained else "sword",
        "role": None if abstained else "equipment",
        "description": "Unclear silhouette." if abstained else "A gold sword.",
        "confidence": 0.1 if abstained else 0.98,
        "abstention_reasons": ["ambiguous_silhouette"] if abstained else [],
        "provider_metadata": {"synthetic": True},
    }


@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory: pytest.TempPathFactory) -> SyntheticProject:
    root = tmp_path_factory.mktemp("sprite-lab-product-complete")
    values = copy.deepcopy(DEFAULT_CONFIG)
    values["project"]["name"] = "Synthetic Sprite Lab"
    values["paths"] = {"runs": "runs", "artifacts": "experiments"}
    values["providers"]["vision"] = {"type": "runpod", "privacy_policy": "ask_before_hosted"}
    values["compute"]["training"] = {"type": "local"}
    (root / "spritelab.yaml").write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    input_root = root / "dataset"
    (input_root / "images").mkdir(parents=True)
    (input_root / "source.txt").write_text(
        "Name: Synthetic pack\nCreator: Synthetic test author\nSource: https://example.test/synthetic-pack\n",
        encoding="utf-8",
    )
    (input_root / "LICENSE").write_text("CC0-1.0\n", encoding="utf-8")
    _sprite(input_root / "images" / "accepted.png")
    _sprite(input_root / "images" / "rejected.png", opaque=True)
    output_root = root / "datasets" / "synthetic-dataset"
    base_context = ProjectContext(root, values, root / "spritelab.yaml", root / "runs")
    dataset_result = build_dataset(input_root, output_root=output_root, context=base_context)

    missing_license = root / "missing-license"
    missing_license.mkdir()
    (missing_license / "source.txt").write_text(
        "Name: Synthetic owned source\nCreator: Synthetic test author\n",
        encoding="utf-8",
    )
    _sprite(missing_license / "image.png")
    legal_output = root / "datasets" / "missing-license-dataset"
    build_dataset(missing_license, output_root=legal_output, context=base_context)

    run = root / "runs" / "train-synthetic"
    checkpoint = run / "checkpoints" / "checkpoint_step_000100_ema.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")
    _write_json(
        run / "state.json",
        {
            "schema_version": "spritelab.v3.run-state.v1",
            "run_id": "train-synthetic",
            "command": "train",
            "stage": "complete",
            "status": "COMPLETE",
            "started_at": "2026-07-13T10:00:00+00:00",
            "ended_at": "2026-07-13T10:05:00+00:00",
            "resumable": False,
            "message": "Synthetic training complete.",
            "backend_identity": {
                "friendly_run_name": "synthetic baseline",
                "training_profile": "test",
                "dataset_identity": "synthetic-dataset-v1",
                "view_identity": "synthetic-view-v1",
            },
            "checkpoints": [
                {
                    "path": "checkpoints/checkpoint_step_000100_ema.pt",
                    "step": 100,
                    "weights": "ema",
                    "sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    _write_json(run / "command.json", {"command": "train", "project_root": str(root)})
    (run / "logs").mkdir()
    (run / "logs" / "run.log").write_text("loss=0.42 validation_loss=0.51\n", encoding="utf-8")
    (run / "report").mkdir()
    (run / "report" / "index.html").write_text("<h1>Synthetic offline report</h1>", encoding="utf-8")

    benchmark = root / "benchmark.jsonl"
    benchmark.write_text(json.dumps({"prompt_id": "p1", "prompt": "gold sword"}) + "\n", encoding="utf-8")
    audit = root / "memorization-audit.json"
    _write_json(audit, {"verdict": "PENDING", "authorization": {"checkpoint_promotion": False}})
    evidence = root / "evaluation" / "candidate_evidence.json"
    _write_json(
        evidence,
        {
            "schema_version": "sprite_lab_memorization_candidate_evidence_v2",
            "pairs": [
                {"pair_id": "hard-pair", "evidence_class": "exact_rgba_nontrivial"},
                {"pair_id": "review-pair", "evidence_class": "near_pixel_review_required"},
            ],
        },
    )
    values["dataset"] = {
        **values["dataset"],
        "output_root": str(output_root),
        "identity": "synthetic-dataset-v1",
        "view_identity": "synthetic-view-v1",
    }
    values["evaluation"] = {
        **values["evaluation"],
        "benchmark": str(benchmark),
        "memorization_audit": str(audit),
    }
    context = ProjectContext(root, values, root / "spritelab.yaml", root / "runs")
    plugins = tuple(
        create_dataset_plugin(folder_chooser=lambda: input_root) if plugin.plugin_id == "dataset.intake" else plugin
        for plugin in build_product_runtime().plugins
    )
    app = create_app(context, plugins=plugins)
    return SyntheticProject(root, input_root, output_root, legal_output, context, TestClient(app), dataset_result)


def _csrf_headers(project: SyntheticProject) -> dict[str, str]:
    return {"x-csrf-token": project.client.app.state.spritelab_csrf_token}


def _approved_input(project: SyntheticProject) -> str:
    response = project.client.post(
        "/dataset/api/folders/choose",
        json={},
        headers=_csrf_headers(project),
    )
    assert response.status_code == 200
    return str(response.json()["approval"]["approval_id"])


def _compute_request(
    project: SyntheticProject, name: str = "synthetic-job", backend_id: str = "fake"
) -> ComputeJobRequest:
    request = compute_request(project.root / "validated-compute" / backend_id / name, backend_id)
    return replace(request, idempotency_key=name)


def _dashboard_event(event_type: str = "progress", **metrics: Any) -> ProductEvent:
    return ProductEvent(
        "synthetic-training",
        datetime.now(timezone.utc).isoformat(),
        "training",
        "seed",
        event_type,
        ProductStatus.RUNNING,
        current=int(metrics.get("optimizer_step", 0)),
        total=1_000,
        metrics=metrics,
    )


def test_01_no_argument_app_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    dispatched: list[list[str]] = []
    monkeypatch.setattr(web_cli, "main", lambda argv=(): dispatched.append(list(argv)))
    package_cli.main(["v3"])
    package_cli.main(["v3", "app", "--no-open"])
    assert dispatched == [[], ["--no-open"]]


def test_02_valid_folder_selection(synthetic_project: SyntheticProject) -> None:
    approval_id = _approved_input(synthetic_project)
    response = synthetic_project.client.post(
        "/dataset/api/inspect",
        json={"approval_id": approval_id},
        headers=_csrf_headers(synthetic_project),
    )
    assert response.status_code == 200
    assert response.json()["image_count"] == 2


def test_03_source_and_license_check(synthetic_project: SyntheticProject) -> None:
    approval_id = _approved_input(synthetic_project)
    response = synthetic_project.client.post(
        "/dataset/api/inspect",
        json={"approval_id": approval_id},
        headers=_csrf_headers(synthetic_project),
    ).json()
    assert response["source_ready"] and response["license_ready"]
    assert response["next_action"] == "Build dataset"


def test_04_dataset_build(synthetic_project: SyntheticProject) -> None:
    assert synthetic_project.dataset_result.status == ProductStatus.COMPLETE
    assert (synthetic_project.output_root / "result.json").is_file()
    assert (synthetic_project.output_root / "raw_extraction" / "extraction_manifest.jsonl").is_file()


def test_web_dataset_build_runs_as_a_polled_background_job(synthetic_project: SyntheticProject) -> None:
    response = synthetic_project.client.post(
        "/dataset/api/build",
        json={"approval_id": _approved_input(synthetic_project)},
        headers=_csrf_headers(synthetic_project),
    )
    assert response.status_code == 202
    status_url = response.json()["status_url"]
    statuses: list[str] = []
    job: dict[str, Any] = {}
    for _ in range(100):
        job = synthetic_project.client.get(status_url).json()
        statuses.append(job["status"])
        if job["status"] in {"complete", "failed"}:
            break
        time.sleep(0.02)
    assert job["status"] == "complete"
    assert job["result"]["status"] == ProductStatus.COMPLETE.value
    assert [entry["message"] for entry in job["logs"]] == [
        "Queued dataset build.",
        "Started dataset build.",
        "Dataset build finished.",
    ]
    assert set(statuses) <= {"queued", "running", "complete"}


def test_web_dataset_build_rejects_stale_staging_tokens_without_cleanup(
    synthetic_project: SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "must-survive.txt"
    marker.write_text("preserve", encoding="utf-8")
    arbitrary_folder = tmp_path / "selected-folder"
    arbitrary_folder.mkdir()

    stale_payloads = (
        ({"staging_token": "expired-token"}, "folder_approval_invalid"),
        ({"staging_token": "expired-token", "folder": ""}, "browser_path_not_allowed"),
        (
            {"staging_token": "expired-token", "folder": str(arbitrary_folder)},
            "browser_path_not_allowed",
        ),
    )
    for payload, expected_error in stale_payloads:
        response = synthetic_project.client.post(
            "/dataset/api/build",
            json=payload,
            headers=_csrf_headers(synthetic_project),
        )
        assert response.status_code == 422
        assert response.json()["error_code"] == expected_error
        assert marker.read_text(encoding="utf-8") == "preserve"
        assert arbitrary_folder.is_dir()


def test_05_accepted_rejected_summary(synthetic_project: SyntheticProject) -> None:
    counts = synthetic_project.dataset_result.data["counts"]
    assert counts["processed"] == 2
    assert counts["accepted"] == 1
    assert counts["rejected"] == 1


def test_06_review_can_be_skipped() -> None:
    journey = next(item for item in USER_JOURNEYS if item.title == "Optional rejection review skipped")
    assert journey.screens[0].shown_primary_action == "Continue without review"
    assert "Rescue images" in journey.screens[0].secondary_actions


def test_07_review_reopens_in_shared_product(synthetic_project: SyntheticProject) -> None:
    page = synthetic_project.client.get("/review")
    assert page.status_code == 200
    assert "Rescue images" in page.text and "Memorization candidates" in page.text
    assert synthetic_project.client.get("/dataset/review").status_code == 200


def test_08_rejected_item_is_rescued(synthetic_project: SyntheticProject) -> None:
    store = DatasetReviewStore(synthetic_project.output_root)
    item = next(row for row in store.queue()["items"] if row["review_rescuable"])
    decision = store.apply(item["item_id"], "keep")
    assert decision["current_disposition"] == "accepted"


def test_09_missing_license_cannot_be_visually_overridden(synthetic_project: SyntheticProject) -> None:
    store = DatasetReviewStore(synthetic_project.legal_output)
    item = store.queue()["items"][0]
    with pytest.raises(ReviewDecisionError, match="source/license evidence"):
        store.apply(item["item_id"], "keep")


def test_10_dataset_build_without_provider_preserves_technical_result(
    synthetic_project: SyntheticProject,
) -> None:
    assert synthetic_project.dataset_result.data["semantic"]["provider_status"] == "not_configured"
    assert synthetic_project.dataset_result.data["counts"]["image_only_eligible"] == 1


def test_11_fake_local_provider() -> None:
    provider = DeterministicMockVisionProvider({"sprite": _provider_label()})
    settings = ProviderSettings(privacy_policy=PrivacyPolicy.LOCAL_ONLY, maximum_retries=0)
    hub = VisionProviderHub(
        VisionProviderRegistry(settings, providers=(provider,), plugin_entry_points=()),
        settings=settings,
    )
    result = hub.label_images(provider, (ImageInput("sprite", b"synthetic"),), prompt="Label conservatively.")
    assert result.successful_count == 1 and provider.call_count == 1


def test_12_hosted_privacy_confirmation() -> None:
    class HostedFake(DeterministicMockVisionProvider):
        privacy_class = PrivacyClass.HOSTED

    provider = HostedFake({"sprite": _provider_label()})
    settings = ProviderSettings(privacy_policy=PrivacyPolicy.ASK_BEFORE_HOSTED, maximum_retries=0)
    hub = VisionProviderHub(
        VisionProviderRegistry(settings, providers=(provider,), plugin_entry_points=()),
        settings=settings,
    )
    prompts: list[str] = []
    hub.label_images(
        provider,
        (ImageInput("sprite", b"synthetic"),),
        prompt="Label.",
        confirm_hosted=lambda prompt: prompts.append(prompt) or True,
    )
    assert len(prompts) == 1 and provider.call_count == 1


def test_13_conservative_semantic_abstention() -> None:
    provider = DeterministicMockVisionProvider({"sprite": _provider_label(abstained=True)})
    settings = ProviderSettings(privacy_policy=PrivacyPolicy.ALLOW_HOSTED, maximum_retries=0)
    result = VisionProviderHub(
        VisionProviderRegistry(settings, providers=(provider,), plugin_entry_points=()),
        settings=settings,
    ).label_images(provider, (ImageInput("sprite", b"synthetic"),), prompt="Label.")
    assert result.results[0].label.state.value == "abstained"
    assert result.results[0].label.abstention_reasons == ("ambiguous_silhouette",)


def test_14_image_only_dataset_is_preserved(synthetic_project: SyntheticProject) -> None:
    capability = next(
        item for item in synthetic_project.dataset_result.capabilities if item.capability_id == "dataset.image_only"
    )
    conditioned = next(
        item for item in synthetic_project.dataset_result.capabilities if item.capability_id == "dataset.conditioned"
    )
    assert capability.status == ProductStatus.READY
    assert conditioned.status == ProductStatus.UNAVAILABLE


def test_15_training_is_blocked_while_audit_is_pending(synthetic_project: SyntheticProject) -> None:
    backend = FakeComputeBackend()
    result = TrainingService(synthetic_project.context, backend).start()
    assert result.status == ProductStatus.BLOCKED
    assert "launch" not in backend.calls and "prepare" not in backend.calls


def test_16_synthetic_ready_training_plan() -> None:
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Synthetic baseline",
        1_800,
        True,
        "fake",
        {"campaign_identity": "synthetic", "seeds": [731001]},
        (),
        ComputeEstimate(60, 0, trustworthy=True),
    )
    assert plan.ready and plan.to_dict()["dataset"]["status"] == "Ready"


def test_17_fake_local_compute_execution(synthetic_project: SyntheticProject) -> None:
    backend = FakeComputeBackend()
    request = _compute_request(synthetic_project)
    prepared = backend.prepare(synthetic_project.context, request)
    job = backend.launch(prepared, request)
    assert job.status == ComputeStatus.RUNNING
    assert backend.calls == ["prepare", "launch"]


def test_18_fake_ssh_lifecycle(synthetic_project: SyntheticProject) -> None:
    class FakeTransport:
        def execute(self, _script: str, payload: dict[str, Any]) -> RemoteResult:
            if "disk_free_bytes" in _script:
                return RemoteResult(0, json.dumps({"python": "3.11", "disk_free_bytes": 10_000}))
            if "job_id" in payload:
                return RemoteResult(0, json.dumps({"status": "STARTING", **payload}))
            if set(payload) == {"state_path"}:
                return RemoteResult(0, json.dumps({"status": "RUNNING"}))
            return RemoteResult(0, json.dumps({"changed": True, **payload}))

        def upload(self, _local: Path, _remote: str) -> RemoteResult:
            return RemoteResult(0)

        def download(self, _remote: str, _local: Path) -> RemoteResult:
            return RemoteResult(0)

    backend = SSHComputeBackend(
        SSHSettings("synthetic.invalid", "trainer", "/workspace/sprite-lab", cloud=False),
        transport=FakeTransport(),
    )
    request = _compute_request(synthetic_project, "ssh-job", "ssh")
    prepared = backend.prepare(synthetic_project.context, request)
    job = backend.launch(prepared, request)
    assert backend.poll(job).status == ComputeStatus.RUNNING


def test_19_runpod_is_unavailable_scaffold_only(
    monkeypatch: pytest.MonkeyPatch, synthetic_project: SyntheticProject
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    backend = RunPodComputeBackend(RunPodSettings(gpu_type_ids=("GPU",), image_name="image:tag"))
    capability = backend.probe(synthetic_project.context)[0]
    assert capability.status == ProductStatus.UNAVAILABLE
    assert capability.details["provider_calls"] == 0


def test_20_training_events_and_charts() -> None:
    dashboard = DashboardState("synthetic-training", "fake")
    dashboard.apply(
        _dashboard_event(
            seed=731001,
            optimizer_step=100,
            total_steps=1_000,
            loss=0.42,
            validation_loss=0.51,
            learning_rate=0.0002,
        )
    )
    state = dashboard.to_dict()
    assert state["loss_curve"] and state["validation_loss_curve"] and state["learning_rate_curve"]


def test_21_safe_pause_and_resume(synthetic_project: SyntheticProject) -> None:
    backend = FakeComputeBackend()
    request = _compute_request(synthetic_project, "resume-job")
    prepared = backend.prepare(synthetic_project.context, request)
    job = backend.launch(prepared, request)
    assert backend.pause(job).changed
    resumed_request = replace(request, idempotency_key="resume-job-safe")
    checkpoint = ArtifactReference(
        "checkpoint.pt",
        "a" * 64,
        prepared.remote_identity,
        synthetic_project.root / "checkpoint.pt",
        downloaded=True,
        hash_verified=True,
        remote_identity_verified=True,
    )
    resumed = backend.resume(prepared, ResumeRequest(resumed_request, checkpoint, safe_resume=True))
    assert resumed.status == ComputeStatus.RUNNING


def test_22_exploratory_previews(synthetic_project: SyntheticProject) -> None:
    scheduler = PreviewScheduler(
        PreviewConfiguration(interval_steps=500, prompts=("gold sword",), generation_seeds=(42,)),
        lambda **kwargs: kwargs["output_path"],
    )
    events = scheduler.generate(
        run_id="synthetic",
        run_root=synthetic_project.root,
        checkpoint=synthetic_project.root / "checkpoint.pt",
        checkpoint_step=500,
        training_seed=731001,
        checkpoint_schedule=[500],
    )
    assert events[0].metrics["exploratory"] is True
    assert events[0].metrics["benchmark_evidence"] is False


def test_23_synthetic_verified_checkpoint(synthetic_project: SyntheticProject) -> None:
    catalog = discover_checkpoint_candidates(synthetic_project.root / "runs", project_root=synthetic_project.root)
    assert catalog.default_checkpoint_id
    assert catalog.find(None).availability == CheckpointAvailability.ELIGIBLE


class _FakeEvaluationGenerator:
    remote = False
    billable = False

    def __init__(self) -> None:
        self.calls = 0

    def generate_benchmark(self, *, output_directory: Path, emit: Any, **_kwargs: Any) -> Path:
        self.calls += 1
        output_directory.mkdir(parents=True, exist_ok=True)
        emit("generation", 1, 1, "Generated synthetic sample.")
        return output_directory


def _fake_evaluator(_generated: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_image_metrics.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "prompt_id": "p1",
                "prompt": "gold sword",
                "category": "weapon",
                "metrics": {"pixel_art": {"unique_palette_size": 8}},
                "conditional_adherence": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "generation_benchmark_v1.0",
        "summary": {
            "sample_count": 1,
            "hard_validity": {"pass_rate": 1.0},
            "conditional": {"represented_rate": 1.0},
            "pixel_art": {"palette_size_mean": 8.0},
            "diversity": {"exact_duplicate_rate": 0.0},
            "memorization": {"hard_evidence_count": 0, "review_required_count": 1},
        },
        "promotion": {"memorization_machine_status": "manual_review_required"},
    }


def test_24_fake_evaluation(synthetic_project: SyntheticProject) -> None:
    generator = _FakeEvaluationGenerator()
    service = EvaluationService(
        project_root=synthetic_project.root,
        config=synthetic_project.context.config,
        generator=generator,
        evaluator=_fake_evaluator,
        output_root=synthetic_project.root / "evaluation-output",
    )
    result = service.run(EvaluationRequest(explicit_action=True))
    assert result.status == ProductStatus.BLOCKED
    assert "incomplete" in result.message.casefold()
    assert generator.calls == 1 and result.data["promotion_actions"] == 0


def test_25_metric_charts() -> None:
    report = {
        "schema_version": "generation_benchmark_v1.0",
        "summary": {
            "sample_count": 1,
            "hard_validity": {"pass_rate": 1.0},
            "conditional": {"represented_rate": 1.0},
            "pixel_art": {"palette_size_mean": 8.0},
            "diversity": {"exact_duplicate_rate": 0.0},
            "memorization": {"hard_evidence_count": 0, "review_required_count": 0},
        },
        "promotion": {"memorization_machine_status": "pass"},
    }
    rows = [
        {
            "sample_id": "sample-1",
            "prompt": "gold sword",
            "category": "weapon",
            "metrics": {"pixel_art": {"unique_palette_size": 8}},
            "conditional_adherence": 1.0,
        }
    ]
    dashboard = build_dashboard(report, rows)
    assert dashboard["charts"] and any(chart["status"] == "AVAILABLE" for chart in dashboard["charts"])


def test_26_memorization_review_link() -> None:
    display = memorization_display([{"pair_id": "pair", "evidence_class": "near_pixel_review_required"}])
    assert display["review_link"] is None
    assert display["evidence_state"] == "incomplete"


def test_27_unsigned_review_is_rejected(synthetic_project: SyntheticProject) -> None:
    log = synthetic_project.root / "unsigned-reviews.jsonl"
    log.write_text(
        json.dumps(
            {
                "schema_version": "sprite_lab_memorization_review_event_v2",
                "pair_id": "pair",
                "review_outcome": "likely_false_positive",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    display = memorization_display(
        [{"pair_id": "pair", "evidence_class": "near_pixel_review_required"}],
        review_log=log,
    )
    assert display["items"] == []
    assert display["review_action_available"] is False


def test_28_hard_evidence_is_not_clearable() -> None:
    display = memorization_display([{"pair_id": "hard", "evidence_class": EvidenceClass.EXACT_RGBA_NONTRIVIAL.value}])
    assert display["items"] == []
    assert display["review_action_available"] is False


class _FakePlaygroundGenerator:
    remote = False
    billable = False

    def generate(self, **kwargs: Any) -> list[GeneratedAsset]:
        return [GeneratedAsset(f"fake-{index}".encode()) for index in range(kwargs["image_count"])]


def _playground_result(project: SyntheticProject) -> dict[str, Any]:
    catalog = discover_checkpoint_candidates(project.root / "runs", project_root=project.root)
    service = PlaygroundService(
        catalog,
        output_root=project.root / "playground-output",
        generator=_FakePlaygroundGenerator(),
    )
    return service.generate(
        GenerationRequest(prompt="gold sword", checkpoint_id=catalog.default_checkpoint_id, image_count=1),
        explicit_action=True,
    )


def test_29_fake_playground_generation(synthetic_project: SyntheticProject) -> None:
    result = _playground_result(synthetic_project)
    assert result["scope"] == "EXPLORATORY" and len(result["results"]) == 1


def test_30_exploratory_output_is_excluded_from_benchmark(synthetic_project: SyntheticProject) -> None:
    result = _playground_result(synthetic_project)
    assert result["excluded_from_frozen_benchmark"] is True
    assert result["excluded_from_promotion_evidence"] is True


def test_31_simple_product_status() -> None:
    result = _product_status(argparse.Namespace(), [])
    assert result.data["next_command"] == "python -m spritelab v3 dataset build <folder>"
    text = result.message.casefold()
    assert "commit" not in text and "audit id" not in text and "manifest path" not in text


def test_32_detailed_developer_status(synthetic_project: SyntheticProject, capsys: pytest.CaptureFixture[str]) -> None:
    config = ProjectConfig(synthetic_project.root, synthetic_project.root / "spritelab.yaml", DEFAULT_CONFIG)
    state = ProjectState(
        "synthetic",
        synthetic_project.root,
        config.path,
        "a" * 40,
        [
            StageState(
                "training-infrastructure-audit",
                "Training audit",
                StageStatus.BLOCKED,
                "Independent audit pending.",
                audit=AuditStatus.NOT_AUDITED,
                source_commit="b" * 40,
            )
        ],
    )
    environment = DeveloperCommandEnvironment(lambda: config, lambda _config: state)
    with pytest.raises(SystemExit) as caught:
        dev_main(["status", "--json"], environment=environment)
    output = capsys.readouterr().out
    assert caught.value.code == 0
    assert "source_commit" in output and '"audit"' in output


def test_33_logs_and_report(synthetic_project: SyntheticProject) -> None:
    logs = synthetic_project.client.get("/runs/train-synthetic/logs")
    report = synthetic_project.client.get("/runs/train-synthetic/report")
    assert logs.status_code == 200 and "loss=0.42" in logs.text
    assert report.status_code == 200 and "Synthetic offline report" in report.text


def test_34_narrow_screen_route_structure(synthetic_project: SyntheticProject) -> None:
    page = synthetic_project.client.get("/").text
    assert 'name="viewport"' in page
    for title in ("Home", "Dataset", "Training", "Evaluation", "Playground", "Runs", "Settings"):
        assert title in page
    assert "Developer audit" not in page


def test_35_keyboard_accessible_controls(synthetic_project: SyntheticProject) -> None:
    page = synthetic_project.client.get("/dataset/review").text
    assert "ArrowLeft" in page and "ArrowRight" in page
    assert "<kbd>K</kbd>" in page and "<kbd>E</kbd>" in page
    assert 'aria-label="Keep' in page


def test_36_no_secrets_or_tracebacks(synthetic_project: SyntheticProject) -> None:
    response = synthetic_project.client.post(
        "/dataset/api/inspect",
        json={"folder": str(synthetic_project.root / "does-not-exist")},
        headers=_csrf_headers(synthetic_project),
    )
    home = synthetic_project.client.get("/")
    combined = response.text + home.text
    assert response.status_code == 422
    assert "Traceback (most recent call last)" not in combined
    assert "api_key" not in combined.casefold() and "bearer " not in combined.casefold()
    assert "content-security-policy" in home.headers
    assert "cdn" not in home.headers["content-security-policy"].casefold()


def test_shared_review_formats_remain_authoritative(synthetic_project: SyntheticProject) -> None:
    routing = discover_review_queues(synthetic_project.context)
    formats = {item["queue_id"]: item["authoritative_format"] for item in routing["queues"]}
    assert formats["dataset"] == "spritelab.dataset.review_queue.v1"
    assert formats["memorization"] == "sprite_lab_memorization_candidate_evidence_v2"
    assert routing["formats_preserved"] is True
