from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from spritelab.product_core import ProductEvent, ProductResult, ProductStatus, ProjectContext
from spritelab.product_features.evaluation.checkpoints import discover_checkpoint_candidates
from spritelab.product_features.training import build_plugin
from spritelab.product_features.training.activation import ConditionedActivationError
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingProfile
from spritelab.product_features.training.plans import TrainingPlanResolver
from spritelab.product_features.training.service import TrainingService, TrainingSession
from spritelab.product_features.training.web import create_router
from spritelab.remote_compute import ComputeEstimate, ComputeStatus, FakeComputeBackend, LocalComputeBackend
from spritelab.training.campaign import DEFAULT_SEEDS, file_sha256, plan_campaign, stable_hash
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, ProjectState, StageState, StageStatus


def _context(root: Path, values: dict | None = None) -> ProjectContext:
    config = values or ProjectConfig.load(root).values
    return ProjectContext(root, config, root / "spritelab.yaml", root / "runs/v3")


def _ready_state(root: Path) -> ProjectState:
    return ProjectState(
        "test",
        root,
        root / "spritelab.yaml",
        "abc",
        [
            StageState(
                "dataset-freeze",
                "freeze",
                StageStatus.COMPLETE,
                "Frozen dataset identity is authoritative.",
                production_authorized=True,
            ),
            StageState(
                "training-infrastructure-audit",
                "audit",
                StageStatus.COMPLETE,
                "Applicable audit passed.",
                audit=AuditStatus.PASS,
            ),
            StageState(
                "training-campaign",
                "campaign",
                StageStatus.READY,
                "Training is authorized.",
                production_authorized=True,
            ),
        ],
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _spec(tmp_path: Path) -> dict:
    inputs = tmp_path / "inputs"
    dataset, split, vocabulary, benchmark = [
        inputs / name for name in ("dataset.json", "split.json", "vocab.json", "benchmark.json")
    ]
    for path, payload in (
        (dataset, {"view": "test"}),
        (split, {"train": ["a"]}),
        (vocabulary, {"tokens": ["sprite"]}),
        (benchmark, {"prompts": ["sprite"]}),
    ):
        _write_json(path, payload)
    optimizer = {"name": "adamw", "learning_rate": 0.001}
    schedule = {"name": "cosine", "warmup_steps": 10}
    loss = {"name": "uniform_velocity"}
    determinism = {"mode": "strict"}
    evaluation = {
        "cadence": 100,
        "include_step_zero": False,
        "benchmark_manifest_hash": file_sha256(benchmark),
        "benchmark_manifest_path": str(benchmark),
        "ema_policy": "both",
        "live_weight_evaluation_policy": "required",
    }
    evaluation["evaluation_config_hash"] = stable_hash(
        {key: value for key, value in evaluation.items() if not key.startswith("benchmark_manifest_")}
    )
    return {
        "campaign_id": "product_training_test",
        "purpose": "Test product training without real execution.",
        "architecture_cells": [{"cell_id": "baseline", "comparison_values": {}}],
        "identities": {
            "dataset_view_manifest_hash": file_sha256(dataset),
            "dataset_view_manifest_path": str(dataset),
            "split_manifest_hash": file_sha256(split),
            "split_manifest_path": str(split),
            "conditioning_vocabulary_hash": file_sha256(vocabulary),
            "conditioning_vocabulary_path": str(vocabulary),
            "model_config_hash": stable_hash({}),
            "optimizer_config_hash": stable_hash(optimizer),
            "schedule_config_hash": stable_hash(schedule),
            "loss_config_hash": stable_hash(loss),
            "determinism_config_hash": stable_hash(determinism),
        },
        "seeds": list(DEFAULT_SEEDS),
        "training": {
            "max_optimizer_steps": 1_000,
            "micro_batch_size": 2,
            "gradient_accumulation": 4,
            "effective_batch_size": 8,
            "precision": "bf16",
            "sampler_policy": "weighted_replacement_v1",
            "positive_sampling_mass_records": 1_800.0,
        },
        "optimizer": optimizer,
        "schedule": schedule,
        "loss": loss,
        "determinism": determinism,
        "evaluation": evaluation,
        "checkpoint": {"cadence": 500},
        "output_root": str(tmp_path / "runs"),
        "executable": True,
        "launch_authorized": True,
    }


class _AuthorizedTestActivation:
    def __init__(self, campaign: dict, selected_spec: dict | None = None) -> None:
        self.campaign = campaign
        self.selected_spec = selected_spec or campaign

    def to_contract_dict(self) -> dict:
        return {
            "schema_version": "spritelab.training.conditioned-dataset-contract.v2",
            "ready": True,
            "campaign_identity_sha256": self.campaign["campaign_identity"],
            "paths_exposed": False,
        }


def _authorize_test_activation(*_args, **kwargs) -> _AuthorizedTestActivation:
    campaign = kwargs["expected_campaign"]
    return _AuthorizedTestActivation(campaign)


def test_current_project_training_is_blocked_and_launches_nothing() -> None:
    root = Path(__file__).resolve().parents[1]
    backend = FakeComputeBackend()
    result = TrainingService(_context(root, deepcopy(DEFAULT_CONFIG)), backend).start()
    assert result.status == ProductStatus.BLOCKED
    assert result.data["backend_launches"] == 0
    assert "launch" not in backend.calls and "prepare" not in backend.calls
    codes = {item.code for item in result.blockers}
    assert {"dataset_freeze", "training_audit_applicability", "authorization"} <= codes


def test_training_plugin_registration_preserves_simple_cli_contract() -> None:
    plugin = build_plugin()
    assert plugin.plugin_id == "training"
    assert plugin.navigation[0].path == "/training"
    assert plugin.cli_registration.__name__ == "register_training_cli"
    assert plugin.web_router_factory is not None


def test_profile_resolves_to_authoritative_campaign_without_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = deepcopy(DEFAULT_CONFIG)
    freeze = tmp_path / "freeze.json"
    campaign_config = tmp_path / "campaign.json"
    _write_json(freeze, {"status": "complete", "production_authorized": True, "image_count": 1_800})
    _write_json(
        campaign_config,
        {
            "product_profiles": {
                "recommended": {
                    "display": {"display_name": "Recommended baseline"},
                    "campaign": _spec(tmp_path),
                }
            }
        },
    )
    values["dataset"]["freeze_manifest"] = str(freeze)
    values["training"]["dataset_freeze"] = str(freeze)
    values["training"]["campaign_config"] = str(campaign_config)
    values["execution"]["allow_training"] = True
    monkeypatch.setattr(
        "spritelab.product_features.training.plans.build_project_state", lambda _config: _ready_state(tmp_path)
    )
    plan = TrainingPlanResolver().resolve(
        _context(tmp_path, values), TrainingProfile.RECOMMENDED, LocalComputeBackend(), probe_backend=False
    )
    assert plan.ready
    assert plan.dataset_count == 1_800
    assert plan.model_label == "Recommended baseline"
    assert plan.campaign and plan.campaign["plan_status"] == "ready"
    assert not (tmp_path / "runs").exists()


def test_fake_local_execution_uses_existing_campaign_execution_gate(tmp_path: Path) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Recommended baseline",
        1_800,
        True,
        "fake",
        campaign,
        (),
        ComputeEstimate(60, 0, trustworthy=True),
        {"safe": True, "runs": []},
    )

    class Resolver:
        def resolve(self, *args, **kwargs):
            return plan

    class RecordingFakeComputeBackend(FakeComputeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []

        def prepare(self, context, request):
            self.requests.append(request)
            return super().prepare(context, request)

    backend = RecordingFakeComputeBackend()
    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    service = TrainingService(
        _context(tmp_path, values),
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
    )
    result = service.start()
    assert result.status == ProductStatus.RUNNING
    assert backend.calls.count("launch") == 3
    assert backend.calls.count("prepare") == 3
    assert {request.event_path.name for request in backend.requests} == {"events.jsonl"}
    assert {request.environment["CUBLAS_WORKSPACE_CONFIG"] for request in backend.requests} == {":4096:8"}
    assert not list((tmp_path / "runs").rglob("product_events.jsonl"))
    for job_id, job in list(backend._jobs.items()):
        backend._jobs[job_id] = replace(job, status=ComputeStatus.COMPLETE)
    refreshed = service.refresh(campaign["campaign_id"])
    assert refreshed.status == ProductStatus.BLOCKED
    assert any("completion artifacts" in warning.lower() for warning in refreshed.data["warnings"])


def test_validated_receipt_view_identity_reaches_product_checkpoint_catalog(tmp_path: Path) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Recommended baseline",
        1_800,
        True,
        "fake",
        campaign,
        (),
        ComputeEstimate(60, 0, trustworthy=True),
        {"safe": True, "runs": []},
    )

    class Resolver:
        def resolve(self, *args, **kwargs):
            return plan

    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    context = _context(tmp_path, values)
    service = TrainingService(
        context,
        FakeComputeBackend(),
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
    )

    started = service.start()
    assert started.status == ProductStatus.RUNNING
    expected_view = campaign["identities"]["dataset_view_manifest_hash"]
    expected_dataset = campaign["identities"].get("dataset_identity_hash", expected_view)
    assert started.data["training_identity"] == {
        "dataset_identity": expected_dataset,
        "view_identity": expected_view,
        "training_view_identity": expected_view,
    }

    run_id = campaign["campaign_id"]
    run_directory = context.runs_directory / run_id  # type: ignore[operator]
    retained = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert retained["dataset_identity"] == expected_dataset
    assert retained["training_view_identity"] == expected_view
    assert retained["backend_identity"]["view_identity"] == expected_view

    checkpoint = run_directory / "checkpoints" / "checkpoint_step_500.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    session = service.sessions[run_id]
    service._apply(
        session,
        ProductEvent(
            run_id=run_id,
            timestamp="2026-01-01T00:00:00+00:00",
            feature="training",
            stage="seed",
            event_type="checkpoint",
            status=ProductStatus.COMPLETE,
            current=500,
            total=500,
            message="Synthetic verified checkpoint.",
            metrics={
                "checkpoint": str(checkpoint),
                "optimizer_step": 500,
                "sha256": file_sha256(checkpoint),
                "downloaded": True,
                "hash_verified": True,
                "remote_identity_verified": True,
                "identity_verified": True,
            },
        ),
    )
    service._persist_session(session)

    persisted = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert persisted["checkpoints"][0]["dataset_identity"] == expected_dataset
    assert persisted["checkpoints"][0]["view_identity"] == expected_view
    catalog = discover_checkpoint_candidates(
        context.runs_directory,  # type: ignore[arg-type]
        project_root=tmp_path,
        active_dataset_identity=expected_dataset,
        active_view_identity=expected_view,
    )
    assert len(catalog.eligible) == 1
    assert catalog.eligible[0].view_identity == expected_view


def test_cloud_launch_requires_confirmation_before_prepare(tmp_path: Path) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Recommended baseline",
        1_800,
        True,
        "fake",
        campaign,
        (),
        ComputeEstimate(60, 0, trustworthy=True),
    )

    class Resolver:
        def resolve(self, *args, **kwargs):
            return plan

    backend = FakeComputeBackend(is_cloud=True)
    result = TrainingService(_context(tmp_path, deepcopy(DEFAULT_CONFIG)), backend, resolver=Resolver()).start()
    assert result.status == ProductStatus.BLOCKED
    assert result.blockers[0].code == "cloud_confirmation"
    assert "prepare" not in backend.calls and "launch" not in backend.calls


def test_training_page_load_never_probes_prepares_or_launches_cloud(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Recommended baseline",
        1_800,
        True,
        "fake",
        campaign,
        (),
        ComputeEstimate(7 * 3600, 0, trustworthy=True),
    )

    class Resolver:
        def resolve(self, *args, **kwargs):
            return plan

    backend = FakeComputeBackend(is_cloud=True)
    context = _context(tmp_path, deepcopy(DEFAULT_CONFIG))
    service = TrainingService(context, backend, resolver=Resolver())
    app = FastAPI()
    app.include_router(create_router(context, service))
    client = TestClient(app)
    assert client.get("/training").status_code == 200
    state = client.get("/training/api/state").json()
    assert state["data"]["advanced_collapsed"] is True
    assert backend.calls == []


def test_web_start_fails_closed_without_a_conditioned_dataset_contract(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class ReadyLookingService:
        def __init__(self) -> None:
            self.starts = 0

        def status(self, _profile: TrainingProfile) -> ProductResult:
            return ProductResult(
                ProductStatus.READY,
                "Synthetic service plan looks ready.",
                feature="training",
                data={"ready": True, "availability_state": "Training available"},
            )

        def start(self, *args, **kwargs) -> ProductResult:
            del args, kwargs
            self.starts += 1
            return ProductResult(ProductStatus.RUNNING, "Started.", feature="training")

    service = ReadyLookingService()
    app = FastAPI()
    app.include_router(create_router(_context(tmp_path, deepcopy(DEFAULT_CONFIG)), service))  # type: ignore[arg-type]
    client = TestClient(app)

    state = client.get("/training/api/state").json()
    assert state["status"] == "BLOCKED"
    assert state["data"]["ready"] is False
    contract = state["data"]["conditioned_dataset_contract"]
    assert contract["schema_version"] == "spritelab.training.conditioned-dataset-contract.v2"
    assert contract["ready"] is False
    assert contract["blockers"]
    assert contract["paths_exposed"] is False

    response = client.post("/training/api/start", json={"profile": "recommended"})
    assert response.status_code == 409
    assert response.json()["error_code"] == "conditioned_dataset_contract_required"
    assert service.starts == 0


def test_web_preserves_safe_start_for_an_exactly_bound_conditioned_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    freeze = tmp_path / "conditioned-freeze.json"
    _write_json(
        freeze,
        {
            "schema_version": "spritelab.dataset.freeze.conditioned.v5",
            "dataset_version": 5,
            "dataset_kind": "conditioned",
            "requires_semantic_labels": True,
            "status": "complete",
            "production_authorized": True,
            "labeling_audit_sha256": "a" * 64,
            "validation_report_sha256": "b" * 64,
        },
    )
    spec = _spec(tmp_path)
    spec["identities"]["dataset_freeze_hash"] = file_sha256(freeze)
    campaign = tmp_path / "conditioned-campaign.json"
    _write_json(
        campaign,
        {
            "product_profiles": {
                "recommended": {
                    "display": {"display_name": "Conditioned Dataset-v5 campaign"},
                    "campaign": spec,
                }
            }
        },
    )
    values = deepcopy(DEFAULT_CONFIG)
    values["dataset"]["freeze_manifest"] = str(freeze)
    values["training"]["dataset_freeze"] = str(freeze)
    values["training"]["campaign_config"] = str(campaign)
    values["execution"]["allow_dataset_production_freeze"] = True
    values["execution"]["allow_training"] = True

    class ConditionedService:
        def __init__(self) -> None:
            self.starts = 0

        def status(self, _profile: TrainingProfile) -> ProductResult:
            return ProductResult(
                ProductStatus.READY,
                "All independent gates passed.",
                feature="training",
                data={"ready": True, "availability_state": "Training available"},
            )

        def start(self, *args, **kwargs) -> ProductResult:
            del args, kwargs
            self.starts += 1
            return ProductResult(ProductStatus.RUNNING, "Started.", feature="training")

    service = ConditionedService()
    seen: list[TrainingProfile] = []

    def exact_contract(_context, profile, *, custom_spec=None):
        del custom_spec
        seen.append(profile)
        return {
            "schema_version": "spritelab.training.conditioned-dataset-contract.v2",
            "ready": True,
            "profile": profile.value,
            "blockers": [],
            "paths_exposed": False,
        }

    monkeypatch.setattr(
        "spritelab.product_features.training.preparation.conditioned_training_contract",
        exact_contract,
    )
    app = FastAPI()
    app.include_router(create_router(_context(tmp_path, values), service))  # type: ignore[arg-type]
    client = TestClient(app)

    state = client.get("/training/api/state").json()
    assert state["status"] == "READY"
    assert state["data"]["ready"] is True
    assert state["data"]["conditioned_dataset_contract"]["ready"] is True
    assert client.post("/training/api/start", json={"profile": "recommended"}).status_code == 200
    assert service.starts == 1
    assert seen == [TrainingProfile.RECOMMENDED, TrainingProfile.RECOMMENDED]


@pytest.mark.parametrize("profile", [TrainingProfile.QUALITY, TrainingProfile.CUSTOM])
def test_start_authorizes_the_exact_quality_or_custom_campaign_before_backend_calls(
    tmp_path: Path,
    profile: TrainingProfile,
) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        profile,
        "Exact selected campaign",
        2_400,
        True,
        "fake",
        campaign,
        (),
        ComputeEstimate(60, 0, trustworthy=True),
    )

    class Resolver:
        def resolve(self, *args, **kwargs):
            return plan

    requested_custom = {"campaign_id": "exact-custom-request"} if profile is TrainingProfile.CUSTOM else None
    observed: list[dict] = []

    def refuse_after_observing(_context, selected_profile, **kwargs):
        observed.append(
            {
                "profile": selected_profile,
                "custom_spec": kwargs.get("custom_spec"),
                "campaign_identity": kwargs["expected_campaign"]["campaign_identity"],
                "require_audit": kwargs.get("require_audit"),
            }
        )
        raise ConditionedActivationError("synthetic_refusal", "Synthetic exact-binding refusal.")

    backend = FakeComputeBackend()
    service = TrainingService(
        _context(tmp_path, deepcopy(DEFAULT_CONFIG)),
        backend,
        resolver=Resolver(),
        activation_loader=refuse_after_observing,
    )
    result = service.start(profile, custom_spec=requested_custom)

    assert result.status is ProductStatus.BLOCKED
    assert observed == [
        {
            "profile": profile,
            "custom_spec": requested_custom,
            "campaign_identity": campaign["campaign_identity"],
            "require_audit": True,
        }
    ]
    assert "prepare" not in backend.calls and "launch" not in backend.calls


def test_resume_reauthorizes_the_retained_profile_activation_and_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.CUSTOM,
        "Retained custom campaign",
        2_400,
        True,
        "fake",
        campaign,
        (),
        ComputeEstimate(60, 0, trustworthy=True),
    )
    dashboard = DashboardState(
        str(campaign["campaign_id"]),
        "fake",
        status=ProductStatus.PAUSED,
        resume_available=True,
    )
    backend = FakeComputeBackend()
    observed: list[dict] = []

    def refuse_after_observing(_context, selected_profile, **kwargs):
        observed.append(
            {
                "profile": selected_profile,
                "campaign_identity": kwargs["expected_campaign"]["campaign_identity"],
                "require_audit": kwargs.get("require_audit"),
            }
        )
        raise ConditionedActivationError("synthetic_resume_refusal", "Synthetic resume refusal.")

    service = TrainingService(
        _context(tmp_path, deepcopy(DEFAULT_CONFIG)),
        backend,
        resolver=object(),  # type: ignore[arg-type]
        activation_loader=refuse_after_observing,
    )
    session = TrainingSession(str(campaign["campaign_id"]), backend, plan, dashboard=dashboard)
    monkeypatch.setattr(service, "_session", lambda _run_id: session)
    monkeypatch.setattr(service, "_verify_session_migrations", lambda _session: None)

    result = service.resume(str(campaign["campaign_id"]))

    assert result.status is ProductStatus.BLOCKED
    assert observed == [
        {
            "profile": TrainingProfile.CUSTOM,
            "campaign_identity": campaign["campaign_identity"],
            "require_audit": True,
        }
    ]
    assert "prepare" not in backend.calls and "resume" not in backend.calls


def test_cli_remains_python_m_spritelab_v3_train() -> None:
    from spritelab.v3.cli import build_parser

    args = build_parser([build_plugin()]).parse_args(["train", "--dry-run"])
    assert args.command == "train"
    assert args.dry_run is True


def test_training_run_reconstructs_from_canonical_events_after_service_recreation(tmp_path: Path) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Recommended baseline",
        1_800,
        True,
        "fake",
        campaign,
        (),
        ComputeEstimate(60, 0, trustworthy=True),
        {"safe": True, "runs": []},
    )

    class Resolver:
        def resolve(self, *args, **kwargs):
            return plan

    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "reconstruction-campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    context = _context(tmp_path, values)
    backend = FakeComputeBackend()
    first = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
    )
    started = first.start()
    assert started.status == ProductStatus.RUNNING
    run_id = campaign["campaign_id"]
    run_directory = context.runs_directory / run_id  # type: ignore[operator]
    assert (run_directory / "events.jsonl").is_file()
    assert not (run_directory / "product_events.jsonl").exists()
    recreated = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
    )
    assert recreated.latest_run_id() == run_id
    dashboard = recreated.dashboard(run_id)
    assert dashboard.status == ProductStatus.RUNNING
    assert dashboard.data["run_id"] == run_id
    assert recreated.sessions == {}


def test_training_action_routes_dispatch_pause_and_resume_once(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class Service:
        def __init__(self) -> None:
            self.pauses = 0
            self.resumes = 0

        def pause(self, run_id: str) -> ProductResult:
            self.pauses += 1
            return ProductResult(ProductStatus.PAUSED, f"Paused {run_id}.", feature="training")

        def resume(self, run_id: str, *, cloud_confirmation: bool = False) -> ProductResult:
            del cloud_confirmation
            self.resumes += 1
            return ProductResult(ProductStatus.RUNNING, f"Resumed {run_id}.", feature="training")

    service = Service()
    app = FastAPI()
    app.include_router(create_router(_context(tmp_path, deepcopy(DEFAULT_CONFIG)), service))  # type: ignore[arg-type]
    client = TestClient(app)
    assert client.post("/training/api/runs/demo/pause").status_code == 200
    assert client.post("/training/api/runs/demo/resume", json={}).status_code == 200
    assert service.pauses == 1
    assert service.resumes == 1
