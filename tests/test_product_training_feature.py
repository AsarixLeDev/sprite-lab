from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from spritelab.product_core import ProductEvent, ProductResult, ProductStatus, ProjectContext
from spritelab.product_features.evaluation.checkpoints import discover_checkpoint_candidates
from spritelab.product_features.training import build_plugin
from spritelab.product_features.training.action_lock import TrainingActionLock, TrainingActionLockError
from spritelab.product_features.training.activation import ConditionedActivationError
from spritelab.product_features.training.dashboard import CheckpointState, DashboardState
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingProfile
from spritelab.product_features.training.plans import (
    TrainingPlanResolver,
    synthetic_training_path_contract_for_tests,
)
from spritelab.product_features.training.service import (
    TrainingService,
    TrainingSession,
    _evaluation_checkpoint_binding,
    _request_from_state,
)
from spritelab.product_features.training.web import create_router
from spritelab.remote_compute import (
    ComputeBackendError,
    ComputeEstimate,
    ComputeJob,
    ComputeJobRequest,
    ComputeStatus,
    FakeComputeBackend,
    LocalComputeBackend,
    PreparedCompute,
    TrainingLaunchRejected,
    verify_compute_job_request,
)
from spritelab.remote_compute.contracts import verify_launch_authorization_capability
from spritelab.training.campaign import DEFAULT_SEEDS, file_sha256, plan_campaign, stable_hash
from spritelab.training.launch import ValidatedTrainingLaunch
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus
from training_launch_test_utils import validated_launch


def _context(root: Path, values: dict | None = None) -> ProjectContext:
    config = values or ProjectConfig.load(root).values
    return ProjectContext(root, config, root / "spritelab.yaml", root / "runs/v3")


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
        self.ready = True
        self.activation_commit = {
            "committed": True,
            "record_identity": "a" * 64,
            "config_after_sha256": "b" * 64,
            "campaign_identity_sha256": campaign["campaign_identity"],
        }

    def to_contract_dict(self) -> dict:
        return {
            "schema_version": "spritelab.training.conditioned-dataset-contract.v2",
            "ready": True,
            "campaign_identity_sha256": self.campaign["campaign_identity"],
            "activation_commit_record_identity": self.activation_commit["record_identity"],
            "paths_exposed": False,
        }


def _authorize_test_activation(*_args, **kwargs) -> _AuthorizedTestActivation:
    campaign = kwargs["expected_campaign"]
    return _AuthorizedTestActivation(campaign)


class _TestTrainingAuditSnapshot:
    status = AuditStatus.PASS
    launch_authorization_evidence_sha256 = "9" * 64

    def __init__(self) -> None:
        self.active = False
        self.stale = False
        self.verify_calls = 0

    def __enter__(self) -> _TestTrainingAuditSnapshot:
        self.active = True
        self.verify_unchanged()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc, traceback
        if exc_type is None:
            self.verify_unchanged()
        self.active = False

    def verify_unchanged(self) -> None:
        self.verify_calls += 1
        if not self.active or self.stale:
            raise ValueError("synthetic retained training audit changed")


def _open_test_training_audit_snapshot(*_args) -> _TestTrainingAuditSnapshot:
    return _TestTrainingAuditSnapshot()


def _execute_test_campaign_directly(campaign, **kwargs):
    launched = []
    for run in campaign["expected_runs"]:
        evidence = str(kwargs["launch_authorization_evidence_sha256"])
        identities = campaign["identities"]
        view_identity = str(identities["dataset_view_manifest_hash"])
        validated = ValidatedTrainingLaunch(
            receipt=SimpleNamespace(
                launch_authorization_evidence_sha256=evidence,
                campaign_identity_sha256=str(campaign["campaign_identity"]),
                execution_spec_sha256=stable_hash({"run_identity": run["run_identity"]}),
                output_root_identity=stable_hash({"output_root": run["output_root"]}),
                dataset_identity=str(identities.get("dataset_identity_hash", view_identity)),
                view_identity=view_identity,
            ),
            validator_context=SimpleNamespace(launch_authorization_evidence_sha256=evidence),
            campaign=campaign,
            run=run,
            argv=("python", "-m", "spritelab", "train", str(run["run_id"])),
            environment=dict(kwargs["execution_environment"]),
            output_root=Path(str(run["output_root"])),
        )
        result = kwargs["runner"](list(validated.argv), validated_launch=validated)
        launched.append({"run_id": str(run["run_id"]), "returncode": int(result.returncode)})
    return {"launched": launched}


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
    values.update(synthetic_training_path_contract_for_tests(tmp_path))
    activated_campaign = plan_campaign(_spec(tmp_path))
    calls = []

    def load_activation(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            audit_status=AuditStatus.PASS,
            campaign=activated_campaign,
            manifest={"image_count": 1_800},
        )

    plan = TrainingPlanResolver(activation_loader=load_activation).resolve(
        _context(tmp_path, values), TrainingProfile.RECOMMENDED, LocalComputeBackend(), probe_backend=False
    )
    assert plan.ready
    assert calls[0][1]["require_audit"] is False
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
        audit_snapshot_opener=_open_test_training_audit_snapshot,
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


def test_coherent_audit_snapshot_reaches_start_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Coherent audit snapshot",
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

    snapshots: list[_TestTrainingAuditSnapshot] = []

    def open_snapshot(*_args) -> _TestTrainingAuditSnapshot:
        with pytest.raises(TrainingActionLockError):
            with TrainingActionLock(tmp_path, timeout_seconds=0.0):
                pass
        snapshot = _TestTrainingAuditSnapshot()
        snapshots.append(snapshot)
        return snapshot

    class RecordingBackend(FakeComputeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []

        def prepare(self, context, request):
            del context
            assert request.launch_authorization_verifier is snapshots[0]
            verify_launch_authorization_capability(request)
            self.requests.append(request)
            self.calls.append("prepare")
            return PreparedCompute(
                self.backend_id,
                request.idempotency_key,
                "/fake",
                stable_hash({"run": request.run_identity}),
            )

        def launch(self, prepared, request, *, cloud_confirmation=False):
            del cloud_confirmation
            verify_launch_authorization_capability(request)
            self.calls.append("launch")
            job = ComputeJob(
                self.backend_id,
                request.idempotency_key,
                request.run_id,
                ComputeStatus.RUNNING,
                prepared.remote_identity,
            )
            self._jobs[request.idempotency_key] = job
            return job

    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "coherent-campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    backend = RecordingBackend()
    service = TrainingService(
        _context(tmp_path, values),
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=open_snapshot,
    )
    monkeypatch.setattr(
        "spritelab.product_features.training.service.execute_campaign",
        _execute_test_campaign_directly,
    )

    result = service.start()

    assert result.status is ProductStatus.RUNNING, result.to_dict()
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.active is False
    assert snapshot.verify_calls >= 20
    assert len(backend.requests) == 3
    assert {request.launch_authorization_evidence_sha256 for request in backend.requests} == {
        snapshot.launch_authorization_evidence_sha256
    }
    assert {id(request.launch_authorization_verifier) for request in backend.requests} == {id(snapshot)}
    state = service.repository.state(str(campaign["campaign_id"]))
    assert state["active_operation"]["schema_version"] == "spritelab.training.backend-operation.v2"
    assert (
        state["active_operation"]["launch_authorization_evidence_sha256"]
        == snapshot.launch_authorization_evidence_sha256
    )
    assert {row["request"]["launch_authorization_evidence_sha256"] for row in state["jobs"]} == {
        snapshot.launch_authorization_evidence_sha256
    }
    restored = _request_from_state(state["jobs"][0]["request"])
    assert restored is not None
    assert restored.launch_authorization_evidence_sha256 == snapshot.launch_authorization_evidence_sha256
    assert restored.launch_authorization_verifier is None
    with pytest.raises(TrainingLaunchRejected, match="receipt and validator context"):
        verify_compute_job_request(restored, backend_id=backend.backend_id)
    with pytest.raises(TrainingLaunchRejected, match="changed"):
        verify_compute_job_request(backend.requests[0], backend_id=backend.backend_id)


def test_audit_swap_fails_before_backend_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Hostile audit swap",
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

    snapshot = _TestTrainingAuditSnapshot()

    class SwapBeforeAdapterBackend(FakeComputeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_attempts = 0

        def prepare(self, context, request):
            del context
            self.prepare_attempts += 1
            assert request.launch_authorization_verifier is snapshot
            snapshot.stale = True
            verify_launch_authorization_capability(request)
            raise AssertionError("stale capability verification must fail")

    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "hostile-swap-campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    backend = SwapBeforeAdapterBackend()
    service = TrainingService(
        _context(tmp_path, values),
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=lambda *_args: snapshot,
    )
    monkeypatch.setattr(
        "spritelab.product_features.training.service.execute_campaign",
        _execute_test_campaign_directly,
    )

    result = service.start()

    assert result.status is ProductStatus.BLOCKED
    assert [blocker.code for blocker in result.blockers] == ["training_audit_snapshot_stale"]
    assert result.data["backend_launches"] == 0
    assert backend.prepare_attempts == 1
    assert "prepare" not in backend.calls and "launch" not in backend.calls


def test_coherent_audit_snapshot_reaches_resume_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = validated_launch(tmp_path / "validated", "fake")
    expected_evidence = _TestTrainingAuditSnapshot.launch_authorization_evidence_sha256
    continuation = replace(
        initial,
        receipt=replace(
            initial.receipt,
            launch_authorization_evidence_sha256=expected_evidence,
            source_checkpoint_identity="c" * 64,
        ),
        validator_context=replace(
            initial.validator_context,
            resume=True,
            launch_authorization_evidence_sha256=expected_evidence,
        ),
    )
    campaign = dict(continuation.campaign)
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Retained coherent audit snapshot",
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

    snapshots: list[_TestTrainingAuditSnapshot] = []

    def open_snapshot(*_args) -> _TestTrainingAuditSnapshot:
        with pytest.raises(TrainingActionLockError):
            with TrainingActionLock(tmp_path, timeout_seconds=0.0):
                pass
        snapshot = _TestTrainingAuditSnapshot()
        snapshots.append(snapshot)
        return snapshot

    class ResumeRecordingBackend(FakeComputeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.resume_requests = []

        def prepare(self, context, request):
            del context
            assert snapshots[0].active
            assert request.launch_authorization_verifier is snapshots[0]
            return PreparedCompute(self.backend_id, request.idempotency_key, "/fake", "resume-remote")

        def resume(self, prepared, resume, *, cloud_confirmation=False):
            del cloud_confirmation
            assert snapshots[0].active
            assert resume.request.launch_authorization_verifier is snapshots[0]
            self.resume_requests.append(resume.request)
            return ComputeJob(
                self.backend_id,
                resume.request.idempotency_key,
                resume.request.run_id,
                ComputeStatus.RUNNING,
                prepared.remote_identity,
            )

    values = deepcopy(DEFAULT_CONFIG)
    values["training"]["campaign_config"] = str(continuation.validator_context.campaign_config_path)
    backend = ResumeRecordingBackend()
    service = TrainingService(
        _context(tmp_path, values),
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=open_snapshot,
    )
    session_id = "coherent-resume-session"
    dashboard = DashboardState(
        session_id,
        backend.backend_id,
        status=ProductStatus.PAUSED,
        resume_available=True,
    )
    dashboard.checkpoints.append(
        CheckpointState(
            checkpoint=str(tmp_path / "checkpoint.pt"),
            seed=int(continuation.run["seed"]),
            optimizer_step=5,
            sha256="c" * 64,
            backend_id=backend.backend_id,
            remote=False,
            downloaded=True,
            hash_verified=True,
            remote_identity_verified=True,
            safe_resume=True,
            synchronization="verified",
            verification="verified",
        )
    )
    previous = ComputeJobRequest(
        run_id=str(continuation.run["run_id"]),
        command=continuation.argv,
        idempotency_key=str(continuation.run["run_id"]),
        campaign_identity=continuation.receipt.campaign_identity_sha256,
        run_identity=str(continuation.run["run_identity"]),
        local_project_root=tmp_path,
        output_root=continuation.output_root,
        event_path=continuation.output_root / "events.jsonl",
        compute_backend_id=backend.backend_id,
    )
    session = TrainingSession(
        session_id,
        backend,
        plan,
        dashboard=dashboard,
        requests={"previous": previous},
    )
    service.sessions[session_id] = session
    claimed: list[str] = []

    def claim_operation(_session, **kwargs) -> None:
        claimed.append(str(kwargs["launch_authorization_evidence_sha256"]))
        _session.operation_nonce = str(kwargs["operation_nonce"])
        _session.operation_action = str(kwargs["action"])
        _session.launch_authorization_evidence_sha256 = str(kwargs["launch_authorization_evidence_sha256"])

    def execute_once(_campaign, **kwargs):
        assert kwargs["launch_authorization_evidence_sha256"] == expected_evidence
        kwargs["runner"](list(continuation.argv), validated_launch=continuation)
        return {"launched": [str(continuation.run["run_id"])]}

    monkeypatch.setattr(service, "_session", lambda _run_id: session)
    monkeypatch.setattr(service, "_verify_session_migrations", lambda _session: None)
    monkeypatch.setattr(service, "_claim_operation", claim_operation)
    monkeypatch.setattr(service, "_idempotent_backend_call", lambda _session, operation: operation())
    monkeypatch.setattr(service, "_persist_session", lambda _session: None)
    monkeypatch.setattr(service, "_finish_operation", lambda _session, _status: None)
    monkeypatch.setattr(service, "_apply", lambda _session, event: _session.dashboard.apply(event))
    monkeypatch.setattr("spritelab.product_features.training.service.execute_campaign", execute_once)

    result = service.resume(session_id)

    assert result.status is ProductStatus.RUNNING
    assert claimed == [expected_evidence]
    assert len(snapshots) == 1 and snapshots[0].active is False
    assert len(backend.resume_requests) == 1
    request = backend.resume_requests[0]
    assert request.launch_authorization_evidence_sha256 == expected_evidence
    assert request.launch_authorization_verifier is snapshots[0]


def test_authoritative_pre_run_refusal_leaves_campaign_retryable(tmp_path: Path) -> None:
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
    campaign_path = tmp_path / "refused-campaign.json"
    _write_json(campaign_path, {})
    values["training"]["campaign_config"] = str(campaign_path)
    context = _context(tmp_path, values)
    backend = FakeComputeBackend()
    service = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=_open_test_training_audit_snapshot,
    )

    first = service.start()
    second = service.start()

    assert first.status is ProductStatus.BLOCKED
    assert second.status is ProductStatus.BLOCKED
    assert [item.code for item in first.blockers] == ["authoritative_launch"]
    assert [item.code for item in second.blockers] == ["authoritative_launch"]
    assert first.data["backend_launches"] == second.data["backend_launches"] == 0
    assert backend.calls == []
    assert service.repository.state(str(campaign["campaign_id"])) == {}
    assert service.sessions == {}


def test_backend_prepare_attempt_claims_durable_non_retryable_campaign_state(tmp_path: Path) -> None:
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

    class PrepareRefusalBackend(FakeComputeBackend):
        def prepare(self, context, request):
            super().prepare(context, request)
            raise ComputeBackendError("Synthetic backend preparation refusal.")

    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    context = _context(tmp_path, values)
    backend = PrepareRefusalBackend()
    service = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=_open_test_training_audit_snapshot,
    )

    first = service.start()
    retained = service.repository.state(str(campaign["campaign_id"]))
    second = service.start()

    assert first.status is ProductStatus.BLOCKED
    assert first.data["backend_launches"] == 0
    assert backend.calls == ["prepare", "prepare"]
    assert retained["status"] == ProductStatus.BLOCKED.value
    assert retained["conditioned_activation"]["campaign_identity_sha256"] == campaign["campaign_identity"]
    assert second.status is ProductStatus.BLOCKED
    assert [item.code for item in second.blockers] == ["existing_run"]
    assert backend.calls == ["prepare", "prepare"]


@pytest.mark.parametrize(
    ("failure_seam", "retained_stage", "cancel_status", "cleanup_stage"),
    [
        ("upload", "prepared", ProductStatus.COMPLETE, "cleaned"),
        ("launch", "possibly_launched", ProductStatus.BLOCKED, "cleanup_uncertain"),
    ],
)
def test_prepared_resource_is_retained_privately_and_cleanup_survives_recreation(
    tmp_path: Path,
    failure_seam: str,
    retained_stage: str,
    cancel_status: ProductStatus,
    cleanup_stage: str,
) -> None:
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

    class SeamFailureBackend(FakeComputeBackend):
        secret_workspace = str(tmp_path / "private" / "operator-workspace")
        secret_token = "provider-token-must-not-be-persisted"

        def prepare(self, context, request):
            prepared = super().prepare(context, request)
            return PreparedCompute(
                prepared.backend_id,
                prepared.operation_id,
                self.secret_workspace,
                prepared.remote_identity,
                {"provider_token": self.secret_token},
            )

        def upload(self, prepared, artifacts, *, remote_subdirectory="inputs"):
            if failure_seam == "upload":
                self.calls.append("upload")
                raise ComputeBackendError("Synthetic upload failure.")
            return super().upload(prepared, artifacts, remote_subdirectory=remote_subdirectory)

        def launch(self, prepared, request, *, cloud_confirmation=False):
            if failure_seam == "launch":
                self.calls.append("launch")
                raise ComputeBackendError("Synthetic launch failure.")
            return super().launch(prepared, request, cloud_confirmation=cloud_confirmation)

    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    context = _context(tmp_path, values)
    backend = SeamFailureBackend()
    first = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=_open_test_training_audit_snapshot,
    )

    blocked = first.start()
    run_id = str(campaign["campaign_id"])
    retained = first.repository.state(run_id)
    serialized = json.dumps(retained, sort_keys=True)

    assert blocked.status is ProductStatus.BLOCKED
    assert retained["jobs"] == []
    assert retained["prepared_resources"][0]["stage"] == retained_stage
    assert "workspace" not in retained["prepared_resources"][0]["reference"]
    assert SeamFailureBackend.secret_workspace not in serialized
    assert SeamFailureBackend.secret_token not in serialized

    recreated = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=_open_test_training_audit_snapshot,
    )
    cancelled = recreated.cancel(run_id)
    after_cleanup = recreated.repository.state(run_id)

    assert cancelled.status is cancel_status
    assert backend.calls[-1] == "cleanup"
    assert after_cleanup["prepared_resources"][0]["stage"] == cleanup_stage
    if failure_seam == "launch":
        assert cancelled.data["cancel_available"] is True
        assert cancelled.data["resource_state_uncertain"] is True


@pytest.mark.parametrize("fail_first_cancel", [False, True])
def test_blocked_partial_campaign_attempts_every_launched_job_and_persists_safe_uncertainty(
    tmp_path: Path,
    fail_first_cancel: bool,
) -> None:
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

    class PartialFailureBackend(FakeComputeBackend):
        private_failure = f"provider-token at {tmp_path / 'private' / 'operator-workspace'}"

        def __init__(self) -> None:
            super().__init__(is_cloud=True)
            self.upload_attempts = 0
            self.cancel_attempts: list[str] = []

        def upload(self, prepared, artifacts, *, remote_subdirectory="inputs"):
            self.upload_attempts += 1
            if self.upload_attempts in {3, 4}:
                self.calls.append("upload")
                raise ComputeBackendError(self.private_failure)
            return super().upload(prepared, artifacts, remote_subdirectory=remote_subdirectory)

        def cancel(self, job):
            self.cancel_attempts.append(job.job_id)
            if fail_first_cancel and len(self.cancel_attempts) == 1:
                raise ComputeBackendError(self.private_failure)
            return super().cancel(job)

        def poll(self, job):
            result = super().poll(job)
            return replace(
                result,
                metadata={"resource_shutdown_verified": result.status is ComputeStatus.CANCELLED},
            )

        def cleanup(self, prepared):
            return replace(
                super().cleanup(prepared),
                metadata={"resource_shutdown_verified": True},
            )

    values = deepcopy(DEFAULT_CONFIG)
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign)
    values["training"]["campaign_config"] = str(campaign_path)
    context = _context(tmp_path, values)
    backend = PartialFailureBackend()
    first = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=_open_test_training_audit_snapshot,
    )

    challenge_result = first.issue_cloud_challenge(action="start")
    challenge = challenge_result.data["challenge_token"]
    blocked = first.start(cloud_challenge=challenge)
    run_id = str(campaign["campaign_id"])
    retained = first.repository.state(run_id)

    assert blocked.status is ProductStatus.BLOCKED
    assert blocked.data["backend_launches"] == 2
    assert len(retained["jobs"]) == 2
    assert {row["stage"] for row in retained["prepared_resources"]} == {"launched", "prepared"}

    recreated = TrainingService(
        context,
        backend,
        resolver=Resolver(),
        activation_loader=_authorize_test_activation,
        audit_snapshot_opener=_open_test_training_audit_snapshot,
    )
    assert recreated.dashboard(run_id).data["cancel_available"] is True

    cancelled = recreated.cancel(run_id)

    assert len(backend.cancel_attempts) == 2
    assert set(backend.cancel_attempts) == {row["job"]["job_id"] for row in retained["jobs"]}
    assert cancelled.data["cancel_attempt_count"] == 2
    if not fail_first_cancel:
        assert cancelled.status is ProductStatus.COMPLETE
        assert cancelled.data["terminal_status"] == "CANCELLED"
        assert recreated.repository.state(run_id)["status"] == "CANCELLED"
        assert recreated.dashboard(run_id).data["cancel_available"] is False
        return

    assert cancelled.status is ProductStatus.BLOCKED
    assert cancelled.data["cancel_unverified_count"] == 1
    assert cancelled.data["may_accrue_cost"] is True
    durable = recreated.dashboard(run_id).data
    serialized_state = json.dumps(recreated.repository.state(run_id), sort_keys=True)
    serialized_events = json.dumps(
        [indexed.event.to_dict() for indexed in recreated.repository.replay(run_id).events],
        sort_keys=True,
    )
    assert durable["status"] == ProductStatus.BLOCKED.value
    assert durable["cancel_available"] is True
    assert durable["remote_resource_uncertain"] is True
    assert durable["may_accrue_cost"] is True
    assert "terminate" in durable["shutdown_guidance"]
    assert PartialFailureBackend.private_failure not in serialized_state
    assert PartialFailureBackend.private_failure not in serialized_events
    assert "provider-token" not in serialized_state
    assert "provider-token" not in serialized_events


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
        audit_snapshot_opener=_open_test_training_audit_snapshot,
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
            status=ProductStatus.RUNNING,
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
    service._apply(
        session,
        ProductEvent(
            run_id=run_id,
            timestamp="2026-01-01T00:01:00+00:00",
            feature="training",
            stage="campaign",
            event_type="backend_state",
            status=ProductStatus.COMPLETE,
            current=3,
            total=3,
            message="Backend jobs are complete.",
            metrics={
                "completion_validated": True,
                "evaluation_checkpoint_binding": _evaluation_checkpoint_binding(
                    session,
                    service.repository.runs_directory,
                ),
            },
        ),
    )
    service._persist_session(session)

    persisted = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert persisted["checkpoints"][0]["dataset_identity"] == expected_dataset
    assert persisted["checkpoints"][0]["view_identity"] == expected_view
    assert persisted["last_durable_event"]["event_type"] == "backend_state"
    terminal_binding = service.repository.replay(run_id).events[-1].event.metrics["evaluation_checkpoint_binding"]
    assert terminal_binding["dataset_identity"] == expected_dataset
    assert terminal_binding["view_identity"] == expected_view
    assert terminal_binding["checkpoints"][0]["path"] == "checkpoints/checkpoint_step_500.pt"
    assert terminal_binding["checkpoints"][0]["sha256"] == file_sha256(checkpoint)
    catalog = discover_checkpoint_candidates(
        context.runs_directory,  # type: ignore[arg-type]
        project_root=tmp_path,
        active_dataset_identity=expected_dataset,
        active_view_identity=expected_view,
    )
    assert len(catalog.eligible) == 1
    assert catalog.eligible[0].view_identity == expected_view


def test_cloud_launch_requires_a_fresh_challenge_before_prepare(tmp_path: Path) -> None:
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

    probes: list[bool] = []

    class Resolver:
        def resolve(self, *args, **kwargs):
            probes.append(kwargs["probe_backend"])
            return plan

    backend = FakeComputeBackend(is_cloud=True)
    service = TrainingService(_context(tmp_path, deepcopy(DEFAULT_CONFIG)), backend, resolver=Resolver())
    result = service.start()
    legacy_boolean = service.start(cloud_confirmation=True)
    assert result.status == ProductStatus.BLOCKED
    assert result.blockers[0].code == "cloud_challenge_required"
    assert legacy_boolean.status == ProductStatus.BLOCKED
    assert legacy_boolean.blockers[0].code == "cloud_challenge_required"
    assert probes == [False, False]
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

    context = _context(tmp_path, deepcopy(DEFAULT_CONFIG))

    class ReadyLookingService:
        def __init__(self) -> None:
            self.context = context
            self.backend = LocalComputeBackend()
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
    app.include_router(create_router(context, service))  # type: ignore[arg-type]
    client = TestClient(app)

    state = client.get("/training/api/state").json()
    assert state["status"] == "BLOCKED"
    assert state["data"]["ready"] is False
    contract = state["data"]["conditioned_dataset_contract"]
    assert contract["schema_version"] == "spritelab.training.conditioned-dataset-contract.v2"
    assert contract["ready"] is False
    assert contract["blockers"]
    assert contract["paths_exposed"] is False

    settings = client.get("/training/api/settings").json()
    response = client.post(
        "/training/api/start",
        json={
            "profile": "recommended",
            "compute_configuration_version": settings["configuration_version"],
            "backend_identity": settings["backend_identity"],
        },
    )
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

    context = _context(tmp_path, values)

    class ConditionedService:
        def __init__(self) -> None:
            self.context = context
            self.backend = LocalComputeBackend()
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
    app.include_router(create_router(context, service))  # type: ignore[arg-type]
    client = TestClient(app)

    state = client.get("/training/api/state").json()
    assert state["status"] == "READY"
    assert state["data"]["ready"] is True
    assert state["data"]["conditioned_dataset_contract"]["ready"] is True
    settings = client.get("/training/api/settings").json()
    assert (
        client.post(
            "/training/api/start",
            json={
                "profile": "recommended",
                "compute_configuration_version": settings["configuration_version"],
                "backend_identity": settings["backend_identity"],
            },
        ).status_code
        == 200
    )
    assert service.starts == 1
    assert seen == [TrainingProfile.RECOMMENDED, TrainingProfile.RECOMMENDED]


@pytest.mark.parametrize("profile", [TrainingProfile.QUALITY, TrainingProfile.CUSTOM])
def test_start_rejects_nonrecommended_production_profiles_before_any_backend_or_activation_call(
    tmp_path: Path,
    profile: TrainingProfile,
) -> None:
    class Resolver:
        def resolve(self, *args, **kwargs):
            raise AssertionError("A nonrecommended production profile must not be resolved.")

    requested_custom = {"campaign_id": "exact-custom-request"} if profile is TrainingProfile.CUSTOM else None
    observed: list[TrainingProfile] = []

    def observe_activation(_context, selected_profile, **kwargs):
        del kwargs
        observed.append(selected_profile)
        raise AssertionError("A nonrecommended production profile must not activate.")

    backend = FakeComputeBackend()
    service = TrainingService(
        _context(tmp_path, deepcopy(DEFAULT_CONFIG)),
        backend,
        resolver=Resolver(),
        activation_loader=observe_activation,
    )
    result = service.start(profile, custom_spec=requested_custom)

    assert result.status is ProductStatus.BLOCKED
    assert [item.code for item in result.blockers] == ["conditioned_profile_ineligible"]
    assert result.data["backend_launches"] == 0
    assert observed == []
    assert backend.calls == []


def test_resume_reauthorizes_the_retained_profile_activation_and_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = plan_campaign(_spec(tmp_path))
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Retained recommended campaign",
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
            "profile": TrainingProfile.RECOMMENDED,
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
        audit_snapshot_opener=_open_test_training_audit_snapshot,
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
        audit_snapshot_opener=_open_test_training_audit_snapshot,
    )
    assert recreated.latest_run_id() == run_id
    dashboard = recreated.dashboard(run_id)
    assert dashboard.status == ProductStatus.RUNNING
    assert dashboard.data["run_id"] == run_id
    assert recreated.sessions == {}


def test_training_action_routes_dispatch_pause_and_resume_once(tmp_path: Path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    context = _context(tmp_path, deepcopy(DEFAULT_CONFIG))

    class Service:
        def __init__(self) -> None:
            self.context = context
            self.backend = LocalComputeBackend()
            self.pauses = 0
            self.resumes = 0

        def pause(self, run_id: str) -> ProductResult:
            self.pauses += 1
            return ProductResult(ProductStatus.PAUSED, f"Paused {run_id}.", feature="training")

        def resume(self, run_id: str, *, cloud_challenge: str | None = None) -> ProductResult:
            del cloud_challenge
            self.resumes += 1
            return ProductResult(ProductStatus.RUNNING, f"Resumed {run_id}.", feature="training")

    service = Service()
    app = FastAPI()
    app.include_router(create_router(context, service))  # type: ignore[arg-type]
    client = TestClient(app)
    assert client.post("/training/api/runs/demo/pause").status_code == 200
    settings = client.get("/training/api/settings").json()
    assert (
        client.post(
            "/training/api/runs/demo/resume",
            json={
                "compute_configuration_version": settings["configuration_version"],
                "backend_identity": settings["backend_identity"],
            },
        ).status_code
        == 200
    )
    assert service.pauses == 1
    assert service.resumes == 1
