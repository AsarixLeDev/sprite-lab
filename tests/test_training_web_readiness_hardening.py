from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from spritelab.product_core import (
    ProductAction,
    ProductEvent,
    ProductResult,
    ProductRun,
    ProductSettingsRepository,
    ProductStatus,
    ProjectContext,
)
from spritelab.product_features.training.config import ComputeSettings, effective_compute_context
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingProfile
from spritelab.product_features.training.service import TrainingService, TrainingSession
from spritelab.product_features.training.web import (
    _is_cloud_backend,
    _launch_challenge,
    _public_action_result,
    _public_dashboard_projection,
    _public_state_projection,
    create_router,
)
from spritelab.product_web.events import EventRepository, IndexedEvent
from spritelab.remote_compute import ComputeEstimate, FakeComputeBackend
from spritelab.v3.config import DEFAULT_CONFIG


class _StateRepository:
    def __init__(self, backend_id: str = "ssh") -> None:
        self.backend_id = backend_id

    def state(self, _run_id: str) -> dict[str, str]:
        return {"backend_id": self.backend_id}


class _TrainingWebService:
    def __init__(self, context: ProjectContext, *, private_path: Path) -> None:
        self.context = context
        self.backend = SimpleNamespace(backend_id="ssh", is_cloud=True)
        self.repository = _StateRepository()
        self.private_path = private_path
        self.starts = 0
        self.resumes = 0
        self.pauses = 0
        self.cancels = 0
        self.challenges = 0
        self.latest_calls = 0
        self.dashboard_calls = 0
        self.private_secret = "TRAINSECRET"

    def latest_run_id(self) -> str:
        self.latest_calls += 1
        return "run-1"

    def status(self, _profile: TrainingProfile) -> ProductResult:
        return ProductResult(
            ProductStatus.READY,
            f"Training is ready. Authorization=Bearer {self.private_secret}",
            feature="training",
            data={
                "ready": True,
                "profile": "recommended",
                "model_label": f"Private config {self.private_path}",
                "dataset": {"images": 2_400, "status": "Ready"},
                "compute": "ssh",
                "estimate": {"duration_seconds": 60},
                "gates": [
                    {
                        "gate_id": "privacy",
                        "passed": True,
                        "message": f"Checked {self.private_path}",
                    }
                ],
                "raw_campaign": {
                    "output_root": str(self.private_path),
                    "argv": ["python", str(self.private_path)],
                },
            },
        )

    def issue_cloud_challenge(self, *, action: str, run_id: str | None, profile: TrainingProfile) -> ProductResult:
        assert action in {"start", "resume"}
        assert (action == "resume") is (run_id is not None)
        assert profile is TrainingProfile.RECOMMENDED
        self.challenges += 1
        return ProductResult(
            ProductStatus.READY,
            "Challenge issued.",
            feature="training",
            data={"challenge_token": f"challenge-{action}-{self.challenges}"},
        )

    def start(self, *_args, cloud_challenge: str | None = None, **_kwargs) -> ProductResult:
        assert (isinstance(cloud_challenge, str) and cloud_challenge.startswith("challenge-start-")) is bool(
            self.backend.is_cloud
        )
        self.starts += 1
        return ProductResult(
            ProductStatus.RUNNING,
            "Started.",
            feature="training",
            run=ProductRun("run-1", "training", "start", ProductStatus.RUNNING, backend_id="ssh"),
            data={
                "execution": {
                    "schema_version": "spritelab_campaign_execution_v1",
                    "campaign_id": "campaign",
                    "launched": [{"run_id": "seed-1", "command": ["python", str(self.private_path)], "returncode": 0}],
                    "resume_report": {"output_root": str(self.private_path)},
                },
                "dashboard": self._private_dashboard(),
                "training_identity": {"dataset_identity": "a" * 64, "view_identity": "b" * 64},
            },
        )

    def dashboard(self, _run_id: str) -> ProductResult:
        self.dashboard_calls += 1
        return ProductResult(
            ProductStatus.PAUSED,
            f"Reconstructed from {self.private_path}; Authorization=Bearer {self.private_secret}",
            feature="training",
            data=self._private_dashboard(),
        )

    def refresh(self, run_id: str) -> ProductResult:
        return self.dashboard(run_id)

    def resume(self, _run_id: str, *, cloud_challenge: str | None = None) -> ProductResult:
        assert (isinstance(cloud_challenge, str) and cloud_challenge.startswith("challenge-resume-")) is bool(
            self.backend.is_cloud
        )
        self.resumes += 1
        return ProductResult(
            ProductStatus.RUNNING,
            "Resumed.",
            feature="training",
            data={
                "execution": {
                    "campaign_id": "campaign",
                    "launched": [{"run_id": "seed-1", "command": [str(self.private_path)]}],
                },
                "unsafe_resume_available": False,
            },
        )

    def pause(self, _run_id: str) -> ProductResult:
        self.pauses += 1
        return ProductResult(ProductStatus.PAUSED, "Paused.", feature="training")

    def cancel(self, _run_id: str) -> ProductResult:
        self.cancels += 1
        return ProductResult(ProductStatus.RUNNING, "Cancelled.", feature="training")

    def _private_dashboard(self) -> dict[str, object]:
        checkpoint = self.private_path / "checkpoints" / "checkpoint.pt"
        checkpoint_row = {
            "checkpoint": str(checkpoint),
            "seed": 1,
            "optimizer_step": 500,
            "sha256": "c" * 64,
            "backend_id": "ssh",
            "remote": True,
            "downloaded": True,
            "hash_verified": True,
            "remote_identity_verified": True,
            "safe_resume": True,
            "synchronization": "downloaded and verified",
            "verification": "verified",
        }
        return {
            "run_id": "run-1",
            "backend_id": "ssh",
            "status": "PAUSED",
            "campaign_progress": {"current": 1, "total": 3},
            "seeds": [{"seed": 1, "status": "PAUSED", "optimizer_step": 500}],
            "checkpoints": [checkpoint_row],
            "latest_verified_checkpoint": checkpoint_row,
            "last_safe_resume_point": checkpoint_row,
            "pause_available": False,
            "resume_available": True,
            "cancel_available": True,
            "logs": [f"Loaded config {self.private_path}; Authorization=Bearer {self.private_secret}"],
            "warnings": [f"Output root {self.private_path}; api_key={self.private_secret}"],
            "previews": [{"output_path": str(self.private_path / "preview.png"), "generation_seed": 2}],
            "argv": ["python", str(self.private_path)],
        }


def _context(tmp_path: Path) -> ProjectContext:
    values = deepcopy(DEFAULT_CONFIG)
    return ProjectContext(tmp_path, values, None, tmp_path / "runs")


def _cloud_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, _TrainingWebService, ProductSettingsRepository]:
    context = _context(tmp_path)
    settings = ProductSettingsRepository(context)
    settings.save(
        "compute",
        {
            "type": "ssh",
            "host": "trainer.example",
            "port": 22,
            "username": "trainer",
            "remote_workspace": "/workspace/sprite-lab",
            "credential_reference": f"file:{tmp_path / 'private-key'}",
            "environment_profile": "python3",
            "cloud": True,
        },
    )
    effective, _configured, _version, _saved = effective_compute_context(context)
    service = _TrainingWebService(effective, private_path=tmp_path)
    monkeypatch.setattr(
        "spritelab.product_features.training.web._conditioned_contract",
        lambda *_args, **_kwargs: {
            "schema_version": "spritelab.training.conditioned-dataset-contract.v2",
            "ready": True,
            "profile": "recommended",
            "paths_exposed": False,
            "blockers": [],
        },
    )
    app = FastAPI()
    app.include_router(create_router(context, service))
    return TestClient(app), service, settings


def _binding(client: TestClient) -> dict[str, object]:
    settings = client.get("/training/api/settings").json()
    return {
        "confirm_cloud": True,
        "compute_configuration_version": settings["configuration_version"],
        "backend_identity": settings["backend_identity"],
    }


def _assert_private_path_absent(payload: object, private_path: Path) -> None:
    serialized = json.dumps(payload, sort_keys=True).replace("\\\\", "\\")
    assert str(private_path) not in serialized
    assert private_path.as_posix() not in serialized


def test_training_service_passes_its_activation_loader_to_the_default_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    class Resolver:
        def __init__(self, *, activation_loader):
            captured.append(activation_loader)

    def loader(*_args, **_kwargs):
        return None

    monkeypatch.setattr("spritelab.product_features.training.service.TrainingPlanResolver", Resolver)

    service = TrainingService(_context(tmp_path), FakeComputeBackend(), activation_loader=loader)

    assert captured == [loader]
    assert service.activation_loader is loader


def test_dashboard_reconstruction_returns_the_exact_last_event_cursor(tmp_path: Path) -> None:
    service = TrainingService(_context(tmp_path), FakeComputeBackend())
    service.repository.create_run(
        "run-1",
        feature="training",
        command="training.start",
        status=ProductStatus.RUNNING.value,
        backend_id="fake",
    )
    event_id = service.repository.append(
        ProductEvent(
            run_id="run-1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            feature="training",
            stage="campaign",
            event_type="progress",
            status=ProductStatus.RUNNING,
            current=1,
            total=3,
        )
    )

    result = service.dashboard("run-1")

    assert result.data["event_cursor"] == event_id


def test_service_cloud_start_and_resume_require_the_exact_true_singleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Ready campaign",
        2_400,
        True,
        "fake",
        {},
        (),
        ComputeEstimate(60, 0, trustworthy=True),
    )

    class Resolver:
        def resolve(self, *_args, **_kwargs):
            return plan

    backend = FakeComputeBackend(is_cloud=True)
    service = TrainingService(_context(tmp_path), backend, resolver=Resolver())

    started = service.start(cloud_confirmation="true")  # type: ignore[arg-type]

    assert started.status is ProductStatus.BLOCKED
    assert started.blockers[0].code == "cloud_challenge_required"
    assert backend.calls == []

    dashboard = DashboardState("run-1", "fake")
    dashboard.status = ProductStatus.PAUSED
    dashboard.resume_available = True
    session = TrainingSession("run-1", backend, plan, dashboard=dashboard)
    monkeypatch.setattr(service, "_session", lambda _run_id: session)
    monkeypatch.setattr(service, "_verify_session_migrations", lambda _session: None)

    resumed = service.resume("run-1", cloud_confirmation="true")  # type: ignore[arg-type]

    assert resumed.status is ProductStatus.BLOCKED
    assert resumed.blockers[0].code == "cloud_challenge_required"
    assert backend.calls == []


def test_cloud_start_binding_rejects_non_boolean_stale_and_mismatched_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, settings = _cloud_client(tmp_path, monkeypatch)
    binding = _binding(client)

    invalid = client.post("/training/api/start", json={**binding, "profile": "recommended", "confirm_cloud": "true"})
    assert invalid.status_code == 422
    assert invalid.json()["error_code"] == "cloud_confirmation_invalid"

    stale = client.post(
        "/training/api/start",
        json={**binding, "profile": "recommended", "compute_configuration_version": -1},
    )
    assert stale.status_code == 409
    assert stale.json()["error_code"] == "compute_authorization_stale"

    mismatch = client.post(
        "/training/api/start",
        json={**binding, "profile": "recommended", "backend_identity": "other-provider"},
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["error_code"] == "compute_backend_mismatch"
    assert service.starts == 0

    service.backend.backend_id = "stale-service-backend"
    stale_service = client.post("/training/api/start", json={**binding, "profile": "recommended"})
    assert stale_service.status_code == 409
    assert stale_service.json()["error_code"] == "compute_backend_mismatch"
    service.backend.backend_id = "ssh"

    matching_context = service.context
    stale_values = deepcopy(dict(matching_context.config))
    stale_compute = deepcopy(dict(stale_values["compute"]))
    stale_training = deepcopy(dict(stale_compute["training"]))
    stale_training["host"] = "stale-cache.example"
    stale_compute["training"] = stale_training
    stale_values["compute"] = stale_compute
    service.context = ProjectContext(
        matching_context.project_root,
        stale_values,
        matching_context.config_path,
        matching_context.runs_directory,
    )
    stale_service_configuration = client.post("/training/api/start", json={**binding, "profile": "recommended"})
    assert stale_service_configuration.status_code == 409
    assert stale_service_configuration.json()["error_code"] == "compute_backend_mismatch"
    service.context = matching_context

    challenge_response = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended"},
    )
    assert challenge_response.status_code == 200
    challenge = challenge_response.json()["data"]["challenge"]
    started = client.post(
        "/training/api/start",
        json={**binding, "profile": "recommended", "cloud_challenge": challenge},
    )
    assert started.status_code == 200
    assert service.starts == 1
    _assert_private_path_absent(started.json(), tmp_path)
    assert started.json()["data"]["launch"]["run_ids"] == ["seed-1"]
    assert "command" not in json.dumps(started.json())

    settings.save(
        "compute",
        {
            "type": "ssh",
            "host": "new-trainer.example",
            "port": 22,
            "username": "trainer",
            "remote_workspace": "/workspace/sprite-lab",
            "credential_reference": "ssh-agent",
            "environment_profile": "python3",
            "cloud": True,
        },
    )
    changed_after_confirmation = client.post(
        "/training/api/start",
        json={**binding, "profile": "recommended", "cloud_challenge": challenge},
    )
    assert changed_after_confirmation.status_code == 409
    assert changed_after_confirmation.json()["error_code"] == "compute_authorization_stale"
    assert service.starts == 1


def test_cloud_resume_binds_durable_backend_and_pause_cancel_use_no_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _settings = _cloud_client(tmp_path, monkeypatch)
    binding = _binding(client)

    invalid = client.post("/training/api/runs/run-1/resume", json={**binding, "confirm_cloud": 1})
    assert invalid.status_code == 422
    assert service.resumes == 0

    service.repository.backend_id = "different-backend"
    mismatch = client.post("/training/api/runs/run-1/resume", json=binding)
    assert mismatch.status_code == 409
    assert mismatch.json()["error_code"] == "training_run_backend_mismatch"
    assert service.resumes == 0

    service.repository.backend_id = "ssh"
    challenge_response = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "resume", "run_id": "run-1", "profile": "recommended"},
    )
    assert challenge_response.status_code == 200
    resumed = client.post(
        "/training/api/runs/run-1/resume",
        json={**binding, "cloud_challenge": challenge_response.json()["data"]["challenge"]},
    )
    assert resumed.status_code == 200
    assert service.resumes == 1
    _assert_private_path_absent(resumed.json(), tmp_path)

    assert client.post("/training/api/runs/run-1/pause", json={"confirm_cloud": "truthy"}).status_code == 200
    assert client.post("/training/api/runs/run-1/cancel", json={"confirm_cloud": "truthy"}).status_code == 200
    assert service.pauses == service.cancels == 1


@pytest.mark.parametrize(
    ("endpoint", "counter"),
    [
        ("/training/api/start", "starts"),
        ("/training/api/runs/run-1/resume", "resumes"),
    ],
)
def test_local_start_and_resume_require_authoritative_compute_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    counter: str,
) -> None:
    client, service, settings = _cloud_client(tmp_path, monkeypatch)
    settings.save("compute", {"type": "local"})
    effective, _configured, _version, _saved = effective_compute_context(service.context)
    service.context = effective
    service.backend = SimpleNamespace(backend_id="local", is_cloud=False)
    service.repository.backend_id = "local"
    current = client.get("/training/api/settings").json()
    binding: dict[str, object] = {
        "compute_configuration_version": current["configuration_version"],
        "backend_identity": "local",
    }
    if endpoint.endswith("/start"):
        binding["profile"] = "recommended"

    missing_version = dict(binding)
    missing_version.pop("compute_configuration_version")
    response = client.post(endpoint, json=missing_version)
    assert response.status_code == 409
    assert response.json()["error_code"] == "compute_authorization_stale"

    response = client.post(endpoint, json={**binding, "backend_identity": "ssh"})
    assert response.status_code == 409
    assert response.json()["error_code"] == "compute_backend_mismatch"

    service.backend.backend_id = "stale-local-service"
    response = client.post(endpoint, json=binding)
    assert response.status_code == 409
    assert response.json()["error_code"] == "compute_backend_mismatch"
    service.backend.backend_id = "local"

    matching_context = service.context
    stale_values = deepcopy(dict(matching_context.config))
    stale_compute = deepcopy(dict(stale_values["compute"]))
    stale_training = deepcopy(dict(stale_compute["training"]))
    stale_training["device_policy"] = "cpu"
    stale_compute["training"] = stale_training
    stale_values["compute"] = stale_compute
    service.context = ProjectContext(
        matching_context.project_root,
        stale_values,
        matching_context.config_path,
        matching_context.runs_directory,
    )
    response = client.post(endpoint, json=binding)
    assert response.status_code == 409
    assert response.json()["error_code"] == "compute_backend_mismatch"
    service.context = matching_context

    response = client.post(endpoint, json=binding)
    assert response.status_code == 200
    assert getattr(service, counter) == 1


def test_training_api_and_page_are_pathless_and_page_performs_zero_event_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _settings = _cloud_client(tmp_path, monkeypatch)
    app = client.app

    def renderer(_request, _plugin_id, _template, values):
        return HTMLResponse(
            json.dumps(
                {
                    "training_run_id": values["training_run_id"],
                    "training_dashboard": values["training_dashboard"],
                    "compute_settings": values["compute_settings"],
                    "compute_backend_identity": values["compute_backend_identity"],
                },
                sort_keys=True,
            )
        )

    app.state.spritelab_render_plugin_template = renderer
    page = client.get("/training")
    assert page.status_code == 200
    _assert_private_path_absent(page.text, tmp_path)
    assert service.latest_calls == 1
    assert service.dashboard_calls == 0
    page_payload = json.loads(page.text)
    assert page_payload["training_dashboard"] is None
    assert page_payload["compute_settings"]["remote_workspace"] == "/workspace/sprite-lab"

    settings = client.get("/training/api/settings")
    assert settings.status_code == 200
    _assert_private_path_absent(settings.json(), tmp_path)
    assert settings.json()["configuration"]["remote_workspace"] == "/workspace/sprite-lab"
    assert settings.json()["configuration"]["credential_reference"] is None
    assert settings.json()["configuration"]["credential_reference_configured"] is True

    state = client.get("/training/api/state")
    assert state.status_code == 200
    _assert_private_path_absent(state.json(), tmp_path)
    assert service.private_secret not in state.text
    assert "[redacted]" in state.text
    assert "raw_campaign" not in state.json()["data"]

    dashboard = client.get("/training/api/runs/run-1")
    assert dashboard.status_code == 200
    _assert_private_path_absent(dashboard.json(), tmp_path)
    assert service.private_secret not in dashboard.text
    assert "[redacted]" in dashboard.text
    assert service.dashboard_calls == 1
    public_checkpoint = dashboard.json()["data"]["latest_verified_checkpoint"]
    assert "checkpoint" not in public_checkpoint
    assert public_checkpoint["checkpoint_label"] == "Checkpoint at step 500"
    assert dashboard.json()["data"]["previews"] == [{"generation_seed": 2}]
    assert dashboard.json()["data"]["cancel_available"] is True


@pytest.mark.parametrize("cloud", ["false", "true", 0, 1, None, [], {}])
def test_compute_settings_reject_non_boolean_cloud_classification(cloud: object) -> None:
    with pytest.raises(ValueError, match="JSON boolean"):
        ComputeSettings.from_mapping({"type": "local", "cloud": cloud})


@pytest.mark.parametrize(
    "backend_id",
    [
        "file:///C:/" + "Users/private/backend",
        "C:\\" + r"Users\private\backend",
        "provider Authorization=Bearer PRIVATESECRET",
        "provider/child",
    ],
)
def test_compute_settings_reject_non_identifier_backend_ids(backend_id: str) -> None:
    with pytest.raises(ValueError, match=r"Backend ID|credentials"):
        ComputeSettings.from_mapping({"type": "other", "backend_id": backend_id})

    configured = ComputeSettings.from_mapping({"type": "other", "backend_id": "provider-x"})
    assert configured.backend_id == "provider-x"


@pytest.mark.parametrize("service_claim", [0, 1, None, "false", "true", [], {}])
def test_malformed_backend_cloud_claim_cannot_suppress_launch_challenge(service_claim: object) -> None:
    configured = ComputeSettings.from_mapping({"type": "local"})
    service = SimpleNamespace(backend=SimpleNamespace(is_cloud=service_claim))

    assert _is_cloud_backend(configured, service) is True
    challenge, error = _launch_challenge({}, configured, service)

    assert challenge is None
    assert error is not None
    assert error.status_code == 422


def test_exact_false_backend_cloud_claim_preserves_local_behavior() -> None:
    configured = ComputeSettings.from_mapping({"type": "local"})
    service = SimpleNamespace(backend=SimpleNamespace(is_cloud=False))

    assert _is_cloud_backend(configured, service) is False
    assert _launch_challenge({}, configured, service) == (None, None)


def test_non_boolean_conditioned_readiness_cannot_enable_state_or_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _settings = _cloud_client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "spritelab.product_features.training.web._conditioned_contract",
        lambda *_args, **_kwargs: {
            "schema_version": "spritelab.training.conditioned-dataset-contract.v2",
            "ready": "false",
            "paths_exposed": "false",
            "blockers": [],
        },
    )

    state = client.get("/training/api/state")
    assert state.status_code == 200
    assert state.json()["status"] == ProductStatus.BLOCKED.value
    assert state.json()["data"]["ready"] is False

    binding = _binding(client)
    challenge_response = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended"},
    )
    assert challenge_response.status_code == 200
    response = client.post(
        "/training/api/start",
        json={
            **binding,
            "profile": "recommended",
            "cloud_challenge": challenge_response.json()["data"]["challenge"],
        },
    )
    assert response.status_code == 409
    assert response.json()["error_code"] == "conditioned_dataset_contract_required"
    assert service.starts == 0


def test_training_public_projections_recursively_redact_and_fail_closed(tmp_path: Path) -> None:
    private_uri = f"file:///{(tmp_path / 'private' / 'checkpoint.pt').as_posix()}"
    secret = "TRAININGPROJECTIONSECRET"
    hostile_text = f"{private_uri} Authorization=Bearer {secret}"
    checkpoint = {
        "optimizer_step": 500,
        "backend_id": {"uri": private_uri, "authorization": f"Bearer {secret}"},
        "remote": "false",
        "downloaded": 1,
        "hash_verified": {"value": True},
        "safe_resume": [True],
        "synchronization": {
            "nested": [
                {
                    "uri": private_uri,
                    "api_key": secret,
                    "privateKey": f"{secret}-PRIVATE-KEY",
                    "accessKey": f"{secret}-ACCESS-KEY",
                    "awsAccessKeyId": f"{secret}-AWS-ACCESS-KEY-ID",
                    "privateKeyPem": f"{secret}-PRIVATE-KEY-PEM",
                    "apiKeyId": f"{secret}-API-KEY-ID",
                    "clientSecretId": f"{secret}-CLIENT-SECRET-ID",
                    "Private Key": f"{secret}-SPACED-PRIVATE-KEY",
                    "Access-Key": f"{secret}-HYPHEN-ACCESS-KEY",
                    "note": (
                        f"awsAccessKeyId={secret}-TEXT-AWS "
                        f"privateKeyPem={secret}-TEXT-PEM Private Key: {secret}-TEXT-PRIVATE"
                    ),
                    "ready": "false",
                    "eligible": "false",
                    "profile_eligible": 1,
                    "validated": "true",
                    "completion_validated": 0,
                    "release_validated": "false",
                    "promotion_evidence": "false",
                    "benchmark_evidence": 1,
                    "narrative_evidence": "recorded",
                    "publicKey": "documented",
                    "accessibilityKey": "keyboard-shortcut",
                    "secretary": "available",
                    "tokenizer": "bpe",
                }
            ]
        },
        "verification": {"message": hostile_text},
    }
    dashboard = _public_dashboard_projection(
        {
            "run_id": {"uri": private_uri},
            "backend_id": hostile_text,
            "status": {"authorization": secret},
            "campaign_progress": {"current": {"path": str(tmp_path)}, "total": 3},
            "seeds": [{"seed": 1, "status": {"uri": private_uri}, "optimizer_step": 500}],
            "checkpoints": [checkpoint],
            "latest_verified_checkpoint": checkpoint,
            "pause_available": "false",
            "resume_available": 1,
            "cancel_available": {"value": True},
            "remote_resource_uncertain": "false",
            "may_accrue_cost": 0,
            "previews": [
                {
                    "generation_seed": 2,
                    "parameters": {
                        "uri": private_uri,
                        "credential": secret,
                        "ready": "false",
                    },
                }
            ],
        },
        tmp_path,
    )
    state = _public_state_projection(
        {
            "schema_version": "test",
            "status": "READY",
            "action": {
                "action_id": "start",
                "feature": "training",
                "title": "Start",
                "requires_confirmation": "false",
            },
            "data": {
                "ready": "false",
                "advanced_collapsed": 0,
                "estimate": {"trustworthy": "false", "message": hostile_text},
                "gates": [{"gate_id": "privacy", "passed": "false", "message": hostile_text}],
                "resume": {
                    "safe": "false",
                    "runs": [{"run_id": "run-1", "next_action": {"uri": private_uri}}],
                },
                "conditioned_dataset_contract": {
                    "ready": "false",
                    "paths_exposed": 1,
                    "audit_status": {
                        "nested": {
                            "uri": private_uri,
                            "authorization": f"Bearer {secret}",
                            "ready": "false",
                        }
                    },
                    "blockers": [{"code": {"uri": private_uri}, "message": hostile_text}],
                },
            },
        },
        tmp_path,
    )

    serialized = json.dumps({"dashboard": dashboard, "state": state}, sort_keys=True)
    _assert_private_path_absent({"dashboard": dashboard, "state": state}, tmp_path)
    assert secret not in serialized
    assert "Authorization=Bearer" not in serialized
    assert dashboard["run_id"] is None
    assert dashboard["status"] is None
    assert dashboard["pause_available"] is False
    assert dashboard["resume_available"] is False
    assert dashboard["cancel_available"] is False
    assert dashboard["remote_resource_uncertain"] is False
    assert dashboard["may_accrue_cost"] is False
    public_checkpoint = dashboard["latest_verified_checkpoint"]
    assert "backend_id" not in public_checkpoint
    assert public_checkpoint["remote"] is False
    assert public_checkpoint["downloaded"] is False
    assert public_checkpoint["hash_verified"] is False
    assert public_checkpoint["safe_resume"] is False
    public_synchronization = public_checkpoint["synchronization"]["nested"][0]
    assert public_synchronization["ready"] is False
    assert public_synchronization["eligible"] is False
    assert public_synchronization["profile_eligible"] is False
    assert public_synchronization["validated"] is False
    assert public_synchronization["completion_validated"] is False
    assert public_synchronization["release_validated"] is False
    assert public_synchronization["promotion_evidence"] is False
    assert public_synchronization["benchmark_evidence"] is False
    assert public_synchronization["narrative_evidence"] == "recorded"
    assert public_synchronization["publicKey"] == "documented"
    assert public_synchronization["accessibilityKey"] == "keyboard-shortcut"
    assert public_synchronization["secretary"] == "available"
    assert public_synchronization["tokenizer"] == "bpe"
    assert public_synchronization["note"] == (
        "awsAccessKeyId=[redacted] privateKeyPem=[redacted] Private Key: [redacted]"
    )
    for private_key in (
        "api_key",
        "privateKey",
        "accessKey",
        "awsAccessKeyId",
        "privateKeyPem",
        "apiKeyId",
        "clientSecretId",
        "Private Key",
        "Access-Key",
    ):
        assert private_key not in public_synchronization
    assert dashboard["previews"][0]["parameters"]["ready"] is False
    assert "credential" not in dashboard["previews"][0]["parameters"]
    public_state = state["data"]
    assert state["action"]["requires_confirmation"] is False
    assert public_state["ready"] is False
    assert public_state["advanced_collapsed"] is True
    assert public_state["estimate"]["trustworthy"] is False
    assert public_state["gates"][0]["passed"] is False
    assert public_state["resume"]["safe"] is False
    contract = public_state["conditioned_dataset_contract"]
    assert contract["ready"] is False
    assert contract["paths_exposed"] is False
    assert contract["audit_status"]["nested"]["ready"] is False
    assert "authorization" not in contract["audit_status"]["nested"]
    assert contract["blockers"][0]["code"] is None


def test_public_event_metrics_expose_only_exact_boolean_flags() -> None:
    event = ProductEvent(
        run_id="run-1",
        timestamp="2026-07-18T00:00:00+00:00",
        feature="training",
        stage="campaign",
        event_type="progress",
        status=ProductStatus.RUNNING,
        metrics={
            "pause_available": "false",
            "resume_available": 1,
            "cancel_available": False,
            "may_accrue_cost": True,
            "eligible": "false",
            "profile_eligible": 1,
            "validated": "true",
            "completion_validated": 0,
            "release_validated": "false",
            "promotion_evidence": "false",
            "benchmark_evidence": 1,
            "narrative_evidence": "recorded",
            "hostile_nested": {
                "eligible": "true",
                "completion_validated": "true",
                "promotion_evidence": 1,
            },
        },
    )

    metrics = IndexedEvent(1, event).public_dict()["metrics"]

    assert metrics == {
        "pause_available": False,
        "resume_available": False,
        "cancel_available": False,
        "may_accrue_cost": True,
        "eligible": False,
        "profile_eligible": False,
        "validated": False,
        "completion_validated": False,
        "release_validated": False,
        "promotion_evidence": False,
        "benchmark_evidence": False,
        "narrative_evidence": "recorded",
    }


def test_generic_training_action_result_rebuilds_exact_public_contract(tmp_path: Path) -> None:
    secret = "ACTIONPROJECTIONSECRET"
    private_uri = f"file:///{(tmp_path / 'private' / 'run').as_posix()}"
    result = ProductResult(
        ProductStatus.RUNNING,
        f"Started {private_uri} Authorization=Bearer {secret}",
        feature=f"training Authorization=Bearer {secret}",
        action=ProductAction(
            "resume",
            "training",
            "Resume",
            requires_confirmation="false",  # type: ignore[arg-type]
        ),
        run=ProductRun(
            "run-1",
            "training",
            "resume",
            ProductStatus.RUNNING,
            backend_id=f"{private_uri} Authorization=Bearer {secret}",
        ),
    )

    public = _public_action_result(result, tmp_path).to_dict()
    serialized = json.dumps(public, sort_keys=True)

    assert public["action"]["requires_confirmation"] is False
    assert secret not in serialized
    _assert_private_path_absent(public, tmp_path)


def test_public_event_snapshot_resumable_flag_requires_exact_boolean(tmp_path: Path) -> None:
    repository = EventRepository(tmp_path / "runs")
    event = ProductEvent(
        run_id="run-1",
        timestamp="2026-07-18T00:00:00+00:00",
        feature="training",
        stage="campaign",
        event_type="progress",
        status=ProductStatus.RUNNING,
    )
    repository.initialize_run(
        "run-1",
        feature="training",
        command="training",
        command_payload={},
        planned_event=event,
        resumable=False,
    )

    repository.update_state("run-1", resumable="false")
    assert repository.snapshot("run-1").resumable is False

    repository.update_state("run-1", resumable=True)
    assert repository.snapshot("run-1").resumable is True
