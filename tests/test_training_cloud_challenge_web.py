from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spritelab.product_core import (
    ProductResult,
    ProductRun,
    ProductSettingsRepository,
    ProductStatus,
    ProjectContext,
)
from spritelab.product_features.training.config import effective_compute_context
from spritelab.product_features.training.models import TrainingProfile
from spritelab.product_features.training.web import create_router
from spritelab.v3.config import DEFAULT_CONFIG


class _RunRepository:
    def state(self, run_id: str) -> dict[str, str]:
        return {"backend_id": "ssh"} if run_id == "run-1" else {}


class _ChallengeService:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.backend = SimpleNamespace(backend_id="ssh", is_cloud=True)
        self.repository = _RunRepository()
        self.challenge_calls: list[dict[str, Any]] = []
        self.start_tokens: list[str | None] = []
        self.resume_tokens: list[tuple[str, str | None]] = []
        self.partial_start = False

    def issue_cloud_challenge(
        self,
        *,
        action: str,
        run_id: str | None,
        profile: TrainingProfile,
    ) -> ProductResult:
        self.challenge_calls.append({"action": action, "run_id": run_id, "profile": profile})
        return ProductResult(
            ProductStatus.READY,
            "issued",
            feature="training",
            data={
                "challenge_token": f"cloud-opaque-{action}",
                "challenge_id": "private-durable-identity",
                "bindings": {"remote_identity": "private-remote-identity"},
                "expires_at": "2099-01-01T00:00:00Z",
            },
        )

    def start(
        self,
        profile: TrainingProfile,
        *,
        custom_spec: object = None,
        cloud_challenge: str | None = None,
    ) -> ProductResult:
        assert profile is TrainingProfile.RECOMMENDED
        assert custom_spec is None
        self.start_tokens.append(cloud_challenge)
        if self.partial_start:
            private_remote = f"private-remote-at-{self.context.project_root}"
            seed_outcomes = [
                {"run_id": "seed-1", "seed": 1, "status": "LAUNCHED", "stage": "launched"},
                {"run_id": "seed-2", "seed": 2, "status": "BLOCKED", "stage": "upload"},
            ]
            job_outcomes = [
                {
                    "job_id": "job-1",
                    "run_id": "seed-1",
                    "status": "RUNNING",
                    "stage": "poll",
                    "remote_identity": private_remote,
                    "may_accrue_cost": True,
                    "resource_shutdown_verified": False,
                }
            ]
            return ProductResult(
                ProductStatus.BLOCKED,
                f"Partial launch blocked at {self.context.project_root}.",
                feature="training",
                data={
                    "run_id": "run-1",
                    "backend_launches": 1,
                    "cancel_available": True,
                    "seed_outcomes": seed_outcomes,
                    "job_outcomes": job_outcomes,
                    "dashboard": {
                        "run_id": "run-1",
                        "backend_id": "ssh",
                        "status": "BLOCKED",
                        "cancel_available": True,
                        "seed_outcomes": seed_outcomes,
                        "job_outcomes": job_outcomes,
                        "remote_resource_uncertain": True,
                        "may_accrue_cost": True,
                    },
                },
            )
        return ProductResult(
            ProductStatus.RUNNING,
            "started",
            feature="training",
            run=ProductRun("run-1", "training", "start", ProductStatus.RUNNING, backend_id="ssh"),
            data={"dashboard": {"run_id": "run-1", "backend_id": "ssh", "status": "RUNNING"}},
        )

    def resume(self, run_id: str, *, cloud_challenge: str | None = None) -> ProductResult:
        self.resume_tokens.append((run_id, cloud_challenge))
        return ProductResult(ProductStatus.RUNNING, "resumed", feature="training")


def _client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, _ChallengeService, dict[str, object]]:
    context = ProjectContext(tmp_path, deepcopy(DEFAULT_CONFIG), None, tmp_path / "runs")
    settings = ProductSettingsRepository(context)
    settings.save(
        "compute",
        {
            "type": "ssh",
            "host": "trainer.example",
            "port": 22,
            "username": "trainer",
            "remote_workspace": "/workspace/sprite-lab",
            "credential_reference": "ssh-agent",
            "environment_profile": "python3",
            "cloud": True,
            "run_profile": "recommended",
        },
    )
    effective, _configured, _version, _saved = effective_compute_context(context)
    service = _ChallengeService(effective)
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
    client = TestClient(app)
    public_settings = client.get("/training/api/settings").json()
    binding: dict[str, object] = {
        "confirm_cloud": True,
        "compute_configuration_version": public_settings["configuration_version"],
        "backend_identity": public_settings["backend_identity"],
    }
    return client, service, binding


@pytest.mark.parametrize("confirmation", [None, False, "true", 1])
def test_challenge_requires_the_exact_fresh_boolean_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confirmation: object,
) -> None:
    client, service, binding = _client(tmp_path, monkeypatch)
    payload = {**binding, "action": "start", "profile": "recommended"}
    if confirmation is None:
        payload.pop("confirm_cloud")
    else:
        payload["confirm_cloud"] = confirmation

    response = client.post("/training/api/cloud-challenge", json=payload)

    assert response.status_code == 422
    assert response.json()["error_code"] in {"cloud_confirmation_invalid", "cloud_confirmation_required"}
    assert service.challenge_calls == []


def test_challenge_is_bound_to_saved_compute_and_projects_only_the_opaque_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, binding = _client(tmp_path, monkeypatch)

    stale = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended", "compute_configuration_version": -1},
    )
    mismatched = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended", "backend_identity": "other"},
    )
    ineligible = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "quality"},
    )
    issued = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended"},
    )

    assert stale.status_code == 409
    assert stale.json()["error_code"] == "compute_authorization_stale"
    assert mismatched.status_code == 409
    assert mismatched.json()["error_code"] == "compute_backend_mismatch"
    assert ineligible.status_code == 422
    assert ineligible.json()["error_code"] == "conditioned_profile_ineligible"
    assert issued.status_code == 200
    assert issued.headers["cache-control"] == "no-store"
    assert issued.json() == {
        "status": "READY",
        "message": "Fresh cloud authorization is ready for this exact action.",
        "data": {"challenge": "cloud-opaque-start"},
    }
    assert service.challenge_calls == [{"action": "start", "run_id": None, "profile": TrainingProfile.RECOMMENDED}]


def test_partial_start_error_projects_durable_outcomes_and_cancel_without_private_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, binding = _client(tmp_path, monkeypatch)
    service.partial_start = True
    issued = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended"},
    )
    response = client.post(
        "/training/api/start",
        json={
            **binding,
            "profile": "recommended",
            "cloud_challenge": issued.json()["data"]["challenge"],
        },
    )

    assert response.status_code == 409
    payload = response.json()
    assert payload["error_code"] == "training_launch_blocked"
    assert payload["details"]["run_id"] == "run-1"
    assert payload["details"]["cancel_available"] is True
    assert [item["status"] for item in payload["details"]["seed_outcomes"]] == ["LAUNCHED", "BLOCKED"]
    assert payload["details"]["dashboard"]["cancel_available"] is True
    serialized = str(payload)
    assert str(tmp_path) not in serialized
    assert "remote_identity" not in serialized


@pytest.mark.parametrize(
    "payload_update",
    [
        {"action": ["start"]},
        {"action": "resume", "run_id": " "},
        {"action": "start", "custom": {}},
    ],
)
def test_challenge_rejects_noncanonical_actions_and_nonproduction_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_update: dict[str, object],
) -> None:
    client, service, binding = _client(tmp_path, monkeypatch)

    response = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended", **payload_update},
    )

    assert response.status_code == 422
    assert service.challenge_calls == []


def test_start_and_resume_reject_the_legacy_boolean_without_the_one_use_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, binding = _client(tmp_path, monkeypatch)

    start = client.post("/training/api/start", json={**binding, "profile": "recommended"})
    resume = client.post("/training/api/runs/run-1/resume", json=binding)

    assert start.status_code == 422
    assert start.json()["error_code"] == "cloud_challenge_required"
    assert resume.status_code == 422
    assert resume.json()["error_code"] == "cloud_challenge_required"
    assert service.start_tokens == []
    assert service.resume_tokens == []


def test_start_and_resume_forward_only_a_fresh_challenge_from_the_dedicated_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, binding = _client(tmp_path, monkeypatch)

    start_challenge = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "start", "profile": "recommended"},
    ).json()["data"]["challenge"]
    started = client.post(
        "/training/api/start",
        json={**binding, "profile": "recommended", "cloud_challenge": start_challenge},
    )
    resume_challenge = client.post(
        "/training/api/cloud-challenge",
        json={**binding, "action": "resume", "run_id": "run-1", "profile": "recommended"},
    ).json()["data"]["challenge"]
    resumed = client.post(
        "/training/api/runs/run-1/resume",
        json={**binding, "cloud_challenge": resume_challenge},
    )

    assert started.status_code == 200
    assert resumed.status_code == 200
    assert service.start_tokens == ["cloud-opaque-start"]
    assert service.resume_tokens == [("run-1", "cloud-opaque-resume")]
    assert service.challenge_calls == [
        {"action": "start", "run_id": None, "profile": TrainingProfile.RECOMMENDED},
        {"action": "resume", "run_id": "run-1", "profile": TrainingProfile.RECOMMENDED},
    ]


def test_template_exposes_only_the_recommended_production_profile() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "src/spritelab/product_features/training/templates/training.html"
    ).read_text(encoding="utf-8")

    assert template.count('<option value="recommended"') == 1
    for ineligible in ("fast_preview", "quality", "custom"):
        assert f'<option value="{ineligible}"' not in template
    assert "short-lived, one-use authorization" in template


def test_browser_requests_and_forwards_a_challenge_without_persisting_it() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "src/spritelab/product_features/training/static/training.js"
    ).read_text(encoding="utf-8")

    assert 'request("/training/api/cloud-challenge"' in javascript
    assert 'body.cloud_challenge=await obtainCloudChallenge("start")' in javascript
    assert 'body.cloud_challenge=await obtainCloudChallenge("resume")' in javascript
    assert "localStorage" not in javascript
    assert "sessionStorage" not in javascript
