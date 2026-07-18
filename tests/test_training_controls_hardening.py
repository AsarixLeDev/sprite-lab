from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_features.training import cloud_challenge as cloud_challenge_module
from spritelab.product_features.training.action_lock import (
    ACTION_LOCK_FILENAME,
    TrainingActionLock,
    TrainingActionLockError,
)
from spritelab.product_features.training.cloud_challenge import CloudChallengeError, CloudChallengeStore
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_features.training.models import ResolvedTrainingPlan, TrainingProfile
from spritelab.product_features.training.service import (
    TrainingService,
    TrainingSession,
    _job_to_state,
    _operation_record,
    _operation_transition,
    _plan_to_state,
    _prepared_reference_to_state,
    _prepared_to_state,
    _request_to_state,
)
from spritelab.product_features.training.web import _action_response, _public_dashboard_projection
from spritelab.remote_compute import (
    ComputeBackendError,
    ComputeEstimate,
    ComputeJob,
    ComputeJobRequest,
    ComputePoll,
    ComputeStatus,
    FakeComputeBackend,
    OperationResult,
    PreparedCompute,
)


def _context(root: Path) -> ProjectContext:
    return ProjectContext(
        root,
        {"compute": {"training": {"type": "local"}}},
        root / "spritelab.yaml",
        root / "runs",
    )


def _plan(backend_id: str, run_ids: tuple[str, ...]) -> ResolvedTrainingPlan:
    campaign = {
        "campaign_id": "campaign-controls",
        "campaign_identity": "c" * 64,
        "expected_runs": [
            {"run_id": run_id, "seed": index, "expected_checkpoint_steps": [500]}
            for index, run_id in enumerate(run_ids, start=1)
        ],
    }
    return ResolvedTrainingPlan(
        TrainingProfile.RECOMMENDED,
        "Controls test",
        3,
        True,
        backend_id,
        campaign,
        (),
        ComputeEstimate(trustworthy=True),
    )


def _seed_service(
    root: Path,
    backend: FakeComputeBackend,
    *,
    job_count: int = 1,
    status: ProductStatus = ProductStatus.RUNNING,
) -> tuple[TrainingService, list[ComputeJob]]:
    context = _context(root)
    service = TrainingService(context, backend)
    run_ids = tuple(f"seed-{index}" for index in range(1, job_count + 1))
    plan = _plan(backend.backend_id, run_ids)
    dashboard = DashboardState("campaign-controls", backend.backend_id, status=status, cancel_available=True)
    jobs: list[ComputeJob] = []
    prepared: dict[str, PreparedCompute] = {}
    requests: dict[str, ComputeJobRequest] = {}
    for index, run_id in enumerate(run_ids, start=1):
        job_id = f"job-{index}"
        remote_identity = f"remote-{index}"
        job = ComputeJob(
            backend.backend_id,
            job_id,
            run_id,
            ComputeStatus.RUNNING,
            remote_identity,
            may_accrue_cost=backend.is_cloud,
        )
        request = ComputeJobRequest(
            run_id,
            ("python", "train.py"),
            job_id,
            "c" * 64,
            f"{index:064x}",
            root,
            root / "outputs" / run_id,
            compute_backend_id=backend.backend_id,
            execution_spec_identity="d" * 64,
            output_root_identity="e" * 64,
            launch_authorization_evidence_sha256="a" * 64,
        )
        resource = PreparedCompute(backend.backend_id, job_id, "/retained", remote_identity)
        jobs.append(job)
        prepared[job_id] = resource
        requests[job_id] = request
        backend._jobs[job_id] = job
    operation = _operation_record(
        action="start",
        run_id="campaign-controls",
        operation_nonce="operation-" + "1" * 32,
        campaign_identity="c" * 64,
        backend_id=backend.backend_id,
        backend_configuration_identity=service._backend_configuration_identity(),
        activation_commit_record_identity="a" * 64,
        project_config_sha256="b" * 64,
        launch_authorization_evidence_sha256="a" * 64,
        status="RUNNING",
    )
    operation = _operation_transition(operation, "LAUNCHED")
    service.repository.create_run(
        "campaign-controls",
        feature="training",
        command="training.start",
        status=status.value,
        backend_id=backend.backend_id,
        backend_identity={
            "backend_id": backend.backend_id,
            "configuration_identity_sha256": service._backend_configuration_identity(),
            "remote_identities": sorted(item.remote_identity for item in prepared.values()),
        },
        extra={
            "plan": _plan_to_state(plan),
            "jobs": [
                {
                    "job": _job_to_state(job),
                    "prepared": _prepared_to_state(prepared[job.job_id]),
                    "request": _request_to_state(requests[job.job_id]),
                }
                for job in jobs
            ],
            "prepared_resources": [
                {
                    "run_id": job.job_id,
                    "stage": "launched",
                    "reference": _prepared_reference_to_state(prepared[job.job_id]),
                }
                for job in jobs
            ],
            "cursors": {},
            "seed_outcomes": {
                job.run_id: {
                    "run_id": job.run_id,
                    "seed": index,
                    "status": "LAUNCHED",
                    "stage": "launched",
                    "paths_exposed": False,
                }
                for index, job in enumerate(jobs, start=1)
            },
            "job_outcomes": {},
            "active_operation": operation,
            "operation_history": [operation],
        },
    )
    service.sessions["campaign-controls"] = TrainingSession(
        "campaign-controls",
        backend,
        plan,
        jobs=jobs,
        dashboard=dashboard,
        prepared=prepared,
        prepared_stages={job.job_id: "launched" for job in jobs},
        requests=requests,
        seed_outcomes={
            job.run_id: {
                "run_id": job.run_id,
                "seed": index,
                "status": "LAUNCHED",
                "stage": "launched",
                "paths_exposed": False,
            }
            for index, job in enumerate(jobs, start=1)
        },
        operation_nonce=str(operation["operation_nonce"]),
        operation_action=str(operation["action"]),
        launch_authorization_evidence_sha256="a" * 64,
    )
    return service, jobs


def test_cancel_is_terminal_durable_and_refresh_safe(tmp_path: Path) -> None:
    backend = FakeComputeBackend()
    original, _jobs = _seed_service(tmp_path, backend)
    service = TrainingService(_context(tmp_path), backend)
    assert service.repository.state("campaign-controls") == original.repository.state("campaign-controls")

    first = service.cancel("campaign-controls")
    call_count = len(backend.calls)
    second = service.cancel("campaign-controls")

    assert first.status == ProductStatus.COMPLETE
    assert first.data["terminal_status"] == "CANCELLED"
    assert service.repository.state("campaign-controls")["status"] == "CANCELLED"
    assert second.status == ProductStatus.COMPLETE
    assert second.data["cancelled"] is True
    assert len(backend.calls) == call_count
    assert service.dashboard("campaign-controls").data["terminal_status"] == "CANCELLED"


class _ChangedFalseBackend(FakeComputeBackend):
    def cancel(self, job: ComputeJob) -> OperationResult:
        self.calls.append(f"cancel:{job.job_id}")
        return OperationResult(False, "No change")


def test_changed_false_cancel_remains_uncertain_and_retryable(tmp_path: Path) -> None:
    backend = _ChangedFalseBackend()
    service, _jobs = _seed_service(tmp_path, backend)

    result = service.cancel("campaign-controls")

    assert result.status == ProductStatus.BLOCKED
    assert result.data["cancelled"] is False
    assert result.data["cancel_available"] is True
    assert result.data["cancel_unverified_count"] == 1
    assert service.dashboard("campaign-controls").data["cancel_available"] is True
    response = _action_response(result, "cancel", tmp_path)
    payload = json.loads(response.body)
    assert response.status_code == 409
    assert payload["error_code"] == "training_cancel_blocked"
    assert payload["details"]["cancel_available"] is True
    assert payload["details"]["job_outcomes"][0]["status"] == "UNCERTAIN"


def test_failed_run_with_retained_jobs_remains_cancelable_after_reconstruction(tmp_path: Path) -> None:
    backend = FakeComputeBackend()
    service, _jobs = _seed_service(tmp_path, backend, status=ProductStatus.FAILED)

    dashboard = service.dashboard("campaign-controls")

    assert dashboard.status == ProductStatus.FAILED
    assert dashboard.data["cancel_available"] is True


class _CloudCancelBackend(FakeComputeBackend):
    def __init__(self, *, shutdown_verified: bool) -> None:
        super().__init__(is_cloud=True)
        self.shutdown_verified = shutdown_verified

    def poll(self, job: ComputeJob) -> ComputePoll:
        current = self._jobs[job.job_id]
        return ComputePoll(
            current.status,
            "Cloud state",
            may_accrue_cost=not self.shutdown_verified,
            metadata={"resource_shutdown_verified": self.shutdown_verified},
        )


@pytest.mark.parametrize(
    ("shutdown_verified", "expected"), [(False, ProductStatus.BLOCKED), (True, ProductStatus.COMPLETE)]
)
def test_cloud_cancel_requires_explicit_resource_shutdown(
    tmp_path: Path,
    shutdown_verified: bool,
    expected: ProductStatus,
) -> None:
    backend = _CloudCancelBackend(shutdown_verified=shutdown_verified)
    service, _jobs = _seed_service(tmp_path, backend)

    result = service.cancel("campaign-controls")

    assert result.status == expected
    assert result.data["cancelled"] is shutdown_verified
    assert result.data.get("may_accrue_cost", False) is (not shutdown_verified)


class _PauseAllBackend(FakeComputeBackend):
    def pause(self, job: ComputeJob) -> OperationResult:
        self.calls.append(f"pause:{job.job_id}")
        if job.job_id == "job-1":
            raise ComputeBackendError("first job failed")
        self._jobs[job.job_id] = replace(job, status=ComputeStatus.PAUSED)
        return OperationResult(True, "paused")


def test_pause_attempts_every_job_after_failure(tmp_path: Path) -> None:
    backend = _PauseAllBackend()
    service, _jobs = _seed_service(tmp_path, backend, job_count=2)

    result = service.pause("campaign-controls")

    assert result.status == ProductStatus.BLOCKED
    assert result.data["pause_attempt_count"] == 2
    assert result.data["pause_unverified_count"] == 1
    assert [item for item in backend.calls if item.startswith("pause:")] == ["pause:job-1", "pause:job-2"]
    assert service.dashboard("campaign-controls").data["cancel_available"] is True


class _MixedPollBackend(FakeComputeBackend):
    def poll(self, job: ComputeJob) -> ComputePoll:
        if job.job_id == "job-1":
            return ComputePoll(ComputeStatus.FAILED, "failed")
        return ComputePoll(ComputeStatus.RUNNING, "running")


def test_mixed_failed_and_active_refresh_stays_cancelable(tmp_path: Path) -> None:
    backend = _MixedPollBackend()
    service, _jobs = _seed_service(tmp_path, backend, job_count=2)

    result = service.refresh("campaign-controls")

    assert result.status == ProductStatus.BLOCKED
    assert result.data["cancel_available"] is True
    assert {item["status"] for item in result.data["job_outcomes"]} == {"FAILED", "RUNNING"}


class _PollFailureBackend(FakeComputeBackend):
    def poll(self, job: ComputeJob) -> ComputePoll:
        raise ComputeBackendError("private host detail")


def test_poll_failure_is_durable_uncertain_without_private_error(tmp_path: Path) -> None:
    backend = _PollFailureBackend()
    service, _jobs = _seed_service(tmp_path, backend)

    result = service.refresh("campaign-controls")
    replay = service.repository.replay("campaign-controls")

    assert result.status == ProductStatus.BLOCKED
    assert result.data["remote_resource_uncertain"] is True
    assert result.data["cancel_available"] is True
    assert any(event.event.metrics.get("resource_state_uncertain") is True for event in replay.events)
    assert "private host detail" not in str(result.to_dict())


def test_public_outcome_projection_keeps_control_state_without_remote_identity(tmp_path: Path) -> None:
    private_remote = f"private-host:{tmp_path}"
    projected = _public_dashboard_projection(
        {
            "status": "BLOCKED",
            "terminal_status": None,
            "seed_outcomes": [{"run_id": "seed-1", "seed": 1, "status": "BLOCKED", "stage": "launch"}],
            "job_outcomes": [
                {
                    "job_id": "job-1",
                    "run_id": "seed-1",
                    "status": "UNCERTAIN",
                    "stage": "cancel",
                    "remote_identity": private_remote,
                    "resource_shutdown_verified": False,
                    "may_accrue_cost": True,
                }
            ],
            "remote_resource_uncertain": True,
            "may_accrue_cost": True,
        },
        tmp_path,
    )

    assert projected["seed_outcomes"][0]["status"] == "BLOCKED"
    assert projected["job_outcomes"][0]["resource_shutdown_verified"] is False
    assert private_remote not in str(projected)
    assert "remote_identity" not in str(projected)


def test_side_effect_then_throw_is_reconciled_under_same_operation_nonce(tmp_path: Path) -> None:
    backend = FakeComputeBackend()
    service, _jobs = _seed_service(tmp_path, backend)
    session = service.sessions["campaign-controls"]
    service._claim_control_operation(session, "pause")
    calls = 0

    def seam() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            backend.reconciled_marker = "same-operation"
            raise ComputeBackendError("transport failed after side effect")
        return backend.reconciled_marker

    assert service._idempotent_backend_call(session, seam) == "same-operation"
    assert calls == 2
    service._finish_operation(session, "COMPLETE")


def test_operation_record_tamper_reaches_zero_control_seams(tmp_path: Path) -> None:
    backend = FakeComputeBackend()
    service, _jobs = _seed_service(tmp_path, backend)
    state = service.repository.state("campaign-controls")
    tampered = dict(state["active_operation"])
    tampered["operation_nonce"] = "operation-" + "9" * 32
    service.repository.update_state("campaign-controls", active_operation=tampered)
    service.sessions.clear()

    result = service.cancel("campaign-controls")

    assert result.status == ProductStatus.UNAVAILABLE
    assert "cancel" not in backend.calls


def test_restart_inside_first_backend_seam_retains_durable_cancelable_uncertainty(tmp_path: Path) -> None:
    backend = FakeComputeBackend(is_cloud=True)
    service, _jobs = _seed_service(tmp_path, backend, job_count=0)
    running = _operation_record(
        action="start",
        run_id="campaign-controls",
        operation_nonce="operation-" + "8" * 32,
        campaign_identity="c" * 64,
        backend_id=backend.backend_id,
        backend_configuration_identity=service._backend_configuration_identity(),
        activation_commit_record_identity="a" * 64,
        project_config_sha256="b" * 64,
        launch_authorization_evidence_sha256="a" * 64,
        status="RUNNING",
    )
    service.repository.update_state(
        "campaign-controls",
        active_operation=running,
        operation_history=[],
    )
    service.sessions.clear()

    first = service.cancel("campaign-controls")
    second = service.cancel("campaign-controls")

    assert first.status == ProductStatus.BLOCKED
    assert second.status == ProductStatus.BLOCKED
    assert first.data["unknown_backend_operation_count"] == 1
    assert second.data["unknown_backend_operation_count"] == 1
    assert first.data["cancel_available"] is True
    assert service.dashboard("campaign-controls").data["cancel_available"] is True
    assert service.repository.state("campaign-controls")["unknown_backend_operation_count"] == 1
    assert "cancel" not in backend.calls


def test_start_while_activation_lock_is_held_reaches_zero_backend_calls(tmp_path: Path) -> None:
    backend = FakeComputeBackend()
    service = TrainingService(_context(tmp_path), backend)

    with TrainingActionLock(tmp_path, timeout_seconds=0.0):
        result = service.start()

    assert result.status == ProductStatus.BLOCKED
    assert result.blockers[0].code == "training_action_conflict"
    assert backend.calls == []


def test_action_lock_rejects_outside_zero_byte_hardlink_without_mutation(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_sentinel = tmp_path / "outside-zero-byte-sentinel.bin"
    outside_sentinel.write_bytes(b"")
    try:
        os.link(outside_sentinel, project_root / ACTION_LOCK_FILENAME)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hard links are unavailable for this filesystem: {exc}")

    with pytest.raises(TrainingActionLockError):
        with TrainingActionLock(project_root, timeout_seconds=0.0):
            pass

    assert outside_sentinel.read_bytes() == b""
    assert outside_sentinel.stat().st_size == 0


def test_action_lock_keeps_one_authority_across_rename_recreate_attack(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    lock_path = project_root / ACTION_LOCK_FILENAME
    displaced = project_root / ".displaced-action-lock"

    if os.name == "nt":
        with TrainingActionLock(project_root, timeout_seconds=0.0):
            with pytest.raises(OSError):
                os.replace(lock_path, displaced)
            with pytest.raises(TrainingActionLockError):
                with TrainingActionLock(project_root, timeout_seconds=0.0):
                    pass
        return

    with pytest.raises(TrainingActionLockError, match="changed"):
        with TrainingActionLock(project_root, timeout_seconds=0.0):
            os.replace(lock_path, displaced)
            lock_path.write_bytes(b"\0")
            with pytest.raises(TrainingActionLockError):
                with TrainingActionLock(project_root, timeout_seconds=0.0):
                    pass


def test_nonrecommended_start_reaches_zero_backend_calls(tmp_path: Path) -> None:
    backend = FakeComputeBackend()
    service = TrainingService(_context(tmp_path), backend)

    result = service.start(TrainingProfile.QUALITY)

    assert result.status == ProductStatus.BLOCKED
    assert result.blockers[0].code == "conditioned_profile_ineligible"
    assert backend.calls == []

    status = service.status(TrainingProfile.CUSTOM)
    assert status.status == ProductStatus.BLOCKED
    assert status.data["ready"] is False
    assert backend.calls == []


def _bindings(*, campaign: str = "c" * 64) -> dict[str, str]:
    return {
        "action": "start",
        "run_id": "campaign-controls",
        "campaign_identity_sha256": campaign,
        "backend_id": "cloud-backend",
        "backend_configuration_identity_sha256": "d" * 64,
        "project_config_sha256": "e" * 64,
        "activation_commit_record_identity": "f" * 64,
        "launch_authorization_evidence_sha256": "a" * 64,
    }


def _challenge_paths(root: Path, challenge_id: str) -> tuple[Path, Path]:
    directory = root / cloud_challenge_module.CLOUD_CHALLENGE_DIRECTORY
    return directory / f"{challenge_id}.json", directory / f"{challenge_id}.consumed.json"


def test_cloud_challenge_requires_concrete_launch_authorization_evidence(tmp_path: Path) -> None:
    store = CloudChallengeStore(tmp_path)
    missing = _bindings()
    missing.pop("launch_authorization_evidence_sha256")
    uppercase = {**_bindings(), "launch_authorization_evidence_sha256": "A" * 64}

    for bindings in (missing, uppercase):
        with pytest.raises(CloudChallengeError) as captured:
            store.issue(bindings)
        assert captured.value.code == "cloud_challenge_binding_invalid"


def test_cloud_challenge_is_one_use_immutable_and_exactly_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CloudChallengeStore(tmp_path)
    issued = store.issue(_bindings())
    token = str(issued["challenge_token"])
    issued_path, consumed_path = _challenge_paths(tmp_path, str(issued["challenge_id"]))
    issued_bytes = issued_path.read_bytes()

    def mutable_replacement_forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cloud challenge consumption must not replace mutable state")

    monkeypatch.setattr(cloud_challenge_module.AnchoredDirectory, "atomic_write_bytes", mutable_replacement_forbidden)
    with TrainingActionLock(tmp_path, timeout_seconds=0.0):
        with pytest.raises(CloudChallengeError, match="does not match"):
            store.consume_locked(
                token,
                expected_bindings=_bindings(campaign="9" * 64),
                operation_nonce="operation-" + "1" * 32,
            )
        consumed = store.consume_locked(
            token,
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "2" * 32,
        )
    assert consumed["status"] == "CONSUMED"
    assert issued_path.read_bytes() == issued_bytes
    marker = json.loads(consumed_path.read_text(encoding="utf-8"))
    issued_record = json.loads(issued_bytes)
    assert marker == consumed
    assert marker["schema_version"] == cloud_challenge_module.CLOUD_CHALLENGE_CONSUMPTION_SCHEMA
    assert marker["challenge_id"] == issued["challenge_id"]
    assert marker["token_sha256"] == issued["challenge_id"]
    assert marker["bindings"] == _bindings()
    assert marker["operation_nonce"] == "operation-" + "2" * 32
    assert marker["issued_record_sha256"] == hashlib.sha256(issued_bytes).hexdigest()
    assert marker["issued_record_identity"] == issued_record["record_identity"]
    assert marker["issued_record_authentication_sha256"] == issued_record["record_authentication_sha256"]
    assert token not in consumed_path.read_text(encoding="utf-8")
    with TrainingActionLock(tmp_path, timeout_seconds=0.0):
        with pytest.raises(CloudChallengeError, match="already consumed"):
            store.consume_locked(
                token,
                expected_bindings=_bindings(),
                operation_nonce="operation-" + "3" * 32,
            )


def test_cloud_challenge_concurrent_consumption_has_one_winner(tmp_path: Path) -> None:
    store = CloudChallengeStore(tmp_path)
    issued = store.issue(_bindings())
    token = str(issued["challenge_token"])
    ready = Barrier(2)

    def consume(index: int) -> str:
        try:
            ready.wait(timeout=2.0)
            store.consume_locked(
                token,
                expected_bindings=_bindings(),
                operation_nonce=f"operation-{index:032x}",
            )
        except CloudChallengeError as exc:
            return exc.code
        return "consumed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(consume, (1, 2)))
    assert sorted(outcomes) == ["cloud_challenge_replayed", "consumed"]


def test_cloud_challenge_expiry_is_fail_closed(tmp_path: Path) -> None:
    now = [datetime(2026, 7, 18, 10, 0, tzinfo=UTC)]
    store = CloudChallengeStore(tmp_path, clock=lambda: now[0], ttl_seconds=2)
    token = str(store.issue(_bindings())["challenge_token"])
    now[0] += timedelta(seconds=3)

    with TrainingActionLock(tmp_path, timeout_seconds=0.0):
        with pytest.raises(CloudChallengeError) as captured:
            store.consume_locked(
                token,
                expected_bindings=_bindings(),
                operation_nonce="operation-" + "4" * 32,
            )
    assert captured.value.code == "cloud_challenge_expired"


def test_cloud_challenge_partial_marker_burns_authority_without_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CloudChallengeStore(tmp_path)
    issued = store.issue(_bindings())
    token = str(issued["challenge_token"])
    issued_path, consumed_path = _challenge_paths(tmp_path, str(issued["challenge_id"]))
    issued_bytes = issued_path.read_bytes()
    outside_sentinel = tmp_path / "outside-cloud-challenge-sentinel.bin"
    outside_sentinel.write_bytes(b"outside-sentinel")

    def interrupted_write(descriptor: int, content: bytes) -> None:
        assert os.write(descriptor, content[:7]) == 7
        raise OSError("simulated process interruption")

    monkeypatch.setattr(cloud_challenge_module, "_write_all", interrupted_write)
    with pytest.raises(CloudChallengeError) as interrupted:
        store.consume_locked(
            token,
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "5" * 32,
        )
    assert interrupted.value.code == "cloud_challenge_unavailable"
    assert consumed_path.read_bytes() == b'{\n  "bi'
    assert issued_path.read_bytes() == issued_bytes
    assert outside_sentinel.read_bytes() == b"outside-sentinel"

    with pytest.raises(CloudChallengeError) as replay:
        store.consume_locked(
            token,
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "6" * 32,
        )
    assert replay.value.code == "cloud_challenge_replayed"
    assert consumed_path.read_bytes() == b'{\n  "bi'
    assert outside_sentinel.read_bytes() == b"outside-sentinel"


def test_cloud_challenge_issued_record_substitution_never_authorizes(tmp_path: Path) -> None:
    store = CloudChallengeStore(tmp_path)
    first = store.issue(_bindings())
    second = store.issue(_bindings(campaign="9" * 64))
    first_issued, first_consumed = _challenge_paths(tmp_path, str(first["challenge_id"]))
    second_issued, _second_consumed = _challenge_paths(tmp_path, str(second["challenge_id"]))
    retained = first_issued.with_name(f"{first_issued.name}.retained")
    outside_sentinel = tmp_path / "outside-substitution-sentinel.bin"
    outside_sentinel.write_bytes(b"outside-substitution")
    first_issued.rename(retained)
    second_issued.rename(first_issued)

    with pytest.raises(CloudChallengeError) as captured:
        store.consume_locked(
            str(first["challenge_token"]),
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "7" * 32,
        )
    assert captured.value.code == "cloud_challenge_invalid"
    assert not first_consumed.exists()
    assert retained.read_bytes() != first_issued.read_bytes()
    assert outside_sentinel.read_bytes() == b"outside-substitution"


def test_cloud_challenge_rejects_rehashed_issued_record_without_token_authentication(tmp_path: Path) -> None:
    store = CloudChallengeStore(tmp_path)
    issued = store.issue(_bindings())
    issued_path, consumed_path = _challenge_paths(tmp_path, str(issued["challenge_id"]))
    record = json.loads(issued_path.read_text(encoding="utf-8"))
    record["expires_at"] = "2099-01-01T00:00:00Z"
    payload = dict(record)
    payload.pop("record_identity")
    record["record_identity"] = cloud_challenge_module.stable_hash(payload)
    issued_path.write_bytes(cloud_challenge_module._canonical_bytes(record))

    with pytest.raises(CloudChallengeError) as captured:
        store.consume_locked(
            str(issued["challenge_token"]),
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "b" * 32,
        )
    assert captured.value.code == "cloud_challenge_invalid"
    assert not consumed_path.exists()


def test_cloud_challenge_hardlinked_issued_record_fails_closed(tmp_path: Path) -> None:
    store = CloudChallengeStore(tmp_path)
    issued = store.issue(_bindings())
    issued_path, consumed_path = _challenge_paths(tmp_path, str(issued["challenge_id"]))
    outside_alias = tmp_path / "outside-issued-hardlink.json"
    outside_sentinel = tmp_path / "outside-hardlink-sentinel.bin"
    outside_sentinel.write_bytes(b"outside-hardlink")
    os.link(issued_path, outside_alias)

    with pytest.raises(CloudChallengeError) as captured:
        store.consume_locked(
            str(issued["challenge_token"]),
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "8" * 32,
        )
    assert captured.value.code == "cloud_challenge_unavailable"
    assert not consumed_path.exists()
    assert outside_sentinel.read_bytes() == b"outside-hardlink"


def test_cloud_challenge_hardlinked_consumption_marker_blocks_without_writing_target(tmp_path: Path) -> None:
    store = CloudChallengeStore(tmp_path)
    issued = store.issue(_bindings())
    _issued_path, consumed_path = _challenge_paths(tmp_path, str(issued["challenge_id"]))
    outside_sentinel = tmp_path / "outside-marker-hardlink.bin"
    outside_sentinel.write_bytes(b"outside-marker-hardlink")
    os.link(outside_sentinel, consumed_path)

    with pytest.raises(CloudChallengeError) as captured:
        store.consume_locked(
            str(issued["challenge_token"]),
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "9" * 32,
        )
    assert captured.value.code == "cloud_challenge_replayed"
    assert outside_sentinel.read_bytes() == b"outside-marker-hardlink"


def test_cloud_challenge_symlinked_consumption_marker_blocks_without_writing_target(tmp_path: Path) -> None:
    store = CloudChallengeStore(tmp_path)
    issued = store.issue(_bindings())
    _issued_path, consumed_path = _challenge_paths(tmp_path, str(issued["challenge_id"]))
    outside_sentinel = tmp_path / "outside-marker-symlink.bin"
    outside_sentinel.write_bytes(b"outside-marker-symlink")
    try:
        consumed_path.symlink_to(outside_sentinel)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(CloudChallengeError) as captured:
        store.consume_locked(
            str(issued["challenge_token"]),
            expected_bindings=_bindings(),
            operation_nonce="operation-" + "a" * 32,
        )
    assert captured.value.code == "cloud_challenge_replayed"
    assert outside_sentinel.read_bytes() == b"outside-marker-symlink"
