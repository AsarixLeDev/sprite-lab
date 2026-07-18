from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import spritelab.training.smoke_runner as runner
import spritelab.training.smoke_worker as worker
from spritelab.training.smoke_bundle import (
    FALSE_ELIGIBILITY,
    SmokeBundleError,
    canonical_json_bytes,
    finalize_identity,
    stable_hash,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _WorkerHarness:
    root: Path
    plan: dict[str, Any]
    launch_identity: str
    state_path: Path
    process_identity: dict[str, Any]
    training_argv: list[str]
    worker_argv: list[str]

    def state(self, status: str = "RUNNING", *, sequence: int | None = None) -> dict[str, Any]:
        device = "cpu"
        limit = int(self.plan["configurations"][device]["wall_clock_limit_seconds"])
        now = datetime.now(timezone.utc)
        started = now - timedelta(seconds=10)
        updated = started if status == "STARTING" else started + timedelta(seconds=1)
        worker_pid: int | None = os.getpid()
        worker_process: dict[str, Any] | None = dict(self.process_identity)
        transition_sequence = 1 if sequence is None else sequence
        current = 0
        exit_code: int | None = None
        receipt_identity: str | None = None
        if status == "STARTING":
            transition_sequence = 0
            worker_pid = None
            worker_process = None
        elif status == "COMPLETE":
            current = 2
            exit_code = 0
            receipt_identity = _digest("receipt")
        elif status == "FAILED":
            exit_code = 70
        elif status == "CANCELLED":
            exit_code = 130
        elif status == "TIMED_OUT":
            started = now - timedelta(seconds=limit + 2)
            updated = started + timedelta(seconds=limit + 1)
            exit_code = 124
        body = {
            "schema_version": worker._EXECUTION_STATE_SCHEMA,
            "smoke_id": self.plan["smoke_id"],
            "device": device,
            "plan_identity": self.plan["plan_identity"],
            "launch_identity": self.launch_identity,
            "status": status,
            "current": current,
            "total": 2,
            "owner_pid": os.getpid(),
            "worker_pid": worker_pid,
            "worker_process": worker_process,
            "portable_argv": list(self.training_argv),
            "worker_argv": list(self.worker_argv),
            "argv_identity": stable_hash(self.training_argv),
            "worker_argv_identity": stable_hash(self.worker_argv),
            "execution_mode": "linux-worker-trainer-v1",
            "confinement": None,
            "environment": dict(self.plan["configurations"][device]["environment"]),
            "environment_identity": self.plan["configurations"][device]["child_environment"]["environment_sha256"],
            "interpreter_identity": self.plan["interpreter"]["interpreter_identity"],
            "writable_roots": [
                worker._directory_identity_record(self.root.joinpath(*Path(relative).parts), self.root)
                for relative in self.plan["configurations"][device]["writable_roots"]
            ],
            "started_at": started.isoformat(),
            "deadline_at": (started + timedelta(seconds=limit)).isoformat(),
            "wall_clock_limit_seconds": limit,
            "updated_at": updated.isoformat(),
            "exit_code": exit_code,
            "receipt_identity": receipt_identity,
            "transition_sequence": transition_sequence,
            "retry_policy": "NEW_BUNDLE_REQUIRED",
            "resumable": False,
            "logs": ["Worker protocol fixture state."],
            **FALSE_ELIGIBILITY,
        }
        body["state_identity"] = stable_hash(body)
        worker._validate_execution_state(body, self.root, self.plan, device, self.launch_identity)
        return body

    def resign(self, state: dict[str, Any]) -> dict[str, Any]:
        body = {key: value for key, value in state.items() if key != "state_identity"}
        body["state_identity"] = stable_hash(body)
        return body

    def publish(self, state: dict[str, Any]) -> None:
        self.state_path.write_bytes(canonical_json_bytes(state, pretty=True))

    def heartbeat(self, state: dict[str, Any], sequence: int) -> dict[str, Any]:
        return worker._publish_heartbeat(
            self.root,
            self.plan,
            "cpu",
            self.launch_identity,
            datetime.now(timezone.utc).isoformat(),
            sequence,
            self.process_identity,
            status=str(state["status"]),
            containment="LINUX_PDEATHSIG",
            execution_state=state,
        )


@pytest.fixture
def worker_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _WorkerHarness:
    root = tmp_path / "project"
    execution = root / "bundle" / "execution" / "cpu"
    output = root / "output"
    execution.mkdir(parents=True)
    output.mkdir()
    training_argv = ["python", "-m", "spritelab", "train"]
    worker_argv = ["python", "-I", "-B", "-S", "-c", "worker"]
    launch_identity = _digest("launch")
    process_identity = {
        "pid": os.getpid(),
        "birth_token": "worker-birth-token",
        "process_image_path_sha256": _digest("worker-image"),
    }
    plan = {
        "smoke_id": "smoke-" + "a" * 20,
        "plan_identity": _digest("plan"),
        "interpreter": {"interpreter_identity": _digest("interpreter")},
        "configurations": {
            "cpu": {
                "environment": {"CUDA_VISIBLE_DEVICES": "-1", "SPRITELAB_PROGRESS": "0"},
                "child_environment": {"environment_sha256": _digest("environment")},
                "wall_clock_limit_seconds": 600,
                "writable_roots": [
                    execution.relative_to(root).as_posix(),
                    output.relative_to(root).as_posix(),
                ],
            }
        },
    }
    monkeypatch.setattr(worker, "artifact_bundle_directory", lambda _root, _smoke_id: root / "bundle")
    monkeypatch.setattr(worker, "smoke_training_argv", lambda _plan, _device: list(training_argv))
    monkeypatch.setattr(worker, "smoke_worker_argv", lambda _plan, _device: list(worker_argv))
    monkeypatch.setattr(
        worker,
        "_process_identity",
        lambda pid: {**process_identity, "pid": pid},
    )
    return _WorkerHarness(
        root=root,
        plan=plan,
        launch_identity=launch_identity,
        state_path=execution / "state.json",
        process_identity=process_identity,
        training_argv=training_argv,
        worker_argv=worker_argv,
    )


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_execution_state_requires_its_exact_identity(
    worker_harness: _WorkerHarness,
    mutation: str,
) -> None:
    state = worker_harness.state()
    if mutation == "missing":
        state.pop("state_identity")
    else:
        state["logs"] = ["Tampered without re-signing."]
    worker_harness.publish(state)

    with pytest.raises(SmokeBundleError, match="execution state is invalid"):
        worker._read_execution_state(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )


def test_execution_state_rejects_a_resigned_foreign_worker(worker_harness: _WorkerHarness) -> None:
    state = worker_harness.state()
    state["worker_process"] = {
        **state["worker_process"],
        "process_image_path_sha256": _digest("foreign-worker-image"),
    }
    state = worker_harness.resign(state)
    worker_harness.publish(state)

    with pytest.raises(SmokeBundleError, match="execution state is invalid"):
        worker._read_execution_state(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )


def test_heartbeat_is_exact_and_rejects_cross_launch(worker_harness: _WorkerHarness) -> None:
    state = worker_harness.state()
    worker_harness.publish(state)
    heartbeat = worker_harness.heartbeat(state, 1)
    assert (
        worker.load_worker_heartbeat(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )
        == heartbeat
    )

    with pytest.raises(SmokeBundleError, match="heartbeat is invalid"):
        worker.load_worker_heartbeat(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            _digest("other-launch"),
        )

    malformed = {key: value for key, value in heartbeat.items() if key != "heartbeat_identity"}
    malformed["unexpected"] = "field"
    malformed = finalize_identity(malformed, "heartbeat_identity")
    worker.heartbeat_path(worker_harness.root, worker_harness.plan["smoke_id"], "cpu").write_bytes(
        canonical_json_bytes(malformed, pretty=True)
    )
    with pytest.raises(SmokeBundleError, match="heartbeat is invalid"):
        worker.load_worker_heartbeat(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )


def test_heartbeat_and_outcome_reject_stale_state_bindings(worker_harness: _WorkerHarness) -> None:
    state = worker_harness.state()
    worker_harness.publish(state)
    heartbeat_one = worker_harness.heartbeat(state, 1)

    newer = dict(state)
    newer["current"] = 1
    newer["transition_sequence"] += 1
    newer["updated_at"] = datetime.now(timezone.utc).isoformat()
    newer = worker_harness.resign(newer)
    worker_harness.publish(newer)
    with pytest.raises(SmokeBundleError, match="heartbeat is stale"):
        worker.load_worker_heartbeat(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )

    heartbeat_two = worker_harness.heartbeat(newer, 2)
    with pytest.raises(SmokeBundleError, match="heartbeat is stale"):
        worker._finish(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
            heartbeat_one,
            70,
        )
    assert (
        worker._finish(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
            heartbeat_two,
            70,
        )
        == 70
    )
    assert (
        worker.load_worker_outcome(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )["status"]
        == "FAILED"
    )

    newest = dict(newer)
    newest["transition_sequence"] += 1
    newest["updated_at"] = datetime.now(timezone.utc).isoformat()
    newest = worker_harness.resign(newest)
    worker_harness.publish(newest)
    with pytest.raises(SmokeBundleError, match="outcome is stale"):
        worker.load_worker_outcome(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )


def test_outcome_is_exact_and_rejects_cross_launch(worker_harness: _WorkerHarness) -> None:
    state = worker_harness.state()
    worker_harness.publish(state)
    heartbeat = worker_harness.heartbeat(state, 1)
    assert (
        worker._finish(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
            heartbeat,
            70,
        )
        == 70
    )
    outcome = worker.load_worker_outcome(
        worker_harness.root,
        worker_harness.plan,
        "cpu",
        worker_harness.launch_identity,
    )

    with pytest.raises(SmokeBundleError, match="outcome is invalid"):
        worker.load_worker_outcome(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            _digest("other-launch"),
        )

    malformed = {key: value for key, value in outcome.items() if key != "outcome_identity"}
    malformed["unexpected"] = "field"
    malformed = finalize_identity(malformed, "outcome_identity")
    worker.outcome_path(worker_harness.root, worker_harness.plan["smoke_id"], "cpu").write_bytes(
        canonical_json_bytes(malformed, pretty=True)
    )
    with pytest.raises(SmokeBundleError, match="outcome is invalid"):
        worker.load_worker_outcome(
            worker_harness.root,
            worker_harness.plan,
            "cpu",
            worker_harness.launch_identity,
        )


def test_finalization_rechecks_cancellation_under_the_state_lock(worker_harness: _WorkerHarness) -> None:
    running = worker_harness.state()
    worker_harness.publish(running)
    heartbeat = worker_harness.heartbeat(running, 1)
    checks = 0

    def cancel_after_check() -> None:
        nonlocal checks
        checks += 1
        if checks != 1:
            return
        cancelled = dict(running)
        cancelled.update(
            {
                "status": "CANCELLED",
                "exit_code": 130,
                "transition_sequence": int(running["transition_sequence"]) + 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        worker_harness.publish(worker_harness.resign(cancelled))

    result = worker._finish(
        worker_harness.root,
        worker_harness.plan,
        "cpu",
        worker_harness.launch_identity,
        heartbeat,
        0,
        operation_check=cancel_after_check,
    )
    outcome = worker.load_worker_outcome(
        worker_harness.root,
        worker_harness.plan,
        "cpu",
        worker_harness.launch_identity,
    )
    assert result == 130
    assert outcome["status"] == "CANCELLED"
    assert outcome["exit_code"] == 130


def test_outcome_idempotence_requires_byte_identical_evidence(
    worker_harness: _WorkerHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = worker_harness.state()
    worker_harness.publish(state)
    fixed = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(worker, "_now", lambda: fixed)
    heartbeat = worker_harness.heartbeat(state, 1)
    arguments = (
        worker_harness.root,
        worker_harness.plan,
        "cpu",
        worker_harness.launch_identity,
        heartbeat,
        0,
    )
    assert worker._finish(*arguments) == 0
    assert worker._finish(*arguments) == 0

    later = (datetime.fromisoformat(fixed) + timedelta(microseconds=1)).isoformat()
    monkeypatch.setattr(worker, "_now", lambda: later)
    assert worker._finish(*arguments) == 70


def test_main_closes_inherited_descriptors_on_pretry_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.open(os.devnull, os.O_RDONLY)
    parsed = SimpleNamespace(
        smoke_id="smoke-" + "b" * 20,
        device="cpu",
        plan_identity="plan",
        launch_identity="launch",
    )
    plan = {"smoke_id": parsed.smoke_id, "plan_identity": parsed.plan_identity}
    monkeypatch.setattr(worker, "_parse_args", lambda: parsed)
    monkeypatch.setattr(worker, "load_plan", lambda *_args: plan)
    monkeypatch.setattr(worker, "smoke_launch_identity", lambda *_args: parsed.launch_identity)
    monkeypatch.setattr(worker, "_parse_writable_root_fds", lambda _value: (descriptor,))
    monkeypatch.setattr(
        worker,
        "validate_smoke_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected validation failure")),
    )

    assert worker.main() == 70
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize(
    ("terminal_status", "exit_code"),
    (("CANCELLED", 130), ("TIMED_OUT", 124)),
)
def test_receipt_validation_rechecks_cancellation_and_deadline_before_complete(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    exit_code: int,
) -> None:
    parsed = SimpleNamespace(
        smoke_id="smoke-" + "c" * 20,
        device="cpu",
        plan_identity="plan",
        launch_identity="launch",
    )
    plan = {"smoke_id": parsed.smoke_id, "plan_identity": parsed.plan_identity}
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    execution_state = {"status": "RUNNING", "deadline_at": future}
    finish_calls: list[tuple[str | None, int]] = []
    receipt_checks: list[object] = []
    terminated: list[int] = []

    class Process:
        pid = os.getpid()

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait() -> int:
            return 0

    class Containment:
        name = "LINUX_PDEATHSIG"

        def __init__(self, process: Process) -> None:
            self.process = process

        def activate(self, *, verifier: Any) -> None:
            verifier(self.process)

        def terminate(self) -> None:
            terminated.append(self.process.pid)

        def close(self) -> None:
            return None

    @contextmanager
    def pinned(_plan: dict[str, Any], **_kwargs: Any):
        yield SimpleNamespace(launch_path="python", pass_fds=())

    def load_receipt(
        _root: Path,
        _plan: dict[str, Any],
        _device: str,
        *,
        operation_check: Any,
    ) -> dict[str, Any]:
        receipt_checks.append(operation_check)
        if terminal_status == "CANCELLED":
            execution_state["status"] = "CANCELLED"
        else:
            execution_state["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        operation_check()
        raise AssertionError("terminal receipt validation unexpectedly continued")

    def finish(
        _root: Path,
        _plan: dict[str, Any],
        _device: str,
        _launch_identity: str,
        _heartbeat: dict[str, Any],
        actual_exit_code: int,
        *,
        status_override: str | None = None,
        **_kwargs: Any,
    ) -> int:
        finish_calls.append((status_override, actual_exit_code))
        return actual_exit_code

    monkeypatch.delenv("SPRITELAB_WRITABLE_ROOT_FDS", raising=False)
    monkeypatch.setattr(worker, "_parse_args", lambda: parsed)
    monkeypatch.setattr(worker, "load_plan", lambda *_args: plan)
    monkeypatch.setattr(worker, "smoke_launch_identity", lambda *_args: parsed.launch_identity)
    monkeypatch.setattr(worker, "validate_smoke_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "validate_smoke_interpreter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "smoke_containment_supported", lambda: True)
    monkeypatch.setattr(worker, "verify_execution_guards", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "_process_identity", lambda pid: {"pid": pid, "birth_token": "1"})
    monkeypatch.setattr(
        worker,
        "_publish_heartbeat",
        lambda *_args, **_kwargs: {"heartbeat_identity": "a" * 64},
    )
    monkeypatch.setattr(worker, "_read_execution_state", lambda *_args, **_kwargs: dict(execution_state))
    monkeypatch.setattr(worker, "smoke_training_argv", lambda *_args: ["python", "-c", "pass"])
    monkeypatch.setattr(worker, "pinned_smoke_interpreter", pinned)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(worker, "_Containment", Containment)
    monkeypatch.setattr(worker, "verify_pinned_process_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "load_device_receipt", load_receipt)
    monkeypatch.setattr(worker, "_finish", finish)

    assert worker.main() == exit_code
    assert len(receipt_checks) == 1
    assert finish_calls == [(terminal_status, exit_code)]
    assert terminated == [Process.pid]


def test_posix_group_escalation_kills_resistant_descendants_after_leader_exit() -> None:
    calls: list[tuple[int, int]] = []

    class ExitedLeader:
        pid = 4242

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            del timeout
            return 0

        @staticmethod
        def kill() -> None:
            raise AssertionError("leader-only kill must not replace group escalation")

    worker._terminate_posix_process_group(
        ExitedLeader(),
        grace_seconds=0,
        kill_group=lambda pid, sig: calls.append((pid, sig)),
    )
    assert calls == [
        (4242, int(getattr(worker.signal, "SIGTERM", 15))),
        (4242, int(getattr(worker.signal, "SIGKILL", 9))),
    ]


def test_recorded_posix_group_requires_an_identity_pin_and_escalates_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "worker_pid": 4242,
        "worker_process": {
            "pid": 4242,
            "birth_token": "recorded-worker",
            "process_image_path_sha256": "a" * 64,
        },
    }
    signals: list[tuple[int, int]] = []
    closed: list[int] = []
    monkeypatch.setattr(runner, "worker_process_matches", lambda _state: True)

    with pytest.raises(runner.SmokeExecutionError, match="does not own"):
        runner._terminate_recorded_linux_process_group(
            state,
            grace_seconds=0,
            pidfd_open=lambda _pid, _flags: 81,
            get_process_group=lambda pid: pid + 1,
            kill_group=lambda pid, sig: signals.append((pid, sig)),
            group_exists=lambda _pid: True,
            close_descriptor=closed.append,
        )
    assert signals == []
    assert closed == [81]

    runner._terminate_recorded_linux_process_group(
        state,
        grace_seconds=0,
        pidfd_open=lambda _pid, _flags: 82,
        get_process_group=lambda pid: pid,
        kill_group=lambda pid, sig: signals.append((pid, sig)),
        group_exists=lambda _pid: True,
        close_descriptor=closed.append,
    )
    assert signals == [
        (4242, int(getattr(runner.signal, "SIGTERM", 15))),
        (4242, int(getattr(runner.signal, "SIGKILL", 9))),
    ]
    assert closed == [81, 82]
