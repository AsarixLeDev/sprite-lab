from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest

import spritelab.training.smoke_runner as runner_module
import spritelab.training.smoke_worker as worker_module
from spritelab.training.smoke_runner import ExploratorySmokeRunner, SmokeExecutionError


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _StateHarness:
    root: Path
    runner: ExploratorySmokeRunner
    state_path: Path
    plan: dict[str, Any]
    training_argv: list[str]
    worker_argv: list[str]
    launch_identity: str

    def state(self, status: str = "RUNNING") -> dict[str, Any]:
        device = "cpu"
        limit = int(self.plan["configurations"][device]["wall_clock_limit_seconds"])
        now = datetime.now(timezone.utc)
        if status == "TIMED_OUT":
            started = now - timedelta(seconds=limit + 2)
            updated = started + timedelta(seconds=limit + 1)
        else:
            started = now - timedelta(seconds=2)
            updated = started if status == "STARTING" else started + timedelta(seconds=1)
        worker_pid: int | None = 40404
        worker_process: dict[str, Any] | None = {
            "pid": worker_pid,
            "birth_token": "fixture-birth-token",
            "process_image_path_sha256": _digest("fixture-worker-image"),
        }
        sequence = 1
        current = 0
        exit_code: int | None = None
        receipt_identity: str | None = None
        if status == "STARTING":
            sequence = 0
            worker_pid = None
            worker_process = None
        elif status == "COMPLETE":
            current = 2
            exit_code = 0
            receipt_identity = _digest("fixture-receipt")
        elif status == "FAILED":
            exit_code = 7
        elif status == "CANCELLED":
            exit_code = 130
        elif status == "TIMED_OUT":
            exit_code = 124
        launched_argv = self.training_argv if os.name == "nt" else self.worker_argv
        execution_mode = "windows-direct-trainer-v1" if os.name == "nt" else "linux-worker-trainer-v1"
        confinement = None
        if os.name == "nt" and status != "STARTING":
            confinement = runner_module._windows_confinement_binding(
                SimpleNamespace(
                    bootstrap_identity_sha256=runner_module.WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256,
                    private_desktop_identity_sha256="a" * 64,
                    restricted_token=True,
                    restricted_sid_hashes_identity_sha256="b" * 64,
                ),
                self.root / "bundle" / "execution" / device,
                self.root / "output",
                project_root=self.root,
            )
        body = {
            "schema_version": runner_module.SMOKE_EXECUTION_SCHEMA,
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
            "worker_argv": list(launched_argv),
            "argv_identity": runner_module.stable_hash(self.training_argv),
            "worker_argv_identity": runner_module.stable_hash(launched_argv),
            "execution_mode": execution_mode,
            "confinement": confinement,
            "environment": dict(self.plan["configurations"][device]["environment"]),
            "environment_identity": self.plan["configurations"][device]["child_environment"]["environment_sha256"],
            "interpreter_identity": self.plan["interpreter"]["interpreter_identity"],
            "writable_roots": [
                runner_module._directory_identity_record(
                    self.root.joinpath(*Path(relative).parts),
                    self.root,
                )
                for relative in self.plan["configurations"][device]["writable_roots"]
            ],
            "started_at": started.isoformat(),
            "deadline_at": (started + timedelta(seconds=limit)).isoformat(),
            "wall_clock_limit_seconds": limit,
            "updated_at": updated.isoformat(),
            "exit_code": exit_code,
            "receipt_identity": receipt_identity,
            "transition_sequence": sequence,
            "retry_policy": "NEW_BUNDLE_REQUIRED",
            "resumable": False,
            "logs": ["Fixture execution state."],
            **runner_module.FALSE_ELIGIBILITY,
        }
        value = runner_module._finalize_state_identity(body)
        self.runner._validate_state(value)
        return value

    def publish(self, state: dict[str, Any]) -> None:
        self.state_path.write_bytes(runner_module.canonical_json_bytes(state, pretty=True))

    def resign(self, state: dict[str, Any]) -> dict[str, Any]:
        body = {key: value for key, value in state.items() if key != "state_identity"}
        return runner_module._finalize_state_identity(body)

    def durable(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))


@pytest.fixture
def state_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _StateHarness:
    root = tmp_path / "project"
    execution = root / "bundle" / "execution" / "cpu"
    output = root / "output"
    execution.mkdir(parents=True)
    output.mkdir()
    smoke_id = "smoke-" + "a" * 20
    plan_identity = _digest("fixture-plan")
    launch_identity = _digest("fixture-launch")
    training_argv = ["python", "-m", "spritelab", "train"]
    worker_argv = ["python", "-I", "-c", "worker"]
    environment = {"CUDA_VISIBLE_DEVICES": "-1", "SPRITELAB_PROGRESS": "0"}
    plan = {
        "smoke_id": smoke_id,
        "plan_identity": plan_identity,
        "interpreter": {"interpreter_identity": _digest("fixture-interpreter")},
        "configurations": {
            "cpu": {
                "environment": environment,
                "child_environment": {"environment_sha256": _digest("fixture-environment")},
                "wall_clock_limit_seconds": 600,
                "writable_roots": [
                    execution.relative_to(root).as_posix(),
                    output.relative_to(root).as_posix(),
                ],
            }
        },
    }

    monkeypatch.setattr(runner_module, "artifact_bundle_directory", lambda _root, _smoke_id: root / "bundle")
    monkeypatch.setattr(runner_module, "load_plan", lambda _root, _smoke_id: plan)
    monkeypatch.setattr(runner_module, "smoke_training_argv", lambda _plan, _device: list(training_argv))
    monkeypatch.setattr(runner_module, "smoke_worker_argv", lambda _plan, _device: list(worker_argv))
    monkeypatch.setattr(runner_module, "smoke_launch_identity", lambda _plan, _device: launch_identity)
    monkeypatch.setattr(runner_module, "worker_process_identity_is_valid", lambda _value, _pid: True)
    runner = ExploratorySmokeRunner(root)
    return _StateHarness(
        root=root,
        runner=runner,
        state_path=execution / "state.json",
        plan=plan,
        training_argv=training_argv,
        worker_argv=worker_argv,
        launch_identity=launch_identity,
    )


def _hostile_replace_fixed_lock(path: Path) -> OSError | None:
    try:
        path.replace(path.with_name(f"{path.name}.attacker-held"))
        path.write_bytes(b"\0")
    except OSError as exc:
        return exc
    return None


def test_transition_lock_replacement_cannot_split_runner_worker_authority(tmp_path: Path) -> None:
    directory = tmp_path / "execution"
    directory.mkdir()
    lock_path = directory / ".state-transition.lock"
    replacement_result: list[OSError | None] = []

    def exercise() -> None:
        with runner_module._state_transition_lock(directory):
            with ThreadPoolExecutor(max_workers=1) as pool:
                replacement_result.append(pool.submit(_hostile_replace_fixed_lock, lock_path).result(timeout=5))
            with pytest.raises(worker_module.SmokeBundleError, match="Timed out waiting"):
                with worker_module._state_transition_lock(directory, timeout=0.05):
                    raise AssertionError("a replacement lock file must not create a second authority")

    if os.name == "nt":
        exercise()
        assert isinstance(replacement_result[0], OSError)
        assert lock_path.is_file()
    else:
        with pytest.raises(SmokeExecutionError, match="transition lock changed"):
            exercise()
        assert replacement_result == [None]


def test_launch_lock_replacement_cannot_create_a_competing_authority(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    lock_path = root / "artifacts" / "training" / "smokes" / ".execution-launch.lock"
    replacement_result: list[OSError | None] = []

    def exercise() -> None:
        with runner_module._project_launch_lock(root):
            with ThreadPoolExecutor(max_workers=1) as pool:
                replacement_result.append(pool.submit(_hostile_replace_fixed_lock, lock_path).result(timeout=5))
            with pytest.raises(SmokeExecutionError, match="Timed out waiting"):
                with runner_module._project_launch_lock(root, timeout=0.05):
                    raise AssertionError("a replacement launch lock must not create a second authority")

    if os.name == "nt":
        exercise()
        assert isinstance(replacement_result[0], OSError)
        assert lock_path.is_file()
    else:
        with pytest.raises(SmokeExecutionError, match="launch lock changed"):
            exercise()
        assert replacement_result == [None]


@pytest.mark.skipif(os.name == "nt", reason="Windows denies the hostile rename before the yielded body can fail.")
def test_transition_lock_rechecks_replacement_when_the_yielded_body_raises(tmp_path: Path) -> None:
    directory = tmp_path / "execution"
    directory.mkdir()
    lock_path = directory / ".state-transition.lock"

    with pytest.raises(SmokeExecutionError, match="transition lock changed"):
        with runner_module._state_transition_lock(directory):
            assert _hostile_replace_fixed_lock(lock_path) is None
            raise RuntimeError("hostile body failure")


@pytest.mark.parametrize(
    "lock_context",
    [runner_module._state_transition_lock, worker_module._state_transition_lock],
)
def test_transition_lock_rejects_a_hard_link_seam(tmp_path: Path, lock_context: Any) -> None:
    directory = tmp_path / "execution"
    directory.mkdir()
    lock_path = directory / ".state-transition.lock"
    lock_path.write_bytes(b"\0")
    os.link(lock_path, directory / "attacker-alias")

    with pytest.raises((SmokeExecutionError, worker_module.SmokeBundleError), match="lock is unsafe"):
        with lock_context(directory):
            raise AssertionError("a multiply linked lock file must never become authoritative")


def test_state_identity_blocks_same_sequence_different_body_aba(state_harness: _StateHarness) -> None:
    current = state_harness.state()
    state_harness.publish(current)
    forged = dict(current)
    forged["logs"] = ["A schema-valid body with the same sequence but different content."]
    forged = state_harness.resign(forged)
    state_harness.runner._validate_state(forged)

    with pytest.raises(SmokeExecutionError, match="changed concurrently"):
        state_harness.runner._transition(forged, status="FAILED", exit_code=9, message="Forged transition.")

    assert state_harness.durable() == current


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_state_validation_requires_the_exact_key_set(state_harness: _StateHarness, mutation: str) -> None:
    value = state_harness.state()
    if mutation == "extra":
        value["unexpected"] = "field"
    else:
        value.pop("owner_pid")
    value = state_harness.resign(value)

    with pytest.raises(SmokeExecutionError, match="invalid"):
        state_harness.runner._validate_state(value)


def test_state_validation_rejects_booleans_in_integer_fields(state_harness: _StateHarness) -> None:
    cases: list[dict[str, Any]] = []
    for key, boolean in (
        ("current", False),
        ("total", True),
        ("owner_pid", True),
        ("transition_sequence", True),
        ("wall_clock_limit_seconds", True),
    ):
        value = state_harness.state()
        value[key] = boolean
        cases.append(state_harness.resign(value))
    worker_boolean = state_harness.state()
    worker_boolean["worker_pid"] = True
    worker_boolean["worker_process"] = {
        **worker_boolean["worker_process"],
        "pid": True,
    }
    cases.append(state_harness.resign(worker_boolean))
    complete = state_harness.state("COMPLETE")
    complete["exit_code"] = False
    cases.append(state_harness.resign(complete))

    for value in cases:
        with pytest.raises(SmokeExecutionError, match="invalid"):
            state_harness.runner._validate_state(value)


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("schema_version", 1),
        ("smoke_id", []),
        ("device", {}),
        ("plan_identity", []),
        ("launch_identity", 1),
        ("status", []),
        ("worker_pid", "40404"),
        ("worker_process", []),
        ("portable_argv", {}),
        ("worker_argv", {}),
        ("argv_identity", 1),
        ("worker_argv_identity", []),
        ("execution_mode", []),
        ("confinement", []),
        ("environment", []),
        ("environment_identity", {}),
        ("interpreter_identity", 1),
        ("writable_roots", {}),
        ("exit_code", False),
        ("receipt_identity", 7),
        ("retry_policy", False),
        ("resumable", 0),
        ("logs", {}),
        (next(iter(runner_module.FALSE_ELIGIBILITY)), 0),
    ],
)
def test_state_validation_requires_exact_top_level_primitive_types(
    state_harness: _StateHarness,
    key: str,
    replacement: Any,
) -> None:
    value = state_harness.state()
    value[key] = replacement
    value = state_harness.resign(value)

    with pytest.raises(SmokeExecutionError, match="invalid"):
        state_harness.runner._validate_state(value)


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("started_at", "not-a-timestamp"),
        ("started_at", "2026-07-17T12:00:00"),
        ("deadline_at", "2026-07-17T12:00:00+01:00"),
        ("updated_at", "2026-07-17T12:00:00Z"),
    ],
)
def test_state_validation_rejects_malformed_or_noncanonical_timestamps(
    state_harness: _StateHarness,
    key: str,
    replacement: str,
) -> None:
    value = state_harness.state()
    value[key] = replacement
    value = state_harness.resign(value)

    with pytest.raises(SmokeExecutionError, match="invalid"):
        state_harness.runner._validate_state(value)


def test_state_validation_requires_the_exact_deadline(state_harness: _StateHarness) -> None:
    value = state_harness.state()
    deadline = datetime.fromisoformat(value["deadline_at"]) + timedelta(microseconds=1)
    value["deadline_at"] = deadline.isoformat()
    value = state_harness.resign(value)

    with pytest.raises(SmokeExecutionError, match="invalid"):
        state_harness.runner._validate_state(value)


def test_state_validation_enforces_status_specific_fields(state_harness: _StateHarness) -> None:
    invalid: list[dict[str, Any]] = []

    starting = state_harness.state("STARTING")
    starting["transition_sequence"] = 1
    invalid.append(state_harness.resign(starting))

    running = state_harness.state("RUNNING")
    running["receipt_identity"] = _digest("early-receipt")
    invalid.append(state_harness.resign(running))

    complete = state_harness.state("COMPLETE")
    complete["worker_pid"] = None
    complete["worker_process"] = None
    invalid.append(state_harness.resign(complete))

    failed = state_harness.state("FAILED")
    failed["exit_code"] = 0
    invalid.append(state_harness.resign(failed))

    cancelled = state_harness.state("CANCELLED")
    cancelled["exit_code"] = 124
    invalid.append(state_harness.resign(cancelled))

    timed_out = state_harness.state("TIMED_OUT")
    timed_out["updated_at"] = timed_out["started_at"]
    invalid.append(state_harness.resign(timed_out))

    for value in invalid:
        with pytest.raises(SmokeExecutionError, match="invalid"):
            state_harness.runner._validate_state(value)


def test_stale_transition_cannot_roll_back_a_newer_nonterminal_state(state_harness: _StateHarness) -> None:
    stale = state_harness.state()
    state_harness.publish(stale)
    newer = state_harness.runner._transition(stale, status="RUNNING", current=1, message="Progress advanced.")

    with pytest.raises(SmokeExecutionError, match="changed concurrently"):
        state_harness.runner._transition(stale, status="RUNNING", message="Stale rollback.")

    assert state_harness.durable() == newer
    assert newer["transition_sequence"] == stale["transition_sequence"] + 1
    assert newer["state_identity"] != stale["state_identity"]


def test_transition_rereads_and_rejects_a_different_postimage(
    state_harness: _StateHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = state_harness.state()
    state_harness.publish(running)
    original_read = state_harness.runner._read_state
    reads = 0

    def changed_postimage(smoke_id: str, device: str, plan: dict[str, Any]) -> dict[str, Any]:
        nonlocal reads
        reads += 1
        value = original_read(smoke_id, device, plan)
        if reads == 2:
            value["logs"] = [*value["logs"], "A different postimage was observed."]
            value = state_harness.resign(value)
            state_harness.runner._validate_state(value)
        return value

    monkeypatch.setattr(state_harness.runner, "_read_state", changed_postimage)

    with pytest.raises(SmokeExecutionError, match="transition changed"):
        state_harness.runner._transition(running, status="RUNNING", current=1, message="Publish progress.")
    assert reads == 2


def test_restart_reconstruction_publishes_one_authenticated_complete_transition(
    state_harness: _StateHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = state_harness.state()
    state_harness.publish(running)
    receipt_identity = _digest("fixture-receipt")

    def no_heartbeat(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise FileNotFoundError

    monkeypatch.setattr(runner_module, "load_device_receipt", lambda *_args: {"receipt_identity": receipt_identity})
    monkeypatch.setattr(runner_module, "load_worker_heartbeat", no_heartbeat)
    monkeypatch.setattr(
        runner_module,
        "load_worker_outcome",
        lambda *_args: {"status": "COMPLETE", "exit_code": 0, "finished_at": datetime.now(timezone.utc).isoformat()},
    )

    restarted = ExploratorySmokeRunner(state_harness.root)
    reconstructed = restarted.status(state_harness.plan["smoke_id"], "cpu")
    durable = state_harness.durable()

    assert reconstructed["status"] == "COMPLETE"
    assert reconstructed["exit_code"] == 0
    assert reconstructed["receipt_identity"] == receipt_identity
    assert durable["status"] == "COMPLETE"
    assert durable["transition_sequence"] == running["transition_sequence"] + 1
    assert durable["state_identity"] != running["state_identity"]
    assert restarted.status(state_harness.plan["smoke_id"], "cpu")["status"] == "COMPLETE"
    assert state_harness.durable() == durable


@pytest.mark.skipif(os.name != "nt", reason="Windows uses a one-process direct-trainer Job boundary.")
def test_windows_restart_without_the_job_owned_trainer_is_interrupted(
    state_harness: _StateHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = state_harness.state()
    limit = int(running["wall_clock_limit_seconds"])
    started = datetime.now(timezone.utc) - timedelta(seconds=runner_module.STARTUP_GRACE_SECONDS + 2)
    running.update(
        {
            "started_at": started.isoformat(),
            "updated_at": (started + timedelta(seconds=1)).isoformat(),
            "deadline_at": (started + timedelta(seconds=limit)).isoformat(),
        }
    )
    running = state_harness.resign(running)
    state_harness.runner._validate_state(running)
    state_harness.publish(running)

    def absent(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise FileNotFoundError

    monkeypatch.setattr(runner_module, "load_device_receipt", absent)
    monkeypatch.setattr(runner_module, "load_worker_heartbeat", absent)
    monkeypatch.setattr(runner_module, "load_worker_outcome", absent)
    monkeypatch.setattr(runner_module, "worker_process_matches", lambda _value: False)

    restarted = ExploratorySmokeRunner(state_harness.root)
    result = restarted.status(state_harness.plan["smoke_id"], "cpu")

    assert result["status"] == "INTERRUPTED"
    assert state_harness.durable()["status"] == "INTERRUPTED"


def test_complete_requires_the_exact_receipt_and_terminal_retries_must_be_compatible(
    state_harness: _StateHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = state_harness.state()
    state_harness.publish(running)
    receipt_identity = _digest("fixture-receipt")
    monkeypatch.setattr(runner_module, "load_device_receipt", lambda *_args: {"receipt_identity": receipt_identity})

    with pytest.raises(SmokeExecutionError, match="receipt identity changed"):
        state_harness.runner._transition(
            running,
            status="COMPLETE",
            current=2,
            exit_code=0,
            receipt_identity=_digest("different-receipt"),
            message="A mismatched receipt must not publish.",
        )
    assert state_harness.durable() == running

    complete = state_harness.runner._transition(
        running,
        status="COMPLETE",
        current=2,
        exit_code=0,
        receipt_identity=receipt_identity,
        message="The exact receipt may publish.",
    )
    repeated = state_harness.runner._transition(
        complete,
        status="COMPLETE",
        current=2,
        exit_code=0,
        receipt_identity=receipt_identity,
        message="An exact terminal retry is idempotent.",
    )
    assert repeated == complete
    assert state_harness.durable() == complete

    with pytest.raises(SmokeExecutionError, match="terminal state already differs"):
        state_harness.runner._transition(
            complete,
            status="COMPLETE",
            current=2,
            exit_code=0,
            receipt_identity=_digest("different-receipt"),
            message="A conflicting duplicate is not idempotent.",
        )
    with pytest.raises(SmokeExecutionError, match="terminal state already differs"):
        state_harness.runner._transition(
            complete,
            status="CANCELLED",
            exit_code=130,
            message="A terminal state is immutable.",
        )
    assert state_harness.durable() == complete


@pytest.mark.parametrize("competing_status", ["CANCELLED", "TIMED_OUT"])
def test_complete_and_stop_race_has_one_exact_terminal_winner(
    state_harness: _StateHarness,
    monkeypatch: pytest.MonkeyPatch,
    competing_status: str,
) -> None:
    running = state_harness.state()
    if competing_status == "TIMED_OUT":
        limit = int(running["wall_clock_limit_seconds"])
        started = datetime.now(timezone.utc) - timedelta(seconds=limit + 2)
        deadline = started + timedelta(seconds=limit)
        running["started_at"] = started.isoformat()
        running["deadline_at"] = deadline.isoformat()
        running["updated_at"] = (deadline - timedelta(seconds=1)).isoformat()
        running = state_harness.resign(running)
        state_harness.runner._validate_state(running)
    state_harness.publish(running)
    other_runner = ExploratorySmokeRunner(state_harness.root)
    barrier = Barrier(2)
    receipt_identity = _digest("fixture-receipt")
    monkeypatch.setattr(runner_module, "load_device_receipt", lambda *_args: {"receipt_identity": receipt_identity})

    def complete() -> dict[str, Any]:
        barrier.wait()
        return state_harness.runner._transition(
            running,
            status="COMPLETE",
            current=2,
            exit_code=0,
            receipt_identity=receipt_identity,
            message="Completion won or lost atomically.",
        )

    def stop() -> dict[str, Any]:
        barrier.wait()
        return other_runner._transition(
            running,
            status=competing_status,
            exit_code=130 if competing_status == "CANCELLED" else 124,
            message="Cancellation or timeout won or lost atomically.",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(complete), pool.submit(stop)]
        terminal: list[dict[str, Any]] = []
        rejected: list[SmokeExecutionError] = []
        for future in futures:
            try:
                terminal.append(future.result(timeout=10))
            except SmokeExecutionError as exc:
                rejected.append(exc)

    durable = state_harness.durable()
    assert durable["status"] in {"COMPLETE", competing_status}
    assert len(terminal) == 1
    assert len(rejected) == 1
    assert "changed concurrently" in str(rejected[0])
    assert terminal[0]["status"] == durable["status"]
    assert terminal[0]["state_identity"] == durable["state_identity"]
    assert durable["transition_sequence"] == running["transition_sequence"] + 1


def test_restart_cancellation_does_not_publish_before_recorded_exit_is_proven(
    state_harness: _StateHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = state_harness.state()
    state_harness.publish(running)
    monkeypatch.setattr(runner_module, "_recalled_process", lambda *_args: None)

    def unverifiable(_state: dict[str, Any]) -> None:
        raise SmokeExecutionError("smoke_process_termination", "Fixture termination was not proven.")

    monkeypatch.setattr(runner_module, "_terminate_recorded_worker", unverifiable)

    with pytest.raises(SmokeExecutionError, match="not proven"):
        state_harness.runner.cancel(
            state_harness.plan["smoke_id"],
            state_harness.plan["plan_identity"],
            "cpu",
            explicit_action=True,
        )

    assert state_harness.durable() == running


@pytest.mark.skipif(os.name != "nt", reason="Windows pointer-sized process handles are platform-specific.")
def test_windows_recorded_termination_uses_pointer_sized_handle_and_proves_exit() -> None:
    import ctypes
    from ctypes import wintypes

    class _Function:
        def __init__(self, implementation: Any) -> None:
            self.implementation = implementation
            self.argtypes: Any = None
            self.restype: Any = None

        def __call__(self, *args: Any) -> Any:
            return self.implementation(*args)

    high_handle = 0x1_0000_1234
    pid = 4242
    birth = 0x12345678
    image = os.path.normcase(r"C:\\fixture\\trainer.exe")
    observed_handles: list[int] = []
    waits: list[int] = []
    terminated: list[int] = []
    closed: list[int] = []

    def value(handle: Any) -> int:
        return int(getattr(handle, "value", handle))

    def times(handle: Any, creation: Any, *_rest: Any) -> bool:
        observed_handles.append(value(handle))
        creation._obj.dwLowDateTime = birth
        creation._obj.dwHighDateTime = 0
        return True

    def image_name(handle: Any, _flags: int, buffer: Any, length: Any) -> bool:
        observed_handles.append(value(handle))
        buffer.value = image
        length._obj.value = len(image)
        return True

    def wait(handle: Any, milliseconds: int) -> int:
        observed_handles.append(value(handle))
        waits.append(int(milliseconds))
        return 0x102 if int(milliseconds) == 0 else 0

    def exit_code(handle: Any, result: Any) -> bool:
        observed_handles.append(value(handle))
        result._obj.value = 130
        return True

    api = SimpleNamespace(
        OpenProcess=_Function(lambda *_args: high_handle),
        GetProcessId=_Function(lambda handle: observed_handles.append(value(handle)) or pid),
        GetProcessTimes=_Function(times),
        QueryFullProcessImageNameW=_Function(image_name),
        TerminateProcess=_Function(lambda handle, _code: terminated.append(value(handle)) or True),
        WaitForSingleObject=_Function(wait),
        GetExitCodeProcess=_Function(exit_code),
        CloseHandle=_Function(lambda handle: closed.append(value(handle)) or True),
    )
    state = {
        "worker_pid": pid,
        "worker_process": {
            "pid": pid,
            "birth_token": str(birth),
            "process_image_path_sha256": hashlib.sha256(image.encode("utf-8", "surrogatepass")).hexdigest(),
        },
    }

    runner_module._terminate_recorded_windows_worker(state, kernel32=api)

    assert api.OpenProcess.restype is wintypes.HANDLE
    assert ctypes.sizeof(wintypes.HANDLE) == ctypes.sizeof(ctypes.c_void_p)
    assert waits == [0, 5_000]
    assert terminated == [high_handle]
    assert closed == [high_handle]
    assert observed_handles and set(observed_handles) == {high_handle}
