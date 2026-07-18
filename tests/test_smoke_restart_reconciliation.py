from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest

import spritelab.training.smoke_runner as runner
import spritelab.training.smoke_worker as worker


def _process_marker(pid: int = 4242) -> dict[str, Any]:
    return {
        "worker_pid": pid,
        "worker_process": {
            "pid": pid,
            "birth_token": "fixture-birth-token",
            "process_image_path_sha256": "a" * 64,
        },
    }


def test_worker_match_distinguishes_definitive_exit_from_unknown_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _process_marker()
    monkeypatch.setattr(worker, "_process_identity", lambda _pid: None)
    monkeypatch.setattr(worker, "_process_is_definitively_absent", lambda _pid: True)
    assert worker.worker_process_matches(state) is False

    monkeypatch.setattr(worker, "_process_is_definitively_absent", lambda _pid: False)
    assert worker.worker_process_matches(state) is None


def test_departed_windows_direct_trainer_terminalizes_without_signaling_reused_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {**_process_marker(), "execution_mode": "windows-direct-trainer-v1"}
    monkeypatch.setattr(runner, "worker_process_matches", lambda _state: False)
    monkeypatch.setattr(
        runner,
        "_terminate_recorded_windows_worker",
        lambda _state: pytest.fail("a departed or reused Windows PID must never be signaled"),
    )
    monkeypatch.setattr(
        runner,
        "_terminate_recorded_linux_process_group",
        lambda _state: pytest.fail("Windows direct-trainer recovery must not dispatch to POSIX"),
    )

    runner._terminate_recorded_worker(state)


def test_unknown_windows_direct_trainer_identity_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {**_process_marker(), "execution_mode": "windows-direct-trainer-v1"}
    monkeypatch.setattr(runner, "worker_process_matches", lambda _state: None)

    with pytest.raises(runner.SmokeExecutionError, match="could not be verified"):
        runner._terminate_recorded_worker(state)


@pytest.mark.skipif(os.name != "nt", reason="Windows process-exit identity is platform-specific.")
def test_windows_exited_process_no_longer_matches_its_captured_identity() -> None:
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        marker = worker.capture_worker_process_identity(process.pid)
        assert marker is not None
        state = {"worker_pid": process.pid, "worker_process": marker}
        process.terminate()
        process.wait(timeout=10)
        assert worker.worker_process_matches(state) is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
