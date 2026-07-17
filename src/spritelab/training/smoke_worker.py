"""Contained worker for one server-owned exploratory smoke trainer."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.training.smoke_bundle import (
    SHA256_PATTERN,
    SmokeBundleError,
    anchored_directory,
    artifact_bundle_directory,
    canonical_json_bytes,
    finalize_identity,
    load_device_receipt,
    load_plan,
    pinned_smoke_interpreter,
    read_stable_single_link_bytes,
    smoke_launch_identity,
    smoke_training_argv,
    validate_identity,
    validate_smoke_environment,
    validate_smoke_interpreter,
    verify_execution_guards,
    verify_pinned_process_image,
    write_exclusive_bytes,
)
from spritelab.utils.pinned_executable import (
    activate_windows_suspended_process,
    close_windows_handle,
    linux_parent_death_signal,
)

SMOKE_WORKER_HEARTBEAT_SCHEMA = "spritelab.training.smoke-worker-heartbeat.v1"
SMOKE_WORKER_OUTCOME_SCHEMA = "spritelab.training.smoke-worker-outcome.v1"
HEARTBEAT_INTERVAL_SECONDS = 1.0
HEARTBEAT_FRESH_SECONDS = 8.0
STARTUP_GRACE_SECONDS = 12.0


def heartbeat_path(project_root: Path, smoke_id: str, device: str) -> Path:
    return artifact_bundle_directory(project_root, smoke_id) / "execution" / f"{device}.heartbeat.json"


def outcome_path(project_root: Path, smoke_id: str, device: str) -> Path:
    return artifact_bundle_directory(project_root, smoke_id) / "execution" / f"{device}.outcome.json"


def load_worker_heartbeat(
    project_root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _read_json(heartbeat_path(project_root, str(plan["smoke_id"]), device), project_root)
    if (
        value.get("schema_version") != SMOKE_WORKER_HEARTBEAT_SCHEMA
        or value.get("smoke_id") != plan["smoke_id"]
        or value.get("device") != device
        or value.get("plan_identity") != plan["plan_identity"]
        or value.get("launch_identity") != launch_identity
        or value.get("status") not in {"STARTING", "RUNNING"}
        or type(value.get("sequence")) is not int
        or int(value["sequence"]) < 1
        or type(value.get("worker_pid")) is not int
        or int(value["worker_pid"]) <= 0
        or not isinstance(value.get("worker_process"), Mapping)
        or value.get("containment") not in {"WINDOWS_JOB", "LINUX_PDEATHSIG"}
    ):
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat is invalid.")
    validate_identity(value, "heartbeat_identity")
    if not _valid_process_identity(value["worker_process"], int(value["worker_pid"])):
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker process identity is invalid.")
    _parse_time(str(value.get("worker_started_at") or ""))
    _parse_time(str(value.get("heartbeat_at") or ""))
    return value


def load_worker_outcome(
    project_root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _read_json(outcome_path(project_root, str(plan["smoke_id"]), device), project_root)
    if (
        value.get("schema_version") != SMOKE_WORKER_OUTCOME_SCHEMA
        or value.get("smoke_id") != plan["smoke_id"]
        or value.get("device") != device
        or value.get("plan_identity") != plan["plan_identity"]
        or value.get("launch_identity") != launch_identity
        or value.get("status") not in {"COMPLETE", "FAILED"}
        or type(value.get("exit_code")) is not int
        or not SHA256_PATTERN.fullmatch(str(value.get("last_heartbeat_identity") or ""))
    ):
        raise SmokeBundleError("smoke_worker_outcome", "The smoke worker outcome is invalid.")
    validate_identity(value, "outcome_identity")
    _parse_time(str(value.get("finished_at") or ""))
    return value


def heartbeat_is_fresh(value: Mapping[str, Any], *, now: datetime | None = None) -> bool:
    heartbeat = _parse_time(str(value.get("heartbeat_at") or ""))
    current = now or datetime.now(timezone.utc)
    return -1.0 <= (current - heartbeat).total_seconds() <= HEARTBEAT_FRESH_SECONDS


def worker_process_matches(value: Mapping[str, Any]) -> bool | None:
    marker = value.get("worker_process")
    pid = value.get("worker_pid")
    if not isinstance(marker, Mapping) or type(pid) is not int:
        return False
    if str(marker.get("birth_token") or "").startswith("unsupported-"):
        return None
    current = _process_identity(pid)
    if current is None:
        return False if smoke_containment_supported() else None
    return dict(marker) == current


def capture_worker_process_identity(pid: int) -> dict[str, Any] | None:
    return _process_identity(pid)


def worker_process_identity_is_valid(value: Any, pid: int) -> bool:
    return _valid_process_identity(value, pid)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-id", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--plan-identity", required=True)
    parser.add_argument("--launch-identity", required=True)
    return parser.parse_args()


def main() -> int:
    parsed = _parse_args()
    root = Path.cwd().resolve()
    plan = load_plan(root, parsed.smoke_id)
    if plan["plan_identity"] != parsed.plan_identity:
        return 64
    expected_launch = smoke_launch_identity(plan, parsed.device)
    if parsed.launch_identity != expected_launch:
        return 64
    validate_smoke_environment(root, plan, parsed.device, os.environ)
    validate_smoke_interpreter(
        plan,
        lexical_path=os.environ.get("SPRITELAB_BOUND_INTERPRETER"),
    )
    if not smoke_containment_supported():
        return 70
    verify_execution_guards(root, plan)
    started_at = _now()
    worker_process = _process_identity(os.getpid())
    if worker_process is None:
        return 70
    sequence = 0
    last_heartbeat: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    containment: _Containment | None = None
    try:
        deadline = time.monotonic() + STARTUP_GRACE_SECONDS
        while True:
            sequence += 1
            last_heartbeat = _publish_heartbeat(
                root,
                plan,
                parsed.device,
                expected_launch,
                started_at,
                sequence,
                worker_process,
                status="STARTING",
                containment=_containment_name(),
            )
            state = _read_execution_state(root, plan, parsed.device, expected_launch)
            if state.get("status") == "RUNNING":
                break
            if state.get("status") in {"FAILED", "INTERRUPTED"} or time.monotonic() >= deadline:
                return _finish(root, plan, parsed.device, expected_launch, last_heartbeat, 70)
            time.sleep(0.1)

        verify_execution_guards(root, plan)
        training = smoke_training_argv(plan, parsed.device)
        with pinned_smoke_interpreter(
            plan,
            lexical_path=os.environ.get("SPRITELAB_BOUND_INTERPRETER"),
        ) as interpreter:
            argv = [interpreter.launch_path, *training[1:]]
            options: dict[str, Any] = {
                "cwd": root,
                "env": dict(os.environ),
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "shell": False,
            }
            if os.name == "nt":
                options["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | 0x00000004
            else:
                options["start_new_session"] = True
                options["pass_fds"] = interpreter.pass_fds
                if sys.platform.startswith("linux"):
                    options["preexec_fn"] = linux_parent_death_signal(os.getpid())
            process = subprocess.Popen(argv, **options)
            containment = _Containment(process)
            containment.activate(verifier=lambda launched: verify_pinned_process_image(launched, interpreter))
        while process.poll() is None:
            sequence += 1
            last_heartbeat = _publish_heartbeat(
                root,
                plan,
                parsed.device,
                expected_launch,
                started_at,
                sequence,
                worker_process,
                status="RUNNING",
                containment=containment.name,
            )
            time.sleep(HEARTBEAT_INTERVAL_SECONDS)
        code = int(process.wait())
        if code == 0:
            try:
                load_device_receipt(root, plan, parsed.device)
            except (OSError, ValueError, SmokeBundleError):
                code = 70
        if last_heartbeat is None:
            return 70
        return _finish(root, plan, parsed.device, expected_launch, last_heartbeat, code)
    except BaseException:
        if containment is not None:
            containment.terminate()
        elif process is not None:
            _terminate_process(process)
        if last_heartbeat is not None:
            try:
                return _finish(root, plan, parsed.device, expected_launch, last_heartbeat, 70)
            except BaseException:
                pass
        return 70
    finally:
        if containment is not None:
            containment.close()


def _read_execution_state(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    path = artifact_bundle_directory(root, str(plan["smoke_id"])) / "execution" / f"{device}.json"
    value = _read_json(path, root)
    if (
        value.get("schema_version") != "spritelab.training.smoke-device-execution.v1"
        or value.get("smoke_id") != plan["smoke_id"]
        or value.get("device") != device
        or value.get("plan_identity") != plan["plan_identity"]
        or value.get("launch_identity") != launch_identity
        or value.get("worker_process") != _process_identity(os.getpid())
    ):
        raise SmokeBundleError("smoke_execution_state", "The smoke execution state is invalid.")
    return value


def _publish_heartbeat(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
    started_at: str,
    sequence: int,
    worker_process: Mapping[str, Any],
    *,
    status: str,
    containment: str,
) -> dict[str, Any]:
    value = finalize_identity(
        {
            "schema_version": SMOKE_WORKER_HEARTBEAT_SCHEMA,
            "smoke_id": plan["smoke_id"],
            "device": device,
            "plan_identity": plan["plan_identity"],
            "launch_identity": launch_identity,
            "status": status,
            "worker_pid": os.getpid(),
            "worker_process": dict(worker_process),
            "worker_started_at": started_at,
            "heartbeat_at": _now(),
            "sequence": sequence,
            "containment": containment,
            "message": "Contained smoke worker is active.",
        },
        "heartbeat_identity",
    )
    path = heartbeat_path(root, str(plan["smoke_id"]), device)
    with anchored_directory(path.parent, root) as anchor:
        anchor.atomic_write_bytes(path.name, canonical_json_bytes(value, pretty=True))
    return value


def _finish(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
    heartbeat: Mapping[str, Any],
    exit_code: int,
) -> int:
    status = "COMPLETE" if exit_code == 0 else "FAILED"
    value = finalize_identity(
        {
            "schema_version": SMOKE_WORKER_OUTCOME_SCHEMA,
            "smoke_id": plan["smoke_id"],
            "device": device,
            "plan_identity": plan["plan_identity"],
            "launch_identity": launch_identity,
            "status": status,
            "exit_code": int(exit_code),
            "finished_at": _now(),
            "last_heartbeat_identity": heartbeat["heartbeat_identity"],
            "message": "Contained smoke worker completed."
            if status == "COMPLETE"
            else "Contained smoke worker failed.",
        },
        "outcome_identity",
    )
    path = outcome_path(root, str(plan["smoke_id"]), device)
    try:
        write_exclusive_bytes(path, canonical_json_bytes(value, pretty=True), boundary=root)
    except FileExistsError:
        existing = load_worker_outcome(root, plan, device, launch_identity)
        if existing != value:
            return 70
    return int(exit_code)


def _read_json(path: Path, root: Path) -> dict[str, Any]:
    payload = read_stable_single_link_bytes(path, boundary=root, max_bytes=4 * 1024 * 1024)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeBundleError("smoke_worker_state", "The smoke worker state is invalid.") from exc
    if not isinstance(value, dict):
        raise SmokeBundleError("smoke_worker_state", "The smoke worker state is invalid.")
    return value


class _Containment:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.name = _containment_name()
        self._job_handle: int | None = None

    def activate(self, *, verifier: Any | None = None) -> None:
        if os.name == "nt":
            self._job_handle = activate_windows_suspended_process(
                self.process,
                verifier=verifier,
            )
        elif verifier is not None:
            verifier(self.process)
        _install_termination_handlers(self)

    def terminate(self) -> None:
        if self.process.poll() is not None:
            return
        if os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        _terminate_process(self.process)

    def close(self) -> None:
        if self._job_handle:
            close_windows_handle(self._job_handle)
            self._job_handle = None


def _containment_name() -> str:
    if os.name == "nt":
        return "WINDOWS_JOB"
    if sys.platform.startswith("linux"):
        return "LINUX_PDEATHSIG"
    raise SmokeBundleError("smoke_containment", "Server-run smoke containment is unavailable on this platform.")


def smoke_containment_supported() -> bool:
    return os.name == "nt" or sys.platform.startswith("linux")


def _install_termination_handlers(containment: _Containment) -> None:
    def terminate(_signum: int, _frame: Any) -> None:
        containment.terminate()
        raise SystemExit(143)

    for name in ("SIGTERM", "SIGINT"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, terminate)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except BaseException:
        try:
            process.kill()
            process.wait(timeout=5)
        except BaseException:
            return


def _process_identity(pid: int) -> dict[str, Any] | None:
    if pid <= 0:
        return None
    if os.name == "nt":
        return _windows_process_identity(pid)
    if sys.platform.startswith("linux"):
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            remainder = stat_text.rsplit(") ", 1)[1].split()
            start_ticks = remainder[19]
            executable = os.readlink(f"/proc/{pid}/exe")
        except (OSError, IndexError, UnicodeError):
            return None
        return {
            "pid": pid,
            "birth_token": start_ticks,
            "process_image_path_sha256": hashlib.sha256(executable.encode("utf-8", "surrogatepass")).hexdigest(),
        }
    if pid == os.getpid():
        return {
            "pid": pid,
            "birth_token": f"unsupported-{time.monotonic_ns()}",
            "process_image_path_sha256": hashlib.sha256(sys.executable.encode("utf-8", "surrogatepass")).hexdigest(),
        }
    return None


def _windows_process_identity(pid: int) -> dict[str, Any] | None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        length = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(length.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            return None
        birth = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        executable = os.path.normcase(buffer.value)
        return {
            "pid": pid,
            "birth_token": str(birth),
            "process_image_path_sha256": hashlib.sha256(executable.encode("utf-8", "surrogatepass")).hexdigest(),
        }
    finally:
        kernel32.CloseHandle(handle)


def _valid_process_identity(value: Any, pid: int) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("pid") == pid
        and isinstance(value.get("birth_token"), str)
        and bool(value["birth_token"])
        and SHA256_PATTERN.fullmatch(str(value.get("process_image_path_sha256") or "")) is not None
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SmokeBundleError("smoke_worker_time", "The smoke worker timestamp is invalid.") from exc
    if parsed.tzinfo is None:
        raise SmokeBundleError("smoke_worker_time", "The smoke worker timestamp is invalid.")
    return parsed.astimezone(timezone.utc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HEARTBEAT_FRESH_SECONDS",
    "SMOKE_WORKER_HEARTBEAT_SCHEMA",
    "SMOKE_WORKER_OUTCOME_SCHEMA",
    "STARTUP_GRACE_SECONDS",
    "capture_worker_process_identity",
    "heartbeat_is_fresh",
    "load_worker_heartbeat",
    "load_worker_outcome",
    "smoke_containment_supported",
    "worker_process_identity_is_valid",
    "worker_process_matches",
]
