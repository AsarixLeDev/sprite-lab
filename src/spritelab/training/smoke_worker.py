"""Contained worker for one server-owned exploratory smoke trainer."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from spritelab.training.smoke_bundle import (
    FALSE_ELIGIBILITY,
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
    smoke_worker_argv,
    stable_hash,
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
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity

SMOKE_WORKER_HEARTBEAT_SCHEMA = "spritelab.training.smoke-worker-heartbeat.v2"
SMOKE_WORKER_OUTCOME_SCHEMA = "spritelab.training.smoke-worker-outcome.v2"
HEARTBEAT_INTERVAL_SECONDS = 1.0
HEARTBEAT_FRESH_SECONDS = 8.0
STARTUP_GRACE_SECONDS = 12.0
_WORKER_TERMINAL_EXIT_CODES = {"CANCELLED": 130, "TIMED_OUT": 124}
_EXECUTION_STATE_SCHEMA = "spritelab.training.smoke-device-execution.v1"
_EXECUTION_TERMINAL = {"COMPLETE", "FAILED", "INTERRUPTED", "CANCELLED", "TIMED_OUT"}
_EXECUTION_STATUSES = {"STARTING", "RUNNING", *_EXECUTION_TERMINAL}
_NONE_TYPE = type(None)
_PROCESS_IDENTITY_KEYS = frozenset({"pid", "birth_token", "process_image_path_sha256"})
_EXECUTION_STATE_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "schema_version": (str,),
    "smoke_id": (str,),
    "device": (str,),
    "plan_identity": (str,),
    "launch_identity": (str,),
    "status": (str,),
    "current": (int,),
    "total": (int,),
    "owner_pid": (int,),
    "worker_pid": (int, _NONE_TYPE),
    "worker_process": (dict, _NONE_TYPE),
    "portable_argv": (list,),
    "worker_argv": (list,),
    "argv_identity": (str,),
    "worker_argv_identity": (str,),
    "execution_mode": (str,),
    "confinement": (dict, _NONE_TYPE),
    "environment": (dict,),
    "environment_identity": (str,),
    "interpreter_identity": (str,),
    "writable_roots": (list,),
    "started_at": (str,),
    "deadline_at": (str,),
    "wall_clock_limit_seconds": (int,),
    "updated_at": (str,),
    "exit_code": (int, _NONE_TYPE),
    "receipt_identity": (str, _NONE_TYPE),
    "transition_sequence": (int,),
    "retry_policy": (str,),
    "resumable": (bool,),
    "logs": (list,),
    "state_identity": (str,),
    **dict.fromkeys(FALSE_ELIGIBILITY, (bool,)),
}
_EXECUTION_STATE_KEYS = frozenset(_EXECUTION_STATE_FIELD_TYPES)
_HEARTBEAT_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "schema_version": (str,),
    "smoke_id": (str,),
    "device": (str,),
    "plan_identity": (str,),
    "launch_identity": (str,),
    "execution_state_sequence": (int,),
    "execution_state_identity": (str,),
    "status": (str,),
    "worker_pid": (int,),
    "worker_process": (dict,),
    "worker_started_at": (str,),
    "heartbeat_at": (str,),
    "sequence": (int,),
    "previous_heartbeat_identity": (str, _NONE_TYPE),
    "containment": (str,),
    "message": (str,),
    "heartbeat_identity": (str,),
}
_HEARTBEAT_KEYS = frozenset(_HEARTBEAT_FIELD_TYPES)
_OUTCOME_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "schema_version": (str,),
    "smoke_id": (str,),
    "device": (str,),
    "plan_identity": (str,),
    "launch_identity": (str,),
    "execution_state_sequence": (int,),
    "execution_state_identity": (str,),
    "worker_pid": (int,),
    "worker_process": (dict,),
    "status": (str,),
    "exit_code": (int,),
    "finished_at": (str,),
    "last_heartbeat_sequence": (int,),
    "last_heartbeat_identity": (str,),
    "message": (str,),
    "outcome_identity": (str,),
}
_OUTCOME_KEYS = frozenset(_OUTCOME_FIELD_TYPES)


class _WorkerTerminal(RuntimeError):
    def __init__(self, status: str) -> None:
        if status not in _WORKER_TERMINAL_EXIT_CODES:
            raise ValueError("The worker terminal status is invalid.")
        super().__init__(status)
        self.status = status

    @property
    def exit_code(self) -> int:
        return _WORKER_TERMINAL_EXIT_CODES[self.status]


def heartbeat_path(project_root: Path, smoke_id: str, device: str) -> Path:
    return artifact_bundle_directory(project_root, smoke_id) / "execution" / device / "heartbeat.json"


def outcome_path(project_root: Path, smoke_id: str, device: str) -> Path:
    return artifact_bundle_directory(project_root, smoke_id) / "execution" / device / "outcome.json"


def load_worker_heartbeat(
    project_root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _load_worker_heartbeat_record(project_root, plan, device, launch_identity)
    state = _load_execution_state(project_root, plan, device, launch_identity)
    if (
        value["execution_state_sequence"] != state["transition_sequence"]
        or value["execution_state_identity"] != state["state_identity"]
        or value["status"] != state["status"]
        or (
            state["status"] == "RUNNING"
            and (value["worker_pid"] != state["worker_pid"] or value["worker_process"] != state["worker_process"])
        )
    ):
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat is stale.")
    return value


def load_worker_outcome(
    project_root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _load_worker_outcome_record(project_root, plan, device, launch_identity)
    state = _load_execution_state(project_root, plan, device, launch_identity)
    heartbeat = _load_worker_heartbeat_record(project_root, plan, device, launch_identity)
    finished_at = _parse_utc(value["finished_at"])
    heartbeat_at = _parse_utc(heartbeat["heartbeat_at"])
    if (
        value["execution_state_sequence"] != state["transition_sequence"]
        or value["execution_state_identity"] != state["state_identity"]
        or value["worker_pid"] != state["worker_pid"]
        or value["worker_process"] != state["worker_process"]
        or value["last_heartbeat_sequence"] != heartbeat["sequence"]
        or value["last_heartbeat_identity"] != heartbeat["heartbeat_identity"]
        or value["worker_pid"] != heartbeat["worker_pid"]
        or value["worker_process"] != heartbeat["worker_process"]
        or finished_at is None
        or heartbeat_at is None
        or finished_at < heartbeat_at
    ):
        raise SmokeBundleError("smoke_worker_outcome", "The smoke worker outcome is stale.")
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
        return False if _process_is_definitively_absent(pid) else None
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
    validation_environment = dict(os.environ)
    writable_fd_value = validation_environment.pop("SPRITELAB_WRITABLE_ROOT_FDS", None)
    last_heartbeat: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    containment: _Containment | None = None
    execution_deadline: datetime | None = None
    monotonic_deadline: float | None = None
    try:
        writable_root_fds = _parse_writable_root_fds(writable_fd_value)
    except SmokeBundleError:
        return 70
    try:
        validate_smoke_environment(root, plan, parsed.device, validation_environment)
        validate_smoke_interpreter(
            plan,
            lexical_path=os.environ.get("SPRITELAB_BOUND_INTERPRETER"),
        )
        if not smoke_containment_supported():
            return 70
        started_at = _now()
        worker_process = _process_identity(os.getpid())
        if worker_process is None:
            return 70
        sequence = 0

        def operation_check() -> None:
            state = _read_execution_state(root, plan, parsed.device, expected_launch)
            status = str(state.get("status") or "")
            if status == "CANCELLED":
                raise _WorkerTerminal("CANCELLED")
            state_deadline = _parse_time(str(state.get("deadline_at") or ""))
            if (
                status == "TIMED_OUT"
                or datetime.now(timezone.utc) >= state_deadline
                or (monotonic_deadline is not None and time.monotonic() >= monotonic_deadline)
            ):
                raise _WorkerTerminal("TIMED_OUT")
            if status != "RUNNING":
                raise SmokeBundleError("smoke_execution_state", "The smoke execution is no longer running.")

        startup_deadline = time.monotonic() + STARTUP_GRACE_SECONDS
        while True:
            state = _read_execution_state(root, plan, parsed.device, expected_launch)
            execution_deadline = _parse_time(str(state.get("deadline_at") or ""))
            remaining = (execution_deadline - datetime.now(timezone.utc)).total_seconds()
            monotonic_deadline = time.monotonic() + max(0.0, remaining)
            if state.get("status") in {"CANCELLED", "TIMED_OUT"}:
                code = 130 if state["status"] == "CANCELLED" else 124
                if last_heartbeat is None:
                    return code
                return _finish(
                    root,
                    plan,
                    parsed.device,
                    expected_launch,
                    last_heartbeat,
                    code,
                    status_override=str(state["status"]),
                )
            if state.get("status") in {"FAILED", "INTERRUPTED"}:
                return 70
            if time.monotonic() >= startup_deadline:
                return 70
            if remaining <= 0:
                return 124
            next_sequence = sequence + 1
            try:
                last_heartbeat = _publish_heartbeat(
                    root,
                    plan,
                    parsed.device,
                    expected_launch,
                    started_at,
                    next_sequence,
                    worker_process,
                    status=str(state["status"]),
                    containment=_containment_name(),
                    execution_state=state,
                )
            except SmokeBundleError as exc:
                if exc.code == "smoke_worker_state_race":
                    continue
                raise
            sequence = next_sequence
            if state.get("status") == "RUNNING":
                break
            time.sleep(0.1)

        operation_check()
        verify_execution_guards(root, plan, operation_check=operation_check)
        operation_check()
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
                options["pass_fds"] = tuple(sorted({*interpreter.pass_fds, *writable_root_fds}))
                if sys.platform.startswith("linux"):
                    options["preexec_fn"] = linux_parent_death_signal(os.getpid())
            operation_check()
            process = subprocess.Popen(argv, **options)
            containment = _Containment(process)
            containment.activate(verifier=lambda launched: verify_pinned_process_image(launched, interpreter))
            operation_check()
        if execution_deadline is None or monotonic_deadline is None:
            raise SmokeBundleError("smoke_deadline", "The smoke wall-clock deadline is unavailable.")
        while process.poll() is None:
            operation_check()
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
        operation_check()
        if code == 0:
            try:
                load_device_receipt(root, plan, parsed.device, operation_check=operation_check)
            except (OSError, ValueError, SmokeBundleError):
                code = 70
        if last_heartbeat is None:
            return 70
        operation_check()
        return _finish(
            root,
            plan,
            parsed.device,
            expected_launch,
            last_heartbeat,
            code,
            operation_check=operation_check,
        )
    except _WorkerTerminal as terminal:
        if containment is not None:
            containment.terminate()
        elif process is not None:
            _terminate_process(process)
        if last_heartbeat is not None:
            try:
                return _finish(
                    root,
                    plan,
                    parsed.device,
                    expected_launch,
                    last_heartbeat,
                    terminal.exit_code,
                    status_override=terminal.status,
                )
            except BaseException:
                pass
        return terminal.exit_code
    except BaseException:
        if containment is not None:
            containment.terminate()
        elif process is not None:
            _terminate_process(process)
        if last_heartbeat is not None:
            try:
                operation_check()
            except _WorkerTerminal as terminal:
                try:
                    return _finish(
                        root,
                        plan,
                        parsed.device,
                        expected_launch,
                        last_heartbeat,
                        terminal.exit_code,
                        status_override=terminal.status,
                    )
                except BaseException:
                    pass
                return terminal.exit_code
            except BaseException:
                pass
            try:
                return _finish(root, plan, parsed.device, expected_launch, last_heartbeat, 70)
            except BaseException:
                pass
        return 70
    finally:
        if containment is not None:
            containment.close()
        for descriptor in writable_root_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _parse_writable_root_fds(value: str | None) -> tuple[int, ...]:
    if not sys.platform.startswith("linux"):
        if value is not None:
            raise SmokeBundleError("smoke_containment", "Unexpected writable-root descriptors were supplied.")
        return ()
    try:
        descriptors = tuple(int(item) for item in str(value).split(","))
    except ValueError as exc:
        raise SmokeBundleError("smoke_containment", "Writable-root descriptors are malformed.") from exc
    if len(descriptors) != 2 or len(set(descriptors)) != 2 or any(descriptor <= 2 for descriptor in descriptors):
        raise SmokeBundleError("smoke_containment", "Writable-root descriptors are malformed.")
    try:
        if any(not os.path.isdir(f"/proc/self/fd/{descriptor}") for descriptor in descriptors):
            raise SmokeBundleError("smoke_containment", "A writable-root descriptor is unavailable.")
    except OSError as exc:
        raise SmokeBundleError("smoke_containment", "A writable-root descriptor is unavailable.") from exc
    return descriptors


def _read_execution_state(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _load_execution_state(root, plan, device, launch_identity)
    expected_worker = _process_identity(os.getpid())
    worker_process = value["worker_process"]
    if expected_worker is None or (worker_process is not None and worker_process != expected_worker):
        raise SmokeBundleError("smoke_execution_state", "The smoke execution state is invalid.")
    return value


def _load_execution_state(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _read_json(_execution_state_path(root, plan, device), root)
    _validate_execution_state(value, root, plan, device, launch_identity)
    return value


def _validate_execution_state(
    value: Mapping[str, Any],
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> None:
    if (
        set(value) != _EXECUTION_STATE_KEYS
        or any(type(value.get(key)) not in allowed for key, allowed in _EXECUTION_STATE_FIELD_TYPES.items())
        or value.get("schema_version") != _EXECUTION_STATE_SCHEMA
        or re.fullmatch(r"smoke-[0-9a-f]{20}", value["smoke_id"]) is None
        or value["device"] not in {"cpu", "cuda"}
        or value["status"] not in _EXECUTION_STATUSES
        or not 0 <= value["current"] <= 2
        or value["total"] != 2
        or value["owner_pid"] <= 0
        or value["transition_sequence"] < 0
        or value["wall_clock_limit_seconds"] <= 0
        or not 1 <= len(value["logs"]) <= 50
        or any(
            type(line) is not str
            or not line
            or len(line) > 1_000
            or line != line.strip()
            or "\r" in line
            or "\n" in line
            for line in value["logs"]
        )
        or value["resumable"] is not False
        or value["retry_policy"] != "NEW_BUNDLE_REQUIRED"
        or any(value[key] is not False for key in FALSE_ELIGIBILITY)
        or not _valid_execution_state_identity(value)
    ):
        raise SmokeBundleError("smoke_execution_state", "The smoke execution state is invalid.")
    try:
        configuration = dict(plan["configurations"])[device]
        expected_environment = dict(configuration["environment"])
        expected_environment_identity = str(configuration["child_environment"]["environment_sha256"])
        expected_interpreter_identity = str(dict(plan["interpreter"])["interpreter_identity"])
        expected_writable_roots = [
            _directory_identity_record(root.joinpath(*Path(relative).parts), root)
            for relative in configuration["writable_roots"]
        ]
        expected_training_argv = smoke_training_argv(plan, device)
        expected_worker_argv = smoke_worker_argv(plan, device)
        expected_limit = int(configuration["wall_clock_limit_seconds"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise SmokeBundleError("smoke_execution_state", "The smoke execution state is invalid.") from exc
    started = _parse_utc(value["started_at"])
    deadline = _parse_utc(value["deadline_at"])
    updated = _parse_utc(value["updated_at"])
    worker_pid = value["worker_pid"]
    worker_process = value["worker_process"]
    worker_is_absent = worker_pid is None and worker_process is None
    worker_is_valid = _valid_process_identity(worker_process, worker_pid) if type(worker_pid) is int else False
    status = value["status"]
    active_fields_are_valid = value["current"] < 2 and value["exit_code"] is None and value["receipt_identity"] is None
    exit_code = value["exit_code"]
    exit_code_is_optional_failure = exit_code is None or (type(exit_code) is int and exit_code != 0)
    if (
        value["smoke_id"] != plan["smoke_id"]
        or value["device"] != device
        or value["plan_identity"] != plan["plan_identity"]
        or value["launch_identity"] != launch_identity
        or value["portable_argv"] != expected_training_argv
        or value["worker_argv"] != expected_worker_argv
        or value["argv_identity"] != stable_hash(expected_training_argv)
        or value["worker_argv_identity"] != stable_hash(expected_worker_argv)
        or value["execution_mode"] != "linux-worker-trainer-v1"
        or value["confinement"] is not None
        or value["environment"] != expected_environment
        or value["environment_identity"] != expected_environment_identity
        or value["interpreter_identity"] != expected_interpreter_identity
        or value["writable_roots"] != expected_writable_roots
        or value["wall_clock_limit_seconds"] != expected_limit
        or started is None
        or deadline is None
        or updated is None
        or deadline != started + timedelta(seconds=expected_limit)
        or updated < started
        or (status in {"STARTING", "RUNNING"} and updated > deadline)
        or (status == "TIMED_OUT" and updated < deadline)
        or (status == "STARTING" and value["transition_sequence"] != 0)
        or (status != "STARTING" and value["transition_sequence"] == 0)
        or (status == "STARTING" and value["current"] != 0)
        or (status == "STARTING" and updated != started)
        or (status == "STARTING" and not worker_is_absent)
        or (status in {"RUNNING", "COMPLETE"} and not worker_is_valid)
        or (status not in {"STARTING", "RUNNING", "COMPLETE"} and not (worker_is_absent or worker_is_valid))
        or (status in {"STARTING", "RUNNING"} and not active_fields_are_valid)
        or (
            status == "COMPLETE"
            and (
                value["current"] != 2
                or type(exit_code) is not int
                or exit_code != 0
                or not SHA256_PATTERN.fullmatch(value["receipt_identity"] or "")
            )
        )
        or (
            status == "FAILED"
            and (value["current"] >= 2 or value["receipt_identity"] is not None or not exit_code_is_optional_failure)
        )
        or (status == "INTERRUPTED" and (not active_fields_are_valid or exit_code is not None))
        or (
            status == "CANCELLED"
            and (value["current"] >= 2 or exit_code != 130 or value["receipt_identity"] is not None)
        )
        or (
            status == "TIMED_OUT"
            and (value["current"] >= 2 or exit_code != 124 or value["receipt_identity"] is not None)
        )
    ):
        raise SmokeBundleError("smoke_execution_state", "The smoke execution state is invalid.")


def _valid_execution_state_identity(value: Mapping[str, Any]) -> bool:
    identity = value.get("state_identity")
    if type(identity) is not str or SHA256_PATTERN.fullmatch(identity) is None:
        return False
    body = {key: item for key, item in value.items() if key != "state_identity"}
    try:
        return stable_hash(body) == identity
    except (TypeError, ValueError):
        return False


def _execution_state_path(root: Path, plan: Mapping[str, Any], device: str) -> Path:
    return artifact_bundle_directory(root, str(plan["smoke_id"])) / "execution" / device / "state.json"


def _directory_identity_record(path: Path, root: Path) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
        raise SmokeBundleError("smoke_execution_state", "A smoke writable root is unsafe.")
    return {
        "relative_path": relative,
        "identity_sha256": stable_hash(
            {
                "device": int(metadata.st_dev),
                "inode": int(metadata.st_ino),
                "mode": int(stat.S_IFMT(metadata.st_mode)),
            }
        ),
    }


def _load_worker_heartbeat_record(
    project_root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _read_json(heartbeat_path(project_root, str(plan["smoke_id"]), device), project_root)
    _validate_worker_heartbeat_record(value, plan, device, launch_identity)
    return value


def _validate_worker_heartbeat_record(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> None:
    if (
        set(value) != _HEARTBEAT_KEYS
        or any(type(value.get(key)) not in allowed for key, allowed in _HEARTBEAT_FIELD_TYPES.items())
        or value["schema_version"] != SMOKE_WORKER_HEARTBEAT_SCHEMA
        or value["smoke_id"] != plan["smoke_id"]
        or value["device"] != device
        or value["plan_identity"] != plan["plan_identity"]
        or value["launch_identity"] != launch_identity
        or value["status"] not in {"STARTING", "RUNNING"}
        or value["execution_state_sequence"] < 0
        or SHA256_PATTERN.fullmatch(value["execution_state_identity"]) is None
        or value["sequence"] < 1
        or not _valid_process_identity(value["worker_process"], value["worker_pid"])
        or value["containment"] not in {"WINDOWS_JOB", "LINUX_PDEATHSIG"}
        or value["message"] != "Contained smoke worker is active."
        or (value["sequence"] == 1 and value["previous_heartbeat_identity"] is not None)
        or (
            value["sequence"] > 1
            and (
                type(value["previous_heartbeat_identity"]) is not str
                or SHA256_PATTERN.fullmatch(value["previous_heartbeat_identity"]) is None
            )
        )
    ):
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat is invalid.")
    try:
        validate_identity(value, "heartbeat_identity")
    except (TypeError, ValueError, SmokeBundleError) as exc:
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat is invalid.") from exc
    worker_started = _parse_utc(value["worker_started_at"])
    heartbeat_at = _parse_utc(value["heartbeat_at"])
    if worker_started is None or heartbeat_at is None or heartbeat_at < worker_started:
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat is invalid.")


def _load_worker_outcome_record(
    project_root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> dict[str, Any]:
    value = _read_json(outcome_path(project_root, str(plan["smoke_id"]), device), project_root)
    _validate_worker_outcome_record(value, plan, device, launch_identity)
    return value


def _validate_worker_outcome_record(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
) -> None:
    status = value.get("status")
    expected_message = {
        "COMPLETE": "Contained smoke worker completed.",
        "FAILED": "Contained smoke worker failed.",
        "CANCELLED": "Contained smoke worker was cancelled and terminated.",
        "TIMED_OUT": "Contained smoke worker reached its fixed deadline and was terminated.",
    }.get(status)
    if (
        set(value) != _OUTCOME_KEYS
        or any(type(value.get(key)) not in allowed for key, allowed in _OUTCOME_FIELD_TYPES.items())
        or value.get("schema_version") != SMOKE_WORKER_OUTCOME_SCHEMA
        or value.get("smoke_id") != plan["smoke_id"]
        or value.get("device") != device
        or value.get("plan_identity") != plan["plan_identity"]
        or value.get("launch_identity") != launch_identity
        or status not in {"COMPLETE", "FAILED", "CANCELLED", "TIMED_OUT"}
        or value.get("execution_state_sequence", -1) < 0
        or SHA256_PATTERN.fullmatch(value.get("execution_state_identity", "")) is None
        or not _valid_process_identity(value.get("worker_process"), value.get("worker_pid"))
        or value.get("last_heartbeat_sequence", 0) < 1
        or SHA256_PATTERN.fullmatch(value.get("last_heartbeat_identity", "")) is None
        or value.get("message") != expected_message
        or (status == "COMPLETE" and value.get("exit_code") != 0)
        or (status == "FAILED" and value.get("exit_code") == 0)
        or (status == "CANCELLED" and value.get("exit_code") != 130)
        or (status == "TIMED_OUT" and value.get("exit_code") != 124)
    ):
        raise SmokeBundleError("smoke_worker_outcome", "The smoke worker outcome is invalid.")
    try:
        validate_identity(value, "outcome_identity")
    except (TypeError, ValueError, SmokeBundleError) as exc:
        raise SmokeBundleError("smoke_worker_outcome", "The smoke worker outcome is invalid.") from exc
    if _parse_utc(value["finished_at"]) is None:
        raise SmokeBundleError("smoke_worker_outcome", "The smoke worker outcome is invalid.")


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
    execution_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = heartbeat_path(root, str(plan["smoke_id"]), device)
    if type(sequence) is not int or sequence < 1:
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat sequence is invalid.")
    if not _valid_process_identity(worker_process, os.getpid()) or dict(worker_process) != _process_identity(
        os.getpid()
    ):
        raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker process identity is invalid.")
    with _state_transition_lock(path.parent):
        current = _load_execution_state(root, plan, device, launch_identity)
        if execution_state is not None and (
            execution_state.get("transition_sequence") != current["transition_sequence"]
            or execution_state.get("state_identity") != current["state_identity"]
        ):
            raise SmokeBundleError("smoke_worker_state_race", "The smoke execution state changed.")
        if status not in {"STARTING", "RUNNING"} or current["status"] != status:
            raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat status is invalid.")
        if status == "RUNNING" and (
            current["worker_pid"] != os.getpid() or current["worker_process"] != dict(worker_process)
        ):
            raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker process identity changed.")
        try:
            previous = _load_worker_heartbeat_record(root, plan, device, launch_identity)
        except FileNotFoundError:
            previous = None
        if sequence == 1:
            if previous is not None:
                raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat already exists.")
            previous_identity = None
        else:
            if (
                previous is None
                or previous["sequence"] != sequence - 1
                or previous["worker_pid"] != os.getpid()
                or previous["worker_process"] != dict(worker_process)
            ):
                raise SmokeBundleError("smoke_worker_heartbeat", "The smoke worker heartbeat chain is invalid.")
            previous_identity = previous["heartbeat_identity"]
        value = finalize_identity(
            {
                "schema_version": SMOKE_WORKER_HEARTBEAT_SCHEMA,
                "smoke_id": plan["smoke_id"],
                "device": device,
                "plan_identity": plan["plan_identity"],
                "launch_identity": launch_identity,
                "execution_state_sequence": current["transition_sequence"],
                "execution_state_identity": current["state_identity"],
                "status": status,
                "worker_pid": os.getpid(),
                "worker_process": dict(worker_process),
                "worker_started_at": started_at,
                "heartbeat_at": _now(),
                "sequence": sequence,
                "previous_heartbeat_identity": previous_identity,
                "containment": containment,
                "message": "Contained smoke worker is active.",
            },
            "heartbeat_identity",
        )
        _validate_worker_heartbeat_record(value, plan, device, launch_identity)
        confirmed = _load_execution_state(root, plan, device, launch_identity)
        if (
            confirmed["transition_sequence"] != current["transition_sequence"]
            or confirmed["state_identity"] != current["state_identity"]
        ):
            raise SmokeBundleError("smoke_worker_heartbeat", "The smoke execution state changed.")
        with anchored_directory(path.parent, path.parent) as anchor:
            anchor.atomic_write_bytes(path.name, canonical_json_bytes(value, pretty=True))
        return value


def _finish(
    root: Path,
    plan: Mapping[str, Any],
    device: str,
    launch_identity: str,
    heartbeat: Mapping[str, Any],
    exit_code: int,
    *,
    status_override: str | None = None,
    operation_check: Callable[[], None] | None = None,
) -> int:
    if status_override not in {None, *_WORKER_TERMINAL_EXIT_CODES}:
        raise SmokeBundleError("smoke_worker_outcome", "The smoke worker outcome status is invalid.")
    if status_override is not None and exit_code != _WORKER_TERMINAL_EXIT_CODES[status_override]:
        raise SmokeBundleError("smoke_worker_outcome", "The smoke worker outcome exit code is invalid.")
    if operation_check is not None:
        operation_check()
    path = outcome_path(root, str(plan["smoke_id"]), device)
    with _state_transition_lock(path.parent):
        state = _read_execution_state(root, plan, device, launch_identity)
        if state["status"] in _WORKER_TERMINAL_EXIT_CODES:
            terminal_status = str(state["status"])
            if status_override is not None and status_override != terminal_status:
                raise SmokeBundleError("smoke_worker_outcome", "The smoke worker terminal state changed.")
            status = terminal_status
            effective_exit_code = _WORKER_TERMINAL_EXIT_CODES[terminal_status]
        else:
            if state["status"] != "RUNNING" or status_override is not None:
                raise SmokeBundleError("smoke_worker_outcome", "The smoke execution is no longer running.")
            if operation_check is not None:
                operation_check()
            status = "COMPLETE" if exit_code == 0 else "FAILED"
            effective_exit_code = int(exit_code)
        latest_heartbeat = _load_worker_heartbeat_record(root, plan, device, launch_identity)
        if latest_heartbeat != dict(heartbeat):
            raise SmokeBundleError("smoke_worker_outcome", "The smoke worker heartbeat is stale.")
        if (
            latest_heartbeat["worker_pid"] != os.getpid()
            or latest_heartbeat["worker_process"] != state["worker_process"]
            or (
                state["status"] == "RUNNING"
                and (
                    latest_heartbeat["execution_state_sequence"] != state["transition_sequence"]
                    or latest_heartbeat["execution_state_identity"] != state["state_identity"]
                )
            )
        ):
            raise SmokeBundleError("smoke_worker_outcome", "The smoke worker heartbeat binding changed.")
        value = finalize_identity(
            {
                "schema_version": SMOKE_WORKER_OUTCOME_SCHEMA,
                "smoke_id": plan["smoke_id"],
                "device": device,
                "plan_identity": plan["plan_identity"],
                "launch_identity": launch_identity,
                "execution_state_sequence": state["transition_sequence"],
                "execution_state_identity": state["state_identity"],
                "worker_pid": os.getpid(),
                "worker_process": dict(latest_heartbeat["worker_process"]),
                "status": status,
                "exit_code": effective_exit_code,
                "finished_at": _now(),
                "last_heartbeat_sequence": latest_heartbeat["sequence"],
                "last_heartbeat_identity": latest_heartbeat["heartbeat_identity"],
                "message": {
                    "COMPLETE": "Contained smoke worker completed.",
                    "FAILED": "Contained smoke worker failed.",
                    "CANCELLED": "Contained smoke worker was cancelled and terminated.",
                    "TIMED_OUT": "Contained smoke worker reached its fixed deadline and was terminated.",
                }[status],
            },
            "outcome_identity",
        )
        _validate_worker_outcome_record(value, plan, device, launch_identity)
        confirmed = _read_execution_state(root, plan, device, launch_identity)
        if (
            confirmed["transition_sequence"] != state["transition_sequence"]
            or confirmed["state_identity"] != state["state_identity"]
        ):
            raise SmokeBundleError("smoke_worker_outcome", "The smoke execution state changed.")
        payload = canonical_json_bytes(value, pretty=True)
        try:
            write_exclusive_bytes(path, payload, boundary=path.parent)
        except FileExistsError:
            existing = read_stable_single_link_bytes(path, boundary=path.parent, max_bytes=4 * 1024 * 1024)
            if existing != payload:
                return 70
        return effective_exit_code


@contextmanager
def _state_transition_lock(directory: Path, *, timeout: float = 5.0) -> Any:
    if not math.isfinite(timeout) or timeout <= 0:
        raise SmokeBundleError("smoke_transition_lock", "The smoke transition-lock timeout is invalid.")
    with AnchoredDirectory(directory, directory) as anchor:
        with _fixed_lock_authority(
            anchor,
            ".state-transition.lock",
            timeout=timeout,
        ):
            yield


@contextmanager
def _fixed_lock_authority(anchor: AnchoredDirectory, name: str, *, timeout: float) -> Any:
    """Hold one worker lock without allowing a second pathname authority."""

    descriptor = -1
    authority_descriptor = -1
    acquired = False
    post_yield_error: SmokeBundleError | None = None
    deadline = time.monotonic() + timeout
    try:
        descriptor = anchor.open_file_immovable(
            name,
            os.O_RDWR | os.O_CREAT | int(getattr(os, "O_BINARY", 0)),
        )
        metadata = os.fstat(descriptor)
        identity = OwnedFileIdentity.from_stat(metadata)
        _verify_fixed_lock_authority(
            anchor,
            name,
            descriptor,
            identity,
            message="The smoke transition lock is unsafe.",
        )
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        if os.name == "nt":
            authority_descriptor = descriptor
        else:
            authority_descriptor = _open_posix_directory_lock_authority(anchor)
        while not acquired:
            try:
                _lock_fixed_authority(authority_descriptor)
                acquired = True
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise SmokeBundleError(
                        "smoke_transition_lock",
                        "Timed out waiting for the smoke transition lock.",
                    ) from None
                time.sleep(0.02)
        _verify_fixed_lock_authority(
            anchor,
            name,
            descriptor,
            identity,
            message="The smoke transition lock changed.",
        )
        try:
            yield
        finally:
            try:
                _verify_fixed_lock_authority(
                    anchor,
                    name,
                    descriptor,
                    identity,
                    message="The smoke transition lock changed.",
                )
            except SmokeBundleError as exc:
                post_yield_error = exc
    finally:
        release_error: SmokeBundleError | None = None
        if acquired:
            try:
                _verify_fixed_lock_authority(
                    anchor,
                    name,
                    descriptor,
                    identity,
                    message="The smoke transition lock changed.",
                )
            except SmokeBundleError as exc:
                release_error = exc
            try:
                _unlock_fixed_authority(authority_descriptor)
            except OSError:
                pass
        if authority_descriptor >= 0 and authority_descriptor != descriptor:
            os.close(authority_descriptor)
        if descriptor >= 0:
            os.close(descriptor)
        if release_error is not None:
            raise release_error
        if post_yield_error is not None:
            raise post_yield_error


def _verify_fixed_lock_authority(
    anchor: AnchoredDirectory,
    name: str,
    descriptor: int,
    identity: OwnedFileIdentity,
    *,
    message: str,
) -> None:
    try:
        anchor.verify()
        opened = os.fstat(descriptor)
        current = anchor.lstat(name)
    except (OSError, ValueError) as exc:
        raise SmokeBundleError("smoke_transition_lock", message) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or int(getattr(opened, "st_nlink", 1)) != 1
        or int(getattr(current, "st_nlink", 1)) != 1
        or not identity.matches(opened)
        or not identity.matches(current)
    ):
        raise SmokeBundleError("smoke_transition_lock", message)


def _open_posix_directory_lock_authority(anchor: AnchoredDirectory) -> int:
    descriptor = -1
    try:
        expected = OwnedFileIdentity.from_stat(anchor.directory_metadata())
        descriptor = os.open(
            anchor.fixed_directory_path(),
            os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_CLOEXEC", 0)),
        )
        opened = os.fstat(descriptor)
        anchor.verify()
        if not stat.S_ISDIR(opened.st_mode) or not expected.matches(opened):
            raise SmokeBundleError("smoke_transition_lock", "The smoke transition lock is unsafe.")
        return descriptor
    except (OSError, ValueError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(exc, SmokeBundleError):
            raise
        raise SmokeBundleError("smoke_transition_lock", "The smoke transition lock is unsafe.") from exc


def _lock_fixed_authority(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fixed_authority(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _read_json(path: Path, root: Path) -> dict[str, Any]:
    del root
    payload = read_stable_single_link_bytes(path, boundary=path.parent, max_bytes=4 * 1024 * 1024)
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
        if os.name != "nt":
            _terminate_posix_process_group(self.process)
            return
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
    # The outer worker intentionally remains Linux-only. Windows uses the
    # runner's one-process direct-trainer path so the Job's process limit can
    # never be bypassed by a nested trainer.
    return sys.platform.startswith("linux")


def _install_termination_handlers(containment: _Containment) -> None:
    def terminate(_signum: int, _frame: Any) -> None:
        containment.terminate()
        raise SystemExit(143)

    for name in ("SIGTERM", "SIGINT"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, terminate)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if os.name != "nt" and type(getattr(process, "pid", None)) is int:
        _terminate_posix_process_group(process)
        return
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


def _terminate_posix_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 5.0,
    kill_group: Callable[[int, int], None] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> None:
    if not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("The process-group termination grace is invalid.")
    pid = getattr(process, "pid", None)
    if type(pid) is not int or pid <= 0:
        return
    send = kill_group or os.killpg
    clock = monotonic or time.monotonic
    pause = sleep or time.sleep
    term_delivered = True
    try:
        send(pid, int(getattr(signal, "SIGTERM", 15)))
    except (OSError, ProcessLookupError):
        term_delivered = False
    if term_delivered:
        deadline = clock() + grace_seconds
        while clock() < deadline:
            pause(min(0.05, max(0.0, deadline - clock())))
        try:
            send(pid, int(getattr(signal, "SIGKILL", 9)))
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=5)
    except BaseException:
        try:
            process.kill()
            process.wait(timeout=5)
        except BaseException:
            pass


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
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
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
    handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        return None
    try:
        if int(kernel32.WaitForSingleObject(handle, 0)) != 0x102:
            return None
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


def _process_is_definitively_absent(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        return _windows_process_is_definitively_absent(pid)
    # Linux identity lookup is backed by the exact /proc PID entry.  The
    # contained worker is the process-group leader, so a missing entry proves
    # that exact recorded leader is no longer active.  Unsupported platforms
    # remain deliberately indeterminate.
    return smoke_containment_supported()


def _windows_process_is_definitively_absent(pid: int) -> bool:
    """Distinguish a departed Windows PID from an unreadable live process."""

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        # ERROR_INVALID_PARAMETER is the documented result for a PID that no
        # longer identifies a process.  Access-denied and other failures stay
        # indeterminate so restart recovery never assumes a live process died.
        return ctypes.get_last_error() == 87
    try:
        result = int(kernel32.WaitForSingleObject(handle, 0))
        if result == 0:
            return True
        if result == 0x102:
            return False
        return False
    finally:
        kernel32.CloseHandle(handle)


def _valid_process_identity(value: Any, pid: int) -> bool:
    return (
        type(pid) is int
        and pid > 0
        and type(value) is dict
        and set(value) == _PROCESS_IDENTITY_KEYS
        and value.get("pid") == pid
        and type(value.get("birth_token")) is str
        and bool(value["birth_token"])
        and type(value.get("process_image_path_sha256")) is str
        and SHA256_PATTERN.fullmatch(value["process_image_path_sha256"]) is not None
    )


def _parse_utc(value: Any) -> datetime | None:
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    normalized = parsed.astimezone(timezone.utc)
    if normalized.isoformat() != value:
        return None
    return normalized


def _parse_time(value: str) -> datetime:
    parsed = _parse_utc(value)
    if parsed is None:
        raise SmokeBundleError("smoke_worker_time", "The smoke worker timestamp is invalid.")
    return parsed


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
