"""Durable, fixed-plan local execution for exploratory smoke bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from spritelab.training.smoke_bundle import (
    FALSE_ELIGIBILITY,
    SmokeBundleError,
    anchored_directory,
    anchored_path_is_absent,
    artifact_bundle_directory,
    begin_device_run,
    begin_device_run_anchored,
    build_smoke_child_environment,
    canonical_json_bytes,
    ensure_managed_directory,
    load_device_receipt,
    load_plan,
    pinned_smoke_interpreter,
    read_stable_single_link_bytes,
    run_bundle_directory,
    smoke_launch_identity,
    smoke_training_argv,
    smoke_worker_argv,
    stable_hash,
    validate_smoke_interpreter,
    verify_execution_guards,
    verify_pinned_process_image,
    write_exclusive_bytes,
)
from spritelab.training.smoke_worker import (
    STARTUP_GRACE_SECONDS,
    capture_worker_process_identity,
    heartbeat_is_fresh,
    load_worker_heartbeat,
    load_worker_outcome,
    smoke_containment_supported,
    worker_process_identity_is_valid,
    worker_process_matches,
)
from spritelab.utils.pinned_executable import (
    activate_windows_suspended_process,
    close_windows_handle,
    linux_parent_death_signal,
)
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity
from spritelab.utils.write_confinement import (
    WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY,
    WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256,
    create_windows_bootstrap_untrusted_process,
    prepare_windows_untrusted_integrity_roots,
)

SMOKE_EXECUTION_SCHEMA = "spritelab.training.smoke-device-execution.v1"
_DEVICES = ("cpu", "cuda")
_TERMINAL = {"COMPLETE", "FAILED", "INTERRUPTED", "CANCELLED", "TIMED_OUT"}
_STATUSES = {"STARTING", "RUNNING", *_TERMINAL}
_STATE_IDENTITY_FIELD = "state_identity"
_NONE_TYPE = type(None)
_STATE_FIELD_TYPES: dict[str, tuple[type, ...]] = {
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
    _STATE_IDENTITY_FIELD: (str,),
    **dict.fromkeys(FALSE_ELIGIBILITY, (bool,)),
}
_STATE_KEYS = frozenset(_STATE_FIELD_TYPES)
_TRANSITION_MUTABLE_FIELDS = {
    "status",
    "current",
    "worker_pid",
    "worker_process",
    "exit_code",
    "receipt_identity",
    "confinement",
}
_LIVE_PROCESS_LOCK = threading.RLock()
_LIVE_PROCESSES: dict[tuple[str, str, str], tuple[str, Any]] = {}


class SmokeExecutionError(SmokeBundleError):
    """A fixed-plan process launch or durable execution-state failure."""


def _runner_containment_supported() -> bool:
    return smoke_containment_supported() or (sys.platform == "win32" and os.name == "nt")


class ExploratorySmokeRunner:
    """Launch only a server-prepared smoke argv and reconstruct it passively."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        process_factory: Callable[..., Any] | None = None,
        windows_process_factory: Callable[..., Any] | None = None,
        windows_suspended_activator: Callable[..., int] | None = None,
        containment_supported: Callable[[], bool] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._process_factory = process_factory or subprocess.Popen
        self._windows_process_factory = windows_process_factory or create_windows_bootstrap_untrusted_process
        self._windows_suspended_activator = windows_suspended_activator or activate_windows_suspended_process
        self._containment_supported = containment_supported or _runner_containment_supported
        self._lock = threading.RLock()

    def launch(
        self,
        smoke_id: str,
        plan_identity: str,
        device: str,
        *,
        explicit_action: bool,
    ) -> dict[str, Any]:
        if explicit_action is not True:
            raise SmokeExecutionError("explicit_smoke_run", "Running a smoke device requires an explicit action.")
        operation_started = datetime.now(timezone.utc)
        normalized = _device(device)
        plan = load_plan(self.project_root, smoke_id)
        if plan.get("plan_identity") != plan_identity:
            raise SmokeExecutionError("smoke_plan_changed", "The selected smoke plan identity changed.")
        validate_smoke_interpreter(plan)
        with self._lock, _project_launch_lock(self.project_root):
            if normalized == "cuda" and self.status(smoke_id, "cpu")["status"] != "COMPLETE":
                raise SmokeExecutionError(
                    "cpu_smoke_required",
                    "The server-run CPU smoke must complete before CUDA can start.",
                )
            current = self.status(smoke_id, normalized)
            if current["status"] != "NOT_STARTED":
                if current["status"] in {"STARTING", "RUNNING", "COMPLETE"}:
                    return current
                raise SmokeExecutionError(
                    "smoke_new_bundle_required",
                    "This device run failed or was interrupted; prepare a fresh smoke bundle.",
                )
            if not self._containment_supported():
                raise SmokeExecutionError(
                    "smoke_containment_unavailable",
                    "Server-run exploratory smoke containment is unavailable on this platform.",
                )
            self._require_no_active_smoke()
            output = run_bundle_directory(self.project_root, smoke_id) / normalized
            if not anchored_path_is_absent(output, self.project_root):
                raise SmokeExecutionError(
                    "smoke_unowned_output",
                    "Device output exists without a server-owned execution; prepare a fresh smoke bundle.",
                )
            bundle = artifact_bundle_directory(self.project_root, smoke_id)
            writable_anchors = ExitStack()
            directory = ensure_managed_directory(bundle, ("execution", normalized), boundary=self.project_root)
            if os.name == "nt":
                directory_anchor = writable_anchors.enter_context(AnchoredDirectory(directory, directory))
                output = ensure_managed_directory(
                    run_bundle_directory(self.project_root, smoke_id),
                    (normalized,),
                    boundary=self.project_root,
                )
                output_anchor = writable_anchors.enter_context(AnchoredDirectory(output, output))
                prepare_windows_untrusted_integrity_roots((directory, output))
                temp_identity = directory_anchor.mkdir("temp")
                if not temp_identity.matches(directory_anchor.lstat("temp")):
                    raise SmokeExecutionError("smoke_writable_root", "The private smoke temp root changed.")
                output = begin_device_run_anchored(output_anchor, plan, normalized)
            else:
                ensure_managed_directory(bundle, ("execution", normalized, "temp"), boundary=self.project_root)
                output = begin_device_run(self.project_root, plan, normalized)
            if os.name != "nt":
                writable_anchors.enter_context(AnchoredDirectory(directory, directory))
                writable_anchors.enter_context(AnchoredDirectory(output, output))
            training_argv = smoke_training_argv(plan, normalized)
            worker_argv = smoke_worker_argv(plan, normalized)
            execution_mode = "windows-direct-trainer-v1" if os.name == "nt" else "linux-worker-trainer-v1"
            launched_argv = training_argv if os.name == "nt" else worker_argv
            launch_identity = smoke_launch_identity(plan, normalized)
            public_environment = dict(plan["configurations"][normalized]["environment"])
            child_environment = build_smoke_child_environment(self.project_root, plan, normalized)
            environment_identity = str(plan["configurations"][normalized]["child_environment"]["environment_sha256"])
            started = operation_started
            wall_clock_limit = int(plan["configurations"][normalized]["wall_clock_limit_seconds"])
            state = _finalize_state_identity(
                {
                    "schema_version": SMOKE_EXECUTION_SCHEMA,
                    "smoke_id": smoke_id,
                    "device": normalized,
                    "plan_identity": plan_identity,
                    "launch_identity": launch_identity,
                    "status": "STARTING",
                    "current": 0,
                    "total": 2,
                    "owner_pid": os.getpid(),
                    "worker_pid": None,
                    "worker_process": None,
                    "portable_argv": training_argv,
                    "worker_argv": launched_argv,
                    "argv_identity": stable_hash(training_argv),
                    "worker_argv_identity": stable_hash(launched_argv),
                    "execution_mode": execution_mode,
                    "confinement": None,
                    "environment": public_environment,
                    "environment_identity": environment_identity,
                    "interpreter_identity": dict(plan["interpreter"])["interpreter_identity"],
                    "writable_roots": [
                        _directory_identity_record(directory, self.project_root),
                        _directory_identity_record(output, self.project_root),
                    ],
                    "started_at": started.isoformat(),
                    "deadline_at": (started + timedelta(seconds=wall_clock_limit)).isoformat(),
                    "wall_clock_limit_seconds": wall_clock_limit,
                    "updated_at": started.isoformat(),
                    "exit_code": None,
                    "receipt_identity": None,
                    "transition_sequence": 0,
                    "retry_policy": "NEW_BUNDLE_REQUIRED",
                    "resumable": False,
                    "logs": ["Contained exploratory smoke launch requested."],
                    **FALSE_ELIGIBILITY,
                }
            )
            self._validate_state(state)
            try:
                if os.name == "nt":
                    _write_anchored_exclusive(
                        directory_anchor,
                        "state.json",
                        canonical_json_bytes(state, pretty=True),
                    )
                else:
                    write_exclusive_bytes(
                        directory / "state.json",
                        canonical_json_bytes(state, pretty=True),
                        boundary=self.project_root,
                    )
            except FileExistsError:
                writable_anchors.close()
                return self.status(smoke_id, normalized)
            process: Any | None = None
            worker_job_handle: int | None = None
            preflight_complete = False
            try:

                def operation_check() -> None:
                    current = self._read_state(smoke_id, normalized, plan)
                    if current["status"] == "CANCELLED":
                        raise SmokeExecutionError("smoke_cancelled", "The smoke was cancelled before process launch.")
                    if current["status"] == "TIMED_OUT" or _deadline_expired(current):
                        if current["status"] not in _TERMINAL:
                            self._transition(
                                current,
                                status="TIMED_OUT",
                                exit_code=124,
                                message="The fixed smoke wall-clock deadline expired during preflight.",
                            )
                        raise SmokeExecutionError("smoke_timed_out", "The smoke deadline expired before launch.")
                    if current["status"] != "STARTING":
                        raise SmokeExecutionError("smoke_state_changed", "The smoke launch state changed safely.")

                operation_check()
                verify_execution_guards(
                    self.project_root,
                    plan,
                    operation_check=operation_check,
                )
                operation_check()
                preflight_complete = True
                writable_fds: tuple[int, ...] = ()
                launch_environment = dict(child_environment)
                if sys.platform.startswith("linux"):
                    opened: list[int] = []
                    flags = (
                        int(getattr(os, "O_PATH", os.O_RDONLY))
                        | int(getattr(os, "O_DIRECTORY", 0))
                        | int(getattr(os, "O_NOFOLLOW", 0))
                    )
                    for writable_root in (directory, output):
                        descriptor = os.open(writable_root, flags)
                        opened.append(descriptor)
                        writable_anchors.callback(os.close, descriptor)
                    writable_fds = tuple(opened)
                    launch_environment["SPRITELAB_WRITABLE_ROOT_FDS"] = ",".join(str(value) for value in writable_fds)
                with pinned_smoke_interpreter(plan) as interpreter:
                    argv = [interpreter.launch_path, *launched_argv[1:]]
                    operation_check()
                    if os.name == "nt":
                        launch_environment["SPRITELAB_CONFINEMENT_PROJECT_ROOT"] = os.fspath(self.project_root)
                        process = self._windows_process_factory(
                            argv,
                            cwd=directory,
                            env=launch_environment,
                            stdin_payload=b"",
                            writable_roots=(directory, output),
                            stdio_root=directory / "temp",
                        )
                        worker_job_handle = self._windows_suspended_activator(
                            process,
                            verifier=lambda launched: verify_pinned_process_image(
                                launched,
                                interpreter,
                            ),
                        )
                    else:
                        process_options: dict[str, Any] = {
                            "cwd": self.project_root,
                            "env": launch_environment,
                            "stdin": subprocess.DEVNULL,
                            "stdout": subprocess.DEVNULL,
                            "stderr": subprocess.DEVNULL,
                            "shell": False,
                        }
                        process_options["start_new_session"] = True
                        process_options["pass_fds"] = tuple(sorted({*interpreter.pass_fds, *writable_fds}))
                        if sys.platform.startswith("linux"):
                            process_options["preexec_fn"] = linux_parent_death_signal(os.getpid())
                        process = self._process_factory(argv, **process_options)
                        verify_pinned_process_image(process, interpreter)
                    operation_check()
            except BaseException as exc:
                if process is not None:
                    _terminate_worker_process(process)
                if worker_job_handle:
                    close_windows_handle(worker_job_handle)
                writable_anchors.close()
                current = self._read_state(smoke_id, normalized, plan)
                if current["status"] in _TERMINAL:
                    return _projection(current)
                failed = self._transition(
                    current,
                    status="FAILED",
                    exit_code=None,
                    message=(
                        f"Contained worker {'launch' if preflight_complete else 'preflight'} failed: "
                        f"{type(exc).__name__}."
                    ),
                )
                if not preflight_complete:
                    raise
                return _projection(failed)
            try:
                worker_process = capture_worker_process_identity(int(process.pid))
                if worker_process is None:
                    raise SmokeExecutionError(
                        "smoke_worker_identity",
                        "The contained smoke worker identity could not be established.",
                    )
                confinement = (
                    _windows_confinement_binding(
                        process,
                        directory,
                        output,
                        project_root=self.project_root,
                    )
                    if os.name == "nt"
                    else None
                )
                state = self._transition(
                    state,
                    status="RUNNING",
                    worker_pid=int(process.pid),
                    worker_process=worker_process,
                    confinement=confinement,
                    message=(
                        f"Contained {normalized.upper()} direct smoke trainer is running."
                        if os.name == "nt"
                        else f"Contained {normalized.upper()} two-step smoke worker is running."
                    ),
                )
            except BaseException as exc:
                _terminate_worker_process(process)
                if worker_job_handle:
                    close_windows_handle(worker_job_handle)
                writable_anchors.close()
                try:
                    self._transition(
                        state,
                        status="FAILED",
                        message="Worker stopped because durable launch-state publication failed.",
                    )
                except BaseException:
                    pass
                raise SmokeExecutionError(
                    "smoke_launch_state",
                    "The contained smoke worker was stopped because durable state could not be published.",
                ) from exc
            try:
                _remember_process(self.project_root, smoke_id, normalized, launch_identity, process)
                monitor = threading.Thread(
                    target=self._monitor,
                    args=(plan, normalized, process, worker_job_handle, writable_anchors),
                    name=f"spritelab-{smoke_id}-{normalized}",
                    daemon=True,
                )
                monitor.start()
            except BaseException as exc:
                _forget_process(self.project_root, smoke_id, normalized, launch_identity)
                _terminate_worker_process(process)
                if worker_job_handle:
                    close_windows_handle(worker_job_handle)
                writable_anchors.close()
                failed = self._transition(
                    state,
                    status="FAILED",
                    message=f"Smoke monitor launch failed safely: {type(exc).__name__}.",
                )
                return _projection(failed)
            return _projection(state)

    def status(self, smoke_id: str, device: str) -> dict[str, Any]:
        normalized = _device(device)
        plan = load_plan(self.project_root, smoke_id)
        state_path = self._state_path(smoke_id, normalized)
        try:
            with AnchoredDirectory(state_path.parent, state_path.parent) as state_anchor:
                state_absent = not state_anchor.lexists(state_path.name)
            if state_absent:
                return _not_started(smoke_id, normalized, str(plan["plan_identity"]))
            state = self._read_state(smoke_id, normalized, plan)
        except FileNotFoundError:
            return _not_started(smoke_id, normalized, str(plan["plan_identity"]))
        if state["status"] not in _TERMINAL and _deadline_expired(state):
            return self._stop_execution(
                plan,
                normalized,
                state,
                status="TIMED_OUT",
                message="The fixed smoke wall-clock deadline expired; contained work was terminated.",
            )
        if state["status"] in _TERMINAL:
            if state["status"] == "COMPLETE":
                try:
                    receipt = load_device_receipt(self.project_root, plan, normalized)
                except (OSError, ValueError, SmokeBundleError):
                    projected = dict(state)
                    projected.update({"status": "FAILED", "updated_at": _now()})
                    projected["logs"] = [*projected["logs"], "The immutable completion receipt is unavailable."][-50:]
                    return _projection(projected)
                if receipt.get("receipt_identity") != state.get("receipt_identity"):
                    projected = dict(state)
                    projected.update({"status": "FAILED", "updated_at": _now()})
                    return _projection(projected)
            return _projection(state)
        try:
            receipt = load_device_receipt(self.project_root, plan, normalized)
        except (OSError, ValueError, SmokeBundleError):
            receipt = None
        launch_identity = str(state["launch_identity"])
        process = _recalled_process(self.project_root, smoke_id, normalized, launch_identity)
        if process is not None:
            code = process.poll()
            if code is None:
                return _projection(state)
            if code == 0 and receipt is not None:
                with self._lock:
                    completed = self._publish_completion(
                        state,
                        plan,
                        normalized,
                        receipt["receipt_identity"],
                        message="Contained worker exit and completion receipt verified.",
                    )
                return _projection(completed)
            with self._lock:
                try:
                    failed = self._transition(
                        state,
                        status="FAILED",
                        exit_code=70 if int(code) == 0 else int(code),
                        message="Contained worker ended without a valid completion receipt.",
                    )
                except SmokeExecutionError as exc:
                    if exc.code != "smoke_transition_race":
                        raise
                    failed = self._read_state(smoke_id, normalized, plan)
                    if failed["status"] not in _TERMINAL:
                        raise
            return _projection(failed)
        try:
            heartbeat = load_worker_heartbeat(self.project_root, plan, normalized, launch_identity)
        except (OSError, ValueError, SmokeBundleError):
            heartbeat = None
        if heartbeat is not None:
            same_process = worker_process_matches(heartbeat)
            if same_process is True or (same_process is None and heartbeat_is_fresh(heartbeat)):
                projected = dict(state)
                projected.update(
                    {
                        "status": "RUNNING",
                        "worker_pid": heartbeat["worker_pid"],
                        "updated_at": heartbeat["heartbeat_at"],
                    }
                )
                projected["logs"] = [*projected["logs"], "Contained worker heartbeat verified after restart."][-50:]
                return _projection(projected)
            if same_process is None:
                projected = dict(state)
                projected.update({"status": "RUNNING", "updated_at": heartbeat["heartbeat_at"]})
                projected["logs"] = [
                    *projected["logs"],
                    "Worker identity cannot be disproved; new smoke launches remain blocked.",
                ][-50:]
                return _projection(projected)
        try:
            outcome = load_worker_outcome(self.project_root, plan, normalized, launch_identity)
        except (OSError, ValueError, SmokeBundleError):
            outcome = None
        if outcome is not None:
            if outcome["status"] == "COMPLETE" and int(outcome["exit_code"]) == 0 and receipt is not None:
                completed = self._publish_completion(
                    state,
                    plan,
                    normalized,
                    receipt["receipt_identity"],
                    message="Contained worker outcome and completion receipt verified after restart.",
                )
                return _projection(completed)
            outcome_status = str(outcome["status"])
            outcome_code = int(outcome["exit_code"])
            if outcome_status == "COMPLETE":
                outcome_status = "FAILED"
                outcome_code = 70
            terminal = self._transition(
                state,
                status=outcome_status,
                exit_code=outcome_code,
                message="Contained smoke worker ended without valid completion.",
            )
            return _projection(terminal)
        worker_pid = state.get("worker_pid")
        if type(worker_pid) is int and worker_process_identity_is_valid(state.get("worker_process"), worker_pid):
            same_process = worker_process_matches(state)
            if same_process is True or same_process is None:
                projected = dict(state)
                projected["status"] = "RUNNING"
                projected["logs"] = [
                    *projected["logs"],
                    "Exact contained worker process identity remains active.",
                ][-50:]
                return _projection(projected)
        if _age_seconds(str(state.get("started_at") or "")) <= STARTUP_GRACE_SECONDS:
            return _projection(state)
        interrupted = self._transition(
            state,
            status="INTERRUPTED",
            message="The contained worker ended before a completion receipt; a fresh bundle is required.",
        )
        return _projection(interrupted)

    def cancel(
        self,
        smoke_id: str,
        plan_identity: str,
        device: str,
        *,
        explicit_action: bool,
    ) -> dict[str, Any]:
        if explicit_action is not True:
            raise SmokeExecutionError("explicit_smoke_cancel", "Cancelling a smoke requires an explicit action.")
        normalized = _device(device)
        plan = load_plan(self.project_root, smoke_id)
        if plan.get("plan_identity") != plan_identity:
            raise SmokeExecutionError("smoke_plan_changed", "The selected smoke plan identity changed.")
        state_path = self._state_path(smoke_id, normalized)
        try:
            with AnchoredDirectory(state_path.parent, state_path.parent) as state_anchor:
                state_absent = not state_anchor.lexists(state_path.name)
        except FileNotFoundError:
            state_absent = True
        if state_absent:
            return _not_started(smoke_id, normalized, plan_identity)
        state = self._read_state(smoke_id, normalized, plan)
        if state["status"] in _TERMINAL:
            return _projection(state)
        return self._stop_execution(
            plan,
            normalized,
            state,
            status="CANCELLED",
            message="Explicit cancellation terminated the contained smoke worker and descendants.",
        )

    def bundle_status(self, smoke_id: str) -> dict[str, Any]:
        plan = load_plan(self.project_root, smoke_id)
        devices = {device: self.status(smoke_id, device) for device in _DEVICES}
        return {
            "smoke_id": smoke_id,
            "plan_identity": plan["plan_identity"],
            "devices": devices,
            "registration_ready": all(value["status"] == "COMPLETE" for value in devices.values()),
            "retry_policy": "NEW_BUNDLE_REQUIRED",
            **FALSE_ELIGIBILITY,
        }

    def require_complete(self, smoke_id: str) -> dict[str, dict[str, Any]]:
        values = {device: self.status(smoke_id, device) for device in _DEVICES}
        if any(value["status"] != "COMPLETE" for value in values.values()):
            raise SmokeExecutionError(
                "smoke_devices_incomplete",
                "Both server-run CPU and CUDA smoke actions must complete before registration.",
            )
        return values

    def _require_no_active_smoke(self) -> None:
        parent = self.project_root / "artifacts" / "training" / "smokes"
        with anchored_directory(parent, self.project_root) as anchor:
            names = anchor.names()
        for name in names:
            if not re.fullmatch(r"smoke-[0-9a-f]{20}", name):
                continue
            for device in _DEVICES:
                state_path = artifact_bundle_directory(self.project_root, name) / "execution" / device / "state.json"
                try:
                    with AnchoredDirectory(state_path.parent, state_path.parent) as state_anchor:
                        state_is_absent = not state_anchor.lexists(state_path.name)
                except FileNotFoundError:
                    state_is_absent = True
                if state_is_absent:
                    continue
                value = self.status(name, device)
                if value["status"] in {"STARTING", "RUNNING"}:
                    raise SmokeExecutionError(
                        "smoke_project_active",
                        "Another contained exploratory smoke worker is active; wait for it to finish.",
                    )

    def _monitor(
        self,
        plan: Mapping[str, Any],
        device: str,
        process: Any,
        worker_job_handle: int | None,
        writable_anchors: ExitStack,
    ) -> None:
        smoke_id = str(plan["smoke_id"])
        try:
            while process.poll() is None:
                with self._lock:
                    state = self._read_state(smoke_id, device, plan)
                    if state["status"] in {"CANCELLED", "TIMED_OUT"}:
                        _terminate_worker_process(process)
                        break
                    if _deadline_expired(state):
                        _terminate_worker_process(process)
                        if process.poll() is None:
                            raise SmokeExecutionError(
                                "smoke_process_termination",
                                "The contained smoke deadline termination could not be verified.",
                            )
                        self._transition(
                            state,
                            status="TIMED_OUT",
                            exit_code=124,
                            message="The fixed smoke wall-clock deadline expired; contained work was terminated.",
                        )
                        break
                time.sleep(0.2)
            code = int(process.wait(timeout=6))
            with self._lock:
                state = self._read_state(smoke_id, device, plan)
                if state["status"] in {"CANCELLED", "TIMED_OUT"}:
                    return
                if code == 0:
                    try:
                        receipt = load_device_receipt(self.project_root, plan, device)
                    except (OSError, ValueError, SmokeBundleError):
                        receipt = None
                    if receipt is not None:
                        self._publish_completion(
                            state,
                            plan,
                            device,
                            receipt["receipt_identity"],
                            message="Fixed two-step smoke completed with an immutable receipt.",
                        )
                        return
                self._transition(
                    state,
                    status="FAILED",
                    exit_code=70 if code == 0 else code,
                    message="Smoke process ended without a valid completion receipt; prepare a fresh bundle.",
                )
        except BaseException as exc:
            try:
                with self._lock:
                    state = self._read_state(smoke_id, device, plan)
                    self._transition(
                        state,
                        status="FAILED",
                        message=f"Smoke monitor failed safely: {type(exc).__name__}.",
                    )
            except BaseException:
                return
        finally:
            _forget_process(self.project_root, smoke_id, device, smoke_launch_identity(plan, device))
            if worker_job_handle:
                close_windows_handle(worker_job_handle)
            writable_anchors.close()

    def _stop_execution(
        self,
        plan: Mapping[str, Any],
        device: str,
        state: Mapping[str, Any],
        *,
        status: str,
        message: str,
    ) -> dict[str, Any]:
        if status not in {"CANCELLED", "TIMED_OUT"}:
            raise SmokeExecutionError("smoke_terminal_status", "The smoke terminal state is invalid.")
        process = _recalled_process(
            self.project_root,
            str(plan["smoke_id"]),
            device,
            smoke_launch_identity(plan, device),
        )
        if process is not None:
            _terminate_worker_process(process)
            if process.poll() is None:
                raise SmokeExecutionError(
                    "smoke_process_termination",
                    "The exact contained worker termination could not be verified.",
                )
        else:
            _terminate_recorded_worker(state)
        try:
            stopped = self._transition(
                state,
                status=status,
                exit_code=124 if status == "TIMED_OUT" else 130,
                message=message,
            )
        except SmokeExecutionError as exc:
            if exc.code != "smoke_transition_race":
                raise
            stopped = self._read_state(str(plan["smoke_id"]), device, plan)
        return _projection(stopped)

    def _publish_completion(
        self,
        state: Mapping[str, Any],
        plan: Mapping[str, Any],
        device: str,
        receipt_identity: Any,
        *,
        message: str,
    ) -> dict[str, Any]:
        try:
            return self._transition(
                state,
                status="COMPLETE",
                current=2,
                exit_code=0,
                receipt_identity=receipt_identity,
                message=message,
            )
        except SmokeExecutionError as exc:
            if exc.code != "smoke_transition_race":
                raise
            winner = self._read_state(str(state["smoke_id"]), device, plan)
            if (
                winner["status"] != "COMPLETE"
                or winner["current"] != 2
                or winner["exit_code"] != 0
                or winner["receipt_identity"] != receipt_identity
            ):
                raise
            self._require_completion_receipt(plan, device, receipt_identity)
            return winner

    def _transition(self, state: Mapping[str, Any], *, message: str, **changes: Any) -> dict[str, Any]:
        self._validate_state(state)
        if not set(changes).issubset(_TRANSITION_MUTABLE_FIELDS):
            raise SmokeExecutionError("smoke_transition_fields", "The smoke transition fields are invalid.")
        directory = self._state_path(str(state["smoke_id"]), str(state["device"])).parent
        with _state_transition_lock(directory):
            plan = load_plan(self.project_root, str(state["smoke_id"]))
            current = self._read_state(
                str(state["smoke_id"]),
                str(state["device"]),
                plan,
            )
            requested_status = changes.get("status", state["status"])
            if type(requested_status) is not str or requested_status not in _STATUSES:
                raise SmokeExecutionError("smoke_transition_status", "The smoke terminal transition was refused.")
            expected_token = (state["transition_sequence"], state[_STATE_IDENTITY_FIELD])
            current_token = (current["transition_sequence"], current[_STATE_IDENTITY_FIELD])
            if current_token != expected_token:
                raise SmokeExecutionError("smoke_transition_race", "The smoke execution state changed concurrently.")
            if current["status"] in _TERMINAL:
                terminal = _compatible_terminal_value(current, str(requested_status), changes)
                if terminal["status"] == "COMPLETE":
                    self._require_completion_receipt(plan, str(terminal["device"]), terminal["receipt_identity"])
                return terminal
            allowed = {
                "STARTING": {"RUNNING", "FAILED", "INTERRUPTED", "CANCELLED", "TIMED_OUT"},
                "RUNNING": {"RUNNING", "COMPLETE", "FAILED", "INTERRUPTED", "CANCELLED", "TIMED_OUT"},
            }
            if requested_status not in allowed.get(str(current["status"]), set()):
                raise SmokeExecutionError("smoke_transition_status", "The smoke terminal transition was refused.")
            value = dict(current)
            value.update(changes)
            value["transition_sequence"] = int(current["transition_sequence"]) + 1
            value["updated_at"] = _now()
            value["logs"] = [*list(value.get("logs") or ()), _safe_log_line(message, self.project_root)][-50:]
            value.pop(_STATE_IDENTITY_FIELD, None)
            value = _finalize_state_identity(value)
            self._validate_state(value)
            if value["status"] == "COMPLETE":
                self._require_completion_receipt(plan, str(value["device"]), value["receipt_identity"])
            with AnchoredDirectory(directory, directory) as anchor:
                anchor.atomic_write_bytes("state.json", canonical_json_bytes(value, pretty=True))
            persisted = self._read_state(
                str(value["smoke_id"]),
                str(value["device"]),
                plan,
            )
            if persisted != value:
                raise SmokeExecutionError("smoke_transition_publish", "The smoke execution transition changed.")
            return persisted

    def _read_state(self, smoke_id: str, device: str, plan: Mapping[str, Any]) -> dict[str, Any]:
        payload = read_stable_single_link_bytes(
            self._state_path(smoke_id, device),
            boundary=self._state_path(smoke_id, device).parent,
            max_bytes=4 * 1024 * 1024,
        )
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmokeExecutionError("smoke_execution_state", "Smoke execution state is invalid.") from exc
        if not isinstance(value, dict):
            raise SmokeExecutionError("smoke_execution_state", "Smoke execution state is invalid.")
        self._validate_state(value)
        if (
            value["smoke_id"] != smoke_id
            or value["device"] != device
            or value["plan_identity"] != plan["plan_identity"]
        ):
            raise SmokeExecutionError("smoke_execution_changed", "Smoke execution state belongs to another plan.")
        return value

    def _validate_state(self, value: Mapping[str, Any]) -> None:
        if (
            set(value) != _STATE_KEYS
            or any(type(value.get(key)) not in allowed for key, allowed in _STATE_FIELD_TYPES.items())
            or value.get("schema_version") != SMOKE_EXECUTION_SCHEMA
            or re.fullmatch(r"smoke-[0-9a-f]{20}", value["smoke_id"]) is None
            or value["device"] not in _DEVICES
            or value["status"] not in _STATUSES
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
            or value.get("resumable") is not False
            or value.get("retry_policy") != "NEW_BUNDLE_REQUIRED"
            or any(value.get(key) is not False for key in FALSE_ELIGIBILITY)
            or not _valid_state_identity(value)
        ):
            raise SmokeExecutionError("smoke_execution_state", "Smoke execution state is invalid.")
        device = value["device"]
        plan = load_plan(self.project_root, value["smoke_id"])
        expected_environment = dict(plan["configurations"][device]["environment"])
        expected_environment_identity = str(plan["configurations"][device]["child_environment"]["environment_sha256"])
        expected_training_argv = smoke_training_argv(plan, device)
        expected_worker_argv = smoke_worker_argv(plan, device)
        expected_mode = "windows-direct-trainer-v1" if os.name == "nt" else "linux-worker-trainer-v1"
        expected_launched_argv = expected_training_argv if os.name == "nt" else expected_worker_argv
        expected_writable_roots = [
            _directory_identity_record(
                self.project_root.joinpath(*Path(relative).parts),
                self.project_root,
            )
            for relative in plan["configurations"][device]["writable_roots"]
        ]
        worker_pid = value.get("worker_pid")
        worker_process = value.get("worker_process")
        worker_is_absent = worker_pid is None and worker_process is None
        worker_is_valid = _valid_worker_process_record(worker_process, worker_pid)
        status = value["status"]
        started = _parse_utc(value["started_at"])
        deadline = _parse_utc(value["deadline_at"])
        updated = _parse_utc(value["updated_at"])
        active_fields_are_valid = (
            value["current"] < 2 and value.get("exit_code") is None and value.get("receipt_identity") is None
        )
        exit_code = value.get("exit_code")
        exit_code_is_optional_failure = exit_code is None or (type(exit_code) is int and exit_code != 0)
        if (
            value.get("plan_identity") != plan["plan_identity"]
            or value.get("launch_identity") != smoke_launch_identity(plan, device)
            or value.get("portable_argv") != expected_training_argv
            or value.get("worker_argv") != expected_launched_argv
            or value.get("argv_identity") != stable_hash(value.get("portable_argv"))
            or value.get("worker_argv_identity") != stable_hash(value.get("worker_argv"))
            or value.get("execution_mode") != expected_mode
            or not _valid_confinement_binding(
                value.get("confinement"),
                mode=expected_mode,
                writable_roots=expected_writable_roots,
                status=status,
                project_root_identity=_project_root_identity(self.project_root),
            )
            or value.get("environment") != expected_environment
            or value.get("environment_identity") != expected_environment_identity
            or value.get("interpreter_identity") != dict(plan["interpreter"])["interpreter_identity"]
            or value.get("writable_roots") != expected_writable_roots
            or value.get("wall_clock_limit_seconds") != int(plan["configurations"][device]["wall_clock_limit_seconds"])
            or not _valid_deadline(value)
            or started is None
            or deadline is None
            or updated is None
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
                    or not re.fullmatch(r"[0-9a-f]{64}", value.get("receipt_identity") or "")
                )
            )
            or (
                status == "FAILED"
                and (
                    value["current"] >= 2
                    or value.get("receipt_identity") is not None
                    or not exit_code_is_optional_failure
                )
            )
            or (status == "INTERRUPTED" and (not active_fields_are_valid or value.get("exit_code") is not None))
            or (
                status == "CANCELLED"
                and (
                    value["current"] >= 2
                    or type(exit_code) is not int
                    or exit_code != 130
                    or value.get("receipt_identity") is not None
                )
            )
            or (
                status == "TIMED_OUT"
                and (
                    value["current"] >= 2
                    or type(exit_code) is not int
                    or exit_code != 124
                    or value.get("receipt_identity") is not None
                )
            )
        ):
            raise SmokeExecutionError("smoke_execution_state", "Smoke execution state is invalid.")

    def _require_completion_receipt(
        self,
        plan: Mapping[str, Any],
        device: str,
        receipt_identity: Any,
    ) -> None:
        if type(receipt_identity) is not str:
            raise SmokeExecutionError("smoke_completion_receipt", "The completion receipt binding is invalid.")
        try:
            receipt = load_device_receipt(self.project_root, plan, device)
        except (OSError, ValueError, SmokeBundleError) as exc:
            raise SmokeExecutionError(
                "smoke_completion_receipt",
                "The exact completion receipt is unavailable.",
            ) from exc
        if type(receipt) is not dict or receipt.get("receipt_identity") != receipt_identity:
            raise SmokeExecutionError("smoke_completion_receipt", "The completion receipt identity changed.")

    def _state_path(self, smoke_id: str, device: str) -> Path:
        return artifact_bundle_directory(self.project_root, smoke_id) / "execution" / _device(device) / "state.json"


def _not_started(smoke_id: str, device: str, plan_identity: str) -> dict[str, Any]:
    return {
        "smoke_id": smoke_id,
        "device": device,
        "plan_identity": plan_identity,
        "status": "NOT_STARTED",
        "current": 0,
        "total": 2,
        "logs": [],
        "resumable": False,
        "retry_policy": "NEW_BUNDLE_REQUIRED",
        **FALSE_ELIGIBILITY,
    }


def _projection(state: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: state.get(key)
        for key in (
            "smoke_id",
            "device",
            "plan_identity",
            "launch_identity",
            "status",
            "current",
            "total",
            "started_at",
            "deadline_at",
            "wall_clock_limit_seconds",
            "updated_at",
            "exit_code",
            "receipt_identity",
            "portable_argv",
            "worker_argv",
            "execution_mode",
            "confinement",
            "logs",
            "resumable",
            "retry_policy",
            *FALSE_ELIGIBILITY,
        )
        if key in state
    }
    if result.get("status") == "COMPLETE":
        result["execution_identity"] = stable_hash(
            {
                "smoke_id": result.get("smoke_id"),
                "device": result.get("device"),
                "plan_identity": result.get("plan_identity"),
                "status": "COMPLETE",
                "receipt_identity": result.get("receipt_identity"),
                "portable_argv": result.get("portable_argv"),
                "worker_argv": result.get("worker_argv"),
                "execution_mode": result.get("execution_mode"),
                "confinement": result.get("confinement"),
                "environment_identity": state.get("environment_identity"),
                "interpreter_identity": state.get("interpreter_identity"),
                "resumable": False,
                "retry_policy": "NEW_BUNDLE_REQUIRED",
            }
        )
    return result


def _finalize_state_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if _STATE_IDENTITY_FIELD in value:
        raise SmokeExecutionError("smoke_execution_state", "Smoke execution state is invalid.")
    try:
        value[_STATE_IDENTITY_FIELD] = stable_hash(value)
    except (TypeError, ValueError) as exc:
        raise SmokeExecutionError("smoke_execution_state", "Smoke execution state is invalid.") from exc
    return value


def _valid_state_identity(value: Mapping[str, Any]) -> bool:
    identity = value.get(_STATE_IDENTITY_FIELD)
    if type(identity) is not str or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        return False
    body = {key: item for key, item in value.items() if key != _STATE_IDENTITY_FIELD}
    try:
        return stable_hash(body) == identity
    except (TypeError, ValueError):
        return False


def _valid_worker_process_record(value: Any, pid: Any) -> bool:
    return (
        type(pid) is int
        and pid > 0
        and type(value) is dict
        and set(value) == {"pid", "birth_token", "process_image_path_sha256"}
        and type(value.get("pid")) is int
        and value["pid"] == pid
        and type(value.get("birth_token")) is str
        and bool(value["birth_token"])
        and type(value.get("process_image_path_sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["process_image_path_sha256"]) is not None
        and worker_process_identity_is_valid(value, pid)
    )


def _compatible_terminal_value(
    current: Mapping[str, Any],
    requested_status: str,
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an exact terminal retry and refuse every conflicting value.

    Compare-and-set mismatches are rejected before this helper is reached, so
    only a caller holding the exact durable terminal sequence and identity can
    request idempotence.  Every explicitly requested field must already match.
    """

    if requested_status != current["status"]:
        raise SmokeExecutionError("smoke_transition_race", "The smoke terminal state already differs.")
    for key, requested in changes.items():
        if key == "status":
            continue
        try:
            matches = canonical_json_bytes(current.get(key)) == canonical_json_bytes(requested)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise SmokeExecutionError("smoke_transition_race", "The smoke terminal state already differs.")
    return dict(current)


def _safe_log_line(value: str, root: Path) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    for private in {str(root), root.as_posix(), str(Path.home()), Path.home().as_posix(), sys.executable}:
        if private:
            text = text.replace(private, "<private-path>")
    text = re.sub(r"(?i)[a-z]:[\\/][^\s\"']+", "<private-path>", text)
    text = re.sub(r"(?<!\w)/(?:[^/\s]+/)+[^\s\"']*", "<private-path>", text)
    return text[:1_000]


def _write_anchored_exclusive(anchor: AnchoredDirectory, name: str, content: bytes) -> Path:
    """Publish one immutable child through the already-held writable-root handle."""

    descriptor = anchor.open_file_immovable(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
    )
    identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            if OwnedFileIdentity.from_stat(os.fstat(handle.fileno())) != identity:
                raise SmokeExecutionError("smoke_writable_root", "An anchored smoke file changed while writing.")
        if not identity.matches(anchor.lstat(name)):
            raise SmokeExecutionError("smoke_writable_root", "An anchored smoke file changed before publication.")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return anchor.directory / name


def _directory_identity_record(path: Path, root: Path) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
        raise SmokeExecutionError("smoke_writable_root", "A smoke writable root is unsafe.")
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


def _project_root_identity(root: Path) -> str:
    return stable_hash(
        {
            "normalized_project_root_sha256": stable_hash(
                os.path.normcase(os.fspath(root.resolve())).encode("utf-8", "surrogatepass").hex()
            )
        }
    )


def _windows_confinement_binding(process: Any, *roots: Path, project_root: Path) -> dict[str, Any]:
    writable_roots = [_directory_identity_record(root, project_root) for root in roots]
    restricted = getattr(process, "restricted_token", None)
    restricted_identity = getattr(process, "restricted_sid_hashes_identity_sha256", None)
    desktop_identity = getattr(process, "private_desktop_identity_sha256", None)
    bootstrap_identity = getattr(process, "bootstrap_identity_sha256", None)
    if (
        type(restricted) is not bool
        or type(restricted_identity) is not str
        or re.fullmatch(r"[0-9a-f]{64}", restricted_identity) is None
        or type(desktop_identity) is not str
        or re.fullmatch(r"[0-9a-f]{64}", desktop_identity) is None
        or bootstrap_identity != WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256
    ):
        raise SmokeExecutionError(
            "smoke_windows_confinement",
            "The Windows smoke process lacks its exact confinement binding.",
        )
    return {
        "schema_version": "spritelab.training.windows-smoke-confinement.v1",
        "strategy": WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY,
        "bootstrap_identity_sha256": bootstrap_identity,
        "project_root_identity_sha256": _project_root_identity(project_root),
        "private_desktop_identity_sha256": desktop_identity,
        "restricted_token": restricted,
        "restricted_sid_hashes_identity_sha256": restricted_identity,
        "job_kill_on_close": True,
        "job_active_process_limit": 1,
        "writable_roots": writable_roots,
        "paths_exposed": False,
    }


def _valid_confinement_binding(
    value: Any,
    *,
    mode: str,
    writable_roots: list[dict[str, str]],
    status: str,
    project_root_identity: str,
) -> bool:
    if mode == "linux-worker-trainer-v1":
        return value is None
    if value is None:
        return status in {"STARTING", "FAILED", "INTERRUPTED", "CANCELLED", "TIMED_OUT"}
    expected_keys = {
        "schema_version",
        "strategy",
        "bootstrap_identity_sha256",
        "project_root_identity_sha256",
        "private_desktop_identity_sha256",
        "restricted_token",
        "restricted_sid_hashes_identity_sha256",
        "job_kill_on_close",
        "job_active_process_limit",
        "writable_roots",
        "paths_exposed",
    }
    return (
        isinstance(value, Mapping)
        and set(value) == expected_keys
        and value.get("schema_version") == "spritelab.training.windows-smoke-confinement.v1"
        and value.get("strategy") == WINDOWS_BOOTSTRAP_UNTRUSTED_STRATEGY
        and value.get("bootstrap_identity_sha256") == WINDOWS_UNTRUSTED_BOOTSTRAP_SHA256
        and value.get("project_root_identity_sha256") == project_root_identity
        and type(value.get("private_desktop_identity_sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["private_desktop_identity_sha256"]) is not None
        and type(value.get("restricted_token")) is bool
        and type(value.get("restricted_sid_hashes_identity_sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["restricted_sid_hashes_identity_sha256"]) is not None
        and value.get("job_kill_on_close") is True
        and value.get("job_active_process_limit") == 1
        and value.get("writable_roots") == writable_roots
        and value.get("paths_exposed") is False
    )


@contextmanager
def _state_transition_lock(directory: Path, *, timeout: float = 5.0) -> Any:
    """Serialize one execution state's compare-and-set publication.

    The lock is an exact single-link file in the already-owned execution
    directory.  Acquisition is bounded so a corrupt or abandoned lock holder
    cannot leave API threads waiting forever.
    """

    if not math.isfinite(timeout) or timeout <= 0:
        raise SmokeExecutionError("smoke_transition_lock", "The smoke transition-lock timeout is invalid.")
    with AnchoredDirectory(directory, directory) as anchor:
        with _fixed_lock_authority(
            anchor,
            ".state-transition.lock",
            timeout=timeout,
            code="smoke_transition_lock",
            unsafe_message="The smoke transition lock is unsafe.",
            changed_message="The smoke transition lock changed.",
            timeout_message="Timed out waiting for the smoke transition lock.",
        ):
            yield


@contextmanager
def _project_launch_lock(root: Path, *, timeout: float = 5.0) -> Any:
    if not math.isfinite(timeout) or timeout <= 0:
        raise SmokeExecutionError("smoke_launch_lock", "The project smoke launch-lock timeout is invalid.")
    parent = ensure_managed_directory(root, ("artifacts", "training", "smokes"), boundary=root)
    with anchored_directory(parent, root) as anchor:
        with _fixed_lock_authority(
            anchor,
            ".execution-launch.lock",
            timeout=timeout,
            code="smoke_launch_lock",
            unsafe_message="The project smoke launch lock is unsafe.",
            changed_message="The project smoke launch lock changed.",
            timeout_message="Timed out waiting for the project smoke launch lock.",
        ):
            yield


@contextmanager
def _fixed_lock_authority(
    anchor: AnchoredDirectory,
    name: str,
    *,
    timeout: float,
    code: str,
    unsafe_message: str,
    changed_message: str,
    timeout_message: str,
) -> Any:
    """Hold one fixed lock without allowing a second pathname authority."""

    descriptor = -1
    authority_descriptor = -1
    acquired = False
    post_yield_error: SmokeExecutionError | None = None
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
            code=code,
            message=unsafe_message,
        )
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        if os.name == "nt":
            authority_descriptor = descriptor
        else:
            authority_descriptor = _open_posix_directory_lock_authority(
                anchor,
                code=code,
                message=unsafe_message,
            )
        while not acquired:
            try:
                _lock_fixed_authority(authority_descriptor)
                acquired = True
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise SmokeExecutionError(code, timeout_message) from None
                time.sleep(0.02)
        _verify_fixed_lock_authority(
            anchor,
            name,
            descriptor,
            identity,
            code=code,
            message=changed_message,
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
                    code=code,
                    message=changed_message,
                )
            except SmokeExecutionError as exc:
                post_yield_error = exc
    finally:
        release_error: SmokeExecutionError | None = None
        if acquired:
            try:
                _verify_fixed_lock_authority(
                    anchor,
                    name,
                    descriptor,
                    identity,
                    code=code,
                    message=changed_message,
                )
            except SmokeExecutionError as exc:
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
    code: str,
    message: str,
) -> None:
    try:
        anchor.verify()
        opened = os.fstat(descriptor)
        current = anchor.lstat(name)
    except (OSError, ValueError) as exc:
        raise SmokeExecutionError(code, message) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or int(getattr(opened, "st_nlink", 1)) != 1
        or int(getattr(current, "st_nlink", 1)) != 1
        or not identity.matches(opened)
        or not identity.matches(current)
    ):
        raise SmokeExecutionError(code, message)


def _open_posix_directory_lock_authority(
    anchor: AnchoredDirectory,
    *,
    code: str,
    message: str,
) -> int:
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
            raise SmokeExecutionError(code, message)
        return descriptor
    except (OSError, ValueError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(exc, SmokeExecutionError):
            raise
        raise SmokeExecutionError(code, message) from exc


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


def _process_key(root: Path, smoke_id: str, device: str) -> tuple[str, str, str]:
    return (os.path.normcase(str(root)), smoke_id, device)


def _remember_process(root: Path, smoke_id: str, device: str, launch_identity: str, process: Any) -> None:
    with _LIVE_PROCESS_LOCK:
        _LIVE_PROCESSES[_process_key(root, smoke_id, device)] = (launch_identity, process)


def _recalled_process(root: Path, smoke_id: str, device: str, launch_identity: str) -> Any | None:
    with _LIVE_PROCESS_LOCK:
        value = _LIVE_PROCESSES.get(_process_key(root, smoke_id, device))
        if value is None or value[0] != launch_identity:
            return None
        return value[1]


def _forget_process(root: Path, smoke_id: str, device: str, launch_identity: str) -> None:
    with _LIVE_PROCESS_LOCK:
        key = _process_key(root, smoke_id, device)
        value = _LIVE_PROCESSES.get(key)
        if value is not None and value[0] == launch_identity:
            _LIVE_PROCESSES.pop(key, None)


def _terminate_worker_process(process: Any) -> None:
    if sys.platform.startswith("linux") and type(getattr(process, "pid", None)) is int:
        _terminate_linux_process_group(process, int(process.pid))
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


def _terminate_linux_process_group(process: Any, process_group_id: int) -> None:
    """Terminate every descendant group member even after its leader exits."""

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        _reap_process(process)
        return
    except OSError:
        # A transient TERM failure must not suppress the bounded KILL attempt.
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            process.poll()
        except BaseException:
            pass
        if not _linux_process_group_exists(process_group_id):
            _reap_process(process)
            return
        time.sleep(0.05)
    if _linux_process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    _reap_process(process)


def _linux_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _reap_process(process: Any) -> None:
    try:
        process.wait(timeout=5)
    except BaseException:
        return


def _terminate_recorded_linux_process_group(
    state: Mapping[str, Any],
    *,
    grace_seconds: float = 5.0,
    pidfd_open: Callable[[int, int], int] | None = None,
    get_process_group: Callable[[int], int] | None = None,
    kill_group: Callable[[int, int], None] | None = None,
    group_exists: Callable[[int], bool] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    close_descriptor: Callable[[int], None] | None = None,
) -> None:
    """Terminate one identity-pinned recorded worker process group.

    The worker is launched as its own POSIX session, so its bound PID must also
    be its process-group ID.  A Linux pidfd retains that exact leader identity
    across the grace interval; without it, numeric PID/PGID reuse could make a
    later group signal target unrelated processes and termination is refused.
    """

    if not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise SmokeExecutionError("smoke_process_group_identity", "The process-group grace is invalid.")
    pid = state.get("worker_pid")
    if type(pid) is not int or pid <= 0:
        raise SmokeExecutionError(
            "smoke_process_group_identity",
            "The recorded worker process-group identity is unavailable.",
        )
    opener = pidfd_open or getattr(os, "pidfd_open", None)
    group_reader = get_process_group or getattr(os, "getpgid", None)
    sender = kill_group or getattr(os, "killpg", None)
    if opener is None or group_reader is None or sender is None:
        raise SmokeExecutionError(
            "smoke_process_group_identity",
            "Exact recorded worker process-group termination is unavailable.",
        )
    exists = group_exists or _linux_process_group_exists
    clock = monotonic or time.monotonic
    pause = sleep or time.sleep
    closer = close_descriptor or os.close
    descriptor = -1
    try:
        descriptor = opener(pid, 0)
        if type(descriptor) is not int or descriptor < 0:
            raise SmokeExecutionError(
                "smoke_process_group_identity",
                "The recorded worker process identity could not be pinned.",
            )
        if worker_process_matches(state) is not True:
            raise SmokeExecutionError(
                "smoke_process_group_identity",
                "The recorded worker process identity changed before termination.",
            )
        process_group_id = group_reader(pid)
        if type(process_group_id) is not int or process_group_id != pid:
            raise SmokeExecutionError(
                "smoke_process_group_identity",
                "The recorded worker does not own its expected process group.",
            )
        try:
            sender(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            # The pinned group remains safe to target with the stronger signal.
            pass
        deadline = clock() + grace_seconds
        while clock() < deadline:
            if not exists(process_group_id):
                return
            pause(min(0.05, max(0.0, deadline - clock())))
        if exists(process_group_id):
            try:
                sender(process_group_id, int(getattr(signal, "SIGKILL", 9)))
            except ProcessLookupError:
                return
            except OSError as exc:
                raise SmokeExecutionError(
                    "smoke_process_group_termination",
                    "The exact recorded worker process group could not be terminated.",
                ) from exc
    except SmokeExecutionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise SmokeExecutionError(
            "smoke_process_group_identity",
            "The recorded worker process-group identity could not be established.",
        ) from exc
    finally:
        if descriptor >= 0:
            try:
                closer(descriptor)
            except OSError:
                pass


def _terminate_recorded_worker(state: Mapping[str, Any]) -> None:
    pid = state.get("worker_pid")
    if type(pid) is not int or not worker_process_identity_is_valid(state.get("worker_process"), pid):
        raise SmokeExecutionError(
            "smoke_process_identity",
            "The exact recorded worker process identity is unavailable.",
        )
    process_matches = worker_process_matches(state)
    if process_matches is False and state.get("execution_mode") == "windows-direct-trainer-v1":
        # Windows smoke runs exactly one direct trainer in a one-process,
        # kill-on-close Job.  A changed or definitively absent PID proves the
        # recorded trainer exited; never signal a potentially reused PID.
        return
    if process_matches is not True:
        raise SmokeExecutionError(
            "smoke_process_identity",
            "The exact recorded worker process identity could not be verified.",
        )
    if sys.platform.startswith("linux"):
        _terminate_recorded_linux_process_group(state)
        return
    if os.name == "nt":
        _terminate_recorded_windows_worker(state)
        return
    raise SmokeExecutionError(
        "smoke_process_termination",
        "Exact recorded worker termination is unavailable on this platform.",
    )


def _terminate_recorded_windows_worker(
    state: Mapping[str, Any],
    *,
    kernel32: Any | None = None,
) -> None:
    """Terminate one identity-bound Windows process through a pointer-sized handle."""

    import ctypes
    from ctypes import wintypes

    pid = state.get("worker_pid")
    marker = state.get("worker_process")
    if type(pid) is not int or not isinstance(marker, Mapping):
        raise SmokeExecutionError("smoke_process_identity", "The recorded Windows worker identity is unavailable.")
    api = kernel32 or ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessId.argtypes = [wintypes.HANDLE]
    api.GetProcessId.restype = wintypes.DWORD
    api.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    api.GetProcessTimes.restype = wintypes.BOOL
    api.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    api.QueryFullProcessImageNameW.restype = wintypes.BOOL
    api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    api.TerminateProcess.restype = wintypes.BOOL
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.GetExitCodeProcess.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.OpenProcess(0x0001 | 0x00100000 | 0x1000, False, pid)
    if not handle:
        raise SmokeExecutionError(
            "smoke_process_identity",
            "The exact recorded Windows worker handle could not be opened.",
        )
    try:
        if int(api.GetProcessId(handle)) != pid or _windows_worker_handle_identity(api, handle, pid) != dict(marker):
            raise SmokeExecutionError(
                "smoke_process_identity",
                "The recorded Windows worker identity changed before termination.",
            )
        wait_result = int(api.WaitForSingleObject(handle, 0))
        if wait_result == 0x102:
            if not api.TerminateProcess(handle, 130):
                raise SmokeExecutionError(
                    "smoke_process_termination",
                    "The exact recorded Windows worker could not be terminated.",
                )
            wait_result = int(api.WaitForSingleObject(handle, 5_000))
        if wait_result != 0:
            raise SmokeExecutionError(
                "smoke_process_termination",
                "The exact recorded Windows worker termination was not observed.",
            )
        exit_code = wintypes.DWORD()
        if not api.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or int(exit_code.value) == 259:
            raise SmokeExecutionError(
                "smoke_process_termination",
                "The exact recorded Windows worker exit could not be verified.",
            )
        if _windows_worker_handle_identity(api, handle, pid) != dict(marker):
            raise SmokeExecutionError(
                "smoke_process_identity",
                "The recorded Windows worker identity changed during termination.",
            )
    finally:
        api.CloseHandle(handle)


def _windows_worker_handle_identity(api: Any, handle: Any, pid: int) -> dict[str, Any]:
    import ctypes
    from ctypes import wintypes

    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not api.GetProcessTimes(
        handle,
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise SmokeExecutionError("smoke_process_identity", "The recorded Windows worker birth time is unavailable.")
    length = wintypes.DWORD(32_768)
    buffer = ctypes.create_unicode_buffer(length.value)
    if not api.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
        raise SmokeExecutionError("smoke_process_identity", "The recorded Windows worker image is unavailable.")
    birth = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
    executable = os.path.normcase(buffer.value)
    return {
        "pid": pid,
        "birth_token": str(birth),
        "process_image_path_sha256": hashlib.sha256(executable.encode("utf-8", "surrogatepass")).hexdigest(),
    }


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


def _valid_deadline(state: Mapping[str, Any]) -> bool:
    started = _parse_utc(state.get("started_at"))
    deadline = _parse_utc(state.get("deadline_at"))
    limit = state.get("wall_clock_limit_seconds")
    return (
        started is not None
        and deadline is not None
        and type(limit) is int
        and limit > 0
        and deadline == started + timedelta(seconds=limit)
    )


def _deadline_expired(state: Mapping[str, Any]) -> bool:
    deadline = _parse_utc(state.get("deadline_at"))
    return deadline is None or datetime.now(timezone.utc) >= deadline


def _age_seconds(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return float("inf")
    if parsed.tzinfo is None:
        return float("inf")
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())


def _device(value: str) -> str:
    normalized = str(value).lower()
    if normalized not in _DEVICES:
        raise SmokeExecutionError("smoke_device", "The smoke device must be CPU or CUDA.")
    return normalized


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["SMOKE_EXECUTION_SCHEMA", "ExploratorySmokeRunner", "SmokeExecutionError"]
