"""Durable, fixed-plan local execution for exploratory smoke bundles."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.training.smoke_bundle import (
    FALSE_ELIGIBILITY,
    SmokeBundleError,
    anchored_directory,
    anchored_path_is_absent,
    artifact_bundle_directory,
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
from spritelab.utils.safe_fs import OwnedFileIdentity

SMOKE_EXECUTION_SCHEMA = "spritelab.training.smoke-device-execution.v1"
_DEVICES = ("cpu", "cuda")
_TERMINAL = {"COMPLETE", "FAILED", "INTERRUPTED"}
_LIVE_PROCESS_LOCK = threading.RLock()
_LIVE_PROCESSES: dict[tuple[str, str, str], tuple[str, Any]] = {}


class SmokeExecutionError(SmokeBundleError):
    """A fixed-plan process launch or durable execution-state failure."""


class ExploratorySmokeRunner:
    """Launch only a server-prepared smoke argv and reconstruct it passively."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        process_factory: Callable[..., Any] | None = None,
        windows_suspended_activator: Callable[..., int] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._process_factory = process_factory or subprocess.Popen
        self._windows_suspended_activator = windows_suspended_activator or activate_windows_suspended_process
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
        normalized = _device(device)
        if not smoke_containment_supported():
            raise SmokeExecutionError(
                "smoke_containment_unavailable",
                "Server-run exploratory smoke containment is unavailable on this platform.",
            )
        plan = load_plan(self.project_root, smoke_id)
        if plan.get("plan_identity") != plan_identity:
            raise SmokeExecutionError("smoke_plan_changed", "The selected smoke plan identity changed.")
        validate_smoke_interpreter(plan)
        verify_execution_guards(self.project_root, plan)
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
            self._require_no_active_smoke()
            output = run_bundle_directory(self.project_root, smoke_id) / normalized
            if not anchored_path_is_absent(output, self.project_root):
                raise SmokeExecutionError(
                    "smoke_unowned_output",
                    "Device output exists without a server-owned execution; prepare a fresh smoke bundle.",
                )
            bundle = artifact_bundle_directory(self.project_root, smoke_id)
            directory = ensure_managed_directory(bundle, ("execution",), boundary=self.project_root)
            ensure_managed_directory(bundle, ("execution", "temp", normalized), boundary=self.project_root)
            training_argv = smoke_training_argv(plan, normalized)
            worker_argv = smoke_worker_argv(plan, normalized)
            launch_identity = smoke_launch_identity(plan, normalized)
            public_environment = dict(plan["configurations"][normalized]["environment"])
            child_environment = build_smoke_child_environment(self.project_root, plan, normalized)
            environment_identity = str(plan["configurations"][normalized]["child_environment"]["environment_sha256"])
            state = {
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
                "worker_argv": worker_argv,
                "argv_identity": stable_hash(training_argv),
                "worker_argv_identity": stable_hash(worker_argv),
                "environment": public_environment,
                "environment_identity": environment_identity,
                "interpreter_identity": dict(plan["interpreter"])["interpreter_identity"],
                "started_at": _now(),
                "updated_at": _now(),
                "exit_code": None,
                "receipt_identity": None,
                "retry_policy": "NEW_BUNDLE_REQUIRED",
                "resumable": False,
                "logs": ["Contained exploratory smoke launch requested."],
                **FALSE_ELIGIBILITY,
            }
            try:
                write_exclusive_bytes(
                    directory / f"{normalized}.json",
                    canonical_json_bytes(state, pretty=True),
                    boundary=self.project_root,
                )
            except FileExistsError:
                return self.status(smoke_id, normalized)
            process: Any | None = None
            worker_job_handle: int | None = None
            try:
                with pinned_smoke_interpreter(plan) as interpreter:
                    argv = [interpreter.launch_path, *worker_argv[1:]]
                    process_options: dict[str, Any] = {
                        "cwd": self.project_root,
                        "env": child_environment,
                        "stdin": subprocess.DEVNULL,
                        "stdout": subprocess.DEVNULL,
                        "stderr": subprocess.DEVNULL,
                        "shell": False,
                    }
                    if os.name == "nt":
                        process_options["creationflags"] = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
                            getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
                        )
                    else:
                        process_options["start_new_session"] = True
                        process_options["pass_fds"] = interpreter.pass_fds
                        if sys.platform.startswith("linux"):
                            process_options["preexec_fn"] = linux_parent_death_signal(os.getpid())
                    process = self._process_factory(argv, **process_options)
                    if os.name == "nt":
                        worker_job_handle = self._windows_suspended_activator(
                            process,
                            verifier=lambda launched: verify_pinned_process_image(
                                launched,
                                interpreter,
                            ),
                        )
                    else:
                        verify_pinned_process_image(process, interpreter)
            except BaseException as exc:
                if process is not None:
                    _terminate_worker_process(process)
                if worker_job_handle:
                    close_windows_handle(worker_job_handle)
                failed = self._transition(
                    state,
                    status="FAILED",
                    exit_code=None,
                    message=f"Contained worker launch failed: {type(exc).__name__}.",
                )
                return _projection(failed)
            try:
                worker_process = capture_worker_process_identity(int(process.pid))
                if worker_process is None:
                    raise SmokeExecutionError(
                        "smoke_worker_identity",
                        "The contained smoke worker identity could not be established.",
                    )
                state = self._transition(
                    state,
                    status="RUNNING",
                    worker_pid=int(process.pid),
                    worker_process=worker_process,
                    message=f"Contained {normalized.upper()} two-step smoke worker is running.",
                )
            except BaseException as exc:
                _terminate_worker_process(process)
                if worker_job_handle:
                    close_windows_handle(worker_job_handle)
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
                    args=(plan, normalized, process, worker_job_handle),
                    name=f"spritelab-{smoke_id}-{normalized}",
                    daemon=True,
                )
                monitor.start()
            except BaseException as exc:
                _forget_process(self.project_root, smoke_id, normalized, launch_identity)
                _terminate_worker_process(process)
                if worker_job_handle:
                    close_windows_handle(worker_job_handle)
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
            if anchored_path_is_absent(state_path, self.project_root):
                return _not_started(smoke_id, normalized, str(plan["plan_identity"]))
            state = self._read_state(smoke_id, normalized, plan)
        except FileNotFoundError:
            return _not_started(smoke_id, normalized, str(plan["plan_identity"]))
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
                return _completed_projection(state, receipt, updated_at=_now())
            projected = dict(state)
            projected.update({"status": "FAILED", "exit_code": int(code), "updated_at": _now()})
            return _projection(projected)
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
                return _completed_projection(state, receipt, updated_at=str(outcome["finished_at"]))
            projected = dict(state)
            projected.update(
                {
                    "status": "FAILED",
                    "exit_code": int(outcome["exit_code"]),
                    "updated_at": str(outcome["finished_at"]),
                }
            )
            projected["logs"] = [*projected["logs"], "Contained smoke worker ended without valid completion."][-50:]
            return _projection(projected)
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
        projected = dict(state)
        projected.update({"status": "INTERRUPTED", "updated_at": _now()})
        projected["logs"] = [
            *projected["logs"],
            "The contained worker ended before a completion receipt; a fresh bundle is required.",
        ][-50:]
        return _projection(projected)

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
                state_path = artifact_bundle_directory(self.project_root, name) / "execution" / f"{device}.json"
                if anchored_path_is_absent(state_path, self.project_root):
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
    ) -> None:
        smoke_id = str(plan["smoke_id"])
        try:
            code = int(process.wait())
            with self._lock:
                state = self._read_state(smoke_id, device, plan)
                if code == 0:
                    try:
                        receipt = load_device_receipt(self.project_root, plan, device)
                    except (OSError, ValueError, SmokeBundleError):
                        receipt = None
                    if receipt is not None:
                        self._transition(
                            state,
                            status="COMPLETE",
                            current=2,
                            exit_code=0,
                            receipt_identity=receipt["receipt_identity"],
                            message="Fixed two-step smoke completed with an immutable receipt.",
                        )
                        return
                self._transition(
                    state,
                    status="FAILED",
                    exit_code=code,
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

    def _transition(self, state: Mapping[str, Any], *, message: str, **changes: Any) -> dict[str, Any]:
        value = dict(state)
        value.update(changes)
        value["updated_at"] = _now()
        value["logs"] = [*list(value.get("logs") or ()), _safe_log_line(message, self.project_root)][-50:]
        self._validate_state(value)
        directory = self._state_path(str(value["smoke_id"]), str(value["device"])).parent
        with anchored_directory(directory, self.project_root) as anchor:
            anchor.atomic_write_bytes(f"{value['device']}.json", canonical_json_bytes(value, pretty=True))
        return value

    def _read_state(self, smoke_id: str, device: str, plan: Mapping[str, Any]) -> dict[str, Any]:
        payload = read_stable_single_link_bytes(
            self._state_path(smoke_id, device),
            boundary=self.project_root,
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
        device = _device(str(value.get("device") or ""))
        plan = load_plan(self.project_root, str(value.get("smoke_id") or ""))
        expected_environment = dict(plan["configurations"][device]["environment"])
        expected_environment_identity = str(plan["configurations"][device]["child_environment"]["environment_sha256"])
        expected_training_argv = smoke_training_argv(plan, device)
        expected_worker_argv = smoke_worker_argv(plan, device)
        worker_pid = value.get("worker_pid")
        worker_process = value.get("worker_process")
        if (
            value.get("schema_version") != SMOKE_EXECUTION_SCHEMA
            or value.get("status") not in {"STARTING", "RUNNING", "COMPLETE", "FAILED", "INTERRUPTED"}
            or not isinstance(value.get("smoke_id"), str)
            or not isinstance(value.get("plan_identity"), str)
            or value.get("launch_identity") != smoke_launch_identity(plan, device)
            or value.get("portable_argv") != expected_training_argv
            or value.get("worker_argv") != expected_worker_argv
            or value.get("argv_identity") != stable_hash(value.get("portable_argv"))
            or value.get("worker_argv_identity") != stable_hash(value.get("worker_argv"))
            or value.get("environment") != expected_environment
            or value.get("environment_identity") != expected_environment_identity
            or value.get("interpreter_identity") != dict(plan["interpreter"])["interpreter_identity"]
            or (
                value.get("status") == "RUNNING"
                and (type(worker_pid) is not int or not worker_process_identity_is_valid(worker_process, worker_pid))
            )
            or (
                worker_process is not None
                and (type(worker_pid) is not int or not worker_process_identity_is_valid(worker_process, worker_pid))
            )
            or value.get("resumable") is not False
            or value.get("retry_policy") != "NEW_BUNDLE_REQUIRED"
            or any(value.get(key) is not False for key in FALSE_ELIGIBILITY)
            or not isinstance(value.get("logs"), list)
        ):
            raise SmokeExecutionError("smoke_execution_state", "Smoke execution state is invalid.")

    def _state_path(self, smoke_id: str, device: str) -> Path:
        return artifact_bundle_directory(self.project_root, smoke_id) / "execution" / f"{_device(device)}.json"


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


def _completed_projection(
    state: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    updated_at: str,
) -> dict[str, Any]:
    projected = dict(state)
    projected.update(
        {
            "status": "COMPLETE",
            "current": 2,
            "exit_code": 0,
            "receipt_identity": receipt["receipt_identity"],
            "updated_at": updated_at,
        }
    )
    projected["logs"] = [*projected["logs"], "Contained worker exit and completion receipt verified."][-50:]
    return _projection(projected)


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
            "updated_at",
            "exit_code",
            "receipt_identity",
            "portable_argv",
            "worker_argv",
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
                "environment_identity": state.get("environment_identity"),
                "interpreter_identity": state.get("interpreter_identity"),
                "resumable": False,
                "retry_policy": "NEW_BUNDLE_REQUIRED",
            }
        )
    return result


def _safe_log_line(value: str, root: Path) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    for private in {str(root), root.as_posix(), str(Path.home()), Path.home().as_posix(), sys.executable}:
        if private:
            text = text.replace(private, "<private-path>")
    text = re.sub(r"(?i)[a-z]:[\\/][^\s\"']+", "<private-path>", text)
    text = re.sub(r"(?<!\w)/(?:[^/\s]+/)+[^\s\"']*", "<private-path>", text)
    return text[:1_000]


@contextmanager
def _project_launch_lock(root: Path) -> Any:
    parent = ensure_managed_directory(root, ("artifacts", "training", "smokes"), boundary=root)
    descriptor = -1
    locked = False
    with anchored_directory(parent, root) as anchor:
        descriptor = anchor.open_file(
            ".execution-launch.lock",
            os.O_RDWR | os.O_CREAT | int(getattr(os, "O_BINARY", 0)),
        )
        try:
            metadata = os.fstat(descriptor)
            identity = OwnedFileIdentity.from_stat(metadata)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or int(getattr(metadata, "st_nlink", 1)) != 1
                or not identity.matches(anchor.lstat(".execution-launch.lock"))
            ):
                raise SmokeExecutionError("smoke_launch_lock", "The project smoke launch lock is unsafe.")
            if metadata.st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            if not identity.matches(anchor.lstat(".execution-launch.lock")):
                raise SmokeExecutionError("smoke_launch_lock", "The project smoke launch lock changed.")
            yield
        finally:
            if descriptor >= 0 and locked:
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            if descriptor >= 0:
                os.close(descriptor)


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
