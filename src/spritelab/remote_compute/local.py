"""Idempotent local-process compute backend."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spritelab.product_core import ProductCapability, ProductEvent, ProductStatus, ProjectContext
from spritelab.remote_compute.contracts import (
    ArtifactReference,
    ArtifactVerificationError,
    ComputeEstimate,
    ComputeJob,
    ComputeJobRequest,
    ComputePoll,
    ComputeStatus,
    OperationResult,
    PreparedCompute,
    ResumeRequest,
    StaleRemoteIdentityError,
    verify_compute_job_request,
    verify_launch_authorization_capability,
    verify_local_project_root_binding,
)
from spritelab.remote_compute.utils import file_sha256, stable_hash, validate_identifier

if TYPE_CHECKING:
    from spritelab.training.launch import TrainingFilesystemCapability

ProcessFactory = Callable[..., Any]

_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
_NVIDIA_SMI_MAX_OUTPUT_BYTES = 16 * 1024
_MAX_LOCAL_EVENT_STREAM_BYTES = 64 * 1024 * 1024
_NVIDIA_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_NVIDIA_COMPUTE_MODES = frozenset({"default", "exclusive_thread", "exclusive_process", "prohibited"})


@dataclass
class _LocalJobRecord:
    job: ComputeJob
    process: Any
    capability: TrainingFilesystemCapability | None
    event_name: str | None
    cached_event_bytes: bytes = b""
    terminal_events_captured: bool = False
    released_by_cleanup: bool = False
    monitoring_uncertain: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


def _local_device_policy(context: ProjectContext) -> str:
    compute = context.config.get("compute") if isinstance(context.config, Mapping) else None
    training = compute.get("training") if isinstance(compute, Mapping) else None
    raw = training.get("device_policy", "auto") if isinstance(training, Mapping) else "auto"
    policy = str(raw or "auto").strip().lower()
    return policy if policy in {"auto", "cpu", "cuda"} else "invalid"


def _nvidia_smi_executable() -> Path | None:
    """Find NVIDIA's utility only in fixed administrator-controlled locations."""

    candidates: list[Path] = []
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot")
        if system_root and Path(system_root).is_absolute():
            candidates.append(Path(system_root) / "System32" / "nvidia-smi.exe")
        for variable in ("ProgramW6432", "ProgramFiles"):
            program_files = os.environ.get(variable)
            if program_files and Path(program_files).is_absolute():
                candidates.append(Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe")
    else:
        candidates.extend((Path("/usr/bin/nvidia-smi"), Path("/usr/local/bin/nvidia-smi")))
    for candidate in dict.fromkeys(candidates):
        try:
            if candidate.is_file():
                return candidate.resolve(strict=True)
        except OSError:
            continue
    return None


def _terminate_probe_process(process: Any) -> None:
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run_bounded_probe_command(command: Sequence[str]) -> dict[str, Any]:
    """Capture at most the declared stdout budget and discard stderr."""

    options: dict[str, Any] = {
        "bufsize": 0,
        "shell": False,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(list(command), **options)
    except OSError:
        return {"status": "unavailable"}
    stream = process.stdout
    if stream is None:
        _terminate_probe_process(process)
        return {"status": "io_error"}
    output = bytearray()
    overflow = threading.Event()
    read_error = threading.Event()

    def reader() -> None:
        try:
            while True:
                remaining = _NVIDIA_SMI_MAX_OUTPUT_BYTES + 1 - len(output)
                if remaining <= 0:
                    overflow.set()
                    return
                chunk = stream.read(min(4_096, remaining))
                if not chunk:
                    return
                if not isinstance(chunk, bytes):
                    read_error.set()
                    return
                output.extend(chunk)
                if len(output) > _NVIDIA_SMI_MAX_OUTPUT_BYTES:
                    overflow.set()
                    return
        except (OSError, ValueError):
            read_error.set()

    thread = threading.Thread(target=reader, name="spritelab-nvidia-probe-reader", daemon=True)
    thread.start()
    deadline = time.monotonic() + _NVIDIA_SMI_TIMEOUT_SECONDS
    timed_out = False
    while not overflow.is_set() and not read_error.is_set():
        if process.poll() is not None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        time.sleep(min(0.01, remaining))
    if timed_out or overflow.is_set() or read_error.is_set():
        _terminate_probe_process(process)
        try:
            stream.close()
        except OSError:
            pass
    thread.join(timeout=1.0)
    if thread.is_alive():
        try:
            stream.close()
        except OSError:
            pass
        thread.join(timeout=0.1)
    try:
        stream.close()
    except OSError:
        pass
    if timed_out:
        return {"status": "timeout"}
    if overflow.is_set():
        return {"status": "output_limit"}
    if read_error.is_set() or thread.is_alive():
        return {"status": "io_error"}
    return {"status": "ok", "returncode": process.returncode, "stdout": bytes(output)}


def _probe_nvidia_host() -> dict[str, Any]:
    """Use a fixed, bounded host command without importing or initializing Torch."""

    executable = _nvidia_smi_executable()
    if executable is None:
        return {"ready": False, "reason": "nvidia_smi_missing"}
    command = [
        str(executable),
        "--id=0",
        "--query-gpu=index,uuid,compute_mode",
        "--format=csv,noheader,nounits",
    ]
    result = _run_bounded_probe_command(command)
    if result["status"] == "timeout":
        return {"ready": False, "reason": "nvidia_smi_timeout"}
    if result["status"] in {"unavailable", "io_error"}:
        return {"ready": False, "reason": "nvidia_smi_unavailable"}
    if result["status"] == "output_limit":
        return {"ready": False, "reason": "nvidia_smi_output_limit"}
    if result.get("returncode") != 0:
        return {"ready": False, "reason": "nvidia_smi_failed"}
    try:
        stdout = bytes(result.get("stdout") or b"").decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return {"ready": False, "reason": "nvidia_smi_malformed"}
    if not stdout:
        return {"ready": False, "reason": "nvidia_smi_malformed"}
    try:
        rows = list(csv.reader(io.StringIO(stdout, newline=""), strict=True))
    except (csv.Error, UnicodeError):
        return {"ready": False, "reason": "nvidia_smi_malformed"}
    if len(rows) != 1:
        return {"ready": False, "reason": "nvidia_smi_malformed"}
    for row in rows:
        values = [value.strip() for value in row]
        if len(values) != 3:
            return {"ready": False, "reason": "nvidia_smi_malformed"}
        index, uuid, mode = values
        normalized_mode = mode.lower().replace(" ", "_")
        if index != "0" or not _NVIDIA_GPU_UUID.fullmatch(uuid):
            return {"ready": False, "reason": "nvidia_smi_malformed"}
        if normalized_mode not in _NVIDIA_COMPUTE_MODES:
            return {"ready": False, "reason": "nvidia_smi_malformed"}
        if normalized_mode == "prohibited":
            return {"ready": False, "reason": "nvidia_smi_compute_prohibited"}
    return {"ready": True, "reason": "nvidia_smi_pass", "usable_device_count": 1}


class LocalComputeBackend:
    backend_id = "local"
    title = "Local computer"
    is_cloud = False

    def __init__(self, *, process_factory: ProcessFactory | None = None) -> None:
        self._process_factory = process_factory or subprocess.Popen
        self._prepared: dict[str, PreparedCompute] = {}
        self._jobs: dict[str, _LocalJobRecord] = {}

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]:
        free = shutil.disk_usage(context.project_root).free
        policy = _local_device_policy(context)
        details: dict[str, Any] = {
            "disk_free_bytes": free,
            "cuda_initialized": False,
            "device_policy": policy,
            "pytorch_compatibility_verified": False,
        }
        if policy == "invalid":
            return (
                ProductCapability(
                    capability_id="compute.local",
                    title=self.title,
                    status=ProductStatus.UNAVAILABLE,
                    message="The local training device policy is invalid.",
                    details={**details, "cuda_host_probe": "NOT_RUN", "reason": "device_policy_invalid"},
                ),
            )
        if policy == "cuda":
            probe = _probe_nvidia_host()
            details.update(
                {
                    "cuda_host_probe": "PASS" if probe["ready"] else "FAIL",
                    "reason": probe["reason"],
                    "usable_nvidia_device_count": int(probe.get("usable_device_count", 0)),
                }
            )
            if not probe["ready"]:
                return (
                    ProductCapability(
                        capability_id="compute.local",
                        title=self.title,
                        status=ProductStatus.UNAVAILABLE,
                        message=(
                            "Explicit CUDA training is unavailable because the bounded NVIDIA host probe did not "
                            "prove a usable device."
                        ),
                        details=details,
                    ),
                )
            return (
                ProductCapability(
                    capability_id="compute.local",
                    title=self.title,
                    status=ProductStatus.READY,
                    message=(
                        "A usable NVIDIA device is visible to the host probe; exact PyTorch/CUDA compatibility "
                        "still requires the bound audit and smoke evidence."
                    ),
                    details=details,
                ),
            )
        details.update(
            {
                "cuda_host_probe": "NOT_REQUIRED",
                "reason": "cpu_forced" if policy == "cpu" else "auto_allows_cpu_fallback",
            }
        )
        return (
            ProductCapability(
                capability_id="compute.local",
                title=self.title,
                status=ProductStatus.READY,
                message=(
                    "Local CPU execution is available and CUDA will be masked."
                    if policy == "cpu"
                    else "Local execution is available; automatic device selection may fall back to CPU."
                ),
                details=details,
            ),
        )

    def estimate(self, context: ProjectContext, campaign: Mapping[str, Any]) -> ComputeEstimate:
        product = campaign.get("product_estimate") if isinstance(campaign.get("product_estimate"), Mapping) else {}
        seconds = product.get("duration_seconds")
        disk = int(product.get("disk_required_bytes", 0) or 0)
        return ComputeEstimate(
            duration_seconds=int(seconds) if isinstance(seconds, (int, float)) and seconds > 0 else None,
            disk_required_bytes=disk,
            trustworthy=isinstance(seconds, (int, float)) and seconds > 0,
            source="campaign product estimate" if seconds else None,
            message="Based on the campaign estimate." if seconds else "Time estimate unavailable for this campaign.",
        )

    def prepare(self, context: ProjectContext, request: ComputeJobRequest) -> PreparedCompute:
        verify_compute_job_request(request, backend_id=self.backend_id)
        operation_id = validate_identifier(request.idempotency_key, label="idempotency key")
        identity = stable_hash(
            {
                "backend": self.backend_id,
                "operation_id": operation_id,
                "campaign_identity": request.campaign_identity,
                "run_identity": request.run_identity,
                "output_root": str(request.output_root.resolve()),
            }
        )
        existing = self._prepared.get(operation_id)
        if existing is not None:
            if existing.remote_identity != identity:
                raise ValueError("Idempotency key is already bound to a different local operation identity.")
            return existing
        prepared = PreparedCompute(self.backend_id, operation_id, str(request.output_root), identity)
        self._prepared[operation_id] = prepared
        return prepared

    def upload(
        self, prepared: PreparedCompute, artifacts: Sequence[Path], *, remote_subdirectory: str = "inputs"
    ) -> OperationResult:
        del remote_subdirectory
        missing = [str(path) for path in artifacts if not path.exists()]
        if missing:
            raise FileNotFoundError("Local input artifact missing: " + ", ".join(missing))
        return OperationResult(False, "Local inputs are already available.", {"artifact_count": len(artifacts)})

    def launch(
        self, prepared: PreparedCompute, request: ComputeJobRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        del cloud_confirmation
        validated = verify_compute_job_request(request, backend_id=self.backend_id)
        existing = self._jobs.get(request.idempotency_key)
        if existing is not None:
            return existing.job
        expected_identity = stable_hash(
            {
                "backend": self.backend_id,
                "operation_id": request.idempotency_key,
                "campaign_identity": request.campaign_identity,
                "run_identity": request.run_identity,
                "output_root": str(request.output_root.resolve()),
            }
        )
        if prepared.backend_id != self.backend_id or prepared.remote_identity != expected_identity:
            raise StaleRemoteIdentityError("Prepared local operation does not match the validated launch request.")
        event_path = request.event_path
        event_name: str | None = None
        if event_path is not None:
            from spritelab.product_web.events import EVENT_FILENAME

            if Path(os.path.abspath(event_path)) != validated.output_root / EVENT_FILENAME:
                raise ValueError("local training event path must be the canonical file in the validated output root")
            event_name = EVENT_FILENAME
        from spritelab.training.launch import TrainingFilesystemCapability

        validated_project_root = verify_local_project_root_binding(request)
        capability = TrainingFilesystemCapability(validated.campaign, validated_project_root)
        capability.__enter__()
        try:
            verified = verify_compute_job_request(
                request,
                backend_id=self.backend_id,
                filesystem_snapshot=capability.filesystem_snapshot,
            )
            child_command = capability.bootstrap_command(verified)
            with capability.launch_inheritance(verified) as (boundary_environment, inheritance_options):
                environment = dict(verified.environment)
                environment.update(boundary_environment)
                process_options = {
                    "cwd": validated_project_root,
                    "env": environment,
                    "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.STDOUT,
                    "shell": False,
                    **inheritance_options,
                }
                if os.name == "nt":
                    process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    process_options["start_new_session"] = True
                verify_launch_authorization_capability(request)
                process = self._process_factory(list(child_command), **process_options)
            job = ComputeJob(
                backend_id=self.backend_id,
                job_id=request.idempotency_key,
                run_id=request.run_id,
                status=ComputeStatus.RUNNING,
                remote_identity=prepared.remote_identity,
                metadata={"pid": getattr(process, "pid", None), "event_filename": event_name},
            )
            self._jobs[request.idempotency_key] = _LocalJobRecord(job, process, capability, event_name)
        except BaseException as exc:
            capability.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return job

    def poll(self, job: ComputeJob) -> ComputePoll:
        record = self._jobs.get(job.job_id)
        if record is None:
            return ComputePoll(
                ComputeStatus.UNCERTAIN, "Local process identity disappeared.", resource_state_uncertain=True
            )
        with record.lock:
            code = record.process.poll()
            if code is None:
                if record.capability is None or record.monitoring_uncertain:
                    return ComputePoll(
                        ComputeStatus.UNCERTAIN,
                        "Local process monitoring capability was released before terminal event capture.",
                        resource_state_uncertain=True,
                    )
                return ComputePoll(ComputeStatus.RUNNING, "Training is running.")
            try:
                self._capture_terminal_event_bytes(record)
            except (OSError, ValueError):
                record.monitoring_uncertain = True
                return ComputePoll(
                    ComputeStatus.UNCERTAIN,
                    "Local process stopped, but terminal event capture could not be verified.",
                    resource_state_uncertain=True,
                    exit_code=int(code),
                )
        if code == 0:
            return ComputePoll(ComputeStatus.COMPLETE, "Training completed.", exit_code=0)
        return ComputePoll(ComputeStatus.FAILED, "Training process failed.", exit_code=int(code))

    def stream_events(self, job: ComputeJob, *, cursor: int = 0) -> tuple[Sequence[ProductEvent], int]:
        record = self._jobs.get(job.job_id)
        if record is None:
            return (), cursor
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Local event cursor must be a non-negative integer.")
        with record.lock:
            if record.monitoring_uncertain:
                raise ValueError("Local event monitoring identity is uncertain.")
            code = record.process.poll()
            if code is None:
                try:
                    content = self._read_retained_event_bytes(record)
                except FileNotFoundError:
                    content = b""
                if record.process.poll() is not None:
                    content = self._capture_terminal_event_bytes(record)
            else:
                content = self._capture_terminal_event_bytes(record)
        lines = content.decode("utf-8", errors="replace").splitlines()
        events: list[ProductEvent] = []
        for line in lines[cursor:]:
            try:
                payload = json.loads(line)
                events.append(ProductEvent.from_dict(payload))
            except (ValueError, json.JSONDecodeError, KeyError):
                events.append(
                    ProductEvent(
                        run_id=job.run_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        feature="training",
                        stage="logs",
                        event_type="log",
                        status=ProductStatus.RUNNING,
                        message=line,
                    )
                )
        return tuple(events), len(lines)

    def pause(self, job: ComputeJob) -> OperationResult:
        record = self._jobs.get(job.job_id)
        if record is None:
            return OperationResult(False, "Local process is no longer known.")
        with record.lock:
            process = record.process
            if process.poll() is not None:
                return OperationResult(False, "Local process already stopped.")
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        return OperationResult(
            True, "Graceful pause requested; safe-resume eligibility still requires checkpoint verification."
        )

    def cancel(self, job: ComputeJob) -> OperationResult:
        record = self._jobs.get(job.job_id)
        if record is None:
            return OperationResult(False, "Local process already stopped.")
        with record.lock:
            if record.process.poll() is not None:
                return OperationResult(False, "Local process already stopped.")
            record.process.terminate()
        return OperationResult(True, "Cancellation requested.")

    def resume(
        self, prepared: PreparedCompute, resume: ResumeRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        if not resume.safe_resume or not resume.checkpoint.hash_verified:
            raise ArtifactVerificationError(
                "Unsafe resume is unavailable; the checkpoint must pass identity and hash checks."
            )
        if resume.checkpoint.remote_identity != prepared.remote_identity:
            raise ArtifactVerificationError("Checkpoint identity does not match the prepared local run.")
        return self.launch(prepared, resume.request, cloud_confirmation=cloud_confirmation)

    def download_artifacts(
        self, job: ComputeJob, artifacts: Sequence[ArtifactReference], destination: Path
    ) -> Sequence[ArtifactReference]:
        destination.mkdir(parents=True, exist_ok=True)
        results = []
        for artifact in artifacts:
            source = Path(artifact.relative_path)
            if not source.is_file():
                raise FileNotFoundError(source)
            target = destination / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            actual = file_sha256(target)
            if actual != artifact.sha256:
                raise ArtifactVerificationError(f"Downloaded artifact hash mismatch: {source.name}")
            results.append(
                replace(
                    artifact,
                    local_path=target,
                    downloaded=True,
                    hash_verified=True,
                    remote_identity_verified=artifact.remote_identity == job.remote_identity,
                )
            )
        return tuple(results)

    def verify_artifacts(self, job: ComputeJob, artifacts: Sequence[ArtifactReference]) -> Sequence[ArtifactReference]:
        verified = []
        for artifact in artifacts:
            path = artifact.local_path or Path(artifact.relative_path)
            matches = path.is_file() and file_sha256(path) == artifact.sha256
            if not matches:
                raise ArtifactVerificationError(f"Artifact hash mismatch: {path}")
            verified.append(
                replace(
                    artifact,
                    hash_verified=True,
                    downloaded=True,
                    remote_identity_verified=artifact.remote_identity == job.remote_identity,
                )
            )
        return tuple(verified)

    def cleanup(self, prepared: PreparedCompute) -> OperationResult:
        removed = self._prepared.pop(prepared.operation_id, None) is not None
        record = self._jobs.get(prepared.operation_id)
        released = False
        if record is not None:
            with record.lock:
                if record.capability is not None:
                    if record.process.poll() is not None:
                        try:
                            self._capture_terminal_event_bytes(record)
                        except (OSError, ValueError):
                            record.monitoring_uncertain = True
                            self._release_capability(record)
                            record.released_by_cleanup = True
                    else:
                        try:
                            self._read_retained_event_bytes(record)
                        except FileNotFoundError:
                            pass
                        except (OSError, ValueError):
                            record.monitoring_uncertain = True
                        finally:
                            self._release_capability(record)
                            record.released_by_cleanup = True
                    released = True
        return OperationResult(
            removed or released,
            "Released local adapter bookkeeping and monitoring handles; run artifacts were preserved.",
        )

    @staticmethod
    def _release_capability(record: _LocalJobRecord) -> None:
        capability, record.capability = record.capability, None
        if capability is not None:
            capability.__exit__(None, None, None)

    @staticmethod
    def _read_retained_event_bytes(record: _LocalJobRecord) -> bytes:
        if record.event_name is None:
            record.cached_event_bytes = b""
            return b""
        if record.capability is None:
            if record.terminal_events_captured or record.released_by_cleanup:
                return record.cached_event_bytes
            raise ValueError("Local event capability is unavailable before terminal capture.")
        content = record.capability.read_run_regular_bytes(
            record.job.run_id,
            record.event_name,
            _MAX_LOCAL_EVENT_STREAM_BYTES,
        )
        record.cached_event_bytes = content
        return content

    @classmethod
    def _capture_terminal_event_bytes(cls, record: _LocalJobRecord) -> bytes:
        if record.terminal_events_captured:
            return record.cached_event_bytes
        if record.released_by_cleanup:
            raise ValueError("Local monitoring was cleaned up before terminal event capture.")
        try:
            content = cls._read_retained_event_bytes(record)
        except FileNotFoundError:
            content = b""
            record.cached_event_bytes = content
        cls._release_capability(record)
        record.terminal_events_captured = True
        record.monitoring_uncertain = False
        return content
