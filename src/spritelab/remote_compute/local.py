"""Idempotent local-process compute backend."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
)
from spritelab.remote_compute.utils import file_sha256, stable_hash, validate_identifier

ProcessFactory = Callable[..., Any]


class LocalComputeBackend:
    backend_id = "local"
    title = "Local computer"
    is_cloud = False

    def __init__(self, *, process_factory: ProcessFactory | None = None) -> None:
        self._process_factory = process_factory or subprocess.Popen
        self._prepared: dict[str, PreparedCompute] = {}
        self._jobs: dict[str, tuple[ComputeJob, Any, Path | None]] = {}

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]:
        free = shutil.disk_usage(context.project_root).free
        return (
            ProductCapability(
                capability_id="compute.local",
                title=self.title,
                status=ProductStatus.READY,
                message="Local process execution is available; no CUDA runtime was initialized.",
                details={"disk_free_bytes": free, "cuda_initialized": False},
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
        verify_compute_job_request(request, backend_id=self.backend_id)
        existing = self._jobs.get(request.idempotency_key)
        if existing is not None:
            return existing[0]
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
        if event_path is not None:
            event_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(request.environment)
        process_options = {
            "cwd": request.local_project_root,
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.STDOUT,
            "shell": False,
        }
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        verify_compute_job_request(request, backend_id=self.backend_id)
        process = self._process_factory(list(request.command), **process_options)
        job = ComputeJob(
            backend_id=self.backend_id,
            job_id=request.idempotency_key,
            run_id=request.run_id,
            status=ComputeStatus.RUNNING,
            remote_identity=prepared.remote_identity,
            metadata={"pid": getattr(process, "pid", None), "event_path": str(event_path) if event_path else None},
        )
        self._jobs[request.idempotency_key] = (job, process, event_path)
        return job

    def poll(self, job: ComputeJob) -> ComputePoll:
        record = self._jobs.get(job.job_id)
        if record is None:
            return ComputePoll(
                ComputeStatus.UNCERTAIN, "Local process identity disappeared.", resource_state_uncertain=True
            )
        code = record[1].poll()
        if code is None:
            return ComputePoll(ComputeStatus.RUNNING, "Training is running.")
        if code == 0:
            return ComputePoll(ComputeStatus.COMPLETE, "Training completed.", exit_code=0)
        return ComputePoll(ComputeStatus.FAILED, "Training process failed.", exit_code=int(code))

    def stream_events(self, job: ComputeJob, *, cursor: int = 0) -> tuple[Sequence[ProductEvent], int]:
        record = self._jobs.get(job.job_id)
        if record is None or record[2] is None or not record[2].is_file():
            return (), cursor
        lines = record[2].read_text(encoding="utf-8", errors="replace").splitlines()
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
        process = record[1]
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
        if record is None or record[1].poll() is not None:
            return OperationResult(False, "Local process already stopped.")
        record[1].terminate()
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
        return OperationResult(removed, "Released local adapter bookkeeping; run artifacts were preserved.")
