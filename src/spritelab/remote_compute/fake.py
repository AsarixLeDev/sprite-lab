"""Deterministic no-process backend used by product and adapter tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from spritelab.product_core import ProductCapability, ProductEvent, ProductStatus, ProjectContext
from spritelab.remote_compute.contracts import (
    ArtifactReference,
    ArtifactVerificationError,
    CloudConfirmationRequired,
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
from spritelab.remote_compute.utils import stable_hash


class FakeComputeBackend:
    backend_id = "fake"
    title = "Deterministic fake compute"

    def __init__(self, events: Sequence[ProductEvent] = (), *, is_cloud: bool = False) -> None:
        self.is_cloud = is_cloud
        self.events = tuple(events)
        self.calls: list[str] = []
        self._jobs: dict[str, ComputeJob] = {}

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]:
        self.calls.append("probe")
        return (ProductCapability("compute.fake", self.title, ProductStatus.READY, "Deterministic test backend."),)

    def estimate(self, context: ProjectContext, campaign: Mapping[str, Any]) -> ComputeEstimate:
        self.calls.append("estimate")
        return ComputeEstimate(60, 1024, trustworthy=True, source="deterministic fake", message="One minute.")

    def prepare(self, context: ProjectContext, request: ComputeJobRequest) -> PreparedCompute:
        verify_compute_job_request(request, backend_id=self.backend_id)
        self.calls.append("prepare")
        identity = stable_hash({"campaign": request.campaign_identity, "run": request.run_identity})
        return PreparedCompute(self.backend_id, request.idempotency_key, "/fake", identity)

    def upload(
        self, prepared: PreparedCompute, artifacts: Sequence[Path], *, remote_subdirectory: str = "inputs"
    ) -> OperationResult:
        self.calls.append("upload")
        return OperationResult(True, "Fake upload complete.", {"count": len(artifacts), "to": remote_subdirectory})

    def launch(
        self, prepared: PreparedCompute, request: ComputeJobRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        if self.is_cloud and not cloud_confirmation:
            raise CloudConfirmationRequired("Explicit cloud confirmation is required.")
        verify_compute_job_request(request, backend_id=self.backend_id)
        existing = self._jobs.get(request.idempotency_key)
        if existing:
            return existing
        expected_identity = stable_hash({"campaign": request.campaign_identity, "run": request.run_identity})
        if prepared.backend_id != self.backend_id or prepared.remote_identity != expected_identity:
            raise StaleRemoteIdentityError("Prepared fake operation does not match the validated launch request.")
        self.calls.append("launch")
        job = ComputeJob(
            self.backend_id,
            request.idempotency_key,
            request.run_id,
            ComputeStatus.RUNNING,
            prepared.remote_identity,
            may_accrue_cost=self.is_cloud,
        )
        self._jobs[job.job_id] = job
        return job

    def poll(self, job: ComputeJob) -> ComputePoll:
        self.calls.append("poll")
        current = self._jobs.get(job.job_id, job)
        return ComputePoll(current.status, "Deterministic fake state.", may_accrue_cost=current.may_accrue_cost)

    def stream_events(self, job: ComputeJob, *, cursor: int = 0) -> tuple[Sequence[ProductEvent], int]:
        self.calls.append("stream_events")
        return self.events[cursor:], len(self.events)

    def pause(self, job: ComputeJob) -> OperationResult:
        self.calls.append("pause")
        self._jobs[job.job_id] = replace(job, status=ComputeStatus.PAUSED)
        return OperationResult(True, "Fake graceful pause completed.")

    def cancel(self, job: ComputeJob) -> OperationResult:
        self.calls.append("cancel")
        self._jobs[job.job_id] = replace(job, status=ComputeStatus.CANCELLED)
        return OperationResult(True, "Fake job cancelled.")

    def resume(
        self, prepared: PreparedCompute, resume: ResumeRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        self.calls.append("resume")
        if not resume.safe_resume or not (
            resume.checkpoint.downloaded
            and resume.checkpoint.hash_verified
            and resume.checkpoint.remote_identity_verified
        ):
            raise ArtifactVerificationError("Unsafe resume is unavailable.")
        return self.launch(prepared, resume.request, cloud_confirmation=cloud_confirmation)

    def download_artifacts(
        self, job: ComputeJob, artifacts: Sequence[ArtifactReference], destination: Path
    ) -> Sequence[ArtifactReference]:
        self.calls.append("download_artifacts")
        return tuple(
            replace(
                item,
                local_path=destination / Path(item.relative_path).name,
                downloaded=True,
                hash_verified=True,
                remote_identity_verified=item.remote_identity == job.remote_identity,
            )
            for item in artifacts
        )

    def verify_artifacts(self, job: ComputeJob, artifacts: Sequence[ArtifactReference]) -> Sequence[ArtifactReference]:
        self.calls.append("verify_artifacts")
        return tuple(
            replace(item, hash_verified=True, remote_identity_verified=item.remote_identity == job.remote_identity)
            for item in artifacts
        )

    def cleanup(self, prepared: PreparedCompute) -> OperationResult:
        self.calls.append("cleanup")
        return OperationResult(True, "Fake cleanup completed.")
