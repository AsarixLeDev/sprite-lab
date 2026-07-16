"""Validated RunPod adapter scaffold that never claims hosted execution works."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.product_core import ProductCapability, ProductEvent, ProductStatus, ProjectContext
from spritelab.remote_compute.contracts import (
    ArtifactReference,
    CapabilityUnavailableError,
    ComputeEstimate,
    ComputeJob,
    ComputeJobRequest,
    ComputePoll,
    OperationResult,
    PreparedCompute,
    ResumeRequest,
    verify_compute_job_request,
)

RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"
RUNPOD_CREATE_POD_DOC = "https://docs.runpod.io/api-reference/pods/POST/pods"
RUNPOD_LIST_PODS_DOC = "https://docs.runpod.io/api-reference/pods/GET/pods"
RUNPOD_DELETE_POD_DOC = "https://docs.runpod.io/api-reference/pods/DELETE/pods/podId"
RUNPOD_SSH_DOC = "https://docs.runpod.io/pods/configuration/use-ssh"

MISSING_IMPLEMENTATION_NOTES = (
    "A reviewed immutable training image/template and its digest are not defined by this repository.",
    "Pod SSH readiness and host-key verification are not bound to a campaign identity.",
    "RunPod Pod identity is not yet bound to the backend artifact/checkpoint identity contract.",
    "A provider quote/availability response is not integrated, so no volatile price is displayed.",
    "Stop/delete reconciliation after connection loss is not implemented and cannot safely report resource shutdown.",
)

_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9./:_@-]{2,254}$")


@dataclass(frozen=True)
class RunPodSettings:
    api_key_env: str = "RUNPOD_API_KEY"
    image_name: str = ""
    gpu_type_ids: tuple[str, ...] = ()
    gpu_count: int = 1
    container_disk_gb: int = 50
    volume_gb: int = 50
    shutdown_policy: str = "terminate_after_artifact_verification"
    cloud_type: str = "SECURE"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RunPodSettings:
        forbidden = sorted(
            key
            for key in value
            if any(mark in str(key).lower() for mark in ("api_key", "token", "secret", "password"))
            and key != "api_key_env"
        )
        if forbidden:
            raise ValueError("RunPod credentials must not be stored in project configuration: " + ", ".join(forbidden))
        gpu_types = value.get("gpu_type_ids") or (() if value.get("gpu_type_id") is None else (value["gpu_type_id"],))
        settings = cls(
            api_key_env=str(value.get("api_key_env") or "RUNPOD_API_KEY"),
            image_name=str(value.get("image_name") or ""),
            gpu_type_ids=tuple(str(item) for item in gpu_types),
            gpu_count=int(value.get("gpu_count", 1)),
            container_disk_gb=int(value.get("container_disk_gb", 50)),
            volume_gb=int(value.get("volume_gb", 50)),
            shutdown_policy=str(value.get("shutdown_policy") or "terminate_after_artifact_verification"),
            cloud_type=str(value.get("cloud_type") or "SECURE").upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not _ENV_RE.fullmatch(self.api_key_env):
            raise ValueError("RunPod api_key_env must name an environment variable, not contain a credential.")
        if not self.image_name or not _IMAGE_RE.fullmatch(self.image_name):
            raise ValueError("RunPod requires an explicit valid image_name.")
        if self.gpu_count < 1:
            raise ValueError("RunPod gpu_count must be positive.")
        if not self.gpu_type_ids or any(not item.strip() for item in self.gpu_type_ids):
            raise ValueError("RunPod requires at least one explicit GPU type selection.")
        if self.container_disk_gb < 20 or self.volume_gb < 20:
            raise ValueError("RunPod container and volume disk requirements must each be at least 20 GB.")
        if self.shutdown_policy not in {"terminate_after_artifact_verification", "manual"}:
            raise ValueError("RunPod shutdown_policy must be 'terminate_after_artifact_verification' or 'manual'.")
        if self.cloud_type not in {"SECURE", "COMMUNITY"}:
            raise ValueError("RunPod cloud_type must be SECURE or COMMUNITY.")


class RunPodComputeBackend:
    """Complete lifecycle-shaped scaffold; mutating methods intentionally fail closed."""

    backend_id = "runpod"
    title = "RunPod"
    is_cloud = True

    def __init__(self, settings: RunPodSettings) -> None:
        settings.validate()
        self.settings = settings

    @property
    def credential_configured(self) -> bool:
        return bool(os.environ.get(self.settings.api_key_env))

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]:
        del context
        return (
            ProductCapability(
                "compute.runpod",
                self.title,
                ProductStatus.UNAVAILABLE,
                "RunPod configuration is valid, but safe end-to-end training is not implemented.",
                {
                    "credential_status": "configured" if self.credential_configured else "missing",
                    "gpu_selection": list(self.settings.gpu_type_ids),
                    "disk_requirement_gb": self.settings.container_disk_gb + self.settings.volume_gb,
                    "shutdown_policy": self.settings.shutdown_policy,
                    "missing_implementation": list(MISSING_IMPLEMENTATION_NOTES),
                    "provider_calls": 0,
                },
            ),
        )

    def estimate(self, context: ProjectContext, campaign: Mapping[str, Any]) -> ComputeEstimate:
        del context, campaign
        return ComputeEstimate(
            disk_required_bytes=(self.settings.container_disk_gb + self.settings.volume_gb) * 1024**3,
            trustworthy=False,
            source=None,
            message="RunPod price is not shown because no current provider quote was fetched.",
        )

    def _unavailable(self) -> CapabilityUnavailableError:
        return CapabilityUnavailableError("RunPod launch is unavailable: " + " ".join(MISSING_IMPLEMENTATION_NOTES))

    def prepare(self, context: ProjectContext, request: ComputeJobRequest) -> PreparedCompute:
        verify_compute_job_request(request, backend_id=self.backend_id)
        del context
        raise self._unavailable()

    def upload(
        self, prepared: PreparedCompute, artifacts: Sequence[Path], *, remote_subdirectory: str = "inputs"
    ) -> OperationResult:
        del prepared, artifacts, remote_subdirectory
        raise self._unavailable()

    def launch(
        self, prepared: PreparedCompute, request: ComputeJobRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        verify_compute_job_request(request, backend_id=self.backend_id)
        del prepared, cloud_confirmation
        raise self._unavailable()

    def poll(self, job: ComputeJob) -> ComputePoll:
        del job
        raise self._unavailable()

    def stream_events(self, job: ComputeJob, *, cursor: int = 0) -> tuple[Sequence[ProductEvent], int]:
        del job, cursor
        raise self._unavailable()

    def pause(self, job: ComputeJob) -> OperationResult:
        del job
        raise self._unavailable()

    def cancel(self, job: ComputeJob) -> OperationResult:
        del job
        raise self._unavailable()

    def resume(
        self, prepared: PreparedCompute, resume: ResumeRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        verify_compute_job_request(resume.request, backend_id=self.backend_id)
        del prepared, cloud_confirmation
        raise self._unavailable()

    def download_artifacts(
        self, job: ComputeJob, artifacts: Sequence[ArtifactReference], destination: Path
    ) -> Sequence[ArtifactReference]:
        del job, artifacts, destination
        raise self._unavailable()

    def verify_artifacts(self, job: ComputeJob, artifacts: Sequence[ArtifactReference]) -> Sequence[ArtifactReference]:
        del job, artifacts
        raise self._unavailable()

    def cleanup(self, prepared: PreparedCompute) -> OperationResult:
        del prepared
        raise self._unavailable()
