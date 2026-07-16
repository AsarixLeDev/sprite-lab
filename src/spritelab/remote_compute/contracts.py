"""Provider-neutral, CPU-only compute contracts for product training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spritelab.product_core import ProductCapability, ProductEvent, ProjectContext
from spritelab.training.campaign import CampaignValidationError, is_concrete_hash
from spritelab.training.launch import (
    TrainingLaunchContext,
    TrainingLaunchReceipt,
    ValidatedTrainingLaunch,
    verify_validated_training_launch,
)


class ComputeStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    READY = "READY"
    PREPARED = "PREPARED"
    UPLOADING = "UPLOADING"
    RUNNING = "RUNNING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"
    UNCERTAIN = "UNCERTAIN"


class ComputeBackendError(RuntimeError):
    """Base error raised without pretending a remote operation succeeded."""


class CapabilityUnavailableError(ComputeBackendError):
    """The adapter deliberately does not implement a required capability."""


class CloudConfirmationRequired(ComputeBackendError):
    """A cloud launch was attempted without a fresh explicit confirmation."""


class ArtifactVerificationError(ComputeBackendError):
    """An artifact did not match its declared identity."""


class StaleRemoteIdentityError(ComputeBackendError):
    """Remote state belongs to a different prepared operation or campaign."""


class TrainingLaunchRejected(ComputeBackendError):
    """A process or remote seam refused an absent, forged, or stale receipt."""


@dataclass(frozen=True)
class ComputeEstimate:
    duration_seconds: int | None = None
    disk_required_bytes: int = 0
    hourly_cost: float | None = None
    estimated_cost: float | None = None
    currency: str | None = None
    trustworthy: bool = False
    source: str | None = None
    message: str = "Estimate unavailable."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComputeJobRequest:
    run_id: str
    command: tuple[str, ...]
    idempotency_key: str
    campaign_identity: str
    run_identity: str
    local_project_root: Path
    output_root: Path
    event_path: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    execution_spec_identity: str = ""
    output_root_identity: str = ""
    compute_backend_id: str = ""
    launch_receipt: TrainingLaunchReceipt | None = None
    validator_context: TrainingLaunchContext | None = None

    def __post_init__(self) -> None:
        if not self.command or any("\x00" in item for item in self.command):
            raise ValueError("Compute commands must be non-empty argument arrays without NUL bytes.")
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key is required.")


def verify_compute_job_request(request: ComputeJobRequest, *, backend_id: str) -> ValidatedTrainingLaunch:
    """Revalidate a request at an adapter's lowest process/transport boundary."""

    if request.launch_receipt is None or request.validator_context is None:
        raise TrainingLaunchRejected("validated training launch receipt and validator context are required")
    receipt = request.launch_receipt
    protected = {
        "campaign_identity": request.campaign_identity,
        "run_identity": request.run_identity,
        "execution_spec_identity": request.execution_spec_identity,
        "output_root_identity": request.output_root_identity,
    }
    malformed = [name for name, value in protected.items() if not is_concrete_hash(value)]
    if malformed:
        raise TrainingLaunchRejected(
            "compute request contains malformed or placeholder protected identities: " + ", ".join(malformed)
        )
    if request.compute_backend_id != backend_id or receipt.compute_backend_id != backend_id:
        raise TrainingLaunchRejected("compute request backend does not match its launch receipt")
    if request.execution_spec_identity != receipt.execution_spec_sha256:
        raise TrainingLaunchRejected("compute request execution specification does not match its receipt")
    if request.output_root_identity != receipt.output_root_identity:
        raise TrainingLaunchRejected("compute request output-root identity does not match its receipt")
    try:
        return verify_validated_training_launch(
            receipt,
            request.validator_context,
            compute_backend_id=backend_id,
            argv=request.command,
            environment=request.environment,
            output_root=request.output_root,
            campaign_identity=request.campaign_identity,
            run_identity=request.run_identity,
        )
    except (CampaignValidationError, OSError, ValueError) as exc:
        raise TrainingLaunchRejected(str(exc)) from exc


@dataclass(frozen=True)
class PreparedCompute:
    backend_id: str
    operation_id: str
    workspace: str
    remote_identity: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeJob:
    backend_id: str
    job_id: str
    run_id: str
    status: ComputeStatus
    remote_identity: str
    may_accrue_cost: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputePoll:
    status: ComputeStatus
    message: str
    may_accrue_cost: bool = False
    resource_state_uncertain: bool = False
    exit_code: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactReference:
    relative_path: str
    sha256: str
    remote_identity: str
    local_path: Path | None = None
    downloaded: bool = False
    hash_verified: bool = False
    remote_identity_verified: bool = False

    @property
    def safe_for_remote_resume(self) -> bool:
        return self.downloaded and self.hash_verified and self.remote_identity_verified


@dataclass(frozen=True)
class ResumeRequest:
    request: ComputeJobRequest
    checkpoint: ArtifactReference
    safe_resume: bool


@dataclass(frozen=True)
class OperationResult:
    changed: bool
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ComputeBackend(Protocol):
    """Full lifecycle required of local and hosted training adapters."""

    backend_id: str
    title: str
    is_cloud: bool

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]: ...

    def estimate(self, context: ProjectContext, campaign: Mapping[str, Any]) -> ComputeEstimate: ...

    def prepare(self, context: ProjectContext, request: ComputeJobRequest) -> PreparedCompute: ...

    def upload(
        self, prepared: PreparedCompute, artifacts: Sequence[Path], *, remote_subdirectory: str = "inputs"
    ) -> OperationResult: ...

    def launch(
        self, prepared: PreparedCompute, request: ComputeJobRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob: ...

    def poll(self, job: ComputeJob) -> ComputePoll: ...

    def stream_events(self, job: ComputeJob, *, cursor: int = 0) -> tuple[Sequence[ProductEvent], int]: ...

    def pause(self, job: ComputeJob) -> OperationResult: ...

    def cancel(self, job: ComputeJob) -> OperationResult: ...

    def resume(
        self, prepared: PreparedCompute, resume: ResumeRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob: ...

    def download_artifacts(
        self, job: ComputeJob, artifacts: Sequence[ArtifactReference], destination: Path
    ) -> Sequence[ArtifactReference]: ...

    def verify_artifacts(
        self, job: ComputeJob, artifacts: Sequence[ArtifactReference]
    ) -> Sequence[ArtifactReference]: ...

    def cleanup(self, prepared: PreparedCompute) -> OperationResult: ...
