"""Remote-compute public surface."""

from spritelab.remote_compute.contracts import (
    ArtifactReference,
    ArtifactVerificationError,
    CapabilityUnavailableError,
    CloudConfirmationRequired,
    ComputeBackend,
    ComputeBackendError,
    ComputeEstimate,
    ComputeJob,
    ComputeJobRequest,
    ComputePoll,
    ComputeStatus,
    OperationResult,
    PreparedCompute,
    ResumeRequest,
    StaleRemoteIdentityError,
    TrainingLaunchRejected,
    verify_compute_job_request,
)
from spritelab.remote_compute.fake import FakeComputeBackend
from spritelab.remote_compute.hosted import HostedBackendRegistry, ReceiptEnforcingComputeBackend, select_hosted_backend
from spritelab.remote_compute.local import LocalComputeBackend
from spritelab.remote_compute.runpod import RunPodComputeBackend, RunPodSettings
from spritelab.remote_compute.ssh import SSHComputeBackend, SSHSettings, SSHTransport, SubprocessSSHTransport

__all__ = [
    "ArtifactReference",
    "ArtifactVerificationError",
    "CapabilityUnavailableError",
    "CloudConfirmationRequired",
    "ComputeBackend",
    "ComputeBackendError",
    "ComputeEstimate",
    "ComputeJob",
    "ComputeJobRequest",
    "ComputePoll",
    "ComputeStatus",
    "FakeComputeBackend",
    "HostedBackendRegistry",
    "LocalComputeBackend",
    "OperationResult",
    "PreparedCompute",
    "ReceiptEnforcingComputeBackend",
    "ResumeRequest",
    "RunPodComputeBackend",
    "RunPodSettings",
    "SSHComputeBackend",
    "SSHSettings",
    "SSHTransport",
    "StaleRemoteIdentityError",
    "SubprocessSSHTransport",
    "TrainingLaunchRejected",
    "select_hosted_backend",
    "verify_compute_job_request",
]
