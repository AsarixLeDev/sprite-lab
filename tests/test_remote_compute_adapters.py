from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spritelab.product_core import ProductEvent, ProductStatus, ProjectContext
from spritelab.remote_compute import (
    ArtifactReference,
    ArtifactVerificationError,
    CapabilityUnavailableError,
    ComputeBackend,
    ComputeJobRequest,
    ComputeStatus,
    FakeComputeBackend,
    HostedBackendRegistry,
    LocalComputeBackend,
    RunPodComputeBackend,
    RunPodSettings,
    TrainingLaunchRejected,
)
from spritelab.remote_compute.utils import redact
from training_launch_test_utils import compute_request


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext(tmp_path, {})


def _request(tmp_path: Path, backend_id: str = "fake") -> ComputeJobRequest:
    return compute_request(tmp_path, backend_id)


def test_fake_backend_satisfies_full_compute_protocol_and_streams_events(tmp_path: Path) -> None:
    event = ProductEvent(
        "seed-1",
        datetime.now(timezone.utc).isoformat(),
        "training",
        "seed",
        "progress",
        ProductStatus.RUNNING,
        metrics={"seed": 1, "loss": 0.5},
    )
    backend = FakeComputeBackend([event])
    assert isinstance(backend, ComputeBackend)
    request = _request(tmp_path)
    prepared = backend.prepare(_context(tmp_path), request)
    job = backend.launch(prepared, request)
    events, cursor = backend.stream_events(job)
    assert events == (event,) and cursor == 1


def test_fake_cloud_confirmation_and_graceful_pause(tmp_path: Path) -> None:
    backend = FakeComputeBackend(is_cloud=True)
    request = _request(tmp_path)
    prepared = backend.prepare(_context(tmp_path), request)
    with pytest.raises(Exception, match="confirmation"):
        backend.launch(prepared, request)
    job = backend.launch(prepared, request, cloud_confirmation=True)
    assert backend.pause(job).changed
    assert backend.poll(backend._jobs[job.job_id]).status == ComputeStatus.PAUSED


def test_fake_resume_refuses_unverified_checkpoint(tmp_path: Path) -> None:
    from spritelab.remote_compute import ResumeRequest

    backend = FakeComputeBackend()
    request = _request(tmp_path)
    prepared = backend.prepare(_context(tmp_path), request)
    checkpoint = ArtifactReference("checkpoint.pt", "a" * 64, prepared.remote_identity)
    with pytest.raises(ArtifactVerificationError, match="Unsafe resume"):
        backend.resume(prepared, ResumeRequest(request, checkpoint, safe_resume=True))


def test_fake_safe_resume_accepts_only_fully_verified_remote_checkpoint(tmp_path: Path) -> None:
    from spritelab.remote_compute import ResumeRequest

    backend = FakeComputeBackend()
    first = _request(tmp_path)
    prepared = backend.prepare(_context(tmp_path), first)
    resumed_request = replace(first, idempotency_key="seed-1-resume")
    checkpoint = ArtifactReference(
        "checkpoint.pt",
        "a" * 64,
        prepared.remote_identity,
        tmp_path / "checkpoint.pt",
        downloaded=True,
        hash_verified=True,
        remote_identity_verified=True,
    )
    job = backend.resume(prepared, ResumeRequest(resumed_request, checkpoint, safe_resume=True))
    assert job.status == ComputeStatus.RUNNING


def test_local_backend_launch_uses_argument_array_and_no_shell(tmp_path: Path) -> None:
    captured = {}

    class Process:
        pid = 123

        def poll(self):
            return None

    def factory(command, **kwargs):
        captured.update({"command": command, **kwargs})
        return Process()

    backend = LocalComputeBackend(process_factory=factory)
    request = _request(tmp_path, "local")
    prepared = backend.prepare(_context(tmp_path), request)
    try:
        backend.launch(prepared, request)
        command = captured["command"]
        assert command[0] == request.command[0]
        assert command[1:3] == ["-I", "-c"]
        assert command[4:] == list(request.command[1:])
        assert captured["shell"] is False
        assert captured["cwd"] == request.validator_context.project_root.resolve(strict=True)
        assert set(captured["env"]) == set(request.environment) | {
            "SPRITELAB_VALIDATED_TRAINING_BOUNDARY",
            "SPRITELAB_VALIDATED_TRAINING_CODE_BUNDLE",
        }
    finally:
        backend.cleanup(prepared)


@pytest.mark.parametrize("hostile_root_kind", ["ancestor", "outside"])
def test_local_backend_rejects_unbound_project_root_before_filesystem_mutation(
    tmp_path: Path,
    hostile_root_kind: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    request = _request(project, "local")
    backend_calls: list[tuple[object, dict[str, object]]] = []

    def factory(command: object, **kwargs: object) -> object:
        backend_calls.append((command, kwargs))
        raise AssertionError("an unbound project root must fail before process construction")

    backend = LocalComputeBackend(process_factory=factory)
    prepared = backend.prepare(_context(project), request)
    hostile_root = tmp_path if hostile_root_kind == "ancestor" else tmp_path / "outside"
    hostile_root.mkdir(exist_ok=True)
    sentinel = hostile_root / f"{hostile_root_kind}.sentinel"
    sentinel.write_bytes(b"preserve-outside")
    unauthorized_bundle_root = hostile_root / ".spritelab" / "training_code_bundles"
    output_root = request.output_root
    assert not unauthorized_bundle_root.exists()
    assert not output_root.exists()

    hostile = replace(request, local_project_root=hostile_root)
    with pytest.raises(TrainingLaunchRejected, match="local project root"):
        backend.launch(prepared, hostile)

    assert sentinel.read_bytes() == b"preserve-outside"
    assert not unauthorized_bundle_root.exists()
    assert not output_root.exists()
    assert backend_calls == []


def test_hosted_registry_accepts_plugin_backend_and_rejects_non_cloud() -> None:
    hosted = FakeComputeBackend(is_cloud=True)
    registry = HostedBackendRegistry([hosted])
    assert registry.get("fake") is not hosted
    assert registry.get("fake").backend_id == hosted.backend_id
    with pytest.raises(TypeError, match="is_cloud"):
        HostedBackendRegistry([FakeComputeBackend(is_cloud=False)])


def test_runpod_configuration_and_unavailable_scaffold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    settings = RunPodSettings.from_mapping(
        {
            "gpu_type_ids": ["NVIDIA RTX A6000"],
            "image_name": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "container_disk_gb": 50,
            "volume_gb": 50,
            "shutdown_policy": "terminate_after_artifact_verification",
        }
    )
    backend = RunPodComputeBackend(settings)
    capability = backend.probe(_context(tmp_path))[0]
    assert capability.status == ProductStatus.UNAVAILABLE
    assert capability.details["credential_status"] == "missing"
    assert capability.details["provider_calls"] == 0
    assert backend.estimate(_context(tmp_path), {}).estimated_cost is None
    with pytest.raises(CapabilityUnavailableError, match="immutable training image"):
        backend.prepare(_context(tmp_path), _request(tmp_path, "runpod"))


def test_runpod_and_ssh_secrets_are_not_accepted_or_exposed() -> None:
    with pytest.raises(ValueError, match="credentials"):
        RunPodSettings.from_mapping({"gpu_type_ids": ["GPU"], "api_key": "secret"})
    value = redact({"api_token": "abc", "nested": {"password": "def"}, "host": "safe"})
    assert value == {"api_token": "<redacted>", "nested": {"password": "<redacted>"}, "host": "safe"}


def test_artifact_reference_requires_all_remote_checks_for_safe_resume() -> None:
    base = ArtifactReference("checkpoint.pt", "a" * 64, "remote")
    assert not base.safe_for_remote_resume
    verified = ArtifactReference(
        "checkpoint.pt",
        "a" * 64,
        "remote",
        Path("checkpoint.pt"),
        downloaded=True,
        hash_verified=True,
        remote_identity_verified=True,
    )
    assert verified.safe_for_remote_resume


def test_no_cuda_module_is_initialized_by_fake_or_runpod_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    before = set(sys.modules)
    FakeComputeBackend().probe(_context(tmp_path))
    backend = RunPodComputeBackend(RunPodSettings(gpu_type_ids=("GPU",), image_name="image:tag"))
    backend.probe(_context(tmp_path))
    assert "torch.cuda" not in set(sys.modules) - before
