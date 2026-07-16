from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spritelab.product_core import ProjectContext
from spritelab.remote_compute import (
    ArtifactReference,
    ArtifactVerificationError,
    ComputeJob,
    ComputeStatus,
    SSHComputeBackend,
    SSHSettings,
    SubprocessSSHTransport,
)
from spritelab.remote_compute.ssh import _POLL_SCRIPT, RemoteResult


class FakeTransport:
    def __init__(self) -> None:
        self.polls = []
        self.upload_result = RemoteResult(0)
        self.download_bytes = b""

    def execute(self, script, payload):
        if script == _POLL_SCRIPT and self.polls:
            return self.polls.pop(0)
        if "disk_free_bytes" in script:
            return RemoteResult(0, json.dumps({"python": "3.11", "disk_free_bytes": 10_000}))
        if "sha256" in script and payload.get("path"):
            return RemoteResult(0, json.dumps({"sha256": payload.get("expected", "a" * 64)}))
        return RemoteResult(
            0,
            json.dumps(
                {
                    "operation_id": payload.get("operation_id"),
                    "remote_identity": payload.get("remote_identity"),
                    "status": "RUNNING",
                    "changed": True,
                    "rows": [],
                    "cursor": payload.get("cursor", 0),
                }
            ),
        )

    def upload(self, local_path, remote_path):
        return self.upload_result

    def download(self, remote_path, local_path):
        local_path.write_bytes(self.download_bytes)
        return RemoteResult(0)


def _backend(transport=None) -> SSHComputeBackend:
    return SSHComputeBackend(
        SSHSettings("example.test", "trainer", "/workspace/sprite-lab", cloud=True),
        transport=transport or FakeTransport(),
    )


def _job() -> ComputeJob:
    return ComputeJob(
        "ssh",
        "job-1",
        "run-1",
        ComputeStatus.RUNNING,
        "remote-id",
        may_accrue_cost=True,
        metadata={
            "state_path": "/workspace/sprite-lab/.spritelab/jobs/job-1.json",
            "log_path": "/workspace/sprite-lab/.spritelab/jobs/job-1.log",
            "event_path": "/workspace/sprite-lab/.spritelab/jobs/job-1.events.jsonl",
        },
    )


def test_ssh_subprocess_uses_safe_argv_and_encoded_untrusted_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def run(command, **kwargs):
        captured.update({"command": command, **kwargs})
        return Completed()

    monkeypatch.setattr("spritelab.remote_compute.ssh.subprocess.run", run)
    transport = SubprocessSSHTransport(SSHSettings("example.test", "trainer"))
    dangerous = "$(touch /tmp/pwned); echo 'unsafe'"
    transport.execute("print('{}')", {"argv": [dangerous]})
    assert captured["shell"] is False
    assert isinstance(captured["command"], list)
    assert dangerous not in " ".join(captured["command"])


def test_ssh_reconnect_reports_uncertainty_then_recovers() -> None:
    transport = FakeTransport()
    transport.polls = [RemoteResult(255, stderr="connection lost"), RemoteResult(0, '{"status":"RUNNING"}')]
    backend = _backend(transport)
    first = backend.poll(_job())
    second = backend.poll(_job())
    assert first.status == ComputeStatus.UNCERTAIN
    assert first.resource_state_uncertain and first.may_accrue_cost
    assert second.status == ComputeStatus.RUNNING


def test_ssh_connection_test_reports_environment_without_cuda(tmp_path: Path) -> None:
    capability = _backend().probe(ProjectContext(tmp_path, {}))[0]
    assert capability.status.value == "READY"
    assert capability.details["cuda_initialized"] is False


def test_ssh_settings_reject_secrets_and_unsafe_paths() -> None:
    with pytest.raises(ValueError, match="secrets"):
        SSHSettings.from_mapping({"host": "example.test", "user": "trainer", "password": "no"})
    with pytest.raises(ValueError, match="Remote paths"):
        SSHSettings("example.test", "trainer", "/workspace/../unsafe").validate()


def test_interrupted_ssh_upload_never_reports_success(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.upload_result = RemoteResult(1, stderr="connection reset")
    backend = _backend(transport)
    local = tmp_path / "input.json"
    local.write_text("{}", encoding="utf-8")
    from spritelab.remote_compute import PreparedCompute

    prepared = PreparedCompute("ssh", "operation", "/workspace/sprite-lab", "remote-id")
    with pytest.raises(Exception, match="connection reset"):
        backend.upload(prepared, [local])


def test_download_hash_mismatch_removes_partial_file(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.download_bytes = b"corrupt"
    backend = _backend(transport)
    artifact = ArtifactReference("runs/checkpoint.pt", hashlib.sha256(b"expected").hexdigest(), "remote-id")
    with pytest.raises(ArtifactVerificationError, match="hash mismatch"):
        backend.download_artifacts(_job(), [artifact], tmp_path)
    assert not (tmp_path / "checkpoint.pt.partial").exists()


def test_remote_resource_disappeared_warns_that_cost_may_continue() -> None:
    transport = FakeTransport()
    transport.polls = [RemoteResult(0, '{"status":"MISSING"}')]
    poll = _backend(transport).poll(_job())
    assert poll.status == ComputeStatus.UNCERTAIN
    assert poll.may_accrue_cost
    assert "shut" in poll.message.lower()
