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
    StaleRemoteIdentityError,
    SubprocessSSHTransport,
)
from spritelab.remote_compute.ssh import (
    _CLEANUP_SCRIPT,
    _FINALIZE_UPLOAD_SCRIPT,
    _POLL_SCRIPT,
    _PREPARE_SCRIPT,
    RemoteResult,
)


class FakeTransport:
    def __init__(self) -> None:
        self.polls = []
        self.upload_result = RemoteResult(0)
        self.download_bytes = b""
        self.uploads: list[str] = []

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
        self.uploads.append(str(remote_path))
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

    prepared = PreparedCompute("ssh", "operation", "/workspace/sprite-lab", "a" * 64)
    with pytest.raises(Exception, match="connection reset"):
        backend.upload(prepared, [local])


def test_ssh_upload_uses_unpredictable_confined_remote_partial(tmp_path: Path) -> None:
    from spritelab.remote_compute import PreparedCompute

    transport = FakeTransport()
    backend = _backend(transport)
    local = tmp_path / "input.json"
    local.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    prepared = PreparedCompute("ssh", "operation", "/workspace/sprite-lab", "a" * 64)

    backend.upload(prepared, [local])
    backend.upload(prepared, [local])

    assert len(transport.uploads) == 2
    assert len(set(transport.uploads)) == 2
    for remote_path in transport.uploads:
        assert remote_path.startswith(f"/workspace/sprite-lab/.spritelab/staging/operation/{digest}.")
        assert remote_path.endswith(".partial")


def test_download_hash_mismatch_removes_partial_file(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.download_bytes = b"corrupt"
    backend = _backend(transport)
    artifact = ArtifactReference("runs/checkpoint.pt", hashlib.sha256(b"expected").hexdigest(), "remote-id")
    with pytest.raises(ArtifactVerificationError, match="hash mismatch"):
        backend.download_artifacts(_job(), [artifact], tmp_path)
    assert not (tmp_path / "checkpoint.pt.partial").exists()


def test_download_uses_unique_partial_and_preserves_preplanted_hard_link(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.download_bytes = b"corrupt"
    backend = _backend(transport)
    outside = tmp_path / "outside.bin"
    predictable = tmp_path / "checkpoint.pt.partial"
    outside.write_bytes(b"preserve")
    try:
        predictable.hardlink_to(outside)
    except OSError:
        pytest.skip("hard links are unavailable in this test session")
    artifact = ArtifactReference("runs/checkpoint.pt", hashlib.sha256(b"expected").hexdigest(), "remote-id")

    with pytest.raises(ArtifactVerificationError, match="hash mismatch"):
        backend.download_artifacts(_job(), [artifact], tmp_path)

    assert outside.read_bytes() == b"preserve"
    assert predictable.read_bytes() == b"preserve"


def test_cleanup_rejects_forged_prepared_workspace_and_traversal() -> None:
    from spritelab.remote_compute import PreparedCompute

    backend = _backend()
    with pytest.raises(StaleRemoteIdentityError, match="operation id"):
        backend.cleanup(PreparedCompute("ssh", "../../danger", "/workspace/sprite-lab", "a" * 64))
    with pytest.raises(StaleRemoteIdentityError, match="workspace"):
        backend.cleanup(PreparedCompute("ssh", "operation", "/workspace/other", "a" * 64))


def test_remote_cleanup_script_has_independent_containment_checks() -> None:
    assert "UNSAFE_OPERATION_ID" in _CLEANUP_SCRIPT
    assert "base.parent.parent!=workspace" in _CLEANUP_SCRIPT
    assert "target.resolve().parent!=base" in _CLEANUP_SCRIPT
    assert "target.is_symlink()" in _CLEANUP_SCRIPT


def test_remote_staging_scripts_reject_links_and_predictable_temporary_files() -> None:
    assert "tempfile.mkstemp" in _PREPARE_SCRIPT
    assert "UNSAFE_WORKSPACE_METADATA" in _PREPARE_SCRIPT
    assert "UNSAFE_STAGING_DIRECTORY" in _PREPARE_SCRIPT
    assert "UNSAFE_OPERATION_DIRECTORY" in _PREPARE_SCRIPT
    assert "UNSAFE_PREPARED_MARKER" in _PREPARE_SCRIPT
    assert "UNSAFE_PARTIAL_FILE" in _FINALIZE_UPLOAD_SCRIPT
    assert "UNSAFE_UPLOAD_DESTINATION" in _FINALIZE_UPLOAD_SCRIPT


def test_remote_resource_disappeared_warns_that_cost_may_continue() -> None:
    transport = FakeTransport()
    transport.polls = [RemoteResult(0, '{"status":"MISSING"}')]
    poll = _backend(transport).poll(_job())
    assert poll.status == ComputeStatus.UNCERTAIN
    assert poll.may_accrue_cost
    assert "shut" in poll.message.lower()
