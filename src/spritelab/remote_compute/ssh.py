"""Generic Unix SSH backend using safe local argv and a fixed remote Python shim."""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from spritelab.product_core import ProductCapability, ProductEvent, ProductStatus, ProjectContext
from spritelab.remote_compute.contracts import (
    ArtifactReference,
    ArtifactVerificationError,
    CloudConfirmationRequired,
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
    verify_compute_job_request,
)
from spritelab.remote_compute.utils import file_sha256, stable_hash, validate_identifier, validate_remote_path

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
_USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]{0,31}$")


@dataclass(frozen=True)
class SSHSettings:
    host: str
    user: str
    workspace: str = "/workspace/sprite-lab"
    port: int = 22
    identity_file: Path | None = None
    python_executable: str = "python3"
    connect_timeout_seconds: int = 10
    cloud: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SSHSettings:
        forbidden = sorted(
            key
            for key in value
            if any(mark in str(key).lower() for mark in ("password", "token", "secret", "private_key"))
        )
        if forbidden:
            raise ValueError("SSH secrets must not be stored in project configuration: " + ", ".join(forbidden))
        settings = cls(
            host=str(value.get("host") or ""),
            user=str(value.get("user") or ""),
            workspace=str(value.get("workspace") or "/workspace/sprite-lab"),
            port=int(value.get("port", 22)),
            identity_file=Path(str(value["identity_file"])).expanduser() if value.get("identity_file") else None,
            python_executable=str(value.get("python_executable") or "python3"),
            connect_timeout_seconds=int(value.get("connect_timeout_seconds", 10)),
            cloud=bool(value.get("cloud", True)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not _HOST_RE.fullmatch(self.host):
            raise ValueError("SSH host must be a DNS name or IPv4 address without command-line syntax.")
        if not _USER_RE.fullmatch(self.user):
            raise ValueError("SSH user contains unsupported characters.")
        if not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535.")
        validate_remote_path(self.workspace)
        if self.python_executable not in {"python3", "python"}:
            raise ValueError("Remote Python executable must be 'python3' or 'python'.")
        if not 1 <= self.connect_timeout_seconds <= 120:
            raise ValueError("SSH connection timeout must be between 1 and 120 seconds.")


@dataclass(frozen=True)
class RemoteResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class SSHTransport(Protocol):
    def execute(self, script: str, payload: Mapping[str, Any]) -> RemoteResult: ...

    def upload(self, local_path: Path, remote_path: str) -> RemoteResult: ...

    def download(self, remote_path: str, local_path: Path) -> RemoteResult: ...


class SubprocessSSHTransport:
    """OpenSSH transport. Local subprocesses always use ``shell=False``."""

    def __init__(self, settings: SSHSettings) -> None:
        self.settings = settings

    @property
    def target(self) -> str:
        return f"{self.settings.user}@{self.settings.host}"

    def _base(self) -> list[str]:
        result = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.settings.connect_timeout_seconds}",
            "-p",
            str(self.settings.port),
        ]
        if self.settings.identity_file:
            result.extend(["-i", str(self.settings.identity_file)])
        result.append(self.target)
        return result

    def execute(self, script: str, payload: Mapping[str, Any]) -> RemoteResult:
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        # OpenSSH sends one command string to the remote login shell. Every token
        # here is trusted adapter code or URL-safe base64 and is POSIX-quoted.
        remote_command = shlex.join([self.settings.python_executable, "-c", script, encoded])
        completed = subprocess.run(
            [*self._base(), remote_command], capture_output=True, text=True, shell=False, check=False
        )
        return RemoteResult(completed.returncode, completed.stdout, completed.stderr)

    def upload(self, local_path: Path, remote_path: str) -> RemoteResult:
        validate_remote_path(remote_path)
        command = ["scp", "-q", "-P", str(self.settings.port)]
        if self.settings.identity_file:
            command.extend(["-i", str(self.settings.identity_file)])
        command.extend([str(local_path), f"{self.target}:{remote_path}"])
        completed = subprocess.run(command, capture_output=True, text=True, shell=False, check=False)
        return RemoteResult(completed.returncode, completed.stdout, completed.stderr)

    def download(self, remote_path: str, local_path: Path) -> RemoteResult:
        validate_remote_path(remote_path)
        command = ["scp", "-q", "-P", str(self.settings.port)]
        if self.settings.identity_file:
            command.extend(["-i", str(self.settings.identity_file)])
        command.extend([f"{self.target}:{remote_path}", str(local_path)])
        completed = subprocess.run(command, capture_output=True, text=True, shell=False, check=False)
        return RemoteResult(completed.returncode, completed.stdout, completed.stderr)


_DECODE = "import base64,json,sys;p=json.loads(base64.urlsafe_b64decode(sys.argv[1].encode()).decode())"
_PROBE_SCRIPT = (
    _DECODE
    + ";import shutil;u=shutil.disk_usage(p['workspace']);print(json.dumps({'python':sys.version.split()[0],'disk_free_bytes':u.free}))"
)
_PREPARE_SCRIPT = (
    _DECODE
    + r"""
import os,re,tempfile
from pathlib import Path
operation=str(p['operation_id'])
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,190}',operation): raise SystemExit('UNSAFE_OPERATION_ID')
w=Path(p['workspace']).resolve(); root=w/'.spritelab'
if root.is_symlink(): raise SystemExit('UNSAFE_WORKSPACE_METADATA')
root.mkdir(parents=True,exist_ok=True)
if not root.is_dir() or root.resolve().parent!=w: raise SystemExit('UNSAFE_WORKSPACE_METADATA')
prepared=root/'prepared'
if prepared.is_symlink(): raise SystemExit('UNSAFE_PREPARED_DIRECTORY')
prepared.mkdir(exist_ok=True)
if not prepared.is_dir() or prepared.resolve().parent!=root.resolve(): raise SystemExit('UNSAFE_PREPARED_DIRECTORY')
staging=root/'staging'
if staging.is_symlink(): raise SystemExit('UNSAFE_STAGING_DIRECTORY')
staging.mkdir(exist_ok=True)
if not staging.is_dir() or staging.resolve().parent!=root.resolve(): raise SystemExit('UNSAFE_STAGING_DIRECTORY')
operation_dir=staging/operation
if operation_dir.is_symlink(): raise SystemExit('UNSAFE_OPERATION_DIRECTORY')
operation_dir.mkdir(exist_ok=True)
if not operation_dir.is_dir() or operation_dir.resolve().parent!=staging.resolve(): raise SystemExit('UNSAFE_OPERATION_DIRECTORY')
marker=prepared/f"{operation}.json"
if marker.is_symlink(): raise SystemExit('UNSAFE_PREPARED_MARKER')
expected={'operation_id':operation,'remote_identity':p['remote_identity']}
if marker.exists() and json.loads(marker.read_text()) != expected:
    raise SystemExit('STALE_REMOTE_IDENTITY')
if not marker.exists():
    fd,name=tempfile.mkstemp(prefix=f'.{marker.name}.',suffix='.partial',dir=marker.parent)
    with os.fdopen(fd,'w') as handle: handle.write(json.dumps(expected,sort_keys=True)); handle.flush(); os.fsync(handle.fileno())
    os.replace(name,marker)
print(json.dumps(expected))
"""
)
_HASH_SCRIPT = (
    _DECODE
    + r"""
import hashlib
from pathlib import Path
x=Path(p['path']); h=hashlib.sha256()
if not x.is_file(): raise SystemExit('MISSING')
with x.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''): h.update(b)
print(json.dumps({'sha256':h.hexdigest()}))
"""
)
_MKDIR_SCRIPT = (
    _DECODE
    + ";from pathlib import Path;Path(p['path']).mkdir(parents=True,exist_ok=True);print(json.dumps({'changed':True}))"
)
_FINALIZE_UPLOAD_SCRIPT = (
    _DECODE
    + r"""
import hashlib,os
from pathlib import Path
operation=str(p['operation_id']); workspace=Path(p['workspace']).resolve()
staging=(workspace/'.spritelab'/'staging'/operation).resolve()
if staging.parent.parent.parent!=workspace: raise SystemExit('UNSAFE_STAGING_DIRECTORY')
src=Path(p['partial']); dst=Path(p['destination']); h=hashlib.sha256()
if src.resolve().parent!=staging: raise SystemExit('UNSAFE_PARTIAL_FILE')
if src.is_symlink() or not src.is_file(): raise SystemExit('UNSAFE_PARTIAL_FILE')
with src.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''): h.update(b)
if h.hexdigest()!=p['sha256']: src.unlink(missing_ok=True); raise SystemExit('HASH_MISMATCH')
parent=dst.parent.resolve()
try: relative=parent.relative_to(workspace)
except ValueError: raise SystemExit('UNSAFE_UPLOAD_DESTINATION')
if not relative.parts: raise SystemExit('UNSAFE_UPLOAD_DESTINATION')
dst.parent.mkdir(parents=True,exist_ok=True)
if dst.parent.resolve()!=parent: raise SystemExit('UNSAFE_UPLOAD_DESTINATION')
os.replace(src,dst); print(json.dumps({'sha256':h.hexdigest()}))
"""
)
_WORKER_SCRIPT = (
    _DECODE
    + r"""
import os,subprocess
from pathlib import Path
s=Path(p['state_path']); state=json.loads(s.read_text()); state['status']='RUNNING'
log=Path(p['log_path']); log.parent.mkdir(parents=True,exist_ok=True)
with log.open('ab') as out:
    child=subprocess.Popen(p['command'],cwd=p['workspace'],env={**os.environ,**p['environment']},stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,shell=False,start_new_session=True)
    state['pid']=child.pid; state['pgid']=child.pid; s.write_text(json.dumps(state,sort_keys=True))
    code=child.wait()
state=json.loads(s.read_text()); state['exit_code']=code
if state.get('status') not in {'PAUSING','CANCELLING'}: state['status']='COMPLETE' if code==0 else 'FAILED'
else: state['status']='PAUSED' if state['status']=='PAUSING' else 'CANCELLED'
s.write_text(json.dumps(state,sort_keys=True))
"""
)
_LAUNCH_SCRIPT = (
    _DECODE
    + r"""
import base64,json,os,subprocess,sys
from pathlib import Path
s=Path(p['state_path']); s.parent.mkdir(parents=True,exist_ok=True)
if s.exists():
    state=json.loads(s.read_text())
    if state.get('remote_identity')!=p['remote_identity']: raise SystemExit('STALE_REMOTE_IDENTITY')
    print(json.dumps(state)); raise SystemExit(0)
state={'job_id':p['job_id'],'run_id':p['run_id'],'status':'STARTING','remote_identity':p['remote_identity'],'log_path':p['log_path'],'event_path':p['event_path']}
s.write_text(json.dumps(state,sort_keys=True))
worker_payload={**p,'state_path':str(s)}
encoded=base64.urlsafe_b64encode(json.dumps(worker_payload,separators=(',',':')).encode()).decode()
subprocess.Popen([p['python_executable'],'-c',p['worker_script'],encoded],cwd=p['workspace'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,close_fds=True)
print(json.dumps(state))
"""
)
_POLL_SCRIPT = (
    _DECODE
    + ";from pathlib import Path;s=Path(p['state_path']);print(s.read_text() if s.is_file() else json.dumps({'status':'MISSING'}))"
)
_SIGNAL_SCRIPT = (
    _DECODE
    + r"""
import os,signal
from pathlib import Path
s=Path(p['state_path'])
if not s.is_file(): print(json.dumps({'changed':False,'status':'MISSING'})); raise SystemExit(0)
state=json.loads(s.read_text())
if state.get('remote_identity')!=p['remote_identity']: raise SystemExit('STALE_REMOTE_IDENTITY')
if state.get('status') not in {'STARTING','RUNNING'}: print(json.dumps({'changed':False,'status':state.get('status')})); raise SystemExit(0)
pgid=state.get('pgid')
if pgid: os.killpg(int(pgid),signal.SIGINT if p['action']=='pause' else signal.SIGTERM)
state['status']='PAUSING' if p['action']=='pause' else 'CANCELLING'; s.write_text(json.dumps(state,sort_keys=True)); print(json.dumps({'changed':True,'status':state['status']}))
"""
)
_READ_EVENTS_SCRIPT = (
    _DECODE
    + r"""
from pathlib import Path
event=Path(p['event_path']); log=Path(p['log_path']); rows=[]
if event.is_file(): rows.extend(event.read_text(errors='replace').splitlines())
if log.is_file(): rows.extend(json.dumps({'_log':x}) for x in log.read_text(errors='replace').splitlines())
print(json.dumps({'rows':rows[p['cursor']:],'cursor':len(rows)}))
"""
)
_CLEANUP_SCRIPT = (
    _DECODE
    + r"""
import re,shutil
from pathlib import Path
operation=str(p['operation_id'])
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,190}',operation): raise SystemExit('UNSAFE_OPERATION_ID')
workspace=Path(p['workspace']).resolve()
base=(workspace/'.spritelab'/'staging').resolve()
if base.parent.parent!=workspace: raise SystemExit('UNSAFE_STAGING_DIRECTORY')
target=base/operation
if target.resolve().parent!=base: raise SystemExit('UNSAFE_CLEANUP_TARGET')
if target.is_symlink(): raise SystemExit('UNSAFE_CLEANUP_LINK')
changed=target.exists()
if changed:
    if not target.is_dir(): raise SystemExit('UNSAFE_CLEANUP_TARGET')
    shutil.rmtree(target)
print(json.dumps({'changed':changed}))
"""
)


class SSHComputeBackend:
    backend_id = "ssh"
    title = "Remote SSH machine"

    def __init__(self, settings: SSHSettings, *, transport: SSHTransport | None = None) -> None:
        settings.validate()
        self.settings = settings
        self.is_cloud = settings.cloud
        self.transport = transport or SubprocessSSHTransport(settings)

    def _execute(self, script: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = self.transport.execute(script, payload)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "remote operation failed").strip()
            if "STALE_REMOTE_IDENTITY" in message:
                raise StaleRemoteIdentityError("Remote workspace identity is stale.")
            raise ComputeBackendError(message)
        try:
            value = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise ComputeBackendError("Remote operation returned invalid JSON.") from exc
        if not isinstance(value, dict):
            raise ComputeBackendError("Remote operation returned an invalid response.")
        return value

    def probe(self, context: ProjectContext) -> Sequence[ProductCapability]:
        del context
        try:
            result = self._execute(_PROBE_SCRIPT, {"workspace": self.settings.workspace})
        except ComputeBackendError as exc:
            return (
                ProductCapability(
                    "compute.ssh",
                    self.title,
                    ProductStatus.UNAVAILABLE,
                    "SSH connection or environment check failed.",
                    {"error": str(exc), "credential_status": "SSH key unavailable or rejected"},
                ),
            )
        return (
            ProductCapability(
                "compute.ssh",
                self.title,
                ProductStatus.READY,
                "SSH connection and remote Python environment are available.",
                {**result, "credential_status": "SSH key accepted", "cuda_initialized": False},
            ),
        )

    def estimate(self, context: ProjectContext, campaign: Mapping[str, Any]) -> ComputeEstimate:
        del context
        product = campaign.get("product_estimate") if isinstance(campaign.get("product_estimate"), Mapping) else {}
        duration = product.get("duration_seconds")
        return ComputeEstimate(
            duration_seconds=int(duration) if isinstance(duration, (int, float)) and duration > 0 else None,
            disk_required_bytes=int(product.get("disk_required_bytes", 0) or 0),
            trustworthy=isinstance(duration, (int, float)) and duration > 0,
            source="campaign product estimate" if duration else None,
            message="Remote-machine cost is not inferred; configure it outside project secrets."
            if not duration
            else "Campaign time estimate; remote cost unavailable.",
        )

    def prepare(self, context: ProjectContext, request: ComputeJobRequest) -> PreparedCompute:
        verify_compute_job_request(request, backend_id=self.backend_id)
        del context
        operation = validate_identifier(request.idempotency_key, label="idempotency key")
        identity = stable_hash(
            {
                "backend": self.backend_id,
                "operation": operation,
                "campaign": request.campaign_identity,
                "run": request.run_identity,
                "workspace": self.settings.workspace,
            }
        )
        verify_compute_job_request(request, backend_id=self.backend_id)
        self._execute(
            _PREPARE_SCRIPT,
            {"workspace": self.settings.workspace, "operation_id": operation, "remote_identity": identity},
        )
        return PreparedCompute(self.backend_id, operation, self.settings.workspace, identity)

    def upload(
        self, prepared: PreparedCompute, artifacts: Sequence[Path], *, remote_subdirectory: str = "inputs"
    ) -> OperationResult:
        self._validate_prepared(prepared)
        subdirectory = validate_identifier(remote_subdirectory, label="remote subdirectory")
        changed = False
        uploaded: list[dict[str, str]] = []
        for local in artifacts:
            if not local.is_file():
                raise FileNotFoundError(f"SSH upload supports files only: {local}")
            digest = file_sha256(local)
            destination = str(PurePosixPath(prepared.workspace, subdirectory, digest, local.name))
            validate_remote_path(destination)
            try:
                current = self._execute(_HASH_SCRIPT, {"path": destination})
            except ComputeBackendError:
                current = {}
            if current.get("sha256") == digest:
                uploaded.append({"path": destination, "sha256": digest})
                continue
            staging = str(
                PurePosixPath(
                    prepared.workspace,
                    ".spritelab",
                    "staging",
                    prepared.operation_id,
                    f"{digest}.{secrets.token_hex(16)}.partial",
                )
            )
            self._execute(
                _PREPARE_SCRIPT,
                {
                    "workspace": self.settings.workspace,
                    "operation_id": prepared.operation_id,
                    "remote_identity": prepared.remote_identity,
                },
            )
            self._execute(_MKDIR_SCRIPT, {"path": str(PurePosixPath(staging).parent)})
            result = self.transport.upload(local, staging)
            if result.returncode != 0:
                raise ComputeBackendError((result.stderr or result.stdout or "SSH upload failed").strip())
            self._execute(
                _FINALIZE_UPLOAD_SCRIPT,
                {
                    "workspace": self.settings.workspace,
                    "operation_id": prepared.operation_id,
                    "partial": staging,
                    "destination": destination,
                    "sha256": digest,
                },
            )
            changed = True
            uploaded.append({"path": destination, "sha256": digest})
        return OperationResult(changed, "SSH inputs uploaded and hash-verified.", {"artifacts": uploaded})

    def _map_command(self, request: ComputeJobRequest, workspace: str) -> list[str]:
        mapped: list[str] = []
        root = request.local_project_root.resolve()
        for index, item in enumerate(request.command):
            if index == 0 and (item == sys.executable or item.lower().endswith(("python.exe", "python3.exe"))):
                mapped.append(self.settings.python_executable)
                continue
            try:
                candidate = Path(item)
                resolved = candidate.resolve() if candidate.is_absolute() else None
                relative = resolved.relative_to(root) if resolved is not None else None
            except (OSError, ValueError):
                relative = None
            mapped.append(str(PurePosixPath(workspace, *relative.parts)) if relative is not None else item)
        return mapped

    def launch(
        self, prepared: PreparedCompute, request: ComputeJobRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        self._validate_prepared(prepared)
        if self.is_cloud and not cloud_confirmation:
            raise CloudConfirmationRequired("Explicit cloud confirmation is required before SSH launch.")
        verify_compute_job_request(request, backend_id=self.backend_id)
        expected_prepared_identity = stable_hash(
            {
                "backend": self.backend_id,
                "operation": request.idempotency_key,
                "campaign": request.campaign_identity,
                "run": request.run_identity,
                "workspace": self.settings.workspace,
            }
        )
        if (
            prepared.backend_id != self.backend_id
            or prepared.operation_id != request.idempotency_key
            or prepared.remote_identity != expected_prepared_identity
        ):
            raise StaleRemoteIdentityError("Prepared SSH operation does not match the validated launch request.")
        job_id = validate_identifier(request.idempotency_key, label="job id")
        state_path = str(PurePosixPath(prepared.workspace, ".spritelab", "jobs", f"{job_id}.json"))
        log_path = str(PurePosixPath(prepared.workspace, ".spritelab", "jobs", f"{job_id}.log"))
        event_path = str(PurePosixPath(prepared.workspace, ".spritelab", "jobs", f"{job_id}.events.jsonl"))
        command = self._map_command(request, prepared.workspace)
        verify_compute_job_request(request, backend_id=self.backend_id)
        state = self._execute(
            _LAUNCH_SCRIPT,
            {
                "job_id": job_id,
                "run_id": request.run_id,
                "remote_identity": prepared.remote_identity,
                "state_path": state_path,
                "log_path": log_path,
                "event_path": event_path,
                "workspace": prepared.workspace,
                "command": command,
                "environment": dict(request.environment),
                "python_executable": self.settings.python_executable,
                "worker_script": _WORKER_SCRIPT,
            },
        )
        return ComputeJob(
            self.backend_id,
            job_id,
            request.run_id,
            _compute_status(state.get("status")),
            prepared.remote_identity,
            may_accrue_cost=self.is_cloud,
            metadata={"state_path": state_path, "log_path": log_path, "event_path": event_path},
        )

    def poll(self, job: ComputeJob) -> ComputePoll:
        try:
            state = self._execute(_POLL_SCRIPT, {"state_path": job.metadata["state_path"]})
        except ComputeBackendError:
            return ComputePoll(
                ComputeStatus.UNCERTAIN,
                "Connection lost; the remote process state is unknown. Check the machine directly and shut it down if appropriate.",
                may_accrue_cost=job.may_accrue_cost,
                resource_state_uncertain=True,
            )
        status = _compute_status(state.get("status"))
        if status == ComputeStatus.UNAVAILABLE:
            return ComputePoll(
                ComputeStatus.UNCERTAIN,
                "Remote resource or job state disappeared. It may still be accruing cost; inspect and shut down the provider resource.",
                may_accrue_cost=job.may_accrue_cost,
                resource_state_uncertain=True,
            )
        return ComputePoll(
            status,
            f"Remote job is {status.value.lower()}.",
            job.may_accrue_cost,
            exit_code=state.get("exit_code"),
            metadata=state,
        )

    def stream_events(self, job: ComputeJob, *, cursor: int = 0) -> tuple[Sequence[ProductEvent], int]:
        result = self._execute(
            _READ_EVENTS_SCRIPT,
            {"event_path": job.metadata["event_path"], "log_path": job.metadata["log_path"], "cursor": cursor},
        )
        events = []
        for raw in result.get("rows", []):
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(raw)
                if "_log" in payload:
                    raise KeyError
                events.append(ProductEvent.from_dict(payload))
            except (ValueError, KeyError, json.JSONDecodeError):
                message = payload.get("_log", raw) if isinstance(payload, dict) else raw
                events.append(
                    ProductEvent(
                        run_id=job.run_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        feature="training",
                        stage="logs",
                        event_type="log",
                        status=ProductStatus.RUNNING,
                        message=str(message),
                    )
                )
        return tuple(events), int(result.get("cursor", cursor))

    def _signal(self, job: ComputeJob, action: str) -> OperationResult:
        result = self._execute(
            _SIGNAL_SCRIPT,
            {
                "state_path": job.metadata["state_path"],
                "remote_identity": job.remote_identity,
                "action": action,
            },
        )
        return OperationResult(bool(result.get("changed")), f"Remote {action} request processed.", result)

    def pause(self, job: ComputeJob) -> OperationResult:
        return self._signal(job, "pause")

    def cancel(self, job: ComputeJob) -> OperationResult:
        return self._signal(job, "cancel")

    def resume(
        self, prepared: PreparedCompute, resume: ResumeRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        checkpoint = resume.checkpoint
        if not resume.safe_resume or not checkpoint.safe_for_remote_resume:
            raise ArtifactVerificationError(
                "Unsafe remote resume is unavailable; download, hash, and identity checks are required."
            )
        if checkpoint.remote_identity != prepared.remote_identity:
            raise StaleRemoteIdentityError("Checkpoint belongs to a stale remote identity.")
        return self.launch(prepared, resume.request, cloud_confirmation=cloud_confirmation)

    def download_artifacts(
        self, job: ComputeJob, artifacts: Sequence[ArtifactReference], destination: Path
    ) -> Sequence[ArtifactReference]:
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = []
        for artifact in artifacts:
            if artifact.remote_identity != job.remote_identity:
                raise StaleRemoteIdentityError("Remote artifact identity does not match the job.")
            relative = PurePosixPath(artifact.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Artifact paths must be relative to the prepared workspace.")
            remote = str(PurePosixPath(self.settings.workspace, *relative.parts))
            target = destination / relative.name
            descriptor, partial_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".partial",
                dir=destination,
            )
            os.close(descriptor)
            partial = Path(partial_name)
            try:
                result = self.transport.download(remote, partial)
                if result.returncode != 0:
                    raise ComputeBackendError((result.stderr or result.stdout or "SSH download failed").strip())
                actual = file_sha256(partial)
                if actual != artifact.sha256:
                    raise ArtifactVerificationError(f"Downloaded artifact hash mismatch: {artifact.relative_path}")
                os.replace(partial, target)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            downloaded.append(
                replace(
                    artifact,
                    local_path=target,
                    downloaded=True,
                    hash_verified=True,
                    remote_identity_verified=artifact.remote_identity == job.remote_identity,
                )
            )
        return tuple(downloaded)

    def verify_artifacts(self, job: ComputeJob, artifacts: Sequence[ArtifactReference]) -> Sequence[ArtifactReference]:
        verified = []
        for artifact in artifacts:
            if artifact.remote_identity != job.remote_identity:
                raise StaleRemoteIdentityError("Remote artifact identity does not match the job.")
            relative = PurePosixPath(artifact.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Artifact paths must be relative to the prepared workspace.")
            remote = str(PurePosixPath(self.settings.workspace, *relative.parts))
            result = self._execute(_HASH_SCRIPT, {"path": remote})
            if result.get("sha256") != artifact.sha256:
                raise ArtifactVerificationError(f"Remote artifact hash mismatch: {artifact.relative_path}")
            local_ok = (
                artifact.local_path is not None
                and artifact.local_path.is_file()
                and file_sha256(artifact.local_path) == artifact.sha256
            )
            verified.append(
                replace(
                    artifact,
                    downloaded=artifact.downloaded and local_ok,
                    hash_verified=artifact.hash_verified and local_ok,
                    remote_identity_verified=True,
                )
            )
        return tuple(verified)

    def cleanup(self, prepared: PreparedCompute) -> OperationResult:
        self._validate_prepared(prepared)
        result = self._execute(
            _CLEANUP_SCRIPT,
            {"workspace": self.settings.workspace, "operation_id": prepared.operation_id},
        )
        return OperationResult(bool(result.get("changed")), "Remote staging cleaned; run artifacts were preserved.")

    def _validate_prepared(self, prepared: PreparedCompute) -> None:
        try:
            operation_id = validate_identifier(prepared.operation_id, label="prepared operation id")
            workspace = validate_remote_path(prepared.workspace)
            configured_workspace = validate_remote_path(self.settings.workspace)
        except ValueError as exc:
            raise StaleRemoteIdentityError(str(exc)) from exc
        if prepared.backend_id != self.backend_id:
            raise StaleRemoteIdentityError("Prepared compute belongs to a different backend.")
        if workspace != configured_workspace:
            raise StaleRemoteIdentityError("Prepared compute workspace does not match configured SSH workspace.")
        if operation_id != prepared.operation_id or not re.fullmatch(r"[a-f0-9]{64}", prepared.remote_identity):
            raise StaleRemoteIdentityError("Prepared compute has an invalid remote identity.")


def _compute_status(value: Any) -> ComputeStatus:
    mapping = {"STARTING": ComputeStatus.RUNNING, "MISSING": ComputeStatus.UNAVAILABLE}
    text = str(value or "MISSING").upper()
    if text in mapping:
        return mapping[text]
    try:
        return ComputeStatus(text)
    except ValueError:
        return ComputeStatus.UNCERTAIN
