"""Validated, non-secret product configuration for training compute."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.product_core import (
    ProductSettingsError,
    ProductSettingsRepository,
    ProjectContext,
    reject_secret_settings,
)

HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
USER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]{0,31}$")
ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BACKEND_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,190}$")


@dataclass(frozen=True)
class ComputeSettings:
    backend_type: str = "local"
    device_policy: str = "auto"
    memory_limit_gb: float | None = None
    cpu_threads: int | None = None
    preview_interval: int = 500
    run_profile: str = "recommended"
    host: str | None = None
    port: int = 22
    username: str | None = None
    remote_workspace: str = "/workspace/sprite-lab"
    credential_reference: str | None = None
    environment_profile: str = "python3"
    artifact_sync_policy: str = "verified_checkpoints"
    cloud: bool = True
    backend_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, allow_unavailable: bool = True) -> ComputeSettings:
        raw = dict(value or {})
        reject_secret_settings(raw)
        backend_type = str(raw.get("type") or raw.get("backend_type") or "local").strip().lower()
        aliases = {"remote_ssh": "ssh", "plugin": "other", "hosted": "other"}
        backend_type = aliases.get(backend_type, backend_type)
        if backend_type not in {"local", "ssh", "runpod", "other"}:
            raise ValueError("Compute type must be local, ssh, runpod, or other.")
        if backend_type == "runpod" and not allow_unavailable:
            raise ValueError("RunPod is not available in this build and cannot be saved as launch-ready compute.")
        device_policy = str(raw.get("device_policy") or "auto").strip().lower()
        if device_policy not in {"auto", "cpu", "cuda"}:
            raise ValueError("Local device policy must be auto, cpu, or cuda.")
        memory_limit = _optional_float(raw.get("memory_limit_gb"), minimum=0.25, maximum=4096)
        cpu_threads = _optional_int(raw.get("cpu_threads"), minimum=1, maximum=1024)
        preview_interval = int(raw.get("preview_interval", 500))
        if not 1 <= preview_interval <= 10_000_000:
            raise ValueError("Preview interval must be between 1 and 10000000 optimizer steps.")
        run_profile = str(raw.get("run_profile") or "recommended").strip().lower()
        if run_profile not in {"recommended", "fast_preview", "quality", "custom"}:
            raise ValueError("Run profile is not supported.")
        host = _optional_text(raw.get("host"))
        username = _optional_text(raw.get("username", raw.get("user")))
        port = int(raw.get("port", 22))
        workspace = str(raw.get("remote_workspace", raw.get("workspace", "/workspace/sprite-lab"))).strip()
        credential_reference = _optional_text(raw.get("credential_reference"))
        environment_profile = str(raw.get("environment_profile", raw.get("python_executable", "python3"))).strip()
        sync_policy = str(raw.get("artifact_sync_policy") or "verified_checkpoints").strip().lower()
        backend_id = _optional_text(raw.get("backend_id"))
        cloud = raw.get("cloud", True)
        if type(cloud) is not bool:
            raise ValueError("Compute cloud classification must be the JSON boolean true or false.")
        if backend_type == "ssh":
            if host is None or not HOST_PATTERN.fullmatch(host):
                raise ValueError("SSH host must be a DNS name or IPv4 address without command-line syntax.")
            if username is None or not USER_PATTERN.fullmatch(username):
                raise ValueError("SSH username contains unsupported characters.")
            if not 1 <= port <= 65535:
                raise ValueError("SSH port must be between 1 and 65535.")
            if not workspace.startswith("/") or "\x00" in workspace or ".." in Path(workspace).parts:
                raise ValueError("Remote workspace must be an absolute safe Unix path.")
            if environment_profile not in {"python", "python3"}:
                raise ValueError("SSH environment profile must select python or python3.")
            if credential_reference and not (
                credential_reference in {"ssh-agent", "default"}
                or credential_reference.startswith("file:")
                or credential_reference.startswith("env:")
            ):
                raise ValueError("SSH credential reference must be ssh-agent, default, file:<path>, or env:<name>.")
            if (
                credential_reference
                and credential_reference.startswith("env:")
                and not ENV_PATTERN.fullmatch(credential_reference[4:])
            ):
                raise ValueError("SSH credential environment-variable name is invalid.")
        if sync_policy not in {"verified_checkpoints", "all_artifacts", "manual"}:
            raise ValueError("Artifact synchronization policy is invalid.")
        if backend_id is not None and BACKEND_ID_PATTERN.fullmatch(backend_id) is None:
            raise ValueError("Backend ID may contain only letters, numbers, '.', '_', and '-'.")
        if backend_type == "other" and backend_id is None:
            raise ValueError("Other provider requires a registered backend ID.")
        return cls(
            backend_type=backend_type,
            device_policy=device_policy,
            memory_limit_gb=memory_limit,
            cpu_threads=cpu_threads,
            preview_interval=preview_interval,
            run_profile=run_profile,
            host=host,
            port=port,
            username=username,
            remote_workspace=workspace,
            credential_reference=credential_reference,
            environment_profile=environment_profile,
            artifact_sync_policy=sync_policy,
            cloud=cloud,
            backend_id=backend_id,
        )

    def to_persisted_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.backend_type,
            "device_policy": self.device_policy,
            "memory_limit_gb": self.memory_limit_gb,
            "cpu_threads": self.cpu_threads,
            "preview_interval": self.preview_interval,
            "run_profile": self.run_profile,
            "artifact_sync_policy": self.artifact_sync_policy,
        }
        if self.backend_type == "ssh":
            value.update(
                {
                    "host": self.host,
                    "port": self.port,
                    "username": self.username,
                    "remote_workspace": self.remote_workspace,
                    "credential_reference": self.credential_reference,
                    "environment_profile": self.environment_profile,
                    "cloud": self.cloud,
                }
            )
        if self.backend_type == "other":
            value["backend_id"] = self.backend_id
        reject_secret_settings(value)
        return value

    def backend_mapping(self) -> dict[str, Any]:
        value = self.to_persisted_dict()
        if self.backend_type == "ssh":
            value.update(
                {
                    "user": self.username,
                    "workspace": self.remote_workspace,
                    "python_executable": self.environment_profile,
                }
            )
            reference = self.credential_reference or ""
            if reference.startswith("file:"):
                value["identity_file"] = reference[5:]
        return value

    def execution_environment(self) -> dict[str, str]:
        environment = {"SPRITELAB_PREVIEW_INTERVAL": str(self.preview_interval)}
        if self.backend_type == "local":
            environment["SPRITELAB_DEVICE_POLICY"] = self.device_policy
            if self.device_policy == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = "-1"
            elif self.device_policy == "cuda":
                environment["CUDA_VISIBLE_DEVICES"] = "0"
            if self.cpu_threads:
                environment["OMP_NUM_THREADS"] = str(self.cpu_threads)
            if self.memory_limit_gb:
                environment["SPRITELAB_MEMORY_LIMIT_GB"] = f"{self.memory_limit_gb:g}"
        return environment


def effective_compute_context(context: ProjectContext) -> tuple[ProjectContext, ComputeSettings, int, bool]:
    repository = ProductSettingsRepository(context)
    raw, version, saved = repository.effective_settings("compute")
    settings = ComputeSettings.from_mapping(raw, allow_unavailable=True)
    effective = repository.effective_context()
    values = dict(effective.config)
    compute = dict(values.get("compute", {})) if isinstance(values.get("compute"), Mapping) else {}
    compute["training"] = settings.backend_mapping()
    values["compute"] = compute
    return (
        ProjectContext(effective.project_root, values, effective.config_path, effective.runs_directory),
        settings,
        version,
        saved,
    )


def passive_compute_projection(context: ProjectContext) -> dict[str, Any]:
    try:
        _effective, settings, version, saved = effective_compute_context(context)
    except (ValueError, ProductSettingsError) as exc:
        return {
            "state": "invalid",
            "message": str(exc),
            "compute_probes": 0,
            "configuration_version": 0,
        }
    available = settings.backend_type != "runpod"
    return {
        "state": "configured_unverified" if available else "unavailable",
        "backend_type": settings.backend_type,
        "message": "Configured; use Test connection for a current check."
        if available
        else "RunPod is not available in this build.",
        "configuration_version": version,
        "saved": saved,
        "compute_probes": 0,
        "remote_calls": 0,
    }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"Compute integer setting must be between {minimum} and {maximum}.")
    return result


def _optional_float(value: Any, *, minimum: float, maximum: float) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"Compute numeric setting must be between {minimum} and {maximum}.")
    return result


__all__ = [
    "ComputeSettings",
    "effective_compute_context",
    "passive_compute_projection",
]
