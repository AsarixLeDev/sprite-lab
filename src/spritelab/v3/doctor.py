"""Non-invasive environment checks for the v3 operator layer."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.v3.config import ProjectConfig
from spritelab.v3.status import build_project_state


@dataclass(frozen=True)
class DoctorCheck:
    key: str
    status: str
    mandatory: bool
    message: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_clean(root: Path) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "FAIL", str(exc)
    if result.returncode != 0:
        return "FAIL", result.stderr.strip() or "Git status failed."
    lines = [line for line in result.stdout.splitlines() if line]
    return (
        ("PASS", "Git worktree is clean.") if not lines else ("WARN", f"Git worktree has {len(lines)} changed path(s).")
    )


def run_doctor(config: ProjectConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    py_ok = sys.version_info >= (3, 10)
    checks.append(DoctorCheck("python", "PASS" if py_ok else "FAIL", True, f"Python {sys.version.split()[0]}"))
    checks.append(
        DoctorCheck(
            "package",
            "PASS" if importlib.util.find_spec("spritelab") else "FAIL",
            True,
            "Sprite Lab package is importable."
            if importlib.util.find_spec("spritelab")
            else "Sprite Lab package is not importable.",
        )
    )
    git_dir = config.root / ".git"
    checks.append(
        DoctorCheck("repository", "PASS" if git_dir.exists() else "FAIL", True, f"Project root: {config.root}")
    )
    git_status, git_message = _git_clean(config.root)
    checks.append(DoctorCheck("git-cleanliness", git_status, False, git_message))
    checks.append(
        DoctorCheck(
            "configuration",
            "PASS" if config.path else "FAIL",
            True,
            f"Configuration: {config.path}" if config.path else "No spritelab.yaml is active.",
        )
    )
    try:
        config.runs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=config.runs_dir, prefix=".doctor-", delete=True):
            pass
        writable = True
        write_message = f"Run directory is writable: {config.runs_dir}"
    except OSError as exc:
        writable = False
        write_message = f"Run directory is not writable: {exc}"
    checks.append(DoctorCheck("runs-writable", "PASS" if writable else "FAIL", True, write_message))
    try:
        usage = shutil.disk_usage(config.runs_dir if config.runs_dir.exists() else config.root)
        free_gib = usage.free / (1024**3)
        disk_status = "FAIL" if free_gib < 0.1 else ("WARN" if free_gib < 1 else "PASS")
        checks.append(DoctorCheck("disk-space", disk_status, disk_status == "FAIL", f"{free_gib:.1f} GiB free."))
    except OSError as exc:
        checks.append(DoctorCheck("disk-space", "FAIL", True, str(exc)))
    raw_inventory = config.path_for("dataset", "raw_inventory")
    checks.append(
        DoctorCheck(
            "source-artifacts",
            "PASS" if raw_inventory and raw_inventory.is_file() else "FAIL",
            True,
            f"Raw inventory: {raw_inventory}" if raw_inventory else "Raw inventory is not configured.",
        )
    )
    state = build_project_state(config)
    audit_values = {stage.key: stage.audit.value for stage in state.stages if stage.audit.value != "NOT AUDITED"}
    stale = [key for key, value in audit_values.items() if value == "STALE"]
    checks.append(
        DoctorCheck(
            "audit-freshness",
            "WARN" if stale else "PASS",
            False,
            f"Stale audits: {', '.join(stale)}" if stale else "Configured audits are identity-applicable.",
            audit_values,
        )
    )
    checkpoint = config.path_for("evaluation", "checkpoint")
    checks.append(
        DoctorCheck(
            "checkpoint",
            "PASS" if checkpoint and checkpoint.exists() else "SKIP",
            False,
            f"Checkpoint: {checkpoint}" if checkpoint else "No evaluation checkpoint configured.",
        )
    )
    torch_visible = importlib.util.find_spec("torch") is not None
    checks.append(
        DoctorCheck(
            "cuda-library",
            "PASS" if torch_visible else "SKIP",
            False,
            "Torch is installed; CUDA was not initialized."
            if torch_visible
            else "Torch is not installed; CUDA was not initialized.",
            {"cuda_initialized": False},
        )
    )
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            gpu = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            gpu_status = "PASS" if gpu.returncode == 0 else "WARN"
            gpu_message = gpu.stdout.strip() or gpu.stderr.strip() or "nvidia-smi returned no devices."
        except (OSError, subprocess.SubprocessError) as exc:
            gpu_status, gpu_message = "WARN", str(exc)
    else:
        gpu_status, gpu_message = "SKIP", "nvidia-smi is unavailable; CPU workflows remain visible."
    checks.append(DoctorCheck("gpu-visibility", gpu_status, False, gpu_message, {"cuda_initialized": False}))
    checks.append(
        DoctorCheck(
            "provider",
            "SKIP",
            False,
            "Optional model-provider connectivity was not probed; no network call was made.",
            {"provider_calls": 0},
        )
    )
    checks.append(
        DoctorCheck(
            "git-executable",
            "PASS" if shutil.which("git") else "FAIL",
            True,
            "Git executable is available." if shutil.which("git") else "Git executable is missing.",
        )
    )
    path_text = str(config.root)
    malformed = any(ord(character) < 32 for character in path_text) or "\x00" in path_text
    checks.append(
        DoctorCheck(
            "path-syntax",
            "FAIL" if malformed else "PASS",
            True,
            "Project path is syntactically safe." if not malformed else "Project path contains control characters.",
        )
    )
    longest = (
        max((len(str(path)) for path in config.root.rglob("*")), default=len(path_text))
        if os.name == "nt"
        else len(path_text)
    )
    checks.append(
        DoctorCheck(
            "windows-path-length",
            "WARN" if os.name == "nt" and longest >= 240 else "PASS",
            False,
            f"Longest observed path is {longest} characters.",
        )
    )
    return checks
