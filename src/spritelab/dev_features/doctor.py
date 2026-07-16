"""Read-only developer environment checks, separate from product doctor."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

from spritelab.dev_features.artifacts import inspect_artifacts
from spritelab.dev_features.repository import list_worktrees, repository_state
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import ProjectState


def _check(key: str, status: str, message: str, *, mandatory: bool = False, detail: Any = None) -> dict[str, Any]:
    return {"key": key, "status": status, "mandatory": mandatory, "message": message, "detail": detail}


def _module_check(name: str, *, mandatory: bool) -> dict[str, Any]:
    available = importlib.util.find_spec(name) is not None
    return _check(
        f"test-dependency:{name}",
        "PASS" if available else ("FAIL" if mandatory else "WARN"),
        f"Python module {name!r} is {'available' if available else 'missing'}.",
        mandatory=mandatory,
    )


def _executable_check(name: str, *, mandatory: bool = False) -> dict[str, Any]:
    path = shutil.which(name)
    return _check(
        f"executable:{name}",
        "PASS" if path else ("FAIL" if mandatory else "SKIP"),
        f"{name}: {path}" if path else f"Optional executable {name!r} is unavailable.",
        mandatory=mandatory,
        detail={"path": path},
    )


def _cache_paths(root: Path) -> list[str]:
    names = {".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__"}
    paths = []
    try:
        for path in root.rglob("*"):
            if path.is_dir() and (path.name in names or path.name.startswith(".pytest_tmp")):
                paths.append(str(path))
    except OSError:
        pass
    return sorted(paths)


def _longest_path(root: Path) -> tuple[int, str]:
    longest = (len(str(root)), str(root))
    try:
        for path in root.rglob("*"):
            candidate = (len(str(path)), str(path))
            if candidate[0] > longest[0]:
                longest = candidate
    except OSError:
        pass
    return longest


def run_developer_doctor(config: ProjectConfig, state: ProjectState) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    repository = repository_state(config.root)
    checks.append(
        _check(
            "repository-state",
            "PASS" if repository["repository_available"] else "FAIL",
            "Repository state is readable."
            if repository["repository_available"]
            else "Repository state is unavailable.",
            mandatory=True,
            detail=repository,
        )
    )
    worktrees = list_worktrees(config.root)
    checks.append(
        _check(
            "worktrees",
            "PASS" if worktrees else "FAIL",
            f"Found {len(worktrees)} registered worktree(s).",
            mandatory=True,
            detail=worktrees,
        )
    )
    checks.extend(
        _executable_check(name, mandatory=name == "git") for name in ("git", "nvidia-smi", "ollama", "ffmpeg")
    )
    checks.extend(_module_check(name, mandatory=name in {"pytest", "ruff"}) for name in ("pytest", "ruff", "mypy"))

    fixture_candidates = [config.root / "tests" / "fixtures", config.path_for("dataset", "raw_inventory")]
    fixtures = [str(path) for path in fixture_candidates if path and path.exists()]
    checks.append(
        _check(
            "fixture-availability",
            "PASS" if fixtures else "WARN",
            f"Found {len(fixtures)} configured fixture location(s).",
            detail={"available": fixtures},
        )
    )

    artifacts = inspect_artifacts(config, state)
    audit_artifacts = [
        item
        for item in artifacts
        if "audit" in item["source"].lower() or "audit" in item["reference"].lower() or item["kind"] == "hashes"
    ]
    audit_issues = [item for item in audit_artifacts if item["identity_status"] not in {"PRESENT", "CURRENT"}]
    checks.append(
        _check(
            "audit-artifact-integrity",
            "FAIL" if audit_issues else "PASS",
            f"Audit artifact inspection found {len(audit_issues)} integrity issue(s).",
            mandatory=bool(audit_issues),
            detail={"issues": audit_issues, "inspected": len(audit_artifacts)},
        )
    )

    longest, longest_path = _longest_path(config.root)
    risk = os.name == "nt" and longest >= 240
    checks.append(
        _check(
            "path-length-risk",
            "WARN" if risk else "PASS",
            f"Longest observed path is {longest} characters.",
            detail={"length": longest, "path": longest_path, "windows": os.name == "nt"},
        )
    )
    caches = _cache_paths(config.root)
    checks.append(
        _check(
            "generated-caches",
            "WARN" if caches else "PASS",
            f"Found {len(caches)} generated cache director{'y' if len(caches) == 1 else 'ies'}.",
            detail={"paths": caches},
        )
    )

    external = os.environ.get("SPRITELAB_EXTERNAL_FIXTURES", "")
    configured_external = [Path(value) for value in external.split(os.pathsep) if value]
    missing_external = [str(path) for path in configured_external if not path.exists()]
    checks.append(
        _check(
            "external-fixtures",
            "WARN" if missing_external else "PASS",
            f"Missing {len(missing_external)} configured external fixture path(s).",
            detail={"configured": [str(path) for path in configured_external], "missing": missing_external},
        )
    )
    return checks
