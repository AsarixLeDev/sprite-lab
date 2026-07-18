from __future__ import annotations

import os
from pathlib import Path

import pytest

from spritelab.utils import pinned_executable as module
from spritelab.utils.pinned_executable import PinnedExecutableError, pinned_git_ls_files

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_pinned_git_inventory_ignores_repository_local_path_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / ("git.exe" if os.name == "nt" else "git")
    shadow.write_bytes(b"must-not-execute")
    monkeypatch.setenv("PATH", str(tmp_path))

    try:
        output = pinned_git_ls_files(
            REPOSITORY_ROOT,
            ("pyproject.toml",),
            timeout_seconds=5.0,
        )
    except PinnedExecutableError as exc:
        if "unavailable" in str(exc):
            pytest.skip(f"fixed system Git is unavailable on this platform: {exc}")
        raise

    assert output == b"pyproject.toml\0"
    assert shadow.read_bytes() == b"must-not-execute"


def test_pinned_git_inventory_rejects_project_local_executable_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / ("git.exe" if os.name == "nt" else "git")
    candidate.write_bytes(b"repository-local executable")
    monkeypatch.setattr(module, "_system_git_candidates", lambda: (candidate,))

    with pytest.raises(PinnedExecutableError, match="unavailable"):
        pinned_git_ls_files(tmp_path, ("tests",), timeout_seconds=1.0)


def test_pinned_git_inventory_checks_operation_before_executable_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = False

    def discover() -> tuple[Path, ...]:
        nonlocal discovered
        discovered = True
        return ()

    def cancelled() -> None:
        raise RuntimeError("cancelled before Git discovery")

    monkeypatch.setattr(module, "_system_git_candidates", discover)
    with pytest.raises(RuntimeError, match="cancelled before Git discovery"):
        pinned_git_ls_files(
            tmp_path,
            ("tests",),
            timeout_seconds=1.0,
            operation_check=cancelled,
        )
    assert discovered is False
