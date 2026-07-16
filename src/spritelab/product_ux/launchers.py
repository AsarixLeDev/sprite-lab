"""Safe project-local launcher generation for novice users."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

LAUNCH_COMMAND = "python -m spritelab v3"

_LAUNCHERS = {
    "Start Sprite Lab.cmd": (f'@echo off\nsetlocal\ncd /d "%~dp0"\n{LAUNCH_COMMAND}\nexit /b %ERRORLEVEL%\n'),
    "start-sprite-lab.sh": (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        'SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        'cd "$SCRIPT_DIR"\n'
        f"exec {LAUNCH_COMMAND}\n"
    ),
}


@dataclass(frozen=True)
class LauncherResult:
    path: Path
    status: str
    message: str


def generate_project_launchers(project_root: Path) -> tuple[LauncherResult, ...]:
    """Create both launchers without replacing any existing file.

    ``open('x')`` supplies the no-overwrite guarantee even if another process
    creates a launcher after an initial directory check.
    """

    root = project_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")

    results: list[LauncherResult] = []
    for filename, content in _LAUNCHERS.items():
        target = root / filename
        try:
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
        except FileExistsError:
            results.append(
                LauncherResult(
                    target,
                    "PRESERVED",
                    f"Existing launcher preserved: {filename}",
                )
            )
            continue
        if target.suffix == ".sh":
            os.chmod(target, target.stat().st_mode | stat.S_IXUSR)
        results.append(LauncherResult(target, "CREATED", f"Created project launcher: {filename}"))
    return tuple(results)
