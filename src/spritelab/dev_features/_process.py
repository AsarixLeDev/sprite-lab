"""Safe process helpers for read-only developer diagnostics."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


def run_process(
    arguments: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    timeout: float | None = 15,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an argument array without invoking a shell."""

    if isinstance(arguments, (str, bytes)):
        raise TypeError("Process arguments must be a sequence, not a shell command string.")
    argv = [os.fspath(argument) for argument in arguments]
    if not argv:
        raise ValueError("Process arguments cannot be empty.")
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        check=False,
        timeout=timeout,
    )
