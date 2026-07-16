"""Named developer test profiles represented only as safe argument arrays."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from spritelab.dev_features._process import run_process

TEST_PROFILES = ("quick", "dataset", "labeling", "training", "evaluation", "full")

_PROFILE_PATTERNS: dict[str, tuple[str, ...]] = {
    "quick": ("tests/test_dev_cli_*.py", "tests/test_product_foundation.py"),
    "dataset": ("tests/test_dataset*.py", "tests/test_v3_config_state.py"),
    "labeling": ("tests/test_label_*.py", "tests/test_semantic*.py"),
    "training": ("tests/test_training_*.py",),
    "evaluation": ("tests/test_memorization_*.py", "tests/test_v3_run_report.py"),
    "full": (),
}


@dataclass(frozen=True)
class TestPlan:
    profile: str
    arguments: tuple[str, ...]
    matched_files: tuple[str, ...]

    @property
    def display_command(self) -> str:
        return subprocess.list2cmdline(list(self.arguments))


def build_test_plan(root: Path, profile: str = "quick", extra_arguments: tuple[str, ...] = ()) -> TestPlan:
    if profile not in TEST_PROFILES:
        raise ValueError(f"Unknown test profile: {profile}")
    matched: set[Path] = set()
    for pattern in _PROFILE_PATTERNS[profile]:
        matched.update(path for path in root.glob(pattern) if path.is_file())
    relative = tuple(sorted(path.relative_to(root).as_posix() for path in matched))
    arguments = (sys.executable, "-m", "pytest", *relative, *extra_arguments)
    return TestPlan(profile=profile, arguments=arguments, matched_files=relative)


def execute_test_plan(
    plan: TestPlan,
    *,
    root: Path,
    dry_run: bool,
    announcement: TextIO | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    if announcement is not None:
        print(f"Planned command: {plan.display_command}", file=announcement, flush=True)
    if dry_run:
        return None
    return run_process(plan.arguments, cwd=root, timeout=None, capture_output=capture_output)
