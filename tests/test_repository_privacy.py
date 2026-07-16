from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

BLOCKED_TRACKED_PATHS = {"spritelab.yaml"}
BLOCKED_TRACKED_PREFIXES = ("artifacts/", "experiments/", "out/", "outputs/", "runs/")
TEXT_RULES = {
    "personal identity": re.compile(r"(?i)\b(?:mathieu|mathieu-pc|asarix)\b"),
    "Windows home path": re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"']+"),
    "Unix home path": re.compile(r"/(?:Users|home)/[^/\s\"']+"),
    "recovery path": re.compile(r"(?i)(?:[A-Z]:[\\/])?Recovery[_\\/-]?20\d{2}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential-shaped token": re.compile(
        r"(?:AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|gh[pousr]_[0-9A-Za-z]{20,}|"
        r"hf_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z_-]{20,})"
    ),
    "RunPod credential assignment": re.compile(
        r"(?m)^\s*(?:export\s+|\$env:)?RUNPOD_(?:API_KEY|TOKEN)\s*[:=]\s*[^\s<>$]+"
    ),
}


def _tracked_files() -> list[Path]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("repository privacy test requires a Git checkout")
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode() for value in result.stdout.split(b"\0") if value]


def test_repository_snapshot_contains_no_private_or_machine_generated_data() -> None:
    tracked = _tracked_files()
    relative_paths = [path.relative_to(ROOT).as_posix() for path in tracked]
    blocked = [
        path for path in relative_paths if path in BLOCKED_TRACKED_PATHS or path.startswith(BLOCKED_TRACKED_PREFIXES)
    ]
    findings: list[str] = [f"blocked tracked path: {path}" for path in blocked]

    for path, relative in zip(tracked, relative_paths, strict=True):
        if path.resolve() == THIS_FILE or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in TEXT_RULES.items():
            if pattern.search(text):
                findings.append(f"{relative}: {label}")

    assert findings == [], "Repository privacy findings:\n" + "\n".join(sorted(findings))
