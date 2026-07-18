from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spritelab.training import smoke_bundle

_SMOKE_ID = "smoke-" + "a" * 20


def _payload(marker: Path) -> bytes:
    return (
        "def _spritelab_smoke_preflight(mode, digest, byte_count):\n"
        "    assert mode == 'main'\n"
        "    assert len(digest) == 64\n"
        "    assert byte_count > 0\n"
        "def _spritelab_run_bound(module_name):\n"
        "    assert module_name == 'spritelab'\n"
        f"    open({str(marker)!r}, 'wb').write(b'executed')\n"
    ).encode()


def _install_payload(project: Path, payload: bytes) -> Path:
    target = project / "artifacts" / "training" / "smokes" / _SMOKE_ID / "bootstrap" / "preflight.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    return target


def _command(
    project: Path,
    payload: bytes,
    *,
    digest: str | None = None,
    byte_count: int | None = None,
    source_prefix: str = "",
) -> list[str]:
    windows_prefix = f"import sys;sys._spritelab_windows_project_root={str(project)!r}\n" if os.name == "nt" else ""
    loader = windows_prefix + source_prefix + smoke_bundle._compact_bootstrap_loader("main")
    result = [
        sys.executable,
        "-I",
        "-B",
        "-S",
        "-c",
        loader,
        digest or hashlib.sha256(payload).hexdigest(),
        str(len(payload) if byte_count is None else byte_count),
        "--smoke-bundle-id",
        _SMOKE_ID,
    ]
    if os.name == "nt":
        result.extend(("--smoke-device", "cpu"))
    return result


def _run(project: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    cwd = project
    if os.name == "nt":
        cwd = project / "artifacts" / "training" / "smokes" / _SMOKE_ID / "execution" / "cpu"
        cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_compact_bootstrap_executes_verified_bytes_below_windows_command_limit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    marker = tmp_path / "executed.marker"
    payload = _payload(marker)
    _install_payload(project, payload)
    command = _command(project, payload)

    assert len(subprocess.list2cmdline(command)) < 8_192
    result = _run(project, command)

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert marker.read_bytes() == b"executed"


@pytest.mark.parametrize("mutation", ["wrong_digest", "oversize", "preloaded_project"])
def test_compact_bootstrap_rejects_hostile_delivery_without_execution(tmp_path: Path, mutation: str) -> None:
    project = tmp_path / "project"
    marker = tmp_path / "must-not-execute.marker"
    payload = _payload(marker)
    if mutation == "oversize":
        payload += b"#" * (smoke_bundle._MAX_BOOTSTRAP_BYTES + 1 - len(payload))
    _install_payload(project, payload)
    digest = "0" * 64 if mutation == "wrong_digest" else None
    prefix = (
        "import sys,types;sys.modules['spritelab']=types.ModuleType('spritelab')\n"
        if mutation == "preloaded_project"
        else ""
    )

    result = _run(project, _command(project, payload, digest=digest, source_prefix=prefix))

    assert result.returncode == 70
    assert not marker.exists()


def test_compact_bootstrap_rejects_hard_link_substitution_without_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    marker = tmp_path / "must-not-execute.marker"
    payload = _payload(marker)
    target = _install_payload(project, payload)
    try:
        os.link(target, target.with_name("substitution.py"))
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {type(exc).__name__}")

    result = _run(project, _command(project, payload))

    assert result.returncode == 70
    assert not marker.exists()
