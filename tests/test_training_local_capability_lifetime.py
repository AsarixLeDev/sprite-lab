from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from spritelab.product_core import ProductEvent, ProductStatus, ProjectContext
from spritelab.product_web.events import EVENT_FILENAME
from spritelab.remote_compute import ComputeStatus, LocalComputeBackend
from spritelab.training import campaign as campaign_module
from spritelab.training.campaign import stable_hash
from training_launch_test_utils import compute_request


class _ControlledProcess:
    pid = 4312

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.signals: list[int] = []

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, value: int) -> None:
        self.signals.append(value)

    def terminate(self) -> None:
        self.returncode = -15


@pytest.fixture(autouse=True)
def _stable_training_code_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository_root = Path(campaign_module.__file__).resolve().parents[3]
    records: list[dict[str, str]] = []
    for relative in ("src/spritelab/__init__.py", "src/spritelab/__main__.py"):
        content = (repository_root / relative).read_bytes()
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
    identity: dict[str, Any] = {
        "schema_version": "synthetic_training_code_identity_v1",
        "files": records,
    }
    identity["sha256"] = stable_hash(identity)
    monkeypatch.setattr(campaign_module, "_code_identity", lambda: deepcopy(identity))


def _launch(
    tmp_path: Path,
) -> tuple[LocalComputeBackend, _ControlledProcess, Any, Any, Path]:
    process = _ControlledProcess()
    backend = LocalComputeBackend(process_factory=lambda *_args, **_kwargs: process)
    request = compute_request(tmp_path, "local")
    event_path = request.output_root / EVENT_FILENAME
    request = replace(request, event_path=event_path)
    prepared = backend.prepare(ProjectContext(tmp_path, {}), request)
    job = backend.launch(prepared, request)
    return backend, process, prepared, job, event_path


def _event_bytes(run_id: str, message: str = "retained event") -> bytes:
    event = ProductEvent(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        feature="training",
        stage="seed",
        event_type="progress",
        status=ProductStatus.RUNNING,
        message=message,
    )
    return (json.dumps(event.to_dict(), sort_keys=True) + "\n").encode("utf-8")


def _replace_with_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable in this test session")


def test_local_capability_is_retained_until_terminal_capture_then_cached(tmp_path: Path) -> None:
    backend, process, prepared, job, event_path = _launch(tmp_path)
    content = _event_bytes(job.run_id)
    event_path.write_bytes(content)
    record = backend._jobs[job.job_id]

    events, cursor = backend.stream_events(job)
    assert [item.message for item in events] == ["retained event"]
    assert cursor == 1
    assert record.capability is not None
    assert record.terminal_events_captured is False
    assert str(tmp_path) not in str(job.metadata)
    assert job.metadata["event_filename"] == EVENT_FILENAME

    process.returncode = 0
    poll = backend.poll(job)
    assert poll.status == ComputeStatus.COMPLETE
    assert record.capability is None
    assert record.terminal_events_captured is True
    assert record.cached_event_bytes == content

    retained = event_path.with_name("retained-events.jsonl")
    event_path.replace(retained)
    outside = tmp_path / "outside-terminal-cache"
    outside.mkdir()
    sentinel = outside / "sentinel.jsonl"
    sentinel.write_bytes(b'{"outside":"must-not-be-read"}\n')
    before = sentinel.read_bytes()
    try:
        os.link(sentinel, event_path)
    except (NotImplementedError, OSError):
        pytest.skip("hard links are unavailable in this test session")

    cached, cached_cursor = backend.stream_events(job)
    assert [item.message for item in cached] == ["retained event"]
    assert cached_cursor == 1
    assert sentinel.read_bytes() == before
    assert retained.read_bytes() == content
    assert backend.cleanup(prepared).changed is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX permits the retained output-root rename seam")
def test_live_output_root_symlink_redirect_fails_closed_and_preserves_outside(tmp_path: Path) -> None:
    backend, _process, prepared, job, event_path = _launch(tmp_path)
    content = _event_bytes(job.run_id)
    event_path.write_bytes(content)
    output_root = event_path.parent
    parked = output_root.with_name(f"{output_root.name}-parked")
    outside = tmp_path / "outside-output-redirect"
    outside.mkdir()
    sentinel = outside / EVENT_FILENAME
    sentinel.write_bytes(b'{"outside":"unchanged"}\n')
    before = sentinel.read_bytes()

    os.replace(output_root, parked)
    _replace_with_symlink(output_root, outside, directory=True)
    try:
        with pytest.raises(ValueError, match=r"mutable|directory|substituted"):
            backend.stream_events(job)
        assert sentinel.read_bytes() == before
        assert (parked / EVENT_FILENAME).read_bytes() == content
    finally:
        output_root.unlink()
        os.replace(parked, output_root)

    assert backend.cleanup(prepared).changed is True
    assert event_path.read_bytes() == content


@pytest.mark.skipif(os.name == "nt", reason="symbolic-link creation is not portable on Windows")
def test_live_event_symlink_redirect_is_rejected_without_reading_outside(tmp_path: Path) -> None:
    backend, _process, prepared, job, event_path = _launch(tmp_path)
    content = _event_bytes(job.run_id)
    event_path.write_bytes(content)
    retained = event_path.with_name("retained-events.jsonl")
    event_path.replace(retained)
    outside = tmp_path / "outside-event-symlink"
    outside.mkdir()
    sentinel = outside / "sentinel.jsonl"
    sentinel.write_bytes(b'{"outside":"unchanged"}\n')
    before = sentinel.read_bytes()

    _replace_with_symlink(event_path, sentinel)
    try:
        with pytest.raises(ValueError, match=r"single-link|event stream"):
            backend.stream_events(job)
        assert sentinel.read_bytes() == before
        assert retained.read_bytes() == content
    finally:
        event_path.unlink()
        retained.replace(event_path)

    assert backend.cleanup(prepared).changed is True


def test_live_event_hardlink_redirect_is_rejected_without_reading_outside(tmp_path: Path) -> None:
    backend, _process, prepared, job, event_path = _launch(tmp_path)
    record = backend._jobs[job.job_id]
    content = _event_bytes(job.run_id)
    event_path.write_bytes(content)
    retained = event_path.with_name("retained-events.jsonl")
    event_path.replace(retained)
    outside = tmp_path / "outside-event-hardlink"
    outside.mkdir()
    sentinel = outside / "sentinel.jsonl"
    sentinel.write_bytes(b'{"outside":"unchanged"}\n')
    before = sentinel.read_bytes()
    try:
        os.link(sentinel, event_path)
    except (NotImplementedError, OSError):
        pytest.skip("hard links are unavailable in this test session")

    try:
        with pytest.raises(ValueError, match=r"single-link|event stream"):
            backend.stream_events(job)
        result = backend.cleanup(prepared)
        assert result.changed is True
        assert record.capability is None
        assert record.released_by_cleanup is True
        assert record.monitoring_uncertain is True
        assert sentinel.read_bytes() == before
        assert retained.read_bytes() == content
    finally:
        event_path.unlink()
        retained.replace(event_path)

    assert backend.poll(job).status == ComputeStatus.UNCERTAIN


def test_cleanup_releases_live_capability_without_deleting_artifacts(tmp_path: Path) -> None:
    backend, _process, prepared, job, event_path = _launch(tmp_path)
    content = _event_bytes(job.run_id)
    event_path.write_bytes(content)
    record = backend._jobs[job.job_id]

    result = backend.cleanup(prepared)

    assert result.changed is True
    assert record.capability is None
    assert record.released_by_cleanup is True
    assert record.terminal_events_captured is False
    assert event_path.read_bytes() == content
    assert backend.poll(job).status == ComputeStatus.UNCERTAIN
    cached, cursor = backend.stream_events(job)
    assert [item.message for item in cached] == ["retained event"]
    assert cursor == 1
