from __future__ import annotations

import threading
from hashlib import sha256
from pathlib import Path

import pytest

import spritelab.product_features.evaluation.playground as playground_module
from spritelab.product_features.evaluation import (
    CheckpointAvailability,
    CheckpointCandidate,
    CheckpointCatalog,
    GenerationRequest,
    PlaygroundService,
)


def _catalog(root: Path) -> CheckpointCatalog:
    checkpoint = root / "checkpoint_step_100_ema.pt"
    checkpoint.write_bytes(b"terminal-cas-checkpoint")
    candidate = CheckpointCandidate(
        checkpoint_id="terminal-cas-checkpoint",
        run_id="training-run",
        friendly_run_name="terminal CAS fixture",
        date="2026-07-17T00:00:00+00:00",
        training_profile="test",
        completion_state="COMPLETE",
        dataset_identity="dataset-v1",
        dataset_identity_summary="Dataset v1",
        view_identity="view-v1",
        view_identity_summary="View v1",
        checkpoint_step=100,
        weights="ema",
        verification_state="VERIFIED",
        availability=CheckpointAvailability.ELIGIBLE,
        checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
        path=checkpoint,
        run_directory=root,
    )
    return CheckpointCatalog((candidate,), (), candidate.checkpoint_id)


def test_failure_and_cancellation_race_has_one_event_first_terminal_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_started = threading.Event()
    release_adapter = threading.Event()
    race_enabled = threading.Event()
    first_preimage = threading.Event()
    terminal_preimages = threading.Barrier(2)

    def lifecycle_checkpoint(stage: str, _directory: Path) -> None:
        if stage == "expected_snapshot_captured" and race_enabled.is_set():
            first_preimage.set()
            terminal_preimages.wait(timeout=10)

    monkeypatch.setattr(playground_module, "_playground_lifecycle_checkpoint", lifecycle_checkpoint)

    class FailingAdapter:
        remote = False
        billable = False

        def generate(self, **_kwargs):
            adapter_started.set()
            assert release_adapter.wait(timeout=10)
            raise RuntimeError("synthetic terminal race")

    service = PlaygroundService(
        _catalog(tmp_path),
        output_root=tmp_path / "playground",
        generator=FailingAdapter(),
    )
    generation_errors: list[BaseException] = []
    cancellation_errors: list[BaseException] = []

    def generate() -> None:
        try:
            service.generate(
                GenerationRequest(
                    prompt="terminal race shield",
                    checkpoint_id="terminal-cas-checkpoint",
                    image_count=1,
                ),
                explicit_action=True,
            )
        except BaseException as exc:
            generation_errors.append(exc)

    worker = threading.Thread(target=generate)
    worker.start()
    assert adapter_started.wait(timeout=10)
    active = service.latest_run()
    assert active is not None
    run_id = active["run_id"]

    def cancel() -> None:
        try:
            service.cancel(run_id, reason="terminal CAS cancellation")
        except BaseException as exc:
            cancellation_errors.append(exc)

    race_enabled.set()
    canceller = threading.Thread(target=cancel)
    canceller.start()
    assert first_preimage.wait(timeout=10)
    release_adapter.set()
    worker.join(timeout=10)
    canceller.join(timeout=10)

    assert not worker.is_alive()
    assert not canceller.is_alive()
    assert len(generation_errors) == 1
    assert isinstance(generation_errors[0], RuntimeError)
    assert cancellation_errors == []

    replay = service.repository.replay(run_id)
    terminal_events = [
        item.event for item in replay.events if item.event.event_type in {"failed", "cancelled", "timed_out"}
    ]
    assert len(terminal_events) == 1
    assert replay.events[-1].event == terminal_events[0]
    terminal = terminal_events[0]
    terminal_status = {"failed": "FAILED", "cancelled": "CANCELLED"}[terminal.event_type]

    state = service.repository.state(run_id)
    assert state["terminal_status"] == terminal_status
    assert state["last_durable_event"]["event_type"] == terminal.event_type
    assert terminal.metrics["terminal"]["terminal_status"] == terminal_status
    reconstructed = service.reconstruct(run_id)
    assert reconstructed["status"] == terminal_status
    assert reconstructed["comparability"] == "CURRENT"
