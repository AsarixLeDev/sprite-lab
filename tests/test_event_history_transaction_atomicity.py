from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import pytest

from spritelab.product_core import ProductEvent, ProductStatus, strict_json_dumps
from spritelab.product_web import events as event_module
from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_FILENAME,
    EVENT_HISTORY_TRANSACTION_FILENAME,
    LEGACY_EVENT_FILENAME,
    LEGACY_MIGRATION_FILENAME,
    EventMigrationState,
    EventRepository,
    LegacyEventMigrationError,
    event_history_transaction_lock,
    record_event_history_origin,
    verify_event_migration,
)
from spritelab.v3.run_state import RUN_SCHEMA, RunState, atomic_write_json


def _event(
    run_id: str,
    *,
    timestamp: str = "2026-07-15T10:00:00+00:00",
    event_type: str = "training_started",
    message: str = "Synthetic transaction test.",
) -> ProductEvent:
    return ProductEvent(
        run_id=run_id,
        timestamp=timestamp,
        feature="training",
        stage="launch",
        event_type=event_type,
        status=ProductStatus.RUNNING,
        current=0,
        total=1,
        message=message,
        metrics={"synthetic": True},
        artifact_references=(),
    )


def _event_line(event: ProductEvent) -> bytes:
    return strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _native_run_state(directory: Path, run_id: str) -> RunState:
    directory.mkdir(parents=True)
    (directory / EVENT_FILENAME).write_bytes(b"")
    atomic_write_json(
        directory / "state.json",
        {
            "schema_version": RUN_SCHEMA,
            "run_id": run_id,
            "command": "training",
            "status": "RUNNING",
            "stage": "launch",
            "resumable": True,
        },
    )
    record_event_history_origin(run_id, directory, expected_origin="native")
    return RunState(directory)


def test_shared_advisory_lock_is_reentrant_and_ignores_stale_sentinel(tmp_path: Path) -> None:
    directory = tmp_path / "lock-run"
    directory.mkdir()
    (directory / ".events.lock").write_bytes(b"stale-owner-metadata")

    with event_history_transaction_lock(directory):
        with event_history_transaction_lock(directory):
            assert (directory / ".events.lock").is_file()


def test_atomic_metadata_barrier_reopens_destination_without_changing_bytes(tmp_path: Path) -> None:
    binary_path = tmp_path / "barrier.bin"
    json_path = tmp_path / "barrier.json"
    event_module._atomic_bytes(binary_path, b"exact durable bytes\n")
    event_module._atomic_json(json_path, {"value": "exact"})

    assert binary_path.read_bytes() == b"exact durable bytes\n"
    assert json_path.read_bytes() == b'{\n  "value": "exact"\n}\n'


def test_replay_of_missing_run_does_not_create_lock_or_run_directory(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    repository = EventRepository(runs)

    assert repository.replay("missing-run").events == ()
    assert not runs.exists()


def test_replay_parses_locked_snapshot_while_later_append_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "locked-replay-snapshot"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    first = _event(run_id)
    second = _event(
        run_id,
        timestamp="2026-07-15T10:01:00+00:00",
        event_type="training_progress",
        message="Later append.",
    )
    repository.append(first)
    snapshot_captured = threading.Event()
    release_snapshot = threading.Event()
    append_finished = threading.Event()
    replay_result: list[object] = []

    def pause_after_snapshot(stage: str, _directory: Path) -> None:
        if stage == "replay_snapshot_captured":
            snapshot_captured.set()
            assert release_snapshot.wait(5)

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", pause_after_snapshot)
    replay_thread = threading.Thread(target=lambda: replay_result.append(repository.replay(run_id)))
    replay_thread.start()
    assert snapshot_captured.wait(5)

    def append_later() -> None:
        repository.append(second)
        append_finished.set()

    append_thread = threading.Thread(target=append_later)
    append_thread.start()
    assert not append_finished.wait(0.1)
    release_snapshot.set()
    replay_thread.join(5)
    append_thread.join(5)

    assert not replay_thread.is_alive()
    assert not append_thread.is_alive()
    assert [indexed.event for indexed in replay_result[0].events] == [first]
    assert [indexed.event for indexed in repository.replay(run_id).events] == [first, second]


def test_product_lock_timeout_returns_controlled_replay_and_append_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "controlled-lock-timeout"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")

    class BusyLock:
        def __enter__(self) -> None:
            raise TimeoutError("synthetic busy lock")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(event_module, "event_history_transaction_lock", lambda _directory: BusyLock())
    replay = repository.replay(run_id)
    assert replay.integrity_status == "NOT_COMPARABLE"
    assert replay.warnings == ("Event history is busy; retry after the active durable write completes.",)
    with pytest.raises(LegacyEventMigrationError, match="Event history is busy"):
        repository.append(_event(run_id))


def test_migration_record_publication_fault_recovers_only_from_live_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "migration-atomic"
    directory = tmp_path / "runs" / run_id
    directory.mkdir(parents=True)
    legacy_bytes = _event_line(_event(run_id))
    (directory / LEGACY_EVENT_FILENAME).write_bytes(legacy_bytes)
    repository = EventRepository(tmp_path / "runs")
    injected = False

    def fail_after_record(stage: str, _directory: Path) -> None:
        nonlocal injected
        if stage == "migration_record_published" and not injected:
            injected = True
            raise OSError("synthetic crash after migration record publication")

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", fail_after_record)
    with pytest.raises(OSError, match="synthetic crash"):
        repository.migrate_legacy_events(run_id)

    assert (directory / EVENT_HISTORY_TRANSACTION_FILENAME).is_file()
    assert (directory / LEGACY_MIGRATION_FILENAME).is_file()
    assert not (directory / EVENT_HISTORY_ORIGIN_FILENAME).exists()
    blocked = verify_event_migration(run_id, directory, migration_required=True)
    assert blocked.state is EventMigrationState.INVALID_RECORD
    assert "controlled recovery" in blocked.message

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", lambda *_args: None)
    record = repository.migrate_legacy_events(run_id)
    assert record is not None
    assert (directory / EVENT_FILENAME).read_bytes() == legacy_bytes
    assert not (directory / EVENT_HISTORY_TRANSACTION_FILENAME).exists()
    assert verify_event_migration(run_id, directory, migration_required=True).migration_verified

    (directory / LEGACY_EVENT_FILENAME).unlink()
    assert (
        verify_event_migration(run_id, directory, migration_required=True).state
        is EventMigrationState.VERIFIED_SOURCE_REMOVED
    )
    (directory / LEGACY_MIGRATION_FILENAME).unlink()
    deleted = verify_event_migration(run_id, directory, migration_required=True)
    assert deleted.state is not EventMigrationState.NO_MIGRATION
    assert not deleted.resume_compatible


def test_event_repository_binding_failure_retries_exact_append_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "repository-append-atomic"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    event = _event(run_id)
    directory = tmp_path / "runs" / run_id
    original_persist = event_module._persist_event_history_origin_bindings
    injected = False

    def fail_binding_once(*args: object, **kwargs: object) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("synthetic binding persistence failure")
        original_persist(*args, **kwargs)

    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", fail_binding_once)
    with pytest.raises(OSError, match="binding persistence"):
        repository.append(event)

    assert (directory / EVENT_FILENAME).read_bytes() == _event_line(event)
    assert (directory / EVENT_HISTORY_TRANSACTION_FILENAME).is_file()
    assert verify_event_migration(run_id, directory).state is EventMigrationState.INVALID_RECORD

    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", original_persist)
    assert repository.append(event) == 1
    assert (directory / EVENT_FILENAME).read_bytes() == _event_line(event)
    assert not (directory / EVENT_HISTORY_TRANSACTION_FILENAME).exists()
    assert repository.state(run_id)["last_durable_event"]["event_id"] == 1
    assert verify_event_migration(run_id, directory).resume_compatible


def test_run_state_binding_failure_retries_original_timestamp_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-state-append-atomic"
    directory = tmp_path / run_id
    run_state = _native_run_state(directory, run_id)
    original_persist = event_module._persist_event_history_origin_bindings
    injected = False

    def fail_binding_once(*args: object, **kwargs: object) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("synthetic run-state binding failure")
        original_persist(*args, **kwargs)

    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", fail_binding_once)
    with pytest.raises(OSError, match="run-state binding"):
        run_state.append_event(
            command="training",
            stage="launch",
            event_type="training_started",
            status="RUNNING",
            message="Synthetic transaction test.",
            total_count=1,
            metrics={"synthetic": True},
        )
    first_bytes = (directory / EVENT_FILENAME).read_bytes()
    assert first_bytes.count(b"\n") == 1
    assert (directory / EVENT_HISTORY_TRANSACTION_FILENAME).is_file()

    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", original_persist)
    run_state.append_event(
        command="training",
        stage="launch",
        event_type="training_started",
        status="RUNNING",
        message="Synthetic transaction test.",
        total_count=1,
        metrics={"synthetic": True},
    )
    assert (directory / EVENT_FILENAME).read_bytes() == first_bytes
    assert not (directory / EVENT_HISTORY_TRANSACTION_FILENAME).exists()
    assert verify_event_migration(run_id, directory).resume_compatible


def test_fresh_repository_replay_recovers_authenticated_event_and_finalizes_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "fresh-replay-recovery"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    event = _event(run_id)
    directory = tmp_path / "runs" / run_id
    original_persist = event_module._persist_event_history_origin_bindings
    injected = False

    def fail_binding_once(*args: object, **kwargs: object) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("synthetic replay recovery failure")
        original_persist(*args, **kwargs)

    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", fail_binding_once)
    with pytest.raises(OSError, match="replay recovery"):
        repository.append(event)
    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", original_persist)

    fresh_repository = EventRepository(tmp_path / "runs")
    replay = fresh_repository.replay(run_id)
    assert replay.integrity_status == "VALID"
    assert [indexed.event for indexed in replay.events] == [event]
    assert fresh_repository.state(run_id)["last_durable_event"] == {
        "event_id": 1,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
    }
    assert not (directory / EVENT_HISTORY_TRANSACTION_FILENAME).exists()


def test_fresh_repository_recovers_old_append_before_accepting_different_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "fresh-repository-different-event"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    first = _event(run_id)
    second = _event(
        run_id,
        timestamp="2026-07-15T10:01:00+00:00",
        event_type="training_progress",
        message="Second synthetic event.",
    )
    original_persist = event_module._persist_event_history_origin_bindings
    injected = False

    def fail_binding_once(*args: object, **kwargs: object) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("synthetic first append failure")
        original_persist(*args, **kwargs)

    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", fail_binding_once)
    with pytest.raises(OSError, match="first append"):
        repository.append(first)
    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", original_persist)

    fresh_repository = EventRepository(tmp_path / "runs")
    assert fresh_repository.append(second) == 2
    assert [indexed.event for indexed in fresh_repository.events(run_id)] == [first, second]
    assert fresh_repository.state(run_id)["last_durable_event"]["event_type"] == second.event_type


def test_fresh_run_state_recovers_old_append_before_accepting_different_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "fresh-run-state-different-event"
    directory = tmp_path / run_id
    run_state = _native_run_state(directory, run_id)
    original_persist = event_module._persist_event_history_origin_bindings
    injected = False

    def fail_binding_once(*args: object, **kwargs: object) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError("synthetic fresh run-state failure")
        original_persist(*args, **kwargs)

    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", fail_binding_once)
    with pytest.raises(OSError, match="fresh run-state"):
        run_state.append_event(
            command="training",
            stage="launch",
            event_type="training_started",
            status="RUNNING",
            message="First synthetic event.",
        )
    monkeypatch.setattr(event_module, "_persist_event_history_origin_bindings", original_persist)

    fresh_run_state = RunState(directory)
    fresh_run_state.append_event(
        command="training",
        stage="launch",
        event_type="training_progress",
        status="RUNNING",
        message="Different synthetic event.",
    )
    rows = [json.loads(line) for line in (directory / EVENT_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == ["training_started", "training_progress"]
    assert not (directory / EVENT_HISTORY_TRANSACTION_FILENAME).exists()


def test_deleted_origin_after_append_intent_is_never_recreated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "deleted-origin-after-intent"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    directory = tmp_path / "runs" / run_id
    event = _event(run_id)

    def fail_before_append(stage: str, _directory: Path) -> None:
        if stage == "append_intent_published":
            raise OSError("synthetic interruption before origin deletion")

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", fail_before_append)
    with pytest.raises(OSError, match="origin deletion"):
        repository.append(event)
    (directory / EVENT_HISTORY_ORIGIN_FILENAME).unlink()
    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", lambda *_args: None)

    fresh_repository = EventRepository(tmp_path / "runs")
    with pytest.raises(LegacyEventMigrationError, match="exact preexisting event-history origin"):
        fresh_repository.append(event)
    assert not (directory / EVENT_HISTORY_ORIGIN_FILENAME).exists()
    assert (directory / EVENT_FILENAME).read_bytes() == b""
    assert (directory / EVENT_HISTORY_TRANSACTION_FILENAME).is_file()


def test_unrelated_current_binding_is_not_healed_during_append_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "tampered-current-binding"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    directory = tmp_path / "runs" / run_id
    event = _event(run_id)

    def fail_before_append(stage: str, _directory: Path) -> None:
        if stage == "append_intent_published":
            raise OSError("synthetic interruption before binding tamper")

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", fail_before_append)
    with pytest.raises(OSError, match="binding tamper"):
        repository.append(event)
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["event_canonical_current_identity_sha256"] = "a" * 64
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", lambda *_args: None)

    with pytest.raises(LegacyEventMigrationError, match="neither the append preimage nor postimage"):
        EventRepository(tmp_path / "runs").append(event)
    assert (directory / EVENT_FILENAME).read_bytes() == b""
    assert (directory / EVENT_HISTORY_TRANSACTION_FILENAME).is_file()


def test_exact_partial_append_is_rolled_back_and_replayed_from_authenticated_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "partial-append-recovery"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    directory = tmp_path / "runs" / run_id
    event = _event(run_id)

    def fail_before_append(stage: str, _directory: Path) -> None:
        if stage == "append_intent_published":
            raise OSError("synthetic crash before partial write")

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", fail_before_append)
    with pytest.raises(OSError, match="partial write"):
        repository.append(event)

    intent = json.loads((directory / EVENT_HISTORY_TRANSACTION_FILENAME).read_text(encoding="utf-8"))
    payload = base64.b64decode(intent["append_payload_base64"], validate=True)
    partial = payload[: max(1, len(payload) // 2)]
    assert len(partial) < len(payload)
    (directory / EVENT_FILENAME).write_bytes(partial)

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", lambda *_args: None)
    assert repository.append(event) == 1
    assert (directory / EVENT_FILENAME).read_bytes() == _event_line(event)
    assert not (directory / EVENT_HISTORY_TRANSACTION_FILENAME).exists()


@pytest.mark.parametrize("surface", ["repository", "run_state"])
def test_tampered_live_append_intent_fails_closed_before_event_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    run_id = f"tampered-intent-{surface}"
    if surface == "repository":
        repository = EventRepository(tmp_path / "runs")
        repository.create_run(run_id, feature="training", command="train")
        directory = tmp_path / "runs" / run_id

        def append() -> None:
            repository.append(_event(run_id))

    else:
        directory = tmp_path / run_id
        run_state = _native_run_state(directory, run_id)

        def append() -> None:
            run_state.append_event(
                command="training",
                stage="launch",
                event_type="training_started",
                status="RUNNING",
                message="Synthetic transaction test.",
                total_count=1,
                metrics={"synthetic": True},
            )

    def fail_before_append(stage: str, _directory: Path) -> None:
        if stage == "append_intent_published":
            raise OSError("synthetic crash before append")

    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", fail_before_append)
    with pytest.raises(OSError, match="before append"):
        append()
    assert (directory / EVENT_FILENAME).read_bytes() == b""

    intent_path = directory / EVENT_HISTORY_TRANSACTION_FILENAME
    value = json.loads(intent_path.read_text(encoding="utf-8"))
    value["canonical_post_sha256"] = "0" * 64
    intent_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(event_module, "_event_transaction_checkpoint", lambda *_args: None)

    with pytest.raises(LegacyEventMigrationError, match="self-hash"):
        append()
    assert (directory / EVENT_FILENAME).read_bytes() == b""
    assert verify_event_migration(run_id, directory).state is EventMigrationState.INVALID_RECORD


def test_malformed_transaction_blocks_read_only_verification_and_append(tmp_path: Path) -> None:
    run_id = "malformed-transaction"
    repository = EventRepository(tmp_path / "runs")
    repository.create_run(run_id, feature="training", command="train")
    directory = tmp_path / "runs" / run_id
    (directory / EVENT_HISTORY_TRANSACTION_FILENAME).write_bytes(b"{malformed")

    verification = verify_event_migration(run_id, directory)
    assert verification.state is EventMigrationState.INVALID_RECORD
    assert not verification.resume_compatible
    with pytest.raises(LegacyEventMigrationError, match="malformed"):
        repository.append(_event(run_id))
    assert (directory / EVENT_FILENAME).read_bytes() == b""
