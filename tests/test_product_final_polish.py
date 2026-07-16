from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spritelab.product_core import ProductEvent, ProductStatus
from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_FILENAME,
    LEGACY_EVENT_FILENAME,
    LEGACY_MIGRATION_FILENAME,
    EventRepository,
    LegacyEventMigrationError,
)


def _make_pre_origin_legacy_fixture(directory: Path) -> None:
    (directory / EVENT_FILENAME).unlink()
    (directory / EVENT_HISTORY_ORIGIN_FILENAME).unlink()
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key in tuple(state):
        if (
            key.startswith("event_history_origin")
            or key.startswith("event_migration_")
            or key.startswith("event_canonical_")
        ):
            state.pop(key)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _event(run_id: str, *, event_type: str, current: int) -> ProductEvent:
    return ProductEvent(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        feature="training",
        stage="seed",
        event_type=event_type,
        status=ProductStatus.RUNNING,
        current=current,
        total=100,
        metrics={"seed": 731001, "optimizer_step": current},
    )


def test_legacy_training_event_migration_is_ordered_recorded_and_idempotent(tmp_path: Path) -> None:
    repository = EventRepository(tmp_path)
    run_id = "legacy-training"
    repository.create_run(run_id, feature="training", command="training.start")
    directory = tmp_path / run_id
    _make_pre_origin_legacy_fixture(directory)
    first = _event(run_id, event_type="first", current=1)
    second = _event(run_id, event_type="second", current=2)
    legacy_rows = [
        json.dumps(first.to_dict(), separators=(",", ":")),
        json.dumps(second.to_dict(), separators=(",", ":")),
    ]
    legacy = directory / LEGACY_EVENT_FILENAME
    legacy.write_text("\n".join(legacy_rows) + "\n", encoding="utf-8")

    migrated = repository.migrate_legacy_events(run_id)
    repeated = repository.migrate_legacy_events(run_id)
    canonical = directory / EVENT_FILENAME
    assert migrated == repeated
    assert canonical.read_text(encoding="utf-8").splitlines() == legacy_rows
    assert [item.event.event_type for item in repository.events(run_id)] == ["first", "second"]
    assert json.loads((directory / LEGACY_MIGRATION_FILENAME).read_text(encoding="utf-8"))["validated_event_count"] == 2

    repository.append(_event(run_id, event_type="third", current=3))
    assert len(canonical.read_text(encoding="utf-8").splitlines()) == 3
    assert legacy.read_text(encoding="utf-8").splitlines() == legacy_rows
    assert repository.state(run_id)["event_stream_migration"]["canonical_relative_path"] == EVENT_FILENAME
    assert repository.snapshot(run_id).event_count == 3


def test_malformed_legacy_training_events_fail_without_partial_migration(tmp_path: Path) -> None:
    repository = EventRepository(tmp_path)
    run_id = "malformed-training"
    repository.create_run(run_id, feature="training", command="training.start")
    directory = tmp_path / run_id
    _make_pre_origin_legacy_fixture(directory)
    (directory / LEGACY_EVENT_FILENAME).write_text('{"broken":\n', encoding="utf-8")
    with pytest.raises(LegacyEventMigrationError, match="invalid"):
        repository.append(_event(run_id, event_type="resume", current=3))
    assert not (directory / EVENT_FILENAME).exists()
    assert not (directory / LEGACY_MIGRATION_FILENAME).exists()


def test_product_event_writer_inventory_has_no_legacy_writer() -> None:
    root = Path(__file__).resolve().parents[1]
    active_roots = (
        root / "src/spritelab/product_features",
        root / "src/spritelab/product_web",
        root / "src/spritelab/remote_compute",
    )
    occurrences: list[Path] = []
    for active_root in active_roots:
        for path in active_root.rglob("*.py"):
            if LEGACY_EVENT_FILENAME in path.read_text(encoding="utf-8"):
                occurrences.append(path.relative_to(root))
    assert occurrences == [Path("src/spritelab/product_web/events.py")]


def test_live_product_events_drive_learning_rate_chart_without_mojibake() -> None:
    root = Path(__file__).resolve().parents[1]
    training_script = (root / "src/spritelab/product_features/training/static/training.js").read_text(encoding="utf-8")
    live_projection = training_script.split('source.addEventListener("product"', 1)[1].split(
        "clearInterval(refreshTimer)", 1
    )[0]
    assert "learning_rate_curve" in live_projection
    assert "item.metrics?.learning_rate??item.metrics?.lr" in live_projection

    evaluation_script = (root / "src/spritelab/product_features/evaluation/static/evaluation.js").read_text(
        encoding="utf-8"
    )
    status_line = next(line for line in evaluation_script.splitlines() if '$("play-run-status").textContent' in line)
    assert "\u00c2\u00b7" not in status_line
    assert " | " in status_line
