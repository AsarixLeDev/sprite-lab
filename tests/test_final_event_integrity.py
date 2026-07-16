from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from spritelab.product_core import (
    ProductEvent,
    ProductEventValidationError,
    ProductStatus,
    ProjectContext,
    api_error,
    strict_json_dumps,
)
from spritelab.product_features.training.dashboard import DashboardState
from spritelab.product_web.app import create_app
from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_NATIVE,
    LEGACY_EVENT_FILENAME,
    LEGACY_MIGRATION_FILENAME,
    LEGACY_MIGRATION_SCHEMA,
    EventRepository,
    LegacyEventMigrationError,
    record_event_history_origin,
)


def _event(run_id: str, **overrides: Any) -> ProductEvent:
    values: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": "2026-07-14T10:00:00+00:00",
        "feature": "training",
        "stage": "seed",
        "event_type": "progress",
        "status": ProductStatus.RUNNING,
        "current": 1,
        "total": 3,
        "message": "Progress recorded.",
        "metrics": {"seed": 42, "optimizer_step": 1, "loss": 0.5},
    }
    values.update(overrides)
    return ProductEvent(**values)


def _strict_parse(value: str) -> Any:
    return json.loads(value, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_product_event_rejects_direct_non_finite_metrics(value: float) -> None:
    with pytest.raises(ProductEventValidationError) as captured:
        _event("finite-direct", metrics={"loss": value})
    assert captured.value.code == "non_finite_number"
    assert captured.value.path == "$.metrics.loss"


def test_product_event_rejects_nested_non_finite_metric() -> None:
    with pytest.raises(ProductEventValidationError, match="Non-finite"):
        _event("finite-nested", metrics={"progress": {"points": [0.1, {"eta": float("inf")}]}})


def test_append_revalidates_mutated_product_event_before_write(tmp_path: Path) -> None:
    metrics: dict[str, Any] = {"loss": 0.5}
    event = _event("append-revalidation", metrics=metrics)
    metrics["loss"] = float("nan")
    repository = EventRepository(tmp_path)
    with pytest.raises(ProductEventValidationError):
        repository.append(event)
    path = tmp_path / event.run_id / "events.jsonl"
    assert not path.exists()


def test_replay_skips_non_finite_legacy_row_and_preserves_neighbors(tmp_path: Path) -> None:
    run_id = "invalid-row-replay"
    directory = tmp_path / run_id
    directory.mkdir()
    first = _event(run_id, current=1, metrics={"loss": 0.5})
    last = _event(run_id, current=3, status=ProductStatus.COMPLETE, metrics={"loss": 0.25})
    invalid = first.to_dict()
    invalid["current"] = 2
    invalid["metrics"] = {"loss": float("inf")}
    content = (
        strict_json_dumps(first.to_dict())
        + "\n"
        + json.dumps(invalid)
        + "\n"
        + strict_json_dumps(last.to_dict())
        + "\n"
    )
    (directory / "events.jsonl").write_text(content, encoding="utf-8")
    record_event_history_origin(run_id, directory, expected_origin=EVENT_HISTORY_ORIGIN_NATIVE)

    replay = EventRepository(tmp_path).replay(run_id)
    assert [item.event_id for item in replay.events] == [1, 3]
    assert [item.event.metrics["loss"] for item in replay.events] == [0.5, 0.25]
    assert replay.invalid_event_count == 1
    assert replay.warnings == ("1 invalid product event row(s) were ignored.",)


def test_sse_payloads_are_strict_browser_json_and_warn_about_invalid_rows(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    run_id = "strict-sse"
    repository = EventRepository(runs)
    repository.create_run(run_id, feature="training", command="training", status="COMPLETE")
    directory = runs / run_id
    valid = _event(run_id, status=ProductStatus.COMPLETE)
    (directory / "events.jsonl").write_text(
        strict_json_dumps(valid.to_dict()) + "\n" + '{"metrics":{"loss":Infinity}}\n',
        encoding="utf-8",
    )
    context = ProjectContext(tmp_path, {}, runs_directory=runs)
    response = TestClient(create_app(context, event_poll_interval=0.01)).get(f"/api/runs/{run_id}/events?once=true")
    assert response.status_code == 200
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert payloads
    decoded = [_strict_parse(payload) for payload in payloads]
    assert all("NaN" not in payload and "Infinity" not in payload for payload in payloads)
    assert any(item.get("invalid_event_count") == 1 for item in decoded)


def test_api_error_omits_invalid_details_and_remains_strict_json() -> None:
    response = api_error(
        422,
        "invalid_metric",
        "Metric validation failed.",
        details={"loss": float("nan")},
        include_details=True,
    )
    text = response.body.decode("utf-8")
    payload = _strict_parse(text)
    assert payload["error_code"] == "invalid_metric"
    assert "details" not in payload
    assert "NaN" not in text and "Infinity" not in text


def test_state_json_rejects_non_finite_updates_without_invalid_tokens(tmp_path: Path) -> None:
    repository = EventRepository(tmp_path)
    repository.create_run("strict-state", feature="training", command="training")
    state_path = tmp_path / "strict-state" / "state.json"
    before = state_path.read_bytes()
    with pytest.raises(ValueError):
        repository.update_state("strict-state", progress={"eta": float("-inf")})
    assert state_path.read_bytes() == before
    assert b"NaN" not in before and b"Infinity" not in before
    _strict_parse(before.decode("utf-8"))


def test_learning_rate_chart_keeps_missing_point_as_gap_not_zero() -> None:
    dashboard = DashboardState("learning-rate-gap", "local")
    dashboard.apply(
        _event(
            dashboard.run_id,
            current=1,
            metrics={"seed": 42, "optimizer_step": 1, "learning_rate": 0.001},
        )
    )
    dashboard.apply(_event(dashboard.run_id, current=2, metrics={"seed": 42, "optimizer_step": 2, "loss": 0.4}))
    dashboard.apply(
        _event(
            dashboard.run_id,
            current=3,
            metrics={"seed": 42, "optimizer_step": 3, "learning_rate": 0.0005},
        )
    )
    assert [point["value"] for point in dashboard.learning_rate_curve] == [0.001, None, 0.0005]
    assert all(point["value"] != 0 for point in dashboard.learning_rate_curve)
    strict_json_dumps(dashboard.to_dict())


def _legacy_rows(run_id: str) -> tuple[bytes, bytes, bytes]:
    rows = tuple(
        strict_json_dumps(
            _event(
                run_id,
                current=index,
                status=ProductStatus.COMPLETE if index == 3 else ProductStatus.RUNNING,
                metrics={"loss": round(1 / index, 4)},
            ).to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        for index in (1, 2, 3)
    )
    return rows  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("source", "summary", "terminal"),
    [
        (lambda rows: b"\n".join(rows) + b"\n", {"lf": 3, "crlf": 0, "unterminated": 0}, True),
        (lambda rows: b"\r\n".join(rows) + b"\r\n", {"lf": 0, "crlf": 3, "unterminated": 0}, True),
        (
            lambda rows: rows[0] + b"\r\n" + rows[1] + b"\n" + rows[2] + b"\r\n",
            {"lf": 1, "crlf": 2, "unterminated": 0},
            True,
        ),
        (lambda rows: b"\n".join(rows), {"lf": 2, "crlf": 0, "unterminated": 1}, False),
    ],
)
def test_legacy_migration_preserves_exact_bytes_and_line_endings(
    tmp_path: Path,
    source: Any,
    summary: dict[str, int],
    terminal: bool,
) -> None:
    run_id = "byte-exact-migration"
    directory = tmp_path / run_id
    directory.mkdir()
    source_bytes = source(_legacy_rows(run_id))
    legacy = directory / LEGACY_EVENT_FILENAME
    legacy.write_bytes(source_bytes)
    repository = EventRepository(tmp_path)

    record = repository.migrate_legacy_events(run_id)

    assert record is not None
    assert record["schema_version"] == LEGACY_MIGRATION_SCHEMA
    assert record["legacy_relative_path"] == LEGACY_EVENT_FILENAME
    assert record["canonical_relative_path"] == EVENT_FILENAME
    assert record["validated_event_count"] == 3
    assert record["line_ending_summary"] == summary
    assert record["had_terminal_newline"] is terminal
    assert record["legacy_size_bytes"] == len(source_bytes)
    assert record["canonical_prefix_size_bytes"] == len(source_bytes)
    assert record["legacy_sha256"] == record["canonical_prefix_sha256"]
    assert legacy.read_bytes() == source_bytes
    assert (directory / EVENT_FILENAME).read_bytes() == source_bytes
    assert [item.event.current for item in repository.replay(run_id).events] == [1, 2, 3]


def test_append_after_unterminated_migration_preserves_prefix_and_adds_one_separator(tmp_path: Path) -> None:
    run_id = "unterminated-append"
    directory = tmp_path / run_id
    directory.mkdir()
    source = b"\r\n".join(_legacy_rows(run_id))
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    repository = EventRepository(tmp_path)
    repository.migrate_legacy_events(run_id)

    event_id = repository.append(_event(run_id, current=4, status=ProductStatus.COMPLETE))

    canonical = (directory / EVENT_FILENAME).read_bytes()
    assert canonical.startswith(source)
    assert canonical[len(source) : len(source) + 1] == b"\n"
    assert not canonical.startswith(source + b"\n\n")
    assert event_id == 4
    assert len(repository.replay(run_id).events) == 4


def test_repeated_migration_is_idempotent_and_does_not_duplicate_rows(tmp_path: Path) -> None:
    run_id = "idempotent-migration"
    directory = tmp_path / run_id
    directory.mkdir()
    source = b"\n".join(_legacy_rows(run_id)) + b"\n"
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    repository = EventRepository(tmp_path)
    first = repository.migrate_legacy_events(run_id)
    canonical_before = (directory / EVENT_FILENAME).read_bytes()
    record_before = (directory / LEGACY_MIGRATION_FILENAME).read_bytes()

    second = repository.migrate_legacy_events(run_id)

    assert second == first
    assert (directory / EVENT_FILENAME).read_bytes() == canonical_before == source
    assert (directory / LEGACY_MIGRATION_FILENAME).read_bytes() == record_before
    assert len(repository.replay(run_id).events) == 3


def test_changed_legacy_bytes_after_migration_are_stale_and_block_append(tmp_path: Path) -> None:
    run_id = "changed-legacy"
    directory = tmp_path / run_id
    directory.mkdir()
    source = b"\n".join(_legacy_rows(run_id)) + b"\n"
    legacy = directory / LEGACY_EVENT_FILENAME
    legacy.write_bytes(source)
    repository = EventRepository(tmp_path)
    repository.migrate_legacy_events(run_id)
    canonical_before = (directory / EVENT_FILENAME).read_bytes()
    legacy.write_bytes(source.replace(b'"feature":"training"', b'"feature":"draining"', 1))

    replay = repository.replay(run_id)

    assert replay.integrity_status == "STALE"
    assert not replay.safe_for_resume
    assert (directory / EVENT_FILENAME).read_bytes() == canonical_before
    with pytest.raises(LegacyEventMigrationError) as captured:
        repository.append(_event(run_id, current=4))
    assert captured.value.status == "STALE"


def test_changed_canonical_prefix_is_not_comparable(tmp_path: Path) -> None:
    run_id = "changed-prefix"
    directory = tmp_path / run_id
    directory.mkdir()
    source = b"\n".join(_legacy_rows(run_id)) + b"\n"
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    repository = EventRepository(tmp_path)
    repository.migrate_legacy_events(run_id)
    canonical = directory / EVENT_FILENAME
    canonical.write_bytes(source.replace(b'"feature":"training"', b'"feature":"draining"', 1))

    replay = repository.replay(run_id)

    assert replay.integrity_status == "NOT_COMPARABLE"
    assert any("prefix hash changed" in warning for warning in replay.warnings)
    with pytest.raises(LegacyEventMigrationError):
        repository.migrate_legacy_events(run_id)


def test_missing_record_is_recreated_only_when_prefix_relationship_is_provable(tmp_path: Path) -> None:
    run_id = "missing-record"
    directory = tmp_path / run_id
    directory.mkdir()
    rows = _legacy_rows(run_id)
    source = b"\n".join(rows)
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    (directory / EVENT_FILENAME).write_bytes(source + b"\n" + rows[-1] + b"\n")
    repository = EventRepository(tmp_path)

    record = repository.migrate_legacy_events(run_id)

    assert record is not None and record["migration_status"] == "reconciled"
    assert (directory / EVENT_FILENAME).read_bytes().startswith(source)


def test_malformed_migration_record_is_not_silently_repaired(tmp_path: Path) -> None:
    run_id = "malformed-record"
    directory = tmp_path / run_id
    directory.mkdir()
    source = b"\n".join(_legacy_rows(run_id)) + b"\n"
    (directory / LEGACY_EVENT_FILENAME).write_bytes(source)
    (directory / EVENT_FILENAME).write_bytes(source)
    record_path = directory / LEGACY_MIGRATION_FILENAME
    record_path.write_bytes(b'{"schema_version":')

    replay = EventRepository(tmp_path).replay(run_id)

    assert replay.integrity_status == "NOT_COMPARABLE"
    assert record_path.read_bytes() == b'{"schema_version":'


def test_conflicting_streams_without_record_are_not_merged(tmp_path: Path) -> None:
    run_id = "conflicting-streams"
    directory = tmp_path / run_id
    directory.mkdir()
    rows = _legacy_rows(run_id)
    (directory / LEGACY_EVENT_FILENAME).write_bytes(rows[0] + b"\n")
    (directory / EVENT_FILENAME).write_bytes(rows[1] + b"\n")

    with pytest.raises(LegacyEventMigrationError, match="conflict"):
        EventRepository(tmp_path).migrate_legacy_events(run_id)
    assert not (directory / LEGACY_MIGRATION_FILENAME).exists()


@pytest.mark.parametrize(
    "invalid_source",
    [
        pytest.param(b"{not-json}\n", id="malformed-json"),
        pytest.param(b"[]\n", id="non-object"),
        pytest.param(
            b'{"schema_version":"spritelab.product.event.v1","metrics":{"loss":NaN}}\n',
            id="non-finite",
        ),
        pytest.param(b"\xff\xfe\n", id="invalid-utf8"),
        pytest.param(b" " * 1_000_001, id="oversized-row"),
    ],
)
def test_invalid_legacy_rows_are_rejected_without_source_mutation(tmp_path: Path, invalid_source: bytes) -> None:
    run_id = "invalid-legacy"
    directory = tmp_path / run_id
    directory.mkdir()
    legacy = directory / LEGACY_EVENT_FILENAME
    legacy.write_bytes(invalid_source)

    with pytest.raises(LegacyEventMigrationError):
        EventRepository(tmp_path).migrate_legacy_events(run_id)

    assert legacy.read_bytes() == invalid_source
    assert not (directory / EVENT_FILENAME).exists()
    assert not (directory / LEGACY_MIGRATION_FILENAME).exists()


def test_verified_migration_survives_deliberate_legacy_source_removal(tmp_path: Path) -> None:
    run_id = "removed-legacy-source"
    directory = tmp_path / run_id
    directory.mkdir()
    source = b"\n".join(_legacy_rows(run_id)) + b"\n"
    legacy = directory / LEGACY_EVENT_FILENAME
    legacy.write_bytes(source)
    repository = EventRepository(tmp_path)
    record = repository.migrate_legacy_events(run_id)
    legacy.unlink()

    replay = repository.replay(run_id)

    assert replay.integrity_status == "VALID"
    assert replay.migration == record
    assert len(replay.events) == 3
