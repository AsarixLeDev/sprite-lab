from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

from spritelab.product_web.events import (
    EVENT_HISTORY_ORIGIN_FILENAME,
    EVENT_HISTORY_ORIGIN_NATIVE,
    EventMigrationState,
    verify_event_migration,
)
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import AuditStatus, Evidence, ProjectState, StageState, StageStatus
from spritelab.v3.report import generate_report, latest_report, open_report
from spritelab.v3.run_state import RunState, atomic_write_json, list_runs, resumable_runs


def _config(tmp_path: Path) -> ProjectConfig:
    values = json.loads(json.dumps(DEFAULT_CONFIG))
    values["paths"]["runs"] = "runs with spaces/über"
    return ProjectConfig(root=tmp_path, path=tmp_path / "spritelab.yaml", values=values)


def _state(tmp_path: Path, *, metrics: dict | None = None) -> ProjectState:
    return ProjectState(
        project_name="Synthetic Ω",
        project_root=tmp_path,
        config_path=tmp_path / "spritelab.yaml",
        source_commit="abc123",
        generated_at="2026-07-13T00:00:00+00:00",
        stages=[
            StageState(
                key="synthetic",
                title="Synthetic stage",
                status=StageStatus.BLOCKED,
                explanation="Waiting safely.",
                blockers=["A gate is closed."],
                warnings=["Synthetic warning."],
                evidence=[Evidence(path=str(tmp_path / "a b.json"), sha256="0" * 64)],
                audit=AuditStatus.FAIL,
                metrics=metrics or {},
            )
        ],
    )


def test_atomic_state_write_replaces_complete_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2, "unicode": "✓"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2, "unicode": "✓"}
    assert not list(tmp_path.glob("*.tmp"))


def test_run_creates_required_layout_and_append_only_events(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = RunState.create(config, command="train", argv=["train", "--dry-run"], source_commit="abc", dry_run=True)
    run.append_event(command="train", stage="plan", event_type="checked", status="READY", message="One")
    before = run.events_path.read_text(encoding="utf-8").splitlines()
    run.append_event(command="train", stage="plan", event_type="checked", status="READY", message="Two")
    after = run.events_path.read_text(encoding="utf-8").splitlines()
    assert after[: len(before)] == before
    assert len(after) == len(before) + 1
    for name in ("state.json", "events.jsonl", "command.json", "logs", "artifacts", "report", "checkpoints"):
        assert (run.directory / name).exists()
    assert (run.directory / EVENT_HISTORY_ORIGIN_FILENAME).is_file()
    verification = verify_event_migration(run.run_id, run.directory, origin_required=True)
    assert verification.state == EventMigrationState.NO_MIGRATION
    assert verification.event_history_origin == EVENT_HISTORY_ORIGIN_NATIVE
    assert verification.resume_compatible is True
    state = run.read_state()
    assert state["event_history_origin"] == EVENT_HISTORY_ORIGIN_NATIVE
    assert state["event_migration_required"] is False


def test_interrupted_run_is_resumable_but_completed_run_is_not(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paused = RunState.create(config, command="train", argv=[], source_commit="abc", dry_run=False)
    paused.finish(command="train", status="PAUSED", exit_code=5, message="paused", stage="campaign", resumable=True)
    complete = RunState.create(config, command="eval", argv=[], source_commit="abc", dry_run=True)
    complete.finish(command="eval", status="COMPLETE", exit_code=0, message="done", stage="plan", resumable=False)
    assert [item["run_id"] for item in resumable_runs(config.runs_dir)] == [paused.run_id]
    assert len(list_runs(config.runs_dir)) == 2


def test_report_is_offline_and_json_matches_state(tmp_path: Path) -> None:
    state = _state(tmp_path, metrics={"curve": [1, 2, 3]})
    index, report_json = generate_report(state, tmp_path / "report")
    html = index.read_text(encoding="utf-8")
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html
    assert "Synthetic Ω" in html
    assert payload["project_state"] == state.to_dict()


def test_report_shows_no_data_yet(tmp_path: Path) -> None:
    index, _ = generate_report(_state(tmp_path), tmp_path / "report")
    assert "No data yet" in index.read_text(encoding="utf-8")


def test_latest_report_uses_latest_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = RunState.create(config, command="train", argv=[], source_commit="abc", dry_run=True)
    generate_report(_state(tmp_path), first.directory / "report")
    first.finish(command="train", status="COMPLETE", exit_code=0, message="done", stage="plan")
    second = RunState.create(config, command="eval", argv=[], source_commit="abc", dry_run=True)
    generate_report(_state(tmp_path), second.directory / "report")
    second.finish(command="eval", status="COMPLETE", exit_code=0, message="done", stage="plan")
    assert latest_report(config.runs_dir) == second.directory / "report" / "index.html"


def test_browser_launch_is_mockable(monkeypatch, tmp_path: Path) -> None:
    called: list[str] = []
    monkeypatch.setattr("spritelab.v3.report.webbrowser.open", lambda uri: called.append(uri) or True)
    path = tmp_path / "report space/index.html"
    path.parent.mkdir()
    path.write_text("ok", encoding="utf-8")
    assert open_report(path) is True
    assert called[0].startswith("file:") and "%20" in called[0]


def test_windows_paths_are_not_string_concatenated() -> None:
    root = PureWindowsPath("C:/Projects/Sprite Lab")
    assert root / "runs" / "v3" == PureWindowsPath("C:/Projects/Sprite Lab/runs/v3")
    assert str(root / "runs").startswith("C:\\")
