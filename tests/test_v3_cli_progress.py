from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from spritelab import __main__ as package_main
from spritelab.product_core import ProductEvent, ProductStatus, strict_json_dumps
from spritelab.product_web.events import (
    EVENT_FILENAME,
    EVENT_HISTORY_ORIGIN_FILENAME,
    LEGACY_EVENT_FILENAME,
    LEGACY_MIGRATION_FILENAME,
    EventMigrationState,
    EventRepository,
)
from spritelab.v3 import cli
from spritelab.v3.config import DEFAULT_CONFIG, ProjectConfig
from spritelab.v3.model import ExitCode
from spritelab.v3.progress import ProgressSnapshot, render_progress
from spritelab.v3.run_state import RunState, atomic_write_json


def _config(tmp_path: Path) -> None:
    (tmp_path / "spritelab.yaml").write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")


def _project_config(tmp_path: Path) -> ProjectConfig:
    values = json.loads(json.dumps(DEFAULT_CONFIG))
    return ProjectConfig(root=tmp_path, path=tmp_path / "spritelab.yaml", values=values)


def _resume_result(monkeypatch, config: ProjectConfig, run_id: str):
    monkeypatch.setattr(cli, "_load", lambda: config)
    monkeypatch.setattr(cli, "build_project_state", lambda _config: SimpleNamespace(source_commit="abc"))
    return cli._handle_resume(argparse.Namespace(run_id=run_id, dry_run=True), [])


def _training_backend_identity() -> dict[str, str]:
    return {
        "dataset_identity": "a" * 64,
        "view_identity": "b" * 64,
        "training_view_identity": "b" * 64,
        "product_training_run_id": "product-training-run",
    }


def _migrated_resumable_run(config: ProjectConfig, run_id: str) -> Path:
    directory = config.runs_dir / run_id
    directory.mkdir(parents=True)
    atomic_write_json(
        directory / "state.json",
        {
            "schema_version": "spritelab.v3.run-state.v1",
            "run_id": run_id,
            "command": "train",
            "status": "PAUSED",
            "stage": "campaign",
            "started_at": "2026-07-14T00:00:00+00:00",
            "resumable": True,
            "source_commit": "abc",
            "backend_identity": _training_backend_identity(),
        },
    )
    event = ProductEvent(
        run_id=run_id,
        timestamp="2026-07-14T00:00:00+00:00",
        feature="training",
        stage="campaign",
        event_type="progress",
        status=ProductStatus.PAUSED,
        message="Synthetic migrated v3 run.",
    )
    (directory / LEGACY_EVENT_FILENAME).write_bytes(
        strict_json_dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    EventRepository(config.runs_dir).migrate_legacy_events(run_id)
    (directory / LEGACY_EVENT_FILENAME).unlink()
    return directory


def _invoke(args: list[str], capsys) -> tuple[int, str, str]:
    with pytest.raises(SystemExit) as caught:
        cli.main(args)
    captured = capsys.readouterr()
    return int(caught.value.code), captured.out, captured.err


def test_python_module_help_works_from_source_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-m", "spritelab", "v3", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "dataset" in result.stdout and "train" in result.stdout and "eval" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["dataset", "--help"],
        ["dataset", "build", "--help"],
        ["train", "--help"],
        ["eval", "--help"],
        ["init", "--help"],
        ["status", "--help"],
        ["doctor", "--help"],
        ["resume", "--help"],
        ["review", "--help"],
        ["report", "--help"],
        ["runs", "--help"],
        ["logs", "--help"],
        ["explain", "--help"],
    ],
)
def test_every_command_has_help(args: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(args)
    assert caught.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_existing_low_level_commands_remain_registered() -> None:
    assert {"curation", "train", "training", "dataset-maker", "harvest", "ml", "eval", "v3"} <= set(
        package_main._COMMANDS
    )


def test_init_dry_run_writes_nothing(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "discover_config", lambda _start: None)
    monkeypatch.setattr(cli, "_project_root_for_init", lambda: tmp_path)
    code, output, _ = _invoke(["init", "--dry-run", "--json"], capsys)
    assert code == 0
    assert json.loads(output)["data"]["created"] is False
    assert not (tmp_path / "spritelab.yaml").exists()


def test_init_creates_once_and_never_overwrites(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "spritelab.yaml"
    monkeypatch.setattr(cli, "discover_config", lambda _start: target if target.exists() else None)
    monkeypatch.setattr(cli, "_project_root_for_init", lambda: tmp_path)
    code, output, _ = _invoke(["init", "--json"], capsys)
    assert code == 0
    first = (tmp_path / "spritelab.yaml").read_bytes()
    assert json.loads(output)["data"]["overwritten"] is False
    code, output, _ = _invoke(["init", "--json"], capsys)
    assert code == 2
    assert (tmp_path / "spritelab.yaml").read_bytes() == first
    assert json.loads(output)["data"]["overwritten"] is False


def test_status_json_schema_has_no_ansi(monkeypatch, tmp_path: Path, capsys) -> None:
    _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    code, output, _ = _invoke(["status", "--json", "--no-color"], capsys)
    payload = json.loads(output)
    assert code == payload["exit_code"] == 0
    assert payload["schema_version"] == "spritelab.v3.result.v1"
    assert "\x1b[" not in output
    assert "evidence" in payload["project_state"]["stages"][0]


def test_status_human_blockers_include_exact_next_command(monkeypatch, tmp_path: Path, capsys) -> None:
    _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    code, output, _ = _invoke(["status", "--no-color"], capsys)
    assert code == 0
    assert "Pipeline status" in output
    assert "Next:" in output
    assert "python -m spritelab v3" in output


def test_internal_traceback_hidden_without_debug(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_handle_status", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    code, output, _ = _invoke(["status"], capsys)
    assert code == 1
    assert "Traceback" not in output


def test_internal_traceback_shown_with_debug(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_handle_status", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    code, output, _ = _invoke(["status", "--debug"], capsys)
    assert code == 1
    assert "Traceback" in output and "RuntimeError: boom" in output


def test_report_browser_launch_is_mocked(monkeypatch, tmp_path: Path, capsys) -> None:
    _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "open_report", lambda _: True)
    code, output, _ = _invoke(["report", "--open", "--json"], capsys)
    assert code == 0
    assert json.loads(output)["data"]["opened"] is True


@pytest.mark.parametrize("mutation", ["delete_origin", "tamper_canonical"])
def test_resume_rejects_invalid_native_event_history_without_backend_handoff(
    monkeypatch, tmp_path: Path, mutation: str
) -> None:
    config = _project_config(tmp_path)
    run = RunState.create(config, command="train", argv=[], source_commit="abc", dry_run=False)
    run.finish(command="train", status="PAUSED", exit_code=5, message="paused", stage="campaign", resumable=True)
    if mutation == "delete_origin":
        (run.directory / EVENT_HISTORY_ORIGIN_FILENAME).unlink()
    else:
        canonical = run.directory / EVENT_FILENAME
        canonical.write_bytes(canonical.read_bytes() + b"tampered")

    result = _resume_result(monkeypatch, config, run.run_id)

    assert result.status == "BLOCKED"
    assert result.exit_code == ExitCode.INVALID
    assert result.data["backend_launches"] == 0
    assert result.data["event_migration_state"] != EventMigrationState.NO_MIGRATION.value


@pytest.mark.parametrize("mutation", ["delete_migration_record", "tamper_canonical_prefix"])
def test_resume_rejects_invalid_migrated_event_history_without_backend_handoff(
    monkeypatch, tmp_path: Path, mutation: str
) -> None:
    config = _project_config(tmp_path)
    run_id = f"migrated-v3-{mutation}"
    directory = _migrated_resumable_run(config, run_id)
    if mutation == "delete_migration_record":
        (directory / LEGACY_MIGRATION_FILENAME).unlink()
    else:
        canonical = directory / EVENT_FILENAME
        canonical_bytes = canonical.read_bytes()
        canonical.write_bytes(b"X" + canonical_bytes[1:])

    result = _resume_result(monkeypatch, config, run_id)

    assert result.status == "BLOCKED"
    assert result.exit_code == ExitCode.INVALID
    assert result.data["backend_launches"] == 0
    assert result.data["event_migration_state"] != EventMigrationState.NO_MIGRATION.value


@pytest.mark.parametrize("origin", ["native", "migrated_legacy"])
def test_resume_accepts_valid_authoritative_event_history_in_dry_run(monkeypatch, tmp_path: Path, origin: str) -> None:
    config = _project_config(tmp_path)
    if origin == "native":
        run = RunState.create(config, command="train", argv=[], source_commit="abc", dry_run=False)
        run.finish(
            command="train",
            status="PAUSED",
            exit_code=5,
            message="paused",
            stage="campaign",
            resumable=True,
            backend_identity=_training_backend_identity(),
        )
        run_id = run.run_id
    else:
        run_id = "migrated-v3-valid"
        _migrated_resumable_run(config, run_id)

    result = _resume_result(monkeypatch, config, run_id)

    assert result.status == "COMPLETE"
    assert result.exit_code == ExitCode.SUCCESS
    assert result.data["backend_launches"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_backend",
        "missing_dataset",
        "numeric_dataset",
        "boolean_view",
        "list_view",
        "padded_view",
        "disagreeing_view",
        "missing_product_run",
    ],
)
def test_v3_training_resume_rejects_incomplete_or_malformed_identity_projection(
    monkeypatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    config = _project_config(tmp_path)
    run = RunState.create(config, command="train", argv=[], source_commit="abc", dry_run=False)
    run.finish(
        command="train",
        status="PAUSED",
        exit_code=5,
        message="paused",
        stage="campaign",
        resumable=True,
        backend_identity=_training_backend_identity(),
    )
    state = run.read_state()
    backend = state["backend_identity"]
    if mutation == "missing_backend":
        state.pop("backend_identity")
    elif mutation == "missing_dataset":
        backend.pop("dataset_identity")
    elif mutation == "numeric_dataset":
        backend["dataset_identity"] = 7
    elif mutation == "boolean_view":
        backend["view_identity"] = True
    elif mutation == "list_view":
        backend["training_view_identity"] = ["b" * 64]
    elif mutation == "padded_view":
        backend["view_identity"] = f" {'b' * 64} "
    elif mutation == "disagreeing_view":
        backend["training_view_identity"] = "c" * 64
    else:
        backend.pop("product_training_run_id")
    run.write_state(state)

    result = _resume_result(monkeypatch, config, run.run_id)

    assert result.status == "BLOCKED"
    assert result.exit_code == ExitCode.INVALID
    assert result.data["backend_launches"] == 0
    assert "identity" in result.message.lower()


def test_progress_tty_bar_and_meaningful_eta() -> None:
    snapshot = ProgressSnapshot("Dataset build", "RUNNING", 75, 100, 30, "Semantic labeling", observations=4)
    output = render_progress(snapshot, tty=True)
    assert "75%" in output and "75 / 100" in output and "ETA:" in output
    assert "█" in output


def test_progress_omits_eta_without_observations() -> None:
    snapshot = ProgressSnapshot("Dataset build", "RUNNING", 1, 100, 1, "Starting", observations=1)
    assert "ETA:" not in render_progress(snapshot, tty=True)


def test_progress_non_tty_is_stable_line_without_controls() -> None:
    snapshot = ProgressSnapshot("Dataset build", "RUNNING", 3, None, 1.5, "Extract", observations=0)
    output = render_progress(snapshot, tty=False)
    assert "stage=Dataset build" in output and "total=?" in output
    assert "\n" not in output and "\x1b[" not in output


def test_progress_no_color_uses_ascii() -> None:
    snapshot = ProgressSnapshot("Train", "RUNNING", 1, 2, 2, "Plan", observations=3)
    output = render_progress(snapshot, tty=True, no_color=True)
    assert "#" in output and "█" not in output


def test_synthetic_end_to_end_operator_flow(monkeypatch, tmp_path: Path, capsys) -> None:
    """Exercise the entire ordinary surface without any backend action."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "spritelab.yaml"
    monkeypatch.setattr(cli, "discover_config", lambda _start: target if target.exists() else None)
    monkeypatch.setattr(cli, "_project_root_for_init", lambda: tmp_path)
    commands = [
        (["init", "--json"], 0),
        (["doctor", "--json"], 7),
        (["status", "--json"], 0),
        (["dataset", "build", "--dry-run", "--json"], 3),
        (["train", "--dry-run", "--json"], 3),
        (["eval", "--dry-run", "--json"], 3),
        (["report", "--json"], 0),
    ]
    observed = []
    for arguments, expected in commands:
        code, output, _ = _invoke(arguments, capsys)
        payload = json.loads(output)
        observed.append(payload["command"])
        assert code == payload["exit_code"] == expected
    assert observed == ["init", "doctor", "status", "dataset build", "train", "eval", "report"]
    assert not list(tmp_path.rglob("*.pt"))
