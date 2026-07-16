from __future__ import annotations

import json
from pathlib import Path

from spritelab.training import live_monitor
from spritelab.training.live_monitor import (
    aggregate,
    collect_run_states,
    render_html,
    render_text_dashboard,
    run_live_monitor,
)


def _write_run(run_dir: Path, *, name: str, losses: list[float], max_steps: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"max_steps": max_steps}), encoding="utf-8")
    lines = []
    for step, loss in enumerate(losses, start=1):
        lines.append(json.dumps({"step": step, "loss": loss, "learning_rate": 1e-3, "elapsed_seconds": step * 0.5}))
    (run_dir / "train_metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_collect_run_states_parses_metrics(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "runs" / "overfit_16_sprites", name="overfit_16_sprites", losses=[1.0, 0.5, 0.25], max_steps=3
    )
    states = collect_run_states(tmp_path)
    assert len(states) == 1
    state = states[0]
    assert state.name == "overfit_16_sprites"
    assert state.step == 3
    assert state.max_steps == 3
    assert state.last_loss == 0.25
    assert state.best_loss == 0.25
    assert state.first_loss == 1.0
    assert state.fraction == 1.0
    # step >= max_steps -> done
    assert state.status == "done"
    assert state.rate is not None and state.rate > 0


def test_collect_run_states_single_run_dir(tmp_path: Path) -> None:
    _write_run(tmp_path, name="solo", losses=[2.0, 1.0], max_steps=10)
    states = collect_run_states(tmp_path)
    assert len(states) == 1
    assert states[0].step == 2
    assert states[0].max_steps == 10
    assert 0.0 < states[0].fraction < 1.0


def test_aggregate_counts_and_fraction(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs" / "a", name="a", losses=[1.0] * 10, max_steps=10)  # done
    _write_run(tmp_path / "runs" / "b", name="b", losses=[1.0] * 3, max_steps=9)  # partial
    states = collect_run_states(tmp_path)
    agg = aggregate(states)
    assert agg["runs"] == 2
    assert agg["total_target"] == 19
    assert agg["total_steps"] == 13
    assert agg["fraction"] == 13 / 19


def test_render_html_contains_runs_and_svg(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "runs" / "overfit_64_sprites", name="overfit_64_sprites", losses=[1.0, 0.8, 0.6, 0.4], max_steps=8
    )
    states = collect_run_states(tmp_path)
    out = render_html(states, refresh_seconds=3)
    assert "overfit_64_sprites" in out
    assert "<svg" in out
    assert 'http-equiv="refresh"' in out
    assert "polyline" in out


def test_render_text_dashboard_handles_empty() -> None:
    assert "no train_metrics" in render_text_dashboard([])


def test_run_live_monitor_once_writes_html(tmp_path: Path) -> None:
    _write_run(tmp_path / "runs" / "r1", name="r1", losses=[1.0, 0.5], max_steps=4)
    html_path = tmp_path / "dash" / "live.html"
    summary = run_live_monitor(tmp_path, html_path=html_path, once=True)
    assert html_path.is_file()
    assert "r1" in html_path.read_text(encoding="utf-8")
    assert summary["runs"] == 1


def test_stalled_run_detected(tmp_path: Path, monkeypatch) -> None:
    _write_run(tmp_path / "runs" / "old", name="old", losses=[1.0, 0.9], max_steps=100)
    # Force the "now" clock far into the future so the run looks stale.
    real = live_monitor.time.time()
    monkeypatch.setattr(live_monitor.time, "time", lambda: real + 10_000)
    states = collect_run_states(tmp_path)
    assert states[0].status == "stalled"
