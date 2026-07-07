from __future__ import annotations

import io

import pytest

from spritelab.training import progress
from spritelab.training.progress import StepProgressBar, format_duration, progress_enabled, sparkline


def test_format_duration_variants() -> None:
    assert format_duration(None) == "--:--"
    assert format_duration(-5) == "--:--"
    assert format_duration(9) == "00:09"
    assert format_duration(75) == "01:15"
    assert format_duration(3725) == "1:02:05"


def test_sparkline_maps_range_to_ramp() -> None:
    line = sparkline([1.0, 2.0, 3.0, 4.0])
    assert len(line) == 4
    # ascending values -> last char is the tallest block
    assert line[0] != line[-1]
    assert sparkline([]) == ""
    flat = sparkline([2.0, 2.0, 2.0])
    assert set(flat) == {"█"}


def test_sparkline_ascii_only_uses_ascii_ramp() -> None:
    line = sparkline([1.0, 5.0], ascii_only=True)
    assert all(ord(ch) < 128 for ch in line)


def test_progress_enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRITELAB_PROGRESS", raising=False)
    assert progress_enabled() is True
    monkeypatch.setenv("SPRITELAB_PROGRESS", "off")
    assert progress_enabled() is False
    monkeypatch.setenv("SPRITELAB_PROGRESS", "1")
    assert progress_enabled() is True


def test_progress_bar_plain_stream_emits_lines_and_summary() -> None:
    stream = io.StringIO()
    bar = StepProgressBar(4, desc="unit", stream=stream, enabled=True, plain_lines=4)
    for step in range(1, 5):
        bar.update(step, loss=1.0 / step, lr=1e-3)
    bar.close(final_loss=0.25)
    text = stream.getvalue()
    assert "unit" in text
    assert "loss" in text
    assert "4/4" in text
    assert "unit done" in text
    assert "best 0.2500" in text


def test_progress_bar_disabled_writes_nothing() -> None:
    stream = io.StringIO()
    bar = StepProgressBar(10, stream=stream, enabled=False)
    for step in range(1, 11):
        bar.update(step, loss=0.5)
    bar.close()
    assert stream.getvalue() == ""


def test_progress_bar_tracks_best_and_ema() -> None:
    bar = StepProgressBar(3, stream=io.StringIO(), enabled=True)
    bar.update(1, 1.0)
    bar.update(2, 0.2)
    bar.update(3, 0.6)
    assert bar.best_loss == pytest.approx(0.2)
    assert bar.ema_loss is not None and 0.2 < bar.ema_loss < 1.0
