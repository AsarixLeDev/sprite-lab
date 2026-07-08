from __future__ import annotations

import pytest

from spritelab.training.cli import main as train_cli
from spritelab.training.v1_gallery_gui import _contact_sheet_paths, _report_summary


def _fake_report(contact_sheets_dir: str) -> dict:
    return {
        "prompt_set": {"prompt_count": 5},
        "sample_count": 5,
        "generated_qa": {"errors": 0, "ok": True},
        "generated_review": {"rare_color_warning_rate": 0.0},
        "projection_summary": {
            "median_visible_colors_before": 32,
            "median_visible_colors_after": 12,
            "destructive_rate": 0.0,
        },
        "output_paths": {
            "samples_dir": "samples",
            "contact_sheets_dir": contact_sheets_dir,
            "report_markdown": "v1_gallery_report.md",
            "contact_sheets": {"overall": "overall.png"},
        },
    }


def test_report_summary_includes_key_metrics() -> None:
    summary = _report_summary(_fake_report("contact_sheets"))
    assert "Prompt count: 5" in summary
    assert "Median visible colors: 32 -> 12" in summary
    assert "QA errors: 0" in summary


def test_contact_sheet_paths_skips_missing_files(tmp_path) -> None:
    contact_sheets_dir = tmp_path / "contact_sheets"
    contact_sheets_dir.mkdir()
    (contact_sheets_dir / "overall.png").write_bytes(b"not a real png but a real file")

    paths = _contact_sheet_paths(_fake_report(str(contact_sheets_dir)))

    assert paths == [str(contact_sheets_dir / "overall.png")]


def test_v1_gallery_gui_cli_help_lists_expected_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        train_cli(["v1-gallery-gui", "--help"])
    assert exc_info.value.code == 0
    text = capsys.readouterr().out
    assert "--out" in text
    assert "--host" in text
    assert "--port" in text


def test_v1_gallery_gui_cli_reports_missing_gradio_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    import spritelab.training.v1_gallery_gui as gui_module

    def _raise_missing_gradio(**_kwargs: object) -> None:
        raise RuntimeError("The v1 gallery GUI requires gradio. Install it with: pip install gradio")

    monkeypatch.setattr(gui_module, "launch_v1_gallery_gui", _raise_missing_gradio)
    with pytest.raises(SystemExit) as exc_info:
        train_cli(["v1-gallery-gui", "--out", "experiments/v1_gallery_gui"])
    assert exc_info.value.code == 1
