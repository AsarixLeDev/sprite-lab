from __future__ import annotations

import json

import pytest

from spritelab.training.cli import main as train_cli
from spritelab.training.v1_gallery_gui import (
    BUILTIN_SOURCE,
    CUSTOM_SOURCE,
    FAMILY_GROUNDED,
    FAMILY_OOD,
    FAMILY_STRESS,
    _checkpoint_status,
    _contact_sheet_paths,
    _families_to_flags,
    _preview_prompt_set,
    _report_summary,
    _resolve_v1_checkpoint,
    _sample_image_items,
)


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


def test_families_to_flags_maps_selected_labels() -> None:
    assert _families_to_flags([FAMILY_GROUNDED, FAMILY_STRESS, FAMILY_OOD]) == (True, True, True)
    assert _families_to_flags([FAMILY_GROUNDED]) == (True, False, False)
    assert _families_to_flags([]) == (False, False, False)


def test_resolve_v1_checkpoint_prefers_ema_sibling(tmp_path) -> None:
    ema = tmp_path / "checkpoint_last_ema.pt"
    ema.write_bytes(b"x")
    resolved = _resolve_v1_checkpoint(tmp_path / "checkpoint_last.pt")
    assert resolved == ema


def test_checkpoint_status_reports_found_and_missing(tmp_path) -> None:
    present = tmp_path / "checkpoint_last_ema.pt"
    present.write_bytes(b"x")
    assert "✅ Found" in _checkpoint_status(str(present))

    missing = _checkpoint_status(str(tmp_path / "nope.pt"))
    assert "❌" in missing
    assert "docs/v1_default.md" in missing


def test_preview_prompt_set_builtin_counts_and_filters() -> None:
    preview = _preview_prompt_set(
        BUILTIN_SOURCE,
        "",
        ["weapon"],
        [FAMILY_GROUNDED],
    )
    assert "prompt(s)." in preview
    assert "weapon" in preview

    empty = _preview_prompt_set(BUILTIN_SOURCE, "", [], [])
    assert "at least one" in empty


def test_preview_prompt_set_custom_missing_file() -> None:
    preview = _preview_prompt_set(CUSTOM_SOURCE, "does_not_exist.jsonl", [], [])
    assert "not found" in preview.lower()


def test_sample_image_items_reads_manifest(tmp_path) -> None:
    from PIL import Image

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    Image.new("RGBA", (32, 32), (10, 20, 30, 255)).save(samples_dir / "sprite_0.png")
    manifest = samples_dir / "generated_manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "red sword", "paths": {"indexed_png": "sprite_0.png"}})
        + "\n"
        + json.dumps({"prompt": "missing", "paths": {"indexed_png": "gone.png"}})
        + "\n",
        encoding="utf-8",
    )

    items = _sample_image_items({"output_paths": {"samples_dir": str(samples_dir)}})

    assert items == [(str(samples_dir / "sprite_0.png"), "red sword")]
