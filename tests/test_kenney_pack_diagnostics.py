from __future__ import annotations

import json
from pathlib import Path

from _harvest_testdata import make_sprite_png
from spritelab.harvest.label_fusion_v2 import FusionThresholds
from spritelab.harvest.label_v2_pipeline import build_label_v2_record
from spritelab.harvest.pack_drilldown import build_pack_drilldown


def test_kenney_coordinate_and_ui_records_remain_review(tmp_path: Path) -> None:
    coord = _label(tmp_path, "r000_c001.png")
    ui = _label(tmp_path, "ui_panel.png")

    assert coord["needs_review"] is True
    assert coord["safe_prefill"]["object_name"] == ""
    assert ui["needs_review"] is True
    assert ui["bucket"].startswith("needs_review")


def test_kenney_drilldown_reports_low_base_object_coverage_and_manual_seed(tmp_path: Path) -> None:
    run = tmp_path / "harvest_runs" / "kenney_micro_roguelike"
    run.mkdir(parents=True)
    imported = [
        {
            "sprite_id": "r000_c001",
            "relative_path": "r000_c001.png",
            "final_png_path": "r000_c001.png",
            "source_id": "kenney_micro_roguelike",
            "source_name": "Kenney",
        }
    ]
    predictions = [
        {
            "sprite_id": "r000_c001",
            "relative_path": "r000_c001.png",
            "bucket": "needs_review",
            "needs_review": True,
            "source_profile": {"name": "kenney_micro_roguelike", "filename_trust": "none"},
            "safe_prefill": {"category": "unknown", "object_name": ""},
            "label_quality": {"bucket": "needs_review", "needs_review": True},
        }
    ]
    _write_jsonl(run / "imported.jsonl", imported)
    _write_jsonl(run / "label_v2_suggestions.jsonl", predictions)

    report = build_pack_drilldown(run)

    assert report["semantic_v3_base_object_coverage"] == 0.0
    assert "base_object_extractor_gap" in report["recommended_fix_classes"]
    assert "needs_manual_golden_seed" in report["recommended_fix_classes"]


def _label(tmp_path: Path, filename: str) -> dict:
    png = make_sprite_png(tmp_path / filename)
    return build_label_v2_record(
        {
            "sprite_id": Path(filename).stem,
            "relative_path": filename,
            "final_png_path": str(png),
            "source_id": "kenney_micro_roguelike",
            "source_name": "Kenney Micro Roguelike",
            "status": "quarantine",
        },
        run_dir=tmp_path,
        vlm=None,
        thresholds=FusionThresholds(),
        vlm_status="skipped_no_backend",
        vlm_stats=("vlm_skipped_no_backend",),
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
