from __future__ import annotations

from pathlib import Path

from _harvest_testdata import make_sprite_png
from spritelab.harvest.label_fusion_v2 import FusionThresholds
from spritelab.harvest.label_v2_pipeline import build_label_v2_record


def test_potion_color_and_author_coordinate_names_fallback_safely(tmp_path: Path) -> None:
    blue = _label(tmp_path, "potion_01_blue.png", "oga_potion_buch")
    assert blue["bucket"] == "auto_filename_trusted"
    assert blue["safe_prefill"]["object_name"] == "potion"
    assert "blue" in blue["safe_prefill"]["tags"]

    buch = _label(tmp_path, "potion-buch_r000_c002.png", "oga_potion_buch")
    rcorre = _label(tmp_path, "potion-rcorre_r001_c001.png", "oga_potion_rcorre")
    assert buch["safe_prefill"]["object_name"] == "potion"
    assert rcorre["safe_prefill"]["object_name"] == "potion"
    assert "buch" not in buch["safe_prefill"]["object_name"]
    assert "rcorre" not in rcorre["safe_prefill"]["object_name"]


def test_coordinate_only_fallback_is_scoped_to_potion_family(tmp_path: Path) -> None:
    potion = _label(tmp_path, "r000_c002.png", "oga_potion_rcorre")
    key = _label(tmp_path, "r000_c002.png", "oga_cc0_key_rcorre")

    assert potion["bucket"] == "auto_filename_trusted"
    assert potion["safe_prefill"]["object_name"] == "potion"
    assert key["needs_review"] is True
    assert key["safe_prefill"]["object_name"] == ""


def _label(tmp_path: Path, filename: str, source_id: str) -> dict:
    png = make_sprite_png(tmp_path / filename)
    return build_label_v2_record(
        {
            "sprite_id": f"{source_id}_{Path(filename).stem}",
            "relative_path": filename,
            "final_png_path": str(png),
            "source_id": source_id,
            "source_name": source_id,
            "status": "quarantine",
        },
        run_dir=tmp_path,
        vlm=None,
        thresholds=FusionThresholds(),
        vlm_status="skipped_no_backend",
        vlm_stats=("vlm_skipped_no_backend",),
    )
