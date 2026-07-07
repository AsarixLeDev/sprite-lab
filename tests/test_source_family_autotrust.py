from __future__ import annotations

from pathlib import Path

from _harvest_testdata import make_sprite_png

from spritelab.harvest.label_fusion_v2 import FusionThresholds
from spritelab.harvest.label_v2_pipeline import build_label_v2_record


def _record(tmp_path: Path, filename: str, source_id: str) -> dict[str, str]:
    png = make_sprite_png(tmp_path / filename)
    return {
        "sprite_id": f"{source_id}_{Path(filename).stem}",
        "relative_path": filename,
        "final_png_path": str(png),
        "source_id": source_id,
        "source_name": source_id,
        "status": "quarantine",
    }


def _label(tmp_path: Path, filename: str, source_id: str) -> dict:
    return build_label_v2_record(
        _record(tmp_path, filename, source_id),
        run_dir=tmp_path,
        vlm=None,
        thresholds=FusionThresholds(),
        vlm_status="skipped_no_backend",
        vlm_stats=("vlm_skipped_no_backend",),
    )


def test_key_family_clean_filename_auto_labels_key(tmp_path: Path) -> None:
    row = _label(tmp_path, "key_01.png", "oga_cc0_key_7soul1")
    assert row["bucket"] == "auto_filename_trusted"
    assert row["safe_prefill"]["object_name"] == "key"


def test_jewelry_family_auto_labels_necklace_and_ring_color_attribute(tmp_path: Path) -> None:
    necklace = _label(tmp_path, "necklace_01.png", "oga_cc0_jewelry_7soul1")
    assert necklace["bucket"] == "auto_filename_trusted"
    assert necklace["safe_prefill"]["object_name"] == "necklace"

    ring = _label(tmp_path, "ring_red.png", "oga_cc0_jewelry_buch")
    assert ring["bucket"] == "auto_filename_trusted"
    assert ring["safe_prefill"]["object_name"] == "ring"
    assert "red" in ring["safe_prefill"]["tags"]
    assert "red" in ring["safe_prefill"]["dominant_colors"]


def test_food_and_tool_family_clean_filenames_auto_label(tmp_path: Path) -> None:
    apple = _label(tmp_path, "apple.png", "oga_cc0_food_arlantr")
    carrot = _label(tmp_path, "carrot.png", "oga_cc0_food_arlantr")
    torch = _label(tmp_path, "torch_01.png", "oga_cc0_tool_dcss")

    assert apple["bucket"] == "auto_filename_trusted"
    assert apple["safe_prefill"]["object_name"] == "apple"
    assert carrot["bucket"] == "auto_filename_trusted"
    assert carrot["safe_prefill"]["object_name"] == "carrot"
    assert torch["bucket"] == "auto_filename_trusted"
    assert torch["safe_prefill"]["object_name"] == "torch"


def test_source_author_token_artifacts_remain_review(tmp_path: Path) -> None:
    for filename, source_id in (
        ("key_rcorre.png", "oga_cc0_key_rcorre"),
        ("tool_dcss.png", "oga_cc0_tool_dcss"),
        ("arlan_tr.png", "oga_cc0_food_arlantr"),
    ):
        row = _label(tmp_path, filename, source_id)
        assert row["needs_review"] is True
        assert row["bucket"].startswith("needs_review")


def test_sheet_coordinate_only_filenames_remain_review(tmp_path: Path) -> None:
    row = _label(tmp_path, "r000_c001.png", "oga_cc0_key_7soul1")
    assert row["needs_review"] is True
    assert row["safe_prefill"]["object_name"] == ""


def test_potion_color_filename_keeps_color_as_attribute(tmp_path: Path) -> None:
    row = _label(tmp_path, "red_potion.png", "oga_potion_7soul1")
    assert row["bucket"] == "auto_filename_trusted"
    assert row["safe_prefill"]["object_name"] == "potion"
    assert "red" in row["safe_prefill"]["tags"]
    assert "red" in row["safe_prefill"]["dominant_colors"]


def test_mushroom_color_filename_keeps_mushroom_as_object(tmp_path: Path) -> None:
    row = _label(tmp_path, "orange_mushroom-1.png", "oga_mushrooms_32")
    assert row["bucket"] == "auto_filename_trusted"
    assert row["safe_prefill"]["object_name"] == "mushroom"
    assert "orange" in row["safe_prefill"]["tags"]


def test_potion_only_author_coordinate_filename_falls_back_to_potion(tmp_path: Path) -> None:
    row = _label(tmp_path, "potion-buch_r000_c002.png", "oga_potion_buch")
    assert row["bucket"] == "auto_filename_trusted"
    assert row["safe_prefill"]["object_name"] == "potion"
    assert "buch" not in row["safe_prefill"]["object_name"]

    outside = _label(tmp_path, "r000_c002.png", "oga_cc0_key_7soul1")
    assert outside["needs_review"] is True
    assert outside["safe_prefill"]["object_name"] == ""
