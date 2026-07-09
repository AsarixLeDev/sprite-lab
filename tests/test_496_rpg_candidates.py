from _harvest_testdata import make_sprite_png
from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2
from spritelab.harvest.label_candidates import candidate_objects_for_record
from spritelab.harvest.label_fusion_v2 import FusionThresholds
from spritelab.harvest.label_schema import LabelSuggestion, label_suggestion_to_json
from spritelab.harvest.label_v2_pipeline import (
    build_label_v2_record,
    format_label_v2_run_report,
    summarize_label_v2_records,
)


def _record(filename: str, source_id: str = "oga_496_rpg_icons_32fix") -> dict[str, str]:
    return {
        "sprite_id": filename.removesuffix(".png").lower(),
        "filename": filename,
        "relative_path": filename,
        "source_id": source_id,
        "source_name": source_id,
    }


def _candidates(filename: str) -> tuple[str, ...]:
    record = _record(filename)
    result = suggest_from_filename_v2(record)
    return candidate_objects_for_record(record, result.profile, label_suggestion_to_json(result.suggestion))


def _assert_contains(filename: str, expected: set[str]) -> None:
    candidates = set(_candidates(filename))
    assert expected <= candidates


def test_496_rpg_candidate_families_from_filename_tokens() -> None:
    _assert_contains("A_Armor05.png", {"armor", "chestplate", "breastplate", "leather_armor"})
    _assert_contains("A_Armour01.png", {"armor", "chestplate", "breastplate", "leather_armor"})
    _assert_contains("A_Shoes05.png", {"shoes", "boots", "boot", "footwear"})
    _assert_contains("AC_Necklace01.png", {"necklace", "amulet", "pendant", "medallion", "charm"})
    _assert_contains("AC_Ring01.png", {"ring", "jewelry_ring"})
    _assert_contains("C_Elm01.png", {"helmet", "helm", "headgear"})
    _assert_contains("C_Hat02.png", {"hat", "wizard_hat", "cap"})
    _assert_contains("E_Bones02.png", {"bones", "bone", "skull"})
    _assert_contains("E_Gold02.png", {"gold", "gold_coin", "coin", "gold_ingot", "gold_nugget"})
    _assert_contains("E_Metal02.png", {"metal", "metal_ore", "ore", "ingot", "stone"})


def test_496_rpg_candidates_are_deduped_and_canonical() -> None:
    malformed = {"sho", "armour", "elm", "ambiguou", "ambiguou_object", "ambiguou_shape"}
    for filename in (
        "A_Armor05.png",
        "A_Armour01.png",
        "A_Shoes05.png",
        "AC_Necklace01.png",
        "AC_Ring01.png",
        "C_Elm01.png",
        "C_Hat02.png",
        "E_Bones02.png",
        "E_Gold02.png",
        "E_Metal02.png",
    ):
        candidates = _candidates(filename)
        assert len(candidates) == len(set(candidates))
        assert not malformed & set(candidates)


def test_496_candidates_flow_to_record_filename_and_vlm_descriptor(tmp_path) -> None:
    run = tmp_path / "run"
    png = make_sprite_png(run / "A_Shoes05.png")
    record = _record("A_Shoes05.png") | {"final_png_path": str(png)}
    vlm = LabelSuggestion("armor", "boot", confidence=0.86, source="vlm_descriptor", source_consistency="consistent")

    row = build_label_v2_record(record, run_dir=run, vlm=vlm, thresholds=FusionThresholds())

    assert {"shoes", "boots", "boot", "footwear"} <= set(row["candidate_object_names"])
    assert row["filename_suggestion"]["candidate_object_names"] == row["candidate_object_names"]
    assert row["vlm_descriptor"]["candidate_object_names"] == row["candidate_object_names"]
    assert row["label_quality"]["bucket"] == "auto_vlm_candidate_ranked"

    summary = summarize_label_v2_records([row])
    assert summary["records_with_candidates"] == 1
    assert summary["records_without_candidates"] == 0
    assert summary["top_candidate_families"]["shoes"] == 1
    report = format_label_v2_run_report(summary)
    assert "## Candidates" in report
    assert "- records_with_candidates: 1" in report
