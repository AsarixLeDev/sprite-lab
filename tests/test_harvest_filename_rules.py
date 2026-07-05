"""Tests for filename-derived harvest metadata suggestions."""

from __future__ import annotations

import random

from _harvest_testdata import make_sprite_png

from spritelab.harvest.catalog import write_jsonl
from spritelab.harvest.filename_rules import (
    filename_suggestion_to_dict,
    metadata_suggestions_differ,
    parse_filename_metadata,
)
from spritelab.harvest.prefill_review_gui import load_prefill_review_items, random_mismatch_index
from spritelab.harvest.prefill_review_gui import _preview_image, _resolve_image_path


def test_filename_rules_parse_banana_item_icon() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_i_c_banana",
        filename="I_C_Banana.png",
    )

    assert filename_suggestion_to_dict(suggestion) == {
        "category": "item_icon",
        "object_name": "banana",
        "tags": ["banana", "fruit", "food", "consumable"],
        "materials": [],
        "mood": [],
        "short_description": "A 32x32 pixel-art banana item icon.",
        "confidence": 0.98,
        "confidence_reason": "recognized filename code 'i' and object token 'banana'",
        "source": "filename_rules",
    }


def test_filename_rules_parse_poison_effect_from_sprite_id() -> None:
    suggestion = parse_filename_metadata("oga_496_rpg_icons_32fix_s_poison01")

    assert suggestion.category == "effect_icon"
    assert suggestion.object_name == "poison"
    assert suggestion.tags == ("poison", "status_effect", "debuff", "magic")
    assert suggestion.short_description == "A 32x32 pixel-art poison status/effect icon."
    assert suggestion.confidence == 0.95
    assert suggestion.source == "filename_rules"


def test_filename_rules_parse_axe_weapon() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_w_axe014",
        filename="W_Axe014.png",
    )

    assert suggestion.category == "weapon"
    assert suggestion.object_name == "axe"
    assert suggestion.tags == ("axe", "weapon", "tool", "melee")
    assert suggestion.materials == ("metal", "wood")
    assert suggestion.short_description == "A 32x32 pixel-art axe weapon icon."
    assert suggestion.confidence == 0.95


def test_filename_rules_parse_accessory_medal_without_prefix_in_object_name() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_ac_medal04",
        filename="Ac_Medal04.png",
    )

    assert suggestion.category == "item_icon"
    assert suggestion.object_name == "medal"
    assert suggestion.tags == ("medal", "accessory", "jewelry", "award")
    assert suggestion.materials == ("metal",)
    assert suggestion.short_description == "A 32x32 pixel-art medal item icon."
    assert suggestion.confidence == 0.95


def test_filename_rules_parse_accessory_necklace_without_prefix_in_object_name() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_ac_necklace01",
        filename="Ac_Necklace01.png",
    )

    assert suggestion.category == "item_icon"
    assert suggestion.object_name == "necklace"
    assert suggestion.tags == ("necklace", "accessory", "jewelry")
    assert suggestion.materials == ("metal",)
    assert suggestion.short_description == "A 32x32 pixel-art necklace item icon."
    assert suggestion.confidence == 0.95


def test_filename_rules_parse_accessory_ring_without_prefix_in_object_name() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_ac_ring01",
        filename="Ac_Ring01.png",
    )

    assert suggestion.object_name == "ring"
    assert suggestion.tags == ("ring", "accessory", "jewelry")
    assert suggestion.materials == ("metal",)
    assert suggestion.confidence == 0.95


def test_filename_rules_strip_structural_c_prefix_when_object_is_known() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_c_hat01",
        filename="C_Hat01.png",
    )

    assert suggestion.category == "armor"
    assert suggestion.object_name == "hat"
    assert suggestion.tags == ("hat", "clothing", "headgear", "armor")
    assert suggestion.confidence == 0.9


def test_filename_rules_include_object_tag_with_subtype_tags() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_i_c_redpepper",
        filename="I_C_RedPepper.png",
    )

    assert suggestion.object_name == "red_pepper"
    assert suggestion.tags == ("red_pepper", "consumable")
    assert suggestion.confidence == 0.75


def test_filename_rules_strip_known_structural_prefix_even_for_unknown_object() -> None:
    suggestion = parse_filename_metadata(
        "oga_496_rpg_icons_32fix_c_elm01",
        filename="C_Elm01.png",
    )

    assert suggestion.object_name == "elm"
    assert suggestion.tags == ("elm",)
    assert suggestion.confidence == 0.75


def test_filename_rules_detect_qwen_object_mismatch() -> None:
    filename_suggestion = parse_filename_metadata("sprite", filename="I_C_Banana.png")

    reasons = metadata_suggestions_differ(
        filename_suggestion,
        {"category": "item_icon", "object_name": "apple", "tags": ["apple"]},
    )

    assert any("object_name" in reason for reason in reasons)


def test_filename_rules_detect_missing_qwen_suggestion() -> None:
    filename_suggestion = parse_filename_metadata("sprite", filename="I_C_Banana.png")

    assert metadata_suggestions_differ(filename_suggestion, {}) == ("missing_qwen_suggestion",)


def test_prefill_review_loads_run_and_random_mismatch(tmp_path) -> None:
    run_dir = tmp_path / "run"
    png_path = make_sprite_png(run_dir / "I_C_Banana.png")
    write_jsonl(
        run_dir / "imported.jsonl",
        [
            {
                "sprite_id": "oga_496_rpg_icons_32fix_i_c_banana",
                "final_png_path": str(png_path),
                "relative_path": "I_C_Banana.png",
                "auto_metadata": {
                    "qwen_suggestion": {
                        "category": "item_icon",
                        "object_name": "apple",
                        "tags": ["apple"],
                    }
                },
            }
        ],
    )

    items = load_prefill_review_items(run_dir)

    assert len(items) == 1
    assert items[0].filename_suggestion.object_name == "banana"
    assert items[0].qwen_suggestion["object_name"] == "apple"
    assert items[0].mismatch_reasons
    assert random_mismatch_index(items, rng=random.Random(1337)) == 0


def test_prefill_review_resolves_project_relative_image_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "harvest_runs" / "run"
    png_path = make_sprite_png(tmp_path / "data_sources" / "fixed" / "I_C_Banana.png")
    record = {"final_png_path": "data_sources\\fixed\\I_C_Banana.png"}

    resolved = _resolve_image_path(run_dir, record)
    preview = _preview_image(resolved)

    assert resolved == png_path.resolve()
    assert preview is not None
    assert preview.mode == "RGB"
    assert preview.size == (256, 256)
