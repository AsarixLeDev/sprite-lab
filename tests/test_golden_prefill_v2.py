import json

from _harvest_testdata import make_sprite_png

from spritelab.harvest.assisted_golden import (
    AssistedGoldenLabel,
    append_golden_label,
    build_label_v2_prefilled_candidates,
    build_assisted_golden_label,
    format_golden_prefill_report,
    summarize_golden_prefill_records,
)
from spritelab.harvest.catalog import read_jsonl, write_jsonl
from spritelab.harvest.cli import main
from spritelab.harvest.golden import load_golden_labels


def test_golden_prefill_v2_cli_reads_label_v2_and_initializes_gold_fields(tmp_path, capsys) -> None:
    run = _run_with_label_v2(tmp_path)
    out = run / "golden_candidates_prefilled.jsonl"

    main(
        [
            "golden-prefill-v2",
            "--run",
            str(run),
            "--prediction-file",
            "label_v2_suggestions.jsonl",
            "--n",
            "1",
            "--seed",
            "496",
            "--out",
            str(out),
        ]
    )

    assert "Candidates: 1" in capsys.readouterr().out
    row = read_jsonl(out)[0]
    assert row["prefill_source"] == "label_v2"
    assert row["prefill_category"] == "armor"
    assert row["prefill_object_name"] == "armor"
    assert row["prefill_tags"] == ["armor", "wearable"]
    assert row["prefill_short_description"] == "A gray armor icon."
    assert row["prefill_materials"] == ["metal"]
    assert row["prefill_mood"] == ["defensive"]
    assert row["gold_category"] == row["prefill_category"]
    assert row["gold_object_name"] == row["prefill_object_name"]
    assert row["gold_tags"] == row["prefill_tags"]
    assert row["candidate_object_names"][:3] == ["armor", "chestplate", "breastplate"]
    assert row["alternative_object_names"] == ["chestplate", "breastplate"]
    assert row["prefill_bucket"] == "auto_prefix_family_trusted"
    assert "prefix_family_trusted" in row["prefill_flags"]
    assert row["vlm_object_name"] == "armor"
    assert row["vlm_short_description"] == "Looks like a chestplate."
    assert row["vlm_source_consistency"] == "consistent"
    assert row["visual_facts"]["dominant_colors"] == ["gray", "black"]
    assert row["prefill_was_corrected"] is False
    assert row["correction_fields"] == []
    assert not (run / "golden_labels.jsonl").exists()


def test_existing_golden_label_is_used_by_default_and_overwrite_restores_prefill(tmp_path) -> None:
    run = _run_with_label_v2(tmp_path)
    append_golden_label(
        run / "golden_labels.jsonl",
        AssistedGoldenLabel(
            sprite_id="armor_05",
            category="armor",
            object_name="chestplate",
            tags=("armor", "metal"),
            short_description="A reviewed chestplate.",
            materials=("metal",),
        ),
    )

    candidate = build_label_v2_prefilled_candidates(run, n=1)[0]
    overwritten = build_label_v2_prefilled_candidates(run, n=1, overwrite=True)[0]

    assert candidate.gold_object_name == "chestplate"
    assert candidate.prefill_object_name == "armor"
    assert candidate.prefill_was_corrected is True
    assert "object_name" in candidate.correction_fields
    assert overwritten.gold_object_name == "armor"
    assert overwritten.prefill_was_corrected is False


def test_saved_assisted_label_preserves_prefill_audit_and_eval_compatibility(tmp_path) -> None:
    run = _run_with_label_v2(tmp_path)
    candidate = build_label_v2_prefilled_candidates(run, n=1)[0]

    label = build_assisted_golden_label(
        candidate,
        category="armor",
        object_name="chestplate",
        tags=("armor", "metal"),
        short_description="A reviewed chestplate.",
        materials=("metal",),
        mood=("defensive",),
        labeler="mathieu",
    )
    append_golden_label(run / "golden_labels.jsonl", label)
    row = read_jsonl(run / "golden_labels.jsonl")[0]

    assert row["object_name"] == "chestplate"
    assert row["prefill_source"] == "label_v2"
    assert row["prefill_object_name"] == "armor"
    assert row["candidate_object_names"][:2] == ["armor", "chestplate"]
    assert row["alternative_object_names"] == ["chestplate", "breastplate"]
    assert row["prefill_was_corrected"] is True
    assert "object_name" in row["correction_fields"]
    assert "short_description" in row["correction_fields"]
    loaded = load_golden_labels(run / "golden_labels.jsonl")
    assert loaded["armor_05"].object_name == "chestplate"


def test_golden_prefill_report_summarizes_corrections() -> None:
    records = [
        {
            "sprite_id": "a",
            "category": "armor",
            "object_name": "armor",
            "tags": ["armor"],
            "prefill_source": "label_v2",
            "prefill_category": "armor",
            "prefill_object_name": "armor",
            "prefill_tags": ["armor"],
            "prefill_bucket": "auto_prefix_family_trusted",
            "prefill_was_corrected": False,
            "correction_fields": [],
        },
        {
            "sprite_id": "b",
            "category": "armor",
            "object_name": "chestplate",
            "tags": ["armor", "metal"],
            "prefill_source": "label_v2",
            "prefill_category": "armor",
            "prefill_object_name": "armor",
            "prefill_tags": ["armor"],
            "prefill_bucket": "auto_prefix_family_trusted",
            "prefill_was_corrected": True,
            "correction_fields": ["object_name", "tags"],
        },
    ]

    summary = summarize_golden_prefill_records(records)
    report = format_golden_prefill_report(summary)

    assert summary["total"] == 2
    assert summary["prefilled_from_label_v2"] == 2
    assert summary["corrected"] == 1
    assert summary["corrections_by_field"]["object_name"] == 1
    assert summary["corrections_by_bucket"]["auto_prefix_family_trusted"] == 1
    assert "Correction rate: 50.0%" in report


def _run_with_label_v2(tmp_path):
    run = tmp_path / "run"
    image = make_sprite_png(run / "A_Armor05.png")
    write_jsonl(
        run / "label_v2_suggestions.jsonl",
        [
            {
                "sprite_id": "armor_05",
                "source_id": "oga_496_rpg_icons_32fix",
                "source_name": "496 RPG Icons",
                "relative_path": "A_Armor05.png",
                "final_png_path": str(image),
                "source_profile": {"name": "oga_496_rpg_icons"},
                "candidate_object_names": ["armor", "chestplate", "breastplate", "leather_armor"],
                "safe_prefill": {
                    "category": "armor",
                    "object_name": "armor",
                    "tags": ["armor", "wearable"],
                    "short_description": "A gray armor icon.",
                    "materials": ["metal"],
                    "mood": ["defensive"],
                    "confidence": 0.75,
                },
                "vlm_descriptor": {
                    "category": "armor",
                    "object_name": "armor",
                    "alternative_object_names": ["chestplate", "breastplate"],
                    "short_description": "Looks like a chestplate.",
                    "source_consistency": "consistent",
                },
                "visual_facts": {
                    "dominant_colors": ["gray", "black"],
                    "content_width": 31,
                    "content_height": 31,
                    "shape_hints": ["wide"],
                },
                "label_quality": {
                    "bucket": "auto_prefix_family_trusted",
                    "flags": ["prefix_family_trusted", "candidate_object_list"],
                    "review_priority": 0.12,
                },
                "bucket": "auto_prefix_family_trusted",
            }
        ],
    )
    return run
