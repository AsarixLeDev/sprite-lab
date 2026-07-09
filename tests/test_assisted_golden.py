"""Tests for assisted golden candidate loading and labels."""

from __future__ import annotations

import json
from pathlib import Path

from _harvest_testdata import make_sprite_png
from spritelab.harvest.assisted_golden import (
    AssistedGoldenCandidate,
    AssistedGoldenLabel,
    append_golden_label,
    assisted_candidate_from_dict,
    assisted_candidate_to_dict,
    build_assisted_golden_label,
    candidate_review_priority,
    choose_best_prefill,
    load_assisted_candidates,
    load_existing_golden_labels,
    normalize_category,
    normalize_object_name,
    normalize_tags,
    write_golden_candidates_jsonl,
)
from spritelab.harvest.catalog import write_jsonl
from spritelab.harvest.golden import load_golden_labels


def test_loads_accepted_candidates_from_run(tmp_path):
    run = _run_fixture(tmp_path)

    candidates = load_assisted_candidates(run)

    assert {candidate.sprite_id for candidate in candidates} == {"sprite_axe", "sprite_banana"}


def test_ignores_rejected_candidates_by_default(tmp_path):
    run = _run_fixture(tmp_path)

    candidates = load_assisted_candidates(run)

    assert "sprite_bad" not in {candidate.sprite_id for candidate in candidates}


def test_include_statuses_can_include_rejected(tmp_path):
    run = _run_fixture(tmp_path)

    candidates = load_assisted_candidates(run, include_statuses=("accepted", "rejected"))

    assert "sprite_bad" in {candidate.sprite_id for candidate in candidates}


def test_loads_rule_suggestion_if_present(tmp_path):
    run = _run_fixture(tmp_path)

    candidate = _by_id(load_assisted_candidates(run), "sprite_axe")

    assert candidate.rule_category == "weapon"
    assert candidate.rule_object_name == "axe"
    assert "melee" in candidate.rule_tags


def test_loads_qwen_suggestion_if_present(tmp_path):
    run = _run_fixture(tmp_path)

    candidate = _by_id(load_assisted_candidates(run), "sprite_banana")

    assert candidate.qwen_category == "item_icon"
    assert candidate.qwen_object_name == "banana"
    assert candidate.qwen_confidence == 0.83


def test_loads_fused_suggestion_if_present(tmp_path):
    run = _run_fixture(tmp_path)

    candidate = _by_id(load_assisted_candidates(run), "sprite_banana")

    assert candidate.fused_category == "item_icon"
    assert candidate.fused_object_name == "banana"
    assert candidate.suggested_source == "fusion"


def test_choose_best_prefill_prefers_fused_non_degenerate():
    candidate = AssistedGoldenCandidate(
        sprite_id="s",
        final_png_path=Path("s.png"),
        rule_category="weapon",
        rule_object_name="axe",
        qwen_category="plant",
        qwen_object_name="mushroom",
        fused_category="item_icon",
        fused_object_name="banana",
        fused_tags=("banana",),
    )

    assert choose_best_prefill(candidate)[:3] == ("item_icon", "banana", ("banana",))


def test_choose_best_prefill_falls_back_to_filename_rules():
    candidate = AssistedGoldenCandidate(
        sprite_id="s",
        final_png_path=Path("s.png"),
        rule_category="weapon",
        rule_object_name="axe",
        rule_tags=("axe", "weapon"),
        fused_category="unknown",
        fused_quality_flags=("degenerate",),
    )

    assert choose_best_prefill(candidate)[4] == "filename_rules"


def test_choose_best_prefill_falls_back_to_qwen():
    candidate = AssistedGoldenCandidate(
        sprite_id="s",
        final_png_path=Path("s.png"),
        qwen_category="plant",
        qwen_object_name="mushroom",
        qwen_tags=("mushroom",),
    )

    assert choose_best_prefill(candidate)[:2] == ("plant", "mushroom")


def test_choose_best_prefill_falls_back_to_existing_metadata():
    candidate = AssistedGoldenCandidate(
        sprite_id="s",
        final_png_path=Path("s.png"),
        existing_category="material",
        existing_tags=("ore",),
    )

    assert choose_best_prefill(candidate) == ("material", "", ("ore",), "", "existing")


def test_normalizes_category():
    assert normalize_category("Item Icon") == "item_icon"
    assert normalize_category("bad category") == "unknown"


def test_normalizes_object_name():
    assert normalize_object_name("Copper Axe!") == "copper_axe"


def test_normalizes_tags():
    assert normalize_tags(" axe, weapon axe  melee ") == ("axe", "weapon", "melee")


def test_append_golden_label_writes_valid_jsonl(tmp_path):
    path = tmp_path / "golden_labels.jsonl"

    append_golden_label(path, AssistedGoldenLabel(sprite_id="s", category="weapon", object_name="axe", tags=("axe",)))

    data = json.loads(path.read_text(encoding="utf-8").strip())
    assert data["sprite_id"] == "s"
    assert data["category"] == "weapon"
    assert data["tags"] == ["axe"]
    assert data["labeled_at"]


def test_existing_golden_loader_reads_assisted_label_extra_keys(tmp_path):
    path = tmp_path / "golden_labels.jsonl"
    append_golden_label(
        path,
        AssistedGoldenLabel(
            sprite_id="s",
            category="weapon",
            object_name="axe",
            tags=("axe",),
            source_id="src",
            prefill_source="fusion",
            prefill_was_corrected=False,
        ),
    )

    labels = load_golden_labels(path)

    assert labels["s"].category == "weapon"
    assert labels["s"].object_name == "axe"


def test_load_golden_labels_last_write_wins(tmp_path):
    path = tmp_path / "golden_labels.jsonl"
    append_golden_label(path, AssistedGoldenLabel(sprite_id="s", category="weapon", object_name="axe", tags=("axe",)))
    append_golden_label(path, AssistedGoldenLabel(sprite_id="s", category="plant", object_name="leaf", tags=("leaf",)))

    labels = load_existing_golden_labels(path)

    assert labels["s"].category == "plant"


def test_correction_tracking_detects_category_correction():
    candidate = _candidate_with_prefill()

    label = build_assisted_golden_label(candidate, category="plant", object_name="axe", tags=("axe",))

    assert label.prefill_was_corrected is True
    assert "category" in label.correction_fields


def test_correction_tracking_detects_object_name_correction():
    candidate = _candidate_with_prefill()

    label = build_assisted_golden_label(candidate, category="weapon", object_name="hammer", tags=("axe",))

    assert "object_name" in label.correction_fields


def test_correction_tracking_detects_tags_correction():
    candidate = _candidate_with_prefill()

    label = build_assisted_golden_label(candidate, category="weapon", object_name="axe", tags=("axe", "metal"))

    assert "tags" in label.correction_fields


def test_candidate_review_priority_ranks_conflicts_higher_than_obvious_cases():
    conflict = AssistedGoldenCandidate(
        sprite_id="conflict",
        final_png_path=Path("c.png"),
        rule_category="weapon",
        rule_object_name="axe",
        qwen_category="plant",
        qwen_object_name="mushroom",
        fused_quality_flags=("filename_qwen_conflict",),
    )
    obvious = AssistedGoldenCandidate(
        sprite_id="obvious",
        final_png_path=Path("o.png"),
        rule_category="weapon",
        rule_object_name="axe",
        suggested_source="filename_rules",
    )

    assert candidate_review_priority(conflict) > candidate_review_priority(obvious)


def test_golden_candidates_jsonl_roundtrip(tmp_path):
    path = tmp_path / "golden_candidates.jsonl"
    candidate = _candidate_with_prefill()

    write_golden_candidates_jsonl(path, [candidate])
    loaded = assisted_candidate_from_dict(json.loads(path.read_text(encoding="utf-8").strip()))

    assert assisted_candidate_to_dict(loaded) == assisted_candidate_to_dict(candidate)


def _run_fixture(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    png_dir = run / "pngs"
    axe = make_sprite_png(png_dir / "W_Axe014.png")
    banana = make_sprite_png(png_dir / "I_C_Banana.png", color=(220, 200, 40, 255))
    bad = make_sprite_png(png_dir / "bad.png")
    write_jsonl(
        run / "imported.jsonl",
        [
            {
                "sprite_id": "sprite_axe",
                "source_id": "src",
                "source_name": "Source",
                "final_png_path": str(axe),
                "relative_path": "W_Axe014.png",
                "status": "accepted",
                "category": "unknown",
                "tags": [],
                "license": "cc0",
                "author": "Author",
                "auto_metadata": {
                    "filename_suggestion": {
                        "category": "weapon",
                        "object_name": "axe",
                        "tags": ["axe", "weapon", "melee"],
                    }
                },
            },
            {
                "sprite_id": "sprite_banana",
                "source_id": "src",
                "source_name": "Source",
                "final_png_path": str(banana),
                "relative_path": "I_C_Banana.png",
                "status": "accepted",
                "category": "unknown",
                "tags": [],
                "license": "cc0",
                "author": "Author",
                "auto_metadata": {},
            },
        ],
    )
    write_jsonl(
        run / "rejected.jsonl",
        [
            {
                "sprite_id": "sprite_bad",
                "source_id": "src",
                "source_name": "Source",
                "final_png_path": str(bad),
                "relative_path": "bad.png",
                "status": "rejected",
                "category": "unknown",
                "tags": [],
                "auto_metadata": {},
            }
        ],
    )
    write_jsonl(
        run / "qwen_suggestions.jsonl",
        [
            {
                "sprite_id": "sprite_banana",
                "category": "item_icon",
                "object_name": "banana",
                "tags": ["banana", "fruit"],
                "short_description": "A banana icon.",
                "confidence": 0.83,
            }
        ],
    )
    write_jsonl(
        run / "fused_suggestions.jsonl",
        [
            {
                "sprite_id": "sprite_banana",
                "fused_suggestion": {
                    "category": "item_icon",
                    "object_name": "banana",
                    "tags": ["banana", "fruit", "food"],
                    "short_description": "A banana icon.",
                },
                "prefill_quality": {"bucket": "fused_automatically", "flags": []},
            }
        ],
    )
    return run


def _by_id(candidates: list[AssistedGoldenCandidate], sprite_id: str) -> AssistedGoldenCandidate:
    return next(candidate for candidate in candidates if candidate.sprite_id == sprite_id)


def _candidate_with_prefill() -> AssistedGoldenCandidate:
    return AssistedGoldenCandidate(
        sprite_id="s",
        final_png_path=Path("s.png"),
        suggested_category="weapon",
        suggested_object_name="axe",
        suggested_tags=("axe",),
        suggested_source="fusion",
    )
