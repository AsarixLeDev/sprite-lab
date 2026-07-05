import json

from _harvest_testdata import make_sprite_png

from spritelab.harvest.assisted_golden import build_assisted_golden_label, load_assisted_candidates
from spritelab.harvest.assisted_golden_gui import accept_as_is, make_initial_state


def test_assisted_golden_uses_safe_prefill_over_old_fused(tmp_path) -> None:
    run = tmp_path / "run"
    image = make_sprite_png(run / "butter.png")
    imported = {
        "sprite_id": "butter",
        "candidate_id": "c",
        "source_id": "oga_cc0_food_ocal",
        "source_name": "Food",
        "final_png_path": str(image),
        "relative_path": "butter.png",
        "status": "accepted",
        "category": "unknown",
        "tags": [],
        "license": "cc0",
        "author": "Tester",
        "auto_metadata": {"fused_suggestion": {"category": "material", "object_name": "gold_bar", "tags": ["gold"]}},
    }
    (run / "imported.jsonl").write_text(json.dumps(imported) + "\n", encoding="utf-8")
    (run / "rejected.jsonl").write_text("", encoding="utf-8")
    label_v2 = {
        "sprite_id": "butter",
        "filename_suggestion": {"category": "item_icon", "object_name": "butter", "tags": ["butter", "food"]},
        "vlm_descriptor": {"category": "material", "object_name": "gold_bar", "tags": ["gold"]},
        "safe_prefill": {"category": "item_icon", "object_name": "butter", "tags": ["butter", "food"]},
        "label_quality": {"bucket": "auto_filename_with_vlm_conflict", "needs_review": False, "flags": ["vlm_conflicts_with_filename"], "review_priority": 0.25},
    }
    (run / "label_v2_suggestions.jsonl").write_text(json.dumps(label_v2) + "\n", encoding="utf-8")

    candidate = load_assisted_candidates(run)[0]
    assert candidate.suggested_object_name == "butter"
    assert candidate.suggested_source == "safe_prefill"
    assert candidate.quality_bucket == "auto_filename_with_vlm_conflict"

    state = make_initial_state([candidate], labels={})
    _, label = accept_as_is(state)
    assert label is not None
    assert label.object_name == "butter"
    corrected = build_assisted_golden_label(candidate, category="item_icon", object_name="cheese", tags=("food",), notes="ambiguous")
    assert corrected.prefill_was_corrected
    assert set(corrected.correction_fields) >= {"object_name", "tags", "notes"}
