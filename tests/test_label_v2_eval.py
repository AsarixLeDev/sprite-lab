from spritelab.harvest.golden import GoldenLabel
from spritelab.harvest.label_v2_eval import evaluate_label_v2, sweep_label_v2_operating_points


def test_label_v2_eval_counts_missing_and_scores_fields() -> None:
    golden = {
        "a": GoldenLabel("a", "item_icon", "apple", ("fruit", "food")),
        "b": GoldenLabel("b", "tool", "compass", ("tool",)),
    }
    records = [
        {
            "sprite_id": "a",
            "safe_prefill": {"category": "item_icon", "object_name": "apple", "tags": ["fruit"]},
            "label_quality": {"bucket": "auto_filename_trusted", "needs_review": False},
        }
    ]
    result = evaluate_label_v2(golden, records)
    assert result["missing_prediction_count"] == 1
    assert result["object_token_f1"] == 1.0
    assert result["tag_precision"] == 1.0
    assert result["auto_coverage"] == 0.5


def test_label_v2_sweep_prefers_highest_valid_coverage() -> None:
    golden = {
        "a": GoldenLabel("a", "item_icon", "apple", ("fruit",)),
        "b": GoldenLabel("b", "item_icon", "orange", ("fruit",)),
    }
    records = [
        {"sprite_id": "a", "safe_prefill": {"category": "item_icon", "object_name": "apple", "confidence": 0.96}, "filename_suggestion": {"confidence": 0.96}, "label_quality": {"bucket": "auto_filename_trusted", "needs_review": False}},
        {"sprite_id": "b", "safe_prefill": {"category": "item_icon", "object_name": "orange", "confidence": 0.86}, "filename_suggestion": {"confidence": 0.86}, "label_quality": {"bucket": "auto_filename_trusted", "needs_review": False}},
    ]
    sweep = sweep_label_v2_operating_points(golden, records, trusted_filename_thresholds=(0.85, 0.95), vlm_thresholds=(0.65,), conflict_policies=("auto_trusted_filename_conflicts",))
    assert sweep["best"]["trusted_filename_threshold"] == 0.85
    assert sweep["best"]["auto_coverage"] == 1.0


def test_label_v2_eval_reports_specificity_breakdowns() -> None:
    golden = {
        "a": GoldenLabel("a", "armor", "golden_chestplate", ("armor",)),
        "b": GoldenLabel("b", "item_icon", "red_potion", ("potion",)),
        "c": GoldenLabel("c", "material", "ruby", ("gem",)),
    }
    records = [
        {"sprite_id": "a", "safe_prefill": {"category": "armor", "object_name": "armor"}, "label_quality": {"bucket": "auto_prefix_family_trusted", "needs_review": False}},
        {"sprite_id": "b", "safe_prefill": {"category": "item_icon", "object_name": "red"}, "label_quality": {"bucket": "auto_prefix_family_trusted", "needs_review": False}},
        {"sprite_id": "c", "safe_prefill": {"category": "item_icon", "object_name": "ruby"}, "label_quality": {"bucket": "auto_prefix_family_trusted", "needs_review": False}},
    ]

    result = evaluate_label_v2(golden, records)

    assert result["object_exact_accuracy_by_bucket"]["auto_prefix_family_trusted"] == 1 / 3
    assert result["category_accuracy_by_bucket"]["auto_prefix_family_trusted"] == 2 / 3
    assert result["broad_to_specific_errors"] == 2
    assert result["broad_to_specific_error_patterns"]["armor->golden_chestplate"] == 1
    assert result["broad_to_specific_error_patterns"]["red->red_potion"] == 1
    assert result["near_miss_compound_errors"]["armor->golden_chestplate"] == 1
    assert result["category_mismatch_pairs"]["material->item_icon"] == 1
    assert result["specificity_gap_rate"] == 2 / 3
