from spritelab.harvest.golden import GoldenLabel
from spritelab.harvest.label_v2_eval import evaluate_label_v2, label_v2_error_records


def _record(
    sprite_id: str, category: str, object_name: str, tags=(), bucket: str = "auto_prefix_family_trusted"
) -> dict:
    return {
        "sprite_id": sprite_id,
        "safe_prefill": {"category": category, "object_name": object_name, "tags": list(tags)},
        "label_quality": {"bucket": bucket, "needs_review": bucket.startswith("needs_review")},
    }


def test_error_class_for_same_object_low_tag_f1_is_tag_only() -> None:
    golden = {"a": GoldenLabel("a", "item_icon", "apple", ("fruit", "food"))}
    records = [_record("a", "item_icon", "apple", ("red",))]

    errors = label_v2_error_records(golden, records)

    assert errors[0]["error_class"] == "tag_only_mismatch"


def test_error_class_for_broad_to_specific_miss() -> None:
    golden = {"a": GoldenLabel("a", "weapon", "gray_bow", ("weapon",))}
    records = [_record("a", "weapon", "bow", ("weapon",))]

    errors = label_v2_error_records(golden, records)

    assert errors[0]["error_class"] == "broad_to_specific_miss"


def test_error_class_for_over_specific_prediction() -> None:
    golden = {"a": GoldenLabel("a", "weapon", "bow", ("weapon",))}
    records = [_record("a", "weapon", "explosive_arrow", ("weapon",))]

    errors = label_v2_error_records(golden, records)

    assert errors[0]["error_class"] == "over_specific_prediction"


def test_error_class_for_category_mismatch_wins() -> None:
    golden = {"a": GoldenLabel("a", "material", "ruby", ("gem",))}
    records = [_record("a", "item_icon", "ruby", ("gem",))]

    errors = label_v2_error_records(golden, records)

    assert errors[0]["error_class"] == "category_mismatch"


def test_errors_mode_hard_excludes_tag_only_mismatches() -> None:
    golden = {
        "a": GoldenLabel("a", "item_icon", "apple", ("fruit", "food")),
        "b": GoldenLabel("b", "weapon", "gray_bow", ("weapon",)),
    }
    records = [
        _record("a", "item_icon", "apple", ("red",)),
        _record("b", "weapon", "bow", ("weapon",)),
    ]

    errors = label_v2_error_records(golden, records, errors_mode="hard")

    assert [error["sprite_id"] for error in errors] == ["b"]
    assert errors[0]["error_class"] == "broad_to_specific_miss"


def test_eval_summary_counts_error_classes() -> None:
    golden = {
        "tag": GoldenLabel("tag", "item_icon", "apple", ("fruit", "food")),
        "broad": GoldenLabel("broad", "weapon", "gray_bow", ("weapon",)),
        "over": GoldenLabel("over", "weapon", "bow", ("weapon",)),
        "category": GoldenLabel("category", "material", "ruby", ("gem",)),
    }
    records = [
        _record("tag", "item_icon", "apple", ("red",)),
        _record("broad", "weapon", "bow", ("weapon",)),
        _record("over", "weapon", "explosive_arrow", ("weapon",)),
        _record("category", "item_icon", "ruby", ("gem",)),
    ]

    result = evaluate_label_v2(golden, records)

    assert result["error_classes"]["tag_only_mismatch"] == 1
    assert result["error_classes"]["broad_to_specific_miss"] == 1
    assert result["error_classes"]["over_specific_prediction"] == 1
    assert result["error_classes"]["category_mismatch"] == 1
    assert result["hard_object_errors"] == 3
    assert result["tag_only_errors"] == 1
    assert result["category_errors"] == 1
