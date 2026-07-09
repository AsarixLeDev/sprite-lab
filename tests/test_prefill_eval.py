"""Tests for the prefill evaluation harness."""

from __future__ import annotations

from spritelab.harvest.golden import GoldenLabel
from spritelab.harvest.prefill_eval import evaluate_prefill, format_eval_report


def _record(
    sprite_id: str,
    *,
    category: str,
    object_name: str = "",
    tags=(),
    bucket: str = "fused_automatically",
    confidence=None,
    warnings=(),
) -> dict:
    suggestion = {
        "category": category,
        "object_name": object_name,
        "tags": list(tags),
        "confidence": confidence,
        "warnings": list(warnings),
    }
    return {
        "sprite_id": sprite_id,
        "qwen_suggestion": dict(suggestion),
        "fused_suggestion": dict(suggestion),
        "prefill_quality": {"bucket": bucket},
    }


def test_perfect_predictions() -> None:
    golden = {
        "a": GoldenLabel(sprite_id="a", category="plant", object_name="mushroom", tags=("mushroom",)),
        "b": GoldenLabel(sprite_id="b", category="weapon", object_name="sword", tags=("sword",)),
    }
    records = [
        _record("a", category="plant", object_name="mushroom", tags=("mushroom",)),
        _record("b", category="weapon", object_name="sword", tags=("sword",)),
    ]
    result = evaluate_prefill(golden, records)
    assert result["matched_count"] == 2
    assert result["qwen"]["category_accuracy"] == 1.0
    assert result["qwen"]["object_name_exact"] == 1.0
    assert result["qwen"]["tags_f1"] == 1.0
    assert result["auto_coverage"] == 1.0
    assert result["auto_precision"] == 1.0


def test_category_confusion_and_macro_f1() -> None:
    golden = {
        "a": GoldenLabel(sprite_id="a", category="plant"),
        "b": GoldenLabel(sprite_id="b", category="plant"),
        "c": GoldenLabel(sprite_id="c", category="weapon"),
    }
    records = [
        _record("a", category="plant"),
        _record("b", category="item_icon"),
        _record("c", category="weapon"),
    ]
    result = evaluate_prefill(golden, records)
    assert result["qwen"]["category_accuracy"] == 2 / 3
    assert result["qwen"]["category_confusion"] == {"plant->item_icon": 1}
    assert 0.0 < result["qwen"]["category_macro_f1"] < 1.0


def test_auto_precision_restricted_to_bucket() -> None:
    golden = {
        "a": GoldenLabel(sprite_id="a", category="plant"),
        "b": GoldenLabel(sprite_id="b", category="weapon"),
    }
    records = [
        _record("a", category="plant", bucket="fused_automatically"),
        _record("b", category="plant", bucket="needs_review"),  # wrong, but not auto
    ]
    result = evaluate_prefill(golden, records)
    assert result["auto_coverage"] == 0.5
    assert result["auto_precision"] == 1.0
    assert result["review_rate"] == 0.5


def test_unmatched_sprites_ignored() -> None:
    golden = {"a": GoldenLabel(sprite_id="a", category="plant"), "z": GoldenLabel(sprite_id="z", category="weapon")}
    records = [_record("a", category="plant"), _record("y", category="block")]
    result = evaluate_prefill(golden, records)
    assert result["matched_count"] == 1
    assert result["golden_count"] == 2
    assert result["record_count"] == 2


def test_calibration_and_ece() -> None:
    golden = {f"s{i}": GoldenLabel(sprite_id=f"s{i}", category="plant") for i in range(4)}
    records = [
        _record("s0", category="plant", confidence=0.9),
        _record("s1", category="weapon", confidence=0.9),
        _record("s2", category="plant", confidence=0.1),
        _record("s3", category="plant", confidence=None),
    ]
    result = evaluate_prefill(golden, records)
    calibration = result["calibration"]
    assert calibration["scored_count"] == 3
    assert calibration["ece"] is not None
    high_bin = next(entry for entry in calibration["bins"] if entry["bin"] == "0.85-1.00")
    assert high_bin["count"] == 2
    assert high_bin["accuracy"] == 0.5


def test_degenerate_rate_from_warnings() -> None:
    golden = {"a": GoldenLabel(sprite_id="a", category="plant"), "b": GoldenLabel(sprite_id="b", category="plant")}
    records = [
        _record("a", category="plant", warnings=("degenerate_response",)),
        _record("b", category="plant"),
    ]
    result = evaluate_prefill(golden, records)
    assert result["degenerate_rate"] == 0.5


def test_degenerate_rate_custom_check() -> None:
    golden = {"a": GoldenLabel(sprite_id="a", category="plant")}
    records = [_record("a", category="plant", object_name="mushroom")]
    result = evaluate_prefill(golden, records, degenerate_check=lambda s: s.get("object_name") == "mushroom")
    assert result["degenerate_rate"] == 1.0


def test_empty_inputs() -> None:
    result = evaluate_prefill({}, [])
    assert result["matched_count"] == 0
    assert result["qwen"]["category_accuracy"] == 0.0
    assert result["auto_coverage"] == 0.0


def test_format_eval_report_smoke() -> None:
    golden = {"a": GoldenLabel(sprite_id="a", category="plant", object_name="fern", tags=("fern",))}
    records = [_record("a", category="plant", object_name="fern", tags=("fern",), confidence=0.9)]
    text = format_eval_report(evaluate_prefill(golden, records))
    assert "category accuracy" in text
    assert "auto precision" in text
