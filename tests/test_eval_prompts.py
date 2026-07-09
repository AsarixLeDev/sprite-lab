from __future__ import annotations

from pathlib import Path

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.eval_prompts import (
    build_eval_prompts,
    summarize_eval_prompts,
    write_eval_prompts,
)


def _dataset(tmp_path: Path) -> Path:
    return make_semantic_dataset(tmp_path / "ds", default_specs())


def test_generates_all_prompt_categories(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_eval_prompts(dataset, seed=4962026, seen_object_count=5, unseen_composition_count=10)
    categories = {p["category"] for p in result.prompts}
    assert {"seen_object", "unseen_composition", "creative_concept", "style_stress", "negative_control"} <= categories


def test_seen_object_prompts_reference_real_objects(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_eval_prompts(dataset, seed=1, seen_object_count=6, unseen_composition_count=0)
    seen = [p for p in result.prompts if p["category"] == "seen_object"]
    assert seen
    for prompt in seen:
        assert prompt["prompt"].strip()
        assert prompt["target_semantics"]["base_object"]
        assert prompt["seen_factors"]["exact_combination_seen"] is True


def test_unseen_composition_factors_seen_but_combination_not(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_eval_prompts(dataset, seed=1, seen_object_count=0, unseen_composition_count=10)
    unseen = [p for p in result.prompts if p["category"] == "unseen_composition"]
    assert unseen
    for prompt in unseen:
        factors = prompt["seen_factors"]
        assert factors["base_object"] is True
        assert factors["color"] is True
        assert factors["exact_combination_seen"] is False
        assert prompt["target_semantics"]["base_object"]


def test_creative_concept_prompts_use_grammar(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_eval_prompts(dataset, seed=1, seen_object_count=0, unseen_composition_count=0)
    creative = [p for p in result.prompts if p["category"] == "creative_concept"]
    assert creative
    by_prompt = {p["prompt"]: p for p in creative}
    assert "charged sinew" in by_prompt
    sinew = by_prompt["charged sinew"]
    assert sinew["target_semantics"]["base_object"] == "sinew"
    effects = sinew["target_semantics"]["attributes"].get("effects", [])
    assert "electric" in effects or "charged" in effects


def test_target_semantics_present_for_all(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_eval_prompts(dataset, seed=1)
    for prompt in result.prompts:
        assert "target_semantics" in prompt
        assert "seen_factors" in prompt
        assert "prompt_id" in prompt


def test_deterministic_with_seed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    a = build_eval_prompts(dataset, seed=99)
    b = build_eval_prompts(dataset, seed=99)
    assert [p["prompt"] for p in a.prompts] == [p["prompt"] for p in b.prompts]


def test_writes_jsonl_and_summary(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_eval_prompts(dataset, seed=1)
    out = dataset / "eval_prompts.jsonl"
    write_eval_prompts(out, result.prompts)
    assert out.is_file()
    summary = summarize_eval_prompts(result)
    assert summary["total_prompts"] == len(result.prompts)
    assert "seen_object" in summary["category_counts"]
