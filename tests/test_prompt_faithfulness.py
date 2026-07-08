from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.training.generated_canonicalizer import (
    canonicalize_generated_rgba,
    write_generated_sprite_artifacts,
    write_generation_reports,
)
from spritelab.training.prompt_faithfulness import PromptFaithfulnessConfig, run_prompt_faithfulness


def _rgba(color: tuple[float, float, float]) -> np.ndarray:
    arr = np.zeros((32, 32, 4), dtype=np.float32)
    arr[8:24, 8:24, :3] = np.array(color, dtype=np.float32)
    arr[8:24, 8:24, 3] = 1.0
    return arr


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "ds"
    dataset.mkdir(exist_ok=True)
    alpha = np.zeros((2, 32, 32), dtype=np.uint8)
    alpha[:, 8:24, 8:24] = 255
    index = np.zeros((2, 32, 32), dtype=np.int16)
    index[:, 8:24, 8:24] = 1
    palette = np.zeros((2, 33, 3), dtype=np.uint8)
    palette[0, 1] = [255, 0, 0]
    palette[1, 1] = [240, 220, 60]
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index,
        role_map=np.zeros_like(index, dtype=np.uint8),
        palette=palette,
        palette_mask=np.ones((2, 33), dtype=bool),
        category_id=np.ones((2,), dtype=np.int64),
        sprite_id=np.array(["red_potion_src", "gold_sword_src"], dtype=np.str_),
    )
    rows = [
        {
            "sprite_id": "red_potion_src",
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": 0,
            "object_name": "red_potion",
            "base_object": "potion",
            "category": "item_icon",
        },
        {
            "sprite_id": "gold_sword_src",
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": 1,
            "object_name": "gold_sword",
            "base_object": "sword",
            "category": "weapon",
        },
    ]
    (dataset / "training_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return dataset


def _prompts(path: Path) -> Path:
    rows = [
        {
            "prompt_id": "red_potion",
            "prompt": "red potion",
            "category": "seen_object",
            "target_semantics": {"base_object": "potion", "attributes": {"colors": ["red"]}},
        },
        {
            "prompt_id": "gold_sword",
            "prompt": "gold sword",
            "category": "seen_object",
            "target_semantics": {"base_object": "sword", "attributes": {"colors": ["gold"]}},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _generated(tmp_path: Path) -> Path:
    out = tmp_path / "generated"
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_text("stub", encoding="utf-8")
    records = []
    for sample_id, prompt_id, prompt in (
        ("sample_000000", "red_potion", "red potion"),
        ("sample_000001", "gold_sword", "gold sword"),
    ):
        records.append(
            write_generated_sprite_artifacts(
                canonicalize_generated_rgba(_rgba((1.0, 0.0, 0.0))),
                out,
                sample_id,
                {"prompt_id": prompt_id, "prompt": prompt, "checkpoint": str(ckpt), "seed": 1},
            )
        )
    write_generation_reports(out_dir=out, records=records, config={}, contact_sheet=None)
    return out


def test_prompt_faithfulness_detects_repeated_silhouettes_color_failure_and_mapping(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    prompts = _prompts(tmp_path / "prompts.jsonl")
    generated = _generated(tmp_path)
    report = run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=generated,
            prompts=prompts,
            dataset=dataset,
            out=generated / "prompt_faithfulness_report.md",
            out_json=generated / "prompt_faithfulness_report.json",
        )
    )
    assert report["repeated_silhouette_rate"] == 1.0
    assert report["generic_potion_collapse_rate"] > 0.0
    assert report["color_prompts_failed"]
    assert (generated / "prompt_faithfulness_mapping.jsonl").is_file()
    rows = [
        json.loads(line)
        for line in (generated / "prompt_faithfulness_mapping.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["prompt_id"] == "red_potion"
    assert rows[0]["nearest_source_object_name"]


_DETERMINISTIC_METRIC_KEYS = (
    "nearest_source_category_consistency_rate",
    "category_consistency_rate",
    "color_consistency_rate",
    "repeated_silhouette_rate",
    "generic_blob_collapse_rate",
    "generic_potion_collapse_rate",
)


def _run(tmp_path: Path, out_name: str, **overrides: object) -> dict:
    dataset = _dataset(tmp_path)
    prompts = _prompts(tmp_path / "prompts.jsonl")
    generated = _generated(tmp_path)
    return run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=generated,
            prompts=prompts,
            dataset=dataset,
            out=generated / f"{out_name}.md",
            out_json=generated / f"{out_name}.json",
            **overrides,  # type: ignore[arg-type]
        )
    )


def test_prompt_faithfulness_is_deterministic_across_runs(tmp_path: Path) -> None:
    first = _run(tmp_path, "run_a", max_sources=0)
    second = _run(tmp_path, "run_b", max_sources=0)
    assert first["source_selection"]["source_candidate_hash"] == second["source_selection"]["source_candidate_hash"]
    for key in _DETERMINISTIC_METRIC_KEYS:
        assert first[key] == second[key]


def test_prompt_faithfulness_max_sources_is_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path, "cap_a", max_sources=1)
    second = _run(tmp_path, "cap_b", max_sources=1)
    assert first["source_selection"]["mode"] == "deterministic_first_n"
    assert first["source_selection"]["source_count_used"] == 1
    assert first["source_selection"]["source_candidate_hash"] == second["source_selection"]["source_candidate_hash"]
    for key in _DETERMINISTIC_METRIC_KEYS:
        assert first[key] == second[key]


def test_prompt_faithfulness_all_sources_uses_every_sprite(tmp_path: Path) -> None:
    report = _run(tmp_path, "all_sources", max_sources=0)
    selection = report["source_selection"]
    assert selection["mode"] == "all"
    assert selection["source_count_total"] == 2
    assert selection["source_count_used"] == 2
    candidate_ids = report["source_candidate_ids"]
    rows = [json.loads(line) for line in Path(candidate_ids).read_text(encoding="utf-8").splitlines()]
    assert {row["sprite_id"] for row in rows} == {"red_potion_src", "gold_sword_src"}


def test_prompt_faithfulness_overwrites_existing_report(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    prompts = _prompts(tmp_path / "prompts.jsonl")
    generated = _generated(tmp_path)
    out_json = generated / "overwrite.json"
    out_json.write_text('{"stale": true}', encoding="utf-8")
    run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=generated,
            prompts=prompts,
            dataset=dataset,
            out=generated / "overwrite.md",
            out_json=out_json,
            max_sources=0,
        )
    )
    on_disk = json.loads(out_json.read_text(encoding="utf-8"))
    assert "stale" not in on_disk
    assert on_disk["schema_version"]


def test_prompt_faithfulness_markdown_names_nearest_source_category(tmp_path: Path) -> None:
    _run(tmp_path, "wording", max_sources=0)
    markdown = (tmp_path / "generated" / "wording.md").read_text(encoding="utf-8")
    assert "Nearest-source category consistency" in markdown
    assert "Source candidate hash" in markdown


def test_prompt_faithfulness_handles_missing_prompt_metadata(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    generated = _generated(tmp_path)
    report = run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=generated,
            prompts=tmp_path / "missing.jsonl",
            dataset=dataset,
            out=generated / "missing_prompt_report.md",
            out_json=generated / "missing_prompt_report.json",
        )
    )
    assert report["sample_count"] == 2
    assert (generated / "missing_prompt_report.json").is_file()


_PRE_EXISTING_SCALAR_KEYS = (
    "schema_version",
    "generated",
    "sample_count",
    "source_selection",
    "nearest_source_category_consistency_rate",
    "category_consistency_rate",
    "color_consistency_rate",
    "shape_bbox_consistency_rate",
    "repeated_silhouette_rate",
    "nearest_neighbor_duplicate_rate",
    "generic_potion_collapse_rate",
    "generic_flame_collapse_rate",
    "generic_blob_collapse_rate",
)

_NEW_CI_KEYS = (
    "nearest_source_category_consistency_ci95",
    "category_consistency_ci95",
    "color_consistency_ci95",
    "repeated_silhouette_rate_ci95",
    "nearest_neighbor_duplicate_rate_ci95",
    "generic_potion_collapse_rate_ci95",
    "generic_blob_collapse_rate_ci95",
    "near_copy_rate",
    "near_copy_rate_ci95",
    "near_copy_criterion",
    "near_copy_distance_threshold",
)


def test_prompt_faithfulness_json_preserves_old_scalar_fields_and_adds_ci_fields(tmp_path: Path) -> None:
    report = _run(tmp_path, "ci_fields", max_sources=0)

    for key in _PRE_EXISTING_SCALAR_KEYS:
        assert key in report

    for key in _NEW_CI_KEYS:
        assert key in report

    assert report["category_consistency_ci95"] == report["nearest_source_category_consistency_ci95"]
    assert report["near_copy_rate"] == report["nearest_neighbor_duplicate_rate"]
    assert report["near_copy_distance_threshold"] is None

    for ci_key, rate_key in (
        ("category_consistency_ci95", "category_consistency_rate"),
        ("color_consistency_ci95", "color_consistency_rate"),
    ):
        ci = report[ci_key]
        rate = report[rate_key]
        if rate is None:
            assert ci is None
        else:
            assert isinstance(ci, list) and len(ci) == 2
            assert 0.0 <= ci[0] <= rate <= ci[1] <= 1.0

    nearest_summary = report["nearest_source_summary"]
    assert "p10_distance" in nearest_summary
    assert "mean_distance" in nearest_summary and "median_distance" in nearest_summary

    on_disk = json.loads((tmp_path / "generated" / "ci_fields.json").read_text(encoding="utf-8"))
    for key in (*_PRE_EXISTING_SCALAR_KEYS, *_NEW_CI_KEYS):
        assert key in on_disk
