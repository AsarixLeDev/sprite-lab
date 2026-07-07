from __future__ import annotations

import json
from pathlib import Path

from spritelab.training.compare_generator_families import (
    CompareGeneratorFamiliesConfig,
    compare_generator_families,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_compare_generator_families_aggregates_metrics_and_writes_recommendation(tmp_path: Path) -> None:
    baseline_run = tmp_path / "runs" / "baseline"
    challenger_run = tmp_path / "runs" / "challenger"
    baseline_generated = tmp_path / "generated" / "baseline"
    challenger_generated = tmp_path / "generated" / "challenger"
    _write_json(baseline_run / "train_report.json", {"final_train_loss": 0.5, "loss_decrease": 0.1})
    _write_json(challenger_run / "train_report.json", {"model_type": "generator_challenger", "final_train_loss": 0.4})
    _write_json(baseline_generated / "generated_qa_report.json", {"ok": True, "errors": [], "warnings": []})
    _write_json(challenger_generated / "generated_qa_report.json", {"ok": True, "errors": [], "warnings": []})
    _write_json(
        baseline_generated / "generated_review_report.json",
        {"overall": {"total_warnings": 4, "mean_alpha_coverage": 0.3, "mean_visible_color_count": 8}},
    )
    _write_json(
        challenger_generated / "generated_review_report.json",
        {"overall": {"total_warnings": 2, "mean_alpha_coverage": 0.35, "mean_visible_color_count": 12}},
    )
    _write_json(
        baseline_generated / "prompt_faithfulness_report.json",
        {"sample_count": 2, "repeated_silhouette_rate": 1.0, "color_consistency_rate": 0.3},
    )
    _write_json(
        challenger_generated / "prompt_faithfulness_report.json",
        {"sample_count": 2, "repeated_silhouette_rate": 0.2, "color_consistency_rate": 0.7},
    )
    _write_json(baseline_generated / "source_match_report.json", {"mean_visible_rgb_mae": 0.2, "mean_alpha_iou": 0.5})
    _write_json(challenger_generated / "source_match_report.json", {"mean_visible_rgb_mae": 0.1, "mean_alpha_iou": 0.8})

    report = compare_generator_families(
        CompareGeneratorFamiliesConfig(
            baseline_run=baseline_run,
            baseline_generated=baseline_generated,
            challenger_run=challenger_run,
            challenger_generated=challenger_generated,
            dataset=tmp_path / "ds",
            prompts=tmp_path / "prompts.jsonl",
            out_dir=tmp_path / "compare",
        )
    )
    assert report["baseline"]["qa_errors"] == 0
    assert report["challenger"]["source_match_mean_alpha_iou"] == 0.8
    assert report["recommendation"]
    assert (tmp_path / "compare" / "compare_generator_families_report.json").is_file()
    assert (tmp_path / "compare" / "compare_generator_families_report.md").is_file()


def test_compare_generator_families_handles_missing_optional_reports(tmp_path: Path) -> None:
    baseline_run = tmp_path / "runs" / "baseline"
    challenger_run = tmp_path / "runs" / "challenger"
    baseline_generated = tmp_path / "generated" / "baseline"
    challenger_generated = tmp_path / "generated" / "challenger"
    _write_json(baseline_run / "train_report.json", {"final_train_loss": 0.5})
    _write_json(challenger_run / "train_report.json", {"final_train_loss": 0.4})
    report = compare_generator_families(
        CompareGeneratorFamiliesConfig(
            baseline_run=baseline_run,
            baseline_generated=baseline_generated,
            challenger_run=challenger_run,
            challenger_generated=challenger_generated,
            dataset=tmp_path / "ds",
            prompts=tmp_path / "prompts.jsonl",
            out_dir=tmp_path / "compare_missing",
        )
    )
    assert report["warnings"]
    assert report["recommendation"]
