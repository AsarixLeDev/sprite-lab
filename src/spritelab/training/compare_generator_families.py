"""Compare regression-generator and challenger-generator experiment families."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "compare_generator_families_v1.0"


@dataclass(frozen=True)
class CompareGeneratorFamiliesConfig:
    baseline_run: Path
    baseline_generated: Path
    challenger_run: Path
    challenger_generated: Path
    dataset: Path
    prompts: Path
    out_dir: Path


def compare_generator_families(config: CompareGeneratorFamiliesConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _family_summary(config.baseline_run, config.baseline_generated, label="baseline")
    challenger = _family_summary(config.challenger_run, config.challenger_generated, label="challenger")
    warnings = [*baseline["warnings"], *challenger["warnings"]]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline,
        "challenger": challenger,
        "deltas": _deltas(baseline, challenger),
        "warnings": warnings,
        "recommendation": _recommendation(baseline, challenger),
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
    }
    (out_dir / "compare_generator_families_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "compare_generator_families_report.md").write_text(
        format_compare_families_markdown(report), encoding="utf-8"
    )
    return report


def format_compare_families_markdown(report: Mapping[str, Any]) -> str:
    baseline = report.get("baseline") if isinstance(report.get("baseline"), Mapping) else {}
    challenger = report.get("challenger") if isinstance(report.get("challenger"), Mapping) else {}
    deltas = report.get("deltas") if isinstance(report.get("deltas"), Mapping) else {}
    lines = [
        "# Generator Family Comparison",
        "",
        f"Recommendation: **{report.get('recommendation', '')}**",
        "",
        "## Summary",
        "",
        "| Metric | Regression baseline | Challenger | Delta challenger-baseline |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "final_train_loss",
        "qa_errors",
        "review_total_warnings",
        "prompt_sensitivity_same_noise_difference",
        "prompt_sensitivity_same_prompt_diversity",
        "faithfulness_repeated_silhouette_rate",
        "faithfulness_generic_potion_collapse_rate",
        "faithfulness_color_consistency_rate",
        "source_match_mean_visible_rgb_mae",
        "source_match_mean_alpha_iou",
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    _fmt(baseline.get(key)),
                    _fmt(challenger.get(key)),
                    _fmt(deltas.get(key)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Missing Optional Reports", ""])
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- (none)")
    lines.extend(["", "## Best/Worst Evidence", ""])
    for label, summary in (("Regression baseline", baseline), ("Challenger", challenger)):
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Run: `{summary.get('run_dir', '')}`")
        lines.append(f"- Generated: `{summary.get('generated_dir', '')}`")
        lines.append(f"- Prompt faithfulness samples: {int(summary.get('faithfulness_sample_count') or 0)}")
        lines.append("")
    return "\n".join(lines)


def _family_summary(run_dir: Path, generated_dir: Path, *, label: str) -> dict[str, Any]:
    warnings: list[str] = []
    train = _read_json(run_dir / "train_report.json", warnings, f"{label}: missing train_report.json")
    qa = _read_json(generated_dir / "generated_qa_report.json", warnings, f"{label}: missing generated_qa_report.json")
    review = _read_first_json(
        [generated_dir / "generated_review_report.json", generated_dir / "review" / "generated_review_report.json"],
        warnings,
        f"{label}: missing generated_review_report.json",
    )
    sensitivity = _read_first_json(
        [
            generated_dir / "prompt_sensitivity_report.json",
            generated_dir.parent / f"{generated_dir.name}_prompt_sensitivity" / "prompt_sensitivity_report.json",
            generated_dir.parent / f"{run_dir.name}_prompt_sensitivity" / "prompt_sensitivity_report.json",
        ],
        warnings,
        f"{label}: missing prompt_sensitivity_report.json",
    )
    faithfulness = _read_json(
        generated_dir / "prompt_faithfulness_report.json",
        warnings,
        f"{label}: missing prompt_faithfulness_report.json",
    )
    source_match = _read_first_json(
        [
            generated_dir / "source_match_report.json",
            generated_dir / "source_match" / "source_match_report.json",
            run_dir / "source_match_report.json",
        ],
        warnings,
        f"{label}: missing source_match_report.json",
    )
    overall = review.get("overall") if isinstance(review.get("overall"), Mapping) else {}
    same_noise = _sensitivity_metrics(sensitivity, "same_noise_different_prompts")
    same_prompt = _sensitivity_metrics(sensitivity, "same_prompt_different_noise")
    return {
        "label": label,
        "run_dir": str(run_dir),
        "generated_dir": str(generated_dir),
        "model_type": train.get("model_type") or train.get("architecture") or "regression_generator",
        "final_train_loss": _num(train.get("final_train_loss")),
        "loss_decrease": _num(train.get("loss_decrease")),
        "qa_ok": qa.get("ok") if isinstance(qa, Mapping) else None,
        "qa_errors": len(qa.get("errors") or []) if isinstance(qa, Mapping) else None,
        "qa_warnings": len(qa.get("warnings") or []) if isinstance(qa, Mapping) else None,
        "review_total_warnings": _num(overall.get("total_warnings")),
        "review_mean_alpha_coverage": _num(overall.get("mean_alpha_coverage")),
        "review_mean_visible_color_count": _num(overall.get("mean_visible_color_count")),
        "prompt_sensitivity_same_noise_difference": _num(same_noise.get("mean_pairwise_difference")),
        "prompt_sensitivity_same_prompt_diversity": _num(same_prompt.get("diversity_score")),
        "faithfulness_sample_count": int(faithfulness.get("sample_count") or 0)
        if isinstance(faithfulness, Mapping)
        else 0,
        "faithfulness_repeated_silhouette_rate": _num(faithfulness.get("repeated_silhouette_rate")),
        "faithfulness_generic_potion_collapse_rate": _num(faithfulness.get("generic_potion_collapse_rate")),
        "faithfulness_color_consistency_rate": _num(faithfulness.get("color_consistency_rate")),
        "source_match_mean_visible_rgb_mae": _num(source_match.get("mean_visible_rgb_mae")),
        "source_match_mean_alpha_iou": _num(source_match.get("mean_alpha_iou")),
        "raw_reports_present": {
            "train": bool(train),
            "qa": bool(qa),
            "review": bool(review),
            "prompt_sensitivity": bool(sensitivity),
            "prompt_faithfulness": bool(faithfulness),
            "source_match": bool(source_match),
        },
        "warnings": warnings,
    }


def _recommendation(baseline: Mapping[str, Any], challenger: Mapping[str, Any]) -> str:
    challenger_errors = challenger.get("qa_errors")
    if isinstance(challenger_errors, (int, float)) and int(challenger_errors) > 0:
        return "fix challenger canonicalization or sampling before model-quality comparison"
    base_mae = baseline.get("source_match_mean_visible_rgb_mae")
    chal_mae = challenger.get("source_match_mean_visible_rgb_mae")
    if _is_number(base_mae) and _is_number(chal_mae) and float(chal_mae) < float(base_mae) * 0.9:
        return "continue challenger architecture"
    base_div = baseline.get("prompt_sensitivity_same_prompt_diversity")
    chal_div = challenger.get("prompt_sensitivity_same_prompt_diversity")
    if _is_number(base_div) and _is_number(chal_div) and float(chal_div) > float(base_div) * 1.2:
        return "continue challenger architecture"
    base_color = baseline.get("faithfulness_color_consistency_rate")
    chal_color = challenger.get("faithfulness_color_consistency_rate")
    if _is_number(base_color) and _is_number(chal_color) and float(chal_color) <= float(base_color):
        return "improve conditioning injection before scaling"
    if not challenger.get("raw_reports_present", {}).get("prompt_faithfulness"):
        return "run prompt faithfulness before choosing the next modeling pass"
    return "increase challenger capacity or tune losses with regression kept as baseline"


def _deltas(baseline: Mapping[str, Any], challenger: Mapping[str, Any]) -> dict[str, Any]:
    keys = sorted(set(baseline) | set(challenger))
    result: dict[str, Any] = {}
    for key in keys:
        left = baseline.get(key)
        right = challenger.get(key)
        if _is_number(left) and _is_number(right):
            result[key] = float(right) - float(left)
    return result


def _sensitivity_metrics(report: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    sets = report.get("sets") if isinstance(report.get("sets"), Mapping) else {}
    item = sets.get(key) if isinstance(sets.get(key), Mapping) else {}
    return item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}


def _read_first_json(paths: list[Path], warnings: list[str], warning: str) -> dict[str, Any]:
    for path in paths:
        value = _read_json(path, [], "")
        if value:
            return value
    warnings.append(warning)
    return {}


def _read_json(path: Path, warnings: list[str], warning: str) -> dict[str, Any]:
    if not path.is_file():
        if warning:
            warnings.append(warning)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        warnings.append(f"invalid JSON: {path}")
        return {}
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.generic)) and math_isfinite(float(value))


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


from spritelab.training.report_utils import fmt_float as _fmt
from spritelab.training.report_utils import jsonable as _jsonable


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare regression and challenger generator families.")
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--baseline-generated", required=True, type=Path)
    parser.add_argument("--challenger-run", required=True, type=Path)
    parser.add_argument("--challenger-generated", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parsed = parser.parse_args(argv)
    report = compare_generator_families(CompareGeneratorFamiliesConfig(**vars(parsed)))
    print(f"Recommendation: {report['recommendation']}")
    print(f"Outputs written to {parsed.out_dir}")


if __name__ == "__main__":
    main()
