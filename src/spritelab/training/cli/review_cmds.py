"""Generated output review and comparison commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_generated_qa(subparsers)
    _register_generated_review(subparsers)
    _register_source_match_review(subparsers)
    _register_dataset_framing_review(subparsers)
    _register_prompt_faithfulness(subparsers)
    _register_compare_generated_runs(subparsers)
    _register_project_generated_palette(subparsers)


def _register_generated_qa(subparsers: argparse._SubParsersAction) -> None:
    generated_qa = subparsers.add_parser("generated-qa", help="QA generated sprite sample folders.")
    generated_qa.add_argument("--generated", required=True, type=Path)
    generated_qa.add_argument("--error-on-fully-transparent", action="store_true")
    generated_qa.set_defaults(func=_run_generated_qa)


def _run_generated_qa(parsed: argparse.Namespace) -> None:
    from spritelab.training.generated_qa import qa_generated_sprites

    result = qa_generated_sprites(
        parsed.generated,
        error_on_fully_transparent=parsed.error_on_fully_transparent,
    )
    print(f"Generated samples: {result.sample_count}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Reports written to {parsed.generated}")
    if not result.ok:
        raise SystemExit(1)


def _register_generated_review(subparsers: argparse._SubParsersAction) -> None:
    generated_review = subparsers.add_parser(
        "generated-review",
        help="Deterministically review generated sprite sample structure.",
    )
    generated_review.add_argument("--generated", required=True, type=Path)
    generated_review.add_argument("--out", type=Path)
    generated_review.add_argument("--out-json", type=Path)
    generated_review.add_argument("--out-dir", type=Path)
    generated_review.add_argument("--group-by", choices=["category", "none"], default="none")
    generated_review.add_argument("--max-samples-per-sheet", type=int, default=64)
    generated_review.add_argument("--compare-raw-indexed", action="store_true")
    generated_review.add_argument("--strict", action="store_true")
    generated_review.set_defaults(func=_run_generated_review)


def _run_generated_review(parsed: argparse.Namespace) -> None:
    from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites

    result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=parsed.generated,
            out=parsed.out,
            out_json=parsed.out_json,
            out_dir=parsed.out_dir,
            group_by=parsed.group_by,
            max_samples_per_sheet=parsed.max_samples_per_sheet,
            compare_raw_indexed=parsed.compare_raw_indexed,
            strict=parsed.strict,
        )
    )
    print(f"Reviewed samples: {result.report['sample_count']}")
    print(f"Markdown report: {result.markdown_path}")
    print(f"JSON report: {result.json_path}")
    print(f"Contact sheets: {len(result.contact_sheets)}")
    if not result.ok:
        raise SystemExit(1)


def _register_source_match_review(subparsers: argparse._SubParsersAction) -> None:
    source_match = subparsers.add_parser("source-match-review", help="Review generated samples against source targets.")
    source_match.add_argument("--generated", required=True, type=Path)
    source_match.add_argument("--dataset", required=True, type=Path)
    source_match.add_argument("--training-manifest", required=True, type=Path)
    source_match.add_argument("--out", required=True, type=Path)
    source_match.add_argument("--out-json", type=Path)
    source_match.set_defaults(func=_run_source_match_review)


def _run_source_match_review(parsed: argparse.Namespace) -> None:
    from spritelab.training.cli._args import _parsed_config_kwargs
    from spritelab.training.source_match_review import SourceMatchReviewConfig, run_source_match_review

    report = run_source_match_review(SourceMatchReviewConfig(**_parsed_config_kwargs(parsed)))
    print(f"Matched sources: {report['matched_source_count']}/{report['sample_count']}")
    print(f"Mean alpha IoU: {report.get('mean_alpha_iou')}")


def _register_dataset_framing_review(subparsers: argparse._SubParsersAction) -> None:
    dataset_framing_review = subparsers.add_parser(
        "dataset-framing-review",
        help="Review source sprite framing in exported training datasets.",
    )
    dataset_framing_review.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    dataset_framing_review.add_argument("--out-dir", type=Path)
    dataset_framing_review.add_argument("--compare-generated", type=Path)
    dataset_framing_review.add_argument("--max-samples-per-sheet", type=int, default=512)
    dataset_framing_review.set_defaults(func=_run_dataset_framing_review)


def _run_dataset_framing_review(parsed: argparse.Namespace) -> None:
    from spritelab.training.dataset_framing_review import (
        DatasetFramingReviewConfig,
        review_dataset_framing,
    )

    result = review_dataset_framing(
        DatasetFramingReviewConfig(
            dataset_dir=parsed.dataset_dir,
            out_dir=parsed.out_dir,
            compare_generated=parsed.compare_generated,
            max_samples_per_sheet=parsed.max_samples_per_sheet,
        )
    )
    print(f"Reviewed source sprites: {result.report['sample_count']}")
    print(f"Markdown report: {result.markdown_path}")
    print(f"JSON report: {result.json_path}")
    print(f"Contact sheets: {len(result.contact_sheets)}")
    if not result.ok:
        raise SystemExit(1)


def _register_prompt_faithfulness(subparsers: argparse._SubParsersAction) -> None:
    prompt_faithfulness = subparsers.add_parser(
        "prompt-faithfulness",
        help="Run dataset-grounded prompt faithfulness diagnostics.",
    )
    prompt_faithfulness.add_argument("--generated", required=True, type=Path)
    prompt_faithfulness.add_argument("--prompts", type=Path)
    prompt_faithfulness.add_argument("--dataset", required=True, type=Path)
    prompt_faithfulness.add_argument("--out", required=True, type=Path)
    prompt_faithfulness.add_argument("--out-json", required=True, type=Path)
    prompt_faithfulness.add_argument("--max-sources", type=int, help="0 or negative means use all sources.")
    prompt_faithfulness.add_argument(
        "--source-selection",
        default="auto",
        choices=["auto", "all", "deterministic_first_n", "deterministic_balanced"],
        help="Deterministic source-candidate selection strategy.",
    )
    prompt_faithfulness.set_defaults(func=_run_prompt_faithfulness)


def _run_prompt_faithfulness(parsed: argparse.Namespace) -> None:
    from spritelab.training.cli._args import _parsed_config_kwargs
    from spritelab.training.prompt_faithfulness import PromptFaithfulnessConfig, run_prompt_faithfulness

    report = run_prompt_faithfulness(PromptFaithfulnessConfig(**_parsed_config_kwargs(parsed)))
    print(f"Prompt faithfulness samples: {report['sample_count']}")
    print(f"Repeated silhouette rate: {report.get('repeated_silhouette_rate')}")
    print(f"Outputs written to {parsed.out}")


def _register_compare_generated_runs(subparsers: argparse._SubParsersAction) -> None:
    compare_generated = subparsers.add_parser(
        "compare-generated-runs",
        help="Compare two generated sprite sample folders.",
    )
    compare_generated.add_argument("--a", required=True, type=Path)
    compare_generated.add_argument("--b", required=True, type=Path)
    compare_generated.add_argument("--out", required=True, type=Path, dest="out_dir")
    compare_generated.add_argument("--max-contact-sheet-pairs", type=int, default=64)
    compare_generated.set_defaults(func=_run_compare_generated_runs)


def _run_compare_generated_runs(parsed: argparse.Namespace) -> None:
    from spritelab.training.compare_generated_runs import CompareGeneratedRunsConfig, compare_generated_runs

    report = compare_generated_runs(
        CompareGeneratedRunsConfig(
            a=parsed.a,
            b=parsed.b,
            out_dir=parsed.out_dir,
            max_contact_sheet_pairs=parsed.max_contact_sheet_pairs,
        )
    )
    deltas = report["deltas"]
    print(f"A samples: {report['a']['sample_count']}")
    print(f"B samples: {report['b']['sample_count']}")
    print(f"Border-touch delta B-A: {deltas['border_touch_rate']:.6f}")
    print(f"Outputs written to {parsed.out_dir}")


def _register_project_generated_palette(subparsers: argparse._SubParsersAction) -> None:
    project_palette = subparsers.add_parser(
        "project-generated-palette",
        help="Project an existing generated folder to cleaner per-image palettes.",
    )
    project_palette.add_argument("--generated", required=True, type=Path)
    project_palette.add_argument("--out", required=True, type=Path)
    project_palette.add_argument("--target-colors", type=int, default=16)
    project_palette.add_argument("--min-pixel-share", type=float, default=0.01)
    project_palette.add_argument("--alpha-threshold", type=float, default=0.5)
    project_palette.add_argument("--method", choices=["deterministic_kmeans"], default="deterministic_kmeans")
    project_palette.set_defaults(func=_run_project_generated_palette)


def _run_project_generated_palette(parsed: argparse.Namespace) -> None:
    from spritelab.training.cli._args import _parsed_config_kwargs
    from spritelab.training.palette_projection import PaletteProjectionConfig, project_generated_palette

    report = project_generated_palette(PaletteProjectionConfig(**_parsed_config_kwargs(parsed)))
    print(f"Projected samples: {report['sample_count']}")
    print(
        "Median visible colors: "
        f"{report['median_visible_color_count_before']} -> {report['median_visible_color_count_after']}"
    )
    print(f"Mean RGB MAE visible: {report['mean_rgb_mae_visible']}")
    print(f"Outputs written to {parsed.out}")
