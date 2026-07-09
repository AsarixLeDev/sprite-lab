"""Data inspection and palette-swap review commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_inspect_data(subparsers)
    _register_make_overfit_subset(subparsers)
    _register_dataset_palette_swap_review(subparsers)


def _register_inspect_data(subparsers: argparse._SubParsersAction) -> None:
    inspect = subparsers.add_parser("inspect-data", help="Inspect training manifest and npz tensor contract.")
    inspect.add_argument("--dataset", required=True, type=Path)
    inspect.add_argument("--training-manifest", required=True, type=Path)
    inspect.add_argument("--split")
    inspect.add_argument("--batch-size", type=int, default=4)
    inspect.add_argument("--max-records", type=int)
    inspect.set_defaults(func=_run_inspect_data)


def _run_inspect_data(parsed: argparse.Namespace) -> None:
    from spritelab.training.inspect_data import inspect_training_data, print_inspection

    summary = inspect_training_data(
        dataset_dir=parsed.dataset,
        training_manifest=parsed.training_manifest,
        split=parsed.split,
        batch_size=parsed.batch_size,
        max_records=parsed.max_records,
    )
    print_inspection(summary)


def _register_make_overfit_subset(subparsers: argparse._SubParsersAction) -> None:
    make_overfit = subparsers.add_parser("make-overfit-subset", help="Write a deterministic sprite-id subset.")
    make_overfit.add_argument("--dataset", required=True, type=Path)
    make_overfit.add_argument("--training-manifest", required=True, type=Path)
    make_overfit.add_argument("--out", required=True, type=Path)
    make_overfit.add_argument("--count", required=True, type=int)
    make_overfit.add_argument("--seed", type=int, default=123)
    make_overfit.add_argument("--split", default="train")
    make_overfit.add_argument("--stratify")
    make_overfit.set_defaults(func=_run_make_overfit_subset)


def _run_make_overfit_subset(parsed: argparse.Namespace) -> None:
    from spritelab.training.overfit_subset import make_overfit_subset

    report = make_overfit_subset(
        dataset=parsed.dataset,
        training_manifest=parsed.training_manifest,
        out=parsed.out,
        count=parsed.count,
        seed=parsed.seed,
        split=parsed.split,
        stratify=parsed.stratify,
    )
    print(f"Selected sprites: {report['selected_sprite_count']}")
    print(f"Selected rows: {report['selected_row_count']}")
    print(f"Output written to {parsed.out}")


def _register_dataset_palette_swap_review(subparsers: argparse._SubParsersAction) -> None:
    from spritelab.training.cli._args import _bool_str
    from spritelab.training.palette_swap import DEFAULT_SWAP_FAMILIES_TEXT

    palette_swap_review = subparsers.add_parser(
        "dataset-palette-swap-review",
        help="Sample and inspect palette-swap augmentation without training.",
    )
    palette_swap_review.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    palette_swap_review.add_argument("--training-manifest", required=True, type=Path)
    palette_swap_review.add_argument("--out-dir", required=True, type=Path)
    palette_swap_review.add_argument("--seed", type=int, default=20260706)
    palette_swap_review.add_argument("--max-samples", type=int, default=256)
    palette_swap_review.add_argument("--draws-per-sprite", type=int, default=1)
    palette_swap_review.add_argument(
        "--review-selection",
        choices=["first", "random", "balanced", "all"],
        default="first",
    )
    palette_swap_review.add_argument("--palette-swap-prob", type=float, default=0.5)
    palette_swap_review.add_argument("--palette-swap-families", default=DEFAULT_SWAP_FAMILIES_TEXT)
    palette_swap_review.add_argument("--palette-swap-target-families", default=None)
    palette_swap_review.add_argument("--palette-swap-source-families", default=None)
    palette_swap_review.add_argument("--palette-swap-category-filter", default=None)
    palette_swap_review.add_argument("--palette-swap-min-color-confidence", type=float, default=0.0)
    palette_swap_review.add_argument("--palette-swap-stochastic", action="store_true", default=False)
    palette_swap_review.add_argument("--palette-swap-keep-original-prob", type=float, default=0.0)
    palette_swap_review.add_argument("--palette-swap-require-role-map", action="store_true", default=False)
    palette_swap_review.add_argument("--palette-swap-require-explicit-color", action="store_true", default=False)
    palette_swap_review.add_argument(
        "--palette-swap-require-explicit-caption-color", action="store_true", default=False
    )
    palette_swap_review.add_argument(
        "--palette-swap-require-explicit-semantic-color", action="store_true", default=False
    )
    palette_swap_review.add_argument(
        "--palette-swap-allow-colorless-caption-if-semantic-color", action="store_true", default=False
    )
    palette_swap_review.add_argument("--palette-swap-no-caption-prepend", action="store_true", default=False)
    palette_swap_review.add_argument("--palette-swap-allow-material-colors", type=_bool_str, default=True)
    palette_swap_review.add_argument(
        "--palette-swap-preserve-outline", action="store_true", default=True, dest="palette_swap_preserve_outline"
    )
    palette_swap_review.add_argument(
        "--no-palette-swap-preserve-outline", action="store_false", dest="palette_swap_preserve_outline"
    )
    palette_swap_review.add_argument(
        "--palette-swap-update-prompts", action="store_true", default=True, dest="palette_swap_update_prompts"
    )
    palette_swap_review.add_argument(
        "--no-palette-swap-update-prompts", action="store_false", dest="palette_swap_update_prompts"
    )
    palette_swap_review.set_defaults(func=_run_dataset_palette_swap_review)


def _run_dataset_palette_swap_review(parsed: argparse.Namespace) -> None:
    from spritelab.training.cli._args import _parsed_config_kwargs
    from spritelab.training.palette_swap_review import (
        PaletteSwapReviewConfig,
        review_palette_swap,
    )

    result = review_palette_swap(PaletteSwapReviewConfig(**_parsed_config_kwargs(parsed)))
    metrics = result.report["metrics"]
    print(f"Evaluated samples: {metrics['sample_count']}")
    print(f"Applied: {metrics['applied_count']} (rate {metrics['applied_rate']:.4f})")
    print(f"Red flags: {len(result.report['red_flags'])}")
    print(f"Markdown report: {result.markdown_path}")
    print(f"Contact sheets: {len(result.contact_sheets)}")
