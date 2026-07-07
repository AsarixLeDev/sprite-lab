"""CLI for semantic training baselines."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from spritelab.training.conditioning import CONDITIONING_MODES, DEFAULT_CONDITIONING_MODE
from spritelab.training.palette_swap import DEFAULT_SWAP_FAMILIES_TEXT


def _parsed_config_kwargs(parsed: argparse.Namespace) -> dict[str, object]:
    values = vars(parsed).copy()
    values.pop("subcommand", None)
    return values


def _add_palette_swap_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared training-time palette-swap augmentation options."""

    parser.add_argument(
        "--palette-swap-augmentation",
        action="store_true",
        default=False,
        dest="palette_swap_augmentation",
        help="Enable deterministic palette-swap augmentation on indexed sprites.",
    )
    parser.add_argument("--palette-swap-prob", type=float, default=0.0, dest="palette_swap_prob")
    parser.add_argument(
        "--palette-swap-families",
        default=DEFAULT_SWAP_FAMILIES_TEXT,
        dest="palette_swap_families",
        help="Comma-separated target color families for augmentation.",
    )
    parser.add_argument(
        "--palette-swap-preserve-outline",
        action="store_true",
        default=True,
        dest="palette_swap_preserve_outline",
    )
    parser.add_argument(
        "--no-palette-swap-preserve-outline",
        action="store_false",
        dest="palette_swap_preserve_outline",
    )
    parser.add_argument(
        "--palette-swap-update-prompts",
        action="store_true",
        default=True,
        dest="palette_swap_update_prompts",
    )
    parser.add_argument(
        "--no-palette-swap-update-prompts",
        action="store_false",
        dest="palette_swap_update_prompts",
    )
    _add_palette_swap_conservative_arguments(parser)


def _bool_str(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _add_palette_swap_conservative_arguments(parser: argparse.ArgumentParser) -> None:
    """Conservative palette-swap filters (defaults preserve permissive behavior)."""

    parser.add_argument(
        "--palette-swap-target-families",
        default=None,
        dest="palette_swap_target_families",
        help="Restrict target color families (overrides --palette-swap-families).",
    )
    parser.add_argument(
        "--palette-swap-source-families",
        default=None,
        dest="palette_swap_source_families",
        help="Only augment sprites whose detected source family is in this set.",
    )
    parser.add_argument(
        "--palette-swap-category-filter",
        default=None,
        dest="palette_swap_category_filter",
        help="Only augment sprites in these categories.",
    )
    parser.add_argument(
        "--palette-swap-min-color-confidence",
        type=float,
        default=0.0,
        dest="palette_swap_min_color_confidence",
    )
    parser.add_argument(
        "--palette-swap-require-role-map",
        action="store_true",
        default=False,
        dest="palette_swap_require_role_map",
    )
    parser.add_argument(
        "--palette-swap-require-explicit-color",
        action="store_true",
        default=False,
        dest="palette_swap_require_explicit_color",
        help="Compatibility alias that requires both explicit caption and semantic colors.",
    )
    parser.add_argument(
        "--palette-swap-require-explicit-caption-color",
        action="store_true",
        default=False,
        dest="palette_swap_require_explicit_caption_color",
        help="Only augment samples whose caption/prompt text contains a known color token.",
    )
    parser.add_argument(
        "--palette-swap-require-explicit-semantic-color",
        action="store_true",
        default=False,
        dest="palette_swap_require_explicit_semantic_color",
        help="Only augment samples whose structured color fields contain a known color token.",
    )
    parser.add_argument(
        "--palette-swap-no-caption-prepend",
        action="store_true",
        default=False,
        dest="palette_swap_no_caption_prepend",
        help="Do not invent a color token by prepending it to colorless captions.",
    )
    parser.add_argument(
        "--palette-swap-allow-material-colors",
        type=_bool_str,
        default=True,
        dest="palette_swap_allow_material_colors",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m spritelab train",
        description="Semantic-manifest training inspection, baseline training, generator training, and evaluation.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    inspect = subparsers.add_parser("inspect-data", help="Inspect training manifest and npz tensor contract.")
    inspect.add_argument("--dataset", required=True, type=Path)
    inspect.add_argument("--training-manifest", required=True, type=Path)
    inspect.add_argument("--split")
    inspect.add_argument("--batch-size", type=int, default=4)
    inspect.add_argument("--max-records", type=int)

    baseline = subparsers.add_parser("baseline", help="Train the conditional autoencoder baseline.")
    baseline.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    baseline.add_argument("--training-manifest", required=True, type=Path)
    baseline.add_argument("--out", required=True, type=Path, dest="out_dir")
    baseline.add_argument("--batch-size", type=int, default=16)
    baseline.add_argument("--max-steps", type=int, default=200)
    baseline.add_argument("--learning-rate", type=float, default=1e-3)
    baseline.add_argument("--device", default="cpu")
    baseline.add_argument("--seed", type=int, default=1337)
    baseline.add_argument("--overfit-batches", type=int, default=0)
    baseline.add_argument("--max-records", type=int)
    baseline.add_argument("--num-workers", type=int, default=0)
    baseline.add_argument("--amp", action="store_true", default=False, help="Enable bf16 autocast (CUDA only).")
    baseline.add_argument("--grad-clip", type=float, default=0.0, help="Clip gradient norm; 0 disables.")
    baseline.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    baseline.add_argument("--lr-warmup-steps", type=int, default=0)

    generator = subparsers.add_parser("generator", help="Train the caption-conditioned RGBA generator.")
    generator.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    generator.add_argument("--training-manifest", required=True, type=Path)
    generator.add_argument("--out", required=True, type=Path, dest="out_dir")
    generator.add_argument("--split", default="train")
    generator.add_argument("--batch-size", type=int, default=32)
    generator.add_argument("--max-steps", type=int, default=1000)
    generator.add_argument("--lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    generator.add_argument("--device", default="cpu")
    generator.add_argument("--seed", type=int, default=123)
    generator.add_argument("--overfit-batches", type=int, default=0)
    generator.add_argument("--num-workers", type=int, default=0)
    generator.add_argument("--latent-dim", type=int, default=32)
    generator.add_argument("--embed-dim", type=int, default=32)
    generator.add_argument("--hidden-channels", type=int, default=48)
    generator.add_argument("--sample-every", type=int, default=20)
    generator.add_argument("--save-every", type=int, default=100)
    generator.add_argument("--caption-policy-filter")
    generator.add_argument("--max-records", type=int)
    generator.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default=DEFAULT_CONDITIONING_MODE)
    generator.add_argument("--border-alpha-weight", type=float, default=0.0)
    generator.add_argument("--alpha-coverage-weight", type=float, default=0.0)
    generator.add_argument("--alpha-coverage-min", type=float)
    generator.add_argument("--alpha-coverage-max", type=float)
    generator.add_argument("--center-weight", type=float, default=0.0)
    generator.add_argument("--margin-band-weight", type=float, default=0.0)
    generator.add_argument("--margin-band-size", type=int, default=2)
    generator.add_argument("--max-train-sprites", type=int)
    generator.add_argument("--sprite-id-list", type=Path)
    generator.add_argument("--overfit-split")
    generator.add_argument("--validation-mode", choices=["auto", "val", "same", "none"], default="auto")
    generator.add_argument("--amp", action="store_true", default=False, help="Enable bf16 autocast (CUDA only).")
    generator.add_argument("--grad-clip", type=float, default=0.0, help="Clip gradient norm; 0 disables.")
    generator.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    generator.add_argument("--lr-warmup-steps", type=int, default=0)

    make_overfit = subparsers.add_parser("make-overfit-subset", help="Write a deterministic sprite-id subset.")
    make_overfit.add_argument("--dataset", required=True, type=Path)
    make_overfit.add_argument("--training-manifest", required=True, type=Path)
    make_overfit.add_argument("--out", required=True, type=Path)
    make_overfit.add_argument("--count", required=True, type=int)
    make_overfit.add_argument("--seed", type=int, default=123)
    make_overfit.add_argument("--split", default="train")
    make_overfit.add_argument("--stratify")

    eval_parser = subparsers.add_parser("eval-baseline", help="Evaluate a baseline checkpoint.")
    eval_parser.add_argument("--dataset", required=True, type=Path)
    eval_parser.add_argument("--training-manifest", required=True, type=Path)
    eval_parser.add_argument("--checkpoint", required=True, type=Path)
    eval_parser.add_argument("--split", default="val")
    eval_parser.add_argument("--out", required=True, type=Path)
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--max-records", type=int)

    eval_generator = subparsers.add_parser("eval-generator", help="Evaluate or sample a generator checkpoint.")
    eval_generator.add_argument("--dataset", type=Path)
    eval_generator.add_argument("--training-manifest", type=Path)
    eval_generator.add_argument("--checkpoint", required=True, type=Path)
    eval_generator.add_argument("--split", default="val")
    eval_generator.add_argument("--prompts", type=Path)
    eval_generator.add_argument("--out", required=True, type=Path)
    eval_generator.add_argument("--batch-size", type=int, default=16)
    eval_generator.add_argument("--device", default="cpu")
    eval_generator.add_argument("--max-records", type=int)

    sample_generator = subparsers.add_parser("sample-generator", help="Generate and canonicalize sprite samples.")
    sample_generator.add_argument("--checkpoint", required=True, type=Path)
    sample_generator.add_argument("--prompts", required=True, type=Path)
    sample_generator.add_argument("--out", required=True, type=Path, dest="out_dir")
    sample_generator.add_argument("--max-samples", type=int, default=64)
    sample_generator.add_argument("--max-colors", type=int, default=32)
    sample_generator.add_argument("--alpha-threshold", type=float, default=0.5)
    sample_generator.add_argument("--device", default="cpu")
    sample_generator.add_argument("--seed", type=int, default=123)
    sample_generator.add_argument("--noise-seed", type=int)
    sample_generator.add_argument("--dither", action="store_true", default=False)
    sample_generator.add_argument("--no-dither", action="store_false", dest="dither")
    sample_generator.add_argument("--write-raw-rgba", action="store_true", dest="write_raw_rgba", default=True)
    sample_generator.add_argument("--no-write-raw-rgba", action="store_false", dest="write_raw_rgba")
    sample_generator.add_argument("--write-hard-rgba", action="store_true", dest="write_hard_rgba", default=True)
    sample_generator.add_argument("--no-write-hard-rgba", action="store_false", dest="write_hard_rgba")
    sample_generator.add_argument("--batch-size", type=int, default=16)
    sample_generator.add_argument(
        "--contact-sheet-labels",
        choices=["prompt", "prompt_and_seed", "prompt_and_nearest_source"],
        default="prompt",
    )

    challenger = subparsers.add_parser("generator-challenger", help="Train a conditional generator challenger.")
    challenger.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    challenger.add_argument("--training-manifest", required=True, type=Path)
    challenger.add_argument("--out", required=True, type=Path, dest="out_dir")
    challenger.add_argument("--architecture", default="rectified_flow")
    challenger.add_argument("--split", default="train")
    challenger.add_argument("--batch-size", type=int, default=32)
    challenger.add_argument("--max-steps", type=int, default=5000)
    challenger.add_argument("--lr", "--learning-rate", type=float, default=2e-4, dest="learning_rate")
    challenger.add_argument("--device", default="cpu")
    challenger.add_argument("--seed", type=int, default=123)
    challenger.add_argument("--num-workers", type=int, default=0)
    challenger.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default=DEFAULT_CONDITIONING_MODE)
    challenger.add_argument("--cfg-dropout", type=float, default=0.1)
    challenger.add_argument("--structured-field-dropout", type=float, default=0.0)
    challenger.add_argument("--ema-decay", type=float, default=0.999)
    challenger.add_argument("--foreground-rgb-loss-weight", type=float, default=1.0)
    challenger.add_argument("--background-rgb-loss-weight", type=float, default=1.0)
    challenger.add_argument("--palette-loss-weight", type=float, default=0.0)
    challenger.add_argument("--palette-loss-temperature", type=float, default=0.05)
    _add_palette_swap_arguments(challenger)
    challenger.add_argument("--base-channels", type=int, default=64)
    challenger.add_argument("--channel-mults", default="1,2,4")
    challenger.add_argument("--res-blocks-per-level", type=int, default=2)
    challenger.add_argument("--embed-dim", type=int, default=64)
    challenger.add_argument("--sample-every", type=int, default=250)
    challenger.add_argument("--save-every", type=int, default=1000)
    challenger.add_argument("--caption-policy-filter")
    challenger.add_argument("--max-records", type=int)
    challenger.add_argument("--max-train-sprites", type=int)
    challenger.add_argument("--sprite-id-list", type=Path)
    challenger.add_argument("--overfit-split")
    challenger.add_argument("--validation-mode", choices=["auto", "val", "same", "none"], default="auto")
    challenger.add_argument("--amp", action="store_true", default=False, help="Enable bf16 autocast (CUDA only).")
    challenger.add_argument("--grad-clip", type=float, default=0.0, help="Clip gradient norm; 0 disables.")
    challenger.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    challenger.add_argument("--lr-warmup-steps", type=int, default=0)

    palette_swap_review = subparsers.add_parser(
        "dataset-palette-swap-review",
        help="Sample and inspect palette-swap augmentation without training.",
    )
    palette_swap_review.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    palette_swap_review.add_argument("--training-manifest", required=True, type=Path)
    palette_swap_review.add_argument("--out-dir", required=True, type=Path)
    palette_swap_review.add_argument("--seed", type=int, default=20260706)
    palette_swap_review.add_argument("--max-samples", type=int, default=256)
    palette_swap_review.add_argument("--palette-swap-prob", type=float, default=0.5)
    palette_swap_review.add_argument("--palette-swap-families", default=DEFAULT_SWAP_FAMILIES_TEXT)
    palette_swap_review.add_argument("--palette-swap-target-families", default=None)
    palette_swap_review.add_argument("--palette-swap-source-families", default=None)
    palette_swap_review.add_argument("--palette-swap-category-filter", default=None)
    palette_swap_review.add_argument("--palette-swap-min-color-confidence", type=float, default=0.0)
    palette_swap_review.add_argument("--palette-swap-require-role-map", action="store_true", default=False)
    palette_swap_review.add_argument("--palette-swap-require-explicit-color", action="store_true", default=False)
    palette_swap_review.add_argument("--palette-swap-require-explicit-caption-color", action="store_true", default=False)
    palette_swap_review.add_argument("--palette-swap-require-explicit-semantic-color", action="store_true", default=False)
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

    sample_challenger = subparsers.add_parser(
        "sample-generator-challenger",
        help="Sample and canonicalize a generator challenger.",
    )
    sample_challenger.add_argument("--checkpoint", required=True, type=Path)
    sample_challenger.add_argument("--prompts", required=True, type=Path)
    sample_challenger.add_argument("--out", required=True, type=Path, dest="out_dir")
    sample_challenger.add_argument("--max-samples", type=int, default=64)
    sample_challenger.add_argument("--steps", type=int, default=30)
    sample_challenger.add_argument("--cfg-scale", type=float, default=2.0)
    sample_challenger.add_argument("--max-colors", type=int, default=32)
    sample_challenger.add_argument("--alpha-threshold", type=float, default=0.5)
    sample_challenger.add_argument("--device", default="cpu")
    sample_challenger.add_argument("--seed", type=int, default=123)
    sample_challenger.add_argument("--noise-seed", type=int)
    sample_challenger.add_argument("--batch-size", type=int, default=16)
    sample_challenger.add_argument("--dither", action="store_true", default=False)
    sample_challenger.add_argument("--no-dither", action="store_false", dest="dither")
    sample_challenger.add_argument("--write-raw-rgba", action="store_true", dest="write_raw_rgba", default=True)
    sample_challenger.add_argument("--no-write-raw-rgba", action="store_false", dest="write_raw_rgba")
    sample_challenger.add_argument("--write-hard-rgba", action="store_true", dest="write_hard_rgba", default=True)
    sample_challenger.add_argument("--no-write-hard-rgba", action="store_false", dest="write_hard_rgba")
    sample_challenger.add_argument(
        "--contact-sheet-labels",
        choices=["prompt", "prompt_and_seed", "prompt_and_nearest_source"],
        default="prompt",
    )

    generated_qa = subparsers.add_parser("generated-qa", help="QA generated sprite sample folders.")
    generated_qa.add_argument("--generated", required=True, type=Path)
    generated_qa.add_argument("--error-on-fully-transparent", action="store_true")

    dataset_framing_review = subparsers.add_parser(
        "dataset-framing-review",
        help="Review source sprite framing in exported training datasets.",
    )
    dataset_framing_review.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    dataset_framing_review.add_argument("--out-dir", type=Path)
    dataset_framing_review.add_argument("--compare-generated", type=Path)
    dataset_framing_review.add_argument("--max-samples-per-sheet", type=int, default=512)

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

    source_match = subparsers.add_parser("source-match-review", help="Review generated samples against source targets.")
    source_match.add_argument("--generated", required=True, type=Path)
    source_match.add_argument("--dataset", required=True, type=Path)
    source_match.add_argument("--training-manifest", required=True, type=Path)
    source_match.add_argument("--out", required=True, type=Path)
    source_match.add_argument("--out-json", type=Path)

    prompt_sensitivity = subparsers.add_parser(
        "prompt-sensitivity",
        help="Generate controlled prompt/noise sensitivity samples and metrics.",
    )
    prompt_sensitivity.add_argument("--checkpoint", required=True, type=Path)
    prompt_sensitivity.add_argument("--prompts", required=True, type=Path)
    prompt_sensitivity.add_argument("--out", required=True, type=Path, dest="out_dir")
    prompt_sensitivity.add_argument("--device", default="cpu")
    prompt_sensitivity.add_argument("--seed", type=int, default=123)
    prompt_sensitivity.add_argument("--max-prompts", type=int, default=32)
    prompt_sensitivity.add_argument("--noise-samples", type=int, default=16)
    prompt_sensitivity.add_argument("--max-pairs", type=int, default=8)
    prompt_sensitivity.add_argument("--max-colors", type=int, default=32)
    prompt_sensitivity.add_argument("--alpha-threshold", type=float, default=0.5)
    prompt_sensitivity.add_argument("--batch-size", type=int, default=16)

    compare_generated = subparsers.add_parser(
        "compare-generated-runs",
        help="Compare two generated sprite sample folders.",
    )
    compare_generated.add_argument("--a", required=True, type=Path)
    compare_generated.add_argument("--b", required=True, type=Path)
    compare_generated.add_argument("--out", required=True, type=Path, dest="out_dir")
    compare_generated.add_argument("--max-contact-sheet-pairs", type=int, default=64)

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

    audit_regression = subparsers.add_parser("audit-regression-generator", help="Run regression generator overfit audit.")
    audit_regression.add_argument("--dataset", required=True, type=Path)
    audit_regression.add_argument("--training-manifest", required=True, type=Path)
    audit_regression.add_argument("--out", required=True, type=Path, dest="out_dir")
    audit_regression.add_argument("--device", default="cpu")
    audit_regression.add_argument("--seed", type=int, default=20260706)

    audit_challenger = subparsers.add_parser("audit-challenger-generator", help="Run challenger generator audit.")
    audit_challenger.add_argument("--dataset", required=True, type=Path)
    audit_challenger.add_argument("--training-manifest", required=True, type=Path)
    audit_challenger.add_argument("--out", required=True, type=Path, dest="out_dir")
    audit_challenger.add_argument("--architecture", default="rectified_flow")
    audit_challenger.add_argument("--device", default="cpu")
    audit_challenger.add_argument("--seed", type=int, default=20260706)

    audit_challenger_full = subparsers.add_parser(
        "audit-challenger-full-v4",
        help="Run the reproducible full-v4 challenger train/sample/diagnostic audit.",
    )
    audit_challenger_full.add_argument("--dataset", required=True, type=Path)
    audit_challenger_full.add_argument("--training-manifest", required=True, type=Path)
    audit_challenger_full.add_argument("--out", required=True, type=Path, dest="out_dir")
    audit_challenger_full.add_argument("--architecture", default="rectified_flow")
    audit_challenger_full.add_argument("--device", default="cpu")
    audit_challenger_full.add_argument("--seed", type=int, default=20260706)
    audit_challenger_full.add_argument("--max-steps", type=int, default=25000)
    audit_challenger_full.add_argument("--batch-size", type=int, default=32)
    audit_challenger_full.add_argument("--num-workers", type=int, default=0)
    audit_challenger_full.add_argument("--lr", "--learning-rate", type=float, default=0.0002, dest="learning_rate")
    audit_challenger_full.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default="caption_semantic")
    audit_challenger_full.add_argument("--cfg-dropout", type=float, default=0.1)
    audit_challenger_full.add_argument("--structured-field-dropout", type=float, default=0.0)
    audit_challenger_full.add_argument("--ema-decay", type=float, default=0.999)
    audit_challenger_full.add_argument("--sample-ema", action="store_true", default=False)
    audit_challenger_full.add_argument("--foreground-rgb-loss-weight", type=float, default=1.0)
    audit_challenger_full.add_argument("--background-rgb-loss-weight", type=float, default=1.0)
    _add_palette_swap_arguments(audit_challenger_full)
    audit_challenger_full.add_argument("--palette-loss-weight", type=float, default=0.0)
    audit_challenger_full.add_argument("--palette-loss-temperature", type=float, default=0.05)
    audit_challenger_full.add_argument("--sample-steps", type=int, default=30)
    audit_challenger_full.add_argument("--cfg-scale", type=float, default=2.0)
    audit_challenger_full.add_argument("--max-colors", type=int, default=32)
    audit_challenger_full.add_argument("--alpha-threshold", type=float, default=0.5)
    audit_challenger_full.add_argument("--max-eval-prompts", type=int, default=128)
    audit_challenger_full.add_argument("--max-sensitivity-prompts", type=int, default=32)
    audit_challenger_full.add_argument(
        "--faithfulness-max-sources",
        type=int,
        default=0,
        help="Source sprites used for prompt-faithfulness nearest-source retrieval. 0 (default) uses all sources.",
    )
    audit_challenger_full.add_argument("--noise-samples", type=int, default=2)
    audit_challenger_full.add_argument("--sample-batch-size", type=int, default=16)
    audit_challenger_full.add_argument("--eval-prompts", type=Path)
    audit_challenger_full.add_argument("--reuse-existing-prompts", action="store_true", default=False)
    audit_challenger_full.add_argument("--run-ood-compositional", action="store_true", default=False)
    audit_challenger_full.add_argument("--ood-prompts", type=Path)
    audit_challenger_full.add_argument("--eval-checkpoints", action="store_true", default=False)
    audit_challenger_full.add_argument("--eval-checkpoint-every", type=int, default=5000)
    audit_challenger_full.add_argument("--eval-checkpoint-steps")
    audit_challenger_full.add_argument("--amp", action="store_true", default=True)
    audit_challenger_full.add_argument("--no-amp", action="store_false", dest="amp")
    audit_challenger_full.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    audit_challenger_full.add_argument("--lr-warmup-steps", type=int, default=500)

    build_ood = subparsers.add_parser(
        "build-ood-compositional-prompts",
        help="Write the default color/object OOD compositional prompt set.",
    )
    build_ood.add_argument("--out", required=True, type=Path)
    build_ood.add_argument("--max-prompts", type=int)

    compare_conditioning = subparsers.add_parser(
        "compare-challenger-conditioning-audits",
        help="Compare baseline and structured full-v4 challenger audit reports.",
    )
    compare_conditioning.add_argument("--baseline", required=True, type=Path)
    compare_conditioning.add_argument("--structured", required=True, type=Path)
    compare_conditioning.add_argument("--out", required=True, type=Path, dest="out_dir")

    monitor = subparsers.add_parser(
        "monitor-training",
        help="Live dashboard that tails train_metrics.jsonl for one or many runs.",
    )
    monitor.add_argument("--dir", required=True, type=Path, dest="root", help="Run dir or experiment dir to watch.")
    monitor.add_argument("--interval", type=float, default=2.0, help="Seconds between refreshes.")
    monitor.add_argument("--html", type=Path, default=None, dest="html_path", help="Also write an auto-refreshing HTML dashboard.")
    monitor.add_argument("--once", action="store_true", help="Render a single snapshot and exit.")
    monitor.add_argument("--no-rich", action="store_true", help="Force plain-text output.")

    compare_families = subparsers.add_parser(
        "compare-generator-families",
        help="Compare regression baseline and challenger generated outputs.",
    )
    compare_families.add_argument("--baseline-run", required=True, type=Path)
    compare_families.add_argument("--baseline-generated", required=True, type=Path)
    compare_families.add_argument("--challenger-run", required=True, type=Path)
    compare_families.add_argument("--challenger-generated", required=True, type=Path)
    compare_families.add_argument("--dataset", required=True, type=Path)
    compare_families.add_argument("--prompts", required=True, type=Path)
    compare_families.add_argument("--out", required=True, type=Path, dest="out_dir")

    parsed = parser.parse_args(argv)
    try:
        if parsed.subcommand == "inspect-data":
            from spritelab.training.train_baseline import inspect_training_data, print_inspection

            summary = inspect_training_data(
                dataset_dir=parsed.dataset,
                training_manifest=parsed.training_manifest,
                split=parsed.split,
                batch_size=parsed.batch_size,
                max_records=parsed.max_records,
            )
            print_inspection(summary)
        elif parsed.subcommand == "baseline":
            from spritelab.training.train_baseline import BaselineTrainConfig, run_baseline_training

            report = run_baseline_training(
                BaselineTrainConfig(
                    dataset_dir=parsed.dataset_dir,
                    training_manifest=parsed.training_manifest,
                    out_dir=parsed.out_dir,
                    batch_size=parsed.batch_size,
                    max_steps=parsed.max_steps,
                    learning_rate=parsed.learning_rate,
                    device=parsed.device,
                    seed=parsed.seed,
                    overfit_batches=parsed.overfit_batches,
                    max_records=parsed.max_records,
                    num_workers=parsed.num_workers,
                    amp=parsed.amp,
                    grad_clip=parsed.grad_clip,
                    lr_schedule=parsed.lr_schedule,
                    lr_warmup_steps=parsed.lr_warmup_steps,
                )
            )
            print(f"Initial train loss: {report['initial_train_loss']:.6f}")
            print(f"Final train loss: {report['final_train_loss']:.6f}")
            if report["val_loss"] is not None:
                print(f"Val loss: {report['val_loss']:.6f}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "generator":
            from spritelab.training.train_generator import GeneratorTrainConfig, run_generator_training

            report = run_generator_training(
                GeneratorTrainConfig(
                    dataset_dir=parsed.dataset_dir,
                    training_manifest=parsed.training_manifest,
                    out_dir=parsed.out_dir,
                    split=parsed.split,
                    batch_size=parsed.batch_size,
                    max_steps=parsed.max_steps,
                    learning_rate=parsed.learning_rate,
                    device=parsed.device,
                    seed=parsed.seed,
                    overfit_batches=parsed.overfit_batches,
                    num_workers=parsed.num_workers,
                    latent_dim=parsed.latent_dim,
                    embed_dim=parsed.embed_dim,
                    hidden_channels=parsed.hidden_channels,
                    sample_every=parsed.sample_every,
                    save_every=parsed.save_every,
                    caption_policy_filter=parsed.caption_policy_filter,
                    max_records=parsed.max_records,
                    conditioning_mode=parsed.conditioning_mode,
                    border_alpha_weight=parsed.border_alpha_weight,
                    alpha_coverage_weight=parsed.alpha_coverage_weight,
                    alpha_coverage_min=parsed.alpha_coverage_min,
                    alpha_coverage_max=parsed.alpha_coverage_max,
                    center_weight=parsed.center_weight,
                    margin_band_weight=parsed.margin_band_weight,
                    margin_band_size=parsed.margin_band_size,
                    max_train_sprites=parsed.max_train_sprites,
                    sprite_id_list=parsed.sprite_id_list,
                    overfit_split=parsed.overfit_split,
                    validation_mode=parsed.validation_mode,
                    amp=parsed.amp,
                    grad_clip=parsed.grad_clip,
                    lr_schedule=parsed.lr_schedule,
                    lr_warmup_steps=parsed.lr_warmup_steps,
                )
            )
            print(f"Initial train loss: {report['initial_train_loss']:.6f}")
            print(f"Final train loss: {report['final_train_loss']:.6f}")
            if report["val_loss"] is not None:
                print(f"Val loss: {report['val_loss']:.6f}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "make-overfit-subset":
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
        elif parsed.subcommand == "eval-baseline":
            from spritelab.training.eval_baseline import evaluate_baseline_checkpoint

            report = evaluate_baseline_checkpoint(
                dataset_dir=parsed.dataset,
                training_manifest=parsed.training_manifest,
                checkpoint=parsed.checkpoint,
                split=parsed.split,
                out_dir=parsed.out,
                batch_size=parsed.batch_size,
                device=parsed.device,
                max_records=parsed.max_records,
            )
            print(f"Evaluated {report['records']} {report['split']} records.")
            print(f"Loss: {report['loss']:.6f}")
            print(f"Outputs written to {parsed.out}")
        elif parsed.subcommand == "eval-generator":
            from spritelab.training.eval_generator import evaluate_generator_checkpoint

            report = evaluate_generator_checkpoint(
                dataset_dir=parsed.dataset,
                training_manifest=parsed.training_manifest,
                checkpoint=parsed.checkpoint,
                split=parsed.split,
                prompts=parsed.prompts,
                out_dir=parsed.out,
                batch_size=parsed.batch_size,
                device=parsed.device,
                max_records=parsed.max_records,
            )
            if report["loss"] is not None:
                print(f"Evaluated {report['records']} {report['split']} records.")
                print(f"Loss: {report['loss']:.6f}")
            if report["prompt_count"]:
                print(
                    f"Generated {report['prompt_samples_written']} prompt samples "
                    f"from {report['prompt_count']} prompts."
                )
            print(f"Outputs written to {parsed.out}")
        elif parsed.subcommand == "sample-generator":
            from spritelab.training.sample_generator import SampleGeneratorConfig, run_sample_generator

            report = run_sample_generator(
                SampleGeneratorConfig(
                    checkpoint=parsed.checkpoint,
                    prompts=parsed.prompts,
                    out_dir=parsed.out_dir,
                    max_samples=parsed.max_samples,
                    max_colors=parsed.max_colors,
                    alpha_threshold=parsed.alpha_threshold,
                    device=parsed.device,
                    seed=parsed.seed,
                    noise_seed=parsed.noise_seed,
                    dither=parsed.dither,
                    write_raw_rgba=parsed.write_raw_rgba,
                    write_hard_rgba=parsed.write_hard_rgba,
                    batch_size=parsed.batch_size,
                    contact_sheet_labels=parsed.contact_sheet_labels,
                )
            )
            print(f"Generated samples: {report['sample_count']}")
            print(f"Max visible colors: {report['max_visible_color_count']}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "generator-challenger":
            from spritelab.training.generator_challenger import ChallengerTrainConfig, run_challenger_training

            report = run_challenger_training(ChallengerTrainConfig(**_parsed_config_kwargs(parsed)))
            print(f"Initial train loss: {report['initial_train_loss']:.6f}")
            print(f"Final train loss: {report['final_train_loss']:.6f}")
            if report["val_loss"] is not None:
                print(f"Val loss: {report['val_loss']:.6f}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "dataset-palette-swap-review":
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
        elif parsed.subcommand == "sample-generator-challenger":
            from spritelab.training.generator_challenger import (
                ChallengerSampleConfig,
                run_sample_generator_challenger,
            )

            report = run_sample_generator_challenger(ChallengerSampleConfig(**_parsed_config_kwargs(parsed)))
            print(f"Generated samples: {report['sample_count']}")
            print(f"Max visible colors: {report['max_visible_color_count']}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "generated-qa":
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
        elif parsed.subcommand == "source-match-review":
            from spritelab.training.source_match_review import SourceMatchReviewConfig, run_source_match_review

            report = run_source_match_review(SourceMatchReviewConfig(**_parsed_config_kwargs(parsed)))
            print(f"Matched sources: {report['matched_source_count']}/{report['sample_count']}")
            print(f"Mean alpha IoU: {report.get('mean_alpha_iou')}")
        elif parsed.subcommand == "dataset-framing-review":
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
        elif parsed.subcommand == "generated-review":
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
        elif parsed.subcommand == "prompt-sensitivity":
            from spritelab.training.prompt_sensitivity import PromptSensitivityConfig, run_prompt_sensitivity

            report = run_prompt_sensitivity(
                PromptSensitivityConfig(
                    checkpoint=parsed.checkpoint,
                    prompts=parsed.prompts,
                    out_dir=parsed.out_dir,
                    device=parsed.device,
                    seed=parsed.seed,
                    max_prompts=parsed.max_prompts,
                    noise_samples=parsed.noise_samples,
                    max_pairs=parsed.max_pairs,
                    max_colors=parsed.max_colors,
                    alpha_threshold=parsed.alpha_threshold,
                    batch_size=parsed.batch_size,
                )
            )
            same_noise = report["sets"]["same_noise_different_prompts"]["metrics"]
            same_prompt = report["sets"]["same_prompt_different_noise"]["metrics"]
            print(f"Same-noise mean difference: {same_noise['mean_pairwise_difference']:.6f}")
            print(f"Same-prompt diversity: {same_prompt['diversity_score']:.6f}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "compare-generated-runs":
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
        elif parsed.subcommand == "prompt-faithfulness":
            from spritelab.training.prompt_faithfulness import PromptFaithfulnessConfig, run_prompt_faithfulness

            report = run_prompt_faithfulness(PromptFaithfulnessConfig(**_parsed_config_kwargs(parsed)))
            print(f"Prompt faithfulness samples: {report['sample_count']}")
            print(f"Repeated silhouette rate: {report.get('repeated_silhouette_rate')}")
            print(f"Outputs written to {parsed.out}")
        elif parsed.subcommand == "audit-regression-generator":
            from spritelab.training.generator_audits import (
                RegressionGeneratorAuditConfig,
                run_regression_generator_audit,
            )

            report = run_regression_generator_audit(RegressionGeneratorAuditConfig(**_parsed_config_kwargs(parsed)))
            print(f"Runs completed: {len(report['runs'])}")
            print(f"Recommendation: {report['decision']['recommendation']}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "audit-challenger-generator":
            from spritelab.training.generator_audits import (
                ChallengerGeneratorAuditConfig,
                run_challenger_generator_audit,
            )

            report = run_challenger_generator_audit(ChallengerGeneratorAuditConfig(**_parsed_config_kwargs(parsed)))
            print(f"Runs completed: {len(report['runs'])}")
            print(f"Recommendation: {report['decision']['recommendation']}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "audit-challenger-full-v4":
            from spritelab.training.generator_audits import (
                FullV4ChallengerAuditConfig,
                run_full_v4_challenger_audit,
            )

            report = run_full_v4_challenger_audit(FullV4ChallengerAuditConfig(**_parsed_config_kwargs(parsed)))
            print(f"Decision: {report['decision']['code']}. {report['decision']['label']}")
            print(f"Markdown report: {parsed.out_dir / 'full_v4_challenger_audit.md'}")
            print(f"JSON report: {parsed.out_dir / 'full_v4_challenger_audit.json'}")
        elif parsed.subcommand == "build-ood-compositional-prompts":
            from spritelab.training.ood_prompts import OodCompositionalPromptConfig, build_ood_compositional_prompts

            report = build_ood_compositional_prompts(
                OodCompositionalPromptConfig(out=parsed.out, max_prompts=parsed.max_prompts)
            )
            print(f"Prompts written: {report['prompt_count']}")
            print(f"Output written to {parsed.out}")
        elif parsed.subcommand == "compare-challenger-conditioning-audits":
            from spritelab.training.generator_audits import compare_challenger_conditioning_audits

            report = compare_challenger_conditioning_audits(
                baseline=parsed.baseline,
                structured=parsed.structured,
                out_dir=parsed.out_dir,
            )
            answers = report["answers"]
            print(f"Dataset-grounded category: {answers['dataset_grounded_category']}")
            print(f"OOD category: {answers['ood_category']}")
            print(f"Dataset-grounded color: {answers['dataset_grounded_color']}")
            print(f"OOD color: {answers['ood_color']}")
            print(f"OOD blob collapse: {answers['ood_blob_collapse']}")
            print(f"Rare-color rate: {answers['rare_color_rate']}")
            print(f"Outputs written to {parsed.out_dir}")
        elif parsed.subcommand == "monitor-training":
            from spritelab.training.live_monitor import run_live_monitor

            summary = run_live_monitor(
                parsed.root,
                interval=parsed.interval,
                html_path=parsed.html_path,
                once=parsed.once,
                use_rich=not parsed.no_rich,
            )
            if parsed.html_path is not None:
                print(f"HTML dashboard written to {parsed.html_path}")
            print(
                f"Runs: {summary['runs']} | done {summary['done']} | "
                f"running {summary['running']} | overall {summary['fraction'] * 100:.1f}%"
            )
        elif parsed.subcommand == "compare-generator-families":
            from spritelab.training.compare_generator_families import (
                CompareGeneratorFamiliesConfig,
                compare_generator_families,
            )

            report = compare_generator_families(CompareGeneratorFamiliesConfig(**_parsed_config_kwargs(parsed)))
            print(f"Recommendation: {report['recommendation']}")
            print(f"Outputs written to {parsed.out_dir}")
    except RuntimeError as exc:
        if "PyTorch is required" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1)
