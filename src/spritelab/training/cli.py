"""CLI for semantic training baselines."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from spritelab.training.conditioning import CONDITIONING_MODES, DEFAULT_CONDITIONING_MODE


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
                )
            )
            print(f"Initial train loss: {report['initial_train_loss']:.6f}")
            print(f"Final train loss: {report['final_train_loss']:.6f}")
            if report["val_loss"] is not None:
                print(f"Val loss: {report['val_loss']:.6f}")
            print(f"Outputs written to {parsed.out_dir}")
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
                )
            )
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
    except RuntimeError as exc:
        if "PyTorch is required" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1)
