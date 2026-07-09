"""CLI for the ML package: validate-dataset, baseline-eval, overfit-smoke."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m spritelab ml",
        description="ML dataset validation, baselines, and overfit smoke test.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    validate = subparsers.add_parser("validate-dataset", help="Validate a dataset split.")
    validate.add_argument("--dataset", required=True, type=Path)
    validate.add_argument("--split", default="train")

    baseline = subparsers.add_parser("baseline-eval", help="Evaluate dumb baselines.")
    baseline.add_argument("--dataset", required=True, type=Path)
    baseline.add_argument("--split", default="train")
    baseline.add_argument("--mask-fraction", type=float, default=0.5)
    baseline.add_argument("--out", required=True, type=Path)
    baseline.add_argument("--max-samples", type=int)

    overfit = subparsers.add_parser("overfit-smoke", help="Tiny overfit smoke test.")
    overfit.add_argument("--dataset", required=True, type=Path)
    overfit.add_argument("--split", default="train")
    overfit.add_argument("--max-samples", type=int, default=16)
    overfit.add_argument("--steps", type=int, default=300)
    overfit.add_argument("--batch-size", type=int, default=8)
    overfit.add_argument("--learning-rate", type=float, default=1e-3)
    overfit.add_argument("--mask-fraction", type=float, default=0.5)
    overfit.add_argument("--seed", type=int, default=1337)
    overfit.add_argument("--device", default="auto")
    overfit.add_argument("--out", required=True, type=Path)

    parsed = parser.parse_args(argv)
    if parsed.subcommand == "validate-dataset":
        _run_validate_dataset(parsed)
    elif parsed.subcommand == "baseline-eval":
        _run_baseline_eval(parsed)
    elif parsed.subcommand == "overfit-smoke":
        _run_overfit_smoke(parsed)


def _run_validate_dataset(parsed: argparse.Namespace) -> None:
    from spritelab.ml.dataset import SpriteBundleDataset

    try:
        dataset = SpriteBundleDataset(parsed.dataset, parsed.split, validate=True)
    except (FileNotFoundError, ValueError) as exc:
        print(f"INVALID: {exc}")
        raise SystemExit(1) from exc

    print(f"Dataset: {parsed.dataset}")
    print(f"Split: {parsed.split}")
    print(f"Samples: {len(dataset)}")
    print(f"Palette rows: {dataset.arrays['palette'].shape[1]}")
    categories = {int(value) for value in dataset.arrays["category_id"]}
    print(f"Categories: {len(categories)}")
    print("Valid.")


def _run_baseline_eval(parsed: argparse.Namespace) -> None:
    from spritelab.ml.baselines import run_baseline_evaluation

    results = run_baseline_evaluation(
        parsed.dataset,
        parsed.split,
        parsed.out,
        mask_fraction=parsed.mask_fraction,
        max_samples=parsed.max_samples,
    )
    print(f"Evaluated {results['sample_count']} samples at mask fraction {results['mask_fraction']}.")
    for name, metrics in results["baselines"].items():
        print(
            f"  {name}: masked_accuracy={metrics['masked_accuracy']:.4f} "
            f"token_accuracy={metrics['token_accuracy']:.4f} "
            f"invalid_token_rate={metrics['invalid_token_rate']:.4f}"
        )
    print(f"Metrics written to {Path(parsed.out) / 'baseline_metrics.json'}")


def _run_overfit_smoke(parsed: argparse.Namespace) -> None:
    from spritelab.ml.overfit import OverfitConfig, run_overfit_smoke_test

    result = run_overfit_smoke_test(
        OverfitConfig(
            dataset_root=parsed.dataset,
            split=parsed.split,
            output_dir=parsed.out,
            max_samples=parsed.max_samples,
            steps=parsed.steps,
            batch_size=parsed.batch_size,
            learning_rate=parsed.learning_rate,
            mask_fraction=parsed.mask_fraction,
            seed=parsed.seed,
            device=parsed.device,
        )
    )
    print(
        json.dumps(
            {
                "initial_loss": result["initial_loss"],
                "final_loss": result["final_loss"],
                "initial_masked_accuracy": result["initial_masked_accuracy"],
                "final_masked_accuracy": result["final_masked_accuracy"],
                "passed": result["passed"],
            },
            indent=2,
        )
    )
    print(f"Outputs written to {parsed.out}")
