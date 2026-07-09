"""Evaluation commands for prompt sensitivity and v2 Phase 0."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_prompt_sensitivity(subparsers)
    _register_run_v2_phase0_eval(subparsers)


def _register_prompt_sensitivity(subparsers: argparse._SubParsersAction) -> None:
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
    prompt_sensitivity.set_defaults(func=_run_prompt_sensitivity)


def _run_prompt_sensitivity(parsed: argparse.Namespace) -> None:
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


def _register_run_v2_phase0_eval(subparsers: argparse._SubParsersAction) -> None:
    v2_phase0 = subparsers.add_parser(
        "run-v2-phase0-eval",
        help="Run the reproducible v2 Phase 0 no-training evaluation harness.",
        description=(
            "Orchestrates sampling, QA, structural review, and prompt-faithfulness "
            "diagnostics for preset and ablation comparison cells across multiple seeds. "
            "Collects metrics, aggregates across seeds, applies decision rules, and writes "
            "summary JSON/CSV/Markdown reports. Never trains a model. "
            "See docs/v2_phase0_diagnostics.md."
        ),
    )
    v2_phase0.add_argument("--out", required=True, type=Path, help="Output directory for runs and summaries.")
    v2_phase0.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to a generator_challenger checkpoint (.pt or directory)."
    )
    v2_phase0.add_argument(
        "--prompts",
        type=Path,
        help="JSONL prompt file (e.g. OOD compositional prompts). Required unless --build-prompts is used.",
    )
    v2_phase0.add_argument(
        "--dataset", required=True, type=Path, help="Training dataset directory (must contain training_manifest.jsonl)."
    )
    v2_phase0.add_argument(
        "--presets", default="v1,v1.1", help="Comma-separated presets, e.g. 'v1,v1.1'. Default: v1,v1.1."
    )
    v2_phase0.add_argument(
        "--seeds",
        default="20260723,20260724,20260725",
        help="Comma-separated integer seeds. Default: 20260723,20260724,20260725.",
    )
    v2_phase0.add_argument("--max-samples", type=int, default=96)
    v2_phase0.add_argument("--device", default="cpu")
    v2_phase0.add_argument("--batch-size", type=int, default=16)
    v2_phase0.add_argument(
        "--include-ablations", action="store_true", default=False, help="Include null-field ablation cells."
    )
    v2_phase0.add_argument(
        "--null-field-sets", default="", help="Comma-separated null-field groups, e.g. 'colors,object_id,category'."
    )
    v2_phase0.add_argument(
        "--factored-grid", default="", help="Grid string, e.g. 'base=1.5,2.0,2.5;color=2.0,3.0,4.5,6.0'."
    )
    v2_phase0.add_argument(
        "--skip-sampling-if-exists",
        action="store_true",
        default=False,
        help="Skip cells whose output directories already exist.",
    )
    v2_phase0.add_argument(
        "--faithfulness-max-sources", type=int, default=0, help="Source sprites for prompt-faithfulness. 0 uses all."
    )
    v2_phase0.add_argument(
        "--no-contact-sheets",
        action="store_true",
        default=False,
        help="Skip contact sheet generation for faster evaluation.",
    )
    v2_phase0.add_argument(
        "--dry-run", action="store_true", default=False, help="Print planned run cells without executing."
    )
    v2_phase0.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Harvest existing run outputs and write summary reports without sampling.",
    )
    v2_phase0.add_argument(
        "--allow-partial-report",
        action="store_true",
        default=False,
        help="When --report-only, allow reports from partial/missing run outputs.",
    )
    v2_phase0.add_argument(
        "--build-prompts",
        action="store_true",
        default=False,
        help="Build eval prompts from the dataset manifest instead of using --prompts.",
    )
    v2_phase0.add_argument(
        "--prompt-count", type=int, default=384, help="Target prompt count when --build-prompts is used. Default: 384."
    )
    v2_phase0.add_argument(
        "--prompt-seed",
        type=int,
        default=20260706,
        help="Seed for prompt building when --build-prompts is used. Default: 20260706.",
    )
    v2_phase0.add_argument(
        "--speed-optimizations",
        action="store_true",
        default=True,
        dest="speed_optimizations",
        help="Enable cuDNN autotuning + TF32 for sampling (CUDA only; no-op on CPU). "
        "On by default since this harness resamples the same checkpoint/shape across "
        "many cells and seeds; use --no-speed-optimizations for the plain numeric path.",
    )
    v2_phase0.add_argument(
        "--no-speed-optimizations",
        action="store_false",
        dest="speed_optimizations",
    )
    v2_phase0.add_argument(
        "--eval-profile",
        default="all",
        choices=["all", "ood_core", "ood_plus_grid"],
        help="Evaluation profile. 'all' includes all prompt families. "
        "'ood_core' excludes in-distribution anchors. Default: all.",
    )
    v2_phase0.add_argument(
        "--profile-weighting",
        default="family",
        choices=["sample", "family"],
        help="Profile weighting method. 'family' gives equal weight to each prompt family. "
        "'sample' weights by sample count. Default: family.",
    )
    v2_phase0.add_argument(
        "--guidance-surgery-grid",
        action="store_true",
        default=False,
        help="Include v2 Phase 2 Exp A guidance surgery variants (rgb-only, late-window, object-id scale).",
    )
    v2_phase0.set_defaults(func=_run_run_v2_phase0_eval)


def _run_run_v2_phase0_eval(parsed: argparse.Namespace) -> None:
    from spritelab.training.v2_phase0_eval import (
        V2Phase0EvalConfig,
        parse_null_field_sets,
        parse_presets,
        parse_seeds,
        run_v2_phase0_eval,
    )

    if parsed.build_prompts and parsed.prompts:
        raise SystemExit(
            "Cannot use both --prompts and --build-prompts. "
            "Use --build-prompts to generate prompts from the dataset, "
            "or --prompts to provide a pre-existing prompt file."
        )
    if not parsed.build_prompts and not parsed.prompts:
        raise SystemExit(
            "Either --prompts or --build-prompts is required. "
            "Use --build-prompts to generate prompts from the dataset, "
            "or --prompts to provide a pre-existing prompt file."
        )

    run_v2_phase0_eval(
        V2Phase0EvalConfig(
            out=parsed.out,
            checkpoint=parsed.checkpoint,
            prompts=parsed.prompts,
            dataset=parsed.dataset,
            presets=parse_presets(parsed.presets),
            seeds=parse_seeds(parsed.seeds),
            max_samples=parsed.max_samples,
            device=parsed.device,
            batch_size=parsed.batch_size,
            include_ablations=parsed.include_ablations,
            null_field_sets=parse_null_field_sets(parsed.null_field_sets),
            factored_grid=parsed.factored_grid,
            skip_sampling_if_exists=parsed.skip_sampling_if_exists,
            faithfulness_max_sources=parsed.faithfulness_max_sources,
            no_contact_sheets=parsed.no_contact_sheets,
            dry_run=parsed.dry_run,
            build_prompts=parsed.build_prompts,
            prompt_count=parsed.prompt_count,
            prompt_seed=parsed.prompt_seed,
            report_only=parsed.report_only,
            allow_partial_report=parsed.allow_partial_report,
            speed_optimizations=parsed.speed_optimizations,
            eval_profile=parsed.eval_profile,
            profile_weighting=parsed.profile_weighting,
            guidance_surgery_grid=parsed.guidance_surgery_grid,
        )
    )
    if not parsed.dry_run:
        print(f"Summary reports: {parsed.out / 'summaries'}")
