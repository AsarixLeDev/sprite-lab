"""Audit and OOD prompt building commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.training.cli._args import (
    _add_export_preset_argument,
    _add_palette_projection_sampling_arguments,
    _add_palette_swap_arguments,
    _add_speed_option_arguments,
    _parse_dropout_rates,
    _parsed_config_kwargs,
)
from spritelab.training.conditioning import CONDITIONING_MODES


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_audit_challenger_generator(subparsers)
    _register_audit_challenger_full_v4(subparsers)
    _register_compare_challenger_conditioning_audits(subparsers)
    _register_build_ood_compositional_prompts(subparsers)


def _register_audit_challenger_generator(subparsers: argparse._SubParsersAction) -> None:
    audit_challenger = subparsers.add_parser("audit-challenger-generator", help="Run challenger generator audit.")
    audit_challenger.add_argument("--dataset", required=True, type=Path)
    audit_challenger.add_argument("--training-manifest", required=True, type=Path)
    audit_challenger.add_argument("--out", required=True, type=Path, dest="out_dir")
    audit_challenger.add_argument("--architecture", default="rectified_flow")
    audit_challenger.add_argument("--device", default="cpu")
    audit_challenger.add_argument("--seed", type=int, default=20260706)
    audit_challenger.set_defaults(func=_run_audit_challenger_generator)


def _run_audit_challenger_generator(parsed: argparse.Namespace) -> None:
    from spritelab.training.generator_audits import (
        ChallengerGeneratorAuditConfig,
        run_challenger_generator_audit,
    )

    report = run_challenger_generator_audit(ChallengerGeneratorAuditConfig(**_parsed_config_kwargs(parsed)))
    print(f"Runs completed: {len(report['runs'])}")
    print(f"Recommendation: {report['decision']['recommendation']}")
    print(f"Outputs written to {parsed.out_dir}")


def _register_audit_challenger_full_v4(subparsers: argparse._SubParsersAction) -> None:
    audit_challenger_full = subparsers.add_parser(
        "audit-challenger-full-v4",
        help="Run the reproducible full-v4 challenger train/sample/diagnostic audit.",
    )
    _add_export_preset_argument(audit_challenger_full)
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
    audit_challenger_full.add_argument("--no-sample-ema", action="store_false", dest="sample_ema")
    audit_challenger_full.add_argument("--foreground-rgb-loss-weight", type=float, default=1.0)
    audit_challenger_full.add_argument("--background-rgb-loss-weight", type=float, default=1.0)
    _add_palette_swap_arguments(audit_challenger_full)
    audit_challenger_full.add_argument("--palette-loss-weight", type=float, default=0.0)
    audit_challenger_full.add_argument("--palette-loss-temperature", type=float, default=0.05)
    audit_challenger_full.add_argument("--sample-steps", type=int, default=30)
    audit_challenger_full.add_argument("--cfg-scale", type=float, default=2.0)
    audit_challenger_full.add_argument("--max-colors", type=int, default=32)
    audit_challenger_full.add_argument("--alpha-threshold", type=float, default=0.5)
    _add_palette_projection_sampling_arguments(audit_challenger_full)
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
    audit_challenger_full.add_argument("--checkpoint-eval-max-samples", type=int)
    audit_challenger_full.add_argument("--amp", action="store_true", default=True)
    audit_challenger_full.add_argument("--no-amp", action="store_false", dest="amp")
    audit_challenger_full.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    audit_challenger_full.add_argument("--lr-warmup-steps", type=int, default=500)
    _add_speed_option_arguments(audit_challenger_full)
    audit_challenger_full.add_argument(
        "--film-conditioning",
        action="store_true",
        default=False,
        help="Enable FiLM conditioning in residual blocks (v2 Phase 1).",
    )
    audit_challenger_full.add_argument(
        "--bottleneck-attention",
        action="store_true",
        default=False,
        help="Enable lightweight self-attention at U-Net bottleneck (v2 Phase 1).",
    )
    audit_challenger_full.add_argument(
        "--structured-field-dropout-rates",
        default=None,
        help="Per-group structured dropout rates, e.g. 'category=0.10,object_id=0.35,colors=0.15'. "
        "Overrides --structured-field-dropout for listed groups.",
    )
    audit_challenger_full.add_argument(
        "--index-head-loss-weight",
        type=float,
        default=0.0,
        help="Weight for index map cross-entropy head loss (v2 Phase 2).",
    )
    audit_challenger_full.add_argument(
        "--palette-head-loss-weight",
        type=float,
        default=0.0,
        help="Weight for palette slot MSE head loss (v2 Phase 2).",
    )
    audit_challenger_full.add_argument(
        "--palette-presence-loss-weight",
        type=float,
        default=0.0,
        help="Weight for palette slot presence BCE head loss (v2 Phase 2).",
    )
    audit_challenger_full.add_argument(
        "--index-head-warmup-steps", type=int, default=0, help="Index head loss inactive before this step (v2 Phase 2)."
    )
    audit_challenger_full.add_argument(
        "--palette-head-use-gt-palette-prob",
        type=float,
        default=1.0,
        help="Probability of using GT palette vs predicted palette for training (v2 Phase 2).",
    )
    audit_challenger_full.set_defaults(func=_run_audit_challenger_full_v4)


def _run_audit_challenger_full_v4(parsed: argparse.Namespace) -> None:
    from spritelab.training.generator_audits import (
        FullV4ChallengerAuditConfig,
        run_full_v4_challenger_audit,
    )

    kwargs = _parsed_config_kwargs(parsed)
    dropout_rates = _parse_dropout_rates(kwargs.pop("structured_field_dropout_rates", None))
    report = run_full_v4_challenger_audit(
        FullV4ChallengerAuditConfig(**kwargs, structured_field_dropout_rates=dropout_rates)
    )
    print(f"Decision: {report['decision']['code']}. {report['decision']['label']}")
    print(f"Markdown report: {parsed.out_dir / 'full_v4_challenger_audit.md'}")
    print(f"JSON report: {parsed.out_dir / 'full_v4_challenger_audit.json'}")


def _register_compare_challenger_conditioning_audits(subparsers: argparse._SubParsersAction) -> None:
    compare_conditioning = subparsers.add_parser(
        "compare-challenger-conditioning-audits",
        help="Compare baseline and structured full-v4 challenger audit reports.",
    )
    compare_conditioning.add_argument("--baseline", required=True, type=Path)
    compare_conditioning.add_argument("--structured", required=True, type=Path)
    compare_conditioning.add_argument("--out", required=True, type=Path, dest="out_dir")
    compare_conditioning.set_defaults(func=_run_compare_challenger_conditioning_audits)


def _run_compare_challenger_conditioning_audits(parsed: argparse.Namespace) -> None:
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


def _register_build_ood_compositional_prompts(subparsers: argparse._SubParsersAction) -> None:
    build_ood = subparsers.add_parser(
        "build-ood-compositional-prompts",
        help="Write the default color/object OOD compositional prompt set.",
    )
    build_ood.add_argument("--out", required=True, type=Path)
    build_ood.add_argument("--max-prompts", type=int)
    build_ood.set_defaults(func=_run_build_ood_compositional_prompts)


def _run_build_ood_compositional_prompts(parsed: argparse.Namespace) -> None:
    from spritelab.training.ood_prompts import OodCompositionalPromptConfig, build_ood_compositional_prompts

    report = build_ood_compositional_prompts(
        OodCompositionalPromptConfig(out=parsed.out, max_prompts=parsed.max_prompts)
    )
    print(f"Prompts written: {report['prompt_count']}")
    print(f"Output written to {parsed.out}")
