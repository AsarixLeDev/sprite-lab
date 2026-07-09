"""Generator challenger training, sampling, and palette/index inspection commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.training.cli._args import (
    _add_export_preset_argument,
    _add_palette_projection_sampling_arguments,
    _add_palette_swap_arguments,
    _add_speed_option_arguments,
    _add_v2_phase0_diagnostic_arguments,
    _parse_dropout_rates,
    _parsed_config_kwargs,
)
from spritelab.training.conditioning import CONDITIONING_MODES, DEFAULT_CONDITIONING_MODE


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_generator_challenger(subparsers)
    _register_sample_generator_challenger(subparsers)
    _register_inspect_palette_index_heads(subparsers)
    _register_probe_palette_index_decode(subparsers)


def _register_generator_challenger(subparsers: argparse._SubParsersAction) -> None:
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
    _add_speed_option_arguments(challenger)
    challenger.add_argument(
        "--film-conditioning",
        action="store_true",
        default=False,
        help="Enable FiLM conditioning in residual blocks (v2 Phase 1).",
    )
    challenger.add_argument(
        "--bottleneck-attention",
        action="store_true",
        default=False,
        help="Enable lightweight self-attention at U-Net bottleneck (v2 Phase 1).",
    )
    challenger.add_argument(
        "--structured-field-dropout-rates",
        default=None,
        help="Per-group structured dropout rates, e.g. 'category=0.10,object_id=0.35,colors=0.15'. "
        "Overrides --structured-field-dropout for listed groups.",
    )
    challenger.add_argument(
        "--index-head-loss-weight",
        type=float,
        default=0.0,
        help="Weight for index map cross-entropy head loss (v2 Phase 2).",
    )
    challenger.add_argument(
        "--palette-head-loss-weight",
        type=float,
        default=0.0,
        help="Weight for palette slot MSE head loss (v2 Phase 2).",
    )
    challenger.add_argument(
        "--palette-presence-loss-weight",
        type=float,
        default=0.0,
        help="Weight for palette slot presence BCE head loss (v2 Phase 2).",
    )
    challenger.add_argument(
        "--index-head-warmup-steps", type=int, default=0, help="Index head loss inactive before this step (v2 Phase 2)."
    )
    challenger.add_argument(
        "--palette-head-use-gt-palette-prob",
        type=float,
        default=1.0,
        help="Probability of using GT palette vs predicted palette for training (v2 Phase 2).",
    )
    challenger.set_defaults(func=_run_generator_challenger)


def _run_generator_challenger(parsed: argparse.Namespace) -> None:
    from spritelab.training.generator_challenger import ChallengerTrainConfig, run_challenger_training

    kwargs = _parsed_config_kwargs(parsed)
    dropout_rates = _parse_dropout_rates(kwargs.pop("structured_field_dropout_rates", None))
    report = run_challenger_training(ChallengerTrainConfig(**kwargs, structured_field_dropout_rates=dropout_rates))
    print(f"Initial train loss: {report['initial_train_loss']:.6f}")
    print(f"Final train loss: {report['final_train_loss']:.6f}")
    if report["val_loss"] is not None:
        print(f"Val loss: {report['val_loss']:.6f}")
    print(f"Outputs written to {parsed.out_dir}")


def _register_sample_generator_challenger(subparsers: argparse._SubParsersAction) -> None:
    sample_challenger = subparsers.add_parser(
        "sample-generator-challenger",
        help="Sample and canonicalize a generator challenger.",
        description=(
            "Sample and canonicalize a generator_challenger checkpoint. "
            "Pass --export-preset v1 to reproduce the official v1 release settings "
            "(Phase 1 EMA checkpoint resolution, CFG 3.0, 30 steps, k16 deterministic "
            "palette projection) -- see docs/v1_default.md."
        ),
    )
    _add_export_preset_argument(sample_challenger, include_v1_1=True)
    sample_challenger.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help=(
            "Path to a generator_challenger checkpoint (.pt). Official v1 path: "
            "experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt "
            "(--export-preset v1 will also resolve a *_last.pt path to its EMA sibling "
            "when present)."
        ),
    )
    sample_challenger.add_argument(
        "--prompts", required=True, type=Path, help="JSONL prompt file, one record per line (see docs/v1_default.md)."
    )
    sample_challenger.add_argument(
        "--out", required=True, type=Path, dest="out_dir", help="Output directory for samples, manifest, and reports."
    )
    sample_challenger.add_argument("--max-samples", type=int, default=64)
    sample_challenger.add_argument("--steps", type=int, default=30)
    sample_challenger.add_argument("--cfg-scale", type=float, default=2.0)
    sample_challenger.add_argument("--max-colors", type=int, default=32)
    sample_challenger.add_argument("--alpha-threshold", type=float, default=0.5)
    sample_challenger.add_argument("--device", default="cpu", help="'cpu' or 'cuda'.")
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
    _add_palette_projection_sampling_arguments(sample_challenger)
    _add_v2_phase0_diagnostic_arguments(sample_challenger)
    sample_challenger.set_defaults(func=_run_sample_generator_challenger)


def _run_sample_generator_challenger(parsed: argparse.Namespace) -> None:
    from spritelab.training.generator_challenger import (
        ChallengerSampleConfig,
        run_sample_generator_challenger,
    )

    report = run_sample_generator_challenger(ChallengerSampleConfig(**_parsed_config_kwargs(parsed)))
    print(f"Generated samples: {report['sample_count']}")
    print(f"Max visible colors: {report['max_visible_color_count']}")
    print(f"Outputs written to {parsed.out_dir}")


def _register_inspect_palette_index_heads(subparsers: argparse._SubParsersAction) -> None:
    inspect_palette_index = subparsers.add_parser(
        "inspect-palette-index-heads",
        help="Evaluate v2 Phase 2 palette/index auxiliary heads on dataset batches (no training).",
    )
    inspect_palette_index.add_argument("--checkpoint", required=True, type=Path)
    inspect_palette_index.add_argument("--dataset", required=True, type=Path)
    inspect_palette_index.add_argument("--training-manifest", required=True, type=Path)
    inspect_palette_index.add_argument("--out", required=True, type=Path)
    inspect_palette_index.add_argument("--device", default="cpu")
    inspect_palette_index.add_argument("--batch-size", type=int, default=32)
    inspect_palette_index.add_argument("--max-batches", type=int, default=32)
    inspect_palette_index.add_argument("--split", default="train")
    inspect_palette_index.add_argument("--cudnn-benchmark", action="store_true", default=False)
    inspect_palette_index.add_argument("--tf32", action="store_true", default=False)
    inspect_palette_index.set_defaults(func=_run_inspect_palette_index_heads)


def _run_inspect_palette_index_heads(parsed: argparse.Namespace) -> None:
    from spritelab.training.palette_index_head_inspect import (
        PaletteIndexHeadInspectConfig,
        run_inspect_palette_index_heads,
    )

    run_inspect_palette_index_heads(PaletteIndexHeadInspectConfig(**_parsed_config_kwargs(parsed)))


def _register_probe_palette_index_decode(subparsers: argparse._SubParsersAction) -> None:
    probe = subparsers.add_parser(
        "probe-palette-index-decode",
        help="Generate samples with palette/index head decode variants (experimental probe, no training).",
    )
    probe.add_argument("--checkpoint", required=True, type=Path)
    probe.add_argument("--prompts", required=True, type=Path)
    probe.add_argument("--dataset", required=True, type=Path)
    probe.add_argument("--out", required=True, type=Path)
    probe.add_argument("--device", default="cpu")
    probe.add_argument("--batch-size", type=int, default=32)
    probe.add_argument("--max-samples", type=int, default=96)
    probe.add_argument("--seed", type=int, default=20260723)
    probe.add_argument("--sample-steps", type=int, default=30)
    probe.add_argument("--cfg-scale", type=float, default=3.0)
    probe.add_argument("--max-colors", type=int, default=16)
    probe.add_argument("--alpha-threshold", type=float, default=0.5)
    probe.add_argument("--cudnn-benchmark", action="store_true", default=False)
    probe.add_argument("--tf32", action="store_true", default=False)
    probe.set_defaults(func=_run_probe_palette_index_decode)


def _run_probe_palette_index_decode(parsed: argparse.Namespace) -> None:
    from spritelab.training.palette_index_decode_probe import (
        DecodeProbeConfig,
        run_decode_probe,
    )

    run_decode_probe(DecodeProbeConfig(**_parsed_config_kwargs(parsed)))
