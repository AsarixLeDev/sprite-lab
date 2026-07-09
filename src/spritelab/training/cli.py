"""CLI for semantic training baselines."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from spritelab.training.conditioning import CONDITIONING_MODES, DEFAULT_CONDITIONING_MODE
from spritelab.training.palette_swap import DEFAULT_SWAP_FAMILIES_TEXT

# Kept as a literal (rather than importing spritelab.training.v1_gallery at module scope) so
# that `spritelab train <subcommand>` argument parsing stays free of heavy sampling imports
# for subcommands that never touch the v1 gallery. Must match v1_gallery.DEFAULT_V1_CHECKPOINT.
DEFAULT_V1_GALLERY_CHECKPOINT = Path("experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt")

# Kept as a literal (rather than importing spritelab.training.generator_challenger at module
# scope) for the same reason as DEFAULT_V1_GALLERY_CHECKPOINT above. Must match
# generator_challenger.NULL_FIELD_CHOICES.
NULL_FIELD_CHOICES: tuple[str, ...] = (
    "caption",
    "semantic",
    "category",
    "object_id",
    "base_object",
    "colors",
    "materials",
    "shapes",
    "function",
    "style",
    "structured",
)

# Kept as literals for the same reason as above. Must match
# generator_challenger.V1_PRESET_ALIASES / V1_1_PRESET_ALIASES / V1_1_CFG_BASE_SCALE /
# V1_1_CFG_COLOR_SCALE. v1.1 is an optional color-strong preset (see
# docs/v1_1_factored_cfg.md); v1 remains the default everywhere.
V1_PRESET_ALIASES: tuple[str, ...] = ("v1", "phase1_v1")
V1_1_PRESET_ALIASES: tuple[str, ...] = ("v1.1", "v1_1", "phase1_v1_1")
V1_1_CFG_BASE_SCALE = 2.5
V1_1_CFG_COLOR_SCALE = 3.0


def _normalize_export_preset(export_preset: str | None) -> str | None:
    normalized = str(export_preset or "").strip().lower()
    if normalized in V1_PRESET_ALIASES:
        return "v1"
    if normalized in V1_1_PRESET_ALIASES:
        return "v1.1"
    return None


def _parse_dropout_rates(raw: str | None) -> dict[str, float] | None:
    """Parse a comma-separated per-group dropout rate string.

    >>> _parse_dropout_rates("category=0.10,object_id=0.35,colors=0.15")
    {"category": 0.10, "object_id": 0.35, "colors": 0.15}
    """
    if not raw or not str(raw).strip():
        return None
    result: dict[str, float] = {}
    known_groups = {
        group[0]
        for group in (
            ("category", ("category_id",)),
            ("object_id", ("object_id",)),
            ("base_object", ("base_object_id",)),
            ("colors", ("primary_color_id", "color_multi_hot")),
            ("materials", ("material_multi_hot",)),
            ("shapes", ("shape_multi_hot",)),
            ("function", ("function_multi_hot",)),
            ("style", ("style_multi_hot",)),
        )
    }
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid dropout rate format: {token!r}. Expected 'group=rate'.")
        group, rate_str = token.split("=", 1)
        group = group.strip()
        rate_str = rate_str.strip()
        if group not in known_groups:
            raise ValueError(f"Unknown dropout group {group!r}. Expected one of {sorted(known_groups)}.")
        try:
            rate = float(rate_str)
        except ValueError:
            raise ValueError(f"Invalid dropout rate {rate_str!r} for group {group!r}.") from None
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"Dropout rate for {group!r} must be in [0, 1], got {rate}.")
        result[group] = rate
    return result if result else None


def _parsed_config_kwargs(parsed: argparse.Namespace) -> dict[str, object]:
    values = vars(parsed).copy()
    values.pop("subcommand", None)
    return values


def _add_speed_option_arguments(parser: argparse.ArgumentParser) -> None:
    """Opt-in training-loop speed knobs; every default reproduces today's behaviour."""

    parser.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="Sync loss to Python and log a train_metrics.jsonl line every N steps "
        "(the final step is always logged). 1 (default) logs every step, unchanged.",
    )
    parser.add_argument(
        "--fused-adamw",
        action="store_true",
        default=False,
        help="Use torch's fused AdamW kernel (CUDA only; falls back with a warning if unsupported). "
        "Numerics can differ slightly from the default optimizer.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        default=False,
        help="Let cuDNN autotune convolution algorithms for the fixed input shapes/batch size. "
        "First few steps are slower while it searches.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        default=False,
        help="Allow TF32 matmul/conv accumulation on Ampere+ GPUs. Minor effect expected when "
        "--amp bf16 autocast is already enabled.",
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=0,
        help="Cap initial/final/val loss evaluation to this many batches; 0 (default) evaluates "
        "the full loader, unchanged. A positive value only changes the *reported* loss estimates.",
    )


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
        "--palette-swap-stochastic",
        action="store_true",
        default=False,
        dest="palette_swap_stochastic",
        help="Include a draw-level index in the palette-swap seed so repeated visits can choose different targets.",
    )
    parser.add_argument(
        "--palette-swap-keep-original-prob",
        type=float,
        default=0.0,
        dest="palette_swap_keep_original_prob",
        help="With stochastic palette swap, keep eligible samples unchanged with this probability.",
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
        "--palette-swap-allow-colorless-caption-if-semantic-color",
        action="store_true",
        default=False,
        dest="palette_swap_allow_colorless_caption_if_semantic_color",
        help="Allow colorless captions when structured color is explicit; structured fields update but captions are not prepended.",
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


def _add_palette_projection_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-palette", action="store_true", default=False)
    parser.add_argument("--no-project-palette", action="store_false", dest="project_palette")
    parser.add_argument("--project-palette-target-colors", type=int, default=16)
    parser.add_argument("--project-palette-min-pixel-share", type=float, default=0.01)
    parser.add_argument("--project-palette-method", choices=["deterministic_kmeans"], default="deterministic_kmeans")


def _add_v2_phase0_diagnostic_arguments(parser: argparse.ArgumentParser) -> None:
    """No-training v2 Phase 0 diagnostic flags (see docs/v2_phase0_diagnostics.md).

    All default to off/empty and reproduce the existing v1 sampling behavior exactly
    unless explicitly used.
    """

    parser.add_argument(
        "--factored-cfg",
        action="store_true",
        default=False,
        dest="factored_cfg",
        help=(
            "Split CFG into independent base (uncond->no-color) and color (no-color->full) "
            "guidance terms instead of the single --cfg-scale term. Off by default."
        ),
    )
    parser.add_argument(
        "--cfg-base-scale",
        type=float,
        default=None,
        dest="cfg_base_scale",
        help="Base guidance scale used when --factored-cfg is set. Defaults to --cfg-scale if omitted.",
    )
    parser.add_argument(
        "--cfg-color-scale",
        type=float,
        default=None,
        dest="cfg_color_scale",
        help="Color guidance scale used when --factored-cfg is set. Defaults to --cfg-scale if omitted.",
    )
    parser.add_argument(
        "--null-fields",
        default="",
        dest="null_fields",
        help=(
            "Comma-separated conditioning fields to null at sample time for diagnostics, "
            f"e.g. 'colors,object_id'. Choices: {', '.join(NULL_FIELD_CHOICES)}. Empty (default) is a no-op."
        ),
    )


def _add_export_preset_argument(
    parser: argparse.ArgumentParser,
    *,
    include_v1_1: bool = False,
    default: str | None = None,
) -> None:
    choices = [*V1_PRESET_ALIASES]
    help_text = "Named export/sampling preset. v1 uses Phase 1 EMA if available, CFG 3.0, 30 steps, and k16 projection."
    if include_v1_1:
        choices = [*choices, *V1_1_PRESET_ALIASES]
        help_text += (
            " v1.1 (aliases v1_1, phase1_v1_1) is an optional color-strong preset: v1 base "
            f"settings plus factored CFG (base={V1_1_CFG_BASE_SCALE}, color={V1_1_CFG_COLOR_SCALE}). "
            "v1 remains the default; v1.1 must be requested explicitly. See docs/v1_1_factored_cfg.md."
        )
    parser.add_argument(
        "--export-preset",
        "--preset",
        choices=choices,
        default=default,
        dest="export_preset",
        help=help_text,
    )


def _argv_has_option(argv: Sequence[str], *names: str) -> bool:
    option_names = tuple(str(name) for name in names)
    for item in argv:
        text = str(item)
        if text in option_names or any(text.startswith(f"{name}=") for name in option_names):
            return True
    return False


def _apply_export_preset_defaults(parsed: argparse.Namespace, argv: Sequence[str]) -> None:
    preset = _normalize_export_preset(getattr(parsed, "export_preset", None))
    if preset is None:
        return
    if parsed.subcommand == "sample-generator-challenger":
        if not _argv_has_option(argv, "--steps"):
            parsed.steps = 30
        if not _argv_has_option(argv, "--cfg-scale"):
            parsed.cfg_scale = 3.0
        if not _argv_has_option(argv, "--max-colors"):
            parsed.max_colors = 32
        if not _argv_has_option(argv, "--alpha-threshold"):
            parsed.alpha_threshold = 0.5
        if not _argv_has_option(argv, "--dither", "--no-dither"):
            parsed.dither = False
        if not _argv_has_option(argv, "--write-raw-rgba", "--no-write-raw-rgba"):
            parsed.write_raw_rgba = True
        if not _argv_has_option(argv, "--write-hard-rgba", "--no-write-hard-rgba"):
            parsed.write_hard_rgba = True
        _apply_projection_preset_defaults(parsed, argv)
        if preset == "v1.1":
            _apply_v1_1_factored_cfg_defaults(parsed, argv)
    elif parsed.subcommand == "audit-challenger-full-v4":
        if not _argv_has_option(argv, "--sample-ema", "--no-sample-ema"):
            parsed.sample_ema = True
        if not _argv_has_option(argv, "--sample-steps"):
            parsed.sample_steps = 30
        if not _argv_has_option(argv, "--cfg-scale"):
            parsed.cfg_scale = 3.0
        if not _argv_has_option(argv, "--max-colors"):
            parsed.max_colors = 32
        if not _argv_has_option(argv, "--alpha-threshold"):
            parsed.alpha_threshold = 0.5
        _apply_projection_preset_defaults(parsed, argv)


def _apply_v1_1_factored_cfg_defaults(parsed: argparse.Namespace, argv: Sequence[str]) -> None:
    """v1.1-only: layer factored CFG on top of the v1 base settings already applied.

    Off-by-default flags (--factored-cfg/--cfg-base-scale/--cfg-color-scale, see
    docs/v2_phase0_diagnostics.md) only get preset values here when the user did not
    already pass them explicitly, so `--export-preset v1.1 --cfg-base-scale 1.0` still
    honors the explicit override.
    """

    if not _argv_has_option(argv, "--factored-cfg"):
        parsed.factored_cfg = True
    if not _argv_has_option(argv, "--cfg-base-scale"):
        parsed.cfg_base_scale = V1_1_CFG_BASE_SCALE
    if not _argv_has_option(argv, "--cfg-color-scale"):
        parsed.cfg_color_scale = V1_1_CFG_COLOR_SCALE


def _apply_projection_preset_defaults(parsed: argparse.Namespace, argv: Sequence[str]) -> None:
    if not _argv_has_option(argv, "--project-palette", "--no-project-palette"):
        parsed.project_palette = True
    if not _argv_has_option(argv, "--project-palette-target-colors"):
        parsed.project_palette_target_colors = 16
    if not _argv_has_option(argv, "--project-palette-min-pixel-share"):
        parsed.project_palette_min_pixel_share = 0.01
    if not _argv_has_option(argv, "--project-palette-method"):
        parsed.project_palette_method = "deterministic_kmeans"


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
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

    make_overfit = subparsers.add_parser("make-overfit-subset", help="Write a deterministic sprite-id subset.")
    make_overfit.add_argument("--dataset", required=True, type=Path)
    make_overfit.add_argument("--training-manifest", required=True, type=Path)
    make_overfit.add_argument("--out", required=True, type=Path)
    make_overfit.add_argument("--count", required=True, type=int)
    make_overfit.add_argument("--seed", type=int, default=123)
    make_overfit.add_argument("--split", default="train")
    make_overfit.add_argument("--stratify")

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

    build_ood = subparsers.add_parser(
        "build-ood-compositional-prompts",
        help="Write the default color/object OOD compositional prompt set.",
    )
    build_ood.add_argument("--out", required=True, type=Path)
    build_ood.add_argument("--max-prompts", type=int)

    build_v1_gallery = subparsers.add_parser(
        "build-v1-gallery",
        help="Build the deterministic v1 demo/release gallery: prompts -> v1 preset sampling -> QA/review -> contact sheets -> report.",
        description=(
            "Build the official v1 demo/release gallery end to end: builds (or reads) a "
            "prompt set, samples it with the v1 export preset (Phase 1 EMA checkpoint, "
            "CFG 3.0, 30 steps, k16 deterministic palette projection), runs QA and "
            "structural review, writes contact sheets, and writes a Markdown/JSON report. "
            "Never trains a model. See docs/v1_default.md."
        ),
    )
    build_v1_gallery.add_argument(
        "--out", required=True, type=Path, dest="out_dir", help="Output directory (see docs/v1_default.md for layout)."
    )
    build_v1_gallery.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_V1_GALLERY_CHECKPOINT,
        help=f"Phase 1 checkpoint to sample. Defaults to the official v1 checkpoint: {DEFAULT_V1_GALLERY_CHECKPOINT}",
    )
    build_v1_gallery.add_argument(
        "--prompts",
        type=Path,
        help="Optional custom JSONL prompt file. Defaults to the built-in deterministic v1 gallery prompt set.",
    )
    _add_export_preset_argument(build_v1_gallery, include_v1_1=True, default="v1")
    build_v1_gallery.add_argument(
        "--device", default="cpu", help="'cpu' or 'cuda'. Use 'cuda' to match the validated release gallery."
    )
    build_v1_gallery.add_argument("--seed", type=int, default=20260723)
    build_v1_gallery.add_argument("--batch-size", type=int, default=32)
    build_v1_gallery.add_argument("--num-samples", type=int, help="Cap the number of prompts/samples.")
    build_v1_gallery.add_argument("--categories", help="Comma-separated category filter for the built-in prompt set.")
    build_v1_gallery.add_argument("--contact-sheet-columns", type=int, default=8)
    build_v1_gallery.add_argument(
        "--include-ood",
        action="store_true",
        default=True,
        help="Include a trimmed OOD compositional prompt slice (default on).",
    )
    build_v1_gallery.add_argument("--no-include-ood", action="store_false", dest="include_ood")
    build_v1_gallery.add_argument("--include-grounded", action="store_true", default=True)
    build_v1_gallery.add_argument("--no-include-grounded", action="store_false", dest="include_grounded")
    build_v1_gallery.add_argument("--include-stress-prompts", action="store_true", default=True)
    build_v1_gallery.add_argument("--no-include-stress-prompts", action="store_false", dest="include_stress_prompts")

    v1_gallery_gui = subparsers.add_parser(
        "v1-gallery-gui",
        help="Launch the local v1 gallery GUI (requires the 'gradio' extra).",
        description=(
            "Launch a local Gradio GUI to build the v1 demo gallery: pick an output "
            "directory and (optionally) a custom prompt file, sample with the official "
            "v1 export preset, and preview the resulting contact sheets. Never trains a "
            "model."
        ),
    )
    v1_gallery_gui.add_argument("--out", default="experiments/v1_gallery_gui", dest="out_dir")
    v1_gallery_gui.add_argument("--host", default="127.0.0.1")
    v1_gallery_gui.add_argument("--port", type=int)

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

    build_eval_prompts = subparsers.add_parser(
        "build-v2-eval-prompts",
        help="Build a larger deterministic OOD/eval prompt suite for v2 Phase 0.",
        description=(
            "Reads the training manifest vocab and builds a JSONL prompt file "
            "covering category-color grids, object-color pairs, rare combos, "
            "style stress, and in-distribution anchors. Deterministic given the "
            "same dataset/manifest/seed/target-count."
        ),
    )
    build_eval_prompts.add_argument("--dataset", required=True, type=Path)
    build_eval_prompts.add_argument("--training-manifest", required=True, type=Path)
    build_eval_prompts.add_argument("--out", required=True, type=Path)
    build_eval_prompts.add_argument("--target-count", type=int, default=384)
    build_eval_prompts.add_argument("--seed", type=int, default=20260706)
    build_eval_prompts.add_argument("--include-grounded-grid", action="store_true", default=True)
    build_eval_prompts.add_argument("--no-include-grounded-grid", action="store_false", dest="include_grounded_grid")
    build_eval_prompts.add_argument("--include-compositional", action="store_true", default=True)
    build_eval_prompts.add_argument("--no-include-compositional", action="store_false", dest="include_compositional")
    build_eval_prompts.add_argument("--include-rare-combos", action="store_true", default=True)
    build_eval_prompts.add_argument("--no-include-rare-combos", action="store_false", dest="include_rare_combos")
    build_eval_prompts.add_argument("--include-style-stress", action="store_true", default=True)
    build_eval_prompts.add_argument("--no-include-style-stress", action="store_false", dest="include_style_stress")
    build_eval_prompts.add_argument("--out-report", action="store_true", default=True)
    build_eval_prompts.add_argument("--no-out-report", action="store_false", dest="out_report")

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
    monitor.add_argument(
        "--html", type=Path, default=None, dest="html_path", help="Also write an auto-refreshing HTML dashboard."
    )
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

    parsed = parser.parse_args(raw_argv)
    _apply_export_preset_defaults(parsed, raw_argv)
    try:
        if parsed.subcommand == "inspect-data":
            from spritelab.training.inspect_data import inspect_training_data, print_inspection

            summary = inspect_training_data(
                dataset_dir=parsed.dataset,
                training_manifest=parsed.training_manifest,
                split=parsed.split,
                batch_size=parsed.batch_size,
                max_records=parsed.max_records,
            )
            print_inspection(summary)
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
        elif parsed.subcommand == "project-generated-palette":
            from spritelab.training.palette_projection import PaletteProjectionConfig, project_generated_palette

            report = project_generated_palette(PaletteProjectionConfig(**_parsed_config_kwargs(parsed)))
            print(f"Projected samples: {report['sample_count']}")
            print(
                "Median visible colors: "
                f"{report['median_visible_color_count_before']} -> {report['median_visible_color_count_after']}"
            )
            print(f"Mean RGB MAE visible: {report['mean_rgb_mae_visible']}")
            print(f"Outputs written to {parsed.out}")
        elif parsed.subcommand == "generator-challenger":
            from spritelab.training.generator_challenger import ChallengerTrainConfig, run_challenger_training

            kwargs = _parsed_config_kwargs(parsed)
            dropout_rates = _parse_dropout_rates(kwargs.pop("structured_field_dropout_rates", None))
            report = run_challenger_training(
                ChallengerTrainConfig(**kwargs, structured_field_dropout_rates=dropout_rates)
            )
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

            kwargs = _parsed_config_kwargs(parsed)
            dropout_rates = _parse_dropout_rates(kwargs.pop("structured_field_dropout_rates", None))
            report = run_full_v4_challenger_audit(
                FullV4ChallengerAuditConfig(**kwargs, structured_field_dropout_rates=dropout_rates)
            )
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
        elif parsed.subcommand == "build-v1-gallery":
            from spritelab.training.v1_gallery import BuildV1GalleryConfig, build_v1_gallery_demo

            categories = None
            if parsed.categories:
                categories = tuple(token.strip() for token in str(parsed.categories).split(",") if token.strip())

            report = build_v1_gallery_demo(
                BuildV1GalleryConfig(
                    out_dir=parsed.out_dir,
                    checkpoint=parsed.checkpoint,
                    prompts=parsed.prompts,
                    export_preset=parsed.export_preset,
                    device=parsed.device,
                    seed=parsed.seed,
                    batch_size=parsed.batch_size,
                    num_samples=parsed.num_samples,
                    categories=categories,
                    contact_sheet_columns=parsed.contact_sheet_columns,
                    include_ood=parsed.include_ood,
                    include_grounded=parsed.include_grounded,
                    include_stress_prompts=parsed.include_stress_prompts,
                )
            )
            print(f"Prompt count: {report['prompt_set']['prompt_count']}")
            print(f"Sample count: {report['sample_count']}")
            print(f"Samples written to {parsed.out_dir / 'samples'}")
            print(f"Contact sheets written to {parsed.out_dir / 'contact_sheets'}")
            print(f"Report written to {parsed.out_dir / 'v1_gallery_report.md'}")
        elif parsed.subcommand == "v1-gallery-gui":
            from spritelab.training.v1_gallery_gui import launch_v1_gallery_gui

            try:
                launch_v1_gallery_gui(out_dir=parsed.out_dir, host=parsed.host, port=parsed.port)
            except RuntimeError as exc:
                if "requires gradio" not in str(exc):
                    raise
                print(str(exc))
                raise SystemExit(1) from exc
        elif parsed.subcommand == "run-v2-phase0-eval":
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

            summary = run_v2_phase0_eval(
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
                )
            )
            if not parsed.dry_run:
                print(f"Summary reports: {parsed.out / 'summaries'}")
        elif parsed.subcommand == "build-v2-eval-prompts":
            from spritelab.training.v2_eval_prompts import V2EvalPromptsConfig, build_v2_eval_prompts

            report = build_v2_eval_prompts(
                V2EvalPromptsConfig(
                    dataset=parsed.dataset,
                    training_manifest=parsed.training_manifest,
                    out=parsed.out,
                    target_count=parsed.target_count,
                    seed=parsed.seed,
                    include_grounded_grid=parsed.include_grounded_grid,
                    include_compositional=parsed.include_compositional,
                    include_rare_combos=parsed.include_rare_combos,
                    include_style_stress=parsed.include_style_stress,
                    out_report=parsed.out_report,
                )
            )
            print(f"Prompts written: {report['prompt_count']} (target: {parsed.target_count})")
            print(f"Families: {report['families']}")
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
        elif parsed.subcommand == "inspect-palette-index-heads":
            from spritelab.training.palette_index_head_inspect import (
                PaletteIndexHeadInspectConfig,
                run_inspect_palette_index_heads,
            )

            run_inspect_palette_index_heads(PaletteIndexHeadInspectConfig(**_parsed_config_kwargs(parsed)))
    except RuntimeError as exc:
        if "PyTorch is required" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc
