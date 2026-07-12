"""Shared argument helpers and constants for training CLI commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from spritelab.training.palette_swap import DEFAULT_SWAP_FAMILIES_TEXT

DEFAULT_V1_GALLERY_CHECKPOINT = Path("experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt")

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
    values.pop("func", None)
    values.pop("verbose", None)
    return values


def _add_speed_option_arguments(parser: argparse.ArgumentParser) -> None:
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


def _add_role_ramp_transplant_arguments(parser: argparse.ArgumentParser) -> None:
    """Add default-off Phase 3 real-ramp transplant training controls."""
    parser.add_argument("--role-ramp-transplant-prob", type=float, default=0.0)
    parser.add_argument("--role-ramp-transplant-keep-original-prob", type=float, default=0.5)
    parser.add_argument("--role-ramp-transplant-exclude-families", default="gold,brown")
    parser.add_argument(
        "--role-ramp-transplant-require-trusted-role-map",
        action="store_true",
        default=True,
        dest="role_ramp_transplant_require_trusted_role_map",
    )
    parser.add_argument(
        "--no-role-ramp-transplant-require-trusted-role-map",
        action="store_false",
        dest="role_ramp_transplant_require_trusted_role_map",
    )
    parser.add_argument("--role-ramp-transplant-debug-samples", type=int, default=0)
    parser.add_argument("--role-ramp-transplant-max-resample-attempts", type=int, default=8)
    parser.add_argument(
        "--role-ramp-transplant-require-fill-target-match",
        action="store_true",
        default=True,
        dest="role_ramp_transplant_require_fill_target_match",
    )
    parser.add_argument(
        "--no-role-ramp-transplant-require-fill-target-match",
        action="store_false",
        dest="role_ramp_transplant_require_fill_target_match",
    )
    parser.add_argument("--role-ramp-transplant-min-primary-fill-coverage", type=float, default=0.03)


def _add_palette_conditioning_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add default-off v3 canonical palette conditioning controls."""
    parser.add_argument("--palette-conditioning", action="store_true", default=False)
    parser.add_argument("--palette-conditioning-dropout", type=float, default=0.0)
    parser.add_argument("--palette-conditioning-dim", type=int, default=64)
    parser.add_argument("--palette-conditioning-inject", choices=["decoder", "all", "bottleneck"], default="decoder")


def _add_palette_conditioning_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--palette-conditioning-source", choices=["none", "source", "retrieved"], default="none")
    parser.add_argument("--palette-conditioning-dataset", type=Path)
    parser.add_argument("--palette-conditioning-training-manifest", type=Path)
    parser.add_argument("--palette-conditioning-exclude-exact-prompt-target", action="store_true", default=False)


def _bool_str(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _add_palette_swap_conservative_arguments(parser: argparse.ArgumentParser) -> None:
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
