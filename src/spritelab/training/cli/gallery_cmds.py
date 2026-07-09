"""V1/V2 gallery and eval prompt building commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.training.cli._args import DEFAULT_V1_GALLERY_CHECKPOINT, _add_export_preset_argument


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_build_v1_gallery(subparsers)
    _register_v1_gallery_gui(subparsers)
    _register_build_v2_eval_prompts(subparsers)


def _register_build_v1_gallery(subparsers: argparse._SubParsersAction) -> None:
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
    build_v1_gallery.set_defaults(func=_run_build_v1_gallery)


def _run_build_v1_gallery(parsed: argparse.Namespace) -> None:
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


def _register_v1_gallery_gui(subparsers: argparse._SubParsersAction) -> None:
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
    v1_gallery_gui.set_defaults(func=_run_v1_gallery_gui)


def _run_v1_gallery_gui(parsed: argparse.Namespace) -> None:
    from spritelab.training.v1_gallery_gui import launch_v1_gallery_gui

    try:
        launch_v1_gallery_gui(out_dir=parsed.out_dir, host=parsed.host, port=parsed.port)
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc


def _register_build_v2_eval_prompts(subparsers: argparse._SubParsersAction) -> None:
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
    build_eval_prompts.set_defaults(func=_run_build_v2_eval_prompts)


def _run_build_v2_eval_prompts(parsed: argparse.Namespace) -> None:
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
