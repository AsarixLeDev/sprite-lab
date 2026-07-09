"""Local Gradio GUI for the official v1 demo/release gallery.

A guided, user-friendly wrapper around
``spritelab.training.v1_gallery.build_v1_gallery_demo``: choose an output
directory and a prompt set (the built-in v1 set or your own JSONL file),
optionally preview the prompts, check that the checkpoint resolves, then
sample with the official v1 export preset and browse the generated sprites
and contact sheets. It never trains a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spritelab.training.sample_generator import read_prompt_records
from spritelab.training.v1_gallery import (
    DEFAULT_V1_CHECKPOINT,
    OFFICIAL_V1_STATEMENT,
    V1_GALLERY_CATEGORY_OBJECTS,
    BuildV1GalleryConfig,
    build_default_v1_gallery_prompts,
    build_v1_gallery_demo,
)

BUILTIN_SOURCE = "Built-in v1 prompt set"
CUSTOM_SOURCE = "Custom prompts file (JSONL)"

FAMILY_GROUNDED = "Grounded & compositional"
FAMILY_STRESS = "Style-stress"
FAMILY_OOD = "OOD compositional"
PROMPT_FAMILY_CHOICES = [FAMILY_GROUNDED, FAMILY_STRESS, FAMILY_OOD]

CATEGORY_CHOICES = list(V1_GALLERY_CATEGORY_OBJECTS)

_INTRO = (
    "Generate a gallery of 32x32 pixel-art sprites with the **official v1 preset** "
    "(Phase 1 EMA checkpoint, CFG 3.0, 30 steps, k16 deterministic palette projection).\n\n"
    "1. Set an **output directory** and confirm the **checkpoint** resolves.\n"
    "2. Choose a **prompt set** — the built-in v1 set or your own JSONL file — and "
    "optionally **Preview prompts** first.\n"
    "3. Click **Generate gallery**. Sampling on CPU is slow; use `cuda` for the "
    "release-quality run.\n\n"
    "This tool never trains a model."
)


def launch_v1_gallery_gui(
    out_dir: str | Path = "experiments/v1_gallery_gui",
    host: str = "127.0.0.1",
    port: int | None = None,
) -> None:
    """Launch the local v1 gallery GUI."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The v1 gallery GUI requires gradio. Install it with: pip install gradio") from exc

    def on_source_change(source: str) -> tuple[Any, Any]:
        is_custom = source == CUSTOM_SOURCE
        return (
            gr.update(visible=is_custom),
            gr.update(visible=not is_custom),
        )

    def check_checkpoint(checkpoint_value: str) -> str:
        return _checkpoint_status(checkpoint_value)

    def preview(
        source: str,
        prompts_value: str,
        categories_value: list[str],
        families_value: list[str],
    ) -> str:
        return _preview_prompt_set(source, prompts_value, categories_value, families_value)

    def build(
        out_dir_value: str,
        checkpoint_value: str,
        device_value: str,
        source: str,
        prompts_value: str,
        categories_value: list[str],
        families_value: list[str],
        seed_value: float,
        batch_size_value: float,
        num_samples_value: float | None,
        contact_columns_value: float,
        progress: Any = gr.Progress(track_tqdm=False),  # noqa: B008
    ) -> tuple[str, list[str], list[tuple[str, str]], str]:
        checkpoint = _resolve_v1_checkpoint(Path(checkpoint_value.strip() or str(DEFAULT_V1_CHECKPOINT)))
        if not checkpoint.is_file():
            return (_checkpoint_status(checkpoint_value), [], [], "")

        use_custom = source == CUSTOM_SOURCE
        if use_custom and not prompts_value.strip():
            return ("### Please choose a prompts JSONL file, or switch to the built-in v1 prompt set.", [], [], "")
        if use_custom and not Path(prompts_value.strip()).is_file():
            return (f"### Prompts file not found\n\n`{prompts_value.strip()}`", [], [], "")

        include_grounded, include_stress, include_ood = _families_to_flags(families_value)
        categories = tuple(categories_value) if (categories_value and not use_custom) else None

        progress(0.05, desc="Preparing prompts and loading checkpoint…")
        try:
            progress(0.15, desc="Sampling sprites with the v1 preset (this can take a while)…")
            report = build_v1_gallery_demo(
                BuildV1GalleryConfig(
                    out_dir=Path(out_dir_value.strip() or "experiments/v1_gallery_gui"),
                    checkpoint=checkpoint,
                    prompts=Path(prompts_value.strip()) if use_custom else None,
                    device=device_value,
                    seed=int(seed_value or 20260723),
                    batch_size=int(batch_size_value or 32),
                    num_samples=int(num_samples_value) if num_samples_value else None,
                    categories=categories,
                    contact_sheet_columns=int(contact_columns_value or 8),
                    include_ood=include_ood,
                    include_grounded=include_grounded,
                    include_stress_prompts=include_stress,
                )
            )
        except FileNotFoundError as exc:
            return (f"### Generation failed\n\n```\n{exc}\n```", [], [], "")
        except Exception as exc:
            return (f"### Generation failed\n\n```\n{type(exc).__name__}: {exc}\n```", [], [], "")

        progress(0.95, desc="Collecting outputs…")
        return (
            _report_summary(report),
            _contact_sheet_paths(report),
            _sample_image_items(report),
            _report_markdown_text(report),
        )

    with gr.Blocks(title="Sprite Lab v1 Gallery") as demo:
        gr.Markdown("# 🎨 Sprite Lab — v1 Gallery")
        gr.Markdown(_INTRO)

        with gr.Accordion("About the v1 preset", open=False):
            gr.Markdown(OFFICIAL_V1_STATEMENT.replace("\n", "  \n"))

        with gr.Group():
            gr.Markdown("### 1. Output & model")
            with gr.Row():
                out_dir_box = gr.Textbox(
                    label="Output directory",
                    value=str(out_dir),
                    scale=3,
                    info="Samples, contact sheets, and reports are written here.",
                )
                device_box = gr.Radio(
                    label="Device",
                    choices=["cpu", "cuda"],
                    value="cpu",
                    scale=1,
                    info="Use 'cuda' for the release-quality run.",
                )
            with gr.Row():
                checkpoint_box = gr.Textbox(
                    label="Checkpoint",
                    value=str(DEFAULT_V1_CHECKPOINT),
                    scale=3,
                    info="Phase 1 EMA checkpoint. A '*_last.pt' path resolves to its EMA sibling.",
                )
                check_button = gr.Button("Check checkpoint", scale=1)
            checkpoint_status = gr.Markdown()

        with gr.Group():
            gr.Markdown("### 2. Prompts")
            source_radio = gr.Radio(
                label="Prompt set",
                choices=[BUILTIN_SOURCE, CUSTOM_SOURCE],
                value=BUILTIN_SOURCE,
            )
            custom_prompts_box = gr.Textbox(
                label="Custom prompts JSONL path",
                placeholder="path\\to\\prompts.jsonl",
                visible=False,
                info="One JSON prompt record per line.",
            )
            with gr.Group(visible=True) as builtin_group:
                category_group = gr.CheckboxGroup(
                    label="Categories (none selected = all)",
                    choices=CATEGORY_CHOICES,
                    value=[],
                )
                family_group = gr.CheckboxGroup(
                    label="Prompt families",
                    choices=PROMPT_FAMILY_CHOICES,
                    value=PROMPT_FAMILY_CHOICES,
                )
            with gr.Row():
                preview_button = gr.Button("Preview prompts")
            preview_markdown = gr.Markdown()

        with gr.Accordion("3. Advanced sampling options", open=False):
            with gr.Row():
                seed_box = gr.Number(label="Seed", value=20260723, precision=0)
                batch_size_box = gr.Number(label="Batch size", value=32, precision=0)
                num_samples_box = gr.Number(label="Max samples (blank = all)", value=None, precision=0)
                contact_columns_box = gr.Number(label="Contact-sheet columns", value=8, precision=0)

        build_button = gr.Button("🚀 Generate gallery", variant="primary", size="lg")

        gr.Markdown("### Results")
        status_markdown = gr.Markdown()
        with gr.Tab("Sprites"):
            sprites_gallery = gr.Gallery(
                label="Generated sprites (projected)", columns=8, height=520, object_fit="contain"
            )
        with gr.Tab("Contact sheets"):
            contact_sheet_gallery = gr.Gallery(label="Contact sheets", columns=2, height=520)
        with gr.Accordion("Full report (Markdown)", open=False):
            report_markdown = gr.Markdown()

        source_radio.change(
            on_source_change,
            inputs=[source_radio],
            outputs=[custom_prompts_box, builtin_group],
        )
        check_button.click(check_checkpoint, inputs=[checkpoint_box], outputs=[checkpoint_status])
        preview_button.click(
            preview,
            inputs=[source_radio, custom_prompts_box, category_group, family_group],
            outputs=[preview_markdown],
        )
        build_button.click(
            build,
            inputs=[
                out_dir_box,
                checkpoint_box,
                device_box,
                source_radio,
                custom_prompts_box,
                category_group,
                family_group,
                seed_box,
                batch_size_box,
                num_samples_box,
                contact_columns_box,
            ],
            outputs=[status_markdown, contact_sheet_gallery, sprites_gallery, report_markdown],
        )

    demo.launch(server_name=host, server_port=port)


def _families_to_flags(families: list[str] | None) -> tuple[bool, bool, bool]:
    selected = set(families or [])
    return (
        FAMILY_GROUNDED in selected,
        FAMILY_STRESS in selected,
        FAMILY_OOD in selected,
    )


def _resolve_v1_checkpoint(checkpoint: Path) -> Path:
    """Mirror the v1 export-preset checkpoint resolution for display/pre-checks.

    Keeps parity with ``generator_challenger._resolve_sample_export_checkpoint``
    without importing the (torch-loading) sampler module.
    """

    path = Path(checkpoint)
    if path.is_file():
        return path
    if path.is_dir():
        for name in (
            "checkpoint_last_ema.pt",
            "checkpoint_best_ema.pt",
            "checkpoint_last.pt",
            "checkpoint_best.pt",
        ):
            candidate = path / name
            if candidate.is_file():
                return candidate
        return path
    if path.suffix == ".pt" and not path.stem.endswith("_ema"):
        sibling = path.with_name(f"{path.stem}_ema{path.suffix}")
        if sibling.is_file():
            return sibling
    return path


def _checkpoint_status(checkpoint_value: str) -> str:
    requested = Path(checkpoint_value.strip() or str(DEFAULT_V1_CHECKPOINT))
    resolved = _resolve_v1_checkpoint(requested)
    if resolved.is_file():
        if resolved != requested:
            return f"✅ Found (resolved to EMA sibling): `{resolved}`"
        return f"✅ Found: `{resolved}`"
    return (
        f"❌ Checkpoint not found: `{requested}`\n\n"
        "The official v1 path is "
        "`experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt` "
        "(see docs/v1_default.md). Point at wherever the checkpoint lives, or train "
        "the Phase 1 challenger first."
    )


def _preview_prompt_set(
    source: str,
    prompts_value: str,
    categories_value: list[str] | None,
    families_value: list[str] | None,
) -> str:
    if source == CUSTOM_SOURCE:
        path = Path(prompts_value.strip()) if prompts_value.strip() else None
        if path is None:
            return "Choose a prompts JSONL file to preview it."
        if not path.is_file():
            return f"### Prompts file not found\n\n`{path}`"
        try:
            rows = read_prompt_records(path)
        except Exception as exc:
            return f"### Could not read prompts file\n\n```\n{exc}\n```"
        return _prompt_list_markdown(rows, note=f"Custom file: `{path}`")

    include_grounded, include_stress, _include_ood = _families_to_flags(families_value)
    categories = tuple(categories_value) if categories_value else None
    if not (include_grounded or include_stress):
        return "Select at least one built-in prompt family (or enable OOD) to preview."
    rows = build_default_v1_gallery_prompts(
        categories=categories,
        include_grounded=include_grounded,
        include_stress_prompts=include_stress,
    )
    note_parts = [f"Categories: {', '.join(categories) if categories else 'all'}"]
    if FAMILY_OOD in set(families_value or []):
        note_parts.append("+ up to 16 OOD compositional prompts appended at generation time")
    return _prompt_list_markdown(rows, note="; ".join(note_parts))


def _prompt_list_markdown(rows: list[dict[str, Any]], *, note: str, limit: int = 24) -> str:
    lines = [f"**{len(rows)} prompt(s).** {note}", ""]
    for row in rows[:limit]:
        text = str(row.get("prompt") or row.get("prompt_id") or "").strip()
        category = str(row.get("category") or "")
        suffix = f" _({category})_" if category else ""
        lines.append(f"- {text}{suffix}")
    if len(rows) > limit:
        lines.append(f"- … and {len(rows) - limit} more")
    return "\n".join(lines)


def _report_summary(report: dict[str, Any]) -> str:
    prompt_set = report.get("prompt_set", {})
    qa = report.get("generated_qa", {})
    review = report.get("generated_review", {})
    projection = report.get("projection_summary") or {}
    output_paths = report.get("output_paths", {})
    qa_errors = qa.get("errors")
    qa_badge = "✅" if qa_errors == 0 else "⚠️"
    lines = [
        "### ✅ Gallery generated",
        "",
        f"- Prompt count: {prompt_set.get('prompt_count')}",
        f"- Sample count: {report.get('sample_count')}",
        f"- {qa_badge} QA errors: {qa_errors} (ok: {qa.get('ok')})",
        f"- Median visible colors: {projection.get('median_visible_colors_before')} -> "
        f"{projection.get('median_visible_colors_after')}",
        f"- Rare-color warning rate: {review.get('rare_color_warning_rate')}",
        f"- Destructive projection rate: {projection.get('destructive_rate')}",
        "",
        "**Files written**",
        "",
        f"- Samples: `{output_paths.get('samples_dir')}`",
        f"- Contact sheets: `{output_paths.get('contact_sheets_dir')}`",
        f"- Report: `{output_paths.get('report_markdown')}`",
    ]
    return "\n".join(lines)


def _contact_sheet_paths(report: dict[str, Any]) -> list[str]:
    output_paths = report.get("output_paths", {})
    contact_sheets_dir = Path(str(output_paths.get("contact_sheets_dir") or ""))
    contact_sheets = output_paths.get("contact_sheets") or {}
    paths: list[str] = []
    for name in contact_sheets.values():
        candidate = contact_sheets_dir / str(name)
        if candidate.is_file():
            paths.append(str(candidate))
    return paths


def _sample_image_items(report: dict[str, Any]) -> list[tuple[str, str]]:
    """Read the generated manifest and return (image_path, caption) pairs for the gallery."""

    output_paths = report.get("output_paths", {})
    samples_dir = Path(str(output_paths.get("samples_dir") or ""))
    manifest = samples_dir / "generated_manifest.jsonl"
    if not manifest.is_file():
        return []
    items: list[tuple[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        paths = record.get("paths") if isinstance(record.get("paths"), dict) else {}
        rel = paths.get("indexed_png") or paths.get("projected_png") or paths.get("hard_rgba")
        if not rel:
            continue
        image_path = samples_dir / str(rel).replace("\\", "/")
        if not image_path.is_file():
            continue
        caption = str(record.get("prompt") or record.get("prompt_id") or "")
        items.append((str(image_path), caption[:80]))
    return items


def _report_markdown_text(report: dict[str, Any]) -> str:
    output_paths = report.get("output_paths", {})
    report_path = Path(str(output_paths.get("report_markdown") or ""))
    if report_path.is_file():
        return report_path.read_text(encoding="utf-8")
    return _report_summary(report)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the local v1 gallery GUI.")
    parser.add_argument("--out", default="experiments/v1_gallery_gui", dest="out_dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parsed = parser.parse_args(argv)
    launch_v1_gallery_gui(out_dir=parsed.out_dir, host=parsed.host, port=parsed.port)


if __name__ == "__main__":
    main()
