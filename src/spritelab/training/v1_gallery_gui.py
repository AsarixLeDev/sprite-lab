"""Local Gradio GUI for the official v1 demo/release gallery.

This is a thin, read-only-safe wrapper around
``spritelab.training.v1_gallery.build_v1_gallery_demo``: pick an output
directory (and optionally a custom prompt file), sample with the official v1
export preset, and preview the resulting contact sheets. It never trains a
model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spritelab.training.v1_gallery import (
    DEFAULT_V1_CHECKPOINT,
    OFFICIAL_V1_STATEMENT,
    BuildV1GalleryConfig,
    build_v1_gallery_demo,
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

    def build(
        out_dir_value: str,
        checkpoint_value: str,
        prompts_value: str,
        device_value: str,
        seed_value: float,
        batch_size_value: float,
        num_samples_value: float | None,
        categories_value: str,
        include_ood_value: bool,
        include_grounded_value: bool,
        include_stress_value: bool,
    ) -> tuple[str, list[str]]:
        categories = None
        if categories_value.strip():
            categories = tuple(token.strip() for token in categories_value.split(",") if token.strip())
        try:
            report = build_v1_gallery_demo(
                BuildV1GalleryConfig(
                    out_dir=Path(out_dir_value.strip() or "experiments/v1_gallery_gui"),
                    checkpoint=Path(checkpoint_value.strip() or str(DEFAULT_V1_CHECKPOINT)),
                    prompts=Path(prompts_value.strip()) if prompts_value.strip() else None,
                    device=device_value,
                    seed=int(seed_value or 20260723),
                    batch_size=int(batch_size_value or 32),
                    num_samples=int(num_samples_value) if num_samples_value else None,
                    categories=categories,
                    include_ood=bool(include_ood_value),
                    include_grounded=bool(include_grounded_value),
                    include_stress_prompts=bool(include_stress_value),
                )
            )
        except Exception as exc:
            return (f"### Generation failed\n\n```\n{exc}\n```", [])
        return (_report_summary(report), _contact_sheet_paths(report))

    with gr.Blocks(title="Sprite Lab v1 Gallery") as demo:
        gr.Markdown("# v1 Gallery")
        gr.Markdown(OFFICIAL_V1_STATEMENT.replace("\n", "  \n"))
        with gr.Row():
            out_dir_box = gr.Textbox(label="Output directory", value=str(out_dir))
            checkpoint_box = gr.Textbox(label="Checkpoint", value=str(DEFAULT_V1_CHECKPOINT))
            device_box = gr.Dropdown(label="Device", choices=["cpu", "cuda"], value="cpu")
        with gr.Row():
            prompts_box = gr.Textbox(
                label="Custom prompts JSONL (optional)",
                placeholder="Leave blank to use the built-in v1 gallery prompt set",
            )
            categories_box = gr.Textbox(
                label="Category filter (optional, comma-separated)",
                placeholder="weapon,armor,item_icon,tool,material,effect_icon,plant",
            )
        with gr.Row():
            seed_box = gr.Number(label="Seed", value=20260723, precision=0)
            batch_size_box = gr.Number(label="Batch size", value=32, precision=0)
            num_samples_box = gr.Number(label="Max samples (optional)", value=None, precision=0)
        with gr.Row():
            include_ood_box = gr.Checkbox(label="Include OOD compositional prompts", value=True)
            include_grounded_box = gr.Checkbox(label="Include grounded/compositional prompts", value=True)
            include_stress_box = gr.Checkbox(label="Include style-stress prompts", value=True)
        build_button = gr.Button("Build v1 gallery", variant="primary")
        summary_markdown = gr.Markdown()
        contact_sheet_gallery = gr.Gallery(label="Contact sheets", columns=2, height="auto")

        build_button.click(
            build,
            inputs=[
                out_dir_box,
                checkpoint_box,
                prompts_box,
                device_box,
                seed_box,
                batch_size_box,
                num_samples_box,
                categories_box,
                include_ood_box,
                include_grounded_box,
                include_stress_box,
            ],
            outputs=[summary_markdown, contact_sheet_gallery],
        )

    demo.launch(server_name=host, server_port=port)


def _report_summary(report: dict[str, Any]) -> str:
    prompt_set = report.get("prompt_set", {})
    qa = report.get("generated_qa", {})
    review = report.get("generated_review", {})
    projection = report.get("projection_summary") or {}
    output_paths = report.get("output_paths", {})
    lines = [
        "### v1 gallery report",
        "",
        f"- Prompt count: {prompt_set.get('prompt_count')}",
        f"- Sample count: {report.get('sample_count')}",
        f"- QA errors: {qa.get('errors')} (ok: {qa.get('ok')})",
        f"- Median visible colors: {projection.get('median_visible_colors_before')} -> "
        f"{projection.get('median_visible_colors_after')}",
        f"- Rare-color warning rate: {review.get('rare_color_warning_rate')}",
        f"- Destructive projection rate: {projection.get('destructive_rate')}",
        "",
        f"Samples: `{output_paths.get('samples_dir')}`  ",
        f"Contact sheets: `{output_paths.get('contact_sheets_dir')}`  ",
        f"Report: `{output_paths.get('report_markdown')}`",
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
