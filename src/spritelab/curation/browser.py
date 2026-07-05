"""Minimal curation browser helpers and optional Gradio launcher."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from spritelab.codec.io import load_bundle
from spritelab.codec.preview import make_preview
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.codec.role_inference import role_map_to_preview_image
from spritelab.curation.manifest import (
    ALLOWED_REASONS,
    ALLOWED_STATUSES,
    CurationDecision,
    append_curation_decision,
    discover_bundle_ids,
    load_latest_curation,
)


@dataclass(frozen=True)
class BrowserSprite:
    """One sprite row prepared for the curation browser."""

    sprite_id: str
    path: Path
    status: str | None
    tags: tuple[str, ...]
    reasons: tuple[str, ...]
    notes: str
    metadata: dict[str, Any]
    quality_issues: tuple[str, ...]
    dedupe_group: str | None


def load_browser_sprites(
    bundle_root: str | Path,
    curation_path: str | Path,
    quality_report_path: str | Path | None = None,
    dedupe_report_path: str | Path | None = None,
) -> list[BrowserSprite]:
    """Discover bundles and attach latest curation plus optional report hints."""

    bundle_ids = discover_bundle_ids(bundle_root)
    latest = load_latest_curation(curation_path)
    quality_issues = _load_quality_issues(quality_report_path)
    dedupe_groups = _load_dedupe_groups(dedupe_report_path)

    sprites: list[BrowserSprite] = []
    for sprite_id, bundle_path in bundle_ids.items():
        decision = latest.get(sprite_id)
        sprites.append(
            BrowserSprite(
                sprite_id=sprite_id,
                path=bundle_path,
                status=decision.status if decision else None,
                tags=decision.tags if decision else (),
                reasons=decision.reasons if decision else (),
                notes=decision.notes if decision else "",
                metadata=_load_metadata(bundle_path),
                quality_issues=quality_issues.get(sprite_id, ()),
                dedupe_group=dedupe_groups.get(sprite_id),
            )
        )
    return sorted(sprites, key=_browser_sprite_sort_key)


def make_browser_preview_image(bundle_path: str | Path, scale: int = 8) -> Image.Image:
    """Return nearest-neighbor reconstructed sprite preview for a bundle."""

    bundle = load_bundle(bundle_path)
    return make_preview(reconstruct_rgba(bundle), scale=scale)


def make_alpha_preview_image(bundle_path: str | Path, scale: int = 8) -> Image.Image:
    """Return a white-on-transparent alpha mask preview."""

    if scale < 1:
        raise ValueError("scale must be at least 1.")
    bundle = load_bundle(bundle_path)
    alpha = np.asarray(bundle.alpha)
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[alpha == 1] = (255, 255, 255, 255)
    image = Image.fromarray(pixels, mode="RGBA")
    return image.resize((32 * scale, 32 * scale), resample=Image.Resampling.NEAREST)


def make_role_preview_image(bundle_path: str | Path, scale: int = 8) -> Image.Image | None:
    """Return a role-map debug preview, or None when the bundle has no role map."""

    bundle = load_bundle(bundle_path)
    if bundle.role_map is None:
        return None
    return role_map_to_preview_image(bundle.role_map, scale=scale)


def make_palette_strip_image(bundle_path: str | Path, swatch_size: int = 24) -> Image.Image:
    """Return a small RGBA strip of all palette slots."""

    if swatch_size < 1:
        raise ValueError("swatch_size must be at least 1.")
    bundle = load_bundle(bundle_path)
    palette = np.asarray(bundle.palette)
    width = int(palette.shape[0]) * swatch_size
    image = Image.new("RGBA", (width, swatch_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for slot, rgb in enumerate(palette):
        x0 = slot * swatch_size
        box = (x0, 0, x0 + swatch_size - 1, swatch_size - 1)
        if slot == 0:
            _draw_checker(draw, box, swatch_size)
            draw.rectangle(box, outline=(120, 120, 120, 255))
            continue
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)
        draw.rectangle(box, fill=color, outline=(0, 0, 0, 255))
    return image


def launch_curation_browser(
    bundle_root: str | Path,
    curation_path: str | Path,
    quality_report_path: str | Path | None = None,
    dedupe_report_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> None:
    """Launch the optional Gradio curation browser."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The curation browser requires gradio. Install with: pip install gradio") from exc

    sprites = load_browser_sprites(
        bundle_root,
        curation_path,
        quality_report_path=quality_report_path,
        dedupe_report_path=dedupe_report_path,
    )
    if not sprites:
        raise ValueError("no SpriteBundle directories found for curation.")

    def filtered(status_filter: str, query: str, uncurated_only: bool) -> list[BrowserSprite]:
        return _filter_sprites(sprites, status_filter=status_filter, query=query, uncurated_only=uncurated_only)

    def render(index: int, status_filter: str, query: str, uncurated_only: bool):
        visible = filtered(status_filter, query, uncurated_only)
        if not visible:
            empty = _empty_image()
            return 0, empty, empty, None, empty, "No sprites match filters.", None, "", [], ""
        index = max(0, min(index, len(visible) - 1))
        sprite = visible[index]
        return (
            index,
            make_browser_preview_image(sprite.path),
            make_alpha_preview_image(sprite.path),
            make_role_preview_image(sprite.path),
            make_palette_strip_image(sprite.path),
            _sprite_markdown(sprite, index=index, total=len(visible)),
            sprite.status,
            ", ".join(sprite.tags),
            list(sprite.reasons),
            sprite.notes,
        )

    def previous(index: int, status_filter: str, query: str, uncurated_only: bool):
        return render(index - 1, status_filter, query, uncurated_only)

    def next_sprite(index: int, status_filter: str, query: str, uncurated_only: bool):
        return render(index + 1, status_filter, query, uncurated_only)

    def save_decision(
        index: int,
        status_filter: str,
        query: str,
        uncurated_only: bool,
        status: str,
        tags_text: str,
        reasons: list[str],
        notes: str,
        reviewer: str,
    ):
        visible = filtered(status_filter, query, uncurated_only)
        if not visible:
            return render(0, status_filter, query, uncurated_only)
        index = max(0, min(index, len(visible) - 1))
        sprite = visible[index]
        decision = CurationDecision(
            sprite_id=sprite.sprite_id,
            status=status,
            tags=tuple(_split_csv(tags_text)),
            reasons=tuple(reasons or ()),
            notes=notes,
            reviewer=reviewer or None,
            source_path=str(sprite.path),
        )
        append_curation_decision(curation_path, decision)
        updated = replace(
            sprite,
            status=decision.status,
            tags=decision.tags,
            reasons=decision.reasons,
            notes=decision.notes,
        )
        original_index = sprites.index(sprite)
        sprites[original_index] = updated
        return render(index + 1, status_filter, query, uncurated_only)

    with gr.Blocks(title="sprite-lab curation") as app:
        index_state = gr.State(0)
        with gr.Row():
            status_filter = gr.Dropdown(["all", "uncurated", *ALLOWED_STATUSES], value="all", label="Filter status")
            query = gr.Textbox(label="Search ID/tags/path")
            uncurated_only = gr.Checkbox(label="Uncurated only", value=False)
        info = gr.Markdown()
        with gr.Row():
            preview = gr.Image(label="Sprite", type="pil")
            alpha = gr.Image(label="Alpha", type="pil")
            role = gr.Image(label="Role map", type="pil")
            palette = gr.Image(label="Palette", type="pil")
        with gr.Row():
            prev_button = gr.Button("Previous")
            next_button = gr.Button("Next")
        status = gr.Dropdown(list(ALLOWED_STATUSES), label="Status")
        tags = gr.Textbox(label="Tags, comma-separated")
        reasons = gr.CheckboxGroup(list(ALLOWED_REASONS), label="Reasons")
        notes = gr.Textbox(label="Notes", lines=3)
        reviewer = gr.Textbox(label="Reviewer")
        save_button = gr.Button("Save decision")

        outputs = [index_state, preview, alpha, role, palette, info, status, tags, reasons, notes]
        inputs = [index_state, status_filter, query, uncurated_only]
        app.load(render, inputs=inputs, outputs=outputs)
        prev_button.click(previous, inputs=inputs, outputs=outputs)
        next_button.click(next_sprite, inputs=inputs, outputs=outputs)
        for control in (status_filter, query, uncurated_only):
            control.change(render, inputs=inputs, outputs=outputs)
        save_button.click(
            save_decision,
            inputs=[index_state, status_filter, query, uncurated_only, status, tags, reasons, notes, reviewer],
            outputs=outputs,
        )

    app.launch(server_name=host, server_port=port)


def _load_metadata(bundle_path: Path) -> dict[str, Any]:
    try:
        return json.loads((bundle_path / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_quality_issues(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    report_path = _resolve_report_path(path, "quality_report.json")
    if report_path is None:
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    issues: dict[str, tuple[str, ...]] = {}
    for record in data.get("records", []):
        sprite_id = record.get("id")
        if sprite_id:
            issues[str(sprite_id)] = tuple(str(issue) for issue in record.get("issue_codes", []))
    return issues


def _load_dedupe_groups(path: str | Path | None) -> dict[str, str]:
    report_path = _resolve_report_path(path, "dedupe_report.json")
    if report_path is None:
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    groups: dict[str, list[str]] = {}
    for index, group in enumerate(data.get("exact_groups", []), start=1):
        label = f"exact:{group.get('kind', 'duplicate')}:{index}"
        for sprite_id in group.get("ids", []):
            groups.setdefault(str(sprite_id), []).append(label)
    for index, group in enumerate(data.get("near_groups", []), start=1):
        label = f"near:{index}"
        for sprite_id in group.get("ids", []):
            groups.setdefault(str(sprite_id), []).append(label)
    return {sprite_id: "; ".join(labels) for sprite_id, labels in groups.items()}


def _resolve_report_path(path: str | Path | None, filename: str) -> Path | None:
    if path is None:
        return None
    report_path = Path(path)
    if report_path.is_dir():
        report_path = report_path / filename
    return report_path if report_path.exists() else None


def _browser_sprite_sort_key(sprite: BrowserSprite) -> tuple[str, str, str]:
    return (sprite.status or "", sprite.sprite_id.lower(), str(sprite.path).lower())


def _draw_checker(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], swatch_size: int) -> None:
    x0, y0, x1, y1 = box
    cell = max(2, swatch_size // 4)
    for y in range(y0, y1 + 1, cell):
        for x in range(x0, x1 + 1, cell):
            parity = ((x - x0) // cell + (y - y0) // cell) % 2
            color = (220, 220, 220, 255) if parity == 0 else (120, 120, 120, 255)
            draw.rectangle((x, y, min(x + cell - 1, x1), min(y + cell - 1, y1)), fill=color)


def _empty_image(size: tuple[int, int] = (256, 256)) -> Image.Image:
    return Image.new("RGBA", size, (32, 32, 32, 255))


def _filter_sprites(
    sprites: list[BrowserSprite],
    *,
    status_filter: str,
    query: str,
    uncurated_only: bool,
) -> list[BrowserSprite]:
    filtered = sprites
    if status_filter == "uncurated" or uncurated_only:
        filtered = [sprite for sprite in filtered if sprite.status is None]
    elif status_filter != "all":
        filtered = [sprite for sprite in filtered if sprite.status == status_filter]
    if query:
        needle = query.lower()
        filtered = [
            sprite
            for sprite in filtered
            if needle in sprite.sprite_id.lower()
            or needle in str(sprite.path).lower()
            or any(needle in tag for tag in sprite.tags)
        ]
    return filtered


def _sprite_markdown(sprite: BrowserSprite, *, index: int, total: int) -> str:
    metadata_json = json.dumps(sprite.metadata, indent=2, sort_keys=True)
    lines = [
        f"### {sprite.sprite_id}",
        f"{index + 1} / {total}",
        f"Path: `{sprite.path}`",
        f"Status: `{sprite.status or 'uncurated'}`",
    ]
    if sprite.quality_issues:
        lines.append(f"Quality issues: `{', '.join(sprite.quality_issues)}`")
    if sprite.dedupe_group:
        lines.append(f"Dedupe group: `{sprite.dedupe_group}`")
    lines.extend(["", "```json", metadata_json, "```"])
    return "\n".join(lines)


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the optional SpriteBundle curation browser.")
    parser.add_argument("--bundles", required=True, type=Path)
    parser.add_argument("--curation", required=True, type=Path)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--dedupe-report", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    launch_curation_browser(
        args.bundles,
        args.curation,
        quality_report_path=args.quality_report,
        dedupe_report_path=args.dedupe_report,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
