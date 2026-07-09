"""Tiny GUI for reviewing Qwen prefill suggestions against filename rules."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.filename_rules import (
    FilenameMetadataSuggestion,
    filename_suggestion_to_dict,
    metadata_suggestions_differ,
    parse_filename_metadata,
)


@dataclass(frozen=True)
class PrefillReviewItem:
    sprite_id: str
    filename: str
    image_path: Path
    qwen_suggestion: dict[str, Any]
    fused_suggestion: dict[str, Any]
    prefill_quality: dict[str, Any]
    filename_suggestion: FilenameMetadataSuggestion
    mismatch_reasons: tuple[str, ...]


def load_prefill_review_items(run_dir: str | Path) -> list[PrefillReviewItem]:
    """Load imported sprites and stored Qwen suggestions from a harvest run."""

    run_dir = Path(run_dir)
    imported = read_jsonl(run_dir / "imported.jsonl")
    qwen_by_id = _qwen_suggestions_by_id(read_jsonl(run_dir / "qwen_suggestions.jsonl"))
    fused_by_id = _fused_suggestions_by_id(read_jsonl(run_dir / "fused_suggestions.jsonl"))
    items: list[PrefillReviewItem] = []
    for record in imported:
        sprite_id = str(record.get("sprite_id", ""))
        image_path = _resolve_image_path(run_dir, record)
        relative_path = str(record.get("relative_path") or image_path.name)
        filename = Path(relative_path).name
        qwen_suggestion = _qwen_suggestion_for_record(record, qwen_by_id)
        fused_suggestion, prefill_quality = _fused_for_record(record, fused_by_id)
        filename_suggestion = parse_filename_metadata(sprite_id, filename=filename)
        items.append(
            PrefillReviewItem(
                sprite_id=sprite_id,
                filename=filename,
                image_path=image_path,
                qwen_suggestion=qwen_suggestion,
                fused_suggestion=fused_suggestion,
                prefill_quality=prefill_quality,
                filename_suggestion=filename_suggestion,
                mismatch_reasons=metadata_suggestions_differ(filename_suggestion, qwen_suggestion),
            )
        )
    return items


def random_mismatch_index(
    items: list[PrefillReviewItem],
    *,
    rng: random.Random | None = None,
) -> int:
    """Return a random index whose filename suggestion differs from Qwen."""

    mismatches = [index for index, item in enumerate(items) if item.mismatch_reasons]
    if not mismatches:
        return 0
    chooser = rng or random
    return int(chooser.choice(mismatches))


def launch_prefill_review_gui(
    run_dir: str | Path = "harvest_runs",
    host: str = "127.0.0.1",
    port: int | None = None,
) -> None:
    """Launch a local Gradio GUI for visual Qwen/filename prefill review."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The prefill review GUI requires gradio. Install with: pip install gradio") from exc

    def load_run(run_dir_value: str) -> tuple[Any, ...]:
        try:
            items = load_prefill_review_items(run_dir_value)
        except Exception as exc:
            return ([], 0, f"Load failed: {exc}", None, {}, {}, {}, {}, "")
        index = random_mismatch_index(items) if items else 0
        return (items, index, _summary(items), *_view(items, index))

    def previous(items: list[PrefillReviewItem], index: int) -> tuple[Any, ...]:
        if not items:
            return (0, None, {}, {}, {}, {}, "No items loaded.")
        next_index = max(0, int(index or 0) - 1)
        return (next_index, *_view(items, next_index))

    def next_item(items: list[PrefillReviewItem], index: int) -> tuple[Any, ...]:
        if not items:
            return (0, None, {}, {}, {}, {}, "No items loaded.")
        next_index = min(len(items) - 1, int(index or 0) + 1)
        return (next_index, *_view(items, next_index))

    def random_mismatch(items: list[PrefillReviewItem]) -> tuple[Any, ...]:
        if not items:
            return (0, None, {}, {}, {}, {}, "No items loaded.")
        index = random_mismatch_index(items)
        return (index, *_view(items, index))

    def random_low_confidence(items: list[PrefillReviewItem]) -> tuple[Any, ...]:
        return _random_by_quality(items, {"low_confidence"})

    def random_weak_filename(items: list[PrefillReviewItem]) -> tuple[Any, ...]:
        if not items:
            return (0, None, {}, {}, {}, {}, "No items loaded.")
        indices = [index for index, item in enumerate(items) if item.filename_suggestion.confidence < 0.8]
        index = int(random.choice(indices)) if indices else 0
        return (index, *_view(items, index))

    def random_auto_fused(items: list[PrefillReviewItem]) -> tuple[Any, ...]:
        return _random_by_quality(items, {"fused_automatically"})

    with gr.Blocks(title="SpriteLab Qwen Prefill Review") as demo:
        gr.Markdown("# Qwen Prefill Review")
        items_state = gr.State([])
        index_state = gr.State(0)
        with gr.Row():
            run_dir_box = gr.Textbox(label="Harvest run directory", value=str(run_dir))
            load_button = gr.Button("Load run", variant="primary")
        summary = gr.Markdown()
        with gr.Row():
            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next")
            random_button = gr.Button("Random filename/Qwen mismatch")
            low_confidence_button = gr.Button("Random low Qwen confidence")
            weak_filename_button = gr.Button("Random weak filename parse")
            auto_fused_button = gr.Button("Random auto-fused")
        image = gr.Image(label="Sprite", type="pil", height=320)
        with gr.Row():
            filename_json = gr.JSON(label="Filename rules")
            qwen_json = gr.JSON(label="Qwen suggestion")
            fused_json = gr.JSON(label="Fused suggestion")
            quality_json = gr.JSON(label="Prefill quality")
        details = gr.Markdown()

        view_outputs = [index_state, image, filename_json, qwen_json, fused_json, quality_json, details]
        load_button.click(
            load_run,
            inputs=run_dir_box,
            outputs=[
                items_state,
                index_state,
                summary,
                image,
                filename_json,
                qwen_json,
                fused_json,
                quality_json,
                details,
            ],
        )
        previous_button.click(previous, inputs=[items_state, index_state], outputs=view_outputs)
        next_button.click(next_item, inputs=[items_state, index_state], outputs=view_outputs)
        random_button.click(random_mismatch, inputs=items_state, outputs=view_outputs)
        low_confidence_button.click(random_low_confidence, inputs=items_state, outputs=view_outputs)
        weak_filename_button.click(random_weak_filename, inputs=items_state, outputs=view_outputs)
        auto_fused_button.click(random_auto_fused, inputs=items_state, outputs=view_outputs)

    demo.launch(server_name=host, server_port=port)


def load_golden_label_items(run_dir: str | Path) -> list[PrefillReviewItem]:
    """Load review items restricted to the sampled golden sprite_ids."""

    run_dir = Path(run_dir)
    sample_ids = {
        str(record.get("sprite_id", ""))
        for record in read_jsonl(run_dir / "golden_sample.jsonl")
        if record.get("sprite_id")
    }
    if not sample_ids:
        raise RuntimeError(f"no golden sample found in {run_dir}. Run `harvest golden-sample --run {run_dir}` first.")
    return [item for item in load_prefill_review_items(run_dir) if item.sprite_id in sample_ids]


def launch_golden_label_gui(
    run_dir: str | Path,
    host: str = "127.0.0.1",
    port: int | None = None,
    labeler: str = "",
) -> None:
    """Launch a local Gradio GUI for labeling the golden evaluation sample."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The golden labeling GUI requires gradio. Install with: pip install gradio") from exc

    from spritelab.dataset_maker.prefill import ALLOWED_CATEGORIES
    from spritelab.harvest.golden import GoldenLabel, append_golden_label, load_golden_labels

    run_path = Path(run_dir)
    labels_path = run_path / "golden_labels.jsonl"
    categories = sorted(ALLOWED_CATEGORIES)

    def load_run() -> tuple[Any, ...]:
        try:
            items = load_golden_label_items(run_path)
        except Exception as exc:
            return ([], 0, f"Load failed: {exc}", None, "unknown", "", "", "")
        index = _first_unlabeled_index(items, load_golden_labels(labels_path))
        return (items, index, _golden_summary(items, labels_path), *_golden_view(items, index, labels_path))

    def go(items: list[PrefillReviewItem], index: int, step: int) -> tuple[Any, ...]:
        if not items:
            return (0, None, "unknown", "", "", "No items loaded.")
        next_index = min(max(0, int(index or 0) + step), len(items) - 1)
        return (next_index, *_golden_view(items, next_index, labels_path))

    def save_label(
        items: list[PrefillReviewItem],
        index: int,
        category: str,
        object_name: str,
        tags_text: str,
        notes: str,
    ) -> tuple[Any, ...]:
        if not items:
            return ("No items loaded.", 0, None, "unknown", "", "", "No items loaded.")
        safe_index = min(max(0, int(index or 0)), len(items) - 1)
        item = items[safe_index]
        label = GoldenLabel(
            sprite_id=item.sprite_id,
            category=category or "unknown",
            object_name=object_name,
            tags=tuple(part for part in (tags_text or "").replace(",", " ").split() if part),
            notes=notes,
            labeler=labeler,
        )
        append_golden_label(labels_path, label)
        next_index = _first_unlabeled_index(items, load_golden_labels(labels_path), start=safe_index + 1)
        return (
            _golden_summary(items, labels_path),
            next_index,
            *_golden_view(items, next_index, labels_path),
        )

    with gr.Blocks(title="SpriteLab Golden Labeling") as demo:
        gr.Markdown("# Golden Set Labeling\nLabels are stored in `golden_labels.jsonl` and used only for evaluation.")
        items_state = gr.State([])
        index_state = gr.State(0)
        summary = gr.Markdown()
        with gr.Row():
            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next")
        image = gr.Image(label="Sprite", type="pil", height=320)
        with gr.Row():
            category_box = gr.Dropdown(label="Category", choices=categories, value="unknown")
            object_box = gr.Textbox(label="Object name (snake_case)")
        tags_box = gr.Textbox(label="Tags (space or comma separated)")
        notes_box = gr.Textbox(label="Notes")
        save_button = gr.Button("Save label and next", variant="primary")
        details = gr.Markdown()

        view_outputs = [index_state, image, category_box, object_box, tags_box, details]
        demo.load(
            load_run, outputs=[items_state, index_state, summary, image, category_box, object_box, tags_box, details]
        )
        previous_button.click(
            lambda items, index: go(items, index, -1), inputs=[items_state, index_state], outputs=view_outputs
        )
        next_button.click(
            lambda items, index: go(items, index, 1), inputs=[items_state, index_state], outputs=view_outputs
        )
        save_button.click(
            save_label,
            inputs=[items_state, index_state, category_box, object_box, tags_box, notes_box],
            outputs=[summary, *view_outputs],
        )

    demo.launch(server_name=host, server_port=port)


def _first_unlabeled_index(items: list[PrefillReviewItem], labels: dict[str, Any], *, start: int = 0) -> int:
    for index in range(start, len(items)):
        if items[index].sprite_id not in labels:
            return index
    return min(max(0, start), len(items) - 1) if items else 0


def _golden_view(
    items: list[PrefillReviewItem],
    index: int,
    labels_path: Path,
) -> tuple[Image.Image | None, str, str, str, str]:
    from spritelab.harvest.golden import load_golden_labels

    if not items:
        return None, "unknown", "", "", "No items loaded."
    safe_index = min(max(0, int(index or 0)), len(items) - 1)
    item = items[safe_index]
    image = _preview_image(item.image_path)
    labels = load_golden_labels(labels_path)
    existing = labels.get(item.sprite_id)
    fused = item.fused_suggestion or item.qwen_suggestion
    category = existing.category if existing else str(fused.get("category", "unknown") or "unknown")
    object_name = existing.object_name if existing else str(fused.get("object_name", ""))
    tags = " ".join(existing.tags) if existing else " ".join(str(tag) for tag in fused.get("tags") or ())
    details = [
        f"## {safe_index + 1} / {len(items)}",
        f"Sprite ID: `{item.sprite_id}`",
        f"Filename: `{item.filename}`",
        f"Already labeled: {'yes' if existing else 'no'}",
        "",
        "Fields are prefilled from the fused suggestion — correct them, do not rubber-stamp.",
    ]
    return image, category, object_name, tags, "\n".join(details)


def _golden_summary(items: list[PrefillReviewItem], labels_path: Path) -> str:
    from spritelab.harvest.golden import load_golden_labels

    labels = load_golden_labels(labels_path)
    labeled = sum(1 for item in items if item.sprite_id in labels)
    return f"Labeled: {labeled} / {len(items)}"


def _view(
    items: list[PrefillReviewItem],
    index: int,
) -> tuple[Image.Image | None, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    if not items:
        return None, {}, {}, {}, {}, "No items loaded."
    safe_index = min(max(0, int(index or 0)), len(items) - 1)
    item = items[safe_index]
    image = _preview_image(item.image_path)
    filename_json = filename_suggestion_to_dict(item.filename_suggestion)
    qwen_json = dict(item.qwen_suggestion)
    fused_json = dict(item.fused_suggestion)
    quality_json = dict(item.prefill_quality)
    details = [
        f"## {safe_index + 1} / {len(items)}",
        f"Sprite ID: `{item.sprite_id}`",
        f"Filename: `{item.filename}`",
        f"Image: `{item.image_path}`",
        f"Quality bucket: `{quality_json.get('bucket', 'none')}`",
        f"Agreement: `{quality_json.get('agreement', 'unknown')}`",
        "",
    ]
    if image is None:
        details.extend(["", "### Image", "- Could not open the sprite image at the resolved path."])
    details.extend(["", "### Mismatch"])
    if item.mismatch_reasons:
        details.extend(f"- {reason}" for reason in item.mismatch_reasons)
    else:
        details.append("- Filename rules and Qwen agree on the checked fields.")
    if quality_json.get("conflict_reasons"):
        details.extend(["", "### Fusion conflict"])
        details.extend(f"- {reason}" for reason in quality_json.get("conflict_reasons", ()))
    return image, filename_json, qwen_json, fused_json, quality_json, "\n".join(details)


def _summary(items: list[PrefillReviewItem]) -> str:
    mismatch_count = sum(1 for item in items if item.mismatch_reasons)
    missing_qwen = sum(1 for item in items if not item.qwen_suggestion)
    quality_counts: dict[str, int] = {}
    for item in items:
        bucket = str(item.prefill_quality.get("bucket", "") or "missing")
        quality_counts[bucket] = quality_counts.get(bucket, 0) + 1
    quality_lines = "\n".join(f"- {bucket}: {count}" for bucket, count in sorted(quality_counts.items()))
    return (
        f"Loaded: {len(items)} sprite(s)\n\n"
        f"Filename/Qwen mismatches: {mismatch_count}\n\n"
        f"Missing Qwen suggestions: {missing_qwen}\n\n"
        f"Quality buckets:\n{quality_lines if quality_lines else '- none'}"
    )


def _random_by_quality(items: list[PrefillReviewItem], buckets: set[str]) -> tuple[Any, ...]:
    if not items:
        return (0, None, {}, {}, {}, {}, "No items loaded.")
    indices = [
        index
        for index, item in enumerate(items)
        if item.prefill_quality.get("bucket") in buckets or bool(set(item.prefill_quality.get("flags") or ()) & buckets)
    ]
    index = int(random.choice(indices)) if indices else 0
    return (index, *_view(items, index))


def _preview_image(path: Path, scale: int = 8) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
    except Exception:
        return None
    preview = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
    checker = _checkerboard(preview.size)
    return Image.alpha_composite(checker, preview).convert("RGB")


def _resolve_image_path(run_dir: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record.get("final_png_path", "")).replace("\\", "/"))
    if path.is_absolute():
        return path
    candidates = (
        path,
        Path.cwd() / path,
        run_dir / path,
        run_dir.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _checkerboard(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size, (238, 238, 238, 255))
    pixels = image.load()
    tile = max(8, width // 16)
    for y in range(height):
        for x in range(width):
            value = 238 if ((x // tile) + (y // tile)) % 2 == 0 else 188
            pixels[x, y] = (value, value, value, 255)
    return image


def _qwen_suggestion_for_record(
    record: dict[str, Any],
    qwen_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    auto_metadata = record.get("auto_metadata")
    if isinstance(auto_metadata, dict):
        suggestion = auto_metadata.get("qwen_suggestion")
        if isinstance(suggestion, dict):
            return dict(suggestion)
    return dict(qwen_by_id.get(str(record.get("sprite_id", "")), {}))


def _fused_for_record(
    record: dict[str, Any],
    fused_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    auto_metadata = record.get("auto_metadata")
    if isinstance(auto_metadata, dict):
        fused = auto_metadata.get("fused_suggestion")
        quality = auto_metadata.get("prefill_quality")
        if isinstance(fused, dict) or isinstance(quality, dict):
            return (dict(fused or {}), dict(quality or {}))
    value = fused_by_id.get(str(record.get("sprite_id", "")), {})
    return (dict(value.get("fused_suggestion") or {}), dict(value.get("prefill_quality") or {}))


def _qwen_suggestions_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        sprite_id = str(record.get("sprite_id", ""))
        if not sprite_id:
            continue
        suggestion = {key: value for key, value in record.items() if key != "sprite_id"}
        result[sprite_id] = json.loads(json.dumps(suggestion, default=str))
    return result


def _fused_suggestions_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("sprite_id", "")): dict(record) for record in records if record.get("sprite_id")}
