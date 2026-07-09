"""Fast correction GUI for assisted golden-set labeling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.assisted_golden import (
    GOLDEN_ASSISTED_STATE_FILENAME,
    GOLDEN_CANDIDATES_FILENAME,
    GOLDEN_CANDIDATES_PREFILLED_FILENAME,
    GOLDEN_CATEGORY_VALUES,
    GOLDEN_LABELS_FILENAME,
    AssistedGoldenCandidate,
    AssistedGoldenLabel,
    append_golden_label,
    build_assisted_golden_label,
    candidate_review_priority,
    load_assisted_candidates,
    load_existing_golden_labels,
    load_golden_candidates_jsonl,
    normalize_category,
    normalize_object_name,
    write_golden_candidates_jsonl,
)
from spritelab.harvest.sources import utc_timestamp

ORDER_MODES = (
    "needs_review_first",
    "random",
    "source_order",
    "uncertain_first",
    "unlabeled_first",
)


@dataclass(frozen=True)
class AssistedGoldenState:
    candidates: tuple[AssistedGoldenCandidate, ...]
    labels: dict[str, AssistedGoldenLabel]
    index: int = 0
    skipped: tuple[str, ...] = ()
    order: str = "needs_review_first"
    seed: int = 1337


def launch_assisted_golden_gui(
    run_dir: str | Path,
    *,
    n: int | None = None,
    seed: int = 1337,
    host: str = "127.0.0.1",
    port: int | None = None,
    labeler: str = "mathieu",
    include_statuses: tuple[str, ...] = ("accepted",),
    order: str = "needs_review_first",
) -> None:
    """Launch the local Gradio assisted golden-label correction GUI."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The assisted golden GUI requires gradio. Install with: pip install gradio") from exc

    run_path = Path(run_dir)
    candidates_path = run_path / GOLDEN_CANDIDATES_FILENAME
    labels_path = run_path / GOLDEN_LABELS_FILENAME
    state_path = run_path / GOLDEN_ASSISTED_STATE_FILENAME
    candidates = _load_or_create_candidates(
        run_path,
        candidates_path=candidates_path,
        n=n,
        seed=seed,
        include_statuses=include_statuses,
    )
    labels = load_existing_golden_labels(labels_path)
    state = make_initial_state(candidates, labels=labels, order=order, seed=seed)
    write_assisted_state_json(state_path, state)

    def current_view(current_state: AssistedGoldenState) -> tuple[Any, ...]:
        return _view_outputs(current_state)

    def previous(current_state: AssistedGoldenState) -> tuple[Any, ...]:
        next_state = move_previous(current_state)
        write_assisted_state_json(state_path, next_state)
        return (next_state, *_view_outputs(next_state))

    def next_item(current_state: AssistedGoldenState) -> tuple[Any, ...]:
        next_state = move_next(current_state)
        write_assisted_state_json(state_path, next_state)
        return (next_state, *_view_outputs(next_state))

    def accept(current_state: AssistedGoldenState) -> tuple[Any, ...]:
        next_state, _ = accept_as_is(current_state, labels_path=labels_path, labeler=labeler)
        write_assisted_state_json(state_path, next_state)
        return (next_state, *_view_outputs(next_state))

    def save(
        current_state: AssistedGoldenState,
        category: str,
        object_name: str,
        tags: str,
        short_description: str,
        materials: str,
        mood: str,
        notes: str,
    ) -> tuple[Any, ...]:
        next_state, _ = save_current_label(
            current_state,
            category=category,
            object_name=object_name,
            tags=tags,
            short_description=short_description,
            materials=materials,
            mood=mood,
            notes=notes,
            labels_path=labels_path,
            labeler=labeler,
        )
        write_assisted_state_json(state_path, next_state)
        return (next_state, *_view_outputs(next_state))

    def skip(current_state: AssistedGoldenState) -> tuple[Any, ...]:
        next_state = skip_current(current_state)
        write_assisted_state_json(state_path, next_state)
        return (next_state, *_view_outputs(next_state))

    def mark_unknown(current_state: AssistedGoldenState) -> tuple[str, str, str]:
        return mark_unknown_fields()

    def add_note_text(notes: str, note: str) -> str:
        return append_note(notes, note)

    def apply_filters(
        current_state: AssistedGoldenState,
        unlabeled_only: bool,
        corrected_only: bool,
        conflicts_only: bool,
        category: str,
        source: str,
        text: str,
    ) -> dict[str, Any]:
        rows = filtered_table_rows(
            current_state,
            unlabeled_only=unlabeled_only,
            corrected_only=corrected_only,
            conflicts_only=conflicts_only,
            category=category,
            source=source,
            text_search=text,
        )
        return {"headers": ["#", "status", "category", "sprite_id", "source"], "data": rows}

    with gr.Blocks(title="SpriteLab Assisted Golden Correction") as demo:
        gr.Markdown("# Assisted Golden Correction")
        state_box = gr.State(state)
        with gr.Row():
            progress = gr.Markdown()
            source_info = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(label="Sprite preview", type="pil", height=384)
                sprite_info = gr.Markdown()
            with gr.Column(scale=3):
                category_box = gr.Dropdown(label="Category", choices=list(GOLDEN_CATEGORY_VALUES), value="unknown")
                object_box = gr.Textbox(label="Object name")
                tags_box = gr.Textbox(label="Tags, comma-separated")
                description_box = gr.Textbox(label="Short description")
                materials_box = gr.Textbox(label="Materials, comma-separated")
                mood_box = gr.Textbox(label="Mood, comma-separated")
                notes_box = gr.Textbox(label="Notes")
                with gr.Row():
                    accept_button = gr.Button("Accept as-is", variant="primary")
                    save_button = gr.Button("Save + next", variant="primary")
                    skip_button = gr.Button("Skip")
                with gr.Row():
                    unknown_button = gr.Button("Mark unknown")
                    ambiguous_button = gr.Button("Add note: ambiguous")
                    bad_crop_button = gr.Button("Add note: bad_crop")
                    tile_button = gr.Button("Add note: tile_not_item")
                with gr.Row():
                    previous_button = gr.Button("Previous")
                    next_button = gr.Button("Next")
            with gr.Column(scale=3):
                filename_json = gr.JSON(label="Filename rules")
                qwen_json = gr.JSON(label="VLM descriptor")
                fused_json = gr.JSON(label="Safe prefill")
                quality = gr.Markdown()
        with gr.Row():
            unlabeled_only = gr.Checkbox(label="Unlabeled only", value=False)
            corrected_only = gr.Checkbox(label="Corrected only", value=False)
            conflicts_only = gr.Checkbox(label="Conflicts only", value=False)
            category_filter = gr.Dropdown(label="Category filter", choices=["", *GOLDEN_CATEGORY_VALUES], value="")
            source_filter = gr.Textbox(label="Source filter")
            search_filter = gr.Textbox(label="Text search")
        table = gr.JSON(label="Candidate status")

        view_outputs = [
            progress,
            source_info,
            image,
            sprite_info,
            category_box,
            object_box,
            tags_box,
            description_box,
            materials_box,
            mood_box,
            notes_box,
            filename_json,
            qwen_json,
            fused_json,
            quality,
            table,
        ]
        demo.load(current_view, inputs=state_box, outputs=view_outputs)
        previous_button.click(previous, inputs=state_box, outputs=[state_box, *view_outputs])
        next_button.click(next_item, inputs=state_box, outputs=[state_box, *view_outputs])
        accept_button.click(accept, inputs=state_box, outputs=[state_box, *view_outputs])
        save_button.click(
            save,
            inputs=[state_box, category_box, object_box, tags_box, description_box, materials_box, mood_box, notes_box],
            outputs=[state_box, *view_outputs],
        )
        skip_button.click(skip, inputs=state_box, outputs=[state_box, *view_outputs])
        unknown_button.click(mark_unknown, outputs=[category_box, object_box, tags_box])
        ambiguous_button.click(lambda notes: add_note_text(notes, "ambiguous"), inputs=notes_box, outputs=notes_box)
        bad_crop_button.click(lambda notes: add_note_text(notes, "bad_crop"), inputs=notes_box, outputs=notes_box)
        tile_button.click(lambda notes: add_note_text(notes, "tile_not_item"), inputs=notes_box, outputs=notes_box)
        for control in (unlabeled_only, corrected_only, conflicts_only, category_filter, source_filter, search_filter):
            control.change(
                apply_filters,
                inputs=[
                    state_box,
                    unlabeled_only,
                    corrected_only,
                    conflicts_only,
                    category_filter,
                    source_filter,
                    search_filter,
                ],
                outputs=table,
            )

    demo.launch(server_name=host, server_port=port)


def make_initial_state(
    candidates: list[AssistedGoldenCandidate] | tuple[AssistedGoldenCandidate, ...],
    *,
    labels: dict[str, AssistedGoldenLabel] | None = None,
    order: str = "needs_review_first",
    seed: int = 1337,
) -> AssistedGoldenState:
    """Create a resumable GUI state at the first unlabeled candidate."""

    ordered = order_candidates(tuple(candidates), labels or {}, order=order, seed=seed)
    state = AssistedGoldenState(candidates=ordered, labels=dict(labels or {}), order=order, seed=seed)
    return replace(state, index=first_unlabeled_index(state))


def order_candidates(
    candidates: tuple[AssistedGoldenCandidate, ...],
    labels: dict[str, AssistedGoldenLabel],
    *,
    order: str,
    seed: int,
) -> tuple[AssistedGoldenCandidate, ...]:
    if order not in ORDER_MODES:
        order = "needs_review_first"
    if order == "random":
        values = list(candidates)
        random.Random(seed).shuffle(values)
        return tuple(values)
    if order == "source_order":
        return tuple(
            sorted(
                candidates, key=lambda candidate: (candidate.source_id, candidate.relative_path, candidate.sprite_id)
            )
        )
    if order == "uncertain_first":
        return tuple(sorted(candidates, key=lambda candidate: (-_uncertainty_score(candidate), candidate.sprite_id)))
    if order == "unlabeled_first":
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.sprite_id in labels,
                    -candidate_review_priority(candidate),
                    candidate.sprite_id,
                ),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                -candidate_review_priority(candidate),
                candidate.sprite_id in labels,
                candidate.sprite_id,
            ),
        )
    )


def save_current_label(
    state: AssistedGoldenState,
    *,
    category: str,
    object_name: str,
    tags: str | tuple[str, ...] | list[str],
    short_description: str = "",
    materials: str | tuple[str, ...] | list[str] = (),
    mood: str | tuple[str, ...] | list[str] = (),
    notes: str = "",
    labels_path: str | Path | None = None,
    labeler: str = "mathieu",
) -> tuple[AssistedGoldenState, AssistedGoldenLabel | None]:
    candidate = current_candidate(state)
    if candidate is None:
        return state, None
    label = build_assisted_golden_label(
        candidate,
        category=category,
        object_name=object_name,
        tags=tags,
        short_description=short_description,
        materials=materials,
        mood=mood,
        notes=notes,
        labeler=labeler,
    )
    if labels_path is not None:
        append_golden_label(labels_path, label)
    labels = dict(state.labels)
    labels[label.sprite_id] = label
    next_state = replace(state, labels=labels)
    return replace(next_state, index=_next_unlabeled_or_next(next_state, state.index + 1)), label


def accept_as_is(
    state: AssistedGoldenState,
    *,
    labels_path: str | Path | None = None,
    labeler: str = "mathieu",
) -> tuple[AssistedGoldenState, AssistedGoldenLabel | None]:
    candidate = current_candidate(state)
    if candidate is None:
        return state, None
    return save_current_label(
        state,
        category=candidate.suggested_category,
        object_name=candidate.suggested_object_name,
        tags=candidate.suggested_tags,
        short_description=candidate.gold_short_description or candidate.suggested_description,
        materials=candidate.gold_materials,
        mood=candidate.gold_mood,
        notes="",
        labels_path=labels_path,
        labeler=labeler,
    )


def skip_current(state: AssistedGoldenState) -> AssistedGoldenState:
    candidate = current_candidate(state)
    if candidate is None:
        return state
    skipped = tuple(sorted({*state.skipped, candidate.sprite_id}))
    next_state = replace(state, skipped=skipped)
    return replace(next_state, index=min(len(state.candidates) - 1, state.index + 1))


def move_previous(state: AssistedGoldenState) -> AssistedGoldenState:
    return replace(state, index=max(0, state.index - 1))


def move_next(state: AssistedGoldenState) -> AssistedGoldenState:
    return replace(state, index=min(max(0, len(state.candidates) - 1), state.index + 1))


def current_candidate(state: AssistedGoldenState) -> AssistedGoldenCandidate | None:
    if not state.candidates:
        return None
    safe_index = min(max(0, int(state.index)), len(state.candidates) - 1)
    return state.candidates[safe_index]


def candidate_gui_model(candidate: AssistedGoldenCandidate, label: AssistedGoldenLabel | None = None) -> dict[str, Any]:
    """Return editable and reference data used by the assisted golden GUI."""

    editable = {
        "category": label.category
        if label
        else candidate.gold_category
        if candidate.gold_category != "unknown"
        else candidate.suggested_category,
        "object_name": label.object_name if label else candidate.gold_object_name or candidate.suggested_object_name,
        "tags": list(label.tags if label else candidate.gold_tags or candidate.suggested_tags),
        "short_description": label.short_description
        if label
        else candidate.gold_short_description or candidate.suggested_description,
        "materials": list(label.materials if label else candidate.gold_materials),
        "mood": list(label.mood if label else candidate.gold_mood),
        "notes": label.notes if label else "",
    }
    object_choices = _dedupe(
        [
            str(editable["object_name"]),
            candidate.suggested_object_name,
            candidate.prefill_object_name,
            candidate.vlm_object_name,
            *candidate.candidate_object_names,
            *candidate.alternative_object_names,
        ]
    )
    return {
        "editable": editable,
        "object_name_choices": object_choices,
        "reference": {
            "candidate_object_names": list(candidate.candidate_object_names),
            "alternative_object_names": list(candidate.alternative_object_names),
            "vlm_object_name": candidate.vlm_object_name,
            "vlm_short_description": candidate.vlm_short_description,
            "vlm_source_consistency": candidate.vlm_source_consistency,
            "visual_facts": dict(candidate.visual_facts or {}),
            "prefill": {
                "source": candidate.prefill_source or candidate.suggested_source,
                "category": candidate.prefill_category,
                "object_name": candidate.prefill_object_name,
                "tags": list(candidate.prefill_tags),
                "short_description": candidate.prefill_short_description,
                "materials": list(candidate.prefill_materials),
                "mood": list(candidate.prefill_mood),
                "bucket": candidate.prefill_bucket or candidate.quality_bucket,
                "flags": list(candidate.prefill_flags or candidate.fused_quality_flags),
            },
        },
    }


def first_unlabeled_index(state: AssistedGoldenState) -> int:
    return _next_unlabeled_or_next(state, 0)


def progress_counts(state: AssistedGoldenState) -> dict[str, int]:
    total = len(state.candidates)
    labeled_ids = {candidate.sprite_id for candidate in state.candidates if candidate.sprite_id in state.labels}
    corrected = sum(1 for sprite_id in labeled_ids if state.labels[sprite_id].prefill_was_corrected)
    skipped = len(set(state.skipped))
    return {
        "total": total,
        "labeled": len(labeled_ids),
        "corrected": corrected,
        "accepted_as_is": len(labeled_ids) - corrected,
        "skipped": skipped,
        "remaining": max(0, total - len(labeled_ids) - skipped),
    }


def filter_candidates(
    candidates: tuple[AssistedGoldenCandidate, ...],
    labels: dict[str, AssistedGoldenLabel],
    *,
    unlabeled_only: bool = False,
    corrected_only: bool = False,
    conflicts_only: bool = False,
    category: str = "",
    source: str = "",
    text_search: str = "",
) -> list[AssistedGoldenCandidate]:
    category = normalize_category(category) if category else ""
    source_query = str(source).strip().lower()
    text_query = str(text_search).strip().lower()
    result: list[AssistedGoldenCandidate] = []
    for candidate in candidates:
        label = labels.get(candidate.sprite_id)
        if unlabeled_only and label is not None:
            continue
        if corrected_only and not (label and label.prefill_was_corrected):
            continue
        if conflicts_only and not _has_conflict(candidate):
            continue
        if category and candidate.suggested_category != category:
            continue
        if (
            source_query
            and source_query not in candidate.source_name.lower()
            and source_query not in candidate.source_id.lower()
        ):
            continue
        searchable = " ".join(
            [
                candidate.sprite_id,
                candidate.relative_path,
                candidate.source_name,
                " ".join(candidate.suggested_tags),
                candidate.suggested_object_name,
                " ".join(candidate.candidate_object_names),
                " ".join(candidate.alternative_object_names),
                candidate.vlm_object_name,
            ]
        ).lower()
        if text_query and text_query not in searchable:
            continue
        result.append(candidate)
    return result


def filtered_table_rows(state: AssistedGoldenState, **filters: Any) -> list[list[Any]]:
    rows = []
    for index, candidate in enumerate(filter_candidates(state.candidates, state.labels, **filters)):
        label = state.labels.get(candidate.sprite_id)
        status = "labeled" if label else "skipped" if candidate.sprite_id in state.skipped else "unlabeled"
        if label and label.prefill_was_corrected:
            status = "corrected"
        rows.append([index + 1, status, candidate.suggested_category, candidate.sprite_id, candidate.source_name])
    return rows


def mark_unknown_fields() -> tuple[str, str, str]:
    return ("unknown", "", "")


def append_note(notes: str, note: str) -> str:
    values = [value.strip() for value in str(notes or "").split(",") if value.strip()]
    normalized = normalize_object_name(note)
    if normalized and normalized not in {normalize_object_name(value) for value in values}:
        values.append(normalized)
    return ", ".join(values)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_object_name(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def write_assisted_state_json(path: str | Path, state: AssistedGoldenState) -> None:
    counts = progress_counts(state)
    output = {
        "current_index": state.index,
        "order": state.order,
        "last_opened": utc_timestamp(),
        "counts": counts,
        "skipped": list(state.skipped),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_or_create_candidates(
    run_dir: Path,
    *,
    candidates_path: Path,
    n: int | None,
    seed: int,
    include_statuses: tuple[str, ...],
) -> list[AssistedGoldenCandidate]:
    prefilled_path = run_dir / GOLDEN_CANDIDATES_PREFILLED_FILENAME
    if prefilled_path.exists():
        return load_golden_candidates_jsonl(prefilled_path)
    if candidates_path.exists():
        return load_golden_candidates_jsonl(candidates_path)
    candidates = load_assisted_candidates(run_dir, n=n, seed=seed, include_statuses=include_statuses)
    write_golden_candidates_jsonl(candidates_path, candidates)
    return candidates


def _view_outputs(state: AssistedGoldenState) -> tuple[Any, ...]:
    candidate = current_candidate(state)
    counts = progress_counts(state)
    progress = (
        f"Labeled: {counts['labeled']} / {counts['total']}\n\n"
        f"Corrected: {counts['corrected']}\n\n"
        f"Accepted as-is: {counts['accepted_as_is']}\n\n"
        f"Skipped: {counts['skipped']}\n\n"
        f"Remaining: {counts['remaining']}"
    )
    if candidate is None:
        return (
            progress,
            "",
            None,
            "No candidates loaded.",
            "unknown",
            "",
            "",
            "",
            "",
            "",
            "",
            {},
            {},
            {},
            "",
            {"data": []},
        )
    label = state.labels.get(candidate.sprite_id)
    gui_model = candidate_gui_model(candidate, label)
    editable = gui_model["editable"]
    category = str(editable["category"])
    object_name = str(editable["object_name"])
    tags = ", ".join(str(tag) for tag in editable["tags"])
    description = str(editable["short_description"])
    materials = ", ".join(str(value) for value in editable["materials"])
    mood = ", ".join(str(value) for value in editable["mood"])
    notes = str(editable["notes"])
    source_info = (
        f"Source: `{candidate.source_name}`\n\n"
        f"Path: `{candidate.relative_path}`\n\n"
        f"License: `{candidate.license}`  Author: `{candidate.author}`"
    )
    sprite_info = (
        f"Sprite ID: `{candidate.sprite_id}`\n\n"
        f"Index: {state.index + 1} / {len(state.candidates)}\n\n"
        f"Prefill source: `{candidate.prefill_source or candidate.suggested_source}`\n\n"
        f"Already labeled: {'yes' if label else 'no'}"
    )
    filename_json = {
        "category": candidate.rule_category,
        "object_name": candidate.rule_object_name,
        "tags": list(candidate.rule_tags),
    }
    qwen_json = {
        "category": candidate.qwen_category,
        "object_name": candidate.qwen_object_name,
        "tags": list(candidate.qwen_tags),
        "short_description": candidate.qwen_description,
        "confidence": candidate.qwen_confidence,
        "warnings": list(candidate.qwen_warnings),
        "alternative_object_names": list(candidate.alternative_object_names),
        "source_consistency": candidate.vlm_source_consistency,
        "vlm_short_description": candidate.vlm_short_description,
    }
    fused_json = {
        "category": candidate.fused_category,
        "object_name": candidate.fused_object_name,
        "tags": list(candidate.fused_tags),
        "short_description": candidate.fused_description,
        "candidate_object_names": list(candidate.candidate_object_names),
        "prefill": gui_model["reference"]["prefill"],
        "visual_facts": gui_model["reference"]["visual_facts"],
    }
    quality = (
        f"Bucket: {candidate.quality_bucket or 'unknown'}\n\n"
        f"Review priority: {candidate.review_priority:.2f}\n\n"
        f"Flags: {', '.join(candidate.fused_quality_flags) or 'none'}\n\n"
        f"Review reason: {candidate.needs_review_reason or 'none'}\n\n"
        f"Object choices: {', '.join(gui_model['object_name_choices']) or 'none'}\n\n"
        f"Candidates: {', '.join(candidate.candidate_object_names) or 'none'}\n\n"
        f"Alternatives: {', '.join(candidate.alternative_object_names) or 'none'}"
    )
    table = {"headers": ["#", "status", "category", "sprite_id", "source"], "data": filtered_table_rows(state)}
    return (
        progress,
        source_info,
        _preview_image(candidate.final_png_path),
        sprite_info,
        category,
        object_name,
        tags,
        description,
        materials,
        mood,
        notes,
        filename_json,
        qwen_json,
        fused_json,
        quality,
        table,
    )


def _next_unlabeled_or_next(state: AssistedGoldenState, start: int) -> int:
    if not state.candidates:
        return 0
    for index in range(max(0, start), len(state.candidates)):
        candidate = state.candidates[index]
        if candidate.sprite_id not in state.labels and candidate.sprite_id not in state.skipped:
            return index
    return min(max(0, start), len(state.candidates) - 1)


def _uncertainty_score(candidate: AssistedGoldenCandidate) -> float:
    score = candidate_review_priority(candidate)
    if candidate.qwen_confidence is None:
        score += 5.0
    else:
        score += max(0.0, 1.0 - candidate.qwen_confidence) * 10.0
    return score


def _has_conflict(candidate: AssistedGoldenCandidate) -> bool:
    return bool(candidate.needs_review_reason) or "filename_qwen_conflict" in candidate.fused_quality_flags


def _preview_image(path: Path, scale: int = 10) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
    except Exception:
        return None
    preview = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
    checker = _checkerboard(preview.size)
    return Image.alpha_composite(checker, preview).convert("RGB")


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
