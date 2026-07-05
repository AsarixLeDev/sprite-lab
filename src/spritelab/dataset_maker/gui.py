"""Local Gradio GUI for building sprite training datasets."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.dataset_maker.exporter import DatasetMakerExportConfig, export_dataset_from_imported_sprites
from spritelab.dataset_maker.importer import ImportOptions, ImportedSprite, import_png_as_dataset_item, import_png_directory
from spritelab.dataset_maker.model import DatasetMakerItem
from spritelab.dataset_maker.prefill import (
    MetadataSuggestion,
    PrefillConfig,
    PrefillRequest,
    apply_suggestion_to_item,
    create_prefill_backend,
    suggestion_to_json_dict,
)
from spritelab.dataset_maker.report import build_dataset_maker_report

LICENSE_CHOICES = [
    "unknown",
    "own_work",
    "cc0",
    "cc_by",
    "cc_by_sa",
    "public_domain",
    "commercial_allowed",
    "custom",
]
STATUS_CHOICES = ["accepted", "rejected", "needs_fix", "quarantine"]
SPLIT_CHOICES = ["auto", "train", "val", "test"]


def launch_dataset_maker_gui(
    output_root: str | Path = "datasets",
    host: str = "127.0.0.1",
    port: int | None = None,
) -> None:
    """Launch the local Dataset Maker GUI."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The dataset maker GUI requires gradio. Install it with: pip install gradio") from exc

    def on_import(
        uploaded_files: Any,
        directory_path: str,
        default_category: str,
        default_tags: str,
        max_palette_slots: float,
        quantize_overcolor: bool,
        infer_role_map: bool,
        canonicalize_palette: bool,
        allow_resize: bool,
        recursive: bool,
    ) -> tuple[Any, ...]:
        options = ImportOptions(
            max_palette_slots=int(max_palette_slots or 32),
            allow_quantize_overcolor=True,
            quantize_overcolor=bool(quantize_overcolor),
            allow_nearest_resize=bool(allow_resize),
            infer_role_map=bool(infer_role_map),
            canonicalize_palette=bool(canonicalize_palette),
            recursive=bool(recursive),
        )
        tags = _parse_tags(default_tags)
        imported: list[ImportedSprite] = []
        seen_paths: set[str] = set()
        for path in _coerce_upload_paths(uploaded_files):
            key = str(Path(path))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            imported.append(
                import_png_as_dataset_item(path, options=options, default_category=default_category, default_tags=tags)
            )
        if directory_path.strip():
            for sprite in import_png_directory(
                directory_path.strip(),
                options=options,
                default_category=default_category,
                default_tags=tags,
            ):
                key = str(sprite.item.source_path)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                imported.append(sprite)
        index = 0
        suggestions: dict[str, Any] = {}
        return (imported, index, suggestions, _import_summary(imported), *_view(imported, index), *_suggestion_view(suggestions, index))

    def previous(imported: list[ImportedSprite], suggestions: dict[str, Any], index: int) -> tuple[Any, ...]:
        if not imported:
            return (0, *_view(imported, 0), *_suggestion_view(suggestions or {}, 0))
        next_index = max(0, int(index or 0) - 1)
        return (next_index, *_view(imported, next_index), *_suggestion_view(suggestions or {}, next_index))

    def next_item(imported: list[ImportedSprite], suggestions: dict[str, Any], index: int) -> tuple[Any, ...]:
        if not imported:
            return (0, *_view(imported, 0), *_suggestion_view(suggestions or {}, 0))
        next_index = min(len(imported) - 1, int(index or 0) + 1)
        return (next_index, *_view(imported, next_index), *_suggestion_view(suggestions or {}, next_index))

    def save_current(
        imported: list[ImportedSprite],
        index: int,
        sprite_id: str,
        status: str,
        category: str,
        tags: str,
        notes: str,
        source_name: str,
        license_value: str,
        author: str,
        split: str,
    ) -> tuple[Any, ...]:
        updated = _save_item(
            imported,
            index,
            sprite_id=sprite_id,
            status=status,
            category=category,
            tags=_parse_tags(tags),
            notes=notes,
            source_name=source_name,
            license_value=license_value,
            author=author,
            split=split,
        )
        return (updated, *_view(updated, int(index or 0)))

    def set_status(imported: list[ImportedSprite], index: int, status: str) -> tuple[Any, ...]:
        if not imported:
            return (imported, *_view(imported, 0))
        sprite = imported[int(index or 0)]
        item = _copy_item(sprite.item, status=status)
        updated = list(imported)
        updated[int(index or 0)] = replace(sprite, item=item)
        return (updated, *_view(updated, int(index or 0)))

    def bulk_apply(
        imported: list[ImportedSprite],
        index: int,
        scope: str,
        filter_status: str,
        filter_category: str,
        filter_search: str,
        filter_errors: bool,
        filter_warnings: bool,
        filter_palette_over: bool,
        filter_palette_limit: float,
        bulk_category: str,
        bulk_tags_add: str,
        bulk_license: str,
        bulk_author: str,
        bulk_status: str,
    ) -> tuple[Any, ...]:
        updated = list(imported or [])
        selected = _bulk_indices(
            updated,
            current_index=int(index or 0),
            scope=scope,
            filter_status=filter_status,
            filter_category=filter_category,
            filter_search=filter_search,
            filter_errors=filter_errors,
            filter_warnings=filter_warnings,
            filter_palette_over=filter_palette_over,
            filter_palette_limit=int(filter_palette_limit or 32),
        )
        add_tags = _parse_tags(bulk_tags_add)
        for item_index in selected:
            sprite = updated[item_index]
            tags = tuple(dict.fromkeys((*sprite.item.tags, *add_tags)))
            item = _copy_item(
                sprite.item,
                category=bulk_category.strip() or sprite.item.category,
                tags=tags,
                license=bulk_license if bulk_license != "no_change" else sprite.item.license,
                author=bulk_author.strip() or sprite.item.author,
                status=bulk_status if bulk_status != "no_change" else sprite.item.status,
            )
            updated[item_index] = replace(sprite, item=item)
        summary = f"Applied bulk edit to {len(selected)} sprite(s)."
        return (updated, summary, *_view(updated, int(index or 0)))

    def export_dataset(
        imported: list[ImportedSprite],
        dataset_name: str,
        output_root_value: str,
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        seed: float,
        max_palette_slots: float,
        overwrite: bool,
    ) -> tuple[str, str]:
        try:
            result = export_dataset_from_imported_sprites(
                imported or [],
                DatasetMakerExportConfig(
                    dataset_name=dataset_name,
                    output_root=Path(output_root_value or output_root),
                    max_palette_slots=int(max_palette_slots or 32),
                    train_fraction=float(train_fraction),
                    val_fraction=float(val_fraction),
                    test_fraction=float(test_fraction),
                    seed=int(seed or 1337),
                    overwrite=bool(overwrite),
                ),
            )
        except Exception as exc:
            return (f"Export blocked: {exc}", build_dataset_maker_report(imported or []))
        summary = (
            f"Output: {result.output_dir}\n\n"
            f"Train: {result.train_count}\n"
            f"Val: {result.val_count}\n"
            f"Test: {result.test_count}\n"
            f"Excluded: {result.excluded_count}"
        )
        return (summary, build_dataset_maker_report(imported or [], result))

    def refresh_report(imported: list[ImportedSprite]) -> str:
        return build_dataset_maker_report(imported or [])

    def prefill_current(
        imported: list[ImportedSprite],
        suggestions: dict[str, Any],
        index: int,
        enabled: bool,
        backend_name: str,
        model: str,
        base_url: str,
        api_key: str,
        runpod_token: str,
        timeout_seconds: float,
        cache_dir: str,
        auto_apply: bool,
        overwrite_existing: bool,
    ) -> tuple[Any, ...]:
        updated = list(imported or [])
        suggestions = dict(suggestions or {})
        if not updated:
            return (updated, suggestions, "No sprites imported.", *_view(updated, 0), *_suggestion_view(suggestions, 0))
        safe_index = min(max(0, int(index or 0)), len(updated) - 1)
        config = _prefill_config(
            enabled=enabled,
            backend_name=backend_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            runpod_token=runpod_token,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
        )
        blocked = _prefill_blocked_warning(config)
        if blocked is not None:
            suggestion = MetadataSuggestion(warnings=(blocked,))
            suggestions[_suggestion_key(safe_index)] = _suggestion_state_value(suggestion)
            summary = _prefill_report(
                [(updated[safe_index].item.sprite_id, suggestion)],
                attempted=0,
                applied=0,
                config=config,
                selected_ids=(updated[safe_index].item.sprite_id,),
                notes=(blocked,),
            )
            return (updated, suggestions, summary, *_view(updated, safe_index), *_suggestion_view(suggestions, safe_index))
        backend = create_prefill_backend(config)
        suggestion = _suggest_for_sprite(backend, updated[safe_index])
        suggestions[_suggestion_key(safe_index)] = _suggestion_state_value(suggestion)
        applied = 0
        if auto_apply:
            sprite = updated[safe_index]
            item = apply_suggestion_to_item(sprite.item, suggestion, overwrite_existing=bool(overwrite_existing))
            updated[safe_index] = replace(sprite, item=item)
            applied = 1
        summary = _prefill_report(
            [(updated[safe_index].item.sprite_id, suggestion)],
            attempted=1,
            applied=applied,
            config=config,
            selected_ids=(updated[safe_index].item.sprite_id,),
        )
        return (updated, suggestions, summary, *_view(updated, safe_index), *_suggestion_view(suggestions, safe_index))

    def prefill_filtered(
        imported: list[ImportedSprite],
        suggestions: dict[str, Any],
        index: int,
        enabled: bool,
        backend_name: str,
        model: str,
        base_url: str,
        api_key: str,
        runpod_token: str,
        timeout_seconds: float,
        cache_dir: str,
        workers: float | None,
        auto_apply: bool,
        overwrite_existing: bool,
        scope: str,
        filter_status: str,
        filter_category: str,
        filter_search: str,
        filter_errors: bool,
        filter_warnings: bool,
        filter_palette_over: bool,
        filter_palette_limit: float,
    ) -> tuple[Any, ...]:
        updated = list(imported or [])
        suggestions = dict(suggestions or {})
        if not updated:
            return (updated, suggestions, "No sprites imported.", *_view(updated, 0), *_suggestion_view(suggestions, 0))
        selected = _bulk_indices(
            updated,
            current_index=int(index or 0),
            scope=scope,
            filter_status=filter_status,
            filter_category=filter_category,
            filter_search=filter_search,
            filter_errors=filter_errors,
            filter_warnings=filter_warnings,
            filter_palette_over=filter_palette_over,
            filter_palette_limit=int(filter_palette_limit or 32),
        )
        config = _prefill_config(
            enabled=enabled,
            backend_name=backend_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            runpod_token=runpod_token,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
        )
        selected_ids = tuple(updated[item_index].item.sprite_id for item_index in selected)
        if not selected:
            safe_index = min(max(0, int(index or 0)), len(updated) - 1)
            summary = _prefill_report(
                [],
                attempted=0,
                applied=0,
                config=config,
                selected_ids=(),
                notes=("No sprites matched the selected bulk prefill scope and filters.",),
            )
            return (updated, suggestions, summary, *_view(updated, safe_index), *_suggestion_view(suggestions, safe_index))
        blocked = _prefill_blocked_warning(config)
        if blocked is not None:
            blocked_results: list[tuple[str, MetadataSuggestion]] = []
            for item_index in selected:
                suggestion = MetadataSuggestion(warnings=(blocked,))
                suggestions[_suggestion_key(item_index)] = _suggestion_state_value(suggestion)
                blocked_results.append((updated[item_index].item.sprite_id, suggestion))
            safe_index = min(max(0, int(index or 0)), len(updated) - 1)
            summary = _prefill_report(
                blocked_results,
                attempted=0,
                applied=0,
                config=config,
                selected_ids=selected_ids,
                notes=(blocked,),
            )
            return (updated, suggestions, summary, *_view(updated, safe_index), *_suggestion_view(suggestions, safe_index))
        backend = create_prefill_backend(config)
        generated: list[tuple[str, MetadataSuggestion]] = []
        applied = 0
        worker_count = max(1, int(workers or 1))
        suggestion_by_index = _bulk_prefill_suggestions(updated, selected, backend, workers=worker_count)
        for item_index in selected:
            suggestion = suggestion_by_index[item_index]
            suggestions[_suggestion_key(item_index)] = _suggestion_state_value(suggestion)
            generated.append((updated[item_index].item.sprite_id, suggestion))
            if auto_apply:
                sprite = updated[item_index]
                item = apply_suggestion_to_item(sprite.item, suggestion, overwrite_existing=bool(overwrite_existing))
                updated[item_index] = replace(sprite, item=item)
                applied += 1
        safe_index = min(max(0, int(index or 0)), len(updated) - 1)
        summary = _prefill_report(
            generated,
            attempted=len(selected),
            applied=applied,
            config=config,
            selected_ids=selected_ids,
            workers=worker_count,
        )
        return (updated, suggestions, summary, *_view(updated, safe_index), *_suggestion_view(suggestions, safe_index))

    def apply_current_suggestion(
        imported: list[ImportedSprite],
        suggestions: dict[str, Any],
        index: int,
        overwrite_existing: bool,
    ) -> tuple[Any, ...]:
        updated = list(imported or [])
        suggestions = dict(suggestions or {})
        if not updated:
            return (updated, "No sprites imported.", *_view(updated, 0), *_suggestion_view(suggestions, 0))
        safe_index = min(max(0, int(index or 0)), len(updated) - 1)
        suggestion = _suggestion_from_state(suggestions.get(_suggestion_key(safe_index)))
        if suggestion is None:
            return (updated, "No suggestion for the current sprite.", *_view(updated, safe_index), *_suggestion_view(suggestions, safe_index))
        sprite = updated[safe_index]
        item = apply_suggestion_to_item(sprite.item, suggestion, overwrite_existing=bool(overwrite_existing))
        updated[safe_index] = replace(sprite, item=item)
        return (updated, "Applied suggestion to current sprite.", *_view(updated, safe_index), *_suggestion_view(suggestions, safe_index))

    def discard_current_suggestion(
        imported: list[ImportedSprite],
        suggestions: dict[str, Any],
        index: int,
    ) -> tuple[Any, ...]:
        suggestions = dict(suggestions or {})
        safe_index = min(max(0, int(index or 0)), max(0, len(imported or []) - 1))
        suggestions.pop(_suggestion_key(safe_index), None)
        return (suggestions, "Discarded current suggestion.", *_suggestion_view(suggestions, safe_index))

    with gr.Blocks(title="Sprite Lab Dataset Maker") as demo:
        imported_state = gr.State([])
        index_state = gr.State(0)
        suggestions_state = gr.State({})

        gr.Markdown("# Dataset Maker GUI")
        with gr.Tab("Import"):
            with gr.Row():
                uploaded_files = gr.File(label="PNG files", file_count="multiple", type="filepath")
                directory_path = gr.Textbox(label="Local PNG directory")
            with gr.Row():
                default_category = gr.Textbox(label="Default category", value="unknown")
                default_tags = gr.Textbox(label="Default tags", placeholder="minecraft,item_icon")
                max_palette_slots = gr.Number(label="Max palette slots", value=32, precision=0)
            with gr.Row():
                quantize_overcolor = gr.Checkbox(label="Quantize over-color sprites", value=True)
                infer_role_map = gr.Checkbox(label="Infer role map", value=True)
                canonicalize_palette = gr.Checkbox(label="Canonicalize palette", value=True)
                allow_resize = gr.Checkbox(label="Allow nearest-neighbor resize to 32x32", value=False)
                recursive = gr.Checkbox(label="Recursive directory import", value=False)
            import_button = gr.Button("Import")
            import_summary = gr.Markdown()

        with gr.Tab("Review"):
            current_label = gr.Markdown("No sprites imported.")
            with gr.Row():
                preview_image = gr.Image(label="Sprite preview", type="pil", height=300)
                alpha_preview_image = gr.Image(label="Alpha preview", type="pil", height=300)
                role_preview_image = gr.Image(label="Role-map preview", type="pil", height=300)
                palette_strip_image = gr.Image(label="Palette", type="pil", height=80)
            source_path_md = gr.Markdown()
            validation_md = gr.Markdown()
            palette_size_md = gr.Markdown()
            with gr.Row():
                previous_button = gr.Button("Previous")
                next_button = gr.Button("Next")
                save_button = gr.Button("Save edits", variant="primary")
            with gr.Row():
                accept_button = gr.Button("Accept")
                reject_button = gr.Button("Reject")
                needs_fix_button = gr.Button("Needs fix")
                quarantine_button = gr.Button("Quarantine")
            with gr.Row():
                sprite_id = gr.Textbox(label="sprite_id")
                status = gr.Dropdown(label="status", choices=STATUS_CHOICES, value="accepted")
                category = gr.Textbox(label="category")
                split = gr.Dropdown(label="split override", choices=SPLIT_CHOICES, value="auto")
            tags = gr.Textbox(label="tags")
            notes = gr.Textbox(label="notes", lines=3)
            with gr.Row():
                source_name = gr.Textbox(label="source name")
                license_value = gr.Dropdown(label="license", choices=LICENSE_CHOICES, value="unknown", allow_custom_value=True)
                author = gr.Textbox(label="author")
            with gr.Accordion("Last metadata suggestion", open=False):
                suggestion_markdown = gr.Markdown("No suggestion for the current sprite.")
                suggestion_raw = gr.Textbox(label="raw response", lines=6, interactive=False)
                with gr.Row():
                    apply_suggestion_button = gr.Button("Apply suggestion to current item")
                    discard_suggestion_button = gr.Button("Discard suggestion")

        with gr.Tab("Bulk edit"):
            with gr.Row():
                scope = gr.Dropdown(label="Scope", choices=["current", "filtered", "all"], value="filtered")
                filter_status = gr.Dropdown(label="Filter status", choices=["any", *STATUS_CHOICES], value="any")
                filter_category = gr.Textbox(label="Filter category")
                filter_search = gr.Textbox(label="Search")
            with gr.Row():
                filter_errors = gr.Checkbox(label="Has errors", value=False)
                filter_warnings = gr.Checkbox(label="Has warnings", value=False)
                filter_palette_over = gr.Checkbox(label="Palette size over limit", value=False)
                filter_palette_limit = gr.Number(label="Palette limit", value=32, precision=0)
            with gr.Row():
                bulk_category = gr.Textbox(label="Apply category")
                bulk_tags_add = gr.Textbox(label="Add tags")
                bulk_license = gr.Dropdown(label="Apply license", choices=["no_change", *LICENSE_CHOICES], value="no_change", allow_custom_value=True)
                bulk_author = gr.Textbox(label="Apply author")
                bulk_status = gr.Dropdown(label="Apply status", choices=["no_change", *STATUS_CHOICES], value="no_change")
            bulk_button = gr.Button("Apply bulk edit")
            bulk_summary = gr.Markdown()

        with gr.Tab("Auto-fill with local Qwen"):
            with gr.Row():
                prefill_enabled = gr.Checkbox(label="Enable auto-fill", value=False)
                prefill_backend = gr.Dropdown(
                    label="Backend",
                    choices=["openai_compatible", "ollama", "rule_based", "none"],
                    value="openai_compatible",
                )
                prefill_timeout = gr.Number(label="Timeout seconds", value=60)
            with gr.Row():
                prefill_model = gr.Textbox(label="Model", value="Qwen/Qwen3-VL-8B-Instruct")
                prefill_base_url = gr.Textbox(label="Base URL", value="http://127.0.0.1:8000/v1")
                prefill_api_key = gr.Textbox(label="API key", value="not-needed", type="password")
                prefill_runpod_token = gr.Textbox(label="RunPod token", value="", type="password")
                prefill_cache_dir = gr.Textbox(label="Cache directory", value=".prefill_cache")
            with gr.Row():
                prefill_auto_apply = gr.Checkbox(label="Apply automatically to empty fields only", value=False)
                prefill_overwrite = gr.Checkbox(label="Overwrite existing fields", value=False)
                prefill_scope = gr.Dropdown(label="Bulk scope", choices=["current", "filtered", "all"], value="filtered")
                prefill_workers = gr.Number(label="Bulk workers", value=1)
            with gr.Row():
                prefill_current_button = gr.Button("Prefill current sprite", variant="primary")
                prefill_filtered_button = gr.Button("Prefill all visible/filter-matched sprites")
            prefill_summary = gr.Markdown(
                "Enable auto-fill, choose a backend, then click a prefill button. "
                "For a quick local smoke test without a model server, choose `rule_based`."
            )

        with gr.Tab("Dataset export"):
            with gr.Row():
                dataset_name = gr.Textbox(label="Dataset name", value="v0")
                output_root_box = gr.Textbox(label="Output root", value=str(output_root))
                export_max_palette_slots = gr.Number(label="Max palette slots", value=32, precision=0)
            with gr.Row():
                train_fraction = gr.Number(label="Train fraction", value=0.8)
                val_fraction = gr.Number(label="Val fraction", value=0.1)
                test_fraction = gr.Number(label="Test fraction", value=0.1)
                seed = gr.Number(label="Seed", value=1337, precision=0)
                overwrite = gr.Checkbox(label="Overwrite", value=False)
            export_button = gr.Button("Export dataset", variant="primary")
            export_summary = gr.Markdown()

        with gr.Tab("Report"):
            report_button = gr.Button("Refresh report")
            report_markdown = gr.Markdown()

        view_outputs = [
            current_label,
            preview_image,
            alpha_preview_image,
            role_preview_image,
            palette_strip_image,
            source_path_md,
            validation_md,
            palette_size_md,
            sprite_id,
            status,
            category,
            tags,
            notes,
            source_name,
            license_value,
            split,
            author,
        ]
        suggestion_outputs = [suggestion_markdown, suggestion_raw]
        import_button.click(
            on_import,
            inputs=[
                uploaded_files,
                directory_path,
                default_category,
                default_tags,
                max_palette_slots,
                quantize_overcolor,
                infer_role_map,
                canonicalize_palette,
                allow_resize,
                recursive,
            ],
            outputs=[imported_state, index_state, suggestions_state, import_summary, *view_outputs, *suggestion_outputs],
        )
        previous_button.click(previous, inputs=[imported_state, suggestions_state, index_state], outputs=[index_state, *view_outputs, *suggestion_outputs])
        next_button.click(next_item, inputs=[imported_state, suggestions_state, index_state], outputs=[index_state, *view_outputs, *suggestion_outputs])
        save_button.click(
            save_current,
            inputs=[
                imported_state,
                index_state,
                sprite_id,
                status,
                category,
                tags,
                notes,
                source_name,
                license_value,
                author,
                split,
            ],
            outputs=[imported_state, *view_outputs],
        )
        accept_button.click(lambda data, index: set_status(data, index, "accepted"), inputs=[imported_state, index_state], outputs=[imported_state, *view_outputs])
        reject_button.click(lambda data, index: set_status(data, index, "rejected"), inputs=[imported_state, index_state], outputs=[imported_state, *view_outputs])
        needs_fix_button.click(lambda data, index: set_status(data, index, "needs_fix"), inputs=[imported_state, index_state], outputs=[imported_state, *view_outputs])
        quarantine_button.click(lambda data, index: set_status(data, index, "quarantine"), inputs=[imported_state, index_state], outputs=[imported_state, *view_outputs])
        bulk_button.click(
            bulk_apply,
            inputs=[
                imported_state,
                index_state,
                scope,
                filter_status,
                filter_category,
                filter_search,
                filter_errors,
                filter_warnings,
                filter_palette_over,
                filter_palette_limit,
                bulk_category,
                bulk_tags_add,
                bulk_license,
                bulk_author,
                bulk_status,
            ],
            outputs=[imported_state, bulk_summary, *view_outputs],
        )
        prefill_current_button.click(
            prefill_current,
            inputs=[
                imported_state,
                suggestions_state,
                index_state,
                prefill_enabled,
                prefill_backend,
                prefill_model,
                prefill_base_url,
                prefill_api_key,
                prefill_runpod_token,
                prefill_timeout,
                prefill_cache_dir,
                prefill_auto_apply,
                prefill_overwrite,
            ],
            outputs=[imported_state, suggestions_state, prefill_summary, *view_outputs, *suggestion_outputs],
        )
        prefill_filtered_button.click(
            prefill_filtered,
            inputs=[
                imported_state,
                suggestions_state,
                index_state,
                prefill_enabled,
                prefill_backend,
                prefill_model,
                prefill_base_url,
                prefill_api_key,
                prefill_runpod_token,
                prefill_timeout,
                prefill_cache_dir,
                prefill_workers,
                prefill_auto_apply,
                prefill_overwrite,
                prefill_scope,
                filter_status,
                filter_category,
                filter_search,
                filter_errors,
                filter_warnings,
                filter_palette_over,
                filter_palette_limit,
            ],
            outputs=[imported_state, suggestions_state, prefill_summary, *view_outputs, *suggestion_outputs],
        )
        apply_suggestion_button.click(
            apply_current_suggestion,
            inputs=[imported_state, suggestions_state, index_state, prefill_overwrite],
            outputs=[imported_state, prefill_summary, *view_outputs, *suggestion_outputs],
        )
        discard_suggestion_button.click(
            discard_current_suggestion,
            inputs=[imported_state, suggestions_state, index_state],
            outputs=[suggestions_state, prefill_summary, *suggestion_outputs],
        )
        export_button.click(
            export_dataset,
            inputs=[
                imported_state,
                dataset_name,
                output_root_box,
                train_fraction,
                val_fraction,
                test_fraction,
                seed,
                export_max_palette_slots,
                overwrite,
            ],
            outputs=[export_summary, report_markdown],
        )
        report_button.click(refresh_report, inputs=[imported_state], outputs=[report_markdown])

    demo.launch(server_name=host, server_port=port)


def _view(imported: list[ImportedSprite], index: int) -> tuple[Any, ...]:
    if not imported:
        return (
            "No sprites imported.",
            None,
            None,
            None,
            None,
            "",
            "",
            "",
            "",
            "accepted",
            "unknown",
            "",
            "",
            "",
            "unknown",
            "auto",
            "",
        )
    safe_index = min(max(0, int(index or 0)), len(imported) - 1)
    sprite = imported[safe_index]
    item = sprite.item
    messages = _messages(imported, safe_index)
    split = item.split or "auto"
    return (
        f"Sprite {safe_index + 1} / {len(imported)}",
        sprite.preview_image,
        sprite.alpha_preview_image,
        sprite.role_preview_image,
        sprite.palette_strip_image,
        f"Source: `{item.source_path}`",
        messages,
        f"Palette size: {item.palette_size if item.palette_size is not None else 'unknown'}",
        item.sprite_id,
        item.status,
        item.category,
        ", ".join(item.tags),
        item.notes,
        item.source_name,
        item.license,
        split,
        item.author,
    )


def _messages(imported: list[ImportedSprite], index: int) -> str:
    sprite = imported[index]
    lines: list[str] = []
    duplicate_ids = {
        sprite_id
        for sprite_id, count in Counter(item.item.sprite_id for item in imported).items()
        if count > 1 and sprite_id
    }
    if sprite.item.sprite_id in duplicate_ids:
        lines.append("- Duplicate sprite_id; export is blocked until this is unique.")
    lines.extend(f"- Error: {error}" for error in sprite.errors)
    lines.extend(f"- Warning: {warning}" for warning in sprite.warnings)
    return "\n".join(lines) if lines else "No validation errors or warnings."


def _prefill_config(
    *,
    enabled: bool,
    backend_name: str,
    model: str,
    base_url: str,
    api_key: str,
    runpod_token: str = "",
    timeout_seconds: float,
    cache_dir: str,
) -> PrefillConfig:
    cache_path = Path(cache_dir.strip()) if str(cache_dir or "").strip() else None
    backend = str(backend_name or "none")
    effective_base_url = base_url or "http://127.0.0.1:8000/v1"
    if backend == "ollama" and effective_base_url.rstrip("/") == "http://127.0.0.1:8000/v1":
        effective_base_url = "http://127.0.0.1:11434"
    return PrefillConfig(
        enabled=bool(enabled) and backend_name != "none",
        backend=backend,
        model=model or "Qwen/Qwen3-VL-8B-Instruct",
        base_url=effective_base_url,
        api_key=api_key or "not-needed",
        runpod_token=runpod_token,
        timeout_seconds=float(timeout_seconds or 60.0),
        cache_dir=cache_path,
    )


def _suggest_for_sprite(backend: Any, sprite: ImportedSprite) -> MetadataSuggestion:
    try:
        image = _image_for_prefill(sprite)
    except Exception as exc:
        return MetadataSuggestion(warnings=(f"Could not prepare image for prefill: {exc}",))
    request = PrefillRequest(
        sprite_id=sprite.item.sprite_id,
        image=image,
        existing_category=sprite.item.category,
        existing_tags=sprite.item.tags,
        source_path=str(sprite.item.source_path),
    )
    try:
        return backend.suggest(request)
    except Exception as exc:
        return MetadataSuggestion(warnings=(f"Prefill failed: {exc}",))


def _bulk_prefill_suggestions(
    sprites: list[ImportedSprite],
    selected_indices: list[int],
    backend: Any,
    *,
    workers: int,
) -> dict[int, MetadataSuggestion]:
    worker_count = max(1, int(workers or 1))
    if worker_count == 1 or len(selected_indices) <= 1:
        return {item_index: _suggest_for_sprite(backend, sprites[item_index]) for item_index in selected_indices}

    results: dict[int, MetadataSuggestion] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_suggest_for_sprite, backend, sprites[item_index]): item_index
            for item_index in selected_indices
        }
        for future in as_completed(future_to_index):
            item_index = future_to_index[future]
            try:
                results[item_index] = future.result()
            except Exception as exc:
                results[item_index] = MetadataSuggestion(warnings=(f"Prefill failed: {exc}",))
    return results


def _image_for_prefill(sprite: ImportedSprite) -> Image.Image:
    if sprite.bundle is not None:
        return reconstruct_rgba(sprite.bundle)
    if sprite.preview_image is not None:
        return sprite.preview_image.convert("RGBA")
    with Image.open(sprite.item.source_path) as image:
        return image.convert("RGBA")


def _prefill_blocked_warning(config: PrefillConfig) -> str | None:
    if config.backend == "none":
        return "Auto-fill backend is set to none. Choose rule_based, ollama, or openai_compatible."
    if not config.enabled:
        return "Auto-fill is disabled. Turn on the Enable auto-fill checkbox before running prefill."
    return None


def _prefill_report(
    results: list[tuple[str, MetadataSuggestion]],
    *,
    attempted: int,
    applied: int,
    config: PrefillConfig,
    selected_ids: tuple[str, ...],
    workers: int = 1,
    notes: tuple[str, ...] = (),
) -> str:
    warning_rows = [
        (sprite_id, warning)
        for sprite_id, suggestion in results
        for warning in suggestion.warnings
    ]
    useful_count = sum(1 for _sprite_id, suggestion in results if _suggestion_has_content(suggestion))
    lines = [
        "## Prefill report",
        f"Backend: `{config.backend}`",
        f"Model: `{config.model}`",
        f"Base URL: `{config.base_url or 'not configured'}`",
        f"Selected sprites: {len(selected_ids)}",
        f"Attempted requests: {attempted}",
        f"Workers: {max(1, int(workers or 1))}",
        f"Suggestions stored: {len(results)}",
        f"Suggestions with metadata: {useful_count}",
        f"Applied to items: {applied}",
    ]
    if selected_ids:
        preview_ids = ", ".join(selected_ids[:12])
        suffix = " ..." if len(selected_ids) > 12 else ""
        lines.append(f"Selected IDs: `{preview_ids}{suffix}`")
    if notes:
        lines.append("")
        lines.append("### Status")
        lines.extend(f"- {note}" for note in notes)
    if warning_rows:
        lines.append("")
        lines.append("### Warnings")
        for sprite_id, warning in warning_rows[:20]:
            lines.append(f"- `{sprite_id}`: {warning}")
        if len(warning_rows) > 20:
            lines.append(f"- ... {len(warning_rows) - 20} more warning(s)")
    if attempted > 0 and results and useful_count == 0 and not warning_rows:
        lines.append("")
        lines.append("### Status")
        lines.append("- The backend returned no metadata fields. Check the backend selection and model response.")
    return "\n".join(lines)


def _suggestion_has_content(suggestion: MetadataSuggestion) -> bool:
    return any(
        [
            suggestion.category != "unknown",
            bool(suggestion.object_name),
            bool(suggestion.tags),
            bool(suggestion.materials),
            bool(suggestion.mood),
            bool(suggestion.dominant_colors),
            bool(suggestion.short_description),
            bool(suggestion.suggested_sprite_id),
            suggestion.confidence is not None,
        ]
    )


def _suggestion_view(suggestions: dict[str, Any], index: int) -> tuple[str, str]:
    suggestion = _suggestion_from_state((suggestions or {}).get(_suggestion_key(index)))
    if suggestion is None:
        return ("No suggestion for the current sprite.", "")
    lines = [
        f"Category: `{suggestion.category}`",
        f"Object name: `{suggestion.object_name or ''}`",
        f"Tags: `{', '.join(suggestion.tags)}`",
        f"Materials: `{', '.join(suggestion.materials)}`",
        f"Mood: `{', '.join(suggestion.mood)}`",
        f"Dominant colors: `{', '.join(suggestion.dominant_colors)}`",
        f"Suggested sprite_id: `{suggestion.suggested_sprite_id}`",
        f"Short description: {suggestion.short_description or ''}",
        f"Confidence: {suggestion.confidence if suggestion.confidence is not None else 'unknown'}",
    ]
    if suggestion.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in suggestion.warnings)
    raw = suggestion.raw_response or json_dumps_for_gui(suggestion_to_json_dict(suggestion, include_raw=False))
    return ("\n\n".join(lines), raw)


def _suggestion_from_state(value: Any) -> MetadataSuggestion | None:
    if value is None:
        return None
    if isinstance(value, MetadataSuggestion):
        return value
    if isinstance(value, dict):
        return MetadataSuggestion(
            category=value.get("category", "unknown"),
            object_name=value.get("object_name", ""),
            tags=tuple(value.get("tags", ())),
            materials=tuple(value.get("materials", ())),
            mood=tuple(value.get("mood", ())),
            dominant_colors=tuple(value.get("dominant_colors", ())),
            short_description=value.get("short_description", ""),
            suggested_sprite_id=value.get("suggested_sprite_id", ""),
            confidence=value.get("confidence"),
            warnings=tuple(value.get("warnings", ())),
            raw_response=value.get("raw_response", ""),
        )
    return None


def _suggestion_state_value(suggestion: MetadataSuggestion) -> dict[str, Any]:
    return suggestion_to_json_dict(suggestion, include_raw=True)


def _suggestion_key(index: int) -> str:
    return str(int(index or 0))


def json_dumps_for_gui(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, sort_keys=True)


def _import_summary(imported: list[ImportedSprite]) -> str:
    counts = Counter(sprite.item.status for sprite in imported)
    warning_counts = Counter(warning for sprite in imported for warning in sprite.warnings)
    lines = [
        f"Imported: {len(imported)}",
        f"Valid: {counts['accepted']}",
        f"Rejected: {counts['rejected']}",
        f"Needs fix: {counts['needs_fix']}",
    ]
    if warning_counts:
        lines.append("")
        lines.append("Common warnings:")
        for warning, count in warning_counts.most_common(5):
            lines.append(f"- {warning}: {count}")
    return "\n".join(lines)


def _save_item(
    imported: list[ImportedSprite],
    index: int,
    *,
    sprite_id: str,
    status: str,
    category: str,
    tags: tuple[str, ...],
    notes: str,
    source_name: str,
    license_value: str,
    author: str,
    split: str,
) -> list[ImportedSprite]:
    if not imported:
        return []
    safe_index = min(max(0, int(index or 0)), len(imported) - 1)
    updated = list(imported)
    sprite = updated[safe_index]
    item = _copy_item(
        sprite.item,
        sprite_id=sprite_id,
        status=status,
        category=category,
        tags=tags,
        notes=notes,
        source_name=source_name,
        license=license_value,
        author=author,
        split=None if split == "auto" else split,
    )
    updated[safe_index] = replace(sprite, item=item)
    return updated


def _copy_item(item: DatasetMakerItem, **changes: Any) -> DatasetMakerItem:
    values = {
        "sprite_id": item.sprite_id,
        "source_path": item.source_path,
        "status": item.status,
        "category": item.category,
        "tags": item.tags,
        "notes": item.notes,
        "source_name": item.source_name,
        "license": item.license,
        "author": item.author,
        "split": item.split,
        "quality_issues": item.quality_issues,
        "palette_size": item.palette_size,
        "has_role_map": item.has_role_map,
    }
    values.update(changes)
    return DatasetMakerItem(**values)


def _bulk_indices(
    imported: list[ImportedSprite],
    *,
    current_index: int,
    scope: str,
    filter_status: str,
    filter_category: str,
    filter_search: str,
    filter_errors: bool,
    filter_warnings: bool,
    filter_palette_over: bool,
    filter_palette_limit: int,
) -> list[int]:
    if scope == "current":
        return [current_index] if imported else []
    if scope == "all":
        return list(range(len(imported)))
    selected: list[int] = []
    for index, sprite in enumerate(imported):
        item = sprite.item
        if filter_status != "any" and item.status != filter_status:
            continue
        if filter_category.strip() and item.category != filter_category.strip().lower().replace(" ", "_"):
            continue
        if filter_errors and not sprite.errors:
            continue
        if filter_warnings and not sprite.warnings:
            continue
        if filter_palette_over and (item.palette_size is None or item.palette_size <= filter_palette_limit):
            continue
        search = filter_search.strip().lower()
        haystack = " ".join([item.sprite_id, str(item.source_path), " ".join(item.tags)]).lower()
        if search and search not in haystack:
            continue
        selected.append(index)
    return selected


def _parse_tags(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _coerce_upload_paths(uploaded_files: Any) -> list[Path]:
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (str, Path)):
        return [Path(uploaded_files)]
    paths: list[Path] = []
    for value in uploaded_files:
        if isinstance(value, (str, Path)):
            paths.append(Path(value))
        elif hasattr(value, "name"):
            paths.append(Path(value.name))
        elif hasattr(value, "path"):
            paths.append(Path(value.path))
    return paths
