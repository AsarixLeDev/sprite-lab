"""Local Gradio GUI for the dataset harvester."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from spritelab.dataset_maker.model import normalize_tag
from spritelab.harvest.autolabel import (
    QwenBatchPrefillConfig,
    batch_prefill_with_qwen,
    merge_auto_labels,
    suggest_metadata_from_path,
)
from spritelab.harvest.catalog import (
    write_candidates_jsonl,
    write_imported_jsonl,
    write_jsonl,
    write_sources_jsonl,
)
from spritelab.harvest.pipeline import (
    HarvestImportOptions,
    HarvestPolicy,
    HarvestedSprite,
    apply_harvest_policy,
    export_harvested_dataset,
    harvest_source_to_imported_sprites,
)
from spritelab.harvest.report import build_harvest_report, write_harvest_reports
from spritelab.harvest.sheets import SheetSliceConfig
from spritelab.harvest.sources import KNOWN_LICENSES, SOURCE_TYPES, SourceLicense, SourceRecord

PAGE_SIZE = 100
STATUS_CHOICES = ["accepted", "rejected", "needs_fix", "quarantine"]


def launch_harvest_gui(
    output_root: str | Path = "datasets",
    run_root: str | Path = "harvest_runs",
    host: str = "127.0.0.1",
    port: int | None = None,
) -> None:
    """Launch the local harvest GUI."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The harvest GUI requires gradio. Install with: pip install gradio") from exc

    state: dict[str, Any] = {"sources": [], "harvested": [], "run_dir": None}

    def on_import(
        source_type: str,
        source_name: str,
        source_id: str,
        source_url: str,
        download_url: str,
        zip_path: str,
        dir_path: str,
        author: str,
        license_name: str,
        license_url: str,
        attribution_required: bool,
        commercial_allowed: bool,
        derivatives_allowed: bool,
        share_alike: bool,
        user_confirmed: bool,
        run_name: str,
        max_palette_slots: float,
        quantize_overcolor: bool,
        infer_role_map: bool,
        canonicalize_palette: bool,
        allow_resize: bool,
        center_pad: bool,
        slice_sheets: bool,
        tile_size: float,
    ) -> str:
        if not run_name.strip():
            return "Run name is required."
        source = SourceRecord(
            source_id=source_id or source_name,
            source_name=source_name,
            source_type=source_type,
            source_url=source_url,
            download_url=download_url,
            local_archive_path=zip_path,
            local_root_path=dir_path,
            author=author,
            license=SourceLicense(
                license=license_name,
                license_url=license_url,
                attribution_required=attribution_required,
                commercial_allowed=commercial_allowed,
                derivatives_allowed=derivatives_allowed,
                share_alike=share_alike,
                user_confirmed=user_confirmed,
            ),
        )
        run_dir = Path(run_root) / run_name.strip()
        run_dir.mkdir(parents=True, exist_ok=True)
        options = HarvestImportOptions(
            max_palette_slots=int(max_palette_slots),
            quantize_overcolor=quantize_overcolor,
            allow_nearest_resize=allow_resize,
            allow_center_pad_to_32=center_pad,
            infer_role_map=infer_role_map,
            canonicalize_palette=canonicalize_palette,
            slice_sheets=slice_sheets,
            sheet_config=SheetSliceConfig(tile_width=int(tile_size), tile_height=int(tile_size)),
        )
        try:
            harvested = harvest_source_to_imported_sprites(source, options=options, work_dir=run_dir)
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            return f"Import failed: {exc}"
        state["sources"].append(source)
        state["harvested"].extend(harvested)
        state["run_dir"] = run_dir
        _persist(state)
        valid = sum(1 for sprite in harvested if sprite.imported.bundle is not None)
        return (
            f"Sources: {len(state['sources'])}\n"
            f"Imported this source: {len(harvested)}\n"
            f"Valid: {valid}\n"
            f"Invalid: {len(harvested) - valid}\n"
            f"Total items: {len(state['harvested'])}"
        )

    def on_autolabel() -> str:
        updated = []
        for sprite in state["harvested"]:
            suggestion = suggest_metadata_from_path(sprite.candidate.relative_path, sprite.source.source_name)
            item = merge_auto_labels(sprite.final_item, [suggestion])
            updated.append(replace(sprite, final_item=item, imported=replace(sprite.imported, item=item)))
        state["harvested"] = updated
        _persist(state)
        return f"Rule-based auto-label applied to {len(updated)} items."

    def on_qwen(
        backend: str,
        base_url: str,
        model: str,
        api_key: str,
        runpod_token: str,
        cache_dir: str,
        max_items: float | None,
        workers: float | None,
    ) -> str:
        effective_base_url = base_url
        if backend == "ollama" and effective_base_url.rstrip("/") == "http://127.0.0.1:8000/v1":
            effective_base_url = "http://127.0.0.1:11434"
        config = QwenBatchPrefillConfig(
            enabled=True,
            backend=backend,
            model=model,
            base_url=effective_base_url,
            api_key=api_key or "not-needed",
            runpod_token=runpod_token,
            cache_dir=Path(cache_dir or ".prefill_cache"),
            max_items=int(max_items) if max_items else None,
            workers=max(1, int(workers or 1)),
        )
        state["harvested"] = batch_prefill_with_qwen(state["harvested"], config)
        _persist(state)
        suggested = sum(1 for s in state["harvested"] if "qwen_suggestion" in s.auto_metadata)
        failed = sum(1 for s in state["harvested"] if "qwen_error" in s.auto_metadata)
        return f"Qwen prefilled: {suggested}, failed: {failed}, workers: {config.workers}."

    def on_browse(status: str, source_filter: str, license_filter: str, category: str, tag: str, search: str, page: float):
        rows = _filter_rows(state["harvested"], status, source_filter, license_filter, category, tag, search)
        start = max(0, int(page)) * PAGE_SIZE
        visible = rows[start : start + PAGE_SIZE]
        table = [
            [
                sprite.final_item.sprite_id,
                sprite.final_item.status,
                sprite.final_item.category,
                " ".join(sprite.final_item.tags[:6]),
                sprite.final_item.license,
                sprite.source.source_id,
                "; ".join(sprite.imported.errors)[:80],
            ]
            for sprite in visible
        ]
        return table, f"{len(rows)} matching items, page {int(page)} ({len(visible)} shown)"

    def on_preview(sprite_id: str):
        sprite = _find(state["harvested"], sprite_id)
        if sprite is None:
            return None, "not found"
        info = {
            "sprite_id": sprite.final_item.sprite_id,
            "status": sprite.final_item.status,
            "category": sprite.final_item.category,
            "tags": list(sprite.final_item.tags),
            "license": sprite.final_item.license,
            "author": sprite.final_item.author,
            "source": sprite.source.source_name,
            "source_url": sprite.source.source_url,
            "errors": list(sprite.imported.errors),
            "warnings": list(sprite.imported.warnings),
        }
        return sprite.imported.preview_image, str(info)

    def on_bulk_status(sprite_ids: str, new_status: str) -> str:
        ids = {token.strip() for token in sprite_ids.replace("\n", ",").split(",") if token.strip()}
        changed = 0
        updated = []
        for sprite in state["harvested"]:
            if sprite.final_item.sprite_id in ids:
                item = replace_status(sprite.final_item, new_status)
                sprite = replace(sprite, final_item=item, imported=replace(sprite.imported, item=item))
                changed += 1
            updated.append(sprite)
        state["harvested"] = updated
        _persist(state)
        return f"Updated {changed} items to {new_status}."

    def on_policy(
        accept_cc0: bool,
        accept_own: bool,
        accept_allowlisted: bool,
        quarantine_unknown: bool,
        quarantine_low_conf: bool,
        threshold: float,
        reject_invalid: bool,
    ) -> str:
        policy = HarvestPolicy(
            auto_accept_valid_cc0=accept_cc0,
            auto_accept_own_work=accept_own,
            auto_accept_allowlisted=accept_allowlisted,
            quarantine_unknown_license=quarantine_unknown,
            quarantine_low_qwen_confidence=quarantine_low_conf,
            qwen_confidence_threshold=float(threshold),
            reject_invalid=reject_invalid,
        )
        state["harvested"] = apply_harvest_policy(state["harvested"], policy)
        _persist(state)
        counts = Counter(sprite.final_item.status for sprite in state["harvested"])
        return "\n".join(f"{status}: {counts.get(status, 0)}" for status in STATUS_CHOICES)

    def on_export(
        dataset_name: str,
        export_root: str,
        train: float,
        val: float,
        test: float,
        seed: float,
        overwrite: bool,
        allow_unknown: bool,
    ) -> str:
        try:
            result = export_harvested_dataset(
                state["harvested"],
                dataset_name=dataset_name,
                output_root=export_root or output_root,
                train_fraction=float(train),
                val_fraction=float(val),
                test_fraction=float(test),
                seed=int(seed),
                overwrite=overwrite,
                allow_unknown_license=allow_unknown,
            )
        except (ValueError, FileExistsError) as exc:
            return f"Export blocked: {exc}"
        lines = [
            f"Output: {result.output_dir}",
            f"Train: {result.train_count}",
            f"Val: {result.val_count}",
            f"Test: {result.test_count}",
            f"Excluded: {result.excluded_count}",
            *(f"Warning: {warning}" for warning in result.warnings),
        ]
        return "\n".join(lines)

    def on_report() -> str:
        if not state["sources"]:
            return "Nothing imported yet."
        return build_harvest_report(state["sources"], state["harvested"])

    with gr.Blocks(title="SpriteLab Harvester") as demo:
        gr.Markdown("# SpriteLab Dataset Harvester")

        with gr.Tab("1. Source setup / 2. Import"):
            source_type = gr.Dropdown(list(SOURCE_TYPES), value="manual_zip", label="Source type")
            source_name = gr.Textbox(label="Source name")
            source_id = gr.Textbox(label="Source ID")
            source_url = gr.Textbox(label="Source page URL")
            download_url = gr.Textbox(label="Direct download URL (optional)")
            zip_path = gr.Textbox(label="Local ZIP path")
            dir_path = gr.Textbox(label="Local directory path")
            author = gr.Textbox(label="Author")
            license_name = gr.Dropdown(list(KNOWN_LICENSES), value="unknown", label="License")
            license_url = gr.Textbox(label="License URL")
            attribution_required = gr.Checkbox(label="Attribution required")
            commercial_allowed = gr.Checkbox(label="Commercial allowed", value=True)
            derivatives_allowed = gr.Checkbox(label="Derivatives allowed", value=True)
            share_alike = gr.Checkbox(label="Share-alike")
            user_confirmed = gr.Checkbox(label="I confirmed this license on the source page")
            run_name = gr.Textbox(label="Run name")
            max_palette_slots = gr.Number(value=32, label="Max palette slots")
            quantize_overcolor = gr.Checkbox(label="Quantize over-color", value=True)
            infer_role_map = gr.Checkbox(label="Infer role map", value=True)
            canonicalize_palette = gr.Checkbox(label="Canonicalize palette", value=True)
            allow_resize = gr.Checkbox(label="Allow nearest resize")
            center_pad = gr.Checkbox(label="Center-pad small sprites to 32x32", value=True)
            slice_sheets = gr.Checkbox(label="Slice sprite sheets", value=True)
            tile_size = gr.Number(value=32, label="Tile size (16/32/custom)")
            import_button = gr.Button("Import source")
            import_output = gr.Textbox(label="Import result", lines=6)
            import_button.click(
                on_import,
                inputs=[
                    source_type, source_name, source_id, source_url, download_url,
                    zip_path, dir_path, author, license_name, license_url,
                    attribution_required, commercial_allowed, derivatives_allowed,
                    share_alike, user_confirmed, run_name, max_palette_slots,
                    quantize_overcolor, infer_role_map, canonicalize_palette,
                    allow_resize, center_pad, slice_sheets, tile_size,
                ],
                outputs=import_output,
            )

        with gr.Tab("3. Auto-label"):
            autolabel_button = gr.Button("Apply rule-based auto-label")
            autolabel_output = gr.Textbox(label="Result")
            autolabel_button.click(on_autolabel, outputs=autolabel_output)
            qwen_backend = gr.Dropdown(["openai_compatible", "ollama", "rule_based"], value="openai_compatible", label="Backend")
            qwen_base_url = gr.Textbox(value="http://127.0.0.1:8000/v1", label="Qwen base URL")
            qwen_model = gr.Textbox(value="Qwen/Qwen3-VL-8B-Instruct", label="Model")
            qwen_api_key = gr.Textbox(value="not-needed", label="API key", type="password")
            qwen_runpod_token = gr.Textbox(value="", label="RunPod token", type="password")
            qwen_cache = gr.Textbox(value=".prefill_cache", label="Cache dir")
            qwen_max = gr.Number(label="Max items (blank = all)", value=None)
            qwen_workers = gr.Number(label="Workers", value=1)
            qwen_button = gr.Button("Run Qwen prefill")
            qwen_output = gr.Textbox(label="Qwen result")
            qwen_button.click(
                on_qwen,
                inputs=[
                    qwen_backend,
                    qwen_base_url,
                    qwen_model,
                    qwen_api_key,
                    qwen_runpod_token,
                    qwen_cache,
                    qwen_max,
                    qwen_workers,
                ],
                outputs=qwen_output,
            )

        with gr.Tab("4. Browse/filter"):
            filter_status = gr.Dropdown(["", *STATUS_CHOICES], value="", label="Status")
            filter_source = gr.Textbox(label="Source ID")
            filter_license = gr.Textbox(label="License")
            filter_category = gr.Textbox(label="Category")
            filter_tag = gr.Textbox(label="Tag")
            filter_search = gr.Textbox(label="Search (path/id/tags)")
            page = gr.Number(value=0, label="Page")
            browse_button = gr.Button("Apply filters")
            table = gr.Dataframe(
                headers=["sprite_id", "status", "category", "tags", "license", "source", "errors"],
                label="Items",
            )
            browse_info = gr.Textbox(label="Result count")
            browse_button.click(
                on_browse,
                inputs=[filter_status, filter_source, filter_license, filter_category, filter_tag, filter_search, page],
                outputs=[table, browse_info],
            )
            preview_id = gr.Textbox(label="Sprite ID to preview")
            preview_button = gr.Button("Preview")
            preview_image = gr.Image(label="Preview", type="pil")
            preview_meta = gr.Textbox(label="Metadata", lines=8)
            preview_button.click(on_preview, inputs=preview_id, outputs=[preview_image, preview_meta])
            bulk_ids = gr.Textbox(label="Sprite IDs (comma/newline separated)", lines=4)
            bulk_status = gr.Dropdown(STATUS_CHOICES, value="accepted", label="Set status")
            bulk_button = gr.Button("Apply status to selected")
            bulk_output = gr.Textbox(label="Bulk result")
            bulk_button.click(on_bulk_status, inputs=[bulk_ids, bulk_status], outputs=bulk_output)

        with gr.Tab("5. Bulk policy"):
            accept_cc0 = gr.Checkbox(label="Auto-accept valid CC0")
            accept_own = gr.Checkbox(label="Auto-accept own_work")
            accept_allow = gr.Checkbox(label="Accept allowlisted licenses")
            quarantine_unknown = gr.Checkbox(label="Quarantine unknown license", value=True)
            quarantine_low = gr.Checkbox(label="Quarantine low Qwen confidence")
            threshold = gr.Number(value=0.3, label="Qwen confidence threshold")
            reject_invalid = gr.Checkbox(label="Reject invalid images", value=True)
            policy_button = gr.Button("Apply policy")
            policy_output = gr.Textbox(label="Status counts", lines=5)
            policy_button.click(
                on_policy,
                inputs=[accept_cc0, accept_own, accept_allow, quarantine_unknown, quarantine_low, threshold, reject_invalid],
                outputs=policy_output,
            )

        with gr.Tab("6. Export"):
            dataset_name = gr.Textbox(label="Dataset name")
            export_root = gr.Textbox(value=str(output_root), label="Output root")
            train = gr.Number(value=0.8, label="Train fraction")
            val = gr.Number(value=0.1, label="Val fraction")
            test = gr.Number(value=0.1, label="Test fraction")
            seed = gr.Number(value=1337, label="Seed")
            overwrite = gr.Checkbox(label="Overwrite")
            allow_unknown = gr.Checkbox(label="Allow unknown license (override)", value=False)
            export_button = gr.Button("Export dataset")
            export_output = gr.Textbox(label="Export result", lines=8)
            export_button.click(
                on_export,
                inputs=[dataset_name, export_root, train, val, test, seed, overwrite, allow_unknown],
                outputs=export_output,
            )

        with gr.Tab("7. Report"):
            report_button = gr.Button("Build report")
            report_output = gr.Markdown()
            report_button.click(on_report, outputs=report_output)

    demo.launch(server_name=host, server_port=port)


def replace_status(item, status: str):
    from spritelab.dataset_maker.model import DatasetMakerItem

    return DatasetMakerItem(
        sprite_id=item.sprite_id,
        source_path=item.source_path,
        status=status,
        category=item.category,
        tags=item.tags,
        notes=item.notes,
        source_name=item.source_name,
        license=item.license,
        author=item.author,
        split=item.split,
        quality_issues=item.quality_issues,
        palette_size=item.palette_size,
        has_role_map=item.has_role_map,
    )


def _filter_rows(
    harvested: list[HarvestedSprite],
    status: str,
    source_filter: str,
    license_filter: str,
    category: str,
    tag: str,
    search: str,
) -> list[HarvestedSprite]:
    rows = harvested
    if status:
        rows = [s for s in rows if s.final_item.status == status]
    if source_filter.strip():
        rows = [s for s in rows if s.source.source_id == source_filter.strip()]
    if license_filter.strip():
        rows = [s for s in rows if s.final_item.license == license_filter.strip()]
    if category.strip():
        rows = [s for s in rows if s.final_item.category == category.strip()]
    if tag.strip():
        token = normalize_tag(tag)
        rows = [s for s in rows if token in s.final_item.tags]
    if search.strip():
        needle = search.strip().lower()
        rows = [
            s
            for s in rows
            if needle in s.final_item.sprite_id
            or needle in s.candidate.relative_path.lower()
            or any(needle in t for t in s.final_item.tags)
        ]
    return rows


def _find(harvested: list[HarvestedSprite], sprite_id: str) -> HarvestedSprite | None:
    wanted = sprite_id.strip()
    for sprite in harvested:
        if sprite.final_item.sprite_id == wanted:
            return sprite
    return None


def _persist(state: dict[str, Any]) -> None:
    run_dir = state.get("run_dir")
    if run_dir is None:
        return
    write_sources_jsonl(run_dir, state["sources"])
    candidates = []
    seen: set[str] = set()
    for sprite in state["harvested"]:
        if sprite.candidate.candidate_id not in seen:
            seen.add(sprite.candidate.candidate_id)
            candidates.append(sprite.candidate)
    write_candidates_jsonl(run_dir, candidates)
    write_imported_jsonl(run_dir, state["harvested"])
    write_harvest_reports(run_dir, state["sources"], state["harvested"])
