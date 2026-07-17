"""Import/ingest commands: import-zip, import-dir, download-zip, import-diagnostics, gui."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spritelab.harvest.cli._args import (
    _add_import_args,
    _add_source_args,
    _build_source,
    _unique_candidates,
)
from spritelab.harvest.source_prefill import build_source_prefill, source_preset_ids


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_source_prefill(subparsers)
    _register_gui(subparsers)
    _register_import_zip(subparsers)
    _register_import_dir(subparsers)
    _register_download_zip(subparsers)
    _register_download_file(subparsers)
    _register_import_diagnostics(subparsers)


def _register_source_prefill(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "source-prefill",
        aliases=("smart-prefill",),
        help="Turn one public pack-page URL into reviewable Harvest source defaults.",
    )
    parser.add_argument("source_page", help="Public HTTPS creator or pack page.")
    parser.add_argument("--preset", default="auto", choices=source_preset_ids())
    parser.add_argument("--format", default="human", choices=("human", "json"))
    parser.set_defaults(func=_run_source_prefill)


def _run_source_prefill(parsed: argparse.Namespace) -> None:
    try:
        prefill = build_source_prefill(parsed.source_page, preset_id=parsed.preset)
    except ValueError as exc:
        raise SystemExit(f"Source prefill failed: {exc}") from exc
    payload = prefill.to_dict()
    if parsed.format == "json":
        print(json.dumps(payload, sort_keys=True, indent=2))
        return
    print(f"Preset: {prefill.preset_label}")
    print(f"Source ID: {prefill.source_id}")
    print(f"Title: {prefill.title}")
    print(f"Creator: {prefill.creator or '[review required]'}")
    print(f"License: {prefill.license_name if prefill.license_name != 'unknown' else '[review required]'}")
    print(f"License evidence: {prefill.license_evidence_url or '[review required]'}")
    print(f"Terms evidence: {prefill.terms_evidence_url or '[review source page]'}")
    print(f"Direct download: {prefill.direct_download_url or '[paste exact creator-posted link]'}")
    print(f"Review required: {', '.join(prefill.review_fields)}")
    print(prefill.guidance)


def _register_gui(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("gui", help="Launch the harvest GUI.")
    p.add_argument("--output-root", default="datasets")
    p.add_argument("--run-root", default="harvest_runs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int)
    p.set_defaults(func=_run_gui)


def _run_gui(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.gui import launch_harvest_gui

    try:
        launch_harvest_gui(
            output_root=parsed.output_root,
            run_root=parsed.run_root,
            host=parsed.host,
            port=parsed.port,
        )
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc


def _register_import_zip(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("import-zip", help="Import a manually downloaded ZIP.")
    _add_source_args(p)
    _add_import_args(p)
    p.add_argument("--zip", required=True, type=Path, dest="zip_path")
    p.set_defaults(func=_run_import_zip)


def _run_import_zip(parsed: argparse.Namespace) -> None:
    _run_import(parsed, kind="zip")


def _register_import_dir(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("import-dir", help="Import a local PNG directory.")
    _add_source_args(p)
    _add_import_args(p)
    p.add_argument("--dir", required=True, type=Path, dest="dir_path")
    p.set_defaults(func=_run_import_dir)


def _run_import_dir(parsed: argparse.Namespace) -> None:
    _run_import(parsed, kind="dir")


def _register_download_zip(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("download-zip", help="Download and import a direct ZIP URL.")
    _add_source_args(p)
    _add_import_args(p)
    p.add_argument("--url", required=True)
    p.set_defaults(func=_run_download_zip)


def _run_download_zip(parsed: argparse.Namespace) -> None:
    _run_import(parsed, kind="url")


def _register_download_file(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("download-file", help="Download and import one direct PNG attachment.")
    _add_source_args(p)
    _add_import_args(p)
    p.add_argument("--url", required=True)
    p.set_defaults(func=_run_download_file)


def _run_download_file(parsed: argparse.Namespace) -> None:
    _run_import(parsed, kind="file")


def _run_import(parsed: argparse.Namespace, *, kind: str) -> None:
    from spritelab.harvest.catalog import (
        append_harvest_event,
        write_candidates_jsonl,
        write_imported_jsonl,
        write_sources_jsonl,
    )
    from spritelab.harvest.pipeline import HarvestImportOptions, harvest_source_to_imported_sprites
    from spritelab.harvest.report import write_harvest_reports
    from spritelab.harvest.sheets import SheetSliceConfig

    source = _build_source(parsed, kind=kind)
    run_dir = Path(parsed.run_root) / parsed.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if kind == "zip" and not source.local_archive_path:
        raise SystemExit("--zip is required for import-zip")

    options = HarvestImportOptions(
        max_palette_slots=parsed.max_palette_slots,
        quantize_overcolor=parsed.quantize_overcolor,
        allow_nearest_resize=parsed.allow_nearest_resize,
        allow_center_pad_to_32=parsed.center_pad,
        infer_role_map=parsed.infer_role_map,
        canonicalize_palette=parsed.canonicalize_palette,
        slice_sheets=parsed.slice_sheets,
        sheet_config=SheetSliceConfig(tile_width=parsed.tile_size, tile_height=parsed.tile_size),
        include_member_globs=tuple(parsed.include_member_glob),
        exclude_member_globs=tuple(parsed.exclude_member_glob),
    )
    harvested = harvest_source_to_imported_sprites(source, options=options, work_dir=run_dir)
    candidates = _unique_candidates(harvested)

    persisted_source = harvested[0].source if harvested else source
    write_sources_jsonl(run_dir, [persisted_source])
    write_candidates_jsonl(run_dir, candidates)
    write_imported_jsonl(run_dir, harvested)
    write_harvest_reports(run_dir, [source], harvested)
    append_harvest_event(run_dir, "import", {"source_id": source.source_id, "count": len(harvested)})

    valid = sum(1 for sprite in harvested if sprite.imported.bundle is not None)
    print(f"Run: {run_dir}")
    print(f"Candidates: {len(candidates)}")
    print(f"Imported: {len(harvested)}")
    print(f"Valid: {valid}")
    print(f"Invalid: {len(harvested) - valid}")


def _register_import_diagnostics(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "import-diagnostics", help="Diagnose empty/import-broken harvest run state without mutating it."
    )
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", type=Path, help="Markdown report path.")
    p.add_argument("--out-json", type=Path, help="JSON report path.")
    p.set_defaults(func=_run_import_diagnostics)


def _run_import_diagnostics(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.import_diagnostics import (
        build_import_diagnostics,
        format_import_diagnostics,
        write_import_diagnostics_reports,
    )

    report = build_import_diagnostics(parsed.run)
    write_import_diagnostics_reports(report, out_md=parsed.out, out_json=parsed.out_json)
    print(format_import_diagnostics(report), end="")
    if parsed.out is not None:
        print(f"Wrote: {parsed.out}")
    if parsed.out_json is not None:
        print(f"Wrote: {parsed.out_json}")
