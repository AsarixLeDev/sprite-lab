"""Command line interface for annotation scheduling and GUI-facing queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spritelab.annotation_scheduler.scheduler import ScheduleConfig, ScheduleView, build_schedule, mark_issued


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--completed-ids", type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spritelab.annotation_scheduler")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build or resume a deterministic schedule.")
    build.add_argument("--pool", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--completed-ids", type=Path)
    build.add_argument("--batch-size", type=int, default=50)
    build.add_argument("--max-shade-share", type=float, default=0.30)
    build.add_argument("--max-pack-share", type=float, default=0.30)
    build.add_argument("--max-artist-share", type=float, default=0.30)
    build.add_argument("--min-broad-types", type=int, default=6)
    build.add_argument("--min-source-packs", type=int, default=5)
    build.add_argument("--strategy", choices=("current_priority", "balanced", "shade_capped"), default="shade_capped")
    export = commands.add_parser("export", help="Export a whole batch or deterministic prefix.")
    _add_paths(export)
    export.add_argument("--batch", type=int, required=True)
    export.add_argument("--limit", type=int)
    export.add_argument("--output", type=Path, required=True)
    cohort = commands.add_parser("export-cohort", help="Export a deterministic balanced cohort from a batch.")
    _add_paths(cohort)
    cohort.add_argument("--batch", type=int, required=True)
    cohort.add_argument("--size", type=int, required=True)
    cohort.add_argument(
        "--mode",
        choices=("semantic_accept_only", "quality_quarantine", "mixed_diagnostic"),
        default="semantic_accept_only",
    )
    cohort.add_argument("--output", type=Path, required=True)
    cohort.add_argument("--manifest", type=Path)
    query = commands.add_parser("query", help="Read GUI integration values.")
    _add_paths(query)
    query.add_argument(
        "what",
        choices=("next-batch", "specific-batch", "remaining-batches", "completed-count", "propagated-variant-count"),
    )
    query.add_argument("--batch", type=int)
    issue = commands.add_parser("issue", help="Mark a batch as immutable/issued.")
    issue.add_argument("--schedule", type=Path, required=True)
    issue.add_argument("--batch", type=int, required=True)
    args = parser.parse_args(argv)

    if args.command == "build":
        result = build_schedule(
            args.pool,
            args.output,
            completed_ids_path=args.completed_ids,
            config=ScheduleConfig(
                batch_size=args.batch_size,
                max_shade_share=args.max_shade_share,
                max_single_pack_share=args.max_pack_share,
                max_single_artist_share=args.max_artist_share,
                min_broad_types=args.min_broad_types,
                min_source_packs=args.min_source_packs,
                strategy=args.strategy,
            ),
        )
    elif args.command == "issue":
        result = mark_issued(args.schedule, args.batch)
    else:
        view = ScheduleView(args.schedule, args.completed_ids)
        if args.command == "export-cohort":
            rows, manifest = view.export_cohort(args.batch, args.size, mode=args.mode)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = {
                "batch": args.batch,
                "records": len(rows),
                "mode": args.mode,
                "output": str(args.output),
                "manifest": str(manifest_path),
            }
        elif args.command == "export":
            rows = view.specific_batch(args.batch)
            if args.limit is not None:
                if args.limit < 1:
                    parser.error("--limit must be positive")
                if args.limit < len(rows):
                    parser.error("partial batch export is unsafe; use export-cohort --size instead")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8"
            )
            result = {"batch": args.batch, "records": len(rows), "output": str(args.output)}
        elif args.what == "next-batch":
            result = view.next_batch()
        elif args.what == "specific-batch":
            if args.batch is None:
                parser.error("specific-batch requires --batch")
            result = view.specific_batch(args.batch)
        elif args.what == "remaining-batches":
            result = view.remaining_batches()
        elif args.what == "completed-count":
            result = view.completed_count()
        else:
            result = view.propagated_variant_count()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
