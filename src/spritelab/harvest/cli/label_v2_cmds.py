"""Label-v2 commands: label-v2, fuse-prefill-v2, label-v2-report, apply-label-v2, label-v2-sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.harvest.cli._args import (
    _add_label_v2_args,
    _parse_runs_arg,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_label_v2(subparsers)
    _register_fuse_prefill_v2(subparsers)
    _register_label_v2_report(subparsers)
    _register_apply_label_v2(subparsers)
    _register_label_v2_sweep(subparsers)


def _register_label_v2(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v2", help="Run filename/source-first safe label v2 suggestions.")
    _add_label_v2_args(p, include_vlm_args=True)
    p.set_defaults(func=_run_label_v2)


def _run_label_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v2_pipeline import (
        build_label_v2_records,
        create_vlm_backend_from_args,
        select_label_v2_input_records,
        summarize_label_v2_records,
        write_label_v2_outputs,
    )

    backend = create_vlm_backend_from_args(parsed) if parsed.use_vlm else None
    input_selection = select_label_v2_input_records(parsed.run, include_statuses=parsed.include_status)
    records = build_label_v2_records(
        parsed.run,
        use_vlm=bool(parsed.use_vlm),
        vlm_only_when_needed=bool(parsed.vlm_only_when_needed),
        max_items=parsed.max_items,
        propagate_dups=bool(parsed.propagate_dups),
        trusted_filename_threshold=parsed.trusted_filename_threshold,
        auto_vlm_threshold=parsed.auto_vlm_threshold,
        review_conflicts=bool(parsed.review_conflicts),
        backend=backend,
        refresh_vlm=bool(parsed.refresh_vlm),
        ignore_existing_vlm=bool(parsed.ignore_existing_vlm),
        workers=parsed.workers,
        include_statuses=parsed.include_status,
    )
    paths = write_label_v2_outputs(parsed.run, records, out=parsed.out, input_selection=input_selection)
    summary = summarize_label_v2_records(records)
    summary["input_selection"] = input_selection.to_summary()
    print(f"Wrote: {paths['suggestions']}")
    print(f"Suggestions: {summary['total']}")
    print(f"Needs review: {summary['needs_review_count']}")
    for reason, count in summary.get("input_selection", {}).get("skipped_by_reason", {}).items():
        print(f"{reason}: {count}")
    print(f"Workers: {max(1, int(parsed.workers or 1))}")
    for key, count in summary.get("vlm_stats", {}).items():
        print(f"{key}: {count}")
    for bucket, count in summary["buckets"].items():
        print(f"{bucket}: {count}")


def _register_fuse_prefill_v2(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("fuse-prefill-v2", help="Safely fuse existing Qwen suggestions through label v2.")
    _add_label_v2_args(p, include_vlm_args=False)
    p.set_defaults(func=_run_fuse_prefill_v2)


def _run_fuse_prefill_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.label_v2_pipeline import (
        build_label_v2_records,
        select_label_v2_input_records,
        summarize_label_v2_records,
        write_label_v2_outputs,
    )

    input_selection = select_label_v2_input_records(parsed.run, include_statuses=parsed.include_status)
    records = build_label_v2_records(
        parsed.run,
        use_vlm=True,
        vlm_only_when_needed=False,
        max_items=parsed.max_items,
        propagate_dups=bool(parsed.propagate_dups),
        trusted_filename_threshold=parsed.trusted_filename_threshold,
        auto_vlm_threshold=parsed.auto_vlm_threshold,
        review_conflicts=bool(parsed.review_conflicts),
        backend=None,
        workers=parsed.workers,
        include_statuses=parsed.include_status,
    )
    paths = write_label_v2_outputs(parsed.run, records, out=parsed.out, input_selection=input_selection)
    summary = summarize_label_v2_records(records)
    summary["input_selection"] = input_selection.to_summary()
    print(f"Wrote: {paths['suggestions']}")
    print(f"Suggestions: {summary['total']}")
    print(f"Needs review: {summary['needs_review_count']}")
    for reason, count in summary.get("input_selection", {}).get("skipped_by_reason", {}).items():
        print(f"{reason}: {count}")
    print(f"Workers: {max(1, int(parsed.workers or 1))}")
    for bucket, count in summary["buckets"].items():
        print(f"{bucket}: {count}")


def _register_label_v2_report(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v2-report", help="Print/write a label-v2 run summary.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.set_defaults(func=_run_label_v2_report)


def _run_label_v2_report(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.catalog import read_jsonl
    from spritelab.harvest.label_v2_pipeline import format_label_v2_run_report, summarize_label_v2_records

    run_dir = Path(parsed.run)
    records = read_jsonl(run_dir / parsed.prediction_file)
    summary = summarize_label_v2_records(records)
    summary_path = run_dir / "label_v2_summary.json"
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if isinstance(previous, dict) and isinstance(previous.get("input_selection"), dict):
            summary["input_selection"] = previous["input_selection"]
    report = format_label_v2_run_report(summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "label_v2_report.md").write_text(report, encoding="utf-8")
    print(report, end="")


def _register_apply_label_v2(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("apply-label-v2", help="Apply label-v2 suggestions to harvest metadata.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.add_argument("--mode", default="auto-only", choices=["auto-only", "all", "review-only"])
    accept_auto = p.add_mutually_exclusive_group()
    accept_auto.add_argument("--accept-auto", action="store_true", dest="accept_auto", default=False)
    accept_auto.add_argument("--no-accept-auto", action="store_false", dest="accept_auto")
    p.add_argument("--out-imported", type=Path)
    p.add_argument("--out-review", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite-human-labels", action="store_true")
    p.set_defaults(func=_run_apply_label_v2)


def _run_apply_label_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.apply_label_v2 import apply_label_v2_predictions, format_apply_summary

    report = apply_label_v2_predictions(
        parsed.run,
        prediction_file=parsed.prediction_file,
        mode=parsed.mode,
        accept_auto=bool(parsed.accept_auto),
        out_imported=parsed.out_imported,
        out_review=parsed.out_review,
        dry_run=bool(parsed.dry_run),
        overwrite_human_labels=bool(parsed.overwrite_human_labels),
    )
    print(format_apply_summary(report), end="")


def _register_label_v2_sweep(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("label-v2-sweep", help="Sweep label-v2 thresholds against golden labels.")
    p.add_argument("--golden", required=True, type=Path)
    p.add_argument("--runs", required=True, help="Comma-separated harvest run directories.")
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=_run_label_v2_sweep)


def _run_label_v2_sweep(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.golden import load_golden_labels
    from spritelab.harvest.label_v2_eval import load_label_v2_predictions, sweep_label_v2_operating_points

    runs = _parse_runs_arg(parsed.runs)
    golden = load_golden_labels(parsed.golden)
    records = load_label_v2_predictions(runs, prediction_file=parsed.prediction_file)
    result = sweep_label_v2_operating_points(golden, records)
    result["golden_path"] = str(parsed.golden)
    result["runs"] = [str(run) for run in runs]
    result["prediction_file"] = parsed.prediction_file
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best = result.get("best") or {}
    print("Best operating point:")
    for key in (
        "trusted_filename_threshold",
        "vlm_threshold",
        "conflict_policy",
        "auto_coverage",
        "auto_precision",
        "object_token_f1",
    ):
        if key in best:
            print(f"{key}: {best[key]}")
    print(f"\nWrote: {parsed.out}")
