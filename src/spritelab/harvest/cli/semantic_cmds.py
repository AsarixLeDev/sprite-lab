"""Semantic commands: semantic-v3, semantic-v3-report, prefill-eval-v2."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.harvest.cli._args import _parse_runs_arg, _resolve_in_run


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_semantic_v3(subparsers)
    _register_semantic_v3_report(subparsers)
    _register_prefill_eval_v2(subparsers)


def _register_semantic_v3(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("semantic-v3", help="Add semantic-v3 compositional metadata to label-v2 predictions.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.add_argument(
        "--out", type=Path, help="Defaults to <prediction-file stem>_semantic_v3.jsonl in the run directory."
    )
    p.add_argument("--max-captions", type=int, default=8)
    p.set_defaults(func=_run_semantic_v3)


def _run_semantic_v3(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.catalog import append_harvest_event, read_jsonl, write_jsonl
    from spritelab.harvest.semantic_v3 import convert_label_v2_predictions, summarize_semantic_v3_records

    run_dir = Path(parsed.run)
    prediction_path = _resolve_in_run(run_dir, parsed.prediction_file)
    if not prediction_path.exists():
        raise SystemExit(f"prediction file not found: {prediction_path}")
    if parsed.out is not None:
        output_path = _resolve_in_run(run_dir, parsed.out)
    else:
        output_path = prediction_path.with_name(f"{prediction_path.stem}_semantic_v3.jsonl")

    predictions = read_jsonl(prediction_path)
    converted = convert_label_v2_predictions(predictions, max_captions=max(1, int(parsed.max_captions)))
    write_jsonl(output_path, converted)
    summary = summarize_semantic_v3_records(converted)
    append_harvest_event(
        run_dir,
        "semantic_v3",
        {
            "prediction_file": prediction_path.name,
            "out": output_path.name,
            "records": summary["records"],
            "records_with_semantic_v3": summary["records_with_semantic_v3"],
        },
    )
    print(f"Wrote: {output_path}")
    print(f"Records: {summary['records']}")
    print(f"Records with semantic_v3: {summary['records_with_semantic_v3']}")
    print(f"Average captions: {summary['average_captions']:.2f}")
    print(f"Base object coverage: {summary['base_object_coverage']:.3f}")
    for warning, count in dict(summary.get("warnings") or {}).items():
        print(f"warning {warning}: {count}")


def _register_semantic_v3_report(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("semantic-v3-report", help="Summarize semantic-v3 coverage for a prediction file.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--prediction-file", default="label_v2_suggestions_semantic_v3.jsonl")
    p.add_argument("--out-json", type=Path)
    p.add_argument("--out-md", type=Path)
    p.set_defaults(func=_run_semantic_v3_report)


def _run_semantic_v3_report(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.catalog import read_jsonl
    from spritelab.harvest.semantic_v3 import format_semantic_v3_report, summarize_semantic_v3_records

    run_dir = Path(parsed.run)
    prediction_path = _resolve_in_run(run_dir, parsed.prediction_file)
    if not prediction_path.exists():
        raise SystemExit(f"prediction file not found: {prediction_path}")
    records = read_jsonl(prediction_path)
    summary = summarize_semantic_v3_records(records)
    report = format_semantic_v3_report(summary)
    if parsed.out_json is not None:
        out_json = _resolve_in_run(run_dir, parsed.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote: {out_json}")
    if parsed.out_md is not None:
        out_md = _resolve_in_run(run_dir, parsed.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report, encoding="utf-8")
        print(f"Wrote: {out_md}")
    print(report, end="")


def _register_prefill_eval_v2(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("prefill-eval-v2", help="Evaluate label-v2 safe prefill against cross-run golden labels.")
    p.add_argument("--golden", required=True, type=Path)
    p.add_argument("--runs", required=True, help="Comma-separated harvest run directories.")
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--errors-out", type=Path)
    p.add_argument("--errors-mode", default="all", choices=["all", "hard", "object", "tag"])
    p.set_defaults(func=_run_prefill_eval_v2)


def _run_prefill_eval_v2(parsed: argparse.Namespace) -> None:
    import json

    from spritelab.harvest.golden import load_golden_labels
    from spritelab.harvest.label_v2_eval import (
        evaluate_label_v2,
        format_label_v2_report,
        label_v2_error_records,
        load_label_v2_predictions,
    )

    runs = _parse_runs_arg(parsed.runs)
    golden = load_golden_labels(parsed.golden)
    records = load_label_v2_predictions(runs, prediction_file=parsed.prediction_file)
    result = evaluate_label_v2(golden, records)
    result["golden_path"] = str(parsed.golden)
    result["runs"] = [str(run) for run in runs]
    result["prediction_file"] = parsed.prediction_file
    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if parsed.errors_out:
        from spritelab.harvest.catalog import write_jsonl

        errors = label_v2_error_records(golden, records, errors_mode=parsed.errors_mode)
        parsed.errors_out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(parsed.errors_out, errors)
    print(format_label_v2_report(result))
    print(f"\nWrote: {parsed.out}")
    if parsed.errors_out:
        print(f"Wrote errors: {parsed.errors_out}")
