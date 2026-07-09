"""Golden label commands: golden-lint, golden-sample, golden-label, golden-prefill-v2, golden-prefill-report, assisted-golden-sample, assisted-golden."""

from __future__ import annotations

import argparse
from pathlib import Path

from spritelab.harvest.cli._args import _parse_include_statuses


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_golden_lint(subparsers)
    _register_golden_sample(subparsers)
    _register_golden_label(subparsers)
    _register_golden_prefill_v2(subparsers)
    _register_golden_prefill_report(subparsers)
    _register_assisted_golden_sample(subparsers)
    _register_assisted_golden(subparsers)


def _register_golden_lint(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("golden-lint", help="Report likely golden label inconsistencies.")
    p.add_argument("--golden", required=True, type=Path)
    p.add_argument("--fix", action="store_true")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=_run_golden_lint)


def _run_golden_lint(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.golden_lint import format_golden_lint_report, lint_golden_file, write_jsonl

    issues, suggestions = lint_golden_file(parsed.golden, fix=bool(parsed.fix))
    print(format_golden_lint_report(issues), end="")
    if parsed.fix:
        if parsed.out is None:
            raise SystemExit("--fix requires --out so the golden labels are not mutated in place.")
        write_jsonl(parsed.out, suggestions)
        print(f"Wrote suggestions: {parsed.out}")


def _register_golden_sample(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("golden-sample", help="Sample sprites for the golden evaluation set.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratify-by", default="source_name", help="Comma-separated record fields to stratify on.")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=_run_golden_sample)


def _run_golden_sample(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.catalog import append_harvest_event, read_jsonl, write_jsonl
    from spritelab.harvest.golden import sample_golden_candidates

    run_dir = Path(parsed.run)
    records = read_jsonl(run_dir / "imported.jsonl")
    stratify_by = tuple(field.strip() for field in str(parsed.stratify_by).split(",") if field.strip())
    sample = sample_golden_candidates(records, parsed.n, stratify_by=stratify_by or ("source_name",), seed=parsed.seed)
    output_path = parsed.out or (run_dir / "golden_sample.jsonl")
    write_jsonl(output_path, sample)
    append_harvest_event(run_dir, "golden_sample", {"count": len(sample), "seed": parsed.seed})
    print(f"Wrote: {output_path}")
    print(f"Sampled: {len(sample)} of {len(records)}")


def _register_golden_label(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("golden-label", help="Launch the golden-set labeling GUI.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int)
    p.add_argument("--labeler", default="")
    p.set_defaults(func=_run_golden_label)


def _run_golden_label(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.prefill_review_gui import launch_golden_label_gui

    try:
        launch_golden_label_gui(parsed.run, host=parsed.host, port=parsed.port, labeler=parsed.labeler)
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc


def _register_golden_prefill_v2(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "golden-prefill-v2", help="Create assisted golden candidates prefilled from label-v2 suggestions."
    )
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--prediction-file", default="label_v2_suggestions.jsonl")
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--seed", type=int, default=496)
    p.add_argument("--stratify-by", default="source_profile.name,bucket,safe_prefill.object_name")
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Initialize gold_* fields from label-v2 even when a human golden label already exists.",
    )
    p.set_defaults(func=_run_golden_prefill_v2)


def _run_golden_prefill_v2(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden import build_label_v2_prefilled_candidates, write_golden_candidates_jsonl
    from spritelab.harvest.catalog import append_harvest_event

    run_dir = Path(parsed.run)
    stratify_by = tuple(field.strip() for field in str(parsed.stratify_by).split(",") if field.strip())
    candidates = build_label_v2_prefilled_candidates(
        run_dir,
        prediction_file=parsed.prediction_file,
        n=parsed.n,
        seed=parsed.seed,
        stratify_by=stratify_by or ("source_profile.name", "bucket", "safe_prefill.object_name"),
        overwrite=bool(parsed.overwrite),
    )
    output_path = parsed.out or (run_dir / "golden_candidates_prefilled.jsonl")
    write_golden_candidates_jsonl(output_path, candidates)
    append_harvest_event(
        run_dir,
        "golden_prefill_v2",
        {
            "count": len(candidates),
            "seed": parsed.seed,
            "prediction_file": str(parsed.prediction_file),
            "stratify_by": list(stratify_by),
            "overwrite": bool(parsed.overwrite),
        },
    )
    print(f"Wrote: {output_path}")
    print(f"Candidates: {len(candidates)}")


def _register_golden_prefill_report(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("golden-prefill-report", help="Summarize correction rates for prefilled golden labels.")
    p.add_argument("--golden", required=True, type=Path)
    p.set_defaults(func=_run_golden_prefill_report)


def _run_golden_prefill_report(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden import format_golden_prefill_report, summarize_golden_prefill_records
    from spritelab.harvest.catalog import read_jsonl

    records = read_jsonl(parsed.golden)
    print(format_golden_prefill_report(summarize_golden_prefill_records(records)), end="")


def _register_assisted_golden_sample(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("assisted-golden-sample", help="Write assisted golden correction candidates.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--include-status", action="append")
    p.set_defaults(func=_run_assisted_golden_sample)


def _run_assisted_golden_sample(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden import (
        GOLDEN_CANDIDATES_FILENAME,
        load_assisted_candidates,
        write_golden_candidates_jsonl,
    )
    from spritelab.harvest.catalog import append_harvest_event

    run_dir = Path(parsed.run)
    include_statuses = _parse_include_statuses(parsed.include_status)
    candidates = load_assisted_candidates(
        run_dir,
        n=parsed.n,
        seed=parsed.seed,
        include_statuses=include_statuses,
    )
    output_path = run_dir / GOLDEN_CANDIDATES_FILENAME
    write_golden_candidates_jsonl(output_path, candidates)
    append_harvest_event(
        run_dir,
        "assisted_golden_sample",
        {"count": len(candidates), "seed": parsed.seed, "include_statuses": list(include_statuses)},
    )
    print(f"Wrote: {output_path}")
    print(f"Candidates: {len(candidates)}")


def _register_assisted_golden(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("assisted-golden", help="Launch assisted golden correction GUI.")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int)
    p.add_argument("--labeler", default="mathieu")
    p.add_argument("--include-status", action="append")
    p.add_argument(
        "--order",
        default="needs_review_first",
        choices=["needs_review_first", "random", "source_order", "uncertain_first", "unlabeled_first"],
    )
    p.set_defaults(func=_run_assisted_golden)


def _run_assisted_golden(parsed: argparse.Namespace) -> None:
    from spritelab.harvest.assisted_golden_gui import launch_assisted_golden_gui

    include_statuses = _parse_include_statuses(parsed.include_status)
    try:
        launch_assisted_golden_gui(
            parsed.run,
            n=parsed.n,
            seed=parsed.seed,
            host=parsed.host,
            port=parsed.port,
            labeler=parsed.labeler,
            include_statuses=include_statuses,
            order=parsed.order,
        )
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc
