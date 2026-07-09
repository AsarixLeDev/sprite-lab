"""Monitoring and comparison commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    _register_monitor_training(subparsers)
    _register_compare_generator_families(subparsers)


def _register_monitor_training(subparsers: argparse._SubParsersAction) -> None:
    monitor = subparsers.add_parser(
        "monitor-training",
        help="Live dashboard that tails train_metrics.jsonl for one or many runs.",
    )
    monitor.add_argument("--dir", required=True, type=Path, dest="root", help="Run dir or experiment dir to watch.")
    monitor.add_argument("--interval", type=float, default=2.0, help="Seconds between refreshes.")
    monitor.add_argument(
        "--html", type=Path, default=None, dest="html_path", help="Also write an auto-refreshing HTML dashboard."
    )
    monitor.add_argument("--once", action="store_true", help="Render a single snapshot and exit.")
    monitor.add_argument("--no-rich", action="store_true", help="Force plain-text output.")
    monitor.set_defaults(func=_run_monitor_training)


def _run_monitor_training(parsed: argparse.Namespace) -> None:
    from spritelab.training.live_monitor import run_live_monitor

    summary = run_live_monitor(
        parsed.root,
        interval=parsed.interval,
        html_path=parsed.html_path,
        once=parsed.once,
        use_rich=not parsed.no_rich,
    )
    if parsed.html_path is not None:
        print(f"HTML dashboard written to {parsed.html_path}")
    print(
        f"Runs: {summary['runs']} | done {summary['done']} | "
        f"running {summary['running']} | overall {summary['fraction'] * 100:.1f}%"
    )


def _register_compare_generator_families(subparsers: argparse._SubParsersAction) -> None:
    compare_families = subparsers.add_parser(
        "compare-generator-families",
        help="Compare regression baseline and challenger generated outputs.",
    )
    compare_families.add_argument("--baseline-run", required=True, type=Path)
    compare_families.add_argument("--baseline-generated", required=True, type=Path)
    compare_families.add_argument("--challenger-run", required=True, type=Path)
    compare_families.add_argument("--challenger-generated", required=True, type=Path)
    compare_families.add_argument("--dataset", required=True, type=Path)
    compare_families.add_argument("--prompts", required=True, type=Path)
    compare_families.add_argument("--out", required=True, type=Path, dest="out_dir")
    compare_families.set_defaults(func=_run_compare_generator_families)


def _run_compare_generator_families(parsed: argparse.Namespace) -> None:
    from spritelab.training.cli._args import _parsed_config_kwargs
    from spritelab.training.compare_generator_families import (
        CompareGeneratorFamiliesConfig,
        compare_generator_families,
    )

    report = compare_generator_families(CompareGeneratorFamiliesConfig(**_parsed_config_kwargs(parsed)))
    print(f"Recommendation: {report['recommendation']}")
    print(f"Outputs written to {parsed.out_dir}")
