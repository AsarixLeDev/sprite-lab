"""CLI for semantic training baselines — entry point and dispatcher."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m spritelab train",
        description="Semantic-manifest training inspection, baseline training, generator training, and evaluation.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    from spritelab.training.cli import (
        audit_cmds,
        challenger_cmds,
        data_cmds,
        eval_cmds,
        gallery_cmds,
        monitor_cmds,
        review_cmds,
    )

    data_cmds.register(subparsers)
    challenger_cmds.register(subparsers)
    review_cmds.register(subparsers)
    audit_cmds.register(subparsers)
    gallery_cmds.register(subparsers)
    eval_cmds.register(subparsers)
    monitor_cmds.register(subparsers)

    from spritelab.training.cli._args import _apply_export_preset_defaults

    parsed = parser.parse_args(raw_argv)
    _apply_export_preset_defaults(parsed, raw_argv)
    try:
        parsed.func(parsed)
    except RuntimeError as exc:
        if "PyTorch is required" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc
