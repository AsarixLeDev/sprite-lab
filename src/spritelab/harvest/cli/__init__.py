"""CLI for the harvest package — entry point and dispatcher."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m spritelab harvest",
        description="License-aware dataset harvester.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Enable DEBUG-level diagnostic logging."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    from spritelab.harvest.cli import (
        dataset_cmds,
        golden_cmds,
        ingest_cmds,
        label_v2_cmds,
        prefill_cmds,
        semantic_cmds,
    )

    ingest_cmds.register(subparsers)
    prefill_cmds.register(subparsers)
    label_v2_cmds.register(subparsers)
    semantic_cmds.register(subparsers)
    golden_cmds.register(subparsers)
    dataset_cmds.register(subparsers)

    parsed = parser.parse_args(raw_argv)
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    try:
        parsed.func(parsed)
    except RuntimeError as exc:
        if "requires gradio" not in str(exc):
            raise
        print(str(exc))
        raise SystemExit(1) from exc
