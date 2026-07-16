"""Re-export for v3 CLI commands from label_v3 package."""

from __future__ import annotations

import argparse

from spritelab.harvest.label_v3.label_v3_cli import register as _register


def register(subparsers: argparse._SubParsersAction) -> None:
    _register(subparsers)
