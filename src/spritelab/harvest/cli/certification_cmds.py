"""Harvest capability-certificate operator commands."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    certificate = subparsers.add_parser("certificate", help="Inspect or refresh Harvest capability evidence.")
    actions = certificate.add_subparsers(dest="certificate_action", required=True)
    refresh = actions.add_parser("refresh", help="Refresh the fixed repository certificate and report.")
    refresh.add_argument("--project-root", type=Path, default=Path.cwd())
    refresh.add_argument("--valid-days", type=int, default=30)
    refresh.add_argument("--rebind-current-implementation", action="store_true")
    refresh.add_argument("--confirm-carry-forward-pass", action="store_true")
    refresh.set_defaults(func=_refresh)


def _refresh(parsed: argparse.Namespace) -> None:
    from spritelab.product_features.harvest.certificate_refresh import refresh_harvest_certificate

    try:
        result = refresh_harvest_certificate(
            parsed.project_root,
            rebind_current_implementation=parsed.rebind_current_implementation,
            confirm_carry_forward_pass=parsed.confirm_carry_forward_pass,
            validity_days=parsed.valid_days,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Harvest certificate refresh refused: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(asdict(result), sort_keys=True))
    if result.restart_required:
        print("Restart Sprite Lab before starting a Harvest probe or acquisition.")


__all__ = ["register"]
