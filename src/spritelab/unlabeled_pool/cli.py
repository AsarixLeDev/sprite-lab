"""CLI for immutable unlabeled candidate pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spritelab.unlabeled_pool.builder import PoolConfig, build_pool, verify_pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spritelab.unlabeled_pool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build and freeze an unlabeled pool.")
    build.add_argument("--harvest-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--reports", type=Path)
    build.add_argument("--pool-name", default="sprite_lab_unlabeled_pool_v1")
    build.add_argument(
        "--provenance-repair",
        type=Path,
        action="append",
        default=[],
        help="Explicit append-only provenance repair artifact (repeatable).",
    )
    verify = subparsers.add_parser("verify", help="Verify a frozen pool.")
    verify.add_argument("--pool", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_pool(
            harvest_root=args.harvest_root,
            output_dir=args.output,
            reports_dir=args.reports,
            config=PoolConfig(pool_name=args.pool_name),
            provenance_repairs=args.provenance_repair,
        )
    else:
        result = verify_pool(args.pool)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
