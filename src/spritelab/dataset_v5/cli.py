"""Command line interface for immutable dataset-v5 assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spritelab.dataset_v5.builder import BuilderConfig, build_dataset, verify_dataset
from spritelab.dataset_v5.policy_v2 import (
    PolicyV2Config,
    build_policy_preview,
    compare_policy_previews,
    verify_policy_preview,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spritelab.dataset_v5.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--v4", required=True)
    build.add_argument("--harvest-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--explicit-manifest", action="append", default=[])
    verify = sub.add_parser("verify")
    verify.add_argument("--dataset", required=True)
    policy_build = sub.add_parser("build-policy-preview")
    policy_build.add_argument("--config", required=True)
    policy_build.add_argument("--v4", required=True)
    policy_build.add_argument("--harvest-root", required=True)
    policy_build.add_argument("--output", required=True)
    policy_build.add_argument("--explicit-manifest", action="append", default=[])
    policy_verify = sub.add_parser("verify-policy-preview")
    policy_verify.add_argument("--dataset", required=True)
    compare = sub.add_parser("compare-policy-previews")
    compare.add_argument("--dataset", action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_dataset(
            v4_dir=args.v4,
            harvest_root=args.harvest_root,
            output_dir=args.output,
            config=BuilderConfig.from_json(args.config),
            explicit_manifests=tuple(args.explicit_manifest),
        )
    elif args.command == "verify":
        result = verify_dataset(Path(args.dataset))
    elif args.command == "build-policy-preview":
        result = build_policy_preview(
            v4_dir=args.v4,
            harvest_root=args.harvest_root,
            output_dir=args.output,
            config=PolicyV2Config.from_json(args.config),
            explicit_manifests=tuple(args.explicit_manifest),
        )
    elif args.command == "verify-policy-preview":
        result = verify_policy_preview(Path(args.dataset))
    else:
        result = compare_policy_previews(args.dataset)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
