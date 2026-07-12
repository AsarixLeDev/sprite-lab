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

VIEW_NAMES = (
    "v5_debug",
    "v5_architecture",
    "v5_scale_check",
    "v5_eval_balanced",
    "v5_source_ood",
    "v5_open_set",
    "v5_unlabeled",
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
    policy_build.add_argument(
        "--legacy-policy-config",
        action="store_true",
        help="explicitly resolve a legacy policy config using documented historical defaults",
    )
    policy_verify = sub.add_parser("verify-policy-preview")
    policy_verify.add_argument("--dataset", required=True)
    compare = sub.add_parser("compare-policy-previews")
    compare.add_argument("--dataset", action="append", required=True)

    contract_validate = sub.add_parser("validate-contract", help="Validate the Dataset-v5 named-view contract.")
    contract_validate.add_argument("--contract-root", type=Path, required=True)
    contract_validate.add_argument("--output", type=Path)

    view_build = sub.add_parser("build-view", help="Build a deterministic non-production Dataset-v5 named view.")
    view_build.add_argument("--contract-root", type=Path, required=True)
    view_build.add_argument("--view", choices=VIEW_NAMES, required=True)
    view_build.add_argument("--policy", type=Path, required=True)
    view_build.add_argument("--source-manifest", type=Path, action="append", required=True)
    view_build.add_argument("--output", type=Path, required=True)

    view_verify = sub.add_parser("verify-view", help="Verify a Dataset-v5 named view without modifying it.")
    view_verify.add_argument("--contract-root", type=Path, required=True)
    view_verify.add_argument("--view-root", type=Path, required=True)
    view_verify.add_argument("--output", type=Path)

    view_freeze = sub.add_parser("freeze-view", help="Freeze a complete, verified Dataset-v5 named view.")
    view_freeze.add_argument("--contract-root", type=Path, required=True)
    view_freeze.add_argument("--view-root", type=Path, required=True)
    view_freeze.add_argument("--approved-decisions", type=Path, required=True)
    view_freeze.add_argument("--command-line", required=True)

    freeze_verify = sub.add_parser("verify-freeze", help="Verify a frozen Dataset-v5 named view.")
    freeze_verify.add_argument("--view-root", type=Path, required=True)
    freeze_verify.add_argument("--output", type=Path)

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
            config=PolicyV2Config.from_json(args.config, legacy=args.legacy_policy_config),
            explicit_manifests=tuple(args.explicit_manifest),
        )
    elif args.command == "verify-policy-preview":
        result = verify_policy_preview(Path(args.dataset))
    elif args.command == "compare-policy-previews":
        result = compare_policy_previews(args.dataset)
    else:
        from spritelab.dataset_v5.named_views import (
            DatasetV5ViewError,
            build_view,
            freeze_view,
            validate_contract,
            verify_freeze,
            verify_view,
            write_report,
        )

        try:
            report_output = getattr(args, "output", None)
            if report_output is not None and report_output.exists():
                raise DatasetV5ViewError(
                    f"report output already exists: {report_output}",
                    exit_code=20 if args.command == "build-view" else 2,
                    reason_code=("existing_output_root" if args.command == "build-view" else "validation_failed"),
                )
            if args.command == "validate-contract":
                result = validate_contract(args.contract_root, output_path=args.output)
            elif args.command == "build-view":
                result = build_view(
                    args.contract_root,
                    args.view,
                    args.policy,
                    tuple(args.source_manifest),
                    args.output,
                )
            elif args.command == "verify-view":
                result = verify_view(args.contract_root, args.view_root)
                if args.output is not None:
                    write_report(args.output, result)
            elif args.command == "freeze-view":
                result = freeze_view(
                    args.contract_root,
                    args.view_root,
                    args.approved_decisions,
                    args.command_line,
                )
            else:
                result = verify_freeze(args.view_root)
                if args.output is not None:
                    write_report(args.output, result)
        except DatasetV5ViewError as exc:
            exit_code = 2 if args.command in {"validate-contract", "verify-view", "verify-freeze"} else exc.exit_code
            failure = {
                "error": str(exc),
                "exit_code": exit_code,
                "ok": False,
                "reason_code": exc.reason_code,
            }
            print(json.dumps(failure, indent=2, sort_keys=True))
            return exit_code
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
