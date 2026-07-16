"""Compact developer command surface for hierarchical labeling workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spritelab.hierarchical_labeling.cohort import CohortCandidate, CohortSelectionPolicy, select_reference_cohort
from spritelab.hierarchical_labeling.pilot import PilotCandidate, plan_pilot
from spritelab.hierarchical_labeling.reporting import build_report_data, write_offline_report
from spritelab.hierarchical_labeling.taxonomy import load_default_taxonomy
from spritelab.v3.model import CommandResult, ExitCode
from spritelab.v3.run_state import atomic_write_json


def register_labeling_commands(
    subparsers: argparse._SubParsersAction,
    *,
    parents: Sequence[argparse.ArgumentParser],
    environment: Any,
) -> None:
    def command(name: str, help_text: str, handler: Any) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(name, help=help_text, parents=list(parents))
        parser.set_defaults(handler=handler, developer_environment=environment, labeling_action=name)
        return parser

    cohort = command("cohort", "Select a deterministic 300-500 record human reference cohort.", _cohort)
    cohort.add_argument("--candidates", type=Path, help="JSON array of cohort candidates.")
    cohort.add_argument("--output", type=Path, default=Path("runs/v3/hierarchical-labeling/reference_cohort.json"))
    cohort.add_argument("--size", type=int, default=400)
    cohort.add_argument("--dataset-identity", default="configured-dataset")
    cohort.add_argument("--embedding-identity", default="configured-embedding")
    cohort.add_argument("--clustering-identity", default="configured-clustering")
    cohort.add_argument("--seed", type=int, default=20260715)

    review = command("review", "Open or locate the semantic review experience.", _review)
    review.add_argument("--url", default="/labeling#semantic-review")

    run = command("run", "Plan the configured automatic-labeling cascade.", _run)
    run.add_argument("--profile", choices=("fast_local", "balanced", "high_quality"), default="balanced")
    run.add_argument("--input-manifest", type=Path)

    reconcile = command("reconcile", "Plan evidence-channel reconciliation.", _reconcile)
    reconcile.add_argument("--run", type=Path)

    calibrate = command("calibrate", "Check readiness for human-truth-only calibration.", _calibrate)
    calibrate.add_argument("--examples", type=Path)
    calibrate.add_argument("--target-precision", type=float, default=0.95)

    report = command("report", "Create a truth-aware offline labeling report.", _report)
    report.add_argument("--records", type=Path)
    report.add_argument("--output", type=Path, default=Path("runs/v3/hierarchical-labeling/report"))

    pilot = command("pilot-plan", "Prepare, but do not execute, a representative 5,000-image pilot.", _pilot)
    pilot.add_argument("--candidates", type=Path)
    pilot.add_argument("--output", type=Path, default=Path("runs/v3/hierarchical-labeling/pilot_5000_plan.json"))
    pilot.add_argument("--records", type=int, default=5000)
    pilot.add_argument("--profile", choices=("fast_local", "balanced", "high_quality"), default="balanced")
    pilot.add_argument("--dataset-identity", default="configured-dataset")
    pilot.add_argument("--reference-size", type=int, default=400)
    pilot.add_argument("--maximum-hosted-calls", type=int, default=0)
    pilot.add_argument("--seed", type=int, default=20260715)


def _root(args: argparse.Namespace) -> Path:
    return args.developer_environment.load_config().root


def _resolved(args: argparse.Namespace, path: Path) -> Path:
    return path if path.is_absolute() else _root(args) / path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cohort(args: argparse.Namespace) -> CommandResult:
    if args.candidates is None:
        return _planned(
            "dev labeling cohort",
            "Reference cohort selection is ready; supply a candidate manifest to materialize it.",
            next_command=(
                "python -m spritelab dev labeling cohort --candidates <cohort-candidates.json> "
                f"--output {args.output} --size {args.size}"
            ),
            data={"recommended_size": args.size, "human_labels_created": 0},
        )
    source = _resolved(args, args.candidates)
    raw = _read_json(source)
    values = raw.get("candidates", ()) if isinstance(raw, dict) else raw
    candidates = tuple(_cohort_candidate(value) for value in values)
    manifest = select_reference_cohort(
        candidates,
        dataset_identity=args.dataset_identity,
        embedding_identity=args.embedding_identity,
        clustering_identity=args.clustering_identity,
        policy=CohortSelectionPolicy(target_size=args.size, seed=args.seed),
    )
    output = _resolved(args, args.output)
    atomic_write_json(output, manifest)
    return _planned(
        "dev labeling cohort",
        f"Selected {manifest['selected_size']} records without creating labels.",
        data={"manifest": str(output), "cohort": manifest},
    )


def _cohort_candidate(value: dict[str, Any]) -> CohortCandidate:
    fields = (
        "record_identity",
        "image_identity",
        "cluster_identity",
        "duplicate_cluster_identity",
        "near_duplicate_cluster_identity",
        "source_identity",
        "style_identity",
        "size_bucket",
        "cluster_size",
        "is_cluster_medoid",
        "novelty",
        "visual_uncertainty",
        "metadata_conflict",
        "taxonomy_confusion",
        "sheet_derived",
        "animation_frame",
        "rare_category_candidate",
        "legally_eligible",
        "technically_usable",
    )
    return CohortCandidate(*(value[name] for name in fields))


def _review(args: argparse.Namespace) -> CommandResult:
    return _planned(
        "dev labeling review",
        "Use the existing product Labeling page for append-only semantic review.",
        next_command="python -m spritelab v3 app",
        data={"path": args.url, "json_authoring_required": False},
    )


def _run(args: argparse.Namespace) -> CommandResult:
    count = None
    if args.input_manifest:
        raw = _read_json(_resolved(args, args.input_manifest))
        rows = raw.get("records", ()) if isinstance(raw, dict) else raw if isinstance(raw, list) else ()
        count = len(rows)
    return _planned(
        "dev labeling run",
        "The automatic-labeling cascade plan is ready; provider execution remains consent- and budget-gated.",
        data={
            "profile": args.profile,
            "records": count,
            "provider_models_from_configuration": True,
            "provider_calls": 0,
            "hosted_calls": 0,
            "production_authorized": False,
        },
    )


def _reconcile(args: argparse.Namespace) -> CommandResult:
    return _planned(
        "dev labeling reconcile",
        "Evidence reconciliation is ready and will preserve visual, metadata, context, retrieval, and human conflicts.",
        data={"run": str(args.run) if args.run else None, "metadata_overrides_visual": False},
    )


def _calibrate(args: argparse.Namespace) -> CommandResult:
    rows = [] if args.examples is None else _read_json(_resolved(args, args.examples))
    examples = rows.get("examples", ()) if isinstance(rows, dict) else rows if isinstance(rows, list) else ()
    count = len(examples)
    message = (
        "Calibration inputs are present; use the strict calibration API to fit and evaluate disjoint partitions."
        if count
        else "Calibration is not ready because no verified human-truth examples were supplied."
    )
    return _planned(
        "dev labeling calibrate",
        message,
        data={
            "truth_rows": count,
            "target_precision": args.target_precision,
            "calibration_state": "ready_for_experiment" if count else "not_ready",
            "model_agreement_is_truth": False,
        },
    )


def _report(args: argparse.Namespace) -> CommandResult:
    values: dict[str, Any] = {"records": [], "truth_rows": []}
    if args.records:
        raw = _read_json(_resolved(args, args.records))
        values = raw if isinstance(raw, dict) else {"records": raw, "truth_rows": []}
    report = build_report_data(
        values.get("records", ()),
        load_default_taxonomy(),
        truth_rows=values.get("truth_rows", ()),
        calibration_state=str(values.get("calibration_state", "not_ready")),
        operational=values.get("operational"),
    )
    json_path, html_path = write_offline_report(report, _resolved(args, args.output))
    return _planned(
        "dev labeling report",
        "Offline labeling report created with truth-aware no-data states.",
        data={
            "json": str(json_path),
            "html": str(html_path),
            "precision_graph": bool(report["claims"]["precision_graph_available"]),
        },
    )


def _pilot(args: argparse.Namespace) -> CommandResult:
    if args.candidates is None:
        return _planned(
            "dev labeling pilot-plan",
            "The 5,000-image pilot planner is ready; supply a candidate manifest to materialize the plan.",
            next_command=(
                "python -m spritelab dev labeling pilot-plan --candidates <pilot-candidates.json> "
                f"--output {args.output} --records {args.records} --profile {args.profile}"
            ),
            data={"pilot_runs_automatically": False, "production_authorized": False},
        )
    raw = _read_json(_resolved(args, args.candidates))
    values = raw.get("candidates", ()) if isinstance(raw, dict) else raw
    candidates = tuple(_pilot_candidate(value) for value in values)
    plan = plan_pilot(
        candidates,
        dataset_identity=args.dataset_identity,
        profile=args.profile,
        target_size=args.records,
        seed=args.seed,
        reference_cohort_size=args.reference_size,
        maximum_hosted_calls=args.maximum_hosted_calls,
    )
    output = _resolved(args, args.output)
    atomic_write_json(output, plan)
    return _planned(
        "dev labeling pilot-plan",
        f"Planned {plan['selected_records']} pilot records without running them.",
        data={"plan": str(output), "pilot": plan},
    )


def _pilot_candidate(value: dict[str, Any]) -> PilotCandidate:
    fields = (
        "record_identity",
        "image_identity",
        "source_identity",
        "pack_identity",
        "cluster_identity",
        "style_identity",
        "image_size_bucket",
        "sheet_derived",
        "technical_bucket",
        "duplicate_cluster_identity",
        "semantic_difficulty",
        "legally_eligible",
        "technically_usable",
    )
    return PilotCandidate(*(value[name] for name in fields))


def _planned(
    command: str,
    message: str,
    *,
    next_command: str | None = None,
    data: dict[str, Any] | None = None,
) -> CommandResult:
    return CommandResult(
        command=command,
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message=message,
        next_command=next_command,
        data=data or {},
        internal_details=True,
    )
