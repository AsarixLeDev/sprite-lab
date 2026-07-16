"""Fail-closed v3 orchestration over existing backend commands."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from spritelab.product_core import ProductResult, ProductStatus, ProjectContext
from spritelab.product_core.audit_evidence import PRODUCTION_CONDITIONED_DATASET_FREEZE
from spritelab.product_features.dataset.certification import authorize_labeling_scope
from spritelab.product_features.training.models import TrainingProfile
from spritelab.product_features.training.service import TrainingService, backend_from_context
from spritelab.training.campaign import CampaignValidationError, is_concrete_hash
from spritelab.training.launch import validate_training_launch_plan
from spritelab.v3.config import ProjectConfig, configured_training_identities
from spritelab.v3.model import AuditStatus, CommandResult, ExitCode, ProjectState, StageStatus
from spritelab.v3.report import generate_report
from spritelab.v3.run_state import RunState
from spritelab.v3.status import build_project_state


@dataclass(frozen=True)
class ExecutionOptions:
    dry_run: bool = False
    yes: bool = False
    non_interactive_confirm: bool = False
    debug: bool = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_product_training_identity(product_result: ProductResult) -> dict[str, object]:
    if product_result.status != ProductStatus.RUNNING or product_result.run is None:
        raise CampaignValidationError("Product training did not return a running, durably identified run.")
    raw = product_result.data.get("training_identity")
    if not isinstance(raw, Mapping):
        raise CampaignValidationError("Product training did not return its validated dataset and view identity.")
    dataset_identity = raw.get("dataset_identity")
    view_identity = raw.get("view_identity")
    view_alias = raw.get("training_view_identity")
    if not is_concrete_hash(dataset_identity):
        raise CampaignValidationError("Product training returned a malformed dataset identity.")
    if not is_concrete_hash(view_identity) or view_alias != view_identity:
        raise CampaignValidationError("Product training returned a malformed or inconsistent training-view identity.")
    run_id = product_result.run.run_id
    if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
        raise CampaignValidationError("Product training returned a malformed durable run identity.")
    return {
        "dataset_identity": dataset_identity,
        "view_identity": view_identity,
        "training_view_identity": view_identity,
        "product_training_run_id": run_id,
    }


def _report_and_finish(
    run: RunState,
    project: ProjectState,
    result: CommandResult,
    *,
    stage: str,
    resumable: bool = False,
    backend_identity: dict[str, object] | None = None,
) -> CommandResult:
    report_path, _ = generate_report(project, run.directory / "report", run=result.to_dict())
    result.run_id = run.run_id
    result.report_path = str(report_path)
    run.finish(
        command=result.command,
        status=result.status,
        exit_code=int(result.exit_code),
        message=result.message,
        stage=stage,
        resumable=resumable,
        backend_identity={**(backend_identity or {}), "source_commit": project.source_commit},
    )
    return result


def _blocked(
    run: RunState,
    project: ProjectState,
    command: str,
    stage: str,
    blockers: list[str],
    *,
    review: bool = False,
    stale: bool = False,
    data: dict[str, object] | None = None,
) -> CommandResult:
    exit_code = ExitCode.REVIEW_REQUIRED if review else (ExitCode.STALE if stale else ExitCode.BLOCKED)
    status = "NEEDS_REVIEW" if review else ("STALE" if stale else "BLOCKED")
    message = (
        "Human review is required."
        if review
        else ("Applicable evidence is stale." if stale else "A mandatory project gate blocked execution.")
    )
    return _report_and_finish(
        run,
        project,
        CommandResult(
            command=command,
            status=status,
            exit_code=exit_code,
            message=message,
            project_state=project,
            blockers=blockers,
            next_command="python -m spritelab v3 status" if not review else "python -m spritelab v3 review",
            data=data or {},
        ),
        stage=stage,
    )


def _run_backend(command: list[str], *, root: Path, run: RunState) -> int:
    """Execute an explicitly configured argument array without invoking a shell."""
    if not command or any("\x00" in argument for argument in command):
        raise ValueError("Backend command must be a non-empty safe argument array.")
    run.log(f"backend argv={command!r}")
    with run.log_path.open("a", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, shell=False, check=False
        )
    return completed.returncode


def _create_run(config: ProjectConfig, command: str, argv: list[str], dry_run: bool, project: ProjectState) -> RunState:
    return RunState.create(
        config,
        command=command,
        argv=argv,
        source_commit=project.source_commit,
        dry_run=dry_run,
    )


def dataset_build(config: ProjectConfig, argv: list[str], options: ExecutionOptions) -> CommandResult:
    project = build_project_state(config)
    run = _create_run(config, "dataset build", argv, options.dry_run, project)
    order = [
        "raw-source-provenance",
        "extraction",
        "suitability",
        "semantic-labeling",
        "semantic-calibration",
        "dataset-v5-view-construction",
        "dataset-freeze",
    ]
    for index, key in enumerate(order, start=1):
        stage = project.stage(key)
        run.append_event(
            command="dataset build",
            stage=key,
            event_type="gate_checked",
            status=stage.status.value,
            current_count=index,
            total_count=len(order),
            message=stage.explanation,
            artifact_identity={"evidence": [item.sha256 for item in stage.evidence]},
        )
        if stage.status == StageStatus.NEEDS_REVIEW:
            return _blocked(run, project, "dataset build", key, stage.blockers, review=True)
        if stage.status in {StageStatus.BLOCKED, StageStatus.FAILED, StageStatus.STALE, StageStatus.INCONCLUSIVE}:
            return _blocked(run, project, "dataset build", key, stage.blockers, stale=stage.status == StageStatus.STALE)
    if options.dry_run:
        return _report_and_finish(
            run,
            project,
            CommandResult(
                command="dataset build",
                status="COMPLETE",
                exit_code=ExitCode.SUCCESS,
                message="Dataset build plan validated; dry-run performed no backend action.",
                project_state=project,
                next_command="python -m spritelab v3 dataset build",
                data={"backend_launches": 0, "production_freezes": 0},
            ),
            stage="plan",
        )
    command = config.values["execution"]["dataset_command"]
    if not command:
        return _blocked(run, project, "dataset build", "execution", ["No typed dataset backend adapter is configured."])
    if config.values["execution"]["allow_dataset_production_freeze"]:
        context = ProjectContext(config.root, config.values, config.path, config.runs_dir)
        authorization = authorize_labeling_scope(context, PRODUCTION_CONDITIONED_DATASET_FREEZE)
        if not authorization.authorized:
            return _blocked(
                run,
                build_project_state(config),
                "dataset build",
                "dataset-freeze",
                ["Current verified labeling evidence does not authorize a conditioned production freeze."],
                stale="stale" in authorization.reason,
            )
    code = _run_backend(command, root=config.root, run=run)
    status = "COMPLETE" if code == 0 else "FAILED"
    result = CommandResult(
        command="dataset build",
        status=status,
        exit_code=ExitCode.SUCCESS if code == 0 else ExitCode.INTERNAL_ERROR,
        message="Dataset backend completed." if code == 0 else "Dataset backend failed; completed work is preserved.",
        project_state=build_project_state(config),
        next_command="python -m spritelab v3 status",
        data={"backend_exit_code": code},
    )
    return _report_and_finish(run, result.project_state, result, stage="backend")


def train(config: ProjectConfig, argv: list[str], options: ExecutionOptions) -> CommandResult:
    project = build_project_state(config)
    run = _create_run(config, "train", argv, options.dry_run, project)
    audit = project.stage("training-infrastructure-audit")
    campaign_stage = project.stage("training-campaign")
    if audit.audit == AuditStatus.STALE:
        return _blocked(run, project, "train", audit.key, audit.blockers, stale=True)
    if audit.audit != AuditStatus.PASS:
        return _blocked(run, project, "train", audit.key, audit.blockers or [audit.explanation])
    if campaign_stage.blockers:
        return _blocked(run, project, "train", campaign_stage.key, campaign_stage.blockers)
    campaign_path = config.path_for("training", "campaign_config")
    if campaign_path is None or not campaign_path.is_file():
        return _blocked(
            run,
            project,
            "train",
            "campaign_configuration",
            [
                "A validated training campaign could not be prepared.",
                "training.campaign_config is missing or is not a regular file.",
                "No process was started.",
            ],
            data={"backend_launches": 0, "training_runs": 0, "cuda_initialized": False},
        )
    context = ProjectContext(config.root, config.values, config.path, config.runs_dir)
    try:
        backend = backend_from_context(context)
        service = TrainingService(context, backend)
        resolved = service.plan(TrainingProfile.RECOMMENDED, before_launch=False)
    except (OSError, TypeError, ValueError, LookupError) as exc:
        return _blocked(run, project, "train", "campaign_resolution", [str(exc), "No process was started."])
    if resolved.blockers:
        return _blocked(
            run,
            project,
            "train",
            "campaign_validation",
            [gate.message for gate in resolved.blockers],
            stale=any(gate.gate_id == "training_audit_applicability" for gate in resolved.blockers),
        )
    plan = resolved.to_dict()
    if options.dry_run:
        try:
            validation = validate_training_launch_plan(
                campaign_path,
                compute_backend_id=backend.backend_id,
                project_root=config.root,
                campaign_profile=TrainingProfile.RECOMMENDED.value,
            )
        except (CampaignValidationError, OSError, ValueError) as exc:
            return _blocked(
                run,
                project,
                "train",
                "campaign_validation",
                [str(exc), "No process was started."],
            )
        return _report_and_finish(
            run,
            project,
            CommandResult(
                command="train",
                status="COMPLETE",
                exit_code=ExitCode.SUCCESS,
                message="Training plan validated; dry-run launched no training process.",
                project_state=project,
                next_command="python -m spritelab v3 train",
                data={
                    "plan": plan,
                    "validation": validation,
                    "training_runs": 0,
                    "backend_launches": 0,
                    "receipts_issued": 0,
                    "cuda_initialized": False,
                    "deprecated_training_command_ignored": bool(config.values["execution"]["training_command"]),
                },
            ),
            stage="plan",
        )
    if not sys.stdin.isatty() and not (options.yes and options.non_interactive_confirm):
        return _blocked(
            run,
            project,
            "train",
            "confirmation",
            ["Noninteractive training requires both --yes and --non-interactive-confirm."],
        )
    if sys.stdin.isatty() and not options.yes:
        answer = input("Start the authorized training campaign? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            result = CommandResult(
                command="train",
                status="PAUSED",
                exit_code=ExitCode.PAUSED,
                message="Training was not started; confirmation defaulted to No.",
                project_state=project,
                next_command="python -m spritelab v3 train",
            )
            return _report_and_finish(run, project, result, stage="confirmation")
    product_result = service.start(TrainingProfile.RECOMMENDED, cloud_confirmation=False)
    if product_result.status == ProductStatus.BLOCKED:
        return _blocked(
            run,
            project,
            "train",
            "validated_launch",
            [item.message for item in product_result.blockers] or [product_result.message, "No process was started."],
        )
    try:
        projected_identity = _validated_product_training_identity(product_result)
    except CampaignValidationError as exc:
        backend_launches = product_result.data.get("backend_launches")
        backend_launches = backend_launches if type(backend_launches) is int and backend_launches >= 0 else 0
        return _blocked(
            run,
            project,
            "train",
            "training_identity",
            [str(exc)],
            data={"backend_launches": backend_launches},
        )
    run.append_event(
        command="train",
        stage="validated_launch",
        event_type="backend_handoff",
        status=product_result.status.value,
        message=product_result.message,
        artifact_identity={"campaign_config": str(campaign_path)},
    )
    result = CommandResult(
        command="train",
        status=product_result.status.value,
        exit_code=ExitCode.SUCCESS,
        message=product_result.message,
        project_state=build_project_state(config),
        next_command="python -m spritelab v3 status",
        data={
            "product_result": product_result.to_dict(),
            "deprecated_training_command_ignored": bool(config.values["execution"]["training_command"]),
        },
    )
    return _report_and_finish(
        run,
        result.project_state,
        result,
        stage="validated_launch",
        backend_identity=projected_identity,
    )


def evaluate(config: ProjectConfig, argv: list[str], options: ExecutionOptions) -> CommandResult:
    project = build_project_state(config)
    run = _create_run(config, "eval", argv, options.dry_run, project)
    generation = project.stage("evaluation-generation")
    memorization = project.stage("memorization-review")
    if generation.blockers:
        return _blocked(run, project, "eval", generation.key, generation.blockers)
    training_dataset_identity, training_view_identity = configured_training_identities(config.values)
    identity_blockers = []
    if training_dataset_identity is None:
        identity_blockers.append("Active training dataset identity is missing or malformed.")
    if training_view_identity is None:
        identity_blockers.append("Active training view identity is missing or malformed.")
    if identity_blockers:
        return _blocked(run, project, "eval", "evaluation_identity", identity_blockers)
    if memorization.audit == AuditStatus.STALE:
        return _blocked(run, project, "eval", memorization.key, memorization.blockers, stale=True)
    if memorization.audit != AuditStatus.PASS:
        return _blocked(run, project, "eval", memorization.key, memorization.blockers)
    checkpoint_path = config.path_for("evaluation", "checkpoint")
    benchmark_path = config.path_for("evaluation", "benchmark")
    evaluation_identity_blockers: list[str] = []
    if checkpoint_path is None or not checkpoint_path.is_file():
        evaluation_identity_blockers.append("Configured evaluation checkpoint is missing.")
    if benchmark_path is None or not benchmark_path.is_file():
        evaluation_identity_blockers.append("Configured evaluation benchmark is missing.")
    if evaluation_identity_blockers:
        return _blocked(run, project, "eval", "evaluation_identity", evaluation_identity_blockers)
    assert checkpoint_path is not None and benchmark_path is not None
    try:
        evaluation_identity: dict[str, object] = {
            "dataset_identity": training_dataset_identity,
            "view_identity": training_view_identity,
            "training_dataset_identity": training_dataset_identity,
            "training_view_identity": training_view_identity,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "benchmark_path": str(benchmark_path),
            "benchmark_sha256": _file_sha256(benchmark_path),
        }
    except OSError as exc:
        return _blocked(run, project, "eval", "evaluation_identity", [f"Evaluation identity read failed: {exc}"])
    plan = {
        "checkpoint": str(checkpoint_path),
        "benchmark": str(benchmark_path),
        "training_dataset_identity": training_dataset_identity,
        "training_view_identity": training_view_identity,
        "backend_argv": config.values["execution"]["evaluation_command"],
        "promotion_authorized": project.stage("promotion-decision").production_authorized,
    }
    if options.dry_run:
        return _report_and_finish(
            run,
            project,
            CommandResult(
                command="eval",
                status="COMPLETE",
                exit_code=ExitCode.SUCCESS,
                message="Evaluation plan validated; dry-run generated and promoted nothing.",
                project_state=project,
                next_command="python -m spritelab v3 eval",
                data={"plan": plan, "generation_runs": 0, "promotion_actions": 0},
            ),
            stage="plan",
            backend_identity=evaluation_identity,
        )
    command = config.values["execution"]["evaluation_command"]
    if not command:
        return _blocked(run, project, "eval", "execution", ["No typed evaluation backend adapter is configured."])
    if not sys.stdin.isatty() and not (options.yes and options.non_interactive_confirm):
        return _blocked(
            run, project, "eval", "confirmation", ["Noninteractive evaluation requires explicit confirmation."]
        )
    code = _run_backend(command, root=config.root, run=run)
    result = CommandResult(
        command="eval",
        status="COMPLETE" if code == 0 else "FAILED",
        exit_code=ExitCode.SUCCESS if code == 0 else ExitCode.INTERNAL_ERROR,
        message="Evaluation backend completed; promotion remains governed by its own gate."
        if code == 0
        else "Evaluation backend failed.",
        project_state=build_project_state(config),
        next_command="python -m spritelab v3 status",
        data={"backend_exit_code": code, "promotion_actions": 0},
    )
    return _report_and_finish(
        run,
        result.project_state,
        result,
        stage="backend",
        backend_identity=evaluation_identity,
    )


def unexpected_result(command: str, exc: BaseException, *, debug: bool) -> CommandResult:
    detail = traceback.format_exc() if debug else None
    return CommandResult(
        command=command,
        status="FAILED",
        exit_code=ExitCode.INTERNAL_ERROR,
        message=f"Unexpected internal failure: {exc}",
        next_command="python -m spritelab v3 doctor",
        data={"traceback": detail} if detail else {},
    )
