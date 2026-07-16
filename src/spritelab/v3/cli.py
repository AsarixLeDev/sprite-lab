"""Command-line interface for the human-compatible Sprite Lab v3 workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from spritelab.product_core import ProductPlugin, ProductPluginRegistry, ProductResult, ProductStatus
from spritelab.product_core.cli import ProductCliRegistry
from spritelab.product_web.events import verify_event_migration
from spritelab.training.campaign import is_concrete_hash
from spritelab.v3.config import CONFIG_NAME, ConfigError, ProjectConfig, discover_config, template_text
from spritelab.v3.doctor import run_doctor
from spritelab.v3.model import CommandResult, ExitCode, ProjectState, StageStatus
from spritelab.v3.orchestration import ExecutionOptions, dataset_build, evaluate, train, unexpected_result
from spritelab.v3.report import generate_report, latest_report, open_report
from spritelab.v3.run_state import RunState, list_runs, resumable_runs
from spritelab.v3.status import build_project_state


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--json", action="store_true", help="Emit the stable machine-readable result schema.")
    parser.add_argument("--no-color", action="store_true", help="Disable color and Unicode decoration.")
    parser.add_argument("--quiet", action="store_true", help="Suppress nonessential human output.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks and diagnostic detail.")
    return parser


def _mutable_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(add_help=False, parents=[common])
    parser.add_argument("--dry-run", action="store_true", help="Validate and plan without launching a backend action.")
    return parser


def build_parser(plugins: Iterable[ProductPlugin] = ()) -> argparse.ArgumentParser:
    common = _common_parser()
    mutable = _mutable_parser()
    parser = argparse.ArgumentParser(
        prog="python -m spritelab v3",
        description="Sprite Lab product commands and local interface.",
        parents=[common],
    )
    parser.set_defaults(handler=_handle_product_default)
    sub = parser.add_subparsers(dest="command")
    registry = ProductCliRegistry(parents=(common,))

    def simple(
        name: str,
        help_text: str,
        handler: Callable[[argparse.Namespace, list[str]], CommandResult],
        *,
        mutable_command: bool = False,
        configure: Callable[[argparse.ArgumentParser], None] | None = None,
    ) -> None:
        parent = mutable if mutable_command else common

        def install(target: argparse._SubParsersAction) -> None:
            command = target.add_parser(name, parents=[parent], help=help_text)
            if configure:
                configure(command)
            command.set_defaults(handler=handler)

        registry.register(name, install, owner="spritelab.v3.legacy")

    def configure_init(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--create-launchers",
            action="store_true",
            help="Create project-local Windows and shell launchers without overwriting files.",
        )
        command.add_argument("--yes", action="store_true", help="Explicitly confirm launcher creation.")

    simple(
        "init",
        "Create a safe project configuration.",
        _handle_init,
        mutable_command=True,
        configure=configure_init,
    )
    simple("status", "Show a simple end-user project status.", _handle_status)
    simple("doctor", "Check the local environment without initializing CUDA.", _handle_doctor)

    def install_dataset(target: argparse._SubParsersAction) -> None:
        dataset = target.add_parser("dataset", parents=[common], help="Dataset-v5 workflows.")
        dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
        build = dataset_sub.add_parser("build", parents=[mutable], help="Orchestrate a fail-closed Dataset-v5 build.")
        build.set_defaults(handler=_handle_dataset_build)

    registry.register("dataset", install_dataset, owner="spritelab.v3.legacy")

    def configure_confirm(command: argparse.ArgumentParser) -> None:
        command.add_argument("--yes", action="store_true", help="Confirm the displayed plan.")
        command.add_argument(
            "--non-interactive-confirm",
            action="store_true",
            help="Required with --yes when an authorized action runs without a TTY.",
        )

    simple(
        "train",
        "Plan or run an authorized training campaign.",
        _handle_train,
        mutable_command=True,
        configure=configure_confirm,
    )
    simple(
        "eval",
        "Plan or run authorized evaluation.",
        _handle_eval,
        mutable_command=True,
        configure=configure_confirm,
    )
    simple(
        "resume",
        "Revalidate and resume a protected incomplete run.",
        _handle_resume,
        mutable_command=True,
        configure=lambda command: command.add_argument(
            "--run-id", help="Explicit run ID (required noninteractively when several runs are resumable)."
        ),
    )
    simple("review", "Discover and launch the actionable review workflow.", _handle_review, mutable_command=True)
    simple(
        "report",
        "Show or open the latest offline HTML report.",
        _handle_report,
        configure=lambda command: command.add_argument(
            "--open", action="store_true", help="Open the report in the platform browser."
        ),
    )
    simple(
        "runs",
        "List recent v3 runs.",
        _handle_runs,
        configure=lambda command: command.add_argument(
            "--limit", type=int, default=20, help="Maximum runs to show (default: 20)."
        ),
    )

    def configure_logs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-id", help="Run ID; defaults to the latest run.")
        command.add_argument("--follow", action="store_true", help="Follow an active run log until it stops.")

    simple("logs", "Show the latest or selected v3 run log.", _handle_logs, configure=configure_logs)
    simple(
        "explain",
        "Explain a pipeline stage or gate.",
        _handle_explain,
        configure=lambda command: command.add_argument(
            "stage", help="Stage name, for example training-audit or memorization."
        ),
    )

    ProductPluginRegistry(plugins).register_cli(registry)
    for feature in ("providers", "settings"):
        if not registry.contains(feature):

            def install_missing(target: argparse._SubParsersAction, feature_name: str = feature) -> None:
                command = target.add_parser(feature_name, parents=[common], help=f"{feature_name.title()} feature.")
                command.set_defaults(handler=_handle_missing_feature, missing_feature=feature_name)

            registry.register(feature, install_missing, owner="spritelab.product.placeholder")
    registry.install(sub)
    return parser


def _handle_product_default(args: argparse.Namespace, argv: list[str]) -> ProductResult:
    return ProductResult(
        status=ProductStatus.UNAVAILABLE,
        feature="web",
        message="The local Sprite Lab web interface is not registered yet.",
        data={"dispatch_contract": "local_web", "feature_registered": False},
    )


def _handle_missing_feature(args: argparse.Namespace, argv: list[str]) -> ProductResult:
    feature = str(args.missing_feature)
    return ProductResult(
        status=ProductStatus.UNAVAILABLE,
        feature=feature,
        message=f"Feature not registered: {feature}.",
        data={"feature_registered": False},
    )


def _as_command_result(value: CommandResult | ProductResult, command: str) -> CommandResult:
    if isinstance(value, CommandResult):
        return value
    exit_codes = {
        ProductStatus.BLOCKED: ExitCode.BLOCKED,
        ProductStatus.FAILED: ExitCode.INTERNAL_ERROR,
        ProductStatus.NEEDS_REVIEW: ExitCode.REVIEW_REQUIRED,
        ProductStatus.PAUSED: ExitCode.PAUSED,
        ProductStatus.UNAVAILABLE: ExitCode.BLOCKED,
    }
    return CommandResult(
        command=command,
        status=value.status.value,
        exit_code=exit_codes.get(value.status, ExitCode.SUCCESS),
        message=value.message,
        blockers=[item.message for item in value.blockers],
        warnings=[item.message for item in value.warnings],
        next_command=str(value.data.get("next_command") or "python -m spritelab v3 status"),
        data={"product_result": value.to_dict()},
    )


def _load(*, required: bool = True) -> ProjectConfig:
    return ProjectConfig.load(Path.cwd(), required=required)


def _options(args: argparse.Namespace) -> ExecutionOptions:
    return ExecutionOptions(
        dry_run=bool(getattr(args, "dry_run", False)),
        yes=bool(getattr(args, "yes", False)),
        non_interactive_confirm=bool(getattr(args, "non_interactive_confirm", False)),
        debug=bool(args.debug),
    )


def _handle_status(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    config = _load(required=False)
    state = build_project_state(config)
    product = state.to_product_dict()
    return CommandResult(
        command="status",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message="Project status is ready.",
        project_state=state,
        blockers=[item for stage in product["stages"] for item in stage["blockers"]],
        warnings=list(product["warnings"]),
        next_command=_next_command(state),
    )


def _handle_doctor(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    config = _load(required=False)
    checks = run_doctor(config)
    failed = [check.message for check in checks if check.status == "FAIL" and check.mandatory]
    return CommandResult(
        command="doctor",
        status="FAILED" if failed else "COMPLETE",
        exit_code=ExitCode.DOCTOR_FAILED if failed else ExitCode.SUCCESS,
        message="Mandatory environment checks failed." if failed else "Mandatory environment checks passed.",
        blockers=failed,
        warnings=[check.message for check in checks if check.status == "WARN"],
        next_command="python -m spritelab v3 init" if config.path is None else "python -m spritelab v3 status",
        data={"checks": [check.to_dict() for check in checks], "cuda_initialized": False, "provider_calls": 0},
    )


def _project_root_for_init() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    return Path(result.stdout.strip()).resolve() if result and result.returncode == 0 else Path.cwd().resolve()


def _handle_init(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    existing = discover_config(Path.cwd())
    target = existing or (_project_root_for_init() / CONFIG_NAME)
    if existing:
        return CommandResult(
            command="init",
            status="BLOCKED",
            exit_code=ExitCode.INVALID,
            message="An existing configuration was preserved; init never overwrites it.",
            blockers=[f"Configuration already exists: {existing}"],
            next_command="python -m spritelab v3 status",
            data={"config_path": str(existing), "overwritten": False},
        )
    if getattr(args, "create_launchers", False) and not getattr(args, "yes", False):
        return CommandResult(
            command="init",
            status="BLOCKED",
            exit_code=ExitCode.INVALID,
            message="Launcher creation requires explicit confirmation.",
            blockers=["Add --yes to confirm creating project-local launcher files."],
            next_command="python -m spritelab v3 init --create-launchers --yes",
            data={"created": False, "launchers_created": 0},
        )
    if args.dry_run:
        return CommandResult(
            command="init",
            status="COMPLETE",
            exit_code=ExitCode.SUCCESS,
            message="Dry-run: configuration would be created without overwriting any file.",
            next_command="python -m spritelab v3 init",
            data={
                "config_path": str(target),
                "created": False,
                "discovered_artifact_root": str(target.parent / "artifacts"),
            },
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(template_text())
    except FileExistsError:
        raise ConfigError(f"Configuration appeared during init and was preserved: {target}") from None
    config = ProjectConfig.load(target.parent)
    launcher_results = ()
    if getattr(args, "create_launchers", False):
        from spritelab.product_ux.launchers import generate_project_launchers

        launcher_results = generate_project_launchers(target.parent)
    state = build_project_state(config)
    run = RunState.create(config, command="init", argv=argv, source_commit=state.source_commit, dry_run=False)
    result = CommandResult(
        command="init",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message="Created a safe v3 project configuration; all production actions default to disabled.",
        project_state=state,
        next_command="python -m spritelab v3 status",
        run_id=run.run_id,
        data={
            "config_path": str(target),
            "created": True,
            "overwritten": False,
            "launchers": [
                {"name": item.path.name, "status": item.status, "message": item.message} for item in launcher_results
            ],
        },
    )
    report_path, _ = generate_report(state, run.directory / "report", run=result.to_dict())
    result.report_path = str(report_path)
    run.finish(command="init", status="COMPLETE", exit_code=0, message=result.message, stage="configuration")
    return result


def _handle_dataset_build(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    return dataset_build(_load(), argv, _options(args))


def _handle_train(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    return train(_load(), argv, _options(args))


def _handle_eval(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    return evaluate(_load(), argv, _options(args))


def _select_run(config: ProjectConfig, run_id: str | None, *, resumable_only: bool = False) -> dict[str, Any] | None:
    candidates = resumable_runs(config.runs_dir) if resumable_only else list_runs(config.runs_dir)
    if run_id:
        return next((run for run in candidates if run.get("run_id") == run_id), None)
    if not candidates:
        return None
    if resumable_only and len(candidates) > 1 and not sys.stdin.isatty():
        raise ConfigError("Several runs are resumable; noninteractive use requires --run-id.")
    if resumable_only and len(candidates) > 1:
        print("Resumable runs:")
        for index, run in enumerate(candidates, start=1):
            print(f"  {index}. {run.get('run_id')}  {run.get('command')}  {run.get('stage')}")
        selection = input("Resume which run? [1] ").strip() or "1"
        try:
            return candidates[int(selection) - 1]
        except (ValueError, IndexError):
            raise ConfigError("Invalid run selection.") from None
    return candidates[0]


def _handle_resume(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    config = _load()
    selected = _select_run(config, args.run_id, resumable_only=True)
    if selected is None:
        return CommandResult(
            command="resume",
            status="BLOCKED",
            exit_code=ExitCode.INVALID,
            message="No safely resumable incomplete run was found.",
            blockers=["Completed, failed, and blocked runs cannot be manufactured into resumable runs."],
            next_command="python -m spritelab v3 runs",
        )
    run_id = str(selected.get("run_id") or "")
    migration = verify_event_migration(
        run_id,
        config.runs_dir / run_id,
        migration_required=bool(selected.get("event_migration_required")),
        origin_required=True,
    )
    if not migration.resume_compatible or (migration.migration_required and not migration.migration_verified):
        return CommandResult(
            command="resume",
            status="BLOCKED",
            exit_code=ExitCode.INVALID,
            message="Safe resume refused because event-history origin or migration evidence is invalid.",
            blockers=[f"{migration.state.value}: {migration.message}"],
            next_command="python -m spritelab v3 runs",
            data={
                "run_id": run_id,
                "event_history_origin": migration.event_history_origin,
                "event_migration_state": migration.state.value,
                "backend_launches": 0,
            },
        )
    identity_blockers = _training_resume_identity_blockers(selected)
    if identity_blockers:
        return CommandResult(
            command="resume",
            status="BLOCKED",
            exit_code=ExitCode.INVALID,
            message="Safe resume refused because the durable training identity is incomplete or invalid.",
            blockers=identity_blockers,
            next_command="python -m spritelab v3 runs",
            data={"run_id": run_id, "backend_launches": 0},
        )
    current = build_project_state(config)
    if selected.get("source_commit") != current.source_commit:
        return CommandResult(
            command="resume",
            status="STALE",
            exit_code=ExitCode.STALE,
            message="Safe resume refused because the protected source identity changed.",
            project_state=current,
            blockers=[
                f"Run source {selected.get('source_commit')} does not match current source {current.source_commit}."
            ],
            next_command="python -m spritelab v3 status",
            data={"run_id": selected.get("run_id"), "backend_launches": 0},
        )
    if args.dry_run:
        return CommandResult(
            command="resume",
            status="COMPLETE",
            exit_code=ExitCode.SUCCESS,
            message="Safe-resume identities match; dry-run did not continue the backend.",
            project_state=current,
            next_command=f"python -m spritelab v3 resume --run-id {selected.get('run_id')}",
            data={"run": selected, "backend_launches": 0},
        )
    return CommandResult(
        command="resume",
        status="PAUSED",
        exit_code=ExitCode.PAUSED,
        message="Identity validation passed, but continuation must be owned by the configured safe backend adapter.",
        project_state=current,
        blockers=["No automatic backend continuation was authorized."],
        next_command="python -m spritelab v3 status",
        data={"run": selected, "backend_launches": 0},
    )


def _training_resume_identity_blockers(selected: Mapping[str, Any]) -> list[str]:
    if str(selected.get("command") or "").lower() not in {"train", "training", "training.start"}:
        return []
    backend = selected.get("backend_identity")
    if not isinstance(backend, Mapping):
        return ["Durable training backend identity is missing or malformed."]
    blockers: list[str] = []
    dataset_identity = backend.get("dataset_identity")
    if not is_concrete_hash(dataset_identity):
        blockers.append("Durable training dataset identity is missing or malformed.")
    view_values: list[str] = []
    for key in ("view_identity", "training_view_identity"):
        value = backend.get(key)
        if not is_concrete_hash(value):
            blockers.append(f"Durable training {key} is missing or malformed.")
        else:
            view_values.append(value)
    optional_view = backend.get("dataset_view_manifest_hash")
    if optional_view is not None:
        if not is_concrete_hash(optional_view):
            blockers.append("Durable training dataset_view_manifest_hash is malformed.")
        else:
            view_values.append(optional_view)
    if len(set(view_values)) > 1:
        blockers.append("Durable training-view identity aliases disagree.")
    product_run_id = backend.get("product_training_run_id")
    if not isinstance(product_run_id, str) or not product_run_id or product_run_id != product_run_id.strip():
        blockers.append("Durable product training run identity is missing or malformed.")
    return blockers


def _handle_review(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    config = _load()
    project = build_project_state(config)
    queues = [
        str(path)
        for value in config.values["labeling"]["review_queues"]
        if (path := config.path_for_value(value)) is not None and path.exists()
    ]
    command = config.values["execution"]["review_command"]
    run = RunState.create(
        config, command="review", argv=argv, source_commit=project.source_commit, dry_run=args.dry_run
    )
    if not queues:
        result = CommandResult(
            command="review",
            status="BLOCKED",
            exit_code=ExitCode.INVALID,
            message="No actionable review queue was discovered.",
            project_state=project,
            blockers=["Configure labeling.review_queues after a backend creates a review queue."],
            next_command="python -m spritelab v3 status",
        )
    elif args.dry_run or not command:
        result = CommandResult(
            command="review",
            status="NEEDS_REVIEW",
            exit_code=ExitCode.REVIEW_REQUIRED,
            message="Actionable review work was discovered; no external interface was launched.",
            project_state=project,
            next_command="python -m spritelab v3 review",
            data={"queues": queues, "launch_command_configured": bool(command), "backend_launches": 0},
        )
    else:
        completed = subprocess.run(command, cwd=config.root, shell=False, check=False)
        result = CommandResult(
            command="review",
            status="COMPLETE" if completed.returncode == 0 else "FAILED",
            exit_code=ExitCode.SUCCESS if completed.returncode == 0 else ExitCode.INTERNAL_ERROR,
            message="Review interface exited."
            if completed.returncode == 0
            else "Review interface failed to launch or exited with an error.",
            project_state=project,
            next_command="python -m spritelab v3 status",
            data={"queues": queues, "backend_exit_code": completed.returncode},
        )
    report_path, _ = generate_report(project, run.directory / "report", run=result.to_dict())
    result.report_path, result.run_id = str(report_path), run.run_id
    run.finish(
        command="review", status=result.status, exit_code=int(result.exit_code), message=result.message, stage="review"
    )
    return result


def _handle_report(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    config = _load(required=False)
    path = latest_report(config.runs_dir)
    if path is None:
        state = build_project_state(config)
        path, _ = generate_report(state, config.runs_dir / "project-report")
    opened = open_report(path) if args.open else False
    return CommandResult(
        command="report",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message="Latest offline report is available." + (" Browser launch requested." if args.open else ""),
        report_path=str(path),
        next_command="python -m spritelab v3 status",
        data={"opened": opened, "offline": True},
    )


def _handle_runs(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    if args.limit < 1:
        raise ConfigError("--limit must be at least 1.")
    config = _load(required=False)
    runs = list_runs(config.runs_dir)[: args.limit]
    return CommandResult(
        command="runs",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message=f"Found {len(runs)} recent run(s).",
        next_command="python -m spritelab v3 logs",
        data={"runs": runs},
    )


def _handle_logs(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    config = _load(required=False)
    run = _select_run(config, args.run_id)
    if run is None:
        return CommandResult(
            command="logs",
            status="BLOCKED",
            exit_code=ExitCode.INVALID,
            message="No v3 run log exists yet.",
            next_command="python -m spritelab v3 status",
        )
    log_path = Path(run["directory"]) / "logs" / "run.log"
    content = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    if args.follow and run.get("status") == "RUNNING" and sys.stdout.isatty():
        position = len(content.encode("utf-8"))
        try:
            while run.get("status") == "RUNNING":
                time.sleep(0.5)
                if log_path.is_file():
                    with log_path.open("r", encoding="utf-8") as handle:
                        handle.seek(position)
                        update = handle.read()
                        position = handle.tell()
                    if update:
                        print(update, end="")
                run = _select_run(config, args.run_id) or run
        except KeyboardInterrupt:
            pass
    return CommandResult(
        command="logs",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message=f"Log for {run.get('run_id')}.",
        next_command="python -m spritelab v3 runs",
        data={"run_id": run.get("run_id"), "log_path": str(log_path), "content": content},
    )


def _handle_explain(args: argparse.Namespace, argv: list[str]) -> CommandResult:
    config = _load(required=False)
    state = build_project_state(config)
    try:
        stage = state.stage(args.stage)
    except KeyError:
        names = ", ".join(item.key for item in state.stages)
        raise ConfigError(f"Unknown stage '{args.stage}'. Available stages: {names}") from None
    dependencies = [item.key for item in state.stages[: state.stages.index(stage)]]
    return CommandResult(
        command="explain",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message=stage.explanation,
        project_state=state,
        blockers=stage.blockers,
        warnings=stage.warnings,
        next_command=stage.next_command,
        data={"stage": stage.to_dict(), "dependencies": dependencies},
    )


def _next_command(state: ProjectState) -> str:
    for stage in state.stages:
        if stage.status == StageStatus.NEEDS_REVIEW:
            return "python -m spritelab v3 review"
        if stage.status in {StageStatus.BLOCKED, StageStatus.FAILED, StageStatus.STALE}:
            return stage.next_command
    return "python -m spritelab v3 dataset build"


def _render_state(state: ProjectState, *, ascii_only: bool) -> list[str]:
    lines = [f"Sprite Lab v3 — {state.project_name}", "", "Pipeline status"]
    symbol = "*" if ascii_only else "●"
    for stage in state.to_product_dict()["stages"]:
        lines.append(f"{symbol} {stage['title']:<36} {stage['status']:<13}")
        lines.append(f"    {stage['explanation']}")
    lines.extend(["", "Detailed evidence is available with: python -m spritelab dev status"])
    return lines


def _render_human(result: CommandResult, args: argparse.Namespace) -> str:
    if args.quiet and result.exit_code == ExitCode.SUCCESS:
        return ""
    lines: list[str] = []
    if result.command == "status" and result.project_state:
        lines.extend(_render_state(result.project_state, ascii_only=args.no_color))
    elif result.command == "doctor":
        lines.append("Sprite Lab v3 doctor")
        for check in result.data.get("checks", []):
            lines.append(f"[{check['status']:<4}] {check['key']:<22} {check['message']}")
    elif result.command == "runs":
        lines.append("Recent Sprite Lab v3 runs")
        for run in result.data.get("runs", []):
            started = str(run.get("started_at", "unknown"))
            ended = run.get("ended_at")
            elapsed = "active"
            if ended:
                try:
                    elapsed = str(datetime.fromisoformat(ended) - datetime.fromisoformat(started)).split(".")[0]
                except ValueError:
                    elapsed = "unknown"
            lines.append(
                f"{run.get('run_id')}  {run.get('command'):<14} {run.get('status'):<12} {started}  {elapsed}  "
                f"resume={str(bool(run.get('resumable'))).lower()} report={str(bool(run.get('report_available'))).lower()}"
            )
    elif result.command == "logs":
        lines.append(result.data.get("content", ""))
    elif result.command == "explain":
        stage = result.data.get("stage", {})
        lines.extend([str(stage.get("title", "Stage")), f"Status: {stage.get('status')}", result.message])
        evidence = stage.get("evidence", [])
        if evidence:
            lines.append("Evidence:")
            lines.extend(f"  {item.get('path')}  sha256={item.get('sha256')}" for item in evidence)
    else:
        heading = result.command.upper()
        if result.exit_code in {ExitCode.BLOCKED, ExitCode.REVIEW_REQUIRED, ExitCode.STALE}:
            heading += f" {result.status}"
        lines.extend([heading, "", result.message])
    if result.blockers:
        lines.extend(["", "What stopped:"])
        lines.extend(f"  - {item}" for item in result.blockers)
        lines.extend(["", "Completed work was preserved."])
    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {item}" for item in result.warnings)
    if result.next_command:
        lines.extend(["", "Next:", f"  {result.next_command}"])
    if result.report_path:
        lines.extend(["", "Details:", f"  {result.report_path}"])
    if args.debug and result.data.get("traceback"):
        lines.extend(["", str(result.data["traceback"])])
    return "\n".join(lines).rstrip()


def main(argv: Sequence[str] | None = None, *, plugins: Iterable[ProductPlugin] = ()) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(plugins)
    args = parser.parse_args(raw)
    try:
        result = _as_command_result(args.handler(args, raw), str(getattr(args, "command", None) or "v3"))
    except ConfigError as exc:
        result = CommandResult(
            command=str(getattr(args, "command", "v3")),
            status="INVALID",
            exit_code=ExitCode.INVALID,
            message=str(exc),
            blockers=[str(exc)],
            next_command="python -m spritelab v3 init",
        )
    except Exception as exc:
        result = unexpected_result(str(getattr(args, "command", "v3")), exc, debug=args.debug)
    output = (
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        if args.json
        else _render_human(result, args)
    )
    if output:
        print(output)
    raise SystemExit(int(result.exit_code))


if __name__ == "__main__":
    main()
