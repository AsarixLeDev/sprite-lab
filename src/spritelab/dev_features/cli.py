"""Developer command extension and standalone CLI assembly."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.dev_features.artifacts import inspect_artifacts
from spritelab.dev_features.audits import collect_audits
from spritelab.dev_features.doctor import run_developer_doctor
from spritelab.dev_features.projection import project_user_status
from spritelab.dev_features.repository import branch_heads, list_local_branches
from spritelab.dev_features.state import build_developer_state
from spritelab.dev_features.test_profiles import TEST_PROFILES, build_test_plan, execute_test_plan
from spritelab.hierarchical_labeling.cli import register_labeling_commands
from spritelab.v3.config import ConfigError, ProjectConfig
from spritelab.v3.model import CommandResult, ExitCode, ProjectState
from spritelab.v3.orchestration import unexpected_result
from spritelab.v3.status import build_project_state

ConfigLoader = Callable[[], ProjectConfig]
ProjectStateBuilder = Callable[[ProjectConfig], ProjectState]


@dataclass(frozen=True)
class DeveloperCommandEnvironment:
    """Dependencies supplied by the foundation when installing developer commands."""

    load_config: ConfigLoader
    build_project_state: ProjectStateBuilder


def default_environment() -> DeveloperCommandEnvironment:
    return DeveloperCommandEnvironment(
        load_config=lambda: ProjectConfig.load(Path.cwd(), required=False),
        build_project_state=build_project_state,
    )


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", help="Emit stable machine-readable output.")
    common.add_argument("--no-color", action="store_true", help="Disable terminal color output.")
    common.add_argument("--quiet", action="store_true", help="Suppress successful human-readable output.")
    common.add_argument("--debug", action="store_true", help="Show tracebacks and diagnostic detail.")
    return common


def register_developer_commands(
    subparsers: argparse._SubParsersAction,
    *,
    parents: Sequence[argparse.ArgumentParser] = (),
    environment: DeveloperCommandEnvironment | None = None,
) -> None:
    """Foundation extension callback for the ``python -m spritelab dev`` namespace."""

    env = environment or default_environment()

    def command(
        name: str, help_text: str, handler: Callable[[argparse.Namespace], CommandResult]
    ) -> argparse.ArgumentParser:
        parser = subparsers.add_parser(name, help=help_text, parents=list(parents))
        parser.set_defaults(handler=handler, developer_environment=env)
        return parser

    command("status", "Show detailed repository, subsystem, audit, artifact, and authorization state.", _status)
    command("audits", "Show independent audit applicability, freshness, gates, and consequences.", _audits)
    command("branches", "List local branches and worktrees without changing them.", _branches)
    command("artifacts", "Inspect reports, manifests, hashes, missing files, and stale identities.", _artifacts)
    command("doctor", "Run read-only developer repository and environment diagnostics.", _doctor)

    test = command("test", "Run a named repository test profile.", _test)
    test.add_argument("profile", choices=TEST_PROFILES, nargs="?", default="quick")
    test.add_argument("--dry-run", action="store_true", help="Print the exact plan without executing it.")

    explain = command("explain", "Explain one subsystem or the recommended engineering action.", _explain)
    explain.add_argument("topic", nargs="?", help="Subsystem key; defaults to the recommended action.")

    report = command("report", "Render combined developer evidence and the safe user projection.", _report)
    report.add_argument("--output", type=Path, help="Optionally write the report to a JSON or Markdown file.")

    labeling = subparsers.add_parser(
        "labeling",
        help="Prepare reference truth, automatic suggestions, calibration, reports, and pilot plans.",
        parents=list(parents),
    )
    labeling_subparsers = labeling.add_subparsers(dest="labeling_action", required=True)
    register_labeling_commands(labeling_subparsers, parents=parents, environment=env)


def build_parser(*, environment: DeveloperCommandEnvironment | None = None) -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="python -m spritelab dev",
        description="Developer-only repository evidence, audits, diagnostics, and tests.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_developer_commands(subparsers, parents=[common], environment=environment)
    return parser


def _inputs(args: argparse.Namespace) -> tuple[ProjectConfig, ProjectState]:
    environment: DeveloperCommandEnvironment = args.developer_environment
    config = environment.load_config()
    return config, environment.build_project_state(config)


def _status(args: argparse.Namespace) -> CommandResult:
    config, state = _inputs(args)
    details = build_developer_state(config, state)
    return CommandResult(
        command="dev status",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message="Detailed developer state loaded.",
        blockers=list(state.blockers),
        warnings=list(state.warnings),
        next_command=str(details["recommended_action"]["command"]),
        data={"developer_state": details},
        internal_details=True,
    )


def _audits(args: argparse.Namespace) -> CommandResult:
    config, state = _inputs(args)
    audits = collect_audits(config, state)
    return CommandResult(
        command="dev audits",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message=f"Inspected {len(audits)} independent audit surface(s).",
        data={"audits": audits, "current_commit": state.source_commit},
        internal_details=True,
    )


def _branches(args: argparse.Namespace) -> CommandResult:
    config, _state = _inputs(args)
    before = _branch_heads(config.root)
    branches = list_local_branches(config.root)
    after = _branch_heads(config.root)
    unchanged = before == after
    return CommandResult(
        command="dev branches",
        status="COMPLETE" if unchanged else "FAILED",
        exit_code=ExitCode.SUCCESS if unchanged else ExitCode.INTERNAL_ERROR,
        message=f"Inspected {len(branches)} local branch(es) without modification.",
        blockers=[] if unchanged else ["Branch references changed during read-only inspection."],
        data={"branches": branches, "read_only": True, "branch_heads_unchanged": unchanged},
        internal_details=True,
    )


def _branch_heads(root: Path) -> dict[str, str]:
    return branch_heads(root)


def _artifacts(args: argparse.Namespace) -> CommandResult:
    config, state = _inputs(args)
    artifacts = inspect_artifacts(config, state)
    issues = [item for item in artifacts if item["identity_status"] not in {"PRESENT", "CURRENT"}]
    return CommandResult(
        command="dev artifacts",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message=f"Inspected {len(artifacts)} artifact reference(s); found {len(issues)} issue(s).",
        warnings=[f"{item['identity_status']}: {item['reference']}" for item in issues],
        data={
            "artifacts": artifacts,
            "missing": [item for item in artifacts if item["identity_status"] == "MISSING"],
            "stale_identities": [item for item in artifacts if item["identity_status"] == "HASH_MISMATCH"],
            "invalid_references": [item for item in artifacts if item["identity_status"] == "INVALID_REFERENCE"],
            "rewritten": False,
        },
        internal_details=True,
    )


def _doctor(args: argparse.Namespace) -> CommandResult:
    config, state = _inputs(args)
    checks = run_developer_doctor(config, state)
    failures = [check for check in checks if check["mandatory"] and check["status"] == "FAIL"]
    return CommandResult(
        command="dev doctor",
        status="FAILED" if failures else "COMPLETE",
        exit_code=ExitCode.DOCTOR_FAILED if failures else ExitCode.SUCCESS,
        message=(
            f"Developer doctor found {len(failures)} mandatory failure(s)."
            if failures
            else "Developer doctor completed without mandatory failures."
        ),
        blockers=[check["message"] for check in failures],
        warnings=[check["message"] for check in checks if check["status"] == "WARN"],
        data={"checks": checks, "read_only": True, "cuda_initialized": False, "provider_calls": 0},
        internal_details=True,
    )


def _test(args: argparse.Namespace) -> CommandResult:
    config, _state = _inputs(args)
    extra = tuple(getattr(args, "pytest_arguments", ()))
    plan = build_test_plan(config.root, args.profile, extra)
    json_output = bool(getattr(args, "json", False))
    quiet = bool(getattr(args, "quiet", False))
    capture = json_output or quiet
    announcement = None if quiet else (sys.stderr if json_output else sys.stdout)
    completed = execute_test_plan(
        plan,
        root=config.root,
        dry_run=bool(args.dry_run),
        announcement=announcement,
        capture_output=capture,
    )
    return_code = completed.returncode if completed else 0
    stdout = completed.stdout if completed and capture else None
    stderr = completed.stderr if completed and capture else None
    if quiet and return_code and (stdout or stderr):
        print((stdout or "") + (stderr or ""), file=sys.stderr, end="")
    return CommandResult(
        command="dev test",
        status="COMPLETE" if return_code == 0 else "FAILED",
        exit_code=return_code,  # Preserve pytest's exact failure code.
        message=(
            f"Dry-run planned the {plan.profile!r} test profile."
            if args.dry_run
            else f"Test profile {plan.profile!r} exited with code {return_code}."
        ),
        data={
            "profile": plan.profile,
            "dry_run": bool(args.dry_run),
            "arguments": list(plan.arguments),
            "display_command": plan.display_command,
            "matched_files": list(plan.matched_files),
            "test_exit_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "shell": False,
        },
        internal_details=True,
    )


def _explain(args: argparse.Namespace) -> CommandResult:
    config, state = _inputs(args)
    details = build_developer_state(config, state)
    topic = args.topic
    if topic is None:
        explanation: dict[str, Any] = {"recommended_action": details["recommended_action"]}
    else:
        normalized = topic.lower().replace("_", "-")
        aliases = {
            "training": "training-campaign",
            "training-audit": "training-infrastructure-audit",
            "labeling": "semantic-labeling",
            "evaluation": "evaluation-generation",
            "memorization": "memorization-review",
            "promotion": "promotion-decision",
            "freeze": "dataset-freeze",
        }
        normalized = aliases.get(normalized, normalized)
        subsystem = next((item for item in details["subsystems"] if item["key"] == normalized), None)
        audit = next(
            (
                item
                for item in details["audits"]
                if item["subsystem"] == normalized or normalized.startswith(item["subsystem"])
            ),
            None,
        )
        if subsystem is None and audit is None:
            return CommandResult(
                command="dev explain",
                status="INVALID",
                exit_code=ExitCode.INVALID,
                message=f"Unknown developer subsystem: {topic}",
                data={"available_topics": [item["key"] for item in details["subsystems"]]},
                internal_details=True,
            )
        explanation = {"subsystem": subsystem, "audit": audit}
    return CommandResult(
        command="dev explain",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message="Developer evidence explanation loaded.",
        data=explanation,
        internal_details=True,
    )


def _report(args: argparse.Namespace) -> CommandResult:
    config, state = _inputs(args)
    details = build_developer_state(config, state)
    projection = project_user_status(details)
    report = {"schema_version": "spritelab.dev.report.v1", "developer": details, "user_projection": projection}
    report_path = None
    if args.output:
        output = args.output if args.output.is_absolute() else config.root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        content = (
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            if output.suffix.lower() == ".json"
            else _report_markdown(report)
        )
        output.write_text(content, encoding="utf-8", newline="\n")
        report_path = str(output)
    return CommandResult(
        command="dev report",
        status="COMPLETE",
        exit_code=ExitCode.SUCCESS,
        message="Combined developer report assembled.",
        report_path=report_path,
        data={"report": report},
        internal_details=True,
    )


def _report_markdown(report: dict[str, Any]) -> str:
    developer = report["developer"]
    repository = developer["repository"]
    lines = [
        "# Sprite Lab developer report",
        "",
        f"Branch: `{repository['branch']}`",
        f"Commit: `{repository['commit']}`",
        f"Worktree clean: `{repository['clean']}`",
        "",
        "## Audits",
        "",
    ]
    for audit in developer["audits"]:
        lines.append(
            f"- {audit['subsystem']}: {audit['verdict']} ({audit['freshness']}); "
            f"consequence={audit['authorization_consequence']}"
        )
    lines.extend(["", "## User projection", ""])
    lines.extend(f"- {area['title']}: {area['message']}" for area in report["user_projection"]["areas"])
    return "\n".join(lines) + "\n"


def _render_human(result: CommandResult) -> str:
    lines = [result.command, "", result.message]
    data = result.data
    if result.command == "dev status" and data:
        details = data["developer_state"]
        repository = details["repository"]
        lines.extend(
            [
                "",
                "Repository",
                f"  branch: {repository['branch']}",
                f"  commit: {repository['commit']}",
                f"  worktree: {(repository['worktree'] or {}).get('path')}",
                f"  clean: {repository['clean']}",
                "",
                "Subsystems",
            ]
        )
        lines.extend(
            f"  {item['key']}: implementation={item['implementation']} status={item['status']} "
            f"audit={item['audit_verdict']}"
            for item in details["subsystems"]
        )
        lines.extend(["", "Independent audits"])
        for item in details["audits"]:
            lines.extend(
                [
                    f"  {item['subsystem']}: verdict={item['verdict']} freshness={item['freshness']} "
                    f"applicability={item['applicability']} failed_gates={','.join(item['failed_gates']) or '-'}",
                    f"    bound_commit={item['bound_commit']} current_code_identity="
                    f"{item.get('current_relevant_code_identity')}",
                    f"    verified_artifact={item.get('verified_artifact_status', 'LEGACY_PROJECTION')} "
                    f"authorized_scopes={','.join(item.get('authorized_scopes', ())) or '-'}",
                    f"    staleness_reasons={','.join(item.get('staleness_reasons', ())) or '-'}",
                    f"    downstream={','.join(item.get('downstream_consequences', ())) or item['authorization_consequence']}",
                    f"    report={item['report']}",
                ]
            )
        lines.extend(["", "Artifact identities"])
        lines.extend(
            f"  {item['identity_status']}: {item['path']} sha256={item['sha256']} expected={item['expected_sha256']}"
            for item in details["artifacts"]
        )
        lines.extend(["", "Dataset freeze identities"])
        lines.extend(
            f"  {item['source']}: {item['path']} sha256={item['sha256']}"
            for item in details["dataset_freeze_identities"]
        )
        authorization = details["authorization"]
        lines.extend(
            [
                "",
                f"Training authorized: {authorization['training']['authorized']}",
                f"Promotion authorized: {authorization['promotion']['authorized']}",
                f"Active developer runs: {len(details['active_developer_runs'])}",
                f"Recommended action: {details['recommended_action']['action']}",
                f"Recommended command: {details['recommended_action']['command']}",
            ]
        )
    elif result.command == "dev audits" and data:
        lines.extend([""])
        for item in data["audits"]:
            lines.extend(
                [
                    item["subsystem"],
                    f"  verdict: {item['verdict']}",
                    f"  bound commit: {item['bound_commit']}",
                    f"  current commit: {item['current_commit']}",
                    f"  current relevant code identity: {item.get('current_relevant_code_identity')}",
                    f"  applicability: {item['applicability']}",
                    f"  freshness: {item['freshness']}",
                    f"  staleness reasons: {', '.join(item.get('staleness_reasons', ())) or '-'}",
                    f"  verified artifact: {item.get('verified_artifact_status', 'LEGACY_PROJECTION')}",
                    f"  authorized scopes: {', '.join(item.get('authorized_scopes', ())) or '-'}",
                    f"  downstream consequences: "
                    f"{', '.join(item.get('downstream_consequences', ())) or item['authorization_consequence']}",
                    f"  failed gates: {', '.join(item['failed_gates']) or '-'}",
                    f"  report: {item['report']}",
                    f"  consequence: {item['authorization_consequence']}",
                ]
            )
    elif result.command == "dev branches" and data:
        lines.extend([""])
        lines.extend(
            f"{item['branch']} commit={item['commit']} worktree={item['worktree']} clean={item['clean']} "
            f"ahead={item['ahead']} behind={item['behind']} merged={item['merged']} contained={item['contained']} "
            f"superseded={item['likely_superseded']}"
            for item in data["branches"]
        )
    elif result.command == "dev artifacts" and data:
        lines.extend([""])
        lines.extend(
            f"{item['identity_status']} {item['kind']} {item['reference']} sha256={item['sha256']} "
            f"expected={item['expected_sha256']} source={item['source']}"
            for item in data["artifacts"]
        )
    elif result.command == "dev doctor" and data:
        lines.extend([""])
        lines.extend(f"{item['status']:4} {item['key']}: {item['message']}" for item in data["checks"])
    elif result.command == "dev test" and data:
        lines.extend(
            [
                "",
                f"Profile: {data['profile']}",
                f"Planned command: {data['display_command']}",
                f"Dry-run: {data['dry_run']}",
                f"Exit code: {data['test_exit_code']}",
            ]
        )
    elif data:
        lines.extend(["", json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)])
    if result.blockers:
        lines.extend(["", "Blockers:", *(f"  - {item}" for item in result.blockers)])
    if result.warnings:
        lines.extend(["", "Warnings:", *(f"  - {item}" for item in result.warnings)])
    if result.next_command:
        lines.extend(["", f"Next command: {result.next_command}"])
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: DeveloperCommandEnvironment | None = None,
) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    pytest_arguments: list[str] = []
    if "--" in raw:
        separator = raw.index("--")
        pytest_arguments = raw[separator + 1 :]
        raw = raw[:separator]
    parser = build_parser(environment=environment)
    args = parser.parse_args(raw)
    if pytest_arguments and args.command != "test":
        parser.error("arguments after -- are only supported by dev test")
    args.pytest_arguments = pytest_arguments
    debug = bool(getattr(args, "debug", False))
    try:
        result = args.handler(args)
    except ConfigError as exc:
        result = CommandResult(
            command=f"dev {getattr(args, 'command', 'status')}",
            status="INVALID",
            exit_code=ExitCode.INVALID,
            message=str(exc),
            blockers=[str(exc)],
            internal_details=True,
        )
    except Exception as exc:
        result = unexpected_result(f"dev {getattr(args, 'command', 'status')}", exc, debug=debug)
        result.internal_details = True

    json_output = bool(getattr(args, "json", False))
    quiet = bool(getattr(args, "quiet", False))
    if json_output:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    elif not quiet:
        print(_render_human(result))
    elif int(result.exit_code) != 0:
        print(result.message, file=sys.stderr)
    raise SystemExit(int(result.exit_code))
