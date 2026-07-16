"""Detailed developer state assembled from v3 state and repository evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spritelab.dev_features.artifacts import inspect_artifacts, sha256_file
from spritelab.dev_features.audits import collect_audits
from spritelab.dev_features.repository import repository_state
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import ProjectState, StageState
from spritelab.v3.run_state import list_runs

DEVELOPER_STATUS_SCHEMA = "spritelab.dev.status.v1"


def _stage_record(stage: StageState) -> dict[str, Any]:
    return {
        "key": stage.key,
        "title": stage.title,
        "implementation": stage.implementation,
        "status": stage.status.value,
        "audit": stage.audit.value,
        "audit_verdict": stage.audit.value,
        "production_authorized": stage.production_authorized,
        "source_commit": stage.source_commit,
        "failed_gates": list(stage.metrics.get("failed_gates", [])),
        "blockers": list(stage.blockers),
        "warnings": list(stage.warnings),
        "evidence": [
            {"path": item.path, "sha256": item.sha256, "source_commit": item.source_commit} for item in stage.evidence
        ],
        "next_action": stage.next_action,
        "next_command": stage.next_command,
    }


def _freeze_identities(config: ProjectConfig) -> list[dict[str, Any]]:
    def configured_path(section: str, key: str) -> Path | None:
        try:
            return config.path_for(section, key)
        except KeyError:
            return None

    values = (
        ("dataset.freeze_manifest", configured_path("dataset", "freeze_manifest")),
        ("training.dataset_freeze", configured_path("training", "dataset_freeze")),
    )
    identities = []
    seen: set[Path] = set()
    for source, path in values:
        if path is None or path in seen:
            continue
        seen.add(path)
        exists = path.is_file()
        identities.append(
            {
                "source": source,
                "path": str(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    return identities


def _authorization(state: ProjectState) -> dict[str, Any]:
    def value(key: str) -> dict[str, Any]:
        try:
            stage = state.stage(key)
        except KeyError:
            return {"authorized": False, "blockers": ["Subsystem state is unavailable."]}
        return {"authorized": stage.production_authorized, "blockers": list(stage.blockers)}

    return {
        "dataset_freeze": value("dataset-freeze"),
        "training": value("training-campaign"),
        "promotion": value("promotion-decision"),
    }


def _active_runs(config: ProjectConfig) -> list[dict[str, Any]]:
    terminal = {"COMPLETE", "FAILED", "BLOCKED", "CANCELLED"}
    try:
        runs_dir = config.runs_dir
    except KeyError:
        return []
    return [run for run in list_runs(runs_dir) if str(run.get("status", "")).upper() not in terminal]


def _recommended_action(state: ProjectState, audits: list[dict[str, Any]]) -> dict[str, Any]:
    for audit in audits:
        if not audit["current_certification"]:
            stage = state.stage(
                {
                    "semantic-labeling": "semantic-labeling",
                    "training-infrastructure": "training-infrastructure-audit",
                    "memorization": "memorization-review",
                }[audit["subsystem"]]
            )
            return {
                "subsystem": audit["subsystem"],
                "action": stage.next_action,
                "command": f"python -m spritelab dev explain {audit['subsystem']}",
                "reason": audit["authorization_consequence"],
            }
    for stage in state.stages:
        if stage.blockers:
            return {
                "subsystem": stage.key,
                "action": stage.next_action,
                "command": stage.next_command,
                "reason": stage.blockers[0],
            }
    return {
        "subsystem": "repository",
        "action": "Run the quick developer test profile.",
        "command": "python -m spritelab dev test quick",
        "reason": "No blocking developer evidence was found.",
    }


def build_developer_state(config: ProjectConfig, state: ProjectState) -> dict[str, Any]:
    repository = repository_state(config.root)
    audits = collect_audits(config, state)
    artifacts = inspect_artifacts(config, state)
    failed_gates = [
        {"subsystem": audit["subsystem"], "gates": audit["failed_gates"]} for audit in audits if audit["failed_gates"]
    ]
    return {
        "schema_version": DEVELOPER_STATUS_SCHEMA,
        "project": state.project_name,
        "project_root": str(state.project_root),
        "config_path": str(state.config_path) if state.config_path else None,
        "generated_at": state.generated_at,
        "repository": repository,
        "subsystems": [_stage_record(stage) for stage in state.stages],
        "audits": audits,
        "failed_gates": failed_gates,
        "artifacts": artifacts,
        "artifact_identities": [item for item in artifacts if item["sha256"]],
        "dataset_freeze_identities": _freeze_identities(config),
        "authorization": _authorization(state),
        "active_developer_runs": _active_runs(config),
        "recommended_action": _recommended_action(state, audits),
        "warnings": list(state.warnings),
    }
