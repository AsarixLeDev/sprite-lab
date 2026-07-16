"""Typed state shared by v3 status, orchestration, rendering, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "spritelab.v3.result.v1"


class StageStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"
    INCONCLUSIVE = "INCONCLUSIVE"
    STALE = "STALE"


class AuditStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    STALE = "STALE"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    NOT_AUDITED = "NOT AUDITED"


class ExitCode(IntEnum):
    SUCCESS = 0
    INTERNAL_ERROR = 1
    INVALID = 2
    BLOCKED = 3
    REVIEW_REQUIRED = 4
    PAUSED = 5
    STALE = 6
    DOCTOR_FAILED = 7


@dataclass(frozen=True)
class Evidence:
    path: str
    sha256: str | None = None
    source_commit: str | None = None


@dataclass
class StageState:
    key: str
    title: str
    status: StageStatus
    explanation: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    source_commit: str | None = None
    next_action: str = "Inspect project status."
    next_command: str = "python -m spritelab v3 status"
    resume_available: bool = False
    audit: AuditStatus = AuditStatus.NOT_AUDITED
    implementation: str = "AVAILABLE"
    production_authorized: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["audit"] = self.audit.value
        return value


@dataclass
class ProjectState:
    project_name: str
    project_root: Path
    config_path: Path | None
    source_commit: str | None
    stages: list[StageState]
    warnings: list[str] = field(default_factory=list)
    generated_at: str | None = None

    def stage(self, key: str) -> StageState:
        normalized = key.lower().replace("_", "-").replace(" ", "-")
        aliases = {
            "provenance": "raw-source-provenance",
            "raw": "raw-source-provenance",
            "labeling": "semantic-labeling",
            "calibration": "semantic-calibration",
            "freeze": "dataset-freeze",
            "training": "training-campaign",
            "training-audit": "training-infrastructure-audit",
            "evaluation": "evaluation-generation",
            "memorization": "memorization-review",
            "promotion": "promotion-decision",
        }
        normalized = aliases.get(normalized, normalized)
        for item in self.stages:
            if item.key == normalized:
                return item
        raise KeyError(key)

    @property
    def blockers(self) -> list[str]:
        return [reason for stage in self.stages for reason in stage.blockers]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.v3.project-state.v1",
            "project": self.project_name,
            "project_root": str(self.project_root),
            "config_path": str(self.config_path) if self.config_path else None,
            "source_commit": self.source_commit,
            "generated_at": self.generated_at,
            "warnings": self.warnings,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_product_dict(self) -> dict[str, Any]:
        """Return a plain product view without hashes, commits, or audit verdicts."""

        groups = (
            (
                "dataset",
                "Dataset",
                self.stages[:7],
                "Dataset preparation is ready.",
                "Dataset preparation needs attention before it can continue.",
                "python -m spritelab v3 dataset",
            ),
            (
                "training",
                "Training",
                self.stages[7:9],
                "Training is ready or complete.",
                "Training is waiting for required project checks or inputs.",
                "python -m spritelab v3 train",
            ),
            (
                "evaluation",
                "Evaluation",
                self.stages[9:12],
                "Evaluation is ready or complete.",
                "Evaluation is waiting for required project checks or inputs.",
                "python -m spritelab v3 eval",
            ),
            (
                "release",
                "Project result",
                self.stages[12:],
                "The project result is ready.",
                "The project result is not ready yet.",
                "python -m spritelab v3 status",
            ),
        )
        product_stages = []
        for key, title, stages, complete_message, blocked_message, next_command in groups:
            status = _product_group_status(stages)
            attention = status in {"BLOCKED", "FAILED", "NEEDS_REVIEW", "UNAVAILABLE"}
            explanation = blocked_message if attention else complete_message
            if key == "dataset":
                explanation = _dataset_labeling_message(stages)
            product_stages.append(
                {
                    "key": key,
                    "title": title,
                    "status": status,
                    "explanation": explanation,
                    "blockers": [explanation] if attention else [],
                    "warnings": [],
                    "evidence": [],
                    "next_action": explanation,
                    "next_command": next_command,
                }
            )
        return {
            "schema_version": "spritelab.product.project-status.v1",
            "project": self.project_name,
            "generated_at": self.generated_at,
            "status": _product_group_status(self.stages),
            "warnings": ["Project warnings are available in developer status."] if self.warnings else [],
            "stages": product_stages,
        }


@dataclass
class CommandResult:
    command: str
    status: str
    exit_code: ExitCode
    message: str
    project_state: ProjectState | None = None
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_command: str | None = None
    run_id: str | None = None
    report_path: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    internal_details: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "command": self.command,
            "status": self.status,
            "exit_code": int(self.exit_code),
            "message": self.message,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "next_command": self.next_command,
            "run_id": self.run_id,
            "report_path": self.report_path,
            "project_state": (
                self.project_state.to_dict() if self.internal_details else self.project_state.to_product_dict()
            )
            if self.project_state
            else None,
            "data": self.data,
        }


def _product_group_status(stages: list[StageState]) -> str:
    if not stages:
        return "UNAVAILABLE"
    mapped = {
        StageStatus.INCONCLUSIVE: "NEEDS_REVIEW",
        StageStatus.STALE: "BLOCKED",
    }
    values = [mapped.get(stage.status, stage.status.value) for stage in stages]
    for candidate in (
        "FAILED",
        "BLOCKED",
        "NEEDS_REVIEW",
        "RUNNING",
        "PAUSED",
        "READY",
        "NOT_STARTED",
        "COMPLETE",
    ):
        if candidate in values:
            return candidate
    return "UNAVAILABLE"


def _dataset_labeling_message(stages: list[StageState]) -> str:
    labeling = next((stage for stage in stages if stage.key == "semantic-labeling"), None)
    if labeling is None:
        return "Image preparation is available. Automatic image descriptions are waiting for a reliability check."
    scopes = set(labeling.metrics.get("authorized_scopes", ()))
    if labeling.audit is AuditStatus.PASS and "conservative_proposal_generation" in scopes:
        return (
            "Automatic image descriptions are available for broad suggestions. "
            "Exact object labels still require more verification."
        )
    if labeling.audit is AuditStatus.FAIL:
        return "Automatic image descriptions are not available because the latest reliability check found problems."
    return "Automatic image descriptions are waiting for a reliability check."
