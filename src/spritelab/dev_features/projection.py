"""Allowlisted projection from detailed developer state to simple product state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

USER_PROJECTION_SCHEMA = "spritelab.dev.user-projection.v1"


def _subsystems(details: Mapping[str, Any], keys: set[str]) -> list[Mapping[str, Any]]:
    values = details.get("subsystems", [])
    if not isinstance(values, Sequence):
        return []
    return [item for item in values if isinstance(item, Mapping) and item.get("key") in keys]


def _simple_status(stages: list[Mapping[str, Any]], *, authorized: bool | None = None) -> tuple[str, str]:
    if authorized is False:
        return "UNAVAILABLE", "Not available yet"
    statuses = {str(stage.get("status", "")) for stage in stages}
    if "RUNNING" in statuses:
        return "RUNNING", "In progress"
    if statuses and statuses <= {"COMPLETE"}:
        return "COMPLETE", "Complete"
    if statuses & {"NEEDS_REVIEW", "INCONCLUSIVE"}:
        return "NEEDS_REVIEW", "Needs your review"
    if statuses & {"FAILED", "BLOCKED", "STALE"}:
        return "UNAVAILABLE", "Not available yet"
    if "READY" in statuses:
        return "READY", "Ready"
    return "NOT_STARTED", "Not started"


def project_user_status(details: Mapping[str, Any]) -> dict[str, Any]:
    """Return an allowlisted user view that cannot leak hashes or repository identities."""

    authorization = details.get("authorization", {})
    authorization = authorization if isinstance(authorization, Mapping) else {}
    groups = (
        (
            "dataset",
            "Dataset",
            {
                "raw-source-provenance",
                "extraction",
                "suitability",
                "semantic-labeling",
                "semantic-calibration",
                "dataset-v5-view-construction",
                "dataset-freeze",
            },
            None,
        ),
        (
            "training",
            "Training",
            {"training-infrastructure-audit", "training-campaign"},
            _authorized(authorization, "training"),
        ),
        (
            "evaluation",
            "Evaluation",
            {"evaluation-generation", "evaluation-metrics", "memorization-review"},
            None,
        ),
        (
            "result",
            "Project result",
            {"promotion-decision"},
            _authorized(authorization, "promotion"),
        ),
    )
    areas = []
    for key, title, stage_keys, authorized in groups:
        status, message = _simple_status(_subsystems(details, stage_keys), authorized=authorized)
        areas.append({"key": key, "title": title, "status": status, "message": message})
    return {
        "schema_version": USER_PROJECTION_SCHEMA,
        "project": str(details.get("project", "Sprite Lab")),
        "areas": areas,
    }


def _authorized(authorization: Mapping[str, Any], key: str) -> bool | None:
    value = authorization.get(key)
    if not isinstance(value, Mapping) or not isinstance(value.get("authorized"), bool):
        return None
    return bool(value["authorized"])
