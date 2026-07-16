"""Additive product preparation and truth-aware status projection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.hierarchical_labeling.renders import RENDER_POLICIES, build_render_bundle
from spritelab.hierarchical_labeling.taxonomy import load_default_taxonomy
from spritelab.hierarchical_labeling.technical import extract_technical_evidence
from spritelab.v3.run_state import atomic_write_json


def labeling_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("labeling", {}) if isinstance(config, Mapping) else {}
    values = raw if isinstance(raw, Mapping) else {}
    return {
        "enabled": values.get("hierarchical_enabled") is True,
        "profile": str(values.get("hierarchical_profile", "fast_local")),
        "reference_cohort_size": int(values.get("reference_cohort_size", 400)),
        "run_root": str(values.get("hierarchical_run_root", "runs/v3/hierarchical-labeling")),
        "report": str(values.get("hierarchical_report", "")),
    }


def prepare_configured_labeling(
    items: Sequence[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Run deterministic stages after intake; semantic stages remain proposals.

    Provider execution stays in the configured cascade.  This intake hook
    deliberately cannot certify labels or condition a dataset.
    """

    settings = labeling_settings(config)
    if not settings["enabled"]:
        return {
            "schema_version": "spritelab.labeling.product-preparation.v1",
            "enabled": False,
            "profile": settings["profile"],
            "status": "disabled",
            "processed": 0,
            "provider_calls": 0,
            "human_labels_created": 0,
            "conditioned_dataset_ready": False,
        }
    graph = load_default_taxonomy()
    profile = settings["profile"]
    policy = RENDER_POLICIES[profile]
    root = Path(output_root) / "hierarchical_labeling"
    completed = 0
    failures: list[dict[str, str]] = []
    for item in items:
        if item.get("current_disposition") != "accepted":
            continue
        try:
            record_identity = str(item.get("item_id") or item.get("relative_path") or "")
            technical = extract_technical_evidence(item["source_path"], record_identity=record_identity)
            renders = build_render_bundle(
                item["source_path"],
                technical,
                root / "renders" / technical.image_identity[:16],
                policy=policy,
            )
            manifest = {
                "schema_version": "spritelab.labeling.product-item-preparation.v1",
                "record_identity": record_identity,
                "technical": technical.to_dict(),
                "render_bundle": renders.to_dict(),
                "taxonomy_identity": graph.identity,
                "profile": profile,
                "semantic_state": "awaiting_strict_description_or_review",
                "truth_status": "not_human_truth",
            }
            artifact = root / "artifacts" / f"{technical.image_identity}.json"
            atomic_write_json(artifact, manifest)
            item["hierarchical_labeling"] = {
                "state": "prepared",
                "profile": profile,
                "technical_evidence_identity": technical.identity,
                "render_bundle_identity": renders.identity,
                "taxonomy_identity": graph.identity,
                "truth_status": "not_human_truth",
                "conditioned_dataset_ready": False,
            }
            completed += 1
        except (OSError, ValueError, KeyError) as exc:
            item["hierarchical_labeling"] = {
                "state": "preparation_failed",
                "truth_status": "unavailable",
                "conditioned_dataset_ready": False,
            }
            failures.append(
                {
                    "record_identity": str(item.get("item_id") or item.get("relative_path") or "unknown"),
                    "error_type": type(exc).__name__,
                }
            )
    summary = {
        "schema_version": "spritelab.labeling.product-preparation.v1",
        "enabled": True,
        "profile": profile,
        "status": "prepared" if not failures else "partial",
        "processed": completed,
        "failures": failures,
        "provider_calls": 0,
        "human_labels_created": 0,
        "calibration_state": "not_ready",
        "conditioned_dataset_ready": False,
        "next_action": "Run automatic suggestions, review a reference cohort, then calibrate.",
    }
    atomic_write_json(root / "preparation_summary.json", summary)
    return summary


def product_status(config: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    settings = labeling_settings(config)
    report_path = Path(settings["report"]).expanduser() if settings["report"] else None
    if report_path is not None and not report_path.is_absolute():
        report_path = Path(project_root) / report_path
    report = None
    if report_path and report_path.is_file():
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
            report = value if isinstance(value, Mapping) else None
        except (OSError, json.JSONDecodeError):
            report = None
    truth = report.get("truth_metrics", {}) if report else {}
    summary = report.get("summary", {}) if report else {}
    truth_size = int(truth.get("sample_size", 0)) if truth.get("state") == "measured_from_human_truth" else 0
    curve = truth.get("precision_coverage_curve") if truth_size else None
    measured = curve[0] if isinstance(curve, list) and curve else None
    precision = measured.get("precision") if isinstance(measured, Mapping) else None
    coverage = measured.get("coverage") if isinstance(measured, Mapping) else summary.get("accepted_coverage")
    return {
        "schema_version": "spritelab.labeling.product-status.v1",
        "enabled": settings["enabled"],
        "profile": settings["profile"],
        "cards": [
            {"key": "image_preparation", "title": "Image preparation", "status": "READY"},
            {
                "key": "reference_truth",
                "title": "Reference truth",
                "status": "READY" if truth_size else "NOT_STARTED",
                "reviewed_records": truth_size,
            },
            {
                "key": "automatic_reliability",
                "title": "Automatic-label reliability",
                "status": "MEASURED" if precision is not None else "NOT_READY",
                "held_out_precision": precision,
                "sample_size": truth_size,
                "truth_source": "human_review" if truth_size else None,
            },
            {
                "key": "accepted_coverage",
                "title": "Accepted semantic coverage",
                "status": "MEASURED" if coverage is not None and truth_size else "NOT_READY",
                "coverage": coverage if truth_size else None,
            },
            {
                "key": "calibration",
                "title": "Calibration",
                "status": str(summary.get("calibration_state", "not_ready")).upper(),
            },
            {
                "key": "conditioned_readiness",
                "title": "Conditioned dataset readiness",
                "status": "NOT_AUTHORIZED",
                "message": "Architecture availability does not authorize conditioned production freezing.",
            },
        ],
        "precision_claim_suppressed_without_truth": precision is None,
        "production_authorized": False,
    }
