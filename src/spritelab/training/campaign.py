"""Versioned, fixed-step training campaign planning and orchestration.

This module deliberately stays above the single-run experiment system.  It does
not import torch or model code; process launching is isolated behind an
injectable runner so planning, validation, status, and tests are CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

CAMPAIGN_SCHEMA_VERSION = "spritelab_training_campaign_v1"
DEFAULT_SEEDS = (731001, 731002, 731003)
REQUIRED_SEED_COUNT = 3

REQUIRED_IDENTITY_HASHES = (
    "dataset_view_manifest_hash",
    "split_manifest_hash",
    "model_config_hash",
    "conditioning_vocabulary_hash",
    "optimizer_config_hash",
    "schedule_config_hash",
    "loss_config_hash",
    "determinism_config_hash",
)

PER_RUN_ARTIFACTS = (
    "experiment_manifest",
    "resolved_config",
    "checkpoint_series",
    "checkpoint_hash_map",
    "training_metrics",
    "validation_metrics",
    "ema_metrics",
    "live_metrics",
    "evaluation_reports",
    "effective_pass_report",
    "resume_report",
    "environment_identity",
    "code_identity",
    "run_completion_marker",
)

CAMPAIGN_ARTIFACTS = (
    "campaign_manifest",
    "run_matrix",
    "fairness_validation",
    "evaluation_schedule",
    "checkpoint_schedule",
    "run_status_summary",
    "cross_seed_aggregate",
    "artifact_hash_map",
)

# The paths are relative to each fully resolved comparison cell.  A differing
# path is allowed only when that exact path (or its leaf name) is declared as an
# experimental variable.
PROTECTED_COMPARISON_FIELDS = (
    "training.max_optimizer_steps",
    "training.micro_batch_size",
    "training.gradient_accumulation",
    "training.effective_batch_size",
    "optimizer",
    "schedule",
    "training.precision",
    "training.sampler_policy",
    "identities.dataset_view_manifest_hash",
    "evaluation.benchmark_manifest_hash",
    "evaluation.cadence",
    "checkpoint.cadence",
    "evaluation.cfg_value",
    "evaluation.sampling_steps",
    "evaluation.ema_policy",
    "determinism",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PLACEHOLDER_RE = re.compile(r"(?:unresolved|placeholder|replace[_ -]?me|tbd|todo|unknown|pending)", re.I)


class CampaignValidationError(ValueError):
    """Raised when a campaign is not safe to execute."""


class CampaignResumeError(CampaignValidationError):
    """Raised when existing output cannot be resumed safely."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for stable identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_concrete_hash(value: Any) -> bool:
    """Return true only for a concrete SHA-256 identity, never a placeholder."""

    text = str(value or "").strip()
    return bool(_SHA256_RE.fullmatch(text)) and not _PLACEHOLDER_RE.search(text)


def evaluation_steps(max_optimizer_steps: int, cadence: int, *, include_step_zero: bool = False) -> list[int]:
    """Build an exact fixed-step evaluation schedule, always including final."""

    return _fixed_step_schedule(max_optimizer_steps, cadence, include_step_zero=include_step_zero)


def checkpoint_steps(max_optimizer_steps: int, cadence: int) -> list[int]:
    """Build an exact checkpoint schedule, always including final."""

    return _fixed_step_schedule(max_optimizer_steps, cadence, include_step_zero=False)


def _fixed_step_schedule(max_steps: int, cadence: int, *, include_step_zero: bool) -> list[int]:
    if isinstance(max_steps, bool) or int(max_steps) <= 0:
        raise ValueError("max optimizer steps must be a positive integer; fixed-epoch fallback is forbidden")
    if isinstance(cadence, bool) or int(cadence) <= 0:
        raise ValueError("cadence must be a positive integer")
    maximum, interval = int(max_steps), int(cadence)
    result = [0] if include_step_zero else []
    result.extend(range(interval, maximum + 1, interval))
    if result[-1:] != [maximum]:
        result.append(maximum)
    return result


def effective_pass_report(
    *,
    optimizer_steps: int,
    effective_batch_size: int,
    positive_sampling_mass_records: float,
    nominal_record_count: int | None = None,
    positive_weight_record_count: int | None = None,
    positive_weight_sum: float | None = None,
) -> dict[str, Any]:
    """Report transparent draw-to-dataset ratios under several denominators.

    These are exposure ratios, not guarantees that every record was observed.
    """

    draws = int(optimizer_steps) * int(effective_batch_size)
    if draws < 0:
        raise ValueError("optimizer_steps and effective_batch_size must be non-negative")
    if float(positive_sampling_mass_records) <= 0:
        raise ValueError("positive_sampling_mass_records must be positive")
    nominal = int(nominal_record_count or 0)
    positive_count = int(positive_weight_record_count or 0)
    weight_sum = float(positive_weight_sum or positive_sampling_mass_records)
    return {
        "optimizer_steps": int(optimizer_steps),
        "effective_batch_size": int(effective_batch_size),
        "optimizer_record_draws": draws,
        "effective_dataset_passes": draws / float(positive_sampling_mass_records),
        "nominal_record_passes": None if nominal <= 0 else draws / nominal,
        "positive_weight_record_passes": None if positive_count <= 0 else draws / positive_count,
        "expected_weighted_exposure_mass": None if weight_sum <= 0 else draws / weight_sum,
        "formulas": {
            "effective_dataset_passes": ("optimizer_steps * effective_batch_size / positive_sampling_mass_records"),
            "nominal_record_passes": "optimizer_steps * effective_batch_size / nominal_record_count",
            "positive_weight_record_passes": ("optimizer_steps * effective_batch_size / positive_weight_record_count"),
            "expected_weighted_exposure_mass": (
                "optimizer_steps * effective_batch_size / sum_of_positive_sampling_weights"
            ),
        },
        "definitions": {
            "positive_sampling_mass_records": (
                "the campaign-declared record-equivalent mass used as the primary weighted-sampling denominator"
            ),
            "nominal_record_count": "all nominal records, including records with zero sampling probability",
            "positive_weight_record_count": "records whose sampling weight is strictly positive",
            "sum_of_positive_sampling_weights": "sum of all strictly positive, pre-normalization sampler weights",
        },
        "interpretation_warning": (
            "Weighted exposure ratios are expectations over draws; they do not mean every record was observed."
        ),
    }


def campaign_schema() -> dict[str, Any]:
    """Return a JSON-Schema-shaped description of the v1 manifest contract."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": CAMPAIGN_SCHEMA_VERSION,
        "title": "Sprite Lab fixed-step training campaign",
        "type": "object",
        "required": [
            "schema_version",
            "campaign_id",
            "purpose",
            "architecture_cells",
            "identities",
            "seeds",
            "training",
            "optimizer",
            "schedule",
            "loss",
            "determinism",
            "evaluation",
            "checkpoint",
            "expected_runs",
            "expected_artifact_contract",
            "abort_conditions",
            "promotion_restrictions",
            "plan_status",
            "executable",
        ],
        "properties": {
            "schema_version": {"const": CAMPAIGN_SCHEMA_VERSION},
            "campaign_id": {"type": "string", "minLength": 1},
            "seeds": {
                "type": "array",
                "minItems": REQUIRED_SEED_COUNT,
                "maxItems": REQUIRED_SEED_COUNT,
                "uniqueItems": True,
                "items": {"type": "integer"},
            },
            "experimental_variables": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "fixed_epoch_fallback": {"const": False},
            "executable": {"type": "boolean"},
            "baseline_launch_authorized": {"type": "boolean"},
        },
        "additionalProperties": True,
    }


def plan_campaign(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a deterministic v1 campaign manifest from an input specification."""

    raw = deepcopy(dict(spec))
    campaign_id = str(raw.get("campaign_id") or "").strip()
    cells = _normalise_cells(raw.get("architecture_cells", []))
    seeds = [int(seed) for seed in raw.get("seeds", DEFAULT_SEEDS)]
    training = deepcopy(dict(raw.get("training") or {}))
    evaluation = deepcopy(dict(raw.get("evaluation") or {}))
    checkpoint = deepcopy(dict(raw.get("checkpoint") or {}))
    identities = deepcopy(dict(raw.get("identities") or {}))

    micro_batch = int(training.get("micro_batch_size", 0) or 0)
    accumulation = int(training.get("gradient_accumulation", 0) or 0)
    calculated_effective = micro_batch * accumulation
    training.setdefault("effective_batch_size", calculated_effective)
    training["effective_batch_size_formula"] = "micro_batch_size * gradient_accumulation"
    max_steps = int(training.get("max_optimizer_steps", 0) or 0)
    eval_cadence = int(evaluation.get("cadence", 0) or 0)
    checkpoint_cadence = int(checkpoint.get("cadence", 0) or 0)
    evaluation.setdefault("include_step_zero", False)
    evaluation.setdefault("ema_policy", "both")
    evaluation.setdefault("live_weight_evaluation_policy", "required")
    checkpoint.setdefault("require_resumability_metadata", True)

    expected_runs: list[dict[str, Any]] = []
    eval_matrix: dict[str, list[int]] = {}
    checkpoint_matrix: dict[str, list[int]] = {}
    schedule_errors: list[str] = []
    try:
        eval_schedule = evaluation_steps(
            max_steps, eval_cadence, include_step_zero=bool(evaluation["include_step_zero"])
        )
    except (TypeError, ValueError) as exc:
        eval_schedule = []
        schedule_errors.append(f"evaluation schedule: {exc}")
    try:
        save_schedule = checkpoint_steps(max_steps, checkpoint_cadence)
    except (TypeError, ValueError) as exc:
        save_schedule = []
        schedule_errors.append(f"checkpoint schedule: {exc}")

    output_base = Path(str(raw.get("output_root") or "experiments/training_campaign_runs"))
    for cell in cells:
        cell_id = cell["cell_id"]
        for seed in seeds:
            run_id = f"{campaign_id}__{cell_id}__seed_{seed}"
            output_root = output_base / campaign_id / cell_id / f"seed_{seed}"
            resolved_config = output_base / campaign_id / "resolved_configs" / f"{cell_id}__seed_{seed}.json"
            run = {
                "run_id": run_id,
                "campaign_id": campaign_id,
                "cell_id": cell_id,
                "seed": seed,
                "output_root": str(output_root),
                "resolved_config_path": str(resolved_config),
                "experiment_command": [
                    sys.executable,
                    "-m",
                    "spritelab",
                    "train",
                    "experiment",
                    "run",
                    "--config",
                    str(resolved_config),
                ],
                "expected_evaluation_steps": eval_schedule,
                "expected_checkpoint_steps": save_schedule,
                "effective_passes": _passes_from_training(training, max_steps),
                "expected_artifacts": list(PER_RUN_ARTIFACTS),
            }
            run["run_identity"] = stable_hash(_run_identity_payload(raw, cell, run, training, evaluation, checkpoint))
            expected_runs.append(run)
            eval_matrix[run_id] = eval_schedule
            checkpoint_matrix[run_id] = save_schedule

    manifest: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "purpose": str(raw.get("purpose") or ""),
        "architecture_cell_ids": [cell["cell_id"] for cell in cells],
        "architecture_cells": cells,
        "experimental_variables": sorted(set(map(str, raw.get("experimental_variables", [])))),
        "identities": identities,
        "seeds": seeds,
        "training": training,
        "optimizer": deepcopy(raw.get("optimizer")),
        "schedule": deepcopy(raw.get("schedule")),
        "loss": deepcopy(raw.get("loss")),
        "determinism": deepcopy(raw.get("determinism")),
        "evaluation": evaluation,
        "checkpoint": checkpoint,
        "evaluation_schedule": eval_matrix,
        "checkpoint_schedule": checkpoint_matrix,
        "expected_run_ids": [run["run_id"] for run in expected_runs],
        "expected_output_roots": [run["output_root"] for run in expected_runs],
        "expected_runs": expected_runs,
        "expected_artifact_contract": {
            "per_run": list(PER_RUN_ARTIFACTS),
            "campaign": list(CAMPAIGN_ARTIFACTS),
            "completion_requires_all": True,
        },
        "abort_conditions": list(raw.get("abort_conditions") or _default_abort_conditions()),
        "promotion_restrictions": list(raw.get("promotion_restrictions") or _default_promotion_restrictions()),
        "fixed_epoch_fallback": False,
        "baseline_launch_authorized": bool(raw.get("baseline_launch_authorized", False)),
        "plan_status": "blocked",
        "executable": False,
        "blockers": schedule_errors,
    }
    manifest["campaign_identity"] = stable_hash(_campaign_identity_payload(manifest))
    validation = validate_campaign(manifest)
    blockers = sorted({*schedule_errors, *validation["errors"], *validation["blockers"]})
    manifest["blockers"] = blockers
    requested_executable = bool(raw.get("executable", False))
    manifest["executable"] = requested_executable and not blockers
    manifest["plan_status"] = "ready" if manifest["executable"] else "blocked"
    # Recompute after status resolution; identity deliberately excludes mutable status fields.
    manifest["campaign_identity"] = stable_hash(_campaign_identity_payload(manifest))
    return manifest


def validate_campaign(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, production identities, schedules, and comparison fairness."""

    errors: list[str] = []
    blockers: list[str] = []
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {CAMPAIGN_SCHEMA_VERSION!r}")
    if not str(campaign.get("campaign_id") or "").strip():
        errors.append("campaign_id is required")
    if not str(campaign.get("purpose") or "").strip():
        errors.append("purpose is required")
    seeds = list(campaign.get("seeds") or [])
    if len(seeds) != REQUIRED_SEED_COUNT:
        errors.append(f"exactly {REQUIRED_SEED_COUNT} seeds are required; found {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        errors.append("seeds must be unique")
    cells = list(campaign.get("architecture_cells") or [])
    cell_ids = [str(cell.get("cell_id") or "") for cell in cells if isinstance(cell, Mapping)]
    if not cells or any(not cell_id for cell_id in cell_ids):
        errors.append("at least one named architecture cell is required")
    if len(set(cell_ids)) != len(cell_ids):
        errors.append("architecture cell IDs must be unique")

    identities = dict(campaign.get("identities") or {})
    for field in REQUIRED_IDENTITY_HASHES:
        value = identities.get(field)
        if not is_concrete_hash(value):
            blockers.append(f"identity {field} is missing, placeholder, or not a concrete SHA-256")
    evaluation = dict(campaign.get("evaluation") or {})
    if not is_concrete_hash(evaluation.get("benchmark_manifest_hash")):
        blockers.append("evaluation benchmark_manifest_hash is missing, placeholder, or not a concrete SHA-256")
    if not is_concrete_hash(evaluation.get("evaluation_config_hash")):
        blockers.append("evaluation evaluation_config_hash is missing, placeholder, or not a concrete SHA-256")
    if evaluation.get("ema_policy") not in {"ema", "live", "both"}:
        errors.append("evaluation ema_policy must be one of: ema, live, both")
    if evaluation.get("ema_policy") == "both" and evaluation.get("live_weight_evaluation_policy") != "required":
        errors.append("ema_policy 'both' requires live_weight_evaluation_policy='required'")

    training = dict(campaign.get("training") or {})
    micro = int(training.get("micro_batch_size", 0) or 0)
    accumulation = int(training.get("gradient_accumulation", 0) or 0)
    effective = int(training.get("effective_batch_size", 0) or 0)
    if micro <= 0 or accumulation <= 0 or effective <= 0:
        errors.append("micro batch, gradient accumulation, and effective batch must be positive")
    elif micro * accumulation != effective:
        errors.append(
            f"effective batch mismatch: {micro} * {accumulation} = {micro * accumulation}, declared {effective}"
        )
    if "max_optimizer_steps" not in training or int(training.get("max_optimizer_steps", 0) or 0) <= 0:
        errors.append("positive max_optimizer_steps is required; fixed-epoch fallback is forbidden")
    if float(training.get("positive_sampling_mass_records", 0) or 0) <= 0:
        errors.append("positive_sampling_mass_records must be positive")
    for field in ("precision", "sampler_policy"):
        if not str(training.get(field) or "").strip():
            errors.append(f"training {field} is required")
    for field in ("optimizer", "schedule", "loss", "determinism"):
        if not isinstance(campaign.get(field), Mapping) or not campaign.get(field):
            errors.append(f"{field} configuration is required")
    if campaign.get("fixed_epoch_fallback") not in {False, None}:
        errors.append("fixed-epoch fallback is forbidden")
    try:
        expected_eval = evaluation_steps(
            int(training.get("max_optimizer_steps", 0)),
            int(evaluation.get("cadence", 0)),
            include_step_zero=bool(evaluation.get("include_step_zero", False)),
        )
        expected_checkpoints = checkpoint_steps(
            int(training.get("max_optimizer_steps", 0)), int(dict(campaign.get("checkpoint") or {}).get("cadence", 0))
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        expected_eval, expected_checkpoints = [], []
    for run in campaign.get("expected_runs") or []:
        if list(run.get("expected_evaluation_steps") or []) != expected_eval:
            errors.append(f"run {run.get('run_id')} has a missing or altered evaluation schedule point")
        if list(run.get("expected_checkpoint_steps") or []) != expected_checkpoints:
            errors.append(f"run {run.get('run_id')} has a missing or altered checkpoint schedule point")

    fairness = validate_fixed_step_fairness(campaign)
    errors.extend(fairness["errors"])
    expected_count = len(cells) * len(seeds)
    runs = list(campaign.get("expected_runs") or [])
    if len(runs) != expected_count:
        errors.append(f"expected_runs must contain {expected_count} cell/seed runs; found {len(runs)}")
    run_ids = [str(run.get("run_id") or "") for run in runs]
    roots = [str(run.get("output_root") or "") for run in runs]
    if len(set(run_ids)) != len(run_ids):
        errors.append("expected run IDs must be unique")
    if len(set(roots)) != len(roots):
        errors.append("expected output roots must be unique")
    if not campaign.get("abort_conditions"):
        errors.append("abort_conditions must be bound")
    if not campaign.get("promotion_restrictions"):
        errors.append("promotion_restrictions must be bound")
    contract = dict(campaign.get("expected_artifact_contract") or {})
    if set(contract.get("per_run") or []) != set(PER_RUN_ARTIFACTS):
        errors.append("per-run artifact contract is incomplete")
    if set(contract.get("campaign") or []) != set(CAMPAIGN_ARTIFACTS):
        errors.append("campaign artifact contract is incomplete")
    stored_campaign_identity = campaign.get("campaign_identity")
    if stored_campaign_identity and stored_campaign_identity != stable_hash(_campaign_identity_payload(campaign)):
        errors.append("campaign identity does not match manifest content")
    for run in runs:
        if run.get("campaign_id") != campaign.get("campaign_id"):
            errors.append(f"run {run.get('run_id')} campaign_id does not match")
        cell = next((item for item in cells if item.get("cell_id") == run.get("cell_id")), None)
        if cell is None:
            errors.append(f"run {run.get('run_id')} references an unknown architecture cell")
            continue
        expected_identity = stable_hash(
            _run_identity_payload(campaign, cell, run, training, evaluation, dict(campaign.get("checkpoint") or {}))
        )
        if run.get("run_identity") != expected_identity:
            errors.append(f"run {run.get('run_id')} identity does not match resolved campaign settings")
    return {
        "schema_version": "spritelab_campaign_validation_v1",
        "campaign_id": campaign.get("campaign_id"),
        "valid": not errors,
        "launch_ready": not errors and not blockers,
        "errors": errors,
        "blockers": blockers,
        "fairness": fairness,
    }


def validate_fixed_step_fairness(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Report every protected and undeclared comparison-cell mismatch."""

    cells = list(campaign.get("architecture_cells") or [])
    declared = set(map(str, campaign.get("experimental_variables") or []))
    mismatches: list[dict[str, Any]] = []
    if len(cells) < 2:
        return {
            "schema_version": "spritelab_fixed_step_fairness_v1",
            "fair": True,
            "declared_experimental_variables": sorted(declared),
            "protected_fields": list(PROTECTED_COMPARISON_FIELDS),
            "mismatches": [],
            "errors": [],
        }
    baseline = _resolved_cell(campaign, cells[0])
    baseline_id = str(cells[0].get("cell_id"))
    for cell in cells[1:]:
        current = _resolved_cell(campaign, cell)
        cell_id = str(cell.get("cell_id"))
        for path in PROTECTED_COMPARISON_FIELDS:
            before, after = _get_dotted(baseline, path), _get_dotted(current, path)
            if before != after:
                allowed = _is_declared(path, declared)
                mismatches.append(
                    {
                        "baseline_cell_id": baseline_id,
                        "cell_id": cell_id,
                        "field": path,
                        "baseline_value": before,
                        "cell_value": after,
                        "declared_experimental_variable": allowed,
                        "protected": True,
                    }
                )
        base_values = dict(cells[0].get("comparison_values") or {})
        cell_values = dict(cell.get("comparison_values") or {})
        for name in sorted(set(base_values) | set(cell_values)):
            if base_values.get(name) != cell_values.get(name) and not _is_declared(name, declared):
                mismatches.append(
                    {
                        "baseline_cell_id": baseline_id,
                        "cell_id": cell_id,
                        "field": name,
                        "baseline_value": base_values.get(name),
                        "cell_value": cell_values.get(name),
                        "declared_experimental_variable": False,
                        "protected": False,
                    }
                )
        protected = set(PROTECTED_COMPARISON_FIELDS)
        flat_baseline = _flatten_leaves(baseline)
        flat_current = _flatten_leaves(current)
        for path in sorted(set(flat_baseline) | set(flat_current)):
            if flat_baseline.get(path) == flat_current.get(path):
                continue
            if any(path == item or path.startswith(f"{item}.") or item.startswith(f"{path}.") for item in protected):
                continue
            if _is_declared(path, declared):
                continue
            mismatches.append(
                {
                    "baseline_cell_id": baseline_id,
                    "cell_id": cell_id,
                    "field": path,
                    "baseline_value": flat_baseline.get(path),
                    "cell_value": flat_current.get(path),
                    "declared_experimental_variable": False,
                    "protected": False,
                }
            )
    errors = [
        f"fairness mismatch {row['baseline_cell_id']} vs {row['cell_id']}: {row['field']} "
        f"({row['baseline_value']!r} != {row['cell_value']!r})"
        for row in mismatches
        if not row["declared_experimental_variable"]
    ]
    return {
        "schema_version": "spritelab_fixed_step_fairness_v1",
        "fair": not errors,
        "declared_experimental_variables": sorted(declared),
        "protected_fields": list(PROTECTED_COMPARISON_FIELDS),
        "mismatches": mismatches,
        "errors": errors,
    }


def audit_resume(campaign: Mapping[str, Any], *, unsafe_resume: bool = False) -> dict[str, Any]:
    """Inspect expected roots without modifying them and classify safe resume work."""

    states: list[dict[str, Any]] = []
    errors: list[str] = []
    for run in campaign.get("expected_runs") or []:
        root = Path(str(run["output_root"]))
        state: dict[str, Any] = {
            "run_id": run["run_id"],
            "output_root": str(root),
            "status": "fresh",
            "next_action": "start",
            "errors": [],
        }
        if not root.exists():
            if unsafe_resume:
                state["errors"].append("unsafe resume is forbidden for fair-comparison campaign cells")
                errors.extend(f"{run['run_id']}: {message}" for message in state["errors"])
            states.append(state)
            continue
        identity_path = root / "run_identity.json"
        if not identity_path.is_file():
            state["errors"].append("existing output root has no run_identity.json and is foreign or unowned")
        else:
            try:
                identity = _read_json(identity_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                state["errors"].append(f"invalid run identity: {exc}")
                identity = {}
            if identity.get("campaign_id") != campaign.get("campaign_id"):
                state["errors"].append("output root is owned by another campaign")
            if identity.get("campaign_identity") != campaign.get("campaign_identity"):
                state["errors"].append("output root campaign identity does not match")
            if identity.get("run_id") != run.get("run_id") or identity.get("run_identity") != run.get("run_identity"):
                state["errors"].append("output root run identity does not match")
        completion = root / "run_completion_marker.json"
        if completion.exists():
            state["status"] = "completed"
            state["next_action"] = "preserve"
        else:
            checkpoint_files = sorted(root.glob("checkpoint_step_*.json"))
            if checkpoint_files:
                state["status"] = "resumable"
                state["next_action"] = "resume"
                parsed_checkpoints: list[tuple[int, Path, dict[str, Any]]] = []
                for checkpoint_path in checkpoint_files:
                    try:
                        checkpoint = _read_json(checkpoint_path)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        state["errors"].append(f"invalid checkpoint identity {checkpoint_path.name}: {exc}")
                        continue
                    if checkpoint.get("campaign_identity") != campaign.get("campaign_identity"):
                        state["errors"].append(
                            f"existing checkpoint campaign identity does not match: {checkpoint_path.name}"
                        )
                    if checkpoint.get("run_identity") != run.get("run_identity"):
                        state["errors"].append(
                            f"existing checkpoint run identity does not match: {checkpoint_path.name}"
                        )
                    if not checkpoint.get("resumability_metadata"):
                        state["errors"].append(
                            f"existing checkpoint is missing resumability metadata: {checkpoint_path.name}"
                        )
                    step = int(checkpoint.get("optimizer_step", -1))
                    if step not in set(run.get("expected_checkpoint_steps") or []):
                        state["errors"].append(
                            f"checkpoint step is not in the campaign schedule: {checkpoint_path.name}"
                        )
                    parsed_checkpoints.append((step, checkpoint_path, checkpoint))
                latest_step, latest, checkpoint = max(parsed_checkpoints, default=(-1, checkpoint_files[-1], {}))
                state["checkpoint"] = str(latest)
                state["resume_step"] = latest_step if latest_step >= 0 else checkpoint.get("optimizer_step")
            else:
                state["status"] = "existing_without_checkpoint"
                state["next_action"] = "refuse_restart"
                state["errors"].append(
                    "existing incomplete run has no valid checkpoint; refusing restart from step zero"
                )
        if unsafe_resume:
            state["errors"].append("unsafe resume is forbidden for fair-comparison campaign cells")
        errors.extend(f"{run['run_id']}: {message}" for message in state["errors"])
        states.append(state)
    return {
        "schema_version": "spritelab_campaign_resume_audit_v1",
        "campaign_id": campaign.get("campaign_id"),
        "safe": not errors,
        "unsafe_resume_requested": bool(unsafe_resume),
        "errors": errors,
        "runs": states,
    }


def execute_campaign(
    campaign: Mapping[str, Any],
    *,
    execute: bool,
    confirm_execute: bool,
    resume: bool = False,
    unsafe_resume: bool = False,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run eligible commands only after all non-interactive safety gates pass."""

    if not execute:
        raise CampaignValidationError("execution requires explicit execute=True / --execute")
    if not confirm_execute:
        raise CampaignValidationError("execution requires explicit --confirm-execute")
    if not campaign.get("executable") or campaign.get("plan_status") != "ready":
        raise CampaignValidationError("campaign is blocked or executable=false")
    if runner is None:
        runner = subprocess.run
    validation = validate_campaign(campaign)
    if not validation["launch_ready"] or validation["errors"] or validation["blockers"]:
        raise CampaignValidationError(
            "campaign validation failed: " + "; ".join([*validation["errors"], *validation["blockers"]])
        )
    resume_report = audit_resume(campaign, unsafe_resume=unsafe_resume)
    if not resume_report["safe"]:
        raise CampaignResumeError("unsafe campaign state: " + "; ".join(resume_report["errors"]))
    states = {row["run_id"]: row for row in resume_report["runs"]}
    missing_configs = [
        str(run["resolved_config_path"])
        for run in campaign.get("expected_runs") or []
        if not Path(str(run["resolved_config_path"])).is_file()
    ]
    if missing_configs:
        raise CampaignValidationError("resolved run configs are missing: " + ", ".join(missing_configs))
    launched: list[dict[str, Any]] = []
    preserved: list[str] = []
    for run in campaign.get("expected_runs") or []:
        state = states[run["run_id"]]
        if state["status"] == "completed":
            preserved.append(run["run_id"])
            continue
        command = list(run["experiment_command"])
        if state["status"] == "resumable":
            if not resume:
                raise CampaignResumeError(f"run {run['run_id']} requires explicit resume")
            command.extend(["--resume", state["checkpoint"]])
        result = runner(command, check=True, cwd=str(Path.cwd()))
        launched.append({"run_id": run["run_id"], "command": command, "returncode": getattr(result, "returncode", 0)})
    return {
        "schema_version": "spritelab_campaign_execution_v1",
        "campaign_id": campaign.get("campaign_id"),
        "launched": launched,
        "preserved_completed_runs": preserved,
        "resume_report": resume_report,
    }


def audit_artifact_completeness(
    campaign: Mapping[str, Any], *, campaign_artifact_root: str | Path | None = None
) -> dict[str, Any]:
    """Require every declared run and campaign artifact before completion."""

    runs: list[dict[str, Any]] = []
    missing_all: list[str] = []
    for run in campaign.get("expected_runs") or []:
        root = Path(str(run["output_root"]))
        mapping = _artifact_paths(root, PER_RUN_ARTIFACTS)
        missing = [name for name, path in mapping.items() if not path.exists()]
        missing_all.extend(f"{run['run_id']}:{name}" for name in missing)
        runs.append({"run_id": run["run_id"], "complete": not missing, "missing": missing})
    campaign_root = Path(campaign_artifact_root) if campaign_artifact_root is not None else None
    campaign_missing = []
    if campaign_root is not None:
        campaign_mapping = _artifact_paths(campaign_root, CAMPAIGN_ARTIFACTS)
        campaign_missing = [name for name, path in campaign_mapping.items() if not path.exists()]
        missing_all.extend(f"campaign:{name}" for name in campaign_missing)
    return {
        "schema_version": "spritelab_campaign_artifact_audit_v1",
        "campaign_id": campaign.get("campaign_id"),
        "complete": not missing_all,
        "runs": runs,
        "campaign_missing": campaign_missing,
        "missing": missing_all,
    }


def aggregate_cross_seed_metrics(
    campaign: Mapping[str, Any], reports: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Deterministically aggregate only identically defined metrics across required seeds."""

    rows: list[dict[str, Any]] = []
    missing_runs: list[str] = []
    by_cell: dict[str, dict[int, Mapping[str, Any]]] = {}
    expected = list(campaign.get("expected_runs") or [])
    for run in expected:
        report = reports.get(str(run["run_id"]))
        if report is None:
            missing_runs.append(str(run["run_id"]))
            continue
        by_cell.setdefault(str(run["cell_id"]), {})[int(run["seed"])] = report
    for cell_id in sorted({str(run["cell_id"]) for run in expected}):
        cell_reports = by_cell.get(cell_id, {})
        metric_names: set[str] = set()
        for report in cell_reports.values():
            metric_names.update((report.get("metrics") or {}).keys())
        for metric_name in sorted(metric_names):
            per_seed: dict[str, float] = {}
            definitions: dict[str, Any] = {}
            missing_metric_seeds: list[int] = []
            for seed in campaign.get("seeds") or []:
                report = cell_reports.get(int(seed))
                metric = None if report is None else (report.get("metrics") or {}).get(metric_name)
                definition = None if report is None else (report.get("metric_definitions") or {}).get(metric_name)
                if metric is None or definition is None:
                    missing_metric_seeds.append(int(seed))
                else:
                    per_seed[str(seed)] = float(metric)
                    definitions[stable_hash(definition)] = definition
            compatible = len(definitions) <= 1
            values = [per_seed[key] for key in sorted(per_seed, key=int)]
            complete = not missing_metric_seeds and len(values) == REQUIRED_SEED_COUNT and compatible
            rows.append(
                {
                    "cell_id": cell_id,
                    "metric": metric_name,
                    "definition": next(iter(definitions.values()), None) if compatible else None,
                    "compatible": compatible,
                    "complete": complete,
                    "per_seed": per_seed,
                    "mean": sum(values) / len(values) if complete else None,
                    "standard_deviation": _population_std(values) if complete else None,
                    "minimum": min(values) if complete else None,
                    "maximum": max(values) if complete else None,
                    "missing_metric_seeds": missing_metric_seeds,
                    "incompatible_definition_hashes": [] if compatible else sorted(definitions),
                }
            )
    complete = not missing_runs and bool(rows) and all(row["complete"] for row in rows)
    return {
        "schema_version": "spritelab_cross_seed_aggregate_v1",
        "campaign_id": campaign.get("campaign_id"),
        "required_seed_count": REQUIRED_SEED_COUNT,
        "per_seed_metrics": rows,
        "missing_runs": missing_runs,
        "complete": complete,
        "promotion_eligible": complete and bool(campaign.get("baseline_launch_authorized", False)),
        "promotion_status": (
            "blocked: complete compatible three-seed evidence and independent promotion authorization required"
            if not complete or not campaign.get("baseline_launch_authorized", False)
            else "eligible_for_independent_review"
        ),
    }


def load_campaign(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path))


def write_json_exclusive(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalise_cells(raw_cells: Any) -> list[dict[str, Any]]:
    result = []
    for raw in raw_cells or []:
        cell = {"cell_id": str(raw)} if isinstance(raw, str) else deepcopy(dict(raw))
        cell["cell_id"] = str(cell.get("cell_id") or "")
        cell.setdefault("comparison_values", {})
        cell.setdefault("overrides", {})
        result.append(cell)
    return result


def _resolved_cell(campaign: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "training": deepcopy(campaign.get("training")),
        "optimizer": deepcopy(campaign.get("optimizer")),
        "schedule": deepcopy(campaign.get("schedule")),
        "loss": deepcopy(campaign.get("loss")),
        "determinism": deepcopy(campaign.get("determinism")),
        "identities": deepcopy(campaign.get("identities")),
        "evaluation": deepcopy(campaign.get("evaluation")),
        "checkpoint": deepcopy(campaign.get("checkpoint")),
    }
    for dotted, value in dict(cell.get("overrides") or {}).items():
        _set_dotted(result, str(dotted), deepcopy(value))
    return result


def _get_dotted(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _flatten_leaves(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            result.update(_flatten_leaves(child, path))
        else:
            result[path] = child
    return result


def _set_dotted(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    keys = dotted.split(".")
    current = value
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = replacement


def _is_declared(path: str, declared: set[str]) -> bool:
    return path in declared or path.rsplit(".", 1)[-1] in declared


def _passes_from_training(training: Mapping[str, Any], max_steps: int) -> dict[str, Any]:
    try:
        return effective_pass_report(
            optimizer_steps=max_steps,
            effective_batch_size=int(training.get("effective_batch_size", 0)),
            positive_sampling_mass_records=float(training.get("positive_sampling_mass_records", 0)),
            nominal_record_count=training.get("nominal_record_count"),
            positive_weight_record_count=training.get("positive_weight_record_count"),
            positive_weight_sum=training.get("positive_weight_sum"),
        )
    except (TypeError, ValueError):
        return {
            "effective_dataset_passes": None,
            "error": "positive sampling mass and valid fixed-step training settings are required",
            "formula": "optimizer_steps * effective_batch_size / positive_sampling_mass_records",
        }


def _run_identity_payload(
    raw: Mapping[str, Any],
    cell: Mapping[str, Any],
    run: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_id": raw.get("campaign_id"),
        "cell": cell,
        "seed": run["seed"],
        "output_root": run["output_root"],
        "identities": raw.get("identities"),
        "training": training,
        "optimizer": raw.get("optimizer"),
        "schedule": raw.get("schedule"),
        "loss": raw.get("loss"),
        "determinism": raw.get("determinism"),
        "evaluation": evaluation,
        "checkpoint": checkpoint,
    }


def _campaign_identity_payload(campaign: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"campaign_identity", "plan_status", "executable", "blockers"}
    return {key: value for key, value in campaign.items() if key not in excluded}


def _artifact_paths(root: Path, names: Sequence[str]) -> dict[str, Path]:
    return {name: root / f"{name}.json" for name in names}


def _population_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _default_abort_conditions() -> list[str]:
    return [
        "any protected fairness field mismatch",
        "any missing or placeholder production identity",
        "any scheduled evaluation or checkpoint point missing",
        "non-finite training or validation metric",
        "checkpoint or output-root identity mismatch",
        "any attempt to restart an incomplete run from step zero",
        "any missing required artifact",
    ]


def _default_promotion_restrictions() -> list[str]:
    return [
        "independent evaluation contract approval is required",
        "all three required seeds must complete",
        "all metrics must have compatible definitions",
        "one-seed promotion is forbidden",
        "all required run and campaign artifacts must be present",
    ]
