"""Versioned, fixed-step training campaign planning and orchestration.

This module deliberately stays above the single-run experiment system.  It does
not import torch or model code; process launching is isolated behind an
injectable runner so planning, validation, status, and tests are CPU-only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from spritelab.utils.safe_fs import UnsafeFilesystemOperation, open_anchored_directory, require_confined_path

CAMPAIGN_SCHEMA_VERSION = "spritelab_training_campaign_v3"
CODE_IDENTITY_SCHEMA_VERSION = "spritelab_training_code_identity_v4"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "spritelab_required_artifact_manifest_v1"
CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION = "spritelab_campaign_artifact_manifest_v1"
CAMPAIGN_ARTIFACT_MANIFEST_NAME = "campaign_artifact_manifest.json"
CAMPAIGN_COMPLETION_REPORT_SCHEMA_VERSION = "spritelab_campaign_completion_report_v1"
CAMPAIGN_COMPLETION_REPORT_NAME = "campaign_completion_report.json"
RESUME_CHECKPOINT_SCHEMA_VERSION = "spritelab_campaign_resume_checkpoint_v1"
DEFAULT_SEEDS = (731001, 731002, 731003)
REQUIRED_SEED_COUNT = 3
_CAMPAIGN_FILESYSTEM_SNAPSHOT_AUTHORITY = object()


class _CampaignFilesystemSnapshot:
    """In-process evidence issued only after a trusted filesystem capture."""

    __slots__ = ("_authority", "_sealed", "campaign_identity", "foreign_run_roots", "runs", "schema_version")

    def __init__(
        self,
        *,
        campaign_identity: str,
        runs: Sequence[Mapping[str, Any]],
        _authority: object,
    ) -> None:
        if _authority is not _CAMPAIGN_FILESYSTEM_SNAPSHOT_AUTHORITY:
            raise CampaignValidationError("campaign filesystem snapshot issuer is not trusted")
        object.__setattr__(self, "schema_version", "spritelab_campaign_filesystem_snapshot_v1")
        object.__setattr__(self, "campaign_identity", campaign_identity)
        object.__setattr__(self, "foreign_run_roots", ())
        object.__setattr__(self, "runs", tuple(_freeze_snapshot_value(state) for state in runs))
        object.__setattr__(self, "_authority", _authority)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("campaign filesystem snapshots are immutable")
        object.__setattr__(self, name, value)


def _freeze_snapshot_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_snapshot_value(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_snapshot_value(item) for item in value)
    return deepcopy(value)


def _thaw_snapshot_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_snapshot_value(item) for item in value]
    return deepcopy(value)


def _native_event_migration_verification(run_id: str) -> dict[str, Any]:
    from spritelab.product_web.events import MIGRATION_EVIDENCE_SCHEMA, EventMigrationState

    empty_sha256 = hashlib.sha256(b"").hexdigest()
    details = {
        "canonical_present": False,
        "canonical_size_bytes": 0,
        "canonical_sha256": empty_sha256,
        "event_history_origin": "native",
        "origin_record_present": False,
        "migration_required": False,
        "canonical_prefix_size_bytes": 0,
        "canonical_prefix_sha256": empty_sha256,
    }
    return {
        "state": EventMigrationState.NO_MIGRATION.value,
        "run_id": run_id,
        "evidence_sha256": stable_hash(
            {
                "schema_version": MIGRATION_EVIDENCE_SCHEMA,
                "state": EventMigrationState.NO_MIGRATION.value,
                "run_id": run_id,
                "details": details,
            }
        ),
        "message": "No legacy migration is recorded or required.",
        "record": None,
        "details": details,
    }


def _capture_fresh_campaign_filesystem_snapshot(
    campaign: Mapping[str, Any],
    project_root: str | Path,
) -> _CampaignFilesystemSnapshot | None:
    """Capture an all-fresh campaign state through one anchored ancestor."""

    root = Path(os.path.abspath(os.fspath(project_root)))
    runs = list(campaign.get("expected_runs") or ())
    if not runs:
        return None
    try:
        output_roots = [require_confined_path(Path(os.path.abspath(str(run.get("output_root")))), root) for run in runs]
    except (OSError, UnsafeFilesystemOperation, ValueError):
        return None
    campaign_roots = {output_root.parents[1] for output_root in output_roots if len(output_root.parents) >= 2}
    if len(campaign_roots) != 1:
        return None
    campaign_root = next(iter(campaign_roots))
    current = campaign_root
    missing_parts: list[str] = []
    while not os.path.lexists(current) and current != root:
        missing_parts.append(current.name)
        current = current.parent
    if not missing_parts or current == campaign_root or current == current.parent:
        return None
    try:
        with open_anchored_directory(current, root) as ancestor:
            if ancestor.lexists(missing_parts[-1]):
                return None
            ancestor.verify()
    except (OSError, UnsafeFilesystemOperation, ValueError):
        return None
    states: list[dict[str, Any]] = []
    for run, output_root in zip(runs, output_roots, strict=True):
        run_id = str(run.get("run_id"))
        states.append(
            {
                "run_id": run_id,
                "output_root": str(output_root),
                "status": "fresh",
                "next_action": "start",
                "errors": [],
                "event_migration_verification": _native_event_migration_verification(run_id),
            }
        )
    return _CampaignFilesystemSnapshot(
        campaign_identity=str(campaign.get("campaign_identity") or ""),
        runs=states,
        _authority=_CAMPAIGN_FILESYSTEM_SNAPSHOT_AUTHORITY,
    )


def _capture_anchored_campaign_filesystem_snapshot(
    campaign: Mapping[str, Any],
    physical_run_roots: Mapping[str, Path],
) -> _CampaignFilesystemSnapshot:
    """Classify exact already-held run roots without reopening lexical roots."""

    runs = list(campaign.get("expected_runs") or ())
    expected_ids = [str(run.get("run_id")) for run in runs]
    if set(physical_run_roots) != set(expected_ids):
        raise CampaignValidationError("anchored campaign filesystem does not cover the exact run set")
    states = [
        _classify_run_root(
            campaign,
            run,
            root_override=Path(physical_run_roots[str(run.get("run_id"))]),
        )
        for run in runs
    ]
    return _CampaignFilesystemSnapshot(
        campaign_identity=str(campaign.get("campaign_identity") or ""),
        runs=states,
        _authority=_CAMPAIGN_FILESYSTEM_SNAPSHOT_AUTHORITY,
    )


TRAINING_CODE_IDENTITY_MANDATORY_FILES = (
    "src/spritelab/__main__.py",
    "src/spritelab/product_core/contracts.py",
    "src/spritelab/product_core/events.py",
    "src/spritelab/product_features/dataset/certification.py",
    "src/spritelab/product_features/training/activation.py",
    "src/spritelab/product_features/training/dashboard.py",
    "src/spritelab/product_features/training/preparation_jobs.py",
    "src/spritelab/product_features/training/service.py",
    "src/spritelab/product_runtime.py",
    "src/spritelab/product_web/app.py",
    "src/spritelab/product_web/cli.py",
    "src/spritelab/product_web/events.py",
    "src/spritelab/training/launch.py",
)

TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS = ("src/spritelab",)

TRAINING_CODE_IDENTITY_SEMANTIC_ROLES = {
    "src/spritelab/product_core/contracts.py": "finite ProductEvent construction and deserialization contract",
    "src/spritelab/product_core/events.py": "strict event JSON parsing, validation, and serialization",
    "src/spritelab/product_features/dataset/certification.py": (
        "labeling-scope authorization used by v3 conditioned-view, freeze, and training gates"
    ),
    "src/spritelab/product_features/training/activation.py": (
        "conditioned Dataset-v5 publication, campaign, audit, launch, and resume applicability"
    ),
    "src/spritelab/product_features/training/dashboard.py": "training event interpretation and chart projection",
    "src/spritelab/product_features/training/preparation_jobs.py": (
        "durable image-only baseline preparation state, recovery, and event history"
    ),
    "src/spritelab/product_features/training/service.py": "training append, replay, reconstruction, pause, and resume adapter",
    "src/spritelab/product_runtime.py": "product plugin composition and training capability registration",
    "src/spritelab/product_web/app.py": "generic product run-action validation and resume routing",
    "src/spritelab/product_web/cli.py": "product server and CLI runtime composition",
    "src/spritelab/product_web/events.py": (
        "canonical filename, transactional append/migration, event-history origin, revalidation, and durable replay semantics"
    ),
}

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
    "run_identity",
    "experiment_manifest",
    "resolved_config",
    "checkpoint_series",
    "artifact_manifest",
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

OUTPUT_ROOT_STATES = (
    "fresh",
    "valid_resumable",
    "complete",
    "partial_valid",
    "partial_invalid",
    "foreign",
    "contradictory",
    "corrupt",
)

_METRIC_ARTIFACT_TYPES = frozenset(
    {"training_metrics", "validation_metrics", "ema_metrics", "live_metrics", "evaluation_reports"}
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
    "identities.split_manifest_hash",
    "identities.conditioning_vocabulary_hash",
    "optimizer",
    "schedule",
    "loss",
    "evaluation.benchmark_manifest_hash",
    "evaluation.cadence",
    "checkpoint.cadence",
    "evaluation.cfg_value",
    "evaluation.sampling_steps",
    "evaluation.ema_policy",
    "evaluation.live_weight_evaluation_policy",
    "expected_artifact_contract",
    "determinism",
)

ALLOWED_EXPERIMENTAL_VARIABLES = {
    "model.auxiliary_heads_mode",
    "model.base_channels",
    "model.channel_mults",
    "model.res_blocks_per_level",
    "model.bottleneck_attention",
    "model.film_conditioning",
    "model.palette_conditioning",
    "architecture.auxiliary_heads_mode",
    "auxiliary_heads_mode",
}

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


def _canonical_json_type_exact_equal(actual: Any, expected: Any) -> bool:
    """Compare canonical JSON values without Python's bool/int/float equality aliases."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _canonical_json_type_exact_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _canonical_json_type_exact_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


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
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max optimizer steps must be a positive integer; fixed-epoch fallback is forbidden")
    if type(cadence) is not int or cadence <= 0:
        raise ValueError("cadence must be a positive integer")
    maximum, interval = max_steps, cadence
    result = [0] if include_step_zero else []
    result.extend(range(interval, maximum + 1, interval))
    if result[-1:] != [maximum]:
        result.append(maximum)
    return result


def _strict_campaign_integer(value: Any, label: str, errors: list[str]) -> int:
    if type(value) is not int:
        errors.append(f"{label} must be an integer; coercion is forbidden")
        return 0
    return value


def _strict_campaign_boolean(spec: Mapping[str, Any], field: str, errors: list[str]) -> bool:
    if field not in spec:
        return False
    value = spec[field]
    if type(value) is not bool:
        errors.append(f"{field} must be a boolean; coercion is forbidden")
        return False
    return value


def _require_runtime_boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise CampaignValidationError(f"{field} must be a boolean; coercion is forbidden")
    return value


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
            "evaluation_schedule",
            "checkpoint_schedule",
            "expected_runs",
            "expected_artifact_contract",
            "abort_conditions",
            "promotion_restrictions",
            "plan_status",
            "executable",
            "launch_authorized",
            "code_identity",
            "campaign_identity",
        ],
        "properties": {
            "schema_version": {"const": CAMPAIGN_SCHEMA_VERSION},
            "campaign_id": {"type": "string", "minLength": 1},
            "purpose": {"type": "string", "minLength": 1},
            "architecture_cells": {"type": "array", "minItems": 1, "items": {"type": "object"}},
            "identities": {"type": "object"},
            "seeds": {
                "type": "array",
                "minItems": REQUIRED_SEED_COUNT,
                "maxItems": REQUIRED_SEED_COUNT,
                "uniqueItems": True,
                "items": {"type": "integer"},
            },
            "training": {"type": "object"},
            "optimizer": {"type": "object"},
            "schedule": {"type": "object"},
            "loss": {"type": "object"},
            "determinism": {"type": "object"},
            "evaluation": {"type": "object"},
            "checkpoint": {"type": "object"},
            "evaluation_schedule": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "integer"}},
            },
            "checkpoint_schedule": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "integer"}},
            },
            "expected_runs": {"type": "array", "items": {"type": "object"}},
            "expected_artifact_contract": {"type": "object"},
            "abort_conditions": {"type": "array", "items": {"type": "string"}},
            "promotion_restrictions": {"type": "array", "items": {"type": "string"}},
            "plan_status": {"enum": ["ready", "blocked"]},
            "experimental_variables": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            "fixed_epoch_fallback": {"const": False},
            "executable": {"type": "boolean"},
            "launch_authorized": {"type": "boolean"},
            "code_identity": {"type": "object"},
            "campaign_identity": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
        },
        "additionalProperties": True,
    }


def plan_campaign(
    spec: Mapping[str, Any],
    *,
    execution_root: str | Path | None = None,
    file_sha256_resolver: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Resolve a deterministic v1 campaign manifest from an input specification."""

    raw = deepcopy(dict(spec))
    campaign_id = str(raw.get("campaign_id") or "").strip()
    cells = _normalise_cells(raw.get("architecture_cells", []))
    seeds = list(raw.get("seeds", DEFAULT_SEEDS))
    training = deepcopy(dict(raw.get("training") or {}))
    evaluation = deepcopy(dict(raw.get("evaluation") or {}))
    checkpoint = deepcopy(dict(raw.get("checkpoint") or {}))
    identities = deepcopy(dict(raw.get("identities") or {}))

    schedule_errors: list[str] = []
    requested_executable = _strict_campaign_boolean(raw, "executable", schedule_errors)
    launch_authorized = _strict_campaign_boolean(raw, "launch_authorized", schedule_errors)
    if any(type(seed) is not int for seed in seeds):
        schedule_errors.append("campaign seeds must be integers; coercion is forbidden")
    micro_batch = _strict_campaign_integer(training.get("micro_batch_size"), "micro_batch_size", schedule_errors)
    accumulation = _strict_campaign_integer(
        training.get("gradient_accumulation"), "gradient_accumulation", schedule_errors
    )
    calculated_effective = micro_batch * accumulation
    training.setdefault("effective_batch_size", calculated_effective)
    training["effective_batch_size_formula"] = "micro_batch_size * gradient_accumulation"
    max_steps = _strict_campaign_integer(training.get("max_optimizer_steps"), "max_optimizer_steps", schedule_errors)
    eval_cadence = _strict_campaign_integer(evaluation.get("cadence"), "evaluation cadence", schedule_errors)
    checkpoint_cadence = _strict_campaign_integer(checkpoint.get("cadence"), "checkpoint cadence", schedule_errors)
    evaluation.setdefault("include_step_zero", False)
    evaluation.setdefault("ema_policy", "both")
    evaluation.setdefault("live_weight_evaluation_policy", "required")
    checkpoint.setdefault("require_resumability_metadata", True)

    expected_runs: list[dict[str, Any]] = []
    eval_matrix: dict[str, list[int]] = {}
    checkpoint_matrix: dict[str, list[int]] = {}
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
    code_identity = _code_identity()
    for cell in cells:
        cell_id = cell["cell_id"]
        for seed in seeds:
            run_id = f"{campaign_id}__{cell_id}__seed_{seed}"
            output_root = output_base / campaign_id / cell_id / f"seed_{seed}"
            resolved_config = output_base / campaign_id / "resolved_configs" / f"{cell_id}__seed_{seed}.json"
            command = [
                sys.executable,
                "-m",
                "spritelab",
                "train",
                "experiment",
                "run",
                "--config",
                str(resolved_config),
            ]
            resolved_run_config = _resolved_run_config(
                raw,
                cell,
                seed=seed,
                output_root=str(output_root),
                execution_root=Path(execution_root or Path.cwd()),
            )
            run = {
                "run_id": run_id,
                "campaign_id": campaign_id,
                "cell_id": cell_id,
                "seed": seed,
                "output_root": str(output_root),
                "resolved_config_path": str(resolved_config),
                "resolved_config": resolved_run_config,
                "resolved_config_sha256": stable_hash(resolved_run_config),
                "experiment_command": command,
                "code_identity": code_identity,
                "expected_evaluation_steps": list(eval_schedule),
                "expected_checkpoint_steps": list(save_schedule),
                "effective_passes": _passes_from_training(training, max_steps),
                "expected_artifacts": list(PER_RUN_ARTIFACTS),
            }
            run["execution_contract_sha256"] = stable_hash(_execution_contract_payload(run))
            run["run_identity"] = stable_hash(_run_identity_payload(raw, cell, run, training, evaluation, checkpoint))
            expected_runs.append(run)
            eval_matrix[run_id] = list(eval_schedule)
            checkpoint_matrix[run_id] = list(save_schedule)

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
        "model": deepcopy(raw.get("model") or {}),
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
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "campaign_manifest_schema_version": CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "campaign_manifest_name": CAMPAIGN_ARTIFACT_MANIFEST_NAME,
            "campaign_completion_report_schema_version": CAMPAIGN_COMPLETION_REPORT_SCHEMA_VERSION,
            "campaign_completion_report_name": CAMPAIGN_COMPLETION_REPORT_NAME,
            "per_run": list(PER_RUN_ARTIFACTS),
            "campaign": list(CAMPAIGN_ARTIFACTS),
            "completion_requires_all": True,
            "exact_checkpoint_set": True,
            "content_hashes_required": True,
        },
        "abort_conditions": list(raw.get("abort_conditions") or _default_abort_conditions()),
        "promotion_restrictions": list(raw.get("promotion_restrictions") or _default_promotion_restrictions()),
        "fixed_epoch_fallback": False,
        "launch_authorized": launch_authorized,
        "campaign_artifact_root": (
            str(raw.get("campaign_artifact_root")) if raw.get("campaign_artifact_root") else None
        ),
        "code_identity": code_identity,
        "plan_status": "blocked",
        "executable": False,
        "blockers": schedule_errors,
    }
    manifest["campaign_identity"] = stable_hash(_campaign_identity_payload(manifest))
    validation = validate_campaign(manifest, file_sha256_resolver=file_sha256_resolver)
    blockers = sorted({*schedule_errors, *validation["errors"], *validation["blockers"]})
    manifest["blockers"] = blockers
    manifest["executable"] = requested_executable and not blockers
    manifest["plan_status"] = "ready" if manifest["executable"] else "blocked"
    # Bind the fully resolved plan after status resolution.
    manifest["campaign_identity"] = stable_hash(_campaign_identity_payload(manifest))
    return manifest


def validate_campaign(
    campaign: Mapping[str, Any],
    *,
    file_sha256_resolver: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Validate schema, production identities, schedules, and comparison fairness."""

    errors: list[str] = []
    blockers: list[str] = []
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {CAMPAIGN_SCHEMA_VERSION!r}")
    for field in ("executable", "launch_authorized"):
        if type(campaign.get(field)) is not bool:
            errors.append(f"{field} must be a boolean; coercion is forbidden")
    if not str(campaign.get("campaign_id") or "").strip():
        errors.append("campaign_id is required")
    if not str(campaign.get("purpose") or "").strip():
        errors.append("purpose is required")
    declared_blockers = campaign.get("blockers")
    if not isinstance(declared_blockers, list):
        errors.append("blockers must be a list")
    else:
        blockers.extend(f"unresolved blocker: {item}" for item in declared_blockers if str(item).strip())
    seeds = list(campaign.get("seeds") or [])
    if len(seeds) != REQUIRED_SEED_COUNT:
        errors.append(f"exactly {REQUIRED_SEED_COUNT} seeds are required; found {len(seeds)}")
    if any(type(seed) is not int for seed in seeds):
        errors.append("seeds must be integers; bool and coercible strings are forbidden")
    elif len(set(seeds)) != len(seeds):
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
    for hash_field, path_field in (
        ("dataset_view_manifest_hash", "dataset_view_manifest_path"),
        ("split_manifest_hash", "split_manifest_path"),
        ("conditioning_vocabulary_hash", "conditioning_vocabulary_path"),
    ):
        _validate_file_binding(
            identities,
            hash_field,
            path_field,
            blockers,
            file_sha256_resolver=file_sha256_resolver,
        )
    configuration_bindings = {
        "model_config_hash": campaign.get("model") or {},
        "optimizer_config_hash": campaign.get("optimizer") or {},
        "schedule_config_hash": campaign.get("schedule") or {},
        "loss_config_hash": campaign.get("loss") or {},
        "determinism_config_hash": campaign.get("determinism") or {},
    }
    for field, content in configuration_bindings.items():
        if identities.get(field) != stable_hash(content):
            blockers.append(f"identity {field} does not match canonical resolved configuration")
    evaluation = dict(campaign.get("evaluation") or {})
    if not is_concrete_hash(evaluation.get("benchmark_manifest_hash")):
        blockers.append("evaluation benchmark_manifest_hash is missing, placeholder, or not a concrete SHA-256")
    if not is_concrete_hash(evaluation.get("evaluation_config_hash")):
        blockers.append("evaluation evaluation_config_hash is missing, placeholder, or not a concrete SHA-256")
    _validate_file_binding(
        evaluation,
        "benchmark_manifest_hash",
        "benchmark_manifest_path",
        blockers,
        prefix="evaluation ",
        file_sha256_resolver=file_sha256_resolver,
    )
    if evaluation.get("evaluation_config_hash") != stable_hash(_evaluation_identity_payload(evaluation)):
        blockers.append("evaluation evaluation_config_hash does not match canonical evaluation configuration")
    if evaluation.get("ema_policy") not in {"ema", "live", "both"}:
        errors.append("evaluation ema_policy must be one of: ema, live, both")
    if evaluation.get("ema_policy") == "both" and evaluation.get("live_weight_evaluation_policy") != "required":
        errors.append("ema_policy 'both' requires live_weight_evaluation_policy='required'")

    training = dict(campaign.get("training") or {})
    micro = training.get("micro_batch_size")
    accumulation = training.get("gradient_accumulation")
    effective = training.get("effective_batch_size")
    if any(type(value) is not int or value <= 0 for value in (micro, accumulation, effective)):
        errors.append("micro batch, gradient accumulation, and effective batch must be positive")
    elif micro * accumulation != effective:
        errors.append(
            f"effective batch mismatch: {micro} * {accumulation} = {micro * accumulation}, declared {effective}"
        )
    if type(training.get("max_optimizer_steps")) is not int or training["max_optimizer_steps"] <= 0:
        errors.append("positive max_optimizer_steps is required; fixed-epoch fallback is forbidden")
    sampling_mass = training.get("positive_sampling_mass_records")
    if isinstance(sampling_mass, bool) or not isinstance(sampling_mass, (int, float)) or sampling_mass <= 0:
        errors.append("positive_sampling_mass_records must be positive")
    for field in ("precision", "sampler_policy"):
        if not isinstance(training.get(field), str) or not training[field].strip():
            errors.append(f"training {field} is required")
    for field in ("optimizer", "schedule", "loss", "determinism"):
        if not isinstance(campaign.get(field), Mapping) or not campaign.get(field):
            errors.append(f"{field} configuration is required")
    if campaign.get("fixed_epoch_fallback") not in {False, None}:
        errors.append("fixed-epoch fallback is forbidden")
    try:
        expected_eval = evaluation_steps(
            training.get("max_optimizer_steps"),
            evaluation.get("cadence"),
            include_step_zero=bool(evaluation.get("include_step_zero", False)),
        )
        expected_checkpoints = checkpoint_steps(
            training.get("max_optimizer_steps"), dict(campaign.get("checkpoint") or {}).get("cadence")
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        expected_eval, expected_checkpoints = [], []
    for run in campaign.get("expected_runs") or []:
        if not _canonical_json_type_exact_equal(run.get("expected_evaluation_steps"), expected_eval):
            errors.append(f"run {run.get('run_id')} has a missing or altered evaluation schedule point")
        if not _canonical_json_type_exact_equal(run.get("expected_checkpoint_steps"), expected_checkpoints):
            errors.append(f"run {run.get('run_id')} has a missing or altered checkpoint schedule point")

    fairness = validate_fixed_step_fairness(campaign)
    errors.extend(fairness["errors"])
    expected_count = len(cells) * len(seeds)
    runs = list(campaign.get("expected_runs") or [])
    expected_schedule_matrices = {
        "evaluation_schedule": {
            str(run.get("run_id") or ""): list(expected_eval) for run in runs if isinstance(run, Mapping)
        },
        "checkpoint_schedule": {
            str(run.get("run_id") or ""): list(expected_checkpoints) for run in runs if isinstance(run, Mapping)
        },
    }
    for field, expected_matrix in expected_schedule_matrices.items():
        actual_matrix = campaign.get(field)
        if not _canonical_json_type_exact_equal(actual_matrix, expected_matrix):
            errors.append(f"{field} does not exactly match the expected run schedule matrix")
    if len(runs) != expected_count:
        errors.append(f"expected_runs must contain {expected_count} cell/seed runs; found {len(runs)}")
    run_ids = [str(run.get("run_id") or "") for run in runs]
    roots = [str(run.get("output_root") or "") for run in runs]
    if len(set(run_ids)) != len(run_ids):
        errors.append("expected run IDs must be unique")
    if len(set(roots)) != len(roots):
        errors.append("expected output roots must be unique")
    expected_pairs = {(cell_id, seed) for cell_id in cell_ids for seed in seeds}
    observed_pairs = [(run.get("cell_id"), run.get("seed")) for run in runs if isinstance(run, Mapping)]
    if len(observed_pairs) != len(set(observed_pairs)):
        errors.append("expected_runs contains a duplicate cell/seed run")
    if set(observed_pairs) != expected_pairs:
        errors.append("expected_runs does not contain the exact declared cell/seed run set")
    if list(campaign.get("expected_run_ids") or []) != run_ids:
        errors.append("expected_run_ids does not exactly match expected_runs")
    if list(campaign.get("expected_output_roots") or []) != roots:
        errors.append("expected_output_roots does not exactly match expected_runs")
    if not campaign.get("abort_conditions"):
        errors.append("abort_conditions must be bound")
    if not campaign.get("promotion_restrictions"):
        errors.append("promotion_restrictions must be bound")
    contract = dict(campaign.get("expected_artifact_contract") or {})
    if set(contract.get("per_run") or []) != set(PER_RUN_ARTIFACTS):
        errors.append("per-run artifact contract is incomplete")
    if set(contract.get("campaign") or []) != set(CAMPAIGN_ARTIFACTS):
        errors.append("campaign artifact contract is incomplete")
    if contract.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        errors.append("artifact manifest contract schema is missing or unsupported")
    if contract.get("campaign_manifest_schema_version") != CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        errors.append("campaign artifact manifest contract schema is missing or unsupported")
    if contract.get("campaign_manifest_name") != CAMPAIGN_ARTIFACT_MANIFEST_NAME:
        errors.append("campaign artifact manifest filename is missing or unsupported")
    if contract.get("campaign_completion_report_schema_version") != CAMPAIGN_COMPLETION_REPORT_SCHEMA_VERSION:
        errors.append("campaign completion report contract schema is missing or unsupported")
    if contract.get("campaign_completion_report_name") != CAMPAIGN_COMPLETION_REPORT_NAME:
        errors.append("campaign completion report filename is missing or unsupported")
    if contract.get("exact_checkpoint_set") is not True or contract.get("content_hashes_required") is not True:
        errors.append("artifact manifest contract must require exact checkpoint sets and content hashes")

    stored_code_identity = campaign.get("code_identity")
    try:
        current_code_identity = _code_identity()
    except (OSError, CampaignValidationError) as exc:
        blockers.append(f"current code identity cannot be verified: {exc}")
        current_code_identity = None
    if not isinstance(stored_code_identity, Mapping):
        errors.append("campaign code identity is mandatory")
    elif current_code_identity is not None and stored_code_identity != current_code_identity:
        blockers.append("campaign code identity is stale; replan and revalidate against current production code")
    stored_campaign_identity = campaign.get("campaign_identity")
    if not is_concrete_hash(stored_campaign_identity):
        errors.append("campaign identity is mandatory")
    elif stored_campaign_identity != stable_hash(_campaign_identity_payload(campaign)):
        errors.append("campaign identity does not match manifest content")
    for run in runs:
        if run.get("campaign_id") != campaign.get("campaign_id"):
            errors.append(f"run {run.get('run_id')} campaign_id does not match")
        cell = next((item for item in cells if item.get("cell_id") == run.get("cell_id")), None)
        if cell is None:
            errors.append(f"run {run.get('run_id')} references an unknown architecture cell")
            continue
        if run.get("code_identity") != stored_code_identity:
            errors.append(f"run {run.get('run_id')} code identity does not match campaign code identity")
        expected_identity = stable_hash(
            _run_identity_payload(campaign, cell, run, training, evaluation, dict(campaign.get("checkpoint") or {}))
        )
        if run.get("run_identity") != expected_identity:
            errors.append(f"run {run.get('run_id')} identity does not match resolved campaign settings")
        resolved = run.get("resolved_config")
        if not isinstance(resolved, Mapping) or run.get("resolved_config_sha256") != stable_hash(resolved):
            errors.append(f"run {run.get('run_id')} resolved config identity does not match")
        elif resolved.get("schema_version") != "spritelab_experiment_config_v1":
            errors.append(f"run {run.get('run_id')} resolved config is not an experiment configuration")
        else:
            required_experiment_fields = {
                "name",
                "ablation",
                "dataset",
                "model",
                "conditioning",
                "loss",
                "optimizer",
                "augmentation",
                "seeds",
                "runtime",
                "sampling",
                "ema",
            }
            missing = sorted(required_experiment_fields - set(resolved))
            if missing:
                errors.append(f"run {run.get('run_id')} resolved experiment config is missing: {', '.join(missing)}")
        if run.get("execution_contract_sha256") != stable_hash(_execution_contract_payload(run)):
            errors.append(f"run {run.get('run_id')} execution contract identity does not match")
    return {
        "schema_version": "spritelab_campaign_validation_v1",
        "campaign_id": campaign.get("campaign_id"),
        "valid": not errors,
        "launch_ready": (
            not errors
            and not blockers
            and campaign.get("executable") is True
            and campaign.get("plan_status") == "ready"
            and campaign.get("launch_authorized") is True
        ),
        "errors": errors,
        "blockers": blockers,
        "fairness": fairness,
    }


def validate_fixed_step_fairness(campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Report every protected and undeclared comparison-cell mismatch."""

    cells = list(campaign.get("architecture_cells") or [])
    declared = set(map(str, campaign.get("experimental_variables") or []))
    declaration_errors = []
    for path in sorted(declared):
        if "*" in path:
            declaration_errors.append(f"experimental variable {path!r} may not contain wildcards")
        elif path not in ALLOWED_EXPERIMENTAL_VARIABLES:
            declaration_errors.append(f"experimental variable {path!r} is not an allowed architecture/model field")
    mismatches: list[dict[str, Any]] = []
    if len(cells) < 2:
        return {
            "schema_version": "spritelab_fixed_step_fairness_v1",
            "fair": not declaration_errors,
            "declared_experimental_variables": sorted(declared),
            "protected_fields": list(PROTECTED_COMPARISON_FIELDS),
            "mismatches": [],
            "errors": declaration_errors,
        }
    baseline = _resolved_cell(campaign, cells[0])
    baseline_id = str(cells[0].get("cell_id"))
    for cell in cells[1:]:
        current = _resolved_cell(campaign, cell)
        cell_id = str(cell.get("cell_id"))
        for path in PROTECTED_COMPARISON_FIELDS:
            before, after = _get_dotted(baseline, path), _get_dotted(current, path)
            if before != after:
                allowed = False
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
    errors = declaration_errors + [
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


def audit_resume(
    campaign: Mapping[str, Any],
    *,
    unsafe_resume: bool = False,
    filesystem_snapshot: _CampaignFilesystemSnapshot | None = None,
) -> dict[str, Any]:
    """Classify all output roots using the closed, versioned state set."""

    unsafe_resume = _require_runtime_boolean(unsafe_resume, "unsafe_resume")
    if filesystem_snapshot is None:
        states = [_classify_run_root(campaign, run) for run in campaign.get("expected_runs") or []]
        foreign_run_roots = _foreign_run_roots(campaign)
    else:
        if (
            type(filesystem_snapshot) is not _CampaignFilesystemSnapshot
            or filesystem_snapshot._authority is not _CAMPAIGN_FILESYSTEM_SNAPSHOT_AUTHORITY
        ):
            raise CampaignValidationError("campaign filesystem snapshot was not issued by the trusted capture seam")
        if filesystem_snapshot.schema_version != "spritelab_campaign_filesystem_snapshot_v1":
            raise CampaignValidationError("campaign filesystem snapshot schema is unsupported")
        if filesystem_snapshot.campaign_identity != campaign.get("campaign_identity"):
            raise CampaignValidationError("campaign filesystem snapshot belongs to a different campaign")
        states = [_thaw_snapshot_value(state) for state in filesystem_snapshot.runs]
        expected_runs = list(campaign.get("expected_runs") or [])
        expected_ids = [str(run.get("run_id")) for run in expected_runs]
        if [str(state.get("run_id")) for state in states] != expected_ids:
            raise CampaignValidationError("campaign filesystem snapshot does not cover the exact run set")
        from spritelab.product_web.events import EventMigrationState

        allowed_statuses = {
            "fresh": "start",
            "valid_resumable": "resume",
            "complete": "refuse_relaunch",
            "corrupt": "refuse",
            "contradictory": "refuse",
            "foreign": "refuse",
            "partial_invalid": "refuse",
        }
        resume_compatible_migrations = {
            EventMigrationState.NO_MIGRATION.value,
            EventMigrationState.VERIFIED_SOURCE_PRESENT.value,
            EventMigrationState.VERIFIED_SOURCE_REMOVED.value,
        }
        for state, run in zip(states, expected_runs, strict=True):
            status = state.get("status")
            errors = state.get("errors")
            if state.get("output_root") != str(run.get("output_root")):
                raise CampaignValidationError("campaign filesystem snapshot output-root binding changed")
            if status not in allowed_statuses or state.get("next_action") != allowed_statuses[status]:
                raise CampaignValidationError("campaign filesystem snapshot contains an invalid closed run state")
            if not isinstance(errors, list) or any(not isinstance(error, str) or not error for error in errors):
                raise CampaignValidationError("campaign filesystem snapshot contains malformed run errors")
            if status in {"fresh", "valid_resumable"} and errors:
                raise CampaignValidationError("campaign filesystem snapshot marks an executable run as erroneous")
            if status in {"corrupt", "contradictory", "foreign", "partial_invalid"} and not errors:
                raise CampaignValidationError("campaign filesystem snapshot omits its refusal reason")
            migration = state.get("event_migration_verification")
            if (
                not isinstance(migration, Mapping)
                or migration.get("run_id") != str(run.get("run_id"))
                or not isinstance(migration.get("state"), str)
                or not isinstance(migration.get("message"), str)
                or not isinstance(migration.get("details"), Mapping)
                or (migration.get("record") is not None and not isinstance(migration.get("record"), Mapping))
            ):
                raise CampaignValidationError("campaign filesystem snapshot contains malformed event evidence")
            expected_evidence = stable_hash(
                {
                    "schema_version": "spritelab.product.event-migration-evidence.v1",
                    "state": migration["state"],
                    "run_id": str(run.get("run_id")),
                    "details": dict(migration["details"]),
                }
            )
            if not hmac.compare_digest(str(migration.get("evidence_sha256") or ""), expected_evidence):
                raise CampaignValidationError("campaign filesystem snapshot event evidence identity changed")
            if status == "fresh" and dict(migration) != _native_event_migration_verification(str(run.get("run_id"))):
                raise CampaignValidationError("campaign filesystem snapshot contains invalid fresh event evidence")
            if status == "valid_resumable":
                checkpoint = state.get("checkpoint")
                logical_root = Path(str(run.get("output_root")))
                if (
                    not isinstance(checkpoint, str)
                    or Path(checkpoint).parent != logical_root
                    or not is_concrete_hash(state.get("checkpoint_content_sha256"))
                    or type(state.get("resume_step")) is not int
                    or state.get("resume_step") not in set(run.get("expected_checkpoint_steps") or ())
                ):
                    raise CampaignValidationError("campaign filesystem snapshot contains an invalid resume binding")
                if migration["state"] not in resume_compatible_migrations:
                    raise CampaignValidationError(
                        "campaign filesystem snapshot resume event evidence is not compatible"
                    )
        foreign_run_roots = list(filesystem_snapshot.foreign_run_roots)
    root_state = _classify_campaign_roots(states)
    errors = [f"{state['run_id']}: {message}" for state in states for message in state["errors"]]
    if foreign_run_roots:
        root_state = "foreign"
        errors.extend(f"foreign campaign run root: {path}" for path in foreign_run_roots)
    if unsafe_resume:
        errors.append("unsafe resume is forbidden for fair-comparison campaigns")
    if root_state not in {"fresh", "valid_resumable"}:
        errors.append(f"campaign output-root state {root_state!r} is not executable")
    return {
        "schema_version": "spritelab_campaign_resume_audit_v2",
        "campaign_id": campaign.get("campaign_id"),
        "root_state": root_state,
        "allowed_execution_states": ["fresh", "valid_resumable"],
        "safe": not errors and root_state in {"fresh", "valid_resumable"},
        "unsafe_resume_requested": unsafe_resume,
        "errors": errors,
        "foreign_run_roots": foreign_run_roots,
        "runs": states,
    }


def execute_campaign(
    campaign: Mapping[str, Any],
    *,
    execute: bool,
    confirm_execute: bool,
    campaign_config_path: str | Path | None = None,
    campaign_profile: str = "recommended",
    compute_backend_id: str = "local",
    execution_environment: Mapping[str, str] | None = None,
    project_root: str | Path | None = None,
    resume: bool = False,
    unsafe_resume: bool = False,
    launch_authorization_evidence_sha256: str | None = None,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run eligible commands only after all non-interactive safety gates pass."""

    for field, value in (
        ("execute", execute),
        ("confirm_execute", confirm_execute),
        ("resume", resume),
        ("unsafe_resume", unsafe_resume),
    ):
        _require_runtime_boolean(value, field)
    if execute is not True:
        raise CampaignValidationError("execution requires explicit execute=True / --execute")
    if confirm_execute is not True:
        raise CampaignValidationError("execution requires explicit --confirm-execute")
    if campaign.get("executable") is not True or campaign.get("plan_status") != "ready":
        raise CampaignValidationError("campaign is blocked or executable=false")
    if campaign.get("launch_authorized") is not True:
        raise CampaignValidationError("campaign launch_authorized must be true")
    validation = validate_campaign(campaign)
    if not validation["launch_ready"] or validation["errors"] or validation["blockers"]:
        raise CampaignValidationError(
            "campaign validation failed: " + "; ".join([*validation["errors"], *validation["blockers"]])
        )
    resume_report = audit_resume(campaign, unsafe_resume=unsafe_resume)
    if not resume_report["safe"]:
        raise CampaignResumeError("unsafe campaign state: " + "; ".join(resume_report["errors"]))
    if resume_report["root_state"] == "valid_resumable" and resume is not True:
        raise CampaignResumeError("valid resumable campaign requires explicit resume / --resume")
    if campaign_config_path is None:
        raise CampaignValidationError("execution requires the exact authoritative campaign configuration path")
    normalised_environment = {str(key): str(value) for key, value in dict(execution_environment or {}).items()}
    determinism_mode = str(dict(campaign.get("determinism") or {}).get("mode", "off")).strip().lower()
    uses_cuda = any(
        str(dict(run.get("resolved_config") or {}).get("runtime", {}).get("device", "auto")) != "cpu"
        for run in campaign.get("expected_runs") or []
    )
    if determinism_mode in {"strict", "warn"} and uses_cuda:
        cublas_config = normalised_environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if cublas_config not in {":4096:8", ":16:8"}:
            raise CampaignValidationError(
                "strict or warning CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    states = {row["run_id"]: row for row in resume_report["runs"]}
    missing_configs = [
        str(run["resolved_config_path"])
        for run in campaign.get("expected_runs") or []
        if not Path(str(run["resolved_config_path"])).is_file()
    ]
    if missing_configs:
        raise CampaignValidationError("resolved run configs are missing: " + ", ".join(missing_configs))
    changed_configs = []
    for run in campaign.get("expected_runs") or []:
        path = Path(str(run["resolved_config_path"]))
        try:
            actual = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            changed_configs.append(str(path))
            continue
        if stable_hash(actual) != run.get("resolved_config_sha256") or actual != run.get("resolved_config"):
            changed_configs.append(str(path))
        if stable_hash(_execution_contract_payload(run)) != run.get("execution_contract_sha256"):
            changed_configs.append(f"{run.get('run_id')}:execution_contract")
    if changed_configs:
        raise CampaignValidationError("resolved config or execution contract changed: " + ", ".join(changed_configs))
    from spritelab.training.launch import (
        TrainingFilesystemCapability,
        prepare_validated_training_launch,
        verify_validated_training_launch,
    )

    def prepare_run_launch(
        run: Mapping[str, Any],
        *,
        filesystem_snapshot: _CampaignFilesystemSnapshot | None = None,
    ) -> Any:
        state = states[run["run_id"]]
        validated = prepare_validated_training_launch(
            campaign_config_path,
            run_id=str(run["run_id"]),
            compute_backend_id=compute_backend_id,
            project_root=project_root or Path.cwd(),
            execute_confirmed=True,
            campaign_profile=campaign_profile,
            environment=normalised_environment,
            resume=state["status"] == "valid_resumable",
            filesystem_snapshot=filesystem_snapshot,
            launch_authorization_evidence_sha256=launch_authorization_evidence_sha256,
        )
        if validated.receipt.campaign_identity_sha256 != campaign.get("campaign_identity"):
            raise CampaignValidationError("loaded campaign configuration does not match the requested campaign")
        return validated

    launched: list[dict[str, Any]] = []
    if runner is None:
        # One capability retains the campaign directory and every per-seed root
        # for the full sequence.  Each launch inherits only its selected root,
        # so child-created outputs in seed N cannot become baseline drift when
        # seed N+1 is checked against its own still-pristine retained snapshot.
        # Receipts are intentionally issued just before each process: one seed
        # may outlive the short receipt TTL before the next seed begins.
        with TrainingFilesystemCapability(campaign, project_root or Path.cwd()) as capability:
            for run in campaign.get("expected_runs") or []:
                validated = prepare_run_launch(run, filesystem_snapshot=capability.filesystem_snapshot)
                command = list(validated.argv)
                verified = verify_validated_training_launch(
                    validated.receipt,
                    validated.validator_context,
                    compute_backend_id=compute_backend_id,
                    argv=command,
                    environment=validated.environment,
                    output_root=validated.output_root,
                    campaign_identity=str(campaign["campaign_identity"]),
                    run_identity=str(run["run_identity"]),
                    filesystem_snapshot=capability.filesystem_snapshot,
                )
                child_command = capability.bootstrap_command(verified)
                with capability.launch_inheritance(verified) as (boundary_environment, inheritance_options):
                    environment = dict(validated.environment)
                    environment.update(boundary_environment)
                    result = subprocess.run(
                        child_command,
                        check=True,
                        cwd=str(project_root or Path.cwd()),
                        env=environment,
                        **inheritance_options,
                    )
                launched.append(
                    {"run_id": run["run_id"], "command": command, "returncode": getattr(result, "returncode", 0)}
                )
    else:
        for run in campaign.get("expected_runs") or []:
            validated = prepare_run_launch(run)
            command = list(validated.argv)
            result = runner(
                command,
                check=True,
                cwd=str(project_root or Path.cwd()),
                validated_launch=validated,
            )
            launched.append(
                {"run_id": run["run_id"], "command": command, "returncode": getattr(result, "returncode", 0)}
            )
    return {
        "schema_version": "spritelab_campaign_execution_v1",
        "campaign_id": campaign.get("campaign_id"),
        "launched": launched,
        "preserved_completed_runs": [],
        "resume_report": resume_report,
    }


def audit_artifact_completeness(
    campaign: Mapping[str, Any], *, campaign_artifact_root: str | Path | None = None
) -> dict[str, Any]:
    """Require both the per-run contract and the exact campaign artifact contract."""

    runs: list[dict[str, Any]] = []
    missing_all: list[str] = []
    validation = validate_campaign(campaign)
    missing_all.extend(f"campaign:{item}" for item in [*validation["errors"], *validation["blockers"]])
    for run in campaign.get("expected_runs") or []:
        root = Path(str(run["output_root"]))
        mapping = _artifact_paths(root, PER_RUN_ARTIFACTS)
        missing = [name for name, path in mapping.items() if not path.is_file()]
        semantic_errors = _run_completion_errors(run, root, mapping, campaign=campaign) if not missing else []
        missing_all.extend(f"{run['run_id']}:{name}" for name in missing)
        missing_all.extend(f"{run['run_id']}:{message}" for message in semantic_errors)
        runs.append(
            {
                "run_id": run["run_id"],
                "complete": not missing and not semantic_errors,
                "missing": missing,
                "errors": semantic_errors,
            }
        )
    foreign_run_roots = _foreign_run_roots(campaign)
    missing_all.extend(f"foreign_run:{path}" for path in foreign_run_roots)
    configured_root = campaign_artifact_root or campaign.get("campaign_artifact_root")
    campaign_audit = _audit_campaign_artifacts(campaign, configured_root)
    missing_all.extend(f"campaign:{item}" for item in campaign_audit["missing"])
    missing_all.extend(f"campaign:{item}" for item in campaign_audit["errors"])
    complete = not missing_all and campaign_audit["complete"] and all(item["complete"] for item in runs)
    if complete:
        status = "complete"
        comparability = "comparable"
    elif campaign_audit["status"] == "not_comparable":
        status = "not_comparable"
        comparability = "not_comparable"
    else:
        status = "incomplete"
        comparability = campaign_audit.get("comparability", "incomplete")
    return {
        "schema_version": "spritelab_campaign_artifact_audit_v3",
        "campaign_id": campaign.get("campaign_id"),
        "complete": complete,
        "status": status,
        "comparability": comparability,
        "runs": runs,
        "campaign_artifacts": campaign_audit,
        "campaign_missing": campaign_audit["missing"],
        "foreign_run_roots": foreign_run_roots,
        "validation": validation,
        "missing": missing_all,
        "reasons": missing_all,
    }


def _campaign_run_matrix(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.get("run_id"),
            "run_identity": run.get("run_identity"),
            "cell_id": run.get("cell_id"),
            "seed": run.get("seed"),
        }
        for run in campaign.get("expected_runs", ())
    ]


def _audit_campaign_artifacts(campaign: Mapping[str, Any], root_value: str | Path | None) -> dict[str, Any]:
    missing: list[str] = []
    errors: list[str] = []
    if root_value is None or not str(root_value).strip():
        return {
            "schema_version": CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "complete": False,
            "status": "incomplete",
            "comparability": "legacy_incomplete",
            "root": None,
            "manifest": None,
            "manifest_sha256": None,
            "missing": ["missing_campaign_artifact_root"],
            "errors": [],
        }
    root = Path(root_value)
    if not root.exists():
        missing.append("campaign_artifact_root")
    elif root.is_symlink() or not root.is_dir():
        errors.append("campaign_artifact_root is not a canonical directory")
    if missing or errors:
        return {
            "schema_version": CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "complete": False,
            "status": "not_comparable" if errors else "incomplete",
            "comparability": "not_comparable" if errors else "incomplete",
            "root": str(root),
            "manifest": None,
            "manifest_sha256": None,
            "missing": missing,
            "errors": errors,
        }
    root_resolved = root.resolve()
    manifest_path = root / CAMPAIGN_ARTIFACT_MANIFEST_NAME
    if not manifest_path.exists():
        missing.append(CAMPAIGN_ARTIFACT_MANIFEST_NAME)
    elif manifest_path.is_symlink() or not manifest_path.is_file():
        errors.append("campaign artifact manifest is not a regular file")
    if missing or errors:
        return {
            "schema_version": CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "complete": False,
            "status": "not_comparable" if errors else "incomplete",
            "comparability": "not_comparable" if errors else "incomplete",
            "root": str(root),
            "manifest": str(manifest_path),
            "manifest_sha256": None,
            "missing": missing,
            "errors": errors,
        }
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid campaign artifact manifest: {exc}")
        manifest = {}
    campaign_identity = campaign.get("campaign_identity")
    code_identity = dict(campaign.get("code_identity") or {}).get("sha256")
    expected_run_ids = list(campaign.get("expected_run_ids") or [])
    expected_seeds = list(campaign.get("seeds") or [])
    run_matrix_sha256 = stable_hash(_campaign_run_matrix(campaign))
    manifest_sha256 = file_sha256(manifest_path)
    completion_report_path = root / CAMPAIGN_COMPLETION_REPORT_NAME
    if not completion_report_path.exists():
        missing.append(CAMPAIGN_COMPLETION_REPORT_NAME)
    elif completion_report_path.is_symlink() or not completion_report_path.is_file():
        errors.append("campaign completion report is not a regular file")
    else:
        try:
            completion_report = _read_json(completion_report_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid campaign completion report: {exc}")
        else:
            for field, expected in (
                ("schema_version", CAMPAIGN_COMPLETION_REPORT_SCHEMA_VERSION),
                ("campaign_identity_sha256", campaign_identity),
                ("training_code_identity_sha256", code_identity),
                ("campaign_artifact_manifest_sha256", manifest_sha256),
                ("expected_run_ids", expected_run_ids),
                ("seeds", expected_seeds),
                ("run_matrix_sha256", run_matrix_sha256),
                ("complete", True),
            ):
                if completion_report.get(field) != expected:
                    errors.append(f"campaign completion report {field} does not match")
    for field, expected in (
        ("schema_version", CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION),
        ("campaign_identity_sha256", campaign_identity),
        ("training_code_identity_sha256", code_identity),
        ("expected_run_ids", expected_run_ids),
        ("seeds", expected_seeds),
        ("run_matrix_sha256", run_matrix_sha256),
    ):
        if manifest.get(field) != expected:
            errors.append(f"campaign artifact manifest {field} does not match")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        errors.append("campaign artifact manifest entries are missing or empty")
        entries = []
    by_type: dict[str, Mapping[str, Any]] = {}
    seen_paths: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            errors.append(f"campaign artifact manifest entry {index} is not an object")
            continue
        artifact_type = str(raw_entry.get("artifact_type") or "")
        relative = raw_entry.get("relative_path")
        if artifact_type in by_type:
            errors.append(f"duplicate campaign artifact type: {artifact_type}")
        else:
            by_type[artifact_type] = raw_entry
        if not isinstance(relative, str) or not _safe_relative_artifact_path(relative):
            errors.append(f"campaign artifact entry has unsafe relative path: {relative!r}")
            continue
        key = PurePosixPath(relative).as_posix().casefold()
        if key in seen_paths:
            errors.append(f"duplicate campaign artifact relative path: {relative}")
        seen_paths.add(key)
    if set(by_type) != set(CAMPAIGN_ARTIFACTS):
        for artifact_type in sorted(set(CAMPAIGN_ARTIFACTS) - set(by_type)):
            missing.append(artifact_type)
        for artifact_type in sorted(set(by_type) - set(CAMPAIGN_ARTIFACTS)):
            errors.append(f"unexpected campaign artifact type: {artifact_type}")
    expected_root_entries = {
        CAMPAIGN_ARTIFACT_MANIFEST_NAME,
        CAMPAIGN_COMPLETION_REPORT_NAME,
        *(f"{artifact_type}.json" for artifact_type in CAMPAIGN_ARTIFACTS),
    }
    try:
        unexpected_root_entries = sorted(path.name for path in root.iterdir() if path.name not in expected_root_entries)
    except OSError as exc:
        errors.append(f"campaign artifact root cannot be enumerated: {exc}")
    else:
        errors.extend(f"unexpected campaign artifact root entry: {name}" for name in unexpected_root_entries)
    for artifact_type in CAMPAIGN_ARTIFACTS:
        entry = by_type.get(artifact_type)
        if entry is None:
            continue
        expected_relative = f"{artifact_type}.json"
        relative = entry.get("relative_path")
        if relative != expected_relative:
            errors.append(f"campaign artifact {artifact_type} has an unexpected replacement path")
            continue
        if entry.get("required") is not True:
            errors.append(f"campaign artifact {artifact_type} is not marked required")
        if entry.get("expected_role") != artifact_type:
            errors.append(f"campaign artifact {artifact_type} has the wrong expected role")
        if not isinstance(entry.get("producing_stage"), str) or not entry["producing_stage"].strip():
            errors.append(f"campaign artifact {artifact_type} has no producing stage")
        for field, expected in (
            ("campaign_identity_sha256", campaign_identity),
            ("training_code_identity_sha256", code_identity),
            ("run_matrix_sha256", run_matrix_sha256),
        ):
            if entry.get(field) != expected:
                errors.append(f"campaign artifact {artifact_type} {field} does not match")
        declared_hash = entry.get("content_sha256")
        if not is_concrete_hash(declared_hash):
            errors.append(f"campaign artifact {artifact_type} has a missing or malformed content hash")
        schema_version = entry.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            errors.append(f"campaign artifact {artifact_type} has no schema version")
        path = root / expected_relative
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            errors.append(f"campaign artifact {artifact_type} cannot be resolved: {exc}")
            continue
        if resolved == root_resolved or root_resolved not in resolved.parents:
            errors.append(f"campaign artifact {artifact_type} escapes the canonical artifact root")
            continue
        if not path.exists():
            missing.append(artifact_type)
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"campaign artifact {artifact_type} is not a regular file")
            continue
        if is_concrete_hash(declared_hash) and file_sha256(path) != declared_hash:
            errors.append(f"campaign artifact {artifact_type} content hash mismatch")
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"campaign artifact {artifact_type} has invalid JSON schema content: {exc}")
            continue
        for field, expected in (
            ("schema_version", schema_version),
            ("campaign_identity_sha256", campaign_identity),
            ("training_code_identity_sha256", code_identity),
            ("expected_run_ids", expected_run_ids),
            ("seeds", expected_seeds),
            ("run_matrix_sha256", run_matrix_sha256),
        ):
            if payload.get(field) != expected:
                errors.append(f"campaign artifact {artifact_type} payload {field} does not match")
    status = "complete" if not missing and not errors else ("not_comparable" if errors else "incomplete")
    return {
        "schema_version": CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "complete": status == "complete",
        "status": status,
        "comparability": "comparable" if status == "complete" else status,
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "completion_report": str(completion_report_path),
        "missing": missing,
        "errors": errors,
        "required_artifacts": list(CAMPAIGN_ARTIFACTS),
        "run_matrix_sha256": run_matrix_sha256,
    }


def aggregate_cross_seed_metrics(
    campaign: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    campaign_artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate only after exact run, identity, seed, and artifact validation."""

    rows: list[dict[str, Any]] = []
    expected = list(campaign.get("expected_runs") or [])
    expected_ids = {str(run["run_id"]) for run in expected}
    report_ids = set(map(str, reports))
    missing_runs = sorted(expected_ids - report_ids)
    foreign_runs = sorted(report_ids - expected_ids)
    identity_errors: list[str] = []
    duplicate_seeds: list[dict[str, Any]] = []
    by_cell: dict[str, dict[int, Mapping[str, Any]]] = {}
    seen_seeds: dict[tuple[str, int], str] = {}
    for run in expected:
        report = reports.get(str(run["run_id"]))
        if report is None:
            continue
        run_id = str(run["run_id"])
        cell_id = str(run["cell_id"])
        seed = report.get("seed")
        if type(seed) is not int or seed != run.get("seed"):
            identity_errors.append(f"{run_id}: report seed does not match expected seed")
        if report.get("run_identity") != run.get("run_identity"):
            identity_errors.append(f"{run_id}: report run identity does not match")
        if report.get("campaign_identity") != campaign.get("campaign_identity"):
            identity_errors.append(f"{run_id}: report campaign identity does not match")
        if type(seed) is int:
            key = (cell_id, seed)
            if key in seen_seeds:
                duplicate_seeds.append({"cell_id": cell_id, "seed": seed, "run_ids": [seen_seeds[key], run_id]})
            else:
                seen_seeds[key] = run_id
        by_cell.setdefault(cell_id, {})[int(run["seed"])] = report
    artifact_audit = audit_artifact_completeness(campaign, campaign_artifact_root=campaign_artifact_root)
    preliminary_reasons = [
        *(f"missing report: {item}" for item in missing_runs),
        *(f"foreign report: {item}" for item in foreign_runs),
        *identity_errors,
        *(f"duplicate seed report: {item}" for item in duplicate_seeds),
        *(f"artifact incomplete: {item}" for item in artifact_audit["missing"]),
    ]
    preliminarily_complete = not preliminary_reasons
    if not preliminarily_complete:
        return {
            "schema_version": "spritelab_cross_seed_aggregate_v3",
            "campaign_id": campaign.get("campaign_id"),
            "required_seed_count": REQUIRED_SEED_COUNT,
            "per_seed_metrics": [],
            "missing_runs": missing_runs,
            "foreign_runs": foreign_runs,
            "duplicate_seeds": duplicate_seeds,
            "identity_errors": identity_errors,
            "incomplete_reasons": preliminary_reasons,
            "not_comparable_reasons": (
                artifact_audit["reasons"] if artifact_audit["status"] == "not_comparable" else []
            ),
            "complete": False,
            "artifact_completion": artifact_audit,
            "metrics_aggregated": False,
            "promotion_eligible": False,
            "promotion_status": "blocked: per-run and campaign-level artifact contracts must pass first",
        }
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
                if (
                    metric is None
                    or definition is None
                    or isinstance(metric, bool)
                    or not isinstance(metric, (int, float))
                ):
                    missing_metric_seeds.append(int(seed))
                else:
                    per_seed[str(seed)] = float(metric)
                    definitions[stable_hash(definition)] = definition
            compatible = len(definitions) <= 1
            values = [per_seed[key] for key in sorted(per_seed, key=int)]
            complete = (
                preliminarily_complete
                and not missing_metric_seeds
                and len(values) == REQUIRED_SEED_COUNT
                and compatible
            )
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
    metric_incomplete_reasons = [
        f"{row['cell_id']}:{row['metric']}: missing seeds {row['missing_metric_seeds']}"
        for row in rows
        if row["missing_metric_seeds"]
    ]
    complete = (
        preliminarily_complete
        and len(expected) == len(campaign.get("architecture_cells") or []) * REQUIRED_SEED_COUNT
        and bool(rows)
        and all(row["complete"] for row in rows)
    )
    not_comparable = [
        f"{row['cell_id']}:{row['metric']}: incompatible metric definitions" for row in rows if not row["compatible"]
    ]
    return {
        "schema_version": "spritelab_cross_seed_aggregate_v3",
        "campaign_id": campaign.get("campaign_id"),
        "required_seed_count": REQUIRED_SEED_COUNT,
        "per_seed_metrics": rows,
        "missing_runs": missing_runs,
        "foreign_runs": foreign_runs,
        "duplicate_seeds": duplicate_seeds,
        "identity_errors": identity_errors,
        "incomplete_reasons": [*preliminary_reasons, *metric_incomplete_reasons],
        "not_comparable_reasons": not_comparable,
        "complete": complete,
        "artifact_completion": artifact_audit,
        "metrics_aggregated": True,
        "promotion_eligible": False,
        "promotion_status": (
            "blocked: complete compatible three-seed evidence and independent promotion authorization required"
            if not complete
            else "complete evidence; promotion remains blocked pending independent authorization"
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
        "resolved_config_sha256": run.get("resolved_config_sha256"),
        "execution_contract_sha256": run.get("execution_contract_sha256"),
        "code_identity": run.get("code_identity"),
        "command": run.get("experiment_command"),
    }


def _campaign_identity_payload(campaign: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"campaign_identity"}
    return {key: value for key, value in campaign.items() if key not in excluded}


def _resolved_run_config(
    raw: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    seed: int,
    output_root: str,
    execution_root: Path,
) -> dict[str, Any]:
    """Translate one campaign run into the schema consumed by experiment run."""

    campaign = {
        "model": deepcopy(raw.get("model") or {}),
        "training": deepcopy(raw.get("training") or {}),
        "optimizer": deepcopy(raw.get("optimizer") or {}),
        "schedule": deepcopy(raw.get("schedule") or {}),
        "loss": deepcopy(raw.get("loss") or {}),
        "determinism": deepcopy(raw.get("determinism") or {}),
        "evaluation": deepcopy(raw.get("evaluation") or {}),
        "checkpoint": deepcopy(raw.get("checkpoint") or {}),
        "identities": deepcopy(raw.get("identities") or {}),
    }
    for dotted, replacement in dict(cell.get("overrides") or {}).items():
        _set_dotted(campaign, str(dotted), deepcopy(replacement))
    for dotted, replacement in dict(cell.get("comparison_values") or {}).items():
        path = str(dotted)
        if "." not in path:
            path = f"model.{path}"
        _set_dotted(campaign, path, deepcopy(replacement))

    identities = dict(campaign["identities"])
    manifest_path = Path(str(identities.get("split_manifest_path") or "training_manifest.jsonl"))
    vocabulary_path = Path(str(identities.get("conditioning_vocabulary_path") or "conditioning_vocabulary.json"))

    def portable(path: str | Path) -> str:
        return os.path.relpath(Path(path), start=execution_root).replace("\\", "/")

    model = {
        "architecture": "rectified_flow",
        "sprite_size": 32,
        "base_channels": 64,
        "channel_mults": [1, 2, 4],
        "res_blocks_per_level": 2,
        "embed_dim": 64,
        "film_conditioning": False,
        "bottleneck_attention": False,
        "auxiliary_heads_mode": "absent",
    }
    model.update(dict(campaign["model"]))
    if model.get("auxiliary_heads_mode") == "off":
        model["auxiliary_heads_mode"] = "absent"
    elif model.get("auxiliary_heads_mode") == "on":
        model["auxiliary_heads_mode"] = "palette_index"

    training = dict(campaign["training"])
    schedule = dict(campaign["schedule"])
    optimizer = {
        "name": "adamw",
        "learning_rate": 0.0002,
        "schedule": str(schedule.get("name", "none")),
        "warmup_steps": int(schedule.get("warmup_steps", 0)),
        "gradient_clip": 0.0,
    }
    optimizer.update(dict(campaign["optimizer"]))
    optimizer["schedule"] = str(schedule.get("name", optimizer.get("schedule", "none")))
    optimizer["warmup_steps"] = int(schedule.get("warmup_steps", optimizer.get("warmup_steps", 0)))

    loss = {
        "strategy": "uniform_velocity",
        "foreground_rgb_weight": 1.0,
        "background_rgb_weight": 1.0,
        "palette_aux_weight": 0.0,
        "auxiliary_heads": False,
        "index_head_weight": 0.0,
        "palette_head_weight": 0.0,
        "palette_presence_weight": 0.0,
    }
    loss.update(dict(campaign["loss"]))
    loss["strategy"] = str(loss.get("strategy") or loss.get("name") or "uniform_velocity")

    evaluation = dict(campaign["evaluation"])
    checkpoint = dict(campaign["checkpoint"])
    determinism = dict(campaign["determinism"])
    resolved: dict[str, Any] = {
        "schema_version": "spritelab_experiment_config_v1",
        "name": f"{raw.get('campaign_id')}__{cell.get('cell_id')}__seed_{seed}",
        "ablation": "baseline",
        "dataset": {
            "directory": portable(manifest_path.parent),
            "training_manifest": portable(manifest_path),
            "split_manifest": portable(manifest_path),
            "split": "train",
        },
        "model": model,
        "conditioning": {
            "mode": "caption_semantic",
            "caption_max_length": 32,
            "semantic_max_length": 48,
            "cfg_dropout": 0.1,
            "field_dropout": {},
            "vocabulary_path": portable(vocabulary_path),
            "palette": {"enabled": False, "dropout": 0.0, "strength": 1.0},
        },
        "loss": loss,
        "optimizer": optimizer,
        "augmentation": {"palette_swap_probability": 0.0, "horizontal_flip_probability": 0.0},
        "seeds": {
            "training": seed,
            "data_loader": seed,
            "evaluation": seed + 1,
            "sampling": seed,
        },
        "runtime": {
            "out_dir": portable(output_root),
            "device": str(training.get("device", "auto")),
            "precision": str(training.get("precision", "fp32")),
            "batch_size": int(training.get("micro_batch_size", 1)),
            "micro_batch_size": int(training.get("micro_batch_size", 1)),
            "gradient_accumulation_steps": int(training.get("gradient_accumulation", 1)),
            "effective_batch_size": int(training.get("effective_batch_size", 1)),
            "max_steps": int(training.get("max_optimizer_steps", 1)),
            "num_workers": 0,
            "validation_mode": "auto",
            "sample_every": int(evaluation.get("cadence", 0)),
            "save_every": int(checkpoint.get("cadence", 0)),
            "determinism": str(determinism.get("mode", "off")),
        },
        "sampling": {
            "cfg_scale": float(evaluation.get("cfg_value", 3.0)),
            "steps": int(evaluation.get("sampling_steps", 30)),
            "max_samples": 64,
        },
        "ema": {"enabled": True, "decay": 0.999, "update_every": 1},
        "sampler": {"name": "stateful_permutation_v1", "shuffle": True},
        "timestep_sampling": {"strategy": "uniform"},
        "noise_schedule": "rectified_flow_linear_path",
        "self_conditioning": False,
        "campaign_bindings": {
            key: identities.get(key)
            for key in (
                "dataset_view_manifest_hash",
                "split_manifest_hash",
                "conditioning_vocabulary_hash",
                "model_config_hash",
                "optimizer_config_hash",
                "schedule_config_hash",
                "loss_config_hash",
                "determinism_config_hash",
            )
        },
        "campaign": {
            "campaign_id": raw.get("campaign_id"),
            "cell_id": cell.get("cell_id"),
            "seed": seed,
        },
        # Retain the authoritative campaign projections used by completion and
        # resume auditing. The experiment loader permits these bound extras.
        "training": training,
        "schedule": schedule,
        "determinism": determinism,
        "evaluation": evaluation,
        "checkpoint": checkpoint,
    }
    experiment_overrides = raw.get("experiment")
    if isinstance(experiment_overrides, Mapping):
        _deep_update(resolved, experiment_overrides)
    return resolved


def _deep_update(target: dict[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        current = target.get(str(key))
        if isinstance(current, dict) and isinstance(value, Mapping):
            _deep_update(current, value)
        else:
            target[str(key)] = deepcopy(value)


def _execution_contract_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "argv": list(run.get("experiment_command") or []),
        "code_identity": run.get("code_identity"),
        "cell_id": run.get("cell_id"),
        "seed": run.get("seed"),
        "output_root": run.get("output_root"),
        "resolved_config_sha256": run.get("resolved_config_sha256"),
    }


def training_code_identity_source_paths(repo_root: str | Path | None = None) -> tuple[Path, ...]:
    """Bind all tracked production Python and reject an incomplete worktree inventory."""

    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    recursive_roots = tuple(root / relative for relative in TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS)
    missing = [path for path in recursive_roots if not path.is_dir()]
    if missing:
        raise CampaignValidationError(
            "bound code-identity source root is missing: " + ", ".join(str(path) for path in missing)
        )
    relative_roots = [path.relative_to(root).as_posix() for path in recursive_roots]
    mandatory = tuple(root / relative for relative in TRAINING_CODE_IDENTITY_MANDATORY_FILES)
    missing_files = [path for path in mandatory if not path.is_file()]
    if missing_files:
        raise CampaignValidationError(
            "bound code-identity source is missing: "
            + ", ".join(path.relative_to(root).as_posix() for path in missing_files)
        )
    try:
        inventory = subprocess.Popen(
            ["git", "ls-files", "-z", "--", *relative_roots, *TRAINING_CODE_IDENTITY_MANDATORY_FILES],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _stderr = inventory.communicate(timeout=10)
    except subprocess.TimeoutExpired as exc:
        inventory.kill()
        inventory.communicate()
        raise CampaignValidationError("tracked code-identity source inventory timed out") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise CampaignValidationError(f"tracked code-identity source inventory failed: {exc}") from exc
    if inventory.returncode != 0:
        raise CampaignValidationError("tracked code-identity source inventory failed")
    tracked = {
        root / relative for raw in stdout.split(b"\0") if raw and (relative := raw.decode("utf-8")).endswith(".py")
    }
    missing_tracked = [path for path in tracked if not path.is_file()]
    if missing_tracked:
        raise CampaignValidationError(
            "tracked production Python source is missing: "
            + ", ".join(path.relative_to(root).as_posix() for path in sorted(missing_tracked))
        )
    source_root = root / "src" / "spritelab"
    discovered: set[Path] = set()
    pending = [source_root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as exc:
            raise CampaignValidationError("production Python inventory could not be scanned safely") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CampaignValidationError("production Python inventory changed while scanning") from exc
            attributes = int(getattr(metadata, "st_file_attributes", 0))
            reparse = bool(attributes & 0x400)
            if entry.is_symlink() or reparse or os.path.ismount(path):
                raise CampaignValidationError(
                    f"production source inventory crosses an unsafe seam: {path.relative_to(root).as_posix()}"
                )
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False) and path.suffix == ".py":
                discovered.add(path)
    untracked = discovered - tracked
    if untracked:
        raise CampaignValidationError(
            "untracked production Python source would escape code identity: "
            + ", ".join(path.relative_to(root).as_posix() for path in sorted(untracked))
        )
    if not set(mandatory).issubset(tracked):
        absent = set(mandatory) - tracked
        raise CampaignValidationError(
            "mandatory production Python source is not tracked: "
            + ", ".join(path.relative_to(root).as_posix() for path in sorted(absent))
        )
    return tuple(sorted(tracked, key=lambda path: path.relative_to(root).as_posix()))


def _code_identity() -> dict[str, Any]:
    """Bind all production code that can plan, authorize, launch, resume, or complete training."""

    repo_root = Path(__file__).resolve().parents[3]
    bound = training_code_identity_source_paths(repo_root)
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in sorted(bound):
        relative = path.relative_to(repo_root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        if not path.is_file():
            raise CampaignValidationError(f"bound code-identity source is missing: {relative}")
        records.append(
            {
                "path": relative,
                "binding": "whole_file",
                "semantic_role": TRAINING_CODE_IDENTITY_SEMANTIC_ROLES.get(
                    relative, "training planning, authorization, execution, reconstruction, or completion"
                ),
                "sha256": file_sha256(path),
            }
        )
    if not records:
        raise CampaignValidationError("code identity has no bound production sources")
    payload = {
        "schema_version": CODE_IDENTITY_SCHEMA_VERSION,
        "contract": "all_tracked_production_python_v5_with_untracked_rejection",
        "files": records,
    }
    payload["sha256"] = stable_hash(payload)
    return payload


def _evaluation_identity_payload(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"evaluation_config_hash", "benchmark_manifest_hash", "benchmark_manifest_path"}
    }


def _validate_file_binding(
    values: Mapping[str, Any],
    hash_field: str,
    path_field: str,
    errors: list[str],
    *,
    prefix: str = "identity ",
    file_sha256_resolver: Callable[[Path], str] | None = None,
) -> None:
    path_value = values.get(path_field)
    if not path_value:
        errors.append(f"{prefix}{path_field} is required to verify {hash_field}")
        return
    path = Path(str(path_value))
    if file_sha256_resolver is not None:
        try:
            actual_sha256 = file_sha256_resolver(path)
        except (KeyError, OSError, ValueError):
            errors.append(f"{prefix}{path_field} is not present in the verified filesystem snapshot")
            return
    elif not path.is_file():
        errors.append(f"{prefix}{path_field} does not exist: {path}")
        return
    else:
        actual_sha256 = file_sha256(path)
    if values.get(hash_field) != actual_sha256:
        errors.append(f"{prefix}{hash_field} does not match actual file content")


def _classify_run_root(
    campaign: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    root_override: Path | None = None,
) -> dict[str, Any]:
    logical_root = Path(str(run["output_root"]))
    root = logical_root if root_override is None else root_override
    state: dict[str, Any] = {
        "run_id": run["run_id"],
        "output_root": str(logical_root),
        "status": "fresh",
        "next_action": "start",
        "errors": [],
        "event_migration_verification": _native_event_migration_verification(str(run["run_id"])),
    }
    if not root.exists():
        return state
    if not root.is_dir():
        state.update(status="corrupt", next_action="refuse")
        state["errors"].append("output root exists but is not a directory")
        return state
    try:
        children = list(root.iterdir())
    except OSError as exc:
        state.update(status="corrupt", next_action="refuse")
        state["errors"].append(f"output root cannot be inspected: {exc}")
        return state
    if not children:
        return state

    identity_path = root / "run_identity.json"
    if not identity_path.is_file():
        state.update(status="foreign", next_action="refuse")
        state["errors"].append("existing output root has no run_identity.json and is foreign or unowned")
        return state
    try:
        identity = _read_json(identity_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        state.update(status="corrupt", next_action="refuse")
        state["errors"].append(f"invalid run identity: {exc}")
        return state
    expected_identity = {
        "campaign_id": campaign.get("campaign_id"),
        "campaign_identity": campaign.get("campaign_identity"),
        "run_id": run.get("run_id"),
        "run_identity": run.get("run_identity"),
        "output_root": str(logical_root),
        "resolved_config_sha256": run.get("resolved_config_sha256"),
        "execution_contract_sha256": run.get("execution_contract_sha256"),
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            state["errors"].append(f"output root {field} identity does not match")
    if identity.get("unsafe_resume") or identity.get("unsafe_resume_record"):
        state["errors"].append("output root contains an unsafe resume revocation")
    revocation_dir = root / "unsafe_resume_revocations"
    if revocation_dir.exists():
        if not revocation_dir.is_dir():
            state.update(status="corrupt", next_action="refuse")
            state["errors"].append("unsafe resume revocation path is not a directory")
            return state
        try:
            if any(revocation_dir.iterdir()):
                state["errors"].append("output root contains append-only unsafe resume revocations")
        except OSError as exc:
            state.update(status="corrupt", next_action="refuse")
            state["errors"].append(f"unsafe resume revocations cannot be inspected: {exc}")
            return state
    if state["errors"]:
        state.update(status="foreign", next_action="refuse")
        return state

    from spritelab.product_web.events import verify_event_migration

    migration = verify_event_migration(str(run["run_id"]), root, origin_required=True)
    state["event_migration_state"] = migration.state.value
    state["event_history_origin"] = migration.event_history_origin
    state["event_migration_verification"] = {
        "state": migration.state.value,
        "run_id": migration.run_id,
        "evidence_sha256": migration.evidence_sha256,
        "message": migration.message,
        "record": deepcopy(migration.record),
        "details": deepcopy(dict(migration.details or {})),
    }
    if not migration.resume_compatible:
        state.update(status="partial_invalid", next_action="refuse")
        state["errors"].append(
            f"event history evidence is not safe for continuation: {migration.state.value}: {migration.message}"
        )
        return state

    completion = root / "run_completion_marker.json"
    if completion.exists():
        try:
            marker = _read_json(completion)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.update(status="corrupt", next_action="refuse")
            state["errors"].append(f"invalid completion marker: {exc}")
            return state
        if marker.get("complete") is not True or marker.get("failed") or marker.get("partial"):
            state.update(status="partial_invalid", next_action="refuse")
            state["errors"].append("completion marker is failed, partial, or not complete")
            return state
        mapping = _artifact_paths(root, PER_RUN_ARTIFACTS)
        missing = [name for name, path in mapping.items() if not path.is_file()]
        completion_errors = [] if missing else _run_completion_errors(run, root, mapping, campaign=campaign)
        if missing or completion_errors:
            state.update(status="contradictory", next_action="refuse")
            state["errors"].extend(f"completion marker contradicts missing artifact: {name}" for name in missing)
            state["errors"].extend(completion_errors)
            return state
        state.update(status="complete", next_action="refuse_relaunch")
        return state

    checkpoint_files = sorted(root.glob("checkpoint_step_*.json"))
    if not checkpoint_files:
        state.update(status="partial_invalid", next_action="refuse")
        state["errors"].append(
            "existing incomplete run has no exact-resume checkpoint; refusing restart from step zero"
        )
        return state
    parsed: list[tuple[int, Path, str, str]] = []
    expected_steps = set(run.get("expected_checkpoint_steps") or [])
    for checkpoint_path in checkpoint_files:
        try:
            checkpoint = _read_json(checkpoint_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state.update(status="corrupt", next_action="refuse")
            state["errors"].append(f"invalid checkpoint identity {checkpoint_path.name}: {exc}")
            continue
        for field, expected in (
            ("campaign_identity", campaign.get("campaign_identity")),
            ("run_identity", run.get("run_identity")),
        ):
            if checkpoint.get(field) != expected:
                state["errors"].append(
                    f"existing checkpoint {field.replace('_', ' ')} does not match: {checkpoint_path.name}"
                )
        metadata = checkpoint.get("resumability_metadata")
        step = checkpoint.get("optimizer_step")
        if type(step) is not int or step not in expected_steps:
            state["errors"].append(f"checkpoint step is not in the campaign schedule: {checkpoint_path.name}")
            continue
        filename_step = _resume_sidecar_step(checkpoint_path.name)
        if filename_step != step:
            state["errors"].append(
                f"checkpoint filename step does not match its optimizer step: {checkpoint_path.name}"
            )
        actual_checkpoint, metadata_errors = _validate_resumability_metadata(
            campaign, run, root, metadata, sidecar=checkpoint_path
        )
        state["errors"].extend(metadata_errors)
        declared_checkpoint_sha256 = (
            metadata.get("checkpoint_content_sha256") if isinstance(metadata, Mapping) else None
        )
        if actual_checkpoint is not None and is_concrete_hash(declared_checkpoint_sha256):
            parsed.append(
                (
                    step,
                    actual_checkpoint,
                    str(declared_checkpoint_sha256),
                    str(metadata["checkpoint_relative_path"]),
                )
            )
    if state["errors"]:
        if state["status"] != "corrupt":
            state.update(status="partial_invalid", next_action="refuse")
        return state
    if not parsed:
        state.update(status="partial_invalid", next_action="refuse")
        state["errors"].append("no checkpoint satisfies the exact campaign resume contract")
        return state
    latest_step, _latest_physical, latest_sha256, latest_relative = max(parsed)
    latest = logical_root.joinpath(*PurePosixPath(latest_relative).parts)
    state.update(
        status="valid_resumable",
        next_action="resume",
        checkpoint=str(latest),
        checkpoint_content_sha256=latest_sha256,
        resume_step=latest_step,
    )
    return state


def _classify_campaign_roots(states: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(state.get("status")) for state in states]
    for terminal in ("corrupt", "contradictory", "foreign", "partial_invalid"):
        if terminal in statuses:
            return terminal
    if statuses and all(status == "fresh" for status in statuses):
        return "fresh"
    if statuses and all(status == "complete" for status in statuses):
        return "complete"
    if "complete" in statuses:
        return "partial_valid"
    if statuses and set(statuses).issubset({"fresh", "valid_resumable"}):
        return "valid_resumable"
    return "partial_invalid"


def _resume_sidecar_step(name: str) -> int | None:
    match = re.fullmatch(r"checkpoint_step_(\d+)\.json", name)
    return None if match is None else int(match.group(1))


def _validate_resumability_metadata(
    campaign: Mapping[str, Any],
    run: Mapping[str, Any],
    root: Path,
    metadata: Any,
    *,
    sidecar: Path,
) -> tuple[Path | None, list[str]]:
    """Verify a resume sidecar binds a real checkpoint and complete replay state."""

    if not isinstance(metadata, Mapping):
        return None, [f"existing checkpoint is missing resumability metadata: {sidecar.name}"]
    errors: list[str] = []
    if metadata.get("schema_version") != RESUME_CHECKPOINT_SCHEMA_VERSION:
        errors.append(f"resumability metadata schema is missing or unsupported: {sidecar.name}")
    relative = metadata.get("checkpoint_relative_path")
    checkpoint_path: Path | None = None
    if not isinstance(relative, str) or not _safe_relative_artifact_path(relative):
        errors.append(f"resumability metadata has an unsafe checkpoint path: {sidecar.name}")
    else:
        checkpoint_path = (root / Path(*PurePosixPath(relative).parts)).resolve()
        resolved_root = root.resolve()
        if checkpoint_path == resolved_root or resolved_root not in checkpoint_path.parents:
            errors.append(f"resumability checkpoint escapes its output root: {sidecar.name}")
            checkpoint_path = None
        elif not checkpoint_path.is_file():
            errors.append(f"resumability checkpoint file is missing: {relative}")
    declared_hash = metadata.get("checkpoint_content_sha256")
    if not is_concrete_hash(declared_hash):
        errors.append(f"resumability checkpoint hash is missing or malformed: {sidecar.name}")
    elif checkpoint_path is not None and file_sha256(checkpoint_path) != declared_hash:
        errors.append(f"resumability checkpoint content hash does not match: {relative}")
    if metadata.get("source_checkpoint_identity") != declared_hash:
        errors.append(f"resumability source checkpoint identity does not match: {sidecar.name}")
    if metadata.get("target_runtime_identity") != run.get("run_identity"):
        errors.append(f"resumability target runtime identity does not match: {sidecar.name}")
    if not is_concrete_hash(metadata.get("experiment_manifest_identity")):
        errors.append(f"resumability experiment manifest identity is missing: {sidecar.name}")
    if metadata.get("exact_replay_eligible") is not True:
        errors.append(f"resumability metadata is not exact-replay eligible: {sidecar.name}")
    if metadata.get("unsafe_resume") is not False:
        errors.append(f"resumability metadata is unsafe or ambiguous: {sidecar.name}")
    if metadata.get("max_optimizer_steps") != dict(campaign.get("training") or {}).get("max_optimizer_steps"):
        errors.append(f"resumability max optimizer steps does not match: {sidecar.name}")
    if metadata.get("gradient_accumulation_position") != 0:
        errors.append(f"resumability accumulation position is not at a safe boundary: {sidecar.name}")
    state_presence = metadata.get("state_presence")
    required_state = {
        "model_state_dict",
        "optimizer_state_dict",
        "rng_states",
        "sampler_state",
        "dataloader_generator_state",
    }
    schedule_name = str(dict(campaign.get("schedule") or {}).get("name", "none")).strip().lower()
    if schedule_name != "none":
        required_state.add("scheduler_state_dict")
    if dict(campaign.get("evaluation") or {}).get("ema_policy") in {"ema", "both"}:
        required_state.add("ema_state_dict")
    if not isinstance(state_presence, Mapping):
        errors.append(f"resumability state-presence section is missing: {sidecar.name}")
    else:
        for field in sorted(required_state):
            if state_presence.get(field) is not True:
                errors.append(f"resumability state is incomplete ({field}): {sidecar.name}")
    return (checkpoint_path if not errors else None), errors


def _run_completion_errors(
    run: Mapping[str, Any],
    root: Path,
    mapping: Mapping[str, Path],
    *,
    campaign: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        marker = _read_json(mapping["run_completion_marker"])
        if marker.get("complete") is not True or marker.get("failed") or marker.get("partial"):
            errors.append("run completion marker is failed, partial, or not complete")
        final_step = marker.get("final_optimizer_step", marker.get("optimizer_step"))
        expected_final = list(run.get("expected_checkpoint_steps") or [])[-1:]
        if not expected_final or final_step != expected_final[-1]:
            errors.append("final optimizer step was not reached")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid run completion marker: {exc}")
    for artifact, expected_key, expected_steps in (
        ("checkpoint_series", "checkpoint_steps", run.get("expected_checkpoint_steps") or []),
        ("evaluation_reports", "evaluation_steps", run.get("expected_evaluation_steps") or []),
    ):
        try:
            payload = _read_json(mapping[artifact])
            actual = payload.get(expected_key, payload.get("steps"))
            if list(actual or []) != list(expected_steps):
                errors.append(f"{artifact} does not exactly match the scheduled steps")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {artifact}: {exc}")
    try:
        evaluation_report = _read_json(mapping["evaluation_reports"])
        policy = dict(run.get("resolved_config") or {}).get("evaluation", {}).get("ema_policy")
        evaluated = set(evaluation_report.get("evaluated_weights") or [])
        required = {"ema", "live"} if policy == "both" else {str(policy)}
        if not required.issubset(evaluated):
            errors.append("evaluation reports do not fulfill the EMA/live evaluation policy")
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        pass
    errors.extend(_validate_required_artifact_manifest(campaign, run, root, mapping))
    return errors


def _validate_required_artifact_manifest(
    campaign: Mapping[str, Any], run: Mapping[str, Any], root: Path, mapping: Mapping[str, Path]
) -> list[str]:
    errors: list[str] = []
    try:
        manifest = _read_json(mapping["artifact_manifest"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"invalid artifact manifest: {exc}"]
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        errors.append("artifact manifest schema is missing or unsupported")
    for field, expected in (
        ("campaign_identity", campaign.get("campaign_identity")),
        ("run_identity", run.get("run_identity")),
        ("seed", run.get("seed")),
    ):
        if manifest.get(field) != expected:
            errors.append(f"artifact manifest {field} does not match")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        return [*errors, "artifact manifest entries are missing or empty"]

    allowed_types = set(PER_RUN_ARTIFACTS) - {"artifact_manifest"}
    allowed_types.add("checkpoint")
    seen_paths: set[str] = set()
    entries_by_path: dict[str, Mapping[str, Any]] = {}
    checkpoint_steps_found: list[int] = []
    checkpoint_paths_found: set[str] = set()
    root_resolved = root.resolve()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            errors.append(f"artifact manifest entry {index} is not an object")
            continue
        entry = dict(raw_entry)
        artifact_type = entry.get("artifact_type")
        relative = entry.get("relative_path")
        expected_hash = entry.get("content_sha256")
        if artifact_type not in allowed_types:
            errors.append(f"artifact manifest entry {index} has unexpected type: {artifact_type!r}")
        if not isinstance(relative, str) or not _safe_relative_artifact_path(relative):
            errors.append(f"artifact manifest entry {index} has unsafe relative path: {relative!r}")
            continue
        normalized = PurePosixPath(relative).as_posix()
        path_key = normalized.casefold()
        if path_key in seen_paths:
            errors.append(f"artifact manifest has duplicate relative path: {normalized}")
            continue
        seen_paths.add(path_key)
        entries_by_path[normalized] = entry
        path = (root / Path(*PurePosixPath(normalized).parts)).resolve()
        if path == root_resolved or root_resolved not in path.parents:
            errors.append(f"artifact path escapes output root: {normalized}")
            continue
        if not is_concrete_hash(expected_hash):
            errors.append(f"artifact is missing a concrete content hash: {normalized}")
        elif not path.is_file():
            errors.append(f"required artifact is missing: {normalized}")
        elif file_sha256(path) != expected_hash:
            errors.append(f"artifact content hash mismatch: {normalized}")
        if entry.get("producing_run_identity") != run.get("run_identity"):
            errors.append(f"artifact has foreign producing run identity: {normalized}")
        if type(entry.get("seed")) is not int or entry.get("seed") != run.get("seed"):
            errors.append(f"artifact has wrong seed: {normalized}")
        scheduled_step = entry.get("scheduled_step")
        final_role = entry.get("final_role")
        if (scheduled_step is None) == (not isinstance(final_role, str) or not final_role.strip()):
            errors.append(f"artifact must declare exactly one scheduled_step or final_role: {normalized}")
        if artifact_type == "checkpoint":
            if type(scheduled_step) is not int or scheduled_step not in set(run.get("expected_checkpoint_steps") or []):
                errors.append(f"checkpoint artifact has wrong scheduled step: {normalized}")
            else:
                checkpoint_steps_found.append(scheduled_step)
                checkpoint_paths_found.add(normalized)
                filename_step = _checkpoint_step_from_name(PurePosixPath(normalized).name)
                if filename_step != scheduled_step:
                    errors.append(f"checkpoint artifact path does not match its scheduled step: {normalized}")
        elif scheduled_step is not None:
            errors.append(f"non-checkpoint artifact has unexpected scheduled step: {normalized}")
        if artifact_type in _METRIC_ARTIFACT_TYPES:
            metric_identity = entry.get("metric_definition_identity")
            if not is_concrete_hash(metric_identity):
                errors.append(f"metric artifact has no definition identity: {normalized}")
            elif path.is_file() and path.suffix.lower() == ".json":
                try:
                    payload = _read_json(path)
                    if metric_identity != _metric_definition_identity(payload):
                        errors.append(f"metric definition identity mismatch: {normalized}")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid metric artifact {normalized}: {exc}")

    required_json_paths = {f"{name}.json" for name in PER_RUN_ARTIFACTS if name != "artifact_manifest"}
    for relative in sorted(required_json_paths - set(entries_by_path)):
        errors.append(f"required artifact has no manifest entry or hash: {relative}")
    for relative in sorted(required_json_paths & set(entries_by_path)):
        if entries_by_path[relative].get("artifact_type") != PurePosixPath(relative).stem:
            errors.append(f"required artifact type does not match its path: {relative}")
    unexpected_json = {
        relative
        for relative, entry in entries_by_path.items()
        if entry.get("artifact_type") != "checkpoint" and relative not in required_json_paths
    }
    errors.extend(f"unexpected required artifact entry: {relative}" for relative in sorted(unexpected_json))
    expected_steps = list(run.get("expected_checkpoint_steps") or [])
    if sorted(checkpoint_steps_found) != sorted(expected_steps):
        errors.append("artifact manifest checkpoint set does not exactly match the schedule")

    on_disk_checkpoint_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("checkpoint*")
        if path.is_file() and path.suffix.lower() in {".bin", ".pt", ".pth", ".ckpt"}
    }
    if on_disk_checkpoint_paths != checkpoint_paths_found:
        errors.append("on-disk checkpoint set contains a missing, unlisted, duplicate, or off-schedule checkpoint")

    for name, path in mapping.items():
        if name == "artifact_manifest" or not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid required JSON artifact {path.name}: {exc}")
            continue
        for field, expected in (
            ("campaign_identity", campaign.get("campaign_identity")),
            ("run_identity", run.get("run_identity")),
            ("seed", run.get("seed")),
        ):
            if payload.get(field) != expected:
                errors.append(f"required artifact {path.name} has missing or foreign {field}")
    return errors


def _safe_relative_artifact_path(value: str) -> bool:
    if not value or "\\" in value or re.match(r"^[A-Za-z]:", value):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and all(part not in {"", "."} for part in path.parts)


def _checkpoint_step_from_name(name: str) -> int | None:
    match = re.fullmatch(r"checkpoint_(?:step_)?(\d+)(?:_ema)?\.(?:bin|pt|pth|ckpt)", name)
    return None if match is None else int(match.group(1))


def _metric_definition_identity(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("metric_definitions"), Mapping) and payload["metric_definitions"]:
        return stable_hash(payload["metric_definitions"])
    definition = payload.get("definition")
    if "definition" in payload and definition is not None and definition != "":
        return stable_hash(definition)
    raise ValueError("metric artifact has no concrete metric definition content")


def _foreign_run_roots(campaign: Mapping[str, Any]) -> list[str]:
    runs = list(campaign.get("expected_runs") or [])
    if not runs:
        return []
    expected = {Path(str(run["output_root"])).resolve() for run in runs}
    campaign_roots = {path.parents[1] for path in expected if len(path.parents) >= 2}
    if len(campaign_roots) != 1:
        return ["expected output roots do not share one campaign root"]
    campaign_root = next(iter(campaign_roots))
    if not campaign_root.exists():
        return []
    found = {path.parent.resolve() for path in campaign_root.rglob("run_identity.json")}
    found.update(
        path.resolve()
        for path in campaign_root.glob("*/seed_*")
        if path.is_dir() and any(child.is_file() for child in path.iterdir())
    )
    return sorted(str(path) for path in found - expected)


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
