"""Strict conditioned Dataset-v5 activation and campaign contracts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from spritelab.product_core import ProjectContext
from spritelab.product_features.training.models import TrainingProfile
from spritelab.training.campaign import (
    DEFAULT_SEEDS,
    TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS,
    file_sha256,
    plan_campaign,
    stable_hash,
    training_code_identity_source_paths,
    validate_campaign,
)
from spritelab.utils.safe_fs import UnsafeFilesystemOperation, require_confined_path
from spritelab.v3.config import ProjectConfig
from spritelab.v3.model import AuditStatus

CONDITIONED_TRAINING_CONTRACT_SCHEMA = "spritelab.training.conditioned-dataset-contract.v2"
CONDITIONED_DATASET_FREEZE_SCHEMA = "spritelab.dataset.freeze.conditioned.v5"
CONDITIONED_PUBLICATION_INVENTORY_SCHEMA = "spritelab.dataset.freeze.inventory.v1"
CONDITIONED_CAMPAIGN_BUILD_SCHEMA = "spritelab.training.conditioned-campaign-build.v1"
TRAINING_AUDIT_REPORT_SCHEMA = "spritelab.training.infrastructure-audit.v2"
TRAINING_AUDIT_HASHES_SCHEMA = "spritelab.training.infrastructure-audit-hashes.v2"
MIN_CONDITIONED_IMAGES = 2_000
MAX_CONDITIONED_IMAGES = 3_000
CONDITIONED_CAMPAIGN_STEPS = 5_000

REQUIRED_ACTIVATION_ARTIFACTS = (
    "view_manifest",
    "split_manifest",
    "conditioning_vocabulary",
    "benchmark_manifest",
    "labeling_audit",
    "validation_report",
)

MANDATORY_TRAINING_AUDIT_GATES = (
    "tracked_code_identity_inventory",
    "no_untracked_production_python",
    "dataset_view_freeze_campaign_vocabulary_identity",
    "dataset_and_training_manifest_qa",
    "production_loader_coverage",
    "campaign_experiment_compatibility",
    "cpu_cuda_smoke_evidence",
    "cuda_driver_torch_device_compatibility",
    "determinism_environment_qualification",
    "launch_receipt_execution_contract_binding",
    "backend_command_safety",
    "idempotency_concurrency_refusal",
    "output_root_resume_safety",
    "event_history_migration_identity",
    "publication_config_atomicity_restart",
    "filesystem_containment_link_defenses",
    "api_ui_privacy",
    "curated_full_test_results",
)


class ConditionedActivationError(ValueError):
    """A conditioned activation contract failed closed."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


@dataclass(frozen=True)
class ConditionedCampaignBuild:
    """Validated runtime campaign plus the relative specification safe to persist."""

    portable_campaign: Mapping[str, Any]
    campaign: Mapping[str, Any]
    validation: Mapping[str, Any]
    schema_version: str = CONDITIONED_CAMPAIGN_BUILD_SCHEMA


@dataclass(frozen=True)
class ConditionedTrainingActivation:
    """Verified activation state consumed by status, launch, and resume."""

    config: ProjectConfig
    profile: TrainingProfile
    freeze_path: Path
    freeze_sha256: str
    campaign_config_path: Path
    campaign_config_sha256: str
    manifest: Mapping[str, Any]
    artifacts: Mapping[str, Path]
    selected_spec: Mapping[str, Any]
    campaign: Mapping[str, Any]
    audit_status: AuditStatus
    schema_version: str = CONDITIONED_TRAINING_CONTRACT_SCHEMA

    @property
    def ready(self) -> bool:
        return self.audit_status is AuditStatus.PASS

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ready": self.ready,
            "profile": self.profile.value,
            "image_count": self.manifest["image_count"],
            "freeze_sha256": self.freeze_sha256,
            "campaign_identity_sha256": self.campaign["campaign_identity"],
            "training_code_identity_sha256": dict(self.campaign["code_identity"])["sha256"],
            "audit_status": self.audit_status.value,
            "paths_exposed": False,
        }


def build_conditioned_three_seed_campaign(
    project_root: str | Path,
    *,
    campaign_directory: str,
    activation_manifest: str,
    activation_manifest_sha256: str,
    view_manifest: str,
    split_manifest: str,
    conditioning_vocabulary: str,
    benchmark_manifest: str,
    output_root: str,
    campaign_id: str,
) -> ConditionedCampaignBuild:
    """Build and validate the exact 5,000-step conditioned three-seed campaign.

    Every supplied path is a canonical project-relative path. Returned
    ``portable_campaign`` paths are relative to ``campaign_directory`` and are
    suitable for a profile entry written there. The helper performs no writes.
    """

    root = Path(project_root).resolve()
    directory = _project_relative_path(root, campaign_directory, require_file=False)
    if not directory.is_dir():
        raise ConditionedActivationError(
            "campaign_directory_missing", "The conditioned campaign directory must already exist."
        )
    activation = _project_relative_file(root, activation_manifest)
    if not _is_sha256(activation_manifest_sha256) or file_sha256(activation) != activation_manifest_sha256:
        raise ConditionedActivationError(
            "activation_manifest_identity", "The supplied activation-manifest identity does not match its bytes."
        )
    activation_document = _read_mapping(activation)
    image_count = activation_document.get("image_count")
    if (
        activation_document.get("schema_version") != CONDITIONED_DATASET_FREEZE_SCHEMA
        or isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or not MIN_CONDITIONED_IMAGES <= image_count <= MAX_CONDITIONED_IMAGES
    ):
        raise ConditionedActivationError(
            "activation_manifest_contract", "The activation manifest is not an eligible conditioned Dataset-v5 freeze."
        )
    view = _project_relative_file(root, view_manifest)
    split = _project_relative_file(root, split_manifest)
    vocabulary = _project_relative_file(root, conditioning_vocabulary)
    benchmark = _project_relative_file(root, benchmark_manifest)
    output = _project_relative_path(root, output_root, require_file=False)
    if output == root:
        raise ConditionedActivationError("campaign_output_root", "The campaign output root must be below the project.")
    if not str(campaign_id).strip() or str(campaign_id) != str(campaign_id).strip():
        raise ConditionedActivationError("campaign_id", "A nonempty unpadded campaign ID is required.")

    model = {
        "architecture": "rectified_flow",
        "sprite_size": 32,
        "base_channels": 32,
        "channel_mults": [1, 2],
        "res_blocks_per_level": 1,
        "embed_dim": 32,
        "film_conditioning": True,
        "bottleneck_attention": False,
        "auxiliary_heads_mode": "absent",
    }
    optimizer = {
        "name": "adamw",
        "learning_rate": 0.0002,
        "schedule": "none",
        "warmup_steps": 0,
        "gradient_clip": 0.0,
    }
    schedule = {"name": "none", "warmup_steps": 0}
    loss = {
        "name": "uniform_velocity",
        "strategy": "uniform_velocity",
        "foreground_rgb_weight": 1.0,
        "background_rgb_weight": 1.0,
        "palette_aux_weight": 0.0,
        "auxiliary_heads": False,
        "index_head_weight": 0.0,
        "palette_head_weight": 0.0,
        "palette_presence_weight": 0.0,
    }
    determinism = {"mode": "strict"}
    evaluation = {
        "cadence": 250,
        "include_step_zero": False,
        "benchmark_manifest_hash": file_sha256(benchmark),
        "benchmark_manifest_path": str(benchmark),
        "cfg_value": 3.0,
        "sampling_steps": 30,
        "ema_policy": "both",
        "live_weight_evaluation_policy": "required",
    }
    evaluation["evaluation_config_hash"] = stable_hash(
        {key: value for key, value in evaluation.items() if not key.startswith("benchmark_manifest_")}
    )
    absolute_spec = {
        "campaign_id": str(campaign_id),
        "purpose": "Conditioned Dataset-v5 three-seed production campaign.",
        "architecture_cells": [{"cell_id": "conditioned_v5", "comparison_values": {}}],
        "identities": {
            "dataset_freeze_path": str(activation),
            "dataset_freeze_hash": activation_manifest_sha256,
            "dataset_view_manifest_hash": file_sha256(view),
            "dataset_view_manifest_path": str(view),
            "split_manifest_hash": file_sha256(split),
            "split_manifest_path": str(split),
            "conditioning_vocabulary_hash": file_sha256(vocabulary),
            "conditioning_vocabulary_path": str(vocabulary),
            "model_config_hash": stable_hash(model),
            "optimizer_config_hash": stable_hash(optimizer),
            "schedule_config_hash": stable_hash(schedule),
            "loss_config_hash": stable_hash(loss),
            "determinism_config_hash": stable_hash(determinism),
        },
        "seeds": list(DEFAULT_SEEDS),
        "model": model,
        "training": {
            "device": "auto",
            "max_optimizer_steps": CONDITIONED_CAMPAIGN_STEPS,
            "micro_batch_size": 4,
            "gradient_accumulation": 1,
            "effective_batch_size": 4,
            "precision": "fp32",
            "sampler_policy": "weighted_replacement_v1",
            "positive_sampling_mass_records": float(image_count),
        },
        "optimizer": optimizer,
        "schedule": schedule,
        "loss": loss,
        "determinism": determinism,
        "evaluation": evaluation,
        "checkpoint": {"cadence": 1_000},
        "output_root": str(output),
        "executable": True,
        "launch_authorized": True,
    }
    campaign = plan_campaign(absolute_spec, execution_root=root)
    validation = validate_campaign(campaign)
    if validation["errors"] or validation["blockers"] or not validation["launch_ready"]:
        raise ConditionedActivationError(
            "conditioned_campaign_invalid", "The conditioned campaign did not pass authoritative validation."
        )

    portable = deepcopy(absolute_spec)
    identities = portable["identities"]
    for field, path in (
        ("dataset_freeze_path", activation),
        ("dataset_view_manifest_path", view),
        ("split_manifest_path", split),
        ("conditioning_vocabulary_path", vocabulary),
    ):
        identities[field] = _portable_path(directory, path)
    portable["evaluation"]["benchmark_manifest_path"] = _portable_path(directory, benchmark)
    portable["output_root"] = _portable_path(directory, output)
    return ConditionedCampaignBuild(portable, campaign, validation)


def load_conditioned_training_activation(
    context: ProjectContext | ProjectConfig,
    profile: TrainingProfile | str = TrainingProfile.RECOMMENDED,
    *,
    custom_spec: Mapping[str, Any] | None = None,
    expected_campaign: Mapping[str, Any] | None = None,
    require_audit: bool = True,
) -> ConditionedTrainingActivation:
    """Load the exact selected activation, refusing every ambiguous or external binding."""

    config = _config(context)
    selected_profile = profile if isinstance(profile, TrainingProfile) else TrainingProfile(str(profile))
    freeze_path = _same_configured_freeze(config)
    campaign_config_path = _configured_project_file(config, "training", "campaign_config")
    manifest = _read_mapping(freeze_path)
    artifacts = _verify_activation_publication(config.root, freeze_path, manifest)
    selected, source_directory = _select_campaign(
        campaign_config_path,
        selected_profile,
        custom_spec=custom_spec,
    )
    resolved = _resolve_campaign_paths(selected, source_directory, config.root)
    campaign = (
        dict(resolved)
        if resolved.get("schema_version") == "spritelab_training_campaign_v3"
        else plan_campaign(resolved, execution_root=config.root)
    )
    _verify_conditioned_campaign(
        campaign,
        freeze_path=freeze_path,
        freeze_sha256=file_sha256(freeze_path),
        artifacts=artifacts,
    )
    if expected_campaign is not None and expected_campaign.get("campaign_identity") != campaign.get(
        "campaign_identity"
    ):
        raise ConditionedActivationError(
            "selected_campaign_changed", "The exact selected campaign differs from the campaign being authorized."
        )
    activation = ConditionedTrainingActivation(
        config=config,
        profile=selected_profile,
        freeze_path=freeze_path,
        freeze_sha256=file_sha256(freeze_path),
        campaign_config_path=campaign_config_path,
        campaign_config_sha256=file_sha256(campaign_config_path),
        manifest=manifest,
        artifacts=artifacts,
        selected_spec=selected,
        campaign=campaign,
        audit_status=AuditStatus.NOT_AUDITED,
    )
    audit = training_audit_status(config, _read_mapping_optional(config, "training", "audit_report"), activation)
    activation = replace(activation, audit_status=audit)
    if require_audit and audit is not AuditStatus.PASS:
        raise ConditionedActivationError(
            "training_audit_applicability",
            f"The exact conditioned activation has no applicable PASS training audit ({audit.value}).",
        )
    return activation


def training_audit_status(
    config: ProjectConfig,
    report: Mapping[str, Any] | None,
    activation: ConditionedTrainingActivation,
) -> AuditStatus:
    """Verify the mandatory independent audit gates and exact activation bindings."""

    if report is None:
        return AuditStatus.NOT_AUDITED
    try:
        report_path = _configured_project_file(config, "training", "audit_report")
        hashes_path = _configured_project_file(config, "training", "audit_hashes")
        stored_report = _read_mapping(report_path)
        hashes = _read_mapping(hashes_path)
    except ConditionedActivationError:
        return AuditStatus.STALE
    if stored_report != dict(report):
        return AuditStatus.STALE
    if report.get("schema_version") != TRAINING_AUDIT_REPORT_SCHEMA:
        return AuditStatus.STALE
    if hashes.get("schema_version") != TRAINING_AUDIT_HASHES_SCHEMA:
        return AuditStatus.STALE
    if hashes.get("audit_report_sha256") != file_sha256(report_path):
        return AuditStatus.STALE
    bindings = {
        "activation_manifest_sha256": activation.freeze_sha256,
        "campaign_config_sha256": activation.campaign_config_sha256,
        "campaign_identity_sha256": activation.campaign.get("campaign_identity"),
        "training_code_identity_sha256": dict(activation.campaign.get("code_identity") or {}).get("sha256"),
    }
    if report.get("bindings") != bindings or hashes.get("bindings") != bindings:
        return AuditStatus.STALE
    if not _verify_audited_code_inventory(config.root, hashes.get("files")):
        return AuditStatus.STALE
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(MANDATORY_TRAINING_AUDIT_GATES):
        return AuditStatus.INCONCLUSIVE
    verdicts = {str(value).upper() for value in gates.values()}
    if "FAIL" in verdicts:
        return AuditStatus.FAIL
    return AuditStatus.PASS if verdicts == {"PASS"} else AuditStatus.INCONCLUSIVE


def _config(context: ProjectContext | ProjectConfig) -> ProjectConfig:
    if isinstance(context, ProjectConfig):
        return context
    if context.config_path is not None and context.config_path.is_file():
        return ProjectConfig.load(context.project_root)
    if context.config:
        return ProjectConfig(context.project_root.resolve(), context.config_path, deepcopy(dict(context.config)))
    return ProjectConfig.load(context.project_root)


def _same_configured_freeze(config: ProjectConfig) -> Path:
    dataset_raw = config.values.get("dataset", {}).get("freeze_manifest")
    training_raw = config.values.get("training", {}).get("dataset_freeze")
    if not isinstance(dataset_raw, str) or not isinstance(training_raw, str) or dataset_raw != training_raw:
        raise ConditionedActivationError(
            "freeze_configuration_mismatch",
            "dataset.freeze_manifest and training.dataset_freeze must name the same project-relative file.",
        )
    return _project_relative_file(config.root, dataset_raw)


def _configured_project_file(config: ProjectConfig, section: str, key: str) -> Path:
    values = config.values.get(section)
    raw = values.get(key) if isinstance(values, Mapping) else None
    if not isinstance(raw, str):
        raise ConditionedActivationError("activation_path_invalid", "A required activation path is invalid.")
    return _project_relative_file(config.root, raw)


def _project_relative_file(root: Path, raw: str) -> Path:
    path = _project_relative_path(root, raw, require_file=True)
    if path.stat().st_nlink != 1:
        raise ConditionedActivationError("activation_hardlink", "Activation files must not be hard-linked.")
    return path


def _project_relative_path(root: Path, raw: str, *, require_file: bool) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\x00" in raw:
        raise ConditionedActivationError(
            "activation_path_invalid", "Activation paths must be canonical relative paths."
        )
    pure = PurePosixPath(raw)
    if pure.is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ConditionedActivationError("activation_path_absolute", "Absolute activation paths are forbidden.")
    if "\\" in raw or any(part in {"", ".", ".."} for part in pure.parts) or pure.as_posix() != raw:
        raise ConditionedActivationError(
            "activation_path_invalid", "Activation paths must be canonical relative paths."
        )
    try:
        path = require_confined_path(root.joinpath(*pure.parts), root)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedActivationError(
            "activation_path_external", "Activation paths must remain inside the project."
        ) from exc
    if require_file and not path.is_file():
        raise ConditionedActivationError("activation_file_missing", "A required activation file is missing.")
    return path


def _publication_file(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\\" in raw or "\x00" in raw:
        raise ConditionedActivationError(
            "publication_path_invalid", "Publication paths must be canonical relative paths."
        )
    pure = PurePosixPath(raw)
    if pure.is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ConditionedActivationError("publication_path_absolute", "Absolute publication paths are forbidden.")
    if any(part in {"", ".", ".."} for part in pure.parts) or pure.as_posix() != raw:
        raise ConditionedActivationError(
            "publication_path_invalid", "Publication paths must be canonical relative paths."
        )
    try:
        path = require_confined_path(root.joinpath(*pure.parts), root)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedActivationError(
            "publication_path_external", "Publication paths must remain confined."
        ) from exc
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ConditionedActivationError(
            "publication_file_invalid", "Publication artifacts must be regular owned files."
        )
    return path


def _verify_activation_publication(
    project_root: Path,
    freeze_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    if manifest.get("schema_version") != CONDITIONED_DATASET_FREEZE_SCHEMA:
        raise ConditionedActivationError(
            "conditioned_dataset_v5_schema", "The Dataset-v5 activation schema is invalid."
        )
    if (
        manifest.get("dataset_version") != 5
        or manifest.get("dataset_kind") != "conditioned"
        or manifest.get("requires_semantic_labels") is not True
        or manifest.get("status") != "complete"
        or manifest.get("production_authorized") is not True
    ):
        raise ConditionedActivationError("conditioned_dataset_v5_state", "The Dataset-v5 activation is incomplete.")
    count = manifest.get("image_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not MIN_CONDITIONED_IMAGES <= count <= MAX_CONDITIONED_IMAGES
    ):
        raise ConditionedActivationError(
            "conditioned_dataset_v5_size", "The conditioned Dataset-v5 must contain between 2,000 and 3,000 images."
        )
    publication_root = freeze_path.parent
    require_confined_path(publication_root, project_root)
    declared_artifacts = manifest.get("artifacts")
    if not isinstance(declared_artifacts, Mapping) or set(declared_artifacts) != set(REQUIRED_ACTIVATION_ARTIFACTS):
        raise ConditionedActivationError("activation_artifacts", "The activation artifact binding is incomplete.")
    artifacts: dict[str, Path] = {}
    for name in REQUIRED_ACTIVATION_ARTIFACTS:
        binding = declared_artifacts[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256", "byte_count"}
            or not _is_sha256(binding.get("sha256"))
            or isinstance(binding.get("byte_count"), bool)
            or not isinstance(binding.get("byte_count"), int)
            or binding["byte_count"] < 0
        ):
            raise ConditionedActivationError(
                "activation_artifact_binding", "An activation artifact binding is invalid."
            )
        path = _publication_file(publication_root, binding.get("path"))
        if file_sha256(path) != binding["sha256"] or path.stat().st_size != binding["byte_count"]:
            raise ConditionedActivationError("activation_artifact_hash", "An activation artifact hash is stale.")
        artifacts[name] = path
    inventory = manifest.get("publication_inventory")
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != {"schema_version", "files", "file_count", "total_bytes", "inventory_sha256"}
        or inventory.get("schema_version") != CONDITIONED_PUBLICATION_INVENTORY_SCHEMA
    ):
        raise ConditionedActivationError("activation_inventory", "The publication inventory schema is invalid.")
    files = inventory.get("files")
    if not isinstance(files, Mapping):
        raise ConditionedActivationError("activation_inventory", "The publication inventory is invalid.")
    actual = _publication_inventory(publication_root, exclude=freeze_path)
    declared_hashes: dict[str, str] = {}
    declared_sizes: dict[str, int] = {}
    for relative, binding in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(binding, Mapping)
            or set(binding) != {"sha256", "byte_count"}
            or not _is_sha256(binding.get("sha256"))
            or isinstance(binding.get("byte_count"), bool)
            or not isinstance(binding.get("byte_count"), int)
            or binding["byte_count"] < 0
        ):
            raise ConditionedActivationError("activation_inventory", "A publication inventory record is invalid.")
        declared_hashes[relative] = str(binding["sha256"])
        declared_sizes[relative] = int(binding["byte_count"])
    actual_hashes = {relative: str(value["sha256"]) for relative, value in actual.items()}
    if declared_hashes != actual_hashes:
        raise ConditionedActivationError("activation_inventory_stale", "The full publication inventory is stale.")
    if any(actual[relative]["byte_size"] != size for relative, size in declared_sizes.items()):
        raise ConditionedActivationError(
            "activation_inventory_stale", "A declared publication inventory byte count is stale."
        )
    declared_total = inventory.get("total_bytes")
    declared_count = inventory.get("file_count")
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total < 0
        or declared_total != sum(int(value["byte_size"]) for value in actual.values())
        or isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(actual)
    ):
        raise ConditionedActivationError(
            "activation_inventory_stale", "The declared publication count or byte total is stale."
        )
    inventory_payload = {
        "schema_version": CONDITIONED_PUBLICATION_INVENTORY_SCHEMA,
        "files": dict(files),
        "file_count": declared_count,
        "total_bytes": declared_total,
    }
    if not _is_sha256(inventory.get("inventory_sha256")) or inventory["inventory_sha256"] != stable_hash(
        inventory_payload
    ):
        raise ConditionedActivationError(
            "activation_inventory_identity", "The publication inventory identity is invalid."
        )
    for name, path in artifacts.items():
        relative = path.relative_to(publication_root).as_posix()
        if actual_hashes.get(relative) != declared_artifacts[name]["sha256"]:
            raise ConditionedActivationError(
                "activation_inventory_binding", "Artifact and inventory bindings disagree."
            )
    return artifacts


def _publication_inventory(root: Path, *, exclude: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            try:
                child = require_confined_path(directory_path / name, root)
            except UnsafeFilesystemOperation as exc:
                raise ConditionedActivationError(
                    "publication_entry_invalid", "The publication contains an unsafe linked directory."
                ) from exc
            if not child.is_dir():
                raise ConditionedActivationError(
                    "publication_entry_invalid", "The publication contains an unsafe entry."
                )
        for name in sorted(file_names):
            try:
                child = require_confined_path(directory_path / name, root)
            except UnsafeFilesystemOperation as exc:
                raise ConditionedActivationError(
                    "publication_entry_invalid", "The publication contains an unsafe linked file."
                ) from exc
            if child == exclude:
                continue
            if not child.is_file() or child.stat().st_nlink != 1:
                raise ConditionedActivationError(
                    "publication_entry_invalid", "The publication contains an unsafe entry."
                )
            files[child.relative_to(root).as_posix()] = {
                "sha256": file_sha256(child),
                "byte_size": child.stat().st_size,
            }
    return dict(sorted(files.items()))


def _select_campaign(
    campaign_path: Path,
    profile: TrainingProfile,
    *,
    custom_spec: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Path]:
    document = _read_mapping(campaign_path)
    selected: Mapping[str, Any] = document
    source_directory = campaign_path.parent
    profiles = document.get("product_profiles")
    if isinstance(profiles, Mapping):
        entry = profiles.get(profile.value)
        if not isinstance(entry, Mapping):
            raise ConditionedActivationError(
                "campaign_profile", "The exact selected campaign profile is not configured."
            )
        if isinstance(entry.get("campaign"), Mapping):
            selected = entry["campaign"]
        elif entry.get("campaign_path"):
            nested = _campaign_relative_file(campaign_path.parent, entry["campaign_path"], campaign_path.parent)
            selected = _read_mapping(nested)
            source_directory = nested.parent
        else:
            raise ConditionedActivationError("campaign_profile", "The exact selected campaign profile is incomplete.")
    elif profile is not TrainingProfile.RECOMMENDED:
        raise ConditionedActivationError("campaign_profile", "Only the recommended direct campaign is supported.")
    chosen = dict(selected)
    if profile is TrainingProfile.CUSTOM:
        if custom_spec is not None and (
            not isinstance(custom_spec, Mapping) or stable_hash(dict(custom_spec)) != stable_hash(chosen)
        ):
            raise ConditionedActivationError(
                "custom_campaign_mismatch", "The custom campaign must exactly match the configured custom profile."
            )
    elif custom_spec is not None:
        raise ConditionedActivationError("custom_campaign_unexpected", "Custom settings require the custom profile.")
    return chosen, source_directory


def _resolve_campaign_paths(spec: Mapping[str, Any], source_directory: Path, project_root: Path) -> dict[str, Any]:
    result = deepcopy(dict(spec))
    identities = result.get("identities")
    if not isinstance(identities, dict):
        raise ConditionedActivationError("campaign_identity", "Campaign identities are missing.")
    for field in (
        "dataset_freeze_path",
        "dataset_view_manifest_path",
        "split_manifest_path",
        "conditioning_vocabulary_path",
    ):
        identities[field] = str(_campaign_relative_file(source_directory, identities.get(field), project_root))
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ConditionedActivationError("campaign_evaluation", "Campaign evaluation settings are missing.")
    evaluation["benchmark_manifest_path"] = str(
        _campaign_relative_file(source_directory, evaluation.get("benchmark_manifest_path"), project_root)
    )
    result["output_root"] = str(
        _campaign_relative_path(source_directory, result.get("output_root"), project_root, require_file=False)
    )
    return result


def _campaign_relative_file(source_directory: Path, raw: Any, project_root: Path) -> Path:
    path = _campaign_relative_path(source_directory, raw, project_root, require_file=True)
    if path.stat().st_nlink != 1:
        raise ConditionedActivationError("campaign_path_hardlink", "Campaign inputs must not be hard-linked.")
    return path


def _campaign_relative_path(
    source_directory: Path,
    raw: Any,
    project_root: Path,
    *,
    require_file: bool,
) -> Path:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\\" in raw or "\x00" in raw:
        raise ConditionedActivationError("campaign_path_invalid", "Campaign paths must be relative and canonical.")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ConditionedActivationError("campaign_path_absolute", "Absolute campaign paths are forbidden.")
    try:
        path = require_confined_path(source_directory.joinpath(*pure.parts), project_root)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedActivationError(
            "campaign_path_external", "Campaign paths must remain inside the project."
        ) from exc
    if require_file and not path.is_file():
        raise ConditionedActivationError("campaign_file_missing", "A bound campaign input is missing.")
    return path


def _verify_conditioned_campaign(
    campaign: Mapping[str, Any],
    *,
    freeze_path: Path,
    freeze_sha256: str,
    artifacts: Mapping[str, Path],
) -> None:
    validation = validate_campaign(campaign)
    if validation["errors"] or validation["blockers"] or not validation["launch_ready"]:
        raise ConditionedActivationError("conditioned_campaign_invalid", "The selected campaign is not launch-ready.")
    if campaign.get("executable") is not True or campaign.get("launch_authorized") is not True:
        raise ConditionedActivationError("conditioned_campaign_authorization", "The campaign is not authorized.")
    if tuple(campaign.get("seeds") or ()) != tuple(DEFAULT_SEEDS) or len(set(campaign.get("seeds") or ())) != 3:
        raise ConditionedActivationError("conditioned_campaign_seeds", "The campaign must bind exactly three seeds.")
    training = campaign.get("training")
    if not isinstance(training, Mapping) or training.get("max_optimizer_steps") != CONDITIONED_CAMPAIGN_STEPS:
        raise ConditionedActivationError("conditioned_campaign_steps", "The campaign must bind exactly 5,000 steps.")
    identities = campaign.get("identities")
    if not isinstance(identities, Mapping):
        raise ConditionedActivationError("conditioned_campaign_identity", "Campaign identities are missing.")
    expected = {
        "dataset_freeze_path": freeze_path,
        "dataset_freeze_hash": freeze_sha256,
        "dataset_view_manifest_path": artifacts["view_manifest"],
        "dataset_view_manifest_hash": file_sha256(artifacts["view_manifest"]),
        "split_manifest_path": artifacts["split_manifest"],
        "split_manifest_hash": file_sha256(artifacts["split_manifest"]),
        "conditioning_vocabulary_path": artifacts["conditioning_vocabulary"],
        "conditioning_vocabulary_hash": file_sha256(artifacts["conditioning_vocabulary"]),
    }
    for field, value in expected.items():
        actual = Path(str(identities.get(field))).resolve() if field.endswith("_path") else identities.get(field)
        expected_value = value.resolve() if isinstance(value, Path) else value
        if actual != expected_value:
            raise ConditionedActivationError(
                "conditioned_campaign_binding", "Campaign dataset, view, split, or vocabulary bindings disagree."
            )
    evaluation = campaign.get("evaluation")
    benchmark = artifacts["benchmark_manifest"]
    if (
        not isinstance(evaluation, Mapping)
        or Path(str(evaluation.get("benchmark_manifest_path"))).resolve() != benchmark.resolve()
        or evaluation.get("benchmark_manifest_hash") != file_sha256(benchmark)
    ):
        raise ConditionedActivationError("conditioned_campaign_benchmark", "Campaign benchmark bindings disagree.")


def _verify_audited_code_inventory(root: Path, value: Any) -> bool:
    if not isinstance(value, list):
        return False
    try:
        required = {
            path.relative_to(root.resolve()).as_posix(): path for path in training_code_identity_source_paths(root)
        }
        production_python: set[str] = set()
        for relative_root in TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS:
            audit_root = _project_relative_path(root, relative_root, require_file=False)
            if not audit_root.is_dir():
                return False
            for directory, directory_names, file_names in os.walk(audit_root, followlinks=False):
                directory_path = Path(directory)
                for name in directory_names:
                    child = require_confined_path(directory_path / name, audit_root)
                    if not child.is_dir():
                        return False
                for name in file_names:
                    if not name.endswith(".py"):
                        continue
                    confined = require_confined_path(directory_path / name, audit_root)
                    if not confined.is_file() or confined.stat().st_nlink != 1:
                        return False
                    production_python.add(confined.relative_to(root.resolve()).as_posix())
        bound_root_prefixes = tuple(f"{relative.rstrip('/')}/" for relative in TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS)
        required_under_roots = {relative for relative in required if relative.startswith(bound_root_prefixes)}
        if production_python != required_under_roots:
            return False
    except (OSError, ValueError, UnsafeFilesystemOperation):
        return False
    observed: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256_before"}:
            return False
        relative, digest = item.get("path"), item.get("sha256_before")
        if not isinstance(relative, str) or relative in observed or not _is_sha256(digest):
            return False
        try:
            target = _project_relative_file(root, relative)
        except ConditionedActivationError:
            return False
        if file_sha256(target) != digest:
            return False
        observed[relative] = str(digest)
    return set(observed) == set(required)


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if path.suffix.casefold() in {".yaml", ".yml"} else json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConditionedActivationError("activation_mapping", "An activation mapping is unreadable.") from exc
    if not isinstance(value, dict):
        raise ConditionedActivationError("activation_mapping", "An activation mapping must be an object.")
    return value


def _read_mapping_optional(config: ProjectConfig, section: str, key: str) -> dict[str, Any] | None:
    try:
        path = _configured_project_file(config, section, key)
        return _read_mapping(path)
    except ConditionedActivationError:
        return None


def _portable_path(directory: Path, target: Path) -> str:
    return os.path.relpath(target, start=directory).replace("\\", "/")


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.casefold())


__all__ = [
    "CONDITIONED_CAMPAIGN_BUILD_SCHEMA",
    "CONDITIONED_DATASET_FREEZE_SCHEMA",
    "CONDITIONED_PUBLICATION_INVENTORY_SCHEMA",
    "CONDITIONED_TRAINING_CONTRACT_SCHEMA",
    "MANDATORY_TRAINING_AUDIT_GATES",
    "TRAINING_AUDIT_HASHES_SCHEMA",
    "TRAINING_AUDIT_REPORT_SCHEMA",
    "ConditionedActivationError",
    "ConditionedCampaignBuild",
    "ConditionedTrainingActivation",
    "build_conditioned_three_seed_campaign",
    "load_conditioned_training_activation",
    "training_audit_status",
]
