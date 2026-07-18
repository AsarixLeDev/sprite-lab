"""Strict conditioned Dataset-v5 activation and campaign contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from spritelab.product_core import ProjectContext
from spritelab.product_features.conditioned_v5.audit_receipts import (
    ConditionedAuditReceiptError,
    validate_audit_action_record,
    validate_audit_receipt,
)
from spritelab.product_features.conditioned_v5.identity import (
    TRUSTED_AUDITOR_IDS,
    trusted_auditor_inventory,
)
from spritelab.product_features.conditioned_v5.publication_commit import (
    PUBLICATION_JOURNAL_NAME,
    PublicationCommitError,
    campaign_commit_name,
    canonical_publication_commit_bytes,
    dataset_commit_name,
    validate_campaign_commit,
    validate_dataset_commit,
    validate_publication_journal,
)
from spritelab.product_features.training.activation_commit import (
    ACTIVATION_PROJECT_COMMIT_NAME,
    ActivationCommitError,
    validate_activation_project_commit,
)
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
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    UnsafeFilesystemOperation,
    open_anchored_directory,
    require_confined_path,
)
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
CONDITIONED_CANDIDATE_SCHEMA = "spritelab.dataset.conditioned-candidate.v1"
CONDITIONED_LABEL_AUDIT_SCHEMA = "spritelab.audit.conditioned-labels.v1"
CONDITIONED_VALIDATION_SCHEMA = "spritelab.audit.conditioned-dataset.v1"
CONDITIONED_AUDIT_SUBJECTS_SCHEMA = "spritelab.audit.conditioned-subjects.v1"
CONDITIONED_VIEW_SCHEMA = "spritelab.dataset.conditioned-view.v1"
CONDITIONED_PROVENANCE_SCHEMA = "spritelab.dataset.conditioned-provenance.v1"
CONDITIONED_RECIPE = "conditioned_filename_taxonomy_v1+local_pixel_vision_v1+near_duplicate_v2"
CONDITIONED_NEAR_DUPLICATE_ALGORITHM = "conditioned_near_duplicate_v2"
MAX_EMBEDDED_EVIDENCE_BYTES = 64 * 1024 * 1024

CONDITIONED_LABEL_AUDIT_GATES = frozenset(
    {
        "deterministic_source_grounding",
        "no_human_truth_claim",
        "taxonomy_contract",
        "semantic_coverage",
        "provenance_and_license_binding",
        "duplicate_family_split_integrity",
        "local_pixel_descriptor_recomputation",
    }
)
CONDITIONED_VALIDATION_GATES = frozenset(
    {
        "phase7_arrays",
        "exact_32x32",
        "manifest_npz_parity",
        "training_loader_all_splits",
        "vocabulary_and_benchmark",
        "count_range",
        "publication_filesystem_safety",
        "portable_paths",
        "provenance_hashes",
        "near_duplicate_retained_pair_recomputation",
    }
)

_ACTIVATION_KEYS = frozenset(
    {
        "schema_version",
        "dataset_version",
        "dataset_kind",
        "requires_semantic_labels",
        "status",
        "production_authorized",
        "immutable",
        "image_count",
        "dataset_identity",
        "publication_identity_sha256",
        "labeling_audit_sha256",
        "validation_report_sha256",
        "artifacts",
        "publication_inventory",
        "licenses",
        "paths_are_relative",
        "paths_exposed",
    }
)

_FIXED_ACTIVATION_ARTIFACT_PATHS = {
    "view_manifest": "view_manifest.json",
    "split_manifest": "training_manifest.jsonl",
    "conditioning_vocabulary": "conditioning_vocabulary.json",
    "benchmark_manifest": "benchmark_manifest.json",
    "labeling_audit": "evidence/label_audit.json",
    "labeling_audit_receipt": "evidence/label_audit_receipt.json",
    "labeling_audit_action_record": "evidence/label_audit_action.json",
    "validation_report": "evidence/dataset_validation.json",
    "validation_receipt": "evidence/dataset_validation_receipt.json",
    "validation_action_record": "evidence/dataset_validation_action.json",
}

CONDITIONED_PRODUCTION_PROFILES = frozenset(
    {
        TrainingProfile.RECOMMENDED,
        TrainingProfile.QUALITY,
        TrainingProfile.CUSTOM,
    }
)

REQUIRED_ACTIVATION_ARTIFACTS = (
    "view_manifest",
    "split_manifest",
    "conditioning_vocabulary",
    "benchmark_manifest",
    "labeling_audit",
    "labeling_audit_receipt",
    "labeling_audit_action_record",
    "validation_report",
    "validation_receipt",
    "validation_action_record",
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


class _DuplicateMappingKeyError(ValueError):
    """An untrusted JSON or YAML mapping contains an ambiguous key."""


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate keys, including nested ones."""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
            keys: set[Any] = set()
            for key_node, _value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    if key in keys:
                        raise _DuplicateMappingKeyError
                    keys.add(key)
                except TypeError:
                    # The base safe constructor reports unsupported mapping keys.
                    pass
        return super().construct_mapping(node, deep=deep)


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
    activation_commit: Mapping[str, Any] | None = None
    schema_version: str = CONDITIONED_TRAINING_CONTRACT_SCHEMA

    @property
    def ready(self) -> bool:
        return self.audit_status is AuditStatus.PASS and bool(
            self.activation_commit and self.activation_commit.get("committed") is True
        )

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
            "activation_commit_record_identity": (
                self.activation_commit.get("record_identity") if self.activation_commit else None
            ),
            "paths_exposed": False,
        }


@dataclass(frozen=True)
class _HeldMappingSnapshot:
    """One mapping parsed and hashed from the same descriptor-held bytes."""

    value: dict[str, Any]
    sha256: str
    byte_count: int


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
    suitable for a profile entry written there. The helper performs no writes,
    so a direct-final publisher may plan against an absent final directory.
    """

    root = Path(project_root).resolve()
    directory = _project_relative_path(root, campaign_directory, require_file=False)
    if directory == root or not directory.parent.is_dir() or (os.path.lexists(directory) and not directory.is_dir()):
        raise ConditionedActivationError(
            "campaign_directory_invalid",
            "The conditioned campaign directory must be an absent or existing directory below a safe parent.",
        )
    activation = _project_relative_file(root, activation_manifest, allow_retained_stage=True)
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
    view = _project_relative_file(root, view_manifest, allow_retained_stage=True)
    split = _project_relative_file(root, split_manifest, allow_retained_stage=True)
    vocabulary = _project_relative_file(root, conditioning_vocabulary, allow_retained_stage=True)
    benchmark = _project_relative_file(root, benchmark_manifest, allow_retained_stage=True)
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
    require_activation_commit: bool = True,
) -> ConditionedTrainingActivation:
    """Load the exact selected activation, refusing every ambiguous or external binding."""

    selected_profile = profile if isinstance(profile, TrainingProfile) else TrainingProfile(str(profile))
    if selected_profile not in CONDITIONED_PRODUCTION_PROFILES:
        raise ConditionedActivationError(
            "conditioned_profile_ineligible",
            "The fast_preview profile is not eligible for conditioned production activation.",
        )
    config = _config(context)
    freeze_path = _same_configured_freeze(config)
    campaign_config_path = _configured_project_file(config, "training", "campaign_config")
    with ExitStack() as held_snapshots:
        freeze_snapshot = held_snapshots.enter_context(
            _held_mapping_snapshot(freeze_path, config.root, allow_retained_stage=True)
        )
        campaign_snapshot = held_snapshots.enter_context(
            _held_mapping_snapshot(campaign_config_path, config.root, allow_retained_stage=True)
        )
        manifest = freeze_snapshot.value
        artifacts = _verify_activation_publication(config.root, freeze_path, manifest)
        selected, source_directory = _select_campaign(
            campaign_config_path,
            selected_profile,
            custom_spec=custom_spec,
            document=campaign_snapshot.value,
            held_snapshots=held_snapshots,
            project_root=config.root,
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
            freeze_sha256=freeze_snapshot.sha256,
            artifacts=artifacts,
        )
        publication_job_id = _verify_publication_pair_commit(
            project_root=config.root,
            freeze_path=freeze_path,
            campaign_config_path=campaign_config_path,
            manifest=manifest,
            artifacts=artifacts,
            held_snapshots=held_snapshots,
        )
        if expected_campaign is not None and expected_campaign.get("campaign_identity") != campaign.get(
            "campaign_identity"
        ):
            raise ConditionedActivationError(
                "selected_campaign_changed",
                "The exact selected campaign differs from the campaign being authorized.",
            )
        activation = ConditionedTrainingActivation(
            config=config,
            profile=selected_profile,
            freeze_path=freeze_path,
            freeze_sha256=freeze_snapshot.sha256,
            campaign_config_path=campaign_config_path,
            campaign_config_sha256=campaign_snapshot.sha256,
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
        commit = (
            _load_activation_commit(config, activation, expected_job_id=publication_job_id)
            if require_activation_commit
            else None
        )
        return replace(activation, activation_commit=commit)


def _verify_publication_pair_commit(
    *,
    project_root: Path,
    freeze_path: Path,
    campaign_config_path: Path,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    held_snapshots: ExitStack,
) -> str:
    publication_identity = str(manifest.get("publication_identity_sha256") or "")
    if not _is_lower_sha256(publication_identity):
        raise ConditionedActivationError(
            "publication_commit_invalid",
            "The conditioned publication identity is invalid.",
        )
    action_documents = []
    for artifact_name in ("labeling_audit_action_record", "validation_action_record"):
        snapshot = held_snapshots.enter_context(
            _held_mapping_snapshot(
                artifacts[artifact_name],
                project_root,
                allow_retained_stage=True,
            )
        )
        action_documents.append(snapshot.value)
    job_ids = {str(document.get("job_id") or "") for document in action_documents}
    if len(job_ids) != 1:
        raise ConditionedActivationError(
            "publication_commit_job",
            "The conditioned publication actions do not identify one job.",
        )
    job_id = next(iter(job_ids))
    if re.fullmatch(r"conditioned-[0-9a-f]{20}", job_id) is None:
        raise ConditionedActivationError(
            "publication_commit_job",
            "The conditioned publication job identity is invalid.",
        )
    journal_path = _project_relative_file(
        project_root,
        f"runs/v3/conditioned-dataset-v5/{job_id}/{PUBLICATION_JOURNAL_NAME}",
        allow_retained_stage=True,
    )
    dataset_marker_path = _project_relative_file(
        project_root,
        f"datasets/{dataset_commit_name(publication_identity)}",
        allow_retained_stage=True,
    )
    campaign_marker_path = _project_relative_file(
        project_root,
        f"campaigns/{campaign_commit_name(publication_identity)}",
        allow_retained_stage=True,
    )
    journal = held_snapshots.enter_context(
        _held_mapping_snapshot(journal_path, project_root, allow_retained_stage=True)
    )
    dataset_marker = held_snapshots.enter_context(
        _held_mapping_snapshot(dataset_marker_path, project_root, allow_retained_stage=True)
    )
    campaign_marker = held_snapshots.enter_context(
        _held_mapping_snapshot(campaign_marker_path, project_root, allow_retained_stage=True)
    )
    if journal.value.get("publication_identity_sha256") != publication_identity:
        raise ConditionedActivationError(
            "publication_commit_invalid",
            "The conditioned publication pair marker is invalid or stale.",
        )
    for snapshot, document in (
        (journal, journal.value),
        (dataset_marker, dataset_marker.value),
        (campaign_marker, campaign_marker.value),
    ):
        canonical = canonical_publication_commit_bytes(document)
        if len(canonical) != snapshot.byte_count or hashlib.sha256(canonical).hexdigest() != snapshot.sha256:
            raise ConditionedActivationError(
                "publication_commit_invalid",
                "A conditioned publication commit document is not canonical.",
            )
    dataset_observed = _publication_inventory(
        freeze_path.parent,
        exclude=None,
        boundary=project_root,
    )
    campaign_observed = _publication_inventory(
        campaign_config_path.parent,
        exclude=None,
        boundary=project_root,
    )
    dataset_inventory = {
        relative: {"sha256": value["sha256"], "byte_count": value["byte_size"]}
        for relative, value in dataset_observed.items()
    }
    campaign_inventory = {
        relative: {"sha256": value["sha256"], "byte_count": value["byte_size"]}
        for relative, value in campaign_observed.items()
    }
    try:
        validate_publication_journal(
            journal.value,
            dataset_inventory=dataset_inventory,
            campaign_inventory=campaign_inventory,
        )
        validate_dataset_commit(
            dataset_marker.value,
            journal=journal.value,
            dataset_inventory=dataset_inventory,
            campaign_inventory=campaign_inventory,
        )
        pair = validate_campaign_commit(
            campaign_marker.value,
            journal=journal.value,
            dataset_commit=dataset_marker.value,
            dataset_inventory=dataset_inventory,
            campaign_inventory=campaign_inventory,
        )
    except PublicationCommitError as exc:
        raise ConditionedActivationError(
            "publication_commit_invalid",
            "The conditioned publication pair marker is invalid or stale.",
        ) from exc
    if pair.get("pair_authority") is not True:
        raise ConditionedActivationError(
            "publication_commit_invalid",
            "The conditioned campaign marker does not authorize the exact pair.",
        )
    return job_id


def _load_activation_commit(
    config: ProjectConfig,
    activation: ConditionedTrainingActivation,
    *,
    expected_job_id: str,
) -> dict[str, Any]:
    if re.fullmatch(r"conditioned-[0-9a-f]{20}", expected_job_id) is None:
        raise ConditionedActivationError(
            "activation_commit_job", "Conditioned audit actions do not identify one activation job."
        )
    job_id = expected_job_id
    job_root = _project_relative_path(
        config.root,
        f"runs/v3/conditioned-dataset-v5/{job_id}/activation_receipt",
        require_file=False,
    )
    if not job_root.is_dir():
        raise ConditionedActivationError(
            "activation_commit_missing", "The conditioned activation has no durable commit record."
        )
    receipt_path = _project_relative_file(
        config.root,
        f"{job_root.relative_to(config.root).as_posix()}/receipt.json",
        allow_retained_stage=True,
    )
    journal_path = _project_relative_file(
        config.root,
        f"{job_root.relative_to(config.root).as_posix()}/journal.json",
        allow_retained_stage=True,
    )
    record_path = _project_relative_file(
        config.root,
        f"{job_root.relative_to(config.root).as_posix()}/record.json",
        allow_retained_stage=True,
    )
    config_path = config.path
    if config_path is None or config_path != config.root / "spritelab.yaml" or not config_path.is_file():
        raise ConditionedActivationError(
            "activation_commit_config", "Committed activation requires the canonical project configuration file."
        )
    marker_path = _project_relative_file(
        config.root,
        ACTIVATION_PROJECT_COMMIT_NAME,
        allow_retained_stage=True,
    )
    try:
        with ExitStack() as snapshots:
            receipt = snapshots.enter_context(
                _held_mapping_snapshot(receipt_path, config.root, allow_retained_stage=True)
            )
            journal = snapshots.enter_context(
                _held_mapping_snapshot(journal_path, config.root, allow_retained_stage=True)
            )
            record = snapshots.enter_context(
                _held_mapping_snapshot(record_path, config.root, allow_retained_stage=True)
            )
            marker = snapshots.enter_context(
                _held_mapping_snapshot(marker_path, config.root, allow_retained_stage=True)
            )
            canonical_config = snapshots.enter_context(_held_mapping_snapshot(config_path, config.root))
            commit, _effective_config = validate_activation_project_commit(
                marker.value,
                receipt=receipt.value,
                journal=journal.value,
                record=record.value,
                current_config_sha256=canonical_config.sha256,
                expected_job_id=job_id,
            )
    except (ActivationCommitError, OSError, ValueError) as exc:
        raise ConditionedActivationError(
            "activation_commit_invalid", "The durable activation commit record is invalid or not committed."
        ) from exc
    expected = {
        "candidate_identity": activation.manifest.get("dataset_identity"),
        "publication_identity_sha256": activation.manifest.get("publication_identity_sha256"),
        "activation_manifest_sha256": activation.freeze_sha256,
        "campaign_config_sha256": activation.campaign_config_sha256,
        "campaign_identity_sha256": activation.campaign.get("campaign_identity"),
    }
    if any(commit.get(name) != value for name, value in expected.items()):
        raise ConditionedActivationError(
            "activation_commit_binding", "The durable activation record differs from the selected freeze or campaign."
        )
    return commit


def training_audit_status(
    config: ProjectConfig,
    report: Mapping[str, Any] | None,
    activation: ConditionedTrainingActivation,
) -> AuditStatus:
    """Require and reverify one server-managed immutable audit receipt."""

    # Keep this import lazy: passive activation/status imports must not load
    # smoke execution machinery, and audit.py imports this module's activation
    # contract for its explicit Phase-I action.
    from spritelab.product_features.training.audit import verify_training_audit_execution

    return verify_training_audit_execution(config, report, activation)


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
    return _project_relative_file(config.root, dataset_raw, allow_retained_stage=True)


def _configured_project_file(config: ProjectConfig, section: str, key: str) -> Path:
    values = config.values.get(section)
    raw = values.get(key) if isinstance(values, Mapping) else None
    if not isinstance(raw, str):
        raise ConditionedActivationError("activation_path_invalid", "A required activation path is invalid.")
    return _project_relative_file(config.root, raw, allow_retained_stage=True)


def _project_relative_file(root: Path, raw: str, *, allow_retained_stage: bool = False) -> Path:
    path = _project_relative_path(root, raw, require_file=True)
    _require_path_link_contract(
        path,
        root,
        allow_retained_stage=allow_retained_stage,
        code="activation_hardlink",
        message="Activation files must be single-link files or exact retained publication stages.",
    )
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
    if not path.is_file():
        raise ConditionedActivationError(
            "publication_file_invalid", "Publication artifacts must be regular owned files."
        )
    _require_path_link_contract(
        path,
        root,
        allow_retained_stage=True,
        code="publication_file_invalid",
        message="Publication artifacts must be regular owned files or exact retained publication stages.",
    )
    return path


def _verify_activation_publication(
    project_root: Path,
    freeze_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    if set(manifest) != _ACTIVATION_KEYS or manifest.get("schema_version") != CONDITIONED_DATASET_FREEZE_SCHEMA:
        raise ConditionedActivationError(
            "conditioned_dataset_v5_schema", "The Dataset-v5 activation schema is invalid."
        )
    if (
        manifest.get("dataset_version") != 5
        or manifest.get("dataset_kind") != "conditioned"
        or manifest.get("requires_semantic_labels") is not True
        or manifest.get("status") != "complete"
        or manifest.get("production_authorized") is not True
        or manifest.get("immutable") is not True
        or manifest.get("paths_are_relative") is not True
        or manifest.get("paths_exposed") is not False
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
        if binding.get("path") != _FIXED_ACTIVATION_ARTIFACT_PATHS[name]:
            raise ConditionedActivationError(
                "activation_artifact_path", "An activation artifact does not use its fixed publication path."
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
    actual = _publication_inventory(publication_root, exclude=freeze_path, boundary=project_root)
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
    _verify_embedded_conditioned_evidence(
        manifest,
        declared_artifacts=declared_artifacts,
        artifacts=artifacts,
        publication_root=publication_root,
        actual=actual,
        project_root=project_root,
    )
    return artifacts


def _verify_embedded_conditioned_evidence(
    manifest: Mapping[str, Any],
    *,
    declared_artifacts: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    publication_root: Path,
    actual: Mapping[str, Mapping[str, Any]],
    project_root: Path,
) -> None:
    evidence_paths = {relative for relative in actual if PurePosixPath(relative).parts[:1] == ("evidence",)}
    if evidence_paths != {
        _FIXED_ACTIVATION_ARTIFACT_PATHS["labeling_audit"],
        _FIXED_ACTIVATION_ARTIFACT_PATHS["labeling_audit_receipt"],
        _FIXED_ACTIVATION_ARTIFACT_PATHS["labeling_audit_action_record"],
        _FIXED_ACTIVATION_ARTIFACT_PATHS["validation_report"],
        _FIXED_ACTIVATION_ARTIFACT_PATHS["validation_receipt"],
        _FIXED_ACTIVATION_ARTIFACT_PATHS["validation_action_record"],
    }:
        raise ConditionedActivationError(
            "conditioned_evidence_inventory", "The embedded conditioned evidence inventory is invalid."
        )
    candidate_files: dict[str, dict[str, Any]] = {}
    for relative, record in actual.items():
        if relative in evidence_paths:
            continue
        if len(PurePosixPath(relative).parts) != 1:
            raise ConditionedActivationError(
                "conditioned_candidate_layout", "The conditioned candidate publication layout is invalid."
            )
        candidate_files[relative] = {
            "sha256": record["sha256"],
            "byte_count": record["byte_size"],
        }
    candidate_files = dict(sorted(candidate_files.items()))
    payload_inventory_sha256 = stable_hash(_conditioned_inventory_payload(candidate_files))

    required_context = {
        "view_manifest.json",
        "training_manifest.jsonl",
        "conditioning_vocabulary.json",
        "benchmark_manifest.json",
        "conditioned_records.jsonl",
        "coverage_report.json",
        "duplicate_report.json",
        "label_audit_subjects.json",
        "provenance_manifest.json",
    }
    if not required_context.issubset(candidate_files):
        raise ConditionedActivationError(
            "conditioned_candidate_context", "The conditioned publication lacks required candidate context."
        )

    view = _read_fixed_publication_mapping(publication_root, "view_manifest.json", actual)
    benchmark = _read_fixed_publication_mapping(publication_root, "benchmark_manifest.json", actual)
    coverage = _read_fixed_publication_mapping(publication_root, "coverage_report.json", actual)
    duplicate = _read_fixed_publication_mapping(publication_root, "duplicate_report.json", actual)
    subjects = _read_fixed_publication_mapping(publication_root, "label_audit_subjects.json", actual)
    provenance = _read_fixed_publication_mapping(publication_root, "provenance_manifest.json", actual)

    image_count = manifest.get("image_count")
    if (
        set(view)
        != {
            "schema_version",
            "view_identity",
            "image_count",
            "records_path",
            "records_sha256",
            "training_manifest_path",
            "training_manifest_sha256",
            "split_integrity_sha256",
            "coverage_report_sha256",
            "requires_semantic_labels",
            "human_truth_claim",
            "paths_are_portable",
        }
        or view.get("schema_version") != CONDITIONED_VIEW_SCHEMA
        or view.get("image_count") != image_count
        or view.get("records_path") != "conditioned_records.jsonl"
        or view.get("records_sha256") != candidate_files["conditioned_records.jsonl"]["sha256"]
        or view.get("training_manifest_path") != "training_manifest.jsonl"
        or view.get("training_manifest_sha256") != candidate_files["training_manifest.jsonl"]["sha256"]
        or view.get("coverage_report_sha256") != candidate_files["coverage_report.json"]["sha256"]
        or view.get("requires_semantic_labels") is not True
        or view.get("human_truth_claim") is not False
        or view.get("paths_are_portable") is not True
    ):
        raise ConditionedActivationError(
            "conditioned_view_binding", "The conditioned view does not match the exact publication."
        )

    category_counts = _exact_count_mapping(coverage.get("category_counts"), image_count)
    source_counts = _exact_count_mapping(coverage.get("source_counts"), image_count)
    split_counts = _exact_count_mapping(coverage.get("split_counts"), image_count)
    benchmark_category_counts = _exact_count_mapping(benchmark.get("category_counts"), None)
    sources = provenance.get("sources")
    if (
        provenance.get("schema_version") != CONDITIONED_PROVENANCE_SCHEMA
        or provenance.get("all_source_files_rehashed") is not True
        or provenance.get("paths_are_portable") is not True
        or not isinstance(sources, list)
        or not sources
        or any(not isinstance(source, Mapping) for source in sources)
    ):
        raise ConditionedActivationError(
            "conditioned_provenance_binding", "The conditioned provenance context is invalid."
        )
    source_bindings = [dict(source) for source in sources]
    managed_receipts = [source.get("managed_intake_receipt_identity") for source in source_bindings]
    expected_view_identity = stable_hash(
        {
            "managed_intake_receipt_identities": managed_receipts,
            "image_count": image_count,
            "records_sha256": candidate_files["conditioned_records.jsonl"]["sha256"],
        }
    )
    if view.get("view_identity") != expected_view_identity:
        raise ConditionedActivationError("conditioned_view_identity", "The conditioned view identity is invalid.")

    raw_licenses = manifest.get("licenses")
    if (
        not isinstance(raw_licenses, list)
        or not raw_licenses
        or any(not isinstance(value, str) or value not in {"cc0-1.0", "public-domain"} for value in raw_licenses)
    ):
        raise ConditionedActivationError("conditioned_license_binding", "The conditioned license binding is invalid.")
    licenses = [value for value in raw_licenses if isinstance(value, str)]
    raw_source_licenses = [source.get("license_id") for source in source_bindings]
    if any(not isinstance(value, str) for value in raw_source_licenses):
        raise ConditionedActivationError("conditioned_license_binding", "The conditioned license binding is invalid.")
    source_licenses = [value for value in raw_source_licenses if isinstance(value, str)]
    if licenses != sorted(set(licenses)) or sorted(set(source_licenses)) != licenses:
        raise ConditionedActivationError("conditioned_license_binding", "The conditioned license binding is invalid.")

    production_code_identity = duplicate.get("near_duplicate_implementation_code_inventory_sha256")
    retained_gate = duplicate.get("retained_near_duplicate_gate")
    if (
        duplicate.get("near_duplicate_algorithm") != CONDITIONED_NEAR_DUPLICATE_ALGORITHM
        or not _is_lower_sha256(duplicate.get("near_duplicate_config_identity"))
        or not _is_lower_sha256(production_code_identity)
        or not isinstance(retained_gate, Mapping)
        or set(retained_gate)
        != {
            "algorithm_id",
            "config",
            "config_identity",
            "retained_count",
            "violation_count",
            "violations",
            "ok",
            "gate_identity",
        }
        or retained_gate.get("algorithm_id") != CONDITIONED_NEAR_DUPLICATE_ALGORITHM
        or retained_gate.get("config_identity") != duplicate.get("near_duplicate_config_identity")
        or retained_gate.get("retained_count") != image_count
        or retained_gate.get("violation_count") != 0
        or retained_gate.get("violations") != []
        or retained_gate.get("ok") is not True
        or not _valid_embedded_identity(retained_gate, "gate_identity")
    ):
        raise ConditionedActivationError(
            "conditioned_duplicate_binding", "The conditioned retained-pair gate is invalid."
        )

    subjects_identity = _validate_conditioned_audit_subjects(
        subjects,
        image_count=image_count,
        category_counts=category_counts,
        source_counts=source_counts,
    )
    candidate_identity = stable_hash(
        {
            "schema_version": CONDITIONED_CANDIDATE_SCHEMA,
            "input_bindings": source_bindings,
            "production_code_identity": production_code_identity,
            "payload_inventory_sha256": payload_inventory_sha256,
            "image_count": image_count,
            "recipe": CONDITIONED_RECIPE,
        }
    )
    publication_identity = stable_hash(
        _conditioned_inventory_payload(
            {
                relative: {"sha256": record["sha256"], "byte_count": record["byte_size"]}
                for relative, record in actual.items()
            }
        )
    )
    if (
        manifest.get("dataset_identity") != candidate_identity
        or manifest.get("publication_identity_sha256") != publication_identity
        or manifest.get("publication_identity_sha256")
        != dict(manifest["publication_inventory"]).get("inventory_sha256")
        or manifest.get("labeling_audit_sha256") != dict(declared_artifacts["labeling_audit"]).get("sha256")
        or manifest.get("validation_report_sha256") != dict(declared_artifacts["validation_report"]).get("sha256")
    ):
        raise ConditionedActivationError(
            "conditioned_activation_binding", "The activation does not bind the exact conditioned publication."
        )

    context = {
        "candidate_identity": candidate_identity,
        "payload_inventory_sha256": payload_inventory_sha256,
        "image_count": image_count,
        "production_code_identity": production_code_identity,
        "label_audit_subjects_identity": subjects_identity,
        "subject_files": candidate_files,
        "audit_subjects": subjects,
        "category_counts": category_counts,
        "source_counts": source_counts,
        "split_counts": split_counts,
        "benchmark_category_counts": benchmark_category_counts,
        "retained_gate": retained_gate,
        "near_duplicate_config_identity": duplicate["near_duplicate_config_identity"],
    }
    evidence = {
        "label_audit": (
            _read_bound_evidence_mapping(
                artifacts["labeling_audit"],
                declared_artifacts["labeling_audit"],
                boundary=publication_root,
            ),
            _read_bound_evidence_mapping(
                artifacts["labeling_audit_receipt"],
                declared_artifacts["labeling_audit_receipt"],
                boundary=publication_root,
            ),
            _read_bound_evidence_mapping(
                artifacts["labeling_audit_action_record"],
                declared_artifacts["labeling_audit_action_record"],
                boundary=publication_root,
            ),
            "labeling_audit",
            "labeling_audit_receipt",
            "labeling_audit_action_record",
        ),
        "dataset_validation": (
            _read_bound_evidence_mapping(
                artifacts["validation_report"],
                declared_artifacts["validation_report"],
                boundary=publication_root,
            ),
            _read_bound_evidence_mapping(
                artifacts["validation_receipt"],
                declared_artifacts["validation_receipt"],
                boundary=publication_root,
            ),
            _read_bound_evidence_mapping(
                artifacts["validation_action_record"],
                declared_artifacts["validation_action_record"],
                boundary=publication_root,
            ),
            "validation_report",
            "validation_receipt",
            "validation_action_record",
        ),
    }
    action_job_id: str | None = None
    for kind, (report, receipt, action, report_artifact, receipt_artifact, action_artifact) in evidence.items():
        current_inventory = _verify_conditioned_evidence_report(kind, report, context)
        report_binding = dict(declared_artifacts[report_artifact])
        receipt_binding = dict(declared_artifacts[receipt_artifact])
        action_binding = dict(declared_artifacts[action_artifact])
        try:
            validated_receipt = validate_audit_receipt(
                receipt,
                kind=kind,
                expected_job_id=None,
                expected_report_sha256=report_binding["sha256"],
                expected_report_byte_count=report_binding["byte_count"],
                report=report,
                candidate=context,
                current_auditor_inventory=current_inventory,
            )
        except ConditionedAuditReceiptError as exc:
            raise ConditionedActivationError(
                "conditioned_evidence_receipt",
                "Embedded conditioned evidence lacks an applicable durable server audit receipt.",
            ) from exc
        try:
            validated_action = validate_audit_action_record(
                action,
                kind=kind,
                expected_job_id=None,
                expected_report_sha256=report_binding["sha256"],
                expected_report_byte_count=report_binding["byte_count"],
                report=report,
                expected_receipt_sha256=receipt_binding["sha256"],
                expected_receipt_byte_count=receipt_binding["byte_count"],
                receipt=validated_receipt,
                candidate=context,
                current_auditor_inventory=current_inventory,
            )
        except ConditionedAuditReceiptError as exc:
            raise ConditionedActivationError(
                "conditioned_evidence_action",
                "Embedded conditioned evidence lacks an applicable durable server audit action record.",
            ) from exc
        if action_binding["sha256"] != file_sha256(artifacts[action_artifact]):
            raise ConditionedActivationError(
                "conditioned_evidence_action", "An embedded audit action-record binding changed."
            )
        try:
            source_action = _project_relative_file(
                project_root,
                (
                    f"runs/v3/conditioned-dataset-v5/{validated_action['job_id']}/audit_actions/"
                    f"{kind}-{validated_action['operation_id']}.json"
                ),
                allow_retained_stage=True,
            )
        except ConditionedActivationError as exc:
            raise ConditionedActivationError(
                "conditioned_evidence_action",
                "The no-replace job-owned server audit action record is absent or invalid.",
            ) from exc
        if (
            file_sha256(source_action) != action_binding["sha256"]
            or source_action.stat().st_size != action_binding["byte_count"]
            or _read_mapping(source_action) != validated_action
        ):
            raise ConditionedActivationError(
                "conditioned_evidence_action",
                "The published audit action differs from its no-replace job-owned server record.",
            )
        current_job_id = str(validated_action["job_id"])
        if action_job_id is None:
            action_job_id = current_job_id
        elif current_job_id != action_job_id:
            raise ConditionedActivationError(
                "conditioned_evidence_action", "Embedded audit action records belong to different jobs."
            )


def _verify_conditioned_evidence_report(
    kind: str,
    report: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "verdict",
        "independent",
        "generated_by_conditioned_workflow",
        "auditor",
        "audit_run_identity",
        "bindings",
        "subject_files",
        "checks",
        "audit_subjects",
        "metrics",
    }
    schema = CONDITIONED_LABEL_AUDIT_SCHEMA if kind == "label_audit" else CONDITIONED_VALIDATION_SCHEMA
    gates = CONDITIONED_LABEL_AUDIT_GATES if kind == "label_audit" else CONDITIONED_VALIDATION_GATES
    if (
        set(report) != expected_keys
        or report.get("schema_version") != schema
        or report.get("verdict") != "PASS"
        or report.get("independent") is not True
        or report.get("generated_by_conditioned_workflow") is not False
        or _contains_private_or_absolute_path({key: value for key, value in report.items() if key != "auditor"})
    ):
        raise ConditionedActivationError(
            "conditioned_evidence_schema", "Embedded conditioned evidence has an invalid schema or verdict."
        )
    auditor = report.get("auditor")
    if not isinstance(auditor, Mapping) or set(auditor) != {
        "auditor_id",
        "code_identity_sha256",
        "implementation_inventory",
    }:
        raise ConditionedActivationError(
            "conditioned_evidence_auditor", "Embedded conditioned evidence lacks a trusted auditor binding."
        )
    try:
        current_inventory = trusted_auditor_inventory(kind)
    except Exception as exc:
        raise ConditionedActivationError(
            "conditioned_evidence_auditor", "The current trusted auditor inventory is unavailable."
        ) from exc
    if (
        auditor.get("auditor_id") != TRUSTED_AUDITOR_IDS[kind]
        or auditor.get("code_identity_sha256") != current_inventory["inventory_sha256"]
        or auditor.get("implementation_inventory") != current_inventory
    ):
        raise ConditionedActivationError(
            "conditioned_evidence_auditor", "Embedded conditioned evidence is stale or from an untrusted auditor."
        )
    expected_bindings = {
        key: context[key]
        for key in (
            "candidate_identity",
            "payload_inventory_sha256",
            "image_count",
            "production_code_identity",
            "label_audit_subjects_identity",
        )
    }
    if (
        report.get("bindings") != expected_bindings
        or report.get("subject_files") != context["subject_files"]
        or report.get("audit_subjects") != context["audit_subjects"]
    ):
        raise ConditionedActivationError(
            "conditioned_evidence_binding", "Embedded conditioned evidence does not bind the exact candidate."
        )
    checks = report.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != gates
        or any(type(value) is not str or value != "PASS" for value in checks.values())
    ):
        raise ConditionedActivationError(
            "conditioned_evidence_checks", "Every exact conditioned evidence gate must literally PASS."
        )
    audit_subjects = dict(context["audit_subjects"])
    if kind == "label_audit":
        expected_metrics = {
            "audited_record_ids": audit_subjects["required_label_audit_ids"],
            "stratified_sample_ids": audit_subjects["stratified_sample_ids"],
            "low_confidence_ids": audit_subjects["low_confidence_ids"],
            "disagreement_ids": audit_subjects["disagreement_ids"],
            "high_impact_ids": audit_subjects["high_impact_ids"],
            "generic_label_ids": audit_subjects["generic_label_ids"],
            "distributions": audit_subjects["distributions"],
            "quality_rates_basis_points": audit_subjects["quality_rates_basis_points"],
            "recomputed_visual_descriptor_bindings": audit_subjects["visual_descriptor_bindings"],
            "local_pixel_vision_config_identity": audit_subjects["local_pixel_vision_config_identity"],
        }
    else:
        expected_metrics = {
            "split_counts": context["split_counts"],
            "category_counts": context["category_counts"],
            "source_counts": context["source_counts"],
            "benchmark_category_counts": context["benchmark_category_counts"],
            "payload_inventory_sha256": context["payload_inventory_sha256"],
            "verified_file_count": len(context["subject_files"]),
            "near_duplicate_recomputation": {
                "algorithm_id": CONDITIONED_NEAR_DUPLICATE_ALGORITHM,
                "config_identity": context["near_duplicate_config_identity"],
                "retained_count": context["image_count"],
                "checked_same_category_pairs": sum(
                    int(count) * (int(count) - 1) // 2 for count in dict(context["category_counts"]).values()
                ),
                "violation_count": 0,
                "gate_identity": dict(context["retained_gate"])["gate_identity"],
            },
        }
    if report.get("metrics") != expected_metrics:
        raise ConditionedActivationError(
            "conditioned_evidence_metrics", "Embedded conditioned evidence metrics do not match the publication."
        )
    if not _valid_embedded_identity(report, "audit_run_identity"):
        raise ConditionedActivationError(
            "conditioned_evidence_identity", "The embedded conditioned audit-run identity is invalid."
        )
    return current_inventory


def _validate_conditioned_audit_subjects(
    subjects: Mapping[str, Any],
    *,
    image_count: int,
    category_counts: Mapping[str, int],
    source_counts: Mapping[str, int],
) -> str:
    expected_keys = {
        "schema_version",
        "stratified_sample_ids",
        "low_confidence_ids",
        "disagreement_ids",
        "high_impact_ids",
        "generic_label_ids",
        "required_label_audit_ids",
        "visual_descriptor_bindings",
        "local_pixel_vision_algorithm",
        "local_pixel_vision_config_identity",
        "distributions",
        "all_low_confidence_required",
        "all_disagreements_required",
        "all_high_impact_required",
        "all_generic_labels_required",
        "all_visual_descriptors_recompute_required",
        "quality_rates_basis_points",
        "human_truth_claim",
        "subjects_identity",
    }
    list_fields = (
        "stratified_sample_ids",
        "low_confidence_ids",
        "disagreement_ids",
        "high_impact_ids",
        "generic_label_ids",
        "required_label_audit_ids",
    )
    if (
        set(subjects) != expected_keys
        or subjects.get("schema_version") != CONDITIONED_AUDIT_SUBJECTS_SCHEMA
        or subjects.get("local_pixel_vision_algorithm") != "local_pixel_vision_v1"
        or not _is_lower_sha256(subjects.get("local_pixel_vision_config_identity"))
        or any(subjects.get(flag) is not True for flag in expected_keys if flag.startswith("all_"))
        or subjects.get("human_truth_claim") is not False
        or not _valid_embedded_identity(subjects, "subjects_identity")
    ):
        raise ConditionedActivationError(
            "conditioned_audit_subjects", "The conditioned audit-subject contract is invalid."
        )
    subject_lists: dict[str, list[str]] = {}
    for field in list_fields:
        value = subjects.get(field)
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or value != sorted(set(value))
        ):
            raise ConditionedActivationError(
                "conditioned_audit_subjects", "The conditioned audit-subject sets are invalid."
            )
        subject_lists[field] = value
    expected_required = sorted(
        set().union(
            subject_lists["stratified_sample_ids"],
            subject_lists["low_confidence_ids"],
            subject_lists["disagreement_ids"],
            subject_lists["high_impact_ids"],
            subject_lists["generic_label_ids"],
        )
    )
    visual = subjects.get("visual_descriptor_bindings")
    if not isinstance(visual, list) or len(visual) != image_count:
        raise ConditionedActivationError(
            "conditioned_audit_subjects", "The visual descriptor audit coverage is incomplete."
        )
    visual_ids: list[str] = []
    for row in visual:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"sprite_id", "descriptor_identity", "decoded_rgba_sha256"}
            or not isinstance(row.get("sprite_id"), str)
            or not row["sprite_id"]
            or not _is_lower_sha256(row.get("descriptor_identity"))
            or not _is_lower_sha256(row.get("decoded_rgba_sha256"))
        ):
            raise ConditionedActivationError(
                "conditioned_audit_subjects", "A visual descriptor audit binding is invalid."
            )
        visual_ids.append(str(row["sprite_id"]))
    if (
        visual_ids != sorted(set(visual_ids))
        or subject_lists["required_label_audit_ids"] != expected_required
        or not set(expected_required).issubset(visual_ids)
    ):
        raise ConditionedActivationError(
            "conditioned_audit_subjects", "The mandatory audit-subject coverage is invalid."
        )
    distributions = subjects.get("distributions")
    if not isinstance(distributions, Mapping) or set(distributions) != {
        "category",
        "source",
        "confidence",
        "confidence_reason",
        "disagreement",
        "generic_label",
    }:
        raise ConditionedActivationError("conditioned_audit_subjects", "The audit-subject distributions are invalid.")
    if (
        _exact_count_mapping(distributions.get("category"), image_count) != category_counts
        or _exact_count_mapping(distributions.get("source"), image_count) != source_counts
    ):
        raise ConditionedActivationError(
            "conditioned_audit_subjects", "The audit-subject distributions do not match the candidate."
        )
    for field in ("confidence", "confidence_reason", "disagreement", "generic_label"):
        _exact_count_mapping(distributions.get(field), image_count)
    quality = subjects.get("quality_rates_basis_points")
    if (
        not isinstance(quality, Mapping)
        or set(quality) != {"unknown_category", "generic_object", "disagreement", "useful_label"}
        or any(type(value) is not int or not 0 <= value <= 10_000 for value in quality.values())
    ):
        raise ConditionedActivationError("conditioned_audit_subjects", "The audit-subject quality rates are invalid.")
    return str(subjects["subjects_identity"])


def _conditioned_inventory_payload(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    normalized = {str(path): dict(record) for path, record in sorted(files.items())}
    return {
        "schema_version": CONDITIONED_PUBLICATION_INVENTORY_SCHEMA,
        "files": normalized,
        "file_count": len(normalized),
        "total_bytes": sum(int(record["byte_count"]) for record in normalized.values()),
    }


def _exact_count_mapping(value: Any, expected_total: int | None) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(
            not isinstance(key, str) or not key or type(count) is not int or count < 0 for key, count in value.items()
        )
    ):
        raise ConditionedActivationError("conditioned_count_binding", "A conditioned count distribution is invalid.")
    result = {str(key): int(count) for key, count in value.items()}
    if expected_total is not None and sum(result.values()) != expected_total:
        raise ConditionedActivationError(
            "conditioned_count_binding", "A conditioned count distribution does not match the image count."
        )
    return result


def _read_fixed_publication_mapping(
    root: Path,
    relative: str,
    actual: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    record = actual.get(relative)
    if not isinstance(record, Mapping):
        raise ConditionedActivationError(
            "conditioned_candidate_context", "A required conditioned candidate artifact is missing."
        )
    path = _publication_file(root, relative)
    return _read_bound_evidence_mapping(
        path,
        {"sha256": record.get("sha256"), "byte_count": record.get("byte_size")},
        boundary=root,
    )


def _read_bound_evidence_mapping(
    path: Path,
    binding: Mapping[str, Any],
    *,
    boundary: Path,
) -> dict[str, Any]:
    expected_sha256 = binding.get("sha256")
    expected_size = binding.get("byte_count")
    if (
        not _is_lower_sha256(expected_sha256)
        or type(expected_size) is not int
        or not 0 <= expected_size <= MAX_EMBEDDED_EVIDENCE_BYTES
    ):
        raise ConditionedActivationError(
            "conditioned_evidence_unreadable", "An embedded conditioned evidence document is unreadable."
        )
    try:
        with _held_mapping_snapshot(path, boundary, allow_retained_stage=True) as snapshot:
            if snapshot.byte_count != expected_size or snapshot.sha256 != expected_sha256:
                raise ValueError
            value = snapshot.value
    except (
        ConditionedActivationError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateMappingKeyError,
        ValueError,
    ) as exc:
        raise ConditionedActivationError(
            "conditioned_evidence_unreadable", "An embedded conditioned evidence document is unreadable."
        ) from exc
    if not isinstance(value, dict):
        raise ConditionedActivationError(
            "conditioned_evidence_unreadable", "An embedded conditioned evidence document must be an object."
        )
    return value


def _valid_embedded_identity(value: Mapping[str, Any], field: str) -> bool:
    identity = value.get(field)
    if not _is_lower_sha256(identity):
        return False
    payload = dict(value)
    payload.pop(field, None)
    return stable_hash(payload) == identity


def _is_lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _contains_private_or_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_private_or_absolute_path(key) or _contains_private_or_absolute_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_or_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    posix = PurePosixPath(candidate.replace("\\", "/"))
    windows = PureWindowsPath(candidate)
    folded = candidate.replace("\\", "/").casefold()
    return (
        "\x00" in candidate
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or folded.startswith("file:")
        or any(part == ".." for part in posix.parts)
        or "/users/" in folded
        or folded.startswith("users/")
        or "/home/" in folded
        or folded.startswith("home/")
    )


def _publication_inventory(
    root: Path,
    *,
    exclude: Path | None,
    boundary: Path,
) -> dict[str, dict[str, Any]]:
    try:
        target = require_confined_path(root, boundary, allow_root=True)
        exclude_relative = None
        if exclude is not None:
            excluded = require_confined_path(exclude, target)
            exclude_relative = excluded.relative_to(target).as_posix()
        current = Path(boundary).absolute()
        for part in target.relative_to(current).parts:
            current /= part
            if current.is_mount():
                raise UnsafeFilesystemOperation(f"publication inventory crosses a mount point: {current}")
        with open_anchored_directory(target, boundary) as anchor:
            root_device = anchor.directory_metadata().st_dev
            passes: list[dict[str, dict[str, Any]]] = []
            for _pass in range(2):
                files: dict[str, dict[str, Any]] = {}
                collision_keys: set[str] = set()
                _publication_inventory_directory(
                    anchor,
                    relative_parts=(),
                    exclude_relative=exclude_relative,
                    root_device=root_device,
                    files=files,
                    collision_keys=collision_keys,
                )
                passes.append(dict(sorted(files.items())))
            if passes[0] != passes[1]:
                raise ConditionedActivationError(
                    "publication_entry_changed",
                    "The publication changed between complete anchored inventory passes.",
                )
            return passes[0]
    except ConditionedActivationError:
        raise
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise ConditionedActivationError(
            "publication_entry_invalid", "The publication contains an unsafe or changing entry."
        ) from exc


def _publication_inventory_directory(
    anchor: AnchoredDirectory,
    *,
    relative_parts: tuple[str, ...],
    exclude_relative: str | None,
    root_device: int,
    files: dict[str, dict[str, Any]],
    collision_keys: set[str],
) -> None:
    try:
        anchor.verify()
        directory_before = anchor.directory_metadata()
        before_names = anchor.names()
        retained_aliases = {
            name for name in before_names if _publication_retained_stage_target(anchor, name, before_names) is not None
        }
        for name in before_names:
            try:
                metadata = anchor.lstat(name)
            except UnsafeFilesystemOperation as exc:
                raise ConditionedActivationError(
                    "publication_entry_invalid",
                    "The publication contains an unsafe linked directory or reparse entry.",
                ) from exc
            if name in retained_aliases:
                anchor.verify()
                continue
            relative = PurePosixPath(*relative_parts, name).as_posix()
            collision = unicodedata.normalize("NFC", relative).casefold()
            if collision in collision_keys:
                raise ConditionedActivationError(
                    "publication_entry_invalid", "The publication contains a case or Unicode path collision."
                )
            collision_keys.add(collision)
            if exclude_relative is not None and relative == exclude_relative:
                _require_publication_file_identity(metadata, metadata, root_device)
                anchor.verify()
                continue
            if stat.S_ISDIR(metadata.st_mode) and not _activation_metadata_is_link_or_reparse(metadata):
                if metadata.st_dev != root_device or (anchor.directory / name).is_mount():
                    raise ConditionedActivationError(
                        "publication_entry_invalid", "The publication contains a linked or mounted directory."
                    )
                with anchor.open_directory_immovable(name) as child:
                    child_metadata = child.directory_metadata()
                    if (
                        child_metadata.st_dev != root_device
                        or child_metadata.st_ino != metadata.st_ino
                        or child.directory.is_mount()
                    ):
                        raise ConditionedActivationError(
                            "publication_entry_invalid", "The publication contains a linked or mounted directory."
                        )
                    _publication_inventory_directory(
                        child,
                        relative_parts=(*relative_parts, name),
                        exclude_relative=exclude_relative,
                        root_device=root_device,
                        files=files,
                        collision_keys=collision_keys,
                    )
            elif stat.S_ISREG(metadata.st_mode) and not _activation_metadata_is_link_or_reparse(metadata):
                files[relative] = _publication_file_identity(anchor, name, root_device)
            else:
                raise ConditionedActivationError(
                    "publication_entry_invalid", "The publication contains an unsafe entry."
                )
            anchor.verify()
        directory_after = anchor.directory_metadata()
        if (
            anchor.names() != before_names
            or directory_after.st_dev != directory_before.st_dev
            or directory_after.st_ino != directory_before.st_ino
            or directory_after.st_mtime_ns != directory_before.st_mtime_ns
        ):
            raise ConditionedActivationError(
                "publication_entry_changed", "The publication changed while it was inventoried."
            )
    except ConditionedActivationError:
        raise
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedActivationError(
            "publication_entry_changed", "The publication changed while it was inventoried."
        ) from exc


def _publication_file_identity(
    anchor: AnchoredDirectory,
    name: str,
    root_device: int,
) -> dict[str, Any]:
    descriptor = -1
    byte_size = 0
    digest = hashlib.sha256()
    retained_alias: str | None = None
    try:
        before = anchor.lstat(name)
        retained_alias = _retained_mapping_stage_alias(anchor, name, before)
        _require_publication_file_identity(before, before, root_device)
        descriptor = anchor.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        opened = os.fstat(descriptor)
        _require_publication_file_identity(before, opened, root_device)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_size += len(chunk)
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        after = anchor.lstat(name)
        anchor.verify()
        _require_publication_file_identity(before, opened_after, root_device)
        _require_publication_file_identity(before, after, root_device)
        if _retained_mapping_stage_alias(anchor, name, after) != retained_alias:
            raise UnsafeFilesystemOperation("retained publication stage changed")
    except ConditionedActivationError:
        raise
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedActivationError(
            "publication_entry_changed", "A publication file changed while it was hashed."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if byte_size != before.st_size:
        raise ConditionedActivationError("publication_entry_changed", "A publication file changed while it was hashed.")
    return {"sha256": digest.hexdigest(), "byte_size": byte_size}


def _require_publication_file_identity(
    expected: os.stat_result,
    actual: os.stat_result,
    root_device: int,
) -> None:
    if (
        not stat.S_ISREG(actual.st_mode)
        or _activation_metadata_is_link_or_reparse(actual)
        or actual.st_nlink != expected.st_nlink
        or expected.st_nlink not in {1, 2}
        or actual.st_dev != root_device
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
        or actual.st_size != expected.st_size
        or actual.st_mtime_ns != expected.st_mtime_ns
    ):
        raise ConditionedActivationError("publication_entry_changed", "A publication file changed while it was hashed.")


def _publication_retained_stage_target(
    anchor: AnchoredDirectory,
    alias_name: str,
    names: Sequence[str],
) -> str | None:
    marker = ".staging-"
    if not alias_name.startswith(".") or marker not in alias_name:
        return None
    target_name, separator, suffix = alias_name[1:].rpartition(marker)
    if (
        separator != marker
        or not target_name
        or re.fullmatch(r"[0-9a-f]{32}", suffix) is None
        or target_name not in names
    ):
        raise ConditionedActivationError(
            "publication_entry_invalid",
            "The publication contains a malformed or orphaned retained-stage alias.",
        )
    target = anchor.lstat(target_name)
    try:
        retained_alias = _retained_mapping_stage_alias(anchor, target_name, target)
    except UnsafeFilesystemOperation as exc:
        raise ConditionedActivationError(
            "publication_entry_invalid",
            "The publication contains an invalid retained-stage alias.",
        ) from exc
    if retained_alias != alias_name:
        raise ConditionedActivationError(
            "publication_entry_invalid",
            "The publication contains an ambiguous retained-stage alias.",
        )
    return target_name


def _activation_metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _require_path_link_contract(
    path: Path,
    boundary: Path,
    *,
    allow_retained_stage: bool,
    code: str,
    message: str,
) -> None:
    """Bind a path to either one link or one exact writer-retained stage."""

    try:
        with open_anchored_directory(path.parent, boundary) as anchor:
            metadata = anchor.lstat(path.name)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _activation_metadata_is_link_or_reparse(metadata)
                or metadata.st_dev != anchor.directory_metadata().st_dev
            ):
                raise UnsafeFilesystemOperation("activation path is not a regular owned file")
            if allow_retained_stage:
                _retained_mapping_stage_alias(anchor, path.name, metadata)
            elif metadata.st_nlink != 1:
                raise UnsafeFilesystemOperation("activation path has an unsafe hard-link count")
            anchor.verify()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedActivationError(code, message) from exc


def _select_campaign(
    campaign_path: Path,
    profile: TrainingProfile,
    *,
    custom_spec: Mapping[str, Any] | None,
    document: Mapping[str, Any],
    held_snapshots: ExitStack,
    project_root: Path,
) -> tuple[dict[str, Any], Path]:
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
            selected = held_snapshots.enter_context(
                _held_mapping_snapshot(nested, project_root, allow_retained_stage=True)
            ).value
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
    _require_path_link_contract(
        path,
        project_root,
        allow_retained_stage=True,
        code="campaign_path_hardlink",
        message="Campaign inputs must be single-link files or exact retained publication stages.",
    )
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
                    if not confined.is_file():
                        return False
                    _require_path_link_contract(
                        confined,
                        root,
                        allow_retained_stage=False,
                        code="training_code_hardlink",
                        message="Audited training source files must remain single-link files.",
                    )
                    production_python.add(confined.relative_to(root.resolve()).as_posix())
        bound_root_prefixes = tuple(f"{relative.rstrip('/')}/" for relative in TRAINING_CODE_IDENTITY_RECURSIVE_ROOTS)
        required_under_roots = {relative for relative in required if relative.startswith(bound_root_prefixes)}
        if production_python != required_under_roots:
            return False
    except (ConditionedActivationError, OSError, ValueError, UnsafeFilesystemOperation):
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
        value = _parse_mapping_bytes(path.read_bytes(), suffix=path.suffix)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        yaml.YAMLError,
        _DuplicateMappingKeyError,
    ) as exc:
        raise ConditionedActivationError("activation_mapping", "An activation mapping is unreadable.") from exc
    if not isinstance(value, dict):
        raise ConditionedActivationError("activation_mapping", "An activation mapping must be an object.")
    return value


@contextmanager
def _held_mapping_snapshot(
    path: Path,
    project_root: Path,
    *,
    allow_retained_stage: bool = False,
) -> Iterator[_HeldMappingSnapshot]:
    """Hold one anchored owned inode while its exact bytes and link topology are consumed."""

    descriptor = -1
    anchor: AnchoredDirectory | None = None
    before: os.stat_result | None = None
    retained_alias: str | None = None
    try:
        anchor_context = open_anchored_directory(path.parent, project_root)
        anchor = anchor_context.__enter__()
        before = anchor.lstat(path.name)
        root_device = anchor.directory_metadata().st_dev
        if allow_retained_stage:
            retained_alias = _retained_mapping_stage_alias(anchor, path.name, before)
        elif before.st_nlink != 1:
            raise ValueError("mapping snapshot has an unsafe hard-link count")
        _require_mapping_snapshot_identity(before, before, root_device)
        if before.st_size > MAX_EMBEDDED_EVIDENCE_BYTES:
            raise ValueError("mapping snapshot exceeds its bounded byte limit")
        descriptor = anchor.open_file_immovable(path.name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        opened = os.fstat(descriptor)
        _require_mapping_snapshot_identity(before, opened, root_device)
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_EMBEDDED_EVIDENCE_BYTES + 1 - byte_count))
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > MAX_EMBEDDED_EVIDENCE_BYTES:
                raise ValueError("mapping snapshot exceeds its bounded byte limit")
        payload = b"".join(chunks)
        opened_after = os.fstat(descriptor)
        path_after = anchor.lstat(path.name)
        _require_mapping_snapshot_identity(before, opened_after, root_device)
        _require_mapping_snapshot_identity(before, path_after, root_device)
        if len(payload) != before.st_size:
            raise ValueError("mapping snapshot size changed")
        snapshot = _HeldMappingSnapshot(
            value=_parse_mapping_bytes(payload, suffix=path.suffix),
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
        )
    except ConditionedActivationError:
        if descriptor >= 0:
            os.close(descriptor)
        if anchor is not None:
            anchor_context.__exit__(None, None, None)
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        yaml.YAMLError,
        _DuplicateMappingKeyError,
    ) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if anchor is not None:
            anchor_context.__exit__(None, None, None)
        raise ConditionedActivationError(
            "activation_mapping",
            "An activation mapping is unreadable.",
        ) from exc
    except (
        OSError,
        UnsafeFilesystemOperation,
        ValueError,
    ) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if anchor is not None:
            anchor_context.__exit__(None, None, None)
        raise ConditionedActivationError(
            "activation_snapshot_changed",
            "An activation mapping changed while its exact bytes were held.",
        ) from exc

    completed = False
    try:
        yield snapshot
        completed = True
    finally:
        try:
            if completed:
                if anchor is None or before is None or descriptor < 0:
                    raise ConditionedActivationError(
                        "activation_snapshot_changed",
                        "An activation mapping snapshot lost its held identity.",
                    )
                try:
                    _require_mapping_snapshot_identity(
                        before,
                        os.fstat(descriptor),
                        anchor.directory_metadata().st_dev,
                    )
                    _require_mapping_snapshot_identity(
                        before,
                        anchor.lstat(path.name),
                        anchor.directory_metadata().st_dev,
                    )
                    if allow_retained_stage:
                        alias_after = _retained_mapping_stage_alias(anchor, path.name, before)
                        if alias_after != retained_alias:
                            raise UnsafeFilesystemOperation("retained mapping stage identity changed")
                except (OSError, UnsafeFilesystemOperation) as exc:
                    raise ConditionedActivationError(
                        "activation_snapshot_changed",
                        "An activation mapping changed before its held snapshot was released.",
                    ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if anchor is not None:
                anchor_context.__exit__(None, None, None)


def _require_mapping_snapshot_identity(
    expected: os.stat_result,
    actual: os.stat_result,
    root_device: int,
) -> None:
    if (
        not stat.S_ISREG(actual.st_mode)
        or _activation_metadata_is_link_or_reparse(actual)
        or actual.st_nlink != expected.st_nlink
        or expected.st_nlink not in {1, 2}
        or actual.st_dev != root_device
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
        or actual.st_size != expected.st_size
        or actual.st_mtime_ns != expected.st_mtime_ns
    ):
        raise ConditionedActivationError(
            "activation_snapshot_changed",
            "An activation mapping changed while its exact bytes were held.",
        )


def _retained_mapping_stage_alias(
    anchor: AnchoredDirectory,
    name: str,
    metadata: os.stat_result,
) -> str | None:
    prefix = f".{name}.staging-"
    candidates = [candidate for candidate in anchor.names() if candidate.startswith(prefix)]
    if metadata.st_nlink == 1:
        if candidates:
            raise UnsafeFilesystemOperation("single-link target has retained-stage residue")
        return None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _activation_metadata_is_link_or_reparse(metadata)
        or metadata.st_nlink != 2
        or len(candidates) != 1
        or re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", candidates[0]) is None
    ):
        raise UnsafeFilesystemOperation("target has an invalid retained publication stage")
    candidate = candidates[0]
    alias = anchor.lstat(candidate)
    if (
        not stat.S_ISREG(alias.st_mode)
        or _activation_metadata_is_link_or_reparse(alias)
        or alias.st_nlink != 2
        or alias.st_dev != metadata.st_dev
        or alias.st_ino != metadata.st_ino
        or alias.st_size != metadata.st_size
        or alias.st_mtime_ns != metadata.st_mtime_ns
    ):
        raise UnsafeFilesystemOperation("retained publication stage does not bind the exact target inode")
    return candidate


def _parse_mapping_bytes(payload: bytes, *, suffix: str) -> dict[str, Any]:
    text = payload.decode("utf-8")
    value = (
        yaml.load(text, Loader=_StrictSafeLoader)
        if suffix.casefold() in {".yaml", ".yml"}
        else json.loads(text, object_pairs_hook=_strict_json_mapping)
    )
    if not isinstance(value, dict):
        raise ConditionedActivationError("activation_mapping", "An activation mapping must be an object.")
    return value


def _strict_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise _DuplicateMappingKeyError
        mapping[key] = value
    return mapping


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
