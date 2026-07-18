"""Durable, fail-closed conditioned Dataset-v5 candidate and freeze workflow.

The service consumes only opaque, DatasetIntake-backed import receipts.  It is
offline after the explicit Harvest-to-Dataset import: preview, build, evidence
verification, and publication never contact a provider or network endpoint.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import stat
import threading
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from spritelab.dataset_maker.exporter import (
    DatasetMakerExportConfig,
    commit_anchored_dataset_maker_export,
    export_dataset_from_imported_sprites_anchored,
    verify_anchored_dataset_maker_export,
)
from spritelab.dataset_maker.importer import (
    ImportedSprite,
    ImportOptions,
    import_png_bytes_as_dataset_item,
)
from spritelab.dataset_maker.model import DatasetMakerItem, normalize_sprite_id
from spritelab.dataset_maker.qa import qa_dataset
from spritelab.dataset_maker.training_manifest import (
    build_training_manifest,
)
from spritelab.dataset_maker.training_manifest_qa import qa_training_manifest
from spritelab.harvest.semantic_v3 import (
    DEFAULT_NEGATIVE_TAGS,
    build_captions,
    build_prompt_phrases,
    build_semantic_v3_record,
    semantic_v3_to_json,
)
from spritelab.product_core import strict_json_dumps, strict_json_loads
from spritelab.product_features.conditioned_v5.audit_receipts import (
    AUDIT_ACTION_RECORD_ARTIFACTS,
    AUDIT_OPERATION_SCHEMA,
    AUDIT_RECEIPT_ARTIFACTS,
    ConditionedAuditReceiptError,
    audit_operation_identity,
    build_audit_action_record,
    build_audit_receipt,
    validate_audit_action_record,
    validate_audit_receipt,
)
from spritelab.product_features.conditioned_v5.identity import (
    TRUSTED_AUDITOR_IDS,
    conditioned_code_inventory,
    trusted_auditor_inventory,
)
from spritelab.product_features.conditioned_v5.intake import (
    ConditionedIntakeError,
    load_managed_intake,
    managed_intake_inventory,
    read_receipt_bound_derived_frame,
)
from spritelab.product_features.conditioned_v5.publication_commit import (
    PUBLICATION_JOURNAL_NAME,
    PublicationCommitError,
    build_campaign_commit,
    build_dataset_commit,
    build_publication_journal,
    campaign_commit_name,
    canonical_publication_commit_bytes,
    dataset_commit_name,
    validate_campaign_commit,
    validate_dataset_commit,
    validate_publication_journal,
)
from spritelab.product_features.harvest.storage import scan_artifacts
from spritelab.product_features.harvest.trusted_backend import AcquiredFile, HarvestLimits
from spritelab.product_features.training.action_lock import (
    ACTION_LOCK_PROTOCOL_IDENTITY,
    TrainingActionLock,
    TrainingActionLockError,
)
from spritelab.product_features.training.activation_commit import (
    ACTIVATION_PROJECT_COMMIT_NAME,
    ActivationCommitError,
    build_activation_commit_documents,
    build_activation_project_commit,
    canonical_activation_commit_bytes,
    validate_activation_commit_documents,
    validate_activation_project_commit,
)
from spritelab.training.campaign import stable_hash
from spritelab.utils.portable_paths import canonical_portable_relative_path
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    ExactPublicationUnsupported,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    open_anchored_directory,
    require_confined_path,
)
from spritelab.v3.config import ProjectConfig
from spritelab.v3.config import _validate as _validate_project_config

HANDOFF_SCHEMA = "spritelab.harvest.dataset-handoff.v2"
CANDIDATE_SCHEMA = "spritelab.dataset.conditioned-candidate.v1"
ACTIVATION_SCHEMA = "spritelab.dataset.freeze.conditioned.v5"
INVENTORY_SCHEMA = "spritelab.dataset.freeze.inventory.v1"
LABEL_AUDIT_SCHEMA = "spritelab.audit.conditioned-labels.v1"
DATASET_VALIDATION_SCHEMA = "spritelab.audit.conditioned-dataset.v1"
AUDIT_SUBJECTS_SCHEMA = "spritelab.audit.conditioned-subjects.v1"

TAXONOMY = (
    "character",
    "creature",
    "weapon",
    "tool",
    "armor",
    "potion",
    "food",
    "plant",
    "terrain",
    "building",
    "furniture",
    "vehicle",
    "effect",
    "icon",
    "interface",
    "unknown",
)
ALLOWED_LICENSES = frozenset({"cc0-1.0", "public-domain"})
RUN_ID_PATTERN = re.compile(r"^harvest-[a-z0-9][a-z0-9-]{5,80}$")
JOB_ID_PATTERN = re.compile(r"^conditioned-[a-f0-9]{20}$")
KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AUDITOR_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){1,7}$")

LABEL_AUDIT_GATES = frozenset(
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
DATASET_VALIDATION_GATES = frozenset(
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

LOCAL_PIXEL_VISION_ALGORITHM = "local_pixel_vision_v1"
CONDITIONING_RECIPE = "conditioned_filename_taxonomy_v1+local_pixel_vision_v1+near_duplicate_v2"
LOCAL_PIXEL_VISION_CONFIG = {
    "schema_version": "spritelab.dataset.local-pixel-vision-config.v1",
    "alpha_threshold": 255,
    "canvas_size": [32, 32],
    "dominant_palette": [
        ["black", [24, 24, 24]],
        ["gray", [128, 128, 128]],
        ["white", [232, 232, 232]],
        ["red", [210, 52, 52]],
        ["orange", [224, 126, 42]],
        ["yellow", [222, 204, 56]],
        ["green", [66, 170, 82]],
        ["cyan", [56, 184, 190]],
        ["blue", [58, 104, 206]],
        ["purple", [142, 74, 190]],
        ["pink", [220, 104, 164]],
        ["brown", [126, 82, 48]],
    ],
    "scale_max_bbox_dimension": {"tiny": 8, "small": 16, "medium": 24, "large": 30},
    "symmetry_mismatch_basis_points": {"high": 1000, "moderate": 2500},
    "edge_density_basis_points": {"low": 2500, "medium": 5000},
}
LOCAL_PIXEL_VISION_CONFIG_IDENTITY = stable_hash(LOCAL_PIXEL_VISION_CONFIG)

NEAR_DUPLICATE_ALGORITHM = "conditioned_near_duplicate_v2"
NEAR_DUPLICATE_CONFIG = {
    "schema_version": "spritelab.dataset.conditioned-near-duplicate-config.v1",
    "same_taxonomy_category": True,
    "max_perceptual_hamming": 2,
    "max_bbox_dimension_delta": 1,
    "max_bbox_center_delta_half_pixels": 2,
    "max_alpha_xor_pixels": 12,
}
NEAR_DUPLICATE_CONFIG_IDENTITY = stable_hash(NEAR_DUPLICATE_CONFIG)
VARIANT_FAMILY_CONFIG = {
    "schema_version": "spritelab.dataset.conditioned-variant-family-config.v1",
    "same_taxonomy_category": True,
    "max_perceptual_hamming": 6,
    "max_bbox_dimension_delta": 3,
    "max_bbox_center_delta_half_pixels": 6,
    "max_alpha_xor_pixels": 64,
}

_CATEGORY_TERMS: dict[str, frozenset[str]] = {
    "character": frozenset({"character", "hero", "player", "npc", "knight", "wizard", "warrior", "archer"}),
    "creature": frozenset({"creature", "monster", "beast", "dragon", "goblin", "orc", "slime", "animal"}),
    "weapon": frozenset({"weapon", "sword", "axe", "bow", "dagger", "mace", "spear", "staff", "gun"}),
    "tool": frozenset({"tool", "hammer", "pickaxe", "shovel", "hoe", "wrench", "saw", "fishing"}),
    "armor": frozenset({"armor", "armour", "helmet", "shield", "boots", "glove", "gauntlet", "chestplate"}),
    "potion": frozenset({"potion", "flask", "vial", "bottle", "elixir", "brew"}),
    "food": frozenset({"food", "bread", "meat", "fish", "fruit", "apple", "cheese", "cake", "meal"}),
    "plant": frozenset({"plant", "flower", "tree", "bush", "grass", "herb", "mushroom", "crop"}),
    "terrain": frozenset({"terrain", "floor", "wall", "ground", "water", "lava", "road", "tile", "cliff"}),
    "building": frozenset({"building", "house", "tower", "castle", "door", "window", "roof", "bridge"}),
    "furniture": frozenset({"furniture", "chair", "table", "bed", "shelf", "cabinet", "barrel", "crate"}),
    "vehicle": frozenset({"vehicle", "car", "cart", "boat", "ship", "wagon", "train"}),
    "effect": frozenset({"effect", "spell", "magic", "spark", "smoke", "fire", "explosion", "aura"}),
    "interface": frozenset({"interface", "ui", "button", "cursor", "panel", "frame", "menu", "checkbox"}),
    "icon": frozenset({"icon", "badge", "symbol", "emblem", "marker"}),
}
_CATEGORY_PATH_HINTS: dict[str, frozenset[str]] = {
    "character": frozenset({"character", "characters", "player", "players", "doll", "dolls"}),
    "creature": frozenset({"creature", "creatures", "mon", "mons", "monster", "monsters"}),
    "weapon": frozenset({"weapon", "weapons"}),
    "tool": frozenset({"tool", "tools"}),
    "armor": frozenset({"armor", "armour", "armors", "armours"}),
    "potion": frozenset({"potion", "potions"}),
    "food": frozenset({"food", "foods"}),
    "plant": frozenset({"plant", "plants", "flora"}),
    "terrain": frozenset({"terrain", "terrains", "dngn", "dungeon", "dungeons"}),
    "building": frozenset({"building", "buildings", "architecture"}),
    "furniture": frozenset({"furniture", "furnishings"}),
    "vehicle": frozenset({"vehicle", "vehicles"}),
    "effect": frozenset({"effect", "effects", "vfx"}),
    "interface": frozenset({"interface", "interfaces", "gui", "hud", "ui"}),
    "icon": frozenset({"icon", "icons"}),
}
_STOP_TOKENS = frozenset(
    {
        "32",
        "32x32",
        "sprite",
        "sprites",
        "pixel",
        "art",
        "png",
        "image",
        "images",
        "tile",
        "tiles",
        "variant",
        "alternate",
        "alt",
        "copy",
        "small",
        "large",
        "new",
        "old",
        "frame",
        "sheet",
    }
)


class ConditionedDatasetError(RuntimeError):
    """A public, privacy-safe workflow refusal."""

    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status_code = status_code


class _ConditionedMutationLock(AbstractContextManager["_ConditionedMutationLock"]):
    """Persistent, descriptor-bound cross-process lock for owned roots."""

    def __init__(self, root: Path, name: str, *, timeout_seconds: float = 5.0) -> None:
        self.root = root
        self.name = name
        self.timeout_seconds = timeout_seconds
        self._handle: Any = None
        self._anchor: AnchoredDirectory | None = None

    def __enter__(self) -> _ConditionedMutationLock:
        try:
            self._anchor = AnchoredDirectory(self.root, self.root)
            self._anchor.__enter__()
            self._handle = _open_conditioned_lock(self._anchor, self.name)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    _lock_conditioned_handle(self._handle)
                    _verify_conditioned_lock(self._handle, self._anchor, self.name)
                    self._anchor.verify()
                    return self
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise ConditionedDatasetError(
                            "mutation_conflict", "Another process is changing this conditioned workflow state."
                        ) from None
                    time.sleep(0.01)
        except BaseException as exc:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            if self._anchor is not None:
                self._anchor.__exit__(type(exc), exc, exc.__traceback__)
                self._anchor = None
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self._handle is not None:
                assert self._anchor is not None
                _verify_conditioned_lock(self._handle, self._anchor, self.name)
                _unlock_conditioned_handle(self._handle)
        finally:
            try:
                if self._handle is not None:
                    self._handle.close()
                    self._handle = None
            finally:
                if self._anchor is not None:
                    self._anchor.__exit__(exc_type, exc_value, traceback)
                    self._anchor = None


@dataclass(frozen=True)
class CandidatePolicy:
    """Server-owned candidate bounds; browsers cannot override them."""

    min_images: int = 2_000
    max_images: int = 3_000
    target_images: int = 2_500
    max_source_files: int = 5_000
    max_source_bytes: int = 512 * 1024 * 1024
    max_file_bytes: int = 16 * 1024 * 1024
    max_depth: int = 8
    max_events: int = 500

    def __post_init__(self) -> None:
        if not 1 <= self.min_images <= self.target_images <= self.max_images:
            raise ValueError("Candidate count policy is invalid.")
        if self.max_images > self.max_source_files:
            raise ValueError("Candidate maximum cannot exceed the source file cap.")
        if min(self.max_source_bytes, self.max_file_bytes, self.max_depth, self.max_events) <= 0:
            raise ValueError("Candidate resource limits must be positive.")


@dataclass(frozen=True)
class _SourceRecord:
    relative_path: str
    path: Path
    byte_count: int
    byte_sha256: str
    pixel_sha256: str
    alpha_sha256: str
    alpha_bitmap: bytes
    alpha_bbox: tuple[int, int, int, int]
    perceptual_hash: int
    category: str
    object_name: str
    tokens: tuple[str, ...]
    source_id: str
    source_title: str
    creator: str
    license_id: str
    license_evidence: Mapping[str, Any]
    visual_descriptor: Mapping[str, Any]
    visual_tags: tuple[str, ...]
    content: bytes = b""
    source_group_identity: str | None = None
    derivation: Mapping[str, Any] | None = None


HandoffLoader = Callable[[str], Mapping[str, Any]]
CampaignBuilder = Callable[..., Any]
ManagedIntakeLoader = Callable[[str], Mapping[str, Any]]
ActivationLoader = Callable[..., Any]
IndependentAuditRunner = Callable[..., Mapping[str, Any]]
TrainingInfrastructureAuditRunner = Callable[..., Any]


class ConditionedDatasetService:
    """Disk-backed candidate workflow with explicit audit and freeze gates."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        handoff_loader: HandoffLoader | None = None,
        managed_intake_loader: ManagedIntakeLoader | None = None,
        campaign_builder: CampaignBuilder | None = None,
        activation_loader: ActivationLoader | None = None,
        independent_audit_runner: IndependentAuditRunner | None = None,
        training_infrastructure_audit_runner: TrainingInfrastructureAuditRunner | None = None,
        policy: CandidatePolicy | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.harvest_root = self.project_root / "harvest_runs"
        self.jobs_root = self.project_root / "runs" / "v3" / "conditioned-dataset-v5"
        self.datasets_root = self.project_root / "datasets"
        self.campaigns_root = self.project_root / "campaigns"
        self.policy = policy or CandidatePolicy()
        self._handoff_loader = handoff_loader
        self._managed_intake_loader = managed_intake_loader
        self._campaign_builder = campaign_builder
        self._activation_loader = activation_loader
        if independent_audit_runner is not None and not callable(independent_audit_runner):
            raise TypeError("independent_audit_runner must be callable")
        self._independent_audit_runner = independent_audit_runner
        if training_infrastructure_audit_runner is not None and not callable(training_infrastructure_audit_runner):
            raise TypeError("training_infrastructure_audit_runner must be callable")
        self._training_infrastructure_audit_runner = training_infrastructure_audit_runner
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._instance_id = uuid.uuid4().hex
        self._lease_seconds = 300

    def inventory(self) -> dict[str, Any]:
        """Return passive published-intake/job inventory without hashing artifacts."""

        return {
            "schema_version": "spritelab.dataset.conditioned-inventory.v1",
            "managed_intakes": managed_intake_inventory(self.project_root),
            "jobs": self._job_inventory(),
            "count_policy": {
                "minimum": self.policy.min_images,
                "target": self.policy.target_images,
                "maximum": self.policy.max_images,
            },
            "taxonomy": list(TAXONOMY),
            "config_sha256": self._config_sha256(),
            "network_actions": 0,
            "paths_exposed": False,
        }

    def _config_sha256(self) -> str | None:
        path = self.project_root / "spritelab.yaml"
        if not os.path.lexists(path):
            return None
        try:
            with AnchoredDirectory(self.project_root, self.project_root) as anchor:
                payload = _read_anchored_regular_bytes(anchor, "spritelab.yaml", max_bytes=16 * 1024 * 1024)
        except (ConditionedDatasetError, UnsafeFilesystemOperation):
            return None
        return hashlib.sha256(payload).hexdigest()

    def preview(self, dataset_references: Sequence[str] | str) -> dict[str, Any]:
        """Re-verify managed imports and deterministically preview conditioning."""

        normalized_ids = _normalize_dataset_references(dataset_references)
        sources = [self._verified_source(reference) for reference in normalized_ids]
        records: list[_SourceRecord] = []
        exclusions: list[str] = []
        for source in sources:
            source_records, source_exclusions = self._inspect_records(source)
            records.extend(source_records)
            exclusions.extend(source_exclusions)
        records, duplicate_exclusions, near_duplicate_exclusions = _deduplicate_records(records)
        exclusions.extend(duplicate_exclusions)
        selected = _representative_selection(records, self.policy.target_images)
        counts = Counter(record.category for record in selected)
        source_counts = Counter(record.source_id for record in selected)
        ready = self.policy.min_images <= len(selected) <= self.policy.max_images
        return {
            "schema_version": "spritelab.dataset.conditioned-preview.v1",
            "dataset_references": list(normalized_ids),
            "handoff_identities": [source["handoff_identity"] for source in sources],
            "managed_intake_receipt_identities": [source["managed_intake_receipt_identity"] for source in sources],
            "source_ids": [source["source_id"] for source in sources],
            "license_ids": sorted({source["license_id"] for source in sources}),
            "eligible_unique_images": len(records),
            "selected_images": len(selected),
            "category_counts": dict(sorted(counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "excluded_counts": dict(sorted(Counter(exclusions).items())),
            "near_duplicate_exclusions": near_duplicate_exclusions,
            "ready_to_build": ready,
            "blockers": [] if ready else ["A conditioned Dataset-v5 candidate requires 2,000-3,000 unique images."],
            "labels_are_human_truth": False,
            "label_evidence": "deterministic_filename_and_relative_path_tokens",
            "paths_exposed": False,
        }

    def start_build(
        self,
        dataset_references: Sequence[str] | str,
        *,
        idempotency_key: str,
        explicit_action: bool,
    ) -> tuple[dict[str, Any], bool]:
        """Start a durable candidate build; repeated identical keys reuse the job."""

        normalized_ids = _normalize_dataset_references(dataset_references)
        if not KEY_PATTERN.fullmatch(str(idempotency_key)):
            raise ConditionedDatasetError(
                "invalid_idempotency_key", "A stable idempotency key is required.", status_code=422
            )
        if explicit_action is not True:
            raise ConditionedDatasetError(
                "explicit_build_action_required", "Candidate build requires an explicit action.", status_code=422
            )
        request_identity = stable_hash({"dataset_references": list(normalized_ids), "idempotency_key": idempotency_key})
        runs_root = _ensure_anchored_child_directory(self.project_root, self.project_root, "runs")
        v3_root = _ensure_anchored_child_directory(runs_root, self.project_root, "v3")
        jobs_root = _ensure_anchored_child_directory(v3_root, self.project_root, "conditioned-dataset-v5")
        if jobs_root != self.jobs_root:
            raise ConditionedDatasetError("jobs_root_unsafe", "The conditioned job directory changed.")
        if not _safe_directory(self.jobs_root):
            raise ConditionedDatasetError("jobs_root_unsafe", "The conditioned job directory is unsafe.")
        with self._lock, _ConditionedMutationLock(self.jobs_root, ".conditioned-jobs.lock"):
            for existing in self._job_inventory():
                if existing.get("idempotency_key") == idempotency_key:
                    if existing.get("request_identity") != request_identity:
                        raise ConditionedDatasetError(
                            "idempotency_conflict",
                            "That idempotency key already belongs to a different candidate request.",
                        )
                    return existing, False
                if existing.get("status") in {"RUNNING", "CANCELLING"}:
                    raise ConditionedDatasetError(
                        "build_conflict", "Another conditioned candidate build is already active."
                    )
            # Fail before mutating job state if the source is not a valid completed handoff.
            sources = [self._verified_source(reference) for reference in normalized_ids]
            job_id = f"conditioned-{uuid.uuid4().hex[:20]}"
            job_root, _job_identity = _create_anchored_named_directory(self.jobs_root, self.project_root, job_id)
            state = {
                "schema_version": "spritelab.dataset.conditioned-job.v1",
                "job_id": job_id,
                "dataset_references": list(normalized_ids),
                "handoff_identities": [source["handoff_identity"] for source in sources],
                "managed_intake_receipt_identities": [source["managed_intake_receipt_identity"] for source in sources],
                "idempotency_key": idempotency_key,
                "request_identity": request_identity,
                "status": "RUNNING",
                "stage": "queued",
                "current": 0,
                "total": sum(int(source["artifact_count"]) for source in sources),
                "message": "Conditioned candidate build queued.",
                "created_at": _now(),
                "updated_at": _now(),
                "events": [],
                "candidate": None,
                "evidence": {},
                "audit_operations": {},
                "audit_operation_history": [],
                "publication": None,
                "freeze_authorization": None,
                "activation_authorization": None,
                "lease": self._lease(),
                "paths_exposed": False,
            }
            self._write_state(job_root, state)
            worker = threading.Thread(
                target=self._run_build,
                args=(job_id,),
                name=f"spritelab-{job_id}",
                daemon=True,
            )
            self._threads[job_id] = worker
            worker.start()
            return self.job(job_id), True

    def job(self, job_id: str) -> dict[str, Any]:
        root = self._job_root(job_id)
        with self._lock, _ConditionedMutationLock(root, ".conditioned-state.lock"):
            state = self._read_state(root)
            thread = self._threads.get(job_id)
        if state.get("status") in {"RUNNING", "CANCELLING"} and thread is not None and thread.is_alive():
            return state
        if state.get("status") in {"RUNNING", "CANCELLING"} and self._lease_is_fresh(state.get("lease")):
            return state
        if state.get("status") in {"RUNNING", "CANCELLING"}:
            projected = dict(state)
            projected["status"] = "INTERRUPTED"
            projected["message"] = (
                "The durable worker lease expired before this build completed; start a new build to retry."
            )
            return projected
        return self._project_activation_state(root, state)

    def cancel(self, job_id: str, *, explicit_action: bool) -> dict[str, Any]:
        if explicit_action is not True:
            raise ConditionedDatasetError(
                "explicit_cancel_action_required", "Cancellation requires an explicit action.", status_code=422
            )
        root = self._job_root(job_id)
        with self._state_transaction(root) as state:
            if state["status"] not in {"RUNNING", "CANCELLING"}:
                raise ConditionedDatasetError("job_not_cancellable", "Only an active candidate build can be cancelled.")
            cancel = {
                "schema_version": "spritelab.dataset.conditioned-cancel.v1",
                "job_id": job_id,
                "requested_at": _now(),
                "explicit_action": True,
                "paths_exposed": False,
            }
            with open_anchored_directory(root, self.project_root) as anchor:
                anchor.atomic_write_bytes(
                    "cancel.json",
                    (strict_json_dumps(cancel, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                )
            state["status"] = "CANCELLING"
            state["stage"] = "cancelling"
            state["message"] = "Cancellation requested; publication remains disabled."
            state["updated_at"] = _now()
        return self.job(job_id)

    def run_independent_audit(self, job_id: str, *, kind: str, explicit_action: bool) -> dict[str, Any]:
        """Own one durable audit lane until its terminal action record is selected."""

        if explicit_action is not True:
            raise ConditionedDatasetError(
                "explicit_audit_action_required",
                "A server-managed independent audit requires an explicit action.",
                status_code=422,
            )
        if kind not in {"label_audit", "dataset_validation"}:
            raise ConditionedDatasetError(
                "evidence_kind", "Evidence kind must be label_audit or dataset_validation.", status_code=422
            )
        root = self._job_root(job_id)
        try:
            with _ConditionedMutationLock(root, f".conditioned-{kind}.lock", timeout_seconds=0.05):
                return self._run_independent_audit_owned(job_id, kind=kind)
        except ConditionedDatasetError as exc:
            if exc.code == "mutation_conflict":
                raise ConditionedDatasetError(
                    "independent_audit_conflict",
                    "That independent audit kind already has a live server operation.",
                ) from exc
            raise

    def _run_independent_audit_owned(self, job_id: str, *, kind: str) -> dict[str, Any]:
        """Run after acquiring the process-released durable ownership lane."""

        root = self._job_root(job_id)
        operation_id = f"audit-{uuid.uuid4().hex}"
        started_at = _now()
        initial_candidate: dict[str, Any]
        initial_inventory: dict[str, Any]
        operation_identity: str
        with self._state_transaction(root) as state:
            if state.get("status") != "NEEDS_REVIEW" or not isinstance(state.get("candidate"), Mapping):
                raise ConditionedDatasetError(
                    "candidate_not_ready", "Complete an unpublished candidate before running its independent audit."
                )
            initial_candidate = self._load_candidate(root, state)
            self._revalidate_candidate_context(root, state, initial_candidate)
            initial_inventory = trusted_auditor_inventory(kind)
            auditor_id = TRUSTED_AUDITOR_IDS[kind]
            inventory_identity = str(initial_inventory.get("inventory_sha256") or "")
            if initial_inventory.get("auditor_id") != auditor_id or not SHA256_PATTERN.fullmatch(inventory_identity):
                raise ConditionedDatasetError(
                    "evidence_auditor", "The current trusted auditor inventory is unavailable."
                )
            try:
                operation_identity = audit_operation_identity(
                    kind=kind,
                    job_id=job_id,
                    operation_id=operation_id,
                    candidate_identity=str(initial_candidate["candidate_identity"]),
                    payload_inventory_sha256=str(initial_candidate["payload_inventory_sha256"]),
                    image_count=int(initial_candidate["image_count"]),
                    auditor_id=auditor_id,
                    auditor_code_identity_sha256=inventory_identity,
                    auditor_inventory_sha256=inventory_identity,
                    started_at=started_at,
                )
            except ConditionedAuditReceiptError as exc:
                raise ConditionedDatasetError(
                    "independent_audit_operation", "The independent audit operation binding is invalid."
                ) from exc
            operations_raw = state.get("audit_operations")
            operations = dict(operations_raw) if isinstance(operations_raw, Mapping) else {}
            previous = operations.get(kind)
            if isinstance(previous, Mapping) and previous.get("terminal_status") == "RUNNING":
                interrupted = {
                    **dict(previous),
                    "terminal_status": "INTERRUPTED",
                    "completed_at": started_at,
                }
                history = list(state.get("audit_operation_history") or ())
                history.append(interrupted)
                state["audit_operation_history"] = history[-100:]
            operations[kind] = {
                "schema_version": AUDIT_OPERATION_SCHEMA,
                "audit_kind": kind,
                "job_id": job_id,
                "operation_id": operation_id,
                "operation_identity": operation_identity,
                "terminal_status": "RUNNING",
                "server_managed": True,
                "owner_instance_id": self._instance_id,
                "candidate_identity": initial_candidate["candidate_identity"],
                "payload_inventory_sha256": initial_candidate["payload_inventory_sha256"],
                "image_count": initial_candidate["image_count"],
                "auditor_id": auditor_id,
                "auditor_code_identity_sha256": inventory_identity,
                "auditor_inventory_sha256": inventory_identity,
                "started_at": started_at,
                "completed_at": None,
                "paths_exposed": False,
            }
            state["audit_operations"] = operations
            evidence = dict(state.get("evidence") or {})
            evidence.pop(kind, None)
            state["evidence"] = evidence
            state["message"] = "A server-managed independent audit is running over the exact candidate."
            state["updated_at"] = _now()

        try:
            runner = self._independent_audit_runner
            if runner is None:
                from spritelab.product_features.conditioned_v5.audit_runner import run_independent_audit

                runner = run_independent_audit
            produced = runner(
                kind,
                root,
                initial_candidate,
                project_root=self.project_root,
                progress=lambda _stage, _current, _total, _message: None,
                cancelled=lambda: False,
            )
            if not isinstance(produced, Mapping):
                raise ConditionedDatasetError(
                    "independent_audit_report", "The trusted independent auditor returned an invalid report."
                )
            report = dict(produced)
        except Exception as exc:
            self._mark_audit_operation_failed(root, kind=kind, operation_identity=operation_identity)
            if isinstance(exc, ConditionedDatasetError):
                raise
            raise ConditionedDatasetError(
                "independent_audit_failed",
                "The server-managed independent audit did not complete; no PASS receipt was published.",
            ) from exc

        try:
            completed_at = _now()
            with self._state_transaction(root) as state:
                active_operations = state.get("audit_operations")
                active = active_operations.get(kind) if isinstance(active_operations, Mapping) else None
                if (
                    not isinstance(active, Mapping)
                    or active.get("operation_id") != operation_id
                    or active.get("operation_identity") != operation_identity
                    or active.get("terminal_status") != "RUNNING"
                ):
                    raise ConditionedDatasetError(
                        "independent_audit_operation_changed",
                        "The independent audit operation changed before terminal publication.",
                    )
                candidate = self._load_candidate(root, state)
                self._revalidate_candidate_context(root, state, candidate)
                if (
                    candidate.get("candidate_identity") != initial_candidate.get("candidate_identity")
                    or candidate.get("payload_inventory_sha256") != initial_candidate.get("payload_inventory_sha256")
                    or candidate.get("image_count") != initial_candidate.get("image_count")
                ):
                    raise ConditionedDatasetError(
                        "independent_audit_candidate_changed",
                        "The conditioned candidate changed while its independent audit was running.",
                    )
                current_inventory = trusted_auditor_inventory(kind)
                if current_inventory != initial_inventory:
                    raise ConditionedDatasetError(
                        "independent_audit_auditor_changed",
                        "The trusted auditor implementation changed while its audit was running.",
                    )
                normalized = self._validate_evidence(kind, report, candidate)
                report_content = (strict_json_dumps(normalized, indent=2, sort_keys=True) + "\n").encode("utf-8")
                report_sha256 = hashlib.sha256(report_content).hexdigest()
                receipt = build_audit_receipt(
                    kind=kind,
                    job_id=job_id,
                    operation_id=operation_id,
                    report_sha256=report_sha256,
                    report_byte_count=len(report_content),
                    report=normalized,
                    candidate=candidate,
                    current_auditor_inventory=current_inventory,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                receipt_content = (strict_json_dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
                receipt_sha256 = hashlib.sha256(receipt_content).hexdigest()
                action_record = build_audit_action_record(
                    kind=kind,
                    job_id=job_id,
                    report_sha256=report_sha256,
                    report_byte_count=len(report_content),
                    report=normalized,
                    receipt_sha256=receipt_sha256,
                    receipt_byte_count=len(receipt_content),
                    receipt=receipt,
                    candidate=candidate,
                    current_auditor_inventory=current_inventory,
                    committed_at=_now(),
                )
                action_content = (strict_json_dumps(action_record, indent=2, sort_keys=True) + "\n").encode("utf-8")
                action_sha256 = hashlib.sha256(action_content).hexdigest()
                evidence_root = _ensure_anchored_child_directory(root, self.project_root, "evidence")
                actions_root = _ensure_anchored_child_directory(root, self.project_root, "audit_actions")
                report_name = f"{kind}-{report_sha256}.json"
                receipt_name = f"{kind}-receipt-{receipt['receipt_identity']}.json"
                action_name = f"{kind}-{operation_id}.json"
                with open_anchored_directory(evidence_root, self.project_root) as anchor:
                    _publish_or_reuse_immutable_file(
                        anchor,
                        report_name,
                        report_content,
                        residue_prefix=f".rollback-{kind}-report-",
                    )
                    _publish_or_reuse_immutable_file(
                        anchor,
                        receipt_name,
                        receipt_content,
                        residue_prefix=f".rollback-{kind}-receipt-",
                    )
                with open_anchored_directory(actions_root, self.project_root) as action_anchor:
                    _publish_fresh_immutable_file(
                        action_anchor,
                        action_name,
                        action_content,
                        residue_prefix=f".rollback-{kind}-action-",
                    )
                evidence = dict(state.get("evidence") or {})
                evidence[kind] = {
                    "relative_path": (evidence_root / report_name).relative_to(root).as_posix(),
                    "sha256": report_sha256,
                    "byte_count": len(report_content),
                    "auditor_id": normalized["auditor"]["auditor_id"],
                    "audit_run_identity": normalized["audit_run_identity"],
                    "operation_id": operation_id,
                    "operation_identity": operation_identity,
                    "receipt": {
                        "relative_path": (evidence_root / receipt_name).relative_to(root).as_posix(),
                        "sha256": receipt_sha256,
                        "byte_count": len(receipt_content),
                        "receipt_identity": receipt["receipt_identity"],
                    },
                    "action": {
                        "relative_path": (actions_root / action_name).relative_to(root).as_posix(),
                        "sha256": action_sha256,
                        "byte_count": len(action_content),
                        "record_identity": action_record["record_identity"],
                    },
                }
                state["evidence"] = evidence
                operations = dict(active_operations)
                operations[kind] = {
                    **dict(active),
                    "terminal_status": "PASS",
                    "completed_at": completed_at,
                    "report_sha256": report_sha256,
                    "audit_run_identity": normalized["audit_run_identity"],
                    "receipt_identity": receipt["receipt_identity"],
                    "action_record_identity": action_record["record_identity"],
                }
                state["audit_operations"] = operations
                state["message"] = (
                    "Server-managed independent PASS evidence recorded; both audit kinds remain required."
                )
                state["updated_at"] = _now()
        except Exception as exc:
            self._mark_audit_operation_failed(root, kind=kind, operation_identity=operation_identity)
            if isinstance(exc, ConditionedDatasetError):
                raise
            if isinstance(exc, ConditionedAuditReceiptError):
                raise ConditionedDatasetError(
                    "independent_audit_receipt", "The server-managed independent audit receipt is invalid."
                ) from exc
            raise
        return self.job(job_id)

    def _mark_audit_operation_failed(self, root: Path, *, kind: str, operation_identity: str) -> None:
        """Terminalize only the still-owned audit operation without masking its error."""

        try:
            with self._state_transaction(root) as state:
                raw_operations = state.get("audit_operations")
                operations = dict(raw_operations) if isinstance(raw_operations, Mapping) else {}
                active = operations.get(kind)
                if (
                    not isinstance(active, Mapping)
                    or active.get("operation_identity") != operation_identity
                    or active.get("terminal_status") != "RUNNING"
                ):
                    return
                operations[kind] = {
                    **dict(active),
                    "terminal_status": "FAILED",
                    "completed_at": _now(),
                }
                state["audit_operations"] = operations
                state["message"] = "The server-managed independent audit failed; no PASS receipt was selected."
                state["updated_at"] = _now()
        except Exception:
            return

    def publish(
        self,
        job_id: str,
        *,
        candidate_identity: str,
        label_audit_sha256: str,
        dataset_validation_sha256: str,
        authorization_id: str,
        explicit_action: bool,
        authorize_one_time_freeze: bool,
    ) -> dict[str, Any]:
        """Publish one fresh content-addressed freeze and its bound campaign."""

        if explicit_action is not True or authorize_one_time_freeze is not True:
            raise ConditionedDatasetError(
                "one_time_freeze_authorization_required",
                "Publication requires an explicit one-time production-freeze authorization.",
                status_code=422,
            )
        if not KEY_PATTERN.fullmatch(str(authorization_id)):
            raise ConditionedDatasetError(
                "freeze_authorization_id",
                "A stable one-time authorization ID is required.",
                status_code=422,
            )
        root = self._job_root(job_id)

        def recover_publication_state() -> None:
            # Direct-final publication is append-only. A failed state write never
            # rolls back marker-bound bytes; an exact retry adopts them instead.
            current = self._read_state(root)
            current["publication"] = None
            current["status"] = "FAILED"
            current["stage"] = "publication_state_failed"
            current["message"] = "Publication state commit failed; exact immutable outputs were retained for retry."
            current["updated_at"] = _now()
            self._write_state_payload(root, current)

        with self._state_transaction(root, write_failure_rollback=recover_publication_state) as state:
            authorization_binding = {
                "authorization_id_sha256": hashlib.sha256(authorization_id.encode("utf-8")).hexdigest(),
                "candidate_identity": candidate_identity,
                "label_audit_sha256": label_audit_sha256,
                "dataset_validation_sha256": dataset_validation_sha256,
                "one_time": True,
            }
            existing_authorization = state.get("freeze_authorization")
            if existing_authorization is not None and (
                not isinstance(existing_authorization, Mapping)
                or any(existing_authorization.get(key) != value for key, value in authorization_binding.items())
            ):
                raise ConditionedDatasetError(
                    "freeze_authorization_consumed",
                    "This candidate's one-time freeze authorization was already consumed by another request.",
                )
            candidate = self._load_candidate(root, state)
            if candidate.get("candidate_identity") != candidate_identity:
                raise ConditionedDatasetError(
                    "candidate_identity_changed",
                    "The candidate identity no longer matches the authorized build.",
                )
            actual_inventory = _inventory(root / "candidate" / "phase7", self.project_root)
            if actual_inventory != candidate.get("payload_inventory"):
                raise ConditionedDatasetError(
                    "candidate_bytes_changed",
                    "Candidate artifact bytes changed after build; publication was refused.",
                )
            evidence = self._verified_selected_evidence(
                root,
                state,
                candidate,
                label_audit_sha256=label_audit_sha256,
                dataset_validation_sha256=dataset_validation_sha256,
            )
            self._revalidate_candidate_context(root, state, candidate)
            if existing_authorization is None:
                state["freeze_authorization"] = {**authorization_binding, "consumed_at": _now()}
            state["status"] = "RUNNING"
            state["stage"] = "publishing"
            state["message"] = "Publishing the authorized immutable freeze."
            state["updated_at"] = _now()
            self._write_state_unlocked(root, state)
            try:
                with _ConditionedMutationLock(self.project_root, ".conditioned-publication.lock"):
                    publication = self._publish(root, candidate, evidence)
            except ConditionedDatasetError:
                state["status"] = "FAILED"
                state["stage"] = "publication_failed"
                state["message"] = "Publication failed closed; the one-time authorization was consumed."
                state["updated_at"] = _now()
                self._write_state_unlocked(root, state)
                raise
            except (OSError, ValueError, TypeError, KeyError) as exc:
                state["status"] = "FAILED"
                state["stage"] = "publication_failed"
                state["message"] = "Publication failed before project configuration was changed."
                state["updated_at"] = _now()
                self._write_state_unlocked(root, state)
                raise ConditionedDatasetError(
                    "publication_failed",
                    "Publication failed before project configuration was changed.",
                    status_code=500,
                ) from exc
            state["publication"] = publication
            state["status"] = "COMPLETE"
            state["stage"] = "published"
            state["message"] = (
                "Conditioned Dataset-v5 freeze and bound campaign are published; project activation remains separate."
            )
            state["updated_at"] = _now()
        return self.job(job_id)

    def run_training_infrastructure_audit(
        self,
        job_id: str,
        *,
        candidate_identity: str,
        publication_identity_sha256: str,
        activation_manifest_sha256: str,
        campaign_config_sha256: str,
        campaign_identity_sha256: str,
        expected_config_sha256: str,
        smoke_id: str,
        operation_nonce: str,
        explicit_action: bool,
    ) -> dict[str, Any]:
        """Audit the exact Phase-J prospective overlay without activating it."""

        if explicit_action is not True:
            raise ConditionedDatasetError(
                "training_audit_explicit_action_required",
                "The training-infrastructure audit requires an explicit action.",
                status_code=422,
            )
        if not KEY_PATTERN.fullmatch(str(smoke_id)) or not KEY_PATTERN.fullmatch(str(operation_nonce)):
            raise ConditionedDatasetError(
                "training_audit_operation",
                "A completed smoke-bundle ID and unique audit operation nonce are required.",
                status_code=422,
            )
        identities = (
            candidate_identity,
            publication_identity_sha256,
            activation_manifest_sha256,
            campaign_config_sha256,
            campaign_identity_sha256,
            expected_config_sha256,
        )
        if any(SHA256_PATTERN.fullmatch(str(value)) is None for value in identities):
            raise ConditionedDatasetError(
                "training_audit_identity",
                "Every selected training-audit identity must be an exact SHA-256.",
                status_code=422,
            )
        if any(os.environ.get(name) for name in ("SPRITELAB_CONFIG", "SPRITELAB_PROJECT_ROOT", "SPRITELAB_RUNS_DIR")):
            raise ConditionedDatasetError(
                "training_audit_config_override",
                "The training audit is unavailable while project configuration path overrides are active.",
            )

        root = self._job_root(job_id)
        state = self.job(job_id)
        if state.get("status") != "COMPLETE":
            raise ConditionedDatasetError(
                "publication_not_ready", "Publish the exact conditioned freeze before its training audit."
            )
        candidate = state.get("candidate")
        publication = state.get("publication")
        if not isinstance(candidate, Mapping) or candidate.get("candidate_identity") != candidate_identity:
            raise ConditionedDatasetError("training_audit_candidate_changed", "The selected candidate changed.")
        expected_publication = {
            "publication_identity_sha256": publication_identity_sha256,
            "activation_manifest_sha256": activation_manifest_sha256,
            "campaign_config_sha256": campaign_config_sha256,
            "campaign_identity_sha256": campaign_identity_sha256,
            "campaign_launch_ready": True,
            "campaign_seeds": [731001, 731002, 731003],
            "campaign_steps": 5_000,
            "configuration_activated": False,
            "training_started": False,
            "paths_exposed": False,
        }
        if not isinstance(publication, Mapping) or any(
            publication.get(key) != value for key, value in expected_publication.items()
        ):
            raise ConditionedDatasetError(
                "training_audit_publication_changed", "The selected publication or campaign changed."
            )
        activation_relative = _canonical_relative(str(publication.get("activation_manifest") or ""))
        campaign_relative = _canonical_relative(str(publication.get("campaign_config") or ""))
        activation_path = require_confined_path(
            self.project_root.joinpath(*PurePosixPath(activation_relative).parts), self.project_root
        )
        campaign_path = require_confined_path(
            self.project_root.joinpath(*PurePosixPath(campaign_relative).parts), self.project_root
        )
        if (
            _anchored_file_sha256(activation_path.parent, activation_path.name, self.project_root)
            != activation_manifest_sha256
        ):
            raise ConditionedDatasetError("training_audit_freeze_changed", "The published freeze bytes changed.")
        if _anchored_file_sha256(campaign_path.parent, campaign_path.name, self.project_root) != campaign_config_sha256:
            raise ConditionedDatasetError("training_audit_campaign_changed", "The campaign configuration changed.")

        config_path = require_confined_path(self.project_root / "spritelab.yaml", self.project_root)
        with (
            _ConditionedMutationLock(self.project_root, ".conditioned-config.lock"),
            AnchoredDirectory(self.project_root, self.project_root) as config_anchor,
        ):
            before = _read_anchored_regular_bytes(
                config_anchor,
                "spritelab.yaml",
                max_bytes=16 * 1024 * 1024,
            )
            before_sha256 = hashlib.sha256(before).hexdigest()
            if before_sha256 != expected_config_sha256:
                raise ConditionedDatasetError(
                    "training_audit_config_changed", "Project configuration changed before the audit."
                )
            try:
                current = _project_config_from_bytes(self.project_root, config_path, before)
            except ValueError as exc:
                raise ConditionedDatasetError(
                    "training_audit_config_invalid", "Project configuration is unavailable or invalid."
                ) from exc
            if current.root != self.project_root or current.path != config_path:
                raise ConditionedDatasetError(
                    "training_audit_config_scope", "The audit requires the canonical project configuration."
                )
            execution_policy = current.values.get("execution")
            if not isinstance(execution_policy, Mapping) or any(
                execution_policy.get(key) is not False for key in ("allow_dataset_production_freeze", "allow_training")
            ):
                raise ConditionedDatasetError(
                    "training_audit_requires_inactive_config",
                    "Phase I requires the still-inactive project configuration.",
                )

            values = deepcopy(current.values)
            view_relative = _canonical_relative(
                (PurePosixPath(activation_relative).parent / "view_manifest.json").as_posix()
            )
            values["dataset"]["view_manifest"] = view_relative
            values["dataset"]["freeze_manifest"] = activation_relative
            values["training"]["dataset_freeze"] = activation_relative
            values["training"]["campaign_config"] = campaign_relative
            values["execution"]["allow_dataset_production_freeze"] = True
            values["execution"]["allow_training"] = True
            prospective = ProjectConfig(self.project_root, config_path, values)

            runner = self._training_infrastructure_audit_runner
            if runner is None:
                from spritelab.product_features.training.audit import run_training_infrastructure_audit

                runner = run_training_infrastructure_audit
            try:
                execution = runner(
                    prospective,
                    operation_nonce=operation_nonce,
                    smoke_id=smoke_id,
                    source_job_id=job_id,
                )
            except ConditionedDatasetError:
                raise
            except ValueError as exc:
                code = str(getattr(exc, "code", "training_audit_failed"))
                message = str(
                    getattr(
                        exc,
                        "public_message",
                        "The training-infrastructure audit failed closed without authorizing training.",
                    )
                )
                raise ConditionedDatasetError(code, message) from exc
            except (OSError, TypeError, KeyError) as exc:
                raise ConditionedDatasetError(
                    "training_audit_failed",
                    "The training-infrastructure audit failed closed without authorizing training.",
                ) from exc

            after = _read_anchored_regular_bytes(
                config_anchor,
                "spritelab.yaml",
                max_bytes=16 * 1024 * 1024,
            )
            if after != before:
                raise ConditionedDatasetError(
                    "training_audit_config_mutated",
                    "The project configuration changed during the audit; its result is not applicable.",
                )

        verdict = getattr(execution, "verdict", None)
        verdict_value = getattr(verdict, "value", verdict)
        operation_identity = getattr(execution, "operation_identity", None)
        if (
            verdict_value not in {"PASS", "FAIL", "INCONCLUSIVE"}
            or not isinstance(operation_identity, str)
            or SHA256_PATTERN.fullmatch(operation_identity) is None
        ):
            raise ConditionedDatasetError(
                "training_audit_result", "The server-managed training audit returned an invalid result."
            )

        def relative_output(name: str) -> str:
            output = getattr(execution, name, None)
            if not isinstance(output, Path):
                raise ConditionedDatasetError(
                    "training_audit_result", "The server-managed training audit returned invalid artifacts."
                )
            return require_confined_path(output, self.project_root).relative_to(self.project_root).as_posix()

        from spritelab.product_features.training.audit import (
            TRAINING_AUDIT_ACTION_RECORD_SCHEMA,
            training_audit_action_record_path,
            training_audit_receipt_path,
        )

        expected_outputs = {
            "report_path": require_confined_path(
                self.project_root.joinpath(*PurePosixPath(str(values["training"]["audit_report"])).parts),
                self.project_root,
            ),
            "hashes_path": require_confined_path(
                self.project_root.joinpath(*PurePosixPath(str(values["training"]["audit_hashes"])).parts),
                self.project_root,
            ),
            "receipt_path": training_audit_receipt_path(prospective),
        }
        output_records: dict[str, dict[str, Any]] = {}
        for name, expected in expected_outputs.items():
            output = getattr(execution, name, None)
            if not isinstance(output, Path) or require_confined_path(output, self.project_root) != expected:
                raise ConditionedDatasetError(
                    "training_audit_result", "The server-managed training audit returned invalid artifacts."
                )
            with AnchoredDirectory(expected.parent, self.project_root) as output_anchor:
                content = _read_anchored_regular_bytes(output_anchor, expected.name, max_bytes=128 * 1024 * 1024)
            output_records[name] = {
                "path": expected.relative_to(self.project_root).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
                "content": content,
            }
        try:
            receipt_document = strict_json_loads(output_records["receipt_path"]["content"])
        except (RecursionError, TypeError, ValueError) as exc:
            raise ConditionedDatasetError(
                "training_audit_result", "The server-managed training audit returned an invalid receipt."
            ) from exc
        receipt_identity = receipt_document.get("receipt_identity") if isinstance(receipt_document, Mapping) else None
        if not isinstance(receipt_identity, str) or SHA256_PATTERN.fullmatch(receipt_identity) is None:
            raise ConditionedDatasetError(
                "training_audit_result", "The server-managed training audit returned an invalid receipt."
            )

        prospective_identity = stable_hash(values)
        record_payload = {
            "schema_version": TRAINING_AUDIT_ACTION_RECORD_SCHEMA,
            "source_job_id": job_id,
            "operation_identity": operation_identity,
            "prospective_configuration_identity_sha256": prospective_identity,
            "base_config_sha256": expected_config_sha256,
            "verdict": verdict_value,
            "report_sha256": output_records["report_path"]["sha256"],
            "hash_inventory_sha256": output_records["hashes_path"]["sha256"],
            "receipt_sha256": output_records["receipt_path"]["sha256"],
            "receipt_identity": receipt_identity,
            "config_unchanged": True,
            "configuration_activated": False,
            "training_started": False,
            "paths_exposed": False,
        }
        action_record = {**record_payload, "record_identity": stable_hash(record_payload)}
        action_record_path = training_audit_action_record_path(self.project_root, job_id, operation_identity)
        with (
            _ConditionedMutationLock(self.project_root, ".conditioned-config.lock"),
            AnchoredDirectory(self.project_root, self.project_root) as config_anchor,
        ):
            final_config = _read_anchored_regular_bytes(
                config_anchor,
                "spritelab.yaml",
                max_bytes=16 * 1024 * 1024,
            )
            if final_config != before:
                raise ConditionedDatasetError(
                    "training_audit_config_mutated",
                    "The project configuration changed before audit commitment; its result is not applicable.",
                )
            action_records_root = _ensure_anchored_child_directory(root, self.project_root, "training_audits")
            if action_records_root != action_record_path.parent:
                raise ConditionedDatasetError(
                    "training_audit_record_scope", "The server-managed audit record location changed."
                )
            with AnchoredDirectory(action_records_root, self.project_root) as record_anchor:
                if record_anchor.lexists(action_record_path.name):
                    raise ConditionedDatasetError(
                        "training_audit_record_exists", "That immutable server audit action already exists."
                    )
                _publish_or_reuse_immutable_file(
                    record_anchor,
                    action_record_path.name,
                    (strict_json_dumps(action_record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                    residue_prefix=".training-audit-record-residue-",
                )

        return {
            "schema_version": "spritelab.training.infrastructure-audit-action.v1",
            "job_id": job_id,
            "smoke_id": smoke_id,
            "operation_identity": operation_identity,
            "verdict": verdict_value,
            "prospective_configuration_identity_sha256": prospective_identity,
            "base_config_sha256": expected_config_sha256,
            "config_unchanged": True,
            "report_path": relative_output("report_path"),
            "hashes_path": relative_output("hashes_path"),
            "receipt_path": relative_output("receipt_path"),
            "action_record_path": action_record_path.relative_to(self.project_root).as_posix(),
            "action_record_identity": action_record["record_identity"],
            "configuration_activated": False,
            "training_started": False,
            "paths_exposed": False,
        }

    def activate(
        self,
        job_id: str,
        *,
        candidate_identity: str,
        publication_identity_sha256: str,
        activation_manifest_sha256: str,
        campaign_config_sha256: str,
        campaign_identity_sha256: str,
        expected_config_sha256: str,
        activation_authorization_id: str,
        explicit_action: bool,
        authorize_dataset_freeze: bool,
        authorize_training: bool,
    ) -> dict[str, Any]:
        """CAS-activate one audited freeze/campaign without starting training."""

        if explicit_action is not True or authorize_dataset_freeze is not True or authorize_training is not True:
            raise ConditionedDatasetError(
                "activation_authorization_required",
                "Activation requires explicit dataset-freeze and training authorization.",
                status_code=422,
            )
        if not KEY_PATTERN.fullmatch(str(activation_authorization_id)):
            raise ConditionedDatasetError(
                "activation_authorization_id", "A fresh activation authorization ID is required.", status_code=422
            )
        identities = (
            candidate_identity,
            publication_identity_sha256,
            activation_manifest_sha256,
            campaign_config_sha256,
            campaign_identity_sha256,
            expected_config_sha256,
        )
        if any(SHA256_PATTERN.fullmatch(str(value)) is None for value in identities):
            raise ConditionedDatasetError(
                "activation_identity", "Every selected activation identity must be an exact SHA-256.", status_code=422
            )
        if any(os.environ.get(name) for name in ("SPRITELAB_CONFIG", "SPRITELAB_PROJECT_ROOT", "SPRITELAB_RUNS_DIR")):
            raise ConditionedDatasetError(
                "activation_config_override",
                "Activation is unavailable while project configuration path overrides are active.",
            )

        try:
            with TrainingActionLock(self.project_root):
                return self._activate_crash_atomic(
                    job_id,
                    candidate_identity=candidate_identity,
                    publication_identity_sha256=publication_identity_sha256,
                    activation_manifest_sha256=activation_manifest_sha256,
                    campaign_config_sha256=campaign_config_sha256,
                    campaign_identity_sha256=campaign_identity_sha256,
                    expected_config_sha256=expected_config_sha256,
                    activation_authorization_id=activation_authorization_id,
                )
        except TrainingActionLockError as exc:
            raise ConditionedDatasetError(
                "activation_launch_conflict",
                "Another activation, Start, or Resume action owns the launch boundary.",
            ) from exc

    def _activate_crash_atomic(
        self,
        job_id: str,
        *,
        candidate_identity: str,
        publication_identity_sha256: str,
        activation_manifest_sha256: str,
        campaign_config_sha256: str,
        campaign_identity_sha256: str,
        expected_config_sha256: str,
        activation_authorization_id: str,
    ) -> dict[str, Any]:
        """Prepare every durable commit byte, then make the config CAS last."""

        root = self._job_root(job_id)
        request_bindings = {
            "candidate_identity": candidate_identity,
            "publication_identity_sha256": publication_identity_sha256,
            "activation_manifest_sha256": activation_manifest_sha256,
            "campaign_config_sha256": campaign_config_sha256,
            "campaign_identity_sha256": campaign_identity_sha256,
        }
        authorization_digest = hashlib.sha256(activation_authorization_id.encode("utf-8")).hexdigest()
        with self._lock, _ConditionedMutationLock(root, ".conditioned-state.lock"):
            state = self._read_state(root)
            candidate = state.get("candidate")
            publication = state.get("publication")
            if state.get("status") != "COMPLETE":
                raise ConditionedDatasetError("publication_not_ready", "Publish the exact freeze before activation.")
            if not isinstance(candidate, Mapping) or candidate.get("candidate_identity") != candidate_identity:
                raise ConditionedDatasetError(
                    "activation_candidate_changed", "The selected candidate identity changed."
                )
            expected_publication = {
                "publication_identity_sha256": publication_identity_sha256,
                "activation_manifest_sha256": activation_manifest_sha256,
                "campaign_config_sha256": campaign_config_sha256,
                "campaign_identity_sha256": campaign_identity_sha256,
                "campaign_launch_ready": True,
                "campaign_seeds": [731001, 731002, 731003],
                "campaign_steps": 5_000,
                "configuration_activated": False,
                "training_started": False,
                "paths_exposed": False,
            }
            if not isinstance(publication, Mapping) or any(
                publication.get(key) != value for key, value in expected_publication.items()
            ):
                projected = self._project_activation_state(root, state)
                projected_publication = projected.get("publication")
                if not isinstance(projected_publication, Mapping) or any(
                    projected_publication.get(key) != value
                    for key, value in {**expected_publication, "configuration_activated": True}.items()
                ):
                    raise ConditionedDatasetError(
                        "activation_publication_changed", "The selected publication or campaign identity changed."
                    )
                return projected

            activation_relative = _canonical_relative(str(publication.get("activation_manifest") or ""))
            campaign_relative = _canonical_relative(str(publication.get("campaign_config") or ""))
            activation_path = require_confined_path(
                self.project_root.joinpath(*PurePosixPath(activation_relative).parts), self.project_root
            )
            campaign_path = require_confined_path(
                self.project_root.joinpath(*PurePosixPath(campaign_relative).parts), self.project_root
            )
            if (
                _anchored_file_sha256(activation_path.parent, activation_path.name, self.project_root)
                != activation_manifest_sha256
            ):
                raise ConditionedDatasetError("activation_freeze_changed", "The published freeze bytes changed.")
            if (
                _anchored_file_sha256(campaign_path.parent, campaign_path.name, self.project_root)
                != campaign_config_sha256
            ):
                raise ConditionedDatasetError("activation_campaign_changed", "The campaign configuration changed.")

            config_path = require_confined_path(self.project_root / "spritelab.yaml", self.project_root)
            with (
                _ConditionedMutationLock(self.project_root, ".conditioned-config.lock"),
                AnchoredDirectory(self.project_root, self.project_root) as config_anchor,
                _held_regular_file_snapshot(
                    config_anchor,
                    "spritelab.yaml",
                    max_bytes=16 * 1024 * 1024,
                ) as held_config,
            ):
                before = held_config.content
                current_sha256 = hashlib.sha256(before).hexdigest()
                receipt_root = require_confined_path(root / "activation_receipt", root)
                if os.path.lexists(receipt_root):
                    summary = self._load_activation_commit_summary(root, current_sha256=current_sha256)
                    self._require_matching_activation_summary(
                        summary,
                        request_bindings=request_bindings,
                        authorization_digest=authorization_digest,
                        expected_config_sha256=expected_config_sha256,
                    )
                    if summary["committed"] is True:
                        return self._project_activation_state(root, state)
                    prepared_state = self._prepared_activation_state(
                        state,
                        publication,
                        summary,
                    )
                    if state != prepared_state:
                        self._write_state_unlocked(root, prepared_state)
                        state = prepared_state
                    config_after_sha256 = str(summary["config_after_sha256"])
                    payload = self._prospective_activation_payload(
                        config_path,
                        before,
                        activation_relative=activation_relative,
                        campaign_relative=campaign_relative,
                        campaign_identity_sha256=campaign_identity_sha256,
                        activation_manifest_sha256=activation_manifest_sha256,
                        campaign_config_sha256=campaign_config_sha256,
                    )
                    if hashlib.sha256(payload).hexdigest() != config_after_sha256:
                        raise ConditionedDatasetError(
                            "activation_recovery_changed",
                            "The exact PREPARED activation no longer produces its intended configuration.",
                        )
                else:
                    if state.get("activation_authorization") is not None:
                        raise ConditionedDatasetError(
                            "activation_authorization_consumed",
                            "This job already consumed its activation authorization.",
                        )
                    if current_sha256 != expected_config_sha256:
                        raise ConditionedDatasetError(
                            "activation_config_changed", "Project configuration changed before activation."
                        )
                    payload = self._prospective_activation_payload(
                        config_path,
                        before,
                        activation_relative=activation_relative,
                        campaign_relative=campaign_relative,
                        campaign_identity_sha256=campaign_identity_sha256,
                        activation_manifest_sha256=activation_manifest_sha256,
                        campaign_config_sha256=campaign_config_sha256,
                    )
                    config_after_sha256 = hashlib.sha256(payload).hexdigest()
                    if config_after_sha256 == current_sha256:
                        raise ConditionedDatasetError(
                            "activation_config_unchanged", "Activation must change the blocked project configuration."
                        )
                    operation_id = f"activation-{uuid.uuid4().hex}"
                    prepared_at = _now()
                    receipt, journal, record = build_activation_commit_documents(
                        job_id=job_id,
                        operation_id=operation_id,
                        **request_bindings,
                        authorization_id_sha256=authorization_digest,
                        config_before_sha256=current_sha256,
                        config_after_sha256=config_after_sha256,
                        prepared_at=prepared_at,
                    )
                    summary = self._publish_activation_commit_documents(root, receipt, journal, record)
                    prepared_state = self._prepared_activation_state(state, publication, summary)
                    self._write_state_unlocked(root, prepared_state)
                    state = prepared_state

                _verify_held_regular_file_snapshot(config_anchor, "spritelab.yaml", held_config)
                if current_sha256 == str(summary["config_before_sha256"]):
                    _replace_held_config_if_supported(
                        config_anchor,
                        held_config,
                        payload,
                        expected_sha256=config_after_sha256,
                    )
                elif current_sha256 != config_after_sha256:
                    raise ConditionedDatasetError(
                        "activation_config_changed",
                        "Project configuration differs from both activation commit boundaries.",
                    )

                receipt, journal, record = self._load_activation_commit_documents(root)
                try:
                    marker = build_activation_project_commit(
                        receipt=receipt,
                        journal=journal,
                        record=record,
                        config_after_bytes=payload,
                    )
                except ActivationCommitError as exc:
                    raise ConditionedDatasetError(
                        "activation_commit_invalid",
                        "The immutable project activation marker could not be built.",
                    ) from exc
                marker_bytes = canonical_activation_commit_bytes(marker)
                _publish_or_reuse_immutable_file(
                    config_anchor,
                    ACTIVATION_PROJECT_COMMIT_NAME,
                    marker_bytes,
                    residue_prefix=".activation-project-marker-residue-",
                )
                marker_summary = self._load_activation_project_commit_summary(
                    root,
                    current_sha256=(config_after_sha256 if held_config.replaced else current_sha256),
                )
                if marker_summary.get("committed") is not True:
                    raise ConditionedDatasetError(
                        "activation_commit_invalid",
                        "The immutable project activation marker did not commit exact bytes.",
                    )
        return self.job(job_id)

    def _prospective_activation_payload(
        self,
        config_path: Path,
        before: bytes,
        *,
        activation_relative: str,
        campaign_relative: str,
        campaign_identity_sha256: str,
        activation_manifest_sha256: str,
        campaign_config_sha256: str,
    ) -> bytes:
        try:
            current = _project_config_from_bytes(self.project_root, config_path, before)
        except ValueError as exc:
            raise ConditionedDatasetError(
                "activation_config_invalid", "Project configuration is unavailable or invalid."
            ) from exc
        if current.root != self.project_root or current.path != config_path:
            raise ConditionedDatasetError(
                "activation_config_scope", "Activation requires the canonical project configuration."
            )
        values = deepcopy(current.values)
        view_relative = _canonical_relative(
            (PurePosixPath(activation_relative).parent / "view_manifest.json").as_posix()
        )
        values["dataset"]["view_manifest"] = view_relative
        values["dataset"]["freeze_manifest"] = activation_relative
        values["training"]["dataset_freeze"] = activation_relative
        values["training"]["campaign_config"] = campaign_relative
        values["execution"]["allow_dataset_production_freeze"] = True
        values["execution"]["allow_training"] = True
        prospective = ProjectConfig(self.project_root, config_path, values)
        activation_loader = self._activation_loader
        if activation_loader is None:
            from spritelab.product_features.training.activation import load_conditioned_training_activation

            activation_loader = load_conditioned_training_activation
        try:
            selected = activation_loader(
                prospective,
                expected_campaign={"campaign_identity": campaign_identity_sha256},
                require_audit=True,
                require_activation_commit=False,
            )
        except ValueError as exc:
            raise ConditionedDatasetError(
                "activation_audit_blocked",
                "The exact prospective freeze and campaign lack an applicable PASS training audit.",
            ) from exc
        selected_training = dict(selected.campaign.get("training") or {})
        audit_status = getattr(getattr(selected, "audit_status", None), "value", None)
        audit_ready = audit_status == "PASS" if audit_status is not None else bool(getattr(selected, "ready", False))
        if (
            not audit_ready
            or selected.freeze_sha256 != activation_manifest_sha256
            or selected.campaign_config_sha256 != campaign_config_sha256
            or selected.campaign.get("campaign_identity") != campaign_identity_sha256
            or selected.campaign.get("seeds") != [731001, 731002, 731003]
            or selected_training.get("max_optimizer_steps") != 5_000
        ):
            raise ConditionedDatasetError(
                "activation_contract_changed", "The prospective training activation contract changed."
            )
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True).encode("utf-8")

    def _publish_activation_commit_documents(
        self,
        root: Path,
        receipt: Mapping[str, Any],
        journal: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        final = require_confined_path(root / "activation_receipt", root)
        with open_anchored_directory(root, self.project_root) as parent:
            if parent.lexists(final.name):
                raise ConditionedDatasetError(
                    "activation_receipt_exists", "An activation PREPARED record already exists for this job."
                )
            receipt_identity = parent.mkdir(final.name, exist_ok=False)
            with parent.open_directory_immovable(final.name) as anchor:
                if not receipt_identity.matches(anchor.directory_metadata()):
                    raise ConditionedDatasetError(
                        "activation_receipt_staging", "Activation PREPARED directory identity changed."
                    )
                for name, value in (
                    ("receipt.json", receipt),
                    ("record.json", record),
                    ("journal.json", journal),
                ):
                    _publish_or_reuse_immutable_file(
                        anchor,
                        name,
                        canonical_activation_commit_bytes(value),
                        residue_prefix=".activation-receipt-residue-",
                    )
                expected_inventory = _inventory_from_anchor(anchor)
                if set(expected_inventory) != {"journal.json", "receipt.json", "record.json"}:
                    raise ConditionedDatasetError(
                        "activation_receipt_inventory", "Activation PREPARED evidence has an unexpected inventory."
                    )
        if _inventory(final, self.project_root) != expected_inventory:
            raise ConditionedDatasetError(
                "activation_receipt_changed", "Activation PREPARED evidence changed during publication."
            )
        return self._load_activation_commit_summary(
            root,
            current_sha256=str(receipt["config_before_sha256"]),
        )

    def _load_activation_commit_summary(self, root: Path, *, current_sha256: str) -> dict[str, Any]:
        receipt, journal, record = self._load_activation_commit_documents(root)
        try:
            prepared = validate_activation_commit_documents(
                receipt=receipt,
                journal=journal,
                record=record,
                expected_job_id=root.name,
                current_config_sha256=current_sha256,
                require_committed=False,
            )
        except ActivationCommitError as exc:
            raise ConditionedDatasetError(
                "activation_commit_invalid", "Activation PREPARED evidence is invalid or conflicts with config."
            ) from exc
        with AnchoredDirectory(self.project_root, self.project_root) as project_anchor:
            marker_exists = project_anchor.lexists(ACTIVATION_PROJECT_COMMIT_NAME)
        if marker_exists:
            return self._load_activation_project_commit_summary(root, current_sha256=current_sha256)
        return {**prepared, "committed": False, "reconciliation_required": False}

    def _load_activation_commit_documents(
        self,
        root: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        directory = require_confined_path(root / "activation_receipt", root)
        if not _safe_directory(directory) or set(_inventory(directory, self.project_root)) != {
            "journal.json",
            "receipt.json",
            "record.json",
        }:
            raise ConditionedDatasetError(
                "activation_commit_invalid", "Activation PREPARED evidence is unavailable or unsafe."
            )
        receipt = _read_json_mapping(directory / "receipt.json", max_bytes=1024 * 1024)
        journal = _read_json_mapping(directory / "journal.json", max_bytes=1024 * 1024)
        record = _read_json_mapping(directory / "record.json", max_bytes=1024 * 1024)
        return receipt, journal, record

    def _load_activation_project_commit_summary(
        self,
        root: Path,
        *,
        current_sha256: str,
    ) -> dict[str, Any]:
        receipt, journal, record = self._load_activation_commit_documents(root)
        with AnchoredDirectory(self.project_root, self.project_root) as project_anchor:
            try:
                marker_bytes = _read_anchored_regular_bytes(
                    project_anchor,
                    ACTIVATION_PROJECT_COMMIT_NAME,
                    max_bytes=32 * 1024 * 1024,
                )
                marker_raw = strict_json_loads(marker_bytes)
            except (OSError, ValueError) as exc:
                raise ConditionedDatasetError(
                    "activation_commit_invalid",
                    "The immutable project activation marker is unavailable or invalid.",
                ) from exc
        if not isinstance(marker_raw, Mapping):
            raise ConditionedDatasetError(
                "activation_commit_invalid",
                "The immutable project activation marker must be an object.",
            )
        marker = dict(marker_raw)
        if canonical_activation_commit_bytes(marker) != marker_bytes:
            raise ConditionedDatasetError(
                "activation_commit_invalid",
                "The immutable project activation marker is not canonical.",
            )
        try:
            summary, config_after = validate_activation_project_commit(
                marker,
                receipt=receipt,
                journal=journal,
                record=record,
                current_config_sha256=current_sha256,
                expected_job_id=root.name,
            )
        except ActivationCommitError as exc:
            raise ConditionedDatasetError(
                "activation_commit_invalid",
                "The immutable project activation marker is invalid or conflicts with config.",
            ) from exc
        if hashlib.sha256(config_after).hexdigest() != summary["config_after_sha256"]:
            raise ConditionedDatasetError(
                "activation_commit_invalid",
                "The immutable project activation marker configuration changed.",
            )
        return summary

    def _require_matching_activation_summary(
        self,
        summary: Mapping[str, Any],
        *,
        request_bindings: Mapping[str, str],
        authorization_digest: str,
        expected_config_sha256: str,
    ) -> None:
        directory = self._job_root(str(summary["job_id"])) / "activation_receipt"
        receipt = _read_json_mapping(directory / "receipt.json", max_bytes=1024 * 1024)
        if (
            any(summary.get(name) != value for name, value in request_bindings.items())
            or receipt.get("authorization_id_sha256") != authorization_digest
        ):
            raise ConditionedDatasetError(
                "activation_recovery_mismatch", "Only the exact PREPARED activation may be recovered."
            )
        if summary.get("config_before_sha256") != expected_config_sha256:
            raise ConditionedDatasetError(
                "activation_recovery_config", "Recovery requires the original blocked configuration identity."
            )

    def _prepared_activation_state(
        self,
        state: Mapping[str, Any],
        publication: Mapping[str, Any],
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        prepared = deepcopy(dict(state))
        prepared["activation_authorization"] = {
            "status": "PREPARED",
            "operation_id": summary["operation_id"],
            "receipt_identity": summary["receipt_identity"],
            "record_identity": summary["record_identity"],
            "journal_identity": summary["journal_identity"],
            "receipt_relative_path": "activation_receipt/receipt.json",
            "record_relative_path": "activation_receipt/record.json",
            "journal_relative_path": "activation_receipt/journal.json",
            "config_before_sha256": summary["config_before_sha256"],
            "config_after_sha256": summary["config_after_sha256"],
            "action_lock_protocol_identity": ACTION_LOCK_PROTOCOL_IDENTITY,
            "one_time": True,
        }
        prepared["publication"] = dict(publication)
        prepared["stage"] = "activation_prepared"
        prepared["message"] = "Activation PREPARED durably; the immutable project commit marker remains pending."
        prepared["updated_at"] = _now()
        return prepared

    def _project_activation_state(self, root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
        publication = state.get("publication")
        if not isinstance(publication, Mapping) or publication.get("configuration_activated") is True:
            return dict(state)
        receipt_root = root / "activation_receipt"
        if not os.path.lexists(receipt_root):
            return dict(state)
        current_sha256 = self._config_sha256()
        if current_sha256 is None:
            raise ConditionedDatasetError(
                "activation_commit_config", "The canonical project configuration is unavailable."
            )
        summary = self._load_activation_commit_summary(root, current_sha256=current_sha256)
        projected = deepcopy(dict(state))
        if summary["committed"] is not True:
            projected["stage"] = "activation_prepared"
            projected["message"] = "Activation PREPARED; project configuration remains blocked."
            return projected
        publication_value = dict(publication)
        publication_value["configuration_activated"] = True
        projected["publication"] = publication_value
        authorization = dict(projected.get("activation_authorization") or {})
        authorization.update(
            {
                "status": "COMMITTED",
                "operation_id": summary["operation_id"],
                "receipt_identity": summary["receipt_identity"],
                "record_identity": summary["record_identity"],
                "journal_identity": summary["journal_identity"],
                "receipt_relative_path": "activation_receipt/receipt.json",
                "record_relative_path": "activation_receipt/record.json",
                "journal_relative_path": "activation_receipt/journal.json",
                "config_before_sha256": summary["config_before_sha256"],
                "config_after_sha256": summary["config_after_sha256"],
                "action_lock_protocol_identity": ACTION_LOCK_PROTOCOL_IDENTITY,
                "one_time": True,
            }
        )
        projected["activation_authorization"] = authorization
        projected["stage"] = "activated"
        projected["message"] = "Conditioned freeze and audited campaign activated; training was not started."
        return projected

    def _handoff_inventory(self) -> list[dict[str, Any]]:
        if not self.harvest_root.is_dir() or _is_link_or_reparse(self.harvest_root):
            return []
        results: list[dict[str, Any]] = []
        for entry in sorted(self.harvest_root.iterdir(), key=lambda path: path.name):
            if not RUN_ID_PATTERN.fullmatch(entry.name) or not _safe_directory(entry):
                continue
            handoff_path = entry / "handoff.json"
            try:
                handoff = _read_json_mapping(handoff_path, max_bytes=16 * 1024 * 1024)
            except ConditionedDatasetError:
                continue
            if handoff.get("schema_version") != HANDOFF_SCHEMA or handoff.get("paths_exposed") is not False:
                continue
            source = handoff.get("source") if isinstance(handoff.get("source"), Mapping) else {}
            results.append(
                {
                    "run_id": entry.name,
                    "source_id": str(source.get("source_id") or handoff.get("source_id") or ""),
                    "source_title": str(source.get("title") or handoff.get("source_title") or ""),
                    "artifact_count": _integer(handoff.get("artifact_count"), default=0),
                    "total_bytes": _integer(handoff.get("total_bytes"), default=0),
                    "status": "COMPLETE",
                    "paths_exposed": False,
                }
            )
        return results

    def _job_inventory(self) -> list[dict[str, Any]]:
        if not self.jobs_root.is_dir() or _is_link_or_reparse(self.jobs_root):
            return []
        results: list[dict[str, Any]] = []
        for entry in sorted(self.jobs_root.iterdir(), key=lambda path: path.name, reverse=True):
            if not JOB_ID_PATTERN.fullmatch(entry.name) or not _safe_directory(entry):
                continue
            try:
                state = self.job(entry.name)
            except ConditionedDatasetError:
                continue
            results.append(state)
        return results

    def _verified_source(self, dataset_reference: str) -> dict[str, Any]:
        """Load only a published, revalidated DatasetIntake-backed import."""

        try:
            if self._managed_intake_loader is not None:
                return dict(self._managed_intake_loader(dataset_reference))
            return load_managed_intake(self.project_root, dataset_reference)
        except ConditionedIntakeError as exc:
            raise ConditionedDatasetError(
                "managed_intake_invalid",
                "The selected managed Dataset import or one of its immutable bindings is unavailable.",
            ) from exc

    def _legacy_verified_harvest_source(self, run_id: str) -> dict[str, Any]:
        """Deprecated verifier retained only for source-contract compatibility tests."""

        raise ConditionedDatasetError(
            "direct_harvest_handoff_forbidden",
            "Conditioned Dataset-v5 accepts only transactional managed Dataset imports.",
        )

        _validate_run_id(run_id)
        try:
            run_root = require_confined_path(self.harvest_root / run_id, self.harvest_root)
        except UnsafeFilesystemOperation as exc:
            raise ConditionedDatasetError(
                "harvest_handoff_unsafe", "The selected Harvest handoff is outside managed storage."
            ) from exc
        if not _safe_directory(run_root):
            raise ConditionedDatasetError(
                "harvest_handoff_missing", "The selected completed Harvest handoff is unavailable.", status_code=404
            )
        handoff_path = run_root / "handoff.json"
        disk_handoff = _read_json_mapping(handoff_path, max_bytes=16 * 1024 * 1024)
        try:
            loaded = dict(self._load_handoff(run_id))
            loaded.pop("dataset_import_available", None)
        except ConditionedDatasetError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ConditionedDatasetError(
                "harvest_handoff_invalid", "The trusted Harvest handoff could not be verified."
            ) from exc
        if loaded != disk_handoff:
            raise ConditionedDatasetError(
                "harvest_handoff_changed", "The durable Harvest handoff disagrees with its trusted service projection."
            )
        handoff = disk_handoff
        if handoff.get("schema_version") != HANDOFF_SCHEMA:
            raise ConditionedDatasetError(
                "harvest_handoff_schema", "Only trusted Harvest dataset handoff v2 records can be conditioned."
            )
        if handoff.get("paths_exposed") is not False:
            raise ConditionedDatasetError(
                "harvest_handoff_privacy", "The Harvest handoff does not satisfy the private-path contract."
            )
        if handoff.get("handoff_ready") is not True or handoff.get("portable_relative_paths") is not True:
            raise ConditionedDatasetError(
                "harvest_handoff_incomplete", "The Harvest handoff is not ready with portable artifact paths."
            )
        if handoff.get("status", "COMPLETE") not in {"COMPLETE", "complete"}:
            raise ConditionedDatasetError("harvest_handoff_incomplete", "The selected Harvest run is not complete.")
        managed = handoff.get("managed_reference")
        if (
            handoff.get("run_id") != run_id
            or not isinstance(managed, Mapping)
            or managed.get("kind") != "harvest_run"
            or managed.get("run_id") != run_id
        ):
            raise ConditionedDatasetError(
                "harvest_handoff_reference", "The Harvest handoff is not bound to its managed run."
            )

        source = handoff.get("source") if isinstance(handoff.get("source"), Mapping) else {}
        source_id = str(source.get("source_id") or handoff.get("source_id") or "")
        source_title = str(source.get("title") or handoff.get("source_title") or source_id)
        creator = str(source.get("creator") or handoff.get("creator") or "")
        license_data = source.get("license") if isinstance(source.get("license"), Mapping) else handoff.get("license")
        license_mapping = dict(license_data) if isinstance(license_data, Mapping) else {}
        license_id = str(license_mapping.get("identifier") or handoff.get("license_id") or "").casefold()
        if license_id not in ALLOWED_LICENSES or license_mapping.get("permissive_policy", True) is not True:
            raise ConditionedDatasetError(
                "harvest_license_not_allowed",
                "Conditioning initially accepts only verified CC0-1.0 or public-domain sources.",
            )
        if not source_id or not source_title.strip() or not creator.strip():
            raise ConditionedDatasetError(
                "harvest_provenance_incomplete",
                "The Harvest handoff lacks explicit source, title, or creator provenance.",
            )
        if handoff.get("source_id") != source_id or handoff.get("license") != license_mapping:
            raise ConditionedDatasetError(
                "harvest_provenance_binding", "The Harvest handoff source or license binding is inconsistent."
            )
        identity_fields = (
            "source_evidence_binding_identity",
            "backend_capability_identity",
            "limits_identity",
            "acquisition_receipt_identity",
            "artifact_manifest_identity",
            "artifact_set_identity",
            "provenance_identity",
        )
        if any(not SHA256_PATTERN.fullmatch(str(handoff.get(name) or "")) for name in identity_fields):
            raise ConditionedDatasetError(
                "harvest_identity_binding", "The Harvest handoff contains an invalid identity binding."
            )
        expected_provenance = stable_hash(
            {
                "source": source,
                "acquisition_receipt_identity": handoff["acquisition_receipt_identity"],
            }
        )
        if handoff.get("provenance_identity") != expected_provenance:
            raise ConditionedDatasetError(
                "harvest_provenance_identity", "The Harvest provenance identity is inconsistent."
            )
        evidence_binding = source.get("evidence_binding")
        if not isinstance(evidence_binding, Mapping) or evidence_binding.get("binding_identity") != handoff.get(
            "source_evidence_binding_identity"
        ):
            raise ConditionedDatasetError(
                "harvest_source_evidence_binding", "The Harvest source-evidence identity is inconsistent."
            )
        if not str(license_mapping.get("evidence_url") or license_mapping.get("evidence_url_sha256") or ""):
            raise ConditionedDatasetError(
                "harvest_license_evidence", "The Harvest handoff lacks bound license evidence."
            )

        manifest_path = run_root / "artifact_manifest.json"
        manifest_bytes = _read_regular_bytes(manifest_path, run_root, max_bytes=64 * 1024 * 1024)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            manifest_value = strict_json_loads(manifest_bytes)
        except ValueError as exc:
            raise ConditionedDatasetError(
                "harvest_artifact_manifest", "The Harvest artifact manifest is invalid."
            ) from exc
        if not isinstance(manifest_value, Mapping):
            raise ConditionedDatasetError(
                "harvest_artifact_manifest", "The Harvest artifact manifest must be an object."
            )
        manifest = dict(manifest_value)
        if manifest.get("schema_version") != "spritelab.harvest.artifact-manifest.v1":
            raise ConditionedDatasetError(
                "harvest_artifact_manifest_schema", "The Harvest artifact manifest schema is unsupported."
            )
        if handoff.get("artifact_manifest_identity") != stable_hash(manifest):
            raise ConditionedDatasetError(
                "harvest_artifact_manifest_identity", "The Harvest artifact manifest changed after handoff."
            )
        if handoff.get("artifact_set_identity") != manifest.get("artifact_set_identity"):
            raise ConditionedDatasetError(
                "harvest_artifact_identity", "The Harvest artifact-set binding is inconsistent."
            )
        if handoff.get("files") != manifest.get("files"):
            raise ConditionedDatasetError(
                "harvest_artifact_file_binding", "The Harvest handoff file inventory disagrees with its manifest."
            )
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) > self.policy.max_source_files:
            raise ConditionedDatasetError(
                "harvest_artifact_manifest_files", "The Harvest artifact file inventory is invalid or oversized."
            )
        expected: list[AcquiredFile] = []
        for raw in files:
            if not isinstance(raw, Mapping):
                raise ConditionedDatasetError(
                    "harvest_artifact_record", "The Harvest artifact inventory contains an invalid record."
                )
            expected_sha = str(raw.get("actual_sha256") or raw.get("sha256") or "")
            if raw.get("expected_sha256") not in {None, "", expected_sha}:
                raise ConditionedDatasetError(
                    "harvest_artifact_hash_disagreement",
                    "A Harvest artifact has disagreeing expected and actual identities.",
                )
            try:
                expected.append(
                    AcquiredFile(
                        relative_path=str(raw.get("relative_path") or ""),
                        byte_count=int(raw.get("byte_count")),
                        sha256=expected_sha,
                        mime_type=str(raw.get("mime_type") or ""),
                        usable=raw.get("usable") is True,
                        quarantine_reason=str(raw.get("quarantine_reason"))
                        if raw.get("quarantine_reason") is not None
                        else None,
                        taxonomy=tuple(str(value) for value in raw.get("taxonomy") or ()),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ConditionedDatasetError(
                    "harvest_artifact_record", "A Harvest artifact record is malformed."
                ) from exc
        artifacts = require_confined_path(run_root / "artifacts", run_root)
        limits = HarvestLimits(
            max_files=self.policy.max_source_files,
            max_file_bytes=self.policy.max_file_bytes,
            max_total_bytes=self.policy.max_source_bytes,
            max_response_bytes=self.policy.max_source_bytes,
            max_depth=self.policy.max_depth,
            max_archive_uncompressed_bytes=max(self.policy.max_source_bytes, 1024 * 1024 * 1024),
        )
        try:
            verified_manifest = scan_artifacts(artifacts, limits, expected_files=tuple(expected))
        except (OSError, ValueError) as exc:
            raise ConditionedDatasetError(
                "harvest_artifacts_changed", "Harvest artifacts failed complete per-file re-verification."
            ) from exc
        if verified_manifest != manifest:
            raise ConditionedDatasetError(
                "harvest_artifacts_changed", "Harvest artifacts no longer reproduce the exact handoff manifest."
            )
        if verified_manifest.get("artifact_set_identity") != manifest.get("artifact_set_identity"):
            raise ConditionedDatasetError(
                "harvest_artifact_identity", "Harvest artifact-set identity no longer matches the handoff."
            )
        if int(verified_manifest.get("artifact_count", -1)) != int(handoff.get("artifact_count", -2)):
            raise ConditionedDatasetError("harvest_artifact_count", "Harvest handoff and artifact counts disagree.")
        if int(verified_manifest.get("total_bytes", -1)) != int(handoff.get("total_bytes", -2)):
            raise ConditionedDatasetError(
                "harvest_artifact_bytes", "Harvest handoff and artifact byte totals disagree."
            )
        return {
            "run_id": run_id,
            "run_root": run_root,
            "artifacts_root": artifacts,
            "handoff": handoff,
            "handoff_identity": hashlib.sha256(_canonical_bytes(handoff)).hexdigest(),
            "artifact_manifest": verified_manifest,
            "artifact_manifest_sha256": manifest_sha256,
            "source_id": source_id,
            "source_title": source_title,
            "creator": creator,
            "license_id": license_id,
            "license_evidence": license_mapping,
            "artifact_count": int(verified_manifest["artifact_count"]),
        }

    def _load_handoff(self, run_id: str) -> Mapping[str, Any]:
        if self._handoff_loader is not None:
            return self._handoff_loader(run_id)
        try:
            from spritelab.product_features.harvest.service import HarvestService
        except ImportError as exc:
            raise ConditionedDatasetError(
                "harvest_service_unavailable", "The trusted Harvest handoff service is unavailable."
            ) from exc
        return HarvestService(self.project_root).handoff(run_id)

    def _inspect_records(self, source: Mapping[str, Any]) -> tuple[list[_SourceRecord], list[str]]:
        records: list[_SourceRecord] = []
        exclusions: list[str] = []
        files = source["artifact_manifest"]["files"]
        accepted_relative_paths = set(source["accepted_relative_paths"])
        covered_source_paths = set(source.get("covered_source_relative_paths") or accepted_relative_paths)
        source_root = Path(source["artifacts_root"])
        raw_derived_records = source.get("derived_sheet_records", [])
        if not isinstance(raw_derived_records, list):
            raise ConditionedDatasetError(
                "derived_manifest_invalid", "Managed derived-frame records are unavailable or malformed."
            )
        with AnchoredDirectory(source_root, source_root) as source_anchor:
            for raw in files:
                relative = _canonical_relative(str(raw.get("relative_path") or ""))
                if relative not in accepted_relative_paths:
                    if relative not in covered_source_paths:
                        exclusions.append("dataset_intake_excluded")
                    continue
                if raw.get("usable") is not True:
                    exclusions.append("harvest_quarantine")
                    continue
                if raw.get("mime_type") != "image/png":
                    exclusions.append("not_png")
                    continue
                try:
                    content = _read_anchored_relative_bytes(
                        source_anchor,
                        relative,
                        max_bytes=self.policy.max_file_bytes,
                    )
                except ConditionedDatasetError:
                    exclusions.append("unsafe_or_unreadable")
                    continue
                digest = hashlib.sha256(content).hexdigest()
                expected = str(raw.get("actual_sha256") or "")
                if digest != expected or len(content) != int(raw.get("byte_count", -1)):
                    raise ConditionedDatasetError(
                        "harvest_artifact_changed", "A Harvest PNG changed during candidate inspection."
                    )
                path = source_root.joinpath(*PurePosixPath(relative).parts)
                record, exclusion = _source_record_from_png(
                    relative=relative,
                    path=path,
                    content=content,
                    source=source,
                )
                if record is None:
                    exclusions.append(str(exclusion))
                else:
                    records.append(record)

            if raw_derived_records:
                derived_root = Path(source["derived_root"])
                with AnchoredDirectory(derived_root, derived_root) as derived_anchor:
                    for raw in raw_derived_records:
                        if not isinstance(raw, Mapping):
                            raise ConditionedDatasetError(
                                "derived_manifest_invalid", "A managed derived-frame record is malformed."
                            )
                        relative = _canonical_relative(str(raw.get("semantic_relative_path") or ""))
                        output_relative = _canonical_relative(str(raw.get("output_relative_path") or ""))
                        try:
                            content = read_receipt_bound_derived_frame(
                                source_anchor=source_anchor,
                                derived_anchor=derived_anchor,
                                record=raw,
                                max_bytes=self.policy.max_file_bytes,
                            )
                        except ConditionedIntakeError as exc:
                            raise ConditionedDatasetError(
                                "derived_frame_changed",
                                "A receipt-bound derived frame or its exact parent changed during inspection.",
                            ) from exc
                        path = derived_root.joinpath(*PurePosixPath(output_relative).parts)
                        record, exclusion = _source_record_from_png(
                            relative=relative,
                            path=path,
                            content=content,
                            source=source,
                            source_group_identity=str(raw.get("source_group_identity") or ""),
                            derivation=dict(raw),
                        )
                        if record is None:
                            exclusions.append(str(exclusion))
                        else:
                            records.append(record)
        return records, exclusions

    def _run_build(self, job_id: str) -> None:
        root = self._job_root(job_id)
        try:
            self._check_cancel(job_id)
            with self._lock, _ConditionedMutationLock(root, ".conditioned-state.lock"):
                state = self._read_state(root)
            sources = [self._verified_source(value) for value in state["dataset_references"]]
            records: list[_SourceRecord] = []
            exclusions: list[str] = []
            for source in sources:
                source_records, source_exclusions = self._inspect_records(source)
                records.extend(source_records)
                exclusions.extend(source_exclusions)
            records, duplicate_exclusions, near_duplicate_exclusions = _deduplicate_records(records)
            exclusions.extend(duplicate_exclusions)
            selected = _representative_selection(records, self.policy.target_images)
            if not self.policy.min_images <= len(selected) <= self.policy.max_images:
                raise ConditionedDatasetError(
                    "conditioned_count_out_of_range",
                    "The verified, known-category unique source set cannot produce 2,000-3,000 conditioned images.",
                )
            self._event(root, "conditioning", 0, len(selected), "Building deterministic filename-grounded labels.")
            candidate = self._build_candidate(
                root,
                sources,
                selected,
                exclusions,
                near_duplicate_exclusions,
            )
            self._check_cancel(job_id)
            with self._state_transaction(root) as state:
                state["candidate"] = {
                    "candidate_identity": candidate["candidate_identity"],
                    "manifest_relative_path": "candidate_manifest.json",
                    "manifest_sha256": _anchored_file_sha256(
                        root,
                        "candidate_manifest.json",
                        self.project_root,
                    ),
                    "image_count": candidate["image_count"],
                    "category_counts": candidate["category_counts"],
                    "source_counts": candidate["source_counts"],
                    "split_counts": candidate["split_counts"],
                }
                state["status"] = "NEEDS_REVIEW"
                state["stage"] = "independent_evidence"
                state["current"] = len(selected)
                state["total"] = len(selected)
                state["message"] = (
                    "Candidate built and locally checked; independent label audit and dataset validation are required."
                )
                state["updated_at"] = _now()
        except ConditionedDatasetError as exc:
            with self._state_transaction(root) as state:
                state["status"] = "CANCELLED" if exc.code == "build_cancelled" else "FAILED"
                state["stage"] = "cancelled" if exc.code == "build_cancelled" else "failed"
                state["message"] = exc.public_message
                state["updated_at"] = _now()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            with self._state_transaction(root) as state:
                state["status"] = "FAILED"
                state["stage"] = "failed"
                state["message"] = (
                    "Candidate build failed safely; no production freeze or project configuration changed "
                    f"({type(exc).__name__})."
                )
                state["updated_at"] = _now()
        finally:
            with self._lock:
                self._threads.pop(job_id, None)

    def _build_candidate(
        self,
        root: Path,
        sources: Sequence[Mapping[str, Any]],
        selected: Sequence[_SourceRecord],
        exclusions: Sequence[str],
        near_duplicate_exclusions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        build_code_inventory = conditioned_code_inventory()
        implementation_code_identity = str(build_code_inventory["inventory_sha256"])
        family_by_key = _family_assignments(selected)
        split_by_key = _split_assignments(selected, family_by_key)
        imported: list[ImportedSprite] = []
        metadata_by_id: dict[str, dict[str, Any]] = {}
        seen_ids: set[str] = set()
        for index, record in enumerate(selected, start=1):
            self._check_cancel(root.name)
            if not record.content or hashlib.sha256(record.content).hexdigest() != record.byte_sha256:
                raise ConditionedDatasetError(
                    "source_changed_during_build", "A held managed PNG differs from its verified source identity."
                )
            imported_record = import_png_bytes_as_dataset_item(
                record.content,
                source_name=record.path.name,
                options=ImportOptions(
                    max_palette_slots=32,
                    allow_quantize_overcolor=False,
                    quantize_overcolor=False,
                    allow_nearest_resize=False,
                    infer_role_map=True,
                    canonicalize_palette=True,
                ),
                default_category=record.category,
                default_tags=tuple(dict.fromkeys((record.category, *record.tokens[:6], *record.visual_tags))),
            )
            if imported_record.errors or imported_record.bundle is None:
                raise ConditionedDatasetError(
                    "phase7_import_rejected",
                    "A selected exact-32x32 PNG failed the lossless Phase-7 import contract.",
                )
            sprite_id = _sprite_id(record)
            if sprite_id in seen_ids:
                raise ConditionedDatasetError(
                    "sprite_identity_collision", "Two selected source records produced the same sprite identity."
                )
            seen_ids.add(sprite_id)
            family_id = family_by_key[_record_key(record)]
            source_group = _source_group(record)
            evidence = {
                "evidence_type": "source_grounding_with_deterministic_local_pixel_facts",
                "inference_method": "conditioned_filename_taxonomy_v1+local_pixel_vision_v1",
                "human_verified": False,
                "human_truth_claim": False,
                "claim_scope": "source_grounded_non_human_proposal",
                "source_relative_path": record.relative_path,
                "source_path_sha256": hashlib.sha256(record.relative_path.encode("utf-8")).hexdigest(),
                "tokens": list(record.tokens),
                "taxonomy_category": record.category,
                "source_id": record.source_id,
                "source_pack": record.source_title,
                "source_group": source_group,
                "source_sha256": record.byte_sha256,
                "source_byte_count": record.byte_count,
                "license_id": record.license_id,
                "creator": record.creator,
                "duplicate_family_id": family_id,
                "local_pixel_vision": dict(record.visual_descriptor),
                "local_pixel_vision_algorithm": LOCAL_PIXEL_VISION_ALGORITHM,
                "local_pixel_vision_config": LOCAL_PIXEL_VISION_CONFIG,
                "local_pixel_vision_config_identity": LOCAL_PIXEL_VISION_CONFIG_IDENTITY,
                "implementation_code_inventory_sha256": implementation_code_identity,
                "semantic_category_from_pixels": False,
            }
            tags = list(dict.fromkeys((record.category, *record.tokens[:6], *record.visual_tags)))
            prediction = {
                "safe_prefill": {
                    "category": record.category,
                    "object_name": record.object_name,
                    "tags": tags,
                    "short_description": _visual_short_description(record),
                    "confidence": "source_grounded_low",
                    "confidence_reason": "filename_path_category_with_verified_local_pixel_descriptors",
                },
                "source_profile": {"name": "conditioned_filename_taxonomy_v1", "domain": "sprite_assets"},
                "bucket": "filename_grounded",
            }
            semantic = build_semantic_v3_record(prediction)
            semantic = replace(
                semantic,
                source_evidence=evidence,
                warnings=tuple(sorted({*semantic.warnings, "non_human_filename_grounded"})),
            )
            semantic = replace(
                semantic,
                captions=build_captions(semantic, max_captions=8),
                prompt_phrases=build_prompt_phrases(semantic),
                negative_tags=DEFAULT_NEGATIVE_TAGS,
            )
            source_storage_kind = "derived" if record.derivation is not None else "artifacts"
            portable_path = Path("managed") / record.source_id / source_storage_kind / Path(record.relative_path)
            item = DatasetMakerItem(
                sprite_id=sprite_id,
                source_path=portable_path,
                status="accepted",
                category=record.category,
                tags=tuple(tags),
                notes="Source-grounded category plus deterministic local RGBA descriptors; not human truth.",
                source_name=record.relative_path,
                license=record.license_id,
                author=record.creator,
                split=split_by_key[_record_key(record)],
                palette_size=imported_record.item.palette_size,
                has_role_map=imported_record.item.has_role_map,
            )
            auto_metadata = {
                "label_v2_applied": True,
                "label_v2_bucket": "filename_grounded",
                "label_v2_label_confidence_tier": "source_grounded_low",
                "label_v2_safe_prefill": prediction["safe_prefill"],
                "semantic_v3": semantic_v3_to_json(semantic),
                "conditioned_v5": evidence,
                "local_pixel_vision": dict(record.visual_descriptor),
            }
            imported.append(replace(imported_record, item=item, auto_metadata=auto_metadata))
            metadata_by_id[sprite_id] = {
                "source_id": record.source_id,
                "source_pack": record.source_title,
                "source_group": source_group,
                "source_relative_path": record.relative_path,
                "source_sha256": record.byte_sha256,
                "source_byte_count": record.byte_count,
                "license_id": record.license_id,
                "creator": record.creator,
                "duplicate_family_id": family_id,
                "label_evidence": evidence,
                "local_pixel_vision": dict(record.visual_descriptor),
                "source_storage_kind": source_storage_kind,
                "source_derivation": dict(record.derivation) if record.derivation is not None else None,
            }
            if index == 1 or index % 50 == 0 or index == len(selected):
                self._event(
                    root,
                    "conditioning",
                    index,
                    len(selected),
                    f"Conditioned {index} of {len(selected)} selected sprites.",
                )

        candidate_root, candidate_root_identity = _create_anchored_named_directory(
            root,
            self.project_root,
            "candidate",
        )
        with open_anchored_directory(candidate_root, self.project_root) as candidate_anchor:
            if not candidate_root_identity.matches(candidate_anchor.directory_metadata()):
                raise ConditionedDatasetError(
                    "candidate_root_changed", "The candidate publication root changed before export."
                )
            anchored_export = export_dataset_from_imported_sprites_anchored(
                imported,
                DatasetMakerExportConfig(
                    dataset_name="phase7",
                    output_root=candidate_root,
                    max_palette_slots=32,
                    train_fraction=0.8,
                    val_fraction=0.1,
                    test_fraction=0.1,
                    seed=731001,
                    overwrite=False,
                ),
                output_parent=candidate_anchor,
            )
            result = anchored_export.result
            phase7 = result.output_dir
            if phase7 != candidate_root / "phase7":
                raise ConditionedDatasetError(
                    "candidate_root_changed", "The candidate exporter returned an unexpected publication root."
                )
            candidate_anchor.verify()
            with candidate_anchor.open_directory_immovable("phase7") as phase7_anchor:
                if not anchored_export.directory_identity.matches(phase7_anchor.directory_metadata()):
                    raise ConditionedDatasetError(
                        "candidate_root_changed", "The candidate output directory changed before enrichment."
                    )
                _enrich_manifests(phase7, phase7_anchor, metadata_by_id)

        def open_bound_phase7() -> AbstractContextManager[AnchoredDirectory]:
            return _open_bound_candidate_phase7(
                candidate_root,
                self.project_root,
                candidate_identity=candidate_root_identity,
                phase7_identity=anchored_export.directory_identity,
            )

        self._check_cancel(root.name)

        training_result = build_training_manifest(
            phase7,
            variants_per_sprite=2,
            caption_policy="mixed",
            seed=731001,
        )
        training_manifest = phase7 / "training_manifest.jsonl"
        with open_bound_phase7() as phase7_anchor:
            _write_training_manifest(
                phase7_anchor,
                training_manifest.name,
                _portable_training_rows(training_result.rows),
            )
            _write_json(
                phase7_anchor,
                "conditioning_vocabulary.json",
                _conditioning_vocabulary(imported, training_result.rows),
            )
            _write_jsonl(
                phase7_anchor,
                "conditioned_records.jsonl",
                [
                    _conditioned_record(item, metadata_by_id[item.item.sprite_id])
                    for item in sorted(imported, key=lambda value: value.item.sprite_id)
                ],
            )
            _write_jsonl(
                phase7_anchor,
                "split_assignments.jsonl",
                [
                    {
                        "sprite_id": item.item.sprite_id,
                        "split": item.item.split,
                        "duplicate_family_id": metadata_by_id[item.item.sprite_id]["duplicate_family_id"],
                        "source_group": metadata_by_id[item.item.sprite_id]["source_group"],
                    }
                    for item in sorted(imported, key=lambda value: value.item.sprite_id)
                ],
            )
        split_check = _split_integrity(phase7 / "split_assignments.jsonl")
        if not split_check["ok"]:
            raise ConditionedDatasetError(
                "duplicate_split_leakage", "A duplicate/source family crossed train, validation, or test splits."
            )
        coverage = _coverage_report(imported, exclusions)
        if len(coverage["category_counts"]) < 4:
            raise ConditionedDatasetError(
                "representative_coverage", "The candidate covers fewer than four known taxonomy categories."
            )
        with open_bound_phase7() as phase7_anchor:
            _write_json(phase7_anchor, "coverage_report.json", coverage)
            _write_json(phase7_anchor, "split_integrity_report.json", split_check)
        retained_near_duplicate_gate = _retained_near_duplicate_gate(selected)
        if retained_near_duplicate_gate["ok"] is not True:
            raise ConditionedDatasetError(
                "retained_near_duplicate",
                "The conditioned candidate retained a pair inside the conservative near-duplicate bound.",
            )
        with open_bound_phase7() as phase7_anchor:
            _write_json(
                phase7_anchor,
                "duplicate_report.json",
                {
                    "schema_version": "spritelab.dataset.conditioned-duplicates.v2",
                    "exact_byte_and_pixel_duplicates_excluded": int(
                        Counter(exclusions)["exact_byte_duplicate"] + Counter(exclusions)["exact_pixel_duplicate"]
                    ),
                    "near_duplicate_algorithm": NEAR_DUPLICATE_ALGORITHM,
                    "near_duplicate_config": NEAR_DUPLICATE_CONFIG,
                    "near_duplicate_config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
                    "near_duplicate_implementation_code_inventory_sha256": implementation_code_identity,
                    "near_duplicate_excluded_count": len(near_duplicate_exclusions),
                    "near_duplicate_exclusions": [dict(value) for value in near_duplicate_exclusions],
                    "retained_near_duplicate_gate": retained_near_duplicate_gate,
                    "broader_variant_family_config": VARIANT_FAMILY_CONFIG,
                    "family_count": len(set(family_by_key.values())),
                    "whole_family_split_gate": split_check["ok"],
                },
            )
        source_evidence = [_conditioned_source_binding(source) for source in sources]
        with open_bound_phase7() as phase7_anchor:
            _write_json(
                phase7_anchor,
                "provenance_manifest.json",
                {
                    "schema_version": "spritelab.dataset.conditioned-provenance.v1",
                    "sources": source_evidence,
                    "all_source_files_rehashed": True,
                    "license_policy": sorted(ALLOWED_LICENSES),
                    "paths_are_portable": True,
                },
            )
        benchmark = _benchmark_manifest(imported, metadata_by_id)
        audit_subjects = _label_audit_subjects(imported)
        with open_bound_phase7() as phase7_anchor:
            _write_json(phase7_anchor, "benchmark_manifest.json", benchmark)
            _write_json(phase7_anchor, "label_audit_subjects.json", audit_subjects)
        dataset_qa = qa_dataset(phase7, require_semantic_v3=True)
        training_qa = qa_training_manifest(phase7, training_manifest)
        dataset_qa_value = dataset_qa.to_json_dict()
        dataset_qa_value["dataset_dir"] = "."
        training_qa_value = training_qa.to_json_dict()
        training_qa_value["dataset_dir"] = "."
        training_qa_value["manifest_path"] = "training_manifest.jsonl"
        with open_bound_phase7() as phase7_anchor:
            _write_json(phase7_anchor, "dataset_qa_report.json", dataset_qa_value)
            _write_json(phase7_anchor, "training_manifest_qa_report.json", training_qa_value)
        if dataset_qa.errors or training_qa.errors:
            raise ConditionedDatasetError(
                "candidate_qa_failed", "The Phase-7 dataset or training manifest failed local structural QA."
            )
        loader = _loader_check(phase7)
        with open_bound_phase7() as phase7_anchor:
            _write_json(phase7_anchor, "loader_check.json", loader)
        if not loader["ok"]:
            raise ConditionedDatasetError(
                "candidate_loader_failed",
                "The production-format loader check did not cover every split and vocabulary binding.",
            )

        with open_bound_phase7() as phase7_anchor:
            phase7_device = phase7_anchor.directory_metadata().st_dev
            records_sha256 = _stable_file_identity(
                phase7_anchor,
                "conditioned_records.jsonl",
                phase7_device,
            )["sha256"]
            training_manifest_sha256 = _stable_file_identity(
                phase7_anchor,
                training_manifest.name,
                phase7_device,
            )["sha256"]
            split_integrity_sha256 = _stable_file_identity(
                phase7_anchor,
                "split_integrity_report.json",
                phase7_device,
            )["sha256"]
            coverage_report_sha256 = _stable_file_identity(
                phase7_anchor,
                "coverage_report.json",
                phase7_device,
            )["sha256"]
        view_manifest = {
            "schema_version": "spritelab.dataset.conditioned-view.v1",
            "view_identity": stable_hash(
                {
                    "managed_intake_receipt_identities": [
                        source["managed_intake_receipt_identity"] for source in sources
                    ],
                    "image_count": len(imported),
                    "records_sha256": records_sha256,
                }
            ),
            "image_count": len(imported),
            "records_path": "conditioned_records.jsonl",
            "records_sha256": records_sha256,
            "training_manifest_path": "training_manifest.jsonl",
            "training_manifest_sha256": training_manifest_sha256,
            "split_integrity_sha256": split_integrity_sha256,
            "coverage_report_sha256": coverage_report_sha256,
            "requires_semantic_labels": True,
            "human_truth_claim": False,
            "paths_are_portable": True,
        }
        with open_bound_phase7() as phase7_anchor:
            _write_json(phase7_anchor, "view_manifest.json", view_manifest)
            payload_inventory = _inventory_from_anchor(phase7_anchor)
        with open_anchored_directory(candidate_root, self.project_root) as candidate_anchor:
            if not candidate_root_identity.matches(candidate_anchor.directory_metadata()):
                raise ConditionedDatasetError(
                    "candidate_root_changed", "The candidate publication root changed before completion."
                )
            phase7_commit = commit_anchored_dataset_maker_export(
                candidate_anchor,
                "phase7",
                expected_parent_identity=candidate_root_identity,
                expected_directory_identity=anchored_export.directory_identity,
                expected_inventory=payload_inventory,
            )
        inventory_identity = _inventory_identity(payload_inventory)
        if conditioned_code_inventory() != build_code_inventory:
            raise ConditionedDatasetError(
                "conditioned_code_changed",
                "Conditioned production code changed during candidate construction.",
            )
        code_inventory = build_code_inventory
        candidate_identity = stable_hash(
            {
                "schema_version": CANDIDATE_SCHEMA,
                "input_bindings": source_evidence,
                "production_code_identity": code_inventory["inventory_sha256"],
                "payload_inventory_sha256": inventory_identity,
                "image_count": len(imported),
                "recipe": CONDITIONING_RECIPE,
            }
        )
        candidate = {
            "schema_version": CANDIDATE_SCHEMA,
            "candidate_identity": candidate_identity,
            "payload_inventory_sha256": inventory_identity,
            "payload_inventory": payload_inventory,
            "phase7_commit_identity": stable_hash(phase7_commit),
            "image_count": len(imported),
            "category_counts": coverage["category_counts"],
            "source_counts": coverage["source_counts"],
            "split_counts": coverage["split_counts"],
            "label_audit_subjects": audit_subjects,
            "label_audit_subjects_identity": audit_subjects["subjects_identity"],
            "benchmark_category_counts": benchmark["category_counts"],
            "input_bindings": source_evidence,
            "production_code_inventory": code_inventory,
            "production_code_identity": code_inventory["inventory_sha256"],
            "conditioning_recipe": CONDITIONING_RECIPE,
            "local_pixel_vision_config_identity": LOCAL_PIXEL_VISION_CONFIG_IDENTITY,
            "near_duplicate_config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
            "near_duplicate_retained_gate": retained_near_duplicate_gate,
            "count_policy": {
                "minimum": self.policy.min_images,
                "target": self.policy.target_images,
                "maximum": self.policy.max_images,
            },
            "handoff_identities": [source["handoff_identity"] for source in sources],
            "dataset_references": [source["dataset_reference"] for source in sources],
            "managed_intake_receipt_identities": [source["managed_intake_receipt_identity"] for source in sources],
            "harvest_import_receipt_identities": [source["harvest_import_receipt_identity"] for source in sources],
            "managed_output_inventory_sha256": [source["managed_output_inventory_sha256"] for source in sources],
            "harvest_run_ids": [source["run_id"] for source in sources],
            "license_ids": sorted({str(source["license_id"]) for source in sources}),
            "local_structural_qa": {
                "dataset_qa_errors": len(dataset_qa.errors),
                "training_manifest_qa_errors": len(training_qa.errors),
                "loader_ok": loader["ok"],
                "split_integrity_ok": split_check["ok"],
            },
            "independent_audit_generated": False,
            "production_authorized": False,
            "paths_exposed": False,
        }
        with open_anchored_directory(root, self.project_root) as anchor:
            _write_anchored_bytes(
                anchor,
                "candidate_manifest.json",
                (strict_json_dumps(candidate, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        return candidate

    def _load_candidate(self, root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
        reference = state.get("candidate")
        if not isinstance(reference, Mapping):
            raise ConditionedDatasetError("candidate_not_ready", "No complete conditioned candidate is available.")
        if reference.get("manifest_relative_path") != "candidate_manifest.json":
            raise ConditionedDatasetError("candidate_schema", "The conditioned candidate manifest location changed.")
        with open_anchored_directory(root, self.project_root) as anchor:
            content = _read_anchored_regular_bytes(
                anchor,
                "candidate_manifest.json",
                max_bytes=128 * 1024 * 1024,
            )
        try:
            value = strict_json_loads(content)
        except ValueError as exc:
            raise ConditionedDatasetError("candidate_schema", "The conditioned candidate manifest is invalid.") from exc
        if not isinstance(value, Mapping):
            raise ConditionedDatasetError("candidate_schema", "The conditioned candidate manifest is invalid.")
        candidate = dict(value)
        if candidate.get("schema_version") != CANDIDATE_SCHEMA:
            raise ConditionedDatasetError("candidate_schema", "The conditioned candidate manifest schema is invalid.")
        if hashlib.sha256(content).hexdigest() != reference.get("manifest_sha256"):
            raise ConditionedDatasetError(
                "candidate_manifest_changed", "The conditioned candidate manifest changed after build."
            )
        count = candidate.get("image_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not self.policy.min_images <= count <= self.policy.max_images
        ):
            raise ConditionedDatasetError(
                "candidate_count", "The conditioned candidate count is outside the authorized range."
            )
        expected_inventory = candidate.get("payload_inventory")
        if not isinstance(expected_inventory, Mapping):
            raise ConditionedDatasetError("candidate_schema", "The conditioned candidate inventory is invalid.")
        candidate_root = root / "candidate"
        actual_inventory = _inventory(candidate_root / "phase7", self.project_root)
        if actual_inventory != expected_inventory:
            raise ConditionedDatasetError(
                "candidate_payload_changed", "The conditioned candidate payload changed after construction."
            )
        try:
            with open_anchored_directory(candidate_root, self.project_root) as candidate_anchor:
                phase7_commit = verify_anchored_dataset_maker_export(
                    candidate_anchor,
                    "phase7",
                    expected_inventory=actual_inventory,
                )
        except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
            raise ConditionedDatasetError(
                "candidate_commit_changed", "The conditioned candidate completion marker is missing or changed."
            ) from exc
        if stable_hash(phase7_commit) != candidate.get("phase7_commit_identity"):
            raise ConditionedDatasetError(
                "candidate_commit_changed", "The conditioned candidate completion marker binding changed."
            )
        return candidate

    def _revalidate_candidate_context(
        self,
        root: Path,
        state: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> None:
        references = state.get("dataset_references")
        if not isinstance(references, list) or references != candidate.get("dataset_references"):
            raise ConditionedDatasetError(
                "candidate_input_selection_changed", "The candidate input selection is missing or changed."
            )
        sources = [self._verified_source(str(reference)) for reference in references]
        bindings = [_conditioned_source_binding(source) for source in sources]
        if bindings != candidate.get("input_bindings"):
            raise ConditionedDatasetError(
                "candidate_input_stale", "A managed input, Harvest handoff, catalog, or certificate binding changed."
            )
        code_inventory = conditioned_code_inventory()
        if (
            candidate.get("production_code_inventory") != code_inventory
            or candidate.get("production_code_identity") != code_inventory["inventory_sha256"]
        ):
            raise ConditionedDatasetError(
                "candidate_code_stale", "Conditioned production code changed after the candidate was built."
            )
        payload_inventory = _inventory(root / "candidate" / "phase7", self.project_root)
        payload_identity = _inventory_identity(payload_inventory)
        if payload_inventory != candidate.get("payload_inventory") or payload_identity != candidate.get(
            "payload_inventory_sha256"
        ):
            raise ConditionedDatasetError("candidate_bytes_changed", "Candidate artifact bytes changed after build.")
        expected_identity = stable_hash(
            {
                "schema_version": CANDIDATE_SCHEMA,
                "input_bindings": bindings,
                "production_code_identity": code_inventory["inventory_sha256"],
                "payload_inventory_sha256": payload_identity,
                "image_count": candidate.get("image_count"),
                "recipe": CONDITIONING_RECIPE,
            }
        )
        if candidate.get("candidate_identity") != expected_identity:
            raise ConditionedDatasetError(
                "candidate_identity_changed", "The candidate identity no longer matches its exact inputs and code."
            )

    def _validate_evidence(
        self,
        kind: str,
        document: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(document, Mapping):
            raise ConditionedDatasetError(
                "evidence_document", "Independent evidence must be a JSON object.", status_code=422
            )
        value = json.loads(strict_json_dumps(dict(document), sort_keys=True))
        schema = LABEL_AUDIT_SCHEMA if kind == "label_audit" else DATASET_VALIDATION_SCHEMA
        gates = LABEL_AUDIT_GATES if kind == "label_audit" else DATASET_VALIDATION_GATES
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
        if set(value) != expected_keys or _contains_private_or_absolute_path(value):
            raise ConditionedDatasetError(
                "evidence_schema", "Independent evidence contains unknown or private-path-bearing fields."
            )
        if value.get("schema_version") != schema or value.get("verdict") != "PASS":
            raise ConditionedDatasetError(
                "evidence_verdict", "The selected independent report is not an applicable PASS report."
            )
        if value.get("independent") is not True or value.get("generated_by_conditioned_workflow") is not False:
            raise ConditionedDatasetError(
                "evidence_independence", "The report must explicitly be independent of the conditioned workflow."
            )
        auditor = value.get("auditor")
        if not isinstance(auditor, Mapping) or set(auditor) != {
            "auditor_id",
            "code_identity_sha256",
            "implementation_inventory",
        }:
            raise ConditionedDatasetError("evidence_auditor", "Independent evidence lacks auditor identity.")
        auditor_id = str(auditor.get("auditor_id") or "")
        code_identity = str(auditor.get("code_identity_sha256") or "")
        expected_auditor = TRUSTED_AUDITOR_IDS[kind]
        expected_inventory = trusted_auditor_inventory(kind)
        if (
            auditor_id != expected_auditor
            or code_identity != expected_inventory["inventory_sha256"]
            or auditor.get("implementation_inventory") != expected_inventory
        ):
            raise ConditionedDatasetError(
                "evidence_auditor",
                "Independent evidence is not bound to the current trusted auditor implementation inventory.",
            )
        bindings = value.get("bindings")
        expected_bindings = {
            "candidate_identity": candidate.get("candidate_identity"),
            "payload_inventory_sha256": candidate.get("payload_inventory_sha256"),
            "image_count": candidate.get("image_count"),
            "production_code_identity": candidate.get("production_code_identity"),
            "label_audit_subjects_identity": candidate.get("label_audit_subjects_identity"),
        }
        if not isinstance(bindings, Mapping) or dict(bindings) != expected_bindings:
            raise ConditionedDatasetError(
                "evidence_candidate_binding", "Independent evidence is not bound to this exact candidate."
            )
        if value.get("subject_files") != candidate.get("payload_inventory"):
            raise ConditionedDatasetError(
                "evidence_file_inventory", "Independent evidence does not enumerate every exact candidate file."
            )
        audit_subjects = candidate.get("label_audit_subjects")
        if not isinstance(audit_subjects, Mapping) or value.get("audit_subjects") != audit_subjects:
            raise ConditionedDatasetError(
                "evidence_audit_subjects",
                "Independent evidence does not cover the exact stratified and mandatory audit subjects.",
            )
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
                "split_counts": candidate.get("split_counts"),
                "category_counts": candidate.get("category_counts"),
                "source_counts": candidate.get("source_counts"),
                "benchmark_category_counts": candidate.get("benchmark_category_counts"),
                "payload_inventory_sha256": candidate.get("payload_inventory_sha256"),
                "verified_file_count": len(candidate.get("payload_inventory") or {}),
                "near_duplicate_recomputation": {
                    "algorithm_id": NEAR_DUPLICATE_ALGORITHM,
                    "config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
                    "retained_count": candidate.get("image_count"),
                    "checked_same_category_pairs": sum(
                        int(count) * (int(count) - 1) // 2
                        for count in dict(candidate.get("category_counts") or {}).values()
                    ),
                    "violation_count": 0,
                    "gate_identity": dict(candidate.get("near_duplicate_retained_gate") or {}).get("gate_identity"),
                },
            }
        if value.get("metrics") != expected_metrics:
            raise ConditionedDatasetError(
                "evidence_metrics", "Independent evidence metrics do not reproduce the exact required distributions."
            )
        checks = value.get("checks")
        if not isinstance(checks, Mapping) or set(checks) != gates:
            raise ConditionedDatasetError(
                "evidence_checks", "Independent evidence does not contain the complete mandatory gate set."
            )
        if any(type(result) is not str or result != "PASS" for result in checks.values()):
            raise ConditionedDatasetError("evidence_checks", "Every mandatory independent evidence gate must PASS.")
        run_payload = dict(value)
        run_identity = str(run_payload.pop("audit_run_identity", ""))
        if not SHA256_PATTERN.fullmatch(run_identity) or stable_hash(run_payload) != run_identity:
            raise ConditionedDatasetError(
                "evidence_run_identity", "Independent evidence audit-run identity does not match its exact report."
            )
        return value

    def _verified_selected_evidence(
        self,
        root: Path,
        state: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        label_audit_sha256: str,
        dataset_validation_sha256: str,
    ) -> dict[str, dict[str, Any]]:
        selected_hashes = {
            "label_audit": label_audit_sha256,
            "dataset_validation": dataset_validation_sha256,
        }
        stored = state.get("evidence")
        if not isinstance(stored, Mapping):
            raise ConditionedDatasetError(
                "independent_evidence_missing", "Both independent evidence reports are required before publication."
            )
        results: dict[str, dict[str, Any]] = {}
        for kind, digest in selected_hashes.items():
            if not SHA256_PATTERN.fullmatch(str(digest)):
                raise ConditionedDatasetError(
                    "evidence_identity", "A selected evidence identity is invalid.", status_code=422
                )
            reference = stored.get(kind)
            if (
                not isinstance(reference, Mapping)
                or set(reference)
                != {
                    "relative_path",
                    "sha256",
                    "byte_count",
                    "auditor_id",
                    "audit_run_identity",
                    "operation_id",
                    "operation_identity",
                    "receipt",
                    "action",
                }
                or reference.get("sha256") != digest
            ):
                raise ConditionedDatasetError(
                    "evidence_selection", "The selected evidence is not the verified report attached to this candidate."
                )
            relative = _canonical_relative(str(reference.get("relative_path") or ""))
            parts = PurePosixPath(relative).parts
            if len(parts) != 2 or parts[0] != "evidence" or parts[1] != f"{kind}-{digest}.json":
                raise ConditionedDatasetError("evidence_changed", "Independent evidence location changed.")
            receipt_reference = reference.get("receipt")
            if not isinstance(receipt_reference, Mapping) or set(receipt_reference) != {
                "relative_path",
                "sha256",
                "byte_count",
                "receipt_identity",
            }:
                raise ConditionedDatasetError(
                    "independent_audit_receipt", "The selected evidence lacks its server-managed audit receipt."
                )
            receipt_relative = _canonical_relative(str(receipt_reference.get("relative_path") or ""))
            receipt_parts = PurePosixPath(receipt_relative).parts
            receipt_identity = str(receipt_reference.get("receipt_identity") or "")
            receipt_digest = str(receipt_reference.get("sha256") or "")
            if (
                len(receipt_parts) != 2
                or receipt_parts[0] != "evidence"
                or receipt_parts[1] != f"{kind}-receipt-{receipt_identity}.json"
                or not SHA256_PATTERN.fullmatch(receipt_identity)
                or not SHA256_PATTERN.fullmatch(receipt_digest)
            ):
                raise ConditionedDatasetError(
                    "independent_audit_receipt", "The selected audit receipt location or identity changed."
                )
            action_reference = reference.get("action")
            if not isinstance(action_reference, Mapping) or set(action_reference) != {
                "relative_path",
                "sha256",
                "byte_count",
                "record_identity",
            }:
                raise ConditionedDatasetError(
                    "independent_audit_action", "The selected evidence lacks its durable server action record."
                )
            action_relative = _canonical_relative(str(action_reference.get("relative_path") or ""))
            action_parts = PurePosixPath(action_relative).parts
            action_identity = str(action_reference.get("record_identity") or "")
            action_digest = str(action_reference.get("sha256") or "")
            expected_action_name = f"{kind}-{reference.get('operation_id')}.json"
            if (
                len(action_parts) != 2
                or action_parts[0] != "audit_actions"
                or action_parts[1] != expected_action_name
                or not SHA256_PATTERN.fullmatch(action_identity)
                or not SHA256_PATTERN.fullmatch(action_digest)
            ):
                raise ConditionedDatasetError(
                    "independent_audit_action", "The selected audit action location or identity changed."
                )
            evidence_root = require_confined_path(root / "evidence", root)
            actions_root = require_confined_path(root / "audit_actions", root)
            path = evidence_root / parts[1]
            receipt_path = evidence_root / receipt_parts[1]
            action_path = actions_root / action_parts[1]
            with open_anchored_directory(evidence_root, self.project_root) as anchor:
                content = _read_anchored_regular_bytes(anchor, parts[1], max_bytes=64 * 1024 * 1024)
                receipt_content = _read_anchored_regular_bytes(
                    anchor,
                    receipt_parts[1],
                    max_bytes=16 * 1024 * 1024,
                )
            with open_anchored_directory(actions_root, self.project_root) as action_anchor:
                action_content = _read_anchored_regular_bytes(
                    action_anchor,
                    action_parts[1],
                    max_bytes=16 * 1024 * 1024,
                )
            if hashlib.sha256(content).hexdigest() != digest or len(content) != reference.get("byte_count"):
                raise ConditionedDatasetError("evidence_changed", "Independent evidence bytes changed after selection.")
            if hashlib.sha256(receipt_content).hexdigest() != receipt_digest or len(
                receipt_content
            ) != receipt_reference.get("byte_count"):
                raise ConditionedDatasetError(
                    "independent_audit_receipt", "The server-managed audit receipt bytes changed after selection."
                )
            if hashlib.sha256(action_content).hexdigest() != action_digest or len(
                action_content
            ) != action_reference.get("byte_count"):
                raise ConditionedDatasetError(
                    "independent_audit_action", "The durable audit action-record bytes changed after selection."
                )
            try:
                value = strict_json_loads(content)
                receipt_value = strict_json_loads(receipt_content)
                action_value = strict_json_loads(action_content)
            except ValueError as exc:
                raise ConditionedDatasetError(
                    "evidence_changed", "Independent evidence is no longer readable."
                ) from exc
            document = self._validate_evidence(kind, value, candidate)
            if content != (strict_json_dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"):
                raise ConditionedDatasetError(
                    "evidence_changed", "Independent evidence is not the exact canonical report."
                )
            current_inventory = trusted_auditor_inventory(kind)
            try:
                validated_receipt = validate_audit_receipt(
                    receipt_value,
                    kind=kind,
                    expected_job_id=root.name,
                    expected_report_sha256=digest,
                    expected_report_byte_count=len(content),
                    report=document,
                    candidate=candidate,
                    current_auditor_inventory=current_inventory,
                )
            except ConditionedAuditReceiptError as exc:
                raise ConditionedDatasetError(
                    "independent_audit_receipt", "The server-managed audit receipt is invalid or stale."
                ) from exc
            if (
                receipt_content
                != (strict_json_dumps(validated_receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
                or validated_receipt.get("receipt_identity") != receipt_identity
                or validated_receipt.get("operation_id") != reference.get("operation_id")
                or validated_receipt.get("operation_identity") != reference.get("operation_identity")
                or validated_receipt.get("audit_run_identity") != reference.get("audit_run_identity")
                or validated_receipt.get("auditor_id") != reference.get("auditor_id")
            ):
                raise ConditionedDatasetError(
                    "independent_audit_receipt", "The selected audit receipt differs from its job-state binding."
                )
            try:
                validated_action = validate_audit_action_record(
                    action_value,
                    kind=kind,
                    expected_job_id=root.name,
                    expected_report_sha256=digest,
                    expected_report_byte_count=len(content),
                    report=document,
                    expected_receipt_sha256=receipt_digest,
                    expected_receipt_byte_count=len(receipt_content),
                    receipt=validated_receipt,
                    candidate=candidate,
                    current_auditor_inventory=current_inventory,
                )
            except ConditionedAuditReceiptError as exc:
                raise ConditionedDatasetError(
                    "independent_audit_action", "The durable server audit action record is invalid or stale."
                ) from exc
            if (
                action_content != (strict_json_dumps(validated_action, indent=2, sort_keys=True) + "\n").encode("utf-8")
                or validated_action.get("record_identity") != action_identity
                or validated_action.get("operation_id") != reference.get("operation_id")
                or validated_action.get("operation_identity") != reference.get("operation_identity")
            ):
                raise ConditionedDatasetError(
                    "independent_audit_action", "The selected audit action differs from its job-state binding."
                )
            results[kind] = {
                "document": document,
                "path": path,
                "sha256": digest,
                "byte_count": len(content),
                "content": content,
                "receipt": {
                    "document": validated_receipt,
                    "path": receipt_path,
                    "sha256": receipt_digest,
                    "byte_count": len(receipt_content),
                    "content": receipt_content,
                },
                "action": {
                    "document": validated_action,
                    "path": action_path,
                    "sha256": action_digest,
                    "byte_count": len(action_content),
                    "content": action_content,
                },
            }
        return results

    def _publish(
        self,
        job_root: Path,
        candidate: Mapping[str, Any],
        evidence: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Publish an exact direct-final pair, authorized only by its final marker."""

        source_root = require_confined_path(job_root / "candidate" / "phase7", job_root)
        source_inventory = _inventory(source_root, self.project_root)
        files: dict[str, dict[str, Any]] = dict(source_inventory)
        dataset_contents: dict[str, bytes] = {}
        with open_anchored_directory(source_root, self.project_root) as source_anchor:
            for relative, expected in sorted(source_inventory.items()):
                if len(PurePosixPath(relative).parts) != 1:
                    raise ConditionedDatasetError(
                        "publication_layout_unsupported",
                        "Conditioned candidate publication accepts only its fixed flat artifact layout.",
                    )
                content = _read_anchored_regular_bytes(
                    source_anchor,
                    relative,
                    max_bytes=max(self.policy.max_source_bytes, 1024 * 1024 * 1024),
                )
                if hashlib.sha256(content).hexdigest() != expected["sha256"] or len(content) != expected["byte_count"]:
                    raise ConditionedDatasetError(
                        "candidate_copy_changed",
                        "A candidate file changed during freeze publication.",
                    )
                dataset_contents[relative] = content

        for kind in ("label_audit", "dataset_validation"):
            record = evidence[kind]
            report_content = record.get("content")
            if not isinstance(report_content, bytes):
                raise ConditionedDatasetError(
                    "evidence_changed",
                    "Independent evidence bytes are unavailable for publication.",
                )
            report_relative = f"evidence/{kind}.json"
            dataset_contents[report_relative] = report_content
            files[report_relative] = {
                "sha256": str(record["sha256"]),
                "byte_count": int(record["byte_count"]),
            }
            _receipt_artifact_name, receipt_relative = AUDIT_RECEIPT_ARTIFACTS[kind]
            receipt = record.get("receipt")
            receipt_content = receipt.get("content") if isinstance(receipt, Mapping) else None
            if not isinstance(receipt, Mapping) or not isinstance(receipt_content, bytes):
                raise ConditionedDatasetError(
                    "independent_audit_receipt",
                    "Server-managed audit receipt bytes are required for publication.",
                )
            dataset_contents[receipt_relative] = receipt_content
            files[receipt_relative] = {
                "sha256": str(receipt["sha256"]),
                "byte_count": int(receipt["byte_count"]),
            }
            _action_artifact_name, action_relative = AUDIT_ACTION_RECORD_ARTIFACTS[kind]
            action = record.get("action")
            action_content = action.get("content") if isinstance(action, Mapping) else None
            if not isinstance(action, Mapping) or not isinstance(action_content, bytes):
                raise ConditionedDatasetError(
                    "independent_audit_action",
                    "A durable server audit action record is required for publication.",
                )
            dataset_contents[action_relative] = action_content
            files[action_relative] = {
                "sha256": str(action["sha256"]),
                "byte_count": int(action["byte_count"]),
            }

        if _content_inventory(dataset_contents) != files:
            raise ConditionedDatasetError(
                "publication_copy_mismatch",
                "The exact publication bytes do not match their authorized inventory.",
            )
        publication_identity = _inventory_identity(files)
        name = f"conditioned-v5-{publication_identity}"
        datasets_root = _ensure_anchored_child_directory(self.project_root, self.project_root, "datasets")
        campaigns_root = _ensure_anchored_child_directory(self.project_root, self.project_root, "campaigns")
        if datasets_root != self.datasets_root or not _safe_directory(self.datasets_root):
            raise ConditionedDatasetError(
                "datasets_root_unsafe",
                "The project datasets directory is unsafe for publication.",
            )
        if campaigns_root != self.campaigns_root or not _safe_directory(self.campaigns_root):
            raise ConditionedDatasetError(
                "campaigns_root_unsafe",
                "The project campaigns directory is unsafe for publication.",
            )
        final = require_confined_path(self.datasets_root / name, self.datasets_root)
        campaign_directory = require_confined_path(self.campaigns_root / name, self.campaigns_root)

        inventory_payload = _inventory_payload(files)
        artifact_names = {
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
        artifacts = {artifact: {"path": path, **files[path]} for artifact, path in artifact_names.items()}
        activation = {
            "schema_version": ACTIVATION_SCHEMA,
            "dataset_version": 5,
            "dataset_kind": "conditioned",
            "requires_semantic_labels": True,
            "status": "complete",
            "production_authorized": True,
            "immutable": True,
            "image_count": candidate["image_count"],
            "dataset_identity": candidate["candidate_identity"],
            "publication_identity_sha256": publication_identity,
            "labeling_audit_sha256": evidence["label_audit"]["sha256"],
            "validation_report_sha256": evidence["dataset_validation"]["sha256"],
            "artifacts": artifacts,
            "publication_inventory": {
                **inventory_payload,
                "inventory_sha256": _inventory_identity(files),
            },
            "licenses": list(candidate["license_ids"]),
            "paths_are_relative": True,
            "paths_exposed": False,
        }
        activation_content = (strict_json_dumps(activation, indent=2, sort_keys=True) + "\n").encode("utf-8")
        dataset_contents["activation.json"] = activation_content
        dataset_inventory = _content_inventory(dataset_contents)
        activation_sha256 = dataset_inventory["activation.json"]["sha256"]

        with open_anchored_directory(self.datasets_root, self.project_root) as dataset_parent:
            with _open_or_create_direct_final_directory(dataset_parent, final.name) as (
                dataset_anchor,
                dataset_created,
            ):
                if dataset_created:
                    _publish_direct_final_contents(
                        dataset_anchor,
                        dataset_contents,
                        residue_prefix=".dataset-publication-residue-",
                    )
                if _inventory_from_anchor(dataset_anchor) != dataset_inventory:
                    raise ConditionedDatasetError(
                        "publication_copy_mismatch",
                        "The direct-final dataset inventory differs from the intended exact bytes.",
                    )

        activation_relative = (final / "activation.json").relative_to(self.project_root).as_posix()
        campaign_relative = campaign_directory.relative_to(self.project_root).as_posix()
        builder = self._campaign_builder
        if builder is None:
            from spritelab.product_features.training.activation import build_conditioned_three_seed_campaign

            builder = build_conditioned_three_seed_campaign
        try:
            built = builder(
                self.project_root,
                campaign_directory=campaign_relative,
                activation_manifest=activation_relative,
                activation_manifest_sha256=activation_sha256,
                view_manifest=(final / "view_manifest.json").relative_to(self.project_root).as_posix(),
                split_manifest=(final / "training_manifest.jsonl").relative_to(self.project_root).as_posix(),
                conditioning_vocabulary=(final / "conditioning_vocabulary.json")
                .relative_to(self.project_root)
                .as_posix(),
                benchmark_manifest=(final / "benchmark_manifest.json").relative_to(self.project_root).as_posix(),
                output_root=f"training-runs/conditioned-v5-{publication_identity}",
                campaign_id=f"conditioned_v5_{publication_identity[:16]}",
            )
            portable = dict(built.portable_campaign)
            resolved_campaign = dict(built.campaign)
            validation = dict(built.validation)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ConditionedDatasetError(
                "campaign_build_failed",
                "The exact conditioned campaign could not be built; uncommitted bytes were retained.",
            ) from exc
        if (
            validation.get("launch_ready") is not True
            or portable.get("seeds") != [731001, 731002, 731003]
            or dict(portable.get("training") or {}).get("max_optimizer_steps") != 5_000
            or not SHA256_PATTERN.fullmatch(str(resolved_campaign.get("campaign_identity") or ""))
        ):
            raise ConditionedDatasetError(
                "campaign_build_failed",
                "The exact conditioned campaign is not launch-ready.",
            )
        campaign_document = {
            "schema_version": "spritelab.training.conditioned-campaign-config.v1",
            "product_profiles": {
                "recommended": {
                    "display": {"display_name": "Conditioned Dataset-v5 · 3 seeds · 5,000 steps"},
                    "campaign": portable,
                }
            },
        }
        campaign_content = (strict_json_dumps(campaign_document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        campaign_contents = {"campaign.json": campaign_content}
        campaign_inventory = _content_inventory(campaign_contents)
        campaign_sha256 = campaign_inventory["campaign.json"]["sha256"]
        with open_anchored_directory(self.campaigns_root, self.project_root) as campaign_parent:
            if campaign_parent.lexists(campaign_directory.name):
                with campaign_parent.open_directory_immovable(campaign_directory.name) as existing_campaign:
                    if _inventory_from_anchor(existing_campaign) != campaign_inventory:
                        raise ConditionedDatasetError(
                            "campaign_copy_mismatch",
                            "A pre-existing campaign directory differs from the intended exact bytes.",
                        )
        try:
            journal = build_publication_journal(
                publication_identity=publication_identity,
                dataset_inventory=dataset_inventory,
                campaign_inventory=campaign_inventory,
            )
            dataset_marker = build_dataset_commit(
                journal=journal,
                dataset_inventory=dataset_inventory,
                campaign_inventory=campaign_inventory,
            )
            campaign_marker = build_campaign_commit(
                journal=journal,
                dataset_commit=dataset_marker,
                dataset_inventory=dataset_inventory,
                campaign_inventory=campaign_inventory,
            )
        except PublicationCommitError as exc:
            raise ConditionedDatasetError(
                "publication_commit_invalid",
                "The conditioned publication commit evidence could not be built.",
            ) from exc

        with open_anchored_directory(job_root, self.project_root) as job_anchor:
            if not dataset_created and not job_anchor.lexists(PUBLICATION_JOURNAL_NAME):
                raise ConditionedDatasetError(
                    "publication_recovery_unbound",
                    "A pre-existing uncommitted dataset directory has no exact PREPARED journal.",
                )
            _publish_or_reuse_immutable_file(
                job_anchor,
                PUBLICATION_JOURNAL_NAME,
                canonical_publication_commit_bytes(journal),
                residue_prefix=".publication-journal-residue-",
            )
        with open_anchored_directory(self.datasets_root, self.project_root) as dataset_parent:
            _publish_or_reuse_immutable_file(
                dataset_parent,
                dataset_commit_name(publication_identity),
                canonical_publication_commit_bytes(dataset_marker),
                residue_prefix=".dataset-commit-residue-",
            )
        _conditioned_publication_checkpoint("dataset_marker_committed")

        with open_anchored_directory(self.campaigns_root, self.project_root) as campaign_parent:
            with _open_or_create_direct_final_directory(campaign_parent, campaign_directory.name) as (
                campaign_anchor,
                campaign_created,
            ):
                if campaign_created:
                    _publish_direct_final_contents(
                        campaign_anchor,
                        campaign_contents,
                        residue_prefix=".campaign-publication-residue-",
                    )
                if _inventory_from_anchor(campaign_anchor) != campaign_inventory:
                    raise ConditionedDatasetError(
                        "campaign_copy_mismatch",
                        "The direct-final campaign inventory differs from the intended exact bytes.",
                    )
            _publish_or_reuse_immutable_file(
                campaign_parent,
                campaign_commit_name(publication_identity),
                canonical_publication_commit_bytes(campaign_marker),
                residue_prefix=".campaign-commit-residue-",
            )
        _conditioned_publication_checkpoint("campaign_marker_committed")

        _verify_conditioned_publication_pair(
            project_root=self.project_root,
            job_root=job_root,
            dataset_directory=final,
            campaign_directory=campaign_directory,
            publication_identity=publication_identity,
            dataset_inventory=dataset_inventory,
            campaign_inventory=campaign_inventory,
        )
        campaign_path = campaign_directory / "campaign.json"
        return {
            "publication_identity_sha256": publication_identity,
            "activation_manifest": activation_relative,
            "activation_manifest_sha256": activation_sha256,
            "campaign_config": campaign_path.relative_to(self.project_root).as_posix(),
            "campaign_config_sha256": campaign_sha256,
            "campaign_identity_sha256": resolved_campaign["campaign_identity"],
            "campaign_launch_ready": True,
            "campaign_seeds": [731001, 731002, 731003],
            "campaign_steps": 5_000,
            "configuration_activated": False,
            "training_started": False,
            "paths_exposed": False,
        }

    def _job_root(self, job_id: str) -> Path:
        if not JOB_ID_PATTERN.fullmatch(str(job_id)):
            raise ConditionedDatasetError("job_id", "The conditioned candidate job ID is invalid.", status_code=404)
        try:
            root = require_confined_path(self.jobs_root / job_id, self.jobs_root)
        except UnsafeFilesystemOperation as exc:
            raise ConditionedDatasetError(
                "job_id", "The conditioned candidate job is unavailable.", status_code=404
            ) from exc
        if not _safe_directory(root):
            raise ConditionedDatasetError("job_id", "The conditioned candidate job is unavailable.", status_code=404)
        return root

    def _read_state(self, root: Path) -> dict[str, Any]:
        with open_anchored_directory(root, self.project_root) as anchor:
            content = _read_anchored_regular_bytes(anchor, "state.json", max_bytes=16 * 1024 * 1024)
        try:
            value = strict_json_loads(content)
        except ValueError as exc:
            raise ConditionedDatasetError("job_state", "The conditioned candidate job state is invalid.") from exc
        if not isinstance(value, Mapping):
            raise ConditionedDatasetError("job_state", "The conditioned candidate job state is invalid.")
        state = dict(value)
        self._validate_state(root, state)
        return state

    def _validate_state(self, root: Path, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "spritelab.dataset.conditioned-job.v1" or state.get("job_id") != root.name:
            raise ConditionedDatasetError("job_state", "The conditioned candidate job state is invalid.")
        if state.get("paths_exposed") is not False:
            raise ConditionedDatasetError(
                "job_state_privacy", "The conditioned candidate job state violates the private-path contract."
            )

    def _write_state(self, root: Path, state: Mapping[str, Any]) -> None:
        with self._lock, _ConditionedMutationLock(root, ".conditioned-state.lock"):
            self._write_state_unlocked(root, state)

    def _write_state_unlocked(self, root: Path, state: Mapping[str, Any]) -> None:
        self._write_state_payload(root, state)

    def _write_state_payload(self, root: Path, state: Mapping[str, Any]) -> None:
        payload = strict_json_dumps(dict(state), indent=2, sort_keys=True) + "\n"
        if str(self.project_root) in payload or str(self.project_root).replace("\\", "/") in payload:
            raise ConditionedDatasetError(
                "private_path_persistence", "A private project path was refused from durable job state."
            )
        with open_anchored_directory(root, self.project_root) as anchor:
            _write_anchored_bytes(anchor, "state.json", payload.encode("utf-8"))

    @contextmanager
    def _state_transaction(
        self,
        root: Path,
        *,
        rollback: Callable[[], None] | None = None,
        write_failure_rollback: Callable[[], None] | None = None,
    ) -> Any:
        with self._lock, _ConditionedMutationLock(root, ".conditioned-state.lock"):
            state = self._read_state(root)
            try:
                yield state
            except BaseException:
                if rollback is not None:
                    rollback()
                raise
            else:
                try:
                    self._write_state_unlocked(root, state)
                except BaseException:
                    active_rollback = write_failure_rollback or rollback
                    if active_rollback is not None:
                        active_rollback()
                    raise

    def _event(self, root: Path, stage: str, current: int, total: int, message: str) -> None:
        with self._state_transaction(root) as state:
            events = list(state.get("events") or ())[-(self.policy.max_events - 1) :]
            events.append(
                {
                    "timestamp": _now(),
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "message": message[:500],
                }
            )
            state["events"] = events
            state["stage"] = stage
            state["current"] = current
            state["total"] = total
            state["message"] = message[:500]
            state["updated_at"] = _now()
            state["lease"] = self._lease()

    def _check_cancel(self, job_id: str) -> None:
        root = self._job_root(job_id)
        cancel = root / "cancel.json"
        if os.path.lexists(cancel):
            value = _read_json_mapping(cancel, max_bytes=16 * 1024)
            if (
                value.get("schema_version") != "spritelab.dataset.conditioned-cancel.v1"
                or value.get("job_id") != job_id
                or value.get("explicit_action") is not True
                or value.get("paths_exposed") is not False
            ):
                raise ConditionedDatasetError("cancel_marker_invalid", "The durable cancellation marker is invalid.")
            raise ConditionedDatasetError(
                "build_cancelled", "Candidate build was cancelled; no production freeze changed."
            )

    def _lease(self) -> dict[str, Any]:
        return {
            "owner_instance_id": self._instance_id,
            "owner_pid": os.getpid(),
            "heartbeat_at": _now(),
            "expires_after_seconds": self._lease_seconds,
        }

    def _lease_is_fresh(self, value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        if value.get("expires_after_seconds") != self._lease_seconds:
            return False
        try:
            heartbeat = datetime.fromisoformat(str(value.get("heartbeat_at")))
        except ValueError:
            return False
        if heartbeat.tzinfo is None:
            return False
        age = (datetime.now(UTC) - heartbeat.astimezone(UTC)).total_seconds()
        return -5 <= age <= self._lease_seconds


def _normalize_dataset_references(values: Sequence[str] | str) -> tuple[str, ...]:
    raw = (values,) if isinstance(values, str) else tuple(values)
    normalized = tuple(str(value) for value in raw)
    if not normalized or len(normalized) > 8 or len(set(normalized)) != len(normalized):
        raise ConditionedDatasetError(
            "managed_intake_selection",
            "Select one to eight distinct completed managed Dataset imports.",
            status_code=422,
        )
    for reference in normalized:
        if re.fullmatch(r"dataset\.[0-9a-f]{24}", reference) is None:
            raise ConditionedDatasetError(
                "managed_intake_reference", "A managed Dataset reference is invalid.", status_code=404
            )
    return tuple(sorted(normalized))


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(str(run_id)):
        raise ConditionedDatasetError("harvest_run_id", "The Harvest run ID is invalid.", status_code=404)


def _integer(value: Any, *, default: int) -> int:
    return value if type(value) is int else default


def _safe_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and not _metadata_is_link_or_reparse(metadata) and not path.is_mount()


def _ensure_anchored_child_directory(parent: Path, boundary: Path, name: str) -> Path:
    """Create or validate one fixed direct-child directory through a stable parent."""

    with open_anchored_directory(parent, boundary) as anchor:
        try:
            anchor.mkdir(name, exist_ok=True)
        except (OSError, UnsafeFilesystemOperation) as exc:
            raise ConditionedDatasetError(
                "managed_directory_unsafe", "A conditioned managed directory is unsafe."
            ) from exc
    return parent / name


def _create_anchored_named_directory(
    parent: Path,
    boundary: Path,
    name: str,
) -> tuple[Path, OwnedFileIdentity]:
    """Publish one fresh fixed-name directory and retain its inode identity."""

    with open_anchored_directory(parent, boundary) as anchor:
        try:
            identity = anchor.mkdir(name, exist_ok=False)
        except FileExistsError as exc:
            raise ConditionedDatasetError(
                "fresh_directory_required", "A fresh conditioned directory is required."
            ) from exc
    return parent / name, identity


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _is_link_or_reparse(path: Path) -> bool:
    try:
        return _metadata_is_link_or_reparse(path.lstat())
    except OSError:
        return False


def _open_conditioned_lock(anchor: AnchoredDirectory, name: str) -> Any:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(8):
        before: os.stat_result | None = None
        handle: Any = None
        if anchor.lexists(name):
            before = anchor.lstat(name)
            _require_safe_lock_metadata(before)
            try:
                descriptor = anchor.open_file(name, flags)
            except FileNotFoundError:
                continue
        else:
            try:
                descriptor = anchor.open_file(name, flags | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            before = os.fstat(descriptor)
        try:
            opened = os.fstat(descriptor)
            assert before is not None
            _require_same_lock_metadata(before, opened, "A conditioned mutation lock changed while opening.")
            _require_same_lock_metadata(opened, anchor.lstat(name), "A conditioned mutation lock path changed.")
            handle = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = -1
            if opened.st_size == 0:
                if handle.write(b"0") != 1:
                    raise ConditionedDatasetError(
                        "mutation_lock_unsafe", "A conditioned mutation lock could not be initialized."
                    )
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            _verify_conditioned_lock(handle, anchor, name)
            return handle
        except BaseException:
            if handle is not None:
                handle.close()
            if descriptor >= 0:
                os.close(descriptor)
            raise
    raise ConditionedDatasetError(
        "mutation_lock_unsafe", "A conditioned mutation lock changed repeatedly while opening."
    )


def _verify_conditioned_lock(handle: Any, anchor: AnchoredDirectory, name: str) -> None:
    _require_same_lock_metadata(
        os.fstat(handle.fileno()),
        anchor.lstat(name),
        "A conditioned mutation lock path changed while held.",
    )


def _lock_conditioned_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_conditioned_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _require_safe_lock_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _metadata_is_link_or_reparse(metadata)
        or int(getattr(metadata, "st_nlink", 1)) != 1
    ):
        raise ConditionedDatasetError("mutation_lock_unsafe", "A conditioned mutation lock is unsafe.")


def _require_same_lock_metadata(before: os.stat_result, after: os.stat_result, message: str) -> None:
    _require_safe_lock_metadata(after)
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ConditionedDatasetError("mutation_lock_changed", message)


def _require_same_directory_metadata(before: os.stat_result, after: os.stat_result, message: str) -> None:
    if (
        not stat.S_ISDIR(after.st_mode)
        or _metadata_is_link_or_reparse(after)
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
    ):
        raise ConditionedDatasetError("mutation_root_changed", message)


def _content_inventory(contents: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        }
        for relative, content in sorted(contents.items())
    }


@contextmanager
def _open_or_create_direct_final_directory(
    parent: AnchoredDirectory,
    name: str,
) -> Iterator[tuple[AnchoredDirectory, bool]]:
    """Create one final directory directly, or retain an existing one for validation."""

    created = not parent.lexists(name)
    identity = parent.mkdir(name, exist_ok=False) if created else None
    with parent.open_directory_immovable(name) as child:
        if identity is not None and not identity.matches(child.directory_metadata()):
            raise ConditionedDatasetError(
                "publication_directory_changed",
                "A direct-final publication directory changed immediately after creation.",
            )
        child.verify()
        yield child, created
        child.verify()


def _publish_direct_final_contents(
    root: AnchoredDirectory,
    contents: Mapping[str, bytes],
    *,
    residue_prefix: str,
) -> None:
    """Populate one newly-created direct-final directory through retained anchors."""

    flat: dict[str, bytes] = {}
    evidence: dict[str, bytes] = {}
    for relative, content in sorted(contents.items()):
        parts = PurePosixPath(_canonical_relative(relative)).parts
        if len(parts) == 1:
            flat[parts[0]] = content
        elif len(parts) == 2 and parts[0] == "evidence":
            evidence[parts[1]] = content
        else:
            raise ConditionedDatasetError(
                "publication_layout_unsupported",
                "Direct-final publication accepts only its fixed flat/evidence layout.",
            )
    for name, content in sorted(flat.items()):
        _publish_or_reuse_immutable_file(
            root,
            name,
            content,
            residue_prefix=residue_prefix,
        )
    if not evidence:
        return
    try:
        identity = root.mkdir("evidence", exist_ok=False)
    except FileExistsError as exc:
        raise ConditionedDatasetError(
            "publication_directory_changed",
            "The direct-final evidence directory appeared concurrently.",
        ) from exc
    with root.open_directory_immovable("evidence") as evidence_anchor:
        if not identity.matches(evidence_anchor.directory_metadata()):
            raise ConditionedDatasetError(
                "publication_directory_changed",
                "The direct-final evidence directory changed immediately after creation.",
            )
        for name, content in sorted(evidence.items()):
            _publish_or_reuse_immutable_file(
                evidence_anchor,
                name,
                content,
                residue_prefix=residue_prefix,
            )
        evidence_anchor.verify()


def _read_canonical_publication_commit(
    anchor: AnchoredDirectory,
    name: str,
) -> dict[str, Any]:
    content = _read_anchored_regular_bytes(anchor, name, max_bytes=64 * 1024 * 1024)
    try:
        value = strict_json_loads(content)
    except ValueError as exc:
        raise ConditionedDatasetError(
            "publication_commit_invalid",
            "A conditioned publication commit document is invalid JSON.",
        ) from exc
    if not isinstance(value, Mapping):
        raise ConditionedDatasetError(
            "publication_commit_invalid",
            "A conditioned publication commit document must be an object.",
        )
    document = dict(value)
    if canonical_publication_commit_bytes(document) != content:
        raise ConditionedDatasetError(
            "publication_commit_invalid",
            "A conditioned publication commit document is not canonical.",
        )
    return document


def _verify_conditioned_publication_pair(
    *,
    project_root: Path,
    job_root: Path,
    dataset_directory: Path,
    campaign_directory: Path,
    publication_identity: str,
    dataset_inventory: Mapping[str, Mapping[str, Any]],
    campaign_inventory: Mapping[str, Mapping[str, Any]],
) -> None:
    if _inventory(dataset_directory, project_root) != dict(dataset_inventory):
        raise ConditionedDatasetError(
            "publication_commit_invalid",
            "The marker-bound dataset inventory changed.",
        )
    if _inventory(campaign_directory, project_root) != dict(campaign_inventory):
        raise ConditionedDatasetError(
            "publication_commit_invalid",
            "The marker-bound campaign inventory changed.",
        )
    with open_anchored_directory(job_root, project_root) as job_anchor:
        journal = _read_canonical_publication_commit(job_anchor, PUBLICATION_JOURNAL_NAME)
    with open_anchored_directory(dataset_directory.parent, project_root) as dataset_parent:
        dataset_marker = _read_canonical_publication_commit(
            dataset_parent,
            dataset_commit_name(publication_identity),
        )
    with open_anchored_directory(campaign_directory.parent, project_root) as campaign_parent:
        campaign_marker = _read_canonical_publication_commit(
            campaign_parent,
            campaign_commit_name(publication_identity),
        )
    try:
        validate_publication_journal(
            journal,
            dataset_inventory=dataset_inventory,
            campaign_inventory=campaign_inventory,
        )
        validate_dataset_commit(
            dataset_marker,
            journal=journal,
            dataset_inventory=dataset_inventory,
            campaign_inventory=campaign_inventory,
        )
        validated = validate_campaign_commit(
            campaign_marker,
            journal=journal,
            dataset_commit=dataset_marker,
            dataset_inventory=dataset_inventory,
            campaign_inventory=campaign_inventory,
        )
    except PublicationCommitError as exc:
        raise ConditionedDatasetError(
            "publication_commit_invalid",
            "The conditioned dataset/campaign pair marker is invalid.",
        ) from exc
    if validated.get("pair_authority") is not True:
        raise ConditionedDatasetError(
            "publication_commit_invalid",
            "The conditioned campaign marker is not pair-authoritative.",
        )


def _conditioned_publication_checkpoint(_step: str) -> None:
    """Deterministic fault-injection seam after durable publication steps."""


def _publish_or_reuse_immutable_file(
    anchor: AnchoredDirectory,
    target_name: str,
    content: bytes,
    *,
    residue_prefix: str,
) -> None:
    """Publish one direct-child artifact once, or accept identical immutable bytes."""

    if anchor.lexists(target_name):
        existing = _read_anchored_regular_bytes(anchor, target_name, max_bytes=max(1, len(content)))
        if existing != content:
            raise ConditionedDatasetError(
                "evidence_identity_conflict", "Existing independent evidence bytes conflict with their identity."
            )
        return
    temporary: str | None = None
    try:
        descriptor = anchor.open_anonymous_file()
    except (UnsafeFilesystemOperation, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno not in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
            raise
        temporary = f".{target_name}.staging-{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
        descriptor = anchor.open_file(temporary, flags, 0o600)
    identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            if not identity.matches(os.fstat(handle.fileno())):
                raise ConditionedDatasetError(
                    "evidence_staging_changed", "An independent evidence staging file changed while written."
                )
        if temporary is not None and not identity.matches(anchor.lstat(temporary)):
            raise ConditionedDatasetError(
                "evidence_staging_changed", "An independent evidence staging path changed before publication."
            )
        try:
            anchor.publish_held_file_no_replace(
                descriptor,
                temporary,
                target_name,
                identity=identity,
            )
        except FileExistsError:
            existing = _read_anchored_regular_bytes(anchor, target_name, max_bytes=max(1, len(content)))
            if existing != content:
                raise ConditionedDatasetError(
                    "evidence_identity_conflict",
                    "Existing independent evidence bytes conflict with their identity.",
                ) from None
            if temporary is not None:
                anchor.quarantine_if_owned(temporary, identity, prefix=residue_prefix)
            return
        if not identity.matches(anchor.lstat(target_name)):
            raise ConditionedDatasetError(
                "evidence_publication_changed", "An independent evidence file changed during publication."
            )
        if _read_anchored_regular_bytes(anchor, target_name, max_bytes=max(1, len(content))) != content:
            raise ConditionedDatasetError(
                "evidence_publication_changed", "Independent evidence bytes changed during publication."
            )
    except BaseException:
        anchor.quarantine_if_owned(target_name, identity, prefix=residue_prefix)
        if temporary is not None:
            anchor.quarantine_if_owned(temporary, identity, prefix=residue_prefix)
        raise
    finally:
        os.close(descriptor)


def _publish_fresh_immutable_file(
    anchor: AnchoredDirectory,
    target_name: str,
    content: bytes,
    *,
    residue_prefix: str,
) -> None:
    """Publish one direct-child action record with strict no-replace semantics."""

    if anchor.lexists(target_name):
        raise ConditionedDatasetError(
            "audit_action_record_exists",
            "A durable independent-audit action record already exists for that operation.",
        )
    temporary: str | None = None
    try:
        descriptor = anchor.open_anonymous_file()
    except (UnsafeFilesystemOperation, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno not in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
            raise
        temporary = f".{target_name}.staging-{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
        descriptor = anchor.open_file(temporary, flags, 0o600)
    identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            if not identity.matches(os.fstat(handle.fileno())):
                raise ConditionedDatasetError(
                    "audit_action_staging_changed", "An audit action staging file changed while written."
                )
        if temporary is not None and not identity.matches(anchor.lstat(temporary)):
            raise ConditionedDatasetError(
                "audit_action_staging_changed", "An audit action staging path changed before publication."
            )
        try:
            anchor.publish_held_file_no_replace(
                descriptor,
                temporary,
                target_name,
                identity=identity,
            )
        except FileExistsError as exc:
            raise ConditionedDatasetError(
                "audit_action_record_exists",
                "A durable independent-audit action record appeared concurrently.",
            ) from exc
        if (
            not identity.matches(anchor.lstat(target_name))
            or _read_anchored_regular_bytes(anchor, target_name, max_bytes=max(1, len(content))) != content
        ):
            raise ConditionedDatasetError(
                "audit_action_publication_changed", "An audit action record changed during no-replace publication."
            )
    except BaseException:
        anchor.quarantine_if_owned(target_name, identity, prefix=residue_prefix)
        if temporary is not None:
            anchor.quarantine_if_owned(temporary, identity, prefix=residue_prefix)
        raise
    finally:
        os.close(descriptor)


@dataclass
class _HeldRegularFileSnapshot:
    descriptor: int
    identity: OwnedFileIdentity
    metadata: os.stat_result
    content: bytes
    replaced: bool = False


@contextmanager
def _held_regular_file_snapshot(
    anchor: AnchoredDirectory,
    name: str,
    *,
    max_bytes: int,
) -> Iterator[_HeldRegularFileSnapshot]:
    """Retain one exact file capability across a compare/commit operation."""

    descriptor = anchor.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size > max_bytes
        ):
            raise ConditionedDatasetError(
                "managed_file_unsafe",
                "A retained managed file is not a singly linked bounded regular file.",
            )
        identity = OwnedFileIdentity.from_stat(metadata)
        os.lseek(descriptor, 0, os.SEEK_SET)
        content = b""
        while len(content) <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(content)))
            if not chunk:
                break
            content += chunk
        if len(content) != metadata.st_size or len(content) > max_bytes:
            raise ConditionedDatasetError("managed_file_changed", "A retained managed file changed while read.")
        snapshot = _HeldRegularFileSnapshot(descriptor, identity, metadata, content)
        _verify_held_regular_file_snapshot(anchor, name, snapshot)
        yield snapshot
        if not snapshot.replaced:
            _verify_held_regular_file_snapshot(anchor, name, snapshot)
    finally:
        os.close(descriptor)


def _verify_held_regular_file_snapshot(
    anchor: AnchoredDirectory,
    name: str,
    snapshot: _HeldRegularFileSnapshot,
) -> None:
    """Verify that the held descriptor and its public name still contain the same bytes."""

    opened = os.fstat(snapshot.descriptor)
    current = anchor.lstat(name)
    expected = snapshot.metadata
    for metadata in (opened, current):
        if (
            not snapshot.identity.matches(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size != expected.st_size
            or metadata.st_mtime_ns != expected.st_mtime_ns
        ):
            raise ConditionedDatasetError(
                "managed_file_changed",
                "A retained managed file changed before its commit boundary.",
            )
    os.lseek(snapshot.descriptor, 0, os.SEEK_SET)
    observed = b""
    while len(observed) <= len(snapshot.content):
        chunk = os.read(snapshot.descriptor, min(1024 * 1024, len(snapshot.content) + 1 - len(observed)))
        if not chunk:
            break
        observed += chunk
    if observed != snapshot.content:
        raise ConditionedDatasetError(
            "managed_file_changed",
            "A retained managed file's exact bytes changed before its commit boundary.",
        )


def _replace_held_config_if_supported(
    anchor: AnchoredDirectory,
    destination: _HeldRegularFileSnapshot,
    payload: bytes,
    *,
    expected_sha256: str,
) -> bool:
    """Perform the exact Windows held-handle CAS; POSIX uses marker authority."""

    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ConditionedDatasetError("activation_reload_failed", "Activation payload identity changed.")
    if os.name != "nt":
        return False
    temporary = f".spritelab.yaml.activation-{uuid.uuid4().hex}"
    descriptor = anchor.open_file(
        temporary,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
        0o600,
    )
    identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_held_regular_file_snapshot(anchor, "spritelab.yaml", destination)
        try:
            anchor.replace_held_file_if_owned(
                descriptor,
                temporary,
                "spritelab.yaml",
                identity=identity,
                destination_descriptor=destination.descriptor,
                destination_identity=destination.identity,
            )
        except ExactPublicationUnsupported:
            return False
        destination.replaced = True
        committed = _read_anchored_regular_bytes(
            anchor,
            "spritelab.yaml",
            max_bytes=16 * 1024 * 1024,
        )
        if committed != payload or hashlib.sha256(committed).hexdigest() != expected_sha256:
            raise ConditionedDatasetError(
                "activation_reload_failed",
                "The exact held configuration replacement did not commit intended bytes.",
            )
        return True
    finally:
        os.close(descriptor)
        anchor.quarantine_if_owned(
            temporary,
            identity,
            prefix=".activation-config-source-residue-",
        )


def _retained_stage_alias(
    anchor: AnchoredDirectory,
    target_name: str,
    metadata: os.stat_result,
) -> str:
    """Return the sole exact named POSIX stage retained for a two-link target."""

    prefix = f".{target_name}.staging-"
    candidates = [
        candidate
        for candidate in anchor.names()
        if re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}", candidate) is not None
    ]
    if len(candidates) != 1:
        raise ConditionedDatasetError(
            "managed_file_unsafe",
            "A two-link managed file lost its sole retained publication stage.",
        )
    candidate = candidates[0]
    current = anchor.lstat(candidate)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _metadata_is_link_or_reparse(metadata)
        or metadata.st_nlink != 2
        or not stat.S_ISREG(current.st_mode)
        or _metadata_is_link_or_reparse(current)
        or current.st_dev != metadata.st_dev
        or current.st_ino != metadata.st_ino
        or current.st_nlink != 2
        or current.st_size != metadata.st_size
    ):
        raise ConditionedDatasetError(
            "managed_file_unsafe",
            "A retained publication stage differs from its exact target inode.",
        )
    return candidate


def _read_anchored_regular_bytes(
    anchor: AnchoredDirectory,
    name: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one bounded child, binding any exact retained POSIX stage alias."""

    try:
        before = anchor.lstat(name)
    except OSError as exc:
        raise ConditionedDatasetError(
            "managed_file_unsafe", "A required managed file is unavailable or unsafe."
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _metadata_is_link_or_reparse(before)
        or before.st_nlink not in {1, 2}
        or before.st_size > max_bytes
    ):
        raise ConditionedDatasetError(
            "managed_file_unsafe", "A required managed file is not a singly linked bounded regular file."
        )
    retained_alias = _retained_stage_alias(anchor, name, before) if before.st_nlink == 2 else None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = anchor.open_file(name, flags)
    except OSError as exc:
        raise ConditionedDatasetError(
            "managed_file_unreadable", "A required managed file could not be opened safely."
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_nlink != before.st_nlink
        ):
            raise ConditionedDatasetError("managed_file_changed", "A managed file changed while it was opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(content) > max_bytes:
        raise ConditionedDatasetError("managed_file_oversized", "A required managed file exceeds its byte limit.")
    after = anchor.lstat(name)
    if (
        len(content) != before.st_size
        or not stat.S_ISREG(after.st_mode)
        or _metadata_is_link_or_reparse(after)
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_nlink != before.st_nlink
        or not stat.S_ISREG(opened_after.st_mode)
        or _metadata_is_link_or_reparse(opened_after)
        or opened_after.st_dev != before.st_dev
        or opened_after.st_ino != before.st_ino
        or opened_after.st_size != before.st_size
        or opened_after.st_mtime_ns != before.st_mtime_ns
        or opened_after.st_nlink != before.st_nlink
    ):
        raise ConditionedDatasetError("managed_file_changed", "A managed file changed while it was read.")
    if retained_alias is not None:
        if _retained_stage_alias(anchor, name, after) != retained_alias:
            raise ConditionedDatasetError(
                "managed_file_changed",
                "A retained managed publication stage changed while its target was read.",
            )
    return content


def _read_anchored_relative_bytes(
    anchor: AnchoredDirectory,
    relative: str,
    *,
    max_bytes: int,
) -> bytes:
    parts = PurePosixPath(_canonical_relative(relative)).parts
    with ExitStack() as stack:
        current = anchor
        for name in parts[:-1]:
            current = stack.enter_context(current.open_directory(name))
        return _read_anchored_regular_bytes(current, parts[-1], max_bytes=max_bytes)


def _anchored_file_sha256(parent: Path, name: str, boundary: Path) -> str:
    with open_anchored_directory(parent, boundary) as anchor:
        content = _read_anchored_regular_bytes(anchor, name, max_bytes=128 * 1024 * 1024)
    return hashlib.sha256(content).hexdigest()


def _read_regular_bytes(path: Path, root: Path, *, max_bytes: int) -> bytes:
    try:
        target = require_confined_path(path, root)
        with open_anchored_directory(target.parent, root) as anchor:
            return _read_anchored_regular_bytes(anchor, target.name, max_bytes=max_bytes)
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "managed_file_unsafe", "A required managed file is unavailable or unsafe."
        ) from exc


def _read_json_mapping(path: Path, *, max_bytes: int) -> dict[str, Any]:
    content = _read_regular_bytes(path, path.parent, max_bytes=max_bytes)
    try:
        value = strict_json_loads(content)
    except ValueError as exc:
        raise ConditionedDatasetError("managed_json_invalid", "A required managed JSON document is invalid.") from exc
    if not isinstance(value, Mapping):
        raise ConditionedDatasetError("managed_json_invalid", "A required managed JSON document must be an object.")
    return dict(value)


def _declared_manifest_hash(handoff: Mapping[str, Any]) -> str:
    candidates: list[Any] = [handoff.get("artifact_manifest_sha256")]
    for key in ("artifact_manifest", "artifact_manifest_binding"):
        value = handoff.get(key)
        if isinstance(value, Mapping):
            candidates.extend((value.get("sha256"), value.get("file_sha256")))
    hashes = {str(value) for value in candidates if SHA256_PATTERN.fullmatch(str(value or ""))}
    if len(hashes) != 1:
        raise ConditionedDatasetError(
            "harvest_artifact_manifest_binding",
            "The Harvest handoff lacks one unambiguous artifact-manifest file identity.",
        )
    return hashes.pop()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return strict_json_dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_relative(value: str) -> str:
    try:
        return canonical_portable_relative_path(value)
    except ValueError as exc:
        raise ConditionedDatasetError(
            "artifact_relative_path", "A Harvest artifact path is not canonical and portable."
        ) from exc


def _contains_private_or_absolute_path(value: Any) -> bool:
    """Reject path-bearing evidence that could leak host-private locations."""

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


def _path_tokens(value: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", value.casefold())
    return tuple(token for token in tokens if token not in _STOP_TOKENS and not token.isdecimal())


def _infer_semantics(relative_path: str) -> tuple[str, str, tuple[str, ...], bool]:
    all_tokens = _path_tokens(relative_path)
    stem_tokens = _path_tokens(PurePosixPath(relative_path).stem)
    parent_tokens = _path_tokens(PurePosixPath(relative_path).parent.as_posix())
    path_matches = [
        category for category, terms in _CATEGORY_PATH_HINTS.items() if any(token in terms for token in parent_tokens)
    ]
    if len(path_matches) > 1:
        return "unknown", "", all_tokens, True
    path_category = path_matches[0] if path_matches else None
    stem_matches = [
        category
        for category, terms in _CATEGORY_TERMS.items()
        if category != "icon" and any(token in terms for token in stem_tokens)
    ]
    if path_category is not None:
        if any(category != path_category for category in stem_matches):
            return "unknown", "", all_tokens, True
        category = path_category
    elif len(stem_matches) > 1:
        return "unknown", "", all_tokens, True
    elif stem_matches:
        category = stem_matches[0]
    elif any(token in _CATEGORY_TERMS["icon"] for token in all_tokens):
        category = "icon"
    else:
        return "unknown", "", all_tokens, False
    meaningful = tuple(token for token in stem_tokens if token not in {"sprite", "tile", "icon"})
    if not meaningful:
        meaningful = stem_tokens
    object_name = normalize_sprite_id("_".join(meaningful[:8]))
    if not object_name or len(object_name) < 2:
        return "unknown", "", all_tokens, False
    return category, object_name, all_tokens, False


def _source_record_from_png(
    *,
    relative: str,
    path: Path,
    content: bytes,
    source: Mapping[str, Any],
    source_group_identity: str | None = None,
    derivation: Mapping[str, Any] | None = None,
) -> tuple[_SourceRecord | None, str | None]:
    digest = hashlib.sha256(content).hexdigest()
    try:
        with Image.open(io.BytesIO(content)) as opened:
            if opened.format != "PNG" or getattr(opened, "n_frames", 1) != 1 or opened.size != (32, 32):
                return None, "not_exact_32x32_png"
            opened.load()
            rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8)
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError):
        return None, "invalid_png"
    alpha = rgba[:, :, 3]
    if not np.all(np.isin(alpha, (0, 255))):
        return None, "soft_alpha"
    if not np.any(alpha == 255):
        return None, "fully_transparent"
    visible_colors = np.unique(rgba[alpha == 255, :3], axis=0)
    if len(visible_colors) > 31:
        return None, "over_palette"
    category, object_name, tokens, disagreement = _infer_semantics(relative)
    if disagreement:
        return None, "taxonomy_disagreement"
    if category == "unknown" or not object_name:
        return None, "unknown_or_low_confidence"
    visual_descriptor = _local_pixel_vision(rgba)
    bbox = tuple(int(value) for value in visual_descriptor["metrics"]["alpha_bbox"])
    if len(bbox) != 4:
        raise ConditionedDatasetError("local_visual_bbox", "Local pixel vision produced an invalid alpha box.")
    if source_group_identity is not None and SHA256_PATTERN.fullmatch(source_group_identity) is None:
        raise ConditionedDatasetError(
            "derived_source_group", "A receipt-bound derived frame has an invalid parent source group."
        )
    return (
        _SourceRecord(
            relative_path=relative,
            path=path,
            byte_count=len(content),
            byte_sha256=digest,
            pixel_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
            alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),
            alpha_bitmap=np.packbits(alpha == 255).tobytes(),
            alpha_bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            perceptual_hash=_perceptual_hash(rgba),
            category=category,
            object_name=object_name,
            tokens=tokens,
            source_id=str(source["source_id"]),
            source_title=str(source["source_title"]),
            creator=str(source["creator"]),
            license_id=str(source["license_id"]),
            license_evidence=dict(source["license_evidence"]),
            visual_descriptor=visual_descriptor,
            visual_tags=tuple(str(value) for value in visual_descriptor["visual_tags"]),
            content=content,
            source_group_identity=source_group_identity,
            derivation=dict(derivation) if derivation is not None else None,
        ),
        None,
    )


def _local_pixel_vision(rgba: np.ndarray) -> dict[str, Any]:
    """Describe exact RGBA geometry/color without inventing semantic category."""

    pixels = np.asarray(rgba, dtype=np.uint8)
    if pixels.shape != (32, 32, 4):
        raise ConditionedDatasetError("local_visual_shape", "Local pixel vision requires exact 32x32 RGBA input.")
    mask = pixels[:, :, 3] == 255
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ConditionedDatasetError("local_visual_blank", "Local pixel vision cannot describe a blank sprite.")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bbox_width, bbox_height = right - left, bottom - top
    foreground_pixels = int(mask.sum())
    occupancy_bp = (foreground_pixels * 10_000 + 512) // 1024

    palette = LOCAL_PIXEL_VISION_CONFIG["dominant_palette"]
    prototypes = np.asarray([entry[1] for entry in palette], dtype=np.int32)
    visible_rgb = pixels[mask, :3].astype(np.int32)
    distances = ((visible_rgb[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    assignments = np.argmin(distances, axis=1)
    color_counts = np.bincount(assignments, minlength=len(palette))
    dominant_index = int(np.argmax(color_counts))
    dominant_color = str(palette[dominant_index][0])
    dominant_color_share_bp = (int(color_counts[dominant_index]) * 10_000 + foreground_pixels // 2) // foreground_pixels

    bbox_max = max(bbox_width, bbox_height)
    scale_thresholds = LOCAL_PIXEL_VISION_CONFIG["scale_max_bbox_dimension"]
    if bbox_max <= int(scale_thresholds["tiny"]):
        silhouette_scale = "tiny"
    elif bbox_max <= int(scale_thresholds["small"]):
        silhouette_scale = "small"
    elif bbox_max <= int(scale_thresholds["medium"]):
        silhouette_scale = "medium"
    elif bbox_max <= int(scale_thresholds["large"]):
        silhouette_scale = "large"
    else:
        silhouette_scale = "full_canvas"

    horizontal_offset_half_pixels = left + right - 32
    vertical_offset_half_pixels = top + bottom - 32
    horizontal_position = (
        "horizontally_centered"
        if abs(horizontal_offset_half_pixels) <= 2
        else "left_weighted"
        if horizontal_offset_half_pixels < 0
        else "right_weighted"
    )
    vertical_position = (
        "vertically_centered"
        if abs(vertical_offset_half_pixels) <= 2
        else "top_weighted"
        if vertical_offset_half_pixels < 0
        else "bottom_weighted"
    )

    symmetry_mismatch_pixels = int(np.count_nonzero(mask != np.fliplr(mask)))
    symmetry_mismatch_bp = (symmetry_mismatch_pixels * 10_000 + 512) // 1024
    symmetry_thresholds = LOCAL_PIXEL_VISION_CONFIG["symmetry_mismatch_basis_points"]
    symmetry = (
        "high_horizontal_symmetry"
        if symmetry_mismatch_bp <= int(symmetry_thresholds["high"])
        else "moderate_horizontal_symmetry"
        if symmetry_mismatch_bp <= int(symmetry_thresholds["moderate"])
        else "asymmetric_silhouette"
    )

    padded = np.pad(mask, 1, constant_values=False)
    interior = padded[1:-1, 1:-1]
    fully_surrounded = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary_pixels = int(np.count_nonzero(interior & ~fully_surrounded))
    edge_density_bp = (boundary_pixels * 10_000 + foreground_pixels // 2) // foreground_pixels
    edge_thresholds = LOCAL_PIXEL_VISION_CONFIG["edge_density_basis_points"]
    edge_density = (
        "low_edge_density"
        if edge_density_bp <= int(edge_thresholds["low"])
        else "medium_edge_density"
        if edge_density_bp <= int(edge_thresholds["medium"])
        else "high_edge_density"
    )
    occupancy = (
        "sparse_occupancy"
        if occupancy_bp < 1500
        else "balanced_occupancy"
        if occupancy_bp < 6000
        else "dense_occupancy"
    )
    visual_tags = (
        f"dominant_{dominant_color}",
        f"{silhouette_scale}_silhouette",
        horizontal_position,
        vertical_position,
        symmetry,
        edge_density,
        occupancy,
    )
    metrics = {
        "alpha_bbox": [left, top, right, bottom],
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "foreground_pixels": foreground_pixels,
        "alpha_occupancy_basis_points": occupancy_bp,
        "dominant_coarse_color": dominant_color,
        "dominant_color_share_basis_points": dominant_color_share_bp,
        "silhouette_scale": silhouette_scale,
        "horizontal_offset_half_pixels": horizontal_offset_half_pixels,
        "vertical_offset_half_pixels": vertical_offset_half_pixels,
        "horizontal_position": horizontal_position,
        "vertical_position": vertical_position,
        "horizontal_symmetry_mismatch_pixels": symmetry_mismatch_pixels,
        "horizontal_symmetry_mismatch_basis_points": symmetry_mismatch_bp,
        "horizontal_symmetry": symmetry,
        "boundary_pixels": boundary_pixels,
        "edge_density_basis_points": edge_density_bp,
        "edge_density": edge_density,
        "occupancy": occupancy,
    }
    payload = {
        "schema_version": "spritelab.dataset.local-pixel-vision.v1",
        "algorithm_id": LOCAL_PIXEL_VISION_ALGORITHM,
        "config_identity": LOCAL_PIXEL_VISION_CONFIG_IDENTITY,
        "decoded_rgba_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "metrics": metrics,
        "visual_tags": list(visual_tags),
        "semantic_category_inferred": False,
        "provider_contacted": False,
        "model_weights_loaded": False,
    }
    return {**payload, "descriptor_identity": stable_hash(payload)}


def _visual_short_description(record: _SourceRecord) -> str:
    metrics = record.visual_descriptor["metrics"]
    object_text = record.object_name.replace("_", " ")
    color = str(metrics["dominant_coarse_color"])
    scale = str(metrics["silhouette_scale"]).replace("_", " ")
    horizontal = str(metrics["horizontal_position"]).replace("_", " ")
    symmetry = str(metrics["horizontal_symmetry"]).replace("_", " ")
    return f"{object_text}; {color}, {scale}, {horizontal}, {symmetry}."


def _perceptual_hash(rgba: np.ndarray) -> int:
    image = (
        Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").convert("L").resize((9, 8), Image.Resampling.BOX)
    )
    values = np.asarray(image, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _record_key(record: _SourceRecord) -> str:
    return f"{record.source_id}:{record.relative_path}"


def _conditioned_source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_reference": source["dataset_reference"],
        "harvest_run_id": source["run_id"],
        "handoff_identity": source["handoff_identity"],
        "harvest_import_receipt_identity": source["harvest_import_receipt_identity"],
        "managed_intake_receipt_identity": source["managed_intake_receipt_identity"],
        "managed_source_inventory_sha256": source["managed_source_inventory_sha256"],
        "managed_output_inventory_sha256": source["managed_output_inventory_sha256"],
        "managed_derived_inventory_sha256": source["managed_derived_inventory_sha256"],
        "derived_sheet_manifest_identity": source["derived_sheet_manifest_identity"],
        "trusted_catalog_identity": source["trusted_catalog_identity"],
        "source_catalog_identity": source["source_catalog_identity"],
        "backend_capability_identity": source["backend_capability_identity"],
        "backend_capability_evidence_identity": source["backend_capability_evidence_identity"],
        "backend_certificate_identity": source["backend_certificate_identity"],
        "backend_audit_report_sha256": source["backend_audit_report_sha256"],
        "backend_audit_report_identity": source["backend_audit_report_identity"],
        "backend_capability_issued_at": source["backend_capability_issued_at"],
        "backend_capability_expires_at": source["backend_capability_expires_at"],
        "authorization_receipt_identity": source["authorization_receipt_identity"],
        "acquisition_receipt_identity": source["acquisition_receipt_identity"],
        "artifact_manifest_sha256": source["artifact_manifest_sha256"],
        "artifact_set_identity": source["artifact_manifest"]["artifact_set_identity"],
        "source_id": source["source_id"],
        "title": source["source_title"],
        "creator": source["creator"],
        "license_id": source["license_id"],
        "license_evidence": dict(source["license_evidence"]),
        "source_document": dict(source["handoff"]["source"]),
        "license_document": dict(source["handoff"]["license"]),
    }


def _deduplicate_records(
    records: Sequence[_SourceRecord],
) -> tuple[list[_SourceRecord], list[str], list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda record: (record.source_id, record.relative_path))
    byte_seen: set[str] = set()
    pixel_seen: set[str] = set()
    kept: list[_SourceRecord] = []
    kept_by_category: dict[str, list[_SourceRecord]] = defaultdict(list)
    exclusions: list[str] = []
    near_exclusions: list[dict[str, Any]] = []
    for record in ordered:
        if record.byte_sha256 in byte_seen:
            exclusions.append("exact_byte_duplicate")
            continue
        byte_seen.add(record.byte_sha256)
        if record.pixel_sha256 in pixel_seen:
            exclusions.append("exact_pixel_duplicate")
            continue
        pixel_seen.add(record.pixel_sha256)
        near_match: tuple[_SourceRecord, dict[str, Any]] | None = None
        for representative in kept_by_category[record.category]:
            metrics = _near_duplicate_metrics(representative, record)
            if metrics["is_near_duplicate"] is True:
                near_match = representative, metrics
                break
        if near_match is not None:
            representative, metrics = near_match
            exclusions.append("near_duplicate")
            pair_keys = sorted((_record_key(representative), _record_key(record)))
            near_exclusions.append(
                {
                    "disposition": "near_duplicate",
                    "family_id": hashlib.sha256("\n".join(pair_keys).encode("utf-8")).hexdigest()[:24],
                    "retained": {
                        "source_id": representative.source_id,
                        "relative_path": representative.relative_path,
                        "pixel_sha256": representative.pixel_sha256,
                    },
                    "excluded": {
                        "source_id": record.source_id,
                        "relative_path": record.relative_path,
                        "pixel_sha256": record.pixel_sha256,
                    },
                    "metric_evidence": metrics,
                }
            )
            continue
        kept.append(record)
        kept_by_category[record.category].append(record)
    return kept, exclusions, near_exclusions


def _near_duplicate_metrics(left: _SourceRecord, right: _SourceRecord) -> dict[str, Any]:
    left_bbox = left.alpha_bbox
    right_bbox = right.alpha_bbox
    width_delta = abs((left_bbox[2] - left_bbox[0]) - (right_bbox[2] - right_bbox[0]))
    height_delta = abs((left_bbox[3] - left_bbox[1]) - (right_bbox[3] - right_bbox[1]))
    center_x_delta_half_pixels = abs((left_bbox[0] + left_bbox[2]) - (right_bbox[0] + right_bbox[2]))
    center_y_delta_half_pixels = abs((left_bbox[1] + left_bbox[3]) - (right_bbox[1] + right_bbox[3]))
    alpha_xor_pixels = sum(
        byte.bit_count() for byte in bytes(a ^ b for a, b in zip(left.alpha_bitmap, right.alpha_bitmap, strict=True))
    )
    perceptual_hamming = (left.perceptual_hash ^ right.perceptual_hash).bit_count()
    same_category = left.category == right.category
    is_near = (
        same_category
        and perceptual_hamming <= int(NEAR_DUPLICATE_CONFIG["max_perceptual_hamming"])
        and width_delta <= int(NEAR_DUPLICATE_CONFIG["max_bbox_dimension_delta"])
        and height_delta <= int(NEAR_DUPLICATE_CONFIG["max_bbox_dimension_delta"])
        and center_x_delta_half_pixels <= int(NEAR_DUPLICATE_CONFIG["max_bbox_center_delta_half_pixels"])
        and center_y_delta_half_pixels <= int(NEAR_DUPLICATE_CONFIG["max_bbox_center_delta_half_pixels"])
        and alpha_xor_pixels <= int(NEAR_DUPLICATE_CONFIG["max_alpha_xor_pixels"])
    )
    return {
        "algorithm_id": NEAR_DUPLICATE_ALGORITHM,
        "config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
        "same_taxonomy_category": same_category,
        "perceptual_hamming": perceptual_hamming,
        "bbox_width_delta": width_delta,
        "bbox_height_delta": height_delta,
        "bbox_center_x_delta_half_pixels": center_x_delta_half_pixels,
        "bbox_center_y_delta_half_pixels": center_y_delta_half_pixels,
        "alpha_xor_pixels": alpha_xor_pixels,
        "is_near_duplicate": is_near,
    }


def _retained_near_duplicate_gate(records: Sequence[_SourceRecord]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    ordered = sorted(records, key=lambda record: (_record_key(record), record.pixel_sha256))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.category != right.category:
                continue
            metrics = _near_duplicate_metrics(left, right)
            if metrics["is_near_duplicate"] is True:
                violations.append(
                    {
                        "left_record_key": _record_key(left),
                        "right_record_key": _record_key(right),
                        "metric_evidence": metrics,
                    }
                )
    payload = {
        "algorithm_id": NEAR_DUPLICATE_ALGORITHM,
        "config": NEAR_DUPLICATE_CONFIG,
        "config_identity": NEAR_DUPLICATE_CONFIG_IDENTITY,
        "retained_count": len(ordered),
        "violation_count": len(violations),
        "violations": violations,
        "ok": not violations,
    }
    return {**payload, "gate_identity": stable_hash(payload)}


def _representative_selection(records: Sequence[_SourceRecord], target: int) -> list[_SourceRecord]:
    by_source: dict[str, list[_SourceRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_id].append(record)
    if len(records) <= target:
        return sorted(records, key=_selection_key)
    sources = sorted(by_source)
    base_quota, remainder = divmod(target, len(sources))
    selected: list[_SourceRecord] = []
    leftovers: list[_SourceRecord] = []
    for index, source_id in enumerate(sources):
        quota = base_quota + (1 if index < remainder else 0)
        chosen, remaining = _category_round_robin(by_source[source_id], quota)
        selected.extend(chosen)
        leftovers.extend(remaining)
    if len(selected) < target:
        selected.extend(sorted(leftovers, key=_selection_key)[: target - len(selected)])
    return sorted(selected[:target], key=lambda record: (record.source_id, record.relative_path))


def _category_round_robin(
    records: Sequence[_SourceRecord], quota: int
) -> tuple[list[_SourceRecord], list[_SourceRecord]]:
    buckets: dict[str, list[_SourceRecord]] = defaultdict(list)
    for record in records:
        buckets[record.category].append(record)
    for values in buckets.values():
        values.sort(key=_selection_key)
    categories = sorted(buckets)
    selected: list[_SourceRecord] = []
    while len(selected) < quota and categories:
        next_categories: list[str] = []
        for category in categories:
            if len(selected) >= quota:
                next_categories.append(category)
                continue
            values = buckets[category]
            if values:
                selected.append(values.pop(0))
            if values:
                next_categories.append(category)
        categories = next_categories
    remaining = [record for values in buckets.values() for record in values]
    return selected, remaining


def _selection_key(record: _SourceRecord) -> str:
    return hashlib.sha256(
        f"conditioned-v5:{record.source_id}:{record.relative_path}:{record.byte_sha256}".encode()
    ).hexdigest()


def _source_group(record: _SourceRecord) -> str:
    if record.source_group_identity is not None:
        if SHA256_PATTERN.fullmatch(record.source_group_identity) is None:
            raise ConditionedDatasetError(
                "derived_source_group", "A receipt-bound derived frame has an invalid parent source group."
            )
        return record.source_group_identity
    parent = PurePosixPath(record.relative_path).parent.as_posix()
    family_stem = re.sub(
        r"(?:[_-](?:north|south|east|west|up|down|left|right|front|back|alt|[0-9]+))+$",
        "",
        PurePosixPath(record.relative_path).stem.casefold(),
    )
    raw = f"{record.source_id}:{parent}:{family_stem or record.object_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class _UnionFind:
    def __init__(self, keys: Sequence[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _family_assignments(records: Sequence[_SourceRecord]) -> dict[str, str]:
    keys = [_record_key(record) for record in records]
    union = _UnionFind(keys)
    by_source_group: dict[str, list[_SourceRecord]] = defaultdict(list)
    by_category: dict[str, list[_SourceRecord]] = defaultdict(list)
    for record in records:
        by_source_group[_source_group(record)].append(record)
        by_category[record.category].append(record)
    for members in by_source_group.values():
        for record in members[1:]:
            union.union(_record_key(members[0]), _record_key(record))
    for members in by_category.values():
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if _variant_family_match(left, right):
                    union.union(_record_key(left), _record_key(right))
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        grouped[union.find(key)].append(key)
    family_by_key: dict[str, str] = {}
    for members in grouped.values():
        family_id = hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:24]
        for key in members:
            family_by_key[key] = family_id
    return family_by_key


def _variant_family_match(left: _SourceRecord, right: _SourceRecord) -> bool:
    if left.category != right.category:
        return False
    left_bbox, right_bbox = left.alpha_bbox, right.alpha_bbox
    return (
        (left.perceptual_hash ^ right.perceptual_hash).bit_count()
        <= int(VARIANT_FAMILY_CONFIG["max_perceptual_hamming"])
        and abs((left_bbox[2] - left_bbox[0]) - (right_bbox[2] - right_bbox[0]))
        <= int(VARIANT_FAMILY_CONFIG["max_bbox_dimension_delta"])
        and abs((left_bbox[3] - left_bbox[1]) - (right_bbox[3] - right_bbox[1]))
        <= int(VARIANT_FAMILY_CONFIG["max_bbox_dimension_delta"])
        and abs((left_bbox[0] + left_bbox[2]) - (right_bbox[0] + right_bbox[2]))
        <= int(VARIANT_FAMILY_CONFIG["max_bbox_center_delta_half_pixels"])
        and abs((left_bbox[1] + left_bbox[3]) - (right_bbox[1] + right_bbox[3]))
        <= int(VARIANT_FAMILY_CONFIG["max_bbox_center_delta_half_pixels"])
        and sum(
            byte.bit_count()
            for byte in bytes(a ^ b for a, b in zip(left.alpha_bitmap, right.alpha_bitmap, strict=True))
        )
        <= int(VARIANT_FAMILY_CONFIG["max_alpha_xor_pixels"])
    )


def _split_assignments(records: Sequence[_SourceRecord], family_by_key: Mapping[str, str]) -> dict[str, str]:
    groups: dict[str, list[_SourceRecord]] = defaultdict(list)
    for record in records:
        groups[family_by_key[_record_key(record)]].append(record)
    total = len(records)
    targets = {
        "train": round(total * 0.8),
        "val": round(total * 0.1),
        "test": total - round(total * 0.8) - round(total * 0.1),
    }
    counts: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode()).hexdigest()))
    for _family_id, members in ordered:
        split = min(
            ("train", "val", "test"),
            key=lambda name: (counts[name] / max(1, targets[name]), {"train": 0, "val": 1, "test": 2}[name]),
        )
        for record in members:
            assignments[_record_key(record)] = split
        counts[split] += len(members)
    if any(counts[name] == 0 for name in ("train", "val", "test")):
        raise ConditionedDatasetError(
            "split_empty", "Whole-family grouping could not produce non-empty train, validation, and test splits."
        )
    return assignments


def _sprite_id(record: _SourceRecord) -> str:
    prefix = normalize_sprite_id(record.object_name)[:36] or record.category
    digest = hashlib.sha256(f"{record.source_id}:{record.relative_path}:{record.byte_sha256}".encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _conditioned_record(item: ImportedSprite, metadata: Mapping[str, Any]) -> dict[str, Any]:
    prefill = item.auto_metadata.get("label_v2_safe_prefill")
    semantic = item.auto_metadata.get("semantic_v3")
    if not isinstance(prefill, Mapping) or not isinstance(semantic, Mapping):
        raise ConditionedDatasetError("conditioned_label_contract", "A conditioned label contract is incomplete.")
    label_contract = {
        "schema_version": "spritelab.dataset.conditioned-label-contract.v1",
        "category": item.item.category,
        "object_name": str(prefill.get("object_name") or ""),
        "tags": [str(value) for value in item.item.tags],
        "short_description": str(prefill.get("short_description") or ""),
        "confidence": str(prefill.get("confidence") or ""),
        "confidence_reason": str(prefill.get("confidence_reason") or ""),
        "captions": [str(value) for value in semantic.get("captions") or ()],
        "prompt_phrases": [str(value) for value in semantic.get("prompt_phrases") or ()],
        "negative_tags": [str(value) for value in semantic.get("negative_tags") or ()],
        "disagreement": bool(dict(metadata["label_evidence"]).get("disagreement")),
        "audit_state": "SOURCE_GROUNDED_REQUIRES_INDEPENDENT_AUDIT",
        "human_truth_claim": False,
    }
    return {
        "sprite_id": item.item.sprite_id,
        "split": item.item.split,
        "category": item.item.category,
        "object_name": label_contract["object_name"],
        **dict(metadata),
        "label_contract": label_contract,
        "semantic_v3": dict(semantic),
    }


def _enrich_manifests(
    dataset: Path,
    anchor: AnchoredDirectory,
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    for split in ("train", "val", "test"):
        name = f"manifest_{split}.jsonl"
        rows = _read_anchored_jsonl(anchor, name)
        enriched: list[dict[str, Any]] = []
        for row in rows:
            sprite_id = str(row.get("sprite_id") or "")
            source = metadata.get(sprite_id)
            if source is None:
                raise ConditionedDatasetError(
                    "manifest_source_binding", "An exported manifest row lacks its source identity binding."
                )
            updated = dict(row)
            updated.update(
                {
                    "source_path": (
                        f"managed/{source['source_id']}/{source['source_storage_kind']}/"
                        f"{source['source_relative_path']}"
                    ),
                    "source_name": source["source_relative_path"],
                    "source_id": source["source_id"],
                    "source_pack": source["source_pack"],
                    "source_group": source["source_group"],
                    "source_sha256": source["source_sha256"],
                    "source_byte_count": source["source_byte_count"],
                    "license": source["license_id"],
                    "author": source["creator"],
                    "duplicate_family_id": source["duplicate_family_id"],
                    "provenance": {
                        "source_id": source["source_id"],
                        "source_pack": source["source_pack"],
                        "source_group": source["source_group"],
                        "relative_path": source["source_relative_path"],
                        "sha256": source["source_sha256"],
                        "byte_count": source["source_byte_count"],
                        "license_id": source["license_id"],
                        "creator": source["creator"],
                    },
                    "label_evidence": dict(source["label_evidence"]),
                    "source_derivation": (
                        dict(source["source_derivation"])
                        if isinstance(source.get("source_derivation"), Mapping)
                        else None
                    ),
                    "human_truth_claim": False,
                }
            )
            enriched.append(updated)
        _write_jsonl(anchor, name, enriched)
    anchor.verify()
    if anchor.directory != dataset:
        raise ConditionedDatasetError("candidate_root_changed", "The candidate publication root changed.")


def _portable_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    portable: list[dict[str, Any]] = []
    for raw in rows:
        row = json.loads(strict_json_dumps(dict(raw), sort_keys=True))
        source = row.get("source")
        if isinstance(source, dict):
            source["dataset_dir"] = "."
            source["inference_path"] = ""
        portable.append(row)
    return portable


def _conditioning_vocabulary(imported: Sequence[ImportedSprite], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories = sorted({item.item.category for item in imported})
    objects = sorted(
        {
            str(item.auto_metadata.get("label_v2_safe_prefill", {}).get("object_name") or "")
            for item in imported
            if str(item.auto_metadata.get("label_v2_safe_prefill", {}).get("object_name") or "")
        }
    )
    captions = sorted({str(row.get("caption") or "") for row in rows if str(row.get("caption") or "")})
    return {
        "schema_version": "spritelab.dataset.conditioning-vocabulary.v1",
        "category_to_id": {value: index for index, value in enumerate(categories)},
        "object_to_id": {value: index for index, value in enumerate(objects)},
        "caption_count": len(captions),
        "negative_tags": list(DEFAULT_NEGATIVE_TAGS),
        "human_truth_claim": False,
    }


def _split_integrity(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    family_splits: dict[str, set[str]] = defaultdict(set)
    source_group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_splits[str(row.get("duplicate_family_id") or "")].add(str(row.get("split") or ""))
        source_group_splits[str(row.get("source_group") or "")].add(str(row.get("split") or ""))
    family_leaks = sorted(key for key, splits in family_splits.items() if key and len(splits) > 1)
    source_group_leaks = sorted(key for key, splits in source_group_splits.items() if key and len(splits) > 1)
    return {
        "schema_version": "spritelab.dataset.conditioned-split-integrity.v1",
        "ok": not family_leaks and not source_group_leaks,
        "duplicate_family_count": len(family_splits),
        "source_group_count": len(source_group_splits),
        "cross_split_duplicate_families": family_leaks,
        "cross_split_source_groups": source_group_leaks,
        "whole_family_assignment": True,
    }


def _coverage_report(imported: Sequence[ImportedSprite], exclusions: Sequence[str]) -> dict[str, Any]:
    category_counts = Counter(item.item.category for item in imported)
    source_counts = Counter(str(item.auto_metadata["conditioned_v5"]["source_id"]) for item in imported)
    split_counts = Counter(str(item.item.split) for item in imported)
    category_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in imported:
        category_source_counts[item.item.category][str(item.auto_metadata["conditioned_v5"]["source_id"])] += 1
    return {
        "schema_version": "spritelab.dataset.conditioned-coverage.v1",
        "image_count": len(imported),
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "category_source_counts": {
            category: dict(sorted(values.items())) for category, values in sorted(category_source_counts.items())
        },
        "excluded_counts": dict(sorted(Counter(exclusions).items())),
        "taxonomy": list(TAXONOMY),
        "unknown_included": 0,
        "taxonomy_disagreement_included": 0,
        "labels_are_human_truth": False,
        "representative_selection": "equal source quotas with deterministic per-source category round-robin",
    }


def _label_audit_subjects(imported: Sequence[ImportedSprite]) -> dict[str, Any]:
    high_impact_categories = frozenset({"character", "creature", "weapon", "armor", "vehicle", "effect"})
    generic_objects = frozenset({"sprite", "pixel_art_sprite", "item", "object", "asset", "unknown"})
    by_category: dict[str, list[str]] = defaultdict(list)
    low_confidence: list[str] = []
    disagreements: list[str] = []
    high_impact: list[str] = []
    generic: list[str] = []
    source_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    confidence_reason_counts: Counter[str] = Counter()
    disagreement_counts: Counter[str] = Counter()
    generic_counts: Counter[str] = Counter()
    visual_bindings: list[dict[str, Any]] = []
    for item in sorted(imported, key=lambda value: value.item.sprite_id):
        sprite_id = item.item.sprite_id
        category = item.item.category
        prefill = item.auto_metadata.get("label_v2_safe_prefill", {})
        evidence = item.auto_metadata.get("conditioned_v5", {})
        confidence = str(prefill.get("confidence") or "unknown")
        confidence_reason = str(prefill.get("confidence_reason") or "unknown")
        object_name = str(prefill.get("object_name") or "").casefold()
        disagreed = bool(evidence.get("disagreement"))
        is_generic = object_name in generic_objects
        visual = item.auto_metadata.get("local_pixel_vision", {})
        descriptor_identity = str(visual.get("descriptor_identity") or "") if isinstance(visual, Mapping) else ""
        decoded_identity = str(visual.get("decoded_rgba_sha256") or "") if isinstance(visual, Mapping) else ""
        if not SHA256_PATTERN.fullmatch(descriptor_identity) or not SHA256_PATTERN.fullmatch(decoded_identity):
            raise ConditionedDatasetError(
                "local_visual_binding",
                "A conditioned label lacks an exact local pixel descriptor binding.",
            )
        visual_bindings.append(
            {
                "sprite_id": sprite_id,
                "descriptor_identity": descriptor_identity,
                "decoded_rgba_sha256": decoded_identity,
            }
        )
        by_category[category].append(sprite_id)
        source_counts[str(evidence.get("source_id") or "unknown")] += 1
        confidence_counts[confidence] += 1
        confidence_reason_counts[confidence_reason] += 1
        disagreement_counts["disagreement" if disagreed else "no_disagreement"] += 1
        generic_counts["generic" if is_generic else "specific"] += 1
        if confidence in {"source_grounded_low", "low", "unknown"}:
            low_confidence.append(sprite_id)
        if disagreed:
            disagreements.append(sprite_id)
        if category in high_impact_categories:
            high_impact.append(sprite_id)
        if is_generic:
            generic.append(sprite_id)
    stratified = sorted(
        sprite_id for category in sorted(by_category) for sprite_id in sorted(by_category[category])[:10]
    )
    required = sorted({*stratified, *low_confidence, *disagreements, *high_impact, *generic})
    base = {
        "schema_version": AUDIT_SUBJECTS_SCHEMA,
        "stratified_sample_ids": stratified,
        "low_confidence_ids": sorted(low_confidence),
        "disagreement_ids": sorted(disagreements),
        "high_impact_ids": sorted(high_impact),
        "generic_label_ids": sorted(generic),
        "required_label_audit_ids": required,
        "visual_descriptor_bindings": visual_bindings,
        "local_pixel_vision_algorithm": LOCAL_PIXEL_VISION_ALGORITHM,
        "local_pixel_vision_config_identity": LOCAL_PIXEL_VISION_CONFIG_IDENTITY,
        "distributions": {
            "category": {key: len(by_category[key]) for key in sorted(by_category)},
            "source": dict(sorted(source_counts.items())),
            "confidence": dict(sorted(confidence_counts.items())),
            "confidence_reason": dict(sorted(confidence_reason_counts.items())),
            "disagreement": dict(sorted(disagreement_counts.items())),
            "generic_label": dict(sorted(generic_counts.items())),
        },
        "all_low_confidence_required": True,
        "all_disagreements_required": True,
        "all_high_impact_required": True,
        "all_generic_labels_required": True,
        "all_visual_descriptors_recompute_required": True,
        "quality_rates_basis_points": {
            "unknown_category": 0,
            "generic_object": (len(generic) * 10_000 + max(1, len(imported)) // 2) // max(1, len(imported)),
            "disagreement": (len(disagreements) * 10_000 + max(1, len(imported)) // 2) // max(1, len(imported)),
            "useful_label": ((len(imported) - len(generic)) * 10_000 + max(1, len(imported)) // 2)
            // max(1, len(imported)),
        },
        "human_truth_claim": False,
    }
    return {**base, "subjects_identity": stable_hash(base)}


def _benchmark_manifest(
    imported: Sequence[ImportedSprite],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    train_groups = {
        str(metadata[item.item.sprite_id]["source_group"]) for item in imported if item.item.split == "train"
    }
    train_families = {
        str(metadata[item.item.sprite_id]["duplicate_family_id"]) for item in imported if item.item.split == "train"
    }
    buckets: dict[str, list[ImportedSprite]] = defaultdict(list)
    for item in imported:
        source = metadata[item.item.sprite_id]
        if (
            item.item.split in {"val", "test"}
            and str(source["source_group"]) not in train_groups
            and str(source["duplicate_family_id"]) not in train_families
        ):
            buckets[item.item.category].append(item)
    for values in buckets.values():
        values.sort(key=lambda item: hashlib.sha256(f"benchmark:{item.item.sprite_id}".encode()).hexdigest())
    seen_groups: set[str] = set()
    seen_families: set[str] = set()
    selected: list[dict[str, Any]] = []
    active = sorted(buckets)
    while active and len(selected) < 128:
        next_active: list[str] = []
        for category in active:
            values = buckets[category]
            while values:
                item = values.pop(0)
                source = metadata[item.item.sprite_id]
                group = str(source["source_group"])
                family = str(source["duplicate_family_id"])
                if group in seen_groups or family in seen_families:
                    continue
                seen_groups.add(group)
                seen_families.add(family)
                selected.append(
                    {
                        "sprite_id": item.item.sprite_id,
                        "split": item.item.split,
                        "category": category,
                        "source_id": source["source_id"],
                        "source_group": group,
                        "duplicate_family_id": family,
                        "source_sha256": source["source_sha256"],
                    }
                )
                break
            if values:
                next_active.append(category)
            if len(selected) >= 128:
                break
        active = next_active
    if not selected:
        raise ConditionedDatasetError(
            "benchmark_empty", "No source-group-disjoint validation/test benchmark could be built."
        )
    selected_groups = {str(item["source_group"]) for item in selected}
    selected_families = {str(item["duplicate_family_id"]) for item in selected}
    candidate_categories = set(buckets) | {str(item["category"]) for item in selected}
    category_counts = Counter(str(item["category"]) for item in selected)
    disjoint = not (selected_groups & train_groups) and not (selected_families & train_families)
    if not disjoint or set(category_counts) != candidate_categories:
        raise ConditionedDatasetError(
            "benchmark_integrity", "The category-stratified benchmark failed recomputed training-group disjointness."
        )
    return {
        "schema_version": "spritelab.dataset.conditioned-benchmark.v1",
        "selection_policy": "deterministic category round-robin with one record per source_group and family",
        "category_stratified": True,
        "category_counts": dict(sorted(category_counts.items())),
        "source_group_disjoint_from_training": disjoint,
        "duplicate_family_disjoint_from_training": disjoint,
        "training_source_group_count": len(train_groups),
        "training_duplicate_family_count": len(train_families),
        "record_count": len(selected),
        "records": selected,
    }


def _loader_check(dataset: Path) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    errors: list[str] = []
    for split in ("train", "val", "test"):
        manifest_rows = _read_jsonl(dataset / f"manifest_{split}.jsonl")
        try:
            with np.load(dataset / f"{split}.npz", allow_pickle=False) as arrays:
                keys = set(arrays.files)
                required = {"alpha", "index_map", "role_map", "palette", "palette_mask", "category_id", "sprite_id"}
                if keys != required:
                    errors.append(f"{split}: array keys differ from Phase-7 contract")
                sprite_ids = [str(value) for value in np.asarray(arrays["sprite_id"])]
                if np.asarray(arrays["alpha"]).shape != (len(sprite_ids), 32, 32):
                    errors.append(f"{split}: alpha shape mismatch")
                if np.asarray(arrays["index_map"]).shape != (len(sprite_ids), 32, 32):
                    errors.append(f"{split}: index-map shape mismatch")
        except (OSError, ValueError, KeyError):
            errors.append(f"{split}: npz unreadable")
            sprite_ids = []
        manifest_ids = [str(row.get("sprite_id") or "") for row in manifest_rows]
        if manifest_ids != sorted(manifest_ids) or set(manifest_ids) != set(sprite_ids):
            errors.append(f"{split}: manifest/npz sprite parity failed")
        split_counts[split] = len(sprite_ids)
    vocabulary = _read_json_mapping(dataset / "conditioning_vocabulary.json", max_bytes=16 * 1024 * 1024)
    benchmark = _read_json_mapping(dataset / "benchmark_manifest.json", max_bytes=64 * 1024 * 1024)
    if not vocabulary.get("category_to_id") or not vocabulary.get("object_to_id"):
        errors.append("conditioning vocabulary is empty")
    if not benchmark.get("records") or benchmark.get("source_group_disjoint_from_training") is not True:
        errors.append("benchmark source-group contract failed")
    return {
        "schema_version": "spritelab.dataset.conditioned-loader-check.v1",
        "ok": not errors and all(split_counts.values()),
        "split_counts": split_counts,
        "checked_all_splits": set(split_counts) == {"train", "val", "test"},
        "vocabulary_loaded": bool(vocabulary),
        "benchmark_loaded": bool(benchmark),
        "errors": errors,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ConditionedDatasetError(
            "managed_jsonl_invalid", "A required managed JSONL document is unreadable."
        ) from exc
    return _parse_jsonl_bytes(content)


def _read_anchored_jsonl(anchor: AnchoredDirectory, name: str) -> list[dict[str, Any]]:
    content = _read_anchored_regular_bytes(anchor, name, max_bytes=1024 * 1024 * 1024)
    return _parse_jsonl_bytes(content)


def _parse_jsonl_bytes(content: bytes) -> list[dict[str, Any]]:
    try:
        lines = content.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConditionedDatasetError(
            "managed_jsonl_invalid", "A required managed JSONL document is unreadable."
        ) from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConditionedDatasetError(
                "managed_jsonl_invalid", "A required managed JSONL document is invalid."
            ) from exc
        if not isinstance(value, dict):
            raise ConditionedDatasetError("managed_jsonl_invalid", "A required managed JSONL row is not an object.")
        rows.append(value)
    return rows


def _write_json(anchor: AnchoredDirectory, name: str, value: Mapping[str, Any]) -> None:
    _write_anchored_bytes(
        anchor,
        name,
        (strict_json_dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


@contextmanager
def _open_bound_candidate_phase7(
    candidate_root: Path,
    project_root: Path,
    *,
    candidate_identity: OwnedFileIdentity,
    phase7_identity: OwnedFileIdentity,
) -> Iterator[AnchoredDirectory]:
    try:
        with open_anchored_directory(candidate_root, project_root) as candidate_anchor:
            if not candidate_identity.matches(candidate_anchor.directory_metadata()):
                raise UnsafeFilesystemOperation("conditioned candidate parent identity changed")
            with candidate_anchor.open_directory_immovable("phase7") as phase7_anchor:
                if not phase7_identity.matches(phase7_anchor.directory_metadata()):
                    raise UnsafeFilesystemOperation("conditioned phase7 directory identity changed")
                yield phase7_anchor
                phase7_anchor.verify()
            candidate_anchor.verify()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "candidate_root_changed",
            "The candidate publication root changed or became unsafe.",
        ) from exc


def _write_jsonl(
    anchor: AnchoredDirectory,
    name: str,
    values: Sequence[Mapping[str, Any]],
) -> None:
    lines = [strict_json_dumps(dict(value), sort_keys=True, separators=(",", ":")) for value in values]
    _write_anchored_bytes(anchor, name, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def _write_training_manifest(
    anchor: AnchoredDirectory,
    name: str,
    values: Sequence[Mapping[str, Any]],
) -> None:
    lines = [json.dumps(dict(value), sort_keys=True) for value in values]
    _write_anchored_bytes(anchor, name, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def _write_anchored_bytes(anchor: AnchoredDirectory, name: str, content: bytes) -> None:
    try:
        anchor.verify()
        anchor.atomic_write_bytes(name, content)
        anchor.verify()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "candidate_publication_unsafe",
            "The conditioned candidate publication root changed or became unsafe.",
        ) from exc


def _inventory(root: Path, boundary: Path) -> dict[str, dict[str, Any]]:
    try:
        target = require_confined_path(root, boundary, allow_root=True)
        current = Path(boundary).absolute()
        for part in target.relative_to(current).parts:
            current /= part
            if current.is_mount():
                raise UnsafeFilesystemOperation(f"artifact inventory crosses a mount point: {current}")
        with open_anchored_directory(root, boundary) as anchor:
            return _inventory_from_anchor(anchor)
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "inventory_root", "The artifact inventory root is unsafe or unavailable."
        ) from exc


def _inventory_from_anchor(root: AnchoredDirectory) -> dict[str, dict[str, Any]]:
    try:
        root_metadata = root.directory_metadata()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "inventory_root", "The artifact inventory root is unsafe or unavailable."
        ) from exc
    passes: list[dict[str, dict[str, Any]]] = []
    for _pass in range(2):
        records: dict[str, dict[str, Any]] = {}
        collision_keys: set[str] = set()
        _inventory_directory(
            root,
            relative_parts=(),
            root_device=root_metadata.st_dev,
            records=records,
            collision_keys=collision_keys,
        )
        passes.append(dict(sorted(records.items())))
    if passes[0] != passes[1]:
        raise ConditionedDatasetError(
            "inventory_changed",
            "An artifact changed between complete anchored inventory passes.",
        )
    return passes[0]


def _inventory_directory(
    anchor: AnchoredDirectory,
    *,
    relative_parts: tuple[str, ...],
    root_device: int,
    records: dict[str, dict[str, Any]],
    collision_keys: set[str],
) -> None:
    try:
        anchor.verify()
        directory_before = anchor.directory_metadata()
        before_names = anchor.names()
        retained_aliases = {
            name for name in before_names if _retained_stage_target_for_alias(anchor, name, before_names) is not None
        }
        for name in before_names:
            metadata = anchor.lstat(name)
            if name in retained_aliases:
                anchor.verify()
                continue
            relative = PurePosixPath(*relative_parts, name).as_posix()
            _canonical_relative(relative)
            collision = unicodedata.normalize("NFC", relative).casefold()
            if collision in collision_keys:
                raise ConditionedDatasetError(
                    "inventory_collision", "Artifact paths contain a case or Unicode collision."
                )
            collision_keys.add(collision)
            if stat.S_ISDIR(metadata.st_mode) and not _metadata_is_link_or_reparse(metadata):
                if metadata.st_dev != root_device or (anchor.directory / name).is_mount():
                    raise ConditionedDatasetError(
                        "inventory_entry", "An artifact directory is linked, mounted, or crosses a device."
                    )
                with anchor.open_directory_immovable(name) as child:
                    child_metadata = child.directory_metadata()
                    if (
                        child_metadata.st_dev != root_device
                        or child_metadata.st_ino != metadata.st_ino
                        or child.directory.is_mount()
                    ):
                        raise ConditionedDatasetError(
                            "inventory_entry", "An artifact directory is linked, mounted, or crosses a device."
                        )
                    _inventory_directory(
                        child,
                        relative_parts=(*relative_parts, name),
                        root_device=root_device,
                        records=records,
                        collision_keys=collision_keys,
                    )
            elif stat.S_ISREG(metadata.st_mode) and not _metadata_is_link_or_reparse(metadata):
                records[relative] = _stable_file_identity(anchor, name, root_device)
            else:
                raise ConditionedDatasetError(
                    "inventory_entry", "Artifact inventory entries must be owned regular files or directories."
                )
            anchor.verify()
        directory_after = anchor.directory_metadata()
        if (
            anchor.names() != before_names
            or directory_after.st_dev != directory_before.st_dev
            or directory_after.st_ino != directory_before.st_ino
            or directory_after.st_mtime_ns != directory_before.st_mtime_ns
        ):
            raise ConditionedDatasetError("inventory_changed", "An artifact changed while it was inventoried.")
    except ConditionedDatasetError:
        raise
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "inventory_entry", "An artifact inventory entry changed or became unsafe."
        ) from exc


def _stable_file_identity(anchor: AnchoredDirectory, name: str, root_device: int) -> dict[str, Any]:
    try:
        before = anchor.lstat(name)
    except OSError as exc:
        raise ConditionedDatasetError(
            "inventory_entry", "An artifact inventory entry changed or became unsafe."
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _metadata_is_link_or_reparse(before)
        or before.st_nlink not in {1, 2}
        or before.st_dev != root_device
    ):
        raise ConditionedDatasetError(
            "inventory_entry", "Artifact inventory entries must be owned regular files on one device."
        )
    if before.st_nlink == 2:
        _retained_stage_alias(anchor, name, before)
    digest = hashlib.sha256()
    descriptor = -1
    byte_count = 0
    try:
        descriptor = anchor.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        opened = os.fstat(descriptor)
        _require_inventory_file_identity(before, opened, root_device)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        after = anchor.lstat(name)
        anchor.verify()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "inventory_changed", "An artifact changed while its identity was computed."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        _require_inventory_file_identity(before, opened_after, root_device)
        _require_inventory_file_identity(before, after, root_device)
    except ConditionedDatasetError:
        raise ConditionedDatasetError(
            "inventory_changed", "An artifact changed while its identity was computed."
        ) from None
    if byte_count != before.st_size:
        raise ConditionedDatasetError("inventory_changed", "An artifact changed while its identity was computed.")
    return {"sha256": digest.hexdigest(), "byte_count": byte_count}


def _require_inventory_file_identity(
    expected: os.stat_result,
    actual: os.stat_result,
    root_device: int,
) -> None:
    if (
        not stat.S_ISREG(actual.st_mode)
        or _metadata_is_link_or_reparse(actual)
        or actual.st_nlink != expected.st_nlink
        or expected.st_nlink not in {1, 2}
        or actual.st_dev != root_device
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
        or actual.st_size != expected.st_size
        or actual.st_mtime_ns != expected.st_mtime_ns
    ):
        raise ConditionedDatasetError("inventory_changed", "An artifact changed while its identity was computed.")


def _retained_stage_target_for_alias(
    anchor: AnchoredDirectory,
    alias_name: str,
    names: Sequence[str],
) -> str | None:
    """Recognize only a publication alias that is exactly paired to one target."""

    marker = ".staging-"
    if not alias_name.startswith(".") or marker not in alias_name:
        return None
    target_name, separator, suffix = alias_name[1:].rpartition(marker)
    if separator != marker or re.fullmatch(r"[0-9a-f]{32}", suffix) is None:
        return None
    if target_name not in names:
        raise ConditionedDatasetError(
            "inventory_entry",
            "A retained publication stage has no exact target.",
        )
    target = anchor.lstat(target_name)
    if _retained_stage_alias(anchor, target_name, target) != alias_name:
        raise ConditionedDatasetError(
            "inventory_entry",
            "A retained publication stage is ambiguous.",
        )
    return target_name


def _inventory_payload(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    normalized = {str(path): dict(record) for path, record in sorted(files.items())}
    return {
        "schema_version": INVENTORY_SCHEMA,
        "files": normalized,
        "file_count": len(normalized),
        "total_bytes": sum(int(record["byte_count"]) for record in normalized.values()),
    }


def _inventory_identity(files: Mapping[str, Mapping[str, Any]]) -> str:
    return stable_hash(_inventory_payload(files))


def _project_config_from_bytes(root: Path, path: Path, content: bytes) -> ProjectConfig:
    """Validate one project configuration from the caller's exact held bytes."""

    raw = yaml.safe_load(content.decode("utf-8"))
    values = _validate_project_config(raw)
    return ProjectConfig(root.resolve(), path, values)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ACTIVATION_SCHEMA",
    "CANDIDATE_SCHEMA",
    "DATASET_VALIDATION_GATES",
    "DATASET_VALIDATION_SCHEMA",
    "HANDOFF_SCHEMA",
    "LABEL_AUDIT_GATES",
    "LABEL_AUDIT_SCHEMA",
    "TAXONOMY",
    "CandidatePolicy",
    "ConditionedDatasetError",
    "ConditionedDatasetService",
]
