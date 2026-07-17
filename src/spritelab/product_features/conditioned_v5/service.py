"""Durable, fail-closed conditioned Dataset-v5 candidate and freeze workflow.

The service consumes only opaque, DatasetIntake-backed import receipts.  It is
offline after the explicit Harvest-to-Dataset import: preview, build, evidence
verification, and publication never contact a provider or network endpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import threading
import unicodedata
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from spritelab.dataset_maker.exporter import (
    DatasetMakerExportConfig,
    export_dataset_from_imported_sprites,
)
from spritelab.dataset_maker.importer import (
    ImportedSprite,
    ImportOptions,
    import_png_as_dataset_item,
)
from spritelab.dataset_maker.model import DatasetMakerItem, normalize_sprite_id
from spritelab.dataset_maker.qa import qa_dataset
from spritelab.dataset_maker.training_manifest import (
    build_training_manifest,
    write_training_manifest,
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
from spritelab.product_features.conditioned_v5.intake import (
    ConditionedIntakeError,
    load_managed_intake,
    managed_intake_inventory,
)
from spritelab.product_features.harvest.storage import scan_artifacts
from spritelab.product_features.harvest.trusted_backend import AcquiredFile, HarvestLimits
from spritelab.training.campaign import file_sha256, stable_hash
from spritelab.utils.safe_fs import (
    UnsafeFilesystemOperation,
    atomic_write_bytes,
    atomic_write_text,
    remove_confined_tree,
    require_confined_path,
)

HANDOFF_SCHEMA = "spritelab.harvest.dataset-handoff.v2"
CANDIDATE_SCHEMA = "spritelab.dataset.conditioned-candidate.v1"
ACTIVATION_SCHEMA = "spritelab.dataset.freeze.conditioned.v5"
INVENTORY_SCHEMA = "spritelab.dataset.freeze.inventory.v1"
LABEL_AUDIT_SCHEMA = "spritelab.audit.conditioned-labels.v1"
DATASET_VALIDATION_SCHEMA = "spritelab.audit.conditioned-dataset.v1"

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
    }
)

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
    perceptual_hash: int
    category: str
    object_name: str
    tokens: tuple[str, ...]
    source_id: str
    source_title: str
    creator: str
    license_id: str
    license_evidence: Mapping[str, Any]


HandoffLoader = Callable[[str], Mapping[str, Any]]
CampaignBuilder = Callable[..., Any]


class ConditionedDatasetService:
    """Disk-backed candidate workflow with explicit audit and freeze gates."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        handoff_loader: HandoffLoader | None = None,
        campaign_builder: CampaignBuilder | None = None,
        policy: CandidatePolicy | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.harvest_root = self.project_root / "harvest_runs"
        self.jobs_root = self.project_root / "runs" / "v3" / "conditioned-dataset-v5"
        self.datasets_root = self.project_root / "datasets"
        self.campaigns_root = self.project_root / "campaigns"
        self.policy = policy or CandidatePolicy()
        self._handoff_loader = handoff_loader
        self._campaign_builder = campaign_builder
        self._threads: dict[str, threading.Thread] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()

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
            "network_actions": 0,
            "paths_exposed": False,
        }

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
        records, duplicate_exclusions = _deduplicate_records(records)
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
        with self._lock:
            for existing in self._job_inventory():
                if existing.get("idempotency_key") == idempotency_key:
                    if existing.get("request_identity") != request_identity:
                        raise ConditionedDatasetError(
                            "idempotency_conflict",
                            "That idempotency key already belongs to a different candidate request.",
                        )
                    return existing, False
            # Fail before mutating job state if the source is not a valid completed handoff.
            sources = [self._verified_source(reference) for reference in normalized_ids]
            job_id = f"conditioned-{uuid.uuid4().hex[:20]}"
            self.jobs_root.mkdir(parents=True, exist_ok=True)
            job_root = require_confined_path(self.jobs_root / job_id, self.jobs_root)
            job_root.mkdir()
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
                "publication": None,
                "freeze_authorization": None,
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
        state = self._read_state(root)
        thread = self._threads.get(job_id)
        if state.get("status") == "RUNNING" and (thread is None or not thread.is_alive()):
            projected = dict(state)
            projected["status"] = "INTERRUPTED"
            projected["message"] = (
                "The process stopped before this durable build completed; start a new build to retry."
            )
            return projected
        return state

    def cancel(self, job_id: str, *, explicit_action: bool) -> dict[str, Any]:
        if explicit_action is not True:
            raise ConditionedDatasetError(
                "explicit_cancel_action_required", "Cancellation requires an explicit action.", status_code=422
            )
        state = self.job(job_id)
        if state["status"] not in {"RUNNING", "INTERRUPTED"}:
            raise ConditionedDatasetError("job_not_cancellable", "Only an active candidate build can be cancelled.")
        with self._lock:
            self._cancelled.add(job_id)
        return self.job(job_id)

    def attach_evidence(self, job_id: str, *, kind: str, document: Mapping[str, Any]) -> dict[str, Any]:
        """Verify and persist externally supplied independent PASS evidence."""

        if kind not in {"label_audit", "dataset_validation"}:
            raise ConditionedDatasetError(
                "evidence_kind", "Evidence kind must be label_audit or dataset_validation.", status_code=422
            )
        root = self._job_root(job_id)
        with self._lock:
            state = self.job(job_id)
            if state["status"] not in {"NEEDS_REVIEW", "COMPLETE"} or not isinstance(state.get("candidate"), Mapping):
                raise ConditionedDatasetError(
                    "candidate_not_ready", "Complete the candidate build before attaching audit evidence."
                )
            candidate = self._load_candidate(root, state)
            normalized = self._validate_evidence(kind, document, candidate)
            payload = (strict_json_dumps(normalized, indent=2, sort_keys=True) + "\n").encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            evidence_root = require_confined_path(root / "evidence", root)
            evidence_root.mkdir(exist_ok=True)
            path = require_confined_path(evidence_root / f"{kind}-{digest}.json", root)
            if path.exists():
                if path.read_bytes() != payload:
                    raise ConditionedDatasetError(
                        "evidence_identity_conflict", "Existing evidence bytes do not match their identity."
                    )
            else:
                atomic_write_bytes(path, payload)
            evidence = dict(state.get("evidence") or {})
            evidence[kind] = {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "byte_count": len(payload),
                "auditor_id": normalized["auditor"]["auditor_id"],
                "audit_run_identity": normalized["audit_run_identity"],
            }
            state["evidence"] = evidence
            state["status"] = "NEEDS_REVIEW"
            state["message"] = "Independent evidence recorded; both reports remain required before publication."
            state["updated_at"] = _now()
            self._write_state(root, state)
            return self.job(job_id)

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
        with self._lock:
            state = self.job(job_id)
            if state.get("freeze_authorization") is not None:
                raise ConditionedDatasetError(
                    "freeze_authorization_consumed",
                    "This candidate's one-time freeze authorization was already consumed.",
                )
            candidate = self._load_candidate(root, state)
            if candidate.get("candidate_identity") != candidate_identity:
                raise ConditionedDatasetError(
                    "candidate_identity_changed",
                    "The candidate identity no longer matches the authorized build.",
                )
            actual_inventory = _inventory(root / "candidate" / "phase7")
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
            state["freeze_authorization"] = {
                "authorization_id_sha256": hashlib.sha256(authorization_id.encode("utf-8")).hexdigest(),
                "candidate_identity": candidate_identity,
                "label_audit_sha256": label_audit_sha256,
                "dataset_validation_sha256": dataset_validation_sha256,
                "consumed_at": _now(),
                "one_time": True,
            }
            state["status"] = "RUNNING"
            state["stage"] = "publishing"
            state["message"] = "Publishing the authorized immutable freeze."
            state["updated_at"] = _now()
            self._write_state(root, state)
            try:
                publication = self._publish(root, candidate, evidence)
            except ConditionedDatasetError:
                state = self._read_state(root)
                state["status"] = "FAILED"
                state["stage"] = "publication_failed"
                state["message"] = "Publication failed closed; the one-time authorization was consumed."
                state["updated_at"] = _now()
                self._write_state(root, state)
                raise
            except (OSError, ValueError, TypeError, KeyError) as exc:
                state = self._read_state(root)
                state["status"] = "FAILED"
                state["stage"] = "publication_failed"
                state["message"] = "Publication failed before project configuration was changed."
                state["updated_at"] = _now()
                self._write_state(root, state)
                raise ConditionedDatasetError(
                    "publication_failed",
                    "Publication failed before project configuration was changed.",
                    status_code=500,
                ) from exc
            state = self._read_state(root)
            state["publication"] = publication
            state["status"] = "COMPLETE"
            state["stage"] = "published"
            state["message"] = (
                "Conditioned Dataset-v5 freeze and bound campaign are published; project activation remains separate."
            )
            state["updated_at"] = _now()
            self._write_state(root, state)
            return self.job(job_id)

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
                state = self._read_state(entry)
            except ConditionedDatasetError:
                continue
            thread = self._threads.get(entry.name)
            if state.get("status") == "RUNNING" and (thread is None or not thread.is_alive()):
                state = dict(state)
                state["status"] = "INTERRUPTED"
                state["message"] = "The process stopped before this durable operation completed."
            results.append(state)
        return results

    def _verified_source(self, dataset_reference: str) -> dict[str, Any]:
        """Load only a published, revalidated DatasetIntake-backed import."""

        try:
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
        for raw in files:
            relative = _canonical_relative(str(raw.get("relative_path") or ""))
            if relative not in accepted_relative_paths:
                exclusions.append("dataset_intake_excluded")
                continue
            if raw.get("usable") is not True:
                exclusions.append("harvest_quarantine")
                continue
            if raw.get("mime_type") != "image/png":
                exclusions.append("not_png")
                continue
            path = require_confined_path(
                source["artifacts_root"].joinpath(*PurePosixPath(relative).parts), source["artifacts_root"]
            )
            try:
                content = _read_regular_bytes(path, source["artifacts_root"], max_bytes=self.policy.max_file_bytes)
            except ConditionedDatasetError:
                exclusions.append("unsafe_or_unreadable")
                continue
            digest = hashlib.sha256(content).hexdigest()
            expected = str(raw.get("actual_sha256") or "")
            if digest != expected or len(content) != int(raw.get("byte_count", -1)):
                raise ConditionedDatasetError(
                    "harvest_artifact_changed", "A Harvest PNG changed during candidate inspection."
                )
            try:
                with Image.open(io.BytesIO(content)) as opened:
                    if opened.format != "PNG" or opened.size != (32, 32):
                        exclusions.append("not_exact_32x32_png")
                        continue
                    rgba = np.asarray(opened.convert("RGBA"), dtype=np.uint8)
            except (OSError, UnidentifiedImageError):
                exclusions.append("invalid_png")
                continue
            alpha = rgba[:, :, 3]
            if not np.all(np.isin(alpha, (0, 255))):
                exclusions.append("soft_alpha")
                continue
            category, object_name, tokens, disagreement = _infer_semantics(relative)
            if disagreement:
                exclusions.append("taxonomy_disagreement")
                continue
            if category == "unknown" or not object_name:
                exclusions.append("unknown_or_low_confidence")
                continue
            records.append(
                _SourceRecord(
                    relative_path=relative,
                    path=path,
                    byte_count=len(content),
                    byte_sha256=digest,
                    pixel_sha256=hashlib.sha256(rgba.tobytes()).hexdigest(),
                    alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),
                    perceptual_hash=_perceptual_hash(rgba),
                    category=category,
                    object_name=object_name,
                    tokens=tokens,
                    source_id=str(source["source_id"]),
                    source_title=str(source["source_title"]),
                    creator=str(source["creator"]),
                    license_id=str(source["license_id"]),
                    license_evidence=dict(source["license_evidence"]),
                )
            )
        return records, exclusions

    def _run_build(self, job_id: str) -> None:
        root = self._job_root(job_id)
        try:
            state = self._read_state(root)
            sources = [self._verified_source(value) for value in state["dataset_references"]]
            records: list[_SourceRecord] = []
            exclusions: list[str] = []
            for source in sources:
                source_records, source_exclusions = self._inspect_records(source)
                records.extend(source_records)
                exclusions.extend(source_exclusions)
            records, duplicate_exclusions = _deduplicate_records(records)
            exclusions.extend(duplicate_exclusions)
            selected = _representative_selection(records, self.policy.target_images)
            if not self.policy.min_images <= len(selected) <= self.policy.max_images:
                raise ConditionedDatasetError(
                    "conditioned_count_out_of_range",
                    "The verified, known-category unique source set cannot produce 2,000-3,000 conditioned images.",
                )
            self._event(root, "conditioning", 0, len(selected), "Building deterministic filename-grounded labels.")
            candidate = self._build_candidate(root, sources, selected, exclusions)
            state = self._read_state(root)
            state["candidate"] = {
                "candidate_identity": candidate["candidate_identity"],
                "manifest_relative_path": "candidate_manifest.json",
                "manifest_sha256": file_sha256(root / "candidate_manifest.json"),
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
            self._write_state(root, state)
        except ConditionedDatasetError as exc:
            state = self._read_state(root)
            state["status"] = "CANCELLED" if exc.code == "build_cancelled" else "FAILED"
            state["stage"] = "cancelled" if exc.code == "build_cancelled" else "failed"
            state["message"] = exc.public_message
            state["updated_at"] = _now()
            self._write_state(root, state)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            state = self._read_state(root)
            state["status"] = "FAILED"
            state["stage"] = "failed"
            state["message"] = (
                "Candidate build failed safely; no production freeze or project configuration changed "
                f"({type(exc).__name__})."
            )
            state["updated_at"] = _now()
            self._write_state(root, state)
        finally:
            with self._lock:
                self._threads.pop(job_id, None)
                self._cancelled.discard(job_id)

    def _build_candidate(
        self,
        root: Path,
        sources: Sequence[Mapping[str, Any]],
        selected: Sequence[_SourceRecord],
        exclusions: Sequence[str],
    ) -> dict[str, Any]:
        family_by_key = _family_assignments(selected)
        split_by_key = _split_assignments(selected, family_by_key)
        imported: list[ImportedSprite] = []
        metadata_by_id: dict[str, dict[str, Any]] = {}
        seen_ids: set[str] = set()
        for index, record in enumerate(selected, start=1):
            self._check_cancel(root.name)
            imported_record = import_png_as_dataset_item(
                record.path,
                options=ImportOptions(
                    max_palette_slots=32,
                    allow_quantize_overcolor=False,
                    quantize_overcolor=False,
                    allow_nearest_resize=False,
                    infer_role_map=True,
                    canonicalize_palette=True,
                ),
                default_category=record.category,
                default_tags=(record.category, *record.tokens[:6]),
            )
            if imported_record.errors or imported_record.bundle is None:
                raise ConditionedDatasetError(
                    "phase7_import_rejected",
                    "A selected exact-32x32 PNG failed the lossless Phase-7 import contract.",
                )
            if file_sha256(record.path) != record.byte_sha256:
                raise ConditionedDatasetError(
                    "source_changed_during_build", "A Harvest PNG changed during Phase-7 import."
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
                "evidence_type": "deterministic_filename_and_relative_path_tokens",
                "inference_method": "conditioned_filename_taxonomy_v1",
                "human_verified": False,
                "human_truth_claim": False,
                "claim_scope": "source_grounded_non_human_proposal",
                "source_relative_path": record.relative_path,
                "source_path_sha256": hashlib.sha256(record.relative_path.encode("utf-8")).hexdigest(),
                "tokens": list(record.tokens),
                "taxonomy_category": record.category,
                "source_id": record.source_id,
                "source_group": source_group,
                "duplicate_family_id": family_id,
            }
            prediction = {
                "safe_prefill": {
                    "category": record.category,
                    "object_name": record.object_name,
                    "tags": [record.category, *record.tokens[:6]],
                    "short_description": record.object_name.replace("_", " "),
                    "confidence": "source_grounded_low",
                    "confidence_reason": "filename_path_tokens_only",
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
            portable_path = Path("harvest") / record.source_id / "artifacts" / Path(record.relative_path)
            item = DatasetMakerItem(
                sprite_id=sprite_id,
                source_path=portable_path,
                status="accepted",
                category=record.category,
                tags=(record.category, *record.tokens[:6]),
                notes="Deterministic filename/path semantic proposal; not human truth.",
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
            }
            if index == 1 or index % 50 == 0 or index == len(selected):
                self._event(
                    root,
                    "conditioning",
                    index,
                    len(selected),
                    f"Conditioned {index} of {len(selected)} selected sprites.",
                )

        candidate_root = require_confined_path(root / "candidate", root)
        candidate_root.mkdir()
        result = export_dataset_from_imported_sprites(
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
        )
        phase7 = result.output_dir
        _enrich_manifests(phase7, metadata_by_id)
        self._check_cancel(root.name)

        training_result = build_training_manifest(
            phase7,
            variants_per_sprite=2,
            caption_policy="mixed",
            seed=731001,
        )
        training_manifest = phase7 / "training_manifest.jsonl"
        write_training_manifest(training_manifest, _portable_training_rows(training_result.rows))
        _write_json(
            phase7 / "conditioning_vocabulary.json",
            _conditioning_vocabulary(imported, training_result.rows),
        )
        _write_jsonl(
            phase7 / "conditioned_records.jsonl",
            [
                {
                    "sprite_id": item.item.sprite_id,
                    "split": item.item.split,
                    "category": item.item.category,
                    "object_name": item.auto_metadata["label_v2_safe_prefill"]["object_name"],
                    **metadata_by_id[item.item.sprite_id],
                    "semantic_v3": item.auto_metadata["semantic_v3"],
                }
                for item in sorted(imported, key=lambda value: value.item.sprite_id)
            ],
        )
        _write_jsonl(
            phase7 / "split_assignments.jsonl",
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
        _write_json(phase7 / "coverage_report.json", coverage)
        _write_json(phase7 / "split_integrity_report.json", split_check)
        _write_json(
            phase7 / "duplicate_report.json",
            {
                "schema_version": "spritelab.dataset.conditioned-duplicates.v1",
                "exact_byte_and_pixel_duplicates_excluded": int(
                    Counter(exclusions)["exact_byte_duplicate"] + Counter(exclusions)["exact_pixel_duplicate"]
                ),
                "near_duplicate_policy": "same source-grounded object, alpha identity, and perceptual Hamming distance <= 1",
                "family_count": len(set(family_by_key.values())),
                "whole_family_split_gate": split_check["ok"],
            },
        )
        source_evidence = []
        for source in sources:
            source_evidence.append(
                {
                    "dataset_reference": source["dataset_reference"],
                    "harvest_run_id": source["run_id"],
                    "handoff_identity": source["handoff_identity"],
                    "harvest_import_receipt_identity": source["harvest_import_receipt_identity"],
                    "managed_intake_receipt_identity": source["managed_intake_receipt_identity"],
                    "managed_source_inventory_sha256": source["managed_source_inventory_sha256"],
                    "managed_output_inventory_sha256": source["managed_output_inventory_sha256"],
                    "artifact_manifest_sha256": source["artifact_manifest_sha256"],
                    "artifact_set_identity": source["artifact_manifest"]["artifact_set_identity"],
                    "source_id": source["source_id"],
                    "title": source["source_title"],
                    "creator": source["creator"],
                    "license_id": source["license_id"],
                    "license_evidence": dict(source["license_evidence"]),
                }
            )
        _write_json(
            phase7 / "provenance_manifest.json",
            {
                "schema_version": "spritelab.dataset.conditioned-provenance.v1",
                "sources": source_evidence,
                "all_source_files_rehashed": True,
                "license_policy": sorted(ALLOWED_LICENSES),
                "paths_are_portable": True,
            },
        )
        benchmark = _benchmark_manifest(imported, metadata_by_id)
        _write_json(phase7 / "benchmark_manifest.json", benchmark)
        dataset_qa = qa_dataset(phase7, require_semantic_v3=True)
        training_qa = qa_training_manifest(phase7, training_manifest)
        dataset_qa_value = dataset_qa.to_json_dict()
        dataset_qa_value["dataset_dir"] = "."
        training_qa_value = training_qa.to_json_dict()
        training_qa_value["dataset_dir"] = "."
        training_qa_value["manifest_path"] = "training_manifest.jsonl"
        _write_json(phase7 / "dataset_qa_report.json", dataset_qa_value)
        _write_json(phase7 / "training_manifest_qa_report.json", training_qa_value)
        if dataset_qa.errors or training_qa.errors:
            raise ConditionedDatasetError(
                "candidate_qa_failed", "The Phase-7 dataset or training manifest failed local structural QA."
            )
        loader = _loader_check(phase7)
        _write_json(phase7 / "loader_check.json", loader)
        if not loader["ok"]:
            raise ConditionedDatasetError(
                "candidate_loader_failed",
                "The production-format loader check did not cover every split and vocabulary binding.",
            )

        view_manifest = {
            "schema_version": "spritelab.dataset.conditioned-view.v1",
            "view_identity": stable_hash(
                {
                    "managed_intake_receipt_identities": [
                        source["managed_intake_receipt_identity"] for source in sources
                    ],
                    "image_count": len(imported),
                    "records_sha256": file_sha256(phase7 / "conditioned_records.jsonl"),
                }
            ),
            "image_count": len(imported),
            "records_path": "conditioned_records.jsonl",
            "records_sha256": file_sha256(phase7 / "conditioned_records.jsonl"),
            "training_manifest_path": "training_manifest.jsonl",
            "training_manifest_sha256": file_sha256(training_manifest),
            "split_integrity_sha256": file_sha256(phase7 / "split_integrity_report.json"),
            "coverage_report_sha256": file_sha256(phase7 / "coverage_report.json"),
            "requires_semantic_labels": True,
            "human_truth_claim": False,
            "paths_are_portable": True,
        }
        _write_json(phase7 / "view_manifest.json", view_manifest)
        payload_inventory = _inventory(phase7)
        inventory_identity = _inventory_identity(payload_inventory)
        candidate_identity = stable_hash(
            {
                "schema_version": CANDIDATE_SCHEMA,
                "handoff_identities": [source["handoff_identity"] for source in sources],
                "managed_intake_receipt_identities": [source["managed_intake_receipt_identity"] for source in sources],
                "managed_output_inventory_sha256": [source["managed_output_inventory_sha256"] for source in sources],
                "payload_inventory_sha256": inventory_identity,
                "image_count": len(imported),
                "recipe": "conditioned_filename_taxonomy_v1",
            }
        )
        candidate = {
            "schema_version": CANDIDATE_SCHEMA,
            "candidate_identity": candidate_identity,
            "payload_inventory_sha256": inventory_identity,
            "payload_inventory": payload_inventory,
            "image_count": len(imported),
            "category_counts": coverage["category_counts"],
            "source_counts": coverage["source_counts"],
            "split_counts": coverage["split_counts"],
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
        _write_json(root / "candidate_manifest.json", candidate)
        return candidate

    def _load_candidate(self, root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
        reference = state.get("candidate")
        if not isinstance(reference, Mapping):
            raise ConditionedDatasetError("candidate_not_ready", "No complete conditioned candidate is available.")
        path = require_confined_path(root / str(reference.get("manifest_relative_path") or ""), root)
        candidate = _read_json_mapping(path, max_bytes=128 * 1024 * 1024)
        if candidate.get("schema_version") != CANDIDATE_SCHEMA:
            raise ConditionedDatasetError("candidate_schema", "The conditioned candidate manifest schema is invalid.")
        if file_sha256(path) != reference.get("manifest_sha256"):
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
        return candidate

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
        if value.get("schema_version") != schema or str(value.get("verdict", "")).upper() != "PASS":
            raise ConditionedDatasetError(
                "evidence_verdict", "The selected independent report is not an applicable PASS report."
            )
        if value.get("independent") is not True or value.get("generated_by_conditioned_workflow") is not False:
            raise ConditionedDatasetError(
                "evidence_independence", "The report must explicitly be independent of the conditioned workflow."
            )
        auditor = value.get("auditor")
        if not isinstance(auditor, Mapping):
            raise ConditionedDatasetError("evidence_auditor", "Independent evidence lacks auditor identity.")
        auditor_id = str(auditor.get("auditor_id") or "")
        code_identity = str(auditor.get("code_identity_sha256") or "")
        if not AUDITOR_ID_PATTERN.fullmatch(auditor_id) or not SHA256_PATTERN.fullmatch(code_identity):
            raise ConditionedDatasetError(
                "evidence_auditor", "Independent evidence has an invalid auditor or code identity."
            )
        if auditor_id.startswith(("spritelab.conditioned", "conditioned.v5")):
            raise ConditionedDatasetError(
                "evidence_self_certification", "The conditioned workflow cannot certify its own candidate."
            )
        if not SHA256_PATTERN.fullmatch(str(value.get("audit_run_identity") or "")):
            raise ConditionedDatasetError(
                "evidence_run_identity", "Independent evidence lacks a stable audit-run identity."
            )
        bindings = value.get("bindings")
        if not isinstance(bindings, Mapping) or bindings.get("candidate_identity") != candidate.get(
            "candidate_identity"
        ):
            raise ConditionedDatasetError(
                "evidence_candidate_binding", "Independent evidence is not bound to this exact candidate."
            )
        if bindings.get("payload_inventory_sha256") != candidate.get("payload_inventory_sha256"):
            raise ConditionedDatasetError(
                "evidence_inventory_binding", "Independent evidence is not bound to this exact artifact inventory."
            )
        if bindings.get("image_count") != candidate.get("image_count"):
            raise ConditionedDatasetError(
                "evidence_count_binding", "Independent evidence is not bound to this exact image count."
            )
        if value.get("subject_files") != candidate.get("payload_inventory"):
            raise ConditionedDatasetError(
                "evidence_file_inventory", "Independent evidence does not enumerate every exact candidate file."
            )
        checks = value.get("checks")
        if not isinstance(checks, Mapping) or set(checks) != gates:
            raise ConditionedDatasetError(
                "evidence_checks", "Independent evidence does not contain the complete mandatory gate set."
            )
        if {str(result).upper() for result in checks.values()} != {"PASS"}:
            raise ConditionedDatasetError("evidence_checks", "Every mandatory independent evidence gate must PASS.")
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
            if not isinstance(reference, Mapping) or reference.get("sha256") != digest:
                raise ConditionedDatasetError(
                    "evidence_selection", "The selected evidence is not the verified report attached to this candidate."
                )
            path = require_confined_path(root / str(reference.get("relative_path") or ""), root)
            content = _read_regular_bytes(path, root, max_bytes=64 * 1024 * 1024)
            if hashlib.sha256(content).hexdigest() != digest or len(content) != reference.get("byte_count"):
                raise ConditionedDatasetError("evidence_changed", "Independent evidence bytes changed after selection.")
            try:
                value = strict_json_loads(content)
            except ValueError as exc:
                raise ConditionedDatasetError(
                    "evidence_changed", "Independent evidence is no longer readable."
                ) from exc
            results[kind] = {
                "document": self._validate_evidence(kind, value, candidate),
                "path": path,
                "sha256": digest,
                "byte_count": len(content),
            }
        return results

    def _publish(
        self,
        job_root: Path,
        candidate: Mapping[str, Any],
        evidence: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        source_root = require_confined_path(job_root / "candidate" / "phase7", job_root)
        source_inventory = _inventory(source_root)
        files: dict[str, dict[str, Any]] = dict(source_inventory)
        for kind in ("label_audit", "dataset_validation"):
            record = evidence[kind]
            files[f"evidence/{kind}.json"] = {
                "sha256": record["sha256"],
                "byte_count": record["byte_count"],
            }
        publication_identity = _inventory_identity(files)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        if not _safe_directory(self.datasets_root):
            raise ConditionedDatasetError(
                "datasets_root_unsafe", "The project datasets directory is unsafe for publication."
            )
        name = f"conditioned-v5-{publication_identity}"
        final = require_confined_path(self.datasets_root / name, self.datasets_root)
        if os.path.lexists(final):
            raise ConditionedDatasetError(
                "fresh_publication_required",
                "This content-addressed publication already exists; it was not overwritten.",
            )
        staging = require_confined_path(self.datasets_root / f".{name}.staging-{uuid.uuid4().hex}", self.datasets_root)
        staging.mkdir()
        try:
            for relative, expected in sorted(source_inventory.items()):
                source = require_confined_path(source_root.joinpath(*PurePosixPath(relative).parts), source_root)
                content = _read_regular_bytes(
                    source, source_root, max_bytes=max(self.policy.max_source_bytes, 1024 * 1024 * 1024)
                )
                if hashlib.sha256(content).hexdigest() != expected["sha256"] or len(content) != expected["byte_count"]:
                    raise ConditionedDatasetError(
                        "candidate_copy_changed", "A candidate file changed during freeze publication."
                    )
                destination = require_confined_path(staging.joinpath(*PurePosixPath(relative).parts), staging)
                atomic_write_bytes(destination, content)
            for kind in ("label_audit", "dataset_validation"):
                record = evidence[kind]
                content = _read_regular_bytes(record["path"], job_root, max_bytes=64 * 1024 * 1024)
                atomic_write_bytes(staging / "evidence" / f"{kind}.json", content)
            actual = _inventory(staging)
            if actual != files:
                raise ConditionedDatasetError(
                    "publication_copy_mismatch", "The staged publication inventory does not match the authorized bytes."
                )
            inventory_payload = _inventory_payload(actual)
            artifact_names = {
                "view_manifest": "view_manifest.json",
                "split_manifest": "training_manifest.jsonl",
                "conditioning_vocabulary": "conditioning_vocabulary.json",
                "benchmark_manifest": "benchmark_manifest.json",
                "labeling_audit": "evidence/label_audit.json",
                "validation_report": "evidence/dataset_validation.json",
            }
            artifacts = {name: {"path": path, **actual[path]} for name, path in artifact_names.items()}
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
                    "inventory_sha256": _inventory_identity(actual),
                },
                "licenses": list(candidate["license_ids"]),
                "paths_are_relative": True,
                "paths_exposed": False,
            }
            _write_json(staging / "activation.json", activation)
            staging.replace(final)
        except BaseException:
            if staging.exists():
                remove_confined_tree(staging, self.datasets_root, missing_ok=True)
            raise

        activation_path = final / "activation.json"
        activation_relative = activation_path.relative_to(self.project_root).as_posix()
        campaign_relative_directory = f"campaigns/conditioned-v5-{publication_identity}"
        self.campaigns_root.mkdir(parents=True, exist_ok=True)
        campaign_directory = require_confined_path(self.project_root / campaign_relative_directory, self.project_root)
        if os.path.lexists(campaign_directory):
            raise ConditionedDatasetError(
                "fresh_campaign_required", "The conditioned campaign directory already exists and was not reused."
            )
        campaign_directory.mkdir()
        builder = self._campaign_builder
        if builder is None:
            from spritelab.product_features.training.activation import build_conditioned_three_seed_campaign

            builder = build_conditioned_three_seed_campaign
        try:
            built = builder(
                self.project_root,
                campaign_directory=campaign_relative_directory,
                activation_manifest=activation_relative,
                activation_manifest_sha256=file_sha256(activation_path),
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
            validation = dict(built.validation)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ConditionedDatasetError(
                "campaign_build_failed",
                "The freeze was published, but its exact three-seed campaign failed authoritative validation; configuration was not changed.",
            ) from exc
        campaign_document = {
            "schema_version": "spritelab.training.conditioned-campaign-config.v1",
            "product_profiles": {
                "recommended": {
                    "display": {"display_name": "Conditioned Dataset-v5 · 3 seeds · 5,000 steps"},
                    "campaign": portable,
                }
            },
        }
        _write_json(campaign_directory / "campaign.json", campaign_document)
        return {
            "publication_identity_sha256": publication_identity,
            "activation_manifest": activation_relative,
            "activation_manifest_sha256": file_sha256(activation_path),
            "campaign_config": (campaign_directory / "campaign.json").relative_to(self.project_root).as_posix(),
            "campaign_config_sha256": file_sha256(campaign_directory / "campaign.json"),
            "campaign_launch_ready": validation.get("launch_ready") is True,
            "campaign_seeds": list(portable.get("seeds") or ()),
            "campaign_steps": dict(portable.get("training") or {}).get("max_optimizer_steps"),
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
        with self._lock:
            state = _read_json_mapping(root / "state.json", max_bytes=16 * 1024 * 1024)
        if state.get("schema_version") != "spritelab.dataset.conditioned-job.v1" or state.get("job_id") != root.name:
            raise ConditionedDatasetError("job_state", "The conditioned candidate job state is invalid.")
        if state.get("paths_exposed") is not False:
            raise ConditionedDatasetError(
                "job_state_privacy", "The conditioned candidate job state violates the private-path contract."
            )
        return state

    def _write_state(self, root: Path, state: Mapping[str, Any]) -> None:
        payload = strict_json_dumps(dict(state), indent=2, sort_keys=True) + "\n"
        if str(self.project_root) in payload or str(self.project_root).replace("\\", "/") in payload:
            raise ConditionedDatasetError(
                "private_path_persistence", "A private project path was refused from durable job state."
            )
        with self._lock:
            atomic_write_text(require_confined_path(root / "state.json", root), payload)

    def _event(self, root: Path, stage: str, current: int, total: int, message: str) -> None:
        state = self._read_state(root)
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
        self._write_state(root, state)

    def _check_cancel(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._cancelled:
                raise ConditionedDatasetError(
                    "build_cancelled", "Candidate build was cancelled; no production freeze changed."
                )


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


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _is_link_or_reparse(path: Path) -> bool:
    try:
        return _metadata_is_link_or_reparse(path.lstat())
    except OSError:
        return False


def _read_regular_bytes(path: Path, root: Path, *, max_bytes: int) -> bytes:
    try:
        target = require_confined_path(path, root)
        before = target.lstat()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedDatasetError(
            "managed_file_unsafe", "A required managed file is unavailable or unsafe."
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _metadata_is_link_or_reparse(before)
        or before.st_nlink != 1
        or before.st_size > max_bytes
    ):
        raise ConditionedDatasetError(
            "managed_file_unsafe", "A required managed file is not a singly linked bounded regular file."
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
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
            or opened.st_nlink != 1
        ):
            raise ConditionedDatasetError("managed_file_changed", "A managed file changed while it was opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if len(content) > max_bytes:
        raise ConditionedDatasetError("managed_file_oversized", "A required managed file exceeds its byte limit.")
    after = target.lstat()
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or _metadata_is_link_or_reparse(after)
    ):
        raise ConditionedDatasetError("managed_file_changed", "A managed file changed while it was read.")
    return content


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
    if not value or value != unicodedata.normalize("NFC", value) or "\\" in value or "\x00" in value:
        raise ConditionedDatasetError(
            "artifact_relative_path", "A Harvest artifact path is not canonical and portable."
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ConditionedDatasetError(
            "artifact_relative_path", "A Harvest artifact path is not canonical and portable."
        )
    if posix.as_posix() != value:
        raise ConditionedDatasetError(
            "artifact_relative_path", "A Harvest artifact path is not canonical and portable."
        )
    return value


def _path_tokens(value: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", value.casefold())
    return tuple(token for token in tokens if token not in _STOP_TOKENS and not token.isdecimal())


def _infer_semantics(relative_path: str) -> tuple[str, str, tuple[str, ...], bool]:
    all_tokens = _path_tokens(relative_path)
    parent_tokens = _path_tokens(PurePosixPath(relative_path).parent.as_posix())
    path_matches = [
        category for category, terms in _CATEGORY_PATH_HINTS.items() if any(token in terms for token in parent_tokens)
    ]
    if len(path_matches) > 1:
        return "unknown", "", all_tokens, True
    path_category = path_matches[0] if path_matches else None
    specific_matches = [
        category
        for category, terms in _CATEGORY_TERMS.items()
        if category != "icon" and any(token in terms for token in all_tokens)
    ]
    if path_category is not None:
        category = path_category
    elif len(specific_matches) > 1:
        return "unknown", "", all_tokens, True
    elif specific_matches:
        category = specific_matches[0]
    elif any(token in _CATEGORY_TERMS["icon"] for token in all_tokens):
        category = "icon"
    else:
        return "unknown", "", all_tokens, False
    stem_tokens = _path_tokens(PurePosixPath(relative_path).stem)
    meaningful = tuple(token for token in stem_tokens if token not in {"sprite", "tile", "icon"})
    if not meaningful:
        meaningful = stem_tokens
    object_name = normalize_sprite_id("_".join(meaningful[:8]))
    if not object_name or len(object_name) < 2:
        return "unknown", "", all_tokens, False
    return category, object_name, all_tokens, False


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


def _deduplicate_records(records: Sequence[_SourceRecord]) -> tuple[list[_SourceRecord], list[str]]:
    ordered = sorted(records, key=lambda record: (record.source_id, record.relative_path))
    byte_seen: set[str] = set()
    pixel_seen: set[str] = set()
    kept: list[_SourceRecord] = []
    exclusions: list[str] = []
    for record in ordered:
        if record.byte_sha256 in byte_seen:
            exclusions.append("exact_byte_duplicate")
            continue
        byte_seen.add(record.byte_sha256)
        if record.pixel_sha256 in pixel_seen:
            exclusions.append("exact_pixel_duplicate")
            continue
        pixel_seen.add(record.pixel_sha256)
        kept.append(record)
    return kept, exclusions


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
    by_near_bucket: dict[tuple[str, str, str], list[_SourceRecord]] = defaultdict(list)
    for record in records:
        by_source_group[_source_group(record)].append(record)
        by_near_bucket[(record.source_id, record.object_name, record.alpha_sha256)].append(record)
    for members in by_source_group.values():
        for record in members[1:]:
            union.union(_record_key(members[0]), _record_key(record))
    for members in by_near_bucket.values():
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if (left.perceptual_hash ^ right.perceptual_hash).bit_count() <= 1:
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


def _enrich_manifests(dataset: Path, metadata: Mapping[str, Mapping[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        path = dataset / f"manifest_{split}.jsonl"
        rows = _read_jsonl(path)
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
                    "source_path": f"harvest/{source['source_id']}/artifacts/{source['source_relative_path']}",
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
                    "human_truth_claim": False,
                }
            )
            enriched.append(updated)
        _write_jsonl(path, enriched)


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


def _benchmark_manifest(
    imported: Sequence[ImportedSprite],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = [item for item in imported if item.item.split in {"val", "test"}]
    candidates.sort(key=lambda item: hashlib.sha256(f"benchmark:{item.item.sprite_id}".encode()).hexdigest())
    seen_groups: set[str] = set()
    selected: list[dict[str, Any]] = []
    for item in candidates:
        source = metadata[item.item.sprite_id]
        group = str(source["source_group"])
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected.append(
            {
                "sprite_id": item.item.sprite_id,
                "split": item.item.split,
                "category": item.item.category,
                "source_id": source["source_id"],
                "source_group": group,
                "duplicate_family_id": source["duplicate_family_id"],
                "source_sha256": source["source_sha256"],
            }
        )
        if len(selected) >= 128:
            break
    if not selected:
        raise ConditionedDatasetError(
            "benchmark_empty", "No source-group-disjoint validation/test benchmark could be built."
        )
    return {
        "schema_version": "spritelab.dataset.conditioned-benchmark.v1",
        "selection_policy": "one deterministic val/test record per source_group",
        "source_group_disjoint_from_training": True,
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
        lines = path.read_text(encoding="utf-8").splitlines()
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, strict_json_dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    lines = [strict_json_dumps(dict(value), sort_keys=True, separators=(",", ":")) for value in values]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    if not _safe_directory(root):
        raise ConditionedDatasetError("inventory_root", "The artifact inventory root is unsafe or unavailable.")
    root_metadata = root.lstat()
    records: dict[str, dict[str, Any]] = {}
    collision_keys: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            child = require_confined_path(directory_path / name, root)
            if not _safe_directory(child) or child.lstat().st_dev != root_metadata.st_dev:
                raise ConditionedDatasetError(
                    "inventory_entry", "An artifact directory is linked, mounted, or crosses a device."
                )
        for name in sorted(file_names):
            child = require_confined_path(directory_path / name, root)
            relative = child.relative_to(root).as_posix()
            _canonical_relative(relative)
            collision = unicodedata.normalize("NFC", relative).casefold()
            if collision in collision_keys:
                raise ConditionedDatasetError(
                    "inventory_collision", "Artifact paths contain a case or Unicode collision."
                )
            collision_keys.add(collision)
            records[relative] = _stable_file_identity(child, root_metadata.st_dev)
    return dict(sorted(records.items()))


def _stable_file_identity(path: Path, root_device: int) -> dict[str, Any]:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or _metadata_is_link_or_reparse(before)
        or before.st_nlink != 1
        or before.st_dev != root_device
    ):
        raise ConditionedDatasetError(
            "inventory_entry", "Artifact inventory entries must be owned regular files on one device."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    after = path.lstat()
    if (
        after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ConditionedDatasetError("inventory_changed", "An artifact changed while its identity was computed.")
    return {"sha256": digest.hexdigest(), "byte_count": before.st_size}


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
