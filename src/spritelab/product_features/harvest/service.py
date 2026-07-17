"""Durable, certified, privacy-safe orchestration for controlled Harvest runs."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps, strict_json_loads
from spritelab.product_features.harvest.catalog import (
    INITIAL_LICENSE_POLICY,
    SHA256_PATTERN,
    SOURCE_ID_PATTERN,
    HarvestSource,
    public_download_url,
    url_identity,
    validate_download_url,
)
from spritelab.product_features.harvest.storage import (
    HarvestStorageError,
    RepositoryMutationLock,
    scan_artifacts,
    scan_legacy_run,
)
from spritelab.product_features.harvest.trusted_backend import (
    AcquiredFile,
    AcquisitionResult,
    BackendFactory,
    CertifiedBackendCapabilities,
    DatasetImportCallback,
    DatasetImportRequest,
    HarvestLimits,
    validate_callback_identity,
)
from spritelab.utils.safe_fs import UnsafeFilesystemOperation, atomic_write_text, require_confined_path

HARVEST_INVENTORY_SCHEMA = "spritelab.harvest.inventory.v2"
HARVEST_REQUEST_SCHEMA = "spritelab.harvest.request.v2"
HARVEST_AUTHORIZATION_SCHEMA = "spritelab.harvest.authorization-receipt.v2"
HARVEST_STATE_SCHEMA = "spritelab.harvest.job-state.v2"
HARVEST_EVENT_SCHEMA = "spritelab.harvest.job-event.v2"
HARVEST_ACQUISITION_RECEIPT_SCHEMA = "spritelab.harvest.acquisition-receipt.v1"
HARVEST_HANDOFF_SCHEMA = "spritelab.harvest.dataset-handoff.v2"
HARVEST_IMPORT_RECEIPT_SCHEMA = "spritelab.harvest.dataset-import-receipt.v1"

RUN_ID_PATTERN = re.compile(r"^harvest-[a-z0-9][a-z0-9-]{5,79}$")
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
STAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")

ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING", "CANCELLING"})
TERMINAL_STATUSES = frozenset({"COMPLETE", "FAILED", "CANCELLED"})
RETRYABLE_STATUSES = frozenset({"FAILED", "CANCELLED", "INTERRUPTED"})
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class HarvestError(RuntimeError):
    """Structured failure whose text is safe for durable/API exposure."""

    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class AcquisitionCancelled(HarvestError):
    def __init__(self) -> None:
        super().__init__("harvest_cancelled", "Harvest acquisition was cancelled.")


@dataclass(frozen=True)
class ReuseEvidence:
    """Explicit evidence that local reusable inventory cannot meet the target."""

    decision: str
    evidence_code: str
    inventory_identity: str
    assessed_usable_items: int
    required_usable_items: int
    deficit_items: int

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | ReuseEvidence) -> ReuseEvidence:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise HarvestError(
                "reuse_evidence_required",
                "Review existing Harvest inventory and record a reuse deficit before acquisition.",
                status_code=422,
            )
        expected = {
            "decision",
            "evidence_code",
            "inventory_identity",
            "assessed_usable_items",
            "required_usable_items",
            "deficit_items",
        }
        if set(value) != expected:
            raise HarvestError(
                "invalid_reuse_evidence",
                "Reuse evidence fields are incomplete or unrecognized.",
                status_code=422,
            )
        try:
            return cls(
                decision=str(value["decision"]),
                evidence_code=str(value["evidence_code"]),
                inventory_identity=str(value["inventory_identity"]),
                assessed_usable_items=_exact_int(value["assessed_usable_items"]),
                required_usable_items=_exact_int(value["required_usable_items"]),
                deficit_items=_exact_int(value["deficit_items"]),
            )
        except (TypeError, ValueError) as exc:
            raise HarvestError(
                "invalid_reuse_evidence",
                "Reuse evidence counts must be exact non-negative integers.",
                status_code=422,
            ) from exc

    @property
    def identity(self) -> str:
        return _json_identity(self.to_dict())

    def validate(self, inventory: Mapping[str, Any]) -> None:
        if self.decision not in {"reuse_exhausted", "deficit_confirmed"}:
            raise HarvestError(
                "invalid_reuse_evidence",
                "Reuse decision must be reuse_exhausted or deficit_confirmed.",
                status_code=422,
            )
        if self.evidence_code not in {"no_reusable_items", "target_deficit"}:
            raise HarvestError(
                "invalid_reuse_evidence",
                "Reuse evidence must use a controlled deficit code.",
                status_code=422,
            )
        if SHA256_PATTERN.fullmatch(self.inventory_identity) is None or self.inventory_identity != inventory.get(
            "inventory_identity"
        ):
            raise HarvestError(
                "harvest_inventory_changed",
                "Harvest inventory changed after review; refresh and reassess reuse before starting.",
            )
        if self.assessed_usable_items < 0 or self.required_usable_items <= 0:
            raise HarvestError("invalid_reuse_evidence", "Reuse assessment counts are invalid.", status_code=422)
        expected_deficit = max(self.required_usable_items - self.assessed_usable_items, 0)
        if self.deficit_items != expected_deficit or self.deficit_items <= 0:
            raise HarvestError(
                "reuse_not_exhausted",
                "Acquisition requires a positive, arithmetically consistent reuse deficit.",
                status_code=422,
            )
        if self.decision == "reuse_exhausted" and (
            self.assessed_usable_items != 0 or self.evidence_code != "no_reusable_items"
        ):
            raise HarvestError(
                "invalid_reuse_evidence",
                "reuse_exhausted requires zero assessed usable items.",
                status_code=422,
            )
        if self.decision == "deficit_confirmed" and self.evidence_code != "target_deficit":
            raise HarvestError(
                "invalid_reuse_evidence",
                "deficit_confirmed requires target_deficit evidence.",
                status_code=422,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.reuse-evidence.v1",
            "decision": self.decision,
            "evidence_code": self.evidence_code,
            "inventory_identity": self.inventory_identity,
            "assessed_usable_items": self.assessed_usable_items,
            "required_usable_items": self.required_usable_items,
            "deficit_items": self.deficit_items,
        }


class HarvestService:
    """Own durable Harvest state while delegating only to a certified backend."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        sources: Iterable[HarvestSource] = (),
        backend_factory: BackendFactory | None = None,
        backend_capabilities: CertifiedBackendCapabilities | None = None,
        limits: HarvestLimits | None = None,
        run_id_factory: Any | None = None,
        dataset_import_callback: DatasetImportCallback | None = None,
    ) -> None:
        self.project_root = Path(project_root).absolute()
        try:
            self.output_root = require_confined_path(self.project_root / "harvest_runs", self.project_root)
        except UnsafeFilesystemOperation as exc:
            raise HarvestError(
                "unsafe_harvest_root",
                "The managed Harvest root is not a safe repository-local directory.",
            ) from exc
        catalog = tuple(sources)
        if len({source.source_id for source in catalog}) != len(catalog):
            raise ValueError("Harvest source identifiers must be unique.")
        if (backend_factory is None) != (backend_capabilities is None):
            raise ValueError("Harvest backend factory and certified capabilities must be configured together.")
        if dataset_import_callback is not None:
            validate_callback_identity(dataset_import_callback)
        self._sources = {source.source_id: source for source in catalog}
        self._backend_factory = backend_factory
        self._backend_capabilities = backend_capabilities
        self.limits = limits or HarvestLimits()
        self._run_id_factory = run_id_factory or _default_run_id
        self._dataset_import_callback = dataset_import_callback
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._instance_id = uuid.uuid4().hex

    @property
    def acquisition_configured(self) -> bool:
        return bool(self._sources) and self._backend_factory is not None and self._backend_capabilities is not None

    def sources(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.sources.v2",
            "sources": [self._sources[key].to_public_dict() for key in sorted(self._sources)],
            "license_policy": sorted(INITIAL_LICENSE_POLICY),
            "backend_configured": self.acquisition_configured,
            "backend_capabilities": self._backend_capabilities.to_dict() if self._backend_capabilities else None,
            "limits": self.limits.to_dict(),
            "network_actions": 0,
            "browser_paths_accepted": False,
        }

    def inventory(self) -> dict[str, Any]:
        """Index immediate managed and legacy runs without mutation or network."""

        with self._lock:
            managed: list[dict[str, Any]] = []
            legacy: list[dict[str, Any]] = []
            unsafe_entries = 0
            collision_keys: set[str] = set()
            if os.path.lexists(self.output_root):
                self._validate_output_root()
                for child in sorted(self.output_root.iterdir(), key=lambda value: value.name):
                    if child.name == ".harvest.lock":
                        continue
                    if len(managed) + len(legacy) + unsafe_entries >= self.limits.max_files:
                        raise HarvestError(
                            "harvest_inventory_limit",
                            "Immediate Harvest inventory exceeds the configured entry limit.",
                        )
                    collision = unicodedata.normalize("NFC", child.name).casefold()
                    if collision in collision_keys:
                        unsafe_entries += 1
                        continue
                    collision_keys.add(collision)
                    try:
                        metadata = child.lstat()
                        if (
                            not stat.S_ISDIR(metadata.st_mode)
                            or _metadata_is_link_or_reparse(metadata)
                            or child.is_mount()
                        ):
                            raise HarvestStorageError("unsafe immediate child")
                        require_confined_path(child, self.output_root)
                        if RUN_ID_PATTERN.fullmatch(child.name):
                            try:
                                managed.append(self._managed_inventory_record(child))
                                continue
                            except HarvestError:
                                pass
                        legacy_record = scan_legacy_run(child)
                        if legacy_record is None:
                            raise HarvestStorageError("unrecognized immediate child")
                        legacy.append(legacy_record)
                    except (OSError, HarvestError, HarvestStorageError, UnsafeFilesystemOperation):
                        unsafe_entries += 1
            managed.sort(key=lambda value: value["run_id"])
            legacy.sort(key=lambda value: value["legacy_id"])
            status_counts = dict(sorted(Counter(item["status"] for item in managed).items()))
            known_usable = sum(
                int(item.get("usable_count", 0)) for item in managed if item["status"] == "COMPLETE"
            ) + sum(int(item.get("imported_records", 0)) for item in legacy)
            identity_payload = {
                "managed_runs": managed,
                "legacy_runs": legacy,
                "unsafe_entries": unsafe_entries,
            }
            return {
                "schema_version": HARVEST_INVENTORY_SCHEMA,
                "run_count": len(managed),
                "legacy_run_count": len(legacy),
                "status_counts": status_counts,
                "unsafe_entries": unsafe_entries,
                "known_usable_items": known_usable,
                "legacy_candidate_records": sum(item["candidate_records"] for item in legacy),
                "legacy_imported_records": sum(item["imported_records"] for item in legacy),
                "inventory_identity": _json_identity(identity_payload),
                "runs": managed,
                "legacy_runs": legacy,
                "scope": "immediate_repository_local_harvest_runs",
                "legacy_entries_read_only": True,
                "network_actions": 0,
                "paths_exposed": False,
            }

    def start(
        self,
        source_id: str,
        *,
        idempotency_key: str,
        explicit_action: bool,
        authorize_zero_cost: bool,
        authorize_permissive_license: bool,
        authorize_existing_inventory_reviewed: bool,
        reuse_evidence: Mapping[str, Any] | ReuseEvidence,
    ) -> tuple[dict[str, Any], bool]:
        return self._start(
            source_id,
            idempotency_key=idempotency_key,
            explicit_action=explicit_action,
            authorize_zero_cost=authorize_zero_cost,
            authorize_permissive_license=authorize_permissive_license,
            authorize_existing_inventory_reviewed=authorize_existing_inventory_reviewed,
            reuse_evidence=reuse_evidence,
            retry_of=None,
        )

    def retry(
        self,
        run_id: str,
        *,
        idempotency_key: str,
        explicit_action: bool,
        authorize_zero_cost: bool,
        authorize_permissive_license: bool,
        authorize_existing_inventory_reviewed: bool,
        reuse_evidence: Mapping[str, Any] | ReuseEvidence,
    ) -> tuple[dict[str, Any], bool]:
        previous = self.job(run_id)
        if previous["status"] not in RETRYABLE_STATUSES:
            raise HarvestError(
                "harvest_retry_not_allowed",
                "Only failed, cancelled, or interrupted Harvest jobs can be retried.",
            )
        return self._start(
            str(previous["source_id"]),
            idempotency_key=idempotency_key,
            explicit_action=explicit_action,
            authorize_zero_cost=authorize_zero_cost,
            authorize_permissive_license=authorize_permissive_license,
            authorize_existing_inventory_reviewed=authorize_existing_inventory_reviewed,
            reuse_evidence=reuse_evidence,
            retry_of=run_id,
        )

    def cancel(self, run_id: str, *, explicit_action: bool) -> dict[str, Any]:
        if explicit_action is not True:
            raise HarvestError(
                "explicit_action_required",
                "Cancelling Harvest requires an explicit user action.",
                status_code=422,
            )
        with self._mutation_guard():
            run_dir = self._run_directory(run_id)
            state = self._read_state(run_dir)
            if state["status"] in TERMINAL_STATUSES:
                return self.job(run_id)
            worker = self._workers.get(run_id)
            cancellation = self._cancellations.setdefault(run_id, threading.Event())
            cancellation.set()
            if state["status"] == "CANCELLING" and worker is not None and worker.is_alive():
                return self.job(run_id)
            if worker is None or not worker.is_alive():
                self._transition_locked(
                    run_dir,
                    "CANCELLED",
                    stage="cancelled",
                    message="Harvest job was cancelled and its durable evidence was retained.",
                    ended=True,
                )
            else:
                self._transition_locked(
                    run_dir,
                    "CANCELLING",
                    stage="cancelling",
                    message="Cancellation is waiting for the certified backend safety boundary.",
                )
            return self.job(run_id)

    def job(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run_dir = self._run_directory(run_id)
            state = dict(self._read_state(run_dir))
            request = self._read_request(run_dir)
            status = self._effective_status(state, run_id)
            if status == "INTERRUPTED":
                state["stage"] = "interrupted"
                state["message"] = "The owning process stopped before this Harvest job finished."
            receipt = self._read_json(run_dir / "authorization_receipt.json")
            if receipt.get("schema_version") != HARVEST_AUTHORIZATION_SCHEMA or receipt.get("run_id") != run_id:
                raise HarvestError("invalid_harvest_authorization", "Harvest authorization evidence is invalid.")
            return {
                "schema_version": "spritelab.harvest.job.v2",
                "run_id": run_id,
                "source_id": request["source_id"],
                "retry_of": request.get("retry_of"),
                "status": status,
                "stage": state.get("stage"),
                "current": state.get("current", 0),
                "total": state.get("total"),
                "message": state.get("message", ""),
                "created_at": state.get("created_at"),
                "started_at": state.get("started_at"),
                "updated_at": state.get("updated_at"),
                "ended_at": state.get("ended_at"),
                "handoff_ready": bool(state.get("handoff_ready", False)),
                "usable_count": int(state.get("usable_count", 0)),
                "quarantined_count": int(state.get("quarantined_count", 0)),
                "taxonomy_counts": dict(state.get("taxonomy_counts", {})),
                "limits": receipt.get("limits"),
                "authorization": {
                    "zero_cost": receipt.get("authorizations", {}).get("zero_cost") is True,
                    "permissive_license": receipt.get("authorizations", {}).get("permissive_license") is True,
                    "existing_inventory_reviewed": receipt.get("authorizations", {}).get("existing_inventory_reviewed")
                    is True,
                    "reuse_evidence": receipt.get("reuse_evidence"),
                },
                "provenance": receipt.get("source"),
                "backend_capabilities": receipt.get("backend_capabilities"),
                "events": self._read_events(run_dir),
                "dataset_import_available": self._dataset_import_callback is not None,
                "paths_exposed": False,
            }

    def handoff(self, run_id: str) -> dict[str, Any]:
        job = self.job(run_id)
        if job["status"] != "COMPLETE" or not job["handoff_ready"]:
            raise HarvestError(
                "harvest_handoff_not_ready",
                "Dataset handoff is available only after certified Harvest completion.",
            )
        source = self._sources.get(str(job["source_id"]))
        if source is None:
            raise HarvestError(
                "catalog_evidence_unavailable",
                "The exact source catalog evidence required by this handoff is unavailable.",
            )
        try:
            source.evidence_binding.validate(source.source_page, source.license_evidence_url)
        except ValueError as exc:
            raise HarvestError(
                "catalog_evidence_stale",
                "Source/license evidence is stale or changed; refresh the signed catalog before handoff.",
            ) from exc
        run_dir = self._run_directory(run_id)
        request = self._read_request(run_dir)
        if (
            request.get("source_catalog_identity") != source.catalog_identity
            or request.get("limits_identity") != self.limits.identity
            or self._backend_capabilities is None
            or request.get("backend_capability_identity") != self._backend_capabilities.identity
        ):
            raise HarvestError(
                "catalog_evidence_changed",
                "Current source, backend, or limit identities no longer match the authorized Harvest request.",
            )
        manifest = self._rehash_manifest(run_dir)
        payload = self._read_json(run_dir / "handoff.json")
        if (
            payload.get("schema_version") != HARVEST_HANDOFF_SCHEMA
            or payload.get("run_id") != run_id
            or payload.get("artifact_manifest_identity") != _json_identity(manifest)
            or payload.get("artifact_set_identity") != manifest["artifact_set_identity"]
        ):
            raise HarvestError("invalid_harvest_handoff", "Harvest handoff identity is invalid.")
        return {**payload, "dataset_import_available": self._dataset_import_callback is not None}

    def evidence(self, run_id: str) -> dict[str, Any]:
        """Return privacy-safe durable authorization, provenance, and limit evidence."""

        run_dir = self._run_directory(run_id)
        job = self.job(run_id)
        authorization = self._read_json(run_dir / "authorization_receipt.json")
        acquisition = (
            self._read_json(run_dir / "acquisition_receipt.json")
            if os.path.lexists(run_dir / "acquisition_receipt.json")
            else None
        )
        manifest = None
        if os.path.lexists(run_dir / "artifact_manifest.json"):
            manifest = (
                self._rehash_manifest(run_dir)
                if job["status"] == "COMPLETE"
                else self._read_json(run_dir / "artifact_manifest.json")
            )
        return {
            "schema_version": "spritelab.harvest.durable-evidence.v1",
            "run_id": run_id,
            "status": job["status"],
            "job": job,
            "authorization_receipt": authorization,
            "acquisition_receipt": acquisition,
            "artifact_manifest": manifest,
            "paths_exposed": False,
        }

    def import_to_dataset(
        self,
        run_id: str,
        *,
        explicit_action: bool,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if explicit_action is not True:
            raise HarvestError(
                "explicit_action_required", "Dataset import requires an explicit action.", status_code=422
            )
        _validate_idempotency_key(idempotency_key)
        callback = self._dataset_import_callback
        if callback is None:
            raise HarvestError(
                "dataset_import_unavailable",
                "No trusted Dataset import callback is configured for Harvest.",
            )
        handoff = self.handoff(run_id)
        run_dir = self._run_directory(run_id)
        manifest = self._rehash_manifest(run_dir)
        request_payload = {
            "schema_version": "spritelab.harvest.dataset-import-request.v1",
            "run_id": run_id,
            "idempotency_key": idempotency_key,
            "callback_id": callback.callback_id,
            "callback_code_identity_sha256": callback.code_identity_sha256,
            "handoff_identity": _json_identity(handoff),
            "artifact_manifest_identity": _json_identity(manifest),
            "created_at": _utc_now(),
        }
        request_path = run_dir / "dataset_import_request.json"
        receipt_path = run_dir / "dataset_import_receipt.json"
        with self._mutation_guard():
            if os.path.lexists(request_path):
                existing = self._read_json(request_path)
                comparable = dict(existing)
                comparable.pop("created_at", None)
                expected = dict(request_payload)
                expected.pop("created_at", None)
                if comparable != expected:
                    raise HarvestError(
                        "idempotency_conflict",
                        "Dataset import idempotency key is bound to different identities.",
                    )
                if os.path.lexists(receipt_path):
                    return self._read_json(receipt_path)
            else:
                self._write_exclusive_json(request_path, request_payload)
        try:
            result = callback.import_harvest(
                DatasetImportRequest(run_id, run_dir / "artifacts", handoff, manifest),
                idempotency_key=idempotency_key,
            )
        except Exception as exc:
            raise HarvestError(
                "dataset_import_failed",
                "The Dataset import callback failed without exposing private details.",
            ) from exc
        receipt = {
            "schema_version": HARVEST_IMPORT_RECEIPT_SCHEMA,
            "run_id": run_id,
            "idempotency_key": idempotency_key,
            "callback_id": callback.callback_id,
            "callback_code_identity_sha256": callback.code_identity_sha256,
            "dataset_reference": result.dataset_reference,
            "accepted_count": result.accepted_count,
            "quarantined_count": result.quarantined_count,
            "artifact_manifest_identity": _json_identity(manifest),
            "paths_exposed": False,
            "created_at": _utc_now(),
        }
        with self._mutation_guard():
            if os.path.lexists(receipt_path):
                return self._read_json(receipt_path)
            self._write_exclusive_json(receipt_path, receipt)
        return receipt

    def _start(
        self,
        source_id: str,
        *,
        idempotency_key: str,
        explicit_action: bool,
        authorize_zero_cost: bool,
        authorize_permissive_license: bool,
        authorize_existing_inventory_reviewed: bool,
        reuse_evidence: Mapping[str, Any] | ReuseEvidence,
        retry_of: str | None,
    ) -> tuple[dict[str, Any], bool]:
        source = self._authorize(
            source_id,
            idempotency_key=idempotency_key,
            explicit_action=explicit_action,
            authorize_zero_cost=authorize_zero_cost,
            authorize_permissive_license=authorize_permissive_license,
            authorize_existing_inventory_reviewed=authorize_existing_inventory_reviewed,
        )
        reuse = ReuseEvidence.from_value(reuse_evidence)
        if retry_of is not None and RUN_ID_PATTERN.fullmatch(retry_of) is None:
            raise HarvestError("invalid_harvest_run", "Harvest run identifier is invalid.", status_code=404)
        fingerprint = _json_identity(
            {
                "source_catalog_identity": source.catalog_identity,
                "backend_capability_identity": self._backend_capabilities.identity,
                "limits_identity": self.limits.identity,
                "reuse_evidence_identity": reuse.identity,
                "retry_of": retry_of,
            }
        )
        # Required passive inventory occurs before even backend construction.
        self.inventory()
        self._create_output_root()
        with self._mutation_guard():
            existing = self._find_idempotency_key(idempotency_key)
            if existing is not None:
                existing_run, existing_fingerprint = existing
                if existing_fingerprint != fingerprint:
                    raise HarvestError(
                        "idempotency_conflict",
                        "That idempotency key is bound to different source, backend, limits, or reuse identities.",
                    )
                existing_job = self.job(existing_run)
                if existing_job["status"] == "COMPLETE":
                    self._rehash_manifest(self._run_directory(existing_run))
                return existing_job, False
            current_inventory = self.inventory()
            reuse.validate(current_inventory)
            conflict = self._active_source_run(source_id)
            if conflict is not None:
                raise HarvestError(
                    "harvest_source_active_conflict",
                    f"Source {source_id!r} already has active managed run {conflict!r}.",
                )
            run_id, run_dir = self._create_run_directory()
            created_at = _utc_now()
            request = {
                "schema_version": HARVEST_REQUEST_SCHEMA,
                "run_id": run_id,
                "source_id": source_id,
                "retry_of": retry_of,
                "idempotency_key": idempotency_key,
                "request_fingerprint": fingerprint,
                "source_catalog_identity": source.catalog_identity,
                "backend_capability_identity": self._backend_capabilities.identity,
                "limits_identity": self.limits.identity,
                "reuse_evidence_identity": reuse.identity,
                "created_at": created_at,
                "browser_paths_accepted": False,
            }
            receipt = {
                "schema_version": HARVEST_AUTHORIZATION_SCHEMA,
                "run_id": run_id,
                "source": source.to_public_dict(),
                "backend_capabilities": {
                    **self._backend_capabilities.to_dict(),
                    "capability_identity": self._backend_capabilities.identity,
                },
                "limits": {**self.limits.to_dict(), "limits_identity": self.limits.identity},
                "reuse_evidence": {**reuse.to_dict(), "reuse_evidence_identity": reuse.identity},
                "authorizations": {
                    "explicit_action": True,
                    "zero_cost": True,
                    "permissive_license": True,
                    "existing_inventory_reviewed": True,
                },
                "inventory_before_action": {
                    "run_count": current_inventory["run_count"],
                    "legacy_run_count": current_inventory["legacy_run_count"],
                    "known_usable_items": current_inventory["known_usable_items"],
                    "unsafe_entries": current_inventory["unsafe_entries"],
                    "inventory_identity": current_inventory["inventory_identity"],
                },
                "network_actions_before_receipt": 0,
                "paths_exposed": False,
                "created_at": created_at,
            }
            self._write_exclusive_json(run_dir / "authorization_receipt.json", receipt)
            self._write_exclusive_json(run_dir / "request.json", request)
            state = {
                "schema_version": HARVEST_STATE_SCHEMA,
                "run_id": run_id,
                "source_id": source_id,
                "status": "QUEUED",
                "stage": "queued",
                "current": 0,
                "total": None,
                "message": "Harvest job queued after certified authorization evidence was recorded.",
                "created_at": created_at,
                "started_at": None,
                "updated_at": created_at,
                "ended_at": None,
                "handoff_ready": False,
                "usable_count": 0,
                "quarantined_count": 0,
                "taxonomy_counts": {},
                "event_count": 1,
                "owner_pid": os.getpid(),
                "owner_instance_id": self._instance_id,
            }
            self._write_state(run_dir, state)
            self._append_event(run_dir, state)
            cancellation = threading.Event()
            worker = threading.Thread(
                target=self._run_worker,
                args=(run_id, source, cancellation),
                name=f"spritelab-{run_id}",
                daemon=True,
            )
            self._cancellations[run_id] = cancellation
            self._workers[run_id] = worker
            worker.start()
            return self.job(run_id), True

    def _authorize(
        self,
        source_id: str,
        *,
        idempotency_key: str,
        explicit_action: bool,
        authorize_zero_cost: bool,
        authorize_permissive_license: bool,
        authorize_existing_inventory_reviewed: bool,
    ) -> HarvestSource:
        _validate_idempotency_key(idempotency_key)
        if explicit_action is not True:
            raise HarvestError(
                "explicit_action_required", "Starting Harvest requires an explicit action.", status_code=422
            )
        source = self._sources.get(source_id)
        if source is None:
            raise HarvestError("unknown_harvest_source", "Select a configured Harvest source.", status_code=404)
        if not self.acquisition_configured:
            raise HarvestError(
                "harvest_backend_unavailable",
                "No separately certified Harvest backend is configured.",
            )
        if authorize_zero_cost is not True or source.zero_cost is not True:
            raise HarvestError(
                "zero_cost_authorization_required",
                "Harvest is limited to an explicitly confirmed zero-cost source.",
                status_code=422,
            )
        if (
            authorize_permissive_license is not True
            or source.permissive is not True
            or source.normalized_license_id not in INITIAL_LICENSE_POLICY
        ):
            raise HarvestError(
                "permissive_license_authorization_required",
                "Harvest initially accepts only explicitly authorized CC0 or public-domain evidence.",
                status_code=422,
            )
        if authorize_existing_inventory_reviewed is not True:
            raise HarvestError(
                "existing_inventory_review_required",
                "Review managed and legacy Harvest inventory before authorizing new acquisition.",
                status_code=422,
            )
        try:
            source.evidence_binding.validate(source.source_page, source.license_evidence_url)
        except ValueError as exc:
            raise HarvestError(
                "catalog_evidence_stale",
                "The signed source/license evidence binding is absent, changed, or stale.",
            ) from exc
        return source

    def _run_worker(self, run_id: str, source: HarvestSource, cancellation: threading.Event) -> None:
        try:
            run_dir = self._run_directory(run_id)
            if cancellation.is_set():
                raise AcquisitionCancelled()
            self._transition(
                run_dir,
                "RUNNING",
                stage="acquiring",
                message="Certified acquisition is enforcing network and resource limits.",
                started=True,
            )
            artifacts = require_confined_path(run_dir / "artifacts", run_dir)
            artifacts.mkdir(exist_ok=False)
            if self._backend_factory is None:
                raise HarvestError("harvest_backend_unavailable", "Certified Harvest backend is unavailable.")
            backend = self._backend_factory()
            started = time.monotonic()
            result = backend.acquire(
                source,
                artifacts,
                self.limits,
                cancel_requested=cancellation.is_set,
                progress=lambda stage, current, total: self._progress(run_dir, cancellation, stage, current, total),
            )
            actual_elapsed = time.monotonic() - started
            if cancellation.is_set():
                raise AcquisitionCancelled()
            acquisition_receipt = self._validate_acquisition_result(source, result, actual_elapsed)
            manifest = scan_artifacts(artifacts, self.limits, expected_files=result.receipt.files)
            acquisition_receipt["artifact_manifest_identity"] = _json_identity(manifest)
            acquisition_receipt["acquisition_receipt_identity"] = _json_identity(acquisition_receipt)
            self._write_exclusive_json(run_dir / "acquisition_receipt.json", acquisition_receipt)
            self._write_exclusive_json(run_dir / "artifact_manifest.json", manifest)
            if cancellation.is_set():
                raise AcquisitionCancelled()
            handoff = self._build_handoff(run_id, source, acquisition_receipt, manifest)
            # Cancellation and handoff publication share the cross-process
            # mutation lock. This closes both pre- and post-publication races.
            with self._mutation_guard():
                if cancellation.is_set():
                    raise AcquisitionCancelled()
                self._write_exclusive_json(run_dir / "handoff.json", handoff)
                if cancellation.is_set():
                    raise AcquisitionCancelled()
                self._transition_locked(
                    run_dir,
                    "COMPLETE",
                    stage="complete",
                    current=manifest["artifact_count"],
                    total=manifest["artifact_count"],
                    message="Harvest completed with verified provenance and a rehashable Dataset handoff.",
                    ended=True,
                    handoff_ready=True,
                    manifest=manifest,
                )
        except Exception:
            status = "CANCELLED" if cancellation.is_set() else "FAILED"
            stage = "cancelled" if cancellation.is_set() else "failed"
            message = (
                "Harvest was cancelled; immutable receipts and artifacts were retained."
                if cancellation.is_set()
                else "Harvest failed closed. Private backend details were not recorded."
            )
            try:
                run_dir = self._run_directory(run_id)
                self._transition(run_dir, status, stage=stage, message=message, ended=True)
            except Exception:
                pass

    def _validate_acquisition_result(
        self,
        source: HarvestSource,
        result: AcquisitionResult,
        actual_elapsed: float,
    ) -> dict[str, Any]:
        if not isinstance(result, AcquisitionResult):
            raise HarvestError("invalid_backend_receipt", "Certified backend returned no valid receipt.")
        receipt = result.receipt
        if receipt.backend_capability_identity != self._backend_capabilities.identity:
            raise HarvestError("backend_identity_changed", "Backend capability identity changed during acquisition.")
        if not 200 <= receipt.http_status < 300:
            raise HarvestError("harvest_http_failed", "Harvest response did not have a successful HTTP status.")
        if receipt.response_mime_type not in self.limits.allowed_response_mime_types:
            raise HarvestError("harvest_response_mime_blocked", "Harvest response MIME type is not allowed.")
        if (
            receipt.expected_response_sha256 != source.expected_response_sha256
            or receipt.actual_response_sha256 != source.expected_response_sha256
        ):
            raise HarvestError("harvest_response_hash_mismatch", "Harvest response hash does not match the catalog.")
        if type(receipt.response_bytes) is not int or not 0 <= receipt.response_bytes <= self.limits.max_response_bytes:
            raise HarvestError("harvest_response_too_large", "Harvest response exceeded the certified byte limit.")
        if (
            not 0 <= receipt.elapsed_seconds <= self.limits.max_duration_seconds
            or actual_elapsed > self.limits.max_duration_seconds
        ):
            raise HarvestError("harvest_duration_exceeded", "Harvest exceeded the certified duration limit.")
        if len(receipt.redirect_chain) > self.limits.max_redirects:
            raise HarvestError("harvest_redirect_limit", "Harvest redirect count exceeded the configured limit.")
        urls = (*receipt.redirect_chain, receipt.final_url)
        redirect_evidence: list[dict[str, Any]] = []
        for value in urls:
            try:
                parsed = validate_download_url(value, source.normalized_download_hosts)
            except ValueError as exc:
                raise HarvestError(
                    "harvest_url_policy_blocked",
                    "Harvest direct or redirect URL violated the certified host/SSRF policy.",
                ) from exc
            if parsed.scheme.casefold() != "https":
                raise HarvestError("harvest_https_required", "Harvest direct and redirect URLs must remain HTTPS.")
            redirect_evidence.append(
                {
                    "url": public_download_url(value, source.normalized_download_hosts),
                    "url_sha256": url_identity(value),
                }
            )
        if (
            type(receipt.archive_members) is not int
            or not 0 <= receipt.archive_members <= self.limits.max_archive_members
            or type(receipt.archive_uncompressed_bytes) is not int
            or not 0 <= receipt.archive_uncompressed_bytes <= self.limits.max_archive_uncompressed_bytes
        ):
            raise HarvestError("harvest_archive_limit", "Harvest archive expansion evidence exceeded its limits.")
        if len(receipt.files) > self.limits.max_files:
            raise HarvestError("harvest_file_limit", "Harvest file receipt exceeded its count limit.")
        return {
            "schema_version": HARVEST_ACQUISITION_RECEIPT_SCHEMA,
            "source_id": source.source_id,
            "source_catalog_identity": source.catalog_identity,
            "source_evidence_binding_identity": source.evidence_binding.identity,
            "backend_capabilities": {
                **self._backend_capabilities.to_dict(),
                "capability_identity": self._backend_capabilities.identity,
            },
            "limits": {**self.limits.to_dict(), "limits_identity": self.limits.identity},
            "final_url": redirect_evidence[-1]["url"],
            "final_url_sha256": redirect_evidence[-1]["url_sha256"],
            "redirect_chain": redirect_evidence[:-1],
            "http_status": receipt.http_status,
            "response_mime_type": receipt.response_mime_type,
            "expected_response_sha256": receipt.expected_response_sha256,
            "actual_response_sha256": receipt.actual_response_sha256,
            "response_bytes": receipt.response_bytes,
            "reported_elapsed_seconds": receipt.elapsed_seconds,
            "observed_elapsed_within_limit": True,
            "archive_members": receipt.archive_members,
            "archive_uncompressed_bytes": receipt.archive_uncompressed_bytes,
            "reported_file_count": len(receipt.files),
            "private_url_components_exposed": False,
            "created_at": _utc_now(),
        }

    def _progress(
        self,
        run_dir: Path,
        cancellation: threading.Event,
        stage: str,
        current: int,
        total: int | None,
    ) -> None:
        if cancellation.is_set():
            raise AcquisitionCancelled()
        safe_stage = stage if STAGE_PATTERN.fullmatch(stage or "") else "acquiring"
        if type(current) is not int or current < 0:
            raise HarvestError("invalid_backend_progress", "Harvest backend progress was invalid.")
        if total is not None and (type(total) is not int or total < current):
            raise HarvestError("invalid_backend_progress", "Harvest backend progress was invalid.")
        self._transition(
            run_dir,
            "RUNNING",
            stage=safe_stage,
            current=current,
            total=total,
            message="Certified acquisition is running within enforced limits.",
        )

    def _build_handoff(
        self,
        run_id: str,
        source: HarvestSource,
        acquisition_receipt: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_snapshot = source.to_public_dict()
        provenance_identity = _json_identity(
            {
                "source": source_snapshot,
                "acquisition_receipt_identity": acquisition_receipt["acquisition_receipt_identity"],
            }
        )
        return {
            "schema_version": HARVEST_HANDOFF_SCHEMA,
            "run_id": run_id,
            "source_id": source.source_id,
            "managed_reference": {"kind": "harvest_run", "run_id": run_id},
            "source": source_snapshot,
            "provenance_identity": provenance_identity,
            "source_evidence_binding_identity": source.evidence_binding.identity,
            "backend_capability_identity": self._backend_capabilities.identity,
            "limits_identity": self.limits.identity,
            "acquisition_receipt_identity": acquisition_receipt["acquisition_receipt_identity"],
            "artifact_manifest_identity": _json_identity(manifest),
            "artifact_set_identity": manifest["artifact_set_identity"],
            "artifact_count": manifest["artifact_count"],
            "usable_count": manifest["usable_count"],
            "quarantined_count": manifest["quarantined_count"],
            "total_bytes": manifest["total_bytes"],
            "taxonomy_counts": manifest["taxonomy_counts"],
            "files": manifest["files"],
            "license": source_snapshot["license"],
            "handoff_ready": True,
            "portable_relative_paths": True,
            "paths_exposed": False,
            "created_at": _utc_now(),
        }

    def _rehash_manifest(self, run_dir: Path) -> dict[str, Any]:
        stored = self._read_json(run_dir / "artifact_manifest.json")
        files = stored.get("files")
        if not isinstance(files, list):
            raise HarvestError("invalid_artifact_manifest", "Harvest artifact manifest is invalid.")
        try:
            expected = tuple(
                AcquiredFile(
                    relative_path=str(item["relative_path"]),
                    byte_count=_exact_int(item["byte_count"]),
                    sha256=str(item["expected_sha256"]),
                    mime_type=str(item["mime_type"]),
                    usable=item.get("usable") is True,
                    quarantine_reason=item.get("quarantine_reason"),
                    taxonomy=tuple(item.get("taxonomy", ())),
                )
                for item in files
            )
            current = scan_artifacts(run_dir / "artifacts", self.limits, expected_files=expected)
        except (KeyError, TypeError, ValueError, HarvestStorageError, OSError) as exc:
            raise HarvestError(
                "harvest_artifact_verification_failed",
                "Harvest artifacts changed or no longer satisfy their manifest and safety limits.",
            ) from exc
        if current != stored:
            raise HarvestError(
                "harvest_artifact_verification_failed",
                "Harvest artifact manifest identity changed during rehash.",
            )
        return current

    def _transition(
        self,
        run_dir: Path,
        status: str,
        *,
        stage: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
        started: bool = False,
        ended: bool = False,
        handoff_ready: bool | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        with self._mutation_guard():
            self._transition_locked(
                run_dir,
                status,
                stage=stage,
                message=message,
                current=current,
                total=total,
                started=started,
                ended=ended,
                handoff_ready=handoff_ready,
                manifest=manifest,
            )

    def _transition_locked(
        self,
        run_dir: Path,
        status: str,
        *,
        stage: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
        started: bool = False,
        ended: bool = False,
        handoff_ready: bool | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        state = self._read_state(run_dir)
        now = _utc_now()
        state.update({"status": status, "stage": stage, "message": message, "updated_at": now})
        if current is not None:
            state["current"] = current
            state["total"] = total
        if started and state.get("started_at") is None:
            state["started_at"] = now
        if ended:
            state["ended_at"] = now
        if handoff_ready is not None:
            state["handoff_ready"] = handoff_ready
        if manifest is not None:
            state["usable_count"] = manifest["usable_count"]
            state["quarantined_count"] = manifest["quarantined_count"]
            state["taxonomy_counts"] = manifest["taxonomy_counts"]
        event_count = int(state.get("event_count", 0))
        terminal = status in TERMINAL_STATUSES
        append_event = event_count < self.limits.max_events
        if not terminal and event_count >= self.limits.max_events - 1:
            raise HarvestError("harvest_event_limit", "Harvest progress event limit was reached.")
        if append_event:
            state["event_count"] = event_count + 1
        self._write_state(run_dir, state)
        if append_event:
            self._append_event(run_dir, state)

    def _managed_inventory_record(self, run_dir: Path) -> dict[str, Any]:
        state = self._read_state(run_dir)
        request = self._read_request(run_dir)
        return {
            "run_id": run_dir.name,
            "kind": "managed",
            "source_id": request["source_id"],
            "status": self._effective_status(state, run_dir.name),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "handoff_ready": bool(state.get("handoff_ready", False)),
            "usable_count": int(state.get("usable_count", 0)),
            "quarantined_count": int(state.get("quarantined_count", 0)),
        }

    def _effective_status(self, state: Mapping[str, Any], run_id: str) -> str:
        status = str(state["status"])
        if status not in ACTIVE_STATUSES:
            return status
        worker = self._workers.get(run_id)
        if worker is not None and worker.is_alive():
            return status
        owner_pid = state.get("owner_pid")
        owner_instance = state.get("owner_instance_id")
        if owner_instance == self._instance_id:
            return "INTERRUPTED"
        if type(owner_pid) is int and _pid_alive(owner_pid):
            return status
        return "INTERRUPTED"

    def _active_source_run(self, source_id: str) -> str | None:
        if not self.output_root.exists():
            return None
        for child in sorted(self.output_root.iterdir(), key=lambda value: value.name):
            if RUN_ID_PATTERN.fullmatch(child.name) is None or _is_link_or_reparse(child):
                continue
            try:
                request = self._read_request(child)
                state = self._read_state(child)
            except (HarvestError, OSError):
                continue
            if request["source_id"] == source_id and self._effective_status(state, child.name) in ACTIVE_STATUSES:
                return child.name
        return None

    def _create_output_root(self) -> None:
        try:
            self.output_root.mkdir(parents=False, exist_ok=True)
            self._validate_output_root()
        except (OSError, HarvestError, UnsafeFilesystemOperation) as exc:
            raise HarvestError(
                "unsafe_harvest_root",
                "The managed Harvest root could not be created safely.",
            ) from exc

    def _validate_output_root(self) -> None:
        require_confined_path(self.output_root, self.project_root)
        metadata = self.output_root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or _metadata_is_link_or_reparse(metadata) or self.output_root.is_mount():
            raise HarvestError("unsafe_harvest_root", "The managed Harvest root is unsafe.")

    @contextmanager
    def _mutation_guard(self) -> Any:
        with self._lock:
            try:
                with RepositoryMutationLock(self.output_root):
                    yield
            except HarvestStorageError as exc:
                raise HarvestError("harvest_mutation_conflict", str(exc)) from exc

    def _create_run_directory(self) -> tuple[str, Path]:
        for _attempt in range(32):
            run_id = self._run_id_factory()
            if RUN_ID_PATTERN.fullmatch(run_id) is None:
                raise HarvestError("invalid_generated_run_id", "Harvest generated an unsafe run identifier.")
            run_dir = require_confined_path(self.output_root / run_id, self.output_root)
            try:
                run_dir.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return run_id, run_dir
        raise HarvestError("harvest_run_collision", "Harvest could not allocate a unique run.")

    def _find_idempotency_key(self, key: str) -> tuple[str, str] | None:
        if not self.output_root.exists():
            return None
        for child in sorted(self.output_root.iterdir(), key=lambda value: value.name):
            if RUN_ID_PATTERN.fullmatch(child.name) is None or _is_link_or_reparse(child):
                continue
            try:
                request = self._read_request(child)
            except (HarvestError, OSError):
                continue
            if request.get("idempotency_key") == key:
                return child.name, str(request.get("request_fingerprint", ""))
        return None

    def _run_directory(self, run_id: str) -> Path:
        if RUN_ID_PATTERN.fullmatch(run_id or "") is None:
            raise HarvestError("invalid_harvest_run", "Harvest run identifier is invalid.", status_code=404)
        try:
            run_dir = require_confined_path(self.output_root / run_id, self.output_root)
        except UnsafeFilesystemOperation as exc:
            raise HarvestError("invalid_harvest_run", "Harvest run identifier is invalid.", status_code=404) from exc
        if not run_dir.is_dir() or _is_link_or_reparse(run_dir) or run_dir.is_mount():
            raise HarvestError("harvest_run_not_found", "Harvest run was not found.", status_code=404)
        return run_dir

    def _read_request(self, run_dir: Path) -> dict[str, Any]:
        payload = self._read_json(run_dir / "request.json")
        if (
            payload.get("schema_version") != HARVEST_REQUEST_SCHEMA
            or payload.get("run_id") != run_dir.name
            or SOURCE_ID_PATTERN.fullmatch(str(payload.get("source_id", ""))) is None
            or SHA256_PATTERN.fullmatch(str(payload.get("request_fingerprint", ""))) is None
        ):
            raise HarvestError("invalid_harvest_request", "Harvest request evidence is invalid.")
        return payload

    def _read_state(self, run_dir: Path) -> dict[str, Any]:
        payload = self._read_json(run_dir / "state.json")
        if (
            payload.get("schema_version") != HARVEST_STATE_SCHEMA
            or payload.get("run_id") != run_dir.name
            or payload.get("status") not in ACTIVE_STATUSES | TERMINAL_STATUSES
        ):
            raise HarvestError("invalid_harvest_state", "Harvest state evidence is invalid.")
        return payload

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            require_confined_path(path, self.output_root)
            metadata = path.lstat()
        except (FileNotFoundError, UnsafeFilesystemOperation) as exc:
            raise HarvestError("missing_harvest_evidence", "Harvest durable evidence is incomplete.") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_METADATA_BYTES
        ):
            raise HarvestError("unsafe_harvest_evidence", "Harvest durable evidence is unsafe.")
        try:
            payload = strict_json_loads(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise HarvestError("invalid_harvest_evidence", "Harvest durable evidence is invalid.") from exc
        if not isinstance(payload, Mapping):
            raise HarvestError("invalid_harvest_evidence", "Harvest durable evidence is invalid.")
        return dict(payload)

    def _read_events(self, run_dir: Path) -> list[dict[str, Any]]:
        path = run_dir / "events.jsonl"
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return []
        maximum = self.limits.max_events * self.limits.max_event_bytes
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum
        ):
            return []
        events: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-200:]:
                payload = strict_json_loads(line)
                if isinstance(payload, Mapping) and payload.get("schema_version") == HARVEST_EVENT_SCHEMA:
                    events.append(dict(payload))
        except (OSError, ValueError):
            return []
        return events

    def _write_state(self, run_dir: Path, state: Mapping[str, Any]) -> None:
        target = require_confined_path(run_dir / "state.json", run_dir)
        atomic_write_text(target, strict_json_dumps(dict(state), sort_keys=True, separators=(",", ":")))

    def _write_exclusive_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        try:
            require_confined_path(path, self.output_root)
        except UnsafeFilesystemOperation as exc:
            raise HarvestError("unsafe_harvest_evidence", "Harvest durable evidence is unsafe.") from exc
        content = strict_json_dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _append_event(self, run_dir: Path, state: Mapping[str, Any]) -> None:
        event = {
            "schema_version": HARVEST_EVENT_SCHEMA,
            "sequence": state["event_count"],
            "run_id": state["run_id"],
            "timestamp": state["updated_at"],
            "status": state["status"],
            "stage": state["stage"],
            "current": state.get("current", 0),
            "total": state.get("total"),
            "message": state.get("message", ""),
            "paths_exposed": False,
        }
        line = strict_json_dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        if len(line.encode("utf-8")) > self.limits.max_event_bytes:
            raise HarvestError("harvest_event_too_large", "Harvest event exceeded its byte limit.")
        path = require_confined_path(run_dir / "events.jsonl", run_dir)
        mode = "x"
        if os.path.lexists(path):
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _metadata_is_link_or_reparse(metadata)
                or metadata.st_nlink != 1
                or metadata.st_size + len(line.encode("utf-8")) > self.limits.max_events * self.limits.max_event_bytes
            ):
                raise HarvestError("unsafe_harvest_evidence", "Harvest event stream is unsafe or capped.")
            mode = "a"
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _validate_idempotency_key(value: str) -> None:
    if IDEMPOTENCY_KEY_PATTERN.fullmatch(value or "") is None:
        raise HarvestError(
            "invalid_idempotency_key",
            "Provide an idempotency key between 8 and 128 safe characters.",
            status_code=422,
        )


def _exact_int(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("not exact int")
    return value


def _default_run_id() -> str:
    return f"harvest-{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_identity(value: Any) -> str:
    payload = strict_json_dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_FLAG)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        return _metadata_is_link_or_reparse(path.lstat())
    except FileNotFoundError:
        return False


__all__ = [
    "HARVEST_ACQUISITION_RECEIPT_SCHEMA",
    "HARVEST_AUTHORIZATION_SCHEMA",
    "HARVEST_HANDOFF_SCHEMA",
    "HARVEST_IMPORT_RECEIPT_SCHEMA",
    "HARVEST_INVENTORY_SCHEMA",
    "HARVEST_REQUEST_SCHEMA",
    "HARVEST_STATE_SCHEMA",
    "AcquisitionCancelled",
    "HarvestError",
    "HarvestService",
    "HarvestSource",
    "ReuseEvidence",
]
