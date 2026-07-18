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
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
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
from spritelab.product_features.harvest.certification import (
    BackendCapabilityEvidence,
    evidence_has_current_validation_snapshot,
)
from spritelab.product_features.harvest.onboarding import (
    ACTIVE_PROBE_STATUSES,
    PROBE_ID_PATTERN,
    CatalogProbeError,
    CatalogProbeService,
    scan_probe_inventory_record,
)
from spritelab.product_features.harvest.storage import (
    HarvestStorageError,
    RepositoryMutationLock,
    append_stable_single_link_bytes,
    read_stable_single_link_bytes,
    scan_artifacts,
    scan_legacy_run,
    write_atomic_stable_bytes,
    write_exclusive_stable_bytes,
)
from spritelab.product_features.harvest.trusted_backend import (
    AcquiredFile,
    AcquisitionResult,
    BackendFactory,
    CertifiedBackendCapabilities,
    DatasetImportCallback,
    DatasetImportCancelled,
    DatasetImportDeadlineExceeded,
    DatasetImportRequest,
    DatasetImportResult,
    HarvestLimits,
    conditioned_dataset_import_callback_binding,
    hardened_backend_code_identity,
    validate_callback_identity,
)
from spritelab.utils.safe_fs import (
    AnchoredDirectory,
    OwnedFileIdentity,
    UnsafeFilesystemOperation,
    open_anchored_directory,
    require_confined_path,
)

HARVEST_INVENTORY_SCHEMA = "spritelab.harvest.inventory.v3"
HARVEST_REQUEST_SCHEMA = "spritelab.harvest.request.v2"
HARVEST_AUTHORIZATION_SCHEMA = "spritelab.harvest.authorization-receipt.v2"
HARVEST_STATE_SCHEMA = "spritelab.harvest.job-state.v2"
HARVEST_EVENT_SCHEMA = "spritelab.harvest.job-event.v2"
HARVEST_ACQUISITION_RECEIPT_SCHEMA = "spritelab.harvest.acquisition-receipt.v2"
HARVEST_HANDOFF_SCHEMA = "spritelab.harvest.dataset-handoff.v2"
HARVEST_IMPORT_RECEIPT_SCHEMA = "spritelab.harvest.dataset-import-receipt.v1"
HARVEST_IMPORT_REQUEST_SCHEMA = "spritelab.harvest.dataset-import-request.v2"
HARVEST_IMPORT_STATE_SCHEMA = "spritelab.harvest.dataset-import-state.v1"
HARVEST_IMPORT_CANCELLATION_SCHEMA = "spritelab.harvest.dataset-import-cancellation.v1"
HARVEST_IMPORT_TERMINAL_COMMIT_SCHEMA = "spritelab.harvest.dataset-import-terminal-commit.v1"
HARVEST_CANCELLATION_SCHEMA = "spritelab.harvest.cancellation-request.v1"
HARVEST_TERMINAL_COMMIT_SCHEMA = "spritelab.harvest.terminal-commit.v1"
HARVEST_LEASE_SCHEMA = "spritelab.harvest.worker-lease.v1"

_IMPORT_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "callback_id",
        "callback_code_identity_sha256",
        "callback_runtime_identity_sha256",
        "handoff_identity",
        "artifact_manifest_identity",
        "request_identity",
        "idempotency_key",
        "created_at",
        "paths_exposed",
    }
)
_IMPORT_STATE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "request_identity",
        "attempt",
        "status",
        "started_at",
        "updated_at",
        "deadline_at",
        "error_code",
        "paths_exposed",
    }
)
_IMPORT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "request_identity",
        "idempotency_key",
        "callback_id",
        "callback_code_identity_sha256",
        "callback_runtime_identity_sha256",
        "dataset_reference",
        "accepted_count",
        "quarantined_count",
        "artifact_manifest_identity",
        "paths_exposed",
        "created_at",
    }
)
_IMPORT_CANCELLATION_KEYS = frozenset(
    {"schema_version", "run_id", "attempt", "explicit_action", "requested_at", "paths_exposed"}
)
_IMPORT_TERMINAL_COMMIT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "request_identity",
        "attempt",
        "status",
        "receipt_identity",
        "state_identity",
        "created_at",
        "paths_exposed",
        "terminal_commit_identity",
    }
)

RUN_ID_PATTERN = re.compile(r"^harvest-[a-z0-9][a-z0-9-]{5,79}$")
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
STAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")

ACTIVE_STATUSES = frozenset({"QUEUED", "RUNNING", "CANCELLING"})
TERMINAL_STATUSES = frozenset({"COMPLETE", "FAILED", "CANCELLED"})
RETRYABLE_STATUSES = frozenset({"FAILED", "CANCELLED", "INTERRUPTED"})
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PROGRESS_EVENTS_PER_STAGE = 200
_DURABLE_CANCELLATION_POLL_SECONDS = 0.05
_LEASE_DURATION_SECONDS = 2.0
_LEASE_HEARTBEAT_SECONDS = 0.25
_LEGAL_TRANSITIONS = {
    "QUEUED": frozenset({"QUEUED", "RUNNING", "CANCELLING", "CANCELLED", "FAILED"}),
    "RUNNING": frozenset({"RUNNING", "CANCELLING", "CANCELLED", "COMPLETE", "FAILED"}),
    "CANCELLING": frozenset({"CANCELLING", "CANCELLED", "FAILED"}),
    "COMPLETE": frozenset({"COMPLETE"}),
    "FAILED": frozenset({"FAILED"}),
    "CANCELLED": frozenset({"CANCELLED"}),
}

LiveConfigurationLoader = Callable[
    [],
    tuple[tuple[HarvestSource, ...], BackendCapabilityEvidence | None],
]
LiveCapabilityEvidenceLoader = Callable[[], BackendCapabilityEvidence | None]


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
        backend_capability_evidence: BackendCapabilityEvidence | None = None,
        live_configuration_loader: LiveConfigurationLoader | None = None,
        live_capability_evidence_loader: LiveCapabilityEvidenceLoader | None = None,
        limits: HarvestLimits | None = None,
        run_id_factory: Any | None = None,
        dataset_import_callback: DatasetImportCallback | None = None,
        dataset_import_callback_factory: Callable[[], DatasetImportCallback] | None = None,
        probe_resolver: Any | None = None,
        probe_transport: Any | None = None,
        probe_downloader: Any | None = None,
        probe_run_id_factory: Any | None = None,
        allow_unverified_test_backend: bool = False,
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
        if backend_capability_evidence is not None and backend_capabilities != backend_capability_evidence.capabilities:
            raise ValueError("Harvest backend capability evidence does not match configured capabilities.")
        if type(allow_unverified_test_backend) is not bool:
            raise ValueError("Harvest test-backend authorization must be an exact boolean.")
        if (
            backend_factory is not None
            and not allow_unverified_test_backend
            and (backend_capability_evidence is None or live_configuration_loader is None)
        ):
            raise ValueError("Production Harvest backends require current independently reloaded capability evidence.")
        if backend_capability_evidence is not None and not allow_unverified_test_backend:
            _require_current_backend_evidence(backend_capability_evidence, backend_capabilities)
        if dataset_import_callback is not None and dataset_import_callback_factory is not None:
            raise ValueError("Dataset import callback and factory are mutually exclusive.")
        if dataset_import_callback is not None:
            _validate_callback_capability_binding(dataset_import_callback, backend_capabilities)
            if backend_capabilities is not None and not allow_unverified_test_backend:
                _validate_production_dataset_callback(dataset_import_callback, self.project_root)
        if dataset_import_callback_factory is not None and not callable(dataset_import_callback_factory):
            raise ValueError("Dataset import callback factory must be callable.")
        self._sources = {source.source_id: source for source in catalog}
        self._backend_factory = backend_factory
        self._backend_capabilities = backend_capabilities
        self._backend_capability_evidence = backend_capability_evidence
        self._live_configuration_loader = live_configuration_loader
        self._live_capability_evidence_loader = live_capability_evidence_loader
        self._allow_unverified_test_backend = allow_unverified_test_backend
        self.limits = limits or HarvestLimits()
        self._run_id_factory = run_id_factory or _default_run_id
        self._dataset_import_callback = dataset_import_callback
        self._dataset_import_callback_factory = dataset_import_callback_factory
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._progress_observations: dict[str, tuple[str, int, int | None, int, int]] = {}
        self._last_cancellation_probe: dict[str, tuple[float, bool]] = {}
        self._lease_stops: dict[str, threading.Event] = {}
        self._lease_failures: dict[str, threading.Event] = {}
        self._lease_threads: dict[str, threading.Thread] = {}
        self._import_cancellations: dict[str, threading.Event] = {}
        self._instance_id = uuid.uuid4().hex
        probe_options: dict[str, Any] = {}
        if probe_downloader is not None:
            probe_options["downloader"] = probe_downloader
        self._probe_service = CatalogProbeService(
            self.project_root,
            limits=self.limits,
            resolver=probe_resolver,
            transport=probe_transport,
            run_id_factory=probe_run_id_factory,
            catalog_refreshed=self._refresh_catalog_sources,
            **probe_options,
        )

    @property
    def acquisition_configured(self) -> bool:
        return bool(self._sources) and self._backend_factory is not None and self._backend_capabilities is not None

    @property
    def dataset_import_available(self) -> bool:
        return self._dataset_import_callback is not None or self._dataset_import_callback_factory is not None

    def _dataset_import_callback_for_action(self) -> DatasetImportCallback:
        """Resolve and attest the callback only at the explicit import boundary."""

        callback = self._dataset_import_callback
        if callback is None and self._dataset_import_callback_factory is not None:
            try:
                callback = self._dataset_import_callback_factory()
            except Exception as exc:
                raise HarvestError(
                    "dataset_import_identity_changed",
                    "The trusted Dataset import callback could not be attested for this action.",
                ) from exc
        if callback is None:
            raise HarvestError(
                "dataset_import_unavailable",
                "No trusted Dataset import callback is configured for Harvest.",
            )
        try:
            _validate_callback_capability_binding(callback, self._backend_capabilities)
            if self._backend_capabilities is not None and not self._allow_unverified_test_backend:
                _validate_production_dataset_callback(callback, self.project_root)
        except ValueError as exc:
            raise HarvestError(
                "dataset_import_identity_changed",
                "The Dataset import callback no longer matches the certified code and runtime binding.",
            ) from exc
        return callback

    def _backend_evidence_record(self) -> dict[str, Any]:
        if self._backend_capabilities is None:
            raise HarvestError("harvest_backend_unavailable", "Certified Harvest backend is unavailable.")
        if self._backend_capability_evidence is not None:
            if not self._allow_unverified_test_backend:
                _require_current_backend_evidence(
                    self._backend_capability_evidence,
                    self._backend_capabilities,
                )
            evidence = self._backend_capability_evidence.to_dict()
            return {**evidence, "evidence_identity": self._backend_capability_evidence.identity}
        if not self._allow_unverified_test_backend:
            raise HarvestError(
                "harvest_backend_evidence_required",
                "Current independent Harvest backend evidence is required.",
            )
        fallback = {
            "schema_version": "spritelab.harvest.explicit-test-backend-evidence.v1",
            "backend_capability_identity": self._backend_capabilities.identity,
            "auditor_id": None,
            "audited_at": None,
            "issued_at": None,
            "expires_at": None,
            "audit_report_sha256": None,
            "audit_report_identity": None,
            "certificate_identity": None,
            "implementation_identity_sha256": self._backend_capabilities.code_identity_sha256,
            "origin": "explicit_test_configuration",
        }
        return {**fallback, "evidence_identity": _json_identity(fallback)}

    def _validate_live_configuration(self, source_id: str) -> HarvestSource:
        source = self._sources.get(source_id)
        if source is None or self._backend_capabilities is None:
            raise HarvestError("harvest_backend_unavailable", "Certified Harvest configuration is unavailable.")
        try:
            source.evidence_binding.validate(source.source_page, source.license_evidence_url)
        except ValueError as exc:
            raise HarvestError(
                "catalog_evidence_stale",
                "Harvest source, license, or automation-terms evidence is stale or invalid.",
            ) from exc
        if self._live_configuration_loader is None:
            if not self._allow_unverified_test_backend:
                raise HarvestError(
                    "harvest_live_configuration_invalid",
                    "Current independent Harvest capability evidence cannot be reloaded.",
                )
            return source
        try:
            live_sources, live_evidence = self._live_configuration_loader()
        except Exception as exc:
            raise HarvestError(
                "harvest_live_configuration_invalid",
                "Repository Harvest catalog or capability evidence is no longer current.",
            ) from exc
        live = {item.source_id: item for item in live_sources}
        if live_evidence is not None and not self._allow_unverified_test_backend:
            try:
                _require_current_backend_evidence(live_evidence, self._backend_capabilities)
            except ValueError as exc:
                raise HarvestError(
                    "harvest_live_configuration_invalid",
                    "Current independent Harvest capability evidence is invalid or expired.",
                ) from exc
        current_source = live.get(source_id)
        if current_source is not None:
            try:
                current_source.evidence_binding.validate(
                    current_source.source_page,
                    current_source.license_evidence_url,
                )
            except ValueError as exc:
                raise HarvestError(
                    "catalog_evidence_stale",
                    "Live Harvest source, license, or automation-terms evidence is stale or invalid.",
                ) from exc
        if (
            current_source is None
            or current_source.catalog_identity != source.catalog_identity
            or live_evidence is None
            or self._backend_capability_evidence is None
            or live_evidence.identity != self._backend_capability_evidence.identity
            or live_evidence.capabilities.identity != self._backend_capabilities.identity
        ):
            raise HarvestError(
                "harvest_live_configuration_changed",
                "Repository Harvest catalog or independent capability evidence changed.",
            )
        return current_source

    def sources(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.sources.v2",
            "sources": [self._sources[key].to_public_dict() for key in sorted(self._sources)],
            "license_policy": sorted(INITIAL_LICENSE_POLICY),
            "backend_configured": self.acquisition_configured,
            "backend_capabilities": self._backend_capabilities.to_dict() if self._backend_capabilities else None,
            "backend_capability_evidence": (
                self._backend_evidence_record() if self._backend_capabilities is not None else None
            ),
            "limits": self.limits.to_dict(),
            "network_actions": 0,
            "browser_paths_accepted": False,
        }

    def _refresh_catalog_sources(self, sources: tuple[HarvestSource, ...]) -> None:
        with self._lock:
            self._sources = {source.source_id: source for source in sources}

    def _current_probe_capability_evidence(self) -> BackendCapabilityEvidence:
        evidence = self._backend_capability_evidence
        if self._live_capability_evidence_loader is not None:
            try:
                evidence = self._live_capability_evidence_loader()
            except Exception as exc:
                raise HarvestError(
                    "catalog_probe_capability_evidence_required",
                    "Current independent Harvest capability evidence could not be loaded.",
                ) from exc
            if self._live_configuration_loader is not None:
                try:
                    live_sources, _live_evidence = self._live_configuration_loader()
                except Exception:
                    pass
                else:
                    self._refresh_catalog_sources(live_sources)
        elif self._live_configuration_loader is not None:
            try:
                live_sources, evidence = self._live_configuration_loader()
            except Exception as exc:
                raise HarvestError(
                    "catalog_probe_capability_evidence_required",
                    "Current independent Harvest capability evidence could not be loaded.",
                ) from exc
            if evidence is not None:
                self._refresh_catalog_sources(live_sources)
        if (
            evidence is None
            or self._backend_capabilities is None
            or evidence.capabilities.identity != self._backend_capabilities.identity
        ):
            raise HarvestError(
                "catalog_probe_capability_evidence_required",
                "Current independent capability evidence is required before catalog onboarding.",
            )
        return evidence

    def source_prefill(
        self,
        source_page: str,
        *,
        preset_id: str,
        authorize_network: bool,
    ) -> Any:
        try:
            return self._probe_service.source_prefill(
                source_page,
                preset_id=preset_id,
                authorize_network=authorize_network,
            )
        except CatalogProbeError as exc:
            raise HarvestError(exc.code, str(exc), status_code=exc.status_code) from exc

    def start_probe(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        with self._lock:
            evidence = self._current_probe_capability_evidence()
            try:
                return self._probe_service.start(
                    payload,
                    capability_evidence=evidence,
                    current_inventory_identity=lambda: str(self.inventory()["inventory_identity"]),
                )
            except CatalogProbeError as exc:
                raise HarvestError(exc.code, str(exc), status_code=exc.status_code) from exc

    def retry_probe(self, probe_id: str, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        with self._lock:
            evidence = self._current_probe_capability_evidence()
            try:
                return self._probe_service.retry(
                    probe_id,
                    payload,
                    capability_evidence=evidence,
                    current_inventory_identity=lambda: str(self.inventory()["inventory_identity"]),
                )
            except CatalogProbeError as exc:
                raise HarvestError(exc.code, str(exc), status_code=exc.status_code) from exc

    def probe(self, probe_id: str) -> dict[str, Any]:
        try:
            return self._probe_service.status(probe_id)
        except CatalogProbeError as exc:
            raise HarvestError(exc.code, str(exc), status_code=exc.status_code) from exc

    def probe_evidence(self, probe_id: str) -> dict[str, Any]:
        try:
            return self._probe_service.evidence(probe_id)
        except CatalogProbeError as exc:
            raise HarvestError(exc.code, str(exc), status_code=exc.status_code) from exc

    def cancel_probe(self, probe_id: str, *, explicit_action: bool) -> dict[str, Any]:
        try:
            return self._probe_service.cancel(probe_id, explicit_action=explicit_action)
        except CatalogProbeError as exc:
            raise HarvestError(exc.code, str(exc), status_code=exc.status_code) from exc

    def promote_probe(
        self,
        probe_id: str,
        *,
        explicit_action: bool,
        authorize_catalog_promotion: bool,
        authorize_zero_cost_evidence_review: bool,
        reviewed_verification_identity: str | None,
        reviewed_source_pack_evidence_sha256: str | None,
    ) -> dict[str, Any]:
        evidence = self._current_probe_capability_evidence()
        try:
            return self._probe_service.promote(
                probe_id,
                explicit_action=explicit_action,
                authorize_catalog_promotion=authorize_catalog_promotion,
                authorize_zero_cost_evidence_review=authorize_zero_cost_evidence_review,
                reviewed_verification_identity=reviewed_verification_identity,
                reviewed_source_pack_evidence_sha256=reviewed_source_pack_evidence_sha256,
                capability_evidence=evidence,
            )
        except CatalogProbeError as exc:
            raise HarvestError(exc.code, str(exc), status_code=exc.status_code) from exc

    def inventory(self) -> dict[str, Any]:
        """Index immediate managed and legacy runs without mutation or network."""

        with self._lock:
            managed: list[dict[str, Any]] = []
            probes: list[dict[str, Any]] = []
            legacy: list[dict[str, Any]] = []
            unsafe_entries = 0
            collision_keys: set[str] = set()
            if os.path.lexists(self.output_root):
                self._validate_output_root()
                with open_anchored_directory(self.output_root, self.project_root) as output_anchor:
                    root_device = output_anchor.directory_metadata().st_dev
                    for name in output_anchor.names():
                        if name == ".harvest.lock":
                            continue
                        if len(managed) + len(probes) + len(legacy) + unsafe_entries >= self.limits.max_files:
                            raise HarvestError(
                                "harvest_inventory_limit",
                                "Immediate Harvest inventory exceeds the configured entry limit.",
                            )
                        collision = unicodedata.normalize("NFC", name).casefold()
                        if collision in collision_keys:
                            unsafe_entries += 1
                            continue
                        collision_keys.add(collision)
                        try:
                            metadata = output_anchor.lstat(name)
                            if (
                                not stat.S_ISDIR(metadata.st_mode)
                                or _metadata_is_link_or_reparse(metadata)
                                or metadata.st_dev != root_device
                            ):
                                raise HarvestStorageError("unsafe immediate child")
                            child = self.output_root / name
                            managed_record: dict[str, Any] | None = None
                            probe_record: dict[str, Any] | None = None
                            legacy_record: dict[str, Any] | None = None
                            with output_anchor.open_directory_immovable(name) as child_anchor:
                                if RUN_ID_PATTERN.fullmatch(name):
                                    managed_record = self._managed_inventory_record(child, child_anchor)
                                elif PROBE_ID_PATTERN.fullmatch(name):
                                    probe_record = scan_probe_inventory_record(
                                        child,
                                        self.output_root,
                                        run_anchor=child_anchor,
                                    )
                                    if probe_record is None:
                                        raise HarvestStorageError("invalid managed probe evidence")
                                else:
                                    legacy_record = scan_legacy_run(
                                        child,
                                        directory_anchor=child_anchor,
                                    )
                                    if legacy_record is None:
                                        raise HarvestStorageError("unrecognized immediate child")
                            if managed_record is not None:
                                managed.append(managed_record)
                            elif probe_record is not None:
                                probes.append(probe_record)
                            elif legacy_record is not None:
                                legacy.append(legacy_record)
                        except (OSError, HarvestError, HarvestStorageError, UnsafeFilesystemOperation):
                            unsafe_entries += 1
            managed.sort(key=lambda value: value["run_id"])
            probes.sort(key=lambda value: value["probe_id"])
            legacy.sort(key=lambda value: value["legacy_id"])
            status_counts = dict(sorted(Counter(item["status"] for item in managed).items()))
            probe_status_counts = dict(sorted(Counter(item["status"] for item in probes).items()))
            known_usable = sum(
                int(item.get("usable_count", 0)) for item in managed if item["status"] == "COMPLETE"
            ) + sum(int(item.get("imported_records", 0)) for item in legacy)
            identity_payload = {
                "managed_runs": managed,
                "probe_runs": probes,
                "legacy_runs": legacy,
                "unsafe_entries": unsafe_entries,
            }
            return {
                "schema_version": HARVEST_INVENTORY_SCHEMA,
                "run_count": len(managed),
                "probe_run_count": len(probes),
                "legacy_run_count": len(legacy),
                "status_counts": status_counts,
                "probe_status_counts": probe_status_counts,
                "unsafe_entries": unsafe_entries,
                "known_usable_items": known_usable,
                "legacy_candidate_records": sum(item["candidate_records"] for item in legacy),
                "legacy_imported_records": sum(item["imported_records"] for item in legacy),
                "inventory_identity": _json_identity(identity_payload),
                "runs": managed,
                "probe_runs": probes,
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
            with open_anchored_directory(run_dir, self.output_root) as run_anchor:
                state = self._read_state(run_dir, run_anchor=run_anchor)
                if state["status"] in TERMINAL_STATUSES:
                    import_summary = self._dataset_import_summary(run_dir, run_anchor)
                    if import_summary["status"] in {"RUNNING", "CANCELLING"}:
                        import_state = self._read_json(
                            run_dir / "dataset_import_state.json",
                            parent_anchor=run_anchor,
                        )
                        attempt = int(import_state["attempt"])
                        cancellation_path = run_dir / f"dataset_import_cancellation_{attempt}.json"
                        if run_anchor.lexists(cancellation_path.name):
                            existing = self._read_json(cancellation_path, parent_anchor=run_anchor)
                            if (
                                set(existing) != _IMPORT_CANCELLATION_KEYS
                                or existing.get("schema_version") != HARVEST_IMPORT_CANCELLATION_SCHEMA
                                or existing.get("run_id") != run_id
                                or existing.get("attempt") != attempt
                                or existing.get("explicit_action") is not True
                                or existing.get("paths_exposed") is not False
                            ):
                                raise HarvestError(
                                    "invalid_harvest_import",
                                    "Dataset import cancellation evidence is invalid.",
                                )
                        else:
                            self._write_exclusive_json(
                                cancellation_path,
                                {
                                    "schema_version": HARVEST_IMPORT_CANCELLATION_SCHEMA,
                                    "run_id": run_id,
                                    "attempt": attempt,
                                    "explicit_action": True,
                                    "requested_at": _utc_now(),
                                    "paths_exposed": False,
                                },
                                parent_anchor=run_anchor,
                            )
                        local_import = self._import_cancellations.get(run_id)
                        if local_import is not None:
                            local_import.set()
                        if import_state["status"] == "RUNNING":
                            self._finish_dataset_import_state(
                                run_dir,
                                run_anchor,
                                import_state,
                                "CANCELLING",
                                None,
                            )
                    return self._job_from_anchor(run_id, run_dir, run_anchor)
                worker = self._workers.get(run_id)
                cancellation = self._cancellations.setdefault(run_id, threading.Event())
                cancellation_path = run_dir / "cancellation_request.json"
                if run_anchor.lexists(cancellation_path.name):
                    existing = self._read_json(cancellation_path, parent_anchor=run_anchor)
                    if (
                        existing.get("schema_version") != HARVEST_CANCELLATION_SCHEMA
                        or existing.get("run_id") != run_id
                    ):
                        raise HarvestError(
                            "invalid_harvest_cancellation",
                            "Harvest cancellation evidence is invalid.",
                        )
                else:
                    self._write_exclusive_json(
                        cancellation_path,
                        {
                            "schema_version": HARVEST_CANCELLATION_SCHEMA,
                            "run_id": run_id,
                            "requested_at": _utc_now(),
                            "explicit_action": True,
                            "paths_exposed": False,
                        },
                        parent_anchor=run_anchor,
                    )
                cancellation.set()
                if state["status"] == "CANCELLING" and worker is not None and worker.is_alive():
                    return self._job_from_anchor(run_id, run_dir, run_anchor)
                lease_live = self._lease_is_live(run_dir, state, run_anchor=run_anchor)
                if (worker is None or not worker.is_alive()) and not lease_live:
                    self._transition_locked(
                        run_dir,
                        "CANCELLED",
                        stage="cancelled",
                        message="Harvest job was cancelled and its durable evidence was retained.",
                        ended=True,
                        run_anchor=run_anchor,
                    )
                else:
                    self._transition_locked(
                        run_dir,
                        "CANCELLING",
                        stage="cancelling",
                        message="Cancellation is waiting for the certified backend safety boundary.",
                        run_anchor=run_anchor,
                    )
                return self._job_from_anchor(run_id, run_dir, run_anchor)

    def _new_lease(self, run_id: str, token: str, *, sequence: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "schema_version": HARVEST_LEASE_SCHEMA,
            "run_id": run_id,
            "owner_pid": os.getpid(),
            "owner_instance_id": self._instance_id,
            "lease_token": token,
            "heartbeat_sequence": sequence,
            "heartbeat_at": now.isoformat().replace("+00:00", "Z"),
            "expires_at": (now + timedelta(seconds=_LEASE_DURATION_SECONDS)).isoformat().replace("+00:00", "Z"),
            "released_at": None,
            "paths_exposed": False,
        }

    def _lease_is_live(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> bool:
        try:
            lease = self._read_json(run_dir / "worker_lease.json", parent_anchor=run_anchor)
            expires_at = _parse_utc_timestamp(str(lease.get("expires_at", "")))
        except (HarvestError, OSError, ValueError):
            return False
        return (
            lease.get("schema_version") == HARVEST_LEASE_SCHEMA
            and lease.get("run_id") == run_dir.name
            and lease.get("lease_token") == state.get("lease_token")
            and lease.get("owner_instance_id") == state.get("owner_instance_id")
            and lease.get("owner_pid") == state.get("owner_pid")
            and lease.get("released_at") is None
            and expires_at > datetime.now(timezone.utc)
        )

    def _start_lease_heartbeat(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> tuple[threading.Event, threading.Event, threading.Thread]:
        token = str(state.get("lease_token", ""))
        if not token or state.get("owner_instance_id") != self._instance_id:
            raise HarvestError("invalid_harvest_lease", "Harvest worker lease ownership is invalid.")
        stop = threading.Event()
        failed = threading.Event()

        def heartbeat() -> None:
            sequence = 0
            while not stop.wait(_LEASE_HEARTBEAT_SECONDS):
                sequence += 1
                try:
                    with self._mutation_guard():
                        current_state = self._read_state(run_dir, run_anchor=run_anchor)
                        if current_state.get("status") in TERMINAL_STATUSES:
                            return
                        if current_state.get("lease_token") != token:
                            raise HarvestError("invalid_harvest_lease", "Harvest worker lease token changed.")
                        current = self._read_json(
                            run_dir / "worker_lease.json",
                            parent_anchor=run_anchor,
                        )
                        if (
                            current.get("schema_version") != HARVEST_LEASE_SCHEMA
                            or current.get("lease_token") != token
                            or current.get("owner_instance_id") != self._instance_id
                        ):
                            raise HarvestError("invalid_harvest_lease", "Harvest worker lease evidence changed.")
                        self._write_atomic_json(
                            run_dir / "worker_lease.json",
                            self._new_lease(run_dir.name, token, sequence=sequence),
                            parent_anchor=run_anchor,
                        )
                except Exception:
                    failed.set()
                    return

        thread = threading.Thread(
            target=heartbeat,
            name=f"spritelab-{run_dir.name}-lease",
            daemon=True,
        )
        with self._lock:
            self._lease_stops[run_dir.name] = stop
            self._lease_failures[run_dir.name] = failed
            self._lease_threads[run_dir.name] = thread
        thread.start()
        return stop, failed, thread

    def _release_lease(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> None:
        token = str(state.get("lease_token", ""))
        try:
            with self._mutation_guard():
                current = self._read_json(run_dir / "worker_lease.json", parent_anchor=run_anchor)
                if current.get("lease_token") != token or current.get("owner_instance_id") != self._instance_id:
                    return
                now = _utc_now()
                current.update({"expires_at": now, "released_at": now})
                self._write_atomic_json(
                    run_dir / "worker_lease.json",
                    current,
                    parent_anchor=run_anchor,
                )
        except Exception:
            return

    def _cancellation_requested(
        self,
        run_dir: Path,
        local: threading.Event,
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> bool:
        """Poll durable cancellation evidence at a hard responsive cadence."""

        if local.is_set():
            return True
        now = time.monotonic()
        with self._lock:
            cached = self._last_cancellation_probe.get(run_dir.name)
            if cached is not None and now - cached[0] < _DURABLE_CANCELLATION_POLL_SECONDS:
                return cached[1]
        state = self._read_state(run_dir, run_anchor=run_anchor)
        cancellation_path = run_dir / "cancellation_request.json"
        durable = False
        cancellation_exists = (
            run_anchor.lexists(cancellation_path.name) if run_anchor is not None else os.path.lexists(cancellation_path)
        )
        if cancellation_exists:
            request = self._read_json(cancellation_path, parent_anchor=run_anchor)
            if (
                request.get("schema_version") != HARVEST_CANCELLATION_SCHEMA
                or request.get("run_id") != run_dir.name
                or request.get("explicit_action") is not True
            ):
                raise HarvestError("invalid_harvest_cancellation", "Harvest cancellation evidence is invalid.")
            durable = True
        result = durable or state["status"] in {"CANCELLING", "CANCELLED"}
        with self._lock:
            self._last_cancellation_probe[run_dir.name] = (now, result)
        return result

    def _raise_if_cancelled(
        self,
        run_dir: Path,
        local: threading.Event,
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> None:
        if self._cancellation_requested(run_dir, local, run_anchor=run_anchor):
            raise AcquisitionCancelled()

    def _raise_if_operation_aborted(
        self,
        run_dir: Path,
        local: threading.Event,
        deadline_monotonic: float,
        *,
        run_anchor: AnchoredDirectory,
    ) -> None:
        self._raise_if_cancelled(run_dir, local, run_anchor=run_anchor)
        if time.monotonic() >= deadline_monotonic:
            raise HarvestError(
                "harvest_duration_exceeded",
                "Harvest exceeded the whole-operation duration limit.",
            )

    def _construct_backend_with_control(
        self,
        run_dir: Path,
        local: threading.Event,
        deadline_monotonic: float,
        *,
        run_anchor: AnchoredDirectory,
    ) -> Any:
        """Keep backend construction inside the run deadline/cancel boundary."""

        factory = self._backend_factory
        if factory is None:
            raise HarvestError("harvest_backend_unavailable", "Certified Harvest backend is unavailable.")
        ready = threading.Event()
        outcome: list[tuple[bool, Any]] = []

        def construct() -> None:
            try:
                outcome.append((True, factory()))
            except BaseException as exc:  # captured and sanitized in the owning worker
                outcome.append((False, exc))
            finally:
                ready.set()

        threading.Thread(
            target=construct,
            name=f"spritelab-{run_dir.name}-backend-construction",
            daemon=True,
        ).start()
        while not ready.is_set():
            self._raise_if_operation_aborted(
                run_dir,
                local,
                deadline_monotonic,
                run_anchor=run_anchor,
            )
            ready.wait(timeout=min(_DURABLE_CANCELLATION_POLL_SECONDS, max(0.0, deadline_monotonic - time.monotonic())))
        self._raise_if_operation_aborted(
            run_dir,
            local,
            deadline_monotonic,
            run_anchor=run_anchor,
        )
        if len(outcome) != 1 or outcome[0][0] is not True:
            error = outcome[0][1] if outcome else None
            raise HarvestError(
                "harvest_backend_construction_failed",
                "Certified Harvest backend construction failed closed.",
            ) from error
        backend = outcome[0][1]
        if not callable(getattr(backend, "acquire", None)):
            raise HarvestError(
                "harvest_backend_construction_failed",
                "Certified Harvest backend construction returned no acquisition contract.",
            )
        return backend

    @staticmethod
    def _ensure_lease_healthy(failed: threading.Event | None) -> None:
        if failed is not None and failed.is_set():
            raise HarvestError("harvest_worker_lease_failed", "Harvest worker lease renewal failed closed.")

    def _lease_checked_cancellation(
        self,
        run_dir: Path,
        local: threading.Event,
        lease_failure: threading.Event | None,
        run_anchor: AnchoredDirectory | None = None,
    ) -> bool:
        self._ensure_lease_healthy(lease_failure)
        return self._cancellation_requested(run_dir, local, run_anchor=run_anchor)

    def _lease_checked_progress(
        self,
        run_dir: Path,
        local: threading.Event,
        lease_failure: threading.Event | None,
        stage: str,
        current: int,
        total: int | None,
        run_anchor: AnchoredDirectory | None = None,
    ) -> None:
        self._ensure_lease_healthy(lease_failure)
        self._progress(run_dir, local, stage, current, total, run_anchor=run_anchor)

    def job(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run_dir = self._run_directory(run_id)
            with open_anchored_directory(run_dir, self.output_root) as run_anchor:
                return self._job_from_anchor(run_id, run_dir, run_anchor)

    def _job_from_anchor(
        self,
        run_id: str,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
    ) -> dict[str, Any]:
        state = dict(self._read_state(run_dir, run_anchor=run_anchor))
        request = self._read_request(run_dir, run_anchor=run_anchor)
        status = self._effective_status(state, run_id, run_dir=run_dir, run_anchor=run_anchor)
        complete_commit_valid = status != "COMPLETE" or self._complete_commit_valid(
            run_dir,
            state,
            run_anchor=run_anchor,
        )
        if status == "COMPLETE" and not complete_commit_valid:
            status = "INTERRUPTED"
        if status == "INTERRUPTED":
            state["stage"] = "interrupted"
            state["message"] = "The owning process stopped before this Harvest job durably committed."
        receipt = self._read_json(run_dir / "authorization_receipt.json", parent_anchor=run_anchor)
        if receipt.get("schema_version") != HARVEST_AUTHORIZATION_SCHEMA or receipt.get("run_id") != run_id:
            raise HarvestError("invalid_harvest_authorization", "Harvest authorization evidence is invalid.")
        import_summary = self._dataset_import_summary(run_dir, run_anchor)
        return {
            "schema_version": "spritelab.harvest.job.v3",
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
            "handoff_ready": bool(state.get("handoff_ready", False)) and complete_commit_valid,
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
            "events": self._read_events(run_dir, run_anchor=run_anchor),
            "dataset_import_available": self.dataset_import_available,
            "dataset_import": import_summary,
            "paths_exposed": False,
        }

    def handoff(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run_dir = self._run_directory(run_id)
            with open_anchored_directory(run_dir, self.output_root) as run_anchor:
                return self._handoff_from_anchor(run_id, run_dir, run_anchor)

    def _handoff_from_anchor(
        self,
        run_id: str,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
    ) -> dict[str, Any]:
        job = self._job_from_anchor(run_id, run_dir, run_anchor)
        if job["status"] != "COMPLETE" or not job["handoff_ready"]:
            raise HarvestError(
                "harvest_handoff_not_ready",
                "Dataset handoff is available only after certified Harvest completion.",
            )
        source = self._validate_live_configuration(str(job["source_id"]))
        request = self._read_request(run_dir, run_anchor=run_anchor)
        backend_evidence = self._backend_evidence_record()
        if (
            request.get("source_catalog_identity") != source.catalog_identity
            or request.get("limits_identity") != self.limits.identity
            or self._backend_capabilities is None
            or request.get("backend_capability_identity") != self._backend_capabilities.identity
            or request.get("backend_capability_evidence_identity") != backend_evidence["evidence_identity"]
            or request.get("backend_capability_certificate_identity") != backend_evidence["certificate_identity"]
            or request.get("backend_capability_audit_report_sha256") != backend_evidence["audit_report_sha256"]
            or request.get("backend_capability_audit_report_identity") != backend_evidence["audit_report_identity"]
            or request.get("backend_capability_issued_at") != backend_evidence["issued_at"]
            or request.get("backend_capability_expires_at") != backend_evidence["expires_at"]
        ):
            raise HarvestError(
                "catalog_evidence_changed",
                "Current source, backend, or limit identities no longer match the authorized Harvest request.",
            )
        manifest = self._rehash_manifest(run_dir, run_anchor=run_anchor)
        payload = self._read_json(run_dir / "handoff.json", parent_anchor=run_anchor)
        if (
            payload.get("schema_version") != HARVEST_HANDOFF_SCHEMA
            or payload.get("run_id") != run_id
            or payload.get("artifact_manifest_identity") != _json_identity(manifest)
            or payload.get("artifact_set_identity") != manifest["artifact_set_identity"]
            or payload.get("backend_capability_evidence_identity") != backend_evidence["evidence_identity"]
        ):
            raise HarvestError("invalid_harvest_handoff", "Harvest handoff identity is invalid.")
        return {
            **payload,
            "dataset_import_available": self.dataset_import_available,
            "dataset_import": self._dataset_import_summary(run_dir, run_anchor),
        }

    def evidence(self, run_id: str) -> dict[str, Any]:
        """Return privacy-safe durable authorization, provenance, and limit evidence."""

        with self._lock:
            run_dir = self._run_directory(run_id)
            with open_anchored_directory(run_dir, self.output_root) as run_anchor:
                job = self._job_from_anchor(run_id, run_dir, run_anchor)
                authorization = self._read_json(
                    run_dir / "authorization_receipt.json",
                    parent_anchor=run_anchor,
                )
                acquisition = (
                    self._read_json(run_dir / "acquisition_receipt.json", parent_anchor=run_anchor)
                    if run_anchor.lexists("acquisition_receipt.json")
                    else None
                )
                manifest = None
                if run_anchor.lexists("artifact_manifest.json"):
                    manifest = (
                        self._rehash_manifest(run_dir, run_anchor=run_anchor)
                        if job["status"] == "COMPLETE"
                        else self._read_json(run_dir / "artifact_manifest.json", parent_anchor=run_anchor)
                    )
                return {
                    "schema_version": "spritelab.harvest.durable-evidence.v2",
                    "run_id": run_id,
                    "status": job["status"],
                    "job": job,
                    "authorization_receipt": authorization,
                    "acquisition_receipt": acquisition,
                    "artifact_manifest": manifest,
                    "dataset_import": self._dataset_import_summary(run_dir, run_anchor),
                    "paths_exposed": False,
                }

    def import_to_dataset(
        self,
        run_id: str,
        *,
        explicit_action: bool,
        idempotency_key: str,
    ) -> dict[str, Any]:
        operation_deadline = time.monotonic() + self.limits.max_duration_seconds
        if explicit_action is not True:
            raise HarvestError(
                "explicit_action_required", "Dataset import requires an explicit action.", status_code=422
            )
        _validate_idempotency_key(idempotency_key)
        callback = self._dataset_import_callback_for_action()
        run_dir = self._run_directory(run_id)
        with open_anchored_directory(run_dir, self.output_root) as run_anchor:
            self._raise_if_import_aborted(run_dir, operation_deadline, run_anchor=run_anchor)
            self._handoff_from_anchor(run_id, run_dir, run_anchor)
            handoff_document = self._read_json(run_dir / "handoff.json", parent_anchor=run_anchor)
            manifest = self._rehash_manifest(run_dir, run_anchor=run_anchor)
            stable_request = {
                "run_id": run_id,
                "callback_id": callback.callback_id,
                "callback_code_identity_sha256": callback.code_identity_sha256,
                "callback_runtime_identity_sha256": callback.runtime_identity_sha256,
                "handoff_identity": _json_identity(handoff_document),
                "artifact_manifest_identity": _json_identity(manifest),
            }
            request_identity = _json_identity(stable_request)
            request_path = run_dir / "dataset_import_request.json"
            receipt_path = run_dir / "dataset_import_receipt.json"
            state_path = run_dir / "dataset_import_state.json"
            with self._mutation_guard():
                # The mutation-lock wait itself can outlast the whole-operation
                # deadline; recheck before any durable publication.
                self._raise_if_import_aborted(run_dir, operation_deadline, run_anchor=run_anchor)
                if run_anchor.lexists(request_path.name):
                    request_payload = self._read_json(request_path, parent_anchor=run_anchor)
                    self._validate_dataset_import_request(request_payload, run_id)
                    if request_payload.get("request_identity") != request_identity or any(
                        request_payload.get(key) != value for key, value in stable_request.items()
                    ):
                        raise HarvestError(
                            "idempotency_conflict",
                            "The durable Dataset import request is bound to different identities.",
                        )
                    effective_key = str(request_payload.get("idempotency_key", ""))
                    _validate_idempotency_key(effective_key)
                else:
                    effective_key = idempotency_key
                    request_payload = {
                        "schema_version": HARVEST_IMPORT_REQUEST_SCHEMA,
                        **stable_request,
                        "request_identity": request_identity,
                        "idempotency_key": effective_key,
                        "created_at": _utc_now(),
                        "paths_exposed": False,
                    }
                    self._write_exclusive_json(request_path, request_payload, parent_anchor=run_anchor)
                if run_anchor.lexists(receipt_path.name):
                    receipt_payload = self._validated_dataset_import_receipt(
                        self._read_json(receipt_path, parent_anchor=run_anchor),
                        run_id,
                        request_identity,
                    )
                    if not self._dataset_import_commit_valid(run_dir, run_anchor, request_identity):
                        self._recover_dataset_import_commit(run_dir, run_anchor, run_id, request_identity)
                    return receipt_payload
                previous_state = (
                    self._read_json(state_path, parent_anchor=run_anchor)
                    if run_anchor.lexists(state_path.name)
                    else None
                )
                if previous_state is not None:
                    self._validate_dataset_import_state(previous_state, run_id, request_identity)
                    previous_status = str(previous_state["status"])
                    previous_deadline = _parse_utc_timestamp(str(previous_state["deadline_at"]))
                    if previous_status in {"RUNNING", "CANCELLING"} and previous_deadline > datetime.now(timezone.utc):
                        raise HarvestError(
                            "dataset_import_in_progress",
                            "This Harvest run already has an active Dataset import.",
                            status_code=409,
                        )
                    attempt = int(previous_state["attempt"]) + 1
                else:
                    attempt = 1
                now = datetime.now(timezone.utc)
                deadline_at = now + timedelta(seconds=max(0.0, operation_deadline - time.monotonic()))
                import_state = {
                    "schema_version": HARVEST_IMPORT_STATE_SCHEMA,
                    "run_id": run_id,
                    "request_identity": request_identity,
                    "attempt": attempt,
                    "status": "RUNNING",
                    "started_at": now.isoformat().replace("+00:00", "Z"),
                    "updated_at": now.isoformat().replace("+00:00", "Z"),
                    "deadline_at": deadline_at.isoformat().replace("+00:00", "Z"),
                    "error_code": None,
                    "paths_exposed": False,
                }
                self._raise_if_import_aborted(run_dir, operation_deadline, run_anchor=run_anchor)
                self._write_atomic_json(state_path, import_state, parent_anchor=run_anchor)
                cancellation = threading.Event()
                self._import_cancellations[run_id] = cancellation
            try:
                result = callback.import_harvest(
                    DatasetImportRequest(run_id, run_dir / "artifacts", handoff_document, manifest),
                    idempotency_key=effective_key,
                    deadline_monotonic=operation_deadline,
                    cancel_requested=lambda: self._dataset_import_cancel_requested(
                        run_dir,
                        attempt,
                        cancellation,
                        run_anchor,
                    ),
                )
                if not isinstance(result, DatasetImportResult):
                    raise HarvestError(
                        "dataset_import_failed",
                        "The Dataset import callback returned no valid result contract.",
                    )
                self._raise_if_import_aborted(
                    run_dir,
                    operation_deadline,
                    run_anchor=run_anchor,
                    attempt=attempt,
                    local=cancellation,
                )
            except (AcquisitionCancelled, DatasetImportCancelled) as exc:
                self._finalize_dataset_import_state(
                    run_dir,
                    run_anchor,
                    request_identity=request_identity,
                    attempt=attempt,
                    status="CANCELLED",
                    error_code="dataset_import_cancelled",
                )
                raise HarvestError(
                    "dataset_import_cancelled",
                    "Dataset import was cancelled before Harvest receipt publication.",
                ) from exc
            except DatasetImportDeadlineExceeded as exc:
                self._finalize_dataset_import_state(
                    run_dir,
                    run_anchor,
                    request_identity=request_identity,
                    attempt=attempt,
                    status="FAILED",
                    error_code="dataset_import_duration_exceeded",
                )
                raise HarvestError(
                    "dataset_import_duration_exceeded",
                    "Dataset import exceeded the whole-operation duration limit.",
                ) from exc
            except HarvestError as exc:
                self._finalize_dataset_import_state(
                    run_dir,
                    run_anchor,
                    request_identity=request_identity,
                    attempt=attempt,
                    status="FAILED",
                    error_code=exc.code,
                )
                raise
            except Exception as exc:
                self._finalize_dataset_import_state(
                    run_dir,
                    run_anchor,
                    request_identity=request_identity,
                    attempt=attempt,
                    status="FAILED",
                    error_code="dataset_import_failed",
                )
                raise HarvestError(
                    "dataset_import_failed",
                    "The Dataset import callback failed without exposing private details.",
                ) from exc
            finally:
                with self._lock:
                    # Pop only this attempt's event; a newer attempt may have
                    # already installed its own local cancellation channel.
                    if self._import_cancellations.get(run_id) is cancellation:
                        self._import_cancellations.pop(run_id, None)
            receipt = {
                "schema_version": HARVEST_IMPORT_RECEIPT_SCHEMA,
                "run_id": run_id,
                "request_identity": request_identity,
                "idempotency_key": effective_key,
                "callback_id": callback.callback_id,
                "callback_code_identity_sha256": callback.code_identity_sha256,
                "callback_runtime_identity_sha256": callback.runtime_identity_sha256,
                "dataset_reference": result.dataset_reference,
                "accepted_count": result.accepted_count,
                "quarantined_count": result.quarantined_count,
                "artifact_manifest_identity": _json_identity(manifest),
                "paths_exposed": False,
                "created_at": _utc_now(),
            }
            try:
                with self._mutation_guard():
                    # The mutation-lock wait itself can outlast cancellation or
                    # the whole-operation deadline; recheck before publication.
                    self._raise_if_import_aborted(
                        run_dir,
                        operation_deadline,
                        run_anchor=run_anchor,
                        attempt=attempt,
                        local=cancellation,
                    )
                    current_state = self._read_json(state_path, parent_anchor=run_anchor)
                    self._validate_dataset_import_state(current_state, run_id, request_identity)
                    if current_state.get("attempt") != attempt:
                        raise HarvestError(
                            "dataset_import_superseded",
                            "A newer Dataset import attempt superseded this one before publication.",
                        )
                    if str(current_state.get("status")) == "CANCELLING":
                        raise AcquisitionCancelled()
                    if str(current_state.get("status")) != "RUNNING":
                        raise HarvestError(
                            "dataset_import_superseded",
                            "This Dataset import attempt was finalized elsewhere before publication.",
                        )
                    if run_anchor.lexists(receipt_path.name):
                        if self._dataset_import_commit_valid(run_dir, run_anchor, request_identity):
                            return self._validated_dataset_import_receipt(
                                self._read_json(receipt_path, parent_anchor=run_anchor),
                                run_id,
                                request_identity,
                            )
                        raise HarvestError(
                            "invalid_harvest_import",
                            "Dataset import receipt evidence is invalid.",
                        )
                    self._write_exclusive_json(receipt_path, receipt, parent_anchor=run_anchor)
                    # COMPLETE becomes visible only through this terminal commit
                    # binding the receipt and final state identities together.
                    self._commit_dataset_import_terminal(run_dir, run_anchor, current_state, receipt)
            except AcquisitionCancelled as exc:
                self._finalize_dataset_import_state(
                    run_dir,
                    run_anchor,
                    request_identity=request_identity,
                    attempt=attempt,
                    status="CANCELLED",
                    error_code="dataset_import_cancelled",
                )
                raise HarvestError(
                    "dataset_import_cancelled",
                    "Dataset import was cancelled before Harvest receipt publication.",
                ) from exc
            except HarvestError as exc:
                self._finalize_dataset_import_state(
                    run_dir,
                    run_anchor,
                    request_identity=request_identity,
                    attempt=attempt,
                    status="FAILED",
                    error_code=exc.code,
                )
                raise
            return receipt

    def _dataset_import_summary(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
    ) -> dict[str, Any]:
        if not run_anchor.lexists("dataset_import_request.json"):
            return {"status": "NOT_STARTED", "completed": False, "paths_exposed": False}
        request = self._read_json(run_dir / "dataset_import_request.json", parent_anchor=run_anchor)
        self._validate_dataset_import_request(request, run_dir.name)
        request_identity = str(request.get("request_identity", ""))
        state: dict[str, Any] | None = None
        if run_anchor.lexists("dataset_import_state.json"):
            state = self._read_json(run_dir / "dataset_import_state.json", parent_anchor=run_anchor)
            self._validate_dataset_import_state(state, run_dir.name, request_identity)
        if run_anchor.lexists("dataset_import_receipt.json"):
            receipt = self._validated_dataset_import_receipt(
                self._read_json(run_dir / "dataset_import_receipt.json", parent_anchor=run_anchor),
                run_dir.name,
                request_identity,
            )
            if self._dataset_import_commit_valid(run_dir, run_anchor, request_identity):
                return {
                    "status": "COMPLETE",
                    "completed": True,
                    "dataset_reference": receipt["dataset_reference"],
                    "accepted_count": receipt["accepted_count"],
                    "quarantined_count": receipt["quarantined_count"],
                    "request_identity": request_identity,
                    "paths_exposed": False,
                }
            # A receipt alone never implies completion: without the bound
            # terminal commit the import is interrupted, recoverable evidence.
            return {
                "status": "INTERRUPTED",
                "completed": False,
                "attempt": state["attempt"] if state is not None else None,
                "error_code": "dataset_import_commit_incomplete",
                "request_identity": request_identity,
                "paths_exposed": False,
            }
        if state is None:
            return {
                "status": "REQUESTED",
                "completed": False,
                "request_identity": request_identity,
                "paths_exposed": False,
            }
        status = str(state["status"])
        if status == "COMPLETE":
            # A COMPLETE state without its receipt and terminal commit is not
            # terminal success evidence.
            status = "INTERRUPTED"
        elif status in {"RUNNING", "CANCELLING"} and _parse_utc_timestamp(str(state["deadline_at"])) <= datetime.now(
            timezone.utc
        ):
            status = "INTERRUPTED"
        return {
            "status": status,
            "completed": False,
            "attempt": state["attempt"],
            "error_code": state.get("error_code"),
            "request_identity": request_identity,
            "paths_exposed": False,
        }

    @staticmethod
    def _validate_dataset_import_request(
        request: Mapping[str, Any],
        run_id: str,
    ) -> None:
        stable = {
            "run_id": request.get("run_id"),
            "callback_id": request.get("callback_id"),
            "callback_code_identity_sha256": request.get("callback_code_identity_sha256"),
            "callback_runtime_identity_sha256": request.get("callback_runtime_identity_sha256"),
            "handoff_identity": request.get("handoff_identity"),
            "artifact_manifest_identity": request.get("artifact_manifest_identity"),
        }
        if (
            set(request) != _IMPORT_REQUEST_KEYS
            or request.get("schema_version") != HARVEST_IMPORT_REQUEST_SCHEMA
            or request.get("run_id") != run_id
            or SOURCE_ID_PATTERN.fullmatch(str(request.get("callback_id", ""))) is None
            or any(
                SHA256_PATTERN.fullmatch(str(request.get(key, ""))) is None
                for key in (
                    "callback_code_identity_sha256",
                    "callback_runtime_identity_sha256",
                    "handoff_identity",
                    "artifact_manifest_identity",
                    "request_identity",
                )
            )
            or request.get("request_identity") != _json_identity(stable)
            or IDEMPOTENCY_KEY_PATTERN.fullmatch(str(request.get("idempotency_key", ""))) is None
            or request.get("paths_exposed") is not False
        ):
            raise HarvestError("invalid_harvest_import", "Dataset import request evidence is invalid.")
        try:
            _parse_utc_timestamp(str(request.get("created_at", "")))
        except ValueError as exc:
            raise HarvestError("invalid_harvest_import", "Dataset import request evidence is invalid.") from exc

    @staticmethod
    def _validated_dataset_import_receipt(
        receipt: Mapping[str, Any],
        run_id: str,
        request_identity: str,
    ) -> dict[str, Any]:
        if (
            set(receipt) != _IMPORT_RECEIPT_KEYS
            or receipt.get("schema_version") != HARVEST_IMPORT_RECEIPT_SCHEMA
            or receipt.get("run_id") != run_id
            or receipt.get("request_identity") != request_identity
            or SHA256_PATTERN.fullmatch(str(receipt.get("request_identity", ""))) is None
            or IDEMPOTENCY_KEY_PATTERN.fullmatch(str(receipt.get("idempotency_key", ""))) is None
            or SOURCE_ID_PATTERN.fullmatch(str(receipt.get("callback_id", ""))) is None
            or SHA256_PATTERN.fullmatch(str(receipt.get("callback_code_identity_sha256", ""))) is None
            or SHA256_PATTERN.fullmatch(str(receipt.get("callback_runtime_identity_sha256", ""))) is None
            or SHA256_PATTERN.fullmatch(str(receipt.get("artifact_manifest_identity", ""))) is None
            or type(receipt.get("accepted_count")) is not int
            or type(receipt.get("quarantined_count")) is not int
            or receipt.get("paths_exposed") is not False
        ):
            raise HarvestError("invalid_harvest_import", "Dataset import receipt evidence is invalid.")
        try:
            DatasetImportResult(
                str(receipt.get("dataset_reference", "")),
                receipt.get("accepted_count"),
                receipt.get("quarantined_count"),
            )
            _parse_utc_timestamp(str(receipt.get("created_at", "")))
        except (TypeError, ValueError) as exc:
            raise HarvestError("invalid_harvest_import", "Dataset import receipt evidence is invalid.") from exc
        return dict(receipt)

    @staticmethod
    def _validate_dataset_import_state(
        state: Mapping[str, Any],
        run_id: str,
        request_identity: str,
    ) -> None:
        if (
            set(state) != _IMPORT_STATE_KEYS
            or state.get("schema_version") != HARVEST_IMPORT_STATE_SCHEMA
            or state.get("run_id") != run_id
            or state.get("request_identity") != request_identity
            or type(state.get("attempt")) is not int
            or state["attempt"] <= 0
            or state.get("status") not in {"RUNNING", "CANCELLING", "FAILED", "CANCELLED", "COMPLETE"}
            or (
                state.get("error_code") is not None
                and (
                    not isinstance(state.get("error_code"), str)
                    or STAGE_PATTERN.fullmatch(str(state.get("error_code"))) is None
                )
            )
            or state.get("paths_exposed") is not False
        ):
            raise HarvestError("invalid_harvest_import", "Dataset import state evidence is invalid.")
        try:
            started = _parse_utc_timestamp(str(state.get("started_at", "")))
            updated = _parse_utc_timestamp(str(state.get("updated_at", "")))
            deadline = _parse_utc_timestamp(str(state.get("deadline_at", "")))
        except ValueError as exc:
            raise HarvestError("invalid_harvest_import", "Dataset import state evidence is invalid.") from exc
        if updated < started or deadline < started:
            raise HarvestError("invalid_harvest_import", "Dataset import state timestamps are invalid.")

    def _finish_dataset_import_state(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        state: Mapping[str, Any],
        status: str,
        error_code: str | None,
    ) -> None:
        updated = {
            **dict(state),
            "status": status,
            "updated_at": _utc_now(),
            "error_code": error_code,
        }
        self._write_atomic_json(
            run_dir / "dataset_import_state.json",
            updated,
            parent_anchor=run_anchor,
        )

    def _finalize_dataset_import_state(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        *,
        request_identity: str,
        attempt: int,
        status: str,
        error_code: str | None,
    ) -> bool:
        """Terminal compare-and-set: a stale attempt never modifies a newer one.

        The write happens only when the durable state still carries this
        attempt's request identity, attempt number, and a finalizable status.
        """

        try:
            with self._mutation_guard():
                current = self._read_json(
                    run_dir / "dataset_import_state.json",
                    parent_anchor=run_anchor,
                )
                self._validate_dataset_import_state(current, run_dir.name, request_identity)
                if (
                    current.get("request_identity") != request_identity
                    or current.get("attempt") != attempt
                    or str(current.get("status")) not in {"RUNNING", "CANCELLING"}
                ):
                    return False
                self._finish_dataset_import_state(
                    run_dir,
                    run_anchor,
                    current,
                    status,
                    error_code,
                )
                return True
        except HarvestError:
            # Finalization is best-effort evidence for this attempt only; the
            # triggering error keeps propagating to the caller unchanged.
            return False

    def _commit_dataset_import_terminal(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        state: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> None:
        """Publish COMPLETE state and the terminal commit binding it to the receipt."""

        completed = {
            **dict(state),
            "status": "COMPLETE",
            "updated_at": _utc_now(),
            "error_code": None,
        }
        self._write_atomic_json(
            run_dir / "dataset_import_state.json",
            completed,
            parent_anchor=run_anchor,
        )
        commit = {
            "schema_version": HARVEST_IMPORT_TERMINAL_COMMIT_SCHEMA,
            "run_id": run_dir.name,
            "request_identity": completed["request_identity"],
            "attempt": completed["attempt"],
            "status": "COMPLETE",
            "receipt_identity": _json_identity(dict(receipt)),
            "state_identity": _json_identity(completed),
            "created_at": completed["updated_at"],
            "paths_exposed": False,
        }
        commit["terminal_commit_identity"] = _json_identity(commit)
        self._write_exclusive_json(
            run_dir / "dataset_import_terminal_commit.json",
            commit,
            parent_anchor=run_anchor,
        )

    def _dataset_import_commit_valid(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        request_identity: str,
    ) -> bool:
        try:
            commit = self._read_json(
                run_dir / "dataset_import_terminal_commit.json",
                parent_anchor=run_anchor,
            )
            receipt = self._read_json(run_dir / "dataset_import_receipt.json", parent_anchor=run_anchor)
            state = self._read_json(run_dir / "dataset_import_state.json", parent_anchor=run_anchor)
            self._validate_dataset_import_state(state, run_dir.name, request_identity)
        except (HarvestError, OSError):
            return False
        recorded_identity = commit.get("terminal_commit_identity")
        identity_payload = {key: value for key, value in commit.items() if key != "terminal_commit_identity"}
        return (
            set(commit) == _IMPORT_TERMINAL_COMMIT_KEYS
            and commit.get("schema_version") == HARVEST_IMPORT_TERMINAL_COMMIT_SCHEMA
            and commit.get("run_id") == run_dir.name
            and commit.get("request_identity") == request_identity
            and commit.get("status") == "COMPLETE"
            and commit.get("paths_exposed") is False
            and type(commit.get("attempt")) is int
            and commit.get("attempt") == state.get("attempt")
            and state.get("status") == "COMPLETE"
            and SHA256_PATTERN.fullmatch(str(recorded_identity or "")) is not None
            and recorded_identity == _json_identity(identity_payload)
            and commit.get("receipt_identity") == _json_identity(receipt)
            and commit.get("state_identity") == _json_identity(state)
        )

    def _recover_dataset_import_commit(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        run_id: str,
        request_identity: str,
    ) -> None:
        """Idempotently rebind a durable receipt to COMPLETE state and commit.

        Runs under the mutation guard after abort rechecks. A receipt is only
        ever published after a successful callback result for this request
        identity, so recovery re-finalizes the interrupted terminal sequence
        without re-invoking the callback.
        """

        if run_anchor.lexists("dataset_import_terminal_commit.json"):
            raise HarvestError(
                "invalid_harvest_import",
                "Dataset import terminal commit evidence is invalid.",
            )
        state = self._read_json(run_dir / "dataset_import_state.json", parent_anchor=run_anchor)
        self._validate_dataset_import_state(state, run_id, request_identity)
        receipt = self._read_json(run_dir / "dataset_import_receipt.json", parent_anchor=run_anchor)
        self._commit_dataset_import_terminal(run_dir, run_anchor, state, receipt)

    def _dataset_import_cancel_requested(
        self,
        run_dir: Path,
        attempt: int,
        local: threading.Event,
        run_anchor: AnchoredDirectory,
    ) -> bool:
        if local.is_set():
            return True
        cancellation_name = f"dataset_import_cancellation_{attempt}.json"
        if not run_anchor.lexists(cancellation_name):
            return False
        cancellation = self._read_json(
            run_dir / cancellation_name,
            parent_anchor=run_anchor,
        )
        if (
            set(cancellation) != _IMPORT_CANCELLATION_KEYS
            or cancellation.get("schema_version") != HARVEST_IMPORT_CANCELLATION_SCHEMA
            or cancellation.get("run_id") != run_dir.name
            or cancellation.get("attempt") != attempt
            or cancellation.get("explicit_action") is not True
            or cancellation.get("paths_exposed") is not False
        ):
            raise HarvestError("invalid_harvest_import", "Dataset import cancellation evidence is invalid.")
        try:
            _parse_utc_timestamp(str(cancellation.get("requested_at", "")))
        except ValueError as exc:
            raise HarvestError("invalid_harvest_import", "Dataset import cancellation evidence is invalid.") from exc
        local.set()
        return True

    def _raise_if_import_aborted(
        self,
        run_dir: Path,
        deadline_monotonic: float,
        *,
        run_anchor: AnchoredDirectory,
        attempt: int | None = None,
        local: threading.Event | None = None,
    ) -> None:
        if (
            attempt is not None
            and local is not None
            and self._dataset_import_cancel_requested(
                run_dir,
                attempt,
                local,
                run_anchor,
            )
        ):
            raise AcquisitionCancelled()
        if time.monotonic() >= deadline_monotonic:
            raise HarvestError(
                "dataset_import_duration_exceeded",
                "Dataset import exceeded the whole-operation duration limit.",
            )

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
                "backend_capability_evidence_identity": self._backend_evidence_record()["evidence_identity"],
                "limits_identity": self.limits.identity,
                "reuse_evidence_identity": reuse.identity,
                "retry_of": retry_of,
            }
        )
        # Required passive inventory occurs before even backend construction.
        self.inventory()
        self._create_output_root()
        with self._mutation_guard():
            locked_source = self._validate_live_configuration(source_id)
            if locked_source.catalog_identity != source.catalog_identity:
                raise HarvestError(
                    "harvest_live_configuration_changed",
                    "Harvest source evidence changed during authorization.",
                )
            backend_evidence = self._backend_evidence_record()
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
            conflict = self._active_managed_run()
            if conflict is not None:
                raise HarvestError(
                    "harvest_single_flight_conflict",
                    f"Managed Harvest run {conflict!r} is already active; wait for it to finish.",
                )
            anchor_stack = ExitStack()
            worker_started = False
            try:
                output_anchor = anchor_stack.enter_context(open_anchored_directory(self.output_root, self.project_root))
                run_id, run_dir, run_anchor = self._create_run_directory(output_anchor, anchor_stack)
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
                    "backend_capability_evidence_identity": backend_evidence["evidence_identity"],
                    "backend_capability_certificate_identity": backend_evidence["certificate_identity"],
                    "backend_capability_audit_report_sha256": backend_evidence["audit_report_sha256"],
                    "backend_capability_audit_report_identity": backend_evidence["audit_report_identity"],
                    "backend_capability_issued_at": backend_evidence["issued_at"],
                    "backend_capability_expires_at": backend_evidence["expires_at"],
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
                    "backend_capability_evidence": backend_evidence,
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
                self._write_exclusive_json(
                    run_dir / "authorization_receipt.json",
                    receipt,
                    parent_anchor=run_anchor,
                )
                self._write_exclusive_json(run_dir / "request.json", request, parent_anchor=run_anchor)
                lease_token = uuid.uuid4().hex
                lease = self._new_lease(run_id, lease_token, sequence=0)
                self._write_exclusive_json(run_dir / "worker_lease.json", lease, parent_anchor=run_anchor)
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
                    "lease_token": lease_token,
                }
                self._write_state(run_dir, state, run_anchor=run_anchor)
                self._append_event(run_dir, state, run_anchor=run_anchor)
                cancellation = threading.Event()
                worker = threading.Thread(
                    target=self._run_worker,
                    args=(run_id, source, cancellation, run_anchor, anchor_stack),
                    name=f"spritelab-{run_id}",
                    daemon=True,
                )
                self._cancellations[run_id] = cancellation
                self._workers[run_id] = worker
                queued_job = self._job_from_anchor(run_id, run_dir, run_anchor)
                worker.start()
                worker_started = True
                return queued_job, True
            finally:
                if not worker_started:
                    anchor_stack.close()

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
        source = self._validate_live_configuration(source_id)
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

    def _run_worker(
        self,
        run_id: str,
        source: HarvestSource,
        cancellation: threading.Event,
        transaction_anchor: AnchoredDirectory,
        anchor_stack: ExitStack,
    ) -> None:
        operation_started = time.monotonic()
        operation_deadline = operation_started + self.limits.max_duration_seconds
        lease_stop: threading.Event | None = None
        lease_failure: threading.Event | None = None
        lease_thread: threading.Thread | None = None
        run_dir: Path | None = transaction_anchor.directory
        try:
            if run_dir.name != run_id or run_dir.parent != self.output_root:
                raise HarvestError("invalid_harvest_run", "Harvest worker anchor does not match its run identity.")
            transaction_anchor.verify()
            artifacts = require_confined_path(run_dir / "artifacts", run_dir)
            if self._backend_factory is None:
                raise HarvestError("harvest_backend_unavailable", "Certified Harvest backend is unavailable.")
            initial_state = self._read_state(run_dir, run_anchor=transaction_anchor)
            lease_stop, lease_failure, lease_thread = self._start_lease_heartbeat(
                run_dir,
                initial_state,
                run_anchor=transaction_anchor,
            )
            self._raise_if_operation_aborted(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            self._transition(
                run_dir,
                "RUNNING",
                stage="verifying_backend",
                message="Harvest is revalidating the certified backend inside the whole-run deadline.",
                started=True,
                run_anchor=transaction_anchor,
            )
            backend = self._construct_backend_with_control(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            self._raise_if_operation_aborted(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            self._transition(
                run_dir,
                "RUNNING",
                stage="acquiring",
                message="Certified acquisition is enforcing network and resource limits.",
                run_anchor=transaction_anchor,
            )
            transaction_anchor.mkdir(artifacts.name)
            acquisition_started = time.monotonic()
            remaining = operation_deadline - acquisition_started
            if remaining <= 0:
                raise HarvestError("harvest_duration_exceeded", "Harvest exceeded the whole-run duration limit.")
            operation_limits = replace(self.limits, max_duration_seconds=remaining)
            backend_arguments: dict[str, Any] = {
                "cancel_requested": lambda: self._lease_checked_cancellation(
                    run_dir,
                    cancellation,
                    lease_failure,
                    transaction_anchor,
                ),
                "progress": lambda stage, current, total: self._lease_checked_progress(
                    run_dir,
                    cancellation,
                    lease_failure,
                    stage,
                    current,
                    total,
                    transaction_anchor,
                ),
            }
            if getattr(backend, "requires_destination_parent_anchor", False) is True:
                backend_arguments["destination_parent_anchor"] = transaction_anchor
            result = backend.acquire(
                source,
                artifacts,
                operation_limits,
                **backend_arguments,
            )
            self._ensure_lease_healthy(lease_failure)
            self._raise_if_operation_aborted(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            actual_elapsed = time.monotonic() - operation_started
            acquisition_receipt = self._validate_acquisition_result(source, result, actual_elapsed)
            with transaction_anchor.open_directory_immovable(artifacts.name) as artifacts_anchor:
                manifest = scan_artifacts(
                    artifacts,
                    self.limits,
                    expected_files=result.receipt.files,
                    artifacts_anchor=artifacts_anchor,
                    cancel_requested=lambda: self._lease_checked_cancellation(
                        run_dir,
                        cancellation,
                        lease_failure,
                        transaction_anchor,
                    ),
                    deadline_monotonic=operation_deadline,
                )
            self._ensure_lease_healthy(lease_failure)
            self._raise_if_operation_aborted(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            acquisition_receipt["artifact_manifest_identity"] = _json_identity(manifest)
            acquisition_receipt["acquisition_receipt_identity"] = _json_identity(acquisition_receipt)
            self._write_exclusive_json(
                run_dir / "acquisition_receipt.json",
                acquisition_receipt,
                parent_anchor=transaction_anchor,
            )
            self._write_exclusive_json(
                run_dir / "artifact_manifest.json",
                manifest,
                parent_anchor=transaction_anchor,
            )
            self._raise_if_operation_aborted(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            self._validate_live_configuration(source.source_id)
            self._raise_if_operation_aborted(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            handoff = self._build_handoff(run_id, source, acquisition_receipt, manifest)
            self._raise_if_operation_aborted(
                run_dir,
                cancellation,
                operation_deadline,
                run_anchor=transaction_anchor,
            )
            # Cancellation, deadline, and handoff publication share the
            # cross-process mutation lock at the final commit boundary.
            with self._mutation_guard():
                self._raise_if_operation_aborted(
                    run_dir,
                    cancellation,
                    operation_deadline,
                    run_anchor=transaction_anchor,
                )
                live_source = self._validate_live_configuration(source.source_id)
                self._raise_if_operation_aborted(
                    run_dir,
                    cancellation,
                    operation_deadline,
                    run_anchor=transaction_anchor,
                )
                if live_source.catalog_identity != source.catalog_identity:
                    raise HarvestError(
                        "harvest_live_configuration_changed",
                        "Harvest configuration changed before handoff publication.",
                    )
                self._write_exclusive_json(
                    run_dir / "handoff.json",
                    handoff,
                    parent_anchor=transaction_anchor,
                )
                self._raise_if_operation_aborted(
                    run_dir,
                    cancellation,
                    operation_deadline,
                    run_anchor=transaction_anchor,
                )
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
                    run_anchor=transaction_anchor,
                )
        except Exception:
            try:
                cancelled = (
                    self._cancellation_requested(run_dir, cancellation, run_anchor=transaction_anchor)
                    if run_dir is not None and transaction_anchor is not None
                    else cancellation.is_set()
                )
            except Exception:
                cancelled = cancellation.is_set()
            status = "CANCELLED" if cancelled else "FAILED"
            stage = "cancelled" if cancelled else "failed"
            message = (
                "Harvest was cancelled; immutable receipts and artifacts were retained."
                if cancelled
                else "Harvest failed closed. Private backend details were not recorded."
            )
            try:
                if run_dir is not None and transaction_anchor is not None:
                    self._transition(
                        run_dir,
                        status,
                        stage=stage,
                        message=message,
                        ended=True,
                        run_anchor=transaction_anchor,
                    )
            except Exception:
                pass
        finally:
            if lease_stop is not None:
                lease_stop.set()
            if lease_thread is not None:
                lease_thread.join(timeout=2)
            try:
                if run_dir is not None and transaction_anchor is not None:
                    self._release_lease(
                        run_dir,
                        self._read_state(run_dir, run_anchor=transaction_anchor),
                        run_anchor=transaction_anchor,
                    )
            except Exception:
                pass
            try:
                anchor_stack.close()
            except Exception:
                pass
            with self._lock:
                self._progress_observations.pop(run_id, None)
                self._last_cancellation_probe.pop(run_id, None)
                self._lease_stops.pop(run_id, None)
                self._lease_failures.pop(run_id, None)
                self._lease_threads.pop(run_id, None)

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
        snapshot_residue: dict[str, Any] | None = None
        if receipt.snapshot_residue is not None:
            residue = dict(receipt.snapshot_residue)
            if (
                set(residue) != {"kind", "relative_path", "byte_count", "sha256", "mode"}
                or residue.get("kind") != "retained_archive_snapshot_evidence"
                or not isinstance(residue.get("relative_path"), str)
                or not residue["relative_path"].startswith("downloads/.spritelab-archive-snapshot-evidence-")
                or Path(residue["relative_path"]).is_absolute()
                or ".." in Path(residue["relative_path"]).parts
                or residue.get("byte_count") != receipt.response_bytes
                or residue.get("sha256") != receipt.actual_response_sha256
                or residue.get("mode") != "0400"
            ):
                raise HarvestError(
                    "invalid_backend_receipt",
                    "Harvest archive snapshot residue evidence is invalid.",
                )
            snapshot_residue = residue
        direct_image_derivation: dict[str, Any] | None = None
        response_kind = "archive"
        if (
            receipt.response_mime_type in {"image/gif", "image/png", "image/webp"}
            and receipt.direct_image_derivation is None
        ):
            raise HarvestError(
                "invalid_backend_receipt",
                "Harvest image responses require exact direct-image derivation evidence.",
            )
        if receipt.direct_image_derivation is not None:
            direct = dict(receipt.direct_image_derivation)
            expected_source_format = {
                "image/gif": "GIF",
                "image/png": "PNG",
                "image/webp": "WEBP",
            }.get(receipt.response_mime_type)
            expected_direct_keys = {
                "schema_version",
                "kind",
                "source_format",
                "source_mime_type",
                "raw_byte_count",
                "raw_sha256",
                "frame_count",
                "width",
                "height",
                "decoded_rgba_sha256",
                "output_relative_path",
                "output_mime_type",
                "output_byte_count",
                "output_sha256",
                "recipe_identity",
                "derived",
                "source_bytes_modified",
            }
            matching_files = [
                item for item in receipt.files if item.relative_path == direct.get("output_relative_path")
            ]
            if (
                set(direct) != expected_direct_keys
                or direct.get("schema_version") != "spritelab.harvest.direct-image-derivation.v1"
                or direct.get("kind") != "direct_static_image"
                or direct.get("source_format") != expected_source_format
                or direct.get("source_mime_type") != receipt.response_mime_type
                or direct.get("raw_byte_count") != receipt.response_bytes
                or direct.get("raw_sha256") != receipt.actual_response_sha256
                or direct.get("frame_count") != 1
                or type(direct.get("width")) is not int
                or type(direct.get("height")) is not int
                or direct["width"] <= 0
                or direct["height"] <= 0
                or direct["width"] * direct["height"] > 16_777_216
                or SHA256_PATTERN.fullmatch(str(direct.get("decoded_rgba_sha256", ""))) is None
                or direct.get("output_relative_path") != "direct-image.png"
                or direct.get("output_mime_type") != "image/png"
                or type(direct.get("output_byte_count")) is not int
                or not 0 < direct["output_byte_count"] <= self.limits.max_file_bytes
                or SHA256_PATTERN.fullmatch(str(direct.get("output_sha256", ""))) is None
                or len(matching_files) != 1
                or matching_files[0].mime_type != "image/png"
                or matching_files[0].byte_count != direct.get("output_byte_count")
                or matching_files[0].sha256 != direct.get("output_sha256")
                or direct.get("recipe_identity") != "spritelab.harvest.direct-static-image-to-png.v1"
                or direct.get("derived") is not (receipt.response_mime_type != "image/png")
                or direct.get("source_bytes_modified") is not False
                or receipt.archive_members != 0
                or receipt.archive_uncompressed_bytes != 0
                or snapshot_residue is not None
            ):
                raise HarvestError(
                    "invalid_backend_receipt",
                    "Harvest direct-image derivation evidence is invalid.",
                )
            direct_image_derivation = direct
            response_kind = "direct_static_image"
        backend_evidence = self._backend_evidence_record()
        return {
            "schema_version": HARVEST_ACQUISITION_RECEIPT_SCHEMA,
            "source_id": source.source_id,
            "source_catalog_identity": source.catalog_identity,
            "source_evidence_binding_identity": source.evidence_binding.identity,
            "backend_capabilities": {
                **self._backend_capabilities.to_dict(),
                "capability_identity": self._backend_capabilities.identity,
            },
            "backend_capability_evidence": backend_evidence,
            "backend_capability_evidence_identity": backend_evidence["evidence_identity"],
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
            "snapshot_residue": snapshot_residue,
            "response_kind": response_kind,
            "direct_image_derivation": direct_image_derivation,
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
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> None:
        self._raise_if_cancelled(run_dir, cancellation, run_anchor=run_anchor)
        safe_stage = stage if STAGE_PATTERN.fullmatch(stage or "") else "acquiring"
        if type(current) is not int or current < 0:
            raise HarvestError("invalid_backend_progress", "Harvest backend progress was invalid.")
        if total is not None and (type(total) is not int or total < current):
            raise HarvestError("invalid_backend_progress", "Harvest backend progress was invalid.")
        with self._lock:
            previous = self._progress_observations.get(run_dir.name)
            if previous is not None and previous[0] == safe_stage and current < previous[1]:
                raise HarvestError("invalid_backend_progress", "Harvest backend progress moved backwards.")
            step = max(1, ((total or 0) + _MAX_PROGRESS_EVENTS_PER_STAGE - 1) // _MAX_PROGRESS_EVENTS_PER_STAGE)
            emitted = previous[4] if previous is not None and previous[0] == safe_stage else 0
            last_emitted = previous[3] if previous is not None and previous[0] == safe_stage else -step
            final = total is not None and current == total
            record = (
                previous is None or previous[0] != safe_stage or current == 0 or final or current - last_emitted >= step
            )
            if (final and emitted >= _MAX_PROGRESS_EVENTS_PER_STAGE) or (
                not final and emitted >= _MAX_PROGRESS_EVENTS_PER_STAGE - 1
            ):
                record = False
            if not record:
                self._progress_observations[run_dir.name] = (
                    safe_stage,
                    current,
                    total,
                    last_emitted,
                    emitted,
                )
                return
            self._progress_observations[run_dir.name] = (
                safe_stage,
                current,
                total,
                current,
                emitted + 1,
            )
        self._transition(
            run_dir,
            "RUNNING",
            stage=safe_stage,
            current=current,
            total=total,
            message="Certified acquisition is running within enforced limits.",
            run_anchor=run_anchor,
        )

    def _build_handoff(
        self,
        run_id: str,
        source: HarvestSource,
        acquisition_receipt: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_snapshot = source.to_public_dict()
        backend_evidence = self._backend_evidence_record()
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
            "backend_capability_evidence": backend_evidence,
            "backend_capability_evidence_identity": backend_evidence["evidence_identity"],
            "limits_identity": self.limits.identity,
            "acquisition_receipt_identity": acquisition_receipt["acquisition_receipt_identity"],
            "acquisition_kind": acquisition_receipt["response_kind"],
            "direct_image_derivation": acquisition_receipt.get("direct_image_derivation"),
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

    def _rehash_manifest(
        self,
        run_dir: Path,
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> dict[str, Any]:
        if run_anchor is None:
            with open_anchored_directory(run_dir, self.output_root) as anchored:
                return self._rehash_manifest(run_dir, run_anchor=anchored)
        stored = self._read_json(run_dir / "artifact_manifest.json", parent_anchor=run_anchor)
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
            with run_anchor.open_directory_immovable("artifacts") as artifacts_anchor:
                current = scan_artifacts(
                    run_dir / "artifacts",
                    self.limits,
                    expected_files=expected,
                    artifacts_anchor=artifacts_anchor,
                )
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
        run_anchor: AnchoredDirectory | None = None,
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
                run_anchor=run_anchor,
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
        run_anchor: AnchoredDirectory | None = None,
    ) -> None:
        state = self._read_state(run_dir, run_anchor=run_anchor)
        previous_status = str(state["status"])
        if status not in _LEGAL_TRANSITIONS.get(previous_status, frozenset()):
            raise HarvestError(
                "illegal_harvest_transition",
                f"Harvest state cannot transition from {previous_status} to {status}.",
            )
        if previous_status == "CANCELLING" and status == "COMPLETE":
            raise HarvestError("illegal_harvest_transition", "Cancelling Harvest cannot complete.")
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
        self._write_state(run_dir, state, run_anchor=run_anchor)
        event: Mapping[str, Any] | None = None
        if append_event:
            event = self._append_event(run_dir, state, run_anchor=run_anchor)
        if status == "COMPLETE":
            if event is None:
                raise HarvestError("harvest_terminal_commit_failed", "Harvest completion has no terminal event.")
            handoff = self._read_json(run_dir / "handoff.json", parent_anchor=run_anchor)
            terminal_commit = {
                "schema_version": HARVEST_TERMINAL_COMMIT_SCHEMA,
                "run_id": run_dir.name,
                "status": "COMPLETE",
                "state_identity": _json_identity(state),
                "terminal_event_identity": _json_identity(event),
                "handoff_identity": _json_identity(handoff),
                "created_at": now,
                "paths_exposed": False,
            }
            terminal_commit["terminal_commit_identity"] = _json_identity(terminal_commit)
            self._write_exclusive_json(
                run_dir / "terminal_commit.json",
                terminal_commit,
                parent_anchor=run_anchor,
            )

    def _complete_commit_valid(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> bool:
        try:
            commit = self._read_json(run_dir / "terminal_commit.json", parent_anchor=run_anchor)
            handoff = self._read_json(run_dir / "handoff.json", parent_anchor=run_anchor)
            events = self._read_events(run_dir, run_anchor=run_anchor)
        except (HarvestError, OSError):
            return False
        if not events:
            return False
        recorded_identity = commit.get("terminal_commit_identity")
        identity_payload = {key: value for key, value in commit.items() if key != "terminal_commit_identity"}
        return (
            commit.get("schema_version") == HARVEST_TERMINAL_COMMIT_SCHEMA
            and commit.get("run_id") == run_dir.name
            and commit.get("status") == "COMPLETE"
            and SHA256_PATTERN.fullmatch(str(recorded_identity or "")) is not None
            and recorded_identity == _json_identity(identity_payload)
            and commit.get("state_identity") == _json_identity(state)
            and commit.get("terminal_event_identity") == _json_identity(events[-1])
            and events[-1].get("status") == "COMPLETE"
            and commit.get("handoff_identity") == _json_identity(handoff)
        )

    def _managed_inventory_record(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
    ) -> dict[str, Any]:
        state = self._read_state(run_dir, run_anchor=run_anchor)
        request = self._read_request(run_dir, run_anchor=run_anchor)
        status = self._effective_status(state, run_dir.name, run_dir=run_dir, run_anchor=run_anchor)
        complete_commit_valid = status != "COMPLETE" or self._complete_commit_valid(
            run_dir,
            state,
            run_anchor=run_anchor,
        )
        if status == "COMPLETE" and not complete_commit_valid:
            status = "INTERRUPTED"
        return {
            "run_id": run_dir.name,
            "kind": "managed",
            "source_id": request["source_id"],
            "status": status,
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "handoff_ready": bool(state.get("handoff_ready", False)) and complete_commit_valid,
            "usable_count": int(state.get("usable_count", 0)),
            "quarantined_count": int(state.get("quarantined_count", 0)),
        }

    def _effective_status(
        self,
        state: Mapping[str, Any],
        run_id: str,
        *,
        run_dir: Path | None = None,
        run_anchor: AnchoredDirectory | None = None,
    ) -> str:
        status = str(state["status"])
        if status not in ACTIVE_STATUSES:
            return status
        worker = self._workers.get(run_id)
        if worker is not None and worker.is_alive():
            return status
        target = run_dir if run_dir is not None else self._run_directory(run_id)
        if self._lease_is_live(target, state, run_anchor=run_anchor):
            return status
        return "INTERRUPTED"

    def _active_source_run(self, source_id: str) -> str | None:
        if not self.output_root.exists():
            return None
        with open_anchored_directory(self.output_root, self.project_root) as output_anchor:
            for name in output_anchor.names():
                if RUN_ID_PATTERN.fullmatch(name) is None:
                    continue
                child = self.output_root / name
                try:
                    metadata = output_anchor.lstat(name)
                    if not stat.S_ISDIR(metadata.st_mode) or _metadata_is_link_or_reparse(metadata):
                        raise HarvestStorageError("unsafe managed run entry")
                    with output_anchor.open_directory_immovable(name) as child_anchor:
                        request = self._read_request(child, run_anchor=child_anchor)
                        state = self._read_state(child, run_anchor=child_anchor)
                        status = self._effective_status(
                            state,
                            name,
                            run_dir=child,
                            run_anchor=child_anchor,
                        )
                        if request["source_id"] == source_id and status in ACTIVE_STATUSES:
                            return name
                except (HarvestError, OSError, HarvestStorageError, UnsafeFilesystemOperation):
                    # A syntactically managed run whose state cannot be bound
                    # is conservatively treated as active.  Ignoring it could
                    # violate the source and project single-flight gates.
                    return name
        return None

    def _active_managed_run(self) -> str | None:
        """Return any live ordinary acquisition or catalog probe."""

        if not self.output_root.exists():
            return None
        with open_anchored_directory(self.output_root, self.project_root) as output_anchor:
            for name in output_anchor.names():
                try:
                    metadata = output_anchor.lstat(name)
                    if not stat.S_ISDIR(metadata.st_mode) or _metadata_is_link_or_reparse(metadata):
                        if RUN_ID_PATTERN.fullmatch(name) or PROBE_ID_PATTERN.fullmatch(name):
                            raise HarvestStorageError("unsafe managed run entry")
                        continue
                    child = self.output_root / name
                    with output_anchor.open_directory_immovable(name) as child_anchor:
                        if RUN_ID_PATTERN.fullmatch(name):
                            state = self._read_state(child, run_anchor=child_anchor)
                            if (
                                self._effective_status(
                                    state,
                                    name,
                                    run_dir=child,
                                    run_anchor=child_anchor,
                                )
                                in ACTIVE_STATUSES
                            ):
                                return name
                        elif PROBE_ID_PATTERN.fullmatch(name):
                            record = scan_probe_inventory_record(
                                child,
                                self.output_root,
                                run_anchor=child_anchor,
                            )
                            if record is None or record.get("status") in ACTIVE_PROBE_STATUSES:
                                return name
                except (OSError, HarvestError, HarvestStorageError, UnsafeFilesystemOperation):
                    if RUN_ID_PATTERN.fullmatch(name) or PROBE_ID_PATTERN.fullmatch(name):
                        return name
                    continue
        return None

    def _create_output_root(self) -> None:
        try:
            with open_anchored_directory(self.project_root, self.project_root) as project_anchor:
                if not project_anchor.lexists(self.output_root.name):
                    observed_identity = project_anchor.mkdir(self.output_root.name, exist_ok=False)
                else:
                    observed_identity = OwnedFileIdentity.from_stat(project_anchor.lstat(self.output_root.name))
                metadata = project_anchor.lstat(self.output_root.name)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or _metadata_is_link_or_reparse(metadata)
                    or not observed_identity.matches(metadata)
                ):
                    raise HarvestError("unsafe_harvest_root", "The managed Harvest root is unsafe.")
                with project_anchor.open_directory_immovable(self.output_root.name) as output_anchor:
                    if not observed_identity.matches(output_anchor.directory_metadata()):
                        raise HarvestError("unsafe_harvest_root", "The managed Harvest root changed while anchoring.")
                    output_anchor.verify()
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
                with RepositoryMutationLock(self.project_root):
                    yield
            except HarvestStorageError as exc:
                raise HarvestError("harvest_mutation_conflict", str(exc)) from exc

    def _create_run_directory(
        self,
        output_anchor: AnchoredDirectory,
        anchor_stack: ExitStack,
    ) -> tuple[str, Path, AnchoredDirectory]:
        output_anchor.verify()
        for _attempt in range(32):
            run_id = self._run_id_factory()
            if RUN_ID_PATTERN.fullmatch(run_id) is None:
                raise HarvestError("invalid_generated_run_id", "Harvest generated an unsafe run identifier.")
            run_dir = self.output_root / run_id
            try:
                created_identity = output_anchor.mkdir(run_id, exist_ok=False)
            except FileExistsError:
                continue
            run_anchor = anchor_stack.enter_context(output_anchor.open_directory_immovable(run_id))
            if not created_identity.matches(run_anchor.directory_metadata()):
                raise HarvestError(
                    "unsafe_harvest_evidence",
                    "Harvest run directory changed between creation and retained anchoring.",
                )
            return run_id, run_dir, run_anchor
        raise HarvestError("harvest_run_collision", "Harvest could not allocate a unique run.")

    def _find_idempotency_key(self, key: str) -> tuple[str, str] | None:
        if not self.output_root.exists():
            return None
        with open_anchored_directory(self.output_root, self.project_root) as output_anchor:
            for name in output_anchor.names():
                if RUN_ID_PATTERN.fullmatch(name) is None:
                    continue
                child = self.output_root / name
                try:
                    metadata = output_anchor.lstat(name)
                    if not stat.S_ISDIR(metadata.st_mode) or _metadata_is_link_or_reparse(metadata):
                        raise HarvestStorageError("unsafe managed run entry")
                    with output_anchor.open_directory_immovable(name) as child_anchor:
                        request = self._read_request(child, run_anchor=child_anchor)
                except (HarvestError, OSError, HarvestStorageError, UnsafeFilesystemOperation) as exc:
                    raise HarvestError(
                        "unsafe_harvest_evidence",
                        "A managed Harvest run could not be safely checked for idempotent reuse.",
                    ) from exc
                if request.get("idempotency_key") == key:
                    return name, str(request.get("request_fingerprint", ""))
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

    def _read_request(
        self,
        run_dir: Path,
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> dict[str, Any]:
        payload = self._read_json(run_dir / "request.json", parent_anchor=run_anchor)
        if (
            payload.get("schema_version") != HARVEST_REQUEST_SCHEMA
            or payload.get("run_id") != run_dir.name
            or SOURCE_ID_PATTERN.fullmatch(str(payload.get("source_id", ""))) is None
            or SHA256_PATTERN.fullmatch(str(payload.get("request_fingerprint", ""))) is None
            or SHA256_PATTERN.fullmatch(str(payload.get("backend_capability_evidence_identity", ""))) is None
        ):
            raise HarvestError("invalid_harvest_request", "Harvest request evidence is invalid.")
        return payload

    def _read_state(
        self,
        run_dir: Path,
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> dict[str, Any]:
        payload = self._read_json(run_dir / "state.json", parent_anchor=run_anchor)
        if (
            payload.get("schema_version") != HARVEST_STATE_SCHEMA
            or payload.get("run_id") != run_dir.name
            or payload.get("status") not in ACTIVE_STATUSES | TERMINAL_STATUSES
        ):
            raise HarvestError("invalid_harvest_state", "Harvest state evidence is invalid.")
        return payload

    def _read_json(
        self,
        path: Path,
        *,
        parent_anchor: AnchoredDirectory | None = None,
    ) -> dict[str, Any]:
        try:
            payload_bytes = read_stable_single_link_bytes(
                path,
                self.output_root,
                max_bytes=_MAX_METADATA_BYTES,
                parent_anchor=parent_anchor,
            )
        except FileNotFoundError as exc:
            raise HarvestError("missing_harvest_evidence", "Harvest durable evidence is incomplete.") from exc
        except (HarvestStorageError, UnsafeFilesystemOperation) as exc:
            raise HarvestError("unsafe_harvest_evidence", "Harvest durable evidence is unsafe.") from exc
        try:
            payload = strict_json_loads(payload_bytes)
        except (OSError, ValueError) as exc:
            raise HarvestError("invalid_harvest_evidence", "Harvest durable evidence is invalid.") from exc
        if not isinstance(payload, Mapping):
            raise HarvestError("invalid_harvest_evidence", "Harvest durable evidence is invalid.")
        return dict(payload)

    def _read_events(
        self,
        run_dir: Path,
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> list[dict[str, Any]]:
        path = run_dir / "events.jsonl"
        maximum = self.limits.max_events * self.limits.max_event_bytes
        try:
            content = read_stable_single_link_bytes(
                path,
                self.output_root,
                max_bytes=maximum,
                parent_anchor=run_anchor,
            )
        except FileNotFoundError:
            return []
        except (HarvestStorageError, UnsafeFilesystemOperation):
            return []
        events: list[dict[str, Any]] = []
        try:
            for line in content.decode("utf-8").splitlines()[-200:]:
                payload = strict_json_loads(line)
                if isinstance(payload, Mapping) and payload.get("schema_version") == HARVEST_EVENT_SCHEMA:
                    events.append(dict(payload))
        except (OSError, ValueError):
            return []
        return events

    def _write_state(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> None:
        target = run_dir / "state.json"
        if run_anchor is None:
            target = require_confined_path(target, run_dir)
        content = strict_json_dumps(dict(state), sort_keys=True, separators=(",", ":")).encode("utf-8")
        deadline = time.monotonic() + 2.0
        while True:
            try:
                write_atomic_stable_bytes(
                    target,
                    self.output_root,
                    content,
                    max_bytes=_MAX_METADATA_BYTES,
                    parent_anchor=run_anchor,
                )
                break
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.002)

    def _write_atomic_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        parent_anchor: AnchoredDirectory | None = None,
    ) -> None:
        try:
            if parent_anchor is None:
                require_confined_path(path, self.output_root)
            content = (strict_json_dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            write_atomic_stable_bytes(
                path,
                self.output_root,
                content,
                max_bytes=_MAX_METADATA_BYTES,
                parent_anchor=parent_anchor,
            )
        except (OSError, HarvestStorageError, UnsafeFilesystemOperation) as exc:
            raise HarvestError("unsafe_harvest_evidence", "Harvest durable evidence publication failed.") from exc

    def _write_exclusive_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        parent_anchor: AnchoredDirectory | None = None,
    ) -> None:
        if parent_anchor is None:
            try:
                require_confined_path(path, self.output_root)
            except UnsafeFilesystemOperation as exc:
                raise HarvestError("unsafe_harvest_evidence", "Harvest durable evidence is unsafe.") from exc
        content = strict_json_dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        try:
            write_exclusive_stable_bytes(
                path,
                self.output_root,
                (content + "\n").encode("utf-8"),
                max_bytes=_MAX_METADATA_BYTES,
                parent_anchor=parent_anchor,
            )
        except HarvestStorageError as exc:
            raise HarvestError("unsafe_harvest_evidence", "Harvest durable evidence is unsafe.") from exc

    def _append_event(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> dict[str, Any]:
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
        line = (strict_json_dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        if len(line) > self.limits.max_event_bytes:
            raise HarvestError("harvest_event_too_large", "Harvest event exceeded its byte limit.")
        path = run_dir / "events.jsonl"
        if run_anchor is None:
            path = require_confined_path(path, run_dir)
        try:
            append_stable_single_link_bytes(
                path,
                run_dir,
                line,
                max_bytes=self.limits.max_event_bytes,
                max_total_bytes=self.limits.max_events * self.limits.max_event_bytes,
                parent_anchor=run_anchor,
            )
        except HarvestStorageError as exc:
            raise HarvestError("unsafe_harvest_evidence", "Harvest event stream is unsafe or capped.") from exc
        return event


def _validate_callback_capability_binding(
    callback: DatasetImportCallback,
    capabilities: CertifiedBackendCapabilities | None,
) -> None:
    validate_callback_identity(callback)
    if capabilities is None:
        return
    if (
        callback.callback_id != capabilities.dataset_import_callback_id
        or callback.code_identity_sha256 != capabilities.dataset_import_callback_code_identity_sha256
        or callback.runtime_identity_sha256 != capabilities.dataset_import_callback_runtime_identity_sha256
    ):
        raise ValueError("Dataset import callback does not match the certified code and runtime binding.")


def _validate_idempotency_key(value: str) -> None:
    if IDEMPOTENCY_KEY_PATTERN.fullmatch(value or "") is None:
        raise HarvestError(
            "invalid_idempotency_key",
            "Provide an idempotency key between 8 and 128 safe characters.",
            status_code=422,
        )


def _validate_production_dataset_callback(
    callback: DatasetImportCallback,
    project_root: Path,
) -> None:
    """Reject identity strings copied onto an injected callback object."""

    from spritelab.product_features.conditioned_v5.intake import ConditionedDatasetImportAdapter

    if type(callback) is not ConditionedDatasetImportAdapter:
        raise ValueError("Production Dataset import requires the audited conditioned callback implementation.")
    callback_root = getattr(callback, "project_root", None)
    if not isinstance(callback_root, Path) or callback_root.absolute() != project_root:
        raise ValueError("Production Dataset import callback is not bound to the trusted project root.")


def _exact_int(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("not exact int")
    return value


def _default_run_id() -> str:
    return f"harvest-{uuid.uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _require_current_backend_evidence(
    evidence: BackendCapabilityEvidence,
    capabilities: CertifiedBackendCapabilities | None,
) -> None:
    if capabilities is None or evidence.capabilities != capabilities:
        raise ValueError("Harvest capability evidence does not match the configured backend.")
    audited = _parse_utc_timestamp(evidence.audited_at)
    issued = _parse_utc_timestamp(evidence.issued_at)
    expires = _parse_utc_timestamp(evidence.expires_at)
    now = datetime.now(timezone.utc)
    if audited > issued or issued > now or expires <= now or expires <= issued:
        raise ValueError("Harvest capability evidence is not currently valid.")
    if (
        SHA256_PATTERN.fullmatch(evidence.audit_report_sha256) is None
        or SHA256_PATTERN.fullmatch(evidence.audit_report_identity) is None
        or SHA256_PATTERN.fullmatch(evidence.certificate_identity) is None
        or evidence.implementation_identity_sha256 != capabilities.code_identity_sha256
    ):
        raise ValueError("Harvest capability evidence identities are invalid.")
    if evidence_has_current_validation_snapshot(evidence, capabilities):
        return
    current_implementation_identity = hardened_backend_code_identity()
    current_callback_binding = conditioned_dataset_import_callback_binding()
    if (
        capabilities.code_identity_sha256 != current_implementation_identity
        or evidence.implementation_identity_sha256 != current_implementation_identity
        or any(getattr(capabilities, key) != value for key, value in current_callback_binding.items())
    ):
        raise ValueError("Harvest capability evidence is not bound to current backend and callback code.")


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
