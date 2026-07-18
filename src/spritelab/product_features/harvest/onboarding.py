"""Durable, bounded source probes and explicit trusted-catalog promotion."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from spritelab.harvest.download import (
    DownloadCancelled,
    DownloadSecurityError,
    HostResolver,
    PinnedHTTPTransport,
    download_file_with_receipt,
)
from spritelab.product_core.events import strict_json_dumps, strict_json_loads
from spritelab.product_features.harvest.catalog import (
    INITIAL_LICENSE_POLICY,
    SHA256_PATTERN,
    SOURCE_ID_PATTERN,
    CatalogAutomationTermsBinding,
    CatalogEvidenceBinding,
    HarvestSource,
    load_trusted_catalog,
    public_download_url,
    public_url,
    url_identity,
)
from spritelab.product_features.harvest.catalog_verifier import (
    CATALOG_EVIDENCE_VERIFIER_ID,
    catalog_evidence_verifier_code_identity,
)
from spritelab.product_features.harvest.catalog_writer import (
    CatalogPromotionError,
    publish_trusted_catalog_source,
)
from spritelab.product_features.harvest.certification import BackendCapabilityEvidence
from spritelab.product_features.harvest.evidence_fetch import (
    MAX_EVIDENCE_PAGE_BYTES,
    MAX_ROBOTS_BYTES,
    EvidenceFetchError,
    FetchSnapshot,
    RobotsDecision,
    RobotsSnapshot,
    canonical_url_string,
    fetch_evidence_page,
    fetch_robots_snapshot,
    read_snapshot_bytes,
    rebuild_robots_snapshot,
    recover_direct_link,
    url_origin,
    verify_automation_terms,
    verify_evidence_pages,
)
from spritelab.product_features.harvest.storage import (
    HarvestStorageError,
    RepositoryMutationLock,
    append_stable_single_link_bytes,
    read_stable_single_link_bytes,
    write_atomic_stable_bytes,
    write_exclusive_stable_bytes,
)
from spritelab.product_features.harvest.trusted_backend import HarvestLimits
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation, require_confined_path

PROBE_REQUEST_SCHEMA = "spritelab.harvest.catalog-probe-request.v1"
PROBE_STATE_SCHEMA = "spritelab.harvest.catalog-probe-state.v1"
PROBE_EVENT_SCHEMA = "spritelab.harvest.catalog-probe-event.v1"
PROBE_RECEIPT_SCHEMA = "spritelab.harvest.catalog-probe-receipt.v1"
PROBE_RESULT_SCHEMA = "spritelab.harvest.catalog-probe-result.v1"
PROBE_TERMINAL_COMMIT_SCHEMA = "spritelab.harvest.catalog-probe-terminal-commit.v1"
PROBE_PROMOTION_RECEIPT_SCHEMA = "spritelab.harvest.catalog-promotion-receipt.v1"
PROBE_LEASE_SCHEMA = "spritelab.harvest.catalog-probe-worker-lease.v1"
PROBE_CANCELLATION_SCHEMA = "spritelab.harvest.catalog-probe-cancellation.v1"

PROBE_ID_PATTERN = re.compile(r"^probe-[0-9a-f]{32}$")
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
TAXONOMY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
ACTIVE_PROBE_STATUSES = frozenset({"QUEUED", "RUNNING", "CANCELLING"})
TERMINAL_PROBE_STATUSES = frozenset({"READY", "FAILED", "CANCELLED", "PROMOTED"})
RETRYABLE_PROBE_STATUSES = frozenset({"FAILED", "CANCELLED", "INTERRUPTED"})

_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_EVENTS = 500
_MAX_EVENT_BYTES = 4096
_PROMOTION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "probe_id",
        "source_id",
        "promoted_at",
        "explicit_action",
        "catalog_changed",
        "catalog_identity",
        "source_catalog_identity",
        "raw_response_sha256",
        "backend_capability_evidence_identity",
        "verifier_id",
        "verifier_code_identity_sha256",
        "probe_receipt_identity",
        "result_identity",
        "zero_cost_evidence_reviewed",
        "reviewed_verification_identity",
        "reviewed_source_pack_evidence_sha256",
        "terminal_commit_identity",
        "private_direct_url_exposed",
        "paths_exposed",
        "promotion_identity",
    }
)
_LEASE_SECONDS = 3.0
_HEARTBEAT_SECONDS = 0.5
_TEXT_LIMITS = {
    "title": (4, 200),
    "creator": (3, 200),
    "attribution_text": (2, 1000),
}
_REQUEST_KEYS = frozenset(
    {
        "source_id",
        "title",
        "creator",
        "source_page",
        "license_id",
        "license_evidence_url",
        "terms_evidence_url",
        "direct_download_url",
        "attribution_text",
        "taxonomy_hints",
        "inventory_identity",
        "backend_capability_evidence_identity",
        "idempotency_key",
        "explicit_action",
        "authorize_network",
        "authorize_hash_probe",
        "authorize_zero_cost",
        "authorize_permissive_license",
    }
)


class CatalogProbeError(RuntimeError):
    """Structured onboarding failure safe for durable/API exposure."""

    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class CatalogProbeCancelled(CatalogProbeError):
    def __init__(self) -> None:
        super().__init__("catalog_probe_cancelled", "Harvest catalog probe was cancelled.")


@dataclass(frozen=True)
class CatalogProbeInput:
    source_id: str
    title: str
    creator: str
    source_page: str
    license_id: str
    license_evidence_url: str
    terms_evidence_url: str | None
    direct_download_url: str
    attribution_text: str
    taxonomy_hints: tuple[str, ...]
    inventory_identity: str
    backend_capability_evidence_identity: str
    idempotency_key: str

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> CatalogProbeInput:
        if not isinstance(value, Mapping) or set(value) != _REQUEST_KEYS:
            raise CatalogProbeError(
                "invalid_catalog_probe",
                "Catalog probe fields are missing or unrecognized.",
                status_code=422,
            )
        for gate in (
            "explicit_action",
            "authorize_network",
            "authorize_hash_probe",
            "authorize_zero_cost",
            "authorize_permissive_license",
        ):
            if value.get(gate) is not True:
                raise CatalogProbeError(
                    "catalog_probe_authorization_required",
                    "Catalog probing requires explicit network, hash-probe, zero-cost, and license authorization.",
                    status_code=422,
                )
        source_id = value.get("source_id")
        if not isinstance(source_id, str) or SOURCE_ID_PATTERN.fullmatch(source_id.strip()) is None:
            raise CatalogProbeError("invalid_catalog_probe", "Catalog probe source_id is invalid.", status_code=422)
        text: dict[str, str] = {}
        for field, (minimum, maximum) in _TEXT_LIMITS.items():
            item = value.get(field)
            if not isinstance(item, str) or not minimum <= len(item.strip()) <= maximum:
                raise CatalogProbeError("invalid_catalog_probe", f"Catalog probe {field} is invalid.", status_code=422)
            if any(ord(character) < 32 and character not in "\n\t" for character in item):
                raise CatalogProbeError("invalid_catalog_probe", f"Catalog probe {field} is invalid.", status_code=422)
            text[field] = item.strip()
        license_id = value.get("license_id")
        if not isinstance(license_id, str) or license_id.strip().casefold() not in INITIAL_LICENSE_POLICY:
            raise CatalogProbeError(
                "unsupported_catalog_license",
                "Catalog onboarding is fixed to CC0-1.0 or explicit public-domain sources.",
                status_code=422,
            )
        source_page = _canonical_evidence_url(value.get("source_page"), "source page")
        license_url = _canonical_evidence_url(value.get("license_evidence_url"), "license evidence")
        raw_terms_url = value.get("terms_evidence_url")
        if raw_terms_url is None or raw_terms_url == "":
            terms_url = None
        else:
            terms_url = _canonical_evidence_url(raw_terms_url, "automation terms evidence")
        direct = _canonical_direct_url(value.get("direct_download_url"))
        taxonomy = value.get("taxonomy_hints")
        if (
            not isinstance(taxonomy, list)
            or len(taxonomy) > 32
            or not all(isinstance(item, str) and TAXONOMY_PATTERN.fullmatch(item) for item in taxonomy)
            or len(set(taxonomy)) != len(taxonomy)
        ):
            raise CatalogProbeError(
                "invalid_catalog_probe", "Catalog probe taxonomy hints are invalid.", status_code=422
            )
        inventory_identity = value.get("inventory_identity")
        evidence_identity = value.get("backend_capability_evidence_identity")
        if (
            not isinstance(inventory_identity, str)
            or SHA256_PATTERN.fullmatch(inventory_identity) is None
            or not isinstance(evidence_identity, str)
            or SHA256_PATTERN.fullmatch(evidence_identity) is None
        ):
            raise CatalogProbeError(
                "invalid_catalog_probe",
                "Catalog probe inventory or capability evidence identity is invalid.",
                status_code=422,
            )
        idempotency_key = value.get("idempotency_key")
        if not isinstance(idempotency_key, str) or IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key) is None:
            raise CatalogProbeError(
                "invalid_idempotency_key",
                "Provide a catalog-probe idempotency key between 8 and 128 safe characters.",
                status_code=422,
            )
        return cls(
            source_id=source_id.strip(),
            title=text["title"],
            creator=text["creator"],
            source_page=source_page,
            license_id=license_id.strip().casefold(),
            license_evidence_url=license_url,
            terms_evidence_url=terms_url,
            direct_download_url=direct,
            attribution_text=text["attribution_text"],
            taxonomy_hints=tuple(taxonomy),
            inventory_identity=inventory_identity,
            backend_capability_evidence_identity=evidence_identity,
            idempotency_key=idempotency_key,
        )

    def durable_payload(self) -> dict[str, Any]:
        parsed_public = public_download_url(self.direct_download_url, (_download_host(self.direct_download_url),))
        return {
            "source_id": self.source_id,
            "title": self.title,
            "creator": self.creator,
            "source_page": self.source_page,
            "license_id": self.license_id,
            "license_evidence_url": self.license_evidence_url,
            "terms_evidence_url": self.terms_evidence_url,
            "direct_download_url_public": parsed_public,
            "direct_download_url_sha256": url_identity(self.direct_download_url),
            "direct_download_host": _download_host(self.direct_download_url),
            "attribution_text": self.attribution_text,
            "taxonomy_hints": list(self.taxonomy_hints),
            "inventory_identity": self.inventory_identity,
            "backend_capability_evidence_identity": self.backend_capability_evidence_identity,
            "idempotency_key": self.idempotency_key,
            "explicit_action": True,
            "authorize_network": True,
            "authorize_hash_probe": True,
            "authorize_zero_cost": True,
            "authorize_permissive_license": True,
            "zero_cost": True,
            "permissive": True,
            "private_direct_url_exposed": False,
        }


class CatalogProbeService:
    """Own probe state separately from ordinary expected-hash acquisitions."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        limits: HarvestLimits,
        resolver: HostResolver | None = None,
        transport: PinnedHTTPTransport | None = None,
        downloader: Any = download_file_with_receipt,
        run_id_factory: Callable[[], str] | None = None,
        catalog_refreshed: Callable[[tuple[HarvestSource, ...]], None] | None = None,
    ) -> None:
        self.project_root = Path(project_root).absolute()
        self.output_root = require_confined_path(self.project_root / "harvest_runs", self.project_root)
        self.limits = limits
        self._resolver = resolver
        self._transport = transport
        self._downloader = downloader
        self._run_id_factory = run_id_factory or (lambda: f"probe-{uuid.uuid4().hex}")
        self._catalog_refreshed = catalog_refreshed
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._cancellations: dict[str, threading.Event] = {}
        self._heartbeat_stops: dict[str, threading.Event] = {}
        self._heartbeat_failures: dict[str, threading.Event] = {}
        self._instance_id = uuid.uuid4().hex

    def start(
        self,
        payload: Mapping[str, Any],
        *,
        capability_evidence: BackendCapabilityEvidence,
        current_inventory_identity: str | Callable[[], str],
        retry_of: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        request = CatalogProbeInput.from_value(payload)
        self._validate_capability_evidence(request, capability_evidence)
        durable = request.durable_payload()
        fingerprint = _identity(durable)
        with self._mutation_guard():
            existing = self._find_idempotency(request.idempotency_key)
            if existing is not None:
                probe_id, prior_fingerprint = existing
                if prior_fingerprint != fingerprint:
                    raise CatalogProbeError(
                        "catalog_probe_idempotency_conflict",
                        "Catalog probe idempotency key was reused for a different request.",
                    )
                return self.status(probe_id), False
            locked_inventory_identity = (
                current_inventory_identity() if callable(current_inventory_identity) else current_inventory_identity
            )
            if request.inventory_identity != locked_inventory_identity:
                raise CatalogProbeError(
                    "harvest_inventory_changed",
                    "Harvest inventory changed after onboarding review; refresh before probing.",
                )
            active = self._active_single_flight()
            if active is not None:
                raise CatalogProbeError(
                    "harvest_single_flight_conflict",
                    f"Managed Harvest run {active!r} is already active; wait for it to finish before probing.",
                )
            probe_id = self._allocate_probe_directory()
            run_dir = self.output_root / probe_id
            token = uuid.uuid4().hex
            now = _utc_now()
            with self._open_probe_anchor(probe_id) as run_anchor:
                run_anchor.mkdir("evidence")
                run_anchor.mkdir("robots")
                run_anchor.mkdir("quarantine")
                request_record = {
                    "schema_version": PROBE_REQUEST_SCHEMA,
                    "probe_id": probe_id,
                    "request_fingerprint": fingerprint,
                    "requested_at": now,
                    "retry_of": retry_of,
                    **durable,
                    "capability_evidence": _capability_record(capability_evidence),
                }
                state = {
                    "schema_version": PROBE_STATE_SCHEMA,
                    "probe_id": probe_id,
                    "source_id": request.source_id,
                    "status": "QUEUED",
                    "stage": "queued",
                    "message": "Catalog probe is queued behind a durable worker lease.",
                    "current": 0,
                    "total": None,
                    "event_count": 1,
                    "created_at": now,
                    "updated_at": now,
                    "ended_at": None,
                    "owner_pid": os.getpid(),
                    "owner_instance_id": self._instance_id,
                    "lease_token": token,
                    "paths_exposed": False,
                }
                self._write_exclusive_json(run_dir / "request.json", request_record, run_anchor)
                self._write_exclusive_json(run_dir / "state.json", state, run_anchor)
                self._write_exclusive_json(
                    run_dir / "worker_lease.json", self._new_lease(probe_id, token, 0), run_anchor
                )
                self._append_event(run_dir, state, run_anchor)
        cancellation = threading.Event()
        worker = threading.Thread(
            target=self._run_worker,
            args=(probe_id, request, capability_evidence, cancellation),
            name=f"spritelab-{probe_id}",
            daemon=True,
        )
        with self._lock:
            self._workers[probe_id] = worker
            self._cancellations[probe_id] = cancellation
        worker.start()
        return self.status(probe_id), True

    def retry(
        self,
        probe_id: str,
        payload: Mapping[str, Any],
        *,
        capability_evidence: BackendCapabilityEvidence,
        current_inventory_identity: str | Callable[[], str],
    ) -> tuple[dict[str, Any], bool]:
        previous = self.status(probe_id)
        if previous["status"] not in RETRYABLE_PROBE_STATUSES:
            raise CatalogProbeError(
                "catalog_probe_retry_not_allowed",
                "Only failed, cancelled, or interrupted catalog probes can be retried.",
            )
        return self.start(
            payload,
            capability_evidence=capability_evidence,
            current_inventory_identity=current_inventory_identity,
            retry_of=probe_id,
        )

    def cancel(self, probe_id: str, *, explicit_action: bool) -> dict[str, Any]:
        if explicit_action is not True:
            raise CatalogProbeError(
                "explicit_action_required",
                "Cancelling a catalog probe requires an explicit user action.",
                status_code=422,
            )
        with self._mutation_guard():
            run_dir = self._probe_directory(probe_id)
            with self._open_probe_anchor(probe_id) as run_anchor:
                state = self._read_state(run_dir, run_anchor)
                if state["status"] in TERMINAL_PROBE_STATUSES:
                    return self.status(probe_id)
                if not run_anchor.lexists("cancellation_request.json"):
                    self._write_exclusive_json(
                        run_dir / "cancellation_request.json",
                        {
                            "schema_version": PROBE_CANCELLATION_SCHEMA,
                            "probe_id": probe_id,
                            "requested_at": _utc_now(),
                            "explicit_action": True,
                            "paths_exposed": False,
                        },
                        run_anchor,
                    )
                self._cancellations.setdefault(probe_id, threading.Event()).set()
                worker = self._workers.get(probe_id)
                if (worker is None or not worker.is_alive()) and not self._lease_is_live(run_dir, state, run_anchor):
                    self._transition(run_dir, run_anchor, "CANCELLED", "cancelled", "Catalog probe was cancelled.")
                else:
                    self._transition(
                        run_dir,
                        run_anchor,
                        "CANCELLING",
                        "cancelling",
                        "Cancellation is waiting for the bounded fetch boundary.",
                    )
        return self.status(probe_id)

    def status(self, probe_id: str) -> dict[str, Any]:
        with self._lock:
            run_dir = self._probe_directory(probe_id)
            with self._open_probe_anchor(probe_id) as run_anchor:
                request = self._read_request(run_dir, run_anchor)
                state = self._read_state(run_dir, run_anchor)
                status = str(state["status"])
                if status in ACTIVE_PROBE_STATUSES and not self._lease_is_live(run_dir, state, run_anchor):
                    status = "INTERRUPTED"
                if status in {"READY", "PROMOTED"} and not self._terminal_commit_valid(run_dir, run_anchor, state):
                    status = "INTERRUPTED"
                result = self._read_optional_json(run_dir / "result.json", run_anchor)
                promotion = self._read_optional_json(run_dir / "promotion_receipt.json", run_anchor)
                return {
                    "schema_version": "spritelab.harvest.catalog-probe-status.v1",
                    "probe_id": probe_id,
                    "source_id": request["source_id"],
                    "title": request["title"],
                    "status": status,
                    "stage": "interrupted" if status == "INTERRUPTED" else state["stage"],
                    "message": (
                        "The probe worker lease expired before a valid terminal commit."
                        if status == "INTERRUPTED"
                        else state["message"]
                    ),
                    "current": state.get("current", 0),
                    "total": state.get("total"),
                    "created_at": state["created_at"],
                    "updated_at": state["updated_at"],
                    "ended_at": state.get("ended_at"),
                    "retry_of": request.get("retry_of"),
                    "promotion_ready": status == "READY",
                    "promoted": status == "PROMOTED" and promotion is not None,
                    "raw_response_sha256": result.get("raw_response_sha256") if result else None,
                    "events": self._read_events(run_dir, run_anchor),
                    "paths_exposed": False,
                }

    def evidence(self, probe_id: str) -> dict[str, Any]:
        run_dir = self._probe_directory(probe_id)
        with self._open_probe_anchor(probe_id) as run_anchor:
            request = self._read_request(run_dir, run_anchor)
            public_request = {key: value for key, value in request.items() if key != "capability_evidence"}
            return {
                "schema_version": "spritelab.harvest.catalog-probe-evidence.v1",
                "probe": self.status(probe_id),
                "request": public_request,
                "capability_evidence": request["capability_evidence"],
                "receipt": self._read_optional_json(run_dir / "probe_receipt.json", run_anchor),
                "result": self._read_optional_json(run_dir / "result.json", run_anchor),
                "terminal_commit": self._read_optional_json(run_dir / "terminal_commit.json", run_anchor),
                "promotion_receipt": self._read_optional_json(run_dir / "promotion_receipt.json", run_anchor),
                "network_actions_recorded": _recorded_network_actions(
                    self._read_optional_json(run_dir / "probe_receipt.json", run_anchor)
                ),
                "network_actions_triggered_by_this_read": 0,
                "private_direct_url_exposed": False,
                "paths_exposed": False,
            }

    def promote(
        self,
        probe_id: str,
        *,
        explicit_action: bool,
        authorize_catalog_promotion: bool,
        authorize_zero_cost_evidence_review: bool,
        reviewed_verification_identity: str | None,
        reviewed_source_pack_evidence_sha256: str | None,
        capability_evidence: BackendCapabilityEvidence,
    ) -> dict[str, Any]:
        if (
            explicit_action is not True
            or authorize_catalog_promotion is not True
            or authorize_zero_cost_evidence_review is not True
        ):
            raise CatalogProbeError(
                "catalog_promotion_authorization_required",
                "Trusted-catalog promotion requires separate explicit catalog and exact zero-cost evidence review authorizations.",
                status_code=422,
            )
        with self._mutation_guard():
            run_dir = self._probe_directory(probe_id)
            with self._open_probe_anchor(probe_id) as run_anchor:
                request = self._read_request(run_dir, run_anchor)
                state = self._read_state(run_dir, run_anchor)
                self._validate_promotion_capability_evidence(request, capability_evidence)
                if run_anchor.lexists("promotion_receipt.json"):
                    promotion_receipt = self._read_json(run_dir / "promotion_receipt.json", run_anchor)
                    probe_receipt = self._read_json(run_dir / "probe_receipt.json", run_anchor)
                    result = self._read_json(run_dir / "result.json", run_anchor)
                    terminal_commit = self._read_json(run_dir / "terminal_commit.json", run_anchor)
                    catalog = load_trusted_catalog(self.project_root)
                    self._validate_promotion_receipt(
                        promotion_receipt,
                        probe_id=probe_id,
                        request=request,
                        probe_receipt=probe_receipt,
                        result=result,
                        terminal_commit=terminal_commit,
                        catalog=catalog,
                        capability_evidence=capability_evidence,
                        reviewed_verification_identity=reviewed_verification_identity,
                        reviewed_source_pack_evidence_sha256=reviewed_source_pack_evidence_sha256,
                    )
                    self._notify_catalog_refreshed(catalog)
                    return promotion_receipt
                if state["status"] != "READY" or not self._terminal_commit_valid(run_dir, run_anchor, state):
                    raise CatalogProbeError(
                        "catalog_probe_not_promotable",
                        "Only an unchanged READY probe with a valid terminal commit can be promoted.",
                    )
                receipt = self._read_json(run_dir / "probe_receipt.json", run_anchor)
                result = self._read_json(run_dir / "result.json", run_anchor)
                self._validate_receipt_records(probe_id, request, receipt, result)
                _validate_exact_zero_cost_review(
                    reviewed_verification_identity=reviewed_verification_identity,
                    reviewed_source_pack_evidence_sha256=reviewed_source_pack_evidence_sha256,
                    expected_verification_identity=receipt.get("verification_identity"),
                    expected_source_pack_evidence_sha256=receipt.get("source_pack_evidence_sha256"),
                )
                try:
                    source = self._reverify_promoted_source(run_dir, run_anchor, request, receipt, result)
                except CatalogProbeError:
                    raise
                except (EvidenceFetchError, OSError, ValueError, UnsafeFilesystemOperation) as exc:
                    raise CatalogProbeError(
                        "catalog_probe_evidence_changed",
                        "Retained catalog-probe evidence changed or became unsafe before promotion.",
                    ) from exc
                try:
                    catalog, changed = publish_trusted_catalog_source(
                        self.project_root,
                        self.project_root,
                        source,
                        lock_held=True,
                    )
                except (CatalogPromotionError, OSError, ValueError, UnsafeFilesystemOperation) as exc:
                    raise CatalogProbeError(
                        "catalog_promotion_conflict",
                        "Trusted catalog changed or conflicts with the promoted source.",
                    ) from exc
                promoted_at = _utc_now()
                promotion = {
                    "schema_version": PROBE_PROMOTION_RECEIPT_SCHEMA,
                    "probe_id": probe_id,
                    "source_id": source.source_id,
                    "promoted_at": promoted_at,
                    "explicit_action": True,
                    "catalog_changed": changed,
                    "catalog_identity": _catalog_identity_from_sources(catalog),
                    "source_catalog_identity": source.catalog_identity,
                    "raw_response_sha256": source.expected_response_sha256,
                    "backend_capability_evidence_identity": capability_evidence.identity,
                    "verifier_id": CATALOG_EVIDENCE_VERIFIER_ID,
                    "verifier_code_identity_sha256": catalog_evidence_verifier_code_identity(),
                    "probe_receipt_identity": _identity(receipt),
                    "result_identity": _identity(result),
                    "zero_cost_evidence_reviewed": True,
                    "reviewed_verification_identity": reviewed_verification_identity,
                    "reviewed_source_pack_evidence_sha256": reviewed_source_pack_evidence_sha256,
                    "terminal_commit_identity": _identity(
                        self._read_json(run_dir / "terminal_commit.json", run_anchor)
                    ),
                    "private_direct_url_exposed": False,
                    "paths_exposed": False,
                }
                promotion["promotion_identity"] = _identity(promotion)
                self._write_exclusive_json(run_dir / "promotion_receipt.json", promotion, run_anchor)
                self._transition(
                    run_dir,
                    run_anchor,
                    "PROMOTED",
                    "promoted",
                    "Probe evidence was reverified and atomically promoted to the trusted catalog.",
                    ended=True,
                )
                self._notify_catalog_refreshed(catalog)
                return promotion

    def _run_worker(
        self,
        probe_id: str,
        request: CatalogProbeInput,
        capability_evidence: BackendCapabilityEvidence,
        cancellation: threading.Event,
    ) -> None:
        run_dir = self.output_root / probe_id
        deadline = time.monotonic() + self.limits.max_duration_seconds
        heartbeat_stop: threading.Event | None = None
        heartbeat_failure: threading.Event | None = None
        heartbeat_thread: threading.Thread | None = None
        run_anchor: AnchoredDirectory | None = None
        try:
            with self._open_probe_anchor(probe_id) as opened_run_anchor:
                run_anchor = opened_run_anchor.detached_duplicate()
                state = self._read_state(run_dir, run_anchor)
                lease_token = str(state["lease_token"])
                heartbeat_stop, heartbeat_failure, heartbeat_thread = self._start_heartbeat(run_dir, state, run_anchor)

                def cancelled() -> bool:
                    if heartbeat_failure is not None and heartbeat_failure.is_set():
                        raise CatalogProbeError(
                            "catalog_probe_lease_failed",
                            "Catalog probe worker lease renewal failed closed.",
                        )
                    if cancellation.is_set():
                        return True
                    if run_anchor is not None and run_anchor.lexists("cancellation_request.json"):
                        return True
                    if time.monotonic() >= deadline:
                        raise CatalogProbeError(
                            "catalog_probe_deadline_exceeded",
                            "Catalog probe exceeded its whole-probe duration limit.",
                        )
                    return False

                def remaining_seconds(stage_limit: float) -> float:
                    if cancelled():
                        raise CatalogProbeCancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CatalogProbeError(
                            "catalog_probe_deadline_exceeded",
                            "Catalog probe exceeded its whole-probe duration limit.",
                        )
                    return min(stage_limit, remaining)

                self._guarded_transition(
                    run_dir,
                    run_anchor,
                    "RUNNING",
                    "robots",
                    "Checking pinned robots policy before each distinct request path.",
                    abort_check=cancelled,
                )
                with ExitStack() as stack:
                    evidence_anchor = stack.enter_context(run_anchor.open_directory("evidence"))
                    robots_anchor = stack.enter_context(run_anchor.open_directory("robots"))
                    quarantine_anchor = stack.enter_context(run_anchor.open_directory("quarantine"))
                    snapshots: dict[str, RobotsSnapshot] = {}
                    decisions: dict[str, list[RobotsDecision]] = {}

                    def authorize_url(url: str) -> None:
                        origin = url_origin(url)
                        snapshot = snapshots.get(origin)
                        if snapshot is None:
                            name = f"robots-{hashlib.sha256(origin.encode('utf-8')).hexdigest()[:20]}.txt"
                            snapshot = fetch_robots_snapshot(
                                url,
                                robots_anchor,
                                name,
                                cancel_requested=cancelled,
                                progress=self._progress_callback(run_dir, run_anchor, cancellation, "robots"),
                                resolver=self._resolver,
                                transport=self._transport,
                                timeout_seconds=remaining_seconds(30.0),
                                downloader=self._downloader,
                            )
                            snapshots[origin] = snapshot
                            decisions[origin] = []
                        decisions[origin].append(snapshot.evaluate(url))

                    authorize_url(request.source_page)
                    self._guarded_transition(
                        run_dir,
                        run_anchor,
                        "RUNNING",
                        "source_evidence",
                        "Fetching the bounded creator source page through pinned public DNS.",
                        abort_check=cancelled,
                    )
                    source_snapshot = fetch_evidence_page(
                        request.source_page,
                        evidence_anchor,
                        "source_page.bin",
                        cancel_requested=cancelled,
                        progress=self._progress_callback(run_dir, run_anchor, cancellation, "source_evidence"),
                        resolver=self._resolver,
                        transport=self._transport,
                        timeout_seconds=remaining_seconds(60.0),
                        downloader=self._downloader,
                    )
                    terms_snapshot: FetchSnapshot | None = None
                    terms_bytes: bytes | None = None
                    if request.terms_evidence_url is not None:
                        authorize_url(request.terms_evidence_url)
                        self._guarded_transition(
                            run_dir,
                            run_anchor,
                            "RUNNING",
                            "automation_terms",
                            "Fetching source-bound automation terms through pinned public DNS.",
                            abort_check=cancelled,
                        )
                        terms_snapshot = fetch_evidence_page(
                            request.terms_evidence_url,
                            evidence_anchor,
                            "terms_page.bin",
                            cancel_requested=cancelled,
                            progress=self._progress_callback(run_dir, run_anchor, cancellation, "automation_terms"),
                            resolver=self._resolver,
                            transport=self._transport,
                            timeout_seconds=remaining_seconds(60.0),
                            downloader=self._downloader,
                        )
                        terms_bytes = read_snapshot_bytes(
                            evidence_anchor,
                            "terms_page.bin",
                            max_bytes=MAX_EVIDENCE_PAGE_BYTES,
                        )
                    authorize_url(request.license_evidence_url)
                    self._guarded_transition(
                        run_dir,
                        run_anchor,
                        "RUNNING",
                        "license_evidence",
                        "Fetching the bounded license page through pinned public DNS.",
                        abort_check=cancelled,
                    )
                    license_snapshot = fetch_evidence_page(
                        request.license_evidence_url,
                        evidence_anchor,
                        "license_page.bin",
                        cancel_requested=cancelled,
                        progress=self._progress_callback(run_dir, run_anchor, cancellation, "license_evidence"),
                        resolver=self._resolver,
                        transport=self._transport,
                        timeout_seconds=remaining_seconds(60.0),
                        downloader=self._downloader,
                    )
                    source_bytes = read_snapshot_bytes(
                        evidence_anchor,
                        "source_page.bin",
                        max_bytes=MAX_EVIDENCE_PAGE_BYTES,
                    )
                    license_bytes = read_snapshot_bytes(
                        evidence_anchor,
                        "license_page.bin",
                        max_bytes=MAX_EVIDENCE_PAGE_BYTES,
                    )
                    verified = verify_evidence_pages(
                        source_url=request.source_page,
                        source_snapshot=source_snapshot,
                        source_bytes=source_bytes,
                        license_url=request.license_evidence_url,
                        license_snapshot=license_snapshot,
                        license_bytes=license_bytes,
                        title=request.title,
                        creator=request.creator,
                        license_id=request.license_id,
                        direct_download_url=request.direct_download_url,
                    )
                    automation_terms = verify_automation_terms(
                        source_url=request.source_page,
                        source_bytes=source_bytes,
                        source_mime_type=source_snapshot.mime_type,
                        source_content_sha256=source_snapshot.content_sha256,
                        terms_url=request.terms_evidence_url,
                        terms_bytes=terms_bytes,
                        terms_snapshot=terms_snapshot,
                    )
                    if automation_terms.decision == "BLOCK":
                        raise EvidenceFetchError("Source automation terms explicitly prohibit automated downloading.")
                    authorize_url(verified.direct_download_url)
                    self._guarded_transition(
                        run_dir,
                        run_anchor,
                        "RUNNING",
                        "hash_probe",
                        "Downloading raw bytes into quarantine without opening, decoding, extracting, or importing.",
                        abort_check=cancelled,
                    )
                    raw_remaining = remaining_seconds(60.0)
                    raw_result = self._downloader(
                        verified.direct_download_url,
                        quarantine_anchor.directory / "raw_payload.bin",
                        allowed_hosts=(verified.direct_download_host,),
                        overwrite=False,
                        timeout_seconds=raw_remaining,
                        max_duration_seconds=raw_remaining,
                        allowed_content_types=self.limits.allowed_response_mime_types,
                        max_bytes=self.limits.max_response_bytes,
                        expected_sha256=None,
                        max_redirects=0,
                        require_https=True,
                        allow_private_hosts=False,
                        cancel_requested=cancelled,
                        progress=self._progress_callback(run_dir, run_anchor, cancellation, "hash_probe"),
                        resolver=self._resolver,
                        transport=self._transport,
                        destination_anchor=quarantine_anchor,
                    )
                    if cancelled():
                        raise CatalogProbeCancelled()
                    robots_records = [
                        snapshots[origin].to_dict(tuple(decisions[origin])) for origin in sorted(snapshots)
                    ]
                    verified_at = _utc_now()
                    raw_receipt = raw_result.receipt
                    receipt = {
                        "schema_version": PROBE_RECEIPT_SCHEMA,
                        "probe_id": probe_id,
                        "source_id": request.source_id,
                        "verified_at": verified_at,
                        "request_fingerprint": _identity(request.durable_payload()),
                        "backend_capability_evidence": _capability_record(capability_evidence),
                        "verifier_id": CATALOG_EVIDENCE_VERIFIER_ID,
                        "verifier_code_identity_sha256": catalog_evidence_verifier_code_identity(),
                        "source_evidence": source_snapshot.to_dict(),
                        "license_evidence": license_snapshot.to_dict(),
                        "robots_evidence": robots_records,
                        "automation_terms": automation_terms.to_dict(terms_snapshot),
                        "source_terms_evidence_url": automation_terms.evidence_url,
                        "terms_policy_decision": automation_terms.decision,
                        "terms_policy_blocked": False,
                        "tos_inference_performed": False,
                        "creator_posted_direct_link_verified": True,
                        "same_pack_evidence_verified": True,
                        "zero_cost_evidence_verified": verified.zero_cost_verified,
                        "license_conflict_checked": verified.license_conflict_checked,
                        "source_pack_evidence_text": verified.source_pack_evidence_text,
                        "source_pack_evidence_sha256": hashlib.sha256(
                            verified.source_pack_evidence_text.encode("utf-8")
                        ).hexdigest(),
                        "direct_download_url_public": public_url(verified.direct_download_url),
                        "direct_download_url_sha256": url_identity(verified.direct_download_url),
                        "raw_response": {
                            "final_url": public_url(raw_receipt.final_url),
                            "final_url_sha256": url_identity(raw_receipt.final_url),
                            "http_status": raw_receipt.http_status,
                            "mime_type": raw_receipt.response_mime_type,
                            "byte_count": raw_receipt.response_bytes,
                            "sha256": raw_receipt.response_sha256,
                            "elapsed_seconds": round(raw_receipt.elapsed_seconds, 6),
                            "redirect_count": len(raw_receipt.redirect_chain),
                            "relative_file": "quarantine/raw_payload.bin",
                            "expected_sha256_absent_by_probe_policy": True,
                        },
                        "verification_identity": verified.verification_identity,
                        "automation_terms_identity": automation_terms.decision_identity,
                        "policy": {
                            "zero_cost": True,
                            "permissive_license": True,
                            "network_authorized": True,
                            "hash_probe_authorized": True,
                            "robots_checked_before_each_path": True,
                        },
                        "extraction_count": 0,
                        "decode_count": 0,
                        "import_count": 0,
                        "candidate_count": 0,
                        "discovery_count": 0,
                        "raw_quarantine_only": True,
                        "private_direct_url_exposed": False,
                        "paths_exposed": False,
                    }
                    receipt["receipt_identity"] = _identity(receipt)
                    result = {
                        "schema_version": PROBE_RESULT_SCHEMA,
                        "probe_id": probe_id,
                        "source_id": request.source_id,
                        "raw_response_sha256": raw_receipt.response_sha256,
                        "raw_response_bytes": raw_receipt.response_bytes,
                        "raw_response_mime_type": raw_receipt.response_mime_type,
                        "source_content_sha256": source_snapshot.content_sha256,
                        "license_content_sha256": license_snapshot.content_sha256,
                        "direct_download_url_sha256": url_identity(verified.direct_download_url),
                        "verification_identity": verified.verification_identity,
                        "automation_terms_identity": automation_terms.decision_identity,
                        "promotion_ready": True,
                        "extraction_count": 0,
                        "decode_count": 0,
                        "import_count": 0,
                        "candidate_count": 0,
                        "paths_exposed": False,
                    }
                    result["result_identity"] = _identity(result)
                    if cancelled():
                        raise CatalogProbeCancelled()
                    with self._mutation_guard():
                        # The mutation-lock wait can outlast cancellation or the
                        # whole-probe deadline; recheck before terminal publication.
                        if cancelled():
                            raise CatalogProbeCancelled()
                        committed_state = self._read_state(run_dir, run_anchor)
                        if (
                            committed_state.get("lease_token") != lease_token
                            or committed_state["status"] not in ACTIVE_PROBE_STATUSES
                        ):
                            raise CatalogProbeError(
                                "invalid_catalog_probe_lease",
                                "Catalog probe ownership changed before its terminal commit.",
                            )
                        if committed_state["status"] == "CANCELLING":
                            raise CatalogProbeCancelled()
                        self._write_exclusive_json(run_dir / "probe_receipt.json", receipt, run_anchor)
                        self._write_exclusive_json(run_dir / "result.json", result, run_anchor)
                        ready_state = self._transition(
                            run_dir,
                            run_anchor,
                            "READY",
                            "ready_for_promotion",
                            "Probe completed in quarantine with zero extraction, decoding, candidates, or import.",
                            ended=True,
                        )
                        # READY becomes visible only through this terminal commit
                        # binding the request, receipt, result, and final state.
                        terminal = {
                            "schema_version": PROBE_TERMINAL_COMMIT_SCHEMA,
                            "probe_id": probe_id,
                            "committed_status": "READY",
                            "committed_at": _utc_now(),
                            "request_identity": _identity(self._read_request(run_dir, run_anchor)),
                            "receipt_identity": _identity(receipt),
                            "result_identity": _identity(result),
                            "state_identity": _identity(ready_state),
                            "raw_response_sha256": raw_receipt.response_sha256,
                            "paths_exposed": False,
                        }
                        terminal["terminal_commit_identity"] = _identity(terminal)
                        self._write_exclusive_json(run_dir / "terminal_commit.json", terminal, run_anchor)
        except (CatalogProbeCancelled, DownloadCancelled):
            if run_anchor is not None:
                self._safe_terminal_transition(
                    run_dir, run_anchor, "CANCELLED", "cancelled", "Catalog probe cancelled."
                )
        except (EvidenceFetchError, DownloadSecurityError, CatalogProbeError, OSError, ValueError) as exc:
            if run_anchor is not None:
                message = _safe_failure_message(exc)
                self._safe_terminal_transition(run_dir, run_anchor, "FAILED", "failed", message)
        finally:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
            if run_anchor is not None:
                self._release_lease(run_dir, run_anchor)
                run_anchor.__exit__(None, None, None)
            with self._lock:
                self._workers.pop(probe_id, None)
                self._cancellations.pop(probe_id, None)
                self._heartbeat_stops.pop(probe_id, None)
                self._heartbeat_failures.pop(probe_id, None)

    def _reverify_promoted_source(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        request: Mapping[str, Any],
        receipt: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> HarvestSource:
        with ExitStack() as stack:
            evidence_anchor = stack.enter_context(run_anchor.open_directory("evidence"))
            robots_anchor = stack.enter_context(run_anchor.open_directory("robots"))
            quarantine_anchor = stack.enter_context(run_anchor.open_directory("quarantine"))
            source_bytes = read_snapshot_bytes(evidence_anchor, "source_page.bin", max_bytes=MAX_EVIDENCE_PAGE_BYTES)
            license_bytes = read_snapshot_bytes(evidence_anchor, "license_page.bin", max_bytes=MAX_EVIDENCE_PAGE_BYTES)
            terms_snapshot: FetchSnapshot | None = None
            terms_bytes: bytes | None = None
            if request.get("terms_evidence_url") is not None:
                terms_record = receipt.get("automation_terms")
                if not isinstance(terms_record, Mapping):
                    raise CatalogProbeError("invalid_catalog_probe_receipt", "Automation terms evidence is invalid.")
                terms_snapshot = _snapshot_from_record(terms_record.get("fetch"))
                terms_bytes = read_snapshot_bytes(
                    evidence_anchor,
                    "terms_page.bin",
                    max_bytes=MAX_EVIDENCE_PAGE_BYTES,
                )
            raw_bytes = read_snapshot_bytes(
                quarantine_anchor,
                "raw_payload.bin",
                max_bytes=self.limits.max_response_bytes,
            )
            source_snapshot = _snapshot_from_record(receipt["source_evidence"])
            license_snapshot = _snapshot_from_record(receipt["license_evidence"])
            if (
                hashlib.sha256(source_bytes).hexdigest() != source_snapshot.content_sha256
                or hashlib.sha256(license_bytes).hexdigest() != license_snapshot.content_sha256
                or (
                    terms_snapshot is not None
                    and (
                        terms_bytes is None or hashlib.sha256(terms_bytes).hexdigest() != terms_snapshot.content_sha256
                    )
                )
                or hashlib.sha256(raw_bytes).hexdigest() != result["raw_response_sha256"]
                or len(raw_bytes) != result["raw_response_bytes"]
            ):
                raise CatalogProbeError(
                    "catalog_probe_evidence_changed",
                    "Retained probe pages or raw quarantine bytes changed after verification.",
                )
            direct = recover_direct_link(
                source_bytes,
                mime_type=source_snapshot.mime_type,
                source_url=str(request["source_page"]),
                expected_url_sha256=str(request["direct_download_url_sha256"]),
            )
            requested_urls = tuple(
                value
                for value in (
                    str(request["source_page"]),
                    str(request["terms_evidence_url"]) if request.get("terms_evidence_url") is not None else None,
                    str(request["license_evidence_url"]),
                    direct,
                )
                if value is not None
            )
            expected_decisions: dict[str, list[dict[str, Any]]] = {}
            for record in receipt["robots_evidence"]:
                if not isinstance(record, Mapping):
                    raise CatalogProbeError("invalid_catalog_probe_receipt", "Robots evidence is invalid.")
                origin = str(record.get("origin", ""))
                fetch = _snapshot_from_record(record.get("fetch"))
                payload = read_snapshot_bytes(
                    robots_anchor,
                    Path(fetch.relative_file).name,
                    max_bytes=MAX_ROBOTS_BYTES,
                )
                snapshot = rebuild_robots_snapshot(origin, fetch, payload)
                if snapshot.identity != record.get("robots_identity"):
                    raise CatalogProbeError(
                        "catalog_probe_evidence_changed",
                        "Retained robots evidence changed after the probe.",
                    )
                expected_decisions[origin] = [dict(item) for item in record.get("decisions", [])]
            actual_decisions: dict[str, list[dict[str, Any]]] = {origin: [] for origin in expected_decisions}
            snapshots_by_origin: dict[str, RobotsSnapshot] = {}
            for record in receipt["robots_evidence"]:
                origin = str(record["origin"])
                fetch = _snapshot_from_record(record["fetch"])
                payload = read_snapshot_bytes(
                    robots_anchor,
                    Path(fetch.relative_file).name,
                    max_bytes=MAX_ROBOTS_BYTES,
                )
                snapshots_by_origin[origin] = rebuild_robots_snapshot(origin, fetch, payload)
            for requested_url in requested_urls:
                origin = url_origin(requested_url)
                snapshot = snapshots_by_origin.get(origin)
                if snapshot is None:
                    raise CatalogProbeError("catalog_probe_evidence_changed", "Robots origin evidence is missing.")
                actual_decisions[origin].append(snapshot.evaluate(requested_url).to_dict())
            if actual_decisions != expected_decisions:
                raise CatalogProbeError(
                    "catalog_probe_evidence_changed",
                    "Robots path decisions changed after the probe.",
                )
            verified = verify_evidence_pages(
                source_url=str(request["source_page"]),
                source_snapshot=source_snapshot,
                source_bytes=source_bytes,
                license_url=str(request["license_evidence_url"]),
                license_snapshot=license_snapshot,
                license_bytes=license_bytes,
                title=str(request["title"]),
                creator=str(request["creator"]),
                license_id=str(request["license_id"]),
                direct_download_url=direct,
            )
            if (
                verified.verification_identity != receipt.get("verification_identity")
                or verified.verification_identity != result.get("verification_identity")
                or receipt.get("same_pack_evidence_verified") is not True
                or receipt.get("zero_cost_evidence_verified") is not True
                or receipt.get("license_conflict_checked") is not True
                or receipt.get("source_pack_evidence_sha256")
                != hashlib.sha256(verified.source_pack_evidence_text.encode("utf-8")).hexdigest()
                or receipt.get("source_pack_evidence_text") != verified.source_pack_evidence_text
            ):
                raise CatalogProbeError(
                    "catalog_probe_evidence_changed",
                    "Retained source-pack, license, or zero-cost verification evidence changed.",
                )
            automation_terms = verify_automation_terms(
                source_url=str(request["source_page"]),
                source_bytes=source_bytes,
                source_mime_type=source_snapshot.mime_type,
                source_content_sha256=source_snapshot.content_sha256,
                terms_url=(str(request["terms_evidence_url"]) if request.get("terms_evidence_url") else None),
                terms_bytes=terms_bytes,
                terms_snapshot=terms_snapshot,
            )
            if automation_terms.decision == "BLOCK":
                raise CatalogProbeError(
                    "catalog_probe_terms_blocked",
                    "Source automation terms explicitly prohibit automated downloading.",
                )
            if automation_terms.to_dict(terms_snapshot) != receipt.get("automation_terms"):
                raise CatalogProbeError(
                    "catalog_probe_evidence_changed",
                    "Source automation terms evidence changed after the probe.",
                )
            verified_at = _parse_utc(str(receipt["verified_at"]))
            expires = verified_at + timedelta(days=30)
            if expires <= datetime.now(timezone.utc):
                raise CatalogProbeError("catalog_probe_stale", "Catalog probe evidence expired before promotion.")
            terms_fetch = terms_snapshot or source_snapshot
            terms_binding = CatalogAutomationTermsBinding(
                mode=automation_terms.mode,
                decision=automation_terms.decision,
                evidence_url=automation_terms.evidence_url,
                evidence_request_url_sha256=url_identity(automation_terms.evidence_url),
                evidence_final_url=terms_fetch.final_url,
                evidence_http_status=terms_fetch.http_status,
                evidence_content_sha256=automation_terms.content_sha256,
                matched_declaration=automation_terms.matched_declaration,
                limited_evidence=automation_terms.limited_evidence,
                decision_identity_sha256=automation_terms.decision_identity,
                verified_at=verified_at.isoformat().replace("+00:00", "Z"),
                expires_at=expires.isoformat().replace("+00:00", "Z"),
            )
            provisional = CatalogEvidenceBinding(
                verifier_id=CATALOG_EVIDENCE_VERIFIER_ID,
                verifier_code_identity_sha256=catalog_evidence_verifier_code_identity(),
                verified_at=verified_at.isoformat().replace("+00:00", "Z"),
                expires_at=expires.isoformat().replace("+00:00", "Z"),
                source_request_url_sha256=url_identity(str(request["source_page"])),
                source_final_url=source_snapshot.final_url,
                source_http_status=source_snapshot.http_status,
                source_content_sha256=source_snapshot.content_sha256,
                license_request_url_sha256=url_identity(str(request["license_evidence_url"])),
                license_final_url=license_snapshot.final_url,
                license_http_status=license_snapshot.http_status,
                license_content_sha256=license_snapshot.content_sha256,
                automation_terms=terms_binding,
                zero_cost_reviewed=True,
                zero_cost_verification_identity_sha256=str(receipt["verification_identity"]),
                zero_cost_evidence_text_sha256=str(receipt["source_pack_evidence_sha256"]),
                zero_cost_probe_receipt_identity_sha256=str(receipt["receipt_identity"]),
                attestation_identity_sha256="0" * 64,
            )
            binding = replace(
                provisional,
                attestation_identity_sha256=provisional.expected_attestation_identity,
            )
            return HarvestSource(
                source_id=str(request["source_id"]),
                title=str(request["title"]),
                creator=str(request["creator"]),
                source_page=str(request["source_page"]),
                license_id=str(request["license_id"]),
                license_evidence_url=str(request["license_evidence_url"]),
                license_evidence_text=verified.license_evidence_text,
                attribution_text=str(request["attribution_text"]),
                acquisition_reference=direct,
                allowed_download_hosts=(verified.direct_download_host,),
                expected_response_sha256=str(result["raw_response_sha256"]),
                evidence_binding=binding,
                taxonomy_hints=tuple(request["taxonomy_hints"]),
            )

    def _validate_capability_evidence(
        self,
        request: CatalogProbeInput,
        evidence: BackendCapabilityEvidence,
    ) -> None:
        if (
            not isinstance(evidence, BackendCapabilityEvidence)
            or evidence.identity != request.backend_capability_evidence_identity
        ):
            raise CatalogProbeError(
                "catalog_probe_capability_evidence_required",
                "Current independent capability evidence is required before any catalog probe request.",
            )
        gates = evidence.capabilities.to_dict().get("enforced_gates", {})
        required = {
            "bounded_evidence_fetch",
            "quarantine_hash_probe",
            "probe_no_decode_extract_import",
            "deterministic_evidence_verification",
            "transactional_catalog_promotion",
        }
        if not isinstance(gates, Mapping) or any(gates.get(name) is not True for name in required):
            raise CatalogProbeError(
                "catalog_probe_capability_evidence_required",
                "Independent capability evidence does not cover the complete onboarding boundary.",
            )

    def _validate_promotion_capability_evidence(
        self,
        request: Mapping[str, Any],
        evidence: BackendCapabilityEvidence,
    ) -> None:
        if (
            not isinstance(evidence, BackendCapabilityEvidence)
            or request.get("backend_capability_evidence_identity") != evidence.identity
            or evidence.to_dict().get("schema_version") != "spritelab.harvest.backend-capability-evidence.v4"
        ):
            raise CatalogProbeError(
                "catalog_capability_evidence_changed",
                "Independent Harvest capability evidence changed after the probe.",
            )
        capabilities = evidence.capabilities.to_dict()
        gates = capabilities.get("enforced_gates")
        if (
            capabilities.get("schema_version") != "spritelab.harvest.backend-capabilities.v4"
            or not isinstance(gates, Mapping)
            or not gates
            or any(value is not True for value in gates.values())
            or _parse_utc(evidence.expires_at) <= datetime.now(timezone.utc)
        ):
            raise CatalogProbeError(
                "catalog_probe_capability_evidence_required",
                "Current full-gate independent capability evidence is required for promotion.",
            )

    def _notify_catalog_refreshed(self, catalog: tuple[HarvestSource, ...] | None = None) -> bool:
        if self._catalog_refreshed is None:
            return True
        try:
            self._catalog_refreshed(catalog if catalog is not None else load_trusted_catalog(self.project_root))
        except Exception:
            # Catalog publication and its promotion receipt are already
            # durable. A live-view refresh is recoverable and must not make a
            # successful POST look failed; replay retries this callback.
            return False
        return True

    def _validate_receipt_records(
        self,
        probe_id: str,
        request: Mapping[str, Any],
        receipt: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        automation = receipt.get("automation_terms")
        if (
            receipt.get("schema_version") != PROBE_RECEIPT_SCHEMA
            or result.get("schema_version") != PROBE_RESULT_SCHEMA
            or receipt.get("probe_id") != probe_id
            or result.get("probe_id") != probe_id
            or receipt.get("request_fingerprint") != request.get("request_fingerprint")
            or receipt.get("receipt_identity")
            != _identity({key: value for key, value in receipt.items() if key != "receipt_identity"})
            or result.get("result_identity")
            != _identity({key: value for key, value in result.items() if key != "result_identity"})
            or any(
                receipt.get(name) != 0
                for name in ("extraction_count", "decode_count", "import_count", "candidate_count")
            )
            or not isinstance(automation, Mapping)
            or automation.get("decision") not in {"ALLOW", "NO_PROHIBITION_OBSERVED"}
            or automation.get("robots_permission_treated_as_terms_permission") is not False
        ):
            raise CatalogProbeError("invalid_catalog_probe_receipt", "Catalog probe receipt is invalid.")
        if receipt.get("verifier_code_identity_sha256") != catalog_evidence_verifier_code_identity():
            raise CatalogProbeError("catalog_probe_stale", "Catalog evidence verifier code changed after the probe.")

    def _validate_promotion_receipt(
        self,
        promotion: Mapping[str, Any],
        *,
        probe_id: str,
        request: Mapping[str, Any],
        probe_receipt: Mapping[str, Any],
        result: Mapping[str, Any],
        terminal_commit: Mapping[str, Any],
        catalog: tuple[HarvestSource, ...],
        capability_evidence: BackendCapabilityEvidence,
        reviewed_verification_identity: str | None,
        reviewed_source_pack_evidence_sha256: str | None,
    ) -> None:
        source_id = str(request.get("source_id", ""))
        self._validate_receipt_records(probe_id, request, probe_receipt, result)
        source = next((item for item in catalog if item.source_id == source_id), None)
        terminal_identity = _identity(terminal_commit)
        promotion_identity = promotion.get("promotion_identity")
        if (
            set(promotion) != _PROMOTION_RECEIPT_KEYS
            or promotion.get("schema_version") != PROBE_PROMOTION_RECEIPT_SCHEMA
            or promotion.get("probe_id") != probe_id
            or promotion.get("source_id") != source_id
            or promotion.get("explicit_action") is not True
            or type(promotion.get("catalog_changed")) is not bool
            or promotion.get("private_direct_url_exposed") is not False
            or promotion.get("paths_exposed") is not False
            or promotion.get("zero_cost_evidence_reviewed") is not True
            or not isinstance(promotion.get("promoted_at"), str)
            or promotion_identity
            != _identity({key: value for key, value in promotion.items() if key != "promotion_identity"})
            or source is None
            or promotion.get("catalog_identity") != _catalog_identity_from_sources(catalog)
            or promotion.get("source_catalog_identity") != source.catalog_identity
            or promotion.get("raw_response_sha256") != result.get("raw_response_sha256")
            or promotion.get("backend_capability_evidence_identity") != capability_evidence.identity
            or promotion.get("verifier_id") != CATALOG_EVIDENCE_VERIFIER_ID
            or promotion.get("verifier_code_identity_sha256") != catalog_evidence_verifier_code_identity()
            or promotion.get("probe_receipt_identity") != _identity(probe_receipt)
            or promotion.get("result_identity") != _identity(result)
            or promotion.get("terminal_commit_identity") != terminal_identity
            or terminal_commit.get("schema_version") != PROBE_TERMINAL_COMMIT_SCHEMA
            or terminal_commit.get("probe_id") != probe_id
            or terminal_commit.get("committed_status") != "READY"
            or terminal_commit.get("request_identity") != _identity(request)
            or terminal_commit.get("receipt_identity") != _identity(probe_receipt)
            or terminal_commit.get("result_identity") != _identity(result)
            or terminal_commit.get("raw_response_sha256") != result.get("raw_response_sha256")
            or terminal_commit.get("paths_exposed") is not False
            or terminal_commit.get("terminal_commit_identity")
            != _identity({key: value for key, value in terminal_commit.items() if key != "terminal_commit_identity"})
            or source.evidence_binding.zero_cost_verification_identity_sha256
            != promotion.get("reviewed_verification_identity")
            or source.evidence_binding.zero_cost_evidence_text_sha256
            != promotion.get("reviewed_source_pack_evidence_sha256")
            or source.evidence_binding.zero_cost_probe_receipt_identity_sha256 != probe_receipt.get("receipt_identity")
        ):
            raise CatalogProbeError("invalid_catalog_promotion", "Catalog promotion evidence is invalid.")
        try:
            _parse_utc(str(promotion["promoted_at"]))
        except ValueError as exc:
            raise CatalogProbeError("invalid_catalog_promotion", "Catalog promotion evidence is invalid.") from exc
        _validate_exact_zero_cost_review(
            reviewed_verification_identity=reviewed_verification_identity,
            reviewed_source_pack_evidence_sha256=reviewed_source_pack_evidence_sha256,
            expected_verification_identity=promotion.get("reviewed_verification_identity"),
            expected_source_pack_evidence_sha256=promotion.get("reviewed_source_pack_evidence_sha256"),
        )

    def _terminal_commit_valid(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        state: Mapping[str, Any],
    ) -> bool:
        try:
            request = self._read_request(run_dir, run_anchor)
            receipt = self._read_json(run_dir / "probe_receipt.json", run_anchor)
            result = self._read_json(run_dir / "result.json", run_anchor)
            terminal = self._read_json(run_dir / "terminal_commit.json", run_anchor)
            # The commit binds the READY state it published; after promotion
            # the promotion receipt carries the follow-on binding instead.
            state_bound = str(state.get("status", "")) != "READY" or terminal.get("state_identity") == _identity(
                dict(state)
            )
            return (
                terminal.get("schema_version") == PROBE_TERMINAL_COMMIT_SCHEMA
                and terminal.get("probe_id") == run_dir.name
                and terminal.get("committed_status") == "READY"
                and terminal.get("request_identity") == _identity(request)
                and terminal.get("receipt_identity") == _identity(receipt)
                and terminal.get("result_identity") == _identity(result)
                and SHA256_PATTERN.fullmatch(str(terminal.get("state_identity", ""))) is not None
                and state_bound
                and terminal.get("raw_response_sha256") == result.get("raw_response_sha256")
                and terminal.get("terminal_commit_identity")
                == _identity({key: value for key, value in terminal.items() if key != "terminal_commit_identity"})
            )
        except (CatalogProbeError, OSError, ValueError):
            return False

    def _progress_callback(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        cancellation: threading.Event,
        stage: str,
    ) -> Callable[[int, int | None], None]:
        observation = {"current": -1, "at": 0.0}

        def update(current: int, total: int | None) -> None:
            if cancellation.is_set():
                raise CatalogProbeCancelled()
            now = time.monotonic()
            if current != total and current - observation["current"] < (1 << 20) and now - observation["at"] < 0.25:
                return
            observation.update({"current": current, "at": now})
            with self._mutation_guard():
                # Cancellation may have landed during the lock wait; recheck
                # local and durable evidence before publishing progress.
                if cancellation.is_set() or run_anchor.lexists("cancellation_request.json"):
                    raise CatalogProbeCancelled()
                state = self._read_state(run_dir, run_anchor)
                if state["status"] not in ACTIVE_PROBE_STATUSES:
                    raise CatalogProbeCancelled()
                self._transition(
                    run_dir,
                    run_anchor,
                    state["status"],
                    stage,
                    f"Bounded {stage.replace('_', ' ')} transfer is in progress.",
                    current=current,
                    total=total,
                )

        return update

    def _start_heartbeat(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        run_anchor: AnchoredDirectory,
    ) -> tuple[threading.Event, threading.Event, threading.Thread]:
        stop = threading.Event()
        failed = threading.Event()
        duplicate = run_anchor.detached_duplicate()
        token = str(state["lease_token"])

        def heartbeat() -> None:
            sequence = 0
            try:
                while not stop.wait(_HEARTBEAT_SECONDS):
                    sequence += 1
                    with self._mutation_guard():
                        current = self._read_state(run_dir, duplicate)
                        if current["status"] in TERMINAL_PROBE_STATUSES:
                            return
                        if current.get("lease_token") != token:
                            raise CatalogProbeError("invalid_catalog_probe_lease", "Catalog probe lease changed.")
                        self._write_atomic_json(
                            run_dir / "worker_lease.json",
                            self._new_lease(run_dir.name, token, sequence),
                            duplicate,
                        )
            except Exception:
                failed.set()
            finally:
                duplicate.__exit__(None, None, None)

        thread = threading.Thread(target=heartbeat, name=f"spritelab-{run_dir.name}-lease", daemon=True)
        with self._lock:
            self._heartbeat_stops[run_dir.name] = stop
            self._heartbeat_failures[run_dir.name] = failed
        thread.start()
        return stop, failed, thread

    def _release_lease(self, run_dir: Path, run_anchor: AnchoredDirectory) -> None:
        try:
            with self._mutation_guard():
                lease = self._read_json(run_dir / "worker_lease.json", run_anchor)
                now = _utc_now()
                lease.update({"released_at": now, "expires_at": now})
                self._write_atomic_json(run_dir / "worker_lease.json", lease, run_anchor)
        except Exception:
            return

    def _new_lease(self, probe_id: str, token: str, sequence: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "schema_version": PROBE_LEASE_SCHEMA,
            "probe_id": probe_id,
            "owner_pid": os.getpid(),
            "owner_instance_id": self._instance_id,
            "lease_token": token,
            "heartbeat_sequence": sequence,
            "heartbeat_at": now.isoformat().replace("+00:00", "Z"),
            "expires_at": (now + timedelta(seconds=_LEASE_SECONDS)).isoformat().replace("+00:00", "Z"),
            "released_at": None,
            "paths_exposed": False,
        }

    def _lease_is_live(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        run_anchor: AnchoredDirectory,
    ) -> bool:
        try:
            lease = self._read_json(run_dir / "worker_lease.json", run_anchor)
            return (
                lease.get("schema_version") == PROBE_LEASE_SCHEMA
                and lease.get("probe_id") == run_dir.name
                and lease.get("lease_token") == state.get("lease_token")
                and lease.get("owner_instance_id") == state.get("owner_instance_id")
                and lease.get("released_at") is None
                and _parse_utc(str(lease["expires_at"])) > datetime.now(timezone.utc)
            )
        except (CatalogProbeError, OSError, ValueError, KeyError):
            return False

    def _safe_terminal_transition(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        status: str,
        stage: str,
        message: str,
    ) -> None:
        try:
            with self._mutation_guard():
                self._transition(run_dir, run_anchor, status, stage, message, ended=True)
        except Exception:
            return

    def _guarded_transition(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        status: str,
        stage: str,
        message: str,
        *,
        current: int = 0,
        total: int | None = None,
        ended: bool = False,
        abort_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        with self._mutation_guard():
            # Cancellation or the deadline may have arrived during the lock
            # wait; recheck before publishing this state.
            if abort_check is not None and abort_check():
                raise CatalogProbeCancelled()
            return self._transition(
                run_dir,
                run_anchor,
                status,
                stage,
                message,
                current=current,
                total=total,
                ended=ended,
            )

    def _transition(
        self,
        run_dir: Path,
        run_anchor: AnchoredDirectory,
        status: str,
        stage: str,
        message: str,
        *,
        current: int = 0,
        total: int | None = None,
        ended: bool = False,
    ) -> dict[str, Any]:
        state = self._read_state(run_dir, run_anchor)
        if status not in ACTIVE_PROBE_STATUSES | TERMINAL_PROBE_STATUSES:
            raise CatalogProbeError("invalid_catalog_probe_state", "Catalog probe status transition is invalid.")
        if state["status"] in TERMINAL_PROBE_STATUSES and state["status"] != status:
            if not (state["status"] == "READY" and status == "PROMOTED"):
                raise CatalogProbeError("invalid_catalog_probe_state", "Catalog probe terminal state is immutable.")
        if state["status"] == "CANCELLING" and status in {"READY", "PROMOTED"}:
            raise CatalogProbeError(
                "invalid_catalog_probe_state",
                "A cancelling catalog probe can never commit a ready or promoted state.",
            )
        count = int(state.get("event_count", 0)) + 1
        if count > _MAX_EVENTS:
            raise CatalogProbeError("catalog_probe_event_limit", "Catalog probe event cap was reached.")
        now = _utc_now()
        updated = {
            **state,
            "status": status,
            "stage": stage,
            "message": message,
            "current": current,
            "total": total,
            "event_count": count,
            "updated_at": now,
            "ended_at": now if ended else None,
        }
        self._write_atomic_json(run_dir / "state.json", updated, run_anchor)
        self._append_event(run_dir, updated, run_anchor)
        return updated

    def _read_request(self, run_dir: Path, run_anchor: AnchoredDirectory) -> dict[str, Any]:
        request = self._read_json(run_dir / "request.json", run_anchor)
        if (
            request.get("schema_version") != PROBE_REQUEST_SCHEMA
            or request.get("probe_id") != run_dir.name
            or SOURCE_ID_PATTERN.fullmatch(str(request.get("source_id", ""))) is None
            or SHA256_PATTERN.fullmatch(str(request.get("request_fingerprint", ""))) is None
            or request.get("private_direct_url_exposed") is not False
        ):
            raise CatalogProbeError("invalid_catalog_probe_request", "Catalog probe request evidence is invalid.")
        return request

    def _read_state(self, run_dir: Path, run_anchor: AnchoredDirectory) -> dict[str, Any]:
        state = self._read_json(run_dir / "state.json", run_anchor)
        if (
            state.get("schema_version") != PROBE_STATE_SCHEMA
            or state.get("probe_id") != run_dir.name
            or state.get("status") not in ACTIVE_PROBE_STATUSES | TERMINAL_PROBE_STATUSES
        ):
            raise CatalogProbeError("invalid_catalog_probe_state", "Catalog probe state evidence is invalid.")
        return state

    def _read_json(self, path: Path, parent_anchor: AnchoredDirectory) -> dict[str, Any]:
        try:
            payload = read_stable_single_link_bytes(
                path,
                self.output_root,
                max_bytes=_MAX_METADATA_BYTES,
                parent_anchor=parent_anchor,
            )
            parsed = strict_json_loads(payload)
        except FileNotFoundError as exc:
            raise CatalogProbeError("missing_catalog_probe_evidence", "Catalog probe evidence is incomplete.") from exc
        except (OSError, ValueError, HarvestStorageError, UnsafeFilesystemOperation) as exc:
            raise CatalogProbeError("unsafe_catalog_probe_evidence", "Catalog probe evidence is unsafe.") from exc
        if not isinstance(parsed, Mapping):
            raise CatalogProbeError("invalid_catalog_probe_evidence", "Catalog probe evidence is invalid.")
        return dict(parsed)

    def _read_optional_json(self, path: Path, parent_anchor: AnchoredDirectory) -> dict[str, Any] | None:
        if not parent_anchor.lexists(path.name):
            return None
        return self._read_json(path, parent_anchor)

    def _write_exclusive_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
        parent_anchor: AnchoredDirectory,
    ) -> None:
        encoded = (strict_json_dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            write_exclusive_stable_bytes(
                path,
                self.output_root,
                encoded,
                max_bytes=_MAX_METADATA_BYTES,
                parent_anchor=parent_anchor,
            )
        except (OSError, HarvestStorageError, UnsafeFilesystemOperation) as exc:
            raise CatalogProbeError("unsafe_catalog_probe_evidence", "Catalog probe publication failed.") from exc

    def _write_atomic_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
        parent_anchor: AnchoredDirectory,
    ) -> None:
        encoded = (strict_json_dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            write_atomic_stable_bytes(
                path,
                self.output_root,
                encoded,
                max_bytes=_MAX_METADATA_BYTES,
                parent_anchor=parent_anchor,
            )
        except (OSError, HarvestStorageError, UnsafeFilesystemOperation) as exc:
            raise CatalogProbeError("unsafe_catalog_probe_evidence", "Catalog probe publication failed.") from exc

    def _append_event(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        run_anchor: AnchoredDirectory,
    ) -> None:
        event = {
            "schema_version": PROBE_EVENT_SCHEMA,
            "probe_id": run_dir.name,
            "sequence": state["event_count"],
            "timestamp": state["updated_at"],
            "status": state["status"],
            "stage": state["stage"],
            "current": state.get("current", 0),
            "total": state.get("total"),
            "message": state.get("message", ""),
            "paths_exposed": False,
        }
        line = (strict_json_dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        if len(line) > _MAX_EVENT_BYTES:
            raise CatalogProbeError("catalog_probe_event_limit", "Catalog probe event exceeded its byte cap.")
        try:
            append_stable_single_link_bytes(
                run_dir / "events.jsonl",
                run_dir,
                line,
                max_bytes=_MAX_EVENT_BYTES,
                max_total_bytes=_MAX_EVENT_BYTES * _MAX_EVENTS,
                parent_anchor=run_anchor,
            )
        except HarvestStorageError as exc:
            raise CatalogProbeError("unsafe_catalog_probe_evidence", "Catalog probe event stream is unsafe.") from exc

    def _read_events(self, run_dir: Path, run_anchor: AnchoredDirectory) -> list[dict[str, Any]]:
        if not run_anchor.lexists("events.jsonl"):
            return []
        try:
            content = read_stable_single_link_bytes(
                run_dir / "events.jsonl",
                self.output_root,
                max_bytes=_MAX_EVENT_BYTES * _MAX_EVENTS,
                parent_anchor=run_anchor,
            )
            events: list[dict[str, Any]] = []
            for line in content.decode("utf-8").splitlines()[-200:]:
                value = strict_json_loads(line)
                if isinstance(value, Mapping) and value.get("schema_version") == PROBE_EVENT_SCHEMA:
                    events.append(dict(value))
            return events
        except (OSError, UnicodeError, ValueError, HarvestStorageError):
            return []

    def _find_idempotency(self, key: str) -> tuple[str, str] | None:
        with AnchoredDirectory(self.output_root, self.project_root) as output_anchor:
            for name in output_anchor.names():
                if PROBE_ID_PATTERN.fullmatch(name) is None:
                    continue
                try:
                    with output_anchor.open_directory_immovable(name) as run_anchor:
                        request = self._read_request(self.output_root / name, run_anchor)
                except (CatalogProbeError, OSError, UnsafeFilesystemOperation) as exc:
                    raise CatalogProbeError(
                        "unsafe_catalog_probe_evidence",
                        "A catalog probe could not be safely checked for idempotent reuse.",
                    ) from exc
                if request.get("idempotency_key") == key:
                    return name, str(request.get("request_fingerprint", ""))
        return None

    def _active_single_flight(self) -> str | None:
        """Return one live ordinary acquisition or probe under the held project lock."""

        with AnchoredDirectory(self.output_root, self.project_root) as output_anchor:
            for name in output_anchor.names():
                if name == ".harvest.lock":
                    continue
                try:
                    metadata = output_anchor.lstat(name)
                    if not stat.S_ISDIR(metadata.st_mode) or _link_or_reparse(metadata):
                        if PROBE_ID_PATTERN.fullmatch(name) or name.startswith("harvest-"):
                            return name
                        continue
                    with output_anchor.open_directory_immovable(name) as run_anchor:
                        if PROBE_ID_PATTERN.fullmatch(name):
                            record = scan_probe_inventory_record(
                                self.output_root / name,
                                self.output_root,
                                run_anchor=run_anchor,
                            )
                            if record is None or record.get("status") in ACTIVE_PROBE_STATUSES:
                                return name
                            continue
                        if not name.startswith("harvest-"):
                            continue
                        state = _read_inventory_json(
                            self.output_root / name / "state.json",
                            self.output_root,
                            run_anchor,
                        )
                        if state.get("status") not in {"QUEUED", "RUNNING", "CANCELLING"}:
                            continue
                        lease = _read_inventory_json(
                            self.output_root / name / "worker_lease.json",
                            self.output_root,
                            run_anchor,
                        )
                        if _generic_lease_live(state, lease):
                            return name
                except (OSError, ValueError, HarvestStorageError, UnsafeFilesystemOperation):
                    if PROBE_ID_PATTERN.fullmatch(name) or name.startswith("harvest-"):
                        return name
                    continue
        return None

    def _allocate_probe_directory(self) -> str:
        with AnchoredDirectory(self.output_root, self.project_root) as output_anchor:
            for _attempt in range(32):
                probe_id = self._run_id_factory()
                if PROBE_ID_PATTERN.fullmatch(probe_id) is None:
                    raise CatalogProbeError("invalid_catalog_probe_id", "Generated catalog probe identifier is unsafe.")
                try:
                    output_anchor.mkdir(probe_id)
                except FileExistsError:
                    continue
                return probe_id
        raise CatalogProbeError("catalog_probe_collision", "Could not allocate a unique catalog probe.")

    def _probe_directory(self, probe_id: str) -> Path:
        if PROBE_ID_PATTERN.fullmatch(probe_id or "") is None:
            raise CatalogProbeError("catalog_probe_not_found", "Catalog probe was not found.", status_code=404)
        path = require_confined_path(self.output_root / probe_id, self.output_root)
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise CatalogProbeError("catalog_probe_not_found", "Catalog probe was not found.", status_code=404) from exc
        if not stat.S_ISDIR(metadata.st_mode) or _link_or_reparse(metadata) or path.is_mount():
            raise CatalogProbeError("catalog_probe_not_found", "Catalog probe was not found.", status_code=404)
        return path

    @contextmanager
    def _open_probe_anchor(self, probe_id: str) -> Any:
        self._probe_directory(probe_id)
        with AnchoredDirectory(self.output_root, self.project_root) as output_anchor:
            with output_anchor.open_directory_immovable(probe_id) as run_anchor:
                yield run_anchor

    @contextmanager
    def _mutation_guard(self) -> Any:
        with self._lock:
            self._ensure_output_root()
            try:
                with RepositoryMutationLock(self.project_root):
                    yield
            except HarvestStorageError as exc:
                raise CatalogProbeError("harvest_mutation_conflict", str(exc)) from exc

    def _ensure_output_root(self) -> None:
        try:
            with AnchoredDirectory(self.project_root, self.project_root) as project_anchor:
                project_anchor.mkdir("harvest_runs", exist_ok=True)
            metadata = self.output_root.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or _link_or_reparse(metadata) or self.output_root.is_mount():
                raise CatalogProbeError("unsafe_harvest_root", "Managed Harvest root is unsafe.")
        except (OSError, UnsafeFilesystemOperation) as exc:
            raise CatalogProbeError("unsafe_harvest_root", "Managed Harvest root could not be safely created.") from exc


def scan_probe_inventory_record(
    run_dir: Path,
    output_root: Path,
    *,
    run_anchor: AnchoredDirectory,
) -> dict[str, Any] | None:
    """Passively recognize one immediate durable probe directory."""

    if PROBE_ID_PATTERN.fullmatch(run_dir.name) is None:
        return None
    try:
        request = _read_inventory_json(run_dir / "request.json", output_root, run_anchor)
        state = _read_inventory_json(run_dir / "state.json", output_root, run_anchor)
        if (
            request.get("schema_version") != PROBE_REQUEST_SCHEMA
            or state.get("schema_version") != PROBE_STATE_SCHEMA
            or request.get("probe_id") != run_dir.name
            or state.get("probe_id") != run_dir.name
        ):
            return None
        status = str(state.get("status", ""))
        if status in ACTIVE_PROBE_STATUSES:
            try:
                lease = _read_inventory_json(run_dir / "worker_lease.json", output_root, run_anchor)
                live = (
                    lease.get("schema_version") == PROBE_LEASE_SCHEMA
                    and lease.get("lease_token") == state.get("lease_token")
                    and lease.get("released_at") is None
                    and _parse_utc(str(lease["expires_at"])) > datetime.now(timezone.utc)
                )
            except Exception:
                live = False
            if not live:
                status = "INTERRUPTED"
        if status in {"READY", "PROMOTED"} and not run_anchor.lexists("terminal_commit.json"):
            # A ready/promoted state without its terminal commit is not
            # terminal success evidence.
            status = "INTERRUPTED"
        result = (
            _read_inventory_json(run_dir / "result.json", output_root, run_anchor)
            if run_anchor.lexists("result.json")
            else {}
        )
        return {
            "probe_id": run_dir.name,
            "source_id": request.get("source_id"),
            "title": request.get("title"),
            "status": status,
            "stage": "interrupted" if status == "INTERRUPTED" else state.get("stage"),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "raw_response_sha256": result.get("raw_response_sha256"),
            "promotion_ready": status == "READY",
            "promoted": status == "PROMOTED",
            "paths_exposed": False,
        }
    except (OSError, ValueError, HarvestStorageError, UnsafeFilesystemOperation):
        return None


def _read_inventory_json(path: Path, root: Path, anchor: AnchoredDirectory) -> dict[str, Any]:
    payload = read_stable_single_link_bytes(path, root, max_bytes=_MAX_METADATA_BYTES, parent_anchor=anchor)
    parsed = strict_json_loads(payload)
    if not isinstance(parsed, Mapping):
        raise ValueError("invalid probe inventory record")
    return dict(parsed)


def _snapshot_from_record(value: Any) -> FetchSnapshot:
    if not isinstance(value, Mapping):
        raise CatalogProbeError("invalid_catalog_probe_receipt", "Catalog fetch evidence is invalid.")
    required = {
        "request_url_sha256",
        "request_public_url",
        "final_url",
        "http_status",
        "mime_type",
        "byte_count",
        "content_sha256",
        "elapsed_seconds",
        "relative_file",
        "redirect_count",
    }
    if set(value) != required or value.get("redirect_count") != 0:
        raise CatalogProbeError("invalid_catalog_probe_receipt", "Catalog fetch evidence is invalid.")
    try:
        return FetchSnapshot(
            request_url_sha256=str(value["request_url_sha256"]),
            request_public_url=str(value["request_public_url"]),
            final_url=str(value["final_url"]),
            http_status=int(value["http_status"]),
            mime_type=str(value["mime_type"]),
            byte_count=int(value["byte_count"]),
            content_sha256=str(value["content_sha256"]),
            elapsed_seconds=float(value["elapsed_seconds"]),
            relative_file=str(value["relative_file"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CatalogProbeError("invalid_catalog_probe_receipt", "Catalog fetch evidence is invalid.") from exc


def _capability_record(evidence: BackendCapabilityEvidence) -> dict[str, Any]:
    return {
        **evidence.to_dict(),
        "evidence_identity": evidence.identity,
        "capabilities": evidence.capabilities.to_dict(),
    }


def _canonical_evidence_url(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CatalogProbeError("invalid_catalog_probe", f"Catalog {label} URL is invalid.", status_code=422)
    try:
        return canonical_url_string(value, allow_query=False)
    except ValueError as exc:
        raise CatalogProbeError(
            "invalid_catalog_probe_url",
            f"Catalog {label} URL must be public canonical HTTPS without credentials, query, or fragment.",
            status_code=422,
        ) from exc


def _canonical_direct_url(value: Any) -> str:
    if not isinstance(value, str):
        raise CatalogProbeError("invalid_catalog_probe_url", "Direct download URL is invalid.", status_code=422)
    try:
        return canonical_url_string(value, allow_query=True)
    except ValueError as exc:
        raise CatalogProbeError(
            "invalid_catalog_probe_url",
            "Direct download URL must be a public HTTPS URL without credentials or fragment.",
            status_code=422,
        ) from exc


def _download_host(value: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(value).hostname or "").casefold().rstrip(".")


def _safe_failure_message(exc: BaseException) -> str:
    if isinstance(exc, EvidenceFetchError):
        message = str(exc)
        if message and len(message) <= 240 and "http" not in message.casefold():
            return message
    if isinstance(exc, CatalogProbeError):
        return str(exc)[:240]
    return "Catalog probe failed closed at a bounded evidence, robots, or quarantine boundary."


def _catalog_identity_from_sources(sources: tuple[HarvestSource, ...]) -> str:
    from spritelab.product_features.harvest.catalog import trusted_catalog_identity

    return trusted_catalog_identity(sources)


def _recorded_network_actions(receipt: Mapping[str, Any] | None) -> int:
    if receipt is None:
        return 0
    robots = receipt.get("robots_evidence")
    terms = receipt.get("automation_terms")
    terms_fetch = terms.get("fetch") if isinstance(terms, Mapping) else None
    return 3 + int(terms_fetch is not None) + len(robots) if isinstance(robots, list) else 0


def _generic_lease_live(state: Mapping[str, Any], lease: Mapping[str, Any]) -> bool:
    try:
        return (
            lease.get("lease_token") == state.get("lease_token")
            and lease.get("owner_instance_id") == state.get("owner_instance_id")
            and lease.get("released_at") is None
            and _parse_utc(str(lease["expires_at"])) > datetime.now(timezone.utc)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _validate_exact_zero_cost_review(
    *,
    reviewed_verification_identity: str | None,
    reviewed_source_pack_evidence_sha256: str | None,
    expected_verification_identity: Any,
    expected_source_pack_evidence_sha256: Any,
) -> None:
    submitted_identity = str(reviewed_verification_identity or "")
    submitted_evidence_sha256 = str(reviewed_source_pack_evidence_sha256 or "")
    if (
        SHA256_PATTERN.fullmatch(submitted_identity) is None
        or SHA256_PATTERN.fullmatch(submitted_evidence_sha256) is None
        or submitted_identity != expected_verification_identity
        or submitted_evidence_sha256 != expected_source_pack_evidence_sha256
    ):
        raise CatalogProbeError(
            "catalog_zero_cost_evidence_review_required",
            "Promotion requires review of the exact unchanged retained zero-cost evidence identity and text hash.",
            status_code=409,
        )


def _identity(value: Any) -> str:
    encoded = strict_json_dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def _link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


__all__ = [
    "ACTIVE_PROBE_STATUSES",
    "PROBE_EVENT_SCHEMA",
    "PROBE_ID_PATTERN",
    "PROBE_PROMOTION_RECEIPT_SCHEMA",
    "PROBE_RECEIPT_SCHEMA",
    "PROBE_REQUEST_SCHEMA",
    "PROBE_RESULT_SCHEMA",
    "PROBE_STATE_SCHEMA",
    "PROBE_TERMINAL_COMMIT_SCHEMA",
    "RETRYABLE_PROBE_STATUSES",
    "TERMINAL_PROBE_STATUSES",
    "CatalogProbeError",
    "CatalogProbeInput",
    "CatalogProbeService",
    "scan_probe_inventory_record",
]
