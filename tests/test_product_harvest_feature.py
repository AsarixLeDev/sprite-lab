from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_features.harvest import create_plugin
from spritelab.product_features.harvest.catalog import (
    CatalogEvidenceBinding,
    HarvestSource,
    url_identity,
)
from spritelab.product_features.harvest.service import HarvestError, HarvestService
from spritelab.product_features.harvest.storage import HarvestStorageError, RepositoryMutationLock
from spritelab.product_features.harvest.trusted_backend import (
    AcquiredFile,
    AcquisitionReceipt,
    AcquisitionResult,
    CertifiedBackendCapabilities,
    DatasetImportRequest,
    DatasetImportResult,
    HarvestLimits,
)
from spritelab.product_web.app import create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"sprite-payload"
RESPONSE = b"certified-archive-response"
SOURCE_URL = "https://catalog.example.test/source"
LICENSE_URL = "https://catalog.example.test/license"
DOWNLOAD_URL = "https://downloads.example.test/archive.zip?token=private"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _binding(
    *,
    verified_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> CatalogEvidenceBinding:
    now = datetime.now(timezone.utc)
    verified = verified_at or now - timedelta(days=1)
    expires = expires_at or now + timedelta(days=7)
    provisional = CatalogEvidenceBinding(
        verifier_id="catalog.verifier",
        verifier_code_identity_sha256=SHA_A,
        verified_at=verified.isoformat().replace("+00:00", "Z"),
        expires_at=expires.isoformat().replace("+00:00", "Z"),
        source_request_url_sha256=url_identity(SOURCE_URL),
        source_final_url=SOURCE_URL,
        source_http_status=200,
        source_content_sha256=hashlib.sha256(b"source-page").hexdigest(),
        license_request_url_sha256=url_identity(LICENSE_URL),
        license_final_url=LICENSE_URL,
        license_http_status=200,
        license_content_sha256=hashlib.sha256(b"license-page").hexdigest(),
        attestation_identity_sha256="0" * 64,
    )
    return replace(
        provisional,
        attestation_identity_sha256=provisional.expected_attestation_identity,
    )


def _source(*, binding: CatalogEvidenceBinding | None = None, source_id: str = "open.source") -> HarvestSource:
    return HarvestSource(
        source_id=source_id,
        title="Verified open sprites",
        creator="Example Artist",
        source_page=SOURCE_URL,
        license_id="cc0-1.0",
        license_evidence_url=LICENSE_URL,
        license_evidence_text="CC0 1.0 Universal public-domain dedication.",
        attribution_text="Example Artist — Verified open sprites",
        acquisition_reference=DOWNLOAD_URL,
        allowed_download_hosts=("downloads.example.test", "cdn.example.test"),
        expected_response_sha256=hashlib.sha256(RESPONSE).hexdigest(),
        evidence_binding=binding or _binding(),
        taxonomy_hints=("item",),
    )


def _capabilities(*, code_identity: str = SHA_A) -> CertifiedBackendCapabilities:
    return CertifiedBackendCapabilities(
        backend_id="fixture.backend",
        backend_version="1.0",
        downloader_id="fixture.downloader",
        downloader_version="1.0",
        code_identity_sha256=code_identity,
        enforces_http_success=True,
        enforces_https_direct_url=True,
        resolves_and_blocks_private_networks=True,
        validates_every_redirect=True,
        enforces_response_mime_allowlist=True,
        enforces_expected_response_hash=True,
        enforces_per_file_hashes=True,
        enforces_file_count_and_byte_limits=True,
        enforces_depth_and_name_policy=True,
        enforces_archive_limits=True,
        enforces_duration_and_cancellation=True,
    )


class FixtureBackend:
    def __init__(
        self,
        capabilities: CertifiedBackendCapabilities,
        calls: list[dict[str, Any]],
        *,
        mode: str = "good",
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.calls = calls
        self.mode = mode
        self.entered = entered
        self.release = release

    def acquire(
        self,
        source: HarvestSource,
        destination: Path,
        limits: HarvestLimits,
        *,
        cancel_requested: Any,
        progress: Any,
    ) -> AcquisitionResult:
        authorization = json.loads((destination.parent / "authorization_receipt.json").read_text(encoding="utf-8"))
        self.calls.append({"authorization": authorization, "private_reference": source.acquisition_reference})
        if self.entered:
            self.entered.set()
        if self.release:
            while not self.release.wait(0.005):
                if cancel_requested():
                    raise RuntimeError("private cancellation URL should not persist")
        if self.mode == "chatty":
            for index in range(20):
                progress("downloading", index, 20)
        else:
            progress("downloading", 1, 1)
        names = ["sprites.png"]
        if self.mode == "too_many":
            names.append("second.png")
        if self.mode == "unicode_name":
            names = ["e\u0301.png"]
        if self.mode == "deep":
            nested = destination / "one" / "two" / "three"
            nested.mkdir(parents=True)
            names = ["one/two/three/sprites.png"]
        files: list[AcquiredFile] = []
        for name in names:
            path = destination / Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(PNG)
            files.append(
                AcquiredFile(
                    name,
                    len(PNG),
                    hashlib.sha256(PNG).hexdigest(),
                    "image/png",
                    usable=self.mode != "quarantine",
                    quarantine_reason="policy_review" if self.mode == "quarantine" else None,
                    taxonomy=("item",),
                )
            )
        final_url = "https://127.0.0.1/private" if self.mode == "private_url" else DOWNLOAD_URL
        actual_hash = SHA_B if self.mode == "bad_hash" else source.expected_response_sha256
        mime = "text/html" if self.mode == "bad_mime" else "application/zip"
        return AcquisitionResult(
            AcquisitionReceipt(
                final_url=final_url,
                redirect_chain=("https://cdn.example.test/redirect",),
                http_status=200,
                response_mime_type=mime,
                expected_response_sha256=source.expected_response_sha256,
                actual_response_sha256=actual_hash,
                response_bytes=len(RESPONSE),
                elapsed_seconds=0.01,
                archive_members=len(files),
                archive_uncompressed_bytes=sum(item.byte_count for item in files),
                backend_capability_identity=self.capabilities.identity,
                files=tuple(files),
            )
        )


def _service(
    project: Path,
    *,
    mode: str = "good",
    limits: HarvestLimits | None = None,
    capabilities: CertifiedBackendCapabilities | None = None,
    calls: list[dict[str, Any]] | None = None,
    entered: threading.Event | None = None,
    release: threading.Event | None = None,
    callback: Any = None,
    service_type: type[HarvestService] = HarvestService,
) -> tuple[HarvestService, list[dict[str, Any]], CertifiedBackendCapabilities]:
    observed = calls if calls is not None else []
    certified = capabilities or _capabilities()
    return (
        service_type(
            project,
            sources=(_source(),),
            backend_capabilities=certified,
            backend_factory=lambda: FixtureBackend(
                certified,
                observed,
                mode=mode,
                entered=entered,
                release=release,
            ),
            limits=limits,
            dataset_import_callback=callback,
        ),
        observed,
        certified,
    )


def _start_arguments(service: HarvestService, key: str) -> dict[str, Any]:
    inventory = service.inventory()
    assessed = inventory["known_usable_items"]
    return {
        "idempotency_key": key,
        "explicit_action": True,
        "authorize_zero_cost": True,
        "authorize_permissive_license": True,
        "authorize_existing_inventory_reviewed": True,
        "reuse_evidence": {
            "decision": "reuse_exhausted" if assessed == 0 else "deficit_confirmed",
            "evidence_code": "no_reusable_items" if assessed == 0 else "target_deficit",
            "inventory_identity": inventory["inventory_identity"],
            "assessed_usable_items": assessed,
            "required_usable_items": assessed + 1,
            "deficit_items": 1,
        },
    }


def _wait(service: HarvestService, run_id: str, *statuses: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service.job(run_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Harvest run did not reach {statuses}")


def test_passive_inventory_indexes_legacy_runs_without_mutation_or_backend(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    legacy = project / "harvest_runs" / "kenney_generic_items"
    legacy.mkdir(parents=True)
    sources = b'{"source_id":"kenney"}\n'
    candidates = b'{"candidate_id":"one"}\n{"candidate_id":"two"}\n'
    (legacy / "sources.jsonl").write_bytes(sources)
    (legacy / "candidates.jsonl").write_bytes(candidates)
    constructed = 0

    def forbidden() -> FixtureBackend:
        nonlocal constructed
        constructed += 1
        raise AssertionError("passive inventory cannot construct a backend")

    service = HarvestService(
        project,
        sources=(_source(),),
        backend_factory=forbidden,
        backend_capabilities=_capabilities(),
    )
    inventory = service.inventory()
    assert inventory["legacy_run_count"] == 1
    assert inventory["legacy_candidate_records"] == 2
    assert inventory["legacy_runs"][0]["legacy_id"] == "kenney_generic_items"
    assert inventory["legacy_runs"][0]["mutation_allowed"] is False
    assert not (project / "harvest_runs" / ".harvest.lock").exists()
    assert (legacy / "sources.jsonl").read_bytes() == sources
    assert (legacy / "candidates.jsonl").read_bytes() == candidates
    assert constructed == 0


def test_real_product_shell_requires_csrf_and_js_supplies_it(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls: list[dict[str, Any]] = []
    capabilities = _capabilities()
    plugin = create_plugin(
        sources=(_source(),),
        backend_capabilities=capabilities,
        backend_factory=lambda: FixtureBackend(capabilities, calls),
    )
    app = create_app(ProjectContext(project), plugins=(plugin,))
    client = TestClient(app)
    page = client.get("/harvest")
    assert page.status_code == 200
    assert "Authorize a measured deficit" in page.text
    inventory = client.get("/harvest/api/inventory").json()
    payload = {
        "source_id": "open.source",
        "idempotency_key": "csrf-harvest-0001",
        "explicit_action": True,
        "authorize_zero_cost": True,
        "authorize_permissive_license": True,
        "authorize_existing_inventory_reviewed": True,
        "reuse_evidence": {
            "decision": "reuse_exhausted",
            "evidence_code": "no_reusable_items",
            "inventory_identity": inventory["inventory_identity"],
            "assessed_usable_items": 0,
            "required_usable_items": 1,
            "deficit_items": 1,
        },
    }
    denied = client.post("/harvest/api/jobs", json=payload)
    assert denied.status_code == 403
    assert denied.json()["error_code"] == "csrf_validation_failed"
    assert calls == []
    assert not (project / "harvest_runs").exists()

    allowed = client.post(
        "/harvest/api/jobs",
        json=payload,
        headers={"X-CSRF-Token": app.state.spritelab_csrf_token},
    )
    assert allowed.status_code == 202
    run_id = allowed.json()["job"]["run_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get(f"/harvest/api/jobs/{run_id}").json()
        if status["status"] == "COMPLETE":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("real-shell Harvest job did not complete")
    evidence = client.get(f"/harvest/api/jobs/{run_id}/evidence").json()
    assert evidence["acquisition_receipt"]["actual_response_sha256"] == hashlib.sha256(RESPONSE).hexdigest()
    javascript = client.get("/harvest/static/harvest.js").text
    assert '"X-CSRF-Token": csrf' in javascript
    assert "inFlight" in javascript


def test_receipts_bind_inventory_source_backend_limits_response_and_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, calls, capabilities = _service(project)
    started, created = service.start("open.source", **_start_arguments(service, "receipt-bind-0001"))
    assert created is True
    complete = _wait(service, started["run_id"], "COMPLETE")
    assert complete["usable_count"] == 1
    assert len(calls) == 1
    authorization = calls[0]["authorization"]
    assert authorization["network_actions_before_receipt"] == 0
    evidence_binding_identity = authorization["source"]["evidence_binding"]["binding_identity"]
    assert len(evidence_binding_identity) == 64
    assert authorization["backend_capabilities"]["capability_identity"] == capabilities.identity
    assert authorization["reuse_evidence"]["deficit_items"] == 1

    evidence = service.evidence(started["run_id"])
    acquisition = evidence["acquisition_receipt"]
    manifest = evidence["artifact_manifest"]
    assert acquisition["http_status"] == 200
    assert acquisition["expected_response_sha256"] == acquisition["actual_response_sha256"]
    assert acquisition["final_url_sha256"] == url_identity(DOWNLOAD_URL)
    assert acquisition["backend_capabilities"]["code_identity_sha256"] == SHA_A
    assert manifest["files"][0]["expected_sha256"] == manifest["files"][0]["actual_sha256"]
    assert manifest["files"][0]["relative_path"] == "sprites.png"
    handoff = service.handoff(started["run_id"])
    assert handoff["schema_version"] == "spritelab.harvest.dataset-handoff.v2"
    assert handoff["portable_relative_paths"] is True
    assert handoff["source_evidence_binding_identity"] == evidence_binding_identity
    assert handoff["files"] == manifest["files"]
    durable = "\n".join(
        path.read_text(encoding="utf-8") for path in (project / "harvest_runs" / started["run_id"]).glob("*.json*")
    )
    assert "token=private" not in durable
    assert str(project) not in durable


def test_reuse_evidence_must_match_exact_legacy_inventory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    legacy = project / "harvest_runs" / "historical_pack"
    legacy.mkdir(parents=True)
    (legacy / "sources.jsonl").write_text('{"source_id":"old"}\n', encoding="utf-8")
    (legacy / "candidates.jsonl").write_text('{"candidate_id":"old-1"}\n', encoding="utf-8")
    service, calls, _capability = _service(project)
    arguments = _start_arguments(service, "reuse-gate-0001")
    arguments["reuse_evidence"]["inventory_identity"] = SHA_B
    with pytest.raises(HarvestError) as changed:
        service.start("open.source", **arguments)
    assert changed.value.code == "harvest_inventory_changed"
    assert calls == []

    arguments = _start_arguments(service, "reuse-gate-0002")
    arguments["authorize_existing_inventory_reviewed"] = False
    with pytest.raises(HarvestError) as unreviewed:
        service.start("open.source", **arguments)
    assert unreviewed.value.code == "existing_inventory_review_required"
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verifier_id", "other.verifier"),
        ("verifier_code_identity_sha256", SHA_B),
        ("verified_at", "2026-07-15T00:00:00Z"),
        ("expires_at", "2026-07-25T00:00:00Z"),
        ("source_request_url_sha256", SHA_B),
        ("source_final_url", "https://catalog.example.test/changed-source"),
        ("source_http_status", 204),
        ("source_content_sha256", SHA_B),
        ("license_request_url_sha256", SHA_B),
        ("license_final_url", "https://catalog.example.test/changed-license"),
        ("license_http_status", 204),
        ("license_content_sha256", SHA_B),
        ("attestation_identity_sha256", SHA_B),
    ],
)
def test_every_catalog_evidence_field_is_attestation_bound(field: str, value: Any) -> None:
    changed = replace(_binding(), **{field: value})
    with pytest.raises(ValueError):
        _source(binding=changed)


def test_catalog_evidence_absence_expiry_long_lifetime_and_source_change_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires a certified"):
        replace(_source(), evidence_binding=None)  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)
    expired = _binding(verified_at=now - timedelta(days=5), expires_at=now - timedelta(days=1))
    with pytest.raises(ValueError, match="stale"):
        _source(binding=expired)
    long_lived = _binding(verified_at=now - timedelta(days=1), expires_at=now + timedelta(days=31))
    with pytest.raises(ValueError, match="stale"):
        _source(binding=long_lived)
    with pytest.raises(ValueError, match="source-page evidence binding changed"):
        replace(_source(), source_page="https://catalog.example.test/changed")


def test_catalog_rejects_unknown_creator_and_broader_license() -> None:
    with pytest.raises(ValueError, match="cannot be Unknown"):
        replace(_source(), creator="Unknown")
    with pytest.raises(ValueError, match="only CC0"):
        replace(_source(), license_id="cc-by-4.0")


@pytest.mark.parametrize("mode", ["private_url", "bad_hash", "bad_mime", "too_many", "unicode_name", "deep"])
def test_backend_or_artifact_policy_violations_fail_closed_and_redact(mode: str, tmp_path: Path) -> None:
    project = tmp_path / mode
    project.mkdir()
    limits = HarvestLimits(max_files=1, max_depth=2) if mode in {"too_many", "deep"} else HarvestLimits()
    service, _calls, _capability = _service(project, mode=mode, limits=limits)
    started, _created = service.start("open.source", **_start_arguments(service, f"bad-{mode}-0001"))
    failed = _wait(service, started["run_id"], "FAILED")
    assert failed["handoff_ready"] is False
    serialized = json.dumps(failed)
    assert "127.0.0.1" not in serialized
    assert "token=private" not in serialized
    assert not (project / "harvest_runs" / started["run_id"] / "handoff.json").exists()


def test_backend_must_have_complete_certification_before_start(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="every required safety gate"):
        replace(_capabilities(), validates_every_redirect=False)
    source = _source()
    service = HarvestService(project, sources=(source,))
    with pytest.raises(HarvestError) as unavailable:
        service.start("open.source", **_start_arguments(service, "backend-missing-0001"))
    assert unavailable.value.code == "harvest_backend_unavailable"
    assert not (project / "harvest_runs").exists()


def test_handoff_and_idempotent_reuse_rehash_every_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, calls, _capability = _service(project)
    arguments = _start_arguments(service, "rehash-reuse-0001")
    started, _created = service.start("open.source", **arguments)
    _wait(service, started["run_id"], "COMPLETE")
    artifact = project / "harvest_runs" / started["run_id"] / "artifacts" / "sprites.png"
    artifact.write_bytes(PNG + b"tampered")
    with pytest.raises(HarvestError) as handoff_error:
        service.handoff(started["run_id"])
    assert handoff_error.value.code == "harvest_artifact_verification_failed"
    with pytest.raises(HarvestError) as reuse_error:
        service.start("open.source", **arguments)
    assert reuse_error.value.code == "harvest_artifact_verification_failed"
    assert len(calls) == 1


def test_same_source_active_conflict_and_cross_instance_lock(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    entered = threading.Event()
    release = threading.Event()
    first, _calls, capability = _service(project, entered=entered, release=release)
    started, _created = first.start("open.source", **_start_arguments(first, "active-first-0001"))
    assert entered.wait(2)
    second, _second_calls, _ = _service(project, capabilities=capability)
    with pytest.raises(HarvestError) as conflict:
        second.start("open.source", **_start_arguments(second, "active-second-0001"))
    assert conflict.value.code == "harvest_source_active_conflict"
    first.cancel(started["run_id"], explicit_action=True)
    release.set()
    _wait(first, started["run_id"], "CANCELLED")

    lock_root = project / "lock-probe"
    lock_root.mkdir()
    with RepositoryMutationLock(lock_root):
        with pytest.raises(HarvestStorageError, match="holds the mutation lock"):
            with RepositoryMutationLock(lock_root, timeout_seconds=0.05):
                pass


def test_idempotency_binds_backend_and_limit_identities(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    first, _calls, _capability = _service(project)
    arguments = _start_arguments(first, "identity-full-0001")
    started, _created = first.start("open.source", **arguments)
    _wait(first, started["run_id"], "COMPLETE")
    changed, _changed_calls, _ = _service(project, capabilities=_capabilities(code_identity=SHA_B))
    with pytest.raises(HarvestError) as conflict:
        changed.start("open.source", **arguments)
    assert conflict.value.code == "idempotency_conflict"


class HandoffBarrierService(HarvestService):
    barrier_entered: threading.Event
    barrier_release: threading.Event

    def _build_handoff(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.barrier_entered.set()
        assert self.barrier_release.wait(2)
        return super()._build_handoff(*args, **kwargs)


def test_cancellation_is_rechecked_before_handoff_publication(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    HandoffBarrierService.barrier_entered = threading.Event()
    HandoffBarrierService.barrier_release = threading.Event()
    service, _calls, _ = _service(project, service_type=HandoffBarrierService)
    started, _created = service.start("open.source", **_start_arguments(service, "cancel-handoff-0001"))
    assert HandoffBarrierService.barrier_entered.wait(2)
    service.cancel(started["run_id"], explicit_action=True)
    HandoffBarrierService.barrier_release.set()
    _wait(service, started["run_id"], "CANCELLED")
    assert not (project / "harvest_runs" / started["run_id"] / "handoff.json").exists()


def test_event_count_and_bytes_are_bounded(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    limits = HarvestLimits(max_events=4, max_event_bytes=1024)
    service, _calls, _ = _service(project, mode="chatty", limits=limits)
    started, _created = service.start("open.source", **_start_arguments(service, "event-cap-0001"))
    _wait(service, started["run_id"], "FAILED")
    event_path = project / "harvest_runs" / started["run_id"] / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= limits.max_events
    assert all(len(line.encode("utf-8")) <= limits.max_event_bytes for line in lines)


class FakeDatasetImport:
    callback_id = "dataset.import"
    code_identity_sha256 = SHA_A

    def __init__(self) -> None:
        self.calls: list[DatasetImportRequest] = []

    def import_harvest(
        self,
        request: DatasetImportRequest,
        *,
        idempotency_key: str,
    ) -> DatasetImportResult:
        assert idempotency_key == "dataset-import-0001"
        assert request.artifacts_directory.name == "artifacts"
        self.calls.append(request)
        return DatasetImportResult("dataset.imported", 1, 0)


def test_dataset_import_callback_is_explicit_rehashed_and_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    callback = FakeDatasetImport()
    service, _calls, _ = _service(project, callback=callback)
    started, _created = service.start("open.source", **_start_arguments(service, "import-source-0001"))
    _wait(service, started["run_id"], "COMPLETE")
    receipt = service.import_to_dataset(started["run_id"], explicit_action=True, idempotency_key="dataset-import-0001")
    repeated = service.import_to_dataset(started["run_id"], explicit_action=True, idempotency_key="dataset-import-0001")
    assert repeated == receipt
    assert receipt["schema_version"] == "spritelab.harvest.dataset-import-receipt.v1"
    assert receipt["paths_exposed"] is False
    assert len(callback.calls) == 1


def test_default_plugin_remains_passive_and_unavailable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    plugin = create_plugin()
    result = plugin.status_provider(ProjectContext(project))
    assert result.status is ProductStatus.UNAVAILABLE
    assert result.data["network_actions"] == 0
    assert not (project / "harvest_runs").exists()
