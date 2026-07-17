from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import spritelab.product_features.harvest.catalog as harvest_catalog_module
import spritelab.product_features.harvest.service as harvest_service_module
from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_features.harvest import create_plugin
from spritelab.product_features.harvest.catalog import (
    CatalogAutomationTermsBinding,
    CatalogEvidenceBinding,
    HarvestSource,
    automation_terms_decision_identity,
    url_identity,
)
from spritelab.product_features.harvest.catalog_verifier import (
    CATALOG_EVIDENCE_VERIFIER_ID,
    catalog_evidence_verifier_code_identity,
)
from spritelab.product_features.harvest.service import HarvestError, HarvestService
from spritelab.product_features.harvest.storage import (
    HarvestStorageError,
    RepositoryMutationLock,
    append_stable_single_link_bytes,
    read_stable_single_link_bytes,
    write_atomic_stable_bytes,
    write_exclusive_stable_bytes,
)
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
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation

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
    verified_text = verified.isoformat().replace("+00:00", "Z")
    expires_text = expires.isoformat().replace("+00:00", "Z")
    source_hash = hashlib.sha256(b"source-page").hexdigest()
    terms = CatalogAutomationTermsBinding(
        mode="source_page_no_governing_terms_link",
        decision="NO_PROHIBITION_OBSERVED",
        evidence_url=SOURCE_URL,
        evidence_request_url_sha256=url_identity(SOURCE_URL),
        evidence_final_url=SOURCE_URL,
        evidence_http_status=200,
        evidence_content_sha256=source_hash,
        matched_declaration=None,
        limited_evidence=True,
        decision_identity_sha256=automation_terms_decision_identity(
            mode="source_page_no_governing_terms_link",
            evidence_url=SOURCE_URL,
            content_sha256=source_hash,
            matched_declaration=None,
            decision="NO_PROHIBITION_OBSERVED",
        ),
        verified_at=verified_text,
        expires_at=expires_text,
    )
    provisional = CatalogEvidenceBinding(
        verifier_id=CATALOG_EVIDENCE_VERIFIER_ID,
        verifier_code_identity_sha256=catalog_evidence_verifier_code_identity(),
        verified_at=verified_text,
        expires_at=expires_text,
        source_request_url_sha256=url_identity(SOURCE_URL),
        source_final_url=SOURCE_URL,
        source_http_status=200,
        source_content_sha256=source_hash,
        license_request_url_sha256=url_identity(LICENSE_URL),
        license_final_url=LICENSE_URL,
        license_http_status=200,
        license_content_sha256=hashlib.sha256(b"license-page").hexdigest(),
        automation_terms=terms,
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


def _capabilities(*, code_identity: str = SHA_A, callback: Any = None) -> CertifiedBackendCapabilities:
    callback_id = callback.callback_id if callback is not None else "dataset.conditioned-intake"
    callback_code_identity = callback.code_identity_sha256 if callback is not None else SHA_A
    callback_runtime_identity = callback.runtime_identity_sha256 if callback is not None else SHA_B
    return CertifiedBackendCapabilities(
        backend_id="fixture.backend",
        backend_version="1.0",
        downloader_id="fixture.downloader",
        downloader_version="1.0",
        code_identity_sha256=code_identity,
        dataset_import_callback_id=callback_id,
        dataset_import_callback_code_identity_sha256=callback_code_identity,
        dataset_import_callback_runtime_identity_sha256=callback_runtime_identity,
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
        enforces_bounded_evidence_fetch=True,
        enforces_quarantine_hash_probe=True,
        enforces_probe_no_decode_extract_import=True,
        enforces_deterministic_evidence_verification=True,
        enforces_transactional_catalog_promotion=True,
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
    certified = capabilities or _capabilities(callback=callback)
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


def test_source_page_terms_binding_cannot_be_rebound_or_expired_independently() -> None:
    binding = _binding()
    changed_url = "https://catalog.example.test/other-terms"
    changed_terms = replace(
        binding.automation_terms,
        evidence_url=changed_url,
        evidence_request_url_sha256=url_identity(changed_url),
        evidence_final_url=changed_url,
        decision_identity_sha256=automation_terms_decision_identity(
            mode="source_page_no_governing_terms_link",
            evidence_url=changed_url,
            content_sha256=binding.automation_terms.evidence_content_sha256,
            matched_declaration=None,
            decision="NO_PROHIBITION_OBSERVED",
        ),
    )
    changed_provisional = replace(
        binding,
        automation_terms=changed_terms,
        attestation_identity_sha256="0" * 64,
    )
    changed = replace(
        changed_provisional,
        attestation_identity_sha256=changed_provisional.expected_attestation_identity,
    )
    with pytest.raises(ValueError, match="source page"):
        _source(binding=changed)

    expired_terms = replace(
        binding.automation_terms,
        verified_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-02T00:00:00Z",
    )
    expired_provisional = replace(
        binding,
        automation_terms=expired_terms,
        attestation_identity_sha256="0" * 64,
    )
    expired = replace(
        expired_provisional,
        attestation_identity_sha256=expired_provisional.expected_attestation_identity,
    )
    with pytest.raises(ValueError, match="automation-terms evidence"):
        _source(binding=expired)
    with pytest.raises(ValueError, match="source-page evidence binding changed"):
        replace(_source(), source_page="https://catalog.example.test/changed")


def test_catalog_binding_becomes_stale_when_live_verifier_identity_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    replacement_identity = "0" * 64
    assert replacement_identity != binding.verifier_code_identity_sha256
    monkeypatch.setattr(
        harvest_catalog_module,
        "catalog_evidence_verifier_code_identity",
        lambda: replacement_identity,
    )

    with pytest.raises(ValueError, match="code identity"):
        _source(binding=binding)


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
    assert conflict.value.code == "harvest_single_flight_conflict"
    first.cancel(started["run_id"], explicit_action=True)
    release.set()
    _wait(first, started["run_id"], "CANCELLED")

    lock_root = project / "lock-probe"
    lock_root.mkdir()
    with RepositoryMutationLock(lock_root):
        with pytest.raises(HarvestStorageError, match="holds the mutation lock"):
            with RepositoryMutationLock(lock_root, timeout_seconds=0.05):
                pass


@pytest.mark.parametrize("operation", ["read", "append", "exclusive", "atomic"])
def test_durable_storage_refuses_parent_rename_symlink_aba(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    root = tmp_path / "root"
    run = root / "run"
    moved = root / "run-held"
    outside = tmp_path / "outside"
    run.mkdir(parents=True)
    outside.mkdir()
    (run / "evidence.bin").write_bytes(b"inside")
    outside_evidence = outside / "evidence.bin"
    outside_evidence.write_bytes(b"outside sentinel")
    real_open = AnchoredDirectory.open_file
    swapped = False

    def swap_before_relative_open(anchor, name, flags, mode=0o600):
        nonlocal swapped
        if not swapped and anchor.directory == run:
            try:
                os.replace(run, moved)
            except OSError:
                pytest.skip("the platform held the durable evidence parent against rename")
            try:
                os.symlink(outside, run, target_is_directory=True)
            except OSError:
                os.replace(moved, run)
                pytest.skip("directory symbolic links are unavailable in this test session")
            swapped = True
        return real_open(anchor, name, flags, mode)

    monkeypatch.setattr(AnchoredDirectory, "open_file", swap_before_relative_open)
    try:
        with pytest.raises((HarvestStorageError, UnsafeFilesystemOperation)):
            if operation == "read":
                read_stable_single_link_bytes(run / "evidence.bin", root, max_bytes=1024)
            elif operation == "append":
                append_stable_single_link_bytes(
                    run / "evidence.bin",
                    root,
                    b"+append",
                    max_bytes=1024,
                    max_total_bytes=2048,
                )
            elif operation == "exclusive":
                write_exclusive_stable_bytes(run / "new.bin", root, b"new", max_bytes=1024)
            else:
                write_atomic_stable_bytes(run / "evidence.bin", root, b"new", max_bytes=1024)
    finally:
        if swapped:
            os.unlink(run)
            os.replace(moved, run)

    assert outside_evidence.read_bytes() == b"outside sentinel"
    assert not (outside / "new.bin").exists()


def test_passive_legacy_inventory_does_not_follow_parent_aba(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    legacy = project / "harvest_runs" / "legacy_pack"
    moved = project / "harvest_runs" / "legacy_pack-held"
    outside = tmp_path / "outside-legacy"
    legacy.mkdir(parents=True)
    outside.mkdir()
    (legacy / "sources.jsonl").write_bytes(b'{"source_id":"inside"}\n')
    outside_payload = b'{"source_id":"outside"}\n'
    (outside / "sources.jsonl").write_bytes(outside_payload)
    real_open = AnchoredDirectory.open_file
    swapped = False

    def swap_before_read(anchor, name, flags, mode=0o600):
        nonlocal swapped
        if not swapped and anchor.directory == legacy and name == "sources.jsonl":
            try:
                os.replace(legacy, moved)
                os.symlink(outside, legacy, target_is_directory=True)
            except OSError:
                if moved.exists() and not legacy.exists():
                    os.replace(moved, legacy)
                pytest.skip("the platform refused the held legacy-run rename")
            swapped = True
        return real_open(anchor, name, flags, mode)

    monkeypatch.setattr(AnchoredDirectory, "open_file", swap_before_read)
    try:
        inventory = HarvestService(project).inventory()
    finally:
        if swapped:
            os.unlink(legacy)
            os.replace(moved, legacy)

    assert inventory["legacy_run_count"] == 0
    assert inventory["unsafe_entries"] == 1
    assert (outside / "sources.jsonl").read_bytes() == outside_payload


def test_worker_transaction_keeps_all_evidence_in_held_run_after_parent_aba(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output_root = project / "harvest_runs"
    outside = tmp_path / "outside-run"
    outside.mkdir()
    (outside / "sentinel.bin").write_bytes(b"outside sentinel")
    (outside / "state.json").write_bytes(b"outside state sentinel")
    capabilities = _capabilities()
    entered = threading.Event()
    release = threading.Event()
    swap: dict[str, Any] = {"complete": False, "error": None, "moved": None, "visible": None}

    class RenamingBackend(FixtureBackend):
        requires_destination_parent_anchor = True

        def acquire(self, source, destination, limits, *, cancel_requested, progress, **_kwargs):
            result = super().acquire(
                source,
                destination,
                limits,
                cancel_requested=cancel_requested,
                progress=progress,
            )
            visible = destination.parent
            moved = output_root / f"{visible.name}-held"
            swap.update({"moved": moved, "visible": visible})
            try:
                os.replace(visible, moved)
                os.symlink(outside, visible, target_is_directory=True)
            except OSError as exc:
                swap["error"] = exc
                if moved.exists() and not visible.exists():
                    os.replace(moved, visible)
            else:
                swap["complete"] = True
            return result

    service = HarvestService(
        project,
        sources=(_source(),),
        backend_capabilities=capabilities,
        backend_factory=lambda: RenamingBackend(
            capabilities,
            [],
            entered=entered,
            release=release,
        ),
    )
    started, _created = service.start("open.source", **_start_arguments(service, "worker-anchor-0001"))
    assert entered.wait(timeout=2)
    worker = service._workers[started["run_id"]]
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    if not swap["complete"]:
        pytest.skip(f"the platform refused the held run rename: {swap['error']}")
    moved = swap["moved"]
    visible = swap["visible"]
    assert isinstance(moved, Path)
    assert isinstance(visible, Path)
    try:
        assert (moved / "acquisition_receipt.json").is_file()
        assert (moved / "artifact_manifest.json").is_file()
        assert (moved / "handoff.json").is_file()
        assert (moved / "terminal_commit.json").is_file()
        assert json.loads((moved / "state.json").read_text(encoding="utf-8"))["status"] == "COMPLETE"
        assert (outside / "sentinel.bin").read_bytes() == b"outside sentinel"
        assert (outside / "state.json").read_bytes() == b"outside state sentinel"
        assert {path.name for path in outside.iterdir()} == {"sentinel.bin", "state.json"}
    finally:
        os.unlink(visible)
        os.replace(moved, visible)

    assert service.job(started["run_id"])["status"] == "COMPLETE"


def test_second_service_persists_cancellation_seen_inside_first_backend(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    entered = threading.Event()
    release = threading.Event()
    first, _calls, capability = _service(project, entered=entered, release=release)
    started, _created = first.start("open.source", **_start_arguments(first, "durable-cancel-0001"))
    assert entered.wait(2)
    second, _second_calls, _ = _service(project, capabilities=capability)
    cancelling = second.cancel(started["run_id"], explicit_action=True)
    assert cancelling["status"] == "CANCELLING"
    cancelled = _wait(first, started["run_id"], "CANCELLED")
    release.set()
    assert cancelled["handoff_ready"] is False
    request = project / "harvest_runs" / started["run_id"] / "cancellation_request.json"
    assert json.loads(request.read_text(encoding="utf-8"))["explicit_action"] is True
    with pytest.raises(HarvestError) as illegal:
        first._transition(
            request.parent,
            "RUNNING",
            stage="acquiring",
            message="must not resurrect a cancelled job",
        )
    assert illegal.value.code == "illegal_harvest_transition"


def test_expired_durable_lease_beats_same_process_pid_liveness(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    entered = threading.Event()
    release = threading.Event()
    first, _calls, capability = _service(project, entered=entered, release=release)
    started, _created = first.start("open.source", **_start_arguments(first, "expired-lease-0001"))
    run_id = started["run_id"]
    assert entered.wait(2)

    heartbeat_stop = first._lease_stops[run_id]
    heartbeat_thread = first._lease_threads[run_id]
    heartbeat_stop.set()
    heartbeat_thread.join(2)
    assert not heartbeat_thread.is_alive()
    time.sleep(2.1)

    second, _second_calls, _ = _service(project, capabilities=capability)
    cancelled = second.cancel(run_id, explicit_action=True)
    assert cancelled["status"] == "CANCELLED"
    release.set()
    _wait(first, run_id, "CANCELLED")


class ScalingBackend(FixtureBackend):
    def acquire(
        self,
        source: HarvestSource,
        destination: Path,
        limits: HarvestLimits,
        *,
        cancel_requested: Any,
        progress: Any,
    ) -> AcquisitionResult:
        del limits
        digest = hashlib.sha256(PNG).hexdigest()
        files: list[AcquiredFile] = []
        total = 2501
        for index in range(total):
            if cancel_requested():
                raise RuntimeError("cancelled")
            name = f"sprites/sprite-{index:04d}.png"
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(PNG)
            files.append(AcquiredFile(name, len(PNG), digest, "image/png", taxonomy=("item",)))
            progress("validating", index + 1, total)
        return AcquisitionResult(
            AcquisitionReceipt(
                final_url=DOWNLOAD_URL,
                redirect_chain=(),
                http_status=200,
                response_mime_type="application/zip",
                expected_response_sha256=source.expected_response_sha256,
                actual_response_sha256=source.expected_response_sha256,
                response_bytes=len(RESPONSE),
                elapsed_seconds=0.01,
                archive_members=total,
                archive_uncompressed_bytes=total * len(PNG),
                backend_capability_identity=self.capabilities.identity,
                files=tuple(files),
            )
        )


def test_progress_is_coalesced_for_more_than_2500_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    capabilities = _capabilities()
    limits = HarvestLimits(
        max_files=3000,
        max_events=1000,
        max_total_bytes=8 * 1024 * 1024,
    )
    service = HarvestService(
        project,
        sources=(_source(),),
        backend_factory=lambda: ScalingBackend(capabilities, []),
        backend_capabilities=capabilities,
        limits=limits,
    )
    started, _created = service.start("open.source", **_start_arguments(service, "scale-events-0001"))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        completed = service.job(started["run_id"])
        if completed["status"] == "COMPLETE":
            break
        if completed["status"] == "FAILED":
            raise AssertionError(completed)
        time.sleep(0.02)
    else:
        raise AssertionError("scaled Harvest run did not complete")
    event_path = project / "harvest_runs" / started["run_id"] / "events.jsonl"
    events = event_path.read_text(encoding="utf-8").splitlines()
    assert len(events) < 250
    assert json.loads(events[-1])["status"] == "COMPLETE"


def test_progress_hard_budget_is_independent_of_elapsed_callback_time(tmp_path: Path, monkeypatch) -> None:
    service, _calls, _capabilities = _service(tmp_path)
    emitted: list[int] = []
    clock = 0.0

    def jump_clock() -> float:
        nonlocal clock
        clock += 2.0
        return clock

    monkeypatch.setattr(harvest_service_module.time, "monotonic", jump_clock)
    monkeypatch.setattr(service, "_raise_if_cancelled", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        service,
        "_transition",
        lambda _run, _status, **fields: emitted.append(int(fields["current"])),
    )
    run_dir = tmp_path / "harvest-slow-progress"
    for current in range(1, 2502):
        service._progress(run_dir, threading.Event(), "validating", current, 2501)

    assert len(emitted) <= 200
    assert emitted[-1] == 2501


def test_complete_event_failure_never_exposes_importable_complete_state(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, _calls, _capabilities = _service(project)
    real_append = service._append_event

    def fail_complete_event(
        run_dir: Path,
        state: dict[str, Any],
        *,
        run_anchor: AnchoredDirectory | None = None,
    ) -> dict[str, Any]:
        if state.get("status") == "COMPLETE":
            raise HarvestError("injected_terminal_event_failure", "terminal event append failed")
        return real_append(run_dir, state, run_anchor=run_anchor)

    monkeypatch.setattr(service, "_append_event", fail_complete_event)
    started, _created = service.start("open.source", **_start_arguments(service, "terminal-event-fail-0001"))

    interrupted = _wait(service, started["run_id"], "INTERRUPTED")
    run = project / "harvest_runs" / started["run_id"]
    assert interrupted["handoff_ready"] is False
    assert not (run / "terminal_commit.json").exists()
    assert all(event["status"] != "COMPLETE" for event in interrupted["events"])
    with pytest.raises(HarvestError, match="handoff"):
        service.handoff(started["run_id"])


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
    runtime_identity_sha256 = SHA_B

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


def test_dataset_import_callback_runtime_drift_fails_before_invocation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    callback = FakeDatasetImport()
    service, _calls, _ = _service(project, callback=callback)
    started, _created = service.start("open.source", **_start_arguments(service, "import-runtime-0001"))
    _wait(service, started["run_id"], "COMPLETE")
    callback.runtime_identity_sha256 = SHA_A

    with pytest.raises(HarvestError) as error:
        service.import_to_dataset(
            started["run_id"],
            explicit_action=True,
            idempotency_key="dataset-import-0001",
        )

    assert error.value.code == "dataset_import_identity_changed"
    assert callback.calls == []


def test_default_plugin_remains_passive_and_unavailable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    plugin = create_plugin()
    result = plugin.status_provider(ProjectContext(project))
    assert result.status is ProductStatus.UNAVAILABLE
    assert result.data["network_actions"] == 0
    assert not (project / "harvest_runs").exists()


def test_context_bound_dataset_callback_factory_receives_trusted_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    observed: list[Path] = []

    def callback_factory(context: ProjectContext) -> FakeDatasetImport:
        observed.append(context.project_root)
        return FakeDatasetImport()

    plugin = create_plugin(sources=(), dataset_import_callback_factory=callback_factory)
    result = plugin.status_provider(ProjectContext(project))
    assert result.status is ProductStatus.UNAVAILABLE
    assert observed == [project]
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_plugin(
            sources=(),
            dataset_import_callback=FakeDatasetImport(),
            dataset_import_callback_factory=callback_factory,
        )
