from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import spritelab.product_features.harvest as harvest_plugin_module
import spritelab.product_features.harvest.onboarding as onboarding_module
import spritelab.product_features.harvest.service as harvest_service_module
from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_features.harvest import create_plugin
from spritelab.product_features.harvest.catalog import (
    TRUSTED_CATALOG_DIRECTORY_RELATIVE_PATH,
    TRUSTED_CATALOG_RELATIVE_PATH,
    TrustedCatalogError,
    load_trusted_catalog,
    trusted_catalog_source_filename,
)
from spritelab.product_features.harvest.certification import (
    BackendCapabilityCertificateError,
    BackendCapabilityEvidence,
)
from spritelab.product_features.harvest.evidence_fetch import (
    EvidenceFetchError,
    FetchSnapshot,
    fetch_robots_snapshot,
    verify_automation_terms,
    verify_evidence_pages,
)
from spritelab.product_features.harvest.service import HarvestError, HarvestService
from spritelab.product_features.harvest.trusted_backend import CertifiedBackendCapabilities
from spritelab.product_web.app import create_app
from spritelab.utils.safe_fs import AnchoredDirectory

ROBOTS_ALLOW = b"User-agent: spritelab-harvest\nAllow: /\n"
OGA_ROBOTS_COMMENTED_RULES = (
    b"User-agent: *\nCrawl-delay: 10\n# CSS, JS, Images\nAllow: /misc/*.css$\nDisallow: /misc/\n"
)
SOURCE_URL = "https://catalog.example.test/source"
LICENSE_URL = "https://catalog.example.test/license"
TERMS_URL = "https://catalog.example.test/terms"
DIRECT_URL = "https://downloads.example.test/pack.zip?token=private"
OGA_SOURCE_URL = "https://opengameart.org/content/behrs-2500-pixel-battle-axes-32x32-archive"
OGA_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
OGA_DIRECT_URL = "https://opengameart.org/sites/default/files/battleaxes_01.zip"
RAW = b"PK\x03\x04raw-probe-only"


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "text/plain",
        peer_ip: str = "8.8.8.8",
    ) -> None:
        self.body = body
        self.offset = 0
        self.status = status
        self.peer_ip = peer_ip
        self.headers: Mapping[str, str] = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.body):
            return b""
        end = len(self.body) if size < 0 else min(self.offset + size, len(self.body))
        chunk = self.body[self.offset : end]
        self.offset = end
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def open(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected network request")
        return self.responses.pop(0)


class BlockingResponse(FakeResponse):
    def __init__(self, body: bytes, entered: threading.Event, release: threading.Event) -> None:
        super().__init__(body)
        self.entered = entered
        self.release = release

    def read(self, size: int = -1) -> bytes:
        if self.offset == 0:
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("test did not release bounded response")
        return super().read(size)


def _capabilities() -> CertifiedBackendCapabilities:
    return CertifiedBackendCapabilities(
        backend_id="onboarding.test",
        backend_version="1",
        downloader_id="pinned.test",
        downloader_version="1",
        code_identity_sha256="a" * 64,
        dataset_import_callback_id="dataset.conditioned-intake",
        dataset_import_callback_code_identity_sha256="b" * 64,
        dataset_import_callback_runtime_identity_sha256="c" * 64,
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
        enforces_direct_static_image_derivation=True,
        enforces_retained_anchored_state=True,
        enforces_whole_operation_deadline=True,
        enforces_durable_import_control=True,
        enforces_same_pack_license_and_zero_cost=True,
        enforces_technical_usability_and_pixel_uniqueness=True,
        enforces_non_self_attested_production_bindings=True,
    )


def _evidence(capabilities: CertifiedBackendCapabilities) -> BackendCapabilityEvidence:
    now = datetime.now(timezone.utc)
    return BackendCapabilityEvidence(
        capabilities=capabilities,
        auditor_id="independent.test",
        audited_at=(now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
        issued_at=(now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        audit_report_sha256="b" * 64,
        audit_report_identity="c" * 64,
        certificate_identity="d" * 64,
        implementation_identity_sha256=capabilities.code_identity_sha256,
    )


def _source_html(*, automation: str = "", include_terms: bool = True, extra_terms_link: str = "") -> bytes:
    terms_link = f'<a href="{TERMS_URL}">Automation terms</a>' if include_terms else ""
    return (
        "<html><head><title>Verified Sprite Pack</title></head><body>"
        "<article>"
        "<h1>Verified Sprite Pack</h1><p>by Example Artist</p>"
        "<p>Free zero-cost download.</p>"
        f"<p>{automation}</p>"
        f'<a href="{LICENSE_URL}">CC0 license</a>'
        f"{terms_link}{extra_terms_link}"
        f'<a href="{DIRECT_URL}">Direct download</a>'
        "</article>"
        "</body></html>"
    ).encode()


def _oga_source_html() -> bytes:
    return (
        "<html><body>"
        '<div class="node node-art view-mode-full clearfix">'
        '<div class="field field-name-title field-type-ds field-label-hidden">'
        "<h1>Behr's 2500+ Pixel Battle Axes 32x32 Archive</h1></div>"
        '<div class="field field-name-author-submitter field-type-ds field-label-hidden">'
        "Submitted by Behrtron</div>"
        '<div class="field field-name-field-art-licenses field-type-taxonomy-term-reference">'
        f'<a href="{OGA_LICENSE_URL}">CC0 1.0 public domain</a></div>'
        '<div class="field field-name-body field-type-text-with-summary">'
        "Hey guys, I have been spriting for some years now, im uploading my archive totally free "
        "in the public domain.</div>"
        '<div class="field field-name-field-art-files field-type-file">'
        f'<a href="{OGA_DIRECT_URL}">battleaxes_01.zip</a></div>'
        "</div>"
        "</body></html>"
    ).encode()


def _verify_oga_source(
    source: bytes,
    *,
    source_url: str = OGA_SOURCE_URL,
    source_snapshot: FetchSnapshot | None = None,
) -> Any:
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"
    return verify_evidence_pages(
        source_url=source_url,
        source_snapshot=source_snapshot or _snapshot(source_url, source),
        source_bytes=source,
        license_url=OGA_LICENSE_URL,
        license_snapshot=_snapshot(OGA_LICENSE_URL, license_page),
        license_bytes=license_page,
        title="Behr's 2500+ Pixel Battle Axes 32x32 Archive",
        creator="Behrtron",
        license_id="cc0-1.0",
        direct_download_url=OGA_DIRECT_URL,
    )


def _responses() -> list[FakeResponse]:
    terms = b"<html><body><h1>Automation policy</h1><p>Automated downloads: allowed</p></body></html>"
    license_page = b"<html><body><h1>CC0 1.0 Universal</h1><p>Public domain dedication</p></body></html>"
    return [
        FakeResponse(ROBOTS_ALLOW),
        FakeResponse(_source_html(), content_type="text/html"),
        FakeResponse(terms, content_type="text/html"),
        FakeResponse(license_page, content_type="text/html"),
        FakeResponse(ROBOTS_ALLOW),
        FakeResponse(RAW, content_type="application/zip"),
    ]


def _service(
    project: Path,
    transport: FakeTransport,
    *,
    limits: Any = None,
) -> tuple[HarvestService, BackendCapabilityEvidence]:
    capabilities = _capabilities()
    evidence = _evidence(capabilities)
    return (
        HarvestService(
            project,
            backend_factory=lambda: None,
            backend_capabilities=capabilities,
            backend_capability_evidence=evidence,
            limits=limits,
            probe_resolver=lambda _host, _port: ("8.8.8.8",),
            probe_transport=transport,
            allow_unverified_test_backend=True,
        ),
        evidence,
    )


def _payload(service: HarvestService, evidence: BackendCapabilityEvidence, *, key: str) -> dict[str, Any]:
    return {
        "source_id": "verified.pack",
        "title": "Verified Sprite Pack",
        "creator": "Example Artist",
        "source_page": SOURCE_URL,
        "license_id": "cc0-1.0",
        "license_evidence_url": LICENSE_URL,
        "terms_evidence_url": TERMS_URL,
        "direct_download_url": DIRECT_URL,
        "attribution_text": "Example Artist - Verified Sprite Pack",
        "taxonomy_hints": ["item"],
        "inventory_identity": service.inventory()["inventory_identity"],
        "backend_capability_evidence_identity": evidence.identity,
        "idempotency_key": key,
        "explicit_action": True,
        "authorize_network": True,
        "authorize_hash_probe": True,
        "authorize_zero_cost": True,
        "authorize_permissive_license": True,
    }


def _wait(service: HarvestService, probe_id: str, expected: set[str]) -> dict[str, Any]:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        status = service.probe(probe_id)
        if status["status"] in expected:
            return status
        time.sleep(0.02)
    raise AssertionError(f"probe did not reach {expected}")


def _promotion_review(service: HarvestService, probe_id: str) -> dict[str, Any]:
    receipt = service.probe_evidence(probe_id)["receipt"]
    return {
        "authorize_zero_cost_evidence_review": True,
        "reviewed_verification_identity": receipt["verification_identity"],
        "reviewed_source_pack_evidence_sha256": receipt["source_pack_evidence_sha256"],
    }


def _snapshot(url: str, payload: bytes, mime: str = "text/html") -> FetchSnapshot:
    return FetchSnapshot(
        request_url_sha256=hashlib.sha256(url.encode()).hexdigest(),
        request_public_url=url,
        final_url=url,
        http_status=200,
        mime_type=mime,
        byte_count=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
        elapsed_seconds=0.01,
        relative_file="page.bin",
    )


def test_probe_is_quarantine_only_and_promotion_is_explicit_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    transport = FakeTransport(_responses())
    service, evidence = _service(project, transport)

    started, created = service.start_probe(_payload(service, evidence, key="probe-success-0001"))
    assert created is True
    ready = _wait(service, started["probe_id"], {"READY", "FAILED"})
    assert ready["status"] == "READY"
    durable = service.probe_evidence(started["probe_id"])
    assert durable["network_actions_recorded"] == 6
    assert durable["receipt"]["terms_policy_decision"] == "ALLOW"
    assert durable["receipt"]["extraction_count"] == durable["receipt"]["decode_count"] == 0
    assert durable["receipt"]["source_pack_evidence_text"] == (
        "Verified Sprite Pack by Example Artist Free zero-cost download. CC0 license Automation terms Direct download"
    )
    assert (
        durable["receipt"]["source_pack_evidence_sha256"]
        == hashlib.sha256(durable["receipt"]["source_pack_evidence_text"].encode("utf-8")).hexdigest()
    )
    run = project / "harvest_runs" / started["probe_id"]
    assert (run / "quarantine" / "raw_payload.bin").read_bytes() == RAW
    assert not (run / "artifacts").exists()
    assert len(transport.calls) == 6

    live_refresh = service._refresh_catalog_sources
    refresh_attempts = 0

    def flaky_refresh(sources: tuple[Any, ...]) -> None:
        nonlocal refresh_attempts
        refresh_attempts += 1
        if refresh_attempts == 1:
            raise RuntimeError("injected live-view refresh failure")
        live_refresh(sources)

    service._probe_service._catalog_refreshed = flaky_refresh
    promotion = service.promote_probe(
        started["probe_id"],
        explicit_action=True,
        authorize_catalog_promotion=True,
        **_promotion_review(service, started["probe_id"]),
    )
    assert promotion["raw_response_sha256"] == hashlib.sha256(RAW).hexdigest()
    assert promotion["zero_cost_evidence_reviewed"] is True
    assert promotion["reviewed_verification_identity"] == durable["receipt"]["verification_identity"]
    assert promotion["reviewed_source_pack_evidence_sha256"] == durable["receipt"]["source_pack_evidence_sha256"]
    assert refresh_attempts == 1
    assert service.sources()["sources"] == []
    assert (
        service.promote_probe(
            started["probe_id"],
            explicit_action=True,
            authorize_catalog_promotion=True,
            **_promotion_review(service, started["probe_id"]),
        )
        == promotion
    )
    assert refresh_attempts == 2
    catalog = load_trusted_catalog(project)
    assert [source.source_id for source in catalog] == ["verified.pack"]
    assert catalog[0].expected_response_sha256 == hashlib.sha256(RAW).hexdigest()
    assert catalog[0].evidence_binding.zero_cost_reviewed is True
    assert (
        catalog[0].evidence_binding.zero_cost_verification_identity_sha256
        == durable["receipt"]["verification_identity"]
    )
    assert (
        catalog[0].evidence_binding.zero_cost_evidence_text_sha256 == durable["receipt"]["source_pack_evidence_sha256"]
    )
    assert catalog[0].evidence_binding.zero_cost_probe_receipt_identity_sha256 == durable["receipt"]["receipt_identity"]
    assert catalog[0].evidence_binding.automation_terms.decision == "ALLOW"
    assert catalog[0].evidence_binding.automation_terms.evidence_url == TERMS_URL
    assert service.sources()["sources"][0]["evidence_binding"]["automation_terms"]["limited_evidence"] is False
    assert service.sources()["sources"][0]["source_id"] == "verified.pack"
    reloaded = HarvestService(
        project,
        sources=load_trusted_catalog(project),
        backend_factory=lambda: None,
        backend_capabilities=_capabilities(),
        allow_unverified_test_backend=True,
    )
    assert reloaded.sources()["sources"][0]["evidence_binding"]["automation_terms"]["decision"] == "ALLOW"

    source = catalog[0]
    expired_terms = replace(
        source.evidence_binding.automation_terms,
        verified_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-02T00:00:00Z",
    )
    expired_provisional = replace(
        source.evidence_binding,
        automation_terms=expired_terms,
        attestation_identity_sha256="0" * 64,
    )
    expired_binding = replace(
        expired_provisional,
        attestation_identity_sha256=expired_provisional.expected_attestation_identity,
    )
    with pytest.raises(ValueError, match="automation-terms evidence"):
        replace(source, evidence_binding=expired_binding)

    catalog_path = project / TRUSTED_CATALOG_DIRECTORY_RELATIVE_PATH / trusted_catalog_source_filename(source.source_id)
    tampered = json.loads(catalog_path.read_text(encoding="utf-8"))
    tampered["source"]["evidence_binding"]["automation_terms"]["decision_identity_sha256"] = "0" * 64
    catalog_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(TrustedCatalogError):
        load_trusted_catalog(project)


def test_promotion_requires_exact_reviewed_zero_cost_evidence_identity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, evidence = _service(project, FakeTransport(_responses()))
    started, _created = service.start_probe(_payload(service, evidence, key="probe-review-binding-0001"))
    assert _wait(service, started["probe_id"], {"READY", "FAILED"})["status"] == "READY"
    review = _promotion_review(service, started["probe_id"])

    for override in (
        {"authorize_zero_cost_evidence_review": False},
        {"reviewed_verification_identity": "0" * 64},
        {"reviewed_source_pack_evidence_sha256": "0" * 64},
    ):
        submitted = {**review, **override}
        with pytest.raises(HarvestError, match=r"authorization|exact unchanged"):
            service.promote_probe(
                started["probe_id"],
                explicit_action=True,
                authorize_catalog_promotion=True,
                **submitted,
            )
    assert load_trusted_catalog(project) == ()


def test_catalog_atomically_embeds_review_binding_before_receipt_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, evidence = _service(project, FakeTransport(_responses()))
    started, _created = service.start_probe(_payload(service, evidence, key="probe-review-crash-0001"))
    probe_id = started["probe_id"]
    assert _wait(service, probe_id, {"READY", "FAILED"})["status"] == "READY"
    review = _promotion_review(service, probe_id)
    writer = service._probe_service._write_exclusive_json

    def fail_promotion_receipt(path: Path, payload: Mapping[str, Any], anchor: AnchoredDirectory) -> None:
        if path.name == "promotion_receipt.json":
            raise OSError("injected promotion-receipt publication failure")
        writer(path, payload, anchor)

    monkeypatch.setattr(service._probe_service, "_write_exclusive_json", fail_promotion_receipt)
    with pytest.raises(OSError, match="injected promotion-receipt"):
        service.promote_probe(
            probe_id,
            explicit_action=True,
            authorize_catalog_promotion=True,
            **review,
        )

    run = project / "harvest_runs" / probe_id
    assert not (run / "promotion_receipt.json").exists()
    source = load_trusted_catalog(project)[0]
    assert source.evidence_binding.zero_cost_reviewed is True
    assert source.evidence_binding.zero_cost_verification_identity_sha256 == review["reviewed_verification_identity"]
    assert source.evidence_binding.zero_cost_evidence_text_sha256 == review["reviewed_source_pack_evidence_sha256"]

    monkeypatch.setattr(service._probe_service, "_write_exclusive_json", writer)
    recovered = service.promote_probe(
        probe_id,
        explicit_action=True,
        authorize_catalog_promotion=True,
        **review,
    )
    assert recovered["catalog_changed"] is False
    assert (run / "promotion_receipt.json").exists()


def test_promotion_replay_rejects_tampered_immutable_receipt(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, evidence = _service(project, FakeTransport(_responses()))
    started, _created = service.start_probe(_payload(service, evidence, key="probe-promotion-tamper-0001"))
    probe_id = started["probe_id"]
    assert _wait(service, probe_id, {"READY", "FAILED"})["status"] == "READY"
    review = _promotion_review(service, probe_id)
    service.promote_probe(
        probe_id,
        explicit_action=True,
        authorize_catalog_promotion=True,
        **review,
    )

    promotion_path = project / "harvest_runs" / probe_id / "promotion_receipt.json"
    tampered = json.loads(promotion_path.read_text(encoding="utf-8"))
    tampered["catalog_identity"] = "0" * 64
    promotion_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(HarvestError, match="promotion evidence is invalid"):
        service.promote_probe(
            probe_id,
            explicit_action=True,
            authorize_catalog_promotion=True,
            **review,
        )
    assert load_trusted_catalog(project)[0].source_id == "verified.pack"


def test_promotion_replay_rejects_reidentified_result_substitution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, evidence = _service(project, FakeTransport(_responses()))
    started, _created = service.start_probe(_payload(service, evidence, key="probe-result-tamper-0001"))
    probe_id = started["probe_id"]
    assert _wait(service, probe_id, {"READY", "FAILED"})["status"] == "READY"
    review = _promotion_review(service, probe_id)
    service.promote_probe(
        probe_id,
        explicit_action=True,
        authorize_catalog_promotion=True,
        **review,
    )

    result_path = project / "harvest_runs" / probe_id / "result.json"
    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["raw_response_bytes"] += 1
    tampered["result_identity"] = onboarding_module._identity(
        {key: value for key, value in tampered.items() if key != "result_identity"}
    )
    result_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(HarvestError, match="promotion evidence is invalid"):
        service.promote_probe(
            probe_id,
            explicit_action=True,
            authorize_catalog_promotion=True,
            **review,
        )


def test_probe_uses_one_monotonic_deadline_for_every_network_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    clock = [100.0]

    class AdvancingTransport(FakeTransport):
        def open(self, **kwargs: Any) -> FakeResponse:
            response = super().open(**kwargs)
            clock[0] += 1.0
            return response

    monkeypatch.setattr(onboarding_module.time, "monotonic", lambda: clock[0])
    transport = AdvancingTransport(_responses())
    service, evidence = _service(
        project,
        transport,
        limits=onboarding_module.HarvestLimits(max_duration_seconds=10.0),
    )

    started, _created = service.start_probe(_payload(service, evidence, key="probe-deadline-success"))
    completed = _wait(service, started["probe_id"], {"READY", "FAILED"})
    timeouts = [float(call["timeout_seconds"]) for call in transport.calls]

    assert completed["status"] == "READY"
    assert timeouts == pytest.approx([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    assert all(left > right > 0 for left, right in pairwise(timeouts))


def test_probe_whole_deadline_fails_with_durable_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    clock = [200.0]

    class AdvancingTransport(FakeTransport):
        def open(self, **kwargs: Any) -> FakeResponse:
            response = super().open(**kwargs)
            clock[0] += 1.0
            return response

    monkeypatch.setattr(onboarding_module.time, "monotonic", lambda: clock[0])
    transport = AdvancingTransport(_responses())
    service, evidence = _service(
        project,
        transport,
        limits=onboarding_module.HarvestLimits(max_duration_seconds=4.5),
    )

    started, _created = service.start_probe(_payload(service, evidence, key="probe-deadline-failure"))
    failed = _wait(service, started["probe_id"], {"FAILED"})
    durable = service.probe_evidence(started["probe_id"])

    assert failed["status"] == "FAILED"
    assert failed["ended_at"] is not None
    assert failed["events"][-1]["status"] == "FAILED"
    assert failed["events"][-1]["stage"] == "failed"
    assert durable["receipt"] is None
    assert durable["result"] is None
    assert len(transport.calls) < len(_responses())


def test_failed_opengameart_probe_offers_exact_retained_page_prefill(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"
    transport = FakeTransport(
        [
            FakeResponse(ROBOTS_ALLOW),
            FakeResponse(_oga_source_html(), content_type="text/html"),
            FakeResponse(ROBOTS_ALLOW),
            FakeResponse(license_page, content_type="text/html"),
        ]
    )
    service, evidence = _service(project, transport)
    payload = _payload(service, evidence, key="probe-oga-prefill-recovery")
    payload.update(
        {
            "source_id": "oga.battle-axes",
            "title": "Behrs Battle Axes",
            "creator": "OpenGameArt",
            "source_page": OGA_SOURCE_URL,
            "license_evidence_url": OGA_LICENSE_URL,
            "terms_evidence_url": None,
            "direct_download_url": OGA_DIRECT_URL,
            "attribution_text": "OpenGameArt",
        }
    )

    started, _created = service.start_probe(payload)
    failed = _wait(service, started["probe_id"], {"FAILED"})

    assert "title field does not exactly identify" in failed["message"]
    assert failed["source_prefill"]["title"] == "Behr's 2500+ Pixel Battle Axes 32x32 Archive"
    assert failed["source_prefill"]["creator"] == "Behrtron"
    assert failed["source_prefill"]["license_id"] == "cc0-1.0"
    assert failed["source_prefill"]["license_evidence_url"] == OGA_LICENSE_URL
    assert failed["source_prefill"]["direct_download_url"] == OGA_DIRECT_URL
    assert len(transport.calls) == 4


def test_probe_cancellation_records_durable_terminal_evidence(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    entered = threading.Event()
    release = threading.Event()
    responses = _responses()
    transport = FakeTransport([BlockingResponse(ROBOTS_ALLOW, entered, release), *responses[1:]])
    service, evidence = _service(project, transport)

    started, _created = service.start_probe(_payload(service, evidence, key="probe-cancel-terminal"))
    assert entered.wait(3)
    cancelling = service.cancel_probe(started["probe_id"], explicit_action=True)
    assert cancelling["status"] == "CANCELLING"
    release.set()
    cancelled = _wait(service, started["probe_id"], {"CANCELLED"})
    durable = service.probe_evidence(started["probe_id"])

    assert cancelled["ended_at"] is not None
    assert cancelled["events"][-1]["status"] == "CANCELLED"
    assert cancelled["events"][-1]["stage"] == "cancelled"
    assert durable["receipt"] is None
    assert durable["result"] is None


def test_legacy_v1_catalog_fails_closed_instead_of_implying_terms_approval(tmp_path: Path) -> None:
    project = tmp_path / "project"
    path = project / TRUSTED_CATALOG_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "spritelab.harvest.trusted-catalog.v1",
                "sources": [],
                "catalog_identity": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrustedCatalogError, match="unsupported"):
        load_trusted_catalog(project)


def test_provenance_license_and_automation_terms_are_source_bound_and_inert() -> None:
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"
    common = (
        f'<html><body><h1>Cart collection</h1><p>CC licenses</p><a href="{DIRECT_URL}">Download</a></body></html>'
    ).encode()
    with pytest.raises(EvidenceFetchError, match="title"):
        verify_evidence_pages(
            source_url=SOURCE_URL,
            source_snapshot=_snapshot(SOURCE_URL, common),
            source_bytes=common,
            license_url=LICENSE_URL,
            license_snapshot=_snapshot(LICENSE_URL, license_page),
            license_bytes=license_page,
            title="Art",
            creator="CC",
            license_id="cc0-1.0",
            direct_download_url=DIRECT_URL,
        )

    unrelated = _source_html().replace(f'href="{LICENSE_URL}"'.encode(), b'href="https://other.example.test"')
    with pytest.raises(EvidenceFetchError, match="neither declares"):
        verify_evidence_pages(
            source_url=SOURCE_URL,
            source_snapshot=_snapshot(SOURCE_URL, unrelated),
            source_bytes=unrelated,
            license_url=LICENSE_URL,
            license_snapshot=_snapshot(LICENSE_URL, license_page),
            license_bytes=license_page,
            title="Verified Sprite Pack",
            creator="Example Artist",
            license_id="cc0-1.0",
            direct_download_url=DIRECT_URL,
        )

    scripted = _source_html(automation="<script>Automated downloads: allowed</script>", include_terms=False)
    silent = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=scripted,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(scripted).hexdigest(),
        terms_url=None,
    )
    assert silent.decision == "NO_PROHIBITION_OBSERVED"
    assert silent.limited_evidence is True
    blocked = _source_html(automation="Automated downloads: prohibited", include_terms=False)
    blocked_result = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=blocked,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(blocked).hexdigest(),
        terms_url=None,
    )
    assert blocked_result.decision == "BLOCK"
    assert blocked_result.limited_evidence is False


def test_evidence_verification_binds_zero_cost_and_license_to_one_pack_block() -> None:
    source = _source_html()
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    verified = verify_evidence_pages(
        source_url=SOURCE_URL,
        source_snapshot=_snapshot(SOURCE_URL, source),
        source_bytes=source,
        license_url=LICENSE_URL,
        license_snapshot=_snapshot(LICENSE_URL, license_page),
        license_bytes=license_page,
        title="Verified Sprite Pack",
        creator="Example Artist",
        license_id="cc0-1.0",
        direct_download_url=DIRECT_URL,
    )

    assert verified.zero_cost_verified is True
    assert verified.license_conflict_checked is True
    assert "Verified Sprite Pack" in verified.source_pack_evidence_text
    assert DIRECT_URL not in verified.source_pack_evidence_text


def test_evidence_verification_accepts_nested_wrappers_within_one_card() -> None:
    source = (
        "<html><body><article>"
        "<div><header><h1>Verified Sprite Pack</h1><p>by Example Artist</p></header></div>"
        f'<div><p>Free zero-cost download.</p><a href="{LICENSE_URL}">CC0 license</a>'
        f'<a href="{DIRECT_URL}">Direct download</a></div>'
        "</article></body></html>"
    ).encode()
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    verified = verify_evidence_pages(
        source_url=SOURCE_URL,
        source_snapshot=_snapshot(SOURCE_URL, source),
        source_bytes=source,
        license_url=LICENSE_URL,
        license_snapshot=_snapshot(LICENSE_URL, license_page),
        license_bytes=license_page,
        title="Verified Sprite Pack",
        creator="Example Artist",
        license_id="cc0-1.0",
        direct_download_url=DIRECT_URL,
    )

    assert verified.zero_cost_verified is True
    assert "Verified Sprite Pack" in verified.source_pack_evidence_text
    assert "Free zero-cost download." in verified.source_pack_evidence_text


def test_evidence_verification_rejects_license_wording_as_zero_cost_on_an_unbound_host() -> None:
    source = _source_html().replace(
        b"Free zero-cost download.",
        b"You can use these tilesets in your program freely.",
    )
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    with pytest.raises(EvidenceFetchError, match="conflict-free explicit zero-cost"):
        verify_evidence_pages(
            source_url=SOURCE_URL,
            source_snapshot=_snapshot(SOURCE_URL, source),
            source_bytes=source,
            license_url=LICENSE_URL,
            license_snapshot=_snapshot(LICENSE_URL, license_page),
            license_bytes=license_page,
            title="Verified Sprite Pack",
            creator="Example Artist",
            license_id="cc0-1.0",
            direct_download_url=DIRECT_URL,
        )


def test_evidence_verification_rejects_oga_cc0_and_direct_file_when_price_is_silent() -> None:
    source_url = "https://opengameart.org/content/dungeon-crawl-32x32-tiles-supplemental"
    license_url = "https://creativecommons.org/publicdomain/zero/1.0/"
    direct_url = "https://opengameart.org/sites/default/files/Dungeon%20Crawl%20Stone%20Soup%20Supplemental.zip"
    source = (
        '<html><body><div class="node node-art view-mode-full clearfix">'
        '<div class="field field-name-title">Dungeon Crawl 32x32 tiles supplemental</div>'
        '<div class="field field-name-author-submitter">Submitted by MedicineStorm</div>'
        '<div class="field field-name-field-art-licenses">'
        f'<a href="{license_url}">CC0 1.0 public domain</a></div>'
        '<div class="field field-name-body">'
        "You can use these tilesets in your program freely. No attribution is required.</div>"
        '<div class="field field-name-field-art-files">'
        f'<a href="{direct_url}">Dungeon Crawl Stone Soup Supplemental.zip</a></div>'
        "</div></body></html>"
    ).encode()
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    with pytest.raises(EvidenceFetchError, match="conflict-free explicit zero-cost"):
        verify_evidence_pages(
            source_url=source_url,
            source_snapshot=_snapshot(source_url, source),
            source_bytes=source,
            license_url=license_url,
            license_snapshot=_snapshot(license_url, license_page),
            license_bytes=license_page,
            title="Dungeon Crawl 32x32 tiles supplemental",
            creator="MedicineStorm",
            license_id="cc0-1.0",
            direct_download_url=direct_url,
        )


def test_opengameart_adapter_accepts_one_live_shaped_record_and_binds_manifest() -> None:
    source = _oga_source_html().replace(
        b"</body>",
        b'<aside class="patreon">Support us on Patreon for $5.</aside>'
        b'<section class="collections">Other collection: all rights reserved.</section>'
        b'<section class="comments">A commenter says this costs 500 points.</section></body>',
    )

    first = _verify_oga_source(source)
    second = _verify_oga_source(source)

    assert first.zero_cost_verified is True
    assert first.verification_identity == second.verification_identity
    assert "source-adapter: spritelab.harvest.source-adapter.opengameart.v1" in first.source_pack_evidence_text
    assert (
        "source-role-manifest: title=.field-name-title|author-submitter=.field-name-author-submitter|"
        "art-licenses=.field-name-field-art-licenses|body=.field-name-body|"
        "art-files=.field-name-field-art-files" in first.source_pack_evidence_text
    )
    assert "Patreon" not in first.source_pack_evidence_text
    assert "collection" not in first.source_pack_evidence_text
    assert "commenter" not in first.source_pack_evidence_text


def test_opengameart_adapter_upgrades_only_the_live_legacy_cc0_deed_link() -> None:
    source = _oga_source_html().replace(
        OGA_LICENSE_URL.encode(),
        b"http://creativecommons.org/publicdomain/zero/1.0/",
        1,
    )

    verified = _verify_oga_source(source)

    assert verified.zero_cost_verified is True


def test_opengameart_adapter_accepts_the_closed_live_legacy_license_selector() -> None:
    selector = (
        b'<a href="http://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>'
        b'<a href="http://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a>'
        b'<a href="http://www.gnu.org/licenses/gpl-3.0.html">GPL 3.0</a>'
        b'<a href="http://www.gnu.org/licenses/old-licenses/gpl-2.0.html">GPL 2.0</a>'
        b'<a href="http://opengameart.org/content/oga-by-30-faq">OGA-BY 3.0</a>'
        b'<a href="http://creativecommons.org/publicdomain/zero/1.0/">CC0 1.0 public domain</a>'
    )
    source = _oga_source_html().replace(
        f'<a href="{OGA_LICENSE_URL}">CC0 1.0 public domain</a>'.encode(),
        selector,
        1,
    )

    verified = _verify_oga_source(source)

    assert verified.zero_cost_verified is True


def test_opengameart_adapter_allows_only_the_live_legacy_opp_body_link() -> None:
    source = _oga_source_html().replace(
        b"in the public domain.</div>",
        b'in the public domain. <a href="http://openpixelproject.com">Open Pixel Project</a></div>',
        1,
    )

    verified = _verify_oga_source(source)

    assert verified.zero_cost_verified is True


def test_opengameart_adapter_rejects_other_insecure_role_links() -> None:
    source = _oga_source_html().replace(
        OGA_LICENSE_URL.encode(),
        b"http://creativecommons.org/licenses/by/4.0/",
        1,
    )

    with pytest.raises(EvidenceFetchError, match="one visible, nonnested full-art node"):
        _verify_oga_source(source)


def test_opengameart_adapter_does_not_take_positive_evidence_from_unknown_fields() -> None:
    source = _oga_source_html().replace(
        b"Hey guys, I have been spriting for some years now, im uploading my archive totally free "
        b"in the public domain.</div>",
        b"Visible neutral description.</div>"
        b'<div class="field field-name-comments">The assets are available for free.</div>',
        1,
    )

    with pytest.raises(EvidenceFetchError, match="conflict-free explicit zero-cost"):
        _verify_oga_source(source)


def test_opengameart_adapter_fails_closed_above_the_visible_link_limit() -> None:
    excess_links = b"".join(b'<a href="https://example.test/preview">preview</a>' for _ in range(20_001))
    source = _oga_source_html().replace(b"</body>", excess_links + b"</body>", 1)

    with pytest.raises(EvidenceFetchError, match="one visible, nonnested full-art node"):
        _verify_oga_source(source)


def test_opengameart_adapter_rejects_duplicate_full_art_roots() -> None:
    source = _oga_source_html()
    node = source.removeprefix(b"<html><body>").removesuffix(b"</body></html>")
    duplicate = b"<html><body>" + node + node + b"</body></html>"

    with pytest.raises(EvidenceFetchError, match="one visible, nonnested full-art node"):
        _verify_oga_source(duplicate)


def test_opengameart_adapter_rejects_cross_record_role_composition() -> None:
    source = _oga_source_html()
    first = source.replace(
        b"field-name-field-art-files field-type-file",
        b"field-name-field-art-files-missing field-type-file",
    )
    second = source.replace(b"field-name-title field-type-ds", b"field-name-title-missing field-type-ds")
    first_node = first.removeprefix(b"<html><body>").removesuffix(b"</body></html>")
    second_node = second.removeprefix(b"<html><body>").removesuffix(b"</body></html>")
    cross_record = b"<html><body>" + first_node + second_node + b"</body></html>"

    with pytest.raises(EvidenceFetchError, match="one visible, nonnested full-art node"):
        _verify_oga_source(cross_record)


def test_opengameart_adapter_rejects_duplicate_authoritative_role() -> None:
    source = _oga_source_html().replace(
        b'<div class="field field-name-title field-type-ds field-label-hidden">',
        b'<div class="field field-name-title field-type-ds">Attacker title</div>'
        b'<div class="field field-name-title field-type-ds field-label-hidden">',
        1,
    )

    with pytest.raises(EvidenceFetchError, match="one exact visible field for every role"):
        _verify_oga_source(source)


def test_opengameart_adapter_uses_only_visible_positive_evidence() -> None:
    source = (
        _oga_source_html()
        .replace(
            b'<div class="field field-name-body field-type-text-with-summary">',
            b'<div class="field field-name-body field-type-text-with-summary">Visible description.'
            b'<span style="display: none">',
            1,
        )
        .replace(b"in the public domain.</div>", b"in the public domain.</span></div>", 1)
    )

    with pytest.raises(EvidenceFetchError, match="conflict-free explicit zero-cost"):
        _verify_oga_source(source)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    (
        (
            b"in the public domain.</div>",
            b"in the public domain. Price: $5.</div>",
            "conflict-free explicit zero-cost",
        ),
        (
            b"CC0 1.0 public domain</a>",
            b"CC0 1.0 public domain; all rights reserved</a>",
            "art-licenses field contains conflicting",
        ),
    ),
)
def test_opengameart_adapter_rejects_conflicts_in_authoritative_roles(
    needle: bytes,
    replacement: bytes,
    message: str,
) -> None:
    source = _oga_source_html().replace(needle, replacement, 1)

    with pytest.raises(EvidenceFetchError, match=message):
        _verify_oga_source(source)


def test_opengameart_adapter_rejects_repeated_direct_link_in_files() -> None:
    repeated = _oga_source_html().replace(
        b"</a></div></div></body></html>",
        f'</a><a href="{OGA_DIRECT_URL}">mirror</a></div></div></body></html>'.encode(),
        1,
    )

    with pytest.raises(EvidenceFetchError, match="once and only in art-files"):
        _verify_oga_source(repeated)


def test_opengameart_profile_rejects_noncanonical_final_host_without_fallback() -> None:
    source = _oga_source_html()
    snapshot = _snapshot(OGA_SOURCE_URL, source)
    attacker_snapshot = replace(snapshot, final_url="https://opengameart.org.attacker.test/content/forged")

    with pytest.raises(EvidenceFetchError, match="final URL is not a canonical content-detail page"):
        _verify_oga_source(source, source_snapshot=attacker_snapshot)


def test_opengameart_adapter_is_not_selected_by_an_attacker_hostname() -> None:
    source = _oga_source_html()
    attacker_url = "https://opengameart.org.attacker.test/content/forged"

    verified = _verify_oga_source(source, source_url=attacker_url)

    assert "source-adapter:" not in verified.source_pack_evidence_text


@pytest.mark.parametrize(
    "declaration",
    (
        "Free high quality pixel art tiles, 32x32.",
        "The assets are available for free.",
        "Over 1000 game components for free!",
        "A series of 100 FREE Tiny Top Down Tiles to use in your games.",
        "This pack includes 170 sprites. ALL FOR FREE.",
        "Included are 50 FREE high quality epic weapon sprites.",
        "im uploading my archive totally free in the public domain.",
        "Hey guys, I have been spriting for some years now, im uploading my archive totally free in the public domain.",
        "The assets are available for free. Focal points guide the sprite composition.",
    ),
)
def test_evidence_verification_accepts_explicit_gratis_pack_wording(declaration: str) -> None:
    source = _source_html().replace(b"Free zero-cost download.", declaration.encode())
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    verified = verify_evidence_pages(
        source_url=SOURCE_URL,
        source_snapshot=_snapshot(SOURCE_URL, source),
        source_bytes=source,
        license_url=LICENSE_URL,
        license_snapshot=_snapshot(LICENSE_URL, license_page),
        license_bytes=license_page,
        title="Verified Sprite Pack",
        creator="Example Artist",
        license_id="cc0-1.0",
        direct_download_url=DIRECT_URL,
    )

    assert verified.zero_cost_verified is True
    assert declaration in verified.source_pack_evidence_text


@pytest.mark.parametrize(
    "declaration",
    (
        "A series of 100 royalty-free tiles.",
        "A series of 100 FREE trial tiles.",
        "A series of 100 FREE tiles after purchase.",
        "The assets are not available for free.",
        "The assets are no longer available for free.",
        "The assets were available for free.",
        "The assets are not all for free.",
        "The documentation is available for free.",
        "The preview is available for free.",
        "The pack was zero-cost.",
        "The pack is no longer zero-cost.",
        "Formerly a free download.",
        "This is not a free download.",
        "Not every asset is available for free.",
        "Only some assets are available for free.",
        "Not all assets are for free.",
        "The documentation is a free download.",
        "Documentation price: 0.",
        "100 free tiles out of 1000.",
        "Assets are available for free; a fee is required.",
        "Free download; a donation is required.",
        "Assets are available for free to members.",
        "Formerly a free pack, now costs 5 credits.",
        "Free asset; payment is required.",
        "No assets are available for free.",
        "None of the assets are available for free.",
        "Half the assets are available for free.",
        "10 of 100 assets are available for free.",
        "Assets are available for free shipping.",
        "Assets are available for free previews.",
        "A surcharge applies; assets are available for free.",
        "Assets are available for free; handling is required.",
        "Assets are available for free after you pay.",
        "Assets are available for free; a tip is required.",
        "Assets are available for free after a contribution.",
        "Assets are available for free to Patreon patrons.",
        "Assets are available for free with a premium account.",
        "Assets are available for free for five tokens.",
        "Assets are available for free for 500 points.",
        "Assets are available for free? No.",
        "Assets are available for free. Not anymore.",
        "Assets are available for free. This offer expired yesterday.",
        "Assets are available for free. Only the first ten qualify.",
        "Assets are available for free. A surcharge applies.",
        "Assets are available for free. You must pay at checkout.",
        "Assets are available for free. A tip is required.",
        "Assets are available for free. A contribution is mandatory.",
        "Assets are available for free. Access is limited to Patreon patrons.",
        "Assets are available for free. A premium account is required.",
        "Assets are available for free. Redeem five tokens.",
        "Assets are available for free. 500 points are required.",
        "Assets are available for free. A coupon code is required.",
    ),
)
def test_evidence_verification_rejects_ambiguous_or_paid_counted_free_wording(declaration: str) -> None:
    source = _source_html().replace(b"Free zero-cost download.", declaration.encode())
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    with pytest.raises(EvidenceFetchError, match="conflict-free explicit zero-cost"):
        verify_evidence_pages(
            source_url=SOURCE_URL,
            source_snapshot=_snapshot(SOURCE_URL, source),
            source_bytes=source,
            license_url=LICENSE_URL,
            license_snapshot=_snapshot(LICENSE_URL, license_page),
            license_bytes=license_page,
            title="Verified Sprite Pack",
            creator="Example Artist",
            license_id="cc0-1.0",
            direct_download_url=DIRECT_URL,
        )


@pytest.mark.parametrize(
    "qualifier",
    (
        "This offer expired yesterday.",
        "Only the first ten qualify.",
        "You must pay at checkout.",
        "A contribution is mandatory.",
        "A premium account is required.",
    ),
)
def test_evidence_verification_rejects_later_sibling_price_qualifier(qualifier: str) -> None:
    source = _source_html().replace(
        b"<p>Free zero-cost download.</p>",
        f"<p>Assets are available for free.</p><p>{qualifier}</p>".encode(),
    )
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    with pytest.raises(EvidenceFetchError, match="conflict-free explicit zero-cost"):
        verify_evidence_pages(
            source_url=SOURCE_URL,
            source_snapshot=_snapshot(SOURCE_URL, source),
            source_bytes=source,
            license_url=LICENSE_URL,
            license_snapshot=_snapshot(LICENSE_URL, license_page),
            license_bytes=license_page,
            title="Verified Sprite Pack",
            creator="Example Artist",
            license_id="cc0-1.0",
            direct_download_url=DIRECT_URL,
        )


@pytest.mark.parametrize(
    ("source", "license_page", "message"),
    [
        (
            _source_html().replace(b"CC0 license", b"CC0 1.0 - all rights reserved"),
            b"<p>CC0 1.0 public domain dedication</p>",
            "source-pack block contains conflicting",
        ),
        (
            _source_html().replace(b"CC0 license", b"not CC0 1.0"),
            b"<p>CC0 1.0 public domain dedication</p>",
            "source-pack block contains conflicting",
        ),
        (
            _source_html().replace(b"Free zero-cost download.", b"Free zero-cost download. Price: $5"),
            b"<p>CC0 1.0 public domain dedication</p>",
            "conflict-free explicit zero-cost",
        ),
        (
            _source_html().replace(
                b"Free zero-cost download.",
                b"You can use these tilesets in your program freely after purchase.",
            ),
            b"<p>CC0 1.0 public domain dedication</p>",
            "conflict-free explicit zero-cost",
        ),
        (
            (
                "<html><body><article><h1>Verified Sprite Pack</h1><p>by Example Artist</p>"
                "<p>You can use these tilesets in your program freely.</p></article><article>"
                f'<a href="{LICENSE_URL}">CC0 1.0</a><a href="{DIRECT_URL}">Direct download</a>'
                "</article></body></html>"
            ).encode(),
            b"<p>CC0 1.0 public domain dedication</p>",
            "bound to one source-pack block",
        ),
        (
            (
                f"<html><body><article><h1>Verified Sprite Pack</h1><p>by Example Artist</p>"
                "<p>Free zero-cost download.</p></article><article>"
                f'<a href="{LICENSE_URL}">CC0 1.0</a><a href="{DIRECT_URL}">Direct download</a>'
                "</article></body></html>"
            ).encode(),
            b"<p>CC0 1.0 public domain dedication</p>",
            "bound to one source-pack block",
        ),
        (
            _source_html(),
            b"<p>CC0 1.0 public domain dedication. All rights reserved.</p>",
            "license page contains conflicting",
        ),
        (
            (
                "<html><body><div>"
                "<article><h1>Verified Sprite Pack</h1><p>by Example Artist</p>"
                "<p>Free zero-cost download.</p></article>"
                f'<article><h2>Other Pack</h2><p>by Other Artist</p><a href="{LICENSE_URL}">CC0 1.0</a>'
                f'<a href="{DIRECT_URL}">Direct download</a></article>'
                "</div></body></html>"
            ).encode(),
            b"<p>CC0 1.0 public domain dedication</p>",
            "composes provenance",
        ),
        (
            (
                "<html><body><div>"
                "<div><article><h1>Verified Sprite Pack</h1><p>by Example Artist</p></article></div>"
                f'<div><p>Free zero-cost download.</p><a href="{LICENSE_URL}">CC0 1.0</a>'
                f'<a href="{DIRECT_URL}">Direct download</a></div>'
                "</div></body></html>"
            ).encode(),
            b"<p>CC0 1.0 public domain dedication</p>",
            "composes provenance",
        ),
        (
            (
                "<html><body>"
                "<article><h1>Verified Sprite Pack</h1><p>by Example Artist</p>"
                f'<p>Free zero-cost download.</p><a href="{LICENSE_URL}">CC0 license</a>'
                f'<a href="{DIRECT_URL}">Direct download</a></article>'
                f'<article><h2>Mirror</h2><a href="{DIRECT_URL}">Mirror download</a></article>'
                "</body></html>"
            ).encode(),
            b"<p>CC0 1.0 public domain dedication</p>",
            "multiple distinct source-page blocks",
        ),
        (
            (
                "<html><body><div>"
                "<article><h1>Verified Sprite Pack</h1><p>by Example Artist</p>"
                f'<p>Free zero-cost download.</p><a href="{LICENSE_URL}">CC0 license</a></article>'
                f'<article><h2>Other Pack</h2><a href="{LICENSE_URL}">CC0 license</a>'
                f'<a href="{DIRECT_URL}">Direct download</a></article>'
                "</div></body></html>"
            ).encode(),
            b"<p>CC0 1.0 public domain dedication</p>",
            "composes provenance",
        ),
    ],
)
def test_evidence_verification_rejects_ambiguous_paid_or_conflicting_pack_evidence(
    source: bytes,
    license_page: bytes,
    message: str,
) -> None:
    with pytest.raises(EvidenceFetchError, match=message):
        verify_evidence_pages(
            source_url=SOURCE_URL,
            source_snapshot=_snapshot(SOURCE_URL, source),
            source_bytes=source,
            license_url=LICENSE_URL,
            license_snapshot=_snapshot(LICENSE_URL, license_page),
            license_bytes=license_page,
            title="Verified Sprite Pack",
            creator="Example Artist",
            license_id="cc0-1.0",
            direct_download_url=DIRECT_URL,
        )


@pytest.mark.parametrize(
    "declaration",
    (
        "Free use is permitted.",
        "Royalty-free tileset.",
        "Direct download available.",
    ),
)
def test_evidence_verification_does_not_infer_zero_cost_from_generic_free_or_price_silence(
    declaration: str,
) -> None:
    source = _source_html().replace(b"Free zero-cost download.", declaration.encode())
    license_page = b"<html><body><p>CC0 1.0 public domain dedication</p></body></html>"

    with pytest.raises(EvidenceFetchError, match="conflict-free explicit zero-cost"):
        verify_evidence_pages(
            source_url=SOURCE_URL,
            source_snapshot=_snapshot(SOURCE_URL, source),
            source_bytes=source,
            license_url=LICENSE_URL,
            license_snapshot=_snapshot(LICENSE_URL, license_page),
            license_bytes=license_page,
            title="Verified Sprite Pack",
            creator="Example Artist",
            license_id="cc0-1.0",
            direct_download_url=DIRECT_URL,
        )


@pytest.mark.parametrize(
    "declaration",
    [
        "Automated downloading is not permitted",
        "Scraping/crawling is prohibited",
        "Bots are forbidden",
        "Use of automated means is prohibited",
        "May not access/use the service through automated or non-human means",
        "No robots/spiders/scrapers",
        "Systematic/bulk downloading is prohibited",
        "Automated data collection/extraction/mining prohibited",
        "You must not use scripts to download",
    ],
)
def test_automation_terms_broad_explicit_prohibitions_block(declaration: str) -> None:
    page = _source_html(automation=declaration, include_terms=False)
    result = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=page,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(page).hexdigest(),
        terms_url=None,
    )
    assert result.decision == "BLOCK"


@pytest.mark.parametrize(
    "declaration",
    [
        "No automated downloads are prohibited",
        "Automated downloading is not prohibited",
        "We do not prohibit scraping",
        "Systematic downloading is not prohibited",
        "Automated data collection is not prohibited",
        "We do not prohibit the use of scripts to download",
        "This article discusses bots and prohibited content",
    ],
)
def test_automation_terms_harmless_and_double_negative_text_does_not_block(declaration: str) -> None:
    page = _source_html(automation=declaration, include_terms=False)
    result = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=page,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(page).hexdigest(),
        terms_url=None,
    )
    assert result.decision == "NO_PROHIBITION_OBSERVED"
    assert result.limited_evidence is True


@pytest.mark.parametrize(
    "page",
    [
        b"<html><body>Automated downloads: prohibited</body></html>",
        b"<html><body><main>Automated downloads: prohibited</main></body></html>",
        b"<html><body><table><tr><td>Automated downloads: prohibited</td></tr></table></body></html>",
        b"<html><body><main>Automated <span>downloads</span>: prohibited</main></body></html>",
        b"<html><body><div>Automated downloads: prohibited",
    ],
)
def test_automation_prohibition_scans_all_visible_html_text(page: bytes) -> None:
    result = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=page,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(page).hexdigest(),
        terms_url=None,
    )
    assert result.decision == "BLOCK"


@pytest.mark.parametrize(
    "declaration",
    [
        "May not access/use the service through automated or non-human means",
        "No robots/spiders/scrapers",
        "Systematic/bulk downloading is prohibited",
        "Automated data collection/extraction/mining prohibited",
        "You must not use scripts to download",
    ],
)
def test_linked_governing_terms_paraphrases_block(declaration: str) -> None:
    terms = f"<html><body><main>{declaration}</main></body></html>".encode()
    result = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=_source_html(),
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(_source_html()).hexdigest(),
        terms_url=TERMS_URL,
        terms_bytes=terms,
        terms_snapshot=_snapshot(TERMS_URL, terms, "text/html"),
    )
    assert result.decision == "BLOCK"


@pytest.mark.parametrize(
    ("label", "terms_url"),
    [
        ("Privacy & Terms", "https://catalog.example.test/policy-center"),
        ("Terms / Privacy", "https://catalog.example.test/legal-hub"),
        ("Conditions of Use", "https://catalog.example.test/about"),
        ("User Agreement", "https://catalog.example.test/agreement-hub"),
        ("Website Terms", "https://catalog.example.test/policy"),
        ("Terms & Policies", "https://catalog.example.test/governance"),
        ("Policy", "https://catalog.example.test/conditions-of-use"),
        ("Policy", "https://catalog.example.test/user-agreement"),
        ("Policy", "https://catalog.example.test/website-terms"),
    ],
)
def test_common_governing_terms_labels_and_paths_require_linked_review(label: str, terms_url: str) -> None:
    source = _source_html(
        include_terms=False,
        extra_terms_link=f'<a href="{terms_url}">{label}</a>',
    )
    with pytest.raises(EvidenceFetchError, match="governing terms"):
        verify_automation_terms(
            source_url=SOURCE_URL,
            source_bytes=source,
            source_mime_type="text/html",
            source_content_sha256=hashlib.sha256(source).hexdigest(),
            terms_url=None,
        )

    terms = b"Automated downloads: allowed"
    result = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=source,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(source).hexdigest(),
        terms_url=terms_url,
        terms_bytes=terms,
        terms_snapshot=_snapshot(terms_url, terms, "text/plain"),
    )
    assert result.decision == "ALLOW"


def test_governing_terms_link_is_required_unambiguous_and_not_a_license_legal_code() -> None:
    linked = _source_html()
    with pytest.raises(EvidenceFetchError, match="governing terms"):
        verify_automation_terms(
            source_url=SOURCE_URL,
            source_bytes=linked,
            source_mime_type="text/html",
            source_content_sha256=hashlib.sha256(linked).hexdigest(),
            terms_url=None,
        )

    license_only = _source_html(include_terms=False)
    with pytest.raises(EvidenceFetchError, match="independently detected"):
        verify_automation_terms(
            source_url=SOURCE_URL,
            source_bytes=license_only,
            source_mime_type="text/html",
            source_content_sha256=hashlib.sha256(license_only).hexdigest(),
            terms_url=LICENSE_URL,
            terms_bytes=b"CC0 1.0 Universal",
            terms_snapshot=_snapshot(LICENSE_URL, b"CC0 1.0 Universal", "text/plain"),
        )

    multiple = _source_html(
        extra_terms_link='<a href="https://catalog.example.test/terms-of-service">Terms of service</a>'
    )
    with pytest.raises(EvidenceFetchError, match="multiple"):
        verify_automation_terms(
            source_url=SOURCE_URL,
            source_bytes=multiple,
            source_mime_type="text/html",
            source_content_sha256=hashlib.sha256(multiple).hexdigest(),
            terms_url=TERMS_URL,
            terms_bytes=b"Automated downloads: allowed",
            terms_snapshot=_snapshot(TERMS_URL, b"Automated downloads: allowed", "text/plain"),
        )

    legal_code = _source_html(
        include_terms=False,
        extra_terms_link='<a href="https://catalog.example.test/licenses/cc0/legalcode">Legal Code</a>',
    )
    result = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=legal_code,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(legal_code).hexdigest(),
        terms_url=None,
    )
    assert result.decision == "NO_PROHIBITION_OBSERVED"

    source_block = _source_html(automation="Bots are forbidden")
    allow_terms = b"Automated downloads: allowed"
    blocked = verify_automation_terms(
        source_url=SOURCE_URL,
        source_bytes=source_block,
        source_mime_type="text/html",
        source_content_sha256=hashlib.sha256(source_block).hexdigest(),
        terms_url=TERMS_URL,
        terms_bytes=allow_terms,
        terms_snapshot=_snapshot(TERMS_URL, allow_terms, "text/plain"),
    )
    assert blocked.decision == "BLOCK"


def test_robots_missing_policy_and_explicit_disallow_are_deterministic(tmp_path: Path) -> None:
    with AnchoredDirectory(tmp_path, tmp_path) as anchor:
        missing = fetch_robots_snapshot(
            SOURCE_URL,
            anchor,
            "missing.txt",
            cancel_requested=lambda: False,
            resolver=lambda _host, _port: ("8.8.8.8",),
            transport=FakeTransport([FakeResponse(b"missing", status=404)]),
        )
        assert missing.policy == "missing_policy_allow"
        assert missing.evaluate(SOURCE_URL).allowed is True

        policy = b"User-agent: spritelab-harvest\nDisallow: /source\nAllow: /\n"
        denied = fetch_robots_snapshot(
            SOURCE_URL,
            anchor,
            "denied.txt",
            cancel_requested=lambda: False,
            resolver=lambda _host, _port: ("8.8.8.8",),
            transport=FakeTransport([FakeResponse(policy)]),
        )
        with pytest.raises(EvidenceFetchError, match="disallows"):
            denied.evaluate(SOURCE_URL)

        malformed = b"Disallow: /\n"
        with pytest.raises(EvidenceFetchError, match="no user-agent"):
            fetch_robots_snapshot(
                SOURCE_URL,
                anchor,
                "malformed.txt",
                cancel_requested=lambda: False,
                resolver=lambda _host, _port: ("8.8.8.8",),
                transport=FakeTransport([FakeResponse(malformed)]),
            )


def test_robots_comment_after_user_agent_does_not_end_group(tmp_path: Path) -> None:
    with AnchoredDirectory(tmp_path, tmp_path) as anchor:
        policy = fetch_robots_snapshot(
            OGA_SOURCE_URL,
            anchor,
            "oga-commented.txt",
            cancel_requested=lambda: False,
            resolver=lambda _host, _port: ("8.8.8.8",),
            transport=FakeTransport([FakeResponse(OGA_ROBOTS_COMMENTED_RULES)]),
        )

        allowed = policy.evaluate("https://opengameart.org/misc/sprite.css")
        assert allowed.matched_directive == "allow"
        assert allowed.matched_pattern == "/misc/*.css$"
        with pytest.raises(EvidenceFetchError, match="disallows"):
            policy.evaluate("https://opengameart.org/misc/private.txt")


def test_project_wide_single_flight_survives_two_service_instances(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    entered = threading.Event()
    release = threading.Event()
    remaining = _responses()
    first_transport = FakeTransport([BlockingResponse(ROBOTS_ALLOW, entered, release), *remaining[1:]])
    first, evidence = _service(project, first_transport)
    started, _created = first.start_probe(_payload(first, evidence, key="probe-race-first"))
    assert entered.wait(3)

    second_transport = FakeTransport([])
    second, second_evidence = _service(project, second_transport)
    second_payload = _payload(second, second_evidence, key="probe-race-second")
    with pytest.raises(HarvestError, match="already active") as error:
        second.start_probe(second_payload)
    assert error.value.code == "harvest_single_flight_conflict"
    assert second_transport.calls == []
    release.set()
    assert _wait(first, started["probe_id"], {"READY", "FAILED"})["status"] == "READY"


@pytest.mark.parametrize("name", ["harvest-corrupt1", f"probe-{'b' * 32}"])
def test_probe_single_flight_fails_closed_for_corrupt_managed_names(tmp_path: Path, name: str) -> None:
    project = tmp_path / "project"
    entry = project / "harvest_runs" / name
    entry.mkdir(parents=True)
    (entry / "sources.jsonl").write_text('{"source_id":"plausible-legacy"}\n', encoding="utf-8")
    service, _evidence_record = _service(project, FakeTransport([]))

    assert service._probe_service._active_single_flight() == name


def test_promotion_rejects_linked_terms_page_drift(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, evidence = _service(project, FakeTransport(_responses()))
    started, _created = service.start_probe(_payload(service, evidence, key="probe-terms-drift-0001"))
    assert _wait(service, started["probe_id"], {"READY", "FAILED"})["status"] == "READY"
    terms = project / "harvest_runs" / started["probe_id"] / "evidence" / "terms_page.bin"
    terms.write_bytes(b"<html><body><p>Bots are forbidden</p></body></html>")

    with pytest.raises(HarvestError, match="changed"):
        service.promote_probe(
            started["probe_id"],
            explicit_action=True,
            authorize_catalog_promotion=True,
            **_promotion_review(service, started["probe_id"]),
        )
    assert load_trusted_catalog(project) == ()


def test_promotion_rejects_hardlinked_raw_quarantine(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, evidence = _service(project, FakeTransport(_responses()))
    started, _created = service.start_probe(_payload(service, evidence, key="probe-tamper-0001"))
    assert _wait(service, started["probe_id"], {"READY", "FAILED"})["status"] == "READY"
    raw = project / "harvest_runs" / started["probe_id"] / "quarantine" / "raw_payload.bin"
    os.link(raw, raw.with_name("attacker-link.bin"))

    with pytest.raises(HarvestError, match=r"changed|unsafe"):
        service.promote_probe(
            started["probe_id"],
            explicit_action=True,
            authorize_catalog_promotion=True,
            **_promotion_review(service, started["probe_id"]),
        )
    assert load_trusted_catalog(project) == ()


def test_catalog_onboarding_ui_is_visible_and_rejects_browser_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    capabilities = _capabilities()
    evidence = _evidence(capabilities)
    plugin = create_plugin(
        sources=(),
        backend_factory=lambda: None,
        backend_capabilities=capabilities,
        backend_capability_evidence=evidence,
        probe_resolver=lambda _host, _port: ("8.8.8.8",),
        probe_transport=FakeTransport([]),
        allow_unverified_test_backend=True,
    )
    app = create_app(ProjectContext(project), plugins=(plugin,))
    client = TestClient(app)
    page = client.get("/harvest")
    assert page.status_code == 200
    assert "Start bounded source probe" in page.text
    assert 'aria-describedby="harvest-probe-availability harvest-probe-progress-status"' in page.text
    assert 'id="harvest-probe-progress-bar"' in page.text
    assert 'aria-labelledby="harvest-probe-progress-status"' in page.text
    assert "Checking whether the certified source-probe backend is available" in page.text
    assert "Automation terms URL" in page.text
    assert "Leave blank only when no governing Terms or ToS link exists" in page.text
    javascript = client.get("/harvest/static/harvest.js").text
    assert "/harvest/api/probes" in javascript
    assert "authorize_catalog_promotion" in javascript
    assert "authorize_zero_cost_evidence_review" in javascript
    assert "reviewed_verification_identity" in javascript
    assert "reviewed_source_pack_evidence_sha256" in javascript
    assert "Automation terms decision" in javascript
    assert 'beginLaunchProgress("probe")' in javascript
    assert 'updateLaunchProgress("probe", value.probe, "probe_id")' in javascript
    assert 'failLaunchProgress("probe", error.message)' in javascript
    assert "no prohibition observed; not affirmative permission" in javascript
    assert "spritelab.harvest.pending-idempotency.v1:" in javascript
    assert (
        "Source probing is unavailable because current independent backend capability evidence is missing or invalid."
        in javascript
    )
    assert "Source probing remains unavailable because backend capability evidence could not be verified:" in javascript
    assert "window.sessionStorage.setItem(storageKey, value)" in javascript
    assert "window.sessionStorage.removeItem(storageKey)" in javascript
    assert "if (error.definitive) clearIdempotency(scope)" in javascript
    assert "sessionStorage.setItem(storageKey, JSON.stringify" not in javascript
    assert ".innerHTML" not in javascript

    denied = client.post(
        "/harvest/api/probes",
        json={"output_path": "C:/private"},
        headers={"X-CSRF-Token": app.state.spritelab_csrf_token},
    )
    assert denied.status_code == 422
    assert denied.json()["error_code"] == "browser_path_not_allowed"


def test_repository_certificate_bootstraps_first_catalog_probe_without_passive_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    capabilities = _capabilities()
    evidence = _evidence(capabilities)
    evidence_loads: list[Path] = []

    def load_evidence(root: Path) -> BackendCapabilityEvidence:
        evidence_loads.append(root)
        return evidence

    monkeypatch.setattr(harvest_plugin_module, "load_trusted_catalog", lambda _root: ())
    monkeypatch.setattr(harvest_plugin_module, "load_backend_capability_evidence", load_evidence)
    monkeypatch.setattr(
        harvest_service_module,
        "hardened_backend_code_identity",
        lambda: capabilities.code_identity_sha256,
    )
    monkeypatch.setattr(
        harvest_service_module,
        "conditioned_dataset_import_callback_binding",
        lambda: {
            "dataset_import_callback_id": capabilities.dataset_import_callback_id,
            "dataset_import_callback_code_identity_sha256": (capabilities.dataset_import_callback_code_identity_sha256),
            "dataset_import_callback_runtime_identity_sha256": (
                capabilities.dataset_import_callback_runtime_identity_sha256
            ),
        },
    )
    plugin = create_plugin(
        load_repository_capabilities=True,
        probe_resolver=lambda _host, _port: ("8.8.8.8",),
        probe_transport=FakeTransport(_responses()),
    )
    app = create_app(ProjectContext(project), plugins=(plugin,))
    client = TestClient(app)
    assert evidence_loads == []

    inventory = client.get("/harvest/api/inventory").json()
    assert evidence_loads == []
    catalog = client.get("/harvest/api/sources").json()
    assert evidence_loads == [project]
    assert client.get("/harvest/api/sources").status_code == 200
    assert evidence_loads == [project]
    assert catalog["sources"] == []
    assert catalog["backend_configured"] is False
    assert catalog["backend_capability_evidence"]["evidence_identity"] == evidence.identity
    assert list(project.iterdir()) == []

    response = client.post(
        "/harvest/api/probes",
        json={
            "source_id": "verified.pack",
            "title": "Verified Sprite Pack",
            "creator": "Example Artist",
            "source_page": SOURCE_URL,
            "license_id": "cc0-1.0",
            "license_evidence_url": LICENSE_URL,
            "terms_evidence_url": TERMS_URL,
            "direct_download_url": DIRECT_URL,
            "attribution_text": "Example Artist - Verified Sprite Pack",
            "taxonomy_hints": ["item"],
            "inventory_identity": inventory["inventory_identity"],
            "backend_capability_evidence_identity": evidence.identity,
            "idempotency_key": "repository-first-probe-0001",
            "explicit_action": True,
            "authorize_network": True,
            "authorize_hash_probe": True,
            "authorize_zero_cost": True,
            "authorize_permissive_license": True,
        },
        headers={"X-CSRF-Token": app.state.spritelab_csrf_token},
    )
    assert response.status_code == 202
    assert evidence_loads == [project, project]
    probe_id = response.json()["probe"]["probe_id"]
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        probe = client.get(f"/harvest/api/probes/{probe_id}").json()
        if probe["status"] in {"READY", "FAILED"}:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("repository-backed first-source probe did not finish")
    assert probe["status"] == "READY"
    assert load_trusted_catalog(project) == ()


@pytest.mark.parametrize("reason", ["invalid", "stale", "mismatched"])
def test_empty_catalog_rejects_unsafe_repository_capability_evidence_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(harvest_plugin_module, "load_trusted_catalog", lambda _root: ())

    def reject_evidence(_root: Path) -> BackendCapabilityEvidence:
        raise BackendCapabilityCertificateError(f"{reason} Harvest capability evidence")

    monkeypatch.setattr(harvest_plugin_module, "load_backend_capability_evidence", reject_evidence)
    plugin = create_plugin(load_repository_capabilities=True)
    status = plugin.status_provider(ProjectContext(project))
    assert status.status is ProductStatus.READY
    assert status.data["backend_certification_validation"] == "deferred"
    app = create_app(ProjectContext(project), plugins=(plugin,))
    catalog = TestClient(app).get("/harvest/api/sources").json()
    assert catalog["sources"] == []
    assert catalog["backend_capabilities"] is None
    assert catalog["backend_capability_evidence"] is None
    assert list(project.iterdir()) == []


def test_probe_cancellation_during_terminal_commit_prevents_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable cancellation landing during the commit lock wait must win."""

    project = tmp_path / "project"
    project.mkdir()
    transport = FakeTransport(_responses())
    capabilities = _capabilities()
    evidence = _evidence(capabilities)
    probe_id = "probe-" + "c" * 32
    service = HarvestService(
        project,
        backend_factory=lambda: None,
        backend_capabilities=capabilities,
        backend_capability_evidence=evidence,
        probe_resolver=lambda _host, _port: ("8.8.8.8",),
        probe_transport=transport,
        probe_run_id_factory=lambda: probe_id,
        allow_unverified_test_backend=True,
    )
    armed = threading.Event()
    real_identity = onboarding_module.catalog_evidence_verifier_code_identity

    def arming_identity() -> str:
        # The worker computes this exactly once while assembling its receipt,
        # after the last network action and before the terminal-commit lock.
        armed.set()
        return real_identity()

    real_lock = onboarding_module.RepositoryMutationLock

    class InjectingLock(real_lock):  # type: ignore[misc,valid-type]
        def __enter__(self) -> Any:
            if armed.is_set() and threading.current_thread().name == f"spritelab-{probe_id}":
                armed.clear()
                (project / "harvest_runs" / probe_id / "cancellation_request.json").write_text(
                    json.dumps(
                        {
                            "schema_version": onboarding_module.PROBE_CANCELLATION_SCHEMA,
                            "probe_id": probe_id,
                            "requested_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            "explicit_action": True,
                            "paths_exposed": False,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
            return super().__enter__()

    monkeypatch.setattr(onboarding_module, "catalog_evidence_verifier_code_identity", arming_identity)
    monkeypatch.setattr(onboarding_module, "RepositoryMutationLock", InjectingLock)

    started, created = service.start_probe(_payload(service, evidence, key="probe-commit-cancel-01"))
    assert created is True
    final = _wait(service, started["probe_id"], {"CANCELLED", "FAILED", "READY", "INTERRUPTED"})

    assert final["status"] == "CANCELLED"
    assert all(event["status"] != "READY" for event in final["events"])
    run = project / "harvest_runs" / probe_id
    assert not (run / "probe_receipt.json").exists()
    assert not (run / "result.json").exists()
    assert not (run / "terminal_commit.json").exists()


def test_probe_ready_without_terminal_commit_is_interrupted_and_retry_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash between READY publication and the commit never exposes READY."""

    project = tmp_path / "project"
    project.mkdir()
    transport = FakeTransport(_responses() + _responses())
    service, evidence = _service(project, transport)
    probe_service = service._probe_service
    real_write = probe_service._write_exclusive_json

    def failing_commit(path: Path, payload: Any, parent_anchor: Any) -> None:
        if path.name == "terminal_commit.json":
            raise onboarding_module.CatalogProbeError(
                "injected_commit_failure",
                "terminal commit publication failed",
            )
        return real_write(path, payload, parent_anchor)

    monkeypatch.setattr(probe_service, "_write_exclusive_json", failing_commit)
    started, _created = service.start_probe(_payload(service, evidence, key="probe-crash-commit-01"))
    interrupted = _wait(service, started["probe_id"], {"INTERRUPTED", "FAILED", "READY"})

    assert interrupted["status"] == "INTERRUPTED"
    assert interrupted["promotion_ready"] is False
    run = project / "harvest_runs" / started["probe_id"]
    assert (run / "probe_receipt.json").exists()
    assert (run / "result.json").exists()
    assert not (run / "terminal_commit.json").exists()
    with pytest.raises(HarvestError) as refused:
        service.promote_probe(
            started["probe_id"],
            explicit_action=True,
            authorize_catalog_promotion=True,
            **_promotion_review(service, started["probe_id"]),
        )
    assert refused.value.code == "catalog_probe_not_promotable"
    inventory = service.inventory()
    record = next(item for item in inventory["probe_runs"] if item["probe_id"] == started["probe_id"])
    assert record["status"] == "INTERRUPTED"
    assert record["promotion_ready"] is False

    # With the fault removed, an explicit retry recovers to a fully committed
    # READY probe while the interrupted evidence stays immutable.
    monkeypatch.setattr(probe_service, "_write_exclusive_json", real_write)
    retried, retry_created = service.retry_probe(
        started["probe_id"],
        _payload(service, evidence, key="probe-crash-commit-02"),
    )
    assert retry_created is True
    ready = _wait(service, retried["probe_id"], {"READY", "FAILED"})
    assert ready["status"] == "READY"
    assert ready["retry_of"] == started["probe_id"]
    assert (project / "harvest_runs" / retried["probe_id"] / "terminal_commit.json").exists()
    assert not (run / "terminal_commit.json").exists()
