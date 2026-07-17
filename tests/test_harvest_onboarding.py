from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from spritelab.product_core import ProjectContext
from spritelab.product_features.harvest import create_plugin
from spritelab.product_features.harvest.catalog import (
    TRUSTED_CATALOG_RELATIVE_PATH,
    TrustedCatalogError,
    load_trusted_catalog,
)
from spritelab.product_features.harvest.certification import BackendCapabilityEvidence
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
SOURCE_URL = "https://catalog.example.test/source"
LICENSE_URL = "https://catalog.example.test/license"
TERMS_URL = "https://catalog.example.test/terms"
DIRECT_URL = "https://downloads.example.test/pack.zip?token=private"
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
        "<h1>Verified Sprite Pack</h1><p>by Example Artist</p>"
        f"<p>{automation}</p>"
        f'<a href="{LICENSE_URL}">CC0 license</a>'
        f"{terms_link}{extra_terms_link}"
        f'<a href="{DIRECT_URL}">Direct download</a>'
        "</body></html>"
    ).encode()


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


def _service(project: Path, transport: FakeTransport) -> tuple[HarvestService, BackendCapabilityEvidence]:
    capabilities = _capabilities()
    evidence = _evidence(capabilities)
    return (
        HarvestService(
            project,
            backend_factory=lambda: None,
            backend_capabilities=capabilities,
            backend_capability_evidence=evidence,
            probe_resolver=lambda _host, _port: ("8.8.8.8",),
            probe_transport=transport,
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
    )
    assert promotion["raw_response_sha256"] == hashlib.sha256(RAW).hexdigest()
    assert refresh_attempts == 1
    assert service.sources()["sources"] == []
    assert (
        service.promote_probe(started["probe_id"], explicit_action=True, authorize_catalog_promotion=True) == promotion
    )
    assert refresh_attempts == 2
    catalog = load_trusted_catalog(project)
    assert [source.source_id for source in catalog] == ["verified.pack"]
    assert catalog[0].expected_response_sha256 == hashlib.sha256(RAW).hexdigest()
    assert catalog[0].evidence_binding.automation_terms.decision == "ALLOW"
    assert catalog[0].evidence_binding.automation_terms.evidence_url == TERMS_URL
    assert service.sources()["sources"][0]["evidence_binding"]["automation_terms"]["limited_evidence"] is False
    assert service.sources()["sources"][0]["source_id"] == "verified.pack"
    reloaded = HarvestService(
        project,
        sources=load_trusted_catalog(project),
        backend_factory=lambda: None,
        backend_capabilities=_capabilities(),
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

    catalog_path = project / TRUSTED_CATALOG_RELATIVE_PATH
    tampered = json.loads(catalog_path.read_text(encoding="utf-8"))
    tampered["sources"][0]["evidence_binding"]["automation_terms"]["decision_identity_sha256"] = "0" * 64
    catalog_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(TrustedCatalogError):
        load_trusted_catalog(project)


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


def test_promotion_rejects_linked_terms_page_drift(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, evidence = _service(project, FakeTransport(_responses()))
    started, _created = service.start_probe(_payload(service, evidence, key="probe-terms-drift-0001"))
    assert _wait(service, started["probe_id"], {"READY", "FAILED"})["status"] == "READY"
    terms = project / "harvest_runs" / started["probe_id"] / "evidence" / "terms_page.bin"
    terms.write_bytes(b"<html><body><p>Bots are forbidden</p></body></html>")

    with pytest.raises(HarvestError, match="changed"):
        service.promote_probe(started["probe_id"], explicit_action=True, authorize_catalog_promotion=True)
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
        service.promote_probe(started["probe_id"], explicit_action=True, authorize_catalog_promotion=True)
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
    )
    app = create_app(ProjectContext(project), plugins=(plugin,))
    client = TestClient(app)
    page = client.get("/harvest")
    assert page.status_code == 200
    assert "Start bounded source probe" in page.text
    assert "Automation terms URL" in page.text
    assert "Leave blank only when no governing Terms or ToS link exists" in page.text
    javascript = client.get("/harvest/static/harvest.js").text
    assert "/harvest/api/probes" in javascript
    assert "authorize_catalog_promotion" in javascript
    assert "Automation terms decision" in javascript
    assert "no prohibition observed; not affirmative permission" in javascript
    assert ".innerHTML" not in javascript

    denied = client.post(
        "/harvest/api/probes",
        json={"output_path": "C:/private"},
        headers={"X-CSRF-Token": app.state.spritelab_csrf_token},
    )
    assert denied.status_code == 422
    assert denied.json()["error_code"] == "browser_path_not_allowed"
