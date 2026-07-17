from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from _harvest_testdata import make_zip_of_pngs
from spritelab.harvest.archive import ArchiveCancelled, extract_archive
from spritelab.harvest.download import (
    DownloadReceipt,
    DownloadSecurityError,
    ReceiptDownloadResult,
    download_file_with_receipt,
)
from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest import build_plugin
from spritelab.product_features.harvest.catalog import (
    TRUSTED_CATALOG_RELATIVE_PATH,
    TRUSTED_CATALOG_SCHEMA,
    CatalogEvidenceBinding,
    HarvestSource,
    TrustedCatalogError,
    load_trusted_catalog,
    trusted_catalog_identity,
    url_identity,
)
from spritelab.product_features.harvest.certification import (
    BACKEND_AUDIT_REPORT_RELATIVE_PATH,
    BACKEND_AUDIT_REPORT_SCHEMA,
    BACKEND_CAPABILITIES_RELATIVE_PATH,
    BACKEND_CAPABILITY_CERTIFICATE_SCHEMA,
    BackendCapabilityCertificateError,
    load_backend_capability_certificate,
)
from spritelab.product_features.harvest.trusted_backend import (
    CertifiedBackendCapabilities,
    HardenedArchiveAcquisitionBackend,
    HarvestLimits,
    hardened_backend_code_identity,
    hardened_backend_module_hashes,
)

SOURCE_PAGE = "https://catalog.example.test/source"
LICENSE_PAGE = "https://catalog.example.test/license"
DOWNLOAD_URL = "https://downloads.example.test/untrusted-remote-name?secret=yes"


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, Any] | None = None,
        peer_ip: str = "8.8.8.8",
    ) -> None:
        self.status = status
        self.headers = headers or {
            "Content-Type": "application/zip",
            "Content-Length": str(len(body)),
        }
        self.peer_ip = peer_ip
        self._body = body
        self._offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def open(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(dict(kwargs))
        return self.responses.pop(0)


def _identity(value: Any) -> str:
    encoded = strict_json_dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding() -> CatalogEvidenceBinding:
    now = datetime.now(timezone.utc)
    provisional = CatalogEvidenceBinding(
        verifier_id="audit.fixture",
        verifier_code_identity_sha256="a" * 64,
        verified_at=(now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(days=7)).isoformat().replace("+00:00", "Z"),
        source_request_url_sha256=url_identity(SOURCE_PAGE),
        source_final_url=SOURCE_PAGE,
        source_http_status=200,
        source_content_sha256=hashlib.sha256(b"source page").hexdigest(),
        license_request_url_sha256=url_identity(LICENSE_PAGE),
        license_final_url=LICENSE_PAGE,
        license_http_status=200,
        license_content_sha256=hashlib.sha256(b"license page").hexdigest(),
        attestation_identity_sha256="0" * 64,
    )
    return replace(provisional, attestation_identity_sha256=provisional.expected_attestation_identity)


def _source(response: bytes) -> HarvestSource:
    return HarvestSource(
        source_id="open.sprites",
        title="Open sprite pack",
        creator="Example Artist",
        source_page=SOURCE_PAGE,
        license_id="cc0-1.0",
        license_evidence_url=LICENSE_PAGE,
        license_evidence_text="CC0 1.0 public-domain dedication.",
        attribution_text="Example Artist - Open sprite pack",
        acquisition_reference=DOWNLOAD_URL,
        allowed_download_hosts=("downloads.example.test", "cdn.example.test"),
        expected_response_sha256=hashlib.sha256(response).hexdigest(),
        evidence_binding=_binding(),
        taxonomy_hints=("item",),
    )


def _capabilities() -> CertifiedBackendCapabilities:
    return CertifiedBackendCapabilities(
        backend_id="audit.backend",
        backend_version="1.0",
        downloader_id="audit.downloader",
        downloader_version="1.0",
        code_identity_sha256=hardened_backend_code_identity(),
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


def _source_record(source: HarvestSource) -> dict[str, Any]:
    binding = source.evidence_binding
    return {
        "source_id": source.source_id,
        "title": source.title,
        "creator": source.creator,
        "source_page": source.source_page,
        "license_id": source.license_id,
        "license_evidence_url": source.license_evidence_url,
        "license_evidence_text": source.license_evidence_text,
        "attribution_text": source.attribution_text,
        "acquisition_reference": source.acquisition_reference,
        "allowed_download_hosts": list(source.allowed_download_hosts),
        "expected_response_sha256": source.expected_response_sha256,
        "evidence_binding": {
            "verifier_id": binding.verifier_id,
            "verifier_code_identity_sha256": binding.verifier_code_identity_sha256,
            "verified_at": binding.verified_at,
            "expires_at": binding.expires_at,
            "source_request_url_sha256": binding.source_request_url_sha256,
            "source_final_url": binding.source_final_url,
            "source_http_status": binding.source_http_status,
            "source_content_sha256": binding.source_content_sha256,
            "license_request_url_sha256": binding.license_request_url_sha256,
            "license_final_url": binding.license_final_url,
            "license_http_status": binding.license_http_status,
            "license_content_sha256": binding.license_content_sha256,
            "attestation_identity_sha256": binding.attestation_identity_sha256,
        },
        "zero_cost": True,
        "permissive": True,
        "taxonomy_hints": list(source.taxonomy_hints),
    }


def _write_catalog(project: Path, source: HarvestSource) -> Path:
    path = project / TRUSTED_CATALOG_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    payload = {
        "schema_version": TRUSTED_CATALOG_SCHEMA,
        "sources": [_source_record(source)],
        "catalog_identity": trusted_catalog_identity((source,)),
    }
    path.write_text(strict_json_dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_capability_evidence(project: Path, capabilities: CertifiedBackendCapabilities) -> None:
    modules = hardened_backend_module_hashes()
    issued = datetime.now(timezone.utc) - timedelta(minutes=5)
    report = {
        "schema_version": BACKEND_AUDIT_REPORT_SCHEMA,
        "outcome": "PASS",
        "auditor_id": "independent.audit",
        "audited_at": (issued - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "implementation_identity_sha256": hardened_backend_code_identity(),
        "module_sha256": modules,
    }
    report["report_identity"] = _identity(report)
    report_bytes = strict_json_dumps(report, sort_keys=True).encode("utf-8")
    report_path = project / BACKEND_AUDIT_REPORT_RELATIVE_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report_bytes)
    certificate = {
        "schema_version": BACKEND_CAPABILITY_CERTIFICATE_SCHEMA,
        "auditor_id": "independent.audit",
        "issued_at": issued.isoformat().replace("+00:00", "Z"),
        "expires_at": (issued + timedelta(days=7)).isoformat().replace("+00:00", "Z"),
        "audit_report_relative_path": BACKEND_AUDIT_REPORT_RELATIVE_PATH.as_posix(),
        "audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "module_sha256": modules,
        "capabilities": {
            "backend_id": capabilities.backend_id,
            "backend_version": capabilities.backend_version,
            "downloader_id": capabilities.downloader_id,
            "downloader_version": capabilities.downloader_version,
            "code_identity_sha256": capabilities.code_identity_sha256,
            "enforces_http_success": True,
            "enforces_https_direct_url": True,
            "resolves_and_blocks_private_networks": True,
            "validates_every_redirect": True,
            "enforces_response_mime_allowlist": True,
            "enforces_expected_response_hash": True,
            "enforces_per_file_hashes": True,
            "enforces_file_count_and_byte_limits": True,
            "enforces_depth_and_name_policy": True,
            "enforces_archive_limits": True,
            "enforces_duration_and_cancellation": True,
        },
    }
    certificate["certificate_identity"] = _identity(certificate)
    (project / BACKEND_CAPABILITIES_RELATIVE_PATH).write_text(
        strict_json_dumps(certificate, sort_keys=True),
        encoding="utf-8",
    )


def test_receipt_downloader_pins_one_dns_result_and_preserves_tls_hostname(tmp_path: Path) -> None:
    body = b"archive"
    resolver_calls: list[tuple[str, int]] = []

    def resolver(host: str, port: int) -> tuple[str, ...]:
        resolver_calls.append((host, port))
        return ("8.8.8.8",) if len(resolver_calls) == 1 else ("127.0.0.1",)

    transport = FakeTransport([FakeResponse(body)])
    result = download_file_with_receipt(
        DOWNLOAD_URL,
        tmp_path / "response.zip",
        allowed_hosts=("downloads.example.test",),
        allowed_content_types=("application/zip",),
        expected_sha256=hashlib.sha256(body).hexdigest(),
        resolver=resolver,
        transport=transport,
    )
    assert resolver_calls == [("downloads.example.test", 443)]
    assert transport.calls[0]["pinned_ip"] == "8.8.8.8"
    assert transport.calls[0]["server_hostname"] == "downloads.example.test"
    assert result.receipt.response_sha256 == hashlib.sha256(body).hexdigest()


def test_receipt_downloader_rejects_any_non_global_answer_and_peer_change(tmp_path: Path) -> None:
    transport = FakeTransport([FakeResponse(b"x")])
    with pytest.raises(DownloadSecurityError, match="non-public"):
        download_file_with_receipt(
            DOWNLOAD_URL,
            tmp_path / "blocked.zip",
            allowed_hosts=("downloads.example.test",),
            resolver=lambda _host, _port: ("8.8.8.8", "127.0.0.1"),
            transport=transport,
        )
    assert transport.calls == []

    mismatch = FakeTransport([FakeResponse(b"x", peer_ip="1.1.1.1")])
    with pytest.raises(DownloadSecurityError, match="did not match pinned"):
        download_file_with_receipt(
            DOWNLOAD_URL,
            tmp_path / "mismatch.zip",
            allowed_hosts=("downloads.example.test",),
            resolver=lambda _host, _port: ("8.8.8.8",),
            transport=mismatch,
        )


def test_receipt_downloader_manual_redirects_and_probe_only_html_allowlist(tmp_path: Path) -> None:
    body = b"<html>evidence</html>"
    transport = FakeTransport(
        [
            FakeResponse(b"", status=302, headers={"Location": "https://cdn.example.test/evidence"}),
            FakeResponse(
                body,
                headers={"Content-Type": "text/html; charset=utf-8", "Content-Length": str(len(body))},
                peer_ip="1.1.1.1",
            ),
        ]
    )
    result = download_file_with_receipt(
        DOWNLOAD_URL,
        tmp_path / "page.html",
        allowed_hosts=("downloads.example.test", "cdn.example.test"),
        allowed_content_types=("text/html",),
        resolver=lambda host, _port: ("8.8.8.8",) if host.startswith("downloads") else ("1.1.1.1",),
        transport=transport,
    )
    assert result.receipt.redirect_chain == (DOWNLOAD_URL,)
    assert result.receipt.final_url == "https://cdn.example.test/evidence"
    assert result.receipt.response_mime_type == "text/html"
    with pytest.raises(DownloadSecurityError, match="HTML"):
        download_file_with_receipt(
            DOWNLOAD_URL,
            tmp_path / "archive.zip",
            allowed_hosts=("downloads.example.test",),
            allowed_content_types=("application/zip",),
            resolver=lambda _host, _port: ("8.8.8.8",),
            transport=FakeTransport(
                [FakeResponse(body, headers={"Content-Type": "text/html", "Content-Length": str(len(body))})]
            ),
        )


def test_hardened_backend_keeps_raw_archive_outside_artifacts_and_refuses_reuse(tmp_path: Path) -> None:
    archive = make_zip_of_pngs(tmp_path / "fixture.zip", ["nested/sprite.png"])
    archive_bytes = archive.read_bytes()
    source = _source(archive_bytes)

    def downloader(_url: str, output: Path, **_kwargs: Any) -> ReceiptDownloadResult:
        output.write_bytes(archive_bytes)
        return ReceiptDownloadResult(
            output,
            DownloadReceipt(
                final_url=DOWNLOAD_URL,
                redirect_chain=(),
                http_status=200,
                response_mime_type="application/zip",
                response_bytes=len(archive_bytes),
                response_sha256=source.expected_response_sha256,
                elapsed_seconds=0.01,
            ),
        )

    run = tmp_path / "harvest-run"
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True)
    backend = HardenedArchiveAcquisitionBackend(_capabilities(), downloader=downloader)
    result = backend.acquire(
        source,
        artifacts,
        HarvestLimits(max_files=5, max_total_bytes=1 << 20),
        cancel_requested=lambda: False,
        progress=lambda _stage, _current, _total: None,
    )
    assert (run / "downloads" / "response.zip").read_bytes() == archive_bytes
    assert (artifacts / "nested" / "sprite.png").is_file()
    assert not (artifacts / "response.zip").exists()
    assert result.receipt.files[0].relative_path == "nested/sprite.png"
    assert result.receipt.archive_members == 1
    with pytest.raises((FileExistsError, ValueError)):
        backend.acquire(
            source,
            artifacts,
            HarvestLimits(max_files=5, max_total_bytes=1 << 20),
            cancel_requested=lambda: False,
            progress=lambda _stage, _current, _total: None,
        )


def test_archive_extraction_cancellation_leaves_destination_unpublished(tmp_path: Path) -> None:
    archive = make_zip_of_pngs(tmp_path / "cancel.zip", ["sprite.png"])
    output = tmp_path / "cancelled"
    with pytest.raises(ArchiveCancelled):
        extract_archive(archive, output, cancel_requested=lambda: True)
    assert not output.exists()


def test_trusted_catalog_loader_is_passive_strict_and_single_link(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    assert load_trusted_catalog(project) == ()
    assert not (project / "artifacts").exists()
    source = _source(b"archive")
    catalog_path = _write_catalog(project, source)
    assert load_trusted_catalog(project) == (source,)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["catalog_identity"] = "0" * 64
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrustedCatalogError):
        load_trusted_catalog(project)

    other = project / "artifacts" / "harvest" / "catalog-hardlink.json"
    os.link(catalog_path, other)
    with pytest.raises(TrustedCatalogError, match="single-link"):
        load_trusted_catalog(project)


def test_independent_certificate_activates_default_plugin_without_network(tmp_path: Path, monkeypatch: Any) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = _source(b"archive")
    _write_catalog(project, source)
    capabilities = _capabilities()
    _write_capability_evidence(project, capabilities)
    loaded = load_backend_capability_certificate(project)
    assert loaded == capabilities

    monkeypatch.setattr("socket.create_connection", lambda *_args, **_kwargs: pytest.fail("network access"))
    result = build_plugin().status_provider(ProjectContext(project))
    assert result.status is ProductStatus.READY
    assert not (project / "harvest_runs").exists()

    certificate_path = project / BACKEND_CAPABILITIES_RELATIVE_PATH
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["module_sha256"]["spritelab.harvest.download"] = "0" * 64
    certificate["certificate_identity"] = _identity(
        {key: value for key, value in certificate.items() if key != "certificate_identity"}
    )
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(BackendCapabilityCertificateError, match="changed after the PASS audit"):
        load_backend_capability_certificate(project)
