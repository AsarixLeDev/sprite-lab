from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import spritelab.harvest.archive as archive_module
import spritelab.harvest.download as download_module
from _harvest_testdata import make_zip_of_pngs
from spritelab.harvest.archive import ArchiveCancelled, ArchiveSnapshot, archive_member_summary, extract_archive
from spritelab.harvest.download import (
    DownloadCancelled,
    DownloadReceipt,
    DownloadSecurityError,
    ReceiptDownloadResult,
    download_file_with_receipt,
)
from spritelab.product_core import ProductStatus, ProjectContext
from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest import build_plugin
from spritelab.product_features.harvest.catalog import (
    TRUSTED_CATALOG_DIRECTORY_RELATIVE_PATH,
    TRUSTED_CATALOG_RELATIVE_PATH,
    TRUSTED_CATALOG_SCHEMA,
    CatalogAutomationTermsBinding,
    CatalogEvidenceBinding,
    HarvestSource,
    TrustedCatalogError,
    automation_terms_decision_identity,
    load_trusted_catalog,
    trusted_catalog_identity,
    trusted_catalog_source_filename,
    trusted_catalog_source_record,
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
from spritelab.product_features.harvest.certification import (
    BACKEND_AUDIT_REPORT_RELATIVE_PATH,
    BACKEND_AUDIT_REPORT_SCHEMA,
    BACKEND_CAPABILITIES_RELATIVE_PATH,
    BACKEND_CAPABILITY_CERTIFICATE_SCHEMA,
    REQUIRED_BACKEND_AUDIT_GATES,
    BackendCapabilityCertificateError,
    load_backend_capability_certificate,
    load_backend_capability_evidence,
)
from spritelab.product_features.harvest.service import HarvestError, HarvestService
from spritelab.product_features.harvest.trusted_backend import (
    CertifiedBackendCapabilities,
    HardenedArchiveAcquisitionBackend,
    HarvestLimits,
    conditioned_dataset_import_callback_binding,
    hardened_backend_code_identity,
    hardened_backend_module_hashes,
    hardened_backend_runtime_dependencies,
)
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, UnsafeFilesystemOperation

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
    verified_at = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    expires_at = (now + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    source_hash = hashlib.sha256(b"source page").hexdigest()
    terms = CatalogAutomationTermsBinding(
        mode="source_page_no_governing_terms_link",
        decision="NO_PROHIBITION_OBSERVED",
        evidence_url=SOURCE_PAGE,
        evidence_request_url_sha256=url_identity(SOURCE_PAGE),
        evidence_final_url=SOURCE_PAGE,
        evidence_http_status=200,
        evidence_content_sha256=source_hash,
        matched_declaration=None,
        limited_evidence=True,
        decision_identity_sha256=automation_terms_decision_identity(
            mode="source_page_no_governing_terms_link",
            evidence_url=SOURCE_PAGE,
            content_sha256=source_hash,
            matched_declaration=None,
            decision="NO_PROHIBITION_OBSERVED",
        ),
        verified_at=verified_at,
        expires_at=expires_at,
    )
    provisional = CatalogEvidenceBinding(
        verifier_id=CATALOG_EVIDENCE_VERIFIER_ID,
        verifier_code_identity_sha256=catalog_evidence_verifier_code_identity(),
        verified_at=verified_at,
        expires_at=expires_at,
        source_request_url_sha256=url_identity(SOURCE_PAGE),
        source_final_url=SOURCE_PAGE,
        source_http_status=200,
        source_content_sha256=source_hash,
        license_request_url_sha256=url_identity(LICENSE_PAGE),
        license_final_url=LICENSE_PAGE,
        license_http_status=200,
        license_content_sha256=hashlib.sha256(b"license page").hexdigest(),
        automation_terms=terms,
        zero_cost_reviewed=True,
        zero_cost_verification_identity_sha256="1" * 64,
        zero_cost_evidence_text_sha256="2" * 64,
        zero_cost_probe_receipt_identity_sha256="3" * 64,
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
        **conditioned_dataset_import_callback_binding(),
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


def _source_record(source: HarvestSource) -> dict[str, Any]:
    return trusted_catalog_source_record(source)


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


def test_append_only_catalog_publication_rejects_staged_inode_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    first = _source(b"first")
    _write_catalog(project, first)
    second = replace(_source(b"second"), source_id="open.second")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"outside bytes must remain unchanged")
    real_publish = AnchoredDirectory.publish_held_file_no_replace
    raced = False

    def substitute_staged_inode(
        anchor: AnchoredDirectory,
        source_descriptor: int,
        source_name: str | None,
        destination_name: str,
        *,
        identity: OwnedFileIdentity,
    ) -> None:
        nonlocal raced
        if not raced:
            raced = True
            assert source_name is not None
            foreign = anchor.directory / f".stage-{'f' * 32}"
            foreign.write_bytes(b"foreign stage must never become trusted")
            os.replace(foreign, anchor.directory / source_name)
        real_publish(
            anchor,
            source_descriptor,
            source_name,
            destination_name,
            identity=identity,
        )

    monkeypatch.setattr(AnchoredDirectory, "publish_held_file_no_replace", substitute_staged_inode)
    with pytest.raises(CatalogPromotionError, match="published exactly"):
        publish_trusted_catalog_source(project, project, second)

    assert raced is True
    record = project / TRUSTED_CATALOG_DIRECTORY_RELATIVE_PATH / trusted_catalog_source_filename(second.source_id)
    assert not record.exists()
    assert sentinel.read_bytes() == b"outside bytes must remain unchanged"
    assert load_trusted_catalog(project)[0].catalog_identity == first.catalog_identity

    monkeypatch.setattr(AnchoredDirectory, "publish_held_file_no_replace", real_publish)
    catalog, changed = publish_trusted_catalog_source(project, project, second)
    assert changed is True
    assert [source.source_id for source in catalog] == ["open.second", "open.sprites"]
    assert record.is_file()
    replay, changed = publish_trusted_catalog_source(project, project, second)
    assert changed is False
    assert replay == catalog
    with pytest.raises(CatalogPromotionError, match="different trusted catalog record"):
        publish_trusted_catalog_source(project, project, replace(second, title="conflict"))
    alias = outside / "record-alias.json"
    os.link(record, alias)
    with pytest.raises(TrustedCatalogError, match=r"hard-link|retained publication stage"):
        load_trusted_catalog(project)
    assert alias.read_bytes() == record.read_bytes()
    assert sentinel.read_bytes() == b"outside bytes must remain unchanged"


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
        "runtime_dependencies": hardened_backend_runtime_dependencies(),
        "gate_results": dict.fromkeys(sorted(REQUIRED_BACKEND_AUDIT_GATES), "PASS"),
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
        "runtime_dependencies": hardened_backend_runtime_dependencies(),
        "capabilities": asdict(capabilities),
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


def test_receipt_downloader_bounds_hanging_resolver_by_duration(tmp_path: Path) -> None:
    release = threading.Event()

    def hanging_resolver(_host: str, _port: int) -> tuple[str, ...]:
        release.wait(5)
        return ("8.8.8.8",)

    started = time.monotonic()
    try:
        with pytest.raises(DownloadSecurityError, match="duration"):
            download_file_with_receipt(
                DOWNLOAD_URL,
                tmp_path / "never.zip",
                allowed_hosts=("downloads.example.test",),
                max_duration_seconds=0.1,
                resolver=hanging_resolver,
                transport=FakeTransport([]),
            )
    finally:
        release.set()

    assert time.monotonic() - started < 1.0
    assert not (tmp_path / "never.zip").exists()


def test_receipt_downloader_observes_cancellation_during_connection_open(tmp_path: Path) -> None:
    release = threading.Event()
    cancelled = threading.Event()
    response = FakeResponse(b"body")

    class BlockingTransport:
        def open(self, **_kwargs: Any) -> FakeResponse:
            release.wait(5)
            return response

    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(DownloadCancelled, match="cancelled"):
            download_file_with_receipt(
                DOWNLOAD_URL,
                tmp_path / "cancelled-open.zip",
                allowed_hosts=("downloads.example.test",),
                max_duration_seconds=5,
                cancel_requested=cancelled.is_set,
                resolver=lambda _host, _port: ("8.8.8.8",),
                transport=BlockingTransport(),
            )
    finally:
        release.set()
        timer.join()

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not response.closed:
        time.sleep(0.01)
    assert time.monotonic() - started < 1.0
    assert response.closed is True
    assert not (tmp_path / "cancelled-open.zip").exists()


def test_receipt_downloader_observes_cancellation_during_blocking_read(tmp_path: Path) -> None:
    release = threading.Event()
    cancelled = threading.Event()

    class BlockingResponse(FakeResponse):
        def read(self, size: int = -1) -> bytes:
            release.wait(5)
            return super().read(size)

    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(DownloadCancelled, match="cancelled"):
            download_file_with_receipt(
                DOWNLOAD_URL,
                tmp_path / "cancelled.zip",
                allowed_hosts=("downloads.example.test",),
                max_duration_seconds=5,
                cancel_requested=cancelled.is_set,
                resolver=lambda _host, _port: ("8.8.8.8",),
                transport=FakeTransport([BlockingResponse(b"body")]),
            )
    finally:
        release.set()
        timer.join()

    assert time.monotonic() - started < 1.0
    assert not (tmp_path / "cancelled.zip").exists()
    assert not list(tmp_path.glob(".cancelled.zip.*.part"))


def test_receipt_downloader_hard_deadline_stops_slow_drip(tmp_path: Path) -> None:
    class SlowDripResponse(FakeResponse):
        def read(self, size: int = -1) -> bytes:
            time.sleep(0.04)
            return super().read(1 if size != 0 else size)

    started = time.monotonic()
    with pytest.raises(DownloadSecurityError, match="duration"):
        download_file_with_receipt(
            DOWNLOAD_URL,
            tmp_path / "slow.zip",
            allowed_hosts=("downloads.example.test",),
            max_duration_seconds=0.12,
            resolver=lambda _host, _port: ("8.8.8.8",),
            transport=FakeTransport([SlowDripResponse(b"slow-drip")]),
        )

    assert time.monotonic() - started < 1.0
    assert not (tmp_path / "slow.zip").exists()
    assert not list(tmp_path.glob(".slow.zip.*.part"))


def test_receipt_downloader_uses_one_reader_worker_for_many_chunks(tmp_path: Path, monkeypatch: Any) -> None:
    body = b"x" * (2 * 1024 * 1024)
    started_workers: list[str] = []
    real_start = threading.Thread.start

    def track_start(thread: threading.Thread) -> None:
        started_workers.append(thread.name)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", track_start)
    result = download_file_with_receipt(
        DOWNLOAD_URL,
        tmp_path / "many-chunks.zip",
        allowed_hosts=("downloads.example.test",),
        allowed_content_types=("application/zip",),
        resolver=lambda _host, _port: ("8.8.8.8",),
        transport=FakeTransport([FakeResponse(body)]),
    )

    assert result.receipt.response_bytes == len(body)
    assert started_workers.count("spritelab-download-response-reader") == 1


def test_repeated_resolver_timeouts_have_a_hard_worker_cap(tmp_path: Path) -> None:
    release = threading.Event()

    def hanging_resolver(_host: str, _port: int) -> tuple[str, ...]:
        release.wait(5)
        return ("8.8.8.8",)

    try:
        for index in range(download_module._BOUNDED_WORKER_CAPACITY + 3):
            with pytest.raises(DownloadSecurityError):
                download_file_with_receipt(
                    DOWNLOAD_URL,
                    tmp_path / f"timeout-{index}.zip",
                    allowed_hosts=("downloads.example.test",),
                    max_duration_seconds=0.05,
                    resolver=hanging_resolver,
                    transport=FakeTransport([]),
                )
        active = [
            thread
            for thread in threading.enumerate()
            if thread.name == "spritelab-download-hostname resolution" and thread.is_alive()
        ]
        assert len(active) <= download_module._BOUNDED_WORKER_CAPACITY
    finally:
        release.set()

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and any(
        thread.name == "spritelab-download-hostname resolution" and thread.is_alive()
        for thread in threading.enumerate()
    ):
        time.sleep(0.01)
    assert not any(
        thread.name == "spritelab-download-hostname resolution" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_receipt_download_holds_trusted_parent_before_network_aba(tmp_path: Path) -> None:
    parent = tmp_path / "trusted-parent"
    moved = tmp_path / "trusted-parent-held"
    outside = tmp_path / "outside"
    parent.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"preserve")
    swapped = False

    def swapping_resolver(_host: str, _port: int) -> tuple[str, ...]:
        nonlocal swapped
        try:
            os.replace(parent, moved)
        except OSError:
            pytest.skip("the platform held the trusted download parent against rename")
        try:
            os.symlink(outside, parent, target_is_directory=True)
        except OSError:
            os.replace(moved, parent)
            pytest.skip("directory symbolic links are unavailable in this test session")
        swapped = True
        return ("8.8.8.8",)

    try:
        with pytest.raises((DownloadSecurityError, UnsafeFilesystemOperation)):
            download_file_with_receipt(
                DOWNLOAD_URL,
                parent / "response.zip",
                allowed_hosts=("downloads.example.test",),
                allowed_content_types=("application/zip",),
                resolver=swapping_resolver,
                transport=FakeTransport([FakeResponse(b"archive")]),
            )
    finally:
        if swapped:
            os.unlink(parent)
            os.replace(moved, parent)

    assert sentinel.read_bytes() == b"preserve"
    assert not (outside / "response.zip").exists()


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


def test_archive_snapshot_defeats_path_aba_between_summary_and_extraction(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    archive = make_zip_of_pngs(tmp_path / "pack.zip", ["sprite.png"])
    original_png = (tmp_path / "pack_staging" / "sprite.png").read_bytes()
    attacker_archive = tmp_path / "attacker.zip"
    with zipfile.ZipFile(attacker_archive, "w") as handle:
        handle.writestr("sprite.png", b"\x89PNG\r\n\x1a\nattacker")
    parked_original = tmp_path / "parked-original.zip"
    real_extract_zip = archive_module._extract_zip

    def aba_extract(*args: Any, **kwargs: Any) -> None:
        archive.replace(parked_original)
        attacker_archive.replace(archive)
        try:
            real_extract_zip(*args, **kwargs)
        finally:
            archive.replace(attacker_archive)
            parked_original.replace(archive)

    monkeypatch.setattr(archive_module, "_extract_zip", aba_extract)
    with ArchiveSnapshot.open(archive) as snapshot:
        assert archive_member_summary(snapshot, include_member_globs=("*.png",))["selected_image_members"] == [
            "sprite.png"
        ]
        extract_archive(snapshot, tmp_path / "out", include_member_globs=("*.png",))
    assert (tmp_path / "out" / "sprite.png").read_bytes() == original_png


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


@pytest.mark.parametrize("evidence_kind", ["catalog", "certificate"])
def test_passive_evidence_loaders_refuse_parent_rename_symlink_aba(
    tmp_path: Path,
    monkeypatch: Any,
    evidence_kind: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = _source(b"archive")
    _write_catalog(project, source)
    if evidence_kind == "certificate":
        _write_capability_evidence(project, _capabilities())
    evidence_parent = project / "artifacts" / "harvest"
    moved_parent = project / "artifacts" / "harvest-held"
    outside = tmp_path / "outside-evidence"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"foreign evidence must remain untouched")
    real_open = AnchoredDirectory.open_file
    swapped = False

    def swap_before_relative_open(
        anchor: AnchoredDirectory,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        nonlocal swapped
        if not swapped and anchor.directory == evidence_parent:
            os.replace(evidence_parent, moved_parent)
            try:
                os.symlink(outside, evidence_parent, target_is_directory=True)
            except OSError:
                os.replace(moved_parent, evidence_parent)
                pytest.skip("directory symbolic links are unavailable in this test session")
            swapped = True
        return real_open(anchor, name, flags, mode)

    monkeypatch.setattr(AnchoredDirectory, "open_file", swap_before_relative_open)
    try:
        if evidence_kind == "catalog":
            with pytest.raises(TrustedCatalogError, match="unsafe"):
                load_trusted_catalog(project)
        else:
            with pytest.raises(BackendCapabilityCertificateError, match="unsafe"):
                load_backend_capability_evidence(project)
    finally:
        if swapped:
            os.unlink(evidence_parent)
            os.replace(moved_parent, evidence_parent)

    assert sentinel.read_bytes() == b"foreign evidence must remain untouched"


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "audit gates fields do not match"),
        ("extra", "audit gates fields do not match"),
        ("failed", "did not each record PASS"),
    ],
)
def test_independent_certificate_requires_exact_individual_pass_for_every_audit_gate(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_capability_evidence(project, _capabilities())
    report_path = project / BACKEND_AUDIT_REPORT_RELATIVE_PATH
    certificate_path = project / BACKEND_CAPABILITIES_RELATIVE_PATH
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["gate_results"]) == REQUIRED_BACKEND_AUDIT_GATES
    assert {
        "direct_static_image_derivation",
        "retained_anchored_state",
        "whole_operation_deadline",
        "durable_import_control",
        "same_pack_license_and_zero_cost",
        "technical_usability_and_pixel_uniqueness",
        "non_self_attested_production_bindings",
    } <= set(report["gate_results"])
    if mutation == "missing":
        report["gate_results"].pop("durable_import_control")
    elif mutation == "extra":
        report["gate_results"]["self_asserted_shortcut"] = "PASS"
    else:
        report["gate_results"]["whole_operation_deadline"] = "FAIL"
    report["report_identity"] = _identity({key: value for key, value in report.items() if key != "report_identity"})
    report_bytes = strict_json_dumps(report, sort_keys=True).encode("utf-8")
    report_path.write_bytes(report_bytes)
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["audit_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    certificate["certificate_identity"] = _identity(
        {key: value for key, value in certificate.items() if key != "certificate_identity"}
    )
    certificate_path.write_text(strict_json_dumps(certificate, sort_keys=True), encoding="utf-8")

    with pytest.raises(BackendCapabilityCertificateError, match=message):
        load_backend_capability_certificate(project)


@pytest.mark.parametrize("distribution", ["NumPy", "PyYAML"])
def test_independent_certificate_rejects_conditioned_runtime_dependency_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distribution: str,
) -> None:
    from spritelab.product_features.conditioned_v5 import identity as conditioned_identity

    project = tmp_path / "project"
    project.mkdir()
    capabilities = _capabilities()
    _write_capability_evidence(project, capabilities)
    original_inventory = conditioned_identity.installed_distribution_inventory

    def drifted_inventory(name: str) -> dict[str, Any]:
        inventory = original_inventory(name)
        if name.casefold() != distribution.casefold():
            return inventory
        replacement = dict(inventory)
        replacement_files = {locator: dict(binding) for locator, binding in inventory["files"].items()}
        changed_locator = min(replacement_files)
        changed_binding = replacement_files[changed_locator]
        changed_binding["sha256"] = "f" * 64 if changed_binding["sha256"] != "f" * 64 else "e" * 64
        replacement["files"] = replacement_files
        replacement["inventory_sha256"] = _identity(
            {key: value for key, value in replacement.items() if key != "inventory_sha256"}
        )
        return replacement

    monkeypatch.setattr(conditioned_identity, "installed_distribution_inventory", drifted_inventory)
    with pytest.raises(BackendCapabilityCertificateError, match="runtime dependencies changed"):
        load_backend_capability_certificate(project)


def test_independent_certificate_rejects_conditioned_callback_runtime_identity_drift(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    capabilities = _capabilities()
    _write_capability_evidence(project, capabilities)
    certificate_path = project / BACKEND_CAPABILITIES_RELATIVE_PATH
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["capabilities"]["dataset_import_callback_runtime_identity_sha256"] = "0" * 64
    certificate["certificate_identity"] = _identity(
        {key: value for key, value in certificate.items() if key != "certificate_identity"}
    )
    certificate_path.write_text(strict_json_dumps(certificate, sort_keys=True), encoding="utf-8")

    with pytest.raises(BackendCapabilityCertificateError, match="Dataset import callback runtime"):
        load_backend_capability_certificate(project)


def test_start_reloads_catalog_and_certificate_and_rejects_drift(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source = _source(b"archive")
    catalog_path = _write_catalog(project, source)
    capabilities = _capabilities()
    _write_capability_evidence(project, capabilities)
    evidence = load_backend_capability_evidence(project)
    assert evidence is not None
    assert evidence.to_dict()["audit_report_sha256"]

    def live_configuration() -> tuple[tuple[HarvestSource, ...], Any]:
        return load_trusted_catalog(project), load_backend_capability_evidence(project)

    service = HarvestService(
        project,
        sources=(source,),
        backend_factory=lambda: pytest.fail("stale configuration must fail before backend construction"),
        backend_capabilities=capabilities,
        backend_capability_evidence=evidence,
        live_configuration_loader=live_configuration,
    )
    changed = replace(source, title="Changed open sprite pack")
    payload = {
        "schema_version": TRUSTED_CATALOG_SCHEMA,
        "sources": [_source_record(changed)],
        "catalog_identity": trusted_catalog_identity((changed,)),
    }
    catalog_path.write_text(strict_json_dumps(payload, sort_keys=True), encoding="utf-8")
    inventory = service.inventory()
    with pytest.raises(HarvestError, match="changed") as error:
        service.start(
            source.source_id,
            idempotency_key="live-drift-0001",
            explicit_action=True,
            authorize_zero_cost=True,
            authorize_permissive_license=True,
            authorize_existing_inventory_reviewed=True,
            reuse_evidence={
                "decision": "reuse_exhausted",
                "evidence_code": "no_reusable_items",
                "inventory_identity": inventory["inventory_identity"],
                "assessed_usable_items": 0,
                "required_usable_items": 1,
                "deficit_items": 1,
            },
        )
    assert error.value.code == "harvest_live_configuration_changed"
    assert not (project / "harvest_runs").exists()
