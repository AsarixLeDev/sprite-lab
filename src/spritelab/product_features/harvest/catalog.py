"""Strict, administrator-owned Harvest source catalog records."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from spritelab.product_core.events import strict_json_dumps
from spritelab.product_features.harvest.catalog_verifier import (
    CATALOG_EVIDENCE_VERIFIER_ID,
    catalog_evidence_verifier_code_identity,
)
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation

SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HOST_PATTERN = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
TAXONOMY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

INITIAL_LICENSE_POLICY = frozenset({"cc0-1.0", "public-domain"})
AUTOMATION_ALLOW_DECLARATIONS = frozenset(
    {
        "automated access: allowed",
        "automated downloads: allowed",
        "downloads may be automated",
        "machine access: allowed",
    }
)
_UNKNOWN_CREATORS = frozenset({"unknown", "n/a", "na", "none", "anonymous", "unspecified"})
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost", ".home", ".arpa")
TRUSTED_CATALOG_SCHEMA = "spritelab.harvest.trusted-catalog.v2"
TRUSTED_CATALOG_RELATIVE_PATH = Path("artifacts") / "harvest" / "trusted_catalog.json"
MAX_TRUSTED_CATALOG_BYTES = 1 << 20
MAX_TRUSTED_CATALOG_SOURCES = 256

_CATALOG_KEYS = frozenset({"schema_version", "sources", "catalog_identity"})
_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "title",
        "creator",
        "source_page",
        "license_id",
        "license_evidence_url",
        "license_evidence_text",
        "attribution_text",
        "acquisition_reference",
        "allowed_download_hosts",
        "expected_response_sha256",
        "evidence_binding",
        "zero_cost",
        "permissive",
        "taxonomy_hints",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "verifier_id",
        "verifier_code_identity_sha256",
        "verified_at",
        "expires_at",
        "source_request_url_sha256",
        "source_final_url",
        "source_http_status",
        "source_content_sha256",
        "license_request_url_sha256",
        "license_final_url",
        "license_http_status",
        "license_content_sha256",
        "automation_terms",
        "attestation_identity_sha256",
    }
)
_AUTOMATION_TERMS_KEYS = frozenset(
    {
        "mode",
        "decision",
        "evidence_url",
        "evidence_request_url_sha256",
        "evidence_final_url",
        "evidence_http_status",
        "evidence_content_sha256",
        "matched_declaration",
        "limited_evidence",
        "decision_identity_sha256",
        "verified_at",
        "expires_at",
    }
)
_AUTOMATION_TERMS_MODES = frozenset({"source_page_no_governing_terms_link", "linked_terms_page"})
_AUTOMATION_TERMS_DECISIONS = frozenset({"ALLOW", "BLOCK", "NO_PROHIBITION_OBSERVED"})


class TrustedCatalogError(ValueError):
    """A repository-local trusted catalog failed strict passive validation."""


@dataclass(frozen=True)
class HarvestSource:
    """A fully evidenced, zero-cost source selected by opaque server-side ID.

    ``acquisition_reference`` may contain a signed URL and is available only to
    the trusted backend. Durable/public records contain its SHA-256 identity,
    never the reference itself.
    """

    source_id: str
    title: str
    creator: str
    source_page: str
    license_id: str
    license_evidence_url: str
    license_evidence_text: str
    attribution_text: str
    acquisition_reference: str
    allowed_download_hosts: tuple[str, ...]
    expected_response_sha256: str
    evidence_binding: CatalogEvidenceBinding
    zero_cost: bool = True
    permissive: bool = True
    taxonomy_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if SOURCE_ID_PATTERN.fullmatch(self.source_id) is None:
            raise ValueError("Harvest source_id is invalid.")
        _validate_text(self.title, "title", minimum=2, maximum=200)
        _validate_text(self.creator, "creator", minimum=2, maximum=200)
        if self.creator.strip().casefold() in _UNKNOWN_CREATORS:
            raise ValueError("Harvest creator must be explicit and cannot be Unknown.")
        _validate_text(self.attribution_text, "attribution_text", minimum=2, maximum=1000)
        _validate_text(self.license_evidence_text, "license_evidence_text", minimum=2, maximum=8000)
        validate_public_evidence_url(self.source_page)
        validate_public_evidence_url(self.license_evidence_url)
        if not isinstance(self.evidence_binding, CatalogEvidenceBinding):
            raise ValueError("Harvest source requires a certified source/license evidence binding.")
        self.evidence_binding.validate(self.source_page, self.license_evidence_url)
        if self.normalized_license_id not in INITIAL_LICENSE_POLICY:
            raise ValueError("Harvest initially accepts only CC0-1.0 or explicit public-domain sources.")
        if self.zero_cost is not True or self.permissive is not True:
            raise ValueError("Harvest catalog sources must be zero-cost and policy-permissive.")
        if SHA256_PATTERN.fullmatch(self.expected_response_sha256) is None:
            raise ValueError("Harvest expected_response_sha256 must be lowercase SHA-256.")
        if not self.allowed_download_hosts:
            raise ValueError("Harvest sources require at least one exact allowed download host.")
        normalized_hosts = tuple(host.casefold().rstrip(".") for host in self.allowed_download_hosts)
        if len(set(normalized_hosts)) != len(normalized_hosts):
            raise ValueError("Harvest allowed download hosts must be unique.")
        for host in normalized_hosts:
            validate_public_hostname(host)
        acquisition = validate_download_url(self.acquisition_reference, normalized_hosts)
        if acquisition.scheme.casefold() != "https":
            raise ValueError("Direct Harvest acquisition references must use HTTPS.")
        if len(set(self.taxonomy_hints)) != len(self.taxonomy_hints):
            raise ValueError("Harvest taxonomy hints must be unique.")
        if any(TAXONOMY_PATTERN.fullmatch(item) is None for item in self.taxonomy_hints):
            raise ValueError("Harvest taxonomy hints must be controlled lowercase tokens.")

    @property
    def normalized_license_id(self) -> str:
        return self.license_id.strip().casefold()

    @property
    def normalized_download_hosts(self) -> tuple[str, ...]:
        return tuple(host.casefold().rstrip(".") for host in self.allowed_download_hosts)

    @property
    def acquisition_reference_sha256(self) -> str:
        return hashlib.sha256(self.acquisition_reference.encode("utf-8")).hexdigest()

    @property
    def catalog_identity(self) -> str:
        return _identity(self._identity_payload())

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.source.v3",
            "source_id": self.source_id,
            "title": self.title.strip(),
            "creator": self.creator.strip(),
            "source_page": public_url(self.source_page),
            "source_page_sha256": url_identity(self.source_page),
            "license": {
                "identifier": self.normalized_license_id,
                "evidence_url": public_url(self.license_evidence_url),
                "evidence_url_sha256": url_identity(self.license_evidence_url),
                "evidence_text": self.license_evidence_text.strip(),
                "attribution_text": self.attribution_text.strip(),
                "permissive_policy": True,
            },
            "zero_cost": True,
            "allowed_download_hosts": list(self.normalized_download_hosts),
            "expected_response_sha256": self.expected_response_sha256,
            "acquisition_reference_sha256": self.acquisition_reference_sha256,
            "evidence_binding": {
                **self.evidence_binding.to_dict(),
                "binding_identity": self.evidence_binding.identity,
            },
            "taxonomy_hints": list(self.taxonomy_hints),
            "catalog_identity": self.catalog_identity,
            "private_reference_exposed": False,
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title.strip(),
            "creator": self.creator.strip(),
            "source_page_sha256": url_identity(self.source_page),
            "license_id": self.normalized_license_id,
            "license_evidence_url_sha256": url_identity(self.license_evidence_url),
            "license_evidence_text": self.license_evidence_text.strip(),
            "attribution_text": self.attribution_text.strip(),
            "allowed_download_hosts": list(self.normalized_download_hosts),
            "expected_response_sha256": self.expected_response_sha256,
            "acquisition_reference_sha256": self.acquisition_reference_sha256,
            "evidence_binding_identity": self.evidence_binding.identity,
            "taxonomy_hints": list(self.taxonomy_hints),
            "zero_cost": True,
            "permissive": True,
        }


@dataclass(frozen=True)
class TrustedCatalogSnapshot:
    """Exact stable file evidence used to parse an existing trusted catalog."""

    sha256: str
    byte_count: int
    device: int
    inode: int
    mode: int
    link_count: int
    modified_ns: int

    @classmethod
    def from_payload(cls, payload: bytes, metadata: os.stat_result) -> TrustedCatalogSnapshot:
        return cls(
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            link_count=metadata.st_nlink,
            modified_ns=metadata.st_mtime_ns,
        )

    def matches(self, payload: bytes, metadata: os.stat_result) -> bool:
        return self == TrustedCatalogSnapshot.from_payload(payload, metadata)


@dataclass(frozen=True)
class CatalogAutomationTermsBinding:
    """Immutable, expiring automation-terms decision retained with a source."""

    mode: str
    decision: str
    evidence_url: str
    evidence_request_url_sha256: str
    evidence_final_url: str
    evidence_http_status: int
    evidence_content_sha256: str
    matched_declaration: str | None
    limited_evidence: bool
    decision_identity_sha256: str
    verified_at: str
    expires_at: str

    def validate(self, source_page: str, *, now: datetime | None = None) -> None:
        if self.mode not in _AUTOMATION_TERMS_MODES or self.decision not in _AUTOMATION_TERMS_DECISIONS:
            raise ValueError("Harvest automation-terms mode or decision is invalid.")
        validate_public_evidence_url(self.evidence_url)
        validate_public_evidence_url(self.evidence_final_url)
        if self.evidence_url != public_url(self.evidence_url) or self.evidence_final_url != public_url(
            self.evidence_final_url
        ):
            raise ValueError("Harvest automation-terms evidence URLs must be canonical and query-free.")
        for value in (
            self.evidence_request_url_sha256,
            self.evidence_content_sha256,
            self.decision_identity_sha256,
        ):
            if SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError("Harvest automation-terms evidence requires lowercase SHA-256 identities.")
        if self.evidence_request_url_sha256 != url_identity(self.evidence_url):
            raise ValueError("Harvest automation-terms request URL identity is invalid.")
        if url_identity(self.evidence_final_url) != url_identity(self.evidence_url):
            raise ValueError("Harvest automation-terms final URL changed despite the no-redirect policy.")
        if not 200 <= self.evidence_http_status < 300:
            raise ValueError("Harvest automation-terms evidence requires a successful retrieval status.")
        if self.mode == "source_page_no_governing_terms_link" and url_identity(source_page) != url_identity(
            self.evidence_url
        ):
            raise ValueError("Harvest source-page automation terms are not bound to the source page.")
        if self.decision == "NO_PROHIBITION_OBSERVED":
            if self.matched_declaration is not None or self.limited_evidence is not True:
                raise ValueError("Harvest silent automation terms must retain an honest limited-evidence decision.")
        elif not isinstance(self.matched_declaration, str) or not self.matched_declaration.strip():
            raise ValueError("Harvest explicit automation-terms decisions require the matched declaration.")
        elif self.limited_evidence is not False:
            raise ValueError("Harvest explicit automation-terms decisions cannot be marked as limited evidence.")
        elif self.decision == "ALLOW" and self.matched_declaration not in AUTOMATION_ALLOW_DECLARATIONS:
            raise ValueError("Harvest automation permission is not an exact supported declaration.")
        verified = _parse_utc(self.verified_at)
        expires = _parse_utc(self.expires_at)
        current = now or datetime.now(timezone.utc)
        if verified > current or expires <= verified or expires - verified > timedelta(days=30) or expires <= current:
            raise ValueError("Harvest automation-terms evidence is absent, future-dated, or stale.")
        if self.decision_identity_sha256 != self.expected_decision_identity:
            raise ValueError("Harvest automation-terms decision identity is invalid.")

    @property
    def expected_decision_identity(self) -> str:
        return automation_terms_decision_identity(
            mode=self.mode,
            evidence_url=self.evidence_url,
            content_sha256=self.evidence_content_sha256,
            matched_declaration=self.matched_declaration,
            decision=self.decision,
        )

    @property
    def identity(self) -> str:
        return _identity(self.attested_payload())

    def attested_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.catalog-automation-terms-attestation.v1",
            "mode": self.mode,
            "decision": self.decision,
            "evidence_url_sha256": url_identity(self.evidence_url),
            "evidence_request_url_sha256": self.evidence_request_url_sha256,
            "evidence_final_url_sha256": url_identity(self.evidence_final_url),
            "evidence_http_status": self.evidence_http_status,
            "evidence_content_sha256": self.evidence_content_sha256,
            "matched_declaration": self.matched_declaration,
            "limited_evidence": self.limited_evidence,
            "decision_identity_sha256": self.decision_identity_sha256,
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "robots_permission_treated_as_terms_permission": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.catalog-automation-terms-binding.v1",
            "mode": self.mode,
            "decision": self.decision,
            "evidence_url": public_url(self.evidence_url),
            "evidence_request_url_sha256": self.evidence_request_url_sha256,
            "evidence_final_url": public_url(self.evidence_final_url),
            "evidence_final_url_sha256": url_identity(self.evidence_final_url),
            "evidence_http_status": self.evidence_http_status,
            "evidence_content_sha256": self.evidence_content_sha256,
            "matched_declaration": self.matched_declaration,
            "limited_evidence": self.limited_evidence,
            "decision_identity_sha256": self.decision_identity_sha256,
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "binding_identity": self.identity,
            "robots_permission_treated_as_terms_permission": False,
        }


@dataclass(frozen=True)
class CatalogEvidenceBinding:
    """Externally verified source/license-page snapshot bound into the catalog."""

    verifier_id: str
    verifier_code_identity_sha256: str
    verified_at: str
    expires_at: str
    source_request_url_sha256: str
    source_final_url: str
    source_http_status: int
    source_content_sha256: str
    license_request_url_sha256: str
    license_final_url: str
    license_http_status: int
    license_content_sha256: str
    automation_terms: CatalogAutomationTermsBinding
    attestation_identity_sha256: str

    def validate(self, source_page: str, license_evidence_url: str, *, now: datetime | None = None) -> None:
        if self.verifier_id != CATALOG_EVIDENCE_VERIFIER_ID:
            raise ValueError("Harvest evidence verifier identifier is not trusted.")
        if self.verifier_code_identity_sha256 != catalog_evidence_verifier_code_identity():
            raise ValueError("Harvest evidence verifier code identity is not current.")
        for value in (
            self.verifier_code_identity_sha256,
            self.source_request_url_sha256,
            self.source_content_sha256,
            self.license_request_url_sha256,
            self.license_content_sha256,
            self.attestation_identity_sha256,
        ):
            if SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError("Harvest evidence binding requires lowercase SHA-256 identities.")
        if not 200 <= self.source_http_status < 300 or not 200 <= self.license_http_status < 300:
            raise ValueError("Harvest evidence pages require successful retrieval status.")
        validate_public_evidence_url(self.source_final_url)
        validate_public_evidence_url(self.license_final_url)
        verified = _parse_utc(self.verified_at)
        expires = _parse_utc(self.expires_at)
        current = now or datetime.now(timezone.utc)
        if verified > current or expires <= verified or expires - verified > timedelta(days=30) or expires <= current:
            raise ValueError("Harvest catalog evidence binding is absent, future-dated, or stale.")
        # The exact requested evidence URLs remain bound even when the verified
        # retrieval followed a public redirect to a final URL.
        if url_identity(source_page) != self.source_request_url_sha256:
            raise ValueError("Harvest source-page evidence binding changed.")
        if url_identity(license_evidence_url) != self.license_request_url_sha256:
            raise ValueError("Harvest license evidence binding changed.")
        if not isinstance(self.automation_terms, CatalogAutomationTermsBinding):
            raise ValueError("Harvest source requires durable automation-terms evidence.")
        self.automation_terms.validate(source_page, now=current)
        if self.automation_terms.verified_at != self.verified_at or self.automation_terms.expires_at != self.expires_at:
            raise ValueError("Harvest automation-terms evidence lifetime is not bound to the catalog attestation.")
        if self.automation_terms.mode == "source_page_no_governing_terms_link" and (
            self.automation_terms.evidence_content_sha256 != self.source_content_sha256
            or self.automation_terms.evidence_http_status != self.source_http_status
            or url_identity(self.automation_terms.evidence_final_url) != url_identity(self.source_final_url)
        ):
            raise ValueError("Harvest source-page automation terms do not match the bound source snapshot.")
        if self.automation_terms.decision == "BLOCK":
            raise ValueError("Harvest source automation terms explicitly prohibit acquisition.")
        if self.attestation_identity_sha256 != self.expected_attestation_identity:
            raise ValueError("Harvest catalog evidence attestation identity is invalid.")

    @property
    def identity(self) -> str:
        return _identity(self.to_dict())

    @property
    def expected_attestation_identity(self) -> str:
        return _identity(self.attested_payload())

    def attested_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.catalog-evidence-attestation.v2",
            "verifier_id": self.verifier_id,
            "verifier_code_identity_sha256": self.verifier_code_identity_sha256,
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "source_request_url_sha256": self.source_request_url_sha256,
            "source_final_url_sha256": url_identity(self.source_final_url),
            "source_http_status": self.source_http_status,
            "source_content_sha256": self.source_content_sha256,
            "license_request_url_sha256": self.license_request_url_sha256,
            "license_final_url_sha256": url_identity(self.license_final_url),
            "license_http_status": self.license_http_status,
            "license_content_sha256": self.license_content_sha256,
            "automation_terms": self.automation_terms.attested_payload(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.catalog-evidence-binding.v2",
            "verifier_id": self.verifier_id,
            "verifier_code_identity_sha256": self.verifier_code_identity_sha256,
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "source_final_url": public_url(self.source_final_url),
            "source_final_url_sha256": url_identity(self.source_final_url),
            "source_request_url_sha256": self.source_request_url_sha256,
            "source_http_status": self.source_http_status,
            "source_content_sha256": self.source_content_sha256,
            "license_final_url": public_url(self.license_final_url),
            "license_final_url_sha256": url_identity(self.license_final_url),
            "license_request_url_sha256": self.license_request_url_sha256,
            "license_http_status": self.license_http_status,
            "license_content_sha256": self.license_content_sha256,
            "automation_terms": self.automation_terms.to_dict(),
            "attestation_identity_sha256": self.attestation_identity_sha256,
        }


def trusted_catalog_identity(sources: Sequence[HarvestSource]) -> str:
    """Return the deterministic identity stored beside a trusted catalog."""

    ordered = sorted(sources, key=lambda source: source.source_id)
    if len({source.source_id for source in ordered}) != len(ordered):
        raise ValueError("Harvest trusted catalog source identifiers must be unique.")
    return _identity(
        {
            "schema_version": TRUSTED_CATALOG_SCHEMA,
            "sources": [
                {
                    "source_id": source.source_id,
                    "source_catalog_identity": source.catalog_identity,
                }
                for source in ordered
            ],
        }
    )


def trusted_catalog_source_record(source: HarvestSource) -> dict[str, Any]:
    """Return the exact private trusted-catalog representation of one source."""

    binding = source.evidence_binding
    return {
        "source_id": source.source_id,
        "title": source.title.strip(),
        "creator": source.creator.strip(),
        "source_page": source.source_page,
        "license_id": source.normalized_license_id,
        "license_evidence_url": source.license_evidence_url,
        "license_evidence_text": source.license_evidence_text.strip(),
        "attribution_text": source.attribution_text.strip(),
        "acquisition_reference": source.acquisition_reference,
        "allowed_download_hosts": list(source.normalized_download_hosts),
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
            "automation_terms": {
                "mode": binding.automation_terms.mode,
                "decision": binding.automation_terms.decision,
                "evidence_url": binding.automation_terms.evidence_url,
                "evidence_request_url_sha256": binding.automation_terms.evidence_request_url_sha256,
                "evidence_final_url": binding.automation_terms.evidence_final_url,
                "evidence_http_status": binding.automation_terms.evidence_http_status,
                "evidence_content_sha256": binding.automation_terms.evidence_content_sha256,
                "matched_declaration": binding.automation_terms.matched_declaration,
                "limited_evidence": binding.automation_terms.limited_evidence,
                "decision_identity_sha256": binding.automation_terms.decision_identity_sha256,
                "verified_at": binding.automation_terms.verified_at,
                "expires_at": binding.automation_terms.expires_at,
            },
            "attestation_identity_sha256": binding.attestation_identity_sha256,
        },
        "zero_cost": True,
        "permissive": True,
        "taxonomy_hints": list(source.taxonomy_hints),
    }


def trusted_catalog_record(sources: Sequence[HarvestSource]) -> dict[str, Any]:
    """Return a deterministic sorted catalog ready for strict JSON publication."""

    ordered = tuple(sorted(sources, key=lambda source: source.source_id))
    if not ordered or len(ordered) > MAX_TRUSTED_CATALOG_SOURCES:
        raise ValueError("Harvest trusted catalog source count is invalid.")
    return {
        "schema_version": TRUSTED_CATALOG_SCHEMA,
        "sources": [trusted_catalog_source_record(source) for source in ordered],
        "catalog_identity": trusted_catalog_identity(ordered),
    }


def load_trusted_catalog(project_root: str | Path) -> tuple[HarvestSource, ...]:
    """Passively load the exact repository-local trusted catalog, if present.

    The loader never creates a directory, follows a link, resolves a hostname,
    or performs a network request. An absent catalog is an empty catalog; an
    existing malformed or unsafe catalog fails closed.
    """

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    try:
        payload, _snapshot = _read_catalog_snapshot(root)
        if payload is None:
            return ()
        parsed = _strict_catalog_json(payload)
        sources = _parse_trusted_catalog(parsed)
    except (OSError, UnicodeError, UnsafeFilesystemOperation, ValueError) as exc:
        if isinstance(exc, TrustedCatalogError):
            raise
        raise TrustedCatalogError("Harvest trusted catalog is unsafe or invalid.") from exc
    return sources


def load_trusted_catalog_snapshot(
    project_root: str | Path,
) -> tuple[tuple[HarvestSource, ...], TrustedCatalogSnapshot | None]:
    """Load sources plus the exact descriptor-read snapshot that produced them."""

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    try:
        payload, snapshot = _read_catalog_snapshot(root)
        if payload is None:
            return (), None
        return _parse_trusted_catalog(_strict_catalog_json(payload)), snapshot
    except (OSError, UnicodeError, UnsafeFilesystemOperation, ValueError) as exc:
        if isinstance(exc, TrustedCatalogError):
            raise
        raise TrustedCatalogError("Harvest trusted catalog is unsafe or invalid.") from exc


def _read_catalog_bytes(root: Path) -> bytes | None:
    return _read_catalog_snapshot(root)[0]


def _read_catalog_snapshot(root: Path) -> tuple[bytes | None, TrustedCatalogSnapshot | None]:
    parts = TRUSTED_CATALOG_RELATIVE_PATH.parts
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in parts[:-1]:
            if not anchor.lexists(part):
                return None, None
            anchor = stack.enter_context(anchor.open_directory(part))
        name = parts[-1]
        if not anchor.lexists(name):
            return None, None
        before = anchor.lstat(name)
        if (
            not stat.S_ISREG(before.st_mode)
            or _metadata_is_link_or_reparse(before)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= MAX_TRUSTED_CATALOG_BYTES
        ):
            raise TrustedCatalogError("Harvest trusted catalog must be one bounded, single-link regular file.")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = anchor.open_file(name, flags)
        try:
            opened = os.fstat(descriptor)
            if not _same_file_metadata(before, opened):
                raise TrustedCatalogError("Harvest trusted catalog changed while it was opened.")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read(MAX_TRUSTED_CATALOG_BYTES + 1)
            after_opened = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = anchor.lstat(name)
        if (
            len(payload) != before.st_size
            or len(payload) > MAX_TRUSTED_CATALOG_BYTES
            or not _same_file_metadata(before, after_opened)
            or not _same_file_metadata(before, after_path)
        ):
            raise TrustedCatalogError("Harvest trusted catalog changed while it was read.")
        return payload, TrustedCatalogSnapshot.from_payload(payload, after_opened)


def _strict_catalog_json(payload: bytes) -> Any:
    def reject_constant(token: str) -> None:
        raise TrustedCatalogError(f"Non-standard JSON constant {token!r} is forbidden.")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TrustedCatalogError(f"Duplicate JSON key {key!r} is forbidden.")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TrustedCatalogError("Harvest trusted catalog is not strict UTF-8 JSON.") from exc


def _parse_trusted_catalog(value: Any) -> tuple[HarvestSource, ...]:
    record = _exact_mapping(value, _CATALOG_KEYS, "catalog")
    if record["schema_version"] != TRUSTED_CATALOG_SCHEMA:
        raise TrustedCatalogError("Harvest trusted catalog schema version is unsupported.")
    source_values = record["sources"]
    if not isinstance(source_values, list) or not 1 <= len(source_values) <= MAX_TRUSTED_CATALOG_SOURCES:
        raise TrustedCatalogError("Harvest trusted catalog source count is invalid.")
    sources = tuple(_parse_catalog_source(item) for item in source_values)
    source_ids = [source.source_id for source in sources]
    if source_ids != sorted(source_ids) or len(set(source_ids)) != len(source_ids):
        raise TrustedCatalogError("Harvest trusted catalog sources must be unique and sorted by source_id.")
    if record["catalog_identity"] != trusted_catalog_identity(sources):
        raise TrustedCatalogError("Harvest trusted catalog identity does not match its sources.")
    return sources


def _parse_catalog_source(value: Any) -> HarvestSource:
    record = _exact_mapping(value, _SOURCE_KEYS, "source")
    evidence_record = _exact_mapping(record["evidence_binding"], _EVIDENCE_KEYS, "evidence_binding")
    automation_record = _exact_mapping(
        evidence_record["automation_terms"],
        _AUTOMATION_TERMS_KEYS,
        "automation_terms",
    )
    text_fields = _SOURCE_KEYS - {
        "allowed_download_hosts",
        "evidence_binding",
        "permissive",
        "taxonomy_hints",
        "zero_cost",
    }
    if any(not isinstance(record[key], str) for key in text_fields):
        raise TrustedCatalogError("Harvest trusted catalog source text fields must be strings.")
    if record["zero_cost"] is not True or record["permissive"] is not True:
        raise TrustedCatalogError("Harvest trusted catalog policy flags must be exactly true.")
    allowed_hosts = record["allowed_download_hosts"]
    taxonomy_hints = record["taxonomy_hints"]
    if (
        not isinstance(allowed_hosts, list)
        or not all(isinstance(value, str) for value in allowed_hosts)
        or not isinstance(taxonomy_hints, list)
        or not all(isinstance(value, str) for value in taxonomy_hints)
    ):
        raise TrustedCatalogError("Harvest trusted catalog tuple fields must be string arrays.")
    evidence_text_fields = _EVIDENCE_KEYS - {"source_http_status", "license_http_status", "automation_terms"}
    if any(not isinstance(evidence_record[key], str) for key in evidence_text_fields) or any(
        type(evidence_record[key]) is not int for key in ("source_http_status", "license_http_status")
    ):
        raise TrustedCatalogError("Harvest trusted catalog evidence field types are invalid.")
    automation_text_fields = _AUTOMATION_TERMS_KEYS - {
        "evidence_http_status",
        "limited_evidence",
        "matched_declaration",
    }
    if (
        any(not isinstance(automation_record[key], str) for key in automation_text_fields)
        or type(automation_record["evidence_http_status"]) is not int
        or type(automation_record["limited_evidence"]) is not bool
        or (
            automation_record["matched_declaration"] is not None
            and not isinstance(automation_record["matched_declaration"], str)
        )
    ):
        raise TrustedCatalogError("Harvest trusted catalog automation-terms field types are invalid.")
    evidence = CatalogEvidenceBinding(
        **{
            **evidence_record,
            "automation_terms": CatalogAutomationTermsBinding(**automation_record),
        }
    )
    return HarvestSource(
        **{
            **record,
            "allowed_download_hosts": tuple(allowed_hosts),
            "evidence_binding": evidence,
            "taxonomy_hints": tuple(taxonomy_hints),
        }
    )


def _exact_mapping(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise TrustedCatalogError(f"Harvest trusted catalog {label} fields do not match the schema.")
    return dict(value)


def _same_file_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
    )


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def validate_public_evidence_url(value: str) -> SplitResult:
    """Validate a public HTTP(S) evidence URL without resolving or contacting it."""

    parsed = _parse_http_url(value)
    validate_public_hostname(parsed.hostname or "")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Harvest evidence URLs cannot contain credentials.")
    return parsed


def validate_download_url(value: str, allowed_hosts: tuple[str, ...]) -> SplitResult:
    """Validate one direct/redirect URL against an exact public-host policy."""

    parsed = _parse_http_url(value)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Harvest download URLs cannot contain credentials.")
    host = (parsed.hostname or "").casefold().rstrip(".")
    validate_public_hostname(host)
    if host not in set(allowed_hosts):
        raise ValueError("Harvest download URL host is not in the source allowlist.")
    return parsed


def validate_public_hostname(host: str) -> None:
    normalized = host.casefold().rstrip(".")
    if not normalized or normalized == "localhost" or normalized.endswith(_BLOCKED_HOST_SUFFIXES):
        raise ValueError("Harvest host is local or otherwise blocked.")
    try:
        ipaddress.ip_address(normalized.strip("[]"))
    except ValueError:
        pass
    else:
        raise ValueError("Harvest host policy requires a DNS name, not an IP literal.")
    if HOST_PATTERN.fullmatch(normalized) is None:
        raise ValueError("Harvest host must be an exact public DNS name.")


def public_url(value: str) -> str:
    """Return query-, fragment-, and credential-free URL evidence."""

    parsed = validate_public_evidence_url(value)
    host = (parsed.hostname or "").casefold().rstrip(".")
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.casefold(), host, parsed.path or "/", "", ""))


def public_download_url(value: str, allowed_hosts: tuple[str, ...]) -> str:
    parsed = validate_download_url(value, allowed_hosts)
    host = (parsed.hostname or "").casefold().rstrip(".")
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.casefold(), host, parsed.path or "/", "", ""))


def url_identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def automation_terms_decision_identity(
    *,
    mode: str,
    evidence_url: str,
    content_sha256: str,
    matched_declaration: str | None,
    decision: str,
) -> str:
    """Bind the exact terms mode, URL, retained bytes, declaration, and decision."""

    return hashlib.sha256(
        "\n".join((mode, evidence_url, content_sha256, matched_declaration or "", decision)).encode("utf-8")
    ).hexdigest()


def _parse_http_url(value: str) -> SplitResult:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Harvest URL is missing or too long.")
    if any(ord(character) < 32 for character in value):
        raise ValueError("Harvest URLs cannot contain control characters.")
    try:
        parsed = urlsplit(value.strip())
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Harvest URL is invalid.") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Harvest URLs must use HTTP(S) and include a host.")
    return parsed


def _validate_text(value: str, name: str, *, minimum: int, maximum: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"Harvest {name} must be text.")
    stripped = value.strip()
    if len(stripped) < minimum or len(stripped) > maximum:
        raise ValueError(f"Harvest {name} has an invalid length.")
    if any(ord(character) < 32 and character not in "\n\t" for character in stripped):
        raise ValueError(f"Harvest {name} cannot contain control characters.")


def _identity(value: Any) -> str:
    encoded = strict_json_dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Harvest evidence timestamp is invalid.") from exc
    if parsed.tzinfo is None:
        raise ValueError("Harvest evidence timestamp must include UTC offset.")
    return parsed.astimezone(timezone.utc)


__all__ = [
    "AUTOMATION_ALLOW_DECLARATIONS",
    "INITIAL_LICENSE_POLICY",
    "MAX_TRUSTED_CATALOG_BYTES",
    "MAX_TRUSTED_CATALOG_SOURCES",
    "SHA256_PATTERN",
    "SOURCE_ID_PATTERN",
    "TRUSTED_CATALOG_RELATIVE_PATH",
    "TRUSTED_CATALOG_SCHEMA",
    "CatalogAutomationTermsBinding",
    "CatalogEvidenceBinding",
    "HarvestSource",
    "TrustedCatalogError",
    "TrustedCatalogSnapshot",
    "automation_terms_decision_identity",
    "load_trusted_catalog",
    "load_trusted_catalog_snapshot",
    "public_download_url",
    "public_url",
    "trusted_catalog_identity",
    "trusted_catalog_record",
    "trusted_catalog_source_record",
    "url_identity",
    "validate_download_url",
    "validate_public_evidence_url",
    "validate_public_hostname",
]
