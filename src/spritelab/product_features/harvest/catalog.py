"""Strict, administrator-owned Harvest source catalog records."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from spritelab.product_core.events import strict_json_dumps
from spritelab.utils.safe_fs import UnsafeFilesystemOperation, require_confined_path

SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HOST_PATTERN = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
TAXONOMY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

INITIAL_LICENSE_POLICY = frozenset({"cc0-1.0", "public-domain"})
_UNKNOWN_CREATORS = frozenset({"unknown", "n/a", "na", "none", "anonymous", "unspecified"})
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost", ".home", ".arpa")
TRUSTED_CATALOG_SCHEMA = "spritelab.harvest.trusted-catalog.v1"
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
        "attestation_identity_sha256",
    }
)


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
            "schema_version": "spritelab.harvest.source.v2",
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
    attestation_identity_sha256: str

    def validate(self, source_page: str, license_evidence_url: str, *, now: datetime | None = None) -> None:
        if SOURCE_ID_PATTERN.fullmatch(self.verifier_id) is None:
            raise ValueError("Harvest evidence verifier identifier is invalid.")
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
            "schema_version": "spritelab.harvest.catalog-evidence-attestation.v1",
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
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.catalog-evidence-binding.v1",
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


def load_trusted_catalog(project_root: str | Path) -> tuple[HarvestSource, ...]:
    """Passively load the exact repository-local trusted catalog, if present.

    The loader never creates a directory, follows a link, resolves a hostname,
    or performs a network request. An absent catalog is an empty catalog; an
    existing malformed or unsafe catalog fails closed.
    """

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(project_root))))
    try:
        catalog_path = require_confined_path(root / TRUSTED_CATALOG_RELATIVE_PATH, root)
        if not _catalog_path_exists_safely(catalog_path, root):
            return ()
        payload = _read_catalog_bytes(catalog_path)
        parsed = _strict_catalog_json(payload)
        sources = _parse_trusted_catalog(parsed)
    except (OSError, UnicodeError, UnsafeFilesystemOperation, ValueError) as exc:
        if isinstance(exc, TrustedCatalogError):
            raise
        raise TrustedCatalogError("Harvest trusted catalog is unsafe or invalid.") from exc
    return sources


def _catalog_path_exists_safely(path: Path, root: Path) -> bool:
    current = root
    relative_parts = path.relative_to(root).parts
    for index, part in enumerate(relative_parts):
        current = current / part
        if not os.path.lexists(current):
            return False
        metadata = current.lstat()
        if _metadata_is_link_or_reparse(metadata):
            raise TrustedCatalogError("Harvest trusted catalog path crosses a link or reparse point.")
        is_target = index == len(relative_parts) - 1
        if not is_target and (not stat.S_ISDIR(metadata.st_mode) or current.is_mount()):
            raise TrustedCatalogError("Harvest trusted catalog path crosses an unsafe directory seam.")
    return True


def _read_catalog_bytes(path: Path) -> bytes:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or _metadata_is_link_or_reparse(before)
        or before.st_nlink != 1
        or not 1 <= before.st_size <= MAX_TRUSTED_CATALOG_BYTES
    ):
        raise TrustedCatalogError("Harvest trusted catalog must be one bounded, single-link regular file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_file_metadata(before, opened):
            raise TrustedCatalogError("Harvest trusted catalog changed while it was opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(MAX_TRUSTED_CATALOG_BYTES + 1)
        after_opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    if (
        len(payload) != before.st_size
        or len(payload) > MAX_TRUSTED_CATALOG_BYTES
        or not _same_file_metadata(before, after_opened)
        or not _same_file_metadata(before, after_path)
    ):
        raise TrustedCatalogError("Harvest trusted catalog changed while it was read.")
    return payload


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
    evidence_text_fields = _EVIDENCE_KEYS - {"source_http_status", "license_http_status"}
    if any(not isinstance(evidence_record[key], str) for key in evidence_text_fields) or any(
        type(evidence_record[key]) is not int for key in ("source_http_status", "license_http_status")
    ):
        raise TrustedCatalogError("Harvest trusted catalog evidence field types are invalid.")
    evidence = CatalogEvidenceBinding(**evidence_record)
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
    "INITIAL_LICENSE_POLICY",
    "MAX_TRUSTED_CATALOG_BYTES",
    "MAX_TRUSTED_CATALOG_SOURCES",
    "SHA256_PATTERN",
    "SOURCE_ID_PATTERN",
    "TRUSTED_CATALOG_RELATIVE_PATH",
    "TRUSTED_CATALOG_SCHEMA",
    "CatalogEvidenceBinding",
    "HarvestSource",
    "TrustedCatalogError",
    "load_trusted_catalog",
    "public_download_url",
    "public_url",
    "trusted_catalog_identity",
    "url_identity",
    "validate_download_url",
    "validate_public_evidence_url",
    "validate_public_hostname",
]
