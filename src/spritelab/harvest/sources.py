"""Source and license records for harvested asset packs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from spritelab.dataset_maker.model import normalize_sprite_id

SOURCE_TYPES = (
    "manual_zip",
    "local_directory",
    "direct_zip_url",
    "direct_file_url",
    "kenney",
    "kenney_manual_zip",
    "opengameart_manual_zip",
    "itch_manual_zip",
)

KNOWN_LICENSES = (
    "cc0",
    "public_domain",
    "own_work",
    "cc_by",
    "cc_by_sa",
    "oga_by",
    "wtfpl",
    "mit",
    "apache_2",
    "bsd",
    "custom_permissive_confirmed",
    "unknown",
    "noncommercial",
    "no_derivatives",
    "all_rights_reserved",
    "custom_unreviewed",
)

TRAINING_ALLOWED_LICENSES = frozenset(
    {
        "cc0",
        "public_domain",
        "own_work",
        "cc_by",
        "oga_by",
        "cc_by_sa",
        "wtfpl",
        "mit",
        "apache_2",
        "bsd",
        "custom_permissive_confirmed",
    }
)

ATTRIBUTION_REQUIRED_LICENSES = frozenset({"cc_by", "cc_by_sa", "oga_by", "mit", "apache_2", "bsd"})
SHARE_ALIKE_LICENSES = frozenset({"cc_by_sa"})

_LICENSE_ALIASES = {
    "cc-0": "cc0",
    "cc_0": "cc0",
    "creative_commons_zero": "cc0",
    "publicdomain": "public_domain",
    "pd": "public_domain",
    "cc-by": "cc_by",
    "cc-by-3.0": "cc_by",
    "cc-by-4.0": "cc_by",
    "cc_by_3_0": "cc_by",
    "cc_by_3.0": "cc_by",
    "cc_by_4.0": "cc_by",
    "cc_by_4_0": "cc_by",
    "cc-by-sa": "cc_by_sa",
    "oga-by": "oga_by",
    "apache": "apache_2",
    "apache2": "apache_2",
    "apache-2.0": "apache_2",
    "apache_2.0": "apache_2",
    "bsd3": "bsd",
    "bsd_3_clause": "bsd",
    "nc": "noncommercial",
    "cc_by_nc": "noncommercial",
    "cc-by-nc": "noncommercial",
    "nd": "no_derivatives",
    "cc_by_nd": "no_derivatives",
    "arr": "all_rights_reserved",
    "copyright": "all_rights_reserved",
    "custom": "custom_unreviewed",
}


def normalize_source_id(value: str) -> str:
    """Normalize a source ID to a filesystem-safe lowercase identifier."""

    return normalize_sprite_id(value)


def normalize_license_name(value: str) -> str:
    """Normalize a free-form license name to a known token, else ``unknown``."""

    token = normalize_sprite_id(str(value))
    token = _LICENSE_ALIASES.get(token, token)
    return token if token in KNOWN_LICENSES else "unknown"


def is_license_allowed_for_training(license_name: str) -> bool:
    """Return whether a normalized license is on the safe training allowlist."""

    return normalize_license_name(license_name) in TRAINING_ALLOWED_LICENSES


def license_requires_attribution(license_name: str) -> bool:
    """Return whether a normalized license requires attribution metadata."""

    return normalize_license_name(license_name) in ATTRIBUTION_REQUIRED_LICENSES


@dataclass(frozen=True)
class SourceLicense:
    license: str
    license_url: str = ""
    attribution_required: bool = False
    commercial_allowed: bool = True
    derivatives_allowed: bool = True
    share_alike: bool = False
    no_ai_training_flag: bool = False
    user_confirmed: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        normalized = normalize_license_name(self.license)
        object.__setattr__(self, "license", normalized)
        if normalized in ATTRIBUTION_REQUIRED_LICENSES:
            object.__setattr__(self, "attribution_required", True)
        if normalized in SHARE_ALIKE_LICENSES:
            object.__setattr__(self, "share_alike", True)


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_name: str
    source_type: str
    source_url: str = ""
    download_url: str = ""
    download_kind: str = ""
    local_archive_path: str = ""
    local_root_path: str = ""
    author: str = ""
    license: SourceLicense = field(default_factory=lambda: SourceLicense(license="unknown"))
    created_at: str = ""
    sha256: str = ""
    download_sha256: str = ""
    download_size_bytes: int = 0
    downloaded_at_utc: str = ""
    original_filename: str = ""
    archive_member_summary: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", normalize_source_id(self.source_id))
        object.__setattr__(self, "source_name", str(self.source_name).strip())
        object.__setattr__(self, "source_type", str(self.source_type).strip().lower())
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_timestamp())
        if self.download_sha256 and not self.sha256:
            object.__setattr__(self, "sha256", self.download_sha256)  # legacy reader compatibility


def utc_timestamp() -> str:
    """Return a compact UTC ISO timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def source_record_to_dict(record: SourceRecord) -> dict[str, Any]:
    """Return a JSON-serializable source record dictionary."""

    return asdict(record)


def source_record_from_dict(data: Mapping[str, Any]) -> SourceRecord:
    """Rebuild a source record from a dictionary loaded from JSON."""

    payload = dict(data)
    license_data = payload.pop("license", {})
    if isinstance(license_data, str):
        license_record = SourceLicense(license=license_data)
    else:
        license_record = SourceLicense(**dict(license_data))
    return SourceRecord(license=license_record, **payload)


def source_warnings(record: SourceRecord) -> list[str]:
    """Return non-blocking provenance/license warnings for a source."""

    warnings: list[str] = []
    license_name = record.license.license
    if license_name in ATTRIBUTION_REQUIRED_LICENSES or record.license.attribution_required:
        warnings.append(f"{record.source_id}: license {license_name} requires attribution.")
        if not record.author:
            warnings.append(f"{record.source_id}: attribution required but author is missing.")
    if license_name in SHARE_ALIKE_LICENSES or record.license.share_alike:
        warnings.append(f"{record.source_id}: license {license_name} is share-alike.")
    if license_name.startswith("custom"):
        warnings.append(f"{record.source_id}: custom license; review manually.")
    if not record.source_url:
        warnings.append(f"{record.source_id}: source URL is missing.")
    if not record.author:
        warnings.append(f"{record.source_id}: author is missing.")
    if record.license.no_ai_training_flag:
        warnings.append(f"{record.source_id}: source is flagged no-AI-training.")
    if not record.license.user_confirmed:
        warnings.append(f"{record.source_id}: license has not been user-confirmed.")
    return warnings


def make_kenney_source(
    source_id: str,
    source_name: str,
    *,
    source_url: str = "",
    download_url: str = "",
    local_archive_path: str = "",
) -> SourceRecord:
    """Conservative Kenney source prefill: CC0, author Kenney, no attribution."""

    return SourceRecord(
        source_id=source_id,
        source_name=source_name,
        source_type="kenney",
        source_url=source_url,
        download_url=download_url,
        local_archive_path=local_archive_path,
        author="Kenney",
        license=SourceLicense(
            license="cc0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            attribution_required=False,
            user_confirmed=True,
        ),
        notes="Kenney packs are usually CC0; verify the pack page.",
    )
