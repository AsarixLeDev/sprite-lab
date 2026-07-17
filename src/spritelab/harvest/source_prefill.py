"""Conservative, network-free source prefills for Harvest inputs.

The prefill layer intentionally derives only metadata that follows from the
submitted public pack-page URL and a small built-in platform profile.  It does
not scrape, download, confirm a license, or satisfy any authorization gate.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from spritelab.harvest.sources import normalize_source_id

SOURCE_PREFILL_SCHEMA = "spritelab.harvest.source-prefill.v1"
CC0_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SEPARATORS = re.compile(r"[-_]+")
_NON_ID = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class SourcePreset:
    preset_id: str
    label: str
    description: str
    source_type: str
    fixed_creator: str = ""
    license_name: str = "unknown"
    license_id: str = ""
    license_evidence_url: str = ""
    terms_evidence_url: str = ""
    attribution_text: str = ""

    def public_dict(self) -> dict[str, str]:
        return {
            "preset_id": self.preset_id,
            "label": self.label,
            "description": self.description,
        }


@dataclass(frozen=True)
class SourcePrefill:
    preset_id: str
    preset_label: str
    recognized_source: bool
    source_page: str
    source_id: str
    title: str
    creator: str
    source_type: str
    license_name: str
    license_id: str
    license_evidence_url: str
    terms_evidence_url: str
    direct_download_url: str
    attribution_text: str
    taxonomy_hints: tuple[str, ...]
    review_fields: tuple[str, ...]
    guidance: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SOURCE_PREFILL_SCHEMA
        payload["taxonomy_hints"] = list(self.taxonomy_hints)
        payload["review_fields"] = list(self.review_fields)
        return payload


_PRESETS: dict[str, SourcePreset] = {
    "opengameart": SourcePreset(
        preset_id="opengameart",
        label="OpenGameArt",
        description="Derives the pack title and ID; the per-pack creator and license still need review.",
        source_type="opengameart_manual_zip",
    ),
    "kenney": SourcePreset(
        preset_id="kenney",
        label="Kenney",
        description="Prefills Kenney, CC0 evidence, attribution, and the linked site terms.",
        source_type="kenney_manual_zip",
        fixed_creator="Kenney",
        license_name="cc0",
        license_id="cc0-1.0",
        license_evidence_url=CC0_LICENSE_URL,
        terms_evidence_url="https://kenney.nl/terms-of-service",
        attribution_text="Kenney",
    ),
    "itchio": SourcePreset(
        preset_id="itchio",
        label="itch.io",
        description="Derives the creator handle, title, ID, and site terms; every asset license remains pack-specific.",
        source_type="itch_manual_zip",
        terms_evidence_url="https://itch.io/docs/legal/terms",
    ),
    "generic": SourcePreset(
        preset_id="generic",
        label="Other creator page",
        description="Derives a title and stable ID from any public HTTPS pack page.",
        source_type="manual_zip",
    ),
}


def available_source_presets() -> tuple[dict[str, str], ...]:
    """Return stable UI/CLI descriptions without any network activity."""

    return tuple(_PRESETS[name].public_dict() for name in ("opengameart", "kenney", "itchio", "generic"))


def source_preset_ids(*, include_auto: bool = True) -> tuple[str, ...]:
    values = tuple(_PRESETS)
    return ("auto", *values) if include_auto else values


def build_source_prefill(source_page: str, *, preset_id: str = "auto") -> SourcePrefill:
    """Build a reviewable source draft from one pack-page URL.

    Explicit values supplied later by the CLI or browser always take
    precedence.  In particular, an OpenGameArt or itch.io hostname never
    implies a license.
    """

    canonical, host, path = _canonical_source_page(source_page)
    requested = str(preset_id or "auto").strip().casefold()
    if requested not in source_preset_ids():
        raise ValueError(f"unknown Harvest source preset: {preset_id}")
    detected = _detect_preset(host)
    resolved = detected if requested == "auto" else requested
    if resolved != "generic" and not _host_matches(resolved, host):
        raise ValueError(f"{_PRESETS[resolved].label} preset does not match the submitted source host")

    preset = _PRESETS[resolved]
    slug = _page_slug(path, host)
    title_slug = re.sub(r"\s+\d+$", "", slug) if resolved == "opengameart" else slug
    title = _title_from_slug(title_slug)
    creator = preset.fixed_creator
    if resolved == "itchio":
        creator = _itch_creator(host)
    source_id = _source_id(resolved, slug, creator)

    review_fields = ["direct_download_url"]
    if not creator or resolved == "itchio":
        review_fields.append("creator")
    if not preset.license_id:
        review_fields.extend(("license_id", "license_evidence_url"))
    if not preset.terms_evidence_url:
        review_fields.append("terms_evidence_url")
    if not preset.attribution_text:
        review_fields.append("attribution_text")

    if resolved == "kenney":
        guidance = (
            "Kenney defaults were filled. Paste the exact Download link from this pack page, then review every field."
        )
    elif resolved == "opengameart":
        guidance = "OpenGameArt licenses vary by pack. Review the displayed author and license, then paste the exact file link."
    elif resolved == "itchio":
        guidance = (
            "itch.io licenses vary by creator. Confirm this pack is zero-cost CC0/public-domain before continuing."
        )
    else:
        guidance = (
            "A generic draft was created. Review provenance, license, terms, and the exact creator-posted file link."
        )

    return SourcePrefill(
        preset_id=resolved,
        preset_label=preset.label,
        recognized_source=detected != "generic",
        source_page=canonical,
        source_id=source_id,
        title=title,
        creator=creator,
        source_type=preset.source_type,
        license_name=preset.license_name,
        license_id=preset.license_id,
        license_evidence_url=preset.license_evidence_url,
        terms_evidence_url=preset.terms_evidence_url,
        direct_download_url="",
        attribution_text=preset.attribution_text or creator,
        taxonomy_hints=(),
        review_fields=tuple(dict.fromkeys(review_fields)),
        guidance=guidance,
    )


def default_license_evidence_url(license_name: str) -> str:
    """Return an unambiguous canonical evidence URL when one is known."""

    return CC0_LICENSE_URL if str(license_name).strip().casefold() == "cc0" else ""


def _detect_preset(host: str) -> str:
    for preset_id in ("opengameart", "kenney", "itchio"):
        if _host_matches(preset_id, host):
            return preset_id
    return "generic"


def _host_matches(preset_id: str, host: str) -> bool:
    if preset_id == "opengameart":
        return host in {"opengameart.org", "www.opengameart.org"}
    if preset_id == "kenney":
        return host in {"kenney.nl", "www.kenney.nl"}
    if preset_id == "itchio":
        return host == "itch.io" or host.endswith(".itch.io")
    return True


def _canonical_source_page(value: str) -> tuple[str, str, str]:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError("source page must be a public HTTPS URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme.casefold() != "https" or parsed.username is not None or parsed.password is not None:
        raise ValueError("source page must be a public HTTPS URL without credentials")
    if parsed.query or parsed.fragment or "\\" in parsed.path:
        raise ValueError("source page cannot contain a query, fragment, or backslash")
    host = (parsed.hostname or "").casefold().rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("source page must use a public hostname, not an IP literal")
    if len(host) > 253 or "." not in host or any(_HOST_LABEL.fullmatch(label) is None for label in host.split(".")):
        raise ValueError("source page hostname is invalid")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("source page port is invalid") from exc
    if port not in (None, 443):
        raise ValueError("source page must use the default HTTPS port")
    path = parsed.path or "/"
    canonical = urlunsplit(("https", host, path, "", ""))
    return canonical, host, path


def _page_slug(path: str, host: str) -> str:
    segments = [unquote(segment).strip() for segment in path.split("/") if segment.strip()]
    candidate = segments[-1] if segments else host.split(".")[0]
    candidate = _SEPARATORS.sub(" ", candidate)
    candidate = " ".join(candidate.split())
    return candidate[:120] or "asset pack"


def _title_from_slug(slug: str) -> str:
    words = " ".join(slug.split())
    title = words.title()
    return title if len(title) >= 4 else f"{title} Art Pack"


def _id_segment(value: str, *, max_parts: int) -> str:
    normalized = _NON_ID.sub("-", value.casefold()).strip("-")
    parts = [part for part in re.split(r"-+", normalized) if part][:max_parts]
    segment = "-".join(parts)[:100].strip("-") or "asset-pack"
    return f"pack-{segment}" if segment[0].isdigit() else segment


def _source_id(preset_id: str, slug: str, creator: str) -> str:
    if preset_id == "itchio" and creator:
        value = f"itchio.{_id_segment(creator, max_parts=2)}.{_id_segment(slug, max_parts=3)}"
    else:
        prefix = {"opengameart": "oga", "kenney": "kenney", "generic": "source"}.get(preset_id, preset_id)
        value = f"{prefix}.{_id_segment(slug, max_parts=6)}"
    return normalize_source_id(value)


def _itch_creator(host: str) -> str:
    if not host.endswith(".itch.io"):
        return ""
    handle = host[: -len(".itch.io")].split(".")[-1]
    return _title_from_slug(handle)


__all__ = [
    "CC0_LICENSE_URL",
    "SOURCE_PREFILL_SCHEMA",
    "SourcePrefill",
    "available_source_presets",
    "build_source_prefill",
    "default_license_evidence_url",
    "source_preset_ids",
]
