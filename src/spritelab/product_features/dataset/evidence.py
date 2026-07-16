"""Translate simple user evidence files into conservative backend metadata."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from spritelab.harvest.sources import is_license_allowed_for_training, normalize_license_name

SOURCE_FILENAMES = (
    "source.yaml",
    "source.yml",
    "metadata.yaml",
    "source.txt",
    "readme",
    "readme.txt",
    "credits.txt",
    "attribution.txt",
)
LICENSE_FILENAMES = (
    "license.yaml",
    "license.yml",
    "license",
    "license.txt",
    "copying",
    "readme",
    "readme.txt",
)
_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_DEDICATED_LICENSE_FILENAMES = frozenset({"license", "license.txt", "license.yaml", "license.yml", "copying"})


def evidence_for_image(image_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the nearest independent source and license evidence for an image."""

    source_records = [
        parse_source_evidence(path, root) for path in _nearest_named_files(image_path.parent, root, SOURCE_FILENAMES)
    ]
    license_records = [
        parse_license_evidence(path, root) for path in _nearest_named_files(image_path.parent, root, LICENSE_FILENAMES)
    ]
    return _aggregate_source_records(source_records), _aggregate_license_records(license_records)


def parse_source_evidence(path: Path | None, root: Path) -> dict[str, Any]:
    if path is None:
        return {"present": False, "path": None, "source_name": None, "creator": None, "source_url": None}
    raw, data = _read_evidence(path)
    urls = _URL_RE.findall(raw)
    mapping = data if isinstance(data, Mapping) else {}
    source_url = _first_text(mapping, "source_url", "url", "homepage", "download_url") or (
        urls[0].rstrip(".,;)") if urls else None
    )
    source_name = _first_text(mapping, "source_name", "name", "title")
    creator = _first_text(mapping, "creator", "author", "publisher")
    if path.suffix.casefold() == ".txt":
        source_name = source_name or _prefixed_value(raw, "name", "title", "pack", "pack title")
        creator = creator or _prefixed_value(raw, "creator", "author", "publisher")
    return {
        "present": bool(raw.strip()),
        "path": _relative(path, root),
        "evidence_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "source_name": source_name,
        "creator": creator,
        "source_url": source_url,
        "notes": raw.strip()[:4000],
        "interpretation": "user_supplied_evidence",
    }


def parse_license_evidence(path: Path | None, root: Path) -> dict[str, Any]:
    if path is None:
        return {
            "present": False,
            "path": None,
            "license": "unknown",
            "license_url": None,
            "training_allowed": False,
        }
    raw, data = _read_evidence(path)
    mapping = data if isinstance(data, Mapping) else {}
    urls = _URL_RE.findall(raw)
    license_url = _first_text(mapping, "license_url", "url") or (urls[0].rstrip(".,;)") if urls else None)
    declared = _first_text(mapping, "license", "spdx", "spdx_id", "identifier", "id")
    public_domain = bool(mapping.get("public_domain")) if mapping else False
    normalized = _recognize_license(declared or raw, public_domain=public_domain)
    return {
        "present": bool(raw.strip()),
        "path": _relative(path, root),
        "evidence_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "license": normalized,
        "license_url": license_url,
        "training_allowed": is_license_allowed_for_training(normalized),
        "raw_excerpt": raw.strip()[:4000],
        "interpretation": "recognized_allowlist" if is_license_allowed_for_training(normalized) else "requires_review",
    }


def evidence_digest_payload(source: Mapping[str, Any], license_record: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable evidence fields used to invalidate resumable preprocessing."""

    return {
        "source": dict(source),
        "license": dict(license_record),
    }


def _nearest_named_files(directory: Path, root: Path, names: tuple[str, ...]) -> list[Path]:
    priorities = {name.casefold(): index for index, name in enumerate(names)}
    wanted = set(priorities)
    current = directory.resolve()
    boundary = root.resolve()
    while True:
        try:
            candidates = sorted(
                (path for path in current.iterdir() if path.is_file() and path.name.casefold() in wanted),
                key=lambda path: (priorities[path.name.casefold()], path.name.casefold()),
            )
        except OSError:
            candidates = []
        if candidates:
            return candidates
        if current == boundary:
            return []
        if boundary not in current.parents:
            return []
        current = current.parent


def _aggregate_source_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        result = parse_source_evidence(None, Path("."))
        result.update(evidence_records=[], conflict=False, conflicting_fields=[], conflict_details=[])
        return result
    primary = next(
        (record for record in records if any(record.get(field) for field in ("source_name", "creator", "source_url"))),
        records[0],
    )
    conflicting_fields: list[str] = []
    conflict_details: list[dict[str, Any]] = []
    for field in ("source_name", "creator", "source_url"):
        values: dict[str, list[str]] = {}
        for record in records:
            raw = record.get(field)
            if not raw:
                continue
            normalized = _normalize_source_value(field, str(raw))
            values.setdefault(normalized, []).append(str(record.get("path") or ""))
        if len(values) > 1:
            conflicting_fields.append(field)
            conflict_details.append(
                {
                    "field": field,
                    "values": [
                        {"normalized": value, "evidence_paths": sorted(paths)}
                        for value, paths in sorted(values.items())
                    ],
                }
            )
    result = dict(primary)
    result.update(
        evidence_records=[dict(record) for record in records],
        conflict=bool(conflicting_fields),
        conflicting_fields=conflicting_fields,
        conflict_details=conflict_details,
    )
    if conflicting_fields:
        result["interpretation"] = "conflicting_user_supplied_evidence"
    return result


def _aggregate_license_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        result = parse_license_evidence(None, Path("."))
        result.update(evidence_records=[], conflict=False, conflicting_fields=[], conflict_details=[])
        return result
    recognized = [record for record in records if record.get("license") not in {None, "", "unknown"}]
    primary = recognized[0] if recognized else records[0]
    by_license: dict[str, list[str]] = {}
    for record in recognized:
        by_license.setdefault(str(record["license"]), []).append(str(record.get("path") or ""))
    for record in records:
        if record in recognized or not _is_unrecognized_dedicated_license(record):
            continue
        by_license.setdefault("unrecognized_dedicated_license", []).append(str(record.get("path") or ""))
    conflict = len(by_license) > 1
    result = dict(primary)
    result.update(
        evidence_records=[dict(record) for record in records],
        conflict=conflict,
        conflicting_fields=["license"] if conflict else [],
        conflict_details=(
            [
                {
                    "field": "license",
                    "values": [
                        {"normalized": value, "evidence_paths": sorted(paths)}
                        for value, paths in sorted(by_license.items())
                    ],
                }
            ]
            if conflict
            else []
        ),
    )
    if conflict:
        result["training_allowed"] = False
        result["interpretation"] = "conflicting_user_supplied_evidence"
    return result


def _is_unrecognized_dedicated_license(record: Mapping[str, Any]) -> bool:
    path = Path(str(record.get("path") or ""))
    excerpt = str(record.get("raw_excerpt") or "")
    return (
        path.name.casefold() in _DEDICATED_LICENSE_FILENAMES
        and str(record.get("license") or "unknown") == "unknown"
        and bool(excerpt.strip())
    )


def _normalize_source_value(field: str, value: str) -> str:
    if field != "source_url":
        return " ".join(value.split()).casefold()
    trimmed = value.strip()
    try:
        parsed = urlsplit(trimmed)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return trimmed
    if not parsed.scheme or not parsed.netloc or not hostname:
        return trimmed
    userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
    normalized_host = f"[{hostname.casefold()}]" if ":" in hostname else hostname.casefold()
    normalized_port = f":{port}" if port is not None else ""
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            f"{userinfo}{normalized_host}{normalized_port}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def _read_evidence(path: Path) -> tuple[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return "", {}
    if path.suffix.casefold() in {".yaml", ".yml"}:
        try:
            return raw, yaml.safe_load(raw) or {}
        except yaml.YAMLError:
            return raw, {}
    return raw, {}


def _recognize_license(value: str, *, public_domain: bool) -> str:
    if public_domain:
        return "public_domain"
    text = str(value).strip()
    normalized = normalize_license_name(text)
    if normalized != "unknown":
        return normalized
    folded = text.casefold().replace("_", "-")
    rules = (
        ("publicdomain/zero", "cc0"),
        ("creative commons zero", "cc0"),
        ("cc0", "cc0"),
        ("public domain", "public_domain"),
        ("licenses/by-sa/", "cc_by_sa"),
        ("cc-by-sa", "cc_by_sa"),
        ("creative commons attribution-sharealike", "cc_by_sa"),
        ("licenses/by/", "cc_by"),
        ("cc-by", "cc_by"),
        ("creative commons attribution", "cc_by"),
        ("mit license", "mit"),
        ("opensource.org/licenses/mit", "mit"),
        ("apache license", "apache_2"),
        ("apache.org/licenses/license-2.0", "apache_2"),
        ("bsd license", "bsd"),
        ("wtfpl", "wtfpl"),
    )
    for needle, result in rules:
        if needle in folded:
            return result
    return "unknown"


def _first_text(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


def _prefixed_value(raw: str, *prefixes: str) -> str | None:
    wanted = {prefix.casefold() for prefix in prefixes}
    for line in raw.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() in wanted and value.strip():
            return value.strip()
    return None


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
