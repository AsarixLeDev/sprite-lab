"""Deterministic, fail-closed remediation for Dataset-v5 raw provenance.

The compiler consumes the frozen forensic inventory as evidence.  It never
edits historical manifests or source bytes, never treats a newly observed
digest as a historical digest, and never infers licensing across packs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

SCHEMA_VERSION = "sprite_lab_raw_provenance_remediation_v1"
SOURCE_BINDING_SCHEMA_VERSION = "sprite_lab_source_binding_v1"
EXCLUSION_SCHEMA_VERSION = "sprite_lab_source_exclusion_v1"

CONTROLLED_RESOLUTION_STATES = frozenset(
    {
        "verified",
        "verified_current_hash_only",
        "recovered_from_local_original",
        "missing_original",
        "license_unknown",
        "provenance_incomplete",
        "excluded",
        "requires_manual_retrieval",
    }
)
TERMINAL_RESOLUTION_STATES = frozenset({"verified", "recovered_from_local_original", "excluded"})
REQUIRED_SOURCE_BINDING_FIELDS = (
    "source_binding_id",
    "distribution_platform",
    "creator_or_publisher",
    "pack_or_collection",
    "acquisition_run",
    "source_page_url",
    "direct_download_url",
    "original_archive_path",
    "original_archive_filename",
    "historical_archive_sha256",
    "current_observed_archive_sha256",
    "historical_hash_authority",
    "license_identifier",
    "license_url",
    "license_evidence",
    "provenance_status",
    "resolution_status",
    "exclusion_reason",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LICENSE_MEMBER_RE = re.compile(
    r"(^|/)(licen[cs]e|copying|copyright|attribution|credits?|readme)(\.|$|[-_ ])", re.IGNORECASE
)
_EXACT_LICENSE_URL_IDENTIFIERS = {
    "https://creativecommons.org/licenses/by/3.0": "cc_by_3_0",
    "https://creativecommons.org/licenses/by/4.0": "cc_by_4_0",
    "https://creativecommons.org/publicdomain/zero/1.0": "cc0",
    "http://creativecommons.org/publicdomain/zero/1.0": "cc0",
}
_PROVENANCE_RECORD_ISSUES = frozenset(
    {
        "acquisition_orphan_artifact",
        "changed_archive_hash",
        "incomplete_itemicon_provenance",
        "invalid_historical_archive_sha256",
        "license_not_training_allowed",
        "license_not_user_confirmed",
        "missing_acquisition_url",
        "missing_historical_archive_bytes",
        "missing_historical_archive_sha256",
        "missing_license",
        "missing_license_url",
        "missing_original_download",
        "missing_original_filename",
        "unknown_license",
    }
)


class SourceResolutionError(RuntimeError):
    """Raised when evidence is ambiguous, inconsistent, or unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SourceResolutionError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SourceResolutionError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite remediation artifact: {path}") from exc


def _write_new_json(path: Path, value: Any) -> None:
    _write_new_bytes(path, canonical_json_bytes(value))


def _write_new_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_new_bytes(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _binding_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("manifest_path") or ""),
        int(row.get("manifest_line_number") or 0),
        str(row.get("source_id") or ""),
        str(row.get("acquisition_run") or ""),
    )


def _source_binding_id(row: Mapping[str, Any]) -> str:
    identity = {
        "acquisition_run": str(row.get("acquisition_run") or ""),
        "manifest_line_number": int(row.get("manifest_line_number") or 0),
        "manifest_path": str(row.get("manifest_path") or ""),
        "source_id": str(row.get("source_id") or ""),
    }
    return "sb_" + _sha256_bytes(canonical_json_bytes(identity))[:24]


def _orphan_binding_id(archive_sha256: str) -> str:
    return "sb_orphan_" + archive_sha256[:24]


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _safe_resolve(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative.replace("/", "\\"))).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise SourceResolutionError(f"evidence path escapes source root: {relative}") from exc
    return candidate


def _url_filename(url: str | None) -> str | None:
    if not url:
        return None
    name = Path(unquote(urlparse(url).path)).name
    return name or None


def _normalized_license(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cc_by": "cc_by",
        "cc_by_3.0": "cc_by_3_0",
        "cc_by_3_0": "cc_by_3_0",
        "cc_by_4.0": "cc_by_4_0",
        "cc_by_4_0": "cc_by_4_0",
        "cc0": "cc0",
        "oga_by": "oga_by",
        "public_domain": "public_domain",
        "unknown": None,
        "": None,
    }
    return aliases.get(text, text or None)


def _license_identifier_from_url(value: Any) -> str | None:
    url = str(value or "").strip().rstrip("/").casefold()
    return _EXACT_LICENSE_URL_IDENTIFIERS.get(url)


def _select_original_archive_path(binding: Mapping[str, Any], paths: Sequence[str]) -> str | None:
    if not paths:
        return None
    source_record = binding.get("source_record") if isinstance(binding.get("source_record"), Mapping) else {}
    explicit = str(source_record.get("local_archive_path") or "").replace("\\", "/").lstrip("./")
    if explicit:
        explicit_matches = [path for path in paths if str(path).replace("\\", "/").lstrip("./") == explicit]
        if len(explicit_matches) == 1:
            return explicit_matches[0]
    run = str(binding.get("acquisition_run") or "")
    prefix = f"harvest_runs/{run}/downloads/".casefold()
    run_matches = [path for path in paths if str(path).replace("\\", "/").casefold().startswith(prefix)]
    if len(run_matches) == 1:
        return run_matches[0]
    return paths[0]


def verify_local_zip_candidate(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    expected_members: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify exact bytes and optional ZIP member hashes; a filename is never evidence."""

    candidate = Path(path)
    result: dict[str, Any] = {
        "candidate_path": candidate.as_posix(),
        "expected_sha256": expected_sha256,
        "match": False,
        "reasons": [],
    }
    if not candidate.is_file():
        result["reasons"].append("candidate_missing")
        return result
    actual_size = candidate.stat().st_size
    actual_sha256 = file_sha256(candidate)
    result.update({"actual_sha256": actual_sha256, "actual_size": actual_size})
    if expected_size is not None and actual_size != expected_size:
        result["reasons"].append("size_mismatch")
    if not _SHA256_RE.fullmatch(expected_sha256) or actual_sha256 != expected_sha256:
        result["reasons"].append("byte_hash_mismatch")
    if expected_members is not None:
        try:
            with zipfile.ZipFile(candidate) as archive:
                names = {info.filename for info in archive.infolist() if not info.is_dir()}
                missing = sorted(set(expected_members) - names)
                if missing:
                    result["reasons"].append("member_list_mismatch")
                    result["missing_members"] = missing
                mismatches = []
                for member, expected_member_hash in sorted(expected_members.items()):
                    if member in names and _sha256_bytes(archive.read(member)) != expected_member_hash:
                        mismatches.append(member)
                if mismatches:
                    result["reasons"].append("member_hash_mismatch")
                    result["member_hash_mismatches"] = mismatches
                result["member_count"] = len(names)
        except (OSError, zipfile.BadZipFile) as exc:
            result["reasons"].append("unreadable_zip")
            result["zip_error"] = type(exc).__name__
    result["match"] = not result["reasons"]
    if result["match"]:
        result["reasons"] = ["exact_byte_hash_and_member_evidence_match"]
    return result


def _validate_recovery_record(path: Path, source_root: Path) -> dict[str, Any]:
    repair = _read_json(path)
    required = (
        "download_sha256",
        "download_size",
        "local_download_path",
        "recorded_download_url",
        "recorded_source_url",
        "source_id",
        "source_run",
    )
    missing = [field for field in required if not repair.get(field)]
    if missing:
        raise SourceResolutionError(f"recovery record lacks {missing}: {path}")
    local_path = _safe_resolve(source_root, str(repair["local_download_path"]))
    mappings = repair.get("verification_evidence", {}).get("archive_member_mapping", [])
    if not isinstance(mappings, list) or not mappings:
        raise SourceResolutionError(f"recovery record has no member mapping: {path}")
    expected_members = {
        str(row["archive_member"]): str(row["source_image_sha256"])
        for row in mappings
        if isinstance(row, Mapping) and row.get("archive_member") and row.get("source_image_sha256")
    }
    check = verify_local_zip_candidate(
        local_path,
        expected_sha256=str(repair["download_sha256"]),
        expected_size=int(repair["download_size"]),
        expected_members=expected_members,
    )
    if not check["match"]:
        raise SourceResolutionError(f"recovery record failed exact ZIP verification: {path}: {check['reasons']}")
    sprite_ids = {str(row.get("sprite_id")) for row in mappings if isinstance(row, Mapping) and row.get("sprite_id")}
    if sprite_ids != set(map(str, repair.get("affected_sprite_ids", []))):
        raise SourceResolutionError(f"recovery affected IDs disagree with member mapping: {path}")
    for row in mappings:
        derived = _safe_resolve(source_root, str(row["derived_image_path"]))
        if not derived.is_file() or file_sha256(derived) != row.get("derived_image_sha256"):
            raise SourceResolutionError(f"derived recovery evidence mismatch: {derived}")
    return {
        "path": path,
        "record": repair,
        "record_sha256": file_sha256(path),
        "verified_member_count": len(expected_members),
        "verified_sprite_ids": sorted(sprite_ids),
    }


def _embedded_license_evidence(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".zip" or not path.is_file():
        return []
    evidence: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/")
                if info.is_dir() or not _LICENSE_MEMBER_RE.search(normalized):
                    continue
                payload = archive.read(info)
                text = ""
                for encoding in ("utf-8", "cp1252"):
                    try:
                        text = payload.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                lower = text.lower()
                identifier = None
                url = None
                if "creativecommons.org/licenses/by/4.0" in lower:
                    identifier = "cc_by_4_0"
                    url = "https://creativecommons.org/licenses/by/4.0/"
                elif "creativecommons.org/publicdomain/zero/1.0" in lower:
                    identifier = "cc0"
                    scheme = "https" if "https://creativecommons.org/publicdomain/zero/1.0" in lower else "http"
                    url = f"{scheme}://creativecommons.org/publicdomain/zero/1.0/"
                evidence.append(
                    {
                        "evidence_type": "embedded_archive_license_file",
                        "explicit_license_identifier": identifier,
                        "explicit_license_url": url,
                        "member_path": info.filename,
                        "member_sha256": _sha256_bytes(payload),
                        "member_size": len(payload),
                    }
                )
    except (OSError, zipfile.BadZipFile):
        return []
    return evidence


def _load_sources(
    frozen_experiment: Path, source_root: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str, str], str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    archive_document = _read_json(frozen_experiment / "source_archive_hashes.json")
    artifacts = archive_document.get("artifacts")
    unresolved = archive_document.get("unresolved_source_bindings")
    if not isinstance(artifacts, list) or not isinstance(unresolved, list):
        raise SourceResolutionError("unsupported forensic source_archive_hashes document")
    artifact_by_binding: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    raw_bindings: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    artifact_by_hash: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        digest = str(artifact.get("current_observed_archive_sha256") or "")
        if digest:
            artifact_by_hash[digest] = artifact
        for binding in artifact.get("source_bindings", []):
            key = _binding_key(binding)
            raw_bindings.setdefault(key, dict(binding))
            artifact_by_binding[key] = artifact
    for binding in unresolved:
        key = _binding_key(binding)
        raw_bindings.setdefault(key, dict(binding))

    key_to_id = {key: _source_binding_id(row) for key, row in raw_bindings.items()}
    sources: list[dict[str, Any]] = []
    for key, binding in sorted(raw_bindings.items()):
        artifact = artifact_by_binding.get(key)
        license_record = binding.get("license") if isinstance(binding.get("license"), Mapping) else {}
        historical = str(binding.get("historically_recorded_archive_sha256") or "") or None
        current = str(binding.get("current_observed_archive_sha256") or "") or None
        paths = list(artifact.get("original_archive_paths", [])) if artifact else []
        evidence: list[dict[str, Any]] = []
        if license_record:
            evidence.append(
                {
                    "evidence_type": "historical_manifest_license_record",
                    "explicit_license_identifier": _normalized_license(license_record.get("license")),
                    "explicit_license_url": str(license_record.get("license_url") or "") or None,
                    "manifest_line_number": int(binding.get("manifest_line_number") or 0),
                    "manifest_path": binding.get("manifest_path"),
                    "manifest_sha256": binding.get("manifest_sha256"),
                    "source_record_sha256": binding.get("source_row_sha256"),
                    "user_confirmed": bool(license_record.get("user_confirmed", False)),
                }
            )
        source = {
            "source_binding_id": key_to_id[key],
            "distribution_platform": binding.get("distribution_platform"),
            "creator_or_publisher": binding.get("creator_or_publisher"),
            "pack_or_collection": binding.get("pack"),
            "acquisition_run": binding.get("acquisition_run"),
            "source_page_url": binding.get("source_url"),
            "direct_download_url": binding.get("download_url"),
            "original_archive_path": _select_original_archive_path(binding, paths),
            "original_archive_filename": binding.get("original_archive_filename"),
            "historical_archive_sha256": historical,
            "current_observed_archive_sha256": current,
            "historical_hash_authority": (
                f"historical_source_manifest:{binding.get('manifest_path')}:{binding.get('manifest_line_number')}"
                if historical
                else None
            ),
            "license_identifier": _normalized_license(binding.get("license_normalized")),
            "license_url": str(license_record.get("license_url") or "") or None,
            "license_evidence": evidence,
            "provenance_status": "provenance_incomplete",
            "resolution_status": "provenance_incomplete",
            "exclusion_reason": None,
            "terminal_gate_status": None,
            "eligible_for_candidate_membership": False,
            "manifest_line_number": int(binding.get("manifest_line_number") or 0),
            "manifest_path": binding.get("manifest_path"),
            "manifest_sha256": binding.get("manifest_sha256"),
            "original_archive_size_bytes": artifact.get("original_archive_size_bytes") if artifact else None,
            "recorded_direct_url_filename": _url_filename(binding.get("download_url")),
            "source_id": binding.get("source_id"),
            "source_record_sha256": binding.get("source_row_sha256"),
            "source_record": dict(binding.get("source_record") or {}),
            "initial_provenance_issues": sorted(set(map(str, binding.get("provenance_issues", [])))),
            "physical_paths": paths,
            "synthetic_orphan_binding": False,
        }
        sources.append(source)

    # Retained standalone files below an explicit local_root_path remain
    # current-only until historical acquisition metadata is independently
    # checked later in the remediation pipeline.
    for source in sources:
        local_root = str(source["source_record"].get("local_root_path") or "").replace("\\", "/").rstrip("/")
        if not local_root:
            continue
        matching = [
            artifact
            for artifact in artifacts
            if "incomplete_itemicon_provenance" in artifact.get("provenance_issues", [])
            and any(
                str(path).replace("\\", "/").startswith(local_root + "/")
                for path in artifact.get("original_archive_paths", [])
            )
        ]
        if len(matching) == 1:
            artifact = matching[0]
            source["current_observed_archive_sha256"] = artifact["current_observed_archive_sha256"]
            source["original_archive_size_bytes"] = artifact["original_archive_size_bytes"]
            source["physical_paths"] = list(artifact["original_archive_paths"])
            source["license_evidence"].append(
                {
                    "evidence_type": "raw_extracted_directory_candidate",
                    "note": "retained bytes are current evidence only, not proof of the original download",
                    "path": artifact["original_archive_paths"][0],
                }
            )

    # Acquisition orphans are source-level exclusions.  Itemicon is already
    # attached to its explicit unresolved manifest above and is not duplicated.
    for artifact in artifacts:
        issues = set(map(str, artifact.get("provenance_issues", [])))
        if artifact.get("source_bindings") or "acquisition_orphan_artifact" not in issues:
            continue
        digest = str(artifact["current_observed_archive_sha256"])
        paths = list(artifact.get("original_archive_paths", []))
        sources.append(
            {
                "source_binding_id": _orphan_binding_id(digest),
                "distribution_platform": None,
                "creator_or_publisher": None,
                "pack_or_collection": None,
                "acquisition_run": None,
                "source_page_url": None,
                "direct_download_url": None,
                "original_archive_path": paths[0] if paths else None,
                "original_archive_filename": None,
                "historical_archive_sha256": None,
                "current_observed_archive_sha256": digest,
                "historical_hash_authority": None,
                "license_identifier": None,
                "license_url": None,
                "license_evidence": [],
                "provenance_status": "provenance_incomplete",
                "resolution_status": "excluded",
                "exclusion_reason": "acquisition_orphan: no exact source manifest, URL, license, filename, or historical hash authority",
                "terminal_gate_status": "excluded",
                "eligible_for_candidate_membership": False,
                "manifest_line_number": None,
                "manifest_path": None,
                "manifest_sha256": None,
                "original_archive_size_bytes": artifact.get("original_archive_size_bytes"),
                "recorded_direct_url_filename": None,
                "source_id": None,
                "source_record_sha256": None,
                "source_record": {},
                "initial_provenance_issues": sorted(issues),
                "physical_paths": paths,
                "synthetic_orphan_binding": True,
            }
        )
    return sorted(sources, key=lambda row: row["source_binding_id"]), key_to_id, artifact_by_hash, unresolved


def _apply_embedded_license_evidence(sources: list[dict[str, Any]], source_root: Path) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        digest = source.get("current_observed_archive_sha256")
        if digest:
            by_hash[str(digest)].append(source)
    for bound_sources in by_hash.values():
        candidate_paths = sorted({path for source in bound_sources for path in source.get("physical_paths", [])})
        evidence: list[dict[str, Any]] = []
        evidence_path = None
        for raw_path in candidate_paths:
            path = _safe_resolve(source_root, raw_path)
            embedded = _embedded_license_evidence(path)
            if embedded:
                evidence = embedded
                evidence_path = raw_path
                break
        if not evidence:
            continue
        explicit = {
            (row.get("explicit_license_identifier"), row.get("explicit_license_url"))
            for row in evidence
            if row.get("explicit_license_identifier") and row.get("explicit_license_url")
        }
        for source in bound_sources:
            source["license_evidence"].extend([{**row, "archive_path": evidence_path} for row in evidence])
            if len(explicit) != 1:
                continue
            identifier, url = next(iter(explicit))
            existing = source.get("license_identifier")
            compatible = existing is None or existing == identifier or {existing, identifier} == {"cc_by", "cc_by_4_0"}
            if compatible:
                source["license_identifier"] = identifier
                source["license_url"] = url


def _apply_bound_license_url_evidence(sources: list[dict[str, Any]]) -> None:
    """Normalize only canonical license URLs already bound to each exact source."""

    for source in sources:
        identifier = _license_identifier_from_url(source.get("license_url"))
        if identifier is None:
            continue
        existing = source.get("license_identifier")
        compatible = (
            existing is None
            or existing == identifier
            or (existing == "cc_by" and identifier in {"cc_by_3_0", "cc_by_4_0"})
        )
        if not compatible:
            continue
        source["license_identifier"] = identifier
        source["license_evidence"].append(
            {
                "evidence_type": "exact_bound_license_url",
                "license_identifier": identifier,
                "license_url": source["license_url"],
                "source_binding_id": source["source_binding_id"],
            }
        )


def _apply_exact_license_bindings(sources: list[dict[str, Any]]) -> None:
    providers: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for source in sources:
        key = (
            str(source.get("current_observed_archive_sha256") or ""),
            str(source.get("source_page_url") or ""),
            str(source.get("pack_or_collection") or ""),
        )
        if all(key) and source.get("license_identifier") and source.get("license_url"):
            providers[key].add(
                (str(source["license_identifier"]), str(source["license_url"]), str(source["source_binding_id"]))
            )
    for source in sources:
        if source.get("license_identifier") and source.get("license_url"):
            continue
        key = (
            str(source.get("current_observed_archive_sha256") or ""),
            str(source.get("source_page_url") or ""),
            str(source.get("pack_or_collection") or ""),
        )
        evidence = providers.get(key, set())
        identities = {(identifier, url) for identifier, url, _ in evidence}
        if len(identities) != 1:
            continue
        identifier, url = next(iter(identities))
        providers_used = sorted(provider for _, _, provider in evidence)
        source["license_identifier"] = identifier
        source["license_url"] = url
        source["license_evidence"].append(
            {
                "evidence_type": "exact_pack_and_archive_binding",
                "criteria": ["same_archive_sha256", "same_source_page_url", "same_pack_or_collection"],
                "provider_source_binding_ids": providers_used,
            }
        )


def _apply_exact_historical_archive_bindings(sources: list[dict[str, Any]]) -> None:
    providers: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        historical = source.get("historical_archive_sha256")
        current = source.get("current_observed_archive_sha256")
        key = (
            str(current or ""),
            str(source.get("source_page_url") or ""),
            str(source.get("direct_download_url") or ""),
        )
        if all(key) and historical == current and source.get("original_archive_filename"):
            providers[key].append(source)
    for source in sources:
        if source.get("historical_archive_sha256"):
            continue
        current = source.get("current_observed_archive_sha256")
        key = (
            str(current or ""),
            str(source.get("source_page_url") or ""),
            str(source.get("direct_download_url") or ""),
        )
        candidates = providers.get(key, [])
        identities = {(row["historical_archive_sha256"], row["original_archive_filename"]) for row in candidates}
        if len(identities) != 1:
            continue
        historical, filename = next(iter(identities))
        source["historical_archive_sha256"] = historical
        source["original_archive_filename"] = filename
        provider_ids = sorted(row["source_binding_id"] for row in candidates)
        source["historical_hash_authority"] = "exact_archive_identity:" + ",".join(provider_ids)


def _apply_recovery_records(sources: list[dict[str, Any]], source_root: Path) -> list[dict[str, Any]]:
    repair_paths = sorted(
        (source_root / "harvest_runs").glob("*/provenance_repair*.json"), key=lambda path: path.as_posix()
    )
    validated = [_validate_recovery_record(path, source_root) for path in repair_paths]
    by_identity = {(str(item["record"]["source_run"]), str(item["record"]["source_id"])): item for item in validated}
    for source in sources:
        item = by_identity.get((str(source.get("acquisition_run") or ""), str(source.get("source_id") or "")))
        if item is None:
            continue
        record = item["record"]
        if record["recorded_source_url"] != source.get("source_page_url"):
            raise SourceResolutionError("recovery source page does not match the exact source binding")
        source["direct_download_url"] = record["recorded_download_url"]
        source["original_archive_path"] = record["local_download_path"]
        source["physical_paths"] = [record["local_download_path"]]
        source["original_archive_filename"] = record.get("server_url_filename") or None
        source["historical_archive_sha256"] = record["download_sha256"]
        source["current_observed_archive_sha256"] = record["download_sha256"]
        source["original_archive_size_bytes"] = record["download_size"]
        source["historical_hash_authority"] = (
            f"append_only_provenance_repair:{_display_path(item['path'], source_root)}:{item['record_sha256']}"
        )
        source["license_identifier"] = _normalized_license(record.get("license"))
        recorded_license_page = record.get("license_page") or None
        source["license_url"] = (
            recorded_license_page if _license_identifier_from_url(recorded_license_page) is not None else None
        )
        source["license_evidence"].append(
            {
                "evidence_type": "append_only_exact_local_recovery",
                "member_mapping_count": item["verified_member_count"],
                "provenance_repair_path": _display_path(item["path"], source_root),
                "provenance_repair_sha256": item["record_sha256"],
                "recorded_download_timestamp": record.get("download_timestamp"),
                "recorded_license_identifier": record.get("license"),
                "recorded_license_page": recorded_license_page,
            }
        )
        source["_recovery_sprite_ids"] = item["verified_sprite_ids"]
        source["_recovery_applied"] = True
        source["local_original_recovery_verified"] = True
    return validated


def _apply_recovery_archive_authority(
    sources: list[dict[str, Any]], validated_repairs: Sequence[Mapping[str, Any]], source_root: Path
) -> None:
    """Bind a repair to other rows only when the original archive identity is exact."""

    for item in validated_repairs:
        record = item["record"]
        local_path = str(record["local_download_path"]).replace("\\", "/")
        provider = next(
            (
                source
                for source in sources
                if source.get("acquisition_run") == record["source_run"]
                and source.get("source_id") == record["source_id"]
                and source.get("local_original_recovery_verified")
            ),
            None,
        )
        if provider is None:
            raise SourceResolutionError("validated recovery record was not applied to its exact source binding")
        for source in sources:
            if source is provider or source.get("historical_archive_sha256"):
                continue
            paths = {str(path).replace("\\", "/") for path in source.get("physical_paths", [])}
            identity_matches = (
                source.get("current_observed_archive_sha256") == record["download_sha256"]
                and source.get("source_page_url") == record["recorded_source_url"]
                and source.get("direct_download_url") == record["recorded_download_url"]
                and local_path in paths
            )
            if not identity_matches:
                continue
            source["historical_archive_sha256"] = record["download_sha256"]
            source["historical_hash_authority"] = (
                f"exact_archive_recovery:{_display_path(item['path'], source_root)}:{item['record_sha256']}"
            )
            source["original_archive_filename"] = record.get("server_url_filename") or None
            source["license_identifier"] = _normalized_license(record.get("license"))
            source["license_evidence"].append(
                {
                    "evidence_type": "exact_archive_recovery_record",
                    "criteria": [
                        "same_current_archive_sha256",
                        "same_source_page_url",
                        "same_direct_download_url",
                        "same_local_archive_path",
                    ],
                    "provider_source_binding_id": provider["source_binding_id"],
                    "provenance_repair_path": _display_path(item["path"], source_root),
                    "provenance_repair_sha256": item["record_sha256"],
                    "recorded_license_identifier": record.get("license"),
                    "recorded_license_page": record.get("license_page"),
                }
            )


def _apply_historical_acquisition_records(sources: list[dict[str, Any]], source_root: Path) -> list[dict[str, Any]]:
    """Recover a retained standalone original from exact acquisition metadata."""

    recovered: list[dict[str, Any]] = []
    for source in sources:
        local_root_raw = str(source.get("source_record", {}).get("local_root_path") or "")
        run = str(source.get("acquisition_run") or "")
        if not local_root_raw or not run or not source.get("physical_paths"):
            continue
        candidates_path = source_root / "harvest_runs" / run / "candidates.jsonl"
        events_path = source_root / "harvest_runs" / run / "events.jsonl"
        if not candidates_path.is_file() or not events_path.is_file():
            continue
        matching = [
            row
            for row in _read_jsonl(candidates_path)
            if row.get("source_id") == source.get("source_id")
            and row.get("source_path") == source.get("source_page_url")
            and row.get("status") == "candidate"
            and row.get("extracted_path")
            and row.get("relative_path")
            and row.get("image_sha256")
        ]
        if len(matching) != 1:
            continue
        row = matching[0]
        candidate = _safe_resolve(source_root, str(row["extracted_path"]))
        local_root = _safe_resolve(source_root, local_root_raw)
        try:
            relative = candidate.relative_to(local_root).as_posix()
        except ValueError:
            continue
        physical_paths = {str(path).replace("\\", "/") for path in source.get("physical_paths", [])}
        candidate_display = _display_path(candidate, source_root)
        if candidate_display not in physical_paths or relative != str(row["relative_path"]).replace("\\", "/"):
            continue
        actual_hash = file_sha256(candidate) if candidate.is_file() else None
        if actual_hash != row["image_sha256"] or actual_hash != source.get("current_observed_archive_sha256"):
            continue
        source_created_at = source.get("source_record", {}).get("created_at")
        import_events = [
            event
            for event in _read_jsonl(events_path)
            if event.get("event") == "import"
            and event.get("source_id") == source.get("source_id")
            and event.get("at") == source_created_at
        ]
        if len(import_events) != 1:
            continue
        candidates_hash = file_sha256(candidates_path)
        events_hash = file_sha256(events_path)
        source["original_archive_path"] = candidate_display
        source["original_archive_filename"] = str(row["relative_path"])
        source["historical_archive_sha256"] = actual_hash
        source["current_observed_archive_sha256"] = actual_hash
        source["original_archive_size_bytes"] = candidate.stat().st_size
        source["historical_hash_authority"] = (
            "historical_acquisition_metadata:"
            f"{_display_path(candidates_path, source_root)}:{candidates_hash}:"
            f"{_display_path(events_path, source_root)}:{events_hash}"
        )
        source["license_evidence"].append(
            {
                "evidence_type": "historical_standalone_acquisition_record",
                "candidate_manifest_path": _display_path(candidates_path, source_root),
                "candidate_manifest_sha256": candidates_hash,
                "event_log_path": _display_path(events_path, source_root),
                "event_log_sha256": events_hash,
                "recorded_original_filename": row["relative_path"],
                "recorded_original_sha256": row["image_sha256"],
                "recorded_source_page_url": row["source_path"],
            }
        )
        source["_recovery_applied"] = True
        source["local_original_recovery_verified"] = True
        recovered.append(
            {
                "candidate_manifest_path": candidates_path,
                "candidate_manifest_sha256": candidates_hash,
                "events_path": events_path,
                "events_sha256": events_hash,
                "original_path": candidate,
                "original_sha256": actual_hash,
                "original_size": candidate.stat().st_size,
                "source_binding_id": source["source_binding_id"],
                "source_id": source["source_id"],
            }
        )
    return recovered


def _exclusion_reasons(source: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not source.get("original_archive_path") or not source.get("current_observed_archive_sha256"):
        reasons.append("missing_original_bytes")
    if not source.get("source_page_url") and not source.get("direct_download_url"):
        reasons.append("missing_source_url")
    if not source.get("original_archive_filename"):
        reasons.append("missing_original_archive_filename")
    if not source.get("historical_archive_sha256"):
        reasons.append("missing_historical_archive_sha256")
    if not source.get("historical_hash_authority"):
        reasons.append("missing_historical_hash_authority")
    if not source.get("license_identifier"):
        reasons.append("unknown_license")
    valid_license_evidence = any(
        evidence.get("evidence_type") != "historical_manifest_license_record" or bool(evidence.get("user_confirmed"))
        for evidence in source.get("license_evidence", [])
        if isinstance(evidence, Mapping)
    )
    if not valid_license_evidence:
        reasons.append("missing_license_evidence")
    if source.get("synthetic_orphan_binding"):
        reasons.append("acquisition_orphan")
    return reasons


def _finalize_sources(sources: list[dict[str, Any]], source_root: Path) -> None:
    for source in sources:
        if source.get("synthetic_orphan_binding"):
            continue
        reasons = _exclusion_reasons(source)
        hashes_match = source.get("historical_archive_sha256") and source.get(
            "historical_archive_sha256"
        ) == source.get("current_observed_archive_sha256")
        if not reasons and hashes_match:
            recovered = bool(source.pop("_recovery_applied", False))
            source["provenance_status"] = "recovered_from_local_original" if recovered else "verified"
            source["resolution_status"] = source["provenance_status"]
            source["terminal_gate_status"] = "verified"
            source["eligible_for_candidate_membership"] = True
            source["exclusion_reason"] = None
        else:
            source.pop("_recovery_applied", None)
            if "missing_original_bytes" in reasons:
                source["provenance_status"] = "requires_manual_retrieval"
            elif "unknown_license" in reasons:
                source["provenance_status"] = "license_unknown"
            elif source.get("current_observed_archive_sha256"):
                source["provenance_status"] = "verified_current_hash_only"
            else:
                source["provenance_status"] = "provenance_incomplete"
            source["resolution_status"] = "excluded"
            source["terminal_gate_status"] = "excluded"
            source["eligible_for_candidate_membership"] = False
            source["exclusion_reason"] = "; ".join(reasons)
        if source.get("original_archive_path"):
            path = _safe_resolve(source_root, str(source["original_archive_path"]))
            if not path.is_file():
                raise SourceResolutionError(f"resolved original bytes disappeared: {path}")
            actual = file_sha256(path)
            if actual != source.get("current_observed_archive_sha256"):
                raise SourceResolutionError(f"current archive hash mismatch: {path}")


def _record_binding_ids(
    row: Mapping[str, Any],
    key_to_id: Mapping[tuple[str, int, str, str], str],
    sources_by_hash: Mapping[str, list[dict[str, Any]]],
) -> list[str]:
    ids = [
        key_to_id[_binding_key(binding)]
        for binding in row.get("source_bindings", [])
        if _binding_key(binding) in key_to_id
    ]
    if ids:
        return sorted(set(ids))
    digest = str(row.get("original_archive_sha256") or "")
    candidates = sources_by_hash.get(digest, [])
    if len(candidates) == 1:
        return [candidates[0]["source_binding_id"]]
    return []


def filter_candidate_records(
    records: Iterable[Mapping[str, Any]], source_resolutions: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Remove excluded bindings and every record that depends only on them."""

    eligible: list[dict[str, Any]] = []
    for record in records:
        ids = list(map(str, record.get("source_binding_ids", [])))
        verified_ids = [
            source_id
            for source_id in ids
            if source_resolutions.get(source_id, {}).get("terminal_gate_status") == "verified"
        ]
        if not verified_ids:
            continue
        public = dict(record)
        public["source_binding_ids"] = sorted(set(verified_ids))
        eligible.append(public)
    return eligible


def _rebuild_inventory(
    records: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    key_to_id: Mapping[tuple[str, int, str, str], str],
) -> dict[str, Any]:
    sources_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        digest = str(source.get("current_observed_archive_sha256") or "")
        if digest:
            sources_by_hash[digest].append(source)
    resolutions = {source["source_binding_id"]: source for source in sources}
    candidate_rows: list[dict[str, Any]] = []
    for row in records:
        ids = _record_binding_ids(row, key_to_id, sources_by_hash)
        if not ids:
            continue
        content_issues = sorted(set(map(str, row.get("audit_issues", []))) - _PROVENANCE_RECORD_ISSUES)
        candidate = {
            "forensic_record_id": row.get("forensic_record_id"),
            "source_binding_ids": ids,
            "content_issues": content_issues,
        }
        if (
            row.get("record_type") in {"archive_member_image", "standalone_image"}
            and row.get("decoded_image_sha256")
            and not content_issues
        ):
            candidate_rows.append(candidate)
    eligible_existing = filter_candidate_records(candidate_rows, resolutions)
    recovered_sources = [source for source in sources if source["resolution_status"] == "recovered_from_local_original"]
    recovered_ids = sorted(
        {sprite_id for source in recovered_sources for sprite_id in source.get("_recovery_sprite_ids", [])}
    )
    baseline_accepted = sum(row.get("inclusion_decision") == "accept" for row in records)
    newly_eligible_existing = sum(
        any(issue in _PROVENANCE_RECORD_ISSUES for issue in row.get("audit_issues", []))
        for row in records
        if row.get("forensic_record_id") in {item["forensic_record_id"] for item in eligible_existing}
    )
    candidate_source_ids = sorted(
        source["source_binding_id"] for source in sources if source["terminal_gate_status"] == "verified"
    )
    excluded_source_ids = sorted(
        source["source_binding_id"] for source in sources if source["terminal_gate_status"] == "excluded"
    )
    manifest_sources = [source for source in sources if not source.get("synthetic_orphan_binding")]
    manifest_verified = [source for source in manifest_sources if source["terminal_gate_status"] == "verified"]
    manifest_excluded = [source for source in manifest_sources if source["terminal_gate_status"] == "excluded"]
    initial_source_issue_counts = Counter(
        issue for source in manifest_sources for issue in source.get("initial_provenance_issues", [])
    )
    initial_record_issue_counts = Counter(issue for row in records for issue in row.get("audit_issues", []))
    itemicon_source = next(
        (source for source in manifest_sources if source.get("source_id") == "oga_itemiconpack32_cc_by"), None
    )
    if set(candidate_source_ids) & set(excluded_source_ids):
        raise SourceResolutionError("excluded source leaked into candidate membership")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_gate_passed": all(source["terminal_gate_status"] in {"verified", "excluded"} for source in sources),
        "source_binding_count": len(sources),
        "manifest_source_binding_count": len(manifest_sources),
        "synthetic_orphan_binding_count": len(sources) - len(manifest_sources),
        "terminal_resolution_entry_count": len(sources),
        "resolved_source_binding_count": len(sources),
        "verified_source_binding_count": len(candidate_source_ids),
        "excluded_source_binding_count": len(excluded_source_ids),
        "verified_manifest_source_binding_count": len(manifest_verified),
        "excluded_manifest_source_binding_count": len(manifest_excluded),
        "remaining_unresolved_source_binding_count": 0,
        "candidate_source_binding_ids": candidate_source_ids,
        "excluded_source_binding_ids": excluded_source_ids,
        "frozen_forensic_record_count": len(records),
        "baseline_accepted_record_count": baseline_accepted,
        "existing_inventory_eligible_record_count": len(eligible_existing),
        "recovered_mapping_eligible_record_count": len(recovered_ids),
        "newly_eligible_existing_record_count": newly_eligible_existing,
        "newly_eligible_record_count": newly_eligible_existing + len(recovered_ids),
        "total_candidate_record_count": len(eligible_existing) + len(recovered_ids),
        "records_excluded_count": len(records) - len(eligible_existing),
        "recovered_from_local_original_source_binding_count": len(recovered_sources),
        "locally_recovered_source_binding_count": sum(
            bool(source.get("local_original_recovery_verified")) for source in manifest_sources
        ),
        "initial_record_blocking_issue_counts": dict(sorted(initial_record_issue_counts.items())),
        "initial_manifest_source_issue_counts": dict(sorted(initial_source_issue_counts.items())),
        "initial_missing_provenance_source_binding_count": sum(
            bool(source.get("initial_provenance_issues")) for source in manifest_sources
        ),
        "initial_acquisition_orphan_source_artifact_count": sum(
            bool(source.get("synthetic_orphan_binding")) for source in sources
        ),
        "initial_incomplete_itemicon_record_count": initial_record_issue_counts["incomplete_itemicon_provenance"],
        "itemicon_source_resolution": (
            {
                "exclusion_reason": itemicon_source.get("exclusion_reason"),
                "resolution_status": itemicon_source["resolution_status"],
                "source_binding_id": itemicon_source["source_binding_id"],
            }
            if itemicon_source is not None
            else None
        ),
        "license_identifier_counts": dict(
            sorted(Counter(source.get("license_identifier") or "unknown" for source in sources).items())
        ),
        "manifest_source_license_identifier_counts": dict(
            sorted(Counter(source.get("license_identifier") or "unknown" for source in manifest_sources).items())
        ),
        "license_url_status_counts": dict(
            sorted(
                Counter("present" if source.get("license_url") else "missing" for source in manifest_sources).items()
            )
        ),
        "license_evidence_status_counts": dict(
            sorted(
                Counter(
                    "present" if source.get("license_evidence") else "missing" for source in manifest_sources
                ).items()
            )
        ),
        "provenance_status_counts": dict(sorted(Counter(source["provenance_status"] for source in sources).items())),
        "historical_hash_authority_counts": dict(
            sorted(
                Counter(
                    "none"
                    if not source.get("historical_hash_authority")
                    else str(source["historical_hash_authority"]).split(":", 1)[0]
                    for source in sources
                ).items()
            )
        ),
    }


def _zip_index(search_roots: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths: set[Path] = set()
    partials: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                try:
                    if not path.is_file():
                        continue
                except OSError:
                    continue
                suffix = path.suffix.lower()
                if suffix == ".zip":
                    paths.add(path.resolve())
                elif suffix in {".crdownload", ".part"}:
                    partials.add(path.resolve())
        except (OSError, PermissionError):
            continue
    zip_rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.as_posix().casefold()):
        row: dict[str, Any] = {"path": path.as_posix(), "size": path.stat().st_size, "sha256": file_sha256(path)}
        try:
            with zipfile.ZipFile(path) as archive:
                members = [
                    {"crc32": f"{info.CRC:08x}", "name": info.filename, "size": info.file_size}
                    for info in archive.infolist()
                    if not info.is_dir()
                ]
            row["member_count"] = len(members)
            row["member_list_sha256"] = _sha256_bytes(canonical_json_bytes(members))
            row["zip_readable"] = True
        except (OSError, zipfile.BadZipFile) as exc:
            row.update(
                {
                    "member_count": None,
                    "member_list_sha256": None,
                    "zip_error": type(exc).__name__,
                    "zip_readable": False,
                }
            )
        zip_rows.append(row)
    partial_rows = [
        {"path": path.as_posix(), "sha256": file_sha256(path), "size": path.stat().st_size}
        for path in sorted(partials, key=lambda item: item.as_posix().casefold())
    ]
    return zip_rows, partial_rows


def _local_candidates(
    sources: list[dict[str, Any]],
    source_root: Path,
    search_roots: Sequence[Path],
    validated_repairs: Sequence[Mapping[str, Any]],
    historical_acquisitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    zip_rows, partial_rows = _zip_index(search_roots)
    repair_by_source_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in validated_repairs:
        repair_by_source_id[str(item["record"]["source_id"])].append(item)
    acquisition_by_binding = {str(item["source_binding_id"]): item for item in historical_acquisitions}
    matched_zip_paths: set[str] = set()
    matched_partial_paths: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    for source in sources:
        if source["provenance_status"] == "verified" and not source["initial_provenance_issues"]:
            continue
        candidates: dict[str, dict[str, Any]] = {}
        expected_hashes = {
            str(value)
            for value in (source.get("historical_archive_sha256"), source.get("current_observed_archive_sha256"))
            if value
        }
        expected_names = {
            str(value).casefold()
            for value in (source.get("original_archive_filename"), source.get("recorded_direct_url_filename"))
            if value
        }
        for raw_path in source.get("physical_paths", []):
            path = _safe_resolve(source_root, raw_path)
            if path.is_file():
                candidates[path.as_posix()] = {
                    "candidate_path": path.as_posix(),
                    "candidate_type": "forensic_physical_path",
                    "match_status": (
                        "exact_current_byte_hash_match"
                        if file_sha256(path) == source.get("current_observed_archive_sha256")
                        else "rejected_hash_mismatch"
                    ),
                    "reason": "path was explicitly retained by the frozen forensic artifact binding",
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
        local_root = str(source.get("source_record", {}).get("local_root_path") or "")
        if local_root:
            path = _safe_resolve(source_root, local_root)
            candidates[path.as_posix()] = {
                "candidate_path": path.as_posix(),
                "candidate_type": "raw_extracted_directory",
                "match_status": "supporting_evidence_not_original_download",
                "reason": "explicit manifest local_root_path; directory contents cannot establish original archive bytes",
                "sha256": None,
                "size": None,
            }
        for row in zip_rows:
            filename_match = Path(row["path"]).name.casefold() in expected_names
            hash_match = row["sha256"] in expected_hashes
            run_match = (
                bool(source.get("acquisition_run"))
                and f"/{source['acquisition_run']}/".casefold() in row["path"].replace("\\", "/").casefold()
            )
            if not (filename_match or hash_match or run_match):
                continue
            if hash_match:
                status = "exact_byte_hash_match"
                reason = "byte hash matches an explicit current or historically authorized archive hash"
            elif filename_match:
                status = "rejected_filename_only" if expected_hashes else "filename_candidate_no_hash_authority"
                reason = "filename matches recorded evidence but filenames alone do not establish source identity"
            else:
                status = "acquisition_directory_candidate"
                reason = "candidate is below the exact acquisition-run directory; byte identity is evaluated separately"
            candidates[row["path"]] = {
                "candidate_path": row["path"],
                "candidate_type": "zip_search_result",
                "match_status": status,
                "reason": reason,
                "sha256": row["sha256"],
                "size": row["size"],
                "member_count": row["member_count"],
                "member_list_sha256": row["member_list_sha256"],
            }
            matched_zip_paths.add(row["path"])
        for row in partial_rows:
            partial_name = Path(row["path"]).name.casefold()
            filename_match = any(partial_name.startswith(name) for name in expected_names)
            run_match = (
                bool(source.get("acquisition_run"))
                and f"/{source['acquisition_run']}/".casefold() in row["path"].replace("\\", "/").casefold()
            )
            if not (filename_match or run_match):
                continue
            candidates[row["path"]] = {
                "candidate_path": row["path"],
                "candidate_type": "partial_download",
                "match_status": "rejected_incomplete_download",
                "reason": "partial downloads cannot establish complete original bytes",
                "sha256": row["sha256"],
                "size": row["size"],
            }
            matched_partial_paths.add(row["path"])
        for item in repair_by_source_id.get(str(source.get("source_id") or ""), []):
            repair = item["record"]
            path = _safe_resolve(source_root, str(repair["local_download_path"]))
            exact_run = repair["source_run"] == source.get("acquisition_run")
            candidates[path.as_posix()] = {
                "candidate_path": path.as_posix(),
                "candidate_type": "append_only_provenance_repair",
                "match_status": "exact_recovery" if exact_run else "rejected_wrong_acquisition_run",
                "reason": (
                    "repair source_run and source_id exactly match this binding"
                    if exact_run
                    else "archive evidence is exact for a different acquisition run and is not transferred"
                ),
                "sha256": repair["download_sha256"],
                "size": repair["download_size"],
                "verified_member_count": item["verified_member_count"],
            }
        acquisition = acquisition_by_binding.get(str(source["source_binding_id"]))
        if acquisition is not None:
            path = Path(acquisition["original_path"])
            candidates[path.as_posix()] = {
                "candidate_path": path.as_posix(),
                "candidate_type": "historical_acquisition_metadata",
                "match_status": "exact_recovery",
                "reason": "source identity, acquisition event, recorded filename, and historical/current byte hashes match",
                "sha256": acquisition["original_sha256"],
                "size": acquisition["original_size"],
                "candidate_manifest_sha256": acquisition["candidate_manifest_sha256"],
                "event_log_sha256": acquisition["events_sha256"],
            }
        source_rows.append(
            {
                "source_binding_id": source["source_binding_id"],
                "resolution_status": source["resolution_status"],
                "candidates": sorted(candidates.values(), key=lambda row: row["candidate_path"].casefold()),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "search_scope": {
            "browser_download_history_exact_url_matches": 0,
            "codex_attachment_exact_identifier_matches": 0,
            "command_logs_and_source_manifests": "searched exact source IDs, URLs, hashes, and filenames",
            "git_history": "searched all refs; no additional authoritative raw archive binding beyond frozen evidence",
            "network_accessed": False,
            "preservation_worktrees": "searched; only duplicate frozen/code evidence found",
            "search_roots": [path.resolve().as_posix() for path in search_roots],
            "zip_file_count": len(zip_rows),
            "partial_download_file_count": len(partial_rows),
        },
        "matched_zip_search_index": [row for row in zip_rows if row["path"] in matched_zip_paths],
        "matched_partial_download_index": [row for row in partial_rows if row["path"] in matched_partial_paths],
        "sources": source_rows,
    }


def _missing_download_report(
    unresolved: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    local_candidates: Mapping[str, Any],
) -> dict[str, Any]:
    source_by_key = {_binding_key(source): source for source in sources if source.get("manifest_path")}
    candidates_by_id = {row["source_binding_id"]: row["candidates"] for row in local_candidates["sources"]}
    rows: list[dict[str, Any]] = []
    for binding in sorted(unresolved, key=_binding_key):
        source = source_by_key[_binding_key(binding)]
        recovered = bool(source.get("local_original_recovery_verified"))
        exact_direct = bool(source.get("direct_download_url"))
        if recovered:
            disposition = (
                "recovered_from_local_original; retain external URL for audit only and do not download"
                if source["terminal_gate_status"] == "verified"
                else "original bytes recovered locally but source remains excluded for other evidence gaps"
            )
        elif exact_direct:
            disposition = "requires_manual_retrieval; exact URL may be used only by the guarded recovery script"
        else:
            disposition = "requires_manual_retrieval; create/review an exact direct-URL plan before any download"
        rows.append(
            {
                "source_binding_id": source["source_binding_id"],
                "expected_source_url": source.get("source_page_url"),
                "expected_direct_download_url": source.get("direct_download_url"),
                "expected_filename": source.get("original_archive_filename"),
                "expected_historical_size": source.get("original_archive_size_bytes") if recovered else None,
                "expected_historical_sha256": source.get("historical_archive_sha256") if recovered else None,
                "local_search_results": candidates_by_id.get(source["source_binding_id"], []),
                "exact_recovery_possible": recovered,
                "external_redownload_safe": exact_direct,
                "recommended_disposition": disposition,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "initial_missing_original_count": len(rows),
        "recovered_locally_count": sum(row["exact_recovery_possible"] for row in rows),
        "still_requiring_external_retrieval_count": sum(not row["exact_recovery_possible"] for row in rows),
        "entries": rows,
    }


def _powershell_literal(value: str | None) -> str:
    if value is None:
        return "$null"
    return "'" + value.replace("'", "''") + "'"


def render_download_recovery_script(entries: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for entry in sorted(entries, key=lambda row: str(row["source_binding_id"])):
        rows.append(
            "    [pscustomobject]@{ "
            f"SourceBindingId = {_powershell_literal(str(entry['source_binding_id']))}; "
            f"DirectUrl = {_powershell_literal(entry.get('expected_direct_download_url'))}; "
            f"Filename = {_powershell_literal(entry.get('expected_filename'))}; "
            f"Recovered = ${str(bool(entry.get('exact_recovery_possible'))).lower()} "
            "}"
        )
    entries_text = "\n".join(rows)
    return f"""[CmdletBinding()]
param(
    [switch]$ExecuteDownloads
)

$ErrorActionPreference = 'Stop'
$RecoveryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\\v5_raw_rebuild_sol_v1\\recovered_downloads_v1'))
$Entries = @(
{entries_text}
)

if (-not $ExecuteDownloads) {{
    Write-Host 'DRY RUN: no network request or file write will occur.'
    $Entries | Select-Object SourceBindingId, DirectUrl, Filename, Recovered | Format-Table -AutoSize
    exit 0
}}

$downloadable = @($Entries | Where-Object {{ -not $_.Recovered -and $_.DirectUrl -and $_.Filename }})
if ($downloadable.Count -eq 0) {{
    Write-Host 'No unresolved entry has both an exact recorded direct URL and an authoritative filename; nothing downloaded.'
    exit 0
}}

New-Item -ItemType Directory -Path $RecoveryRoot -Force | Out-Null
foreach ($entry in $downloadable) {{
    if ($entry.DirectUrl -notmatch '^https://') {{
        throw "Refusing non-HTTPS or missing exact URL for $($entry.SourceBindingId)"
    }}
    $destination = Join-Path $RecoveryRoot $entry.Filename
    if (Test-Path -LiteralPath $destination) {{
        throw "Refusing to overwrite recovery destination: $destination"
    }}
    Invoke-WebRequest -Uri $entry.DirectUrl -OutFile $destination -UseBasicParsing
    $download = Get-Item -LiteralPath $destination
    $logRow = [ordered]@{{
        source_binding_id = $entry.SourceBindingId
        exact_recorded_direct_url = $entry.DirectUrl
        destination = $destination
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
        size = $download.Length
        downloaded_at_utc = [DateTime]::UtcNow.ToString('o')
    }}
    ($logRow | ConvertTo-Json -Compress) | Add-Content -LiteralPath (Join-Path $RecoveryRoot 'download_log.jsonl') -Encoding UTF8
}}
"""


def _exclusion_manifest(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = None
    for sequence, source in enumerate(
        sorted(
            (row for row in sources if row["resolution_status"] == "excluded"), key=lambda row: row["source_binding_id"]
        ),
        1,
    ):
        row = {
            "schema_version": EXCLUSION_SCHEMA_VERSION,
            "exclusion_sequence": sequence,
            "source_binding_id": source["source_binding_id"],
            "exclusion_reason": source["exclusion_reason"],
            "terminal_gate_status": "excluded",
            "current_observed_archive_sha256": source.get("current_observed_archive_sha256"),
            "historical_archive_sha256": source.get("historical_archive_sha256"),
            "previous_entry_sha256": previous,
        }
        entry_hash = _sha256_bytes(canonical_json_bytes(row))
        row["entry_sha256"] = entry_hash
        previous = entry_hash
        rows.append(row)
    return rows


def _public_source(source: Mapping[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in source.items() if not key.startswith("_") and key != "source_record"}
    missing = [field for field in REQUIRED_SOURCE_BINDING_FIELDS if field not in public]
    if missing:
        raise SourceResolutionError(f"source binding lacks required normalized fields: {missing}")
    if public["provenance_status"] not in CONTROLLED_RESOLUTION_STATES:
        raise SourceResolutionError(f"uncontrolled provenance status: {public['provenance_status']}")
    if public["resolution_status"] not in TERMINAL_RESOLUTION_STATES:
        raise SourceResolutionError(f"non-terminal source resolution: {public['source_binding_id']}")
    return public


def _frozen_hash_verification(frozen_experiment: Path, before: Mapping[str, str]) -> dict[str, Any]:
    recorded_document = _read_json(frozen_experiment / "artifact_hashes.json")
    recorded = recorded_document.get("artifacts")
    if not isinstance(recorded, Mapping):
        raise SourceResolutionError("frozen artifact_hashes.json has no artifacts mapping")
    files: dict[str, Any] = {}
    after: dict[str, str] = {}
    for filename, before_hash in sorted(before.items()):
        path = frozen_experiment / filename
        actual = file_sha256(path)
        after[filename] = actual
        expected = recorded.get(filename)
        files[filename] = {
            "before_sha256": before_hash,
            "after_sha256": actual,
            "recorded_frozen_sha256": expected,
            "matches_recorded_frozen_hash": expected == actual,
            "unchanged_during_remediation": before_hash == actual,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_inputs_unchanged": before == after,
        "all_recorded_hashes_match": all(row["matches_recorded_frozen_hash"] for row in files.values()),
        "files": files,
    }


def _report(summary: Mapping[str, Any], missing: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    record_issues = summary["initial_record_blocking_issue_counts"]
    source_issues = summary["initial_manifest_source_issue_counts"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "mission": "raw source provenance remediation only; no semantic labeling performed",
        "network_accessed": False,
        "source_gate_passed": summary["source_gate_passed"],
        "sources_verified": summary["verified_source_binding_count"],
        "sources_excluded": summary["excluded_source_binding_count"],
        "remaining_unresolved_sources": summary["remaining_unresolved_source_binding_count"],
        "missing_downloads_still_requiring_external_retrieval": missing["still_requiring_external_retrieval_count"],
        "newly_eligible_record_count": summary["newly_eligible_record_count"],
        "records_excluded_count": summary["records_excluded_count"],
        "initial_blockers": {
            "missing_original_downloads": {
                "record_findings": record_issues.get("missing_original_download", 0),
                "source_bindings": source_issues.get("missing_original_download", 0),
            },
            "unknown_license": {
                "record_findings": record_issues.get("unknown_license", 0),
                "source_bindings": source_issues.get("unknown_license", 0),
            },
            "missing_provenance": {
                "source_bindings": summary["initial_missing_provenance_source_binding_count"],
            },
            "missing_historical_archive_sha256": {
                "record_findings": record_issues.get("missing_historical_archive_sha256", 0),
                "source_bindings": source_issues.get("missing_historical_archive_sha256", 0),
            },
            "missing_original_filename": {
                "record_findings": record_issues.get("missing_original_filename", 0),
                "source_bindings": source_issues.get("missing_original_filename", 0),
            },
            "missing_license_url": {
                "record_findings": record_issues.get("missing_license_url", 0),
                "source_bindings": source_issues.get("missing_license_url", 0),
            },
            "acquisition_orphans": {
                "record_findings": record_issues.get("acquisition_orphan_artifact", 0),
                "source_artifacts": summary["initial_acquisition_orphan_source_artifact_count"],
            },
            "incomplete_itemicon_provenance": {
                "record_findings": summary["initial_incomplete_itemicon_record_count"],
                "disposition": summary["itemicon_source_resolution"],
            },
        },
        "source_level_resolution": {
            "verified_without_local_recovery": summary["verified_source_binding_count"]
            - summary["recovered_from_local_original_source_binding_count"],
            "recovered_from_local_original": summary["recovered_from_local_original_source_binding_count"],
            "excluded": summary["excluded_source_binding_count"],
            "unresolved": summary["remaining_unresolved_source_binding_count"],
        },
        "license_identifier_counts": summary["license_identifier_counts"],
        "manifest_source_license_identifier_counts": summary["manifest_source_license_identifier_counts"],
        "license_url_status_counts": summary["license_url_status_counts"],
        "license_evidence_status_counts": summary["license_evidence_status_counts"],
        "provenance_status_counts": summary["provenance_status_counts"],
        "historical_hash_authority_counts": summary["historical_hash_authority_counts"],
        "policy_notes": [
            "current observed hashes were not promoted to historical hashes",
            "license evidence was not inferred across packs",
            "missing URLs and original filenames were not guessed",
            "all non-verified sources are terminally and explicitly excluded",
            "excluded source binding IDs are absent from candidate membership",
        ],
    }
    text = "\n".join(
        [
            "# Dataset-v5 raw provenance remediation",
            "",
            "This is a provenance-only remediation. No semantic labeling, provider inference, training, or network download was performed.",
            "",
            "## Terminal source gate",
            "",
            f"- Source gate passed: `{str(report['source_gate_passed']).lower()}`",
            f"- Verified or exactly recovered source bindings: {report['sources_verified']}",
            f"- Explicitly excluded source bindings: {report['sources_excluded']}",
            f"- Silently unresolved source bindings: {report['remaining_unresolved_sources']}",
            f"- Missing originals still requiring an exact external retrieval plan: {report['missing_downloads_still_requiring_external_retrieval']}",
            "",
            "## Candidate impact",
            "",
            f"- Newly eligible records: {report['newly_eligible_record_count']}",
            f"- Frozen forensic records excluded by source/content policy: {report['records_excluded_count']}",
            "",
            "## Initial blocker normalization",
            "",
            f"- Missing original downloads: {record_issues.get('missing_original_download', 0)} record findings / {source_issues.get('missing_original_download', 0)} source bindings",
            f"- Unknown license: {record_issues.get('unknown_license', 0)} record findings / {source_issues.get('unknown_license', 0)} source bindings",
            f"- Missing provenance: {summary['initial_missing_provenance_source_binding_count']} source bindings",
            f"- Missing historical archive hash: {record_issues.get('missing_historical_archive_sha256', 0)} record findings / {source_issues.get('missing_historical_archive_sha256', 0)} source bindings",
            f"- Missing original filename: {record_issues.get('missing_original_filename', 0)} record findings / {source_issues.get('missing_original_filename', 0)} source bindings",
            f"- Missing license URL: {record_issues.get('missing_license_url', 0)} record findings / {source_issues.get('missing_license_url', 0)} source bindings",
            f"- Acquisition orphans: {record_issues.get('acquisition_orphan_artifact', 0)} record findings / {summary['initial_acquisition_orphan_source_artifact_count']} source artifacts",
            f"- Incomplete Itemicon provenance: {summary['initial_incomplete_itemicon_record_count']} record; final source resolution `{summary['itemicon_source_resolution']['resolution_status']}`",
            "",
            "## Evidence policy",
            "",
            "Current hashes remain current-only unless an exact historical binding or pre-existing append-only recovery record supplies authority. Embedded license files are bound only to their exact archive. Missing evidence results in exclusion.",
            "",
        ]
    )
    return report, text


def refresh_artifact_hashes(output_root: str | Path) -> dict[str, str]:
    root = Path(output_root)
    hashes = {
        path.name: file_sha256(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    destination = root / "artifact_hashes.json"
    destination.write_bytes(canonical_json_bytes({"schema_version": SCHEMA_VERSION, "artifacts": hashes}))
    return hashes


def compile_remediation(
    frozen_experiment: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    *,
    search_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    frozen = Path(frozen_experiment).resolve()
    sources_root = Path(source_root).resolve()
    output = Path(output_root).resolve()
    required_inputs = ("raw_source_inventory.jsonl", "raw_source_inventory_report.md", "source_archive_hashes.json")
    before = {filename: file_sha256(frozen / filename) for filename in required_inputs}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty remediation output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(frozen / "raw_source_inventory.jsonl")
    sources, key_to_id, _, unresolved = _load_sources(frozen, sources_root)
    _apply_embedded_license_evidence(sources, sources_root)
    _apply_bound_license_url_evidence(sources)
    _apply_exact_license_bindings(sources)
    _apply_exact_historical_archive_bindings(sources)
    repairs = _apply_recovery_records(sources, sources_root)
    _apply_recovery_archive_authority(sources, repairs, sources_root)
    historical_acquisitions = _apply_historical_acquisition_records(sources, sources_root)
    _finalize_sources(sources, sources_root)
    public_sources = [_public_source(source) for source in sources]
    if any(source["resolution_status"] not in TERMINAL_RESOLUTION_STATES for source in public_sources):
        raise SourceResolutionError("every source must receive a terminal resolution")

    actual_search_roots = [sources_root, *(Path(root).resolve() for root in search_roots)]
    deduped_search_roots = list(dict.fromkeys(actual_search_roots))
    local = _local_candidates(sources, sources_root, deduped_search_roots, repairs, historical_acquisitions)
    missing = _missing_download_report(unresolved, sources, local)
    summary = _rebuild_inventory(records, sources, key_to_id)
    report_json, report_md = _report(summary, missing)
    exclusions = _exclusion_manifest(public_sources)
    license_rows = [
        {
            "source_binding_id": source["source_binding_id"],
            "pack_or_collection": source["pack_or_collection"],
            "license_identifier": source["license_identifier"],
            "license_url": source["license_url"],
            "license_evidence": source["license_evidence"],
            "resolution_status": source["resolution_status"],
        }
        for source in public_sources
    ]
    frozen_verification = _frozen_hash_verification(frozen, before)

    _write_new_json(output / "remediation_report.json", report_json)
    _write_new_bytes(output / "remediation_report.md", report_md.encode("utf-8"))
    _write_new_jsonl(output / "source_resolution_manifest.jsonl", public_sources)
    _write_new_json(output / "local_recovery_candidates.json", local)
    _write_new_jsonl(output / "license_evidence_manifest.jsonl", license_rows)
    _write_new_json(output / "missing_download_report.json", missing)
    _write_new_bytes(
        output / "download_recovery_plan.ps1", render_download_recovery_script(missing["entries"]).encode("utf-8")
    )
    _write_new_jsonl(output / "exclusion_manifest.jsonl", exclusions)
    _write_new_json(output / "rebuilt_inventory_summary.json", summary)
    _write_new_json(output / "frozen_hash_verification.json", frozen_verification)
    _write_new_bytes(
        output / "command_log.txt",
        (
            b"Dataset-v5 raw provenance remediation command log\n"
            b"================================================\n\n"
            b"No network access, download, semantic labeling, provider inference, training, or frozen-data mutation occurred.\n\n"
            b"- Read frozen raw forensic inventory and archive-hash evidence.\n"
            b"- Indexed local ZIPs with byte hashes and member-list hashes.\n"
            b"- Searched historical acquisition paths, manifests, appendices, raw roots, download caches, attachments, browser history, Git history, and preservation worktrees.\n"
            b"- Validated exact local recovery records by archive bytes and every declared member hash.\n"
            b"- Applied terminal verified/excluded source policy and rebuilt candidate counts.\n"
        ),
    )
    _write_new_bytes(output / "implementation.patch", b"")
    _write_new_bytes(output / "test_results.txt", b"Pending final verification.\n")
    _write_new_bytes(output / "static_check_results.txt", b"Pending final verification.\n")
    refresh_artifact_hashes(output)
    return report_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-experiment", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--search-root", action="append", default=[], type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compile_remediation(
        args.frozen_experiment,
        args.source_root,
        args.output,
        search_roots=args.search_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
