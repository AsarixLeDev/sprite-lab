"""Strict discovery of original source bytes for the Dataset-v5 raw rebuild.

The inventory is intentionally provenance-facing.  Original filenames and
source metadata are retained here, but callers must not pass this record to a
blind semantic provider.  Discovery never uses modification times and never
downloads or mutates a source.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from spritelab.dataset_v5.identity import canonical_json_bytes
from spritelab.harvest.sources import (
    TRAINING_ALLOWED_LICENSES,
    license_requires_attribution,
    normalize_license_name,
)
from spritelab.utils.safe_fs import remove_confined_tree

RAW_SOURCE_INVENTORY_SCHEMA_VERSION = "sprite_lab_raw_source_inventory_v1"
SOURCE_ARCHIVE_HASH_SCHEMA_VERSION = "sprite_lab_source_archive_hashes_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RawInventoryError(RuntimeError):
    """Base error for fail-closed raw source discovery."""


class AmbiguousRawSourceError(RawInventoryError):
    """Raised when more than one distinct source byte stream could be used."""


class RawSourceHashMismatchError(RawInventoryError):
    """Raised when no discovered source matches its recorded digest."""


@dataclass(frozen=True)
class RawSourceRecord:
    """One verified occurrence from a harvest ``sources.jsonl`` manifest."""

    acquisition_run: str
    source_id: str
    source_name: str
    source_type: str
    source_url: str
    download_url: str
    distribution_platform: str
    creator_or_publisher: str
    original_filename: str
    manifest_path: str
    manifest_sha256: str
    source_row_sha256: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    expected_archive_sha256: str | None
    resolution_method: str
    provenance_status: str
    license: Mapping[str, Any]
    source_record: Mapping[str, Any]
    resolved_archive_path: Path = field(repr=False, compare=False)
    schema_version: str = RAW_SOURCE_INVENTORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic provenance representation."""

        return {
            "acquisition_run": self.acquisition_run,
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "creator_or_publisher": self.creator_or_publisher,
            "distribution_platform": self.distribution_platform,
            "download_url": self.download_url,
            "expected_archive_sha256": self.expected_archive_sha256,
            "license": dict(self.license),
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "original_filename": self.original_filename,
            "provenance_status": self.provenance_status,
            "resolution_method": self.resolution_method,
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_record": dict(self.source_record),
            "source_row_sha256": self.source_row_sha256,
            "source_type": self.source_type,
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class _PathCandidate:
    priority: int
    method: str
    path: Path


def file_sha256(path: str | Path) -> str:
    """Hash a file without relying on filesystem metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_raw_sources(
    workspace_root: str | Path,
    *,
    harvest_root: str | Path | None = None,
    acquisition_download_roots: Iterable[str | Path] = (),
) -> list[RawSourceRecord]:
    """Discover and verify every ``harvest_runs/*/sources.jsonl`` row.

    Resolution checks, in order, an explicitly recorded local path, the
    acquisition run's ``downloads`` directory, URL/original basenames, and
    other declared acquisition download roots.  A digest match can recover a
    byte-identical download whose cache filename changed; distinct plausible
    byte streams are never guessed between.
    """

    workspace = Path(workspace_root).resolve()
    harvest = _resolve_from(workspace, harvest_root) if harvest_root is not None else workspace / "harvest_runs"
    harvest = harvest.resolve()
    if not harvest.is_dir():
        raise RawInventoryError(f"missing harvest root: {harvest}")
    manifests = sorted(harvest.glob("*/sources.jsonl"), key=lambda path: path.as_posix())
    if not manifests:
        raise RawInventoryError(f"no harvest source manifests found under {harvest}")

    extra_roots = _download_roots(workspace, harvest, acquisition_download_roots)
    records: list[RawSourceRecord] = []
    for manifest in manifests:
        rows = _read_jsonl_strict(manifest)
        if not rows:
            raise RawInventoryError(f"empty source manifest: {manifest}")
        manifest_hash = file_sha256(manifest)
        seen_source_ids: set[str] = set()
        for line_number, row in rows:
            source_id = _required_text(row, "source_id", manifest, line_number)
            if source_id in seen_source_ids:
                raise RawInventoryError(f"duplicate source_id {source_id!r} in {manifest}")
            seen_source_ids.add(source_id)
            records.append(
                _build_source_record(
                    workspace=workspace,
                    manifest=manifest,
                    manifest_hash=manifest_hash,
                    line_number=line_number,
                    row=row,
                    extra_roots=extra_roots,
                )
            )
    return sorted(records, key=lambda item: (item.acquisition_run, item.source_id, item.source_row_sha256))


def write_raw_source_inventory(records: Iterable[RawSourceRecord], output_root: str | Path) -> Path:
    """Write a new immutable inventory directory and refuse any existing root."""

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"raw inventory output already exists: {destination}")
    rows = sorted((record.to_dict() for record in records), key=_inventory_sort_key)
    if not rows:
        raise RawInventoryError("cannot write an empty raw source inventory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        _write_new_text(
            staging / "raw_source_inventory.jsonl",
            "".join(canonical_json_bytes(row).decode("utf-8") + "\n" for row in rows),
        )
        archive_hashes: dict[str, str] = {}
        for row in rows:
            path = str(row["archive_path"])
            digest = str(row["archive_sha256"])
            previous = archive_hashes.setdefault(path, digest)
            if previous != digest:
                raise RawInventoryError(f"one archive path has multiple hashes: {path}")
        _write_new_json(
            staging / "source_archive_hashes.json",
            {
                "archives": dict(sorted(archive_hashes.items())),
                "schema_version": SOURCE_ARCHIVE_HASH_SCHEMA_VERSION,
            },
        )
        staging.replace(destination)
    except Exception:
        remove_confined_tree(staging, destination.parent, missing_ok=True)
        raise
    return destination


def write_raw_inventory(
    workspace_root: str | Path,
    output_root: str | Path,
    *,
    harvest_root: str | Path | None = None,
    acquisition_download_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Discover, verify, and transactionally write one fresh inventory.

    This convenience entry point is suitable for the versioned CLI.  The
    lower-level functions remain available when orchestration needs to inspect
    verified records before deciding whether any output should be written.
    """

    records = discover_raw_sources(
        workspace_root,
        harvest_root=harvest_root,
        acquisition_download_roots=acquisition_download_roots,
    )
    destination = write_raw_source_inventory(records, output_root)
    return {
        "archive_count": len({record.archive_sha256 for record in records}),
        "expected_hash_verified_count": sum(record.expected_archive_sha256 is not None for record in records),
        "output_root": str(destination),
        "schema_version": RAW_SOURCE_INVENTORY_SCHEMA_VERSION,
        "source_record_count": len(records),
    }


def _build_source_record(
    *,
    workspace: Path,
    manifest: Path,
    manifest_hash: str,
    line_number: int,
    row: dict[str, Any],
    extra_roots: tuple[Path, ...],
) -> RawSourceRecord:
    source_id = _required_text(row, "source_id", manifest, line_number)
    source_name = _required_text(row, "source_name", manifest, line_number)
    source_type = _required_text(row, "source_type", manifest, line_number).lower()
    creator = str(row.get("author") or row.get("creator") or row.get("publisher") or "").strip()
    if not creator:
        raise RawInventoryError(f"missing creator/publisher for {source_id} in {manifest}:{line_number}")
    source_url = str(row.get("source_url") or "").strip()
    download_url = str(row.get("download_url") or "").strip()
    if not any((source_url, download_url, str(row.get("local_archive_path") or "").strip())):
        raise RawInventoryError(f"missing acquisition provenance for {source_id} in {manifest}:{line_number}")

    license_record = _validated_license(row, source_id=source_id, manifest=manifest, line_number=line_number)
    expected_hash = _expected_hash(row, source_id=source_id)
    selected, actual_hash, resolution_method = _resolve_source_bytes(
        workspace=workspace,
        manifest=manifest,
        row=row,
        expected_hash=expected_hash,
        extra_roots=extra_roots,
    )
    recorded_size = row.get("download_size_bytes")
    if recorded_size not in (None, "", 0):
        if not isinstance(recorded_size, int) or recorded_size < 0:
            raise RawInventoryError(f"invalid download_size_bytes for {source_id}")
        if selected.stat().st_size != recorded_size:
            raise RawInventoryError(
                f"download size mismatch for {source_id}: expected {recorded_size}, got {selected.stat().st_size}"
            )

    canonical_row = json.loads(canonical_json_bytes(row).decode("utf-8"))
    provenance_status = "expected_archive_hash_verified" if expected_hash else "source_bytes_hashed_no_expected_digest"
    return RawSourceRecord(
        acquisition_run=manifest.parent.name,
        source_id=source_id,
        source_name=source_name,
        source_type=source_type,
        source_url=source_url,
        download_url=download_url,
        distribution_platform=_distribution_platform(source_url, download_url, source_type),
        creator_or_publisher=creator,
        original_filename=str(row.get("original_filename") or selected.name),
        manifest_path=_display_path(manifest, workspace),
        manifest_sha256=manifest_hash,
        source_row_sha256=hashlib.sha256(canonical_json_bytes(canonical_row)).hexdigest(),
        archive_path=_display_path(selected, workspace),
        archive_sha256=actual_hash,
        archive_size_bytes=selected.stat().st_size,
        expected_archive_sha256=expected_hash,
        resolution_method=resolution_method,
        provenance_status=provenance_status,
        license=license_record,
        source_record=canonical_row,
        resolved_archive_path=selected,
    )


def _validated_license(row: Mapping[str, Any], *, source_id: str, manifest: Path, line_number: int) -> dict[str, Any]:
    raw = row.get("license")
    if isinstance(raw, str):
        license_data: dict[str, Any] = {"license": raw}
    elif isinstance(raw, Mapping):
        license_data = dict(raw)
    else:
        raise RawInventoryError(f"missing license for {source_id} in {manifest}:{line_number}")
    name = normalize_license_name(str(license_data.get("license") or ""))
    if name not in TRAINING_ALLOWED_LICENSES:
        raise RawInventoryError(f"unknown or disallowed license {name!r} for {source_id}")
    confirmed = bool(license_data.get("user_confirmed", row.get("license_confirmed", False)))
    if not confirmed:
        raise RawInventoryError(f"unconfirmed license for {source_id}")
    if bool(license_data.get("no_ai_training_flag", False)):
        raise RawInventoryError(f"no-AI-training source cannot enter the rebuild: {source_id}")
    if license_data.get("derivatives_allowed") is False:
        raise RawInventoryError(f"no-derivatives source cannot enter the rebuild: {source_id}")
    creator = str(row.get("author") or row.get("creator") or row.get("publisher") or "").strip()
    if license_requires_attribution(name) and not creator:
        raise RawInventoryError(f"attribution-required source lacks a creator: {source_id}")
    license_data["license"] = name
    license_data["user_confirmed"] = True
    return json.loads(canonical_json_bytes(license_data).decode("utf-8"))


def _expected_hash(row: Mapping[str, Any], *, source_id: str) -> str | None:
    values = {
        str(row.get(field) or "").strip().lower()
        for field in ("download_sha256", "sha256", "archive_sha256", "expected_sha256")
        if str(row.get(field) or "").strip()
    }
    if len(values) > 1:
        raise RawInventoryError(f"conflicting expected archive hashes for {source_id}: {sorted(values)}")
    if not values:
        return None
    value = values.pop()
    if _SHA256_RE.fullmatch(value) is None:
        raise RawInventoryError(f"invalid expected SHA-256 for {source_id}: {value!r}")
    return value


def _resolve_source_bytes(
    *,
    workspace: Path,
    manifest: Path,
    row: Mapping[str, Any],
    expected_hash: str | None,
    extra_roots: tuple[Path, ...],
) -> tuple[Path, str, str]:
    candidates: list[_PathCandidate] = []
    raw_local = str(row.get("local_archive_path") or "").strip()
    if raw_local:
        local = Path(raw_local.replace("\\", "/"))
        local_paths = (local,) if local.is_absolute() else (manifest.parent / local, workspace / local)
        for path in local_paths:
            candidates.append(_PathCandidate(0, "local_archive_path", path))

    basenames = _source_basenames(row)
    run_downloads = manifest.parent / "downloads"
    search_roots = (run_downloads, *extra_roots)
    run_download_files = _iter_files(run_downloads) if run_downloads.is_dir() else []
    if len(run_download_files) == 1:
        candidates.append(_PathCandidate(15, "run_downloads_unique_file", run_download_files[0]))
    for root_index, root in enumerate(search_roots):
        if not root.is_dir():
            continue
        method = "run_downloads" if root.resolve() == run_downloads.resolve() else "acquisition_downloads"
        priority = 10 + root_index
        for path in _iter_files(root):
            if path.name.casefold() in basenames:
                candidates.append(_PathCandidate(priority, f"{method}_basename", path))

    existing = _unique_candidates(candidate for candidate in candidates if candidate.path.is_file())
    if expected_hash:
        existing_paths = {candidate.path for candidate in existing}
        for root_index, root in enumerate(search_roots):
            if not root.is_dir():
                continue
            for path in _iter_files(root):
                resolved = path.resolve()
                if resolved not in existing_paths:
                    existing.append(_PathCandidate(100 + root_index, "acquisition_downloads_digest", resolved))
                    existing_paths.add(resolved)
    if not existing:
        attempted = sorted({candidate.path.as_posix() for candidate in candidates})
        raise RawInventoryError(f"missing original source bytes; attempted: {attempted}")

    hashed: list[tuple[_PathCandidate, str]] = []
    for candidate in existing:
        hashed.append((candidate, file_sha256(candidate.path)))
    if expected_hash:
        matches = [(candidate, digest) for candidate, digest in hashed if digest == expected_hash]
        if not matches:
            observed = sorted({digest for _, digest in hashed})
            raise RawSourceHashMismatchError(
                f"no source candidate matches expected SHA-256 {expected_hash}; observed {observed}"
            )
    else:
        observed = {digest for _, digest in hashed}
        if len(observed) != 1:
            details = [(item.path.as_posix(), digest) for item, digest in hashed]
            raise AmbiguousRawSourceError(f"ambiguous original source bytes without an expected hash: {details}")
        matches = hashed
    selected, digest = min(matches, key=lambda item: (item[0].priority, item[0].path.as_posix().casefold()))
    return selected.path.resolve(), digest, selected.method


def _download_roots(
    workspace: Path, harvest: Path, acquisition_download_roots: Iterable[str | Path]
) -> tuple[Path, ...]:
    roots = [
        workspace / "acquisition_downloads",
        workspace / "attachment_downloads",
        workspace / "downloads",
    ]
    roots.extend(
        sorted((path / "downloads" for path in harvest.iterdir() if path.is_dir()), key=lambda p: p.as_posix())
    )
    roots.extend(_resolve_from(workspace, path) for path in acquisition_download_roots)
    unique: dict[Path, None] = {}
    for root in roots:
        unique[root.resolve()] = None
    return tuple(unique)


def _source_basenames(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    original = str(row.get("original_filename") or "").strip()
    if original:
        values.add(Path(original.replace("\\", "/")).name.casefold())
    for field_name in ("download_url", "source_url"):
        url = str(row.get(field_name) or "").strip()
        if url:
            name = Path(unquote(urlparse(url).path)).name
            if name:
                values.add(name.casefold())
    return values


def _read_jsonl_strict(path: Path) -> list[tuple[int, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RawInventoryError(f"cannot read source manifest {path}: {exc}") from exc
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RawInventoryError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise RawInventoryError(f"source row must be an object in {path}:{line_number}")
        rows.append((line_number, value))
    return rows


def _required_text(row: Mapping[str, Any], field_name: str, manifest: Path, line_number: int) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise RawInventoryError(f"missing or invalid {field_name} in {manifest}:{line_number}")
    return value.strip()


def _iter_files(root: Path) -> list[Path]:
    return sorted((path.resolve() for path in root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())


def _unique_candidates(candidates: Iterable[_PathCandidate]) -> list[_PathCandidate]:
    unique: dict[Path, _PathCandidate] = {}
    for candidate in candidates:
        resolved = candidate.path.resolve()
        current = unique.get(resolved)
        normalized = _PathCandidate(candidate.priority, candidate.method, resolved)
        if current is None or (normalized.priority, normalized.method) < (current.priority, current.method):
            unique[resolved] = normalized
    return sorted(unique.values(), key=lambda item: (item.priority, item.path.as_posix().casefold()))


def _distribution_platform(source_url: str, download_url: str, source_type: str) -> str:
    for value in (source_url, download_url):
        host = urlparse(value).hostname
        if host:
            return host.casefold()
    return source_type


def _inventory_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["acquisition_run"]), str(row["source_id"]), str(row["source_row_sha256"])


def _display_path(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_from(workspace: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _write_new_json(path: Path, value: Any) -> None:
    _write_new_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
