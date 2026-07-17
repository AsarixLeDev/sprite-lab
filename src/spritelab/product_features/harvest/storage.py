"""Fail-closed filesystem primitives for the Harvest product feature."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
import unicodedata
from collections import Counter
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps, strict_json_loads
from spritelab.product_features.harvest.trusted_backend import AcquiredFile, HarvestLimits
from spritelab.utils.safe_fs import require_confined_path

_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
)
_LEGACY_FILES = ("sources.jsonl", "candidates.jsonl", "imported.jsonl")
_MAX_LEGACY_FILE_BYTES = 64 * 1024 * 1024
_MAX_LEGACY_RECORDS = 100_000
_QUARANTINE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TAXONOMY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class HarvestStorageError(ValueError):
    pass


class RepositoryMutationLock(AbstractContextManager["RepositoryMutationLock"]):
    """Bounded cross-process lock created only for an explicit mutation."""

    def __init__(self, root: Path, *, timeout_seconds: float = 5.0) -> None:
        self.root = root
        self.timeout_seconds = timeout_seconds
        self._handle: Any = None

    def __enter__(self) -> RepositoryMutationLock:
        lock_path = require_confined_path(self.root / ".harvest.lock", self.root)
        self._handle = _open_owned_lock(lock_path)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                _lock_handle(self._handle)
                return self
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise HarvestStorageError("Another Harvest process holds the mutation lock.") from None
                time.sleep(0.01)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        if self._handle is not None:
            try:
                _unlock_handle(self._handle)
            finally:
                self._handle.close()
                self._handle = None


def scan_artifacts(
    artifacts: Path,
    limits: HarvestLimits,
    *,
    expected_files: tuple[AcquiredFile, ...] | None = None,
) -> dict[str, Any]:
    """Hash a confined tree with mount/link/name/size/depth/TOCTOU checks."""

    root_metadata = artifacts.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or _metadata_is_link_or_reparse(root_metadata):
        raise HarvestStorageError("Harvest artifact root is not a safe directory.")
    if artifacts.is_mount():
        raise HarvestStorageError("Harvest artifact root cannot be a mount point.")
    root_device = root_metadata.st_dev
    expected = {item.relative_path: item for item in expected_files or ()}
    if len(expected) != len(expected_files or ()):
        raise HarvestStorageError("Backend artifact receipt contains duplicate paths.")

    records: list[dict[str, Any]] = []
    collision_keys: set[str] = set()
    total_bytes = 0
    stack: list[tuple[Path, int]] = [(artifacts, 0)]
    while stack:
        directory, directory_depth = stack.pop()
        if directory_depth > limits.max_depth:
            raise HarvestStorageError("Harvest artifact directory depth exceeds the configured limit.")
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            require_confined_path(item, artifacts)
            relative = item.relative_to(artifacts)
            _validate_relative_path(relative)
            depth = len(relative.parts)
            if depth > limits.max_depth:
                raise HarvestStorageError("Harvest artifact depth exceeds the configured limit.")
            collision_key = "/".join(unicodedata.normalize("NFC", part).casefold() for part in relative.parts)
            if collision_key in collision_keys:
                raise HarvestStorageError("Harvest artifacts contain a case or Unicode name collision.")
            collision_keys.add(collision_key)

            metadata = item.lstat()
            if _metadata_is_link_or_reparse(metadata):
                raise HarvestStorageError("Harvest artifacts cannot contain links or reparse points.")
            if metadata.st_dev != root_device or item.is_mount():
                raise HarvestStorageError("Harvest artifacts cannot cross a device or mount boundary.")
            if stat.S_ISDIR(metadata.st_mode):
                stack.append((item, depth))
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise HarvestStorageError("Harvest artifacts must be singly linked regular files.")
            if len(records) >= limits.max_files:
                raise HarvestStorageError("Harvest artifact count exceeds the configured limit.")
            if metadata.st_size > limits.max_file_bytes:
                raise HarvestStorageError("A Harvest artifact exceeds the per-file byte limit.")
            total_bytes += metadata.st_size
            if total_bytes > limits.max_total_bytes:
                raise HarvestStorageError("Harvest artifact bytes exceed the whole-run limit.")

            digest, prefix = _stable_file_hash(item, metadata)
            mime_type = _sniff_mime(prefix)
            if mime_type not in limits.allowed_artifact_mime_types:
                raise HarvestStorageError("Harvest artifact MIME type is outside the configured allowlist.")
            path_text = relative.as_posix()
            claimed = expected.get(path_text)
            if expected_files is not None and claimed is None:
                raise HarvestStorageError("Backend receipt omitted a Harvest artifact.")
            if claimed is not None:
                if claimed.byte_count != metadata.st_size or claimed.sha256 != digest or claimed.mime_type != mime_type:
                    raise HarvestStorageError("Harvest artifact does not match its backend receipt.")
                usable = claimed.usable
                quarantine_reason = claimed.quarantine_reason
                taxonomy = claimed.taxonomy
            else:
                usable = True
                quarantine_reason = None
                taxonomy = ()
            if quarantine_reason is not None and _QUARANTINE_PATTERN.fullmatch(quarantine_reason) is None:
                raise HarvestStorageError("Harvest quarantine reason is not a controlled token.")
            if any(_TAXONOMY_PATTERN.fullmatch(value) is None for value in taxonomy):
                raise HarvestStorageError("Harvest taxonomy contains an uncontrolled value.")
            records.append(
                {
                    "relative_path": path_text,
                    "byte_count": metadata.st_size,
                    "expected_sha256": claimed.sha256 if claimed else digest,
                    "actual_sha256": digest,
                    "mime_type": mime_type,
                    "usable": usable,
                    "quarantine_reason": quarantine_reason,
                    "taxonomy": list(taxonomy),
                }
            )
    records.sort(key=lambda value: value["relative_path"])
    if expected_files is not None and set(expected) != {item["relative_path"] for item in records}:
        raise HarvestStorageError("Backend receipt names an artifact that is not present.")
    taxonomy_counts: Counter[str] = Counter()
    for record in records:
        taxonomy_counts.update(record["taxonomy"])
    identity_payload = [
        {
            "relative_path": record["relative_path"],
            "byte_count": record["byte_count"],
            "sha256": record["actual_sha256"],
            "mime_type": record["mime_type"],
            "usable": record["usable"],
            "quarantine_reason": record["quarantine_reason"],
            "taxonomy": record["taxonomy"],
        }
        for record in records
    ]
    return {
        "schema_version": "spritelab.harvest.artifact-manifest.v1",
        "artifact_count": len(records),
        "usable_count": sum(record["usable"] for record in records),
        "quarantined_count": sum(not record["usable"] for record in records),
        "total_bytes": total_bytes,
        "max_depth_observed": max((len(Path(record["relative_path"]).parts) for record in records), default=0),
        "artifact_set_identity": _identity(identity_payload),
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "files": records,
        "paths_are_relative": True,
        "absolute_paths_exposed": False,
    }


def scan_legacy_run(directory: Path) -> dict[str, Any] | None:
    """Index recognized immediate legacy JSONL files without traversing them."""

    if _is_link_or_reparse(directory) or not directory.is_dir() or directory.is_mount():
        raise HarvestStorageError("Legacy Harvest entry is not a safe directory.")
    records: dict[str, Any] = {}
    for name in _LEGACY_FILES:
        path = directory / name
        if not os.path.lexists(path):
            continue
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_LEGACY_FILE_BYTES
        ):
            raise HarvestStorageError("Legacy Harvest JSONL evidence is unsafe or oversized.")
        digest, _prefix, line_count, valid_count = _stable_jsonl_summary(path, metadata)
        records[name] = {
            "byte_count": metadata.st_size,
            "sha256": digest,
            "line_count": line_count,
            "valid_record_count": valid_count,
            "all_records_valid": line_count == valid_count,
        }
    if not records:
        return None
    return {
        "schema_version": "spritelab.harvest.legacy-run-index.v1",
        "legacy_id": directory.name,
        "status": "LEGACY_READ_ONLY",
        "files": records,
        "source_records": records.get("sources.jsonl", {}).get("valid_record_count", 0),
        "candidate_records": records.get("candidates.jsonl", {}).get("valid_record_count", 0),
        "imported_records": records.get("imported.jsonl", {}).get("valid_record_count", 0),
        "legacy_identity": _identity(records),
        "mutation_allowed": False,
        "paths_exposed": False,
    }


def _validate_relative_path(relative: Path) -> None:
    if relative.is_absolute() or not relative.parts:
        raise HarvestStorageError("Harvest artifact path is not relative.")
    for part in relative.parts:
        if part in {"", ".", ".."} or part != unicodedata.normalize("NFC", part):
            raise HarvestStorageError("Harvest artifact names must be canonical NFC components.")
        if part.endswith((" ", ".")) or any(ord(character) < 32 for character in part):
            raise HarvestStorageError("Harvest artifact name is unsafe.")
        if any(character in part for character in '<>:"/\\|?*'):
            raise HarvestStorageError("Harvest artifact name contains a reserved character.")
        if part.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise HarvestStorageError("Harvest artifact name is reserved on Windows.")


def _stable_file_hash(path: Path, before: os.stat_result) -> tuple[str, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    prefix = b""
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise HarvestStorageError("Harvest artifact changed while it was being opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                if len(prefix) < 32:
                    prefix += chunk[: 32 - len(prefix)]
                digest.update(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or _metadata_is_link_or_reparse(after)
    ):
        raise HarvestStorageError("Harvest artifact changed while it was being hashed.")
    return digest.hexdigest(), prefix


def _stable_jsonl_summary(path: Path, before: os.stat_result) -> tuple[str, bytes, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    prefix = b""
    line_count = 0
    valid_count = 0
    try:
        opened = os.fstat(descriptor)
        if opened.st_ino != before.st_ino or opened.st_dev != before.st_dev or opened.st_nlink != 1:
            raise HarvestStorageError("Legacy Harvest evidence changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for raw_line in handle:
                if line_count >= _MAX_LEGACY_RECORDS:
                    raise HarvestStorageError("Legacy Harvest evidence exceeds the record cap.")
                digest.update(raw_line)
                if len(prefix) < 32:
                    prefix += raw_line[: 32 - len(prefix)]
                if raw_line.strip():
                    line_count += 1
                    try:
                        parsed = strict_json_loads(raw_line)
                    except ValueError:
                        continue
                    if isinstance(parsed, dict):
                        valid_count += 1
    finally:
        os.close(descriptor)
    after = path.lstat()
    if after.st_ino != before.st_ino or after.st_dev != before.st_dev or after.st_mtime_ns != before.st_mtime_ns:
        raise HarvestStorageError("Legacy Harvest evidence changed while hashing.")
    return digest.hexdigest(), prefix, line_count, valid_count


def _sniff_mime(prefix: bytes) -> str:
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _open_owned_lock(path: Path) -> Any:
    if os.path.lexists(path):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or _metadata_is_link_or_reparse(metadata) or metadata.st_nlink != 1:
            raise HarvestStorageError("Harvest mutation lock is unsafe.")
        handle = path.open("r+b", buffering=0)
    else:
        try:
            handle = path.open("x+b", buffering=0)
        except FileExistsError:
            return _open_owned_lock(path)
        handle.write(b"0")
        handle.flush()
        os.fsync(handle.fileno())
    opened = os.fstat(handle.fileno())
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        handle.close()
        raise HarvestStorageError("Harvest mutation lock changed while opening.")
    if opened.st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    return handle


def _lock_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _identity(value: Any) -> str:
    payload = strict_json_dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_FLAG)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        return _metadata_is_link_or_reparse(path.lstat())
    except FileNotFoundError:
        return False


__all__ = [
    "HarvestStorageError",
    "RepositoryMutationLock",
    "scan_artifacts",
    "scan_legacy_run",
]
