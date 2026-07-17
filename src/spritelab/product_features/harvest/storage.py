"""Fail-closed filesystem primitives for the Harvest product feature."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Any

from spritelab.product_core.events import strict_json_dumps, strict_json_loads
from spritelab.product_features.harvest.trusted_backend import AcquiredFile, HarvestLimits
from spritelab.utils.safe_fs import AnchoredDirectory, open_anchored_directory, require_confined_path

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
        self._lock_path: Path | None = None
        self._root_metadata: os.stat_result | None = None
        self._root_parent_metadata: os.stat_result | None = None
        self._anchor: AnchoredDirectory | None = None

    def __enter__(self) -> RepositoryMutationLock:
        self._root_metadata = self.root.lstat()
        self._root_parent_metadata = self.root.parent.lstat()
        if (
            not stat.S_ISDIR(self._root_metadata.st_mode)
            or _metadata_is_link_or_reparse(self._root_metadata)
            or self.root.is_mount()
            or not stat.S_ISDIR(self._root_parent_metadata.st_mode)
            or _metadata_is_link_or_reparse(self._root_parent_metadata)
        ):
            raise HarvestStorageError("Harvest mutation-lock root is unsafe.")
        lock_path = require_confined_path(self.root / ".harvest.lock", self.root)
        self._anchor = AnchoredDirectory(self.root, self.root)
        try:
            self._anchor.__enter__()
            self._handle = _open_owned_lock(self._anchor, lock_path.name)
        except BaseException:
            if self._anchor is not None:
                self._anchor.__exit__(None, None, None)
                self._anchor = None
            raise
        self._lock_path = lock_path
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                _lock_handle(self._handle)
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    self._anchor.__exit__(None, None, None)
                    self._anchor = None
                    raise HarvestStorageError("Another Harvest process holds the mutation lock.") from None
                time.sleep(0.01)
                continue
            try:
                _verify_open_lock(self._handle, self._anchor, lock_path.name)
                _require_same_directory(
                    self._root_metadata,
                    self.root.lstat(),
                    "Harvest mutation-lock root changed while acquiring the lock.",
                )
                _require_same_directory(
                    self._root_parent_metadata,
                    self.root.parent.lstat(),
                    "Harvest mutation-lock root was renamed while acquiring the lock.",
                    compare_times=self.root.parent != self.root,
                )
                return self
            except BaseException:
                try:
                    _unlock_handle(self._handle)
                finally:
                    self._handle.close()
                    self._handle = None
                    self._anchor.__exit__(None, None, None)
                    self._anchor = None
                raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        if self._handle is not None:
            try:
                if (
                    self._lock_path is None
                    or self._root_metadata is None
                    or self._root_parent_metadata is None
                    or self._anchor is None
                ):
                    raise HarvestStorageError("Harvest mutation lock lost its ownership evidence.")
                _verify_open_lock(self._handle, self._anchor, self._lock_path.name)
                _require_same_directory(
                    self._root_metadata,
                    self.root.lstat(),
                    "Harvest mutation-lock root changed while held.",
                )
                _require_same_directory(
                    self._root_parent_metadata,
                    self.root.parent.lstat(),
                    "Harvest mutation-lock root was renamed while held.",
                    compare_times=self.root.parent != self.root,
                )
                _unlock_handle(self._handle)
            finally:
                self._handle.close()
                self._handle = None
                self._lock_path = None
                self._root_metadata = None
                self._root_parent_metadata = None
                if self._anchor is not None:
                    self._anchor.__exit__(None, None, None)
                    self._anchor = None


def read_stable_single_link_bytes(
    path: Path,
    root: Path,
    *,
    max_bytes: int,
    parent_anchor: AnchoredDirectory | None = None,
) -> bytes:
    """Read one confined file through a verified no-follow descriptor."""

    if max_bytes <= 0:
        raise ValueError("stable read byte limit must be positive")
    with _anchored_file_parent(path, root, parent_anchor=parent_anchor) as (parent, name):
        before = parent.lstat(name)
        _require_safe_regular(before, max_bytes=max_bytes, message="Harvest durable evidence is unsafe.")
        descriptor = parent.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            opened = os.fstat(descriptor)
            _require_same_regular_file(before, opened, "Harvest durable evidence changed while opening.")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read(max_bytes + 1)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = parent.lstat(name)
        _require_same_regular_file(before, opened_after, "Harvest durable evidence changed while reading.")
        _require_same_regular_file(before, path_after, "Harvest durable evidence path changed while reading.")
        if len(payload) != before.st_size or len(payload) > max_bytes:
            raise HarvestStorageError("Harvest durable evidence changed size while reading.")
        return payload


def append_stable_single_link_bytes(
    path: Path,
    root: Path,
    payload: bytes,
    *,
    max_bytes: int,
    max_total_bytes: int,
    parent_anchor: AnchoredDirectory | None = None,
) -> None:
    """Append one bounded record with a verified persistent path/inode binding."""

    if not payload or len(payload) > max_bytes or max_total_bytes < len(payload):
        raise HarvestStorageError("Harvest event payload exceeds its bounded append limit.")
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with _anchored_file_parent(path, root, parent_anchor=parent_anchor) as (parent, name):
        before: os.stat_result | None = None
        for _attempt in range(8):
            if parent.lexists(name):
                before = parent.lstat(name)
                _require_safe_regular(before, max_bytes=None, message="Harvest event log is unsafe.")
                try:
                    descriptor = parent.open_file(name, flags)
                except FileNotFoundError:
                    continue
            else:
                try:
                    descriptor = parent.open_file(name, flags | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    continue
                before = os.fstat(descriptor)
            break
        else:
            raise HarvestStorageError("Harvest event log changed repeatedly while opening.")
        try:
            opened = os.fstat(descriptor)
            if before is None:
                raise HarvestStorageError("Harvest event log identity is unavailable.")
            _require_same_regular_file(before, opened, "Harvest event log changed while opening.")
            if opened.st_size + len(payload) > max_total_bytes:
                raise HarvestStorageError("Harvest event log exceeds its total byte cap.")
            path_opened = parent.lstat(name)
            _require_same_regular_file(opened, path_opened, "Harvest event log path changed while opening.")
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise HarvestStorageError("Harvest event append was incomplete.")
            os.fsync(descriptor)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = parent.lstat(name)
        if opened_after.st_size != opened.st_size + len(payload):
            raise HarvestStorageError("Harvest event log size changed unexpectedly during append.")
        _require_same_regular_file(
            opened_after,
            path_after,
            "Harvest event log path changed during append.",
            compare_size=True,
        )


def write_exclusive_stable_bytes(
    path: Path,
    root: Path,
    payload: bytes,
    *,
    max_bytes: int,
    parent_anchor: AnchoredDirectory | None = None,
) -> None:
    """Create one durable file exclusively and bind its path to the written inode."""

    if not payload or len(payload) > max_bytes:
        raise HarvestStorageError("Harvest durable evidence payload is empty or oversized.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with _anchored_file_parent(path, root, parent_anchor=parent_anchor) as (parent, name):
        descriptor = parent.open_file(name, flags, 0o600)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_size != 0:
                raise HarvestStorageError("Harvest durable evidence descriptor is unsafe.")
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = parent.lstat(name)
        if opened_after.st_size != len(payload):
            raise HarvestStorageError("Harvest durable evidence write was incomplete.")
        _require_same_regular_file(
            opened_after,
            path_after,
            "Harvest durable evidence path changed during exclusive publication.",
        )


def write_atomic_stable_bytes(
    path: Path,
    root: Path,
    payload: bytes,
    *,
    max_bytes: int,
    parent_anchor: AnchoredDirectory | None = None,
) -> None:
    """Atomically replace one bounded file through a root-held parent chain."""

    if not payload or len(payload) > max_bytes:
        raise HarvestStorageError("Harvest durable evidence payload is empty or oversized.")
    with _anchored_file_parent(path, root, parent_anchor=parent_anchor) as (parent, name):
        parent.atomic_write_bytes(name, payload)
        after = parent.lstat(name)
        _require_safe_regular(after, max_bytes=max_bytes, message="Harvest durable evidence publication is unsafe.")


@contextmanager
def _anchored_file_parent(
    path: Path,
    root: Path,
    *,
    parent_anchor: AnchoredDirectory | None = None,
) -> Iterator[tuple[AnchoredDirectory, str]]:
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(root))))
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if parent_anchor is not None:
        if path.parent != parent_anchor.directory or Path(path.name).name != path.name:
            raise HarvestStorageError("Supplied Harvest evidence anchor does not match its file parent.")
        parent_anchor.verify()
        yield parent_anchor, path.name
        parent_anchor.verify()
        return
    path = require_confined_path(path, root)
    relative = path.relative_to(root)
    if not relative.parts:
        raise HarvestStorageError("Harvest durable evidence path has no file name.")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root, root))
        for part in relative.parts[:-1]:
            if not anchor.lexists(part):
                raise FileNotFoundError(path.parent)
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
        yield anchor, relative.parts[-1]


def scan_artifacts(
    artifacts: Path,
    limits: HarvestLimits,
    *,
    expected_files: tuple[AcquiredFile, ...] | None = None,
    artifacts_anchor: AnchoredDirectory | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Hash a confined tree with mount/link/name/size/depth/TOCTOU checks."""

    artifacts = Path(os.path.abspath(os.path.expanduser(os.fspath(artifacts))))
    if artifacts_anchor is None:
        with open_anchored_directory(artifacts, artifacts) as anchored:
            return scan_artifacts(
                artifacts,
                limits,
                expected_files=expected_files,
                artifacts_anchor=anchored,
                cancel_requested=cancel_requested,
                deadline_monotonic=deadline_monotonic,
            )
    artifacts_anchor.verify()
    if artifacts_anchor.directory != artifacts:
        raise HarvestStorageError("Harvest artifact root does not match its supplied anchor.")
    root_metadata = artifacts_anchor.directory_metadata()
    if not stat.S_ISDIR(root_metadata.st_mode) or _metadata_is_link_or_reparse(root_metadata):
        raise HarvestStorageError("Harvest artifact root is not a safe directory.")
    root_device = root_metadata.st_dev
    expected = {item.relative_path: item for item in expected_files or ()}
    if len(expected) != len(expected_files or ()):
        raise HarvestStorageError("Backend artifact receipt contains duplicate paths.")

    records: list[dict[str, Any]] = []
    collision_keys: set[str] = set()
    total_bytes = 0

    def visit(directory: AnchoredDirectory, prefix: tuple[str, ...], directory_depth: int) -> None:
        nonlocal total_bytes
        _check_storage_abort(cancel_requested, deadline_monotonic)
        if directory_depth > limits.max_depth:
            raise HarvestStorageError("Harvest artifact directory depth exceeds the configured limit.")
        directory.verify()
        for name in directory.names():
            _check_storage_abort(cancel_requested, deadline_monotonic)
            relative = Path(*prefix, name)
            _validate_relative_path(relative)
            depth = len(relative.parts)
            if depth > limits.max_depth:
                raise HarvestStorageError("Harvest artifact depth exceeds the configured limit.")
            collision_key = "/".join(unicodedata.normalize("NFC", part).casefold() for part in relative.parts)
            if collision_key in collision_keys:
                raise HarvestStorageError("Harvest artifacts contain a case or Unicode name collision.")
            collision_keys.add(collision_key)

            metadata = directory.lstat(name)
            if _metadata_is_link_or_reparse(metadata):
                raise HarvestStorageError("Harvest artifacts cannot contain links or reparse points.")
            if metadata.st_dev != root_device:
                raise HarvestStorageError("Harvest artifacts cannot cross a device or mount boundary.")
            if stat.S_ISDIR(metadata.st_mode):
                with directory.open_directory_immovable(name) as child:
                    visit(child, (*prefix, name), depth)
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

            digest, file_prefix = _stable_anchored_file_hash(
                directory,
                name,
                metadata,
                cancel_requested=cancel_requested,
                deadline_monotonic=deadline_monotonic,
            )
            mime_type = _sniff_mime(file_prefix)
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
        directory.verify()

    visit(artifacts_anchor, (), 0)
    artifacts_anchor.verify()
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


def scan_legacy_run(
    directory: Path,
    *,
    directory_anchor: AnchoredDirectory | None = None,
) -> dict[str, Any] | None:
    """Index recognized immediate legacy JSONL files without traversing them."""

    directory = Path(os.path.abspath(os.path.expanduser(os.fspath(directory))))
    if directory_anchor is None:
        with open_anchored_directory(directory, directory) as anchored:
            return scan_legacy_run(directory, directory_anchor=anchored)
    directory_anchor.verify()
    if directory_anchor.directory != directory:
        raise HarvestStorageError("Legacy Harvest entry does not match its supplied anchor.")
    records: dict[str, Any] = {}
    for name in _LEGACY_FILES:
        if not directory_anchor.lexists(name):
            continue
        metadata = directory_anchor.lstat(name)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _metadata_is_link_or_reparse(metadata)
            or metadata.st_nlink != 1
            or metadata.st_size > _MAX_LEGACY_FILE_BYTES
        ):
            raise HarvestStorageError("Legacy Harvest JSONL evidence is unsafe or oversized.")
        digest, _prefix, line_count, valid_count = _stable_anchored_jsonl_summary(
            directory_anchor,
            name,
            metadata,
        )
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


def _stable_anchored_file_hash(
    parent: AnchoredDirectory,
    name: str,
    before: os.stat_result,
    *,
    cancel_requested: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> tuple[str, bytes]:
    descriptor = parent.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    digest = hashlib.sha256()
    prefix = b""
    try:
        opened = os.fstat(descriptor)
        _require_same_regular_file(before, opened, "Harvest artifact changed while it was being opened.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                _check_storage_abort(cancel_requested, deadline_monotonic)
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                if len(prefix) < 32:
                    prefix += chunk[: 32 - len(prefix)]
                digest.update(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = parent.lstat(name)
    _require_same_regular_file(before, opened_after, "Harvest artifact changed while it was being hashed.")
    _require_same_regular_file(before, path_after, "Harvest artifact path changed while it was being hashed.")
    return digest.hexdigest(), prefix


def _check_storage_abort(
    cancel_requested: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    if cancel_requested is not None and cancel_requested():
        raise HarvestStorageError("Harvest artifact verification was cancelled.")
    if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
        raise HarvestStorageError("Harvest artifact verification exceeded its duration limit.")


def _stable_anchored_jsonl_summary(
    parent: AnchoredDirectory,
    name: str,
    before: os.stat_result,
) -> tuple[str, bytes, int, int]:
    descriptor = parent.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    digest = hashlib.sha256()
    prefix = b""
    line_count = 0
    valid_count = 0
    try:
        opened = os.fstat(descriptor)
        _require_same_regular_file(before, opened, "Legacy Harvest evidence changed while opening.")
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
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = parent.lstat(name)
    _require_same_regular_file(before, opened_after, "Legacy Harvest evidence changed while hashing.")
    _require_same_regular_file(before, after, "Legacy Harvest evidence path changed while hashing.")
    return digest.hexdigest(), prefix, line_count, valid_count


def _sniff_mime(prefix: bytes) -> str:
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _open_owned_lock(parent: AnchoredDirectory, name: str) -> Any:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(8):
        before: os.stat_result | None = None
        handle: Any = None
        if parent.lexists(name):
            before = parent.lstat(name)
            _require_safe_regular(before, max_bytes=None, message="Harvest mutation lock is unsafe.")
            try:
                descriptor = parent.open_file(name, flags)
            except FileNotFoundError:
                continue
        else:
            try:
                descriptor = parent.open_file(name, flags | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            before = os.fstat(descriptor)
        try:
            opened = os.fstat(descriptor)
            if before is None:
                raise HarvestStorageError("Harvest mutation lock identity is unavailable.")
            _require_same_regular_file(before, opened, "Harvest mutation lock changed while opening.")
            path_after = parent.lstat(name)
            _require_same_regular_file(opened, path_after, "Harvest mutation lock path changed while opening.")
            handle = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = -1
            if opened.st_size == 0:
                written = handle.write(b"0")
                if written != 1:
                    raise HarvestStorageError("Harvest mutation lock initialization was incomplete.")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            _verify_open_lock(handle, parent, name)
            return handle
        except BaseException:
            if handle is not None:
                handle.close()
            if descriptor >= 0:
                os.close(descriptor)
            raise
    raise HarvestStorageError("Harvest mutation lock changed repeatedly while opening.")


def _verify_open_lock(handle: Any, parent: AnchoredDirectory, name: str) -> None:
    opened = os.fstat(handle.fileno())
    path_metadata = parent.lstat(name)
    _require_same_regular_file(opened, path_metadata, "Harvest mutation lock path changed while held.")


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


def _require_safe_regular(
    metadata: os.stat_result,
    *,
    max_bytes: int | None,
    message: str,
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _metadata_is_link_or_reparse(metadata)
        or metadata.st_nlink != 1
        or metadata.st_size < 0
        or (max_bytes is not None and metadata.st_size > max_bytes)
    ):
        raise HarvestStorageError(message)


def _require_same_regular_file(
    before: os.stat_result,
    after: os.stat_result,
    message: str,
    *,
    compare_size: bool = True,
) -> None:
    if (
        not stat.S_ISREG(after.st_mode)
        or _metadata_is_link_or_reparse(after)
        or after.st_nlink != 1
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or (compare_size and after.st_size != before.st_size)
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise HarvestStorageError(message)


def _require_same_directory(
    before: os.stat_result,
    after: os.stat_result,
    message: str,
    *,
    compare_times: bool = False,
) -> None:
    if (
        not stat.S_ISDIR(after.st_mode)
        or _metadata_is_link_or_reparse(after)
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or (compare_times and after.st_mtime_ns != before.st_mtime_ns)
    ):
        raise HarvestStorageError(message)


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
    "append_stable_single_link_bytes",
    "read_stable_single_link_bytes",
    "scan_artifacts",
    "scan_legacy_run",
    "write_atomic_stable_bytes",
    "write_exclusive_stable_bytes",
]
