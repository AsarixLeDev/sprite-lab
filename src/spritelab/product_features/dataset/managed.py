"""Fail-closed validation and writes for managed product dataset outputs."""

from __future__ import annotations

import json
import os
import stat
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from spritelab.product_core.contracts import ProjectContext
from spritelab.utils.safe_fs import (
    UnsafeFilesystemOperation,
    atomic_write_bytes,
    atomic_write_text,
    require_confined_path,
)

INTAKE_SCHEMA = "spritelab.dataset.intake.v1"
QUEUE_SCHEMA = "spritelab.dataset.review_queue.v1"
REPORT_SCHEMA = "spritelab.dataset.report_data.v1"
RESULT_SCHEMA = "spritelab.product.result.v1"

_REQUIRED_DOCUMENTS = ("items.jsonl", "review_queue.json", "result.json", "report_data.json")
_MUTABLE_DOCUMENTS = (*_REQUIRED_DOCUMENTS, "review_log.jsonl")
_MAX_MANAGED_DOCUMENT_BYTES = 128 * 1024 * 1024
_WINDOWS_RESERVED = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class ManagedDatasetError(ValueError):
    """A selected dataset cannot be proven safe to mutate or reuse."""


class _ManagedDatasetMutationLock(AbstractContextManager["_ManagedDatasetMutationLock"]):
    """Persistent descriptor-backed lock; the lock path is never unlinked."""

    def __init__(self, root: Path, *, timeout_seconds: float = 1.0) -> None:
        self.root = root
        self.timeout_seconds = timeout_seconds
        self._handle: Any = None
        self._path: Path | None = None
        self._root_metadata: os.stat_result | None = None

    def __enter__(self) -> _ManagedDatasetMutationLock:
        self._root_metadata = self.root.lstat()
        _require_safe_directory(self.root)
        path = require_confined_path(self.root / ".spritelab-dataset-write.lock", self.root)
        self._handle = _open_persistent_lock(path)
        self._path = path
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                _lock_handle(self._handle)
                _verify_open_lock(self._handle, path)
                _require_same_directory(
                    self._root_metadata,
                    self.root.lstat(),
                    "The managed dataset directory changed while acquiring its write lock.",
                )
                return self
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise ManagedDatasetError("Another process holds the managed dataset write lock.") from None
                time.sleep(0.01)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        if self._handle is not None:
            try:
                assert self._path is not None
                _verify_open_lock(self._handle, self._path)
                _unlock_handle(self._handle)
            finally:
                self._handle.close()
                self._handle = None
                self._path = None
                self._root_metadata = None


def validate_managed_dataset_output(
    output_root: str | Path,
    *,
    context: ProjectContext | None = None,
    require_datasets_root: bool = False,
) -> Path:
    """Validate one completed intake dataset and return its lexical absolute root.

    A production context confines the output to the repository. Native selection
    is stricter and accepts only a direct descendant of ``datasets/``. Required
    writable documents must be ordinary single-link files, and their identities
    must agree before any later write is allowed.
    """

    value = os.fspath(output_root)
    if not value or not value.strip():
        raise ManagedDatasetError("The managed dataset path is empty.")
    lexical = Path(os.path.abspath(os.path.expanduser(value)))
    if context is not None:
        project_root = Path(os.path.abspath(context.project_root))
        boundary = project_root / "datasets" if require_datasets_root else project_root
        try:
            lexical = require_confined_path(lexical, boundary)
        except UnsafeFilesystemOperation as exc:
            scope = "the project datasets directory" if require_datasets_root else "the project directory"
            raise ManagedDatasetError(f"The selected dataset must stay inside {scope}.") from exc
    _require_safe_directory(lexical)

    for name in _REQUIRED_DOCUMENTS:
        _require_safe_regular_file(lexical / name, label=name)
    log_path = lexical / "review_log.jsonl"
    if os.path.lexists(log_path):
        _require_safe_regular_file(log_path, label="review_log.jsonl")

    result = _read_json(lexical / "result.json", label="result.json")
    report = _read_json(lexical / "report_data.json", label="report_data.json")
    queue = _read_json(lexical / "review_queue.json", label="review_queue.json")
    items = _read_jsonl(lexical / "items.jsonl")
    _validate_documents(lexical, result=result, report=report, queue=queue, items=items)
    return lexical


@contextmanager
def dataset_write_guard(
    output_root: str | Path,
    *,
    context: ProjectContext | None = None,
) -> Iterator[Path]:
    """Serialize one managed-dataset mutation and revalidate both boundaries."""

    output = validate_managed_dataset_output(output_root, context=context)
    with _ManagedDatasetMutationLock(output):
        output = validate_managed_dataset_output(output, context=context)
        yield output
        validate_managed_dataset_output(output, context=context)


def atomic_write_json_document(path: str | Path, value: Mapping[str, Any], *, output_root: str | Path) -> Path:
    """Atomically replace one confined JSON document with strict JSON bytes."""

    output = _validated_output_root(output_root)
    target = require_confined_path(path, output)
    return atomic_write_text(
        target,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )


def atomic_write_jsonl_document(
    path: str | Path,
    values: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
) -> Path:
    """Atomically replace one confined canonical JSONL document."""

    output = _validated_output_root(output_root)
    target = require_confined_path(path, output)
    payload = "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
        for value in values
    )
    return atomic_write_text(target, payload)


def atomic_append_jsonl_event(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> Path:
    """Append one event through an atomic prefix-preserving replacement."""

    output = _validated_output_root(output_root)
    target = require_confined_path(path, output)
    if os.path.lexists(target):
        _require_safe_regular_file(target, label=target.name)
        prefix = _read_stable_single_link_bytes(target, output, max_bytes=_MAX_MANAGED_DOCUMENT_BYTES)
        if prefix and not prefix.endswith(b"\n"):
            raise ManagedDatasetError("The append-only dataset review log has an incomplete final record.")
    else:
        prefix = b""
    row = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    return atomic_write_bytes(target, prefix + row)


def _validate_documents(
    output: Path,
    *,
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    queue: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ManagedDatasetError("The selected dataset result has an unsupported schema.")
    data = result.get("data")
    if not isinstance(data, Mapping) or data.get("schema_version") != INTAKE_SCHEMA:
        raise ManagedDatasetError("The selected dataset result is not a completed intake result.")
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ManagedDatasetError("The selected dataset report has an unsupported schema.")
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or summary.get("schema_version") != INTAKE_SCHEMA:
        raise ManagedDatasetError("The selected dataset report is incomplete.")
    if queue.get("schema_version") != QUEUE_SCHEMA or not isinstance(queue.get("items"), list):
        raise ManagedDatasetError("The selected dataset review queue has an unsupported schema.")
    if not _stored_path_matches(data.get("output_root"), output) or not _stored_path_matches(
        queue.get("output_root"), output
    ):
        raise ManagedDatasetError("The selected dataset output identity does not match its directory.")
    input_root = queue.get("input_root")
    if (
        not isinstance(input_root, str)
        or not input_root.strip()
        or not _stored_path_matches(data.get("input_root"), input_root)
    ):
        raise ManagedDatasetError("The selected dataset source identity is incomplete or inconsistent.")
    expected_log = output / "review_log.jsonl"
    if not _stored_path_matches(queue.get("append_only_log"), expected_log):
        raise ManagedDatasetError("The selected dataset review-log identity is inconsistent.")
    if data.get("counts") != summary.get("counts"):
        raise ManagedDatasetError("The selected dataset result and report counts disagree.")

    item_ids: set[str] = set()
    relative_identities: set[str] = set()
    for item in items:
        if item.get("schema_version") != INTAKE_SCHEMA:
            raise ManagedDatasetError("The selected dataset item manifest has an unsupported schema.")
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in item_ids:
            raise ManagedDatasetError("The selected dataset item identities are missing or duplicated.")
        item_ids.add(item_id)
        relative = _safe_relative_path(item.get("relative_path"))
        identity = unicodedata.normalize("NFC", relative).casefold()
        if identity in relative_identities:
            raise ManagedDatasetError("The selected dataset contains a case or Unicode-normalization path collision.")
        relative_identities.add(identity)
    for queued in queue["items"]:
        if not isinstance(queued, Mapping) or str(queued.get("item_id") or "") not in item_ids:
            raise ManagedDatasetError("The selected dataset review queue references an unknown item.")


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ManagedDatasetError("The selected dataset contains an invalid relative image path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ManagedDatasetError("The selected dataset contains an unsafe relative image path.")
    for part in relative.parts:
        stem = part.rstrip(" .").split(".", 1)[0].casefold()
        if stem in _WINDOWS_RESERVED or part.endswith((" ", ".")):
            raise ManagedDatasetError("The selected dataset contains a platform-reserved image path.")
    return relative.as_posix()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_stable_single_link_bytes(
                path,
                path.parent,
                max_bytes=_MAX_MANAGED_DOCUMENT_BYTES,
            ).decode("utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManagedDatasetError(f"The selected dataset {label} is unreadable.") from exc
    if not isinstance(value, Mapping):
        raise ManagedDatasetError(f"The selected dataset {label} is malformed.")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        payload = _read_stable_single_link_bytes(
            path,
            path.parent,
            max_bytes=_MAX_MANAGED_DOCUMENT_BYTES,
        )
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ManagedDatasetError("The selected dataset item manifest contains a non-object row.")
            rows.append(dict(value))
    except ManagedDatasetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManagedDatasetError("The selected dataset item manifest is unreadable.") from exc
    if not rows:
        raise ManagedDatasetError("The selected dataset item manifest is empty.")
    return rows


def _require_safe_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ManagedDatasetError("The selected dataset directory is unavailable.") from exc
    if _is_link_or_reparse(path, metadata) or not stat.S_ISDIR(metadata.st_mode) or path.is_mount():
        raise ManagedDatasetError("The selected dataset directory is linked, mounted, or not a regular directory.")


def _require_safe_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ManagedDatasetError(f"The selected dataset is missing its required {label} document.") from exc
    if _is_link_or_reparse(path, metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ManagedDatasetError(f"The selected dataset {label} document is linked or not a regular file.")
    if int(getattr(metadata, "st_nlink", 1)) != 1:
        raise ManagedDatasetError(f"The selected dataset {label} document has an unsafe hard-link count.")
    return metadata


def _is_link_or_reparse(path: Path, metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(attributes & reparse)
        or bool(getattr(path, "is_junction", lambda: False)())
    )


def _read_stable_single_link_bytes(path: Path, root: Path, *, max_bytes: int) -> bytes:
    """Read a confined ordinary file through a stable no-follow descriptor."""

    path = require_confined_path(path, root)
    root_before = root.lstat()
    parent_before = path.parent.lstat()
    before = _require_safe_regular_file(path, label=path.name)
    if before.st_size < 0 or before.st_size > max_bytes:
        raise ManagedDatasetError(f"The selected dataset {path.name} document exceeds its byte limit.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManagedDatasetError(f"The selected dataset {path.name} document is unreadable.") from exc
    try:
        opened = os.fstat(descriptor)
        _require_same_regular_file(before, opened, "A managed dataset document changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    _require_same_regular_file(before, opened_after, "A managed dataset document changed while reading.")
    _require_same_regular_file(before, after, "A managed dataset path changed while reading.")
    _require_same_directory(parent_before, path.parent.lstat(), "A managed dataset directory changed while reading.")
    _require_same_directory(root_before, root.lstat(), "A managed dataset root changed while reading.")
    if len(payload) != before.st_size or len(payload) > max_bytes:
        raise ManagedDatasetError("A managed dataset document changed size while reading.")
    return payload


def _open_persistent_lock(path: Path) -> Any:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(8):
        before: os.stat_result | None = None
        handle: Any = None
        if os.path.lexists(path):
            before = _require_safe_regular_file(path, label=path.name)
            try:
                descriptor = os.open(path, flags)
            except FileNotFoundError:
                continue
        else:
            try:
                descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            before = os.fstat(descriptor)
        try:
            opened = os.fstat(descriptor)
            assert before is not None
            _require_same_regular_file(before, opened, "The managed dataset lock changed while opening.")
            _require_same_regular_file(opened, path.lstat(), "The managed dataset lock path changed while opening.")
            handle = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = -1
            if opened.st_size == 0:
                if handle.write(b"0") != 1:
                    raise ManagedDatasetError("The managed dataset lock could not be initialized.")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            _verify_open_lock(handle, path)
            return handle
        except BaseException:
            if handle is not None:
                handle.close()
            if descriptor >= 0:
                os.close(descriptor)
            raise
    raise ManagedDatasetError("The managed dataset lock changed repeatedly while opening.")


def _verify_open_lock(handle: Any, path: Path) -> None:
    _require_same_regular_file(
        os.fstat(handle.fileno()),
        path.lstat(),
        "The managed dataset lock path changed while held.",
    )


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


def _require_same_regular_file(before: os.stat_result, after: os.stat_result, message: str) -> None:
    if (
        not stat.S_ISREG(after.st_mode)
        or _is_metadata_link_or_reparse(after)
        or int(getattr(after, "st_nlink", 1)) != 1
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ManagedDatasetError(message)


def _require_same_directory(before: os.stat_result, after: os.stat_result, message: str) -> None:
    if (
        not stat.S_ISDIR(after.st_mode)
        or _is_metadata_link_or_reparse(after)
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
    ):
        raise ManagedDatasetError(message)


def _is_metadata_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _stored_path_matches(value: Any, expected: str | Path) -> bool:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
        return False
    left = os.path.normcase(os.path.abspath(os.path.expanduser(os.fspath(value))))
    right = os.path.normcase(os.path.abspath(os.path.expanduser(os.fspath(expected))))
    return left == right


def _validated_output_root(output_root: str | Path) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_root))))
    _require_safe_directory(output)
    return output


__all__ = [
    "ManagedDatasetError",
    "atomic_append_jsonl_event",
    "atomic_write_json_document",
    "atomic_write_jsonl_document",
    "dataset_write_guard",
    "validate_managed_dataset_output",
]
