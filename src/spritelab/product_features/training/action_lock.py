"""Shared interprocess exclusion for activation and Training launch claims.

Conditioned activation holds this lock from prospective configuration
validation through its single configuration compare-and-swap.  Training Start
and Resume acquire the same lock before reading activation/configuration state
and retain it until durable launch ownership has been claimed.  The fixed
repository-local file is persistent; lock release is handled by the kernel on
normal exit or process death.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Final

from spritelab.utils.safe_fs import AnchoredDirectory

ACTION_LOCK_FILENAME: Final = ".spritelab-activation-launch.lock"
ACTION_LOCK_PROTOCOL: Final = {
    "schema_version": "spritelab.training.activation-launch-lock.v1",
    "relative_path": ACTION_LOCK_FILENAME,
    "activation_scope": "prospective-validation-through-config-cas",
    "launch_scope": "activation-validation-through-durable-launch-claim",
    "process_death_releases": True,
    "paths_exposed": False,
}
ACTION_LOCK_PROTOCOL_IDENTITY: Final = hashlib.sha256(
    json.dumps(ACTION_LOCK_PROTOCOL, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
).hexdigest()


class TrainingActionLockError(RuntimeError):
    """The activation/launch exclusion lane is busy or unsafe."""


class TrainingActionLock(AbstractContextManager["TrainingActionLock"]):
    """Descriptor-bound repository-local activation/launch action lock."""

    def __init__(self, project_root: str | Path, *, timeout_seconds: float = 5.0) -> None:
        self.project_root = Path(project_root).resolve()
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self._anchor: AnchoredDirectory | None = None
        self._handle: Any = None

    def __enter__(self) -> TrainingActionLock:
        try:
            self._anchor = AnchoredDirectory(self.project_root, self.project_root)
            self._anchor.__enter__()
            self._handle = _open_lock(self._anchor)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    _lock_handle(self._handle)
                    _verify_lock(self._handle, self._anchor)
                    self._anchor.verify()
                    return self
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise TrainingActionLockError(
                            "Another activation, Start, or Resume action currently owns the launch boundary."
                        ) from None
                    time.sleep(0.01)
        except BaseException as exc:
            self._close(type(exc), exc, exc.__traceback__)
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self._handle is not None:
                if self._anchor is None:
                    raise TrainingActionLockError("The activation/launch lock anchor is unavailable.")
                _verify_lock(self._handle, self._anchor)
                _unlock_handle(self._handle)
        finally:
            self._close(exc_type, exc_value, traceback)

    def _close(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
        finally:
            if self._anchor is not None:
                self._anchor.__exit__(exc_type, exc_value, traceback)
                self._anchor = None


def _open_lock(anchor: AnchoredDirectory) -> Any:
    flags = os.O_RDWR | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = anchor.open_file(ACTION_LOCK_FILENAME, flags)
    except FileNotFoundError:
        try:
            descriptor = anchor.open_file(ACTION_LOCK_FILENAME, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            descriptor = anchor.open_file(ACTION_LOCK_FILENAME, flags)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        opened = os.fstat(handle.fileno())
        if opened.st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        _verify_lock(handle, anchor)
        return handle
    except BaseException:
        handle.close()
        raise


def _safe_metadata(metadata: os.stat_result) -> bool:
    reparse = getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not reparse
        and metadata.st_nlink == 1
        and metadata.st_size == 1
    )


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _safe_metadata(left)
        and _safe_metadata(right)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _verify_lock(handle: Any, anchor: AnchoredDirectory) -> None:
    try:
        opened = os.fstat(handle.fileno())
        current = anchor.lstat(ACTION_LOCK_FILENAME)
    except OSError as exc:
        raise TrainingActionLockError("The activation/launch lock is unavailable.") from exc
    if not _same_file(opened, current):
        raise TrainingActionLockError("The activation/launch lock changed while open.")


def _lock_handle(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "ACTION_LOCK_FILENAME",
    "ACTION_LOCK_PROTOCOL",
    "ACTION_LOCK_PROTOCOL_IDENTITY",
    "TrainingActionLock",
    "TrainingActionLockError",
]
