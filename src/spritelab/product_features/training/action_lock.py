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

from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, UnsafeFilesystemOperation

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
        self._identity: OwnedFileIdentity | None = None
        self._authority_descriptor: int | None = None
        self._acquired = False

    def __enter__(self) -> TrainingActionLock:
        try:
            self._anchor = AnchoredDirectory(self.project_root, self.project_root)
            self._anchor.__enter__()
            self._handle, self._identity = _open_lock(self._anchor)
            self._authority_descriptor = (
                self._handle.fileno() if os.name == "nt" else _open_posix_directory_lock_authority(self._anchor)
            )
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    _lock_descriptor(self._authority_descriptor)
                    self._acquired = True
                    _verify_lock(self._handle, self._anchor, self._identity, allowed_sizes={1})
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
                if self._anchor is None or self._identity is None:
                    raise TrainingActionLockError("The activation/launch lock anchor is unavailable.")
                _verify_lock(self._handle, self._anchor, self._identity, allowed_sizes={1})
        finally:
            self._close(exc_type, exc_value, traceback)

    def _close(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            try:
                if self._acquired and self._authority_descriptor is not None:
                    _unlock_descriptor(self._authority_descriptor)
                    self._acquired = False
            finally:
                if (
                    self._authority_descriptor is not None
                    and self._handle is not None
                    and self._authority_descriptor != self._handle.fileno()
                ):
                    os.close(self._authority_descriptor)
                self._authority_descriptor = None
                if self._handle is not None:
                    self._handle.close()
                    self._handle = None
                self._identity = None
        finally:
            if self._anchor is not None:
                self._anchor.__exit__(exc_type, exc_value, traceback)
                self._anchor = None


def _open_lock(anchor: AnchoredDirectory) -> tuple[Any, OwnedFileIdentity]:
    flags = os.O_RDWR | os.O_CREAT | int(getattr(os, "O_BINARY", 0))
    descriptor = anchor.open_file_immovable(ACTION_LOCK_FILENAME, flags, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        opened = os.fstat(handle.fileno())
        identity = OwnedFileIdentity.from_stat(opened)
        _verify_lock(handle, anchor, identity, allowed_sizes={0, 1})
        if opened.st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        _verify_lock(handle, anchor, identity, allowed_sizes={1})
        return handle, identity
    except BaseException:
        handle.close()
        raise


def _safe_metadata(metadata: os.stat_result, *, allowed_sizes: set[int]) -> bool:
    reparse = getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not reparse
        and metadata.st_nlink == 1
        and metadata.st_size in allowed_sizes
    )


def _verify_lock(
    handle: Any,
    anchor: AnchoredDirectory,
    identity: OwnedFileIdentity,
    *,
    allowed_sizes: set[int],
) -> None:
    try:
        anchor.verify()
        opened = os.fstat(handle.fileno())
        current = anchor.lstat(ACTION_LOCK_FILENAME)
    except (OSError, UnsafeFilesystemOperation, ValueError) as exc:
        raise TrainingActionLockError("The activation/launch lock is unavailable.") from exc
    parent = anchor.directory_metadata()
    if (
        not _safe_metadata(opened, allowed_sizes=allowed_sizes)
        or not _safe_metadata(current, allowed_sizes=allowed_sizes)
        or opened.st_dev != parent.st_dev
        or current.st_dev != parent.st_dev
        or not identity.matches(opened)
        or not identity.matches(current)
    ):
        raise TrainingActionLockError("The activation/launch lock changed while open.")


def _open_posix_directory_lock_authority(anchor: AnchoredDirectory) -> int:
    descriptor = -1
    try:
        expected = OwnedFileIdentity.from_stat(anchor.directory_metadata())
        descriptor = os.open(
            anchor.fixed_directory_path(),
            os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_CLOEXEC", 0)),
        )
        opened = os.fstat(descriptor)
        anchor.verify()
        if not stat.S_ISDIR(opened.st_mode) or not expected.matches(opened):
            raise TrainingActionLockError("The activation/launch lock authority is unsafe.")
        return descriptor
    except (TrainingActionLockError, OSError, UnsafeFilesystemOperation, ValueError) as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(exc, TrainingActionLockError):
            raise
        raise TrainingActionLockError("The activation/launch lock authority is unsafe.") from exc


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


__all__ = [
    "ACTION_LOCK_FILENAME",
    "ACTION_LOCK_PROTOCOL",
    "ACTION_LOCK_PROTOCOL_IDENTITY",
    "TrainingActionLock",
    "TrainingActionLockError",
]
