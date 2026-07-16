"""Fail-closed helpers for destructive and replacement filesystem operations."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path


class UnsafeFilesystemOperation(ValueError):
    """Raised when a filesystem mutation cannot prove its target is confined."""


def require_confined_path(
    path: str | Path,
    root: str | Path,
    *,
    allow_root: bool = False,
) -> Path:
    """Return a lexical absolute path only when it is safely below ``root``.

    Both lexical and resolved containment are checked. Existing descendant
    components may not be symbolic links or Windows reparse points. The root is
    treated as the caller-approved boundary and may itself resolve elsewhere.
    """

    root_path = _absolute(root)
    target = _absolute(path)
    try:
        relative = target.relative_to(root_path)
    except ValueError as exc:
        raise UnsafeFilesystemOperation(f"target escapes its approved root: {target}") from exc
    if not relative.parts and not allow_root:
        raise UnsafeFilesystemOperation(f"refusing to mutate the approved root itself: {root_path}")

    resolved_root = root_path.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    try:
        resolved_relative = resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafeFilesystemOperation(f"resolved target escapes its approved root: {target}") from exc
    if not resolved_relative.parts and not allow_root:
        raise UnsafeFilesystemOperation(f"refusing to mutate the approved root itself: {root_path}")

    current = root_path
    for part in relative.parts:
        current = current / part
        if not _lexists(current):
            break
        if _is_link_or_reparse_point(current):
            raise UnsafeFilesystemOperation(f"target crosses a link or reparse point: {current}")
    return target


def remove_confined_tree(path: str | Path, root: str | Path, *, missing_ok: bool = False) -> None:
    """Recursively remove one verified directory strictly below ``root``."""

    target = require_confined_path(path, root)
    if not _lexists(target):
        if missing_ok:
            return
        raise FileNotFoundError(target)
    if _is_link_or_reparse_point(target):
        raise UnsafeFilesystemOperation(f"refusing to recursively remove a linked path: {target}")
    if target.is_mount():
        raise UnsafeFilesystemOperation(f"refusing to recursively remove a mount point: {target}")
    if not target.is_dir():
        raise UnsafeFilesystemOperation(f"recursive removal requires a directory: {target}")
    shutil.rmtree(target)


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Atomically replace one file via an exclusive unpredictable sibling."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Encode and atomically replace one text file."""

    return atomic_write_bytes(path, content.encode(encoding))


def _absolute(path: str | Path) -> Path:
    value = os.fspath(path)
    if not value or not value.strip():
        raise UnsafeFilesystemOperation("filesystem target must not be empty")
    return Path(os.path.abspath(os.path.expanduser(value)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


__all__ = [
    "UnsafeFilesystemOperation",
    "atomic_write_bytes",
    "atomic_write_text",
    "remove_confined_tree",
    "require_confined_path",
]
