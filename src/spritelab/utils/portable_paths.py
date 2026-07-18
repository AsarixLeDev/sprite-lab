"""One strict cross-platform grammar for persisted relative artifact paths."""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath, PureWindowsPath

_WINDOWS_RESERVED = re.compile(r"(?i)^(?:con|prn|aux|nul|clock\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?$")
_WINDOWS_FORBIDDEN = frozenset('<>:"|?*')


def canonical_portable_relative_path(value: str) -> str:
    """Return ``value`` only when every platform can preserve it exactly."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("A portable relative path must be a non-empty exact string.")
    if value != unicodedata.normalize("NFC", value) or "\\" in value:
        raise ValueError("A portable relative path must use NFC and POSIX separators.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("A portable relative path cannot contain control characters.")
    if any(character in _WINDOWS_FORBIDDEN for character in value):
        raise ValueError("A portable relative path contains a Windows-forbidden character.")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or posix.as_posix() != value
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or bool(windows.anchor)
    ):
        raise ValueError("A portable relative path cannot be absolute, drive-relative, UNC, or device-qualified.")
    parts = value.split("/")
    if not parts:
        raise ValueError("A portable relative path is empty.")
    for part in parts:
        if part in {"", ".", ".."} or part != part.rstrip(". ") or _WINDOWS_RESERVED.fullmatch(part) is not None:
            raise ValueError("A portable relative path contains an unsafe component.")
    return value


def is_portable_relative_path(value: object) -> bool:
    """Return whether ``value`` satisfies :func:`canonical_portable_relative_path`."""

    if not isinstance(value, str):
        return False
    try:
        canonical_portable_relative_path(value)
    except ValueError:
        return False
    return True


def portable_path_collision_key(value: str) -> str:
    """Return the NFC/casefold key used to reject cross-platform collisions."""

    return unicodedata.normalize("NFC", canonical_portable_relative_path(value)).casefold()


__all__ = [
    "canonical_portable_relative_path",
    "is_portable_relative_path",
    "portable_path_collision_key",
]
