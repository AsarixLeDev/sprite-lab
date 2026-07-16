"""Small safety and identity helpers shared by compute adapters."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,190}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value.startswith("/") or not SAFE_REMOTE_PATH.fullmatch(value) or ".." in path.parts:
        raise ValueError(
            "Remote paths must be absolute POSIX paths containing only letters, numbers, '.', '_', and '-'."
        )
    return str(path)


def validate_identifier(value: str, *, label: str = "identifier") -> str:
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {label}; use letters, numbers, '.', '_', and '-'.")
    return value


def redact(value: Any) -> Any:
    """Redact credential-looking fields before returning diagnostics."""

    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            result[key] = (
                "<redacted>"
                if any(mark in normalized for mark in ("token", "secret", "password", "key"))
                else redact(child)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value
