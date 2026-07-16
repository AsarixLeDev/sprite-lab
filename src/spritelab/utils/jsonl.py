"""Canonical JSONL read/write helpers — single source of truth for all modules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path, *, validate: bool = False) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    With ``validate=False`` (default): silently returns ``[]`` for missing files
    and skips blank lines without validation.

    With ``validate=True``: raises ``ValueError`` on invalid JSON or non-dict rows
    with the file path and line number.
    """
    path = Path(path)
    if not validate:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        records.append(value)
    return records


def write_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> Path:
    """Write a sequence of dicts to a JSONL file, creating parent dirs as needed.

    ``sort_keys=True`` (default) matches all existing call sites.
    ``ensure_ascii=False`` (default) matches all existing call sites.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n")
    return path


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Lazily iterate over JSONL records without loading the entire file into memory."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
