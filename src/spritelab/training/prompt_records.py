"""Shared prompt records reader (migrated from sample_generator.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_prompt_records(path: str | Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    """Read eval prompt JSONL while preserving metadata fields."""

    path = Path(path)
    if max_records is not None and int(max_records) <= 0:
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if isinstance(value, str):
            record = {"prompt": value, "prompt_id": f"prompt_{line_no:04d}"}
        elif isinstance(value, dict):
            record = dict(value)
            record["prompt"] = str(record.get("prompt") or record.get("caption") or "")
            record.setdefault("prompt_id", f"prompt_{line_no:04d}")
        else:
            raise ValueError(f"{path}:{line_no}: expected JSON object or string")
        if not str(record.get("prompt", "")).strip():
            raise ValueError(f"{path}:{line_no}: prompt is empty")
        records.append(record)
        if max_records is not None and len(records) >= int(max_records):
            break
    return records
