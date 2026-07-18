"""Strict JSON parsing for evaluation authority-bearing artifacts."""

from __future__ import annotations

import json
from typing import Any


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """Reject duplicate object keys and non-finite constants at every depth."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(payload, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
