"""Strict JSON primitives shared by durable product-event boundaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

_OMIT = object()


class StrictJSONError(ValueError):
    """Structured failure for a value that is not standards-compliant JSON."""

    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def to_dict(self) -> dict[str, str]:
        return {"error_code": self.code, "path": self.path, "message": str(self)}


class ProductEventValidationError(StrictJSONError):
    """A ProductEvent failed the finite, standards-compliant JSON contract."""


def validate_finite_json(value: Any, *, path: str = "$") -> None:
    """Reject non-finite numbers and values that cannot be represented as JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictJSONError("non_finite_number", path, f"Non-finite numeric value at {path}.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrictJSONError("invalid_json_key", path, f"JSON object key at {path} is not a string.")
            validate_finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_finite_json(item, path=f"{path}[{index}]")
        return
    raise StrictJSONError(
        "unsupported_json_value",
        path,
        f"Value at {path} cannot be represented as standards-compliant JSON.",
    )


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize only finite standards-compliant JSON."""

    validate_finite_json(value)
    options = dict(kwargs)
    options["allow_nan"] = False
    try:
        return json.dumps(value, **options)
    except (TypeError, ValueError) as exc:
        raise StrictJSONError("json_serialization_failed", "$", "Value cannot be serialized as strict JSON.") from exc


def strict_json_bytes(value: Any, **kwargs: Any) -> bytes:
    return strict_json_dumps(value, **kwargs).encode("utf-8")


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting Python's non-standard numeric constants."""

    def reject_constant(token: str) -> None:
        raise StrictJSONError("non_finite_number", "$", f"Non-standard JSON numeric constant {token!r}.")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except UnicodeDecodeError as exc:
        raise StrictJSONError("invalid_utf8", "$", "JSON payload is not valid UTF-8.") from exc
    validate_finite_json(value)
    return value


def finite_json_copy(value: Any) -> Any:
    """Copy a legacy/public payload while omitting values invalid under strict JSON."""

    def clean(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else _OMIT
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    continue
                cleaned = clean(child)
                if cleaned is not _OMIT:
                    result[key] = cleaned
            return result
        if isinstance(item, (list, tuple)):
            result = []
            for child in item:
                cleaned = clean(child)
                if cleaned is not _OMIT:
                    result.append(cleaned)
            return result
        return _OMIT

    cleaned = clean(value)
    return None if cleaned is _OMIT else cleaned


__all__ = [
    "ProductEventValidationError",
    "StrictJSONError",
    "finite_json_copy",
    "strict_json_bytes",
    "strict_json_dumps",
    "strict_json_loads",
    "validate_finite_json",
]
