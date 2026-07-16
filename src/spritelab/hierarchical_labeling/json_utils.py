"""Strict JSON, validation, and content-identity helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ContractValidationError(ValueError):
    """A versioned labeling record failed closed during validation."""


def strict_json_value(value: Any, *, path: str = "$") -> Any:
    """Return a JSON-compatible projection and reject non-finite/opaque values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Enum):
        return strict_json_value(value.value, path=path)
    if isinstance(value, StrictRecord):
        return value.to_dict()
    if is_dataclass(value):
        return {
            item.name: strict_json_value(getattr(value, item.name), path=f"{path}.{item.name}")
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ContractValidationError(f"{path} contains a non-string or empty object key")
            result[key] = strict_json_value(child, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [strict_json_value(child, path=f"{path}[{index}]") for index, child in enumerate(value)]
    raise ContractValidationError(f"{path} contains unsupported value type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        strict_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_identity(kind: str, value: Any) -> str:
    if not kind or "\x00" in kind:
        raise ContractValidationError("identity kind must be a non-empty text value")
    digest = hashlib.sha256()
    digest.update(b"spritelab-hierarchical-labeling-v1\0")
    digest.update(kind.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def require_text(value: Any, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractValidationError(f"{name} must be non-empty text without surrounding whitespace")
    if identifier and not IDENTIFIER_PATTERN.fullmatch(value):
        raise ContractValidationError(f"{name} must be a controlled identifier")
    return value


def require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ContractValidationError(f"{name} must be a full lowercase SHA-256")
    return value


def require_probability(value: Any, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractValidationError(f"{name} must be from 0 through 1")
    return result


def require_finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractValidationError(f"{name} is outside its valid range")
    return result


def require_unique_text(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or (not allow_empty and not value.strip()) or value != value.strip():
            raise ContractValidationError(f"{name}[{index}] must be normalized text")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ContractValidationError(f"{name} cannot contain duplicates")
    return tuple(normalized)


class StrictRecord:
    """Mixin for frozen schema-versioned, serializable, content-hashable records."""

    SCHEMA_VERSION: ClassVar[str]
    IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = ()

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("StrictRecord subclasses must be dataclasses")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            **{item.name: strict_json_value(getattr(self, item.name), path=f"$.{item.name}") for item in fields(self)},
        }
        strict_json_value(payload)
        return payload

    @property
    def identity(self) -> str:
        return content_identity(self.SCHEMA_VERSION, self.to_dict())

    def validate_record(self) -> None:
        require_text(self.SCHEMA_VERSION, "schema version")
        for name in self.IDENTITY_FIELDS:
            value = getattr(self, name, None)
            if value in (None, "", (), []):
                raise ContractValidationError(f"{type(self).__name__}.{name} is required for identity binding")
        strict_json_value(self.to_dict())

    def __hash__(self) -> int:
        return int(self.identity[:16], 16)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and isinstance(other, StrictRecord) and self.to_dict() == other.to_dict()
