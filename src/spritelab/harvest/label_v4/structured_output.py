"""Bounded, provenance-preserving recovery for provider JSON objects."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructuredOutputRecovery:
    value: dict[str, Any] | None
    raw_response_hash: str
    parse_error: dict[str, Any] | None
    repair_actions: tuple[str, ...]
    repaired_json_hash: str | None
    schema_validation_result: dict[str, Any]

    @property
    def repaired(self) -> bool:
        return bool(self.repair_actions and self.value is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_response_hash": self.raw_response_hash,
            "parse_error": self.parse_error,
            "repair_actions": list(self.repair_actions),
            "repaired_json_hash": self.repaired_json_hash,
            "schema_validation_result": dict(self.schema_validation_result),
        }


def recover_json_object(
    raw_response: str,
    *,
    schema_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> StructuredOutputRecovery:
    """Apply the bounded v4 policy without any model call.

    Recovery is intentionally narrow: wrappers around exactly one object,
    trailing commas, Python-style booleans/nulls, and an incomplete terminal
    container may be repaired.  The last case is recorded explicitly and only
    removes the unterminated terminal member; it never invents its contents.
    """

    raw = str(raw_response)
    raw_hash = _hash(raw)
    first_error: dict[str, Any] | None = None
    candidates: list[tuple[str, tuple[str, ...]]] = [(raw.strip(), ())]

    extracted = _extract_exactly_one_object(raw)
    if extracted is not None and extracted != raw.strip():
        candidates.append((extracted, ("extract_single_json_object",)))

    for text, actions in list(candidates):
        parsed, error = _strict_parse(text)
        first_error = first_error or error
        if parsed is not None:
            return _validated(parsed, raw_hash, first_error, actions, schema_validator)

        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        if repaired != text:
            parsed, _ = _strict_parse(repaired)
            if parsed is not None:
                return _validated(parsed, raw_hash, first_error, (*actions, "remove_trailing_commas"), schema_validator)

        literals = re.sub(r"\bTrue\b", "true", repaired)
        literals = re.sub(r"\bFalse\b", "false", literals)
        literals = re.sub(r"\bNone\b", "null", literals)
        if literals != repaired:
            parsed, _ = _strict_parse(literals)
            if parsed is not None:
                return _validated(
                    parsed, raw_hash, first_error, (*actions, "normalize_json_literals"), schema_validator
                )

        closed = _close_incomplete_terminal_containers(literals)
        if closed is not None:
            repaired_text, close_actions = closed
            parsed, _ = _strict_parse(repaired_text)
            if parsed is not None:
                return _validated(parsed, raw_hash, first_error, actions + close_actions, schema_validator)

    return StructuredOutputRecovery(
        value=None,
        raw_response_hash=raw_hash,
        parse_error=first_error or {"error_type": "empty_response"},
        repair_actions=(),
        repaired_json_hash=None,
        schema_validation_result={"valid": False, "error": "json_unrecoverable"},
    )


def _strict_parse(text: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, {
            "error_type": "invalid_json",
            "message": exc.msg,
            "line": exc.lineno,
            "column": exc.colno,
            "position": exc.pos,
        }
    if not isinstance(value, dict):
        return None, {"error_type": "json_root_not_object"}
    return value, None


def _extract_exactly_one_object(text: str) -> str | None:
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    remainder = (text[:start] + text[index + 1 :]).strip()
                    if "{" not in remainder and "}" not in remainder:
                        return candidate
                    return None
    return None


def _close_incomplete_terminal_containers(text: str) -> tuple[str, tuple[str, ...]] | None:
    """Close only EOF truncation, dropping a wholly incomplete last member."""

    value = text.rstrip()
    actions: list[str] = []
    # The pilot response ends immediately after an array's new object opener.
    if re.search(r"\[\s*\{\s*$", value):
        value = re.sub(r"\{\s*$", "", value)
        actions.append("drop_incomplete_terminal_array_member")
    elif re.search(r",\s*$", value):
        value = re.sub(r",\s*$", "", value)
        actions.append("drop_incomplete_terminal_separator")

    stack: list[str] = []
    quoted = False
    escaped = False
    for char in value:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or (stack[-1], char) not in {("[", "]"), ("{", "}")}:
                return None
            stack.pop()
    if quoted or not stack:
        return None
    value += "".join("]" if char == "[" else "}" for char in reversed(stack))
    return value, (*actions, "close_unterminated_containers")


def _validated(
    value: dict[str, Any],
    raw_hash: str,
    parse_error: dict[str, Any] | None,
    actions: tuple[str, ...],
    validator: Callable[[Mapping[str, Any]], Any] | None,
) -> StructuredOutputRecovery:
    result: dict[str, Any] = {"valid": True}
    if validator is not None:
        try:
            validator(value)
        except (TypeError, ValueError, KeyError) as exc:
            result = {"valid": False, "error_type": type(exc).__name__, "message": str(exc)}
            value = None  # type: ignore[assignment]
    canonical = None if value is None else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return StructuredOutputRecovery(
        value=value,
        raw_response_hash=raw_hash,
        parse_error=parse_error,
        repair_actions=actions,
        repaired_json_hash=_hash(canonical) if canonical is not None and actions else None,
        schema_validation_result=result,
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
