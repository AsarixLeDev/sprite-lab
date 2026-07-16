"""Strict normalized-label schema and response validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from spritelab.product_features.providers.contracts import LabelState, NormalizedLabel
from spritelab.product_features.providers.errors import ProviderInvalidOutputError

LABEL_FIELDS = (
    "state",
    "domain",
    "category",
    "canonical_object",
    "role",
    "description",
    "confidence",
    "abstention_reasons",
    "provider_metadata",
)

NORMALIZED_LABEL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "state": {"type": "string", "enum": [state.value for state in LabelState]},
        "domain": {"type": ["string", "null"]},
        "category": {"type": ["string", "null"]},
        "canonical_object": {"type": ["string", "null"]},
        "role": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "abstention_reasons": {"type": "array", "items": {"type": "string"}},
        "provider_metadata": {"type": "object"},
    },
    "required": list(LABEL_FIELDS),
}

BATCH_LABEL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"image_id": {"type": "string"}, **NORMALIZED_LABEL_JSON_SCHEMA["properties"]},
                "required": ["image_id", *LABEL_FIELDS],
            },
        }
    },
    "required": ["results"],
}


def parse_json_object(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProviderInvalidOutputError("The provider returned malformed JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ProviderInvalidOutputError("The provider response root must be a JSON object.")
    return parsed


def validate_label(value: Mapping[str, Any]) -> NormalizedLabel:
    if set(value) != set(LABEL_FIELDS):
        raise ProviderInvalidOutputError("The provider label fields do not match the required schema.")
    try:
        state = LabelState(value["state"])
    except (ValueError, TypeError) as exc:
        raise ProviderInvalidOutputError("The provider returned an invalid label state.") from exc
    text_fields: dict[str, str | None] = {}
    for name in ("domain", "category", "canonical_object", "role", "description"):
        item = value[name]
        if item is not None and not isinstance(item, str):
            raise ProviderInvalidOutputError(f"The provider returned a non-string {name}.")
        if isinstance(item, str) and not item.strip():
            raise ProviderInvalidOutputError(f"The provider returned an empty {name}.")
        text_fields[name] = item
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ProviderInvalidOutputError("The provider returned an invalid confidence.")
    reasons = value["abstention_reasons"]
    if not isinstance(reasons, list) or not all(isinstance(reason, str) and reason.strip() for reason in reasons):
        raise ProviderInvalidOutputError("The provider returned invalid abstention reasons.")
    if state == LabelState.ABSTAINED and not reasons:
        raise ProviderInvalidOutputError("An abstained label must preserve at least one abstention reason.")
    metadata = value["provider_metadata"]
    if not isinstance(metadata, Mapping):
        raise ProviderInvalidOutputError("provider_metadata must be a JSON object.")
    return NormalizedLabel(
        state=state,
        domain=text_fields["domain"],
        category=text_fields["category"],
        canonical_object=text_fields["canonical_object"],
        role=text_fields["role"],
        description=text_fields["description"],
        confidence=float(confidence),
        abstention_reasons=tuple(reasons),
        provider_metadata=dict(metadata),
    )


def validate_batch_payload(value: Mapping[str, Any], image_ids: Sequence[str]) -> dict[str, NormalizedLabel]:
    if set(value) != {"results"} or not isinstance(value["results"], list):
        raise ProviderInvalidOutputError("The provider batch response must contain only a results array.")
    expected = set(image_ids)
    normalized: dict[str, NormalizedLabel] = {}
    for item in value["results"]:
        if not isinstance(item, Mapping) or set(item) != {"image_id", *LABEL_FIELDS}:
            raise ProviderInvalidOutputError("A provider result does not match the required schema.")
        image_id = item["image_id"]
        if not isinstance(image_id, str) or image_id not in expected or image_id in normalized:
            raise ProviderInvalidOutputError("The provider returned an unknown or duplicate image identity.")
        normalized[image_id] = validate_label({name: item[name] for name in LABEL_FIELDS})
    return normalized


def validate_batch_items(
    value: Mapping[str, Any], image_ids: Sequence[str]
) -> tuple[dict[str, NormalizedLabel], dict[str, str]]:
    """Validate items independently so one invalid item cannot erase valid siblings."""

    if set(value) != {"results"} or not isinstance(value["results"], list):
        raise ProviderInvalidOutputError("The provider batch response must contain only a results array.")
    expected = set(image_ids)
    normalized: dict[str, NormalizedLabel] = {}
    failures: dict[str, str] = {}
    for item in value["results"]:
        if not isinstance(item, Mapping):
            continue
        image_id = item.get("image_id")
        if not isinstance(image_id, str) or image_id not in expected or image_id in normalized or image_id in failures:
            continue
        try:
            if set(item) != {"image_id", *LABEL_FIELDS}:
                raise ProviderInvalidOutputError("A provider result does not match the required schema.")
            normalized[image_id] = validate_label({name: item[name] for name in LABEL_FIELDS})
        except (KeyError, ProviderInvalidOutputError) as exc:
            failures[image_id] = str(exc)
    return normalized, failures
