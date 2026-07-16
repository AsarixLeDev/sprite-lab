"""Explicit, risk-penalized adapters for pre-v4 labeling artifacts.

Legacy values remain readable, but this module makes it structurally impossible
to confuse them with a new deterministic extraction or a fresh model call.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

LEGACY_ADAPTER_SCHEMA_VERSION = "label_legacy_adapter_v4.1"
LEGACY_FIELDS = (
    "domain",
    "category",
    "canonical_object",
    "surface_alias",
    "color",
    "material",
    "shape",
    "role",
    "description",
)

DEFAULT_V3_RISK_PENALTY = 4
DEFAULT_V2_RISK_PENALTY = 6
DEFAULT_V3_UNCERTAINTY_FLOOR = 13
DEFAULT_V2_UNCERTAINTY_FLOOR = 15


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _artifact_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _original_policy_hash(record: Mapping[str, Any]) -> str:
    for value in (
        record.get("policy_hash"),
        record.get("config_hash"),
        _nested(record, "lineage", "policy_hash"),
        _nested(record, "metadata", "policy_hash"),
    ):
        if value:
            return str(value)
    return ""


def _original_model_identity(record: Mapping[str, Any], override: str) -> str:
    if override:
        return override
    for value in (
        record.get("model_identity"),
        record.get("model"),
        _nested(record, "vlm", "model_identity"),
        _nested(record, "lineage", "model_identity"),
        _nested(record, "metadata", "model_identity"),
    ):
        if value:
            return str(value)
    return "unknown_legacy_model"


def _field_value_v3(record: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    prefills = record.get("prefills")
    prefill = prefills.get(name) if isinstance(prefills, Mapping) else None
    decision = record.get(name)
    if isinstance(prefill, Mapping):
        raw_candidates = copy.deepcopy(list(prefill.get("raw_candidates") or ()))
        raw_values = [item.get("value") for item in raw_candidates if isinstance(item, Mapping) and "value" in item]
        value = copy.deepcopy(prefill.get("value"))
        normalized = copy.deepcopy(prefill.get("normalized_value", value))
        return {
            "value": value,
            "raw_open_vocabulary_value": raw_values[0] if raw_values else value,
            "raw_candidates": raw_candidates,
            "normalized_controlled_value": normalized,
            "legacy_state": str(prefill.get("open_set_state", "unknown")),
            "legacy_warnings": list(prefill.get("warnings") or ()),
        }
    if isinstance(decision, Mapping):
        value = copy.deepcopy(decision.get("accepted_value"))
        if value is None:
            candidates = decision.get("candidates") or ()
            value = copy.deepcopy(candidates[0]) if candidates else None
        if value is None and not decision.get("candidates"):
            return None
        return {
            "value": value,
            "raw_open_vocabulary_value": value,
            "raw_candidates": copy.deepcopy(list(decision.get("candidates") or ())),
            "normalized_controlled_value": value,
            "legacy_state": str(decision.get("state", "unknown")),
            "legacy_warnings": [],
        }
    if decision is not None:
        return {
            "value": copy.deepcopy(decision),
            "raw_open_vocabulary_value": copy.deepcopy(decision),
            "raw_candidates": [],
            "normalized_controlled_value": copy.deepcopy(decision),
            "legacy_state": "unknown",
            "legacy_warnings": [],
        }
    return None


def _field_value_v2(record: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    for container_name in ("fused_label", "label", "fields", "proposal", "suggestion"):
        container = record.get(container_name)
        if isinstance(container, Mapping):
            candidates.append(container)
    candidates.append(record)
    aliases = {
        "canonical_object": ("canonical_object", "object_name", "object"),
        "surface_alias": ("surface_alias", "specific_name", "name"),
    }
    keys = aliases.get(name, (name,))
    value: Any = None
    found = False
    for container in candidates:
        for key in keys:
            if key in container:
                value = copy.deepcopy(container[key])
                found = True
                break
        if found:
            break
    if not found:
        return None
    return {
        "value": value,
        "raw_open_vocabulary_value": copy.deepcopy(value),
        "raw_candidates": [],
        "normalized_controlled_value": copy.deepcopy(value),
        "legacy_state": str(record.get("status", "unknown")),
        "legacy_warnings": [],
    }


def adapt_legacy_artifact(
    record: Mapping[str, Any],
    legacy_version: str,
    *,
    risk_penalty: int | None = None,
    uncertainty_floor_1_20: int | None = None,
    original_model_identity: str = "",
) -> dict[str, Any]:
    """Adapt v2/v3 without granting the artifact fresh-evidence status."""

    version = str(legacy_version).strip().lower().removeprefix("labeling_")
    if version not in {"v2", "v3"}:
        raise ValueError("legacy_version must be v2 or v3")
    penalty = (
        risk_penalty
        if risk_penalty is not None
        else (DEFAULT_V3_RISK_PENALTY if version == "v3" else DEFAULT_V2_RISK_PENALTY)
    )
    floor = (
        uncertainty_floor_1_20
        if uncertainty_floor_1_20 is not None
        else (DEFAULT_V3_UNCERTAINTY_FLOOR if version == "v3" else DEFAULT_V2_UNCERTAINTY_FLOOR)
    )
    if not 0 <= int(penalty) <= 20:
        raise ValueError("risk_penalty must be in [0, 20]")
    if not 1 <= int(floor) <= 20:
        raise ValueError("uncertainty_floor_1_20 must be in [1, 20]")

    extractor = _field_value_v3 if version == "v3" else _field_value_v2
    adapted_fields: dict[str, Any] = {}
    for name in LEGACY_FIELDS:
        extracted = extractor(record, name)
        if extracted is None:
            continue
        adapted_fields[name] = {
            **extracted,
            "legacy_source": True,
            "uncalibrated": True,
            "calibration_state": "uncalibrated",
            "risk_penalty": int(penalty),
            "uncertainty_floor_1_20": int(floor),
            "fresh_model_evidence": False,
            "eligible_for_independent_support": False,
        }

    raw_copy = copy.deepcopy(dict(record))
    result = {
        "schema_version": LEGACY_ADAPTER_SCHEMA_VERSION,
        "sprite_id": str(record.get("sprite_id", "")),
        "legacy_version": version,
        "legacy_source": True,
        "uncalibrated": True,
        "calibration_state": "uncalibrated",
        "risk_penalty": int(penalty),
        "uncertainty_floor_1_20": int(floor),
        "original_policy_hash": _original_policy_hash(record),
        "original_model_identity": _original_model_identity(record, original_model_identity),
        "original_schema_version": str(record.get("schema_version", version)),
        "raw_artifact_hash": _artifact_hash(record),
        "raw_legacy_artifact": raw_copy,
        "evidence_family": f"legacy_{version}_adapter",
        "dependency_group": f"legacy:{version}",
        "fresh_model_evidence": False,
        "independent_evidence": False,
        "eligible_for_independent_support": False,
        "cache_namespace": f"legacy_{version}_isolated",
        "adapted_fields": adapted_fields,
        "risk_signals": ["legacy_adapter_use", "uncalibrated", "original_model_not_reexecuted"],
    }
    validate_legacy_isolation(result)
    return result


def adapt_v3_artifact(record: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    return adapt_legacy_artifact(record, "v3", **kwargs)


def adapt_v2_artifact(record: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    return adapt_legacy_artifact(record, "v2", **kwargs)


def validate_legacy_isolation(value: Mapping[str, Any]) -> None:
    """Reject any adapter payload that could masquerade as fresh evidence."""

    required = {
        "legacy_source": True,
        "uncalibrated": True,
        "fresh_model_evidence": False,
        "independent_evidence": False,
        "eligible_for_independent_support": False,
    }
    for key, expected in required.items():
        if value.get(key) is not expected:
            raise ValueError(f"legacy adapter isolation flag invalid: {key}")
    if value.get("calibration_state") != "uncalibrated":
        raise ValueError("legacy adapter must remain uncalibrated")
    if not str(value.get("evidence_family", "")).startswith("legacy_"):
        raise ValueError("legacy adapter needs a legacy evidence family")
    if not str(value.get("cache_namespace", "")).startswith("legacy_"):
        raise ValueError("legacy adapter needs an isolated cache namespace")


def detect_and_adapt_legacy(record: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    schema = str(record.get("schema_version", "")).lower()
    if "v3" in schema or "prefills" in record:
        return adapt_v3_artifact(record, **kwargs)
    return adapt_v2_artifact(record, **kwargs)
