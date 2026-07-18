"""Canonical compatibility identity for evaluation metric definitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


class IncompatibleMetricDefinitions(ValueError):
    """Raised before incomplete or incompatible metrics could be compared."""


_LEGACY_DEFINITION_FIELDS = (
    "schema_version",
    "thresholds",
    "detector_policy_version",
    "detector_policy_sha256",
    "comparison_method",
    "comparison_parameters_sha256",
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IncompatibleMetricDefinitions("Metric definitions are not canonical finite JSON.") from error


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.strip()
        and all(character in "0123456789abcdef" for character in value)
    )


def _computed_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_embedded_policy_hashes(report: Mapping[str, Any]) -> None:
    comparison_parameters = report.get("comparison_parameters")
    if comparison_parameters is not None:
        if not isinstance(comparison_parameters, Mapping) or not comparison_parameters:
            raise IncompatibleMetricDefinitions("Embedded comparison parameters are malformed.")
        declared = report.get("comparison_parameters_sha256")
        if not _valid_sha256(declared) or _computed_identity(comparison_parameters) != declared:
            raise IncompatibleMetricDefinitions("Comparison-parameter SHA-256 does not agree with its fields.")

    detector_policy = report.get("detector_policy")
    if detector_policy is not None:
        if not isinstance(detector_policy, Mapping):
            raise IncompatibleMetricDefinitions("Embedded detector policy is malformed.")
        policy_payload = {
            field: detector_policy.get(field)
            for field in (
                "detector_policy_version",
                "comparison_method",
                "thresholds",
                "diagnostic_semantics",
            )
        }
        declared = report.get("detector_policy_sha256")
        embedded_declared = detector_policy.get("detector_policy_sha256")
        if (
            not _valid_sha256(declared)
            or embedded_declared != declared
            or _computed_identity(policy_payload) != declared
        ):
            raise IncompatibleMetricDefinitions("Detector-policy SHA-256 does not agree with its fields.")


def metric_definition_identity(report: Mapping[str, Any]) -> str:
    """Return a complete identity or reject incomplete/hash-inconsistent definitions.

    Newer reports may publish an explicit ``metric_definitions`` mapping.  The
    redacted product projection may replace that mapping with its verified
    SHA-256.  Legacy generation-benchmark reports instead bind their complete
    threshold and memorization-detector policy tuple.
    """

    _validate_embedded_policy_hashes(report)
    explicit = report.get("metric_definitions")
    public_identity = report.get("metric_definitions_sha256")
    if explicit is not None:
        if not isinstance(explicit, Mapping) or not explicit:
            raise IncompatibleMetricDefinitions("Explicit metric definitions are missing or malformed.")
        schema_version = report.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version or schema_version != schema_version.strip():
            raise IncompatibleMetricDefinitions("Explicit metric-definition schema is missing or malformed.")
        payload: dict[str, Any] = {
            "schema_version": schema_version,
            "metric_definitions": explicit,
        }
        # Explicit reports may also carry detector policy.  If they do, it is
        # comparison-relevant and cannot be hidden behind the same metric hash.
        for field in _LEGACY_DEFINITION_FIELDS[1:]:
            if field in report:
                payload[field] = report[field]
        identity = _computed_identity(payload)
        if public_identity is not None and (not _valid_sha256(public_identity) or public_identity != identity):
            raise IncompatibleMetricDefinitions("Explicit metric-definition SHA-256 does not agree with its fields.")
        return identity

    legacy_present = any(field in report for field in _LEGACY_DEFINITION_FIELDS[1:])
    if legacy_present:
        schema_version = report.get("schema_version")
        thresholds = report.get("thresholds")
        if not isinstance(schema_version, str) or not schema_version or schema_version != schema_version.strip():
            raise IncompatibleMetricDefinitions("Metric-definition schema version is missing or malformed.")
        if not isinstance(thresholds, Mapping) or not thresholds:
            raise IncompatibleMetricDefinitions("Metric thresholds are missing or malformed.")
        for field in ("detector_policy_version", "comparison_method"):
            value = report.get(field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise IncompatibleMetricDefinitions(f"{field} is missing or malformed.")
        for field in ("detector_policy_sha256", "comparison_parameters_sha256"):
            if not _valid_sha256(report.get(field)):
                raise IncompatibleMetricDefinitions(f"{field} is missing or malformed.")
        payload = {field: report[field] for field in _LEGACY_DEFINITION_FIELDS}
        identity = _computed_identity(payload)
        if public_identity is not None and (not _valid_sha256(public_identity) or public_identity != identity):
            raise IncompatibleMetricDefinitions("Metric-definition SHA-256 does not agree with its fields.")
        return identity

    if _valid_sha256(public_identity):
        return public_identity
    raise IncompatibleMetricDefinitions("Evaluation report has no complete metric-definition identity.")
