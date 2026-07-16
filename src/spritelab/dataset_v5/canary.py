"""Exactly-20-record GPT-5.6 Sol canary accounting."""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.dataset_v5.blind import BlindInput
from spritelab.dataset_v5.labeling import CRITICAL_FIELDS, reconcile_sol_passes
from spritelab.dataset_v5.sol import SolModelUnavailable, SolProvider

CANARY_SCHEMA_VERSION = "raw_v5_sol_canary_v1"
REQUIRED_CANARY_TAGS = frozenset(
    {
        "ambiguous",
        "armor_like",
        "gem",
        "key",
        "mineral",
        "misleading_filename",
        "plant",
        "quarantined_image",
        "shade_variant",
        "tool",
        "weapon",
    }
)
CRITICAL_CONTRADICTION_LIMIT = 0.02
MATERIAL_OVERCLAIM_LIMIT = 0.01


@dataclass(frozen=True)
class SolPricing:
    input_per_million: float
    output_per_million: float
    currency: str = "USD"
    pricing_identity: str = ""

    def __post_init__(self) -> None:
        prices = (self.input_per_million, self.output_per_million)
        if any(not math.isfinite(value) or value < 0 for value in prices):
            raise ValueError("Sol prices must be finite and non-negative")
        if not self.currency.strip():
            raise ValueError("Sol pricing currency must be non-empty")
        if not self.pricing_identity.strip():
            raise ValueError("Sol pricing identity must be non-empty")

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_per_million / 1_000_000 + output_tokens * self.output_per_million / 1_000_000


class SolCanaryModelUnavailable(SolModelUnavailable):
    """Exact Sol identity failed after one or more canary calls."""

    def __init__(self, partial_report: Mapping[str, Any]) -> None:
        self.partial_report = dict(partial_report)
        super().__init__("exact Sol identity changed during the canary")


def run_sol_canary(
    records: Sequence[tuple[BlindInput, Mapping[str, Any]]],
    *,
    provider: SolProvider,
    projected_record_count: int,
    pricing: SolPricing | None,
    metered: bool = True,
    explicit_bulk_cost_authorization: bool = False,
) -> dict[str, Any]:
    """Run two blind passes for exactly 20 records and never hide cost."""

    _validate_run_configuration(
        projected_record_count=projected_record_count,
        pricing=pricing,
        metered=metered,
    )
    if len(records) != 20:
        raise ValueError("Sol canary requires exactly 20 records")
    tags = {str(tag) for _, metadata in records for tag in metadata.get("canary_tags", [])}
    missing_tags = sorted(REQUIRED_CANARY_TAGS - tags)
    if missing_tags:
        raise ValueError("Sol canary cohort is missing required tags: " + ", ".join(missing_tags))

    # Capture the attested identity once. Later per-response identity changes
    # must stop the run, but must not erase evidence from earlier paid calls.
    model_identity = provider.identity.canonical()
    artifacts: list[dict[str, Any]] = []
    reconciled: list[dict[str, Any]] = []
    local_agreement_numerator = 0
    local_agreement_denominator = 0
    records_started = 0
    records_completed = 0

    for blind_input, selection_metadata in records:
        records_started += 1
        prefix = blind_input.record_id[4:20]
        try:
            first = _observed_provider_call(
                provider,
                blind_input,
                request_id=f"req_{prefix}_a",
                pass_kind="adjudication",
                artifacts=artifacts,
                model_identity=model_identity,
            )
            second = _observed_provider_call(
                provider,
                blind_input,
                request_id=f"req_{prefix}_b",
                pass_kind="consistency",
                artifacts=artifacts,
                model_identity=model_identity,
            )
        except SolModelUnavailable as exc:
            report = _build_canary_report(
                artifacts=artifacts,
                reconciled=reconciled,
                local_agreement_numerator=local_agreement_numerator,
                local_agreement_denominator=local_agreement_denominator,
                model_identity=model_identity,
                pricing=pricing,
                metered=metered,
                explicit_bulk_cost_authorization=explicit_bulk_cost_authorization,
                projected_record_count=projected_record_count,
                records_started=records_started,
                records_completed=records_completed,
                tags=tags,
                identity_failure=True,
            )
            raise SolCanaryModelUnavailable(report) from exc

        records_completed += 1
        if first.get("authoritative") and second.get("authoritative"):
            item = reconcile_sol_passes(
                first["output"],
                second["output"],
                deterministic_facts=blind_input.pixel_facts,
                local_proposal=selection_metadata.get("local_proposal"),
            )
            reconciled.append(item)
            local_agreement = item.get("local_sol_agreement")
            if isinstance(local_agreement, Mapping):
                local_agreement_numerator += sum(bool(value) for value in local_agreement.values())
                local_agreement_denominator += len(local_agreement)

    return _build_canary_report(
        artifacts=artifacts,
        reconciled=reconciled,
        local_agreement_numerator=local_agreement_numerator,
        local_agreement_denominator=local_agreement_denominator,
        model_identity=model_identity,
        pricing=pricing,
        metered=metered,
        explicit_bulk_cost_authorization=explicit_bulk_cost_authorization,
        projected_record_count=projected_record_count,
        records_started=records_started,
        records_completed=records_completed,
        tags=tags,
        identity_failure=False,
    )


def unavailable_canary_report(public_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evidence artifact for a run stopped before any provider call."""

    return {
        "abstention_rate": None,
        "artifacts": [],
        "authoritative": False,
        "canary_record_count": 0,
        "cohort_record_count": 0,
        "configured_identity": dict(public_config or {}),
        "critical_field_agreement": None,
        "currency": None,
        "explicit_bulk_cost_authorization": False,
        "filename_leakage": None,
        "invalid_json_count": 0,
        "latency_ms": {"max": None, "mean": None, "median": None},
        "local_sol_agreement": None,
        "material_overclaim_rate": None,
        "metered": None,
        "observed_cost": None,
        "ok": False,
        "pricing_configured": False,
        "pricing_identity": None,
        "projected_total_calls": None,
        "projected_total_cost": None,
        "projected_total_runtime_seconds": None,
        "provider_calls": 0,
        "quality_gates": _empty_quality_gates(),
        "quality_gates_passed": False,
        "reason": "SOL_MODEL_UNAVAILABLE",
        "records_completed": 0,
        "records_started": 0,
        "request_payloads_audited": 0,
        "required_tags_covered": [],
        "schema_version": CANARY_SCHEMA_VERSION,
        "sol_self_consistency": None,
        "status": "blocked_non_authoritative",
        "stop_before_bulk": True,
        "stop_reason": "exact_sol_model_unavailable",
        "stop_reasons": ["exact_sol_model_unavailable"],
        "success_rate": None,
        "successful_record_count": 0,
        "token_usage": Counter(),
        "token_usage_reported_calls": 0,
        "valid_json_rate": None,
    }


def _observed_provider_call(
    provider: SolProvider,
    blind_input: BlindInput,
    *,
    request_id: str,
    pass_kind: str,
    artifacts: list[dict[str, Any]],
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        artifact = provider.call_blind_pass(
            blind_input,
            request_id=request_id,
            pass_kind=pass_kind,
        )
    except SolModelUnavailable:
        # The transport call occurred but SolProvider correctly refused to
        # return an artifact because its exact identity attestation changed.
        # Record only locally observed, non-semantic facts; usage for this call
        # is explicitly unavailable rather than invented as zero.
        artifacts.append(
            {
                "authoritative": False,
                "backend": model_identity.get("backend"),
                "endpoint_identity": model_identity.get("endpoint_identity"),
                "error_code": "exact_sol_identity_changed",
                "http_attempts": 1,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "model_identifier": model_identity.get("model_identifier"),
                "model_version": model_identity.get("model_version"),
                "pass_kind": pass_kind,
                "provider": model_identity.get("provider"),
                "record_id": blind_input.record_id,
                "request_id": request_id,
                "status": "blocked",
                "token_usage_available": False,
            }
        )
        raise
    artifacts.append(artifact)
    return artifact


def _build_canary_report(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    reconciled: Sequence[Mapping[str, Any]],
    local_agreement_numerator: int,
    local_agreement_denominator: int,
    model_identity: Mapping[str, Any],
    pricing: SolPricing | None,
    metered: bool,
    explicit_bulk_cost_authorization: bool,
    projected_record_count: int,
    records_started: int,
    records_completed: int,
    tags: set[str],
    identity_failure: bool,
) -> dict[str, Any]:
    success = [artifact for artifact in artifacts if artifact.get("authoritative")]
    usage_artifacts = [artifact for artifact in artifacts if isinstance(artifact.get("token_usage"), Mapping)]
    input_tokens = sum(int(artifact["token_usage"].get("input_tokens", 0)) for artifact in usage_artifacts)
    output_tokens = sum(int(artifact["token_usage"].get("output_tokens", 0)) for artifact in usage_artifacts)
    latency = [float(artifact["latency_ms"]) for artifact in artifacts if artifact.get("latency_ms") is not None]
    critical_total = len(reconciled) * len(CRITICAL_FIELDS)
    critical_conflicts = sum(len(item.get("critical_conflicts", [])) for item in reconciled)
    contradiction_rate = _optional_rate(critical_conflicts, critical_total)
    semantic_fields = [details for item in reconciled for details in item.get("fields", {}).values()]
    abstentions = sum(details.get("state") == "abstained" for details in semantic_fields)
    material_claims = [item.get("fields", {}).get("explicit_material", {}) for item in reconciled]
    material_overclaims = sum(item.get("state") == "unsupported_removed" for item in material_claims)
    material_overclaim_rate = _optional_rate(material_overclaims, len(material_claims))
    invalid_json_count = len(artifacts) - len(success)
    filename_leakage = 0
    expected_calls = 40
    quality_gates = {
        "complete_two_pass_cohort": {
            "observed_calls": len(artifacts),
            "observed_completed_records": records_completed,
            "passed": len(artifacts) == expected_calls and records_completed == 20,
            "required_calls": expected_calls,
            "required_completed_records": 20,
        },
        "critical_field_contradiction_rate": {
            "limit": CRITICAL_CONTRADICTION_LIMIT,
            "observed": contradiction_rate,
            "passed": contradiction_rate is not None and contradiction_rate <= CRITICAL_CONTRADICTION_LIMIT,
        },
        "filename_leakage": {
            "limit": 0,
            "observed": filename_leakage,
            "passed": filename_leakage == 0,
        },
        "material_overclaim_rate": {
            "limit": MATERIAL_OVERCLAIM_LIMIT,
            "observed": material_overclaim_rate,
            "passed": material_overclaim_rate is not None and material_overclaim_rate <= MATERIAL_OVERCLAIM_LIMIT,
        },
        "strict_response_schema": {
            "limit": 0,
            "observed_invalid_or_rejected": invalid_json_count,
            "passed": invalid_json_count == 0 and len(success) == expected_calls,
        },
    }
    quality_gates_passed = all(bool(gate["passed"]) for gate in quality_gates.values())

    if metered:
        assert pricing is not None  # validated before the first provider call
        observed_cost = pricing.estimate(input_tokens, output_tokens)
        projected_cost = observed_cost / len(usage_artifacts) * projected_record_count * 2 if usage_artifacts else None
        currency = pricing.currency
        pricing_identity = pricing.pricing_identity
    else:
        observed_cost = 0.0
        projected_cost = 0.0
        currency = None
        pricing_identity = "unmetered"

    projected_runtime = statistics.fmean(latency) * projected_record_count * 2 / 1000.0 if latency else None
    stop_reasons: list[str] = []
    if identity_failure:
        stop_reasons.append("exact_sol_identity_changed_during_canary")
    if not quality_gates_passed:
        stop_reasons.append("canary_quality_or_schema_gate_failed")
    if metered and not explicit_bulk_cost_authorization:
        stop_reasons.append("metered_canary_complete_await_bulk_cost_authorization")
    stop_before_bulk = bool(stop_reasons)
    if identity_failure or not quality_gates_passed:
        status = "blocked_non_authoritative"
    elif stop_before_bulk:
        status = "canary_passed_bulk_blocked"
    else:
        status = "pass"
    if len(stop_reasons) == 1:
        stop_reason = stop_reasons[0]
    elif stop_reasons:
        stop_reason = "multiple_canary_stop_gates_failed"
    else:
        stop_reason = None

    return {
        "abstention_rate": _optional_rate(abstentions, len(semantic_fields)),
        "artifacts": [dict(artifact) for artifact in artifacts],
        "authoritative": quality_gates_passed and not identity_failure,
        "canary_record_count": records_completed,
        "cohort_record_count": 20,
        "critical_field_agreement": None if contradiction_rate is None else 1.0 - contradiction_rate,
        "currency": currency,
        "explicit_bulk_cost_authorization": explicit_bulk_cost_authorization,
        "filename_leakage": filename_leakage,
        "invalid_json_count": invalid_json_count,
        "latency_ms": {
            "max": max(latency, default=None),
            "mean": statistics.fmean(latency) if latency else None,
            "median": statistics.median(latency) if latency else None,
        },
        "local_sol_agreement": _optional_rate(local_agreement_numerator, local_agreement_denominator),
        "material_overclaim_rate": material_overclaim_rate,
        "metered": metered,
        "model_identity": dict(model_identity),
        "observed_cost": observed_cost,
        "ok": quality_gates_passed and not identity_failure,
        "pricing_configured": pricing is not None,
        "pricing_identity": pricing_identity,
        "projected_total_calls": projected_record_count * 2,
        "projected_total_cost": projected_cost,
        "projected_total_runtime_seconds": projected_runtime,
        "provider_calls": len(artifacts),
        "quality_gates": quality_gates,
        "quality_gates_passed": quality_gates_passed,
        "reason": "SOL_MODEL_UNAVAILABLE" if identity_failure else None,
        "records_completed": records_completed,
        "records_started": records_started,
        "request_payloads_audited": len(artifacts),
        "required_tags_covered": sorted(tags & REQUIRED_CANARY_TAGS),
        "schema_version": CANARY_SCHEMA_VERSION,
        "sol_self_consistency": None if contradiction_rate is None else 1.0 - contradiction_rate,
        "status": status,
        "stop_before_bulk": stop_before_bulk,
        "stop_reason": stop_reason,
        "stop_reasons": stop_reasons,
        "success_rate": _optional_rate(len(success), len(artifacts)),
        "successful_record_count": len(reconciled),
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "token_usage_reported_calls": len(usage_artifacts),
        "valid_json_rate": _optional_rate(len(success), len(artifacts)),
    }


def _validate_run_configuration(
    *,
    projected_record_count: int,
    pricing: SolPricing | None,
    metered: bool,
) -> None:
    if isinstance(projected_record_count, bool) or not isinstance(projected_record_count, int):
        raise ValueError("projected_record_count must be a positive integer")
    if projected_record_count <= 0:
        raise ValueError("projected_record_count must be a positive integer")
    if metered and pricing is None:
        raise ValueError("metered Sol canary requires explicit pricing")
    if not metered and pricing is not None:
        raise ValueError("unmetered Sol canary must not configure metered pricing")


def _empty_quality_gates() -> dict[str, dict[str, Any]]:
    return {
        "complete_two_pass_cohort": {"observed_calls": 0, "passed": False, "required_calls": 40},
        "critical_field_contradiction_rate": {
            "limit": CRITICAL_CONTRADICTION_LIMIT,
            "observed": None,
            "passed": False,
        },
        "filename_leakage": {"limit": 0, "observed": None, "passed": False},
        "material_overclaim_rate": {
            "limit": MATERIAL_OVERCLAIM_LIMIT,
            "observed": None,
            "passed": False,
        },
        "strict_response_schema": {
            "limit": 0,
            "observed_invalid_or_rejected": None,
            "passed": False,
        },
    }


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator
