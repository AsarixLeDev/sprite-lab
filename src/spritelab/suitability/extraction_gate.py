"""Fail-closed bridge between extraction and pixel suitability."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from spritelab.dataset_v5.raw_forensics import RAW_OUTPUT_OPERATIONS, RawExtractionOperation


def gate_suitability_result(
    operation: RawExtractionOperation,
    suitability_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Preserve accept/quarantine/reject without promoting terminal extraction records."""

    if operation.operation not in RAW_OUTPUT_OPERATIONS:
        return {
            "extraction_operation_id": operation.operation_id,
            "extraction_operation": operation.operation,
            "reason_codes": [f"EXTRACTION_{operation.operation.upper()}"],
            "status": "reject",
            "suitability_evaluated": False,
        }
    if suitability_result is None:
        raise ValueError("output extraction operations require a suitability result")
    status = str(suitability_result.get("status") or "")
    if status not in {"accept", "quarantine", "reject"}:
        raise ValueError(f"invalid suitability status: {status!r}")
    result = dict(suitability_result)
    result["extraction_operation_id"] = operation.operation_id
    result["extraction_operation"] = operation.operation
    result["suitability_evaluated"] = True
    return result


def apply_extraction_gate(
    operations: Iterable[RawExtractionOperation],
    suitability_by_operation_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Gate a complete operation set in deterministic operation-ID order."""

    results = []
    for operation in sorted(operations, key=lambda item: item.operation_id):
        suitability = suitability_by_operation_id.get(operation.operation_id)
        results.append(gate_suitability_result(operation, suitability))
    return tuple(results)
