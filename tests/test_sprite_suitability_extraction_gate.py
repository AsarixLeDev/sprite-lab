from __future__ import annotations

from spritelab.dataset_v5.raw_forensics import (
    RAW_EXTRACTION_OPERATION_SCHEMA_VERSION,
    RawExtractionOperation,
)
from spritelab.suitability import gate_suitability_result


def _operation(operation: str, decoded_hash: str | None, reason: str | None) -> RawExtractionOperation:
    return RawExtractionOperation(
        operation_version=RAW_EXTRACTION_OPERATION_SCHEMA_VERSION,
        operation=operation,
        source_archive_sha256="a" * 64,
        archive_member_path="sprite.png",
        source_member_sha256="b" * 64,
        frame_index=None,
        crop_rectangle=None,
        sheet_row=None,
        sheet_column=None,
        cell_width=None,
        cell_height=None,
        padding_dimensions=None,
        interpolation_policy="none",
        decoded_rgba_sha256=decoded_hash,
        terminal_reason=reason,
        candidate_coordinates=(),
    )


def test_rejected_extraction_can_never_become_suitability_accepted() -> None:
    rejected = _operation("reject_resource_fork", None, "metadata_resource_fork_not_sprite")
    result = gate_suitability_result(rejected, {"status": "accept", "reason_codes": []})
    assert result["status"] == "reject"
    assert result["suitability_evaluated"] is False


def test_suitability_states_and_large_palette_soft_reason_are_preserved() -> None:
    accepted_output = _operation("direct_decode", "c" * 64, None)
    result = gate_suitability_result(
        accepted_output,
        {"status": "quarantine", "reason_codes": ["LARGE_PALETTE"]},
    )
    assert result["status"] == "quarantine"
    assert result["reason_codes"] == ["LARGE_PALETTE"]
