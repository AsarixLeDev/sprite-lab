from __future__ import annotations

from copy import deepcopy

import pytest

from spritelab.evaluation.memorization import MemorizationMachineStatus
from spritelab.evaluation.suite import DEFAULT_GATES, evaluate_gates


def _summary(evidence_class: str) -> dict:
    hard = evidence_class == "exact_rgba_nontrivial"
    review = evidence_class.endswith("_review_required")
    return {
        "sample_count": 1,
        "hard_validity": {"malformed_count": 0},
        "pixel_art": {"semi_transparent_ratio_mean": 0.0, "palette_size_mean": 8.0},
        "diversity": {"exact_duplicate_rate": 0.0, "repeated_template_rate": 0.0},
        "conditional": {"represented_rate": 1.0},
        "memorization": {
            "detector_policy_version": "memorization_detector_v2",
            "comparison_method": "deterministic_rgba_alpha_occupancy_v2",
            "hard_evidence_count": int(hard),
            "review_required_count": int(review),
            "warning_count": int(not hard and not review),
            "low_evidence_collision_count": int(not hard and not review),
            "unresolved_candidate_count": int(hard or review),
            "evidence_class_counts": {evidence_class: 1},
        },
    }


def test_suite_exposes_five_state_memorization_outcome() -> None:
    assert (
        evaluate_gates(_summary("exact_rgba_nontrivial"), DEFAULT_GATES)["memorization_machine_status"]
        == MemorizationMachineStatus.HARD_FAIL
    )
    assert (
        evaluate_gates(_summary("exact_alpha_review_required"), DEFAULT_GATES)["memorization_machine_status"]
        == MemorizationMachineStatus.MANUAL_REVIEW_REQUIRED
    )
    assert (
        evaluate_gates(_summary("generic_sparse_collision"), DEFAULT_GATES)["memorization_machine_status"]
        == MemorizationMachineStatus.PASS
    )
    assert (
        evaluate_gates(_summary("invented"), DEFAULT_GATES)["memorization_machine_status"]
        == MemorizationMachineStatus.NOT_COMPARABLE
    )


def test_missing_evidence_counts_are_incomplete_and_cannot_pass() -> None:
    summary = _summary("generic_sparse_collision")
    del summary["memorization"]["evidence_class_counts"]
    result = evaluate_gates(summary, DEFAULT_GATES)
    assert result["memorization_machine_status"] == MemorizationMachineStatus.INCOMPLETE
    assert result["pass"] is False
    assert any("missing evidence counts" in reason for reason in result["memorization_outcome_reasons"])


@pytest.mark.parametrize("value", ["not-an-integer", -1, True])
def test_malformed_negative_and_bool_evidence_counts_are_not_comparable(value: object) -> None:
    summary = _summary("generic_sparse_collision")
    summary["memorization"]["evidence_class_counts"]["generic_sparse_collision"] = value
    result = evaluate_gates(summary, DEFAULT_GATES)
    assert result["memorization_machine_status"] == MemorizationMachineStatus.NOT_COMPARABLE
    assert result["pass"] is False


def test_unknown_evidence_key_and_inconsistent_total_are_not_comparable() -> None:
    unknown = _summary("generic_sparse_collision")
    unknown["memorization"]["evidence_class_counts"] = {"invented": 1}
    assert (
        evaluate_gates(unknown, DEFAULT_GATES)["memorization_machine_status"]
        == MemorizationMachineStatus.NOT_COMPARABLE
    )


def test_malformed_present_count_takes_precedence_over_another_missing_count() -> None:
    summary = _summary("generic_sparse_collision")
    del summary["memorization"]["warning_count"]
    summary["memorization"]["hard_evidence_count"] = True
    result = evaluate_gates(summary, DEFAULT_GATES)
    assert result["memorization_machine_status"] == MemorizationMachineStatus.NOT_COMPARABLE

    inconsistent = deepcopy(_summary("generic_sparse_collision"))
    inconsistent["memorization"]["evidence_class_counts"]["generic_sparse_collision"] = 2
    assert (
        evaluate_gates(inconsistent, DEFAULT_GATES)["memorization_machine_status"]
        == MemorizationMachineStatus.NOT_COMPARABLE
    )
