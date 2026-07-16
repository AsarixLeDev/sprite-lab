"""Test configuration for running pytest directly from the repository."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Historical integration fixtures are intentionally not committed because they
# contain machine-specific experiment output. Unit tests remain runnable on a
# clean clone; only tests that explicitly require those optional archives skip.
_MODULE_FIXTURES = {
    "test_dataset_v5_named_views.py": (ROOT / "experiments" / "v5_view_contract_v1",),
    "test_label_v4_audit_prefill_gui.py": (
        ROOT / "experiments" / "label_v4_calibration_wave1" / "audit_manifest.jsonl",
        ROOT / "experiments" / "label_v4_real_pilot_15_v1",
        ROOT / "experiments" / "label_v4_pilot_replay_v2",
    ),
    "test_label_v4_two_pass_workflow.py": (
        ROOT / "experiments" / "label_v4_calibration_wave1" / "audit_manifest.jsonl",
        ROOT / "experiments" / "label_v4_real_pilot_15_v1",
        ROOT / "experiments" / "label_v4_pilot_replay_v2",
    ),
}

_TEST_FIXTURES = {
    ("test_dataset_v5_conditional_abstention.py", "test_original_campaign_artifact_remains_byte_identical"): (
        ROOT / "experiments" / "v5_codex_blind_labeling_v1",
    ),
    ("test_dataset_v5_conservative_labeling.py", "test_salvage_manifest_contains_no_strong_or_human_labels"): (
        ROOT / "experiments" / "v3_labeling_calibration_remediation_v1",
    ),
    (
        "test_dataset_v5_conservative_labeling.py",
        "test_historical_schema_remains_readable_without_strength_inference",
    ): (ROOT / "experiments" / "v5_codex_blind_labeling_v1",),
    ("test_dataset_v5_conservative_labeling.py", "test_original_pass_a_shards_remain_byte_identical"): (
        ROOT / "experiments" / "v5_codex_blind_labeling_v1",
    ),
    ("test_label_v4_canary_hardening.py", "test_targeted_manifest_is_exact_and_mock_regression_covers_all_15"): (
        ROOT / "experiments" / "label_v4_canary_hardening" / "targeted_smoke_manifest.jsonl",
        ROOT / "out" / "r2_annotation_batch_0001_semantic_accept_only_25.jsonl",
        ROOT / "out" / "r2_assisted_v3_batch_0001" / "scheduler_resolved_candidates.jsonl",
    ),
    ("test_label_v4_pipeline.py", "test_mocked_same_cohort_improves_coverage_without_paid_calls"): (
        ROOT / "out" / "r2_annotation_batch_0001_semantic_accept_only_25.jsonl",
    ),
    ("test_label_v4_replay_calibration.py", "test_replay_has_zero_http_and_is_byte_identical"): (
        ROOT / "experiments" / "label_v4_real_pilot_15_v1",
    ),
    ("test_label_v4_replay_calibration.py", "test_replay_incompatible_cache_fails_closed"): (
        ROOT / "experiments" / "label_v4_real_pilot_15_v1",
    ),
    ("test_label_v4_replay_calibration.py", "test_small_purple_repair_and_fallback_provenance"): (
        ROOT / "experiments" / "label_v4_real_pilot_15_v1",
    ),
    ("test_label_v4_replay_calibration.py", "test_pilot_description_regressions"): (
        ROOT / "experiments" / "label_v4_real_pilot_15_v1",
    ),
    ("test_product_user_journeys_contract.py", "test_committed_contract_artifacts_match_the_reusable_payloads"): (
        ROOT / "experiments" / "v3_novice_ux_v1",
    ),
    (
        "test_label_v4_smoke_selection.py",
        "test_three_record_smoke_marks_unexecuted_named_recoveries_not_evaluated",
    ): (ROOT / "out" / "r2_annotation_batch_0001_semantic_accept_only_25.jsonl",),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        filename = item.path.name
        required = _MODULE_FIXTURES.get(filename, ()) + _TEST_FIXTURES.get((filename, item.name), ())
        missing = [path for path in required if not path.exists()]
        if missing:
            item.add_marker(pytest.mark.skip(reason="optional external experiment fixtures are not installed"))
