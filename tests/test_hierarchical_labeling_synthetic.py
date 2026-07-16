from __future__ import annotations

import json

from spritelab.hierarchical_labeling.synthetic_demo import run_synthetic_demo


def test_synthetic_end_to_end_architecture_demonstration(tmp_path) -> None:
    result = run_synthetic_demo(tmp_path / "synthetic")
    assert result["corpus"]["records"] >= 30
    assert result["corpus"]["clear_broad_categories"]
    assert result["corpus"]["ambiguous_leaves"]
    assert result["corpus"]["metadata_conflicts"]
    assert result["corpus"]["duplicate_clusters"]
    assert result["corpus"]["near_duplicates"]
    assert result["corpus"]["sheet_derived"]
    assert result["corpus"]["novel_outliers"]
    assert result["corpus"]["animation_frames"]
    assert result["demonstrations"]["hierarchy_increases_broad_accepted_coverage"]
    assert result["demonstrations"]["uncertain_leaves_abstain"]
    assert result["demonstrations"]["metadata_conflicts_visible"]
    assert result["demonstrations"]["retrieval_reviewed_neighbors_authoritative_only"]
    assert result["demonstrations"]["calibration_truth_source"] == "synthetic oracle fixture holdout partition"
    assert result["demonstrations"]["holdout_sample_size"] == 5
    assert result["demonstrations"]["model_model_agreement_is_truth"] is False
    assert result["demonstrations"]["image_only_eligibility_preserved"]
    assert result["demonstrations"]["report_precision_graph_available"] is False
    assert result["demonstrations"]["synthetic_oracle_precision_available"] is True
    assert not result["production_authorization"]


def test_synthetic_demo_safety_counters_and_artifacts(tmp_path) -> None:
    output = tmp_path / "synthetic"
    result = run_synthetic_demo(output)
    assert result["safety"] == {
        "real_provider_calls": 0,
        "fake_provider_calls": result["corpus"]["records"] * 2,
        "hosted_calls": 0,
        "network_calls": 0,
        "gpu_initializations": 0,
        "training_runs": 0,
        "production_freezes": 0,
        "synthetic_oracle_labels": 30,
        "human_review_events": 1,
        "human_review_truth_events": 0,
        "human_labels_auto_created": 0,
    }
    persisted = json.loads((output / "synthetic_end_to_end_results.json").read_text(encoding="utf-8"))
    assert persisted["result_identity"] == result["result_identity"]
    assert (output / "human_review_events.jsonl").is_file()
    ledger = json.loads((output / "synthetic_runtime_events.json").read_text(encoding="utf-8"))
    assert ledger["ledger_identity"] == result["artifacts"]["runtime_event_ledger_identity"]
    assert ledger["summary"] == result["safety"]
    assert (output / "report" / "labeling_report.html").is_file()
    assert "do not generalize" in " ".join(result["limitations"]).lower()
