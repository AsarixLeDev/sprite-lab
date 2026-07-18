from __future__ import annotations

import json
from pathlib import Path

import pytest

from spritelab.evaluation.metric_definitions import metric_definition_identity
from spritelab.product_features.evaluation import (
    IncompatibleMetricDefinitions,
    build_dashboard,
    compare_evaluations,
    filter_gallery,
    memorization_display,
    promotion_integrity_display,
)
from spritelab.product_features.evaluation.dashboard import public_evaluation_projection


def _report(*, definition: str = "v1", review: int = 0, hard: int = 0) -> dict:
    return {
        "schema_version": "generation_benchmark_v1.0",
        "metric_definitions": {"structural": definition},
        "summary": {
            "sample_count": 2,
            "hard_validity": {"pass_rate": 1.0},
            "conditional": {"represented_rate": 0.75},
            "pixel_art": {"palette_size_mean": 8.0},
            "diversity": {"exact_duplicate_rate": 0.0, "repeated_template_rate": 0.1},
            "memorization": {"review_required_count": review, "hard_evidence_count": hard},
        },
        "promotion": {"memorization_machine_status": "manual_review_required" if review else "pass"},
    }


def _rows() -> list[dict]:
    return [
        {
            "sample_id": "sample-1",
            "prompt_id": "p1",
            "prompt": "red sword",
            "seed": 10,
            "category": "weapon",
            "source_id": "public-pack-a",
            "image": "C:/private/source/generated.png",
            "checkpoint": "C:/private/checkpoint.pt",
            "metrics": {"pixel_art": {"unique_palette_size": 6, "silhouette_occupancy": 0.3}},
            "conditional_adherence": 1.0,
            "generation_failed": False,
        },
        {
            "sample_id": "sample-2",
            "prompt_id": "p2",
            "prompt": "blue potion",
            "seed": 11,
            "category": "consumable",
            "source_id": "public-pack-b",
            "metrics": {"pixel_art": {"unique_palette_size": 10, "silhouette_occupancy": 0.4}},
            "conditional_adherence": 0.5,
            "generation_failed": False,
        },
    ]


def test_metric_charts_use_real_rows_and_gallery_redacts_private_paths() -> None:
    dashboard = build_dashboard(_report(), _rows())
    assert all(chart["status"] == "AVAILABLE" for chart in dashboard["charts"])
    assert dashboard["charts"][0]["series"]
    assert len(dashboard["gallery"]) == 2
    assert "C:/private/source" not in json.dumps(dashboard)
    assert dashboard["per_source"] == []


def test_dashboard_recursively_allowlists_metrics_and_exact_booleans() -> None:
    secret = "DASHBOARD-ADAPTER-SECRET"
    posix_path = "/srv/private/evaluator/metrics.json"
    windows_path = r"C:\private\evaluator\metrics.json"
    file_uri = "file:///srv/private/evaluator/metrics.json"
    report = _report()
    report["summary"]["pixel_art"]["adapter_payload"] = {
        "api_key": secret,
        "posix": posix_path,
        "windows": windows_path,
        "uri": file_uri,
    }
    rows = _rows()
    rows[0]["metrics"]["hard_validity"] = {"pass": True, "adapter_secret": secret}
    rows[0]["metrics"]["pixel_art"]["adapter_payload"] = {
        "authorization": f"Bearer {secret}",
        "path": posix_path,
    }
    rows[0]["adapter_payload"] = {"file_uri": file_uri, "password": secret}

    dashboard = build_dashboard(report, rows)
    serialized = json.dumps(dashboard, sort_keys=True)

    assert dashboard["source_results_allowed"] is False
    assert dashboard["gallery"][0]["metrics"]["hard_validity"]["pass"] is True
    assert dashboard["gallery"][0]["metrics"]["pixel_art"]["unique_palette_size"] == 6
    assert "adapter_payload" not in serialized
    for private in (secret, posix_path, windows_path, file_uri):
        assert private not in serialized


def test_dashboard_projects_rows_before_aggregation() -> None:
    row = _rows()[0]
    row["category"] = {"password": "CATEGORY-SECRET"}
    row["source_id"] = r"C:\private\source-pack"
    row["generation_failed"] = "false"

    dashboard = build_dashboard(_report(), [row], allow_source_results=True)

    assert dashboard["per_category"] == [
        {
            "name": "unknown",
            "sample_count": 1,
            "structural_validity_rate": None,
            "conditional_adherence": 1.0,
        }
    ]
    assert dashboard["per_source"][0]["name"] == "source-pack"
    assert "CATEGORY-SECRET" not in json.dumps(dashboard)
    assert "C:\\private" not in json.dumps(dashboard)


def test_no_data_chart_is_explicit() -> None:
    dashboard = build_dashboard(_report(), [])
    assert {chart["status"] for chart in dashboard["charts"]} == {"NO_DATA"}
    assert all(chart["no_data_message"] for chart in dashboard["charts"])


def test_sample_gallery_filters_and_sorts() -> None:
    gallery = build_dashboard(_report(), _rows())["gallery"]
    filtered = filter_gallery(gallery, prompt="sword", category="weapon")
    assert [item["sample_id"] for item in filtered] == ["sample-1"]
    sorted_rows = filter_gallery(gallery, sort_metric="conditional_adherence")
    assert [item["sample_id"] for item in sorted_rows] == ["sample-1", "sample-2"]


def test_side_by_side_comparison_reports_metric_category_and_sample_changes() -> None:
    left = _report()
    right = _report()
    right["summary"]["conditional"]["represented_rate"] = 0.9
    comparison = compare_evaluations(left, right, _rows(), _rows())
    assert comparison["compatible"] is True
    assert len(comparison["sample_pairs"]) == 2
    assert comparison["category_changes"]
    conditional = next(item for item in comparison["metrics"] if item["metric"] == "conditional.represented_rate")
    assert conditional["change"] == pytest.approx(0.15)


def test_incompatible_metric_definitions_are_rejected_before_averaging() -> None:
    with pytest.raises(IncompatibleMetricDefinitions, match="incompatible"):
        compare_evaluations(_report(definition="v1"), _report(definition="v2"))


def test_explicit_metric_definitions_are_bound_to_report_schema() -> None:
    left = _report()
    right = _report()
    right["schema_version"] = "generation_benchmark_v2.0"

    with pytest.raises(IncompatibleMetricDefinitions, match="incompatible"):
        compare_evaluations(left, right)


def test_missing_metric_definitions_are_never_treated_as_compatible() -> None:
    with pytest.raises(IncompatibleMetricDefinitions, match="no complete"):
        compare_evaluations({}, {})


def test_detector_policy_identity_is_comparison_relevant() -> None:
    def report(detector_sha256: str) -> dict:
        return {
            "schema_version": "generation_benchmark_v1.0",
            "thresholds": {"near": 0.1},
            "detector_policy_version": "detector-v1",
            "detector_policy_sha256": detector_sha256,
            "comparison_method": "rgba-v1",
            "comparison_parameters_sha256": "c" * 64,
            "summary": {},
        }

    with pytest.raises(IncompatibleMetricDefinitions, match="incompatible"):
        compare_evaluations(report("a" * 64), report("b" * 64))


def test_explicit_metric_definition_hash_must_agree_with_all_definition_fields() -> None:
    report = _report()
    report.update(
        detector_policy_version="detector-v1",
        detector_policy_sha256="a" * 64,
        comparison_method="rgba-v1",
        comparison_parameters_sha256="b" * 64,
        thresholds={"near": 0.1},
    )
    report["metric_definitions_sha256"] = metric_definition_identity(report)
    drifted = json.loads(json.dumps(report))
    drifted["detector_policy_sha256"] = "c" * 64

    with pytest.raises(IncompatibleMetricDefinitions, match="does not agree"):
        compare_evaluations(report, drifted)


def test_embedded_comparison_parameter_hash_must_agree_with_fields() -> None:
    report = {
        "schema_version": "generation_benchmark_v1.0",
        "thresholds": {"near": 0.1},
        "detector_policy_version": "detector-v1",
        "detector_policy_sha256": "a" * 64,
        "comparison_method": "rgba-v1",
        "comparison_parameters": {"near": 0.1},
        "comparison_parameters_sha256": "b" * 64,
    }

    with pytest.raises(IncompatibleMetricDefinitions, match="Comparison-parameter"):
        metric_definition_identity(report)


def test_public_reports_preserve_metric_definition_compatibility_identity() -> None:
    left = public_evaluation_projection(_report(definition="v1"), surface="report")
    right = public_evaluation_projection(_report(definition="v2"), surface="report")

    assert "metric_definitions" not in left
    assert len(left["metric_definitions_sha256"]) == 64
    assert left["metric_definitions_sha256"] != right["metric_definitions_sha256"]
    with pytest.raises(IncompatibleMetricDefinitions, match="incompatible"):
        compare_evaluations(left, right)


def test_hard_memorization_evidence_never_exposes_clear_action() -> None:
    display = memorization_display([{"pair_id": "pair-1", "evidence_class": "exact_rgba_nontrivial"}])
    assert display["items"] == []
    assert display["review_action_available"] is False
    assert display["evidence_state"] == "incomplete"


def test_review_required_state_links_to_common_review_feature() -> None:
    display = memorization_display(
        [{"pair_id": f"pair-{index}", "evidence_class": "near_pixel_review_required"} for index in range(12)]
    )
    assert display["review_message"].startswith("Memorization review evidence is incomplete")
    assert display["review_link"] is None
    assert display["writes_review_log"] is False


def test_unsigned_incomplete_review_is_not_authoritative(tmp_path: Path) -> None:
    review_log = tmp_path / "reviews.jsonl"
    review_log.write_text(
        json.dumps(
            {
                "schema_version": "sprite_lab_memorization_review_event_v2",
                "pair_id": "pair-1",
                "review_outcome": "likely_false_positive",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    display = memorization_display(
        [{"pair_id": "pair-1", "evidence_class": "near_pixel_review_required"}],
        review_log=review_log,
    )
    assert display["items"] == []
    assert display["review_action_available"] is False


def test_failed_memorization_audit_blocks_promotion_authorization(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"verdict": "FAIL", "authorization": {"checkpoint_promotion": True}}),
        encoding="utf-8",
    )
    display = promotion_integrity_display(audit)
    assert display["promotion_authorized"] is False
    assert display["integrity_certified"] is False
    assert display["message"] == "Promotion integrity is not currently certified."
    assert display["actions"] == []
