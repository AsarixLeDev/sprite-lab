from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spritelab.dataset_v5.conservative_labeling import (
    REVIEW_EVENT_VERSION,
    adapt_historical_label,
    append_review_event,
    build_calibration_inputs,
    build_review_queue,
    calibration_readiness,
    compare_field,
    conservative_output_schema,
    conservative_prompt_v2,
    field_health_report,
    filename_leakage_findings,
    provider_neutral_request,
    reconcile_proposals,
    validate_field,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments" / "v5_codex_blind_labeling_v1"
REMEDIATION = ROOT / "experiments" / "v3_labeling_calibration_remediation_v1"


def _labeled(value: object, **evidence: object) -> dict[str, object]:
    return {"state": "labeled", "value": value, **evidence}


def _abstained(reason: str = "visually_ambiguous") -> dict[str, object]:
    return {"state": "model_abstained", "value": None, "reason": reason}


def _record(
    *,
    record_id: str = "rec_test",
    domain: dict[str, object] | None = None,
    broad_category: dict[str, object] | None = None,
    canonical_object: dict[str, object] | None = None,
    role: dict[str, object] | None = None,
    description: str = "a visible object",
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "image_sha256": "a" * 64,
        "fields": {
            "domain": domain or _labeled("equipment_icon"),
            "broad_category": broad_category or _labeled("weapon"),
            "canonical_object": canonical_object or _abstained("multiple_plausible_objects"),
            "role": role or _abstained("role_not_visually_demonstrated"),
            "visual_form": _labeled("one compact silhouette"),
            "visual_material_cue": _labeled("metal-like"),
            "colors": _labeled(["gray"]),
            "description": _labeled(description),
        },
    }


def _pair(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    return {"record_id": left["record_id"], "pass_a": left, "pass_b": right}


def test_exact_core_agreement_is_only_a_pending_candidate() -> None:
    result = reconcile_proposals(_record(), _record())
    assert result["fields"]["domain"]["state"] == "labeled"
    assert result["fields"]["broad_category"]["value"] == "weapon"
    assert result["fields"]["broad_category"]["candidate_status"] == "pending_new_independent_health_gate"
    assert result["fields"]["broad_category"]["training_eligible"] is False


def test_explicit_parent_child_category_uses_safer_parent() -> None:
    comparison = compare_field("broad_category", _labeled("weapon"), _labeled("sword"))
    assert comparison["classification"] == "compatible_hierarchy"
    assert comparison["safe_parent"] == "weapon"


def test_incompatible_categories_are_true_contradiction() -> None:
    comparison = compare_field("broad_category", _labeled("helmet"), _labeled("mineral"))
    assert comparison["classification"] == "true_contradiction"
    assert comparison["safe_parent"] is None


def test_one_side_core_abstention_reconciles_to_abstention() -> None:
    left = _record()
    right = _record(domain=_abstained("insufficient_resolution"))
    result = reconcile_proposals(left, right)
    assert result["comparisons"]["domain"]["classification"] == "one_side_abstention"
    assert result["fields"]["domain"]["state"] == "model_abstained"


def test_both_side_abstention_is_normal_valid_outcome() -> None:
    comparison = compare_field("domain", _abstained(), _abstained("image_unidentifiable"))
    assert comparison["classification"] == "both_abstained"


def test_object_disagreement_does_not_invalidate_broad_category() -> None:
    left = _record(canonical_object=_labeled("sword", visually_unmistakable=True, evidence="visible blade"))
    right = _record(canonical_object=_labeled("dagger", visually_unmistakable=True, evidence="visible blade"))
    result = reconcile_proposals(left, right)
    assert result["fields"]["broad_category"]["value"] == "weapon"
    assert result["fields"]["canonical_object"]["state"] == "model_abstained"
    assert result["mandatory_review"] is False


def test_role_disagreement_does_not_invalidate_broad_category() -> None:
    left = _record(role=_labeled("combat", visually_demonstrated=True, evidence="visible action"))
    right = _record(role=_labeled("decoration", visually_demonstrated=True, evidence="visible display"))
    result = reconcile_proposals(left, right)
    assert result["comparisons"]["role"]["classification"] == "true_contradiction"
    assert result["fields"]["role"]["state"] == "model_abstained"
    assert result["fields"]["broad_category"]["state"] == "labeled"
    assert result["mandatory_review"] is False


def test_invalid_taxonomy_state_fails_closed() -> None:
    comparison = compare_field(
        "broad_category",
        {"state": "successful", "value": "weapon"},
        _labeled("weapon"),
    )
    assert comparison["classification"] == "invalid_comparison"


def test_unknown_taxonomy_value_is_invalid_not_similarity_matched() -> None:
    comparison = compare_field("broad_category", _labeled("weaponish"), _labeled("weapon"))
    assert comparison["classification"] == "invalid_comparison"
    assert comparison["taxonomy_valid"] is False


def test_invalid_output_is_queued_only_when_semantic_label_is_required() -> None:
    left = _record(broad_category=_labeled("invented_value"))
    right = _record()
    image_only = reconcile_proposals(left, right)
    required = reconcile_proposals(left, right, semantic_label_required=True)
    assert image_only["mandatory_review"] is False
    assert build_review_queue([image_only]) == []
    assert required["mandatory_review"] is True
    assert len(build_review_queue([required])) == 1


def test_material_overclaim_is_invalid_output() -> None:
    validation = validate_field("visual_material_cue", _labeled("steel"))
    assert validation.valid is False
    assert validation.reason == "exact_material_overclaim"


def test_filename_leakage_is_detected_before_adapter_use() -> None:
    findings = filename_leakage_findings({"record_id": "rec_opaque", "filename": "sword.png"})
    assert findings == [{"location": "$.filename", "reason": "forbidden_metadata_key"}]


def test_field_specific_denominator_reports_missing_records() -> None:
    left = _record(record_id="rec_1")
    right = _record(record_id="rec_1")
    report = field_health_report([_pair(left, right)], expected_record_ids=["rec_1", "rec_2", "rec_3"])
    assert report["record_denominator"] == 3
    assert report["fields"]["domain"]["true_contradiction"]["denominator"] == 3
    assert report["metrics"]["missing_record"] == {"numerator": 2, "denominator": 3, "rate": 2 / 3}


def test_compatible_hierarchy_is_excluded_from_contradiction_rate() -> None:
    left = _record(broad_category=_labeled("weapon"))
    right = _record(broad_category=_labeled("sword"))
    report = field_health_report([_pair(left, right)])
    field = report["fields"]["broad_category"]
    assert field["compatible_hierarchy"]["numerator"] == 1
    assert field["true_contradiction"]["numerator"] == 0


def test_conditional_exact_agreement_without_direct_evidence_abstains() -> None:
    left = _record(canonical_object=_labeled("sword"))
    right = _record(canonical_object=_labeled("sword"))
    result = reconcile_proposals(left, right)
    assert result["comparisons"]["canonical_object"]["classification"] == "exact_agreement"
    assert result["fields"]["canonical_object"]["state"] == "model_abstained"


def test_true_core_contradiction_creates_prefilled_review_item() -> None:
    left = _record(broad_category=_labeled("helmet"))
    right = _record(broad_category=_labeled("mineral"))
    result = reconcile_proposals(left, right)
    result["pass_a_proposal"] = left["fields"]
    result["pass_b_proposal"] = right["fields"]
    queue = build_review_queue([result])
    assert len(queue) == 1
    item = queue[0]
    assert item["image_identity"]["record_id"] == "rec_test"
    assert item["current_conservative_result"]
    assert item["pass_a_proposal"]
    assert item["health_check_or_pass_b_proposal"]
    assert item["field_level_disagreement"]
    assert item["recommended_safe_action"] in item["allowed_actions"]
    assert item["review_reason"]


def test_auxiliary_disagreement_does_not_create_mandatory_review() -> None:
    result = reconcile_proposals(_record(description="one"), _record(description="two"))
    assert result["fields"]["description"]["state"] == "conflict"
    assert result["mandatory_review"] is False
    assert build_review_queue([result]) == []


def test_image_only_eligibility_survives_semantic_abstention() -> None:
    result = reconcile_proposals(
        _record(domain=_abstained(), broad_category=_abstained()),
        _record(domain=_abstained(), broad_category=_abstained()),
        image_only_eligible=True,
    )
    assert result["image_only_eligible"] is True
    assert result["semantic_abstention_changes_image_only_eligibility"] is False


def test_salvage_manifest_contains_no_strong_or_human_labels() -> None:
    payload = (REMEDIATION / "salvage_manifest.jsonl").read_text(encoding="utf-8").casefold()
    assert "supervised_strong" not in payload
    assert "human_verified" not in payload
    assert "ground_truth" not in payload
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == 100
    assert all(row["training_eligible"] is False for row in rows)


def test_review_event_is_append_only_and_cannot_change_identity(tmp_path: Path) -> None:
    left = _record(broad_category=_labeled("helmet"))
    right = _record(broad_category=_labeled("mineral"))
    item = build_review_queue([reconcile_proposals(left, right)])[0]
    path = tmp_path / "events.jsonl"
    first = append_review_event(
        path,
        item,
        action="abstain",
        field_name="broad_category",
        reviewer_id="reviewer",
    )
    before = path.read_bytes()
    second = append_review_event(
        path,
        item,
        action="exclude_semantic_supervision",
        field_name="broad_category",
        reviewer_id="reviewer",
    )
    after = path.read_bytes()
    assert after.startswith(before)
    assert len(after.splitlines()) == 2
    assert first["event_id"] != second["event_id"]
    assert first["changes_provenance_license_or_image_identity"] is False


def test_model_model_agreement_is_never_calibration_truth() -> None:
    reconciled = reconcile_proposals(_record(), _record())
    rows = build_calibration_inputs([reconciled])
    assert any(row["calibration_state"] == "model_agreement_candidate" for row in rows)
    assert all(row["model_agreement_treated_as_truth"] is False for row in rows)
    readiness = calibration_readiness(rows, min_truth_per_field=1)
    assert readiness["status"] == "not_ready"
    assert readiness["model_agreement_candidate_rows_counted_as_truth"] == 0


def test_insufficient_explicit_human_truth_refuses_fit() -> None:
    reconciled = reconcile_proposals(_record(), _record())
    event = {
        "schema_version": REVIEW_EVENT_VERSION,
        "record_id": "rec_test",
        "field": "domain",
        "action": "accept_broad_label",
        "selected_value": "equipment_icon",
    }
    readiness = calibration_readiness(build_calibration_inputs([reconciled], [event]), min_truth_per_field=2)
    assert readiness["status"] == "insufficient_truth"
    assert readiness["fit_calibration_model"] is False


def test_historical_schema_remains_readable_without_strength_inference() -> None:
    historical = json.loads(
        (CAMPAIGN / "pass_a" / "shard_0000" / "labels.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    adapted = adapt_historical_label(historical)
    assert adapted["record_id"] == historical["record_id"]
    assert adapted["fields"]["broad_category"]["value"] == historical["fields"]["category"]["value"]
    assert adapted["migration"]["confidence_or_supervision_strength_added"] is False
    assert adapted["migration"]["source_record_rewritten"] is False


def test_original_pass_a_shards_remain_byte_identical() -> None:
    manifest = json.loads((CAMPAIGN / "artifact_hashes.json").read_text(encoding="utf-8"))["artifacts"]
    for shard in range(4):
        relative = f"pass_a/shard_{shard:04d}/labels.jsonl"
        actual = hashlib.sha256((CAMPAIGN / relative).read_bytes()).hexdigest()
        assert actual == manifest[relative]["sha256"]


def test_structured_abstention_reason_is_required() -> None:
    validation = validate_field("domain", {"state": "model_abstained", "value": None})
    assert validation.valid is False
    assert validation.reason == "invalid_or_missing_abstention_reason"


def test_current_prompt_contains_every_conservative_instruction() -> None:
    prompt = conservative_prompt_v2().casefold()
    for phrase in (
        "broad correct class",
        "multiple objects are plausible",
        "canonical_object",
        "role",
        "gameplay use",
        "filename",
        "exact material",
        "pairs, sets",
        "shared parent",
        "never emit `unknown` as a labeled value",
        "exact agreement on `unknown` is still",
        "low-resolution ambiguity",
    ):
        assert phrase in prompt


def test_provider_neutral_request_has_no_provider_http_logic() -> None:
    request = provider_neutral_request(
        record_id="rec_" + "a" * 64,
        image_sha256="b" * 64,
        image_reference="opaque-image-token",
    )
    assert request["prompt_version"] == "sprite_lab_conservative_prompt_v3"
    assert "url" not in request
    assert "headers" not in request


def test_v2_output_schema_requires_controlled_states_and_abstention_reason() -> None:
    schema = conservative_output_schema()
    domain = schema["properties"]["fields"]["properties"]["domain"]
    assert "model_abstained" in domain["properties"]["state"]["enum"]
    assert domain["allOf"][0]["then"]["properties"]["value"] == {"type": "null"}
    assert "visually_ambiguous" in domain["allOf"][0]["then"]["properties"]["reason"]["enum"]


@pytest.mark.parametrize("reason", ["", "made_up_reason"])
def test_uncontrolled_abstention_reasons_fail_closed(reason: str) -> None:
    assert validate_field("role", {"state": "model_abstained", "value": None, "reason": reason}).valid is False
