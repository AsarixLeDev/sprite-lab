from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spritelab.dataset_v5.conservative_labeling import (
    adapt_historical_label,
    build_calibration_inputs,
    build_conservative_salvage_summary,
    build_review_queue,
    calibration_readiness,
    compare_field,
    conditional_identity_health,
    conservative_output_schema,
    conservative_prompt_v3,
    is_controlled_conditional_unknown,
    normalize_provider_field,
    normalize_provider_output,
    reconcile_proposals,
    validate_field,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "experiments" / "v5_codex_blind_labeling_v1"


def _labeled(value: object, **evidence: object) -> dict[str, object]:
    return {"state": "labeled", "value": value, "reason": None, **evidence}


def _unknown(**evidence: object) -> dict[str, object]:
    return _labeled("unknown", **evidence)


def _abstained(reason: str = "visually_ambiguous") -> dict[str, object]:
    return {"state": "model_abstained", "value": None, "reason": reason}


def _record(
    *,
    record_id: str = "rec_test",
    domain: dict[str, object] | None = None,
    broad_category: dict[str, object] | None = None,
    canonical_object: dict[str, object] | None = None,
    role: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "image_sha256": "a" * 64,
        "fields": {
            "domain": domain or _labeled("equipment_icon"),
            "broad_category": broad_category or _labeled("weapon"),
            "canonical_object": canonical_object or _abstained("multiple_plausible_objects"),
            "role": role or _abstained("role_not_visually_demonstrated"),
            "visual_form": _labeled("compact silhouette"),
            "visual_material_cue": _labeled("metal-like"),
            "colors": _labeled(["gray"]),
            "description": _labeled("a visible object"),
        },
    }


def test_canonical_object_labeled_unknown_normalizes_to_abstention() -> None:
    result = normalize_provider_field("canonical_object", _unknown(visually_unmistakable=True))
    assert (result["state"], result["value"], result["reason"]) == (
        "model_abstained",
        None,
        "provider_returned_unknown",
    )


def test_role_labeled_unknown_normalizes_to_abstention() -> None:
    result = normalize_provider_field("role", _unknown(visually_demonstrated=True))
    assert result["state"] == "model_abstained"
    assert result["value"] is None


def test_unknown_unknown_reconciliation_abstains() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_unknown(visually_unmistakable=True, evidence="ambiguous")),
        _record(canonical_object=_unknown(visually_unmistakable=True, evidence="ambiguous")),
    )
    assert result["comparisons"]["canonical_object"]["classification"] == "both_abstained"
    assert result["fields"]["canonical_object"]["state"] == "model_abstained"


def test_unknown_abstained_reconciliation_abstains() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_unknown()),
        _record(canonical_object=_abstained("insufficient_resolution")),
    )
    assert result["comparisons"]["canonical_object"]["classification"] == "both_abstained"
    assert result["fields"]["canonical_object"]["value"] is None


def test_abstained_unknown_reconciliation_abstains() -> None:
    result = reconcile_proposals(
        _record(role=_abstained("role_not_visually_demonstrated")),
        _record(role=_unknown()),
    )
    assert result["comparisons"]["role"]["classification"] == "both_abstained"
    assert result["fields"]["role"]["state"] == "model_abstained"


def test_unknown_concrete_reconciliation_abstains() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_unknown()),
        _record(canonical_object=_labeled("sword", visually_unmistakable=True, evidence="clear blade")),
    )
    assert result["comparisons"]["canonical_object"]["classification"] == "one_side_abstention"
    assert result["fields"]["canonical_object"]["state"] == "model_abstained"


def test_two_different_concrete_values_abstain() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_labeled("sword", visually_unmistakable=True, evidence="blade")),
        _record(canonical_object=_labeled("dagger", visually_unmistakable=True, evidence="blade")),
    )
    assert result["fields"]["canonical_object"]["state"] == "model_abstained"


def test_same_concrete_identity_with_sufficient_evidence_may_label() -> None:
    proposal = _labeled("sword", visually_unmistakable=True, evidence="unmistakable blade and hilt")
    result = reconcile_proposals(_record(canonical_object=proposal), _record(canonical_object=proposal))
    assert result["fields"]["canonical_object"]["state"] == "labeled"
    assert result["fields"]["canonical_object"]["value"] == "sword"


def test_same_concrete_identity_with_weak_evidence_abstains() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_labeled("sword", evidence="maybe a blade")),
        _record(canonical_object=_labeled("sword", evidence="maybe a blade")),
    )
    assert result["fields"]["canonical_object"]["state"] == "model_abstained"


def test_labeled_null_is_invalid_output() -> None:
    normalized = normalize_provider_field("canonical_object", _labeled(None))
    assert normalized["state"] == "invalid_output"
    assert normalized["value"] is None
    assert normalized["reason"] == "empty_successful_label"


def test_abstained_non_null_discards_value_fail_closed() -> None:
    normalized = normalize_provider_field(
        "role", {"state": "model_abstained", "value": "combat", "reason": "visually_ambiguous"}
    )
    assert normalized["state"] == "invalid_output"
    assert normalized["value"] is None
    assert normalized["normalization"]["raw_value"] == "combat"


def test_not_applicable_requires_null() -> None:
    assert validate_field("role", {"state": "not_applicable", "value": None}).valid
    invalid = validate_field("role", {"state": "not_applicable", "value": "combat"})
    assert not invalid.valid
    assert invalid.value is None


def test_legacy_unknown_is_read_as_abstention_with_diagnostics() -> None:
    historical = {
        "schema_version": "sprite_lab_codex_blind_label_v1",
        "record_id": "rec_legacy",
        "fields": {
            "canonical_object": {"state": "known", "value": "unknown"},
            "role": {"state": "known", "value": "unknown"},
        },
    }
    adapted = adapt_historical_label(historical, source_artifact_identity="artifact:legacy.jsonl")
    canonical = adapted["fields"]["canonical_object"]
    assert (canonical["state"], canonical["value"], canonical["reason"]) == (
        "model_abstained",
        None,
        "legacy_unknown_normalized",
    )
    diagnostics = canonical["diagnostic_metadata"]
    assert diagnostics["original_raw_state"] == "known"
    assert diagnostics["original_raw_value"] == "unknown"
    assert diagnostics["source_artifact_identity"] == "artifact:legacy.jsonl"


def test_historical_file_bytes_remain_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "historical.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "sprite_lab_codex_blind_label_v1",
                "record_id": "rec_legacy",
                "fields": {"canonical_object": {"state": "known", "value": "unknown"}},
            }
        ),
        encoding="utf-8",
    )
    before = path.read_bytes()
    adapt_historical_label(json.loads(before), source_artifact_identity=str(path))
    assert path.read_bytes() == before


def test_exact_unknown_does_not_count_as_exact_identity_agreement() -> None:
    comparison = compare_field("canonical_object", _unknown(), _unknown())
    assert comparison["classification"] == "both_abstained"


def test_unknown_contributes_to_abstention_denominator() -> None:
    normalized = normalize_provider_output(_record(canonical_object=_unknown(), role=_unknown()))
    health = conditional_identity_health([normalized])
    assert health["conditional_abstention_rate"] == {"numerator": 2, "denominator": 2, "rate": 1.0}


def test_labeled_unknown_health_gate_fails_without_boundary_normalization() -> None:
    health = conditional_identity_health([_record(canonical_object=_unknown(), role=_unknown())])
    assert health["conditional_labeled_unknown_count"] == 2
    assert health["conditional_invalid_state_count"] == 2
    assert health["passed"] is False


def test_normalized_output_passes_labeled_unknown_invariant() -> None:
    normalized = normalize_provider_output(_record(canonical_object=_unknown(), role=_unknown()))
    health = conditional_identity_health([normalized])
    assert health["conditional_labeled_unknown_count"] == 0
    assert health["conditional_invalid_state_count"] == 0
    assert health["passed"] is True


def test_conditional_abstention_creates_no_mandatory_review() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_unknown(), role=_unknown()),
        _record(canonical_object=_unknown(), role=_unknown()),
    )
    assert result["mandatory_review"] is False
    assert build_review_queue([result]) == []


def test_core_contradiction_still_creates_review() -> None:
    result = reconcile_proposals(
        _record(broad_category=_labeled("helmet")),
        _record(broad_category=_labeled("mineral")),
    )
    assert result["mandatory_review"] is True
    assert len(build_review_queue([result])) == 1


def test_image_only_eligibility_survives_conditional_abstention() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_unknown(), role=_unknown()),
        _record(canonical_object=_unknown(), role=_unknown()),
        image_only_eligible=True,
    )
    assert result["image_only_eligible"] is True


def test_salvage_excludes_unknown_identity() -> None:
    result = reconcile_proposals(
        _record(canonical_object=_unknown(), role=_unknown()),
        _record(canonical_object=_unknown(), role=_unknown()),
    )
    summary = build_conservative_salvage_summary([result])
    assert summary["unknown_identity_candidate_count"] == 0
    assert summary["conditional_identity_candidates"] == []


def test_calibration_receives_no_truth_or_correctness_row_from_unknown() -> None:
    reconciled = reconcile_proposals(
        _record(canonical_object=_unknown(), role=_unknown()),
        _record(canonical_object=_unknown(), role=_unknown()),
    )
    conditional_rows = [
        row for row in build_calibration_inputs([reconciled]) if row["field"] in {"canonical_object", "role"}
    ]
    assert all(row["calibration_state"] == "unreviewed" for row in conditional_rows)
    assert all(row["truth_value"] is None and not row["correctness_eligible"] for row in conditional_rows)
    assert all(row["abstention_contribution"] == 1 for row in conditional_rows)
    assert calibration_readiness(conditional_rows)["status"] == "not_ready"


def test_prompt_contract_forbids_labeled_unknown() -> None:
    prompt = conservative_prompt_v3().casefold()
    assert "never emit `unknown` as a labeled value" in prompt
    assert "exact agreement on `unknown` is still" in prompt
    assert "directly demonstrated visually" in prompt
    conditional_schema = conservative_output_schema()["properties"]["fields"]["properties"]
    for field_name in ("canonical_object", "role"):
        labeled_enum = conditional_schema[field_name]["oneOf"][0]["properties"]["value"]["enum"]
        assert "unknown" not in labeled_enum


@pytest.mark.parametrize("value", ["unknown relic", "Unknown", " unknown", "unknown "])
def test_controlled_sentinel_matching_does_not_use_substrings_or_fuzzy_matching(value: str) -> None:
    assert is_controlled_conditional_unknown("canonical_object", value) is False


def test_object_containing_unknown_text_is_not_misclassified_as_abstention() -> None:
    normalized = normalize_provider_field("canonical_object", _labeled("unknown relic"))
    assert normalized["state"] == "invalid_output"
    assert normalized["reason"] == "unknown_taxonomy_value"
    assert normalized["normalization"]["marker"] == "invalid_provider_output_preserved"


@pytest.mark.parametrize("placeholder", ["unspecified", "not sure", "ambiguous", "none"])
def test_unrecognized_placeholders_remain_invalid_output(placeholder: str) -> None:
    normalized = normalize_provider_field("role", _labeled(placeholder))
    assert normalized["state"] == "invalid_output"
    assert normalized["reason"] == "unknown_taxonomy_value"


def test_field_health_numerators_and_denominators_are_correct() -> None:
    normalized_unknown = normalize_provider_output(
        _record(record_id="rec_1", canonical_object=_unknown(), role=_unknown())
    )
    normalized_concrete = normalize_provider_output(
        _record(
            record_id="rec_2",
            canonical_object=_labeled("sword", visually_unmistakable=True, evidence="blade"),
            role=_labeled("combat", visually_demonstrated=True, evidence="visible combat action"),
        )
    )
    health = conditional_identity_health([normalized_unknown, normalized_concrete])
    assert health["pass_field_denominator"] == 4
    assert health["conditional_abstention_rate"] == {"numerator": 2, "denominator": 4, "rate": 0.5}
    assert health["metric_details"]["conditional_concrete_label_count"] == {
        "numerator": 2,
        "denominator": 4,
        "rate": 0.5,
    }


def test_original_campaign_artifact_remains_byte_identical() -> None:
    manifest = json.loads((CAMPAIGN / "artifact_hashes.json").read_text(encoding="utf-8"))["artifacts"]
    relative = "pass_a/shard_0000/labels.jsonl"
    assert hashlib.sha256((CAMPAIGN / relative).read_bytes()).hexdigest() == manifest[relative]["sha256"]
