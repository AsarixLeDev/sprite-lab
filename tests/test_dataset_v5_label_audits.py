from __future__ import annotations

from copy import deepcopy

import pytest

from spritelab.dataset_v5.audits import (
    audit_label_batch,
    audit_label_health,
    cache_identity_collisions,
    hard_relation_leakage,
    verify_label_drift,
    wilson_interval,
)
from spritelab.dataset_v5.labeling import RECONCILIATION_VERSION, SUPERVISION_POLICY_VERSION


def _field(value: object, *, state: str = "known", supervision: str = "supervised_weak") -> dict[str, object]:
    return {
        "agreement": True,
        "adjudication_value": value,
        "consistency_value": value,
        "negative_target": False,
        "reason": "test_fixture",
        "state": state,
        "supervision_class": supervision,
        "target_mask": 1 if supervision in {"supervised_strong", "supervised_weak"} else 0,
        "value": value,
    }


def _artifact(pass_kind: str, seed: str) -> dict[str, object]:
    return {
        "authoritative": True,
        "backend": "openai_responses_v1",
        "cache_key": seed * 64,
        "endpoint_identity": "https://sol.example.test/v1",
        "model_family": "GPT-5.6 Sol",
        "model_identifier": "gpt-5.6-sol-2026-07-01",
        "model_version": "2026-07-01.1",
        "pass_kind": pass_kind,
        "prompt_version": f"raw_v5_sol_{pass_kind}_prompt_v1",
        "provider": "Fixture Sol Provider",
        "provider_schema_version": "sol_provider_transport_v1",
        "request_schema_version": "blind_semantic_request_v1",
        "request_sha256": seed.upper() * 64,
        "response_schema_version": "blind_semantic_output_v1",
        "status": "success",
    }


def _record(record_id: str = "rec_" + "1" * 64) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_binding_valid": True,
        "blind_request_payload": {
            "metadata": {"record_id": record_id},
            "model": "gpt-5.6-sol-2026-07-01",
            "request_id": "req_fixture",
        },
        "reconciliation": {
            "critical_conflicts": [],
            "deterministic_fields": {},
            "fields": {
                "category": _field("tool"),
                "canonical_object": _field("hammer"),
                "domain": _field("inventory_icon"),
                "role": _field("functional_tool"),
                "visual_form": _field(["compact head", "straight handle"]),
                "material_applicability": _field("applicable"),
                "explicit_material": _field(None, state="missing", supervision="unlabeled"),
                "color_roles": _field({"primary": ["blue gray"]}),
                "description": _field("A compact object with a straight handle."),
            },
            "inclusion_decision": "candidate",
            "reconciliation_version": RECONCILIATION_VERSION,
            "supervision_policy_version": SUPERVISION_POLICY_VERSION,
        },
        "adjudication_artifact": _artifact("adjudication", "a"),
        "consistency_artifact": _artifact("consistency", "c"),
    }


def test_clean_batch_passes_and_health_is_healthy() -> None:
    report = audit_label_batch([_record()], batch_id="batch-0001")
    assert report["status"] == "pass"
    assert report["authoritative"] is True
    assert report["metrics"]["abstention_rate"] == 0.0
    assert report["metrics"]["field_disagreement_rate"] == 0.0
    assert report["metrics"]["invalid_json_rate"] == 0.0
    assert report["metrics"]["repair_rate"] == 0.0
    assert audit_label_health([report])["ok"] is True


def test_empty_batch_and_missing_health_reports_fail_closed() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        audit_label_batch([], batch_id="empty")
    health = audit_label_health([])
    assert health["ok"] is False
    assert health["status"] == "blocked_non_authoritative"


def test_health_rejects_malformed_pass_report() -> None:
    health = audit_label_health([{"status": "pass"}])

    assert health["ok"] is False
    assert health["status"] == "blocked_non_authoritative"
    assert health["invalid_reports"]


def test_health_rejects_overlapping_record_coverage() -> None:
    first = audit_label_batch([_record()], batch_id="batch-0001")
    second = deepcopy(first)
    second["batch_id"] = "batch-0002"

    health = audit_label_health([first, second])

    assert health["ok"] is False
    assert health["coverage"]["overlapping_record_ids"] == ["rec_" + "1" * 64]


def test_batch_audit_stops_on_any_filename_leakage() -> None:
    record = _record()
    record["blind_request_payload"] = {
        "model": "gpt-5.6-sol",
        "original_source_filename": "misleading_helmet.png",
    }
    report = audit_label_batch([record], batch_id="batch-leak")
    assert report["status"] == "blocked_non_authoritative"
    assert report["authoritative"] is False
    assert report["metrics"]["filename_leakage"] == 1


def test_batch_audit_stops_on_critical_contradiction() -> None:
    record = _record()
    canonical = record["reconciliation"]["fields"]["canonical_object"]
    canonical.update(
        agreement=False,
        consistency_value="pickaxe",
        state="conflicted",
        supervision_class="auxiliary_only",
        target_mask=0,
        value=None,
    )
    record["reconciliation"]["critical_conflicts"] = ["canonical_object"]
    record["reconciliation"]["inclusion_decision"] = "quarantine"
    report = audit_label_batch([record], batch_id="batch-conflict")
    assert report["status"] == "blocked_non_authoritative"
    assert report["metrics"]["critical_field_contradiction_rate"] == 0.25
    assert record["record_id"] in report["review_sample_record_ids"]


def test_batch_audit_rejects_silent_repair_and_invalid_json() -> None:
    record = _record()
    record["adjudication_artifact"]["repair_used"] = True
    record["adjudication_artifact"]["status"] = "invalid"
    record["adjudication_artifact"]["authoritative"] = False
    report = audit_label_batch([record], batch_id="batch-repair")
    assert report["metrics"]["silent_repairs"] == 1
    assert report["metrics"]["invalid_json"] == 1
    assert report["status"] == "blocked_non_authoritative"


def test_batch_audit_fails_closed_on_missing_semantic_evidence() -> None:
    record = {
        "record_id": "rec_" + "9" * 64,
        "source_binding_valid": True,
    }

    report = audit_label_batch([record], batch_id="batch-missing-evidence")

    assert report["status"] == "blocked_non_authoritative"
    assert report["authoritative"] is False
    assert report["metrics"]["schema_invalidity"] > 0
    assert report["metrics"]["critical_field_agreement"] == 0.0
    assert report["schema_findings"]


def test_batch_audit_gates_new_taxonomy_values() -> None:
    record = _record()
    record["new_taxonomy_values"] = ["invented_category"]

    report = audit_label_batch([record], batch_id="batch-new-taxonomy")

    assert report["metrics"]["new_taxonomy_value_count"] == 1
    assert report["metrics"]["taxonomy_invalidity"] == 1
    assert report["status"] == "blocked_non_authoritative"


def test_cache_collision_detection() -> None:
    rows = [
        {"cache_key": "same", "request_sha256": "first"},
        {"cache_key": "same", "request_sha256": "second"},
        {"cache_key": "other", "request_sha256": "third"},
    ]
    assert cache_identity_collisions(rows) == 1


def test_hard_relation_leakage_detection() -> None:
    relations = [{"hard_split_constraint": True, "members": ["a", "b"]}]
    assert hard_relation_leakage(relations, {"a": "train", "b": "test"}) == 1
    assert hard_relation_leakage(relations, {"a": "train", "b": "train"}) == 0


def test_label_drift_detects_changed_field() -> None:
    baseline = _record()
    current = deepcopy(baseline)
    current["reconciliation"]["fields"]["category"]["value"] = "weapon"
    report = verify_label_drift([current], [baseline])
    assert report["ok"] is False
    assert report["drift_count"] == 1


def test_label_drift_detects_membership_changes() -> None:
    first = _record("rec_" + "1" * 64)
    second = _record("rec_" + "2" * 64)
    report = verify_label_drift([first, second], [first])
    assert report["ok"] is False
    assert report["added_record_ids"] == [second["record_id"]]
    assert report["drift_count"] == 1


def test_label_drift_detects_pack_and_creator_distribution_changes() -> None:
    baseline = _record()
    baseline["source_pack"] = "pack_a"
    baseline["source_creator"] = "creator_a"
    current = deepcopy(baseline)
    current["source_pack"] = "pack_b"
    current["source_creator"] = "creator_b"

    report = verify_label_drift([current], [baseline])

    assert report["ok"] is False
    assert report["pack_drift"]
    assert report["creator_drift"]


def test_label_drift_detects_supervision_mask_and_provider_prompt_changes() -> None:
    baseline = _record()
    current = deepcopy(baseline)
    category = current["reconciliation"]["fields"]["category"]
    category["supervision_class"] = "supervised_strong"
    category["target_mask"] = 1
    current["adjudication_artifact"]["provider"] = "Different Sol Provider"
    current["consistency_artifact"]["prompt_version"] = "raw_v5_sol_consistency_prompt_v2"

    report = verify_label_drift([current], [baseline])

    assert report["ok"] is False
    assert report["drift_count"] == 1
    change = report["changed"][0]
    assert change["before"]["fields"]["category"]["supervision_class"] == "supervised_weak"
    assert change["after"]["fields"]["category"]["supervision_class"] == "supervised_strong"
    assert change["after"]["passes"]["adjudication"]["provider"] == "Different Sol Provider"
    assert change["after"]["passes"]["consistency"]["prompt_version"].endswith("_v2")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"record_id": ""}], "missing or non-opaque"),
        ([_record(), _record()], "duplicate record_id"),
    ],
)
def test_label_drift_rejects_missing_or_duplicate_record_ids(
    records: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_label_drift(records, [])


def test_wilson_interval_reports_observed_rate() -> None:
    interval = wilson_interval(2, 100)
    assert interval["observed"] == 0.02
    assert interval["lower"] <= 0.02 <= interval["upper"]
