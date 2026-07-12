from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from spritelab.harvest.label_v4.assisted_v4_gui import (
    gui_mode_contract,
    load_assisted_records,
    require_quality_eligible_for_semantic,
    review_resume_index,
)
from spritelab.harvest.label_v4.audit_prefill import prepare_audit
from spritelab.harvest.label_v4.review import (
    ReviewEvent,
    accept_model_abstention,
    accept_proposal,
    load_review_events,
    record_quality_decision,
)
from spritelab.harvest.label_v4.two_pass import (
    QualityResolution,
    audit_existing_events,
    calibration_denominator_report,
    freeze_inference_queue,
    resolve_quality_decisions,
    semantic_completion,
    semantic_readiness,
    validate_accept_all,
    validate_semantic_field,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiments" / "label_v4_calibration_wave1" / "audit_manifest.jsonl"
PILOT = ROOT / "experiments" / "label_v4_real_pilot_15_v1"
REPLAY = ROOT / "experiments" / "label_v4_pilot_replay_v2"
MINERAL = "acq_craftpix_minerals_icon29"
KEY = "oga_cc0_key_rcorre_key_01"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _rehash_queue(root: Path) -> None:
    queue_path = root / "inference_queue.jsonl"
    rows = _rows(queue_path)
    queue_id = _stable([{key: value for key, value in row.items() if key != "queue_id"} for row in rows])
    for row in rows:
        row["queue_id"] = queue_id
    queue_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_path = root / "inference_queue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "queue_id": queue_id,
            "records": len(rows),
            "ordered_sprite_ids_hash": _stable([row["sprite_id"] for row in rows]),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_path = root / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["queue_id"] = queue_id
    freeze["files"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "freeze_manifest.json"
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selection(tmp_path: Path, ids: tuple[str, ...]) -> Path:
    index = {row["sprite_id"]: row for row in _rows(AUDIT)}
    path = tmp_path / "selection.jsonl"
    path.write_text("".join(json.dumps(index[sprite_id]) + "\n" for sprite_id in ids), encoding="utf-8")
    return path


def _prepared(tmp_path: Path, ids: tuple[str, ...], *, cached: bool = False) -> tuple[Path, list[dict]]:
    selection = _selection(tmp_path, ids)
    output = tmp_path / "prepared"
    prepare_audit(
        selection,
        output,
        inference_policy="cached-only" if cached else "deterministic-only",
        artifact_roots=[PILOT, REPLAY] if cached else [],
    )
    return selection, _rows(output / "audit_prefilled_records.jsonl")


def _quality(path: Path, record: dict, outcome: str) -> None:
    record_quality_decision(
        path,
        record,
        outcome,
        session_id=record["audit_id"],
        metadata={"audit_id": record["audit_id"], "review_mode": "quality_only"},
    )


def _material_not_required(record: dict) -> None:
    record["fields"]["explicit_material"]["applicability"] = {
        "state": "not_required",
        "reason": "explicit_axis_not_applicable",
        "provenance": "test_taxonomy_axis",
    }


def _valid_abstention(record: dict, field_name: str = "canonical_object", *, repaired: bool = False) -> None:
    image_hash = "a" * 64
    status = "success_after_json_repair" if repaired else "success"
    record["model_provenance"]["image_hash"] = image_hash
    record["model_provenance"]["stage_outcomes"] = [
        {"stage": "B_blind_vlm_proposal", "stage_status": "success", "fallback_used": False},
        {"stage": "C_text_reconciliation", "stage_status": status, "fallback_used": False},
    ]
    record["model_provenance"]["stage_ledger"] = [
        {
            "stage": "B_blind_vlm_proposal",
            "image_hash": image_hash,
            "provider_output_valid": True,
            "failure_diagnostics": {},
        },
        {
            "stage": "C_text_reconciliation",
            "image_hash": image_hash,
            "provider_output_valid": not repaired,
            "failure_diagnostics": {"error_type": "invalid_json"} if repaired else {},
        },
    ]
    update = {
        "value": None,
        "value_state": "model_abstained",
        "reason": "model_stage_completed_without_promoted_value",
    }
    record["fields"][field_name].update(update)
    record["field_proposals"][field_name].update(update)


def test_quality_only_contract_has_no_semantic_controls() -> None:
    contract = gui_mode_contract("quality-only")
    assert contract["semantic_controls_present"] is False
    assert contract["banner"].startswith("QUALITY REVIEW ONLY")


def test_quality_actions_append_advance_resume_and_have_explicit_outcomes(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL, KEY))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    event = load_review_events(truth)[0]
    assert event.human_outcome == "quality_suitable"
    assert event.human_outcome != "not_applicable"
    assert review_resume_index(records, truth, mode="quality-only") == 1
    _quality(truth, records[1], "quality_uncertain_usable")
    states = resolve_quality_decisions(records, load_review_events(truth))
    assert states[MINERAL].effective_state == "quality_suitable"
    assert states[KEY].effective_state == "quality_uncertain_usable"


def test_uncertain_usable_and_unusable_are_distinct_and_latest_valid_event_wins(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_uncertain_usable")
    _quality(truth, records[0], "quality_uncertain_not_usable")
    resolution = resolve_quality_decisions(records, load_review_events(truth))[MINERAL]
    assert resolution.effective_state == "quality_uncertain_not_usable"
    assert resolution.valid_event_count == 2


def test_freeze_is_immutable_ordered_and_quality_eligible_only(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (MINERAL, KEY))
    prefilled = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    _quality(truth, records[1], "quality_uncertain_not_usable")
    output = tmp_path / "queue"
    result = freeze_inference_queue(selection, prefilled, truth, output)
    queue = _rows(output / "inference_queue.jsonl")
    assert result["included"] == 1 and queue[0]["sprite_id"] == MINERAL
    assert queue[0]["audit_order"] == 0 and queue[0]["quality_risk_penalty"] == 0.0
    assert json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))["frozen"] is True
    with pytest.raises(FileExistsError, match="immutable inference queue"):
        freeze_inference_queue(selection, prefilled, truth, output)


def test_incomplete_quality_review_blocks_freeze_without_partial_flag(tmp_path: Path) -> None:
    selection, _records = _prepared(tmp_path, (MINERAL, KEY))
    truth = tmp_path / "truth.jsonl"
    truth.touch()
    with pytest.raises(ValueError, match="quality review incomplete for 2 records"):
        freeze_inference_queue(
            selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, tmp_path / "queue"
        )


def test_missing_prediction_rejected_by_semantic_gui_and_cannot_be_accepted(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    path = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    with pytest.raises(ValueError, match=f"Record {MINERAL} is not semantic-review ready"):
        load_assisted_records(path, mode="semantic-assisted")
    with pytest.raises(ValueError, match="forbidden for value_state=missing_prediction"):
        accept_proposal(tmp_path / "truth.jsonl", records[0], "canonical_object")


def test_semantic_assisted_requires_prior_eligible_quality(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    with pytest.raises(ValueError, match="eligible quality decisions"):
        require_quality_eligible_for_semantic(records, [])
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    require_quality_eligible_for_semantic(records, load_review_events(truth))


def test_genuine_model_abstention_has_distinct_acceptance(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = copy.deepcopy(records[0])
    _valid_abstention(record)
    event = accept_model_abstention(tmp_path / "truth.jsonl", record, "canonical_object")
    assert event.action == "accept_model_abstention"
    assert event.reviewed_value is None
    assert event.human_outcome == "model_abstention_accepted"


def test_accept_all_and_semantic_completion_refuse_unresolved_critical_fields(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    with pytest.raises(ValueError, match="accept all refused"):
        validate_accept_all(records[0])
    quality = QualityResolution(MINERAL, "quality_suitable", None, 1, 0)
    completion = semantic_completion(records[0], [], quality)
    assert completion["complete"] is False
    assert "missing_required_model_stage" in completion["reasons"]


def test_semantic_completion_requires_and_accepts_all_terminal_critical_judgments(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, record, "quality_suitable")
    required = validate_accept_all(record)
    for name in required:
        accept_proposal(truth, record, name)
    events = load_review_events(truth)
    quality = resolve_quality_decisions(records, events)[KEY]
    assert semantic_completion(record, events, quality)["complete"] is True


def test_denominator_excludes_missing_null_incomplete_and_quality_ineligible(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    unsafe = ReviewEvent(
        sprite_id=MINERAL,
        action="accept_proposed_value",
        field_name="canonical_object",
        proposed_value=None,
        reviewed_value=None,
        human_outcome="correct",
        proposal_hash="",
    )
    report = calibration_denominator_report(records, [unsafe])
    assert report["missing_prediction_records"] == 1
    assert report["scorable_field_judgments"] == 0
    assert report["excluded_field_judgments"] == 1


def test_blind_semantic_mode_requires_real_proposal(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    record = records[0]
    record["prediction_state"] = "complete_deterministic"
    record["missing_stages"] = []
    for name in ("canonical_object", "category", "domain", "role"):
        record["fields"][name].update({"value": name, "value_state": "known", "reason": "test"})
        record["field_proposals"][name].update({"value": name, "value_state": "known", "reason": "test"})
    record["review_mode"] = "blind"
    _material_not_required(record)
    path = tmp_path / "blind.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot use blind review without a real proposal"):
        load_assisted_records(path, mode="semantic-assisted")


def test_event_audit_preserves_append_history_and_reports_incomplete(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    _quality(truth, records[0], "quality_unsuitable")
    before = truth.read_bytes()
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert report["category_counts"]["valid_quality_event"] == 2
    assert report["incomplete_count"] == 0
    assert truth.read_bytes() == before


def test_mineral_is_quality_only_and_no_provider_calls_occur(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("provider call"))
    )
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    record = records[0]
    assert record["source_suitability"]["status"] == "quarantine"
    assert record["source_suitability"]["reason_codes"] == ["LARGE_PALETTE"]
    assert record["prediction_state"] == "missing_required_model_stage"
    assert record["fields"]["canonical_object"]["value_state"] == "missing_prediction"
    assert gui_mode_contract("quality-only")["semantic_controls_present"] is False


def test_wrong_schema_quality_event_fails_closed_and_does_not_resume(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    valid = load_review_events(truth)[0]
    raw = valid.to_dict()
    raw["schema_version"] = "wrong_event_schema"
    truth.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported review-event schema"):
        load_review_events(truth)
    assert load_review_events(truth, strict=False) == ()
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert report["category_counts"]["schema_incompatible"] == 1
    assert report["category_counts"]["valid_quality_event"] == 0
    assert report["incomplete_count"] == 1
    assert review_resume_index(records, truth, mode="quality-only") == 0
    bad = copy.deepcopy(valid)
    object.__setattr__(bad, "schema_version", "wrong_event_schema")
    resolution = resolve_quality_decisions(records, [bad])[MINERAL]
    assert resolution.effective_state == "quality_unreviewed"
    assert resolution.valid_event_count == 0 and resolution.ignored_event_count == 1


def test_wrong_schema_semantic_event_never_completes_a_field(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    event = ReviewEvent(
        sprite_id=KEY,
        action="accept_proposed_value",
        field_name="canonical_object",
        human_outcome="correct",
    )
    object.__setattr__(event, "schema_version", "future_semantic_schema")
    quality = QualityResolution(KEY, "quality_suitable", None, 1, 0)
    completion = semantic_completion(record, [event], quality)
    assert "canonical_object" not in completion["terminal_fields"]
    assert completion["complete"] is False
    raw = event.to_dict()
    raw["schema_version"] = "future_semantic_schema"
    truth = tmp_path / "wrong-semantic.jsonl"
    truth.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert report["category_counts"]["schema_incompatible"] == 1
    assert report["category_counts"]["valid_semantic_event"] == 0


def test_partial_and_complete_queue_manifests_have_explicit_finality(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (MINERAL, KEY))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    partial = tmp_path / "partial"
    freeze_inference_queue(
        selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, partial, allow_partial=True
    )
    for name in ("inference_queue_manifest.json", "freeze_manifest.json"):
        manifest = json.loads((partial / name).read_text(encoding="utf-8"))
        assert manifest["allow_partial"] is True
        assert manifest["quality_review_complete"] is False
        assert manifest["queue_status"] == "partial_nonfinal"
        assert manifest["eligible_for_semantic_preparation"] is False
        assert manifest["total_input_records"] == 2
        assert manifest["reviewed_records"] == 1 and manifest["unreviewed_records"] == 1
        assert len(manifest["unreviewed_ids_sha256"]) == 64
    _quality(truth, records[1], "quality_unsuitable")
    complete = tmp_path / "complete"
    freeze_inference_queue(selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, complete)
    for name in ("inference_queue_manifest.json", "freeze_manifest.json"):
        manifest = json.loads((complete / name).read_text(encoding="utf-8"))
        assert manifest["allow_partial"] is False
        assert manifest["quality_review_complete"] is True
        assert manifest["queue_status"] == "final"
        assert manifest["eligible_for_semantic_preparation"] is True


def test_queue_rebuilds_are_byte_identical(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (MINERAL, KEY))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    _quality(truth, records[1], "quality_unsuitable")
    first, second = tmp_path / "queue-a", tmp_path / "queue-b"
    for output in (first, second):
        freeze_inference_queue(selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, output)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_partial_missing_and_tampered_queues_are_rejected_before_provider_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("provider call"))
    )
    selection, records = _prepared(tmp_path, (MINERAL, KEY))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    partial = tmp_path / "partial"
    freeze_inference_queue(
        selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, partial, allow_partial=True
    )
    with pytest.raises(ValueError, match="nonfinal"):
        prepare_audit(partial / "inference_queue.jsonl", tmp_path / "partial_out", inference_policy="cached-only")
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "inference_queue.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires sibling"):
        prepare_audit(missing / "inference_queue.jsonl", tmp_path / "missing_out", inference_policy="cached-only")
    _quality(truth, records[1], "quality_unsuitable")
    complete = tmp_path / "complete"
    freeze_inference_queue(selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, complete)
    with (complete / "inference_queue.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(" \n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_audit(complete / "inference_queue.jsonl", tmp_path / "tampered_out", inference_policy="cached-only")


def test_queue_with_omitted_finality_or_unreviewed_row_is_rejected(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (KEY,), cached=True)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    queue = tmp_path / "queue"
    freeze_inference_queue(selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, queue)
    manifest_path = queue / "inference_queue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["queue_status"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = json.loads((queue / "freeze_manifest.json").read_text(encoding="utf-8"))
    freeze["files"][manifest_path.name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (queue / "freeze_manifest.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing finality field"):
        prepare_audit(queue / "inference_queue.jsonl", tmp_path / "no-finality", inference_policy="cached-only")

    queue2 = tmp_path / "queue2"
    freeze_inference_queue(selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, queue2)
    row = _rows(queue2 / "inference_queue.jsonl")[0]
    row["quality_state"] = "quality_unreviewed"
    (queue2 / "inference_queue.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_queue(queue2)
    with pytest.raises(ValueError, match="unknown or unreviewed quality state"):
        prepare_audit(queue2 / "inference_queue.jsonl", tmp_path / "unreviewed", inference_policy="cached-only")


def test_queue_identity_and_partition_integrity_survive_semantic_prefill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("provider call"))
    )
    selection, records = _prepared(tmp_path, (KEY,), cached=True)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    queue = tmp_path / "queue"
    freeze_inference_queue(selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, queue)
    source_row = _rows(queue / "inference_queue.jsonl")[0]
    manifest = prepare_audit(
        queue / "inference_queue.jsonl",
        tmp_path / "semantic",
        inference_policy="cached-only",
        artifact_roots=[PILOT, REPLAY],
    )
    outputs = []
    for name in (
        "semantic_ready_records.jsonl",
        "semantic_pending_inference.jsonl",
        "semantic_failed_records.jsonl",
        "semantic_excluded_quality.jsonl",
    ):
        outputs.extend(_rows(tmp_path / "semantic" / name))
    assert len(outputs) == 1
    for field in (
        "queue_id",
        "audit_order",
        "prefill_record_hash",
        "audit_id",
        "sprite_id",
        "exported_rgba_hash",
        "quality_state",
        "quality_event",
    ):
        assert outputs[0][field] == source_row[field]
    integrity = manifest["partition_integrity"]
    assert integrity["disjoint"] is True and integrity["exhaustive"] is True
    assert sum(integrity["partition_counts"].values()) == 1
    assert manifest["queue_id"] == source_row["queue_id"]
    assert manifest["queue_file_sha256"] == hashlib.sha256((queue / "inference_queue.jsonl").read_bytes()).hexdigest()


def test_quality_unsuitable_queue_row_is_excluded_and_never_ready(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (KEY,), cached=True)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    queue = tmp_path / "queue"
    freeze_inference_queue(selection, tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth, queue)
    row = _rows(queue / "inference_queue.jsonl")[0]
    row["quality_state"] = "quality_unsuitable"
    row["quality_event"]["action"] = "quality_unsuitable"
    row["quality_event"]["human_outcome"] = "quality_unsuitable"
    (queue / "inference_queue.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_queue(queue)
    manifest = prepare_audit(queue / "inference_queue.jsonl", tmp_path / "semantic", inference_policy="cached-only")
    assert _rows(tmp_path / "semantic" / "semantic_ready_records.jsonl") == []
    excluded = _rows(tmp_path / "semantic" / "semantic_excluded_quality.jsonl")
    assert len(excluded) == 1 and excluded[0]["quality_state"] == "quality_unsuitable"
    assert manifest["partition_integrity"]["partition_counts"]["excluded"] == 1


@pytest.mark.parametrize(
    ("state", "value", "ready"),
    [
        ("known", "sword", True),
        ("known", 0, True),
        ("known", False, True),
        ("known", None, False),
        ("known", "   ", False),
        ("known", [], False),
        ("missing_prediction", None, False),
        ("provider_failed", None, False),
        ("not_scorable", None, False),
        ("unsupported", None, False),
        ("not_applicable", None, False),
    ],
)
def test_strict_semantic_state_matrix(tmp_path: Path, state: str, value: object, ready: bool) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    record["fields"]["canonical_object"].update({"value_state": state, "value": value})
    assert semantic_readiness(record)[0] is ready
    assert validate_semantic_field(record, "canonical_object").valid is ready


def test_model_abstention_requires_bound_terminal_stage_and_acceptance_can_complete(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    fake = copy.deepcopy(records[0])
    _material_not_required(fake)
    fake["fields"]["canonical_object"].update(
        {"value": None, "value_state": "model_abstained", "reason": "model_stage_completed_without_promoted_value"}
    )
    assert semantic_readiness(fake)[0] is False
    with pytest.raises(ValueError, match="requires valid model_abstained"):
        accept_model_abstention(tmp_path / "fake.jsonl", fake, "canonical_object")

    record = records[0]
    _material_not_required(record)
    _valid_abstention(record, repaired=True)
    assert semantic_readiness(record)[0] is True
    truth = tmp_path / "truth.jsonl"
    _quality(truth, record, "quality_suitable")
    accept_model_abstention(truth, record, "canonical_object")
    for name in ("category", "domain", "role"):
        accept_proposal(truth, record, name)
    events = load_review_events(truth)
    quality = resolve_quality_decisions([record], events)[KEY]
    assert semantic_completion(record, events, quality)["complete"] is True
    report = calibration_denominator_report([record], events)
    assert report["model_abstention_appropriateness_judgments"] == 1
    assert report["scorable_field_judgments"] == 3
    assert all(item["field"] != "canonical_object" for item in report["scorable"])


def test_missing_material_is_unresolved_not_automatically_not_applicable(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    material = record["fields"]["explicit_material"]
    assert material["value"] is None
    assert material["value_state"] != "not_applicable"
    ready, reasons = semantic_readiness(record)
    assert ready is False
    assert "explicit_material:material_applicability_not_established" in reasons
    quality = QualityResolution(KEY, "quality_suitable", None, 1, 0)
    assert semantic_completion(record, [], quality)["complete"] is False
