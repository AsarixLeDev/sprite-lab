from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from spritelab.harvest.label_v4.assisted_v4_gui import (
    gui_mode_contract,
    load_assisted_records,
    require_quality_eligible_for_semantic,
    review_resume_index,
    review_resume_state,
)
from spritelab.harvest.label_v4.audit_prefill import bind_model_stage_proof, prepare_audit
from spritelab.harvest.label_v4.pixel_evidence import exact_rgba_content_hash
from spritelab.harvest.label_v4.review import (
    ReviewEvent,
    ReviewEventSchemaError,
    abstain_field,
    accept_model_abstention,
    accept_proposal,
    edit_field,
    load_review_events,
    mark_not_applicable,
    mark_unsupported,
    record_quality_decision,
)
from spritelab.harvest.label_v4.two_pass import (
    QualityResolution,
    audit_existing_events,
    calibration_denominator_report,
    freeze_inference_queue,
    quality_resume_state,
    resolve_quality_decisions,
    semantic_completion,
    semantic_readiness,
    validate_accept_all,
    validate_semantic_field,
    verify_frozen_inference_queue,
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


def _rehash_queue(root: Path, selection: Path, prefilled: Path, truth: Path) -> None:
    queue_path = root / "inference_queue.jsonl"
    rows = _rows(queue_path)
    base_rows = [{key: value for key, value in row.items() if key != "queue_id"} for row in rows]
    queue_id = _stable(
        {
            "identity_version": "label_v4_source_bound_queue_id_v2",
            "source_sha256": {
                "audit_selection": hashlib.sha256(selection.read_bytes()).hexdigest(),
                "prefilled_records": hashlib.sha256(prefilled.read_bytes()).hexdigest(),
                "human_truth": hashlib.sha256(truth.read_bytes()).hexdigest(),
            },
            "records": base_rows,
        }
    )
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


def _rehash_freeze(root: Path) -> None:
    freeze_path = root / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["files"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "freeze_manifest.json"
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_source_binding(root: Path, name: str, source: Path) -> None:
    manifest_path = root / "inference_queue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["source_bindings"][name] = {"path": str(source.resolve()), "sha256": digest}
    manifest[f"{name}_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_freeze(root)


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


def _prepare_bound_queue(
    queue: Path,
    output: Path,
    selection: Path,
    prefilled: Path,
    truth: Path,
    **kwargs: object,
) -> dict:
    return prepare_audit(
        queue,
        output,
        bound_audit_selection=selection,
        bound_prefilled_records=prefilled,
        bound_human_truth=truth,
        **kwargs,
    )


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
    image_hash = exact_rgba_content_hash(record["image_path"])
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
    bind_model_stage_proof(record)


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
    assert report["excluded_field_judgments"] == 0


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
    assert report["category_counts"]["valid_quality"] == 2
    assert report["categories_exhaustive"] is True
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
    assert report["category_counts"]["valid_quality"] == 0
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
    assert report["category_counts"]["valid_semantic"] == 0


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
    prefilled = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    freeze_inference_queue(selection, prefilled, truth, queue2)
    row = _rows(queue2 / "inference_queue.jsonl")[0]
    row["quality_state"] = "quality_unreviewed"
    (queue2 / "inference_queue.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_queue(queue2, selection, prefilled, truth)
    with pytest.raises(ValueError, match="bound source projection"):
        _prepare_bound_queue(
            queue2 / "inference_queue.jsonl",
            tmp_path / "unreviewed",
            selection,
            prefilled,
            truth,
            inference_policy="cached-only",
        )


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
    prefilled = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    freeze_inference_queue(selection, prefilled, truth, queue)
    source_row = _rows(queue / "inference_queue.jsonl")[0]
    manifest = _prepare_bound_queue(
        queue / "inference_queue.jsonl",
        tmp_path / "semantic",
        selection,
        prefilled,
        truth,
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


def test_tampered_quality_state_is_rejected_against_bound_sources(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (KEY,), cached=True)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    queue = tmp_path / "queue"
    prefilled = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    freeze_inference_queue(selection, prefilled, truth, queue)
    row = _rows(queue / "inference_queue.jsonl")[0]
    row["quality_state"] = "quality_unsuitable"
    row["quality_event"]["action"] = "quality_unsuitable"
    row["quality_event"]["human_outcome"] = "quality_unsuitable"
    (queue / "inference_queue.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_queue(queue, selection, prefilled, truth)
    with pytest.raises(ValueError, match="bound source projection"):
        _prepare_bound_queue(
            queue / "inference_queue.jsonl",
            tmp_path / "semantic",
            selection,
            prefilled,
            truth,
            inference_policy="cached-only",
        )


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


def test_missing_schema_quality_event_is_incompatible_and_never_advances_resume(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    raw = json.loads(truth.read_text(encoding="utf-8"))
    del raw["schema_version"]
    truth.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ReviewEventSchemaError, match="missing review-event schema_version"):
        ReviewEvent.from_dict(raw)
    with pytest.raises(ValueError, match="missing review-event schema_version"):
        load_review_events(truth)
    assert load_review_events(truth, strict=False) == ()
    assert quality_resume_state(records, [])["next_index"] == 0
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert report["category_counts"]["schema_incompatible"] == 1
    assert report["category_counts"]["valid_quality"] == 0


def test_missing_schema_semantic_event_is_incompatible_and_never_terminal(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    truth = tmp_path / "truth.jsonl"
    accept_proposal(truth, record, "canonical_object")
    raw = json.loads(truth.read_text(encoding="utf-8"))
    del raw["schema_version"]
    truth.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    assert load_review_events(truth, strict=False) == ()
    quality = QualityResolution(KEY, "quality_suitable", None, 1, 0)
    completion = semantic_completion(record, [], quality)
    assert completion["terminal_fields"] == []
    assert completion["complete"] is False
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert report["category_counts"]["schema_incompatible"] == 1
    assert report["category_counts"]["valid_semantic"] == 0


@pytest.mark.parametrize(
    ("case", "expected_category"),
    [
        ("identity_free", "identity_mismatch"),
        ("empty_proposal_hash", "proposal_hash_mismatch"),
        ("wrong_audit_record", "identity_mismatch"),
        ("wrong_sprite_id", "identity_mismatch"),
        ("wrong_field_identity", "identity_mismatch"),
    ],
)
def test_semantic_event_identity_bindings_are_exact_and_invalid_events_never_count(
    tmp_path: Path, case: str, expected_category: str
) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    truth = tmp_path / "truth.jsonl"
    for name in ("canonical_object", "category", "domain", "role"):
        accept_proposal(truth, record, name)
    raw_events = _rows(truth)
    forged = raw_events[0]
    if case == "identity_free":
        forged["session_id"] = ""
        forged["metadata"] = {}
    elif case == "empty_proposal_hash":
        forged["proposal_hash"] = ""
    elif case == "wrong_audit_record":
        forged["session_id"] = "wrong-audit"
        forged["metadata"]["audit_id"] = "wrong-audit"
        forged["metadata"]["audit_record_id"] = "wrong-audit"
    elif case == "wrong_sprite_id":
        forged["sprite_id"] = "foreign-sprite"
    else:
        forged["field_name"] = "foreign_field"
    truth.write_text("".join(json.dumps(item) + "\n" for item in raw_events), encoding="utf-8")

    events = load_review_events(truth)
    quality = QualityResolution(KEY, "quality_suitable", None, 1, 0)
    completion = semantic_completion(record, events, quality)
    assert completion["complete"] is False
    assert "canonical_object" not in completion["terminal_fields"]
    denominator = calibration_denominator_report([record], events)
    assert denominator["scorable_field_judgments"] == 0
    assert all(item["field"] != "canonical_object" for item in denominator["excluded"])
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert report["category_counts"][expected_category] == 1
    assert report["category_counts"]["valid_semantic"] == 3
    assert report["categories_exhaustive"] is True


def test_event_audit_uses_exhaustive_primary_categories(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    valid_path = tmp_path / "valid.jsonl"
    _quality(valid_path, records[0], "quality_suitable")
    valid = json.loads(valid_path.read_text(encoding="utf-8"))
    ignored = ReviewEvent(
        sprite_id=MINERAL,
        action="mark_suitable_image",
        proposal_hash=valid["proposal_hash"],
        session_id=records[0]["audit_id"],
        metadata={"audit_id": records[0]["audit_id"]},
    ).to_dict()
    truth = tmp_path / "audit-matrix.jsonl"
    truth.write_text("{malformed\n" + json.dumps(valid) + "\n" + json.dumps(ignored) + "\n", encoding="utf-8")
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert set(report["category_counts"]) == {
        "schema_incompatible",
        "identity_mismatch",
        "proposal_hash_mismatch",
        "malformed",
        "valid_quality",
        "valid_semantic",
        "ignored_non_authoritative",
    }
    assert report["category_counts"]["malformed"] == 1
    assert report["category_counts"]["valid_quality"] == 1
    assert report["category_counts"]["ignored_non_authoritative"] == 1
    assert report["categories_exhaustive"] is True


def test_quality_proposal_hash_mismatch_is_invalid_in_audit_and_resolver(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL,))
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    raw = json.loads(truth.read_text(encoding="utf-8"))
    raw["proposal_hash"] = "f" * 64
    truth.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    events = load_review_events(truth)
    resolution = resolve_quality_decisions(records, events)[MINERAL]
    assert resolution.effective_state == "quality_unreviewed"
    assert resolution.valid_event_count == 0
    report = audit_existing_events(tmp_path / "prepared" / "audit_prefilled_records.jsonl", truth)
    assert report["category_counts"]["proposal_hash_mismatch"] == 1
    assert report["category_counts"]["valid_quality"] == 0


def test_queue_verification_requires_and_reprojects_actual_bound_sources(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (MINERAL, KEY))
    prefilled = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    _quality(truth, records[1], "quality_unsuitable")
    queue = tmp_path / "queue"
    freeze_inference_queue(selection, prefilled, truth, queue)
    verified = verify_frozen_inference_queue(
        queue / "inference_queue.jsonl",
        audit_selection=selection,
        prefilled_records=prefilled,
        human_truth=truth,
    )
    assert verified["source_bindings_verified"] is True
    with pytest.raises(ValueError, match="requires actual source inputs"):
        verify_frozen_inference_queue(queue / "inference_queue.jsonl")

    copied_queue = tmp_path / "copied-queue"
    copied_selection = tmp_path / "copied-selection.jsonl"
    copied_prefilled = tmp_path / "copied-prefilled.jsonl"
    copied_truth = tmp_path / "copied-truth.jsonl"
    shutil.copytree(queue, copied_queue)
    shutil.copy2(selection, copied_selection)
    shutil.copy2(prefilled, copied_prefilled)
    shutil.copy2(truth, copied_truth)
    with pytest.raises(ValueError, match="source path mismatch"):
        verify_frozen_inference_queue(
            copied_queue / "inference_queue.jsonl",
            audit_selection=copied_selection,
            prefilled_records=copied_prefilled,
            human_truth=copied_truth,
        )


@pytest.mark.parametrize("source_name", ["audit_selection", "prefilled_records", "human_truth"])
def test_regenerated_local_source_manifest_cannot_rebind_queue(tmp_path: Path, source_name: str) -> None:
    selection, records = _prepared(tmp_path, (MINERAL, KEY))
    prefilled = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    _quality(truth, records[1], "quality_unsuitable")
    queue = tmp_path / "queue"
    freeze_inference_queue(selection, prefilled, truth, queue)
    forged_queue = tmp_path / f"forged-{source_name}"
    shutil.copytree(queue, forged_queue)
    sources = {
        "audit_selection": selection,
        "prefilled_records": prefilled,
        "human_truth": truth,
    }
    changed = tmp_path / f"changed-{source_name}.jsonl"
    changed.write_bytes(sources[source_name].read_bytes())
    if source_name == "human_truth":
        changed.write_text(changed.read_text(encoding="utf-8") + " \n", encoding="utf-8")
    else:
        rows = _rows(changed)
        rows[0]["adversarial_change"] = source_name
        changed.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _rewrite_source_binding(forged_queue, source_name, changed)
    sources[source_name] = changed
    with pytest.raises(ValueError, match=r"bound source projection|exactly match|incompatible schema"):
        verify_frozen_inference_queue(
            forged_queue / "inference_queue.jsonl",
            audit_selection=sources["audit_selection"],
            prefilled_records=sources["prefilled_records"],
            human_truth=sources["human_truth"],
        )


def test_queue_id_mismatch_is_rejected_even_when_local_manifest_is_rehashed(tmp_path: Path) -> None:
    selection, records = _prepared(tmp_path, (KEY,), cached=True)
    prefilled = tmp_path / "prepared" / "audit_prefilled_records.jsonl"
    truth = tmp_path / "truth.jsonl"
    _quality(truth, records[0], "quality_suitable")
    queue = tmp_path / "queue"
    freeze_inference_queue(selection, prefilled, truth, queue)
    manifest_path = queue / "inference_queue_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["queue_id"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze_path = queue / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["queue_id"] = "f" * 64
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_freeze(queue)
    with pytest.raises(ValueError, match="bound source projection"):
        verify_frozen_inference_queue(
            queue / "inference_queue.jsonl",
            audit_selection=selection,
            prefilled_records=prefilled,
            human_truth=truth,
        )


def test_resume_after_final_record_has_explicit_completed_state(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (MINERAL, KEY))
    truth = tmp_path / "truth.jsonl"
    for record in records:
        _quality(truth, record, "quality_suitable")
    events = load_review_events(truth)
    expected = {"next_index": None, "review_complete": True, "remaining": 0, "completed": 2, "total": 2}
    assert quality_resume_state(records, events) == expected
    assert review_resume_state(records, truth, mode="quality-only") == expected
    assert review_resume_index(records, truth, mode="quality-only") is None


def test_semantic_resume_after_final_record_is_explicitly_complete(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, record, "quality_suitable")
    for name in validate_accept_all(record):
        accept_proposal(truth, record, name)
    state = review_resume_state(records, truth, mode="semantic-assisted")
    assert state == {"next_index": None, "review_complete": True, "remaining": 0, "completed": 1, "total": 1}
    assert review_resume_index(records, truth, mode="semantic-assisted") is None


def test_rejected_model_abstention_correction_supersedes_without_acceptance(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    _valid_abstention(record)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, record, "quality_suitable")
    correction = edit_field(truth, record, "canonical_object", "corrected-key")
    for name in ("category", "domain", "role"):
        accept_proposal(truth, record, name)
    events = load_review_events(truth)
    quality = resolve_quality_decisions([record], events)[KEY]
    completion = semantic_completion(record, events, quality)
    assert correction.human_outcome == "incorrect"
    assert completion["complete"] is True
    assert not any(reason.startswith("unjudged_model_abstention") for reason in completion["reasons"])
    assert len(events) == 5
    report = calibration_denominator_report([record], events)
    assert report["model_abstention_appropriateness_judgments"] == 0
    assert report["model_abstention_rejected_with_correction_judgments"] == 1
    assert all(item["field"] != "canonical_object" for item in report["scorable"])


@pytest.mark.parametrize(
    ("outcome", "counter"),
    [
        ("human_abstained", "human_abstention_judgments"),
        ("unsupported", "unsupported_judgments"),
        ("not_applicable", "not_applicable_judgments"),
    ],
)
def test_terminal_non_value_outcomes_never_enter_semantic_value_denominator(
    tmp_path: Path, outcome: str, counter: str
) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    truth = tmp_path / "truth.jsonl"
    _quality(truth, record, "quality_suitable")
    if outcome == "human_abstained":
        abstain_field(truth, record, "canonical_object")
    elif outcome == "unsupported":
        mark_unsupported(truth, record, "canonical_object")
    else:
        mark_not_applicable(truth, record, "canonical_object")
    for name in ("category", "domain", "role"):
        accept_proposal(truth, record, name)
    events = load_review_events(truth)
    quality = resolve_quality_decisions([record], events)[KEY]
    assert semantic_completion(record, events, quality)["complete"] is True
    report = calibration_denominator_report([record], events)
    assert report["scorable_field_judgments"] == 3
    assert report[counter] == 1
    assert report["non_value_outcome_counts"][outcome] == 1
    assert all(item["field"] != "canonical_object" for item in report["scorable"])


def test_internally_rebound_wrong_exported_rgba_hash_fails_abstention_proof(tmp_path: Path) -> None:
    _selection_path, records = _prepared(tmp_path, (KEY,), cached=True)
    record = records[0]
    _material_not_required(record)
    _valid_abstention(record)
    assert semantic_readiness(record)[0] is True
    forged = copy.deepcopy(record)
    forged["exported_rgba_hash"] = "f" * 64
    bind_model_stage_proof(forged)
    ready, reasons = semantic_readiness(forged)
    assert ready is False
    assert any("exported_rgba_hash_does_not_match_image" in reason for reason in reasons)
