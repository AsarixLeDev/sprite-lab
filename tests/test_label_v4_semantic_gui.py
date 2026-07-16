from __future__ import annotations

import copy
import hashlib
import inspect
import json
import socket
import struct
from pathlib import Path

import pytest
from PIL import Image

from spritelab.harvest.label_v4.assisted_v4_gui import (
    build_semantic_gui,
    field_action_availability,
    gui_mode_contract,
    preview_field_correction,
)
from spritelab.harvest.label_v4.audit_prefill import PREFILL_FIELDS, bind_model_stage_proof
from spritelab.harvest.label_v4.pixel_evidence import exact_rgba_content_hash
from spritelab.harvest.label_v4.review import (
    accept_model_abstention,
    accept_proposal,
    edit_field,
    load_review_events,
    parse_field_correction,
    save_field_correction,
)
from spritelab.harvest.label_v4.two_pass import (
    QualityResolution,
    semantic_field_progress,
    validate_semantic_field,
)


def _field(value: object, state: str = "known", *, shape: str | None = None) -> dict:
    result = {
        "value": value,
        "value_state": state,
        "reason": "normalized_available_evidence" if state == "known" else "no_supported_value_for_optional_field",
        "alternatives": [],
        "evidence": ["vlm_visual"],
        "uncertainty_1_20": 8,
    }
    if shape:
        result["value_shape"] = shape
    return result


def _synthetic_record(tmp_path: Path, sprite_id: str = "synthetic_bracers") -> dict:
    image_path = tmp_path / f"{sprite_id}.png"
    Image.new("RGBA", (8, 8), (120, 90, 65, 255)).save(image_path)
    with Image.open(image_path) as source:
        rgba = source.convert("RGBA")
        exported = b"spritelab-exported-rgba-v1\0" + struct.pack(">II", rgba.width, rgba.height) + rgba.tobytes()
    fields = {
        "canonical_object": _field(None, "model_abstained"),
        "category": _field("armor"),
        "domain": _field("equipment_icon"),
        "role": _field("wearable_accessory"),
        "explicit_material": _field(None, "not_applicable"),
        "surface_alias": _field("brown bracers"),
        "filename_color_hints": _field(["brown"]),
        "palette_colors": _field(["gray", "brown"]),
        "primary_colors": _field(["brown"]),
        "secondary_colors": _field(["gray"]),
        "outline_colors": _field(["dark_brown"]),
        "highlight_colors": _field(["tan"]),
        "shadow_colors": _field(["dark_brown"]),
        "size_hint": _field("compact"),
        "condition": _field(["intact"]),
        "shape": _field({"silhouette": ["paired"]}),
        "visual_form": _field(["pair"]),
        "parts": _field(["left_bracer", "right_bracer"]),
        "description": _field("A visually recognizable pair of bracers."),
    }
    fields["canonical_object"].update(
        {"reason": "model_stage_completed_without_promoted_value", "uncertainty_1_20": 17}
    )
    fields["explicit_material"]["applicability"] = {
        "state": "not_required",
        "reason": "explicit_axis_not_applicable",
        "provenance": "synthetic_demo_contract",
    }
    record = {
        "schema_version": "label_v4_prefilled_audit_record_v1",
        "audit_id": f"audit-{sprite_id}",
        "sprite_id": sprite_id,
        "image_path": str(image_path),
        "exported_rgba_hash": hashlib.sha256(exported).hexdigest(),
        "fields": copy.deepcopy(fields),
        "field_proposals": copy.deepcopy(fields),
        "label_quality": {"fields": {name: {} for name in fields}},
        "prediction_state": "completed_valid",
        "prediction_origin": "semantic_minimal_provider",
        "missing_stages": [],
        "review_mode": "assisted",
        "quality_state": "quality_suitable",
        "model_provenance": {},
    }
    image_hash = exact_rgba_content_hash(image_path)
    record["model_provenance"] = {
        "image_hash": image_hash,
        "sprite_id": sprite_id,
        "stage_outcomes": [
            {"stage": "B_blind_vlm_proposal", "stage_status": "success", "fallback_used": False},
            {"stage": "C_text_reconciliation", "stage_status": "success", "fallback_used": False},
        ],
        "stage_ledger": [
            {
                "stage": "B_blind_vlm_proposal",
                "image_hash": image_hash,
                "provider_output_valid": True,
                "failure_diagnostics": {},
            },
            {
                "stage": "C_text_reconciliation",
                "image_hash": image_hash,
                "provider_output_valid": True,
                "failure_diagnostics": {},
            },
        ],
    }
    bind_model_stage_proof(record)
    return record


def _context(record: dict, *, token: str = "submission-1", field: str = "canonical_object") -> dict:
    return {
        "displayed_field": field,
        "sprite_id": record["sprite_id"],
        "audit_record_id": record["audit_id"],
        "submission_token": token,
        "review_mode": "semantic_assisted",
        "quality": QualityResolution(record["sprite_id"], "quality_suitable", None, 0, 0),
    }


def test_model_abstained_canonical_object_correction_is_one_terminal_event(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    result = save_field_correction(record, "canonical_object", " bracers ", corrections, _context(record))
    events = load_review_events(corrections)
    assert len(events) == 1
    assert events[0].reviewed_value == "bracers"
    assert events[0].metadata["audit_record_id"] == record["audit_id"]
    assert events[0].metadata["proposal_input_hash"] == record["model_provenance"]["image_hash"]
    assert events[0].metadata["exported_rgba_hash"] == record["exported_rgba_hash"]
    assert events[0].review_mode == "semantic_assisted"
    assert result.parsed_value == "bracers"
    assert result.human_outcome == "incorrect"
    assert "canonical_object" in result.record_completion_state["terminal_fields"]
    assert 'Saved canonical_object = "bracers"' in result.message


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_scalar_correction_writes_nothing_and_has_visible_error(tmp_path: Path, raw: str) -> None:
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    with pytest.raises(ValueError, match="canonical_object requires a nonempty string value"):
        save_field_correction(record, "canonical_object", raw, corrections, _context(record))
    assert load_review_events(corrections) == ()
    assert "Nothing will be saved" in preview_field_correction(record, "canonical_object", raw)


def test_scalar_and_list_parsing_and_invalid_json(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    assert parse_field_correction(record, "canonical_object", " bracers ") == "bracers"
    assert parse_field_correction(record, "primary_colors", '["gray", "brown"]') == ["gray", "brown"]
    assert parse_field_correction(record, "primary_colors", "gray, brown") == ["gray", "brown"]
    with pytest.raises(ValueError, match="requires valid JSON"):
        parse_field_correction(record, "primary_colors", '["gray"')


def test_numeric_zero_and_boolean_false_are_preserved(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    record["fields"]["synthetic_score"] = _field(1, shape="integer")
    record["field_proposals"]["synthetic_score"] = copy.deepcopy(record["fields"]["synthetic_score"])
    record["fields"]["synthetic_flag"] = _field(True, shape="boolean")
    record["field_proposals"]["synthetic_flag"] = copy.deepcopy(record["fields"]["synthetic_flag"])
    bind_model_stage_proof(record)
    assert parse_field_correction(record, "synthetic_score", "0") == 0
    assert parse_field_correction(record, "synthetic_flag", "false") is False


def test_stale_field_selection_cannot_save_text_to_new_field(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    with pytest.raises(ValueError, match="stale field selection"):
        save_field_correction(
            record,
            "category",
            "bracers",
            corrections,
            _context(record, field="canonical_object"),
        )
    assert load_review_events(corrections) == ()


def test_double_click_submission_token_is_idempotent(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    first = save_field_correction(record, "canonical_object", "bracers", corrections, _context(record))
    second = save_field_correction(record, "canonical_object", "bracers", corrections, _context(record))
    assert len(load_review_events(corrections)) == 1
    assert first.event_id == second.event_id
    assert second.duplicate_submission is True
    with pytest.raises(ValueError, match="submission token was already used"):
        save_field_correction(record, "canonical_object", "gauntlets", corrections, _context(record))
    assert len(load_review_events(corrections)) == 1


def test_rejected_abstention_then_correction_is_authoritative(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    accept_model_abstention(
        corrections,
        record,
        "canonical_object",
        session_id=record["audit_id"],
        metadata={"audit_id": record["audit_id"]},
    )
    result = save_field_correction(record, "canonical_object", "bracers", corrections, _context(record, token="next"))
    progress = semantic_field_progress(
        record,
        load_review_events(corrections),
        QualityResolution(record["sprite_id"], "quality_suitable", None, 0, 0),
    )
    canonical = next(row for row in progress["fields"] if row["field"] == "canonical_object")
    assert result.human_outcome == "incorrect"
    assert canonical["status"] == "corrected"
    assert len(load_review_events(corrections)) == 2


def test_abstention_is_separate_and_missing_prediction_actions_fail_closed(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    assert field_action_availability(record, "canonical_object")["model_abstention"] is True
    event = accept_model_abstention(
        tmp_path / "events.jsonl",
        record,
        "canonical_object",
        session_id=record["audit_id"],
        metadata={"audit_id": record["audit_id"]},
    )
    assert event.human_outcome == "model_abstention_accepted"
    missing = copy.deepcopy(record)
    missing["fields"]["canonical_object"].update(
        {"value": None, "value_state": "missing_prediction", "reason": "rich_vlm_stage_not_executed"}
    )
    missing["field_proposals"]["canonical_object"].update(missing["fields"]["canonical_object"])
    assert field_action_availability(missing, "canonical_object")["accept"] is False
    assert field_action_availability(missing, "canonical_object")["model_abstention"] is False
    with pytest.raises(ValueError, match="forbidden for value_state=missing_prediction"):
        accept_proposal(tmp_path / "missing.jsonl", missing, "canonical_object")


def test_progress_updates_next_field_and_completion(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    save_field_correction(record, "canonical_object", "bracers", corrections, _context(record))
    quality = QualityResolution(record["sprite_id"], "quality_suitable", None, 0, 0)
    progress = semantic_field_progress(record, load_review_events(corrections), quality)
    assert progress["next_unresolved_field"] == "category"
    assert progress["required_reviewed"] == 1
    for name in ("category", "domain", "role"):
        edit_field(
            corrections,
            record,
            name,
            record["fields"][name]["value"],
            session_id=record["audit_id"],
            metadata={"audit_id": record["audit_id"]},
        )
    complete = semantic_field_progress(record, load_review_events(corrections), quality)
    assert complete["record_complete"] is True
    assert complete["next_unresolved_field"] is None


def test_gradio_callback_contract_has_exact_seven_inputs_and_outputs(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    record = _synthetic_record(tmp_path)
    demo = build_semantic_gui(
        gr,
        [record],
        tmp_path / "events.jsonl",
        0,
        {},
        gui_mode_contract("semantic-assisted"),
        "semantic_assisted",
    )
    dependency = next(item for item in demo.config["dependencies"] if item.get("api_name") == "save_field_correction")
    assert len(dependency["inputs"]) == 7
    assert len(dependency["outputs"]) == 28
    fn = demo.fns[dependency["id"]].fn
    assert len(inspect.signature(fn).parameters) == 7
    assert "selected" not in inspect.signature(fn).parameters
    demo.close()


def test_gui_callback_surfaces_empty_error_and_appends_nothing(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    demo = build_semantic_gui(
        gr,
        [record],
        corrections,
        0,
        {},
        gui_mode_contract("semantic-assisted"),
        "semantic_assisted",
    )
    dependency = next(item for item in demo.config["dependencies"] if item.get("api_name") == "save_field_correction")
    callback = demo.fns[dependency["id"]].fn
    result = callback(0, "canonical_object", None, "   ", 12, "empty-click", "canonical_object")
    assert result[0].startswith("Nothing was saved.")
    assert "canonical_object requires a nonempty string value" in result[0]
    assert load_review_events(corrections) == ()
    demo.close()


def test_complete_record_enables_next_record_button(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    first = _synthetic_record(tmp_path, "first_complete")
    second = _synthetic_record(tmp_path, "second_pending")
    corrections = tmp_path / "events.jsonl"
    save_field_correction(first, "canonical_object", "bracers", corrections, _context(first))
    for name in ("category", "domain", "role"):
        edit_field(
            corrections,
            first,
            name,
            first["fields"][name]["value"],
            session_id=first["audit_id"],
            metadata={"audit_id": first["audit_id"]},
        )
    demo = build_semantic_gui(
        gr,
        [first, second],
        corrections,
        0,
        {},
        gui_mode_contract("semantic-assisted"),
        "semantic_assisted",
    )
    load_dependency = demo.config["dependencies"][0]
    rendered = demo.fns[load_dependency["id"]].fn()
    assert rendered[-1]["interactive"] is True
    assert rendered[-1]["value"] == "Next record"
    demo.close()


@pytest.mark.filterwarnings("ignore:.*HTTP_422_UNPROCESSABLE_ENTITY.*:Warning")
@pytest.mark.filterwarnings("ignore:.*asyncio.iscoroutinefunction.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*future.no_silent_downcasting.*:Warning")
@pytest.mark.filterwarnings("ignore:.*copy keyword is deprecated.*:Warning")
def test_local_gradio_correction_endpoint_has_no_http_422_and_writes_one_event(tmp_path: Path) -> None:
    gr = pytest.importorskip("gradio")
    client_module = pytest.importorskip("gradio_client")
    record = _synthetic_record(tmp_path)
    corrections = tmp_path / "events.jsonl"
    demo = build_semantic_gui(
        gr,
        [record],
        corrections,
        0,
        {},
        gui_mode_contract("semantic-assisted"),
        "semantic_assisted",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            quiet=True,
            css=".sprite-pixel-viewer img{image-rendering:pixelated!important}",
        )
        client = client_module.Client(f"http://127.0.0.1:{port}", verbose=False)
        result = client.predict(
            "canonical_object",
            None,
            "bracers",
            12,
            api_name="/save_field_correction",
        )
        assert result[0].startswith('Saved canonical_object = "bracers"')
        assert len(load_review_events(corrections)) == 1
    finally:
        demo.close()


def test_unproven_abstention_is_disabled(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    record["model_provenance"].pop("review_record_binding")
    assert validate_semantic_field(record, "canonical_object").valid is False
    assert field_action_availability(record, "canonical_object")["model_abstention"] is False


def test_all_production_prefill_fields_have_a_parser_shape(tmp_path: Path) -> None:
    record = _synthetic_record(tmp_path)
    assert set(record["fields"]) == set(PREFILL_FIELDS)
    for field_name in PREFILL_FIELDS:
        preview_field_correction(record, field_name, json.dumps(record["fields"][field_name]["value"]))
