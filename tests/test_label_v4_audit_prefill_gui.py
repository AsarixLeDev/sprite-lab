from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from spritelab.harvest.label_v4.assisted_v4_gui import (
    DEFAULT_ZOOM,
    gui_field_view,
    gui_record_summary,
    load_assisted_records,
    pixel_preview_html,
    review_resume_index,
)
from spritelab.harvest.label_v4.audit_prefill import (
    PREFILL_FIELDS,
    prepare_audit,
    require_prefilled_records,
)
from spritelab.harvest.label_v4.review import (
    abstain_field,
    accept_proposal,
    load_review_events,
    mark_suitable_image,
    mark_unsuitable_image,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments" / "label_v4_calibration_wave1" / "audit_manifest.jsonl"
REAL_PILOT = ROOT / "experiments" / "label_v4_real_pilot_15_v1"
REPLAY = ROOT / "experiments" / "label_v4_pilot_replay_v2"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _selection(tmp_path: Path, sprite_ids: set[str]) -> Path:
    rows = [row for row in _rows(FROZEN) if row["sprite_id"] in sprite_ids]
    path = tmp_path / "selection.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_raw_audit_manifest_is_rejected_by_gui() -> None:
    with pytest.raises(ValueError, match="requires label_v4_prefilled_audit_record_v1"):
        load_assisted_records(FROZEN)


def test_prepare_prefills_named_deterministic_and_cached_rich_without_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(tmp_path, {"acq_gem_thekingphoenix_diamond", "oga_cc0_key_rcorre_key_01"})
    before = selection.read_bytes()
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("provider call"))
    )
    result = prepare_audit(
        selection, tmp_path / "out", inference_policy="cached-only", artifact_roots=[REAL_PILOT, REPLAY]
    )
    rows = {row["sprite_id"]: row for row in _rows(tmp_path / "out" / "audit_prefilled_records.jsonl")}
    assert result["provider_calls_made"] == 0
    assert selection.read_bytes() == before
    assert rows["acq_gem_thekingphoenix_diamond"]["fields"]["canonical_object"]["value"] == "diamond"
    assert rows["oga_cc0_key_rcorre_key_01"]["prediction_origin"] == "compatible_cached_rich_vlm"
    require_prefilled_records(list(rows.values()))


def test_missing_prediction_is_distinct_from_abstention_and_every_null_has_reason(tmp_path: Path) -> None:
    selection = _selection(tmp_path, {"acq_craftpix_minerals_icon29"})
    prepare_audit(selection, tmp_path / "out", inference_policy="deterministic-only")
    record = _rows(tmp_path / "out" / "audit_prefilled_records.jsonl")[0]
    canonical = record["fields"]["canonical_object"]
    assert canonical["value"] is None
    assert canonical["value_state"] == "missing_prediction"
    assert canonical["reason"] == "rich_vlm_stage_not_executed"
    assert record["missing_stages"] == ["B_blind_vlm_proposal", "C_text_reconciliation"]
    assert all(field["reason"] for field in record["fields"].values() if field["value"] is None)


def test_all_fields_alternatives_abstention_and_record_summary_render(tmp_path: Path) -> None:
    selection = _selection(tmp_path, {"oga_cc0_key_rcorre_key_01"})
    prepare_audit(selection, tmp_path / "out", inference_policy="cached-only", artifact_roots=[REAL_PILOT])
    record = _rows(tmp_path / "out" / "audit_prefilled_records.jsonl")[0]
    truth = tmp_path / "truth.jsonl"
    assert set(PREFILL_FIELDS) == set(record["fields"])
    assert "Canonical object" in gui_record_summary(record)
    panel = gui_field_view(record, "canonical_object", truth)
    assert "alternatives" in panel
    mark_suitable_image(truth, record)
    abstain_field(truth, record, "canonical_object")
    assert load_review_events(truth)[-1].human_outcome == "human_abstained"


def test_unsuitable_is_record_level_not_scorable_and_append_only(tmp_path: Path) -> None:
    selection = _selection(tmp_path, {"acq_craftpix_minerals_icon29"})
    prepare_audit(selection, tmp_path / "out", inference_policy="deterministic-only")
    record = _rows(tmp_path / "out" / "audit_prefilled_records.jsonl")[0]
    original = copy.deepcopy(record)
    truth = tmp_path / "truth.jsonl"
    event = mark_unsuitable_image(truth, record, metadata={"record_completed": True})
    assert event.human_outcome == "not_scorable_due_to_image"
    assert event.field_name == ""
    assert record == original
    assert len(load_review_events(truth)) == 1


def test_resume_skips_only_completed_records(tmp_path: Path) -> None:
    selection = _selection(tmp_path, {"acq_craftpix_minerals_icon29", "acq_gem_thekingphoenix_diamond"})
    prepare_audit(selection, tmp_path / "out", inference_policy="deterministic-only")
    records = _rows(tmp_path / "out" / "audit_prefilled_records.jsonl")
    truth = tmp_path / "truth.jsonl"
    mark_unsuitable_image(truth, records[0], metadata={"record_completed": True})
    # Legacy generic completion metadata is not a valid two-pass quality decision.
    assert review_resume_index(records, truth) == 0


def test_pixel_viewer_uses_nearest_neighbor_checkerboard_zoom_and_native_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "sprite.png"
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((15, 15), (255, 0, 0, 255))
    image.save(path)
    rendered = pixel_preview_html(path, DEFAULT_ZOOM)
    assert "image-rendering:pixelated" in rendered
    assert "linear-gradient" in rendered
    assert "Decoded exported canvas" in rendered
    assert "32\u00d732" in rendered
    assert "zoom 12\u00d7" in rendered
    assert 'width="384"' in rendered and 'height="384"' in rendered


def test_blind_mode_hides_proposal_risk_and_evidence_until_first_critical_judgment(tmp_path: Path) -> None:
    selection = _selection(tmp_path, {"acq_craftpix_minerals_icon29"})
    prepare_audit(selection, tmp_path / "out", inference_policy="deterministic-only")
    record = _rows(tmp_path / "out" / "audit_prefilled_records.jsonl")[0]
    record["review_mode"] = "blind"
    record["proposal_visible_before_judgment"] = False
    truth = tmp_path / "truth.jsonl"
    hidden = gui_field_view(record, "canonical_object", truth)
    # Missing predictions never enter blind semantic review; there is no proposal to hide/reveal.
    assert hidden["blind_locked"] is False
    with pytest.raises(ValueError, match="forbidden for value_state=missing_prediction"):
        accept_proposal(truth, record, "canonical_object")
