from __future__ import annotations

import json

import numpy as np
import pytest

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.curation.manifest import (
    CurationDecision,
    append_curation_decision,
    discover_bundle_ids,
    format_curation_summary,
    load_curation_events,
    load_latest_curation,
    main as curation_manifest_main,
    summarize_curation,
    validate_curation_against_bundles,
    write_curation_events,
)


def test_creating_valid_decision_fills_timestamp() -> None:
    decision = CurationDecision(sprite_id="copper_vial_001", status="accepted")

    assert decision.sprite_id == "copper_vial_001"
    assert decision.status == "accepted"
    assert decision.timestamp.endswith("Z")


def test_invalid_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="status"):
        CurationDecision(sprite_id="sprite_a", status="maybe")


def test_empty_sprite_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="sprite_id"):
        CurationDecision(sprite_id="", status="accepted")


def test_tags_are_normalized_and_deduplicated() -> None:
    decision = CurationDecision(
        sprite_id="sprite_a",
        status="accepted",
        tags=(" Item Icon ", "item icon", "Copper"),
    )

    assert decision.tags == ("item_icon", "copper")


def test_reasons_are_normalized_and_deduplicated() -> None:
    decision = CurationDecision(
        sprite_id="sprite_a",
        status="rejected",
        reasons=(" Too Noisy ", "too noisy", "bad_alpha"),
    )

    assert decision.reasons == ("too_noisy", "bad_alpha")


def test_unknown_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown curation reason"):
        CurationDecision(sprite_id="sprite_a", status="rejected", reasons=("mystery",))


def test_decision_to_dict_from_dict_roundtrip() -> None:
    decision = CurationDecision(
        sprite_id="sprite_a",
        status="needs_fix",
        tags=("item icon",),
        reasons=("bad palette",),
        notes="Needs cleanup.",
        reviewer="tester",
        source_path="bundles/sprite_a",
    )

    loaded = CurationDecision.from_dict(decision.to_dict())

    assert loaded == decision


def test_from_dict_tolerates_missing_optional_fields() -> None:
    decision = CurationDecision.from_dict(
        {
            "sprite_id": "sprite_a",
            "status": "accepted",
        }
    )

    assert decision.tags == ()
    assert decision.reasons == ()
    assert decision.notes == ""
    assert decision.timestamp


def test_loading_missing_jsonl_returns_empty_list(tmp_path) -> None:
    assert load_curation_events(tmp_path / "missing.jsonl") == []
    assert load_latest_curation(tmp_path / "missing.jsonl") == {}


def test_append_creates_file_and_parent_directories(tmp_path) -> None:
    path = tmp_path / "nested" / "curation.jsonl"
    append_curation_decision(path, CurationDecision(sprite_id="sprite_a", status="accepted"))

    assert path.exists()
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_loading_jsonl_returns_all_events(tmp_path) -> None:
    path = tmp_path / "curation.jsonl"
    decisions = [
        CurationDecision(sprite_id="sprite_a", status="accepted"),
        CurationDecision(sprite_id="sprite_b", status="rejected", reasons=("duplicate",)),
    ]
    write_curation_events(path, decisions)

    assert load_curation_events(path) == decisions


def test_latest_decision_wins_for_same_sprite_id(tmp_path) -> None:
    path = tmp_path / "curation.jsonl"
    append_curation_decision(path, CurationDecision(sprite_id="sprite_a", status="accepted"))
    append_curation_decision(path, CurationDecision(sprite_id="sprite_a", status="rejected"))

    latest = load_latest_curation(path)

    assert latest["sprite_a"].status == "rejected"


def test_malformed_json_line_raises_value_error_with_line_number(tmp_path) -> None:
    path = tmp_path / "curation.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_curation_events(path)


def test_invalid_decision_line_raises_value_error_with_line_number(tmp_path) -> None:
    path = tmp_path / "curation.jsonl"
    path.write_text(json.dumps({"sprite_id": "sprite_a", "status": "invalid"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_curation_events(path)


def test_summary_counts_statuses_tags_and_reasons() -> None:
    decisions = [
        CurationDecision(sprite_id="sprite_a", status="accepted", tags=("item_icon",)),
        CurationDecision(
            sprite_id="sprite_b",
            status="rejected",
            tags=("item_icon", "vial"),
            reasons=("too_noisy", "duplicate"),
        ),
        CurationDecision(sprite_id="sprite_c", status="quarantine", reasons=("bad_source",)),
        CurationDecision(sprite_id="sprite_d", status="needs_fix", reasons=("bad_roles",)),
    ]

    summary = summarize_curation(decisions)
    text = format_curation_summary(summary)

    assert summary.total_event_count == 4
    assert summary.accepted_count == 1
    assert summary.rejected_count == 1
    assert summary.quarantine_count == 1
    assert summary.needs_fix_count == 1
    assert summary.count_by_tag["item_icon"] == 2
    assert summary.count_by_reason["duplicate"] == 1
    assert "Curation summary" in text


def test_validation_reports_unknown_uncurated_and_status_sets(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    _write_bundle(bundles / "sprite_a", "sprite_a")
    _write_bundle(bundles / "sprite_b", "sprite_b")
    _write_bundle(bundles / "sprite_c", "sprite_c")
    _write_bundle(bundles / "sprite_d", "sprite_d")
    _write_bundle(bundles / "sprite_e", "sprite_e")

    latest = {
        "sprite_a": CurationDecision(sprite_id="sprite_a", status="accepted"),
        "sprite_b": CurationDecision(sprite_id="sprite_b", status="rejected"),
        "sprite_c": CurationDecision(sprite_id="sprite_c", status="quarantine"),
        "sprite_d": CurationDecision(sprite_id="sprite_d", status="needs_fix"),
        "missing": CurationDecision(sprite_id="missing", status="accepted"),
    }

    result = validate_curation_against_bundles(latest, discover_bundle_ids(bundles))

    assert result.unknown_curated_sprite_ids == ("missing",)
    assert result.uncurated_bundle_ids == ("sprite_e",)
    assert result.accepted_bundle_ids == ("sprite_a",)
    assert result.rejected_bundle_ids == ("sprite_b",)
    assert result.quarantine_bundle_ids == ("sprite_c",)
    assert result.needs_fix_bundle_ids == ("sprite_d",)


def test_validation_reports_bundle_id_collisions(tmp_path) -> None:
    bundles = tmp_path / "bundles"
    _write_bundle(bundles / "first", "same_id")
    _write_bundle(bundles / "second", "same_id")

    bundle_ids = discover_bundle_ids(bundles)
    result = validate_curation_against_bundles({}, bundle_ids)

    assert result.collision_issues
    assert "same_id" in result.collision_issues[0]


def test_manifest_decide_cli_appends_decision(tmp_path) -> None:
    path = tmp_path / "curation.jsonl"

    curation_manifest_main(
        [
            "decide",
            "--curation",
            str(path),
            "--sprite-id",
            "sprite_a",
            "--status",
            "accepted",
            "--tag",
            "Item Icon",
            "--notes",
            "Good.",
        ]
    )

    events = load_curation_events(path)
    assert len(events) == 1
    assert events[0].sprite_id == "sprite_a"
    assert events[0].tags == ("item_icon",)


def test_manifest_summary_cli_prints_summary(tmp_path, capsys) -> None:
    path = tmp_path / "curation.jsonl"
    append_curation_decision(path, CurationDecision("sprite_a", "accepted"))

    curation_manifest_main(["summary", "--curation", str(path)])

    captured = capsys.readouterr()
    assert "Curation summary" in captured.out
    assert "accepted: 1" in captured.out


def _write_bundle(directory, sprite_id: str) -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[10:14, 10:14] = 1
    index_map[10:14, 10:14] = 1
    palette = np.array([[0, 0, 0], [120, 80, 160]], dtype=np.uint8)
    bundle = SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id, palette_size=1),
    )
    save_bundle(bundle, directory)
