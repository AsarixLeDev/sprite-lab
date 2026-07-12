from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spritelab.harvest.label_v4.assisted_v4_gui import audit_review_metrics
from spritelab.harvest.label_v4.calibration_wave import build_calibration_wave1
from spritelab.harvest.label_v4.description import choose_or_regenerate_description, validate_description
from spritelab.harvest.label_v4.replay import OfflineReplayError, replay_pilot, scan_semantic_channels
from spritelab.harvest.label_v4.routing import AdaptiveRoutingSignals, decide_profile_routing
from spritelab.harvest.label_v4.semantic_axes import normalize_visual_color_roles
from spritelab.harvest.label_v4.structured_output import recover_json_object

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments" / "label_v4_real_pilot_15_v1"
CACHE = PILOT / "shared_bc_cache_v1"
POOL = ROOT / "datasets" / "sprite_lab_unlabeled_pool_v1_r2" / "candidate_manifest.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_replay_has_zero_http_and_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline replay attempted HTTP")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    output = tmp_path / "replay"
    first = replay_pilot(
        PILOT, output, shared_cache_root=CACHE, require_complete_cache=True, allow_deterministic_fallback=True
    )
    payload = (output / "replayed_records.jsonl").read_bytes()
    second = replay_pilot(
        PILOT, output, shared_cache_root=CACHE, require_complete_cache=True, allow_deterministic_fallback=True
    )
    assert (output / "replayed_records.jsonl").read_bytes() == payload
    assert first["http_attempts"] == second["new_provider_calls"] == 0
    assert first["deterministic_output_hash"] == second["deterministic_output_hash"]
    assert first["original_pilot_unchanged"] is True


def test_replay_incompatible_cache_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "cache"
    shutil.copytree(CACHE, copied)
    artifact = next((copied / "blind_vlm_proposal_v4").glob("*/*.json"))
    envelope = json.loads(artifact.read_text(encoding="utf-8"))
    envelope["identity"]["prompt_version"] = "incompatible"
    artifact.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(OfflineReplayError, match="identity mismatch"):
        replay_pilot(PILOT, tmp_path / "out", shared_cache_root=copied, require_complete_cache=True)


def test_small_purple_repair_and_fallback_provenance(tmp_path: Path) -> None:
    replay_pilot(
        PILOT, tmp_path, shared_cache_root=CACHE, require_complete_cache=True, allow_deterministic_fallback=True
    )
    row = next(value for value in _rows(tmp_path / "replayed_records.jsonl") if "small_purple" in value["sprite_id"])
    stage = next(value for value in row["stage_outcomes"] if value["stage"] == "C_text_reconciliation")
    assert stage["stage_status"] == "success_after_json_repair"
    assert stage["raw_response_hash"]
    assert stage["repair_actions"] == ["drop_incomplete_terminal_array_member", "close_unterminated_containers"]
    assert row["record_status"] == "completed_with_repaired_stage"
    assert row["canonical_object"] == "gem"
    assert set(row["deterministic_fallback_fields"]) == {"domain", "explicit_material", "surface_alias", "colors"}
    assert row["reconciliation_provider_artifact"]["raw_output"].endswith('"unresolved_conflicts": [\n    {\n')


def test_pilot_description_regressions(tmp_path: Path) -> None:
    replay_pilot(
        PILOT, tmp_path, shared_cache_root=CACHE, require_complete_cache=True, allow_deterministic_fallback=True
    )
    rows = {row["sprite_id"]: row for row in _rows(tmp_path / "replayed_records.jsonl")}
    assert rows["acq_gem_ettingrinder_small_purple"]["description"] == (
        "A small dark-purple gem icon with a rounded, slightly flattened silhouette."
    )
    assert rows["oga_cc0_gem_7soul1_agate"]["description"] == "A pink oval agate icon with a dark outline."
    assert rows["shade_16x16_weapons_bronze-weapons_r000_c019"]["description"] == (
        "An elongated bronze-colored weapon icon with a dark outline."
    )
    for sprite_id in ("oga_cc0_food_ocal_eggplant", "oga_cc0_key_rcorre_key_01"):
        assert rows[sprite_id]["description"]
        assert rows[sprite_id]["description_validation"]["claims_rejected"] == []


@pytest.mark.parametrize(
    ("raw", "action"),
    [
        ('wrapper {"a": 1} tail', "extract_single_json_object"),
        ('{"a": 1,}', "remove_trailing_commas"),
        ('{"a": True}', "normalize_json_literals"),
        ('{"a": [{', "drop_incomplete_terminal_array_member"),
    ],
)
def test_bounded_malformed_json_repairs(raw: str, action: str) -> None:
    recovered = recover_json_object(raw)
    assert recovered.value is not None
    assert action in recovered.repair_actions
    assert recovered.raw_response_hash


def test_routing_profiles_skip_named_semantics_but_keep_ambiguous_rich() -> None:
    named = decide_profile_routing(
        AdaptiveRoutingSignals(shape_weak=True, description_missing=True), critical_semantics_complete=True
    )
    assert named.run_stage_b is False
    assert named.run_stage_c is False
    ambiguous = decide_profile_routing(
        AdaptiveRoutingSignals(canonical_object_missing=True, object_open_set=True), critical_semantics_complete=False
    )
    assert ambiguous.run_stage_b is True
    assert ambiguous.run_stage_c is True


def test_description_generator_and_validator_share_palette_facts() -> None:
    facts = {
        "canonical_object": "gem",
        "category": "gem",
        "size_hint": "small",
        "shape": {"silhouette": ["rounded_or_oval"], "aspect": ["slightly_flattened"]},
        "colors": {
            "palette_colors": ["dark_purple", "black"],
            "primary_colors": ["dark_purple"],
            "outline_colors": ["black"],
        },
    }
    result = choose_or_regenerate_description(["an amethyst jewel"], facts)
    assert result["description"] == "A small dark-purple gem icon with a rounded, slightly flattened silhouette."
    assert validate_description(result["description"], facts) == (True, ())
    assert result["candidate_claims_rejected"]


def test_role_aware_outline_normalization_prefers_black() -> None:
    result = normalize_visual_color_roles({"outline": ["dark pink or black"]}, ["pink", "black"])
    assert result.color_roles.outline_colors == ("black",)
    assert result.role_evidence["outline_colors"][0]["confidence"] == 0.65
    assert not result.color_roles.role_membership_conflicts()


def test_semantic_scanner_separates_raw_evidence() -> None:
    rows = [
        {
            "sprite_id": "x",
            "semantics": {"domain": "inventory_icon", "description": "A key icon."},
            "description": "A key icon.",
            "description_validation": {"claims_rejected": []},
            "vlm_proposal": {"raw_output": "domain = weapon; CC0 Gem Icons; rejected material guess"},
            "training_channels": {"description_text": {"training_state": "active", "value": "A key icon."}},
        }
    ]
    result = scan_semantic_channels(rows)
    assert result["final_semantic_violations"] == []
    assert result["training_target_violations"] == []
    assert len(result["raw_evidence_observations"]) == 3


def test_calibration_wave_is_exact_grouped_deterministic_and_has_no_gold(tmp_path: Path) -> None:
    first = build_calibration_wave1(POOL, tmp_path / "one")
    second = build_calibration_wave1(POOL, tmp_path / "two")
    one = (tmp_path / "one" / "audit_manifest.jsonl").read_bytes()
    two = (tmp_path / "two" / "audit_manifest.jsonl").read_bytes()
    assert first["selected"] == second["selected"] == 100
    assert first["audit_set_hash"] == second["audit_set_hash"]
    assert one == two
    report = json.loads((tmp_path / "one" / "sampling_report.json").read_text(encoding="utf-8"))
    assert report["geometry_groups_preserved"] is True
    assert report["declared_variant_groups_preserved"] is True
    assert all(row["human_truth"] is None for row in _rows(tmp_path / "one" / "audit_manifest.jsonl"))


def test_empty_append_only_gui_truth_has_null_correctness(tmp_path: Path) -> None:
    build_calibration_wave1(POOL, tmp_path)
    truth = tmp_path / "human_truth.jsonl"
    before = truth.read_bytes()
    metrics = audit_review_metrics(tmp_path / "audit_manifest.jsonl", truth)
    assert metrics["field_correctness"] is None
    assert metrics["events"] == 0
    assert truth.read_bytes() == before
