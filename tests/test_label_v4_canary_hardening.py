from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from spritelab.harvest.label_v4.cache import LabelV4Cache, make_cache_identity
from spritelab.harvest.label_v4.cohort import TARGETED_SMOKE_IDS, run_same_cohort_comparison
from spritelab.harvest.label_v4.pipeline import (
    LabelV4PipelineConfig,
    RequiredSharedCacheMiss,
    label_record_v4,
)
from spritelab.harvest.label_v4.providers import MockJSONProvider
from spritelab.harvest.label_v4.semantic_axes import DOMAIN_VALUES

TARGET_MANIFEST = Path("experiments/label_v4_canary_hardening/targeted_smoke_manifest.jsonl")


def _sprite(path: Path, color: tuple[int, int, int, int] = (110, 75, 40, 255)) -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    for x in range(1, 7):
        for y in range(2, 6):
            image.putpixel((x, y), color)
    image.save(path)


def _visual_proposal() -> dict:
    return {
        "canonical_object_candidates": [{"value": "gem", "visual_support": ["faceted silhouette", "bright center"]}],
        "visual_form": [],
        "category_candidates": ["gem"],
        "surface_alias_candidates": ["blue gem"],
        "role_candidates": ["resource"],
        "shape": {"silhouette": ["diamond"], "parts": ["facets"]},
        "color_roles": {"primary": ["blue"], "outline": ["black"]},
        "material_visual_cues": ["crystalline"],
        "description_candidates": ["A blue gem."],
        "uncertainties": [],
        "alternative_interpretations": [],
        "unsupported_fields": ["explicit_material"],
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_targeted_manifest_is_exact_and_mock_regression_covers_all_15(tmp_path: Path) -> None:
    manifest_ids = [row["sprite_id"] for row in _read_jsonl(TARGET_MANIFEST)]
    assert manifest_ids == list(TARGETED_SMOKE_IDS)

    output = tmp_path / "targeted"
    result = run_same_cohort_comparison(
        output_dir=output,
        record_manifest=TARGET_MANIFEST,
        max_records=15,
    )
    assert result["ordered_sprite_ids"] == list(TARGETED_SMOKE_IDS)
    assert result["record_count"] == 15
    assert result["paid_provider_calls"] == 0
    assert (output / "targeted_mock_results.json").is_file()
    assert (
        (output / "targeted_mock_results.md")
        .read_text(encoding="utf-8")
        .startswith("# Labeling v4 targeted mock results")
    )

    rows_by_mode = {
        mode: {row["sprite_id"]: row for row in _read_jsonl(output / f"cohort_{mode}_v1.jsonl")}
        for mode in ("A", "B", "C")
    }
    for rows in rows_by_mode.values():
        assert list(rows) == list(TARGETED_SMOKE_IDS)
        assert all(row["actual_http_attempts"] == 0 for row in rows.values())
        assert all(row["domain"] in DOMAIN_VALUES for row in rows.values())

    for mode in ("A", "B", "C"):
        buckler = rows_by_mode[mode]["acq_idylwild_armory_iron_buckler"]
        assert (
            buckler["canonical_object"],
            buckler["category"],
            buckler["explicit_material"],
            buckler["role"],
            buckler["surface_alias"],
        ) == ("buckler", "armor", "iron", "defensive_equipment", "iron buckler")

        helmet = rows_by_mode[mode]["acq_idylwild_armory_platemail_helmet"]
        assert (
            helmet["canonical_object"],
            helmet["category"],
            helmet["explicit_material"],
            helmet["role"],
        ) == ("helmet", "armor", "plate_metal", "wearable_equipment")

        small = rows_by_mode[mode]["acq_gem_ettingrinder_small_purple"]
        assert small["semantics"]["size_hint"] == "small"
        assert "purple" in small["semantics"]["colors"]["filename_color_hints"]

        agate = rows_by_mode[mode]["oga_cc0_gem_7soul1_agate"]
        assert agate["legacy_evidence_used"] is False
        assert agate["explicit_material"] is None
        assert agate["domain"] == "resource_icon"

    for mode in ("B", "C"):
        crystal = rows_by_mode[mode]["oga_496_rpg_icons_32fix_i_crystal01"]
        assert "cluster" in crystal["canonical_object"]

        shade = rows_by_mode[mode]["shade_16x16_weapons_bronze-weapons_r000_c019"]
        assert (shade["category"], shade["role"], shade["explicit_material"]) == (
            "weapon",
            "weapon",
            "bronze",
        )
        assert shade["canonical_object"] is None
        assert any(value in {"rod", "elongated_form", "cylinder", "stick"} for value in shade["visual_form"])
        assert "rod" in shade["canonical_object_alternatives"]
        assert shade["surface_alias"] is None
        assert not re.search(r"\b(?:pencil|pen|matchstick)\b", shade["description"].lower())
        assert shade["field_risks"]["canonical_object"]["uncertainty_1_20"] >= 13
        assert shade["record_risk"]["record_uncertainty_1_20"] >= 13
        assert shade["verification"]["artifact"] is None

    shade_c = rows_by_mode["C"]["shade_16x16_weapons_bronze-weapons_r000_c019"]
    assert shade_c["new_provider_calls"] == 0
    assert shade_c["shared_cache_hits"] == 2


def test_stage_c_invalid_domain_raw_is_persisted_but_cannot_reach_fusion(tmp_path: Path) -> None:
    image = tmp_path / "iron_buckler.png"
    _sprite(image)
    vlm = MockJSONProvider({"B_blind_vlm_proposal": _visual_proposal()}, model_identity="mock-vlm")
    raw_text = {
        "field_proposals": {
            "domain": {
                "raw_open_vocabulary_value": "weapon",
                "normalized_controlled_value": "weapon",
                "support": ["taxonomy: weapon is a valid domain"],
            }
        },
        "taxonomy_mapping_actions": [{"field": "domain", "raw": "weapon", "action": "weapon is a valid domain"}],
        "unresolved_conflicts": [],
        "claims_accepted": [],
        "claims_rejected": [],
        "claims_unresolved": [],
    }
    text = MockJSONProvider({"C_text_reconciliation": raw_text}, model_identity="mock-text")
    row = label_record_v4(
        {"sprite_id": "buckler", "relative_path": "iron_buckler.png"},
        image_path=image,
        config=LabelV4PipelineConfig(mode="B"),
        vlm_provider=vlm,
        text_provider=text,
    )

    artifact = row["reconciliation_provider_artifact"]
    assert "weapon is a valid domain" in artifact["raw_output"]
    normalized = row["reconciliation"]
    domain = normalized["field_proposals"]["domain"]
    assert domain["raw_open_vocabulary_value"] == "weapon"
    assert domain["normalized_controlled_value"] is None
    assert domain["value"] is None
    assert domain["decision"] == "rejected"
    assert domain["support"] == []
    assert any(
        conflict["code"] == "invalid_taxonomy_provider_output" for conflict in normalized["unresolved_conflicts"]
    )
    assert "weapon is a valid domain" not in json.dumps(normalized["taxonomy_mapping_actions"], ensure_ascii=False)
    assert row["domain"] == "equipment_icon"
    assert row["actual_http_attempts"] == 0


def test_shared_bc_cache_reuses_exact_artifacts_and_fail_closed_miss_never_calls_provider(tmp_path: Path) -> None:
    image = tmp_path / "small_purple.png"
    _sprite(image, (70, 40, 150, 255))
    record = {
        "sprite_id": "small_purple",
        "relative_path": "small_purple.png",
        "pack_name": "gem pack",
    }
    shared = tmp_path / "shared-cache"
    provider = MockJSONProvider(
        {"B_blind_vlm_proposal": _visual_proposal()},
        model_identity="same-mock-vlm",
    )
    row_b = label_record_v4(
        record,
        image_path=image,
        config=LabelV4PipelineConfig(mode="B", cache_dir=shared, shared_cache=True),
        vlm_provider=provider,
    )
    assert row_b["new_provider_calls"] == 2
    assert row_b["actual_http_attempts"] == 0
    assert provider.call_count == 1

    row_c = label_record_v4(
        record,
        image_path=image,
        config=LabelV4PipelineConfig(
            mode="C",
            cache_dir=shared,
            shared_cache=True,
            require_shared_bc_cache=True,
        ),
        vlm_provider=provider,
    )
    assert provider.call_count == 1
    assert row_c["new_provider_calls"] == 0
    assert row_c["actual_http_attempts"] == 0
    assert row_c["shared_cache_hits"] == 2
    assert row_c["verification"]["artifact"] is None
    assert all(stage["new_token_usage"] == {} for stage in row_c["stage_ledger"][1:])

    empty = tmp_path / "empty-shared-cache"
    stale_lock = empty / ".locks" / "blind_vlm_proposal_v4" / "unrelated.lock"
    stale_lock.parent.mkdir(parents=True)
    stale_lock.write_text("stale", encoding="utf-8")
    never_called = MockJSONProvider(
        {"B_blind_vlm_proposal": _visual_proposal()},
        model_identity="same-mock-vlm",
    )
    with pytest.raises(RequiredSharedCacheMiss):
        label_record_v4(
            record,
            image_path=image,
            config=LabelV4PipelineConfig(
                mode="C",
                cache_dir=empty,
                shared_cache=True,
                require_shared_bc_cache=True,
            ),
            vlm_provider=never_called,
        )
    assert never_called.call_count == 0


def test_cache_lock_without_identity_validated_json_is_never_a_hit(tmp_path: Path) -> None:
    identity = make_cache_identity(
        namespace="test",
        stage="B",
        image=b"\x00\x00\x00\x00",
        model_identity="mock",
        prompt_version="v1",
        prompt="prompt",
        schema_version="v1",
        request={"x": 1},
        provider="mock",
        width=1,
        height=1,
    )
    cache = LabelV4Cache(tmp_path / "cache")
    lock = cache.root / ".locks" / identity.namespace / f"{identity.key}.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("stale", encoding="utf-8")
    assert cache.get(identity) is None
