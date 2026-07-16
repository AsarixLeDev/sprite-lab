from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.label_v4.batch import merge_label_v4_shards, run_label_v4_shard, stable_shard
from spritelab.harvest.label_v4.cohort import run_same_cohort_comparison
from spritelab.harvest.label_v4.filename_parser import parse_filename_semantics
from spritelab.harvest.label_v4.pipeline import (
    LabelV4PipelineConfig,
    _merge_terminal_claims,
    label_record_v4,
)
from spritelab.harvest.label_v4.providers import MockJSONProvider
from spritelab.harvest.label_v4.reconciliation import ReconciliationResult, build_reconciliation_prompt


def _sprite(path: Path, color: tuple[int, int, int, int] = (100, 100, 120, 255)) -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    for x in range(1, 7):
        for y in range(1, 7):
            if (x - 3.5) ** 2 + (y - 3.5) ** 2 <= 10:
                image.putpixel((x, y), color)
    image.save(path)


def _proposal(
    *,
    object_name: str = "buckler",
    category: str = "armor",
    primary: str = "gray",
) -> dict[str, Any]:
    return {
        "object_candidates": [
            {"value": object_name, "visual_support": ["round silhouette", "raised center"]},
            {"value": "shield", "visual_support": ["defensive circular form"]},
        ],
        "category_candidates": [category],
        "surface_alias_candidates": ["round shield"],
        "role_candidates": ["defensive_equipment"],
        "shape": {
            "silhouette": ["round"],
            "aspect": ["compact"],
            "orientation": ["front_facing"],
            "structure": ["rimmed", "bossed"],
            "edge_profile": ["rounded"],
            "parts": ["rim", "central_boss"],
        },
        "color_roles": {
            "primary": [primary],
            "secondary": [],
            "outline": ["black"],
            "shadow": ["dark_gray"],
            "highlight": ["light_gray"],
        },
        "material_visual_cues": ["metallic"],
        "description_candidates": ["A round buckler with a dark rim and raised central boss."],
        "uncertainties": ["shield subtype could be broader"],
        "alternative_interpretations": ["shield"],
        "unsupported_fields": ["explicit_material"],
    }


class _SpyMock(MockJSONProvider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def call_json(self, **kwargs: Any):
        self.calls.append(dict(kwargs))
        return super().call_json(**kwargs)


def test_pipeline_a_recovers_named_semantics_and_conservative_risk(tmp_path: Path) -> None:
    image = tmp_path / "iron_buckler.png"
    _sprite(image)
    row = label_record_v4(
        {"sprite_id": "iron_buckler", "relative_path": "iron_buckler.png", "source_id": "armory"},
        image_path=image,
        config=LabelV4PipelineConfig(mode="A"),
    )
    assert (row["canonical_object"], row["category"], row["explicit_material"], row["role"]) == (
        "buckler",
        "armor",
        "iron",
        "defensive_equipment",
    )
    assert row["domain"] == "equipment_icon"
    assert "pixel_art" in row["semantics"]["style"]
    assert row["record_risk"]["record_uncertainty_1_20"] >= 13
    assert row["label_quality"]["critical_field_max_uncertainty"] >= 13
    assert row["provider_call_count"] == 0


def test_pipeline_abstains_from_source_record_object_identity_alone(tmp_path: Path) -> None:
    image = tmp_path / "opaque_asset.png"
    _sprite(image)
    row = label_record_v4(
        {
            "sprite_id": "opaque_asset",
            "relative_path": "opaque_asset.png",
            "declared_canonical_object": "sword",
            "declared_category": "weapon",
            "declared_role": "weapon",
        },
        image_path=image,
        config=LabelV4PipelineConfig(mode="A"),
    )
    assert row["canonical_object"] is None
    assert row["category"] == "weapon"
    assert row["role"] == "weapon"
    assert row["reconciliation"]["field_proposals"]["canonical_object"]["decision"] == "abstained"


def test_policy_terminal_state_precedes_provider_acceptance() -> None:
    claim = {"field": "canonical_object", "value": "cylinder"}
    provider = ReconciliationResult(claims_accepted=({**claim, "reason": "provider"},))

    accepted, rejected, unresolved = _merge_terminal_claims(
        ReconciliationResult(claims_rejected=({**claim, "reason": "promotion_policy"},)),
        provider,
    )
    assert not accepted and not unresolved
    assert rejected == ({**claim, "reason": "promotion_policy"},)

    accepted, rejected, unresolved = _merge_terminal_claims(
        ReconciliationResult(claims_unresolved=({**claim, "reason": "insufficient_evidence"},)),
        provider,
    )
    assert not accepted and not rejected
    assert unresolved == ({**claim, "reason": "insufficient_evidence"},)


def test_blind_stage_receives_pixels_without_filename_or_scheduler_context(tmp_path: Path) -> None:
    image = tmp_path / "secret_iron_buckler.png"
    _sprite(image)
    provider = _SpyMock(
        {"B_blind_vlm_proposal": _proposal()},
        model_identity="mock-vlm",
        namespace="blind_vlm_proposal_v4",
    )
    row = label_record_v4(
        {
            "sprite_id": "s",
            "relative_path": "iron_buckler.png",
            "scheduler_context": {"broad_type": "armor"},
        },
        image_path=image,
        config=LabelV4PipelineConfig(mode="B"),
        vlm_provider=provider,
    )
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["image_path"] == image
    assert call["payload"] is None
    assert "iron_buckler" not in call["prompt"]
    assert "scheduler" not in call["prompt"]
    assert row["vlm_proposal"]["proposal"]["canonical_object_candidates"][1]["value"] == "shield"


def test_text_reconciliation_is_text_only_and_preserves_raw_normalized_values(tmp_path: Path) -> None:
    image = tmp_path / "sprite.png"
    _sprite(image)
    vlm = MockJSONProvider({"B_blind_vlm_proposal": _proposal(object_name="shield", category="shield")})
    text_response = {
        "field_proposals": {
            "canonical_object": {
                "value": "buckler",
                "raw_open_vocabulary_value": "small round shield",
                "normalized_controlled_value": "buckler",
                "alternatives": ["shield"],
                "support": ["filename", "vlm_visual"],
                "conflicts": [],
            },
            "category": {
                "raw_open_vocabulary_value": "shield",
                "normalized_controlled_value": "armor",
                "alternatives": [],
                "support": ["taxonomy"],
                "conflicts": [],
            },
            "role": {
                "raw_open_vocabulary_value": "defensive_equipment",
                "normalized_controlled_value": "defensive_equipment",
                "alternatives": [],
                "support": ["filename"],
                "conflicts": [],
            },
        },
        "open_set_terms": ["small round shield"],
        "taxonomy_mapping_actions": [],
        "unresolved_conflicts": [],
        "claims_rejected": [],
    }
    text = _SpyMock({"C_text_reconciliation": text_response}, model_identity="mock-text")
    row = label_record_v4(
        {"sprite_id": "s", "relative_path": "iron_buckler.png"},
        image_path=image,
        config=LabelV4PipelineConfig(mode="B"),
        vlm_provider=vlm,
        text_provider=text,
    )
    assert text.calls[0]["image_path"] is None
    assert text.calls[0]["max_tokens"] == 1536
    proposal = row["reconciliation"]["field_proposals"]["canonical_object"]
    assert proposal["raw_open_vocabulary_value"] == "small round shield"
    assert proposal["normalized_controlled_value"] == "buckler"
    assert "small round shield" in row["open_set_terms"]


def test_invalid_provider_json_is_not_cached_or_masked_by_lock_cleanup(tmp_path: Path) -> None:
    image = tmp_path / "iron_buckler.png"
    _sprite(image)
    vlm = MockJSONProvider({"B_blind_vlm_proposal": _proposal()})
    truncated = MockJSONProvider({"C_text_reconciliation": '{"field_proposals":{"canonical_object":'})
    config = LabelV4PipelineConfig(mode="B", cache_dir=tmp_path / "cache")

    first = label_record_v4(
        {"sprite_id": "iron_buckler", "relative_path": "iron_buckler.png"},
        image_path=image,
        config=config,
        vlm_provider=vlm,
        text_provider=truncated,
    )
    second = label_record_v4(
        {"sprite_id": "iron_buckler", "relative_path": "iron_buckler.png"},
        image_path=image,
        config=config,
        vlm_provider=vlm,
        text_provider=truncated,
    )

    assert first["canonical_object"] == "buckler"
    assert second["canonical_object"] == "buckler"
    assert first["stage_ledger"][2]["failure_diagnostics"]["error_type"] == "invalid_json"
    assert second["stage_ledger"][2]["failure_diagnostics"]["error_type"] == "invalid_json"
    assert truncated.call_count == 2
    assert not list((tmp_path / "cache" / "text_reconciliation_v4").rglob("*.json"))


def test_reconciliation_prompt_groups_repeated_filename_provenance() -> None:
    mapping = {
        "category": "weapon",
        "domain": "weapon",
        "mapping_name": "shade_weapons",
        "material": "bronze",
        "native_resolution": "16x16",
        "sheet_coordinate": "r000_c019",
        "source_sheet": "16x16 Weapons RPG Icons/bronze-weapons.png",
        "variant_group_id": "shade:r000:c019",
    }
    record = {
        "relative_path": "16x16 Weapons RPG Icons/bronze-weapons.png",
        "archive_member": "16x16 Weapons RPG Icons/bronze-weapons.png",
        "source_sheet": "16x16 Weapons RPG Icons/bronze-weapons.png",
        "pack_name": "16x16 Weapon RPG Icons",
        "declared_material": "bronze",
        "auto_metadata": {"sheet_mapping": mapping},
    }
    deterministic = parse_filename_semantics(record, pack_context={"category": "weapon", "role": "weapon"})
    prompt = build_reconciliation_prompt(
        deterministic,
        _proposal(object_name="spear", category="weapon"),
        source_metadata={"pack_name": record["pack_name"]},
        declarative_mappings=mapping,
    )

    assert len(prompt) < 9000
    assert '"source_text"' not in prompt
    assert '"source_values"' not in prompt
    assert "Do not copy, quote, summarize, or echo" in prompt
    assert '"token_evidence"' in prompt


def test_filename_visual_color_conflict_is_policy_resolved_without_verifier(tmp_path: Path) -> None:
    image = tmp_path / "small_purple.png"
    _sprite(image, (40, 70, 180, 255))
    proposal = _proposal(object_name="gem", category="gem", primary="blue")
    proposal["role_candidates"] = ["resource"]
    proposal["surface_alias_candidates"] = ["small blue gemstone"]
    vlm = MockJSONProvider({"B_blind_vlm_proposal": proposal})
    row = label_record_v4(
        {"sprite_id": "small_purple", "relative_path": "small_purple.png", "pack_name": "gem pack"},
        image_path=image,
        config=LabelV4PipelineConfig(mode="C"),
        vlm_provider=vlm,
    )
    assert any(conflict["code"] == "filename_visual_color_conflict" for conflict in row["unresolved_conflicts"])
    assert row["verification"]["artifact"] is None
    assert row["verification"]["independent_prompt"] is None
    assert any(
        claim["reason"] == "deterministic_policy_already_decides_claim"
        for claim in row["verification"]["claims_not_routed"]
    )
    color_risk = row["field_risks"]["filename_color_hints"]
    assert color_risk["uncertainty_band"] != "strong"
    assert row["semantics"]["colors"]["filename_color_hints"] == ["purple"]
    assert row["semantics"]["colors"]["primary_colors"] == ["blue"]


def test_contradicted_verifier_claim_is_removed_from_final_semantics(tmp_path: Path) -> None:
    image = tmp_path / "iron_buckler.png"
    _sprite(image)
    vlm = MockJSONProvider(
        {"B_blind_vlm_proposal": _proposal(object_name="spear", category="weapon")},
        model_identity="mock-upstream",
    )
    verifier = MockJSONProvider(
        {
            "D_independent_verifier": {
                "claim_results": [
                    {
                        "claim_id": "verify-conflict-000",
                        "verdict": "contradicted",
                        "visible_support": ["the claimed object is not visibly established"],
                    }
                ],
                "unsupported_fields": [],
            }
        },
        model_identity="mock-verifier",
    )
    row = label_record_v4(
        {"sprite_id": "iron_buckler", "relative_path": "iron_buckler.png"},
        image_path=image,
        config=LabelV4PipelineConfig(mode="C"),
        vlm_provider=vlm,
        verifier_provider=verifier,
    )

    assert row["verification"]["decision_effects"]["claim_effects"][0]["effects"] == [
        "claim_rejected",
        "conflict_retained",
    ]
    assert row["canonical_object"] is None
    assert row["surface_alias"] is None
    assert "buckler" not in row["description"].lower()
    proposal = row["reconciliation"]["field_proposals"]["canonical_object"]
    assert proposal["decision"] == "rejected"
    assert row["reconciliation"]["field_proposals"]["surface_alias"]["decision"] == "abstained"
    assert any(
        claim.get("field") == "canonical_object"
        and claim.get("value") == "buckler"
        and claim.get("reason") == "verifier_contradicted"
        for claim in row["reconciliation"]["claims_rejected"]
    )
    assert not any(
        claim.get("field") == "canonical_object" and claim.get("value") == "buckler"
        for claim in row["reconciliation"]["claims_accepted"]
    )
    assert any(
        claim.get("field") == "surface_alias"
        and claim.get("value") == "iron buckler"
        and claim.get("reason") == "canonical_identity_rejected"
        for claim in row["reconciliation"]["claims_unresolved"]
    )


def test_mocked_same_cohort_improves_coverage_without_paid_calls(tmp_path: Path) -> None:
    result = run_same_cohort_comparison(output_dir=tmp_path / "cohort", max_records=2)
    metrics = result["metrics"]
    assert metrics["B"]["field_coverage"] > metrics["A"]["field_coverage"]
    assert metrics["C"]["field_coverage"] >= metrics["B"]["field_coverage"]
    assert metrics["B"]["unsupported_material_rate"] == 0.0
    assert result["paid_provider_calls"] == 0
    assert result["legacy_cache_reuse"] is False
    assert (tmp_path / "cohort" / "cohort_A_v1.jsonl").is_file()
    rows = [json.loads(line) for line in (tmp_path / "cohort" / "cohort_C_v1.jsonl").read_text().splitlines()]
    assert all(row["legacy_evidence_used"] is False for row in rows)


def test_resumable_shards_merge_deterministically_without_input_mutation(tmp_path: Path) -> None:
    records = []
    for index, name in enumerate(("iron_buckler", "cloth_pants", "copper_ring", "leather_cap")):
        image = tmp_path / f"{name}.png"
        _sprite(image, (80 + index * 20, 90, 120, 255))
        records.append(
            {
                "sprite_id": name,
                "relative_path": f"{name}.png",
                "final_png_path": str(image),
                "source_id": "fixture",
            }
        )
    input_path = tmp_path / "records.jsonl"
    input_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    input_path.write_text(input_text, encoding="utf-8")
    output = tmp_path / "batch"
    first = []
    for shard_index in range(2):
        first.append(
            run_label_v4_shard(
                input_path,
                output,
                shard_index=shard_index,
                shard_count=2,
                workers=2,
            )
        )
    merged = merge_label_v4_shards(output, shard_count=2)
    assert merged["record_count"] == 4
    first_hash = merged["output_sha256"]
    resumed = run_label_v4_shard(input_path, output, shard_index=0, shard_count=2, workers=2)
    assert resumed.resumed_records == first[0].completed_records
    assert merge_label_v4_shards(output, shard_count=2)["output_sha256"] == first_hash
    assert input_path.read_text(encoding="utf-8") == input_text
    assert all(stable_shard(row["sprite_id"], 2) in {0, 1} for row in records)


def test_batch_failure_is_isolated_and_not_emitted_as_label(tmp_path: Path) -> None:
    image = tmp_path / "iron_buckler.png"
    _sprite(image)
    rows = [
        {"sprite_id": "good", "relative_path": "iron_buckler.png", "final_png_path": str(image)},
        {"sprite_id": "missing", "relative_path": "missing.png", "final_png_path": str(tmp_path / "no.png")},
    ]
    input_path = tmp_path / "records.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = run_label_v4_shard(input_path, tmp_path / "batch", shard_index=0, shard_count=1)
    assert result.completed_records == 1
    assert result.failure_records == 1
    failures = [json.loads(line) for line in result.failure_path.read_text().splitlines()]
    assert failures[0]["sprite_id"] == "missing"
    labels = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    assert [row["sprite_id"] for row in labels] == ["good"]
