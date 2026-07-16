from __future__ import annotations

import json
from pathlib import Path

import pytest

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.harvest.label_v4.training_quality import (
    FIELD_VALUE_STATE_IDS,
    TRAINING_QUALITY_FIELDS,
    apply_conditioning_policy,
    dataset_quality_score,
    dataset_uncertainty_report,
    effective_sample_size,
    evaluation_uncertainty_report,
    extract_training_quality,
    loss_weight_for_risk,
    normalize_training_quality,
    quality_grade_breakdown,
    quality_tensor_payload,
    training_uncertainty_report,
    uncertainty_from_risk_upper,
)
from spritelab.training.structured_conditioning import (
    build_structured_conditioning_vocab,
    encode_structured_conditioning,
    extract_structured_fields,
)


def _quality(score: int, *, calibration_state: str = "calibrated", state: str = "known") -> dict:
    risk = score / 20.0
    fields = {
        name: {
            "uncertainty_1_20": score,
            "risk_upper_95": risk,
            "calibration_state": calibration_state,
            "value_state": state,
        }
        for name in TRAINING_QUALITY_FIELDS
    }
    return {
        "record_uncertainty_1_20": score,
        "fields": fields,
    }


def _record(score: int, *, sprite_id: str = "sprite", **extra: object) -> dict:
    row = {
        "sprite_id": sprite_id,
        "canonical_object": "sword",
        "object_name": "sword",
        "category": "weapon",
        "domain": "inventory_icon",
        "role": "weapon",
        "label_quality": _quality(score),
    }
    row.update(extra)
    return row


def test_uncertainty_mapping_and_loss_weight_are_monotonic() -> None:
    assert uncertainty_from_risk_upper(0.0) == 1
    assert uncertainty_from_risk_upper(0.08) == 2
    assert uncertainty_from_risk_upper(1.0) == 20
    weights = [loss_weight_for_risk(score / 20.0) for score in range(1, 21)]
    assert weights == sorted(weights, reverse=True)
    assert weights[-1] == 0.05


def test_field_policy_masks_primary_weak_excluded_and_abstained_states() -> None:
    strong = normalize_training_quality(_quality(3), record=_record(3))
    usable_weak = normalize_training_quality(_quality(6), record=_record(6))
    weak = normalize_training_quality(_quality(10), record=_record(10))
    excluded = normalize_training_quality(_quality(14), record=_record(14))
    abstained_raw = _quality(18)
    abstained_raw["fields"]["category"]["value_state"] = "abstained"
    abstained = normalize_training_quality(abstained_raw, record=_record(18))

    assert strong["fields"]["category"]["supervision_mask"] == 1
    assert usable_weak["fields"]["category"]["supervision_mask"] == 1
    assert usable_weak["fields"]["category"]["negative_target_mask"] == 0
    assert weak["fields"]["category"]["supervision_mask"] == 0
    assert weak["fields"]["category"]["auxiliary_mask"] == 1
    assert weak["fields"]["category"]["negative_target_mask"] == 0
    assert weak["fields"]["category"]["unknown_is_negative"] is False
    assert excluded["fields"]["category"]["conditioning_mask"] == 0
    assert abstained["fields"]["category"]["loss_weight"] == 0.0


def test_unresolved_contradiction_cannot_remain_strong() -> None:
    raw = _quality(2)
    raw["unresolved_conflicts"] = ["filename_visual_color"]
    normalized = normalize_training_quality(raw, record=_record(2))
    assert normalized["record_uncertainty_1_20"] >= 9
    assert normalized["record_uncertainty_band"] == "auxiliary_only"
    assert normalized["record_loss_weight"] <= 0.55
    assert normalized["fields"]["canonical_object"]["supervision_mask"] == 0


def test_uncalibrated_field_without_risk_gets_conservative_score() -> None:
    raw = {
        "fields": {
            "category": {"value_state": "known", "calibration_state": "uncalibrated"},
        }
    }
    normalized = normalize_training_quality(raw, record={"category": "weapon"})
    assert normalized["fields"]["category"]["uncertainty_1_20"] == 18
    assert normalized["fields"]["category"]["supervision_mask"] == 0


def test_legacy_rows_remain_loadable_but_are_not_claimed_as_scored() -> None:
    assert extract_training_quality({"category": "weapon"}) is None
    payload = quality_tensor_payload({"category": "weapon"})
    assert payload["quality_present"] is False
    assert payload["record_loss_weight"] == 1.0
    assert not any(payload["field_supervision_mask"])
    assert not any(payload["field_negative_target_mask"])


def test_known_unknown_missing_abstained_and_oov_are_distinct() -> None:
    vocab = build_structured_conditioning_vocab([{"category": "weapon", "object_name": "sword"}])
    known = encode_structured_conditioning({"category": "weapon"}, vocab)
    missing = encode_structured_conditioning({}, vocab)
    unknown = encode_structured_conditioning({"category": "unknown"}, vocab)
    oov = encode_structured_conditioning({"category": "armor"}, vocab)
    abstained_quality = _quality(18)
    abstained_quality["fields"]["category"]["value_state"] = "abstained"
    abstained = encode_structured_conditioning({"category": "weapon", "label_quality": abstained_quality}, vocab)
    ids = {
        known["category_status_id"],
        missing["category_status_id"],
        unknown["category_status_id"],
        abstained["category_status_id"],
        oov["category_status_id"],
    }
    assert len(ids) == 5
    assert missing["category_status_id"] == FIELD_VALUE_STATE_IDS["missing"]
    assert known["category_status_id"] == FIELD_VALUE_STATE_IDS["known"]
    assert unknown["category_status_id"] == FIELD_VALUE_STATE_IDS["unknown"]
    assert abstained["category_status_id"] == FIELD_VALUE_STATE_IDS["abstained"]
    assert oov["category_status_id"] == FIELD_VALUE_STATE_IDS["out_of_vocabulary"]
    assert abstained["category_id"] == 0


def test_structured_conditioning_reads_v4_axes_without_promoting_visual_material() -> None:
    record = {
        "conditioning": {
            "semantic_v4": {
                "canonical_object": "buckler",
                "category": "armor",
                "role": "defensive_equipment",
                "explicit_material": "iron",
                "visual_material_cue": "metallic",
                "shape": {"silhouette": "round", "structure": ["rimmed", "bossed"]},
                "color_roles": {"primary": ["gray"], "outline": ["black"]},
            }
        }
    }
    fields = extract_structured_fields(record)
    assert fields["object_name"] == "buckler"
    assert fields["materials"] == ["iron"]
    assert "metallic" not in fields["materials"]
    assert fields["colors"] == ["gray"]
    assert fields["shapes"] == ["round", "rimmed", "bossed"]
    assert fields["functions"] == ["defensive_equipment"]


def test_field_masks_remove_excluded_values_from_text_and_semantic_conditioning() -> None:
    raw_quality = _quality(3)
    raw_quality["fields"]["explicit_material"].update(uncertainty_1_20=14, risk_upper_95=0.70)
    record = {
        "canonical_object": "sword",
        "category": "weapon",
        "explicit_material": "iron",
        "conditioning": {
            "semantic_v4": {
                "canonical_object": "sword",
                "category": "weapon",
                "explicit_material": "iron",
            },
            "semantic_v3": {"attributes": {"materials": ["iron"]}},
            "dropped_attributes": {"materials": ["iron"]},
        },
    }
    record["label_quality"] = normalize_training_quality(raw_quality, record=record)
    filtered, caption = apply_conditioning_policy(record, "iron sword icon")
    assert caption == "sword icon"
    assert "explicit_material" not in filtered
    assert "explicit_material" not in filtered["conditioning"]["semantic_v4"]
    assert "materials" not in filtered["conditioning"]["semantic_v3"]["attributes"]
    assert "materials" not in filtered["conditioning"]["dropped_attributes"]
    assert filtered["canonical_object"] == "sword"


def test_excluded_canonical_object_produces_unconditioned_caption() -> None:
    quality = _quality(14)
    record = {"canonical_object": "sword", "object_name": "sword"}
    record["label_quality"] = normalize_training_quality(quality, record=record)
    filtered, caption = apply_conditioning_policy(record, "red sword icon")
    assert caption == ""
    assert "canonical_object" not in filtered
    assert "object_name" not in filtered


def test_effective_sample_size_uses_kish_formula() -> None:
    assert effective_sample_size([1.0, 1.0, 1.0]) == pytest.approx(3.0)
    assert effective_sample_size([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_sample_size([]) == 0.0


def test_dataset_training_and_evaluation_reports_stratify_quality() -> None:
    rows = [
        _record(3, sprite_id="strong", source_pack="pack_a", split="train"),
        _record(7, sprite_id="weak", source_pack="pack_b", split="validation", unseen_pack=True),
        _record(11, sprite_id="aux", propagation_relation="recolor", split="open_set_test"),
        {"sprite_id": "legacy", "category": "misc_item", "split": "train"},
    ]
    dataset = dataset_uncertainty_report(rows)
    training = training_uncertainty_report(rows)
    evaluation = evaluation_uncertainty_report(rows)

    assert dataset["record_uncertainty_histogram"]["3"] == 1
    assert dataset["legacy_unscored_count"] == 1
    assert "pack_b" in dataset["uncertainty_by"]["pack"]
    contribution = training["training_contribution"]
    assert contribution["raw_records"] == 4
    assert 0 < contribution["effective_sample_size"] <= 4
    assert contribution["sample_count_by_uncertainty_band"]["strong"] == 1
    assert evaluation["strata"]["strong_labels"] == 1
    assert evaluation["strata"]["open_set"] == 1
    assert evaluation["strata"]["propagated_labels"] == 1


def test_dataset_grade_is_rule_based_and_explained() -> None:
    perfect = {
        "strong_label_coverage": 0.9,
        "calibrated_coverage": 0.9,
        "abstention_rate": 0.01,
        "critical_field_risk_upper": 0.04,
    }
    breakdown = quality_grade_breakdown(perfect)
    assert breakdown["quality_grade"] == "A"
    assert breakdown["rules_version"] == "dataset_quality_grade_v1"
    score = dataset_quality_score([_record(3, source_pack="pack_a")])
    assert "grade_breakdown" in score
    assert "grade_lowering_dimensions" in score


def test_training_manifest_carries_v4_quality_without_changing_legacy_rows(tmp_path: Path) -> None:
    dataset = make_semantic_dataset(tmp_path / "dataset", default_specs())
    manifest_path = dataset / "manifest_train.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["label_v4"] = {"label_quality": _quality(4)}
    manifest_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    result = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=4)
    v4_rows = [row for row in result.rows if row["sprite_id"] == rows[0]["sprite_id"]]
    assert v4_rows and v4_rows[0]["label_quality"]["record_uncertainty_1_20"] == 4
    assert any("label_quality" not in row for row in result.rows if row["sprite_id"] != rows[0]["sprite_id"])


def test_loader_carries_quality_vectors_and_normalized_record_weight(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch

    dataset = make_semantic_dataset(tmp_path / "dataset", default_specs())
    built = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=3)
    built.rows[0]["label_quality"] = normalize_training_quality(_quality(6), record=built.rows[0])
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, built.rows)
    loaded = SpriteTrainingDataset(dataset, manifest, max_records=2)
    sample = loaded[0]
    batch = collate_sprite_batch([loaded[0], loaded[1]])

    assert sample["label_field_loss_weight"].shape == (len(TRAINING_QUALITY_FIELDS),)
    assert sample["record_loss_weight"].dtype == torch.float32
    assert batch["record_loss_weight"].shape == (2,)
    assert batch["label_field_supervision_mask"].shape == (2, len(TRAINING_QUALITY_FIELDS))


def test_weighted_reduction_is_normalized_not_divided_by_batch_size() -> None:
    torch = pytest.importorskip("torch")
    from spritelab.training.generator_challenger import _normalized_weighted_mean

    values = torch.tensor([1.0, 3.0])
    weights = torch.tensor([1.0, 3.0])
    assert float(_normalized_weighted_mean(values, weights)) == pytest.approx(2.5)
    assert float(_normalized_weighted_mean(values, torch.zeros(2))) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        _normalized_weighted_mean(values, torch.tensor([1.0, -1.0]))
