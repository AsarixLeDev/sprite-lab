from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.dataset_v5.builder import alpha_mask_sha256, canonical_rgba_sha256, validate_no_leakage
from spritelab.dataset_v5.policy_v2 import (
    PolicyV2Config,
    WeightingPolicy,
    _hard_cap_subset,
    assign_policy_splits,
    build_policy_groups,
    compute_sampling_weights,
    exclusion_summary,
    source_sheet_identity,
    verify_policy_preview,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PREVIEW = ROOT / "datasets/sprite_lab_multisource_v5_policy_v2_core_plus_weighted_sampling_preview"


def _record(
    sprite_id: str,
    *,
    pack: str = "pack_a",
    artist: str = "artist_a",
    obj: str = "sword",
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: np.ndarray | None = None,
    source_image: str | None = None,
    archive_member: str | None = None,
    source_hash: str | None = None,
    strict: bool = True,
) -> dict:
    mask = np.zeros((32, 32), dtype=np.uint8) if alpha is None else np.asarray(alpha, dtype=np.uint8)
    if alpha is None:
        mask[8:24, 12:20] = 1
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[mask > 0, :3] = color
    rgba[mask > 0, 3] = 255
    return {
        "sprite_id": sprite_id,
        "source_pack": pack,
        "source_id": pack,
        "source_family": "family_a",
        "sub_artist": artist,
        "author": artist,
        "object_name": obj,
        "category": "weapon",
        "is_supervised": True,
        "strict_quality": strict,
        "source_image": source_image or f"icons/{sprite_id}.png",
        "archive_member": archive_member or f"icons/{sprite_id}.png",
        "downloaded_file_hash": source_hash or f"hash-{sprite_id}",
        "source_sheet": f"{pack}:icons",
        "cell_coordinates": None,
        "animation_group": "",
        "known_variant_family": "",
        "declared_variant_ids": [],
        "exported_rgba_hash": canonical_rgba_sha256(rgba),
        "alpha_mask_hash": alpha_mask_sha256(mask),
        "_rgba": rgba,
        "_alpha": mask,
    }


def _distinct_records(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[1 + index % 20 : 3 + index % 20, 1 + (index * 3) % 25 : 3 + (index * 3) % 25] = 1
        rows.append(
            _record(
                f"r{index}",
                pack=f"pack_{index % 5}",
                artist=f"artist_{index % 7}",
                obj=f"object_{index % 4}",
                color=(index, 20, 30),
                alpha=mask,
            )
        )
    return rows


def test_soft_pack_cap_downweights_without_deleting_membership():
    records = [_record(f"a{i}", pack="dominant", color=(i, 0, 0)) for i in range(8)] + [
        _record(f"b{i}", pack=f"other{i}", color=(0, i, 0)) for i in range(4)
    ]
    families = {row["sprite_id"]: row["sprite_id"] for row in records}
    weights, report = compute_sampling_weights(
        records,
        families,
        WeightingPolicy(
            mode="soft_cap",
            soft_target_share=0.5,
            artist_exponent=0,
            source_family_exponent=0,
            canonical_object_exponent=0,
        ),
    )
    assert len(weights) == len(records)
    dominant = next(row for row in report["effective_pack_distribution"] if row["value"] == "dominant")
    assert dominant["share"] < 8 / 12


def test_soft_artist_cap_downweights_without_deleting_membership():
    records = [_record(f"a{i}", artist="dominant", color=(i, 0, 0)) for i in range(8)] + [
        _record(f"b{i}", artist=f"other{i}", color=(0, i, 0)) for i in range(4)
    ]
    families = {row["sprite_id"]: row["sprite_id"] for row in records}
    weights, report = compute_sampling_weights(
        records,
        families,
        WeightingPolicy(
            mode="soft_cap",
            soft_target_share=0.5,
            pack_exponent=0,
            source_family_exponent=0,
            canonical_object_exponent=0,
        ),
    )
    assert len(weights) == len(records)
    dominant = next(row for row in report["effective_artist_distribution"] if row["value"] == "dominant")
    assert dominant["share"] < 8 / 12


def test_inverse_frequency_weighting_favors_rare_pack():
    records = [_record(f"a{i}", pack="common", color=(i, 0, 0)) for i in range(4)] + [_record("rare", pack="rare")]
    families = {row["sprite_id"]: row["sprite_id"] for row in records}
    weights, _ = compute_sampling_weights(
        records,
        families,
        WeightingPolicy(artist_exponent=0, source_family_exponent=0, canonical_object_exponent=0),
    )
    assert weights["rare"] > weights["a0"]


def test_temperature_weighting_is_less_aggressive_below_one():
    records = [_record(f"a{i}", pack="common", color=(i, 0, 0)) for i in range(9)] + [_record("rare", pack="rare")]
    families = {row["sprite_id"]: row["sprite_id"] for row in records}
    inverse, _ = compute_sampling_weights(
        records,
        families,
        WeightingPolicy(artist_exponent=0, source_family_exponent=0, canonical_object_exponent=0),
    )
    temperature, _ = compute_sampling_weights(
        records,
        families,
        WeightingPolicy(
            mode="temperature_sampling",
            temperature=0.5,
            artist_exponent=0,
            source_family_exponent=0,
            canonical_object_exponent=0,
        ),
    )
    assert temperature["rare"] / temperature["a0"] < inverse["rare"] / inverse["a0"]


def test_geometry_family_normalization_prevents_recolor_gain():
    records = [_record(f"recolor{i}", color=(i, 0, 0)) for i in range(4)]
    other_mask = np.zeros((32, 32), dtype=np.uint8)
    other_mask[1:4, 1:4] = 1
    records.append(_record("single", alpha=other_mask))
    families = {row["sprite_id"]: "recolors" if row["sprite_id"].startswith("recolor") else "single" for row in records}
    weights, _ = compute_sampling_weights(
        records,
        families,
        WeightingPolicy(pack_exponent=0, artist_exponent=0, source_family_exponent=0, canonical_object_exponent=0),
    )
    assert sum(weights[f"recolor{i}"] for i in range(4)) == pytest.approx(weights["single"])


def test_sampling_weights_are_bounded_and_deterministic():
    records = _distinct_records(30)
    families = {row["sprite_id"]: f"g{index % 3}" for index, row in enumerate(records)}
    policy = WeightingPolicy(minimum_weight=0.2, maximum_weight=2.0)
    first, first_report = compute_sampling_weights(records, families, policy)
    second, second_report = compute_sampling_weights(list(reversed(records)), families, policy)
    assert first == second
    assert first_report["weight_digest"] == second_report["weight_digest"]
    assert min(first.values()) >= 0.2
    assert max(first.values()) <= 2.0


def test_hard_caps_are_evaluation_only_for_recommended_policy():
    records = [_record(f"dominant{i}", pack="dominant", artist="dominant", color=(i, 0, 0)) for i in range(30)]
    records += [_record(f"other{i}", pack=f"pack{i}", artist=f"artist{i}", color=(0, i, 0)) for i in range(10)]
    subset = _hard_cap_subset(records, 0.30, 0.30, seed=7)
    assert 0 < len(subset) < len(records)
    assert len(records) == 40


def test_split_construction_has_non_empty_regular_test_and_validation():
    records = []
    for index in range(80):
        mask = np.zeros((32, 32), dtype=np.uint8)
        height, width = 1 + index // 25, 1 + index % 25
        mask[1 : 1 + height, 1 : 1 + width] = 1
        records.append(
            _record(
                f"split{index}",
                pack=f"pack_{index % 5}",
                artist=f"artist_{index % 7}",
                obj=f"object_{index % 4}",
                color=(index, 20, 30),
                alpha=mask,
            )
        )
    _relations, groups, _audit = build_policy_groups(records)
    assignments = assign_policy_splits(records, groups, PolicyV2Config(source_ood_packs=()))
    assert "test" in assignments.values()
    assert "validation" in assignments.values()
    train_objects = {row["object_name"] for row in records if assignments[row["sprite_id"]] == "train"}
    assert all(
        row["object_name"] in train_objects
        for row in records
        if assignments[row["sprite_id"]] in {"test", "validation"}
    )


def test_precise_sheet_grouping_uses_archive_hash_for_slices():
    one = _record("one", source_image="run/sliced/sheet/a.png", archive_member="sheet.png", source_hash="same")
    two = _record(
        "two", color=(0, 0, 1), source_image="run/sliced/sheet/b.png", archive_member="sheet.png", source_hash="same"
    )
    assert source_sheet_identity(one) == source_sheet_identity(two)
    relations, _groups, audit = build_policy_groups([one, two])
    assert any(row["kind"] == "source_sheet_siblings" for row in relations)
    assert audit["fallback_record_count"] == 0


def test_directory_fallback_does_not_group_unrelated_standalone_images():
    one = _record("one", source_image="icons/one.png", archive_member="one.png", source_hash="one")
    two = _record("two", color=(0, 0, 1), source_image="icons/two.png", archive_member="two.png", source_hash="two")
    relations, groups, audit = build_policy_groups([one, two])
    assert not any(row["kind"] == "source_sheet_siblings" for row in relations)
    assert len(groups) == 1  # exact alpha/recolor still correctly groups them
    assert audit["fallback_record_count"] == 0


def test_unique_exclusion_and_multiple_reason_occurrence_counting():
    exclusions = [
        {
            "sprite_id": "a",
            "stage": "validate",
            "reason_code": "first",
            "details": {"all_reasons": ["first", "second"]},
        },
        {"sprite_id": "b", "stage": "dedupe", "reason_code": "duplicate"},
    ]
    occurrences, report = exclusion_summary(exclusions)
    assert report["unique_excluded_records"] == 2
    assert report["reason_occurrences"] == 3
    assert report["primary_exclusion_reasons"] == {"duplicate": 1, "first": 1}
    assert report["secondary_reason_occurrences"] == {"second": 1}
    assert len(occurrences) == 3


def test_all_previous_leakage_guarantees_remain_hard():
    hard_kinds = [
        "exact_exported_rgba",
        "exact_alpha_mask_recolor",
        "declared_variant_family",
        "source_sheet_siblings",
    ]
    for kind in hard_kinds:
        with pytest.raises(ValueError, match="hard split leakage"):
            validate_no_leakage(
                {"a": "train", "b": "test"},
                [{"relation_id": kind, "kind": kind, "members": ["a", "b"], "hard_split_constraint": True}],
            )


def test_loader_weighted_sampler_compatibility_when_preview_exists():
    if not POLICY_PREVIEW.is_dir():
        pytest.skip("generated policy preview is built by the preview smoke stage")
    result = verify_policy_preview(POLICY_PREVIEW)
    assert result["training_loader_contract"]["ok"]
    rows = [
        json.loads(line)
        for line in (POLICY_PREVIEW / "training_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    train = [row for row in rows if row["split"] == "train"]
    torch = pytest.importorskip("torch")
    sampler = torch.utils.data.WeightedRandomSampler(
        [row["sampling_weight"] for row in train], num_samples=min(8, len(train)), replacement=True
    )
    assert len(list(sampler)) == min(8, len(train))
