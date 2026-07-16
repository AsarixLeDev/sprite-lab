from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.codec.roles import ROLE_LIGHT, ROLE_MIDTONE, ROLE_OUTLINE
from spritelab.training.data import SpriteTrainingDataset
from spritelab.training.role_ramp_transplant import (
    RampDonor,
    RampSlot,
    RoleRampLibrary,
    RoleRampTransplantAuditConfig,
    RoleRampTransplantConfig,
    RoleRampTransplantReviewConfig,
    apply_role_ramp_transplant,
    audit_role_ramp_transplant,
    build_role_ramp_library,
    primary_fill_slots,
    review_role_ramp_transplant,
    summarize_role_ramp_audit,
)


def _dataset(tmp_path: Path, *, trusted: bool = True) -> tuple[Path, Path, list[dict[str, object]]]:
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True)
    alpha = np.zeros((2, 32, 32), dtype=np.uint8)
    alpha[:, 6:26, 6:26] = 255
    index = np.zeros((2, 32, 32), dtype=np.int16)
    index[:, 6:26, 6:18] = 1
    index[:, 6:26, 18:26] = 2
    index[:, 6:9, 6:26] = 3
    roles = np.zeros_like(index, dtype=np.uint8)
    roles[:, 6:26, 6:18] = ROLE_MIDTONE
    roles[:, 6:26, 18:26] = ROLE_LIGHT
    roles[:, 6:9, 6:26] = ROLE_OUTLINE
    if not trusted:
        roles.fill(0)
    palette = np.zeros((2, 8, 3), dtype=np.uint8)
    # Recipient: red ramp. Donor: blue ramp. Slot zero deliberately has a tempting green color.
    palette[0] = [[0, 255, 0], [180, 45, 45], [240, 110, 110], [24, 18, 18], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    palette[1] = [[0, 255, 0], [35, 70, 190], [100, 145, 245], [18, 22, 40], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    mask = np.zeros((2, 8), dtype=bool)
    mask[:, :4] = True
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index,
        role_map=roles,
        palette=palette,
        palette_mask=mask,
        category_id=np.array([1, 1], dtype=np.int64),
        sprite_id=np.array(["red_sword", "blue_sword"], dtype=np.str_),
    )
    records: list[dict[str, object]] = [
        {
            "sprite_id": "red_sword",
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": 0,
            "caption": "red sword",
            "colors": ["red"],
            "primary_color": "red",
            "target_semantics": {"base_object": "sword", "attributes": {"colors": ["red"]}},
        },
        {
            "sprite_id": "blue_sword",
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": 1,
            "caption": "blue sword",
            "colors": ["blue"],
            "primary_color": "blue",
            "target_semantics": {"base_object": "sword", "attributes": {"colors": ["blue"]}},
        },
    ]
    manifest = dataset / "training_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return dataset, manifest, records


def _arrays(dataset: Path) -> dict[str, np.ndarray]:
    with np.load(dataset / "train.npz", allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def test_ramp_library_excludes_slot_zero_and_groups_role_buckets(tmp_path: Path) -> None:
    dataset, _manifest, records = _dataset(tmp_path)
    library = build_role_ramp_library(dataset, records, exclude_families=())
    assert {donor.color_family for donor in library.donors} == {"red", "blue"}
    for donor in library.donors:
        assert all(slot.palette_index != 0 for slot in donor.slots)
        assert {slot.role_bucket for slot in donor.slots} >= {"midtone", "light", "outline"}


def test_primary_fill_slots_exclude_outline_shadow_and_light() -> None:
    slots = [
        RampSlot(1, "midtone", 80, 0.40, (0.5, 0.1, 0.1)),
        RampSlot(2, "light", 90, 0.45, (0.8, 0.1, 0.1)),
        RampSlot(3, "shadow", 20, 0.10, (0.2, 0.1, 0.1)),
        RampSlot(4, "outline", 10, 0.05, (0.1, 0.0, 0.0)),
    ]
    assert [slot.palette_index for slot in primary_fill_slots(slots, min_coverage=0.03)] == [1]


def test_transplant_requires_matching_target_family_and_preserves_geometry(tmp_path: Path) -> None:
    dataset, _manifest, records = _dataset(tmp_path)
    arrays = _arrays(dataset)
    library = build_role_ramp_library(dataset, records, exclude_families=())
    alpha = arrays["alpha"][0].copy()
    index = arrays["index_map"][0].copy()
    roles = arrays["role_map"][0].copy()
    palette = arrays["palette"][0].astype(np.float32) / 255.0
    result = apply_role_ramp_transplant(
        index_map=index,
        alpha=alpha,
        role_map=roles,
        palette_rgb=palette,
        palette_mask=arrays["palette_mask"][0],
        record=records[0],
        caption="red sword",
        sprite_id="red_sword",
        library=library,
        config=RoleRampTransplantConfig(enabled=True, prob=1.0, keep_original_prob=0.0, exclude_families=(), seed=5),
    )
    assert result.applied is True
    assert result.target_color_family == "blue"
    assert result.donor_sprite_id == "blue_sword"
    assert np.array_equal(alpha, arrays["alpha"][0])
    assert np.array_equal(index, arrays["index_map"][0])
    assert np.array_equal(roles, arrays["role_map"][0])
    assert not np.array_equal(result.palette_rgb[1:3], palette[1:3])
    assert result.record["primary_color"] == "blue"
    assert result.record["colors"] == ["blue"]
    assert "blue" in result.caption
    assert result.safety["alpha_unchanged_exact"] is True
    assert result.safety["index_map_unchanged_exact"] is True
    assert result.safety["role_map_unchanged_exact"] is True


def test_fill_mismatch_skips_without_mutating_prompt_or_palette(tmp_path: Path) -> None:
    dataset, _manifest, records = _dataset(tmp_path)
    arrays = _arrays(dataset)
    # Claim a blue donor family but give every donor role a neutral ramp. The
    # post-transplant primary midtone therefore cannot truthfully be blue.
    neutral_slots = tuple(
        RampSlot(index, role, 100, 0.25, (0.7, 0.0, 0.0))
        for index, role in ((1, "midtone"), (2, "light"), (3, "outline"))
    )
    library = RoleRampLibrary((RampDonor("neutral", "blue", neutral_slots, True, primary_fill_family="blue"),))
    original = arrays["palette"][0].astype(np.float32) / 255.0
    result = apply_role_ramp_transplant(
        index_map=arrays["index_map"][0],
        alpha=arrays["alpha"][0],
        role_map=arrays["role_map"][0],
        palette_rgb=original,
        palette_mask=arrays["palette_mask"][0],
        record=records[0],
        caption="red sword",
        sprite_id="red_sword",
        library=library,
        config=RoleRampTransplantConfig(
            enabled=True, prob=1.0, keep_original_prob=0.0, exclude_families=(), max_resample_attempts=2, seed=1
        ),
    )
    assert result.applied is False
    assert result.ineligibility_reason == "post_transplant_fill_family_mismatch"
    assert np.array_equal(result.palette_rgb, original)
    assert result.record["primary_color"] == "red"
    assert result.caption == "red sword"


def test_excluded_families_and_trusted_role_requirement_are_enforced(tmp_path: Path) -> None:
    dataset, _manifest, records = _dataset(tmp_path)
    arrays = _arrays(dataset)
    library = build_role_ramp_library(dataset, records, exclude_families=())
    result = apply_role_ramp_transplant(
        index_map=arrays["index_map"][0],
        alpha=arrays["alpha"][0],
        role_map=arrays["role_map"][0],
        palette_rgb=arrays["palette"][0].astype(np.float32) / 255.0,
        palette_mask=arrays["palette_mask"][0],
        record=records[0],
        caption="red sword",
        sprite_id="red_sword",
        library=library,
        config=RoleRampTransplantConfig(
            enabled=True, prob=1.0, keep_original_prob=0.0, exclude_families=("blue",), seed=1
        ),
    )
    assert result.applied is False
    assert result.ineligibility_reason == "no_target_family_donor"

    untrusted_dataset, _manifest, untrusted_records = _dataset(tmp_path / "untrusted", trusted=False)
    untrusted = build_role_ramp_library(
        untrusted_dataset, untrusted_records, require_trusted_role_map=True, exclude_families=()
    )
    assert untrusted.donors == ()


def test_review_writes_required_outputs_and_dataset_applies_augmentation(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset, manifest, _records = _dataset(tmp_path)
    out = tmp_path / "review"
    summary = review_role_ramp_transplant(
        RoleRampTransplantReviewConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            max_samples=2,
            role_ramp_transplant_prob=1.0,
            role_ramp_transplant_keep_original_prob=0.0,
            role_ramp_transplant_exclude_families="",
        )
    )
    assert summary["applied_count"] >= 1
    assert (out / "summary.json").is_file()
    assert (out / "transplant_decisions.jsonl").is_file()
    assert (out / "preview_contact_sheet.png").is_file()

    ds = SpriteTrainingDataset(
        dataset,
        manifest,
        palette_swap=None,
        role_ramp_transplant=RoleRampTransplantConfig(
            enabled=True, prob=1.0, keep_original_prob=0.0, exclude_families=(), seed=5
        ),
    )
    sample = ds[0]
    assert sample["role_ramp_transplant"]["role_ramp_transplant_applied"] is True
    assert not np.array_equal(sample["palette_u8"].numpy()[1:3], _arrays(dataset)["palette"][0, 1:3])


def test_audit_distinguishes_donor_prompt_pixel_families_and_writes_outputs(tmp_path: Path) -> None:
    dataset, manifest, _records = _dataset(tmp_path)
    out = tmp_path / "audit"
    summary = audit_role_ramp_transplant(
        RoleRampTransplantAuditConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            max_samples=2,
            role_ramp_transplant_prob=1.0,
            role_ramp_transplant_keep_original_prob=0.0,
            role_ramp_transplant_exclude_families="",
        )
    )
    decisions = [
        json.loads(line) for line in (out / "transplant_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["applied_count"] >= 1
    assert {"recipient_original_family", "donor_family", "target_prompt_family", "dominant_fill_new_family"} <= set(
        decisions[0]
    )
    assert decisions[0]["donor_family"] != decisions[0]["recipient_original_family"]
    assert (out / "summary.json").is_file()
    assert (out / "family_confusion.csv").is_file()
    assert (out / "role_coverage.csv").is_file()
    assert (out / "contact_sheet.png").is_file()


def test_audit_summary_detects_exclusions_prompt_pixel_mismatch_and_role_coverage_failure() -> None:
    rows = [
        {
            "applied": True,
            "excluded_family_violation": True,
            "dominant_fill_target_match": False,
            "prompt_pixel_family_match": False,
            "role_coverage_success": False,
            "target_prompt_family": "blue",
            "dominant_fill_new_family": "red",
            "donor_family": "gray",
            "recipient_original_family": "red",
            "mean_chroma_before": 0.1,
            "mean_chroma_after": 0.2,
            "mean_chroma_delta": 0.1,
            "mean_lightness_before": 0.5,
            "mean_lightness_after": 0.5,
        }
    ]
    summary = summarize_role_ramp_audit(
        rows,
        RoleRampTransplantConfig(enabled=True, prob=0.3, keep_original_prob=0.5, exclude_families=("gray",)),
    )
    assert summary["excluded_family_violation_count"] == 1
    assert summary["dominant_fill_target_match_rate"] == 0.0
    assert summary["prompt_pixel_family_match_rate"] == 0.0
    assert summary["role_coverage_success_rate"] == 0.0
    assert summary["failed"] is True
