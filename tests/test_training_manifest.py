from __future__ import annotations

import random
from pathlib import Path

import pytest
from _semantic_dataset import default_specs, make_semantic_dataset

from spritelab.dataset_maker.training_manifest import (
    CAPTION_POLICIES,
    SCHEMA_VERSION,
    build_training_manifest,
    sample_caption_variants,
)
from spritelab.harvest.semantic_v3 import build_semantic_v3_record


def _dataset(tmp_path: Path, *, write_semantic: bool = True) -> Path:
    return make_semantic_dataset(tmp_path / "ds", default_specs(), write_semantic=write_semantic)


def _semantic(object_name: str, category: str = "item_icon", **kw):
    prediction = {
        "sprite_id": f"s_{object_name}",
        "candidate_object_names": [],
        "safe_prefill": {
            "category": category,
            "object_name": object_name,
            "tags": kw.get("tags", [object_name, category]),
            "materials": kw.get("materials", []),
            "mood": [],
        },
        "visual_facts": {"dominant_colors": kw.get("dominant_colors", []), "shape_hints": []},
        "vlm_descriptor": {"object_name": "", "alternative_object_names": []},
        "source_profile": {"name": "test", "domain": "rpg_icons"},
        "bucket": "auto_filename_trusted",
    }
    return build_semantic_v3_record(prediction)


def test_build_emits_exactly_variants_per_sprite(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=8, caption_policy="mixed", seed=1)
    assert result.source_records == 6
    assert result.unique_sprites == 6
    assert len(result.rows) == 6 * 8
    from collections import Counter

    per_sprite = Counter(row["sprite_id"] for row in result.rows)
    assert set(per_sprite.values()) == {8}


def test_rows_include_split_npz_file_and_row(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=4, caption_policy="mixed", seed=1)
    for row in result.rows:
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["split"] in {"train", "val", "test"}
        assert row["npz_file"] == f"{row['split']}.npz"
        assert isinstance(row["npz_row"], int) and row["npz_row"] >= 0
        assert row["caption"]
        assert row["caption_type"]
        assert row["base_object"]


def test_npz_row_points_at_matching_sprite(tmp_path: Path) -> None:
    import numpy as np

    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=1)
    npz_ids = {}
    for split in ("train", "val", "test"):
        with np.load(dataset / f"{split}.npz", allow_pickle=False) as data:
            npz_ids[split] = [str(v) for v in np.asarray(data["sprite_id"])]
    for row in result.rows:
        assert npz_ids[row["split"]][row["npz_row"]] == row["sprite_id"]


def test_build_is_deterministic_with_same_seed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    a = build_training_manifest(dataset, variants_per_sprite=8, caption_policy="mixed", seed=7)
    b = build_training_manifest(dataset, variants_per_sprite=8, caption_policy="mixed", seed=7)
    assert [r["caption"] for r in a.rows] == [r["caption"] for r in b.rows]


def test_different_seed_changes_selection(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    a = build_training_manifest(dataset, variants_per_sprite=8, caption_policy="mixed", seed=1)
    b = build_training_manifest(dataset, variants_per_sprite=8, caption_policy="mixed", seed=2)
    assert [r["caption"] for r in a.rows] != [r["caption"] for r in b.rows]


@pytest.mark.parametrize("policy", list(CAPTION_POLICIES))
def test_all_policies_produce_nonempty_captions(tmp_path: Path, policy: str) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=6, caption_policy=policy, seed=3)
    assert len(result.rows) == 6 * 6
    for row in result.rows:
        assert row["caption"].strip()


def test_object_only_policy_prefers_object_captions(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=8, caption_policy="object_only", seed=3)
    types = {row["caption_type"] for row in result.rows}
    assert types <= {"object", "minimal"}


def test_style_aware_policy_uses_style_captions(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=6, caption_policy="style_aware", seed=3)
    assert all(row["caption_type"] == "style_aware" for row in result.rows)
    assert any("pixel art" in row["caption"] for row in result.rows)


def test_minimal_policy_produces_short_captions(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=4, caption_policy="minimal", seed=3)
    assert all(row["caption_type"] in {"minimal", "object"} for row in result.rows)


def test_captions_do_not_invent_unrelated_object_terms() -> None:
    record = _semantic("red_potion", "item_icon", dominant_colors=["red", "black", "white"])
    rng = random.Random("seed")
    variants = sample_caption_variants(record, policy="mixed", rng=rng, count=12)
    allowed = {
        "32x32", "pixel", "art", "fantasy", "rpg", "icon", "centered", "made", "of",
        "outline", "transparent", "background", "item",
    }
    allowed.update(record.open_name.split())
    allowed.update(record.base_object.split("_"))
    attrs = record.attributes
    for group in (attrs.colors, attrs.materials, attrs.shapes, attrs.effects, attrs.state, attrs.function, attrs.mood, attrs.parts):
        for value in group:
            allowed.update(value.split("_"))
    for variant in variants:
        for word in variant.caption.replace(",", " ").lower().split():
            assert word in allowed, f"ungrounded word {word!r} in {variant.caption!r}"


def test_semantic_dropout_never_removes_all_meaning() -> None:
    record = _semantic("golden_sword", "weapon", dominant_colors=["gold", "yellow", "black"], materials=["metal"])
    rng = random.Random("seed")
    variants = sample_caption_variants(record, policy="mixed", rng=rng, count=12)
    base = record.base_object.replace("_", " ")
    for variant in variants:
        text = variant.caption.lower()
        assert text.strip()
        # every caption keeps a noun: base object, open name, or a category/icon word
        assert (
            base in text
            or record.open_name.lower() in text
            or "icon" in text
            or record.category in text
        )


def test_dropout_mask_is_recorded(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=8, caption_policy="mixed", seed=3)
    for row in result.rows:
        conditioning = row["conditioning"]
        assert "kept_attributes" in conditioning
        assert "dropped_attributes" in conditioning
        mask = row["dropout_mask"]
        assert any(key.startswith("kept_") for key in mask)
    # at least some rows actually drop the colour
    dropped_color = [
        row for row in result.rows if row["conditioning"]["dropped_attributes"].get("colors")
    ]
    assert dropped_color


def test_negative_tags_are_copied(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=3)
    for row in result.rows:
        assert row["negative_tags"]
        assert "watermark" in row["negative_tags"]


def test_attribute_policy_can_decompose(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = build_training_manifest(dataset, variants_per_sprite=6, caption_policy="attribute", seed=3)
    assert all(row["caption_type"] == "attribute" for row in result.rows)


def test_works_without_semantic_v3(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, write_semantic=False)
    result = build_training_manifest(dataset, variants_per_sprite=4, caption_policy="mixed", seed=3)
    assert len(result.rows) == 6 * 4
    for row in result.rows:
        assert row["caption"].strip()
        assert row["base_object"]


def test_unknown_policy_raises(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(ValueError):
        build_training_manifest(dataset, variants_per_sprite=4, caption_policy="bogus", seed=1)
