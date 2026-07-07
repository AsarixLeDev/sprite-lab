from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.dataset_maker.qa import qa_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.dataset_maker.training_manifest_qa import qa_training_manifest
from spritelab.harvest.merge_datasets import MergeError, derive_source_pack, merge_datasets

SPLITS = ("train", "val", "test")


def _bundle_arrays(sprite_ids: list[str], *, palette_rows: int = 33) -> dict[str, np.ndarray]:
    count = len(sprite_ids)
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, palette_rows, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, palette_rows), dtype=bool)
    for row in range(count):
        # Give each sprite a slightly different opaque block so we can verify
        # rasters are concatenated in the right row order.
        offset = row % 6
        alpha[row, 10 + offset, 10:14] = 1
        index_map[row, 10 + offset, 10:14] = 1
        palette[row, 1] = [10 + row, 50, 80]
        palette_mask[row, 0] = True
        palette_mask[row, 1] = True
    return {
        "alpha": alpha,
        "index_map": index_map,
        "role_map": role_map,
        "palette": palette,
        "palette_mask": palette_mask,
        "category_id": np.ones((count,), dtype=np.int64),
        "sprite_id": np.array(sprite_ids, dtype=np.str_),
    }


def _semantic(object_name: str, base_object: str, color: str = "red") -> dict:
    return {
        "schema_version": "semantic_v3.0",
        "category": "item_icon",
        "object_name": object_name,
        "base_object": base_object,
        "open_name": object_name.replace("_", " "),
        "attributes": {
            "colors": [color],
            "materials": ["glass"],
            "shapes": [],
            "effects": [],
            "state": [],
            "function": ["consumable"],
            "mood": ["fantasy"],
            "style": ["32x32", "pixel_art", "rpg_icon"],
            "parts": [],
            "environment": [],
        },
        "aliases": [base_object],
        "captions": [
            object_name.replace("_", " "),
            f"{color} {base_object} made of glass",
            f"32x32 pixel art {object_name.replace('_', ' ')} icon",
        ],
        "prompt_phrases": [f"32x32 pixel art {object_name.replace('_', ' ')}"],
        "negative_tags": ["photorealistic", "large_scene", "text", "watermark"],
        "source_evidence": {},
        "warnings": [],
    }


def _record(sprite_id: str, split: str, *, object_name: str, base_object: str, color: str = "red") -> dict:
    return {
        "sprite_id": sprite_id,
        "split": split,
        "category": "item_icon",
        "category_id": 1,
        "object_name": object_name,
        "tags": [base_object, "glass", color],
        "materials": ["glass"],
        "mood": ["fantasy"],
        "source_name": "test-source",
        "source_path": f"data/{sprite_id}.png",
        "license": "cc0",
        "author": "tester",
        "notes": "",
        "short_description": object_name.replace("_", " "),
        "has_role_map": True,
        "palette_size": 1,
        "label_v2": {"applied": True, "bucket": "auto_filename_trusted", "flags": ["auto_filename_trusted"]},
        "semantic_v3": _semantic(object_name, base_object, color),
    }


def _write_dataset(
    root: Path, name: str, records_by_split: dict[str, list[dict]], *, palette_rows: int = 33, max_palette_slots: int = 32
) -> Path:
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    categories = {"unknown": 0}
    for split in SPLITS:
        records = records_by_split.get(split, [])
        (dataset_dir / f"manifest_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in records) + ("\n" if records else ""),
            encoding="utf-8",
        )
        ids = [str(r["sprite_id"]) for r in records]
        np.savez_compressed(dataset_dir / f"{split}.npz", **_bundle_arrays(ids, palette_rows=palette_rows))
        for r in records:
            categories.setdefault(str(r["category"]), len(categories))
    (dataset_dir / "dataset_config.json").write_text(
        json.dumps({"dataset_name": name, "max_palette_slots": max_palette_slots}), encoding="utf-8"
    )
    (dataset_dir / "vocab.json").write_text(json.dumps({"category_to_id": categories}), encoding="utf-8")
    return dataset_dir


def _dataset_a(root: Path) -> Path:
    records = {
        "train": [
            _record("a_ruby", "train", object_name="ruby_gem", base_object="gem", color="red"),
            _record("a_sapphire", "train", object_name="sapphire_gem", base_object="gem", color="blue"),
            _record("a_emerald", "train", object_name="emerald_gem", base_object="gem", color="green"),
        ],
        "val": [_record("a_topaz", "val", object_name="topaz_gem", base_object="gem", color="yellow")],
        "test": [_record("a_opal", "test", object_name="opal_gem", base_object="gem", color="white")],
    }
    return _write_dataset(root, "packA_label_v2_semantic_v3", records)


def _dataset_b(root: Path) -> Path:
    records = {
        "train": [
            _record("b_red", "train", object_name="red_potion", base_object="potion", color="red"),
            _record("b_blue", "train", object_name="blue_potion", base_object="potion", color="blue"),
        ],
        "val": [_record("b_green", "val", object_name="green_potion", base_object="potion", color="green")],
        "test": [_record("b_pink", "test", object_name="pink_potion", base_object="potion", color="pink")],
    }
    return _write_dataset(root, "packB_label_v2_semantic_v3", records)


# ---------------------------------------------------------------------------


def test_derive_source_pack_strips_known_suffixes() -> None:
    assert derive_source_pack("packA_label_v2_semantic_v3") == "packA"
    assert derive_source_pack("packB_semantic_v3") == "packB"
    assert derive_source_pack("packC_label_v2") == "packC"
    assert derive_source_pack("raw_pack") == "raw_pack"


def test_merges_two_tiny_datasets(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    result = merge_datasets([a, b], out, split_policy="preserve")
    assert result.ok, result.errors
    assert result.total_records == 9
    assert out.is_dir()
    assert (out / "train.npz").is_file()
    assert (out / "manifest_train.jsonl").is_file()


def test_preserve_split_keeps_source_assignment(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    merge_datasets([a, b], out, split_policy="preserve")
    split_by_source_id: dict[str, str] = {}
    for split in SPLITS:
        for line in (out / f"manifest_{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            split_by_source_id[record["source_sprite_id"]] = split
    assert split_by_source_id["a_ruby"] == "train"
    assert split_by_source_id["a_topaz"] == "val"
    assert split_by_source_id["a_opal"] == "test"
    assert split_by_source_id["b_green"] == "val"


def test_reshuffle_is_deterministic(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)

    def run(out: Path) -> dict[str, str]:
        merge_datasets([a, b], out, split_policy="reshuffle", seed=7, overwrite=True)
        mapping: dict[str, str] = {}
        for split in SPLITS:
            for line in (out / f"manifest_{split}.jsonl").read_text().splitlines():
                record = json.loads(line)
                mapping[record["source_sprite_id"]] = split
        return mapping

    first = run(tmp_path / "m1")
    second = run(tmp_path / "m2")
    assert first == second
    # A reshuffle should still populate all three splits given enough records.
    assert set(first.values()) == {"train", "val", "test"}


def test_duplicate_sprite_ids_across_sources_are_prefixed(tmp_path: Path) -> None:
    records = {
        "train": [
            _record("s0", "train", object_name="ruby_gem", base_object="gem"),
            _record("s1", "train", object_name="sapphire_gem", base_object="gem", color="blue"),
        ],
        "val": [_record("s2", "val", object_name="topaz_gem", base_object="gem", color="yellow")],
        "test": [_record("s3", "test", object_name="opal_gem", base_object="gem", color="white")],
    }
    a = _write_dataset(tmp_path, "packA_label_v2_semantic_v3", records)
    records_b = {
        "train": [
            _record("s0", "train", object_name="red_potion", base_object="potion"),
            _record("s9", "train", object_name="blue_potion", base_object="potion", color="blue"),
        ],
        "val": [_record("s8", "val", object_name="green_potion", base_object="potion", color="green")],
        "test": [_record("s7", "test", object_name="pink_potion", base_object="potion", color="pink")],
    }
    b = _write_dataset(tmp_path, "packB_label_v2_semantic_v3", records_b)

    out = tmp_path / "merged"
    result = merge_datasets([a, b], out, split_policy="preserve")
    assert result.ok, result.errors
    assert result.prefixed_sprite_ids == 2  # both "s0" records get prefixed

    ids: list[str] = []
    for split in SPLITS:
        for line in (out / f"manifest_{split}.jsonl").read_text().splitlines():
            ids.append(json.loads(line)["sprite_id"])
    assert len(ids) == len(set(ids))  # globally unique
    assert "packA__s0" in ids
    assert "packB__s0" in ids
    assert "s1" in ids  # non-colliding ids untouched


def test_provenance_fields_added(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    merge_datasets([a, b], out, split_policy="preserve")
    record = json.loads((out / "manifest_val.jsonl").read_text().splitlines()[0])
    for field in ("source_dataset", "source_pack", "source_sprite_id", "source_split", "source_npz_row"):
        assert field in record
    assert isinstance(record["provenance"], dict)
    assert record["provenance"]["source_pack"] in {"packA", "packB"}


def test_semantic_and_label_v2_preserved(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    merge_datasets([a, b], out, split_policy="preserve")
    by_source: dict[str, dict] = {}
    for split in SPLITS:
        for line in (out / f"manifest_{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            by_source[record["source_sprite_id"]] = record
    ruby = by_source["a_ruby"]
    assert ruby["semantic_v3"]["base_object"] == "gem"
    assert ruby["semantic_v3"]["object_name"] == "ruby_gem"
    assert ruby["label_v2"]["applied"] is True
    assert ruby["label_v2"]["bucket"] == "auto_filename_trusted"


def test_npz_arrays_concatenated_and_aligned(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    merge_datasets([a, b], out, split_policy="preserve")
    # train has 3 (A) + 2 (B) = 5 records
    with np.load(out / "train.npz", allow_pickle=False) as data:
        assert data["alpha"].shape == (5, 32, 32)
        assert data["palette"].shape == (5, 33, 3)
        npz_ids = [str(v) for v in data["sprite_id"]]
        alpha = np.asarray(data["alpha"])
    manifest_ids = [json.loads(line)["sprite_id"] for line in (out / "manifest_train.jsonl").read_text().splitlines()]
    assert npz_ids == manifest_ids
    assert alpha.sum() > 0


def test_merged_dataset_passes_dataset_qa(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    merge_datasets([a, b], out, split_policy="preserve")
    result = qa_dataset(out, require_semantic_v3=True)
    assert result.ok, result.errors


def test_merged_training_manifest_builds_and_passes_qa(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    merge_datasets([a, b], out, split_policy="preserve")
    manifest_result = build_training_manifest(out, variants_per_sprite=4, caption_policy="mixed", seed=1)
    manifest_path = out / "training_manifest.jsonl"
    write_training_manifest(manifest_path, manifest_result.rows)
    qa = qa_training_manifest(out, manifest_path)
    assert not qa.errors, qa.errors
    assert qa.unique_sprites == 9


def test_merge_report_lists_source_contributions(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    out = tmp_path / "merged"
    result = merge_datasets([a, b], out, split_policy="preserve")
    report = json.loads((out / "merge_report.json").read_text())
    assert report["source_contributions"]["packA_label_v2_semantic_v3"] == 5
    assert report["source_contributions"]["packB_label_v2_semantic_v3"] == 4
    assert result.semantic_v3_coverage == 1.0
    assert "Source contributions" in (out / "merge_report.md").read_text()


def test_duplicate_source_record_leakage_is_detected(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    out = tmp_path / "merged"
    # Passing the same dataset twice yields identical (pack, id) pairs that
    # cannot be de-collided by prefixing -> duplicate ids must be reported.
    result = merge_datasets([a, a], out, split_policy="preserve")
    assert not result.ok
    assert any("duplicate sprite_id after merge" in error for error in result.errors)
    # The broken dataset must not be written out.
    assert not (out / "train.npz").is_file()


def test_missing_source_dataset_fails_clearly(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    with pytest.raises(MergeError, match="does not exist"):
        merge_datasets([a, tmp_path / "nope_label_v2_semantic_v3"], tmp_path / "merged")


def test_unknown_split_policy_fails_clearly(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    with pytest.raises(MergeError, match="split policy"):
        merge_datasets([a, b], tmp_path / "merged", split_policy="nonsense")


def test_source_datasets_not_mutated(tmp_path: Path) -> None:
    a = _dataset_a(tmp_path)
    b = _dataset_b(tmp_path)
    before = {p.name: p.read_bytes() for p in a.iterdir()}
    merge_datasets([a, b], tmp_path / "merged", split_policy="preserve")
    after = {p.name: p.read_bytes() for p in a.iterdir()}
    assert before == after
