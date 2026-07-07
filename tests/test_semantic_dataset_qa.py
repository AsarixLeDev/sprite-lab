from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.dataset_maker.qa import qa_dataset


def _bundle_arrays(sprite_ids: list[str]) -> dict[str, np.ndarray]:
    count = len(sprite_ids)
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, 33, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, 33), dtype=bool)
    for row in range(count):
        alpha[row, 10:14, 10:14] = 1
        index_map[row, 10:14, 10:14] = 1
        palette[row, 1] = [120, 50, 80]
        palette_mask[row, 0] = True
        palette_mask[row, 1] = True
    return {
        "alpha": alpha,
        "index_map": index_map,
        "role_map": role_map,
        "palette": palette,
        "palette_mask": palette_mask,
        "category_id": np.zeros((count,), dtype=np.int64),
        "sprite_id": np.array(sprite_ids, dtype=np.str_),
    }


def _semantic(**overrides) -> dict:
    semantic = {
        "schema_version": "semantic_v3.0",
        "category": "item_icon",
        "object_name": "red_potion",
        "base_object": "potion",
        "open_name": "red potion",
        "attributes": {
            "colors": ["red"],
            "materials": ["glass", "liquid"],
            "shapes": [],
            "effects": [],
            "state": [],
            "function": ["consumable"],
            "mood": ["fantasy"],
            "style": ["32x32", "pixel_art", "rpg_icon"],
            "parts": ["cork"],
            "environment": [],
        },
        "aliases": ["potion"],
        "captions": ["red potion", "red potion made of glass", "32x32 pixel art red potion icon"],
        "prompt_phrases": ["32x32 pixel art red potion"],
        "negative_tags": ["photorealistic", "large_scene", "text", "watermark"],
        "source_evidence": {},
        "warnings": [],
    }
    semantic.update(overrides)
    return semantic


def _record(sprite_id: str, split: str, *, semantic: dict | None = None, **overrides) -> dict:
    record = {
        "sprite_id": sprite_id,
        "split": split,
        "category": "item_icon",
        "category_id": 1,
        "object_name": "red_potion",
        "tags": ["potion", "glass", "red"],
        "source_name": "test-source",
        "source_path": f"data/{sprite_id}.png",
        "license": "cc0",
        "label_v2": {"applied": True, "bucket": "auto_test", "flags": ["auto_test"]},
    }
    if semantic is not None:
        record["semantic_v3"] = semantic
    record.update(overrides)
    return record


def _write_dataset(tmp_path: Path, records_by_split: dict[str, list[dict]]) -> Path:
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        records = records_by_split.get(split, [])
        (dataset_dir / f"manifest_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in records) + ("\n" if records else ""),
            encoding="utf-8",
        )
        ids = [str(r["sprite_id"]) for r in records]
        np.savez_compressed(dataset_dir / f"{split}.npz", **_bundle_arrays(ids))
    (dataset_dir / "dataset_config.json").write_text(
        json.dumps({"dataset_name": "ds", "max_palette_slots": 32}), encoding="utf-8"
    )
    (dataset_dir / "vocab.json").write_text(
        json.dumps({"category_to_id": {"unknown": 0, "item_icon": 1}}), encoding="utf-8"
    )
    return dataset_dir


def _semantic_dataset(tmp_path: Path, *, train_overrides: dict | None = None) -> Path:
    train = [_record(f"s{i}", "train", semantic=_semantic()) for i in range(8)]
    if train_overrides is not None:
        train[0]["semantic_v3"] = {**_semantic(), **train_overrides}
    records = {
        "train": train,
        "val": [_record("v0", "val", semantic=_semantic())],
        "test": [_record("t0", "test", semantic=_semantic())],
    }
    return _write_dataset(tmp_path, records)


def test_qa_passes_on_valid_semantic_dataset(tmp_path: Path) -> None:
    result = qa_dataset(_semantic_dataset(tmp_path), require_semantic_v3=True)
    assert result.ok, result.errors
    assert result.semantic_v3_checks["is_semantic_v3_dataset"] is True
    assert result.semantic_v3_checks["records_with_semantic_v3"] == 10
    assert result.semantic_v3_checks["missing_semantic_v3"] == []


def test_dataset_without_semantic_v3_passes_unless_required(tmp_path: Path) -> None:
    records = {
        "train": [_record(f"s{i}", "train") for i in range(8)],
        "val": [_record("v0", "val")],
        "test": [_record("t0", "test")],
    }
    dataset_dir = _write_dataset(tmp_path, records)

    result = qa_dataset(dataset_dir)
    assert result.ok, result.errors
    assert result.semantic_v3_checks["is_semantic_v3_dataset"] is False

    required = qa_dataset(dataset_dir, require_semantic_v3=True)
    assert not required.ok
    assert any("missing required semantic_v3" in error for error in required.errors)


def test_missing_base_object_is_error_when_semantic_required(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(tmp_path, train_overrides={"base_object": ""})
    result = qa_dataset(dataset_dir, require_semantic_v3=True)
    assert not result.ok
    assert "s0" in result.semantic_v3_checks["missing_base_object"]
    assert any("semantic_v3.base_object missing" in error for error in result.errors)


def test_missing_captions_is_error(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(tmp_path, train_overrides={"captions": []})
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert "s0" in result.semantic_v3_checks["missing_captions"]


def test_non_string_caption_is_error(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(tmp_path, train_overrides={"captions": ["red potion", 42, "potion"]})
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert any("caption invalid" in error for error in result.errors)


def test_overlong_caption_is_error(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(tmp_path, train_overrides={"captions": ["red potion", "x" * 400, "potion"]})
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert any("longer than" in error for error in result.errors)


def test_forbidden_caption_content_is_error(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(
        tmp_path,
        train_overrides={"captions": ["red potion", "photorealistic red potion", "potion"]},
    )
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert any("forbidden content" in error for error in result.errors)


def test_category_mismatch_is_error(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(tmp_path, train_overrides={"category": "weapon"})
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert any("semantic_v3.category does not match" in error for error in result.errors)


def test_object_name_mismatch_is_error(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(tmp_path, train_overrides={"object_name": "blue_potion"})
    result = qa_dataset(dataset_dir)
    assert not result.ok
    assert any("semantic_v3.object_name does not match" in error for error in result.errors)


def test_few_captions_and_missing_negative_tags_are_warnings(tmp_path: Path) -> None:
    dataset_dir = _semantic_dataset(
        tmp_path,
        train_overrides={"captions": ["red potion"], "negative_tags": []},
    )
    result = qa_dataset(dataset_dir)
    assert result.ok, result.errors
    assert any("fewer than 3 captions" in warning for warning in result.warnings)
    assert any("no negative_tags" in warning for warning in result.warnings)


def test_colorless_gem_records_produce_warning(tmp_path: Path) -> None:
    semantic = _semantic(
        object_name="gem",
        base_object="gem",
        open_name="gem",
        attributes={**_semantic()["attributes"], "colors": []},
    )
    records = {
        "train": [_record(f"s{i}", "train", semantic=semantic, object_name="gem") for i in range(8)],
        "val": [_record("v0", "val", semantic=semantic, object_name="gem")],
        "test": [_record("t0", "test", semantic=semantic, object_name="gem")],
    }
    dataset_dir = _write_dataset(tmp_path, records)
    result = qa_dataset(dataset_dir)
    assert result.ok, result.errors
    assert any("no color information" in warning for warning in result.warnings)


def test_semantic_checks_present_in_json_and_markdown_reports(tmp_path: Path) -> None:
    result = qa_dataset(_semantic_dataset(tmp_path), require_semantic_v3=True)
    payload = result.to_json_dict()
    assert payload["semantic_v3_checks"]["records_with_semantic_v3"] == 10
    markdown = result.to_markdown()
    assert "Semantic-v3 Checks" in markdown
    assert "records with semantic_v3: 10" in markdown
