from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=7)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_loader_returns_expected_tensor_shapes(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    dataset = SpriteTrainingDataset(dataset_dir, manifest, split="train")
    sample = dataset[0]
    assert sample["alpha"].shape == (1, 32, 32)
    assert sample["index_map"].shape == (32, 32)
    assert sample["role_map"].shape == (32, 32)
    assert sample["palette"].shape == (33, 3)
    assert sample["palette_mask"].shape == (33,)
    assert sample["caption_tokens"].dim() == 1
    assert sample["semantic_tokens"].dim() == 1


def test_split_filtering_works(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    train = SpriteTrainingDataset(dataset_dir, manifest, split="train")
    val = SpriteTrainingDataset(dataset_dir, manifest, split="val")
    assert len(train) == 8
    assert len(val) == 2
    assert all(sample["split"] == "val" for sample in [val[i] for i in range(len(val))])


def test_npz_row_lookup_matches_sprite_id(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    dataset = SpriteTrainingDataset(dataset_dir, manifest)
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["sprite_id"] == sample["manifest_record"]["sprite_id"]
        assert sample["npz_file"] == f"{sample['split']}.npz"


def test_missing_npz_file_raises_useful_error(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    rows = _read_rows(manifest)
    rows[0]["npz_file"] = "missing.npz"
    broken = dataset_dir / "broken_missing.jsonl"
    write_training_manifest(broken, rows)
    dataset = SpriteTrainingDataset(dataset_dir, broken)
    with pytest.raises(FileNotFoundError, match="missing npz file"):
        dataset[0]


def test_out_of_range_npz_row_raises_useful_error(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    rows = _read_rows(manifest)
    rows[0]["npz_row"] = 999
    broken = dataset_dir / "broken_row.jsonl"
    write_training_manifest(broken, rows)
    dataset = SpriteTrainingDataset(dataset_dir, broken)
    with pytest.raises(IndexError, match="npz_row 999 out of range"):
        dataset[0]


def test_collate_keeps_metadata_and_stacks_tensors(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    dataset = SpriteTrainingDataset(dataset_dir, manifest, split="train", max_records=2)
    batch = collate_sprite_batch([dataset[0], dataset[1]])
    assert batch["alpha"].shape == (2, 1, 32, 32)
    assert batch["index_map"].shape == (2, 32, 32)
    assert len(batch["caption"]) == 2
    assert len(batch["manifest_record"]) == 2
