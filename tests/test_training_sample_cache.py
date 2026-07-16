"""Regression tests for the dataset sample cache and shared npz cache.

The sample cache must be *compute-identical*: a cached sample has to equal a
freshly-built one, so training that relies on the cache stays numerically the
same as before the cache existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.data import SpriteTrainingDataset

_TENSOR_KEYS = (
    "rgba",
    "rgb",
    "alpha",
    "index_map",
    "role_map",
    "palette",
    "palette_u8",
    "palette_mask",
    "category_id",
    "caption_tokens",
    "semantic_tokens",
)
_SCALAR_KEYS = ("caption", "sprite_id", "split", "npz_file", "npz_row")


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=7)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def _assert_samples_equal(a: dict, b: dict) -> None:
    for key in _TENSOR_KEYS:
        assert torch.equal(a[key], b[key]), f"tensor mismatch for {key}"
    for key in _SCALAR_KEYS:
        assert a[key] == b[key], f"scalar mismatch for {key}"


def test_sample_cache_matches_uncached(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    cached = SpriteTrainingDataset(dataset_dir, manifest, split="train", cache_samples=True)
    uncached = SpriteTrainingDataset(dataset_dir, manifest, split="train", cache_samples=False)
    assert len(cached) == len(uncached)
    for index in range(len(cached)):
        _assert_samples_equal(cached[index], uncached[index])


def test_cache_reuses_object_and_uncached_rebuilds(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    cached = SpriteTrainingDataset(dataset_dir, manifest, split="train", cache_samples=True)
    assert cached[0] is cached[0]  # memoized object is reused

    uncached = SpriteTrainingDataset(dataset_dir, manifest, split="train", cache_samples=False)
    first, second = uncached[0], uncached[0]
    assert first is not second  # rebuilt each access
    _assert_samples_equal(first, second)  # ...but byte-for-byte equal


def test_shared_npz_cache_across_splits(tmp_path: Path) -> None:
    dataset_dir, manifest = _dataset_with_manifest(tmp_path)
    shared: dict = {}
    train = SpriteTrainingDataset(dataset_dir, manifest, split="train", npz_cache=shared)
    val = SpriteTrainingDataset(dataset_dir, manifest, split="val", npz_cache=shared)
    _ = train[0]
    _ = val[0]
    assert train._npz_cache is shared and val._npz_cache is shared
    assert "train.npz" in shared and "val.npz" in shared
