"""Tests for spritelab.ml.masking."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _ml_testdata import write_synthetic_dataset

from spritelab.ml.dataset import SpriteBundleDataset
from spritelab.ml.masking import INDEX_MASK, FixedOpaqueMask, FullOpaqueMask, RandomOpaqueMask


def _sample(tmp_path, transform=None):
    dataset = SpriteBundleDataset(write_synthetic_dataset(tmp_path), "train", transform=transform)
    return dataset[0]


def test_random_mask_masks_only_opaque(tmp_path):
    sample = _sample(tmp_path, RandomOpaqueMask(seed=7))
    masked = sample["input_index_map"] == INDEX_MASK
    assert bool(masked.any())
    assert bool((sample["alpha"][masked] == 1).all())


def test_transparent_pixels_remain_zero(tmp_path):
    sample = _sample(tmp_path, RandomOpaqueMask(seed=7))
    transparent = sample["alpha"] == 0
    assert bool((sample["input_index_map"][transparent] == 0).all())


def test_loss_mask_only_masked_opaque(tmp_path):
    sample = _sample(tmp_path, FixedOpaqueMask(mask_fraction=0.5, seed=3))
    loss_mask = sample["loss_mask"]
    assert bool((sample["alpha"][loss_mask] == 1).all())
    assert bool((sample["input_index_map"][loss_mask] == INDEX_MASK).all())
    unmasked = ~loss_mask
    assert bool(
        (sample["input_index_map"][unmasked] == sample["index_map"][unmasked]).all()
    )


def test_fixed_mask_deterministic(tmp_path):
    a = _sample(tmp_path, FixedOpaqueMask(mask_fraction=0.5, seed=42))
    b = _sample(tmp_path, FixedOpaqueMask(mask_fraction=0.5, seed=42))
    assert torch.equal(a["loss_mask"], b["loss_mask"])
    assert torch.equal(a["input_index_map"], b["input_index_map"])


def test_full_mask_masks_all_opaque(tmp_path):
    sample = _sample(tmp_path, FullOpaqueMask())
    opaque = sample["alpha"] == 1
    assert bool((sample["input_index_map"][opaque] == INDEX_MASK).all())
    assert torch.equal(sample["loss_mask"], opaque)
    assert sample["mask_fraction"] == 1.0


def test_empty_alpha_does_not_crash():
    sample = {
        "alpha": torch.zeros(32, 32, dtype=torch.long),
        "index_map": torch.zeros(32, 32, dtype=torch.long),
        "sample_index": 0,
    }
    for transform in (RandomOpaqueMask(seed=1), FixedOpaqueMask(0.5, seed=1), FullOpaqueMask()):
        result = transform(dict(sample))
        assert not bool(result["loss_mask"].any())
        assert result["mask_fraction"] == 0.0


def test_target_equals_original(tmp_path):
    sample = _sample(tmp_path, FixedOpaqueMask(mask_fraction=0.7, seed=9))
    assert torch.equal(sample["target_index_map"], sample["index_map"])
