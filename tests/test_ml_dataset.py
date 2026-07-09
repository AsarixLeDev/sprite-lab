"""Tests for spritelab.ml.dataset."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _ml_testdata import PALETTE_ROWS, make_synthetic_arrays, write_synthetic_dataset
from spritelab.ml.dataset import (
    SpriteBundleDataset,
    assert_valid_npz_dataset_arrays,
    validate_npz_dataset_arrays,
)


def test_valid_dataset_loads(tmp_path):
    dataset_dir = write_synthetic_dataset(tmp_path)
    dataset = SpriteBundleDataset(dataset_dir, "train")
    assert len(dataset) == 3


def test_all_split_paths_resolve(tmp_path):
    dataset_dir = write_synthetic_dataset(tmp_path)
    for split in ("train", "val", "test"):
        assert len(SpriteBundleDataset(dataset_dir, split)) == 3


def test_dataset_length(tmp_path):
    dataset_dir = write_synthetic_dataset(tmp_path, count=5)
    assert len(SpriteBundleDataset(dataset_dir, "train")) == 5


def test_item_has_required_keys(tmp_path):
    dataset = SpriteBundleDataset(write_synthetic_dataset(tmp_path), "train")
    sample = dataset[0]
    for key in (
        "alpha",
        "index_map",
        "role_map",
        "palette",
        "palette_u8",
        "palette_mask",
        "category_id",
        "sprite_id",
        "sample_index",
        "metadata",
    ):
        assert key in sample


def test_tensor_shapes(tmp_path):
    sample = SpriteBundleDataset(write_synthetic_dataset(tmp_path), "train")[0]
    assert sample["alpha"].shape == (32, 32)
    assert sample["index_map"].shape == (32, 32)
    assert sample["role_map"].shape == (32, 32)
    assert sample["palette"].shape == (PALETTE_ROWS, 3)
    assert sample["palette_u8"].shape == (PALETTE_ROWS, 3)
    assert sample["palette_mask"].shape == (PALETTE_ROWS,)
    assert sample["category_id"].shape == ()
    assert sample["sprite_id"] == "sample_000"


def test_palette_normalized_float32(tmp_path):
    sample = SpriteBundleDataset(write_synthetic_dataset(tmp_path), "train")[0]
    assert sample["palette"].dtype == torch.float32
    assert float(sample["palette"].max()) <= 1.0
    assert float(sample["palette"].min()) >= 0.0


def test_palette_u8_preserves_values(tmp_path):
    sample = SpriteBundleDataset(write_synthetic_dataset(tmp_path), "train")[0]
    assert sample["palette_u8"].dtype == torch.uint8
    assert sample["palette_u8"][1].tolist() == [200, 40, 40]


def test_missing_split_raises(tmp_path):
    dataset_dir = write_synthetic_dataset(tmp_path)
    (dataset_dir / "val.npz").unlink()
    with pytest.raises(FileNotFoundError):
        SpriteBundleDataset(dataset_dir, "val")


def test_missing_required_key_raises():
    arrays = make_synthetic_arrays()
    del arrays["role_map"]
    with pytest.raises(ValueError, match="role_map"):
        assert_valid_npz_dataset_arrays(arrays)


def test_invalid_alpha_shape_raises():
    arrays = make_synthetic_arrays()
    arrays["alpha"] = arrays["alpha"][:, :16, :]
    with pytest.raises(ValueError, match="alpha"):
        assert_valid_npz_dataset_arrays(arrays)


def test_invalid_index_map_shape_raises():
    arrays = make_synthetic_arrays()
    arrays["index_map"] = arrays["index_map"].reshape(-1, 16, 64)
    with pytest.raises(ValueError, match="index_map"):
        assert_valid_npz_dataset_arrays(arrays)


def test_transparent_pixel_with_nonzero_index_raises():
    arrays = make_synthetic_arrays()
    arrays["index_map"][0, 0, 0] = 1  # alpha is 0 there
    with pytest.raises(ValueError, match="transparent"):
        assert_valid_npz_dataset_arrays(arrays)


def test_opaque_pixel_with_zero_index_raises():
    arrays = make_synthetic_arrays()
    arrays["index_map"][0, 12, 12] = 0  # alpha is 1 there
    with pytest.raises(ValueError, match="opaque"):
        assert_valid_npz_dataset_arrays(arrays)


def test_index_into_masked_palette_row_raises():
    arrays = make_synthetic_arrays()
    arrays["index_map"][0, 12, 12] = 5  # palette_mask row 5 is False
    errors = validate_npz_dataset_arrays(arrays)
    assert any("palette_mask" in error for error in errors)


def test_missing_manifest_vocab_config_tolerated(tmp_path):
    dataset_dir = write_synthetic_dataset(tmp_path)
    dataset = SpriteBundleDataset(dataset_dir, "train", load_manifests=True)
    assert dataset.manifest == []
    assert dataset.vocab == {}
    assert dataset.dataset_config == {}
