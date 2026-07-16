"""Tests for spritelab.ml.previews."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _ml_testdata import write_synthetic_dataset
from spritelab.ml.dataset import SpriteBundleDataset
from spritelab.ml.masking import FixedOpaqueMask
from spritelab.ml.previews import decode_index_map_to_rgba, save_prediction_grid


def _palette():
    palette = np.zeros((4, 3), dtype=np.uint8)
    palette[1] = (255, 0, 0)
    palette[2] = (0, 255, 0)
    palette[3] = (0, 0, 255)
    return palette


def test_decode_returns_32x32_rgba():
    index_map = np.zeros((32, 32), dtype=np.int64)
    image = decode_index_map_to_rgba(index_map, _palette())
    assert image.size == (32, 32)
    assert image.mode == "RGBA"


def test_transparent_index_alpha_zero():
    index_map = np.zeros((32, 32), dtype=np.int64)
    index_map[5, 5] = 1
    image = decode_index_map_to_rgba(index_map, _palette())
    pixels = np.asarray(image)
    assert pixels[0, 0, 3] == 0
    assert pixels[5, 5].tolist() == [255, 0, 0, 255]


def test_invalid_index_renders_error_pixel():
    index_map = np.zeros((32, 32), dtype=np.int64)
    index_map[3, 3] = 200
    image = decode_index_map_to_rgba(index_map, _palette())
    pixels = np.asarray(image)
    assert pixels[3, 3].tolist() == [255, 0, 255, 255]


def test_save_prediction_grid_writes_file(tmp_path):
    dataset = SpriteBundleDataset(
        write_synthetic_dataset(tmp_path),
        "train",
        transform=FixedOpaqueMask(mask_fraction=0.5, seed=1),
    )
    samples = [dataset[i] for i in range(len(dataset))]
    predictions = [sample["target_index_map"] for sample in samples]
    output_path = tmp_path / "grids" / "grid.png"
    save_prediction_grid(samples, predictions, output_path, scale=2)
    assert output_path.exists()
