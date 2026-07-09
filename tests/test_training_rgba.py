from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.rgba import npz_row_to_rgba


def _full(shape: tuple[int, int], value: int | float) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_npz_row_to_rgba_converts_index_palette_alpha() -> None:
    index_map = _full((32, 32), 1)
    alpha = _full((32, 32), 1)
    palette = np.array([[0, 0, 0], [255, 128, 0]], dtype=np.uint8)
    rgba = npz_row_to_rgba(index_map=index_map, alpha=alpha, palette=palette)
    assert rgba.shape == (4, 32, 32)
    assert rgba.dtype == np.float32
    assert float(rgba.min()) >= 0.0
    assert float(rgba.max()) <= 1.0
    assert np.allclose(rgba[0], 1.0)
    assert np.allclose(rgba[1], 128.0 / 255.0)
    assert np.allclose(rgba[2], 0.0)
    assert np.array_equal(rgba[3], alpha.astype(np.float32))


def test_npz_row_to_rgba_keeps_normalized_palette_values_and_zeroes_transparent_rgb() -> None:
    index_map = _full((32, 32), 1)
    alpha = _full((32, 32), 1)
    alpha[0, 0] = 0
    palette = np.array([[0.0, 0.0, 0.0, 0.0], [0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    rgba = npz_row_to_rgba(index_map=index_map, alpha=alpha, palette=palette)
    assert np.allclose(rgba[:3, 1, 1], [0.25, 0.5, 0.75])
    assert np.allclose(rgba[:3, 0, 0], [0.0, 0.0, 0.0])
    assert rgba[3, 0, 0] == 0.0


def test_npz_row_to_rgba_rejects_invalid_palette_index() -> None:
    index_map = _full((32, 32), 2)
    alpha = _full((32, 32), 1)
    palette = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="invalid palette index"):
        npz_row_to_rgba(index_map=index_map, alpha=alpha, palette=palette)


def test_npz_row_to_rgba_honors_palette_mask() -> None:
    index_map = _full((32, 32), 1)
    alpha = _full((32, 32), 1)
    palette = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="disabled by palette_mask"):
        npz_row_to_rgba(index_map=index_map, alpha=alpha, palette=palette, palette_mask=np.array([True, False]))


def test_training_dataset_sample_includes_rgba_rgb_alpha(tmp_path: Path) -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    from spritelab.training.data import SpriteTrainingDataset

    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=7)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    sample = SpriteTrainingDataset(dataset, manifest, split="train")[0]
    assert sample["rgba"].shape == (4, 32, 32)
    assert sample["rgb"].shape == (3, 32, 32)
    assert sample["alpha"].shape == (1, 32, 32)


def test_training_dataset_rgba_collates(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch

    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=8)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    ds = SpriteTrainingDataset(dataset, manifest, split="train", max_records=2)
    batch = collate_sprite_batch([ds[0], ds[1]])
    assert isinstance(batch["rgba"], torch.Tensor)
    assert batch["rgba"].shape == (2, 4, 32, 32)
    assert batch["rgb"].shape == (2, 3, 32, 32)
