"""Shared synthetic dataset helpers for spritelab.ml tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PALETTE_ROWS = 8  # max_palette_slots (7) + 1


def make_synthetic_arrays(count: int = 3) -> dict[str, np.ndarray]:
    """Return valid Dataset Maker arrays with an 8x8 opaque square per sample."""

    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, PALETTE_ROWS, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, PALETTE_ROWS), dtype=bool)
    category_id = np.ones((count,), dtype=np.int64)
    sprite_id = np.array([f"sample_{i:03d}" for i in range(count)], dtype=np.str_)

    for i in range(count):
        alpha[i, 12:20, 12:20] = 1
        square = index_map[i, 12:20, 12:20]
        square[:] = 1
        square[2:6, 2:6] = 2
        square[3:5, 3:5] = 3
        role_map[i][alpha[i] == 1] = 255
        palette[i, 1] = (200, 40, 40)
        palette[i, 2] = (40, 200, 40)
        palette[i, 3] = (40, 40, 200)
        palette_mask[i, :4] = True

    return {
        "alpha": alpha,
        "index_map": index_map,
        "role_map": role_map,
        "palette": palette,
        "palette_mask": palette_mask,
        "category_id": category_id,
        "sprite_id": sprite_id,
    }


def write_synthetic_dataset(root: Path, count: int = 3) -> Path:
    """Write train/val/test npz files (no manifests) under ``root / 'dataset'``."""

    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        np.savez_compressed(dataset_dir / f"{split}.npz", **make_synthetic_arrays(count))
    return dataset_dir
