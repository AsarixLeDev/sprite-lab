"""PyTorch dataset over Dataset Maker ``.npz`` exports."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

SPRITE_SHAPE = (32, 32)
VALID_SPLITS = ("train", "val", "test")
REQUIRED_NPZ_KEYS = (
    "alpha",
    "index_map",
    "role_map",
    "palette",
    "palette_mask",
    "category_id",
    "sprite_id",
)


def load_npz_split(path: str | Path) -> dict[str, np.ndarray]:
    """Load a split ``.npz`` file into plain arrays."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_jsonl_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest; each line becomes one dict."""

    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_vocab(path: str | Path) -> dict[str, Any]:
    """Load ``vocab.json``."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load ``dataset_config.json``."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_npz_dataset_arrays(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Check the Dataset Maker array contract and return all found errors."""

    errors: list[str] = []
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in arrays]
    if missing:
        errors.append(f"missing required npz keys: {', '.join(missing)}")
        return errors

    alpha = np.asarray(arrays["alpha"])
    index_map = np.asarray(arrays["index_map"])
    role_map = np.asarray(arrays["role_map"])
    palette = np.asarray(arrays["palette"])
    palette_mask = np.asarray(arrays["palette_mask"])
    category_id = np.asarray(arrays["category_id"])
    sprite_id = np.asarray(arrays["sprite_id"])

    counts = {
        "alpha": alpha.shape[0] if alpha.ndim else -1,
        "index_map": index_map.shape[0] if index_map.ndim else -1,
        "role_map": role_map.shape[0] if role_map.ndim else -1,
        "palette": palette.shape[0] if palette.ndim else -1,
        "palette_mask": palette_mask.shape[0] if palette_mask.ndim else -1,
        "category_id": category_id.shape[0] if category_id.ndim else -1,
        "sprite_id": sprite_id.shape[0] if sprite_id.ndim else -1,
    }
    if len(set(counts.values())) != 1:
        errors.append(f"arrays disagree on sample count N: {counts}")

    if alpha.ndim != 3 or alpha.shape[1:] != SPRITE_SHAPE:
        errors.append(f"alpha shape must be [N, 32, 32], got {alpha.shape}")
    if index_map.ndim != 3 or index_map.shape[1:] != SPRITE_SHAPE:
        errors.append(f"index_map shape must be [N, 32, 32], got {index_map.shape}")
    if role_map.ndim != 3 or role_map.shape[1:] != SPRITE_SHAPE:
        errors.append(f"role_map shape must be [N, 32, 32], got {role_map.shape}")
    if palette.ndim != 3 or palette.shape[2:] != (3,):
        errors.append(f"palette shape must be [N, K, 3], got {palette.shape}")
    if palette_mask.ndim != 2:
        errors.append(f"palette_mask shape must be [N, K], got {palette_mask.shape}")
    if category_id.ndim != 1:
        errors.append(f"category_id shape must be [N], got {category_id.shape}")
    if sprite_id.ndim != 1:
        errors.append(f"sprite_id shape must be [N], got {sprite_id.shape}")
    if palette.ndim == 3 and palette_mask.ndim == 2 and palette.shape[:2] != palette_mask.shape:
        errors.append(f"palette rows {palette.shape[:2]} and palette_mask {palette_mask.shape} disagree")
    if errors:
        return errors

    if not np.all(np.isin(alpha, [0, 1])):
        errors.append("alpha values must be only 0 or 1")
    if np.any(index_map < 0):
        errors.append("index_map values must be >= 0")
    if np.any(index_map >= palette.shape[1]):
        errors.append(f"index_map values must be < palette row count {palette.shape[1]}")
    if np.any((alpha == 0) & (index_map != 0)):
        errors.append("transparent pixels must have index_map == 0")
    if np.any((alpha == 1) & (index_map < 1)):
        errors.append("opaque pixels must have index_map >= 1")
    if palette.shape[0] and np.any(palette[:, 0] != 0):
        errors.append("palette row 0 must be [0, 0, 0]")
    if palette_mask.shape[0] and not np.all(palette_mask[:, 0]):
        errors.append("palette_mask row 0 must be True")

    if not np.any(index_map >= palette.shape[1]):
        clipped = np.clip(index_map, 0, palette.shape[1] - 1).astype(np.int64)
        used = np.take_along_axis(
            palette_mask,
            clipped.reshape(clipped.shape[0], -1),
            axis=1,
        )
        if not np.all(used):
            errors.append("index_map uses palette rows where palette_mask is False")

    return errors


def assert_valid_npz_dataset_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Raise ``ValueError`` listing every contract violation found."""

    errors = validate_npz_dataset_arrays(arrays)
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"invalid dataset arrays:\n  - {joined}")


class SpriteBundleDataset(torch.utils.data.Dataset):
    """Dataset over one exported split of a Dataset Maker dataset."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        *,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        validate: bool = True,
        load_manifests: bool = True,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform

        self.arrays = load_npz_split(self.dataset_root / f"{split}.npz")
        if validate:
            assert_valid_npz_dataset_arrays(self.arrays)

        self.manifest: list[dict[str, Any]] = []
        self.vocab: dict[str, Any] = {}
        self.dataset_config: dict[str, Any] = {}
        if load_manifests:
            manifest_path = self.dataset_root / f"manifest_{split}.jsonl"
            if manifest_path.exists():
                self.manifest = load_jsonl_manifest(manifest_path)
            vocab_path = self.dataset_root / "vocab.json"
            if vocab_path.exists():
                self.vocab = load_vocab(vocab_path)
            config_path = self.dataset_root / "dataset_config.json"
            if config_path.exists():
                self.dataset_config = load_dataset_config(config_path)

        self._manifest_by_id = {record.get("sprite_id"): record for record in self.manifest}

    def __len__(self) -> int:
        return int(self.arrays["alpha"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        sprite_id = str(self.arrays["sprite_id"][index])
        palette_u8 = np.asarray(self.arrays["palette"][index], dtype=np.uint8)
        sample: dict[str, Any] = {
            "alpha": torch.as_tensor(np.asarray(self.arrays["alpha"][index], dtype=np.int64)),
            "index_map": torch.as_tensor(np.asarray(self.arrays["index_map"][index], dtype=np.int64)),
            "role_map": torch.as_tensor(np.asarray(self.arrays["role_map"][index], dtype=np.int64)),
            "palette": torch.as_tensor(palette_u8.astype(np.float32) / 255.0),
            "palette_u8": torch.as_tensor(palette_u8),
            "palette_mask": torch.as_tensor(np.asarray(self.arrays["palette_mask"][index], dtype=bool)),
            "category_id": torch.as_tensor(int(self.arrays["category_id"][index]), dtype=torch.int64),
            "sprite_id": sprite_id,
            "sample_index": int(index),
            "metadata": dict(self._manifest_by_id.get(sprite_id, {})),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
