"""PyTorch data loading for semantic training manifests."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:  # torch is an optional project dependency.
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]

from spritelab.training.palette_swap import PaletteSwapConfig, apply_palette_swap
from spritelab.training.tokenization import SpriteTextTokenizer
from spritelab.training.rgba import npz_row_to_rgba
from spritelab.training.structured_conditioning import (
    STRUCTURED_BATCH_KEYS,
    StructuredConditioningVocab,
    encode_structured_conditioning,
)

SPRITE_SIZE = 32
REQUIRED_NPZ_KEYS = (
    "alpha",
    "index_map",
    "role_map",
    "palette",
    "palette_mask",
    "category_id",
    "sprite_id",
)

_DatasetBase = torch.utils.data.Dataset if torch is not None else object


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab training data loading.")
    return torch


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        records.append(value)
    return records


class SpriteTrainingDataset(_DatasetBase):
    """Dataset over ``training_manifest.jsonl`` rows and split ``.npz`` arrays."""

    def __init__(
        self,
        dataset_dir: Path,
        training_manifest: Path,
        split: str | None = None,
        max_records: int | None = None,
        *,
        tokenizer: SpriteTextTokenizer | None = None,
        caption_max_length: int = 32,
        semantic_max_length: int = 48,
        caption_policy_filter: str | None = None,
        sprite_ids: Sequence[str] | None = None,
        structured_vocab: StructuredConditioningVocab | None = None,
        cache_samples: bool = True,
        npz_cache: dict[str, dict[str, np.ndarray]] | None = None,
        palette_swap: PaletteSwapConfig | None = None,
    ) -> None:
        _require_torch()
        self.dataset_dir = Path(dataset_dir)
        self.training_manifest = Path(training_manifest)
        self.split = split
        self.caption_max_length = int(caption_max_length)
        self.semantic_max_length = int(semantic_max_length)
        self.caption_policy_filter = caption_policy_filter
        self.sprite_ids = None if sprite_ids is None else tuple(str(sprite_id) for sprite_id in sprite_ids)
        self.structured_vocab = structured_vocab
        self.cache_samples = bool(cache_samples)
        self.palette_swap = palette_swap if (palette_swap is not None and palette_swap.active()) else None

        all_records = read_jsonl(self.training_manifest)
        self.all_records = list(all_records)
        sprite_id_set = None if self.sprite_ids is None else set(self.sprite_ids)
        records = [
            record
            for record in all_records
            if (split is None or record.get("split") == split)
            and _matches_caption_policy(record, caption_policy_filter)
            and (sprite_id_set is None or str(record.get("sprite_id", "")) in sprite_id_set)
        ]
        if max_records is not None:
            records = records[: max(0, int(max_records))]
        self.records = records
        self.tokenizer = tokenizer or SpriteTextTokenizer.build_from_records(
            all_records,
            max_length=self.caption_max_length,
        )
        # ``npz_cache`` may be shared across split datasets so each ``.npz`` file
        # is decompressed at most once per run. ``_sample_cache`` memoizes the
        # fully-built per-sample dict: ``__getitem__`` has no randomness, so a
        # cached sample is byte-for-byte identical to a freshly built one.
        self._npz_cache: dict[str, dict[str, np.ndarray]] = npz_cache if npz_cache is not None else {}
        self._sample_cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_samples:
            cached = self._sample_cache.get(index)
            if cached is not None:
                return cached
        th = _require_torch()
        record = self.records[index]
        npz_file = str(record.get("npz_file") or f"{record.get('split', '')}.npz")
        npz_row = int(record.get("npz_row", -1))
        arrays = self._load_npz(npz_file)
        row_count = int(arrays["alpha"].shape[0])
        if npz_row < 0 or npz_row >= row_count:
            sprite_id = record.get("sprite_id", f"manifest row {index}")
            raise IndexError(f"{sprite_id}: npz_row {npz_row} out of range for {npz_file} with {row_count} rows")

        sprite_id = str(record.get("sprite_id", ""))
        npz_sprite_id = str(np.asarray(arrays["sprite_id"])[npz_row])
        if sprite_id and npz_sprite_id != sprite_id:
            raise ValueError(
                f"{sprite_id}: npz_row {npz_row} in {npz_file} holds sprite_id {npz_sprite_id!r}"
            )

        alpha_np = np.asarray(arrays["alpha"][npz_row], dtype=np.float32)
        index_np = np.asarray(arrays["index_map"][npz_row], dtype=np.int64)
        role_np = np.asarray(arrays["role_map"][npz_row], dtype=np.int64)
        palette_raw = np.asarray(arrays["palette"][npz_row])
        palette = _palette_rgb_float(palette_raw)
        palette_u8 = np.rint(np.clip(palette, 0.0, 1.0) * 255.0).astype(np.uint8)
        palette_mask_np = np.asarray(arrays["palette_mask"][npz_row], dtype=bool)
        category_id = int(np.asarray(arrays["category_id"])[npz_row])
        caption = str(record.get("caption", ""))
        active_record: Mapping[str, Any] = record

        palette_swap_meta: dict[str, Any] | None = None
        if self.palette_swap is not None:
            swap = apply_palette_swap(
                index_map=index_np,
                alpha=alpha_np,
                role_map=role_np,
                palette_rgb=palette,
                palette_mask=palette_mask_np,
                record=record,
                caption=caption,
                sprite_id=sprite_id,
                config=self.palette_swap,
            )
            palette_swap_meta = swap.metadata()
            if swap.applied:
                palette = np.asarray(swap.palette_rgb, dtype=np.float32)
                palette_u8 = np.rint(np.clip(palette, 0.0, 1.0) * 255.0).astype(np.uint8)
                caption = swap.caption
                active_record = swap.record

        rgba_np = npz_row_to_rgba(
            index_map=index_np,
            alpha=alpha_np,
            palette=palette,
            palette_mask=palette_mask_np,
        )

        sample = {
            "rgba": th.as_tensor(rgba_np, dtype=th.float32),
            "rgb": th.as_tensor(rgba_np[:3], dtype=th.float32),
            "alpha": th.as_tensor(rgba_np[3:4], dtype=th.float32),
            "index_map": th.as_tensor(index_np, dtype=th.long),
            "role_map": th.as_tensor(role_np, dtype=th.long),
            "palette": th.as_tensor(palette, dtype=th.float32),
            "palette_u8": th.as_tensor(palette_u8, dtype=th.uint8),
            "palette_mask": th.as_tensor(palette_mask_np, dtype=th.bool),
            "category_id": th.as_tensor(category_id, dtype=th.long),
            "caption": caption,
            "caption_tokens": th.as_tensor(
                self.tokenizer.encode(caption, max_length=self.caption_max_length),
                dtype=th.long,
            ),
            "semantic_tokens": th.as_tensor(
                self.tokenizer.encode_record_semantics(active_record, max_length=self.semantic_max_length),
                dtype=th.long,
            ),
            "sprite_id": sprite_id,
            "split": str(record.get("split", "")),
            "npz_file": npz_file,
            "npz_row": npz_row,
            "manifest_record": dict(record),
        }
        if palette_swap_meta is not None:
            sample["palette_swap"] = palette_swap_meta
        if self.structured_vocab is not None:
            structured = encode_structured_conditioning(active_record, self.structured_vocab)
            for key in STRUCTURED_BATCH_KEYS:
                field = key.removeprefix("structured_")
                dtype = th.float32 if key.endswith("_multi_hot") else th.long
                sample[key] = th.as_tensor(structured[field], dtype=dtype)
            sample["structured_present"] = th.as_tensor(bool(structured["structured_present"]), dtype=th.bool)
        if self.cache_samples:
            self._sample_cache[index] = sample
        return sample

    def split_counts(self) -> dict[str, int]:
        return dict(Counter(str(record.get("split", "")) for record in self.all_records))

    def _load_npz(self, npz_file: str) -> dict[str, np.ndarray]:
        cached = self._npz_cache.get(npz_file)
        if cached is not None:
            return cached
        path = self.dataset_dir / npz_file
        if not path.is_file():
            raise FileNotFoundError(f"manifest references missing npz file: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
            if missing:
                raise ValueError(f"{path}: missing required arrays: {', '.join(missing)}")
            arrays = {key: data[key] for key in data.files}
        self._validate_npz_shapes(path, arrays)
        self._npz_cache[npz_file] = arrays
        return arrays

    @staticmethod
    def _validate_npz_shapes(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        alpha = np.asarray(arrays["alpha"])
        index_map = np.asarray(arrays["index_map"])
        role_map = np.asarray(arrays["role_map"])
        palette = np.asarray(arrays["palette"])
        palette_mask = np.asarray(arrays["palette_mask"])
        if alpha.ndim != 3 or alpha.shape[1:] != (SPRITE_SIZE, SPRITE_SIZE):
            raise ValueError(f"{path}: alpha must have shape [N, 32, 32], got {alpha.shape}")
        if index_map.shape != alpha.shape:
            raise ValueError(f"{path}: index_map shape {index_map.shape} does not match alpha {alpha.shape}")
        if role_map.shape != alpha.shape:
            raise ValueError(f"{path}: role_map shape {role_map.shape} does not match alpha {alpha.shape}")
        if palette.ndim != 3 or palette.shape[0] != alpha.shape[0] or palette.shape[2] not in (3, 4):
            raise ValueError(f"{path}: palette must have shape [N, K, 3] or [N, K, 4], got {palette.shape}")
        if palette_mask.shape != palette.shape[:2]:
            raise ValueError(f"{path}: palette_mask shape {palette_mask.shape} does not match palette {palette.shape[:2]}")


def collate_sprite_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    th = _require_torch()
    if not samples:
        raise ValueError("cannot collate an empty sprite batch")
    tensor_keys = (
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
    batch: dict[str, Any] = {}
    for key in tensor_keys:
        batch[key] = th.stack([sample[key] for sample in samples])
    for key in (*STRUCTURED_BATCH_KEYS, "structured_present"):
        if key in samples[0]:
            batch[key] = th.stack([sample[key] for sample in samples])
    for key in ("caption", "sprite_id", "split", "npz_file"):
        batch[key] = [sample[key] for sample in samples]
    if "palette_swap" in samples[0]:
        batch["palette_swap"] = [dict(sample.get("palette_swap") or {}) for sample in samples]
    batch["npz_row"] = [int(sample["npz_row"]) for sample in samples]
    batch["manifest_record"] = [dict(sample["manifest_record"]) for sample in samples]
    return batch


def _matches_caption_policy(record: Mapping[str, Any], caption_policy_filter: str | None) -> bool:
    if not caption_policy_filter:
        return True
    audit = record.get("audit") if isinstance(record.get("audit"), Mapping) else {}
    return str(audit.get("caption_policy", "")) == str(caption_policy_filter)


def _palette_rgb_float(palette: np.ndarray) -> np.ndarray:
    value = np.asarray(palette)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"palette must have shape [K, 3] or [K, 4], got {value.shape}")
    rgb = value[:, :3].astype(np.float32, copy=False)
    if value.dtype.kind in "ui" or (rgb.size and float(np.nanmax(rgb)) > 1.0):
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def describe_array(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    result: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.size and value.dtype.kind in "biuf?":
        result["min"] = float(value.min()) if value.dtype.kind == "f" else int(value.min())
        result["max"] = float(value.max()) if value.dtype.kind == "f" else int(value.max())
    return result
