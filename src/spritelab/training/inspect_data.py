"""Shared training data inspection helpers (migrated from train_baseline.py and data.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]

from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch, read_jsonl
from spritelab.training.tokenization import SpriteTextTokenizer


def describe_array(array: np.ndarray) -> dict[str, Any]:
    """Return a shape/dtype/value-range summary for a single numpy array."""
    value = np.asarray(array)
    result: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.size and value.dtype.kind in "biuf?":
        result["min"] = float(value.min()) if value.dtype.kind == "f" else int(value.min())
        result["max"] = float(value.max()) if value.dtype.kind == "f" else int(value.max())
    return result


def inspect_training_data(
    *,
    dataset_dir: str | Path,
    training_manifest: str | Path,
    split: str | None = None,
    batch_size: int = 4,
    max_records: int | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    training_manifest = Path(training_manifest)
    records = read_jsonl(training_manifest)
    tokenizer = SpriteTextTokenizer.build_from_records(records)
    filtered_records = [record for record in records if split is None or record.get("split") == split]
    if max_records is not None:
        filtered_records = filtered_records[: max(0, int(max_records))]
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[str(record.get("split", ""))] = split_counts.get(str(record.get("split", "")), 0) + 1

    npz_summary: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        path = dataset_dir / f"{split_name}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            npz_summary[split_name] = {key: describe_array(data[key]) for key in data.files}

    batch_shapes: dict[str, Any] = {}
    warnings: list[str] = []
    if torch is None:
        warnings.append("PyTorch is unavailable; skipped DataLoader batch tensor shape inspection.")
    elif filtered_records:
        th = torch
        dataset = SpriteTrainingDataset(
            dataset_dir,
            training_manifest,
            split=split,
            max_records=max_records,
            tokenizer=tokenizer,
        )
        loader = th.utils.data.DataLoader(
            dataset, batch_size=min(batch_size, len(dataset)), collate_fn=collate_sprite_batch
        )
        batch = next(iter(loader))
        for key, value in batch.items():
            if isinstance(value, th.Tensor):
                batch_shapes[key] = list(value.shape)

    summary = {
        "records": len(records),
        "loaded_records": len(filtered_records),
        "splits": split_counts,
        "npz": npz_summary,
        "caption_examples": [str(record.get("caption", "")) for record in records[:5]],
        "token_vocabulary_size": len(tokenizer),
        "batch_tensor_shapes": batch_shapes,
        "warnings": warnings,
    }
    return summary


def print_inspection(summary: dict[str, Any]) -> None:
    print(f"records: {summary['records']}")
    print(f"loaded_records: {summary['loaded_records']}")
    print("splits:")
    for split, count in sorted(summary["splits"].items()):
        print(f"  {split}: {count}")
    for split, arrays in summary["npz"].items():
        print(f"{split}.npz:")
        for key in ("alpha", "index_map", "palette", "palette_mask", "role_map", "category_id", "sprite_id"):
            if key in arrays:
                desc = arrays[key]
                range_text = ""
                if "min" in desc and "max" in desc:
                    range_text = f" range=[{desc['min']}, {desc['max']}]"
                print(f"  {key}: shape={desc['shape']} dtype={desc['dtype']}{range_text}")
    print("caption examples:")
    for caption in summary["caption_examples"]:
        print(f"  - {caption}")
    print(f"token vocabulary size: {summary['token_vocabulary_size']}")
    print("batch tensor shapes:")
    for key, shape in sorted(summary["batch_tensor_shapes"].items()):
        print(f"  {key}: {shape}")
    warnings = summary.get("warnings") or []
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
