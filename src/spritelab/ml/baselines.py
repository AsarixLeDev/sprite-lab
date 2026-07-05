"""Dumb reconstruction baselines for masked index-map prediction."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from spritelab.codec.roles import (
    ROLE_DEEP_SHADOW,
    ROLE_EMISSIVE,
    ROLE_HIGHLIGHT,
    ROLE_LIGHT,
    ROLE_OUTLINE,
    ROLE_SHADOW,
    ROLE_TRANSPARENT,
)
from spritelab.ml.dataset import SpriteBundleDataset
from spritelab.ml.masking import FixedOpaqueMask
from spritelab.ml.metrics import (
    average_reconstruction_metrics,
    compute_reconstruction_metrics,
    metrics_to_dict,
)
from spritelab.ml.previews import save_prediction_grid

_LOW_ROLES = {ROLE_OUTLINE, ROLE_DEEP_SHADOW, ROLE_SHADOW}
_HIGH_ROLES = {ROLE_LIGHT, ROLE_HIGHLIGHT, ROLE_EMISSIVE}


def _visible_index_counts(sample: dict[str, Any]) -> Counter:
    index_map = sample["index_map"]
    opaque = sample["alpha"] == 1
    values = index_map[opaque & (index_map >= 1)]
    return Counter(int(value) for value in values)


def _majority(counter: Counter, fallback: int = 1) -> int:
    if not counter:
        return fallback
    return max(counter.items(), key=lambda pair: (pair[1], -pair[0]))[0]


def _base_prediction(sample: dict[str, Any]) -> torch.Tensor:
    """Start from the unmasked visible pixels, zero elsewhere."""

    input_map = sample.get("input_index_map", sample["index_map"]).long()
    loss_mask = sample.get("loss_mask")
    prediction = input_map.clone()
    if loss_mask is not None:
        prediction[loss_mask] = 0
    prediction[sample["alpha"] == 0] = 0
    return prediction


def _fill_masked(sample: dict[str, Any], prediction: torch.Tensor, value: int) -> torch.Tensor:
    loss_mask = sample.get("loss_mask")
    fill = loss_mask if loss_mask is not None else (sample["alpha"] == 1)
    prediction[fill] = int(value)
    return prediction


class MajorityIndexBaseline:
    """Predict the globally most common visible index on masked pixels."""

    def __init__(self) -> None:
        self.majority_index = 1

    def fit(self, dataset: Iterable[dict[str, Any]]) -> "MajorityIndexBaseline":
        counts: Counter = Counter()
        for sample in dataset:
            counts.update(_visible_index_counts(sample))
        self.majority_index = _majority(counts)
        return self

    def predict(self, sample: dict[str, Any]) -> torch.Tensor:
        return _fill_masked(sample, _base_prediction(sample), self.majority_index)


class PerCategoryMajorityIndexBaseline:
    """Majority visible index per category, global fallback for unseen ones."""

    def __init__(self) -> None:
        self.global_majority = 1
        self.per_category: dict[int, int] = {}

    def fit(self, dataset: Iterable[dict[str, Any]]) -> "PerCategoryMajorityIndexBaseline":
        global_counts: Counter = Counter()
        category_counts: dict[int, Counter] = {}
        for sample in dataset:
            counts = _visible_index_counts(sample)
            global_counts.update(counts)
            category = int(sample["category_id"])
            category_counts.setdefault(category, Counter()).update(counts)
        self.global_majority = _majority(global_counts)
        self.per_category = {
            category: _majority(counts, fallback=self.global_majority)
            for category, counts in category_counts.items()
        }
        return self

    def predict(self, sample: dict[str, Any]) -> torch.Tensor:
        category = int(sample["category_id"])
        value = self.per_category.get(category, self.global_majority)
        return _fill_masked(sample, _base_prediction(sample), value)


class PaletteRampBaseline:
    """Map shadow/midtone/highlight roles onto low/middle/high palette slots."""

    def fit(self, dataset: Iterable[dict[str, Any]]) -> "PaletteRampBaseline":
        return self

    def predict(self, sample: dict[str, Any]) -> torch.Tensor:
        prediction = _base_prediction(sample)
        palette_mask = sample["palette_mask"].bool()
        valid_rows = palette_mask.nonzero(as_tuple=False).flatten().tolist()
        visible_rows = [row for row in valid_rows if row >= 1] or [0]
        low = visible_rows[0]
        high = visible_rows[-1]
        middle = visible_rows[len(visible_rows) // 2]

        role_map = sample.get("role_map")
        loss_mask = sample.get("loss_mask")
        fill = loss_mask if loss_mask is not None else (sample["alpha"] == 1)
        positions = fill.nonzero(as_tuple=False)
        for position in positions:
            y, x = int(position[0]), int(position[1])
            role = int(role_map[y, x]) if role_map is not None else -1
            if role == ROLE_TRANSPARENT:
                value = middle
            elif role in _LOW_ROLES:
                value = low
            elif role in _HIGH_ROLES:
                value = high
            else:
                value = middle
            prediction[y, x] = value
        prediction[sample["alpha"] == 0] = 0
        return prediction


class CopyVisibleBaseline:
    """Copy unmasked pixels, fill masked ones with the sample-local majority."""

    def __init__(self) -> None:
        self.global_majority = 1

    def fit(self, dataset: Iterable[dict[str, Any]]) -> "CopyVisibleBaseline":
        counts: Counter = Counter()
        for sample in dataset:
            counts.update(_visible_index_counts(sample))
        self.global_majority = _majority(counts)
        return self

    def predict(self, sample: dict[str, Any]) -> torch.Tensor:
        input_map = sample.get("input_index_map", sample["index_map"]).long()
        loss_mask = sample.get("loss_mask")
        visible = (
            (sample["alpha"] == 1)
            & (input_map >= 1)
            & (~loss_mask if loss_mask is not None else True)
        )
        counts = Counter(int(value) for value in input_map[visible])
        value = _majority(counts, fallback=self.global_majority)
        return _fill_masked(sample, _base_prediction(sample), value)


def run_baseline_evaluation(
    dataset_root: str | Path,
    split: str,
    output_dir: str | Path,
    mask_fraction: float = 0.5,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Fit and evaluate all baselines; write metrics JSON and preview grids."""

    dataset = SpriteBundleDataset(
        dataset_root,
        split,
        transform=FixedOpaqueMask(mask_fraction=mask_fraction, seed=1337),
    )
    count = len(dataset)
    if max_samples is not None:
        count = min(count, max_samples)
    samples = [dataset[index] for index in range(count)]

    baselines = {
        "majority_index": MajorityIndexBaseline().fit(samples),
        "per_category_majority_index": PerCategoryMajorityIndexBaseline().fit(samples),
        "palette_ramp": PaletteRampBaseline().fit(samples),
        "copy_visible": CopyVisibleBaseline().fit(samples),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "split": split,
        "mask_fraction": mask_fraction,
        "sample_count": count,
        "baselines": {},
    }
    for name, baseline in baselines.items():
        predictions = [baseline.predict(sample) for sample in samples]
        metrics = [
            compute_reconstruction_metrics(
                prediction,
                sample["target_index_map"],
                sample["alpha"],
                sample["palette_mask"],
                sample["loss_mask"],
            )
            for sample, prediction in zip(samples, predictions)
        ]
        results["baselines"][name] = metrics_to_dict(
            average_reconstruction_metrics(metrics)
        )
        if name in ("copy_visible", "palette_ramp"):
            save_prediction_grid(
                samples, predictions, output_dir / f"preview_{name}.png"
            )

    metrics_path = output_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results
