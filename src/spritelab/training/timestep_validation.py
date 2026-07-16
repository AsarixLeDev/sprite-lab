"""Deterministic, support-reporting validation buckets for flow timesteps."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

TIMESTEP_BUCKET_NAMES = ("early", "low-mid", "mid", "high-mid", "late")
DEFAULT_TIMESTEP_BOUNDARIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TIMESTEP_METRIC_NAMES = ("loss_velocity", "loss_palette_aux", "loss_index_head")


def validate_timestep_boundaries(boundaries: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in boundaries)
    if len(values) != len(TIMESTEP_BUCKET_NAMES) + 1:
        raise ValueError("timestep bucket boundaries must contain exactly six values")
    if values[0] != 0.0 or values[-1] != 1.0 or any(a >= b for a, b in pairwise(values)):
        raise ValueError("timestep bucket boundaries must be strictly increasing from 0.0 to 1.0")
    return values


def timestep_bucket_index(value: float, boundaries: Sequence[float] = DEFAULT_TIMESTEP_BOUNDARIES) -> int:
    values = validate_timestep_boundaries(boundaries)
    timestep = float(value)
    if not 0.0 <= timestep <= 1.0:
        raise ValueError(f"timestep must be in [0, 1], got {timestep}")
    for index, upper in enumerate(values[1:]):
        if timestep < upper or index == len(TIMESTEP_BUCKET_NAMES) - 1:
            return index
    raise AssertionError("unreachable")


class TimestepBucketAccumulator:
    def __init__(self, boundaries: Sequence[float] = DEFAULT_TIMESTEP_BOUNDARIES) -> None:
        self.boundaries = validate_timestep_boundaries(boundaries)
        self.counts = [0] * len(TIMESTEP_BUCKET_NAMES)
        self.sums: list[dict[str, float]] = [{} for _ in TIMESTEP_BUCKET_NAMES]
        self.metric_names = set(TIMESTEP_METRIC_NAMES)

    def add(self, timestep: float, metrics: dict[str, Any]) -> None:
        index = timestep_bucket_index(timestep, self.boundaries)
        self.counts[index] += 1
        for key, value in metrics.items():
            self.metric_names.add(str(key))
            self.sums[index][str(key)] = self.sums[index].get(str(key), 0.0) + float(value)

    def report(self) -> dict[str, Any]:
        buckets: dict[str, Any] = {}
        for index, name in enumerate(TIMESTEP_BUCKET_NAMES):
            count = self.counts[index]
            metrics = {
                key: (self.sums[index].get(key, 0.0) / count if count else None) for key in sorted(self.metric_names)
            }
            buckets[name] = {
                "lower": self.boundaries[index],
                "upper": self.boundaries[index + 1],
                "upper_inclusive": index == len(TIMESTEP_BUCKET_NAMES) - 1,
                "sample_count": count,
                **metrics,
            }
        return {"boundaries": list(self.boundaries), "buckets": buckets}
