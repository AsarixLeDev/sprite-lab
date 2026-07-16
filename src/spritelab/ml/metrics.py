"""Reconstruction metrics for masked index-map prediction.

Metrics over empty pixel sets (for example ``masked_accuracy`` when no
pixel is masked) are reported as ``0.0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ReconstructionMetrics:
    token_accuracy: float
    opaque_accuracy: float
    masked_accuracy: float
    invalid_token_rate: float
    transparent_accuracy: float
    loss_pixel_count: int


def _masked_accuracy(correct: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum())
    if count == 0:
        return 0.0
    return float(correct[mask].float().mean())


def compute_reconstruction_metrics(
    prediction: torch.Tensor,
    target_index_map: torch.Tensor,
    alpha: torch.Tensor,
    palette_mask: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> ReconstructionMetrics:
    """Compute per-sample reconstruction metrics for one [32, 32] prediction."""

    prediction = prediction.long()
    target = target_index_map.long()
    correct = prediction == target
    opaque = alpha == 1
    transparent = alpha == 0

    num_rows = int(palette_mask.shape[0])
    out_of_range = (prediction < 0) | (prediction >= num_rows)
    clamped = prediction.clamp(0, num_rows - 1)
    masked_out = ~palette_mask.bool()[clamped]
    invalid = out_of_range | masked_out

    if loss_mask is None:
        masked_accuracy = 0.0
        loss_pixel_count = 0
    else:
        masked_accuracy = _masked_accuracy(correct, loss_mask.bool())
        loss_pixel_count = int(loss_mask.sum())

    return ReconstructionMetrics(
        token_accuracy=float(correct.float().mean()),
        opaque_accuracy=_masked_accuracy(correct, opaque),
        masked_accuracy=masked_accuracy,
        invalid_token_rate=float(invalid.float().mean()),
        transparent_accuracy=_masked_accuracy(correct, transparent),
        loss_pixel_count=loss_pixel_count,
    )


def average_reconstruction_metrics(
    metrics: Sequence[ReconstructionMetrics],
) -> ReconstructionMetrics:
    """Return the unweighted mean of the metrics (sum for pixel counts)."""

    if not metrics:
        return ReconstructionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    count = len(metrics)
    return ReconstructionMetrics(
        token_accuracy=sum(m.token_accuracy for m in metrics) / count,
        opaque_accuracy=sum(m.opaque_accuracy for m in metrics) / count,
        masked_accuracy=sum(m.masked_accuracy for m in metrics) / count,
        invalid_token_rate=sum(m.invalid_token_rate for m in metrics) / count,
        transparent_accuracy=sum(m.transparent_accuracy for m in metrics) / count,
        loss_pixel_count=sum(m.loss_pixel_count for m in metrics),
    )


def metrics_to_dict(metrics: ReconstructionMetrics) -> dict[str, Any]:
    """Return metrics as a JSON-serializable dictionary."""

    return asdict(metrics)
