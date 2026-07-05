"""Tests for spritelab.ml.metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spritelab.ml.metrics import (
    ReconstructionMetrics,
    average_reconstruction_metrics,
    compute_reconstruction_metrics,
    metrics_to_dict,
)


def _fixture():
    alpha = torch.zeros(32, 32, dtype=torch.long)
    alpha[10:20, 10:20] = 1
    target = torch.zeros(32, 32, dtype=torch.long)
    target[10:20, 10:20] = 2
    palette_mask = torch.tensor([True, True, True, True, False, False])
    return alpha, target, palette_mask


def test_perfect_prediction():
    alpha, target, palette_mask = _fixture()
    metrics = compute_reconstruction_metrics(target.clone(), target, alpha, palette_mask)
    assert metrics.token_accuracy == 1.0
    assert metrics.opaque_accuracy == 1.0
    assert metrics.transparent_accuracy == 1.0
    assert metrics.invalid_token_rate == 0.0


def test_wrong_opaque_lowers_opaque_accuracy():
    alpha, target, palette_mask = _fixture()
    prediction = target.clone()
    prediction[10, 10] = 3
    metrics = compute_reconstruction_metrics(prediction, target, alpha, palette_mask)
    assert metrics.opaque_accuracy < 1.0
    assert metrics.transparent_accuracy == 1.0


def test_wrong_transparent_lowers_transparent_accuracy():
    alpha, target, palette_mask = _fixture()
    prediction = target.clone()
    prediction[0, 0] = 1
    metrics = compute_reconstruction_metrics(prediction, target, alpha, palette_mask)
    assert metrics.transparent_accuracy < 1.0
    assert metrics.opaque_accuracy == 1.0


def test_masked_accuracy_uses_loss_mask():
    alpha, target, palette_mask = _fixture()
    prediction = target.clone()
    prediction[10, 10] = 3  # wrong, inside loss mask
    prediction[19, 19] = 3  # wrong, outside loss mask
    loss_mask = torch.zeros(32, 32, dtype=torch.bool)
    loss_mask[10, 10] = True
    loss_mask[10, 11] = True
    metrics = compute_reconstruction_metrics(prediction, target, alpha, palette_mask, loss_mask)
    assert metrics.masked_accuracy == 0.5
    assert metrics.loss_pixel_count == 2


def test_invalid_token_out_of_range():
    alpha, target, palette_mask = _fixture()
    prediction = target.clone()
    prediction[0, 0] = 99
    metrics = compute_reconstruction_metrics(prediction, target, alpha, palette_mask)
    assert metrics.invalid_token_rate > 0.0


def test_invalid_token_palette_mask_false():
    alpha, target, palette_mask = _fixture()
    prediction = target.clone()
    prediction[0, 0] = 4  # palette_mask[4] is False
    metrics = compute_reconstruction_metrics(prediction, target, alpha, palette_mask)
    assert metrics.invalid_token_rate == pytest.approx(1 / 1024)


def test_aggregation_averages():
    a = ReconstructionMetrics(1.0, 1.0, 1.0, 0.0, 1.0, 10)
    b = ReconstructionMetrics(0.5, 0.0, 0.5, 0.2, 0.0, 30)
    average = average_reconstruction_metrics([a, b])
    assert average.token_accuracy == 0.75
    assert average.opaque_accuracy == 0.5
    assert average.masked_accuracy == 0.75
    assert average.invalid_token_rate == pytest.approx(0.1)
    assert average.loss_pixel_count == 40
    assert metrics_to_dict(average)["token_accuracy"] == 0.75
