"""Tests for spritelab.ml.overfit."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _ml_testdata import write_synthetic_dataset

from spritelab.codec.bundle import INDEX_MASK
from spritelab.ml.overfit import (
    OverfitConfig,
    TinyIndexMapModel,
    compute_masked_index_loss,
    run_overfit_smoke_test,
)


def test_forward_shape():
    model = TinyIndexMapModel(num_tokens=8)
    input_map = torch.zeros(2, 32, 32, dtype=torch.long)
    input_map[:, 10:20, 10:20] = INDEX_MASK
    alpha = torch.zeros(2, 32, 32, dtype=torch.long)
    alpha[:, 10:20, 10:20] = 1
    logits = model(input_map, alpha, torch.tensor([0, 1]))
    assert logits.shape == (2, 8, 32, 32)


def test_masked_loss_scalar():
    logits = torch.randn(1, 8, 32, 32, requires_grad=True)
    target = torch.ones(1, 32, 32, dtype=torch.long)
    loss_mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    loss_mask[0, 5, 5] = True
    loss = compute_masked_index_loss(logits, target, loss_mask)
    assert loss.dim() == 0
    assert float(loss) > 0
    loss.backward()


def test_masked_loss_empty_mask():
    logits = torch.randn(1, 8, 32, 32, requires_grad=True)
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    loss_mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    loss = compute_masked_index_loss(logits, target, loss_mask)
    assert float(loss) == 0.0
    loss.backward()  # must stay connected to the graph


@pytest.fixture()
def smoke_result(tmp_path):
    dataset_dir = write_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "out"
    config = OverfitConfig(
        dataset_root=dataset_dir,
        output_dir=out_dir,
        max_samples=3,
        steps=4,
        batch_size=2,
    )
    return run_overfit_smoke_test(config), out_dir


def test_runner_works_few_steps(smoke_result):
    result, _ = smoke_result
    assert result["sample_count"] == 3
    assert result["steps"] == 4


def test_runner_writes_metrics_json(smoke_result):
    result, out_dir = smoke_result
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded["final_loss"] == result["final_loss"]


def test_runner_writes_preview_grid(smoke_result):
    _, out_dir = smoke_result
    assert (out_dir / "predictions.png").exists()


def test_result_keys(smoke_result):
    result, _ = smoke_result
    for key in ("initial_loss", "final_loss", "passed"):
        assert key in result
    assert isinstance(result["passed"], bool)
