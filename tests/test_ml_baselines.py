"""Tests for spritelab.ml.baselines."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _ml_testdata import write_synthetic_dataset

from spritelab.codec.bundle import INDEX_MASK, INDEX_PAD
from spritelab.ml.baselines import (
    CopyVisibleBaseline,
    MajorityIndexBaseline,
    PaletteRampBaseline,
    PerCategoryMajorityIndexBaseline,
    run_baseline_evaluation,
)
from spritelab.ml.dataset import SpriteBundleDataset
from spritelab.ml.masking import FixedOpaqueMask


@pytest.fixture()
def samples(tmp_path):
    dataset = SpriteBundleDataset(
        write_synthetic_dataset(tmp_path),
        "train",
        transform=FixedOpaqueMask(mask_fraction=0.5, seed=1),
    )
    return [dataset[i] for i in range(len(dataset))]


def test_majority_baseline_fit_predict(samples):
    baseline = MajorityIndexBaseline().fit(samples)
    assert baseline.majority_index == 1  # index 1 dominates the square
    prediction = baseline.predict(samples[0])
    masked = samples[0]["loss_mask"]
    assert bool((prediction[masked] == 1).all())


def test_prediction_shape(samples):
    prediction = MajorityIndexBaseline().fit(samples).predict(samples[0])
    assert prediction.shape == (32, 32)


def test_transparent_predicts_zero(samples):
    for baseline in (
        MajorityIndexBaseline().fit(samples),
        PerCategoryMajorityIndexBaseline().fit(samples),
        PaletteRampBaseline().fit(samples),
        CopyVisibleBaseline().fit(samples),
    ):
        prediction = baseline.predict(samples[0])
        assert bool((prediction[samples[0]["alpha"] == 0] == 0).all())


def test_no_mask_or_pad_tokens(samples):
    for baseline in (
        MajorityIndexBaseline().fit(samples),
        PerCategoryMajorityIndexBaseline().fit(samples),
        PaletteRampBaseline().fit(samples),
        CopyVisibleBaseline().fit(samples),
    ):
        prediction = baseline.predict(samples[0])
        assert not bool((prediction == INDEX_MASK).any())
        assert not bool((prediction == INDEX_PAD).any())


def test_per_category_fallback(samples):
    baseline = PerCategoryMajorityIndexBaseline().fit(samples)
    unseen = dict(samples[0])
    unseen["category_id"] = torch.tensor(999)
    prediction = baseline.predict(unseen)
    masked = unseen["loss_mask"]
    assert bool((prediction[masked] == baseline.global_majority).all())


def test_palette_ramp_clamps_to_valid_rows(samples):
    prediction = PaletteRampBaseline().predict(samples[0])
    palette_mask = samples[0]["palette_mask"]
    assert bool((prediction >= 0).all())
    assert bool((prediction < palette_mask.shape[0]).all())
    assert bool(palette_mask[prediction.flatten()].all())


def test_copy_visible_preserves_unmasked(samples):
    prediction = CopyVisibleBaseline().fit(samples).predict(samples[0])
    unmasked = (samples[0]["alpha"] == 1) & ~samples[0]["loss_mask"]
    assert bool((prediction[unmasked] == samples[0]["index_map"][unmasked]).all())


def test_evaluation_writes_metrics_json(tmp_path):
    dataset_dir = write_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "out"
    results = run_baseline_evaluation(dataset_dir, "train", out_dir, mask_fraction=0.5)
    metrics_path = out_dir / "baseline_metrics.json"
    assert metrics_path.exists()
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(loaded["baselines"]) == set(results["baselines"])
    assert (out_dir / "preview_copy_visible.png").exists()
    assert (out_dir / "preview_palette_ramp.png").exists()
