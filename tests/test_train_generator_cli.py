from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset

from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.cli import main as train_cli
from spritelab.training.train_generator import GeneratorTrainConfig, run_generator_training


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def test_generator_training_loop_writes_metrics_checkpoint_and_sample(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "run"
    report = run_generator_training(
        GeneratorTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=2,
            device="cpu",
            seed=123,
            latent_dim=4,
            embed_dim=8,
            hidden_channels=8,
            sample_every=1,
            save_every=0,
        )
    )
    assert report["steps_completed"] == 2
    assert (out / "train_metrics.jsonl").is_file()
    assert (out / "train_report.json").is_file()
    assert (out / "checkpoint_last.pt").is_file()
    assert (out / "samples_step_000001.png").is_file()
    assert (out / "samples_final.png").is_file()
    assert len((out / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_generator_overfit_tiny_batch_reduces_loss(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "overfit"
    report = run_generator_training(
        GeneratorTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=16,
            learning_rate=0.05,
            overfit_batches=1,
            device="cpu",
            seed=456,
            latent_dim=4,
            embed_dim=8,
            hidden_channels=8,
            sample_every=0,
            save_every=0,
        )
    )
    assert report["final_train_loss"] < report["initial_train_loss"]
    assert report["loss_decreased"] is True


def test_generator_cli_runs_two_steps(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "cli_run"
    train_cli(
        [
            "generator",
            "--dataset",
            str(dataset),
            "--training-manifest",
            str(manifest),
            "--out",
            str(out),
            "--batch-size",
            "2",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--seed",
            "321",
            "--latent-dim",
            "4",
            "--embed-dim",
            "8",
            "--hidden-channels",
            "8",
            "--sample-every",
            "0",
            "--save-every",
            "0",
        ]
    )
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["steps_completed"] == 2
    assert (out / "checkpoint_last.pt").is_file()


def test_generator_cli_accepts_framing_options_and_writes_metrics(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "cli_framing_run"
    train_cli(
        [
            "generator",
            "--dataset",
            str(dataset),
            "--training-manifest",
            str(manifest),
            "--out",
            str(out),
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--device",
            "cpu",
            "--seed",
            "321",
            "--latent-dim",
            "4",
            "--embed-dim",
            "8",
            "--hidden-channels",
            "8",
            "--sample-every",
            "0",
            "--save-every",
            "0",
            "--border-alpha-weight",
            "0.5",
            "--center-weight",
            "0.05",
            "--margin-band-weight",
            "0.1",
            "--margin-band-size",
            "1",
        ]
    )
    metric = json.loads((out / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "loss_border_alpha" in metric
    assert "loss_center" in metric
    assert "loss_margin_band" in metric
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["framing_loss_config"]["border_alpha_weight"] == 0.5
