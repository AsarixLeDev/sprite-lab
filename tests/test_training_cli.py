from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset

from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.cli import main as train_cli
from spritelab.training.eval_baseline import evaluate_baseline_checkpoint
from spritelab.training.train_baseline import BaselineTrainConfig, inspect_training_data, run_baseline_training


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def test_inspect_data_cli_runs(tmp_path: Path, capsys) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    train_cli(["inspect-data", "--dataset", str(dataset), "--training-manifest", str(manifest), "--max-records", "3"])
    output = capsys.readouterr().out
    assert "records:" in output
    assert "alpha:" in output
    assert "token vocabulary size:" in output
    assert "batch tensor shapes:" in output


def test_training_loop_writes_metrics_and_checkpoint(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "run"
    report = run_baseline_training(
        BaselineTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=2,
            device="cpu",
            seed=123,
            hidden_dim=16,
        )
    )
    assert report["steps_completed"] == 2
    assert (out / "train_metrics.jsonl").is_file()
    assert (out / "train_report.json").is_file()
    assert (out / "checkpoint_last.pt").is_file()
    assert len((out / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_overfit_tiny_batch_reduces_loss(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "overfit"
    report = run_baseline_training(
        BaselineTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=12,
            overfit_batches=1,
            device="cpu",
            seed=456,
            hidden_dim=16,
        )
    )
    assert report["final_train_loss"] < report["initial_train_loss"]
    assert report["loss_decreased"] is True


def test_eval_baseline_loads_checkpoint(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "run"
    run_baseline_training(
        BaselineTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=2,
            device="cpu",
            seed=789,
            hidden_dim=16,
        )
    )
    eval_report = evaluate_baseline_checkpoint(
        dataset_dir=dataset,
        training_manifest=manifest,
        checkpoint=out / "checkpoint_last.pt",
        split="val",
        out_dir=tmp_path / "eval",
        batch_size=2,
        device="cpu",
    )
    assert eval_report["records"] == 2
    assert (tmp_path / "eval" / "eval_report.json").is_file()


def test_baseline_cli_runs_two_steps(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "cli_run"
    train_cli(
        [
            "baseline",
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
        ]
    )
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["steps_completed"] == 2


def test_real_496_training_manifest_can_be_inspected_if_present() -> None:
    dataset = Path("datasets/oga_496_rpg_icons_32fix_label_v2_semantic_v3")
    manifest = dataset / "training_manifest.jsonl"
    if not manifest.is_file():
        pytest.skip("reference 496 dataset is not present")
    summary = inspect_training_data(dataset_dir=dataset, training_manifest=manifest, max_records=8)
    assert summary["records"] >= 1
    assert summary["token_vocabulary_size"] >= 4
    assert "alpha" in summary["npz"]["train"]
