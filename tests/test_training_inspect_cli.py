from __future__ import annotations

from pathlib import Path

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.cli import main as train_cli


def test_inspect_data_cli_runs_without_requiring_torch(tmp_path: Path, capsys) -> None:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)

    train_cli(["inspect-data", "--dataset", str(dataset), "--training-manifest", str(manifest), "--max-records", "3"])

    output = capsys.readouterr().out
    assert "records: 12" in output
    assert "alpha:" in output
    assert "token vocabulary size:" in output
    assert "batch tensor shapes:" in output
