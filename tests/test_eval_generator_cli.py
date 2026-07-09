from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.eval_generator import evaluate_generator_checkpoint
from spritelab.training.train_generator import GeneratorTrainConfig, run_generator_training


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=12)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def _train_checkpoint(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "run"
    run_generator_training(
        GeneratorTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=1,
            device="cpu",
            seed=789,
            latent_dim=4,
            embed_dim=8,
            hidden_channels=8,
            sample_every=0,
            save_every=0,
        )
    )
    return dataset, manifest, out / "checkpoint_last.pt"


def test_eval_generator_loads_checkpoint_and_writes_report_and_samples(tmp_path: Path) -> None:
    dataset, manifest, checkpoint = _train_checkpoint(tmp_path)
    eval_report = evaluate_generator_checkpoint(
        dataset_dir=dataset,
        training_manifest=manifest,
        checkpoint=checkpoint,
        split="val",
        out_dir=tmp_path / "eval",
        batch_size=2,
        device="cpu",
    )
    assert eval_report["records"] == 2
    assert eval_report["loss"] is not None
    assert (tmp_path / "eval" / "eval_report.json").is_file()
    assert (tmp_path / "eval" / "eval_samples.png").is_file()


def test_eval_generator_prompt_only_generation_writes_samples(tmp_path: Path) -> None:
    _dataset, _manifest, checkpoint = _train_checkpoint(tmp_path)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                json.dumps({"prompt_id": "p0", "prompt": "red potion"}),
                json.dumps({"prompt_id": "p1", "prompt": "gold sword"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = evaluate_generator_checkpoint(
        checkpoint=checkpoint,
        prompts=prompts,
        out_dir=tmp_path / "prompt_eval",
        batch_size=2,
        device="cpu",
    )
    assert report["records"] == 0
    assert report["prompt_count"] == 2
    assert (tmp_path / "prompt_eval" / "eval_report.json").is_file()
    assert (tmp_path / "prompt_eval" / "prompt_samples.png").is_file()
