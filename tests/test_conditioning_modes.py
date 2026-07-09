from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.cli import main as train_cli
from spritelab.training.conditioning import (
    DEFAULT_CONDITIONING_MODE,
    apply_conditioning_mode,
    checkpoint_conditioning_mode,
)
from spritelab.training.tokenization import semantic_strings_from_record
from spritelab.training.train_generator import GeneratorTrainConfig, run_generator_training


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def test_semantic_tokens_include_requested_manifest_fields() -> None:
    record = {
        "category": "weapon",
        "base_object": "sword",
        "object_name": "iron_sword",
        "colors": ["gray"],
        "materials": ["iron"],
        "effects": ["charged"],
        "function": ["attack"],
        "style": ["pixel_art"],
    }
    text = " ".join(semantic_strings_from_record(record))
    for token in ("weapon", "sword", "iron_sword", "gray", "iron", "charged", "attack", "pixel_art"):
        assert token in text


def test_conditioning_modes_route_caption_and_semantic_streams() -> None:
    caption = torch.tensor([[2, 10, 3, 0], [2, 11, 3, 0]], dtype=torch.long)
    semantic = torch.tensor([[2, 20, 3, 0], [2, 21, 3, 0]], dtype=torch.long)

    caption_only = apply_conditioning_mode(
        caption_tokens=caption,
        semantic_tokens=semantic,
        mode="caption",
        pad_token_id=0,
    )
    assert torch.equal(caption_only["caption_tokens"], caption)
    assert caption_only["semantic_tokens"] is None

    semantic_only = apply_conditioning_mode(
        caption_tokens=caption,
        semantic_tokens=semantic,
        mode="semantic",
        pad_token_id=0,
    )
    assert torch.count_nonzero(semantic_only["caption_tokens"]) == 0
    assert torch.equal(semantic_only["semantic_tokens"], semantic)

    unconditioned = apply_conditioning_mode(
        caption_tokens=caption,
        semantic_tokens=semantic,
        mode="none",
        pad_token_id=0,
    )
    assert torch.count_nonzero(unconditioned["caption_tokens"]) == 0
    assert unconditioned["semantic_tokens"] is None

    structured = {"category_id": torch.tensor([1, 2], dtype=torch.long)}
    structured_mode = apply_conditioning_mode(
        caption_tokens=caption,
        semantic_tokens=semantic,
        structured_conditioning=structured,
        mode="caption_semantic_structured",
        pad_token_id=0,
    )
    assert torch.equal(structured_mode["caption_tokens"], caption)
    assert torch.equal(structured_mode["semantic_tokens"], semantic)
    assert structured_mode["structured_conditioning"] is structured


def test_old_checkpoint_conditioning_mode_falls_back_to_default() -> None:
    assert checkpoint_conditioning_mode({"checkpoint_type": "caption_rgba_generator_v0"}) == DEFAULT_CONDITIONING_MODE


def test_training_records_conditioning_mode_and_checkpoint(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "none_run"
    report = run_generator_training(
        GeneratorTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=1,
            device="cpu",
            seed=123,
            latent_dim=4,
            embed_dim=8,
            hidden_channels=8,
            sample_every=0,
            save_every=0,
            conditioning_mode="none",
        )
    )
    assert report["conditioning_mode"] == "none"
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["conditioning_mode"] == "none"
    ckpt = torch.load(out / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    assert ckpt["conditioning_mode"] == "none"
    assert ckpt["train_config"]["conditioning_mode"] == "none"


def test_generator_cli_accepts_all_conditioning_modes(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    for mode in ("caption", "semantic", "caption_semantic", "none"):
        out = tmp_path / f"cli_{mode}"
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
                "--conditioning-mode",
                mode,
            ]
        )
        report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
        assert report["conditioning_mode"] == mode
