from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.generator_models import TinyCaptionSpriteGenerator
from spritelab.training.sample_generator import SampleGeneratorConfig, run_sample_generator
from spritelab.training.tokenization import SpriteTextTokenizer


def _checkpoint(path: Path) -> Path:
    tokenizer = SpriteTextTokenizer.build(["red potion"], max_length=8)
    model = TinyCaptionSpriteGenerator(
        vocab_size=len(tokenizer),
        embed_dim=8,
        latent_dim=4,
        hidden_channels=8,
        pad_token_id=tokenizer.pad_id,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "vocab": tokenizer.to_json_dict(),
            "checkpoint_type": "caption_rgba_generator_v0",
            "conditioning_mode": "caption_semantic",
        },
        path,
    )
    return path


def test_sample_generator_writes_contact_sheet_label_mapping(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt_id": "p0", "prompt": "red potion"}) + "\n", encoding="utf-8")
    out = tmp_path / "generated"
    run_sample_generator(
        SampleGeneratorConfig(
            checkpoint=_checkpoint(tmp_path / "checkpoint.pt"),
            prompts=prompts,
            out_dir=out,
            max_samples=1,
            max_colors=8,
            device="cpu",
            seed=1,
            contact_sheet_labels="prompt_and_seed",
        )
    )
    labels = json.loads((out / "contact_sheet_labels.json").read_text(encoding="utf-8"))
    assert labels[0]["sample_id"] == "sample_000000"
    assert labels[0]["prompt"] == "red potion"
    assert labels[0]["label_mode"] == "prompt_and_seed"
    assert (out / "contact_sheet_labels.md").is_file()
