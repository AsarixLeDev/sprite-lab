from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.cli import main as train_cli
from spritelab.training.generator_models import TinyCaptionSpriteGenerator
from spritelab.training.sample_generator import SampleGeneratorConfig, run_sample_generator
from spritelab.training.tokenization import SpriteTextTokenizer


def _fake_checkpoint(path: Path) -> Path:
    tokenizer = SpriteTextTokenizer.build(["red potion", "gold sword"], max_length=8)
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
            "step": 0,
        },
        path,
    )
    return path


def _prompts(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_id": "p0",
                        "prompt": "red potion",
                        "category": "seen_object",
                        "target_semantics": {"base_object": "potion"},
                    }
                ),
                json.dumps({"prompt_id": "p1", "prompt": "gold sword", "category": "seen_object"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_sample_generator_writes_manifest_reports_contact_sheet_and_metadata(tmp_path: Path) -> None:
    ckpt = _fake_checkpoint(tmp_path / "checkpoint.pt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    out = tmp_path / "generated"
    report = run_sample_generator(
        SampleGeneratorConfig(
            checkpoint=ckpt,
            prompts=prompts,
            out_dir=out,
            max_samples=2,
            max_colors=8,
            device="cpu",
            seed=77,
            noise_seed=1000,
            batch_size=2,
        )
    )
    assert report["sample_count"] == 2
    assert (out / "generated_manifest.jsonl").is_file()
    assert (out / "generation_report.json").is_file()
    assert (out / "generation_report.md").is_file()
    assert (out / "generation_contact_sheet.png").is_file()

    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["prompt_id"] == "p0"
    assert rows[0]["prompt"] == "red potion"
    assert rows[0]["target_semantics"] == {"base_object": "potion"}
    assert rows[0]["noise_seed"] == 1000
    assert (out / rows[0]["paths"]["raw_rgba"]).is_file()
    assert (out / rows[0]["paths"]["hard_rgba"]).is_file()
    assert (out / rows[0]["paths"]["indexed_png"]).is_file()


def test_sample_generator_cli_runs(tmp_path: Path) -> None:
    ckpt = _fake_checkpoint(tmp_path / "checkpoint.pt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    out = tmp_path / "cli_generated"
    train_cli(
        [
            "sample-generator",
            "--checkpoint",
            str(ckpt),
            "--prompts",
            str(prompts),
            "--out",
            str(out),
            "--max-samples",
            "1",
            "--max-colors",
            "8",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--noise-seed",
            "2000",
        ]
    )
    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["noise_seed"] == 2000
