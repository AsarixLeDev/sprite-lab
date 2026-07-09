from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.cli import main as train_cli
from spritelab.training.generator_challenger import RectifiedFlowUNet
from spritelab.training.prompt_sensitivity import (
    PromptSensitivityConfig,
    discover_prompt_pairs,
    is_near_duplicate,
    pairwise_image_metrics,
    run_prompt_sensitivity,
)
from spritelab.training.tokenization import SpriteTextTokenizer


def _fake_checkpoint(path: Path) -> Path:
    tokenizer = SpriteTextTokenizer.build(
        [
            "red potion",
            "blue potion",
            "gold sword",
            "iron sword",
            "seen object colors red blue gold iron",
        ],
        max_length=8,
    )
    model = RectifiedFlowUNet(
        vocab_size=len(tokenizer),
        embed_dim=8,
        base_channels=16,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=tokenizer.pad_id,
    )
    torch.save(
        {
            "model_type": "generator_challenger",
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "vocab": tokenizer.to_json_dict(),
            "checkpoint_type": "generator_challenger_rectified_flow_v0",
            "conditioning_mode": "caption_semantic",
            "train_config": {"conditioning_mode": "caption_semantic", "semantic_max_length": 12},
            "step": 0,
        },
        path,
    )
    return path


def _prompts(path: Path) -> Path:
    rows = [
        {
            "prompt_id": "red_potion",
            "prompt": "red potion",
            "category": "seen_object",
            "target_semantics": {"base_object": "potion", "attributes": {"colors": ["red"]}},
        },
        {
            "prompt_id": "blue_potion",
            "prompt": "blue potion",
            "category": "seen_object",
            "target_semantics": {"base_object": "potion", "attributes": {"colors": ["blue"]}},
        },
        {
            "prompt_id": "gold_sword",
            "prompt": "gold sword",
            "category": "seen_object",
            "target_semantics": {"base_object": "sword", "attributes": {"colors": ["gold"]}},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_pairwise_metrics_and_near_duplicate_detection() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:16, 8:16, :3] = [255, 0, 0]
    rgba[8:16, 8:16, 3] = 255
    same = pairwise_image_metrics(rgba, rgba.copy())
    assert same["alpha_iou"] == pytest.approx(1.0)
    assert same["rgb_mae_visible_union"] == pytest.approx(0.0)
    assert is_near_duplicate(same)

    shifted = np.zeros((32, 32, 4), dtype=np.uint8)
    shifted[10:18, 10:18, :3] = [0, 0, 255]
    shifted[10:18, 10:18, 3] = 255
    changed = pairwise_image_metrics(rgba, shifted)
    assert changed["alpha_iou"] < 1.0
    assert changed["rgb_histogram_distance"] > 0.0


def test_discovers_exact_prompt_pair_from_eval_prompts(tmp_path: Path) -> None:
    records = [
        json.loads(line) for line in _prompts(tmp_path / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    pairs = discover_prompt_pairs(records, max_pairs=2)
    assert pairs[0]["pair_id"] == "red_potion__blue_potion"
    assert pairs[0]["source"] == "exact_eval_prompt"


def test_prompt_sensitivity_writes_sets_reports_contact_sheets_and_metadata(tmp_path: Path) -> None:
    ckpt = _fake_checkpoint(tmp_path / "checkpoint.pt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    out = tmp_path / "sensitivity"
    report = run_prompt_sensitivity(
        PromptSensitivityConfig(
            checkpoint=ckpt,
            prompts=prompts,
            out_dir=out,
            device="cpu",
            seed=77,
            max_prompts=3,
            noise_samples=2,
            max_pairs=1,
            max_colors=8,
            batch_size=2,
        )
    )
    assert report["conditioning_mode"] == "caption_semantic"
    assert (out / "prompt_sensitivity_report.json").is_file()
    assert (out / "prompt_sensitivity_report.md").is_file()
    assert (out / "prompt_sensitivity_contact_sheet.png").is_file()

    for folder in ("same_noise_different_prompts", "same_prompt_different_noise", "prompt_pairs"):
        assert (out / folder / "generated_manifest.jsonl").is_file()
        assert (out / folder / "generation_report.json").is_file()
        assert (out / folder / "generation_contact_sheet.png").is_file()

    rows = [
        json.loads(line)
        for line in (out / "same_noise_different_prompts" / "generated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["prompt_id"] == "red_potion"
    assert rows[0]["target_semantics"]["base_object"] == "potion"
    assert rows[0]["same_noise"] is True
    assert report["sets"]["same_noise_different_prompts"]["metrics"]["pair_count"] == 3


def test_prompt_sensitivity_cli_runs(tmp_path: Path) -> None:
    ckpt = _fake_checkpoint(tmp_path / "checkpoint.pt")
    prompts = _prompts(tmp_path / "prompts.jsonl")
    out = tmp_path / "cli_sensitivity"
    train_cli(
        [
            "prompt-sensitivity",
            "--checkpoint",
            str(ckpt),
            "--prompts",
            str(prompts),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--seed",
            "77",
            "--max-prompts",
            "2",
            "--noise-samples",
            "2",
            "--max-pairs",
            "1",
            "--max-colors",
            "8",
        ]
    )
    assert (out / "prompt_sensitivity_report.json").is_file()
