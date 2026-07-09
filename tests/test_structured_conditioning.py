from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch, read_jsonl
from spritelab.training.generator_challenger import RectifiedFlowUNet
from spritelab.training.ood_prompts import OodCompositionalPromptConfig, build_ood_compositional_prompts
from spritelab.training.structured_conditioning import (
    build_structured_conditioning_vocab,
    encode_structured_conditioning,
)


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def test_structured_vocab_construction_and_missing_fields() -> None:
    rows = [
        {
            "category": "weapon",
            "object_name": "iron_sword",
            "base_object": "sword",
            "conditioning": {
                "semantic_v3": {
                    "attributes": {
                        "colors": ["gray", "blue"],
                        "materials": ["metal"],
                        "shapes": ["long"],
                        "function": ["attack"],
                        "style": ["pixel_art"],
                    }
                }
            },
        },
        {"category": "item_icon", "object_name": "red_potion", "base_object": "potion", "colors": ["red"]},
    ]
    vocab = build_structured_conditioning_vocab(rows)

    assert "weapon" in vocab.categories
    assert "iron_sword" in vocab.objects
    assert "sword" in vocab.base_objects
    assert "gray" in vocab.colors
    assert "attack" in vocab.functions

    encoded = encode_structured_conditioning(rows[0], vocab)
    assert encoded["category_id"] > 0
    assert encoded["object_id"] > 0
    assert sum(encoded["color_multi_hot"]) == 2.0

    missing = encode_structured_conditioning({}, vocab)
    assert missing["category_id"] == 0
    assert missing["object_id"] == 0
    assert sum(missing["color_multi_hot"]) == 0.0


def test_ood_compositional_prompt_builder_writes_default_96_rows(tmp_path: Path) -> None:
    out = tmp_path / "ood.jsonl"
    summary = build_ood_compositional_prompts(OodCompositionalPromptConfig(out=out))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert summary["prompt_count"] == 96
    assert len(rows) == 96
    first = rows[0]
    assert first["prompt"] == "red sword 32x32 pixel art icon"
    assert first["target_sprite_id"] == "ood_red_sword"
    assert first["conditioning"]["semantic_v3"]["base_object"] == "sword"
    assert first["conditioning"]["semantic_v3"]["attributes"]["colors"] == ["red"]


def test_dataset_returns_structured_tensors_when_vocab_is_supplied(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    rows = read_jsonl(manifest)
    vocab = build_structured_conditioning_vocab(row for row in rows if row.get("split") == "train")
    ds = SpriteTrainingDataset(dataset, manifest, split="train", structured_vocab=vocab, max_records=2)

    sample = ds[0]
    assert sample["structured_category_id"].dtype == torch.long
    assert sample["structured_color_multi_hot"].ndim == 1

    batch = collate_sprite_batch([ds[0], ds[1]])
    assert batch["structured_category_id"].shape == (2,)
    assert batch["structured_color_multi_hot"].shape[0] == 2
    assert batch["structured_present"].dtype == torch.bool


def test_rectified_flow_unet_forward_with_and_without_structured_conditioning() -> None:
    old_model = RectifiedFlowUNet(
        vocab_size=12,
        embed_dim=8,
        base_channels=8,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    structured_model = RectifiedFlowUNet(
        vocab_size=12,
        embed_dim=8,
        base_channels=8,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
        pad_token_id=0,
        structured_vocab_sizes={
            "category_vocab_size": 3,
            "object_vocab_size": 4,
            "base_object_vocab_size": 4,
            "color_vocab_size": 5,
            "material_vocab_size": 2,
            "shape_vocab_size": 2,
            "function_vocab_size": 2,
            "style_vocab_size": 2,
        },
    )
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    caption = torch.tensor([[2, 4, 3, 0], [2, 5, 3, 0]], dtype=torch.long)
    semantic = torch.tensor([[2, 6, 3, 0], [2, 7, 3, 0]], dtype=torch.long)
    structured = {
        "category_id": torch.tensor([1, 2], dtype=torch.long),
        "object_id": torch.tensor([1, 2], dtype=torch.long),
        "base_object_id": torch.tensor([1, 2], dtype=torch.long),
        "primary_color_id": torch.tensor([1, 2], dtype=torch.long),
        "color_multi_hot": torch.zeros(2, 5),
        "material_multi_hot": torch.zeros(2, 2),
        "shape_multi_hot": torch.zeros(2, 2),
        "function_multi_hot": torch.zeros(2, 2),
        "style_multi_hot": torch.zeros(2, 2),
    }

    assert old_model(x, t, caption_tokens=caption, semantic_tokens=semantic).shape == (2, 4, 32, 32)
    assert structured_model(
        x,
        t,
        caption_tokens=caption,
        semantic_tokens=semantic,
        structured_conditioning=structured,
    ).shape == (2, 4, 32, 32)
    assert structured_model(x, t, caption_tokens=caption, semantic_tokens=semantic).shape == (2, 4, 32, 32)
