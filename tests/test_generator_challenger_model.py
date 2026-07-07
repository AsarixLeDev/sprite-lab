from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset

from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.conditioning import apply_conditioning_mode
from spritelab.training.generator_challenger import (
    ChallengerSampleConfig,
    ChallengerTrainConfig,
    RectifiedFlowUNet,
    _apply_cfg_dropout,
    _apply_structured_field_dropout,
    _velocity_loss_components,
    palette_soft_min_auxiliary_loss,
    run_challenger_training,
    run_sample_generator_challenger,
)
from spritelab.training.generated_qa import qa_generated_sprites


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


def _prompts(path: Path) -> Path:
    rows = [
        {"prompt_id": "p0", "prompt": "red potion", "category": "seen_object"},
        {"prompt_id": "p1", "prompt": "gold sword", "category": "seen_object"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _structured_prompts(path: Path) -> Path:
    rows = [
        {
            "prompt_id": "p0",
            "prompt": "red potion",
            "target_sprite_id": "p0",
            "category": "item_icon",
            "object_name": "potion",
            "base_object": "potion",
            "colors": ["red"],
            "conditioning": {
                "semantic_v3": {
                    "category": "item_icon",
                    "object_name": "potion",
                    "open_name": "potion",
                    "base_object": "potion",
                    "attributes": {"colors": ["red"], "materials": [], "shapes": [], "function": []},
                }
            },
        },
        {
            "prompt_id": "p1",
            "prompt": "gold sword",
            "target_sprite_id": "p1",
            "category": "weapon",
            "object_name": "sword",
            "base_object": "sword",
            "colors": ["gold"],
            "conditioning": {
                "semantic_v3": {
                    "category": "weapon",
                    "object_name": "sword",
                    "open_name": "sword",
                    "base_object": "sword",
                    "attributes": {"colors": ["gold"], "materials": ["metal"], "shapes": [], "function": ["attack"]},
                }
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_challenger_forward_shape_and_conditioning_modes() -> None:
    model = RectifiedFlowUNet(
        vocab_size=12,
        embed_dim=8,
        base_channels=8,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    caption = torch.tensor([[2, 4, 3, 0], [2, 5, 3, 0]], dtype=torch.long)
    semantic = torch.tensor([[2, 6, 3, 0], [2, 7, 3, 0]], dtype=torch.long)
    for mode in ("caption", "semantic", "caption_semantic", "none"):
        inputs = apply_conditioning_mode(
            caption_tokens=caption,
            semantic_tokens=semantic,
            mode=mode,
            pad_token_id=0,
        )
        out = model(x, t, caption_tokens=inputs["caption_tokens"], semantic_tokens=inputs["semantic_tokens"])
        assert out.shape == (2, 4, 32, 32)


def test_cfg_dropout_can_make_unconditional_branch() -> None:
    caption = torch.tensor([[2, 4, 3], [2, 5, 3]], dtype=torch.long)
    semantic = torch.tensor([[2, 6, 3], [2, 7, 3]], dtype=torch.long)
    dropped = _apply_cfg_dropout(
        {"caption_tokens": caption, "semantic_tokens": semantic},
        dropout=1.0,
        pad_token_id=0,
    )
    assert torch.count_nonzero(dropped["caption_tokens"]).item() == 0
    assert torch.count_nonzero(dropped["semantic_tokens"]).item() == 0


def _structured_inputs(batch_size: int = 4) -> dict[str, object]:
    caption = torch.ones(batch_size, 3, dtype=torch.long)
    structured = {
        "category_id": torch.ones(batch_size, dtype=torch.long),
        "object_id": torch.ones(batch_size, dtype=torch.long),
        "base_object_id": torch.ones(batch_size, dtype=torch.long),
        "primary_color_id": torch.ones(batch_size, dtype=torch.long),
        "color_multi_hot": torch.ones(batch_size, 3),
        "material_multi_hot": torch.ones(batch_size, 2),
        "shape_multi_hot": torch.ones(batch_size, 2),
        "function_multi_hot": torch.ones(batch_size, 2),
        "style_multi_hot": torch.ones(batch_size, 2),
    }
    return {"caption_tokens": caption, "semantic_tokens": caption.clone(), "structured_conditioning": structured}


def test_structured_field_dropout_only_applies_in_train_mode() -> None:
    inputs = _structured_inputs()
    unchanged = _apply_structured_field_dropout(inputs, dropout=1.0, training=False)
    assert unchanged is inputs

    dropped = _apply_structured_field_dropout(inputs, dropout=1.0, training=True)
    structured = dropped["structured_conditioning"]
    assert torch.count_nonzero(structured["category_id"]).item() == 0
    assert torch.count_nonzero(structured["color_multi_hot"]).item() == 0


def test_structured_field_dropout_uses_independent_field_masks() -> None:
    torch.manual_seed(0)
    inputs = _structured_inputs(batch_size=64)
    dropped = _apply_structured_field_dropout(inputs, dropout=0.5, training=True)
    structured = dropped["structured_conditioning"]
    category_mask = structured["category_id"].eq(0)
    object_mask = structured["object_id"].eq(0)
    color_mask = structured["color_multi_hot"].sum(dim=1).eq(0)

    assert category_mask.any()
    assert object_mask.any()
    assert color_mask.any()
    assert not torch.equal(category_mask, object_mask)
    assert torch.equal(structured["primary_color_id"].eq(0), color_mask)


def test_velocity_loss_defaults_match_global_mse_and_keep_alpha_stable() -> None:
    pred = torch.zeros(1, 4, 2, 2)
    velocity = torch.ones(1, 4, 2, 2)
    target_rgba = torch.zeros(1, 4, 2, 2)
    target_rgba[:, 3:, :, :1] = 1.0

    default = _velocity_loss_components(pred, velocity, target_rgba=target_rgba)
    weighted = _velocity_loss_components(
        pred,
        velocity,
        target_rgba=target_rgba,
        foreground_rgb_loss_weight=2.0,
        background_rgb_loss_weight=0.25,
    )

    assert torch.isclose(default["loss_velocity"], torch.mean((pred - velocity) ** 2))
    assert torch.isclose(default["loss_alpha"], weighted["loss_alpha"])
    assert weighted["loss_rgb"] != default["loss_rgb"]


def test_palette_soft_min_loss_handles_masks_and_rewards_palette_matches() -> None:
    target = torch.zeros(1, 4, 2, 2)
    target[:, 3] = 1.0
    palette = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    palette_mask = torch.tensor([[True, True]])
    red_x1 = torch.zeros(1, 4, 2, 2)
    red_x1[:, 0] = 1.0
    red_x1[:, 1] = -1.0
    red_x1[:, 2] = -1.0
    blue_x1 = red_x1.clone()
    blue_x1[:, 0] = -1.0
    blue_x1[:, 2] = 1.0

    red_loss = palette_soft_min_auxiliary_loss(
        x1_hat=red_x1,
        target_rgba=target,
        palette=palette,
        palette_mask=palette_mask,
    )
    blue_loss = palette_soft_min_auxiliary_loss(
        x1_hat=blue_x1,
        target_rgba=target,
        palette=palette,
        palette_mask=palette_mask,
    )
    empty_loss = palette_soft_min_auxiliary_loss(
        x1_hat=blue_x1,
        target_rgba=target,
        palette=palette,
        palette_mask=torch.zeros_like(palette_mask),
    )
    transparent_loss = palette_soft_min_auxiliary_loss(
        x1_hat=blue_x1,
        target_rgba=torch.zeros_like(target),
        palette=palette,
        palette_mask=palette_mask,
    )

    assert red_loss < blue_loss
    assert float(empty_loss) == 0.0
    assert float(transparent_loss) == 0.0


def test_challenger_cpu_smoke_train_checkpoint_sample_and_qa(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "challenger_run"
    report = run_challenger_training(
        ChallengerTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=run_dir,
            batch_size=2,
            max_steps=1,
            device="cpu",
            seed=7,
            base_channels=8,
            channel_mults="1,2",
            res_blocks_per_level=1,
            embed_dim=8,
            sample_every=0,
            save_every=0,
            validation_mode="none",
        )
    )
    assert report["steps_completed"] == 1
    ckpt = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    assert ckpt["model_type"] == "generator_challenger"
    assert ckpt["architecture"] == "rectified_flow"
    assert ckpt["conditioning_mode"] == "caption_semantic"
    assert ckpt["train_config"]["seed"] == 7

    out = tmp_path / "generated"
    sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=run_dir / "checkpoint_last.pt",
            prompts=_prompts(tmp_path / "prompts.jsonl"),
            out_dir=out,
            max_samples=2,
            steps=2,
            cfg_scale=1.0,
            max_colors=8,
            device="cpu",
            seed=9,
            batch_size=2,
        )
    )
    assert sample_report["sample_count"] == 2
    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (out / rows[0]["paths"]["indexed_png"]).is_file()
    qa = qa_generated_sprites(out)
    assert qa.ok
    assert (out / "contact_sheet_labels.json").is_file()

    ema_ckpt = torch.load(run_dir / "checkpoint_last_ema.pt", map_location="cpu", weights_only=False)
    assert ema_ckpt["ema_weights"] is True
    assert ema_ckpt["ema_decay"] == pytest.approx(0.999)
    ema_out = tmp_path / "generated_ema"
    ema_sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=run_dir / "checkpoint_last_ema.pt",
            prompts=_prompts(tmp_path / "prompts_ema.jsonl"),
            out_dir=ema_out,
            max_samples=2,
            steps=2,
            cfg_scale=1.0,
            max_colors=8,
            device="cpu",
            seed=9,
            batch_size=2,
        )
    )
    assert ema_sample_report["sample_count"] == 2


def test_challenger_ema_decay_zero_disables_ema_checkpoint(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "no_ema_run"
    report = run_challenger_training(
        ChallengerTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=run_dir,
            batch_size=2,
            max_steps=1,
            device="cpu",
            seed=8,
            base_channels=8,
            channel_mults="1,2",
            res_blocks_per_level=1,
            embed_dim=8,
            sample_every=0,
            save_every=0,
            validation_mode="none",
            ema_decay=0.0,
        )
    )
    assert report["ema_enabled"] is False
    assert (run_dir / "checkpoint_last.pt").is_file()
    assert not (run_dir / "checkpoint_last_ema.pt").exists()


def test_challenger_interval_checkpoint_steps_save_normal_and_ema(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "interval_run"
    report = run_challenger_training(
        ChallengerTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=run_dir,
            batch_size=2,
            max_steps=2,
            device="cpu",
            seed=18,
            base_channels=8,
            channel_mults="1,2",
            res_blocks_per_level=1,
            embed_dim=8,
            sample_every=0,
            save_every=0,
            checkpoint_steps=(1, 2),
            validation_mode="none",
            ema_decay=0.9,
        )
    )

    assert report["checkpoint_steps"] == [1, 2]
    assert (run_dir / "checkpoint_step_000001.pt").is_file()
    assert (run_dir / "checkpoint_step_000001_ema.pt").is_file()
    assert (run_dir / "checkpoint_step_000002.pt").is_file()
    assert (run_dir / "checkpoint_step_000002_ema.pt").is_file()
    step_ema = torch.load(run_dir / "checkpoint_step_000002_ema.pt", map_location="cpu", weights_only=False)
    assert step_ema["ema_weights"] is True
    assert step_ema["checkpoint_variant"] == "step_ema"


def test_structured_challenger_cpu_smoke_train_checkpoint_sample_and_qa(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "structured_challenger_run"
    report = run_challenger_training(
        ChallengerTrainConfig(
            dataset_dir=dataset,
            training_manifest=manifest,
            out_dir=run_dir,
            batch_size=2,
            max_steps=1,
            device="cpu",
            seed=17,
            base_channels=8,
            channel_mults="1,2",
            res_blocks_per_level=1,
            embed_dim=8,
            sample_every=0,
            save_every=0,
            validation_mode="none",
            conditioning_mode="caption_semantic_structured",
            structured_field_dropout=0.1,
            palette_loss_weight=0.1,
            foreground_rgb_loss_weight=2.0,
            background_rgb_loss_weight=0.25,
        )
    )
    assert report["steps_completed"] == 1
    assert report["conditioning_mode"] == "caption_semantic_structured"
    assert report["structured_field_dropout"] == 0.1
    assert report["palette_loss_weight"] == 0.1
    assert "loss_palette_aux" in report["last_step_loss_components"]
    assert report["structured_vocab_sizes"]["category_vocab_size"] > 1
    ckpt = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    assert ckpt["conditioning_mode"] == "caption_semantic_structured"
    assert ckpt["structured_vocab_sizes"]["category_vocab_size"] > 1

    out = tmp_path / "structured_generated"
    sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=run_dir / "checkpoint_last.pt",
            prompts=_structured_prompts(tmp_path / "structured_prompts.jsonl"),
            out_dir=out,
            max_samples=2,
            steps=2,
            cfg_scale=1.0,
            max_colors=8,
            device="cpu",
            seed=19,
            batch_size=2,
        )
    )
    assert sample_report["sample_count"] == 2
    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["conditioning_mode"] == "caption_semantic_structured"
    assert (out / rows[0]["paths"]["indexed_png"]).is_file()
    assert qa_generated_sprites(out).ok
