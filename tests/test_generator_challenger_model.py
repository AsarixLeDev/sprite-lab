from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.conditioning import apply_conditioning_mode
from spritelab.training.generated_qa import qa_generated_sprites
from spritelab.training.generator_challenger import (
    NULL_FIELD_CHOICES,
    V1_1_CFG_BASE_SCALE,
    V1_1_CFG_COLOR_SCALE,
    ChallengerSampleConfig,
    ChallengerTrainConfig,
    RectifiedFlowUNet,
    _apply_cfg_dropout,
    _apply_structured_field_dropout,
    _velocity_loss_components,
    apply_conditioning_field_ablations,
    color_token_ids_for_tokenizer,
    integrate_rectified_flow,
    normalize_export_preset,
    palette_soft_min_auxiliary_loss,
    run_challenger_training,
    run_sample_generator_challenger,
    strip_color_conditioning,
)
from spritelab.training.tokenization import SpriteTextTokenizer


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


def test_challenger_sample_config_factored_cfg_and_null_fields_default_off() -> None:
    config = ChallengerSampleConfig(
        checkpoint=Path("checkpoint.pt"),
        prompts=Path("prompts.jsonl"),
        out_dir=Path("out"),
    )
    assert config.factored_cfg is False
    assert config.cfg_base_scale is None
    assert config.cfg_color_scale is None
    assert config.null_fields == ""


def _tiny_model_and_inputs() -> tuple[RectifiedFlowUNet, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = RectifiedFlowUNet(
        vocab_size=12,
        embed_dim=8,
        base_channels=8,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
        pad_token_id=0,
    ).eval()
    initial = torch.randn(2, 4, 32, 32)
    caption = torch.tensor([[2, 4, 3, 0], [2, 5, 3, 0]], dtype=torch.long)
    semantic = torch.tensor([[2, 6, 3, 0], [2, 7, 3, 0]], dtype=torch.long)
    return model, initial, caption, semantic


def test_integrate_rectified_flow_normal_cfg_path_still_callable_without_factored_args() -> None:
    model, initial, caption, semantic = _tiny_model_and_inputs()
    out = integrate_rectified_flow(
        model,
        initial,
        caption_tokens=caption,
        semantic_tokens=semantic,
        steps=3,
        cfg_scale=2.0,
        pad_token_id=0,
    )
    assert out.shape == (2, 4, 32, 32)
    assert torch.isfinite(out).all()


def test_factored_cfg_reduces_to_uncond_and_cond_at_boundary_scales() -> None:
    """factored CFG's base/color decomposition telescopes back to the plain CFG formula
    at base=color=0 (pure v_uncond) and base=color=1 (pure v_cond), independent of what
    the color-stripped branch predicts -- a strong regression check on the combination
    math itself, not just shapes."""

    model, initial, caption, semantic = _tiny_model_and_inputs()

    uncond_only = integrate_rectified_flow(
        model, initial.clone(), caption_tokens=caption, semantic_tokens=semantic, steps=3, cfg_scale=0.0, pad_token_id=0
    )
    factored_zero = integrate_rectified_flow(
        model,
        initial.clone(),
        caption_tokens=caption,
        semantic_tokens=semantic,
        steps=3,
        cfg_scale=2.0,
        pad_token_id=0,
        factored_cfg=True,
        cfg_base_scale=0.0,
        cfg_color_scale=0.0,
        color_token_ids=(4,),
    )
    assert torch.allclose(uncond_only, factored_zero, atol=1e-6)

    cond_only = integrate_rectified_flow(
        model, initial.clone(), caption_tokens=caption, semantic_tokens=semantic, steps=3, cfg_scale=1.0, pad_token_id=0
    )
    factored_one = integrate_rectified_flow(
        model,
        initial.clone(),
        caption_tokens=caption,
        semantic_tokens=semantic,
        steps=3,
        cfg_scale=2.0,
        pad_token_id=0,
        factored_cfg=True,
        cfg_base_scale=1.0,
        cfg_color_scale=1.0,
        color_token_ids=(4,),
    )
    assert torch.allclose(cond_only, factored_one, atol=1e-6)


def test_factored_cfg_defaults_base_color_scale_from_cfg_scale_at_call_site(tmp_path: Path) -> None:
    """run_sample_generator_challenger resolves cfg_base_scale/cfg_color_scale from
    cfg_scale when the factored-only flags are omitted; exercised end to end via a CPU
    smoke sample so the resolution + wiring both get covered."""

    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = dataset.parent / "factored_run"
    run_challenger_training(
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
    out = dataset.parent / "factored_generated"
    sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=run_dir / "checkpoint_last.pt",
            prompts=_prompts(dataset.parent / "factored_prompts.jsonl"),
            out_dir=out,
            max_samples=2,
            steps=2,
            cfg_scale=2.0,
            max_colors=8,
            device="cpu",
            seed=9,
            batch_size=2,
            factored_cfg=True,
        )
    )
    assert sample_report["sample_count"] == 2
    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["factored_cfg"] is True
    assert rows[0]["cfg_base_scale"] is None
    assert rows[0]["cfg_color_scale"] is None
    assert qa_generated_sprites(out).ok


def test_strip_color_conditioning_removes_color_without_mutating_inputs() -> None:
    tokenizer = SpriteTextTokenizer.build(["red potion", "gold sword", "category item_icon"], max_length=8)
    color_ids = color_token_ids_for_tokenizer(tokenizer)
    assert tokenizer.token_to_id["red"] in color_ids
    assert tokenizer.token_to_id["gold"] in color_ids

    caption = torch.as_tensor([tokenizer.encode("red potion", max_length=8)], dtype=torch.long)
    caption_before = caption.clone()
    structured = {
        "category_id": torch.tensor([1]),
        "primary_color_id": torch.tensor([2]),
        "color_multi_hot": torch.tensor([[1.0, 0.0]]),
    }
    structured_before = {key: value.clone() for key, value in structured.items()}

    stripped = strip_color_conditioning(
        caption_tokens=caption,
        semantic_tokens=None,
        structured_conditioning=structured,
        color_token_ids=color_ids,
        pad_token_id=tokenizer.pad_id,
    )

    decoded = tokenizer.decode(stripped["caption_tokens"][0].tolist())
    assert "red" not in decoded.split()
    assert "potion" in decoded.split()
    assert torch.count_nonzero(stripped["structured_conditioning"]["primary_color_id"]).item() == 0
    assert torch.count_nonzero(stripped["structured_conditioning"]["color_multi_hot"]).item() == 0
    assert torch.equal(stripped["structured_conditioning"]["category_id"], structured["category_id"])

    # inputs must not be mutated
    assert torch.equal(caption, caption_before)
    for key, value in structured.items():
        assert torch.equal(value, structured_before[key])


def test_apply_conditioning_field_ablations_colors_only_affects_color_fields() -> None:
    structured = {
        "category_id": torch.ones(2, dtype=torch.long),
        "object_id": torch.ones(2, dtype=torch.long),
        "primary_color_id": torch.ones(2, dtype=torch.long),
        "color_multi_hot": torch.ones(2, 3),
        "material_multi_hot": torch.ones(2, 2),
    }
    structured_before = {key: value.clone() for key, value in structured.items()}
    caption = torch.ones(2, 4, dtype=torch.long)
    semantic = torch.ones(2, 4, dtype=torch.long)

    result = apply_conditioning_field_ablations(
        caption_tokens=caption,
        semantic_tokens=semantic,
        structured_conditioning=structured,
        fields=("colors",),
        pad_token_id=0,
    )
    out_structured = result["structured_conditioning"]
    assert torch.count_nonzero(out_structured["primary_color_id"]).item() == 0
    assert torch.count_nonzero(out_structured["color_multi_hot"]).item() == 0
    assert torch.count_nonzero(out_structured["category_id"]).item() == 2
    assert torch.count_nonzero(out_structured["object_id"]).item() == 2
    assert torch.count_nonzero(out_structured["material_multi_hot"]).item() == 4
    assert torch.equal(result["caption_tokens"], caption)
    assert torch.equal(result["semantic_tokens"], semantic)
    for key, value in structured.items():
        assert torch.equal(value, structured_before[key])


def test_apply_conditioning_field_ablations_object_id_only_affects_object_id() -> None:
    structured = {
        "category_id": torch.ones(2, dtype=torch.long),
        "object_id": torch.ones(2, dtype=torch.long),
        "base_object_id": torch.ones(2, dtype=torch.long),
        "primary_color_id": torch.ones(2, dtype=torch.long),
        "color_multi_hot": torch.ones(2, 3),
    }
    result = apply_conditioning_field_ablations(
        caption_tokens=torch.ones(2, 4, dtype=torch.long),
        semantic_tokens=torch.ones(2, 4, dtype=torch.long),
        structured_conditioning=structured,
        fields=("object_id",),
        pad_token_id=0,
    )
    out_structured = result["structured_conditioning"]
    assert torch.count_nonzero(out_structured["object_id"]).item() == 0
    assert torch.count_nonzero(out_structured["category_id"]).item() == 2
    assert torch.count_nonzero(out_structured["base_object_id"]).item() == 2
    assert torch.count_nonzero(out_structured["primary_color_id"]).item() == 2
    assert torch.count_nonzero(out_structured["color_multi_hot"]).item() == 6


def test_apply_conditioning_field_ablations_caption_and_structured_whole_null() -> None:
    structured = {"category_id": torch.ones(2, dtype=torch.long), "primary_color_id": torch.ones(2, dtype=torch.long)}
    caption = torch.ones(2, 4, dtype=torch.long)
    semantic = torch.ones(2, 4, dtype=torch.long)
    result = apply_conditioning_field_ablations(
        caption_tokens=caption,
        semantic_tokens=semantic,
        structured_conditioning=structured,
        fields=("caption", "structured"),
        pad_token_id=5,
    )
    assert bool(torch.all(result["caption_tokens"] == 5))
    assert torch.equal(result["semantic_tokens"], semantic)
    for value in result["structured_conditioning"].values():
        assert torch.count_nonzero(value).item() == 0


def test_apply_conditioning_field_ablations_empty_fields_is_noop_identity() -> None:
    caption = torch.ones(2, 4, dtype=torch.long)
    semantic = torch.ones(2, 4, dtype=torch.long)
    structured = {"category_id": torch.ones(2, dtype=torch.long)}
    result = apply_conditioning_field_ablations(
        caption_tokens=caption,
        semantic_tokens=semantic,
        structured_conditioning=structured,
        fields=(),
        pad_token_id=0,
    )
    assert result["caption_tokens"] is caption
    assert result["semantic_tokens"] is semantic
    assert result["structured_conditioning"] is structured


def test_apply_conditioning_field_ablations_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        apply_conditioning_field_ablations(
            caption_tokens=torch.ones(1, 2, dtype=torch.long),
            semantic_tokens=None,
            structured_conditioning=None,
            fields=("not_a_real_field",),
            pad_token_id=0,
        )


@pytest.mark.parametrize("alias", ["v1.1", "v1_1", "phase1_v1_1", "V1.1", " v1_1 "])
def test_normalize_export_preset_v1_1_aliases_all_resolve(alias: str) -> None:
    assert normalize_export_preset(alias) == "v1.1"


@pytest.mark.parametrize("alias", ["v1", "phase1_v1", "V1", " v1 "])
def test_normalize_export_preset_v1_aliases_all_resolve(alias: str) -> None:
    assert normalize_export_preset(alias) == "v1"


def test_normalize_export_preset_unknown_returns_none() -> None:
    assert normalize_export_preset("not_a_preset") is None
    assert normalize_export_preset(None) is None
    assert normalize_export_preset("") is None


def test_v1_1_cfg_scale_constants_match_validated_confirmation() -> None:
    assert V1_1_CFG_BASE_SCALE == pytest.approx(2.5)
    assert V1_1_CFG_COLOR_SCALE == pytest.approx(3.0)


@pytest.mark.parametrize("preset", ["v1.1", "v1_1", "phase1_v1_1"])
def test_sample_manifest_records_v1_1_factored_cfg_metadata(tmp_path: Path, preset: str) -> None:
    """End-to-end (real run_sample_generator_challenger, CPU): the v1.1 preset's
    factored-CFG fields and export_preset must show up on every generated manifest row,
    not just in the in-memory config, so a v1.1 run is unambiguous after the fact."""

    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "v1_1_manifest_run"
    run_challenger_training(
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
    out = tmp_path / "v1_1_manifest_generated"
    sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=run_dir / "checkpoint_last.pt",
            prompts=_prompts(tmp_path / "v1_1_manifest_prompts.jsonl"),
            out_dir=out,
            export_preset=preset,
            max_samples=2,
            steps=2,
            cfg_scale=3.0,
            max_colors=8,
            device="cpu",
            seed=9,
            batch_size=2,
            factored_cfg=True,
            cfg_base_scale=V1_1_CFG_BASE_SCALE,
            cfg_color_scale=V1_1_CFG_COLOR_SCALE,
        )
    )
    assert sample_report["sample_count"] == 2
    rows = [json.loads(line) for line in (out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["export_preset"] == preset
        assert row["factored_cfg"] is True
        assert row["cfg_base_scale"] == pytest.approx(2.5)
        assert row["cfg_color_scale"] == pytest.approx(3.0)
    assert qa_generated_sprites(out).ok

    # v1 (no factored CFG) must remain unaffected by these new fields' presence/values.
    v1_out = tmp_path / "v1_manifest_generated"
    run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=run_dir / "checkpoint_last.pt",
            prompts=_prompts(tmp_path / "v1_manifest_prompts.jsonl"),
            out_dir=v1_out,
            export_preset="v1",
            max_samples=2,
            steps=2,
            cfg_scale=3.0,
            max_colors=8,
            device="cpu",
            seed=9,
            batch_size=2,
        )
    )
    v1_rows = [
        json.loads(line) for line in (v1_out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in v1_rows:
        assert row["export_preset"] == "v1"
        assert row["factored_cfg"] is False
        assert row["cfg_base_scale"] is None
        assert row["cfg_color_scale"] is None


def test_null_field_choices_are_stable() -> None:
    assert set(NULL_FIELD_CHOICES) == {
        "caption",
        "semantic",
        "category",
        "object_id",
        "base_object",
        "colors",
        "materials",
        "shapes",
        "function",
        "style",
        "structured",
    }


# ── v2 Phase 1 conditioning architecture tests ──────────────────────────────


def test_model_film_off_is_default_and_shapes_match() -> None:
    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=0,
        film_conditioning=False,
    )
    assert model.film_conditioning is False
    cfg = model.config()
    assert cfg["film_conditioning"] is False
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    caps = torch.randint(0, 64, (2, 8))
    out = model(x, t, caption_tokens=caps)
    assert out.shape == (2, 4, 32, 32)


def test_model_film_on_shapes_match() -> None:
    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=0,
        film_conditioning=True,
    )
    assert model.film_conditioning is True
    cfg = model.config()
    assert cfg["film_conditioning"] is True
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    caps = torch.randint(0, 64, (2, 8))
    out = model(x, t, caption_tokens=caps)
    assert out.shape == (2, 4, 32, 32)


def test_model_bottleneck_attention_on_shapes_match() -> None:
    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
        pad_token_id=0,
        bottleneck_attention=True,
    )
    assert model.bottleneck_attention is True
    cfg = model.config()
    assert cfg["bottleneck_attention"] is True
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    caps = torch.randint(0, 64, (2, 8))
    out = model(x, t, caption_tokens=caps)
    assert out.shape == (2, 4, 32, 32)


def test_model_film_and_attention_together() -> None:
    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
        pad_token_id=0,
        film_conditioning=True,
        bottleneck_attention=True,
    )
    assert model.film_conditioning is True
    assert model.bottleneck_attention is True
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    caps = torch.randint(0, 64, (2, 8))
    out = model(x, t, caption_tokens=caps)
    assert out.shape == (2, 4, 32, 32)


def test_old_default_config_has_features_off() -> None:
    config = ChallengerTrainConfig(
        dataset_dir=Path("ds"),
        training_manifest=Path("m.jsonl"),
        out_dir=Path("out"),
    )
    assert config.film_conditioning is False
    assert config.bottleneck_attention is False
    assert config.structured_field_dropout_rates is None


def test_per_group_dropout_accepts_valid_string() -> None:
    from spritelab.training.cli import _parse_dropout_rates

    rates = _parse_dropout_rates("category=0.10,object_id=0.35,colors=0.15")
    assert rates == {"category": 0.10, "object_id": 0.35, "colors": 0.15}


def test_per_group_dropout_rejects_unknown_group() -> None:
    from spritelab.training.cli import _parse_dropout_rates

    with pytest.raises(ValueError, match="Unknown"):
        _parse_dropout_rates("unknown=0.5")


def test_per_group_dropout_rejects_invalid_rate() -> None:
    from spritelab.training.cli import _parse_dropout_rates

    with pytest.raises(ValueError, match="must be in"):
        _parse_dropout_rates("category=1.5")
    with pytest.raises(ValueError, match="must be in"):
        _parse_dropout_rates("category=-0.5")


def test_per_group_dropout_returns_none_for_empty() -> None:
    from spritelab.training.cli import _parse_dropout_rates

    assert _parse_dropout_rates(None) is None
    assert _parse_dropout_rates("") is None


def test_cli_accepts_film_conditioning_flag() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--film-conditioning", action="store_true", default=False)
    parsed = parser.parse_args(["--film-conditioning"])
    assert parsed.film_conditioning is True
    parsed2 = parser.parse_args([])
    assert parsed2.film_conditioning is False


def test_cli_accepts_bottleneck_attention_flag() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bottleneck-attention", action="store_true", default=False)
    parsed = parser.parse_args(["--bottleneck-attention"])
    assert parsed.bottleneck_attention is True
    parsed2 = parser.parse_args([])
    assert parsed2.bottleneck_attention is False


def test_cli_accepts_dropout_rates_flag() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-field-dropout-rates", default=None)
    parsed = parser.parse_args(["--structured-field-dropout-rates", "category=0.10,colors=0.15"])
    assert parsed.structured_field_dropout_rates == "category=0.10,colors=0.15"
    parsed2 = parser.parse_args([])
    assert parsed2.structured_field_dropout_rates is None


# ── v2 Phase 2: palette/index head tests ────────────────────────────────────


def _dummy_batch(bs: int = 2) -> dict[str, Any]:
    """Create a minimal batch with palette/index_map tensors."""
    K = 16
    H = W = 32
    return {
        "rgba": torch.rand(bs, 4, H, W),
        "palette": torch.rand(bs, K, 3),
        "palette_mask": torch.zeros(bs, K, dtype=torch.bool),
        "index_map": torch.zeros(bs, H, W, dtype=torch.long),
        "caption_tokens": torch.randint(0, 64, (bs, 8)),
        "semantic_tokens": torch.randint(0, 64, (bs, 8)),
    }


def test_model_forward_returns_velocity_only_by_default() -> None:
    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    x = torch.randn(2, 4, 32, 32)
    out = model(x, torch.rand(2), caption_tokens=torch.randint(0, 64, (2, 8)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 32, 32)


def test_model_forward_return_aux_returns_dict() -> None:
    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    x = torch.randn(2, 4, 32, 32)
    out = model(
        x,
        torch.rand(2),
        caption_tokens=torch.randint(0, 64, (2, 8)),
        return_aux=True,
    )
    assert isinstance(out, dict)
    assert "velocity" in out
    assert out["velocity"].shape == (2, 4, 32, 32)
    assert "palette_rgb" in out
    assert out["palette_rgb"].shape == (2, 16, 3)
    assert "palette_presence_logits" in out
    assert out["palette_presence_logits"].shape == (2, 16)
    assert "index_logits" in out
    assert out["index_logits"].shape == (2, 16, 32, 32)


def test_default_config_has_head_loss_weights_zero() -> None:
    config = ChallengerTrainConfig(
        dataset_dir=Path("ds"),
        training_manifest=Path("m.jsonl"),
        out_dir=Path("out"),
    )
    assert config.index_head_loss_weight == 0.0
    assert config.palette_head_loss_weight == 0.0
    assert config.palette_presence_loss_weight == 0.0
    assert config.index_head_warmup_steps == 0


def test_loss_skips_heads_when_weights_zero() -> None:
    """When all head weights are zero, only velocity + palette_aux losses appear."""
    from spritelab.training.generator_challenger import rectified_flow_loss

    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    batch = _dummy_batch(bs=2)
    losses = rectified_flow_loss(
        model,
        batch,
        conditioning_mode="caption_semantic",
        cfg_dropout=0.0,
        pad_token_id=0,
        index_head_loss_weight=0.0,
        palette_head_loss_weight=0.0,
        palette_presence_loss_weight=0.0,
    )
    assert "loss_velocity" in losses
    assert "loss_palette_aux" in losses
    assert losses["index_head_active"] is False


def test_palette_head_loss_finite() -> None:
    from spritelab.training.generator_challenger import (
        rectified_flow_loss,
    )

    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    K = 16
    batch = _dummy_batch(bs=2)
    batch["palette_mask"] = torch.ones(2, K, dtype=torch.bool)
    losses = rectified_flow_loss(
        model,
        batch,
        conditioning_mode="caption_semantic",
        cfg_dropout=0.0,
        pad_token_id=0,
        palette_head_loss_weight=0.1,
        palette_presence_loss_weight=0.05,
        index_head_loss_weight=0.0,
    )
    assert "loss_palette_head" in losses
    assert "loss_palette_presence" in losses
    assert torch.isfinite(losses["loss_palette_head"])
    assert torch.isfinite(losses["loss_palette_presence"])
    # Total loss should be finite
    assert torch.isfinite(losses["loss"])


def test_index_head_loss_finite() -> None:
    from spritelab.training.generator_challenger import (
        rectified_flow_loss,
    )

    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    batch = _dummy_batch(bs=2)
    batch["index_map"] = torch.randint(0, 16, (2, 32, 32))
    # Set some pixels as invisible
    batch["rgba"][:, 3:4] = 0.0
    losses = rectified_flow_loss(
        model,
        batch,
        conditioning_mode="caption_semantic",
        cfg_dropout=0.0,
        pad_token_id=0,
        index_head_loss_weight=0.25,
        palette_head_loss_weight=0.0,
        palette_presence_loss_weight=0.0,
        index_head_warmup_steps=0,
    )
    assert "loss_index_head" in losses
    assert torch.isfinite(losses["loss_index_head"])
    assert losses["index_head_active"] is True


def test_index_head_inactive_before_warmup() -> None:
    from spritelab.training.generator_challenger import (
        rectified_flow_loss,
    )

    model = RectifiedFlowUNet(
        vocab_size=64,
        embed_dim=32,
        base_channels=32,
        channel_mults=(1,),
        res_blocks_per_level=1,
        pad_token_id=0,
    )
    batch = _dummy_batch(bs=2)
    losses = rectified_flow_loss(
        model,
        batch,
        conditioning_mode="caption_semantic",
        cfg_dropout=0.0,
        pad_token_id=0,
        index_head_loss_weight=0.25,
        palette_head_loss_weight=0.0,
        palette_presence_loss_weight=0.0,
        global_step=50,
        index_head_warmup_steps=100,
    )
    assert losses["index_head_active"] is False


# ── v2 Phase 2: palette/index head inspection tests ──────────────────────────


def test_cli_accepts_inspect_palette_index_heads() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="test")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    inspect = sub.add_parser("inspect-palette-index-heads")
    inspect.add_argument("--checkpoint", required=True, type=Path)
    inspect.add_argument("--dataset", required=True, type=Path)
    inspect.add_argument("--training-manifest", required=True, type=Path)
    inspect.add_argument("--out", required=True, type=Path)
    inspect.add_argument("--device", default="cpu")
    inspect.add_argument("--batch-size", type=int, default=32)
    inspect.add_argument("--max-batches", type=int, default=32)
    inspect.add_argument("--cudnn-benchmark", action="store_true", default=False)
    inspect.add_argument("--tf32", action="store_true", default=False)

    parsed = parser.parse_args(
        [
            "inspect-palette-index-heads",
            "--checkpoint",
            "ckpt.pt",
            "--dataset",
            "ds",
            "--training-manifest",
            "manifest.jsonl",
            "--out",
            "out",
        ]
    )
    assert parsed.subcommand == "inspect-palette-index-heads"
    assert parsed.device == "cpu"
    assert parsed.cudnn_benchmark is False
    assert parsed.tf32 is False


def test_cli_accepts_speed_flags_for_inspect() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="test")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    inspect = sub.add_parser("inspect-palette-index-heads")
    inspect.add_argument("--checkpoint", required=True, type=Path)
    inspect.add_argument("--dataset", required=True, type=Path)
    inspect.add_argument("--training-manifest", required=True, type=Path)
    inspect.add_argument("--out", required=True, type=Path)
    inspect.add_argument("--device", default="cpu")
    inspect.add_argument("--batch-size", type=int, default=32)
    inspect.add_argument("--max-batches", type=int, default=32)
    inspect.add_argument("--cudnn-benchmark", action="store_true", default=False)
    inspect.add_argument("--tf32", action="store_true", default=False)

    parsed = parser.parse_args(
        [
            "inspect-palette-index-heads",
            "--checkpoint",
            "ckpt.pt",
            "--dataset",
            "ds",
            "--training-manifest",
            "manifest.jsonl",
            "--out",
            "out",
            "--cudnn-benchmark",
            "--tf32",
        ]
    )
    assert parsed.cudnn_benchmark is True
    assert parsed.tf32 is True


def test_inspect_index_metrics_visible_pixel_accuracy() -> None:
    from spritelab.training.palette_index_head_inspect import compute_index_head_metrics

    B, C, H, W = 2, 16, 32, 32
    logits = torch.zeros(B, C, H, W)
    target_indices = torch.tensor([0, 1]).view(B, 1, 1).expand(B, H, W)
    for b in range(B):
        logits[b, target_indices[b, 0, 0], :, :] = 10.0

    alpha = torch.ones(B, 1, H, W) * 0.8

    metrics = compute_index_head_metrics(logits, target_indices, alpha)
    assert metrics.visible_pixel_accuracy == pytest.approx(1.0)
    assert metrics.top2_accuracy == pytest.approx(1.0)
    assert metrics.invalid_target_count == 0
    assert metrics.visible_pixel_count == B * H * W


def test_inspect_index_metrics_ignore_invisible() -> None:
    from spritelab.training.palette_index_head_inspect import compute_index_head_metrics

    B, C, H, W = 2, 16, 32, 32
    logits = torch.zeros(B, C, H, W)
    target_indices = torch.tensor([3, 5]).view(B, 1, 1).expand(B, H, W)

    alpha = torch.zeros(B, 1, H, W)

    metrics = compute_index_head_metrics(logits, target_indices, alpha)
    assert metrics.visible_pixel_count == 0
    assert metrics.ignored_pixel_count == B * H * W


def test_inspect_palette_rgb_metrics_finite() -> None:
    from spritelab.training.palette_index_head_inspect import compute_palette_rgb_metrics

    B, K = 4, 16
    predicted = torch.randn(B, K, 3)
    target = torch.rand(B, K, 3)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, 0] = False
    mask[1, 0] = False

    metrics = compute_palette_rgb_metrics(predicted, target, mask)
    assert metrics.mse > 0
    assert metrics.mae > 0
    assert torch.tensor(metrics.mse).isfinite()
    assert torch.tensor(metrics.mae).isfinite()
    assert metrics.active_slot_count_mean > 0


def test_inspect_palette_rgb_zero_for_empty_mask() -> None:
    from spritelab.training.palette_index_head_inspect import compute_palette_rgb_metrics

    B, K = 4, 16
    predicted = torch.randn(B, K, 3)
    target = torch.rand(B, K, 3)
    mask = torch.zeros(B, K, dtype=torch.bool)

    metrics = compute_palette_rgb_metrics(predicted, target, mask)
    assert metrics.mse == 0.0
    assert metrics.mae == 0.0
    assert metrics.active_slot_count_mean == 0.0


def test_inspect_presence_metrics_finite() -> None:
    from spritelab.training.palette_index_head_inspect import compute_palette_presence_metrics

    B, K = 4, 16
    logits = torch.randn(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, 0] = False
    mask[0, 1] = False

    metrics = compute_palette_presence_metrics(logits, mask)
    assert 0.0 <= metrics.bce
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert metrics.predicted_active_mean >= 0
    assert metrics.target_active_mean >= 0


def test_inspect_presence_perfect_prediction() -> None:
    from spritelab.training.palette_index_head_inspect import compute_palette_presence_metrics

    B, K = 2, 16
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :4] = True
    logits = torch.where(mask, torch.tensor(5.0), torch.tensor(-5.0))

    metrics = compute_palette_presence_metrics(logits, mask)
    assert metrics.bce < 0.01
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.predicted_active_mean == pytest.approx(metrics.target_active_mean)


def test_inspect_report_writes_json_and_markdown(tmp_path: Path) -> None:
    from spritelab.training.palette_index_head_inspect import (
        PaletteIndexHeadInspectConfig,
        _BatchMetrics,
        _IndexMetrics,
        _PalettePresenceMetrics,
        _PaletteRGBMetrics,
        write_inspect_report,
    )

    batch = _BatchMetrics(
        index=_IndexMetrics(
            visible_pixel_accuracy=0.95,
            top2_accuracy=0.98,
            cross_entropy=0.1,
            visible_pixel_count=1000,
        ),
        palette_rgb=_PaletteRGBMetrics(mse=0.001, mae=0.02, active_slot_count_mean=4.5),
        palette_presence=_PalettePresenceMetrics(
            bce=0.3,
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1=0.86,
            predicted_active_mean=5.0,
            target_active_mean=4.5,
            false_positives_rate=0.05,
            false_negatives_rate=0.1,
        ),
    )

    config = PaletteIndexHeadInspectConfig(
        checkpoint=Path("ckpt.pt"),
        dataset=Path("ds"),
        training_manifest=Path("manifest.jsonl"),
        out=tmp_path / "inspect_out",
    )

    json_path = write_inspect_report([batch], config, config.out)
    assert json_path.exists()
    assert json_path.suffix == ".json"

    md_path = config.out / "palette_index_head_inspect.md"
    assert md_path.exists()

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["aggregate"]["index_head"]["visible_pixel_accuracy"] == pytest.approx(0.95)

    md_text = md_path.read_text(encoding="utf-8")
    assert "0.9500" in md_text or "0.95" in md_text


def test_inspect_cmd_fails_for_non_challenger_checkpoint(tmp_path: Path) -> None:
    from spritelab.training.palette_index_head_inspect import (
        PaletteIndexHeadInspectConfig,
        run_inspect_palette_index_heads,
    )

    ckpt_path = tmp_path / "fake.pt"
    torch.save({"model_type": "other_type"}, ckpt_path)

    config = PaletteIndexHeadInspectConfig(
        checkpoint=ckpt_path,
        dataset=tmp_path / "ds",
        training_manifest=tmp_path / "manifest.jsonl",
        out=tmp_path / "out",
        device="cpu",
        batch_size=2,
        max_batches=1,
    )

    with pytest.raises(SystemExit, match="generator_challenger"):
        run_inspect_palette_index_heads(config)


def test_inspect_unsupported_for_model_without_heads(tmp_path: Path) -> None:
    class OldModel(torch.nn.Module):
        def config(self):
            return {
                "vocab_size": 64,
                "embed_dim": 32,
                "base_channels": 32,
                "channel_mults": [1],
                "res_blocks_per_level": 1,
                "pad_token_id": 0,
                "structured_vocab_sizes": None,
                "film_conditioning": False,
                "bottleneck_attention": False,
            }

        def forward(
            self, x, t, caption_tokens=None, semantic_tokens=None, structured_conditioning=None, return_aux=False
        ):
            return x[:, :4]

    old = OldModel()
    ckpt_path = tmp_path / "old_model.pt"
    torch.save(
        {
            "model_type": "generator_challenger",
            "architecture": "rectified_flow",
            "model_state_dict": old.state_dict(),
            "model_config": old.config(),
            "vocab": {},
            "conditioning_mode": "caption_semantic",
        },
        ckpt_path,
    )

    from spritelab.training.palette_index_head_inspect import (
        _has_palette_index_heads,
    )

    assert not _has_palette_index_heads(old)
