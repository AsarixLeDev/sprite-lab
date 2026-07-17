from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spritelab.training.experiment_system import (
    CONDITIONING_SCHEMA_VERSION,
    IncompatibleResumeError,
    build_experiment_manifest,
    canonical_json,
    capture_rng_state,
    conditioning_schema,
    restore_rng_state,
    stable_hash,
    validate_ablation_config,
    validate_inference_parity,
    validate_resume_compatibility,
    windows_command,
)
from spritelab.training.generator_challenger import (
    ChallengerSampleConfig,
    ChallengerTrainConfig,
    RectifiedFlowUNet,
    _apply_cfg_dropout,
    _init_ema_state,
    _save_checkpoint,
    rectified_flow_loss,
    run_challenger_training,
    run_sample_generator_challenger,
)
from spritelab.training.structured_conditioning import (
    StructuredConditioningVocab,
    structured_vocab_from_checkpoint,
)
from spritelab.training.tokenization import SpriteTextTokenizer


def _config() -> dict:
    return {
        "name": "test",
        "ablation": "baseline",
        "dataset": {},
        "model": {"base_channels": 8, "channel_mults": [1, 2], "embed_dim": 8},
        "conditioning": {"mode": "caption_semantic", "cfg_dropout": 0.1},
        "loss": {"name": "masked_velocity_mse"},
        "optimizer": {"name": "adamw", "schedule": "none", "warmup_steps": 0},
        "augmentation": {},
        "seeds": {"training": 1},
        "runtime": {
            "precision": "fp32",
            "micro_batch_size": 1,
            "batch_size": 1,
            "effective_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sample_every": 1,
            "save_every": 1,
            "max_steps": 2,
        },
        "sampling": {"cfg_scale": 3.0, "steps": 30},
        "ema": {"enabled": True, "decay": 0.999},
        "timestep_sampling": {"strategy": "uniform"},
        "noise_schedule": "rectified_flow_linear_path",
        "self_conditioning": False,
    }


def _manifest(tmp_path: Path) -> dict:
    source = tmp_path / "manifest.jsonl"
    source.write_text('{"sprite_id":"a","split":"train"}\n', encoding="utf-8")
    tokenizer = SpriteTextTokenizer.build(["red sword"])
    return build_experiment_manifest(
        _config(), dataset_manifest=source, tokenizer=tokenizer.to_json_dict(), repo=tmp_path
    )


def _model() -> RectifiedFlowUNet:
    return RectifiedFlowUNet(
        vocab_size=8,
        embed_dim=8,
        base_channels=8,
        channel_mults=(1, 2),
        res_blocks_per_level=1,
    )


def test_manifest_hash_and_config_serialization_are_deterministic(tmp_path: Path) -> None:
    first = _manifest(tmp_path)
    second = _manifest(tmp_path)
    assert first["experiment_hash"] == second["experiment_hash"]
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert stable_hash({"a": [1, 2]}) == stable_hash({"a": [1, 2]})


def test_conditioning_vocabulary_schema_and_order_are_stable() -> None:
    tokenizer = SpriteTextTokenizer.build(["sword red", "red shield"])
    schema = conditioning_schema(mode="caption_semantic", tokenizer=tokenizer.to_json_dict())
    assert schema["version"] == CONDITIONING_SCHEMA_VERSION
    assert schema["field_order"] == ["caption", "semantic"]
    assert tokenizer.token_to_id == SpriteTextTokenizer.build(["red shield", "sword red"]).token_to_id
    with pytest.raises(ValueError, match="unsupported"):
        StructuredConditioningVocab.from_json_dict({"schema_version": "future_v9"})


def test_legacy_checkpoint_structured_vocabulary_uses_compatibility_adapter() -> None:
    checkpoint = {
        "structured_conditioning_vocab": {
            "schema_version": "structured_conditioning_vocab_v1",
            "categories": ["<unk>", "weapon"],
            "objects": ["<unk>", "sword"],
        }
    }
    vocab = structured_vocab_from_checkpoint(checkpoint, allow_schema_v1_adapter=True)
    assert vocab is not None
    assert vocab.schema_version == "structured_conditioning_vocab_v1"
    assert vocab.categories == ("<unk>", "weapon")


def test_cfg_dropout_behavior_is_joint_and_null() -> None:
    inputs = {
        "caption_tokens": torch.ones(4, 3, dtype=torch.long),
        "semantic_tokens": torch.ones(4, 3, dtype=torch.long),
    }
    dropped = _apply_cfg_dropout(inputs, dropout=1.0, pad_token_id=0)
    assert torch.count_nonzero(dropped["caption_tokens"]) == 0
    assert torch.count_nonzero(dropped["semantic_tokens"]) == 0
    assert dropped["cfg_dropout_fraction"] == 1.0


def test_auxiliary_loss_toggles_and_smoke_forward_backward() -> None:
    model = _model()
    batch = {
        "rgba": torch.rand(2, 4, 32, 32),
        "caption_tokens": torch.ones(2, 4, dtype=torch.long),
        "semantic_tokens": torch.ones(2, 4, dtype=torch.long),
        "palette": torch.rand(2, 16, 3),
        "palette_mask": torch.ones(2, 16),
        "indices": torch.zeros(2, 32, 32, dtype=torch.long),
    }
    off = rectified_flow_loss(model, batch, conditioning_mode="caption_semantic", cfg_dropout=0, pad_token_id=0)
    off["loss"].backward()
    assert off["loss_index_head"].item() == 0
    model.zero_grad(set_to_none=True)
    on = rectified_flow_loss(
        model,
        batch,
        conditioning_mode="caption_semantic",
        cfg_dropout=0,
        pad_token_id=0,
        index_head_loss_weight=0.1,
        palette_head_loss_weight=0.1,
        palette_presence_loss_weight=0.1,
    )
    on["loss"].backward()
    assert on["index_head_active"] is True
    assert on["loss_index_head"].item() >= 0


def test_resume_compatibility_and_rejection(tmp_path: Path) -> None:
    saved = _manifest(tmp_path)
    validate_resume_compatibility(saved, dict(saved))
    changed = dict(saved)
    changed["conditioning_schema_hash"] = "different"
    with pytest.raises(IncompatibleResumeError, match="conditioning_schema_hash"):
        validate_resume_compatibility(changed, saved)
    assert validate_resume_compatibility(
        changed, saved, unsafe=True, unsafe_reason="test-only incompatible conditioning override"
    ) == ["conditioning_schema_hash"]


def test_rng_state_restoration() -> None:
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(2))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_checkpoint_loader_requires_weights_only_without_unsafe_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from spritelab.training import checkpoint_io

    calls: list[dict[str, object]] = []
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"synthetic")

    class SafeTorch:
        @staticmethod
        def load(source, **kwargs):
            calls.append({"file_like": callable(getattr(source, "read", None)), **kwargs})
            return {"model_type": "synthetic"}

    monkeypatch.setattr(checkpoint_io, "torch", SafeTorch())
    expected_hash = sha256(checkpoint_path.read_bytes()).hexdigest()
    assert checkpoint_io.load_checkpoint(checkpoint_path, expected_sha256=expected_hash) == {"model_type": "synthetic"}
    assert calls == [
        {
            "file_like": True,
            "map_location": "cpu",
            "weights_only": True,
        }
    ]
    with pytest.raises(ValueError, match="SHA-256"):
        checkpoint_io.load_checkpoint(checkpoint_path, expected_sha256="0" * 64)
    assert len(calls) == 1

    class UnsupportedTorch:
        @staticmethod
        def load(source, **kwargs):
            del source, kwargs
            raise TypeError("weights_only unsupported")

    monkeypatch.setattr(checkpoint_io, "torch", UnsupportedTorch())
    with pytest.raises(RuntimeError, match="safe weights-only"):
        checkpoint_io.load_checkpoint(checkpoint_path)


def test_checkpoint_saves_ema_optimizer_scheduler_scaler_and_manifest(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    class Scaler:
        def state_dict(self) -> dict:
            return {"scale": 7.0}

    tokenizer = SpriteTextTokenizer.build(["sword"])
    manifest = _manifest(tmp_path)
    path = tmp_path / "checkpoint.pt"
    ema = _init_ema_state(model)
    _save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config_json={"experiment_manifest": manifest},
        step=2,
        scheduler=scheduler,
        scaler=Scaler(),
        ema_state=ema,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 2
    assert checkpoint["optimizer_state_dict"] == optimizer.state_dict()
    assert checkpoint["scheduler_state_dict"] == scheduler.state_dict()
    assert checkpoint["scaler_state_dict"] == {"scale": 7.0}
    assert checkpoint["ema_state_dict"].keys() == ema.keys()
    assert checkpoint["experiment_manifest"]["experiment_hash"] == manifest["experiment_hash"]


def test_inference_config_parity_and_validation_guards(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert manifest["sampling"] == {"cfg_scale": 3.0, "steps": 30}
    validate_inference_parity(manifest, {"experiment_manifest": manifest})
    incompatible = dict(manifest)
    incompatible["model_architecture_hash"] = "other"
    with pytest.raises(IncompatibleResumeError, match="model_architecture_hash"):
        validate_inference_parity(incompatible, {"experiment_manifest": manifest})
    validate_ablation_config(_config())
    unsafe = _config()
    unsafe["self_conditioning"] = True
    with pytest.raises(ValueError, match="self-conditioning"):
        validate_ablation_config(unsafe)


def test_windows_safe_cli_construction() -> None:
    command = windows_command(["python", "-m", "spritelab", "train", "--out", Path("C:/run dir/out")])
    assert command == "python -m spritelab train --out 'C:\\run dir\\out'"
    assert '\\"' not in command


def test_smoke_checkpoint_rejects_max_step_extension_and_exports_paired_seed(tmp_path: Path) -> None:
    from _semantic_dataset import default_specs, make_semantic_dataset
    from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest

    dataset = make_semantic_dataset(tmp_path / "dataset", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=11)
    manifest_path = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest_path, result.rows)
    manifest = build_experiment_manifest(
        _config(),
        dataset_manifest=manifest_path,
        tokenizer=SpriteTextTokenizer.build(["sword", "potion"]).to_json_dict(),
        repo=tmp_path,
    )
    out = tmp_path / "run"
    base = {
        "dataset_dir": dataset,
        "training_manifest": manifest_path,
        "out_dir": out,
        "batch_size": 2,
        "device": "cpu",
        "base_channels": 8,
        "channel_mults": "1,2",
        "res_blocks_per_level": 1,
        "embed_dim": 8,
        "sample_every": 0,
        "save_every": 1,
        "validation_mode": "same",
        "max_records": 4,
        "experiment_manifest": manifest,
    }
    first = run_challenger_training(ChallengerTrainConfig(**base, max_steps=1))
    checkpoint = out / "checkpoint_last.pt"
    assert first["steps_completed"] == 1
    with pytest.raises(IncompatibleResumeError, match="max_optimizer_steps"):
        run_challenger_training(ChallengerTrainConfig(**base, max_steps=2, resume_from=checkpoint))

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"prompt_id": "p0", "prompt": "red sword"}) + "\n",
        encoding="utf-8",
    )
    noise_seeds = []
    for seed in (71, 72):
        sample_out = tmp_path / f"sample_{seed}"
        result = run_sample_generator_challenger(
            ChallengerSampleConfig(
                checkpoint=checkpoint,
                prompts=prompts,
                out_dir=sample_out,
                max_samples=1,
                steps=2,
                cfg_scale=1.0,
                seed=seed,
                noise_seed=seed,
                batch_size=1,
            )
        )
        assert result["sample_count"] == 1
        record = json.loads((sample_out / "generated_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
        noise_seeds.append(record["noise_seed"])
    assert noise_seeds == [71, 72]
