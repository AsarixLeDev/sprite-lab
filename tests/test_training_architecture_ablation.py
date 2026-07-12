from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from spritelab.training.experiment_system import (
    RESUME_HARD_FIELDS,
    IncompatibleResumeError,
    measured_architecture_identity,
    validate_resume_compatibility,
)
from spritelab.training.generator_challenger import (
    AUXILIARY_HEAD_PREFIXES,
    AuxiliaryHeadsMode,
    RectifiedFlowUNet,
    _init_ema_state,
    load_challenger_from_checkpoint,
    rectified_flow_loss,
    resolve_auxiliary_heads_mode,
)
from spritelab.training.tokenization import SpriteTextTokenizer


def _model(mode: str | None, *, full_size: bool = False) -> RectifiedFlowUNet:
    kwargs = {
        "vocab_size": 32,
        "embed_dim": 64 if full_size else 8,
        "base_channels": 64 if full_size else 8,
        "channel_mults": (1, 2, 4) if full_size else (1, 2),
        "res_blocks_per_level": 2 if full_size else 1,
        "pad_token_id": 0,
    }
    if mode is not None:
        kwargs["auxiliary_heads_mode"] = mode
    return RectifiedFlowUNet(**kwargs)


def _is_auxiliary(name: str) -> bool:
    return name.startswith(AUXILIARY_HEAD_PREFIXES)


def _forward_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(2, 4, 32, 32), torch.rand(2), torch.randint(0, 32, (2, 4))


def _resume_manifest(model: RectifiedFlowUNet) -> dict:
    architecture = measured_architecture_identity(model)
    manifest = dict.fromkeys(RESUME_HARD_FIELDS, "same")
    manifest.update(
        {
            "model_architecture": architecture,
            "model_architecture_hash": architecture["hash"],
            "auxiliary_heads_mode": model.auxiliary_heads_mode.value,
        }
    )
    return manifest


def test_physical_parameter_optimizer_state_and_ema_ownership() -> None:
    torch.manual_seed(4)
    absent = _model("absent", full_size=True)
    torch.manual_seed(4)
    enabled = _model("palette_index", full_size=True)

    absent_named = dict(absent.named_parameters())
    enabled_named = dict(enabled.named_parameters())
    auxiliary_named = {name: value for name, value in enabled_named.items() if _is_auxiliary(name)}
    derived_auxiliary_count = sum(parameter.numel() for parameter in auxiliary_named.values())
    assert derived_auxiliary_count > 0
    assert (
        sum(parameter.numel() for parameter in enabled.parameters())
        - sum(parameter.numel() for parameter in absent.parameters())
        == derived_auxiliary_count
    )
    assert not any(_is_auxiliary(name) for name in absent_named)
    assert auxiliary_named

    optimizer = torch.optim.AdamW(absent.parameters(), lr=1e-3)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in absent.parameters() if parameter.requires_grad}
    assert set(_init_ema_state(absent)) == set(absent.state_dict())
    assert not any(_is_auxiliary(key) for key in absent.state_dict())
    assert any(_is_auxiliary(key) for key in enabled.state_dict())


def test_forward_schema_distinguishes_absent_present_and_loss_disabled() -> None:
    x, t, tokens = _forward_batch()
    absent = _model("absent")
    absent_output = absent(x, t, caption_tokens=tokens, return_aux=True)
    assert absent_output["schema_version"] == "spritelab_generator_forward_v2"
    assert absent_output["auxiliary_heads_available"] is False
    assert absent_output["palette_rgb"] is None
    assert absent_output["palette_presence_logits"] is None
    assert absent_output["index_logits"] is None

    enabled = _model("palette_index")
    enabled_output = enabled(x, t, caption_tokens=tokens, return_aux=True)
    assert enabled_output["auxiliary_heads_available"] is True
    assert enabled_output["palette_rgb"].shape == (2, 16, 3)
    assert enabled_output["palette_presence_logits"].shape == (2, 16)
    assert enabled_output["index_logits"].shape == (2, 16, 32, 32)

    minimal_batch = {"rgba": torch.rand(2, 4, 32, 32), "caption_tokens": tokens, "semantic_tokens": tokens}
    loss = rectified_flow_loss(
        absent, minimal_batch, conditioning_mode="caption_semantic", cfg_dropout=0.0, pad_token_id=0
    )
    assert loss["auxiliary_heads_available"] is False
    assert loss["auxiliary_loss_enabled"] is False


def test_absent_nonzero_auxiliary_loss_fails_and_enabled_zero_keeps_heads() -> None:
    tokens = torch.randint(0, 32, (1, 4))
    batch = {"rgba": torch.rand(1, 4, 32, 32), "caption_tokens": tokens, "semantic_tokens": tokens}
    with pytest.raises(ValueError, match="palette_index"):
        rectified_flow_loss(
            _model("absent"),
            batch,
            conditioning_mode="caption_semantic",
            cfg_dropout=0.0,
            pad_token_id=0,
            palette_head_loss_weight=0.1,
        )
    enabled = _model("palette_index")
    loss = rectified_flow_loss(enabled, batch, conditioning_mode="caption_semantic", cfg_dropout=0.0, pad_token_id=0)
    assert loss["auxiliary_heads_available"] is True
    assert loss["auxiliary_loss_enabled"] is False
    assert any(_is_auxiliary(name) for name, _parameter in enabled.named_parameters())


def test_initialization_and_architecture_hashes_are_stable() -> None:
    torch.manual_seed(91)
    first = _model("absent")
    torch.manual_seed(91)
    second = _model("absent")
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())
    first_identity = measured_architecture_identity(first)
    second_identity = measured_architecture_identity(second)
    assert first_identity == second_identity

    torch.manual_seed(91)
    enabled = _model("palette_index")
    for key, value in first.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[key])
    assert first_identity["hash"] != measured_architecture_identity(enabled)["hash"]


def test_legacy_loss_zero_is_enabled_legacy_not_headless() -> None:
    legacy = _model(None)
    identity = measured_architecture_identity(legacy)
    assert legacy.auxiliary_heads_mode is AuxiliaryHeadsMode.PALETTE_INDEX
    assert legacy.legacy_auxiliary_heads_adapter is True
    assert identity["identity_kind"] == "legacy_adapter"
    assert identity["auxiliary_heads_instantiated"] is True
    assert identity["promotion_eligible"] is False


def test_explicit_manifest_mode_reaches_production_constructor_adapter() -> None:
    manifest = _resume_manifest(_model("absent"))
    mode, legacy = resolve_auxiliary_heads_mode(None, manifest)
    assert mode is AuxiliaryHeadsMode.ABSENT
    assert legacy is False
    legacy_manifest = _resume_manifest(_model(None))
    mode, legacy = resolve_auxiliary_heads_mode(None, legacy_manifest)
    assert mode is AuxiliaryHeadsMode.PALETTE_INDEX
    assert legacy is True


def test_cross_mode_legacy_and_tampered_safe_resume_fail() -> None:
    absent = _resume_manifest(_model("absent"))
    enabled = _resume_manifest(_model("palette_index"))
    legacy = _resume_manifest(_model(None))
    validate_resume_compatibility(absent, deepcopy(absent))
    with pytest.raises(IncompatibleResumeError, match=r"model_architecture_hash|auxiliary_heads_mode"):
        validate_resume_compatibility(absent, enabled)
    with pytest.raises(IncompatibleResumeError, match=r"model_architecture_hash|auxiliary_heads_mode"):
        validate_resume_compatibility(absent, legacy)
    tampered = deepcopy(absent)
    tampered["model_architecture"]["parameter_count"] += 1
    with pytest.raises(IncompatibleResumeError, match="tampered"):
        validate_resume_compatibility(absent, tampered)


@pytest.mark.parametrize(
    "field",
    [
        "optimizer_identity_hash",
        "schedule_identity_hash",
        "global_batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "ema_identity_hash",
        "evaluation_cadence",
        "checkpoint_cadence",
        "max_optimizer_steps",
    ],
)
def test_fair_resume_fields_are_hard(field: str) -> None:
    saved = _resume_manifest(_model("absent"))
    current = deepcopy(saved)
    current[field] = "changed"
    with pytest.raises(IncompatibleResumeError, match=field):
        validate_resume_compatibility(current, saved)


def test_unsafe_resume_records_every_bypassed_mismatch() -> None:
    saved = _resume_manifest(_model("absent"))
    current = deepcopy(saved)
    current["optimizer_identity_hash"] = "other-optimizer"
    current["schedule_identity_hash"] = "other-schedule"
    current["evaluation_cadence"] = 17
    record: dict = {}
    mismatches = validate_resume_compatibility(
        current,
        saved,
        unsafe=True,
        unsafe_reason="recovery inspection only",
        unsafe_record=record,
    )
    assert record["mismatches"] == mismatches
    assert set(mismatches) == {"optimizer_identity_hash", "schedule_identity_hash", "evaluation_cadence"}
    assert record["exact_replay_claimed"] is False
    assert record["fair_architecture_comparison_eligible"] is False
    assert record["checkpoint_promotion_eligible"] is False


def test_checkpoint_loader_is_strict_and_blocks_cross_mode() -> None:
    enabled = _model("palette_index")
    tokenizer = SpriteTextTokenizer.build(["sprite"])
    checkpoint = {
        "model_type": "generator_challenger",
        "model_config": enabled.config(),
        "model_state_dict": enabled.state_dict(),
        "vocab": tokenizer.to_json_dict(),
        "conditioning_mode": "caption_semantic",
    }
    loaded, _tokenizer, _mode, _length = load_challenger_from_checkpoint(checkpoint, device="cpu")
    assert loaded.auxiliary_heads_mode is AuxiliaryHeadsMode.PALETTE_INDEX
    with pytest.raises(RuntimeError, match="safe resume/import is blocked"):
        load_challenger_from_checkpoint(checkpoint, device="cpu", auxiliary_heads_mode="absent")

    missing_auxiliary = deepcopy(checkpoint)
    missing_auxiliary["model_state_dict"] = {
        key: value for key, value in checkpoint["model_state_dict"].items() if not _is_auxiliary(key)
    }
    with pytest.raises(RuntimeError, match=r"missing_key_classes=.*auxiliary"):
        load_challenger_from_checkpoint(missing_auxiliary, device="cpu")

    legacy = deepcopy(missing_auxiliary)
    legacy["model_config"].pop("auxiliary_heads_mode")
    imported, *_rest = load_challenger_from_checkpoint(legacy, device="cpu", legacy_evaluation_import=True)
    assert imported.checkpoint_import_mode == "legacy_missing_auxiliary_evaluation_only"
    assert imported.safe_resume_eligible is False
    assert imported.fair_architecture_comparison_eligible is False
    assert imported.checkpoint_promotion_eligible is False
