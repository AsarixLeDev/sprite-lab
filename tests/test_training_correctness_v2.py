from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.determinism import (
    DeterminismQualificationError,
    configure_determinism,
)
from spritelab.training.experiment_system import (
    CONDITIONING_SCHEMA_VERSION,
    CONDITIONING_SCHEMA_VERSION_V1,
    IncompatibleResumeError,
    adapt_conditioning_schema_v1_manifest,
    stable_hash,
    validate_inference_parity,
)
from spritelab.training.generator_challenger import (
    ChallengerTrainConfig,
    _apply_structured_field_dropout,
    run_challenger_training,
)
from spritelab.training.sampler_resume import (
    SAMPLER_STATE_VERSION,
    StatefulPermutationSampler,
    UnsupportedExactResumeError,
    validate_worker_mode,
)
from spritelab.training.structured_conditioning import (
    MISSING_TOKEN,
    STRUCTURED_VOCAB_SCHEMA_V1,
    STRUCTURED_VOCAB_SCHEMA_V2,
    UNKNOWN_TOKEN,
    StructuredConditioningVocab,
    adapt_schema_v1_vocab,
    build_structured_conditioning_vocab,
    encode_structured_conditioning,
    structured_vocab_from_checkpoint,
)
from spritelab.training.timestep_validation import (
    TimestepBucketAccumulator,
    timestep_bucket_index,
)


def test_missing_oov_ids_stable_hash_and_field_specific_encoding() -> None:
    records = [{"category": "weapon", "object_name": "sword", "colors": ["red"]}]
    first = build_structured_conditioning_vocab(records)
    second = build_structured_conditioning_vocab(reversed(records))
    assert first.schema_version == STRUCTURED_VOCAB_SCHEMA_V2
    assert first.categories[:2] == (MISSING_TOKEN, UNKNOWN_TOKEN)
    assert stable_hash(first.to_json_dict()) == stable_hash(second.to_json_dict())
    malformed = first.to_json_dict()
    malformed["categories"] = [UNKNOWN_TOKEN, MISSING_TOKEN, "weapon"]
    with pytest.raises(ValueError, match="ID 0"):
        StructuredConditioningVocab.from_json_dict(malformed)

    missing = encode_structured_conditioning({}, first)
    oov = encode_structured_conditioning({"category": "armor", "object_name": "sword", "colors": ["violet"]}, first)
    assert missing["category_id"] == 0
    assert oov["category_id"] == 1
    assert missing["color_multi_hot"][0] == 1.0
    assert oov["color_multi_hot"][0] == 0.0
    assert oov["color_multi_hot"][1] == 1.0
    assert oov["object_id"] == first.objects.index("sword")


def test_schema_v1_adapter_is_explicit_and_never_remaps() -> None:
    raw = {
        "schema_version": STRUCTURED_VOCAB_SCHEMA_V1,
        "categories": [UNKNOWN_TOKEN, "weapon"],
        "objects": [UNKNOWN_TOKEN],
        "base_objects": [UNKNOWN_TOKEN],
        "colors": [UNKNOWN_TOKEN],
        "materials": [UNKNOWN_TOKEN],
        "shapes": [UNKNOWN_TOKEN],
        "functions": [UNKNOWN_TOKEN],
        "styles": [UNKNOWN_TOKEN],
    }
    with pytest.raises(ValueError, match="explicit"):
        StructuredConditioningVocab.from_json_dict(raw)
    adapted = adapt_schema_v1_vocab(raw)
    assert adapted is not None
    assert adapted.categories == (UNKNOWN_TOKEN, "weapon")
    with pytest.raises(ValueError, match="explicit"):
        structured_vocab_from_checkpoint({"structured_conditioning_vocab": raw})
    loaded = structured_vocab_from_checkpoint({"structured_conditioning_vocab": raw}, allow_schema_v1_adapter=True)
    assert loaded == adapted


def test_schema_v2_checkpoint_rejected_by_schema_v1_inference() -> None:
    current = {
        "conditioning_schema": {"version": CONDITIONING_SCHEMA_VERSION_V1},
        "conditioning_schema_hash": "v1",
    }
    saved = {
        "conditioning_schema": {"version": CONDITIONING_SCHEMA_VERSION},
        "conditioning_schema_hash": "v2",
    }
    with pytest.raises(IncompatibleResumeError, match="schema mismatch"):
        validate_inference_parity(current, {"experiment_manifest": saved})
    legacy = {"conditioning_schema": {"version": CONDITIONING_SCHEMA_VERSION_V1}}
    adapted = adapt_conditioning_schema_v1_manifest(legacy)
    assert adapted["compatibility_adapter"]["from"] == CONDITIONING_SCHEMA_VERSION_V1
    assert "compatibility_adapter" not in legacy


def test_benchmark_checkpoint_vocabulary_loading_uses_recorded_schema_v2() -> None:
    vocab = build_structured_conditioning_vocab([{"category": "weapon", "object_name": "sword"}])
    checkpoint = {
        "conditioning_schema_version": CONDITIONING_SCHEMA_VERSION,
        "structured_conditioning_vocab": vocab.to_json_dict(),
    }
    loaded = structured_vocab_from_checkpoint(checkpoint)
    assert loaded == vocab
    assert encode_structured_conditioning({"category": "unseen"}, loaded)["category_id"] == 1


def test_structured_dropout_uses_missing_not_unknown() -> None:
    inputs = {
        "caption_tokens": torch.ones(2, 2, dtype=torch.long),
        "structured_conditioning": {
            "category_id": torch.tensor([2, 2]),
            "color_multi_hot": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        },
    }
    result = _apply_structured_field_dropout(
        inputs,
        dropout=1.0,
        training=True,
        dropout_rates={"category": 1.0, "colors": 1.0},
    )["structured_conditioning"]
    assert result["category_id"].tolist() == [0, 0]
    assert result["color_multi_hot"].tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def _sampler_state(size: int = 7) -> tuple[list[int], dict, torch.Generator]:
    sampler_generator = torch.Generator().manual_seed(4)
    loader_generator = torch.Generator().manual_seed(5)
    sampler = StatefulPermutationSampler(range(size), generator=sampler_generator)
    iterator = iter(sampler)
    prefix = [next(iterator), next(iterator), next(iterator)]
    state = sampler.state_dict(batch_size=2, loader_generator=loader_generator)
    suffix = list(iterator)
    restored_loader = torch.Generator()
    restored = StatefulPermutationSampler(range(size), generator=torch.Generator())
    restored.load_state_dict(state, batch_size=2, loader_generator=restored_loader)
    assert list(restored) == suffix
    assert restored_loader.get_state().equal(state["dataloader_generator_state"])
    return prefix + suffix, state, restored_loader


def test_sampler_permutation_cursor_and_loader_generator_restore() -> None:
    order, state, _loader = _sampler_state()
    assert sorted(order) == list(range(7))
    assert state["schema_version"] == SAMPLER_STATE_VERSION
    assert state["sample_cursor"] == 3
    assert state["batch_index"] == 1


def test_accumulation_position_and_worker_mode_are_not_silent() -> None:
    _order, state, _loader = _sampler_state()
    state["gradient_accumulation_position"] = 1
    with pytest.raises(UnsupportedExactResumeError, match="gradient_accumulation_position"):
        StatefulPermutationSampler(range(7), generator=torch.Generator()).load_state_dict(
            state,
            batch_size=2,
            loader_generator=torch.Generator(),
        )
    with pytest.raises(UnsupportedExactResumeError, match="num_workers"):
        validate_worker_mode(num_workers=2, exact_resume=True)
    with pytest.warns(RuntimeWarning, match="num_workers"):
        assert validate_worker_mode(num_workers=2, exact_resume=True, unsafe=True)


def test_cuda_determinism_strict_and_warn_behavior() -> None:
    with pytest.raises(DeterminismQualificationError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_determinism("strict", device="cuda", torch_module=torch, environ={})
    with pytest.warns(RuntimeWarning, match="not guaranteed"):
        report = configure_determinism("warn", device="cuda", torch_module=torch, environ={})
    assert report["qualified"] is False
    assert report["cross_gpu_or_version_identity_claimed"] is False
    configure_determinism("off", device="cpu", torch_module=torch)


def test_timestep_bucket_boundaries_and_support() -> None:
    assert [timestep_bucket_index(value) for value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)] == [0, 1, 2, 3, 4, 4]
    accumulator = TimestepBucketAccumulator()
    accumulator.add(0.1, {"loss_velocity": 2, "loss_palette_aux": 1, "loss_index_head": 3})
    report = accumulator.report()
    assert report["buckets"]["early"]["sample_count"] == 1
    assert report["buckets"]["low-mid"]["sample_count"] == 0
    assert report["buckets"]["low-mid"]["loss_velocity"] is None


def _tiny_training_config(tmp_path: Path, out: str, **overrides: object) -> ChallengerTrainConfig:
    dataset = make_semantic_dataset(tmp_path / "dataset", default_specs())
    built = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, built.rows)
    values: dict[str, object] = {
        "dataset_dir": dataset,
        "training_manifest": manifest,
        "out_dir": tmp_path / out,
        "batch_size": 1,
        "max_steps": 4,
        "device": "cpu",
        "base_channels": 8,
        "channel_mults": "1,2",
        "res_blocks_per_level": 1,
        "embed_dim": 8,
        "sample_every": 0,
        "save_every": 1,
        "validation_mode": "same",
        "max_records": 4,
        "lr_schedule": "cosine",
        "experiment_manifest": {
            "dataset_manifest_hash": "tiny-dataset",
            "split_manifest_hash": "tiny-split",
            "model_architecture_hash": "tiny-model",
            "conditioning_schema_hash": "tiny-conditioning-v2",
            "conditioning_schema": {"version": CONDITIONING_SCHEMA_VERSION},
        },
    }
    values.update(overrides)
    return ChallengerTrainConfig(**values)


def _assert_nested_equal(left: object, right: object, path: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping) and set(left) == set(right), path
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_nested_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def test_uninterrupted_and_mid_epoch_resumed_states_are_exact(tmp_path: Path) -> None:
    full_report = run_challenger_training(_tiny_training_config(tmp_path, "full"))
    run_challenger_training(_tiny_training_config(tmp_path, "resumed", stop_after_step=2))
    interrupt = tmp_path / "resumed" / "checkpoint_step_000002.pt"
    resumed_report = run_challenger_training(_tiny_training_config(tmp_path, "resumed", resume_from=interrupt))
    full = torch.load(tmp_path / "full" / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "ema_state_dict",
        "scheduler_state_dict",
        "rng_states",
        "sampler_state",
        "training_metrics_summary",
    ):
        _assert_nested_equal(full[key], resumed[key], key)
    full_metrics = [json.loads(line) for line in (tmp_path / "full" / "train_metrics.jsonl").read_text().splitlines()]
    resumed_metrics = [
        json.loads(line) for line in (tmp_path / "resumed" / "train_metrics.jsonl").read_text().splitlines()
    ]
    for rows in (full_metrics, resumed_metrics):
        for row in rows:
            row.pop("elapsed_seconds")
    assert full_metrics == resumed_metrics
    assert full_report["timestep_validation"]["non_ema"] is not None
    assert full_report["timestep_validation"]["ema"] is not None
    assert resumed_report["steps_completed"] == 4
