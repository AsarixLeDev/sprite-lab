from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest

from spritelab.training import generator_challenger as generator_module
from spritelab.training.cli import main as train_cli
from spritelab.training.experiment_system import (
    EXPERIMENT_MANIFEST_VERSION,
    RESUME_HARD_FIELDS,
    IncompatibleResumeError,
    UnsafeResumeMismatches,
    build_experiment_manifest,
    create_unsafe_resume_revocation,
    validate_resume_against_runtime,
    validate_resume_compatibility,
)
from spritelab.training.generator_challenger import ChallengerTrainConfig, run_challenger_training
from spritelab.training.tokenization import SpriteTextTokenizer


def _runtime(tmp_path: Path) -> tuple[dict, dict, Path, Path, dict]:
    dataset = tmp_path / "dataset.jsonl"
    split = tmp_path / "split.json"
    dataset.write_text('{"sprite_id":"a"}\n', encoding="utf-8")
    split.write_text('{"train":["a"]}\n', encoding="utf-8")
    tokenizer = SpriteTextTokenizer.build(["sprite"]).to_json_dict()
    config = {
        "model": {"base_channels": 8, "channel_mults": [1, 2], "embed_dim": 8},
        "conditioning": {"mode": "caption_semantic"},
        "optimizer": {"name": "adamw", "schedule": "cosine", "warmup_steps": 2},
        "loss": {"name": "velocity"},
        "seeds": {"training": 7, "sampler": 8},
        "sampler": {"name": "weighted"},
        "runtime": {
            "micro_batch_size": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "precision": "fp32",
            "determinism": "strict",
            "sample_every": 5,
            "save_every": 10,
            "max_steps": 20,
        },
        "ema": {"enabled": True, "decay": 0.99},
    }
    manifest = build_experiment_manifest(
        config,
        dataset_manifest=dataset,
        split_manifest=split,
        tokenizer=tokenizer,
        repo=tmp_path,
    )
    return manifest, config, dataset, split, tokenizer


def test_all_resume_hard_groups_missing_fail_closed() -> None:
    with pytest.raises(IncompatibleResumeError, match="missing"):
        validate_resume_compatibility(
            {"manifest_version": EXPERIMENT_MANIFEST_VERSION},
            {"manifest_version": EXPERIMENT_MANIFEST_VERSION},
        )


@pytest.mark.parametrize("field", RESUME_HARD_FIELDS)
def test_each_resume_hard_field_is_mandatory(tmp_path: Path, field: str) -> None:
    manifest, *_ = _runtime(tmp_path)
    saved = deepcopy(manifest)
    saved.pop(field)
    with pytest.raises(IncompatibleResumeError, match=field):
        validate_resume_compatibility(manifest, saved)


def test_self_consistent_claims_do_not_override_runtime_files(tmp_path: Path) -> None:
    manifest, config, dataset, split, tokenizer = _runtime(tmp_path)
    forged = deepcopy(manifest)
    forged["dataset_manifest_hash"] = "a" * 64
    with pytest.raises(IncompatibleResumeError, match="dataset_manifest_hash"):
        validate_resume_against_runtime(
            forged,
            config,
            dataset_manifest=dataset,
            split_manifest=split,
            tokenizer=tokenizer,
        )


def test_changed_runtime_config_rejects_stored_identity(tmp_path: Path) -> None:
    manifest, config, dataset, split, tokenizer = _runtime(tmp_path)
    config["optimizer"]["name"] = "sgd"
    with pytest.raises(IncompatibleResumeError, match="optimizer_identity_hash"):
        validate_resume_against_runtime(
            manifest,
            config,
            dataset_manifest=dataset,
            split_manifest=split,
            tokenizer=tokenizer,
        )


def test_every_low_level_unsafe_resume_returns_or_writes_revocation(tmp_path: Path) -> None:
    manifest, *_ = _runtime(tmp_path)
    changed = deepcopy(manifest)
    changed["loss_configuration_hash"] = "b" * 64
    result = validate_resume_compatibility(changed, manifest, unsafe=True, unsafe_reason="adversarial test")
    assert isinstance(result, UnsafeResumeMismatches)
    record = result.revocation_record
    assert record["unsafe_resume"] is True
    assert record["all_mismatches"]
    assert record["exact_replay_eligible"] is False
    assert record["fair_comparison_eligible"] is False
    assert record["promotion_eligible"] is False


def test_legacy_schema_requires_explicit_unsafe_record(tmp_path: Path) -> None:
    manifest, *_ = _runtime(tmp_path)
    legacy = deepcopy(manifest)
    legacy["manifest_version"] = "spritelab_experiment_v2"
    with pytest.raises(IncompatibleResumeError, match="manifest_version"):
        validate_resume_compatibility(manifest, legacy)
    destination: dict = {}
    validate_resume_compatibility(
        manifest,
        legacy,
        unsafe=True,
        unsafe_reason="explicit legacy migration",
        unsafe_record=destination,
    )
    assert destination["unsafe_resume"] is True
    assert destination["all_mismatches"]


def test_public_unsafe_resume_entry_points_are_centralized_or_forbidden() -> None:
    compatibility_source = inspect.getsource(validate_resume_compatibility)
    training_source = inspect.getsource(run_challenger_training)
    assert "create_unsafe_resume_revocation" in compatibility_source
    assert "create_unsafe_resume_revocation" in training_source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_optimizer_steps", "20"),
        ("max_optimizer_steps", True),
        ("evaluation_cadence", -1),
        ("sampler_policy_hash", "not-a-sha256"),
        ("precision_policy", {}),
        ("lineage_parent_identity", []),
    ],
)
def test_malformed_current_schema_values_are_rejected(tmp_path: Path, field: str, value: object) -> None:
    manifest, *_ = _runtime(tmp_path)
    current = deepcopy(manifest)
    saved = deepcopy(manifest)
    current[field] = deepcopy(value)
    saved[field] = deepcopy(value)
    with pytest.raises(IncompatibleResumeError) as caught:
        validate_resume_compatibility(current, saved)
    assert any(field in mismatch for mismatch in caught.value.mismatches)


@pytest.mark.parametrize("version", [None, "spritelab_experiment_v999"])
def test_versionless_or_unsupported_checkpoint_never_qualifies_as_current(tmp_path: Path, version: str | None) -> None:
    manifest, *_ = _runtime(tmp_path)
    saved = deepcopy(manifest)
    if version is None:
        saved.pop("manifest_version")
    else:
        saved["manifest_version"] = version
    with pytest.raises(IncompatibleResumeError) as caught:
        validate_resume_compatibility(manifest, saved)
    assert any("checkpoint.manifest_version" in mismatch for mismatch in caught.value.mismatches)


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_unsafe_resume_requires_nonblank_human_reason(tmp_path: Path, reason: str | None) -> None:
    manifest, *_ = _runtime(tmp_path)
    changed = deepcopy(manifest)
    changed["max_optimizer_steps"] += 1
    with pytest.raises(IncompatibleResumeError, match="reason"):
        validate_resume_compatibility(changed, manifest, unsafe=True, unsafe_reason=reason)


def test_unsafe_resume_without_a_detected_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, *_ = _runtime(tmp_path)
    with pytest.raises(IncompatibleResumeError, match="unnecessary"):
        validate_resume_compatibility(
            manifest,
            deepcopy(manifest),
            unsafe=True,
            unsafe_reason="operator requested an unnecessary override",
        )


def test_complete_mismatch_list_and_revocation_are_preserved_immutably() -> None:
    mismatches = ["dataset_manifest_hash", "optimizer_identity_hash", "max_optimizer_steps"]
    record = create_unsafe_resume_revocation(
        reason="operator accepted all three explicitly reviewed mismatches",
        mismatches=mismatches,
        source_checkpoint_identity="a" * 64,
        target_runtime_identity="b" * 64,
        event_identity="2026-07-13T12:00:00+00:00",
    )
    assert list(record["all_mismatches"]) == mismatches
    assert record["schema_version"] == "spritelab_unsafe_resume_revocation_v1"
    assert record["timestamp"] == "2026-07-13T12:00:00+00:00"
    with pytest.raises(TypeError):
        record["reason"] = "mutated"  # type: ignore[index]


def test_low_level_training_loader_cannot_bypass_blank_reason_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_loader(*args: object, **kwargs: object) -> dict:
        nonlocal called
        called = True
        raise AssertionError("checkpoint loader must not be reached")

    monkeypatch.setattr(generator_module, "_load_checkpoint", forbidden_loader)
    config = ChallengerTrainConfig(
        dataset_dir=tmp_path,
        training_manifest=tmp_path / "absent.jsonl",
        out_dir=tmp_path / "out",
        resume_from=tmp_path / "checkpoint.pt",
        unsafe_resume=True,
        unsafe_resume_reason=" ",
    )
    with pytest.raises(ValueError, match="unsafe-resume-reason"):
        run_challenger_training(config)
    assert called is False


@pytest.mark.parametrize("reason_args", [[], ["--unsafe-resume-reason", "   "]])
def test_low_level_cli_requires_explicit_nonblank_unsafe_reason(tmp_path: Path, reason_args: list[str]) -> None:
    with pytest.raises(ValueError, match="unsafe-resume-reason"):
        train_cli(
            [
                "experiment",
                "run",
                "--config",
                str(tmp_path / "not-read.yaml"),
                "--resume",
                str(tmp_path / "not-loaded.pt"),
                "--unsafe-resume",
                *reason_args,
            ]
        )
