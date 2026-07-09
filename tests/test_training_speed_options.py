from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.cli import main as train_cli
from spritelab.training.generator_challenger import (
    ChallengerTrainConfig,
    _init_ema_fast_state,
    _init_ema_state,
    _update_ema_state,
    _update_ema_state_fast,
    run_challenger_training,
)
from spritelab.training.optim_utils import apply_backend_speed_flags, build_adamw


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=1, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    return dataset, manifest


class _FakeModel:
    """Minimal stand-in exposing only the ``state_dict()`` surface the EMA helpers use."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.tensors


def test_ema_fast_path_matches_legacy_update_exactly_including_nonfloat_tensor() -> None:
    torch.manual_seed(0)
    tensors = {
        "weight": torch.randn(4, 4),
        "bias": torch.randn(4),
        "step_count": torch.tensor(0, dtype=torch.int64),
    }
    model = _FakeModel(tensors)

    ema_legacy = _init_ema_state(model)
    ema_fast = _init_ema_state(model)
    fast_cache = _init_ema_fast_state(ema_fast, model)

    decay = 0.9
    for step in range(5):
        # Simulate optimizer steps: in-place mutation, same storage (as real params).
        tensors["weight"].add_(torch.randn(4, 4) * 0.01)
        tensors["bias"].add_(torch.randn(4) * 0.01)
        tensors["step_count"].fill_(step + 1)

        _update_ema_state(ema_legacy, model, decay=decay)
        _update_ema_state_fast(fast_cache, decay=decay)

    assert set(ema_legacy) == set(ema_fast)
    for key in ema_legacy:
        assert torch.equal(ema_legacy[key], ema_fast[key]), key


def test_build_adamw_default_matches_plain_adamw() -> None:
    params = [torch.nn.Parameter(torch.randn(3, 3))]
    optimizer = build_adamw(params, lr=1e-3, fused=False)
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0].get("fused") in (None, False)


def test_build_adamw_fused_falls_back_on_cpu_without_raising() -> None:
    params = [torch.nn.Parameter(torch.randn(3, 3))]
    optimizer = build_adamw(params, lr=1e-3, fused=True)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_apply_backend_speed_flags_default_is_noop() -> None:
    before_benchmark = torch.backends.cudnn.benchmark
    before_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    before_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        apply_backend_speed_flags(cudnn_benchmark=False, tf32=False)
        assert torch.backends.cudnn.benchmark == before_benchmark
        assert torch.backends.cuda.matmul.allow_tf32 == before_tf32_matmul
        assert torch.backends.cudnn.allow_tf32 == before_tf32_cudnn
    finally:
        torch.backends.cudnn.benchmark = before_benchmark
        torch.backends.cuda.matmul.allow_tf32 = before_tf32_matmul
        torch.backends.cudnn.allow_tf32 = before_tf32_cudnn


def test_apply_backend_speed_flags_opt_in_sets_flags() -> None:
    before_benchmark = torch.backends.cudnn.benchmark
    before_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    before_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        apply_backend_speed_flags(cudnn_benchmark=True, tf32=True)
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cudnn.benchmark = before_benchmark
        torch.backends.cuda.matmul.allow_tf32 = before_tf32_matmul
        torch.backends.cudnn.allow_tf32 = before_tf32_cudnn


def _base_train_config(dataset: Path, manifest: Path, out_dir: Path, **overrides: object) -> ChallengerTrainConfig:
    kwargs = {
        "dataset_dir": dataset,
        "training_manifest": manifest,
        "out_dir": out_dir,
        "batch_size": 2,
        "max_steps": 5,
        "device": "cpu",
        "seed": 7,
        "base_channels": 8,
        "channel_mults": "1,2",
        "res_blocks_per_level": 1,
        "embed_dim": 8,
        "sample_every": 0,
        "save_every": 0,
        "validation_mode": "none",
    }
    kwargs.update(overrides)
    return ChallengerTrainConfig(**kwargs)


def test_metrics_every_logs_only_synced_steps_and_keeps_final_step(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "metrics_every_run"
    report = run_challenger_training(_base_train_config(dataset, manifest, run_dir, metrics_every=2))
    lines = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in lines] == [2, 4, 5]
    assert report["steps_completed"] == 5
    assert report["last_step_loss_components"]
    assert isinstance(report["final_train_loss"], float)


def test_metrics_every_one_matches_default_every_step_logging(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "metrics_every_default_run"
    run_challenger_training(_base_train_config(dataset, manifest, run_dir, metrics_every=1))
    lines = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in lines] == [1, 2, 3, 4, 5]


def test_metrics_every_rejects_values_below_one(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "metrics_every_invalid_run"
    with pytest.raises(ValueError):
        run_challenger_training(_base_train_config(dataset, manifest, run_dir, metrics_every=0))


def test_eval_max_batches_zero_matches_full_pass_report_shape(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "eval_full_run"
    report = run_challenger_training(_base_train_config(dataset, manifest, run_dir, eval_max_batches=0))
    assert isinstance(report["initial_train_loss"], float)
    assert isinstance(report["final_train_loss"], float)


def test_eval_max_batches_caps_batches_without_crashing(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_dir = tmp_path / "eval_capped_run"
    report = run_challenger_training(_base_train_config(dataset, manifest, run_dir, eval_max_batches=1))
    assert isinstance(report["initial_train_loss"], float)
    assert isinstance(report["final_train_loss"], float)
    assert report["initial_train_loss_components"]
    assert report["final_train_loss_components"]


def test_fused_adamw_and_backend_flags_default_off_train_identically(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_a = tmp_path / "defaults_a"
    run_b = tmp_path / "defaults_b"
    report_a = run_challenger_training(_base_train_config(dataset, manifest, run_a))
    report_b = run_challenger_training(
        _base_train_config(dataset, manifest, run_b, fused_adamw=False, cudnn_benchmark=False, tf32=False)
    )
    assert report_a["final_train_loss"] == pytest.approx(report_b["final_train_loss"])
    assert report_a["last_step_loss"] == pytest.approx(report_b["last_step_loss"])


def test_speed_defaults_produce_bit_identical_train_metrics_jsonl(tmp_path: Path) -> None:
    """CPU A/B: explicit defaults for every new speed knob vs. omitting them entirely.

    ``run_a`` exercises ``ChallengerTrainConfig``'s bare dataclass defaults (as any
    caller written before these options existed would); ``run_b`` passes the same
    values explicitly through the new fields (metrics_every=1, fused_adamw=False,
    cudnn_benchmark=False, tf32=False, eval_max_batches=0). Every field in
    train_metrics.jsonl besides the wall-clock ``elapsed_seconds`` must match
    exactly on CPU, since all five knobs are no-ops at these values.
    """
    dataset, manifest = _dataset_with_manifest(tmp_path)
    run_a = tmp_path / "ab_a"
    run_b = tmp_path / "ab_b"
    run_challenger_training(_base_train_config(dataset, manifest, run_a, max_steps=20))
    run_challenger_training(
        _base_train_config(
            dataset,
            manifest,
            run_b,
            max_steps=20,
            metrics_every=1,
            fused_adamw=False,
            cudnn_benchmark=False,
            tf32=False,
            eval_max_batches=0,
        )
    )
    lines_a = [json.loads(line) for line in (run_a / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    lines_b = [json.loads(line) for line in (run_b / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines_a) == len(lines_b)
    for row_a, row_b in zip(lines_a, lines_b, strict=False):
        row_a.pop("elapsed_seconds")
        row_b.pop("elapsed_seconds")
        assert row_a == row_b


def test_cli_generator_challenger_threads_speed_flags_into_config_json(tmp_path: Path) -> None:
    dataset, manifest = _dataset_with_manifest(tmp_path)
    out = tmp_path / "cli_run"
    train_cli(
        [
            "generator-challenger",
            "--dataset",
            str(dataset),
            "--training-manifest",
            str(manifest),
            "--out",
            str(out),
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--device",
            "cpu",
            "--base-channels",
            "8",
            "--channel-mults",
            "1,2",
            "--res-blocks-per-level",
            "1",
            "--embed-dim",
            "8",
            "--sample-every",
            "0",
            "--save-every",
            "0",
            "--validation-mode",
            "none",
            "--metrics-every",
            "3",
            "--fused-adamw",
            "--cudnn-benchmark",
            "--tf32",
            "--eval-max-batches",
            "1",
        ]
    )
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["metrics_every"] == 3
    assert config["fused_adamw"] is True
    assert config["cudnn_benchmark"] is True
    assert config["tf32"] is True
    assert config["eval_max_batches"] == 1
