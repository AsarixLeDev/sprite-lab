"""Training loop for the caption-conditioned RGBA sprite generator."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]

from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch, read_jsonl
from spritelab.training.eval_baseline import move_batch_to_device, resolve_device
from spritelab.training.eval_generator import evaluate_generator_model
from spritelab.training.conditioning import (
    CONDITIONING_MODES,
    DEFAULT_CONDITIONING_MODE,
    apply_conditioning_mode,
    validate_conditioning_mode,
)
from spritelab.training.generator_losses import rgba_generator_loss
from spritelab.training.generator_models import TinyCaptionSpriteGenerator
from spritelab.training.rgba import save_rgba_contact_sheet
from spritelab.training.tokenization import SpriteTextTokenizer


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab generator training.")
    return torch


@dataclass(frozen=True)
class GeneratorTrainConfig:
    dataset_dir: Path
    training_manifest: Path
    out_dir: Path
    split: str = "train"
    batch_size: int = 32
    max_steps: int = 1000
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 123
    overfit_batches: int = 0
    num_workers: int = 0
    latent_dim: int = 32
    embed_dim: int = 32
    hidden_channels: int = 48
    sample_every: int = 20
    save_every: int = 100
    caption_policy_filter: str | None = None
    max_records: int | None = None
    conditioning_mode: str = DEFAULT_CONDITIONING_MODE
    caption_max_length: int = 32
    semantic_max_length: int = 48
    border_alpha_weight: float = 0.0
    alpha_coverage_weight: float = 0.0
    alpha_coverage_min: float | None = None
    alpha_coverage_max: float | None = None
    center_weight: float = 0.0
    margin_band_weight: float = 0.0
    margin_band_size: int = 2


def set_deterministic_seed(seed: int) -> None:
    th = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    try:
        th.use_deterministic_algorithms(True)
    except Exception:
        pass


def run_generator_training(config: GeneratorTrainConfig) -> dict[str, Any]:
    th = _require_torch()
    started = time.perf_counter()
    set_deterministic_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditioning_mode = validate_conditioning_mode(config.conditioning_mode)

    manifest_rows = read_jsonl(config.training_manifest)
    token_rows = [row for row in manifest_rows if _matches_caption_policy(row, config.caption_policy_filter)]
    train_rows = [row for row in token_rows if row.get("split") == config.split]
    tokenizer = SpriteTextTokenizer.build_from_records(
        train_rows or token_rows or manifest_rows,
        max_length=config.caption_max_length,
    )
    tokenizer.save(out_dir / "vocab.json")

    train_dataset = SpriteTrainingDataset(
        config.dataset_dir,
        config.training_manifest,
        split=config.split,
        max_records=config.max_records,
        tokenizer=tokenizer,
        caption_max_length=config.caption_max_length,
        semantic_max_length=config.semantic_max_length,
        caption_policy_filter=config.caption_policy_filter,
    )
    val_dataset = SpriteTrainingDataset(
        config.dataset_dir,
        config.training_manifest,
        split="val",
        tokenizer=tokenizer,
        caption_max_length=config.caption_max_length,
        semantic_max_length=config.semantic_max_length,
        caption_policy_filter=config.caption_policy_filter,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"training manifest has no records for split {config.split!r}")

    train_source = _overfit_subset(train_dataset, config.batch_size, config.overfit_batches)
    shuffle = config.overfit_batches <= 0
    loader_generator = th.Generator().manual_seed(config.seed)
    train_loader = th.utils.data.DataLoader(
        train_source,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=loader_generator if shuffle else None,
        num_workers=max(0, int(config.num_workers)),
        collate_fn=collate_sprite_batch,
    )
    eval_train_loader = th.utils.data.DataLoader(
        train_source,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(0, int(config.num_workers)),
        collate_fn=collate_sprite_batch,
    )
    val_loader = th.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(0, int(config.num_workers)),
        collate_fn=collate_sprite_batch,
    )

    model_config = {
        "vocab_size": len(tokenizer),
        "embed_dim": int(config.embed_dim),
        "latent_dim": int(config.latent_dim),
        "hidden_channels": int(config.hidden_channels),
        "pad_token_id": tokenizer.pad_id,
    }
    model = TinyCaptionSpriteGenerator(**model_config).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)

    config_json = {
        **{key: _jsonable(value) for key, value in asdict(config).items()},
        "conditioning_mode": conditioning_mode,
        "model_config": model_config,
        "train_records": len(train_dataset),
        "effective_train_records": len(train_source),
        "val_records": len(val_dataset),
    }
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    initial_metrics = evaluate_generator_model(
        model,
        eval_train_loader,
        device=device,
        conditioning_mode=conditioning_mode,
        pad_token_id=tokenizer.pad_id,
    )
    metrics_path = out_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    preview_batch_cpu = next(iter(eval_train_loader))
    preview_batch = move_batch_to_device(preview_batch_cpu, device)
    step = 0
    last_loss = initial_metrics["loss"]
    last_loss_components: dict[str, float] = {}
    framing_config = _framing_config(config)
    model.train()
    while step < config.max_steps:
        for batch in train_loader:
            if step >= config.max_steps:
                break
            batch = move_batch_to_device(batch, device)
            noise = _training_noise(model, batch, device=device, overfit=config.overfit_batches > 0)
            model_inputs = apply_conditioning_mode(
                caption_tokens=batch["caption_tokens"],
                semantic_tokens=batch["semantic_tokens"],
                mode=conditioning_mode,
                pad_token_id=tokenizer.pad_id,
            )
            outputs = model(
                **model_inputs,
                noise=noise,
            )
            losses = rgba_generator_loss(outputs, batch, framing_config=framing_config)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            step += 1
            last_loss = float(losses["loss"].detach().cpu())
            loss_metrics = _loss_metrics(losses)
            last_loss_components = dict(loss_metrics)
            _append_jsonl(
                metrics_path,
                {
                    "step": step,
                    **loss_metrics,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            if config.sample_every > 0 and step % int(config.sample_every) == 0:
                _write_sample_sheet(
                    model,
                    preview_batch,
                    out_dir / f"samples_step_{step:06d}.png",
                    device=device,
                    conditioning_mode=conditioning_mode,
                    pad_token_id=tokenizer.pad_id,
                )
            if config.save_every > 0 and step % int(config.save_every) == 0:
                _save_checkpoint(
                    out_dir / f"checkpoint_step_{step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    config_json=config_json,
                    step=step,
                )
        if len(train_loader) == 0:
            break

    final_metrics = evaluate_generator_model(
        model,
        eval_train_loader,
        device=device,
        conditioning_mode=conditioning_mode,
        pad_token_id=tokenizer.pad_id,
    )
    val_metrics = (
        evaluate_generator_model(
            model,
            val_loader,
            device=device,
            conditioning_mode=conditioning_mode,
            pad_token_id=tokenizer.pad_id,
        )
        if len(val_dataset)
        else None
    )
    _save_checkpoint(
        out_dir / "checkpoint_last.pt",
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config_json=config_json,
        step=step,
    )
    _write_sample_sheet(
        model,
        preview_batch,
        out_dir / "samples_final.png",
        device=device,
        conditioning_mode=conditioning_mode,
        pad_token_id=tokenizer.pad_id,
    )

    report = {
        "dataset": str(config.dataset_dir),
        "training_manifest": str(config.training_manifest),
        "model_config": model_config,
        "conditioning_mode": conditioning_mode,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "steps_completed": step,
        "device": str(device),
        "split": config.split,
        "overfit_batches": config.overfit_batches,
        "train_records": len(train_dataset),
        "effective_train_records": len(train_source),
        "val_records": len(val_dataset),
        "initial_train_loss": initial_metrics["loss"],
        "final_train_loss": final_metrics["loss"],
        "loss_decrease": initial_metrics["loss"] - final_metrics["loss"],
        "last_step_loss": last_loss,
        "last_step_loss_components": last_loss_components,
        "framing_loss_config": framing_config,
        "val_loss": None if val_metrics is None else val_metrics["loss"],
        "loss_decreased": final_metrics["loss"] < initial_metrics["loss"],
        "elapsed_seconds": time.perf_counter() - started,
        "warnings": [],
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _framing_config(config: GeneratorTrainConfig) -> dict[str, Any]:
    return {
        "border_alpha_weight": float(config.border_alpha_weight),
        "alpha_coverage_weight": float(config.alpha_coverage_weight),
        "alpha_coverage_min": None if config.alpha_coverage_min is None else float(config.alpha_coverage_min),
        "alpha_coverage_max": None if config.alpha_coverage_max is None else float(config.alpha_coverage_max),
        "center_weight": float(config.center_weight),
        "margin_band_weight": float(config.margin_band_weight),
        "margin_band_size": int(config.margin_band_size),
    }


def _loss_metrics(losses: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in losses.items():
        if not key.startswith("loss"):
            continue
        if hasattr(value, "detach"):
            metrics[key] = float(value.detach().cpu())
    return metrics


def _training_noise(model: TinyCaptionSpriteGenerator, batch: dict[str, Any], *, device: Any, overfit: bool) -> Any:
    th = _require_torch()
    batch_size = int(batch["caption_tokens"].shape[0])
    if overfit:
        return th.zeros(batch_size, int(model.latent_dim), device=device)
    return model.sample_noise(batch_size, device=device)


def _write_sample_sheet(
    model: TinyCaptionSpriteGenerator,
    batch: dict[str, Any],
    path: Path,
    *,
    device: Any,
    conditioning_mode: str,
    pad_token_id: int,
) -> None:
    th = _require_torch()
    model.eval()
    with th.no_grad():
        noise = th.zeros(int(batch["caption_tokens"].shape[0]), int(model.latent_dim), device=device)
        model_inputs = apply_conditioning_mode(
            caption_tokens=batch["caption_tokens"],
            semantic_tokens=batch["semantic_tokens"],
            mode=conditioning_mode,
            pad_token_id=pad_token_id,
        )
        outputs = model(
            **model_inputs,
            noise=noise,
        )
    save_rgba_contact_sheet(outputs=outputs, batch=batch, path=path)
    model.train()


def _save_checkpoint(
    path: Path,
    *,
    model: TinyCaptionSpriteGenerator,
    optimizer: Any,
    tokenizer: SpriteTextTokenizer,
    config_json: dict[str, Any],
    step: int,
) -> None:
    th = _require_torch()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config(),
        "vocab": tokenizer.to_json_dict(),
        "train_config": config_json,
        "conditioning_mode": str(config_json.get("conditioning_mode", DEFAULT_CONDITIONING_MODE)),
        "step": int(step),
        "checkpoint_type": "caption_rgba_generator_v0",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    th.save(checkpoint, path)


def _overfit_subset(dataset: Any, batch_size: int, overfit_batches: int) -> Any:
    th = _require_torch()
    if overfit_batches <= 0:
        return dataset
    count = min(len(dataset), max(1, int(batch_size) * int(overfit_batches)))
    return th.utils.data.Subset(dataset, list(range(count)))


def _matches_caption_policy(record: dict[str, Any], caption_policy_filter: str | None) -> bool:
    if not caption_policy_filter:
        return True
    audit = record.get("audit") if isinstance(record.get("audit"), dict) else {}
    return str(audit.get("caption_policy", "")) == str(caption_policy_filter)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the caption-conditioned RGBA sprite generator.")
    parser.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overfit-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--caption-policy-filter")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default=DEFAULT_CONDITIONING_MODE)
    parser.add_argument("--border-alpha-weight", type=float, default=0.0)
    parser.add_argument("--alpha-coverage-weight", type=float, default=0.0)
    parser.add_argument("--alpha-coverage-min", type=float)
    parser.add_argument("--alpha-coverage-max", type=float)
    parser.add_argument("--center-weight", type=float, default=0.0)
    parser.add_argument("--margin-band-weight", type=float, default=0.0)
    parser.add_argument("--margin-band-size", type=int, default=2)
    parsed = parser.parse_args(argv)
    report = run_generator_training(GeneratorTrainConfig(**vars(parsed)))
    print(f"Initial train loss: {report['initial_train_loss']:.6f}")
    print(f"Final train loss: {report['final_train_loss']:.6f}")
    if report["val_loss"] is not None:
        print(f"Val loss: {report['val_loss']:.6f}")
    print(f"Outputs written to {parsed.out_dir}")
