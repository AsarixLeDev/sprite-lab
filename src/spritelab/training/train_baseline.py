"""Tiny semantic-manifest conditional autoencoder training loop."""

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

from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch, describe_array, read_jsonl
from spritelab.training.eval_baseline import evaluate_model, move_batch_to_device, resolve_device, save_reconstruction_sheet
from spritelab.training.losses import sprite_reconstruction_loss
from spritelab.training.models import SpriteCondAutoencoder
from spritelab.training.optim_utils import (
    amp_autocast,
    build_lr_scheduler,
    clip_gradients,
    dataloader_perf_kwargs,
    device_type,
)
from spritelab.training.tokenization import SpriteTextTokenizer


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab baseline training.")
    return torch


@dataclass(frozen=True)
class BaselineTrainConfig:
    dataset_dir: Path
    training_manifest: Path
    out_dir: Path
    batch_size: int = 16
    max_steps: int = 200
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 1337
    overfit_batches: int = 0
    max_records: int | None = None
    caption_max_length: int = 32
    semantic_max_length: int = 48
    hidden_dim: int = 48
    num_workers: int = 0
    # Opt-in speed/quality knobs; defaults keep training numerically identical.
    amp: bool = False
    grad_clip: float = 0.0
    lr_schedule: str = "none"
    lr_warmup_steps: int = 0


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


def run_baseline_training(config: BaselineTrainConfig) -> dict[str, Any]:
    th = _require_torch()
    started = time.perf_counter()
    set_deterministic_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(config.training_manifest)
    train_rows = [row for row in manifest_rows if row.get("split") == "train"]
    tokenizer = SpriteTextTokenizer.build_from_records(train_rows or manifest_rows, max_length=config.caption_max_length)
    tokenizer.save(out_dir / "vocab.json")

    shared_npz_cache: dict[str, Any] = {}
    train_dataset = SpriteTrainingDataset(
        config.dataset_dir,
        config.training_manifest,
        split="train",
        max_records=config.max_records,
        tokenizer=tokenizer,
        caption_max_length=config.caption_max_length,
        semantic_max_length=config.semantic_max_length,
        npz_cache=shared_npz_cache,
    )
    val_dataset = SpriteTrainingDataset(
        config.dataset_dir,
        config.training_manifest,
        split="val",
        tokenizer=tokenizer,
        caption_max_length=config.caption_max_length,
        semantic_max_length=config.semantic_max_length,
        npz_cache=shared_npz_cache,
    )
    if len(train_dataset) == 0:
        raise ValueError("training manifest has no train records")

    train_source = _overfit_subset(train_dataset, config.batch_size, config.overfit_batches)
    shuffle = config.overfit_batches <= 0
    generator = th.Generator().manual_seed(config.seed)
    loader_perf = dataloader_perf_kwargs(device, num_workers=config.num_workers)
    train_loader = th.utils.data.DataLoader(
        train_source,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_sprite_batch,
        **loader_perf,
    )
    eval_train_loader = th.utils.data.DataLoader(
        train_source,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_sprite_batch,
        **loader_perf,
    )
    val_loader = th.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_sprite_batch,
        **loader_perf,
    )

    first_sample = train_dataset[0]
    num_palette_slots = int(first_sample["palette_mask"].shape[0])
    num_roles = max(16, _max_role_id(train_dataset) + 1)
    max_category = max(1, _max_category_id(train_dataset), _max_category_id(val_dataset))
    model_config = {
        "num_palette_slots": num_palette_slots,
        "vocab_size": len(tokenizer),
        "num_roles": num_roles,
        "num_categories": max(64, max_category + 2),
        "hidden_dim": int(config.hidden_dim),
        "pad_token_id": tokenizer.pad_id,
        "predict_roles": True,
    }
    model = SpriteCondAutoencoder(**model_config).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=config.lr_schedule,
        max_steps=config.max_steps,
        warmup_steps=config.lr_warmup_steps,
    )
    non_blocking = device_type(device) == "cuda"

    config_json = {
        **{key: _jsonable(value) for key, value in asdict(config).items()},
        "model_config": model_config,
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
    }
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    initial_metrics = evaluate_model(model, eval_train_loader, device=device)
    metrics_path = out_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    step = 0
    last_loss = initial_metrics["loss"]
    model.train()
    # Keep the metrics file open for the whole run instead of reopening it every
    # step; the line content and order are unchanged, so the file is identical.
    metrics_handle = metrics_path.open("a", encoding="utf-8")
    try:
        while step < config.max_steps:
            for batch in train_loader:
                if step >= config.max_steps:
                    break
                batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
                with amp_autocast(device, config.amp):
                    outputs = model(
                        index_map=batch["index_map"],
                        alpha=batch["alpha"],
                        role_map=batch["role_map"],
                        caption_tokens=batch["caption_tokens"],
                        semantic_tokens=batch["semantic_tokens"],
                        category_id=batch["category_id"],
                    )
                    losses = sprite_reconstruction_loss(outputs, batch)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                clip_gradients(model, config.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                last_loss = float(losses["loss"].detach().cpu())
                _write_jsonl_line(
                    metrics_handle,
                    {
                        "step": step,
                        "loss": last_loss,
                        "loss_alpha": float(losses["loss_alpha"].detach().cpu()),
                        "loss_index": float(losses["loss_index"].detach().cpu()),
                        "loss_role": float(losses["loss_role"].detach().cpu()),
                    },
                )
            if len(train_loader) == 0:
                break
    finally:
        metrics_handle.close()

    final_metrics = evaluate_model(model, eval_train_loader, device=device)
    val_metrics = evaluate_model(model, val_loader, device=device) if len(val_dataset) else None

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "vocab": tokenizer.to_json_dict(),
        "train_config": config_json,
        "step": step,
    }
    th.save(checkpoint, out_dir / "checkpoint_last.pt")
    if val_metrics is not None:
        th.save(checkpoint, out_dir / "checkpoint_best.pt")

    preview_batch = next(iter(eval_train_loader))
    preview_batch = move_batch_to_device(preview_batch, device)
    model.eval()
    with th.no_grad():
        preview_outputs = model(
            index_map=preview_batch["index_map"],
            alpha=preview_batch["alpha"],
            role_map=preview_batch["role_map"],
            caption_tokens=preview_batch["caption_tokens"],
            semantic_tokens=preview_batch["semantic_tokens"],
            category_id=preview_batch["category_id"],
        )
    save_reconstruction_sheet(preview_batch, preview_outputs, out_dir / "reconstructions.png")

    report = {
        "dataset": str(config.dataset_dir),
        "training_manifest": str(config.training_manifest),
        "model_config": model_config,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "steps_completed": step,
        "device": str(device),
        "overfit_batches": config.overfit_batches,
        "train_records": len(train_dataset),
        "effective_train_records": len(train_source),
        "val_records": len(val_dataset),
        "initial_train_loss": initial_metrics["loss"],
        "final_train_loss": final_metrics["loss"],
        "last_step_loss": last_loss,
        "val_loss": None if val_metrics is None else val_metrics["loss"],
        "loss_decreased": final_metrics["loss"] < initial_metrics["loss"],
        "elapsed_seconds": time.perf_counter() - started,
        "warnings": [],
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def inspect_training_data(
    *,
    dataset_dir: str | Path,
    training_manifest: str | Path,
    split: str | None = None,
    batch_size: int = 4,
    max_records: int | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    training_manifest = Path(training_manifest)
    records = read_jsonl(training_manifest)
    tokenizer = SpriteTextTokenizer.build_from_records(records)
    filtered_records = [record for record in records if split is None or record.get("split") == split]
    if max_records is not None:
        filtered_records = filtered_records[: max(0, int(max_records))]
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[str(record.get("split", ""))] = split_counts.get(str(record.get("split", "")), 0) + 1

    npz_summary: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        path = dataset_dir / f"{split_name}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            npz_summary[split_name] = {key: describe_array(data[key]) for key in data.files}

    batch_shapes: dict[str, Any] = {}
    warnings: list[str] = []
    if torch is None:
        warnings.append("PyTorch is unavailable; skipped DataLoader batch tensor shape inspection.")
    elif filtered_records:
        th = torch
        dataset = SpriteTrainingDataset(
            dataset_dir,
            training_manifest,
            split=split,
            max_records=max_records,
            tokenizer=tokenizer,
        )
        loader = th.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), collate_fn=collate_sprite_batch)
        batch = next(iter(loader))
        for key, value in batch.items():
            if isinstance(value, th.Tensor):
                batch_shapes[key] = list(value.shape)

    summary = {
        "records": len(records),
        "loaded_records": len(filtered_records),
        "splits": split_counts,
        "npz": npz_summary,
        "caption_examples": [str(record.get("caption", "")) for record in records[:5]],
        "token_vocabulary_size": len(tokenizer),
        "batch_tensor_shapes": batch_shapes,
        "warnings": warnings,
    }
    return summary


def print_inspection(summary: dict[str, Any]) -> None:
    print(f"records: {summary['records']}")
    print(f"loaded_records: {summary['loaded_records']}")
    print("splits:")
    for split, count in sorted(summary["splits"].items()):
        print(f"  {split}: {count}")
    for split, arrays in summary["npz"].items():
        print(f"{split}.npz:")
        for key in ("alpha", "index_map", "palette", "palette_mask", "role_map", "category_id", "sprite_id"):
            if key in arrays:
                desc = arrays[key]
                range_text = ""
                if "min" in desc and "max" in desc:
                    range_text = f" range=[{desc['min']}, {desc['max']}]"
                print(f"  {key}: shape={desc['shape']} dtype={desc['dtype']}{range_text}")
    print("caption examples:")
    for caption in summary["caption_examples"]:
        print(f"  - {caption}")
    print(f"token vocabulary size: {summary['token_vocabulary_size']}")
    print("batch tensor shapes:")
    for key, shape in sorted(summary["batch_tensor_shapes"].items()):
        print(f"  {key}: {shape}")
    warnings = summary.get("warnings") or []
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def _overfit_subset(dataset: Any, batch_size: int, overfit_batches: int) -> Any:
    th = _require_torch()
    if overfit_batches <= 0:
        return dataset
    count = min(len(dataset), max(1, int(batch_size) * int(overfit_batches)))
    return th.utils.data.Subset(dataset, list(range(count)))


def _max_role_id(dataset: SpriteTrainingDataset) -> int:
    max_role = 0
    for npz_file in sorted({str(record.get("npz_file") or f"{record.get('split', '')}.npz") for record in dataset.records}):
        arrays = dataset._load_npz(npz_file)
        if arrays["role_map"].size:
            max_role = max(max_role, int(np.asarray(arrays["role_map"]).max()))
    return max_role


def _max_category_id(dataset: SpriteTrainingDataset) -> int:
    max_category = 0
    for npz_file in sorted({str(record.get("npz_file") or f"{record.get('split', '')}.npz") for record in dataset.records}):
        arrays = dataset._load_npz(npz_file)
        if arrays["category_id"].size:
            max_category = max(max_category, int(np.asarray(arrays["category_id"]).max()))
    return max_category


def _write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the semantic sprite reconstruction baseline.")
    parser.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overfit-batches", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False, help="Enable bf16 autocast (CUDA only).")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Clip gradient norm; 0 disables.")
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parsed = parser.parse_args(argv)
    report = run_baseline_training(BaselineTrainConfig(**vars(parsed)))
    print(f"Initial train loss: {report['initial_train_loss']:.6f}")
    print(f"Final train loss: {report['final_train_loss']:.6f}")
    if report["val_loss"] is not None:
        print(f"Val loss: {report['val_loss']:.6f}")
    print(f"Outputs written to {parsed.out_dir}")
