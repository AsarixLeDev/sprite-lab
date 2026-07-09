"""Evaluation helpers for the semantic sprite reconstruction baseline."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]

from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch
from spritelab.training.losses import sprite_reconstruction_loss
from spritelab.training.models import SpriteCondAutoencoder
from spritelab.training.tokenization import SpriteTextTokenizer


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab baseline evaluation.")
    return torch


def resolve_device(device: str) -> Any:
    th = _require_torch()
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


def evaluate_baseline_checkpoint(
    *,
    dataset_dir: str | Path,
    training_manifest: str | Path,
    checkpoint: str | Path,
    split: str = "val",
    out_dir: str | Path,
    batch_size: int = 16,
    device: str = "cpu",
    max_records: int | None = None,
) -> dict[str, Any]:
    th = _require_torch()
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        ckpt = th.load(Path(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = th.load(Path(checkpoint), map_location="cpu")
    tokenizer_data = ckpt.get("vocab")
    if isinstance(tokenizer_data, Mapping):
        vocab_path = out_path / "vocab_from_checkpoint.json"
        vocab_path.write_text(json.dumps(dict(tokenizer_data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tokenizer = SpriteTextTokenizer.load(vocab_path)
    else:
        tokenizer = SpriteTextTokenizer.load(Path(checkpoint).parent / "vocab.json")

    dataset = SpriteTrainingDataset(
        Path(dataset_dir),
        Path(training_manifest),
        split=split,
        max_records=max_records,
        tokenizer=tokenizer,
    )
    loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sprite_batch)

    model_config = dict(ckpt["model_config"])
    model = SpriteCondAutoencoder(**model_config).to(resolve_device(device))
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = evaluate_model(model, loader, device=resolve_device(device))

    eval_prompts_path = Path(dataset_dir) / "eval_prompts.jsonl"
    prompt_count = 0
    if eval_prompts_path.is_file():
        prompt_count = sum(1 for line in eval_prompts_path.read_text(encoding="utf-8").splitlines() if line.strip())

    report = {
        "dataset": str(dataset_dir),
        "training_manifest": str(training_manifest),
        "checkpoint": str(checkpoint),
        "split": split,
        "records": len(dataset),
        "batch_size": int(batch_size),
        "device": str(resolve_device(device)),
        "loss": metrics["loss"],
        "loss_alpha": metrics["loss_alpha"],
        "loss_index": metrics["loss_index"],
        "loss_role": metrics["loss_role"],
        "eval_prompt_count": prompt_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out_path / "eval_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if len(dataset):
        first_batch = next(iter(loader))
        first_batch = move_batch_to_device(first_batch, resolve_device(device))
        model.eval()
        with th.no_grad():
            outputs = model(
                index_map=first_batch["index_map"],
                alpha=first_batch["alpha"],
                role_map=first_batch["role_map"],
                caption_tokens=first_batch["caption_tokens"],
                semantic_tokens=first_batch["semantic_tokens"],
                category_id=first_batch["category_id"],
            )
        save_reconstruction_sheet(first_batch, outputs, out_path / "reconstructions.png")
    return report


def move_batch_to_device(batch: dict[str, Any], device: Any, *, non_blocking: bool = False) -> dict[str, Any]:
    th = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if isinstance(value, th.Tensor) else value
    return moved


def evaluate_model(model: Any, loader: Any, *, device: Any) -> dict[str, float]:
    th = _require_torch()
    totals = {"loss": 0.0, "loss_alpha": 0.0, "loss_index": 0.0, "loss_role": 0.0}
    count = 0
    model.eval()
    with th.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                index_map=batch["index_map"],
                alpha=batch["alpha"],
                role_map=batch["role_map"],
                caption_tokens=batch["caption_tokens"],
                semantic_tokens=batch["semantic_tokens"],
                category_id=batch["category_id"],
            )
            losses = sprite_reconstruction_loss(outputs, batch)
            batch_size = int(batch["index_map"].shape[0])
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_size
            count += batch_size
    model.train()
    if count == 0:
        return dict.fromkeys(totals, 0.0)
    return {key: value / count for key, value in totals.items()}


def save_reconstruction_sheet(
    batch: dict[str, Any], outputs: dict[str, Any], path: str | Path, *, max_items: int = 16, scale: int = 6
) -> None:
    """Write target/prediction/alpha contact sheet when Pillow is available."""

    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - Pillow is a base dependency.
        return

    predictions = outputs["index_logits"].argmax(dim=1).detach().cpu()
    alphas = (outputs["alpha_logits"].sigmoid() > 0.5).squeeze(1).detach().cpu()
    targets = batch["index_map"].detach().cpu()
    target_alpha = batch["alpha"].squeeze(1).detach().cpu()
    palettes = batch["palette_u8"].detach().cpu()
    count = min(int(targets.shape[0]), max_items)
    if count <= 0:
        return

    def render(index_map: Any, palette: Any, alpha: Any) -> Image.Image:
        index_np = index_map.numpy().astype("int64")
        palette_np = palette.numpy().astype("uint8")
        alpha_np = alpha.numpy().astype("uint8")
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        clipped = np.clip(index_np, 0, palette_np.shape[0] - 1)
        rgba[..., :3] = palette_np[clipped]
        rgba[..., 3] = np.where(alpha_np > 0, 255, 0).astype(np.uint8)
        return Image.fromarray(rgba, mode="RGBA")

    def render_alpha(alpha: Any) -> Image.Image:
        alpha_np = alpha.numpy().astype("uint8")
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[..., :3] = np.where(alpha_np[..., None] > 0, 255, 0).astype(np.uint8)
        rgba[..., 3] = 255
        return Image.fromarray(rgba, mode="RGBA")

    import numpy as np

    cell = 32 * scale
    padding = scale
    columns = 3
    sheet = Image.new(
        "RGBA", (columns * cell + (columns + 1) * padding, count * cell + (count + 1) * padding), (36, 36, 40, 255)
    )
    for row in range(count):
        images = [
            render(targets[row], palettes[row], target_alpha[row]),
            render(predictions[row], palettes[row], alphas[row]),
            render_alpha(target_alpha[row]),
        ]
        top = padding + row * (cell + padding)
        for col, image in enumerate(images):
            left = padding + col * (cell + padding)
            sheet.paste(image.resize((cell, cell), Image.NEAREST), (left, top))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a baseline sprite reconstruction checkpoint.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-records", type=int)
    parsed = parser.parse_args(argv)
    report = evaluate_baseline_checkpoint(
        dataset_dir=parsed.dataset,
        training_manifest=parsed.training_manifest,
        checkpoint=parsed.checkpoint,
        split=parsed.split,
        out_dir=parsed.out,
        batch_size=parsed.batch_size,
        device=parsed.device,
        max_records=parsed.max_records,
    )
    print(f"Evaluated {report['records']} {report['split']} records.")
    print(f"Loss: {report['loss']:.6f}")
