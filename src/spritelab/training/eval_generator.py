"""Evaluation and prompt sampling for the caption-conditioned RGBA generator."""

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

from spritelab.training.checkpoint_io import load_checkpoint as _load_checkpoint
from spritelab.training.checkpoint_io import tokenizer_from_checkpoint as _tokenizer_from_checkpoint
from spritelab.training.conditioning import (
    DEFAULT_CONDITIONING_MODE,
    apply_conditioning_mode,
    checkpoint_conditioning_mode,
    checkpoint_semantic_max_length,
    validate_conditioning_mode,
)
from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch
from spritelab.training.eval_baseline import move_batch_to_device, resolve_device
from spritelab.training.generator_losses import rgba_generator_loss
from spritelab.training.generator_models import TinyCaptionSpriteGenerator
from spritelab.training.rgba import save_rgba_contact_sheet
from spritelab.training.tokenization import SpriteTextTokenizer


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab generator evaluation.")
    return torch


def evaluate_generator_checkpoint(
    *,
    checkpoint: str | Path,
    out_dir: str | Path,
    dataset_dir: str | Path | None = None,
    training_manifest: str | Path | None = None,
    split: str = "val",
    prompts: str | Path | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    max_records: int | None = None,
) -> dict[str, Any]:
    th = _require_torch()
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if (dataset_dir is None) != (training_manifest is None):
        raise ValueError("--dataset and --training-manifest must be provided together")
    if dataset_dir is None and prompts is None:
        raise ValueError("provide either --dataset/--training-manifest or --prompts")

    ckpt = _load_checkpoint(checkpoint)
    tokenizer = _tokenizer_from_checkpoint(ckpt)
    conditioning_mode = checkpoint_conditioning_mode(ckpt)
    semantic_max_length = checkpoint_semantic_max_length(ckpt)
    model = TinyCaptionSpriteGenerator(**dict(ckpt["model_config"])).to(resolve_device(device))
    model.load_state_dict(ckpt["model_state_dict"])

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "dataset": None if dataset_dir is None else str(dataset_dir),
        "training_manifest": None if training_manifest is None else str(training_manifest),
        "split": split,
        "records": 0,
        "batch_size": int(batch_size),
        "device": str(resolve_device(device)),
        "conditioning_mode": conditioning_mode,
        "loss": None,
        "loss_alpha": None,
        "loss_rgb_opaque": None,
        "loss_rgb_all": None,
        "prompt_count": 0,
        "prompt_samples_written": 0,
        "elapsed_seconds": None,
    }

    if dataset_dir is not None and training_manifest is not None:
        dataset = SpriteTrainingDataset(
            Path(dataset_dir),
            Path(training_manifest),
            split=split,
            max_records=max_records,
            tokenizer=tokenizer,
            caption_max_length=tokenizer.max_length,
            semantic_max_length=semantic_max_length,
        )
        loader = th.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_sprite_batch,
        )
        metrics = evaluate_generator_model(
            model,
            loader,
            device=resolve_device(device),
            conditioning_mode=conditioning_mode,
            pad_token_id=tokenizer.pad_id,
        )
        report.update(
            {
                "records": len(dataset),
                "loss": metrics["loss"],
                "loss_alpha": metrics["loss_alpha"],
                "loss_rgb_opaque": metrics["loss_rgb_opaque"],
                "loss_rgb_all": metrics["loss_rgb_all"],
            }
        )
        if len(dataset):
            first_batch = move_batch_to_device(next(iter(loader)), resolve_device(device))
            model.eval()
            with th.no_grad():
                noise = th.zeros(
                    int(first_batch["caption_tokens"].shape[0]),
                    int(model.latent_dim),
                    device=resolve_device(device),
                )
                model_inputs = apply_conditioning_mode(
                    caption_tokens=first_batch["caption_tokens"],
                    semantic_tokens=first_batch["semantic_tokens"],
                    mode=conditioning_mode,
                    pad_token_id=tokenizer.pad_id,
                )
                outputs = model(
                    **model_inputs,
                    noise=noise,
                )
            save_rgba_contact_sheet(outputs=outputs, batch=first_batch, path=out_path / "eval_samples.png")

    if prompts is not None:
        prompt_records = _read_prompt_records(Path(prompts), max_records=max_records)
        report["prompt_count"] = len(prompt_records)
        if prompt_records:
            sample_records = prompt_records[: min(16, len(prompt_records))]
            report["prompt_samples_written"] = len(sample_records)
            outputs = generate_prompt_batch(
                model,
                tokenizer,
                [record["prompt"] for record in sample_records],
                device=resolve_device(device),
                conditioning_mode=conditioning_mode,
                semantic_records=sample_records,
                semantic_max_length=semantic_max_length,
            )
            save_rgba_contact_sheet(outputs=outputs, path=out_path / "prompt_samples.png")

    report["elapsed_seconds"] = time.perf_counter() - started
    (out_path / "eval_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_generator_model(
    model: Any,
    loader: Any,
    *,
    device: Any,
    conditioning_mode: str = DEFAULT_CONDITIONING_MODE,
    pad_token_id: int = 0,
) -> dict[str, float]:
    th = _require_torch()
    mode = validate_conditioning_mode(conditioning_mode)
    totals = {"loss": 0.0, "loss_alpha": 0.0, "loss_rgb_opaque": 0.0, "loss_rgb_all": 0.0}
    count = 0
    model.eval()
    with th.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            noise = th.zeros(int(batch["caption_tokens"].shape[0]), int(model.latent_dim), device=device)
            model_inputs = apply_conditioning_mode(
                caption_tokens=batch["caption_tokens"],
                semantic_tokens=batch["semantic_tokens"],
                mode=mode,
                pad_token_id=pad_token_id,
            )
            outputs = model(
                **model_inputs,
                noise=noise,
            )
            losses = rgba_generator_loss(outputs, batch)
            batch_size = int(batch["caption_tokens"].shape[0])
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_size
            count += batch_size
    model.train()
    if count == 0:
        return dict.fromkeys(totals, 0.0)
    return {key: value / count for key, value in totals.items()}


def generate_prompt_batch(
    model: TinyCaptionSpriteGenerator,
    tokenizer: SpriteTextTokenizer,
    prompts: list[str],
    *,
    device: Any,
    seed: int = 0,
    conditioning_mode: str = DEFAULT_CONDITIONING_MODE,
    semantic_records: list[Mapping[str, Any]] | None = None,
    semantic_max_length: int = 48,
) -> dict[str, Any]:
    th = _require_torch()
    mode = validate_conditioning_mode(conditioning_mode)
    model.eval()
    tokens = th.as_tensor(
        [tokenizer.encode(prompt, max_length=tokenizer.max_length) for prompt in prompts],
        dtype=th.long,
        device=device,
    )
    semantic_source = semantic_records if semantic_records is not None else [{"prompt": prompt} for prompt in prompts]
    semantic_tokens = th.as_tensor(
        [tokenizer.encode_record_semantics(record, max_length=semantic_max_length) for record in semantic_source],
        dtype=th.long,
        device=device,
    )
    model_inputs = apply_conditioning_mode(
        caption_tokens=tokens,
        semantic_tokens=semantic_tokens,
        mode=mode,
        pad_token_id=tokenizer.pad_id,
    )
    with th.no_grad():
        outputs = model(**model_inputs, noise=model.sample_noise(len(prompts), device=device, seed=seed))
    return outputs


def _read_prompt_records(path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    if max_records is not None and int(max_records) <= 0:
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if isinstance(value, str):
            prompt = value
            prompt_id = f"prompt_{line_no:04d}"
            record = {"prompt": prompt, "prompt_id": prompt_id}
        elif isinstance(value, Mapping):
            record = dict(value)
            prompt = str(value.get("prompt") or value.get("caption") or "")
            prompt_id = str(value.get("prompt_id") or f"prompt_{line_no:04d}")
            record["prompt"] = prompt
            record["prompt_id"] = prompt_id
        else:
            raise ValueError(f"{path}:{line_no}: expected JSON object or string")
        if not prompt:
            raise ValueError(f"{path}:{line_no}: prompt is empty")
        records.append(record)
        if max_records is not None and len(records) >= max(0, int(max_records)):
            break
    return records


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a caption-conditioned RGBA generator checkpoint.")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-records", type=int)
    parsed = parser.parse_args(argv)
    report = evaluate_generator_checkpoint(
        dataset_dir=parsed.dataset,
        training_manifest=parsed.training_manifest,
        checkpoint=parsed.checkpoint,
        split=parsed.split,
        prompts=parsed.prompts,
        out_dir=parsed.out,
        batch_size=parsed.batch_size,
        device=parsed.device,
        max_records=parsed.max_records,
    )
    if report["loss"] is not None:
        print(f"Evaluated {report['records']} {report['split']} records.")
        print(f"Loss: {report['loss']:.6f}")
    if report["prompt_count"]:
        print(f"Generated {report['prompt_samples_written']} prompt samples from {report['prompt_count']} prompts.")
