"""Benchmark the generator_challenger hot training loop under opt-in speed variants.

This script intentionally reaches into private challenger training internals so it can
time the real hot path; re-check those imports after challenger refactors.

Loads an existing run's ``config.json`` (e.g. a ``train_25k/config.json`` from a past
``audit-challenger-full-v4`` run), rebuilds the exact training objects (dataset,
tokenizer, model, optimizer, EMA state) using the real in-tree code, then times a
warmup + measured window of steps per variant. Writes only to stdout plus an
optional results JSONL -- it never writes checkpoints, samples, or metrics files.

Usage:
    python scripts/benchmark_train_step.py --config experiments/challenger_full_v4_v2_phase1_conditioning/train_25k/config.json
    python scripts/benchmark_train_step.py --config <...>/config.json --variants baseline,fused_adamw,all_on --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import multiprocessing
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from spritelab.training.conditioning import uses_structured_conditioning, validate_conditioning_mode
from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch, read_jsonl
from spritelab.training.eval_baseline import move_batch_to_device, resolve_device
from spritelab.training.generator_challenger import (
    ChallengerTrainConfig,
    RectifiedFlowUNet,
    _init_ema_state,
    _loss_metrics,
    _matches_caption_policy,
    _parse_channel_mults,
    _update_ema_state,
    rectified_flow_loss,
)
from spritelab.training.optim_utils import amp_autocast, build_adamw, dataloader_perf_kwargs
from spritelab.training.structured_conditioning import build_structured_conditioning_vocab
from spritelab.training.tokenization import SpriteTextTokenizer

# Measured on the archived challenger_full_v4_v2_phase1_conditioning/train_25k run
# (RTX 5060 Ti, torch 2.11.0+cu128): median per-step dt from train_metrics.jsonl.
REFERENCE_BASELINE_MS_PER_STEP = 40.7

DEFAULT_VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "baseline"},
    {"name": "no_metrics_sync", "metrics_sync": False},
    {"name": "no_ema", "ema_mode": "off"},
    {"name": "ema_foreach", "ema_mode": "foreach"},
    {"name": "fused_adamw", "fused_adamw": True},
    {"name": "cudnn_benchmark", "cudnn_benchmark": True},
    {"name": "tf32", "tf32": True},
    {"name": "workers0", "num_workers": 0},
    {"name": "batch64", "batch_size": 64},
    {
        "name": "all_on",
        "metrics_sync": False,
        "ema_mode": "foreach",
        "fused_adamw": True,
        "cudnn_benchmark": True,
        "tf32": True,
    },
)


def _load_train_config(config_path: Path, *, out_dir: Path) -> ChallengerTrainConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    field_names = {field.name for field in dataclasses.fields(ChallengerTrainConfig)}
    kwargs: dict[str, Any] = {key: value for key, value in raw.items() if key in field_names}
    kwargs["dataset_dir"] = Path(kwargs["dataset_dir"])
    kwargs["training_manifest"] = Path(kwargs["training_manifest"])
    kwargs["out_dir"] = out_dir
    if kwargs.get("sprite_id_list"):
        kwargs["sprite_id_list"] = Path(kwargs["sprite_id_list"])
    if "checkpoint_steps" in kwargs:
        kwargs["checkpoint_steps"] = tuple(kwargs["checkpoint_steps"] or ())
    return ChallengerTrainConfig(**kwargs)


def _prepare_shared_objects(config: ChallengerTrainConfig) -> tuple[Any, Any, str, Any]:
    manifest_rows = read_jsonl(config.training_manifest)
    token_rows = [row for row in manifest_rows if _matches_caption_policy(row, config.caption_policy_filter)]
    effective_split = str(config.overfit_split or config.split)
    train_rows = [row for row in token_rows if row.get("split") == effective_split]
    tokenizer = SpriteTextTokenizer.build_from_records(
        train_rows or token_rows or manifest_rows, max_length=config.caption_max_length
    )
    conditioning_mode = validate_conditioning_mode(config.conditioning_mode)
    structured_vocab = (
        build_structured_conditioning_vocab(train_rows or token_rows)
        if uses_structured_conditioning(conditioning_mode)
        else None
    )
    npz_cache: dict[str, Any] = {}
    train_dataset = SpriteTrainingDataset(
        config.dataset_dir,
        config.training_manifest,
        split=effective_split,
        max_records=config.max_records,
        tokenizer=tokenizer,
        caption_max_length=config.caption_max_length,
        semantic_max_length=config.semantic_max_length,
        caption_policy_filter=config.caption_policy_filter,
        structured_vocab=structured_vocab,
        npz_cache=npz_cache,
    )
    return tokenizer, structured_vocab, conditioning_mode, train_dataset


def _build_model(
    config: ChallengerTrainConfig, tokenizer: Any, structured_vocab: Any, device: Any
) -> RectifiedFlowUNet:
    model = RectifiedFlowUNet(
        vocab_size=len(tokenizer),
        embed_dim=int(config.embed_dim),
        base_channels=int(config.base_channels),
        channel_mults=_parse_channel_mults(config.channel_mults),
        res_blocks_per_level=int(config.res_blocks_per_level),
        pad_token_id=tokenizer.pad_id,
        structured_vocab_sizes=None if structured_vocab is None else structured_vocab.sizes(),
        film_conditioning=config.film_conditioning,
        bottleneck_attention=config.bottleneck_attention,
    ).to(device)
    return model


def _build_loader(train_dataset: Any, *, batch_size: int, num_workers: int, device: Any, seed: int) -> Any:
    generator = torch.Generator().manual_seed(seed)
    perf = dataloader_perf_kwargs(device, num_workers=num_workers)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_sprite_batch,
        **perf,
    )


def _infinite_batches(loader: Any):
    while True:
        for batch in loader:
            yield batch


def _init_ema_fast(model: RectifiedFlowUNet) -> dict[str, Any]:
    """Prototype foreach EMA cache; mirrors the fast path added to generator_challenger."""

    state = model.state_dict()
    ema_state = {key: value.detach().clone() for key, value in state.items()}
    float_keys = [key for key, value in ema_state.items() if value.dtype.is_floating_point]
    nonfloat_keys = [key for key in ema_state if key not in float_keys]
    return {
        "state": ema_state,
        "ema_float": [ema_state[key] for key in float_keys],
        "src_float": [state[key] for key in float_keys],
        "nonfloat_keys": nonfloat_keys,
        "nonfloat_src": [state[key] for key in nonfloat_keys],
    }


def _update_ema_fast(cache: dict[str, Any], decay: float) -> None:
    clipped = min(1.0, max(0.0, float(decay)))
    with torch.no_grad():
        if cache["ema_float"]:
            torch._foreach_mul_(cache["ema_float"], clipped)
            torch._foreach_add_(cache["ema_float"], cache["src_float"], alpha=1.0 - clipped)
        for key, source in zip(cache["nonfloat_keys"], cache["nonfloat_src"]):
            cache["state"][key].copy_(source)


def _run_variant(
    variant: dict[str, Any],
    *,
    config: ChallengerTrainConfig,
    tokenizer: Any,
    structured_vocab: Any,
    conditioning_mode: str,
    train_dataset: Any,
    device: Any,
    warmup: int,
    steps: int,
    seed: int,
) -> dict[str, float]:
    metrics_sync = bool(variant.get("metrics_sync", True))
    ema_mode = variant.get("ema_mode", "legacy" if config.ema_decay > 0 else "off")
    fused_adamw = bool(variant.get("fused_adamw", False))
    cudnn_benchmark = bool(variant.get("cudnn_benchmark", False))
    tf32 = bool(variant.get("tf32", False))
    num_workers = int(variant.get("num_workers", config.num_workers))
    batch_size = int(variant.get("batch_size", config.batch_size))

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    model = _build_model(config, tokenizer, structured_vocab, device)
    # build_adamw(fused=False) omits the `fused` kwarg entirely (matching the
    # production default construction) rather than passing `fused=False`
    # explicitly -- torch's own None/None auto-selection for foreach/fused
    # differs from an explicit `fused=False`, so passing the latter here would
    # silently make every non-fused_adamw variant slower than production's
    # actual default and invalidate the whole comparison.
    optimizer = build_adamw(model.parameters(), lr=float(config.learning_rate), fused=fused_adamw)

    ema_state = _init_ema_state(model) if ema_mode == "legacy" else None
    ema_cache = _init_ema_fast(model) if ema_mode == "foreach" else None

    loader = _build_loader(train_dataset, batch_size=batch_size, num_workers=num_workers, device=device, seed=seed)
    batches = _infinite_batches(loader)
    non_blocking = device.type == "cuda"
    model.train()

    def step_once() -> None:
        batch = move_batch_to_device(next(batches), device, non_blocking=non_blocking)
        with amp_autocast(device, config.amp):
            losses = rectified_flow_loss(
                model,
                batch,
                conditioning_mode=conditioning_mode,
                cfg_dropout=config.cfg_dropout,
                structured_field_dropout=config.structured_field_dropout,
                structured_field_dropout_rates=config.structured_field_dropout_rates,
                pad_token_id=tokenizer.pad_id,
                foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
                background_rgb_loss_weight=config.background_rgb_loss_weight,
                palette_loss_weight=config.palette_loss_weight,
                palette_loss_temperature=config.palette_loss_temperature,
            )
            loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ema_mode == "legacy":
            _update_ema_state(ema_state, model, decay=float(config.ema_decay))
        elif ema_mode == "foreach":
            _update_ema_fast(ema_cache, float(config.ema_decay))
        if metrics_sync:
            _ = float(loss.detach().cpu())
            _ = _loss_metrics(losses)

    try:
        for _ in range(warmup):
            step_once()
        if device.type == "cuda":
            torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(steps):
            step_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        # Persistent DataLoader workers (Windows spawn) only shut down once every
        # reference to the loader/iterator is dropped; without this, worker
        # processes from one variant can outlive it and contend with the next.
        batches.close()
        del batches
        del loader
        gc.collect()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return {
        "variant": variant["name"],
        "ms_per_step": elapsed / steps * 1000.0,
        "it_per_s": steps / elapsed,
        "samples_per_s": steps * batch_size / elapsed,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Path to a run's config.json")
    parser.add_argument("--steps", type=int, default=400, help="Measured steps per variant")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup steps per variant (untimed)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None, help="Override the device recorded in config.json")
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant names to run (default: all). "
        f"Available: {', '.join(v['name'] for v in DEFAULT_VARIANTS)}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Results JSONL path (default: a file under the system temp directory)",
    )
    args = parser.parse_args()

    scratch_out_dir = Path(tempfile.mkdtemp(prefix="sprite_lab_benchmark_train_step_"))
    config = _load_train_config(args.config, out_dir=scratch_out_dir)
    device = resolve_device(args.device or config.device)
    print(f"Device: {device} | amp={config.amp} | batch_size={config.batch_size} | num_workers={config.num_workers}")

    variant_names = (
        None if args.variants is None else {name.strip() for name in args.variants.split(",") if name.strip()}
    )
    variants = [v for v in DEFAULT_VARIANTS if variant_names is None or v["name"] in variant_names]
    if not variants:
        raise SystemExit(f"No matching variants for --variants {args.variants!r}")

    print("Preparing dataset/tokenizer/vocab (shared across variants)...")
    tokenizer, structured_vocab, conditioning_mode, train_dataset = _prepare_shared_objects(config)
    print(f"Train records: {len(train_dataset)}")

    results: list[dict[str, float]] = []
    for variant in variants:
        print(f"Running variant: {variant['name']} ...")
        result = _run_variant(
            variant,
            config=config,
            tokenizer=tokenizer,
            structured_vocab=structured_vocab,
            conditioning_mode=conditioning_mode,
            train_dataset=train_dataset,
            device=device,
            warmup=args.warmup,
            steps=args.steps,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"  {result['ms_per_step']:.2f} ms/step | {result['it_per_s']:.2f} it/s | "
            f"{result['samples_per_s']:.1f} samples/s | batch={result['batch_size']} workers={result['num_workers']}"
        )

    print()
    print(f"{'variant':<18} {'ms/step':>10} {'it/s':>10} {'samples/s':>12} {'batch':>7} {'workers':>8}")
    for result in results:
        print(
            f"{result['variant']:<18} {result['ms_per_step']:>10.2f} {result['it_per_s']:>10.2f} "
            f"{result['samples_per_s']:>12.1f} {result['batch_size']:>7} {result['num_workers']:>8}"
        )
    baseline = next((r for r in results if r["variant"] == "baseline"), None)
    if baseline is not None:
        print(
            f"\nbaseline vs. archived reference: {baseline['ms_per_step']:.2f} ms/step measured, "
            f"{REFERENCE_BASELINE_MS_PER_STEP:.1f} ms/step archived (median dt, train_metrics.jsonl)."
        )

    out_path = args.out or (scratch_out_dir / "results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"\nResults JSONL written to {out_path}")

    # Defensive net: a persistent-worker DataLoader occasionally leaves spawned
    # worker processes alive past this point on Windows even after per-variant
    # cleanup, which would otherwise contend with the next invocation's GPU work.
    stragglers = multiprocessing.active_children()
    for child in stragglers:
        child.terminate()
    for child in stragglers:
        child.join(timeout=5)


if __name__ == "__main__":
    main()
