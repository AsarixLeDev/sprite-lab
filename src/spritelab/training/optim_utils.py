"""Opt-in training speed/quality helpers shared by the trainers.

Every helper here is a no-op in its default/off configuration, so a trainer that
leaves the new options at their defaults follows exactly the same numeric path as
before these helpers existed. The GPU/AMP/schedule behaviour only activates when
the caller explicitly opts in.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

try:  # torch is an optional project dependency.
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]


def device_type(device: Any) -> str:
    return str(getattr(device, "type", device))


def amp_autocast(device: Any, enabled: bool):
    """bf16 autocast on CUDA when ``enabled``; a no-op context otherwise.

    bfloat16 shares float32's exponent range, so no ``GradScaler`` is required and
    the backward / optimizer-step code is identical to the fp32 path — only the
    forward pass runs in mixed precision. When disabled (the default) this returns
    ``contextlib.nullcontext()`` which changes nothing.
    """

    if enabled and torch is not None and device_type(device) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_lr_scheduler(optimizer: Any, *, schedule: str, max_steps: int, warmup_steps: int = 0):
    """Return a per-step LR scheduler, or ``None`` for ``schedule='none'``.

    ``None`` means the learning rate stays constant, i.e. identical to today.
    """

    normalized = str(schedule or "none").strip().lower()
    if normalized == "none" or torch is None:
        return None
    total = max(1, int(max_steps))
    warmup = max(0, int(warmup_steps))
    if normalized == "cosine":

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"unknown lr schedule: {schedule!r} (expected 'none' or 'cosine')")


def clip_gradients(model: Any, max_norm: float) -> None:
    """Clip gradient norm in place; a no-op when ``max_norm`` is 0 or negative."""

    if torch is not None and max_norm and float(max_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))


def dataloader_perf_kwargs(device: Any, *, num_workers: int, pin_memory: bool | None = None) -> dict[str, Any]:
    """DataLoader kwargs for the GPU fast path.

    Defaults preserve current behaviour: ``num_workers`` defaults to 0 (no worker
    processes), ``pin_memory`` defaults on only for CUDA devices, and the
    worker-only options are added solely when workers are actually used.
    """

    workers = max(0, int(num_workers))
    if pin_memory is None:
        pin_memory = device_type(device) == "cuda"
    kwargs: dict[str, Any] = {"num_workers": workers, "pin_memory": bool(pin_memory)}
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs
