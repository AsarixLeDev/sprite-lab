"""Opt-in training speed/quality helpers shared by the trainers.

Every helper here is a no-op in its default/off configuration, so a trainer that
leaves the new options at their defaults follows exactly the same numeric path as
before these helpers existed. The GPU/AMP/schedule behaviour only activates when
the caller explicitly opts in.
"""

from __future__ import annotations

import contextlib
import math
import warnings
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


def build_adamw(params: Any, *, lr: float, fused: bool = False) -> Any:
    """``AdamW`` with an opt-in fused CUDA kernel path.

    ``fused=False`` (the default) constructs exactly today's ``torch.optim.AdamW``.
    ``fused=True`` requests the fused kernel, which updates parameters with fewer,
    larger CUDA kernel launches; its numerics can differ slightly from the
    unfused path, so it is never selected implicitly. If the installed torch
    build or device doesn't support ``fused=True`` (e.g. CPU-only), this falls
    back to the unfused optimizer with a warning instead of raising.
    """

    if not fused:
        return torch.optim.AdamW(params, lr=lr)
    try:
        return torch.optim.AdamW(params, lr=lr, fused=True)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(f"fused AdamW unavailable ({exc}); falling back to fused=False", stacklevel=2)
        return torch.optim.AdamW(params, lr=lr)


def apply_backend_speed_flags(*, cudnn_benchmark: bool = False, tf32: bool = False) -> None:
    """Opt-in cuDNN/TF32 backend flags; a no-op at the default ``False`` values.

    ``cudnn_benchmark`` lets cuDNN autotune convolution algorithms for the
    observed input shapes, which pays off when shapes and batch size stay fixed
    (as they do here) at the cost of extra search time on the first few steps.
    ``tf32`` allows TF32 matmul/conv accumulation on Ampere+ GPUs; since the
    training forward pass already runs under bf16 autocast when ``--amp`` is
    set, TF32 only affects the (already low-precision-tolerant) parts that run
    outside autocast, so its effect is expected to be minor. At the defaults,
    this function does not touch ``torch.backends`` at all.
    """

    if not cudnn_benchmark and not tf32:
        return
    if torch is None:
        return
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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
