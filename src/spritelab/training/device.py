"""Shared device helpers (migrated from eval_baseline.py)."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab device operations.")
    return torch


def resolve_device(device: str) -> Any:
    th = _require_torch()
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


def move_batch_to_device(batch: dict[str, Any], device: Any, *, non_blocking: bool = False) -> dict[str, Any]:
    th = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if isinstance(value, th.Tensor) else value
    return moved
