"""Losses for diagnostic sprite reconstruction."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for spritelab training losses.")
    return torch, nn


def _masked_cross_entropy(logits: Any, targets: Any, mask: Any) -> Any:
    _th, nn_mod = _require_torch()
    mask = mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    classes = int(logits.shape[1])
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, classes)
    flat_targets = targets.long().reshape(-1)
    flat_mask = mask.reshape(-1)
    return nn_mod.functional.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


def _target_palette_valid(index_map: Any, palette_mask: Any) -> Any:
    th, _nn_mod = _require_torch()
    target = index_map.long()
    max_index = palette_mask.shape[1] - 1
    in_range = (target >= 0) & (target <= max_index)
    clipped = target.clamp(min=0, max=max_index)
    valid = th.gather(palette_mask.bool(), 1, clipped.reshape(clipped.shape[0], -1))
    valid = valid.reshape_as(target)
    return in_range & valid


def sprite_reconstruction_loss(
    outputs: dict[str, Any], batch: dict[str, Any], *, weights: dict[str, float] | None = None
) -> dict[str, Any]:
    """Compute alpha/index/optional-role reconstruction losses.

    Transparent pixels are ignored for index cross entropy. The alpha head is
    responsible for reconstructing transparency, while index loss only judges
    opaque pixels whose target palette row is valid for that sample.
    """

    th, nn_mod = _require_torch()
    weights = {"alpha": 1.0, "index": 1.0, "role": 0.2, **(weights or {})}

    alpha_target = batch["alpha"].float()
    alpha_logits = outputs["alpha_logits"]
    loss_alpha = nn_mod.functional.binary_cross_entropy_with_logits(alpha_logits, alpha_target)

    index_logits = outputs["index_logits"]
    index_target = batch["index_map"].long()
    alpha_mask = alpha_target.squeeze(1) > 0.5
    palette_valid = _target_palette_valid(index_target, batch["palette_mask"])
    index_mask = alpha_mask & palette_valid
    loss_index = _masked_cross_entropy(
        index_logits, index_target.clamp(min=0, max=index_logits.shape[1] - 1), index_mask
    )

    role_logits = outputs.get("role_logits")
    if role_logits is not None and "role_map" in batch and float(weights.get("role", 0.0)) != 0.0:
        role_target = batch["role_map"].long().clamp(min=0, max=role_logits.shape[1] - 1)
        loss_role = nn_mod.functional.cross_entropy(role_logits, role_target)
    else:
        loss_role = alpha_logits.sum() * 0.0

    total = (
        float(weights.get("alpha", 1.0)) * loss_alpha
        + float(weights.get("index", 1.0)) * loss_index
        + float(weights.get("role", 0.2)) * loss_role
    )
    return {
        "loss": total,
        "loss_alpha": loss_alpha,
        "loss_index": loss_index,
        "loss_role": loss_role,
        "index_loss_pixels": th.as_tensor(int(index_mask.sum().detach().cpu()), device=total.device),
    }
