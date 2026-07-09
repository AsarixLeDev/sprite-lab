"""Losses for caption-conditioned RGBA sprite generation."""

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
        raise RuntimeError("PyTorch is required for spritelab generator losses.")
    return torch, nn


def rgba_generator_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
    framing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute RGBA generator losses against ``batch['rgb']`` and ``batch['alpha']``."""

    th, nn_mod = _require_torch()
    weights = {"alpha": 1.0, "rgb_opaque": 1.0, "rgb_all": 0.25, **(weights or {})}

    rgb_target = batch["rgb"].float()
    alpha_target = batch["alpha"].float()
    rgb_logits = outputs["rgb_logits"]
    alpha_logits = outputs["alpha_logits"]

    loss_alpha = nn_mod.functional.binary_cross_entropy_with_logits(alpha_logits, alpha_target)
    rgb_pred = th.sigmoid(rgb_logits)
    rgb_abs = (rgb_pred - rgb_target).abs()
    opaque_mask = alpha_target.gt(0.5).float()
    if bool(opaque_mask.any()):
        denom = opaque_mask.sum().clamp(min=1.0) * int(rgb_target.shape[1])
        loss_rgb_opaque = (rgb_abs * opaque_mask).sum() / denom
    else:
        loss_rgb_opaque = rgb_logits.sum() * 0.0
    loss_rgb_all = nn_mod.functional.l1_loss(rgb_pred, rgb_target)

    total = (
        float(weights.get("alpha", 1.0)) * loss_alpha
        + float(weights.get("rgb_opaque", 1.0)) * loss_rgb_opaque
        + float(weights.get("rgb_all", 0.25)) * loss_rgb_all
    )
    result = {
        "loss": total,
        "loss_alpha": loss_alpha,
        "loss_rgb_opaque": loss_rgb_opaque,
        "loss_rgb_all": loss_rgb_all,
        "opaque_pixels": th.as_tensor(int(opaque_mask.sum().detach().cpu()), device=total.device),
    }
    framing = framing_regularization_loss(outputs, batch, config=framing_config)
    if framing:
        total = result["loss"] + framing["loss_framing"]
        result["loss"] = total
        result.update(framing)
    return result


def framing_regularization_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optional differentiable alpha framing losses for generator training.

    Components are returned only when their corresponding weight is nonzero and
    their required options are present. The returned component values are
    unweighted; ``loss_framing`` is the weighted sum added to the main loss.
    """

    th, _nn_mod = _require_torch()
    cfg = dict(config or {})
    border_weight = float(cfg.get("border_alpha_weight", 0.0) or 0.0)
    coverage_weight = float(cfg.get("alpha_coverage_weight", 0.0) or 0.0)
    center_weight = float(cfg.get("center_weight", 0.0) or 0.0)
    margin_weight = float(cfg.get("margin_band_weight", 0.0) or 0.0)

    if border_weight == 0.0 and coverage_weight == 0.0 and center_weight == 0.0 and margin_weight == 0.0:
        return {}

    alpha_prob = th.sigmoid(outputs["alpha_logits"].float())
    if alpha_prob.ndim != 4 or alpha_prob.shape[1] != 1:
        raise ValueError(f"alpha_logits must have shape [B, 1, H, W], got {tuple(alpha_prob.shape)}")
    _batch_size, _channels, height, width = alpha_prob.shape
    device = alpha_prob.device
    total = alpha_prob.sum() * 0.0
    result: dict[str, Any] = {}

    if border_weight != 0.0:
        mask = th.zeros((height, width), dtype=th.bool, device=device)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        loss_border = alpha_prob[:, :, mask].mean()
        result["loss_border_alpha"] = loss_border
        total = total + border_weight * loss_border

    if coverage_weight != 0.0 and (
        cfg.get("alpha_coverage_min") is not None or cfg.get("alpha_coverage_max") is not None
    ):
        coverage = alpha_prob.mean(dim=(1, 2, 3))
        loss_coverage = coverage.sum() * 0.0
        if cfg.get("alpha_coverage_min") is not None:
            min_coverage = float(cfg["alpha_coverage_min"])
            loss_coverage = loss_coverage + th.relu(th.as_tensor(min_coverage, device=device) - coverage).pow(2)
        if cfg.get("alpha_coverage_max") is not None:
            max_coverage = float(cfg["alpha_coverage_max"])
            loss_coverage = loss_coverage + th.relu(coverage - th.as_tensor(max_coverage, device=device)).pow(2)
        loss_coverage = loss_coverage.mean()
        result["loss_alpha_coverage"] = loss_coverage
        total = total + coverage_weight * loss_coverage

    if center_weight != 0.0:
        y = th.arange(height, dtype=alpha_prob.dtype, device=device).view(1, 1, height, 1)
        x = th.arange(width, dtype=alpha_prob.dtype, device=device).view(1, 1, 1, width)
        mass = alpha_prob.sum(dim=(1, 2, 3))
        denom = mass.clamp(min=1.0e-6)
        center_x = (alpha_prob * x).sum(dim=(1, 2, 3)) / denom
        center_y = (alpha_prob * y).sum(dim=(1, 2, 3)) / denom
        target_x = (width - 1) / 2.0
        target_y = (height - 1) / 2.0
        normalizer = max(1.0, target_x * target_x + target_y * target_y)
        dist = ((center_x - target_x).pow(2) + (center_y - target_y).pow(2)) / normalizer
        valid = mass > 1.0e-6
        loss_center = th.where(valid, dist, th.zeros_like(dist)).mean()
        result["loss_center"] = loss_center
        total = total + center_weight * loss_center

    if margin_weight != 0.0:
        band_size = max(0, int(cfg.get("margin_band_size", 2) or 0))
        if band_size > 0:
            band_size = min(band_size, height // 2, width // 2)
            mask = th.zeros((height, width), dtype=th.bool, device=device)
            mask[:band_size, :] = True
            mask[-band_size:, :] = True
            mask[:, :band_size] = True
            mask[:, -band_size:] = True
            loss_margin = alpha_prob[:, :, mask].mean()
            result["loss_margin_band"] = loss_margin
            total = total + margin_weight * loss_margin

    if result:
        result["loss_framing"] = total
    return result
