from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.losses import sprite_reconstruction_loss


def _batch(alpha_value: float = 1.0) -> dict:
    alpha = torch.full((1, 1, 32, 32), alpha_value)
    index_map = torch.ones(1, 32, 32, dtype=torch.long)
    role_map = torch.zeros(1, 32, 32, dtype=torch.long)
    palette_mask = torch.zeros(1, 4, dtype=torch.bool)
    palette_mask[:, :2] = True
    return {
        "alpha": alpha,
        "index_map": index_map,
        "role_map": role_map,
        "palette_mask": palette_mask,
    }


def test_loss_returns_total_and_components() -> None:
    outputs = {
        "alpha_logits": torch.randn(1, 1, 32, 32, requires_grad=True),
        "index_logits": torch.randn(1, 4, 32, 32, requires_grad=True),
        "role_logits": torch.randn(1, 4, 32, 32, requires_grad=True),
    }
    losses = sprite_reconstruction_loss(outputs, _batch())
    assert set(losses) >= {"loss", "loss_alpha", "loss_index", "loss_role"}
    assert torch.isfinite(losses["loss"])


def test_transparent_pixels_are_ignored_for_index_loss() -> None:
    batch = _batch(alpha_value=0.0)
    batch["index_map"][:, :, :] = 3
    outputs = {
        "alpha_logits": torch.zeros(1, 1, 32, 32, requires_grad=True),
        "index_logits": torch.zeros(1, 4, 32, 32, requires_grad=True),
    }
    losses = sprite_reconstruction_loss(outputs, batch)
    assert float(losses["loss_index"]) == 0.0
    losses["loss"].backward()
    assert outputs["alpha_logits"].grad is not None


def test_backward_computes_gradients() -> None:
    outputs = {
        "alpha_logits": torch.randn(1, 1, 32, 32, requires_grad=True),
        "index_logits": torch.randn(1, 4, 32, 32, requires_grad=True),
        "role_logits": torch.randn(1, 4, 32, 32, requires_grad=True),
    }
    losses = sprite_reconstruction_loss(outputs, _batch())
    losses["loss"].backward()
    assert outputs["alpha_logits"].grad is not None
    assert outputs["index_logits"].grad is not None
    assert outputs["role_logits"].grad is not None
