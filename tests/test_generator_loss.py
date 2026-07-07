from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.generator_losses import rgba_generator_loss


def _batch(alpha_value: float = 1.0, rgb_value: float = 0.0) -> dict:
    return {
        "rgb": torch.full((1, 3, 32, 32), rgb_value),
        "alpha": torch.full((1, 1, 32, 32), alpha_value),
    }


def test_rgba_generator_loss_returns_finite_components() -> None:
    outputs = {
        "rgb_logits": torch.randn(1, 3, 32, 32, requires_grad=True),
        "alpha_logits": torch.randn(1, 1, 32, 32, requires_grad=True),
    }
    losses = rgba_generator_loss(outputs, _batch())
    assert set(losses) >= {"loss", "loss_alpha", "loss_rgb_opaque", "loss_rgb_all"}
    assert torch.isfinite(losses["loss"])


def test_rgba_generator_loss_backward_computes_gradients() -> None:
    outputs = {
        "rgb_logits": torch.randn(1, 3, 32, 32, requires_grad=True),
        "alpha_logits": torch.randn(1, 1, 32, 32, requires_grad=True),
    }
    losses = rgba_generator_loss(outputs, _batch())
    losses["loss"].backward()
    assert outputs["rgb_logits"].grad is not None
    assert outputs["alpha_logits"].grad is not None


def test_opaque_rgb_loss_ignores_transparent_pixels() -> None:
    outputs = {
        "rgb_logits": torch.zeros(1, 3, 32, 32, requires_grad=True),
        "alpha_logits": torch.zeros(1, 1, 32, 32, requires_grad=True),
    }
    transparent = rgba_generator_loss(outputs, _batch(alpha_value=0.0, rgb_value=1.0))
    opaque = rgba_generator_loss(outputs, _batch(alpha_value=1.0, rgb_value=1.0))
    assert float(transparent["loss_rgb_opaque"].detach()) == 0.0
    assert float(opaque["loss_rgb_opaque"].detach()) > 0.0
    assert float(transparent["loss_rgb_all"].detach()) > 0.0
