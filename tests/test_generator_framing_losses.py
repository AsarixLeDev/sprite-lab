from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.generator_losses import framing_regularization_loss, rgba_generator_loss


def _outputs(alpha_logits: torch.Tensor) -> dict:
    return {
        "rgb_logits": torch.zeros(alpha_logits.shape[0], 3, 32, 32, requires_grad=True),
        "alpha_logits": alpha_logits.clone().detach().requires_grad_(True),
    }


def _batch() -> dict:
    return {
        "rgb": torch.zeros(1, 3, 32, 32),
        "alpha": torch.zeros(1, 1, 32, 32),
    }


def _alpha_logits_for_square(y: int, x: int, size: int, inside: float = 10.0, outside: float = -10.0) -> torch.Tensor:
    logits = torch.full((1, 1, 32, 32), outside)
    logits[:, :, y : y + size, x : x + size] = inside
    return logits


def test_border_alpha_loss_is_higher_when_border_alpha_is_high() -> None:
    high = _outputs(torch.full((1, 1, 32, 32), 10.0))
    low = _outputs(_alpha_logits_for_square(8, 8, 8))
    cfg = {"border_alpha_weight": 1.0}
    high_loss = framing_regularization_loss(high, _batch(), config=cfg)["loss_border_alpha"]
    low_loss = framing_regularization_loss(low, _batch(), config=cfg)["loss_border_alpha"]
    assert float(high_loss.detach()) > float(low_loss.detach())


def test_border_alpha_loss_is_lower_when_border_alpha_is_zero() -> None:
    low = _outputs(_alpha_logits_for_square(8, 8, 8, outside=-20.0))
    cfg = {"border_alpha_weight": 1.0}
    loss = framing_regularization_loss(low, _batch(), config=cfg)["loss_border_alpha"]
    assert float(loss.detach()) < 1.0e-6


def test_center_loss_is_lower_for_centered_alpha_mass() -> None:
    centered = _outputs(_alpha_logits_for_square(12, 12, 8))
    off_center = _outputs(_alpha_logits_for_square(0, 0, 8))
    cfg = {"center_weight": 1.0}
    centered_loss = framing_regularization_loss(centered, _batch(), config=cfg)["loss_center"]
    off_center_loss = framing_regularization_loss(off_center, _batch(), config=cfg)["loss_center"]
    assert float(centered_loss.detach()) < float(off_center_loss.detach())


def test_center_loss_is_higher_for_off_center_alpha_mass() -> None:
    centered = _outputs(_alpha_logits_for_square(12, 12, 8))
    off_center = _outputs(_alpha_logits_for_square(0, 20, 8))
    cfg = {"center_weight": 1.0}
    centered_loss = framing_regularization_loss(centered, _batch(), config=cfg)["loss_center"]
    off_center_loss = framing_regularization_loss(off_center, _batch(), config=cfg)["loss_center"]
    assert float(off_center_loss.detach()) > float(centered_loss.detach())


def test_alpha_coverage_loss_is_zero_inside_range() -> None:
    outputs = _outputs(torch.zeros((1, 1, 32, 32)))
    cfg = {"alpha_coverage_weight": 1.0, "alpha_coverage_min": 0.4, "alpha_coverage_max": 0.6}
    loss = framing_regularization_loss(outputs, _batch(), config=cfg)["loss_alpha_coverage"]
    assert float(loss.detach()) == pytest.approx(0.0)


def test_alpha_coverage_loss_positive_outside_range() -> None:
    outputs = _outputs(torch.full((1, 1, 32, 32), 10.0))
    cfg = {"alpha_coverage_weight": 1.0, "alpha_coverage_min": 0.1, "alpha_coverage_max": 0.4}
    loss = framing_regularization_loss(outputs, _batch(), config=cfg)["loss_alpha_coverage"]
    assert float(loss.detach()) > 0.0


def test_framing_losses_are_finite_when_alpha_mass_is_near_zero() -> None:
    outputs = _outputs(torch.full((1, 1, 32, 32), -80.0))
    cfg = {
        "border_alpha_weight": 1.0,
        "alpha_coverage_weight": 1.0,
        "alpha_coverage_min": 0.1,
        "alpha_coverage_max": 0.9,
        "center_weight": 1.0,
        "margin_band_weight": 1.0,
        "margin_band_size": 2,
    }
    losses = framing_regularization_loss(outputs, _batch(), config=cfg)
    assert torch.isfinite(losses["loss_framing"])


def test_framing_components_appear_in_total_loss_when_weights_nonzero() -> None:
    outputs = _outputs(torch.full((1, 1, 32, 32), 10.0))
    losses = rgba_generator_loss(
        outputs,
        _batch(),
        framing_config={"border_alpha_weight": 0.5, "center_weight": 0.05, "margin_band_weight": 0.1},
    )
    assert "loss_border_alpha" in losses
    assert "loss_center" in losses
    assert "loss_margin_band" in losses
    assert "loss_framing" in losses
    assert float(losses["loss_framing"].detach()) > 0.0


def test_framing_components_are_absent_when_weights_are_zero() -> None:
    outputs = _outputs(torch.full((1, 1, 32, 32), 10.0))
    losses = rgba_generator_loss(outputs, _batch())
    assert "loss_border_alpha" not in losses
    assert "loss_alpha_coverage" not in losses
    assert "loss_center" not in losses
    assert "loss_margin_band" not in losses
