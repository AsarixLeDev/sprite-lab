from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.generator_models import TinyCaptionSpriteGenerator


def _tokens() -> torch.Tensor:
    return torch.tensor(
        [
            [2, 4, 5, 3, 0, 0],
            [2, 6, 7, 8, 9, 3],
        ],
        dtype=torch.long,
    )


def test_generator_forward_shapes_on_cpu() -> None:
    model = TinyCaptionSpriteGenerator(vocab_size=16, embed_dim=8, latent_dim=4, hidden_channels=8)
    outputs = model(caption_tokens=_tokens(), semantic_tokens=_tokens(), noise=torch.zeros(2, 4))
    assert next(model.parameters()).device.type == "cpu"
    assert outputs["rgb_logits"].shape == (2, 3, 32, 32)
    assert outputs["alpha_logits"].shape == (2, 1, 32, 32)


def test_fixed_noise_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(1)
    model = TinyCaptionSpriteGenerator(vocab_size=16, embed_dim=8, latent_dim=4, hidden_channels=8)
    model.eval()
    noise = torch.randn(2, 4)
    with torch.no_grad():
        first = model(caption_tokens=_tokens(), noise=noise)
        second = model(caption_tokens=_tokens(), noise=noise)
    assert torch.allclose(first["rgb_logits"], second["rgb_logits"])
    assert torch.allclose(first["alpha_logits"], second["alpha_logits"])


def test_different_noise_can_change_outputs() -> None:
    torch.manual_seed(2)
    model = TinyCaptionSpriteGenerator(vocab_size=16, embed_dim=8, latent_dim=4, hidden_channels=8)
    model.eval()
    with torch.no_grad():
        first = model(caption_tokens=_tokens(), noise=torch.zeros(2, 4))
        second = model(caption_tokens=_tokens(), noise=torch.ones(2, 4))
    assert not torch.allclose(first["rgb_logits"], second["rgb_logits"])
