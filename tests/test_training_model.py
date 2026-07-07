from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.models import SpriteCondAutoencoder


def _batch() -> dict:
    caption_tokens = torch.tensor(
        [
            [2, 4, 5, 3, 0, 0],
            [2, 6, 7, 8, 9, 3],
        ],
        dtype=torch.long,
    )
    return {
        "index_map": torch.ones(2, 32, 32, dtype=torch.long),
        "alpha": torch.ones(2, 1, 32, 32),
        "role_map": torch.zeros(2, 32, 32, dtype=torch.long),
        "caption_tokens": caption_tokens,
        "semantic_tokens": caption_tokens,
        "category_id": torch.tensor([1, 2], dtype=torch.long),
    }


def test_forward_pass_output_shapes_on_cpu() -> None:
    model = SpriteCondAutoencoder(num_palette_slots=8, vocab_size=16, num_roles=4, hidden_dim=16)
    outputs = model(**_batch())
    assert next(model.parameters()).device.type == "cpu"
    assert outputs["alpha_logits"].shape == (2, 1, 32, 32)
    assert outputs["index_logits"].shape == (2, 8, 32, 32)
    assert outputs["role_logits"].shape == (2, 4, 32, 32)


def test_model_handles_variable_caption_lengths_via_padding() -> None:
    model = SpriteCondAutoencoder(num_palette_slots=8, vocab_size=16, num_roles=4, hidden_dim=16)
    batch = _batch()
    batch["caption_tokens"][0, 3:] = 0
    outputs = model(**batch)
    assert torch.isfinite(outputs["alpha_logits"]).all()
    assert torch.isfinite(outputs["index_logits"]).all()
