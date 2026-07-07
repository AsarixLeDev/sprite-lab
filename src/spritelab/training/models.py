"""Small diagnostic PyTorch models for semantic sprite training."""

from __future__ import annotations

from typing import Any

try:  # torch is optional for the base package.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for spritelab training models.")
    return torch, nn


_ModuleBase = nn.Module if nn is not None else object


class SpriteCondAutoencoder(_ModuleBase):
    """Tiny conditional autoencoder for 32x32 alpha/index reconstruction."""

    def __init__(
        self,
        *,
        num_palette_slots: int = 33,
        vocab_size: int = 256,
        num_roles: int = 16,
        num_categories: int = 64,
        index_embed_dim: int = 16,
        role_embed_dim: int = 8,
        text_embed_dim: int = 32,
        category_embed_dim: int = 8,
        hidden_dim: int = 48,
        pad_token_id: int = 0,
        predict_roles: bool = True,
    ) -> None:
        _torch, nn_mod = _require_torch()
        super().__init__()
        self.num_palette_slots = int(num_palette_slots)
        self.vocab_size = int(vocab_size)
        self.num_roles = int(num_roles)
        self.num_categories = int(num_categories)
        self.pad_token_id = int(pad_token_id)
        self.predict_roles = bool(predict_roles)
        self.index_embed_dim = int(index_embed_dim)
        self.role_embed_dim = int(role_embed_dim)
        self.text_embed_dim = int(text_embed_dim)
        self.category_embed_dim = int(category_embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.index_embedding = nn_mod.Embedding(self.num_palette_slots, index_embed_dim)
        self.role_embedding = nn_mod.Embedding(max(self.num_roles, 1), role_embed_dim)
        self.text_embedding = nn_mod.Embedding(self.vocab_size, text_embed_dim, padding_idx=self.pad_token_id)
        self.category_embedding = nn_mod.Embedding(self.num_categories, category_embed_dim)

        in_channels = index_embed_dim + role_embed_dim + 1
        self.encoder = nn_mod.Sequential(
            nn_mod.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn_mod.ReLU(),
            nn_mod.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn_mod.ReLU(),
            nn_mod.AvgPool2d(2),
            nn_mod.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn_mod.ReLU(),
            nn_mod.AvgPool2d(2),
        )
        cond_dim = text_embed_dim * 2 + category_embed_dim
        self.film = nn_mod.Linear(cond_dim, hidden_dim * 4)
        self.decoder = nn_mod.Sequential(
            nn_mod.Upsample(scale_factor=2, mode="nearest"),
            nn_mod.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn_mod.ReLU(),
            nn_mod.Upsample(scale_factor=2, mode="nearest"),
            nn_mod.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn_mod.ReLU(),
        )
        self.alpha_head = nn_mod.Conv2d(hidden_dim, 1, kernel_size=1)
        self.index_head = nn_mod.Conv2d(hidden_dim, self.num_palette_slots, kernel_size=1)
        self.role_head = nn_mod.Conv2d(hidden_dim, self.num_roles, kernel_size=1) if self.predict_roles else None

    def config(self) -> dict[str, Any]:
        return {
            "num_palette_slots": self.num_palette_slots,
            "vocab_size": self.vocab_size,
            "num_roles": self.num_roles,
            "num_categories": self.num_categories,
            "index_embed_dim": self.index_embed_dim,
            "role_embed_dim": self.role_embed_dim,
            "text_embed_dim": self.text_embed_dim,
            "category_embed_dim": self.category_embed_dim,
            "hidden_dim": self.hidden_dim,
            "pad_token_id": self.pad_token_id,
            "predict_roles": self.predict_roles,
        }

    def forward(
        self,
        *,
        index_map: Any,
        alpha: Any,
        role_map: Any | None = None,
        caption_tokens: Any | None = None,
        semantic_tokens: Any | None = None,
        category_id: Any | None = None,
    ) -> dict[str, Any]:
        th, _nn_mod = _require_torch()
        index = index_map.long().clamp(min=0, max=self.num_palette_slots - 1)
        index_features = self.index_embedding(index).permute(0, 3, 1, 2)

        if alpha.dim() == 3:
            alpha_channel = alpha.float().unsqueeze(1)
        else:
            alpha_channel = alpha.float()

        if role_map is None:
            role_map = th.zeros_like(index)
        role = role_map.long().clamp(min=0, max=max(self.num_roles - 1, 0))
        role_features = self.role_embedding(role).permute(0, 3, 1, 2)

        features = th.cat([index_features, role_features, alpha_channel], dim=1)
        encoded = self.encoder(features)

        batch = index.shape[0]
        if caption_tokens is None:
            caption_cond = th.zeros(batch, self.text_embedding.embedding_dim, device=index.device)
        else:
            caption_cond = self._mean_pool_tokens(caption_tokens)
        if semantic_tokens is None:
            semantic_cond = th.zeros(batch, self.text_embedding.embedding_dim, device=index.device)
        else:
            semantic_cond = self._mean_pool_tokens(semantic_tokens)
        if category_id is None:
            category_id = th.zeros(batch, dtype=th.long, device=index.device)
        category = self.category_embedding(category_id.long().clamp(min=0, max=self.num_categories - 1))

        cond = th.cat([caption_cond, semantic_cond, category], dim=1)
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        encoded = encoded * (1.0 + gamma) + beta

        decoded = self.decoder(encoded)
        outputs = {
            "alpha_logits": self.alpha_head(decoded),
            "index_logits": self.index_head(decoded),
        }
        if self.role_head is not None:
            outputs["role_logits"] = self.role_head(decoded)
        return outputs

    def _mean_pool_tokens(self, tokens: Any) -> Any:
        th, _nn_mod = _require_torch()
        token_ids = tokens.long().clamp(min=0, max=self.vocab_size - 1)
        embedded = self.text_embedding(token_ids)
        mask = token_ids.ne(self.pad_token_id).float().unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
