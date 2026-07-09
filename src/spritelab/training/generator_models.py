"""Caption-conditioned sprite generator models."""

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
        raise RuntimeError("PyTorch is required for spritelab generator models.")
    return torch, nn


_ModuleBase = nn.Module if nn is not None else object


class TinyCaptionSpriteGenerator(_ModuleBase):
    """Tiny caption/semantic/noise conditioned RGBA generator for 32x32 sprites."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 32,
        latent_dim: int = 32,
        hidden_channels: int = 48,
        pad_token_id: int = 0,
    ) -> None:
        _torch, nn_mod = _require_torch()
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_channels = int(hidden_channels)
        self.pad_token_id = int(pad_token_id)

        self.token_embedding = nn_mod.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_token_id)
        cond_dim = self.embed_dim * 2 + self.latent_dim
        self.grid_mlp = nn_mod.Sequential(
            nn_mod.Linear(cond_dim, self.hidden_channels * 4),
            nn_mod.ReLU(),
            nn_mod.Linear(self.hidden_channels * 4, self.hidden_channels * 8 * 8),
        )
        self.learned_grid = nn_mod.Parameter(_torch.randn(1, self.hidden_channels, 8, 8) * 0.02)
        self.decoder = nn_mod.Sequential(
            nn_mod.Upsample(scale_factor=2, mode="nearest"),
            nn_mod.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn_mod.ReLU(),
            nn_mod.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn_mod.ReLU(),
            nn_mod.Upsample(scale_factor=2, mode="nearest"),
            nn_mod.Conv2d(self.hidden_channels, max(8, self.hidden_channels // 2), kernel_size=3, padding=1),
            nn_mod.ReLU(),
            nn_mod.Conv2d(
                max(8, self.hidden_channels // 2), max(8, self.hidden_channels // 2), kernel_size=3, padding=1
            ),
            nn_mod.ReLU(),
        )
        out_channels = max(8, self.hidden_channels // 2)
        self.rgb_head = nn_mod.Conv2d(out_channels, 3, kernel_size=1)
        self.alpha_head = nn_mod.Conv2d(out_channels, 1, kernel_size=1)

    def config(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "hidden_channels": self.hidden_channels,
            "pad_token_id": self.pad_token_id,
        }

    def forward(
        self,
        *,
        caption_tokens: Any,
        semantic_tokens: Any | None = None,
        noise: Any | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        th, _nn_mod = _require_torch()
        caption_cond = self._mean_pool_tokens(caption_tokens)
        batch = int(caption_cond.shape[0])
        if semantic_tokens is None:
            semantic_cond = th.zeros(batch, self.embed_dim, device=caption_cond.device, dtype=caption_cond.dtype)
        else:
            semantic_cond = self._mean_pool_tokens(semantic_tokens)
        if noise is None:
            noise = self.sample_noise(batch, device=caption_cond.device, seed=seed)
        else:
            noise = noise.to(device=caption_cond.device, dtype=caption_cond.dtype)

        cond = th.cat([caption_cond, semantic_cond, noise], dim=1)
        grid = self.grid_mlp(cond).reshape(batch, self.hidden_channels, 8, 8)
        grid = grid + self.learned_grid
        decoded = self.decoder(grid)
        return {
            "rgb_logits": self.rgb_head(decoded),
            "alpha_logits": self.alpha_head(decoded),
        }

    def sample_noise(self, batch_size: int, *, device: Any | None = None, seed: int | None = None) -> Any:
        th, _nn_mod = _require_torch()
        if device is None:
            device = next(self.parameters()).device
        generator = None
        if seed is not None:
            try:
                generator = th.Generator(device=device)
            except TypeError:  # pragma: no cover - older torch fallback.
                generator = th.Generator()
            generator.manual_seed(int(seed))
        return th.randn(int(batch_size), self.latent_dim, device=device, generator=generator)

    def _mean_pool_tokens(self, tokens: Any) -> Any:
        th, _nn_mod = _require_torch()
        token_ids = tokens.long().clamp(min=0, max=self.vocab_size - 1)
        embedded = self.token_embedding(token_ids)
        mask = token_ids.ne(self.pad_token_id).float().unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
