"""Opaque-pixel masking transforms for masked index-map training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from spritelab.codec.bundle import INDEX_MASK, INDEX_PAD, INDEX_TRANSPARENT

__all__ = [
    "INDEX_MASK",
    "INDEX_PAD",
    "INDEX_TRANSPARENT",
    "FixedOpaqueMask",
    "FullOpaqueMask",
    "RandomOpaqueMask",
]


def _apply_mask(sample: dict[str, Any], keep: torch.Tensor, mask_token: int) -> dict[str, Any]:
    """Attach masked input/target/loss-mask keys given a boolean mask of masked pixels."""

    index_map = sample["index_map"]
    alpha = sample["alpha"]
    opaque = alpha == 1
    masked = keep & opaque

    input_index_map = index_map.clone()
    input_index_map[masked] = mask_token

    opaque_count = int(opaque.sum())
    masked_count = int(masked.sum())
    result = dict(sample)
    result["input_index_map"] = input_index_map
    result["target_index_map"] = index_map.clone()
    result["loss_mask"] = masked
    result["mask_fraction"] = masked_count / opaque_count if opaque_count else 0.0
    return result


def _sample_seed_generator(seed: int | None, sample: dict[str, Any]) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed + int(sample.get("sample_index", 0)))
    return generator


def _mask_opaque_fraction(sample: dict[str, Any], fraction: float, mask_token: int, seed: int | None) -> dict[str, Any]:
    alpha = sample["alpha"]
    opaque = alpha == 1
    opaque_positions = opaque.nonzero(as_tuple=False)
    masked = torch.zeros_like(opaque)
    count = opaque_positions.shape[0]
    if count:
        num_masked = int(round(fraction * count))
        num_masked = max(0, min(count, num_masked))
        if num_masked:
            generator = _sample_seed_generator(seed, sample)
            order = torch.randperm(count, generator=generator)[:num_masked]
            chosen = opaque_positions[order]
            masked[chosen[:, 0], chosen[:, 1]] = True
    return _apply_mask(sample, masked, mask_token)


@dataclass(frozen=True)
class RandomOpaqueMask:
    """Mask a uniformly random fraction of opaque pixels."""

    mask_fraction_min: float = 0.15
    mask_fraction_max: float = 0.75
    mask_token: int = INDEX_MASK
    seed: int | None = None

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        generator = _sample_seed_generator(self.seed, sample)
        span = self.mask_fraction_max - self.mask_fraction_min
        fraction = self.mask_fraction_min + span * float(torch.rand((), generator=generator))
        return _mask_opaque_fraction(sample, fraction, self.mask_token, self.seed)


@dataclass(frozen=True)
class FixedOpaqueMask:
    """Mask a fixed fraction of opaque pixels; deterministic given a seed."""

    mask_fraction: float
    mask_token: int = INDEX_MASK
    seed: int | None = None

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return _mask_opaque_fraction(sample, self.mask_fraction, self.mask_token, self.seed)


@dataclass(frozen=True)
class FullOpaqueMask:
    """Mask every opaque pixel."""

    mask_token: int = INDEX_MASK

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return _apply_mask(sample, sample["alpha"] == 1, self.mask_token)
