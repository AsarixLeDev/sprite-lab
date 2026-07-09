"""Shared checkpoint I/O helpers (migrated from eval_generator.py)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]

from spritelab.training.tokenization import SPECIAL_TOKENS, SpriteTextTokenizer


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab checkpoint I/O.")
    return torch


def load_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    th = _require_torch()
    try:
        return th.load(Path(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        return th.load(Path(checkpoint), map_location="cpu")


def tokenizer_from_checkpoint(checkpoint: Mapping[str, Any]) -> SpriteTextTokenizer:
    data = checkpoint.get("vocab")
    if not isinstance(data, Mapping):
        raise ValueError("checkpoint does not contain a tokenizer vocabulary")
    token_to_id = {str(token): int(index) for token, index in dict(data["token_to_id"]).items()}
    for index, token in enumerate(SPECIAL_TOKENS):
        if token_to_id.get(token) != index:
            raise ValueError(f"vocabulary special token {token!r} must have id {index}")
    return SpriteTextTokenizer(token_to_id=token_to_id, max_length=int(data.get("max_length", 32)))
