"""Conditioning-mode helpers for the RGBA sprite generator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STRUCTURED_CONDITIONING_MODE = "caption_semantic_structured"
CONDITIONING_MODES: tuple[str, ...] = (
    "caption",
    "semantic",
    "caption_semantic",
    STRUCTURED_CONDITIONING_MODE,
    "none",
)
DEFAULT_CONDITIONING_MODE = "caption_semantic"


def validate_conditioning_mode(mode: str | None) -> str:
    """Return a normalized conditioning mode or raise for unsupported values."""

    normalized = str(mode or DEFAULT_CONDITIONING_MODE).strip().lower().replace("-", "_")
    if normalized not in CONDITIONING_MODES:
        allowed = ", ".join(CONDITIONING_MODES)
        raise ValueError(f"conditioning_mode must be one of: {allowed}")
    return normalized


def checkpoint_conditioning_mode(checkpoint: Mapping[str, Any]) -> str:
    """Resolve conditioning mode from a checkpoint, with old-checkpoint fallback."""

    direct = checkpoint.get("conditioning_mode")
    if direct:
        return validate_conditioning_mode(str(direct))
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, Mapping) and train_config.get("conditioning_mode"):
        return validate_conditioning_mode(str(train_config["conditioning_mode"]))
    return DEFAULT_CONDITIONING_MODE


def checkpoint_semantic_max_length(checkpoint: Mapping[str, Any], *, fallback: int = 48) -> int:
    """Resolve semantic token length from checkpoint metadata."""

    train_config = checkpoint.get("train_config")
    if isinstance(train_config, Mapping):
        value = train_config.get("semantic_max_length")
        if value is not None:
            return max(1, int(value))
    return max(1, int(fallback))


def uses_structured_conditioning(mode: str | None) -> bool:
    return validate_conditioning_mode(mode) == STRUCTURED_CONDITIONING_MODE


def apply_conditioning_mode(
    *,
    caption_tokens: Any,
    semantic_tokens: Any | None,
    mode: str,
    pad_token_id: int = 0,
    structured_conditioning: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return model inputs for a selected conditioning mode.

    ``none`` and stream-ablated modes use all-pad caption tensors. The model's
    mean-pool denominator clamps at one, so all-pad inputs produce a zero
    conditioning vector without requiring architecture changes.
    """

    normalized = validate_conditioning_mode(mode)
    null_caption = caption_tokens.new_full(caption_tokens.shape, int(pad_token_id))

    if normalized == "caption":
        return {"caption_tokens": caption_tokens, "semantic_tokens": None}
    if normalized == "semantic":
        return {"caption_tokens": null_caption, "semantic_tokens": semantic_tokens}
    if normalized == "caption_semantic":
        return {"caption_tokens": caption_tokens, "semantic_tokens": semantic_tokens}
    if normalized == STRUCTURED_CONDITIONING_MODE:
        result = {"caption_tokens": caption_tokens, "semantic_tokens": semantic_tokens}
        if structured_conditioning is not None:
            result["structured_conditioning"] = structured_conditioning
        return result
    return {"caption_tokens": null_caption, "semantic_tokens": None}
