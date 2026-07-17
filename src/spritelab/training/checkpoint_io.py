"""Shared checkpoint I/O helpers (migrated from eval_generator.py)."""

from __future__ import annotations

import os
import stat
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
    path = Path(checkpoint)
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(getattr(before, "st_nlink", 1)) != 1:
            raise ValueError("checkpoint must be one regular single-link file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            try:
                loaded = th.load(handle, map_location="cpu", weights_only=True)
            except TypeError as exc:
                raise RuntimeError("This PyTorch build does not support safe weights-only checkpoint loading.") from exc
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_nlink,
            getattr(before, "st_mtime_ns", None),
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_nlink,
            getattr(after, "st_mtime_ns", None),
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_nlink,
            getattr(current, "st_mtime_ns", None),
        )
        if before_identity != after_identity or after_identity != current_identity:
            raise RuntimeError("checkpoint changed while it was loaded")
    finally:
        os.close(descriptor)
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return dict(loaded)


def tokenizer_from_checkpoint(checkpoint: Mapping[str, Any]) -> SpriteTextTokenizer:
    data = checkpoint.get("vocab")
    if not isinstance(data, Mapping):
        raise ValueError("checkpoint does not contain a tokenizer vocabulary")
    token_to_id = {str(token): int(index) for token, index in dict(data["token_to_id"]).items()}
    for index, token in enumerate(SPECIAL_TOKENS):
        if token_to_id.get(token) != index:
            raise ValueError(f"vocabulary special token {token!r} must have id {index}")
    return SpriteTextTokenizer(token_to_id=token_to_id, max_length=int(data.get("max_length", 32)))
