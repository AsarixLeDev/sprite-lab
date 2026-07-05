"""Deterministic dataset ID and hash helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SAFE_ID_RE = re.compile(r"[^a-z0-9]+")


def make_sprite_id(path: str | Path, *, root: str | Path | None = None) -> str:
    """Create a deterministic filesystem-safe sprite ID from a PNG path."""

    source = Path(path)
    if root is not None:
        try:
            source = source.resolve().relative_to(Path(root).resolve())
        except ValueError:
            source = Path(path)

    parts = list(source.with_suffix("").parts)
    raw = "_".join(parts)
    safe = SAFE_ID_RE.sub("_", raw.lower()).strip("_")
    safe = re.sub(r"_+", "_", safe)
    return safe or "sprite"


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hex digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_path_hash(path: str | Path, *, root: str | Path | None = None) -> str:
    """Return a short stable hash for path-based collision suffixes."""

    source = Path(path)
    if root is not None:
        try:
            text = source.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            text = source.as_posix()
    else:
        text = source.as_posix()

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
