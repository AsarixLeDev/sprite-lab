"""Streaming download helpers for direct ZIP URLs."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


def compute_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 hex digest of a file, streamed."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    timeout_seconds: float = 60.0,
) -> Path:
    """Stream a URL to disk via a ``.part`` file and atomic rename."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "spritelab-harvest/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        content_type = str(response.headers.get("Content-Type", ""))
        if "text/html" in content_type.lower():
            raise ValueError(
                f"URL returned HTML instead of a file ({content_type}); "
                "this is probably a landing page, not a direct download."
            )
        total = response.headers.get("Content-Length")
        progress = _make_progress(int(total) if total else None, url)
        try:
            with part_path.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()

    if output_path.exists():
        output_path.unlink()
    shutil.move(str(part_path), str(output_path))
    return output_path


def _make_progress(total: int | None, url: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, unit="B", unit_scale=True, desc=Path(url).name or "download")
