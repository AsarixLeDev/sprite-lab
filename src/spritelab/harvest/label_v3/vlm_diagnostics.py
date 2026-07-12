"""Bounded, credential-safe diagnostics for failed VLM responses."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\brpa_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE),
)


def _sanitize(text: str) -> str:
    value = " ".join(str(text).replace("\x00", " ").split())
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "<redacted>", value)
    return value


def write_failure_diagnostic(
    root: str | Path | None,
    *,
    enabled: bool,
    provider: str,
    model: str,
    stage: str,
    content: str | bytes = "",
    status_code: int | None = None,
    content_type: str = "",
    exception: BaseException | str,
    prompt_hash: str,
    model_hash: str,
    cache_hash: str,
    excerpt_chars: int = 160,
) -> Path | None:
    if not enabled or not root:
        return None
    raw = content if isinstance(content, bytes) else str(content).encode("utf-8", errors="replace")
    decoded = raw.decode("utf-8", errors="replace")
    limit = max(32, min(512, int(excerpt_chars)))
    sanitized = _sanitize(decoded)
    first = sanitized[:limit]
    last = sanitized[-limit:] if len(sanitized) > limit else first
    response_sha256 = hashlib.sha256(raw).hexdigest()
    artifact: dict[str, Any] = {
        "schema_version": "vlm_failure_diagnostic_v1",
        "provider": _sanitize(provider)[:128],
        "model": _sanitize(model)[:256],
        "stage": _sanitize(stage)[:96],
        "status_code": status_code,
        "response_content_type": _sanitize(content_type)[:128],
        "response_length": len(raw),
        "response_sha256": response_sha256,
        "first_excerpt": first,
        "last_excerpt": last,
        "exception_type": _sanitize(type(exception).__name__ if isinstance(exception, BaseException) else exception)[
            :128
        ],
        "prompt_hash": _sanitize(prompt_hash)[:128],
        "model_hash": _sanitize(model_hash)[:128],
        "cache_hash": _sanitize(cache_hash)[:128],
    }
    destination = Path(root) / _sanitize(stage).replace("/", "_")
    destination.mkdir(parents=True, exist_ok=True)
    name = f"{artifact['cache_hash'] or 'no_cache'}_{response_sha256[:12]}_{artifact['exception_type']}.json"
    path = destination / name
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return path
