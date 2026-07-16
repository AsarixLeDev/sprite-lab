"""Deterministic request and response identities for safe resume behavior."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from spritelab.product_features.providers.contracts import ImageInput


def image_digest(image: ImageInput) -> str:
    return hashlib.sha256(image.data).hexdigest()


def request_identity(
    provider_id: str,
    model_id: str,
    prompt: str,
    images: tuple[ImageInput, ...],
    *,
    options: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema": "spritelab.vision.request-identity.v1",
        "provider_id": provider_id,
        "model_id": model_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "images": [{"image_id": image.image_id, "sha256": image_digest(image)} for image in images],
        "options": dict(options or {}),
    }
    return _hash_json(payload)


def response_identity(request_id: str, payload: object) -> str:
    return _hash_json({"schema": "spritelab.vision.response-identity.v1", "request_id": request_id, "payload": payload})


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
