"""Auto-Labeling v3: hashing and content-addressing utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_short(data: bytes | str, length: int = 16) -> str:
    return sha256_hex(data)[:length]


def dict_hash(data: Mapping[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
    return sha256_short(canonical)


def file_hash(path: str | Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def image_rgba_hash(rgba_bytes: bytes, width: int, height: int) -> str:
    digest = hashlib.sha256()
    digest.update(width.to_bytes(2, "big"))
    digest.update(height.to_bytes(2, "big"))
    digest.update(rgba_bytes)
    return digest.hexdigest()[:16]


def config_identity_hash(
    taxonomy_hash: str = "",
    prompt_hash: str = "",
    model_identity: str = "",
    image_view: str = "",
    calibration_hash: str = "",
    fusion_policy_hash: str = "",
    contradiction_policy_hash: str = "",
    source_profiles_hash: str = "",
    sheet_mapping_hash: str = "",
) -> str:
    """Produce a composite config hash from all relevant component hashes."""
    parts = [
        taxonomy_hash,
        prompt_hash,
        model_identity,
        image_view,
        calibration_hash,
        fusion_policy_hash,
        contradiction_policy_hash,
        source_profiles_hash,
        sheet_mapping_hash,
    ]
    return sha256_short("|".join(parts))


def deterministic_stage_hash(stage_name: str, config: Mapping[str, Any]) -> str:
    data = json.dumps({"stage": stage_name, "config": config}, sort_keys=True, default=str)
    return sha256_short(data)


def stable_evidence_id(
    sprite_id: str,
    evidence_family: str,
    stage_hash: str,
    *,
    variant: str = "",
) -> str:
    seed = f"{sprite_id}|{evidence_family}|{stage_hash}|{variant}"
    return sha256_short(seed, length=12)
