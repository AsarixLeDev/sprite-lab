"""Detection of source metadata that could bias a later semantic decision."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

TAINT_POLICY_VERSION = "filename_taint_v1"

# These tokens are evidence hazards, not asserted labels.  A match means the
# metadata must stay outside the blind boundary and be reconciled afterward.
DEFAULT_SEMANTIC_TOKENS = frozenset(
    {
        "armor",
        "armour",
        "axe",
        "boot",
        "cap",
        "crystal",
        "gem",
        "helmet",
        "key",
        "mineral",
        "ore",
        "plant",
        "rod",
        "shield",
        "sword",
        "tool",
        "weapon",
    }
)


def detect_filename_taint(
    source_filename: str,
    *,
    semantic_tokens: Iterable[str] = DEFAULT_SEMANTIC_TOKENS,
) -> dict[str, Any]:
    """Mark possible semantic evidence without interpreting it as truth."""

    basename = PurePosixPath(source_filename.replace("\\", "/")).name
    stem = basename.rsplit(".", 1)[0]
    tokens = [token.casefold() for token in re.findall(r"[A-Za-z]+", stem)]
    controlled = {str(token).casefold() for token in semantic_tokens}
    matches = sorted(set(tokens) & controlled)
    return {
        "policy_version": TAINT_POLICY_VERSION,
        "semantic_tokens": matches,
        "status": "tainted_metadata" if matches else "no_known_semantic_token",
    }


def reconcile_metadata_taint(
    taint: Mapping[str, Any],
    blind_label: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Record conflicts after blind labeling without overwriting the label."""

    semantic_tokens = {str(value).casefold() for value in taint.get("semantic_tokens", [])}
    values = set()
    if blind_label:
        for key in ("category", "canonical_object", "role", "explicit_material"):
            value = blind_label.get(key)
            if isinstance(value, str):
                values.update(re.findall(r"[a-z]+", value.casefold()))
    return {
        "blind_label_unchanged": True,
        "filename_taint_status": taint.get("status", "unknown"),
        "metadata_conflict": bool(semantic_tokens and values and semantic_tokens.isdisjoint(values)),
        "semantic_tokens": sorted(semantic_tokens),
    }
