"""Duplicate propagation helpers for label v2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class LabelDuplicateGroup:
    representative_sprite_id: str
    member_sprite_ids: tuple[str, ...]
    member_indices: tuple[int, ...]
    kind: str = "exact"
    rgba_sha256: str = ""

    @property
    def representative_index(self) -> int:
        return self.member_indices[0]


def exact_rgba_sha256(path: str | Path) -> str:
    """SHA256 of decoded RGBA bytes, independent of PNG encoding details."""

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(rgba.size[0].to_bytes(2, "big"))
    digest.update(rgba.size[1].to_bytes(2, "big"))
    digest.update(rgba.tobytes())
    return digest.hexdigest()


def group_label_records_by_exact_rgba(
    records: Sequence[Mapping[str, Any]],
    *,
    run_dir: str | Path | None = None,
) -> list[LabelDuplicateGroup]:
    """Group records by decoded RGBA hash with deterministic representatives."""

    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        path = _resolve_path(record, run_dir=run_dir)
        if path is None or not path.exists():
            key = f"missing:{index}"
        else:
            key = exact_rgba_sha256(path)
        groups.setdefault(key, []).append(index)

    result: list[LabelDuplicateGroup] = []
    for key, indices in groups.items():
        sorted_indices = tuple(sorted(indices, key=lambda item: (str(records[item].get("sprite_id", "")), item)))
        member_ids = tuple(str(records[index].get("sprite_id", "")) for index in sorted_indices)
        result.append(
            LabelDuplicateGroup(
                representative_sprite_id=member_ids[0] if member_ids else "",
                member_sprite_ids=member_ids,
                member_indices=sorted_indices,
                kind="exact" if len(sorted_indices) > 1 and not key.startswith("missing:") else "single",
                rgba_sha256="" if key.startswith("missing:") else key,
            )
        )
    result.sort(key=lambda group: group.representative_index)
    return result


def duplicate_metadata_for_member(group: LabelDuplicateGroup, sprite_id: str) -> dict[str, Any]:
    """Return additive JSON metadata for a propagated duplicate member."""

    if group.kind != "exact" or sprite_id == group.representative_sprite_id:
        return {}
    return {
        "prefill_propagated_from": group.representative_sprite_id,
        "duplicate_group_size": len(group.member_sprite_ids),
        "duplicate_propagation": "exact",
    }


def _resolve_path(record: Mapping[str, Any], *, run_dir: str | Path | None) -> Path | None:
    raw = str(record.get("final_png_path") or record.get("path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    if run_dir is not None:
        run_path = Path(run_dir)
        candidates.extend([run_path / path, run_path.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]
