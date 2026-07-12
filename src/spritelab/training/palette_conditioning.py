"""Canonical palette conditioning features and simple real-palette retrieval.

The condition is deliberately slot-aligned: canonical palette slot position keeps
its existing role/luminance meaning, while the explicit role feature makes rows
with sparse or imperfect canonicalisation less ambiguous.  Slot zero is always
represented but never treated as a visible colour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.training.palette_swap import nearest_family

PALETTE_CONDITION_SLOTS = 16
PALETTE_CONDITION_CHANNELS = 6  # OKLab L,a,b; presence; coverage; normalized role id.


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0, 1] to OKLab without changing array shape."""
    value = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    lms = np.einsum(
        "...c,dc->...d",
        linear,
        np.asarray(
            (
                (0.4122214708, 0.5363325363, 0.0514459929),
                (0.2119034982, 0.6806995451, 0.1073969566),
                (0.0883024619, 0.2817188376, 0.6299787005),
            ),
            dtype=np.float32,
        ),
    )
    lms = np.cbrt(np.clip(lms, 0.0, None))
    return np.einsum(
        "...c,dc->...d",
        lms,
        np.asarray(
            (
                (0.2104542553, 0.7936177850, -0.0040720468),
                (1.9779984951, -2.4285922050, 0.4505937099),
                (0.0259040371, 0.7827717662, -0.8086757660),
            ),
            dtype=np.float32,
        ),
    ).astype(np.float32)


def build_palette_condition(
    *,
    palette_rgb: np.ndarray | None,
    palette_mask: np.ndarray | None,
    index_map: np.ndarray | None,
    role_map: np.ndarray | None,
    allow_missing: bool = False,
) -> np.ndarray:
    """Build a ``[16, 6]`` float32 condition from indexed sprite data.

    Missing source fields are an error unless an explicit all-zero fallback is
    requested.  This prevents a silently unconditioned v3 training run.
    """
    required = {"palette_rgb": palette_rgb, "palette_mask": palette_mask, "index_map": index_map, "role_map": role_map}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        if allow_missing:
            return np.zeros((PALETTE_CONDITION_SLOTS, PALETTE_CONDITION_CHANNELS), dtype=np.float32)
        raise ValueError(f"palette conditioning requires {', '.join(missing)}")
    palette = np.asarray(palette_rgb, dtype=np.float32)
    mask = np.asarray(palette_mask, dtype=bool).reshape(-1)
    index = np.asarray(index_map, dtype=np.int64)
    roles = np.asarray(role_map, dtype=np.int64)
    if palette.ndim != 2 or palette.shape[1] < 3:
        raise ValueError(f"palette conditioning palette_rgb must be [K, 3], got {palette.shape}")
    if index.shape != roles.shape:
        raise ValueError("palette conditioning index_map and role_map must have identical shapes")
    if palette.size and float(np.nanmax(palette[:, :3])) > 1.5:
        palette = palette / 255.0
    result = np.zeros((PALETTE_CONDITION_SLOTS, PALETTE_CONDITION_CHANNELS), dtype=np.float32)
    count_total = float(max(1, index.size))
    for slot in range(min(PALETTE_CONDITION_SLOTS, palette.shape[0], mask.size)):
        if slot == 0 or not bool(mask[slot]):
            continue
        slot_pixels = index == slot
        coverage = float(np.count_nonzero(slot_pixels)) / count_total
        # A mask entry alone is not enough: the condition describes the actual
        # indexed sprite, so unused slots must not claim presence.
        if coverage <= 0.0:
            continue
        lab = rgb_to_oklab(palette[slot, :3])
        role_values = roles[slot_pixels]
        role = int(np.bincount(np.clip(role_values, 0, 9), minlength=10).argmax()) if role_values.size else 0
        result[slot, :3] = lab
        result[slot, 3] = 1.0
        result[slot, 4] = coverage
        result[slot, 5] = float(role) / 9.0
    return result


def prompt_color_family(record: Mapping[str, Any]) -> str:
    """Best-effort requested family from structured fields, then text."""
    candidates: list[str] = []
    for key in ("primary_color", "color", "color_family"):
        value = record.get(key)
        if isinstance(value, str):
            candidates.append(value.lower())
    conditioning = record.get("conditioning")
    if isinstance(conditioning, Mapping):
        for key in ("primary_color", "color", "color_family"):
            value = conditioning.get(key)
            if isinstance(value, str):
                candidates.append(value.lower())
        semantic = conditioning.get("semantic_v3")
        if isinstance(semantic, Mapping):
            values = semantic.get("colors") or semantic.get("color") or ()
            if isinstance(values, str):
                candidates.append(values.lower())
            elif isinstance(values, Sequence):
                candidates.extend(str(value).lower() for value in values)
    text = " ".join(str(record.get(key) or "") for key in ("prompt", "caption")).lower()
    candidates.extend(text.split())
    for family in ("red", "blue", "green", "yellow", "purple", "brown", "gold", "gray", "black", "white"):
        if family in candidates:
            return family
    return ""


@dataclass(frozen=True)
class PaletteConditionEntry:
    sprite_id: str
    family: str
    category: str
    condition: np.ndarray


class PaletteConditionLibrary:
    def __init__(self, entries: Sequence[PaletteConditionEntry]) -> None:
        self.entries = tuple(entries)

    def retrieve(self, record: Mapping[str, Any], *, exclude_sprite_id: bool = False) -> PaletteConditionEntry | None:
        family = prompt_color_family(record)
        category = str(record.get("category") or "")
        target_id = str(record.get("sprite_id") or record.get("prompt_id") or "")
        candidates = [
            entry for entry in self.entries if not (exclude_sprite_id and target_id and entry.sprite_id == target_id)
        ]
        by_family = [entry for entry in candidates if family and entry.family == family]
        candidates = by_family or candidates
        by_category = [entry for entry in candidates if category and entry.category == category]
        return (by_category or candidates or [None])[0]


def build_palette_condition_library(
    dataset_dir: Path, records: Sequence[Mapping[str, Any]], *, max_entries: int = 0
) -> PaletteConditionLibrary:
    """Build a deterministic real-palette library from manifest rows."""
    root = Path(dataset_dir)
    cache: dict[str, dict[str, np.ndarray]] = {}
    entries: list[PaletteConditionEntry] = []
    for record in records:
        if max_entries > 0 and len(entries) >= int(max_entries):
            break
        npz_file = str(record.get("npz_file") or f"{record.get('split', '')}.npz")
        path = root / npz_file
        if not path.is_file():
            continue
        if npz_file not in cache:
            with np.load(path, allow_pickle=False) as loaded:
                cache[npz_file] = {key: np.asarray(loaded[key]) for key in loaded.files}
        arrays = cache[npz_file]
        row = int(record.get("npz_row", -1))
        if row < 0 or row >= len(arrays.get("palette", ())):
            continue
        try:
            condition = build_palette_condition(
                palette_rgb=arrays["palette"][row],
                palette_mask=arrays["palette_mask"][row],
                index_map=arrays["index_map"][row],
                role_map=arrays["role_map"][row],
            )
        except (KeyError, ValueError):
            continue
        visible = condition[:, 3] > 0.0
        if not bool(visible.any()):
            continue
        dominant = int(np.argmax(condition[:, 4]))
        palette = np.asarray(arrays["palette"][row], dtype=np.float32)
        if palette.size and float(np.nanmax(palette)) > 1.5:
            palette = palette / 255.0
        entries.append(
            PaletteConditionEntry(
                sprite_id=str(record.get("sprite_id") or ""),
                family=nearest_family(palette[dominant, :3]),
                category=str(record.get("category") or ""),
                condition=condition,
            )
        )
    return PaletteConditionLibrary(entries)
