"""Deterministic palette canonicalization for sprite bundles.

The project convention is that palette slot 0 is a dummy transparent RGB value
and ``index_map == 0`` means transparent. This module never moves slot 0.
Visible slots are sorted with a small explainable heuristic so future indexed
models can rely on more stable palette positions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from spritelab.codec.bundle import SPRITE_HEIGHT, SPRITE_WIDTH, SpriteBundle, SpriteMetadata
from spritelab.codec.color import rgb_hue_degrees, rgb_saturation, srgb_luminance
from spritelab.codec.role_inference import infer_palette_slot_roles_v2
from spritelab.codec.roles import role_name
from spritelab.codec.validate import assert_valid_bundle

CANONICALIZER_VERSION = "v1"

ROLE_BUCKET_ORDER = {
    "outline": 0,
    "deep_shadow": 1,
    "shadow": 2,
    "midtone": 3,
    "light": 4,
    "highlight": 5,
    "accent": 6,
    "emissive": 7,
    "texture_detail": 8,
    "unknown": 9,
    "unused": 9,
}


@dataclass(frozen=True)
class PaletteSlotStats:
    """Debuggable statistics for one visible palette slot."""

    slot: int
    rgb: tuple[int, int, int]
    count: int
    frequency: float
    luminance: float
    saturation: float
    edge_contact_ratio: float
    alpha_neighbor_ratio: float
    mean_x: float
    mean_y: float
    role_hint: str
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class CanonicalizationResult:
    """Result of canonicalizing a bundle palette."""

    bundle: SpriteBundle
    old_to_new: dict[int, int]
    new_to_old: dict[int, int]
    slot_stats: list[PaletteSlotStats]
    warnings: list[str]


def compute_palette_slot_stats(bundle: SpriteBundle) -> list[PaletteSlotStats]:
    """Compute deterministic per-slot statistics for visible palette slots.

    ``edge_contact_ratio`` is the share of pixels for a slot that touch
    transparency in an 8-neighbor search. ``alpha_neighbor_ratio`` is currently
    the same v1 signal and is kept as a named field for future refinement.
    """

    assert_valid_bundle(bundle)

    total_opaque = int(np.count_nonzero(bundle.alpha == 1))
    role_result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)
    stats: list[PaletteSlotStats] = []

    for slot in range(1, int(bundle.palette.shape[0])):
        rgb = _rgb_tuple(bundle.palette[slot])
        feature = role_result.slot_features[slot]
        count = feature.pixel_count
        frequency = count / total_opaque if total_opaque > 0 else 0.0
        luminance = srgb_luminance(rgb)
        saturation = rgb_saturation(rgb)
        edge_contact_ratio = feature.edge_contact_ratio
        mean_x = feature.mean_x
        mean_y = feature.mean_y
        role_hint = "unused" if count == 0 else role_name(role_result.slot_roles.get(slot, 255))
        sort_key = _sort_key(slot=slot, rgb=rgb, role_hint=role_hint, luminance=luminance, saturation=saturation)

        stats.append(
            PaletteSlotStats(
                slot=slot,
                rgb=rgb,
                count=count,
                frequency=frequency,
                luminance=luminance,
                saturation=saturation,
                edge_contact_ratio=edge_contact_ratio,
                alpha_neighbor_ratio=edge_contact_ratio,
                mean_x=mean_x,
                mean_y=mean_y,
                role_hint=role_hint,
                sort_key=sort_key,
            )
        )

    return stats


def canonical_palette_order(bundle: SpriteBundle) -> list[int]:
    """Return old palette slots in their new canonical order.

    The returned list always includes slot 0 as the first entry. Remaining
    entries are old visible slot IDs sorted into canonical order.
    """

    stats = compute_palette_slot_stats(bundle)
    return [0] + [stat.slot for stat in sorted(stats, key=lambda stat: stat.sort_key)]


def canonicalize_bundle_palette(bundle: SpriteBundle) -> CanonicalizationResult:
    """Return a copy of ``bundle`` with a canonical palette order.

    The reconstructed image is preserved exactly for valid indexed bundles:
    palette rows are reordered and every ``index_map`` value is remapped to the
    row now holding the same RGB color.
    """

    assert_valid_bundle(bundle)

    old_order = canonical_palette_order(bundle)
    old_to_new = {old_slot: new_slot for new_slot, old_slot in enumerate(old_order)}
    new_to_old = {new_slot: old_slot for new_slot, old_slot in enumerate(old_order)}

    new_palette = np.asarray(bundle.palette[old_order]).copy()
    new_index_map = remap_index_map(bundle.index_map, old_to_new)
    metadata = _metadata_with_canonicalization_info(bundle.metadata, old_to_new, new_to_old)

    canonical_bundle = SpriteBundle(
        alpha=np.asarray(bundle.alpha).copy(),
        palette=new_palette,
        index_map=new_index_map,
        role_map=None if bundle.role_map is None else np.asarray(bundle.role_map).copy(),
        metadata=metadata,
    )

    assert_valid_bundle(canonical_bundle)

    return CanonicalizationResult(
        bundle=canonical_bundle,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        slot_stats=compute_palette_slot_stats(bundle),
        warnings=[],
    )


def remap_index_map(index_map: np.ndarray, old_to_new: dict[int, int]) -> np.ndarray:
    """Return a copy of ``index_map`` with palette indices remapped."""

    remapped = np.empty_like(index_map)
    for old_slot in np.unique(index_map):
        old_int = int(old_slot)
        if old_int not in old_to_new:
            raise ValueError(f"missing palette remap entry for old slot {old_int}.")
        remapped[index_map == old_int] = old_to_new[old_int]
    return remapped


def _rgb_tuple(value: np.ndarray) -> tuple[int, int, int]:
    red, green, blue = value
    return (int(red), int(green), int(blue))


def _touches_transparency(alpha: np.ndarray, y: int, x: int) -> bool:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny = y + dy
            nx = x + dx
            if ny < 0 or nx < 0 or ny >= SPRITE_HEIGHT or nx >= SPRITE_WIDTH:
                return True
            if int(alpha[ny, nx]) == 0:
                return True
    return False


def _role_hint(
    *,
    count: int,
    frequency: float,
    luminance: float,
    saturation: float,
    edge_contact_ratio: float,
) -> str:
    if count == 0:
        return "unused"

    if luminance <= 0.30 and edge_contact_ratio >= 0.45:
        return "outline"
    if luminance <= 0.16:
        return "deep_shadow"
    if luminance <= 0.36:
        return "shadow"
    if saturation >= 0.70 and luminance >= 0.70:
        return "emissive"
    if saturation >= 0.55 and frequency <= 0.16:
        return "accent"
    if luminance >= 0.82:
        return "highlight"
    if luminance >= 0.62:
        return "light"
    if frequency <= 0.03:
        return "detail"
    return "midtone"


def _sort_key(
    *,
    slot: int,
    rgb: tuple[int, int, int],
    role_hint: str,
    luminance: float,
    saturation: float,
) -> tuple[Any, ...]:
    bucket = ROLE_BUCKET_ORDER[role_hint]
    hue = rgb_hue_degrees(rgb)

    if role_hint in {"accent", "emissive", "texture_detail", "unknown", "unused"}:
        return (bucket, hue, -saturation, luminance, rgb, slot)

    return (bucket, luminance, hue, -saturation, rgb, slot)


def _metadata_with_canonicalization_info(
    metadata: SpriteMetadata,
    old_to_new: dict[int, int],
    new_to_old: dict[int, int],
) -> SpriteMetadata:
    metadata_data = copy.deepcopy(metadata.to_dict())
    extra = dict(metadata_data.get("extra") or {})
    extra["palette_canonicalized"] = True
    extra["palette_old_to_new"] = {str(old): new for old, new in old_to_new.items()}
    extra["palette_new_to_old"] = {str(new): old for new, old in new_to_old.items()}
    extra["palette_canonicalizer_version"] = CANONICALIZER_VERSION
    metadata_data["extra"] = extra
    return SpriteMetadata.from_dict(metadata_data)
