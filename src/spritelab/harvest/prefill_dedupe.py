"""Duplicate grouping so each unique sprite image is VLM-labeled once."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spritelab.data.dedupe_report import (
    average_hash_image,
    decoded_rgba_sha256,
    difference_hash_image,
    hamming_distance_hex,
)

if TYPE_CHECKING:
    from spritelab.harvest.pipeline import HarvestedSprite


@dataclass(frozen=True)
class PrefillGroup:
    """One VLM call per group: the representative is labeled, members reuse it."""

    representative_index: int
    member_indices: tuple[int, ...]
    kind: str  # "single", "exact", or "near"


def group_sprites_for_prefill(
    harvested: Sequence[HarvestedSprite],
    selected_indices: Sequence[int],
    *,
    exact_duplicates: bool = True,
    near_duplicates: bool = False,
    near_dup_threshold: int = 2,
) -> list[PrefillGroup]:
    """Group selected sprites by decoded-RGBA hash, optionally near-dup hashes.

    Deterministic: the representative is the member with the lowest sprite_id.
    Near-duplicate merging compares perceptual hashes across exact groups
    (quadratic in unique images, intended for per-run scales).
    """

    if near_dup_threshold < 0:
        raise ValueError("near_dup_threshold must be non-negative.")

    exact_groups: dict[str, list[int]] = {}
    decoded_key_by_group: dict[str, str] = {}
    order: list[str] = []
    for index in selected_indices:
        bundle = harvested[index].imported.bundle
        if bundle is None:
            continue
        decoded_key = decoded_rgba_sha256(bundle)
        key = decoded_key if exact_duplicates else f"{decoded_key}:{index}"
        if key not in exact_groups:
            exact_groups[key] = []
            decoded_key_by_group[key] = decoded_key
            order.append(key)
        exact_groups[key].append(index)

    merged: list[tuple[list[str], str]] = [
        ([key], "exact" if len(exact_groups[key]) > 1 else "single") for key in order
    ]
    if near_duplicates and len(order) > 1:
        merged = _merge_near_duplicates(
            harvested,
            exact_groups,
            order,
            near_dup_threshold,
            decoded_key_by_group=decoded_key_by_group,
            exact_duplicates=exact_duplicates,
        )

    groups: list[PrefillGroup] = []
    for keys, kind in merged:
        indices = sorted(
            (index for key in keys for index in exact_groups[key]),
            key=lambda index: (harvested[index].final_item.sprite_id, index),
        )
        groups.append(
            PrefillGroup(
                representative_index=indices[0],
                member_indices=tuple(indices),
                kind=kind if len(indices) > 1 else "single",
            )
        )
    groups.sort(key=lambda group: group.representative_index)
    return groups


def _merge_near_duplicates(
    harvested: Sequence[HarvestedSprite],
    exact_groups: dict[str, list[int]],
    order: list[str],
    threshold: int,
    *,
    decoded_key_by_group: dict[str, str],
    exact_duplicates: bool,
) -> list[tuple[list[str], str]]:
    from spritelab.codec.reconstruct import reconstruct_rgba

    fingerprints: dict[str, tuple[str, str]] = {}
    for key in order:
        bundle = harvested[exact_groups[key][0]].imported.bundle
        image = reconstruct_rgba(bundle)
        fingerprints[key] = (average_hash_image(image), difference_hash_image(image))

    parent = {key: key for key in order}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for first_position, key_a in enumerate(order):
        ahash_a, dhash_a = fingerprints[key_a]
        for key_b in order[first_position + 1 :]:
            if not exact_duplicates and decoded_key_by_group[key_a] == decoded_key_by_group[key_b]:
                continue
            ahash_b, dhash_b = fingerprints[key_b]
            distance = min(
                hamming_distance_hex(ahash_a, ahash_b),
                hamming_distance_hex(dhash_a, dhash_b),
            )
            if distance <= threshold:
                parent[find(key_b)] = find(key_a)

    components: dict[str, list[str]] = {}
    for key in order:
        components.setdefault(find(key), []).append(key)

    merged: list[tuple[list[str], str]] = []
    for root in order:
        if root not in components:
            continue
        keys = components[root]
        if len(keys) > 1:
            kind = "near"
        elif len(exact_groups[keys[0]]) > 1:
            kind = "exact"
        else:
            kind = "single"
        merged.append((keys, kind))
    return merged
