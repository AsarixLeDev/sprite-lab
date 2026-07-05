from __future__ import annotations

from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.data.dedupe_report import (
    bundle_content_sha256,
    decoded_rgba_sha256,
    sha256_bytes,
    sha256_file,
)


def test_sha256_bytes_is_deterministic() -> None:
    assert sha256_bytes(b"sprite") == sha256_bytes(b"sprite")
    assert sha256_bytes(b"sprite") != sha256_bytes(b"sprite!")


def test_sha256_file_changes_when_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "source.png"
    path.write_bytes(b"first")
    first = sha256_file(path)

    path.write_bytes(b"second")
    second = sha256_file(path)

    assert first != second


def test_decoded_rgba_hash_matches_when_palette_order_differs() -> None:
    first = _two_color_bundle(
        "first",
        palette=np.array([[0, 0, 0], [255, 0, 0], [5, 5, 5]], dtype=np.uint8),
        left_slot=1,
        right_slot=2,
    )
    reordered = _two_color_bundle(
        "reordered",
        palette=np.array([[0, 0, 0], [5, 5, 5], [255, 0, 0]], dtype=np.uint8),
        left_slot=2,
        right_slot=1,
    )

    assert decoded_rgba_sha256(first) == decoded_rgba_sha256(reordered)
    assert bundle_content_sha256(first) != bundle_content_sha256(reordered)


def test_bundle_content_hash_matches_identical_arrays_and_differs_on_index_change() -> None:
    first = _two_color_bundle(
        "first",
        palette=np.array([[0, 0, 0], [255, 0, 0], [5, 5, 5]], dtype=np.uint8),
        left_slot=1,
        right_slot=2,
    )
    same_arrays_different_metadata = _two_color_bundle(
        "different_metadata",
        palette=np.array([[0, 0, 0], [255, 0, 0], [5, 5, 5]], dtype=np.uint8),
        left_slot=1,
        right_slot=2,
    )
    changed_index = _two_color_bundle(
        "changed",
        palette=np.array([[0, 0, 0], [255, 0, 0], [5, 5, 5]], dtype=np.uint8),
        left_slot=2,
        right_slot=2,
    )

    assert bundle_content_sha256(first) == bundle_content_sha256(same_arrays_different_metadata)
    assert bundle_content_sha256(first) != bundle_content_sha256(changed_index)


def _two_color_bundle(
    sprite_id: str,
    *,
    palette: np.ndarray,
    left_slot: int,
    right_slot: int,
) -> SpriteBundle:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[15, 15] = 1
    alpha[15, 16] = 1
    index_map[15, 15] = left_slot
    index_map[15, 16] = right_slot
    return SpriteBundle(
        alpha=alpha,
        palette=palette,
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id, palette_size=2),
    )
