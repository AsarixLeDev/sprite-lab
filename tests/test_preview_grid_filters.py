from __future__ import annotations

from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, save_manifest
from spritelab.data.preview_grid import (
    PreviewGridOptions,
    PreviewGridRecord,
    create_preview_grid,
    filter_preview_records,
    sort_preview_records,
)


def test_filter_by_category_split_id_and_palette_size() -> None:
    records = [
        PreviewGridRecord("red_mushroom", Path("a"), None, "item_icon", "train", 4),
        PreviewGridRecord("blue_mushroom", Path("b"), None, "npc", "val", 8),
        PreviewGridRecord("green_leaf", Path("c"), None, "item_icon", "test", None),
    ]

    assert [r.id for r in filter_preview_records(records, filter_category="item_icon")] == [
        "red_mushroom",
        "green_leaf",
    ]
    assert [r.id for r in filter_preview_records(records, filter_split="val")] == ["blue_mushroom"]
    assert [r.id for r in filter_preview_records(records, filter_id_contains="MUSH")] == [
        "red_mushroom",
        "blue_mushroom",
    ]
    assert [r.id for r in filter_preview_records(records, min_palette_size=5)] == ["blue_mushroom"]
    assert [r.id for r in filter_preview_records(records, max_palette_size=4)] == ["red_mushroom"]


def test_sort_by_id_palette_size_and_descending() -> None:
    records = [
        PreviewGridRecord("b", Path("b"), None, None, "test", 8, source_path="raw/b.png"),
        PreviewGridRecord("a", Path("a"), None, None, "train", 4, source_path="raw/a.png"),
        PreviewGridRecord("c", Path("c"), None, None, None, None, source_path=None),
    ]

    assert [r.id for r in sort_preview_records(records, sort_by="id")] == ["a", "b", "c"]
    assert [r.id for r in sort_preview_records(records, sort_by="palette_size")] == ["a", "b", "c"]
    assert [r.id for r in sort_preview_records(records, sort_by="id", descending=True)] == ["c", "b", "a"]


def test_max_items_is_applied_after_sorting_and_filtering(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    records = []
    for sprite_id, palette_size in [("c", 3), ("a", 1), ("b", 2)]:
        bundle_dir = _write_bundle(dataset_dir / "bundles" / sprite_id, sprite_id, palette_size)
        records.append(
            IngestedSpriteRecord(
                id=sprite_id,
                source_path=f"raw/{sprite_id}.png",
                bundle_dir=str(bundle_dir),
                width=32,
                height=32,
                category="item_icon",
                subtype=None,
                license=None,
                palette_size=palette_size,
                sha256="0" * 64,
            )
        )
    save_manifest(
        DatasetManifest("dataset", records=records, rejected_count=0, total_seen=3, options={}),
        dataset_dir / "manifest.json",
    )

    grid = create_preview_grid(
        PreviewGridOptions(
            dataset_path=dataset_dir,
            output_path=tmp_path / "grid.png",
            scale=1,
            columns=1,
            padding=0,
            label=False,
            max_items=2,
            sort_by="id",
        )
    )

    assert grid.size == (32, 64)


def _write_bundle(bundle_dir: Path, sprite_id: str, palette_size: int) -> Path:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[0, 0] = 1
    index_map[0, 0] = 1
    visible_rows = [[index, 0, 0] for index in range(1, palette_size + 1)]
    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], *visible_rows], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id, category="item_icon", palette_size=palette_size),
    )
    save_bundle(bundle, bundle_dir)
    return bundle_dir
