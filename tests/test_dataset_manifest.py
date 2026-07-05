from __future__ import annotations

from pathlib import Path

from spritelab.data.manifest import (
    DatasetManifest,
    IngestedSpriteRecord,
    load_manifest,
    save_manifest,
)


def test_manifest_save_load_roundtrip(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset_name="items_v0",
        records=[
            IngestedSpriteRecord(
                id="mushroom_red",
                source_path="data/raw/mushroom_red.png",
                bundle_dir="data/processed/items_v0/bundles/mushroom_red",
                width=32,
                height=32,
                category="item_icon",
                subtype=None,
                license="CC0",
                palette_size=4,
                sha256="abc123",
                split="train",
            )
        ],
        rejected_count=1,
        total_seen=2,
        options={"category": "item_icon", "max_visible_colors": 32, "canonicalize_palette": True},
    )

    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded == manifest
    assert loaded.records[0].palette_size == 4
    assert loaded.rejected_count == 1
    assert loaded.options["canonicalize_palette"] is True
