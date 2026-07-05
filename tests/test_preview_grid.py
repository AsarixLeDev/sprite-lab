from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, save_manifest
from spritelab.data.preview_grid import (
    PreviewGridOptions,
    PreviewGridRecord,
    create_preview_grid,
    load_preview_records,
    make_preview_grid,
)


def test_make_preview_grid_dimensions_and_nearest_neighbor(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundles" / "red", "red", (255, 0, 0))
    record = PreviewGridRecord(
        id="red",
        bundle_dir=bundle_dir,
        image_path=bundle_dir / "reconstructed.png",
        category="item_icon",
        split=None,
        palette_size=1,
    )

    grid = make_preview_grid([record], scale=2, columns=1, padding=1, label=False)

    assert grid.mode == "RGBA"
    assert grid.size == (66, 66)
    assert grid.getpixel((1, 1)) == (255, 0, 0, 255)
    assert grid.getpixel((2, 2)) == (255, 0, 0, 255)
    assert grid.getpixel((3, 1)) == (32, 32, 32, 255)


def test_make_preview_grid_output_png_can_be_saved(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path / "bundles" / "blue", "blue", (0, 0, 255))
    record = PreviewGridRecord("blue", bundle_dir, bundle_dir / "reconstructed.png", None, None, 1)
    grid = make_preview_grid([record], scale=1, columns=1, padding=0, label=False)
    output = tmp_path / "grid.png"

    grid.save(output)

    with Image.open(output) as image:
        assert image.size == (32, 32)
        assert image.mode == "RGBA"


def test_load_preview_records_from_dataset_directory_manifest_and_bundles_dir(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    bundle_dir = _write_bundle(dataset_dir / "bundles" / "sprite_a", "sprite_a", (255, 0, 0))
    manifest_path = dataset_dir / "manifest.json"
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                IngestedSpriteRecord(
                    id="sprite_a",
                    source_path="raw/sprite_a.png",
                    bundle_dir=str(bundle_dir),
                    width=32,
                    height=32,
                    category="item_icon",
                    subtype=None,
                    license="CC0",
                    palette_size=1,
                    sha256="0" * 64,
                    split="train",
                )
            ],
            rejected_count=0,
            total_seen=1,
            options={},
        ),
        manifest_path,
    )

    from_dataset = load_preview_records(dataset_dir)
    from_manifest = load_preview_records(manifest_path)
    from_bundles = load_preview_records(dataset_dir / "bundles")

    assert [record.id for record in from_dataset] == ["sprite_a"]
    assert [record.id for record in from_manifest] == ["sprite_a"]
    assert [record.id for record in from_bundles] == ["sprite_a"]
    assert from_dataset[0].category == "item_icon"
    assert from_dataset[0].split == "train"


def test_create_preview_grid_saves_output_file(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    bundle_dir = _write_bundle(dataset_dir / "bundles" / "sprite_a", "sprite_a", (255, 0, 0))
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                IngestedSpriteRecord(
                    id="sprite_a",
                    source_path="raw/sprite_a.png",
                    bundle_dir=str(bundle_dir),
                    width=32,
                    height=32,
                    category="item_icon",
                    subtype=None,
                    license="CC0",
                    palette_size=1,
                    sha256="0" * 64,
                )
            ],
            rejected_count=0,
            total_seen=1,
            options={},
        ),
        dataset_dir / "manifest.json",
    )
    output = tmp_path / "grid.png"

    grid = create_preview_grid(
        PreviewGridOptions(
            dataset_path=dataset_dir,
            output_path=output,
            scale=1,
            columns=1,
            padding=0,
            label=False,
        )
    )

    assert output.exists()
    assert grid.size == (32, 32)


def _write_bundle(bundle_dir: Path, sprite_id: str, color: tuple[int, int, int]) -> Path:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[0, 0] = 1
    index_map[0, 0] = 1
    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], list(color)], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id, category="item_icon", palette_size=1),
    )
    save_bundle(bundle, bundle_dir)
    return bundle_dir
