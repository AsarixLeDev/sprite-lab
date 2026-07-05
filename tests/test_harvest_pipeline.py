"""Tests for spritelab.harvest.pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from _harvest_testdata import make_sheet_png, make_source, make_sprite_png, make_zip_of_pngs

from spritelab.harvest.pipeline import (
    HarvestImportOptions,
    export_harvested_dataset,
    harvest_source_to_imported_sprites,
)


def test_manual_zip_source_imports(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png", "b.png"])
    source = make_source("zip_source", source_type="manual_zip", local_archive_path=str(zip_path))
    harvested = harvest_source_to_imported_sprites(
        source, options=HarvestImportOptions(), work_dir=tmp_path / "work"
    )
    valid = [s for s in harvested if s.imported.bundle is not None]
    assert len(valid) == 2


def test_local_directory_source_imports(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "one.png")
    make_sprite_png(root / "two.png")
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(
        source, options=HarvestImportOptions(), work_dir=tmp_path / "work"
    )
    assert len([s for s in harvested if s.imported.bundle is not None]) == 2


def test_wrong_size_sliced_or_padded(tmp_path):
    root = tmp_path / "pngs"
    make_sheet_png(root / "sheet.png", rows=2, cols=2)  # 64x64 -> 4 tiles
    make_sprite_png(root / "small.png", size=16)  # -> center-padded
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(
        source, options=HarvestImportOptions(), work_dir=tmp_path / "work"
    )
    valid = [s for s in harvested if s.imported.bundle is not None]
    assert len(valid) == 5
    for sprite in valid:
        assert np.asarray(sprite.imported.bundle.alpha).shape == (32, 32)


def test_invalid_images_rejected_not_crash(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "good.png")
    bad = root / "bad.png"
    bad.write_bytes(b"not a png at all")
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(
        source, options=HarvestImportOptions(), work_dir=tmp_path / "work"
    )
    statuses = {s.final_item.sprite_id: s.final_item.status for s in harvested}
    assert any(status == "accepted" for status in statuses.values())
    assert any(status == "rejected" for status in statuses.values())


def test_source_metadata_propagates(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "one.png")
    source = make_source("kenney_pack", license_name="cc0", author="Kenney", local_root_path=str(root), source_type="kenney")
    harvested = harvest_source_to_imported_sprites(
        source, options=HarvestImportOptions(), work_dir=tmp_path / "work"
    )
    item = harvested[0].final_item
    assert item.license == "cc0"
    assert item.author == "Kenney"
    assert item.source_name == source.source_name
    assert item.sprite_id.startswith("kenney_pack_")


def _harvested(tmp_path, license_name):
    root = tmp_path / f"pngs_{license_name}"
    for index in range(3):
        make_sprite_png(root / f"sprite_{index}.png", colors=2 + index % 2)
    source = make_source(f"src_{license_name}", license_name=license_name, local_root_path=str(root))
    return harvest_source_to_imported_sprites(
        source, options=HarvestImportOptions(), work_dir=tmp_path / f"work_{license_name}"
    )


def test_export_blocks_unknown_license(tmp_path):
    harvested = _harvested(tmp_path, "unknown")
    with pytest.raises(ValueError, match="license"):
        export_harvested_dataset(
            harvested,
            dataset_name="blocked",
            output_root=tmp_path / "datasets",
        )


def test_export_allows_unknown_with_override(tmp_path):
    harvested = _harvested(tmp_path, "unknown")
    result = export_harvested_dataset(
        harvested,
        dataset_name="override",
        output_root=tmp_path / "datasets",
        allow_unknown_license=True,
    )
    assert result.accepted_count == 3
    manifest = (result.output_dir / "manifest_train.jsonl").read_text(encoding="utf-8")
    assert "license_override" in manifest


def test_export_uses_dataset_maker_format(tmp_path):
    harvested = _harvested(tmp_path, "cc0")
    result = export_harvested_dataset(
        harvested,
        dataset_name="cc0_pack",
        output_root=tmp_path / "datasets",
    )
    npz_path = result.output_dir / "train.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as data:
        assert set(data.files) >= {
            "alpha",
            "index_map",
            "role_map",
            "palette",
            "palette_mask",
            "category_id",
            "sprite_id",
        }
    assert (result.output_dir / "dataset_config.json").exists()
    assert (result.output_dir / "vocab.json").exists()
