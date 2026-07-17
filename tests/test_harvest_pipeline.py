"""Tests for spritelab.harvest.pipeline."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from _harvest_testdata import make_sheet_png, make_source, make_sprite_png, make_zip_of_pngs
from spritelab.harvest.download import compute_sha256
from spritelab.harvest.pipeline import (
    HarvestImportOptions,
    export_harvested_dataset,
    harvest_source_to_imported_sprites,
)


def test_manual_zip_source_imports(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png", "b.png"])
    source = make_source("zip_source", source_type="manual_zip", local_archive_path=str(zip_path))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
    valid = [s for s in harvested if s.imported.bundle is not None]
    assert len(valid) == 2


def test_local_archive_is_revalidated_instead_of_reusing_stale_extraction(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png"])
    source = make_source("zip_source", source_type="manual_zip", local_archive_path=str(zip_path))
    work_dir = tmp_path / "work"

    first = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=work_dir)
    assert [sprite.candidate.relative_path for sprite in first] == ["a.png"]

    make_zip_of_pngs(zip_path, ["b.png"])
    second = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=work_dir)

    assert [sprite.candidate.relative_path for sprite in second] == ["b.png"]
    assert not (work_dir / "extracted" / "zip_source" / "a.png").exists()


def test_checksum_mismatch_preserves_previous_extraction(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png"])
    expected_digest = compute_sha256(zip_path)
    source = make_source(
        "zip_source",
        source_type="manual_zip",
        local_archive_path=str(zip_path),
        download_sha256=expected_digest,
    )
    work_dir = tmp_path / "work"
    harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=work_dir)

    make_zip_of_pngs(zip_path, ["b.png"])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=work_dir)

    extracted = work_dir / "extracted" / "zip_source"
    assert (extracted / "a.png").exists()
    assert not (extracted / "b.png").exists()


def test_remote_cache_is_bound_to_url_kind_hash_and_size(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_download(url, output_path, **_kwargs):
        calls.append(url)
        output_path = Path(output_path)
        tag = "one" if url.endswith("one.zip") else "two"
        png = make_sprite_png(tmp_path / "payloads" / f"{tag}.png")
        with zipfile.ZipFile(output_path, "w") as archive:
            archive.write(png, arcname=f"{tag}.png")
        return output_path

    monkeypatch.setattr("spritelab.harvest.pipeline.download_file", fake_download)
    work_dir = tmp_path / "work"
    first_source = make_source(
        "remote",
        source_type="direct_zip_url",
        download_url="https://example.test/one.zip",
        download_kind="zip",
    )

    first = harvest_source_to_imported_sprites(first_source, options=HarvestImportOptions(), work_dir=work_dir)
    reused = harvest_source_to_imported_sprites(first_source, options=HarvestImportOptions(), work_dir=work_dir)
    assert [sprite.candidate.relative_path for sprite in first] == ["one.png"]
    assert [sprite.candidate.relative_path for sprite in reused] == ["one.png"]
    assert calls == ["https://example.test/one.zip"]

    changed_source = make_source(
        "remote",
        source_type="direct_zip_url",
        download_url="https://example.test/two.zip",
        download_kind="zip",
    )
    changed = harvest_source_to_imported_sprites(changed_source, options=HarvestImportOptions(), work_dir=work_dir)

    assert [sprite.candidate.relative_path for sprite in changed] == ["two.png"]
    assert calls == ["https://example.test/one.zip", "https://example.test/two.zip"]


def test_local_directory_source_imports(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "one.png")
    make_sprite_png(root / "two.png")
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
    assert len([s for s in harvested if s.imported.bundle is not None]) == 2


def test_work_subdirectory_symlink_is_rejected_without_touching_outside_tree(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "one.png")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"outside")
    try:
        os.symlink(outside, work_dir / "sliced", target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable in this test session")

    source = make_source(local_root_path=str(root))
    with pytest.raises(ValueError, match="link or reparse"):
        harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=work_dir)

    assert sentinel.read_bytes() == b"outside"


def test_wrong_size_sliced_or_padded(tmp_path):
    root = tmp_path / "pngs"
    make_sheet_png(root / "sheet.png", rows=2, cols=2)  # 64x64 -> 4 tiles
    make_sprite_png(root / "small.png", size=16)  # -> center-padded
    source = make_source(local_root_path=str(root))
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
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
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
    statuses = {s.final_item.sprite_id: s.final_item.status for s in harvested}
    assert any(status == "accepted" for status in statuses.values())
    assert any(status == "rejected" for status in statuses.values())


def test_source_metadata_propagates(tmp_path):
    root = tmp_path / "pngs"
    make_sprite_png(root / "one.png")
    source = make_source(
        "kenney_pack", license_name="cc0", author="Kenney", local_root_path=str(root), source_type="kenney"
    )
    harvested = harvest_source_to_imported_sprites(source, options=HarvestImportOptions(), work_dir=tmp_path / "work")
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
