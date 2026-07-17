from __future__ import annotations

import zipfile

import pytest

from _harvest_testdata import make_sprite_png, make_zip_of_pngs
from spritelab.harvest.archive import ArchiveSecurityError, archive_member_summary, extract_archive
from spritelab.harvest.sheet_mappings import metadata_for_sheet_cell
from spritelab.harvest.sources import SourceRecord, normalize_license_name, source_record_from_dict


def test_zip_member_filters_are_case_sensitive_and_reported(tmp_path):
    pack = make_zip_of_pngs(tmp_path / "pack.zip", ["iron.png", "steel.png", "all-assets-preview.png"])
    summary = archive_member_summary(
        pack, include_member_globs=["*.png"], exclude_member_globs=["all-assets-preview.png"]
    )
    assert summary["selected_image_members"] == ["iron.png", "steel.png"]
    root = extract_archive(
        pack, tmp_path / "out", include_member_globs=["*.png"], exclude_member_globs=["all-assets-preview.png"]
    )
    assert sorted(path.name for path in root.iterdir()) == ["iron.png", "steel.png"]


def test_unsafe_members_fail_before_filter_selection(tmp_path):
    pack = tmp_path / "pack.zip"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("../unsafe.png", b"x")
        archive.writestr("readme.txt", "x")
    with pytest.raises(ArchiveSecurityError, match="unsafe archive member"):
        archive_member_summary(pack, include_member_globs=["*.png"])
    assert not (tmp_path / "unsafe.png").exists()


def test_cc_by_spellings_and_legacy_provenance_are_compatible():
    for value in ("CC0", "cc-by-3.0", "CC-BY-3.0", "CC BY 3.0", "cc-by-4.0", "CC BY 4.0"):
        assert normalize_license_name(value) in {"cc0", "cc_by"}
    restored = source_record_from_dict(
        {"source_id": "legacy", "source_name": "legacy", "source_type": "manual_zip", "license": "cc0"}
    )
    assert restored.download_sha256 == ""
    assert (
        SourceRecord(source_id="new", source_name="new", source_type="direct_file_url", download_sha256="abc").sha256
        == "abc"
    )


def test_declarative_farming_mapping_excludes_blanks_and_groups_variants(tmp_path):
    tile = tmp_path / "farm tool icons calciumtrice__r001_c002.png"
    make_sprite_png(tile)
    mapped = metadata_for_sheet_cell("farming_tools_calciumtrice", "farm tool icons calciumtrice.png", tile)
    assert mapped["object_name"] == "shovel"
    assert mapped["variant_group_id"] == "farming:shovel"
    blank = metadata_for_sheet_cell(
        "farming_tools_calciumtrice", "farm tool icons calciumtrice.png", tmp_path / "x__r001_c000.png"
    )
    assert blank["mapping_excluded"] == "true"


def test_declarative_shade_mapping_groups_recolors(tmp_path):
    iron = metadata_for_sheet_cell(
        "shade_16x16_weapons", "dir/iron-weapons.png", tmp_path / "iron-weapons__r003_c007.png"
    )
    gold = metadata_for_sheet_cell(
        "shade_16x16_weapons", "dir/gold-weapons.png", tmp_path / "gold-weapons__r003_c007.png"
    )
    assert iron["material"] == "iron"
    assert gold["material"] == "gold"
    assert iron["variant_group_id"] == gold["variant_group_id"]
