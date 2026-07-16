from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from spritelab.product_core import ProductStatus
from spritelab.product_features.dataset.intake import DatasetImportInterrupted, DatasetIntakeService, build_dataset
from test_product_dataset_helpers import (
    make_animated_png,
    make_configured,
    make_png,
    make_sheet,
    save_same_rgba_with_metadata,
    tree_hashes,
)


def _items(output: Path) -> list[dict]:
    return [json.loads(line) for line in (output / "items.jsonl").read_text(encoding="utf-8").splitlines()]


def test_minimum_images_source_license_layout(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "images" / "one.png")
    result = build_dataset(root, output_root=tmp_path / "result")
    assert result.status == ProductStatus.COMPLETE
    assert result.data["counts"]["accepted"] == 1
    assert (tmp_path / "result" / "raw_extraction" / "extraction_manifest.jsonl").is_file()


def test_png_directly_under_root(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "root.png")
    assert build_dataset(root, output_root=tmp_path / "out").data["counts"]["processed"] == 1


def test_nested_source_packs_use_nearest_evidence(tmp_path: Path) -> None:
    root = tmp_path / "packs"
    for name, color in (("pack_a", (220, 40, 30, 255)), ("pack_b", (40, 150, 230, 255))):
        pack = make_configured(root / name)
        make_png(pack / "images" / "same.png", color=color)
    result = build_dataset(root, output_root=tmp_path / "out")
    items = _items(tmp_path / "out")
    assert result.data["counts"]["accepted"] == 2
    assert {item["source"]["path"] for item in items} == {"pack_a/source.txt", "pack_b/source.txt"}


def test_source_txt_url_is_preserved(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    (root / "source.txt").write_text(
        "Name: A Pack\nCreator: Test Artist\nhttps://example.test/a-pack\n", encoding="utf-8"
    )
    make_png(root / "one.png")
    build_dataset(root, output_root=tmp_path / "out")
    assert _items(tmp_path / "out")[0]["source"]["source_url"] == "https://example.test/a-pack"


def test_source_yaml_is_supported(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "source.yaml").write_text(
        yaml.safe_dump({"name": "YAML Pack", "creator": "Artist", "url": "https://example.test/yaml"}),
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    make_png(root / "one.png")
    build_dataset(root, output_root=tmp_path / "out")
    assert _items(tmp_path / "out")[0]["source"]["source_name"] == "YAML Pack"


@pytest.mark.parametrize(
    ("license_text", "expected"),
    [
        ("Permission is hereby granted, free of charge. MIT License", "mit"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("This work is explicitly dedicated to the public domain.", "public_domain"),
    ],
)
def test_license_text_url_and_public_domain(tmp_path: Path, license_text: str, expected: str) -> None:
    root = make_configured(tmp_path / expected, license_text=license_text)
    make_png(root / "one.png")
    build_dataset(root, output_root=tmp_path / f"out-{expected}")
    assert _items(tmp_path / f"out-{expected}")[0]["license"]["license"] == expected


def test_license_yaml_is_supported(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "source.txt").write_text("Name: Commissioned Pack\nCreator: Test Artist\n", encoding="utf-8")
    (root / "license.yaml").write_text("spdx: Apache-2.0\n", encoding="utf-8")
    make_png(root / "one.png")
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["accepted"] == 1


def test_missing_source_is_quarantined_without_traceback(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "LICENSE").write_text("CC0\n", encoding="utf-8")
    make_png(root / "one.png")
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.status == ProductStatus.NEEDS_REVIEW
    assert result.data["counts"]["quarantined"] == 1
    assert "1 images have no source information" in result.message


def test_missing_license_is_quarantined_without_traceback(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "source.txt").write_text("Name: Own Archive\nCreator: Test Artist\n", encoding="utf-8")
    make_png(root / "one.png")
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["quarantined"] == 1
    assert "1 images have no license information" in result.message


def test_duplicate_bytes_excludes_later_path(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    first = make_png(root / "a.png")
    (root / "b.png").write_bytes(first.read_bytes())
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["duplicates"] == 1
    assert [item["relative_path"] for item in _items(tmp_path / "out") if "duplicate" in item["reasons"]] == ["b.png"]


def test_duplicate_decoded_rgba_excludes_different_png_bytes(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    first = make_png(root / "a.png")
    save_same_rgba_with_metadata(first, root / "b.png")
    build_dataset(root, output_root=tmp_path / "out")
    duplicate = next(item for item in _items(tmp_path / "out") if item.get("duplicate_kind"))
    assert duplicate["duplicate_kind"] == "duplicate_decoded_rgba"


def test_blank_and_corrupt_pngs_have_controlled_reasons(tmp_path: Path) -> None:
    from PIL import Image

    root = make_configured(tmp_path / "input")
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(root / "blank.png")
    (root / "corrupt.png").write_bytes(b"not a png")
    build_dataset(root, output_root=tmp_path / "out")
    reasons = {item["relative_path"]: item["reasons"] for item in _items(tmp_path / "out")}
    assert reasons == {"blank.png": ["blank"], "corrupt.png": ["unreadable"]}


def test_truncated_png_is_unreadable_not_a_traceback(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    complete = make_png(root / "complete.png")
    payload = complete.read_bytes()
    (root / "truncated.png").write_bytes(payload[: max(20, len(payload) // 3)])
    complete.unlink()
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["rejected"] == 1
    assert _items(tmp_path / "out")[0]["reasons"] == ["unreadable"]


def test_appledouble_is_policy_excluded(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    (root / "._sprite.png").write_bytes(b"AppleDouble metadata")
    build_dataset(root, output_root=tmp_path / "out")
    assert _items(tmp_path / "out")[0]["reasons"] == ["policy_excluded"]


def test_animated_png_requires_special_extraction(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_animated_png(root / "animated.png")
    build_dataset(root, output_root=tmp_path / "out")
    item = _items(tmp_path / "out")[0]
    assert item["current_disposition"] == "requires_special_extraction"
    assert item["reasons"] == ["unsupported_animation"]


def test_unambiguous_sheet_is_extracted_automatically(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_sheet(root / "sheet.png")
    result = build_dataset(root, output_root=tmp_path / "out")
    items = _items(tmp_path / "out")
    assert items[0]["current_disposition"] == "sheet_split"
    assert all(item["current_disposition"] == "accepted" for item in items[1:])
    assert result.data["counts"]["processed"] == 1
    assert result.data["counts"]["extracted_from_sheets"] == 4
    assert result.data["counts"]["needs_sheet_review"] == 0
    extraction = items[1]["sheet_extraction"]
    assert extraction["crop_rectangle"] == [0, 0, 16, 16]
    assert extraction["output_decoded_rgba_sha256"] == items[1]["decoded_rgba_sha256"]
    assert extraction["source_sheet_modified"] is False


def test_unicode_paths_spaces_and_filename_collisions(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "dataset with spaces ü")
    make_png(root / "créatures" / "same.png", color=(250, 60, 60, 255))
    make_png(root / "objets" / "same.png", color=(60, 100, 250, 255))
    result = build_dataset(root, output_root=tmp_path / "output with spaces")
    assert result.data["counts"]["processed"] == 2
    assert len({item["item_id"] for item in _items(tmp_path / "output with spaces")}) == 2


def test_input_folder_remains_byte_identical(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "images" / "one.png")
    before = tree_hashes(root)
    result = build_dataset(root, output_root=tmp_path / "out")
    assert tree_hashes(root) == before
    assert result.data["input_mutated"] is False


def test_include_exclude_and_originals_are_supported(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "originals" / "keep.png")
    make_png(root / "images" / "drop.png", color=(30, 200, 160, 255))
    (root / "include.txt").write_text("originals/*.png\n", encoding="utf-8")
    (root / "exclude.txt").write_text("**/drop.png\n", encoding="utf-8")
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["processed"] == 2
    assert result.data["counts"]["accepted"] == 1


def test_unusual_dimensions_receive_a_controlled_disposition(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "wide.png", size=(600, 8))
    build_dataset(root, output_root=tmp_path / "out")
    item = _items(tmp_path / "out")[0]
    assert item["current_disposition"] in {"rejected", "uncertain", "requires_special_extraction"}
    assert "unusual_dimensions" in item["reasons"]


def test_large_directory_discovery_is_complete(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    for index in range(80):
        make_png(
            root / f"pack-{index // 20}" / f"sprite-{index:03d}.png",
            color=((index * 17) % 240 + 10, (index * 31) % 240 + 10, (index * 47) % 240 + 10, 255),
        )
    result = build_dataset(root, output_root=tmp_path / "out")
    assert result.data["counts"]["processed"] == 80


def test_labels_jsonl_groups_and_readme_are_optional_inputs(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "one.png")
    (root / "README").write_text("Human notes about this pack.\n", encoding="utf-8")
    (root / "labels.jsonl").write_text(
        json.dumps({"filename": "one.png", "category": "object", "canonical_object": "key"}) + "\n",
        encoding="utf-8",
    )
    (root / "groups.csv").write_text("filename,group\none.png,keys\n", encoding="utf-8")
    result = build_dataset(root, output_root=tmp_path / "out")
    item = _items(tmp_path / "out")[0]
    assert result.data["counts"]["semantically_labeled"] == 1
    assert item["groups"]["group"] == "keys"


def test_interrupted_import_resumes_and_reuses_completed_stage(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "a.png")
    make_png(root / "b.png", color=(20, 180, 220, 255))
    output = tmp_path / "out"
    with pytest.raises(DatasetImportInterrupted):
        DatasetIntakeService().build(root, output_root=output, interrupt_after=1)
    result = DatasetIntakeService().build(root, output_root=output)
    assert result.data["counts"]["processed"] == 2
    assert result.data["resumability"]["reused"] == 1


def test_changed_and_deleted_files_do_not_remain_eligible(tmp_path: Path) -> None:
    root = make_configured(tmp_path / "input")
    make_png(root / "a.png")
    make_png(root / "b.png", color=(20, 180, 220, 255))
    output = tmp_path / "out"
    build_dataset(root, output_root=output)
    (root / "a.png").unlink()
    make_png(root / "b.png", color=(210, 100, 40, 255))
    result = build_dataset(root, output_root=output)
    assert result.data["counts"]["processed"] == 1
    assert result.data["resumability"]["deleted"] == ["a.png"]
    assert {item["relative_path"] for item in _items(output)} == {"b.png"}
