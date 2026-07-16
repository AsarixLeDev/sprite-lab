from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.dataset_v5.identity import RecordBinding, canonical_rgba_bytes, decoded_rgba_sha256, make_record_id
from spritelab.dataset_v5.raw_extraction import (
    ExtractionTransform,
    RawExtractionError,
    RawExtractionSpec,
    TransparentPadding,
    UnsafeArchiveError,
    build_raw_extraction,
    build_twice_and_verify,
    list_source_image_members,
)
from spritelab.dataset_v5.raw_inventory import (
    RawInventoryError,
    RawSourceHashMismatchError,
    discover_raw_sources,
    write_raw_source_inventory,
)


def _png_bytes(rgba: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _source_row(filename: str, digest: str, *, source_type: str = "direct_zip_url") -> dict[str, object]:
    return {
        "author": "Fixture Creator",
        "download_sha256": digest,
        "download_size_bytes": 0,
        "download_url": f"https://assets.example.test/files/{filename}",
        "license": {"license": "cc0", "user_confirmed": True},
        "original_filename": filename,
        "source_id": "fixture_source",
        "source_name": "Fixture Pack",
        "source_type": source_type,
        "source_url": "https://assets.example.test/source-page",
    }


def _write_source_manifest(workspace: Path, row: dict[str, object], payload: bytes) -> Path:
    run = workspace / "harvest_runs" / "acquisition_run_01"
    downloads = run / "downloads"
    downloads.mkdir(parents=True)
    archive = downloads / str(row["original_filename"])
    archive.write_bytes(payload)
    (run / "sources.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return archive


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
    return buffer.getvalue()


def _rgba_fixture() -> np.ndarray:
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[0, 1] = [255, 0, 0, 255]
    rgba[0, 2] = [0, 255, 0, 255]
    rgba[1, 1] = [0, 0, 255, 255]
    rgba[1, 2] = [255, 255, 0, 128]
    return rgba


def test_raw_inventory_resolves_url_basename_and_writes_only_fresh_output(tmp_path: Path) -> None:
    rgba = _rgba_fixture()
    archive_bytes = _zip_bytes([("sprites/item.png", _png_bytes(rgba))])
    digest = hashlib.sha256(archive_bytes).hexdigest()
    row = _source_row("original_download.zip", digest)
    archive = _write_source_manifest(tmp_path, row, archive_bytes)

    records = discover_raw_sources(tmp_path)

    assert len(records) == 1
    source = records[0]
    assert source.resolved_archive_path == archive.resolve()
    assert source.archive_sha256 == digest
    assert source.expected_archive_sha256 == digest
    assert source.provenance_status == "expected_archive_hash_verified"
    assert source.resolution_method == "run_downloads_basename"
    assert source.distribution_platform == "assets.example.test"

    output = tmp_path / "inventory"
    write_raw_source_inventory(records, output)
    inventory_rows = [json.loads(line) for line in (output / "raw_source_inventory.jsonl").read_text().splitlines()]
    assert inventory_rows[0]["archive_sha256"] == digest
    archive_hashes = json.loads((output / "source_archive_hashes.json").read_text())
    assert list(archive_hashes["archives"].values()) == [digest]
    with pytest.raises(FileExistsError):
        write_raw_source_inventory(records, output)


def test_raw_inventory_fails_closed_on_hash_license_and_provenance(tmp_path: Path) -> None:
    payload = _zip_bytes([("sprite.png", _png_bytes(_rgba_fixture()))])
    wrong_hash = "0" * 64
    row = _source_row("source.zip", wrong_hash)
    _write_source_manifest(tmp_path, row, payload)
    with pytest.raises(RawSourceHashMismatchError):
        discover_raw_sources(tmp_path)

    actual_hash = hashlib.sha256(payload).hexdigest()
    row["download_sha256"] = actual_hash
    row["license"] = {"license": "unknown", "user_confirmed": True}
    manifest = tmp_path / "harvest_runs" / "acquisition_run_01" / "sources.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RawInventoryError, match="license"):
        discover_raw_sources(tmp_path)

    row["license"] = {"license": "cc0", "user_confirmed": True}
    row["author"] = ""
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RawInventoryError, match="creator/publisher"):
        discover_raw_sources(tmp_path)


def test_raw_inventory_resolves_local_path_unique_run_file_and_acquisition_root(tmp_path: Path) -> None:
    payload = _zip_bytes([("sprite.png", _png_bytes(_rgba_fixture()))])
    digest = hashlib.sha256(payload).hexdigest()

    local_workspace = tmp_path / "local_workspace"
    local_run = local_workspace / "harvest_runs" / "run"
    local_run.mkdir(parents=True)
    originals = local_workspace / "originals"
    originals.mkdir()
    (originals / "source.zip").write_bytes(payload)
    local_row = _source_row("source.zip", digest)
    local_row["local_archive_path"] = "originals/source.zip"
    (local_run / "sources.jsonl").write_text(json.dumps(local_row) + "\n", encoding="utf-8")
    assert discover_raw_sources(local_workspace)[0].resolution_method == "local_archive_path"

    unique_workspace = tmp_path / "unique_workspace"
    unique_row = _source_row("unused-name.zip", digest)
    unique_row.pop("original_filename")
    unique_row.pop("download_url")
    unique_run = unique_workspace / "harvest_runs" / "run"
    (unique_run / "downloads").mkdir(parents=True)
    (unique_run / "downloads" / "opaque-cache-entry.bin").write_bytes(payload)
    (unique_run / "sources.jsonl").write_text(json.dumps(unique_row) + "\n", encoding="utf-8")
    assert discover_raw_sources(unique_workspace)[0].resolution_method == "run_downloads_unique_file"

    acquisition_workspace = tmp_path / "acquisition_workspace"
    acquisition_run = acquisition_workspace / "harvest_runs" / "run"
    acquisition_run.mkdir(parents=True)
    acquisition_row = _source_row("source.zip", digest)
    (acquisition_run / "sources.jsonl").write_text(json.dumps(acquisition_row) + "\n", encoding="utf-8")
    acquisition_root = tmp_path / "acquisition_downloads"
    acquisition_root.mkdir()
    (acquisition_root / "source.zip").write_bytes(payload)
    source = discover_raw_sources(
        acquisition_workspace,
        acquisition_download_roots=(acquisition_root,),
    )[0]
    assert source.resolution_method == "acquisition_downloads_basename"


def test_zip_extraction_is_content_addressed_and_two_builds_are_identical(tmp_path: Path) -> None:
    rgba = _rgba_fixture()
    archive_bytes = _zip_bytes([("sprites/item.png", _png_bytes(rgba))])
    row = _source_row("source.zip", hashlib.sha256(archive_bytes).hexdigest())
    _write_source_manifest(tmp_path, row, archive_bytes)
    source = discover_raw_sources(tmp_path)[0]
    padding = TransparentPadding(left=1, top=2, right=0, bottom=1)
    transform = ExtractionTransform(crop_coordinates=(1, 0, 3, 2), padding=padding)
    spec = RawExtractionSpec(source=source, archive_member_path="sprites/item.png", transform=transform)

    report = build_twice_and_verify((spec,), tmp_path / "build_a", tmp_path / "build_b")

    assert report["byte_identical"] is True
    row_out = json.loads((tmp_path / "build_a" / "extraction_manifest.jsonl").read_text())
    assert row_out["source_width"] == 4
    assert row_out["source_height"] == 3
    assert row_out["output_width"] == 3
    assert row_out["output_height"] == 5
    assert row_out["interpolation_policy"] == "none"
    assert row_out["crop_coordinates"] == [1, 0, 3, 2]

    cropped = rgba[0:2, 1:3]
    expected = np.zeros((5, 3, 4), dtype=np.uint8)
    expected[2:4, 1:3] = cropped
    blob_id = decoded_rgba_sha256(expected)
    assert row_out["blob_id"] == blob_id
    assert (tmp_path / "build_a" / row_out["blob_path"]).read_bytes() == canonical_rgba_bytes(expected)
    binding = RecordBinding(
        source_archive_sha256=source.archive_sha256,
        archive_member_path="sprites/item.png",
        extraction_operation="sprite_lab_raw_extraction_v1:decode_rgba_crop_then_pad",
        crop_coordinates=(1, 0, 3, 2),
        decoded_rgba_sha256=blob_id,
        padding_operation=padding.canonical(),
    )
    assert row_out["record_id"] == make_record_id(binding)
    with pytest.raises(FileExistsError):
        build_raw_extraction((spec,), tmp_path / "build_a")


@pytest.mark.parametrize(
    "entries",
    [
        [("same.png", b"one"), ("same.png", b"two")],
        [("../escape.png", b"unsafe")],
        [("Case.png", b"one"), ("case.png", b"two")],
    ],
)
def test_zip_duplicate_unsafe_and_case_colliding_members_fail_closed(
    tmp_path: Path, entries: list[tuple[str, bytes]]
) -> None:
    archive_bytes = _zip_bytes(entries)
    row = _source_row("source.zip", hashlib.sha256(archive_bytes).hexdigest())
    _write_source_manifest(tmp_path, row, archive_bytes)
    source = discover_raw_sources(tmp_path)[0]

    with pytest.raises(UnsafeArchiveError):
        list_source_image_members(source)


def test_missing_member_invalid_crop_and_source_tampering_leave_no_output(tmp_path: Path) -> None:
    rgba = _rgba_fixture()
    archive_bytes = _zip_bytes([("sprite.png", _png_bytes(rgba))])
    row = _source_row("source.zip", hashlib.sha256(archive_bytes).hexdigest())
    archive_path = _write_source_manifest(tmp_path, row, archive_bytes)
    source = discover_raw_sources(tmp_path)[0]

    missing = RawExtractionSpec(
        source=source,
        archive_member_path="missing.png",
        transform=ExtractionTransform.whole_image(),
    )
    missing_output = tmp_path / "missing_output"
    with pytest.raises(RawExtractionError, match="member missing"):
        build_raw_extraction((missing,), missing_output)
    assert not missing_output.exists()

    invalid_crop = RawExtractionSpec(
        source=source,
        archive_member_path="sprite.png",
        transform=ExtractionTransform(crop_coordinates=(0, 0, 99, 99), padding=None),
    )
    crop_output = tmp_path / "crop_output"
    with pytest.raises(RawExtractionError, match="outside decoded image bounds"):
        build_raw_extraction((invalid_crop,), crop_output)
    assert not crop_output.exists()

    archive_path.write_bytes(archive_bytes + b"tampered")
    tamper_output = tmp_path / "tamper_output"
    with pytest.raises(RawExtractionError, match="changed after inventory"):
        build_raw_extraction(
            (
                RawExtractionSpec(
                    source=source,
                    archive_member_path="sprite.png",
                    transform=ExtractionTransform.whole_image(),
                ),
            ),
            tamper_output,
        )
    assert not tamper_output.exists()


def test_direct_image_extraction_requires_explicit_whole_image_operation(tmp_path: Path) -> None:
    payload = _png_bytes(_rgba_fixture())
    row = _source_row("direct.png", hashlib.sha256(payload).hexdigest(), source_type="direct_file_url")
    _write_source_manifest(tmp_path, row, payload)
    source = discover_raw_sources(tmp_path)[0]
    assert list_source_image_members(source) == ("direct.png",)

    spec = RawExtractionSpec(
        source=source,
        archive_member_path=None,
        transform=ExtractionTransform.whole_image(),
    )
    build_raw_extraction((spec,), tmp_path / "direct_build")
    extracted = json.loads((tmp_path / "direct_build" / "extraction_manifest.jsonl").read_text())
    assert extracted["archive_member_path"] == "direct.png"
    assert extracted["transformation"]["operation"] == "decode_rgba_whole_image"

    with pytest.raises(ValueError, match="forbidden"):
        ExtractionTransform(crop_coordinates=None, padding=None, interpolation_policy="nearest")
    with pytest.raises(ValueError, match="transparent black"):
        TransparentPadding(left=1, top=0, right=0, bottom=0, fill_rgba=(255, 255, 255, 0))
