from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from _harvest_testdata import make_source
from spritelab.dataset_v5.identity import decoded_rgba_sha256
from spritelab.dataset_v5.raw_forensics import (
    RAW_EXTRACTION_OPERATION_SCHEMA_VERSION,
    CandidateSheetCoordinate,
    ExtractionOperationError,
    RawExtractionOperation,
    classify_image_payload,
    execute_extraction_operation,
    extraction_operation_json_schema,
    make_extraction_relation_id,
    operation_manifest_bytes,
    verify_extraction_operation_manifest,
)
from spritelab.harvest.archive import extract_archive, is_appledouble_record, iter_archive_pngs
from spritelab.harvest.extract import discover_png_candidates


def _png_bytes(pixels: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGBA").save(output, format="PNG")
    return output.getvalue()


def _gif_bytes() -> bytes:
    output = io.BytesIO()
    frames = [Image.new("RGBA", (3, 2), color) for color in ((255, 0, 0, 255), (0, 0, 255, 255))]
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=[20, 40], loop=0)
    return output.getvalue()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _operation(
    *,
    operation: str,
    decoded_hash: str | None,
    frame_index: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
    sheet: tuple[int, int, int, int] | None = None,
    padding: tuple[int, int, int, int] | None = None,
    terminal_reason: str | None = None,
    candidates: tuple[CandidateSheetCoordinate, ...] = (),
    interpolation: str = "none",
    member: str = "sprites/item.png",
) -> RawExtractionOperation:
    row, column, cell_width, cell_height = sheet or (None, None, None, None)
    return RawExtractionOperation(
        operation_version=RAW_EXTRACTION_OPERATION_SCHEMA_VERSION,
        operation=operation,
        source_archive_sha256="a" * 64,
        archive_member_path=member,
        source_member_sha256="b" * 64,
        frame_index=frame_index,
        crop_rectangle=crop,
        sheet_row=row,
        sheet_column=column,
        cell_width=cell_width,
        cell_height=cell_height,
        padding_dimensions=padding,
        interpolation_policy=interpolation,
        decoded_rgba_sha256=decoded_hash,
        terminal_reason=terminal_reason,
        candidate_coordinates=candidates,
    )


def test_appledouble_rejected_by_path_and_structure_without_candidate_decode(tmp_path: Path) -> None:
    metadata = b"\x00\x05\x16\x07" + b"metadata"
    archive = tmp_path / "pack.zip"
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[:, :] = (20, 40, 60, 255)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("sprites/ok.png", _png_bytes(pixels))
        bundle.writestr("__MACOSX/sprites/._ok.png", metadata)
    assert is_appledouble_record("ordinary.png", metadata)
    assert iter_archive_pngs(archive) == ["sprites/ok.png"]

    extracted = extract_archive(archive, tmp_path / "extracted")
    assert not (extracted / "__MACOSX" / "sprites" / "._ok.png").exists()

    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "._resource.png").write_bytes(metadata)
    candidates = discover_png_candidates(direct, make_source(), include_hidden=True)
    assert candidates[0].artifact_kind == "metadata_resource_fork"
    assert candidates[0].extraction_disposition == "reject_resource_fork"
    assert candidates[0].status == "rejected"


def test_multiframe_requires_explicit_frame_and_explicit_frame_is_reproducible() -> None:
    payload = _gif_bytes()
    disposition = classify_image_payload(
        "animation.gif", payload, expected_size=len(payload), member_bytes_complete=True
    )
    assert disposition.payload_classification == "multi_frame_image"
    assert disposition.terminal_operation == "exclude_ambiguous"
    assert disposition.frame_count == 2

    with Image.open(io.BytesIO(payload)) as image:
        image.seek(1)
        expected = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    frame = _operation(
        operation="frame_select",
        decoded_hash=decoded_rgba_sha256(expected),
        frame_index=1,
        member="animation.gif",
    )
    assert np.array_equal(execute_extraction_operation(payload, frame), expected)
    assert np.array_equal(execute_extraction_operation(payload, frame), expected)

    direct = _operation(operation="direct_decode", decoded_hash=decoded_rgba_sha256(expected), member="animation.gif")
    with pytest.raises(ExtractionOperationError, match="explicit frame_select"):
        execute_extraction_operation(payload, direct)


def test_crop_sheet_coordinates_and_interpolation_are_explicitly_required() -> None:
    with pytest.raises(ExtractionOperationError, match="crop requires"):
        _operation(operation="crop", decoded_hash="c" * 64)
    with pytest.raises(ExtractionOperationError, match="sheet_cell requires"):
        _operation(operation="sheet_cell", decoded_hash="c" * 64, crop=(0, 0, 2, 2))
    with pytest.raises(ExtractionOperationError, match="interpolation"):
        _operation(operation="direct_decode", decoded_hash="c" * 64, interpolation="nearest")


def test_duplicate_pixels_at_different_coordinates_have_distinct_operation_identity() -> None:
    decoded_hash = "c" * 64
    first = _operation(operation="sheet_cell", decoded_hash=decoded_hash, crop=(0, 0, 2, 2), sheet=(0, 0, 2, 2))
    second = _operation(operation="sheet_cell", decoded_hash=decoded_hash, crop=(2, 0, 4, 2), sheet=(0, 1, 2, 2))
    assert first.operation_id != second.operation_id
    assert make_extraction_relation_id("exact_rgba", [first.operation_id, second.operation_id]).startswith("xrel_")


def test_center_padding_is_exact_and_never_interpolates() -> None:
    source = np.zeros((2, 3, 4), dtype=np.uint8)
    source[:, :] = (10, 20, 30, 255)
    expected = np.zeros((5, 6, 4), dtype=np.uint8)
    expected[1:3, 1:4] = source
    operation = _operation(
        operation="center_pad",
        decoded_hash=decoded_rgba_sha256(expected),
        padding=(1, 1, 2, 2),
    )
    actual = execute_extraction_operation(_png_bytes(source), operation)
    assert np.array_equal(actual, expected)
    assert set(map(tuple, actual.reshape(-1, 4))) == {(0, 0, 0, 0), (10, 20, 30, 255)}

    with pytest.raises(ExtractionOperationError, match="center padding"):
        _operation(operation="center_pad", decoded_hash="c" * 64, padding=(2, 1, 0, 1))


def test_two_rebuilds_and_manifests_are_byte_identical() -> None:
    pixels = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    pixels[:, :, 3] = 255
    expected = pixels[1:3, 1:4].copy()
    operation = _operation(operation="crop", decoded_hash=decoded_rgba_sha256(expected), crop=(1, 1, 4, 3))
    payload = _png_bytes(pixels)
    first_pixels = execute_extraction_operation(payload, operation).tobytes()
    second_pixels = execute_extraction_operation(payload, operation).tobytes()
    first_manifest = operation_manifest_bytes([operation])
    second_manifest = operation_manifest_bytes([operation])
    assert first_pixels == second_pixels
    assert first_manifest == second_manifest
    assert b"timestamp" not in first_manifest
    assert str(Path.cwd()).encode() not in first_manifest


def test_unreadable_payload_terminal_classification_is_specific() -> None:
    metadata = classify_image_payload(
        "._sprite.png", b"\x00\x05\x16\x07data", expected_size=8, member_bytes_complete=True
    )
    corrupt = classify_image_payload("sprite.png", b"not png", expected_size=7, member_bytes_complete=True)
    truncated = classify_image_payload("sprite.png", b"short", expected_size=99, member_bytes_complete=True)
    nonimage = classify_image_payload("README.txt", b"hello", expected_size=5, member_bytes_complete=True)
    cmyk = io.BytesIO()
    Image.new("CMYK", (2, 2)).save(cmyk, format="TIFF")
    unsupported = classify_image_payload(
        "sprite.tif", cmyk.getvalue(), expected_size=len(cmyk.getvalue()), member_bytes_complete=True
    )
    assert metadata.payload_classification == "appledouble_resource_fork"
    assert corrupt.payload_classification == "corrupt_image"
    assert truncated.payload_classification == "truncated_archive_member"
    assert nonimage.payload_classification == "non_image_payload"
    assert unsupported.payload_classification == "unsupported_image_mode"


def test_operation_schema_and_tamper_detection() -> None:
    operation = _operation(operation="direct_decode", decoded_hash="c" * 64)
    row = operation.to_dict()
    assert extraction_operation_json_schema()["$id"] == RAW_EXTRACTION_OPERATION_SCHEMA_VERSION
    assert verify_extraction_operation_manifest([row]) == (operation,)

    tampered = json.loads(json.dumps(row))
    tampered["archive_member_path"] = "sprites/changed.png"
    with pytest.raises(ExtractionOperationError, match="identity mismatch"):
        verify_extraction_operation_manifest([tampered])


def test_exclude_ambiguous_coordinate_preserves_every_candidate() -> None:
    operation = _operation(
        operation="exclude_ambiguous_coordinate",
        decoded_hash=None,
        terminal_reason="multiple_coordinates_without_authoritative_evidence",
        candidates=(CandidateSheetCoordinate(0, 0), CandidateSheetCoordinate(0, 1)),
    )
    assert operation.to_dict()["candidate_coordinates"] == [
        {"column": 0, "row": 0},
        {"column": 1, "row": 0},
    ]
    assert _hash(operation_manifest_bytes([operation]))
