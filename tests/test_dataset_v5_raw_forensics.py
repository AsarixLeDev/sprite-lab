from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from spritelab.dataset_v5.raw_forensics import (
    RAW_FORENSIC_INVENTORY_SCHEMA_VERSION,
    audit_raw_source_inventory,
    write_raw_forensic_inventory,
)


def _png_bytes(color: tuple[int, int, int, int] = (30, 80, 120, 255)) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (3, 2), color).save(output, format="PNG")
    return output.getvalue()


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)


def _write_source_manifest(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw-source"
    archive = root / "harvest_runs" / "run_alpha" / "downloads" / "pack.zip"
    _zip(
        archive,
        {
            "sprites/safe.png": _png_bytes(),
            "__MACOSX/sprites/._fake.png": b"\x00\x05\x16\x07AppleDouble metadata",
            "LICENSE.txt": b"test only",
        },
    )
    row: dict[str, object] = {
        "author": "Example Creator",
        "license": {
            "license": "unknown",
            "license_url": "",
            "user_confirmed": False,
        },
        "local_archive_path": "harvest_runs/run_alpha/downloads/pack.zip",
        "original_filename": "pack.zip",
        "source_id": "source_alpha",
        "source_name": "Example Pack",
        "source_type": "manual_zip",
    }
    _write_source_manifest(root / "harvest_runs" / "run_alpha" / "sources.jsonl", row)

    orphan = root / "experiments" / "acquisition_diversity_wave_v1" / "downloads" / "orphan.zip"
    _zip(orphan, {"orphan.png": _png_bytes((200, 10, 20, 255))})

    itemicon = root / "data_sources" / "cc_by_itemiconpack32" / "itemiconpack32.png"
    itemicon.parent.mkdir(parents=True, exist_ok=True)
    itemicon.write_bytes(_png_bytes((10, 200, 20, 255)))
    return root


def test_unknown_license_and_missing_historical_hash_are_recorded_not_fatal(tmp_path: Path) -> None:
    inventory = audit_raw_source_inventory(_fixture_source_root(tmp_path))

    assert inventory.schema_version == RAW_FORENSIC_INVENTORY_SCHEMA_VERSION
    assert inventory.summary["source_binding_count"] == 1
    assert inventory.summary["resolved_source_binding_count"] == 1
    assert inventory.summary["raw_source_gate_passed"] is False
    assert inventory.summary["blocking_issue_counts"]["unknown_license"] >= 1
    assert inventory.summary["blocking_issue_counts"]["missing_historical_archive_sha256"] >= 1
    assert inventory.summary["blocking_issue_counts"]["missing_acquisition_url"] >= 1

    safe = next(row for row in inventory.records if row.get("archive_member_path") == "sprites/safe.png")
    assert safe["decoded_image_sha256"] == safe["output_decoded_rgba_sha256"]
    assert safe["width"] == 3
    assert safe["height"] == 2
    assert safe["crop_coordinates"] == [0, 0, 3, 2]
    assert safe["interpolation_policy"] == "none"
    assert safe["inclusion_decision"] == "quarantine"
    assert "unknown_license" in safe["provenance_issues"]
    binding = safe["source_bindings"][0]
    assert binding["historically_recorded_archive_sha256"] is None
    assert binding["current_observed_archive_sha256"] == safe["original_archive_sha256"]


def test_orphan_itemicon_and_appledouble_are_explicit_evidence(tmp_path: Path) -> None:
    inventory = audit_raw_source_inventory(_fixture_source_root(tmp_path))

    fake = next(row for row in inventory.records if row.get("archive_member_path") == "__MACOSX/sprites/._fake.png")
    assert fake["original_byte_sha256"]
    assert fake["decoded_image_sha256"] is None
    assert fake["inclusion_decision"] == "reject"
    assert "appledouble_resource_fork" in fake["audit_issues"]
    assert "unreadable_image_payload" in fake["audit_issues"]

    source_archive = next(row for row in inventory.artifacts if row["source_bindings"])
    assert source_archive["archive_crc_ok"] is True
    assert source_archive["unsafe_archive_member_count"] == 0

    orphan = next(
        row
        for row in inventory.artifacts
        if "experiments/acquisition_diversity_wave_v1/downloads/orphan.zip" in row["original_archive_paths"]
    )
    assert "acquisition_orphan_artifact" in orphan["provenance_issues"]
    assert inventory.summary["acquisition_orphan_artifact_count"] == 1

    itemicon = next(row for row in inventory.records if row.get("original_filename") == "itemiconpack32.png")
    assert itemicon["inclusion_decision"] == "quarantine"
    assert "incomplete_itemicon_provenance" in itemicon["provenance_issues"]


def test_artifact_hash_dedup_retains_every_physical_path(tmp_path: Path) -> None:
    root = _fixture_source_root(tmp_path)
    original = root / "harvest_runs" / "run_alpha" / "downloads" / "pack.zip"
    copy = root / "experiments" / "acquisition_diversity_wave_v1" / "downloads" / "pack-copy.zip"
    shutil.copyfile(original, copy)

    inventory = audit_raw_source_inventory(root)

    artifact = next(row for row in inventory.artifacts if row["source_bindings"])
    assert artifact["original_archive_paths"] == [
        "experiments/acquisition_diversity_wave_v1/downloads/pack-copy.zip",
        "harvest_runs/run_alpha/downloads/pack.zip",
    ]
    assert inventory.summary["physical_path_count"] == 4
    assert inventory.summary["unique_artifact_count"] == 3


def test_outputs_are_deterministic_and_refuse_overwrite(tmp_path: Path) -> None:
    root = _fixture_source_root(tmp_path)
    first = audit_raw_source_inventory(root)
    second = audit_raw_source_inventory(root)

    assert first.inventory_jsonl_bytes() == second.inventory_jsonl_bytes()
    assert first.archive_hashes_bytes() == second.archive_hashes_bytes()
    assert first.report_text() == second.report_text()
    assert b"mtime" not in first.inventory_jsonl_bytes().lower()

    paths_a = write_raw_forensic_inventory(first, tmp_path / "evidence-a")
    paths_b = write_raw_forensic_inventory(second, tmp_path / "evidence-b")
    for key in paths_a:
        assert paths_a[key].read_bytes() == paths_b[key].read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_raw_forensic_inventory(first, tmp_path / "evidence-a")
