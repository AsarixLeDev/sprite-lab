from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.unlabeled_pool.builder import alpha_mask_sha256, canonical_rgba_sha256
from spritelab.unlabeled_pool.provenance_repair import (
    apply_provenance_repair,
    deterministic_json,
    file_sha256,
    load_provenance_repairs,
)


def _fixture(tmp_path: Path, *, method: str = "original_file_recovered") -> tuple[Path, dict, Path]:
    source = np.zeros((34, 34, 4), dtype=np.uint8)
    source[1:33, 1:33, :3] = (20, 80, 160)
    source[5:29, 8:26, 3] = 255
    derived = source[1:33, 1:33]
    source_path = tmp_path / "source.png"
    derived_path = tmp_path / "derived.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(derived).save(derived_path)
    archive_path = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source_path, "pack/source.png")
    source_bytes = source_path.read_bytes()
    mapping = {
        "alpha_mask_sha256": alpha_mask_sha256(derived[..., 3]),
        "archive_member": "pack/source.png",
        "crop_box": [1, 1, 33, 33],
        "derived_image_path": "derived.png",
        "derived_image_sha256": file_sha256(derived_path),
        "exported_rgba_sha256": canonical_rgba_sha256(derived),
        "source_dimensions": {"height": 34, "width": 34},
        "source_image_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "sprite_id": "pack_sprite",
    }
    status = {
        "original_file_recovered": "original_download_verified",
        "exact_source_reacquired": "reacquired_exact_source_verified",
        "upstream_changed": "reacquired_equivalent",
    }[method]
    artifact = {
        "repair_schema_version": "spritelab_provenance_repair_v1",
        "source_id": "pack",
        "source_run": "pack_run",
        "recorded_source_url": "https://example.invalid/source",
        "downloaded_filename": "pack.zip",
        "download_sha256": file_sha256(archive_path),
        "download_hash_scope": "downloaded_file_bytes",
        "download_size": archive_path.stat().st_size,
        "local_download_path": "pack.zip",
        "recovery_method": method,
        "verification_evidence": {"archive_member_mapping": [mapping]},
        "affected_sprite_ids": ["pack_sprite"],
        "old_provenance_status": "blocked_provenance",
        "new_provenance_status": status,
        "timestamp": "2026-07-11T00:00:00Z",
        "tool_config_hash": "a" * 64,
    }
    if method == "upstream_changed":
        artifact["historical_download_sha256"] = "b" * 64
    repair_path = tmp_path / "repair.json"
    repair_path.write_text(deterministic_json(artifact), encoding="utf-8")
    return repair_path, artifact, archive_path


def test_original_file_recovery(tmp_path: Path):
    repair_path, artifact, _ = _fixture(tmp_path)
    repairs, hashes = load_provenance_repairs([repair_path], workspace_root=tmp_path)
    row = apply_provenance_repair(
        {"source_run": "pack_run", "sprite_id": "pack_sprite", "downloaded_file_hash": ""}, repairs
    )
    assert row["downloaded_file_hash"] == artifact["download_sha256"]
    assert row["provenance_status"] == "original_download_verified"
    assert row["archive_member"] == "pack/source.png"
    assert row["native_dimensions"] == {"height": 34, "width": 34}
    assert row["resize_policy"] == "exact_rgba_crop_34x34_box_1_1_33_33_to_32x32"
    assert hashes[repair_path.name] == file_sha256(repair_path)


def test_reacquired_exact_source(tmp_path: Path):
    repair_path, artifact, _ = _fixture(tmp_path, method="exact_source_reacquired")
    repairs, _ = load_provenance_repairs([repair_path], workspace_root=tmp_path)
    row = apply_provenance_repair({"source_run": "pack_run", "sprite_id": "pack_sprite"}, repairs)
    assert row["downloaded_file_hash"] == artifact["download_sha256"]
    assert row["provenance_status"] == "reacquired_exact_source_verified"


def test_upstream_changed_source(tmp_path: Path):
    repair_path, artifact, _ = _fixture(tmp_path, method="upstream_changed")
    repairs, _ = load_provenance_repairs([repair_path], workspace_root=tmp_path)
    row = apply_provenance_repair({"source_run": "pack_run", "sprite_id": "pack_sprite"}, repairs)
    assert artifact["historical_download_sha256"] != artifact["download_sha256"]
    assert row["provenance_status"] == "reacquired_equivalent"


def test_mismatched_archive_rejected(tmp_path: Path):
    repair_path, artifact, _ = _fixture(tmp_path)
    artifact["verification_evidence"]["archive_member_mapping"][0]["source_image_sha256"] = "c" * 64
    repair_path.write_text(deterministic_json(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="archive member hash mismatch"):
        load_provenance_repairs([repair_path], workspace_root=tmp_path)


def test_exported_sprite_hash_not_accepted_as_download_hash(tmp_path: Path):
    repair_path, artifact, _ = _fixture(tmp_path)
    mapping = artifact["verification_evidence"]["archive_member_mapping"][0]
    artifact["download_sha256"] = mapping["exported_rgba_sha256"]
    repair_path.write_text(deterministic_json(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="exported-sprite hash"):
        load_provenance_repairs([repair_path], workspace_root=tmp_path)


def test_append_only_repair_loading(tmp_path: Path):
    repair_path, _, _ = _fixture(tmp_path)
    historical = tmp_path / "sources.jsonl"
    historical.write_text('{"sha256":""}\n', encoding="utf-8")
    before = historical.read_bytes()
    first, _ = load_provenance_repairs([repair_path], workspace_root=tmp_path)
    second, _ = load_provenance_repairs([repair_path], workspace_root=tmp_path)
    assert first == second
    assert historical.read_bytes() == before


def test_historical_manifest_unchanged(tmp_path: Path):
    repair_path, _, _ = _fixture(tmp_path)
    historical = tmp_path / "sources.jsonl"
    historical.write_text(json.dumps({"source_id": "pack", "sha256": ""}) + "\n", encoding="utf-8")
    before_hash = file_sha256(historical)
    load_provenance_repairs([repair_path], workspace_root=tmp_path)
    assert file_sha256(historical) == before_hash


def test_deterministic_repair_serialization(tmp_path: Path):
    _, artifact, _ = _fixture(tmp_path)
    reversed_artifact = dict(reversed(list(artifact.items())))
    assert deterministic_json(artifact) == deterministic_json(reversed_artifact)
    assert deterministic_json(artifact).endswith("\n")
