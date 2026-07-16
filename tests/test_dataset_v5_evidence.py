from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.dataset_v5.evidence import VIEW_NAMES, EvidenceCompileError, _decode_rgba, compile_blocked_evidence
from spritelab.dataset_v5.identity import canonical_json_bytes, decoded_rgba_sha256


def _png_bytes(rgba: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(stream, format="PNG", compress_level=9, optimize=False)
    return stream.getvalue()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(canonical_json_bytes(row).decode("utf-8") + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, name: str) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "originals"
    downloads = source / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    pixels = np.array(
        [
            [[0, 0, 0, 0], [180, 80, 20, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[180, 80, 20, 255], [240, 170, 70, 255], [180, 80, 20, 255], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [180, 80, 20, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    payload = _png_bytes(pixels)
    archive = downloads / "misleading_pack.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("sprites/helmet.png", payload)
    direct = downloads / "mineral.png"
    direct.write_bytes(payload)
    archive_hash = _sha256(archive.read_bytes())
    direct_hash = _sha256(payload)
    decoded_hash = decoded_rgba_sha256(pixels)

    source_binding = {
        "creator_or_publisher": "Fixture Publisher",
        "license": {"license": "cc0", "user_confirmed": True},
        "pack": "Misleading Fixture Pack",
        "source_url": "https://invalid.example/source",
    }
    common = {
        "acquisition_runs": ["fixture"],
        "audit_issues": [],
        "creator_or_publishers": ["Fixture Publisher"],
        "crop_coordinates": [0, 0, 4, 4],
        "decoded_image_sha256": decoded_hash,
        "extraction_operation": "identity_decode_to_rgba",
        "height": 4,
        "image_format": "PNG",
        "image_mode": "RGBA",
        "interpolation_policy": "none",
        "licenses": [{"license": "cc0", "user_confirmed": True}],
        "original_byte_sha256": direct_hash,
        "original_byte_size": len(payload),
        "output_decoded_rgba_sha256": decoded_hash,
        "packs": ["Misleading Fixture Pack"],
        "padding_operation": "none",
        "provenance_issues": [],
        "provenance_status": "verified_fixture",
        "schema_version": "sprite_lab_raw_forensic_inventory_v1",
        "source_bindings": [source_binding],
        "source_urls": ["https://invalid.example/source"],
        "width": 4,
    }
    rows = [
        common
        | {
            "archive_member_path": "sprites/helmet.png",
            "forensic_record_id": "fr_fixture_archive",
            "inclusion_decision": "accept",
            "original_archive_path": "downloads/misleading_pack.zip",
            "original_archive_sha256": archive_hash,
            "original_filename": "helmet.png",
            "record_type": "archive_member_image",
        },
        common
        | {
            "archive_member_path": None,
            "forensic_record_id": "fr_fixture_direct",
            "inclusion_decision": "quarantine",
            "original_archive_path": "downloads/mineral.png",
            "original_archive_sha256": direct_hash,
            "original_filename": "mineral.png",
            "record_type": "standalone_image",
        },
    ]
    experiment = tmp_path / name
    experiment.mkdir()
    _write_jsonl(experiment / "raw_source_inventory.jsonl", rows)
    _write_json(
        experiment / "source_archive_hashes.json",
        {
            "archives": {
                "downloads/mineral.png": direct_hash,
                "downloads/misleading_pack.zip": archive_hash,
            },
            "schema_version": "fixture_archive_hashes_v1",
        },
    )
    historical = {"candidate_rows_byte_hash_matched": 2, "status": "read_only_fixture_observation"}
    return experiment, source, historical


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix())
        if path.name not in {"raw_source_inventory.jsonl", "source_archive_hashes.json"}
    }


def test_unreadable_decode_error_is_stable_and_address_free() -> None:
    errors = []
    for _ in range(2):
        with pytest.raises(EvidenceCompileError) as caught:
            _decode_rgba(b"not an image")
        errors.append(str(caught.value))

    assert errors == ["unidentified_image", "unidentified_image"]
    assert "0x" not in errors[0]


def test_blocked_evidence_redecodes_originals_and_keeps_names_in_provenance(tmp_path: Path) -> None:
    experiment, source, historical = _fixture(tmp_path, "evidence_a")

    result = compile_blocked_evidence(
        experiment,
        source_root=source,
        historical_reproduction=historical,
    )

    assert result["raw_source_gate_passed"] is False
    assert result["sol_status"] == "SOL_MODEL_UNAVAILABLE"
    assert result["provider_calls"] == 0
    assert result["candidate_dataset_created"] is False
    assert result["production_frozen"] is False
    assert result["training_authorized"] is False

    extraction = [json.loads(line) for line in (experiment / "extraction_manifest.jsonl").read_text().splitlines()]
    assert len(extraction) == 2
    assert all(row["record_id"].startswith("rec_") and len(row["record_id"]) == 68 for row in extraction)
    assert all(row["decode_status"] == "verified_from_original" for row in extraction)
    assert all(row["source_bytes_verified"] is True for row in extraction)
    assert len({row["record_id"] for row in extraction}) == 2
    assert len({row["blob_id"] for row in extraction}) == 1
    assert len({row["geometry_family_id"] for row in extraction}) == 1

    provenance_text = (experiment / "provenance_manifest.jsonl").read_text(encoding="utf-8")
    assert "helmet.png" in provenance_text
    assert "mineral.png" in provenance_text
    blind_evidence_paths = [
        experiment / "extraction_manifest.jsonl",
        experiment / "blob_manifest.jsonl",
        experiment / "suitability_manifest.jsonl",
        experiment / "relation_manifest.jsonl",
    ]
    blind_evidence = "\n".join(path.read_text(encoding="utf-8") for path in blind_evidence_paths)
    assert "helmet" not in blind_evidence.casefold()
    assert "mineral" not in blind_evidence.casefold()
    assert "misleading" not in blind_evidence.casefold()

    blob = next((experiment / "raw_audit_blobs").glob("*.rgba"))
    assert blob.name == f"{_sha256(blob.read_bytes())}.rgba"
    relations = [json.loads(line) for line in (experiment / "relation_manifest.jsonl").read_text().splitlines()]
    exact = [row for row in relations if row["relation_type"] == "exact_rgba"]
    packs = [row for row in relations if row["relation_type"] == "pack"]
    assert len(exact) == 1
    assert len(packs) == 1
    assert not any(row["relation_type"] == "source_pack" for row in relations)
    assert exact[0]["hard_split_constraint"] is True
    assert exact[0]["members"] == exact[0]["member_record_ids"]
    assert all(row["declared_variant_status"] == "unknown_not_inferred" for row in relations)

    report = json.loads((experiment / "rebuild_report.json").read_text(encoding="utf-8"))
    assert report["forensic_inventory"] == {
        "accept_count": 1,
        "quarantine_count": 1,
        "record_count": 2,
        "reject_count": 0,
    }
    assert report["raw_audit"]["two_complete_audit_rebuilds_byte_identical"] is True
    assert report["raw_audit"]["unique_geometry_count"] == 1
    assert report["raw_audit"]["suitability_status_counts"]["reject"] == 2
    external = report["external_read_only_audit_evidence"]
    assert external["authoritative_for_new_rebuild"] is False
    assert external["evidence"] == historical

    canary = json.loads((experiment / "sol_canary_report.json").read_text(encoding="utf-8"))
    assert canary["reason"] == "SOL_MODEL_UNAVAILABLE"
    assert canary["provider_calls"] == 0
    assert canary["stop_before_bulk"] is True
    assert canary["valid_json_rate"] is None
    assert canary["configured_identity"]["configured_model_identifier"] is None
    assert canary["configured_identity"]["provider_identity_status"] == "unavailable_not_attested"
    assert canary["configured_identity"]["blind_request_schema_version"] == "blind_semantic_request_v1"
    conflicts = json.loads((experiment / "source_conflict_report.json").read_text(encoding="utf-8"))
    assert conflicts["conflict_count"] is None
    assert conflicts["historical_synthetic_fixture_taint_cases"]["count"] == 4

    contact_index = json.loads((experiment / "contact_sheets" / "index.json").read_text(encoding="utf-8"))
    assert set(contact_index["groupings"]) == {
        "category",
        "source_pack",
        "creator_lineage",
        "uncertainty",
        "conflict_status",
        "quarantine_reason",
    }
    for view in VIEW_NAMES:
        manifest = json.loads((experiment / "candidate_view_manifests" / f"{view}.json").read_text())
        assert manifest["member_record_ids"] == []
        assert manifest["candidate_dataset_created"] is False
    assert not (tmp_path / "datasets" / "sprite_lab_v5_raw_rebuild_candidate_v1").exists()
    assert not (tmp_path / "datasets" / "sprite_lab_v5_raw_rebuild_frozen_v1").exists()


def test_blocked_evidence_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    first, source, historical = _fixture(tmp_path, "evidence_first")
    second, _, _ = _fixture(tmp_path, "evidence_second")

    compile_blocked_evidence(first, source_root=source, historical_reproduction=historical)
    compile_blocked_evidence(second, source_root=source, historical_reproduction=historical)

    assert _tree_hashes(first) == _tree_hashes(second)
    before = _tree_hashes(first)
    with pytest.raises(FileExistsError):
        compile_blocked_evidence(first, source_root=source, historical_reproduction=historical)
    assert _tree_hashes(first) == before


def test_blocked_evidence_fails_closed_on_archive_hash_conflict(tmp_path: Path) -> None:
    experiment, source, historical = _fixture(tmp_path, "evidence_bad_hash")
    hash_document = json.loads((experiment / "source_archive_hashes.json").read_text(encoding="utf-8"))
    hash_document["archives"]["downloads/misleading_pack.zip"] = "0" * 64
    _write_json(experiment / "source_archive_hashes.json", hash_document)

    with pytest.raises(EvidenceCompileError, match="inventory/archive hash manifest conflict"):
        compile_blocked_evidence(experiment, source_root=source, historical_reproduction=historical)

    assert not (experiment / "rebuild_report.json").exists()
    assert not (experiment / "raw_audit_blobs").exists()


def test_inventory_only_evidence_marks_pixel_claims_unverified(tmp_path: Path) -> None:
    experiment, _source, historical = _fixture(tmp_path, "evidence_inventory_only")

    compile_blocked_evidence(experiment, historical_reproduction=historical)

    extraction = [json.loads(line) for line in (experiment / "extraction_manifest.jsonl").read_text().splitlines()]
    assert all(row["decode_status"] == "inventory_observation_unverified" for row in extraction)
    assert all(row["source_bytes_verified"] is False for row in extraction)
    assert all(row["geometry_family_id"] is None for row in extraction)
    report = json.loads((experiment / "rebuild_report.json").read_text(encoding="utf-8"))
    assert report["raw_audit"]["mode"] == "inventory_observation_only"
    assert report["raw_audit"]["authoritative_dataset_rebuild"] is False
    assert report["raw_audit"]["materialized_blob_count"] == 0
