from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.dataset_v5.raw_cli import main


def _png_bytes() -> bytes:
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[1, 1:3] = [20, 80, 160, 255]
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG")
    return output.getvalue()


def _workspace(tmp_path: Path) -> tuple[Path, str]:
    image = _png_bytes()
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("opaque/member.png", image)
    archive_bytes = archive_buffer.getvalue()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    run = tmp_path / "harvest_runs" / "run_01"
    downloads = run / "downloads"
    downloads.mkdir(parents=True)
    (downloads / "source.zip").write_bytes(archive_bytes)
    source = {
        "author": "Fixture Author",
        "download_sha256": digest,
        "download_url": "https://example.test/source.zip",
        "license": {"license": "cc0", "user_confirmed": True},
        "original_filename": "source.zip",
        "source_id": "fixture_source",
        "source_name": "Fixture Source",
        "source_type": "direct_zip_url",
        "source_url": "https://example.test/source",
    }
    (run / "sources.jsonl").write_text(json.dumps(source) + "\n", encoding="utf-8")
    return tmp_path, digest


def test_raw_cli_requires_explicit_plan_and_verifies_two_builds(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    inventory = tmp_path / "inventory"
    assert (
        main(
            [
                "inventory-raw-sources",
                "--source-root",
                str(workspace),
                "--output",
                str(inventory),
            ]
        )
        == 0
    )
    inventory_row = json.loads((inventory / "raw_source_inventory.jsonl").read_text(encoding="utf-8"))
    plan = tmp_path / "plan.jsonl"
    plan.write_text(
        json.dumps(
            {
                "archive_member_path": "opaque/member.png",
                "crop_coordinates": [1, 1, 3, 2],
                "padding": {"bottom": 1, "left": 1, "right": 0, "top": 0},
                "schema_version": "sprite_lab_raw_extraction_plan_v1",
                "source_row_sha256": inventory_row["source_row_sha256"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "build_a"
    second = tmp_path / "build_b"

    result = main(
        [
            "rebuild-from-raw",
            "--source-root",
            str(workspace),
            "--inventory",
            str(inventory),
            "--plan",
            str(plan),
            "--output",
            str(first),
            "--verification-output",
            str(second),
        ]
    )

    assert result == 0
    assert json.loads((first / "extraction_manifest.jsonl").read_text())["interpolation_policy"] == "none"
    assert (first / "build_manifest.json").read_bytes() == (second / "build_manifest.json").read_bytes()


def test_raw_forensics_cli_records_blockers_without_aborting(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    output = tmp_path / "forensic_evidence"

    result = main(
        [
            "inventory-raw-forensics",
            "--source-root",
            str(workspace),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = (output / "raw_source_inventory_report.md").read_text(encoding="utf-8")
    assert "raw_source_gate_passed: `false`" in report
    assert (output / "raw_source_inventory.jsonl").is_file()
    assert (output / "source_archive_hashes.json").is_file()


def test_sol_canary_unavailable_writes_blocking_report_without_reading_cohort(tmp_path: Path, monkeypatch) -> None:
    for key in (
        "SPRITELAB_SOL_BACKEND",
        "SPRITELAB_SOL_MODEL",
        "SPRITELAB_SOL_BASE_URL",
        "SPRITELAB_SOL_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    missing_cohort = tmp_path / "does_not_exist.jsonl"
    output = tmp_path / "sol_canary_report.json"

    result = main(
        [
            "sol-canary",
            "--cohort",
            str(missing_cohort),
            "--image-root",
            str(tmp_path),
            "--projected-record-count",
            "2409",
            "--output",
            str(output),
        ]
    )

    assert result == 78
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["reason"] == "SOL_MODEL_UNAVAILABLE"
    assert report["provider_calls"] == 0
    assert report["canary_record_count"] == 0


def test_extraction_plan_rejects_semantic_or_inferred_fields(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    inventory = tmp_path / "inventory"
    assert main(["inventory-raw-sources", "--source-root", str(workspace), "--output", str(inventory)]) == 0
    inventory_row = json.loads((inventory / "raw_source_inventory.jsonl").read_text(encoding="utf-8"))
    plan = tmp_path / "tainted_plan.jsonl"
    plan.write_text(
        json.dumps(
            {
                "archive_member_path": "opaque/member.png",
                "crop_coordinates": None,
                "padding": None,
                "predicted_class": "helmet",
                "source_row_sha256": inventory_row["source_row_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        main(
            [
                "rebuild-from-raw",
                "--source-root",
                str(workspace),
                "--inventory",
                str(inventory),
                "--plan",
                str(plan),
                "--output",
                str(tmp_path / "first"),
                "--verification-output",
                str(tmp_path / "second"),
            ]
        )
    except ValueError as exc:
        assert "unsupported extraction-plan fields" in str(exc)
    else:
        raise AssertionError("semantic extraction-plan data was accepted")


def test_batch_cli_requires_and_audits_provenance_relations_and_splits(tmp_path: Path) -> None:
    first = "rec_" + "1" * 64
    second = "rec_" + "2" * 64
    records = tmp_path / "records.jsonl"
    provenance = tmp_path / "provenance.jsonl"
    relations = tmp_path / "relations.jsonl"
    splits = tmp_path / "splits.json"
    output = tmp_path / "batch_report.json"
    records.write_text(
        json.dumps(
            {
                "blind_request_payload": {
                    "metadata": {"record_id": first},
                    "request_id": "helmet.png",
                },
                "record_id": first,
                "source_binding_valid": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance.write_text(
        json.dumps({"original_filename": "helmet.png", "record_id": first}) + "\n",
        encoding="utf-8",
    )
    relations.write_text(
        json.dumps({"hard_split_constraint": True, "members": [first, second]}) + "\n",
        encoding="utf-8",
    )
    splits.write_text(json.dumps({first: "train", second: "eval"}), encoding="utf-8")

    result = main(
        [
            "audit-label-batch",
            "--records",
            str(records),
            "--batch-id",
            "batch-taint",
            "--provenance",
            str(provenance),
            "--relation-manifest",
            str(relations),
            "--split-map",
            str(splits),
            "--output",
            str(output),
        ]
    )

    assert result == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["filename_leakage"] == 1
    assert report["metrics"]["hard_relation_leakage"] == 1
    assert report["status"] == "blocked_non_authoritative"


def test_random_sample_cli_is_provider_free_preparation(tmp_path: Path, monkeypatch) -> None:
    for key in (
        "SPRITELAB_SOL_BACKEND",
        "SPRITELAB_SOL_MODEL",
        "SPRITELAB_SOL_BASE_URL",
        "SPRITELAB_SOL_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(json.dumps({"record_id": f"rec_{index:064x}"}) + "\n" for index in range(4)),
        encoding="utf-8",
    )
    output = tmp_path / "sample.jsonl"

    result = main(
        [
            "audit-random-sample",
            "--records",
            str(records),
            "--count",
            "2",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
