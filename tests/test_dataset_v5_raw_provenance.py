from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from spritelab.provenance.raw_remediation import (
    SourceResolutionError,
    _apply_bound_license_url_evidence,
    _apply_exact_historical_archive_bindings,
    _apply_exact_license_bindings,
    _apply_historical_acquisition_records,
    _apply_recovery_archive_authority,
    _exclusion_reasons,
    _public_source,
    filter_candidate_records,
    render_download_recovery_script,
    verify_local_zip_candidate,
)


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(**updates: object) -> dict[str, object]:
    source: dict[str, object] = {
        "source_binding_id": "sb_test",
        "distribution_platform": "example.invalid",
        "creator_or_publisher": "creator",
        "pack_or_collection": "pack-a",
        "acquisition_run": "run-a",
        "source_page_url": "https://example.invalid/pack-a",
        "direct_download_url": "https://example.invalid/pack-a.zip",
        "original_archive_path": "run-a/pack.zip",
        "original_archive_filename": "pack-a.zip",
        "historical_archive_sha256": "a" * 64,
        "current_observed_archive_sha256": "a" * 64,
        "historical_hash_authority": "historical_source_manifest:test:1",
        "license_identifier": "cc0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_evidence": [{"evidence_type": "test"}],
        "provenance_status": "verified",
        "resolution_status": "verified",
        "exclusion_reason": None,
        "terminal_gate_status": "verified",
        "eligible_for_candidate_membership": True,
    }
    source.update(updates)
    return source


def test_current_hash_is_not_promoted_to_historical() -> None:
    source = _source(
        historical_archive_sha256=None,
        historical_hash_authority=None,
        original_archive_filename=None,
    )

    _apply_exact_historical_archive_bindings([source])

    assert source["current_observed_archive_sha256"] == "a" * 64
    assert source["historical_archive_sha256"] is None
    assert source["historical_hash_authority"] is None


def test_license_is_not_inferred_across_packs() -> None:
    provider = _source(source_binding_id="sb_provider", pack_or_collection="pack-a")
    target = _source(
        source_binding_id="sb_target",
        pack_or_collection="pack-b",
        license_identifier=None,
        license_url=None,
        license_evidence=[],
    )

    _apply_exact_license_bindings([provider, target])

    assert target["license_identifier"] is None
    assert target["license_url"] is None


def test_exact_bound_license_url_resolves_only_its_own_binding() -> None:
    source = _source(
        license_identifier=None,
        license_url="https://creativecommons.org/licenses/by/3.0/",
        license_evidence=[{"evidence_type": "historical_manifest_license_record", "user_confirmed": True}],
    )

    _apply_bound_license_url_evidence([source])

    assert source["license_identifier"] == "cc_by_3_0"
    assert source["license_evidence"][-1]["source_binding_id"] == "sb_test"


def test_missing_license_url_is_not_guessed_from_source_page() -> None:
    source = _source(license_url=None, license_evidence=[])

    _apply_exact_license_bindings([source])

    assert source["source_page_url"]
    assert source["license_url"] is None


def test_exact_local_zip_recovery_uses_bytes_and_members(tmp_path: Path) -> None:
    archive = tmp_path / "pack.zip"
    members = {"sprites/a.png": b"a", "LICENSE.txt": b"CC0"}
    _zip(archive, members)

    result = verify_local_zip_candidate(
        archive,
        expected_sha256=_digest(archive),
        expected_size=archive.stat().st_size,
        expected_members={name: hashlib.sha256(payload).hexdigest() for name, payload in members.items()},
    )

    assert result["match"] is True
    assert result["reasons"] == ["exact_byte_hash_and_member_evidence_match"]


def test_exact_historical_standalone_recovery(tmp_path: Path) -> None:
    original = tmp_path / "data_sources" / "itempack" / "sheet.png"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"historical original sheet")
    digest = _digest(original)
    run = tmp_path / "harvest_runs" / "run-a"
    run.mkdir(parents=True)
    candidate = {
        "extracted_path": "data_sources/itempack/sheet.png",
        "image_sha256": digest,
        "relative_path": "sheet.png",
        "source_id": "source-a",
        "source_path": "https://example.invalid/pack-a",
        "status": "candidate",
    }
    (run / "candidates.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    event = {"at": "2026-01-02T03:04:05Z", "event": "import", "source_id": "source-a"}
    (run / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    source = _source(
        acquisition_run="run-a",
        current_observed_archive_sha256=digest,
        direct_download_url=None,
        historical_archive_sha256=None,
        historical_hash_authority=None,
        original_archive_filename=None,
        original_archive_path=None,
        physical_paths=["data_sources/itempack/sheet.png"],
        source_id="source-a",
        source_record={"created_at": event["at"], "local_root_path": "data_sources/itempack"},
    )

    recovered = _apply_historical_acquisition_records([source], tmp_path)

    assert len(recovered) == 1
    assert source["historical_archive_sha256"] == digest
    assert source["original_archive_filename"] == "sheet.png"
    assert source["local_original_recovery_verified"] is True


def test_exact_recovery_archive_authority_requires_full_identity(tmp_path: Path) -> None:
    provider = _source(
        acquisition_run="repair-run",
        direct_download_url="https://example.invalid/original.zip",
        historical_archive_sha256="b" * 64,
        current_observed_archive_sha256="b" * 64,
        local_original_recovery_verified=True,
        physical_paths=["downloads/original.zip"],
        source_id="repair-source",
    )
    target = _source(
        source_binding_id="sb_target",
        direct_download_url="https://example.invalid/original.zip",
        historical_archive_sha256=None,
        historical_hash_authority=None,
        current_observed_archive_sha256="b" * 64,
        original_archive_filename=None,
        physical_paths=["downloads/original.zip"],
        source_id="original-source",
    )
    repair = {
        "path": tmp_path / "repair.json",
        "record_sha256": "c" * 64,
        "record": {
            "download_sha256": "b" * 64,
            "license": "cc0",
            "license_page": "https://example.invalid/pack-a",
            "local_download_path": "downloads/original.zip",
            "recorded_download_url": "https://example.invalid/original.zip",
            "recorded_source_url": "https://example.invalid/pack-a",
            "server_url_filename": "original.zip",
            "source_id": "repair-source",
            "source_run": "repair-run",
        },
    }

    _apply_recovery_archive_authority([provider, target], [repair], tmp_path)

    assert target["historical_archive_sha256"] == "b" * 64
    assert target["historical_hash_authority"].startswith("exact_archive_recovery:")
    assert target["original_archive_filename"] == "original.zip"


def test_wrong_zip_with_same_filename_is_rejected(tmp_path: Path) -> None:
    expected = tmp_path / "expected" / "pack.zip"
    wrong = tmp_path / "wrong" / "pack.zip"
    _zip(expected, {"sprite.png": b"expected"})
    _zip(wrong, {"sprite.png": b"wrong"})

    result = verify_local_zip_candidate(wrong, expected_sha256=_digest(expected))

    assert expected.name == wrong.name
    assert result["match"] is False
    assert "byte_hash_mismatch" in result["reasons"]


def test_source_exclusion_removes_all_dependent_records() -> None:
    records = [
        {"record_id": "excluded", "source_binding_ids": ["sb_excluded"]},
        {"record_id": "verified", "source_binding_ids": ["sb_verified"]},
    ]
    resolutions = {
        "sb_excluded": {"terminal_gate_status": "excluded"},
        "sb_verified": {"terminal_gate_status": "verified"},
    }

    filtered = filter_candidate_records(records, resolutions)

    assert [row["record_id"] for row in filtered] == ["verified"]
    assert all("sb_excluded" not in row["source_binding_ids"] for row in filtered)


def test_download_script_defaults_to_dry_run(tmp_path: Path) -> None:
    script = tmp_path / "remediation" / "download_recovery_plan.ps1"
    script.parent.mkdir(parents=True)
    script.write_text(
        render_download_recovery_script(
            [
                {
                    "source_binding_id": "sb_download",
                    "expected_direct_download_url": "https://example.invalid/original.zip",
                    "expected_filename": "original.zip",
                    "exact_recovery_possible": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert not (tmp_path / "v5_raw_rebuild_sol_v1" / "recovered_downloads_v1").exists()


def test_download_script_refuses_destination_overwrite_before_network(tmp_path: Path) -> None:
    script = tmp_path / "remediation" / "download_recovery_plan.ps1"
    destination = tmp_path / "v5_raw_rebuild_sol_v1" / "recovered_downloads_v1" / "original.zip"
    script.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"historical")
    script.write_text(
        render_download_recovery_script(
            [
                {
                    "source_binding_id": "sb_download",
                    "expected_direct_download_url": "https://example.invalid/original.zip",
                    "expected_filename": "original.zip",
                    "exact_recovery_possible": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-ExecuteDownloads",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to overwrite recovery destination" in result.stderr
    assert destination.read_bytes() == b"historical"


def test_unknown_license_blocks_inclusion() -> None:
    source = _source(license_identifier=None, license_url=None, license_evidence=[])

    reasons = _exclusion_reasons(source)

    assert "unknown_license" in reasons
    assert "missing_license_evidence" in reasons


def test_missing_provenance_blocks_inclusion() -> None:
    source = _source(historical_archive_sha256=None, historical_hash_authority=None)

    reasons = _exclusion_reasons(source)

    assert "missing_historical_archive_sha256" in reasons
    assert "missing_historical_hash_authority" in reasons


def test_every_source_requires_terminal_resolution() -> None:
    source = _source(resolution_status="provenance_incomplete")

    with pytest.raises(SourceResolutionError, match="non-terminal source resolution"):
        _public_source(source)


def test_download_script_contains_only_exact_recorded_urls() -> None:
    entry = {
        "source_binding_id": "sb_manual",
        "expected_direct_download_url": None,
        "expected_filename": None,
        "exact_recovery_possible": False,
    }

    script = render_download_recovery_script([entry])

    assert "DirectUrl = $null" in script
    assert "Filename = $null" in script
    assert "Invoke-WebRequest" in script
    assert "Where-Object" in script


def test_normalized_binding_has_every_required_field() -> None:
    public = _public_source(_source())

    assert public["resolution_status"] == "verified"
    json.dumps(public, sort_keys=True)
