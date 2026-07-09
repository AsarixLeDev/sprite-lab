from __future__ import annotations

import json
from pathlib import Path

from spritelab.harvest.import_diagnostics import build_import_diagnostics, write_import_diagnostics_reports


def test_import_diagnostics_handles_missing_and_empty_imported(tmp_path: Path) -> None:
    run = tmp_path / "harvest_runs" / "empty_pack"
    run.mkdir(parents=True)
    _write_jsonl(
        run / "sources.jsonl",
        [
            {
                "source_id": "empty_pack",
                "source_name": "Empty Pack",
                "source_type": "local_directory",
                "local_root_path": "data/empty",
                "license": {"license": "cc0"},
            }
        ],
    )
    _write_jsonl(run / "candidates.jsonl", [])
    report_missing = build_import_diagnostics(run)
    assert report_missing["imported_jsonl_exists"] is False

    (run / "imported.jsonl").write_text("", encoding="utf-8")
    report_empty = build_import_diagnostics(run)
    assert report_empty["imported_jsonl_exists"] is True
    assert report_empty["imported_count"] == 0
    assert "import-dir" in report_empty["recommended_reimport_command"]


def test_import_diagnostics_reports_rejections_writes_outputs_and_does_not_mutate(tmp_path: Path) -> None:
    run = tmp_path / "harvest_runs" / "broken_pack"
    run.mkdir(parents=True)
    _write_jsonl(
        run / "sources.jsonl",
        [
            {
                "source_id": "broken_pack",
                "source_name": "Broken Pack",
                "source_type": "manual_zip",
                "local_archive_path": "data/broken.zip",
                "license": {"license": "cc0"},
            }
        ],
    )
    _write_jsonl(
        run / "candidates.jsonl",
        [
            {
                "candidate_id": "c0",
                "width": 4096,
                "height": 4096,
                "extracted_path": "missing.png",
                "rejection_reasons": ["image too large", "palette has too many colors"],
            }
        ],
    )
    _write_jsonl(run / "imported.jsonl", [])
    _write_jsonl(run / "rejected.jsonl", [{"sprite_id": "s0", "errors": ["bad license gate"]}])
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    out_md = tmp_path / "reports" / "diag.md"
    out_json = tmp_path / "reports" / "diag.json"

    report = build_import_diagnostics(run)
    write_import_diagnostics_reports(report, out_md=out_md, out_json=out_json)

    assert report["rejected_count"] == 1
    assert report["top_rejection_reasons"]["palette has too many colors"] == 1
    assert report["invalid_image_sizes"]
    assert report["missing_files"]
    assert "bad license gate" in report["bad_license_gate_reasons"]
    assert "import-zip" in report["recommended_reimport_command"]
    assert out_md.is_file()
    assert json.loads(out_json.read_text(encoding="utf-8"))["candidate_count"] == 1
    after = {path.name: path.read_bytes() for path in run.iterdir()}
    assert before == after


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
