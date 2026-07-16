from __future__ import annotations

import json
from pathlib import Path

from spritelab.harvest.cli import main
from spritelab.harvest.pack_drilldown import build_pack_drilldown


def _write_run(tmp_path: Path) -> Path:
    run = tmp_path / "harvest_runs" / "oga_cc0_key_rcorre"
    run.mkdir(parents=True)
    imported = [
        {
            "sprite_id": "key_rcorre",
            "relative_path": "key_rcorre.png",
            "final_png_path": "key_rcorre.png",
            "status": "quarantine",
            "source_id": "oga_cc0_key_rcorre",
            "source_name": "Keys",
            "errors": [],
        },
        {
            "sprite_id": "r000_c001",
            "relative_path": "r000_c001.png",
            "final_png_path": "r000_c001.png",
            "status": "quarantine",
            "source_id": "oga_cc0_key_rcorre",
            "source_name": "Keys",
            "errors": [],
        },
    ]
    predictions = [
        {
            "sprite_id": "key_rcorre",
            "relative_path": "key_rcorre.png",
            "bucket": "needs_review",
            "needs_review": True,
            "candidate_object_names": ["key"],
            "source_profile": {"name": "cc0_key", "filename_trust": "exact"},
            "safe_prefill": {"category": "item_icon", "object_name": "key_rcorre"},
            "label_quality": {"bucket": "needs_review", "needs_review": True},
        },
        {
            "sprite_id": "r000_c001",
            "relative_path": "r000_c001.png",
            "bucket": "needs_review",
            "needs_review": True,
            "candidate_object_names": ["key"],
            "source_profile": {"name": "cc0_key", "filename_trust": "exact"},
            "safe_prefill": {"category": "unknown", "object_name": ""},
            "label_quality": {"bucket": "needs_review", "needs_review": True},
        },
    ]
    _write_jsonl(run / "imported.jsonl", imported)
    _write_jsonl(run / "label_v2_suggestions_triage.jsonl", predictions)
    return run


def test_pack_drilldown_reports_blocking_classes_and_writes_outputs(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    out_md = tmp_path / "reports" / "pack.md"
    out_json = tmp_path / "reports" / "pack.json"
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    main(
        [
            "pack-drilldown",
            "--run",
            str(run),
            "--prediction-file",
            "label_v2_suggestions_triage.jsonl",
            "--out",
            str(out_md),
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["raw_prediction_records"] == 2
    assert payload["needs_review_count"] == 2
    assert "all_predictions_need_review" in payload["recommended_fix_classes"]
    assert "source_author_token_misparsed" in payload["recommended_fix_classes"]
    assert "sheet_coordinate_only" in payload["recommended_fix_classes"]
    assert out_md.is_file()
    assert "Pack Drilldown" in out_md.read_text(encoding="utf-8")
    after = {path.name: path.read_bytes() for path in run.iterdir()}
    assert before == after


def test_pack_drilldown_flags_semantic_prediction_file_used_as_raw(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    semantic = run / "label_v2_suggestions_triage_semantic_v3.jsonl"
    semantic.write_text((run / "label_v2_suggestions_triage.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    report = build_pack_drilldown(run, prediction_file=semantic.name)

    assert "semantic_prediction_file_used_as_raw" in report["recommended_fix_classes"]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
