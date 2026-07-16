from __future__ import annotations

import json
from pathlib import Path

from spritelab.harvest.acceptance_gap_report import build_acceptance_gap_report, write_acceptance_gap_reports


def test_acceptance_gap_report_ranks_raw_auto_gap_and_writes_reports(tmp_path: Path) -> None:
    runs = tmp_path / "harvest_runs"
    datasets = tmp_path / "datasets"
    run = runs / "gap_pack"
    run.mkdir(parents=True)
    _write_jsonl(
        run / "imported.jsonl",
        [
            {"sprite_id": "s0", "status": "quarantine"},
            {"sprite_id": "s1", "status": "quarantine"},
            {"sprite_id": "s2", "status": "quarantine"},
        ],
    )
    _write_jsonl(
        run / "label_v2_suggestions.jsonl",
        [
            _prediction("s0", "apple"),
            _prediction("s1", "carrot"),
            _prediction("s2", "", bucket="needs_review", needs_review=True),
        ],
    )
    (run / "label_v2_apply_report.json").write_text(
        json.dumps(
            {
                "applied_auto_labels": 1,
                "accepted_auto_labels": 1,
                "auto_skip_reasons": {"missing_object_name": 1},
                "auto_validation_counts": {"missing_object_name": 1},
            }
        ),
        encoding="utf-8",
    )

    out_md = tmp_path / "reports" / "gap.md"
    out_json = tmp_path / "reports" / "gap.json"
    report = build_acceptance_gap_report(runs, datasets)
    write_acceptance_gap_reports(report, out_md=out_md, out_json=out_json)

    assert report["packs"][0]["pack"] == "gap_pack"
    assert report["packs"][0]["gap_raw_auto_to_accepted"] == 1
    assert report["packs"][0]["recommended_fix_class"] in {"raw_auto_not_applied", "base_object_extractor_gap"}
    assert out_md.is_file()
    assert json.loads(out_json.read_text(encoding="utf-8"))["pack_count"] == 1


def test_acceptance_gap_report_reports_accepted_to_exported_gap(tmp_path: Path) -> None:
    runs = tmp_path / "harvest_runs"
    datasets = tmp_path / "datasets"
    run = runs / "export_gap"
    run.mkdir(parents=True)
    _write_jsonl(run / "imported.jsonl", [{"sprite_id": "s0", "status": "accepted"}])
    _write_jsonl(run / "label_v2_suggestions.jsonl", [_prediction("s0", "apple")])
    (run / "label_v2_apply_report.json").write_text(
        json.dumps({"applied_auto_labels": 2, "accepted_auto_labels": 2}),
        encoding="utf-8",
    )
    dataset = datasets / "export_gap_label_v2_semantic_v3"
    dataset.mkdir(parents=True)
    for split in ("train", "val", "test"):
        _write_jsonl(dataset / f"manifest_{split}.jsonl", [_manifest("s0")] if split == "train" else [])
    (dataset / "train.npz").write_bytes(b"placeholder")
    (dataset / "dataset_qa_report.json").write_text(json.dumps({"errors": []}), encoding="utf-8")
    (dataset / "training_manifest_qa_report.json").write_text(json.dumps({"errors": []}), encoding="utf-8")

    report = build_acceptance_gap_report(runs, datasets)
    row = next(pack for pack in report["packs"] if pack["pack"] == "export_gap")

    assert row["gap_accepted_to_exported"] == 1
    assert row["exported_count"] == 1


def _prediction(
    sprite_id: str, object_name: str, *, bucket: str = "auto_filename_trusted", needs_review: bool = False
) -> dict:
    return {
        "sprite_id": sprite_id,
        "bucket": bucket,
        "needs_review": needs_review,
        "safe_prefill": {
            "category": "item_icon",
            "object_name": object_name,
            "tags": [object_name] if object_name else [],
        },
        "label_quality": {"bucket": bucket, "needs_review": needs_review},
    }


def _manifest(sprite_id: str) -> dict:
    return {
        "sprite_id": sprite_id,
        "split": "train",
        "category": "item_icon",
        "object_name": "apple",
        "tags": ["apple"],
        "semantic_v3": {"schema_version": "semantic_v3.0", "base_object": "apple"},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
