from __future__ import annotations

import json
from pathlib import Path

import pytest

from _harvest_testdata import make_sprite_png

from spritelab.harvest.apply_label_v2 import apply_label_v2_predictions
from spritelab.harvest.build_semantic_dataset import build_semantic_dataset
from spritelab.harvest.cli import main


def test_apply_label_v2_counts_auto_skip_reasons(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_jsonl(
        run / "imported.jsonl",
        [
            {"sprite_id": "missing_object", "status": "quarantine"},
            {"sprite_id": "missing_category", "status": "quarantine"},
            {"sprite_id": "missing_semantic", "status": "quarantine"},
        ],
    )
    _write_jsonl(
        run / "predictions.jsonl",
        [
            _prediction("missing_object", object_name="", category="item_icon", semantic=True),
            _prediction("missing_category", object_name="apple", category="unknown", semantic=True),
            _prediction("missing_semantic", object_name="apple", category="item_icon", semantic=False),
            _prediction("orphan", object_name="apple", category="item_icon", semantic=True),
        ],
    )

    report = apply_label_v2_predictions(
        run,
        prediction_file="predictions.jsonl",
        accept_auto=True,
        require_semantic_v3_for_auto=True,
    )

    assert report["raw_auto_rows_seen"] == 4
    assert report["auto_rows_skipped"] == 4
    assert report["auto_skip_reasons"]["missing_object_name"] == 1
    assert report["auto_skip_reasons"]["missing_category"] == 1
    assert report["auto_skip_reasons"]["missing_semantic_v3"] == 1
    assert report["auto_skip_reasons"]["sprite_id_mismatch"] == 1
    assert report["auto_validation_counts"]["missing_object_name"] == 1
    assert report["auto_validation_counts"]["missing_category"] == 1
    assert report["auto_validation_counts"]["missing_semantic_v3"] == 1


def test_build_semantic_dataset_report_includes_auto_skip_reasons(tmp_path: Path) -> None:
    png_dir = tmp_path / "pngs"
    make_sprite_png(png_dir / "apple.png")
    make_sprite_png(png_dir / "mystery.png", color=(40, 80, 120))
    run_root = tmp_path / "harvest_runs"
    main(
        [
            "import-dir",
            "--dir", str(png_dir),
            "--run-name", "food_pack",
            "--run-root", str(run_root),
            "--source-id", "oga_cc0_food_ocal",
            "--source-name", "Food",
            "--license", "cc0",
            "--author", "Tester",
            "--user-confirmed-license",
        ]
    )
    run = run_root / "food_pack"
    imported = [json.loads(line) for line in (run / "imported.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _write_jsonl(
        run / "label_v2_suggestions.jsonl",
        [
            _prediction(imported[0]["sprite_id"], object_name="apple", category="item_icon", semantic=False),
            _prediction(imported[1]["sprite_id"], object_name="", category="item_icon", semantic=False),
        ],
    )

    report = build_semantic_dataset(
        run,
        dataset_name="food_pack_label_v2_semantic_v3",
        output_root=tmp_path / "datasets",
        variants_per_sprite=2,
        overwrite=True,
    )

    assert report.ok
    payload = json.loads((Path(report.output_dir) / "semantic_dataset_build_report.json").read_text(encoding="utf-8"))
    assert payload["auto_rows_skipped"] == 1
    assert payload["auto_skip_reasons"]["missing_object_name"] == 1


def _prediction(sprite_id: str, *, object_name: str, category: str, semantic: bool) -> dict:
    row = {
        "sprite_id": sprite_id,
        "bucket": "auto_filename_trusted",
        "needs_review": False,
        "safe_prefill": {
            "category": category,
            "object_name": object_name,
            "tags": [object_name] if object_name else [],
            "short_description": object_name,
        },
        "label_quality": {"bucket": "auto_filename_trusted", "needs_review": False},
    }
    if semantic:
        row["semantic_v3"] = {"schema_version": "semantic_v3.0", "base_object": object_name or ""}
    return row


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
