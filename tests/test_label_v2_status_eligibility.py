from __future__ import annotations

import json
from pathlib import Path

from _harvest_testdata import make_sprite_png
from spritelab.harvest.cli import main


def _make_status_run(tmp_path: Path) -> Path:
    pngs = tmp_path / "pngs"
    for name in ("apple.png", "carrot.png", "meat.png", "missing.png"):
        make_sprite_png(pngs / name)
    run_root = tmp_path / "harvest_runs"
    main(
        [
            "import-dir",
            "--dir",
            str(pngs),
            "--run-name",
            "oga_cc0_food_arlantr",
            "--run-root",
            str(run_root),
            "--source-id",
            "oga_cc0_food_arlantr",
            "--source-name",
            "Food Arlantr",
            "--license",
            "cc0",
            "--author",
            "Tester",
            "--user-confirmed-license",
        ]
    )
    run = run_root / "oga_cc0_food_arlantr"
    records = [
        json.loads(line) for line in (run / "imported.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_name = {Path(record["relative_path"]).name: record for record in records}
    by_name["apple.png"]["status"] = "accepted"
    by_name["carrot.png"]["status"] = "quarantine"
    by_name["meat.png"]["status"] = "needs_fix"
    by_name["missing.png"]["status"] = "quarantine"
    by_name["missing.png"]["final_png_path"] = str(run / "does_not_exist.png")
    (run / "imported.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return run


def test_label_v2_includes_onboarding_statuses_by_default(tmp_path: Path, capsys) -> None:
    run = _make_status_run(tmp_path)

    main(["label-v2", "--run", str(run), "--out", "label_v2_suggestions_fresh_novlm.jsonl", "--no-vlm"])

    output = capsys.readouterr().out
    rows = [
        json.loads(line)
        for line in (run / "label_v2_suggestions_fresh_novlm.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    assert {Path(row["relative_path"]).name for row in rows} == {"apple.png", "carrot.png", "meat.png"}
    assert "skipped_missing_png: 1" in output
    summary = json.loads((run / "label_v2_summary.json").read_text(encoding="utf-8"))
    assert summary["input_selection"]["skipped_by_reason"]["skipped_missing_png"] == 1


def test_label_v2_include_status_filters_when_requested(tmp_path: Path) -> None:
    run = _make_status_run(tmp_path)

    main(["label-v2", "--run", str(run), "--out", "accepted_only.jsonl", "--no-vlm", "--include-status", "accepted"])

    rows = [
        json.loads(line)
        for line in (run / "accepted_only.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert Path(rows[0]["relative_path"]).name == "apple.png"


def test_label_v2_no_vlm_does_not_reuse_existing_qwen_suggestions(tmp_path: Path) -> None:
    run = _make_status_run(tmp_path)
    rows = [
        json.loads(line) for line in (run / "imported.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    sprite_id = next(row["sprite_id"] for row in rows if Path(row["relative_path"]).name == "apple.png")
    (run / "qwen_suggestions.jsonl").write_text(
        json.dumps({"sprite_id": sprite_id, "category": "material", "object_name": "coin", "confidence": 0.99}) + "\n",
        encoding="utf-8",
    )

    main(["label-v2", "--run", str(run), "--out", "novlm.jsonl", "--no-vlm"])

    predictions = [
        json.loads(line) for line in (run / "novlm.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    apple = next(row for row in predictions if Path(row["relative_path"]).name == "apple.png")
    assert apple["vlm_status"] == "skipped_no_backend"
    assert apple["vlm_descriptor"] is None
