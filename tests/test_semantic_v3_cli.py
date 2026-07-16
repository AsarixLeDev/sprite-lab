from __future__ import annotations

import json
from pathlib import Path

from spritelab.harvest.apply_label_v2 import apply_label_v2_predictions
from spritelab.harvest.cli import main


def _prediction(sprite_id: str, object_name: str, category: str, *, bucket: str = "auto_filename_trusted") -> dict:
    return {
        "sprite_id": sprite_id,
        "relative_path": f"{sprite_id}.png",
        "candidate_object_names": [object_name],
        "safe_prefill": {
            "category": category,
            "object_name": object_name,
            "tags": [object_name, category],
            "short_description": f"A {object_name.replace('_', ' ')} icon.",
            "materials": [],
            "mood": [],
        },
        "visual_facts": {"dominant_colors": ["red", "black"], "shape_hints": ["roundish"]},
        "vlm_descriptor": {"object_name": object_name, "alternative_object_names": []},
        "source_profile": {"name": "oga_496_rpg_icons", "domain": "rpg_icons"},
        "bucket": bucket,
        "needs_review": False,
        "flags": [],
        "label_quality": {"bucket": bucket, "flags": [], "needs_review": False},
    }


def _imported_record(sprite_id: str, *, status: str = "accepted") -> dict:
    return {
        "sprite_id": sprite_id,
        "candidate_id": f"candidate_{sprite_id}",
        "source_id": "source_pack",
        "final_png_path": f"{sprite_id}.png",
        "relative_path": f"{sprite_id}.png",
        "status": status,
        "category": "unknown",
        "object_name": "",
        "tags": [],
        "notes": "",
        "source_name": "Source Pack",
        "license": "cc0",
        "author": "Artist",
        "auto_metadata": {},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    _write_jsonl(
        run / "preds.jsonl",
        [
            _prediction("sprite_a", "red_potion", "item_icon"),
            _prediction("sprite_b", "golden_chestplate", "armor", bucket="auto_rpg_496_specialized"),
        ],
    )
    return run


def test_semantic_v3_cli_writes_enriched_prediction_jsonl(tmp_path: Path, capsys) -> None:
    run = _write_run(tmp_path)

    main(["semantic-v3", "--run", str(run), "--prediction-file", "preds.jsonl", "--out", "preds_semantic.jsonl"])

    output = capsys.readouterr().out
    assert "Records: 2" in output
    assert "Records with semantic_v3: 2" in output

    records = _read_jsonl(run / "preds_semantic.jsonl")
    assert len(records) == 2
    for record in records:
        semantic = record["semantic_v3"]
        assert semantic["schema_version"].startswith("semantic_v3")
        assert semantic["base_object"]
        assert semantic["captions"]
        # primary labels are untouched
        assert record["safe_prefill"]["object_name"] == semantic["object_name"]
        assert record["safe_prefill"]["category"] == semantic["category"]
    by_id = {record["sprite_id"]: record for record in records}
    assert by_id["sprite_a"]["semantic_v3"]["base_object"] == "potion"
    assert by_id["sprite_b"]["semantic_v3"]["base_object"] == "chestplate"


def test_semantic_v3_cli_default_output_name(tmp_path: Path) -> None:
    run = _write_run(tmp_path)

    main(["semantic-v3", "--run", str(run), "--prediction-file", "preds.jsonl"])

    assert (run / "preds_semantic_v3.jsonl").exists()


def test_semantic_v3_report_prints_summary(tmp_path: Path, capsys) -> None:
    run = _write_run(tmp_path)
    main(["semantic-v3", "--run", str(run), "--prediction-file", "preds.jsonl", "--out", "preds_semantic.jsonl"])
    capsys.readouterr()

    main(
        [
            "semantic-v3-report",
            "--run",
            str(run),
            "--prediction-file",
            "preds_semantic.jsonl",
            "--out-json",
            "semantic_summary.json",
        ]
    )

    output = capsys.readouterr().out
    assert "Semantic v3 Report" in output
    assert "Records with semantic_v3: 2" in output
    assert "Top base objects" in output
    summary = json.loads((run / "semantic_summary.json").read_text(encoding="utf-8"))
    assert summary["records_with_semantic_v3"] == 2
    assert summary["top_base_objects"]["potion"] == 1


def test_apply_label_v2_preserves_semantic_v3_metadata(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    _write_jsonl(run / "imported.jsonl", [_imported_record("sprite_a"), _imported_record("sprite_b")])
    main(["semantic-v3", "--run", str(run), "--prediction-file", "preds.jsonl", "--out", "preds_semantic.jsonl"])

    apply_label_v2_predictions(run, prediction_file="preds_semantic.jsonl", mode="auto-only", accept_auto=True)

    imported = {record["sprite_id"]: record for record in _read_jsonl(run / "imported.jsonl")}
    for sprite_id, base in (("sprite_a", "potion"), ("sprite_b", "chestplate")):
        record = imported[sprite_id]
        assert record["auto_metadata"]["label_v2_applied"] is True
        semantic = record["auto_metadata"]["semantic_v3"]
        assert semantic["base_object"] == base
        assert semantic["captions"]
        assert record["object_name"] == semantic["object_name"]


def test_apply_label_v2_still_works_without_semantic_v3(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    _write_jsonl(run / "imported.jsonl", [_imported_record("sprite_a"), _imported_record("sprite_b")])

    apply_label_v2_predictions(run, prediction_file="preds.jsonl", mode="auto-only", accept_auto=True)

    imported = {record["sprite_id"]: record for record in _read_jsonl(run / "imported.jsonl")}
    assert imported["sprite_a"]["auto_metadata"]["label_v2_applied"] is True
    assert "semantic_v3" not in imported["sprite_a"]["auto_metadata"]
