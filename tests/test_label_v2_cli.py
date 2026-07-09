import json

from _harvest_testdata import make_sprite_png
from spritelab.harvest.cli import main


def test_fuse_prefill_v2_writes_safe_suggestions(tmp_path, capsys) -> None:
    root = tmp_path / "pngs"
    make_sprite_png(root / "butter.png")
    run_root = tmp_path / "harvest_runs"
    main(
        [
            "import-dir",
            "--dir",
            str(root),
            "--run-name",
            "oga_cc0_food_ocal",
            "--run-root",
            str(run_root),
            "--source-id",
            "oga_cc0_food_ocal",
            "--source-name",
            "Food",
            "--license",
            "cc0",
            "--author",
            "Tester",
            "--user-confirmed-license",
        ]
    )
    run = run_root / "oga_cc0_food_ocal"
    imported = json.loads((run / "imported.jsonl").read_text(encoding="utf-8").splitlines()[0])
    (run / "qwen_suggestions.jsonl").write_text(
        json.dumps(
            {
                "sprite_id": imported["sprite_id"],
                "category": "material",
                "object_name": "gold_bar",
                "tags": ["gold"],
                "confidence": 0.85,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = run / "label_v2_suggestions.jsonl"
    main(["fuse-prefill-v2", "--run", str(run), "--out", str(out)])
    assert "Suggestions: 1" in capsys.readouterr().out
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["safe_prefill"]["object_name"] == "butter"
    assert row["vlm_descriptor"]["object_name"] == "gold_bar"


def test_label_v2_report_and_eval_write_outputs(tmp_path, capsys) -> None:
    run = tmp_path / "run"
    run.mkdir()
    prediction = {
        "sprite_id": "apple",
        "safe_prefill": {"category": "item_icon", "object_name": "apple", "tags": ["fruit"]},
        "label_quality": {"bucket": "auto_filename_trusted", "needs_review": False},
    }
    (run / "label_v2_suggestions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    main(["label-v2-report", "--run", str(run)])
    assert "Total: 1" in capsys.readouterr().out

    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"sprite_id": "apple", "category": "item_icon", "object_name": "apple", "tags": ["fruit"]}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "eval.json"
    errors_out = tmp_path / "errors.jsonl"
    main(
        [
            "prefill-eval-v2",
            "--golden",
            str(golden),
            "--runs",
            str(run),
            "--out",
            str(out),
            "--errors-out",
            str(errors_out),
        ]
    )
    assert out.exists()
    assert errors_out.exists()


def test_golden_lint_cli_writes_fix_suggestions(tmp_path, capsys) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "sprite_id": "a",
                "category": "effect_icon",
                "object_name": "ice_cream_sandwich",
                "tags": ["ice_cream_sandwich"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "golden_fixed_suggestions.jsonl"
    main(["golden-lint", "--golden", str(golden), "--fix", "--out", str(out)])
    assert "Golden lint issues" in capsys.readouterr().out
    assert out.exists()


def test_old_fuse_prefill_help_still_works(capsys) -> None:
    try:
        main(["fuse-prefill", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "--min-qwen-confidence" in capsys.readouterr().out
