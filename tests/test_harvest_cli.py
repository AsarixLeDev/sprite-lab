"""Tests for spritelab.harvest.cli."""

from __future__ import annotations

import json

import numpy as np
import pytest

from _harvest_testdata import make_sprite_png, make_zip_of_pngs
from spritelab.harvest.cli import main


def _import_dir_args(tmp_path, *, license_name="cc0", run_name="run_dir"):
    root = tmp_path / "pngs"
    make_sprite_png(root / "one.png")
    make_sprite_png(root / "two.png")
    return [
        "import-dir",
        "--dir",
        str(root),
        "--run-name",
        run_name,
        "--run-root",
        str(tmp_path / "harvest_runs"),
        "--source-id",
        "cli_source",
        "--source-name",
        "CLI Source",
        "--license",
        license_name,
        "--author",
        "Tester",
        "--user-confirmed-license",
    ]


def test_import_dir(tmp_path, capsys):
    main(_import_dir_args(tmp_path))
    output = capsys.readouterr().out
    assert "Valid: 2" in output
    run_dir = tmp_path / "harvest_runs" / "run_dir"
    assert (run_dir / "sources.jsonl").exists()
    assert (run_dir / "imported.jsonl").exists()
    assert (run_dir / "harvest_report.md").exists()


def test_import_zip(tmp_path, capsys):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png", "b.png", "c.png"])
    main(
        [
            "import-zip",
            "--zip",
            str(zip_path),
            "--run-name",
            "run_zip",
            "--run-root",
            str(tmp_path / "harvest_runs"),
            "--source-id",
            "zip_source",
            "--source-name",
            "Zip Source",
            "--license",
            "cc0",
            "--author",
            "Tester",
            "--user-confirmed-license",
        ]
    )
    output = capsys.readouterr().out
    assert "Valid: 3" in output


def test_export_writes_dataset(tmp_path, capsys):
    main(_import_dir_args(tmp_path))
    # accept everything valid via policy first
    run = str(tmp_path / "harvest_runs" / "run_dir")
    main(["apply-policy", "--run", run, "--auto-accept-valid-cc0", "--reject-invalid"])
    main(
        [
            "export",
            "--run",
            run,
            "--dataset-name",
            "cli_dataset",
            "--output-root",
            str(tmp_path / "datasets"),
        ]
    )
    npz = tmp_path / "datasets" / "cli_dataset" / "train.npz"
    assert npz.exists()
    with np.load(npz, allow_pickle=False) as data:
        assert "index_map" in data.files


def test_export_blocked_without_override(tmp_path, capsys):
    main(_import_dir_args(tmp_path, license_name="unknown", run_name="run_unknown"))
    run = str(tmp_path / "harvest_runs" / "run_unknown")
    with pytest.raises(SystemExit):
        main(
            [
                "export",
                "--run",
                run,
                "--dataset-name",
                "blocked",
                "--output-root",
                str(tmp_path / "datasets"),
            ]
        )
    assert "Export blocked" in capsys.readouterr().out
    main(
        [
            "export",
            "--run",
            run,
            "--dataset-name",
            "unblocked",
            "--output-root",
            str(tmp_path / "datasets"),
            "--allow-unknown-license",
        ]
    )
    assert (tmp_path / "datasets" / "unblocked" / "train.npz").exists()


def test_cli_help():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_qwen_prefill_alias_help_includes_ollama_and_runpod_token(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["qwen_prefill", "--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "ollama" in output
    assert "--runpod-token" in output
    assert "--workers" in output
    assert "--retry-attempts" in output
    assert "--no-filename-hint" in output
    assert "--no-propagate-dups" in output
    assert "--propagate-near-dups" in output


def test_assisted_golden_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["assisted-golden", "--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--include-status" in output
    assert "--order" in output


def test_assisted_golden_sample_cli_writes_candidates(tmp_path, capsys):
    main(_import_dir_args(tmp_path))
    run = tmp_path / "harvest_runs" / "run_dir"

    main(["assisted-golden-sample", "--run", str(run), "--n", "1", "--seed", "1"])

    output = capsys.readouterr().out
    assert "Candidates: 1" in output
    rows = [json.loads(line) for line in (run / "golden_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert "suggested_category" in rows[0]


def test_filename_prefill_cli_outputs_filename_rule_json(capsys):
    main(
        [
            "filename-prefill",
            "--sprite-id",
            "oga_496_rpg_icons_32fix_i_c_banana",
            "--filename",
            "I_C_Banana.png",
        ]
    )

    data = json.loads(capsys.readouterr().out)
    assert data["source"] == "filename_rules"
    assert data["category"] == "item_icon"
    assert data["object_name"] == "banana"
    assert data["tags"] == ["banana", "fruit", "food", "consumable"]


def test_fuse_prefill_cli_writes_fused_suggestions(tmp_path, capsys):
    main(_import_dir_args(tmp_path))
    run = tmp_path / "harvest_runs" / "run_dir"
    out = run / "fused.jsonl"

    main(["fuse-prefill", "--run", str(run), "--out", str(out)])

    output = capsys.readouterr().out
    assert "Suggestions: 2" in output
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert "filename_suggestion" in rows[0]
    assert "fused_suggestion" in rows[0]
    assert "prefill_quality" in rows[0]
