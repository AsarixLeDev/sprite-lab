from __future__ import annotations

from pathlib import Path

from _harvest_testdata import make_sprite_png
from spritelab.harvest.cli import main


def _make_run(tmp_path: Path) -> Path:
    pngs = tmp_path / "pngs"
    make_sprite_png(pngs / "apple.png")
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
    return run_root / "oga_cc0_food_arlantr"


def test_label_v2_relative_out_writes_inside_run_dir(tmp_path: Path, capsys) -> None:
    run = _make_run(tmp_path)

    main(["label-v2", "--run", str(run), "--out", "label_v2_suggestions_fresh_novlm.jsonl", "--no-vlm"])

    output = capsys.readouterr().out
    expected = run / "label_v2_suggestions_fresh_novlm.jsonl"
    assert expected.is_file()
    assert f"Wrote: {expected}" in output
    assert not (Path.cwd() / "label_v2_suggestions_fresh_novlm.jsonl").exists()


def test_label_v2_absolute_out_is_respected(tmp_path: Path, capsys) -> None:
    run = _make_run(tmp_path)
    out = tmp_path / "absolute_predictions.jsonl"

    main(["label-v2", "--run", str(run), "--out", str(out), "--no-vlm"])

    assert out.is_file()
    assert f"Wrote: {out}" in capsys.readouterr().out
