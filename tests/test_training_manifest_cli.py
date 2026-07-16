from __future__ import annotations

import json
from pathlib import Path

import pytest

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.harvest.cli import main


def _dataset(tmp_path: Path) -> Path:
    return make_semantic_dataset(tmp_path / "ds", default_specs())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_training_manifest_cli(tmp_path: Path, capsys) -> None:
    dataset = _dataset(tmp_path)
    out = dataset / "training_manifest.jsonl"
    main(
        [
            "build-training-manifest",
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--caption-policy",
            "mixed",
            "--variants-per-sprite",
            "8",
            "--seed",
            "4962026",
        ]
    )
    output = capsys.readouterr().out
    assert "Total rows: 48" in output
    assert "Unique sprites: 6" in output
    rows = _read_jsonl(out)
    assert len(rows) == 48
    assert (dataset / "training_manifest_report.json").is_file()
    assert (dataset / "training_manifest_report.md").is_file()


def test_build_training_manifest_default_out(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    main(["build-training-manifest", "--dataset", str(dataset), "--variants-per-sprite", "2"])
    assert (dataset / "training_manifest.jsonl").is_file()


def test_build_training_manifest_per_split(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    main(
        [
            "build-training-manifest",
            "--dataset",
            str(dataset),
            "--variants-per-sprite",
            "3",
            "--per-split",
        ]
    )
    for split in ("train", "val", "test"):
        assert (dataset / f"training_manifest_{split}.jsonl").is_file()


def test_training_manifest_qa_cli_passes(tmp_path: Path, capsys) -> None:
    dataset = _dataset(tmp_path)
    main(["build-training-manifest", "--dataset", str(dataset), "--variants-per-sprite", "8"])
    capsys.readouterr()

    main(["training-manifest-qa", "--dataset", str(dataset)])
    output = capsys.readouterr().out
    assert "Errors: 0" in output
    assert (dataset / "training_manifest_qa_report.json").is_file()


def test_training_manifest_qa_cli_exits_nonzero_on_error(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    # A manifest that does not exist -> error -> non-zero exit.
    with pytest.raises(SystemExit):
        main(["training-manifest-qa", "--dataset", str(dataset), "--manifest", str(dataset / "nope.jsonl")])


def test_build_eval_prompts_cli(tmp_path: Path, capsys) -> None:
    dataset = _dataset(tmp_path)
    main(["build-eval-prompts", "--dataset", str(dataset), "--seed", "4962026"])
    output = capsys.readouterr().out
    assert "Total prompts:" in output
    prompts = _read_jsonl(dataset / "eval_prompts.jsonl")
    categories = {p["category"] for p in prompts}
    assert "seen_object" in categories
    assert "unseen_composition" in categories
    assert "creative_concept" in categories
    assert "negative_control" in categories
    assert (dataset / "eval_prompts_report.json").is_file()


def test_training_manifest_report_cli(tmp_path: Path, capsys) -> None:
    dataset = _dataset(tmp_path)
    out = dataset / "training_manifest.jsonl"
    main(["build-training-manifest", "--dataset", str(dataset), "--out", str(out), "--variants-per-sprite", "4"])
    capsys.readouterr()

    main(["training-manifest-report", "--manifest", str(out)])
    output = capsys.readouterr().out
    assert "Training Manifest Report" in output
    assert "Total rows: 24" in output
