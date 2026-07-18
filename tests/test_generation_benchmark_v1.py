from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.evaluation import suite as evaluation_suite_module
from spritelab.evaluation.cli import main
from spritelab.evaluation.conditional import score_conditions
from spritelab.evaluation.memorization import TrainingImage, retrieve_neighbors
from spritelab.evaluation.metric_definitions import IncompatibleMetricDefinitions
from spritelab.evaluation.metrics import batch_metrics, score_image, score_rgba
from spritelab.evaluation.suite import compare_reports, human_package, score_suite


def _sprite(color: tuple[int, int, int] = (200, 50, 40), *, shift: int = 0) -> np.ndarray:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:22, 10 + shift : 20 + shift, :3] = color
    rgba[8:22, 10 + shift : 20 + shift, 3] = 255
    return rgba


def _metadata(sample_id: str = "s0", noise_seed: int = 10) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "prompt_id": "p0",
        "prompt": "red sword",
        "checkpoint": "checkpoint.pt",
        "seed": 1,
        "noise_seed": noise_seed,
        "steps": 30,
        "cfg_scale": 3.0,
        "model_output_finite": True,
        "category": "weapon",
    }


def _run(root: Path, rgba: np.ndarray, *, sample_id: str = "s0", noise_seed: int = 10) -> Path:
    image_dir = root / "hard_rgba"
    image_dir.mkdir(parents=True)
    Image.fromarray(rgba, "RGBA").save(image_dir / f"{sample_id}.png")
    row = {**_metadata(sample_id, noise_seed), "paths": {"hard_rgba": f"hard_rgba/{sample_id}.png"}}
    (root / "generated_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


def test_malformed_png(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"not a png")
    assert score_image(path, _metadata())["hard_validity"]["corrupt_output"] is True


def test_wrong_size(tmp_path: Path) -> None:
    path = tmp_path / "wrong.png"
    Image.new("RGBA", (31, 32)).save(path)
    result = score_image(path, _metadata())
    assert result["hard_validity"]["exact_dimensions"] is False
    assert result["hard_validity"]["pass"] is False


def test_empty_alpha() -> None:
    result = score_rgba(np.zeros((32, 32, 4), dtype=np.uint8), _metadata())
    assert result["hard_validity"]["fully_transparent"] is True
    assert result["hard_validity"]["pass"] is False


def test_exact_and_alpha_duplicate() -> None:
    a = _sprite()
    b = a.copy()
    batch = batch_metrics([_metadata("a"), _metadata("b")], [a, b])
    assert batch["exact_duplicate_rate"] == 0.5
    assert batch["alpha_mask_duplicate_rate"] == 0.5


def test_translated_duplicate() -> None:
    a = score_rgba(_sprite(), _metadata())["pixel_art"]
    b = score_rgba(_sprite(shift=2), _metadata("b"))["pixel_art"]
    assert a["alpha_sha256"] != b["alpha_sha256"]
    assert a["normalized_alpha_sha256"] == b["normalized_alpha_sha256"]


def test_recolor_geometry_near_duplicate() -> None:
    batch = batch_metrics([_metadata("a"), _metadata("b")], [_sprite(), _sprite((30, 90, 220))])
    assert batch["geometry_recolor_duplicate_rate"] == 1.0


def test_palette_overflow() -> None:
    rgba = _sprite()
    coords = np.argwhere(rgba[..., 3] > 0)
    for index, (y, x) in enumerate(coords[:40]):
        rgba[y, x, :3] = (index, index * 3 % 256, index * 7 % 256)
    assert score_rgba(rgba, _metadata())["pixel_art"]["unique_palette_size"] > 32


def test_antialiased_edge() -> None:
    rgba = _sprite()
    rgba[8, 10:20, 3] = 128
    assert score_rgba(rgba, _metadata())["pixel_art"]["antialiased_edge_ratio"] > 0


def test_clipped_silhouette() -> None:
    rgba = _sprite(shift=-10)
    assert score_rgba(rgba, _metadata())["pixel_art"]["border_clipping"] is True


def test_deterministic_score() -> None:
    assert score_rgba(_sprite(), _metadata()) == score_rgba(_sprite(), _metadata())


def test_training_neighbor_retrieval() -> None:
    target = TrainingImage("train-1", "d", "train.npz", 0, _sprite())
    nearest = retrieve_neighbors(_sprite(), [target])[0]
    assert nearest["exact_rgba"] is True
    assert nearest["sprite_id"] == "train-1"


def test_unscorable_conditional_label() -> None:
    result = score_conditions({"conditions": {"category": "weapon"}}, _sprite(), None)
    assert result["category"] == "unscorable"


def test_paired_seed_comparison(tmp_path: Path) -> None:
    base_run = _run(tmp_path / "base_run", _sprite())
    cand_run = _run(tmp_path / "cand_run", _sprite((30, 90, 220)))
    base_score = tmp_path / "base_score"
    cand_score = tmp_path / "cand_score"
    score_suite(base_run, base_score)
    score_suite(cand_run, cand_score)
    result = compare_reports(base_score, cand_score, tmp_path / "compare")
    assert result["paired_count"] == 1
    assert (tmp_path / "compare" / "paired_contact_sheet.html").is_file()


def test_paired_comparison_rejects_incompatible_metric_definitions_before_output(tmp_path: Path) -> None:
    base_run = _run(tmp_path / "base_run", _sprite())
    cand_run = _run(tmp_path / "cand_run", _sprite((30, 90, 220)))
    base_score = tmp_path / "base_score"
    cand_score = tmp_path / "cand_score"
    score_suite(base_run, base_score)
    score_suite(cand_run, cand_score)
    summary_path = cand_score / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["thresholds"]["near_train_pixel_distance"] = 999
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    comparison = tmp_path / "compare"

    with pytest.raises(IncompatibleMetricDefinitions, match="incompatible"):
        compare_reports(base_score, cand_score, comparison)

    assert not comparison.exists()


@pytest.mark.parametrize(
    "payload",
    (
        b'{"schema_version":"generation_benchmark_v1.0","schema_version":"generation_benchmark_v1.0"}',
        b'{"schema_version":"generation_benchmark_v1.0","thresholds":NaN}',
        b"[]",
        b'{"schema_version":"unsupported.v0"}',
    ),
)
def test_paired_comparison_strictly_rejects_malformed_report_before_output(
    tmp_path: Path,
    payload: bytes,
) -> None:
    base_run = _run(tmp_path / "base_run", _sprite())
    candidate_run = _run(tmp_path / "candidate_run", _sprite())
    base_score = tmp_path / "base_score"
    candidate_score = tmp_path / "candidate_score"
    score_suite(base_run, base_score)
    score_suite(candidate_run, candidate_score)
    (candidate_score / "summary.json").write_bytes(payload)
    output = tmp_path / "compare"

    with pytest.raises(ValueError):
        compare_reports(base_score, candidate_score, output)

    assert not output.exists()


def test_paired_comparison_rejects_preopen_report_substitution_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _run(tmp_path / "run", _sprite())
    score = tmp_path / "score"
    score_suite(run, score)
    summary_path = score / "summary.json"
    original_lstat = Path.lstat

    def substituted_lstat(path: Path):
        metadata = original_lstat(path)
        if path == summary_path:
            values = list(metadata)
            values[1] = int(metadata.st_ino) + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(Path, "lstat", substituted_lstat)

    with pytest.raises(ValueError, match="changed"):
        compare_reports(score, score, tmp_path / "compare")

    assert not (tmp_path / "compare").exists()


@pytest.mark.parametrize(
    "kind",
    ("duplicate", "nonfinite", "nonobject", "numeric_identity", "whitespace_identity", "collision"),
)
def test_paired_comparison_strictly_rejects_malformed_metric_rows_before_output(
    tmp_path: Path,
    kind: str,
) -> None:
    run = _run(tmp_path / "run", _sprite())
    score = tmp_path / "score"
    score_suite(run, score)
    row = {
        "prompt_id": "p0",
        "noise_seed": 10,
        "metrics": {
            "pixel_art": {
                "unique_palette_size": 2,
                "silhouette_occupancy": 0.2,
                "foreground_fragmentation": 0.1,
                "high_frequency_pixel_noise": 0.0,
            }
        },
    }
    if kind == "duplicate":
        payload = json.dumps(row).replace('"prompt_id": "p0"', '"prompt_id":"p0","prompt_id":"other"')
    elif kind == "nonfinite":
        payload = json.dumps(row).replace('"silhouette_occupancy": 0.2', '"silhouette_occupancy": NaN')
    elif kind == "nonobject":
        payload = "[]"
    elif kind == "numeric_identity":
        row["prompt_id"] = 7
        row["prompt"] = "fallback must not mask malformed prompt_id"
        payload = json.dumps(row)
    elif kind == "whitespace_identity":
        row["prompt_id"] = " padded "
        payload = json.dumps(row)
    else:
        payload = json.dumps(row) + "\n" + json.dumps(row)
    (score / "per_image_metrics.jsonl").write_text(payload + "\n", encoding="utf-8")
    output = tmp_path / "compare"

    with pytest.raises(ValueError):
        compare_reports(score, score, output)

    assert not output.exists()


def test_paired_comparison_rejects_oversize_metric_rows_before_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _run(tmp_path / "run", _sprite())
    score = tmp_path / "score"
    score_suite(run, score)
    monkeypatch.setattr(evaluation_suite_module, "_MAX_COMPARISON_JSONL_BYTES", 128)
    (score / "per_image_metrics.jsonl").write_bytes(b"x" * 129)
    output = tmp_path / "compare"

    with pytest.raises(ValueError, match="bounded"):
        compare_reports(score, score, output)

    assert not output.exists()


def test_paired_comparison_rejects_preopen_metric_row_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _run(tmp_path / "run", _sprite())
    score = tmp_path / "score"
    score_suite(run, score)
    metrics_path = score / "per_image_metrics.jsonl"
    original_lstat = Path.lstat

    def substituted_lstat(path: Path):
        metadata = original_lstat(path)
        if path == metrics_path:
            values = list(metadata)
            values[1] = int(metadata.st_ino) + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(Path, "lstat", substituted_lstat)
    output = tmp_path / "compare"

    with pytest.raises(ValueError, match="changed"):
        compare_reports(score, score, output)

    assert not output.exists()


def test_human_package_randomization_is_deterministic(tmp_path: Path) -> None:
    a = _run(tmp_path / "a", _sprite())
    b = _run(tmp_path / "b", _sprite((30, 90, 220)))
    human_package(a, b, tmp_path / "h1", seed=55)
    human_package(a, b, tmp_path / "h2", seed=55)
    key1 = (tmp_path / "h1" / "blind_key.jsonl").read_text(encoding="utf-8")
    key2 = (tmp_path / "h2" / "blind_key.jsonl").read_text(encoding="utf-8")
    assert '"hidden_left_source"' in key1
    assert key1.replace("h1", "h2") == key2


def test_cli_execution(tmp_path: Path) -> None:
    run = _run(tmp_path / "run", _sprite())
    out = tmp_path / "score"
    main(["score-suite", "--generated", str(run), "--out", str(out)])
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["summary"]["sample_count"] == 1
