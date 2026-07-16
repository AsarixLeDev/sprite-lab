from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.evaluation.cli import main
from spritelab.evaluation.conditional import score_conditions
from spritelab.evaluation.memorization import TrainingImage, retrieve_neighbors
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
