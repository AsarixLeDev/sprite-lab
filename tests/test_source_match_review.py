from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.training.generated_canonicalizer import (
    canonicalize_generated_rgba,
    write_generated_sprite_artifacts,
    write_generation_reports,
)
from spritelab.training.source_match_review import SourceMatchReviewConfig, run_source_match_review


def _rgba(color: tuple[float, float, float]) -> np.ndarray:
    arr = np.zeros((32, 32, 4), dtype=np.float32)
    arr[8:24, 8:24, :3] = np.array(color, dtype=np.float32)
    arr[8:24, 8:24, 3] = 1.0
    return arr


def _source_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    alpha = np.zeros((2, 32, 32), dtype=np.uint8)
    alpha[:, 8:24, 8:24] = 255
    index = np.zeros((2, 32, 32), dtype=np.int16)
    index[:, 8:24, 8:24] = 1
    palette = np.zeros((2, 33, 3), dtype=np.uint8)
    palette[0, 1] = [255, 0, 0]
    palette[1, 1] = [0, 0, 255]
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index,
        role_map=np.zeros_like(index, dtype=np.uint8),
        palette=palette,
        palette_mask=np.ones((2, 33), dtype=bool),
        category_id=np.ones((2,), dtype=np.int64),
        sprite_id=np.array(["red_source", "blue_source"], dtype=np.str_),
    )
    manifest = dataset / "training_manifest.jsonl"
    rows = [
        {
            "sprite_id": "red_source",
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": 0,
            "object_name": "red_potion",
            "category": "item_icon",
            "caption": "red potion",
        },
        {
            "sprite_id": "blue_source",
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": 1,
            "object_name": "blue_gem",
            "category": "material",
            "caption": "blue gem",
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return dataset, manifest


def _generated(tmp_path: Path) -> Path:
    out = tmp_path / "generated"
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_text("stub", encoding="utf-8")
    records = [
        write_generated_sprite_artifacts(
            canonicalize_generated_rgba(_rgba((1.0, 0.0, 0.0))),
            out,
            "copy_red",
            {"prompt_id": "p0", "prompt": "red potion", "checkpoint": str(ckpt), "source_sprite_id": "red_source"},
        ),
        write_generated_sprite_artifacts(
            canonicalize_generated_rgba(_rgba((0.0, 0.0, 1.0))),
            out,
            "wrong_blue",
            {"prompt_id": "p1", "prompt": "red potion", "checkpoint": str(ckpt), "source_sprite_id": "red_source"},
        ),
    ]
    write_generation_reports(out_dir=out, records=records, config={}, contact_sheet=None)
    return out


def test_source_match_scores_copy_better_than_different_sprite(tmp_path: Path) -> None:
    dataset, manifest = _source_dataset(tmp_path)
    generated = _generated(tmp_path)
    report = run_source_match_review(
        SourceMatchReviewConfig(
            generated=generated,
            dataset=dataset,
            training_manifest=manifest,
            out=tmp_path / "source_match",
        )
    )
    samples = {sample["sample_id"]: sample for sample in report["samples"]}
    assert samples["copy_red"]["metrics"]["visible_rgb_mae"] == 0.0
    assert samples["copy_red"]["metrics"]["alpha_iou"] == 1.0
    assert samples["wrong_blue"]["metrics"]["visible_rgb_mae"] > samples["copy_red"]["metrics"]["visible_rgb_mae"]
    assert report["near_match_rate"] < 1.0
    assert (tmp_path / "source_match" / "source_match_report.json").is_file()
    assert (tmp_path / "source_match" / "source_match_report.md").is_file()
    assert report["best_examples"]
    assert report["worst_examples"]
