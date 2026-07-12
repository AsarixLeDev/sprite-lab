from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.evaluation.memorization_review import (
    ReviewPair,
    append_review,
    initialize_review,
    load_latest_reviews,
    load_review_pairs,
    resume_index,
)


def _fixture_report(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir()
    alpha = np.zeros((1, 32, 32), dtype=np.uint8)
    alpha[:, 8:20, 10:22] = 1
    index_map = np.zeros((1, 32, 32), dtype=np.uint8)
    palette = np.array([[[20, 40, 60]]], dtype=np.uint8)
    palette_mask = np.ones((1, 1), dtype=bool)
    np.savez(dataset / "train.npz", alpha=alpha, index_map=index_map, palette=palette, palette_mask=palette_mask)
    manifest_row = {
        "sprite_id": "train_sprite",
        "split": "train",
        "npz_file": "train.npz",
        "npz_row": 0,
        "schema_version": "training_manifest_v1.0",
        "source": {"dataset_dir": str(dataset), "manifest_file": "source.jsonl", "manifest_row": 17},
    }
    manifest = dataset / "training_manifest.jsonl"
    manifest.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
    generated = np.zeros((32, 32, 4), dtype=np.uint8)
    generated[8:20, 10:22] = (200, 100, 10, 255)
    image = root / "generated.png"
    Image.fromarray(generated, "RGBA").save(image)
    report = root / "report"
    report.mkdir()
    summary = {"schema_version": "generation_benchmark_v1.0", "training_manifests": [str(manifest)]}
    (report / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    metric = {
        "sample_id": "sample_0",
        "prompt": "orange block",
        "seed": 3,
        "noise_seed": 30,
        "checkpoint": "checkpoint.pt",
        "run": str(root),
        "image": str(image),
        "suspicious_memorization": "exact_alpha",
        "training_neighbors": [
            {
                "sprite_id": "train_sprite",
                "dataset": str(dataset),
                "npz_file": "train.npz",
                "npz_row": 0,
                "exact_rgba": False,
                "exact_alpha": True,
                "translated_duplicate": False,
                "pixel_distance": 0.01,
                "perceptual_distance": 0.02,
                "geometry_iou": 1.0,
            }
        ],
    }
    (report / "per_image_metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
    return report


def test_loading_exact_alpha_pair_and_provenance(tmp_path: Path) -> None:
    pairs = load_review_pairs(_fixture_report(tmp_path), project_root=tmp_path)
    assert len(pairs) == 1
    assert pairs[0].nearest["exact_alpha"] is True
    assert pairs[0].training_provenance["source"]["manifest_row"] == 17
    assert np.array_equal(pairs[0].generated_rgba[..., 3], pairs[0].training_rgba[..., 3])


def test_saving_is_append_only_and_latest_revision_wins(tmp_path: Path) -> None:
    pair = load_review_pairs(_fixture_report(tmp_path), project_root=tmp_path)[0]
    out = tmp_path / "review"
    append_review(
        out,
        pair,
        classification="uncertain",
        notes="first",
        block_promotion=False,
        rule_needs_review=True,
        current_index=0,
        pair_count=1,
    )
    append_review(
        out,
        pair,
        classification="same_silhouette_different_render",
        notes="revised",
        block_promotion=False,
        rule_needs_review=False,
        current_index=0,
        pair_count=1,
    )
    assert len((out / "review_results.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    latest = load_latest_reviews(out / "review_results.jsonl")
    assert latest[pair.pair_id]["revision"] == 2
    assert latest[pair.pair_id]["notes"] == "revised"


def test_resume_starts_at_first_unreviewed() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    pairs = [ReviewPair(str(i), {"sample_id": str(i)}, {}, {}, rgba, rgba) for i in range(3)]
    assert resume_index(pairs, {"0": {}}) == 1
    assert resume_index(pairs, {"0": {}, "1": {}, "2": {}}) == 2


def test_initialize_persists_resumable_state(tmp_path: Path) -> None:
    pair = load_review_pairs(_fixture_report(tmp_path), project_root=tmp_path)[0]
    out = tmp_path / "review"
    assert initialize_review(out, [pair]) == 0
    state = json.loads((out / "review_state.json").read_text(encoding="utf-8"))
    assert state == {
        "schema_version": "memorization_review_v1.0",
        "current_index": 0,
        "pair_count": 1,
        "completed_pair_ids": [],
        "completed_count": 0,
    }
    assert (out / "review_summary.json").is_file()


def test_summary_counts_latest_decisions(tmp_path: Path) -> None:
    pair = load_review_pairs(_fixture_report(tmp_path), project_root=tmp_path)[0]
    out = tmp_path / "review"
    append_review(
        out,
        pair,
        classification="likely_false_positive",
        notes="generic",
        block_promotion=False,
        rule_needs_review=True,
        current_index=0,
        pair_count=2,
    )
    summary = json.loads((out / "review_summary.json").read_text(encoding="utf-8"))
    assert summary["reviewed_count"] == 1
    assert summary["remaining_count"] == 1
    assert summary["classification_counts"]["likely_false_positive"] == 1
    assert summary["classification_counts"]["same_sprite_or_memorized"] == 0
    assert summary["threshold_or_rule_review_count"] == 1
    assert (out / "review_summary.md").is_file()
