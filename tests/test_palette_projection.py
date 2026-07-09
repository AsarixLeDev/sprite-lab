from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.training.cli import main as train_cli
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites
from spritelab.training.palette_projection import (
    PaletteProjectionConfig,
    project_generated_palette,
    project_rgba_array,
)
from spritelab.training.prompt_faithfulness import PromptFaithfulnessConfig, run_prompt_faithfulness


def _gradient_rgba(color_count: int = 24) -> np.ndarray:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    for index in range(color_count):
        y = 4 + index // 8
        x = 4 + index % 8
        rgba[y : y + 3, x : x + 3, :3] = [
            (index * 37) % 256,
            (index * 73) % 256,
            (index * 109) % 256,
        ]
        rgba[y : y + 3, x : x + 3, 3] = 255
    return rgba


def _write_generated_dir(tmp_path: Path, specs: list[dict[str, Any]]) -> Path:
    out = tmp_path / "generated"
    out.mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        sample_id = spec.get("sample_id", f"sample_{index:06d}")
        rgba = np.asarray(spec.get("rgba", _gradient_rgba()), dtype=np.uint8)
        paths: dict[str, str] = {}
        for key in ("raw_rgba", "hard_rgba", "indexed_png"):
            rel = f"{key}/{sample_id}.png"
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, mode="RGBA").save(path)
            paths[key] = rel
        visible = rgba[..., 3] >= 128
        visible_colors = int(np.unique(rgba[..., :3][visible], axis=0).shape[0]) if bool(np.any(visible)) else 0
        records.append(
            {
                "sample_id": sample_id,
                "prompt_id": spec.get("prompt_id", f"prompt_{index:04d}"),
                "prompt": spec.get("prompt", f"red potion {index}"),
                "category": spec.get("category", "seen_object"),
                "checkpoint": str(checkpoint),
                "max_colors": spec.get("max_colors", 32),
                "visible_color_count": visible_colors,
                "alpha_opaque_count": int(np.count_nonzero(rgba[..., 3] >= 128)),
                "paths": paths,
                "target_semantics": spec.get(
                    "target_semantics", {"base_object": "potion", "attributes": {"colors": ["red"]}}
                ),
            }
        )
    (out / "generated_manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (out / "generation_report.json").write_text(
        json.dumps({"sample_count": len(records), "manifest": "generated_manifest.jsonl"}) + "\n",
        encoding="utf-8",
    )
    return out


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    alpha = np.zeros((1, 32, 32), dtype=np.uint8)
    alpha[:, 8:24, 8:24] = 255
    index_map = np.zeros((1, 32, 32), dtype=np.int16)
    index_map[:, 8:24, 8:24] = 1
    palette = np.zeros((1, 33, 3), dtype=np.uint8)
    palette[0, 1] = [255, 0, 0]
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index_map,
        role_map=np.zeros_like(index_map, dtype=np.uint8),
        palette=palette,
        palette_mask=np.ones((1, 33), dtype=bool),
        category_id=np.ones((1,), dtype=np.int64),
        sprite_id=np.array(["red_potion_src"], dtype=np.str_),
    )
    row = {
        "sprite_id": "red_potion_src",
        "split": "train",
        "npz_file": "train.npz",
        "npz_row": 0,
        "object_name": "red_potion",
        "base_object": "potion",
        "category": "item_icon",
    }
    (dataset / "training_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return dataset


def _prompts(path: Path) -> Path:
    row = {
        "prompt_id": "red_potion",
        "prompt": "red potion",
        "category": "seen_object",
        "target_semantics": {"base_object": "potion", "attributes": {"colors": ["red"]}},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_projection_preserves_alpha_exactly() -> None:
    rgba = _gradient_rgba(12)
    rgba[0, 0] = [3, 4, 5, 0]
    rgba[0, 1] = [6, 7, 8, 127]
    rgba[0, 2] = [9, 10, 11, 128]
    result = project_rgba_array(rgba, target_colors=4, min_pixel_share=0.01, alpha_threshold=0.5)
    assert np.array_equal(result.rgba[..., 3], rgba[..., 3])
    assert result.metrics["alpha_changed_pixels"] == 0


def test_projection_visible_color_count_decreases_to_target() -> None:
    result = project_rgba_array(_gradient_rgba(24), target_colors=4, min_pixel_share=0.0, alpha_threshold=0.5)
    assert result.metrics["visible_color_count_before"] > 4
    assert result.metrics["visible_color_count_after"] <= 4


def test_projection_ignores_transparent_pixels() -> None:
    rgba = _gradient_rgba(8)
    rgba[0, 0] = [201, 202, 203, 0]
    result = project_rgba_array(rgba, target_colors=2, min_pixel_share=0.0, alpha_threshold=0.5)
    assert np.array_equal(result.rgba[0, 0], rgba[0, 0])


def test_projection_is_deterministic_for_same_input() -> None:
    rgba = _gradient_rgba(20)
    first = project_rgba_array(rgba, target_colors=6, min_pixel_share=0.01, alpha_threshold=0.5)
    second = project_rgba_array(rgba, target_colors=6, min_pixel_share=0.01, alpha_threshold=0.5)
    assert np.array_equal(first.rgba, second.rgba)
    assert first.metrics == second.metrics


def test_projection_merges_tiny_clusters() -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[4:20, 4:20, :3] = [255, 0, 0]
    rgba[4:20, 4:20, 3] = 255
    rgba[5, 5, :3] = [0, 0, 255]
    result = project_rgba_array(rgba, target_colors=16, min_pixel_share=0.05, alpha_threshold=0.5)
    assert result.metrics["visible_color_count_before"] == 2
    assert result.metrics["visible_color_count_after"] == 1
    assert result.metrics["tiny_cluster_merge_count"] == 1


def test_project_generated_palette_writes_report_schema_and_contact_sheets(tmp_path: Path) -> None:
    generated = _write_generated_dir(tmp_path, [{"sample_id": "sample_000000", "prompt_id": "red_potion"}])
    out = tmp_path / "projected"
    report = project_generated_palette(
        PaletteProjectionConfig(
            generated=generated,
            out=out,
            target_colors=4,
            min_pixel_share=0.01,
            alpha_threshold=0.5,
        )
    )
    assert report["schema_version"] == "palette_projection_v1.0"
    assert report["sample_count"] == 1
    assert report["median_visible_color_count_after"] <= 4
    assert {"safe_count", "moderate_count", "destructive_count"} <= set(report)
    assert (out / "palette_projection_report.json").is_file()
    assert (out / "palette_projection_report.md").is_file()
    assert (out / "palette_projection_samples.jsonl").is_file()
    assert (out / "palette_projection_before_contact_sheet.png").is_file()
    assert (out / "palette_projection_after_contact_sheet.png").is_file()
    sample = json.loads((out / "palette_projection_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sample["visible_color_count_after"] <= 4
    assert sample["alpha_changed_pixels"] == 0


def test_project_generated_palette_cli_runs(tmp_path: Path) -> None:
    generated = _write_generated_dir(tmp_path, [{"sample_id": "sample_000000", "prompt_id": "red_potion"}])
    out = tmp_path / "cli_projected"
    train_cli(
        [
            "project-generated-palette",
            "--generated",
            str(generated),
            "--out",
            str(out),
            "--target-colors",
            "4",
            "--min-pixel-share",
            "0.01",
            "--alpha-threshold",
            "0.5",
            "--method",
            "deterministic_kmeans",
        ]
    )
    assert (out / "generated_manifest.jsonl").is_file()
    assert (out / "palette_projection_report.json").is_file()


def test_generated_review_runs_on_projected_directory(tmp_path: Path) -> None:
    generated = _write_generated_dir(tmp_path, [{"sample_id": "sample_000000", "prompt_id": "red_potion"}])
    out = tmp_path / "projected"
    project_generated_palette(PaletteProjectionConfig(generated=generated, out=out, target_colors=4))
    result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=out,
            out=out / "generated_review_report.md",
            out_json=out / "generated_review_report.json",
            out_dir=out / "review",
            compare_raw_indexed=True,
            strict=True,
        )
    )
    assert result.ok
    assert result.report["sample_count"] == 1


def test_prompt_faithfulness_runs_on_projected_directory(tmp_path: Path) -> None:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24, :3] = [255, 0, 0]
    rgba[8:24, 8:24, 3] = 255
    generated = _write_generated_dir(
        tmp_path,
        [{"sample_id": "sample_000000", "prompt_id": "red_potion", "prompt": "red potion", "rgba": rgba}],
    )
    out = tmp_path / "projected"
    project_generated_palette(PaletteProjectionConfig(generated=generated, out=out, target_colors=4))
    report = run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=out,
            prompts=_prompts(tmp_path / "prompts.jsonl"),
            dataset=_dataset(tmp_path),
            out=out / "prompt_faithfulness_report.md",
            out_json=out / "prompt_faithfulness_report.json",
            max_sources=0,
        )
    )
    assert report["sample_count"] == 1
    assert (out / "prompt_faithfulness_report.json").is_file()
