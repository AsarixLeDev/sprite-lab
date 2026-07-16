from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from spritelab.training.cli import main as train_cli
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites


def _write_generated_dir(tmp_path: Path, specs: list[dict[str, Any]]) -> Path:
    out = tmp_path / "generated"
    out.mkdir()
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        sample_id = spec.get("sample_id", f"sample_{index:06d}")
        index_map = np.asarray(spec.get("index_map", np.zeros((32, 32), dtype=np.uint8)), dtype=np.uint8)
        palette = dict(spec.get("palette", {1: (255, 0, 0)}))
        raw_palette = dict(spec.get("raw_palette", palette))
        paths: dict[str, str] = {}

        if spec.get("include_raw", True):
            raw_rel = f"raw_rgba/{sample_id}.png"
            _write_rgba_from_indices(out / raw_rel, index_map, raw_palette)
            paths["raw_rgba"] = raw_rel
        if spec.get("include_hard", True):
            hard_rel = f"hard_rgba/{sample_id}.png"
            _write_rgba_from_indices(out / hard_rel, index_map, palette)
            paths["hard_rgba"] = hard_rel
        if spec.get("include_indexed", True):
            indexed_rel = f"indexed_png/{sample_id}.png"
            _write_indexed_png(out / indexed_rel, index_map, palette)
            paths["indexed_png"] = indexed_rel

        records.append(
            {
                "sample_id": sample_id,
                "prompt_id": spec.get("prompt_id", f"prompt_{index:04d}"),
                "prompt": spec.get("prompt", f"prompt {index}"),
                "category": spec.get("category", "seen_object"),
                "paths": paths,
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


def _write_rgba_from_indices(path: Path, index_map: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    for index, color in palette.items():
        mask = index_map == int(index)
        rgba[mask, :3] = np.asarray(color, dtype=np.uint8)
        rgba[mask, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)


def _write_indexed_png(path: Path, index_map: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(index_map.astype(np.uint8), mode="P")
    flat = [0, 0, 0] * 256
    for index, color in palette.items():
        offset = int(index) * 3
        flat[offset : offset + 3] = [int(color[0]), int(color[1]), int(color[2])]
    image.putpalette(flat)
    image.info["transparency"] = 0
    image.save(path)


def _review_one(tmp_path: Path, index_map: np.ndarray, **spec: Any) -> dict[str, Any]:
    generated = _write_generated_dir(tmp_path, [{**spec, "index_map": index_map}])
    result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=generated,
            out=generated / "generated_review_report.md",
            out_json=generated / "generated_review_report.json",
            out_dir=generated / "review",
            compare_raw_indexed=spec.get("compare_raw_indexed", False),
        )
    )
    assert result.ok
    return result.report["samples"][0]


def test_generated_review_command_loads_valid_generated_folder(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[8:16, 8:16] = 1
    generated = _write_generated_dir(tmp_path, [{"index_map": index_map}])

    train_cli(
        [
            "generated-review",
            "--generated",
            str(generated),
            "--out-dir",
            str(generated / "review"),
            "--group-by",
            "category",
        ]
    )

    assert (generated / "review" / "generated_review_report.json").is_file()
    assert (generated / "review" / "generated_review_report.md").is_file()
    assert (generated / "review" / "review_contact_sheet.png").is_file()


def test_generated_review_computes_alpha_coverage(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[2:4, 3:5] = 1
    metrics = _review_one(tmp_path, index_map)["metrics"]
    assert metrics["opaque_pixels"] == 4
    assert metrics["alpha_coverage"] == pytest.approx(4 / 1024)


def test_generated_review_computes_bounding_box(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[10:14, 5:9] = 1
    metrics = _review_one(tmp_path, index_map)["metrics"]
    assert metrics["bounding_box"] == {"x_min": 5, "y_min": 10, "x_max": 8, "y_max": 13}
    assert metrics["bbox_width"] == 4
    assert metrics["bbox_height"] == 4
    assert metrics["bbox_area"] == 16


def test_generated_review_computes_center_of_mass_and_offset(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[15:17, 15:17] = 1
    metrics = _review_one(tmp_path, index_map)["metrics"]
    assert metrics["center_of_mass_x"] == pytest.approx(15.5)
    assert metrics["center_of_mass_y"] == pytest.approx(15.5)
    assert metrics["center_offset_from_image_center"] == pytest.approx(0.0)


def test_generated_review_detects_border_touch(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[12:16, 0:2] = 1
    metrics = _review_one(tmp_path, index_map)["metrics"]
    assert metrics["touches_border"] is True


def test_generated_review_computes_connected_components(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[2, 2] = 1
    index_map[2, 12] = 1
    index_map[12, 2] = 1
    metrics = _review_one(tmp_path, index_map)["metrics"]
    assert metrics["connected_components"] == 3


def test_generated_review_computes_largest_component_ratio(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[2:4, 2:4] = 1
    index_map[10, 10] = 1
    metrics = _review_one(tmp_path, index_map)["metrics"]
    assert metrics["largest_component_pixels"] == 4
    assert metrics["largest_component_ratio"] == pytest.approx(4 / 5)


def test_generated_review_computes_visible_color_count(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[1, 1] = 1
    index_map[1, 2] = 2
    index_map[1, 3] = 3
    sample = _review_one(
        tmp_path,
        index_map,
        palette={1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)},
    )
    assert sample["metrics"]["visible_color_count"] == 3


def test_generated_review_computes_rare_color_count(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[1, 1] = 1
    index_map[2, 1:3] = 2
    index_map[3, 1:4] = 3
    sample = _review_one(
        tmp_path,
        index_map,
        palette={1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)},
    )
    assert sample["metrics"]["rare_color_count"] == 2


def test_generated_review_computes_raw_vs_indexed_mae(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[8:16, 8:16] = 1
    sample = _review_one(
        tmp_path,
        index_map,
        palette={1: (0, 0, 0)},
        raw_palette={1: (255, 0, 0)},
        compare_raw_indexed=True,
    )
    assert sample["metrics"]["raw_indexed_rgb_mae_visible"] == pytest.approx(1 / 3)
    assert "quantization_destructive" in sample["warnings"]


def test_generated_review_emits_empty_warning(tmp_path: Path) -> None:
    sample = _review_one(tmp_path, np.zeros((32, 32), dtype=np.uint8))
    assert "empty_or_nearly_empty" in sample["warnings"]


def test_generated_review_emits_too_full_warning(tmp_path: Path) -> None:
    sample = _review_one(tmp_path, np.ones((32, 32), dtype=np.uint8))
    assert "too_full_canvas" in sample["warnings"]


def test_generated_review_emits_fragmented_warning(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    for y, x in [(2, 2), (2, 8), (2, 14), (8, 2), (8, 8), (8, 14), (14, 2), (14, 8)]:
        index_map[y, x] = 1
    sample = _review_one(tmp_path, index_map)
    assert "fragmented" in sample["warnings"]


def test_generated_review_emits_too_few_colors_warning(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[8:16, 8:16] = 1
    sample = _review_one(tmp_path, index_map)
    assert "too_few_colors" in sample["warnings"]


def test_generated_review_groups_samples_by_category(tmp_path: Path) -> None:
    a = np.zeros((32, 32), dtype=np.uint8)
    b = np.zeros((32, 32), dtype=np.uint8)
    a[8:16, 8:16] = 1
    b[10:18, 10:18] = 1
    generated = _write_generated_dir(
        tmp_path,
        [
            {"index_map": a, "category": "seen_object"},
            {"index_map": b, "category": "creative_concept"},
        ],
    )
    result = review_generated_sprites(
        GeneratedReviewConfig(generated_dir=generated, out_dir=generated / "review", group_by="category")
    )
    assert result.report["groups"]["seen_object"]["count"] == 1
    assert result.report["groups"]["creative_concept"]["count"] == 1


def test_generated_review_writes_json_markdown_and_contact_sheet(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[8:16, 8:16] = 1
    generated = _write_generated_dir(tmp_path, [{"index_map": index_map}])
    result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=generated,
            out=generated / "generated_review_report.md",
            out_json=generated / "generated_review_report.json",
            out_dir=generated / "review",
            group_by="category",
        )
    )
    assert result.ok
    assert (generated / "generated_review_report.json").is_file()
    assert (generated / "generated_review_report.md").is_file()
    assert (generated / "review" / "review_contact_sheet.png").is_file()
    assert (generated / "review" / "review_contact_sheet_seen_object.png").is_file()


def test_generated_review_missing_optional_raw_warns_without_crashing(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[8:16, 8:16] = 1
    generated = _write_generated_dir(tmp_path, [{"index_map": index_map, "include_raw": False}])
    result = review_generated_sprites(
        GeneratedReviewConfig(generated_dir=generated, out_dir=generated / "review", compare_raw_indexed=True)
    )
    assert result.ok
    assert "raw_missing" in result.report["samples"][0]["warnings"]


def test_generated_review_missing_required_indexed_errors_in_strict_mode(tmp_path: Path) -> None:
    index_map = np.zeros((32, 32), dtype=np.uint8)
    index_map[8:16, 8:16] = 1
    generated = _write_generated_dir(tmp_path, [{"index_map": index_map, "include_indexed": False}])
    result = review_generated_sprites(
        GeneratedReviewConfig(generated_dir=generated, out_dir=generated / "review", strict=True)
    )
    assert not result.ok
    assert "indexed_missing" in result.report["samples"][0]["warnings"]
    assert any("indexed_png is required" in error for error in result.errors)
