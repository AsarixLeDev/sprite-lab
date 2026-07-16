from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.training.cli import main as train_cli
from spritelab.training.compare_generated_runs import CompareGeneratedRunsConfig, compare_generated_runs
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites


def _write_generated_dir(tmp_path: Path, name: str, specs: list[dict[str, Any]]) -> Path:
    out = tmp_path / name
    out.mkdir()
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        sample_id = spec.get("sample_id", f"sample_{index:06d}")
        prompt_id = spec.get("prompt_id", f"prompt_{index:04d}")
        prompt = spec.get("prompt", f"prompt {index}")
        index_map = np.zeros((32, 32), dtype=np.uint8)
        x0, y0, x1, y1 = spec.get("rect", (8, 8, 16, 16))
        index_map[y0:y1, x0:x1] = 1
        color = spec.get("color", (255, 0, 0))
        paths = {
            "raw_rgba": f"raw_rgba/{sample_id}.png",
            "hard_rgba": f"hard_rgba/{sample_id}.png",
            "indexed_png": f"indexed_png/{sample_id}.png",
        }
        _write_rgba(out / paths["raw_rgba"], index_map, color)
        _write_rgba(out / paths["hard_rgba"], index_map, color)
        _write_indexed(out / paths["indexed_png"], index_map, color)
        records.append(
            {
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "category": spec.get("category", "seen_object"),
                "checkpoint": str(tmp_path / "checkpoint.pt"),
                "visible_color_count": 1,
                "alpha_opaque_count": int(np.count_nonzero(index_map)),
                "max_colors": 8,
                "paths": paths,
            }
        )
    (out / "generated_manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (out / "generation_report.json").write_text(
        json.dumps({"sample_count": len(records), "manifest": "generated_manifest.jsonl", "contact_sheet": ""}) + "\n",
        encoding="utf-8",
    )
    (out / "generated_qa_report.json").write_text(
        json.dumps({"ok": True, "errors": [], "warnings": [], "sample_count": len(records)}) + "\n",
        encoding="utf-8",
    )
    return out


def _write_rgba(path: Path, index_map: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    mask = index_map == 1
    rgba[mask, :3] = np.asarray(color, dtype=np.uint8)
    rgba[mask, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)


def _write_indexed(path: Path, index_map: np.ndarray, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(index_map.astype(np.uint8), mode="P")
    palette = [0, 0, 0] * 256
    palette[3:6] = [int(color[0]), int(color[1]), int(color[2])]
    image.putpalette(palette)
    image.info["transparency"] = 0
    image.save(path)


def _write_review(generated: Path) -> None:
    review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=generated,
            out=generated / "generated_review_report.md",
            out_json=generated / "generated_review_report.json",
            out_dir=generated / "review",
            group_by="category",
            compare_raw_indexed=True,
        )
    )


def test_compare_generated_runs_uses_existing_reviews_and_writes_reports(tmp_path: Path) -> None:
    a = _write_generated_dir(
        tmp_path,
        "a",
        [{"sample_id": "s0", "prompt_id": "p0", "prompt": "red potion", "rect": (0, 8, 8, 16)}],
    )
    b = _write_generated_dir(
        tmp_path,
        "b",
        [{"sample_id": "s0", "prompt_id": "p0", "prompt": "red potion", "rect": (8, 8, 16, 16), "color": (0, 0, 255)}],
    )
    _write_review(a)
    _write_review(b)

    out = tmp_path / "compare"
    report = compare_generated_runs(CompareGeneratedRunsConfig(a=a, b=b, out_dir=out))
    assert report["a"]["review_source"] == str(a / "generated_review_report.json")
    assert report["b"]["review_source"] == str(b / "generated_review_report.json")
    assert report["deltas"]["border_touch_rate"] < 0.0
    assert report["matched_image_count"] == 1
    assert report["matched_images"][0]["metrics"]["rgb_histogram_distance"] > 0.0
    assert (out / "compare_report.json").is_file()
    assert (out / "compare_report.md").is_file()
    assert (out / "compare_contact_sheet.png").is_file()


def test_compare_generated_runs_computes_review_when_missing(tmp_path: Path) -> None:
    a = _write_generated_dir(tmp_path, "a", [{"sample_id": "s0", "prompt_id": "p0"}])
    b = _write_generated_dir(tmp_path, "b", [{"sample_id": "s0", "prompt_id": "p0", "rect": (10, 10, 18, 18)}])
    out = tmp_path / "compare_missing_review"
    report = compare_generated_runs(CompareGeneratedRunsConfig(a=a, b=b, out_dir=out))
    assert "derived_reviews" in report["a"]["review_source"]
    assert "derived_reviews" in report["b"]["review_source"]
    assert report["a"]["generated_qa"]["ok"] is True
    assert report["matched_image_count"] == 1


def test_compare_generated_runs_cli_runs(tmp_path: Path) -> None:
    a = _write_generated_dir(tmp_path, "a", [{"sample_id": "s0", "prompt_id": "p0"}])
    b = _write_generated_dir(tmp_path, "b", [{"sample_id": "s0", "prompt_id": "p0", "rect": (10, 10, 18, 18)}])
    out = tmp_path / "cli_compare"
    train_cli(["compare-generated-runs", "--a", str(a), "--b", str(b), "--out", str(out)])
    assert (out / "compare_report.json").is_file()
    assert (out / "compare_report.md").is_file()
    assert (out / "compare_contact_sheet.png").is_file()
