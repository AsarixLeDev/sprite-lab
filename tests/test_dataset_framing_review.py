from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spritelab.training.cli import main as train_cli
from spritelab.training.dataset_framing_review import (
    DatasetFramingReviewConfig,
    review_dataset_framing,
)


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    specs = {
        "train": [
            ("train_center", "item_icon", "potion", _square(12, 12, 8)),
            ("train_border", "weapon", "sword", _square(0, 12, 6)),
        ],
        "val": [("val_center", "item_icon", "vial", _square(10, 10, 6))],
        "test": [("test_center", "material", "gem", _square(14, 14, 4))],
    }
    for split, rows in specs.items():
        records = []
        alpha = np.zeros((len(rows), 32, 32), dtype=np.uint8)
        index_map = np.zeros((len(rows), 32, 32), dtype=np.int16)
        role_map = np.zeros((len(rows), 32, 32), dtype=np.uint8)
        palette = np.zeros((len(rows), 33, 3), dtype=np.uint8)
        palette_mask = np.zeros((len(rows), 33), dtype=bool)
        category_id = np.zeros((len(rows),), dtype=np.int64)
        sprite_ids: list[str] = []
        for index, (sprite_id, category, base_object, mask) in enumerate(rows):
            alpha[index][mask] = 255
            index_map[index][mask] = 1
            palette[index, 1] = [255, 0, 0]
            palette_mask[index, :2] = True
            category_id[index] = index + 1
            sprite_ids.append(sprite_id)
            records.append(
                {
                    "sprite_id": sprite_id,
                    "split": split,
                    "category": category,
                    "object_name": base_object,
                    "semantic_v3": {"base_object": base_object, "category": category},
                    "source_path": f"{sprite_id}.png",
                }
            )
        (dataset / f"manifest_{split}.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        np.savez_compressed(
            dataset / f"{split}.npz",
            alpha=alpha,
            index_map=index_map,
            role_map=role_map,
            palette=palette,
            palette_mask=palette_mask,
            category_id=category_id,
            sprite_id=np.asarray(sprite_ids, dtype=np.str_),
        )
    return dataset


def _square(y: int, x: int, size: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[y : y + size, x : x + size] = True
    return mask


def _generated_review(tmp_path: Path) -> Path:
    generated = tmp_path / "generated"
    generated.mkdir()
    report = {
        "schema_version": "generated_review_v1.0",
        "sample_count": 2,
        "samples": [
            {
                "sample_id": "g0",
                "metrics": {
                    "touches_border": True,
                    "alpha_coverage": 0.5,
                    "bbox_width": 32,
                    "bbox_height": 20,
                    "center_offset_from_image_center": 1.0,
                    "visible_color_count": 16,
                },
            },
            {
                "sample_id": "g1",
                "metrics": {
                    "touches_border": True,
                    "alpha_coverage": 0.6,
                    "bbox_width": 32,
                    "bbox_height": 21,
                    "center_offset_from_image_center": 2.0,
                    "visible_color_count": 15,
                },
            },
        ],
    }
    (generated / "generated_review_report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    return generated


def test_tiny_exported_dataset_fixture_can_be_reviewed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = review_dataset_framing(DatasetFramingReviewConfig(dataset_dir=dataset, out_dir=dataset / "framing_review"))
    assert result.ok
    assert result.report["sample_count"] == 4


def test_dataset_framing_report_json_md_and_contact_sheet_are_written(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = review_dataset_framing(DatasetFramingReviewConfig(dataset_dir=dataset, out_dir=dataset / "framing_review"))
    assert result.json_path.is_file()
    assert result.markdown_path.is_file()
    assert (dataset / "framing_review" / "framing_contact_sheet.png").is_file()


def test_dataset_framing_split_grouping_works(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = review_dataset_framing(DatasetFramingReviewConfig(dataset_dir=dataset, out_dir=dataset / "framing_review"))
    assert result.report["groups"]["split"]["train"]["count"] == 2
    assert result.report["groups"]["split"]["val"]["count"] == 1
    assert result.report["groups"]["split"]["test"]["count"] == 1


def test_dataset_framing_category_grouping_works(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    result = review_dataset_framing(DatasetFramingReviewConfig(dataset_dir=dataset, out_dir=dataset / "framing_review"))
    assert result.report["groups"]["category"]["item_icon"]["count"] == 2
    assert result.report["groups"]["category"]["weapon"]["count"] == 1
    assert result.report["groups"]["category"]["material"]["count"] == 1


def test_dataset_framing_generated_comparison_works(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    generated = _generated_review(tmp_path)
    result = review_dataset_framing(
        DatasetFramingReviewConfig(
            dataset_dir=dataset,
            out_dir=dataset / "framing_review",
            compare_generated=generated,
        )
    )
    comparison = result.report["comparison"]
    assert comparison["source_count"] == 4
    assert comparison["generated_count"] == 2
    assert comparison["generated_border_touch_rate"] == 1.0
    assert comparison["diagnosis"] in {
        "generated_border_touch_rate_is_much_higher_than_source",
        "generated_border_touch_rate_is_somewhat_higher_than_source",
        "source_and_generated_are_both_border_heavy",
        "generated_border_touch_rate_matches_source_distribution",
    }


def test_dataset_framing_cli_runs(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    train_cli(
        [
            "dataset-framing-review",
            "--dataset",
            str(dataset),
            "--out-dir",
            str(dataset / "framing_review"),
        ]
    )
    assert (dataset / "framing_review" / "framing_review_report.json").is_file()
