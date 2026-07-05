from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.harvest.cli import main as harvest_main


def _bundle_arrays(sprite_ids: list[str]) -> dict[str, np.ndarray]:
    count = len(sprite_ids)
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    palette = np.zeros((count, 33, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, 33), dtype=bool)
    for row in range(count):
        alpha[row, 10:14, 10:14] = 1
        index_map[row, 10:14, 10:14] = 1
        palette[row, 1] = [10, 20, 30]
        palette_mask[row, 0] = True
        palette_mask[row, 1] = True
    return {
        "alpha": alpha,
        "index_map": index_map,
        "role_map": np.zeros((count, 32, 32), dtype=np.uint8),
        "palette": palette,
        "palette_mask": palette_mask,
        "category_id": np.zeros((count,), dtype=np.int64),
        "sprite_id": np.array(sprite_ids, dtype=np.str_),
    }


def _record(sprite_id: str, split: str, **overrides) -> dict:
    record = {
        "sprite_id": sprite_id,
        "split": split,
        "category": "item_icon",
        "object_name": "potion",
        "tags": ["potion", "glass"],
        "source_name": "test-source",
        "source_path": f"data/{sprite_id}.png",
        "license": "cc0",
        "label_v2": {"applied": True, "bucket": "auto_test", "flags": []},
    }
    record.update(overrides)
    return record


def _make_dataset(tmp_path: Path, records_by_split: dict[str, list[dict]]) -> Path:
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        recs = records_by_split.get(split, [])
        (dataset_dir / f"manifest_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in recs) + ("\n" if recs else ""),
            encoding="utf-8",
        )
        np.savez_compressed(
            dataset_dir / f"{split}.npz", **_bundle_arrays([str(r["sprite_id"]) for r in recs])
        )
    (dataset_dir / "dataset_config.json").write_text(
        json.dumps({"dataset_name": "ds", "max_palette_slots": 32}), encoding="utf-8"
    )
    return dataset_dir


def _valid(tmp_path: Path) -> Path:
    return _make_dataset(
        tmp_path,
        {
            "train": [_record(f"s{i}", "train") for i in range(8)],
            "val": [_record("v0", "val")],
            "test": [_record("t0", "test")],
        },
    )


def test_cli_exits_zero_and_writes_reports(tmp_path: Path, capsys) -> None:
    dataset_dir = _valid(tmp_path)
    harvest_main(["dataset-qa", "--dataset", str(dataset_dir)])
    out = capsys.readouterr().out
    assert "Records: 10" in out
    assert "Errors: 0" in out
    assert (dataset_dir / "dataset_qa_report.json").exists()
    assert (dataset_dir / "dataset_qa_report.md").exists()
    assert (dataset_dir / "dataset_qa_contact_sheet.png").exists()


def test_cli_exits_nonzero_on_errors(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(
        tmp_path,
        {
            "train": [_record("s0", "train", object_name="")],
            "val": [_record("v0", "val")],
            "test": [_record("t0", "test")],
        },
    )
    with pytest.raises(SystemExit) as excinfo:
        harvest_main(["dataset-qa", "--dataset", str(dataset_dir), "--no-contact-sheet"])
    assert excinfo.value.code == 1


def test_cli_exits_zero_with_warnings_by_default(tmp_path: Path) -> None:
    # A single all-transparent sprite triggers a warning (non-strict) but not an error.
    dataset_dir = _valid(tmp_path)
    arrays = _bundle_arrays([f"s{i}" for i in range(8)])
    arrays["alpha"][0] = 0
    arrays["index_map"][0] = 0
    np.savez_compressed(dataset_dir / "train.npz", **arrays)

    harvest_main(["dataset-qa", "--dataset", str(dataset_dir), "--no-contact-sheet"])
    result_json = json.loads((dataset_dir / "dataset_qa_report.json").read_text(encoding="utf-8"))
    assert result_json["warnings"]
    assert not result_json["errors"]


def test_cli_fail_on_warning_exits_nonzero(tmp_path: Path) -> None:
    dataset_dir = _valid(tmp_path)
    arrays = _bundle_arrays([f"s{i}" for i in range(8)])
    arrays["alpha"][0] = 0
    arrays["index_map"][0] = 0
    np.savez_compressed(dataset_dir / "train.npz", **arrays)

    with pytest.raises(SystemExit) as excinfo:
        harvest_main(
            ["dataset-qa", "--dataset", str(dataset_dir), "--no-contact-sheet", "--fail-on-warning"]
        )
    assert excinfo.value.code == 1


def test_cli_review_queue_overlap_exits_nonzero(tmp_path: Path) -> None:
    dataset_dir = _valid(tmp_path)
    queue = tmp_path / "review.jsonl"
    queue.write_text(json.dumps({"sprite_id": "s2"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        harvest_main(
            [
                "dataset-qa",
                "--dataset",
                str(dataset_dir),
                "--review-queue",
                str(queue),
                "--no-contact-sheet",
            ]
        )
    assert excinfo.value.code == 1
