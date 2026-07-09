from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from _semantic_dataset import default_specs, make_semantic_dataset
from spritelab.dataset_maker.training_manifest import build_training_manifest, write_training_manifest
from spritelab.training.data import SpriteTrainingDataset
from spritelab.training.overfit_subset import make_overfit_subset, select_overfit_subset


def _dataset_with_manifest(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    dataset = make_semantic_dataset(tmp_path / "ds", default_specs())
    result = build_training_manifest(dataset, variants_per_sprite=2, caption_policy="mixed", seed=11)
    manifest = dataset / "training_manifest.jsonl"
    write_training_manifest(manifest, result.rows)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    return dataset, manifest, rows


def test_deterministic_subset_with_seed() -> None:
    rows = [
        {"sprite_id": f"s{i}", "split": "train", "category": "a" if i % 2 else "b", "object_name": f"obj{i}"}
        for i in range(8)
    ]
    first = select_overfit_subset(rows, count=4, split="train", seed=77, stratify="category")
    second = select_overfit_subset(rows, count=4, split="train", seed=77, stratify="category")
    assert first.sprite_ids == second.sprite_ids
    assert first.to_report()["selected_sprite_count"] == 4
    assert first.to_report()["selected_row_count"] == 4


def test_max_train_sprites_filters_rows_consistently(tmp_path: Path) -> None:
    dataset, manifest, rows = _dataset_with_manifest(tmp_path)
    selection = select_overfit_subset(rows, count=2, split="train", seed=123)
    ds = SpriteTrainingDataset(dataset, manifest, split="train", sprite_ids=selection.sprite_ids)
    assert {sample["sprite_id"] for sample in (ds[i] for i in range(len(ds)))} == set(selection.sprite_ids)
    assert len(ds) == selection.to_report()["selected_row_count"]


def test_sprite_id_list_filters_exact_ids_and_reports_categories(tmp_path: Path) -> None:
    _dataset, manifest, rows = _dataset_with_manifest(tmp_path)
    wanted = ["t_red_potion", "t_gold_sword"]
    selection = select_overfit_subset(rows, sprite_ids=wanted, split="train", seed=5)
    report = selection.to_report()
    assert selection.sprite_ids == tuple(wanted)
    assert set(report["categories"])
    assert report["object_names"]["t_red_potion"]

    out = tmp_path / "subset_ids.txt"
    written = make_overfit_subset(
        dataset=tmp_path / "ds",
        training_manifest=manifest,
        out=out,
        count=2,
        seed=5,
        stratify="category",
    )
    assert out.is_file()
    assert written["selected_sprite_count"] == 2
    assert "categories" in written


def test_read_sprite_id_list_accepts_persisted_json_report(tmp_path: Path) -> None:
    from spritelab.training.overfit_subset import read_sprite_id_list

    path = tmp_path / "overfit_sprite_ids.json"
    path.write_text(json.dumps({"sprite_ids": ["s0", "s1"]}), encoding="utf-8")
    assert read_sprite_id_list(path) == ["s0", "s1"]
