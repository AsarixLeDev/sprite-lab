from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.data.dedupe_report import DedupeReportOptions, create_dedupe_report
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, save_manifest


def test_dedupe_report_filters_max_items_and_optional_writes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    train_bundle = _write_bundle(dataset_dir / "bundles" / "train_item", "train_item")
    test_bundle = _write_bundle(dataset_dir / "bundles" / "test_item", "test_item")
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                _record("train_item", train_bundle, category="item_icon", split="train"),
                _record("test_item", test_bundle, category="item_icon", split="test"),
            ],
            rejected_count=0,
            total_seen=2,
            options={},
        ),
        dataset_dir / "manifest.json",
    )
    output_dir = tmp_path / "dedupe"

    report = create_dedupe_report(
        DedupeReportOptions(
            dataset_path=dataset_dir,
            output_dir=output_dir,
            filter_split="train",
            max_items=1,
            include_json=False,
            include_markdown=False,
            write_group_files=False,
        )
    )

    assert report.summary.total_records == 1
    assert [record.id for record in report.records] == ["train_item"]
    assert not (output_dir / "dedupe_report.json").exists()
    assert not (output_dir / "dedupe_report.md").exists()
    assert not (output_dir / "duplicate_groups").exists()


def test_dedupe_report_detects_duplicate_ids(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    bundle_a = _write_bundle(dataset_dir / "bundles" / "duplicate_a", "duplicate_a")
    bundle_b = _write_bundle(dataset_dir / "bundles" / "duplicate_b", "duplicate_b")
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                _record("duplicate", bundle_a, category="item_icon", split="train"),
                _record("duplicate", bundle_b, category="item_icon", split="val"),
            ],
            rejected_count=0,
            total_seen=2,
            options={},
        ),
        dataset_dir / "manifest.json",
    )

    report = create_dedupe_report(DedupeReportOptions(dataset_path=dataset_dir, output_dir=tmp_path / "dedupe"))

    assert report.summary.duplicate_id_count == 1
    assert "duplicate" in report.duplicate_ids
    assert len(report.duplicate_ids["duplicate"]) == 2


def test_dedupe_report_records_load_errors_and_can_fail_fast(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                IngestedSpriteRecord(
                    id="broken",
                    source_path="raw/broken.png",
                    bundle_dir=str(dataset_dir / "bundles" / "broken"),
                    width=32,
                    height=32,
                    category=None,
                    subtype=None,
                    license=None,
                    palette_size=1,
                    sha256="broken",
                    split=None,
                )
            ],
            rejected_count=0,
            total_seen=1,
            options={},
        ),
        dataset_dir / "manifest.json",
    )

    report = create_dedupe_report(DedupeReportOptions(dataset_path=dataset_dir, output_dir=tmp_path / "dedupe"))

    assert report.summary.total_records == 1
    assert report.summary.analyzed_records == 0
    assert report.summary.failed_records == 1
    assert report.failed[0].id == "broken"

    with pytest.raises(Exception):  # noqa: B017
        create_dedupe_report(
            DedupeReportOptions(
                dataset_path=dataset_dir,
                output_dir=tmp_path / "dedupe_fast",
                fail_on_load_error=True,
            )
        )


def _record(sprite_id: str, bundle_dir: Path, *, category: str, split: str) -> IngestedSpriteRecord:
    return IngestedSpriteRecord(
        id=sprite_id,
        source_path=f"raw/{sprite_id}.png",
        bundle_dir=str(bundle_dir),
        width=32,
        height=32,
        category=category,
        subtype=None,
        license=None,
        palette_size=1,
        sha256=f"{sprite_id}-{bundle_dir.name}",
        split=split,
    )


def _write_bundle(bundle_dir: Path, sprite_id: str) -> Path:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[15, 15] = 1
    index_map[15, 15] = 1
    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], [180, 80, 40]], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id, category="item_icon", palette_size=1),
    )
    save_bundle(bundle, bundle_dir)
    return bundle_dir
