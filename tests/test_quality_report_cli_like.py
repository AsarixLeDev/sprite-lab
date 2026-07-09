from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, save_manifest
from spritelab.data.quality_report import QualityReportOptions, create_quality_report


def test_quality_report_filters_and_max_items(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    bundle_a = _write_bundle(dataset_dir / "bundles" / "train_item", "train_item")
    bundle_b = _write_bundle(dataset_dir / "bundles" / "test_item", "test_item")
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                _record("train_item", bundle_a, category="item_icon", split="train"),
                _record("test_item", bundle_b, category="item_icon", split="test"),
            ],
            rejected_count=0,
            total_seen=2,
            options={},
        ),
        dataset_dir / "manifest.json",
    )

    report = create_quality_report(
        QualityReportOptions(
            dataset_path=dataset_dir,
            output_dir=tmp_path / "quality",
            filter_split="train",
            max_items=1,
            include_markdown=False,
            include_json=False,
            write_flag_files=False,
        )
    )

    assert report.summary.total_records == 1
    assert [record.id for record in report.records] == ["train_item"]
    assert not (tmp_path / "quality" / "quality_report.json").exists()
    assert not (tmp_path / "quality" / "quality_report.md").exists()


def test_quality_report_fail_on_load_error(tmp_path: Path) -> None:
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
                    palette_size=2,
                    sha256="hash",
                )
            ],
            rejected_count=0,
            total_seen=1,
            options={},
        ),
        dataset_dir / "manifest.json",
    )

    with pytest.raises(Exception):  # noqa: B017
        create_quality_report(
            QualityReportOptions(
                dataset_path=dataset_dir,
                output_dir=tmp_path / "quality",
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
        palette_size=2,
        sha256=sprite_id,
        split=split,
    )


def _write_bundle(bundle_dir: Path, sprite_id: str) -> Path:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    alpha[15, 15] = 1
    index_map[15, 15] = 1
    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id),
    )
    save_bundle(bundle, bundle_dir)
    return bundle_dir
