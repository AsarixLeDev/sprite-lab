from __future__ import annotations

from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, save_manifest
from spritelab.data.quality_report import (
    QualityReportOptions,
    create_quality_report,
    load_quality_report_json,
    render_quality_report_markdown,
)


def test_create_quality_report_writes_outputs_and_roundtrips(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    bundle_a = _write_bundle(dataset_dir / "bundles" / "mostly_empty", "mostly_empty", [(0, 0)])
    bundle_b = _write_bundle(dataset_dir / "bundles" / "edge", "edge", [(0, 0), (1, 0), (0, 1)])
    manifest = DatasetManifest(
        dataset_name="dataset",
        records=[
            _record("mostly_empty", bundle_a, "samehash"),
            _record("edge", bundle_b, "samehash"),
            IngestedSpriteRecord(
                id="broken",
                source_path="raw/broken.png",
                bundle_dir=str(dataset_dir / "bundles" / "broken"),
                width=32,
                height=32,
                category="item_icon",
                subtype=None,
                license="CC0",
                palette_size=2,
                sha256="brokenhash",
            ),
        ],
        rejected_count=0,
        total_seen=3,
        options={},
    )
    save_manifest(manifest, dataset_dir / "manifest.json")
    output_dir = tmp_path / "quality"

    report = create_quality_report(QualityReportOptions(dataset_path=dataset_dir, output_dir=output_dir))

    assert report.summary.total_records == 3
    assert report.summary.analyzed_records == 2
    assert report.summary.failed_records == 1
    assert "MOSTLY_EMPTY" in report.summary.issue_counts
    assert report.summary.duplicate_sha256_groups == {"samehash": ["edge", "mostly_empty"]}
    assert (output_dir / "quality_report.json").exists()
    assert (output_dir / "quality_report.md").exists()
    assert (output_dir / "flagged" / "MOSTLY_EMPTY.txt").exists()

    loaded = load_quality_report_json(output_dir / "quality_report.json")
    assert loaded == report

    markdown = render_quality_report_markdown(report)
    assert "# Dataset Quality Report" in markdown
    assert "## Issue counts" in markdown
    assert "## Failed records" in markdown


def _record(sprite_id: str, bundle_dir: Path, sha256: str) -> IngestedSpriteRecord:
    return IngestedSpriteRecord(
        id=sprite_id,
        source_path=f"raw/{sprite_id}.png",
        bundle_dir=str(bundle_dir),
        width=32,
        height=32,
        category="item_icon",
        subtype=None,
        license="CC0",
        palette_size=2,
        sha256=sha256,
    )


def _write_bundle(bundle_dir: Path, sprite_id: str, pixels: list[tuple[int, int]]) -> Path:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    for x, y in pixels:
        alpha[y, x] = 1
        index_map[y, x] = 1
    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], [40, 40, 40]], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id, category="item_icon", palette_size=1),
    )
    save_bundle(bundle, bundle_dir)
    return bundle_dir
