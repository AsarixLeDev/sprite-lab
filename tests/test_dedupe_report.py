from __future__ import annotations

from pathlib import Path

import numpy as np

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.io import save_bundle
from spritelab.data.dedupe_report import (
    DECODED_RGBA_SHA256,
    DedupeReportOptions,
    create_dedupe_report,
    load_dedupe_report_json,
    render_dedupe_report_markdown,
)
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, save_manifest


def test_create_dedupe_report_writes_outputs_and_detects_groups(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    bundle_a = _write_square_bundle(dataset_dir / "bundles" / "red_a", "red_a", (220, 30, 30))
    bundle_b = _write_square_bundle(dataset_dir / "bundles" / "red_b_duplicate", "red_b_duplicate", (220, 30, 30))
    bundle_c = _write_square_bundle(
        dataset_dir / "bundles" / "red_c_near_duplicate",
        "red_c_near_duplicate",
        (220, 30, 30),
        extra_pixel=(23, 23),
    )
    bundle_d = _write_square_bundle(dataset_dir / "bundles" / "blue_unique", "blue_unique", (20, 40, 230), offset=2)
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                _record("red_a", bundle_a, source_path="raw/red.png", sha256="same-source", split="train"),
                _record("red_b_duplicate", bundle_b, source_path="raw/red.png", sha256="same-source", split="val"),
                _record("red_c_near_duplicate", bundle_c, source_path="raw/red_c.png", sha256="near-source", split="train"),
                _record("blue_unique", bundle_d, source_path="raw/blue.png", sha256="blue-source", split="test"),
            ],
            rejected_count=0,
            total_seen=4,
            options={},
        ),
        dataset_dir / "manifest.json",
    )
    output_dir = tmp_path / "dedupe"

    report = create_dedupe_report(DedupeReportOptions(dataset_path=dataset_dir, output_dir=output_dir))

    assert report.summary.total_records == 4
    assert report.summary.analyzed_records == 4
    assert report.summary.failed_records == 0
    assert report.summary.exact_decoded_duplicate_groups >= 1
    assert report.summary.exact_bundle_duplicate_groups >= 1
    assert report.summary.exact_source_duplicate_groups >= 1
    assert report.summary.near_duplicate_groups >= 1
    assert report.summary.cross_split_exact_groups >= 1
    assert report.summary.duplicate_source_path_count == 1
    assert any(group.kind == DECODED_RGBA_SHA256 and group.crosses_splits for group in report.exact_groups)
    assert (output_dir / "dedupe_report.json").exists()
    assert (output_dir / "dedupe_report.md").exists()
    assert (output_dir / "duplicate_groups" / "exact_decoded_sprite_duplicates.txt").exists()
    assert (output_dir / "duplicate_groups" / "near_duplicates.txt").exists()

    loaded = load_dedupe_report_json(output_dir / "dedupe_report.json")
    assert loaded == report

    markdown = render_dedupe_report_markdown(report)
    assert "# Dataset Dedupe Report" in markdown
    assert "## Critical split leakage" in markdown
    assert "## Exact duplicate groups" in markdown
    assert "## Near-duplicate groups" in markdown


def test_dedupe_report_can_disable_near_duplicate_detection(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    bundle_a = _write_square_bundle(dataset_dir / "bundles" / "a", "a", (220, 30, 30))
    bundle_b = _write_square_bundle(dataset_dir / "bundles" / "b", "b", (220, 30, 30), extra_pixel=(23, 23))
    save_manifest(
        DatasetManifest(
            dataset_name="dataset",
            records=[
                _record("a", bundle_a, source_path="raw/a.png", sha256="a", split="train"),
                _record("b", bundle_b, source_path="raw/b.png", sha256="b", split="train"),
            ],
            rejected_count=0,
            total_seen=2,
            options={},
        ),
        dataset_dir / "manifest.json",
    )

    report = create_dedupe_report(
        DedupeReportOptions(
            dataset_path=dataset_dir,
            output_dir=tmp_path / "dedupe",
            near_duplicate=False,
        )
    )

    assert report.summary.near_duplicate_groups == 0
    assert report.near_groups == []
    assert all(record.average_hash is None for record in report.records)


def _record(
    sprite_id: str,
    bundle_dir: Path,
    *,
    source_path: str,
    sha256: str,
    split: str,
) -> IngestedSpriteRecord:
    return IngestedSpriteRecord(
        id=sprite_id,
        source_path=source_path,
        bundle_dir=str(bundle_dir),
        width=32,
        height=32,
        category="item_icon",
        subtype=None,
        license="CC0",
        palette_size=1,
        sha256=sha256,
        split=split,
    )


def _write_square_bundle(
    bundle_dir: Path,
    sprite_id: str,
    color: tuple[int, int, int],
    *,
    offset: int = 0,
    extra_pixel: tuple[int, int] | None = None,
) -> Path:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    index_map = np.zeros((32, 32), dtype=np.uint8)
    for y in range(10 + offset, 22 + offset):
        for x in range(10 + offset, 22 + offset):
            alpha[y, x] = 1
            index_map[y, x] = 1
    if extra_pixel is not None:
        x, y = extra_pixel
        alpha[y, x] = 1
        index_map[y, x] = 1
    bundle = SpriteBundle(
        alpha=alpha,
        palette=np.array([[0, 0, 0], list(color)], dtype=np.uint8),
        index_map=index_map,
        role_map=None,
        metadata=SpriteMetadata(id=sprite_id, category="item_icon", palette_size=1),
    )
    save_bundle(bundle, bundle_dir)
    return bundle_dir
