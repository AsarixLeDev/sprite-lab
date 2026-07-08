"""Demonstrate duplicate and near-duplicate dataset reporting."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spritelab.data.dedupe_report import DedupeReportOptions, create_dedupe_report
from spritelab.data.ingest_clean_pngs import IngestOptions, ingest_clean_png_folder
from spritelab.data.manifest import DatasetManifest, IngestedSpriteRecord, load_manifest, save_manifest


def write_demo_inputs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_red_sprite(raw_dir / "red_a.png")
    _write_red_sprite(raw_dir / "red_b_duplicate.png")
    _write_red_sprite(raw_dir / "red_c_near_duplicate.png", extra_pixel=(23, 23))
    _write_blue_unique(raw_dir / "blue_unique.png")


def _write_red_sprite(path: Path, *, extra_pixel: tuple[int, int] | None = None) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            color = (92, 18, 24, 255) if x in (10, 21) or y in (10, 21) else (220, 46, 52, 255)
            image.putpixel((x, y), color)
    if extra_pixel is not None:
        image.putpixel(extra_pixel, (240, 80, 80, 255))
    image.save(path)


def _write_blue_unique(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(8, 20):
        for x in range(14, 26):
            color = (12, 34, 120, 255) if x in (14, 25) or y in (8, 19) else (30, 110, 230, 255)
            image.putpixel((x, y), color)
    image.save(path)


def _force_demo_splits(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    split_by_id = {
        "red_a": "train",
        "red_b_duplicate": "val",
        "red_c_near_duplicate": "train",
        "blue_unique": "test",
    }
    records = [
        IngestedSpriteRecord(
            id=record.id,
            source_path=record.source_path,
            bundle_dir=record.bundle_dir,
            width=record.width,
            height=record.height,
            category=record.category,
            subtype=record.subtype,
            license=record.license,
            palette_size=record.palette_size,
            sha256=record.sha256,
            split=split_by_id.get(record.id, record.split),
        )
        for record in manifest.records
    ]
    save_manifest(
        DatasetManifest(
            dataset_name=manifest.dataset_name,
            records=records,
            rejected_count=manifest.rejected_count,
            total_seen=manifest.total_seen,
            options=manifest.options,
        ),
        manifest_path,
    )


def main() -> None:
    demo_root = ROOT / "outputs" / "dedupe_report_demo"
    raw_dir = demo_root / "raw"
    dataset_dir = demo_root / "dataset"
    dedupe_dir = demo_root / "dedupe"

    write_demo_inputs(raw_dir)
    ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=dataset_dir,
            category="item_icon",
            license="CC0",
        )
    )
    _force_demo_splits(dataset_dir / "manifest.json")

    report = create_dedupe_report(
        DedupeReportOptions(
            dataset_path=dataset_dir,
            output_dir=dedupe_dir,
        )
    )

    print(f"dataset: {dataset_dir}")
    print(f"analyzed count: {report.summary.analyzed_records}")
    print(f"exact decoded duplicate groups: {report.summary.exact_decoded_duplicate_groups}")
    print(f"near duplicate groups: {report.summary.near_duplicate_groups}")
    print(f"cross-split exact groups: {report.summary.cross_split_exact_groups}")
    print(f"cross-split near groups: {report.summary.cross_split_near_groups}")
    print(f"json: {dedupe_dir / 'dedupe_report.json'}")
    print(f"markdown: {dedupe_dir / 'dedupe_report.md'}")


if __name__ == "__main__":
    main()
