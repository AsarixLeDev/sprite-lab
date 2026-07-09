"""Demonstrate dataset quality report generation."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spritelab.data.ingest_clean_pngs import IngestOptions, ingest_clean_png_folder
from spritelab.data.quality_report import QualityReportOptions, create_quality_report


def write_demo_inputs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_centered_square(raw_dir / "normal_square.png")
    _write_mostly_empty(raw_dir / "mostly_empty.png")
    _write_off_center(raw_dir / "off_center.png")
    _write_low_contrast(raw_dir / "low_contrast.png")
    _write_touching_edge(raw_dir / "touching_edge.png")
    _write_speckles(raw_dir / "speckles.png")


def _write_centered_square(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            image.putpixel((x, y), (42, 40, 56, 255) if x in (10, 21) or y in (10, 21) else (160, 92, 210, 255))
    image.save(path)


def _write_mostly_empty(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((15, 15), (220, 80, 80, 255))
    image.save(path)


def _write_off_center(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(2, 8):
        for x in range(24, 30):
            image.putpixel((x, y), (80, 180, 220, 255))
    image.save(path)


def _write_low_contrast(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(11, 21):
        for x in range(11, 21):
            image.putpixel((x, y), (100, 100, 100, 255) if x < 16 else (108, 108, 108, 255))
    image.save(path)


def _write_touching_edge(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(0, 10):
        for x in range(0, 10):
            image.putpixel((x, y), (210, 160, 60, 255))
    image.save(path)


def _write_speckles(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y, x in [(2, 2), (5, 8), (8, 14), (11, 20), (14, 26), (20, 6), (23, 12), (26, 18)]:
        image.putpixel((x, y), (230, 230, 240, 255))
    image.save(path)


def main() -> None:
    demo_root = ROOT / "outputs" / "quality_report_demo"
    raw_dir = demo_root / "raw"
    dataset_dir = demo_root / "dataset"
    quality_dir = demo_root / "quality"

    write_demo_inputs(raw_dir)
    ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=dataset_dir,
            category="item_icon",
            license="CC0",
        )
    )
    report = create_quality_report(
        QualityReportOptions(
            dataset_path=dataset_dir,
            output_dir=quality_dir,
        )
    )

    print(f"dataset: {dataset_dir}")
    print(f"analyzed count: {report.summary.analyzed_records}")
    print(f"issue counts: {report.summary.issue_counts}")
    print(f"json: {quality_dir / 'quality_report.json'}")
    print(f"markdown: {quality_dir / 'quality_report.md'}")


if __name__ == "__main__":
    main()
