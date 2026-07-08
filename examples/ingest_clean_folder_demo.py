"""Demonstrate batch ingestion of clean 32x32 PNG sprites."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spritelab.data.ingest_clean_pngs import IngestOptions, ingest_clean_png_folder


def write_demo_inputs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_diamond(raw_dir / "red_diamond.png", fill=(200, 40, 80, 255))
    _write_diamond(raw_dir / "blue_diamond.png", fill=(40, 100, 220, 255))
    _write_wrong_size(raw_dir / "wrong_size.png")


def _write_diamond(path: Path, *, fill: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    outline = (18, 16, 24, 255)
    center_x = 16
    center_y = 16

    for y in range(32):
        for x in range(32):
            distance = abs(x - center_x) + abs(y - center_y)
            if distance <= 8:
                image.putpixel((x, y), outline if distance == 8 else fill)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_wrong_size(path: Path) -> None:
    image = Image.new("RGBA", (31, 32), (255, 0, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    demo_root = ROOT / "outputs" / "ingest_clean_folder_demo"
    raw_dir = demo_root / "raw"
    output_dir = demo_root / "processed"

    write_demo_inputs(raw_dir)
    manifest = ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=output_dir,
            category="item_icon",
            license="CC0",
        )
    )

    print(f"total seen: {manifest.total_seen}")
    print(f"encoded count: {len(manifest.records)}")
    print(f"rejected count: {manifest.rejected_count}")
    print(f"manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
