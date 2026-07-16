"""Demonstrate preview grid generation for an ingested SpriteBundle dataset."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spritelab.data.ingest_clean_pngs import IngestOptions, ingest_clean_png_folder
from spritelab.data.preview_grid import PreviewGridOptions, create_preview_grid


def write_demo_inputs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        ("ruby", (210, 46, 74, 255)),
        ("emerald", (54, 180, 96, 255)),
        ("sapphire", (58, 110, 220, 255)),
        ("amber", (230, 160, 40, 255)),
        ("violet", (150, 86, 210, 255)),
        ("silver", (190, 200, 210, 255)),
    ]
    for name, fill in colors:
        _write_token(raw_dir / f"{name}_token.png", fill=fill)


def _write_token(path: Path, *, fill: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    outline = (18, 18, 24, 255)
    highlight = (245, 245, 230, 255)
    center_x = 16
    center_y = 16

    for y in range(32):
        for x in range(32):
            distance = abs(x - center_x) + abs(y - center_y)
            if distance <= 9:
                image.putpixel((x, y), outline if distance == 9 else fill)

    for point in ((14, 11), (15, 11), (14, 12)):
        image.putpixel(point, highlight)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    demo_root = ROOT / "outputs" / "preview_grid_demo"
    raw_dir = demo_root / "raw"
    dataset_dir = demo_root / "dataset"
    grid_path = demo_root / "grid.png"

    write_demo_inputs(raw_dir)
    manifest = ingest_clean_png_folder(
        IngestOptions(
            input_dir=raw_dir,
            output_dir=dataset_dir,
            category="item_icon",
            license="CC0",
        )
    )
    create_preview_grid(
        PreviewGridOptions(
            dataset_path=dataset_dir,
            output_path=grid_path,
            columns=4,
            scale=8,
        )
    )

    print(f"dataset: {dataset_dir}")
    print(f"record count: {len(manifest.records)}")
    print(f"grid: {grid_path}")


if __name__ == "__main__":
    main()
