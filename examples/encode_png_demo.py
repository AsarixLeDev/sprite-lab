"""Create a clean PNG and encode it into a SpriteBundle."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spritelab.codec.bundle import SpriteMetadata
from spritelab.codec.encode import encode_png_to_bundle
from spritelab.codec.io import save_bundle


def create_source_png(path: Path) -> None:
    """Create a tiny clean 32x32 vial-like pixel-art source image."""

    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))

    outline = (20, 18, 28, 255)
    shadow = (62, 42, 86, 255)
    midtone = (96, 84, 188, 255)
    light = (134, 168, 255, 255)
    highlight = (238, 248, 255, 255)

    for y in range(8, 26):
        for x in range(11, 21):
            if y < 11 and x not in range(14, 18):
                continue
            if y > 22 and (x < 13 or x > 18):
                continue

            if x in (11, 20) or y in (8, 25):
                image.putpixel((x, y), outline)
            elif y > 19 or x < 14:
                image.putpixel((x, y), shadow)
            elif y < 14:
                image.putpixel((x, y), light)
            else:
                image.putpixel((x, y), midtone)

    for point in ((17, 12), (18, 13), (17, 14)):
        image.putpixel(point, highlight)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    output_root = ROOT / "outputs" / "encode_png_demo"
    source_path = output_root / "source.png"
    bundle_dir = output_root / "bundle"

    create_source_png(source_path)

    metadata = SpriteMetadata(
        id="encode_png_demo_vial",
        category="demo",
        subtype="vial",
        source=str(source_path),
        license="CC0-1.0",
    )
    bundle = encode_png_to_bundle(
        source_path,
        metadata=metadata,
        max_visible_colors=32,
        canonicalize_palette=True,
        generate_role_map=True,
    )
    save_bundle(bundle, bundle_dir)

    print(f"palette size: {bundle.metadata.palette_size}")
    print(f"metadata id: {bundle.metadata.id}")
    print("canonicalization used: True")
    print(f"source: {source_path}")
    print(f"bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
