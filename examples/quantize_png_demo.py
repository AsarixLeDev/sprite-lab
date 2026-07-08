"""Demonstrate OKLab quantization for an over-color 32x32 sprite."""

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
from spritelab.codec.quantize import QuantizationOptions, encode_png_to_quantized_bundle


def make_over_color_sprite() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(5, 27):
        for x in range(5, 27):
            edge = x in (5, 26) or y in (5, 26)
            if edge:
                red = 32 + x * 2
                green = 28 + y * 2
                blue = 48 + ((x + y) % 12)
            else:
                red = 90 + x * 5
                green = 48 + y * 6
                blue = 110 + ((x * 4 + y * 3) % 60)
            image.putpixel((x, y), (red % 256, green % 256, blue % 256, 255))
    return image


def main() -> None:
    demo_root = ROOT / "outputs" / "quantize_png_demo"
    source_path = demo_root / "source_over_color.png"
    input_folder = demo_root / "input_folder"
    bundle_dir = demo_root / "quantized_bundle"

    demo_root.mkdir(parents=True, exist_ok=True)
    input_folder.mkdir(parents=True, exist_ok=True)
    source = make_over_color_sprite()
    source.save(source_path)
    source.save(input_folder / "source_over_color.png")

    try:
        encode_png_to_bundle(
            source_path,
            metadata=SpriteMetadata(id="source_over_color", category="item_icon"),
            max_visible_colors=16,
        )
    except ValueError as exc:
        print(f"strict encoding failed as expected: {exc}")

    bundle = encode_png_to_quantized_bundle(
        source_path,
        metadata=SpriteMetadata(id="source_over_color", category="item_icon", source=str(source_path)),
        options=QuantizationOptions(target_visible_colors=16),
    )
    save_bundle(bundle, bundle_dir)
    extra = bundle.metadata.extra

    print(f"original visible colors: {extra['original_visible_color_count']}")
    print(f"quantized visible colors: {extra['quantized_visible_color_count']}")
    print(f"mean OKLab error: {extra['mean_oklab_error']:.6f}")
    print(f"max OKLab error: {extra['max_oklab_error']:.6f}")
    print(f"source: {source_path}")
    print(f"input folder: {input_folder}")
    print(f"bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
