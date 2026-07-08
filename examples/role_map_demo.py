"""Demonstrate deterministic role-map inference and preview generation."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spritelab.codec.bundle import SpriteMetadata
from spritelab.codec.encode import encode_rgba_image_to_bundle
from spritelab.codec.io import save_bundle
from spritelab.codec.role_inference import (
    describe_role_inference,
    infer_palette_slot_roles_v2,
    save_role_map_preview,
)


def make_demo_sprite() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(7, 25):
        for x in range(8, 24):
            if x in (8, 23) or y in (7, 24):
                color = (12, 10, 18, 255)
            elif y >= 18:
                color = (54, 36, 84, 255)
            elif y <= 11:
                color = (180, 130, 214, 255)
            else:
                color = (116, 72, 152, 255)
            image.putpixel((x, y), color)

    for x, y in [(17, 10), (18, 10), (18, 11)]:
        image.putpixel((x, y), (252, 236, 120, 255))
    for x, y in [(12, 19), (20, 17), (15, 14)]:
        image.putpixel((x, y), (255, 70, 170, 255))
    image.putpixel((21, 12), (80, 240, 255, 255))
    return image


def main() -> None:
    output_root = ROOT / "outputs" / "role_map_demo"
    source_path = output_root / "source.png"
    bundle_dir = output_root / "bundle"
    role_preview_path = output_root / "role_preview_8x.png"

    output_root.mkdir(parents=True, exist_ok=True)
    image = make_demo_sprite()
    image.save(source_path)

    bundle = encode_rgba_image_to_bundle(
        image,
        SpriteMetadata(id="role_map_demo", category="item_icon", source=str(source_path)),
        canonicalize_palette=True,
        generate_role_map=True,
    )
    save_bundle(bundle, bundle_dir)
    if bundle.role_map is not None:
        save_role_map_preview(bundle.role_map, role_preview_path, scale=8)

    result = infer_palette_slot_roles_v2(bundle.palette, bundle.index_map, bundle.alpha)
    print(f"source: {source_path}")
    print(f"bundle: {bundle_dir}")
    print(f"role preview: {role_preview_path}")
    for line in describe_role_inference(result):
        print(line)


if __name__ == "__main__":
    main()
