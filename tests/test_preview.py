from __future__ import annotations

from PIL import Image

from spritelab.codec.preview import make_preview


def make_test_image() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((1, 0), (0, 0, 255, 255))
    return image


def test_default_preview_size_is_256() -> None:
    preview = make_preview(make_test_image())

    assert preview.size == (256, 256)
    assert preview.mode == "RGBA"


def test_preview_uses_nearest_neighbor_behavior() -> None:
    preview = make_preview(make_test_image(), scale=8)

    assert preview.getpixel((7, 0)) == (255, 0, 0, 255)
    assert preview.getpixel((8, 0)) == (0, 0, 255, 255)


def test_custom_preview_scale() -> None:
    preview = make_preview(make_test_image(), scale=3)

    assert preview.size == (96, 96)
