from pathlib import Path

from PIL import Image

from spritelab.harvest.visual_facts import (
    dominant_color_names_from_rgba,
    extract_visual_facts_from_png,
    shape_hints_from_alpha,
)


def test_visual_facts_bbox_and_colors(tmp_path: Path) -> None:
    path = tmp_path / "wide.png"
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 14):
        for x in range(4, 22):
            image.putpixel((x, y), (240, 40, 40, 255))
    image.save(path)

    facts = extract_visual_facts_from_png(path)
    assert facts.content_bbox == (4, 10, 22, 14)
    assert facts.content_width == 18
    assert facts.content_height == 4
    assert facts.opaque_pixel_count == 72
    assert facts.alpha_hard
    assert "red" in dominant_color_names_from_rgba(path)
    assert "wide" in shape_hints_from_alpha(path)


def test_visual_facts_transparent_and_roundish(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(empty)
    assert extract_visual_facts_from_png(empty).content_bbox is None

    roundish = tmp_path / "roundish.png"
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 20):
        for x in range(10, 20):
            image.putpixel((x, y), (20, 180, 40, 255))
    image.save(roundish)
    assert "roundish" in shape_hints_from_alpha(roundish)
