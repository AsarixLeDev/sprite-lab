from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, PngImagePlugin


def make_png(
    path: Path, *, color: tuple[int, int, int, int] = (240, 80, 60, 255), size: tuple[int, int] = (16, 16)
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    left, top = max(1, size[0] // 4), max(1, size[1] // 4)
    right, bottom = max(left + 1, size[0] * 3 // 4), max(top + 1, size[1] * 3 // 4)
    for y in range(top, bottom):
        for x in range(left, right):
            image.putpixel((x, y), color)
    image.save(path)
    return path


def make_configured(root: Path, *, license_text: str = "CC0-1.0") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.txt").write_text(
        "Name: Test Pack\nCreator: Test Artist\nhttps://example.test/source\n",
        encoding="utf-8",
    )
    (root / "LICENSE").write_text(license_text + "\n", encoding="utf-8")
    return root


def make_opaque_background(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (32, 32), (20, 40, 80, 255))
    for y in range(9, 23):
        for x in range(9, 23):
            image.putpixel((x, y), (240, 190, 40, 255))
    image.save(path)
    return path


def make_sheet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (64, 16), (0, 0, 0, 0))
    for start in (2, 18, 34, 50):
        for y in range(4, 12):
            for x in range(start, start + 8):
                image.putpixel((x, y), (80 + start, 210, 100, 255))
    image.save(path)
    return path


def make_animated_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    second = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for index, image in enumerate((first, second)):
        for y in range(4, 12):
            for x in range(4 + index, 12 + index):
                if x < 16:
                    image.putpixel((x, y), (250, 90, 40, 255))
    first.save(path, save_all=True, append_images=[second], duration=100, loop=0, format="PNG")
    return path


def save_same_rgba_with_metadata(source: Path, destination: Path) -> Path:
    with Image.open(source) as image:
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("variant", "same decoded pixels, different source bytes")
        image.save(destination, pnginfo=png_info)
    return destination


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }
