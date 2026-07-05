"""Shared synthetic fixtures for harvest tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.harvest.sources import SourceLicense, SourceRecord


def make_sprite_png(
    path: Path,
    *,
    size: int = 32,
    colors: int = 3,
    empty: bool = False,
    color: tuple[int, int, int, int] | tuple[int, int, int] | None = None,
) -> Path:
    """Write a valid hard-alpha sprite PNG with a small colored square."""

    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    if not empty:
        palette = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40)]
        if color is not None:
            palette[0] = tuple(int(channel) for channel in color[:3])
        block = max(2, size // 4)
        for index in range(min(colors, len(palette))):
            r, g, b = palette[index]
            y = size // 4 + (index % 2)
            x = size // 4 + index * 2
            pixels[y : y + block, x : x + 2] = (r, g, b, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGBA").save(path)
    return path


def make_sheet_png(path: Path, *, rows: int = 2, cols: int = 2, tile: int = 32, empty_tiles: tuple[tuple[int, int], ...] = ()) -> Path:
    """Write a sprite sheet with distinct colored tiles."""

    pixels = np.zeros((rows * tile, cols * tile, 4), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if (row, col) in empty_tiles:
                continue
            color = (60 + row * 80, 60 + col * 80, 120, 255)
            y0, x0 = row * tile + 4, col * tile + 4
            pixels[y0 : y0 + tile // 2, x0 : x0 + tile // 2] = color
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGBA").save(path)
    return path


def make_zip_of_pngs(zip_path: Path, names: list[str], *, size: int = 32) -> Path:
    """Write a ZIP whose PNG members are synthetic sprites."""

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    staging = zip_path.parent / (zip_path.stem + "_staging")
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in names:
            png = make_sprite_png(staging / Path(name).name, size=size)
            archive.write(png, arcname=name)
    return zip_path


def make_source(
    source_id: str = "test_source",
    *,
    license_name: str = "cc0",
    user_confirmed: bool = True,
    **kwargs,
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        source_name=source_id.replace("_", " ").title(),
        source_type=kwargs.pop("source_type", "local_directory"),
        author=kwargs.pop("author", "Tester"),
        license=SourceLicense(license=license_name, user_confirmed=user_confirmed),
        **kwargs,
    )
