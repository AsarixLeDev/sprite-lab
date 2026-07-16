"""Sprite-sheet slicing into individual tiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SheetSliceConfig:
    enabled: bool = True
    tile_width: int = 32
    tile_height: int = 32
    margin_x: int = 0
    margin_y: int = 0
    spacing_x: int = 0
    spacing_y: int = 0
    skip_empty: bool = True
    min_opaque_pixels: int = 8


@dataclass(frozen=True)
class ExplicitSheetCell:
    """A sheet coordinate and its independently recorded pixel rectangle."""

    row: int
    column: int
    cell_width: int
    cell_height: int
    crop_rectangle: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (self.row, self.column)):
            raise ValueError("sheet row and column must be non-negative integers")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.cell_width, self.cell_height)
        ):
            raise ValueError("sheet cell dimensions must be positive integers")
        left, top, right, bottom = self.crop_rectangle
        if min(left, top) < 0 or right - left != self.cell_width or bottom - top != self.cell_height:
            raise ValueError("crop rectangle must explicitly match the sheet cell dimensions")


def slice_sheet_to_pngs(
    image_path: str | Path,
    output_dir: str | Path,
    config: SheetSliceConfig,
) -> list[Path]:
    """Crop a sheet into tiles and save non-empty ones as PNGs.

    Tiles are named ``<stem>__r{row:03d}_c{col:03d}.png``; no resizing.
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as opened:
        sheet = opened.convert("RGBA")

    step_x = config.tile_width + config.spacing_x
    step_y = config.tile_height + config.spacing_y
    usable_width = sheet.width - config.margin_x
    usable_height = sheet.height - config.margin_y

    written: list[Path] = []
    row = 0
    y = config.margin_y
    while y + config.tile_height <= config.margin_y + usable_height:
        col = 0
        x = config.margin_x
        while x + config.tile_width <= config.margin_x + usable_width:
            tile = sheet.crop((x, y, x + config.tile_width, y + config.tile_height))
            if not config.skip_empty or _opaque_pixel_count(tile) >= config.min_opaque_pixels:
                path = output_dir / f"{image_path.stem}__r{row:03d}_c{col:03d}.png"
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite historical sheet extraction: {path}")
                tile.save(path)
                written.append(path)
            col += 1
            x += step_x
        row += 1
        y += step_y
    return written


def looks_like_sprite_sheet(path: str | Path) -> bool:
    """Cheap heuristic: larger than one tile with grid-friendly dimensions."""

    with Image.open(path) as image:
        width, height = image.size
    if width <= 32 and height <= 32:
        return False
    divisible = (width % 32 == 0 and height % 32 == 0) or (width % 16 == 0 and height % 16 == 0)
    return divisible


def center_pad_to_32(image_path: str | Path, output_path: str | Path) -> Path:
    """Place a smaller-than-32 sprite centered on a transparent 32x32 canvas."""

    image_path = Path(image_path)
    output_path = Path(output_path)
    with Image.open(image_path) as opened:
        rgba = opened.convert("RGBA")
    if rgba.width > 32 or rgba.height > 32:
        raise ValueError(f"cannot center-pad {rgba.size} to 32x32; image is larger than 32.")
    canvas = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    left, top, _, _ = center_padding(rgba.width, rgba.height, 32, 32)
    offset = (left, top)
    canvas.paste(rgba, offset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite historical padded extraction: {output_path}")
    canvas.save(output_path)
    return output_path


def center_padding(width: int, height: int, target_width: int, target_height: int) -> tuple[int, int, int, int]:
    """Return exact left/top/right/bottom padding with the odd pixel on right/bottom."""

    values = (width, height, target_width, target_height)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("source and target dimensions must be positive integers")
    if width > target_width or height > target_height:
        raise ValueError("source dimensions exceed target dimensions")
    horizontal = target_width - width
    vertical = target_height - height
    left = horizontal // 2
    top = vertical // 2
    return left, top, horizontal - left, vertical - top


def crop_explicit_sheet_cell(image: Image.Image, cell: ExplicitSheetCell) -> Image.Image:
    """Crop one explicitly bound cell without deriving coordinates or resampling."""

    _, _, right, bottom = cell.crop_rectangle
    if right > image.width or bottom > image.height:
        raise ValueError(f"explicit sheet crop is out of bounds: {cell.crop_rectangle} for {image.size}")
    return image.convert("RGBA").crop(cell.crop_rectangle)


def _opaque_pixel_count(tile: Image.Image) -> int:
    alpha = np.asarray(tile, dtype=np.uint8)[:, :, 3]
    return int((alpha > 0).sum())
