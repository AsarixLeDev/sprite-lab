"""Tests for spritelab.harvest.sheets."""

from __future__ import annotations

from PIL import Image

from _harvest_testdata import make_sheet_png, make_sprite_png

from spritelab.harvest.sheets import SheetSliceConfig, looks_like_sprite_sheet, slice_sheet_to_pngs


def test_slices_64x64_into_four_tiles(tmp_path):
    sheet = make_sheet_png(tmp_path / "sheet.png", rows=2, cols=2)
    tiles = slice_sheet_to_pngs(sheet, tmp_path / "tiles", SheetSliceConfig())
    assert len(tiles) == 4
    for tile in tiles:
        with Image.open(tile) as image:
            assert image.size == (32, 32)


def test_skips_empty_tiles(tmp_path):
    sheet = make_sheet_png(tmp_path / "sheet.png", rows=2, cols=2, empty_tiles=((1, 1),))
    tiles = slice_sheet_to_pngs(sheet, tmp_path / "tiles", SheetSliceConfig())
    assert len(tiles) == 3
    assert not any(t.name.endswith("r001_c001.png") for t in tiles)


def test_deterministic_tile_names(tmp_path):
    sheet = make_sheet_png(tmp_path / "sheet.png", rows=2, cols=2)
    tiles = slice_sheet_to_pngs(sheet, tmp_path / "tiles", SheetSliceConfig())
    names = sorted(t.name for t in tiles)
    assert names == [
        "sheet__r000_c000.png",
        "sheet__r000_c001.png",
        "sheet__r001_c000.png",
        "sheet__r001_c001.png",
    ]


def test_detects_likely_sheet(tmp_path):
    sheet = make_sheet_png(tmp_path / "sheet.png", rows=2, cols=2)
    single = make_sprite_png(tmp_path / "single.png", size=32)
    assert looks_like_sprite_sheet(sheet)
    assert not looks_like_sprite_sheet(single)


def test_custom_spacing_margins(tmp_path):
    # 2x2 grid of 16px tiles with 4px margins and 2px spacing: 4+16+2+16 = 38.
    import numpy as np

    pixels = np.zeros((38, 38, 4), dtype=np.uint8)
    for row in range(2):
        for col in range(2):
            y0 = 4 + row * 18
            x0 = 4 + col * 18
            pixels[y0 : y0 + 16, x0 : x0 + 16] = (100 + row * 50, 100 + col * 50, 90, 255)
    path = tmp_path / "custom.png"
    Image.fromarray(pixels, mode="RGBA").save(path)

    config = SheetSliceConfig(tile_width=16, tile_height=16, margin_x=4, margin_y=4, spacing_x=2, spacing_y=2)
    tiles = slice_sheet_to_pngs(path, tmp_path / "tiles", config)
    assert len(tiles) == 4
    for tile in tiles:
        with Image.open(tile) as image:
            assert image.size == (16, 16)
            assert image.getpixel((8, 8))[3] == 255
