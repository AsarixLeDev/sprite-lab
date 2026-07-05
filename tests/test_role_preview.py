from __future__ import annotations

import numpy as np
from PIL import Image

from spritelab.codec.role_inference import role_map_to_preview_image, save_role_map_preview
from spritelab.codec.roles import ROLE_ACCENT, ROLE_MIDTONE, ROLE_OUTLINE, ROLE_PREVIEW_COLORS, ROLE_TRANSPARENT


def test_role_map_preview_default_and_custom_scale() -> None:
    role_map = np.full((32, 32), ROLE_TRANSPARENT, dtype=np.uint8)
    role_map[0, 0] = ROLE_OUTLINE
    role_map[0, 1] = ROLE_MIDTONE

    default = role_map_to_preview_image(role_map)
    custom = role_map_to_preview_image(role_map, scale=2)

    assert default.mode == "RGBA"
    assert default.size == (256, 256)
    assert custom.size == (64, 64)


def test_role_map_preview_uses_stable_role_colors_and_nearest_neighbor() -> None:
    role_map = np.full((32, 32), ROLE_TRANSPARENT, dtype=np.uint8)
    role_map[0, 0] = ROLE_OUTLINE
    role_map[0, 1] = ROLE_ACCENT

    preview = role_map_to_preview_image(role_map, scale=3)

    assert preview.getpixel((0, 0)) == ROLE_PREVIEW_COLORS[ROLE_OUTLINE]
    assert preview.getpixel((2, 2)) == ROLE_PREVIEW_COLORS[ROLE_OUTLINE]
    assert preview.getpixel((3, 0)) == ROLE_PREVIEW_COLORS[ROLE_ACCENT]
    assert preview.getpixel((6, 0)) == ROLE_PREVIEW_COLORS[ROLE_TRANSPARENT]


def test_save_role_map_preview_writes_loadable_png(tmp_path) -> None:
    role_map = np.full((32, 32), ROLE_TRANSPARENT, dtype=np.uint8)
    role_map[10, 10] = ROLE_MIDTONE
    output = tmp_path / "roles.png"

    save_role_map_preview(role_map, output, scale=4)

    with Image.open(output) as image:
        assert image.mode == "RGBA"
        assert image.size == (128, 128)
