"""Deterministic RGB-to-name mapping and dominant sprite colors.

Dominant colors for prefill metadata are computed here from the bundle
palette and index-map frequencies instead of being asked from a VLM.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spritelab.codec.bundle import SpriteBundle
from spritelab.codec.oklab import rgb_u8_array_to_oklab

# Small fixed vocabulary; nearest neighbor is computed in OKLab so
# perceptually-close pixels land on the same name.
COLOR_NAME_TABLE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (0, 0, 0)),
    ("dark_gray", (72, 72, 72)),
    ("gray", (128, 128, 128)),
    ("light_gray", (192, 192, 192)),
    ("white", (255, 255, 255)),
    ("dark_red", (128, 24, 24)),
    ("red", (220, 40, 40)),
    ("orange", (240, 140, 40)),
    ("gold", (212, 164, 32)),
    ("brown", (130, 84, 44)),
    ("tan", (196, 160, 112)),
    ("beige", (232, 216, 176)),
    ("cream", (252, 236, 190)),
    ("yellow", (240, 220, 60)),
    ("olive", (120, 120, 40)),
    ("dark_green", (32, 96, 40)),
    ("green", (60, 168, 70)),
    ("lime", (140, 220, 80)),
    ("teal", (40, 140, 140)),
    ("cyan", (80, 210, 220)),
    ("dark_blue", (28, 44, 120)),
    ("blue", (60, 100, 220)),
    ("light_blue", (140, 180, 240)),
    ("navy", (20, 24, 72)),
    ("purple", (130, 60, 190)),
    ("magenta", (220, 60, 200)),
    ("pink", (240, 150, 190)),
)

_TABLE_NAMES = tuple(name for name, _ in COLOR_NAME_TABLE)
_TABLE_OKLAB = rgb_u8_array_to_oklab(np.array([rgb for _, rgb in COLOR_NAME_TABLE], dtype=np.uint8))


def color_name(rgb: Sequence[int]) -> str:
    """Return the nearest table name for one RGB color."""

    values = np.asarray(rgb, dtype=np.float64).reshape(3)
    point = rgb_u8_array_to_oklab(np.clip(np.rint(values), 0, 255).astype(np.uint8))
    distances = np.linalg.norm(_TABLE_OKLAB - point, axis=-1)
    return _TABLE_NAMES[int(np.argmin(distances))]


def dominant_colors_from_bundle(
    bundle: SpriteBundle,
    *,
    max_colors: int = 4,
    min_coverage: float = 0.15,
) -> tuple[str, ...]:
    """Return the dominant opaque color names by pixel coverage.

    Palette slots are mapped to names, same-name counts merged, and names
    covering at least ``min_coverage`` of opaque pixels returned (the top
    name is always kept so the result is never empty for visible sprites).
    """

    index_map = np.asarray(bundle.index_map)
    palette = np.asarray(bundle.palette)
    opaque = index_map[index_map >= 1]
    if opaque.size == 0:
        return ()

    slot_counts = np.bincount(opaque, minlength=palette.shape[0])
    name_counts: dict[str, int] = {}
    for slot in range(1, palette.shape[0]):
        count = int(slot_counts[slot]) if slot < slot_counts.shape[0] else 0
        if count == 0:
            continue
        name = color_name(palette[slot])
        name_counts[name] = name_counts.get(name, 0) + count

    total = int(opaque.size)
    ranked = sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    result = [
        name for index, (name, count) in enumerate(ranked[:max_colors]) if index == 0 or count / total >= min_coverage
    ]
    return tuple(result)
