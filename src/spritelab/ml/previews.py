"""Preview rendering for masked index-map predictions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ERROR_COLOR = (255, 0, 255, 255)


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def decode_index_map_to_rgba(
    index_map: np.ndarray | torch.Tensor,
    palette: np.ndarray | torch.Tensor,
    alpha: np.ndarray | torch.Tensor | None = None,
) -> Image.Image:
    """Render an index map to a strict 32x32 RGBA image.

    Index 0 is transparent, invalid indices become magenta error pixels,
    and an explicit ``alpha`` overrides the derived alpha channel.
    """

    indices = _to_numpy(index_map).astype(np.int64)
    palette_array = _to_numpy(palette)
    if palette_array.dtype != np.uint8:
        # Accept normalized float palettes as well.
        if np.issubdtype(palette_array.dtype, np.floating):
            palette_array = np.clip(palette_array * 255.0, 0, 255).astype(np.uint8)
        else:
            palette_array = np.clip(palette_array, 0, 255).astype(np.uint8)

    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    invalid = (indices < 0) | (indices >= palette_array.shape[0])
    valid = ~invalid
    clamped = np.clip(indices, 0, palette_array.shape[0] - 1)
    rgba[..., :3] = palette_array[clamped]
    rgba[..., 3] = np.where(indices == 0, 0, 255)
    rgba[invalid] = ERROR_COLOR
    rgba[valid & (indices == 0)] = (0, 0, 0, 0)

    if alpha is not None:
        alpha_array = _to_numpy(alpha).astype(np.uint8)
        rgba[..., 3] = np.where(alpha_array > 0, 255, 0).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


def save_prediction_grid(
    samples: Sequence[dict[str, Any]],
    predictions: Sequence[torch.Tensor],
    output_path: str | Path,
    scale: int = 8,
    max_items: int = 32,
) -> None:
    """Write a grid image with one row per sample.

    Columns: masked input | prediction | target | alpha.
    """

    count = min(len(samples), len(predictions), max_items)
    if count == 0:
        raise ValueError("no samples to render")

    cell = 32 * scale
    padding = scale
    columns = 4
    width = columns * cell + (columns + 1) * padding
    height = count * cell + (count + 1) * padding
    grid = Image.new("RGBA", (width, height), (40, 40, 48, 255))

    for row in range(count):
        sample = samples[row]
        prediction = predictions[row]
        palette = sample.get("palette_u8", sample["palette"])
        alpha = sample["alpha"]
        input_map = sample.get("input_index_map", sample["index_map"])
        target_map = sample.get("target_index_map", sample["index_map"])

        alpha_array = _to_numpy(alpha).astype(np.uint8)
        alpha_rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        alpha_rgba[..., :3] = (alpha_array[..., None] * 255).astype(np.uint8)
        alpha_rgba[..., 3] = 255

        cells = [
            decode_index_map_to_rgba(input_map, palette, alpha),
            decode_index_map_to_rgba(prediction, palette, alpha),
            decode_index_map_to_rgba(target_map, palette, alpha),
            Image.fromarray(alpha_rgba, mode="RGBA"),
        ]
        top = padding + row * (cell + padding)
        for column, image in enumerate(cells):
            left = padding + column * (cell + padding)
            grid.paste(image.resize((cell, cell), Image.NEAREST), (left, top))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
