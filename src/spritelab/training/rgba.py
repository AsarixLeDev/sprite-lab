"""RGBA target conversion and sample-sheet helpers for generator training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SPRITE_SIZE = 32


def npz_row_to_rgba(
    *,
    index_map: np.ndarray,
    alpha: np.ndarray,
    palette: np.ndarray,
    palette_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert one exported indexed-palette row to float RGBA ``[4, 32, 32]``.

    RGB values are looked up from the per-sprite palette and normalized to
    ``[0, 1]``. Transparent pixels keep alpha 0 and have RGB zeroed so the
    generator has a stable background target independent of palette row 0.
    """

    index = np.asarray(index_map)
    alpha_arr = _normalize_alpha(alpha)
    palette_rgb = _normalize_palette_rgb(palette)

    if index.shape != (SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"index_map must have shape [32, 32], got {index.shape}")
    if alpha_arr.shape != (SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"alpha must have shape [32, 32] or [1, 32, 32], got {np.asarray(alpha).shape}")
    if index.dtype.kind not in "biu":
        raise ValueError(f"index_map must contain integer palette indices, got dtype {index.dtype}")

    min_index = int(index.min()) if index.size else 0
    max_index = int(index.max()) if index.size else 0
    if min_index < 0 or max_index >= palette_rgb.shape[0]:
        raise ValueError(
            "invalid palette index in index_map: "
            f"range [{min_index}, {max_index}] outside palette rows [0, {palette_rgb.shape[0] - 1}]"
        )

    if palette_mask is not None:
        mask = np.asarray(palette_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != palette_rgb.shape[0]:
            raise ValueError(f"palette_mask must have shape [{palette_rgb.shape[0]}], got {mask.shape}")
        used = np.unique(index.astype(np.int64, copy=False))
        invalid_used = [int(value) for value in used if not bool(mask[int(value)])]
        if invalid_used:
            preview = ", ".join(str(value) for value in invalid_used[:8])
            raise ValueError(f"index_map uses palette indices disabled by palette_mask: {preview}")

    rgb_hwc = palette_rgb[index.astype(np.int64, copy=False)]
    opaque = alpha_arr > 0.0
    rgb_hwc = rgb_hwc.copy()
    rgb_hwc[~opaque] = 0.0

    rgba = np.empty((4, SPRITE_SIZE, SPRITE_SIZE), dtype=np.float32)
    rgba[:3] = np.moveaxis(rgb_hwc, -1, 0).astype(np.float32, copy=False)
    rgba[3] = alpha_arr.astype(np.float32, copy=False)
    return rgba


def save_rgba_contact_sheet(
    *,
    outputs: dict[str, Any],
    path: str | Path,
    batch: dict[str, Any] | None = None,
    max_items: int = 16,
    scale: int = 6,
) -> None:
    """Write a generated RGBA contact sheet.

    With a target batch, each sample row is target, generated, predicted alpha.
    Without a target batch, each row is generated and predicted alpha.
    """

    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - Pillow is a project dependency.
        return

    rgb_logits = _to_numpy(outputs["rgb_logits"])
    alpha_logits = _to_numpy(outputs["alpha_logits"])
    rgb = _sigmoid(rgb_logits)
    alpha = _sigmoid(alpha_logits)
    count = min(int(rgb.shape[0]), int(max_items))
    if count <= 0:
        return

    target_rgba = None
    if batch is not None and "rgba" in batch:
        target_rgba = _to_numpy(batch["rgba"])
        count = min(count, int(target_rgba.shape[0]))

    columns = 3 if target_rgba is not None else 2
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    sheet = Image.new(
        "RGBA",
        (columns * cell + (columns + 1) * padding, count * cell + (count + 1) * padding),
        (36, 36, 40, 255),
    )
    for row in range(count):
        images = []
        if target_rgba is not None:
            images.append(_rgba_chw_to_image(target_rgba[row]))
        images.append(_rgba_chw_to_image(np.concatenate([rgb[row], alpha[row]], axis=0)))
        images.append(_alpha_chw_to_image(alpha[row]))

        top = padding + row * (cell + padding)
        for col, image in enumerate(images):
            left = padding + col * (cell + padding)
            sheet.paste(image.resize((cell, cell), Image.NEAREST), (left, top))

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    value = np.asarray(alpha)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    value = value.astype(np.float32, copy=False)
    if value.size and float(np.nanmax(value)) > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_palette_rgb(palette: np.ndarray) -> np.ndarray:
    value = np.asarray(palette)
    if value.ndim != 2 or value.shape[1] < 3:
        raise ValueError(f"palette must have shape [K, 3] or [K, 4], got {value.shape}")
    rgb = value[:, :3].astype(np.float32, copy=False)
    if value.dtype.kind in "ui" or (rgb.size and float(np.nanmax(rgb)) > 1.0):
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


def _rgba_chw_to_image(rgba: np.ndarray) -> Any:
    from PIL import Image

    arr = np.asarray(rgba, dtype=np.float32)
    if arr.shape != (4, SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"RGBA image must have shape [4, 32, 32], got {arr.shape}")
    hwc = np.moveaxis(np.clip(arr, 0.0, 1.0), 0, -1)
    return Image.fromarray(np.rint(hwc * 255.0).astype(np.uint8), mode="RGBA")


def _alpha_chw_to_image(alpha: np.ndarray) -> Any:
    from PIL import Image

    arr = np.asarray(alpha, dtype=np.float32)
    if arr.shape == (SPRITE_SIZE, SPRITE_SIZE):
        arr = arr[None, :, :]
    if arr.shape != (1, SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"alpha image must have shape [1, 32, 32], got {arr.shape}")
    gray = np.clip(arr[0], 0.0, 1.0)
    rgba = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(gray[..., None] * 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    return Image.fromarray(rgba, mode="RGBA")
