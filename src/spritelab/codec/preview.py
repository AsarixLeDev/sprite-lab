"""Nearest-neighbor preview generation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from spritelab.codec.bundle import SpriteBundle
from spritelab.codec.role_inference import role_map_to_preview_image, save_role_map_preview
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.utils.image import ensure_rgba


def make_preview(image: Image.Image, scale: int = 8) -> Image.Image:
    """Scale an image with nearest-neighbor sampling only."""

    if scale < 1:
        raise ValueError("scale must be at least 1.")

    rgba = ensure_rgba(image)
    size = (rgba.width * scale, rgba.height * scale)
    return rgba.resize(size, resample=Image.Resampling.NEAREST)


def save_preview(bundle: SpriteBundle, path: str | Path, scale: int = 8) -> None:
    """Save a nearest-neighbor preview PNG for a sprite bundle."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = make_preview(reconstruct_rgba(bundle), scale=scale)
    preview.save(output_path)
