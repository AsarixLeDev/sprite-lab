"""Small Pillow image helpers."""

from __future__ import annotations

from PIL import Image


def ensure_rgba(image: Image.Image) -> Image.Image:
    """Return an RGBA image, converting only when needed."""

    if image.mode == "RGBA":
        return image
    return image.convert("RGBA")


def assert_exact_size(image: Image.Image, size: tuple[int, int] = (32, 32)) -> None:
    """Raise ValueError when an image does not have the exact requested size."""

    if image.size != size:
        raise ValueError(f"expected image size {size}, got {image.size}.")
