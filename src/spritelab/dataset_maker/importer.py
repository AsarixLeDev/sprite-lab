"""PNG import and validation for the Dataset Maker GUI."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from spritelab.codec.bundle import SPRITE_HEIGHT, SPRITE_WIDTH, SpriteBundle, SpriteMetadata
from spritelab.codec.encode import encode_rgba_image_to_bundle
from spritelab.codec.palette import visible_palette_size
from spritelab.codec.preview import make_preview
from spritelab.codec.quantize import QuantizationOptions, encode_rgba_image_to_quantized_bundle
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.codec.role_inference import role_map_to_preview_image
from spritelab.codec.validate import validate_bundle
from spritelab.dataset_maker.model import DatasetMakerItem, normalize_sprite_id


@dataclass(frozen=True)
class ImportOptions:
    max_palette_slots: int = 32
    allow_quantize_overcolor: bool = True
    quantize_overcolor: bool = True
    allow_nearest_resize: bool = False
    infer_role_map: bool = True
    canonicalize_palette: bool = True
    recursive: bool = False


@dataclass(frozen=True)
class ImportedSprite:
    item: DatasetMakerItem
    bundle: SpriteBundle | None
    preview_image: Image.Image | None
    alpha_preview_image: Image.Image | None
    role_preview_image: Image.Image | None
    palette_strip_image: Image.Image | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    auto_metadata: Mapping[str, Any] = field(default_factory=dict)


def import_png_as_dataset_item(
    path: str | Path,
    *,
    options: ImportOptions,
    default_category: str = "unknown",
    default_tags: Sequence[str] = (),
) -> ImportedSprite:
    """Import one PNG path as an editable Dataset Maker sprite."""

    source_path = Path(path)
    sprite_id = normalize_sprite_id(source_path.stem)
    base_item = DatasetMakerItem(
        sprite_id=sprite_id,
        source_path=source_path,
        status="rejected",
        category=default_category,
        tags=tuple(default_tags),
        source_name=source_path.name,
    )

    if source_path.suffix.lower() != ".png":
        return _failed_import(base_item, [f"unsupported file type for {source_path.name}; expected .png."])

    if options.max_palette_slots < 1:
        return _failed_import(base_item, ["max_palette_slots must be at least 1."])

    warnings: list[str] = []
    try:
        with Image.open(source_path) as opened:
            rgba = opened.convert("RGBA")
    except FileNotFoundError:
        return _failed_import(base_item, [f"file not found: {source_path}."])
    except (OSError, UnidentifiedImageError) as exc:
        return _failed_import(base_item, [f"could not load PNG: {exc}."])

    return _import_loaded_rgba(
        base_item,
        source_path,
        rgba,
        options=options,
        default_category=default_category,
        default_tags=default_tags,
        warnings=warnings,
    )


def import_png_bytes_as_dataset_item(
    content: bytes,
    *,
    source_name: str,
    options: ImportOptions,
    default_category: str = "unknown",
    default_tags: Sequence[str] = (),
) -> ImportedSprite:
    """Import already-held PNG bytes without reopening a mutable pathname."""

    source_path = Path(source_name)
    base_item = DatasetMakerItem(
        sprite_id=normalize_sprite_id(source_path.stem),
        source_path=source_path,
        status="rejected",
        category=default_category,
        tags=tuple(default_tags),
        source_name=source_path.name,
    )
    if source_path.suffix.lower() != ".png":
        return _failed_import(base_item, [f"unsupported file type for {source_path.name}; expected .png."])
    if options.max_palette_slots < 1:
        return _failed_import(base_item, ["max_palette_slots must be at least 1."])
    if not isinstance(content, bytes) or not content:
        return _failed_import(base_item, [f"could not load PNG bytes for {source_path.name}."])
    try:
        with Image.open(io.BytesIO(content)) as opened:
            if opened.format != "PNG" or getattr(opened, "n_frames", 1) != 1:
                return _failed_import(base_item, [f"could not load a static PNG: {source_path.name}."])
            opened.load()
            rgba = opened.convert("RGBA")
    except (OSError, UnidentifiedImageError) as exc:
        return _failed_import(base_item, [f"could not load PNG: {exc}."])
    return _import_loaded_rgba(
        base_item,
        source_path,
        rgba,
        options=options,
        default_category=default_category,
        default_tags=default_tags,
    )


def _import_loaded_rgba(
    base_item: DatasetMakerItem,
    source_path: Path,
    rgba: Image.Image,
    *,
    options: ImportOptions,
    default_category: str,
    default_tags: Sequence[str],
    warnings: Sequence[str] = (),
) -> ImportedSprite:
    warnings = list(warnings)
    if rgba.size != (SPRITE_WIDTH, SPRITE_HEIGHT):
        if not options.allow_nearest_resize:
            return _failed_import(base_item, [f"expected image size (32, 32), got {rgba.size}."])
        original_size = rgba.size
        rgba = rgba.resize((SPRITE_WIDTH, SPRITE_HEIGHT), resample=Image.Resampling.NEAREST)
        warnings.append(f"resized to 32x32 with nearest neighbor from {original_size}.")

    alpha_values = np.asarray(rgba, dtype=np.uint8)[:, :, 3]
    soft_alpha_values = sorted(int(value) for value in np.unique(alpha_values) if int(value) not in {0, 255})
    if soft_alpha_values:
        return _failed_import(
            base_item,
            [f"alpha must be hard 0 or 255; found soft alpha values including {soft_alpha_values[:8]}."],
            warnings=warnings,
        )

    metadata = SpriteMetadata(
        id=base_item.sprite_id,
        category=base_item.category,
        source=str(source_path),
        license=base_item.license,
    )
    bundle: SpriteBundle | None = None
    try:
        bundle = encode_rgba_image_to_bundle(
            rgba,
            metadata,
            max_visible_colors=options.max_palette_slots,
            canonicalize_palette=options.canonicalize_palette,
            generate_role_map=options.infer_role_map,
        )
    except ValueError as exc:
        strict_error = str(exc)
        if not _looks_like_overcolor_error(strict_error):
            return _failed_import(base_item, [strict_error], warnings=warnings)
        if not (options.allow_quantize_overcolor and options.quantize_overcolor):
            return _failed_import(base_item, [strict_error], warnings=warnings)
        try:
            bundle = encode_rgba_image_to_quantized_bundle(
                rgba,
                metadata,
                options=QuantizationOptions(
                    target_visible_colors=options.max_palette_slots,
                    canonicalize_palette=options.canonicalize_palette,
                    generate_role_map=options.infer_role_map,
                ),
            )
        except ValueError as quantize_exc:
            return _failed_import(base_item, [f"quantization failed: {quantize_exc}"], warnings=warnings)
        original_count = bundle.metadata.extra.get("original_visible_color_count")
        quantized_count = bundle.metadata.extra.get("quantized_visible_color_count")
        warnings.append(f"quantized over-color sprite from {original_count} to {quantized_count} visible colors.")

    errors = validate_bundle(bundle)
    if errors:
        return _failed_import(base_item, errors, warnings=warnings)

    palette_size = visible_palette_size(np.asarray(bundle.palette))
    item = DatasetMakerItem(
        sprite_id=base_item.sprite_id,
        source_path=source_path,
        status="accepted",
        category=default_category,
        tags=tuple(default_tags),
        source_name=source_path.name,
        palette_size=palette_size,
        has_role_map=bundle.role_map is not None,
    )
    preview_image, alpha_preview_image, role_preview_image, palette_strip_image = _preview_images(bundle)
    return ImportedSprite(
        item=item,
        bundle=bundle,
        preview_image=preview_image,
        alpha_preview_image=alpha_preview_image,
        role_preview_image=role_preview_image,
        palette_strip_image=palette_strip_image,
        errors=(),
        warnings=tuple(warnings),
    )


def import_png_directory(
    root: str | Path,
    *,
    options: ImportOptions,
    default_category: str = "unknown",
    default_tags: Sequence[str] = (),
) -> list[ImportedSprite]:
    """Import deterministically sorted PNG files from a directory."""

    root_path = Path(root)
    if not root_path.exists():
        item = DatasetMakerItem(
            sprite_id=normalize_sprite_id(root_path.stem),
            source_path=root_path,
            status="rejected",
            category=default_category,
            tags=tuple(default_tags),
            source_name=root_path.name,
        )
        return [_failed_import(item, [f"directory not found: {root_path}."])]
    if not root_path.is_dir():
        return [
            import_png_as_dataset_item(
                root_path,
                options=options,
                default_category=default_category,
                default_tags=default_tags,
            )
        ]

    iterator = root_path.rglob("*.png") if options.recursive else root_path.glob("*.png")
    paths = sorted(
        (path for path in iterator if path.is_file() and not _is_hidden_path(path, root_path)),
        key=lambda value: value.relative_to(root_path).as_posix().lower(),
    )
    return [
        import_png_as_dataset_item(
            path,
            options=options,
            default_category=default_category,
            default_tags=default_tags,
        )
        for path in paths
    ]


def _failed_import(
    item: DatasetMakerItem,
    errors: Sequence[str],
    *,
    warnings: Sequence[str] = (),
) -> ImportedSprite:
    return ImportedSprite(
        item=item,
        bundle=None,
        preview_image=None,
        alpha_preview_image=None,
        role_preview_image=None,
        palette_strip_image=None,
        errors=tuple(str(error) for error in errors),
        warnings=tuple(str(warning) for warning in warnings),
    )


def _looks_like_overcolor_error(error: str) -> bool:
    return "visible colors" in error and "above max_visible_colors" in error


def _preview_images(bundle: SpriteBundle) -> tuple[Image.Image, Image.Image, Image.Image | None, Image.Image]:
    preview = make_preview(reconstruct_rgba(bundle), scale=8)
    alpha_preview = _alpha_preview(bundle.alpha)
    role_preview = role_map_to_preview_image(bundle.role_map, scale=8) if bundle.role_map is not None else None
    palette_strip = _palette_strip(bundle.palette)
    return preview, alpha_preview, role_preview, palette_strip


def _alpha_preview(alpha: np.ndarray) -> Image.Image:
    pixels = np.asarray(alpha, dtype=np.uint8) * 255
    image = Image.fromarray(pixels, mode="L").convert("RGBA")
    return image.resize((SPRITE_WIDTH * 8, SPRITE_HEIGHT * 8), resample=Image.Resampling.NEAREST)


def _palette_strip(palette: np.ndarray, *, swatch_size: int = 16) -> Image.Image:
    rows = np.asarray(palette, dtype=np.uint8)
    width = max(1, int(rows.shape[0])) * swatch_size
    image = Image.new("RGBA", (width, swatch_size), (0, 0, 0, 0))
    for index, color in enumerate(rows):
        red, green, blue = (int(channel) for channel in color)
        fill = (red, green, blue, 255 if index > 0 else 96)
        for y in range(swatch_size):
            for x in range(index * swatch_size, (index + 1) * swatch_size):
                image.putpixel((x, y), fill)
    return image


def _is_hidden_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)
