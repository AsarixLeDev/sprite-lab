"""Canonicalize generated RGBA samples into strict sprite artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.codec.quantize import QuantizationOptions, quantize_rgba_image_to_palette_indices

SPRITE_SIZE = 32
TRANSPARENT_INDEX = 0


@dataclass(frozen=True)
class GeneratedSprite:
    """A generated sprite canonicalized into hard-alpha indexed form.

    ``max_colors`` is interpreted as the maximum number of visible colors.
    Palette row 0 is the project-standard transparent dummy slot, so the
    palette and mask contain ``max_colors + 1`` rows.
    """

    rgba_raw: np.ndarray
    rgba_hard: np.ndarray
    index_map: np.ndarray
    palette: np.ndarray
    palette_mask: np.ndarray
    visible_color_count: int
    alpha_opaque_count: int
    warnings: tuple[str, ...]


def canonicalize_generated_rgba(
    rgba: np.ndarray,
    *,
    max_colors: int = 32,
    alpha_threshold: float = 0.5,
    dither: bool = False,
) -> GeneratedSprite:
    """Convert continuous generated RGBA to hard-alpha indexed sprite data."""

    if max_colors < 1:
        raise ValueError("max_colors must be at least 1 visible color")
    if max_colors > 255:
        raise ValueError("max_colors must be at most 255 for uint8 index maps")
    if not 0.0 <= float(alpha_threshold) <= 1.0:
        raise ValueError("alpha_threshold must be in 0..1")

    warnings: list[str] = []
    if dither:
        warnings.append("dither requested but generated canonicalizer v0 uses no dithering")

    rgba_raw = _normalize_rgba_hwc(rgba)
    alpha_hard = (rgba_raw[..., 3] >= float(alpha_threshold)).astype(np.float32)
    rgba_hard = rgba_raw.copy()
    rgba_hard[..., 3] = alpha_hard
    rgba_hard[alpha_hard == 0.0, :3] = 0.0

    alpha_opaque_count = int(alpha_hard.sum())
    rows = int(max_colors) + 1
    if alpha_opaque_count == 0:
        warnings.append("generated sprite is fully transparent after alpha thresholding")
        palette = np.zeros((rows, 4), dtype=np.float32)
        palette[0, 3] = 0.0
        palette_mask = np.zeros((rows,), dtype=bool)
        palette_mask[0] = True
        return GeneratedSprite(
            rgba_raw=rgba_raw,
            rgba_hard=rgba_hard,
            index_map=np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=np.uint8),
            palette=palette,
            palette_mask=palette_mask,
            visible_color_count=0,
            alpha_opaque_count=0,
            warnings=tuple(warnings),
        )

    hard_image = _float_rgba_to_image(rgba_hard)
    result = quantize_rgba_image_to_palette_indices(
        hard_image,
        options=QuantizationOptions(
            target_visible_colors=int(max_colors),
            alpha_threshold=128,
            preserve_exact_if_under_limit=True,
            canonicalize_palette=False,
            generate_role_map=False,
        ),
    )
    visible_count = int(result.quantized_visible_color_count)
    if visible_count > max_colors:
        raise ValueError(f"quantized visible color count {visible_count} exceeds max_colors={max_colors}")

    palette = np.zeros((rows, 4), dtype=np.float32)
    palette_mask = np.zeros((rows,), dtype=bool)
    palette_rgb = np.asarray(result.palette, dtype=np.float32) / 255.0
    palette[: palette_rgb.shape[0], :3] = palette_rgb
    palette[0, 3] = 0.0
    if visible_count:
        palette[1 : visible_count + 1, 3] = 1.0
    palette_mask[: visible_count + 1] = True

    return GeneratedSprite(
        rgba_raw=rgba_raw,
        rgba_hard=rgba_hard.astype(np.float32, copy=False),
        index_map=np.asarray(result.index_map, dtype=np.uint8),
        palette=palette,
        palette_mask=palette_mask,
        visible_color_count=visible_count,
        alpha_opaque_count=alpha_opaque_count,
        warnings=tuple(warnings),
    )


def reconstruct_indexed_rgba(*, index_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Reconstruct HWC float RGBA from an index map and RGBA palette."""

    index = np.asarray(index_map)
    pal = np.asarray(palette, dtype=np.float32)
    if index.shape != (SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"index_map must have shape [32, 32], got {index.shape}")
    if pal.ndim != 2 or pal.shape[1] != 4:
        raise ValueError(f"palette must have shape [K, 4], got {pal.shape}")
    if index.size and (int(index.min()) < 0 or int(index.max()) >= pal.shape[0]):
        raise ValueError("index_map contains values outside the palette range")
    return pal[index.astype(np.int64, copy=False)].astype(np.float32, copy=False)


def write_generated_sprite_artifacts(
    sprite: GeneratedSprite,
    out_dir: Path,
    sample_id: str,
    metadata: Mapping[str, Any],
    *,
    write_raw_rgba: bool = True,
    write_hard_rgba: bool = True,
) -> dict[str, Any]:
    """Write one generated sprite's PNG artifacts and return its manifest record."""

    safe_id = _safe_sample_id(sample_id)
    out_dir = Path(out_dir)
    paths: dict[str, str] = {}
    if write_raw_rgba:
        rel = Path("raw_rgba") / f"{safe_id}.png"
        _save_rgba_png(sprite.rgba_raw, out_dir / rel)
        paths["raw_rgba"] = rel.as_posix()
    if write_hard_rgba:
        rel = Path("hard_rgba") / f"{safe_id}.png"
        _save_rgba_png(sprite.rgba_hard, out_dir / rel)
        paths["hard_rgba"] = rel.as_posix()

    indexed_rel = Path("indexed_png") / f"{safe_id}.png"
    _save_indexed_png(sprite, out_dir / indexed_rel)
    paths["indexed_png"] = indexed_rel.as_posix()

    record = {
        "sample_id": safe_id,
        "width": SPRITE_SIZE,
        "height": SPRITE_SIZE,
        "alpha_threshold": float(metadata.get("alpha_threshold", 0.5)),
        "max_colors": int(metadata.get("max_colors", max(0, int(sprite.palette.shape[0]) - 1))),
        "visible_color_count": int(sprite.visible_color_count),
        "alpha_opaque_count": int(sprite.alpha_opaque_count),
        "paths": paths,
        "warnings": list(sprite.warnings),
    }
    for key, value in metadata.items():
        if key not in record:
            record[key] = _jsonable(value)
    return record


def write_generation_reports(
    *,
    out_dir: Path,
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    contact_sheet: str | None,
) -> dict[str, Any]:
    """Write generation manifest plus JSON and Markdown reports."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "generated_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    warnings = sum(len(record.get("warnings") or []) for record in records)
    fully_transparent = sum(1 for record in records if int(record.get("alpha_opaque_count", 0)) == 0)
    color_counts = [int(record.get("visible_color_count", 0)) for record in records]
    report = {
        "sample_count": len(records),
        "warnings": warnings,
        "fully_transparent_count": fully_transparent,
        "max_visible_color_count": max(color_counts) if color_counts else 0,
        "contact_sheet": contact_sheet,
        "manifest": "generated_manifest.jsonl",
        "config": dict(config),
    }
    (out_dir / "generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "generation_report.md").write_text(_generation_report_md(report, records), encoding="utf-8")
    return report


def build_generation_contact_sheet(
    generated_dir: Path,
    records: list[Mapping[str, Any]],
    out_path: Path,
    *,
    include_raw: bool = True,
    max_items: int = 64,
    scale: int = 4,
    columns: int = 4,
) -> Path | None:
    """Write a contact sheet of raw/hard/indexed generated outputs."""

    if not records:
        return None

    rows = records[: max(0, int(max_items))]
    if not rows:
        return None

    tile_images: list[list[Image.Image]] = []
    for record in rows:
        paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
        images: list[Image.Image] = []
        for key in ("raw_rgba",) if include_raw else ():
            rel = paths.get(key)
            if rel:
                images.append(Image.open(Path(generated_dir) / str(rel)).convert("RGBA"))
        for key in ("hard_rgba", "indexed_png"):
            rel = paths.get(key)
            if rel:
                images.append(Image.open(Path(generated_dir) / str(rel)).convert("RGBA"))
        if images:
            tile_images.append(images)

    if not tile_images:
        return None

    subtiles = max(len(images) for images in tile_images)
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    columns = max(1, int(columns))
    tile_w = subtiles * cell + (subtiles - 1) * padding
    tile_h = cell
    sheet_rows = (len(tile_images) + columns - 1) // columns
    sheet = Image.new(
        "RGBA",
        (columns * tile_w + (columns + 1) * padding, sheet_rows * tile_h + (sheet_rows + 1) * padding),
        (36, 36, 40, 255),
    )
    for index, images in enumerate(tile_images):
        col = index % columns
        row = index // columns
        left = padding + col * (tile_w + padding)
        top = padding + row * (tile_h + padding)
        for sub_index, image in enumerate(images):
            sheet.alpha_composite(image.resize((cell, cell), Image.NEAREST), (left + sub_index * (cell + padding), top))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _normalize_rgba_hwc(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba)
    if arr.shape == (4, SPRITE_SIZE, SPRITE_SIZE):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape != (SPRITE_SIZE, SPRITE_SIZE, 4):
        raise ValueError(f"rgba must have shape [4, 32, 32] or [32, 32, 4], got {arr.shape}")
    value = arr.astype(np.float32, copy=False)
    if value.size and (arr.dtype.kind in "ui" or float(np.nanmax(value)) > 1.0):
        value = value / 255.0
    return np.clip(value, 0.0, 1.0).astype(np.float32, copy=False)


def _float_rgba_to_image(rgba_hwc: np.ndarray) -> Image.Image:
    value = np.asarray(rgba_hwc, dtype=np.float32)
    if value.shape != (SPRITE_SIZE, SPRITE_SIZE, 4):
        raise ValueError(f"rgba image must have shape [32, 32, 4], got {value.shape}")
    return Image.fromarray(np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")


def _save_rgba_png(rgba_hwc: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _float_rgba_to_image(rgba_hwc).save(path)


def _save_indexed_png(sprite: GeneratedSprite, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index = np.asarray(sprite.index_map, dtype=np.uint8)
    image = Image.fromarray(index, mode="P")
    rgb = np.rint(np.clip(sprite.palette[:, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    flat_palette = rgb.reshape(-1).tolist()
    flat_palette.extend([0] * (256 * 3 - len(flat_palette)))
    image.putpalette(flat_palette[: 256 * 3])
    image.info["transparency"] = TRANSPARENT_INDEX
    image.save(path)


def _safe_sample_id(sample_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(sample_id).strip())
    if not cleaned:
        raise ValueError("sample_id must be non-empty")
    return cleaned


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _generation_report_md(report: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str:
    lines = [
        "# Generated Sprite Report",
        "",
        f"Samples: {int(report.get('sample_count', 0))}",
        f"Warnings: {int(report.get('warnings', 0))}",
        f"Fully transparent: {int(report.get('fully_transparent_count', 0))}",
        f"Max visible colors: {int(report.get('max_visible_color_count', 0))}",
        f"Contact sheet: `{report.get('contact_sheet') or ''}`",
        "",
        "## Prompts",
    ]
    for record in records[:100]:
        prompt_id = str(record.get("prompt_id", ""))
        prompt = str(record.get("prompt", ""))
        sample_id = str(record.get("sample_id", ""))
        colors = int(record.get("visible_color_count", 0))
        opaque = int(record.get("alpha_opaque_count", 0))
        lines.append(f"- `{sample_id}` `{prompt_id}` colors={colors} opaque={opaque}: {prompt}")
    if len(records) > 100:
        lines.append(f"- ... {len(records) - 100} more")
    lines.append("")
    return "\n".join(lines)
