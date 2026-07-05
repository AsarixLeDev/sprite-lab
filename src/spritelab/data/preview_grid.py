"""Preview grid generation for SpriteBundle datasets."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from spritelab.codec.io import load_bundle
from spritelab.codec.reconstruct import reconstruct_rgba
from spritelab.data.manifest import IngestedSpriteRecord, load_manifest
from spritelab.utils.image import ensure_rgba

SPRITE_SIZE = (32, 32)
SORT_FIELDS = {"id", "palette_size", "source_path", "split"}


@dataclass(frozen=True)
class PreviewGridOptions:
    dataset_path: Path
    output_path: Path
    scale: int = 8
    columns: int = 8
    padding: int = 4
    background: tuple[int, int, int, int] = (32, 32, 32, 255)
    label: bool = True
    label_height: int = 12
    max_items: int | None = None
    sort_by: Literal["id", "palette_size", "source_path", "split"] = "id"
    descending: bool = False
    filter_category: str | None = None
    filter_split: str | None = None
    filter_id_contains: str | None = None
    min_palette_size: int | None = None
    max_palette_size: int | None = None


@dataclass(frozen=True)
class PreviewGridRecord:
    id: str
    bundle_dir: Path
    image_path: Path | None
    category: str | None
    split: str | None
    palette_size: int | None
    source_path: str | None = None


def load_preview_records(dataset_path: str | Path) -> list[PreviewGridRecord]:
    """Load preview records from a dataset directory, manifest, or bundles directory."""

    path = Path(dataset_path)
    if path.is_file():
        if path.name != "manifest.json":
            raise ValueError(f"expected manifest.json file, got {path}.")
        return _records_from_manifest(path)

    if not path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"dataset path is not a directory: {path}")

    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        return _records_from_manifest(manifest_path)

    return _records_from_bundles_dir(path)


def filter_preview_records(
    records: list[PreviewGridRecord],
    *,
    filter_category: str | None = None,
    filter_split: str | None = None,
    filter_id_contains: str | None = None,
    min_palette_size: int | None = None,
    max_palette_size: int | None = None,
) -> list[PreviewGridRecord]:
    """Return records matching the provided filters."""

    filtered = records
    if filter_category is not None:
        filtered = [record for record in filtered if record.category == filter_category]
    if filter_split is not None:
        filtered = [record for record in filtered if record.split == filter_split]
    if filter_id_contains is not None:
        needle = filter_id_contains.lower()
        filtered = [record for record in filtered if needle in record.id.lower()]
    if min_palette_size is not None:
        filtered = [
            record
            for record in filtered
            if record.palette_size is not None and record.palette_size >= min_palette_size
        ]
    if max_palette_size is not None:
        filtered = [
            record
            for record in filtered
            if record.palette_size is not None and record.palette_size <= max_palette_size
        ]
    return filtered


def sort_preview_records(
    records: list[PreviewGridRecord],
    *,
    sort_by: str = "id",
    descending: bool = False,
) -> list[PreviewGridRecord]:
    """Return deterministically sorted records."""

    if sort_by not in SORT_FIELDS:
        allowed = ", ".join(sorted(SORT_FIELDS))
        raise ValueError(f"sort_by must be one of: {allowed}.")

    return sorted(records, key=lambda record: _sort_key(record, sort_by), reverse=descending)


def load_record_image(record: PreviewGridRecord) -> Image.Image:
    """Load a record's exact 32x32 RGBA sprite image."""

    reconstructed_path = record.bundle_dir / "reconstructed.png"
    if reconstructed_path.exists():
        image = _try_load_exact_sprite(reconstructed_path)
        if image is not None:
            return image

    if (record.bundle_dir / "bundle.npz").exists() and (record.bundle_dir / "metadata.json").exists():
        return reconstruct_rgba(load_bundle(record.bundle_dir))

    preview_path = record.bundle_dir / "preview_8x.png"
    if preview_path.exists():
        with Image.open(preview_path) as preview:
            rgba = ensure_rgba(preview)
            if rgba.size[0] % 32 != 0 or rgba.size[1] % 32 != 0:
                raise ValueError(f"preview fallback is not an integer 32x32 scale: {preview_path}")
            return rgba.resize(SPRITE_SIZE, resample=Image.Resampling.NEAREST)

    raise FileNotFoundError(f"could not load sprite image for record {record.id} from {record.bundle_dir}")


def make_preview_grid(
    records: list[PreviewGridRecord],
    *,
    scale: int = 8,
    columns: int = 8,
    padding: int = 4,
    background: tuple[int, int, int, int] = (32, 32, 32, 255),
    label: bool = True,
    label_height: int = 12,
) -> Image.Image:
    """Create a contact-sheet PNG from 32x32 sprite records."""

    if scale < 1:
        raise ValueError("scale must be at least 1.")
    if columns < 1:
        raise ValueError("columns must be at least 1.")
    if padding < 0:
        raise ValueError("padding must be non-negative.")
    if label_height < 0:
        raise ValueError("label_height must be non-negative.")
    if not records:
        raise ValueError("cannot create preview grid with no records.")

    cell_sprite_size = 32 * scale
    actual_label_height = label_height if label else 0
    cell_width = cell_sprite_size + padding
    cell_height = cell_sprite_size + padding + actual_label_height
    rows = math.ceil(len(records) / columns)
    grid_width = columns * cell_width + padding
    grid_height = rows * cell_height + padding

    grid = Image.new("RGBA", (grid_width, grid_height), background)
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for index, record in enumerate(records):
        row = index // columns
        column = index % columns
        x = padding + column * cell_width
        y = padding + row * cell_height

        sprite = load_record_image(record)
        scaled = sprite.resize((cell_sprite_size, cell_sprite_size), resample=Image.Resampling.NEAREST)
        grid.alpha_composite(scaled, dest=(x, y))

        if label:
            text = _truncate_label(_label_for_record(record), max_width=cell_sprite_size, font=font)
            draw.text((x, y + cell_sprite_size + 1), text, fill=(235, 235, 235, 255), font=font)

    return grid


def create_preview_grid(options: PreviewGridOptions) -> Image.Image:
    """Load, filter, sort, render, save, and return a preview grid image."""

    records = load_preview_records(options.dataset_path)
    records = filter_preview_records(
        records,
        filter_category=options.filter_category,
        filter_split=options.filter_split,
        filter_id_contains=options.filter_id_contains,
        min_palette_size=options.min_palette_size,
        max_palette_size=options.max_palette_size,
    )
    records = sort_preview_records(records, sort_by=options.sort_by, descending=options.descending)
    if options.max_items is not None:
        records = records[: options.max_items]

    loadable_records, warnings = _loadable_records(records)
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")
    if not loadable_records:
        raise ValueError("no loadable records after filtering.")

    grid = make_preview_grid(
        loadable_records,
        scale=options.scale,
        columns=options.columns,
        padding=options.padding,
        background=options.background,
        label=options.label,
        label_height=options.label_height,
    )
    output_path = Path(options.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return grid


def _records_from_manifest(manifest_path: Path) -> list[PreviewGridRecord]:
    manifest = load_manifest(manifest_path)
    return [
        _record_from_manifest_record(record, manifest_path=manifest_path)
        for record in manifest.records
    ]


def _record_from_manifest_record(record: IngestedSpriteRecord, *, manifest_path: Path) -> PreviewGridRecord:
    bundle_dir = _resolve_bundle_dir(record, manifest_path=manifest_path)
    image_path = bundle_dir / "reconstructed.png"
    return PreviewGridRecord(
        id=record.id,
        bundle_dir=bundle_dir,
        image_path=image_path if image_path.exists() else None,
        category=record.category,
        split=record.split,
        palette_size=record.palette_size,
        source_path=record.source_path,
    )


def _resolve_bundle_dir(record: IngestedSpriteRecord, *, manifest_path: Path) -> Path:
    raw = Path(record.bundle_dir)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw

    manifest_dir = manifest_path.parent
    candidates = [
        manifest_dir / raw,
        manifest_dir / "bundles" / record.id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _records_from_bundles_dir(bundles_dir: Path) -> list[PreviewGridRecord]:
    records: list[PreviewGridRecord] = []
    for bundle_dir in sorted(bundles_dir.iterdir(), key=lambda path: path.name.lower()):
        if not bundle_dir.is_dir():
            continue
        if not (bundle_dir / "bundle.npz").exists() or not (bundle_dir / "metadata.json").exists():
            continue

        metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        image_path = bundle_dir / "reconstructed.png"
        records.append(
            PreviewGridRecord(
                id=str(metadata.get("id") or bundle_dir.name),
                bundle_dir=bundle_dir,
                image_path=image_path if image_path.exists() else None,
                category=metadata.get("category"),
                split=metadata.get("extra", {}).get("split"),
                palette_size=metadata.get("palette_size"),
                source_path=metadata.get("source"),
            )
        )
    return records


def _try_load_exact_sprite(path: Path) -> Image.Image | None:
    with Image.open(path) as image:
        rgba = ensure_rgba(image)
        if rgba.size != SPRITE_SIZE:
            return None
        return rgba.copy()


def _sort_key(record: PreviewGridRecord, sort_by: str) -> tuple[bool, object, str]:
    value = getattr(record, sort_by)
    normalized: object
    if isinstance(value, str):
        normalized = value.lower()
    else:
        normalized = value
    return (value is None, normalized if normalized is not None else "", record.id)


def _label_for_record(record: PreviewGridRecord) -> str:
    parts = [record.id]
    if record.palette_size is not None:
        parts.append(f"p={record.palette_size}")
    if record.split:
        parts.append(record.split)
    return " | ".join(parts)


def _truncate_label(text: str, *, max_width: int, font: ImageFont.ImageFont) -> str:
    if _text_width(text, font) <= max_width:
        return text

    suffix = "..."
    available = max(0, max_width - _text_width(suffix, font))
    truncated = ""
    for char in text:
        if _text_width(truncated + char, font) > available:
            break
        truncated += char
    return truncated.rstrip() + suffix


def _text_width(text: str, font: ImageFont.ImageFont) -> int:
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0])


def _loadable_records(records: list[PreviewGridRecord]) -> tuple[list[PreviewGridRecord], list[str]]:
    loadable: list[PreviewGridRecord] = []
    warnings: list[str] = []
    for record in records:
        try:
            load_record_image(record)
        except Exception as exc:
            warnings.append(f"skipping {record.id}: {exc}")
        else:
            loadable.append(record)
    return loadable, warnings


def _parse_args() -> PreviewGridOptions:
    parser = argparse.ArgumentParser(description="Create a PNG preview grid from a SpriteBundle dataset.")
    parser.add_argument("--dataset", required=True, dest="dataset_path", type=Path)
    parser.add_argument("--output", required=True, dest="output_path", type=Path)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--no-label", action="store_false", dest="label")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--sort-by", choices=sorted(SORT_FIELDS), default="id")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--category", dest="filter_category")
    parser.add_argument("--split", dest="filter_split")
    parser.add_argument("--id-contains", dest="filter_id_contains")
    parser.add_argument("--min-palette-size", type=int)
    parser.add_argument("--max-palette-size", type=int)
    args = parser.parse_args()

    return PreviewGridOptions(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        scale=args.scale,
        columns=args.columns,
        padding=args.padding,
        label=args.label,
        max_items=args.max_items,
        sort_by=args.sort_by,
        descending=args.descending,
        filter_category=args.filter_category,
        filter_split=args.filter_split,
        filter_id_contains=args.filter_id_contains,
        min_palette_size=args.min_palette_size,
        max_palette_size=args.max_palette_size,
    )


def main() -> None:
    options = _parse_args()
    loaded = load_preview_records(options.dataset_path)
    filtered = filter_preview_records(
        loaded,
        filter_category=options.filter_category,
        filter_split=options.filter_split,
        filter_id_contains=options.filter_id_contains,
        min_palette_size=options.min_palette_size,
        max_palette_size=options.max_palette_size,
    )
    sorted_records = sort_preview_records(filtered, sort_by=options.sort_by, descending=options.descending)
    if options.max_items is not None:
        sorted_records = sorted_records[: options.max_items]

    loadable_records, warnings = _loadable_records(sorted_records)
    for warning in warnings:
        print(f"Warning: {warning}")
    if not loadable_records:
        raise ValueError("no loadable records after filtering.")

    grid = make_preview_grid(
        loadable_records,
        scale=options.scale,
        columns=options.columns,
        padding=options.padding,
        background=options.background,
        label=options.label,
        label_height=options.label_height,
    )
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(options.output_path)

    rows = math.ceil(len(loadable_records) / options.columns)
    print(f"Loaded records: {len(loaded)}")
    print(f"After filters: {len(sorted_records)}")
    print(f"Grid: {options.columns} columns x {rows} rows")
    print(f"Output: {options.output_path}")


if __name__ == "__main__":
    main()
