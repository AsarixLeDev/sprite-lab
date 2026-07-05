"""Batch ingestion for already-clean 32x32 PNG sprite folders."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.codec.bundle import SpriteBundle, SpriteMetadata
from spritelab.codec.encode import encode_png_to_bundle
from spritelab.codec.io import save_bundle
from spritelab.codec.quantize import QuantizationOptions, encode_png_to_quantized_bundle
from spritelab.data.ids import make_sprite_id, sha256_file, short_path_hash
from spritelab.data.manifest import (
    DatasetManifest,
    IngestedSpriteRecord,
    RejectedSpriteRecord,
    save_manifest,
    save_rejected_report,
)


@dataclass(frozen=True)
class IngestOptions:
    input_dir: Path
    output_dir: Path
    category: str | None = None
    subtype: str | None = None
    license: str | None = None
    recursive: bool = True
    alpha_threshold: int = 128
    max_visible_colors: int = 32
    canonicalize_palette: bool = True
    generate_role_map: bool = True
    skip_existing: bool = False
    write_reconstructed: bool = True
    write_preview: bool = True
    create_split: bool = False
    split_seed: int = 12345
    quantize_over_color: bool = False
    target_visible_colors: int = 16
    quantization_seed: int = 12345
    quantization_max_iterations: int = 32


def find_png_files(input_dir: str | Path, recursive: bool = True) -> list[Path]:
    """Return deterministically sorted PNG files from ``input_dir``."""

    root = Path(input_dir)
    candidates = root.rglob("*") if recursive else root.glob("*")
    files = [path for path in candidates if path.is_file() and path.suffix.lower() == ".png"]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().lower())


def ingest_clean_png_folder(options: IngestOptions) -> DatasetManifest:
    """Batch-encode clean 32x32 PNGs into SpriteBundle directories."""

    input_dir = Path(options.input_dir)
    output_dir = Path(options.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir = output_dir / "bundles"
    previews_dir = output_dir / "previews"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    if options.write_preview:
        previews_dir.mkdir(parents=True, exist_ok=True)

    png_files = find_png_files(input_dir, recursive=options.recursive)
    ids = _resolve_unique_ids(png_files, root=input_dir)
    records: list[IngestedSpriteRecord] = []
    rejected: list[RejectedSpriteRecord] = []
    quantized_count = 0

    for source_path in png_files:
        sprite_id = ids[source_path]
        bundle_dir = bundles_dir / sprite_id

        try:
            source_sha = sha256_file(source_path)
            metadata = SpriteMetadata(
                id=sprite_id,
                category=options.category,
                subtype=options.subtype,
                source=str(source_path),
                license=options.license,
            )

            if not options.skip_existing or not _bundle_exists(bundle_dir):
                bundle = _encode_png_for_ingestion(source_path, metadata=metadata, options=options)
                save_bundle(
                    bundle,
                    bundle_dir,
                    write_reconstructed=options.write_reconstructed,
                    write_preview=options.write_preview,
                )
            else:
                bundle = _encode_png_for_ingestion(source_path, metadata=metadata, options=options)

            if bool(bundle.metadata.extra.get("quantized")):
                quantized_count += 1

            if options.write_preview:
                _copy_preview(bundle_dir, previews_dir, sprite_id)

            records.append(
                IngestedSpriteRecord(
                    id=sprite_id,
                    source_path=str(source_path),
                    bundle_dir=str(bundle_dir),
                    width=bundle.metadata.width,
                    height=bundle.metadata.height,
                    category=bundle.metadata.category,
                    subtype=bundle.metadata.subtype,
                    license=bundle.metadata.license,
                    palette_size=int(bundle.metadata.palette_size or 0),
                    sha256=source_sha,
                )
            )
        except Exception as exc:
            rejected.append(RejectedSpriteRecord(source_path=str(source_path), reason=str(exc)))

    if options.create_split:
        records = _with_deterministic_splits(records, seed=options.split_seed)

    manifest_options = _options_for_manifest(options)
    manifest_options["quantized_count"] = quantized_count
    manifest = DatasetManifest(
        dataset_name=output_dir.name,
        records=records,
        rejected_count=len(rejected),
        total_seen=len(png_files),
        options=manifest_options,
    )
    save_manifest(manifest, output_dir / "manifest.json")
    save_rejected_report(rejected, output_dir / "rejected.json")

    return manifest


def _encode_png_for_ingestion(
    source_path: Path,
    *,
    metadata: SpriteMetadata,
    options: IngestOptions,
) -> SpriteBundle:
    try:
        return encode_png_to_bundle(
            source_path,
            metadata=metadata,
            alpha_threshold=options.alpha_threshold,
            max_visible_colors=options.max_visible_colors,
            canonicalize_palette=options.canonicalize_palette,
            generate_role_map=options.generate_role_map,
        )
    except ValueError as strict_error:
        if not options.quantize_over_color or not _is_too_many_colors_error(strict_error):
            raise

        quantization_options = QuantizationOptions(
            target_visible_colors=options.target_visible_colors,
            max_iterations=options.quantization_max_iterations,
            seed=options.quantization_seed,
            canonicalize_palette=options.canonicalize_palette,
            generate_role_map=options.generate_role_map,
            alpha_threshold=options.alpha_threshold,
        )
        try:
            return encode_png_to_quantized_bundle(
                source_path,
                metadata=metadata,
                options=quantization_options,
            )
        except Exception as quantization_error:
            raise ValueError(
                "strict encoder failed due to too many colors; "
                f"quantization failed: {quantization_error}"
            ) from quantization_error


def _is_too_many_colors_error(error: Exception) -> bool:
    message = str(error)
    return "visible colors" in message and "above max_visible_colors" in message


def _resolve_unique_ids(paths: list[Path], *, root: Path) -> dict[Path, str]:
    base_ids: dict[Path, str] = {path: make_sprite_id(path, root=root) for path in paths}
    counts: dict[str, int] = {}
    for sprite_id in base_ids.values():
        counts[sprite_id] = counts.get(sprite_id, 0) + 1

    resolved: dict[Path, str] = {}
    for path in paths:
        sprite_id = base_ids[path]
        if counts[sprite_id] > 1:
            sprite_id = f"{sprite_id}_{short_path_hash(path, root=root)}"
        resolved[path] = sprite_id
    return resolved


def _bundle_exists(bundle_dir: Path) -> bool:
    return (bundle_dir / "bundle.npz").exists() and (bundle_dir / "metadata.json").exists()


def _copy_preview(bundle_dir: Path, previews_dir: Path, sprite_id: str) -> None:
    source = bundle_dir / "preview_8x.png"
    if source.exists():
        shutil.copy2(source, previews_dir / f"{sprite_id}_preview_8x.png")


def _with_deterministic_splits(records: list[IngestedSpriteRecord], *, seed: int) -> list[IngestedSpriteRecord]:
    rng = random.Random(seed)
    shuffled_ids = [record.id for record in records]
    rng.shuffle(shuffled_ids)

    total = len(records)
    train_count = int(total * 0.90)
    val_count = int(total * 0.05)
    split_by_id: dict[str, str] = {}

    for index, sprite_id in enumerate(shuffled_ids):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        split_by_id[sprite_id] = split

    return [
        IngestedSpriteRecord(
            id=record.id,
            source_path=record.source_path,
            bundle_dir=record.bundle_dir,
            width=record.width,
            height=record.height,
            category=record.category,
            subtype=record.subtype,
            license=record.license,
            palette_size=record.palette_size,
            sha256=record.sha256,
            split=split_by_id[record.id],
        )
        for record in records
    ]


def _options_for_manifest(options: IngestOptions) -> dict[str, Any]:
    data = asdict(options)
    data["input_dir"] = str(options.input_dir)
    data["output_dir"] = str(options.output_dir)
    return data


def _parse_args() -> IngestOptions:
    parser = argparse.ArgumentParser(description="Ingest clean 32x32 PNG sprites into SpriteBundles.")
    parser.add_argument("--input", required=True, dest="input_dir", type=Path)
    parser.add_argument("--output", required=True, dest="output_dir", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--subtype")
    parser.add_argument("--license")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    parser.add_argument("--alpha-threshold", type=int, default=128)
    parser.add_argument("--max-visible-colors", type=int, default=32)
    parser.add_argument("--no-canonicalize", action="store_false", dest="canonicalize_palette")
    parser.add_argument("--no-role-map", action="store_false", dest="generate_role_map")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--quantize-over-color", action="store_true")
    parser.add_argument("--target-visible-colors", type=int, default=16)
    parser.add_argument("--quantization-seed", type=int, default=12345)
    parser.add_argument("--quantization-max-iterations", type=int, default=32)
    args = parser.parse_args()

    return IngestOptions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        category=args.category,
        subtype=args.subtype,
        license=args.license,
        recursive=args.recursive,
        alpha_threshold=args.alpha_threshold,
        max_visible_colors=args.max_visible_colors,
        canonicalize_palette=args.canonicalize_palette,
        generate_role_map=args.generate_role_map,
        skip_existing=args.skip_existing,
        create_split=args.create_split,
        split_seed=args.split_seed,
        quantize_over_color=args.quantize_over_color,
        target_visible_colors=args.target_visible_colors,
        quantization_seed=args.quantization_seed,
        quantization_max_iterations=args.quantization_max_iterations,
    )


def main() -> None:
    options = _parse_args()
    manifest = ingest_clean_png_folder(options)
    output_dir = Path(options.output_dir)

    print(f"Seen: {manifest.total_seen} PNGs")
    print(f"Encoded: {len(manifest.records)}")
    print(f"Rejected: {manifest.rejected_count}")
    print(f"Quantized: {manifest.options.get('quantized_count', 0)}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Rejected report: {output_dir / 'rejected.json'}")


if __name__ == "__main__":
    main()
