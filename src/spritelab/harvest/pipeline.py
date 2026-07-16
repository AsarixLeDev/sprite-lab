"""End-to-end harvest pipeline: source -> candidates -> imports -> export."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from spritelab.dataset_maker.exporter import (
    DatasetMakerExportConfig,
    DatasetMakerExportResult,
    export_dataset_from_imported_sprites,
)
from spritelab.dataset_maker.importer import ImportedSprite, ImportOptions, import_png_as_dataset_item
from spritelab.dataset_maker.model import DatasetMakerItem, normalize_sprite_id
from spritelab.harvest.archive import archive_member_summary, extract_archive
from spritelab.harvest.autolabel import merge_auto_labels, suggest_metadata_from_path
from spritelab.harvest.download import compute_sha256, download_file
from spritelab.harvest.extract import HarvestCandidate, discover_png_candidates, filter_candidate_basic
from spritelab.harvest.sheet_mappings import metadata_for_sheet_cell
from spritelab.harvest.sheets import SheetSliceConfig, center_pad_to_32, looks_like_sprite_sheet, slice_sheet_to_pngs
from spritelab.harvest.sources import SourceRecord, is_license_allowed_for_training, utc_timestamp


@dataclass(frozen=True)
class HarvestImportOptions:
    max_palette_slots: int = 32
    allow_quantize_overcolor: bool = True
    quantize_overcolor: bool = True
    allow_nearest_resize: bool = False
    allow_center_pad_to_32: bool = True
    infer_role_map: bool = True
    canonicalize_palette: bool = True
    recursive: bool = True
    slice_sheets: bool = True
    sheet_config: SheetSliceConfig = field(default_factory=SheetSliceConfig)
    include_member_globs: tuple[str, ...] = ()
    exclude_member_globs: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarvestedSprite:
    source: SourceRecord
    candidate: HarvestCandidate
    imported: ImportedSprite
    auto_metadata: dict[str, Any]
    final_item: DatasetMakerItem


def harvest_source_to_imported_sprites(
    source: SourceRecord,
    *,
    options: HarvestImportOptions,
    work_dir: str | Path,
) -> list[HarvestedSprite]:
    """Resolve a source to PNGs, slice/pad as configured, and import each one."""

    work_dir = Path(work_dir)
    root, source = _resolve_source_root(source, work_dir, options=options)
    candidates = [
        filter_candidate_basic(candidate)
        for candidate in discover_png_candidates(root, source, recursive=options.recursive)
    ]

    final_pngs: list[tuple[HarvestCandidate, Path, list[str]]] = []
    sliced_dir = work_dir / "sliced"
    padded_dir = work_dir / "padded"
    for candidate in candidates:
        if candidate.status == "rejected":
            final_pngs.append((candidate, candidate.extracted_path, list(candidate.rejection_reasons)))
            continue
        size = (candidate.width, candidate.height)
        if size == (32, 32):
            final_pngs.append((candidate, candidate.extracted_path, []))
        elif (
            options.slice_sheets and options.sheet_config.enabled and looks_like_sprite_sheet(candidate.extracted_path)
        ):
            tile_dir = sliced_dir / candidate.candidate_id
            tiles = slice_sheet_to_pngs(candidate.extracted_path, tile_dir, options.sheet_config)
            for tile in tiles:
                if (
                    options.allow_center_pad_to_32
                    and options.sheet_config.tile_width <= 32
                    and options.sheet_config.tile_height <= 32
                ):
                    tile = center_pad_to_32(tile, padded_dir / f"{tile.stem}.png")
                final_pngs.append((candidate, tile, []))
        elif options.allow_center_pad_to_32 and candidate.width <= 32 and candidate.height <= 32:
            padded = center_pad_to_32(
                candidate.extracted_path,
                padded_dir / f"{candidate.candidate_id}.png",
            )
            final_pngs.append((candidate, padded, []))
        else:
            # Let the Dataset Maker importer handle it (resize or reject).
            final_pngs.append((candidate, candidate.extracted_path, []))

    import_options = ImportOptions(
        max_palette_slots=options.max_palette_slots,
        allow_quantize_overcolor=options.allow_quantize_overcolor,
        quantize_overcolor=options.quantize_overcolor,
        allow_nearest_resize=options.allow_nearest_resize,
        infer_role_map=options.infer_role_map,
        canonicalize_palette=options.canonicalize_palette,
    )

    harvested: list[HarvestedSprite] = []
    used_ids: set[str] = set()
    for candidate, png_path, extra_errors in final_pngs:
        imported = import_png_as_dataset_item(png_path, options=import_options)
        suggestion = suggest_metadata_from_path(candidate.relative_path, source.source_name)
        mapping = metadata_for_sheet_cell(source.source_id, candidate.relative_path, png_path)
        if mapping.get("mapping_excluded") == "true":
            extra_errors = [*extra_errors, "excluded by declarative sheet mapping"]
        item = _apply_source_metadata(imported.item, source, candidate, used_ids)
        item = merge_auto_labels(item, [suggestion])
        errors = tuple(imported.errors) + tuple(extra_errors)
        if errors and item.status == "accepted":
            item = _with_status(item, "rejected")
        harvested.append(
            HarvestedSprite(
                source=source,
                candidate=candidate,
                imported=replace(imported, item=item, errors=errors),
                auto_metadata={
                    "rule_suggestion": suggestion.__dict__ | {"tags": list(suggestion.tags)},
                    "import_options": {"allow_nearest_resize": options.allow_nearest_resize},
                    **({"sheet_mapping": mapping} if mapping else {}),
                },
                final_item=item,
            )
        )
    return harvested


@dataclass(frozen=True)
class HarvestPolicy:
    auto_accept_valid_cc0: bool = False
    auto_accept_own_work: bool = False
    auto_accept_allowlisted: bool = False
    quarantine_unknown_license: bool = True
    quarantine_low_qwen_confidence: bool = False
    qwen_confidence_threshold: float = 0.3
    reject_invalid: bool = True
    reject_too_empty: bool = False
    min_opaque_pixels: int = 8


def apply_harvest_policy(
    harvested: Sequence[HarvestedSprite],
    policy: HarvestPolicy,
) -> list[HarvestedSprite]:
    """Apply a bulk accept/quarantine/reject policy over harvested sprites."""

    results: list[HarvestedSprite] = []
    for sprite in harvested:
        item = sprite.final_item
        license_name = sprite.source.license.license
        valid = sprite.imported.bundle is not None and not sprite.imported.errors

        if policy.reject_invalid and not valid:
            item = _with_status(item, "rejected")
        elif policy.quarantine_unknown_license and not is_license_allowed_for_training(license_name):
            item = _with_status(item, "quarantine")
        elif (
            policy.quarantine_low_qwen_confidence
            and _qwen_confidence(sprite) is not None
            and (_qwen_confidence(sprite) < policy.qwen_confidence_threshold)
        ):
            item = _with_status(item, "quarantine")
        elif valid and (
            (policy.auto_accept_valid_cc0 and license_name == "cc0")
            or (policy.auto_accept_own_work and license_name == "own_work")
            or (policy.auto_accept_allowlisted and is_license_allowed_for_training(license_name))
        ):
            item = _with_status(item, "accepted")

        results.append(replace(sprite, final_item=item, imported=replace(sprite.imported, item=item)))
    return results


def export_harvested_dataset(
    harvested: Sequence[HarvestedSprite],
    *,
    dataset_name: str,
    output_root: str | Path,
    max_palette_slots: int = 32,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1337,
    overwrite: bool = False,
    allow_unknown_license: bool = False,
) -> DatasetMakerExportResult:
    """Export accepted, license-checked sprites via the Dataset Maker exporter."""

    accepted = [sprite for sprite in harvested if sprite.final_item.status == "accepted"]
    if not accepted:
        raise ValueError("no accepted sprites to export.")

    blocked = [sprite for sprite in accepted if not is_license_allowed_for_training(sprite.source.license.license)]
    if blocked and not allow_unknown_license:
        offenders = sorted({f"{s.source.source_id} ({s.source.license.license})" for s in blocked})
        raise ValueError(
            "export blocked: accepted sprites come from sources with unreviewed/unsafe licenses: "
            + ", ".join(offenders)
            + ". Pass --allow-unknown-license to override (samples will be marked)."
        )

    imported_sprites: list[ImportedSprite] = []
    for sprite in accepted:
        if sprite.imported.bundle is None:
            raise ValueError(f"{sprite.final_item.sprite_id}: accepted sprite has no valid bundle.")
        item = sprite.final_item
        if sprite in blocked:
            notes = (item.notes + " " if item.notes else "") + "[license_override: unreviewed license exported]"
            item = DatasetMakerItem(
                sprite_id=item.sprite_id,
                source_path=item.source_path,
                status=item.status,
                category=item.category,
                tags=(*item.tags, "license_override"),
                notes=notes,
                source_name=item.source_name,
                license=item.license,
                author=item.author,
                split=item.split,
                quality_issues=(*item.quality_issues, "license_override"),
                palette_size=item.palette_size,
                has_role_map=item.has_role_map,
            )
        imported_sprites.append(replace(sprite.imported, item=item, auto_metadata=sprite.auto_metadata))

    return export_dataset_from_imported_sprites(
        imported_sprites,
        DatasetMakerExportConfig(
            dataset_name=dataset_name,
            output_root=Path(output_root),
            max_palette_slots=max_palette_slots,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            overwrite=overwrite,
        ),
    )


def _resolve_source_root(
    source: SourceRecord, work_dir: Path, *, options: HarvestImportOptions
) -> tuple[Path, SourceRecord]:
    if source.local_root_path:
        return Path(source.local_root_path), source
    if source.local_archive_path:
        extracted = work_dir / "extracted" / source.source_id
        if not (extracted.exists() and any(extracted.iterdir())):
            summary = archive_member_summary(
                source.local_archive_path,
                include_member_globs=options.include_member_globs,
                exclude_member_globs=options.exclude_member_globs,
            )
            extract_archive(
                source.local_archive_path,
                extracted,
                overwrite=True,
                include_member_globs=options.include_member_globs,
                exclude_member_globs=options.exclude_member_globs,
            )
        else:
            summary = archive_member_summary(
                source.local_archive_path,
                include_member_globs=options.include_member_globs,
                exclude_member_globs=options.exclude_member_globs,
            )
        return extracted, replace(
            source,
            download_sha256=compute_sha256(source.local_archive_path),
            download_size_bytes=Path(source.local_archive_path).stat().st_size,
            original_filename=Path(source.local_archive_path).name,
            archive_member_summary=summary,
        )
    if source.download_url:
        downloads = work_dir / "downloads"
        is_direct_file = source.download_kind == "file"
        suffix = ".png" if is_direct_file else ".zip"
        original_filename = Path(unquote(urlparse(source.download_url).path)).name or f"{source.source_id}{suffix}"
        archive_path = downloads / original_filename
        if not archive_path.exists():
            download_file(
                source.download_url, archive_path, allowed_content_types=("image/png",) if is_direct_file else ()
            )
        downloaded = replace(
            source,
            download_sha256=compute_sha256(archive_path),
            download_size_bytes=archive_path.stat().st_size,
            downloaded_at_utc=utc_timestamp(),
            original_filename=original_filename,
        )
        if is_direct_file:
            from PIL import Image

            try:
                with Image.open(archive_path) as image:
                    if image.format != "PNG":
                        raise ValueError(f"downloaded attachment is {image.format}, expected PNG")
            except OSError as exc:
                raise ValueError("downloaded attachment is not a decodable PNG") from exc
            extracted = work_dir / "extracted" / source.source_id
            extracted.mkdir(parents=True, exist_ok=True)
            target = extracted / archive_path.name
            if not target.exists():
                target.write_bytes(archive_path.read_bytes())
            return extracted, downloaded
        extracted = work_dir / "extracted" / source.source_id
        summary = archive_member_summary(
            archive_path,
            include_member_globs=options.include_member_globs,
            exclude_member_globs=options.exclude_member_globs,
        )
        if not (extracted.exists() and any(extracted.iterdir())):
            extract_archive(
                archive_path,
                extracted,
                overwrite=True,
                include_member_globs=options.include_member_globs,
                exclude_member_globs=options.exclude_member_globs,
            )
        return extracted, replace(downloaded, archive_member_summary=summary)
    raise ValueError(f"{source.source_id}: source has no local_root_path, local_archive_path, or download_url.")


def _apply_source_metadata(
    item: DatasetMakerItem,
    source: SourceRecord,
    candidate: HarvestCandidate,
    used_ids: set[str],
) -> DatasetMakerItem:
    sprite_id = normalize_sprite_id(f"{source.source_id}__{Path(item.source_path).stem}")
    unique_id = sprite_id
    counter = 1
    while unique_id in used_ids:
        unique_id = f"{sprite_id}_{counter:03d}"
        counter += 1
    used_ids.add(unique_id)
    return DatasetMakerItem(
        sprite_id=unique_id,
        source_path=item.source_path,
        status=item.status,
        category=item.category,
        tags=item.tags,
        notes=item.notes,
        source_name=source.source_name,
        license=source.license.license,
        author=source.author,
        split=item.split,
        quality_issues=item.quality_issues,
        palette_size=item.palette_size,
        has_role_map=item.has_role_map,
    )


def _with_status(item: DatasetMakerItem, status: str) -> DatasetMakerItem:
    return DatasetMakerItem(
        sprite_id=item.sprite_id,
        source_path=item.source_path,
        status=status,
        category=item.category,
        tags=item.tags,
        notes=item.notes,
        source_name=item.source_name,
        license=item.license,
        author=item.author,
        split=item.split,
        quality_issues=item.quality_issues,
        palette_size=item.palette_size,
        has_role_map=item.has_role_map,
    )


def _qwen_confidence(sprite: HarvestedSprite) -> float | None:
    suggestion = sprite.auto_metadata.get("qwen_suggestion")
    if isinstance(suggestion, dict):
        confidence = suggestion.get("confidence")
        if confidence is not None:
            return float(confidence)
    return None
