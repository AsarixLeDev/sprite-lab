"""End-to-end harvest pipeline: source -> candidates -> imports -> export."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from spritelab.dataset_maker.exporter import (
    DatasetMakerExportConfig,
    DatasetMakerExportResult,
    export_dataset_from_imported_sprites,
)
from spritelab.dataset_maker.importer import ImportedSprite, ImportOptions, import_png_as_dataset_item
from spritelab.dataset_maker.model import DatasetMakerItem, normalize_sprite_id
from spritelab.harvest.archive import DEFAULT_MAX_ARCHIVE_BYTES, archive_member_summary, extract_archive
from spritelab.harvest.autolabel import merge_auto_labels, suggest_metadata_from_path
from spritelab.harvest.download import DEFAULT_MAX_DOWNLOAD_BYTES, compute_sha256, download_file
from spritelab.harvest.extract import HarvestCandidate, discover_png_candidates, filter_candidate_basic
from spritelab.harvest.sheet_mappings import metadata_for_sheet_cell
from spritelab.harvest.sheets import SheetSliceConfig, center_pad_to_32, looks_like_sprite_sheet, slice_sheet_to_pngs
from spritelab.harvest.sources import SourceRecord, is_license_allowed_for_training, utc_timestamp
from spritelab.utils.safe_fs import UnsafeFilesystemOperation, atomic_write_text, require_confined_path


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

    work_dir = _validated_work_root(work_dir)
    root, source = _resolve_source_root(source, work_dir, options=options)
    candidates = [
        filter_candidate_basic(candidate)
        for candidate in discover_png_candidates(root, source, recursive=options.recursive)
    ]

    final_pngs: list[tuple[HarvestCandidate, Path, list[str]]] = []
    sliced_dir = _ensure_confined_directory(work_dir / "sliced", work_dir)
    padded_dir = _ensure_confined_directory(work_dir / "padded", work_dir)
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
            tile_dir = _ensure_confined_directory(sliced_dir / candidate.candidate_id, sliced_dir)
            tiles = slice_sheet_to_pngs(candidate.extracted_path, tile_dir, options.sheet_config)
            for tile in tiles:
                if (
                    options.allow_center_pad_to_32
                    and options.sheet_config.tile_width <= 32
                    and options.sheet_config.tile_height <= 32
                ):
                    padded_tile = require_confined_path(padded_dir / f"{tile.stem}.png", padded_dir)
                    tile = center_pad_to_32(tile, padded_tile)
                final_pngs.append((candidate, tile, []))
        elif options.allow_center_pad_to_32 and candidate.width <= 32 and candidate.height <= 32:
            padded_path = require_confined_path(padded_dir / f"{candidate.candidate_id}.png", padded_dir)
            padded = center_pad_to_32(candidate.extracted_path, padded_path)
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
        extracted_root = _ensure_confined_directory(work_dir / "extracted", work_dir)
        extracted = require_confined_path(extracted_root / source.source_id, extracted_root)
        summary = archive_member_summary(
            source.local_archive_path,
            include_member_globs=options.include_member_globs,
            exclude_member_globs=options.exclude_member_globs,
        )
        digest = compute_sha256(source.local_archive_path, max_bytes=DEFAULT_MAX_ARCHIVE_BYTES)
        _verify_expected_source_digest(source, digest)
        extract_archive(
            source.local_archive_path,
            extracted,
            overwrite=True,
            include_member_globs=options.include_member_globs,
            exclude_member_globs=options.exclude_member_globs,
            expected_sha256=digest,
        )
        return extracted, replace(
            source,
            download_sha256=digest,
            download_size_bytes=Path(source.local_archive_path).stat().st_size,
            original_filename=Path(source.local_archive_path).name,
            archive_member_summary=summary,
        )
    if source.download_url:
        downloads = _ensure_confined_directory(work_dir / "downloads", work_dir)
        source_cache = _ensure_confined_directory(downloads / source.source_id, downloads)
        is_direct_file = source.download_kind == "file"
        suffix = ".png" if is_direct_file else ".zip"
        original_filename = _download_original_filename(source.download_url, source.source_id, suffix)
        identity = hashlib.sha256(f"{source.download_kind}\0{source.download_url}".encode()).hexdigest()[:24]
        cache_dir = _ensure_confined_directory(source_cache / identity, source_cache)
        archive_path = require_confined_path(
            cache_dir / _cache_payload_name(original_filename, source.source_id, suffix),
            cache_dir,
        )
        binding_path = require_confined_path(cache_dir / "binding.json", cache_dir)
        _assert_cache_directory_entries(cache_dir, {archive_path.name, binding_path.name})
        expected_digest = _expected_source_digest(source)
        cached = _load_bound_download(
            archive_path,
            binding_path,
            source=source,
            expected_digest=expected_digest,
        )
        if cached is None:
            download_file(
                source.download_url,
                archive_path,
                overwrite=True,
                allowed_content_types=("image/png",) if is_direct_file else (),
                expected_sha256=expected_digest,
            )
            digest = compute_sha256(archive_path, max_bytes=DEFAULT_MAX_DOWNLOAD_BYTES)
            size = archive_path.stat().st_size
            downloaded_at = utc_timestamp()
            _write_download_binding(
                binding_path,
                source=source,
                digest=digest,
                size=size,
                downloaded_at=downloaded_at,
            )
        else:
            digest, size, downloaded_at = cached
        _assert_cache_directory_entries(cache_dir, {archive_path.name, binding_path.name})
        downloaded = replace(
            source,
            download_sha256=digest,
            download_size_bytes=size,
            downloaded_at_utc=downloaded_at,
            original_filename=original_filename,
        )
        if is_direct_file:
            from PIL import Image

            try:
                with Image.open(archive_path) as image:
                    if image.format != "PNG":
                        raise ValueError(f"downloaded attachment is {image.format}, expected PNG")
                    image.load()
            except OSError as exc:
                raise ValueError("downloaded attachment is not a decodable PNG") from exc
            return cache_dir, downloaded
        extracted_root = _ensure_confined_directory(work_dir / "extracted", work_dir)
        extracted = require_confined_path(extracted_root / source.source_id, extracted_root)
        summary = archive_member_summary(
            archive_path,
            include_member_globs=options.include_member_globs,
            exclude_member_globs=options.exclude_member_globs,
        )
        extract_archive(
            archive_path,
            extracted,
            overwrite=True,
            include_member_globs=options.include_member_globs,
            exclude_member_globs=options.exclude_member_globs,
            expected_sha256=digest,
        )
        return extracted, replace(downloaded, archive_member_summary=summary)
    raise ValueError(f"{source.source_id}: source has no local_root_path, local_archive_path, or download_url.")


def _ensure_confined_directory(path: Path, root: Path) -> Path:
    try:
        path = require_confined_path(path, root)
    except UnsafeFilesystemOperation as exc:
        raise ValueError(f"harvest work path crosses a link or reparse boundary: {path}") from exc
    if not os.path.lexists(path):
        path.mkdir()
    metadata = path.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode) or path.is_mount():
        raise ValueError(f"harvest work path is not a confined directory: {path}")
    return path


def _validated_work_root(path: str | Path) -> Path:
    raw_path = os.fspath(path)
    if not raw_path.strip() or raw_path.strip() in {".", ".."}:
        raise ValueError("harvest work directory must be a specific non-root path")
    work_dir = Path(os.path.abspath(os.path.expanduser(raw_path)))
    existing_ancestor = work_dir.parent
    while not os.path.lexists(existing_ancestor):
        parent = existing_ancestor.parent
        if parent == existing_ancestor:
            raise ValueError(f"could not find an existing ancestor for harvest work directory: {work_dir}")
        existing_ancestor = parent
    metadata = existing_ancestor.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"harvest work directory crosses an unsafe ancestor: {existing_ancestor}")
    work_dir = require_confined_path(work_dir, existing_ancestor)
    _create_work_directories(work_dir, existing_ancestor)
    work_dir = require_confined_path(work_dir, existing_ancestor)
    metadata = work_dir.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"harvest work directory is not a confined directory: {work_dir}")
    return work_dir


def _create_work_directories(target: Path, root: Path) -> None:
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        metadata = current.lstat()
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode) or current.is_mount():
            raise ValueError(f"harvest work directory crosses an unsafe directory seam: {current}")
        require_confined_path(current, root)


def _download_original_filename(url: str, source_id: str, suffix: str) -> str:
    path = unquote(urlparse(url).path).replace("\\", "/")
    name = PurePosixPath(path).name
    name = "".join(character if ord(character) >= 32 else "_" for character in name).strip()
    return name[:255] or f"{source_id}{suffix}"


def _cache_payload_name(original_filename: str, source_id: str, suffix: str) -> str:
    stem = normalize_sprite_id(PurePosixPath(original_filename.replace("\\", "/")).stem)
    return f"{stem or source_id}{suffix}"


def _expected_source_digest(source: SourceRecord) -> str | None:
    value = source.download_sha256 or source.sha256
    if not value:
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{source.source_id}: expected download SHA256 is not a valid digest")
    return normalized


def _verify_expected_source_digest(source: SourceRecord, actual_digest: str) -> None:
    expected = _expected_source_digest(source)
    if expected is not None and actual_digest != expected:
        raise ValueError(f"{source.source_id}: archive SHA256 mismatch: expected {expected}, got {actual_digest}")


def _load_bound_download(
    archive_path: Path,
    binding_path: Path,
    *,
    source: SourceRecord,
    expected_digest: str | None,
) -> tuple[str, int, str] | None:
    archive_exists = os.path.lexists(archive_path)
    binding_exists = os.path.lexists(binding_path)
    if not archive_exists or not binding_exists:
        return None
    _require_safe_cache_file(archive_path)
    _require_safe_cache_file(binding_path)
    try:
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != "spritelab.harvest-download-cache.v1"
        or payload.get("source_id") != source.source_id
        or payload.get("download_url") != source.download_url
        or payload.get("download_kind") != source.download_kind
    ):
        return None
    size = archive_path.stat().st_size
    if size > DEFAULT_MAX_DOWNLOAD_BYTES:
        return None
    digest = compute_sha256(archive_path, max_bytes=DEFAULT_MAX_DOWNLOAD_BYTES)
    if payload.get("sha256") != digest or payload.get("size_bytes") != size:
        return None
    if expected_digest is not None and digest != expected_digest:
        return None
    downloaded_at = payload.get("downloaded_at_utc")
    if not isinstance(downloaded_at, str) or not downloaded_at:
        return None
    return digest, size, downloaded_at


def _write_download_binding(
    binding_path: Path,
    *,
    source: SourceRecord,
    digest: str,
    size: int,
    downloaded_at: str,
) -> None:
    payload = {
        "download_kind": source.download_kind,
        "download_url": source.download_url,
        "downloaded_at_utc": downloaded_at,
        "schema_version": "spritelab.harvest-download-cache.v1",
        "sha256": digest,
        "size_bytes": size,
        "source_id": source.source_id,
    }
    atomic_write_text(binding_path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _assert_cache_directory_entries(cache_dir: Path, allowed_names: set[str]) -> None:
    for path in cache_dir.iterdir():
        if path.name not in allowed_names:
            raise ValueError(f"download cache contains an unbound entry: {path}")
        _require_safe_cache_file(path)


def _require_safe_cache_file(path: Path) -> None:
    metadata = path.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"download cache entry is not a confined regular file: {path}")


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


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
