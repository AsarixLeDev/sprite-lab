"""Export Dataset Maker imports to Phase 7-ready fixed arrays."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.codec.bundle import INDEX_MASK, INDEX_PAD, SpriteBundle
from spritelab.codec.palette import visible_palette_size
from spritelab.codec.roles import ROLE_NAMES, ROLE_TRANSPARENT, ROLE_UNKNOWN
from spritelab.codec.validate import validate_bundle
from spritelab.dataset_maker.importer import ImportedSprite
from spritelab.dataset_maker.model import (
    ALLOWED_SPLITS,
    DatasetMakerItem,
    normalize_sprite_id,
    validate_dataset_maker_item,
)


@dataclass(frozen=True)
class DatasetMakerExportConfig:
    dataset_name: str
    output_root: Path
    max_palette_slots: int = 32
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 1337
    overwrite: bool = False


@dataclass(frozen=True)
class DatasetMakerExportResult:
    output_dir: Path
    train_count: int
    val_count: int
    test_count: int
    accepted_count: int
    excluded_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedSprite:
    item: DatasetMakerItem
    bundle: SpriteBundle
    category_id: int
    split: str
    auto_metadata: Mapping[str, Any]


def export_dataset_from_imported_sprites(
    imported: Sequence[ImportedSprite],
    config: DatasetMakerExportConfig,
) -> DatasetMakerExportResult:
    """Export accepted Dataset Maker sprites into npz arrays and manifests."""

    _validate_config(config)
    dataset_name = normalize_sprite_id(config.dataset_name)
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty and filesystem-safe.")

    output_dir = Path(config.output_root) / dataset_name
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")

    accepted = [sprite for sprite in imported if sprite.item.status == "accepted"]
    excluded = [sprite for sprite in imported if sprite.item.status != "accepted"]
    if not accepted:
        raise ValueError("no accepted sprites to export.")

    _validate_unique_sprite_ids([sprite.item for sprite in accepted])
    category_to_id = _category_vocab([sprite.item for sprite in accepted])
    split_lookup = make_dataset_maker_split(
        [sprite.item for sprite in accepted],
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    if not any(split == "train" for split in split_lookup.values()):
        raise ValueError("train split would be empty; adjust split overrides or fractions.")

    prepared: list[_PreparedSprite] = []
    for sprite in accepted:
        _validate_imported_for_export(sprite, max_palette_slots=config.max_palette_slots)
        prepared.append(
            _PreparedSprite(
                item=sprite.item,
                bundle=sprite.bundle,
                category_id=category_to_id[sprite.item.category],
                split=split_lookup[sprite.item.sprite_id],
                auto_metadata=dict(sprite.auto_metadata) if isinstance(sprite.auto_metadata, Mapping) else {},
            )
        )

    if output_dir.exists() and config.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings = _split_warnings(prepared)
    for split_name in ("train", "val", "test"):
        split_sprites = [sprite for sprite in prepared if sprite.split == split_name]
        _write_split_npz(output_dir / f"{split_name}.npz", split_sprites, max_palette_slots=config.max_palette_slots)
        _write_manifest_jsonl(output_dir / f"manifest_{split_name}.jsonl", _manifest_records(split_sprites))

    _write_json(output_dir / "vocab.json", _vocab(category_to_id, prepared))
    _write_json(output_dir / "dataset_config.json", _dataset_config(dataset_name, config.max_palette_slots))
    _write_rejected_jsonl(output_dir / "rejected.jsonl", excluded)

    result = DatasetMakerExportResult(
        output_dir=output_dir,
        train_count=sum(1 for sprite in prepared if sprite.split == "train"),
        val_count=sum(1 for sprite in prepared if sprite.split == "val"),
        test_count=sum(1 for sprite in prepared if sprite.split == "test"),
        accepted_count=len(prepared),
        excluded_count=len(excluded),
        warnings=tuple(warnings),
    )
    from spritelab.dataset_maker.report import build_dataset_maker_report

    (output_dir / "dataset_report.md").write_text(build_dataset_maker_report(imported, result), encoding="utf-8")
    return result


def make_dataset_maker_split(
    items: Sequence[DatasetMakerItem],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Return a deterministic ``sprite_id -> split`` mapping."""

    _validate_fractions(train_fraction, val_fraction, test_fraction)
    ordered_items = sorted(items, key=lambda item: item.sprite_id)
    split_by_id: dict[str, str] = {}
    auto_items: list[DatasetMakerItem] = []
    for item in ordered_items:
        if item.split is None:
            auto_items.append(item)
        elif item.split in ALLOWED_SPLITS:
            split_by_id[item.sprite_id] = item.split
        else:
            raise ValueError(f"{item.sprite_id}: invalid split override {item.split!r}.")

    total_count = len(ordered_items)
    if total_count == 0:
        return {}
    if total_count < 3:
        for item in auto_items:
            split_by_id[item.sprite_id] = "train"
        return split_by_id

    target_counts = _target_split_counts(
        total_count,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    current_counts = Counter(split_by_id.values())
    shuffled = list(auto_items)
    random.Random(seed).shuffle(shuffled)

    for split_name in ("train", "val", "test"):
        if target_counts[split_name] <= 0 or current_counts[split_name] > 0:
            continue
        if not shuffled:
            break
        item = shuffled.pop(0)
        split_by_id[item.sprite_id] = split_name
        current_counts[split_name] += 1

    for item in shuffled:
        split_name = _best_split(current_counts, target_counts)
        split_by_id[item.sprite_id] = split_name
        current_counts[split_name] += 1

    return split_by_id


def _validate_imported_for_export(imported: ImportedSprite, *, max_palette_slots: int) -> None:
    item = imported.item
    item_errors = validate_dataset_maker_item(item)
    if item_errors:
        raise ValueError(f"{item.sprite_id}: invalid metadata: {'; '.join(item_errors)}")
    if imported.errors:
        raise ValueError(f"{item.sprite_id}: accepted sprite has import errors: {'; '.join(imported.errors)}")
    if imported.bundle is None:
        raise ValueError(f"{item.sprite_id}: accepted sprite has no SpriteBundle.")

    bundle = imported.bundle
    bundle_errors = validate_bundle(bundle)
    if bundle_errors:
        raise ValueError(f"{item.sprite_id}: invalid SpriteBundle: {'; '.join(bundle_errors)}")
    palette_size = visible_palette_size(np.asarray(bundle.palette))
    if palette_size > max_palette_slots:
        raise ValueError(
            f"{item.sprite_id}: palette has {palette_size} visible slots, above max_palette_slots={max_palette_slots}."
        )
    sample_errors = _validate_sample_contract(bundle, max_palette_slots=max_palette_slots)
    if sample_errors:
        raise ValueError(f"{item.sprite_id}: {'; '.join(sample_errors)}")


def _validate_sample_contract(bundle: SpriteBundle, *, max_palette_slots: int) -> list[str]:
    errors: list[str] = []
    alpha = np.asarray(bundle.alpha)
    index_map = np.asarray(bundle.index_map)
    palette = np.asarray(bundle.palette)
    visible_rows = int(palette.shape[0] - 1)

    if alpha.shape != (32, 32):
        errors.append("alpha shape must be [32, 32].")
    if index_map.shape != (32, 32):
        errors.append("index_map shape must be [32, 32].")
    if not np.all(np.isin(alpha, [0, 1])):
        errors.append("alpha values must be 0 or 1.")
    if bool(np.any((alpha == 0) & (index_map != 0))):
        errors.append("transparent pixels must have index_map == 0.")
    if bool(np.any((alpha == 1) & (index_map < 1))):
        errors.append("opaque pixels must have index_map >= 1.")
    if palette.shape[0] and not np.array_equal(palette[0], np.array([0, 0, 0], dtype=np.uint8)):
        errors.append("palette row 0 must be [0, 0, 0].")
    if visible_rows > max_palette_slots:
        errors.append("palette has more visible rows than max_palette_slots.")
    if index_map.size and int(np.max(index_map)) > visible_rows:
        errors.append("index_map points outside the true palette rows.")
    return errors


def _write_split_npz(path: Path, sprites: Sequence[_PreparedSprite], *, max_palette_slots: int) -> None:
    count = len(sprites)
    palette_rows = max_palette_slots + 1
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, palette_rows, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, palette_rows), dtype=bool)
    category_id = np.zeros((count,), dtype=np.int64)
    sprite_id = np.array([sprite.item.sprite_id for sprite in sprites], dtype=np.str_)

    for row, sprite in enumerate(sprites):
        bundle = sprite.bundle
        bundle_palette = np.asarray(bundle.palette, dtype=np.uint8)
        actual_rows = int(bundle_palette.shape[0])
        alpha[row] = np.asarray(bundle.alpha, dtype=np.uint8)
        index_map[row] = np.asarray(bundle.index_map, dtype=np.int16)
        role_map[row] = _role_map_for_export(bundle)
        palette[row, :actual_rows] = bundle_palette
        palette_mask[row, :actual_rows] = True
        category_id[row] = int(sprite.category_id)

    np.savez_compressed(
        path,
        alpha=alpha,
        index_map=index_map,
        role_map=role_map,
        palette=palette,
        palette_mask=palette_mask,
        category_id=category_id,
        sprite_id=sprite_id,
    )


def _role_map_for_export(bundle: SpriteBundle) -> np.ndarray:
    if bundle.role_map is not None:
        return np.asarray(bundle.role_map, dtype=np.uint8)
    fallback = np.zeros((32, 32), dtype=np.uint8)
    alpha = np.asarray(bundle.alpha)
    fallback[alpha == 0] = ROLE_TRANSPARENT
    fallback[alpha == 1] = ROLE_UNKNOWN
    return fallback


def _manifest_records(sprites: Sequence[_PreparedSprite]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sprite in sorted(sprites, key=lambda value: value.item.sprite_id):
        item = sprite.item
        label_v2 = _label_v2_manifest_metadata(sprite.auto_metadata)
        safe_prefill = _mapping(sprite.auto_metadata.get("label_v2_safe_prefill")) if isinstance(sprite.auto_metadata, Mapping) else {}
        records.append(
            {
                "sprite_id": item.sprite_id,
                "split": sprite.split,
                "category": item.category,
                "category_id": sprite.category_id,
                "object_name": str(safe_prefill.get("object_name", "")),
                "tags": list(item.tags),
                "short_description": str(safe_prefill.get("short_description") or item.notes),
                "materials": [str(value) for value in safe_prefill.get("materials") or ()],
                "mood": [str(value) for value in safe_prefill.get("mood") or ()],
                "palette_size": int(visible_palette_size(np.asarray(sprite.bundle.palette))),
                "has_role_map": sprite.bundle.role_map is not None,
                "source_path": str(item.source_path),
                "source_name": item.source_name,
                "license": item.license,
                "author": item.author,
                "notes": item.notes,
                "label_v2": label_v2,
            }
        )
    return records


def _label_v2_manifest_metadata(auto_metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(auto_metadata, Mapping) or not auto_metadata.get("label_v2_applied"):
        return {}
    result: dict[str, Any] = {
        "applied": bool(auto_metadata.get("label_v2_applied", False)),
        "prediction_file": str(auto_metadata.get("label_v2_prediction_file", "")),
        "bucket": str(auto_metadata.get("label_v2_bucket", "")),
        "flags": [str(value) for value in auto_metadata.get("label_v2_flags") or ()],
        "candidate_object_names": [str(value) for value in auto_metadata.get("label_v2_candidate_object_names") or ()],
    }
    safe_prefill = _mapping(auto_metadata.get("label_v2_safe_prefill"))
    if safe_prefill:
        result["safe_prefill"] = safe_prefill
    vlm_descriptor = _mapping(auto_metadata.get("label_v2_vlm_descriptor"))
    if vlm_descriptor:
        result["vlm_descriptor"] = vlm_descriptor
        result["vlm_object_name"] = str(vlm_descriptor.get("object_name", ""))
        result["vlm_alternative_object_names"] = [
            str(value) for value in vlm_descriptor.get("alternative_object_names") or ()
        ]
        result["vlm_source_consistency"] = str(vlm_descriptor.get("source_consistency", ""))
    label_quality = _mapping(auto_metadata.get("label_v2_label_quality"))
    if label_quality:
        result["label_quality"] = label_quality
    conflict_reasons = [str(value) for value in auto_metadata.get("label_v2_conflict_reasons") or ()]
    if conflict_reasons:
        result["conflict_reasons"] = conflict_reasons
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_manifest_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_rejected_jsonl(path: Path, sprites: Sequence[ImportedSprite]) -> None:
    lines: list[str] = []
    for sprite in sorted(sprites, key=lambda value: value.item.sprite_id):
        item = sprite.item
        lines.append(
            json.dumps(
                {
                    "sprite_id": item.sprite_id,
                    "status": item.status,
                    "category": item.category,
                    "tags": list(item.tags),
                    "source_path": str(item.source_path),
                    "source_name": item.source_name,
                    "license": item.license,
                    "author": item.author,
                    "notes": item.notes,
                    "errors": list(sprite.errors),
                    "warnings": list(sprite.warnings),
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _vocab(category_to_id: Mapping[str, int], sprites: Sequence[_PreparedSprite]) -> dict[str, Any]:
    tags = sorted({tag for sprite in sprites for tag in sprite.item.tags})
    role_order = sorted(ROLE_NAMES)
    return {
        "category_to_id": dict(sorted(category_to_id.items(), key=lambda item: item[1])),
        "tag_to_id": {tag: index for index, tag in enumerate(tags, start=1)},
        "role_names": {str(role_id): ROLE_NAMES[role_id] for role_id in role_order},
    }


def _dataset_config(dataset_name: str, max_palette_slots: int) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "sprite_size": 32,
        "max_palette_slots": max_palette_slots,
        "index_transparent": 0,
        "index_pad": INDEX_PAD,
        "index_mask": INDEX_MASK,
        "role_transparent": ROLE_TRANSPARENT,
        "role_unknown": ROLE_UNKNOWN,
        "alpha_shape": [32, 32],
        "index_map_shape": [32, 32],
        "created_by": "spritelab.dataset_maker",
        "format_version": "1.0",
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _category_vocab(items: Sequence[DatasetMakerItem]) -> dict[str, int]:
    categories = {item.category or "unknown" for item in items}
    ordered = ["unknown", *sorted(category for category in categories if category != "unknown")]
    return {category: index for index, category in enumerate(ordered)}


def _validate_unique_sprite_ids(items: Sequence[DatasetMakerItem]) -> None:
    counts = Counter(item.sprite_id for item in items)
    duplicates = sorted(sprite_id for sprite_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sprite_id values among accepted sprites: {', '.join(duplicates)}")


def _validate_config(config: DatasetMakerExportConfig) -> None:
    if config.max_palette_slots < 1:
        raise ValueError("max_palette_slots must be at least 1.")
    _validate_fractions(config.train_fraction, config.val_fraction, config.test_fraction)


def _validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(value < 0.0 for value in fractions):
        raise ValueError("split fractions must be non-negative.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.")


def _target_split_counts(
    total_count: int,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    names = ("train", "val", "test")
    fractions = {"train": train_fraction, "val": val_fraction, "test": test_fraction}
    raw = {name: total_count * fractions[name] for name in names}
    counts = {name: int(np.floor(raw[name])) for name in names}
    remainder = total_count - sum(counts.values())
    for name in sorted(names, key=lambda split: (raw[split] - counts[split], split), reverse=True):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1

    positive = [name for name in names if fractions[name] > 0.0]
    if total_count >= len(positive):
        for name in positive:
            if counts[name] == 0:
                donor = max((other for other in names if counts[other] > 1), key=lambda other: counts[other])
                counts[donor] -= 1
                counts[name] += 1
    return counts


def _best_split(current_counts: Mapping[str, int], target_counts: Mapping[str, int]) -> str:
    names = ("train", "val", "test")

    def score(name: str) -> tuple[float, int]:
        target = max(1, int(target_counts[name]))
        count = int(current_counts.get(name, 0))
        return ((target - count) / target, -count)

    return max(names, key=score)


def _split_warnings(sprites: Sequence[_PreparedSprite]) -> list[str]:
    counts = Counter(sprite.split for sprite in sprites)
    warnings: list[str] = []
    if len(sprites) < 3:
        warnings.append("Tiny dataset: val/test splits may be empty.")
    elif counts["val"] == 0 or counts["test"] == 0:
        warnings.append("One or more validation/test splits are empty.")
    return warnings
