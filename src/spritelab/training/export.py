"""Dedupe-safe fixed-array training export for curated SpriteBundles."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.codec.bundle import BUNDLE_SCHEMA_VERSION, CODEC_VERSION, SpriteBundle
from spritelab.codec.io import load_bundle
from spritelab.codec.roles import ROLE_NAMES, ROLE_TRANSPARENT, ROLE_UNKNOWN
from spritelab.codec.validate import assert_valid_bundle
from spritelab.curation.manifest import discover_bundle_ids, load_latest_curation, summarize_curation
from spritelab.training.palette_report import (
    build_palette_semantics_report,
    write_palette_semantics_report_json,
    write_palette_semantics_report_markdown,
)
from spritelab.training.readiness import (
    build_training_readiness_report,
    write_training_readiness_markdown,
)
from spritelab.training.splits import DEFAULT_SPLIT_SEED, SplitAssignment, make_group_aware_split

KNOWN_CATEGORY_TAGS = ("item_icon", "block", "plant", "ui_icon", "entity", "terrain", "weapon", "armor")


@dataclass(frozen=True)
class TrainingExportConfig:
    bundle_root: Path
    curation_path: Path
    output_dir: Path
    dataset_name: str = "sprite_dataset"
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = DEFAULT_SPLIT_SEED
    max_palette_slots: int = 32
    quality_report_path: Path | None = None
    dedupe_report_path: Path | None = None
    fail_on_duplicate_leakage: bool = True
    fail_on_invalid_accepted: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class TrainingExportResult:
    output_dir: Path
    train_count: int
    val_count: int
    test_count: int
    accepted_count: int
    excluded_count: int
    warnings: tuple[str, ...]
    readiness_passed: bool


@dataclass(frozen=True)
class _ExportSprite:
    sprite_id: str
    bundle_path: Path
    bundle: SpriteBundle
    tags: tuple[str, ...]
    category: str
    category_id: int
    quality_issues: tuple[str, ...]
    dedupe_group: str
    has_role_map: bool


def export_training_dataset(config: TrainingExportConfig) -> TrainingExportResult:
    """Export latest accepted curated bundles to fixed-shape training arrays."""

    if config.max_palette_slots < 1:
        raise ValueError("max_palette_slots must be at least 1.")
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output_dir already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_ids = discover_bundle_ids(config.bundle_root)
    collision_map = getattr(bundle_ids, "collisions", {})
    if collision_map:
        messages = []
        for sprite_id, paths in sorted(collision_map.items()):
            joined = ", ".join(str(path) for path in paths)
            messages.append(f"{sprite_id}: {joined}")
        raise ValueError("bundle ID collision detected: " + "; ".join(messages))
    latest = load_latest_curation(config.curation_path)
    summarize_curation(latest)
    accepted_ids = tuple(sorted(sprite_id for sprite_id, decision in latest.items() if decision.status == "accepted"))
    missing_accepted = [sprite_id for sprite_id in accepted_ids if sprite_id not in bundle_ids]
    if missing_accepted:
        message = ", ".join(missing_accepted)
        raise ValueError(f"accepted curation decisions reference missing bundles: {message}")

    dedupe_report_provided = _report_exists(config.dedupe_report_path, "dedupe_report.json")
    quality_report_provided = _report_exists(config.quality_report_path, "quality_report.json")
    dedupe_groups = load_dedupe_groups(config.dedupe_report_path)
    quality_issues = load_quality_issues(config.quality_report_path)

    warnings: list[str] = []
    if not dedupe_report_provided:
        warnings.append("No dedupe groups were provided; split cannot protect against near-duplicate leakage.")
    if not quality_report_provided:
        warnings.append("No quality report was provided; quality issues will be empty.")

    invalid_accepted: list[tuple[str, str]] = []
    loaded: list[tuple[str, Path, SpriteBundle]] = []
    for sprite_id in accepted_ids:
        path = bundle_ids[sprite_id]
        try:
            bundle = load_bundle(path)
            assert_valid_bundle(bundle)
            if int(np.asarray(bundle.palette).shape[0] - 1) > config.max_palette_slots:
                raise ValueError(
                    f"palette has {int(np.asarray(bundle.palette).shape[0] - 1)} visible slots, above max_palette_slots={config.max_palette_slots}"
                )
            loaded.append((sprite_id, path, bundle))
        except Exception as exc:
            invalid_accepted.append((sprite_id, str(exc)))
    if invalid_accepted and config.fail_on_invalid_accepted:
        joined = "; ".join(f"{sprite_id}: {reason}" for sprite_id, reason in invalid_accepted)
        raise ValueError(f"Invalid accepted bundles: {joined}")

    category_to_id = _category_vocab(loaded, latest)
    sprites = [
        _export_sprite(
            sprite_id=sprite_id,
            bundle_path=path,
            bundle=bundle,
            decision=latest[sprite_id],
            category_to_id=category_to_id,
            dedupe_groups=dedupe_groups,
            quality_issues=quality_issues,
        )
        for sprite_id, path, bundle in loaded
    ]

    group_by_id = {sprite.sprite_id: sprite.dedupe_group for sprite in sprites}
    split_assignment = make_group_aware_split(
        [sprite.sprite_id for sprite in sprites],
        group_by_sprite_id=group_by_id,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    leakage = _duplicate_leakage(split_assignment)
    if leakage and config.fail_on_duplicate_leakage:
        joined = "; ".join(f"{sprite_id}: {left}/{right}" for sprite_id, left, right in leakage)
        raise ValueError(f"Duplicate leakage across splits: {joined}")

    split_lookup = _split_lookup(split_assignment)
    records = [
        _manifest_record(
            sprite, split_lookup[sprite.sprite_id], config, dedupe_report_provided, quality_report_provided
        )
        for sprite in sprites
    ]
    for split_name, split_ids in {
        "train": split_assignment.train,
        "val": split_assignment.val,
        "test": split_assignment.test,
    }.items():
        split_sprites = [sprite for sprite in sprites if sprite.sprite_id in set(split_ids)]
        _write_split_npz(output_dir / f"{split_name}.npz", split_sprites, max_palette_slots=config.max_palette_slots)
        _write_manifest_jsonl(
            output_dir / f"manifest_{split_name}.jsonl",
            [record for record in records if record["split"] == split_name],
        )

    vocab = _vocab(category_to_id, sprites)
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    palette_report = build_palette_semantics_report([sprite.bundle_path for sprite in sprites])
    write_palette_semantics_report_json(palette_report, output_dir / "palette_semantics_report.json")
    write_palette_semantics_report_markdown(palette_report, output_dir / "palette_semantics_report.md")

    readiness = build_training_readiness_report(
        records,
        split_assignment,
        palette_report,
        accepted_count=len(accepted_ids),
        exported_count=len(sprites),
        invalid_accepted=invalid_accepted,
        duplicate_leakage=leakage,
    )
    write_training_readiness_markdown(readiness, output_dir / "training_readiness_report.md")
    _write_export_config(
        output_dir / "export_config.json", config, accepted_count=len(accepted_ids), exported_count=len(sprites)
    )

    warnings.extend(issue.message for issue in readiness.issues if issue.severity == "warning")
    excluded_count = max(0, len(bundle_ids) - len(sprites))
    return TrainingExportResult(
        output_dir=output_dir,
        train_count=len(split_assignment.train),
        val_count=len(split_assignment.val),
        test_count=len(split_assignment.test),
        accepted_count=len(accepted_ids),
        excluded_count=excluded_count,
        warnings=tuple(dict.fromkeys(warnings)),
        readiness_passed=readiness.passed,
    )


def load_dedupe_groups(path: Path | None) -> dict[str, str]:
    """Best-effort parser for current dedupe report duplicate groups."""

    report_path = _resolve_report_path(path, "dedupe_report.json")
    if report_path is None:
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed dedupe report JSON: {report_path}") from exc

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for group in list(data.get("exact_groups", [])) + list(data.get("near_groups", [])):
        ids = [str(sprite_id) for sprite_id in group.get("ids", [])]
        if not ids:
            continue
        first = ids[0]
        find(first)
        for sprite_id in ids[1:]:
            union(first, sprite_id)

    grouped: dict[str, list[str]] = defaultdict(list)
    for sprite_id in sorted(parent):
        grouped[find(sprite_id)].append(sprite_id)
    result: dict[str, str] = {}
    for index, (_root, ids) in enumerate(sorted(grouped.items(), key=lambda item: item[1]), start=1):
        group_id = f"group_{index:04d}"
        for sprite_id in ids:
            result[sprite_id] = group_id
    return result


def load_quality_issues(path: Path | None) -> dict[str, tuple[str, ...]]:
    """Best-effort parser for current quality report issue codes."""

    report_path = _resolve_report_path(path, "quality_report.json")
    if report_path is None:
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed quality report JSON: {report_path}") from exc
    issues: dict[str, tuple[str, ...]] = {}
    for record in data.get("records", []):
        sprite_id = record.get("id")
        if sprite_id is not None:
            issues[str(sprite_id)] = tuple(str(issue) for issue in record.get("issue_codes", []))
    return issues


def _export_sprite(
    *,
    sprite_id: str,
    bundle_path: Path,
    bundle: SpriteBundle,
    decision: Any,
    category_to_id: Mapping[str, int],
    dedupe_groups: Mapping[str, str],
    quality_issues: Mapping[str, tuple[str, ...]],
) -> _ExportSprite:
    category = _category_for(bundle, decision.tags)
    return _ExportSprite(
        sprite_id=sprite_id,
        bundle_path=bundle_path,
        bundle=bundle,
        tags=decision.tags,
        category=category,
        category_id=category_to_id[category],
        quality_issues=quality_issues.get(sprite_id, ()),
        dedupe_group=dedupe_groups.get(sprite_id, sprite_id),
        has_role_map=bundle.role_map is not None,
    )


def _category_vocab(loaded: Sequence[tuple[str, Path, SpriteBundle]], latest: Mapping[str, Any]) -> dict[str, int]:
    categories = {_category_for(bundle, latest[sprite_id].tags) for sprite_id, _path, bundle in loaded}
    ordered = ["unknown", *sorted(category for category in categories if category != "unknown")]
    return {category: index for index, category in enumerate(ordered)}


def _category_for(bundle: SpriteBundle, tags: Sequence[str]) -> str:
    if bundle.metadata.category:
        return str(bundle.metadata.category)
    for tag in tags:
        if tag in KNOWN_CATEGORY_TAGS:
            return tag
    return "unknown"


def _write_split_npz(path: Path, sprites: Sequence[_ExportSprite], *, max_palette_slots: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(sprites)
    palette_rows = max_palette_slots + 1
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, palette_rows, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, palette_rows), dtype=bool)
    category_id = np.zeros((count,), dtype=np.int64)
    sprite_id = np.array([sprite.sprite_id for sprite in sprites], dtype=np.str_)

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


def _manifest_record(
    sprite: _ExportSprite,
    split: str,
    config: TrainingExportConfig,
    dedupe_report_provided: bool,
    quality_report_provided: bool,
) -> dict[str, Any]:
    bundle = sprite.bundle
    return {
        "sprite_id": sprite.sprite_id,
        "bundle_path": str(sprite.bundle_path),
        "split": split,
        "category": sprite.category,
        "category_id": sprite.category_id,
        "tags": list(sprite.tags),
        "palette_size": int(np.asarray(bundle.palette).shape[0] - 1),
        "max_palette_slots": config.max_palette_slots,
        "has_role_map": sprite.has_role_map,
        "dedupe_group": sprite.dedupe_group,
        "source_path": bundle.metadata.source,
        "quality_issues": list(sprite.quality_issues),
        "curation_status": "accepted",
        "dedupe_report_provided": dedupe_report_provided,
        "quality_report_provided": quality_report_provided,
    }


def _write_manifest_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _vocab(category_to_id: Mapping[str, int], sprites: Sequence[_ExportSprite]) -> dict[str, Any]:
    tags = sorted({tag for sprite in sprites for tag in sprite.tags})
    return {
        "category_to_id": dict(sorted(category_to_id.items(), key=lambda item: item[1])),
        "tag_to_id": {tag: index for index, tag in enumerate(tags, start=1)},
        "status_to_id": {"accepted": 1},
        "role_names": {str(role_id): name.upper() for role_id, name in sorted(ROLE_NAMES.items())},
    }


def _write_export_config(path: Path, config: TrainingExportConfig, *, accepted_count: int, exported_count: int) -> None:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    data.update(
        {
            "created_timestamp": _utc_timestamp(),
            "accepted_count": accepted_count,
            "exported_count": exported_count,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "codec_version": CODEC_VERSION,
        }
    )
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_lookup(split_assignment: SplitAssignment) -> dict[str, str]:
    lookup = dict.fromkeys(split_assignment.train, "train")
    lookup.update(dict.fromkeys(split_assignment.val, "val"))
    lookup.update(dict.fromkeys(split_assignment.test, "test"))
    return lookup


def _duplicate_leakage(split_assignment: SplitAssignment) -> list[tuple[str, str, str]]:
    split_lookup = _split_lookup(split_assignment)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for sprite_id, group_id in split_assignment.group_by_sprite_id.items():
        group_splits[group_id].add(split_lookup[sprite_id])
    leakage: list[tuple[str, str, str]] = []
    for sprite_id, group_id in sorted(split_assignment.group_by_sprite_id.items()):
        splits = sorted(group_splits[group_id])
        if len(splits) > 1:
            leakage.append((sprite_id, splits[0], splits[-1]))
    return leakage


def _report_exists(path: Path | None, filename: str) -> bool:
    return _resolve_report_path(path, filename) is not None


def _resolve_report_path(path: Path | None, filename: str) -> Path | None:
    if path is None:
        return None
    report_path = Path(path)
    if report_path.is_dir():
        report_path = report_path / filename
    return report_path if report_path.exists() else None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export accepted SpriteBundles to fixed training arrays.")
    parser.add_argument("--bundles", required=True, type=Path, dest="bundle_root")
    parser.add_argument("--curation", required=True, type=Path, dest="curation_path")
    parser.add_argument("--out", required=True, type=Path, dest="output_dir")
    parser.add_argument("--dataset-name", default="sprite_dataset")
    parser.add_argument("--train", type=float, default=0.8, dest="train_fraction")
    parser.add_argument("--val", type=float, default=0.1, dest="val_fraction")
    parser.add_argument("--test", type=float, default=0.1, dest="test_fraction")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-palette-slots", type=int, default=32)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--dedupe-report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = export_training_dataset(
        TrainingExportConfig(
            bundle_root=args.bundle_root,
            curation_path=args.curation_path,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            max_palette_slots=args.max_palette_slots,
            quality_report_path=args.quality_report,
            dedupe_report_path=args.dedupe_report,
            overwrite=args.overwrite,
        )
    )
    print(f"Accepted: {result.accepted_count}")
    print(f"Train: {result.train_count}")
    print(f"Val: {result.val_count}")
    print(f"Test: {result.test_count}")
    print(f"Readiness passed: {result.readiness_passed}")
    print(f"Output: {result.output_dir}")


if __name__ == "__main__":
    main()
