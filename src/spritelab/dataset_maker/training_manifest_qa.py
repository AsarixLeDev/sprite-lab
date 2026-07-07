"""QA gate for generated training manifests.

Validates a ``training_manifest.jsonl`` (produced by
:mod:`spritelab.dataset_maker.training_manifest`) against the exported
semantic-v3 dataset it was built from: every row must reference a real split,
a real npz raster row, an existing base manifest record, and carry a valid
grounded caption plus the semantic/audit metadata a future trainer needs.

Read-only, offline, no torch.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.dataset_maker.training_manifest import (
    FORBIDDEN_CAPTION_CONTENT,
    MAX_CAPTION_LENGTH,
    SCHEMA_VERSION,
    SPLIT_NAMES,
)


@dataclass
class TrainingManifestQAResult:
    dataset_dir: Path
    manifest_path: Path
    total_rows: int = 0
    unique_sprites: int = 0
    split_rows: dict[str, int] = field(default_factory=dict)
    variants_per_sprite: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_json_dict(self) -> dict[str, Any]:
        variant_counts = list(self.variants_per_sprite.values())
        return {
            "dataset_dir": str(self.dataset_dir),
            "manifest_path": str(self.manifest_path),
            "total_rows": self.total_rows,
            "unique_sprites": self.unique_sprites,
            "split_rows": dict(self.split_rows),
            "variants_per_sprite_min": min(variant_counts) if variant_counts else 0,
            "variants_per_sprite_max": max(variant_counts) if variant_counts else 0,
            "variants_per_sprite_avg": (sum(variant_counts) / len(variant_counts)) if variant_counts else 0.0,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


def qa_training_manifest(
    dataset_dir: Path,
    manifest_path: Path,
    *,
    allow_duplicate_captions: bool = False,
) -> TrainingManifestQAResult:
    """Validate a training manifest against its source dataset."""

    dataset_dir = Path(dataset_dir)
    manifest_path = Path(manifest_path)
    result = TrainingManifestQAResult(dataset_dir=dataset_dir, manifest_path=manifest_path)

    if not manifest_path.is_file():
        result.add_error(f"training manifest not found: {manifest_path}")
        return result

    rows = _read_jsonl(manifest_path, result)
    result.total_rows = len(rows)
    if not rows:
        result.add_error("training manifest is empty")
        return result

    source_records, source_splits = _load_source_records(dataset_dir)
    npz_sprite_rows = _load_npz_sprite_rows(dataset_dir)

    per_sprite: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    duplicate_pairs: Counter[tuple[str, str]] = Counter()

    forbidden_content: list[str] = []
    bad_caption: list[str] = []
    missing_schema: list[str] = []
    bad_split: list[str] = []
    npz_mismatch: list[str] = []
    npz_out_of_range: list[str] = []
    sprite_id_mismatch: list[str] = []
    missing_source_record: list[str] = []
    missing_caption_type: list[str] = []
    missing_core_field: list[str] = []
    missing_semantic_schema: list[str] = []
    missing_negative_tags: list[str] = []

    for line_no, row in enumerate(rows, start=1):
        sprite_id = str(row.get("sprite_id", "")).strip()
        location = sprite_id or f"row {line_no}"
        per_sprite[sprite_id] += 1

        if str(row.get("schema_version", "")).strip() != SCHEMA_VERSION:
            missing_schema.append(location)

        split = str(row.get("split", "")).strip()
        if split not in SPLIT_NAMES:
            bad_split.append(f"{location}: split={split!r}")
        else:
            split_rows[split] += 1

        caption = row.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            bad_caption.append(f"{location}: missing/empty caption")
        else:
            lowered = caption.lower()
            if len(caption) > MAX_CAPTION_LENGTH:
                bad_caption.append(f"{location}: caption longer than {MAX_CAPTION_LENGTH} chars")
            for forbidden in FORBIDDEN_CAPTION_CONTENT:
                if forbidden in lowered:
                    forbidden_content.append(f"{location}: '{forbidden}' in caption")
            if _has_repeated_word(lowered):
                bad_caption.append(f"{location}: duplicated adjacent word in caption")
            duplicate_pairs[(sprite_id, caption.strip().lower())] += 1

        if not str(row.get("caption_type", "")).strip():
            missing_caption_type.append(location)

        for field_name in ("category", "object_name", "base_object"):
            if not str(row.get(field_name, "")).strip():
                missing_core_field.append(f"{location}: {field_name}")

        audit = row.get("audit") if isinstance(row.get("audit"), Mapping) else {}
        if not str(audit.get("semantic_schema_version", "")).strip():
            missing_semantic_schema.append(location)

        negative_tags = row.get("negative_tags")
        if not isinstance(negative_tags, list) or not negative_tags:
            missing_negative_tags.append(location)

        # Split / npz cross-checks against the source dataset.
        npz_file = str(row.get("npz_file", "")).strip()
        npz_path = dataset_dir / npz_file if npz_file else None
        if npz_file and (npz_path is None or not npz_path.is_file()):
            npz_mismatch.append(f"{location}: npz_file {npz_file!r} does not exist")
        elif split in SPLIT_NAMES and npz_file and npz_file != f"{split}.npz":
            npz_mismatch.append(f"{location}: npz_file {npz_file!r} does not match split {split!r}")

        sprite_rows = npz_sprite_rows.get(split)
        npz_row = row.get("npz_row")
        if sprite_rows is not None and isinstance(npz_row, int):
            if npz_row < 0 or npz_row >= len(sprite_rows):
                npz_out_of_range.append(f"{location}: npz_row {npz_row} out of range [0,{len(sprite_rows)})")
            elif sprite_id and sprite_rows[npz_row] != sprite_id:
                sprite_id_mismatch.append(
                    f"{location}: npz_row {npz_row} holds {sprite_rows[npz_row]!r}, not {sprite_id!r}"
                )

        source_key = (split, sprite_id)
        if source_records and source_key not in source_records:
            missing_source_record.append(location)

    duplicate_captions = sorted(
        f"{sprite_id}: {caption!r} x{count}"
        for (sprite_id, caption), count in duplicate_pairs.items()
        if count > 1
    )

    result.unique_sprites = len([sid for sid in per_sprite if sid])
    result.split_rows = dict(split_rows)
    result.variants_per_sprite = dict(per_sprite)

    variant_values = [count for sid, count in per_sprite.items() if sid]
    result.checks = {
        "schema_version": SCHEMA_VERSION,
        "variants_per_sprite_min": min(variant_values) if variant_values else 0,
        "variants_per_sprite_max": max(variant_values) if variant_values else 0,
        "source_split_rows": dict(source_splits),
        "manifest_split_rows": dict(split_rows),
        "missing_schema_version": missing_schema,
        "bad_split": bad_split,
        "npz_file_mismatch": npz_mismatch,
        "npz_row_out_of_range": npz_out_of_range,
        "sprite_id_mismatch": sprite_id_mismatch,
        "missing_source_record": missing_source_record,
        "bad_caption": bad_caption,
        "forbidden_caption_content": forbidden_content,
        "missing_caption_type": missing_caption_type,
        "missing_core_field": missing_core_field,
        "missing_semantic_schema_version": missing_semantic_schema,
        "missing_negative_tags": missing_negative_tags,
        "duplicate_captions": duplicate_captions,
        "allow_duplicate_captions": bool(allow_duplicate_captions),
    }

    # Errors.
    for location in missing_schema:
        result.add_error(f"row has wrong/missing schema_version: {location}")
    for entry in bad_split:
        result.add_error(f"row references invalid split: {entry}")
    for entry in npz_mismatch:
        result.add_error(f"npz_file problem: {entry}")
    for entry in npz_out_of_range:
        result.add_error(f"npz_row out of range: {entry}")
    for entry in sprite_id_mismatch:
        result.add_error(f"sprite_id does not match npz row: {entry}")
    for location in missing_source_record:
        result.add_error(f"row references sprite absent from source manifest: {location}")
    for entry in bad_caption:
        result.add_error(f"invalid caption: {entry}")
    for entry in forbidden_content:
        result.add_error(f"forbidden caption content: {entry}")
    for location in missing_caption_type:
        result.add_error(f"row missing caption_type: {location}")
    for entry in missing_core_field:
        result.add_error(f"row missing core field: {entry}")
    for location in missing_semantic_schema:
        result.add_error(f"row missing audit.semantic_schema_version: {location}")

    # Warnings (or errors when duplicates are disallowed).
    for location in missing_negative_tags:
        result.add_warning(f"row has no negative_tags: {location}")

    if duplicate_captions:
        if allow_duplicate_captions:
            result.add_warning(f"{len(duplicate_captions)} duplicate (sprite_id, caption) pairs")
        else:
            for entry in duplicate_captions:
                result.add_error(f"duplicate (sprite_id, caption): {entry}")

    _check_split_distribution(result, split_rows, source_splits)
    _check_variant_consistency(result, variant_values)

    return result


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------


def _check_split_distribution(
    result: TrainingManifestQAResult,
    manifest_split_rows: Mapping[str, int],
    source_splits: Mapping[str, int],
) -> None:
    manifest_total = sum(manifest_split_rows.values())
    source_total = sum(source_splits.values())
    if not manifest_total or not source_total:
        return
    for split in SPLIT_NAMES:
        manifest_fraction = manifest_split_rows.get(split, 0) / manifest_total
        source_fraction = source_splits.get(split, 0) / source_total
        if abs(manifest_fraction - source_fraction) > 0.02:
            result.add_warning(
                f"split {split} fraction {manifest_fraction:.3f} deviates from source {source_fraction:.3f}"
            )


def _check_variant_consistency(result: TrainingManifestQAResult, variant_values: Sequence[int]) -> None:
    if not variant_values:
        return
    low, high = min(variant_values), max(variant_values)
    if low != high:
        result.add_warning(f"variants per sprite not uniform: min={low} max={high}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path, result: TrainingManifestQAResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            result.add_error(f"{path.name}:{line_no}: invalid JSON line")
            continue
        if isinstance(record, dict):
            rows.append(record)
        else:
            result.add_error(f"{path.name}:{line_no}: row is not a JSON object")
    return rows


def _load_source_records(dataset_dir: Path) -> tuple[set[tuple[str, str]], dict[str, int]]:
    keys: set[tuple[str, str]] = set()
    split_counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"manifest_{split}.jsonl"
        if not path.is_file():
            continue
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                keys.add((split, str(record.get("sprite_id", "")).strip()))
                count += 1
        split_counts[split] = count
    return keys, split_counts


def _load_npz_sprite_rows(dataset_dir: Path) -> dict[str, list[str] | None]:
    rows: dict[str, list[str] | None] = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"{split}.npz"
        if not path.is_file():
            rows[split] = None
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if "sprite_id" not in data.files:
                    rows[split] = None
                    continue
                rows[split] = [str(value) for value in np.asarray(data["sprite_id"])]
        except Exception:  # pragma: no cover - defensive
            rows[split] = None
    return rows


def _has_repeated_word(lowered_caption: str) -> bool:
    words = lowered_caption.replace(",", " ").split()
    return any(words[i] == words[i + 1] for i in range(len(words) - 1))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_markdown(result: TrainingManifestQAResult) -> str:
    variant_values = [count for sid, count in result.variants_per_sprite.items() if sid]
    low = min(variant_values) if variant_values else 0
    high = max(variant_values) if variant_values else 0
    avg = (sum(variant_values) / len(variant_values)) if variant_values else 0.0
    lines = [
        "# Training Manifest QA Report",
        "",
        f"Manifest: `{result.manifest_path}`",
        f"Dataset: `{result.dataset_dir}`",
        f"Status: **{'PASS' if result.ok else 'FAIL'}**",
        f"Rows: {result.total_rows}",
        f"Unique sprites: {result.unique_sprites}",
        f"Variants per sprite: min={low} max={high} avg={avg:.1f}",
        f"Errors: {len(result.errors)}",
        f"Warnings: {len(result.warnings)}",
        "",
        "## Split rows",
    ]
    for split in SPLIT_NAMES:
        lines.append(f"- {split}: {result.split_rows.get(split, 0)}")
    lines.extend(["", "## Warnings"])
    if result.warnings:
        for warning in result.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Errors"])
    if result.errors:
        for error in result.errors:
            lines.append(f"- {error}")
    else:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


def write_training_manifest_qa_reports(
    result: TrainingManifestQAResult, *, out_json: Path, out_md: Path
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(result.to_markdown(), encoding="utf-8")
