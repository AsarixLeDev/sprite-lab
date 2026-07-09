"""Merge exported semantic-v3 datasets into a single multi-source dataset.

This module concatenates the ``.npz`` rasters and per-split JSONL manifests of
several *already exported* datasets (the layout produced by
:mod:`spritelab.dataset_maker.exporter`) into one merged dataset directory that
still satisfies the strict dataset QA contract:

* semantic_v3 coverage is preserved verbatim;
* label_v2 metadata is preserved verbatim;
* every merged record carries provenance
  (``source_dataset`` / ``source_pack`` / ``source_sprite_id`` /
  ``source_split`` / ``source_npz_row``);
* sprite ids that collide across sources are prefixed with the source pack so
  the merged dataset has globally-unique ids;
* review/quarantine records never leak (they were never exported to begin with,
  and the merge report re-checks this).

The merge is deterministic given the input datasets and seed and never mutates
any source dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# npz arrays written by the exporter, in the order we re-emit them.
_NPZ_ARRAY_KEYS: tuple[str, ...] = (
    "alpha",
    "index_map",
    "role_map",
    "palette",
    "palette_mask",
    "category_id",
    "sprite_id",
)

# Suffixes stripped from a dataset directory name to recover the source pack.
_PACK_SUFFIXES: tuple[str, ...] = (
    "_label_v2_semantic_v3",
    "_semantic_v3",
    "_label_v2",
)

# Statuses / flags that indicate a record leaked from review or quarantine.
_REVIEW_STATUS_TOKENS: frozenset[str] = frozenset({"needs_review", "quarantine", "needs_fix", "rejected", "review"})

SPLIT_POLICIES: tuple[str, ...] = ("preserve", "reshuffle")


class MergeError(RuntimeError):
    """Raised when a merge cannot be completed safely."""


@dataclass
class MergeResult:
    output_dir: Path
    source_datasets: list[str] = field(default_factory=list)
    split_policy: str = "preserve"
    seed: int = 0
    max_palette_slots: int = 32
    total_records: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    source_contributions: dict[str, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)
    base_object_distribution: dict[str, int] = field(default_factory=dict)
    prefixed_sprite_ids: int = 0
    semantic_v3_coverage: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "source_datasets": list(self.source_datasets),
            "split_policy": self.split_policy,
            "seed": self.seed,
            "max_palette_slots": self.max_palette_slots,
            "total_records": self.total_records,
            "split_counts": dict(self.split_counts),
            "source_contributions": dict(self.source_contributions),
            "category_distribution": dict(self.category_distribution),
            "base_object_distribution": dict(self.base_object_distribution),
            "prefixed_sprite_ids": self.prefixed_sprite_ids,
            "semantic_v3_coverage": round(self.semantic_v3_coverage, 6),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass
class _LoadedRecord:
    source_dataset: str
    source_pack: str
    source_split: str
    source_npz_row: int
    original_sprite_id: str
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]  # single-row (unbatched) arrays keyed by npz key


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_datasets(
    dataset_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 20260706,
    split_policy: str = "preserve",
    max_palette_slots: int = 32,
    overwrite: bool = False,
) -> MergeResult:
    """Merge exported semantic-v3 datasets into ``output_dir``.

    Raises :class:`MergeError` for hard, up-front failures (missing source,
    palette overflow, bad split policy). Softer problems are recorded on the
    returned :class:`MergeResult` (which also drives the merge report).
    """

    if split_policy not in SPLIT_POLICIES:
        raise MergeError(f"unknown split policy: {split_policy!r} (expected one of {SPLIT_POLICIES})")
    if not dataset_dirs:
        raise MergeError("no source datasets provided")
    if max_palette_slots < 1:
        raise MergeError("max_palette_slots must be at least 1")

    output_dir = Path(output_dir)
    target_palette_rows = max_palette_slots + 1

    source_paths = [Path(dataset_dir) for dataset_dir in dataset_dirs]
    for source_path in source_paths:
        if not source_path.is_dir():
            raise MergeError(f"source dataset does not exist: {source_path}")

    result = MergeResult(
        output_dir=output_dir,
        source_datasets=[path.name for path in source_paths],
        split_policy=split_policy,
        seed=seed,
        max_palette_slots=max_palette_slots,
    )

    # Load every record from every source, preserving order.
    loaded: list[_LoadedRecord] = []
    for source_path in source_paths:
        loaded.extend(_load_source_dataset(source_path, target_palette_rows=target_palette_rows))

    if not loaded:
        raise MergeError("no records found across the provided source datasets")

    # Resolve sprite-id collisions across sources by prefixing with the pack.
    final_ids = _resolve_sprite_ids(loaded, result)

    # Assign target splits.
    target_splits = _assign_target_splits(loaded, split_policy=split_policy, seed=seed)

    # Build merged category vocab from the union of record categories.
    category_to_id = _merged_category_vocab(loaded)

    # Assemble per-split payloads in a deterministic order (by final sprite id).
    order = sorted(range(len(loaded)), key=lambda i: (target_splits[i], final_ids[i]))

    per_split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    per_split_arrays: dict[str, list[dict[str, np.ndarray]]] = {split: [] for split in SPLIT_NAMES}

    source_contributions: Counter[str] = Counter()
    category_distribution: Counter[str] = Counter()
    base_object_distribution: Counter[str] = Counter()
    semantic_present = 0

    for i in order:
        record = loaded[i]
        final_id = final_ids[i]
        split = target_splits[i]

        manifest = dict(record.manifest)
        manifest["sprite_id"] = final_id
        manifest["split"] = split
        manifest["category_id"] = int(category_to_id.get(str(manifest.get("category", "")) or "unknown", 0))
        manifest["provenance"] = {
            "source_dataset": record.source_dataset,
            "source_pack": record.source_pack,
            "source_sprite_id": record.original_sprite_id,
            "source_split": record.source_split,
            "source_npz_row": record.source_npz_row,
        }
        # Also surface the top-level provenance fields for easy querying.
        manifest["source_dataset"] = record.source_dataset
        manifest["source_pack"] = record.source_pack
        manifest["source_sprite_id"] = record.original_sprite_id
        manifest["source_split"] = record.source_split
        manifest["source_npz_row"] = record.source_npz_row

        arrays = dict(record.arrays)
        arrays["sprite_id"] = np.array(final_id, dtype=np.str_)
        arrays["category_id"] = np.asarray(manifest["category_id"], dtype=np.int64)

        per_split_records[split].append(manifest)
        per_split_arrays[split].append(arrays)

        source_contributions[record.source_dataset] += 1
        category_distribution[str(manifest.get("category", "")) or "unknown"] += 1
        semantic = manifest.get("semantic_v3")
        if isinstance(semantic, Mapping) and semantic:
            semantic_present += 1
            base_object_distribution[str(semantic.get("base_object", "")) or "(none)"] += 1
        else:
            base_object_distribution["(missing)"] += 1

    result.total_records = len(loaded)
    result.split_counts = {split: len(per_split_records[split]) for split in SPLIT_NAMES}
    result.source_contributions = dict(sorted(source_contributions.items()))
    result.category_distribution = dict(sorted(category_distribution.items()))
    result.base_object_distribution = dict(sorted(base_object_distribution.items(), key=lambda kv: (-kv[1], kv[0])))
    result.semantic_v3_coverage = semantic_present / len(loaded) if loaded else 0.0

    _validate_merge(result, loaded, final_ids, per_split_records, per_split_arrays)

    # Write the merged dataset only when the structural contract holds.
    if result.errors:
        return result

    _write_merged_dataset(
        output_dir,
        per_split_records=per_split_records,
        per_split_arrays=per_split_arrays,
        category_to_id=category_to_id,
        max_palette_slots=max_palette_slots,
        overwrite=overwrite,
    )
    write_merge_reports(result, output_dir / "merge_report.json", output_dir / "merge_report.md")
    return result


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_source_dataset(source_path: Path, *, target_palette_rows: int) -> list[_LoadedRecord]:
    source_dataset = source_path.name
    source_pack = derive_source_pack(source_dataset)
    records: list[_LoadedRecord] = []

    for split in SPLIT_NAMES:
        manifest_path = source_path / f"manifest_{split}.jsonl"
        npz_path = source_path / f"{split}.npz"
        if not manifest_path.is_file() and not npz_path.is_file():
            continue
        if not manifest_path.is_file():
            raise MergeError(f"{source_dataset}: {split}.npz present but manifest_{split}.jsonl missing")
        if not npz_path.is_file():
            raise MergeError(f"{source_dataset}: manifest_{split}.jsonl present but {split}.npz missing")

        manifest_records = _read_jsonl(manifest_path)
        arrays_by_row, npz_ids = _load_split_npz(npz_path, target_palette_rows=target_palette_rows)
        row_by_id = {sprite_id: row for row, sprite_id in enumerate(npz_ids)}

        if len(manifest_records) != len(npz_ids):
            raise MergeError(
                f"{source_dataset}:{split}: manifest has {len(manifest_records)} records "
                f"but npz has {len(npz_ids)} rows"
            )

        for manifest in manifest_records:
            sprite_id = str(manifest.get("sprite_id", "")).strip()
            if not sprite_id:
                raise MergeError(f"{source_dataset}:{split}: manifest record with empty sprite_id")
            row = row_by_id.get(sprite_id)
            if row is None:
                raise MergeError(f"{source_dataset}:{split}: manifest sprite {sprite_id} has no matching npz raster")
            records.append(
                _LoadedRecord(
                    source_dataset=source_dataset,
                    source_pack=source_pack,
                    source_split=split,
                    source_npz_row=row,
                    original_sprite_id=sprite_id,
                    manifest=dict(manifest),
                    arrays=arrays_by_row[row],
                )
            )
    return records


def _load_split_npz(npz_path: Path, *, target_palette_rows: int) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    missing = set(_NPZ_ARRAY_KEYS) - set(arrays)
    if missing:
        raise MergeError(f"{npz_path.name}: missing arrays {sorted(missing)}")

    sprite_ids = [str(value) for value in arrays["sprite_id"]]
    count = len(sprite_ids)

    palette = arrays["palette"]
    palette_mask = arrays["palette_mask"]
    source_rows = int(palette.shape[1]) if palette.ndim == 3 else 0
    if source_rows > target_palette_rows:
        raise MergeError(
            f"{npz_path.name}: palette has {source_rows} rows, above target {target_palette_rows} "
            f"(max_palette_slots too small for this source)"
        )

    per_row: list[dict[str, np.ndarray]] = []
    for row in range(count):
        row_arrays: dict[str, np.ndarray] = {
            "alpha": np.asarray(arrays["alpha"][row], dtype=np.uint8),
            "index_map": np.asarray(arrays["index_map"][row], dtype=np.int16),
            "role_map": np.asarray(arrays["role_map"][row], dtype=np.uint8),
        }
        padded_palette = np.zeros((target_palette_rows, 3), dtype=np.uint8)
        padded_mask = np.zeros((target_palette_rows,), dtype=bool)
        padded_palette[:source_rows] = np.asarray(palette[row], dtype=np.uint8)
        padded_mask[:source_rows] = np.asarray(palette_mask[row], dtype=bool)
        row_arrays["palette"] = padded_palette
        row_arrays["palette_mask"] = padded_mask
        row_arrays["category_id"] = np.asarray(arrays["category_id"][row], dtype=np.int64)
        row_arrays["sprite_id"] = np.array(sprite_ids[row], dtype=np.str_)
        per_row.append(row_arrays)
    return per_row, sprite_ids


# ---------------------------------------------------------------------------
# Sprite id + split + vocab resolution
# ---------------------------------------------------------------------------


def _resolve_sprite_ids(loaded: Sequence[_LoadedRecord], result: MergeResult) -> list[str]:
    """Return final sprite ids, prefixing ids that collide across sources.

    An id is prefixed with ``{source_pack}__`` when the same original id appears
    in more than one source dataset, guaranteeing global uniqueness while
    keeping non-colliding ids (e.g. the validated 496 set) untouched.
    """

    id_to_sources: dict[str, set[str]] = {}
    for record in loaded:
        id_to_sources.setdefault(record.original_sprite_id, set()).add(record.source_dataset)
    colliding = {sprite_id for sprite_id, sources in id_to_sources.items() if len(sources) > 1}

    final_ids: list[str] = []
    prefixed = 0
    for record in loaded:
        if record.original_sprite_id in colliding:
            final_ids.append(f"{record.source_pack}__{record.original_sprite_id}")
            prefixed += 1
        else:
            final_ids.append(record.original_sprite_id)
    result.prefixed_sprite_ids = prefixed
    return final_ids


def _assign_target_splits(loaded: Sequence[_LoadedRecord], *, split_policy: str, seed: int) -> list[str]:
    if split_policy == "preserve":
        return [record.source_split for record in loaded]

    # reshuffle: deterministic 0.8/0.1/0.1 split over all records.
    import random

    order = sorted(range(len(loaded)), key=lambda i: (loaded[i].source_dataset, loaded[i].original_sprite_id))
    rng = random.Random(seed)
    rng.shuffle(order)

    total = len(order)
    n_val = int(total * 0.1)
    n_test = int(total * 0.1)
    n_train = total - n_val - n_test
    # Guarantee non-empty val/test when there is enough data.
    if total >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        n_train = total - n_val - n_test

    assignment: list[str] = [""] * len(loaded)
    for rank, i in enumerate(order):
        if rank < n_train:
            assignment[i] = "train"
        elif rank < n_train + n_val:
            assignment[i] = "val"
        else:
            assignment[i] = "test"
    return assignment


def _merged_category_vocab(loaded: Sequence[_LoadedRecord]) -> dict[str, int]:
    categories = {str(record.manifest.get("category", "")) or "unknown" for record in loaded}
    ordered = ["unknown", *sorted(category for category in categories if category != "unknown")]
    return {category: index for index, category in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Validation (merge-specific checks)
# ---------------------------------------------------------------------------


def _validate_merge(
    result: MergeResult,
    loaded: Sequence[_LoadedRecord],
    final_ids: Sequence[str],
    per_split_records: Mapping[str, list[dict[str, Any]]],
    per_split_arrays: Mapping[str, list[dict[str, np.ndarray]]],
) -> None:
    # No duplicate sprite id after merge.
    id_counts = Counter(final_ids)
    duplicates = sorted(sprite_id for sprite_id, count in id_counts.items() if count > 1)
    for sprite_id in duplicates:
        result.errors.append(f"duplicate sprite_id after merge: {sprite_id}")

    # All merged records traceable + provenance complete + no review leakage.
    for split in SPLIT_NAMES:
        records = per_split_records[split]
        arrays = per_split_arrays[split]
        if len(records) != len(arrays):
            result.errors.append(f"{split}: record/array count mismatch ({len(records)} vs {len(arrays)})")
        for record, row_arrays in zip(records, arrays):
            sprite_id = str(record.get("sprite_id", ""))
            for field_name in ("source_dataset", "source_pack", "source_sprite_id", "source_split"):
                if not str(record.get(field_name, "")).strip():
                    result.errors.append(f"{sprite_id}: missing provenance field {field_name}")
            if str(row_arrays.get("sprite_id")) != sprite_id:
                result.errors.append(f"{sprite_id}: npz row sprite_id does not match manifest")
            leak = _review_status_leak(record)
            if leak:
                result.errors.append(f"review/quarantine leaked into merge: {sprite_id}: {leak}")

    # semantic_v3 coverage must be 100%.
    if result.semantic_v3_coverage < 1.0:
        missing = round((1.0 - result.semantic_v3_coverage) * len(loaded))
        result.errors.append(
            f"semantic_v3 coverage is {result.semantic_v3_coverage:.3f} ({missing} records lack semantic_v3)"
        )

    # Splits that existed in a source should not vanish.
    source_splits = {record.source_split for record in loaded}
    for split in source_splits:
        if result.split_counts.get(split, 0) == 0:
            result.warnings.append(f"split {split} existed in a source but has zero merged records")


def _review_status_leak(record: Mapping[str, Any]) -> str:
    status = str(record.get("status", "")).strip().lower()
    if status and status not in {"accepted", "auto", "auto_accepted"} and status in _REVIEW_STATUS_TOKENS:
        return f"status={status}"
    if record.get("needs_review") is True:
        return "needs_review=true"
    if record.get("quarantine") is True:
        return "quarantine=true"
    label_v2 = record.get("label_v2")
    if isinstance(label_v2, Mapping):
        label_quality = label_v2.get("label_quality")
        if isinstance(label_quality, Mapping) and label_quality.get("needs_review") is True:
            return "label_v2.label_quality.needs_review=true"
    return ""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_merged_dataset(
    output_dir: Path,
    *,
    per_split_records: Mapping[str, list[dict[str, Any]]],
    per_split_arrays: Mapping[str, list[dict[str, np.ndarray]]],
    category_to_id: Mapping[str, int],
    max_palette_slots: int,
    overwrite: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise MergeError(f"output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLIT_NAMES:
        records = per_split_records[split]
        arrays = per_split_arrays[split]
        _write_jsonl(output_dir / f"manifest_{split}.jsonl", records)
        _write_stacked_npz(output_dir / f"{split}.npz", arrays, max_palette_slots=max_palette_slots)

    _write_json(
        output_dir / "vocab.json",
        {
            "category_to_id": dict(sorted(category_to_id.items(), key=lambda kv: kv[1])),
            "tag_to_id": {},
            "role_names": {},
        },
    )
    _write_json(
        output_dir / "dataset_config.json",
        {
            "dataset_name": output_dir.name,
            "sprite_size": 32,
            "max_palette_slots": max_palette_slots,
            "created_by": "spritelab.harvest.merge_datasets",
            "format_version": "1.0",
        },
    )


def _write_stacked_npz(path: Path, arrays: Sequence[Mapping[str, np.ndarray]], *, max_palette_slots: int) -> None:
    count = len(arrays)
    palette_rows = max_palette_slots + 1
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    index_map = np.zeros((count, 32, 32), dtype=np.int16)
    role_map = np.zeros((count, 32, 32), dtype=np.uint8)
    palette = np.zeros((count, palette_rows, 3), dtype=np.uint8)
    palette_mask = np.zeros((count, palette_rows), dtype=bool)
    category_id = np.zeros((count,), dtype=np.int64)
    sprite_id = np.array([str(row["sprite_id"]) for row in arrays], dtype=np.str_)

    for row, row_arrays in enumerate(arrays):
        alpha[row] = np.asarray(row_arrays["alpha"], dtype=np.uint8)
        index_map[row] = np.asarray(row_arrays["index_map"], dtype=np.int16)
        role_map[row] = np.asarray(row_arrays["role_map"], dtype=np.uint8)
        palette[row] = np.asarray(row_arrays["palette"], dtype=np.uint8)
        palette_mask[row] = np.asarray(row_arrays["palette_mask"], dtype=bool)
        category_id[row] = int(np.asarray(row_arrays["category_id"]))

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


def write_merge_reports(result: MergeResult, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(format_merge_report(result), encoding="utf-8")


def format_merge_report(result: MergeResult) -> str:
    lines: list[str] = []
    lines.append("# Dataset Merge Report")
    lines.append("")
    lines.append(f"Output: `{result.output_dir}`")
    lines.append(f"Status: **{'PASS' if result.ok else 'FAIL'}**")
    lines.append(f"Split policy: {result.split_policy}")
    lines.append(f"Seed: {result.seed}")
    lines.append(f"Max palette slots: {result.max_palette_slots}")
    lines.append(f"Total records: {result.total_records}")
    lines.append(f"Prefixed sprite ids: {result.prefixed_sprite_ids}")
    lines.append(f"semantic_v3 coverage: {result.semantic_v3_coverage:.3f}")
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    for split in SPLIT_NAMES:
        lines.append(f"- {split}: {result.split_counts.get(split, 0)}")
    lines.append("")
    lines.append("## Source contributions")
    lines.append("")
    for name, count in result.source_contributions.items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Category distribution")
    lines.append("")
    for name, count in result.category_distribution.items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Base object distribution (top 20)")
    lines.append("")
    for name, count in list(result.base_object_distribution.items())[:20]:
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if result.warnings:
        for warning in result.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Errors")
    lines.append("")
    if result.errors:
        for error in result.errors:
            lines.append(f"- {error}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def derive_source_pack(dataset_name: str) -> str:
    """Recover the source pack name from an exported dataset directory name."""

    name = str(dataset_name)
    for suffix in _PACK_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(record), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
