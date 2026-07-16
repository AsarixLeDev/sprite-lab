"""Dataset QA gate for exported Dataset Maker datasets.

This module validates an *exported* dataset directory (the layout produced by
:mod:`spritelab.dataset_maker.exporter`) without mutating it, without any
network access, without a GPU, and without any fresh Qwen/VLM calls.

The exporter writes, per dataset directory:

* ``manifest_train.jsonl`` / ``manifest_val.jsonl`` / ``manifest_test.jsonl`` --
  one JSON record per accepted sprite (there is no unified manifest);
* ``train.npz`` / ``val.npz`` / ``test.npz`` -- the raster payload, with arrays
  ``alpha`` ``(N, 32, 32)``, ``index_map`` ``(N, 32, 32)``, ``role_map``,
  ``palette`` ``(N, P, 3)``, ``palette_mask`` ``(N, P)``, ``category_id`` and
  ``sprite_id``;
* ``dataset_config.json`` / ``vocab.json`` / ``rejected.jsonl`` /
  ``dataset_report.md``.

There are **no per-sprite PNG files** -- every sprite raster lives inside the
split ``.npz``. The "image checks" below therefore operate on the npz rasters.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.harvest.config_loader import labeling_config_metadata, load_hallucination_denylist_config
from spritelab.harvest.label_schema import confidence_tier_for_bucket

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# Malformed object names that must never reach an exported dataset.
FORBIDDEN_OBJECT_NAMES: frozenset[str] = frozenset({"sho", "armour", "elm"})

# Bare colour tokens that must not become object names for 496 potion records.
POTION_COLOR_ONLY_NAMES: frozenset[str] = frozenset({"red", "blue", "green", "yellow", "pink", "white", "orange"})

# Statuses that indicate a record leaked out of review/quarantine into the
# accepted export.
REVIEW_STATUS_TOKENS: frozenset[str] = frozenset({"needs_review", "quarantine", "needs_fix", "rejected", "review"})

_EXPECTED_NPZ_KEYS: frozenset[str] = frozenset(
    {"alpha", "index_map", "role_map", "palette", "palette_mask", "category_id", "sprite_id"}
)

# semantic_v3 caption hygiene: these must never appear inside a caption.
FORBIDDEN_CAPTION_CONTENT: tuple[str, ...] = ("photorealistic", "watermark", "text overlay")

# Hard cap on caption length; anything longer is a caption-generation bug.
MAX_SEMANTIC_CAPTION_LENGTH = 300

# Base objects whose sprites are expected to carry color information.
_COLOR_EXPECTED_BASE_OBJECTS: frozenset[str] = frozenset({"gem", "crystal", "potion", "vial", "bottle", "flask", "jar"})


@dataclass
class DatasetQAResult:
    """Structured outcome of :func:`qa_dataset`."""

    dataset_dir: Path
    total_records: int = 0
    total_images: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    image_checks: dict[str, Any] = field(default_factory=dict)
    manifest_checks: dict[str, Any] = field(default_factory=dict)
    label_v2_checks: dict[str, Any] = field(default_factory=dict)
    label_audits: dict[str, Any] = field(default_factory=dict)
    semantic_v3_checks: dict[str, Any] = field(default_factory=dict)
    split_checks: dict[str, Any] = field(default_factory=dict)
    review_queue_overlap: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "total_records": self.total_records,
            "total_images": self.total_images,
            "splits": dict(self.splits),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "counts": {
                "categories": dict(self.counts.get("categories", {})),
                "objects": dict(self.counts.get("objects", {})),
                "sources": dict(self.counts.get("sources", {})),
                "label_v2_buckets": dict(self.counts.get("label_v2_buckets", {})),
                "splits": dict(self.counts.get("splits", {})),
            },
            "image_checks": dict(self.image_checks),
            "manifest_checks": dict(self.manifest_checks),
            "label_v2_checks": dict(self.label_v2_checks),
            "label_audits": dict(self.label_audits),
            "labeling_v2_config": labeling_config_metadata(),
            "semantic_v3_checks": dict(self.semantic_v3_checks),
            "split_checks": dict(self.split_checks),
            "review_queue_overlap": list(self.review_queue_overlap),
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


def qa_dataset(
    dataset_dir: Path,
    *,
    sample_limit: int = 64,
    review_queue: Path | None = None,
    expected_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_object_share: float = 0.20,
    strict: bool = False,
    require_semantic_v3: bool = False,
) -> DatasetQAResult:
    """Validate an exported dataset directory. Never mutates the dataset.

    ``strict`` escalates the normally-soft raster warnings (all-transparent
    sprites, single-tag records) to errors. ``require_semantic_v3`` makes
    missing ``semantic_v3`` metadata an error on every record; without it,
    semantic checks only apply to records that carry the metadata.
    """

    dataset_dir = Path(dataset_dir)
    result = DatasetQAResult(dataset_dir=dataset_dir)

    if not dataset_dir.is_dir():
        result.add_error(f"dataset directory does not exist: {dataset_dir}")
        return result

    config = _load_json(dataset_dir / "dataset_config.json")
    vocab = _load_json(dataset_dir / "vocab.json")
    max_palette_slots = int(config.get("max_palette_slots", 32)) if config else 32

    manifests = _load_manifests(dataset_dir, result)
    npz_by_split = _load_npz(dataset_dir, result)

    all_records: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        all_records.extend(manifests.get(split, []))

    result.total_records = len(all_records)
    result.splits = {split: len(manifests.get(split, [])) for split in SPLIT_NAMES}

    _check_images(result, manifests, npz_by_split, max_palette_slots=max_palette_slots, strict=strict)
    _check_manifests(result, dataset_dir, all_records, config, strict=strict)
    _check_label_v2(result, all_records)
    _run_label_v2_audits(result, all_records)
    _check_semantic_v3(result, all_records, require_semantic_v3=require_semantic_v3)
    _check_splits(result, manifests, npz_by_split, expected_fractions=expected_fractions)
    _build_counts(result, all_records, vocab)
    _check_distribution(result, all_records, vocab, max_object_share=max_object_share)
    _check_review_queue(result, all_records, review_queue)

    result.total_images = sum(int(payload["count"]) for payload in npz_by_split.values() if payload is not None)
    return result


def compare_dataset_manifest_parity(before_dir: Path, after_dir: Path) -> dict[str, Any]:
    """Compare labeling/semantic outputs while ignoring additive v2 metadata.

    This is intentionally read-only and reports every changed sprite row. It
    is suitable for a pre/post export gate as well as an in-place no-change
    smoke check.
    """

    before = _manifest_rows_by_id(Path(before_dir))
    after = _manifest_rows_by_id(Path(after_dir))
    changed: list[dict[str, Any]] = []
    for sprite_id in sorted(set(before) | set(after)):
        if sprite_id not in before or sprite_id not in after:
            changed.append(
                {
                    "sprite_id": sprite_id,
                    "reason": "missing_row",
                    "before_present": sprite_id in before,
                    "after_present": sprite_id in after,
                }
            )
            continue
        old = _parity_payload(before[sprite_id])
        new = _parity_payload(after[sprite_id])
        if old != new:
            changed.append({"sprite_id": sprite_id, "reason": "semantic_output_changed", "before": old, "after": new})
    return {
        "before_dir": str(before_dir),
        "after_dir": str(after_dir),
        "checked_rows": len(set(before) | set(after)),
        "unexpected_changed_rows": changed,
        "unexpected_changed_count": len(changed),
        "ok": not changed,
    }


def _manifest_rows_by_id(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"manifest_{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and str(record.get("sprite_id", "")):
                rows[str(record["sprite_id"])] = record
    return rows


def _parity_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    label_v2 = record.get("label_v2") if isinstance(record.get("label_v2"), Mapping) else {}
    safe = label_v2.get("safe_prefill") if isinstance(label_v2.get("safe_prefill"), Mapping) else {}
    quality = label_v2.get("label_quality") if isinstance(label_v2.get("label_quality"), Mapping) else {}
    return {
        "split": record.get("split"),
        "category": record.get("category"),
        "object_name": record.get("object_name"),
        "tags": record.get("tags"),
        "materials": record.get("materials"),
        "dominant_colors": safe.get("dominant_colors", record.get("dominant_colors")),
        "description": safe.get("short_description", record.get("short_description")),
        "confidence": safe.get("confidence"),
        "confidence_reason": safe.get("confidence_reason"),
        "review_status": {
            "needs_review": quality.get("needs_review", label_v2.get("needs_review")),
            "bucket": label_v2.get("bucket"),
        },
        "semantic_v3": record.get("semantic_v3"),
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_manifests(dataset_dir: Path, result: DatasetQAResult) -> dict[str, list[dict[str, Any]]]:
    manifests: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"manifest_{split}.jsonl"
        if not path.is_file():
            result.add_error(f"missing split manifest: {path.name}")
            manifests[split] = []
            continue
        records: list[dict[str, Any]] = []
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
                records.append(record)
            else:
                result.add_error(f"{path.name}:{line_no}: record is not a JSON object")
        manifests[split] = records
    return manifests


def _load_npz(dataset_dir: Path, result: DatasetQAResult) -> dict[str, dict[str, Any] | None]:
    payloads: dict[str, dict[str, Any] | None] = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"{split}.npz"
        if not path.is_file():
            result.add_error(f"missing split npz: {path.name}")
            payloads[split] = None
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
        except Exception as exc:  # pragma: no cover - defensive
            result.add_error(f"unreadable npz {path.name}: {exc}")
            payloads[split] = None
            continue
        missing = _EXPECTED_NPZ_KEYS - set(arrays)
        if missing:
            result.add_error(f"{path.name}: missing arrays {sorted(missing)}")
        sprite_ids = [str(value) for value in arrays.get("sprite_id", np.array([]))]
        payloads[split] = {"arrays": arrays, "sprite_ids": sprite_ids, "count": len(sprite_ids)}
    return payloads


# ---------------------------------------------------------------------------
# Image checks (operate on the npz rasters)
# ---------------------------------------------------------------------------


def _check_images(
    result: DatasetQAResult,
    manifests: Mapping[str, list[dict[str, Any]]],
    npz_by_split: Mapping[str, dict[str, Any] | None],
    *,
    max_palette_slots: int,
    strict: bool,
) -> None:
    bad_dimensions: list[str] = []
    unreadable: list[str] = []
    missing_images: list[str] = []
    unreferenced: list[str] = []
    empty_images: list[str] = []
    bad_alpha: list[str] = []
    palette_overflow: list[str] = []
    contract_violations: list[str] = []
    max_palette_seen = 0

    for split in SPLIT_NAMES:
        payload = npz_by_split.get(split)
        manifest_ids = [str(record.get("sprite_id", "")) for record in manifests.get(split, [])]
        if payload is None:
            unreadable.append(f"{split}.npz")
            missing_images.extend(manifest_ids)
            continue

        arrays = payload["arrays"]
        npz_ids: list[str] = payload["sprite_ids"]
        npz_id_set = set(npz_ids)
        manifest_id_set = set(manifest_ids)

        missing_images.extend(sorted(manifest_id_set - npz_id_set))
        unreferenced.extend(sorted(npz_id_set - manifest_id_set))

        alpha = arrays.get("alpha")
        index_map = arrays.get("index_map")
        palette = arrays.get("palette")
        palette_mask = arrays.get("palette_mask")

        if alpha is None or alpha.ndim != 3 or alpha.shape[1:] != (32, 32):
            shape = None if alpha is None else tuple(alpha.shape)
            bad_dimensions.append(f"{split}: alpha shape {shape} is not (N, 32, 32)")
        if index_map is None or index_map.ndim != 3 or index_map.shape[1:] != (32, 32):
            shape = None if index_map is None else tuple(index_map.shape)
            bad_dimensions.append(f"{split}: index_map shape {shape} is not (N, 32, 32)")

        # Per-sprite raster contract checks (only if shapes are usable).
        if (
            alpha is not None
            and alpha.ndim == 3
            and alpha.shape[1:] == (32, 32)
            and index_map is not None
            and index_map.shape == alpha.shape
        ):
            for row in range(alpha.shape[0]):
                sprite_id = npz_ids[row] if row < len(npz_ids) else f"{split}#{row}"
                a = alpha[row]
                idx = index_map[row]
                if not np.all(np.isin(a, (0, 1))):
                    bad_alpha.append(sprite_id)
                if not bool(np.any(a == 1)):
                    empty_images.append(sprite_id)
                if bool(np.any((a == 0) & (idx != 0))) or bool(np.any((a == 1) & (idx < 1))):
                    contract_violations.append(sprite_id)

                if palette_mask is not None and row < palette_mask.shape[0]:
                    visible = int(palette_mask[row].sum()) - 1  # row 0 is the transparent slot
                    max_palette_seen = max(max_palette_seen, visible)
                    if visible > max_palette_slots:
                        palette_overflow.append(f"{sprite_id}: {visible} slots")
                    max_index = int(idx.max()) if idx.size else 0
                    if max_index > visible:
                        contract_violations.append(f"{sprite_id}: index {max_index} > {visible} palette rows")
                if palette is not None and row < palette.shape[0]:
                    if not np.array_equal(palette[row, 0], np.array([0, 0, 0], dtype=palette.dtype)):
                        contract_violations.append(f"{sprite_id}: palette row 0 is not [0, 0, 0]")

    all_32x32 = not bad_dimensions
    result.image_checks = {
        "all_32x32": all_32x32,
        "max_palette_slots_seen": max_palette_seen,
        "max_palette_slots_allowed": max_palette_slots,
        "missing_images": missing_images,
        "unreferenced_images": unreferenced,
        "bad_dimensions": bad_dimensions,
        "unreadable_images": unreadable,
        "empty_images": empty_images,
        "bad_alpha": bad_alpha,
        "palette_overflow": palette_overflow,
        "contract_violations": contract_violations,
    }

    for name in missing_images:
        result.add_error(f"manifest references sprite with no raster in npz: {name}")
    for name in unreferenced:
        result.add_error(f"npz raster not present in manifest: {name}")
    for message in bad_dimensions:
        result.add_error(f"image dimensions: {message}")
    for name in unreadable:
        result.add_error(f"unreadable image payload: {name}")
    for name in bad_alpha:
        result.add_error(f"sprite has non-binary alpha (soft alpha): {name}")
    for name in palette_overflow:
        result.add_error(f"palette exceeds max_palette_slots: {name}")
    for message in contract_violations:
        result.add_error(f"raster contract violation: {message}")
    for name in empty_images:
        message = f"sprite is fully transparent (empty raster): {name}"
        result.add_error(message) if strict else result.add_warning(message)


# ---------------------------------------------------------------------------
# Manifest checks
# ---------------------------------------------------------------------------


def _check_manifests(
    result: DatasetQAResult,
    dataset_dir: Path,
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    strict: bool,
) -> None:
    seen_ids: Counter[str] = Counter()
    duplicate_sprite_ids: list[str] = []
    missing_object_name: list[str] = []
    missing_tags: list[str] = []
    single_tag: list[str] = []
    missing_category: list[str] = []
    missing_source: list[str] = []
    missing_license: list[str] = []
    review_status_leaks: list[str] = []
    forbidden_object_names: list[str] = []
    color_only_potions: list[str] = []

    dataset_name = str(config.get("dataset_name", "")) if config else ""
    is_496 = "496" in dataset_name or "496" in dataset_dir.name

    for record in records:
        sprite_id = str(record.get("sprite_id", "")).strip()
        if not sprite_id:
            result.add_error("manifest record has empty sprite_id")
            continue
        seen_ids[sprite_id] += 1

        object_name = str(record.get("object_name", "")).strip()
        if not object_name:
            missing_object_name.append(sprite_id)
        elif object_name.lower() in FORBIDDEN_OBJECT_NAMES:
            forbidden_object_names.append(f"{sprite_id}: {object_name}")

        tags = record.get("tags")
        if not isinstance(tags, list) or not [str(tag) for tag in tags if str(tag).strip()]:
            missing_tags.append(sprite_id)
        elif len([str(tag) for tag in tags if str(tag).strip()]) == 1:
            single_tag.append(sprite_id)

        if not str(record.get("category", "")).strip():
            missing_category.append(sprite_id)
        if not str(record.get("source_name", "")).strip():
            missing_source.append(sprite_id)
        if not str(record.get("license", "")).strip():
            missing_license.append(sprite_id)

        leak = _review_status_leak(record)
        if leak:
            review_status_leaks.append(f"{sprite_id}: {leak}")

        if is_496 and _is_potion_record(record) and object_name.lower() in POTION_COLOR_ONLY_NAMES:
            color_only_potions.append(f"{sprite_id}: {object_name}")

    duplicate_sprite_ids = sorted(sprite_id for sprite_id, count in seen_ids.items() if count > 1)

    result.manifest_checks = {
        "duplicate_sprite_ids": duplicate_sprite_ids,
        "missing_object_name": missing_object_name,
        "missing_tags": missing_tags,
        "single_tag_records": single_tag,
        "missing_category": missing_category,
        "missing_source_name": missing_source,
        "missing_license": missing_license,
        "review_status_leaks": review_status_leaks,
        "forbidden_object_names": forbidden_object_names,
        "color_only_potion_object_names": color_only_potions,
    }

    for sprite_id in duplicate_sprite_ids:
        result.add_error(f"duplicate sprite_id across manifests: {sprite_id}")
    for sprite_id in missing_object_name:
        result.add_error(f"record missing object_name: {sprite_id}")
    for sprite_id in missing_tags:
        result.add_error(f"record has empty tags: {sprite_id}")
    for sprite_id in missing_category:
        result.add_error(f"record missing category: {sprite_id}")
    for sprite_id in missing_source:
        result.add_error(f"record missing source_name: {sprite_id}")
    for sprite_id in missing_license:
        result.add_error(f"record missing license: {sprite_id}")
    for entry in review_status_leaks:
        result.add_error(f"review/quarantine status leaked into export: {entry}")
    for entry in forbidden_object_names:
        result.add_error(f"forbidden malformed object_name: {entry}")
    for entry in color_only_potions:
        result.add_error(f"colour-only potion object_name (496): {entry}")
    for sprite_id in single_tag:
        message = f"record has only one tag: {sprite_id}"
        result.add_error(message) if strict else result.add_warning(message)


def _review_status_leak(record: Mapping[str, Any]) -> str:
    """Return a non-empty reason if the record looks like it leaked from review."""

    status = str(record.get("status", "")).strip().lower()
    if status and status not in {"accepted", "auto", "auto_accepted"}:
        if status in REVIEW_STATUS_TOKENS:
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


def _is_potion_record(record: Mapping[str, Any]) -> bool:
    source_path = str(record.get("source_path", ""))
    name = Path(source_path.replace("\\", "/")).name
    if name[:2].lower() == "p_":
        return True
    sprite_id = str(record.get("sprite_id", ""))
    return "_p_" in sprite_id.lower()


# ---------------------------------------------------------------------------
# Label-v2 checks
# ---------------------------------------------------------------------------


def _check_label_v2(result: DatasetQAResult, records: Sequence[Mapping[str, Any]]) -> None:
    records_with_label_v2 = [r for r in records if isinstance(r.get("label_v2"), Mapping) and r.get("label_v2")]
    is_label_v2_dataset = bool(records_with_label_v2)

    missing_label_v2: list[str] = []
    not_applied: list[str] = []
    missing_bucket: list[str] = []
    missing_flags: list[str] = []

    if is_label_v2_dataset:
        for record in records:
            sprite_id = str(record.get("sprite_id", ""))
            label_v2 = record.get("label_v2")
            if not isinstance(label_v2, Mapping) or not label_v2:
                missing_label_v2.append(sprite_id)
                continue
            if label_v2.get("applied") is not True:
                not_applied.append(sprite_id)
            if not str(label_v2.get("bucket", "")).strip():
                missing_bucket.append(sprite_id)
            if "flags" not in label_v2:
                missing_flags.append(sprite_id)

    result.label_v2_checks = {
        "is_label_v2_dataset": is_label_v2_dataset,
        "records_with_label_v2": len(records_with_label_v2),
        "missing_label_v2": missing_label_v2,
        "applied_not_true": not_applied,
        "missing_bucket": missing_bucket,
        "missing_flags": missing_flags,
    }

    for sprite_id in missing_label_v2:
        result.add_error(f"label-v2 dataset record missing label_v2 metadata: {sprite_id}")
    for sprite_id in not_applied:
        result.add_error(f"label_v2.applied is not true: {sprite_id}")
    for sprite_id in missing_bucket:
        result.add_error(f"label_v2.bucket missing: {sprite_id}")
    for sprite_id in missing_flags:
        result.add_error(f"label_v2.flags missing: {sprite_id}")


def _run_label_v2_audits(result: DatasetQAResult, records: Sequence[Mapping[str, Any]]) -> None:
    """Emit report-only Labeling v2 audit entries; never rewrite or reject rows."""

    fallback = {"vlm_hallucination_objects": [], "malformed_objects": [], "generic_objects": []}
    denylist = load_hallucination_denylist_config(fallback)
    hallucinations = {str(value) for value in denylist["vlm_hallucination_objects"]}
    entries: list[dict[str, Any]] = []
    unavailable: Counter[str] = Counter()

    filename_vlm_mismatch: list[tuple[str, dict[str, Any]]] = []
    hallucination_hits: list[tuple[str, dict[str, Any]]] = []
    hallucination_suppressed: list[tuple[str, dict[str, Any]]] = []
    color_contradictions: list[tuple[str, dict[str, Any]]] = []
    role_contradictions: list[tuple[str, dict[str, Any]]] = []
    shape_contradictions: list[tuple[str, dict[str, Any]]] = []
    potion_color_only: list[tuple[str, dict[str, Any]]] = []
    gem_color_only: list[tuple[str, dict[str, Any]]] = []
    color_expected_missing: list[tuple[str, dict[str, Any]]] = []
    color_dominates: list[tuple[str, dict[str, Any]]] = []
    colors = POTION_COLOR_ONLY_NAMES
    color_words = colors | {
        "black",
        "brown",
        "gray",
        "grey",
        "purple",
        "violet",
        "pink",
        "magenta",
        "cyan",
        "navy",
        "teal",
        "gold",
        "silver",
        "lime",
        "crimson",
    }

    category_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        sprite_id = str(record.get("sprite_id", ""))
        category = str(record.get("category", "") or "unknown")
        category_records.setdefault(category, []).append(record)
        label_v2 = record.get("label_v2") if isinstance(record.get("label_v2"), Mapping) else {}
        filename = (
            label_v2.get("filename_suggestion") if isinstance(label_v2.get("filename_suggestion"), Mapping) else None
        )
        vlm = label_v2.get("vlm_descriptor") if isinstance(label_v2.get("vlm_descriptor"), Mapping) else None
        safe = label_v2.get("safe_prefill") if isinstance(label_v2.get("safe_prefill"), Mapping) else {}
        tier = _record_label_tier(record, label_v2)

        quality = label_v2.get("label_quality") if isinstance(label_v2.get("label_quality"), Mapping) else {}
        fusion_flags = {str(value) for value in quality.get("flags") or label_v2.get("flags") or ()}
        conflict_reasons = [
            str(value) for value in quality.get("conflict_reasons") or label_v2.get("conflict_reasons") or ()
        ]
        if vlm is None:
            unavailable["filename_vlm_category_mismatch"] += 1
        elif "vlm_conflicts_with_filename" in fusion_flags:
            filename_vlm_mismatch.append(
                (
                    sprite_id,
                    {
                        "tier": tier,
                        "evidence_source": "fusion_flags",
                        "fusion_flags": sorted(fusion_flags),
                        "conflict_reasons": conflict_reasons,
                    },
                )
            )
        elif filename is not None:
            try:
                filename_confidence = float(filename.get("confidence", 0.0))
            except (TypeError, ValueError):
                filename_confidence = 0.0
            filename_category = str(filename.get("category", "unknown"))
            vlm_category = str(vlm.get("category", vlm.get("possible_category", "unknown")))
            if (
                filename_confidence >= 0.85
                and filename_category != "unknown"
                and vlm_category != "unknown"
                and filename_category != vlm_category
            ):
                filename_vlm_mismatch.append(
                    (
                        sprite_id,
                        {
                            "filename_category": filename_category,
                            "vlm_category": vlm_category,
                            "filename_confidence": filename_confidence,
                            "tier": tier,
                            "evidence_source": "raw_filename_suggestion",
                        },
                    )
                )

        if vlm is None:
            unavailable["vlm_hallucination_denylist_hit"] += 1
        else:
            object_name = str(vlm.get("object_name") or vlm.get("possible_object_name") or "")
            if object_name in hallucinations:
                evidence = {
                    "vlm_object_name": object_name,
                    "safe_object_name": str(safe.get("object_name") or record.get("object_name") or ""),
                    "tier": tier,
                    "fusion_flags": sorted(fusion_flags),
                }
                if _trusted_same_object_family(evidence, fusion_flags):
                    hallucination_suppressed.append((sprite_id, {**evidence, "reason": "trusted_same_object_family"}))
                else:
                    hallucination_hits.append((sprite_id, evidence))

        object_name = str(record.get("object_name", ""))
        tokens = object_name.split("_")
        is_color_only = object_name in colors
        if is_color_only and (_is_potion_record(record) or {"potion", "vial", "bottle", "flask", "jar"} & set(tokens)):
            potion_color_only.append((sprite_id, {"object_name": object_name}))
        if category == "material" and is_color_only:
            gem_color_only.append((sprite_id, {"object_name": object_name, "category": category}))
        if tokens and tokens[0] in color_words and len(tokens) > 1:
            color_dominates.append(
                (
                    sprite_id,
                    {
                        "object_name": object_name,
                        "suggestion": "store color as an attribute when canonical representation exists",
                    },
                )
            )

        dominant_colors = _dominant_colors(record, label_v2, safe, vlm)
        base = tokens[-1] if tokens else ""
        if base in _COLOR_EXPECTED_BASE_OBJECTS:
            if dominant_colors is None:
                unavailable["color_expected_but_missing"] += 1
            elif not set(dominant_colors) - {"black", "white", "gray", "grey", "silver", "transparent"}:
                color_expected_missing.append(
                    (sprite_id, {"object_name": object_name, "dominant_colors": dominant_colors})
                )

        named_colors = set(tokens) & color_words
        if named_colors:
            if dominant_colors is None:
                unavailable["vlm_color_contradiction"] += 1
            elif not _color_tokens_compatible(named_colors, dominant_colors):
                color_contradictions.append(
                    (
                        sprite_id,
                        {
                            "label_color_tokens": sorted(named_colors),
                            "dominant_color_tokens": dominant_colors,
                            "label_color_families": sorted(_color_families(named_colors)),
                            "dominant_color_families": sorted(_color_families(dominant_colors)),
                        },
                    )
                )

        visual_facts = record.get("visual_facts") if isinstance(record.get("visual_facts"), Mapping) else None
        compact_audit_codes = {str(value) for value in quality.get("audit_codes") or ()}
        if "role_inference_contradiction" in compact_audit_codes:
            role_contradictions.append(
                (
                    sprite_id,
                    {"evidence_source": "upstream_compact_audit_code", "audit_code": "role_inference_contradiction"},
                )
            )
        if "shape_hint_contradiction" in compact_audit_codes:
            shape_contradictions.append(
                (
                    sprite_id,
                    {"evidence_source": "upstream_compact_audit_code", "audit_code": "shape_hint_contradiction"},
                )
            )
        if visual_facts is None:
            if "role_inference_contradiction" not in compact_audit_codes:
                unavailable["role_inference_contradiction"] += 1
            if "shape_hint_contradiction" not in compact_audit_codes:
                unavailable["shape_hint_contradiction"] += 1
        else:
            shape_hints = {str(value) for value in visual_facts.get("shape_hints") or ()}
            if "solid" in set(record.get("tags") or ()) and "small_content" in shape_hints:
                role_contradictions.append(
                    (sprite_id, {"tags": list(record.get("tags") or ()), "shape_hints": sorted(shape_hints)})
                )
            if "round" in set(tokens) and ({"tall", "wide"} & shape_hints):
                shape_contradictions.append(
                    (sprite_id, {"object_name": object_name, "shape_hints": sorted(shape_hints)})
                )

    _append_audit(entries, "filename_vlm_category_mismatch", filename_vlm_mismatch, severity_by_tier=True)
    _append_audit(entries, "vlm_hallucination_denylist_hit", hallucination_hits, severity="error")
    _append_audit(entries, "vlm_hallucination_denylist_suppressed", hallucination_suppressed, severity="info")
    _append_audit(entries, "vlm_color_contradiction", color_contradictions, severity="warning")
    _append_audit(entries, "role_inference_contradiction", role_contradictions, severity="warning")
    _append_audit(entries, "shape_hint_contradiction", shape_contradictions, severity="warning")
    _append_audit(entries, "potion_is_color_only", potion_color_only, severity="error")
    _append_audit(entries, "gem_is_color_only", gem_color_only, severity="error")
    _append_audit(entries, "color_expected_but_missing", color_expected_missing, severity="warning")
    _append_audit(entries, "color_dominates_object_name", color_dominates, severity="info")

    total = len(records)
    for category, category_rows in sorted(category_records.items()):
        count = len(category_rows)
        if total and count / total < 0.02:
            _append_audit(
                entries,
                "category_underrepresented",
                [(category, {"category": category, "count": count, "total": total})],
                severity="warning",
            )
        donors = Counter(str(row.get("source_name", "") or "(unknown)") for row in category_rows)
        if donors:
            donor, donor_count = donors.most_common(1)[0]
            if donor_count / count > 0.60:
                _append_audit(
                    entries,
                    "single_donor_dominates_category",
                    [
                        (
                            category,
                            {"category": category, "source_name": donor, "count": donor_count, "category_count": count},
                        )
                    ],
                    severity="warning",
                )
        weak = [
            row
            for row in category_rows
            if _record_label_tier(row, row.get("label_v2") if isinstance(row.get("label_v2"), Mapping) else {})
            in {"T3", "T4"}
        ]
        if count and len(weak) / count > 0.40:
            _append_audit(
                entries,
                "vlm_only_dominates_category",
                [
                    (
                        str(row.get("sprite_id", "")),
                        {
                            "category": category,
                            "tier": _record_label_tier(
                                row, row.get("label_v2") if isinstance(row.get("label_v2"), Mapping) else {}
                            ),
                        },
                    )
                    for row in weak
                ],
                severity="warning",
            )

    result.label_audits = {
        "report_only": True,
        "entries": entries,
        "evidence_unavailable": dict(sorted(unavailable.items())),
        "evidence_coverage": _audit_evidence_coverage(len(records), unavailable),
    }


def _record_label_tier(record: Mapping[str, Any], label_v2: Mapping[str, Any]) -> str:
    tier = str(record.get("label_confidence_tier") or label_v2.get("label_confidence_tier", "")).strip()
    if tier:
        return tier
    quality = label_v2.get("label_quality") if isinstance(label_v2.get("label_quality"), Mapping) else {}
    tier = str(quality.get("label_confidence_tier", "")).strip()
    if tier:
        return tier
    bucket = str(label_v2.get("bucket") or quality.get("bucket") or "").strip()
    return confidence_tier_for_bucket(bucket) if bucket else ""


def _dominant_colors(
    record: Mapping[str, Any], label_v2: Mapping[str, Any], safe: Mapping[str, Any], vlm: Mapping[str, Any] | None
) -> list[str] | None:
    for source in (record, safe, vlm or {}):
        value = source.get("dominant_colors") if isinstance(source, Mapping) else None
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [str(color) for color in value]
    return None


def _trusted_same_object_family(evidence: Mapping[str, Any], fusion_flags: set[str]) -> bool:
    """Suppress a denylist guess only when independent trusted evidence agrees."""

    if str(evidence.get("tier", "")) not in {"T0", "T1"}:
        return False
    if "needs_review_candidate_conflict" in fusion_flags:
        return False
    return bool(
        _object_families(str(evidence.get("vlm_object_name", "")))
        & _object_families(str(evidence.get("safe_object_name", "")))
    )


def _object_families(value: str) -> set[str]:
    tokens = set(str(value).lower().split("_"))
    families: set[str] = set()
    if {"potion", "vial", "bottle", "flask", "jar"} & tokens:
        families.add("potion_container")
    if {"coin", "bar", "ingot"} & tokens:
        families.add("currency_material")
    return families


_COLOR_FAMILY_BY_TOKEN: dict[str, frozenset[str]] = {
    "blue": frozenset({"blue"}),
    "light_blue": frozenset({"blue"}),
    "dark_blue": frozenset({"blue"}),
    "navy": frozenset({"blue"}),
    "cyan": frozenset({"blue"}),
    "teal": frozenset({"blue", "green"}),
    "pink": frozenset({"pink"}),
    "magenta": frozenset({"pink", "purple"}),
    "purple": frozenset({"purple"}),
    "violet": frozenset({"purple"}),
    "red": frozenset({"red"}),
    "dark_red": frozenset({"red"}),
    "crimson": frozenset({"red"}),
    "orange": frozenset({"orange"}),
    "yellow": frozenset({"yellow"}),
    "gold": frozenset({"yellow"}),
    "green": frozenset({"green"}),
    "dark_green": frozenset({"green"}),
    "lime": frozenset({"green"}),
    "black": frozenset({"grayscale"}),
    "white": frozenset({"grayscale"}),
    "gray": frozenset({"grayscale"}),
    "grey": frozenset({"grayscale"}),
    "dark_gray": frozenset({"grayscale"}),
    "light_gray": frozenset({"grayscale"}),
    "silver": frozenset({"grayscale"}),
}


def _color_families(tokens: Sequence[str] | set[str]) -> set[str]:
    families: set[str] = set()
    for token in tokens:
        families.update(_COLOR_FAMILY_BY_TOKEN.get(str(token).lower(), ()))
    return families


def _color_tokens_compatible(label_tokens: set[str], observed_tokens: Sequence[str]) -> bool:
    label_families = _color_families(label_tokens)
    observed_families = _color_families(observed_tokens)
    return bool(label_families and observed_families and label_families & observed_families)


def _audit_evidence_coverage(total: int, unavailable: Mapping[str, int]) -> dict[str, dict[str, float | int]]:
    return {
        code: {
            "total_records": total,
            "unavailable_records": int(count),
            "available_records": max(0, total - int(count)),
            "coverage": (max(0, total - int(count)) / total) if total else 0.0,
        }
        for code, count in sorted(unavailable.items())
    }


def _append_audit(
    entries: list[dict[str, Any]],
    code: str,
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    severity: str = "warning",
    severity_by_tier: bool = False,
) -> None:
    if not rows:
        return
    examples = [{"sprite_id": sprite_id, "evidence": dict(evidence)} for sprite_id, evidence in rows]
    if severity_by_tier:
        errors = [example for example in examples if example["evidence"].get("tier") in {"T0", "T1"}]
        warnings = [example for example in examples if example not in errors]
        if errors:
            entries.append(
                {
                    "code": code,
                    "severity": "error",
                    "count": len(errors),
                    "sprite_ids": [e["sprite_id"] for e in errors],
                    "examples": errors,
                }
            )
        if warnings:
            entries.append(
                {
                    "code": code,
                    "severity": "warning",
                    "count": len(warnings),
                    "sprite_ids": [e["sprite_id"] for e in warnings],
                    "examples": warnings,
                }
            )
        return
    entries.append(
        {
            "code": code,
            "severity": severity,
            "count": len(examples),
            "sprite_ids": [e["sprite_id"] for e in examples],
            "examples": examples,
        }
    )


# ---------------------------------------------------------------------------
# Semantic-v3 checks
# ---------------------------------------------------------------------------


def _check_semantic_v3(
    result: DatasetQAResult,
    records: Sequence[Mapping[str, Any]],
    *,
    require_semantic_v3: bool,
) -> None:
    records_with = [r for r in records if isinstance(r.get("semantic_v3"), Mapping) and r.get("semantic_v3")]
    is_semantic_dataset = bool(records_with)

    missing_semantic: list[str] = []
    missing_schema_version: list[str] = []
    missing_base_object: list[str] = []
    missing_open_name: list[str] = []
    missing_attributes: list[str] = []
    missing_captions: list[str] = []
    bad_captions: list[str] = []
    forbidden_captions: list[str] = []
    category_mismatch: list[str] = []
    object_name_mismatch: list[str] = []
    few_captions: list[str] = []
    missing_negative_tags: list[str] = []
    no_attributes: list[str] = []
    base_equals_compound_name: list[str] = []
    missing_expected_colors: list[str] = []

    for record in records:
        sprite_id = str(record.get("sprite_id", ""))
        semantic = record.get("semantic_v3")
        if not isinstance(semantic, Mapping) or not semantic:
            missing_semantic.append(sprite_id)
            continue

        if not str(semantic.get("schema_version", "")).strip():
            missing_schema_version.append(sprite_id)
        base_object = str(semantic.get("base_object", "")).strip()
        if not base_object:
            missing_base_object.append(sprite_id)
        if not str(semantic.get("open_name", "")).strip():
            missing_open_name.append(sprite_id)

        attributes = semantic.get("attributes")
        if not isinstance(attributes, Mapping):
            missing_attributes.append(sprite_id)
            attributes = {}

        captions = semantic.get("captions")
        caption_texts: list[str] = []
        if not isinstance(captions, list) or not captions:
            missing_captions.append(sprite_id)
        else:
            for caption in captions:
                if not isinstance(caption, str) or not caption.strip():
                    bad_captions.append(f"{sprite_id}: non-string or empty caption")
                    continue
                if len(caption) > MAX_SEMANTIC_CAPTION_LENGTH:
                    bad_captions.append(f"{sprite_id}: caption longer than {MAX_SEMANTIC_CAPTION_LENGTH} chars")
                    continue
                caption_texts.append(caption)
            for caption in caption_texts:
                lowered = caption.lower()
                for forbidden in FORBIDDEN_CAPTION_CONTENT:
                    if forbidden in lowered:
                        forbidden_captions.append(f"{sprite_id}: '{forbidden}' in caption")
            if len(caption_texts) < 3:
                few_captions.append(sprite_id)

        semantic_category = str(semantic.get("category", "")).strip()
        manifest_category = str(record.get("category", "")).strip()
        if semantic_category and manifest_category and semantic_category != manifest_category:
            category_mismatch.append(f"{sprite_id}: semantic={semantic_category} manifest={manifest_category}")
        semantic_object = str(semantic.get("object_name", "")).strip()
        manifest_object = str(record.get("object_name", "")).strip()
        if semantic_object and manifest_object and semantic_object != manifest_object:
            object_name_mismatch.append(f"{sprite_id}: semantic={semantic_object} manifest={manifest_object}")

        negative_tags = semantic.get("negative_tags")
        if not isinstance(negative_tags, list) or not negative_tags:
            missing_negative_tags.append(sprite_id)

        attribute_lists = [value for value in attributes.values() if isinstance(value, list) and value]
        if not attribute_lists:
            no_attributes.append(sprite_id)
        if base_object and base_object == manifest_object and "_" in manifest_object:
            base_equals_compound_name.append(sprite_id)
        colors = attributes.get("colors") if isinstance(attributes, Mapping) else None
        if base_object in _COLOR_EXPECTED_BASE_OBJECTS and not (isinstance(colors, list) and colors):
            missing_expected_colors.append(sprite_id)

    result.semantic_v3_checks = {
        "is_semantic_v3_dataset": is_semantic_dataset,
        "require_semantic_v3": bool(require_semantic_v3),
        "records_with_semantic_v3": len(records_with),
        "missing_semantic_v3": missing_semantic,
        "missing_schema_version": missing_schema_version,
        "missing_base_object": missing_base_object,
        "missing_open_name": missing_open_name,
        "missing_attributes": missing_attributes,
        "missing_captions": missing_captions,
        "bad_captions": bad_captions,
        "forbidden_caption_content": forbidden_captions,
        "category_mismatch": category_mismatch,
        "object_name_mismatch": object_name_mismatch,
        "few_captions": few_captions,
        "missing_negative_tags": missing_negative_tags,
        "no_attributes_extracted": no_attributes,
        "base_object_equals_compound_name": base_equals_compound_name,
        "missing_expected_colors": missing_expected_colors,
    }

    if require_semantic_v3:
        for sprite_id in missing_semantic:
            result.add_error(f"record missing required semantic_v3 metadata: {sprite_id}")
    elif is_semantic_dataset and missing_semantic:
        result.add_warning(f"{len(missing_semantic)}/{len(records)} records lack semantic_v3 in a semantic dataset")

    for sprite_id in missing_schema_version:
        result.add_error(f"semantic_v3.schema_version missing: {sprite_id}")
    for sprite_id in missing_base_object:
        result.add_error(f"semantic_v3.base_object missing: {sprite_id}")
    for sprite_id in missing_open_name:
        result.add_error(f"semantic_v3.open_name missing: {sprite_id}")
    for sprite_id in missing_attributes:
        result.add_error(f"semantic_v3.attributes missing: {sprite_id}")
    for sprite_id in missing_captions:
        result.add_error(f"semantic_v3.captions missing or empty: {sprite_id}")
    for entry in bad_captions:
        result.add_error(f"semantic_v3 caption invalid: {entry}")
    for entry in forbidden_captions:
        result.add_error(f"semantic_v3 caption contains forbidden content: {entry}")
    for entry in category_mismatch:
        result.add_error(f"semantic_v3.category does not match manifest: {entry}")
    for entry in object_name_mismatch:
        result.add_error(f"semantic_v3.object_name does not match manifest: {entry}")

    if few_captions:
        result.add_warning(f"{len(few_captions)} semantic_v3 records have fewer than 3 captions")
    if missing_negative_tags:
        result.add_warning(f"{len(missing_negative_tags)} semantic_v3 records have no negative_tags")
    if no_attributes:
        result.add_warning(f"{len(no_attributes)} semantic_v3 records have no extracted attributes")
    if records_with and len(base_equals_compound_name) > max(2, len(records_with) // 4):
        result.add_warning(
            f"{len(base_equals_compound_name)} semantic_v3 records keep the full compound name as base_object"
        )
    if missing_expected_colors:
        result.add_warning(
            f"{len(missing_expected_colors)} semantic_v3 gem/container records have no color information"
        )


# ---------------------------------------------------------------------------
# Split checks
# ---------------------------------------------------------------------------


def _check_splits(
    result: DatasetQAResult,
    manifests: Mapping[str, list[dict[str, Any]]],
    npz_by_split: Mapping[str, dict[str, Any] | None],
    *,
    expected_fractions: tuple[float, float, float],
) -> None:
    id_to_splits: dict[str, list[str]] = {}
    mislabeled_split: list[str] = []
    count_mismatch: list[str] = []

    for split in SPLIT_NAMES:
        for record in manifests.get(split, []):
            sprite_id = str(record.get("sprite_id", ""))
            id_to_splits.setdefault(sprite_id, []).append(split)
            record_split = str(record.get("split", "")).strip()
            if record_split and record_split != split:
                mislabeled_split.append(f"{sprite_id}: in {split} manifest but split={record_split}")

        payload = npz_by_split.get(split)
        manifest_count = len(manifests.get(split, []))
        npz_count = payload["count"] if payload is not None else 0
        if payload is not None and npz_count != manifest_count:
            count_mismatch.append(f"{split}: manifest={manifest_count} npz={npz_count}")

    overlap = sorted(sprite_id for sprite_id, splits in id_to_splits.items() if len(splits) > 1)

    total = sum(len(manifests.get(split, [])) for split in SPLIT_NAMES)
    fractions = {split: (len(manifests.get(split, [])) / total if total else 0.0) for split in SPLIT_NAMES}
    expected = dict(zip(SPLIT_NAMES, expected_fractions, strict=False))
    ratio_warnings: list[str] = []
    tolerance = 0.08
    if total >= 20:
        for split in SPLIT_NAMES:
            if abs(fractions[split] - expected[split]) > tolerance:
                ratio_warnings.append(
                    f"{split} fraction {fractions[split]:.3f} deviates from expected {expected[split]:.3f}"
                )

    empty_splits = [split for split in SPLIT_NAMES if len(manifests.get(split, [])) == 0]

    result.split_checks = {
        "counts": {split: len(manifests.get(split, [])) for split in SPLIT_NAMES},
        "fractions": {split: round(fractions[split], 4) for split in SPLIT_NAMES},
        "expected_fractions": expected,
        "overlap": overlap,
        "mislabeled_split": mislabeled_split,
        "npz_count_mismatch": count_mismatch,
        "empty_splits": empty_splits,
        "total": total,
    }

    for sprite_id in overlap:
        result.add_error(f"sprite appears in multiple splits: {sprite_id} ({id_to_splits[sprite_id]})")
    for entry in mislabeled_split:
        result.add_error(f"split field mismatch: {entry}")
    for entry in count_mismatch:
        result.add_error(f"npz row count does not match manifest count: {entry}")
    for split in empty_splits:
        result.add_warning(f"split has zero records: {split}")
    for message in ratio_warnings:
        result.add_warning(f"split ratio: {message}")


# ---------------------------------------------------------------------------
# Distribution / counts
# ---------------------------------------------------------------------------


def _build_counts(
    result: DatasetQAResult,
    records: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, Any],
) -> None:
    categories: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    splits: Counter[str] = Counter()

    for record in records:
        categories[str(record.get("category", "")) or "unknown"] += 1
        objects[str(record.get("object_name", "")) or "(empty)"] += 1
        sources[str(record.get("source_name", "")) or "(unknown)"] += 1
        splits[str(record.get("split", "")) or "(none)"] += 1
        label_v2 = record.get("label_v2")
        if isinstance(label_v2, Mapping):
            bucket = str(label_v2.get("bucket", "")).strip()
            if bucket:
                buckets[bucket] += 1

    result.counts = {
        "categories": dict(sorted(categories.items())),
        "objects": dict(sorted(objects.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sources": dict(sorted(sources.items())),
        "label_v2_buckets": dict(sorted(buckets.items())),
        "splits": dict(sorted(splits.items())),
    }


def _check_distribution(
    result: DatasetQAResult,
    records: Sequence[Mapping[str, Any]],
    vocab: Mapping[str, Any],
    *,
    max_object_share: float,
) -> None:
    total = len(records)
    if total == 0:
        result.add_error("dataset has zero records")
        return

    categories = result.counts.get("categories", {})
    category_to_id = vocab.get("category_to_id") if isinstance(vocab, Mapping) else None
    if isinstance(category_to_id, Mapping):
        for category in category_to_id:
            if category == "unknown":
                continue  # structural placeholder id 0
            if categories.get(category, 0) == 0:
                result.add_warning(f"vocab category has zero records: {category}")

    objects = result.counts.get("objects", {})
    if objects:
        top_object, top_count = next(iter(objects.items()))
        share = top_count / total
        if share > max_object_share:
            result.add_warning(
                f"object_name '{top_object}' dominates: {top_count}/{total} = {share:.1%} (> {max_object_share:.0%})"
            )

    single_tag = len(result.manifest_checks.get("single_tag_records", []))
    if single_tag and single_tag / total > 0.10:
        result.add_warning(f"{single_tag}/{total} records have only one tag ({single_tag / total:.1%})")


# ---------------------------------------------------------------------------
# Review queue overlap
# ---------------------------------------------------------------------------


def _check_review_queue(
    result: DatasetQAResult,
    records: Sequence[Mapping[str, Any]],
    review_queue: Path | None,
) -> None:
    if review_queue is None:
        result.review_queue_overlap = []
        return
    review_queue = Path(review_queue)
    if not review_queue.is_file():
        result.add_warning(f"review queue not found: {review_queue}")
        result.review_queue_overlap = []
        return

    queue_ids: set[str] = set()
    for line in review_queue.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, Mapping):
            sprite_id = str(record.get("sprite_id", "")).strip()
            if sprite_id:
                queue_ids.add(sprite_id)

    exported_ids = {str(record.get("sprite_id", "")).strip() for record in records}
    overlap = sorted(exported_ids & queue_ids)
    result.review_queue_overlap = overlap
    for sprite_id in overlap:
        result.add_error(f"exported sprite is in review queue: {sprite_id}")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(result: DatasetQAResult) -> str:
    lines: list[str] = []
    ok = "PASS" if result.ok else "FAIL"
    lines.append("# Dataset QA Report")
    lines.append("")
    lines.append(f"Dataset: `{result.dataset_dir}`")
    lines.append(f"Status: **{ok}**")
    lines.append(f"Records: {result.total_records}")
    lines.append(f"Images: {result.total_images}")
    lines.append(f"Errors: {len(result.errors)}")
    lines.append(f"Warnings: {len(result.warnings)}")
    lines.append("")

    lines.append("## Splits")
    lines.append("")
    for split in SPLIT_NAMES:
        frac = result.split_checks.get("fractions", {}).get(split)
        frac_text = f" ({frac:.1%})" if isinstance(frac, (int, float)) else ""
        lines.append(f"- {split}: {result.splits.get(split, 0)}{frac_text}")
    lines.append(f"- total: {result.split_checks.get('total', result.total_records)}")
    lines.append("")

    lines.append("## Image Checks")
    lines.append("")
    ic = result.image_checks
    lines.append(f"- all 32x32: {ic.get('all_32x32')}")
    lines.append(
        f"- max palette slots seen: {ic.get('max_palette_slots_seen')} (allowed {ic.get('max_palette_slots_allowed')})"
    )
    lines.append(f"- missing images: {len(ic.get('missing_images', []))}")
    lines.append(f"- unreferenced images: {len(ic.get('unreferenced_images', []))}")
    lines.append(f"- bad dimensions: {len(ic.get('bad_dimensions', []))}")
    lines.append(f"- unreadable payloads: {len(ic.get('unreadable_images', []))}")
    lines.append(f"- empty (fully transparent) images: {len(ic.get('empty_images', []))}")
    lines.append(f"- soft-alpha images: {len(ic.get('bad_alpha', []))}")
    lines.append("")

    lines.append("## Manifest Checks")
    lines.append("")
    mc = result.manifest_checks
    lines.append(f"- duplicate sprite_ids: {len(mc.get('duplicate_sprite_ids', []))}")
    lines.append(f"- missing object_name: {len(mc.get('missing_object_name', []))}")
    lines.append(f"- empty tags: {len(mc.get('missing_tags', []))}")
    lines.append(f"- single-tag records: {len(mc.get('single_tag_records', []))}")
    lines.append(f"- review/quarantine leaks: {len(mc.get('review_status_leaks', []))}")
    lines.append(f"- forbidden object names: {len(mc.get('forbidden_object_names', []))}")
    lines.append(f"- colour-only potion names: {len(mc.get('color_only_potion_object_names', []))}")
    lines.append("")

    lines.append("## Label-v2 Checks")
    lines.append("")

    lines.append("## Labeling v2 Audits (report-only)")
    lines.append("")
    audits = result.label_audits
    lines.append(f"- report-only: {audits.get('report_only', True)}")
    for entry in audits.get("entries", []):
        lines.append(f"- {entry.get('severity', 'warning')} {entry.get('code', '')}: {entry.get('count', 0)}")
    unavailable = dict(audits.get("evidence_unavailable", {}) or {})
    if unavailable:
        lines.append("- unavailable evidence:")
        for code, count in unavailable.items():
            lines.append(f"  - {code}: {count}")
    lines.append("")
    lc = result.label_v2_checks
    lines.append(f"- label-v2 dataset: {lc.get('is_label_v2_dataset')}")
    lines.append(f"- records with label_v2: {lc.get('records_with_label_v2', 0)}")
    lines.append(f"- missing label_v2: {len(lc.get('missing_label_v2', []))}")
    lines.append(f"- applied != true: {len(lc.get('applied_not_true', []))}")
    lines.append(f"- missing bucket: {len(lc.get('missing_bucket', []))}")
    lines.append(f"- missing flags: {len(lc.get('missing_flags', []))}")
    lines.append("")

    lines.append("## Semantic-v3 Checks")
    lines.append("")
    sc = result.semantic_v3_checks
    lines.append(f"- semantic-v3 dataset: {sc.get('is_semantic_v3_dataset')}")
    lines.append(f"- required: {sc.get('require_semantic_v3')}")
    lines.append(f"- records with semantic_v3: {sc.get('records_with_semantic_v3', 0)}")
    lines.append(f"- missing semantic_v3: {len(sc.get('missing_semantic_v3', []))}")
    lines.append(f"- missing base_object: {len(sc.get('missing_base_object', []))}")
    lines.append(f"- missing captions: {len(sc.get('missing_captions', []))}")
    lines.append(f"- invalid captions: {len(sc.get('bad_captions', []))}")
    lines.append(f"- forbidden caption content: {len(sc.get('forbidden_caption_content', []))}")
    lines.append(f"- category mismatches: {len(sc.get('category_mismatch', []))}")
    lines.append(f"- object_name mismatches: {len(sc.get('object_name_mismatch', []))}")
    lines.append("")

    lines.append("## Distribution")
    lines.append("")
    lines.append("### Categories")
    for name, count in result.counts.get("categories", {}).items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("### label_v2 buckets")
    for name, count in result.counts.get("label_v2_buckets", {}).items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("### Top object names")
    for name, count in list(result.counts.get("objects", {}).items())[:20]:
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("### Sources")
    for name, count in result.counts.get("sources", {}).items():
        lines.append(f"- {name}: {count}")
    lines.append("")

    lines.append("## Review Queue Overlap")
    lines.append("")
    lines.append(f"Overlap count: {len(result.review_queue_overlap)}")
    for sprite_id in result.review_queue_overlap:
        lines.append(f"- {sprite_id}")
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
# Report + contact-sheet writers
# ---------------------------------------------------------------------------


def write_reports(result: DatasetQAResult, *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(result.to_markdown(), encoding="utf-8")


def build_contact_sheet(
    dataset_dir: Path,
    out_path: Path,
    *,
    sample_limit: int = 64,
    scale: int = 6,
    columns: int = 8,
    label: bool = True,
) -> Path | None:
    """Render a deterministic grid of exported sprites for eyeball QA.

    Returns the written path, or ``None`` if there is nothing to render or
    Pillow is unavailable.
    """

    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - Pillow is a project dependency
        return None

    dataset_dir = Path(dataset_dir)
    samples = _contact_sheet_samples(dataset_dir, sample_limit)
    if not samples:
        return None

    cell = 32 * scale
    label_h = 10 if label else 0
    pad = 2
    cell_w = cell + pad
    cell_h = cell + label_h + pad
    columns = max(1, columns)
    rows = (len(samples) + columns - 1) // columns

    sheet = Image.new("RGBA", (columns * cell_w + pad, rows * cell_h + pad), (32, 32, 32, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        from PIL import ImageFont

        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - defensive
        font = None

    for index, (sprite_id, rgba, object_name, split) in enumerate(samples):
        col = index % columns
        row = index // columns
        x = pad + col * cell_w
        y = pad + row * cell_h
        big = rgba.resize((cell, cell), Image.NEAREST)
        sheet.alpha_composite(big, (x, y))
        if label:
            suffix = sprite_id.rsplit("_", 1)[-1][:8]
            text = f"{object_name[:10]} {split[:1]} {suffix}"
            draw.text((x + 1, y + cell + 1), text, fill=(220, 220, 220, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.convert("RGB").save(out_path)
    return out_path


def _contact_sheet_samples(dataset_dir: Path, sample_limit: int) -> list[tuple[str, Any, str, str]]:
    from PIL import Image

    object_by_id: dict[str, str] = {}
    for split in SPLIT_NAMES:
        path = dataset_dir / f"manifest_{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                object_by_id[str(record.get("sprite_id", ""))] = str(record.get("object_name", ""))

    collected: list[tuple[str, Any, str, str]] = []
    for split in SPLIT_NAMES:
        path = dataset_dir / f"{split}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            if not {"alpha", "index_map", "palette", "sprite_id"} <= set(data.files):
                continue
            alpha = np.asarray(data["alpha"])
            index_map = np.asarray(data["index_map"])
            palette = np.asarray(data["palette"])
            sprite_ids = [str(value) for value in np.asarray(data["sprite_id"])]
        for row, sprite_id in enumerate(sprite_ids):
            rgb = palette[row][np.clip(index_map[row], 0, palette[row].shape[0] - 1)]
            a = alpha[row].astype(np.uint8) * 255
            rgba_array = np.dstack([rgb.astype(np.uint8), a]).astype(np.uint8)
            image = Image.fromarray(rgba_array, mode="RGBA")
            collected.append((sprite_id, image, object_by_id.get(sprite_id, ""), split))

    collected.sort(key=lambda item: item[0])
    if sample_limit > 0 and len(collected) > sample_limit:
        # Deterministic evenly-spaced sample across the sorted ids.
        step = len(collected) / sample_limit
        collected = [collected[int(i * step)] for i in range(sample_limit)]
    return collected
