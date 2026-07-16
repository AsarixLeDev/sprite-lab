"""Leakage-safe, read-only split planning for exported Sprite Lab datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.training.splits import GroupSplitRecord, make_balanced_group_aware_split

SPLITS = ("train", "val", "test")
_EXPLICIT_VARIANT_FIELDS = (
    "explicit_variant_group",
    "variant_group",
    "variant_group_id",
    "duplicate_group",
    "base_family",
    "base_sprite_id",
    "source_variant_group",
)


@dataclass(frozen=True)
class GroupAwareSplitConfig:
    seed: int = 1337
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    exact_rgba: bool = True
    alpha_mask: bool = True
    explicit_variant_family: bool = True
    source_ood_holdout_packs: tuple[str, ...] = ()
    source_ood_holdout_authors: tuple[str, ...] = ()
    source_ood_min_pack_size: int = 12
    source_ood_allow_category_ood: bool = False
    overwrite_output: bool = False


@dataclass(frozen=True)
class SplitPlanRecord:
    manifest: dict[str, Any]
    current_split: str
    alpha_hash: str
    rgba_hash: str
    explicit_variant_values: tuple[str, ...]
    source_sheet: str

    @property
    def sprite_id(self) -> str:
        return str(self.manifest.get("sprite_id", ""))


def load_exported_split_records(dataset_dir: str | Path) -> list[SplitPlanRecord]:
    """Load manifests/NPZ rasters without modifying the exported dataset."""

    root = Path(dataset_dir)
    records: list[SplitPlanRecord] = []
    for split in SPLITS:
        manifest_path = root / f"manifest_{split}.jsonl"
        npz_path = root / f"{split}.npz"
        if not manifest_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(f"dataset requires {manifest_path.name} and {npz_path.name}")
        manifest = _read_jsonl(manifest_path)
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        ids = [str(value) for value in arrays.get("sprite_id", ())]
        by_id = {sprite_id: index for index, sprite_id in enumerate(ids)}
        for row in manifest:
            sprite_id = str(row.get("sprite_id", ""))
            if sprite_id not in by_id:
                raise ValueError(f"{split}: manifest sprite {sprite_id!r} has no NPZ row")
            index = by_id[sprite_id]
            alpha = np.asarray(arrays["alpha"][index], dtype=np.uint8)
            index_map = np.asarray(arrays["index_map"][index], dtype=np.int64)
            palette = np.asarray(arrays["palette"][index], dtype=np.uint8)
            records.append(
                SplitPlanRecord(
                    manifest=dict(row),
                    current_split=split,
                    alpha_hash=_hash_array(alpha),
                    rgba_hash=_hash_rgba(alpha, index_map, palette),
                    explicit_variant_values=_explicit_variant_values(row),
                    source_sheet=_source_sheet_key(row),
                )
            )
    return sorted(records, key=lambda record: record.sprite_id)


def build_variant_groups(
    records: Sequence[SplitPlanRecord], config: GroupAwareSplitConfig
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Union hard leakage signals into canonical stable variant-group IDs."""

    parent = {record.sprite_id: record.sprite_id for record in records}
    evidence: dict[str, set[str]] = defaultdict(set)

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(values: Iterable[str], reason: str) -> None:
        ids = sorted(set(values))
        if len(ids) < 2:
            return
        root = find(ids[0])
        for sprite_id in ids[1:]:
            other = find(sprite_id)
            if root != other:
                parent[other] = root
        for sprite_id in ids:
            evidence[sprite_id].add(reason)

    if config.exact_rgba:
        for values in _groups_by(records, lambda record: record.rgba_hash).values():
            union((record.sprite_id for record in values), "exact_rgba")
    if config.alpha_mask:
        for values in _groups_by(records, lambda record: record.alpha_hash).values():
            union((record.sprite_id for record in values), "alpha_mask")
    if config.explicit_variant_family:
        explicit: dict[str, list[str]] = defaultdict(list)
        for record in records:
            for value in record.explicit_variant_values:
                explicit[value].append(record.sprite_id)
        for values in explicit.values():
            union(values, "explicit_variant_family")

    components: dict[str, list[str]] = defaultdict(list)
    for record in records:
        components[find(record.sprite_id)].append(record.sprite_id)
    group_by_id: dict[str, str] = {}
    reasons_by_id: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        canonical = "|".join(sorted(members))
        group_id = f"variant_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
        reasons = tuple(sorted({reason for sprite_id in members for reason in evidence[sprite_id]}))
        for sprite_id in members:
            group_by_id[sprite_id] = group_id
            reasons_by_id[sprite_id] = reasons
    return group_by_id, reasons_by_id


def plan_group_aware_split(records: Sequence[SplitPlanRecord], config: GroupAwareSplitConfig) -> dict[str, Any]:
    """Return a main compositional plan and separate Source-OOD plan."""

    group_by_id, reasons_by_id = build_variant_groups(records, config)
    assignments = make_balanced_group_aware_split(
        [
            GroupSplitRecord(
                sprite_id=record.sprite_id,
                group_id=group_by_id[record.sprite_id],
                category=str(record.manifest.get("category", "unknown")),
                object_name=str(record.manifest.get("object_name", "")),
                source_pack=_source_pack(record.manifest),
                split_override=_manual_split_override(record.manifest),
            )
            for record in records
        ],
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    source_ood = _plan_source_ood(records, config)
    current = {record.sprite_id: record.current_split for record in records}
    proposed = assignments.split_by_sprite_id
    before = leakage_audit(records, current, group_by_id)
    after = leakage_audit(records, proposed, group_by_id)
    return {
        "assignments": proposed,
        "group_by_sprite_id": group_by_id,
        "group_reasons_by_sprite_id": reasons_by_id,
        "manual_group_overrides": assignments.group_overrides,
        "main": split_metrics(records, proposed),
        "before": split_metrics(records, current),
        "degradation": _degradation(split_metrics(records, current), split_metrics(records, proposed)),
        "leakage_before": before,
        "leakage_after": after,
        "source_ood": source_ood,
        "changed_split_count": sum(1 for sprite_id, split in proposed.items() if current.get(sprite_id) != split),
    }


def leakage_audit(
    records: Sequence[SplitPlanRecord], assignments: Mapping[str, str], group_by_id: Mapping[str, str]
) -> dict[str, Any]:
    """Return hard leakage gates plus informational same-sheet crossings."""

    def crossings(groups: Mapping[str, Sequence[SplitPlanRecord]]) -> list[dict[str, Any]]:
        return [
            {
                "key": key,
                "sprite_ids": [record.sprite_id for record in values],
                "splits": sorted({assignments[record.sprite_id] for record in values}),
            }
            for key, values in sorted(groups.items())
            if len(values) > 1 and len({assignments[record.sprite_id] for record in values}) > 1
        ]

    rgba = crossings(_groups_by(records, lambda record: record.rgba_hash))
    alpha = crossings(_groups_by(records, lambda record: record.alpha_hash))
    variants = crossings(_explicit_variant_groups(records))
    sheets = crossings(_groups_by(records, lambda record: record.source_sheet))
    return {
        "cross_split_exact_rgba_groups": rgba,
        "cross_split_alpha_mask_groups": alpha,
        "cross_split_explicit_variant_groups": variants,
        "cross_split_source_sheet_groups": sheets,
        "manual_override_group_conflicts": [],
        "gates_pass": not rgba and not alpha and not variants,
    }


def _explicit_variant_groups(records: Sequence[SplitPlanRecord]) -> dict[str, list[SplitPlanRecord]]:
    groups: dict[str, list[SplitPlanRecord]] = defaultdict(list)
    for record in records:
        for value in record.explicit_variant_values:
            groups[value].append(record)
    return groups


def _degradation(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_objects = set(before["val_test_only_objects"])
    after_objects = set(after["val_test_only_objects"])
    missing_train_categories = sorted(
        category
        for category, counts in after["category_distribution"].items()
        if before["category_distribution"].get(category, {}).get("train", 0) > 0 and counts.get("train", 0) == 0
    )
    return {
        "new_val_test_only_objects": sorted(after_objects - before_objects),
        "resolved_val_test_only_objects": sorted(before_objects - after_objects),
        "categories_lost_from_train": missing_train_categories,
    }


def split_metrics(records: Sequence[SplitPlanRecord], assignments: Mapping[str, str]) -> dict[str, Any]:
    total = len(records)
    rows = Counter(assignments.values())
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    objects: dict[str, Counter[str]] = defaultdict(Counter)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = assignments[record.sprite_id]
        categories[str(record.manifest.get("category", "unknown"))][split] += 1
        objects[str(record.manifest.get("object_name", ""))][split] += 1
        sources[_source_pack(record.manifest)][split] += 1
    val_test_only = sorted(
        name for name, counts in objects.items() if name and not counts["train"] and (counts["val"] or counts["test"])
    )
    return {
        "row_counts": {split: int(rows[split]) for split in SPLITS},
        "fractions": {split: (rows[split] / total if total else 0.0) for split in SPLITS},
        "category_distribution": {
            name: {split: int(counts[split]) for split in SPLITS} for name, counts in sorted(categories.items())
        },
        "object_distribution": {
            name: {split: int(counts[split]) for split in SPLITS} for name, counts in sorted(objects.items())
        },
        "source_pack_distribution": {
            name: {split: int(counts[split]) for split in SPLITS} for name, counts in sorted(sources.items())
        },
        "val_test_only_objects": val_test_only,
    }


def write_group_aware_dry_run(
    dataset_dir: str | Path,
    output_dir: str | Path,
    config: GroupAwareSplitConfig | None = None,
) -> dict[str, Any]:
    """Write only proposal/report artifacts; never mutate ``dataset_dir``."""

    config = config or GroupAwareSplitConfig()
    records = load_exported_split_records(dataset_dir)
    plan = plan_group_aware_split(records, config)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not config.overwrite_output:
        raise FileExistsError(f"dry-run output directory already exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    assignments = []
    for record in records:
        sprite_id = record.sprite_id
        assignments.append(
            {
                "sprite_id": sprite_id,
                "current_split": record.current_split,
                "proposed_split": plan["assignments"][sprite_id],
                "variant_group": plan["group_by_sprite_id"][sprite_id],
                "grouping_reasons": list(plan["group_reasons_by_sprite_id"][sprite_id]),
            }
        )
    _write_jsonl(out / "proposed_assignments.jsonl", assignments)
    _write_json(out / "split_summary.json", _json_summary(dataset_dir, config, plan))
    _write_json(out / "leakage_before_after.json", {"before": plan["leakage_before"], "after": plan["leakage_after"]})
    _write_source_ood_manifests(out, records, plan["source_ood"])
    (out / "split_report.md").write_text(_render_report(dataset_dir, plan), encoding="utf-8")
    return plan


def _plan_source_ood(records: Sequence[SplitPlanRecord], config: GroupAwareSplitConfig) -> dict[str, Any]:
    by_pack: dict[str, list[SplitPlanRecord]] = defaultdict(list)
    for record in records:
        by_pack[_source_pack(record.manifest)].append(record)
    category_sources: dict[str, set[str]] = defaultdict(set)
    for pack, members in by_pack.items():
        for record in members:
            category_sources[str(record.manifest.get("category", "unknown"))].add(pack)
    explicit = set(config.source_ood_holdout_packs)
    explicit_authors = set(config.source_ood_holdout_authors)
    if explicit_authors:
        explicit.update(
            pack
            for pack, members in by_pack.items()
            if any(str(member.manifest.get("author", "")) in explicit_authors for member in members)
        )
    candidates: list[dict[str, Any]] = []
    for pack, members in sorted(by_pack.items()):
        categories = Counter(str(member.manifest.get("category", "unknown")) for member in members)
        critical = sorted(category for category in categories if len(category_sources[category]) == 1)
        eligible = len(members) >= config.source_ood_min_pack_size and (
            config.source_ood_allow_category_ood or not critical
        )
        candidates.append(
            {
                "source_pack": pack,
                "count": len(members),
                "category_counts": dict(sorted(categories.items())),
                "critical_categories": critical,
                "eligible": eligible,
            }
        )
    if explicit:
        holdout = sorted(explicit & set(by_pack))
    else:
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        holdout = (
            [
                min(
                    eligible,
                    key=lambda candidate: (-candidate["count"], _seed_key(config.seed, candidate["source_pack"])),
                )["source_pack"]
            ]
            if eligible
            else []
        )
    assignments = {
        record.sprite_id: ("eval" if _source_pack(record.manifest) in holdout else "train") for record in records
    }
    return {"holdout_packs": holdout, "candidate_holdouts": candidates, "assignments": assignments}


def _groups_by(records: Sequence[SplitPlanRecord], key_fn: Any) -> dict[str, list[SplitPlanRecord]]:
    groups: dict[str, list[SplitPlanRecord]] = defaultdict(list)
    for record in records:
        groups[str(key_fn(record))].append(record)
    return groups


def _hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _hash_rgba(alpha: np.ndarray, index_map: np.ndarray, palette: np.ndarray) -> str:
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha * 255
    opaque = alpha == 1
    rgba[opaque, :3] = palette[index_map[opaque]]
    return _hash_array(rgba)


def _explicit_variant_values(row: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for field in _EXPLICIT_VARIANT_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            values.append(f"{field}:{str(value).strip()}")
    provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
    for field in _EXPLICIT_VARIANT_FIELDS:
        value = provenance.get(field)
        if value is not None and str(value).strip():
            values.append(f"provenance.{field}:{str(value).strip()}")
    return tuple(sorted(set(values)))


def _manual_split_override(row: Mapping[str, Any]) -> str | None:
    value = row.get("split_override") or row.get("manual_split")
    return str(value) if value is not None and str(value).strip() else None


def _source_pack(row: Mapping[str, Any]) -> str:
    return str(row.get("source_pack") or row.get("source_dataset") or row.get("source_name") or "unknown")


def _source_sheet_key(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_path", "")).replace("\\", "/")
    parent = str(Path(source).parent).replace("\\", "/")
    return f"{_source_pack(row)}:{parent}"


def _seed_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")


def _write_source_ood_manifests(
    output_dir: Path, records: Sequence[SplitPlanRecord], source_ood: Mapping[str, Any]
) -> None:
    assignments = source_ood["assignments"]
    for split in ("train", "eval"):
        rows = [
            {**record.manifest, "source_ood_split": split}
            for record in records
            if assignments[record.sprite_id] == split
        ]
        _write_jsonl(output_dir / f"manifest_source_ood_{split}.jsonl", rows)
    _write_json(
        output_dir / "source_ood_split_report.json",
        {key: value for key, value in source_ood.items() if key != "assignments"},
    )


def _json_summary(dataset_dir: str | Path, config: GroupAwareSplitConfig, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir),
        "split_policy": "group_aware",
        "split_seed": config.seed,
        "split_grouping": {
            "exact_rgba": config.exact_rgba,
            "alpha_mask": config.alpha_mask,
            "explicit_variant_family": config.explicit_variant_family,
        },
        **{
            key: value
            for key, value in plan.items()
            if key not in {"assignments", "group_by_sprite_id", "group_reasons_by_sprite_id", "source_ood"}
        },
    }


def _render_report(dataset_dir: str | Path, plan: Mapping[str, Any]) -> str:
    before = plan["leakage_before"]
    after = plan["leakage_after"]
    main = plan["main"]
    lines = [
        "# Group-aware split dry run",
        "",
        f"Dataset: `{dataset_dir}`",
        "",
        "## Recommendation",
        "",
        "Rebuild only after reviewing changed assignments and Source-OOD candidate coverage. The proposed main split passes hard leakage gates."
        if after["gates_pass"]
        else "Do not rebuild: hard leakage gates still fail.",
        "",
        "## Leakage before / after",
        "",
    ]
    for key in (
        "cross_split_exact_rgba_groups",
        "cross_split_alpha_mask_groups",
        "cross_split_explicit_variant_groups",
        "cross_split_source_sheet_groups",
    ):
        lines.append(f"- {key}: {len(before[key])} -> {len(after[key])}")
    lines.extend(["", "## Main compositional split", ""])
    for split in SPLITS:
        lines.append(f"- {split}: {main['row_counts'][split]} ({main['fractions'][split]:.1%})")
    lines.append(f"- sprites changing split: {plan['changed_split_count']}")
    lines.append(f"- val/test-only objects: {len(main['val_test_only_objects'])}")
    lines.append(
        f"- new val/test-only objects: {', '.join(plan['degradation']['new_val_test_only_objects']) or '(none)'}"
    )
    lines.append(
        f"- categories lost from train: {', '.join(plan['degradation']['categories_lost_from_train']) or '(none)'}"
    )
    lines.extend(
        [
            "",
            "## Source-OOD",
            "",
            f"- holdout packs: {', '.join(plan['source_ood']['holdout_packs']) or '(none)'}",
            "- This benchmark is separate from main train/val/test.",
            "",
        ]
    )
    return "\n".join(lines)
