"""Deterministic candidate-view construction for the raw Dataset-v5 rebuild."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.dataset_v5.identity import assert_opaque_id
from spritelab.dataset_v5.raw_relations import (
    HardRelationLeakageError,
    assert_no_hard_relation_crossing,
    assign_component_splits,
)

VIEW_POLICY_VERSION = "sprite_lab_raw_candidate_views_v1"
VIEW_BUNDLE_SCHEMA_VERSION = "sprite_lab_raw_candidate_view_bundle_v1"
VIEW_MANIFEST_SCHEMA_VERSION = "sprite_lab_raw_candidate_view_manifest_v1"

VIEW_NAMES = (
    "v5_debug",
    "v5_architecture",
    "v5_scale_check",
    "v5_eval_balanced",
    "v5_source_ood",
    "v5_open_set",
    "v5_unlabeled",
)

SUPERVISION_CLASSES = frozenset({"supervised_strong", "supervised_weak", "auxiliary_only", "unlabeled"})
_NORMAL_SPLITS = frozenset({"train", "validation", "test", "unsplit"})


class RawViewError(ValueError):
    """Raised when candidate membership cannot be made safe and deterministic."""


def build_candidate_views(records: Sequence[Mapping[str, Any]], relation_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build all seven views without forcing a target record count.

    Exact RGBA duplicates are represented once in every architecture-bearing
    view.  Selection order is unique geometry, meaningful material variants,
    recolors, and then other unique blobs.  Quarantined, rejected, or
    unresolved-critical records are never promoted into a candidate view.
    """

    rows = _validated_records(records)
    prepared = _with_partition_intents(rows)
    try:
        split_by_record = assign_component_splits(prepared, relation_manifest)
    except HardRelationLeakageError as exc:
        raise RawViewError(str(exc)) from exc
    assert_no_hard_relation_crossing(relation_manifest, split_by_record, require_all=True)

    component_by_record = {
        str(record_id): str(component_id)
        for record_id, component_id in dict(relation_manifest.get("record_to_component") or {}).items()
    }
    if set(component_by_record) != {str(row["record_id"]) for row in rows}:
        raise RawViewError("relation manifest record_to_component is not an exact record partition")

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        record_id = str(row["record_id"])
        decision = _suitability_decision(row)
        if decision != "accept":
            excluded.append({"reason": f"suitability_{decision}", "record_id": record_id})
            continue
        if _has_unresolved_critical_conflict(row):
            excluded.append({"reason": "unresolved_critical_conflict", "record_id": record_id})
            continue
        eligible.append(row)

    unique_rows, duplicate_exclusions = _deduplicate_exact_rgba(eligible)
    excluded.extend(duplicate_exclusions)
    geometry_counts = Counter(_geometry_key(row) for row in unique_rows)
    relation_kinds = _relation_kinds_by_record(relation_manifest)

    entries = {
        str(row["record_id"]): _view_entry(
            row,
            split=split_by_record[str(row["record_id"])],
            component_id=component_by_record[str(row["record_id"])],
            geometry_count=geometry_counts[_geometry_key(row)],
            relation_kinds=relation_kinds.get(str(row["record_id"]), frozenset()),
        )
        for row in unique_rows
    }

    normal = [row for row in unique_rows if split_by_record[str(row["record_id"])] in _NORMAL_SPLITS]
    source_ood = [row for row in unique_rows if split_by_record[str(row["record_id"])] == "source_ood_test"]
    open_set = [row for row in unique_rows if split_by_record[str(row["record_id"])] == "open_set_test"]

    architecture = sorted((entries[str(row["record_id"])] for row in normal), key=_architecture_sort_key)
    component_representatives: dict[str, dict[str, Any]] = {}
    for entry in architecture:
        component_representatives.setdefault(str(entry["hard_relation_component_id"]), entry)
    debug = sorted(component_representatives.values(), key=_architecture_sort_key)
    scale_check = sorted(architecture, key=lambda entry: (int(entry["pixel_area"]), str(entry["record_id"])))
    eval_balanced = _balanced_eval_entries(normal, entries)
    unlabeled = sorted(
        (entries[str(row["record_id"])] for row in normal if _supervision_class(row) == "unlabeled"),
        key=lambda entry: str(entry["record_id"]),
    )

    views = {
        "v5_architecture": architecture,
        "v5_debug": debug,
        "v5_eval_balanced": eval_balanced,
        "v5_open_set": sorted(
            (entries[str(row["record_id"])] for row in open_set), key=lambda entry: str(entry["record_id"])
        ),
        "v5_scale_check": scale_check,
        "v5_source_ood": sorted(
            (entries[str(row["record_id"])] for row in source_ood), key=lambda entry: str(entry["record_id"])
        ),
        "v5_unlabeled": unlabeled,
    }
    exact_duplicate_exclusions = [
        row for row in excluded if row["reason"] == "exact_rgba_duplicate_zero_architecture_value"
    ]
    return {
        "candidate_only": True,
        "exact_duplicate_exclusions": sorted(
            exact_duplicate_exclusions, key=lambda row: (row["record_id"], row["reason"])
        ),
        "excluded_records": sorted(excluded, key=lambda row: (row["record_id"], row["reason"])),
        "production_frozen": False,
        "promotion_forbidden": True,
        "schema_version": VIEW_BUNDLE_SCHEMA_VERSION,
        "split_by_record": split_by_record,
        "training_authorized": False,
        "view_policy_version": VIEW_POLICY_VERSION,
        "views": {
            name: {
                "candidate_only": True,
                "production_frozen": False,
                "promotion_forbidden": True,
                "record_count": len(views[name]),
                "records": views[name],
                "schema_version": VIEW_MANIFEST_SCHEMA_VERSION,
                "training_authorized": False,
                "view_name": name,
            }
            for name in VIEW_NAMES
        },
    }


def write_candidate_view_manifests(
    records: Sequence[Mapping[str, Any]],
    relation_manifest: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Write a fresh, deterministic directory containing all view manifests."""

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"candidate view output already exists: {destination}")
    bundle = build_candidate_views(records, relation_manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        for view_name in VIEW_NAMES:
            _write_json(staging / f"{view_name}.json", bundle["views"][view_name])
        index = {key: value for key, value in bundle.items() if key != "views"}
        index["view_files"] = {name: f"{name}.json" for name in VIEW_NAMES}
        _write_json(staging / "view_index.json", index)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return bundle


def _validated_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        row = dict(record)
        record_id = str(row.get("record_id") or "")
        assert_opaque_id(record_id)
        if record_id in seen:
            raise RawViewError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        _supervision_class(row)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["record_id"]))


def _with_partition_intents(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for source in rows:
        row = dict(source)
        open_set = _is_open_set(row)
        source_ood = _is_source_ood(row)
        if open_set and source_ood:
            raise RawViewError(f"record {row['record_id']} cannot be both source-OOD and open-set")
        requested = str(row.get("requested_split") or "").strip()
        evaluation = row.get("evaluation_candidate") is True
        inferred = "open_set_test" if open_set else "source_ood_test" if source_ood else "test" if evaluation else ""
        if requested and inferred and requested != inferred:
            raise RawViewError(
                f"record {row['record_id']} has incompatible requested split {requested!r} and {inferred!r} intent"
            )
        if not requested and inferred:
            row["requested_split"] = inferred
        prepared.append(row)
    return prepared


def _suitability_decision(row: Mapping[str, Any]) -> str:
    value: Any = None
    for field in ("suitability_status", "suitability_decision"):
        if row.get(field) is not None:
            value = row[field]
            break
    if value is None:
        nested = row.get("source_suitability")
        if nested is not None and not isinstance(nested, Mapping):
            raise RawViewError(f"source_suitability for {row.get('record_id')} must be an object")
        if isinstance(nested, Mapping):
            value = nested.get("status", nested.get("decision"))
    if value is None or not str(value).strip():
        raise RawViewError(f"missing explicit suitability result for {row.get('record_id')}")
    value = str(value).strip().casefold()
    if value not in {"accept", "quarantine", "reject"}:
        raise RawViewError(f"invalid suitability decision for {row.get('record_id')}: {value!r}")
    return value


def _has_unresolved_critical_conflict(row: Mapping[str, Any]) -> bool:
    for field in (
        "critical_conflict",
        "unresolved_critical_conflict",
        "critical_field_conflict",
    ):
        if row.get(field) is True:
            return True
    for field in ("unresolved_critical_conflicts", "critical_field_conflicts"):
        value = row.get(field)
        if isinstance(value, Sequence) and not isinstance(value, str) and len(value) > 0:
            return True
        if isinstance(value, int) and value > 0:
            return True
    return False


def _deduplicate_exact_rgba(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        blob_id = _blob_id(row)
        buckets[blob_id].append(row)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for members in buckets.values():
        ordered = sorted(members, key=lambda row: str(row["record_id"]))
        representative = ordered[0]
        kept.append(representative)
        for duplicate in ordered[1:]:
            excluded.append(
                {
                    "reason": "exact_rgba_duplicate_zero_architecture_value",
                    "record_id": str(duplicate["record_id"]),
                    "representative_record_id": str(representative["record_id"]),
                }
            )
    return sorted(kept, key=lambda row: str(row["record_id"])), excluded


def _blob_id(row: Mapping[str, Any]) -> str:
    for field in ("blob_id", "output_decoded_rgba_sha256", "decoded_rgba_sha256"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    raise RawViewError(f"record {row.get('record_id')} has no decoded RGBA content identity")


def _geometry_key(row: Mapping[str, Any]) -> str:
    value = str(row.get("geometry_family_id") or "").strip()
    return value or f"singleton:{row['record_id']}"


def _relation_kinds_by_record(relation_manifest: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for relation in relation_manifest.get("relations") or []:
        if not isinstance(relation, Mapping):
            raise RawViewError("relation manifest contains a non-object relation")
        kind = str(relation.get("kind") or "")
        for member in relation.get("members") or []:
            result[str(member)].add(kind)
    return {record_id: frozenset(kinds) for record_id, kinds in result.items()}


def _architecture_priority(
    row: Mapping[str, Any], geometry_count: int, relation_kinds: frozenset[str]
) -> tuple[int, str]:
    if geometry_count == 1:
        return (0, "unique_geometry")
    variant_type = str(row.get("variant_type") or row.get("declared_variant_type") or "").casefold()
    if row.get("meaningful_material_variant") is True or variant_type == "material":
        return (1, "meaningful_material_variant")
    if "alpha_recolor" in relation_kinds or variant_type == "recolor":
        return (2, "recolor")
    return (3, "other_unique_blob")


def _view_entry(
    row: Mapping[str, Any],
    *,
    split: str,
    component_id: str,
    geometry_count: int,
    relation_kinds: frozenset[str],
) -> dict[str, Any]:
    rank, priority = _architecture_priority(row, geometry_count, relation_kinds)
    width = _positive_int(row, ("output_width", "width"))
    height = _positive_int(row, ("output_height", "height"))
    result: dict[str, Any] = {
        "architecture_priority": priority,
        "architecture_priority_rank": rank,
        "blob_id": _blob_id(row),
        "geometry_family_id": str(row.get("geometry_family_id") or ""),
        "hard_relation_component_id": component_id,
        "height": height,
        "pixel_area": width * height,
        "record_id": str(row["record_id"]),
        "split": split,
        "supervision_class": _supervision_class(row),
        "width": width,
    }
    category = _category(row)
    if category is not None:
        result["category"] = category
    return result


def _positive_int(row: Mapping[str, Any], fields: Sequence[str]) -> int:
    for field in fields:
        value = row.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise RawViewError(f"record {row.get('record_id')} has no positive {fields[0]}")


def _supervision_class(row: Mapping[str, Any]) -> str:
    value = str(row.get("supervision_class") or "").strip()
    if not value:
        value = "supervised_weak" if _has_blind_label(row) else "unlabeled"
    if value not in SUPERVISION_CLASSES:
        raise RawViewError(f"invalid supervision class for {row.get('record_id')}: {value!r}")
    return value


def _has_blind_label(row: Mapping[str, Any]) -> bool:
    return any(isinstance(row.get(field), Mapping) and bool(row[field]) for field in ("blind_sol_label", "label"))


def _category(row: Mapping[str, Any]) -> str | None:
    direct = str(row.get("category") or "").strip()
    if direct:
        return direct
    for field in ("blind_sol_label", "label"):
        value = row.get(field)
        if isinstance(value, Mapping):
            category = str(value.get("category") or "").strip()
            if category:
                return category
    return None


def _is_source_ood(row: Mapping[str, Any]) -> bool:
    tags = row.get("candidate_views") or row.get("view_tags") or []
    return (
        row.get("source_ood") is True
        or str(row.get("requested_split") or "") == "source_ood_test"
        or (isinstance(tags, Sequence) and not isinstance(tags, str) and "v5_source_ood" in tags)
    )


def _is_open_set(row: Mapping[str, Any]) -> bool:
    tags = row.get("candidate_views") or row.get("view_tags") or []
    state = str(row.get("target_state") or row.get("semantic_state") or "").casefold()
    return (
        row.get("open_set") is True
        or state in {"unknown", "oov"}
        or str(row.get("requested_split") or "") == "open_set_test"
        or (isinstance(tags, Sequence) and not isinstance(tags, str) and "v5_open_set" in tags)
    )


def _balanced_eval_entries(
    rows: Sequence[Mapping[str, Any]], entries: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    labeled = [
        row
        for row in rows
        if _supervision_class(row) in {"supervised_strong", "supervised_weak"} and _category(row) is not None
    ]
    explicit = [row for row in labeled if row.get("evaluation_candidate") is True]
    pool = (
        explicit
        if explicit
        else [row for row in labeled if str(entries[str(row["record_id"])]["split"]) in {"validation", "test"}]
    )
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool:
        by_category[str(_category(row))].append(row)
    if not by_category:
        return []
    per_category = min(len(values) for values in by_category.values())
    selected = []
    for category in sorted(by_category):
        ordered = sorted(by_category[category], key=lambda row: str(row["record_id"]))
        selected.extend(entries[str(row["record_id"])] for row in ordered[:per_category])
    return sorted(
        (dict(entry) for entry in selected), key=lambda entry: (str(entry["category"]), str(entry["record_id"]))
    )


def _architecture_sort_key(entry: Mapping[str, Any]) -> tuple[int, str]:
    return (int(entry["architecture_priority_rank"]), str(entry["record_id"]))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
