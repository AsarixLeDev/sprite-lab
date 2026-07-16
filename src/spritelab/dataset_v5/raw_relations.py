"""Hard-relation discovery and split closure for the raw Dataset-v5 rebuild.

Relation evidence can include post-label provenance such as pack and creator
lineage.  Canonical relation and component identifiers never include those
values: they are hashes of the relation policy, relation kind, and sorted
opaque record identities only.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.dataset_v5.identity import assert_opaque_id, canonical_json_bytes

RELATION_POLICY_VERSION = "sprite_lab_raw_hard_relations_v1"
RELATION_MANIFEST_SCHEMA_VERSION = "sprite_lab_raw_relation_manifest_v1"

HARD_RELATION_KINDS = (
    "exact_rgba",
    "alpha_recolor",
    "translation",
    "known_flip",
    "declared_variant",
    "geometry_family",
    "sheet",
    "pack",
    "creator_lineage",
    "other_hard",
)


class RawRelationError(ValueError):
    """Raised when relation evidence cannot be closed deterministically."""


class HardRelationLeakageError(RawRelationError):
    """Raised when members of one hard component would cross splits."""


@dataclass(frozen=True)
class _RelationSpec:
    kind: str
    scalar_fields: tuple[str, ...]
    collection_fields: tuple[str, ...] = ()
    reference_fields: tuple[str, ...] = ()


_RELATION_SPECS = (
    _RelationSpec(
        "exact_rgba",
        ("blob_id", "output_decoded_rgba_sha256", "decoded_rgba_sha256", "exported_rgba_hash"),
    ),
    _RelationSpec(
        "alpha_recolor",
        ("alpha_mask_sha256", "alpha_mask_hash", "exact_alpha_family_id"),
    ),
    _RelationSpec(
        "translation",
        ("translation_family_id", "translation_normalized_alpha_sha256", "translation_group_id"),
    ),
    _RelationSpec(
        "known_flip",
        (
            "known_flip_family_id",
            "flip_family_id",
            "flip_canonical_geometry_id",
            "horizontal_flip_family_id",
            "vertical_flip_family_id",
            "declared_flip_group_id",
        ),
        reference_fields=("known_flip_of", "horizontal_flip_of", "vertical_flip_of"),
    ),
    _RelationSpec(
        "declared_variant",
        ("declared_variant_group_id", "declared_variant_family", "known_variant_family"),
        collection_fields=("declared_variant_ids",),
        reference_fields=("declared_variant_of",),
    ),
    _RelationSpec("geometry_family", ("geometry_family_id",)),
    _RelationSpec("sheet", ("sheet_id", "source_sheet", "sheet_membership_id")),
    _RelationSpec("pack", ("source_pack_id", "source_pack")),
    _RelationSpec("creator_lineage", ("creator_lineage_id", "creator_lineage", "source_creator")),
    _RelationSpec(
        "other_hard",
        ("other_hard_relation_id",),
        collection_fields=("hard_relation_group_ids", "other_hard_relation_ids"),
        reference_fields=("hard_relation_with",),
    ),
)


class _DSU:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # The lexical rule makes closure independent of input/edge order.
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def build_relation_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Detect every required hard relation and compute its DSU closure.

    The function accepts extraction, suitability, and reconciled provenance
    rows.  Missing optional evidence does not invent a relation; malformed or
    dangling evidence fails closed.
    """

    rows = [dict(record) for record in records]
    record_ids = _validated_record_ids(rows)
    known_ids = set(record_ids)
    relation_members: dict[tuple[str, tuple[str, ...]], None] = {}

    for spec in _RELATION_SPECS:
        buckets: dict[str, list[str]] = defaultdict(list)
        references: set[tuple[str, str]] = set()
        for row in rows:
            record_id = str(row["record_id"])
            if spec.kind == "exact_rgba":
                _validate_exact_content_aliases(row)
            scalar = _first_scalar(row, spec.scalar_fields)
            if spec.kind == "sheet" and scalar is None:
                scalar = _inferred_sheet_key(row)
            if scalar is not None:
                buckets[scalar].append(record_id)
            for field in spec.collection_fields:
                for value in _collection_values(row.get(field), field=field):
                    if value in known_ids:
                        references.add(tuple(sorted((record_id, value))))
                    else:
                        buckets[value].append(record_id)
            for field in spec.reference_fields:
                for target in _reference_values(row.get(field), field=field):
                    if target not in known_ids:
                        raise RawRelationError(f"{field} for {record_id} references unknown opaque record {target!r}")
                    if target != record_id:
                        references.add(tuple(sorted((record_id, target))))

        for members in buckets.values():
            _remember_relation(relation_members, spec.kind, members)
        for pair in references:
            _remember_relation(relation_members, spec.kind, pair)

    # The extractor's geometry identity is translation-normalized.  Distinct
    # exact alpha masks in one geometry family are therefore translations even
    # when a separate translation-family field was not emitted.
    _infer_geometry_relations(rows, relation_members)

    relations = [_relation_row(kind, members) for kind, members in relation_members]
    relations.sort(key=lambda row: str(row["relation_id"]))
    components = _closed_components(record_ids, relations)
    record_to_component = {
        member: str(component["component_id"]) for component in components for member in component["members"]
    }
    return {
        "hard_relation_components": components,
        "hard_relation_kinds": list(HARD_RELATION_KINDS),
        "record_count": len(record_ids),
        "record_to_component": dict(sorted(record_to_component.items())),
        "relation_policy_version": RELATION_POLICY_VERSION,
        "relations": relations,
        "schema_version": RELATION_MANIFEST_SCHEMA_VERSION,
    }


def validate_relation_manifest(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require an exact deterministic replay of a relation manifest."""

    expected = build_relation_manifest(records)
    actual = dict(manifest)
    if actual != expected:
        raise RawRelationError("relation manifest does not match deterministic relation discovery")
    return {
        "component_count": len(expected["hard_relation_components"]),
        "ok": True,
        "relation_count": len(expected["relations"]),
        "schema_version": "sprite_lab_raw_relation_verification_v1",
    }


def assign_component_splits(
    records: Sequence[Mapping[str, Any]],
    relation_manifest: Mapping[str, Any],
    *,
    requested_split_field: str = "requested_split",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, str]:
    """Assign whole hard components to train/validation/test deterministically.

    Explicit split requests are honored only when the entire component agrees.
    Any disagreement fails instead of silently moving an evidence-linked row.
    Components without a request use a stable hash bucket; target counts are
    never forced.
    """

    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-12:
        raise ValueError("split ratios must be three non-negative values summing to one")
    index = _record_index(records)
    components = _validated_components(relation_manifest, set(index))
    result: dict[str, str] = {}
    valid_splits = {"train", "validation", "test", "source_ood_test", "open_set_test", "unsplit"}
    cut_train = int(ratios[0] * 10_000)
    cut_validation = int((ratios[0] + ratios[1]) * 10_000)

    for component in components:
        members = [str(value) for value in component["members"]]
        requests = {
            str(index[member].get(requested_split_field) or "").strip()
            for member in members
            if str(index[member].get(requested_split_field) or "").strip()
        }
        invalid = sorted(requests - valid_splits)
        if invalid:
            raise RawRelationError(f"unsupported requested splits for component {component['component_id']}: {invalid}")
        if len(requests) > 1:
            raise HardRelationLeakageError(
                f"hard component {component['component_id']} has conflicting split requests: {sorted(requests)}"
            )
        if requests:
            split = next(iter(requests))
        else:
            bucket = (
                int(
                    hashlib.sha256(
                        b"sprite_lab_raw_split_v1\0" + str(component["component_id"]).encode("ascii")
                    ).hexdigest()[:8],
                    16,
                )
                % 10_000
            )
            split = "train" if bucket < cut_train else "validation" if bucket < cut_validation else "test"
        result.update(dict.fromkeys(members, split))

    assert_no_hard_relation_crossing(relation_manifest, result, require_all=True)
    return dict(sorted(result.items()))


def assert_no_hard_relation_crossing(
    relation_manifest: Mapping[str, Any],
    split_by_record: Mapping[str, str],
    *,
    require_all: bool = True,
) -> None:
    """Fail if a closed hard component is missing or appears in two splits."""

    components = relation_manifest.get("hard_relation_components")
    if not isinstance(components, list):
        raise RawRelationError("relation manifest has no hard_relation_components list")
    for component in components:
        if not isinstance(component, Mapping) or not isinstance(component.get("members"), list):
            raise RawRelationError("invalid hard relation component")
        members = [str(value) for value in component["members"]]
        missing = [member for member in members if member not in split_by_record]
        if missing and require_all:
            raise HardRelationLeakageError(
                f"hard component {component.get('component_id')} has records without split assignment: {missing}"
            )
        splits = {str(split_by_record[member]) for member in members if member in split_by_record}
        if len(splits) > 1:
            raise HardRelationLeakageError(
                f"hard component {component.get('component_id')} crosses forbidden splits: {sorted(splits)}"
            )


def _validated_record_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        record_id = str(row.get("record_id") or "")
        assert_opaque_id(record_id)
        if record_id in seen:
            raise RawRelationError(f"duplicate record_id: {record_id}")
        geometry_id = row.get("geometry_family_id")
        if geometry_id is not None:
            assert_opaque_id(str(geometry_id), kind="geometry")
        seen.add(record_id)
        values.append(record_id)
    return sorted(values)


def _record_index(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    rows = [dict(record) for record in records]
    _validated_record_ids(rows)
    return {str(row["record_id"]): row for row in rows}


def _first_scalar(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise RawRelationError(f"{field} must be a string when present")
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _validate_exact_content_aliases(row: Mapping[str, Any]) -> None:
    fields = ("blob_id", "output_decoded_rgba_sha256", "decoded_rgba_sha256", "exported_rgba_hash")
    values: dict[str, str] = {}
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RawRelationError(f"{field} must be a lowercase SHA-256 for {row.get('record_id')}")
        values[field] = value
    if len(set(values.values())) > 1:
        raise RawRelationError(
            f"inconsistent decoded RGBA content identities for {row.get('record_id')}: {sorted(values)}"
        )


def _collection_values(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RawRelationError(f"{field} must be a sequence of nonempty strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RawRelationError(f"{field} must contain only nonempty strings")
        result.append(item.strip())
    return tuple(sorted(set(result)))


def _reference_values(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        raise RawRelationError(f"{field} must be an opaque record ID or sequence of IDs")
    result = []
    for item in values:
        if not isinstance(item, str):
            raise RawRelationError(f"{field} must contain only opaque record IDs")
        assert_opaque_id(item)
        result.append(item)
    return tuple(sorted(set(result)))


def _remember_relation(relations: dict[tuple[str, tuple[str, ...]], None], kind: str, members: Iterable[str]) -> None:
    values = tuple(sorted(set(members)))
    if len(values) >= 2:
        relations[(kind, values)] = None


def _infer_geometry_relations(
    rows: Sequence[Mapping[str, Any]],
    relations: dict[tuple[str, tuple[str, ...]], None],
) -> None:
    geometry_buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        geometry = _first_scalar(row, ("geometry_family_id",))
        if geometry:
            geometry_buckets[geometry].append(row)
    for members in geometry_buckets.values():
        explicit_alpha = [_first_scalar(row, ("alpha_mask_sha256", "alpha_mask_hash")) for row in members]
        if all(value is not None for value in explicit_alpha):
            position_keys = [str(value) for value in explicit_alpha]
        else:
            position_keys = [_alpha_position_key(row) for row in members]
        if all(value is not None for value in position_keys):
            position_buckets: dict[str, list[str]] = defaultdict(list)
            for row, position in zip(members, position_keys, strict=True):
                position_buckets[str(position)].append(str(row["record_id"]))
            for alpha_members in position_buckets.values():
                _remember_relation(relations, "alpha_recolor", alpha_members)
        if len({value for value in position_keys if value is not None}) > 1:
            _remember_relation(relations, "translation", (str(row["record_id"]) for row in members))


def _alpha_position_key(row: Mapping[str, Any]) -> str | None:
    bbox = row.get("tight_foreground_bbox")
    width = row.get("output_width", row.get("width"))
    height = row.get("output_height", row.get("height"))
    if bbox is None or not isinstance(bbox, Sequence) or isinstance(bbox, str) or len(bbox) != 4:
        return None
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in bbox):
        raise RawRelationError(f"invalid tight_foreground_bbox for {row.get('record_id')}")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise RawRelationError(f"invalid output width for {row.get('record_id')}")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise RawRelationError(f"invalid output height for {row.get('record_id')}")
    return canonical_json_bytes({"bbox": list(bbox), "height": height, "width": width}).hex()


def _inferred_sheet_key(row: Mapping[str, Any]) -> str | None:
    if row.get("crop_coordinates") is None:
        return None
    archive_hash = str(row.get("source_archive_sha256") or "").strip()
    member = str(row.get("archive_member_path") or "").replace("\\", "/").strip()
    if not archive_hash or not member:
        raise RawRelationError(f"cropped record {row.get('record_id')} lacks archive/member evidence for sheet closure")
    return canonical_json_bytes(
        {"archive_sha256": archive_hash, "archive_member_path": member, "kind": "cropped_source_sheet"}
    ).hex()


def _relation_row(kind: str, members: tuple[str, ...]) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "members": list(members),
        "policy_version": RELATION_POLICY_VERSION,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {
        "hard_split_constraint": True,
        "kind": kind,
        "members": list(members),
        "relation_id": f"rel_{digest}",
    }


def _closed_components(record_ids: Sequence[str], relations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dsu = _DSU(record_ids)
    for relation in relations:
        members = [str(value) for value in relation["members"]]
        for member in members[1:]:
            dsu.union(members[0], member)
    buckets: dict[str, list[str]] = defaultdict(list)
    for record_id in record_ids:
        buckets[dsu.find(record_id)].append(record_id)
    result = []
    for members in buckets.values():
        values = sorted(members)
        kinds = sorted(
            {
                str(relation["kind"])
                for relation in relations
                if {str(value) for value in relation["members"]} & set(values)
            }
        )
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "members": values,
                    "policy_version": RELATION_POLICY_VERSION,
                }
            )
        ).hexdigest()
        result.append(
            {
                "component_id": f"grp_{digest}",
                "members": values,
                "relation_kinds": kinds,
            }
        )
    return sorted(result, key=lambda row: str(row["component_id"]))


def _validated_components(relation_manifest: Mapping[str, Any], known_ids: set[str]) -> list[Mapping[str, Any]]:
    components = relation_manifest.get("hard_relation_components")
    if not isinstance(components, list):
        raise RawRelationError("relation manifest has no hard_relation_components list")
    found: list[str] = []
    for component in components:
        if not isinstance(component, Mapping) or not isinstance(component.get("members"), list):
            raise RawRelationError("invalid hard relation component")
        members = [str(value) for value in component["members"]]
        if members != sorted(set(members)) or not members:
            raise RawRelationError("hard relation component members must be sorted and unique")
        found.extend(members)
    if len(found) != len(set(found)) or set(found) != known_ids:
        raise RawRelationError("hard relation components are not an exact partition of record IDs")
    return sorted(components, key=lambda row: str(row.get("component_id") or ""))
