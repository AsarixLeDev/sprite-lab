"""Non-destructive pack and variant-family consistency audit (Stage E)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

FAMILY_AUDIT_SCHEMA_VERSION = "label_family_audit_v1.0"
IDENTITY_FIELDS: tuple[str, ...] = ("canonical_object", "category", "domain", "role")


def audit_families(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flag family outliers without altering any individual record."""

    exact_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    recolor_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    geometry_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    declared_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pack_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        image_hash = _nested(record, "pixel_evidence", "image_hash") or record.get("image_hash")
        alpha_hash = _nested(record, "pixel_evidence", "alpha_mask_hash") or record.get("alpha_mask_hash")
        geometry = record.get("geometry_group") or record.get("variant_geometry_group")
        declared = record.get("declared_variant_group") or record.get("declared_variant_family")
        pack = record.get("pack") or record.get("pack_id") or record.get("source_id")
        if image_hash:
            exact_groups[str(image_hash)].append(record)
        if alpha_hash:
            recolor_groups[str(alpha_hash)].append(record)
        if geometry:
            geometry_groups[str(geometry)].append(record)
        if declared:
            declared_groups[str(declared)].append(record)
        if pack:
            pack_groups[str(pack)].append(record)

    findings: list[dict[str, Any]] = []
    findings.extend(_audit_group_map(exact_groups, "exact_duplicate", require_size=2))
    findings.extend(_audit_group_map(recolor_groups, "recolor", require_size=2, ignore_fields=("surface_alias",)))
    findings.extend(_audit_group_map(geometry_groups, "geometry", require_size=2, advisory_only=True))
    findings.extend(_audit_group_map(declared_groups, "declared_variant", require_size=2))
    findings.extend(_pack_outliers(pack_groups))
    findings.sort(key=lambda row: (str(row.get("relation")), str(row.get("group")), str(row.get("sprite_id"))))
    return {
        "schema_version": FAMILY_AUDIT_SCHEMA_VERSION,
        "record_count": len(records),
        "findings": findings,
        "finding_counts": dict(Counter(str(row.get("code")) for row in findings)),
        "automatic_overwrites": 0,
    }


def variant_family_consistent(record: Mapping[str, Any], audit: Mapping[str, Any]) -> bool:
    sprite_id = str(record.get("sprite_id", ""))
    return not any(
        str(row.get("sprite_id", "")) == sprite_id and str(row.get("severity")) in {"warning", "quarantine"}
        for row in audit.get("findings") or ()
        if isinstance(row, Mapping)
    )


def _audit_group_map(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    relation: str,
    *,
    require_size: int,
    ignore_fields: Sequence[str] = (),
    advisory_only: bool = False,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    checked_fields = tuple(field for field in IDENTITY_FIELDS if field not in set(ignore_fields))
    for group_id, rows in sorted(groups.items()):
        if len(rows) < require_size:
            continue
        for field_name in checked_fields:
            values: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                value = _semantic_value(row, field_name)
                if value not in {None, "", "unknown"}:
                    values[_stable_value(value)].append(str(row.get("sprite_id", "")))
            if len(values) <= 1:
                continue
            majority_value, majority_ids = max(values.items(), key=lambda pair: (len(pair[1]), pair[0]))
            for value, sprite_ids in sorted(values.items()):
                if value == majority_value:
                    continue
                for sprite_id in sprite_ids:
                    findings.append(
                        {
                            "sprite_id": sprite_id,
                            "relation": relation,
                            "group": group_id,
                            "field": field_name,
                            "code": f"{relation}_field_outlier",
                            "observed_value": value,
                            "family_majority_value": majority_value,
                            "family_majority_support_n": len(majority_ids),
                            "severity": "advisory" if advisory_only else "warning",
                            "action": "flag_only_no_overwrite",
                        }
                    )
    return findings


def _pack_outliers(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for pack, rows in sorted(groups.items()):
        if len(rows) < 4:
            continue
        categories = Counter(
            str(_semantic_value(row, "category"))
            for row in rows
            if _semantic_value(row, "category") not in {None, "", "unknown"}
        )
        if not categories:
            continue
        majority, support = categories.most_common(1)[0]
        homogeneity = support / len(rows)
        if homogeneity < 0.75:
            continue
        for row in rows:
            category = str(_semantic_value(row, "category") or "")
            if category and category != majority:
                findings.append(
                    {
                        "sprite_id": str(row.get("sprite_id", "")),
                        "relation": "pack",
                        "group": pack,
                        "field": "category",
                        "code": "pack_category_outlier",
                        "observed_value": category,
                        "family_majority_value": majority,
                        "family_majority_support_n": support,
                        "pack_homogeneity": homogeneity,
                        "severity": "advisory",
                        "action": "flag_only_no_overwrite",
                    }
                )
    return findings


def _semantic_value(record: Mapping[str, Any], field_name: str) -> Any:
    semantics = record.get("semantics")
    if isinstance(semantics, Mapping) and field_name in semantics:
        return semantics.get(field_name)
    reconciliation = record.get("reconciliation")
    proposals = reconciliation.get("field_proposals") if isinstance(reconciliation, Mapping) else None
    proposal = proposals.get(field_name) if isinstance(proposals, Mapping) else None
    if isinstance(proposal, Mapping):
        return proposal.get("normalized_controlled_value", proposal.get("value"))
    return record.get(field_name)


def _nested(record: Mapping[str, Any], parent: str, key: str) -> Any:
    value = record.get(parent)
    return value.get(key) if isinstance(value, Mapping) else None


def _stable_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "|".join(sorted(str(item) for item in value))
    return str(value)
