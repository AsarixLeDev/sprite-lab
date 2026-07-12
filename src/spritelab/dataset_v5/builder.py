"""Auditable, deterministic assembly for Sprite Lab dataset v5 previews."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BUILDER_VERSION = "dataset_v5_builder.1"
RGBA_MARKER = b"spritelab-exported-rgba-v1\0"
PARTITIONS = ("train", "validation", "test", "source_ood_test", "open_set_test")
REQUIRED_EXPORTS = (
    "dataset_manifest.jsonl",
    "dataset_summary.json",
    "split_manifest.json",
    "group_manifest.jsonl",
    "excluded_manifest.jsonl",
    "license_manifest.json",
    "README.md",
)
REQUIRED_PROVENANCE = (
    "source_id",
    "source_pack",
    "source_url",
    "license",
    "attribution",
    "downloaded_file_hash",
    "archive_member",
    "source_image",
    "author",
    "resize_policy",
    "original_width",
    "original_height",
    "label_provenance",
    "suitability_status",
)


@dataclass(frozen=True)
class BalancePolicy:
    max_pack_share: float = 0.15
    max_artist_share: float = 0.15
    max_source_family_share: float = 1.0
    max_exact_object_share: float = 1.0
    max_recolor_family_share: float = 1.0
    max_near_duplicate_family_share: float = 1.0


@dataclass(frozen=True)
class BuilderConfig:
    dataset_name: str = "sprite_lab_multisource_v5_preview"
    seed: int = 20260711
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    source_ood_packs: tuple[str, ...] = ()
    held_out_artists: tuple[str, ...] = ()
    open_set_objects: tuple[str, ...] = ()
    balance: BalancePolicy = field(default_factory=BalancePolicy)
    strict_quality_buckets: tuple[str, ...] = ("human", "golden", "reviewed")

    @classmethod
    def from_json(cls, path: str | Path) -> BuilderConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        balance = BalancePolicy(**raw.pop("balance", {}))
        for key in ("source_ood_packs", "held_out_artists", "open_set_objects", "strict_quality_buckets"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(balance=balance, **raw)

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


def canonical_rgba_sha256(rgba: np.ndarray) -> str:
    """Hash the versioned, dimensioned canonical exported decoded RGBA representation."""

    value = np.ascontiguousarray(rgba, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 4:
        raise ValueError(f"RGBA must have shape [H,W,4], got {value.shape}")
    height, width = value.shape[:2]
    digest = hashlib.sha256()
    digest.update(RGBA_MARKER)
    digest.update(struct.pack(">II", width, height))
    digest.update(value.tobytes())
    return digest.hexdigest()


def alpha_mask_sha256(alpha: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(alpha) > 0, dtype=np.uint8)
    height, width = value.shape
    return hashlib.sha256(
        b"spritelab-alpha-mask-v1\0" + struct.pack(">II", width, height) + value.tobytes()
    ).hexdigest()


def source_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _DSU:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_dataset(
    *,
    v4_dir: str | Path,
    harvest_root: str | Path,
    output_dir: str | Path,
    config: BuilderConfig | None = None,
    explicit_manifests: tuple[str | Path, ...] = (),
    freeze_timestamp: str | None = None,
) -> dict[str, Any]:
    """Run discover through freeze and return the verification report."""

    config = config or BuilderConfig()
    v4 = Path(v4_dir).resolve()
    harvest = Path(harvest_root).resolve()
    output = Path(output_dir).resolve()
    if (output / "FREEZE.json").exists():
        raise FileExistsError(f"refusing to overwrite frozen dataset: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    config_hash = _hash_json(config.canonical())

    discovered = _discover(v4, harvest, tuple(Path(p).resolve() for p in explicit_manifests))
    _stage(output, "01_discover", "discovered_inputs.json", discovered)
    records, exclusions = _load_v4(v4, harvest, config_hash)
    records.extend(_load_explicit(explicit_manifests, config_hash, exclusions))
    _write_jsonl(output / "audit/02_validate/validated_records.jsonl", [_public(row) for row in records])
    _write_jsonl(output / "audit/02_validate/validation_exclusions.jsonl", exclusions)

    records, duplicate_exclusions, duplicate_report = _deduplicate(records)
    exclusions.extend(duplicate_exclusions)
    _stage(output, "03_deduplicate", "deduplication.json", duplicate_report)

    relations, groups = build_groups(records)
    _write_jsonl(output / "audit/04_group/relation_graph.jsonl", relations)
    _write_jsonl(output / "audit/04_group/final_split_groups.jsonl", groups)
    group_by_id = {member: row["split_group_id"] for row in groups for member in row["members"]}
    relation_family = _relation_family_maps(relations)

    before_balance = distribution_report(records)
    balanced, balance_exclusions, balance_report = enforce_balance(
        records, config.balance, config.seed, relation_family
    )
    exclusions.extend(balance_exclusions)
    balance_report["before"] = before_balance
    balance_report["after"] = distribution_report(balanced)
    _stage(output, "05_balance", "balance_report.json", balance_report)
    _write_jsonl(output / "audit/05_balance/overflow_manifest.jsonl", balance_exclusions)

    assignments = assign_splits(balanced, group_by_id, config)
    split_report = {"assignments": assignments, "counts": dict(Counter(assignments.values()))}
    _stage(output, "06_split", "split_plan.json", split_report)

    export_report = _export(
        output, balanced, records, assignments, groups, relations, exclusions, discovered, config, config_hash
    )
    _stage(output, "07_export", "export_report.json", export_report)
    verification = verify_dataset(output, v4_dir=v4, expected_v4_hashes=discovered["v4_files"])
    _stage(output, "08_verify", "verification.json", verification)
    if not verification["ok"]:
        raise ValueError(f"dataset verification failed: {verification['errors']}")
    freeze = _freeze(output, discovered, config_hash, verification, freeze_timestamp)
    _stage(output, "09_freeze", "freeze_result.json", freeze)
    return {**verification, "freeze": freeze, "summary": _read_json(output / "dataset_summary.json")}


def _discover(v4: Path, harvest: Path, explicit: tuple[Path, ...]) -> dict[str, Any]:
    if not v4.is_dir():
        raise FileNotFoundError(v4)
    v4_files = {path.name: source_file_sha256(path) for path in sorted(v4.iterdir()) if path.is_file()}
    manifest_paths = [v4 / f"manifest_{name}.jsonl" for name in ("train", "val", "test")]
    manifest_paths.extend(explicit)
    harvest_manifests: list[Path] = []
    for run in sorted(harvest.iterdir()) if harvest.is_dir() else ():
        if run.is_dir():
            harvest_manifests.extend(
                run / name for name in ("sources.jsonl", "candidates.jsonl", "imported.jsonl") if (run / name).is_file()
            )
    source_hashes = {str(path): source_file_sha256(path) for path in sorted(manifest_paths + harvest_manifests)}
    return {
        "adapter": "immutable_v4_plus_harvest_v1",
        "v4_files": v4_files,
        "source_manifest_hashes": source_hashes,
        "explicit_manifests": [str(path) for path in explicit],
    }


def _load_v4(v4: Path, harvest: Path, config_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_id, candidates, imported = _harvest_indexes(harvest)
    training_by_id: dict[str, dict[str, Any]] = {}
    training_path = v4 / "training_manifest.jsonl"
    if training_path.is_file():
        for row in _read_jsonl(training_path):
            training_by_id.setdefault(str(row.get("sprite_id", "")), row)
    valid: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for old_split in ("train", "val", "test"):
        manifests = _read_jsonl(v4 / f"manifest_{old_split}.jsonl")
        with np.load(v4 / f"{old_split}.npz", allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        npz_ids = {str(value): index for index, value in enumerate(arrays["sprite_id"])}
        for manifest in manifests:
            sprite_id = str(manifest.get("sprite_id", ""))
            index = npz_ids.get(sprite_id)
            if index is None:
                excluded.append(_exclude(sprite_id, "missing_npz_row", "validate"))
                continue
            row_arrays = {key: np.asarray(value[index]) for key, value in arrays.items()}
            rgba = _rgba_from_arrays(row_arrays)
            alpha = np.asarray(row_arrays["alpha"], dtype=np.uint8)
            source_id = str(manifest.get("source_pack") or manifest.get("source_dataset") or "")
            imp = imported.get(sprite_id, {})
            relative = str(imp.get("relative_path") or Path(str(manifest.get("source_path", ""))).name)
            candidate = candidates.get((source_id, _norm(relative)), {})
            source = source_by_id.get(source_id, {})
            license_data = source.get("license") if isinstance(source.get("license"), dict) else {}
            author = str(manifest.get("author") or imp.get("author") or source.get("author") or "").strip()
            record: dict[str, Any] = {
                "sprite_id": sprite_id,
                "source_id": source_id,
                "source_pack": source_id,
                "source_family": str(manifest.get("source_family") or source.get("source_type") or source_id),
                "source_url": str(source.get("source_url") or source.get("download_url") or ""),
                "license": str(manifest.get("license") or license_data.get("license") or ""),
                "license_url": str(license_data.get("license_url") or ""),
                "license_confirmed": bool(license_data.get("user_confirmed")),
                "attribution": author,
                "downloaded_file_hash": str(candidate.get("image_sha256") or source.get("sha256") or ""),
                "archive_member": relative,
                "source_image": str(manifest.get("source_path") or candidate.get("extracted_path") or relative),
                "source_sheet": str(
                    manifest.get("source_sheet")
                    or manifest.get("sheet_id")
                    or f"{source_id}:{Path(str(manifest.get('source_path') or relative)).parent.as_posix()}"
                ),
                "cell_coordinates": manifest.get("cell_coordinates"),
                "author": author,
                "sub_artist": str(manifest.get("sub_artist") or author),
                "resize_policy": str(manifest.get("resize_policy") or "preserve_existing_32x32_export"),
                "original_width": candidate.get("width"),
                "original_height": candidate.get("height"),
                "exported_width": int(rgba.shape[1]),
                "exported_height": int(rgba.shape[0]),
                "exported_rgba_hash": canonical_rgba_sha256(rgba),
                "alpha_mask_hash": alpha_mask_sha256(alpha),
                "object_name": str(manifest.get("object_name") or ""),
                "category": str(manifest.get("category") or "unknown"),
                "label_provenance": _label_provenance(manifest),
                "label_quality": _label_quality(manifest, training_by_id.get(sprite_id)),
                "is_supervised": bool(manifest.get("object_name") and manifest.get("category")),
                "suitability_status": "approved_existing_dataset",
                "dataset_builder_version": BUILDER_VERSION,
                "config_hash": config_hash,
                "declared_variant_ids": _variant_values(manifest),
                "animation_group": str(manifest.get("animation_group") or ""),
                "known_variant_family": str(manifest.get("known_variant_family") or ""),
                "prior_split": {"val": "validation"}.get(old_split, old_split),
                "source_dataset": str(manifest.get("source_dataset") or ""),
                "strict_quality": _strict_quality(manifest),
                "training_record": training_by_id.get(sprite_id),
                "_rgba": rgba,
                "_alpha": alpha,
                "_arrays": row_arrays,
            }
            reasons = _validation_reasons(record)
            if "flare" in source_id.lower():
                reasons.append("rejected_flare_source")
            if reasons:
                excluded.append(_exclude(sprite_id, reasons[0], "validate", details={"all_reasons": reasons}))
            else:
                valid.append(record)
    return sorted(valid, key=lambda row: row["sprite_id"]), sorted(excluded, key=lambda row: row["sprite_id"])


def _harvest_indexes(
    harvest: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    sources: dict[str, dict[str, Any]] = {}
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    imported: dict[str, dict[str, Any]] = {}
    if not harvest.is_dir():
        return sources, candidates, imported
    for run in sorted(harvest.iterdir()):
        if not run.is_dir():
            continue
        for row in _safe_jsonl(run / "sources.jsonl"):
            sources[str(row.get("source_id", ""))] = row
        for row in _safe_jsonl(run / "candidates.jsonl"):
            candidates[(str(row.get("source_id", "")), _norm(str(row.get("relative_path", ""))))] = row
        for row in _safe_jsonl(run / "imported.jsonl"):
            imported[str(row.get("sprite_id", ""))] = row
    return sources, candidates, imported


def _load_explicit(
    paths: tuple[str | Path, ...], config_hash: str, exclusions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        for raw in _read_jsonl(path):
            sprite_id = str(raw.get("sprite_id", ""))
            rgba_path = Path(str(raw.get("rgba_path", "")))
            if not rgba_path.is_absolute():
                rgba_path = path.parent / rgba_path
            if not rgba_path.is_file():
                exclusions.append(_exclude(sprite_id, "missing_rgba_path", "validate"))
                continue
            from PIL import Image

            with Image.open(rgba_path) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            alpha = (rgba[:, :, 3] > 0).astype(np.uint8)
            row = dict(raw)
            row.update(
                exported_rgba_hash=canonical_rgba_sha256(rgba),
                alpha_mask_hash=alpha_mask_sha256(alpha),
                exported_width=rgba.shape[1],
                exported_height=rgba.shape[0],
                dataset_builder_version=BUILDER_VERSION,
                config_hash=config_hash,
                declared_variant_ids=_variant_values(raw),
                _rgba=rgba,
                _alpha=alpha,
                _arrays=None,
                training_record=None,
            )
            reasons = _validation_reasons(row)
            if reasons:
                exclusions.append(_exclude(sprite_id, reasons[0], "validate", details={"all_reasons": reasons}))
            else:
                records.append(row)
    return records


def _validation_reasons(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(record.get("license", "")).lower() in {"", "unknown", "none"}:
        reasons.append("missing_or_unknown_license")
    if not record.get("license_confirmed") and not record.get("license_url"):
        reasons.append("unconfirmed_license")
    for field_name in REQUIRED_PROVENANCE:
        value = record.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            reasons.append(f"missing_provenance:{field_name}")
    if (record.get("exported_width"), record.get("exported_height")) != (32, 32):
        reasons.append("export_not_32x32")
    if not record.get("is_supervised") and not record.get("unlabeled_pool"):
        reasons.append("unlabeled_without_explicit_pool")
    return reasons


def _deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["exported_rgba_hash"]].append(row)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for rgba_hash, members in sorted(groups.items()):
        members.sort(key=lambda row: row["sprite_id"])
        representative = members[0]
        representative["duplicate_source_records"] = [_public(row) for row in members]
        kept.append(representative)
        if len(members) > 1:
            duplicate_groups.append(
                {
                    "exported_rgba_hash": rgba_hash,
                    "representative": representative["sprite_id"],
                    "members": [row["sprite_id"] for row in members],
                }
            )
            for row in members[1:]:
                excluded.append(
                    _exclude(
                        row["sprite_id"],
                        "exact_exported_rgba_duplicate",
                        "deduplicate",
                        representative=representative["sprite_id"],
                    )
                )
    return (
        kept,
        excluded,
        {"input_count": len(records), "output_count": len(kept), "exact_duplicate_groups": duplicate_groups},
    )


def build_groups(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relations: list[dict[str, Any]] = []
    hard_edges: list[tuple[str, str]] = []

    def add(kind: str, members: list[str], *, hard: bool, key: str) -> None:
        values = sorted(set(members))
        if len(values) < 2:
            return
        relation_id = "rel_" + hashlib.sha256(f"{kind}:{key}".encode()).hexdigest()[:20]
        relations.append(
            {"relation_id": relation_id, "kind": kind, "key": key, "members": values, "hard_split_constraint": hard}
        )
        if hard:
            hard_edges.extend((values[0], value) for value in values[1:])

    for field_name, kind, hard in (
        ("exported_rgba_hash", "exact_exported_rgba", True),
        ("alpha_mask_hash", "exact_alpha_mask_recolor", True),
        ("source_sheet", "source_sheet_siblings", True),
        ("animation_group", "animation_siblings", True),
        ("known_variant_family", "known_variant_family", True),
        ("source_pack", "same_source_pack", False),
        ("sub_artist", "same_artist_or_subartist", False),
    ):
        buckets: dict[str, list[str]] = defaultdict(list)
        for row in records:
            value = str(row.get(field_name) or "")
            if value:
                buckets[value].append(row["sprite_id"])
        for key, members in sorted(buckets.items()):
            add(kind, members, hard=hard, key=key)
    declared: dict[str, list[str]] = defaultdict(list)
    translated: dict[str, list[str]] = defaultdict(list)
    for row in records:
        for value in row.get("declared_variant_ids", []):
            declared[value].append(row["sprite_id"])
        translated[_normalized_alpha_key(row["_alpha"])].append(row["sprite_id"])
    for key, members in sorted(declared.items()):
        add("declared_variant_family", members, hard=True, key=key)
    for key, members in sorted(translated.items()):
        add("translation_padding_variant", members, hard=True, key=key)

    guarded: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        guarded[(row["source_pack"], row["object_name"])].append(row)
    for (pack, object_name), members in sorted(guarded.items()):
        if not object_name or len(members) > 250:
            continue
        near_members: set[str] = set()
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if int(np.count_nonzero(left["_alpha"] != right["_alpha"])) <= 4:
                    near_members.update((left["sprite_id"], right["sprite_id"]))
        add("near_identical_geometry_guarded", sorted(near_members), hard=True, key=f"{pack}:{object_name}")

    dsu = _DSU([row["sprite_id"] for row in records])
    for left, right in hard_edges:
        dsu.union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for row in records:
        components[dsu.find(row["sprite_id"])].append(row["sprite_id"])
    groups = []
    for members in sorted(components.values(), key=lambda values: sorted(values)[0]):
        values = sorted(members)
        group_id = "split_" + hashlib.sha256("|".join(values).encode()).hexdigest()[:20]
        reasons = sorted(
            {rel["kind"] for rel in relations if rel["hard_split_constraint"] and set(rel["members"]) & set(values)}
        )
        groups.append({"split_group_id": group_id, "members": values, "hard_relation_kinds": reasons})
    return sorted(relations, key=lambda row: row["relation_id"]), groups


def _relation_family_maps(relations: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {"recolor": {}, "near": {}}
    for row in relations:
        target = (
            "recolor"
            if row["kind"] == "exact_alpha_mask_recolor"
            else "near"
            if row["kind"] == "near_identical_geometry_guarded"
            else ""
        )
        if target:
            for member in row["members"]:
                result[target][member] = row["relation_id"]
    return result


def enforce_balance(
    records: list[dict[str, Any]], policy: BalancePolicy, seed: int, families: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    remaining = list(records)
    excluded: list[dict[str, Any]] = []
    dimensions = (
        ("source_pack", policy.max_pack_share, lambda row: row["source_pack"], "pack_share_exceeded"),
        ("artist", policy.max_artist_share, lambda row: row["sub_artist"], "artist_share_exceeded"),
        (
            "source_family",
            policy.max_source_family_share,
            lambda row: row["source_family"],
            "source_family_share_exceeded",
        ),
        ("exact_object", policy.max_exact_object_share, lambda row: row["object_name"], "exact_object_share_exceeded"),
        (
            "recolor_family",
            policy.max_recolor_family_share,
            lambda row: families["recolor"].get(row["sprite_id"], row["sprite_id"]),
            "recolor_family_share_exceeded",
        ),
        (
            "near_duplicate_family",
            policy.max_near_duplicate_family_share,
            lambda row: families["near"].get(row["sprite_id"], row["sprite_id"]),
            "near_duplicate_family_share_exceeded",
        ),
    )
    infeasible: list[dict[str, Any]] = []
    for name, cap, key_fn, reason in dimensions:
        if cap >= 1.0 or not remaining:
            continue
        while remaining:
            counts = Counter(key_fn(row) for row in remaining)
            dominant, count = max(counts.items(), key=lambda item: (item[1], str(item[0])))
            if count / len(remaining) <= cap + 1e-12:
                break
            if len(counts) == 1:
                infeasible.append({"dimension": name, "value": dominant, "share": 1.0, "limit": cap})
                break
            candidates = [row for row in remaining if key_fn(row) == dominant]
            victim = max(candidates, key=lambda row: _stable_key(seed, f"balance:{name}:{row['sprite_id']}"))
            remaining.remove(victim)
            excluded.append(
                _exclude(
                    victim["sprite_id"], reason, "balance", details={"dimension": name, "value": dominant, "limit": cap}
                )
            )
    return (
        sorted(remaining, key=lambda row: row["sprite_id"]),
        sorted(excluded, key=lambda row: row["sprite_id"]),
        {"policy": asdict(policy), "infeasible": infeasible, "excluded_count": len(excluded)},
    )


def assign_splits(records: list[dict[str, Any]], group_by_id: dict[str, str], config: BuilderConfig) -> dict[str, str]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[group_by_id[row["sprite_id"]]].append(row)
    assignments: dict[str, str] = {}
    source_holdouts = set(config.source_ood_packs)
    artist_holdouts = set(config.held_out_artists)
    open_objects = set(config.open_set_objects)
    for group_id, members in sorted(by_group.items()):
        packs = {row["source_pack"] for row in members}
        artists = {row["sub_artist"] for row in members}
        objects = {row["object_name"] for row in members}
        if packs & source_holdouts or artists & artist_holdouts:
            partition = "source_ood_test"
        elif objects and objects <= open_objects:
            partition = "open_set_test"
        else:
            point = int(_stable_key(config.seed, group_id)[:16], 16) / float(16**16)
            if point < config.train_fraction:
                partition = "train"
            elif point < config.train_fraction + config.validation_fraction:
                partition = "validation"
            else:
                partition = "test"
        for row in members:
            assignments[row["sprite_id"]] = partition
    train_objects = {row["object_name"] for row in records if assignments[row["sprite_id"]] == "train"}
    missing = {
        row["object_name"]
        for row in records
        if assignments[row["sprite_id"]] in {"validation", "test"} and row["object_name"] not in train_objects
    }
    for _group_id, members in by_group.items():
        if any(row["object_name"] in missing for row in members) and assignments[members[0]["sprite_id"]] in {
            "validation",
            "test",
        }:
            for row in members:
                assignments[row["sprite_id"]] = "train"
    return assignments


def validate_no_leakage(assignments: dict[str, str], relations: list[dict[str, Any]]) -> None:
    """Fail when a declared hard relation crosses immutable partitions."""

    crossings: list[str] = []
    for relation in relations:
        if not relation.get("hard_split_constraint"):
            continue
        splits = {assignments[member] for member in relation.get("members", []) if member in assignments}
        if len(splits) > 1:
            crossings.append(str(relation.get("relation_id") or relation.get("kind")))
    if crossings:
        raise ValueError(f"hard split leakage: {sorted(crossings)}")


def _export(
    output: Path,
    balanced: list[dict[str, Any]],
    all_valid: list[dict[str, Any]],
    assignments: dict[str, str],
    groups: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    discovered: dict[str, Any],
    config: BuilderConfig,
    config_hash: str,
) -> dict[str, Any]:
    blobs = output / "blobs"
    blobs.mkdir(parents=True)
    for row in all_valid:
        blob = blobs / f"{row['exported_rgba_hash']}.rgba"
        if not blob.exists():
            blob.write_bytes(np.ascontiguousarray(row["_rgba"], dtype=np.uint8).tobytes())
    balanced_ids = {row["sprite_id"] for row in balanced}
    dataset_rows = []
    for row in balanced:
        public = _public(row)
        public["split"] = assignments[row["sprite_id"]]
        public["blob_path"] = f"blobs/{row['exported_rgba_hash']}.rgba"
        dataset_rows.append(public)
    _write_jsonl(output / "dataset_manifest.jsonl", dataset_rows)
    split_manifest = {
        "partitions": {
            partition: sorted(sprite_id for sprite_id, value in assignments.items() if value == partition)
            for partition in PARTITIONS
        },
        "seed": config.seed,
        "fractions": {
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": config.test_fraction,
        },
        "source_ood_packs": list(config.source_ood_packs),
        "held_out_artists": list(config.held_out_artists),
        "open_set_objects": list(config.open_set_objects),
    }
    _write_json(output / "split_manifest.json", split_manifest)
    group_rows = list(groups) + relations
    _write_jsonl(output / "group_manifest.jsonl", group_rows)
    _write_jsonl(
        output / "excluded_manifest.jsonl",
        sorted(exclusions, key=lambda row: (row["sprite_id"], row["stage"], row["reason_code"])),
    )
    licenses: dict[str, dict[str, Any]] = {}
    for row in balanced:
        key = row["source_id"]
        licenses[key] = {
            "source_id": key,
            "license": row["license"],
            "license_url": row["license_url"],
            "source_url": row["source_url"],
            "attribution": row["attribution"],
        }
    _write_json(output / "license_manifest.json", {"sources": [licenses[key] for key in sorted(licenses)]})
    variants = output / "variants"
    variant_members = {
        "supervised_core": [row for row in all_valid if row.get("is_supervised")],
        "supervised_plus_unlabeled": list(all_valid),
        "source_balanced": balanced,
        "strict_quality": [row for row in all_valid if row.get("strict_quality")],
    }
    for name, members in variant_members.items():
        target = variants / name
        target.mkdir(parents=True)
        _write_jsonl(
            target / "dataset_manifest.jsonl",
            [
                {
                    "sprite_id": row["sprite_id"],
                    "blob_path": f"../../blobs/{row['exported_rgba_hash']}.rgba",
                    "supervision": "supervised" if row.get("is_supervised") else "unlabeled",
                }
                for row in sorted(members, key=lambda item: item["sprite_id"])
            ],
        )
    _write_loader_adapter(output, balanced, assignments)
    summary = {
        "dataset_name": config.dataset_name,
        "preview": True,
        "builder_version": BUILDER_VERSION,
        "config_hash": config_hash,
        "input_valid_count": len(all_valid),
        "exported_count": len(balanced),
        "excluded_count": len(exclusions),
        "split_counts": {
            partition: sum(value == partition for value in assignments.values()) for partition in PARTITIONS
        },
        "exclusions_by_reason": dict(sorted(Counter(row["reason_code"] for row in exclusions).items())),
        "distribution": distribution_report(balanced),
        "variant_counts": {name: len(members) for name, members in variant_members.items()},
        "source_manifest_hashes": discovered["source_manifest_hashes"],
    }
    _write_json(output / "dataset_summary.json", summary)
    (output / "README.md").write_text(_readme(config, summary), encoding="utf-8", newline="\n")
    return {
        "exported_count": len(balanced),
        "blob_count": len(list(blobs.glob("*.rgba"))),
        "balanced_ids": len(balanced_ids),
    }


def _write_loader_adapter(output: Path, records: list[dict[str, Any]], assignments: dict[str, str]) -> None:
    training_rows: list[dict[str, Any]] = []
    for public_split, npz_name in (("train", "train"), ("validation", "val"), ("test", "test")):
        members = [row for row in records if assignments[row["sprite_id"]] == public_split]
        members.sort(key=lambda row: row["sprite_id"])
        arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        for index, row in enumerate(members):
            source_arrays = row.get("_arrays")
            if source_arrays is None:
                source_arrays = _encode_rgba_adapter(row["_rgba"], row["sprite_id"], row["category"])
            for key in ("alpha", "index_map", "role_map", "palette", "palette_mask", "category_id", "sprite_id"):
                arrays[key].append(np.asarray(source_arrays[key]))
            original = dict(row.get("training_record") or {})
            label_quality = original.get("label_quality") or row.get("label_quality")
            training_row = {
                **original,
                "schema_version": str(original.get("schema_version") or "training_manifest_v1.0"),
                "sprite_id": row["sprite_id"],
                # Loader split names follow their NPZ names. The public dataset
                # partition remains ``validation`` in dataset_manifest.jsonl.
                "split": npz_name,
                "npz_file": f"{npz_name}.npz",
                "npz_row": index,
                "caption": str(original.get("caption") or row["object_name"]),
                "object_name": row["object_name"],
                "category": row["category"],
                "source_id": row.get("source_id", ""),
                "source_pack": row.get("source_pack", ""),
                "artist": row.get("author") or row.get("sub_artist") or "",
                "label_provenance": row.get("label_provenance", {}),
                "suitability_status": row.get("suitability_status", ""),
            }
            if isinstance(label_quality, dict):
                training_row["label_quality"] = label_quality
            training_rows.append(training_row)
        stacked = _stack_adapter_arrays(arrays)
        _write_deterministic_npz(output / f"{npz_name}.npz", stacked)
    _write_jsonl(
        output / "training_manifest.jsonl", sorted(training_rows, key=lambda row: (row["npz_file"], row["npz_row"]))
    )


def _stack_adapter_arrays(arrays: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    if arrays:
        return {key: np.stack(values) for key, values in arrays.items()}
    return {
        "alpha": np.zeros((0, 32, 32), dtype=np.uint8),
        "index_map": np.zeros((0, 32, 32), dtype=np.uint8),
        "role_map": np.zeros((0, 32, 32), dtype=np.uint8),
        "palette": np.zeros((0, 32, 3), dtype=np.uint8),
        "palette_mask": np.zeros((0, 32), dtype=bool),
        "category_id": np.zeros((0,), dtype=np.int64),
        "sprite_id": np.zeros((0,), dtype="<U1"),
    }


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def verify_dataset(
    output_dir: str | Path, *, v4_dir: str | Path | None = None, expected_v4_hashes: dict[str, str] | None = None
) -> dict[str, Any]:
    output = Path(output_dir)
    errors: list[str] = []
    rows = _read_jsonl(output / "dataset_manifest.jsonl")
    assignments = {row["sprite_id"]: row["split"] for row in rows}
    groups = _read_jsonl(output / "group_manifest.jsonl")
    leakage: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        if not group.get("hard_split_constraint") and "split_group_id" not in group:
            continue
        members = group.get("members", [])
        splits = {assignments[member] for member in members if member in assignments}
        if len(splits) > 1:
            kind = str(group.get("kind") or "final_split_group")
            leakage[kind].append(str(group.get("relation_id") or group.get("split_group_id")))
    for row in rows:
        blob = output / row["blob_path"]
        if not blob.is_file():
            errors.append(f"missing blob for {row['sprite_id']}")
            continue
        rgba = np.frombuffer(blob.read_bytes(), dtype=np.uint8).reshape(
            row["exported_height"], row["exported_width"], 4
        )
        if canonical_rgba_sha256(rgba) != row["exported_rgba_hash"]:
            errors.append(f"blob hash mismatch for {row['sprite_id']}")
        for field_name in REQUIRED_PROVENANCE:
            if row.get(field_name) is None or (isinstance(row.get(field_name), str) and not row[field_name].strip()):
                errors.append(f"missing {field_name} for {row['sprite_id']}")
    split_manifest = _read_json(output / "split_manifest.json")
    heldout = set(split_manifest.get("source_ood_packs", []))
    for row in rows:
        if row["source_pack"] in heldout and row["split"] != "source_ood_test":
            errors.append(f"source OOD leak: {row['sprite_id']}")
        if (
            row["source_pack"] not in heldout
            and row["split"] == "source_ood_test"
            and not split_manifest.get("held_out_artists")
        ):
            errors.append(f"unexpected source OOD member: {row['sprite_id']}")
    if leakage:
        errors.append("hard relation crossings detected")
    loader = _verify_loader_contract(output)
    errors.extend(loader["errors"])
    v4_unchanged = True
    if v4_dir is not None and expected_v4_hashes is not None:
        actual = {path.name: source_file_sha256(path) for path in Path(v4_dir).iterdir() if path.is_file()}
        v4_unchanged = actual == expected_v4_hashes
        if not v4_unchanged:
            errors.append("v4 input changed during build")
    return {
        "ok": not errors,
        "errors": errors,
        "leakage": dict(sorted(leakage.items())),
        "leakage_counts": {
            "exact_exported_rgba": len(leakage.get("exact_exported_rgba", [])),
            "alpha_mask": len(leakage.get("exact_alpha_mask_recolor", [])),
            "declared_variant": len(leakage.get("declared_variant_family", [])),
            "source_sheet": len(leakage.get("source_sheet_siblings", [])),
        },
        "v4_unchanged": v4_unchanged,
        "training_loader_contract": loader,
    }


def _verify_loader_contract(output: Path) -> dict[str, Any]:
    errors: list[str] = []
    rows = _read_jsonl(output / "training_manifest.jsonl")
    for name in ("train", "val", "test"):
        with np.load(output / f"{name}.npz", allow_pickle=False) as payload:
            missing = {"alpha", "index_map", "role_map", "palette", "palette_mask", "category_id", "sprite_id"} - set(
                payload.files
            )
            if missing:
                errors.append(f"{name}.npz missing {sorted(missing)}")
    for row in rows:
        if row.get("npz_file") not in {"train.npz", "val.npz", "test.npz"}:
            errors.append(f"bad npz_file for {row.get('sprite_id')}")
        expected_splits = {
            "train.npz": {"train"},
            # ``validation`` is the frozen v5-preview spelling. New adapters
            # emit ``val`` so the trainer consumes them without a silent empty
            # validation loader, while old immutable previews remain readable.
            "val.npz": {"val", "validation"},
            "test.npz": {"test"},
        }.get(row.get("npz_file"))
        if expected_splits is not None and row.get("split") not in expected_splits:
            errors.append(f"split/npz mismatch for {row.get('sprite_id')}")
        quality = row.get("label_quality")
        if isinstance(quality, dict) and str(quality.get("schema_version", "")).startswith("label_training_quality"):
            score = quality.get("record_uncertainty_1_20")
            if score is not None and not 1 <= int(score) <= 20:
                errors.append(f"bad record uncertainty for {row.get('sprite_id')}")
    return {"ok": not errors, "errors": errors, "row_count": len(rows)}


def _freeze(
    output: Path, discovered: dict[str, Any], config_hash: str, verification: dict[str, Any], timestamp: str | None
) -> dict[str, Any]:
    if (output / "FREEZE.json").exists():
        raise FileExistsError(f"refusing to overwrite frozen dataset: {output}")
    hashes = {name: source_file_sha256(output / name) for name in REQUIRED_EXPORTS}
    hashes.update(
        {
            name: source_file_sha256(output / name)
            for name in ("training_manifest.jsonl", "train.npz", "val.npz", "test.npz")
        }
    )
    split_hash = hashes["split_manifest.json"]
    content_hash = hashlib.sha256("".join(f"{name}:{hashes[name]}\n" for name in sorted(hashes)).encode()).hexdigest()
    freeze = {
        "content_manifest_hash": content_hash,
        "manifest_hashes": hashes,
        "config_hash": config_hash,
        "code_version": BUILDER_VERSION,
        "source_manifest_hashes": discovered["source_manifest_hashes"],
        "split_manifest_hash": split_hash,
        "label_manifest_hashes": {
            path: value
            for path, value in discovered["source_manifest_hashes"].items()
            if "label" in Path(path).name.lower() or Path(path).name.startswith("manifest_")
        },
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "timestamp_excluded_from_content_hash": True,
        "reproducibility_command": "$env:PYTHONPATH='src'; python -m spritelab.dataset_v5.cli build --config experiments/dataset_v5_builder/preview_config.json --v4 datasets/sprite_lab_multisource_v4 --harvest-root harvest_runs --output datasets/sprite_lab_multisource_v5_preview_rebuild",
        "verification_result": {
            "ok": verification["ok"],
            "leakage_counts": verification["leakage_counts"],
            "v4_unchanged": verification["v4_unchanged"],
        },
    }
    _write_json(output / "FREEZE.json", freeze)
    return freeze


def distribution_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)

    def dist(field_name: str) -> list[dict[str, Any]]:
        counts = Counter(str(row.get(field_name) or "unknown") for row in records)
        return [
            {"value": value, "count": count, "share": count / total if total else 0.0}
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    return {
        "total": total,
        "source": dist("source_id"),
        "pack": dist("source_pack"),
        "artist": dist("sub_artist"),
        "object": dist("object_name"),
    }


def _rgba_from_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    alpha = np.asarray(arrays["alpha"], dtype=np.uint8)
    index_map = np.asarray(arrays["index_map"], dtype=np.int64)
    palette = np.asarray(arrays["palette"])
    if palette.dtype.kind == "f":
        palette = np.rint(np.clip(palette, 0, 1) * 255)
    palette = np.asarray(palette, dtype=np.uint8)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    opaque = alpha > 0
    rgba[:, :, 3] = opaque.astype(np.uint8) * 255
    rgba[opaque, :3] = palette[index_map[opaque], :3]
    return rgba


def _encode_rgba_adapter(rgba: np.ndarray, sprite_id: str, category: str) -> dict[str, np.ndarray]:
    alpha = (rgba[:, :, 3] > 0).astype(np.uint8)
    colors = np.unique(rgba[alpha > 0, :3], axis=0)[:32]
    palette = np.zeros((32, 3), dtype=np.uint8)
    palette[: len(colors)] = colors
    index_map = np.zeros((32, 32), dtype=np.uint8)
    lookup = {tuple(color): index for index, color in enumerate(colors)}
    for y, x in zip(*np.nonzero(alpha), strict=True):
        index_map[y, x] = lookup[tuple(rgba[y, x, :3])]
    return {
        "alpha": alpha,
        "index_map": index_map,
        "role_map": np.zeros((32, 32), dtype=np.uint8),
        "palette": palette,
        "palette_mask": np.arange(32) < len(colors),
        "category_id": np.asarray(0, dtype=np.int64),
        "sprite_id": np.asarray(sprite_id),
    }


def _normalized_alpha_key(alpha: np.ndarray) -> str:
    mask = np.asarray(alpha) > 0
    ys, xs = np.nonzero(mask)
    if not len(xs):
        payload = b"empty"
    else:
        crop = np.ascontiguousarray(mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], dtype=np.uint8)
        payload = struct.pack(">II", crop.shape[1], crop.shape[0]) + crop.tobytes()
    return hashlib.sha256(b"translation-normalized-alpha-v1\0" + payload).hexdigest()


def _variant_values(row: dict[str, Any]) -> list[str]:
    fields = (
        "explicit_variant_group",
        "variant_group",
        "variant_group_id",
        "duplicate_group",
        "base_family",
        "base_sprite_id",
        "source_variant_group",
    )
    values: list[str] = []
    for field_name in fields:
        if row.get(field_name):
            values.append(f"{field_name}:{row[field_name]}")
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for field_name in fields:
        if provenance.get(field_name):
            values.append(f"provenance.{field_name}:{provenance[field_name]}")
    return sorted(set(values))


def _label_provenance(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("label_v4"), dict):
        return {
            "adapter": "label_v4",
            "schema_version": row["label_v4"].get("schema_version"),
            "risk_model_version": row["label_v4"].get("risk_model_version", "label_risk_v1"),
        }
    if str(row.get("schema_version", "")).startswith("label_record_v4"):
        return {
            "adapter": "label_v4",
            "schema_version": row.get("schema_version"),
            "risk_model_version": _nested_dict(row, "label_quality", "risk_model_version") or "label_risk_v1",
        }
    if isinstance(row.get("label_v3"), dict) and row["label_v3"].get("approved"):
        return {"adapter": "label_v3", "record": row["label_v3"]}
    if isinstance(row.get("label_v2"), dict) and row["label_v2"].get("applied"):
        return {
            "adapter": "legacy_label_v2",
            "bucket": row["label_v2"].get("bucket"),
            "quality": row["label_v2"].get("label_quality", {}),
        }
    if isinstance(row.get("semantic_v3"), dict):
        return {"adapter": "existing_semantic_manifest", "schema_version": row["semantic_v3"].get("schema_version")}
    return {}


def _strict_quality(row: dict[str, Any]) -> bool:
    v4_quality = row.get("label_quality") if isinstance(row.get("label_quality"), dict) else {}
    if v4_quality:
        critical = v4_quality.get("fields") if isinstance(v4_quality.get("fields"), dict) else {}
        critical_names = ("canonical_object", "category", "domain", "role")
        return bool(critical) and all(
            isinstance(critical.get(name), dict)
            and critical[name].get("calibration_state") == "calibrated"
            and 1 <= int(critical[name].get("uncertainty_1_20", 20)) <= 4
            and bool(critical[name].get("supervision_mask", True))
            for name in critical_names
        )
    label = row.get("label_v2") if isinstance(row.get("label_v2"), dict) else {}
    quality = label.get("label_quality") if isinstance(label.get("label_quality"), dict) else {}
    return bool(label.get("applied") and not quality.get("needs_review", True))


def _label_quality(manifest: dict[str, Any], training_record: dict[str, Any] | None) -> dict[str, Any] | None:
    direct = manifest.get("label_quality")
    if isinstance(direct, dict):
        return dict(direct)
    label_v4 = manifest.get("label_v4")
    if isinstance(label_v4, dict) and isinstance(label_v4.get("label_quality"), dict):
        return dict(label_v4["label_quality"])
    if isinstance(training_record, dict) and isinstance(training_record.get("label_quality"), dict):
        return dict(training_record["label_quality"])
    return None


def _nested_dict(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_") and key not in {"training_record"}}


def _exclude(
    sprite_id: str, reason: str, stage: str, *, representative: str | None = None, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {"sprite_id": sprite_id, "reason_code": reason, "stage": stage}
    if representative:
        row["representative_sprite_id"] = representative
    if details:
        row["details"] = details
    return row


def _strict_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_strict_json(value).encode("utf-8")).hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _norm(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").lower()


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return _read_jsonl(path)
    except (json.JSONDecodeError, TypeError):
        return []


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_strict_json(value) + "\n" for value in values)
    path.write_text(text, encoding="utf-8", newline="\n")


def _stage(output: Path, stage: str, name: str, value: Any) -> None:
    _write_json(output / "audit" / stage / name, value)


def _readme(config: BuilderConfig, summary: dict[str, Any]) -> str:
    return f"""# {config.dataset_name}

This is an immutable preview assembled by `{BUILDER_VERSION}`. It is not the final production v5.

- Exported sprites: {summary["exported_count"]}
- Excluded records: {summary["excluded_count"]}
- Canonical identity: versioned dimensions plus decoded exported RGBA bytes
- Public partitions: {", ".join(PARTITIONS)}
- Loader adapter: `training_manifest.jsonl` with `train.npz`, `val.npz`, and `test.npz`

Rebuild with the command recorded in `FREEZE.json`. Never edit a frozen directory in place.
"""
