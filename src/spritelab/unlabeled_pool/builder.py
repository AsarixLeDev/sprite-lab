"""Deterministic assembly of an immutable, strictly unlabeled sprite pool."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.harvest.suitability import SuitabilityInput, audit_sprite, load_config
from spritelab.unlabeled_pool.provenance_repair import apply_provenance_repair, load_provenance_repairs

BUILDER_VERSION = "unlabeled_candidate_pool_builder_v1"
ACQUISITION_POLICY_VERSION = "unlabeled_acquisition_policy_v1"
RGBA_MARKER = b"spritelab-exported-rgba-v1\0"
ALPHA_MARKER = b"spritelab-alpha-mask-v1\0"
STATUSES = {
    "ready_for_labeling",
    "quarantine_quality",
    "blocked_provenance",
    "duplicate_representative",
    "duplicate_variant",
    "excluded",
}
SUPERVISED_FIELDS = {
    "category",
    "object_name",
    "label",
    "labels",
    "split",
    "partition",
    "train",
    "validation",
    "test",
    "semantic_v3",
    "label_v2",
    "label_v3",
    "vlm_descriptor",
    "safe_prefill",
}
REQUIRED_PROVENANCE = (
    "source_id",
    "pack_id",
    "author",
    "license",
    "attribution",
    "source_url",
    "downloaded_file_hash",
    "archive_member",
    "source_image",
    "cell_coordinates",
    "native_dimensions",
    "resize_policy",
    "exported_rgba_hash",
    "alpha_mask_hash",
    "suitability_status",
    "variant_geometry_group",
    "acquisition_policy_version",
)
EXPORT_FILES = (
    "candidate_manifest.jsonl",
    "group_manifest.jsonl",
    "annotation_queue.jsonl",
    "quarantine_manifest.jsonl",
    "excluded_manifest.jsonl",
    "license_manifest.json",
    "summary.json",
    "README.md",
)


@dataclass(frozen=True)
class PoolConfig:
    pool_name: str = "sprite_lab_unlabeled_pool_v1"
    suitability_profile: str = "single_object_32px"
    source_priority: tuple[str, ...] = (
        "sota_v2_shade_weapons_validated",
        "sota_v2_farming_tools_final3",
    )
    deprioritized_types: tuple[str, ...] = ("potion", "food")

    def canonical(self) -> dict[str, Any]:
        return asdict(self)


def strict_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_json(value: Any) -> str:
    return hashlib.sha256(strict_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rgba_sha256(rgba: np.ndarray) -> str:
    value = np.ascontiguousarray(rgba, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 4:
        raise ValueError(f"RGBA must have shape [H,W,4], got {value.shape}")
    height, width = value.shape[:2]
    return hashlib.sha256(RGBA_MARKER + struct.pack(">II", width, height) + value.tobytes()).hexdigest()


def alpha_mask_sha256(alpha: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(alpha) > 0, dtype=np.uint8)
    height, width = value.shape
    return hashlib.sha256(ALPHA_MARKER + struct.pack(">II", width, height) + value.tobytes()).hexdigest()


def normalized_alpha_sha256(alpha: np.ndarray) -> str:
    value = np.asarray(alpha) > 0
    ys, xs = np.where(value)
    if not len(xs):
        return hashlib.sha256(b"empty-alpha").hexdigest()
    cropped = np.ascontiguousarray(value[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], dtype=np.uint8)
    return hashlib.sha256(struct.pack(">II", cropped.shape[1], cropped.shape[0]) + cropped.tobytes()).hexdigest()


def build_pool(
    *,
    harvest_root: str | Path,
    output_dir: str | Path,
    reports_dir: str | Path | None = None,
    config: PoolConfig | None = None,
    provenance_repairs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Run discover through immutable freeze, then verify the result."""

    config = config or PoolConfig()
    harvest_root = Path(harvest_root).resolve()
    output = Path(output_dir).resolve()
    reports = Path(reports_dir).resolve() if reports_dir else None
    repair_paths = tuple(provenance_repairs)
    if (output / "freeze_manifest.json").is_file():
        raise FileExistsError(f"refusing to overwrite frozen pool: {output}")
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty pool: {output}")
        output.rmdir()
    output.mkdir(parents=True)
    config_hash = hash_json(config.canonical())

    inventory, occurrences, source_hashes = discover(harvest_root, provenance_repairs=repair_paths)
    selected, superseded = collapse_occurrences(occurrences, config)
    source_total = len(occurrences)
    provenance_ready, provenance_blocked = validate_provenance(selected)
    suitable, quality_quarantine, quality_rejected = load_suitability(provenance_ready, config)
    retained = suitable + quality_quarantine
    candidate_rows, group_rows, group_stats = group_variants(retained, config)
    queue_rows = annotation_queues(candidate_rows, provenance_blocked)

    excluded = []
    excluded.extend(superseded)
    excluded.extend(quality_rejected)
    excluded.extend(row for row in selected if row.get("_flare_excluded"))
    blocked_rows = [_public_blocked(row) for row in provenance_blocked]
    excluded.extend(blocked_rows)
    excluded = _unique_exclusions(excluded)
    _export_blobs(output, candidate_rows)
    _write_jsonl(output / "candidate_manifest.jsonl", [_public_candidate(row) for row in candidate_rows])
    _write_jsonl(output / "group_manifest.jsonl", group_rows)
    _write_jsonl(output / "annotation_queue.jsonl", queue_rows)
    _write_jsonl(
        output / "quarantine_manifest.jsonl",
        [_public_candidate(row) for row in candidate_rows if row["acquisition_status"] == "quarantine_quality"]
        + blocked_rows,
    )
    _write_jsonl(output / "excluded_manifest.jsonl", excluded)
    _write_json(output / "license_manifest.json", license_manifest(candidate_rows))
    summary = build_summary(
        inventory=inventory,
        source_total=source_total,
        selected=selected,
        candidates=candidate_rows,
        blocked=provenance_blocked,
        rejected=quality_rejected,
        queues=queue_rows,
        group_stats=group_stats,
        config_hash=config_hash,
        source_hashes=source_hashes,
    )
    _write_json(output / "summary.json", summary)
    (output / "README.md").write_text(_readme(summary), encoding="utf-8", newline="\n")
    verification = verify_pool(output, require_frozen=False)
    if not verification["ok"]:
        raise ValueError(f"pool verification failed: {verification['errors']}")
    freeze = freeze_pool(
        output,
        source_hashes,
        config_hash,
        rebuild_command=_rebuild_command(repair_paths, workspace_root=harvest_root.parent),
    )
    verification = verify_pool(output)
    if not verification["ok"]:
        raise ValueError(f"frozen pool verification failed: {verification['errors']}")
    if reports is not None:
        write_reports(reports, inventory, summary, candidate_rows, provenance_blocked, group_rows, freeze)
    return {**verification, "summary": summary, "freeze": freeze}


def discover(
    harvest_root: Path, *, provenance_repairs: Iterable[str | Path] = ()
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    if not harvest_root.is_dir():
        raise FileNotFoundError(harvest_root)
    inventory: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    repair_index, repair_hashes = load_provenance_repairs(provenance_repairs, workspace_root=harvest_root.parent)
    source_hashes.update(repair_hashes)
    for run in sorted(path for path in harvest_root.iterdir() if path.is_dir()):
        sources_path = run / "sources.jsonl"
        imported_path = run / "imported.jsonl"
        candidates_path = run / "candidates.jsonl"
        for path in (sources_path, imported_path, candidates_path):
            if path.is_file():
                source_hashes[path.relative_to(harvest_root.parent).as_posix()] = file_sha256(path)
        for path in sorted((run / "downloads").glob("*")) if (run / "downloads").is_dir() else ():
            if path.is_file():
                source_hashes[path.relative_to(harvest_root.parent).as_posix()] = file_sha256(path)
        sources = {str(row.get("source_id") or ""): row for row in _safe_jsonl(sources_path)}
        candidates = {
            (str(row.get("source_id") or ""), _norm(str(row.get("relative_path") or ""))): row
            for row in _safe_jsonl(candidates_path)
        }
        imported = _safe_jsonl(imported_path)
        accepted = [row for row in imported if row.get("status") == "accepted"]
        flare = "flare" in run.name.lower() or any("flare" in key.lower() for key in sources)
        inventory.append(
            {
                "run_id": run.name,
                "source_ids": sorted(sources),
                "imported_count": len(imported),
                "accepted_count": len(accepted),
                "flare_rejected_pack": flare,
                "has_sources": bool(sources),
                "has_candidates": bool(candidates),
            }
        )
        for raw in accepted:
            source_id = str(raw.get("source_id") or "")
            source = sources.get(source_id, {})
            candidate = candidates.get((source_id, _norm(str(raw.get("relative_path") or ""))), {})
            occurrence = _adapt_occurrence(run, raw, source, candidate, flare)
            occurrences.append(apply_provenance_repair(occurrence, repair_index))
    return inventory, occurrences, dict(sorted(source_hashes.items()))


def _adapt_occurrence(
    run: Path, imported: dict[str, Any], source: dict[str, Any], candidate: dict[str, Any], flare: bool
) -> dict[str, Any]:
    metadata = imported.get("auto_metadata") if isinstance(imported.get("auto_metadata"), dict) else {}
    mapping = metadata.get("sheet_mapping") if isinstance(metadata.get("sheet_mapping"), dict) else {}
    license_data = source.get("license") if isinstance(source.get("license"), dict) else {}
    author = str(imported.get("author") or source.get("author") or "").strip()
    source_id = str(imported.get("source_id") or source.get("source_id") or "").strip()
    relative = str(imported.get("relative_path") or candidate.get("relative_path") or "").replace("\\", "/")
    source_sheet = str(mapping.get("source_sheet") or relative).replace("\\", "/")
    coord = mapping.get("sheet_coordinate") or _coordinate_from_path(str(imported.get("final_png_path") or ""))
    native = _native_dimensions(mapping, imported)
    final_path = _resolve_image_path(run, str(imported.get("final_png_path") or ""))
    downloaded_hash = str(source.get("download_sha256") or source.get("sha256") or "").lower()
    if not downloaded_hash:
        downloaded_hash = _recover_download_hash(run, source_id, str(source.get("original_filename") or ""))
    source_url = str(source.get("source_url") or source.get("download_url") or candidate.get("source_path") or "")
    attribution = str(source.get("attribution") or author).strip()
    return {
        "sprite_id": str(imported.get("sprite_id") or ""),
        "source_id": source_id,
        "pack_id": source_id,
        "pack_name": str(source.get("source_name") or imported.get("source_name") or source_id),
        "author": author,
        "sub_artist": str(imported.get("sub_artist") or author).strip(),
        "license": str(imported.get("license") or license_data.get("license") or "").lower(),
        "license_url": str(license_data.get("license_url") or ""),
        "license_confirmed": bool(license_data.get("user_confirmed")),
        "attribution": attribution,
        "source_url": source_url,
        "downloaded_file_hash": downloaded_hash,
        "archive_member": source_sheet,
        "source_image": source_sheet,
        "cell_coordinates": str(coord or "whole_image"),
        "native_dimensions": native,
        "resize_policy": _resize_policy(native, imported),
        "declared_variant_group": str(mapping.get("variant_group_id") or ""),
        "declared_material": str(mapping.get("material") or ""),
        "source_sheet": source_sheet,
        "source_run": run.name,
        "source_runs": [run.name],
        "broad_pack_type": _broad_pack_type(source_id, run.name, source_sheet),
        "image_path": str(final_path),
        "acquisition_policy_version": ACQUISITION_POLICY_VERSION,
        "_flare_excluded": flare,
        "_source_priority": 0,
    }


def collapse_occurrences(
    occurrences: list[dict[str, Any]], config: PoolConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sprite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        by_sprite[row["sprite_id"]].append(row)
    selected: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    priorities = {name: index for index, name in enumerate(config.source_priority)}
    for sprite_id, members in sorted(by_sprite.items()):
        members.sort(
            key=lambda row: (
                row["_flare_excluded"],
                priorities.get(row["source_run"], len(priorities)),
                -_run_revision(row["source_run"]),
                row["source_run"],
            )
        )
        chosen = dict(members[0])
        chosen["source_runs"] = sorted({row["source_run"] for row in members})
        chosen["_source_priority"] = priorities.get(chosen["source_run"], len(priorities))
        selected.append(chosen)
        for row in members[1:]:
            superseded.append(
                {
                    "sprite_id": sprite_id,
                    "source_run": row["source_run"],
                    "acquisition_status": "excluded",
                    "reason_code": "superseded_harvest_occurrence",
                    "selected_source_run": chosen["source_run"],
                }
            )
    return selected, superseded


def validate_provenance(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ready: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    licensing = ("license", "attribution", "source_url", "downloaded_file_hash")
    for row in records:
        if row.get("_flare_excluded"):
            continue
        missing = [
            name
            for name in (
                "sprite_id",
                "source_id",
                "pack_id",
                "author",
                "attribution",
                "source_url",
                "downloaded_file_hash",
                "archive_member",
                "source_image",
                "cell_coordinates",
                "native_dimensions",
                "resize_policy",
            )
            if not row.get(name)
        ]
        if row.get("license") in {"", "unknown", "none"}:
            missing.append("license")
        if not row.get("license_confirmed") and not row.get("license_url"):
            missing.append("license_confirmation_or_url")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("downloaded_file_hash") or "")):
            missing.append("downloaded_file_hash_valid_sha256")
        image_value = str(row.get("image_path") or "")
        image = Path(image_value) if image_value else None
        if image is None or not image.is_file():
            missing.append("exported_image_file")
        if missing:
            blocked_row = dict(row)
            blocked_row["missing_provenance_fields"] = sorted(set(missing))
            blocked_row["license_failure"] = bool(set(missing) & set(licensing)) or "license" in missing
            blocked.append(blocked_row)
        else:
            ready.append(row)
    return ready, blocked


def load_suitability(
    records: list[dict[str, Any]], config: PoolConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    suitability_config = load_config(config.suitability_profile)
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: item["sprite_id"]):
        result = audit_sprite(
            SuitabilityInput(row["sprite_id"], Path(row["image_path"]), source_run=row["source_run"]),
            suitability_config,
        ).to_dict()
        enriched = dict(row)
        enriched["suitability_status"] = result["status"]
        enriched["suitability_score"] = result["score"]
        enriched["suitability_reason_codes"] = result["reason_codes"]
        enriched["suitability_config_hash"] = result["config_hash"]
        enriched["quality_confidence"] = round(float(result["score"]), 6)
        if result["status"] == "reject":
            rejected.append(
                {
                    "sprite_id": row["sprite_id"],
                    "source_id": row["source_id"],
                    "source_run": row["source_run"],
                    "acquisition_status": "excluded",
                    "reason_code": "suitability_reject",
                    "suitability_reason_codes": result["reason_codes"],
                }
            )
            continue
        with Image.open(row["image_path"]) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        enriched["exported_width"] = int(rgba.shape[1])
        enriched["exported_height"] = int(rgba.shape[0])
        enriched["exported_rgba_hash"] = canonical_rgba_sha256(rgba)
        enriched["alpha_mask_hash"] = alpha_mask_sha256(rgba[..., 3])
        enriched["normalized_alpha_hash"] = normalized_alpha_sha256(rgba[..., 3])
        enriched["_rgba"] = rgba
        (accepted if result["status"] == "accept" else quarantined).append(enriched)
    return accepted, quarantined, rejected


class _DSU:
    def __init__(self, values: Iterable[str]) -> None:
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


def group_variants(
    records: list[dict[str, Any]], config: PoolConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["sprite_id"]: row for row in records}
    relations: list[dict[str, Any]] = []
    geometry_dsu = _DSU(by_id)

    def add_groups(kind: str, field: str, *, geometry: bool) -> None:
        values: dict[str, list[str]] = defaultdict(list)
        for row in records:
            if row.get(field):
                values[str(row[field])].append(row["sprite_id"])
        for key, members in sorted(values.items()):
            members = sorted(set(members))
            if len(members) < 2:
                continue
            group_id = f"{kind}__{hashlib.sha256((kind + ':' + key).encode()).hexdigest()[:20]}"
            relations.append({"group_id": group_id, "group_kind": kind, "group_key": key, "members": members})
            if geometry:
                for member in members[1:]:
                    geometry_dsu.union(members[0], member)

    add_groups("exact_duplicate", "exported_rgba_hash", geometry=True)
    add_groups("alpha_mask_recolor", "alpha_mask_hash", geometry=True)
    add_groups("translation_padding_variant", "normalized_alpha_hash", geometry=True)
    add_groups("declared_material_variant", "declared_variant_group", geometry=True)
    add_groups("source_sheet_siblings", "source_sheet", geometry=False)
    families: dict[str, list[str]] = defaultdict(list)
    for sprite_id in sorted(by_id):
        families[geometry_dsu.find(sprite_id)].append(sprite_id)
    geometry_rows: list[dict[str, Any]] = []
    representatives: dict[str, str] = {}
    for members in sorted(families.values(), key=lambda values: values[0]):
        members = sorted(members)
        family_hash = hashlib.sha256("\n".join(members).encode()).hexdigest()[:20]
        family_id = f"geometry__{family_hash}"
        representative = choose_representative([by_id[value] for value in members], config)
        representatives[family_id] = representative["sprite_id"]
        geometry_rows.append(
            {
                "group_id": family_id,
                "group_kind": "geometry_family",
                "members": members,
                "representative_sprite_id": representative["sprite_id"],
                "variant_count": len(members),
            }
        )
        for sprite_id in members:
            row = by_id[sprite_id]
            row["variant_geometry_group"] = family_id
            row["annotation_representative"] = sprite_id == representative["sprite_id"]
            if row["suitability_status"] == "quarantine":
                row["acquisition_status"] = "quarantine_quality"
            elif len(members) == 1:
                row["acquisition_status"] = "ready_for_labeling"
            elif row["annotation_representative"]:
                row["acquisition_status"] = "duplicate_representative"
            else:
                row["acquisition_status"] = "duplicate_variant"
    counts_type = Counter(row["broad_pack_type"] for row in records)
    counts_source = Counter(row["source_id"] for row in records)
    counts_artist = Counter(row["sub_artist"] for row in records)
    family_sizes = {row["group_id"]: len(row["members"]) for row in geometry_rows}
    for row in records:
        components = {
            "unique_geometry": 30 if family_sizes[row["variant_geometry_group"]] == 1 else 15,
            "underrepresented_source_artist": _rarity_points(counts_source[row["source_id"]], len(records), 12)
            + _rarity_points(counts_artist[row["sub_artist"]], len(records), 10),
            "underrepresented_broad_pack_type": _rarity_points(counts_type[row["broad_pack_type"]], len(records), 12),
            "quality_confidence": round(10 * row["quality_confidence"], 6),
            "provenance_completeness": 10,
            "nonduplicate_status": 8 if row["acquisition_status"] == "ready_for_labeling" else 4,
            "potential_taxonomy_expansion": 8 if row["broad_pack_type"] not in config.deprioritized_types else 0,
            "variant_propagation_value": min(10, family_sizes[row["variant_geometry_group"]] - 1),
        }
        row["annotation_priority_components"] = components
        row["annotation_priority_score"] = round(sum(components.values()), 6)
    recolor_groups = [row for row in relations if row["group_kind"] == "alpha_mask_recolor"]
    return (
        sorted(records, key=lambda row: row["sprite_id"]),
        sorted(relations + geometry_rows, key=lambda row: (row["group_kind"], row["group_id"])),
        {
            "unique_geometry_families": len(geometry_rows),
            "recolor_families": len(recolor_groups),
            "annotation_representatives": len(geometry_rows),
            "expected_label_propagation_savings": sum(max(0, len(row["members"]) - 1) for row in geometry_rows),
        },
    )


def choose_representative(records: list[dict[str, Any]], config: PoolConfig) -> dict[str, Any]:
    priorities = {name: index for index, name in enumerate(config.source_priority)}
    return min(
        records,
        key=lambda row: (
            row.get("suitability_status") != "accept",
            -float(row.get("quality_confidence") or 0),
            priorities.get(row.get("source_run", ""), len(priorities)),
            row["sprite_id"],
        ),
    )


def annotation_queues(candidates: list[dict[str, Any]], blocked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidates:
        if row["acquisition_status"] == "quarantine_quality":
            queue = "quality_quarantine"
        elif row["annotation_representative"] and row["acquisition_status"] == "duplicate_representative":
            queue = "variant_representatives"
        elif row["annotation_representative"]:
            queue = "high_priority_unique_geometry"
        else:
            continue
        rows.append(
            {
                "queue": queue,
                "sprite_id": row["sprite_id"],
                "variant_geometry_group": row["variant_geometry_group"],
                "annotation_priority_score": row["annotation_priority_score"],
                "annotation_priority_components": row["annotation_priority_components"],
                "acquisition_status": row["acquisition_status"],
            }
        )
    for row in blocked:
        rows.append(
            {
                "queue": "provenance_blocked",
                "sprite_id": row["sprite_id"],
                "source_run": row["source_run"],
                "missing_provenance_fields": row["missing_provenance_fields"],
                "acquisition_status": "blocked_provenance",
            }
        )
    order = {
        name: index
        for index, name in enumerate(
            ("high_priority_unique_geometry", "variant_representatives", "quality_quarantine", "provenance_blocked")
        )
    }
    return sorted(
        rows, key=lambda row: (order[row["queue"]], -float(row.get("annotation_priority_score", 0)), row["sprite_id"])
    )


def license_manifest(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for row in candidates:
        sources[row["source_id"]] = {
            "source_id": row["source_id"],
            "pack_id": row["pack_id"],
            "author": row["author"],
            "license": row["license"],
            "license_url": row["license_url"],
            "attribution": row["attribution"],
            "source_url": row["source_url"],
            "downloaded_file_hash": row["downloaded_file_hash"],
        }
    return {"sources": [sources[key] for key in sorted(sources)]}


def build_summary(
    *,
    inventory: list[dict[str, Any]],
    source_total: int,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    queues: list[dict[str, Any]],
    group_stats: dict[str, Any],
    config_hash: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    suitability = Counter(row.get("suitability_status") for row in candidates)
    suitability["reject"] = len(rejected)
    representatives = [row for row in candidates if row.get("annotation_representative")]
    before = _distribution(candidates)
    after = _distribution(representatives)
    exact_unique = len({row["exported_rgba_hash"] for row in candidates})
    return {
        "pool_name": "sprite_lab_unlabeled_pool_v1",
        "builder_version": BUILDER_VERSION,
        "acquisition_policy_version": ACQUISITION_POLICY_VERSION,
        "config_hash": config_hash,
        "total_harvest_runs": len(inventory),
        "total_source_sprites": source_total,
        "distinct_discovered_sprite_ids": len(selected),
        "retained_candidates": len(candidates),
        "suitability": {name: suitability.get(name, 0) for name in ("accept", "quarantine", "reject")},
        "exact_unique_images": exact_unique,
        **group_stats,
        "queue_sizes": dict(sorted(Counter(row["queue"] for row in queues).items())),
        "candidate_status_counts": dict(sorted(Counter(row["acquisition_status"] for row in candidates).items())),
        "distribution": before,
        "representative_distribution": after,
        "pack_dominance": {
            "before_representative_selection": _dominance(before["pack"]),
            "after_representative_selection": _dominance(after["pack"]),
        },
        "license_distribution": _counter_rows(Counter(row["license"] for row in candidates), len(candidates)),
        "missing_provenance_count": len(blocked),
        "missing_provenance_fields": dict(
            sorted(Counter(field for row in blocked for field in row["missing_provenance_fields"]).items())
        ),
        "flare_retained_count": sum("flare" in row["source_run"].lower() for row in candidates),
        "source_manifest_hashes": source_hashes,
        "semantic_labels_present": False,
        "supervised_partitions_present": False,
    }


def freeze_pool(
    output: Path,
    source_hashes: dict[str, str],
    config_hash: str,
    *,
    rebuild_command: str | None = None,
) -> dict[str, Any]:
    freeze_path = output / "freeze_manifest.json"
    if freeze_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen pool: {output}")
    artifact_hashes = {name: file_sha256(output / name) for name in EXPORT_FILES}
    blob_hashes = {
        path.relative_to(output).as_posix(): file_sha256(path) for path in sorted((output / "blobs").glob("*.rgba"))
    }
    content_hash = hashlib.sha256(
        "".join(f"{name}:{digest}\n" for name, digest in sorted(artifact_hashes.items())).encode()
        + "".join(f"{name}:{digest}\n" for name, digest in sorted(blob_hashes.items())).encode()
    ).hexdigest()
    freeze = {
        "schema_version": "unlabeled_pool_freeze_v1",
        "builder_version": BUILDER_VERSION,
        "acquisition_policy_version": ACQUISITION_POLICY_VERSION,
        "config_hash": config_hash,
        "content_manifest_hash": content_hash,
        "artifact_hashes": artifact_hashes,
        "blob_hashes": blob_hashes,
        "source_manifest_hashes": source_hashes,
        "rebuild_command": rebuild_command
        or "$env:PYTHONPATH='src'; python -m spritelab.unlabeled_pool build --harvest-root harvest_runs --output datasets/sprite_lab_unlabeled_pool_v1_rebuild --reports experiments/unlabeled_candidate_pool_v1_rebuild",
    }
    _write_json(freeze_path, freeze)
    return freeze


def verify_pool(output_dir: str | Path, *, require_frozen: bool = True) -> dict[str, Any]:
    output = Path(output_dir)
    errors: list[str] = []
    for name in EXPORT_FILES:
        if not (output / name).is_file():
            errors.append(f"missing export: {name}")
    candidates = _safe_jsonl(output / "candidate_manifest.jsonl")
    groups = _safe_jsonl(output / "group_manifest.jsonl")
    queues = _safe_jsonl(output / "annotation_queue.jsonl")
    for row in candidates:
        missing = [name for name in REQUIRED_PROVENANCE if row.get(name) is None or row.get(name) == ""]
        if missing:
            errors.append(f"missing provenance for {row.get('sprite_id')}: {missing}")
        forbidden = sorted(SUPERVISED_FIELDS & set(row))
        if forbidden:
            errors.append(f"supervised fields for {row.get('sprite_id')}: {forbidden}")
        if row.get("acquisition_status") not in STATUSES:
            errors.append(f"invalid status for {row.get('sprite_id')}")
        if "flare" in str(row.get("source_run", "")).lower() or "flare" in str(row.get("source_id", "")).lower():
            errors.append(f"Flare candidate retained: {row.get('sprite_id')}")
        blob = output / str(row.get("blob_path") or "")
        if not blob.is_file():
            errors.append(f"missing blob for {row.get('sprite_id')}")
        else:
            expected_size = int(row.get("exported_width", 0)) * int(row.get("exported_height", 0)) * 4
            payload = blob.read_bytes()
            if len(payload) != expected_size:
                errors.append(f"bad blob size for {row.get('sprite_id')}")
            elif canonical_rgba_sha256(
                np.frombuffer(payload, dtype=np.uint8).reshape(row["exported_height"], row["exported_width"], 4)
            ) != row.get("exported_rgba_hash"):
                errors.append(f"blob hash mismatch for {row.get('sprite_id')}")
    geometry = [row for row in groups if row.get("group_kind") == "geometry_family"]
    candidate_ids = {row["sprite_id"] for row in candidates}
    represented = {row.get("representative_sprite_id") for row in geometry}
    if represented - candidate_ids:
        errors.append("geometry representative missing from candidate manifest")
    queue_ids = {
        row["sprite_id"]
        for row in queues
        if row["queue"] in {"high_priority_unique_geometry", "variant_representatives", "quality_quarantine"}
    }
    if represented - queue_ids:
        errors.append("geometry representative missing from annotation queues")
    freeze_path = output / "freeze_manifest.json"
    freeze_ok = False
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        actual = {name: file_sha256(output / name) for name in EXPORT_FILES if (output / name).is_file()}
        if actual != freeze.get("artifact_hashes"):
            errors.append("frozen artifact hash mismatch")
        actual_blobs = {
            path.relative_to(output).as_posix(): file_sha256(path) for path in sorted((output / "blobs").glob("*.rgba"))
        }
        if actual_blobs != freeze.get("blob_hashes"):
            errors.append("frozen blob hash mismatch")
        freeze_ok = not errors
    elif require_frozen:
        errors.append("missing freeze_manifest.json")
    return {
        "ok": not errors,
        "errors": errors,
        "candidate_count": len(candidates),
        "geometry_family_count": len(geometry),
        "frozen": freeze_ok,
        "v5_blob_store_compatible": all(str(row.get("blob_path", "")).startswith("blobs/") for row in candidates),
    }


def write_reports(
    reports: Path,
    inventory: list[dict[str, Any]],
    summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    freeze: dict[str, Any],
) -> None:
    if reports.exists() and any(reports.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty reports directory: {reports}")
    reports.mkdir(parents=True, exist_ok=True)
    inventory_lines = [
        "# Source inventory",
        "",
        "| Run | Imported | Accepted | Sources | Flare excluded |",
        "|---|---:|---:|---|---|",
    ]
    inventory_lines.extend(
        f"| `{row['run_id']}` | {row['imported_count']} | {row['accepted_count']} | {', '.join(row['source_ids']) or 'none'} | {'yes' if row['flare_rejected_pack'] else 'no'} |"
        for row in inventory
    )
    _write_md(reports / "source_inventory.md", inventory_lines)
    _write_md(
        reports / "inclusion_policy.md",
        [
            "# Inclusion policy",
            "",
            f"Policy: `{ACQUISITION_POLICY_VERSION}`.",
            "",
            "Accepted harvest outputs are admitted only with complete licensing and acquisition provenance. Suitability rejects and all Flare runs are excluded. Quality warnings are quarantined. No semantic field is copied into this pool.",
        ],
    )
    _write_md(
        reports / "annotation_priority_policy.md",
        [
            "# Annotation priority policy",
            "",
            "Scores are deterministic sums of geometry uniqueness, source/artist/type rarity, suitability quality, provenance completeness, nonduplicate state, taxonomy-expansion potential, and variant-propagation value. VLM confidence and semantic prefills are not inputs.",
        ],
    )
    shade = [row for row in candidates if row["author"].lower() == "shade"]
    shade_geometry = {row["variant_geometry_group"] for row in shade}
    _write_md(
        reports / "shade_variant_report.md",
        [
            "# Shade variant report",
            "",
            f"- Retained variants: {len(shade)}",
            f"- Geometry families: {len(shade_geometry)}",
            f"- Annotation representatives: {sum(row['annotation_representative'] for row in shade)}",
            "- Family semantics: deliberately unresolved; no weapon-family labels were invented.",
            "- Material recolors remain linked through declared sheet-coordinate variant groups.",
        ],
    )
    _write_md(reports / "pool_summary.md", _summary_markdown(summary))
    failure_counts = Counter(field for row in blocked for field in row["missing_provenance_fields"])
    _write_md(
        reports / "provenance_failures.md",
        ["# Provenance failures", "", f"Blocked records: {len(blocked)}", ""]
        + [f"- `{field}`: {count}" for field, count in sorted(failure_counts.items())]
        + ["", "Licensing fields fail closed; blocked records have no blob in the pool."],
    )
    geometry = [row for row in groups if row["group_kind"] == "geometry_family"]
    _write_md(
        reports / "reproducibility_report.md",
        [
            "# Reproducibility report",
            "",
            f"- Content manifest hash: `{freeze['content_manifest_hash']}`",
            f"- Artifact count: {len(freeze['artifact_hashes'])}",
            f"- Blob count: {len(freeze['blob_hashes'])}",
            f"- Geometry families: {len(geometry)}",
            "- Freeze content excludes timestamps and uses canonical JSON plus deterministic raw-RGBA blobs.",
            f"- Rebuild: `{freeze['rebuild_command']}`",
        ],
    )
    (reports / "command_log.txt").write_text(
        freeze["rebuild_command"]
        + "\n$env:PYTHONPATH='src'; python -m spritelab.unlabeled_pool verify --pool datasets/sprite_lab_unlabeled_pool_v1_rebuild\n",
        encoding="utf-8",
        newline="\n",
    )


def _public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    allowed_private = set()
    public = {key: value for key, value in row.items() if not key.startswith("_") and key not in SUPERVISED_FIELDS}
    del allowed_private
    public.pop("image_path", None)
    public["blob_path"] = f"blobs/{row['exported_rgba_hash']}.rgba"
    return public


def _public_blocked(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprite_id": row["sprite_id"],
        "source_id": row["source_id"],
        "pack_id": row["pack_id"],
        "source_run": row["source_run"],
        "acquisition_status": "blocked_provenance",
        "missing_provenance_fields": row["missing_provenance_fields"],
        "license_failure": row["license_failure"],
    }


def _unique_exclusions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        public = {key: value for key, value in row.items() if not key.startswith("_") and key not in SUPERVISED_FIELDS}
        public.pop("image_path", None)
        if "acquisition_status" not in public:
            public["acquisition_status"] = "excluded"
        if "reason_code" not in public:
            public["reason_code"] = "rejected_flare_pack" if row.get("_flare_excluded") else "excluded"
        key = hash_json(public)
        values[key] = public
    return sorted(
        values.values(),
        key=lambda row: (row.get("sprite_id", ""), row.get("source_run", ""), row.get("reason_code", "")),
    )


def _export_blobs(output: Path, candidates: list[dict[str, Any]]) -> None:
    blobs = output / "blobs"
    blobs.mkdir()
    for row in candidates:
        path = blobs / f"{row['exported_rgba_hash']}.rgba"
        payload = np.ascontiguousarray(row["_rgba"], dtype=np.uint8).tobytes()
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"content-address collision: {path.name}")
        if not path.exists():
            path.write_bytes(payload)


def _distribution(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "source": _counter_rows(Counter(row["source_id"] for row in records), len(records)),
        "pack": _counter_rows(Counter(row["pack_id"] for row in records), len(records)),
        "artist": _counter_rows(Counter(row["sub_artist"] for row in records), len(records)),
        "broad_pack_type": _counter_rows(Counter(row["broad_pack_type"] for row in records), len(records)),
    }


def _counter_rows(counts: Counter[str], total: int) -> list[dict[str, Any]]:
    return [
        {"value": value or "unknown", "count": count, "share": round(count / total, 8) if total else 0.0}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _dominance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {"value": "none", "count": 0, "share": 0.0}


def _rarity_points(count: int, total: int, maximum: int) -> float:
    return round(maximum * max(0.0, 1.0 - count / max(1, total)), 6)


def _broad_pack_type(*values: str) -> str:
    text = " ".join(values).lower()
    for name, tokens in (
        ("weapon", ("weapon", "sword")),
        ("tool", ("tool", "farming")),
        ("gem", ("gem", "crystal")),
        ("key", ("key",)),
        ("plant", ("plant", "mushroom")),
        ("armor", ("armor", "headgear", "shield")),
        ("material", ("material", "jewelry")),
        ("potion", ("potion",)),
        ("food", ("food",)),
    ):
        if any(token in text for token in tokens):
            return name
    return "mixed_item_pack"


def _native_dimensions(mapping: dict[str, Any], imported: dict[str, Any]) -> dict[str, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", str(mapping.get("native_resolution") or "").lower())
    if match:
        return {"width": int(match.group(1)), "height": int(match.group(2))}
    warnings = " ".join(str(value) for value in imported.get("warnings", []))
    match = re.search(r"from \((\d+),\s*(\d+)\)", warnings)
    if match:
        return {"width": int(match.group(1)), "height": int(match.group(2))}
    return {"width": 32, "height": 32}


def _resize_policy(native: dict[str, int], imported: dict[str, Any]) -> str:
    width, height = native["width"], native["height"]
    warnings = " ".join(str(value).lower() for value in imported.get("warnings", []))
    if "resized" in warnings or width > 32 or height > 32:
        return f"nearest_neighbor_resize_{width}x{height}_to_32x32"
    if width < 32 or height < 32:
        return f"transparent_center_pad_{width}x{height}_to_32x32"
    return "preserve_native_32x32_rgba"


def _resolve_image_path(run: Path, value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path.resolve()
    workspace = run.parent.parent
    workspace_candidate = (workspace / path).resolve()
    if workspace_candidate.is_file():
        return workspace_candidate
    return (run / path.name).resolve()


def _recover_download_hash(run: Path, source_id: str, original_filename: str) -> str:
    downloads = run / "downloads"
    if not downloads.is_dir():
        return ""
    files = sorted(path for path in downloads.iterdir() if path.is_file())
    if not files:
        return ""
    normalized_original = _norm(original_filename)
    normalized_source = re.sub(r"[^a-z0-9]+", "", source_id.lower())
    ranked = sorted(
        files,
        key=lambda path: (
            _norm(path.name) != normalized_original if normalized_original else True,
            normalized_source not in re.sub(r"[^a-z0-9]+", "", path.stem.lower()),
            path.name.lower(),
        ),
    )
    if len(files) > 1 and normalized_original and _norm(ranked[0].name) != normalized_original:
        return ""
    return file_sha256(ranked[0])


def _coordinate_from_path(value: str) -> str:
    match = re.search(r"__(r\d+_c\d+)", value)
    return match.group(1) if match else ""


def _run_revision(value: str) -> int:
    match = re.search(r"(?:final|retry)(\d+)$", value)
    return int(match.group(1)) if match else 0


def _norm(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").lower()


def _rebuild_command(repair_paths: Iterable[str | Path], *, workspace_root: Path) -> str:
    command = "$env:PYTHONPATH='src'; python -m spritelab.unlabeled_pool build --harvest-root harvest_runs"
    for raw_path in repair_paths:
        path = Path(raw_path).resolve()
        try:
            display = path.relative_to(workspace_root).as_posix()
        except ValueError:
            display = path.as_posix()
        quoted = "'" + display.replace("'", "''") + "'" if any(char.isspace() for char in display) else display
        command += f" --provenance-repair {quoted}"
    return (
        command
        + " --output datasets/sprite_lab_unlabeled_pool_v1_rebuild"
        + " --reports experiments/unlabeled_candidate_pool_v1_rebuild"
    )


def _safe_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, OSError, TypeError):
        return []


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(strict_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


def _write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _readme(summary: dict[str, Any]) -> str:
    return f"""# Sprite Lab unlabeled pool v1

This is an immutable acquisition and annotation-priority artifact. It is not a supervised dataset and contains no semantic labels or train/validation/test assignment.

- Retained candidates: {summary["retained_candidates"]}
- Exact unique images: {summary["exact_unique_images"]}
- Geometry families: {summary["unique_geometry_families"]}
- Recolor families: {summary["recolor_families"]}
- Annotation representatives: {summary["annotation_representatives"]}
- Acquisition policy: `{ACQUISITION_POLICY_VERSION}`
- Blob format: v5-compatible content-addressed decoded raw RGBA (`blobs/<exported_rgba_hash>.rgba`)

Licensing provenance fails closed. VLM output is not used as truth. Never edit or overwrite this frozen directory; rebuild to a new path and compare `freeze_manifest.json`.
"""


def _summary_markdown(summary: dict[str, Any]) -> list[str]:
    return [
        "# Pool summary",
        "",
        f"- Total source sprite occurrences: {summary['total_source_sprites']}",
        f"- Retained candidates: {summary['retained_candidates']}",
        f"- Suitability accept/quarantine/reject: {summary['suitability']['accept']}/{summary['suitability']['quarantine']}/{summary['suitability']['reject']}",
        f"- Exact unique images: {summary['exact_unique_images']}",
        f"- Unique geometry families: {summary['unique_geometry_families']}",
        f"- Recolor families: {summary['recolor_families']}",
        f"- Annotation representatives: {summary['annotation_representatives']}",
        f"- Expected label-propagation savings: {summary['expected_label_propagation_savings']}",
        f"- Provenance blocked: {summary['missing_provenance_count']}",
        f"- Flare retained: {summary['flare_retained_count']}",
    ]
