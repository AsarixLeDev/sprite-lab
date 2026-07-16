"""Deterministic balanced annotation-batch scheduler.

This module deliberately treats pool metadata as evidence, not semantic labels.  It
never opens images and only reads the six artifacts listed in ``POOL_ARTIFACTS``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEDULER_VERSION = "annotation_scheduler_v1"
COHORT_POLICY_VERSION = "balanced_cohort_v1"
COHORT_MODES = ("semantic_accept_only", "quality_quarantine", "mixed_diagnostic")
POOL_ARTIFACTS = (
    "candidate_manifest.jsonl",
    "group_manifest.jsonl",
    "annotation_queue.jsonl",
    "quarantine_manifest.jsonl",
    "summary.json",
    "freeze_manifest.json",
)
PREFERRED_TYPES = ("armor", "plant", "gem", "material", "key", "tool", "weapon")
EARLY_LIMITED_TYPES = frozenset({"potion", "food"})


@dataclass(frozen=True)
class ScheduleConfig:
    """Stable policy parameters. Shares are soft upper bounds per batch."""

    batch_size: int = 50
    max_shade_share: float = 0.30
    max_single_pack_share: float = 0.30
    max_single_artist_share: float = 0.30
    min_broad_types: int = 6
    min_source_packs: int = 5
    early_batch_count: int = 5
    allow_duplicate_variants: bool = False
    strategy: str = "shade_capped"

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in ("max_shade_share", "max_single_pack_share", "max_single_artist_share"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.min_broad_types < 1 or self.min_source_packs < 1:
            raise ValueError("minimum diversity targets must be positive")
        if self.strategy not in {"current_priority", "balanced", "shade_capped"}:
            raise ValueError("strategy must be current_priority, balanced, or shade_capped")


@dataclass
class PoolData:
    representatives: list[dict[str, Any]]
    baseline_order: dict[str, int]
    summary: dict[str, Any]
    freeze: dict[str, Any]
    artifact_sha256: dict[str, str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pool(pool_dir: str | Path) -> PoolData:
    """Load and cross-check a frozen pool without reading any non-contract artifact."""

    pool = Path(pool_dir)
    missing = [name for name in POOL_ARTIFACTS if not (pool / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing immutable pool artifacts: {', '.join(missing)}")
    candidates = _read_jsonl(pool / "candidate_manifest.jsonl")
    groups = _read_jsonl(pool / "group_manifest.jsonl")
    queue = _read_jsonl(pool / "annotation_queue.jsonl")
    quarantine = _read_jsonl(pool / "quarantine_manifest.jsonl")
    summary = json.loads((pool / "summary.json").read_text(encoding="utf-8"))
    freeze = json.loads((pool / "freeze_manifest.json").read_text(encoding="utf-8"))

    by_id = {row["sprite_id"]: row for row in candidates}
    representatives = {row["sprite_id"]: dict(row) for row in candidates if row.get("annotation_representative")}
    recolor_by_member: dict[str, str] = {}
    geometry: dict[str, dict[str, Any]] = {}
    for group in groups:
        if group.get("group_kind") == "alpha_mask_recolor":
            for sprite_id in group.get("members", []):
                recolor_by_member[sprite_id] = group["group_id"]
        elif group.get("group_kind") == "geometry_family":
            geometry[group["group_id"]] = group

    for sprite_id, row in representatives.items():
        group_id = row.get("variant_geometry_group", "")
        group = geometry.get(group_id)
        if group is None or group.get("representative_sprite_id") != sprite_id:
            raise ValueError(f"representative/geometry mismatch for {sprite_id}")
        row["_recolor_family"] = next(
            (recolor_by_member[member] for member in group.get("members", []) if member in recolor_by_member), ""
        )
        row["_propagation_count"] = max(0, int(group.get("variant_count", len(group.get("members", [])))) - 1)

    raw_queue_ids = [row["sprite_id"] for row in queue]
    if len(raw_queue_ids) != len(set(raw_queue_ids)):
        raise ValueError("annotation_queue contains duplicate IDs")
    if any(row["sprite_id"] not in by_id and row.get("queue") != "provenance_blocked" for row in queue):
        raise ValueError("annotation_queue contains an unexpected unknown ID")
    representative_by_geometry = {group_id: group["representative_sprite_id"] for group_id, group in geometry.items()}
    queue_ids: list[str] = []
    seen_queue_representatives: set[str] = set()
    for queue_row in queue:
        representative_id = representative_by_geometry.get(queue_row.get("variant_geometry_group", ""))
        if representative_id and representative_id not in seen_queue_representatives:
            queue_ids.append(representative_id)
            seen_queue_representatives.add(representative_id)
    quarantine_ids = {row["sprite_id"] for row in quarantine}
    expected_quarantine = {sid for sid, row in representatives.items() if row.get("suitability_status") == "quarantine"}
    if not expected_quarantine <= quarantine_ids:
        raise ValueError("quarantine manifest is missing representative records")
    if summary.get("annotation_representatives") != len(representatives):
        raise ValueError("summary representative count disagrees with candidate manifest")
    if len(by_id) != len(candidates):
        raise ValueError("candidate_manifest contains duplicate sprite IDs")

    # The queue contains the immediately labelable subset. Complete the legacy
    # baseline with the other representatives in their existing priority order.
    queued = set(queue_ids)
    baseline_tail = sorted(
        (row for sid, row in representatives.items() if sid not in queued),
        key=lambda row: (-float(row.get("annotation_priority_score", 0)), row["sprite_id"]),
    )
    complete_baseline = queue_ids + [row["sprite_id"] for row in baseline_tail]
    return PoolData(
        representatives=[representatives[sid] for sid in sorted(representatives)],
        baseline_order={sid: index for index, sid in enumerate(complete_baseline)},
        summary=summary,
        freeze=freeze,
        artifact_sha256={name: _sha256(pool / name) for name in POOL_ARTIFACTS},
    )


def read_completed_ids(path: str | Path | None) -> set[str]:
    """Read append-only plain-ID or JSONL completion events without inspecting labels."""

    if path is None or not Path(path).exists():
        return set()
    completed: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            event = json.loads(line)
            sprite_id = event.get("representative_id") or event.get("sprite_id")
        else:
            sprite_id = line
        if not isinstance(sprite_id, str) or not sprite_id:
            raise ValueError("completed-ID entries must contain a non-empty sprite ID")
        completed.add(sprite_id)
    return completed


def _artist(row: dict[str, Any]) -> str:
    return str(row.get("sub_artist") or row.get("author") or row.get("attribution") or "unknown")


def _pack(row: dict[str, Any]) -> str:
    return str(row.get("pack_id") or row.get("pack_name") or row.get("source_id") or "unknown")


def _is_shade(row: dict[str, Any]) -> bool:
    return _artist(row).casefold() == "shade" or _pack(row).casefold().startswith("shade_")


def _preferred_rank(row: dict[str, Any], batch_number: int, early_count: int) -> int:
    broad_type = str(row.get("broad_pack_type") or "unknown")
    if broad_type == "tool" and _is_shade(row):
        return 8
    if broad_type == "weapon" and _is_shade(row):
        return 9
    try:
        rank = PREFERRED_TYPES.index(broad_type)
    except ValueError:
        rank = len(PREFERRED_TYPES) + 2
    if batch_number <= early_count and broad_type in EARLY_LIMITED_TYPES:
        rank += 20
    return rank


def _cap_count(share: float, target_size: int) -> int:
    return max(1, math.floor(share * target_size + 1e-12))


def _candidate_key(
    row: dict[str, Any],
    batch: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
    config: ScheduleConfig,
    batch_number: int,
    target_size: int,
    type_counts: Counter[str],
    pack_counts: Counter[str],
    artist_counts: Counter[str],
) -> tuple[Any, ...]:
    broad_type = str(row.get("broad_pack_type") or "unknown")
    pack = _pack(row)
    artist = _artist(row)
    batch_types = {str(item.get("broad_pack_type") or "unknown") for item in batch}
    batch_packs = {_pack(item) for item in batch}
    slots_left = target_size - len(batch)
    types_needed = max(0, min(config.min_broad_types, len(type_counts)) - len(batch_types))
    packs_needed = max(0, min(config.min_source_packs, len(pack_counts)) - len(batch_packs))
    diversity_urgency = (
        int(slots_left <= types_needed and broad_type in batch_types)
        + int(slots_left <= packs_needed and pack in batch_packs)
        + int(
            config.strategy == "shade_capped"
            and slots_left == 1
            and not any(_is_shade(item) for item in batch)
            and not _is_shade(row)
            and any(_is_shade(item) for item in remaining)
        )
    )

    violations = 0
    overage = 0
    if config.strategy != "current_priority":
        pack_over = (
            len([item for item in batch if _pack(item) == pack])
            + 1
            - _cap_count(config.max_single_pack_share, target_size)
        )
        artist_over = (
            len([item for item in batch if _artist(item) == artist])
            + 1
            - _cap_count(config.max_single_artist_share, target_size)
        )
        for value in (pack_over, artist_over):
            if value > 0:
                violations += 1
                overage += value
        if config.strategy == "shade_capped" and _is_shade(row):
            shade_over = sum(_is_shade(item) for item in batch) + 1 - _cap_count(config.max_shade_share, target_size)
            if shade_over > 0:
                violations += 1
                overage += shade_over

    if config.strategy == "current_priority":
        return (0, 0, 0, row["_baseline_order"], row["sprite_id"])
    components = row.get("annotation_priority_components", {})
    provenance = float(components.get("provenance_completeness", 0))
    propagation = int(row.get("_propagation_count", 0))
    suitability_rank = 0 if row.get("suitability_status") == "accept" else 1
    # Rare categories lead; hashes provide deterministic silhouette/palette tie breaks.
    return (
        violations,
        overage,
        diversity_urgency,
        _preferred_rank(row, batch_number, config.early_batch_count),
        type_counts[broad_type],
        pack_counts[pack],
        artist_counts[artist],
        -propagation,
        -provenance,
        suitability_rank,
        str(row.get("normalized_alpha_hash", "")),
        str(row.get("exported_rgba_hash", "")),
        row["sprite_id"],
    )


def _reason(row: dict[str, Any], relaxed: list[str], batch_number: int, config: ScheduleConfig) -> str:
    reasons = ["unique geometry representative"]
    broad_type = str(row.get("broad_pack_type") or "unknown")
    if broad_type in PREFERRED_TYPES and batch_number <= config.early_batch_count:
        if broad_type in {"tool", "weapon"} and _is_shade(row):
            pass
        else:
            reasons.append(f"early underrepresented type: {broad_type}")
    if row.get("_propagation_count", 0):
        reasons.append(f"propagates to {row['_propagation_count']} variants")
    reasons.append("accept suitability" if row.get("suitability_status") == "accept" else "quarantine fallback")
    if relaxed:
        reasons.append("soft fallback: " + ", ".join(relaxed))
    return "; ".join(reasons)


def _output_row(row: dict[str, Any], batch_number: int, reason: str) -> dict[str, Any]:
    return {
        "sprite_id": row["sprite_id"],
        "geometry_group": row.get("variant_geometry_group", ""),
        "recolor_family": row.get("_recolor_family", ""),
        "representative_id": row["sprite_id"],
        "source": row.get("source_id", ""),
        "pack": _pack(row),
        "artist": _artist(row),
        "broad_type": row.get("broad_pack_type", "unknown"),
        "suitability": {
            "status": row.get("suitability_status", "unknown"),
            "score": row.get("suitability_score"),
            "reason_codes": row.get("suitability_reason_codes", []),
        },
        "annotation_priority_components": row.get("annotation_priority_components", {}),
        "propagation_count": int(row.get("_propagation_count", 0)),
        "scheduling_reason": reason,
        "batch_number": batch_number,
    }


def _batch_stats(rows: list[dict[str, Any]], deferred_records: int) -> dict[str, Any]:
    size = len(rows)
    types = Counter(row["broad_type"] for row in rows)
    packs = Counter(row["pack"] for row in rows)
    artists = Counter(row["artist"] for row in rows)
    suitability = Counter(row["suitability"]["status"] for row in rows)
    shade = sum(row["artist"].casefold() == "shade" or row["pack"].casefold().startswith("shade_") for row in rows)
    return {
        "batch_number": rows[0]["batch_number"] if rows else 0,
        "records": size,
        "broad_type_coverage": len(types),
        "broad_type_distribution": dict(sorted(types.items())),
        "pack_coverage": len(packs),
        "artist_coverage": len(artists),
        "shade_count": shade,
        "shade_share": round(shade / size, 8) if size else 0.0,
        "estimated_propagated_variants": sum(row["propagation_count"] for row in rows),
        "unique_geometries": len({row["geometry_group"] for row in rows}),
        "suitability_status": dict(sorted(suitability.items())),
        "deferred_records": deferred_records,
    }


def _load_existing(output_dir: Path) -> tuple[list[list[dict[str, Any]]], list[int]]:
    plan_path = output_dir / "annotation_plan.json"
    if not plan_path.exists():
        return [], []
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    issued = sorted({int(value) for value in plan.get("issued_batches", [])})
    if issued != list(range(1, max(issued, default=0) + 1)):
        raise ValueError("issued_batches must be a contiguous prefix")
    batches = []
    for number in issued:
        path = output_dir / "batches" / f"batch_{number:04d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"issued batch is missing: {path}")
        batches.append(_read_jsonl(path))
    return batches, issued


def build_schedule(
    pool_dir: str | Path,
    output_dir: str | Path,
    *,
    config: ScheduleConfig | None = None,
    completed_ids_path: str | Path | None = None,
    preserve_issued: bool = True,
) -> dict[str, Any]:
    """Build a complete schedule, preserving explicitly issued batch files verbatim."""

    config = config or ScheduleConfig()
    config.validate()
    pool = load_pool(pool_dir)
    output = Path(output_dir)
    old_batches, issued = _load_existing(output) if preserve_issued else ([], [])
    completed = read_completed_ids(completed_ids_path)
    known_ids = {row["sprite_id"] for row in pool.representatives}
    unknown_completed = completed - known_ids
    if unknown_completed:
        raise ValueError(f"completed IDs are not pool representatives: {sorted(unknown_completed)[0]}")
    preserved_ids = {row["sprite_id"] for batch in old_batches for row in batch}
    if len(preserved_ids) != sum(map(len, old_batches)):
        raise ValueError("issued batches contain duplicate representatives")

    remaining = []
    for row in pool.representatives:
        if row["sprite_id"] in completed or row["sprite_id"] in preserved_ids:
            continue
        copied = dict(row)
        copied["_baseline_order"] = pool.baseline_order[row["sprite_id"]]
        remaining.append(copied)
    type_counts = Counter(str(row.get("broad_pack_type") or "unknown") for row in remaining)
    pack_counts = Counter(_pack(row) for row in remaining)
    artist_counts = Counter(_artist(row) for row in remaining)

    batches = [list(batch) for batch in old_batches]
    relaxations: list[dict[str, Any]] = []
    next_number = max(issued, default=0) + 1
    while remaining:
        target_size = min(config.batch_size, len(remaining))
        selected_raw: list[dict[str, Any]] = []
        selected_out: list[dict[str, Any]] = []
        while len(selected_raw) < target_size:
            ranked = sorted(
                remaining,
                key=lambda row: _candidate_key(
                    row,
                    selected_raw,
                    remaining,
                    config,
                    next_number,
                    target_size,
                    type_counts,
                    pack_counts,
                    artist_counts,
                ),
            )
            chosen = ranked[0]
            key = _candidate_key(
                chosen,
                selected_raw,
                remaining,
                config,
                next_number,
                target_size,
                type_counts,
                pack_counts,
                artist_counts,
            )
            relaxed: list[str] = []
            if config.strategy != "current_priority" and key[0]:
                pack_count = sum(_pack(item) == _pack(chosen) for item in selected_raw) + 1
                artist_count = sum(_artist(item) == _artist(chosen) for item in selected_raw) + 1
                if pack_count > _cap_count(config.max_single_pack_share, target_size):
                    relaxed.append("pack cap")
                if artist_count > _cap_count(config.max_single_artist_share, target_size):
                    relaxed.append("artist cap")
                shade_count = sum(_is_shade(item) for item in selected_raw) + int(_is_shade(chosen))
                if config.strategy == "shade_capped" and shade_count > _cap_count(config.max_shade_share, target_size):
                    relaxed.append("Shade cap")
            if relaxed:
                relaxations.append(
                    {"batch_number": next_number, "sprite_id": chosen["sprite_id"], "constraints": relaxed}
                )
            selected_raw.append(chosen)
            selected_out.append(_output_row(chosen, next_number, _reason(chosen, relaxed, next_number, config)))
            remaining.remove(chosen)
        batches.append(selected_out)
        next_number += 1

    # Issued files are never rewritten. New/unissued files are deterministic replacements.
    output.mkdir(parents=True, exist_ok=True)
    batch_dir = output / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(batches, 1):
        if index in issued:
            continue
        jsonl = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in batch)
        (batch_dir / f"batch_{index:04d}.jsonl").write_text(jsonl, encoding="utf-8")
        stats = _batch_stats(batch, sum(len(value) for value in batches[index:]))
        md = _batch_markdown(stats, batch)
        (batch_dir / f"batch_{index:04d}.md").write_text(md, encoding="utf-8")

    all_rows = [row for batch in batches for row in batch]
    deferred = []
    for row in all_rows:
        if row["batch_number"] > 1:
            deferred.append(
                {
                    **row,
                    "deferred_from_batch": 1,
                    "deferred_reason": "retained for a later balanced batch",
                    "scheduled_batch": row["batch_number"],
                }
            )
    (output / "deferred_candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in deferred), encoding="utf-8"
    )
    batch_summaries = [
        _batch_stats(batch, sum(len(value) for value in batches[index + 1 :])) for index, batch in enumerate(batches)
    ]
    summary = {
        "scheduler_version": SCHEDULER_VERSION,
        "strategy": config.strategy,
        "total_batches": len(batches),
        "scheduled_representatives": len(all_rows),
        "completed_count": len(completed),
        "issued_batches": issued,
        "estimated_propagated_variants": sum(row["propagation_count"] for row in all_rows),
        "expected_annotation_actions_saved": sum(row["propagation_count"] for row in all_rows),
        "pool_shade_share": pool.summary.get("pack_dominance", {})
        .get("after_representative_selection", {})
        .get("share"),
        "scheduled_shade_share": round(sum(row["artist"].casefold() == "shade" for row in all_rows) / len(all_rows), 8)
        if all_rows
        else 0.0,
        "soft_constraint_relaxations": relaxations,
        "batches": batch_summaries,
    }
    plan = {
        "schema_version": 1,
        "scheduler_version": SCHEDULER_VERSION,
        "pool_artifact_sha256": pool.artifact_sha256,
        "pool_content_manifest_hash": pool.freeze.get("content_manifest_hash"),
        "config": asdict(config),
        "completed_ids_file": str(completed_ids_path) if completed_ids_path else None,
        "issued_batches": issued,
        "total_batches": len(batches),
        "batch_files": [f"batches/batch_{index:04d}.jsonl" for index in range(1, len(batches) + 1)],
        "fallback_policy": "Meet all caps when feasible; otherwise minimize cap overage, record the relaxation, and retain overflow for later batches.",
    }
    (output / "annotation_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "schedule_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def mark_issued(output_dir: str | Path, batch_number: int) -> dict[str, Any]:
    """Record an issued batch in the plan without changing its contents."""

    output = Path(output_dir)
    plan_path = output / "annotation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if batch_number < 1 or batch_number > int(plan["total_batches"]):
        raise IndexError("batch number is outside the schedule")
    expected = max(plan.get("issued_batches", []), default=0) + 1
    if batch_number != expected and batch_number not in plan.get("issued_batches", []):
        raise ValueError(f"batches must be issued in order; expected batch {expected}")
    batch_path = output / "batches" / f"batch_{batch_number:04d}.jsonl"
    if not batch_path.exists():
        raise FileNotFoundError(batch_path)
    plan["issued_batches"] = sorted({*plan.get("issued_batches", []), batch_number})
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"batch_number": batch_number, "records": len(_read_jsonl(batch_path)), "issued": True}


def _batch_markdown(stats: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Annotation batch {stats['batch_number']:04d}",
        "",
        f"Records: {stats['records']}  ",
        f"Broad types: {stats['broad_type_coverage']}  ",
        f"Packs: {stats['pack_coverage']}  ",
        f"Artists: {stats['artist_coverage']}  ",
        f"Shade share: {stats['shade_share']:.2%}  ",
        f"Estimated propagated variants: {stats['estimated_propagated_variants']}",
        "",
        "| # | Representative | Type | Pack | Artist | Suitability | Propagation |",
        "|---:|---|---|---|---|---|---:|",
    ]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | `{row['representative_id']}` | {row['broad_type']} | {row['pack']} | "
            f"{row['artist']} | {row['suitability']['status']} | {row['propagation_count']} |"
        )
    return "\n".join(lines) + "\n"


class ScheduleView:
    """Read-only GUI integration facade over generated schedule artifacts."""

    def __init__(self, output_dir: str | Path, completed_ids_path: str | Path | None = None):
        self.output_dir = Path(output_dir)
        self.plan = json.loads((self.output_dir / "annotation_plan.json").read_text(encoding="utf-8"))
        self.completed_ids_path = Path(completed_ids_path) if completed_ids_path else None

    @property
    def completed_ids(self) -> set[str]:
        return read_completed_ids(self.completed_ids_path)

    def specific_batch(self, batch_number: int) -> list[dict[str, Any]]:
        if batch_number < 1 or batch_number > int(self.plan["total_batches"]):
            raise IndexError("batch number is outside the schedule")
        return _read_jsonl(self.output_dir / "batches" / f"batch_{batch_number:04d}.jsonl")

    def remaining_batches(self) -> list[int]:
        completed = self.completed_ids
        return [
            number
            for number in range(1, int(self.plan["total_batches"]) + 1)
            if any(row["representative_id"] not in completed for row in self.specific_batch(number))
        ]

    def next_batch(self) -> list[dict[str, Any]]:
        remaining = self.remaining_batches()
        if not remaining:
            return []
        completed = self.completed_ids
        return [row for row in self.specific_batch(remaining[0]) if row["representative_id"] not in completed]

    def completed_count(self) -> int:
        return len(self.completed_ids)

    def propagation_metrics(self) -> dict[str, int]:
        """Return explicitly named estimated, completed, and remaining values."""

        completed = self.completed_ids
        rows = [row for number in range(1, int(self.plan["total_batches"]) + 1) for row in self.specific_batch(number)]
        estimated = sum(int(row.get("propagation_count", 0)) for row in rows)
        completed_value = sum(
            int(row.get("propagation_count", 0)) for row in rows if row["representative_id"] in completed
        )
        return {
            "estimated_propagated_variants": estimated,
            "completed_propagated_variants": completed_value,
            "remaining_propagation_value": estimated - completed_value,
        }

    def propagated_variant_count(self) -> dict[str, int | str]:
        """Compatibility query whose value is now unambiguously completed-only."""

        metrics: dict[str, int | str] = dict(self.propagation_metrics())
        metrics["metric"] = "completed_propagated_variants"
        metrics["propagated_variant_count"] = metrics["completed_propagated_variants"]
        return metrics

    def export_cohort(
        self,
        batch_number: int,
        cohort_size: int,
        *,
        mode: str = "semantic_accept_only",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Select a deterministic balanced cohort from an entire stored batch."""

        if cohort_size < 1:
            raise ValueError("cohort_size must be positive")
        if mode not in COHORT_MODES:
            raise ValueError(f"cohort mode must be one of: {', '.join(COHORT_MODES)}")
        batch = self.specific_batch(batch_number)
        eligible = [row for row in batch if _cohort_mode_allows(row, mode)]
        if cohort_size > len(eligible):
            raise ValueError(f"cohort size {cohort_size} exceeds {len(eligible)} eligible records for mode {mode}")

        batch_order = {row["representative_id"]: index for index, row in enumerate(batch)}
        remaining = list(eligible)
        selected: list[dict[str, Any]] = []
        global_types = Counter(str(row.get("broad_type", "unknown")) for row in eligible)
        global_packs = Counter(str(row.get("pack", "unknown")) for row in eligible)
        global_artists = Counter(str(row.get("artist", "unknown")) for row in eligible)
        while len(selected) < cohort_size:
            type_counts = Counter(str(row.get("broad_type", "unknown")) for row in selected)
            pack_counts = Counter(str(row.get("pack", "unknown")) for row in selected)
            artist_counts = Counter(str(row.get("artist", "unknown")) for row in selected)
            selected_ids = {row["representative_id"] for row in selected}

            def key(
                row: dict[str, Any],
                type_counts: Counter[str] = type_counts,
                pack_counts: Counter[str] = pack_counts,
                artist_counts: Counter[str] = artist_counts,
            ) -> tuple[Any, ...]:
                broad_type = str(row.get("broad_type", "unknown"))
                pack = str(row.get("pack", "unknown"))
                artist = str(row.get("artist", "unknown"))
                suitability = str((row.get("suitability") or {}).get("status", "unknown"))
                is_shade = artist.casefold() == "shade" or pack.casefold().startswith("shade_")
                shade_count = sum(
                    str(item.get("artist", "")).casefold() == "shade"
                    or str(item.get("pack", "")).casefold().startswith("shade_")
                    for item in selected
                )
                shade_cap = max(1, math.floor(cohort_size * 0.30))
                return (
                    int(is_shade and shade_count >= shade_cap),
                    type_counts[broad_type],
                    pack_counts[pack],
                    artist_counts[artist],
                    global_types[broad_type],
                    global_packs[pack],
                    global_artists[artist],
                    0 if suitability == "accept" else 1,
                    -int(row.get("propagation_count", 0)),
                    batch_order[row["representative_id"]],
                    row["representative_id"],
                )

            chosen = min((row for row in remaining if row["representative_id"] not in selected_ids), key=key)
            selected.append(chosen)
            remaining.remove(chosen)

        output_rows = []
        for selection_index, row in enumerate(selected):
            output_rows.append(
                {
                    **row,
                    "cohort_context": {
                        "mode": mode,
                        "selection_index": selection_index,
                        "original_batch_index": batch_order[row["representative_id"]],
                        "broad_type_is_scheduling_metadata_only": True,
                    },
                }
            )
        selected_ids = [row["representative_id"] for row in output_rows]
        selected_set = set(selected_ids)
        excluded = [row["representative_id"] for row in batch if row["representative_id"] not in selected_set]

        def distribution(field: str) -> dict[str, int]:
            return dict(sorted(Counter(str(row.get(field, "unknown")) for row in output_rows).items()))

        suitability = dict(
            sorted(Counter(str((row.get("suitability") or {}).get("status", "unknown")) for row in output_rows).items())
        )
        batch_path = self.output_dir / "batches" / f"batch_{batch_number:04d}.jsonl"
        plan_path = self.output_dir / "annotation_plan.json"
        manifest = {
            "schema_version": 1,
            "schedule_hash": _sha256(plan_path),
            "batch_hash": _sha256(batch_path),
            "batch_number": batch_number,
            "cohort_size": cohort_size,
            "cohort_mode": mode,
            "selection_policy": COHORT_POLICY_VERSION,
            "selected_ids": selected_ids,
            "excluded_deferred_ids": excluded,
            "original_batch_order": [row["representative_id"] for row in batch],
            "broad_type_distribution": distribution("broad_type"),
            "pack_distribution": distribution("pack"),
            "artist_distribution": distribution("artist"),
            "suitability_distribution": suitability,
            "estimated_propagated_variants": sum(int(row.get("propagation_count", 0)) for row in output_rows),
            "quality_decision_values": ["quality_accept", "quality_reject", "quality_uncertain"]
            if mode == "quality_quarantine"
            else [],
            "semantic_fields_are_separate_from_quality_decisions": True,
        }
        return output_rows, manifest


def _cohort_mode_allows(row: dict[str, Any], mode: str) -> bool:
    status = str((row.get("suitability") or {}).get("status", "unknown"))
    if mode == "semantic_accept_only":
        return status == "accept"
    if mode == "quality_quarantine":
        return status == "quarantine"
    return True


def compare_strategies(pool_dir: str | Path, batch_size: int = 50) -> dict[str, list[dict[str, Any]]]:
    """Return in-memory per-batch evaluations for all required strategies."""

    import tempfile

    result: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="annotation-scheduler-") as root:
        for strategy in ("current_priority", "balanced", "shade_capped"):
            config = ScheduleConfig(
                batch_size=batch_size,
                strategy=strategy,
                max_shade_share=0.30,
                max_single_pack_share=0.35 if strategy == "balanced" else 0.30,
                max_single_artist_share=0.35 if strategy == "balanced" else 0.30,
            )
            summary = build_schedule(pool_dir, Path(root) / strategy, config=config, preserve_issued=False)
            result[strategy] = summary["batches"]
    return result
