"""Auto-Labeling v3: sharded, resumable, streaming pipeline execution.

This module adds the operational scale layer required by Phase 6 on top of the
per-record fusion in ``pipeline_v3``:

* deterministic sharding by a stable hash of ``sprite_id`` (never input order);
* append-safe, per-record persistence with fsync (a crash loses at most the one
  record currently being written);
* a completion ledger so re-running skips finished records (resume);
* a structured failure queue with retryable/terminal classification;
* deterministic shard merge that rejects non-identical duplicate ids and yields
  a byte-identical canonical output regardless of shard count;
* streaming per-pack and global reports that never hold all records in memory.

Everything here is deterministic in ``--no-vlm`` mode: identical inputs and
config produce a byte-identical canonical ``v3_records.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v3.calibration import CalibrationArtifact
from spritelab.harvest.label_v3.config_v3 import V3PipelineConfig
from spritelab.harvest.label_v3.pipeline_v3 import (
    compute_record_decision,
    record_decision_to_json,
)
from spritelab.harvest.label_v3.record_decisions import record_decision_from_json
from spritelab.harvest.label_v3.stage_cache_v3 import (
    StageCache,
    record_content_hash,
    record_decision_cache_key,
)
from spritelab.utils.jsonl import iter_jsonl

logger = logging.getLogger(__name__)

SHARD_DIR = "shards"
CANONICAL_RECORDS_NAME = "v3_records.jsonl"
STAGE_META_NAME = "v3_stage_meta.json"
PIPELINE_STAGE_VERSION = "v3.1.0"


# ---------------------------------------------------------------------------
# Deterministic sharding
# ---------------------------------------------------------------------------


def stable_shard(sprite_id: str, shard_count: int) -> int:
    """Assign a record to a shard from a stable hash of its immutable id.

    Never derives the shard from input order, so shard membership is stable
    across runs and independent of how records were listed.
    """
    n = max(1, int(shard_count))
    digest = hashlib.sha256(str(sprite_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def record_in_shard(sprite_id: str, shard_index: int, shard_count: int) -> bool:
    return stable_shard(sprite_id, shard_count) == (int(shard_index) % max(1, int(shard_count)))


def _shard_stem(shard_index: int, shard_count: int) -> str:
    return f"shard_{int(shard_index):04d}_of_{int(shard_count):04d}"


@dataclass
class ShardPaths:
    records: Path
    ledger: Path
    failures: Path

    @classmethod
    def for_shard(cls, output_root: str | Path, shard_index: int, shard_count: int) -> ShardPaths:
        base = Path(output_root) / SHARD_DIR
        stem = _shard_stem(shard_index, shard_count)
        return cls(
            records=base / f"{stem}.records.jsonl",
            ledger=base / f"{stem}.ledger.jsonl",
            failures=base / f"{stem}.failures.jsonl",
        )


# ---------------------------------------------------------------------------
# Atomic append
# ---------------------------------------------------------------------------


def _append_line(path: Path, obj: Mapping[str, Any]) -> None:
    """Append one canonical JSON line and fsync so a crash cannot tear it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_ledger(path: Path) -> set[str]:
    done: set[str] = set()
    for row in iter_jsonl(path):
        sid = str(row.get("sprite_id", ""))
        if sid:
            done.add(sid)
    return done


# ---------------------------------------------------------------------------
# Shard execution (resumable)
# ---------------------------------------------------------------------------


@dataclass
class ShardRunResult:
    shard_index: int
    shard_count: int
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    cache_hits: int = 0
    state_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "state_counts": dict(self.state_counts),
        }


def _stage_meta_path(output_root: str | Path) -> Path:
    return Path(output_root) / STAGE_META_NAME


def _check_or_write_meta(output_root: str | Path, config: V3PipelineConfig, *, resume: bool) -> None:
    """Guard against silently mixing outputs from different configs."""
    meta_path = _stage_meta_path(output_root)
    current = {
        "policy_hash": config.policy.policy_hash(),
        "pipeline_hash": config.pipeline_hash(),
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("pipeline_hash") != current["pipeline_hash"]:
            raise ValueError(
                "v3 output root was produced with a different config "
                f"(existing pipeline_hash={existing.get('pipeline_hash')}, "
                f"current={current['pipeline_hash']}). Use a fresh --output-root."
            )
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(current, sort_keys=True, indent=2), encoding="utf-8")


def run_v3_shard(
    run_dir: str | Path,
    output_root: str | Path,
    config: V3PipelineConfig,
    *,
    calibration: CalibrationArtifact | None = None,
    use_vlm: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    resume: bool = True,
    max_records: int | None = None,
    cache_dir: str | Path | None = None,
) -> ShardRunResult:
    """Process one shard, resumably, writing per-record outputs + ledger.

    Records already present in this shard's ledger are skipped (resume). Each
    decision is appended and fsynced before its ledger entry, so an interrupted
    run re-computes at most the one in-flight record on the next resume.

    When ``cache_dir`` is given, fully-fused decisions are content-addressed
    there and reused across runs/output-roots as long as record content + policy
    + calibration are unchanged (a config change changes the key, so stale
    entries are never served).
    """
    run_path = Path(run_dir)
    imported_path = run_path / "imported.jsonl"
    paths = ShardPaths.for_shard(output_root, shard_index, shard_count)
    Path(output_root).mkdir(parents=True, exist_ok=True)

    _check_or_write_meta(output_root, config, resume=resume)

    if not resume:
        for p in (paths.records, paths.ledger, paths.failures):
            if p.exists():
                p.unlink()

    already_done = _read_ledger(paths.ledger) if resume else set()
    policy_hash = config.policy.policy_hash()

    cache: StageCache | None = None
    calibration_hash = calibration.artifact_hash() if calibration is not None else "none"
    if cache_dir is not None:
        cache = StageCache(cache_dir)

    result = ShardRunResult(shard_index=int(shard_index), shard_count=int(shard_count))
    state_counts: Counter[str] = Counter()
    seen = 0

    for record in iter_jsonl(imported_path):
        sprite_id = str(record.get("sprite_id", ""))
        if not sprite_id:
            continue
        if not record_in_shard(sprite_id, shard_index, shard_count):
            continue
        if max_records is not None and seen >= max_records:
            break
        seen += 1

        if sprite_id in already_done:
            result.skipped += 1
            continue

        try:
            cache_key = None
            if cache is not None:
                cache_key = record_decision_cache_key(
                    input_content_hash=record_content_hash(record),
                    policy_hash=policy_hash,
                    calibration_hash=calibration_hash,
                    stage_version=PIPELINE_STAGE_VERSION,
                )
                cached = cache.get(cache_key)
            else:
                cached = None

            if cached is not None:
                decision_json = cached
                decision = record_decision_from_json(cached)
                result.cache_hits += 1
            else:
                decision = compute_record_decision(
                    record, config, calibration=calibration, use_vlm=use_vlm, run_dir=run_path
                )
                decision_json = record_decision_to_json(decision)
                if cache is not None and cache_key is not None:
                    cache.put(cache_key, decision_json)
        except Exception as exc:
            _append_line(
                paths.failures,
                {
                    "sprite_id": sprite_id,
                    "stage": "fusion",
                    "reason": "record_processing_error",
                    "retryable": True,
                    "retry_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "policy_hash": policy_hash,
                },
            )
            result.failed += 1
            continue

        # Record first, then ledger: a crash between the two only re-does this
        # record (deterministic, so the redo is byte-identical).
        _append_line(paths.records, decision_json)
        _append_line(paths.ledger, {"sprite_id": sprite_id, "policy_hash": policy_hash})
        result.processed += 1
        state_counts[decision.record_state] += 1

    result.state_counts = dict(state_counts)
    return result


# ---------------------------------------------------------------------------
# Deterministic merge
# ---------------------------------------------------------------------------


def _canonical_line(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def merge_v3_shards(
    output_root: str | Path,
    *,
    out_path: str | Path | None = None,
    allow_duplicates: bool = False,
) -> dict[str, Any]:
    """Merge all shard record logs into one canonical, sorted output.

    Duplicate sprite ids are rejected unless byte-identical (or
    ``allow_duplicates``). The result is sorted by sprite_id and serialized
    canonically, so the merged file is byte-identical regardless of how many
    shards produced it.
    """
    base = Path(output_root) / SHARD_DIR
    target = Path(out_path) if out_path is not None else Path(output_root) / CANONICAL_RECORDS_NAME

    by_id: dict[str, str] = {}
    duplicates = 0
    for log in sorted(base.glob("*.records.jsonl")):
        for row in iter_jsonl(log):
            sid = str(row.get("sprite_id", ""))
            if not sid:
                continue
            canonical = _canonical_line(row)
            if sid in by_id:
                if by_id[sid] == canonical:
                    duplicates += 1
                    continue
                if allow_duplicates:
                    by_id[sid] = canonical  # last wins
                    duplicates += 1
                    continue
                raise ValueError(f"conflicting duplicate sprite_id in shard merge: {sid}")
            by_id[sid] = canonical

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for sid in sorted(by_id):
            handle.write(by_id[sid] + "\n")

    return {"merged_records": len(by_id), "duplicate_lines": duplicates, "output": str(target)}


# ---------------------------------------------------------------------------
# Failure-queue retry
# ---------------------------------------------------------------------------


def retry_v3_failures(
    run_dir: str | Path,
    output_root: str | Path,
    config: V3PipelineConfig,
    *,
    calibration: CalibrationArtifact | None = None,
    use_vlm: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Re-attempt only the retryable, not-yet-completed failures for one shard."""
    run_path = Path(run_dir)
    paths = ShardPaths.for_shard(output_root, shard_index, shard_count)
    if not paths.failures.exists():
        return {"retried": 0, "recovered": 0, "remaining": 0}

    failures = list(iter_jsonl(paths.failures))
    already_done = _read_ledger(paths.ledger)
    # Index imported records for the ids we need.
    wanted = {
        str(f.get("sprite_id", ""))
        for f in failures
        if bool(f.get("retryable")) and str(f.get("sprite_id", "")) not in already_done
    }
    records_by_id: dict[str, Mapping[str, Any]] = {}
    if wanted:
        for record in iter_jsonl(run_path / "imported.jsonl"):
            sid = str(record.get("sprite_id", ""))
            if sid in wanted:
                records_by_id[sid] = record

    policy_hash = config.policy.policy_hash()
    recovered = 0
    remaining: list[dict[str, Any]] = []
    retried = 0
    for f in failures:
        sid = str(f.get("sprite_id", ""))
        if not bool(f.get("retryable")) or sid in already_done or sid not in records_by_id:
            if sid not in already_done:
                remaining.append(dict(f))
            continue
        retried += 1
        try:
            decision = compute_record_decision(
                records_by_id[sid], config, calibration=calibration, use_vlm=use_vlm, run_dir=run_path
            )
        except Exception as exc:
            row = dict(f)
            row["retry_count"] = int(f.get("retry_count", 0)) + 1
            row["error"] = f"{type(exc).__name__}: {exc}"
            remaining.append(row)
            continue
        _append_line(paths.records, record_decision_to_json(decision))
        _append_line(paths.ledger, {"sprite_id": sid, "policy_hash": policy_hash})
        already_done.add(sid)
        recovered += 1

    # Rewrite the failure queue with only the still-failing entries.
    tmp = paths.failures.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in remaining:
            handle.write(_canonical_line(row) + "\n")
    tmp.replace(paths.failures)

    return {"retried": retried, "recovered": recovered, "remaining": len(remaining)}


# ---------------------------------------------------------------------------
# Streaming reports
# ---------------------------------------------------------------------------


def stream_v3_report(records_path: str | Path) -> dict[str, Any]:
    """Aggregate a v3 records JSONL in a single streaming pass (bounded memory).

    Never materializes the full record list, so it scales to 100k+ records
    without quadratic behavior.
    """
    record_states: Counter[str] = Counter()
    field_states: dict[str, Counter[str]] = defaultdict(Counter)
    reason_codes: Counter[str] = Counter()
    per_pack_state: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_categories: Counter[str] = Counter()
    accepted_objects: Counter[str] = Counter()
    total = 0

    field_names = ("domain", "category", "canonical_object", "color", "material", "shape")

    for row in iter_jsonl(records_path):
        total += 1
        state = str(row.get("record_state", "unknown"))
        record_states[state] += 1
        pack = str((row.get("lineage") or {}).get("pack_id", "") or "unknown")
        per_pack_state[pack][state] += 1
        for fname in field_names:
            fd = row.get(fname) or {}
            field_states[fname][str(fd.get("state", "unlabeled"))] += 1
        cat = row.get("category") or {}
        if cat.get("state") == "accepted" and cat.get("accepted_value"):
            accepted_categories[str(cat.get("accepted_value"))] += 1
        obj = row.get("canonical_object") or {}
        if obj.get("state") == "accepted" and obj.get("accepted_value"):
            accepted_objects[str(obj.get("accepted_value"))] += 1
        for code in row.get("reason_codes") or ():
            reason_codes[str(code)] += 1

    accepted = record_states.get("auto_accept", 0) + record_states.get("partial_accept", 0)
    return {
        "total_records": total,
        "record_states": dict(record_states),
        "acceptance_rate": accepted / max(1, total),
        "field_states": {k: dict(v) for k, v in field_states.items()},
        "reason_codes": dict(reason_codes),
        "per_pack_state": {k: dict(v) for k, v in per_pack_state.items()},
        "accepted_categories": dict(accepted_categories),
        "accepted_objects": dict(accepted_objects),
    }


def format_stream_report_md(report: Mapping[str, Any]) -> str:
    lines: list[str] = ["# Auto-Labeling v3 — Streaming Report", ""]
    lines.append(f"Total records: {report.get('total_records', 0)}")
    lines.append("")
    lines.append("## Record states")
    for state, count in sorted((report.get("record_states") or {}).items()):
        lines.append(f"- {state}: {count}")
    lines.append("")
    lines.append("## Field states")
    for fname, states in sorted((report.get("field_states") or {}).items()):
        rendered = ", ".join(f"{s}={c}" for s, c in sorted(states.items()))
        lines.append(f"- {fname}: {rendered}")
    lines.append("")
    lines.append("## Per-pack record states")
    for pack, states in sorted((report.get("per_pack_state") or {}).items()):
        rendered = ", ".join(f"{s}={c}" for s, c in sorted(states.items()))
        lines.append(f"- {pack}: {rendered}")
    lines.append("")
    lines.append("## Top reason codes")
    for code, count in sorted((report.get("reason_codes") or {}).items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {code}: {count}")
    return "\n".join(lines) + "\n"


def iter_canonical_records(records_path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream canonical records (thin wrapper kept for API symmetry)."""
    yield from iter_jsonl(records_path)
