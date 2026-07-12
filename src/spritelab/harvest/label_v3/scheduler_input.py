"""Bridge immutable annotation-scheduler cohorts into assisted-v3 review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.label_v3.assisted_golden_v3 import load_all_v3_records
from spritelab.harvest.label_v3.config_v3 import V3PipelineConfig
from spritelab.harvest.label_v3.pipeline_v3 import compute_record_decision
from spritelab.harvest.label_v3.record_decisions import RecordDecision, record_decision_to_json
from spritelab.harvest.sources import utc_timestamp
from spritelab.utils.jsonl import read_jsonl

SCHEDULER_RECORDS_FILENAME = "scheduler_v3_records.jsonl"
RESOLVED_CANDIDATES_FILENAME = "scheduler_resolved_candidates.jsonl"
QUALITY_DECISIONS_FILENAME = "quality_decisions.jsonl"


@dataclass(frozen=True)
class SchedulerPreparation:
    cohort_path: Path
    pool_path: Path
    work_dir: Path
    records_path: Path
    resolved_candidates_path: Path
    candidate_ids: tuple[str, ...]
    reused_prefills: int
    generated_deterministic_prefills: int
    harvest_resolved: int
    blob_resolved: int
    estimated_propagated_variants: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort": str(self.cohort_path),
            "pool": str(self.pool_path),
            "work_dir": str(self.work_dir),
            "records_path": str(self.records_path),
            "resolved_candidates_path": str(self.resolved_candidates_path),
            "candidate_ids": list(self.candidate_ids),
            "records": len(self.candidate_ids),
            "reused_prefills": self.reused_prefills,
            "generated_deterministic_prefills": self.generated_deterministic_prefills,
            "paid_vlm_calls": 0,
            "harvest_resolved": self.harvest_resolved,
            "blob_resolved": self.blob_resolved,
            "estimated_propagated_variants": self.estimated_propagated_variants,
        }


def prepare_scheduler_v3(
    cohort_path: str | Path,
    pool_path: str | Path,
    work_dir: str | Path,
    *,
    harvest_root: str | Path = "harvest_runs",
    config: V3PipelineConfig | None = None,
) -> SchedulerPreparation:
    """Resolve cohort sprites and create/reuse deterministic v3 prefills in cohort order."""

    cohort = Path(cohort_path)
    pool = Path(pool_path)
    work = Path(work_dir)
    rows = read_jsonl(cohort)
    if not rows:
        raise ValueError("scheduler cohort is empty")
    ids = [str(row.get("representative_id") or row.get("sprite_id") or "") for row in rows]
    if any(not sprite_id for sprite_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("scheduler cohort must contain unique non-empty representative IDs")
    modes = {str((row.get("cohort_context") or {}).get("mode", "")) for row in rows}
    if len(modes - {""}) > 1:
        raise ValueError("scheduler cohort mixes review modes")
    mode = next(iter(modes - {""}), "semantic_accept_only")
    if mode == "semantic_accept_only":
        quarantined = [
            sprite_id
            for sprite_id, row in zip(ids, rows, strict=True)
            if str((row.get("suitability") or {}).get("status", "")) != "accept"
        ]
        if quarantined:
            raise ValueError(f"semantic_accept_only cohort contains quarantined/non-accept record: {quarantined[0]}")

    pool_candidates = {str(row.get("sprite_id", "")): row for row in read_jsonl(pool / "candidate_manifest.jsonl")}
    missing_pool = [sprite_id for sprite_id in ids if sprite_id not in pool_candidates]
    if missing_pool:
        raise ValueError(f"cohort representative is not present in pool: {missing_pool[0]}")

    work.mkdir(parents=True, exist_ok=True)
    resolved_dir = work / "resolved_png"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    relevant_runs = {
        str(run)
        for sprite_id in ids
        for run in (pool_candidates[sprite_id].get("source_runs") or [pool_candidates[sprite_id].get("source_run", "")])
        if run
    }
    imported, existing_prefills = _index_harvest_inputs(Path(harvest_root), relevant_runs, set(ids))

    previous = load_all_v3_records(work / SCHEDULER_RECORDS_FILENAME)
    existing_prefills = {**existing_prefills, **previous}
    resolved_rows: list[dict[str, Any]] = []
    decisions: list[RecordDecision] = []
    reused = generated = harvest_resolved = blob_resolved = 0
    cfg = config or V3PipelineConfig()
    for order, (sprite_id, cohort_row) in enumerate(zip(ids, rows, strict=True)):
        pool_row = dict(pool_candidates[sprite_id])
        imported_row = imported.get(sprite_id, {})
        record = {**pool_row, **imported_row, "sprite_id": sprite_id}
        # Scheduling type is prioritization context, never semantic evidence.
        record.pop("broad_pack_type", None)
        image_path, resolution = _resolve_image(pool, pool_row, imported_row, resolved_dir)
        record["final_png_path"] = str(image_path)
        record["scheduler_context"] = {
            "batch_number": cohort_row.get("batch_number"),
            "cohort_mode": mode,
            "cohort_order": order,
            "original_batch_index": (cohort_row.get("cohort_context") or {}).get("original_batch_index"),
            "broad_type": cohort_row.get("broad_type", "unknown"),
            "broad_type_is_reviewed_truth": False,
            "geometry_group": cohort_row.get("geometry_group", ""),
            "propagation_count": int(cohort_row.get("propagation_count", 0)),
            "suitability": cohort_row.get("suitability", {}),
        }
        if resolution == "harvest_run":
            harvest_resolved += 1
        else:
            blob_resolved += 1
        resolved_rows.append(record)

        decision = existing_prefills.get(sprite_id)
        if decision is None:
            decision = compute_record_decision(record, cfg, use_vlm=False, run_dir=work)
            generated += 1
        else:
            reused += 1
        metadata = {
            **dict(decision.prefill_metadata),
            "scheduler_context": record["scheduler_context"],
            "resolved_png_path": str(image_path),
            "sprite_resolution": resolution,
        }
        decisions.append(replace(decision, prefill_metadata=metadata))

    resolved_path = work / RESOLVED_CANDIDATES_FILENAME
    records_path = work / SCHEDULER_RECORDS_FILENAME
    _write_jsonl(resolved_path, resolved_rows)
    _write_jsonl(records_path, [record_decision_to_json(decision) for decision in decisions])
    return SchedulerPreparation(
        cohort_path=cohort,
        pool_path=pool,
        work_dir=work,
        records_path=records_path,
        resolved_candidates_path=resolved_path,
        candidate_ids=tuple(ids),
        reused_prefills=reused,
        generated_deterministic_prefills=generated,
        harvest_resolved=harvest_resolved,
        blob_resolved=blob_resolved,
        estimated_propagated_variants=sum(int(row.get("propagation_count", 0)) for row in rows),
    )


def append_completed_id(path: str | Path, sprite_id: str, *, cohort_hash: str = "") -> bool:
    """Append one representative completion event once; never rewrite prior events."""

    target = Path(path)
    existing: set[str] = set()
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line) if line.lstrip().startswith("{") else {"representative_id": line.strip()}
            existing.add(str(value.get("representative_id") or value.get("sprite_id") or ""))
    if sprite_id in existing:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "representative_id": sprite_id,
        "completed_at": utc_timestamp(),
        "cohort_hash": cohort_hash,
    }
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    return True


def append_quality_decision(
    path: str | Path,
    sprite_id: str,
    decision: str,
    *,
    suitability_reason_codes: Sequence[str] = (),
    reviewer_id: str = "",
) -> None:
    """Append a quality-only decision without writing semantic fields."""

    if decision not in {"quality_accept", "quality_reject", "quality_uncertain"}:
        raise ValueError("invalid quality decision")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "sprite_id": sprite_id,
        "quality_decision": decision,
        "suitability_reason_codes": list(suitability_reason_codes),
        "reviewer_id": reviewer_id,
        "timestamp": utc_timestamp(),
    }
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def cohort_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _index_harvest_inputs(
    harvest_root: Path, relevant_runs: set[str], wanted_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, RecordDecision]]:
    imported: dict[str, dict[str, Any]] = {}
    prefills: dict[str, RecordDecision] = {}
    for run_name in sorted(relevant_runs):
        run = harvest_root / run_name
        imported_path = run / "imported.jsonl"
        if imported_path.is_file():
            for row in read_jsonl(imported_path):
                sprite_id = str(row.get("sprite_id", ""))
                if sprite_id in wanted_ids and sprite_id not in imported:
                    imported[sprite_id] = row
        for path in sorted((run / "v3_output").glob("*v3_records.jsonl")) if (run / "v3_output").is_dir() else ():
            prefills.update({sid: value for sid, value in load_all_v3_records(path).items() if sid in wanted_ids})
    return imported, prefills


def _resolve_image(
    pool: Path,
    pool_row: Mapping[str, Any],
    imported_row: Mapping[str, Any],
    resolved_dir: Path,
) -> tuple[Path, str]:
    raw = str(imported_row.get("final_png_path") or "")
    if raw:
        path = Path(raw)
        candidates = [path, Path.cwd() / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve(), "harvest_run"
    blob = pool / str(pool_row.get("blob_path") or "")
    if not blob.is_file():
        raise FileNotFoundError(f"no harvest image or pool blob for {pool_row.get('sprite_id')}")
    width = int(pool_row.get("exported_width", 0))
    height = int(pool_row.get("exported_height", 0))
    payload = blob.read_bytes()
    if width < 1 or height < 1 or len(payload) != width * height * 4:
        raise ValueError(f"invalid RGBA pool blob for {pool_row.get('sprite_id')}")
    output = resolved_dir / f"{pool_row.get('exported_rgba_hash')}.png"
    if not output.is_file():
        Image.frombytes("RGBA", (width, height), payload).save(output, format="PNG", optimize=False)
    return output.resolve(), "pool_blob"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")
