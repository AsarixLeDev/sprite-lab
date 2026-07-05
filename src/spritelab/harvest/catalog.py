"""Resumable JSONL state files for harvest runs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.harvest.extract import HarvestCandidate
from spritelab.harvest.pipeline import HarvestedSprite
from spritelab.harvest.sources import (
    SourceRecord,
    source_record_from_dict,
    source_record_to_dict,
    utc_timestamp,
)

RUN_FILES = (
    "sources.jsonl",
    "candidates.jsonl",
    "imported.jsonl",
    "rejected.jsonl",
    "qwen_suggestions.jsonl",
)


@dataclass(frozen=True)
class HarvestCatalog:
    sources: tuple[SourceRecord, ...]
    candidates: tuple[HarvestCandidate, ...]
    harvested_count: int
    accepted_count: int
    rejected_count: int
    created_at: str
    updated_at: str


def candidate_to_dict(candidate: HarvestCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_id": candidate.source_id,
        "source_path": candidate.source_path,
        "extracted_path": str(candidate.extracted_path),
        "relative_path": candidate.relative_path,
        "image_sha256": candidate.image_sha256,
        "width": candidate.width,
        "height": candidate.height,
        "mode": candidate.mode,
        "status": candidate.status,
        "rejection_reasons": list(candidate.rejection_reasons),
        "warnings": list(candidate.warnings),
    }


def candidate_from_dict(data: Mapping[str, Any]) -> HarvestCandidate:
    payload = dict(data)
    payload["extracted_path"] = Path(payload["extracted_path"])
    payload["rejection_reasons"] = tuple(payload.get("rejection_reasons", ()))
    payload["warnings"] = tuple(payload.get("warnings", ()))
    return HarvestCandidate(**payload)


def harvested_to_record(sprite: HarvestedSprite) -> dict[str, Any]:
    """One JSONL-friendly record per harvested sprite (no image bytes)."""

    item = sprite.final_item
    return {
        "sprite_id": item.sprite_id,
        "candidate_id": sprite.candidate.candidate_id,
        "source_id": sprite.source.source_id,
        "final_png_path": str(item.source_path),
        "relative_path": sprite.candidate.relative_path,
        "status": item.status,
        "category": item.category,
        "tags": list(item.tags),
        "notes": item.notes,
        "source_name": item.source_name,
        "license": item.license,
        "author": item.author,
        "palette_size": item.palette_size,
        "has_role_map": item.has_role_map,
        "errors": list(sprite.imported.errors),
        "warnings": list(sprite.imported.warnings),
        "auto_metadata": _json_safe(sprite.auto_metadata),
    }


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_sources_jsonl(run_dir: str | Path, sources: Sequence[SourceRecord]) -> Path:
    return write_jsonl(Path(run_dir) / "sources.jsonl", [source_record_to_dict(s) for s in sources])


def write_candidates_jsonl(run_dir: str | Path, candidates: Sequence[HarvestCandidate]) -> Path:
    return write_jsonl(Path(run_dir) / "candidates.jsonl", [candidate_to_dict(c) for c in candidates])


def write_imported_jsonl(run_dir: str | Path, harvested: Sequence[HarvestedSprite]) -> Path:
    run_dir = Path(run_dir)
    valid = [s for s in harvested if s.imported.bundle is not None]
    rejected = [s for s in harvested if s.imported.bundle is None]
    write_jsonl(run_dir / "rejected.jsonl", [harvested_to_record(s) for s in rejected])
    return write_jsonl(run_dir / "imported.jsonl", [harvested_to_record(s) for s in valid])


def load_harvest_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a run's JSONL state without loading any image data."""

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"harvest run directory not found: {run_dir}")
    return {
        "run_dir": run_dir,
        "sources": [source_record_from_dict(d) for d in read_jsonl(run_dir / "sources.jsonl")],
        "candidates": [candidate_from_dict(d) for d in read_jsonl(run_dir / "candidates.jsonl")],
        "imported": read_jsonl(run_dir / "imported.jsonl"),
        "rejected": read_jsonl(run_dir / "rejected.jsonl"),
        "qwen_suggestions": read_jsonl(run_dir / "qwen_suggestions.jsonl"),
    }


def append_harvest_event(run_dir: str | Path, event: str, details: Mapping[str, Any] | None = None) -> None:
    """Append one timestamped event line to ``events.jsonl``."""

    path = Path(run_dir) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "at": utc_timestamp(), **(dict(details) if details else {})}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_catalog(
    sources: Sequence[SourceRecord],
    candidates: Sequence[HarvestCandidate],
    harvested: Sequence[HarvestedSprite],
) -> HarvestCatalog:
    now = utc_timestamp()
    return HarvestCatalog(
        sources=tuple(sources),
        candidates=tuple(candidates),
        harvested_count=len(harvested),
        accepted_count=sum(1 for s in harvested if s.final_item.status == "accepted"),
        rejected_count=sum(1 for s in harvested if s.final_item.status == "rejected"),
        created_at=now,
        updated_at=now,
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))
