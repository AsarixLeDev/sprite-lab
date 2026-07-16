"""Deterministic, resumable shard and merge orchestration for Labeling v4."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.pipeline import LabelV4PipelineConfig, label_record_v4

BATCH_SCHEMA_VERSION = "label_batch_v4.1"


@dataclass(frozen=True)
class BatchShardResult:
    shard_index: int
    shard_count: int
    selected_records: int
    completed_records: int
    resumed_records: int
    failure_records: int
    output_path: Path
    failure_path: Path
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BATCH_SCHEMA_VERSION,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "selected_records": self.selected_records,
            "completed_records": self.completed_records,
            "resumed_records": self.resumed_records,
            "failure_records": self.failure_records,
            "output_path": str(self.output_path),
            "failure_path": str(self.failure_path),
            "manifest_path": str(self.manifest_path),
        }


def run_label_v4_shard(
    input_path: str | Path,
    output_root: str | Path,
    *,
    shard_index: int,
    shard_count: int,
    config: LabelV4PipelineConfig | None = None,
    vlm_provider: Any | None = None,
    text_provider: Any | None = None,
    verifier_provider: Any | None = None,
    workers: int = 1,
    resume: bool = True,
    max_records: int | None = None,
) -> BatchShardResult:
    """Run one stable shard and atomically checkpoint canonical JSONL output."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    input_path = Path(input_path)
    output_root = Path(output_root)
    if input_path.resolve() == output_root.resolve():
        raise ValueError("output_root must not overwrite the input artifact")
    rows = _read_jsonl(input_path)
    selected = [row for row in rows if stable_shard(str(row.get("sprite_id", "")), shard_count) == shard_index]
    if max_records is not None:
        selected = selected[: max(0, int(max_records))]
    basename = f"shard_{shard_index:05d}_of_{shard_count:05d}"
    output_path = output_root / "shards" / f"{basename}.jsonl"
    failure_path = output_root / "failures" / f"{basename}.jsonl"
    manifest_path = output_root / "manifests" / f"{basename}.json"
    existing = _read_jsonl(output_path) if resume and output_path.is_file() else []
    completed = {str(row.get("sprite_id", "")): row for row in existing}
    pending = [row for row in selected if str(row.get("sprite_id", "")) not in completed]
    pipeline_config = config or LabelV4PipelineConfig(
        mode="A",
        cache_dir=output_root / "cache_v1",
        use_cache=True,
    )

    def process(record: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        sprite_id = str(record.get("sprite_id", ""))
        try:
            output = label_record_v4(
                record,
                config=pipeline_config,
                vlm_provider=vlm_provider,
                text_provider=text_provider,
                verifier_provider=verifier_provider,
            )
            return sprite_id, output, None
        except Exception as exc:  # failure isolation is part of the batch contract
            return (
                sprite_id,
                None,
                {
                    "schema_version": BATCH_SCHEMA_VERSION,
                    "sprite_id": sprite_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:240],
                    "retryable": isinstance(exc, (TimeoutError, OSError)),
                    "input_record_hash": _stable_hash(record),
                },
            )

    failures: list[dict[str, Any]] = []
    if max(1, int(workers)) == 1:
        results = [process(row) for row in pending]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {executor.submit(process, row): str(row.get("sprite_id", "")) for row in pending}
            for future in as_completed(futures):
                results.append(future.result())
    for sprite_id, output, failure in results:
        if output is not None:
            completed[sprite_id] = output
        if failure is not None:
            failures.append(failure)

    order = {str(row.get("sprite_id", "")): index for index, row in enumerate(selected)}
    canonical = sorted(
        completed.values(),
        key=lambda row: (order.get(str(row.get("sprite_id", "")), 10**12), str(row.get("sprite_id", ""))),
    )
    failures.sort(key=lambda row: (order.get(str(row.get("sprite_id", "")), 10**12), str(row.get("sprite_id", ""))))
    _atomic_write_jsonl(output_path, canonical)
    _atomic_write_jsonl(failure_path, failures)
    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "input_path": str(input_path),
        "input_sha256": _file_hash(input_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selection": "sha256(sprite_id) modulo shard_count",
        "selected_sprite_ids": [str(row.get("sprite_id", "")) for row in selected],
        "selected_count": len(selected),
        "completed_count": len(canonical),
        "resumed_count": len(existing),
        "failure_count": len(failures),
        "workers": max(1, int(workers)),
        "resume": bool(resume),
        "pipeline_mode": pipeline_config.mode,
        "output_sha256": _file_hash(output_path),
        "failure_sha256": _file_hash(failure_path),
        "paid_inference_enabled_by_runner": False,
    }
    _atomic_write_json(manifest_path, manifest)
    return BatchShardResult(
        shard_index=shard_index,
        shard_count=shard_count,
        selected_records=len(selected),
        completed_records=len(canonical),
        resumed_records=len(existing),
        failure_records=len(failures),
        output_path=output_path,
        failure_path=failure_path,
        manifest_path=manifest_path,
    )


def merge_label_v4_shards(
    output_root: str | Path,
    *,
    shard_count: int,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge a complete shard set into one sprite-id-sorted immutable artifact."""

    output_root = Path(output_root)
    target = Path(output_path) if output_path is not None else output_root / "label_v4_records.jsonl"
    rows: list[dict[str, Any]] = []
    shard_hashes: dict[str, str] = {}
    for index in range(int(shard_count)):
        path = output_root / "shards" / f"shard_{index:05d}_of_{shard_count:05d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing Labeling-v4 shard: {path}")
        shard_hashes[path.name] = _file_hash(path)
        rows.extend(_read_jsonl(path))
    ids = [str(row.get("sprite_id", "")) for row in rows]
    duplicates = sorted(sprite_id for sprite_id, count in _counts(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sprite ids across shards: {duplicates[:10]}")
    canonical = sorted(rows, key=lambda row: str(row.get("sprite_id", "")))
    _atomic_write_jsonl(target, canonical)
    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "shard_count": int(shard_count),
        "record_count": len(canonical),
        "shard_hashes": shard_hashes,
        "output_path": str(target),
        "output_sha256": _file_hash(target),
        "canonical_order": "sprite_id",
    }
    _atomic_write_json(target.with_suffix(".manifest.json"), manifest)
    return manifest


def stable_shard(sprite_id: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(str(sprite_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(shard_count)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True, default=str) + "\n" for row in rows).encode("utf-8")
    _atomic_write(path, payload)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result
