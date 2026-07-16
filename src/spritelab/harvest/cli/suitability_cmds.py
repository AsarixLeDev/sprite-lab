"""CLI registration for the read-only sprite suitability gate."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace, _SubParsersAction
from collections import Counter
from pathlib import Path
from typing import Any


def register(subparsers: _SubParsersAction) -> None:
    parser: ArgumentParser = subparsers.add_parser(
        "suitability-audit", help="Audit sprite visual suitability without changing source files."
    )
    parser.add_argument("--run", action="append", type=Path, default=[], help="Harvest run directory; repeatable.")
    parser.add_argument("--runs", default="", help="Comma-separated harvest run directories.")
    parser.add_argument("--profile", default="single_object_32px")
    parser.add_argument("--config", type=Path, help="Optional versioned JSON threshold overrides.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Explicitly confirm read-only operation (the audit is always read-only).",
    )
    parser.add_argument("--limit", type=int, help="Deterministic per-run diagnostic limit.")
    parser.set_defaults(func=_run)


def _run(parsed: Namespace) -> None:
    from spritelab.harvest.suitability import audit_inputs, load_config, write_audit_output

    run_dirs = [*parsed.run, *(Path(value.strip()) for value in str(parsed.runs).split(",") if value.strip())]
    if not run_dirs:
        raise ValueError("provide at least one --run or --runs value")
    resolved_runs = [_resolve_run(path) for path in run_dirs]
    inputs = _collect_inputs(resolved_runs, limit=parsed.limit)
    config = load_config(parsed.profile, parsed.config)
    output = audit_inputs(inputs, config)
    write_audit_output(output, parsed.out_dir)
    counts = output.summary["status_counts"]
    print(f"Audited: {output.summary['total']}")
    print(f"Accept: {counts['accept']}")
    print(f"Quarantine: {counts['quarantine']}")
    print(f"Reject: {counts['reject']}")
    print(f"Output: {parsed.out_dir}")


def _resolve_run(path: Path) -> Path:
    candidates = [path, Path("harvest_runs") / path]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"harvest run directory not found: {path}")


def _collect_inputs(run_dirs: list[Path], *, limit: int | None) -> list[Any]:
    from spritelab.harvest.catalog import read_jsonl
    from spritelab.harvest.suitability import SuitabilityInput

    records: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in sorted(run_dirs, key=lambda path: path.as_posix().lower()):
        manifest = [*read_jsonl(run_dir / "imported.jsonl"), *read_jsonl(run_dir / "rejected.jsonl")]
        represented_candidates = {str(record.get("candidate_id") or "") for record in manifest}
        for candidate in read_jsonl(run_dir / "candidates.jsonl"):
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id and candidate_id not in represented_candidates:
                manifest.append(
                    {
                        "sprite_id": candidate_id,
                        "final_png_path": candidate.get("extracted_path", ""),
                        "candidate_id": candidate_id,
                        "warnings": candidate.get("warnings", []),
                    }
                )
        manifest.sort(key=lambda record: (str(record.get("sprite_id") or ""), str(record.get("final_png_path") or "")))
        if limit is not None:
            if limit < 0:
                raise ValueError("--limit must be non-negative")
            manifest = manifest[:limit]
        records.extend((run_dir, dict(record)) for record in manifest)

    raw_ids = [str(record.get("sprite_id") or record.get("candidate_id") or "unnamed") for _, record in records]
    duplicate_ids = {sprite_id for sprite_id, count in Counter(raw_ids).items() if count > 1}
    inputs: list[SuitabilityInput] = []
    for run_dir, record in records:
        raw_id = str(record.get("sprite_id") or record.get("candidate_id") or "unnamed")
        sprite_id = f"{run_dir.name}::{raw_id}" if raw_id in duplicate_ids else raw_id
        raw_path = str(record.get("final_png_path") or record.get("path") or "")
        image_path = _resolve_image_path(raw_path, run_dir)
        resize_history = _resize_history(record)
        inputs.append(
            SuitabilityInput(
                sprite_id=sprite_id,
                image_path=image_path,
                resize_history=resize_history,
                source_run=run_dir.name,
            )
        )
    return inputs


def _resolve_image_path(raw: str, run_dir: Path) -> Path:
    if not raw:
        return run_dir / "__missing_image_path__"
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, run_dir / path, run_dir.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def _resize_history(record: dict[str, Any]) -> tuple[str, ...]:
    history: list[str] = []
    for warning in record.get("warnings") or ():
        text = str(warning)
        if "resize" in text.lower() or "pad" in text.lower():
            history.append(text)
    auto = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), dict) else {}
    mapping = auto.get("sheet_mapping") if isinstance(auto.get("sheet_mapping"), dict) else {}
    native = str(mapping.get("native_resolution") or "")
    if native:
        history.append(f"declared_native_resolution={native}")
    path = str(record.get("final_png_path") or "").lower()
    if "padded" in Path(path).parts:
        history.append("harvest_center_padding")
    return tuple(sorted(set(history)))
