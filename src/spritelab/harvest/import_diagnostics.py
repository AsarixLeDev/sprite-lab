"""Read-only diagnostics for empty or import-broken harvest runs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.catalog import read_jsonl

RUN_FILES = ("sources.jsonl", "candidates.jsonl", "imported.jsonl", "rejected.jsonl", "events.jsonl")


def build_import_diagnostics(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    sources = read_jsonl(run_path / "sources.jsonl")
    candidates = read_jsonl(run_path / "candidates.jsonl")
    imported = read_jsonl(run_path / "imported.jsonl")
    rejected = read_jsonl(run_path / "rejected.jsonl")
    events = read_jsonl(run_path / "events.jsonl")
    rejection_counts, rejection_examples = _rejection_reasons(candidates, rejected)
    missing_files = _missing_files(run_path, candidates, imported, rejected)
    invalid_sizes = _invalid_image_sizes(candidates)
    over_color = _matching_reasons(rejection_counts, ("palette", "color", "colour", "over-color", "over_color"))
    license_gate = _matching_reasons(rejection_counts, ("license", "licence", "cc0", "attribution"))
    slicing = _slicing_diagnostics(candidates, rejection_counts)

    return {
        "run_dir": str(run_path),
        "files": {
            name: {
                "exists": (run_path / name).exists(),
                "bytes": (run_path / name).stat().st_size if (run_path / name).exists() else 0,
            }
            for name in RUN_FILES
        },
        "sources_jsonl_exists": (run_path / "sources.jsonl").is_file(),
        "candidates_jsonl_exists": (run_path / "candidates.jsonl").is_file(),
        "imported_jsonl_exists": (run_path / "imported.jsonl").is_file(),
        "rejected_jsonl_exists": (run_path / "rejected.jsonl").is_file(),
        "harvest_events": events,
        "sources": sources,
        "source_archive_path": _first_source_value(sources, "local_archive_path"),
        "source_root_path": _first_source_value(sources, "local_root_path"),
        "source_url": _first_source_value(sources, "source_url") or _first_source_value(sources, "download_url"),
        "candidate_count": len(candidates),
        "imported_count": len(imported),
        "rejected_count": len(rejected),
        "top_rejection_reasons": dict(rejection_counts.most_common(25)),
        "rejection_examples": rejection_examples[:20],
        "invalid_image_sizes": invalid_sizes[:20],
        "over_color_reasons": over_color,
        "spritesheet_slicing_issue": slicing,
        "missing_files": missing_files[:20],
        "bad_license_gate_reasons": license_gate,
        "recommended_reimport_command": _infer_reimport_command(sources, run_path),
    }


def write_import_diagnostics_reports(report: Mapping[str, Any], *, out_md: Path | None, out_json: Path | None) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(format_import_diagnostics(report), encoding="utf-8")


def format_import_diagnostics(report: Mapping[str, Any]) -> str:
    lines = [
        "# Import Diagnostics",
        "",
        f"Run: `{report.get('run_dir', '')}`",
        f"Candidates: {int(report.get('candidate_count', 0))}",
        f"Imported: {int(report.get('imported_count', 0))}",
        f"Rejected: {int(report.get('rejected_count', 0))}",
        "",
        "## Files",
    ]
    for name, info in dict(report.get("files") or {}).items():
        info = dict(info or {})
        lines.append(f"- {name}: exists={bool(info.get('exists'))} bytes={int(info.get('bytes', 0))}")
    lines.extend(["", "## Source"])
    lines.append(f"- archive/path/url: `{report.get('source_archive_path') or report.get('source_root_path') or report.get('source_url') or ''}`")
    command = str(report.get("recommended_reimport_command") or "")
    lines.append(f"- recommended re-import command: `{command}`" if command else "- recommended re-import command: (not inferable)")
    lines.extend(["", "## Rejection Reasons"])
    reasons = dict(report.get("top_rejection_reasons") or {})
    if reasons:
        for reason, count in reasons.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- (none)")
    for title, key in (
        ("Invalid Image Sizes", "invalid_image_sizes"),
        ("Over-Color Reasons", "over_color_reasons"),
        ("Missing Files", "missing_files"),
        ("Bad License Gate Reasons", "bad_license_gate_reasons"),
    ):
        lines.extend(["", f"## {title}"])
        values = report.get(key) or ()
        if isinstance(values, Mapping):
            values = [f"{name}: {count}" for name, count in values.items()]
        if values:
            for value in list(values)[:20]:
                lines.append(f"- {value}")
        else:
            lines.append("- (none)")
    lines.extend(["", "## Spritesheet Slicing"])
    slicing = dict(report.get("spritesheet_slicing_issue") or {})
    for key, value in slicing.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Events"])
    for event in report.get("harvest_events") or ():
        lines.append(f"- {dict(event)}")
    return "\n".join(lines) + "\n"


def _rejection_reasons(
    candidates: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]
) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    examples: list[str] = []
    for record in [*candidates, *rejected]:
        reasons = [
            *[str(value) for value in record.get("rejection_reasons") or () if str(value)],
            *[str(value) for value in record.get("errors") or () if str(value)],
        ]
        for reason in reasons:
            counts[reason] += 1
            if len(examples) < 50:
                examples.append(f"{record.get('candidate_id') or record.get('sprite_id') or '?'}: {reason}")
    return counts, examples


def _invalid_image_sizes(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    examples: list[str] = []
    for record in candidates:
        width = int(record.get("width", 0) or 0)
        height = int(record.get("height", 0) or 0)
        if width and height and (width != 32 or height != 32):
            examples.append(f"{record.get('candidate_id', '?')}: {width}x{height}")
    return examples


def _missing_files(
    run_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    imported: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
) -> list[str]:
    examples: list[str] = []
    for record in [*candidates, *imported, *rejected]:
        for field in ("extracted_path", "final_png_path"):
            raw = str(record.get(field, "")).strip()
            if not raw:
                continue
            if not _resolve_existing(raw, run_path=run_path):
                examples.append(f"{record.get('candidate_id') or record.get('sprite_id') or '?'}: {field}={raw}")
                break
    return examples


def _resolve_existing(raw: str, *, run_path: Path) -> Path | None:
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, run_path / path, run_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _matching_reasons(counts: Mapping[str, int], terms: Sequence[str]) -> dict[str, int]:
    lowered_terms = tuple(str(term).lower() for term in terms)
    return {
        reason: int(count)
        for reason, count in counts.items()
        if any(term in str(reason).lower() for term in lowered_terms)
    }


def _slicing_diagnostics(candidates: Sequence[Mapping[str, Any]], rejection_counts: Mapping[str, int]) -> dict[str, Any]:
    sheet_like = 0
    for record in candidates:
        width = int(record.get("width", 0) or 0)
        height = int(record.get("height", 0) or 0)
        if width > 32 or height > 32:
            sheet_like += 1
    slicing_reasons = _matching_reasons(rejection_counts, ("sheet", "slice", "tile"))
    return {
        "sheet_like_candidate_count": sheet_like,
        "slicing_reasons": slicing_reasons,
        "likely_issue": bool(sheet_like and slicing_reasons),
    }


def _first_source_value(sources: Sequence[Mapping[str, Any]], key: str) -> str:
    for source in sources:
        value = str(source.get(key, "")).strip()
        if value:
            return value
    return ""


def _infer_reimport_command(sources: Sequence[Mapping[str, Any]], run_path: Path) -> str:
    if not sources:
        return ""
    source = dict(sources[0])
    common = (
        f"--run-name {run_path.name} --run-root {run_path.parent} "
        f"--source-id {source.get('source_id', '')} --source-name \"{source.get('source_name', '')}\" "
        f"--license {dict(source.get('license') or {}).get('license', source.get('license', 'unknown'))} "
        "--user-confirmed-license"
    )
    archive = str(source.get("local_archive_path", "")).strip()
    root = str(source.get("local_root_path", "")).strip()
    url = str(source.get("download_url") or source.get("source_url") or "").strip()
    if archive:
        return f"python -m spritelab harvest import-zip --zip \"{archive}\" {common}"
    if root:
        return f"python -m spritelab harvest import-dir --dir \"{root}\" {common}"
    if url:
        return f"python -m spritelab harvest download-zip --url \"{url}\" {common}"
    return ""
