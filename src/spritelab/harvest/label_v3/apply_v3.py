"""Auto-Labeling v3: explicit, versioned apply / export to a NEW output.

Applying v3 never mutates historical artifacts. It:

* defaults to dry-run (reports what *would* be written, writes nothing);
* only ever writes *new* filenames (``label_v3_suggestions.jsonl`` sidecar and a
  migration report) — never ``imported.jsonl``, ``manifest_*.jsonl`` or any
  Labeling v2 output;
* only exposes ``auto_accept`` / ``partial_accept`` records, and only their
  genuinely accepted fields (via the legacy adapter, which never claims masked
  or abstained fields as accepted);
* excludes ``quarantine`` / ``hard_reject`` / ``unknown`` records entirely;
* emits a migration report enumerating every applied record and field so the
  change is auditable and reversible (delete the new output to roll back).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v3.adapter import build_legacy_safe_fused_label
from spritelab.harvest.label_v3.record_decisions import record_decision_from_json
from spritelab.utils.jsonl import iter_jsonl

logger = logging.getLogger(__name__)

SUGGESTIONS_NAME = "label_v3_suggestions.jsonl"
MIGRATION_REPORT_JSON = "v3_migration_report.json"
MIGRATION_REPORT_MD = "v3_migration_report.md"

APPLIED_STATES = ("auto_accept", "partial_accept")


@dataclass
class ApplyReport:
    output_root: str
    dry_run: bool
    total_records: int = 0
    applied: int = 0
    excluded: int = 0
    state_counts: dict[str, int] = field(default_factory=dict)
    applied_field_counts: dict[str, int] = field(default_factory=dict)
    written_files: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "dry_run": self.dry_run,
            "total_records": self.total_records,
            "applied": self.applied,
            "excluded": self.excluded,
            "state_counts": dict(self.state_counts),
            "applied_field_counts": dict(self.applied_field_counts),
            "written_files": list(self.written_files),
        }


def apply_v3_records(
    records_path: str | Path,
    output_root: str | Path,
    *,
    dry_run: bool = True,
    include_partial: bool = True,
    force: bool = False,
) -> ApplyReport:
    """Apply v3 decisions to a new output root (dry-run by default).

    Returns an :class:`ApplyReport`. In dry-run mode nothing is written.
    """
    records_path = Path(records_path)
    out_root = Path(output_root)
    report = ApplyReport(output_root=str(out_root), dry_run=dry_run)

    if not records_path.is_file():
        raise FileNotFoundError(f"v3 records not found: {records_path}")

    # Never write into a location that holds historical harvest/v2 artifacts.
    for protected in ("imported.jsonl", "label_v2_suggestions.jsonl"):
        if (out_root / protected).exists() and not force:
            raise ValueError(
                f"refusing to apply into {out_root} — it contains historical '{protected}'. "
                "Choose a fresh --output-root (or pass force=True to a non-historical dir)."
            )

    suggestions_path = out_root / SUGGESTIONS_NAME
    if suggestions_path.exists() and not force and not dry_run:
        raise ValueError(f"{suggestions_path} already exists; pass force=True to overwrite the v3 sidecar.")

    state_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    applied_rows: list[dict[str, Any]] = []
    migration_rows: list[dict[str, Any]] = []

    for row in iter_jsonl(records_path):
        report.total_records += 1
        record = record_decision_from_json(row)
        state_counts[record.record_state] += 1

        if record.record_state not in APPLIED_STATES:
            report.excluded += 1
            continue
        if record.record_state == "partial_accept" and not include_partial:
            report.excluded += 1
            continue

        report.applied += 1
        accepted_fields = list(record.accepted_fields)
        for fname in accepted_fields:
            field_counts[fname] += 1

        applied_rows.append(
            {
                "sprite_id": record.sprite_id,
                "label_v3": build_legacy_safe_fused_label(record, include_partial=include_partial),
            }
        )
        migration_rows.append(
            {
                "sprite_id": record.sprite_id,
                "record_state": record.record_state,
                "applied_fields": accepted_fields,
                "policy_hash": record.policy_hash,
            }
        )

    report.state_counts = dict(state_counts)
    report.applied_field_counts = dict(field_counts)

    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        with suggestions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for r in applied_rows:
                handle.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        report.written_files.append(str(suggestions_path))

        migration = {
            "summary": report.to_json_dict(),
            "records": migration_rows,
        }
        (out_root / MIGRATION_REPORT_JSON).write_text(
            json.dumps(migration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out_root / MIGRATION_REPORT_MD).write_text(format_migration_md(report), encoding="utf-8")
        report.written_files.append(str(out_root / MIGRATION_REPORT_JSON))
        report.written_files.append(str(out_root / MIGRATION_REPORT_MD))

    return report


def format_migration_md(report: ApplyReport) -> str:
    lines = [
        "# Auto-Labeling v3 Migration Report",
        "",
        f"Output root: {report.output_root}",
        f"Dry run: {report.dry_run}",
        "",
        f"- Total v3 records: {report.total_records}",
        f"- Applied (auto/partial): {report.applied}",
        f"- Excluded (quarantine/hard_reject/unknown/partial-off): {report.excluded}",
        "",
        "## Record states",
    ]
    for state, count in sorted(report.state_counts.items()):
        lines.append(f"- {state}: {count}")
    lines.append("")
    lines.append("## Applied field counts")
    for fname, count in sorted(report.applied_field_counts.items()):
        lines.append(f"- {fname}: {count}")
    lines.append("")
    lines.append("Rollback: delete the new output root. No historical artifact was modified.")
    return "\n".join(lines) + "\n"
