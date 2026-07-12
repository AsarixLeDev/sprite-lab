"""Auto-Labeling v3: frozen suite creation CLI.

Selects records from harvest runs and creates disjoint frozen evaluation
partitions: calibration, development, frozen_in_domain_test,
frozen_unseen_pack_test, frozen_source_ood_test, frozen_open_set_test.
"""

from __future__ import annotations

import argparse
import random as _random
from pathlib import Path
from typing import Any

from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.label_v3.frozen_suites_v3 import (
    SUITE_SCHEMA_VERSION,
    FrozenSuiteManifest,
    LeakageReport,
    check_suite_leakage,
)
from spritelab.harvest.sources import utc_timestamp

# Partition names for frozen suites.
ALL_PARTITIONS = (
    "calibration",
    "development",
    "frozen_in_domain_test",
    "frozen_unseen_pack_test",
    "frozen_source_ood_test",
    "frozen_open_set_test",
)

# Fractions for each partition (must sum to <= 1.0)
DEFAULT_FRACTIONS = (0.25, 0.25, 0.20, 0.15, 0.10, 0.05)


def _get_variant_group(record: dict[str, Any]) -> str:
    """Return a variant group key from sprite_id by stripping trailing
    recoloring digits. Same-sheet recolors share the same prefix."""
    sid = str(record.get("sprite_id", ""))
    if not sid:
        return sid
    parts = sid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return sid


def _get_source_group(record: dict[str, Any]) -> str:
    """Return source group from source_id or source_name."""
    return str(record.get("source_id", record.get("source_name", "")))


def _get_sheet_group(record: dict[str, Any]) -> str:
    """Extract sheet name from relative_path."""
    rel = str(record.get("relative_path", ""))
    if not rel:
        return _get_source_group(record)
    parts = Path(rel).parts
    if parts:
        base = parts[0]
        if base.endswith(".png") or "__" in base:
            return base.rsplit("__", 1)[0] if "__" in base else Path(base).stem
        return base
    return _get_source_group(record)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "freeze-v3-suite",
        help="Create disjoint frozen evaluation suites from harvest runs.",
    )
    p.add_argument("--runs", nargs="+", required=True, type=Path, help="One or more harvest run directories")
    p.add_argument("--suite-name", default="v3_frozen", help="Suite name prefix")
    p.add_argument("--suite-dir", default=None, type=Path, help="Directory for frozen suite output files")
    p.add_argument("--seed", type=int, default=496, help="Deterministic random seed")
    p.add_argument("--n", type=int, default=None, help="Maximum total sample size")
    p.add_argument(
        "--fractions",
        nargs=6,
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Six partition fractions: calibration development in_domain unseen_pack source_ood open_set",
    )
    p.add_argument("--include-status", action="append", default=["accepted"])
    p.add_argument(
        "--no-leakage-check",
        action="store_true",
        default=False,
        help="Skip leakage check (not recommended)",
    )
    p.set_defaults(func=_run_freeze_v3_suite)


def _run_freeze_v3_suite(parsed: argparse.Namespace) -> None:

    rng = _random.Random(parsed.seed)

    # Collect all records from all runs
    all_records: list[dict[str, Any]] = []
    run_source_sets: dict[str, set[str]] = {}
    for run_dir in parsed.runs:
        run_path = Path(run_dir)
        imported_path = run_path / "imported.jsonl"
        if not imported_path.is_file():
            print(f"Warning: no imported.jsonl in {run_path}, skipping")
            continue
        records = read_jsonl(imported_path)
        include_statuses = {s.strip().lower() for s in (parsed.include_status or ["accepted"])}
        for r in records:
            sid = str(r.get("sprite_id", ""))
            status = str(r.get("status", "accepted")).strip().lower()
            if status not in include_statuses:
                continue
            r["_run_dir"] = str(run_path)
            r["_run_name"] = run_path.name
            all_records.append(r)
            src = _get_source_group(r)
            run_source_sets.setdefault(run_path.name, set()).add(src)

    if not all_records:
        raise SystemExit("No records found across specified runs")

    n_total = parsed.n if parsed.n is not None else len(all_records)
    n_total = min(n_total, len(all_records))
    fractions = parsed.fractions

    # Build sprite sets
    ids_by_source: dict[str, list[str]] = {}
    ids_by_variant: dict[str, list[str]] = {}
    ids_by_sheet: dict[str, list[str]] = {}
    record_by_id: dict[str, dict[str, Any]] = {}

    for r in all_records:
        sid = str(r["sprite_id"])
        record_by_id[sid] = r
        src = _get_source_group(r)
        vg = _get_variant_group(r)
        sheet = _get_sheet_group(r)
        ids_by_source.setdefault(src, []).append(sid)
        ids_by_variant.setdefault(vg, []).append(sid)
        ids_by_sheet.setdefault(sheet, []).append(sid)

    source_ids = list(ids_by_source.keys())
    rng.shuffle(source_ids)

    # Assign sources to partitions
    calibration_sources: list[str] = []
    development_sources: list[str] = []
    eval_sources: list[str] = []

    n_sources = len(source_ids)
    idx = 0
    n_calib = max(1, int(n_sources * fractions[0]))
    n_dev = max(1, int(n_sources * fractions[1]))
    n_eval = max(1, n_sources - n_calib - n_dev)

    calibration_sources = source_ids[idx : idx + n_calib]
    idx += n_calib
    development_sources = source_ids[idx : idx + n_dev]
    idx += n_dev
    eval_sources = source_ids[idx : idx + n_eval]

    # Collect sprite IDs per partition
    def _collect_ids(sources: list[str], max_per_source: int = 200) -> list[str]:
        result: list[str] = []
        for src in sources:
            sids = list(ids_by_source.get(src, []))
            rng.shuffle(sids)
            result.extend(sids[:max_per_source])
        return result

    calib_ids = _collect_ids(calibration_sources, 50)
    dev_ids = _collect_ids(development_sources, 200)

    # Split eval sources into in-domain and unseen
    n_eval_src = len(eval_sources)
    n_in_domain = max(1, int(n_eval_src * 0.5))
    n_unseen = max(1, int(n_eval_src * 0.3))
    n_ood = max(1, n_eval_src - n_in_domain - n_unseen)
    rng.shuffle(eval_sources)
    in_domain_src = eval_sources[:n_in_domain]
    unseen_src = eval_sources[n_in_domain : n_in_domain + n_unseen]
    ood_src = eval_sources[n_in_domain + n_unseen : n_in_domain + n_unseen + n_ood]
    open_set_src = eval_sources[n_in_domain + n_unseen + n_ood :][: max(1, n_ood // 2)]

    in_domain_ids = _collect_ids(in_domain_src, 100)
    unseen_ids = _collect_ids(unseen_src, 100)
    ood_ids = _collect_ids(ood_src, 100)
    open_set_ids = _collect_ids(open_set_src, 50) if open_set_src else []

    # Build partitions map
    partitions: dict[str, tuple[str, ...]] = {
        "calibration": tuple(calib_ids[:n_total]),
        "development": tuple(dev_ids[:n_total]),
        "frozen_in_domain_test": tuple(in_domain_ids),
        "frozen_unseen_pack_test": tuple(unseen_ids),
        "frozen_source_ood_test": tuple(ood_ids),
    }
    if open_set_ids:
        partitions["frozen_open_set_test"] = tuple(open_set_ids)

    # Run leakage check
    manifest = FrozenSuiteManifest(
        suite_name=parsed.suite_name,
        taxonomy_version="v3.1.0",
        annotation_guidance="created from freeze-v3-suite CLI",
        created_at=utc_timestamp(),
        partitions=partitions,
    )

    report = check_suite_leakage(manifest)
    if not report.ok:
        _print_leakage_report(report)
        if not parsed.no_leakage_check:
            raise SystemExit("Frozen suite has leakage. Review the report above or pass --no-leakage-check.")
        print("WARNING: --no-leakage-check bypassed leakage failures.")

    # Save manifest
    suite_dir = Path(parsed.suite_dir) if parsed.suite_dir else Path(parsed.runs[0])
    suite_dir.mkdir(parents=True, exist_ok=True)
    json_path = suite_dir / f"{parsed.suite_name}_suite.json"
    manifest.save(json_path)

    # Save Markdown summary
    md_lines = [
        f"# Frozen Suite: {parsed.suite_name}",
        "",
        f"- **Schema**: {SUITE_SCHEMA_VERSION}",
        "- **Taxonomy**: v3.1.0",
        f"- **Seed**: {parsed.seed}",
        f"- **Created**: {manifest.created_at}",
        f"- **Total records scanned**: {len(all_records)}",
        "",
        "## Partitions",
    ]
    for part_name in partitions:
        md_lines.append(f"### {part_name}")
        md_lines.append(f"- Count: {len(partitions[part_name])}")
        md_lines.append("")
    md_lines.append("## Leakage")
    md_lines.append(f"- OK: {report.ok}")
    if report.cross_partition_overlaps:
        md_lines.append("- Cross-partition overlaps detected")
    if report.tuning_overlap:
        md_lines.append(f"- Tuning overlap: {len(report.tuning_overlap)} IDs")

    md_path = suite_dir / f"{parsed.suite_name}_suite.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Suite manifest: {json_path}")
    print(f"Suite report: {md_path}")
    print(f"Leakage OK: {report.ok}")
    for part_name in partitions:
        print(f"  {part_name}: {len(partitions[part_name])} IDs")


def _print_leakage_report(report: LeakageReport) -> None:
    print(f"LEAKAGE REPORT for: {report.suite_name}")
    print(f"  OK: {report.ok}")
    if report.cross_partition_overlaps:
        print("  Cross-partition overlaps:")
        for pair, ids in report.cross_partition_overlaps.items():
            print(f"    {pair}: {len(ids)} shared IDs ({', '.join(ids[:5])}...)")
    if report.tuning_overlap:
        print(f"  Tuning overlap ({len(report.tuning_overlap)} IDs): {', '.join(report.tuning_overlap[:10])}...")
