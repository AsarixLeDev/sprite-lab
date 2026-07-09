"""Training-readiness gate for exported SpriteBundle datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.training.palette_report import PaletteSemanticsReport, PaletteSlotStats
from spritelab.training.splits import SplitAssignment


@dataclass(frozen=True)
class ReadinessIssue:
    severity: str
    code: str
    message: str
    sprite_id: str | None = None


@dataclass(frozen=True)
class TrainingReadinessReport:
    passed: bool
    issues: tuple[ReadinessIssue, ...]
    counts: dict[str, int]
    split_counts: dict[str, int]
    palette_size_counts: dict[int, int]


def build_training_readiness_report(
    export_records: Sequence[Mapping[str, Any]],
    split_assignment: SplitAssignment,
    palette_report: PaletteSemanticsReport,
    *,
    accepted_count: int,
    exported_count: int,
    invalid_accepted: Sequence[tuple[str, str]] = (),
    duplicate_leakage: Sequence[tuple[str, str, str]] = (),
) -> TrainingReadinessReport:
    """Build deterministic readiness issues and counts for an export."""

    issues: list[ReadinessIssue] = []
    records = list(export_records)
    if accepted_count == 0:
        issues.append(ReadinessIssue("error", "NO_ACCEPTED_SPRITES", "No accepted sprites are available."))
    if accepted_count > 0 and exported_count == 0:
        issues.append(
            ReadinessIssue("error", "ZERO_EXPORTED_SPRITES", "Accepted sprites exist but zero sprites were exported.")
        )
    for sprite_id, reason in invalid_accepted:
        issues.append(ReadinessIssue("error", "INVALID_ACCEPTED_BUNDLE", reason, sprite_id=sprite_id))
    for sprite_id, left_split, right_split in duplicate_leakage:
        issues.append(
            ReadinessIssue(
                "error",
                "DUPLICATE_LEAKAGE",
                f"Duplicate group crosses splits: {left_split} and {right_split}.",
                sprite_id=sprite_id,
            )
        )

    for record in records:
        palette_size = int(record.get("palette_size", 0))
        max_palette_slots = int(record.get("max_palette_slots", 10_000))
        if palette_size > max_palette_slots:
            issues.append(
                ReadinessIssue(
                    "error",
                    "PALETTE_TOO_LARGE",
                    f"Palette size {palette_size} exceeds max_palette_slots={max_palette_slots}.",
                    sprite_id=str(record.get("sprite_id")),
                )
            )
        if record.get("invalid_tokens"):
            issues.append(
                ReadinessIssue(
                    "error",
                    "INVALID_TOKEN_VALUES",
                    "Exported arrays contain invalid token values.",
                    sprite_id=str(record.get("sprite_id")),
                )
            )
        if record.get("missing_required_split_file"):
            issues.append(ReadinessIssue("error", "MISSING_SPLIT_FILE", "A required split file is missing."))
        if record.get("id_collision"):
            issues.append(ReadinessIssue("error", "BUNDLE_ID_COLLISION", "Metadata or bundle ID collision detected."))

    if exported_count < 500:
        issues.append(
            ReadinessIssue("warning", "TINY_DATASET", "Fewer than 500 sprites exported; useful mainly for smoke tests.")
        )
    elif exported_count < 2000:
        issues.append(
            ReadinessIssue(
                "warning", "SMALL_DATASET", "Fewer than 2000 sprites exported; likely narrow generation coverage."
            )
        )

    if records and not any(bool(record.get("dedupe_report_provided")) for record in records):
        issues.append(
            ReadinessIssue(
                "warning",
                "NO_DEDUPE_REPORT",
                "No dedupe report was provided; split cannot protect against unknown near-duplicate leakage.",
            )
        )
    if records and not any(bool(record.get("quality_report_provided")) for record in records):
        issues.append(ReadinessIssue("warning", "NO_QUALITY_REPORT", "No quality report was provided."))

    if records:
        unknown_categories = sum(1 for record in records if record.get("category") == "unknown")
        if unknown_categories / len(records) > 0.5:
            issues.append(
                ReadinessIssue(
                    "warning", "MANY_UNKNOWN_CATEGORIES", "More than 50% of exported sprites have unknown category."
                )
            )
        missing_roles = sum(1 for record in records if not bool(record.get("has_role_map")))
        if missing_roles / len(records) > 0.5:
            issues.append(
                ReadinessIssue(
                    "warning", "MANY_MISSING_ROLE_MAPS", "More than 50% of exported sprites have no original role_map."
                )
            )
        category_counts = Counter(str(record.get("category", "unknown")) for record in records)
        if category_counts:
            most_common = category_counts.most_common(1)[0][1]
            if len(category_counts) > 1 and most_common / len(records) > 0.9:
                issues.append(
                    ReadinessIssue("warning", "IMBALANCED_CATEGORIES", "Category distribution is very imbalanced.")
                )

    split_counts = {
        "train": len(split_assignment.train),
        "val": len(split_assignment.val),
        "test": len(split_assignment.test),
    }
    if exported_count >= 10:
        largest_split = max(split_counts.values())
        smallest_split = min(split_counts.values())
        if smallest_split == 0 or largest_split / max(1, smallest_split) > 20:
            issues.append(ReadinessIssue("warning", "IMBALANCED_SPLITS", "Split sizes are very imbalanced."))

    for slot in palette_report.slot_stats:
        if slot.role_entropy is not None and slot.pixel_count >= 10 and slot.role_entropy > 1.5:
            issues.append(
                ReadinessIssue(
                    "warning",
                    "HIGH_SLOT_ROLE_ENTROPY",
                    f"Palette slot {slot.slot_id} has high role entropy.",
                )
            )

    if _palette_size_variance_high(palette_report.palette_size_counts):
        issues.append(ReadinessIssue("warning", "HIGH_PALETTE_SIZE_VARIANCE", "Palette sizes vary widely."))

    status_counts = Counter(str(record.get("curation_status", "accepted")) for record in records)
    if status_counts.get("needs_fix", 0) + status_counts.get("quarantine", 0) > exported_count:
        issues.append(
            ReadinessIssue("warning", "HIGH_UNSTABLE_CURATION_COUNT", "Many sprites are quarantine or needs_fix.")
        )

    counts = {
        "accepted": int(accepted_count),
        "exported": int(exported_count),
        "excluded": max(0, int(accepted_count) - int(exported_count)),
        "issues": len(issues),
        "errors": sum(1 for issue in issues if issue.severity == "error"),
        "warnings": sum(1 for issue in issues if issue.severity == "warning"),
    }
    passed = counts["errors"] == 0
    return TrainingReadinessReport(
        passed=passed,
        issues=tuple(sorted(issues, key=lambda issue: (issue.severity, issue.code, issue.sprite_id or ""))),
        counts=counts,
        split_counts=split_counts,
        palette_size_counts=dict(sorted(palette_report.palette_size_counts.items())),
    )


def format_training_readiness_markdown(report: TrainingReadinessReport) -> str:
    """Render a training-readiness report as Markdown."""

    errors = [issue for issue in report.issues if issue.severity == "error"]
    warnings = [issue for issue in report.issues if issue.severity == "warning"]
    lines = [
        "# Training Readiness Report",
        "",
        "## Verdict",
        "",
        "PASSED" if report.passed else "FAILED",
        "",
        "## Counts",
        "",
        f"- Accepted: {report.counts.get('accepted', 0)}",
        f"- Exported: {report.counts.get('exported', 0)}",
        f"- Excluded: {report.counts.get('excluded', 0)}",
        f"- Train: {report.split_counts.get('train', 0)}",
        f"- Val: {report.split_counts.get('val', 0)}",
        f"- Test: {report.split_counts.get('test', 0)}",
        "",
        "## Hard errors",
        "",
    ]
    lines.extend(_issue_lines(errors))
    lines.extend(["", "## Warnings", ""])
    lines.extend(_issue_lines(warnings))
    lines.extend(["", "## Split safety", ""])
    lines.append(
        "- Duplicate leakage detected."
        if any(issue.code == "DUPLICATE_LEAKAGE" for issue in errors)
        else "- No duplicate leakage was reported."
    )
    lines.extend(["", "## Palette readiness", ""])
    lines.append(f"- Palette size distribution: {_format_distribution(report.palette_size_counts)}")
    high_entropy = [issue for issue in warnings if issue.code == "HIGH_SLOT_ROLE_ENTROPY"]
    lines.append(f"- High entropy slot warnings: {len(high_entropy)}")
    lines.extend(["", "## Category distribution", ""])
    lines.append("- See per-split manifests for category details.")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(_recommendations(report.issues))
    return "\n".join(lines)


def write_training_readiness_markdown(report: TrainingReadinessReport, path: Path) -> None:
    """Write a training-readiness report Markdown file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_training_readiness_markdown(report) + "\n", encoding="utf-8")


def build_readiness_report_from_export(export_dir: Path) -> TrainingReadinessReport:
    """Load an export directory and rebuild the readiness report."""

    export_path = Path(export_dir)
    records = _load_export_records(export_path)
    split_assignment = SplitAssignment(
        train=tuple(record["sprite_id"] for record in records if record.get("split") == "train"),
        val=tuple(record["sprite_id"] for record in records if record.get("split") == "val"),
        test=tuple(record["sprite_id"] for record in records if record.get("split") == "test"),
        group_by_sprite_id={
            str(record["sprite_id"]): str(record.get("dedupe_group") or record["sprite_id"]) for record in records
        },
    )
    palette_report = _load_palette_report(export_path / "palette_semantics_report.json")
    config = _load_export_config(export_path / "export_config.json")
    accepted_count = int(config.get("accepted_count", len(records)))
    missing_records: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_path = export_path / f"{split_name}.npz"
        if not split_path.exists():
            missing_records.append({"sprite_id": f"{split_name}.npz", "missing_required_split_file": True})
    return build_training_readiness_report(
        [*records, *missing_records],
        split_assignment,
        palette_report,
        accepted_count=accepted_count,
        exported_count=len(records),
    )


def _palette_size_variance_high(counts: Mapping[int, int]) -> bool:
    if not counts:
        return False
    sizes = list(counts)
    return max(sizes) - min(sizes) > 16


def _issue_lines(issues: Sequence[ReadinessIssue]) -> list[str]:
    if not issues:
        return ["- None"]
    return [
        f"- `{issue.code}`{f' ({issue.sprite_id})' if issue.sprite_id else ''}: {issue.message}" for issue in issues
    ]


def _format_distribution(values: Mapping[int, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in sorted(values.items()))


def _recommendations(issues: Sequence[ReadinessIssue]) -> list[str]:
    codes = {issue.code for issue in issues}
    recommendations: list[str] = []
    if "TINY_DATASET" in codes or "SMALL_DATASET" in codes or "NO_ACCEPTED_SPRITES" in codes:
        recommendations.append("- Curate more accepted sprites before broad training.")
    if "INVALID_ACCEPTED_BUNDLE" in codes:
        recommendations.append("- Fix or unaccept invalid accepted bundles.")
    if "NO_DEDUPE_REPORT" in codes or "DUPLICATE_LEAKAGE" in codes:
        recommendations.append("- Generate a dedupe report and keep duplicate groups within one split.")
    if "HIGH_SLOT_ROLE_ENTROPY" in codes or "HIGH_PALETTE_SIZE_VARIANCE" in codes:
        recommendations.append("- Improve palette canonicalization and role-map consistency.")
    if "MANY_UNKNOWN_CATEGORIES" in codes or "IMBALANCED_CATEGORIES" in codes:
        recommendations.append("- Add categories or curation tags for better dataset balance.")
    if not recommendations:
        recommendations.append("- Dataset passed current readiness checks.")
    return recommendations


def _load_export_records(export_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        path = export_dir / f"manifest_{split_name}.jsonl"
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return sorted(records, key=lambda record: str(record.get("sprite_id", "")))


def _load_palette_report(path: Path) -> PaletteSemanticsReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PaletteSemanticsReport(
        bundle_count=int(data["bundle_count"]),
        accepted_count=int(data["accepted_count"]),
        palette_size_counts={int(key): int(value) for key, value in data["palette_size_counts"].items()},
        slot_stats=[PaletteSlotStats(**record) for record in data["slot_stats"]],
        warnings=list(data["warnings"]),
    )


def _load_export_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the training-readiness report for an export directory.")
    parser.add_argument("--export", required=True, type=Path, dest="export_dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = build_readiness_report_from_export(args.export_dir)
    output_path = args.export_dir / "training_readiness_report.md"
    write_training_readiness_markdown(report, output_path)
    print("Verdict: " + ("PASSED" if report.passed else "FAILED"))
    print(f"Errors: {report.counts.get('errors', 0)}")
    print(f"Warnings: {report.counts.get('warnings', 0)}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
