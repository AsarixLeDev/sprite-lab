"""Markdown reporting for Dataset Maker imports and exports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from spritelab.dataset_maker.importer import ImportedSprite

if TYPE_CHECKING:
    from spritelab.dataset_maker.exporter import DatasetMakerExportResult


def build_dataset_maker_report(
    imported: Sequence[ImportedSprite],
    result: DatasetMakerExportResult | None = None,
) -> str:
    """Build a concise Markdown report for a Dataset Maker run."""

    status_counts = Counter(sprite.item.status for sprite in imported)
    accepted = status_counts["accepted"]
    rejected = status_counts["rejected"]
    needs_fix = status_counts["needs_fix"]
    quarantine = status_counts["quarantine"]
    exported = result.accepted_count if result is not None else 0
    train = result.train_count if result is not None else 0
    val = result.val_count if result is not None else 0
    test = result.test_count if result is not None else 0

    lines = [
        "# Dataset Maker Report",
        "",
        "## Summary",
        f"Imported: {len(imported)}",
        f"Accepted: {accepted}",
        f"Rejected: {rejected}",
        f"Needs fix: {needs_fix}",
        f"Quarantine: {quarantine}",
        f"Exported: {exported}",
        f"Train: {train}",
        f"Val: {val}",
        f"Test: {test}",
        "",
        "## Categories",
        *_counter_lines(Counter(sprite.item.category for sprite in imported)),
        "",
        "## Tags",
        *_counter_lines(Counter(tag for sprite in imported for tag in sprite.item.tags)),
        "",
        "## Palette sizes",
        *_counter_lines(Counter(_palette_size_label(sprite) for sprite in imported)),
        "",
        "## Warnings",
        *_warning_lines(imported, result),
        "",
        "## Rejected / excluded examples",
        *_excluded_lines(imported),
        "",
    ]
    return "\n".join(lines)


def _counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["None."]
    return [f"- {name}: {count}" for name, count in sorted(counter.items(), key=lambda item: (str(item[0]), item[1]))]


def _warning_lines(imported: Sequence[ImportedSprite], result: DatasetMakerExportResult | None) -> list[str]:
    warnings: list[str] = []
    for sprite in imported:
        for warning in sprite.warnings:
            warnings.append(f"- {sprite.item.sprite_id}: {warning}")
    if result is not None:
        warnings.extend(f"- Export: {warning}" for warning in result.warnings)
    return warnings or ["None."]


def _excluded_lines(imported: Sequence[ImportedSprite]) -> list[str]:
    lines: list[str] = []
    for sprite in imported:
        if sprite.item.status == "accepted":
            continue
        reason = "; ".join(sprite.errors or sprite.warnings or ("excluded by status",))
        lines.append(f"- {sprite.item.sprite_id} ({sprite.item.status}): {reason}")
        if len(lines) >= 20:
            lines.append("- ...")
            break
    return lines or ["None."]


def _palette_size_label(sprite: ImportedSprite) -> str:
    if sprite.item.palette_size is None:
        return "unknown"
    return str(sprite.item.palette_size)
