"""Training-readiness reporting and fixed-array export helpers."""

from spritelab.training.export import (
    TrainingExportConfig,
    TrainingExportResult,
    export_training_dataset,
    load_dedupe_groups,
    load_quality_issues,
)
from spritelab.training.palette_report import (
    PaletteSemanticsReport,
    PaletteSlotStats,
    build_palette_semantics_report,
    build_palette_semantics_report_from_curation,
    format_palette_semantics_report_markdown,
    write_palette_semantics_report_json,
    write_palette_semantics_report_markdown,
)
from spritelab.training.readiness import (
    ReadinessIssue,
    TrainingReadinessReport,
    build_readiness_report_from_export,
    build_training_readiness_report,
    format_training_readiness_markdown,
    write_training_readiness_markdown,
)
from spritelab.training.splits import DEFAULT_SPLIT_SEED, SplitAssignment, make_group_aware_split

__all__ = [
    "DEFAULT_SPLIT_SEED",
    "PaletteSemanticsReport",
    "PaletteSlotStats",
    "ReadinessIssue",
    "SplitAssignment",
    "TrainingExportConfig",
    "TrainingExportResult",
    "TrainingReadinessReport",
    "build_palette_semantics_report",
    "build_palette_semantics_report_from_curation",
    "build_readiness_report_from_export",
    "build_training_readiness_report",
    "export_training_dataset",
    "format_palette_semantics_report_markdown",
    "format_training_readiness_markdown",
    "load_dedupe_groups",
    "load_quality_issues",
    "make_group_aware_split",
    "write_palette_semantics_report_json",
    "write_palette_semantics_report_markdown",
    "write_training_readiness_markdown",
]
