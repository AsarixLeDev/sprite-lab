"""Dataset quality diagnostics for SpriteBundle datasets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.codec.bundle import SPRITE_HEIGHT, SPRITE_WIDTH, SpriteBundle
from spritelab.codec.color import srgb_luminance
from spritelab.codec.io import load_bundle
from spritelab.data.manifest import DatasetManifest, load_manifest
from spritelab.data.preview_grid import (
    PreviewGridRecord,
    filter_preview_records,
    load_preview_records,
)

EMPTY_SPRITE = "EMPTY_SPRITE"
MOSTLY_EMPTY = "MOSTLY_EMPTY"
MOSTLY_FULL = "MOSTLY_FULL"
TINY_BBOX = "TINY_BBOX"
HUGE_BBOX = "HUGE_BBOX"
OFF_CENTER = "OFF_CENTER"
MANY_COMPONENTS = "MANY_COMPONENTS"
FRAGMENTED = "FRAGMENTED"
MANY_SINGLE_PIXELS = "MANY_SINGLE_PIXELS"
HAS_ALPHA_HOLES = "HAS_ALPHA_HOLES"
TOUCHES_EDGE = "TOUCHES_EDGE"
LOW_CONTRAST = "LOW_CONTRAST"
TINY_PALETTE = "TINY_PALETTE"
LARGE_PALETTE = "LARGE_PALETTE"
INVALID_LOAD = "INVALID_LOAD"

ISSUE_DESCRIPTIONS = {
    EMPTY_SPRITE: "No opaque pixels.",
    MOSTLY_EMPTY: "Very little opaque coverage.",
    MOSTLY_FULL: "Nearly all pixels are opaque.",
    TINY_BBOX: "Opaque bounding box is very small in at least one dimension.",
    HUGE_BBOX: "Opaque bounding box nearly spans the canvas.",
    OFF_CENTER: "Opaque center of mass is far from the sprite center.",
    MANY_COMPONENTS: "Many disconnected opaque components.",
    FRAGMENTED: "Largest opaque component is relatively small.",
    MANY_SINGLE_PIXELS: "Many one-pixel opaque components.",
    HAS_ALPHA_HOLES: "Transparent holes exist inside opaque regions.",
    TOUCHES_EDGE: "Opaque pixels touch the image boundary.",
    LOW_CONTRAST: "Used palette luminance range is low.",
    TINY_PALETTE: "Palette has very few slots.",
    LARGE_PALETTE: "Palette has many slots.",
    INVALID_LOAD: "Bundle could not be loaded or validated.",
}


@dataclass(frozen=True)
class QualityReportOptions:
    dataset_path: Path
    output_dir: Path | None = None
    max_items: int | None = None
    filter_category: str | None = None
    filter_split: str | None = None
    include_markdown: bool = True
    include_json: bool = True
    write_flag_files: bool = True
    fail_on_load_error: bool = False


@dataclass(frozen=True)
class SpriteQualityMetrics:
    id: str
    bundle_dir: str

    width: int
    height: int

    palette_size: int
    opaque_pixel_count: int
    opaque_pixel_ratio: float

    bbox_x: int | None
    bbox_y: int | None
    bbox_width: int | None
    bbox_height: int | None
    bbox_area: int | None
    bbox_fill_ratio: float | None

    center_x: float | None
    center_y: float | None
    center_offset_x: float | None
    center_offset_y: float | None
    center_distance: float | None

    connected_component_count: int
    largest_component_ratio: float | None

    single_pixel_component_count: int
    single_pixel_component_ratio: float | None

    alpha_hole_count: int
    edge_touching_opaque_count: int
    edge_touching_opaque_ratio: float

    contrast_score: float | None
    luminance_min: float | None
    luminance_max: float | None
    luminance_range: float | None

    issue_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetQualitySummary:
    total_records: int
    analyzed_records: int
    failed_records: int

    palette_size_min: int | None
    palette_size_max: int | None
    palette_size_mean: float | None

    opaque_ratio_min: float | None
    opaque_ratio_max: float | None
    opaque_ratio_mean: float | None

    issue_counts: dict[str, int]
    category_counts: dict[str, int]
    split_counts: dict[str, int]

    duplicate_sha256_groups: dict[str, list[str]]


@dataclass(frozen=True)
class FailedQualityRecord:
    id: str | None
    bundle_dir: str
    reason: str


@dataclass(frozen=True)
class DatasetQualityReport:
    dataset_name: str
    summary: DatasetQualitySummary
    records: list[SpriteQualityMetrics]
    failed: list[FailedQualityRecord]
    options: dict[str, Any]


def compute_sprite_quality_metrics(
    bundle: SpriteBundle,
    *,
    bundle_dir: str | Path,
) -> SpriteQualityMetrics:
    """Compute deterministic quality metrics for one SpriteBundle."""

    alpha = np.asarray(bundle.alpha)
    index_map = np.asarray(bundle.index_map)
    palette = np.asarray(bundle.palette)
    opaque_mask = alpha == 1
    opaque_pixel_count = int(np.count_nonzero(opaque_mask))
    opaque_pixel_ratio = opaque_pixel_count / float(SPRITE_WIDTH * SPRITE_HEIGHT)

    bbox = _compute_bbox(opaque_mask, opaque_pixel_count)
    center = _compute_center(opaque_mask, opaque_pixel_count)
    components = _component_sizes(opaque_mask)
    connected_component_count = len(components)
    largest_component_ratio = max(components) / opaque_pixel_count if opaque_pixel_count > 0 and components else None
    single_pixel_component_count = sum(1 for size in components if size == 1)
    single_pixel_component_ratio = (
        single_pixel_component_count / connected_component_count if connected_component_count > 0 else None
    )
    alpha_hole_count = _count_alpha_holes(alpha)
    edge_touching_opaque_count = _count_edge_opaque(opaque_mask)
    edge_touching_opaque_ratio = edge_touching_opaque_count / opaque_pixel_count if opaque_pixel_count > 0 else 0.0
    luminance = _compute_luminance_metrics(alpha, index_map, palette)

    issue_codes = _issue_codes(
        palette_size=int(palette.shape[0]),
        opaque_pixel_count=opaque_pixel_count,
        opaque_pixel_ratio=opaque_pixel_ratio,
        bbox_width=bbox["bbox_width"],
        bbox_height=bbox["bbox_height"],
        center_distance=center["center_distance"],
        connected_component_count=connected_component_count,
        largest_component_ratio=largest_component_ratio,
        single_pixel_component_count=single_pixel_component_count,
        alpha_hole_count=alpha_hole_count,
        edge_touching_opaque_count=edge_touching_opaque_count,
        contrast_score=luminance["contrast_score"],
    )

    return SpriteQualityMetrics(
        id=bundle.metadata.id,
        bundle_dir=str(bundle_dir),
        width=bundle.metadata.width,
        height=bundle.metadata.height,
        palette_size=int(palette.shape[0]),
        opaque_pixel_count=opaque_pixel_count,
        opaque_pixel_ratio=opaque_pixel_ratio,
        bbox_x=bbox["bbox_x"],
        bbox_y=bbox["bbox_y"],
        bbox_width=bbox["bbox_width"],
        bbox_height=bbox["bbox_height"],
        bbox_area=bbox["bbox_area"],
        bbox_fill_ratio=bbox["bbox_fill_ratio"],
        center_x=center["center_x"],
        center_y=center["center_y"],
        center_offset_x=center["center_offset_x"],
        center_offset_y=center["center_offset_y"],
        center_distance=center["center_distance"],
        connected_component_count=connected_component_count,
        largest_component_ratio=largest_component_ratio,
        single_pixel_component_count=single_pixel_component_count,
        single_pixel_component_ratio=single_pixel_component_ratio,
        alpha_hole_count=alpha_hole_count,
        edge_touching_opaque_count=edge_touching_opaque_count,
        edge_touching_opaque_ratio=edge_touching_opaque_ratio,
        contrast_score=luminance["contrast_score"],
        luminance_min=luminance["luminance_min"],
        luminance_max=luminance["luminance_max"],
        luminance_range=luminance["luminance_range"],
        issue_codes=issue_codes,
    )


def create_quality_report(options: QualityReportOptions) -> DatasetQualityReport:
    """Load a dataset, compute quality metrics, write reports, and return them."""

    records = load_preview_records(options.dataset_path)
    records = filter_preview_records(
        records,
        filter_category=options.filter_category,
        filter_split=options.filter_split,
    )
    if options.max_items is not None:
        records = records[: options.max_items]

    metrics: list[SpriteQualityMetrics] = []
    failed: list[FailedQualityRecord] = []
    for record in records:
        try:
            bundle = load_bundle(record.bundle_dir)
            metrics.append(compute_sprite_quality_metrics(bundle, bundle_dir=record.bundle_dir))
        except Exception as exc:
            if options.fail_on_load_error:
                raise
            failed.append(
                FailedQualityRecord(
                    id=record.id,
                    bundle_dir=str(record.bundle_dir),
                    reason=str(exc),
                )
            )

    sha_groups = _duplicate_sha256_groups(options.dataset_path, {record.id for record in records})
    summary = _build_summary(records=records, metrics=metrics, failed=failed, duplicate_sha256_groups=sha_groups)
    report = DatasetQualityReport(
        dataset_name=_dataset_name(options.dataset_path),
        summary=summary,
        records=metrics,
        failed=failed,
        options=_options_dict(options),
    )

    output_dir = _output_dir(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_report.json"
    md_path = output_dir / "quality_report.md"
    if options.include_json:
        save_quality_report_json(report, json_path)
    if options.include_markdown:
        save_quality_report_markdown(report, md_path)
    if options.write_flag_files:
        _write_flag_files(report, output_dir / "flagged")

    return report


def save_quality_report_json(report: DatasetQualityReport, path: str | Path) -> None:
    """Write a quality report as stable, readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_quality_report_json(path: str | Path) -> DatasetQualityReport:
    """Load a quality report from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = DatasetQualitySummary(**data["summary"])
    records = [SpriteQualityMetrics(**record) for record in data["records"]]
    failed = [FailedQualityRecord(**record) for record in data["failed"]]
    return DatasetQualityReport(
        dataset_name=data["dataset_name"],
        summary=summary,
        records=records,
        failed=failed,
        options=dict(data["options"]),
    )


def render_quality_report_markdown(report: DatasetQualityReport) -> str:
    """Render a quality report as simple Markdown."""

    summary = report.summary
    lines = [
        "# Dataset Quality Report",
        "",
        f"Dataset: {report.dataset_name}",
        "",
        "## Summary",
        "",
        f"- Total records: {summary.total_records}",
        f"- Analyzed records: {summary.analyzed_records}",
        f"- Failed records: {summary.failed_records}",
        f"- Palette size min/max/mean: {_fmt_int(summary.palette_size_min)} / {_fmt_int(summary.palette_size_max)} / {_fmt_float(summary.palette_size_mean)}",
        f"- Opaque ratio min/max/mean: {_fmt_float(summary.opaque_ratio_min)} / {_fmt_float(summary.opaque_ratio_max)} / {_fmt_float(summary.opaque_ratio_mean)}",
        "",
        "## Issue counts",
        "",
        "| Issue | Count |",
        "|---|---:|",
    ]
    lines.extend(_count_rows(summary.issue_counts, empty_label="None"))
    lines.extend(
        [
            "",
            "## Category counts",
            "",
            "| Category | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(_count_rows(summary.category_counts, empty_label="None"))
    lines.extend(
        [
            "",
            "## Split counts",
            "",
            "| Split | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(_count_rows(summary.split_counts, empty_label="None"))
    lines.extend(
        [
            "",
            "## Top flagged sprites",
            "",
            "| ID | Issues | Palette | Opaque ratio | Components | Contrast |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    flagged = [record for record in report.records if record.issue_codes]
    flagged.sort(key=lambda record: (-len(record.issue_codes), record.id))
    for record in flagged[:25]:
        lines.append(
            f"| {record.id} | {', '.join(record.issue_codes)} | {record.palette_size} | "
            f"{record.opaque_pixel_ratio:.3f} | {record.connected_component_count} | "
            f"{_fmt_float(record.contrast_score)} |"
        )
    if not flagged:
        lines.append("| None |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Failed records",
            "",
            "| ID | Bundle dir | Reason |",
            "|---|---|---|",
        ]
    )
    if report.failed:
        for record in report.failed:
            lines.append(f"| {record.id or ''} | {record.bundle_dir} | {record.reason} |")
    else:
        lines.append("| None |  |  |")

    lines.extend(
        [
            "",
            "## Issue code meanings",
            "",
            "| Issue | Meaning |",
            "|---|---|",
        ]
    )
    for issue, description in ISSUE_DESCRIPTIONS.items():
        lines.append(f"| {issue} | {description} |")

    return "\n".join(lines) + "\n"


def save_quality_report_markdown(report: DatasetQualityReport, path: str | Path) -> None:
    """Write a quality report as Markdown."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_quality_report_markdown(report), encoding="utf-8")


def _compute_bbox(opaque_mask: np.ndarray, opaque_pixel_count: int) -> dict[str, int | float | None]:
    if opaque_pixel_count == 0:
        return {
            "bbox_x": None,
            "bbox_y": None,
            "bbox_width": None,
            "bbox_height": None,
            "bbox_area": None,
            "bbox_fill_ratio": None,
        }

    coords = np.argwhere(opaque_mask)
    min_y = int(np.min(coords[:, 0]))
    max_y = int(np.max(coords[:, 0]))
    min_x = int(np.min(coords[:, 1]))
    max_x = int(np.max(coords[:, 1]))
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    area = width * height
    return {
        "bbox_x": min_x,
        "bbox_y": min_y,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "bbox_fill_ratio": opaque_pixel_count / area,
    }


def _compute_center(opaque_mask: np.ndarray, opaque_pixel_count: int) -> dict[str, float | None]:
    if opaque_pixel_count == 0:
        return {
            "center_x": None,
            "center_y": None,
            "center_offset_x": None,
            "center_offset_y": None,
            "center_distance": None,
        }

    coords = np.argwhere(opaque_mask)
    center_y = float(np.mean(coords[:, 0]))
    center_x = float(np.mean(coords[:, 1]))
    offset_x = center_x - 15.5
    offset_y = center_y - 15.5
    return {
        "center_x": center_x,
        "center_y": center_y,
        "center_offset_x": offset_x,
        "center_offset_y": offset_y,
        "center_distance": math.sqrt(offset_x * offset_x + offset_y * offset_y),
    }


def _component_sizes(mask: np.ndarray) -> list[int]:
    visited = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    for y in range(SPRITE_HEIGHT):
        for x in range(SPRITE_WIDTH):
            if visited[y, x] or not bool(mask[y, x]):
                continue
            size = 0
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for ny, nx in _neighbors4(cy, cx):
                    if not visited[ny, nx] and bool(mask[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            sizes.append(size)
    return sizes


def _count_alpha_holes(alpha: np.ndarray) -> int:
    transparent = alpha == 0
    visited = np.zeros(transparent.shape, dtype=bool)
    holes = 0
    for y in range(SPRITE_HEIGHT):
        for x in range(SPRITE_WIDTH):
            if visited[y, x] or not bool(transparent[y, x]):
                continue
            touches_boundary = False
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                if cy == 0 or cy == SPRITE_HEIGHT - 1 or cx == 0 or cx == SPRITE_WIDTH - 1:
                    touches_boundary = True
                for ny, nx in _neighbors4(cy, cx):
                    if not visited[ny, nx] and bool(transparent[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if not touches_boundary:
                holes += 1
    return holes


def _neighbors4(y: int, x: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    if y > 0:
        neighbors.append((y - 1, x))
    if y < SPRITE_HEIGHT - 1:
        neighbors.append((y + 1, x))
    if x > 0:
        neighbors.append((y, x - 1))
    if x < SPRITE_WIDTH - 1:
        neighbors.append((y, x + 1))
    return neighbors


def _count_edge_opaque(opaque_mask: np.ndarray) -> int:
    edge = np.zeros_like(opaque_mask, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    return int(np.count_nonzero(opaque_mask & edge))


def _compute_luminance_metrics(
    alpha: np.ndarray, index_map: np.ndarray, palette: np.ndarray
) -> dict[str, float | None]:
    if int(np.count_nonzero(alpha == 1)) == 0:
        return {
            "contrast_score": None,
            "luminance_min": None,
            "luminance_max": None,
            "luminance_range": None,
        }

    used_slots = sorted(int(slot) for slot in np.unique(index_map[alpha == 1]) if int(slot) > 0)
    if not used_slots:
        return {
            "contrast_score": None,
            "luminance_min": None,
            "luminance_max": None,
            "luminance_range": None,
        }

    values = [srgb_luminance(tuple(int(channel) for channel in palette[slot])) for slot in used_slots]
    luminance_min = min(values)
    luminance_max = max(values)
    luminance_range = luminance_max - luminance_min if len(values) >= 2 else 0.0
    return {
        "contrast_score": luminance_range,
        "luminance_min": luminance_min,
        "luminance_max": luminance_max,
        "luminance_range": luminance_range,
    }


def _issue_codes(
    *,
    palette_size: int,
    opaque_pixel_count: int,
    opaque_pixel_ratio: float,
    bbox_width: int | None,
    bbox_height: int | None,
    center_distance: float | None,
    connected_component_count: int,
    largest_component_ratio: float | None,
    single_pixel_component_count: int,
    alpha_hole_count: int,
    edge_touching_opaque_count: int,
    contrast_score: float | None,
) -> list[str]:
    issues: list[str] = []
    if opaque_pixel_count == 0:
        issues.append(EMPTY_SPRITE)
    if opaque_pixel_ratio < 0.03:
        issues.append(MOSTLY_EMPTY)
    if opaque_pixel_ratio > 0.90:
        issues.append(MOSTLY_FULL)
    if bbox_width is not None and bbox_height is not None and (bbox_width <= 4 or bbox_height <= 4):
        issues.append(TINY_BBOX)
    if bbox_width is not None and bbox_height is not None and (bbox_width >= 31 or bbox_height >= 31):
        issues.append(HUGE_BBOX)
    if center_distance is not None and center_distance > 7.0:
        issues.append(OFF_CENTER)
    if connected_component_count >= 8:
        issues.append(MANY_COMPONENTS)
    if largest_component_ratio is not None and largest_component_ratio < 0.75:
        issues.append(FRAGMENTED)
    if single_pixel_component_count >= 5:
        issues.append(MANY_SINGLE_PIXELS)
    if alpha_hole_count >= 1:
        issues.append(HAS_ALPHA_HOLES)
    if edge_touching_opaque_count >= 1:
        issues.append(TOUCHES_EDGE)
    if contrast_score is not None and contrast_score < 0.12:
        issues.append(LOW_CONTRAST)
    if palette_size <= 2:
        issues.append(TINY_PALETTE)
    if palette_size > 24:
        issues.append(LARGE_PALETTE)
    return issues


def _build_summary(
    *,
    records: list[PreviewGridRecord],
    metrics: list[SpriteQualityMetrics],
    failed: list[FailedQualityRecord],
    duplicate_sha256_groups: dict[str, list[str]],
) -> DatasetQualitySummary:
    palette_sizes = [record.palette_size for record in metrics]
    opaque_ratios = [record.opaque_pixel_ratio for record in metrics]
    issue_counter: Counter[str] = Counter()
    for record in metrics:
        issue_counter.update(record.issue_codes)

    category_counts = Counter(record.category or "unknown" for record in records)
    split_counts = Counter(record.split or "unknown" for record in records)
    return DatasetQualitySummary(
        total_records=len(records),
        analyzed_records=len(metrics),
        failed_records=len(failed),
        palette_size_min=min(palette_sizes) if palette_sizes else None,
        palette_size_max=max(palette_sizes) if palette_sizes else None,
        palette_size_mean=_mean(palette_sizes),
        opaque_ratio_min=min(opaque_ratios) if opaque_ratios else None,
        opaque_ratio_max=max(opaque_ratios) if opaque_ratios else None,
        opaque_ratio_mean=_mean(opaque_ratios),
        issue_counts=dict(sorted(issue_counter.items())),
        category_counts=dict(sorted(category_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
        duplicate_sha256_groups=duplicate_sha256_groups,
    )


def _duplicate_sha256_groups(dataset_path: Path, included_ids: set[str]) -> dict[str, list[str]]:
    manifest = _load_manifest_for_dataset(dataset_path)
    if manifest is None:
        return {}

    groups: dict[str, list[str]] = defaultdict(list)
    for record in manifest.records:
        if record.id in included_ids and record.sha256:
            groups[record.sha256].append(record.id)

    return {digest: sorted(ids) for digest, ids in sorted(groups.items()) if len(ids) >= 2}


def _load_manifest_for_dataset(dataset_path: Path) -> DatasetManifest | None:
    path = Path(dataset_path)
    if path.is_file() and path.name == "manifest.json":
        return load_manifest(path)
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        return load_manifest(manifest_path)
    return None


def _mean(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _dataset_name(dataset_path: Path) -> str:
    path = Path(dataset_path)
    if path.is_file():
        return path.parent.name
    return path.name


def _output_dir(options: QualityReportOptions) -> Path:
    if options.output_dir is not None:
        return Path(options.output_dir)
    dataset_path = Path(options.dataset_path)
    if dataset_path.is_file():
        return dataset_path.parent / "quality_report"
    return dataset_path / "quality_report"


def _options_dict(options: QualityReportOptions) -> dict[str, Any]:
    data = asdict(options)
    data["dataset_path"] = str(options.dataset_path)
    data["output_dir"] = None if options.output_dir is None else str(options.output_dir)
    return data


def _write_flag_files(report: DatasetQualityReport, directory: Path) -> None:
    issue_to_ids: dict[str, list[str]] = defaultdict(list)
    for record in report.records:
        for issue in record.issue_codes:
            issue_to_ids[issue].append(record.id)

    if not issue_to_ids:
        directory.mkdir(parents=True, exist_ok=True)
        return

    directory.mkdir(parents=True, exist_ok=True)
    for issue, ids in sorted(issue_to_ids.items()):
        (directory / f"{issue}.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")


def _count_rows(counts: dict[str, int], *, empty_label: str) -> list[str]:
    if not counts:
        return [f"| {empty_label} | 0 |"]
    return [f"| {key} | {value} |" for key, value in sorted(counts.items())]


from spritelab.training.report_utils import fmt_int as _fmt_int


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _parse_args() -> QualityReportOptions:
    parser = argparse.ArgumentParser(description="Generate quality diagnostics for a SpriteBundle dataset.")
    parser.add_argument("--dataset", required=True, dest="dataset_path", type=Path)
    parser.add_argument("--output", dest="output_dir", type=Path)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--category", dest="filter_category")
    parser.add_argument("--split", dest="filter_split")
    parser.add_argument("--no-markdown", action="store_false", dest="include_markdown")
    parser.add_argument("--no-json", action="store_false", dest="include_json")
    parser.add_argument("--no-flag-files", action="store_false", dest="write_flag_files")
    parser.add_argument("--fail-on-load-error", action="store_true")
    args = parser.parse_args()
    return QualityReportOptions(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_items=args.max_items,
        filter_category=args.filter_category,
        filter_split=args.filter_split,
        include_markdown=args.include_markdown,
        include_json=args.include_json,
        write_flag_files=args.write_flag_files,
        fail_on_load_error=args.fail_on_load_error,
    )


def main() -> None:
    options = _parse_args()
    report = create_quality_report(options)
    output_dir = _output_dir(options)
    flagged = sum(1 for record in report.records if record.issue_codes)
    print(f"Dataset: {report.dataset_name}")
    print(f"Analyzed: {report.summary.analyzed_records} / {report.summary.total_records}")
    print(f"Failed: {report.summary.failed_records}")
    print(f"Flagged: {flagged}")
    if options.include_json:
        print(f"JSON: {output_dir / 'quality_report.json'}")
    if options.include_markdown:
        print(f"Markdown: {output_dir / 'quality_report.md'}")


if __name__ == "__main__":
    main()
