"""Deterministic framing review for exported source sprite datasets."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.training.data import read_jsonl
from spritelab.training.framing_metrics import (
    CONNECTIVITY,
    SPRITE_SIZE,
    checkerboard_rgba,
    compute_sprite_framing_metrics,
    jsonable,
    rgba_array_to_image,
)
from spritelab.training.rgba import npz_row_to_rgba

SCHEMA_VERSION = "dataset_framing_review_v1.0"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetFramingReviewConfig:
    dataset_dir: Path
    out_dir: Path | None = None
    compare_generated: Path | None = None
    max_samples_per_sheet: int = 512


@dataclass
class DatasetFramingReviewResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    contact_sheets: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def review_dataset_framing(config: DatasetFramingReviewConfig) -> DatasetFramingReviewResult:
    dataset_dir = Path(config.dataset_dir)
    out_dir = Path(config.out_dir) if config.out_dir is not None else dataset_dir / "framing_review"
    json_path = out_dir / "framing_review_report.json"
    markdown_path = out_dir / "framing_review_report.md"
    errors: list[str] = []

    loaded = _load_source_samples(dataset_dir, errors)
    samples = [item["sample"] for item in loaded]
    overall = _summarize_samples(samples)
    groups = {
        "split": _summarize_group(samples, "split"),
        "category": _summarize_group(samples, "category"),
        "base_object": _summarize_group(samples, "base_object"),
    }
    comparison = (
        _compare_with_generated(samples, Path(config.compare_generated), errors)
        if config.compare_generated is not None
        else None
    )

    contact_sheets = _write_contact_sheets(
        out_dir,
        loaded,
        max_samples=max(0, int(config.max_samples_per_sheet)),
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(dataset_dir),
        "sample_count": len(samples),
        "connectivity": CONNECTIVITY,
        "overall": overall,
        "groups": groups,
        "comparison": comparison,
        "examples": {
            "worst_border_touch": _border_touch_examples(samples),
            "largest_alpha_coverage": _rank_examples(samples, "alpha_coverage", reverse=True),
            "smallest_alpha_coverage": _rank_examples(samples, "alpha_coverage", reverse=False),
        },
        "contact_sheets": {key: str(path) for key, path in sorted(contact_sheets.items())},
        "errors": errors,
        "samples": samples,
    }

    _write_reports(report, json_path, markdown_path)
    return DatasetFramingReviewResult(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        contact_sheets=contact_sheets,
        errors=errors,
    )


def _load_source_samples(dataset_dir: Path, errors: list[str]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    if not dataset_dir.is_dir():
        errors.append(f"dataset directory does not exist: {dataset_dir}")
        return loaded

    for split in SPLITS:
        npz_path = dataset_dir / f"{split}.npz"
        manifest_path = dataset_dir / f"manifest_{split}.jsonl"
        if not npz_path.is_file():
            errors.append(f"missing npz split file: {npz_path}")
            continue
        if not manifest_path.is_file():
            errors.append(f"missing split manifest: {manifest_path}")
            continue
        try:
            records = read_jsonl(manifest_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
        except Exception as exc:
            errors.append(f"{npz_path}: failed to load npz: {exc}")
            continue

        required = ("alpha", "index_map", "palette", "palette_mask", "sprite_id")
        missing = [key for key in required if key not in arrays]
        if missing:
            errors.append(f"{npz_path}: missing required arrays: {', '.join(missing)}")
            continue
        row_count = int(np.asarray(arrays["alpha"]).shape[0])
        if len(records) != row_count:
            errors.append(f"{split}: manifest row count {len(records)} does not match npz rows {row_count}")
        for row_index, record in enumerate(records[:row_count]):
            try:
                rgba = npz_row_to_rgba(
                    index_map=np.asarray(arrays["index_map"][row_index]),
                    alpha=np.asarray(arrays["alpha"][row_index]),
                    palette=np.asarray(arrays["palette"][row_index]),
                    palette_mask=np.asarray(arrays["palette_mask"][row_index], dtype=bool),
                )
            except Exception as exc:
                errors.append(f"{split}:{row_index}: failed to reconstruct RGBA: {exc}")
                continue
            sprite_id = str(record.get("sprite_id") or np.asarray(arrays["sprite_id"])[row_index])
            metrics = compute_sprite_framing_metrics(rgba)
            sample = {
                "sprite_id": sprite_id,
                "split": str(record.get("split") or split),
                "category": str(record.get("category") or "unknown"),
                "base_object": _base_object(record),
                "object_name": str(record.get("object_name") or ""),
                "source_path": str(record.get("source_path") or ""),
                "npz_file": f"{split}.npz",
                "npz_row": int(row_index),
                "metrics": metrics,
            }
            loaded.append({"sample": sample, "rgba": rgba})
    return loaded


def _base_object(record: Mapping[str, Any]) -> str:
    semantic = record.get("semantic_v3") if isinstance(record.get("semantic_v3"), Mapping) else {}
    value = semantic.get("base_object") if isinstance(semantic, Mapping) else None
    return str(value or record.get("base_object") or record.get("object_name") or "unknown")


def _summarize_samples(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [sample.get("metrics", {}) for sample in samples if isinstance(sample.get("metrics"), Mapping)]
    touches = [bool(metric.get("touches_border")) for metric in metrics]
    return {
        "count": len(samples),
        "border_touch_count": int(sum(1 for value in touches if value)),
        "border_touch_rate": float(sum(1 for value in touches if value) / len(touches)) if touches else 0.0,
        "alpha_coverage": _distribution(metrics, "alpha_coverage"),
        "bbox_width": _distribution(metrics, "bbox_width"),
        "bbox_height": _distribution(metrics, "bbox_height"),
        "bbox_area": _distribution(metrics, "bbox_area"),
        "center_offset_from_image_center": _distribution(metrics, "center_offset_from_image_center"),
        "visible_color_count": _distribution(metrics, "visible_color_count"),
        "connected_components": _distribution(metrics, "connected_components"),
        "alpha_edge_density": _distribution(metrics, "alpha_edge_density"),
    }


def _summarize_group(samples: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample.get(key) or "unknown"), []).append(sample)
    return {name: _summarize_samples(grouped[name]) for name in sorted(grouped)}


def _distribution(metrics: list[Mapping[str, Any]], key: str) -> dict[str, float | int | None]:
    values = _numeric_values(metrics, key)
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "p05": _percentile(values, 5),
        "p25": _percentile(values, 25),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "p75": _percentile(values, 75),
        "p95": _percentile(values, 95),
        "max": float(max(values)),
    }


def _numeric_values(metrics: list[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in metrics:
        value = item.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(sorted(values), dtype=np.float64)
    return float(np.percentile(arr, percentile))


def _border_touch_examples(samples: list[Mapping[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    touched = [sample for sample in samples if bool(sample.get("metrics", {}).get("touches_border"))]
    touched = sorted(
        touched,
        key=lambda sample: (
            -float(sample.get("metrics", {}).get("alpha_coverage") or 0.0),
            str(sample.get("sprite_id", "")),
        ),
    )
    return [_example(sample) for sample in touched[:limit]]


def _rank_examples(
    samples: list[Mapping[str, Any]], metric: str, *, reverse: bool, limit: int = 20
) -> list[dict[str, Any]]:
    ranked = sorted(
        samples,
        key=lambda sample: float(sample.get("metrics", {}).get(metric) or 0.0),
        reverse=reverse,
    )
    return [_example(sample) for sample in ranked[:limit]]


def _example(sample: Mapping[str, Any]) -> dict[str, Any]:
    metrics = sample.get("metrics") if isinstance(sample.get("metrics"), Mapping) else {}
    return {
        "sprite_id": sample.get("sprite_id", ""),
        "split": sample.get("split", ""),
        "category": sample.get("category", ""),
        "base_object": sample.get("base_object", ""),
        "alpha_coverage": metrics.get("alpha_coverage"),
        "bbox_width": metrics.get("bbox_width"),
        "bbox_height": metrics.get("bbox_height"),
        "center_offset_from_image_center": metrics.get("center_offset_from_image_center"),
        "touches_border": metrics.get("touches_border"),
    }


def _compare_with_generated(
    source_samples: list[Mapping[str, Any]],
    generated_dir: Path,
    errors: list[str],
) -> dict[str, Any] | None:
    generated_report_path = generated_dir / "generated_review_report.json"
    if not generated_report_path.is_file():
        try:
            from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites

            result = review_generated_sprites(
                GeneratedReviewConfig(
                    generated_dir=generated_dir,
                    out=generated_dir / "generated_review_report.md",
                    out_json=generated_report_path,
                    out_dir=generated_dir / "review",
                    group_by="category",
                    compare_raw_indexed=True,
                )
            )
            generated_report = result.report
        except Exception as exc:
            errors.append(f"failed to build generated review for comparison: {exc}")
            return None
    else:
        try:
            generated_report = json.loads(generated_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"failed to read generated review report: {exc}")
            return None

    generated_samples = generated_report.get("samples") if isinstance(generated_report, Mapping) else []
    if not isinstance(generated_samples, list):
        errors.append("generated review report has no samples list")
        return None

    source_metrics = [
        sample.get("metrics", {}) for sample in source_samples if isinstance(sample.get("metrics"), Mapping)
    ]
    generated_metrics = [
        sample.get("metrics", {})
        for sample in generated_samples
        if isinstance(sample, Mapping) and isinstance(sample.get("metrics"), Mapping)
    ]
    return {
        "generated_dir": str(generated_dir),
        "source_count": len(source_metrics),
        "generated_count": len(generated_metrics),
        "source_border_touch_rate": _border_touch_rate(source_metrics),
        "generated_border_touch_rate": _border_touch_rate(generated_metrics),
        "source_mean_alpha_coverage": _mean_metric(source_metrics, "alpha_coverage"),
        "generated_mean_alpha_coverage": _mean_metric(generated_metrics, "alpha_coverage"),
        "source_mean_bbox_width": _mean_metric(source_metrics, "bbox_width"),
        "generated_mean_bbox_width": _mean_metric(generated_metrics, "bbox_width"),
        "source_mean_bbox_height": _mean_metric(source_metrics, "bbox_height"),
        "generated_mean_bbox_height": _mean_metric(generated_metrics, "bbox_height"),
        "source_mean_center_offset": _mean_metric(source_metrics, "center_offset_from_image_center"),
        "generated_mean_center_offset": _mean_metric(generated_metrics, "center_offset_from_image_center"),
        "source_mean_visible_color_count": _mean_metric(source_metrics, "visible_color_count"),
        "generated_mean_visible_color_count": _mean_metric(generated_metrics, "visible_color_count"),
        "diagnosis": _diagnosis(source_metrics, generated_metrics),
    }


def _border_touch_rate(metrics: list[Mapping[str, Any]]) -> float:
    if not metrics:
        return 0.0
    return float(sum(1 for metric in metrics if bool(metric.get("touches_border"))) / len(metrics))


def _mean_metric(metrics: list[Mapping[str, Any]], key: str) -> float | None:
    values = _numeric_values(metrics, key)
    return float(statistics.fmean(values)) if values else None


def _diagnosis(source_metrics: list[Mapping[str, Any]], generated_metrics: list[Mapping[str, Any]]) -> str:
    source_border = _border_touch_rate(source_metrics)
    generated_border = _border_touch_rate(generated_metrics)
    if generated_border >= source_border + 0.25:
        return "generated_border_touch_rate_is_much_higher_than_source"
    if source_border >= 0.75 and generated_border >= 0.75:
        return "source_and_generated_are_both_border_heavy"
    if generated_border <= source_border + 0.10:
        return "generated_border_touch_rate_matches_source_distribution"
    return "generated_border_touch_rate_is_somewhat_higher_than_source"


def _write_contact_sheets(
    out_dir: Path,
    loaded: list[dict[str, Any]],
    *,
    max_samples: int,
) -> dict[str, Path]:
    contact_sheets: dict[str, Path] = {}
    overall = out_dir / "framing_contact_sheet.png"
    if _build_contact_sheet(loaded, overall, max_samples=max_samples):
        contact_sheets["overall"] = overall
    for split in SPLITS:
        subset = [item for item in loaded if item["sample"].get("split") == split]
        path = out_dir / f"framing_contact_sheet_{split}.png"
        if _build_contact_sheet(subset, path, max_samples=max_samples):
            contact_sheets[split] = path
    return contact_sheets


def _build_contact_sheet(items: list[dict[str, Any]], out_path: Path, *, max_samples: int, scale: int = 4) -> bool:
    rows = items[:max_samples] if max_samples > 0 else []
    if not rows:
        return False
    columns = min(8, max(1, len(rows)))
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGBA",
        (columns * cell + (columns + 1) * padding, sheet_rows * cell + (sheet_rows + 1) * padding),
        (38, 38, 42, 255),
    )
    for index, item in enumerate(rows):
        image = checkerboard_rgba(rgba_array_to_image(np.asarray(item["rgba"]))).resize(
            (cell, cell),
            Image.Resampling.NEAREST,
        )
        col = index % columns
        row = index // columns
        sheet.alpha_composite(image, (padding + col * (cell + padding), padding + row * (cell + padding)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def _write_reports(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(format_dataset_framing_markdown(report), encoding="utf-8")


def format_dataset_framing_markdown(report: Mapping[str, Any]) -> str:
    overall = report.get("overall") if isinstance(report.get("overall"), Mapping) else {}
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    comparison = report.get("comparison") if isinstance(report.get("comparison"), Mapping) else None
    lines = [
        "# Dataset Framing Review",
        "",
        "## Summary",
        "",
        f"- Samples: {int(report.get('sample_count', 0))}",
        f"- Connectivity: {report.get('connectivity', CONNECTIVITY)}",
        f"- Border-touch rate: {_fmt_rate(overall.get('border_touch_rate'))}",
        f"- Mean alpha coverage: {_fmt_dist_mean(overall.get('alpha_coverage'))}",
        f"- Mean bbox width: {_fmt_dist_mean(overall.get('bbox_width'))}",
        f"- Mean bbox height: {_fmt_dist_mean(overall.get('bbox_height'))}",
        f"- Mean center offset: {_fmt_dist_mean(overall.get('center_offset_from_image_center'))}",
        f"- Mean visible colors: {_fmt_dist_mean(overall.get('visible_color_count'))}",
        "",
        "## Split Summary",
        "",
        "| Split | Count | Border touch | Alpha coverage | BBox W | BBox H | Center offset | Colors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    split_groups = groups.get("split") if isinstance(groups.get("split"), Mapping) else {}
    for split in SPLITS:
        summary = split_groups.get(split) if isinstance(split_groups.get(split), Mapping) else None
        if summary is None:
            continue
        lines.append(_summary_row(split, summary))

    lines.extend(
        [
            "",
            "## Category Summary",
            "",
            "| Category | Count | Border touch | Alpha coverage | BBox W | BBox H | Center offset | Colors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    category_groups = groups.get("category") if isinstance(groups.get("category"), Mapping) else {}
    for category, summary in sorted(category_groups.items()):
        if isinstance(summary, Mapping):
            lines.append(_summary_row(str(category), summary))

    if comparison is not None:
        lines.extend(
            [
                "",
                "## Source vs Generated",
                "",
                f"- Source border-touch rate: {_fmt_rate(comparison.get('source_border_touch_rate'))}",
                f"- Generated border-touch rate: {_fmt_rate(comparison.get('generated_border_touch_rate'))}",
                f"- Source mean alpha coverage: {_fmt_float(comparison.get('source_mean_alpha_coverage'))}",
                f"- Generated mean alpha coverage: {_fmt_float(comparison.get('generated_mean_alpha_coverage'))}",
                f"- Source mean bbox: {_fmt_float(comparison.get('source_mean_bbox_width'))} x {_fmt_float(comparison.get('source_mean_bbox_height'))}",
                f"- Generated mean bbox: {_fmt_float(comparison.get('generated_mean_bbox_width'))} x {_fmt_float(comparison.get('generated_mean_bbox_height'))}",
                f"- Source mean center offset: {_fmt_float(comparison.get('source_mean_center_offset'))}",
                f"- Generated mean center offset: {_fmt_float(comparison.get('generated_mean_center_offset'))}",
                f"- Source mean visible colors: {_fmt_float(comparison.get('source_mean_visible_color_count'))}",
                f"- Generated mean visible colors: {_fmt_float(comparison.get('generated_mean_visible_color_count'))}",
                f"- Diagnosis: `{comparison.get('diagnosis')}`",
            ]
        )

    examples = report.get("examples") if isinstance(report.get("examples"), Mapping) else {}
    lines.extend(
        [
            "",
            "## Worst Border-Touch Examples",
            "",
            "| Sprite | Split | Category | Base object | Alpha | BBox |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for example in (examples.get("worst_border_touch") or [])[:10]:
        lines.append(_example_row(example))
    lines.extend(
        [
            "",
            "## Largest Alpha-Coverage Examples",
            "",
            "| Sprite | Split | Category | Base object | Alpha | BBox |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for example in (examples.get("largest_alpha_coverage") or [])[:10]:
        lines.append(_example_row(example))
    lines.extend(
        [
            "",
            "## Smallest Alpha-Coverage Examples",
            "",
            "| Sprite | Split | Category | Base object | Alpha | BBox |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for example in (examples.get("smallest_alpha_coverage") or [])[:10]:
        lines.append(_example_row(example))

    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    if errors:
        lines.extend(["", "## Input Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)


def _summary_row(name: str, summary: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {int(summary.get('count', 0))} | {_fmt_rate(summary.get('border_touch_rate'))} | "
        f"{_fmt_dist_mean(summary.get('alpha_coverage'))} | {_fmt_dist_mean(summary.get('bbox_width'))} | "
        f"{_fmt_dist_mean(summary.get('bbox_height'))} | {_fmt_dist_mean(summary.get('center_offset_from_image_center'))} | "
        f"{_fmt_dist_mean(summary.get('visible_color_count'))} |"
    )


def _example_row(example: Mapping[str, Any]) -> str:
    bbox = f"{example.get('bbox_width')}x{example.get('bbox_height')}"
    return (
        f"| `{example.get('sprite_id', '')}` | {example.get('split', '')} | {example.get('category', '')} | "
        f"{example.get('base_object', '')} | {_fmt_float(example.get('alpha_coverage'))} | {bbox} |"
    )


def _fmt_dist_mean(value: Any) -> str:
    if isinstance(value, Mapping):
        return _fmt_float(value.get("mean"))
    return "n/a"


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Review exported dataset sprite framing.")
    parser.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--compare-generated", type=Path)
    parser.add_argument("--max-samples-per-sheet", type=int, default=512)
    parsed = parser.parse_args(argv)
    result = review_dataset_framing(DatasetFramingReviewConfig(**vars(parsed)))
    print(f"Reviewed source sprites: {result.report['sample_count']}")
    print(f"Markdown report: {result.markdown_path}")
    print(f"JSON report: {result.json_path}")
    print(f"Contact sheets: {len(result.contact_sheets)}")
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
