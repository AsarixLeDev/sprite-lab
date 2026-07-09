"""Deterministic structural review for generated sprite sample folders."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.training.framing_metrics import (
    alpha_mask_from_image as _shared_alpha_mask_from_image,
)
from spritelab.training.framing_metrics import (
    checkerboard_rgba as _shared_checkerboard_rgba,
)
from spritelab.training.framing_metrics import (
    compute_alpha_metrics as _shared_compute_alpha_metrics,
)
from spritelab.training.framing_metrics import (
    compute_color_metrics as _shared_compute_color_metrics,
)
from spritelab.training.framing_metrics import (
    compute_connected_components as _shared_compute_connected_components,
)
from spritelab.training.framing_metrics import (
    compute_edge_metrics as _shared_compute_edge_metrics,
)
from spritelab.training.framing_metrics import (
    image_to_rgba_array as _shared_image_to_rgba_array,
)
from spritelab.training.framing_metrics import (
    transparent_index_used as _shared_transparent_index_used,
)

SPRITE_SIZE = 32
SCHEMA_VERSION = "generated_review_v1.0"
CONNECTIVITY = "8-neighbor"

KNOWN_CATEGORIES = (
    "seen_object",
    "unseen_composition",
    "creative_concept",
    "style_stress",
    "negative_control",
    "unknown",
)

WARNING_ORDER = (
    "empty_or_nearly_empty",
    "too_full_canvas",
    "touches_border",
    "off_center",
    "fragmented",
    "single_blob",
    "too_few_colors",
    "too_many_rare_colors",
    "quantization_destructive",
    "raw_missing",
    "indexed_missing",
)


@dataclass(frozen=True)
class GeneratedReviewConfig:
    generated_dir: Path
    out: Path | None = None
    out_json: Path | None = None
    out_dir: Path | None = None
    group_by: str = "none"
    max_samples_per_sheet: int = 64
    compare_raw_indexed: bool = False
    strict: bool = False


@dataclass
class GeneratedReviewResult:
    report: dict[str, Any]
    markdown_path: Path | None = None
    json_path: Path | None = None
    contact_sheets: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def review_generated_sprites(config: GeneratedReviewConfig) -> GeneratedReviewResult:
    generated_dir = Path(config.generated_dir)
    artifact_dir, markdown_path, json_path = _resolve_output_paths(config)
    errors: list[str] = []
    manifest_path = generated_dir / "generated_manifest.jsonl"

    if not generated_dir.is_dir():
        errors.append(f"generated directory does not exist: {generated_dir}")
        report = _empty_report(generated_dir, errors)
        return GeneratedReviewResult(report=report, markdown_path=markdown_path, json_path=json_path, errors=errors)

    if not manifest_path.is_file():
        errors.append(f"generated_manifest.jsonl is missing: {manifest_path}")
        report = _empty_report(generated_dir, errors)
        _write_reports(report, markdown_path, json_path)
        return GeneratedReviewResult(report=report, markdown_path=markdown_path, json_path=json_path, errors=errors)

    records = _read_manifest(manifest_path, errors)
    generation_report = _read_generation_report(generated_dir / "generation_report.json", errors, strict=config.strict)

    samples: list[dict[str, Any]] = []
    for record in records:
        sample = _review_sample(
            generated_dir,
            record,
            compare_raw_indexed=config.compare_raw_indexed,
            strict=config.strict,
            errors=errors,
        )
        samples.append(sample)

    overall = _summarize_samples(samples)
    groups = _summarize_groups(samples, group_by=config.group_by)
    contact_sheets = _write_contact_sheets(
        generated_dir,
        artifact_dir,
        samples,
        group_by=config.group_by,
        compare_raw_indexed=config.compare_raw_indexed,
        max_samples=config.max_samples_per_sheet,
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_dir": str(generated_dir),
        "sample_count": len(samples),
        "connectivity": CONNECTIVITY,
        "compare_raw_indexed": bool(config.compare_raw_indexed),
        "strict": bool(config.strict),
        "source_generation_report_present": generation_report is not None,
        "contact_sheets": {key: str(path) for key, path in sorted(contact_sheets.items())},
        "overall": overall,
        "groups": groups,
        "errors": list(errors),
        "samples": samples,
    }
    report["recommendations"] = _recommendations(report)

    _write_reports(report, markdown_path, json_path)
    return GeneratedReviewResult(
        report=report,
        markdown_path=markdown_path,
        json_path=json_path,
        contact_sheets=contact_sheets,
        errors=errors,
    )


def compute_alpha_silhouette_metrics(mask: np.ndarray) -> dict[str, Any]:
    return _shared_compute_alpha_metrics(mask)


def compute_connected_component_metrics(mask: np.ndarray) -> dict[str, Any]:
    return _shared_compute_connected_components(mask)


def compute_color_metrics(image: Image.Image | None) -> dict[str, Any]:
    return _shared_compute_color_metrics(image)


def compute_edge_metrics(mask: np.ndarray, image: Image.Image | None) -> dict[str, Any]:
    return _shared_compute_edge_metrics(mask, image)


def compute_raw_indexed_difference(raw_image: Image.Image | None, indexed_image: Image.Image | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "raw_indexed_rgb_mae_visible": None,
        "raw_indexed_alpha_diff": None,
        "quantization_changed_visible_pixels_ratio": None,
    }
    if raw_image is None or indexed_image is None:
        return metrics

    raw = _image_to_rgba_array(raw_image).astype(np.float32) / 255.0
    indexed = _image_to_rgba_array(indexed_image).astype(np.float32) / 255.0
    visible = indexed[..., 3] > 0.0
    if bool(np.any(visible)):
        diff = np.abs(raw[..., :3] - indexed[..., :3])
        metrics["raw_indexed_rgb_mae_visible"] = float(np.mean(diff[visible]))
        metrics["quantization_changed_visible_pixels_ratio"] = float(
            np.mean(np.any(diff[visible] > (1.0 / 255.0), axis=1))
        )
    else:
        metrics["raw_indexed_rgb_mae_visible"] = 0.0
        metrics["quantization_changed_visible_pixels_ratio"] = 0.0
    metrics["raw_indexed_alpha_diff"] = float(np.mean(np.abs(raw[..., 3] - indexed[..., 3])))
    return metrics


def warning_labels(metrics: Mapping[str, Any], *, compare_raw_indexed: bool = False) -> list[str]:
    warnings: list[str] = []
    alpha_coverage = float(metrics.get("alpha_coverage") or 0.0)
    center_offset = metrics.get("center_offset_from_image_center")
    raw_indexed_mae = metrics.get("raw_indexed_rgb_mae_visible")

    if alpha_coverage < 0.03:
        warnings.append("empty_or_nearly_empty")
    if alpha_coverage > 0.85:
        warnings.append("too_full_canvas")
    if bool(metrics.get("touches_border")):
        warnings.append("touches_border")
    if center_offset is not None and float(center_offset) > 8.0:
        warnings.append("off_center")
    if (
        int(metrics.get("connected_components") or 0) >= 8
        and float(metrics.get("largest_component_ratio") or 0.0) < 0.75
    ):
        warnings.append("fragmented")
    if (
        alpha_coverage > 0.05
        and float(metrics.get("bbox_fill_ratio") or 0.0) > 0.85
        and float(metrics.get("alpha_edge_density") or 0.0) < 0.25
    ):
        warnings.append("single_blob")
    if int(metrics.get("visible_color_count") or 0) <= 2 and alpha_coverage > 0.05:
        warnings.append("too_few_colors")
    if int(metrics.get("rare_color_count") or 0) >= 8:
        warnings.append("too_many_rare_colors")
    if compare_raw_indexed and raw_indexed_mae is not None and float(raw_indexed_mae) > 0.12:
        warnings.append("quantization_destructive")
    return warnings


def _review_sample(
    generated_dir: Path,
    record: Mapping[str, Any],
    *,
    compare_raw_indexed: bool,
    strict: bool,
    errors: list[str],
) -> dict[str, Any]:
    sample_id = str(record.get("sample_id") or "").strip()
    if not sample_id:
        sample_id = "unknown_sample"

    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    raw_path = _record_path(generated_dir, paths, "raw_rgba", sample_id)
    hard_path = _record_path(generated_dir, paths, "hard_rgba", sample_id)
    indexed_path = _record_path(generated_dir, paths, "indexed_png", sample_id)

    sample_warnings: list[str] = []
    raw_image = _load_optional_png(raw_path, sample_id, "raw_rgba", strict=strict, errors=errors) if raw_path else None
    hard_image = (
        _load_optional_png(hard_path, sample_id, "hard_rgba", strict=strict, errors=errors) if hard_path else None
    )
    indexed_image = (
        _load_optional_png(indexed_path, sample_id, "indexed_png", strict=strict, errors=errors)
        if indexed_path
        else None
    )

    if compare_raw_indexed and raw_image is None:
        sample_warnings.append("raw_missing")
    if indexed_image is None:
        sample_warnings.append("indexed_missing")
        if strict:
            errors.append(f"{sample_id}: indexed_png is required in strict mode")

    alpha_source = hard_image or indexed_image
    color_source = indexed_image or hard_image
    mask = _alpha_mask(alpha_source)

    metrics: dict[str, Any] = {}
    metrics.update(compute_alpha_silhouette_metrics(mask))
    metrics.update(compute_connected_component_metrics(mask))
    metrics.update(compute_color_metrics(color_source))
    metrics.update(compute_edge_metrics(mask, color_source))
    metrics.update(compute_raw_indexed_difference(raw_image, indexed_image) if compare_raw_indexed else {})

    sample_warnings.extend(warning_labels(metrics, compare_raw_indexed=compare_raw_indexed))
    sample_warnings = _ordered_unique(sample_warnings)

    return {
        "sample_id": sample_id,
        "prompt_id": str(record.get("prompt_id") or ""),
        "prompt": str(record.get("prompt") or ""),
        "category": _category(record.get("category")),
        "paths": {
            "raw_rgba": _path_for_report(raw_path),
            "hard_rgba": _path_for_report(hard_path),
            "indexed_png": _path_for_report(indexed_path),
        },
        "metrics": _jsonable(metrics),
        "warnings": sample_warnings,
    }


def _resolve_output_paths(config: GeneratedReviewConfig) -> tuple[Path, Path, Path]:
    generated_dir = Path(config.generated_dir)
    if config.out_dir is not None:
        artifact_dir = Path(config.out_dir)
    elif config.out is not None:
        artifact_dir = Path(config.out).parent
    else:
        artifact_dir = generated_dir / "review"

    markdown_path = Path(config.out) if config.out is not None else artifact_dir / "generated_review_report.md"
    json_path = Path(config.out_json) if config.out_json is not None else markdown_path.with_suffix(".json")
    return artifact_dir, markdown_path, json_path


def _empty_report(generated_dir: Path, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_dir": str(generated_dir),
        "sample_count": 0,
        "connectivity": CONNECTIVITY,
        "overall": _summarize_samples([]),
        "groups": {},
        "errors": list(errors),
        "samples": [],
        "recommendations": ["Fix input artifact errors before structural review."],
    }


def _read_manifest(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path.name}:{line_no}: expected JSON object")
            continue
        records.append(value)
    return records


def _read_generation_report(path: Path, errors: list[str], *, strict: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if strict:
            errors.append(f"generation_report.json is missing: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"generation_report.json is invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append("generation_report.json is not a JSON object")
        return None
    return value


def _record_path(generated_dir: Path, paths: Mapping[str, Any], key: str, sample_id: str) -> Path | None:
    value = paths.get(key)
    if value:
        return _resolve_generated_path(generated_dir, str(value))
    fallback = generated_dir / key / f"{sample_id}.png"
    return fallback if fallback.is_file() else None


def _resolve_generated_path(generated_dir: Path, value: str) -> Path:
    raw = str(value).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return generated_dir / Path(raw.replace("\\", "/"))


def _load_optional_png(
    path: Path, sample_id: str, label: str, *, strict: bool, errors: list[str]
) -> Image.Image | None:
    if not path.is_file():
        if strict:
            errors.append(f"{sample_id}: {label} PNG is missing: {path}")
        return None
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        errors.append(f"{sample_id}: {label} PNG is unreadable: {exc}")
        return None
    copied = image.copy()
    if copied.size != (SPRITE_SIZE, SPRITE_SIZE):
        message = f"{sample_id}: {label} expected 32x32, got {copied.size}"
        if strict:
            errors.append(message)
        return None
    return copied


def _alpha_mask(image: Image.Image | None) -> np.ndarray:
    return _shared_alpha_mask_from_image(image)


def _image_to_rgba_array(image: Image.Image) -> np.ndarray:
    return _shared_image_to_rgba_array(image)


def _transparent_index_used(image: Image.Image) -> bool:
    return _shared_transparent_index_used(image)


def _summarize_samples(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [sample.get("metrics", {}) for sample in samples if isinstance(sample.get("metrics"), Mapping)]
    warning_counts: Counter[str] = Counter()
    for sample in samples:
        warning_counts.update(str(warning) for warning in sample.get("warnings", []))

    return {
        "count": len(samples),
        "mean_alpha_coverage": _mean_metric(metrics, "alpha_coverage"),
        "median_alpha_coverage": _median_metric(metrics, "alpha_coverage"),
        "mean_visible_color_count": _mean_metric(metrics, "visible_color_count"),
        "median_visible_color_count": _median_metric(metrics, "visible_color_count"),
        "mean_connected_components": _mean_metric(metrics, "connected_components"),
        "median_connected_components": _median_metric(metrics, "connected_components"),
        "mean_raw_indexed_rgb_mae_visible": _mean_metric(metrics, "raw_indexed_rgb_mae_visible"),
        "median_raw_indexed_rgb_mae_visible": _median_metric(metrics, "raw_indexed_rgb_mae_visible"),
        "warning_counts": dict(sorted(warning_counts.items())),
        "total_warnings": int(sum(warning_counts.values())),
    }


def _summarize_groups(samples: list[Mapping[str, Any]], *, group_by: str) -> dict[str, Any]:
    if group_by != "category":
        return {}
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        by_category.setdefault(str(sample.get("category") or "unknown"), []).append(sample)
    ordered: dict[str, Any] = {}
    for category in _ordered_categories(by_category):
        ordered[category] = _summarize_samples(list(by_category[category]))
    return ordered


def _ordered_categories(groups: Mapping[str, Any]) -> list[str]:
    known = [category for category in KNOWN_CATEGORIES if category in groups]
    rest = sorted(category for category in groups if category not in KNOWN_CATEGORIES)
    return [*known, *rest]


def _mean_metric(metrics: list[Mapping[str, Any]], key: str) -> float | None:
    values = _numeric_values(metrics, key)
    return float(statistics.fmean(values)) if values else None


def _median_metric(metrics: list[Mapping[str, Any]], key: str) -> float | None:
    values = _numeric_values(metrics, key)
    return float(statistics.median(values)) if values else None


def _numeric_values(metrics: list[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in metrics:
        value = item.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _write_contact_sheets(
    generated_dir: Path,
    artifact_dir: Path,
    samples: list[Mapping[str, Any]],
    *,
    group_by: str,
    compare_raw_indexed: bool,
    max_samples: int,
) -> dict[str, Path]:
    contact_sheets: dict[str, Path] = {}
    overall = artifact_dir / "review_contact_sheet.png"
    if _build_review_contact_sheet(
        generated_dir, samples, overall, compare_raw_indexed=compare_raw_indexed, max_samples=max_samples
    ):
        contact_sheets["overall"] = overall

    if group_by == "category":
        by_category: dict[str, list[Mapping[str, Any]]] = {}
        for sample in samples:
            by_category.setdefault(str(sample.get("category") or "unknown"), []).append(sample)
        for category in _ordered_categories(by_category):
            filename = f"review_contact_sheet_{_safe_name(category)}.png"
            out_path = artifact_dir / filename
            if _build_review_contact_sheet(
                generated_dir,
                by_category[category],
                out_path,
                compare_raw_indexed=compare_raw_indexed,
                max_samples=max_samples,
            ):
                contact_sheets[category] = out_path
    return contact_sheets


def _build_review_contact_sheet(
    generated_dir: Path,
    samples: list[Mapping[str, Any]],
    out_path: Path,
    *,
    compare_raw_indexed: bool,
    max_samples: int,
    scale: int = 4,
) -> bool:
    rows = samples[: max(0, int(max_samples))]
    if not rows:
        return False

    tile_rows: list[list[Image.Image]] = []
    for sample in rows:
        images: list[Image.Image] = []
        paths = sample.get("paths") if isinstance(sample.get("paths"), Mapping) else {}
        if compare_raw_indexed:
            raw = _contact_image_path(generated_dir, paths.get("raw_rgba"))
            raw_image = _open_contact_image(raw)
            if raw_image is not None:
                images.append(raw_image)
        indexed = _contact_image_path(generated_dir, paths.get("indexed_png"))
        hard = _contact_image_path(generated_dir, paths.get("hard_rgba"))
        raw = _contact_image_path(generated_dir, paths.get("raw_rgba"))
        for candidate in (indexed, hard, raw):
            image = _open_contact_image(candidate)
            if image is not None:
                images.append(image)
                break
        if images:
            tile_rows.append(images)

    if not tile_rows:
        return False

    columns = 4 if compare_raw_indexed else 8
    columns = min(columns, max(1, len(tile_rows)))
    subtiles = max(len(images) for images in tile_rows)
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    tile_w = subtiles * cell + (subtiles - 1) * padding
    tile_h = cell
    sheet_rows = (len(tile_rows) + columns - 1) // columns
    sheet = Image.new(
        "RGBA",
        (columns * tile_w + (columns + 1) * padding, sheet_rows * tile_h + (sheet_rows + 1) * padding),
        (38, 38, 42, 255),
    )
    for index, images in enumerate(tile_rows):
        col = index % columns
        row = index // columns
        left = padding + col * (tile_w + padding)
        top = padding + row * (tile_h + padding)
        for sub_index, image in enumerate(images):
            preview = _checkerboard_rgba(image).resize((cell, cell), Image.Resampling.NEAREST)
            sheet.alpha_composite(preview, (left + sub_index * (cell + padding), top))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def _contact_image_path(generated_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    return _resolve_generated_path(generated_dir, str(value))


def _open_contact_image(path: Path | None) -> Image.Image | None:
    if path is None or not path.is_file():
        return None
    try:
        image = Image.open(path)
        image.load()
    except Exception:
        return None
    if image.size != (SPRITE_SIZE, SPRITE_SIZE):
        return None
    return image.convert("RGBA")


def _checkerboard_rgba(image: Image.Image) -> Image.Image:
    return _shared_checkerboard_rgba(image)


def _write_reports(report: Mapping[str, Any], markdown_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(format_generated_review_markdown(report), encoding="utf-8")


def format_generated_review_markdown(report: Mapping[str, Any]) -> str:
    overall = report.get("overall") if isinstance(report.get("overall"), Mapping) else {}
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    samples = report.get("samples") if isinstance(report.get("samples"), list) else []
    warning_counts = overall.get("warning_counts") if isinstance(overall.get("warning_counts"), Mapping) else {}
    categories = ", ".join(groups.keys()) if groups else "(not grouped)"
    common_warnings = _common_warning_text(warning_counts)

    lines = [
        "# Generated Sample Review",
        "",
        "## Summary",
        "",
        f"- Samples: {int(report.get('sample_count', 0))}",
        f"- Categories: {categories}",
        f"- Connectivity: {report.get('connectivity', CONNECTIVITY)}",
        f"- Mean alpha coverage: {_fmt_float(overall.get('mean_alpha_coverage'))}",
        f"- Median visible colors: {_fmt_float(overall.get('median_visible_color_count'))}",
        f"- Most common warnings: {common_warnings}",
        "",
        "## Group Summary",
        "",
        "| Category | Count | Alpha coverage | Colors | Components | Warnings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if groups:
        for category, summary in groups.items():
            if not isinstance(summary, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(str(category)),
                        str(int(summary.get("count", 0))),
                        _fmt_float(summary.get("mean_alpha_coverage")),
                        _fmt_float(summary.get("mean_visible_color_count")),
                        _fmt_float(summary.get("mean_connected_components")),
                        str(int(summary.get("total_warnings", 0))),
                    ]
                )
                + " |"
            )
    else:
        lines.append(
            "| all | "
            + str(int(overall.get("count", 0)))
            + " | "
            + " | ".join(
                [
                    _fmt_float(overall.get("mean_alpha_coverage")),
                    _fmt_float(overall.get("mean_visible_color_count")),
                    _fmt_float(overall.get("mean_connected_components")),
                    str(int(overall.get("total_warnings", 0))),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Worst Samples by Warning Count",
            "",
            "| Sample | Prompt | Category | Warnings |",
            "|---|---|---|---|",
        ]
    )
    worst = sorted(samples, key=lambda sample: (-len(sample.get("warnings", [])), str(sample.get("sample_id", ""))))[
        :10
    ]
    if worst:
        for sample in worst:
            warnings = ", ".join(str(warning) for warning in sample.get("warnings", [])) or "(none)"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_md_escape(str(sample.get('sample_id', '')))}`",
                        _md_escape(_shorten(str(sample.get("prompt", "")), 72)),
                        _md_escape(str(sample.get("category", ""))),
                        _md_escape(warnings),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| (none) |  |  |  |")

    lines.extend(["", "## Quantization Impact", ""])
    mean_mae = overall.get("mean_raw_indexed_rgb_mae_visible")
    if mean_mae is None:
        lines.append("Raw-vs-indexed comparison was not computed.")
    else:
        destructive = int(warning_counts.get("quantization_destructive", 0))
        lines.append(f"- Mean raw/indexed RGB MAE on visible pixels: {_fmt_float(mean_mae)}")
        lines.append(f"- Quantization destructive warnings: {destructive}")

    lines.extend(["", "## Recommendations", ""])
    recommendations = report.get("recommendations") if isinstance(report.get("recommendations"), list) else []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- No deterministic recommendation was triggered.")

    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    if errors:
        lines.extend(["", "## Input Errors", ""])
        lines.extend(f"- {_md_escape(str(error))}" for error in errors)

    lines.append("")
    return "\n".join(lines)


def _recommendations(report: Mapping[str, Any]) -> list[str]:
    overall = report.get("overall") if isinstance(report.get("overall"), Mapping) else {}
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    sample_count = max(1, int(report.get("sample_count") or 0))
    warning_counts = overall.get("warning_counts") if isinstance(overall.get("warning_counts"), Mapping) else {}
    recs: list[str] = []

    if int(warning_counts.get("fragmented", 0)) / sample_count >= 0.25:
        recs.append(
            "Many samples are fragmented; later model work should consider stronger alpha/silhouette loss or connectedness regularization."
        )
    if int(warning_counts.get("empty_or_nearly_empty", 0)) / sample_count >= 0.20:
        recs.append("Many samples are empty or nearly empty; later training should consider positive-alpha weighting.")
    if int(warning_counts.get("too_full_canvas", 0)) / sample_count >= 0.20:
        recs.append(
            "Many samples fill too much canvas; later training should strengthen framing and background separation."
        )
    if int(warning_counts.get("touches_border", 0)) / sample_count >= 0.30:
        recs.append(
            "Many samples touch the canvas border; later training or sampling should improve icon framing and sprite margins."
        )
    if int(warning_counts.get("quantization_destructive", 0)) / sample_count >= 0.10:
        recs.append(
            "Raw-vs-indexed differences are high; later training should consider quantization-aware or palette-aware losses."
        )
    if int(warning_counts.get("too_many_rare_colors", 0)) / sample_count >= 0.25:
        recs.append(
            "Many samples contain rare colors; later training should reduce chroma noise or use palette-aware output constraints."
        )

    if groups and "seen_object" in groups:
        seen = groups.get("seen_object")
        unseen_names = ("unseen_composition", "creative_concept", "style_stress")
        unseen_summaries = [groups[name] for name in unseen_names if isinstance(groups.get(name), Mapping)]
        if isinstance(seen, Mapping) and unseen_summaries:
            seen_rate = _warning_rate(seen)
            unseen_total_count = sum(int(summary.get("count", 0)) for summary in unseen_summaries)
            unseen_total_warnings = sum(int(summary.get("total_warnings", 0)) for summary in unseen_summaries)
            unseen_rate = unseen_total_warnings / float(unseen_total_count) if unseen_total_count else 0.0
            if unseen_total_count and unseen_rate >= seen_rate + 0.5:
                recs.append(
                    "Seen prompts are structurally cleaner than unseen or creative prompts; add broader source-pack coverage before escalating architecture."
                )

    if not recs:
        recs.append("No high-frequency structural warning crossed deterministic recommendation thresholds.")
    return recs


def _warning_rate(summary: Mapping[str, Any]) -> float:
    count = int(summary.get("count", 0))
    if count <= 0:
        return 0.0
    return int(summary.get("total_warnings", 0)) / float(count)


def _common_warning_text(warning_counts: Mapping[str, Any]) -> str:
    if not warning_counts:
        return "(none)"
    items = sorted(
        ((str(key), int(value)) for key, value in warning_counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(f"{key}={count}" for key, count in items[:5])


def _category(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value.strip())
    return cleaned or "unknown"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    order = {name: index for index, name in enumerate(WARNING_ORDER)}
    for value in sorted(values, key=lambda item: (order.get(item, 999), item)):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _path_for_report(path: Path | None) -> str | None:
    return None if path is None else str(path)


from spritelab.training.report_utils import fmt_float as _fmt_float


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


from spritelab.training.report_utils import jsonable as _jsonable


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Deterministically review generated sprite sample folders.")
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--group-by", choices=["category", "none"], default="none")
    parser.add_argument("--max-samples-per-sheet", type=int, default=64)
    parser.add_argument("--compare-raw-indexed", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parsed = parser.parse_args(argv)

    result = review_generated_sprites(GeneratedReviewConfig(**vars(parsed)))
    print(f"Reviewed samples: {result.report['sample_count']}")
    print(f"Markdown report: {result.markdown_path}")
    print(f"JSON report: {result.json_path}")
    if result.contact_sheets:
        print(f"Contact sheets: {len(result.contact_sheets)}")
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
