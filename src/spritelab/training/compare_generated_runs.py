"""Compare two generated sprite sample folders."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.training.framing_metrics import checkerboard_rgba
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites
from spritelab.training.prompt_sensitivity import pairwise_image_metrics

SCHEMA_VERSION = "compare_generated_runs_v1.0"
SPRITE_SIZE = 32


@dataclass(frozen=True)
class CompareGeneratedRunsConfig:
    a: Path
    b: Path
    out_dir: Path
    max_contact_sheet_pairs: int = 64


def compare_generated_runs(config: CompareGeneratedRunsConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_dir = Path(config.a)
    b_dir = Path(config.b)

    a_review, a_review_source = _load_or_compute_review(a_dir, out_dir, "a")
    b_review, b_review_source = _load_or_compute_review(b_dir, out_dir, "b")
    a_manifest = _read_manifest(a_dir / "generated_manifest.jsonl")
    b_manifest = _read_manifest(b_dir / "generated_manifest.jsonl")
    a_summary = _summarize_run(a_dir, a_review, a_manifest, review_source=a_review_source)
    b_summary = _summarize_run(b_dir, b_review, b_manifest, review_source=b_review_source)
    image_deltas = _matched_image_deltas(a_dir, b_dir, a_review, b_review)
    contact_sheet = _write_compare_contact_sheet(
        a_dir,
        b_dir,
        image_deltas,
        out_dir / "compare_contact_sheet.png",
        max_pairs=config.max_contact_sheet_pairs,
    )
    warnings = []
    if int(a_summary["sample_count"]) != int(b_summary["sample_count"]):
        warnings.append("different_sample_counts")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "a": a_summary,
        "b": b_summary,
        "deltas": _summary_deltas(a_summary, b_summary),
        "warnings": warnings,
        "matched_image_count": len(image_deltas),
        "matched_image_summary": _summarize_image_deltas(image_deltas),
        "matched_images": image_deltas,
        "contact_sheet": None if contact_sheet is None else contact_sheet.name,
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
    }
    (out_dir / "compare_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "compare_report.md").write_text(format_compare_markdown(report), encoding="utf-8")
    return report


def format_compare_markdown(report: Mapping[str, Any]) -> str:
    a = report.get("a") if isinstance(report.get("a"), Mapping) else {}
    b = report.get("b") if isinstance(report.get("b"), Mapping) else {}
    deltas = report.get("deltas") if isinstance(report.get("deltas"), Mapping) else {}
    image_summary = report.get("matched_image_summary") if isinstance(report.get("matched_image_summary"), Mapping) else {}
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []

    lines = [
        "# Generated Run Comparison",
        "",
        f"A: `{a.get('generated_dir', '')}`",
        f"B: `{b.get('generated_dir', '')}`",
        "",
        "## Summary",
        "",
        "| Metric | A | B | Delta B-A |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "sample_count",
        "border_touch_rate",
        "mean_alpha_coverage",
        "mean_connected_components",
        "mean_visible_color_count",
        "mean_rare_color_count",
        "mean_quantization_mae",
        "total_warnings",
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    _fmt(a.get(key)),
                    _fmt(b.get(key)),
                    _fmt(deltas.get(key)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## QA",
            "",
            f"- A QA: {_qa_text(a.get('generated_qa'))}",
            f"- B QA: {_qa_text(b.get('generated_qa'))}",
            "",
            "## Matched Images",
            "",
            f"- Matched image count: {int(report.get('matched_image_count') or 0)}",
            f"- Mean matched alpha IoU: {_fmt(image_summary.get('mean_alpha_iou'))}",
            f"- Mean matched RGB distance: {_fmt(image_summary.get('mean_rgb_distance'))}",
            f"- Mean matched combined difference: {_fmt(image_summary.get('mean_combined_difference'))}",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def _load_or_compute_review(generated_dir: Path, out_dir: Path, label: str) -> tuple[dict[str, Any], str]:
    for candidate in (
        generated_dir / "generated_review_report.json",
        generated_dir / "review" / "generated_review_report.json",
    ):
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value, str(candidate)

    review_dir = out_dir / "derived_reviews" / label
    result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=generated_dir,
            out=review_dir / "generated_review_report.md",
            out_json=review_dir / "generated_review_report.json",
            out_dir=review_dir,
            group_by="category",
            compare_raw_indexed=True,
        )
    )
    return result.report, str(review_dir / "generated_review_report.json")


def _summarize_run(
    generated_dir: Path,
    review: Mapping[str, Any],
    manifest_records: Sequence[Mapping[str, Any]],
    *,
    review_source: str,
) -> dict[str, Any]:
    samples = review.get("samples") if isinstance(review.get("samples"), list) else []
    metrics = [sample.get("metrics", {}) for sample in samples if isinstance(sample, Mapping)]
    warning_counts: Counter[str] = Counter()
    border_touch = 0
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        warning_counts.update(str(warning) for warning in sample.get("warnings", []))
        sample_metrics = sample.get("metrics") if isinstance(sample.get("metrics"), Mapping) else {}
        if bool(sample_metrics.get("touches_border")):
            border_touch += 1

    sample_count = int(review.get("sample_count") or len(samples) or len(manifest_records))
    return {
        "generated_dir": str(generated_dir),
        "sample_count": sample_count,
        "review_source": review_source,
        "generated_qa": _read_generated_qa(generated_dir),
        "border_touch_count": int(border_touch),
        "border_touch_rate": border_touch / float(sample_count) if sample_count else 0.0,
        "mean_alpha_coverage": _mean_metric(metrics, "alpha_coverage"),
        "mean_connected_components": _mean_metric(metrics, "connected_components"),
        "mean_visible_color_count": _mean_metric(metrics, "visible_color_count"),
        "mean_rare_color_count": _mean_metric(metrics, "rare_color_count"),
        "mean_quantization_mae": _mean_metric(metrics, "raw_indexed_rgb_mae_visible"),
        "warning_counts": dict(sorted(warning_counts.items())),
        "total_warnings": int(sum(warning_counts.values())),
        "prompt_category_summaries": review.get("groups") if isinstance(review.get("groups"), Mapping) else {},
        "generation_report": _read_json_if_present(generated_dir / "generation_report.json"),
    }


def _summary_deltas(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for key in (
        "sample_count",
        "border_touch_rate",
        "mean_alpha_coverage",
        "mean_connected_components",
        "mean_visible_color_count",
        "mean_rare_color_count",
        "mean_quantization_mae",
        "total_warnings",
    ):
        a_value = a.get(key)
        b_value = b.get(key)
        if isinstance(a_value, (int, float)) and isinstance(b_value, (int, float)):
            deltas[key] = float(b_value) - float(a_value)
        else:
            deltas[key] = None
    return deltas


def _matched_image_deltas(
    a_dir: Path,
    b_dir: Path,
    a_review: Mapping[str, Any],
    b_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    a_samples = [sample for sample in a_review.get("samples", []) if isinstance(sample, Mapping)]
    b_samples = [sample for sample in b_review.get("samples", []) if isinstance(sample, Mapping)]
    b_by_sample_id = {str(sample.get("sample_id", "")): sample for sample in b_samples if sample.get("sample_id")}
    b_by_prompt_id = {str(sample.get("prompt_id", "")): sample for sample in b_samples if sample.get("prompt_id")}
    matched: list[dict[str, Any]] = []
    used_b: set[str] = set()
    for a_sample in a_samples:
        match_key = "sample_id"
        b_sample = b_by_sample_id.get(str(a_sample.get("sample_id", "")))
        if b_sample is None:
            match_key = "prompt_id"
            b_sample = b_by_prompt_id.get(str(a_sample.get("prompt_id", "")))
        if b_sample is None:
            continue
        b_id = str(b_sample.get("sample_id", ""))
        if b_id in used_b:
            continue
        used_b.add(b_id)
        image_a = _open_sample_image(a_dir, a_sample)
        image_b = _open_sample_image(b_dir, b_sample)
        if image_a is None or image_b is None:
            continue
        metrics = pairwise_image_metrics(image_a, image_b)
        matched.append(
            {
                "match_key": match_key,
                "sample_id_a": str(a_sample.get("sample_id", "")),
                "sample_id_b": str(b_sample.get("sample_id", "")),
                "prompt_id_a": str(a_sample.get("prompt_id", "")),
                "prompt_id_b": str(b_sample.get("prompt_id", "")),
                "prompt": str(a_sample.get("prompt") or b_sample.get("prompt") or ""),
                "metrics": metrics,
            }
        )
    return matched


def _summarize_image_deltas(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [item.get("metrics", {}) for item in items if isinstance(item.get("metrics"), Mapping)]
    return {
        "mean_alpha_iou": _mean_metric(metrics, "alpha_iou"),
        "mean_rgb_distance": _mean_metric(metrics, "rgb_mae_visible_union"),
        "mean_histogram_distance": _mean_metric(metrics, "rgb_histogram_distance"),
        "mean_combined_difference": _mean_metric(metrics, "combined_difference_score"),
        "near_duplicate_rate": _mean([1.0 if metric.get("near_duplicate") else 0.0 for metric in metrics]),
    }


def _write_compare_contact_sheet(
    a_dir: Path,
    b_dir: Path,
    image_deltas: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    max_pairs: int,
    scale: int = 4,
) -> Path | None:
    rows = list(image_deltas)[: max(0, int(max_pairs))]
    if not rows:
        return None
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    columns = 4
    tile_w = cell * 2 + padding
    tile_h = cell
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGBA",
        (columns * tile_w + (columns + 1) * padding, sheet_rows * tile_h + (sheet_rows + 1) * padding),
        (38, 38, 42, 255),
    )
    a_samples = _sample_index(a_dir)
    b_samples = _sample_index(b_dir)
    for index, item in enumerate(rows):
        a_sample = a_samples.get(str(item.get("sample_id_a", "")))
        b_sample = b_samples.get(str(item.get("sample_id_b", "")))
        image_a = _open_sample_image(a_dir, a_sample) if a_sample is not None else None
        image_b = _open_sample_image(b_dir, b_sample) if b_sample is not None else None
        if image_a is None or image_b is None:
            continue
        col = index % columns
        row = index // columns
        left = padding + col * (tile_w + padding)
        top = padding + row * (tile_h + padding)
        sheet.alpha_composite(checkerboard_rgba(image_a).resize((cell, cell), Image.Resampling.NEAREST), (left, top))
        sheet.alpha_composite(
            checkerboard_rgba(image_b).resize((cell, cell), Image.Resampling.NEAREST),
            (left + cell + padding, top),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _sample_index(generated_dir: Path) -> dict[str, Mapping[str, Any]]:
    return {str(record.get("sample_id", "")): record for record in _read_manifest(generated_dir / "generated_manifest.jsonl")}


def _open_sample_image(generated_dir: Path, sample: Mapping[str, Any] | None) -> Image.Image | None:
    if sample is None:
        return None
    paths = sample.get("paths") if isinstance(sample.get("paths"), Mapping) else {}
    rel = paths.get("indexed_png") or paths.get("hard_rgba") or paths.get("raw_rgba")
    if not rel:
        return None
    path = _resolve_sample_path(generated_dir, str(rel))
    if not path.is_file():
        return None
    image = Image.open(path).convert("RGBA")
    image.load()
    return image.copy()


def _resolve_sample_path(generated_dir: Path, value: str) -> Path:
    raw = str(value).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return path
    joined = generated_dir / path
    if joined.is_file():
        return joined
    if path.is_file():
        return path
    parts = path.parts
    generated_name = generated_dir.name
    if generated_name in parts:
        start = parts.index(generated_name) + 1
        candidate = generated_dir.joinpath(*parts[start:])
        if candidate.is_file():
            return candidate
    return joined


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_generated_qa(generated_dir: Path) -> dict[str, Any] | None:
    value = _read_json_if_present(generated_dir / "generated_qa_report.json")
    if not isinstance(value, Mapping):
        return None
    return {
        "ok": bool(value.get("ok")),
        "errors": len(value.get("errors") or []),
        "warnings": len(value.get("warnings") or []),
        "sample_count": int(value.get("sample_count") or 0),
    }


def _read_json_if_present(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _mean_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(metric[key])
        for metric in metrics
        if isinstance(metric.get(key), (int, float)) and math.isfinite(float(metric[key]))
    ]
    return float(statistics.fmean(values)) if values else None


def _mean(values: Sequence[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(data)) if data else 0.0


def _qa_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "not available"
    status = "PASS" if value.get("ok") else "FAIL"
    return f"{status}, errors={int(value.get('errors') or 0)}, warnings={int(value.get('warnings') or 0)}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare two generated sprite sample folders.")
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--max-contact-sheet-pairs", type=int, default=64)
    parsed = parser.parse_args(argv)
    report = compare_generated_runs(CompareGeneratedRunsConfig(**vars(parsed)))
    deltas = report["deltas"]
    print(f"A samples: {report['a']['sample_count']}")
    print(f"B samples: {report['b']['sample_count']}")
    print(f"Border-touch delta B-A: {deltas['border_touch_rate']:.6f}")
    print(f"Outputs written to {parsed.out_dir}")


if __name__ == "__main__":
    main()
