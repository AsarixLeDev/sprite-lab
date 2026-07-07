"""Compare generated samples against their exact source sprites."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.training.data import REQUIRED_NPZ_KEYS, read_jsonl
from spritelab.training.prompt_sensitivity import pairwise_image_metrics
from spritelab.training.rgba import npz_row_to_rgba

SCHEMA_VERSION = "source_match_review_v1.0"
SPRITE_SIZE = 32


@dataclass(frozen=True)
class SourceMatchReviewConfig:
    generated: Path
    dataset: Path
    training_manifest: Path
    out: Path
    out_json: Path | None = None


def run_source_match_review(config: SourceMatchReviewConfig) -> dict[str, Any]:
    generated_dir = Path(config.generated)
    manifest_records = _read_generated_manifest(generated_dir / "generated_manifest.jsonl")
    source_index = load_source_sprite_index(config.dataset, config.training_manifest)

    samples: list[dict[str, Any]] = []
    for record in manifest_records:
        sample = _review_generated_record(generated_dir, record, source_index)
        samples.append(sample)

    matched = [sample for sample in samples if sample.get("matched_source")]
    metric_rows = [sample.get("metrics", {}) for sample in matched if isinstance(sample.get("metrics"), Mapping)]
    exactish = [sample for sample in matched if bool(sample.get("exactish_match"))]
    near = [sample for sample in matched if bool(sample.get("near_match"))]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated": str(generated_dir),
        "dataset": str(config.dataset),
        "training_manifest": str(config.training_manifest),
        "sample_count": len(samples),
        "matched_source_count": len(matched),
        "missing_source_count": len(samples) - len(matched),
        "mean_visible_rgb_mae": _mean_metric(metric_rows, "visible_rgb_mae"),
        "mean_alpha_iou": _mean_metric(metric_rows, "alpha_iou"),
        "mean_alpha_mae": _mean_metric(metric_rows, "alpha_mae"),
        "mean_bbox_center_distance": _mean_metric(metric_rows, "bbox_center_distance"),
        "mean_palette_overlap": _mean_metric(metric_rows, "palette_overlap"),
        "mean_visible_color_histogram_distance": _mean_metric(metric_rows, "rgb_histogram_distance"),
        "exactish_match_rate": len(exactish) / float(len(matched)) if matched else 0.0,
        "near_match_rate": len(near) / float(len(matched)) if matched else 0.0,
        "best_examples": _ranked_examples(matched, best=True),
        "worst_examples": _ranked_examples(matched, best=False),
        "samples": samples,
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
    }
    md_path, json_path = _resolve_output_paths(config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(format_source_match_markdown(report), encoding="utf-8")
    return report


def load_source_sprite_index(dataset: str | Path, training_manifest: str | Path) -> dict[str, dict[str, Any]]:
    """Load one canonical RGBA source image and metadata row per sprite ID."""

    dataset_dir = Path(dataset)
    rows = read_jsonl(training_manifest)
    first_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        sprite_id = str(row.get("sprite_id", "")).strip()
        if sprite_id and sprite_id not in first_rows:
            first_rows[sprite_id] = dict(row)

    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    index: dict[str, dict[str, Any]] = {}
    for sprite_id, row in first_rows.items():
        npz_file = str(row.get("npz_file") or f"{row.get('split', '')}.npz")
        npz_row = int(row.get("npz_row", -1))
        arrays = npz_cache.get(npz_file)
        if arrays is None:
            with np.load(dataset_dir / npz_file, allow_pickle=False) as data:
                missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
                if missing:
                    raise ValueError(f"{dataset_dir / npz_file}: missing required arrays: {', '.join(missing)}")
                arrays = {key: data[key] for key in data.files}
            npz_cache[npz_file] = arrays
        rgba = npz_row_to_rgba(
            index_map=arrays["index_map"][npz_row],
            alpha=arrays["alpha"][npz_row],
            palette=arrays["palette"][npz_row],
            palette_mask=arrays["palette_mask"][npz_row],
        )
        index[sprite_id] = {
            "sprite_id": sprite_id,
            "metadata": row,
            "image": _rgba_chw_to_image(rgba),
        }
    return index


def compute_source_match_metrics(generated: Image.Image, source: Image.Image) -> dict[str, Any]:
    metrics = pairwise_image_metrics(generated, source)
    metrics["visible_rgb_mae"] = metrics["rgb_mae_visible_union"]
    metrics["palette_overlap"] = _palette_overlap(generated, source)
    return metrics


def format_source_match_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Source Match Review",
        "",
        f"Generated: `{report.get('generated', '')}`",
        f"Samples: {int(report.get('sample_count') or 0)}",
        f"Matched sources: {int(report.get('matched_source_count') or 0)}",
        "",
        "## Summary",
        "",
        f"- Mean visible RGB MAE: {_fmt(report.get('mean_visible_rgb_mae'))}",
        f"- Mean alpha IoU: {_fmt(report.get('mean_alpha_iou'))}",
        f"- Mean alpha MAE: {_fmt(report.get('mean_alpha_mae'))}",
        f"- Mean bbox center distance: {_fmt(report.get('mean_bbox_center_distance'))}",
        f"- Mean palette overlap: {_fmt(report.get('mean_palette_overlap'))}",
        f"- Exact-ish match rate: {_fmt(report.get('exactish_match_rate'))}",
        f"- Near-match rate: {_fmt(report.get('near_match_rate'))}",
        "",
        "## Best Examples",
        "",
        "| Sample | Source | Object | RGB MAE | Alpha IoU | Near |",
        "|---|---|---|---:|---:|---|",
    ]
    lines.extend(_example_rows(report.get("best_examples")))
    lines.extend(
        [
            "",
            "## Worst Examples",
            "",
            "| Sample | Source | Object | RGB MAE | Alpha IoU | Near |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    lines.extend(_example_rows(report.get("worst_examples")))
    lines.append("")
    return "\n".join(lines)


def _review_generated_record(
    generated_dir: Path,
    record: Mapping[str, Any],
    source_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sample_id = str(record.get("sample_id", ""))
    target_sprite_id = _target_sprite_id(record, source_index)
    source = source_index.get(target_sprite_id) if target_sprite_id else None
    base = {
        "sample_id": sample_id,
        "prompt_id": str(record.get("prompt_id") or ""),
        "prompt": str(record.get("prompt") or ""),
        "target_sprite_id": target_sprite_id,
        "matched_source": source is not None,
    }
    if source is None:
        return {**base, "metrics": {}, "warnings": ["missing_source_mapping"]}
    image = _open_generated_image(generated_dir, record)
    metrics = compute_source_match_metrics(image, source["image"])
    metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
    exactish = _is_exactish(metrics)
    near = _is_near_match(metrics)
    return {
        **base,
        "source_object_name": str(metadata.get("object_name") or ""),
        "source_category": str(metadata.get("category") or ""),
        "metrics": _jsonable(metrics),
        "exactish_match": exactish,
        "near_match": near,
        "warnings": [] if near else ["not_near_match"],
    }


def _target_sprite_id(record: Mapping[str, Any], source_index: Mapping[str, Any]) -> str:
    for key in ("target_sprite_id", "source_sprite_id", "sprite_id"):
        value = str(record.get(key) or "").strip()
        if value in source_index:
            return value
    prompt_id = str(record.get("prompt_id") or "").strip()
    if prompt_id in source_index:
        return prompt_id
    return str(record.get("target_sprite_id") or record.get("source_sprite_id") or record.get("sprite_id") or "")


def _open_generated_image(generated_dir: Path, record: Mapping[str, Any]) -> Image.Image:
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    rel = paths.get("indexed_png") or paths.get("hard_rgba") or paths.get("raw_rgba")
    if not rel:
        raise ValueError(f"{record.get('sample_id', '')}: generated record has no image path")
    image = Image.open(generated_dir / str(rel)).convert("RGBA")
    image.load()
    return image.copy()


def _read_generated_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"generated_manifest.jsonl is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def _palette_overlap(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGBA"), dtype=np.uint8)
    arr_b = np.asarray(b.convert("RGBA"), dtype=np.uint8)
    colors_a = {tuple(int(v) for v in rgb) for rgb in arr_a[..., :3][arr_a[..., 3] > 0]}
    colors_b = {tuple(int(v) for v in rgb) for rgb in arr_b[..., :3][arr_b[..., 3] > 0]}
    union = colors_a | colors_b
    if not union:
        return 1.0
    return len(colors_a & colors_b) / float(len(union))


def _is_exactish(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics.get("alpha_iou") or 0.0) >= 0.995
        and float(metrics.get("visible_rgb_mae") or 0.0) <= 0.005
        and float(metrics.get("alpha_mae") or 0.0) <= 0.005
    )


def _is_near_match(metrics: Mapping[str, Any]) -> bool:
    center = metrics.get("bbox_center_distance")
    center_ok = center is None or float(center) <= 4.0
    return (
        float(metrics.get("alpha_iou") or 0.0) >= 0.90
        and float(metrics.get("visible_rgb_mae") or 0.0) <= 0.08
        and center_ok
    )


def _ranked_examples(samples: Sequence[Mapping[str, Any]], *, best: bool) -> list[dict[str, Any]]:
    def score(sample: Mapping[str, Any]) -> float:
        metrics = sample.get("metrics") if isinstance(sample.get("metrics"), Mapping) else {}
        value = metrics.get("combined_difference_score")
        return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else float("inf")

    ordered = sorted(samples, key=lambda sample: (score(sample), str(sample.get("sample_id", ""))))
    if not best:
        ordered = list(reversed(ordered))
    return [dict(item) for item in ordered[:10]]


def _example_rows(examples: Any) -> list[str]:
    rows: list[str] = []
    for sample in examples or []:
        if not isinstance(sample, Mapping):
            continue
        metrics = sample.get("metrics") if isinstance(sample.get("metrics"), Mapping) else {}
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_escape(str(sample.get('sample_id', '')))}`",
                    f"`{_md_escape(str(sample.get('target_sprite_id', '')))}`",
                    _md_escape(str(sample.get("source_object_name") or "")),
                    _fmt(metrics.get("visible_rgb_mae")),
                    _fmt(metrics.get("alpha_iou")),
                    "yes" if sample.get("near_match") else "no",
                ]
            )
            + " |"
        )
    return rows or ["| (none) |  |  |  |  |  |"]


def _mean_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(metric[key])
        for metric in metrics
        if isinstance(metric.get(key), (int, float)) and math.isfinite(float(metric[key]))
    ]
    return float(statistics.fmean(values)) if values else None


def _resolve_output_paths(config: SourceMatchReviewConfig) -> tuple[Path, Path]:
    out = Path(config.out)
    if out.suffix.lower() == ".md":
        md_path = out
        json_path = config.out_json or out.with_suffix(".json")
    elif out.suffix.lower() == ".json":
        json_path = out
        md_path = out.with_suffix(".md")
    else:
        md_path = out / "source_match_report.md"
        json_path = config.out_json or out / "source_match_report.json"
    return md_path, Path(json_path)


def _rgba_chw_to_image(rgba: np.ndarray) -> Image.Image:
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.shape != (4, SPRITE_SIZE, SPRITE_SIZE):
        raise ValueError(f"RGBA image must have shape [4, 32, 32], got {arr.shape}")
    hwc = np.moveaxis(np.clip(arr, 0.0, 1.0), 0, -1)
    return Image.fromarray(np.rint(hwc * 255.0).astype(np.uint8), mode="RGBA")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items() if key != "image"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Review generated sprites against exact source targets.")
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    parsed = parser.parse_args(argv)
    report = run_source_match_review(SourceMatchReviewConfig(**vars(parsed)))
    print(f"Matched sources: {report['matched_source_count']}/{report['sample_count']}")
    print(f"Mean alpha IoU: {_fmt(report.get('mean_alpha_iou'))}")


if __name__ == "__main__":
    main()
