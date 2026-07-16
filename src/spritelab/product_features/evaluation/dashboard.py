"""Dashboard projections over existing evaluation report data."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spritelab.product_core import finite_json_copy, strict_json_bytes, validate_finite_json


class IncompatibleMetricDefinitions(ValueError):
    """Raised before incompatible metrics could be averaged or compared."""


@dataclass(frozen=True)
class MetricCard:
    metric_id: str
    title: str
    value: int | float | str | None
    unit: str = ""
    status: str = "AVAILABLE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChartSpec:
    chart_id: str
    title: str
    kind: str
    status: str
    series: tuple[dict[str, Any], ...] = ()
    no_data_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["series"] = list(self.series)
        return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metric_cards(report: Mapping[str, Any]) -> list[MetricCard]:
    summary = _mapping(report.get("summary"))
    memo = _mapping(summary.get("memorization"))
    values = (
        ("samples", "Samples", _number(summary.get("sample_count")), ""),
        ("validity", "Structural validity", _number(_nested(summary, "hard_validity", "pass_rate")), "%"),
        ("conditional", "Conditional adherence", _number(_nested(summary, "conditional", "represented_rate")), "%"),
        ("palette", "Mean palette size", _number(_nested(summary, "pixel_art", "palette_size_mean")), " colors"),
        ("diversity", "Exact duplicate rate", _number(_nested(summary, "diversity", "exact_duplicate_rate")), "%"),
        (
            "memorization",
            "Memorization",
            report.get("promotion", {}).get("memorization_machine_status")
            if isinstance(report.get("promotion"), Mapping)
            else memo.get("machine_status"),
            "",
        ),
    )
    cards: list[MetricCard] = []
    for metric_id, title, value, unit in values:
        if value is None:
            cards.append(MetricCard(metric_id, title, None, unit, "NO_DATA"))
        else:
            display: int | float | str = value
            if unit == "%" and isinstance(value, (int, float)):
                display = round(float(value) * 100.0, 2)
            cards.append(MetricCard(metric_id, title, display, unit))
    return cards


def _values(rows: Sequence[Mapping[str, Any]], *path: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = _nested(row, *path)
        if (number := _number(value)) is not None:
            result.append(float(number))
    return result


def _histogram(values: Sequence[float], *, bins: int = 8) -> tuple[dict[str, Any], ...]:
    if not values:
        return ()
    low, high = min(values), max(values)
    if low == high:
        return ({"label": f"{low:.3g}", "value": len(values)},)
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] += 1
    return tuple(
        {"label": f"{low + index * width:.3g}-{low + (index + 1) * width:.3g}", "value": count}
        for index, count in enumerate(counts)
    )


def _chart(chart_id: str, title: str, values: Sequence[float]) -> ChartSpec:
    series = _histogram(values)
    return ChartSpec(
        chart_id=chart_id,
        title=title,
        kind="histogram",
        status="AVAILABLE" if series else "NO_DATA",
        series=series,
        no_data_message=None if series else "No comparable data is available for this chart.",
    )


def _aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get(key) or "unknown")
        groups[label].append(row)
    result: list[dict[str, Any]] = []
    for label, group in sorted(groups.items()):
        adherence = [
            float(number) for row in group if (number := _number(row.get("conditional_adherence"))) is not None
        ]
        valid = [not bool(row.get("generation_failed")) for row in group]
        result.append(
            {
                "name": label,
                "sample_count": len(group),
                "structural_validity_rate": sum(valid) / len(valid) if valid else None,
                "conditional_adherence": sum(adherence) / len(adherence) if adherence else None,
            }
        )
    return result


def sample_gallery(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return public gallery records without source or training filesystem paths."""

    gallery: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("generated_sample_id") or "sample")
        nearest = row.get("training_neighbors") if isinstance(row.get("training_neighbors"), list) else []
        raw_checkpoint = str(row.get("checkpoint_identity") or row.get("checkpoint") or "")
        public_checkpoint = raw_checkpoint.replace("\\", "/").rsplit("/", 1)[-1]
        gallery.append(
            {
                "sample_id": sample_id,
                "image_reference": str(row.get("artifact_id") or sample_id),
                "prompt": str(row.get("prompt") or ""),
                "seed": row.get("seed"),
                "checkpoint": public_checkpoint,
                "weights": str(
                    row.get("weights") or ("ema" if str(row.get("checkpoint", "")).endswith("_ema.pt") else "live")
                ),
                "category": str(row.get("category") or "unknown"),
                "metrics": dict(_mapping(row.get("metrics"))),
                "conditional_adherence": row.get("conditional_adherence"),
                "memorization_evidence_class": row.get("memorization_evidence_class"),
                "nearest_match_summary": {
                    "evidence_class": nearest[0].get("evidence_class"),
                    "evidence_strength": nearest[0].get("evidence_strength"),
                }
                if nearest and isinstance(nearest[0], Mapping)
                else None,
            }
        )
    return gallery


def filter_gallery(
    samples: Sequence[Mapping[str, Any]],
    *,
    prompt: str | None = None,
    seed: int | None = None,
    checkpoint: str | None = None,
    weights: str | None = None,
    category: str | None = None,
    sort_metric: str | None = None,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Filter/sort gallery data; unknown metrics sort as no-data instead of failing."""

    selected = []
    for raw in samples:
        row = dict(raw)
        if prompt and prompt.lower() not in str(row.get("prompt") or "").lower():
            continue
        if seed is not None and row.get("seed") != seed:
            continue
        if checkpoint and row.get("checkpoint") != checkpoint:
            continue
        if weights and row.get("weights") != weights:
            continue
        if category and row.get("category") != category:
            continue
        selected.append(row)
    if sort_metric:
        path = tuple(sort_metric.split("."))

        def key(row: Mapping[str, Any]) -> tuple[bool, float]:
            value = _nested(row, *path)
            number = _number(value)
            return (number is None, float(number) if number is not None else 0.0)

        selected.sort(key=key, reverse=descending)
    return selected


def build_dashboard(
    report: Mapping[str, Any],
    per_image_rows: Sequence[Mapping[str, Any]] = (),
    *,
    allow_source_results: bool = False,
) -> dict[str, Any]:
    rows = tuple(per_image_rows)
    charts = [
        _chart(
            "palette_size", "Palette-size distribution", _values(rows, "metrics", "pixel_art", "unique_palette_size")
        ),
        _chart("silhouette", "Silhouette occupancy", _values(rows, "metrics", "pixel_art", "silhouette_occupancy")),
        _chart("conditional", "Conditional adherence", _values(rows, "conditional_adherence")),
    ]
    summary = _mapping(report.get("summary"))
    payload = {
        "schema_version": "spritelab.product.evaluation-dashboard.v1",
        "metric_cards": [card.to_dict() for card in _metric_cards(report)],
        "charts": [chart.to_dict() for chart in charts],
        "per_category": _aggregate(rows, "category"),
        "per_source": _aggregate(rows, "source_id") if allow_source_results else [],
        "source_results_allowed": allow_source_results,
        "gallery": sample_gallery(rows),
        "review_queue": {
            "required": int(_nested(summary, "memorization", "review_required_count") or 0),
            "complete": int(_nested(summary, "memorization", "review_required_count") or 0) == 0,
        },
        "report_data_download": "/evaluation/api/report-data",
    }
    payload = finite_json_copy(payload)
    validate_finite_json(payload)
    return payload


def _definition_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    explicit = report.get("metric_definitions")
    if explicit is not None:
        return {"metric_definitions": explicit}
    return {
        "schema_version": report.get("schema_version"),
        "thresholds": report.get("thresholds"),
        "detector_policy_version": report.get("detector_policy_version"),
        "comparison_method": report.get("comparison_method"),
        "comparison_parameters_sha256": report.get("comparison_parameters_sha256"),
    }


def metric_definition_identity(report: Mapping[str, Any]) -> str:
    payload = _definition_payload(report)
    encoded = strict_json_bytes(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded).hexdigest()


def _numeric_summary(summary: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in summary.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_numeric_summary(value, path))
        elif (number := _number(value)) is not None:
            result[path] = float(number)
    return result


def compare_evaluations(
    left_report: Mapping[str, Any],
    right_report: Mapping[str, Any],
    left_rows: Sequence[Mapping[str, Any]] = (),
    right_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compare reports only when metric definitions are exactly compatible."""

    left_identity = metric_definition_identity(left_report)
    right_identity = metric_definition_identity(right_report)
    if left_identity != right_identity:
        raise IncompatibleMetricDefinitions(
            "Evaluation reports use incompatible metric definitions; no averages or deltas were computed."
        )
    left_metrics = _numeric_summary(_mapping(left_report.get("summary")))
    right_metrics = _numeric_summary(_mapping(right_report.get("summary")))
    common = sorted(set(left_metrics) & set(right_metrics))
    left_pairs = {(row.get("prompt_id"), row.get("seed"), row.get("category")): row for row in left_rows}
    right_pairs = {(row.get("prompt_id"), row.get("seed"), row.get("category")): row for row in right_rows}
    paired_keys = sorted(set(left_pairs) & set(right_pairs), key=lambda value: tuple(str(item) for item in value))
    return {
        "schema_version": "spritelab.product.evaluation-comparison.v1",
        "compatible": True,
        "metric_definition_identity": left_identity,
        "metrics": [
            {
                "metric": name,
                "left": left_metrics[name],
                "right": right_metrics[name],
                "change": right_metrics[name] - left_metrics[name],
            }
            for name in common
        ],
        "sample_pairs": [
            {
                "prompt_id": key[0],
                "seed": key[1],
                "category": key[2],
                "left": sample_gallery([left_pairs[key]])[0],
                "right": sample_gallery([right_pairs[key]])[0],
            }
            for key in paired_keys
        ],
        "category_changes": _category_changes(left_rows, right_rows),
        "diversity_changes": {
            name.removeprefix("diversity."): right_metrics[name] - left_metrics[name]
            for name in common
            if name.startswith("diversity.")
        },
        "memorization": {
            "left": _nested(_mapping(left_report.get("summary")), "memorization"),
            "right": _nested(_mapping(right_report.get("summary")), "memorization"),
        },
        "run_metadata": {"left": left_report.get("run_metadata", {}), "right": right_report.get("run_metadata", {})},
    }


def _category_changes(
    left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    left = {row["name"]: row for row in _aggregate(left_rows, "category")}
    right = {row["name"]: row for row in _aggregate(right_rows, "category")}
    result: list[dict[str, Any]] = []
    for name in sorted(set(left) | set(right)):
        left_value = left.get(name, {}).get("conditional_adherence")
        right_value = right.get(name, {}).get("conditional_adherence")
        result.append(
            {
                "category": name,
                "left": left_value,
                "right": right_value,
                "change": right_value - left_value
                if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float))
                else None,
            }
        )
    return result
