"""Dashboard projections over existing evaluation report data."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.evaluation.metric_definitions import IncompatibleMetricDefinitions, metric_definition_identity
from spritelab.product_core import finite_json_copy, validate_finite_json
from spritelab.product_web.events import sanitize_public_text


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


_OMIT = object()
_TEXT = "text"
_OPTIONAL_TEXT = "optional_text"
_NUMBER = "number"
_OPTIONAL_NUMBER = "optional_number"
_BOOLEAN = "boolean"
_SCALAR = "scalar"
_BASENAME = "basename"


def _same_spec(names: Sequence[str], spec: Any) -> dict[str, Any]:
    return dict.fromkeys(names, spec)


_DISTRIBUTION_SPEC = _same_spec(("min", "p10", "median", "mean", "p90", "max"), _OPTIONAL_NUMBER)
_HARD_VALIDITY_METRICS_SPEC = {
    **_same_spec(
        (
            "pass",
            "is_png",
            "exact_dimensions",
            "alpha_range_valid",
            "fully_transparent",
            "fully_opaque_rectangle",
            "corrupt_output",
            "model_output_finite",
            "unexpected_interpolation",
            "wrong_export_scaling",
            "missing_generation_metadata",
        ),
        _BOOLEAN,
    ),
    "missing_metadata_fields": (_TEXT,),
}
_PIXEL_ART_METRICS_SPEC = {
    **_same_spec(
        (
            "unique_palette_size",
            "palette_concentration",
            "semi_transparent_pixel_ratio",
            "antialiased_edge_ratio",
            "local_color_transition_sharpness",
            "isolated_pixel_noise",
            "small_component_count",
            "silhouette_occupancy",
            "border_pixel_ratio",
            "empty_padding",
            "foreground_fragmentation",
            "connected_component_count",
            "disconnected_shadow_components",
            "palette_adherence",
            "alpha_mask_compactness",
            "high_frequency_pixel_noise",
            "bbox_area",
        ),
        _OPTIONAL_NUMBER,
    ),
    "border_clipping": _BOOLEAN,
}
_ROW_METRICS_SPEC = {
    "hard_validity": _HARD_VALIDITY_METRICS_SPEC,
    "pixel_art": _PIXEL_ART_METRICS_SPEC,
}
_PUBLIC_ROW_SPEC = {
    **_same_spec(
        (
            "sample_id",
            "generated_sample_id",
            "artifact_id",
            "image_reference",
            "prompt_id",
            "prompt",
            "weights",
            "category",
            "source_id",
            "memorization_evidence_class",
            "evidence_strength",
            "low_evidence_reason",
            "suspicious_memorization",
            "label_quality",
            "split",
            "propagation_relation",
        ),
        _TEXT,
    ),
    **_same_spec(("seed", "noise_seed"), _OPTIONAL_NUMBER),
    **_same_spec(
        (
            "requires_human_review",
            "machine_hard_block_candidate",
            "warning_only",
            "open_set",
            "unseen_pack",
            "memorization_indicator",
            "generation_failed",
        ),
        _BOOLEAN,
    ),
    "checkpoint": _BASENAME,
    "checkpoint_identity": _BASENAME,
    "metrics": _ROW_METRICS_SPEC,
    "conditional_adherence": _OPTIONAL_NUMBER,
    "nearest_match_summary": {
        "evidence_class": _OPTIONAL_TEXT,
        "evidence_strength": _OPTIONAL_TEXT,
    },
}
_SUMMARY_SPEC = {
    "sample_count": _OPTIONAL_NUMBER,
    "hard_validity": {
        "malformed_count": _OPTIONAL_NUMBER,
        "pass_rate": _OPTIONAL_NUMBER,
    },
    "pixel_art": _same_spec(
        (
            "palette_size_mean",
            "palette_concentration_mean",
            "semi_transparent_ratio_mean",
            "antialiased_edge_ratio_mean",
            "silhouette_occupancy_mean",
            "border_clipping_rate",
            "fragmentation_mean",
            "high_frequency_noise_mean",
            "palette_adherence_mean",
        ),
        _OPTIONAL_NUMBER,
    ),
    "diversity": {
        **_same_spec(
            (
                "sample_count",
                "exact_duplicate_rate",
                "alpha_mask_duplicate_rate",
                "perceptual_near_duplicate_rate",
                "geometry_recolor_duplicate_rate",
                "unique_silhouettes",
                "palette_diversity",
                "repeated_template_rate",
                "palette_consistency_mean_jaccard",
            ),
            _OPTIONAL_NUMBER,
        ),
        "pairwise_distance": _DISTRIBUTION_SPEC,
        "seed_sensitivity": {"scorable_groups": _OPTIONAL_NUMBER, **_DISTRIBUTION_SPEC},
        "prompt_sensitivity": {"scorable_groups": _OPTIONAL_NUMBER, **_DISTRIBUTION_SPEC},
    },
    "memorization": {
        **_same_spec(
            (
                "detector_policy_version",
                "comparison_method",
                "comparison_parameters_sha256",
                "detector_policy_sha256",
                "evidence_contract_state",
            ),
            _OPTIONAL_TEXT,
        ),
        **_same_spec(
            (
                "hard_evidence_count",
                "review_required_count",
                "low_evidence_collision_count",
                "warning_count",
                "unresolved_candidate_count",
                "suspicious_count",
                "suspicious_rate",
                "exact_rgba_count",
                "candidate_count",
            ),
            _OPTIONAL_NUMBER,
        ),
        "evidence_contract_reasons": (_TEXT,),
    },
    "conditional": {
        "represented_rate": _OPTIONAL_NUMBER,
        "scorable_decisions": _OPTIONAL_NUMBER,
    },
}
_PROMOTION_REPORT_SPEC = {
    **_same_spec(
        (
            "policy_version",
            "detector_policy_version",
            "comparison_method",
            "comparison_parameters_sha256",
            "detector_policy_sha256",
            "memorization_machine_status",
        ),
        _OPTIONAL_TEXT,
    ),
    **_same_spec(
        (
            "pass",
            "manual_review_required",
        ),
        _BOOLEAN,
    ),
    **_same_spec(
        (
            "hard_evidence_count",
            "review_required_count",
            "low_evidence_collision_count",
            "unresolved_candidate_count",
        ),
        _OPTIONAL_NUMBER,
    ),
    "checks": _same_spec(
        (
            "detector_policy_supported",
            "memorization_evidence_complete",
            "malformed",
            "memorization_hard_evidence",
            "memorization_reviews_resolved",
            "exact_train_duplicates",
            "near_train_duplicates",
            "alpha_quality",
            "palette",
            "exact_duplicates",
            "template_collapse",
            "conditional_not_worse",
        ),
        _BOOLEAN,
    ),
    "memorization_outcome_reasons": (_TEXT,),
    "memorization_warnings": (_TEXT,),
}
_REPORT_SPEC = {
    **_same_spec(
        (
            "schema_version",
            "detector_policy_version",
            "comparison_method",
            "comparison_parameters_sha256",
            "detector_policy_sha256",
        ),
        _OPTIONAL_TEXT,
    ),
    "metric_definitions_sha256": _OPTIONAL_TEXT,
    "thresholds": {
        **_same_spec(
            (
                "perceptual_near_duplicate",
                "geometry_duplicate_iou",
                "near_train_pixel_distance",
                "suspicious_geometry_iou",
                "palette_adherence_rgb_tolerance",
            ),
            _SCALAR,
        )
    },
    "summary": _SUMMARY_SPEC,
    "promotion": _PROMOTION_REPORT_SPEC,
}
_PROMOTION_DISPLAY_SPEC = {
    **_same_spec(
        ("schema_version", "message", "audit_verdict", "audit_applicability", "audit_identity"), _OPTIONAL_TEXT
    ),
    **_same_spec(("integrity_certified", "promotion_authorized", "audit_code_identity_current"), _BOOLEAN),
    "audit_applicability_reasons": (_TEXT,),
    "actions": (),
    "promotion_actions_performed": _OPTIONAL_NUMBER,
}
_MEMORIZATION_ITEM_SPEC = {
    **_same_spec(
        (
            "pair_id",
            "evidence_class",
            "display_state",
            "current_review_state",
            "authoritative_event_sha256",
            "event_chain_status",
            "action_unavailable_reason",
        ),
        _OPTIONAL_TEXT,
    ),
    **_same_spec(
        (
            "review_authoritative",
            "event_chain_valid",
            "review_action_available",
            "clear_action_available",
        ),
        _BOOLEAN,
    ),
    "controlled_review_outcomes": (_TEXT,),
}
_MEMORIZATION_SPEC = {
    **_same_spec(
        (
            "schema_version",
            "evidence_state",
            "candidate_evidence_sha256",
            "review_message",
            "review_contract",
            "action_unavailable_reason",
        ),
        _OPTIONAL_TEXT,
    ),
    **_same_spec(("review_action_available", "writes_review_log"), _BOOLEAN),
    "review_required_count": _OPTIONAL_NUMBER,
    "items": (_MEMORIZATION_ITEM_SPEC,),
    "counts": _same_spec(
        (
            "Hard evidence",
            "Review required",
            "Warnings",
            "Cleared by valid review",
            "Invalid review",
            "Not comparable",
            "No material match",
        ),
        _OPTIONAL_NUMBER,
    ),
    "review_link": {
        **_same_spec(("feature", "action_id", "queue_id", "href", "label"), _OPTIONAL_TEXT),
    },
    "validation_reasons": (_TEXT,),
}
_STAGE_SPEC = {
    **_same_spec(("key", "title", "status", "message"), _OPTIONAL_TEXT),
    "current": _OPTIONAL_NUMBER,
    "total": _OPTIONAL_NUMBER,
    "metrics": {},
}
_METRIC_CARD_SPEC = {
    **_same_spec(("metric_id", "title", "unit", "status"), _OPTIONAL_TEXT),
    "value": _SCALAR,
}
_CHART_SPEC = {
    **_same_spec(("chart_id", "title", "kind", "status", "no_data_message"), _OPTIONAL_TEXT),
    "series": ({"label": _OPTIONAL_TEXT, "value": _OPTIONAL_NUMBER},),
}
_AGGREGATE_SPEC = {
    "name": _OPTIONAL_TEXT,
    "sample_count": _OPTIONAL_NUMBER,
    "structural_validity_rate": _OPTIONAL_NUMBER,
    "conditional_adherence": _OPTIONAL_NUMBER,
}
_DASHBOARD_SPEC = {
    "schema_version": _OPTIONAL_TEXT,
    "metric_cards": (_METRIC_CARD_SPEC,),
    "charts": (_CHART_SPEC,),
    "per_category": (_AGGREGATE_SPEC,),
    "per_source": (_AGGREGATE_SPEC,),
    "source_results_allowed": _BOOLEAN,
    "gallery": (_PUBLIC_ROW_SPEC,),
    "review_queue": {"required": _OPTIONAL_NUMBER, "complete": _BOOLEAN},
    "report_data_download": _OPTIONAL_TEXT,
    **_same_spec(("run_id", "status", "message"), _OPTIONAL_TEXT),
    "stale": _BOOLEAN,
    "stale_reasons": (_TEXT,),
    "memorization": _MEMORIZATION_SPEC,
}
_RUN_DATA_SPEC = {
    "schema_version": _OPTIONAL_TEXT,
    "stages": (_STAGE_SPEC,),
    "progress": {"completed": _OPTIONAL_NUMBER, "total": _OPTIONAL_NUMBER},
    "promotion": _PROMOTION_DISPLAY_SPEC,
    "generation_runs": _OPTIONAL_NUMBER,
    "promotion_actions": _OPTIONAL_NUMBER,
    "dry_run": _BOOLEAN,
    "dashboard": _DASHBOARD_SPEC,
    "memorization": _MEMORIZATION_SPEC,
    "report": _REPORT_SPEC,
}
_REPORT_DATA_SPEC = {
    "report": _REPORT_SPEC,
    "per_image_metrics": (_PUBLIC_ROW_SPEC,),
    "stages": (_STAGE_SPEC,),
    **_same_spec(("run_id", "status", "message"), _OPTIONAL_TEXT),
    "promotion_actions": _OPTIONAL_NUMBER,
}
_DURABLE_RUN_SPEC = {
    **_same_spec(("run_id", "status", "message"), _OPTIONAL_TEXT),
    "stages": (_STAGE_SPEC,),
    "dashboard": _DASHBOARD_SPEC,
}
_RESULT_SPEC = {
    **_same_spec(("schema_version", "status", "message", "feature"), _OPTIONAL_TEXT),
    "action": {
        **_same_spec(("action_id", "feature", "title"), _OPTIONAL_TEXT),
        "requires_confirmation": _BOOLEAN,
    },
    "run": {
        **_same_spec(
            ("run_id", "feature", "action_id", "status", "backend_id", "started_at", "ended_at"),
            _OPTIONAL_TEXT,
        ),
        "artifact_references": (_BASENAME,),
    },
    "blockers": ({**_same_spec(("code", "message", "resolution"), _OPTIONAL_TEXT)},),
    "warnings": ({**_same_spec(("code", "message", "resolution"), _OPTIONAL_TEXT)},),
    "data": _RUN_DATA_SPEC,
}
_COMPARISON_SPEC = {
    "schema_version": _OPTIONAL_TEXT,
    "compatible": _BOOLEAN,
    "metric_definition_identity": _OPTIONAL_TEXT,
    "metrics": (
        {
            "metric": _OPTIONAL_TEXT,
            "left": _OPTIONAL_NUMBER,
            "right": _OPTIONAL_NUMBER,
            "change": _OPTIONAL_NUMBER,
        },
    ),
    "sample_pairs": (
        {
            "prompt_id": _OPTIONAL_TEXT,
            "seed": _OPTIONAL_NUMBER,
            "category": _OPTIONAL_TEXT,
            "left": _PUBLIC_ROW_SPEC,
            "right": _PUBLIC_ROW_SPEC,
        },
    ),
    "category_changes": (
        {
            "category": _OPTIONAL_TEXT,
            "left": _OPTIONAL_NUMBER,
            "right": _OPTIONAL_NUMBER,
            "change": _OPTIONAL_NUMBER,
        },
    ),
    "diversity_changes": _same_spec(
        (
            "sample_count",
            "exact_duplicate_rate",
            "alpha_mask_duplicate_rate",
            "perceptual_near_duplicate_rate",
            "geometry_recolor_duplicate_rate",
            "unique_silhouettes",
            "palette_diversity",
            "repeated_template_rate",
            "palette_consistency_mean_jaccard",
        ),
        _OPTIONAL_NUMBER,
    ),
    "memorization": {"left": _SUMMARY_SPEC["memorization"], "right": _SUMMARY_SPEC["memorization"]},
}
_PUBLIC_SURFACE_SPECS = {
    "row": _PUBLIC_ROW_SPEC,
    "report": _REPORT_SPEC,
    "memorization": _MEMORIZATION_SPEC,
    "stage": _STAGE_SPEC,
    "dashboard": _DASHBOARD_SPEC,
    "run_data": _RUN_DATA_SPEC,
    "report_data": _REPORT_DATA_SPEC,
    "durable_run": _DURABLE_RUN_SPEC,
    "result": _RESULT_SPEC,
    "comparison": _COMPARISON_SPEC,
}


def _project_public_value(value: Any, spec: Any, private_roots: tuple[Path, ...]) -> Any:
    if spec == _TEXT or spec == _OPTIONAL_TEXT:
        if value is None and spec == _OPTIONAL_TEXT:
            return None
        return sanitize_public_text(value, private_roots) if isinstance(value, str) else _OMIT
    if spec == _NUMBER or spec == _OPTIONAL_NUMBER:
        if value is None and spec == _OPTIONAL_NUMBER:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _OMIT
        if isinstance(value, float) and not math.isfinite(value):
            return _OMIT
        return value
    if spec == _BOOLEAN:
        return value if type(value) is bool else _OMIT
    if spec == _SCALAR:
        if value is None or type(value) is bool:
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value if not isinstance(value, float) or math.isfinite(value) else _OMIT
        return sanitize_public_text(value, private_roots) if isinstance(value, str) else _OMIT
    if spec == _BASENAME:
        if not isinstance(value, str):
            return _OMIT
        public = sanitize_public_text(value, private_roots).replace("\\", "/").rstrip("/")
        return public.rsplit("/", 1)[-1] or "artifact"
    if isinstance(spec, tuple):
        if len(spec) != 1 or not isinstance(value, (list, tuple)):
            return []
        rows = []
        for child in value:
            projected = _project_public_value(child, spec[0], private_roots)
            if projected is not _OMIT:
                rows.append(projected)
        return rows
    if isinstance(spec, Mapping):
        if value is None:
            return None
        if not isinstance(value, Mapping):
            return _OMIT
        projected: dict[str, Any] = {}
        for key, child_spec in spec.items():
            if key not in value:
                continue
            child = _project_public_value(value[key], child_spec, private_roots)
            if child is not _OMIT:
                projected[key] = child
        if spec is _REPORT_SPEC and value.get("metric_definitions") is not None:
            try:
                identity = metric_definition_identity(value)
            except IncompatibleMetricDefinitions:
                projected.pop("metric_definitions_sha256", None)
            else:
                projected["metric_definitions_sha256"] = identity
                # The verified digest is the complete public definition
                # envelope.  Do not retain a second, partial policy envelope
                # that could later drift while reusing the digest.
                for field in (
                    "thresholds",
                    "detector_policy_version",
                    "detector_policy_sha256",
                    "comparison_method",
                    "comparison_parameters_sha256",
                ):
                    projected.pop(field, None)
        return projected
    return _OMIT


def public_evaluation_projection(
    value: Any,
    *,
    surface: str,
    private_roots: Sequence[Path] = (),
) -> Any:
    """Project one ordinary Evaluation surface through a closed recursive schema."""

    try:
        spec = _PUBLIC_SURFACE_SPECS[surface]
    except KeyError as exc:
        raise ValueError(f"Unsupported public Evaluation surface: {surface}") from exc
    projected = _project_public_value(value, spec, tuple(Path(root) for root in private_roots))
    if projected is _OMIT:
        return {} if isinstance(spec, Mapping) else []
    public = finite_json_copy(projected)
    validate_finite_json(public)
    return public


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
        raw_label = row.get(key)
        label = raw_label if isinstance(raw_label, str) and raw_label else "unknown"
        groups[label].append(row)
    result: list[dict[str, Any]] = []
    for label, group in sorted(groups.items()):
        adherence = [
            float(number) for row in group if (number := _number(row.get("conditional_adherence"))) is not None
        ]
        valid = [not failed for row in group if type(failed := row.get("generation_failed")) is bool]
        result.append(
            {
                "name": label,
                "sample_count": len(group),
                "structural_validity_rate": sum(valid) / len(valid) if valid else None,
                "conditional_adherence": sum(adherence) / len(adherence) if adherence else None,
            }
        )
    return result


def sample_gallery(
    rows: Sequence[Mapping[str, Any]],
    *,
    private_roots: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """Return public gallery records without source or training filesystem paths."""

    gallery: list[dict[str, Any]] = []
    for row in rows:
        public_row = public_evaluation_projection(row, surface="row", private_roots=private_roots)
        sample_id = str(public_row.get("sample_id") or public_row.get("generated_sample_id") or "sample")
        nearest = row.get("training_neighbors") if isinstance(row.get("training_neighbors"), list) else []
        raw_checkpoint = str(public_row.get("checkpoint_identity") or public_row.get("checkpoint") or "")
        candidate = public_evaluation_projection(
            {
                "sample_id": sample_id,
                "image_reference": str(public_row.get("artifact_id") or sample_id),
                "prompt": str(public_row.get("prompt") or ""),
                "seed": public_row.get("seed"),
                "checkpoint": raw_checkpoint,
                "weights": str(
                    public_row.get("weights")
                    or ("ema" if str(public_row.get("checkpoint", "")).endswith("_ema.pt") else "live")
                ),
                "category": str(public_row.get("category") or "unknown"),
                "metrics": public_row.get("metrics", {}),
                "conditional_adherence": public_row.get("conditional_adherence"),
                "memorization_evidence_class": public_row.get("memorization_evidence_class"),
                "nearest_match_summary": {
                    "evidence_class": nearest[0].get("evidence_class"),
                    "evidence_strength": nearest[0].get("evidence_strength"),
                }
                if nearest and isinstance(nearest[0], Mapping)
                else None,
            },
            surface="row",
            private_roots=private_roots,
        )
        gallery.append(candidate)
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
    private_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    rows = tuple(
        public_evaluation_projection(row, surface="row", private_roots=private_roots) for row in per_image_rows
    )
    public_report = public_evaluation_projection(report, surface="report", private_roots=private_roots)
    charts = [
        _chart(
            "palette_size", "Palette-size distribution", _values(rows, "metrics", "pixel_art", "unique_palette_size")
        ),
        _chart("silhouette", "Silhouette occupancy", _values(rows, "metrics", "pixel_art", "silhouette_occupancy")),
        _chart("conditional", "Conditional adherence", _values(rows, "conditional_adherence")),
    ]
    summary = _mapping(public_report.get("summary"))
    source_results_allowed = allow_source_results is True
    payload = {
        "schema_version": "spritelab.product.evaluation-dashboard.v1",
        "metric_cards": [card.to_dict() for card in _metric_cards(public_report)],
        "charts": [chart.to_dict() for chart in charts],
        "per_category": _aggregate(rows, "category"),
        "per_source": _aggregate(rows, "source_id") if source_results_allowed else [],
        "source_results_allowed": source_results_allowed,
        "gallery": sample_gallery(rows, private_roots=private_roots),
        "review_queue": {
            "required": int(_nested(summary, "memorization", "review_required_count") or 0),
            "complete": int(_nested(summary, "memorization", "review_required_count") or 0) == 0,
        },
        "report_data_download": "/evaluation/api/report-data",
    }
    return public_evaluation_projection(payload, surface="dashboard", private_roots=private_roots)


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
    public_left_report = public_evaluation_projection(left_report, surface="report")
    public_right_report = public_evaluation_projection(right_report, surface="report")
    public_left_rows = tuple(public_evaluation_projection(row, surface="row") for row in left_rows)
    public_right_rows = tuple(public_evaluation_projection(row, surface="row") for row in right_rows)
    left_metrics = _numeric_summary(_mapping(public_left_report.get("summary")))
    right_metrics = _numeric_summary(_mapping(public_right_report.get("summary")))
    common = sorted(set(left_metrics) & set(right_metrics))
    left_pairs = {(row.get("prompt_id"), row.get("seed"), row.get("category")): row for row in public_left_rows}
    right_pairs = {(row.get("prompt_id"), row.get("seed"), row.get("category")): row for row in public_right_rows}
    paired_keys = sorted(set(left_pairs) & set(right_pairs), key=lambda value: tuple(str(item) for item in value))
    payload = {
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
        "category_changes": _category_changes(public_left_rows, public_right_rows),
        "diversity_changes": {
            name.removeprefix("diversity."): right_metrics[name] - left_metrics[name]
            for name in common
            if name.startswith("diversity.")
        },
        "memorization": {
            "left": _nested(_mapping(public_left_report.get("summary")), "memorization"),
            "right": _nested(_mapping(public_right_report.get("summary")), "memorization"),
        },
    }
    return public_evaluation_projection(payload, surface="comparison")


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
