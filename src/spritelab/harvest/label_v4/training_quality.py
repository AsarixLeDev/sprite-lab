"""Risk-aware Labeling-v4 training metadata and report helpers.

This module is deliberately independent from provider/model code.  It turns
calibrated *error risk* into versioned masks and monotonically decreasing loss
weights; it never treats model self-confidence as calibration.  Legacy
training rows without Labeling-v4 quality remain readable and are reported as
``not_scorable`` rather than being assigned invented precision.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from statistics import mean, median
from typing import Any

from spritelab.harvest.label_v4.risk import (
    CRITICAL_FIELDS,
    DEFAULT_BANDS,
    RISK_MODEL_VERSION,
    SEMANTIC_FIELDS,
    RiskBandThresholds,
    risk_score_from_upper,
)

TRAINING_QUALITY_SCHEMA_VERSION = "label_training_quality_v1"

FIELD_VALUE_STATES: tuple[str, ...] = (
    "missing",
    "known",
    "unknown",
    "abstained",
    "out_of_vocabulary",
)
FIELD_VALUE_STATE_IDS: dict[str, int] = {value: index for index, value in enumerate(FIELD_VALUE_STATES)}

TRAINING_QUALITY_FIELDS: tuple[str, ...] = SEMANTIC_FIELDS

RISK_BANDS: tuple[tuple[int, int, str], ...] = (
    (1, 4, "strong"),
    (5, 8, "usable_weak"),
    (9, 12, "auxiliary_only"),
    (13, 16, "excluded_from_primary_supervision"),
    (17, 20, "abstain_or_quarantine"),
)

# Ordered best to worst.  A dataset receives the first grade whose documented
# minimum coverage / maximum risk requirements all pass.
DEFAULT_GRADE_RULES: tuple[tuple[str, dict[str, float]], ...] = (
    (
        "A",
        {
            "min_strong_label_coverage": 0.80,
            "min_calibrated_coverage": 0.75,
            "max_abstention_rate": 0.05,
            "max_critical_field_risk_upper": 0.05,
        },
    ),
    (
        "B",
        {
            "min_strong_label_coverage": 0.60,
            "min_calibrated_coverage": 0.50,
            "max_abstention_rate": 0.15,
            "max_critical_field_risk_upper": 0.10,
        },
    ),
    (
        "C",
        {
            "min_strong_label_coverage": 0.40,
            "min_calibrated_coverage": 0.30,
            "max_abstention_rate": 0.30,
            "max_critical_field_risk_upper": 0.20,
        },
    ),
    (
        "D",
        {
            "min_strong_label_coverage": 0.20,
            "min_calibrated_coverage": 0.10,
            "max_abstention_rate": 0.50,
            "max_critical_field_risk_upper": 0.35,
        },
    ),
)


def uncertainty_band(score: int, *, thresholds: RiskBandThresholds = DEFAULT_BANDS) -> str:
    """Return the configurable-default band for an uncertainty score."""

    return thresholds.band(score)


def uncertainty_from_risk_upper(risk_upper_95: float) -> int:
    """Map an error-risk upper bound to 1..20 without implying score 1 is zero risk."""

    return risk_score_from_upper(risk_upper_95)


def loss_weight_for_risk(risk_upper_95: float, *, min_weight: float = 0.05) -> float:
    """Monotonic ``1-risk`` weight, clipped to a documented non-zero floor."""

    floor = max(0.0, min(1.0, float(min_weight)))
    risk = max(0.0, min(1.0, float(risk_upper_95)))
    return min(1.0, max(floor, 1.0 - risk))


def effective_sample_size(weights: Iterable[float]) -> float:
    """Kish effective sample size: ``(sum(w)**2) / sum(w**2)``."""

    values = [max(0.0, float(value)) for value in weights]
    denominator = sum(value * value for value in values)
    return (sum(values) ** 2 / denominator) if denominator else 0.0


def extract_training_quality(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return normalized v4 quality, or ``None`` for an unscored legacy row."""

    direct = record.get("label_quality")
    if _is_normalized_training_quality(direct):
        return dict(direct)
    if _looks_like_training_quality(direct):
        return normalize_training_quality(direct, record=record)
    direct = record.get("training_quality")
    if _is_normalized_training_quality(direct):
        return dict(direct)
    if _looks_like_training_quality(direct):
        return normalize_training_quality(direct, record=record)
    label_v4 = record.get("label_v4")
    if not isinstance(label_v4, Mapping):
        return None
    nested = label_v4.get("label_quality") or label_v4.get("training_quality") or label_v4.get("quality")
    if _is_normalized_training_quality(nested):
        return dict(nested)
    if isinstance(nested, Mapping):
        return normalize_training_quality(nested, record=record)
    if any(key in label_v4 for key in ("fields", "field_quality", "record_uncertainty_1_20")):
        return normalize_training_quality(label_v4, record=record)
    return None


def normalize_training_quality(
    quality: Mapping[str, Any],
    *,
    record: Mapping[str, Any] | None = None,
    min_weight: float = 0.05,
) -> dict[str, Any]:
    """Normalize provider-independent quality into the training contract.

    Missing calibrated risk in an explicitly v4 record is conservative: the
    field becomes ``not_scorable`` or receives uncertainty 18 when it is marked
    uncalibrated.  Caller-provided loss weights never override the monotonic
    risk-derived weight.
    """

    record = record or {}
    raw_fields = quality.get("fields") or quality.get("field_quality") or quality.get("field_uncertainties") or {}
    raw_fields = raw_fields if isinstance(raw_fields, Mapping) else {}
    field_names = list(TRAINING_QUALITY_FIELDS)
    field_names.extend(sorted(str(name) for name in raw_fields if str(name) not in field_names))
    normalized_fields: dict[str, dict[str, Any]] = {}
    for field_name in field_names:
        raw = raw_fields.get(field_name)
        raw = raw if isinstance(raw, Mapping) else {}
        present, value = _field_value(record, field_name)
        value_state = _value_state(raw, present=present, value=value)
        calibration_state = str(raw.get("calibration_state") or quality.get("calibration_state") or "").strip()
        score_raw = raw.get("uncertainty_1_20")
        risk_raw = raw.get("risk_upper_95")
        if risk_raw is not None:
            risk_upper = max(0.0, min(1.0, float(risk_raw)))
            score = uncertainty_from_risk_upper(risk_upper) if score_raw is None else _score(score_raw)
            score_state = "scored"
        elif score_raw is not None:
            score = _score(score_raw)
            risk_upper = score / 20.0
            score_state = "scored"
        elif calibration_state in {"uncalibrated", "borrowed_calibration"} or raw.get("uncalibrated") is True:
            score = 18
            risk_upper = 0.90
            score_state = "scored_conservative"
        else:
            score = None
            risk_upper = None
            score_state = "not_scorable"
        if raw.get("conflicts") and score is not None:
            score = max(9, score)
            risk_upper = max(0.45, float(risk_upper or 0.0))
        normalized_fields[field_name] = _field_policy(
            raw,
            value_state=value_state,
            score=score,
            risk_upper=risk_upper,
            score_state=score_state,
            calibration_state=calibration_state or "not_scorable",
            min_weight=min_weight,
        )

    unresolved = list(quality.get("unresolved_conflicts") or record.get("unresolved_conflicts") or ())
    if unresolved:
        named = {name for name in normalized_fields if any(name in str(conflict).lower() for conflict in unresolved)}
        for name in named or set(CRITICAL_FIELDS):
            field = normalized_fields[name]
            if field["uncertainty_1_20"] is None:
                continue
            normalized_fields[name] = _field_policy(
                field,
                value_state=str(field["value_state"]),
                score=max(9, int(field["uncertainty_1_20"])),
                risk_upper=max(0.45, float(field["risk_upper_95"] or 0.0)),
                score_state=str(field["uncertainty_state"]),
                calibration_state=str(field["calibration_state"]),
                min_weight=min_weight,
            )
    critical_scored = [
        normalized_fields[name]
        for name in CRITICAL_FIELDS
        if normalized_fields[name]["uncertainty_state"] != "not_scorable"
    ]
    supplied_score = quality.get("record_uncertainty_1_20")
    if supplied_score is not None:
        record_score: int | None = _score(supplied_score)
    elif critical_scored:
        record_score = max(int(item["uncertainty_1_20"]) for item in critical_scored)
    else:
        record_score = None
    if unresolved and record_score is not None:
        record_score = max(9, record_score)
    record_risk = (
        max(float(item["risk_upper_95"]) for item in critical_scored)
        if critical_scored
        else (record_score / 20.0 if record_score is not None else None)
    )
    if record_risk is not None and record_score is not None:
        record_risk = max(record_risk, record_score / 20.0)
    record_weight = 1.0 if record_risk is None else loss_weight_for_risk(record_risk, min_weight=min_weight)
    return {
        "schema_version": TRAINING_QUALITY_SCHEMA_VERSION,
        "risk_model_version": str(quality.get("risk_model_version") or RISK_MODEL_VERSION),
        "record_uncertainty_1_20": record_score,
        "record_uncertainty_state": "scored" if record_score is not None else "not_scorable",
        "record_uncertainty_band": uncertainty_band(record_score) if record_score is not None else "not_scorable",
        "critical_field_max_uncertainty": max(
            (int(item["uncertainty_1_20"]) for item in critical_scored), default=None
        ),
        "record_risk_upper_95": record_risk,
        "record_loss_weight": record_weight,
        "unresolved_conflict_count": len(unresolved),
        "fields": normalized_fields,
    }


def quality_tensor_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return fixed-order numeric arrays for a training loader.

    Legacy rows keep record loss weight 1.0 for behavioral compatibility, but
    their v4 field masks are all zero and ``quality_present`` is false.  This
    prevents reports from misrepresenting legacy compatibility as calibration.
    """

    quality = extract_training_quality(record)
    if quality is None:
        count = len(TRAINING_QUALITY_FIELDS)
        return {
            "quality_present": False,
            "record_loss_weight": 1.0,
            "field_uncertainty": [0] * count,
            "field_loss_weight": [0.0] * count,
            "field_supervision_mask": [0.0] * count,
            "field_auxiliary_mask": [0.0] * count,
            "field_conditioning_mask": [0.0] * count,
            "field_negative_target_mask": [0.0] * count,
            "field_state_ids": [FIELD_VALUE_STATE_IDS["missing"]] * count,
        }
    fields = quality["fields"]
    selected = [fields[name] for name in TRAINING_QUALITY_FIELDS]
    return {
        "quality_present": True,
        "record_loss_weight": float(quality["record_loss_weight"]),
        "field_uncertainty": [int(item.get("uncertainty_1_20") or 0) for item in selected],
        "field_loss_weight": [float(item["loss_weight"]) for item in selected],
        "field_supervision_mask": [float(item["supervision_mask"]) for item in selected],
        "field_auxiliary_mask": [float(item["auxiliary_mask"]) for item in selected],
        "field_conditioning_mask": [float(item["conditioning_mask"]) for item in selected],
        "field_negative_target_mask": [float(item["negative_target_mask"]) for item in selected],
        "field_state_ids": [FIELD_VALUE_STATE_IDS[str(item["value_state"])] for item in selected],
    }


def apply_conditioning_policy(record: Mapping[str, Any], caption: str) -> tuple[dict[str, Any], str]:
    """Remove excluded/abstained field values from model conditioning.

    A field mask must affect the actual model input, not merely appear in a
    report.  The transformation works on a copy and never mutates the manifest
    row.  Legacy rows have no v4 contract and retain their historical behavior.
    """

    quality = extract_training_quality(record)
    if quality is None:
        return dict(record), str(caption)
    blocked = {
        name
        for name, field in quality["fields"].items()
        if not bool(field.get("conditioning_mask")) and field.get("value_state") != "missing"
    }
    if not blocked:
        return dict(record), str(caption)
    result = deepcopy(dict(record))
    phrases: list[str] = []
    for field_name in blocked:
        _present, value = _field_value(record, field_name)
        phrases.extend(_text_values(value))
    _remove_blocked_semantics(result, blocked)
    if "canonical_object" in blocked:
        filtered_caption = ""
    else:
        filtered_caption = _remove_caption_phrases(str(caption), phrases)
    return result, filtered_caption


def dataset_uncertainty_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Dataset-wide uncertainty distributions and documented quality score."""

    rows = list(records)
    base = _base_report(rows)
    score = dataset_quality_score(rows, base_report=base)
    return {"schema_version": "dataset_label_quality_report_v1", **base, "dataset_quality_score": score}


def training_uncertainty_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Training contribution summary including masks, weighted mass, and ESS."""

    rows = list(records)
    base = _base_report(rows)
    qualities = [extract_training_quality(row) for row in rows]
    weights = [float(item["record_loss_weight"]) if item is not None else 1.0 for item in qualities]
    band_count: Counter[str] = Counter()
    band_weight: defaultdict[str, float] = defaultdict(float)
    masked_by_field: Counter[str] = Counter()
    weak_by_field: defaultdict[str, float] = defaultdict(float)
    for quality, weight in zip(qualities, weights, strict=True):
        band = "not_scorable" if quality is None else str(quality["record_uncertainty_band"])
        band_count[band] += 1
        band_weight[band] += weight
        if quality is None:
            continue
        for name, field in quality["fields"].items():
            if not field["supervision_mask"]:
                masked_by_field[name] += 1
            if field["auxiliary_mask"] and not field["supervision_mask"]:
                weak_by_field[name] += float(field["loss_weight"])
    return {
        "schema_version": "training_label_quality_report_v1",
        **base,
        "training_contribution": {
            "raw_records": len(rows),
            "effective_weighted_records": sum(weights),
            "effective_sample_size": effective_sample_size(weights),
            "sample_count_by_uncertainty_band": dict(sorted(band_count.items())),
            "loss_contribution_by_uncertainty_band": {
                key: round(value, 9) for key, value in sorted(band_weight.items())
            },
            "masked_labels_by_field": dict(sorted(masked_by_field.items())),
            "weak_label_contribution_by_field": {key: round(value, 9) for key, value in sorted(weak_by_field.items())},
        },
    }


def evaluation_uncertainty_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluation support counts for the required quality/OOD/propagation strata."""

    rows = list(records)
    pairs = [(row, extract_training_quality(row)) for row in rows]

    def count(predicate: Any) -> int:
        return sum(bool(predicate(row, quality)) for row, quality in pairs)

    strata = {
        "strong_labels": count(lambda _row, q: q is not None and q["record_uncertainty_band"] == "strong"),
        "usable_weak_labels": count(lambda _row, q: q is not None and q["record_uncertainty_band"] == "usable_weak"),
        "all_labels": len(rows),
        "source_ood": count(lambda row, _q: str(row.get("split") or "").lower() in {"source_ood", "source_ood_test"}),
        "open_set": count(lambda row, _q: bool(row.get("open_set")) or str(row.get("split") or "") == "open_set_test"),
        "unseen_pack": count(lambda row, _q: bool(row.get("unseen_pack"))),
        "propagated_labels": count(lambda row, _q: bool(_dimension(row, "propagation_relation"))),
        "non_propagated_labels": count(lambda row, _q: not bool(_dimension(row, "propagation_relation"))),
    }
    return {"schema_version": "evaluation_label_quality_report_v1", **_base_report(rows), "strata": strata}


def uncertainty_correlation_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Report non-causal Pearson relationships when paired observations exist."""

    rows = list(records)
    targets = {
        "training_loss": (("training_loss",), ("metrics", "training_loss"), ("loss",)),
        "validation_loss": (("validation_loss",), ("metrics", "validation_loss"), ("val_loss",)),
        "conditional_adherence": (
            ("conditional_adherence",),
            ("conditional", "represented_rate"),
            ("metrics", "conditional_adherence"),
        ),
        "memorization_indicator": (
            ("memorization_indicator",),
            ("suspicious_memorization",),
            ("metrics", "memorization_indicator"),
        ),
        "generation_failure_rate": (
            ("generation_failure_rate",),
            ("generation_failed",),
            ("metrics", "generation_failure_rate"),
        ),
    }
    relationships: dict[str, Any] = {}
    for name, paths in targets.items():
        pairs: list[tuple[float, float]] = []
        for row in rows:
            quality = extract_training_quality(row)
            if quality is None or quality.get("record_uncertainty_1_20") is None:
                continue
            target = _first_numeric(row, paths)
            if target is not None:
                pairs.append((float(quality["record_uncertainty_1_20"]), target))
        relationships[name] = {
            "pearson_r": _pearson(pairs),
            "paired_support_n": len(pairs),
            "available": len(pairs) >= 2,
        }
    return {
        "schema_version": "label_uncertainty_correlation_v1",
        "relationships": relationships,
        "interpretation": "descriptive_correlation_only",
        "causal_claim": False,
        "warning": "Correlation is non-causal; null values mean paired observations were unavailable.",
    }


def dataset_quality_score(
    records: Iterable[Mapping[str, Any]], *, base_report: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Produce transparent dataset scores plus the rule-by-rule grade breakdown."""

    rows = list(records)
    base = dict(base_report or _base_report(rows))
    qualities = [extract_training_quality(row) for row in rows]
    scored = [item for item in qualities if item is not None and item["record_uncertainty_1_20"] is not None]
    total = len(rows)
    strong = sum(item["record_uncertainty_band"] == "strong" for item in scored)
    weak = sum(item["record_uncertainty_band"] == "usable_weak" for item in scored)
    abstained = sum(item["record_uncertainty_band"] == "abstain_or_quarantine" for item in scored)
    calibrated = sum(_quality_calibrated(item) for item in scored)
    weights = [float(item["record_loss_weight"]) if item is not None else 0.0 for item in qualities]
    critical_risks = [
        float(field["risk_upper_95"])
        for item in scored
        for name, field in item["fields"].items()
        if name in CRITICAL_FIELDS and field["risk_upper_95"] is not None
    ]
    metrics = {
        "semantic_coverage": len(scored) / total if total else 0.0,
        "strong_label_coverage": strong / total if total else 0.0,
        "weak_label_coverage": weak / total if total else 0.0,
        "abstention_rate": abstained / total if total else 0.0,
        "calibrated_coverage": calibrated / total if total else 0.0,
        "effective_supervised_records": sum(weights),
        "effective_sample_size": effective_sample_size(weights),
        "critical_field_risk_upper": max(critical_risks, default=1.0),
    }
    grade = quality_grade_breakdown(metrics)
    lowering = _grade_lowering_dimensions(base)
    return {
        **metrics,
        "quality_grade": grade["quality_grade"],
        "grade_breakdown": grade,
        "grade_lowering_dimensions": lowering,
    }


def quality_grade_breakdown(
    metrics: Mapping[str, Any],
    *,
    rules: Sequence[tuple[str, Mapping[str, float]]] = DEFAULT_GRADE_RULES,
) -> dict[str, Any]:
    """Return a documented, configurable, non-opaque quality grade decision."""

    evaluated: list[dict[str, Any]] = []
    selected = "F"
    for grade, rule in rules:
        checks = {
            "strong_label_coverage": float(metrics.get("strong_label_coverage") or 0.0)
            >= float(rule["min_strong_label_coverage"]),
            "calibrated_coverage": float(metrics.get("calibrated_coverage") or 0.0)
            >= float(rule["min_calibrated_coverage"]),
            "abstention_rate": float(metrics.get("abstention_rate") or 0.0) <= float(rule["max_abstention_rate"]),
            "critical_field_risk_upper": float(metrics.get("critical_field_risk_upper") or 1.0)
            <= float(rule["max_critical_field_risk_upper"]),
        }
        evaluated.append({"grade": grade, "rule": dict(rule), "checks": checks, "passed": all(checks.values())})
        if all(checks.values()):
            selected = grade
            break
    return {
        "quality_grade": selected,
        "rules_version": "dataset_quality_grade_v1",
        "evaluated_rules": evaluated,
        "explanation": "Grade uses published coverage, abstention, calibration, and critical-risk thresholds.",
    }


def _field_policy(
    raw: Mapping[str, Any],
    *,
    value_state: str,
    score: int | None,
    risk_upper: float | None,
    score_state: str,
    calibration_state: str,
    min_weight: float,
) -> dict[str, Any]:
    band = uncertainty_band(score) if score is not None else "not_scorable"
    usable_value = value_state == "known"
    primary = usable_value and score is not None and score <= 8
    auxiliary = usable_value and score is not None and score <= 12
    conditioning = usable_value and score is not None and score <= 12
    weight = 0.0 if risk_upper is None else loss_weight_for_risk(risk_upper, min_weight=min_weight)
    if not auxiliary:
        weight = 0.0
    return {
        **dict(raw),
        "value_state": value_state,
        "uncertainty_state": score_state,
        "uncertainty_1_20": score,
        "uncertainty_band": band,
        "risk_upper_95": risk_upper,
        "calibration_state": calibration_state,
        "loss_weight": weight,
        "training_state": band,
        "supervision_mask": int(primary),
        "auxiliary_mask": int(auxiliary),
        "conditioning_mask": int(conditioning),
        # Only a supported known primary target may create class negatives.
        # Unknown, missing, abstained, OOV, and auxiliary-only values never do.
        "negative_target_mask": int(usable_value and score is not None and score <= 4),
        "unknown_is_negative": False,
    }


def _base_report(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    record_hist = {str(score): 0 for score in range(1, 21)}
    field_hists: dict[str, dict[str, int]] = {}
    scores: list[int] = []
    field_scores: defaultdict[str, list[int]] = defaultdict(list)
    record_calibration: Counter[str] = Counter()
    field_calibration: Counter[str] = Counter()
    dimensions: dict[str, defaultdict[str, list[int]]] = {
        name: defaultdict(list)
        for name in ("source", "pack", "artist", "category", "split", "inference_path", "propagation_relation")
    }
    legacy = 0
    for row in rows:
        quality = extract_training_quality(row)
        if quality is None:
            legacy += 1
            record_calibration["not_scorable"] += 1
            continue
        score = quality.get("record_uncertainty_1_20")
        if score is not None:
            value = int(score)
            scores.append(value)
            record_hist[str(value)] += 1
            for dimension in dimensions:
                dimensions[dimension][_dimension(row, dimension) or "unknown"].append(value)
        for name, field in quality["fields"].items():
            field_score = field.get("uncertainty_1_20")
            if field_score is not None:
                field_scores[name].append(int(field_score))
            field_calibration[str(field.get("calibration_state") or "not_scorable")] += 1
        critical_states = {
            str(quality["fields"][name].get("calibration_state") or "not_scorable")
            for name in CRITICAL_FIELDS
            if name in quality["fields"]
        }
        if critical_states == {"calibrated"}:
            record_calibration["calibrated"] += 1
        elif "uncalibrated" in critical_states:
            record_calibration["uncalibrated"] += 1
        elif "borrowed_calibration" in critical_states:
            record_calibration["borrowed_calibration"] += 1
        else:
            record_calibration["not_scorable"] += 1
    for name, values in field_scores.items():
        hist = {str(score): 0 for score in range(1, 21)}
        for value in values:
            hist[str(value)] += 1
        field_hists[name] = hist
    return {
        "record_count": len(rows),
        "scored_record_count": len(scores),
        "legacy_unscored_count": legacy,
        "record_uncertainty_histogram": record_hist,
        "field_uncertainty_histograms": field_hists,
        "record_uncertainty_summary": _score_summary(scores),
        "field_uncertainty_summaries": {name: _score_summary(values) for name, values in field_scores.items()},
        "uncertainty_by": {
            dimension: {key: {"count": len(values), **_score_summary(values)} for key, values in sorted(groups.items())}
            for dimension, groups in dimensions.items()
        },
        "calibration_state_counts": dict(sorted(record_calibration.items())),
        "field_calibration_state_counts": dict(sorted(field_calibration.items())),
    }


def _score_summary(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    index = max(0, math.ceil(0.90 * len(ordered)) - 1)
    return {"count": len(ordered), "mean": mean(ordered), "median": median(ordered), "p90": ordered[index]}


def _looks_like_training_quality(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        value.get("schema_version") == TRAINING_QUALITY_SCHEMA_VERSION
        or "record_uncertainty_1_20" in value
        or (
            isinstance(value.get("fields"), Mapping)
            and any("uncertainty_1_20" in item for item in value["fields"].values() if isinstance(item, Mapping))
        )
    )


def _is_normalized_training_quality(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("schema_version") == TRAINING_QUALITY_SCHEMA_VERSION
        and isinstance(value.get("fields"), Mapping)
        and all(name in value["fields"] for name in TRAINING_QUALITY_FIELDS)
        and "record_loss_weight" in value
        and all(
            isinstance(field, Mapping)
            and "conditioning_mask" in field
            and "supervision_mask" in field
            and "loss_weight" in field
            for field in value["fields"].values()
        )
    )


def _score(value: Any) -> int:
    return max(1, min(20, int(value)))


def _value_state(raw: Mapping[str, Any], *, present: bool, value: Any) -> str:
    requested = str(raw.get("value_state") or raw.get("field_state") or "").strip().lower()
    aliases = {"oov": "out_of_vocabulary", "unsupported": "abstained", "abstain": "abstained"}
    requested = aliases.get(requested, requested)
    if requested in FIELD_VALUE_STATE_IDS:
        return requested
    if raw.get("abstained") is True or str(raw.get("training_state") or "") == "abstain_or_quarantine":
        return "abstained"
    if raw.get("out_of_vocabulary") is True:
        return "out_of_vocabulary"
    if not present or value is None or value == "" or value == []:
        return "missing"
    if isinstance(value, str) and value.strip().lower() == "unknown":
        return "unknown"
    return "known"


def _field_value(record: Mapping[str, Any], field_name: str) -> tuple[bool, Any]:
    aliases: dict[str, tuple[str, ...]] = {
        "canonical_object": ("canonical_object", "object_name", "base_object"),
        "surface_alias": ("surface_alias", "open_name"),
        "description": ("description", "short_description", "caption"),
        "primary_colors": ("primary_colors", "colors", "color"),
        "shape": ("shape", "shapes"),
        "style": ("style", "styles"),
        "condition": ("condition", "state"),
    }
    containers: list[Mapping[str, Any]] = [record]
    for key in ("field_values", "semantic_fields", "final_label", "label_v4"):
        value = record.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    conditioning = record.get("conditioning")
    if isinstance(conditioning, Mapping):
        semantic = conditioning.get("semantic_v4") or conditioning.get("semantic_v3")
        if isinstance(semantic, Mapping):
            containers.append(semantic)
            if isinstance(semantic.get("attributes"), Mapping):
                containers.append(semantic["attributes"])
    for container in containers:
        for key in aliases.get(field_name, (field_name,)):
            if key in container:
                return True, container[key]
        for parent in ("shape", "color_roles", "colors"):
            nested = container.get(parent)
            if isinstance(nested, Mapping) and field_name in nested:
                return True, nested[field_name]
    return False, None


def _remove_blocked_semantics(record: dict[str, Any], blocked: set[str]) -> None:
    key_groups: dict[str, tuple[str, ...]] = {
        "canonical_object": ("canonical_object", "object_name", "base_object"),
        "surface_alias": ("surface_alias", "open_name"),
        "category": ("category",),
        "domain": ("domain",),
        "role": ("role", "function", "functions"),
        "explicit_material": ("explicit_material", "material", "materials"),
        "visual_material_cue": ("visual_material_cue", "material_visual_cues"),
        "style": ("style", "styles"),
        "description": ("description", "short_description"),
        "size_hint": ("size_hint",),
        "filename_color_hints": ("filename_color_hints", "filename_color_hint"),
    }
    shape_fields = {"silhouette", "aspect", "orientation", "structure", "edge_profile", "parts"}
    color_fields = {
        "palette_colors",
        "primary_colors",
        "secondary_colors",
        "outline_colors",
        "shadow_colors",
        "highlight_colors",
    }

    def clean(container: dict[str, Any]) -> None:
        for field_name in blocked:
            for key in key_groups.get(field_name, (field_name,)):
                container.pop(key, None)
        if blocked & shape_fields:
            container.pop("shapes", None)
            shape = container.get("shape")
            if isinstance(shape, dict):
                for field_name in blocked & shape_fields:
                    shape.pop(field_name, None)
        if blocked & color_fields:
            container.pop("colors", None)
            roles = container.get("color_roles")
            if isinstance(roles, dict):
                role_names = {
                    "primary_colors": "primary",
                    "secondary_colors": "secondary",
                    "outline_colors": "outline",
                    "shadow_colors": "shadow",
                    "highlight_colors": "highlight",
                }
                for field_name in blocked & color_fields:
                    roles.pop(role_names.get(field_name, field_name), None)

    clean(record)
    conditioning = record.get("conditioning")
    if isinstance(conditioning, dict):
        for semantic_name in ("semantic_v4", "semantic_v3"):
            semantic = conditioning.get(semantic_name)
            if isinstance(semantic, dict):
                clean(semantic)
                attributes = semantic.get("attributes")
                if isinstance(attributes, dict):
                    clean(attributes)
        dropout_groups: set[str] = set()
        if "explicit_material" in blocked or "visual_material_cue" in blocked:
            dropout_groups.add("materials")
        if blocked & shape_fields:
            dropout_groups.add("shapes")
        if blocked & color_fields or "filename_color_hints" in blocked:
            dropout_groups.add("colors")
        if "role" in blocked:
            dropout_groups.add("function")
        if "style" in blocked:
            dropout_groups.add("style")
        for section_name in ("kept_attributes", "dropped_attributes"):
            section = conditioning.get(section_name)
            if isinstance(section, dict):
                for group in dropout_groups:
                    section.pop(group, None)
    for semantic_name in ("semantic_v4", "semantic_v3", "semantics"):
        semantic = record.get(semantic_name)
        if isinstance(semantic, dict):
            clean(semantic)
            attributes = semantic.get("attributes")
            if isinstance(attributes, dict):
                clean(attributes)


def _text_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [text for nested in value.values() for text in _text_values(nested)]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [text for nested in value for text in _text_values(nested)]
    text = str(value or "").strip().replace("_", " ")
    return [text] if text and text.lower() != "unknown" else []


def _remove_caption_phrases(caption: str, phrases: Sequence[str]) -> str:
    result = str(caption)
    for phrase in sorted(set(phrases), key=lambda value: (-len(value), value)):
        words = [re.escape(word) for word in phrase.split() if word]
        if not words:
            continue
        result = re.sub(r"(?<!\w)" + r"[ _-]+".join(words) + r"(?!\w)", " ", result, flags=re.IGNORECASE)
    return " ".join(result.split()).strip(" ,.;:-")


def _quality_calibrated(quality: Mapping[str, Any]) -> bool:
    critical = [quality["fields"][name] for name in CRITICAL_FIELDS if name in quality["fields"]]
    return bool(critical) and all(field.get("calibration_state") == "calibrated" for field in critical)


def _dimension(record: Mapping[str, Any], name: str) -> str:
    source = record.get("source") if isinstance(record.get("source"), Mapping) else {}
    aliases: dict[str, tuple[str, ...]] = {
        "source": ("source_id", "source_dataset"),
        "pack": ("source_pack", "pack"),
        "artist": ("artist", "author", "sub_artist"),
        "category": ("category",),
        "split": ("split",),
        "inference_path": ("inference_path",),
        "propagation_relation": ("propagation_relation",),
    }
    for key in aliases[name]:
        value = record.get(key, source.get(key))
        if value not in (None, ""):
            return str(value)
    return ""


def _grade_lowering_dimensions(base: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    field_summaries = (
        base.get("field_uncertainty_summaries") if isinstance(base.get("field_uncertainty_summaries"), Mapping) else {}
    )
    result["fields"] = sorted(
        (
            {"value": value, **dict(summary)}
            for value, summary in field_summaries.items()
            if isinstance(summary, Mapping) and summary.get("mean") is not None
        ),
        key=lambda item: (-float(item["mean"]), -int(item["count"]), str(item["value"])),
    )[:5]
    by = base.get("uncertainty_by") if isinstance(base.get("uncertainty_by"), Mapping) else {}
    for dimension in ("source", "pack", "artist", "category", "split", "inference_path", "propagation_relation"):
        groups = by.get(dimension) if isinstance(by.get(dimension), Mapping) else {}
        ranked = sorted(
            (
                {"value": value, **dict(summary)}
                for value, summary in groups.items()
                if isinstance(summary, Mapping) and summary.get("mean") is not None
            ),
            key=lambda item: (-float(item["mean"]), -int(item["count"]), str(item["value"])),
        )
        result[dimension] = ranked[:5]
    return result


def _first_numeric(record: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> float | None:
    for path in paths:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        if isinstance(value, str) and value in {"exact_rgba", "near_pixel", "suspicious_geometry"}:
            return 1.0
    return None


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else None
