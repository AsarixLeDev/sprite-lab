"""Sparse audit allocation and monotonic error-risk calibration for v4."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from spritelab.harvest.label_v4.risk import CalibrationState

CALIBRATION_SCHEMA_VERSION = "label_risk_calibration_v1.0"
AUDIT_SET_SCHEMA_VERSION = "label_audit_set_v1.0"
DEFAULT_AUDIT_DIMENSIONS: tuple[str, ...] = (
    "uncertainty_1_20",
    "field",
    "category",
    "pack",
    "artist",
    "source",
    "pack_seen_state",
    "open_set_state",
    "inference_path",
    "propagation_relation",
    "suitability_status",
)


@dataclass(frozen=True)
class IsotonicBin:
    lower_score: float
    upper_score: float
    p_error: float
    support_n: int
    error_n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_score": self.lower_score,
            "upper_score": self.upper_score,
            "p_error": self.p_error,
            "support_n": self.support_n,
            "error_n": self.error_n,
        }


@dataclass(frozen=True)
class StratumCalibration:
    field_name: str
    stratum: str
    calibration_state: CalibrationState
    support_n: int
    error_n: int
    bins: tuple[IsotonicBin, ...] = ()
    borrowed_from: str = ""
    brier_score: float | None = None
    expected_calibration_error: float | None = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def predict(self, raw_risk: float, *, confidence_level: float = 0.95) -> dict[str, Any]:
        if self.calibration_state == "not_scorable" or not self.bins:
            return {
                "calibration_state": "uncalibrated",
                "calibration_stratum": self.stratum,
                "calibration_support_n": self.support_n,
            }
        score = _clip(raw_risk)
        selected = self.bins[-1]
        for bin_ in self.bins:
            if score <= bin_.upper_score:
                selected = bin_
                break
        upper = wilson_upper_bound(selected.error_n, selected.support_n, confidence_level=confidence_level)
        return {
            "p_error_estimate": selected.p_error,
            "risk_upper_95": max(selected.p_error, upper),
            "calibration_state": self.calibration_state,
            "calibration_stratum": self.stratum,
            "calibration_support_n": selected.support_n,
            "borrowed_from": self.borrowed_from,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field": self.field_name,
            "stratum": self.stratum,
            "calibration_state": self.calibration_state,
            "support_n": self.support_n,
            "error_n": self.error_n,
            "bins": [bin_.to_dict() for bin_ in self.bins],
            "borrowed_from": self.borrowed_from,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
        }


@dataclass(frozen=True)
class CalibrationBundle:
    strata: tuple[StratumCalibration, ...]
    audit_set_hash: str
    feature_definition_hash: str
    risk_model_version: str = "label_risk_v1"
    schema_version: str = CALIBRATION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "risk_model_version": self.risk_model_version,
            "audit_set_hash": self.audit_set_hash,
            "feature_definition_hash": self.feature_definition_hash,
            "strata": [row.to_dict() for row in self.strata],
            "metadata": dict(self.metadata),
        }


def fit_isotonic(
    raw_risks: Sequence[float],
    errors: Sequence[int | bool],
) -> tuple[IsotonicBin, ...]:
    """Fit a deterministic PAV isotonic calibrator for error probability."""

    if len(raw_risks) != len(errors):
        raise ValueError("raw_risks and errors must have equal length")
    if not raw_risks:
        return ()
    ordered = sorted((_clip(risk), int(bool(error))) for risk, error in zip(raw_risks, errors, strict=True))
    # Aggregate ties first.  PAV then merges adjacent decreasing means.
    tied: list[dict[str, float | int]] = []
    for score, error in ordered:
        if tied and float(tied[-1]["upper"]) == score:
            tied[-1]["support"] = int(tied[-1]["support"]) + 1
            tied[-1]["errors"] = int(tied[-1]["errors"]) + error
        else:
            tied.append({"lower": score, "upper": score, "support": 1, "errors": error})

    blocks: list[dict[str, float | int]] = []
    for row in tied:
        blocks.append(dict(row))
        while len(blocks) >= 2 and _block_mean(blocks[-2]) > _block_mean(blocks[-1]):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                {
                    "lower": float(left["lower"]),
                    "upper": float(right["upper"]),
                    "support": int(left["support"]) + int(right["support"]),
                    "errors": int(left["errors"]) + int(right["errors"]),
                }
            )
    return tuple(
        IsotonicBin(
            lower_score=float(block["lower"]),
            upper_score=float(block["upper"]),
            p_error=_block_mean(block),
            support_n=int(block["support"]),
            error_n=int(block["errors"]),
        )
        for block in blocks
    )


def build_calibration_bundle(
    reviewed_rows: Sequence[Mapping[str, Any]],
    *,
    min_support: int = 30,
    borrow_min_support: int = 60,
    feature_definition: Mapping[str, Any] | str = "label_risk_v1_features",
) -> CalibrationBundle:
    """Build per-field/per-stratum calibrators from human error judgments.

    Required row keys are ``field``, ``stratum``, ``raw_risk`` and
    ``is_error``.  Rows may additionally include human/source metadata.  A
    stratum below ``min_support`` is never marked calibrated; it may borrow a
    sufficiently supported field-global calibrator, otherwise it remains
    uncalibrated.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    field_global: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    canonical_rows: list[dict[str, Any]] = []
    for raw in reviewed_rows:
        field_name = str(raw.get("field", "")).strip()
        stratum = str(raw.get("stratum", "")).strip() or "global"
        if not field_name or raw.get("raw_risk") is None or raw.get("is_error") is None:
            continue
        row = {
            "field": field_name,
            "stratum": stratum,
            "raw_risk": _clip(float(raw["raw_risk"])),
            "is_error": int(bool(raw["is_error"])),
            "review_id": str(raw.get("review_id", "")),
        }
        canonical_rows.append(row)
        grouped[(field_name, stratum)].append(row)
        field_global[field_name].append(row)

    global_models: dict[str, StratumCalibration] = {}
    for field_name, rows in sorted(field_global.items()):
        if len(rows) >= borrow_min_support:
            global_models[field_name] = _fit_stratum(field_name, "global", rows, state="calibrated")

    results: list[StratumCalibration] = []
    for (field_name, stratum), rows in sorted(grouped.items()):
        if len(rows) >= min_support:
            results.append(_fit_stratum(field_name, stratum, rows, state="calibrated"))
        elif field_name in global_models:
            global_model = global_models[field_name]
            results.append(
                StratumCalibration(
                    field_name=field_name,
                    stratum=stratum,
                    calibration_state="borrowed_calibration",
                    support_n=len(rows),
                    error_n=sum(int(row["is_error"]) for row in rows),
                    bins=global_model.bins,
                    borrowed_from=f"{field_name}/global",
                    brier_score=global_model.brier_score,
                    expected_calibration_error=global_model.expected_calibration_error,
                )
            )
        else:
            results.append(
                StratumCalibration(
                    field_name=field_name,
                    stratum=stratum,
                    calibration_state="uncalibrated",
                    support_n=len(rows),
                    error_n=sum(int(row["is_error"]) for row in rows),
                )
            )

    present_keys = {(row.field_name, row.stratum) for row in results}
    for field_name, model in global_models.items():
        if (field_name, "global") not in present_keys:
            results.append(model)

    audit_hash = _stable_hash(canonical_rows)
    feature_hash = _stable_hash(feature_definition)
    return CalibrationBundle(
        strata=tuple(sorted(results, key=lambda row: (row.field_name, row.stratum))),
        audit_set_hash=audit_hash,
        feature_definition_hash=feature_hash,
        metadata={
            "reviewed_rows": len(canonical_rows),
            "min_support": int(min_support),
            "borrow_min_support": int(borrow_min_support),
            "calibration_states": dict(Counter(row.calibration_state for row in results)),
        },
    )


def allocate_sparse_audit(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_size: int = 400,
    seed: int = 17,
    dimensions: Sequence[str] = DEFAULT_AUDIT_DIMENSIONS,
) -> dict[str, Any]:
    """Select a frozen sparse audit with risk/sparsity-aware coverage.

    The selector greedily covers rare dimension values, then high uncertainty
    and calibration-sparse strata.  It never mutates input records and emits a
    content hash over the ordered selected identifiers and sampling policy.
    """

    target = max(0, min(int(target_size), len(candidates)))
    rng = random.Random(int(seed))
    rows = [dict(row) for row in candidates]
    value_counts: dict[tuple[str, str], int] = Counter()
    for row in rows:
        for dimension in dimensions:
            value_counts[(dimension, _dimension_value(row, dimension))] += 1

    jitter = {id(row): rng.random() * 1.0e-6 for row in rows}

    def priority(row: Mapping[str, Any]) -> tuple[float, str]:
        rarity = sum(
            1.0 / max(1, value_counts[(dimension, _dimension_value(row, dimension))]) for dimension in dimensions
        )
        uncertainty = float(row.get("uncertainty_1_20", row.get("record_uncertainty_1_20", 10))) / 20.0
        calibration_sparse = 1.0 if str(row.get("calibration_state", "uncalibrated")) != "calibrated" else 0.0
        open_set = 0.5 if str(row.get("open_set_state", "")) in {"open_set", "novel"} else 0.0
        score = rarity + uncertainty * 1.5 + calibration_sparse * 1.25 + open_set + jitter[id(row)]
        return (-score, _row_id(row))

    selected = sorted(rows, key=priority)[:target]
    selected_ids = [_row_id(row) for row in selected]
    policy = {
        "schema_version": AUDIT_SET_SCHEMA_VERSION,
        "seed": int(seed),
        "target_size": target,
        "dimensions": list(dimensions),
        "selector": "rare-strata+high-risk+calibration-sparse_v1",
    }
    return {
        **policy,
        "selected_ids": selected_ids,
        "selected_records": selected,
        "audit_set_hash": _stable_hash({"policy": policy, "selected_ids": selected_ids}),
        "coverage": audit_coverage(selected, dimensions=dimensions),
    }


def adaptive_audit_allocation(
    candidates: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    seed: int = 17,
) -> dict[str, Any]:
    """Allocate periodic audits toward new and calibration-sparse strata."""

    return allocate_sparse_audit(candidates, target_size=budget, seed=seed)


def audit_coverage(
    rows: Iterable[Mapping[str, Any]],
    *,
    dimensions: Sequence[str] = DEFAULT_AUDIT_DIMENSIONS,
) -> dict[str, dict[str, int]]:
    coverage: dict[str, Counter[str]] = {dimension: Counter() for dimension in dimensions}
    for row in rows:
        for dimension in dimensions:
            coverage[dimension][_dimension_value(row, dimension)] += 1
    return {dimension: dict(sorted(counter.items())) for dimension, counter in coverage.items()}


def reliability_diagram(
    predicted_risks: Sequence[float],
    errors: Sequence[int | bool],
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    if len(predicted_risks) != len(errors):
        raise ValueError("predictions and errors must have equal length")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(max(1, int(bins)))]
    for risk, error in zip(predicted_risks, errors, strict=True):
        value = _clip(risk)
        index = min(len(buckets) - 1, int(value * len(buckets)))
        buckets[index].append((value, int(bool(error))))
    diagram: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        support = len(bucket)
        observed = sum(error for _, error in bucket) / support if support else None
        mean_predicted = sum(risk for risk, _ in bucket) / support if support else None
        diagram.append(
            {
                "bin": index,
                "lower": index / len(buckets),
                "upper": (index + 1) / len(buckets),
                "support_n": support,
                "mean_predicted_risk": mean_predicted,
                "observed_error_rate": observed,
                "risk_upper_95": wilson_upper_bound(sum(error for _, error in bucket), support) if support else None,
            }
        )
    return diagram


def selective_risk_curve(
    predicted_risks: Sequence[float],
    errors: Sequence[int | bool],
) -> list[dict[str, Any]]:
    """Return coverage-versus-error when accepting lowest-risk labels first."""

    if len(predicted_risks) != len(errors):
        raise ValueError("predictions and errors must have equal length")
    ordered = sorted((_clip(risk), int(bool(error))) for risk, error in zip(predicted_risks, errors, strict=True))
    curve: list[dict[str, Any]] = []
    cumulative_errors = 0
    total = len(ordered)
    for index, (threshold, error) in enumerate(ordered, start=1):
        cumulative_errors += error
        curve.append(
            {
                "coverage": index / total if total else 0.0,
                "accepted": index,
                "risk_threshold": threshold,
                "observed_error_rate": cumulative_errors / index,
                "risk_upper_95": wilson_upper_bound(cumulative_errors, index),
            }
        )
    return curve


def calibration_metrics(predicted_risks: Sequence[float], errors: Sequence[int | bool]) -> dict[str, Any]:
    if len(predicted_risks) != len(errors):
        raise ValueError("predictions and errors must have equal length")
    if not predicted_risks:
        return {
            "support_n": 0,
            "brier_score": None,
            "expected_calibration_error": None,
            "reliability_diagram": [],
            "selective_risk_curve": [],
        }
    predictions = [_clip(value) for value in predicted_risks]
    labels = [int(bool(value)) for value in errors]
    return {
        "support_n": len(predictions),
        "brier_score": brier_score(predictions, labels),
        "expected_calibration_error": expected_calibration_error(predictions, labels),
        "reliability_diagram": reliability_diagram(predictions, labels),
        "selective_risk_curve": selective_risk_curve(predictions, labels),
    }


def brier_score(predicted_risks: Sequence[float], errors: Sequence[int | bool]) -> float:
    if not predicted_risks or len(predicted_risks) != len(errors):
        raise ValueError("non-empty equal-length predictions and errors required")
    return sum((_clip(p) - int(bool(y))) ** 2 for p, y in zip(predicted_risks, errors, strict=True)) / len(
        predicted_risks
    )


def expected_calibration_error(
    predicted_risks: Sequence[float], errors: Sequence[int | bool], *, bins: int = 10
) -> float:
    if not predicted_risks or len(predicted_risks) != len(errors):
        raise ValueError("non-empty equal-length predictions and errors required")
    total = len(predicted_risks)
    return sum(
        row["support_n"] / total * abs(float(row["mean_predicted_risk"]) - float(row["observed_error_rate"]))
        for row in reliability_diagram(predicted_risks, errors, bins=bins)
        if row["support_n"]
    )


def wilson_upper_bound(error_count: int, total: int, *, confidence_level: float = 0.95) -> float:
    """One-sided Wilson upper confidence bound for a binomial error rate."""

    total = int(total)
    error_count = max(0, min(total, int(error_count)))
    if total <= 0:
        return 1.0
    p = error_count / total
    z = _one_sided_z(confidence_level)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return _clip(center + margin)


def _fit_stratum(
    field_name: str,
    stratum: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    state: CalibrationState,
) -> StratumCalibration:
    risks = [float(row["raw_risk"]) for row in rows]
    errors = [int(row["is_error"]) for row in rows]
    bins = fit_isotonic(risks, errors)
    predictions = [_predict_bins(bins, risk) for risk in risks]
    return StratumCalibration(
        field_name=field_name,
        stratum=stratum,
        calibration_state=state,
        support_n=len(rows),
        error_n=sum(errors),
        bins=bins,
        brier_score=brier_score(predictions, errors),
        expected_calibration_error=expected_calibration_error(predictions, errors),
    )


def _predict_bins(bins: Sequence[IsotonicBin], raw_risk: float) -> float:
    value = _clip(raw_risk)
    for bin_ in bins:
        if value <= bin_.upper_score:
            return bin_.p_error
    return bins[-1].p_error if bins else 1.0


def _block_mean(block: Mapping[str, float | int]) -> float:
    return int(block["errors"]) / max(1, int(block["support"]))


def _dimension_value(row: Mapping[str, Any], dimension: str) -> str:
    value = row.get(dimension)
    if value is None and dimension == "field":
        value = row.get("field_name")
    if isinstance(value, (list, tuple, set)):
        return "|".join(sorted(str(item) for item in value)) or "<missing>"
    return str(value) if value not in {None, ""} else "<missing>"


def _row_id(row: Mapping[str, Any]) -> str:
    sprite_id = str(row.get("sprite_id", ""))
    field_name = str(row.get("field", row.get("field_name", "")))
    return f"{sprite_id}:{field_name}" if field_name else sprite_id


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _one_sided_z(confidence_level: float) -> float:
    lookup = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.96, 0.99: 2.3263, 0.995: 2.5758}
    return lookup.get(round(float(confidence_level), 3), 1.6449)
