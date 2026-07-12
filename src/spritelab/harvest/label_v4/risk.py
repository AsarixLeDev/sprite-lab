"""Calibrated field and record error-risk contracts for Labeling v4.

The public score is an uncertainty score, not model confidence:

``1`` is the lowest estimated error risk and ``20`` is the highest.  Even a
score of 1 is not a claim of zero error.  Scores are derived from the upper
95% error-risk bound so sparse evidence cannot look more certain than it is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

RISK_MODEL_VERSION = "label_risk_v1.1"
RISK_SCHEMA_VERSION = "label_field_risk_v1.0"
RECORD_RISK_SCHEMA_VERSION = "label_record_risk_v1.0"

CalibrationState = Literal["calibrated", "borrowed_calibration", "uncalibrated", "not_scorable"]

CRITICAL_FIELDS: tuple[str, ...] = ("canonical_object", "category", "domain", "role")

# Every semantic leaf is scored independently.  Parent structures such as
# ``shape`` and ``color_roles`` may additionally have summaries, but these
# leaves are never hidden inside one aggregate confidence value.
SEMANTIC_FIELDS: tuple[str, ...] = (
    "domain",
    "category",
    "canonical_object",
    "surface_alias",
    "role",
    "explicit_material",
    "visual_material_cue",
    "silhouette",
    "aspect",
    "orientation",
    "structure",
    "edge_profile",
    "parts",
    "palette_colors",
    "primary_colors",
    "secondary_colors",
    "outline_colors",
    "shadow_colors",
    "highlight_colors",
    "filename_color_hints",
    "size_hint",
    "condition",
    "style",
    "description",
)


@dataclass(frozen=True)
class RiskBandThresholds:
    """Inclusive configurable thresholds for the five training bands."""

    strong_max: int = 4
    usable_weak_max: int = 8
    auxiliary_only_max: int = 12
    excluded_max: int = 16
    abstain_max: int = 20

    def __post_init__(self) -> None:
        values = (
            self.strong_max,
            self.usable_weak_max,
            self.auxiliary_only_max,
            self.excluded_max,
            self.abstain_max,
        )
        if values != tuple(sorted(values)) or values[-1] != 20 or values[0] < 1:
            raise ValueError("risk-band thresholds must be increasing, start above 0, and end at 20")

    def band(self, score: int) -> str:
        score = clamp_score(score)
        if score <= self.strong_max:
            return "strong"
        if score <= self.usable_weak_max:
            return "usable_weak"
        if score <= self.auxiliary_only_max:
            return "auxiliary_only"
        if score <= self.excluded_max:
            return "excluded_from_primary_supervision"
        return "abstain_or_quarantine"


DEFAULT_BANDS = RiskBandThresholds()


@dataclass(frozen=True)
class FieldRisk:
    field_name: str
    p_error_estimate: float | None
    risk_upper_95: float | None
    uncertainty_1_20: int | None
    uncertainty_band: str
    calibration_support_n: int
    calibration_stratum: str
    calibration_state: CalibrationState
    risk_signals: tuple[str, ...] = ()
    risk_model_version: str = RISK_MODEL_VERSION
    not_scorable_reason: str = ""
    schema_version: str = RISK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.calibration_state == "not_scorable":
            if self.uncertainty_1_20 is not None or self.risk_upper_95 is not None:
                raise ValueError("not_scorable fields cannot carry a numeric risk score")
            if not self.not_scorable_reason:
                raise ValueError("not_scorable fields require a reason")
            return
        if self.uncertainty_1_20 is None or not 1 <= self.uncertainty_1_20 <= 20:
            raise ValueError("scorable fields require uncertainty_1_20 in [1, 20]")
        for value in (self.p_error_estimate, self.risk_upper_95):
            if value is None or not 0.0 <= value <= 1.0:
                raise ValueError("scorable fields require probabilities in [0, 1]")
        if self.p_error_estimate > self.risk_upper_95:
            raise ValueError("point risk cannot exceed its upper confidence bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field": self.field_name,
            "p_error_estimate": self.p_error_estimate,
            "risk_upper_95": self.risk_upper_95,
            "uncertainty_1_20": self.uncertainty_1_20,
            "uncertainty_band": self.uncertainty_band,
            "calibration_support_n": self.calibration_support_n,
            "calibration_stratum": self.calibration_stratum,
            "calibration_state": self.calibration_state,
            "risk_signals": list(self.risk_signals),
            "risk_model_version": self.risk_model_version,
            "not_scorable_reason": self.not_scorable_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FieldRisk:
        return cls(
            field_name=str(data.get("field", "")),
            p_error_estimate=_optional_probability(data.get("p_error_estimate")),
            risk_upper_95=_optional_probability(data.get("risk_upper_95")),
            uncertainty_1_20=int(data["uncertainty_1_20"]) if data.get("uncertainty_1_20") is not None else None,
            uncertainty_band=str(data.get("uncertainty_band", "not_scorable")),
            calibration_support_n=max(0, int(data.get("calibration_support_n", 0))),
            calibration_stratum=str(data.get("calibration_stratum", "")),
            calibration_state=str(data.get("calibration_state", "uncalibrated")),
            risk_signals=tuple(str(value) for value in data.get("risk_signals") or ()),
            risk_model_version=str(data.get("risk_model_version", RISK_MODEL_VERSION)),
            not_scorable_reason=str(data.get("not_scorable_reason", "")),
            schema_version=str(data.get("schema_version", RISK_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class RecordRisk:
    critical_field_max_uncertainty: int | None
    mean_field_uncertainty: float | None
    weighted_mean_uncertainty: float | None
    record_uncertainty_1_20: int | None
    record_risk_upper_95: float | None
    critical_field_risk_upper: float | None
    contradiction_penalty: float
    critical_field_scores: dict[str, int | None] = field(default_factory=dict)
    schema_version: str = RECORD_RISK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "critical_field_max_uncertainty": self.critical_field_max_uncertainty,
            "mean_field_uncertainty": self.mean_field_uncertainty,
            "weighted_mean_uncertainty": self.weighted_mean_uncertainty,
            "record_uncertainty_1_20": self.record_uncertainty_1_20,
            "record_risk_upper_95": self.record_risk_upper_95,
            "critical_field_risk_upper": self.critical_field_risk_upper,
            "contradiction_penalty": self.contradiction_penalty,
            "critical_field_scores": dict(self.critical_field_scores),
        }


@dataclass(frozen=True)
class RiskPolicy:
    """Conservative defaults used until human calibration is available."""

    bands: RiskBandThresholds = DEFAULT_BANDS
    min_calibration_support: int = 30
    uncalibrated_critical_upper: float = 0.72
    uncalibrated_optional_upper: float = 0.58
    borrowed_penalty: float = 0.08
    contradiction_floor_upper: float = 0.45
    contradiction_penalty_per_conflict: float = 0.05
    max_contradiction_penalty: float = 0.25
    forced_exclusion_floor_upper: float = 0.65
    forced_abstention_floor_upper: float = 0.85


DEFAULT_RISK_POLICY = RiskPolicy()


def risk_score_from_upper(risk_upper_95: float) -> int:
    """Map an upper error bound to 1--20 using the public contract."""

    upper = min(1.0, max(0.0, float(risk_upper_95)))
    return clamp_score(math.ceil(20.0 * upper))


def clamp_score(score: int) -> int:
    return max(1, min(20, int(score)))


def not_scorable_field(field_name: str, reason: str) -> FieldRisk:
    return FieldRisk(
        field_name=field_name,
        p_error_estimate=None,
        risk_upper_95=None,
        uncertainty_1_20=None,
        uncertainty_band="not_scorable",
        calibration_support_n=0,
        calibration_stratum="not_scorable",
        calibration_state="not_scorable",
        not_scorable_reason=reason,
    )


def estimate_field_risk(
    field_name: str,
    *,
    value_present: bool,
    risk_features: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    policy: RiskPolicy = DEFAULT_RISK_POLICY,
) -> FieldRisk:
    """Estimate field error risk without treating model confidence as calibration.

    ``risk_features`` are observable agreement/conflict/provenance signals.
    A model's self-reported confidence is deliberately ignored.  When no
    supported calibration row exists, the function returns a conservative
    high uncertainty and the explicit ``uncalibrated`` state.
    """

    features = dict(risk_features or {})
    if not value_present and not (
        features.get("explicit_abstention")
        or features.get("force_abstention")
        or features.get("generic_visual_form_promoted")
    ):
        return not_scorable_field(field_name, "field_has_no_proposed_or_accepted_value")
    signals = _risk_signals(features)
    critical = field_name in CRITICAL_FIELDS
    state: CalibrationState = "uncalibrated"
    support = 0
    stratum = "uncalibrated"
    point = 0.5 if critical else 0.4
    upper = policy.uncalibrated_critical_upper if critical else policy.uncalibrated_optional_upper

    if calibration:
        requested_state = str(calibration.get("calibration_state", "uncalibrated"))
        support = max(0, int(calibration.get("calibration_support_n", calibration.get("sample_count", 0))))
        stratum = str(calibration.get("calibration_stratum", calibration.get("stratum", "uncalibrated")))
        calibrated_point = _optional_probability(calibration.get("p_error_estimate"))
        calibrated_upper = _optional_probability(calibration.get("risk_upper_95"))
        if requested_state == "calibrated" and support >= policy.min_calibration_support:
            state = "calibrated"
            if calibrated_point is not None:
                point = calibrated_point
            if calibrated_upper is not None:
                upper = max(point, calibrated_upper)
        elif requested_state in {"calibrated", "borrowed_calibration"} and calibrated_upper is not None:
            state = "borrowed_calibration"
            point = calibrated_point if calibrated_point is not None else min(calibrated_upper, point)
            upper = max(point, calibrated_upper + policy.borrowed_penalty)
        else:
            # Unsupported calibration metadata is retained for audit, but does
            # not lower the conservative uncalibrated bound.
            state = "uncalibrated"
            stratum = stratum or "uncalibrated"

    feature_delta = _feature_risk_delta(features)
    # Positive deltas increase both point and upper risk.  Negative deltas may
    # lower a *calibrated* estimate, but never lower conservative uncalibrated
    # defaults merely because several models agreed with one another.
    point = _clip_probability(point + feature_delta)
    if state == "uncalibrated":
        upper = max(upper, point)
    else:
        upper = max(point, _clip_probability(upper + feature_delta))

    contradiction_count = max(0, int(features.get("contradiction_count", 0)))
    if contradiction_count or bool(features.get("unresolved_conflict")):
        upper = max(upper, policy.contradiction_floor_upper)
    upper = max(upper, _forced_risk_floor(features, policy))
    point = min(upper, point)
    upper = _clip_probability(upper)
    score = risk_score_from_upper(upper)
    return FieldRisk(
        field_name=field_name,
        p_error_estimate=point,
        risk_upper_95=upper,
        uncertainty_1_20=score,
        uncertainty_band=policy.bands.band(score),
        calibration_support_n=support,
        calibration_stratum=stratum,
        calibration_state=state,
        risk_signals=signals,
    )


def ensure_all_field_risks(
    values: Mapping[str, Any],
    supplied: Mapping[str, FieldRisk | Mapping[str, Any]],
) -> dict[str, FieldRisk]:
    """Return an explicit risk or ``not_scorable`` state for every field."""

    result: dict[str, FieldRisk] = {}
    for field_name in SEMANTIC_FIELDS:
        raw = supplied.get(field_name)
        if isinstance(raw, FieldRisk):
            result[field_name] = raw
        elif isinstance(raw, Mapping):
            payload = dict(raw)
            payload.setdefault("field", field_name)
            result[field_name] = FieldRisk.from_dict(payload)
        else:
            present = _value_present(values.get(field_name))
            result[field_name] = (
                estimate_field_risk(field_name, value_present=True)
                if present
                else not_scorable_field(field_name, "field_not_emitted")
            )
    return result


def summarize_record_risk(
    field_risks: Mapping[str, FieldRisk | Mapping[str, Any]],
    *,
    contradiction_count: int = 0,
    field_weights: Mapping[str, float] | None = None,
    policy: RiskPolicy = DEFAULT_RISK_POLICY,
) -> RecordRisk:
    """Summarize without allowing easy optional fields to hide critical risk."""

    parsed = {
        name: value if isinstance(value, FieldRisk) else FieldRisk.from_dict({"field": name, **dict(value)})
        for name, value in field_risks.items()
    }
    scorable = [risk for risk in parsed.values() if risk.uncertainty_1_20 is not None]
    critical = [parsed[name] for name in CRITICAL_FIELDS if name in parsed and parsed[name].risk_upper_95 is not None]
    critical_scores = {name: parsed[name].uncertainty_1_20 if name in parsed else None for name in CRITICAL_FIELDS}
    if not scorable or not critical:
        return RecordRisk(
            critical_field_max_uncertainty=max(
                (score for score in critical_scores.values() if score is not None), default=None
            ),
            mean_field_uncertainty=_mean([risk.uncertainty_1_20 for risk in scorable]),
            weighted_mean_uncertainty=_weighted_mean(parsed, field_weights or {}),
            record_uncertainty_1_20=None,
            record_risk_upper_95=None,
            critical_field_risk_upper=max(
                (risk.risk_upper_95 for risk in critical if risk.risk_upper_95 is not None), default=None
            ),
            contradiction_penalty=0.0,
            critical_field_scores=critical_scores,
        )

    critical_upper = max(float(risk.risk_upper_95) for risk in critical if risk.risk_upper_95 is not None)
    contradiction_penalty = min(
        policy.max_contradiction_penalty,
        max(0, int(contradiction_count)) * policy.contradiction_penalty_per_conflict,
    )
    record_upper = _clip_probability(critical_upper + contradiction_penalty)
    record_score = risk_score_from_upper(record_upper)
    return RecordRisk(
        critical_field_max_uncertainty=max(int(risk.uncertainty_1_20) for risk in critical if risk.uncertainty_1_20),
        mean_field_uncertainty=_mean([risk.uncertainty_1_20 for risk in scorable]),
        weighted_mean_uncertainty=_weighted_mean(parsed, field_weights or {}),
        record_uncertainty_1_20=record_score,
        record_risk_upper_95=record_upper,
        critical_field_risk_upper=critical_upper,
        contradiction_penalty=contradiction_penalty,
        critical_field_scores=critical_scores,
    )


def _risk_signals(features: Mapping[str, Any]) -> tuple[str, ...]:
    signals: list[str] = []
    named_flags = (
        "deterministic_evidence_strong",
        "vlm_verifier_agreement",
        "filename_vlm_agreement",
        "taxonomy_compatible",
        "open_set_novelty",
        "variant_family_inconsistent",
        "image_quality_low",
        "provenance_incomplete",
        "legacy_adapter_used",
        "cache_version_mismatch",
        "description_claim_invalid",
        "material_not_explicit",
        "color_role_inconsistent",
        "unresolved_conflict",
        "generic_visual_form_promoted",
        "pack_only_category",
        "source_category_visual_ambiguity",
        "surface_alias_from_pack_metadata",
        "description_uses_alternative_interpretation",
        "same_model_verifier",
        "verifier_no_decision_change",
        "invalid_taxonomy_provider_output",
        "color_role_outside_palette",
        "explicit_abstention",
    )
    for name in named_flags:
        if bool(features.get(name)):
            signals.append(name)
    dependency_groups = int(features.get("independent_dependency_groups", 0) or 0)
    if dependency_groups:
        signals.append(f"independent_dependency_groups:{dependency_groups}")
    contradictions = int(features.get("contradiction_count", 0) or 0)
    if contradictions:
        signals.append(f"contradiction_count:{contradictions}")
    return tuple(signals)


def _feature_risk_delta(features: Mapping[str, Any]) -> float:
    delta = 0.0
    if bool(features.get("deterministic_evidence_strong")):
        delta -= 0.06
    groups = max(0, int(features.get("independent_dependency_groups", 0) or 0))
    delta -= min(0.08, max(0, groups - 1) * 0.025)
    for name, amount in (
        ("vlm_verifier_agreement", -0.04),
        ("filename_vlm_agreement", -0.04),
        ("taxonomy_compatible", -0.02),
        ("open_set_novelty", 0.10),
        ("variant_family_inconsistent", 0.12),
        ("image_quality_low", 0.10),
        ("provenance_incomplete", 0.08),
        ("legacy_adapter_used", 0.18),
        ("cache_version_mismatch", 0.20),
        ("description_claim_invalid", 0.14),
        ("material_not_explicit", 0.12),
        ("color_role_inconsistent", 0.10),
        ("unresolved_conflict", 0.18),
        ("generic_visual_form_promoted", 0.30),
        ("pack_only_category", 0.10),
        ("source_category_visual_ambiguity", 0.14),
        ("surface_alias_from_pack_metadata", 0.24),
        ("description_uses_alternative_interpretation", 0.30),
        ("same_model_verifier", 0.08),
        ("verifier_no_decision_change", 0.08),
        ("invalid_taxonomy_provider_output", 0.24),
        ("color_role_outside_palette", 0.16),
    ):
        if bool(features.get(name)):
            delta += amount
    delta += min(0.30, max(0, int(features.get("contradiction_count", 0) or 0)) * 0.08)
    novelty = features.get("open_set_novelty_score")
    if novelty is not None:
        delta += 0.10 * _clip_probability(float(novelty))
    return max(-0.18, min(0.55, delta))


def _forced_risk_floor(features: Mapping[str, Any], policy: RiskPolicy) -> float:
    requested = _optional_probability(features.get("risk_upper_floor")) or 0.0
    if bool(features.get("force_exclusion")) or bool(features.get("description_uses_alternative_interpretation")):
        requested = max(requested, policy.forced_exclusion_floor_upper)
    if (
        bool(features.get("force_abstention"))
        or bool(features.get("generic_visual_form_promoted"))
        or bool(features.get("explicit_abstention"))
    ):
        requested = max(requested, policy.forced_abstention_floor_upper)
    return _clip_probability(requested)


def _weighted_mean(risks: Mapping[str, FieldRisk], weights: Mapping[str, float]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for name, risk in risks.items():
        if risk.uncertainty_1_20 is None:
            continue
        weight = max(0.0, float(weights.get(name, 2.0 if name in CRITICAL_FIELDS else 1.0)))
        numerator += float(risk.uncertainty_1_20) * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _mean(values: Sequence[int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _optional_probability(value: Any) -> float | None:
    if value is None:
        return None
    return _clip_probability(float(value))
