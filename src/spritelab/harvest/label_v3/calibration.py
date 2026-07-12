"""Auto-Labeling v3: versioned calibration artifacts.

Per-field calibration using hierarchical fallback:
  1. source-specific (if sample support sufficient)
  2. source-profile
  3. domain
  4. global
  5. uncalibrated (not eligible for auto-accept)

Calibration artifacts record all dependencies for cache invalidation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from spritelab.harvest.label_v3.sha256_utils import dict_hash, sha256_short

logger = logging.getLogger(__name__)

CALIBRATION_SCHEMA_VERSION = "calibration_v3.1"


@dataclass(frozen=True)
class CalibrationStratumData:
    field: str
    stratum: str
    sample_count: int
    error_count: int
    observed_precision: float
    calibrated_probability: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ece: float | None = None
    sufficient: bool = False


@dataclass(frozen=True)
class CalibrationArtifact:
    """Versioned calibration data for one field/stratum combination."""

    schema_version: str = CALIBRATION_SCHEMA_VERSION
    field_name: str = ""
    evidence_policy: str = ""
    calibration_split_identity: str = ""
    source_strata: tuple[str, ...] = ()
    domain_strata: tuple[str, ...] = ()
    model_identity: str = ""
    prompt_hash: str = ""
    prompt_version: str = ""
    taxonomy_hash: str = ""
    taxonomy_version: str = ""
    feature_definition_hash: str = ""
    strata_data: tuple[CalibrationStratumData, ...] = ()
    observed_errors: dict[str, int] = field(default_factory=dict)
    calibration_metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    build_identity: str = ""
    invalidation_dependencies: dict[str, str] = field(default_factory=dict)

    def artifact_hash(self) -> str:
        return dict_hash(self.to_json_dict())

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field": self.field_name,
            "evidence_policy": self.evidence_policy,
            "calibration_split_identity": self.calibration_split_identity,
            "source_strata": list(self.source_strata),
            "domain_strata": list(self.domain_strata),
            "model_identity": self.model_identity,
            "prompt_hash": self.prompt_hash,
            "prompt_version": self.prompt_version,
            "taxonomy_hash": self.taxonomy_hash,
            "taxonomy_version": self.taxonomy_version,
            "feature_definition_hash": self.feature_definition_hash,
            "strata_data": [
                {
                    "field": s.field,
                    "stratum": s.stratum,
                    "sample_count": s.sample_count,
                    "error_count": s.error_count,
                    "observed_precision": s.observed_precision,
                    "calibrated_probability": s.calibrated_probability,
                    "ci_lower": s.ci_lower,
                    "ci_upper": s.ci_upper,
                    "ece": s.ece,
                    "sufficient": s.sufficient,
                }
                for s in self.strata_data
            ],
            "observed_errors": dict(self.observed_errors),
            "calibration_metrics": dict(self.calibration_metrics),
            "created_at": self.created_at,
            "build_identity": self.build_identity,
            "invalidation_dependencies": dict(self.invalidation_dependencies),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> CalibrationArtifact:
        strata = tuple(
            CalibrationStratumData(
                field=str(s.get("field", "")),
                stratum=str(s.get("stratum", "")),
                sample_count=int(s.get("sample_count", 0)),
                error_count=int(s.get("error_count", 0)),
                observed_precision=float(s.get("observed_precision", 0.0)),
                calibrated_probability=float(s["calibrated_probability"])
                if s.get("calibrated_probability") is not None
                else None,
                ci_lower=float(s["ci_lower"]) if s.get("ci_lower") is not None else None,
                ci_upper=float(s["ci_upper"]) if s.get("ci_upper") is not None else None,
                ece=float(s["ece"]) if s.get("ece") is not None else None,
                sufficient=bool(s.get("sufficient", False)),
            )
            for s in data.get("strata_data") or ()
        )
        return cls(
            schema_version=str(data.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
            field_name=str(data.get("field", "")),
            evidence_policy=str(data.get("evidence_policy", "")),
            calibration_split_identity=str(data.get("calibration_split_identity", "")),
            source_strata=tuple(str(v) for v in data.get("source_strata") or ()),
            domain_strata=tuple(str(v) for v in data.get("domain_strata") or ()),
            model_identity=str(data.get("model_identity", "")),
            prompt_hash=str(data.get("prompt_hash", "")),
            prompt_version=str(data.get("prompt_version", "")),
            taxonomy_hash=str(data.get("taxonomy_hash", "")),
            taxonomy_version=str(data.get("taxonomy_version", "")),
            feature_definition_hash=str(data.get("feature_definition_hash", "")),
            strata_data=strata,
            observed_errors={str(k): int(v) for k, v in (data.get("observed_errors") or {}).items()},
            calibration_metrics=dict(data.get("calibration_metrics") or {}),
            created_at=str(data.get("created_at", "")),
            build_identity=str(data.get("build_identity", "")),
            invalidation_dependencies={
                str(k): str(v) for k, v in (data.get("invalidation_dependencies") or {}).items()
            },
        )


def compute_lower_confidence_bound(
    accepted_correct: int,
    accepted_total: int,
    confidence_level: float = 0.95,
) -> float:
    """One-sided lower confidence bound on precision using Wilson score interval.

    When accepted_total is 0, returns 0.0.
    """
    import math

    if accepted_total == 0:
        return 0.0

    p = accepted_correct / accepted_total
    z = _z_score_one_sided(confidence_level)
    denominator = 1 + z * z / accepted_total
    center = (p + z * z / (2 * accepted_total)) / denominator
    margin = (
        z * math.sqrt((p * (1 - p) / accepted_total) + (z * z / (4 * accepted_total * accepted_total))) / denominator
    )
    return max(0.0, center - margin)


def compute_ece(
    predicted_probs: Sequence[float],
    true_labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE)."""
    import math

    if not predicted_probs or len(predicted_probs) != len(true_labels):
        return 1.0

    data = list(zip(predicted_probs, true_labels, strict=True))
    data.sort(key=lambda x: x[0])

    bin_size = max(1, math.ceil(len(data) / n_bins))
    ece = 0.0
    for i in range(0, len(data), bin_size):
        bin_data = data[i : i + bin_size]
        if not bin_data:
            continue
        avg_conf = sum(p for p, _ in bin_data) / len(bin_data)
        avg_acc = sum(t for _, t in bin_data) / len(bin_data)
        weight = len(bin_data) / len(data)
        ece += weight * abs(avg_acc - avg_conf)
    return ece


def stratum_sufficient(
    sample_count: int,
    *,
    min_samples: int = 30,
    min_samples_per_stratum: int = 10,
) -> bool:
    return sample_count >= min_samples and sample_count >= min_samples_per_stratum


def calibration_support_for_field(
    field: str,
    source_id: str,
    source_profile_name: str,
    domain: str,
    artifact: CalibrationArtifact,
    *,
    min_samples: int = 30,
) -> dict[str, Any] | None:
    """Find the best calibration stratum for a field given source/domain context.

    Returns None if no sufficient calibration is available.
    """

    strata_order = [
        f"source:{source_id}",
        f"profile:{source_profile_name}",
        f"domain:{domain}",
        "global",
    ]

    for target_stratum in strata_order:
        for stratum_data in artifact.strata_data:
            if stratum_data.field == field and stratum_data.stratum == target_stratum:
                if stratum_data.sufficient and stratum_data.sample_count >= min_samples:
                    return {
                        "stratum": target_stratum,
                        "sample_count": stratum_data.sample_count,
                        "observed_precision": stratum_data.observed_precision,
                        "calibrated_probability": stratum_data.calibrated_probability,
                        "ci_lower": stratum_data.ci_lower,
                        "ci_upper": stratum_data.ci_upper,
                        "ece": stratum_data.ece,
                    }
    return None


def _z_score_one_sided(confidence: float) -> float:

    _Z_LOOKUP = {
        0.90: 1.2816,
        0.95: 1.6449,
        0.975: 1.9600,
        0.99: 2.3263,
        0.995: 2.5758,
        0.999: 3.0902,
    }
    return _Z_LOOKUP.get(confidence, 1.6449)


def build_empty_calibration_artifact(
    field: str,
    *,
    taxonomy_hash: str = "",
    prompt_hash: str = "",
    model_identity: str = "",
) -> CalibrationArtifact:
    return CalibrationArtifact(
        field_name=field,
        taxonomy_hash=taxonomy_hash or sha256_short("no_taxonomy"),
        prompt_hash=prompt_hash or sha256_short("no_prompt"),
        model_identity=model_identity or "unknown",
        evidence_policy="v3_deterministic+vlm_staged",
        calibration_split_identity="uncalibrated",
        feature_definition_hash=sha256_short(f"v3_fusion_features_{field}"),
        created_at="",
        build_identity="",
    )
