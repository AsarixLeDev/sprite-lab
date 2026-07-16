"""Auto-Labeling v3: versioned evidence contract.

Each evidence item is an immutable, versioned record of one piece of evidence
contributed by a named producer stage. Evidence items may carry proposed values,
raw scores, structured observations, and metadata sufficient to reproduce the
evidence independently.

Do not import this module inside hot-paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "evidence_v3.1"

EvidenceFamily = Literal[
    "source_profile",
    "filename",
    "archive_path",
    "declarative_sheet_mapping",
    "sheet_coordinate",
    "variant_group",
    "deterministic_visual",
    "color_palette",
    "image_embedding",
    "nearest_neighbor",
    "blind_vlm_descriptor",
    "constrained_vlm_classification",
    "vlm_open_set_verification",
    "pack_consistency",
    "human_calibration",
    "legacy_v2_derived",
]

EvidenceProducer = Literal[
    "deterministic",
    "vlm_stage_a_blind_descriptor",
    "vlm_stage_b_morphology",
    "vlm_stage_c_constrained_classification",
    "vlm_stage_d_open_set_verify",
    "vlm_stage_e_consistency",
    "embedding_index",
    "pack_analysis",
    "fusion",
    "human_correction",
]

CalibrationStratum = Literal[
    "source_specific",
    "source_profile",
    "domain",
    "global",
    "uncalibrated",
]

TargetField = Literal[
    "domain",
    "category",
    "canonical_object",
    "surface_alias",
    "color",
    "material",
    "shape",
    "role",
    "tags",
    "description",
]


@dataclass(frozen=True)
class EvidenceItem:
    """One versioned piece of evidence for one or more target fields."""

    schema_version: str = SCHEMA_VERSION
    evidence_id: str = ""
    sprite_id: str = ""
    source_id: str = ""
    pack_id: str = ""
    evidence_family: EvidenceFamily = "deterministic_visual"
    producer_stage: EvidenceProducer = "deterministic"
    target_fields: tuple[TargetField, ...] = ()
    proposed_value: Any = None
    candidate_distribution: dict[str, float] = field(default_factory=dict)
    raw_score: float | None = None
    calibration_stratum: CalibrationStratum = "uncalibrated"
    dependency_group: str = ""
    source_hints_exposed: bool = False
    candidate_hints_exposed: bool = False
    input_artifact_refs: tuple[str, ...] = ()
    image_hash: str = ""
    image_view: str = ""
    model_identity: str = ""
    prompt_hash: str = ""
    prompt_version: str = ""
    config_hash: str = ""
    stage_hash: str = ""
    deterministic: bool = False
    stochastic: bool = False
    warnings: tuple[str, ...] = ()
    contradiction_codes: tuple[str, ...] = ()
    timestamp: str = ""
    build_identity: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def evidence_item_to_json(item: EvidenceItem) -> dict[str, Any]:
    return {
        "schema_version": item.schema_version,
        "evidence_id": item.evidence_id,
        "sprite_id": item.sprite_id,
        "source_id": item.source_id,
        "pack_id": item.pack_id,
        "evidence_family": item.evidence_family,
        "producer_stage": item.producer_stage,
        "target_fields": list(item.target_fields),
        "proposed_value": item.proposed_value,
        "candidate_distribution": dict(item.candidate_distribution),
        "raw_score": item.raw_score,
        "calibration_stratum": item.calibration_stratum,
        "dependency_group": item.dependency_group,
        "source_hints_exposed": item.source_hints_exposed,
        "candidate_hints_exposed": item.candidate_hints_exposed,
        "input_artifact_refs": list(item.input_artifact_refs),
        "image_hash": item.image_hash,
        "image_view": item.image_view,
        "model_identity": item.model_identity,
        "prompt_hash": item.prompt_hash,
        "prompt_version": item.prompt_version,
        "config_hash": item.config_hash,
        "stage_hash": item.stage_hash,
        "deterministic": item.deterministic,
        "stochastic": item.stochastic,
        "warnings": list(item.warnings),
        "contradiction_codes": list(item.contradiction_codes),
        "timestamp": item.timestamp,
        "build_identity": item.build_identity,
        "provenance": dict(item.provenance),
    }


def evidence_item_from_json(data: Mapping[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        evidence_id=str(data.get("evidence_id", "")),
        sprite_id=str(data.get("sprite_id", "")),
        source_id=str(data.get("source_id", "")),
        pack_id=str(data.get("pack_id", "")),
        evidence_family=str(data.get("evidence_family", "deterministic_visual")),
        producer_stage=str(data.get("producer_stage", "deterministic")),
        target_fields=tuple(str(v) for v in data.get("target_fields") or ()),
        proposed_value=data.get("proposed_value"),
        candidate_distribution={str(k): float(v) for k, v in (data.get("candidate_distribution") or {}).items()},
        raw_score=float(data["raw_score"]) if data.get("raw_score") is not None else None,
        calibration_stratum=str(data.get("calibration_stratum", "uncalibrated")),
        dependency_group=str(data.get("dependency_group", "")),
        source_hints_exposed=bool(data.get("source_hints_exposed")),
        candidate_hints_exposed=bool(data.get("candidate_hints_exposed")),
        input_artifact_refs=tuple(str(v) for v in data.get("input_artifact_refs") or ()),
        image_hash=str(data.get("image_hash", "")),
        image_view=str(data.get("image_view", "")),
        model_identity=str(data.get("model_identity", "")),
        prompt_hash=str(data.get("prompt_hash", "")),
        prompt_version=str(data.get("prompt_version", "")),
        config_hash=str(data.get("config_hash", "")),
        stage_hash=str(data.get("stage_hash", "")),
        deterministic=bool(data.get("deterministic")),
        stochastic=bool(data.get("stochastic")),
        warnings=tuple(str(v) for v in data.get("warnings") or ()),
        contradiction_codes=tuple(str(v) for v in data.get("contradiction_codes") or ()),
        timestamp=str(data.get("timestamp", "")),
        build_identity=str(data.get("build_identity", "")),
        provenance=dict(data.get("provenance") or {}),
    )
