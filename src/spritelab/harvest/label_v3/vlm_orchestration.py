"""Auto-Labeling v3: Multi-stage VLM orchestration.

Staged cascade: blind description → morphology extraction → constrained
classification → open-set verification → consistency verification.

Source hints, candidates, and pack context are strictly controlled per stage.
Every response records exposed context so fusion can discount correlated agreement.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable

from spritelab.harvest.label_v3.evidence import (
    SCHEMA_VERSION,
    EvidenceItem,
)
from spritelab.harvest.label_v3.sha256_utils import sha256_short, stable_evidence_id

logger = logging.getLogger(__name__)

VLM_STAGE_HASH = "vlm_staged_v3_2"

VlmStageId = Literal[
    "stage_a_blind_descriptor",
    "stage_b_morphology",
    "stage_c_constrained_classification",
    "stage_d_open_set_verify",
    "stage_e_consistency",
]


@dataclass(frozen=True)
class VlmRequestMetrics:
    """Reconciled counters for one logical stage request."""

    logical_stage_requests: int = 0
    successful_stage_outputs: int = 0
    cache_hits: int = 0
    http_attempts: int = 0
    retries: int = 0
    timeouts: int = 0
    transport_failures: int = 0
    json_parse_failures: int = 0
    schema_validation_failures: int = 0
    fallbacks: int = 0
    abstentions_caused_by_backend_failure: int = 0

    def merged(self, other: VlmRequestMetrics) -> VlmRequestMetrics:
        return VlmRequestMetrics(
            **{name: getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__}
        )

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, eq=False)
class VlmBackendResponse(Mapping[str, Any]):
    data: dict[str, Any]
    metrics: VlmRequestMetrics = field(default_factory=VlmRequestMetrics)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VlmBackendResponse):
            return self.data == other.data and self.metrics == other.metrics
        return self.data == other


@dataclass(frozen=True)
class VlmStageResult:
    """Result of one VLM stage for one sprite, or an unavailable marker."""

    stage_id: VlmStageId
    evidence: EvidenceItem | None = None
    available: bool = True
    failure_reason: str = ""
    source_hints_exposed: bool = False
    candidate_hints_exposed: bool = False
    context_views: tuple[str, ...] = ()
    model_identity: str = ""
    prompt_hash: str = ""
    cache_key: str = ""
    cache_hit: bool = False
    retry_count: int = 0
    metrics: VlmRequestMetrics = field(default_factory=VlmRequestMetrics)


@dataclass(frozen=True)
class VlmCascadeResult:
    """Complete multi-stage VLM results for one sprite."""

    sprite_id: str
    stage_a: VlmStageResult = field(default_factory=lambda: VlmStageResult(stage_id="stage_a_blind_descriptor"))
    stage_b: VlmStageResult = field(default_factory=lambda: VlmStageResult(stage_id="stage_b_morphology"))
    stage_c: VlmStageResult = field(
        default_factory=lambda: VlmStageResult(stage_id="stage_c_constrained_classification")
    )
    stage_d: VlmStageResult = field(default_factory=lambda: VlmStageResult(stage_id="stage_d_open_set_verify"))
    stage_e: VlmStageResult = field(default_factory=lambda: VlmStageResult(stage_id="stage_e_consistency"))
    all_failed: bool = False

    def available_stages(self) -> tuple[VlmStageResult, ...]:
        return tuple(
            s
            for s in [self.stage_a, self.stage_b, self.stage_c, self.stage_d, self.stage_e]
            if s.available and s.evidence is not None
        )

    def all_evidence(self) -> tuple[EvidenceItem, ...]:
        return tuple(s.evidence for s in self.available_stages() if s.evidence is not None)


STAGE_CONFIGS: dict[VlmStageId, dict[str, Any]] = {
    "stage_a_blind_descriptor": {
        "producer_stage": "vlm_stage_a_blind_descriptor",
        "evidence_family": "blind_vlm_descriptor",
        "target_fields": ("color", "shape", "tags", "description"),
        "source_hints_exposed": False,
        "candidate_hints_exposed": False,
        "context_views": ("isolated_checkerboard", "nearest_neighbor_scale", "tight_foreground_crop"),
        "description": "Blind visual description without source hints",
    },
    "stage_b_morphology": {
        "producer_stage": "vlm_stage_b_morphology",
        "evidence_family": "blind_vlm_descriptor",
        "target_fields": ("domain", "category", "shape"),
        "source_hints_exposed": False,
        "candidate_hints_exposed": False,
        "context_views": ("isolated_checkerboard", "nearest_neighbor_scale", "tight_foreground_crop"),
        "description": "Morphology extraction: shape, parts, composition",
    },
    "stage_c_constrained_classification": {
        "producer_stage": "vlm_stage_c_constrained_classification",
        "evidence_family": "constrained_vlm_classification",
        "target_fields": ("category", "canonical_object", "surface_alias"),
        "source_hints_exposed": True,
        "candidate_hints_exposed": True,
        "context_views": ("isolated_checkerboard", "nearest_neighbor_scale", "tight_foreground_crop"),
        "description": "Constrained classification from candidate pool",
    },
    "stage_d_open_set_verify": {
        "producer_stage": "vlm_stage_d_open_set_verify",
        "evidence_family": "vlm_open_set_verification",
        "target_fields": ("category", "canonical_object"),
        "source_hints_exposed": True,
        "candidate_hints_exposed": True,
        "context_views": ("isolated_checkerboard", "nearest_neighbor_scale", "tight_foreground_crop"),
        "description": "Independent visual verification of the proposed class; supported, compatible_but_too_specific, contradicted, or unknown",
    },
    "stage_e_consistency": {
        "producer_stage": "vlm_stage_e_consistency",
        "evidence_family": "vlm_open_set_verification",
        "target_fields": (),
        "source_hints_exposed": True,
        "candidate_hints_exposed": True,
        "context_views": ("sheet_context", "variant_group", "nearest_neighbors"),
        "description": "Reconcile prior stage outputs without proposing replacement prefills",
    },
}


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict) and isinstance(item.get("value"), str) and item["value"].strip():
                return item["value"].strip()
    return ""


def normalise_stage_fields(stage_id: VlmStageId, response_data: dict[str, Any]) -> dict[str, Any]:
    """Expose model output as field proposals without losing the raw artifact.

    Raw model scores stay in ``stage_output`` and are never treated as
    calibrated probabilities.  Keeping the normalised values at the top level
    makes the existing field-specific fusion contract consume VLM evidence
    instead of silently ignoring its nested JSON.
    """
    data = dict(response_data)
    result: dict[str, Any] = {"stage_output": data, "field_proposals": {}}
    if stage_id == "stage_e_consistency":
        result["consistency_result"] = data.get("consistency_result", "insufficient_context")
        result["confirmed_fields"] = list(data.get("confirmed_fields", ()))
        result["conflicts"] = list(data.get("conflicts", ()))
        return result
    color_roles = {
        "primary_colors": data.get("primary_colors") or (),
        "secondary_colors": data.get("secondary_colors") or (),
        "highlight_colors": data.get("highlight_colors") or (),
        "shadow_colors": data.get("shadow_colors") or (),
        "outline_color": data.get("outline_color") or "",
    }
    flattened_colors: list[str] = []
    for value in (*color_roles.values(), data.get("colors") or data.get("visible_colors") or ()):
        values_to_add = value if isinstance(value, (list, tuple)) else (value,)
        for color in values_to_add:
            if str(color).strip() and str(color).strip() not in flattened_colors:
                flattened_colors.append(str(color).strip())
    values = {
        "domain": data.get("domain") or _first_text(data.get("domain_candidates")),
        "category": data.get("category") or _first_text(data.get("category_candidates")),
        "canonical_object": data.get("specific_object")
        or data.get("canonical_object")
        or data.get("top_1")
        or _first_text(data.get("object_candidates")),
        "surface_alias": data.get("surface_alias_candidate"),
        # Flat color remains the legacy field; structured roles travel beside
        # it for GUI/detail consumers.
        "color": data.get("color") or _first_text(color_roles["primary_colors"]) or _first_text(flattened_colors),
        "material": data.get("material") or _first_text(data.get("materials")),
        "shape": data.get("shape")
        or _first_text(data.get("normalized_shape_candidates"))
        or _first_text(data.get("shape_features")),
        "role": data.get("role") or _first_text(data.get("role_candidates")),
        "description": data.get("literal_description") or data.get("description"),
    }
    if stage_id == "stage_b_morphology":
        values["shape"] = data.get("silhouette_family") or values["shape"]
    if stage_id == "stage_c_constrained_classification":
        values["canonical_object"] = data.get("top_1") or data.get("canonical_object") or data.get("choice")
        values["category"] = data.get("category") or values["category"]
        alternatives = data.get("top_3") or data.get("alternatives") or ()
        names: list[str] = []
        for alternative in alternatives:
            name = (
                str(alternative.get("value", alternative.get("candidate", "")))
                if isinstance(alternative, dict)
                else str(alternative)
            )
            if name:
                names.append(name)
        result["candidate_object_names"] = names
    for field_name, value in values.items():
        if isinstance(value, str) and value.strip() and value.strip().lower() not in {"unknown", "none_of_the_above"}:
            result[field_name] = value.strip()
            result["field_proposals"][field_name] = {
                "value": value.strip(),
                "alternatives": data.get("top_3") or data.get(f"{field_name}_candidates") or (),
                "raw_model_score": data.get("confidence"),
                "stage": stage_id,
            }
    if isinstance(data.get("tags"), (list, tuple)):
        result["tags"] = [str(v) for v in data["tags"] if str(v)]
    result.update(color_roles)
    result["colors"] = flattened_colors
    return result


def build_stage_evidence(
    stage_id: VlmStageId,
    sprite_id: str,
    source_id: str,
    pack_id: str,
    response_data: dict[str, Any] | None,
    *,
    model_identity: str = "",
    prompt_hash: str = "",
    image_hash: str = "",
    image_view: str = "magenta_matte",
) -> EvidenceItem:
    """Build an evidence item from a VLM stage response."""

    config = STAGE_CONFIGS[stage_id]
    eid = stable_evidence_id(sprite_id, stage_id, VLM_STAGE_HASH)

    if response_data is None:
        return EvidenceItem(
            schema_version=SCHEMA_VERSION,
            evidence_id=eid,
            sprite_id=sprite_id,
            source_id=source_id,
            pack_id=pack_id,
            evidence_family=config["evidence_family"],
            producer_stage=config["producer_stage"],
            target_fields=config["target_fields"],
            proposed_value={"unavailable": True},
            source_hints_exposed=config["source_hints_exposed"],
            candidate_hints_exposed=config["candidate_hints_exposed"],
            image_hash=image_hash,
            image_view=image_view,
            model_identity=model_identity,
            prompt_hash=prompt_hash,
            stage_hash=VLM_STAGE_HASH,
            deterministic=False,
            stochastic=True,
            warnings=("vlm_unavailable",),
        )

    normalized = normalise_stage_fields(stage_id, response_data)
    verification = str(response_data.get("verification_result", response_data.get("result", ""))).lower()
    consistency = str(response_data.get("consistency_result", "")).lower()
    warnings = [str(w) for w in response_data.get("warnings", ())]
    contradictions: tuple[str, ...] = ()
    if verification == "contradicted":
        warnings.append("verification_contradicted")
        # This code is deliberately scoped to the VLM dependency group.  The
        # pipeline uses it to abstain the VLM-derived proposal; it is not a
        # blanket claim that deterministic evidence is false.
        contradictions = ("vlm_verification_contradicted",)
    if consistency == "conflict":
        warnings.append("vlm_consistency_conflict")
        contradictions = ("vlm_consistency_conflict",)
    return EvidenceItem(
        schema_version=SCHEMA_VERSION,
        evidence_id=eid,
        sprite_id=sprite_id,
        source_id=source_id,
        pack_id=pack_id,
        evidence_family=config["evidence_family"],
        producer_stage=config["producer_stage"],
        target_fields=config["target_fields"],
        proposed_value=normalized,
        source_hints_exposed=config["source_hints_exposed"],
        candidate_hints_exposed=config["candidate_hints_exposed"],
        image_hash=image_hash,
        image_view=image_view,
        model_identity=model_identity,
        prompt_hash=prompt_hash,
        stage_hash=VLM_STAGE_HASH,
        deterministic=False,
        stochastic=True,
        raw_score=None,
        warnings=tuple(warnings),
        contradiction_codes=contradictions,
        provenance={"raw_model_score": response_data.get("confidence"), "stage": stage_id},
    )


def create_unavailable_cascade(
    sprite_id: str,
    *,
    reason: str = "no_vlm_backend",
) -> VlmCascadeResult:
    """Create a cascade result where all VLM stages are unavailable."""
    return VlmCascadeResult(
        sprite_id=sprite_id,
        stage_a=VlmStageResult(
            stage_id="stage_a_blind_descriptor",
            available=False,
            failure_reason=reason,
        ),
        stage_b=VlmStageResult(
            stage_id="stage_b_morphology",
            available=False,
            failure_reason=reason,
        ),
        stage_c=VlmStageResult(
            stage_id="stage_c_constrained_classification",
            available=False,
            failure_reason=reason,
        ),
        stage_d=VlmStageResult(
            stage_id="stage_d_open_set_verify",
            available=False,
            failure_reason=reason,
        ),
        stage_e=VlmStageResult(
            stage_id="stage_e_consistency",
            available=False,
            failure_reason=reason,
        ),
        all_failed=True,
    )


# ---------------------------------------------------------------------------
# Pluggable backend + real multi-stage orchestration
# ---------------------------------------------------------------------------

STAGE_ORDER: tuple[VlmStageId, ...] = (
    "stage_a_blind_descriptor",
    "stage_b_morphology",
    "stage_c_constrained_classification",
    "stage_d_open_set_verify",
    "stage_e_consistency",
)

# Explicit cache-identity contract. ``geometry_only`` is intentionally limited
# to Stage B, whose contract excludes colour, markings, and object identity.
VLM_STAGE_CACHE_IDENTITY: dict[VlmStageId, dict[str, bool]] = {
    "stage_a_blind_descriptor": {
        "exact_rgba": True,
        "geometry_only": False,
        "filename_source_context": False,
        "taxonomy": False,
        "prompt_version": True,
        "model_provider": True,
        "preprocessing_views": True,
        "context": False,
    },
    "stage_b_morphology": {
        "exact_rgba": False,
        "geometry_only": True,
        "filename_source_context": False,
        "taxonomy": False,
        "prompt_version": True,
        "model_provider": True,
        "preprocessing_views": True,
        "context": False,
    },
    "stage_c_constrained_classification": {
        "exact_rgba": True,
        "geometry_only": False,
        "filename_source_context": False,
        "taxonomy": True,
        "prompt_version": True,
        "model_provider": True,
        "preprocessing_views": True,
        "context": False,
    },
    "stage_d_open_set_verify": {
        "exact_rgba": True,
        "geometry_only": False,
        "filename_source_context": False,
        "taxonomy": True,
        "prompt_version": True,
        "model_provider": True,
        "preprocessing_views": True,
        "context": False,
    },
    "stage_e_consistency": {
        "exact_rgba": True,
        "geometry_only": False,
        "filename_source_context": False,
        "taxonomy": True,
        "prompt_version": True,
        "model_provider": True,
        "preprocessing_views": True,
        "context": True,
    },
}


def build_vlm_stage_cache_key(
    stage_id: VlmStageId,
    *,
    exact_rgba_hash: str,
    geometry_hash: str,
    image_view: str,
    preprocessing_hash: str,
    model_identity: str,
    prompt_version: str,
    prompt_hash: str,
    taxonomy_hash: str = "",
    context_hash: str = "",
) -> str:
    """Build a VLM cache key from the stage's declared visual contract."""
    identity = VLM_STAGE_CACHE_IDENTITY[stage_id]
    visual_hash = geometry_hash if identity["geometry_only"] else exact_rgba_hash
    if not visual_hash:
        raise ValueError(f"missing visual hash for {stage_id}")
    payload = {
        "stage": stage_id,
        "visual_kind": "geometry_alpha" if identity["geometry_only"] else "exact_exported_rgba",
        "visual_hash": visual_hash,
        "image_view": image_view,
        "preprocessing_hash": preprocessing_hash,
        "model_identity": model_identity,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "taxonomy_hash": taxonomy_hash if identity["taxonomy"] else "",
        "context_hash": context_hash if identity["context"] else "",
    }
    return sha256_short(json.dumps(payload, sort_keys=True), length=24)


# Fixed reliability given to a VLM stage's evidence in fusion. Intentionally
# independent of the model's self-reported confidence (which is untrusted).
VLM_STAGE_RELIABILITY = 0.7


class VlmUnavailable(RuntimeError):
    """Raised by a backend that cannot serve (transport/absent model)."""

    def __init__(self, message: str, metrics: VlmRequestMetrics | None = None):
        super().__init__(message)
        self.metrics = metrics or VlmRequestMetrics()


class VlmStageContractError(ValueError):
    """Raised when a backend response violates a stage schema."""

    def __init__(self, stage_id: str, raw: Any, metrics: VlmRequestMetrics):
        super().__init__(f"invalid_stage_contract:{stage_id}")
        self.stage_id = stage_id
        self.raw = raw
        self.metrics = metrics


@runtime_checkable
class VlmBackend(Protocol):
    """A minimal, mockable inference backend.

    ``model_identity`` participates in cache identity and dependency grouping.
    ``infer`` returns a structured dict or raises ``VlmUnavailable``. It must
    NOT be given source or candidate hints for the blind stage (the orchestrator
    enforces this by only passing candidates to stages that expose them).
    """

    model_identity: str

    def infer(
        self,
        *,
        stage_id: str,
        image_ref: str,
        prompt: str,
        prompt_hash: str,
        candidates: tuple[str, ...] | None = None,
    ) -> dict[str, Any] | VlmBackendResponse: ...


class UnavailableVlmBackend:
    """Default backend: always unavailable. No model, no network, no inference."""

    model_identity = "unavailable"

    def infer(self, **_kwargs: Any) -> dict[str, Any]:
        raise VlmUnavailable("no VLM backend configured")


def build_stage_prompt(
    stage_id: VlmStageId,
    candidates: tuple[str, ...] | None,
    proposed_classification: str = "",
    stage_context: str = "",
) -> str:
    """Return a strict stage prompt; the blind stages cannot receive hints."""
    common = (
        "Return JSON only. Report visible pixels, not lore, powers, game mechanics, style, "
        "or unsupported identity. Use unknown or none_of_the_above when the image does not support a claim. "
        "Model scores are uncalibrated evidence features, not probabilities."
    )
    if stage_id == "stage_a_blind_descriptor":
        return (
            common
            + " This is a blind isolated-sprite task: no filename, source, pack, taxonomy, mapping, or candidate is available. "
            + 'Do not classify or name the object. Schema: {"complete_object":bool,"likely_asset_type":string,'
            + '"primary_colors":[string],"secondary_colors":[string],"highlight_colors":[string],"shadow_colors":[string],"outline_color":string,'
            + '"visible_material_cues":[string],"normalized_shape_candidates":[string],'
            + '"raw_morphology":string,"components":[string],"orientation":string,"style_attributes":[string],'
            + '"literal_description":string,"object_free_description":string,"warnings":[string]}'
        )
    if stage_id == "stage_b_morphology":
        return (
            common
            + ' Extract morphology and select only a broad category/domain, never exact identity. Schema: {"silhouette_family":string,"aspect_ratio":number,'
            + '"major_components":[string],"relationships":[string],"symmetry":string,'
            + '"multipart_or_fragment":string,"complete_object":bool,"broad_visual_family":string,'
            + '"domain":string,"category":string,"warnings":[string]}'
        )
    if stage_id == "stage_d_open_set_verify":
        return (
            common
            + " Verify only whether the supplied proposed classification is visually supported. "
            + f"Proposed classification: {proposed_classification or 'unknown'}. "
            + 'Schema: {"verification_result":"supported|compatible_but_too_specific|contradicted|unknown",'
            + '"reason":string,"open_set":bool,"warnings":[string]}. '
            + "Do not propose a new identity."
        )
    if stage_id == "stage_e_consistency":
        return (
            common
            + " Reconcile the supplied prior-stage evidence. Do not classify the image and do not propose replacement prefills. "
            + f"Prior-stage evidence: {stage_context or 'unavailable'}. "
            + 'Schema: {"consistency_result":"consistent|conflict|insufficient_context",'
            + '"confirmed_fields":[string],"conflicts":[{"field":string,"values":[string]}],'
            + '"reason":string,"warnings":[string]}.'
        )
    choices = ", ".join(sorted(candidates or ()))
    return (
        common
        + " Constrained classification using only this vocabulary: ["
        + choices
        + "]. Include unknown and none_of_the_above. "
        + 'Separate what it looks like from what it is. Schema: {"broad_object":string,"specific_object":string,'
        + '"surface_alias_candidate":string,"top_1":string,"top_3":[{"value":string,"raw_score":number}],'
        + '"category":string,"hierarchy_path":[string],"too_specific":bool,"reason":string,'
        + '"none_of_the_above":bool,"warnings":[string]}'
    )


def parse_stage_output(stage_id: VlmStageId, raw: Any) -> dict[str, Any] | None:
    """Validate a backend response. Returns ``None`` for malformed output.

    Malformed output must never be cached as a successful decision — the caller
    quarantines the stage instead.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    if raw.get("_malformed"):
        return None
    required_signal = {
        "stage_a_blind_descriptor": {
            "literal_description",
            "description",
            "object_candidates",
            "colors",
            "primary_colors",
            "shape_features",
        },
        "stage_b_morphology": {"silhouette_family", "aspect_ratio", "major_components", "complete_object"},
    }
    if stage_id in required_signal and not any(key in raw for key in required_signal[stage_id]):
        return None
    if stage_id == "stage_c_constrained_classification":
        # Constrained/open-set stages must express a choice or an explicit
        # none-of-the-above / cannot_tell escape.
        has_choice = any(k in raw for k in ("canonical_object", "category", "choice"))
        has_escape = bool(raw.get("none_of_the_above") or raw.get("cannot_tell"))
        if not (has_choice or has_escape):
            return None
    if stage_id == "stage_d_open_set_verify":
        if str(raw.get("verification_result", raw.get("result", ""))).lower() not in {
            "supported",
            "compatible_but_too_specific",
            "contradicted",
            "unknown",
        }:
            return None
    if stage_id == "stage_e_consistency":
        if str(raw.get("consistency_result", "")).lower() not in {"consistent", "conflict", "insufficient_context"}:
            return None
        if not isinstance(raw.get("confirmed_fields"), list) or not isinstance(raw.get("conflicts"), list):
            return None
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("field"), str)
            or not isinstance(item.get("values"), list)
            for item in raw["conflicts"]
        ):
            return None
    return raw


def run_vlm_cascade(
    sprite_id: str,
    *,
    backend: VlmBackend | None = None,
    image_ref: str = "",
    image_hash: str = "",
    image_view: str = "magenta_matte",
    candidates: tuple[str, ...] = (),
    cache: Any | None = None,
    preprocessing_hash: str = "",
    taxonomy_hash: str = "",
    context_hash: str = "",
    geometry_hash: str = "",
    prompt_version: str = "vlm_prefill_v3_2",
    profile: str = "full",
) -> VlmCascadeResult:
    """Run the staged cascade against a pluggable backend.

    * The blind descriptor stage receives NO source/candidate hints.
    * Self-reported ``confidence`` is retained only as a feature (raw_score),
      never as an acceptance probability.
    * All stages from one model share dependency group ``vlm_<model>`` so
      correlated calls cannot multiply their vote in fusion.
    * Malformed output quarantines that stage (marked unavailable + reason),
      it is not stored as a successful decision.
    * Absent backend yields a fully-unavailable, resumable cascade.
    """
    if backend is None:
        backend = UnavailableVlmBackend()

    model_id = getattr(backend, "model_identity", "unknown")
    dep_group = f"vlm_{model_id}"
    stage_results: dict[str, VlmStageResult] = {}
    active_stages = {
        "fast": {"stage_a_blind_descriptor", "stage_c_constrained_classification"},
        "balanced": {
            "stage_a_blind_descriptor",
            "stage_b_morphology",
            "stage_c_constrained_classification",
            "stage_d_open_set_verify",
        },
        "full": set(STAGE_ORDER),
    }.get(profile)
    if active_stages is None:
        raise ValueError(f"unknown VLM cascade profile: {profile}")

    for stage_id in STAGE_ORDER:
        if stage_id not in active_stages:
            stage_results[stage_id] = VlmStageResult(
                stage_id=stage_id,
                available=False,
                failure_reason=f"skipped:{profile}_profile",
                model_identity=model_id,
            )
            continue
        cfg = STAGE_CONFIGS[stage_id]
        exposed = candidates if cfg["candidate_hints_exposed"] else None
        proposal = ""
        stage_context = ""
        if stage_id == "stage_d_open_set_verify":
            previous = stage_results.get("stage_c_constrained_classification")
            if previous and previous.evidence and isinstance(previous.evidence.proposed_value, dict):
                proposal = str(previous.evidence.proposed_value.get("canonical_object", ""))
        if stage_id == "stage_e_consistency":
            prior = {
                name: result.evidence.proposed_value.get("stage_output", {})
                for name, result in stage_results.items()
                if result.evidence is not None and isinstance(result.evidence.proposed_value, dict)
            }
            stage_context = json.dumps(prior, sort_keys=True, default=str)[:4000]
        prompt = build_stage_prompt(stage_id, exposed, proposal, stage_context)
        prompt_hash = sha256_short(prompt, length=16)
        cache_key = build_vlm_stage_cache_key(
            stage_id,
            exact_rgba_hash=image_hash or sha256_short(f"missing-image:{sprite_id}", length=24),
            geometry_hash=geometry_hash or image_hash or sha256_short(f"missing-geometry:{sprite_id}", length=24),
            image_view=image_view,
            preprocessing_hash=preprocessing_hash or "unspecified",
            model_identity=model_id,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            taxonomy_hash=taxonomy_hash,
            context_hash=sha256_short(f"{context_hash}|{stage_context}", length=16) if stage_context else context_hash,
        )

        transport_metrics = VlmRequestMetrics()

        def compute(
            _stage_id: VlmStageId = stage_id,
            _prompt: str = prompt,
            _prompt_hash: str = prompt_hash,
            _exposed: tuple[str, ...] | None = exposed,
            _cache_key: str = cache_key,
        ) -> dict[str, Any]:
            nonlocal transport_metrics
            response = backend.infer(
                stage_id=_stage_id,
                image_ref=image_ref,
                prompt=_prompt,
                prompt_hash=_prompt_hash,
                candidates=_exposed,
            )
            if isinstance(response, VlmBackendResponse):
                transport_metrics = response.metrics
                raw = response.data
            else:
                raw = response
            parsed = parse_stage_output(_stage_id, raw)
            if parsed is None:
                schema_metrics = transport_metrics.merged(VlmRequestMetrics(schema_validation_failures=1))
                recorder = getattr(backend, "record_schema_failure", None)
                if callable(recorder):
                    try:
                        recorder(stage_id=_stage_id, raw=raw, prompt_hash=_prompt_hash, cache_hash=_cache_key)
                    except Exception as exc:
                        logger.warning("VLM diagnostic capture failed for %s: %s", _stage_id, type(exc).__name__)
                raise VlmStageContractError(_stage_id, raw, schema_metrics)
            return parsed

        try:
            if cache is not None:
                parsed, cache_hit = cache.get_or_compute(cache_key, compute)
            else:
                parsed, cache_hit = compute(), False
        except VlmUnavailable as exc:
            metrics = VlmRequestMetrics(
                logical_stage_requests=1, fallbacks=1, abstentions_caused_by_backend_failure=1
            ).merged(exc.metrics)
            stage_results[stage_id] = VlmStageResult(
                stage_id=stage_id,
                available=False,
                failure_reason=f"unavailable:{exc}",
                model_identity=model_id,
                prompt_hash=prompt_hash,
                cache_key=cache_key,
                retry_count=metrics.retries,
                metrics=metrics,
            )
            continue
        except VlmStageContractError as exc:
            metrics = VlmRequestMetrics(logical_stage_requests=1, fallbacks=1).merged(exc.metrics)
            stage_results[stage_id] = VlmStageResult(
                stage_id=stage_id,
                available=False,
                failure_reason="invalid_stage_contract",
                model_identity=model_id,
                prompt_hash=prompt_hash,
                cache_key=cache_key,
                retry_count=metrics.retries,
                metrics=metrics,
            )
            continue
        except Exception as exc:
            metrics = VlmRequestMetrics(
                logical_stage_requests=1, fallbacks=1, abstentions_caused_by_backend_failure=1
            ).merged(transport_metrics)
            stage_results[stage_id] = VlmStageResult(
                stage_id=stage_id,
                available=False,
                failure_reason=f"error:{type(exc).__name__}",
                model_identity=model_id,
                prompt_hash=prompt_hash,
                cache_key=cache_key,
                retry_count=metrics.retries,
                metrics=metrics,
            )
            continue

        metrics = VlmRequestMetrics(
            logical_stage_requests=1,
            successful_stage_outputs=1,
            cache_hits=int(cache_hit),
        ).merged(VlmRequestMetrics() if cache_hit else transport_metrics)

        evidence = build_stage_evidence(
            stage_id,
            sprite_id,
            "",
            "",
            parsed,
            model_identity=model_id,
            prompt_hash=prompt_hash,
            image_hash=image_hash,
            image_view=image_view,
        )
        # Correlated calls from one model share a dependency group. The VLM's
        # self-reported confidence is retained only as a provenance FEATURE — it
        # is never used as reliability weight or acceptance probability; the
        # stage's fixed reliability is used instead, and acceptance is gated by
        # per-field calibration downstream.
        self_conf = parsed.get("confidence")
        evidence = replace(
            evidence,
            dependency_group=dep_group,
            raw_score=VLM_STAGE_RELIABILITY,
            provenance={
                **dict(evidence.provenance),
                "self_reported_confidence": self_conf,
                "confidence_is_feature_only": True,
            },
        )
        stage_results[stage_id] = VlmStageResult(
            stage_id=stage_id,
            evidence=evidence,
            available=True,
            source_hints_exposed=cfg["source_hints_exposed"],
            candidate_hints_exposed=cfg["candidate_hints_exposed"],
            context_views=tuple(cfg["context_views"]),
            model_identity=model_id,
            prompt_hash=prompt_hash,
            cache_key=cache_key,
            cache_hit=cache_hit,
            retry_count=metrics.retries,
            metrics=metrics,
        )

    result = VlmCascadeResult(
        sprite_id=sprite_id,
        stage_a=stage_results["stage_a_blind_descriptor"],
        stage_b=stage_results["stage_b_morphology"],
        stage_c=stage_results["stage_c_constrained_classification"],
        stage_d=stage_results["stage_d_open_set_verify"],
        stage_e=stage_results["stage_e_consistency"],
        all_failed=all(not s.available for s in stage_results.values()),
    )
    return result
