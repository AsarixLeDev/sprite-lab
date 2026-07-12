"""End-to-end risk-aware Labeling v4 pipeline.

The pipeline implements the sequence:

    high-freedom proposal -> controlled normalization -> evidence
    reconciliation -> independent verification when routed -> calibrated risk
    estimation -> training policy metadata

Provider output is always evidence, never truth.  The default execution path
uses deterministic mocks and makes no network calls.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from spritelab.harvest.label_v4.cache import (
    INDEPENDENT_VERIFIER_NAMESPACE,
    TEXT_RECONCILIATION_NAMESPACE,
    LabelV4Cache,
    blind_proposal_identity,
    make_cache_identity,
    verifier_identity,
)
from spritelab.harvest.label_v4.description import choose_or_regenerate_description
from spritelab.harvest.label_v4.filename_parser import FilenameParseResult, parse_filename_semantics
from spritelab.harvest.label_v4.pixel_evidence import analyze_pixels
from spritelab.harvest.label_v4.proposal import (
    BLIND_VLM_PROMPT_VERSION,
    VLM_PROPOSAL_SCHEMA_VERSION,
    VLMProposalArtifact,
    build_blind_vlm_prompt,
    is_generic_visual_form,
    parse_blind_vlm_response,
)
from spritelab.harvest.label_v4.providers import MockJSONProvider, ProviderArtifact
from spritelab.harvest.label_v4.reconciliation import (
    RECONCILIATION_PROMPT_VERSION,
    RECONCILIATION_SCHEMA_VERSION,
    FieldProposal,
    ReconciliationResult,
    build_reconciliation_prompt,
    parse_reconciliation_response,
    reconcile_evidence,
)
from spritelab.harvest.label_v4.risk import (
    CRITICAL_FIELDS,
    SEMANTIC_FIELDS,
    FieldRisk,
    RiskPolicy,
    ensure_all_field_risks,
    estimate_field_risk,
    summarize_record_risk,
)
from spritelab.harvest.label_v4.routing import (
    DEFAULT_ROUTING_PROFILE,
    INDEPENDENT_VERIFIER_PROMPT_VERSION,
    AdaptiveRoutingDecision,
    AdaptiveRoutingSignals,
    RoutingProfile,
    decide_adaptive_routing,
    decide_profile_routing,
    field_coverage_after_stage_a,
)
from spritelab.harvest.label_v4.semantic_axes import (
    CATEGORY_VALUES,
    DOMAIN_VALUES,
    ROLE_VALUES,
    normalize_semantic_term,
    normalize_visual_color_roles,
)
from spritelab.harvest.label_v4.training_quality import normalize_training_quality
from spritelab.harvest.label_v4.verifier import (
    ParsedVerifierResponse,
    VerifierDecisionSummary,
    classify_verifier_independence,
    derive_verifier_effects,
    parse_verifier_response,
)

LABEL_V4_RECORD_SCHEMA_VERSION = "label_record_v4.1"
PIPELINE_VERSION = "label_pipeline_v4.2"
VERIFIER_SCHEMA_VERSION = "label_verifier_v4.1"

PipelineMode = Literal["adaptive", "A", "B", "C"]


@dataclass(frozen=True)
class LabelV4PipelineConfig:
    mode: PipelineMode = "adaptive"
    cache_dir: Path | None = None
    use_cache: bool = True
    verifier_risk_threshold: float = 0.40
    novelty_threshold: float = 0.65
    force_vlm_for_comparison: bool = False
    risk_policy: RiskPolicy = field(default_factory=RiskPolicy)
    calibration_by_field: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    known_object_vocabulary: tuple[str, ...] = ()
    blind_max_tokens: int = 1024
    reconciliation_max_tokens: int = 1536
    verifier_max_tokens: int = 1024
    shared_cache: bool = False
    require_shared_bc_cache: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    routing_profile: RoutingProfile = DEFAULT_ROUTING_PROFILE

    def __post_init__(self) -> None:
        if self.mode not in {"adaptive", "A", "B", "C"}:
            raise ValueError("mode must be adaptive, A, B, or C")
        if self.routing_profile not in {"semantic_minimal", "semantic_plus_visual", "full_diagnostic"}:
            raise ValueError("invalid routing_profile")
        for name in ("blind_max_tokens", "reconciliation_max_tokens", "verifier_max_tokens"):
            value = int(getattr(self, name))
            if not 1 <= value <= 8192:
                raise ValueError(f"{name} must be between 1 and 8192")
        if self.require_shared_bc_cache and (self.mode != "C" or not self.shared_cache or self.cache_dir is None):
            raise ValueError("require_shared_bc_cache requires mode C and an enabled shared cache")
        prices = (self.input_cost_per_million, self.output_cost_per_million)
        if any(value is not None and float(value) < 0.0 for value in prices):
            raise ValueError("provider prices must be non-negative")


class _ProviderFailure(RuntimeError):
    """Internal cache escape carrying a failed artifact without caching it.

    Exception instances must remain mutable because ``contextlib`` assigns
    ``__traceback__`` while unwinding generator-based context managers.
    """

    def __init__(self, artifact: ProviderArtifact) -> None:
        super().__init__(f"provider stage failed: {artifact.stage}")
        self.artifact = artifact


class RequiredSharedCacheMiss(RuntimeError):
    """Fail closed before a paid B/C call when exact shared reuse is required."""

    def __init__(self, stage: str, cache_key: str) -> None:
        super().__init__(f"required shared cache artifact missing for {stage}: {cache_key}")
        self.stage = stage
        self.cache_key = cache_key


@dataclass(frozen=True)
class _ProviderExecution:
    artifact: ProviderArtifact
    cache_hit: bool
    cache_key: str
    shared_cache_hit: bool
    execution_latency_ms: float


DEFAULT_PIPELINE_CONFIG = LabelV4PipelineConfig()


def label_record_v4(
    record: Mapping[str, Any],
    *,
    image_path: str | Path | None = None,
    config: LabelV4PipelineConfig = DEFAULT_PIPELINE_CONFIG,
    vlm_provider: Any | None = None,
    text_provider: Any | None = None,
    verifier_provider: Any | None = None,
) -> dict[str, Any]:
    """Label one record with deterministic evidence and adaptively routed mocks/providers."""

    source_record = dict(record)
    sprite_id = str(source_record.get("sprite_id", ""))
    resolved_image = _resolve_image_path(source_record, image_path)
    pixel_evidence = analyze_pixels(resolved_image)
    pack_context = infer_pack_context(source_record)
    deterministic = parse_filename_semantics(source_record, pack_context=pack_context)
    initial_signals = _initial_routing_signals(source_record, deterministic, pixel_evidence)
    stage_a_coverage = field_coverage_after_stage_a(deterministic)
    initial_route = decide_profile_routing(
        initial_signals,
        critical_semantics_complete=bool(stage_a_coverage["critical_semantics_complete"]),
        profile=config.routing_profile,
    )
    run_b = (
        config.mode in {"B", "C"}
        or config.force_vlm_for_comparison
        or (config.mode == "adaptive" and initial_route.run_stage_b)
    )
    cache = LabelV4Cache(config.cache_dir) if config.cache_dir and config.use_cache else None
    ledger: list[dict[str, Any]] = [
        {
            "stage": "A_deterministic",
            "provider_call": False,
            "cache_hit": False,
            "image_hash": pixel_evidence["image_hash"],
        }
    ]

    vlm_artifact: VLMProposalArtifact | None = None
    if run_b:
        provider = vlm_provider or _default_blind_mock(pixel_evidence)
        execution = _call_provider_cached(
            provider,
            cache=cache,
            cache_kind="blind",
            stage="B_blind_vlm_proposal",
            prompt=build_blind_vlm_prompt(),
            prompt_version=BLIND_VLM_PROMPT_VERSION,
            schema_version=VLM_PROPOSAL_SCHEMA_VERSION,
            image_path=resolved_image,
            payload=None,
            max_tokens=config.blind_max_tokens,
            shared_cache=config.shared_cache,
            require_cache_hit=config.require_shared_bc_cache,
        )
        provider_artifact = execution.artifact
        vlm_artifact = parse_blind_vlm_response(
            provider_artifact.raw_output,
            model_identity=provider_artifact.model_identity,
            request_hash=provider_artifact.request_hash,
            image_hash=provider_artifact.image_hash,
            prompt_version=provider_artifact.prompt_version,
            latency_ms=provider_artifact.latency_ms,
            token_usage=provider_artifact.token_usage,
        )
        if provider_artifact.failure_diagnostics and vlm_artifact.failure is None:
            vlm_artifact = parse_blind_vlm_response(
                "",
                model_identity=provider_artifact.model_identity,
                request_hash=provider_artifact.request_hash,
                image_hash=provider_artifact.image_hash,
                prompt_version=provider_artifact.prompt_version,
                latency_ms=provider_artifact.latency_ms,
                token_usage=provider_artifact.token_usage,
            )
        ledger.append(_ledger_row("B_blind_vlm_proposal", execution))

    run_c = (
        config.mode in {"B", "C"} or (config.mode == "adaptive" and initial_route.run_stage_c)
    ) and config.mode != "A"
    baseline_reconciliation = reconcile_evidence(
        deterministic,
        vlm_artifact,
        known_object_vocabulary=config.known_object_vocabulary,
        promotion_evidence=_visual_promotion_evidence(vlm_artifact),
    )
    reconciliation = baseline_reconciliation
    reconciliation_artifact: ProviderArtifact | None = None
    if run_c:
        reconciliation_prompt = build_reconciliation_prompt(
            deterministic,
            vlm_artifact,
            source_metadata=_safe_source_metadata(source_record),
            declarative_mappings=_declarative_mapping(source_record),
            known_conflicts=reconciliation.unresolved_conflicts,
            candidate_vocabularies={"canonical_object": config.known_object_vocabulary},
            approved_prior_labels=_approved_field_safe_priors(source_record),
        )
        provider = text_provider or MockJSONProvider(
            {"C_text_reconciliation": reconciliation.to_dict()},
            model_identity="mock-text-reconciler-v1",
            namespace=TEXT_RECONCILIATION_NAMESPACE,
        )
        execution = _call_provider_cached(
            provider,
            cache=cache,
            cache_kind="text",
            stage="C_text_reconciliation",
            prompt=reconciliation_prompt,
            prompt_version=RECONCILIATION_PROMPT_VERSION,
            schema_version=RECONCILIATION_SCHEMA_VERSION,
            image_path=resolved_image,
            payload=None,
            max_tokens=config.reconciliation_max_tokens,
            shared_cache=config.shared_cache,
            require_cache_hit=config.require_shared_bc_cache,
        )
        reconciliation_artifact = execution.artifact
        if reconciliation_artifact.parsed_output is not None:
            try:
                provider_reconciliation = parse_reconciliation_response(reconciliation_artifact.parsed_output)
                reconciliation = _merge_reconciliation_for_fusion(
                    baseline_reconciliation,
                    provider_reconciliation,
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                # The deterministic reconciliation remains visible and the
                # failed provider response remains in the stage artifact.
                pass
        ledger.append(_ledger_row("C_text_reconciliation", execution))

    reconciliation = _apply_color_role_policy(reconciliation, pixel_evidence)

    provisional_semantics = _assemble_semantics(deterministic, pixel_evidence, reconciliation, pack_context)
    disputed_claims, verifier_skips = _disputed_claims(reconciliation, provisional_semantics, deterministic)
    post_signals = _post_reconciliation_signals(
        initial_signals,
        deterministic,
        vlm_artifact,
        reconciliation,
        provisional_semantics,
        verifier_eligible_claim_count=len(disputed_claims),
    )
    final_route = decide_adaptive_routing(
        post_signals,
        verifier_risk_threshold=config.verifier_risk_threshold,
        novelty_threshold=config.novelty_threshold,
    )
    run_d = config.mode == "C" and bool(disputed_claims)
    if config.mode == "adaptive":
        run_d = final_route.run_stage_d and bool(disputed_claims)

    verifier_artifact: ProviderArtifact | None = None
    verifier_raw_output: dict[str, Any] | None = None
    verifier_normalized: ParsedVerifierResponse | None = None
    verifier_effects: VerifierDecisionSummary | None = None
    verifier_independence: str | None = None
    if run_d:
        verifier_prompt = build_independent_verifier_prompt(disputed_claims)
        provider = verifier_provider or MockJSONProvider(
            {"D_independent_verifier": _default_verifier_response(disputed_claims)},
            model_identity="mock-independent-verifier-v1",
            namespace=INDEPENDENT_VERIFIER_NAMESPACE,
        )
        execution = _call_provider_cached(
            provider,
            cache=cache,
            cache_kind="verifier",
            stage="D_independent_verifier",
            prompt=verifier_prompt,
            prompt_version=INDEPENDENT_VERIFIER_PROMPT_VERSION,
            schema_version=VERIFIER_SCHEMA_VERSION,
            image_path=resolved_image,
            payload={"disputed_claims": disputed_claims},
            max_tokens=config.verifier_max_tokens,
            shared_cache=config.shared_cache,
        )
        verifier_artifact = execution.artifact
        verifier_raw_output = verifier_artifact.parsed_output
        verifier_normalized = parse_verifier_response(verifier_raw_output, known_claims=disputed_claims)
        verifier_effects = derive_verifier_effects(verifier_normalized, known_claims=disputed_claims)
        upstream_identities = [
            artifact.model_identity
            for artifact in (vlm_artifact, reconciliation_artifact)
            if artifact is not None and artifact.model_identity
        ]
        if upstream_identities:
            verifier_independence = classify_verifier_independence(
                verifier_artifact.model_identity,
                upstream_identities,
            )
        ledger.append(_ledger_row("D_independent_verifier", execution))

    reconciliation = _apply_verifier_claim_effects(
        reconciliation,
        disputed_claims=disputed_claims,
        verifier_effects=verifier_effects,
    )
    effective_conflicts = _effective_conflicts(reconciliation, verifier_effects)
    semantics = _assemble_semantics(deterministic, pixel_evidence, reconciliation, pack_context)
    description_candidates = _description_candidates(reconciliation)
    description_facts = {
        **semantics,
        "object_alternatives": list(semantics.get("canonical_object_alternatives") or ()),
        "object_claim_candidates": (
            [candidate.value for candidate in vlm_artifact.proposal.object_candidates]
            if vlm_artifact and vlm_artifact.proposal
            else []
        ),
    }
    description = choose_or_regenerate_description(description_candidates, description_facts)
    semantics["description"] = description["description"]
    field_risks = _estimate_all_risks(
        semantics,
        deterministic=deterministic,
        vlm=vlm_artifact,
        reconciliation=reconciliation,
        verifier=verifier_normalized.to_dict() if verifier_normalized else None,
        verifier_effects=verifier_effects,
        verifier_independence=verifier_independence,
        effective_conflicts=effective_conflicts,
        pixel_evidence=pixel_evidence,
        config=config,
        description_valid=not description["claims_rejected"],
        description_uses_alternative_interpretation=any(
            "unsupported_object:" in str(rejection) for rejection in description.get("claims_rejected") or ()
        ),
    )
    record_risk = summarize_record_risk(
        field_risks,
        contradiction_count=len(effective_conflicts),
        policy=config.risk_policy,
    )
    quality_input = {
        **record_risk.to_dict(),
        "risk_model_version": "label_risk_v1",
        "fields": {name: risk.to_dict() for name, risk in field_risks.items()},
        "unresolved_conflicts": effective_conflicts,
    }
    training_quality = normalize_training_quality(
        quality_input,
        record={**source_record, **semantics, **_flatten_semantic_values(semantics)},
    )
    inference_path = "A" + ("+B+C" if run_b and run_c else "+C" if run_c else "") + ("+D" if run_d else "")
    route = _combined_route(initial_route, final_route, run_b=run_b, run_c=run_c, run_d=run_d)
    provider_accounting = _provider_accounting(ledger, config)

    output = {
        "schema_version": LABEL_V4_RECORD_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "sprite_id": sprite_id,
        "source_id": str(source_record.get("source_id", "")),
        "pack_id": str(
            source_record.get("pack_id") or source_record.get("pack") or source_record.get("source_name") or ""
        ),
        "artist": str(
            source_record.get("artist") or source_record.get("author") or source_record.get("sub_artist") or ""
        ),
        "image_path": str(resolved_image),
        "image_hash": pixel_evidence["image_hash"],
        "deterministic_evidence": {
            "filename": deterministic.to_dict(),
            "pixels": pixel_evidence,
            "pack_context": pack_context,
        },
        "vlm_proposal": vlm_artifact.to_dict() if vlm_artifact else None,
        "reconciliation": reconciliation.to_dict(),
        "reconciliation_provider_artifact": (reconciliation_artifact.to_dict() if reconciliation_artifact else None),
        "verification": {
            "disputed_claims": disputed_claims,
            "claims_not_routed": verifier_skips,
            "raw_output": verifier_raw_output,
            "output": verifier_normalized.to_dict() if verifier_normalized else None,
            "decision_effects": verifier_effects.to_dict() if verifier_effects else None,
            "artifact": verifier_artifact.to_dict() if verifier_artifact else None,
            "independent_prompt": True if run_d else None,
            "independence": verifier_independence,
            "independent_cache_namespace": INDEPENDENT_VERIFIER_NAMESPACE if run_d else None,
        },
        "semantics": semantics,
        "description_validation": description,
        "field_risks": {name: risk.to_dict() for name, risk in field_risks.items()},
        "record_risk": record_risk.to_dict(),
        "label_quality": training_quality,
        "unresolved_conflicts": effective_conflicts,
        "open_set_terms": list(reconciliation.open_set_terms),
        "routing": route.to_dict(),
        "routing_profile": config.routing_profile,
        "stage_a_field_coverage": stage_a_coverage,
        "inference_path": inference_path,
        "stage_ledger": ledger,
        "provider_accounting": provider_accounting,
        "logical_stage_count": provider_accounting["logical_stage_count"],
        "provider_call_count": provider_accounting["new_provider_calls"],
        "new_provider_calls": provider_accounting["new_provider_calls"],
        "actual_http_attempts": provider_accounting["actual_http_attempts"],
        "cache_hit_count": provider_accounting["cache_hits"],
        "shared_cache_hits": provider_accounting["shared_cache_hits"],
        "total_tokens": provider_accounting["total_tokens"],
        "estimated_provider_cost": provider_accounting["estimated_provider_cost"],
        "legacy_evidence_used": False,
        "source_record_hash": _stable_hash(source_record),
    }
    output["stage_outcomes"] = _terminal_stage_outcomes(ledger, run_b=run_b, run_c=run_c, run_d=run_d)
    output["record_status"] = _terminal_record_status(output["stage_outcomes"])
    output["training_channels"] = _separate_training_channels(output, semantics, training_quality, description)
    # Compatibility fields are projections of v4 semantics, not mutated v3
    # artifacts.  They make existing manifest adapters able to carry quality.
    output.update(
        {
            "domain": semantics.get("domain"),
            "category": semantics.get("category"),
            "canonical_object": semantics.get("canonical_object"),
            "canonical_object_alternatives": semantics.get("canonical_object_alternatives"),
            "visual_form": semantics.get("visual_form"),
            "surface_alias": semantics.get("surface_alias"),
            "role": semantics.get("role"),
            "explicit_material": semantics.get("explicit_material"),
            "description": semantics.get("description"),
            "raw_visual_color_roles": semantics.get("raw_visual_color_roles"),
        }
    )
    return output


def infer_pack_context(record: Mapping[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(record.get(key, "")).lower() for key in ("pack", "pack_id", "pack_name", "source_id", "source_name")
    )
    if any(token in text for token in ("jewelry", "jewellery")):
        return {"category": "jewelry", "role": "wearable_equipment", "source": "pack_metadata"}
    if any(token in text for token in ("gem", "jewel", "crystal")):
        return {"category": "gem", "role": "resource", "source": "pack_metadata"}
    if "food" in text:
        return {"category": "food", "role": "consumable", "source": "pack_metadata"}
    if "key" in text:
        return {"category": "key", "role": "quest_item", "source": "pack_metadata"}
    if any(token in text for token in ("armor", "armour", "armory")):
        return {"category": "armor", "role": "wearable_equipment", "source": "pack_metadata"}
    if "weapon" in text:
        return {"category": "weapon", "role": "weapon", "source": "pack_metadata"}
    if "plant" in text:
        return {"category": "plant", "role": "resource", "source": "pack_metadata"}
    if "potion" in text:
        return {"category": "potion", "role": "consumable", "source": "pack_metadata"}
    return {"category": "unknown", "role": "unknown", "source": "unknown_pack_profile"}


def build_independent_verifier_prompt(disputed_claims: Sequence[Mapping[str, Any]]) -> str:
    return (
        "Inspect the sprite pixels with an independent prompt. Evaluate only each directional claimed_value below; "
        "do not infer from a prior model narrative and do not emit self-confidence. Return JSON with claim_results, "
        "each containing claim_id, verdict (supported, contradicted, or unresolved), and visible_support, plus "
        "unsupported_fields. Use unresolved when isolated pixels cannot adjudicate the concrete claim. "
        "Disputed claims: "
        + json.dumps(
            [dict(claim) for claim in disputed_claims],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _resolve_image_path(record: Mapping[str, Any], supplied: str | Path | None) -> Path:
    candidates = [
        supplied,
        record.get("final_png_path"),
        record.get("image_path"),
        record.get("source_image"),
        record.get("rgba_path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file():
            return path
    raise FileNotFoundError(f"no image found for sprite {record.get('sprite_id', '')!r}")


def _initial_routing_signals(
    record: Mapping[str, Any], deterministic: FilenameParseResult, pixels: Mapping[str, Any]
) -> AdaptiveRoutingSignals:
    context = infer_pack_context(record)
    shape = pixels.get("shape") if isinstance(pixels.get("shape"), Mapping) else {}
    structure_weak = not bool(shape.get("structure") or shape.get("parts"))
    return AdaptiveRoutingSignals(
        canonical_object_missing=not bool(deterministic.canonical_object),
        filename_generic=deterministic.generic,
        pack_heterogeneous=bool(record.get("pack_heterogeneous")),
        deterministic_conflicts=len(deterministic.explicit_material_candidates) > 1,
        shape_weak=structure_weak,
        description_missing=not bool(record.get("description") or record.get("short_description")),
        object_open_set=bool(deterministic.open_set_tokens and not deterministic.canonical_object),
        source_profile_unknown=context.get("source") == "unknown_pack_profile",
    )


def _post_reconciliation_signals(
    initial: AdaptiveRoutingSignals,
    deterministic: FilenameParseResult,
    vlm: VLMProposalArtifact | None,
    reconciliation: ReconciliationResult,
    semantics: Mapping[str, Any],
    *,
    verifier_eligible_claim_count: int | None = None,
) -> AdaptiveRoutingSignals:
    conflict_codes = {str(conflict.get("code")) for conflict in reconciliation.unresolved_conflicts}
    category = str(semantics.get("category", "unknown"))
    object_name = str(semantics.get("canonical_object", ""))
    inconsistent = not _object_category_compatible(object_name, category)
    raw_risk = 0.12
    raw_risk += 0.18 if initial.canonical_object_missing else 0.0
    raw_risk += min(0.45, 0.12 * len(conflict_codes))
    raw_risk += 0.12 if reconciliation.open_set_terms else 0.0
    return AdaptiveRoutingSignals(
        **{
            name: getattr(initial, name)
            for name in (
                "canonical_object_missing",
                "filename_generic",
                "pack_heterogeneous",
                "deterministic_conflicts",
                "shape_weak",
                "description_missing",
                "object_open_set",
                "source_profile_unknown",
            )
        },
        critical_field_risk_upper=min(1.0, raw_risk),
        vlm_deterministic_disagreement=any("vlm" in code and "disagreement" in code for code in conflict_codes),
        object_category_inconsistent=inconsistent,
        filename_visual_color_conflict="filename_visual_color_conflict" in conflict_codes,
        material_visual_only=bool(
            vlm and vlm.proposal and vlm.proposal.material_visual_cues and not deterministic.explicit_material
        ),
        open_set_novelty=0.8 if reconciliation.open_set_terms else 0.0,
        variant_family_inconsistent=bool(semantics.get("variant_family_inconsistent")),
        policy_abstains_object_identity=not bool(semantics.get("canonical_object")),
        source_category_authoritative=bool(deterministic.category and deterministic.category != "unknown"),
        verifier_eligible_claim_count=verifier_eligible_claim_count,
    )


def _visual_promotion_evidence(artifact: VLMProposalArtifact | None) -> dict[str, Any]:
    proposal = artifact.proposal if artifact and artifact.available else None
    if proposal is None:
        return {}
    concrete = [candidate for candidate in proposal.object_candidates if candidate.value not in proposal.visual_form]
    strong = (
        len(concrete) == 1
        and len(concrete[0].visual_support) >= 2
        and not proposal.alternative_interpretations
        and not proposal.uncertainties
    )
    return {
        "strong_visual_identity": strong,
        "visual_identity_margin": 1.0 if len(concrete) == 1 else 0.0,
        "minimum_visual_identity_margin": 0.35,
        "independent_evidence_groups": 1 if concrete else 0,
    }


def _merge_reconciliation_for_fusion(
    baseline: ReconciliationResult,
    provider: ReconciliationResult,
) -> ReconciliationResult:
    """Keep provider raw reasoning while reapplying deterministic policy gates."""

    fields = dict(provider.field_proposals)
    policy_fields = {
        "canonical_object",
        "surface_alias",
        "category",
        "role",
        "explicit_material",
        "visual_form",
    }
    for field_name in policy_fields:
        safe = baseline.field_proposals.get(field_name)
        raw = provider.field_proposals.get(field_name)
        if safe is None:
            if raw is not None and field_name in {"canonical_object", "surface_alias", "explicit_material"}:
                fields[field_name] = FieldProposal(
                    raw_open_vocabulary_value=raw.raw_open_vocabulary_value,
                    normalized_controlled_value=None,
                    alternatives=raw.alternatives,
                    support=(),
                    conflicts=tuple(dict.fromkeys((*raw.conflicts, f"{field_name}_policy_abstained"))),
                    decision="abstained",
                )
            continue
        fields[field_name] = FieldProposal(
            raw_open_vocabulary_value=(
                raw.raw_open_vocabulary_value if raw is not None else safe.raw_open_vocabulary_value
            ),
            normalized_controlled_value=safe.normalized_controlled_value,
            alternatives=tuple(
                dict.fromkeys(
                    [
                        *safe.alternatives,
                        *(raw.alternatives if raw is not None else ()),
                    ]
                )
            ),
            support=safe.support,
            conflicts=tuple(
                dict.fromkeys(
                    [
                        *safe.conflicts,
                        *(raw.conflicts if raw is not None else ()),
                    ]
                )
            ),
            decision=safe.decision,
            promotion_basis=safe.promotion_basis,
        )

    accepted, rejected, unresolved = _merge_terminal_claims(baseline, provider)
    return ReconciliationResult(
        field_proposals=fields,
        taxonomy_mapping_actions=tuple(
            _dedupe_mapping_rows([*baseline.taxonomy_mapping_actions, *provider.taxonomy_mapping_actions])
        ),
        unresolved_conflicts=tuple(
            _dedupe_mapping_rows([*baseline.unresolved_conflicts, *provider.unresolved_conflicts])
        ),
        open_set_terms=tuple(dict.fromkeys([*baseline.open_set_terms, *provider.open_set_terms])),
        claims_accepted=accepted,
        claims_rejected=rejected,
        claims_unresolved=unresolved,
    )


def _merge_terminal_claims(
    baseline: ReconciliationResult,
    provider: ReconciliationResult,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    terminal: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    # Baseline claims have already passed deterministic policy.  Provider
    # bookkeeping may add new claims, but it must never upgrade a claim that
    # policy rejected or left unresolved.
    groups = (
        ("accepted", baseline.claims_accepted, True),
        ("rejected", baseline.claims_rejected, True),
        ("unresolved", baseline.claims_unresolved, True),
        ("rejected", provider.claims_rejected, False),
        ("unresolved", provider.claims_unresolved, False),
        ("accepted", provider.claims_accepted, False),
    )
    for state, claims, policy_authoritative in groups:
        for raw_claim in claims:
            claim = dict(raw_claim)
            key = _terminal_claim_key(claim)
            if policy_authoritative or key not in terminal:
                terminal[key] = (state, claim)
    return tuple(
        tuple(claim for state, claim in terminal.values() if state == wanted)
        for wanted in ("accepted", "rejected", "unresolved")
    )  # type: ignore[return-value]


def _terminal_claim_key(claim: Mapping[str, Any]) -> tuple[str, str]:
    field_name = normalize_semantic_term(claim.get("field")) or "claim"
    value = claim.get("value", claim.get("values", claim.get("claim")))
    return field_name, json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _dedupe_mapping_rows(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        row = dict(value)
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def _apply_color_role_policy(
    reconciliation: ReconciliationResult,
    pixels: Mapping[str, Any],
) -> ReconciliationResult:
    fields = dict(reconciliation.field_proposals)
    raw_field = fields.get("raw_visual_color_roles")
    raw_roles = raw_field.raw_open_vocabulary_value if raw_field is not None else None
    if not isinstance(raw_roles, Mapping):
        color_field = fields.get("color_roles")
        raw_roles = color_field.raw_open_vocabulary_value if color_field is not None else None
    if not isinstance(raw_roles, Mapping):
        return reconciliation
    normalized = normalize_visual_color_roles(raw_roles, pixels.get("palette_colors") or ())
    fields["raw_visual_color_roles"] = FieldProposal(
        raw_open_vocabulary_value={role: list(values) for role, values in normalized.raw_visual_color_roles.items()},
        normalized_controlled_value=None,
        support=("vlm_visual",),
    )
    existing_color = fields.get("color_roles")
    fields["color_roles"] = FieldProposal(
        raw_open_vocabulary_value=dict(raw_roles),
        normalized_controlled_value=normalized.color_roles.to_dict(),
        support=("deterministic_palette", "vlm_visual"),
        conflicts=tuple(
            dict.fromkeys(
                [
                    *(existing_color.conflicts if existing_color is not None else ()),
                    *(str(conflict.get("code", "")) for conflict in normalized.conflicts),
                ]
            )
        ),
    )
    claims_unresolved = [*reconciliation.claims_unresolved]
    for conflict in normalized.conflicts:
        claims_unresolved.append(
            {
                "field": "color",
                "value": conflict.get("raw_visual_color"),
                "reason": "color_role_outside_palette",
            }
        )
    return ReconciliationResult(
        field_proposals=fields,
        taxonomy_mapping_actions=reconciliation.taxonomy_mapping_actions,
        unresolved_conflicts=tuple(_dedupe_mapping_rows([*reconciliation.unresolved_conflicts, *normalized.conflicts])),
        open_set_terms=reconciliation.open_set_terms,
        claims_accepted=reconciliation.claims_accepted,
        claims_rejected=reconciliation.claims_rejected,
        claims_unresolved=tuple(_dedupe_mapping_rows(claims_unresolved)),
    )


def _assemble_semantics(
    deterministic: FilenameParseResult,
    pixels: Mapping[str, Any],
    reconciliation: ReconciliationResult,
    pack_context: Mapping[str, Any],
) -> dict[str, Any]:
    proposals = reconciliation.field_proposals

    def value(name: str, fallback: Any = None) -> Any:
        proposal = proposals.get(name)
        if proposal is not None and proposal.decision != "accepted":
            return None
        return proposal.value if proposal is not None and _has_value(proposal.value) else fallback

    category = str(value("category", deterministic.category or pack_context.get("category") or "unknown"))
    if category not in CATEGORY_VALUES:
        category = "unknown"
    role = str(value("role", deterministic.role or pack_context.get("role") or "unknown"))
    if role not in ROLE_VALUES:
        role = "unknown"
    domain = _domain_for_category(category)
    object_name = value("canonical_object", deterministic.canonical_object)
    alias = value("surface_alias", deterministic.surface_alias)
    visual_form = value("visual_form", [])
    visual_form = _dedupe_strings(visual_form if isinstance(visual_form, (list, tuple)) else (visual_form,))
    canonical_proposal = proposals.get("canonical_object")
    canonical_alternatives = list(canonical_proposal.alternatives) if canonical_proposal is not None else []

    pixel_shape = pixels.get("shape") if isinstance(pixels.get("shape"), Mapping) else {}
    proposed_shape = value("shape", {})
    proposed_shape = proposed_shape if isinstance(proposed_shape, Mapping) else {}
    shape: dict[str, list[str]] = {}
    for axis in ("silhouette", "aspect", "orientation", "structure", "edge_profile", "parts"):
        shape[axis] = _dedupe_strings([*(proposed_shape.get(axis) or ()), *(pixel_shape.get(axis) or ())])

    color_roles = value("color_roles", {})
    color_roles = color_roles if isinstance(color_roles, Mapping) else {}
    raw_visual_color_roles = value("raw_visual_color_roles", {})
    raw_visual_color_roles = raw_visual_color_roles if isinstance(raw_visual_color_roles, Mapping) else {}
    colors = {
        "palette_colors": list(pixels.get("palette_colors") or ()),
        "primary_colors": _list_field(color_roles, "primary_colors", "primary"),
        "secondary_colors": _list_field(color_roles, "secondary_colors", "secondary"),
        "outline_colors": _list_field(color_roles, "outline_colors", "outline"),
        "shadow_colors": _list_field(color_roles, "shadow_colors", "shadow"),
        "highlight_colors": _list_field(color_roles, "highlight_colors", "highlight"),
        "filename_color_hints": list(deterministic.filename_color_hints),
    }
    if not colors["primary_colors"] and colors["palette_colors"]:
        colors["primary_colors"] = colors["palette_colors"][:1]
    styles = _dedupe_strings(["pixel_art", *deterministic.style_modifiers])
    return {
        "domain": domain,
        "category": category,
        "canonical_object": object_name,
        "canonical_object_alternatives": canonical_alternatives,
        "visual_form": visual_form,
        "surface_alias": alias,
        "role": role,
        "explicit_material": value("explicit_material", deterministic.explicit_material),
        "visual_material_cue": value("visual_material_cue", []),
        "shape": shape,
        "colors": colors,
        "raw_visual_color_roles": dict(raw_visual_color_roles),
        "size_hint": deterministic.size_hint,
        "condition": list(deterministic.condition_hints),
        "style": styles,
        "description": "",
    }


def _estimate_all_risks(
    semantics: Mapping[str, Any],
    *,
    deterministic: FilenameParseResult,
    vlm: VLMProposalArtifact | None,
    reconciliation: ReconciliationResult,
    verifier: Mapping[str, Any] | None,
    verifier_effects: VerifierDecisionSummary | None,
    verifier_independence: str | None,
    effective_conflicts: Sequence[Mapping[str, Any]],
    pixel_evidence: Mapping[str, Any],
    config: LabelV4PipelineConfig,
    description_valid: bool,
    description_uses_alternative_interpretation: bool,
) -> dict[str, FieldRisk]:
    values = _flatten_semantic_values(semantics)
    conflicts_by_field: dict[str, list[Mapping[str, Any]]] = {}
    for conflict in effective_conflicts:
        field_name = str(conflict.get("field", ""))
        conflicts_by_field.setdefault(field_name, []).append(conflict)
    verified_fields = {
        effect.field for effect in (verifier_effects.claim_effects if verifier_effects else ()) if effect.field
    }
    verifier_no_change = bool(verifier_effects and verifier_effects.decision_change == "no_decision_change")
    canonical_proposal = reconciliation.field_proposals.get("canonical_object")
    canonical_abstained = bool(canonical_proposal and canonical_proposal.decision == "abstained")
    generic_promoted = bool(
        semantics.get("canonical_object")
        and is_generic_visual_form(semantics.get("canonical_object"))
        and not (canonical_proposal and canonical_proposal.promotion_basis)
    )
    category_source = str(deterministic.field_sources.get("category", ""))
    alias_source = str(deterministic.surface_alias_source or "")
    pack_only_category = category_source in {"pack_name", "pack_context"}
    source_visual_ambiguity = bool(
        deterministic.category != "unknown"
        and canonical_abstained
        and vlm
        and vlm.proposal
        and vlm.proposal.object_candidates
    )
    invalid_taxonomy = any(
        str(conflict.get("code", "")) == "invalid_taxonomy_provider_output"
        for conflict in reconciliation.unresolved_conflicts
    )
    color_outside_palette = any(
        str(conflict.get("code", "")) == "color_role_outside_palette"
        for conflict in reconciliation.unresolved_conflicts
    )
    supplied: dict[str, FieldRisk] = {}
    for field_name in SEMANTIC_FIELDS:
        present = _present(values.get(field_name))
        deterministic_support = _deterministic_supports(field_name, deterministic, pixel_evidence)
        visual_support = _vlm_supports(field_name, vlm)
        field_conflicts = conflicts_by_field.get(field_name, [])
        if field_name in {"primary_colors", "secondary_colors", "filename_color_hints"}:
            field_conflicts += conflicts_by_field.get("color", [])
        field_verifier_agreement = field_name in verified_fields and _verifier_agreement(verifier, field_name)
        verifier_is_independent_model = (
            field_name in verified_fields and verifier_independence == "different_model_independent_prompt"
        )
        features = {
            "deterministic_evidence_strong": deterministic_support,
            "independent_dependency_groups": (
                int(deterministic_support) + int(visual_support) + int(verifier_is_independent_model)
            ),
            "vlm_verifier_agreement": field_verifier_agreement,
            "filename_vlm_agreement": deterministic_support and visual_support and not field_conflicts,
            "taxonomy_compatible": field_name not in CRITICAL_FIELDS
            or _object_category_compatible(
                str(semantics.get("canonical_object") or ""), str(semantics.get("category") or "unknown")
            ),
            "open_set_novelty": field_name == "canonical_object" and bool(reconciliation.open_set_terms),
            "image_quality_low": bool(pixel_evidence.get("quality_signals")),
            "provenance_incomplete": not deterministic.token_provenance,
            "description_claim_invalid": field_name == "description" and not description_valid,
            "material_not_explicit": field_name == "visual_material_cue" and not deterministic.explicit_material,
            "color_role_inconsistent": any(
                str(conflict.get("code")) == "filename_visual_color_conflict" for conflict in field_conflicts
            ),
            "unresolved_conflict": bool(field_conflicts),
            "contradiction_count": len(field_conflicts),
            "same_model_verifier": (
                field_name in verified_fields and verifier_independence == "same_model_independent_prompt"
            ),
            "verifier_no_decision_change": field_name in verified_fields and verifier_no_change,
            "explicit_abstention": field_name == "canonical_object" and canonical_abstained,
            "generic_visual_form_promoted": field_name == "canonical_object" and generic_promoted,
            "pack_only_category": field_name == "category" and pack_only_category,
            "source_category_visual_ambiguity": (
                field_name in {"canonical_object", "category"} and source_visual_ambiguity
            ),
            "surface_alias_from_pack_metadata": (
                field_name == "surface_alias" and alias_source in {"sheet_name", "pack_name", "pack_context"}
            ),
            "description_uses_alternative_interpretation": (
                field_name == "description" and description_uses_alternative_interpretation
            ),
            "invalid_taxonomy_provider_output": (field_name in {"domain", "category", "role"} and invalid_taxonomy),
            "color_role_outside_palette": (
                field_name
                in {"primary_colors", "secondary_colors", "outline_colors", "shadow_colors", "highlight_colors"}
                and color_outside_palette
            ),
        }
        supplied[field_name] = estimate_field_risk(
            field_name,
            value_present=present,
            risk_features=features,
            calibration=config.calibration_by_field.get(field_name),
            policy=config.risk_policy,
        )
    return ensure_all_field_risks(values, supplied)


def _call_provider_cached(
    provider: Any,
    *,
    cache: LabelV4Cache | None,
    cache_kind: Literal["blind", "text", "verifier"],
    stage: str,
    prompt: str,
    prompt_version: str,
    schema_version: str,
    image_path: Path,
    payload: Mapping[str, Any] | None,
    max_tokens: int = 1024,
    shared_cache: bool = False,
    require_cache_hit: bool = False,
) -> _ProviderExecution:
    started = time.perf_counter()
    model_identity = str(getattr(provider, "model_identity", type(provider).__name__))
    request = {
        "stage": stage,
        "prompt_version": prompt_version,
        "payload": payload,
        "max_tokens": int(max_tokens),
        "provider_request_policy": str(getattr(provider, "request_policy_version", "unspecified")),
    }
    identity_kwargs = {
        "stage": stage,
        "image": image_path,
        "model_identity": model_identity,
        "prompt_version": prompt_version,
        "prompt": prompt,
        "schema_version": schema_version,
        "request": request,
        "provider": type(provider).__name__,
    }
    if cache_kind == "blind":
        identity = blind_proposal_identity(**identity_kwargs)
    elif cache_kind == "verifier":
        identity = verifier_identity(**identity_kwargs)
    else:
        identity = make_cache_identity(namespace=TEXT_RECONCILIATION_NAMESPACE, **identity_kwargs)

    def compute() -> dict[str, Any]:
        artifact = provider.call_json(
            stage=stage,
            prompt=prompt,
            prompt_version=prompt_version,
            image_path=image_path if cache_kind != "text" else None,
            payload=payload,
            max_tokens=int(max_tokens),
        )
        if not artifact.ok:
            raise _ProviderFailure(artifact)
        return artifact.to_dict()

    if cache is None:
        if require_cache_hit:
            raise RequiredSharedCacheMiss(stage, identity.key)
        try:
            raw = compute()
        except _ProviderFailure as exc:
            artifact = exc.artifact
        else:
            artifact = _provider_artifact_from_dict(raw)
        return _ProviderExecution(
            artifact=artifact,
            cache_hit=False,
            cache_key=identity.key,
            shared_cache_hit=False,
            execution_latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    if require_cache_hit:
        raw = cache.get(identity)
        if raw is None:
            raise RequiredSharedCacheMiss(stage, identity.key)
        return _ProviderExecution(
            artifact=_provider_artifact_from_dict(raw),
            cache_hit=True,
            cache_key=identity.key,
            shared_cache_hit=bool(shared_cache),
            execution_latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    try:
        raw, hit = cache.get_or_compute(identity, compute)
    except _ProviderFailure as exc:
        return _ProviderExecution(
            artifact=exc.artifact,
            cache_hit=False,
            cache_key=identity.key,
            shared_cache_hit=False,
            execution_latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return _ProviderExecution(
        artifact=_provider_artifact_from_dict(raw),
        cache_hit=hit,
        cache_key=identity.key,
        shared_cache_hit=bool(shared_cache and hit),
        execution_latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def _provider_artifact_from_dict(data: Mapping[str, Any]) -> ProviderArtifact:
    return ProviderArtifact(
        stage=str(data.get("stage", "")),
        raw_output=str(data.get("raw_output", "")),
        parsed_output=dict(data["parsed_output"]) if isinstance(data.get("parsed_output"), Mapping) else None,
        model_identity=str(data.get("model_identity", "")),
        request_hash=str(data.get("request_hash", "")),
        image_hash=str(data.get("image_hash", "")),
        prompt_version=str(data.get("prompt_version", "")),
        prompt_hash=str(data.get("prompt_hash", "")),
        latency_ms=float(data.get("latency_ms", 0.0)),
        http_attempts=max(0, int(data.get("http_attempts", 0) or 0)),
        request_policy_version=str(data.get("request_policy_version", "")),
        token_usage={str(key): int(value) for key, value in dict(data.get("token_usage") or {}).items()},
        cache_namespace=str(data.get("cache_namespace", "")),
        failure_diagnostics=dict(data.get("failure_diagnostics") or {}),
        schema_version=str(data.get("schema_version", "label_provider_artifact_v1.0")),
    )


def _default_blind_mock(pixels: Mapping[str, Any]) -> MockJSONProvider:
    shape = pixels.get("shape") if isinstance(pixels.get("shape"), Mapping) else {}
    colors = list(pixels.get("palette_colors") or ())
    response = {
        "canonical_object_candidates": [],
        "category_candidates": [],
        "surface_alias_candidates": [],
        "role_candidates": [],
        "shape": _controlled_mock_shape(shape),
        "color_roles": {
            "primary": colors[:1],
            "secondary": colors[1:2],
            "outline": ["black"] if "black" in colors else [],
            "shadow": [value for value in colors if value.startswith("dark_")][:1],
            "highlight": [value for value in colors if value.startswith("light_")][:1],
        },
        "material_visual_cues": [],
        "description_candidates": [],
        "uncertainties": ["identity unsupported by deterministic mock"],
        "alternative_interpretations": [],
        "unsupported_fields": ["canonical_object", "category", "role", "explicit_material"],
    }
    return MockJSONProvider(
        {"B_blind_vlm_proposal": response},
        model_identity="mock-blind-pixel-observer-v1",
        namespace="blind_vlm_proposal_v4",
    )


def _controlled_mock_shape(shape: Mapping[str, Any]) -> dict[str, list[str]]:
    silhouette_map = {
        "round_or_compact": "compact",
        "elongated_horizontal": "elongated",
        "elongated_vertical": "elongated",
        "open_or_fragmented": "irregular",
        "irregular_compact": "irregular",
    }
    orientation_map = {"front_facing_or_unknown": "unknown"}
    result = {
        name: list(shape.get(name) or ())
        for name in ("silhouette", "aspect", "orientation", "structure", "edge_profile", "parts")
    }
    result["silhouette"] = [silhouette_map.get(value, value) for value in result["silhouette"]]
    result["orientation"] = [orientation_map.get(value, value) for value in result["orientation"]]
    return result


def _default_verifier_response(disputed_claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": VERIFIER_SCHEMA_VERSION,
        "claim_results": [
            {
                "claim_id": str(claim.get("claim_id", "")),
                "verdict": "unresolved",
                "visible_support": [],
            }
            for claim in disputed_claims
        ],
        "unsupported_fields": [],
    }


def _disputed_claims(
    reconciliation: ReconciliationResult,
    semantics: Mapping[str, Any],
    deterministic: FilenameParseResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return only directional, visible, decision-relevant verifier claims."""

    claims: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, conflict in enumerate(reconciliation.unresolved_conflicts):
        field_name = str(conflict.get("field", ""))
        code = str(conflict.get("code", ""))
        conflict_id = str(conflict.get("conflict_id") or f"conflict-{index:03d}")
        values = list(conflict.get("values") or ())
        skip_reason = ""
        if code in {
            "invalid_taxonomy_provider_output",
            "color_role_outside_palette",
            "filename_visual_color_conflict",
            "material_inferred_not_explicit",
            "surface_alias_from_pack_metadata",
            "generic_visual_form_promoted",
        }:
            skip_reason = "deterministic_policy_already_decides_claim"
        elif (
            field_name in {"category", "role"}
            and getattr(deterministic, field_name) != "unknown"
            and not semantics.get("canonical_object")
        ):
            skip_reason = "source_category_authoritative_object_identity_abstained"
        elif field_name == "canonical_object" and not semantics.get("canonical_object"):
            skip_reason = "canonical_object_already_abstained"
        elif field_name not in {"canonical_object", "category", "role"}:
            skip_reason = "claim_not_visually_adjudicable"
        elif len(values) < 2:
            skip_reason = "claim_not_concrete"
        if skip_reason:
            skipped.append(
                {
                    "conflict_id": conflict_id,
                    "field": field_name,
                    "dispute_code": code,
                    "reason": skip_reason,
                }
            )
            continue
        claimed_value = semantics.get(field_name)
        alternatives = [value for value in values if value != claimed_value]
        if not _has_value(claimed_value) or not alternatives:
            skipped.append(
                {
                    "conflict_id": conflict_id,
                    "field": field_name,
                    "dispute_code": code,
                    "reason": "claim_not_directional",
                }
            )
            continue
        claims.append(
            {
                "claim_id": f"verify-{conflict_id}",
                "conflict_id": conflict_id,
                "field": field_name,
                "claimed_value": claimed_value,
                "competing_values": alternatives,
                "dispute_code": code,
                "visible_adjudication": True,
                "resolve_on_supported": True,
            }
        )
    if semantics.get("visual_material_cue") and not semantics.get("explicit_material"):
        skipped.append(
            {
                "field": "visual_material_cue",
                "dispute_code": "material_inferred_not_explicit",
                "reason": "policy_already_abstains_explicit_material",
            }
        )
    return claims, skipped


def _apply_verifier_claim_effects(
    reconciliation: ReconciliationResult,
    *,
    disputed_claims: Sequence[Mapping[str, Any]],
    verifier_effects: VerifierDecisionSummary | None,
) -> ReconciliationResult:
    """Apply exact verifier rejections to normalized fields and claim state.

    Raw reconciliation/provider artifacts remain unchanged.  A contradicted
    routed claim cannot remain as an accepted final semantic merely because it
    was the provisional value before stage D.
    """

    if verifier_effects is None:
        return reconciliation
    claim_index = {
        str(claim.get("claim_id", "")): dict(claim) for claim in disputed_claims if str(claim.get("claim_id", ""))
    }
    rejected_claim_ids = {
        effect.claim_id for effect in verifier_effects.claim_effects if "claim_rejected" in effect.effects
    }
    if not rejected_claim_ids:
        return reconciliation

    fields = dict(reconciliation.field_proposals)
    accepted = [dict(claim) for claim in reconciliation.claims_accepted]
    rejected = [dict(claim) for claim in reconciliation.claims_rejected]
    unresolved = [dict(claim) for claim in reconciliation.claims_unresolved]
    for claim_id in rejected_claim_ids:
        claim = claim_index.get(claim_id)
        if claim is None:
            continue
        field_name = normalize_semantic_term(claim.get("field"))
        claimed_value = claim.get("claimed_value")
        target_key = _terminal_claim_key({"field": field_name, "value": claimed_value})
        accepted = [value for value in accepted if _terminal_claim_key(value) != target_key]
        unresolved = [value for value in unresolved if _terminal_claim_key(value) != target_key]
        rejected = [value for value in rejected if _terminal_claim_key(value) != target_key]
        rejected.append(
            {
                "field": field_name,
                "value": claimed_value,
                "reason": "verifier_contradicted",
                "claim_id": claim_id,
            }
        )

        proposal = fields.get(field_name)
        if proposal is not None and proposal.value == claimed_value:
            fields[field_name] = FieldProposal(
                raw_open_vocabulary_value=proposal.raw_open_vocabulary_value,
                normalized_controlled_value=None,
                alternatives=tuple(
                    dict.fromkeys(
                        [
                            *proposal.alternatives,
                            *(claim.get("competing_values") or ()),
                        ]
                    )
                ),
                support=proposal.support,
                conflicts=tuple(dict.fromkeys((*proposal.conflicts, "verifier_claim_rejected"))),
                decision="rejected",
                promotion_basis=proposal.promotion_basis,
            )

        # A surface alias is an identity-bearing projection.  Once the exact
        # canonical identity has been contradicted, retaining its alias would
        # let the rejected noun leak back into the generated description.
        if field_name == "canonical_object":
            alias_proposal = fields.get("surface_alias")
            alias_value = alias_proposal.value if alias_proposal is not None else None
            if alias_proposal is not None and _has_value(alias_value):
                alias_key = _terminal_claim_key({"field": "surface_alias", "value": alias_value})
                accepted = [value for value in accepted if _terminal_claim_key(value) != alias_key]
                rejected = [value for value in rejected if _terminal_claim_key(value) != alias_key]
                unresolved = [value for value in unresolved if _terminal_claim_key(value) != alias_key]
                unresolved.append(
                    {
                        "field": "surface_alias",
                        "value": alias_value,
                        "reason": "canonical_identity_rejected",
                        "claim_id": claim_id,
                    }
                )
                fields["surface_alias"] = FieldProposal(
                    raw_open_vocabulary_value=alias_proposal.raw_open_vocabulary_value,
                    normalized_controlled_value=None,
                    alternatives=alias_proposal.alternatives,
                    support=alias_proposal.support,
                    conflicts=tuple(dict.fromkeys((*alias_proposal.conflicts, "canonical_identity_rejected"))),
                    decision="abstained",
                    promotion_basis=alias_proposal.promotion_basis,
                )

    return ReconciliationResult(
        field_proposals=fields,
        taxonomy_mapping_actions=reconciliation.taxonomy_mapping_actions,
        unresolved_conflicts=reconciliation.unresolved_conflicts,
        open_set_terms=reconciliation.open_set_terms,
        claims_accepted=tuple(_dedupe_mapping_rows(accepted)),
        claims_rejected=tuple(_dedupe_mapping_rows(rejected)),
        claims_unresolved=tuple(_dedupe_mapping_rows(unresolved)),
    )


def _effective_conflicts(
    reconciliation: ReconciliationResult,
    verifier_effects: VerifierDecisionSummary | None,
) -> list[dict[str, Any]]:
    conflicts = []
    for index, value in enumerate(reconciliation.unresolved_conflicts):
        conflict = dict(value)
        conflict.setdefault("conflict_id", f"conflict-{index:03d}")
        conflicts.append(conflict)
    if verifier_effects is None:
        return conflicts
    resolved = set(verifier_effects.resolved_conflict_ids)
    return [conflict for conflict in conflicts if str(conflict.get("conflict_id", "")) not in resolved]


def _flatten_semantic_values(semantics: Mapping[str, Any]) -> dict[str, Any]:
    shape = semantics.get("shape") if isinstance(semantics.get("shape"), Mapping) else {}
    colors = semantics.get("colors") if isinstance(semantics.get("colors"), Mapping) else {}
    return {
        "domain": semantics.get("domain"),
        "category": semantics.get("category"),
        "canonical_object": semantics.get("canonical_object"),
        "surface_alias": semantics.get("surface_alias"),
        "role": semantics.get("role"),
        "explicit_material": semantics.get("explicit_material"),
        "visual_material_cue": semantics.get("visual_material_cue"),
        "silhouette": shape.get("silhouette"),
        "aspect": shape.get("aspect"),
        "orientation": shape.get("orientation"),
        "structure": shape.get("structure"),
        "edge_profile": shape.get("edge_profile"),
        "parts": shape.get("parts"),
        "palette_colors": colors.get("palette_colors"),
        "primary_colors": colors.get("primary_colors"),
        "secondary_colors": colors.get("secondary_colors"),
        "outline_colors": colors.get("outline_colors"),
        "shadow_colors": colors.get("shadow_colors"),
        "highlight_colors": colors.get("highlight_colors"),
        "filename_color_hints": colors.get("filename_color_hints"),
        "size_hint": semantics.get("size_hint"),
        "condition": semantics.get("condition"),
        "style": semantics.get("style"),
        "description": semantics.get("description"),
    }


def _deterministic_supports(field_name: str, deterministic: FilenameParseResult, pixels: Mapping[str, Any]) -> bool:
    if field_name in {"canonical_object", "category", "surface_alias", "role"}:
        return bool(getattr(deterministic, field_name, None)) and getattr(deterministic, field_name, None) != "unknown"
    if field_name == "explicit_material":
        return bool(deterministic.explicit_material)
    if field_name == "filename_color_hints":
        return bool(deterministic.filename_color_hints)
    if field_name == "size_hint":
        return bool(deterministic.size_hint)
    if field_name == "condition":
        return bool(deterministic.condition_hints)
    if field_name in {"palette_colors", "silhouette", "aspect", "orientation"}:
        return bool(pixels.get("visible_pixel_count"))
    if field_name == "style":
        return True
    if field_name == "domain":
        return True
    return False


def _vlm_supports(field_name: str, artifact: VLMProposalArtifact | None) -> bool:
    proposal = artifact.proposal if artifact and artifact.available else None
    if proposal is None:
        return False
    mapping = {
        "canonical_object": bool(proposal.object_candidates),
        "category": bool(proposal.category_candidates),
        "surface_alias": bool(proposal.surface_alias_candidates),
        "role": bool(proposal.role_candidates),
        "visual_material_cue": bool(proposal.material_visual_cues),
        "description": bool(proposal.description_candidates),
        "silhouette": bool(proposal.shape.silhouette),
        "aspect": bool(proposal.shape.aspect),
        "orientation": bool(proposal.shape.orientation),
        "structure": bool(proposal.shape.structure),
        "edge_profile": bool(proposal.shape.edge_profile),
        "parts": bool(proposal.shape.parts),
        "primary_colors": bool(proposal.color_roles.primary_colors),
        "secondary_colors": bool(proposal.color_roles.secondary_colors),
        "outline_colors": bool(proposal.color_roles.outline_colors),
        "shadow_colors": bool(proposal.color_roles.shadow_colors),
        "highlight_colors": bool(proposal.color_roles.highlight_colors),
    }
    return mapping.get(field_name, False)


def _verifier_agreement(verifier: Mapping[str, Any] | None, _field_name: str = "") -> bool:
    if not verifier:
        return False
    results = [value for value in verifier.get("claim_results") or () if isinstance(value, Mapping)]
    return bool(results) and all(str(value.get("verdict")) == "supported" for value in results)


def _object_category_compatible(object_name: str, category: str) -> bool:
    if not object_name or category == "unknown":
        return True
    expected: dict[str, set[str]] = {
        "buckler": {"armor"},
        "shield": {"armor"},
        "helmet": {"armor", "clothing"},
        "pants": {"clothing", "armor"},
        "shirt": {"clothing", "armor"},
        "jacket": {"clothing", "armor"},
        "ring": {"jewelry"},
        "key": {"key"},
        "gem": {"gem"},
        "diamond": {"gem"},
        "crystal_cluster": {"gem", "material"},
    }
    return category in expected.get(object_name, {category})


def _domain_for_category(category: str) -> str:
    if category in {"weapon", "armor", "clothing", "jewelry", "tool"}:
        value = "equipment_icon"
    elif category in {"gem", "material"}:
        value = "resource_icon"
    elif category == "food":
        value = "food_icon"
    elif category == "plant":
        value = "plant_icon"
    elif category == "spell":
        value = "spell_icon"
    elif category == "unknown":
        value = "unknown"
    else:
        value = "inventory_icon"
    if value not in DOMAIN_VALUES:
        raise AssertionError(value)
    return value


def _description_candidates(reconciliation: ReconciliationResult) -> list[str]:
    proposal = reconciliation.field_proposals.get("description")
    if proposal is None:
        return []
    values = [proposal.value, *proposal.alternatives]
    return [str(value) for value in values if isinstance(value, str) and value.strip()]


def _safe_source_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "sprite_id",
        "source_id",
        "pack_id",
        "pack_name",
        "source_name",
        "relative_path",
        "archive_member",
        "source_sheet",
        "declared_material",
        "declared_variant_group",
    )
    return {key: record.get(key) for key in keys if _has_value(record.get(key))}


def _declarative_mapping(record: Mapping[str, Any]) -> dict[str, Any]:
    auto = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), Mapping) else {}
    mapping = auto.get("sheet_mapping") if isinstance(auto.get("sheet_mapping"), Mapping) else {}
    return dict(mapping)


def _approved_field_safe_priors(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    priors = record.get("approved_v4_priors")
    if not isinstance(priors, Sequence) or isinstance(priors, (str, bytes, bytearray)):
        return []
    return [dict(value) for value in priors if isinstance(value, Mapping) and not value.get("legacy_source")]


def _list_field(mapping: Mapping[str, Any], *names: str) -> list[str]:
    for name in names:
        value = mapping.get(name)
        if value:
            values = value if isinstance(value, (list, tuple, set)) else (value,)
            return _dedupe_strings(values)
    return []


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip() != "unknown"
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    return not (isinstance(value, str) and not value.strip())


def _ledger_row(stage: str, execution: _ProviderExecution) -> dict[str, Any]:
    artifact = execution.artifact
    new_usage = {} if execution.cache_hit else dict(artifact.token_usage)
    failure = dict(artifact.failure_diagnostics)
    stage_status = (
        "cache_hit_success"
        if execution.cache_hit
        else "success_after_retry"
        if int(getattr(artifact, "http_attempts", 0)) > 1 and not failure
        else "success"
        if not failure
        else "failed"
    )
    return {
        "stage": stage,
        "provider_call": not execution.cache_hit,
        "new_provider_call": not execution.cache_hit,
        "actual_http_attempts": 0 if execution.cache_hit else int(getattr(artifact, "http_attempts", 0)),
        "cache_hit": execution.cache_hit,
        "shared_cache_hit": execution.shared_cache_hit,
        "cache_key": execution.cache_key,
        "cache_source": "shared" if execution.shared_cache_hit else "local" if execution.cache_hit else "provider",
        "model_identity": artifact.model_identity,
        "request_hash": artifact.request_hash,
        "image_hash": artifact.image_hash,
        "prompt_version": artifact.prompt_version,
        "latency_ms": 0.0 if execution.cache_hit else artifact.latency_ms,
        "execution_latency_ms": execution.execution_latency_ms,
        "artifact_origin_latency_ms": artifact.latency_ms,
        "token_usage": new_usage,
        "new_token_usage": new_usage,
        "artifact_token_usage": dict(artifact.token_usage),
        "failure_diagnostics": dict(artifact.failure_diagnostics),
        "stage_status": stage_status,
        "provider_output_valid": not failure and artifact.parsed_output is not None,
        "fallback_used": False,
        "fallback_reason": failure.get("error_type") if failure else None,
        "training_consequence": "artifact_replayed"
        if execution.cache_hit
        else "provider_evidence_available"
        if not failure
        else "stage_output_excluded",
        "cache_namespace": artifact.cache_namespace,
    }


def _terminal_stage_outcomes(
    ledger: Sequence[Mapping[str, Any]], *, run_b: bool, run_c: bool, run_d: bool
) -> list[dict[str, Any]]:
    result = [
        {
            "stage": "A_deterministic",
            "stage_status": "success",
            "provider_output_valid": True,
            "fallback_used": False,
            "fallback_reason": None,
            "training_consequence": "deterministic_evidence_available",
        }
    ]
    routed = {"B_blind_vlm_proposal": run_b, "C_text_reconciliation": run_c, "D_independent_verifier": run_d}
    by_stage = {str(row.get("stage")): row for row in ledger}
    for name, active in routed.items():
        if not active:
            result.append(
                {
                    "stage": name,
                    "stage_status": "not_routed",
                    "provider_output_valid": False,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "training_consequence": "not_routed",
                }
            )
        else:
            row = by_stage[name]
            result.append(
                {
                    key: row.get(key)
                    for key in (
                        "stage",
                        "stage_status",
                        "provider_output_valid",
                        "fallback_used",
                        "fallback_reason",
                        "training_consequence",
                    )
                }
            )
    return result


def _terminal_record_status(stages: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(stage.get("stage_status")) for stage in stages}
    if "failed" in statuses:
        return "failed"
    if "abstained_after_failure" in statuses:
        return "completed_with_abstention"
    if "deterministic_fallback" in statuses:
        return "completed_with_fallback"
    if "success_after_json_repair" in statuses:
        return "completed_with_repaired_stage"
    return "completed_valid"


def _separate_training_channels(
    output: Mapping[str, Any],
    semantics: Mapping[str, Any],
    quality: Mapping[str, Any],
    description: Mapping[str, Any],
) -> dict[str, Any]:
    critical = ("canonical_object", "category", "domain", "role", "explicit_material", "surface_alias")
    quality_fields = quality.get("fields") if isinstance(quality.get("fields"), Mapping) else {}
    description_valid = bool(semantics.get("description")) and not description.get("claims_rejected")
    return {
        "critical_semantics": {
            "values": {name: semantics.get(name) for name in critical},
            "field_masks": {
                name: int(dict(quality_fields.get(name) or {}).get("supervision_mask", 0)) for name in critical
            },
            "training_state": "field_masked_usable",
        },
        "optional_visual_attributes": {
            "values": semantics.get("shape") or {},
            "training_state": "auxiliary_only" if semantics.get("shape") else "not_available",
        },
        "description_text": {
            "value": semantics.get("description") if description_valid else None,
            "training_state": "active" if description_valid else "excluded_invalid",
        },
        "raw_open_vocabulary_evidence": {
            "values": list(output.get("open_set_terms") or ()),
            "training_state": "provenance_only",
        },
    }


def _provider_accounting(ledger: Sequence[Mapping[str, Any]], config: LabelV4PipelineConfig) -> dict[str, Any]:
    provider_rows = [row for row in ledger if str(row.get("stage", "")).startswith(("B_", "C_", "D_"))]
    per_stage: dict[str, Any] = {}
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for row in provider_rows:
        usage = dict(row.get("new_token_usage") or {})
        stage_input = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        stage_output = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        stage_total = int(usage.get("total_tokens", stage_input + stage_output) or 0)
        input_tokens += stage_input
        output_tokens += stage_output
        total_tokens += stage_total
        per_stage[str(row.get("stage", ""))] = {
            "new_provider_calls": int(bool(row.get("new_provider_call"))),
            "actual_http_attempts": int(row.get("actual_http_attempts", 0) or 0),
            "cache_hit": bool(row.get("cache_hit")),
            "shared_cache_hit": bool(row.get("shared_cache_hit")),
            "execution_latency_ms": float(row.get("execution_latency_ms", 0.0) or 0.0),
            "provider_latency_ms": float(row.get("latency_ms", 0.0) or 0.0),
            "new_token_usage": usage,
            "artifact_token_usage": dict(row.get("artifact_token_usage") or {}),
            "cache_key": str(row.get("cache_key", "")),
        }
    pricing_configured = config.input_cost_per_million is not None and config.output_cost_per_million is not None
    estimated_cost = None
    if pricing_configured:
        estimated_cost = (
            input_tokens * float(config.input_cost_per_million or 0.0)
            + output_tokens * float(config.output_cost_per_million or 0.0)
        ) / 1_000_000.0
    return {
        "logical_stage_count": len(provider_rows),
        "actual_http_attempts": sum(int(row.get("actual_http_attempts", 0) or 0) for row in provider_rows),
        "new_provider_calls": sum(bool(row.get("new_provider_call")) for row in provider_rows),
        "shared_cache_hits": sum(bool(row.get("shared_cache_hit")) for row in provider_rows),
        "cache_hits": sum(bool(row.get("cache_hit")) for row in provider_rows),
        "per_stage": per_stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "pricing_configured": pricing_configured,
        "estimated_provider_cost": estimated_cost,
    }


def _combined_route(
    initial: AdaptiveRoutingDecision,
    final: AdaptiveRoutingDecision,
    *,
    run_b: bool,
    run_c: bool,
    run_d: bool,
) -> AdaptiveRoutingDecision:
    return AdaptiveRoutingDecision(
        run_stage_b=run_b,
        run_stage_c=run_c,
        run_stage_d=run_d,
        stage_b_reasons=initial.stage_b_reasons or (("comparison_mode",) if run_b else ()),
        stage_c_reasons=final.stage_c_reasons or (("proposal_requires_reconciliation",) if run_c else ()),
        stage_d_reasons=(final.stage_d_reasons or ("comparison_dispute_verification",)) if run_d else (),
        stage_d_skipped_reasons=final.stage_d_skipped_reasons if not run_d else (),
    )


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
