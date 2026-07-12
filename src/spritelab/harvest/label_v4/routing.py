"""Adaptive inference routing for Labeling v4 stages A through D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ROUTING_POLICY_VERSION = "adaptive_routing_v4.2"
INDEPENDENT_VERIFIER_PROMPT_VERSION = "independent_dispute_verifier_v4.2"
INDEPENDENT_VERIFIER_CACHE_NAMESPACE = "label_v4_verifier_independent_v1"

RoutingProfile = Literal["semantic_minimal", "semantic_plus_visual", "full_diagnostic"]
DEFAULT_ROUTING_PROFILE: RoutingProfile = "semantic_minimal"
CRITICAL_SEMANTIC_FIELDS = (
    "canonical_object",
    "category",
    "domain",
    "role",
    "explicit_material",
    "surface_alias",
)


def field_coverage_after_stage_a(deterministic: Any) -> dict[str, Any]:
    """Describe critical support without treating inapplicable optionals as missing."""

    values = {name: getattr(deterministic, name, None) for name in CRITICAL_SEMANTIC_FIELDS}
    category = str(values.get("category") or "unknown")
    material_applicable = category in {"armor", "clothing", "jewelry", "weapon"}
    surface_applicable = bool(values.get("surface_alias"))
    supported = {
        "canonical_object": bool(values["canonical_object"]),
        "category": category != "unknown",
        "domain": str(values.get("domain") or "unknown") != "unknown",
        "role": str(values.get("role") or "unknown") != "unknown",
        "explicit_material": bool(values["explicit_material"]) or not material_applicable,
        "surface_alias": bool(values["surface_alias"]) or not surface_applicable,
    }
    return {
        "fields": {name: {"supported": supported[name], "value": values[name]} for name in CRITICAL_SEMANTIC_FIELDS},
        "critical_semantics_complete": all(supported.values()),
        "material_applicable": material_applicable,
        "surface_alias_applicable": surface_applicable,
    }


def decide_profile_routing(
    signals: AdaptiveRoutingSignals,
    *,
    critical_semantics_complete: bool,
    profile: RoutingProfile = DEFAULT_ROUTING_PROFILE,
) -> AdaptiveRoutingDecision:
    """Apply cost-aware profile policy after deterministic field coverage."""

    if profile not in {"semantic_minimal", "semantic_plus_visual", "full_diagnostic"}:
        raise ValueError(f"unknown routing profile: {profile}")
    if profile == "full_diagnostic":
        decision = decide_adaptive_routing(signals)
        return AdaptiveRoutingDecision(
            run_stage_b=True,
            run_stage_c=True,
            run_stage_d=decision.run_stage_d,
            stage_b_reasons=("full_diagnostic",),
            stage_c_reasons=("full_diagnostic",),
            stage_d_reasons=decision.stage_d_reasons,
            stage_d_skipped_reasons=decision.stage_d_skipped_reasons,
        )
    ambiguous = (
        signals.canonical_object_missing
        or signals.filename_generic
        or signals.deterministic_conflicts
        or signals.object_open_set
        or signals.pack_heterogeneous
    )
    run_b = profile == "semantic_plus_visual" or ambiguous or not critical_semantics_complete
    run_c = ambiguous or signals.deterministic_conflicts
    reasons_b = ("optional_visual_enrichment",) if profile == "semantic_plus_visual" and not ambiguous else ()
    if ambiguous:
        reasons_b += ("critical_semantic_ambiguity",)
    elif not critical_semantics_complete:
        reasons_b += ("critical_semantic_coverage_incomplete",)
    return AdaptiveRoutingDecision(
        run_stage_b=run_b,
        run_stage_c=run_c,
        run_stage_d=False,
        stage_b_reasons=reasons_b,
        stage_c_reasons=("semantic_conflict_or_open_set_mapping",) if run_c else (),
        stage_d_skipped_reasons=("no_post_reconciliation_dispute",),
    )


@dataclass(frozen=True)
class AdaptiveRoutingSignals:
    """Cheap signals available before or immediately after reconciliation."""

    canonical_object_missing: bool = False
    filename_generic: bool = False
    pack_heterogeneous: bool = False
    deterministic_conflicts: bool = False
    shape_weak: bool = False
    description_missing: bool = False
    object_open_set: bool = False
    source_profile_unknown: bool = False

    critical_field_risk_upper: float | None = None
    vlm_deterministic_disagreement: bool = False
    object_category_inconsistent: bool = False
    filename_visual_color_conflict: bool = False
    material_visual_only: bool = False
    open_set_novelty: float = 0.0
    variant_family_inconsistent: bool = False
    policy_abstains_object_identity: bool = False
    source_category_authoritative: bool = False
    verifier_eligible_claim_count: int | None = None

    def __post_init__(self) -> None:
        if self.critical_field_risk_upper is not None and not 0.0 <= self.critical_field_risk_upper <= 1.0:
            raise ValueError("critical_field_risk_upper must be in [0, 1]")
        if not 0.0 <= self.open_set_novelty <= 1.0:
            raise ValueError("open_set_novelty must be in [0, 1]")
        if self.verifier_eligible_claim_count is not None and self.verifier_eligible_claim_count < 0:
            raise ValueError("verifier_eligible_claim_count must be non-negative")


@dataclass(frozen=True)
class AdaptiveRoutingDecision:
    policy_version: str = ROUTING_POLICY_VERSION
    run_stage_a: bool = True
    run_stage_b: bool = False
    run_stage_c: bool = False
    run_stage_d: bool = False
    stage_b_reasons: tuple[str, ...] = ()
    stage_c_reasons: tuple[str, ...] = ()
    stage_d_reasons: tuple[str, ...] = ()
    stage_d_skipped_reasons: tuple[str, ...] = ()
    verifier_prompt_version: str = INDEPENDENT_VERIFIER_PROMPT_VERSION
    verifier_cache_namespace: str = INDEPENDENT_VERIFIER_CACHE_NAMESPACE
    verifier_must_be_independent: bool = True

    @property
    def disputed_claims_required(self) -> bool:
        return self.run_stage_d

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "run_stage_a": self.run_stage_a,
            "run_stage_b": self.run_stage_b,
            "run_stage_c": self.run_stage_c,
            "run_stage_d": self.run_stage_d,
            "stage_b_reasons": list(self.stage_b_reasons),
            "stage_c_reasons": list(self.stage_c_reasons),
            "stage_d_reasons": list(self.stage_d_reasons),
            "stage_d_skipped_reasons": list(self.stage_d_skipped_reasons),
            "verifier_prompt_version": self.verifier_prompt_version,
            "verifier_cache_namespace": self.verifier_cache_namespace,
            "verifier_must_be_independent": self.verifier_must_be_independent,
        }


def decide_adaptive_routing(
    signals: AdaptiveRoutingSignals,
    *,
    verifier_risk_threshold: float = 0.40,
    novelty_threshold: float = 0.65,
) -> AdaptiveRoutingDecision:
    """Choose only the expensive stages justified by current risk signals.

    Stage A is always deterministic and always runs. Stage B is one blind rich
    proposal, Stage C reconciles every proposal and any deterministic conflict,
    and Stage D independently checks only disputed/high-risk claims.
    """

    if not 0.0 <= verifier_risk_threshold <= 1.0:
        raise ValueError("verifier_risk_threshold must be in [0, 1]")
    if not 0.0 <= novelty_threshold <= 1.0:
        raise ValueError("novelty_threshold must be in [0, 1]")

    stage_b_flags = (
        ("canonical_object_missing", signals.canonical_object_missing),
        ("filename_generic", signals.filename_generic),
        ("pack_heterogeneous", signals.pack_heterogeneous),
        ("deterministic_conflicts", signals.deterministic_conflicts),
        # Weak shape/prose alone never justifies paid semantic inference.
        ("object_open_set", signals.object_open_set),
        ("source_profile_unknown", signals.source_profile_unknown),
    )
    stage_b_reasons = tuple(name for name, active in stage_b_flags if active)
    run_stage_b = bool(stage_b_reasons)

    stage_c_reasons: list[str] = []
    if run_stage_b:
        stage_c_reasons.append("vlm_proposal_requires_reconciliation")
    if signals.deterministic_conflicts:
        stage_c_reasons.append("deterministic_conflict_requires_reconciliation")

    high_risk = (
        signals.critical_field_risk_upper is not None and signals.critical_field_risk_upper > verifier_risk_threshold
    )
    policy_already_abstains = signals.policy_abstains_object_identity and signals.source_category_authoritative
    stage_d_flags = (
        (
            "vlm_deterministic_disagreement",
            signals.vlm_deterministic_disagreement and not policy_already_abstains,
        ),
        ("object_category_inconsistent", signals.object_category_inconsistent),
        ("filename_visual_color_conflict", signals.filename_visual_color_conflict),
        ("material_inferred_not_explicit", signals.material_visual_only),
        ("variant_family_inconsistent", signals.variant_family_inconsistent),
    )
    stage_d_reasons = [name for name, active in stage_d_flags if active]
    stage_d_skipped_reasons: list[str] = []

    # Risk and novelty prioritize an already-concrete dispute; neither creates
    # a visually adjudicable claim by itself. If normalization has already
    # chosen to abstain, another open-set model call adds no semantic value.
    if stage_d_reasons and high_risk:
        stage_d_reasons.insert(0, "critical_field_risk_above_threshold")
    elif high_risk:
        stage_d_skipped_reasons.append("risk_without_concrete_dispute")
    if stage_d_reasons and signals.open_set_novelty >= novelty_threshold:
        stage_d_reasons.append("open_set_novelty_high")
    elif signals.open_set_novelty >= novelty_threshold:
        stage_d_skipped_reasons.append("open_set_novelty_without_concrete_dispute")
    if policy_already_abstains and signals.vlm_deterministic_disagreement:
        stage_d_skipped_reasons.append("policy_abstains_source_category_visual_form_disagreement")
    if signals.verifier_eligible_claim_count == 0:
        stage_d_reasons = []
        stage_d_skipped_reasons.append("no_verifier_eligible_claims")

    # A post-reconciliation risk signal can route directly to verification. If
    # C has not yet run, ensure it runs first so the verifier receives a compact
    # list of disputed claims rather than the first model's narrative.
    if stage_d_reasons and not stage_c_reasons:
        stage_c_reasons.append("prepare_disputed_claims_for_verifier")

    return AdaptiveRoutingDecision(
        run_stage_b=run_stage_b,
        run_stage_c=bool(stage_c_reasons),
        run_stage_d=bool(stage_d_reasons),
        stage_b_reasons=stage_b_reasons,
        stage_c_reasons=tuple(stage_c_reasons),
        stage_d_reasons=tuple(stage_d_reasons),
        stage_d_skipped_reasons=tuple(dict.fromkeys(stage_d_skipped_reasons)),
    )


# Pipeline-friendly aliases.
route_labeling_stages = decide_adaptive_routing
route_stages = decide_adaptive_routing
