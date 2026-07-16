"""Deterministic, traceable, per-depth hierarchical decision engine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from spritelab.hierarchical_labeling.calibration import CalibrationModel
from spritelab.hierarchical_labeling.contracts import (
    CalibrationState,
    DecisionState,
    EvidenceChannel,
    FieldDecision,
    HierarchicalLabelDecision,
    LabelEvidenceBundle,
    ScoreComponent,
    SemanticHypothesis,
)
from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    content_identity,
    require_probability,
)
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph

DECISION_ENGINE_VERSION = "spritelab-hierarchical-decision-engine-v1"
APPEND_ONLY_HUMAN_TRUTH = "append_only_human_review"
SYNTHETIC_ORACLE_TRUTH = "synthetic_oracle"
SYNTHETIC_ORACLE_CONFLICT = "synthetic_oracle_calibration_non_production"
CONTROLLED_NON_NODE_ABSTENTIONS = frozenset(
    {"reviewer_abstained", "taxonomy_gap", "technically_or_semantically_unusable"}
)


@dataclass(frozen=True)
class DecisionPolicy:
    visual_weight: float = 0.62
    description_compatibility_weight: float = 0.08
    retrieval_weight: float = 0.18
    metadata_weight: float = 0.04
    context_weight: float = 0.04
    provider_validity_weight: float = 0.04
    conflict_penalty_weight: float = 0.28
    novelty_penalty_weight: float = 0.12
    maximum_novelty_for_leaf: float = 0.72
    reviewed_retrieval_conflict_fraction: float = 0.6

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            require_probability(value, name.replace("_", " "))
        positive = (
            self.visual_weight
            + self.description_compatibility_weight
            + self.retrieval_weight
            + self.metadata_weight
            + self.context_weight
            + self.provider_validity_weight
        )
        if not math.isclose(positive, 1.0):
            raise ContractValidationError("positive decision component weights must sum to 1")

    @property
    def identity(self) -> str:
        return content_identity(
            DECISION_ENGINE_VERSION,
            {"version": DECISION_ENGINE_VERSION, **self.__dict__},
        )


def _all_hypotheses(bundle: LabelEvidenceBundle) -> tuple[SemanticHypothesis, ...]:
    return tuple(item for path in bundle.taxonomy_hypotheses for item in path.hypotheses)


def _candidate_path(bundle: LabelEvidenceBundle, graph: TaxonomyGraph) -> tuple[str, ...]:
    candidates = [hypothesis for hypothesis in _all_hypotheses(bundle) if hypothesis.rank == 1]
    deepest = max(candidates, key=lambda item: (item.depth, item.raw_model_confidence or 0.0), default=None)
    return graph.path(deepest.node_id) if deepest is not None else ()


def _hypothesis_support(
    node_id: str, hypotheses: Sequence[SemanticHypothesis], graph: TaxonomyGraph
) -> tuple[float, tuple[str, ...]]:
    direct = [item for item in hypotheses if item.node_id == node_id]
    reasons: list[str] = []
    if direct:
        usable = [item.raw_model_confidence for item in direct if item.raw_model_confidence is not None]
        score = max(usable, default=0.0)
        reasons.append(f"{len(direct)} direct visual hypothesis pass(es)")
        if any(item.abstention_recommended for item in direct):
            score *= 0.75
            reasons.append("provider recommended abstention at this node")
        return score, tuple(reasons)
    descendants = [
        item
        for item in hypotheses
        if graph.is_ancestor(node_id, item.node_id) and item.raw_model_confidence is not None
    ]
    if descendants:
        reasons.append("conservative ancestor support projected from a deeper visual hypothesis")
        return max(float(item.raw_model_confidence or 0.0) for item in descendants) * 0.9, tuple(reasons)
    return 0.0, ("no visual hypothesis supports this node",)


def _description_compatibility(node_id: str, hypotheses: Sequence[SemanticHypothesis]) -> tuple[float, tuple[str, ...]]:
    direct = [item for item in hypotheses if item.node_id == node_id]
    citations = sum(len(item.evidence_citations) for item in direct)
    contradictions = sum(len(item.contradicting_observations) for item in direct)
    if citations + contradictions == 0:
        return 0.0, ("no cited visual observation",)
    score = citations / (citations + contradictions)
    return score, (f"{citations} cited observation(s); {contradictions} contradiction(s)",)


def _retrieval_support(bundle: LabelEvidenceBundle, node_id: str) -> tuple[float, tuple[str, ...], float]:
    if bundle.retrieval is None:
        return 0.0, ("retrieval evidence unavailable",), 0.0
    reviewed = [neighbor for neighbor in bundle.retrieval.neighbors if neighbor.review_status == "reviewed"]
    proposal_count = sum(neighbor.review_status == "proposal" for neighbor in bundle.retrieval.neighbors)
    if not reviewed:
        reason = "no reviewed neighbors"
        if proposal_count:
            reason += f"; {proposal_count} proposal neighbor(s) excluded from authoritative support"
        return 0.0, (reason,), 0.0
    total_weight = sum(max(0.0, neighbor.similarity) for neighbor in reviewed)
    supporting = sum(
        max(0.0, neighbor.similarity) for neighbor in reviewed if node_id in neighbor.verified_taxonomy_path
    )
    score = supporting / total_weight if total_weight else 0.0
    conflict_fraction = 1.0 - score
    return (
        score,
        (
            f"{sum(node_id in neighbor.verified_taxonomy_path for neighbor in reviewed)}/{len(reviewed)} reviewed neighbors support node",
            f"{proposal_count} proposal neighbor(s) kept non-authoritative",
        ),
        conflict_fraction,
    )


def _claim_support(
    claims: Sequence[tuple[str, str]], node_id: str, graph: TaxonomyGraph, *, verified: bool
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    resolved = [(field, graph.resolve(value)) for field, value in claims]
    known = [(field, value) for field, value in resolved if value is not None]
    if not known:
        return 0.0, ("no controlled taxonomy claim",), ()
    supporting = [value for _field, value in known if node_id in graph.path(value or "")]
    conflicts = [
        value
        for _field, value in known
        if value not in supporting and graph.lowest_common_ancestor(node_id, value or "") != node_id
    ]
    strength = 0.8 if verified else 0.45
    return (
        strength * len(supporting) / len(known),
        (f"{len(supporting)}/{len(known)} controlled claim(s) support node",),
        tuple(value or "" for value in conflicts),
    )


def _components_for_node(
    bundle: LabelEvidenceBundle,
    node_id: str,
    graph: TaxonomyGraph,
    policy: DecisionPolicy,
) -> tuple[tuple[ScoreComponent, ...], tuple[str, ...]]:
    hypotheses = _all_hypotheses(bundle)
    visual, visual_reasons = _hypothesis_support(node_id, hypotheses, graph)
    description, description_reasons = _description_compatibility(node_id, hypotheses)
    retrieval, retrieval_reasons, retrieval_conflict = _retrieval_support(bundle, node_id)
    metadata, metadata_reasons, metadata_conflicts = (
        _claim_support(bundle.metadata.claims, node_id, graph, verified=bundle.metadata.verified)
        if bundle.metadata
        else (0.0, ("metadata channel unavailable",), ())
    )
    context, context_reasons, context_conflicts = (
        _claim_support(bundle.context.claims, node_id, graph, verified=False)
        if bundle.context and bundle.context.permitted_by_policy
        else (0.0, ("pack context unavailable or not permitted",), ())
    )
    novelty = bundle.retrieval.novelty_score if bundle.retrieval else 1.0
    conflict_penalty = retrieval_conflict if visual >= 0.6 else 0.0
    conflicts: list[str] = []
    if metadata_conflicts and visual >= 0.6:
        conflicts.append(f"visual_metadata_conflict:{node_id}:{','.join(metadata_conflicts)}")
    if context_conflicts and visual >= 0.6:
        conflicts.append(f"visual_context_conflict:{node_id}:{','.join(context_conflicts)}")
    if retrieval_conflict >= policy.reviewed_retrieval_conflict_fraction and visual >= 0.6:
        conflicts.append(f"visual_retrieval_conflict:{node_id}")
    provider_valid = float(bool(hypotheses))
    components = (
        ScoreComponent(EvidenceChannel.VISUAL_ONLY, "visual_hypothesis", visual, policy.visual_weight, visual_reasons),
        ScoreComponent(
            EvidenceChannel.VISUAL_ONLY,
            "description_compatibility",
            description,
            policy.description_compatibility_weight,
            description_reasons,
        ),
        ScoreComponent(
            EvidenceChannel.RETRIEVAL, "reviewed_retrieval", retrieval, policy.retrieval_weight, retrieval_reasons
        ),
        ScoreComponent(
            EvidenceChannel.METADATA, "metadata_support", metadata, policy.metadata_weight, metadata_reasons
        ),
        ScoreComponent(
            EvidenceChannel.PACK_CONTEXT, "context_support", context, policy.context_weight, context_reasons
        ),
        ScoreComponent(
            EvidenceChannel.VISUAL_ONLY,
            "provider_validity",
            provider_valid,
            policy.provider_validity_weight,
            ("structured hypothesis output validated" if provider_valid else "no valid structured hypothesis output",),
        ),
        ScoreComponent(
            EvidenceChannel.RETRIEVAL,
            "retrieval_conflict_penalty",
            -conflict_penalty,
            policy.conflict_penalty_weight,
            (f"reviewed-neighbor conflict fraction {retrieval_conflict:.3f}",),
        ),
        ScoreComponent(
            EvidenceChannel.RETRIEVAL,
            "novelty_penalty",
            -novelty,
            policy.novelty_penalty_weight,
            (f"novelty score {novelty:.3f}",),
        ),
    )
    return components, tuple(conflicts)


def _raw_score(components: Sequence[ScoreComponent]) -> float:
    value = sum(component.value * component.weight for component in components)
    return round(max(0.0, min(1.0, value)), 8)


def validate_bundle_for_graph(bundle: LabelEvidenceBundle, graph: TaxonomyGraph) -> None:
    """Revalidate graph-sensitive bindings at every decision/export trust boundary."""

    bundle.__post_init__()
    if bundle.taxonomy_identity != graph.identity:
        raise ContractValidationError("evidence bundle taxonomy does not bind the selected graph")
    for path_hypothesis in bundle.taxonomy_hypotheses:
        if path_hypothesis.path and graph.path(path_hypothesis.path[-1]) != path_hypothesis.path:
            raise ContractValidationError("taxonomy hypothesis path does not bind the selected graph")
        response_bindings = {
            (
                item.provider_identity,
                item.model_identity,
                item.prompt_identity,
                item.render_bundle_identity,
                item.taxonomy_identity,
            )
            for item in path_hypothesis.hypotheses
        }
        if len(response_bindings) > 1:
            raise ContractValidationError("semantic hypotheses from one response do not share provider bindings")
        for item in path_hypothesis.hypotheses:
            if (
                graph.resolve(item.node_id) != item.node_id
                or graph.depth(item.node_id) != item.depth
                or item.taxonomy_identity != graph.identity
            ):
                raise ContractValidationError("semantic hypothesis does not bind a valid graph node/depth")
    if bundle.retrieval is not None:
        if bundle.retrieval.taxonomy_identity != graph.identity:
            raise ContractValidationError("retrieval evidence taxonomy does not bind the selected graph")
        for neighbor in bundle.retrieval.neighbors:
            if neighbor.taxonomy_identity != graph.identity:
                raise ContractValidationError("retrieval neighbor taxonomy does not bind the selected graph")
            for path_name, candidate_path in (
                ("verified", neighbor.verified_taxonomy_path),
                ("proposal", neighbor.proposal_taxonomy_path),
            ):
                if candidate_path and graph.path(candidate_path[-1]) != candidate_path:
                    raise ContractValidationError(
                        f"retrieval neighbor {path_name} path does not bind the selected graph"
                    )
    human = bundle.human
    if human is None:
        return
    verification = human.verification
    if verification is None or not human.verified_append_only:
        raise ContractValidationError("human decisions require verified append-only review truth")
    verification.__post_init__()
    human.__post_init__()
    if (
        human.record_identity != bundle.record_identity
        or human.taxonomy_identity != graph.identity
        or verification.image_identity != bundle.image_identity
        or verification.evidence_bundle_identity != bundle.nonhuman_identity
    ):
        raise ContractValidationError("human truth does not cross-bind the decision evidence bundle")
    if human.taxonomy_path and graph.path(human.taxonomy_path[-1]) != human.taxonomy_path:
        raise ContractValidationError("human truth path does not bind the selected graph")
    for abstention in human.explicit_abstentions:
        resolved = graph.resolve(abstention)
        if resolved is None:
            if abstention not in CONTROLLED_NON_NODE_ABSTENTIONS:
                raise ContractValidationError("human truth contains an uncontrolled abstention")
            continue
        if resolved != abstention:
            raise ContractValidationError("human truth abstention names a deprecated taxonomy node")
        if resolved in human.taxonomy_path or any(
            graph.is_ancestor(resolved, accepted) for accepted in human.taxonomy_path
        ):
            raise ContractValidationError("human truth cannot accept a node below an explicit abstention")


def _decision_path(bundle: LabelEvidenceBundle, graph: TaxonomyGraph) -> tuple[str, ...]:
    candidate = _candidate_path(bundle, graph)
    human_path = bundle.human.taxonomy_path if bundle.human is not None else ()
    if not human_path:
        return candidate
    if not candidate:
        return human_path
    human_deepest = human_path[-1]
    candidate_deepest = candidate[-1]
    if human_deepest == candidate_deepest or graph.is_ancestor(human_deepest, candidate_deepest):
        return candidate
    return human_path


def _human_abstention_blocker(bundle: LabelEvidenceBundle, node_id: str, graph: TaxonomyGraph) -> str | None:
    human = bundle.human
    if human is None:
        return None
    for abstention in human.explicit_abstentions:
        resolved = graph.resolve(abstention)
        if resolved is not None and (resolved == node_id or graph.is_ancestor(resolved, node_id)):
            return resolved
    non_node = next(
        (value for value in human.explicit_abstentions if value in CONTROLLED_NON_NODE_ABSTENTIONS),
        None,
    )
    if non_node is not None and node_id not in human.taxonomy_path:
        return non_node
    return None


def decide_hierarchical_label(
    bundle: LabelEvidenceBundle,
    graph: TaxonomyGraph,
    calibration: CalibrationModel,
    *,
    policy: DecisionPolicy | None = None,
) -> HierarchicalLabelDecision:
    """Decide each depth independently, then stop at the deepest safe node."""

    selected_policy = policy or DecisionPolicy()
    validate_bundle_for_graph(bundle, graph)
    if calibration.taxonomy_identity != graph.identity:
        raise ContractValidationError("decision inputs do not bind the selected taxonomy")
    truth_scope = str(getattr(calibration, "truth_scope", APPEND_ONLY_HUMAN_TRUTH))
    synthetic_oracle = truth_scope == SYNTHETIC_ORACLE_TRUTH
    effective_calibration_state = calibration.state
    if synthetic_oracle and effective_calibration_state == CalibrationState.VALIDATED_FOR_SCOPE:
        effective_calibration_state = CalibrationState.READY_FOR_EXPERIMENT
    path = _decision_path(bundle, graph)
    all_hypotheses = _all_hypotheses(bundle)
    alternatives = tuple(
        dict.fromkeys(
            item.node_id
            for item in sorted(all_hypotheses, key=lambda value: (value.depth, value.rank, value.node_id))
            if item.rank > 1
        )
    )
    initial_conflicts = (SYNTHETIC_ORACLE_CONFLICT,) if synthetic_oracle else ()
    if not path:
        return HierarchicalLabelDecision(
            bundle.record_identity,
            graph.identity,
            bundle.identity,
            (),
            None,
            None,
            alternatives,
            (),
            bundle.contributed_channels,
            initial_conflicts,
            effective_calibration_state,
        )
    level_decisions: list[FieldDecision] = []
    accepted_path: list[str] = []
    conflicts = list(initial_conflicts)
    parent_accepted = True
    human_path = bundle.human.taxonomy_path if bundle.human else ()
    blank = bool(bundle.technical.feature("empty_blank_status", False))
    novelty = bundle.retrieval.novelty_score if bundle.retrieval else 1.0
    for node_id in path:
        node = graph.node(node_id)
        depth = graph.depth(node_id)
        components, node_conflicts = _components_for_node(bundle, node_id, graph, selected_policy)
        conflicts.extend(node_conflicts)
        score = _raw_score(components)
        threshold, threshold_source = calibration.threshold_for(node_id, graph)
        probability = calibration.calibrated_probability(node_id, score, graph)
        reasons = [f"threshold source {threshold_source}"]
        blocker = _human_abstention_blocker(bundle, node_id, graph)
        if blank:
            state = DecisionState.ABSTAINED
            reasons.append("technically blank image")
        elif blocker is not None:
            state = DecisionState.HUMAN_ABSTAINED
            reasons.append(f"verified human abstention at {blocker} blocks this node and descendants")
        elif bundle.human is not None and node_id in human_path:
            state = DecisionState.HUMAN_VERIFIED
            reasons.append("append-only human review accepted this taxonomy level")
        elif not parent_accepted:
            state = DecisionState.ABSTAINED
            reasons.append("parent taxonomy level was not accepted")
        elif node.human_truth_required:
            state = DecisionState.ABSTAINED
            reasons.append("taxonomy node requires direct human truth")
        elif not node.may_be_automatically_accepted:
            state = DecisionState.ABSTAINED
            reasons.append("taxonomy node is not eligible for automatic acceptance")
        elif effective_calibration_state not in {
            CalibrationState.READY_FOR_EXPERIMENT,
            CalibrationState.VALIDATED_FOR_SCOPE,
        }:
            state = DecisionState.ABSTAINED
            reasons.append("human-truth calibration is not ready")
        elif depth >= 3 and novelty > selected_policy.maximum_novelty_for_leaf:
            state = DecisionState.ABSTAINED
            reasons.append("high novelty requires a safer ancestor or human review")
        elif node_conflicts and depth >= 2:
            state = DecisionState.ABSTAINED
            reasons.append("strong cross-channel conflict blocks this depth")
        elif score >= threshold:
            state = DecisionState.ACCEPTED
            reasons.append("calibrated raw-score threshold passed")
            if synthetic_oracle:
                reasons.append("synthetic-oracle calibration is experimental and non-production")
        else:
            state = DecisionState.ABSTAINED
            reasons.append("calibrated raw-score threshold did not pass")
        accepted = state in {DecisionState.ACCEPTED, DecisionState.HUMAN_VERIFIED}
        if accepted:
            accepted_path.append(node_id)
        parent_accepted = accepted
        level_decisions.append(
            FieldDecision(
                node_id,
                depth,
                state,
                score,
                probability,
                threshold,
                components,
                tuple(reasons),
                selected_policy.identity,
                bundle.identity,
            )
        )
    deepest = accepted_path[-1] if accepted_path else None
    abstained_below = next((node_id for node_id in path if node_id not in accepted_path), None)
    return HierarchicalLabelDecision(
        bundle.record_identity,
        graph.identity,
        bundle.identity,
        tuple(accepted_path),
        deepest,
        abstained_below,
        alternatives,
        tuple(level_decisions),
        bundle.contributed_channels,
        tuple(dict.fromkeys(conflicts)),
        effective_calibration_state,
    )


def conditional_field_decision(
    field_name: str,
    value: str | None,
    *,
    direct_visual_support: bool,
    calibrated_probability: float | None,
    threshold: float,
    contradiction: bool,
    decision_policy_identity: str,
    evidence_bundle_identity: str,
) -> FieldDecision:
    """Fail closed for conditional canonical-object and role fields."""

    if field_name not in {"canonical_object", "role"}:
        raise ContractValidationError("conditional decisions are limited to canonical_object and role")
    require_probability(threshold, "conditional threshold")
    if calibrated_probability is not None:
        require_probability(calibrated_probability, "conditional calibrated probability")
    accepted = bool(
        value
        and direct_visual_support
        and calibrated_probability is not None
        and calibrated_probability >= threshold
        and not contradiction
    )
    reasons = []
    if not value:
        reasons.append("no concrete controlled value")
    if not direct_visual_support:
        reasons.append("not directly visually supportable")
    if calibrated_probability is None or calibrated_probability < threshold:
        reasons.append("calibrated precision threshold unavailable or not passed")
    if contradiction:
        reasons.append("strong contradiction")
    if accepted:
        reasons.append("conditional identity passed every mandatory gate")
    return FieldDecision(
        field_name,
        0,
        DecisionState.ACCEPTED if accepted else DecisionState.MODEL_ABSTAINED,
        calibrated_probability,
        calibrated_probability,
        threshold,
        (),
        tuple(reasons),
        decision_policy_identity,
        evidence_bundle_identity,
    )
