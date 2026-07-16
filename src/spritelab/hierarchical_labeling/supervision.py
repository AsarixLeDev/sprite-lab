"""Explicit supervision tiers and conservative export policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spritelab.hierarchical_labeling.contracts import (
    CalibrationState,
    DecisionState,
    HierarchicalLabelDecision,
    LabelEvidenceBundle,
    SupervisionExport,
    SupervisionTier,
)
from spritelab.hierarchical_labeling.decision import (
    CONTROLLED_NON_NODE_ABSTENTIONS,
    SYNTHETIC_ORACLE_CONFLICT,
    validate_bundle_for_graph,
)
from spritelab.hierarchical_labeling.json_utils import ContractValidationError
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph
from spritelab.hierarchical_labeling.technical import technical_supervision

DEFAULT_TRAINING_WEIGHTS: Mapping[SupervisionTier, float] = {
    SupervisionTier.HUMAN_VERIFIED: 1.0,
    SupervisionTier.HUMAN_ABSTAINED: 0.0,
    SupervisionTier.CALIBRATED_CORE: 0.7,
    SupervisionTier.RETRIEVAL_SUPPORTED_WEAK: 0.3,
    SupervisionTier.MODEL_PROPOSAL: 0.0,
    SupervisionTier.AUXILIARY_VISUAL: 0.1,
    SupervisionTier.TECHNICAL_DETERMINISTIC: 0.5,
    SupervisionTier.UNUSABLE: 0.0,
}


def _verified_human(bundle: LabelEvidenceBundle) -> bool:
    return bool(
        bundle.human is not None and bundle.human.verification is not None and bundle.human.verified_append_only
    )


def _deepest_level(decision: HierarchicalLabelDecision):
    if decision.deepest_accepted_node is None:
        return None
    return next(
        (item for item in decision.level_decisions if item.node_id == decision.deepest_accepted_node),
        None,
    )


def _verified_human_abstains_at(bundle: LabelEvidenceBundle, node_id: str, graph: TaxonomyGraph) -> bool:
    if not _verified_human(bundle) or bundle.human is None:
        return False
    for abstention in bundle.human.explicit_abstentions:
        resolved = graph.resolve(abstention)
        if resolved is not None and (resolved == node_id or graph.is_ancestor(resolved, node_id)):
            return True
    return node_id not in bundle.human.taxonomy_path and any(
        abstention in CONTROLLED_NON_NODE_ABSTENTIONS for abstention in bundle.human.explicit_abstentions
    )


def _validate_export_bindings(
    bundle: LabelEvidenceBundle,
    decision: HierarchicalLabelDecision,
    graph: TaxonomyGraph,
) -> None:
    validate_bundle_for_graph(bundle, graph)
    decision.__post_init__()
    if (
        decision.record_identity != bundle.record_identity
        or decision.taxonomy_identity != graph.identity
        or decision.evidence_bundle_identity != bundle.identity
    ):
        raise ContractValidationError("supervision decision does not bind the evidence bundle and taxonomy")
    if decision.contributed_channels != bundle.contributed_channels:
        raise ContractValidationError("supervision decision channels do not bind the evidence bundle")
    if decision.taxonomy_path and graph.path(decision.taxonomy_path[-1]) != decision.taxonomy_path:
        raise ContractValidationError("supervision decision accepted path is invalid for the taxonomy")
    levels = decision.level_decisions
    if levels:
        level_path = tuple(item.node_id for item in levels)
        if graph.path(level_path[-1]) != level_path:
            raise ContractValidationError("supervision level decisions do not form one taxonomy path")
    accepted_path: list[str] = []
    seen_abstention = False
    for level in levels:
        level.__post_init__()
        if level.evidence_bundle_identity != bundle.identity or level.depth != graph.depth(level.node_id):
            raise ContractValidationError("supervision level does not bind the evidence bundle node/depth")
        accepted = level.state in {DecisionState.ACCEPTED, DecisionState.HUMAN_VERIFIED}
        if accepted and seen_abstention:
            raise ContractValidationError("supervision decision accepts a descendant below an abstained level")
        if accepted:
            accepted_path.append(level.node_id)
        else:
            seen_abstention = True
        if level.state == DecisionState.HUMAN_VERIFIED and (
            not _verified_human(bundle) or bundle.human is None or level.node_id not in bundle.human.taxonomy_path
        ):
            raise ContractValidationError("human-verified supervision lacks exact append-only human truth")
        if level.state == DecisionState.HUMAN_ABSTAINED and not _verified_human_abstains_at(
            bundle, level.node_id, graph
        ):
            raise ContractValidationError("human-abstained supervision lacks an exact append-only human blocker")
    if _verified_human(bundle) and bundle.human is not None:
        states_by_node = {item.node_id: item.state for item in levels}
        if any(
            node_id in decision.taxonomy_path and states_by_node.get(node_id) != DecisionState.HUMAN_VERIFIED
            for node_id in bundle.human.taxonomy_path
        ):
            raise ContractValidationError("supervision decision does not preserve the verified human path")
    if tuple(accepted_path) != decision.taxonomy_path:
        raise ContractValidationError("supervision decision path does not match its per-depth states")
    expected_deepest = accepted_path[-1] if accepted_path else None
    if decision.deepest_accepted_node != expected_deepest:
        raise ContractValidationError("supervision decision deepest node does not match its per-depth states")
    expected_abstention = next(
        (item.node_id for item in levels if item.state not in {DecisionState.ACCEPTED, DecisionState.HUMAN_VERIFIED}),
        None,
    )
    if decision.abstained_below_node != expected_abstention:
        raise ContractValidationError("supervision decision abstention boundary does not match its per-depth states")
    if any(graph.resolve(node_id) != node_id for node_id in decision.top_k_alternatives):
        raise ContractValidationError("supervision alternatives contain an invalid or deprecated taxonomy node")


def _tier(bundle: LabelEvidenceBundle, decision: HierarchicalLabelDecision) -> SupervisionTier:
    deepest_level = _deepest_level(decision)
    if deepest_level is not None and deepest_level.state == DecisionState.HUMAN_VERIFIED:
        return SupervisionTier.HUMAN_VERIFIED
    if bool(bundle.technical.feature("empty_blank_status", False)):
        return SupervisionTier.UNUSABLE
    deepest_evaluated = decision.level_decisions[-1] if decision.level_decisions else None
    if decision.deepest_accepted_node is None and _verified_human(bundle) and bundle.human is not None:
        if (deepest_evaluated is not None and deepest_evaluated.state == DecisionState.HUMAN_ABSTAINED) or (
            deepest_evaluated is None and bundle.human.deepest_accepted_node is None
        ):
            return SupervisionTier.HUMAN_ABSTAINED
    if SYNTHETIC_ORACLE_CONFLICT in decision.conflicts:
        return SupervisionTier.MODEL_PROPOSAL
    if decision.deepest_accepted_node and decision.calibration_state == CalibrationState.VALIDATED_FOR_SCOPE:
        return SupervisionTier.CALIBRATED_CORE
    if (
        decision.deepest_accepted_node
        and bundle.retrieval is not None
        and any(neighbor.review_status == "reviewed" for neighbor in bundle.retrieval.neighbors)
    ):
        return SupervisionTier.RETRIEVAL_SUPPORTED_WEAK
    if bundle.taxonomy_hypotheses:
        return SupervisionTier.MODEL_PROPOSAL
    if bundle.visual_description is not None:
        return SupervisionTier.AUXILIARY_VISUAL
    return SupervisionTier.TECHNICAL_DETERMINISTIC


def _evidence_identities(bundle: LabelEvidenceBundle) -> tuple[str, ...]:
    values = [bundle.technical.identity]
    if bundle.visual_description:
        values.append(bundle.visual_description.identity)
    if bundle.visual_attributes:
        values.append(bundle.visual_attributes.identity)
    values.extend(item.identity for item in bundle.taxonomy_hypotheses)
    for item in (bundle.retrieval, bundle.metadata, bundle.context, bundle.human):
        if item is not None:
            values.append(item.identity)
    return tuple(dict.fromkeys(values))


def export_supervision(
    bundle: LabelEvidenceBundle,
    decision: HierarchicalLabelDecision,
    graph: TaxonomyGraph,
    *,
    training_weights: Mapping[SupervisionTier, float] | None = None,
) -> SupervisionExport:
    """Export one tiered record without turning proposals into human truth."""

    _validate_export_bindings(bundle, decision, graph)
    tier = _tier(bundle, decision)
    weights = dict(DEFAULT_TRAINING_WEIGHTS)
    if training_weights:
        weights.update(training_weights)
    weight = float(weights[tier])
    deepest = decision.deepest_accepted_node
    canonical_object: str | None = None
    if deepest:
        node = graph.node(deepest)
        deepest_level = _deepest_level(decision)
        exact_human_verified = bool(
            deepest_level is not None
            and deepest_level.state == DecisionState.HUMAN_VERIFIED
            and _verified_human(bundle)
            and bundle.human is not None
            and bundle.human.deepest_accepted_node == deepest
        )
        if not node.allowed_children and not decision.conflicts and exact_human_verified:
            canonical_object = deepest
    visual_attributes: dict[str, Any] = {}
    if bundle.visual_attributes:
        visual_attributes = {
            "entity_count": bundle.visual_attributes.entity_count,
            "colors": list(bundle.visual_attributes.colors),
            "forms": list(bundle.visual_attributes.forms),
            "parts": list(bundle.visual_attributes.parts),
            "orientations": list(bundle.visual_attributes.orientations),
            "material_like_cues": list(bundle.visual_attributes.material_like_cues),
            "uncertainty_terms": list(bundle.visual_attributes.uncertainty_terms),
        }
    caption = bundle.visual_description.caption_short if bundle.visual_description else None
    keywords = tuple(
        dict.fromkeys(
            [
                *(bundle.visual_attributes.colors if bundle.visual_attributes else ()),
                *(bundle.visual_attributes.forms if bundle.visual_attributes else ()),
                *(bundle.visual_attributes.parts if bundle.visual_attributes else ()),
            ]
        )
    )
    return SupervisionExport(
        bundle.record_identity,
        decision.identity,
        graph.identity,
        decision.taxonomy_path,
        deepest,
        canonical_object,
        None,
        technical_supervision(bundle.technical),
        visual_attributes,
        caption,
        keywords,
        decision.top_k_alternatives,
        _evidence_identities(bundle),
        decision.calibration_state,
        tier,
        weight,
    )
