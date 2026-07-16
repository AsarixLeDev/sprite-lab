"""Strict open-description and top-k taxonomy-hypothesis stages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from spritelab.hierarchical_labeling.contracts import (
    SemanticHypothesis,
    StructuredVisualAttributes,
    TaxonomyPathHypothesis,
    VisualDescription,
)
from spritelab.hierarchical_labeling.json_utils import ContractValidationError, content_identity
from spritelab.hierarchical_labeling.renders import MultiViewRenderBundle, audit_visual_only_payload
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph

DESCRIPTION_SCHEMA_VERSION = "spritelab.visual-description.provider-output.v1"
HYPOTHESIS_SCHEMA_VERSION = "spritelab.semantic-hypotheses.provider-output.v1"
DESCRIPTION_PROMPT_VERSION = "spritelab-open-visual-description-prompt-v1"
HYPOTHESIS_PROMPT_VERSION = "spritelab-hierarchical-top-k-prompt-v1"

DESCRIPTION_FIELDS = {
    "schema_version",
    "visible_observations",
    "visible_entities",
    "entity_count",
    "shape_terms",
    "visual_forms",
    "dominant_colors",
    "secondary_colors",
    "material_like_cues",
    "orientation",
    "symmetry",
    "visible_parts",
    "possible_interpretations",
    "ambiguities",
    "resolution_limitations",
    "scene_or_icon_context",
    "caption_short",
    "caption_detailed",
}
HYPOTHESIS_ITEM_FIELDS = {
    "node_id",
    "depth",
    "rank",
    "raw_model_confidence",
    "evidence_citations",
    "contradicting_observations",
    "abstention_recommended",
}


def visual_description_prompt() -> str:
    return (
        "Inspect only the supplied controlled sprite renders. Report what is visibly present before interpreting it. "
        "Keep visible_observations separate from possible_interpretations and ambiguities. Material words are visual "
        "cues only and must be one of metal-like, wood-like, stone-like, glass-like, fabric-like. Do not select a "
        "taxonomy label, use filenames or metadata, or silently turn uncertainty into an exact object claim. Return "
        f"strict JSON matching {DESCRIPTION_SCHEMA_VERSION}."
    )


def semantic_hypothesis_prompt(graph: TaxonomyGraph, description: VisualDescription, *, top_k: int = 3) -> str:
    if not 1 <= top_k <= 20:
        raise ContractValidationError("top_k must be from 1 through 20")
    taxonomy_summary = [
        {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "definition": node.definition,
            "positive_visual_criteria": list(node.positive_visual_criteria),
            "negative_visual_criteria": list(node.negative_visual_criteria),
        }
        for node in graph.nodes
    ]
    return (
        "Map the supplied visual description and controlled renders to ranked taxonomy hypotheses at each relevant "
        f"depth. Return at most {top_k} candidates per depth. Cite visible observations, record contradictions, and "
        "recommend abstention when no safe candidate exists. Raw confidence is not calibrated probability. Unknown "
        "is abstention and must never be emitted as a node. Return strict JSON matching "
        f"{HYPOTHESIS_SCHEMA_VERSION}. Taxonomy={taxonomy_summary!r}; description={description.to_dict()!r}"
    )


def prompt_identity(kind: str, prompt: str) -> str:
    return content_identity(kind, {"prompt": prompt})


def _string_array(value: Mapping[str, Any], name: str) -> tuple[str, ...]:
    raw = value.get(name)
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() == item and item for item in raw):
        raise ContractValidationError(f"description field {name} must be an array of normalized strings")
    if len(raw) != len(set(raw)):
        raise ContractValidationError(f"description field {name} cannot contain duplicates")
    return tuple(raw)


def _reject_material_overclaim(value: Mapping[str, Any]) -> None:
    exact_claim = re.compile(r"\b(?:made\s+of\s+)?(?:metal|wood|stone|glass|fabric)(?!-like)\b", re.IGNORECASE)
    for field in ("visible_observations", "caption_short", "caption_detailed"):
        raw = value.get(field, ())
        texts = raw if isinstance(raw, list) else (raw,)
        if any(isinstance(text, str) and exact_claim.search(text) for text in texts):
            raise ContractValidationError("visual description makes an exact material claim instead of a -like cue")


def parse_visual_description(
    value: Mapping[str, Any],
    *,
    record_identity: str,
    image_identity: str,
    render_bundle_identity: str,
    provider_identity: str,
    model_identity: str,
    prompt_identity_value: str,
) -> VisualDescription:
    if set(value) != DESCRIPTION_FIELDS or value.get("schema_version") != DESCRIPTION_SCHEMA_VERSION:
        raise ContractValidationError("provider visual description does not match the exact schema")
    _reject_material_overclaim(value)
    entity_count = value.get("entity_count")
    if entity_count is not None and (type(entity_count) is not int or entity_count < 0):
        raise ContractValidationError("description entity_count must be a non-negative integer or null")
    for name in ("scene_or_icon_context", "caption_short", "caption_detailed"):
        if not isinstance(value.get(name), str) or not str(value[name]).strip():
            raise ContractValidationError(f"description field {name} must be non-empty text")
    description = VisualDescription(
        record_identity,
        image_identity,
        render_bundle_identity,
        provider_identity,
        model_identity,
        prompt_identity_value,
        _string_array(value, "visible_observations"),
        _string_array(value, "visible_entities"),
        entity_count,
        _string_array(value, "shape_terms"),
        _string_array(value, "visual_forms"),
        _string_array(value, "dominant_colors"),
        _string_array(value, "secondary_colors"),
        _string_array(value, "material_like_cues"),
        _string_array(value, "orientation"),
        _string_array(value, "symmetry"),
        _string_array(value, "visible_parts"),
        _string_array(value, "possible_interpretations"),
        _string_array(value, "ambiguities"),
        _string_array(value, "resolution_limitations"),
        value["scene_or_icon_context"],
        value["caption_short"],
        value["caption_detailed"],
    )
    if not description.visible_observations:
        raise ContractValidationError("visual description must preserve at least one visible observation")
    return description


def structured_attributes(description: VisualDescription) -> StructuredVisualAttributes:
    return StructuredVisualAttributes(
        description.identity,
        description.image_identity,
        description.entity_count,
        tuple(dict.fromkeys((*description.dominant_colors, *description.secondary_colors))),
        tuple(dict.fromkeys((*description.shape_terms, *description.visual_forms))),
        description.visible_parts,
        description.orientation,
        description.material_like_cues,
        tuple(dict.fromkeys((*description.ambiguities, *description.resolution_limitations))),
    )


def parse_semantic_hypotheses(
    value: Mapping[str, Any],
    *,
    record_identity: str,
    graph: TaxonomyGraph,
    description: VisualDescription,
    provider_identity: str,
    model_identity: str,
    prompt_identity_value: str,
    render_bundle_identity: str,
    maximum_per_depth: int = 3,
) -> TaxonomyPathHypothesis:
    if set(value) != {"schema_version", "no_safe_hypothesis", "reason", "hypotheses"}:
        raise ContractValidationError("provider hypotheses do not match the exact schema")
    if value.get("schema_version") != HYPOTHESIS_SCHEMA_VERSION:
        raise ContractValidationError("provider hypothesis schema version is unsupported")
    no_safe = value.get("no_safe_hypothesis")
    if type(no_safe) is not bool:
        raise ContractValidationError("no_safe_hypothesis must be a JSON boolean")
    raw_items = value.get("hypotheses")
    if not isinstance(raw_items, list):
        raise ContractValidationError("hypotheses must be a JSON array")
    reason = value.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ContractValidationError("hypothesis reason must be non-empty text or null")
    if no_safe:
        if raw_items:
            raise ContractValidationError("no_safe_hypothesis cannot include candidates")
        return TaxonomyPathHypothesis(
            record_identity,
            graph.identity,
            description.identity,
            (),
            (),
            True,
            reason or "provider_reported_no_safe_hypothesis",
        )
    hypotheses: list[SemanticHypothesis] = []
    nodes_seen: set[str] = set()
    depth_ranks: set[tuple[int, int]] = set()
    depth_counts: dict[int, int] = {}
    for raw in raw_items:
        if not isinstance(raw, Mapping) or set(raw) != HYPOTHESIS_ITEM_FIELDS:
            raise ContractValidationError("a semantic hypothesis item does not match the exact schema")
        node_id = raw.get("node_id")
        resolved = graph.resolve(node_id if isinstance(node_id, str) else None)
        if resolved is None or resolved != node_id:
            raise ContractValidationError(f"hypothesis contains invalid, unknown, or deprecated node {node_id!r}")
        depth = raw.get("depth")
        rank = raw.get("rank")
        if type(depth) is not int or depth != graph.depth(resolved) or type(rank) is not int or rank < 1:
            raise ContractValidationError("hypothesis depth/rank is invalid for the taxonomy node")
        if resolved in nodes_seen or (depth, rank) in depth_ranks:
            raise ContractValidationError("hypotheses cannot repeat a node or depth/rank pair")
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        if depth_counts[depth] > maximum_per_depth:
            raise ContractValidationError("provider exceeded the configured top-k per taxonomy depth")
        confidence = raw.get("raw_model_confidence")
        citations = raw.get("evidence_citations")
        contradictions = raw.get("contradicting_observations")
        abstention = raw.get("abstention_recommended")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))):
            raise ContractValidationError("raw model confidence must be numeric or null")
        if not isinstance(citations, list) or not all(isinstance(item, str) and item for item in citations):
            raise ContractValidationError("hypothesis evidence citations must be strings")
        if not isinstance(contradictions, list) or not all(isinstance(item, str) and item for item in contradictions):
            raise ContractValidationError("hypothesis contradictions must be strings")
        if type(abstention) is not bool:
            raise ContractValidationError("hypothesis abstention recommendation must be a boolean")
        hypotheses.append(
            SemanticHypothesis(
                resolved,
                depth,
                rank,
                float(confidence) if confidence is not None else None,
                tuple(citations),
                tuple(contradictions),
                abstention,
                provider_identity,
                model_identity,
                prompt_identity_value,
                render_bundle_identity,
                graph.identity,
            )
        )
        nodes_seen.add(resolved)
        depth_ranks.add((depth, rank))
    if not hypotheses:
        raise ContractValidationError("a non-abstained hypothesis result requires candidates")
    deepest_rank_one = max(
        (item for item in hypotheses if item.rank == 1),
        key=lambda item: item.depth,
        default=None,
    )
    if deepest_rank_one is None:
        raise ContractValidationError("top-k hypotheses require a rank-one candidate")
    path = graph.path(deepest_rank_one.node_id)
    rank_one_by_depth = {item.depth: item.node_id for item in hypotheses if item.rank == 1}
    for node_id in path:
        depth = graph.depth(node_id)
        if depth in rank_one_by_depth and rank_one_by_depth[depth] != node_id:
            raise ContractValidationError("rank-one hypotheses do not form a compatible hierarchy path")
    return TaxonomyPathHypothesis(
        record_identity,
        graph.identity,
        description.identity,
        path,
        tuple(sorted(hypotheses, key=lambda item: (item.depth, item.rank, item.node_id))),
        False,
        reason,
    )


def visual_stage_payload(bundle: MultiViewRenderBundle, description: VisualDescription | None = None) -> dict[str, Any]:
    """Build an audited semantic-stage payload without metadata/retrieval labels."""

    payload = {
        "render_bundle_identity": bundle.identity,
        "views": [view.blind_manifest() for view in bundle.views],
        "description": description.to_dict() if description else None,
    }
    audit_visual_only_payload(payload)
    return payload
