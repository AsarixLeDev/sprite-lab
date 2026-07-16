from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spritelab.hierarchical_labeling.calibration import CalibrationModel
from spritelab.hierarchical_labeling.contracts import (
    CalibrationState,
    LabelEvidenceBundle,
    MetadataEvidence,
    RetrievalEvidence,
    RetrievalNeighbor,
)
from spritelab.hierarchical_labeling.json_utils import content_identity
from spritelab.hierarchical_labeling.renders import FAST_LOCAL_POLICY, build_render_bundle
from spritelab.hierarchical_labeling.semantic import (
    DESCRIPTION_SCHEMA_VERSION,
    HYPOTHESIS_SCHEMA_VERSION,
    parse_semantic_hypotheses,
    parse_visual_description,
    structured_attributes,
)
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph, load_default_taxonomy
from spritelab.hierarchical_labeling.technical import extract_technical_evidence


def write_sprite(path: Path, *, kind: str = "bottle", shift: int = 0, size: tuple[int, int] = (16, 16)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if kind == "blank":
        pass
    elif kind == "bottle":
        draw.rectangle((6 + shift, 2, 9 + shift, 5), fill=(30, 40, 55, 255))
        draw.rectangle((4 + shift, 5, 11 + shift, 13), fill=(30, 40, 55, 255))
        draw.rectangle((5 + shift, 6, 10 + shift, 12), fill=(40, 100, 210, 255))
    elif kind == "sword":
        draw.rectangle((7 + shift, 1, 8 + shift, 12), fill=(190, 210, 225, 255))
        draw.rectangle((4 + shift, 11, 11 + shift, 12), fill=(45, 50, 65, 255))
        draw.rectangle((7 + shift, 13, 8 + shift, 15), fill=(120, 75, 40, 255))
    elif kind == "components":
        for x, y in ((1, 1), (10, 1), (1, 10), (10, 10)):
            draw.rectangle((x, y, x + 2, y + 2), fill=(80, 170, 90, 255))
    elif kind == "sheet":
        for y in range(2, size[1] - 3, 8):
            for x in range(2, size[0] - 3, 8):
                draw.rectangle((x, y, x + 2, y + 2), fill=(80, 170, 90, 255))
    image.save(path)
    return path


def description_value(*, kind: str = "bottle", exact_material: bool = False) -> dict[str, Any]:
    observation = (
        "upright blue container-like form with a narrow neck"
        if kind == "bottle"
        else "elongated blade-like form with a cross guard"
    )
    if exact_material:
        observation = "object made of metal"
    return {
        "schema_version": DESCRIPTION_SCHEMA_VERSION,
        "visible_observations": [observation],
        "visible_entities": ["one visible form"],
        "entity_count": 1,
        "shape_terms": ["upright" if kind == "bottle" else "elongated"],
        "visual_forms": [f"{kind}-like"],
        "dominant_colors": ["blue"],
        "secondary_colors": ["dark outline"],
        "material_like_cues": ["metal-like"] if kind == "sword" else [],
        "orientation": ["upright"],
        "symmetry": ["approximately vertical"],
        "visible_parts": ["outline", "interior region"],
        "possible_interpretations": [kind, "uncertain alternative"],
        "ambiguities": ["low-resolution ambiguity"],
        "resolution_limitations": ["few pixels"],
        "scene_or_icon_context": "isolated sprite icon",
        "caption_short": observation,
        "caption_detailed": f"Low-resolution isolated icon: {observation}.",
    }


def make_bundle(
    tmp_path: Path,
    *,
    record_identity: str = "record-1",
    node_id: str = "bottle",
    confidence: float = 0.9,
    abstention_recommended: bool = False,
    novelty: float = 0.1,
    neighbors: tuple[RetrievalNeighbor, ...] = (),
    metadata_node: str | None = None,
) -> tuple[LabelEvidenceBundle, TaxonomyGraph, Any]:
    graph = load_default_taxonomy()
    kind = "sword" if node_id in {"sword", "weapon", "equipment"} else "bottle"
    path = write_sprite(tmp_path / f"{record_identity}.png", kind=kind)
    technical = extract_technical_evidence(path, record_identity=record_identity)
    renders = build_render_bundle(path, technical, tmp_path / f"renders-{record_identity}", policy=FAST_LOCAL_POLICY)
    description = parse_visual_description(
        description_value(kind=kind),
        record_identity=record_identity,
        image_identity=technical.image_identity,
        render_bundle_identity=renders.identity,
        provider_identity="fake-provider",
        model_identity="fake-model",
        prompt_identity_value="description-prompt",
    )
    hypotheses = []
    for path_node in graph.path(node_id):
        deepest = path_node == node_id
        hypotheses.append(
            {
                "node_id": path_node,
                "depth": graph.depth(path_node),
                "rank": 1,
                "raw_model_confidence": confidence if deepest else 0.95,
                "evidence_citations": ["visible observation"],
                "contradicting_observations": ["ambiguous leaf"] if deepest and abstention_recommended else [],
                "abstention_recommended": deepest and abstention_recommended,
            }
        )
    hypothesis = parse_semantic_hypotheses(
        {
            "schema_version": HYPOTHESIS_SCHEMA_VERSION,
            "no_safe_hypothesis": False,
            "reason": "test fixture",
            "hypotheses": hypotheses,
        },
        record_identity=record_identity,
        graph=graph,
        description=description,
        provider_identity="fake-provider",
        model_identity="fake-model",
        prompt_identity_value="hypothesis-prompt",
        render_bundle_identity=renders.identity,
    )
    retrieval = (
        RetrievalEvidence(
            record_identity,
            technical.image_identity,
            "query-embedding",
            "index-identity",
            graph.identity,
            next(
                (neighbor.reference_cohort_identity for neighbor in neighbors if neighbor.review_status == "reviewed"),
                None,
            ),
            next(
                (neighbor.review_log_identity for neighbor in neighbors if neighbor.review_status == "reviewed"),
                None,
            ),
            neighbors,
            (("structural", 1.0),),
            novelty,
        )
        if neighbors
        else None
    )
    metadata = (
        MetadataEvidence(
            record_identity,
            content_identity("test-metadata-v1", {"record": record_identity}),
            (("category", metadata_node),),
            False,
        )
        if metadata_node
        else None
    )
    bundle = LabelEvidenceBundle(
        record_identity,
        technical.image_identity,
        graph.identity,
        technical,
        description,
        structured_attributes(description),
        (hypothesis,),
        retrieval,
        metadata,
    )
    return bundle, graph, renders


def calibration_model(graph: TaxonomyGraph, *, threshold: float = 0.5) -> CalibrationModel:
    return CalibrationModel(
        graph.identity,
        CalibrationState.VALIDATED_FOR_SCOPE,
        0.95,
        1,
        1,
        threshold,
        (),
        (),
        (),
        (),
        0,
        (),
        (),
        (),
        (),
    )
