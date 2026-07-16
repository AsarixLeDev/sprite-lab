from __future__ import annotations

from dataclasses import replace

import pytest
from PIL import Image

from hierarchical_labeling_support import write_sprite
from spritelab.hierarchical_labeling.cache import (
    NO_CONTEXT_IDENTITY,
    TECHNICAL_SCHEMA_IDENTITY,
    LabelingCacheIdentity,
)
from spritelab.hierarchical_labeling.contracts import ContextEvidence
from spritelab.hierarchical_labeling.json_utils import ContractValidationError
from spritelab.hierarchical_labeling.renders import (
    BALANCED_POLICY,
    FAST_LOCAL_POLICY,
    RenderType,
    audit_visual_only_payload,
    build_render_bundle,
    visual_only_provider_payload,
)
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph, load_default_taxonomy
from spritelab.hierarchical_labeling.technical import extract_technical_evidence


def test_taxonomy_valid_tree_and_required_example_paths() -> None:
    graph = load_default_taxonomy()
    assert len(graph.nodes) >= 20
    assert graph.path("sword") == ("entity", "object", "equipment", "weapon", "sword")
    assert graph.path("bottle") == ("entity", "object", "container", "bottle")


def test_taxonomy_rejects_cycle_or_disconnected_cycle() -> None:
    graph = load_default_taxonomy()
    nodes = {node.node_id: node for node in graph.nodes}
    nodes["object"] = replace(
        nodes["object"], allowed_children=tuple(x for x in nodes["object"].allowed_children if x != "equipment")
    )
    nodes["equipment"] = replace(nodes["equipment"], parent_id="sword")
    nodes["sword"] = replace(nodes["sword"], allowed_children=("equipment",))
    with pytest.raises(ContractValidationError):
        TaxonomyGraph(graph.version, nodes.values())


def test_taxonomy_rejects_orphan() -> None:
    graph = load_default_taxonomy()
    nodes = [replace(node, parent_id="missing_parent") if node.node_id == "bottle" else node for node in graph.nodes]
    with pytest.raises(ContractValidationError, match="orphan"):
        TaxonomyGraph(graph.version, nodes)


def test_taxonomy_rejects_duplicate_alias() -> None:
    graph = load_default_taxonomy()
    nodes = [replace(node, deprecated_aliases=("vial",)) if node.node_id == "sword" else node for node in graph.nodes]
    with pytest.raises(ContractValidationError, match="duplicate"):
        TaxonomyGraph(graph.version, nodes)


def test_taxonomy_deprecated_alias_lca_and_deepest_defensible() -> None:
    graph = load_default_taxonomy()
    assert graph.resolve("vial") == "bottle"
    assert graph.migrate_deprecated("vial")["current_node"] == "bottle"
    assert graph.lowest_common_ancestor("sword", "axe") == "weapon"
    evidence = dict.fromkeys(graph.path("bottle"), 0.9)
    evidence["bottle"] = 0.3
    assert graph.deepest_defensible_node(evidence, {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.8}) == "container"


def test_taxonomy_unknown_is_abstention_and_invalid_node_fails() -> None:
    graph = load_default_taxonomy()
    assert graph.resolve("unknown") is None
    assert graph.resolve("not_a_node") is None
    with pytest.raises(ContractValidationError, match="unknown taxonomy node"):
        graph.node("not_a_node")


def _cache_identity(**overrides: str) -> LabelingCacheIdentity:
    values = {
        "artifact_stage": "decision",
        "source_image_identity": "source",
        "decoded_rgba_identity": "rgba",
        "render_bundle_identity": "render",
        "provider_identity": "provider",
        "model_identity": "model",
        "prompt_identity": "prompt",
        "taxonomy_identity": "taxonomy-v1",
        "description_schema": "description",
        "hypothesis_schema": "hypothesis",
        "embedding_identity": "embedding",
        "retrieval_index_identity": "index",
        "reference_set_identity": "reference",
        "decision_policy_identity": "decision",
        "calibration_identity": "calibration",
        "metadata_identity": "metadata",
        "provider_configuration_identity": "provider-config",
        "reviewed_truth_identity": "truth",
        "context_identity": NO_CONTEXT_IDENTITY,
        "technical_evidence_identity": "technical-evidence",
        "technical_extraction_identity": "technical-extraction",
        "technical_schema_identity": TECHNICAL_SCHEMA_IDENTITY,
    }
    values.update(overrides)
    return LabelingCacheIdentity(**values)


def test_taxonomy_version_change_invalidates_cache_identity() -> None:
    assert _cache_identity().identity != _cache_identity(taxonomy_identity="taxonomy-v2").identity
    assert _cache_identity().legacy_cache_identity().namespace == "hierarchical_labeling_v2"


def _evidence_bound_cache(technical, context=None) -> LabelingCacheIdentity:
    dimensions = dict(_cache_identity().__dict__)
    for name in (
        "context_identity",
        "technical_evidence_identity",
        "technical_extraction_identity",
        "technical_schema_identity",
    ):
        dimensions.pop(name)
    return LabelingCacheIdentity.from_evidence(technical=technical, context=context, **dimensions)


def test_cache_factory_binds_full_context_and_technical_content(tmp_path) -> None:
    technical = extract_technical_evidence(write_sprite(tmp_path / "bound.png"), record_identity="r")
    no_context = _evidence_bound_cache(technical)
    assert no_context.context_identity == NO_CONTEXT_IDENTITY
    assert no_context.technical_evidence_identity == technical.identity
    assert no_context.technical_extraction_identity == technical.extraction_identity
    assert no_context.technical_schema_identity == TECHNICAL_SCHEMA_IDENTITY

    context = ContextEvidence("r", "pack-a", "pack", (("category", "container"),), True)
    with_context = _evidence_bound_cache(technical, context)
    assert with_context.context_identity == context.identity
    assert with_context.identity != no_context.identity
    assert (
        _evidence_bound_cache(technical, replace(context, permitted_by_policy=False)).identity != with_context.identity
    )
    assert (
        _evidence_bound_cache(technical, replace(context, claims=(("category", "weapon"),))).identity
        != with_context.identity
    )

    changed_feature = replace(technical.features[0], value=999)
    changed_technical = replace(technical, features=(changed_feature, *technical.features[1:]))
    assert _evidence_bound_cache(changed_technical).identity != no_context.identity
    assert (
        _evidence_bound_cache(replace(technical, extraction_identity="changed-extraction")).identity
        != no_context.identity
    )


def test_technical_dimensions_alpha_bbox_palette_components_duplicate_and_identity(tmp_path) -> None:
    path = write_sprite(tmp_path / "components.png", kind="components")
    first = extract_technical_evidence(path, record_identity="r", duplicate_cluster_identity="duplicate-a")
    second = extract_technical_evidence(path, record_identity="r", duplicate_cluster_identity="duplicate-a")
    assert first.feature("image_width") == 16 and first.feature("image_height") == 16
    assert first.feature("alpha_coverage") > 0
    assert first.feature("opaque_bounding_box") == [1, 1, 13, 13]
    assert first.feature("palette_size") >= 2
    assert first.feature("connected_component_count") == 4
    assert first.feature("duplicate_cluster_identity") == "duplicate-a"
    assert first.identity == second.identity


def test_technical_blank_and_sheet_detection(tmp_path) -> None:
    blank = extract_technical_evidence(write_sprite(tmp_path / "blank.png", kind="blank"), record_identity="blank")
    sheet_path = write_sprite(tmp_path / "sheet.png", kind="sheet", size=(64, 32))
    sheet = extract_technical_evidence(sheet_path, record_identity="sheet")
    assert blank.feature("empty_blank_status") is True
    assert sheet.feature("sheet_grid_status")["state"] == "likely_sheet_or_grid"


def test_technical_multiframe_status(tmp_path) -> None:
    path = tmp_path / "animated.gif"
    frames = [Image.new("RGBA", (8, 8), (255, 0, 0, 255)), Image.new("RGBA", (8, 8), (0, 0, 255, 255))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    evidence = extract_technical_evidence(path, record_identity="animated")
    assert evidence.feature("frame_count") == 2
    assert evidence.feature("animation_status") is True


def test_multiview_native_enlarged_checker_silhouette_crop_and_no_leakage(tmp_path) -> None:
    path = write_sprite(tmp_path / "secret-filename-bottle.png")
    technical = extract_technical_evidence(path, record_identity="r")
    bundle = build_render_bundle(path, technical, tmp_path / "renders", policy=FAST_LOCAL_POLICY, scale=4)
    types = {view.render_type for view in bundle.views}
    assert {
        RenderType.NATIVE.value,
        RenderType.ENLARGED.value,
        RenderType.CHECKERBOARD.value,
        RenderType.SILHOUETTE.value,
        RenderType.BOUNDING_BOX_CROP.value,
    }.issubset(types)
    payload = visual_only_provider_payload(bundle)
    assert all("artifact_path" not in row and "filename" not in row for row in payload)
    audit_visual_only_payload(payload)


def test_multiview_animation_sheet_context_and_identity_changes(tmp_path) -> None:
    source = write_sprite(tmp_path / "source.png")
    frame = write_sprite(tmp_path / "frame.png", shift=1)
    sheet = write_sprite(tmp_path / "sheet.png", kind="sheet", size=(64, 32))
    technical = extract_technical_evidence(source, record_identity="r")
    bundle = build_render_bundle(
        source,
        technical,
        tmp_path / "context",
        policy=BALANCED_POLICY,
        scale=4,
        sheet_context=sheet,
        animation_frames=(frame,),
    )
    assert RenderType.ANIMATION_CONTACT_SHEET.value in {view.render_type for view in bundle.views}
    assert RenderType.SOURCE_SHEET_CONTEXT.value in {view.render_type for view in bundle.views}
    changed = build_render_bundle(source, technical, tmp_path / "changed", policy=FAST_LOCAL_POLICY, scale=5)
    assert bundle.identity != changed.identity


def test_multiview_leakage_audit_rejects_filename() -> None:
    with pytest.raises(ContractValidationError, match="forbidden"):
        audit_visual_only_payload({"filename": "secret.png"})
