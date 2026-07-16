from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from hierarchical_labeling_support import description_value, make_bundle, write_sprite
from spritelab.hierarchical_labeling.cohort import (
    CohortCandidate,
    CohortSelectionPolicy,
    cohort_membership,
    select_reference_cohort,
)
from spritelab.hierarchical_labeling.json_utils import ContractValidationError
from spritelab.hierarchical_labeling.renders import FAST_LOCAL_POLICY, build_render_bundle
from spritelab.hierarchical_labeling.retrieval import (
    DeterministicMockEmbeddingBackend,
    EmbeddingSample,
    EmbeddingStore,
    EmbeddingVector,
    ExactRetrievalIndex,
    RetrievalIndexRecord,
    StructuralEmbeddingBackend,
)
from spritelab.hierarchical_labeling.review import (
    GENESIS_EVENT_HASH,
    create_review_event,
    human_reference_label,
    synthetic_oracle_reference_label,
)
from spritelab.hierarchical_labeling.semantic import (
    HYPOTHESIS_SCHEMA_VERSION,
    parse_semantic_hypotheses,
    parse_visual_description,
    structured_attributes,
)
from spritelab.hierarchical_labeling.technical import extract_technical_evidence


def _description(tmp_path):
    bundle, graph, renders = make_bundle(tmp_path)
    return bundle.visual_description, graph, renders


def test_description_valid_observation_interpretation_ambiguity_and_material_cue(tmp_path) -> None:
    bundle, _graph, _renders = make_bundle(tmp_path, node_id="sword")
    description = bundle.visual_description
    assert description.visible_observations != description.possible_interpretations
    assert description.ambiguities
    assert description.material_like_cues == ("metal-like",)
    attributes = structured_attributes(description)
    assert "metal-like" in attributes.material_like_cues


def test_description_invalid_structured_output_is_not_coerced(tmp_path) -> None:
    bundle, _graph, renders = make_bundle(tmp_path)
    raw = description_value()
    raw.pop("visible_entities")
    with pytest.raises(ContractValidationError, match="exact schema"):
        parse_visual_description(
            raw,
            record_identity="r",
            image_identity=bundle.image_identity,
            render_bundle_identity=renders.identity,
            provider_identity="fake",
            model_identity="fake",
            prompt_identity_value="prompt",
        )


def test_description_exact_material_overclaim_rejected(tmp_path) -> None:
    bundle, _graph, renders = make_bundle(tmp_path)
    with pytest.raises(ContractValidationError, match="exact material"):
        parse_visual_description(
            description_value(exact_material=True),
            record_identity="r",
            image_identity=bundle.image_identity,
            render_bundle_identity=renders.identity,
            provider_identity="fake",
            model_identity="fake",
            prompt_identity_value="prompt",
        )


def _hypothesis_raw(description, graph, node_id="bottle"):
    return {
        "schema_version": HYPOTHESIS_SCHEMA_VERSION,
        "no_safe_hypothesis": False,
        "reason": "test",
        "hypotheses": [
            {
                "node_id": node,
                "depth": graph.depth(node),
                "rank": 1,
                "raw_model_confidence": 0.9,
                "evidence_citations": [description.visible_observations[0]],
                "contradicting_observations": [],
                "abstention_recommended": False,
            }
            for node in graph.path(node_id)
        ],
    }


def _parse(raw, description, graph, renders):
    return parse_semantic_hypotheses(
        raw,
        record_identity=description.record_identity,
        graph=graph,
        description=description,
        provider_identity="fake",
        model_identity="fake",
        prompt_identity_value="prompt",
        render_bundle_identity=renders.identity,
    )


def test_top_k_hypothesis_valid_and_hierarchy_path(tmp_path) -> None:
    description, graph, renders = _description(tmp_path)
    parsed = _parse(_hypothesis_raw(description, graph), description, graph, renders)
    assert parsed.path == graph.path("bottle")
    assert all(item.rank == 1 for item in parsed.hypotheses)


@pytest.mark.parametrize("invalid_node", ["unknown", "not_a_node", "vial"])
def test_hypothesis_invalid_unknown_or_deprecated_node_fails(tmp_path, invalid_node) -> None:
    description, graph, renders = _description(tmp_path)
    raw = _hypothesis_raw(description, graph)
    raw["hypotheses"][-1]["node_id"] = invalid_node
    with pytest.raises(ContractValidationError):
        _parse(raw, description, graph, renders)


def test_hypothesis_duplicate_and_nonfinite_confidence_fail(tmp_path) -> None:
    description, graph, renders = _description(tmp_path)
    duplicate = _hypothesis_raw(description, graph)
    duplicate["hypotheses"].append(copy.deepcopy(duplicate["hypotheses"][-1]))
    with pytest.raises(ContractValidationError, match="repeat"):
        _parse(duplicate, description, graph, renders)
    nonfinite = _hypothesis_raw(description, graph)
    nonfinite["hypotheses"][-1]["raw_model_confidence"] = float("nan")
    with pytest.raises(ContractValidationError):
        _parse(nonfinite, description, graph, renders)


def test_hypothesis_no_safe_state(tmp_path) -> None:
    description, graph, renders = _description(tmp_path)
    result = _parse(
        {
            "schema_version": HYPOTHESIS_SCHEMA_VERSION,
            "no_safe_hypothesis": True,
            "reason": "no defensible node",
            "hypotheses": [],
        },
        description,
        graph,
        renders,
    )
    assert result.no_safe_hypothesis and not result.path


def test_hypothesis_incompatible_rank_one_hierarchy_fails(tmp_path) -> None:
    description, graph, renders = _description(tmp_path)
    raw = _hypothesis_raw(description, graph)
    raw["hypotheses"][2]["node_id"] = "equipment"
    with pytest.raises(ContractValidationError, match="compatible"):
        _parse(raw, description, graph, renders)


def _embedding_fixture(tmp_path):
    samples = []
    for index, kind in enumerate(("bottle", "bottle", "sword")):
        path = write_sprite(tmp_path / f"{index}.png", kind=kind, shift=1 if index == 1 else 0)
        technical = extract_technical_evidence(path, record_identity=f"r{index}")
        renders = build_render_bundle(path, technical, tmp_path / f"render-{index}", policy=FAST_LOCAL_POLICY)
        samples.append(EmbeddingSample(f"r{index}", technical.image_identity, technical, renders.views))
    return tuple(samples)


def _verified_reference(tmp_path, *, record_identity: str = "r0"):
    bundle, graph, renders = make_bundle(tmp_path, record_identity=record_identity)
    candidate = CohortCandidate(
        record_identity,
        bundle.image_identity,
        f"cluster-{record_identity}",
        f"duplicate-{record_identity}",
        f"near-{record_identity}",
        f"source-{record_identity}",
        "style",
        "16x16",
        1,
        True,
        0.1,
        0.1,
        False,
        0.1,
        False,
        False,
        False,
        True,
        True,
    )
    manifest = select_reference_cohort(
        (candidate,),
        dataset_identity="dataset",
        embedding_identity="embedding",
        clustering_identity="clustering",
        policy=CohortSelectionPolicy(
            target_size=1,
            reference_fraction=1.0,
            calibration_fraction=0.0,
            holdout_fraction=0.0,
        ),
    )
    event = create_review_event(
        bundle,
        graph,
        action="accept_suggested_path",
        reviewer_identity="reviewer",
        partition="reference",
        previous_event_hash=GENESIS_EVENT_HASH,
        selected_node="bottle",
        render_identities=(renders.identity,),
        timestamp="2026-07-15T00:00:00+00:00",
        submission_token=f"verified:{record_identity}",
    )
    reference = human_reference_label(
        event,
        graph=graph,
        verified_events=(event,),
        membership=cohort_membership(manifest, record_identity),
    )
    return reference, bundle, graph


def test_retrieval_deterministic_embeddings_and_cache_changed_image_encoder_delete(tmp_path) -> None:
    samples = _embedding_fixture(tmp_path)
    backend = StructuralEmbeddingBackend()
    vectors = (*backend.embed_images(samples), *backend.embed_views(samples))
    repeated = (*StructuralEmbeddingBackend().embed_images(samples), *StructuralEmbeddingBackend().embed_views(samples))
    assert [item.vector for item in vectors] == [item.vector for item in repeated]
    store = EmbeddingStore(tmp_path / "embeddings.sqlite")
    assert store.put_batch(vectors) == len(vectors)
    cached = store.get("r0", image_identity=samples[0].image_identity, backend_identity=backend.cache_identity)
    assert len(cached) == 3
    assert not store.get("r0", image_identity="changed-image", backend_identity=backend.cache_identity)
    changed_encoder = StructuralEmbeddingBackend()
    changed_encoder.model_identity = "changed-model"
    assert changed_encoder.cache_identity != backend.cache_identity
    assert store.delete_missing(("r0", "r1"), backend_identity=backend.cache_identity) == 1


def test_retrieval_neighbors_truth_separation_clusters_medoids_and_novelty(tmp_path) -> None:
    samples = _embedding_fixture(tmp_path)
    backend = DeterministicMockEmbeddingBackend(dimensions=8)
    vectors = backend.embed_images(samples)
    human, reviewed_bundle, graph = _verified_reference(tmp_path / "review", record_identity="r0")
    assert reviewed_bundle.image_identity == vectors[0].image_identity
    assert human.verification is not None
    reference_cohort_identity = human.verification.cohort_identity
    records = (
        RetrievalIndexRecord.from_vectors((vectors[0],), taxonomy_identity=graph.identity, human_label=human),
        RetrievalIndexRecord.from_vectors(
            (vectors[1],),
            taxonomy_identity=graph.identity,
            reference_cohort_identity=reference_cohort_identity,
            proposal_taxonomy_path=graph.path("bottle"),
        ),
        RetrievalIndexRecord.from_vectors(
            (vectors[2],), taxonomy_identity=graph.identity, reference_cohort_identity=reference_cohort_identity
        ),
    )
    index = ExactRetrievalIndex(records, backend_identity=backend.cache_identity, fusion_weights={"mock": 1.0})
    neighbors = index.nearest_neighbors({"mock": vectors[0].vector}, k=3)
    reviewed = next(item for item in neighbors if item.review_status == "reviewed")
    proposal = next(item for item in neighbors if item.review_status == "proposal")
    assert reviewed.verified_taxonomy_path == graph.path("bottle")
    assert not proposal.verified_taxonomy_path and proposal.proposal_taxonomy_path
    assignments = index.cluster_assignments(similarity_threshold=0.0)
    medoids = index.cluster_medoids(assignments)
    assert medoids and set(medoids.values()).issubset({"r0", "r1", "r2"})
    assert 0 <= index.novelty_score({"mock": vectors[0].vector}) <= 1


def test_authoritative_retrieval_rejects_unverified_or_nonreference_truth(tmp_path) -> None:
    reference, bundle, graph = _verified_reference(tmp_path, record_identity="r")
    assert reference.verification is not None
    verification = reference.verification
    vector = EmbeddingVector("r", bundle.image_identity, "backend", "model", "mock", (1.0, 0.0))
    indexed = RetrievalIndexRecord.from_vectors((vector,), taxonomy_identity=graph.identity, human_label=reference)
    assert indexed.review_status == "reviewed"

    direct = replace(reference, verification=None)
    with pytest.raises(ContractValidationError, match="verified reference-partition"):
        RetrievalIndexRecord.from_vectors((vector,), taxonomy_identity=graph.identity, human_label=direct)

    for partition in ("calibration", "holdout"):
        changed_verification = replace(verification, partition=partition)
        changed = replace(reference, partition=partition, verification=changed_verification)
        with pytest.raises(ContractValidationError, match="reference-partition"):
            RetrievalIndexRecord.from_vectors((vector,), taxonomy_identity=graph.identity, human_label=changed)

    foreign_verification = replace(verification, taxonomy_identity="foreign-taxonomy")
    foreign = replace(reference, taxonomy_identity="foreign-taxonomy", verification=foreign_verification)
    with pytest.raises(ContractValidationError, match=r"taxonomy|verified"):
        RetrievalIndexRecord.from_vectors((vector,), taxonomy_identity=graph.identity, human_label=foreign)

    oracle = synthetic_oracle_reference_label(
        record_identity="r",
        taxonomy_identity=graph.identity,
        taxonomy_path=graph.path("bottle"),
        deepest_accepted_node="bottle",
        explicit_abstentions=(),
        partition="reference",
        oracle_set_identity="oracle-set",
        image_identity="image-r",
        evidence_bundle_identity="bundle-r",
        cohort_identity="oracle-cohort",
        source_identity="source-r",
        cluster_identity="cluster-r",
        leakage_group_identity="group-r",
    )
    with pytest.raises(ContractValidationError, match="synthetic oracle"):
        RetrievalIndexRecord.from_vectors((vector,), taxonomy_identity=graph.identity, human_label=oracle)  # type: ignore[arg-type]


def test_retrieval_50000_scalability_contract_without_allocation() -> None:
    expected = 50_000 * (13 + 64 + 64) * 4
    assert (
        ExactRetrievalIndex.estimated_memory_bytes(
            50_000, {"technical_feature": 13, "alpha_silhouette": 64, "palette_composition": 64}
        )
        == expected
    )
    assert expected < 100_000_000


def _cohort_candidates() -> list[CohortCandidate]:
    result = []
    for index in range(12):
        result.append(
            CohortCandidate(
                f"r{index}",
                f"i{index}",
                f"cluster-{index}",
                "duplicate-shared" if index in {0, 1} else f"duplicate-{index}",
                f"near-{index // 3}" if index < 6 else None,
                f"source-{index}",
                f"style-{index % 2}",
                f"size-{index % 3}",
                20 if index % 4 == 0 else 2,
                index % 4 == 0,
                0.9 if index == 11 else 0.2,
                0.8 if index == 10 else 0.2,
                index == 9,
                0.7 if index == 8 else 0.1,
                index == 7,
                index == 6,
                index == 11,
                index != 5,
                True,
            )
        )
    return result


def test_reference_cohort_medoids_rare_source_diversity_duplicates_and_partitions() -> None:
    candidates = _cohort_candidates()
    policy = CohortSelectionPolicy(target_size=8, seed=17)
    first = select_reference_cohort(
        candidates,
        dataset_identity="dataset",
        embedding_identity="embedding",
        clustering_identity="clustering",
        policy=policy,
    )
    second = select_reference_cohort(
        candidates,
        dataset_identity="dataset",
        embedding_identity="embedding",
        clustering_identity="clustering",
        policy=policy,
    )
    assert first == second
    rows = [row for values in first["partitions"].values() for row in values]
    assert len({row["record_identity"] for row in rows}) == len(rows)
    assert len({row["source_identity"] for row in rows}) >= 2
    assert not {"r0", "r1"}.issubset({row["record_identity"] for row in rows})
    assert first["partition_identities_disjoint"] is True
    for field in (
        "source_identity",
        "cluster_identity",
        "duplicate_cluster_identity",
        "near_duplicate_cluster_identity",
        "leakage_group_identity",
    ):
        owners = {}
        for partition, values in first["partitions"].items():
            for row in values:
                value = row[field]
                if value is not None:
                    assert owners.setdefault(value, partition) == partition
    assert first["human_labels_created"] == 0
