from __future__ import annotations

import copy

import pytest

from hierarchical_labeling_support import make_bundle
from spritelab.hierarchical_labeling.cohort import (
    COHORT_SCHEMA_VERSION,
    CohortCandidate,
    CohortSelectionPolicy,
    cohort_membership,
    select_reference_cohort,
)
from spritelab.hierarchical_labeling.json_utils import ContractValidationError, content_identity
from spritelab.hierarchical_labeling.review import (
    GENESIS_EVENT_HASH,
    create_review_event,
    human_reference_label,
)


def _candidate(record_identity: str, image_identity: str, suffix: str) -> CohortCandidate:
    return CohortCandidate(
        record_identity,
        image_identity,
        f"cluster-{suffix}",
        f"duplicate-{suffix}",
        f"near-{suffix}",
        f"source-{suffix}",
        f"style-{suffix}",
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


def _reference_manifest(*candidates: CohortCandidate):
    return select_reference_cohort(
        candidates,
        dataset_identity="dataset",
        embedding_identity="embedding",
        clustering_identity="clustering",
        policy=CohortSelectionPolicy(
            target_size=len(candidates),
            reference_fraction=1.0,
            calibration_fraction=0.0,
            holdout_fraction=0.0,
        ),
    )


def test_truth_projection_requires_complete_chain_and_authoritative_event(tmp_path) -> None:
    bundle, graph, renders = make_bundle(tmp_path)
    manifest = _reference_manifest(_candidate(bundle.record_identity, bundle.image_identity, "a"))
    membership = cohort_membership(manifest, bundle.record_identity)
    first = create_review_event(
        bundle,
        graph,
        action="accept_suggested_path",
        reviewer_identity="reviewer",
        partition="reference",
        previous_event_hash=GENESIS_EVENT_HASH,
        selected_node="bottle",
        render_identities=(renders.identity,),
        timestamp="2026-07-15T00:00:00+00:00",
        submission_token="first",
    )
    second = create_review_event(
        bundle,
        graph,
        action="choose_parent",
        reviewer_identity="reviewer",
        partition="reference",
        previous_event_hash=first.event_hash,
        selected_node="container",
        render_identities=(renders.identity,),
        timestamp="2026-07-15T00:00:01+00:00",
        submission_token="second",
    )

    with pytest.raises(ContractValidationError, match="authoritative"):
        human_reference_label(first, graph=graph, verified_events=(first, second), membership=membership)
    with pytest.raises(ContractValidationError, match="complete valid hash chain"):
        human_reference_label(second, graph=graph, verified_events=(second,), membership=membership)
    with pytest.raises(ContractValidationError, match="complete valid hash chain"):
        human_reference_label(second, graph=graph, verified_events=(second, first), membership=membership)

    projected = human_reference_label(
        second,
        graph=graph,
        verified_events=(first, second),
        membership=membership,
    )
    assert projected.verified_append_only
    assert projected.deepest_accepted_node == "container"
    assert projected.verification is not None
    assert projected.verification.cohort_identity == manifest["cohort_identity"]
    assert projected.verification.render_identities == (renders.identity,)


def test_cohort_membership_rejects_content_tamper_and_recomputed_partition_leakage(tmp_path) -> None:
    first, _graph, _renders = make_bundle(tmp_path / "first", record_identity="first")
    second, _graph, _renders = make_bundle(tmp_path / "second", record_identity="second", node_id="sword")
    manifest = _reference_manifest(
        _candidate(first.record_identity, first.image_identity, "a"),
        _candidate(second.record_identity, second.image_identity, "b"),
    )

    tampered = copy.deepcopy(manifest)
    tampered["partitions"]["reference"][0]["source_identity"] = "forged-source"
    with pytest.raises(ContractValidationError, match="identity"):
        cohort_membership(tampered, first.record_identity)

    leaking = copy.deepcopy(manifest)
    moved = leaking["partitions"]["reference"].pop()
    moved["source_identity"] = leaking["partitions"]["reference"][0]["source_identity"]
    leaking["partitions"]["holdout"].append(moved)
    leaking["cohort_identity"] = content_identity(
        COHORT_SCHEMA_VERSION,
        {key: value for key, value in leaking.items() if key != "cohort_identity"},
    )
    with pytest.raises(ContractValidationError, match="leaks across"):
        cohort_membership(leaking, first.record_identity)

    image_leaking = copy.deepcopy(manifest)
    moved = image_leaking["partitions"]["reference"].pop()
    moved["image_identity"] = image_leaking["partitions"]["reference"][0]["image_identity"]
    image_leaking["partitions"]["calibration"].append(moved)
    image_leaking["cohort_identity"] = content_identity(
        COHORT_SCHEMA_VERSION,
        {key: value for key, value in image_leaking.items() if key != "cohort_identity"},
    )
    with pytest.raises(ContractValidationError, match="image identity leaks across"):
        cohort_membership(image_leaking, first.record_identity)
