from __future__ import annotations

from dataclasses import replace

import pytest

from hierarchical_labeling_support import calibration_model, make_bundle
from spritelab.hierarchical_labeling.calibration import CalibrationExample, fit_calibration
from spritelab.hierarchical_labeling.cohort import (
    CohortCandidate,
    CohortSelectionPolicy,
    cohort_membership,
    select_reference_cohort,
)
from spritelab.hierarchical_labeling.contracts import (
    CalibrationState,
    DecisionState,
    HumanReferenceLabel,
    HumanTruthVerification,
    SupervisionTier,
)
from spritelab.hierarchical_labeling.decision import (
    SYNTHETIC_ORACLE_CONFLICT,
    decide_hierarchical_label,
)
from spritelab.hierarchical_labeling.json_utils import ContractValidationError
from spritelab.hierarchical_labeling.reporting import build_report_data
from spritelab.hierarchical_labeling.retrieval import EmbeddingVector, RetrievalIndexRecord
from spritelab.hierarchical_labeling.review import (
    GENESIS_EVENT_HASH,
    create_review_event,
    human_reference_label,
    synthetic_oracle_reference_label,
)
from spritelab.hierarchical_labeling.supervision import export_supervision


def _reviewed_bundle(
    tmp_path,
    *,
    candidate_node: str,
    selected_node: str,
    explicit_abstentions: tuple[str, ...] = (),
    partition: str = "reference",
):
    bundle, graph, renders = make_bundle(tmp_path, node_id=candidate_node)
    event = create_review_event(
        bundle,
        graph,
        action="choose_parent" if selected_node != candidate_node else "accept_suggested_path",
        reviewer_identity="reviewer-a",
        partition=partition,
        previous_event_hash=GENESIS_EVENT_HASH,
        selected_node=selected_node,
        explicit_abstentions=explicit_abstentions,
        render_identities=(renders.identity,),
        timestamp="2026-07-15T00:00:00+00:00",
        submission_token=f"{candidate_node}:{selected_node}:{partition}",
    )
    fractions = {
        "reference": (1.0, 0.0, 0.0),
        "calibration": (0.0, 1.0, 0.0),
        "holdout": (0.0, 0.0, 1.0),
    }[partition]
    candidate = CohortCandidate(
        bundle.record_identity,
        bundle.image_identity,
        "cluster-a",
        "duplicate-a",
        "near-a",
        "source-a",
        "style-a",
        "size-a",
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
        dataset_identity="dataset-a",
        embedding_identity="embedding-a",
        clustering_identity="clustering-a",
        policy=CohortSelectionPolicy(
            target_size=1,
            reference_fraction=fractions[0],
            calibration_fraction=fractions[1],
            holdout_fraction=fractions[2],
        ),
    )
    human = human_reference_label(
        event,
        graph=graph,
        verified_events=(event,),
        membership=cohort_membership(manifest, bundle.record_identity),
    )
    return replace(bundle, human=human), graph, bundle


def _report_record(record_identity: str, path: tuple[str, ...]) -> dict[str, object]:
    return {
        "record_identity": record_identity,
        "accepted_path": list(path),
        "abstained": not path,
        "source_identity": "source-a",
        "cluster_identity": "cluster-a",
    }


def _truth_row(bundle, path: tuple[str, ...]) -> dict[str, object]:
    assert bundle.human is not None and bundle.human.verification is not None
    return {
        "record_identity": bundle.record_identity,
        "evidence_bundle_identity": bundle.human.verification.evidence_bundle_identity,
        "predicted_path": list(path),
        "calibrated_probability": 0.9,
        "source_identity": bundle.human.verification.source_identity,
        "cluster_identity": bundle.human.verification.cluster_identity,
        "leakage_group_identity": bundle.human.verification.leakage_group_identity,
        "human_reference": bundle.human,
    }


def _fabricated_human_reference(bundle, graph, *, partition: str) -> HumanReferenceLabel:
    event_identity = "1" * 64
    verification = HumanTruthVerification(
        "append_only_human_review",
        bundle.record_identity,
        graph.identity,
        graph.path("bottle"),
        (),
        partition,
        "fabricated-reviewer",
        event_identity,
        "2" * 64,
        "3" * 64,
        bundle.image_identity,
        ("fabricated-render",),
        bundle.nonhuman_identity,
        "fabricated-cohort",
        "fabricated-source",
        "fabricated-cluster",
        "fabricated-leakage-group",
    )
    return HumanReferenceLabel(
        bundle.record_identity,
        event_identity,
        graph.identity,
        graph.path("bottle"),
        "bottle",
        (),
        partition,
        "fabricated-reviewer",
        verification,
    )


def test_verified_abstention_blocks_the_named_node_and_every_descendant(tmp_path) -> None:
    bundle, graph, _nonhuman = _reviewed_bundle(
        tmp_path,
        candidate_node="sword",
        selected_node="equipment",
        explicit_abstentions=("weapon",),
    )
    decision = decide_hierarchical_label(bundle, graph, calibration_model(graph))

    assert decision.taxonomy_path == graph.path("equipment")
    assert decision.deepest_accepted_node == "equipment"
    assert decision.abstained_below_node == "weapon"
    states = {level.node_id: level.state for level in decision.level_decisions}
    assert states["equipment"] == DecisionState.HUMAN_VERIFIED
    assert states["weapon"] == DecisionState.HUMAN_ABSTAINED
    assert states["sword"] == DecisionState.HUMAN_ABSTAINED
    assert all(
        "verified human abstention at weapon" in " ".join(level.reasons)
        for level in decision.level_decisions
        if level.node_id in {"weapon", "sword"}
    )
    exported = export_supervision(bundle, decision, graph)
    assert exported.canonical_object is None
    assert exported.supervision_tier == SupervisionTier.HUMAN_VERIFIED


def test_model_leaf_below_human_parent_uses_deepest_model_tier(tmp_path) -> None:
    bundle, graph, _nonhuman = _reviewed_bundle(
        tmp_path,
        candidate_node="container",
        selected_node="object",
    )
    decision = decide_hierarchical_label(bundle, graph, calibration_model(graph))

    assert decision.deepest_accepted_node == "container"
    assert decision.level_decisions[-1].state == DecisionState.ACCEPTED
    exported = export_supervision(bundle, decision, graph)
    assert exported.supervision_tier == SupervisionTier.CALIBRATED_CORE
    assert exported.recommended_training_weight < 1.0
    assert exported.canonical_object is None


def test_supervision_rejects_decision_and_level_evidence_identity_spoofing(tmp_path) -> None:
    bundle, graph, _renders = make_bundle(tmp_path)
    decision = decide_hierarchical_label(bundle, graph, calibration_model(graph))

    with pytest.raises(ContractValidationError, match="does not bind"):
        export_supervision(bundle, replace(decision, evidence_bundle_identity="forged-bundle"), graph)

    forged_level = replace(decision.level_decisions[-1], evidence_bundle_identity="forged-bundle")
    forged_levels = (*decision.level_decisions[:-1], forged_level)
    with pytest.raises(ContractValidationError, match="does not bind"):
        export_supervision(bundle, replace(decision, level_decisions=forged_levels), graph)

    forged_human_abstention = replace(decision.level_decisions[-1], state=DecisionState.HUMAN_ABSTAINED)
    forged_levels = (*decision.level_decisions[:-1], forged_human_abstention)
    forged_decision = replace(
        decision,
        taxonomy_path=decision.taxonomy_path[:-1],
        deepest_accepted_node=decision.taxonomy_path[-2],
        abstained_below_node=decision.taxonomy_path[-1],
        level_decisions=forged_levels,
    )
    with pytest.raises(ContractValidationError, match="human-abstained"):
        export_supervision(bundle, forged_decision, graph)


def test_decision_revalidates_semantic_provider_binding_at_trust_boundary(tmp_path) -> None:
    bundle, graph, _renders = make_bundle(tmp_path)
    path_hypothesis = bundle.taxonomy_hypotheses[0]
    object.__setattr__(path_hypothesis.hypotheses[-1], "provider_identity", "foreign-provider")

    with pytest.raises(ContractValidationError, match="provider"):
        decide_hierarchical_label(bundle, graph, calibration_model(graph))


def test_raw_truth_mapping_cannot_enable_verified_precision(tmp_path) -> None:
    bundle, graph, _renders = make_bundle(tmp_path)
    records = (_report_record(bundle.record_identity, graph.path("bottle")),)
    raw = {
        "record_identity": bundle.record_identity,
        "predicted_path": list(graph.path("bottle")),
        "truth_path": list(graph.path("bottle")),
        "calibrated_probability": 0.9,
        "truth_source": "human_review",
        "source_identity": "source-a",
        "cluster_identity": "cluster-a",
    }

    with pytest.raises(ContractValidationError, match="raw truth mappings"):
        build_report_data(records, graph, truth_rows=(raw,))


def test_self_attested_verification_cannot_cross_any_human_truth_boundary(tmp_path) -> None:
    bundle, graph, _renders = make_bundle(tmp_path)

    calibration_label = _fabricated_human_reference(bundle, graph, partition="calibration")
    assert not calibration_label.verified_append_only
    example = CalibrationExample(
        bundle.record_identity,
        "bottle",
        1.0,
        bundle.nonhuman_identity,
        "fabricated-source",
        "fabricated-cluster",
        calibration_label,
        leakage_group_identity="fabricated-leakage-group",
    )
    with pytest.raises(ContractValidationError, match="verified append-only"):
        fit_calibration((example,), graph, minimum_global_samples=1, minimum_class_samples=1)

    holdout_label = _fabricated_human_reference(bundle, graph, partition="holdout")
    records = (_report_record(bundle.record_identity, graph.path("bottle")),)
    row = {
        "record_identity": bundle.record_identity,
        "evidence_bundle_identity": bundle.nonhuman_identity,
        "predicted_path": list(graph.path("bottle")),
        "calibrated_probability": 1.0,
        "source_identity": "fabricated-source",
        "cluster_identity": "fabricated-cluster",
        "leakage_group_identity": "fabricated-leakage-group",
        "human_reference": holdout_label,
    }
    with pytest.raises(ContractValidationError, match="identity-bound truth projection"):
        build_report_data(records, graph, truth_rows=(row,))

    reference_label = _fabricated_human_reference(bundle, graph, partition="reference")
    vector = EmbeddingVector(
        bundle.record_identity,
        bundle.image_identity,
        "backend",
        "model",
        "mock",
        (1.0, 0.0),
    )
    with pytest.raises(ContractValidationError, match="verified reference-partition"):
        RetrievalIndexRecord.from_vectors((vector,), taxonomy_identity=graph.identity, human_label=reference_label)
    with pytest.raises(ContractValidationError, match="verified append-only"):
        replace(bundle, human=reference_label)


def test_verified_holdout_projection_enables_precision_with_provenance(tmp_path) -> None:
    bundle, graph, _nonhuman = _reviewed_bundle(
        tmp_path,
        candidate_node="bottle",
        selected_node="bottle",
        partition="holdout",
    )
    records = (_report_record(bundle.record_identity, graph.path("bottle")),)
    report = build_report_data(
        records,
        graph,
        truth_rows=(_truth_row(bundle, graph.path("bottle")),),
        calibration_state="validated_for_scope",
    )

    assert report["claims"]["precision_graph_available"] is True
    assert report["truth_metrics"]["state"] == "measured_from_verified_human_truth"
    assert report["truth_metrics"]["truth_source"] == "verified_append_only_human_review"
    assert bundle.human is not None and bundle.human.verification is not None
    assert report["truth_provenance"]["cohort_identity"] == bundle.human.verification.cohort_identity
    forged = _truth_row(bundle, graph.path("bottle"))
    forged["leakage_group_identity"] = "forged-leakage-group"
    with pytest.raises(ContractValidationError, match="leakage group"):
        build_report_data(records, graph, truth_rows=(forged,))


def test_synthetic_oracle_is_separate_and_cannot_validate_production(tmp_path) -> None:
    bundle, graph, _renders = make_bundle(tmp_path)
    oracle = synthetic_oracle_reference_label(
        record_identity=bundle.record_identity,
        taxonomy_identity=graph.identity,
        taxonomy_path=graph.path("bottle"),
        deepest_accepted_node="bottle",
        explicit_abstentions=(),
        partition="holdout",
        oracle_set_identity="oracle-set",
        image_identity=bundle.image_identity,
        evidence_bundle_identity=bundle.identity,
        cohort_identity="oracle-cohort",
        source_identity="source-a",
        cluster_identity="cluster-a",
        leakage_group_identity="leakage-a",
    )
    row = {
        "record_identity": bundle.record_identity,
        "evidence_bundle_identity": bundle.identity,
        "predicted_path": list(graph.path("bottle")),
        "calibrated_probability": 0.9,
        "source_identity": "source-a",
        "cluster_identity": "cluster-a",
        "leakage_group_identity": "leakage-a",
        "human_reference": oracle,
    }
    records = (_report_record(bundle.record_identity, graph.path("bottle")),)

    with pytest.raises(ContractValidationError, match="identity-bound truth projection"):
        build_report_data(records, graph, truth_rows=(row,))
    report = build_report_data(
        records,
        graph,
        synthetic_oracle_rows=(row,),
        calibration_state="validated_for_scope",
        calibration_truth_scope="synthetic_oracle",
    )
    assert report["claims"]["precision_graph_available"] is False
    assert report["claims"]["synthetic_oracle_can_validate_production"] is False
    assert report["summary"]["calibration_state"] == "ready_for_experiment"
    assert report["synthetic_oracle_metrics"]["truth_source"] == "synthetic_oracle_fixture"

    synthetic_model = replace(
        calibration_model(graph),
        truth_scope="synthetic_oracle",
        state=CalibrationState.VALIDATED_FOR_SCOPE,
    )
    decision = decide_hierarchical_label(bundle, graph, synthetic_model)
    assert decision.calibration_state == CalibrationState.READY_FOR_EXPERIMENT
    assert SYNTHETIC_ORACLE_CONFLICT in decision.conflicts
    assert export_supervision(bundle, decision, graph).supervision_tier == SupervisionTier.MODEL_PROPOSAL
