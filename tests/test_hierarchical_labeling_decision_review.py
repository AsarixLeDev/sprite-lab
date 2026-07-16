from __future__ import annotations

from dataclasses import replace

import pytest

from hierarchical_labeling_support import calibration_model, make_bundle
from spritelab.hierarchical_labeling.calibration import (
    SYNTHETIC_ORACLE_SCOPE,
    CalibrationExample,
    cross_validate,
    evaluate_holdout,
    fit_calibration,
    precision_coverage_curve,
)
from spritelab.hierarchical_labeling.contracts import HumanReferenceLabel, RetrievalNeighbor
from spritelab.hierarchical_labeling.decision import conditional_field_decision, decide_hierarchical_label
from spritelab.hierarchical_labeling.json_utils import ContractValidationError
from spritelab.hierarchical_labeling.review import (
    GENESIS_EVENT_HASH,
    BatchReviewItem,
    append_batch_review,
    append_review_action,
    create_review_event,
    load_review_events,
    review_consensus,
    synthetic_oracle_reference_label,
)
from spritelab.hierarchical_labeling.taxonomy import load_default_taxonomy


def _reviewed_neighbor(path, *, similarity=0.95, record="neighbor"):
    return RetrievalNeighbor(
        record,
        f"image-{record}",
        "embedding",
        load_default_taxonomy().identity,
        1 - similarity,
        similarity,
        "reviewed",
        path,
        "reference-cohort",
        "truth-projection",
        "review-log",
    )


def test_decision_leaf_rejected_parent_accepted_and_component_trace(tmp_path) -> None:
    bundle, graph, _renders = make_bundle(tmp_path, node_id="bottle", confidence=0.2, abstention_recommended=True)
    decision = decide_hierarchical_label(bundle, graph, calibration_model(graph))
    assert decision.deepest_accepted_node == "container"
    assert decision.abstained_below_node == "bottle"
    assert decision.taxonomy_path == graph.path("container")
    assert all(level.components and level.reasons for level in decision.level_decisions)


def test_decision_high_novelty_prefers_safe_ancestor(tmp_path) -> None:
    graph_path = ("entity", "object", "container", "bottle")
    bundle, graph, _renders = make_bundle(
        tmp_path,
        node_id="bottle",
        neighbors=(_reviewed_neighbor(graph_path),),
        novelty=0.95,
    )
    decision = decide_hierarchical_label(bundle, graph, calibration_model(graph))
    assert decision.deepest_accepted_node == "container"
    assert "high novelty" in " ".join(decision.level_decisions[-1].reasons)


def test_decision_strong_retrieval_conflict_and_metadata_conflict_visible(tmp_path) -> None:
    sword = ("entity", "object", "equipment", "weapon", "sword")
    retrieval_bundle, graph, _renders = make_bundle(
        tmp_path / "retrieval",
        node_id="bottle",
        neighbors=(_reviewed_neighbor(sword),),
        novelty=0.1,
    )
    retrieval_decision = decide_hierarchical_label(retrieval_bundle, graph, calibration_model(graph))
    assert any("visual_retrieval_conflict" in value for value in retrieval_decision.conflicts)
    metadata_bundle, graph, _renders = make_bundle(tmp_path / "metadata", node_id="bottle", metadata_node="resource")
    metadata_decision = decide_hierarchical_label(metadata_bundle, graph, calibration_model(graph))
    assert any("visual_metadata_conflict" in value for value in metadata_decision.conflicts)
    assert "metadata" in metadata_decision.contributed_channels


def test_decision_reviewed_retrieval_support_can_pass_threshold(tmp_path) -> None:
    bottle = ("entity", "object", "container", "bottle")
    bundle, graph, _renders = make_bundle(
        tmp_path,
        node_id="bottle",
        confidence=0.65,
        neighbors=(_reviewed_neighbor(bottle),),
        novelty=0.1,
    )
    decision = decide_hierarchical_label(bundle, graph, calibration_model(graph, threshold=0.6))
    assert decision.deepest_accepted_node == "bottle"
    leaf = decision.level_decisions[-1]
    retrieval = next(component for component in leaf.components if component.name == "reviewed_retrieval")
    assert retrieval.value == 1.0


def test_decision_never_forces_unknown_and_conditional_fields_abstain(tmp_path) -> None:
    bundle, graph, _renders = make_bundle(tmp_path)
    decision = decide_hierarchical_label(bundle, graph, calibration_model(graph))
    assert "unknown" not in decision.taxonomy_path
    field = conditional_field_decision(
        "canonical_object",
        "bottle",
        direct_visual_support=True,
        calibrated_probability=0.7,
        threshold=0.9,
        contradiction=False,
        decision_policy_identity="policy",
        evidence_bundle_identity=bundle.identity,
    )
    assert field.state.value == "model_abstained"


def _human(graph, record, partition, path=None):
    selected = path or graph.path("bottle")
    return synthetic_oracle_reference_label(
        record_identity=record,
        taxonomy_identity=graph.identity,
        taxonomy_path=selected,
        deepest_accepted_node=selected[-1],
        explicit_abstentions=(),
        partition=partition,
        oracle_set_identity="test-oracle-set",
        image_identity=f"image-{record}",
        evidence_bundle_identity=f"bundle-{record}",
        cohort_identity="test-oracle-cohort",
        source_identity=f"source-{record}",
        cluster_identity=f"cluster-{record}",
        leakage_group_identity=f"group-{record}",
        duplicate_cluster_identity=f"duplicate-{record}",
        near_duplicate_cluster_identity=f"near-{record}",
    )


def _example(graph, index, *, partition="calibration", score=0.8, node="object", path=None):
    record = f"r-{partition}-{index}"
    return CalibrationExample(
        record,
        node,
        score,
        f"bundle-{record}",
        f"source-{record}",
        f"cluster-{record}",
        _human(graph, record, partition, path),
        f"duplicate-{record}",
        f"near-{record}",
        f"group-{record}",
    )


def test_calibration_zero_truth_and_insufficient_class_samples(tmp_path) -> None:
    _bundle, graph, _renders = make_bundle(tmp_path)
    empty = fit_calibration([], graph)
    assert empty.state.value == "not_ready" and empty.fit_example_count == 0
    sparse = fit_calibration(
        [_example(graph, index) for index in range(5)],
        graph,
        minimum_global_samples=3,
        minimum_class_samples=10,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    assert "object" not in dict(sparse.class_thresholds)
    assert any("falls back" in limitation for limitation in sparse.limitations)


def test_calibration_holdout_precision_coverage_risk_and_no_leakage(tmp_path) -> None:
    _bundle, graph, _renders = make_bundle(tmp_path)
    fit = [_example(graph, index, score=0.55 + index * 0.01) for index in range(20)]
    model = fit_calibration(
        fit,
        graph,
        minimum_global_samples=20,
        minimum_class_samples=10,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    holdout = [_example(graph, index + 100, partition="holdout", score=0.9) for index in range(5)]
    evaluated, result = evaluate_holdout(
        model,
        holdout,
        graph,
        minimum_holdout_samples=5,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    assert evaluated.state.value == "ready_for_experiment"
    assert result.precision == 1.0 and result.coverage == 1.0 and result.risk == 0.0
    leaked_label = _human(graph, fit[0].record_identity, "holdout")
    leaked = CalibrationExample(
        fit[0].record_identity,
        "object",
        0.9,
        f"bundle-{fit[0].record_identity}",
        f"source-{fit[0].record_identity}",
        f"cluster-{fit[0].record_identity}",
        leaked_label,
        f"duplicate-{fit[0].record_identity}",
        f"near-{fit[0].record_identity}",
        f"group-{fit[0].record_identity}",
    )
    with pytest.raises(ContractValidationError, match="overlap"):
        evaluate_holdout(
            model,
            [leaked],
            graph,
            minimum_holdout_samples=1,
            truth_scope=SYNTHETIC_ORACLE_SCOPE,
        )


def test_calibration_curve_and_cross_validation_are_explicitly_synthetic(tmp_path) -> None:
    _bundle, graph, _renders = make_bundle(tmp_path)
    examples = [_example(graph, index, score=0.95 - index * 0.01) for index in range(25)]
    curve = precision_coverage_curve(examples, graph, truth_scope=SYNTHETIC_ORACLE_SCOPE)
    assert curve[-1]["coverage"] == 1.0
    assert all(point["truth_source"] == SYNTHETIC_ORACLE_SCOPE for point in curve)
    cross = cross_validate(
        examples,
        graph,
        folds=3,
        minimum_global_samples=5,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    assert all(fold["identity_overlap"] == 0 for fold in cross["folds"])
    assert cross["truth_source"] == "synthetic_oracle_cross_validation"
    assert cross["production_validation"] is False


def test_calibration_precision_target_beats_coverage_goal(tmp_path) -> None:
    _bundle, graph, _renders = make_bundle(tmp_path)
    correct = [_example(graph, index, score=0.9, node="bottle") for index in range(18)]
    wrong_path = graph.path("sword")
    wrong = [_example(graph, index + 30, score=0.4, node="bottle", path=wrong_path) for index in range(2)]
    model = fit_calibration(
        correct + wrong,
        graph,
        target_precision=0.95,
        minimum_global_samples=5,
        minimum_class_samples=5,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    assert model.global_threshold > 0.4


def test_calibration_rejects_unverified_direct_human_label(tmp_path) -> None:
    _bundle, graph, _renders = make_bundle(tmp_path)
    record = "unverified-calibration"
    direct = HumanReferenceLabel(
        record,
        "unverified-event",
        graph.identity,
        graph.path("bottle"),
        "bottle",
        (),
        "calibration",
        "reviewer",
    )
    row = CalibrationExample(record, "object", 0.9, "bundle", "source", "cluster", direct)
    with pytest.raises(ContractValidationError, match="verified append-only"):
        fit_calibration((row,), graph, minimum_global_samples=1, minimum_class_samples=1)


@pytest.mark.parametrize(
    "field",
    [
        "source_identity",
        "cluster_identity",
        "duplicate_cluster_identity",
        "near_duplicate_cluster_identity",
        "leakage_group_identity",
        "image_identity",
    ],
)
def test_calibration_holdout_rejects_every_group_overlap(tmp_path, field) -> None:
    _bundle, graph, _renders = make_bundle(tmp_path)
    fit_row = _example(graph, 1)
    model = fit_calibration(
        (fit_row,),
        graph,
        minimum_global_samples=1,
        minimum_class_samples=1,
        truth_scope=SYNTHETIC_ORACLE_SCOPE,
    )
    holdout = _example(graph, 101, partition="holdout")
    fit_label = fit_row.human_label
    holdout_label = holdout.human_label
    replacement = getattr(fit_label, field)
    changed_label = replace(holdout_label, **{field: replacement})
    changed = replace(holdout, human_label=changed_label)
    if hasattr(changed, field):
        changed = replace(changed, **{field: replacement})
    with pytest.raises(ContractValidationError, match="overlap"):
        evaluate_holdout(
            model,
            (changed,),
            graph,
            minimum_holdout_samples=1,
            truth_scope=SYNTHETIC_ORACLE_SCOPE,
        )


def test_human_review_append_choose_parent_abstain_and_taxonomy_gap(tmp_path) -> None:
    bundle, graph, renders = make_bundle(tmp_path)
    log = tmp_path / "events.jsonl"
    parent = append_review_action(
        log,
        bundle,
        graph,
        action="choose_parent",
        reviewer_identity="reviewer-a",
        partition="reference",
        selected_node="container",
        render_identities=(renders.identity,),
        submission_token="parent",
    )
    abstain = append_review_action(
        log,
        bundle,
        graph,
        action="abstain",
        reviewer_identity="reviewer-b",
        partition="calibration",
        render_identities=(renders.identity,),
        submission_token="abstain",
    )
    gap = append_review_action(
        log,
        bundle,
        graph,
        action="flag_taxonomy_gap",
        reviewer_identity="reviewer-c",
        partition="holdout",
        render_identities=(renders.identity,),
        submission_token="gap",
    )
    events = load_review_events(log)
    assert len(events) == 3 and events[1].previous_event_hash == parent.event_hash
    assert abstain.explicit_abstentions == ("reviewer_abstained",)
    assert gap.explicit_abstentions == ("taxonomy_gap",)


def test_human_review_batch_identity_and_no_legal_override(tmp_path) -> None:
    first, graph, renders = make_bundle(tmp_path / "first", record_identity="first")
    second, _graph, second_renders = make_bundle(tmp_path / "second", record_identity="second")
    log = tmp_path / "batch.jsonl"
    exemplar = append_review_action(
        log,
        first,
        graph,
        action="accept_suggested_path",
        reviewer_identity="reviewer",
        partition="reference",
        selected_node="bottle",
        render_identities=(renders.identity,),
        submission_token="exemplar",
    )
    appended = append_batch_review(
        log,
        graph,
        cluster_identity="cluster-bottle",
        exemplar_event=exemplar,
        items=(BatchReviewItem(second, (second_renders.identity,), True),),
        selected_node="bottle",
        reviewer_identity="reviewer",
        partition_by_record={"second": "reference"},
        explicit_confirmation=True,
    )
    assert appended[0].batch_identity and appended[0].batch_identity != exemplar.event_hash
    with pytest.raises(ContractValidationError, match="provenance"):
        append_batch_review(
            log,
            graph,
            cluster_identity="cluster-bottle",
            exemplar_event=exemplar,
            items=(BatchReviewItem(second, (second_renders.identity,), False),),
            selected_node="bottle",
            reviewer_identity="reviewer",
            partition_by_record={"second": "reference"},
            explicit_confirmation=True,
        )


def test_human_review_conflicting_double_review_and_adjudication(tmp_path) -> None:
    bundle, graph, renders = make_bundle(tmp_path)
    first = create_review_event(
        bundle,
        graph,
        action="choose_alternative",
        reviewer_identity="reviewer-a",
        partition="reference",
        previous_event_hash=GENESIS_EVENT_HASH,
        selected_node="bottle",
        render_identities=(renders.identity,),
        timestamp="2026-01-01T00:00:00+00:00",
        submission_token="a",
    )
    agreeing = create_review_event(
        bundle,
        graph,
        action="choose_alternative",
        reviewer_identity="reviewer-b",
        partition="reference",
        previous_event_hash=first.event_hash,
        selected_node="bottle",
        render_identities=(renders.identity,),
        timestamp="2026-01-01T00:00:01+00:00",
        submission_token="b",
    )
    assert review_consensus((first, agreeing), bundle.record_identity)["state"] == "double_review_agreement"
    conflicting = create_review_event(
        bundle,
        graph,
        action="choose_parent",
        reviewer_identity="reviewer-b",
        partition="reference",
        previous_event_hash=first.event_hash,
        selected_node="container",
        render_identities=(renders.identity,),
        timestamp="2026-01-01T00:00:02+00:00",
        submission_token="c",
    )
    assert review_consensus((first, conflicting), bundle.record_identity)["state"] == "adjudication_required"
    adjudicated = create_review_event(
        bundle,
        graph,
        action="adjudicate",
        reviewer_identity="adjudicator",
        partition="reference",
        previous_event_hash=conflicting.event_hash,
        selected_node="container",
        render_identities=(renders.identity,),
        adjudicates_event_ids=(first.event_id, conflicting.event_id),
        timestamp="2026-01-01T00:00:03+00:00",
        submission_token="d",
    )
    assert review_consensus((first, conflicting, adjudicated), bundle.record_identity)["state"] == "adjudicated"


def test_human_review_cannot_override_legal_ineligibility(tmp_path) -> None:
    bundle, graph, renders = make_bundle(tmp_path)
    with pytest.raises(ContractValidationError, match="provenance"):
        create_review_event(
            bundle,
            graph,
            action="choose_alternative",
            reviewer_identity="reviewer",
            partition="reference",
            previous_event_hash=GENESIS_EVENT_HASH,
            selected_node="bottle",
            render_identities=(renders.identity,),
            legal_and_provenance_eligible=False,
        )
