"""Human-truth-only calibration and precision/coverage evaluation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any

from spritelab.hierarchical_labeling.contracts import (
    CalibrationResult,
    CalibrationState,
    HumanReferenceLabel,
    SyntheticOracleLabel,
)
from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    content_identity,
    require_probability,
    require_text,
)
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph

CALIBRATION_MODEL_SCHEMA = "spritelab.labeling.calibration-model.v1"
HUMAN_TRUTH_SCOPE = "append_only_human_review"
SYNTHETIC_ORACLE_SCOPE = "synthetic_oracle"
TRUTH_SCOPES = frozenset({HUMAN_TRUTH_SCOPE, SYNTHETIC_ORACLE_SCOPE})


@dataclass(frozen=True, eq=False)
class CalibrationExample(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.calibration-example.v1"
    IDENTITY_FIELDS = ("record_identity", "node_id", "evidence_bundle_identity", "human_review_identity")

    record_identity: str
    node_id: str
    raw_score: float
    evidence_bundle_identity: str
    source_identity: str
    cluster_identity: str
    human_label: HumanReferenceLabel | SyntheticOracleLabel
    duplicate_cluster_identity: str | None = None
    near_duplicate_cluster_identity: str | None = None
    leakage_group_identity: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "record_identity",
            "node_id",
            "evidence_bundle_identity",
            "source_identity",
            "cluster_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        require_probability(self.raw_score, "calibration raw score")
        if self.human_label.record_identity != self.record_identity:
            raise ContractValidationError("calibration example human label identity mismatch")
        for name in ("duplicate_cluster_identity", "near_duplicate_cluster_identity", "leakage_group_identity"):
            value = getattr(self, name)
            if value is not None:
                require_text(value, name.replace("_", " "))
        self.validate_record()

    @property
    def human_review_identity(self) -> str:
        return self.human_label.review_event_identity


@dataclass(frozen=True)
class MonotonicBin:
    lower_score: float
    upper_score: float
    probability_correct: float
    sample_size: int
    correct_count: int

    def __post_init__(self) -> None:
        require_probability(self.lower_score, "calibration bin lower score")
        require_probability(self.upper_score, "calibration bin upper score")
        require_probability(self.probability_correct, "calibration bin probability")
        if self.lower_score > self.upper_score:
            raise ContractValidationError("calibration bin score bounds are reversed")
        if type(self.sample_size) is not int or self.sample_size < 1:
            raise ContractValidationError("calibration bin sample size must be positive")
        if type(self.correct_count) is not int or not 0 <= self.correct_count <= self.sample_size:
            raise ContractValidationError("calibration bin correct count is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_score": self.lower_score,
            "upper_score": self.upper_score,
            "probability_correct": self.probability_correct,
            "sample_size": self.sample_size,
            "correct_count": self.correct_count,
        }


@dataclass(frozen=True)
class CalibrationModel:
    taxonomy_identity: str
    state: CalibrationState
    target_precision: float
    minimum_global_samples: int
    minimum_class_samples: int
    global_threshold: float
    depth_thresholds: tuple[tuple[int, float], ...]
    class_thresholds: tuple[tuple[str, float], ...]
    depth_bins: tuple[tuple[int, tuple[MonotonicBin, ...]], ...]
    fit_record_identities: tuple[str, ...]
    fit_example_count: int
    class_sample_counts: tuple[tuple[str, int], ...]
    source_diagnostics: tuple[tuple[str, int, int], ...]
    cluster_diagnostics: tuple[tuple[str, int, int], ...]
    limitations: tuple[str, ...]
    truth_scope: str = HUMAN_TRUTH_SCOPE
    cohort_identity: str | None = None
    fit_source_identities: tuple[str, ...] = ()
    fit_cluster_identities: tuple[str, ...] = ()
    fit_duplicate_cluster_identities: tuple[str, ...] = ()
    fit_near_duplicate_cluster_identities: tuple[str, ...] = ()
    fit_leakage_group_identities: tuple[str, ...] = ()
    fit_image_identities: tuple[str, ...] = ()
    review_log_identity: str | None = None
    chain_tip_identity: str | None = None

    def __post_init__(self) -> None:
        require_text(self.taxonomy_identity, "calibration taxonomy identity")
        require_probability(self.target_precision, "target precision")
        require_probability(self.global_threshold, "global threshold")
        if type(self.fit_example_count) is not int or self.fit_example_count < 0:
            raise ContractValidationError("calibration fit example count must be non-negative")
        if len(self.fit_record_identities) != len(set(self.fit_record_identities)):
            raise ContractValidationError("calibration fit record identities cannot repeat")
        if self.truth_scope not in TRUTH_SCOPES:
            raise ContractValidationError("calibration truth scope is not controlled")
        if self.fit_example_count and self.cohort_identity is None:
            raise ContractValidationError("calibration with truth rows must bind a cohort identity")
        if self.cohort_identity is not None:
            require_text(self.cohort_identity, "calibration cohort identity")
        for name in (
            "fit_source_identities",
            "fit_cluster_identities",
            "fit_duplicate_cluster_identities",
            "fit_near_duplicate_cluster_identities",
            "fit_leakage_group_identities",
            "fit_image_identities",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ContractValidationError(f"calibration {name.replace('_', ' ')} must be unique text")
        if self.truth_scope == HUMAN_TRUTH_SCOPE and self.fit_example_count:
            if self.review_log_identity is None or self.chain_tip_identity is None:
                raise ContractValidationError("human calibration must bind one verified review-log snapshot")
        for name in ("review_log_identity", "chain_tip_identity"):
            value = getattr(self, name)
            if value is not None:
                require_text(value, name.replace("_", " "))

    @property
    def identity(self) -> str:
        return content_identity(CALIBRATION_MODEL_SCHEMA, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_MODEL_SCHEMA,
            "taxonomy_identity": self.taxonomy_identity,
            "state": self.state.value,
            "target_precision": self.target_precision,
            "minimum_global_samples": self.minimum_global_samples,
            "minimum_class_samples": self.minimum_class_samples,
            "global_threshold": self.global_threshold,
            "depth_thresholds": {str(depth): value for depth, value in self.depth_thresholds},
            "class_thresholds": dict(self.class_thresholds),
            "depth_bins": {str(depth): [item.to_dict() for item in bins] for depth, bins in self.depth_bins},
            "fit_record_identities": list(self.fit_record_identities),
            "fit_example_count": self.fit_example_count,
            "class_sample_counts": dict(self.class_sample_counts),
            "source_diagnostics": [
                {"source_identity": key, "examples": total, "errors": errors}
                for key, total, errors in self.source_diagnostics
            ],
            "cluster_diagnostics": [
                {"cluster_identity": key, "examples": total, "errors": errors}
                for key, total, errors in self.cluster_diagnostics
            ],
            "limitations": list(self.limitations),
            "truth_scope": self.truth_scope,
            "cohort_identity": self.cohort_identity,
            "fit_source_identities": list(self.fit_source_identities),
            "fit_cluster_identities": list(self.fit_cluster_identities),
            "fit_duplicate_cluster_identities": list(self.fit_duplicate_cluster_identities),
            "fit_near_duplicate_cluster_identities": list(self.fit_near_duplicate_cluster_identities),
            "fit_leakage_group_identities": list(self.fit_leakage_group_identities),
            "fit_image_identities": list(self.fit_image_identities),
            "review_log_identity": self.review_log_identity,
            "chain_tip_identity": self.chain_tip_identity,
        }

    def threshold_for(self, node_id: str, graph: TaxonomyGraph) -> tuple[float, str]:
        class_map = dict(self.class_thresholds)
        if node_id in class_map:
            return class_map[node_id], f"class:{node_id}"
        depth_map = dict(self.depth_thresholds)
        current = graph.node(node_id)
        while current is not None:
            depth = graph.depth(current.node_id)
            if depth in depth_map:
                return depth_map[depth], f"depth:{depth}"
            current = graph.parent(current.node_id)
        return self.global_threshold, "global"

    def calibrated_probability(self, node_id: str, score: float, graph: TaxonomyGraph) -> float | None:
        bins_by_depth = dict(self.depth_bins)
        current = graph.node(node_id)
        bins: tuple[MonotonicBin, ...] = ()
        while current is not None and not bins:
            bins = bins_by_depth.get(graph.depth(current.node_id), ())
            current = graph.parent(current.node_id)
        if not bins:
            return None
        for item in bins:
            if score <= item.upper_score:
                return item.probability_correct
        return bins[-1].probability_correct


def example_correct(example: CalibrationExample, graph: TaxonomyGraph) -> bool:
    resolved = graph.resolve(example.node_id)
    if resolved is None:
        return False
    return resolved in example.human_label.taxonomy_path


def _validate_examples(
    examples: Sequence[CalibrationExample],
    graph: TaxonomyGraph,
    *,
    required_partition: str | None,
    truth_scope: str,
) -> tuple[CalibrationExample, ...]:
    if truth_scope not in TRUTH_SCOPES:
        raise ContractValidationError("calibration truth scope is not controlled")
    seen: set[tuple[str, str]] = set()
    validated: list[CalibrationExample] = []
    cohorts: set[str] = set()
    review_logs: set[tuple[str, str]] = set()
    for example in examples:
        label = example.human_label
        if required_partition is not None and label.partition != required_partition:
            raise ContractValidationError(
                f"{required_partition} calibration operation received {label.partition} truth"
            )
        if truth_scope == HUMAN_TRUTH_SCOPE:
            if (
                not isinstance(label, HumanReferenceLabel)
                or not label.verified_append_only
                or label.verification is None
            ):
                raise ContractValidationError("calibration requires a verified append-only human truth projection")
            verification = label.verification
            cohorts.add(verification.cohort_identity)
            review_logs.add((verification.review_log_identity, verification.chain_tip_identity))
            bindings = (
                verification.record_identity == example.record_identity,
                verification.taxonomy_identity == graph.identity,
                verification.evidence_bundle_identity == example.evidence_bundle_identity,
                verification.source_identity == example.source_identity,
                verification.cluster_identity == example.cluster_identity,
                verification.duplicate_cluster_identity == example.duplicate_cluster_identity,
                verification.near_duplicate_cluster_identity == example.near_duplicate_cluster_identity,
                verification.leakage_group_identity == example.leakage_group_identity,
            )
            if not all(bindings):
                raise ContractValidationError("verified human truth does not bind the calibration example")
        else:
            if not isinstance(label, SyntheticOracleLabel):
                raise ContractValidationError("synthetic calibration requires an explicit synthetic oracle label")
            cohorts.add(label.cohort_identity)
            bindings = (
                label.record_identity == example.record_identity,
                label.taxonomy_identity == graph.identity,
                label.evidence_bundle_identity == example.evidence_bundle_identity,
                label.source_identity == example.source_identity,
                label.cluster_identity == example.cluster_identity,
                label.duplicate_cluster_identity == example.duplicate_cluster_identity,
                label.near_duplicate_cluster_identity == example.near_duplicate_cluster_identity,
                label.leakage_group_identity == example.leakage_group_identity,
            )
            if not all(bindings):
                raise ContractValidationError("synthetic oracle does not bind the calibration example")
        if label.taxonomy_identity != graph.identity:
            raise ContractValidationError("truth taxonomy identity does not match calibration taxonomy")
        if label.deepest_accepted_node is not None and label.taxonomy_path != graph.path(label.deepest_accepted_node):
            raise ContractValidationError("calibration truth path is not canonical for the taxonomy")
        for abstention in label.explicit_abstentions:
            if (
                abstention
                not in {
                    "reviewer_abstained",
                    "taxonomy_gap",
                    "technically_or_semantically_unusable",
                }
                and graph.resolve(abstention) != abstention
            ):
                raise ContractValidationError("calibration truth contains an invalid explicit abstention")
        if graph.resolve(example.node_id) != example.node_id:
            raise ContractValidationError("calibration example contains an invalid or deprecated node")
        key = (example.record_identity, example.node_id)
        if key in seen:
            raise ContractValidationError("calibration examples contain duplicate record/node rows")
        seen.add(key)
        validated.append(example)
    if len(cohorts) > 1:
        raise ContractValidationError("calibration examples must bind one cohort identity")
    if truth_scope == HUMAN_TRUTH_SCOPE and len(review_logs) > 1:
        raise ContractValidationError("calibration examples must bind one review-log snapshot")
    return tuple(validated)


def _select_threshold(
    examples: Sequence[CalibrationExample],
    graph: TaxonomyGraph,
    *,
    target_precision: float,
    minimum_samples: int,
) -> float | None:
    if len(examples) < minimum_samples:
        return None
    candidates = sorted({float(example.raw_score) for example in examples}, reverse=True)
    valid: list[tuple[float, int]] = []
    for threshold in candidates:
        accepted = [example for example in examples if example.raw_score >= threshold]
        if len(accepted) < minimum_samples:
            continue
        correct = sum(example_correct(example, graph) for example in accepted)
        precision = correct / len(accepted)
        if precision >= target_precision:
            valid.append((threshold, len(accepted)))
    if not valid:
        return 1.0
    return min(valid, key=lambda item: (item[0], -item[1]))[0]


def fit_monotonic_bins(
    examples: Sequence[CalibrationExample], graph: TaxonomyGraph, *, minimum_samples: int = 20
) -> tuple[MonotonicBin, ...]:
    if len(examples) < minimum_samples:
        return ()
    blocks: list[dict[str, Any]] = []
    for example in sorted(examples, key=lambda item: (item.raw_score, item.record_identity, item.node_id)):
        correct = int(example_correct(example, graph))
        blocks.append(
            {
                "lower": float(example.raw_score),
                "upper": float(example.raw_score),
                "count": 1,
                "correct": correct,
            }
        )
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = previous["correct"] / previous["count"]
            current_mean = current["correct"] / current["count"]
            if previous_mean <= current_mean:
                break
            blocks[-2:] = [
                {
                    "lower": previous["lower"],
                    "upper": current["upper"],
                    "count": previous["count"] + current["count"],
                    "correct": previous["correct"] + current["correct"],
                }
            ]
    return tuple(
        MonotonicBin(
            round(block["lower"], 8),
            round(block["upper"], 8),
            round(block["correct"] / block["count"], 8),
            block["count"],
            block["correct"],
        )
        for block in blocks
    )


def fit_calibration(
    examples: Sequence[CalibrationExample],
    graph: TaxonomyGraph,
    *,
    target_precision: float = 0.95,
    minimum_global_samples: int = 30,
    minimum_class_samples: int = 10,
    truth_scope: str = HUMAN_TRUTH_SCOPE,
) -> CalibrationModel:
    require_probability(target_precision, "calibration target precision")
    if minimum_global_samples < 1 or minimum_class_samples < 1:
        raise ContractValidationError("calibration minimum samples must be positive")
    rows = _validate_examples(
        examples,
        graph,
        required_partition="calibration",
        truth_scope=truth_scope,
    )
    limitations: list[str] = []
    if not rows:
        return CalibrationModel(
            taxonomy_identity=graph.identity,
            state=CalibrationState.NOT_READY,
            target_precision=target_precision,
            minimum_global_samples=minimum_global_samples,
            minimum_class_samples=minimum_class_samples,
            global_threshold=1.0,
            depth_thresholds=(),
            class_thresholds=(),
            depth_bins=(),
            fit_record_identities=(),
            fit_example_count=0,
            class_sample_counts=(),
            source_diagnostics=(),
            cluster_diagnostics=(),
            limitations=("zero truth rows in the requested calibration scope",),
            truth_scope=truth_scope,
        )
    by_depth: dict[int, list[CalibrationExample]] = defaultdict(list)
    by_class: dict[str, list[CalibrationExample]] = defaultdict(list)
    by_source: dict[str, list[CalibrationExample]] = defaultdict(list)
    by_cluster: dict[str, list[CalibrationExample]] = defaultdict(list)
    for row in rows:
        by_depth[graph.depth(row.node_id)].append(row)
        by_class[row.node_id].append(row)
        by_source[row.source_identity].append(row)
        by_cluster[row.cluster_identity].append(row)
    global_threshold = _select_threshold(
        rows, graph, target_precision=target_precision, minimum_samples=minimum_global_samples
    )
    if global_threshold is None:
        global_threshold = 1.0
        limitations.append(f"fewer than {minimum_global_samples} global calibration examples")
    depth_thresholds: list[tuple[int, float]] = []
    depth_bins: list[tuple[int, tuple[MonotonicBin, ...]]] = []
    for depth, depth_rows in sorted(by_depth.items()):
        threshold = _select_threshold(
            depth_rows,
            graph,
            target_precision=target_precision,
            minimum_samples=minimum_global_samples,
        )
        if threshold is not None:
            depth_thresholds.append((depth, threshold))
        depth_bins.append((depth, fit_monotonic_bins(depth_rows, graph)))
    class_thresholds: list[tuple[str, float]] = []
    for node_id, class_rows in sorted(by_class.items()):
        threshold = _select_threshold(
            class_rows,
            graph,
            target_precision=target_precision,
            minimum_samples=minimum_class_samples,
        )
        if threshold is not None:
            class_thresholds.append((node_id, threshold))
        else:
            limitations.append(f"class {node_id} falls back to parent/depth/global threshold")
    state = (
        CalibrationState.READY_FOR_EXPERIMENT
        if len(rows) >= minimum_global_samples and any(bins for _depth, bins in depth_bins)
        else CalibrationState.INSUFFICIENT_TRUTH
    )
    source_diagnostics = tuple(
        (key, len(values), sum(not example_correct(row, graph) for row in values))
        for key, values in sorted(by_source.items())
    )
    cluster_diagnostics = tuple(
        (key, len(values), sum(not example_correct(row, graph) for row in values))
        for key, values in sorted(by_cluster.items())
    )
    first_label = rows[0].human_label
    if isinstance(first_label, HumanReferenceLabel):
        assert first_label.verification is not None
        cohort_identity = first_label.verification.cohort_identity
        review_log_identity = first_label.verification.review_log_identity
        chain_tip_identity = first_label.verification.chain_tip_identity
        images = tuple(
            sorted(
                {
                    row.human_label.verification.image_identity
                    for row in rows
                    if isinstance(row.human_label, HumanReferenceLabel) and row.human_label.verification is not None
                }
            )
        )
    else:
        cohort_identity = first_label.cohort_identity
        review_log_identity = None
        chain_tip_identity = None
        images = tuple(
            sorted(
                {row.human_label.image_identity for row in rows if isinstance(row.human_label, SyntheticOracleLabel)}
            )
        )
    return CalibrationModel(
        taxonomy_identity=graph.identity,
        state=state,
        target_precision=target_precision,
        minimum_global_samples=minimum_global_samples,
        minimum_class_samples=minimum_class_samples,
        global_threshold=global_threshold,
        depth_thresholds=tuple(depth_thresholds),
        class_thresholds=tuple(class_thresholds),
        depth_bins=tuple(depth_bins),
        fit_record_identities=tuple(sorted({row.record_identity for row in rows})),
        fit_example_count=len(rows),
        class_sample_counts=tuple((key, len(values)) for key, values in sorted(by_class.items())),
        source_diagnostics=source_diagnostics,
        cluster_diagnostics=cluster_diagnostics,
        limitations=tuple(dict.fromkeys(limitations)),
        truth_scope=truth_scope,
        cohort_identity=cohort_identity,
        fit_source_identities=tuple(sorted({row.source_identity for row in rows})),
        fit_cluster_identities=tuple(sorted({row.cluster_identity for row in rows})),
        fit_duplicate_cluster_identities=tuple(
            sorted({row.duplicate_cluster_identity for row in rows if row.duplicate_cluster_identity is not None})
        ),
        fit_near_duplicate_cluster_identities=tuple(
            sorted(
                {row.near_duplicate_cluster_identity for row in rows if row.near_duplicate_cluster_identity is not None}
            )
        ),
        fit_leakage_group_identities=tuple(
            sorted({row.leakage_group_identity for row in rows if row.leakage_group_identity is not None})
        ),
        fit_image_identities=images,
        review_log_identity=review_log_identity,
        chain_tip_identity=chain_tip_identity,
    )


def _calibration_error(
    examples: Sequence[CalibrationExample], model: CalibrationModel, graph: TaxonomyGraph
) -> float | None:
    rows: list[tuple[float, int]] = []
    for example in examples:
        probability = model.calibrated_probability(example.node_id, example.raw_score, graph)
        if probability is not None:
            rows.append((probability, int(example_correct(example, graph))))
    if not rows:
        return None
    return round(sum(abs(probability - correct) for probability, correct in rows) / len(rows), 8)


def evaluate_holdout(
    model: CalibrationModel,
    examples: Sequence[CalibrationExample],
    graph: TaxonomyGraph,
    *,
    minimum_holdout_samples: int = 20,
    truth_scope: str | None = None,
) -> tuple[CalibrationModel, CalibrationResult]:
    if model.taxonomy_identity != graph.identity:
        raise ContractValidationError("calibration model taxonomy does not match holdout taxonomy")
    selected_scope = model.truth_scope if truth_scope is None else truth_scope
    if selected_scope != model.truth_scope:
        raise ContractValidationError("holdout truth scope does not match the calibration model")
    rows = _validate_examples(
        examples,
        graph,
        required_partition="holdout",
        truth_scope=selected_scope,
    )
    dimensions = {
        "record": (set(model.fit_record_identities), {row.record_identity for row in rows}),
        "source": (set(model.fit_source_identities), {row.source_identity for row in rows}),
        "cluster": (set(model.fit_cluster_identities), {row.cluster_identity for row in rows}),
        "exact duplicate cluster": (
            set(model.fit_duplicate_cluster_identities),
            {row.duplicate_cluster_identity for row in rows if row.duplicate_cluster_identity is not None},
        ),
        "near duplicate cluster": (
            set(model.fit_near_duplicate_cluster_identities),
            {row.near_duplicate_cluster_identity for row in rows if row.near_duplicate_cluster_identity is not None},
        ),
        "leakage group": (
            set(model.fit_leakage_group_identities),
            {row.leakage_group_identity for row in rows if row.leakage_group_identity is not None},
        ),
    }
    holdout_images = {
        row.human_label.verification.image_identity
        if isinstance(row.human_label, HumanReferenceLabel) and row.human_label.verification is not None
        else row.human_label.image_identity
        for row in rows
    }
    dimensions["image"] = (set(model.fit_image_identities), holdout_images)
    for name, (fit_values, holdout_values) in dimensions.items():
        if fit_values & holdout_values:
            raise ContractValidationError(f"calibration fit and held-out evaluation {name} identities overlap")
    if rows:
        first = rows[0].human_label
        cohort_identity = (
            first.verification.cohort_identity
            if isinstance(first, HumanReferenceLabel) and first.verification is not None
            else first.cohort_identity
        )
        if cohort_identity != model.cohort_identity:
            raise ContractValidationError("holdout truth does not belong to the calibration cohort")
        if isinstance(first, HumanReferenceLabel):
            assert first.verification is not None
            if (
                first.verification.review_log_identity != model.review_log_identity
                or first.verification.chain_tip_identity != model.chain_tip_identity
            ):
                raise ContractValidationError("holdout truth does not bind the calibration review-log snapshot")
    accepted: list[CalibrationExample] = []
    for row in rows:
        threshold, _fallback = model.threshold_for(row.node_id, graph)
        if row.raw_score >= threshold:
            accepted.append(row)
    correct = sum(example_correct(row, graph) for row in accepted)
    accepted_count = len(accepted)
    precision = correct / accepted_count if accepted_count else None
    coverage = accepted_count / len(rows) if rows else None
    risk = (accepted_count - correct) / accepted_count if accepted_count else None
    calibration_error = _calibration_error(rows, model, graph)
    validated = (
        model.state == CalibrationState.READY_FOR_EXPERIMENT
        and len(rows) >= minimum_holdout_samples
        and precision is not None
        and precision >= model.target_precision
    )
    state = CalibrationState.VALIDATED_FOR_SCOPE if validated and selected_scope == HUMAN_TRUTH_SCOPE else model.state
    limitations = list(model.limitations)
    if not rows:
        limitations.append("zero independent human-reviewed holdout rows")
    elif len(rows) < minimum_holdout_samples:
        limitations.append(f"fewer than {minimum_holdout_samples} independent holdout examples")
    if precision is None:
        limitations.append("no labels were accepted on holdout; precision is undefined")
    elif precision < model.target_precision:
        limitations.append("held-out accepted precision is below the configured target")
    if validated and selected_scope == SYNTHETIC_ORACLE_SCOPE:
        limitations.append("synthetic oracle metrics cannot validate a production human-truth scope")
    evaluated_model = replace(model, state=state, limitations=tuple(dict.fromkeys(limitations)))
    truth_set_identity = content_identity(
        "spritelab-calibration-holdout-truth-v2",
        {"truth_scope": selected_scope, "labels": [row.human_label.identity for row in rows]},
    )
    bins = tuple(
        {
            "depth": depth,
            "bins": [item.to_dict() for item in depth_values],
        }
        for depth, depth_values in model.depth_bins
    )
    result = CalibrationResult(
        evaluated_model.identity,
        graph.identity,
        truth_set_identity,
        state,
        "calibration",
        "holdout",
        len(rows),
        accepted_count,
        correct,
        round(precision, 8) if precision is not None else None,
        round(coverage, 8) if coverage is not None else None,
        round(risk, 8) if risk is not None else None,
        calibration_error,
        tuple(
            [("global", model.global_threshold)]
            + [(f"depth:{depth}", value) for depth, value in model.depth_thresholds]
            + [(f"class:{node}", value) for node, value in model.class_thresholds]
        ),
        bins,
        (
            "append-only human review holdout partition"
            if selected_scope == HUMAN_TRUTH_SCOPE
            else "synthetic oracle fixture holdout partition"
        ),
        tuple(dict.fromkeys(limitations)),
    )
    return evaluated_model, result


def precision_coverage_curve(
    examples: Sequence[CalibrationExample],
    graph: TaxonomyGraph,
    *,
    truth_scope: str = HUMAN_TRUTH_SCOPE,
) -> tuple[dict[str, Any], ...]:
    if not examples:
        return ()
    validated = _validate_examples(examples, graph, required_partition=None, truth_scope=truth_scope)
    rows = sorted(validated, key=lambda item: (-item.raw_score, item.record_identity, item.node_id))
    points: list[dict[str, Any]] = []
    correct = 0
    for index, row in enumerate(rows, start=1):
        correct += int(example_correct(row, graph))
        points.append(
            {
                "threshold": row.raw_score,
                "accepted": index,
                "total": len(rows),
                "correct_accepted": correct,
                "wrong_accepted": index - correct,
                "precision": correct / index,
                "coverage": index / len(rows),
                "truth_source": truth_scope,
            }
        )
    return tuple(points)


def cross_validate(
    examples: Sequence[CalibrationExample],
    graph: TaxonomyGraph,
    *,
    folds: int = 5,
    target_precision: float = 0.95,
    minimum_global_samples: int = 10,
    truth_scope: str = HUMAN_TRUTH_SCOPE,
) -> dict[str, Any]:
    if type(folds) is not int or folds < 2:
        raise ContractValidationError("cross-validation requires at least two folds")
    if truth_scope != SYNTHETIC_ORACLE_SCOPE:
        raise ContractValidationError(
            "verified human partitions are immutable; use independent holdout evaluation instead of cross-validation"
        )
    rows = _validate_examples(examples, graph, required_partition=None, truth_scope=truth_scope)
    parent = {row.record_identity: row.record_identity for row in rows}

    def find(record_identity: str) -> str:
        while parent[record_identity] != record_identity:
            parent[record_identity] = parent[parent[record_identity]]
            record_identity = parent[record_identity]
        return record_identity

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_by_key: dict[tuple[str, str], str] = {}
    for row in rows:
        keys = [("source", row.source_identity), ("cluster", row.cluster_identity)]
        if row.duplicate_cluster_identity is not None:
            keys.append(("duplicate", row.duplicate_cluster_identity))
        if row.near_duplicate_cluster_identity is not None:
            keys.append(("near_duplicate", row.near_duplicate_cluster_identity))
        for key in keys:
            union(first_by_key.setdefault(key, row.record_identity), row.record_identity)
    if len({find(row.record_identity) for row in rows}) < folds:
        raise ContractValidationError("cross-validation has fewer leakage-disjoint groups than folds")
    fold_by_record = {
        row.record_identity: int(hashlib.sha256(find(row.record_identity).encode()).hexdigest(), 16) % folds
        for row in rows
    }
    metrics: list[dict[str, Any]] = []
    for fold in range(folds):
        fit_rows = [row for row in rows if fold_by_record[row.record_identity] != fold]
        test_rows = [row for row in rows if fold_by_record[row.record_identity] == fold]
        fit_rows = [_with_partition(row, "calibration") for row in fit_rows]
        test_rows = [_with_partition(row, "holdout") for row in test_rows]
        model = fit_calibration(
            fit_rows,
            graph,
            target_precision=target_precision,
            minimum_global_samples=minimum_global_samples,
            minimum_class_samples=max(2, minimum_global_samples // 2),
            truth_scope=truth_scope,
        )
        _evaluated, result = evaluate_holdout(
            model,
            test_rows,
            graph,
            minimum_holdout_samples=1,
            truth_scope=truth_scope,
        )
        metrics.append(
            {
                "fold": fold,
                "fit_records": len({row.record_identity for row in fit_rows}),
                "test_records": len({row.record_identity for row in test_rows}),
                "identity_overlap": 0,
                "precision": result.precision,
                "coverage": result.coverage,
                "risk": result.risk,
            }
        )
    precisions = [item["precision"] for item in metrics if item["precision"] is not None]
    coverages = [item["coverage"] for item in metrics if item["coverage"] is not None]
    return {
        "schema_version": "spritelab.labeling.cross-validation.v1",
        "folds": metrics,
        "mean_precision": fmean(precisions) if precisions else None,
        "mean_coverage": fmean(coverages) if coverages else None,
        "truth_source": "synthetic_oracle_cross_validation",
        "production_validation": False,
    }


def _with_partition(example: CalibrationExample, partition: str) -> CalibrationExample:
    if not isinstance(example.human_label, SyntheticOracleLabel):
        raise ContractValidationError("immutable human review partitions cannot be rewritten")
    from spritelab.hierarchical_labeling.review import synthetic_oracle_reference_label

    source = example.human_label
    label = synthetic_oracle_reference_label(
        record_identity=source.record_identity,
        taxonomy_identity=source.taxonomy_identity,
        taxonomy_path=source.taxonomy_path,
        deepest_accepted_node=source.deepest_accepted_node,
        explicit_abstentions=source.explicit_abstentions,
        partition=partition,
        oracle_set_identity=source.oracle_set_identity,
        image_identity=source.image_identity,
        evidence_bundle_identity=source.evidence_bundle_identity,
        cohort_identity=source.cohort_identity,
        source_identity=source.source_identity,
        cluster_identity=source.cluster_identity,
        leakage_group_identity=source.leakage_group_identity,
        duplicate_cluster_identity=source.duplicate_cluster_identity,
        near_duplicate_cluster_identity=source.near_duplicate_cluster_identity,
    )
    return CalibrationExample(
        example.record_identity,
        example.node_id,
        example.raw_score,
        example.evidence_bundle_identity,
        example.source_identity,
        example.cluster_identity,
        label,
        example.duplicate_cluster_identity,
        example.near_duplicate_cluster_identity,
        example.leakage_group_identity,
    )
