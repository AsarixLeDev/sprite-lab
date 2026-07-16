"""Append-only, identity-bound semantic human review events."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spritelab.hierarchical_labeling.cohort import CohortMembership
from spritelab.hierarchical_labeling.contracts import (
    HumanReferenceLabel,
    HumanTruthVerification,
    LabelEvidenceBundle,
    SyntheticOracleLabel,
    _seal_verified_human_truth_projection,
)
from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    canonical_json,
    content_identity,
    require_probability,
    require_text,
    require_unique_text,
)
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph
from spritelab.v3.run_state import lock_file

HUMAN_REVIEW_EVENT_SCHEMA = "spritelab.labeling.human-review-event.v1"
GENESIS_EVENT_HASH = "0" * 64
REVIEW_ACTIONS = frozenset(
    {
        "accept_suggested_path",
        "choose_parent",
        "choose_alternative",
        "abstain",
        "mark_unusable",
        "flag_taxonomy_gap",
        "adjudicate",
    }
)
NON_NODE_ABSTENTIONS = frozenset({"reviewer_abstained", "taxonomy_gap", "technically_or_semantically_unusable"})


@dataclass(frozen=True, eq=False)
class HumanReviewEvent(StrictRecord):
    SCHEMA_VERSION = HUMAN_REVIEW_EVENT_SCHEMA
    IDENTITY_FIELDS = ("event_id", "record_identity", "taxonomy_identity", "evidence_bundle_identity")

    event_id: str
    record_identity: str
    action: str
    taxonomy_version: str
    taxonomy_identity: str
    selected_taxonomy_path: tuple[str, ...]
    deepest_accepted_node: str | None
    explicit_abstentions: tuple[str, ...]
    reviewer_identity: str
    timestamp: str
    previous_event_hash: str
    event_hash: str
    image_identity: str
    render_identities: tuple[str, ...]
    evidence_bundle_identity: str
    partition: str
    review_notes: str | None
    review_confidence: float | None
    exclude_semantic_supervision: bool
    legal_and_provenance_eligible: bool
    batch_identity: str | None = None
    adjudicates_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "record_identity",
            "taxonomy_version",
            "taxonomy_identity",
            "reviewer_identity",
            "timestamp",
            "previous_event_hash",
            "event_hash",
            "image_identity",
            "evidence_bundle_identity",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        if self.action not in REVIEW_ACTIONS:
            raise ContractValidationError("human review action is not controlled")
        if self.partition not in {"reference", "calibration", "holdout"}:
            raise ContractValidationError("human review truth partition is not controlled")
        require_unique_text(self.selected_taxonomy_path, "selected taxonomy path")
        require_unique_text(self.explicit_abstentions, "review explicit abstentions")
        require_unique_text(self.render_identities, "review render identities")
        require_unique_text(self.adjudicates_event_ids, "adjudicated event IDs")
        if not self.render_identities:
            raise ContractValidationError("human review event requires at least one image/render identity")
        if self.deepest_accepted_node is not None:
            require_text(self.deepest_accepted_node, "deepest accepted node")
            if not self.selected_taxonomy_path or self.selected_taxonomy_path[-1] != self.deepest_accepted_node:
                raise ContractValidationError("deepest accepted node must terminate the selected path")
        elif not self.explicit_abstentions and self.action != "mark_unusable":
            raise ContractValidationError("review without an accepted node must preserve an explicit abstention")
        if self.review_notes is not None and (not isinstance(self.review_notes, str) or not self.review_notes.strip()):
            raise ContractValidationError("review notes must be non-empty text or null")
        require_probability(self.review_confidence, "review confidence", optional=True)
        if self.action == "adjudicate" and len(self.adjudicates_event_ids) < 2:
            raise ContractValidationError("adjudication must bind at least two prior review events")
        if not self.legal_and_provenance_eligible and self.deepest_accepted_node is not None:
            raise ContractValidationError("semantic review cannot override provenance or license ineligibility")
        if self.batch_identity is not None:
            require_text(self.batch_identity, "batch identity")
        expected = content_identity(HUMAN_REVIEW_EVENT_SCHEMA, self.hash_payload())
        if self.event_hash != expected:
            raise ContractValidationError("human review event hash does not match its immutable payload")
        self.validate_record()

    def hash_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "record_identity": self.record_identity,
            "action": self.action,
            "taxonomy_version": self.taxonomy_version,
            "taxonomy_identity": self.taxonomy_identity,
            "selected_taxonomy_path": list(self.selected_taxonomy_path),
            "deepest_accepted_node": self.deepest_accepted_node,
            "explicit_abstentions": list(self.explicit_abstentions),
            "reviewer_identity": self.reviewer_identity,
            "timestamp": self.timestamp,
            "previous_event_hash": self.previous_event_hash,
            "image_identity": self.image_identity,
            "render_identities": list(self.render_identities),
            "evidence_bundle_identity": self.evidence_bundle_identity,
            "partition": self.partition,
            "review_notes": self.review_notes,
            "review_confidence": self.review_confidence,
            "exclude_semantic_supervision": self.exclude_semantic_supervision,
            "legal_and_provenance_eligible": self.legal_and_provenance_eligible,
            "batch_identity": self.batch_identity,
            "adjudicates_event_ids": list(self.adjudicates_event_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HumanReviewEvent:
        expected = {
            "schema_version",
            "event_id",
            "record_identity",
            "action",
            "taxonomy_version",
            "taxonomy_identity",
            "selected_taxonomy_path",
            "deepest_accepted_node",
            "explicit_abstentions",
            "reviewer_identity",
            "timestamp",
            "previous_event_hash",
            "event_hash",
            "image_identity",
            "render_identities",
            "evidence_bundle_identity",
            "partition",
            "review_notes",
            "review_confidence",
            "exclude_semantic_supervision",
            "legal_and_provenance_eligible",
            "batch_identity",
            "adjudicates_event_ids",
        }
        if set(value) != expected or value.get("schema_version") != HUMAN_REVIEW_EVENT_SCHEMA:
            raise ContractValidationError("human review event does not match the exact schema")
        for name in ("selected_taxonomy_path", "explicit_abstentions", "render_identities", "adjudicates_event_ids"):
            if not isinstance(value[name], list) or not all(isinstance(item, str) for item in value[name]):
                raise ContractValidationError(f"human review field {name} must be an array of strings")
        for name in ("exclude_semantic_supervision", "legal_and_provenance_eligible"):
            if type(value[name]) is not bool:
                raise ContractValidationError(f"human review field {name} must be a boolean")
        return cls(
            value["event_id"],
            value["record_identity"],
            value["action"],
            value["taxonomy_version"],
            value["taxonomy_identity"],
            tuple(value["selected_taxonomy_path"]),
            value["deepest_accepted_node"],
            tuple(value["explicit_abstentions"]),
            value["reviewer_identity"],
            value["timestamp"],
            value["previous_event_hash"],
            value["event_hash"],
            value["image_identity"],
            tuple(value["render_identities"]),
            value["evidence_bundle_identity"],
            value["partition"],
            value["review_notes"],
            value["review_confidence"],
            value["exclude_semantic_supervision"],
            value["legal_and_provenance_eligible"],
            value["batch_identity"],
            tuple(value["adjudicates_event_ids"]),
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_review_event(
    bundle: LabelEvidenceBundle,
    graph: TaxonomyGraph,
    *,
    action: str,
    reviewer_identity: str,
    partition: str,
    previous_event_hash: str,
    selected_node: str | None = None,
    explicit_abstentions: Sequence[str] = (),
    render_identities: Sequence[str] = (),
    review_notes: str | None = None,
    review_confidence: float | None = None,
    exclude_semantic_supervision: bool = False,
    legal_and_provenance_eligible: bool = True,
    batch_identity: str | None = None,
    adjudicates_event_ids: Sequence[str] = (),
    timestamp: str | None = None,
    submission_token: str | None = None,
) -> HumanReviewEvent:
    if action not in REVIEW_ACTIONS:
        raise ContractValidationError("human review action is not controlled")
    if bundle.human is not None:
        raise ContractValidationError("review events must bind the non-human evidence snapshot")
    if graph.identity != bundle.taxonomy_identity:
        raise ContractValidationError("review bundle and taxonomy identities disagree")
    resolved = graph.resolve(selected_node)
    if selected_node is not None and resolved is None:
        raise ContractValidationError("review selected an invalid or unknown taxonomy node")
    path = graph.path(resolved) if resolved else ()
    abstentions = tuple(explicit_abstentions)
    if action in {"abstain", "flag_taxonomy_gap"} and not abstentions:
        abstentions = ("taxonomy_gap" if action == "flag_taxonomy_gap" else "reviewer_abstained",)
    if action == "mark_unusable":
        abstentions = tuple(dict.fromkeys((*abstentions, "technically_or_semantically_unusable")))
        exclude_semantic_supervision = True
    for abstention in abstentions:
        if abstention not in NON_NODE_ABSTENTIONS and graph.resolve(abstention) != abstention:
            raise ContractValidationError("review explicit abstention is not a taxonomy node or controlled reason")
    if set(path) & set(abstentions):
        raise ContractValidationError("review cannot accept and explicitly abstain at the same taxonomy node")
    selected_renders = tuple(render_identities)
    if (
        bundle.visual_description is not None
        and bundle.visual_description.render_bundle_identity not in selected_renders
    ):
        raise ContractValidationError("review renders do not include the evidence bundle's bound render")
    moment = timestamp or utc_now()
    event_id = content_identity(
        "spritelab-human-review-event-id-v1",
        {
            "record_identity": bundle.record_identity,
            "reviewer_identity": reviewer_identity,
            "timestamp": moment,
            "submission_token": submission_token,
            "previous_event_hash": previous_event_hash,
        },
    )
    provisional = HumanReviewEvent.__new__(HumanReviewEvent)
    values = {
        "event_id": event_id,
        "record_identity": bundle.record_identity,
        "action": action,
        "taxonomy_version": graph.version,
        "taxonomy_identity": graph.identity,
        "selected_taxonomy_path": tuple(path),
        "deepest_accepted_node": resolved,
        "explicit_abstentions": tuple(abstentions),
        "reviewer_identity": reviewer_identity,
        "timestamp": moment,
        "previous_event_hash": previous_event_hash,
        "event_hash": "pending",
        "image_identity": bundle.image_identity,
        "render_identities": selected_renders,
        "evidence_bundle_identity": bundle.identity,
        "partition": partition,
        "review_notes": review_notes,
        "review_confidence": review_confidence,
        "exclude_semantic_supervision": exclude_semantic_supervision,
        "legal_and_provenance_eligible": legal_and_provenance_eligible,
        "batch_identity": batch_identity,
        "adjudicates_event_ids": tuple(adjudicates_event_ids),
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    values["event_hash"] = content_identity(HUMAN_REVIEW_EVENT_SCHEMA, provisional.hash_payload())
    return HumanReviewEvent(**values)


def load_review_events(path: str | Path, *, strict: bool = True) -> tuple[HumanReviewEvent, ...]:
    source = Path(path)
    if not source.is_file():
        return ()
    events: list[HumanReviewEvent] = []
    previous = GENESIS_EVENT_HASH
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ContractValidationError("human review row must be a JSON object")
            event = HumanReviewEvent.from_dict(raw)
            if event.previous_event_hash != previous:
                raise ContractValidationError("human review hash chain is broken")
        except (json.JSONDecodeError, ContractValidationError, KeyError, TypeError) as exc:
            if strict:
                raise ContractValidationError(f"invalid human review event at line {line_number}: {exc}") from exc
            break
        events.append(event)
        previous = event.event_hash
    return tuple(events)


def append_review_event(path: str | Path, event: HumanReviewEvent) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with lock_file(target.with_suffix(target.suffix + ".lock")):
        events = load_review_events(target)
        previous = events[-1].event_hash if events else GENESIS_EVENT_HASH
        if event.previous_event_hash != previous:
            raise ContractValidationError("review event previous hash does not match the append-only log tip")
        payload = canonical_json(event.to_dict()) + "\n"
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def append_review_action(
    path: str | Path,
    bundle: LabelEvidenceBundle,
    graph: TaxonomyGraph,
    **kwargs: Any,
) -> HumanReviewEvent:
    events = load_review_events(path)
    previous = events[-1].event_hash if events else GENESIS_EVENT_HASH
    event = create_review_event(bundle, graph, previous_event_hash=previous, **kwargs)
    append_review_event(path, event)
    return event


def review_log_identity(events: Sequence[HumanReviewEvent]) -> str:
    """Validate a complete chain and return its immutable content identity."""

    previous = GENESIS_EVENT_HASH
    event_ids: set[str] = set()
    event_hashes: set[str] = set()
    for event in events:
        event.__post_init__()
        if event.previous_event_hash != previous:
            raise ContractValidationError("human review projection requires a complete valid hash chain")
        if event.event_id in event_ids or event.event_hash in event_hashes:
            raise ContractValidationError("human review log event identities/hashes cannot repeat")
        event_ids.add(event.event_id)
        event_hashes.add(event.event_hash)
        previous = event.event_hash
    return content_identity(
        "spritelab-human-review-log-v1",
        {"event_hashes": [event.event_hash for event in events], "chain_tip": previous},
    )


def _authoritative_event(events: Sequence[HumanReviewEvent], record_identity: str) -> HumanReviewEvent | None:
    matching = [event for event in events if event.record_identity == record_identity]
    adjudications = [event for event in matching if event.action == "adjudicate"]
    if adjudications:
        return adjudications[-1]
    latest_by_reviewer: dict[str, HumanReviewEvent] = {}
    for event in matching:
        latest_by_reviewer[event.reviewer_identity] = event
    latest = tuple(latest_by_reviewer.values())
    if not latest:
        return None
    outcomes = {(event.selected_taxonomy_path, event.explicit_abstentions) for event in latest}
    if len(latest) >= 2 and len(outcomes) != 1:
        return None
    return max(latest, key=lambda item: item.timestamp)


def human_reference_label(
    event: HumanReviewEvent,
    *,
    graph: TaxonomyGraph,
    verified_events: Sequence[HumanReviewEvent],
    membership: CohortMembership,
) -> HumanReferenceLabel:
    """Project one authoritative event from a verified append-only log into trusted truth."""

    events = tuple(verified_events)
    log_identity = review_log_identity(events)
    authoritative = _authoritative_event(events, event.record_identity)
    if authoritative is None or authoritative.event_hash != event.event_hash:
        raise ContractValidationError("human truth projection requires the authoritative reviewed event")
    if event.exclude_semantic_supervision or not event.legal_and_provenance_eligible:
        raise ContractValidationError("excluded or provenance-ineligible review cannot become human truth")
    if event.taxonomy_identity != graph.identity:
        raise ContractValidationError("review truth taxonomy does not match the projection taxonomy")
    expected_path = graph.path(event.deepest_accepted_node) if event.deepest_accepted_node is not None else ()
    if event.selected_taxonomy_path != expected_path:
        raise ContractValidationError("review truth path is not canonical for the projection taxonomy")
    for abstention in event.explicit_abstentions:
        if abstention not in NON_NODE_ABSTENTIONS and graph.resolve(abstention) != abstention:
            raise ContractValidationError("review truth contains an invalid explicit abstention")
    if (
        membership.record_identity != event.record_identity
        or membership.image_identity != event.image_identity
        or membership.partition != event.partition
    ):
        raise ContractValidationError("review truth does not bind the verified cohort membership")
    verification = _seal_verified_human_truth_projection(
        HumanTruthVerification(
            "append_only_human_review",
            event.record_identity,
            event.taxonomy_identity,
            event.selected_taxonomy_path,
            event.explicit_abstentions,
            event.partition,
            event.reviewer_identity,
            event.event_hash,
            log_identity,
            events[-1].event_hash if events else GENESIS_EVENT_HASH,
            event.image_identity,
            event.render_identities,
            event.evidence_bundle_identity,
            membership.cohort_identity,
            membership.source_identity,
            membership.cluster_identity,
            membership.leakage_group_identity,
            membership.duplicate_cluster_identity,
            membership.near_duplicate_cluster_identity,
        )
    )
    return HumanReferenceLabel(
        event.record_identity,
        event.event_hash,
        event.taxonomy_identity,
        event.selected_taxonomy_path,
        event.deepest_accepted_node,
        event.explicit_abstentions,
        event.partition,
        event.reviewer_identity,
        verification,
    )


def synthetic_oracle_reference_label(
    *,
    record_identity: str,
    taxonomy_identity: str,
    taxonomy_path: Sequence[str],
    deepest_accepted_node: str | None,
    explicit_abstentions: Sequence[str],
    partition: str,
    oracle_set_identity: str,
    image_identity: str,
    evidence_bundle_identity: str,
    cohort_identity: str,
    source_identity: str,
    cluster_identity: str,
    leakage_group_identity: str,
    duplicate_cluster_identity: str | None = None,
    near_duplicate_cluster_identity: str | None = None,
) -> SyntheticOracleLabel:
    """Create an explicitly non-human oracle fixture for synthetic architecture tests."""

    path = tuple(taxonomy_path)
    abstentions = tuple(explicit_abstentions)
    event_identity = content_identity(
        "spritelab-synthetic-oracle-event-v1",
        {
            "oracle_set_identity": oracle_set_identity,
            "record_identity": record_identity,
            "taxonomy_identity": taxonomy_identity,
            "taxonomy_path": path,
            "explicit_abstentions": abstentions,
            "partition": partition,
            "evidence_bundle_identity": evidence_bundle_identity,
        },
    )
    return SyntheticOracleLabel(
        record_identity,
        event_identity,
        taxonomy_identity,
        path,
        deepest_accepted_node,
        abstentions,
        partition,
        oracle_set_identity,
        image_identity,
        evidence_bundle_identity,
        cohort_identity,
        source_identity,
        cluster_identity,
        leakage_group_identity,
        duplicate_cluster_identity,
        near_duplicate_cluster_identity,
    )


def review_consensus(
    events: Sequence[HumanReviewEvent],
    record_identity: str,
    *,
    graph: TaxonomyGraph | None = None,
    membership: CohortMembership | None = None,
) -> dict[str, Any]:
    matching = [event for event in events if event.record_identity == record_identity]
    adjudications = [event for event in matching if event.action == "adjudicate"]
    if adjudications:
        winner = adjudications[-1]
        state = "adjudicated"
        winner_event: HumanReviewEvent | None = winner
    else:
        latest_by_reviewer: dict[str, HumanReviewEvent] = {}
        for item in matching:
            latest_by_reviewer[item.reviewer_identity] = item
        latest = tuple(latest_by_reviewer.values())
        if not latest:
            return {"state": "unreviewed", "event": None, "human_reference": None}
        outcomes = {(item.selected_taxonomy_path, item.explicit_abstentions) for item in latest}
        if len(latest) >= 2 and len(outcomes) == 1:
            state = "double_review_agreement"
            winner_event = max(latest, key=lambda item: item.timestamp)
        elif len(latest) >= 2:
            return {"state": "adjudication_required", "event": None, "human_reference": None}
        else:
            state = "single_review"
            winner_event = latest[0]
    projection = None
    if membership is not None:
        if graph is None:
            raise ContractValidationError("verified consensus projection requires the taxonomy graph")
        projection = human_reference_label(
            winner_event,
            graph=graph,
            verified_events=events,
            membership=membership,
        )
    return {"state": state, "event": winner_event, "human_reference": projection}


@dataclass(frozen=True)
class BatchReviewItem:
    bundle: LabelEvidenceBundle
    render_identities: tuple[str, ...]
    legal_and_provenance_eligible: bool


def batch_review_identity(
    cluster_identity: str,
    exemplar_event: HumanReviewEvent,
    items: Sequence[BatchReviewItem],
    selected_node: str | None,
) -> str:
    return content_identity(
        "spritelab-human-batch-review-v1",
        {
            "cluster_identity": cluster_identity,
            "exemplar_event_hash": exemplar_event.event_hash,
            "record_identities": [item.bundle.record_identity for item in items],
            "evidence_bundle_identities": [item.bundle.identity for item in items],
            "selected_node": selected_node,
        },
    )


def append_batch_review(
    path: str | Path,
    graph: TaxonomyGraph,
    *,
    cluster_identity: str,
    exemplar_event: HumanReviewEvent,
    items: Sequence[BatchReviewItem],
    selected_node: str | None,
    reviewer_identity: str,
    partition_by_record: Mapping[str, str],
    explicit_confirmation: bool,
) -> tuple[HumanReviewEvent, ...]:
    if not explicit_confirmation:
        raise ContractValidationError("batch review requires explicit confirmation after preview")
    if exemplar_event.deepest_accepted_node is None:
        raise ContractValidationError("batch review requires a reviewed exemplar with an accepted taxonomy path")
    if not items:
        raise ContractValidationError("batch review requires affected records")
    if any(not item.legal_and_provenance_eligible for item in items):
        raise ContractValidationError("batch review cannot override provenance or license restrictions")
    batch_id = batch_review_identity(cluster_identity, exemplar_event, items, selected_node)
    appended: list[HumanReviewEvent] = []
    for item in items:
        event = append_review_action(
            path,
            item.bundle,
            graph,
            action="choose_alternative"
            if selected_node != exemplar_event.deepest_accepted_node
            else "accept_suggested_path",
            reviewer_identity=reviewer_identity,
            partition=partition_by_record[item.bundle.record_identity],
            selected_node=selected_node,
            render_identities=item.render_identities,
            review_notes=f"Explicit batch review for cluster {cluster_identity}.",
            legal_and_provenance_eligible=True,
            batch_identity=batch_id,
            submission_token=f"{batch_id}:{item.bundle.record_identity}",
        )
        appended.append(event)
    return tuple(appended)


def latest_events_by_record(events: Sequence[HumanReviewEvent]) -> dict[str, HumanReviewEvent]:
    grouped: dict[str, list[HumanReviewEvent]] = defaultdict(list)
    for event in events:
        grouped[event.record_identity].append(event)
    return {record: values[-1] for record, values in grouped.items()}
