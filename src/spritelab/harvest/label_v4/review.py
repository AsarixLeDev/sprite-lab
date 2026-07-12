"""Append-only human review support for Labeling v4.

The functions in this module deliberately build a review *overlay*.  They never
rewrite a model proposal or a reconciled record.  A review UI can therefore
show the latest decision while the complete proposal and correction history
remain independently auditable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REVIEW_EVENT_SCHEMA_VERSION = "label_v4_human_truth_v1"


class ReviewEventSchemaError(ValueError):
    """Raised when a review event does not use the canonical event schema."""


ACCEPT_PROPOSAL = "accept_proposed_value"
ACCEPT_MODEL_ABSTENTION = "accept_model_abstention"
SELECT_ALTERNATIVE = "select_alternative"
EDIT = "edit"
ABSTAIN = "mark_human_abstention"
MARK_UNSUPPORTED = "mark_unsupported"
MARK_WRONG_TAXONOMY = "mark_wrong_taxonomy"
MARK_UNSUITABLE_IMAGE = "mark_unsuitable_image"
MARK_SUITABLE_IMAGE = "mark_suitable_image"
MARK_UNCERTAIN_QUALITY = "mark_uncertain_quality"
MARK_NOT_APPLICABLE = "mark_not_applicable"
QUALITY_SUITABLE = "quality_suitable"
QUALITY_UNCERTAIN_USABLE = "quality_uncertain_usable"
QUALITY_UNSUITABLE = "quality_unsuitable"
QUALITY_UNCERTAIN_NOT_USABLE = "quality_uncertain_not_usable"

QUALITY_ACTIONS = frozenset(
    {QUALITY_SUITABLE, QUALITY_UNCERTAIN_USABLE, QUALITY_UNSUITABLE, QUALITY_UNCERTAIN_NOT_USABLE}
)

REVIEW_ACTIONS = frozenset(
    {
        ACCEPT_PROPOSAL,
        ACCEPT_MODEL_ABSTENTION,
        SELECT_ALTERNATIVE,
        EDIT,
        ABSTAIN,
        MARK_UNSUPPORTED,
        MARK_WRONG_TAXONOMY,
        MARK_UNSUITABLE_IMAGE,
        MARK_SUITABLE_IMAGE,
        MARK_UNCERTAIN_QUALITY,
        MARK_NOT_APPLICABLE,
        *QUALITY_ACTIONS,
    }
)

_ACTION_STATES = {
    ACCEPT_PROPOSAL: "accepted",
    ACCEPT_MODEL_ABSTENTION: "accepted_abstention",
    SELECT_ALTERNATIVE: "accepted",
    EDIT: "accepted",
    ABSTAIN: "abstained",
    MARK_UNSUPPORTED: "unsupported",
    MARK_WRONG_TAXONOMY: "wrong_taxonomy",
    MARK_UNSUITABLE_IMAGE: "unsuitable_image",
    MARK_SUITABLE_IMAGE: "suitable_image",
    MARK_UNCERTAIN_QUALITY: "uncertain_quality",
    MARK_NOT_APPLICABLE: "not_applicable",
    QUALITY_SUITABLE: "quality_suitable",
    QUALITY_UNCERTAIN_USABLE: "quality_uncertain_usable",
    QUALITY_UNSUITABLE: "quality_unsuitable",
    QUALITY_UNCERTAIN_NOT_USABLE: "quality_uncertain_not_usable",
}

_RECORD_ACTIONS = frozenset({MARK_UNSUITABLE_IMAGE, MARK_SUITABLE_IMAGE, MARK_UNCERTAIN_QUALITY, *QUALITY_ACTIONS})
_HUMAN_OUTCOMES = {
    ACCEPT_PROPOSAL: "correct",
    ACCEPT_MODEL_ABSTENTION: "model_abstention_accepted",
    SELECT_ALTERNATIVE: "incorrect",
    EDIT: "incorrect",
    ABSTAIN: "human_abstained",
    MARK_UNSUPPORTED: "unsupported",
    MARK_WRONG_TAXONOMY: "unsupported",
    MARK_UNSUITABLE_IMAGE: "not_scorable_due_to_image",
    MARK_SUITABLE_IMAGE: "not_applicable",
    MARK_UNCERTAIN_QUALITY: "not_scorable_due_to_image",
    MARK_NOT_APPLICABLE: "not_applicable",
    QUALITY_SUITABLE: "quality_suitable",
    QUALITY_UNCERTAIN_USABLE: "quality_uncertain_usable",
    QUALITY_UNSUITABLE: "quality_unsuitable",
    QUALITY_UNCERTAIN_NOT_USABLE: "quality_uncertain_not_usable",
}

_UNSAFE_SEMANTIC_STATES = frozenset(
    {"missing_prediction", "provider_failed", "not_scorable", "not_scorable_due_to_image"}
)

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_MISSING = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def immutable_proposal_digest(record: Mapping[str, Any]) -> str:
    """Return a full SHA-256 for the raw proposal represented by ``record``.

    Raw provider text is hashed byte-for-byte.  Structured proposals use a
    canonical JSON representation so dictionary insertion order is irrelevant.
    """

    raw: Any = record.get("raw_proposal", _MISSING)
    if raw is _MISSING:
        proposal = record.get("vlm_proposal")
        if isinstance(proposal, Mapping):
            raw = proposal.get("raw_output", proposal.get("parsed_output", proposal))
        else:
            raw = record.get("proposal", record.get("field_proposals", record))
    if isinstance(raw, bytes):
        return _sha256(raw)
    if isinstance(raw, str):
        return _sha256(raw.encode("utf-8"))
    return _sha256(_canonical_json(raw))


@dataclass(frozen=True)
class ReviewEvent:
    """One immutable field- or record-level human review action."""

    sprite_id: str
    action: str
    field_name: str = ""
    proposed_value: Any = None
    reviewed_value: Any = None
    alternatives_visible: tuple[Any, ...] = ()
    original_state: str = "proposed"
    reviewed_state: str = ""
    uncertainty_before: int | None = None
    risk_band_before: str = ""
    evidence_refs_visible: tuple[str, ...] = ()
    conflicts_visible: tuple[str, ...] = ()
    propagation_scope_visible: str = ""
    training_consequence_visible: str = ""
    proposal_hash: str = ""
    proposal_schema_version: str = ""
    taxonomy_hash: str = ""
    risk_model_version: str = ""
    reviewer_id: str = ""
    session_id: str = ""
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = ""
    timestamp: str = ""
    schema_version: str = REVIEW_EVENT_SCHEMA_VERSION
    human_outcome: str = ""
    review_mode: str = "assisted"
    proposal_visible_before_judgment: bool = True
    review_started_at: str = ""
    review_completed_at: str = ""
    review_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_EVENT_SCHEMA_VERSION:
            raise ReviewEventSchemaError(
                f"unsupported review-event schema {self.schema_version!r}; expected {REVIEW_EVENT_SCHEMA_VERSION!r}"
            )
        if self.action not in REVIEW_ACTIONS:
            raise ValueError(f"unsupported review action: {self.action}")
        if self.action not in _RECORD_ACTIONS and not str(self.field_name).strip():
            raise ValueError(f"{self.action} requires field_name")
        if self.uncertainty_before is not None and not 1 <= int(self.uncertainty_before) <= 20:
            raise ValueError("uncertainty_before must be in [1, 20]")
        object.__setattr__(self, "sprite_id", str(self.sprite_id).strip())
        object.__setattr__(self, "field_name", str(self.field_name).strip())
        object.__setattr__(self, "reviewed_state", self.reviewed_state or _ACTION_STATES[self.action])
        object.__setattr__(self, "event_id", self.event_id or uuid.uuid4().hex)
        object.__setattr__(self, "timestamp", self.timestamp or _utc_now())
        object.__setattr__(self, "alternatives_visible", tuple(copy.deepcopy(self.alternatives_visible)))
        object.__setattr__(self, "evidence_refs_visible", tuple(str(v) for v in self.evidence_refs_visible))
        object.__setattr__(self, "conflicts_visible", tuple(str(v) for v in self.conflicts_visible))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))
        metadata = dict(self.metadata)
        object.__setattr__(
            self,
            "human_outcome",
            self.human_outcome or str(metadata.get("human_outcome") or _HUMAN_OUTCOMES[self.action]),
        )
        object.__setattr__(self, "review_mode", str(metadata.get("review_mode") or self.review_mode))
        object.__setattr__(
            self,
            "proposal_visible_before_judgment",
            bool(metadata.get("proposal_visible_before_judgment", self.proposal_visible_before_judgment)),
        )
        object.__setattr__(
            self,
            "review_started_at",
            str(metadata.get("review_started_at") or self.review_started_at or self.timestamp),
        )
        object.__setattr__(
            self,
            "review_completed_at",
            str(metadata.get("review_completed_at") or self.review_completed_at or self.timestamp),
        )
        duration = metadata.get("review_duration_seconds", self.review_duration_seconds)
        object.__setattr__(self, "review_duration_seconds", float(duration) if duration is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "sprite_id": self.sprite_id,
            "action": self.action,
            "field_name": self.field_name,
            "proposed_value": copy.deepcopy(self.proposed_value),
            "reviewed_value": copy.deepcopy(self.reviewed_value),
            "alternatives_visible": copy.deepcopy(list(self.alternatives_visible)),
            "original_state": self.original_state,
            "reviewed_state": self.reviewed_state,
            "uncertainty_before": self.uncertainty_before,
            "risk_band_before": self.risk_band_before,
            "evidence_refs_visible": list(self.evidence_refs_visible),
            "conflicts_visible": list(self.conflicts_visible),
            "propagation_scope_visible": self.propagation_scope_visible,
            "training_consequence_visible": self.training_consequence_visible,
            "proposal_hash": self.proposal_hash,
            "proposal_schema_version": self.proposal_schema_version,
            "taxonomy_hash": self.taxonomy_hash,
            "risk_model_version": self.risk_model_version,
            "reviewer_id": self.reviewer_id,
            "session_id": self.session_id,
            "notes": self.notes,
            "metadata": copy.deepcopy(dict(self.metadata)),
            "timestamp": self.timestamp,
            "human_outcome": self.human_outcome,
            "review_mode": self.review_mode,
            "proposal_visible_before_judgment": self.proposal_visible_before_judgment,
            "review_started_at": self.review_started_at,
            "review_completed_at": self.review_completed_at,
            "review_duration_seconds": self.review_duration_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReviewEvent:
        if "schema_version" not in value or not str(value.get("schema_version") or "").strip():
            raise ReviewEventSchemaError(
                f"missing review-event schema_version; expected {REVIEW_EVENT_SCHEMA_VERSION!r}"
            )
        schema_version = str(value["schema_version"])
        if schema_version != REVIEW_EVENT_SCHEMA_VERSION:
            raise ReviewEventSchemaError(
                f"unsupported review-event schema {schema_version!r}; expected {REVIEW_EVENT_SCHEMA_VERSION!r}"
            )
        return cls(
            schema_version=schema_version,
            event_id=str(value.get("event_id", "")),
            sprite_id=str(value.get("sprite_id", "")),
            action=str(value.get("action", "")),
            field_name=str(value.get("field_name", "")),
            proposed_value=copy.deepcopy(value.get("proposed_value")),
            reviewed_value=copy.deepcopy(value.get("reviewed_value")),
            alternatives_visible=tuple(copy.deepcopy(value.get("alternatives_visible") or ())),
            original_state=str(value.get("original_state", "proposed")),
            reviewed_state=str(value.get("reviewed_state", "")),
            uncertainty_before=(
                int(value["uncertainty_before"]) if value.get("uncertainty_before") is not None else None
            ),
            risk_band_before=str(value.get("risk_band_before", "")),
            evidence_refs_visible=tuple(str(v) for v in value.get("evidence_refs_visible") or ()),
            conflicts_visible=tuple(str(v) for v in value.get("conflicts_visible") or ()),
            propagation_scope_visible=str(value.get("propagation_scope_visible", "")),
            training_consequence_visible=str(value.get("training_consequence_visible", "")),
            proposal_hash=str(value.get("proposal_hash", "")),
            proposal_schema_version=str(value.get("proposal_schema_version", "")),
            taxonomy_hash=str(value.get("taxonomy_hash", "")),
            risk_model_version=str(value.get("risk_model_version", "")),
            reviewer_id=str(value.get("reviewer_id", "")),
            session_id=str(value.get("session_id", "")),
            notes=str(value.get("notes", "")),
            metadata=copy.deepcopy(dict(value.get("metadata") or {})),
            timestamp=str(value.get("timestamp", "")),
            human_outcome=str(value.get("human_outcome", "")),
            review_mode=str(value.get("review_mode", "assisted")),
            proposal_visible_before_judgment=bool(value.get("proposal_visible_before_judgment", True)),
            review_started_at=str(value.get("review_started_at", "")),
            review_completed_at=str(value.get("review_completed_at", "")),
            review_duration_seconds=(
                float(value["review_duration_seconds"]) if value.get("review_duration_seconds") is not None else None
            ),
        )


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_process_lock(path: Path) -> Iterator[None]:
    """Use a one-byte sidecar lock for cross-process append serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":  # pragma: no cover - branch depends on host OS
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised on non-Windows CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_review_event(path: str | Path, event: ReviewEvent) -> None:
    """Append one complete JSONL event with thread and process serialization."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(event.to_dict()) + "\n").encode("utf-8")
    with _thread_lock(target):
        with _exclusive_process_lock(target.with_suffix(target.suffix + ".lock")):
            descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def load_review_events(path: str | Path, *, strict: bool = True) -> tuple[ReviewEvent, ...]:
    target = Path(path)
    if not target.is_file():
        return ()
    events: list[ReviewEvent] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(ReviewEvent.from_dict(json.loads(line)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if strict:
                raise ValueError(f"invalid review event at {target}:{line_number}: {exc}") from None
    return tuple(events)


def _proposal_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    direct = record.get("field_proposals")
    if isinstance(direct, Mapping):
        return dict(direct)
    reconciliation = record.get("reconciliation")
    if isinstance(reconciliation, Mapping) and isinstance(reconciliation.get("field_proposals"), Mapping):
        return dict(reconciliation["field_proposals"])
    fields = record.get("fields")
    return dict(fields) if isinstance(fields, Mapping) else {}


def _quality_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    quality = record.get("label_quality")
    if isinstance(quality, Mapping) and isinstance(quality.get("fields"), Mapping):
        return dict(quality["fields"])
    quality = record.get("field_uncertainty")
    return dict(quality) if isinstance(quality, Mapping) else {}


def _alternatives(field_value: Any) -> list[Any]:
    if not isinstance(field_value, Mapping):
        return []
    result: list[Any] = []
    for item in field_value.get("alternatives") or ():
        result.append(
            copy.deepcopy(item.get("value")) if isinstance(item, Mapping) and "value" in item else copy.deepcopy(item)
        )
    return result


def _score_band(score: int | None) -> str:
    if score is None:
        return "not_scorable"
    if score <= 4:
        return "strong"
    if score <= 8:
        return "usable_weak"
    if score <= 12:
        return "auxiliary_only"
    if score <= 16:
        return "excluded_from_primary_supervision"
    return "abstain_or_quarantine"


def _training_consequence(score: int | None, quality: Mapping[str, Any]) -> str:
    explicit = quality.get("training_state") or quality.get("training_consequence")
    if explicit:
        return str(explicit)
    if score is None:
        return "excluded_not_scorable"
    if score <= 4:
        return "full_supervised_weight"
    if score <= 8:
        return "reduced_supervised_weight"
    if score <= 12:
        return "auxiliary_or_weak_supervision_only"
    if score <= 16:
        return "field_masked_from_supervised_target"
    return "field_abstained"


def latest_review_events(events: Sequence[ReviewEvent], sprite_id: str) -> dict[str, ReviewEvent]:
    latest: dict[str, ReviewEvent] = {}
    for event in events:
        if event.sprite_id == sprite_id:
            latest[event.field_name or "__record__"] = event
    return latest


def compact_review_presenter(record: Mapping[str, Any], events: Sequence[ReviewEvent] = ()) -> dict[str, Any]:
    """Return the small, stable view model intended for the assisted GUI."""

    sprite_id = str(record.get("sprite_id", ""))
    proposals = _proposal_fields(record)
    qualities = _quality_fields(record)
    latest = latest_review_events(events, sprite_id)
    propagation = record.get("propagation") if isinstance(record.get("propagation"), Mapping) else {}
    rendered: dict[str, Any] = {}
    for field_name in sorted(set(proposals) | set(qualities)):
        proposal = proposals.get(field_name)
        proposal_map = proposal if isinstance(proposal, Mapping) else {"value": proposal}
        quality = qualities.get(field_name)
        quality_map = quality if isinstance(quality, Mapping) else {}
        raw_score = quality_map.get("uncertainty_1_20", proposal_map.get("uncertainty_1_20"))
        score = int(raw_score) if raw_score is not None else None
        if score is not None and not 1 <= score <= 20:
            raise ValueError(f"{field_name}: uncertainty_1_20 must be in [1, 20]")
        review = latest.get(field_name)
        support = proposal_map.get("support") or proposal_map.get("evidence_summary") or ()
        refs = proposal_map.get("evidence_refs") or ()
        evidence_summary = [str(value) for value in support]
        evidence_summary.extend(str(value) for value in refs if str(value) not in evidence_summary)
        field_propagation = (
            propagation.get("fields", {}).get(field_name, {}) if isinstance(propagation, Mapping) else {}
        )
        scope = proposal_map.get("propagation_scope") or (
            field_propagation.get("scope") if isinstance(field_propagation, Mapping) else ""
        )
        if not scope and isinstance(propagation, Mapping):
            scope = propagation.get("propagation_relation", "none")
        rendered[field_name] = {
            "proposed_value": copy.deepcopy(proposal_map.get("value")),
            "alternatives": _alternatives(proposal_map),
            "reviewed_value": copy.deepcopy(review.reviewed_value) if review else None,
            "review_state": review.reviewed_state if review else "unreviewed",
            "uncertainty_1_20": score,
            "uncertainty_state": str(quality_map.get("calibration_state", "scored" if score else "not_scorable")),
            "risk_band": str(quality_map.get("uncertainty_band") or _score_band(score)),
            "evidence_summary": evidence_summary,
            "conflicts": [str(value) for value in proposal_map.get("conflicts") or ()],
            "propagation_scope": str(scope or "none"),
            "training_consequence": _training_consequence(score, quality_map),
            "loss_weight": quality_map.get("loss_weight"),
        }
    record_quality = record.get("label_quality") if isinstance(record.get("label_quality"), Mapping) else {}
    unsuitable = latest.get("__record__")
    return {
        "sprite_id": sprite_id,
        "record_uncertainty_1_20": record_quality.get("record_uncertainty_1_20"),
        "critical_field_max_uncertainty": record_quality.get("critical_field_max_uncertainty"),
        "record_review_state": unsuitable.reviewed_state if unsuitable else "reviewable",
        "raw_proposal_hash": immutable_proposal_digest(record),
        "fields": rendered,
    }


def create_review_event(
    record: Mapping[str, Any],
    action: str,
    field_name: str = "",
    *,
    reviewed_value: Any = _MISSING,
    reviewer_id: str = "",
    session_id: str = "",
    notes: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ReviewEvent:
    """Create an event from exactly what the reviewer was shown."""

    view = compact_review_presenter(record)
    field_view = view["fields"].get(field_name, {})
    proposed = copy.deepcopy(field_view.get("proposed_value"))
    source_field = record.get("fields", {}).get(field_name, {}) if isinstance(record.get("fields"), Mapping) else {}
    value_state = str(source_field.get("value_state", "known" if proposed is not None else "unsupported"))
    semantic_action = action not in _RECORD_ACTIONS
    if semantic_action and value_state in _UNSAFE_SEMANTIC_STATES:
        raise ValueError(f"{field_name}: semantic action {action} forbidden for value_state={value_state}")
    if action in {ACCEPT_PROPOSAL, ACCEPT_MODEL_ABSTENTION}:
        # Local import avoids a module cycle while keeping every acceptance path on
        # the same strict state validator used by readiness and completion.
        from spritelab.harvest.label_v4.two_pass import validate_semantic_field

        validation = validate_semantic_field(record, field_name)
        expected = "known" if action == ACCEPT_PROPOSAL else "model_abstained"
        if not validation.valid or validation.value_state != expected:
            raise ValueError(f"{field_name}: {action} requires valid {expected}: {validation.reason}")
    if reviewed_value is _MISSING:
        reviewed_value = proposed if action in {ACCEPT_PROPOSAL, ACCEPT_MODEL_ABSTENTION} else None
    alternatives = tuple(copy.deepcopy(field_view.get("alternatives") or ()))
    if action == SELECT_ALTERNATIVE and reviewed_value not in alternatives:
        raise ValueError(f"selected value is not a visible alternative for {field_name}")
    proposal_version = str(record.get("proposal_schema_version") or "")
    vlm_proposal = record.get("vlm_proposal")
    if not proposal_version and isinstance(vlm_proposal, Mapping):
        proposal_version = str(vlm_proposal.get("schema_version", ""))
    sprite_id = str(record.get("sprite_id", "")).strip()
    audit_record_id = str(record.get("audit_id", "")).strip()
    event_metadata = copy.deepcopy(dict(metadata or {}))
    if audit_record_id:
        event_metadata.setdefault("audit_id", audit_record_id)
        event_metadata.setdefault("audit_record_id", audit_record_id)
    event_metadata.setdefault("sprite_id", sprite_id)
    event_metadata.setdefault("field_name", str(field_name).strip())
    event_metadata.setdefault("proposal_hash", view["raw_proposal_hash"])
    return ReviewEvent(
        sprite_id=sprite_id,
        action=action,
        field_name=field_name,
        proposed_value=proposed,
        reviewed_value=copy.deepcopy(reviewed_value),
        alternatives_visible=alternatives,
        original_state=str(field_view.get("review_state", "proposed")),
        uncertainty_before=field_view.get("uncertainty_1_20"),
        risk_band_before=str(field_view.get("risk_band", "")),
        evidence_refs_visible=tuple(field_view.get("evidence_summary") or ()),
        conflicts_visible=tuple(field_view.get("conflicts") or ()),
        propagation_scope_visible=str(field_view.get("propagation_scope", "")),
        training_consequence_visible=str(field_view.get("training_consequence", "")),
        proposal_hash=view["raw_proposal_hash"],
        proposal_schema_version=proposal_version,
        taxonomy_hash=str(record.get("taxonomy_hash", "")),
        risk_model_version=str(record.get("risk_model_version", "")),
        reviewer_id=reviewer_id,
        session_id=str(session_id or audit_record_id),
        notes=notes,
        metadata=event_metadata,
    )


def record_review_action(
    path: str | Path,
    record: Mapping[str, Any],
    action: str,
    field_name: str = "",
    **kwargs: Any,
) -> ReviewEvent:
    """Create and append an event while asserting proposal immutability."""

    before = immutable_proposal_digest(record)
    event = create_review_event(record, action, field_name, **kwargs)
    append_review_event(path, event)
    if immutable_proposal_digest(record) != before:  # defensive invariant
        raise RuntimeError("raw proposal mutated while recording review action")
    return event


def accept_proposal(path: str | Path, record: Mapping[str, Any], field_name: str, **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, ACCEPT_PROPOSAL, field_name, **kwargs)


def accept_model_abstention(path: str | Path, record: Mapping[str, Any], field_name: str, **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, ACCEPT_MODEL_ABSTENTION, field_name, **kwargs)


def select_alternative(
    path: str | Path, record: Mapping[str, Any], field_name: str, value: Any, **kwargs: Any
) -> ReviewEvent:
    return record_review_action(path, record, SELECT_ALTERNATIVE, field_name, reviewed_value=value, **kwargs)


def edit_field(path: str | Path, record: Mapping[str, Any], field_name: str, value: Any, **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, EDIT, field_name, reviewed_value=value, **kwargs)


def abstain_field(path: str | Path, record: Mapping[str, Any], field_name: str, **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, ABSTAIN, field_name, reviewed_value=None, **kwargs)


def mark_unsupported(path: str | Path, record: Mapping[str, Any], field_name: str, **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, MARK_UNSUPPORTED, field_name, reviewed_value=None, **kwargs)


def mark_wrong_taxonomy(path: str | Path, record: Mapping[str, Any], field_name: str, **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, MARK_WRONG_TAXONOMY, field_name, reviewed_value=None, **kwargs)


def mark_unsuitable_image(path: str | Path, record: Mapping[str, Any], **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, MARK_UNSUITABLE_IMAGE, reviewed_value=None, **kwargs)


def mark_suitable_image(path: str | Path, record: Mapping[str, Any], **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, MARK_SUITABLE_IMAGE, reviewed_value="suitable", **kwargs)


def mark_uncertain_quality(path: str | Path, record: Mapping[str, Any], **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, MARK_UNCERTAIN_QUALITY, reviewed_value="uncertain_quality", **kwargs)


def mark_not_applicable(path: str | Path, record: Mapping[str, Any], field_name: str, **kwargs: Any) -> ReviewEvent:
    return record_review_action(path, record, MARK_NOT_APPLICABLE, field_name, reviewed_value=None, **kwargs)


def record_quality_decision(path: str | Path, record: Mapping[str, Any], outcome: str, **kwargs: Any) -> ReviewEvent:
    if outcome not in QUALITY_ACTIONS:
        raise ValueError(f"invalid quality outcome: {outcome}")
    return record_review_action(path, record, outcome, reviewed_value=outcome, **kwargs)
