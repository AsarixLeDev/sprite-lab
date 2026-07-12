"""Auto-Labeling v3: assisted golden integration.

Extends the existing assisted golden workflow with v3 record prefetching,
sample selection, and correction-event lineage for calibration rebuilds.

Operates as add-ons to ``assisted_golden.py`` — never modifies its existing
data structures.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.label_v3.field_decisions import FieldDecision
from spritelab.harvest.label_v3.record_decisions import (
    RecordDecision,
    record_decision_from_json,
)
from spritelab.harvest.sources import utc_timestamp

V3_CORRECTIONS_FILENAME = "v3_corrections.jsonl"
V3_RECORDS_FILENAME = "v3_records.jsonl"
V3_CALIBRATION_SAMPLE_FILENAME = "v3_calibration_sample.jsonl"


@dataclass(frozen=True)
class V3CorrectionEvent:
    """One field-level correction applied during the GUI calibration workflow.

    Appended, never overwritten. Carries enough state to reproduce what the
    reviewer saw and why the sample was selected.
    """

    sprite_id: str
    field_name: str
    original_value: Any
    corrected_value: Any
    original_state: str
    corrected_state: str
    evidence_refs_visible: tuple[str, ...] = ()
    selection_reason: str = ""
    reviewer_id: str = ""
    session_id: str = ""
    timestamp: str = ""
    calibration_policy_hash: str = ""
    taxonomy_hash: str = ""
    review_action: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", str(self.timestamp) or utc_timestamp())
        object.__setattr__(self, "selection_reason", str(self.selection_reason).strip())
        object.__setattr__(self, "reviewer_id", str(self.reviewer_id).strip())
        object.__setattr__(self, "evidence_refs_visible", tuple(str(v) for v in self.evidence_refs_visible))
        if not self.review_action:
            action = (
                "accepted_as_prefilled"
                if self.corrected_state == "accepted" and self.corrected_value == self.original_value
                else "corrected"
                if self.corrected_state == "accepted"
                else "cleared"
                if self.corrected_value in (None, "")
                else "marked_unknown"
                if self.corrected_state == "unknown"
                else "abstained"
                if self.corrected_state == "abstained"
                else self.corrected_state
            )
            object.__setattr__(self, "review_action", action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprite_id": self.sprite_id,
            "field_name": self.field_name,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "original_state": self.original_state,
            "corrected_state": self.corrected_state,
            "evidence_refs_visible": list(self.evidence_refs_visible),
            "selection_reason": self.selection_reason,
            "reviewer_id": self.reviewer_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "calibration_policy_hash": self.calibration_policy_hash,
            "taxonomy_hash": self.taxonomy_hash,
            "review_action": self.review_action,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> V3CorrectionEvent:
        return cls(
            sprite_id=str(data.get("sprite_id", "")),
            field_name=str(data.get("field_name", "")),
            original_value=data.get("original_value"),
            corrected_value=data.get("corrected_value"),
            original_state=str(data.get("original_state", "")),
            corrected_state=str(data.get("corrected_state", "")),
            evidence_refs_visible=tuple(str(v) for v in data.get("evidence_refs_visible") or ()),
            selection_reason=str(data.get("selection_reason", "")),
            reviewer_id=str(data.get("reviewer_id", "")),
            session_id=str(data.get("session_id", "")),
            timestamp=str(data.get("timestamp", "")),
            calibration_policy_hash=str(data.get("calibration_policy_hash", "")),
            taxonomy_hash=str(data.get("taxonomy_hash", "")),
            review_action=str(data.get("review_action", "")),
        )


def append_v3_correction(path: str | Path, event: V3CorrectionEvent) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def load_v3_corrections(path: str | Path) -> list[V3CorrectionEvent]:
    p = Path(path)
    if not p.is_file():
        return []
    events: list[V3CorrectionEvent] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = V3CorrectionEvent.from_dict(json.loads(line))
        except (json.JSONDecodeError, KeyError):
            continue
        events.append(event)
    return events


def v3_correction_summary(events: Sequence[V3CorrectionEvent]) -> dict[str, Any]:
    total = len(events)
    field_counts: Counter[str] = Counter()
    state_changes: Counter[str] = Counter()
    corrected_sprite_ids: set[str] = set()
    for event in events:
        field_counts[event.field_name] += 1
        state_changes[f"{event.original_state}->{event.corrected_state}"] += 1
        corrected_sprite_ids.add(event.sprite_id)
    return {
        "total_corrections": total,
        "unique_sprites_corrected": len(corrected_sprite_ids),
        "fields_corrected": dict(field_counts.most_common()),
        "state_transitions": dict(state_changes.most_common()),
    }


def select_v3_calibration_sample(
    v3_records: Sequence[RecordDecision],
    n: int,
    *,
    seed: int = 496,
    uncertainty_fraction: float = 0.35,
    random_fraction: float = 0.30,
    conflict_fraction: float = 0.15,
    open_set_fraction: float = 0.20,
    stratify_source: bool = True,
) -> list[str]:
    """Select a diverse, uncertainty-focused calibration sample.

    Returns sprite_ids in priority order. The caller may cap at ``n``.
    """

    import random as _random

    if n <= 0:
        return []

    rng = _random.Random(seed)
    records_by_id = {r.sprite_id: r for r in v3_records if r.sprite_id}
    total = len(records_by_id)
    if total == 0:
        return []

    n = min(n, total)

    priority: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}

    for record in records_by_id.values():
        pid = record.sprite_id
        reasons.setdefault(pid, [])
        score = 0

        # Priority to borderline decisions
        if record.canonical_object.state == "novel":
            score += 3
            reasons[pid].append("open_set_novel")
        elif record.canonical_object.state == "unknown":
            score += 2
            reasons[pid].append("open_set_unknown")
        if record.record_state == "quarantine":
            score += 2
            reasons[pid].append("quarantine")
        if record.reason_codes:
            score += 1
            reasons[pid].append("has_reason_codes")
        if record.canonical_object.contradiction_codes:
            score += 2
            reasons[pid].append("object_contradiction")
        if record.canonical_object.state == "abstained":
            score += 1
            reasons[pid].append("object_abstained")

        # Stratify by presence in accepted/partial
        if record.record_state in ("auto_accept", "partial_accept"):
            score += 1
            reasons[pid].append("accepted_or_partial")

        priority[pid] = score

    sorted_ids = sorted(records_by_id.keys(), key=lambda pid: (-priority[pid], rng.random()))

    # Allocate fractions
    n_uncertainty = max(1, int(n * uncertainty_fraction))
    max(1, int(n * random_fraction))
    n_conflict = max(1, int(n * conflict_fraction))
    n_open_set = max(1, int(n * open_set_fraction))

    selected: list[str] = []
    selected_set: set[str] = set()

    # Uncertainty-first
    for pid in sorted_ids:
        if len(selected) >= n_uncertainty:
            break
        if priority.get(pid, 0) >= 2 and pid not in selected_set:
            selected.append(pid)
            selected_set.add(pid)

    # Conflict
    for pid in sorted_ids:
        if len(selected) >= n_uncertainty + n_conflict:
            break
        r = records_by_id.get(pid)
        if r and r.canonical_object.contradiction_codes and pid not in selected_set:
            selected.append(pid)
            selected_set.add(pid)

    # Open-set
    for pid in sorted_ids:
        if len(selected) >= n_uncertainty + n_conflict + n_open_set:
            break
        r = records_by_id.get(pid)
        if r and r.canonical_object.state in ("novel", "unknown") and pid not in selected_set:
            selected.append(pid)
            selected_set.add(pid)

    # Random remainder
    shuffled = list(records_by_id.keys())
    rng.shuffle(shuffled)
    for pid in shuffled:
        if len(selected) >= n:
            break
        if pid not in selected_set:
            selected.append(pid)
            selected_set.add(pid)

    return selected[:n]


def prefetch_v3_records_for_candidates(
    run_dir: str | Path,
    sprite_ids: Sequence[str],
    *,
    v3_records_path: str | Path | None = None,
) -> dict[str, RecordDecision]:
    """Load v3 RecordDecision objects for a set of sprite IDs."""
    run_path = Path(run_dir)
    v3_path = Path(v3_records_path) if v3_records_path else run_path / V3_RECORDS_FILENAME

    result: dict[str, RecordDecision] = {}
    if not v3_path.is_file():
        return result

    for record in read_jsonl(v3_path):
        sid = str(record.get("sprite_id", ""))
        if sid in set(sprite_ids):
            try:
                result[sid] = record_decision_from_json(record)
            except Exception:
                continue
    return result


def load_all_v3_records(path: str | Path) -> dict[str, RecordDecision]:
    """Load every v3 RecordDecision from a v3_records.jsonl file."""
    p = Path(path)
    result: dict[str, RecordDecision] = {}
    if not p.is_file():
        return result
    for record in read_jsonl(p):
        try:
            rd = record_decision_from_json(record)
            if rd.sprite_id:
                result[rd.sprite_id] = rd
        except Exception:
            continue
    return result


def _list_value(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if value not in (None, "") else []


def v3_candidate_summary_for_gui(
    record: RecordDecision, corrections: Sequence[V3CorrectionEvent] = ()
) -> dict[str, Any]:
    """Resolve GUI values from correction → prefill → legacy → calibrated.

    ``FieldDecision`` is deliberately conservative and may abstain while a
    useful review prefill exists.  This presenter must never turn that valid
    serialized proposal into an empty control.
    """

    by_field = {
        correction.field_name: correction
        for correction in corrections
        if correction.sprite_id == record.sprite_id and correction.field_name != "__review__"
    }

    def resolved_value(fd: FieldDecision, field_name: str) -> tuple[Any, str, V3CorrectionEvent | None]:
        correction = by_field.get(field_name)
        if correction is not None:
            return correction.corrected_value, "reviewed_correction", correction
        prefill = record.prefills.get(field_name)
        if prefill is not None and prefill.value not in (None, "", []):
            kind = (
                "compatible_legacy_prefill"
                if "legacy_v3_uncalibrated_score_unavailable" in prefill.warnings
                else "prefill"
            )
            return prefill.value, kind, None
        return fd.accepted_value, "accepted_calibrated", None

    def _field_summary(fd: FieldDecision, fd_name: str) -> dict[str, Any]:
        prefill = record.prefills.get(fd_name)
        value, value_source, correction = resolved_value(fd, fd_name)
        return {
            "field": fd_name,
            "state": correction.corrected_state if correction is not None else fd.state,
            "value": value,
            "value_source": value_source,
            "accepted_value": fd.accepted_value,
            "prefill_confidence": prefill.confidence if prefill is not None else 0.0,
            "confidence_kind": prefill.confidence_kind if prefill is not None else "prefill_ranking_score",
            "alternatives": [a.to_json() for a in prefill.alternatives] if prefill is not None else [],
            "score_components": dict(prefill.score_components) if prefill is not None else {},
            "evidence_refs": list(prefill.evidence_refs) if prefill is not None else list(fd.evidence_refs),
            "supporting_sources": list(prefill.supporting_dependency_groups) if prefill is not None else [],
            "conflicting_sources": list(prefill.conflicting_dependency_groups) if prefill is not None else [],
            "normalization_actions": list(prefill.normalization_actions) if prefill is not None else [],
            "prefill_warnings": list(prefill.warnings) if prefill is not None else [],
            "hierarchy_node": fd.hierarchy_node,
            "candidates": list(fd.candidates),
            "calibrated_estimate": fd.calibrated_estimate,
            "ci_lower": fd.confidence_interval[0] if fd.confidence_interval else None,
            "ci_upper": fd.confidence_interval[1] if fd.confidence_interval else None,
            "decision_reason": fd.decision_reason,
            "contradiction_codes": list(fd.contradiction_codes),
            "evidence_count": len(fd.evidence_refs),
        }

    fields = {
        "domain": _field_summary(record.domain, "domain"),
        "category": _field_summary(record.category, "category"),
        "canonical_object": _field_summary(record.canonical_object, "canonical_object"),
        "surface_alias": _field_summary(record.surface_alias, "surface_alias"),
        "color": _field_summary(record.color, "color"),
        "material": _field_summary(record.material, "material"),
        "shape": _field_summary(record.shape, "shape"),
        "role": _field_summary(record.role, "role"),
        "description": _field_summary(record.description, "description"),
    }
    changed_dependencies = {name for name in ("canonical_object", "color", "material", "shape") if name in by_field}
    description_artifact = dict(record.description_artifact)
    if changed_dependencies and "description" not in by_field:
        from spritelab.harvest.label_v3.description_enrichment import canonical_description_from_facts

        facts = {name: fields[name]["value"] for name in ("canonical_object", "color", "material", "shape")}
        color_roles = record.prefill_metadata.get("color_roles") or {}
        facts.update(
            {
                name: color_roles[name]
                for name in ("primary_colors", "highlight_colors", "outline_color")
                if color_roles.get(name)
            }
        )
        canonical = canonical_description_from_facts(facts)
        description_artifact.update(
            {"canonical_description": canonical, "enriched_description": canonical, "regenerated_from_review": True}
        )
        fields["description"]["value"] = canonical
        fields["description"]["value_source"] = "regenerated_from_review"

    tag_correction = by_field.get("tags")
    if tag_correction is not None:
        display_tags = _list_value(tag_correction.corrected_value)
    elif changed_dependencies:
        stale = set()
        for name in ("canonical_object", "color", "shape", "role"):
            stale.update(_list_value(record.prefills.get(name).value if record.prefills.get(name) else None))
        display_tags = [tag for tag in record.prefill_tags if tag not in stale]
        for name in ("canonical_object", "color", "shape", "role"):
            for value in _list_value(fields[name]["value"]):
                if value not in display_tags:
                    display_tags.append(value)
    else:
        display_tags = list(record.prefill_tags) or list(record.tags.accepted_tags)

    return {
        "sprite_id": record.sprite_id,
        "record_state": record.record_state,
        "reason_codes": list(record.reason_codes),
        "reason_details": list(record.reason_details),
        "fields": fields,
        "accepted_tags": list(record.tags.accepted_tags),
        "prefill_tags": display_tags,
        "accepted_fields": list(record.accepted_fields),
        "abstained_fields": list(record.abstained_fields),
        "description_artifact": description_artifact,
        "prefill_metadata": dict(record.prefill_metadata),
    }


def apply_v3_corrections_to_record(
    record: RecordDecision,
    corrections: Sequence[V3CorrectionEvent],
) -> RecordDecision:
    """Apply a set of correction events to a RecordDecision.

    Only corrections matching the sprite_id are applied. Returns a new record.
    """
    record_corrections = [c for c in corrections if c.sprite_id == record.sprite_id]
    if not record_corrections:
        return record

    from spritelab.harvest.label_v3.field_decisions import AcceptedTagSet, FieldDecision, TagDecision

    field_map = {
        "domain": record.domain,
        "category": record.category,
        "canonical_object": record.canonical_object,
        "surface_alias": record.surface_alias,
        "color": record.color,
        "material": record.material,
        "shape": record.shape,
        "role": record.role,
        "description": record.description,
    }

    for correction in record_corrections:
        if correction.field_name not in field_map:
            continue
        old_fd = field_map[correction.field_name]
        new_fd = FieldDecision(
            sprite_id=old_fd.sprite_id,
            field_name=old_fd.field_name,
            state=correction.corrected_state,
            accepted_value=correction.corrected_value if correction.corrected_state == "accepted" else None,
            candidates=old_fd.candidates,
            evidence_refs=old_fd.evidence_refs,
            decision_reason="human_correction",
            policy_hash=old_fd.policy_hash,
        )
        field_map[correction.field_name] = new_fd

    tags = record.tags
    tag_corrections = [c for c in record_corrections if c.field_name == "tags"]
    if tag_corrections:
        latest = tag_corrections[-1]
        values = latest.corrected_value if isinstance(latest.corrected_value, (list, tuple)) else ()
        tags = AcceptedTagSet(
            decisions=tuple(
                TagDecision(tag=str(tag), state="accepted", provenance={"review_action": latest.review_action})
                for tag in values
            ),
            provenance={"human_reviewed": True},
        )

    kwargs = {
        "sprite_id": record.sprite_id,
        "domain": field_map.get("domain", record.domain),
        "category": field_map.get("category", record.category),
        "canonical_object": field_map.get("canonical_object", record.canonical_object),
        "surface_alias": field_map.get("surface_alias", record.surface_alias),
        "color": field_map.get("color", record.color),
        "material": field_map.get("material", record.material),
        "shape": field_map.get("shape", record.shape),
        "role": field_map.get("role", record.role),
        "description": field_map.get("description", record.description),
        "tags": tags,
        "description_artifact": dict(record.description_artifact),
        "prefills": dict(record.prefills),
        "prefill_tags": tuple(record.prefill_tags),
        "prefill_metadata": dict(record.prefill_metadata),
        "policy_hash": record.policy_hash,
        "lineage": dict(record.lineage),
        "reason_details": record.reason_details,
    }

    from spritelab.harvest.label_v3.record_decisions import derive_record_state

    decisions_map = {
        "domain": kwargs["domain"],
        "category": kwargs["category"],
        "canonical_object": kwargs["canonical_object"],
        "color": kwargs["color"],
        "material": kwargs["material"],
        "shape": kwargs["shape"],
    }
    kwargs["record_state"] = derive_record_state(decisions_map)

    return RecordDecision(**kwargs)
