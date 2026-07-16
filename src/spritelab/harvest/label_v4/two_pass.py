"""Strict two-pass quality and semantic calibration contracts for Labeling v4."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from spritelab.harvest.label_v4.audit_prefill import (
    AUDIT_SELECTION_SCHEMA,
    detect_audit_schema,
    require_prefilled_records,
)
from spritelab.harvest.label_v4.review import (
    QUALITY_ACTIONS,
    REVIEW_EVENT_SCHEMA_VERSION,
    ReviewEvent,
    immutable_proposal_digest,
)

INFERENCE_QUEUE_SCHEMA = "label_v4_inference_queue_v1"
QUALITY_ELIGIBLE = frozenset({"quality_suitable", "quality_uncertain_usable"})
QUALITY_EXCLUDED = frozenset({"quality_unsuitable", "quality_uncertain_not_usable"})
QUALITY_TERMINAL = QUALITY_ELIGIBLE | QUALITY_EXCLUDED
SEMANTIC_TERMINAL_OUTCOMES = frozenset(
    {
        "correct",
        "correct_but_normalization_changed",
        "incorrect",
        "unsupported",
        "human_abstained",
        "not_applicable",
        "model_abstention_accepted",
    }
)
SEMANTIC_VALUE_OUTCOMES = frozenset({"correct", "correct_but_normalization_changed", "incorrect"})
SEMANTIC_NON_VALUE_OUTCOMES = frozenset(
    {
        "model_abstention_accepted",
        "human_abstained",
        "unsupported",
        "not_applicable",
        "not_scorable",
        "missing_prediction",
        "provider_failed",
    }
)
REQUIRED_CRITICAL_FIELDS = ("canonical_object", "category", "domain", "role")
UNSCORABLE_FIELD_STATES = frozenset(
    {"missing_prediction", "provider_failed", "not_scorable", "not_scorable_due_to_image"}
)
SUCCESSFUL_STAGE_STATUSES = frozenset(
    {"success", "success_after_retry", "cache_hit_success", "success_after_json_repair", "deterministic_fallback"}
)
REPAIRED_STAGE_STATUSES = frozenset({"success_after_json_repair", "deterministic_fallback"})
MATERIAL_NOT_REQUIRED_REASONS = frozenset(
    {"explicit_axis_not_applicable", "taxonomy_rule_not_material_bearing", "deterministic_rule_not_material_bearing"}
)


@dataclass(frozen=True)
class QualityResolution:
    sprite_id: str
    effective_state: str
    event: ReviewEvent | None
    valid_event_count: int
    ignored_event_count: int

    @property
    def complete(self) -> bool:
        return self.effective_state in QUALITY_TERMINAL and self.event is not None


@dataclass(frozen=True)
class SemanticFieldValidation:
    field_name: str
    value_state: str
    valid: bool
    reason: str


@dataclass(frozen=True)
class MaterialRequirement:
    state: str
    reason: str
    provenance: str


def _compatible_event(event: ReviewEvent) -> bool:
    return event.schema_version == REVIEW_EVENT_SCHEMA_VERSION


def _common_event_authority(
    record: Mapping[str, Any] | None,
    event: ReviewEvent,
    *,
    semantic: bool,
) -> tuple[str | None, str]:
    """Return an invalid audit category/reason, or ``(None, "authoritative")``.

    This is the shared identity boundary used by replay resolvers and the
    existing-event audit.  Bindings are deliberately exact and non-optional.
    """

    if not _compatible_event(event):
        return "schema_incompatible", "unsupported_review_event_schema"
    if record is None:
        return "identity_mismatch", "sprite_not_in_review_manifest"
    sprite_id = str(record.get("sprite_id") or "")
    if not sprite_id or event.sprite_id != sprite_id:
        return "identity_mismatch", "sprite_id_mismatch"
    audit_record_id = str(record.get("audit_id") or "")
    metadata_audit_id = str(event.metadata.get("audit_record_id") or event.metadata.get("audit_id") or "")
    if not audit_record_id:
        return "identity_mismatch", "review_record_missing_audit_identity"
    if not event.session_id or event.session_id != audit_record_id:
        return "identity_mismatch", "session_identity_mismatch"
    if not metadata_audit_id or metadata_audit_id != audit_record_id:
        return "identity_mismatch", "audit_record_identity_mismatch"
    if semantic:
        fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
        if not fields and isinstance(record.get("field_proposals"), Mapping):
            fields = record["field_proposals"]
        if not event.field_name or event.field_name not in fields:
            return "identity_mismatch", "semantic_field_identity_mismatch"
    elif event.field_name:
        return "identity_mismatch", "quality_event_has_field_identity"
    expected_hash = immutable_proposal_digest(record)
    if not event.proposal_hash or event.proposal_hash != expected_hash:
        return "proposal_hash_mismatch", "missing_or_changed_proposal_hash"
    return None, "authoritative"


def _quality_event_authority(record: Mapping[str, Any] | None, event: ReviewEvent) -> tuple[str, str]:
    category, reason = _common_event_authority(record, event, semantic=False)
    if category:
        return category, reason
    if event.action not in QUALITY_ACTIONS or event.human_outcome != event.action:
        return "ignored_non_authoritative", "invalid_quality_action_or_outcome"
    return "valid_quality", "authoritative_quality_event"


def resolve_quality_decisions(
    records: Sequence[Mapping[str, Any]], events: Sequence[ReviewEvent]
) -> dict[str, QualityResolution]:
    """Resolve the latest schema- and proposal-compatible terminal quality event."""

    by_id = {str(record.get("sprite_id", "")): record for record in records}
    valid: dict[str, list[ReviewEvent]] = {sprite_id: [] for sprite_id in by_id}
    ignored = Counter()
    for event in events:
        record = by_id.get(event.sprite_id)
        category, _reason = _quality_event_authority(record, event)
        if category != "valid_quality":
            ignored[event.sprite_id] += 1
            continue
        valid[event.sprite_id].append(event)
    return {
        sprite_id: QualityResolution(
            sprite_id=sprite_id,
            effective_state=(items[-1].human_outcome if items else "quality_unreviewed"),
            event=(items[-1] if items else None),
            valid_event_count=len(items),
            ignored_event_count=int(ignored[sprite_id]),
        )
        for sprite_id, items in valid.items()
    }


def quality_resume_state(records: Sequence[Mapping[str, Any]], events: Sequence[ReviewEvent]) -> dict[str, Any]:
    resolved = resolve_quality_decisions(records, events)
    remaining_indices = [
        index for index, record in enumerate(records) if not resolved[str(record.get("sprite_id", ""))].complete
    ]
    return {
        "next_index": remaining_indices[0] if remaining_indices else None,
        "review_complete": not remaining_indices,
        "remaining": len(remaining_indices),
        "completed": len(records) - len(remaining_indices),
        "total": len(records),
    }


def quality_resume_index(records: Sequence[Mapping[str, Any]], events: Sequence[ReviewEvent]) -> int | None:
    return quality_resume_state(records, events)["next_index"]


def _actual_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return bool(value)
    return True


def _validated_model_stages(record: Mapping[str, Any]) -> tuple[bool, str]:
    provenance = record.get("model_provenance")
    if not isinstance(provenance, Mapping):
        return False, "missing_model_provenance"
    audit_record_id = str(record.get("audit_id") or "")
    sprite_id = str(record.get("sprite_id") or "")
    exported_rgba_hash = str(record.get("exported_rgba_hash") or "")
    if not audit_record_id or not sprite_id:
        return False, "missing_canonical_record_identity"
    if len(exported_rgba_hash) != 64 or any(
        character not in "0123456789abcdef" for character in exported_rgba_hash.lower()
    ):
        return False, "missing_exported_rgba_identity"
    input_hash = str(provenance.get("image_hash") or "")
    if not input_hash:
        return False, "missing_model_input_binding"
    if str(provenance.get("sprite_id") or "") != sprite_id:
        return False, "model_provenance_sprite_identity_mismatch"
    image_path = Path(str(record.get("image_path") or ""))
    if not image_path.is_file():
        return False, "missing_bound_review_image"
    try:
        with Image.open(image_path) as source:
            rgba = source.convert("RGBA")
            exported_payload = (
                b"spritelab-exported-rgba-v1\0" + struct.pack(">II", rgba.width, rgba.height) + rgba.tobytes()
            )
    except (OSError, ValueError):
        return False, "unreadable_bound_review_image"
    if hashlib.sha256(exported_payload).hexdigest() != exported_rgba_hash:
        return False, "exported_rgba_hash_does_not_match_image"
    from spritelab.harvest.label_v4.pixel_evidence import exact_rgba_content_hash

    if exact_rgba_content_hash(image_path) != input_hash:
        return False, "model_input_hash_does_not_match_image"
    binding = provenance.get("review_record_binding")
    if not isinstance(binding, Mapping):
        return False, "missing_review_record_binding"
    expected_binding = {
        "audit_record_id": audit_record_id,
        "sprite_id": sprite_id,
        "exported_rgba_hash": exported_rgba_hash,
        "proposal_hash": immutable_proposal_digest(record),
        "proposal_input_hash": input_hash,
    }
    for name, expected in expected_binding.items():
        if not expected or str(binding.get(name) or "") != expected:
            return False, f"review_record_binding_mismatch:{name}"
    for optional_identity in ("queue_id", "prefill_record_hash"):
        record_value = str(record.get(optional_identity) or "")
        if record_value and str(binding.get(optional_identity) or "") != record_value:
            return False, f"review_record_binding_mismatch:{optional_identity}"
    stage_proofs = binding.get("stage_artifacts")
    if not isinstance(stage_proofs, Mapping):
        return False, "missing_stage_artifact_identities"
    outcomes = {
        str(item.get("stage")): item for item in provenance.get("stage_outcomes") or () if isinstance(item, Mapping)
    }
    ledger = {
        str(item.get("stage")): item for item in provenance.get("stage_ledger") or () if isinstance(item, Mapping)
    }
    for stage in ("B_blind_vlm_proposal", "C_text_reconciliation"):
        outcome = outcomes.get(stage)
        artifact = ledger.get(stage)
        if not outcome or str(outcome.get("stage_status")) not in SUCCESSFUL_STAGE_STATUSES:
            return False, f"{stage}:not_successfully_terminal"
        if not artifact:
            return False, f"{stage}:missing_stage_artifact"
        proof = stage_proofs.get(stage)
        if not isinstance(proof, Mapping):
            return False, f"{stage}:missing_stage_identity"
        if str(proof.get("artifact_sha256") or "") != _stable_hash(artifact):
            return False, f"{stage}:stage_artifact_identity_mismatch"
        if str(proof.get("stage_status") or "") != str(outcome.get("stage_status") or ""):
            return False, f"{stage}:stage_status_binding_mismatch"
        if str(proof.get("proposal_input_hash") or "") != input_hash:
            return False, f"{stage}:proposal_input_identity_mismatch"
        if stage == "B_blind_vlm_proposal":
            if str(artifact.get("image_hash") or "") != input_hash:
                return False, f"{stage}:artifact_input_binding_mismatch"
        else:
            embedded = provenance.get("reconciliation_provider_artifact")
            request_hash = str(artifact.get("request_hash") or "")
            image_bound = str(artifact.get("image_hash") or "") == input_hash
            request_bound = (
                isinstance(embedded, Mapping)
                and bool(request_hash)
                and request_hash == str(embedded.get("request_hash") or "")
            )
            if not image_bound and not request_bound:
                return False, f"{stage}:artifact_input_binding_mismatch"
        status = str(outcome.get("stage_status"))
        failure = bool(artifact.get("failure_diagnostics"))
        if failure and status not in REPAIRED_STAGE_STATUSES:
            return False, f"{stage}:unresolved_provider_failure"
        if not failure and status != "deterministic_fallback" and artifact.get("provider_output_valid") is not True:
            return False, f"{stage}:provider_output_not_valid"
        if status == "deterministic_fallback" and outcome.get("fallback_used") is not True:
            return False, f"{stage}:unvalidated_fallback"
    return True, "completed_model_stages_bound_to_record"


def validate_semantic_field(record: Mapping[str, Any], field_name: str) -> SemanticFieldValidation:
    """Authoritative semantic-state validator for review, completion, and metrics."""

    fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
    if not fields and isinstance(record.get("field_proposals"), Mapping):
        fields = record["field_proposals"]
    field = fields.get(field_name)
    if not isinstance(field, Mapping):
        return SemanticFieldValidation(field_name, "missing_prediction", False, "missing_field")
    state = str(field.get("value_state", "known" if _actual_value(field.get("value")) else "missing_prediction"))
    value = field.get("value")
    if state == "known":
        return SemanticFieldValidation(
            field_name,
            state,
            _actual_value(value),
            "known_value_present" if _actual_value(value) else "known_without_value",
        )
    if state == "model_abstained":
        if value is not None:
            return SemanticFieldValidation(field_name, state, False, "model_abstention_has_value")
        stages_valid, reason = _validated_model_stages(record)
        if not stages_valid:
            return SemanticFieldValidation(field_name, state, False, reason)
        if str(field.get("reason")) != "model_stage_completed_without_promoted_value":
            return SemanticFieldValidation(field_name, state, False, "no_explicit_unpromoted_value_result")
        return SemanticFieldValidation(field_name, state, True, reason)
    if state in {"not_applicable", "unsupported"}:
        return SemanticFieldValidation(field_name, state, False, f"unscorable_{state}")
    return SemanticFieldValidation(field_name, state, False, state or "missing_prediction")


def _semantic_event_authority(record: Mapping[str, Any] | None, event: ReviewEvent) -> tuple[str, str]:
    category, reason = _common_event_authority(record, event, semantic=True)
    if category:
        return category, reason
    assert record is not None  # narrowed by the common authority check
    if event.action in QUALITY_ACTIONS or event.human_outcome not in SEMANTIC_TERMINAL_OUTCOMES:
        return "ignored_non_authoritative", "nonterminal_or_quality_semantic_event"
    allowed_actions = {
        "correct": {"accept_proposed_value"},
        "correct_but_normalization_changed": {
            "accept_proposed_value",
            "select_alternative",
            "edit",
        },
        "incorrect": {"select_alternative", "edit"},
        "model_abstention_accepted": {"accept_model_abstention"},
        "human_abstained": {"mark_human_abstention"},
        "unsupported": {"mark_unsupported", "mark_wrong_taxonomy"},
        "not_applicable": {"mark_not_applicable"},
    }
    if event.action not in allowed_actions.get(event.human_outcome, set()):
        return "ignored_non_authoritative", "action_outcome_mismatch"
    validation = validate_semantic_field(record, event.field_name)
    if event.human_outcome == "correct" and (not validation.valid or validation.value_state != "known"):
        return "ignored_non_authoritative", "accepted_semantic_value_is_not_known"
    if event.human_outcome == "model_abstention_accepted" and (
        not validation.valid or validation.value_state != "model_abstained"
    ):
        return "ignored_non_authoritative", "accepted_model_abstention_is_not_proven"
    if event.human_outcome in {"incorrect", "correct_but_normalization_changed"} and not _actual_value(
        event.reviewed_value
    ):
        return "ignored_non_authoritative", "corrected_semantic_value_missing"
    return "valid_semantic", "authoritative_semantic_event"


def material_requirement(record: Mapping[str, Any]) -> MaterialRequirement:
    """Return required/not_required/unresolved from explicit or deterministic evidence.

    Absence of a material proposal is deliberately unresolved. Non-applicability
    requires a structured applicability result or a named deterministic/taxonomy rule.
    """

    fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
    field = fields.get("explicit_material")
    if not isinstance(field, Mapping):
        return MaterialRequirement("unresolved", "material_field_missing", "none")
    applicability = field.get("applicability")
    if isinstance(applicability, Mapping):
        state = str(applicability.get("state") or "")
        reason = str(applicability.get("reason") or "")
        provenance = str(applicability.get("provenance") or "")
        if state == "not_required" and reason and provenance:
            return MaterialRequirement(state, reason, provenance)
        if state == "required" and reason and provenance:
            return MaterialRequirement(state, reason, provenance)
    if _actual_value(field.get("value")) or str(field.get("value_state")) == "known":
        return MaterialRequirement("required", "applicable_material_value", "semantic_field")
    if (
        str(field.get("value_state")) == "model_abstained"
        and validate_semantic_field(record, "explicit_material").valid
    ):
        return MaterialRequirement("required", "applicable_stage_abstention", "validated_model_stage")
    reason = str(field.get("reason") or "")
    if str(field.get("value_state")) == "not_applicable" and reason in MATERIAL_NOT_REQUIRED_REASONS:
        return MaterialRequirement("not_required", reason, "deterministic_or_taxonomy_rule")
    return MaterialRequirement("unresolved", "material_applicability_not_established", "none")


def semantic_readiness(record: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if str(record.get("prediction_state", "")) == "missing_required_model_stage" or record.get("missing_stages"):
        reasons.append("missing_required_model_stage")
    for name in REQUIRED_CRITICAL_FIELDS:
        validation = validate_semantic_field(record, name)
        if not validation.valid:
            reasons.append(f"{name}:{validation.reason}")
    material = material_requirement(record)
    if material.state == "required":
        validation = validate_semantic_field(record, "explicit_material")
        if not validation.valid:
            reasons.append(f"explicit_material:{validation.reason}")
    elif material.state == "unresolved":
        reasons.append(f"explicit_material:{material.reason}")
    return not reasons, tuple(dict.fromkeys(reasons))


def require_semantic_ready_records(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        ready, reasons = semantic_readiness(record)
        if not ready:
            raise ValueError(
                f"Record {record.get('sprite_id', '')} is not semantic-review ready.\n"
                f"Prediction state: {record.get('prediction_state', 'unknown')}.\n"
                "Reasons: " + ", ".join(reasons) + ".\nRun semantic prediction preparation first."
            )


def has_real_semantic_proposal(record: Mapping[str, Any]) -> bool:
    if str(record.get("prediction_origin", "")) not in {"compatible_cached_rich_vlm", "semantic_minimal_provider"}:
        return False
    return any(validate_semantic_field(record, name).valid for name in REQUIRED_CRITICAL_FIELDS)


def _valid_semantic_events(record: Mapping[str, Any], events: Sequence[ReviewEvent]) -> dict[str, ReviewEvent]:
    latest: dict[str, ReviewEvent] = {}
    for event in events:
        category, _reason = _semantic_event_authority(record, event)
        if category == "valid_semantic":
            latest[event.field_name] = event
    return latest


def _required_semantic_fields(record: Mapping[str, Any]) -> tuple[list[str], MaterialRequirement]:
    required = list(REQUIRED_CRITICAL_FIELDS)
    material = material_requirement(record)
    if material.state == "required":
        required.append("explicit_material")
    return required, material


def semantic_completion(
    record: Mapping[str, Any], events: Sequence[ReviewEvent], quality: QualityResolution
) -> dict[str, Any]:
    ready, readiness_reasons = semantic_readiness(record)
    latest = _valid_semantic_events(record, events)
    required, material = _required_semantic_fields(record)
    unresolved = [name for name in required if name not in latest]
    abstentions_unjudged = [
        name
        for name in required
        if validate_semantic_field(record, name).value_state == "model_abstained" and name not in latest
    ]
    reasons = [*readiness_reasons]
    if material.state == "unresolved":
        reasons.append(f"material_applicability:{material.reason}")
    if quality.effective_state not in QUALITY_ELIGIBLE:
        reasons.append(f"ineligible_quality:{quality.effective_state}")
    reasons.extend(f"unreviewed:{name}" for name in unresolved)
    reasons.extend(f"unjudged_model_abstention:{name}" for name in abstentions_unjudged)
    complete = bool(
        ready
        and material.state != "unresolved"
        and quality.effective_state in QUALITY_ELIGIBLE
        and not unresolved
        and not abstentions_unjudged
    )
    return {
        "complete": complete,
        "required_fields": required,
        "terminal_fields": sorted(latest),
        "unresolved_fields": unresolved,
        "material_requirement": material.__dict__,
        "reasons": list(dict.fromkeys(reasons)),
    }


def semantic_field_progress(
    record: Mapping[str, Any], events: Sequence[ReviewEvent], quality: QualityResolution
) -> dict[str, Any]:
    """Return controlled per-field and record progress for reviewer-facing UIs."""

    completion = semantic_completion(record, events, quality)
    latest = _valid_semantic_events(record, events)
    required = set(completion["required_fields"])
    fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
    outcome_status = {
        "correct": "accepted",
        "correct_but_normalization_changed": "corrected",
        "incorrect": "corrected",
        "model_abstention_accepted": "model_abstention_accepted",
        "human_abstained": "human_abstained",
        "unsupported": "unsupported",
        "not_applicable": "not_applicable",
    }
    rows: list[dict[str, Any]] = []
    for name, field in fields.items():
        source = field if isinstance(field, Mapping) else {"value": field}
        validation = validate_semantic_field(record, str(name))
        event = latest.get(str(name))
        status = outcome_status.get(event.human_outcome, "invalid") if event else "unreviewed"
        human_state = status
        if event and status == "corrected":
            human_state = f"corrected: {json.dumps(event.reviewed_value, ensure_ascii=False)}"
        elif event and status == "accepted":
            human_state = "accepted"
        model_state = str(source.get("value_state") or validation.value_state)
        rows.append(
            {
                "field": str(name),
                "required": str(name) in required,
                "model_state": model_state,
                "human_state": human_state,
                "status": status,
                "event_id": event.event_id if event else None,
            }
        )
    unresolved = list(completion["unresolved_fields"])
    return {
        "fields": rows,
        "required_total": len(completion["required_fields"]),
        "required_reviewed": len(completion["required_fields"]) - len(unresolved),
        "remaining_required_fields": unresolved,
        "record_complete": completion["complete"],
        "next_unresolved_field": unresolved[0] if unresolved else None,
        "completion": completion,
    }


def validate_accept_all(record: Mapping[str, Any]) -> tuple[str, ...]:
    ready, reasons = semantic_readiness(record)
    if not ready:
        raise ValueError("accept all refused: " + ", ".join(reasons))
    required, material = _required_semantic_fields(record)
    if material.state == "unresolved":
        raise ValueError("accept all refused: unresolved material applicability")
    unresolved = [name for name in required if validate_semantic_field(record, name).value_state != "known"]
    if unresolved:
        raise ValueError("accept all refused; unresolved critical fields: " + ", ".join(unresolved))
    return tuple(required)


def calibration_denominator_report(
    records: Sequence[Mapping[str, Any]], events: Sequence[ReviewEvent]
) -> dict[str, Any]:
    quality = resolve_quality_decisions(records, events)
    scorable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    abstention_judgments: list[dict[str, Any]] = []
    abstention_corrections: list[dict[str, Any]] = []
    non_value_outcomes: Counter[str] = Counter()
    semantic_complete = 0
    for record in records:
        sprite_id = str(record.get("sprite_id", ""))
        completion = semantic_completion(record, events, quality[sprite_id])
        semantic_complete += int(completion["complete"])
        latest = _valid_semantic_events(record, events)
        for name, event in latest.items():
            validation = validate_semantic_field(record, name)
            if event.human_outcome == "model_abstention_accepted":
                abstention_judgments.append({"sprite_id": sprite_id, "field": name, "outcome": event.human_outcome})
                continue
            if (
                validation.value_state == "model_abstained"
                and event.human_outcome in SEMANTIC_VALUE_OUTCOMES
                and _actual_value(event.reviewed_value)
            ):
                correction = {
                    "sprite_id": sprite_id,
                    "field": name,
                    "outcome": event.human_outcome,
                    "reason": "model_abstention_rejected_with_human_correction",
                }
                abstention_corrections.append(correction)
                excluded.append(correction)
                continue
            reason = ""
            if event.human_outcome not in SEMANTIC_VALUE_OUTCOMES:
                non_value_outcomes[event.human_outcome] += 1
                reason = f"non_value_outcome:{event.human_outcome}"
            elif quality[sprite_id].effective_state not in QUALITY_ELIGIBLE:
                reason = quality[sprite_id].effective_state
            elif not completion["complete"]:
                reason = "incomplete_semantic_record"
            elif not validation.valid or validation.value_state != "known":
                reason = validation.reason
            (scorable if not reason else excluded).append(
                {"sprite_id": sprite_id, "field": name, "outcome": event.human_outcome, "reason": reason or None}
            )
    return {
        "quality_reviewed_records": sum(value.complete for value in quality.values()),
        "semantic_ready_records": sum(semantic_readiness(record)[0] for record in records),
        "semantic_complete_records": semantic_complete,
        "missing_prediction_records": sum(
            record.get("prediction_state") == "missing_required_model_stage" for record in records
        ),
        "provider_failure_records": sum(
            any(
                (field or {}).get("value_state") == "provider_failed" for field in (record.get("fields") or {}).values()
            )
            for record in records
        ),
        "scorable_field_judgments": len(scorable),
        "excluded_field_judgments": len(excluded),
        "model_abstention_appropriateness_judgments": len(abstention_judgments),
        "model_abstention_rejected_with_correction_judgments": len(abstention_corrections),
        "human_abstention_judgments": non_value_outcomes["human_abstained"],
        "unsupported_judgments": non_value_outcomes["unsupported"],
        "not_applicable_judgments": non_value_outcomes["not_applicable"],
        "non_value_outcome_counts": {
            outcome: non_value_outcomes[outcome] for outcome in sorted(SEMANTIC_NON_VALUE_OUTCOMES)
        },
        "scorable": scorable,
        "excluded": excluded,
        "abstention_judgments": abstention_judgments,
        "abstention_corrections": abstention_corrections,
    }


def _project_inference_queue(
    selection: Sequence[Mapping[str, Any]],
    prefilled_by_id: Mapping[str, Mapping[str, Any]],
    resolved: Mapping[str, QualityResolution],
    policy: frozenset[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_queue: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for order, selected in enumerate(selection):
        sprite_id = str(selected.get("sprite_id", ""))
        decision = resolved[sprite_id]
        if decision.effective_state in policy:
            base_queue.append(
                {
                    **dict(selected),
                    "schema_version": INFERENCE_QUEUE_SCHEMA,
                    "audit_order": order,
                    "quality_state": decision.effective_state,
                    "quality_risk_penalty": (0.15 if decision.effective_state == "quality_uncertain_usable" else 0.0),
                    "quality_event": decision.event.to_dict() if decision.event else None,
                    "prefill_record_hash": _stable_hash(prefilled_by_id[sprite_id]),
                }
            )
        else:
            excluded.append(
                {
                    "schema_version": "label_v4_inference_queue_exclusion_v1",
                    "sprite_id": sprite_id,
                    "audit_id": selected.get("audit_id"),
                    "audit_order": order,
                    "quality_state": decision.effective_state,
                    "reason": "quality_unreviewed" if not decision.complete else "excluded_by_quality_policy",
                }
            )
    return base_queue, excluded


def freeze_inference_queue(
    audit_selection: str | Path,
    prefilled_records: str | Path,
    human_truth: str | Path,
    output_root: str | Path,
    *,
    inclusion_policy: Sequence[str] = tuple(sorted(QUALITY_ELIGIBLE)),
    allow_partial: bool = False,
) -> dict[str, Any]:
    selection_path, prefilled_path, truth_path = (
        Path(value).resolve() for value in (audit_selection, prefilled_records, human_truth)
    )
    source_paths = {
        "audit_selection": selection_path,
        "prefilled_records": prefilled_path,
        "human_truth": truth_path,
    }
    for name, source_path in source_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"inference queue source input is missing: {name}={source_path}")
    selection = _read_jsonl(selection_path)
    prefilled = _read_jsonl(prefilled_path)
    require_prefilled_records(prefilled)
    if any(detect_audit_schema(row) != AUDIT_SELECTION_SCHEMA for row in selection):
        raise ValueError("inference queue input must be label_v4_audit_selection_v1")
    if len({str(row.get("sprite_id", "")) for row in selection}) != len(selection):
        raise ValueError("duplicate sprite IDs in audit selection")
    events = _load_events(truth_path)
    by_id = {str(row.get("sprite_id", "")): row for row in prefilled}
    if set(by_id) != {str(row.get("sprite_id", "")) for row in selection}:
        raise ValueError("prefilled records do not exactly match audit selection")
    ordered_records = [by_id[str(row.get("sprite_id", ""))] for row in selection]
    resolved = resolve_quality_decisions(ordered_records, events)
    unreviewed = [sprite_id for sprite_id, value in resolved.items() if not value.complete]
    if unreviewed and not allow_partial:
        raise ValueError(
            f"quality review incomplete for {len(unreviewed)} records; use --allow-partial to freeze reviewed subset"
        )
    policy = frozenset(inclusion_policy)
    if not policy or not policy <= QUALITY_ELIGIBLE:
        raise ValueError("inclusion policy may contain only quality_suitable and quality_uncertain_usable")
    base_queue, excluded = _project_inference_queue(selection, by_id, resolved, policy)
    source_bindings = {
        name: {"path": str(source_path), "sha256": _file_hash(source_path)}
        for name, source_path in source_paths.items()
    }
    source_hashes = {name: value["sha256"] for name, value in source_bindings.items()}
    queue_id = _stable_hash(
        {
            "identity_version": "label_v4_source_bound_queue_id_v2",
            "source_sha256": source_hashes,
            "records": base_queue,
        }
    )
    queue = [{**row, "queue_id": queue_id} for row in base_queue]
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"immutable inference queue output already exists: {output}")
    output.mkdir(parents=True)
    queue_path = output / "inference_queue.jsonl"
    excluded_path = output / "excluded_records.jsonl"
    _write_jsonl(queue_path, queue)
    _write_jsonl(excluded_path, excluded)
    complete = not unreviewed
    finality = {
        "allow_partial": bool(allow_partial),
        "quality_review_complete": complete,
        "queue_status": "final" if complete else "partial_nonfinal",
        "eligible_for_semantic_preparation": complete,
        "total_input_records": len(selection),
        "reviewed_records": len(selection) - len(unreviewed),
        "unreviewed_records": len(unreviewed),
        "included_records": len(queue),
        "excluded_records": len(excluded),
        "unreviewed_ids_sha256": _stable_hash(sorted(unreviewed)),
    }
    report = {
        "schema_version": "label_v4_inference_queue_inclusion_report_v1",
        "queue_id": queue_id,
        "audit_records": len(selection),
        "included": len(queue),
        "excluded": len(excluded),
        "unreviewed": len(unreviewed),
        "inclusion_policy": sorted(policy),
        "quality_state_counts": dict(sorted(Counter(value.effective_state for value in resolved.values()).items())),
        **finality,
    }
    _write_json(output / "inclusion_report.json", report)
    (output / "inclusion_report.md").write_text(
        "# Labeling-v4 inference queue inclusion\n\n"
        f"Queue `{queue_id}` includes {len(queue)} of {len(selection)} records. "
        f"Status: `{finality['queue_status']}`. Unreviewed: {len(unreviewed)}.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "label_v4_inference_queue_manifest_v1",
        "queue_id": queue_id,
        "queue_identity_version": "label_v4_source_bound_queue_id_v2",
        "records": len(queue),
        "inclusion_policy": sorted(policy),
        "audit_selection_sha256": _file_hash(selection_path),
        "prefilled_records_sha256": _file_hash(prefilled_path),
        "human_truth_sha256": _file_hash(truth_path),
        "source_bindings": source_bindings,
        "ordered_sprite_ids_hash": _stable_hash([row["sprite_id"] for row in queue]),
        "geometry_groups_preserved": True,
        "variant_groups_preserved": True,
        **finality,
    }
    _write_json(output / "inference_queue_manifest.json", manifest)
    files = {
        path.name: _file_hash(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "freeze_manifest.json"
    }
    _write_json(
        output / "freeze_manifest.json",
        {
            "schema_version": "label_v4_inference_queue_freeze_v1",
            "queue_id": queue_id,
            "files": files,
            "frozen": True,
            **finality,
        },
    )
    return {**report, "output": str(queue_path)}


def verify_frozen_inference_queue(
    queue_path: str | Path,
    *,
    audit_selection: str | Path | None = None,
    prefilled_records: str | Path | None = None,
    human_truth: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(queue_path).resolve()
    manifest_path = path.parent / "inference_queue_manifest.json"
    freeze_path = path.parent / "freeze_manifest.json"
    if path.name != "inference_queue.jsonl" or not manifest_path.is_file() or not freeze_path.is_file():
        raise ValueError(
            "frozen inference queue requires sibling inference_queue_manifest.json and freeze_manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed frozen inference queue manifest: {exc}") from None
    for name, value in (
        ("queue_status", "final"),
        ("quality_review_complete", True),
        ("eligible_for_semantic_preparation", True),
    ):
        if manifest.get(name) != value or freeze.get(name) != value:
            raise ValueError(f"inference queue is nonfinal or missing finality field: {name}")
    if freeze.get("frozen") is not True:
        raise ValueError("inference queue is not frozen")
    queue_id = str(manifest.get("queue_id") or "")
    if not queue_id or str(freeze.get("queue_id") or "") != queue_id:
        raise ValueError("inference queue ID binding mismatch")
    expected_hash = str((freeze.get("files") or {}).get(path.name) or "")
    actual_hash = _file_hash(path)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("inference queue SHA-256 mismatch")
    frozen_files = freeze.get("files") if isinstance(freeze.get("files"), Mapping) else {}
    for required in ("inference_queue_manifest.json", "inclusion_report.json", "excluded_records.jsonl"):
        expected = str(frozen_files.get(required) or "")
        sibling = path.parent / required
        if not expected or not sibling.is_file() or _file_hash(sibling) != expected:
            raise ValueError(f"frozen queue provenance hash mismatch: {required}")
    for binding in ("audit_selection_sha256", "prefilled_records_sha256", "human_truth_sha256"):
        value = str(manifest.get(binding) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError(f"inference queue source binding missing or malformed: {binding}")
    supplied_sources = {
        "audit_selection": audit_selection,
        "prefilled_records": prefilled_records,
        "human_truth": human_truth,
    }
    missing_sources = [name for name, value in supplied_sources.items() if value is None]
    if missing_sources:
        raise ValueError(
            "frozen inference queue verification requires actual source inputs: " + ", ".join(missing_sources)
        )
    bound_sources = manifest.get("source_bindings")
    if not isinstance(bound_sources, Mapping):
        raise ValueError("inference queue source path bindings are missing")
    scalar_hash_fields = {
        "audit_selection": "audit_selection_sha256",
        "prefilled_records": "prefilled_records_sha256",
        "human_truth": "human_truth_sha256",
    }
    resolved_sources: dict[str, Path] = {}
    for name, supplied in supplied_sources.items():
        actual_path = Path(supplied).resolve()  # type: ignore[arg-type]
        resolved_sources[name] = actual_path
        if not actual_path.is_file():
            raise FileNotFoundError(f"inference queue source input is missing: {name}={actual_path}")
        expected = bound_sources.get(name)
        if not isinstance(expected, Mapping):
            raise ValueError(f"inference queue source path binding is missing: {name}")
        if str(expected.get("path") or "") != str(actual_path):
            raise ValueError(f"inference queue source path mismatch: {name}")
        actual_source_hash = _file_hash(actual_path)
        expected_hash = str(expected.get("sha256") or "")
        if (
            not expected_hash
            or expected_hash != actual_source_hash
            or expected_hash != str(manifest.get(scalar_hash_fields[name]) or "")
        ):
            raise ValueError(f"inference queue source SHA-256 mismatch: {name}")
    if manifest.get("queue_identity_version") != "label_v4_source_bound_queue_id_v2":
        raise ValueError("inference queue uses an unsupported or missing source-bound queue identity")
    policy_value = manifest.get("inclusion_policy")
    if not isinstance(policy_value, list):
        raise ValueError("inference queue inclusion policy binding is missing")
    policy = frozenset(str(value) for value in policy_value)
    if not policy or not policy <= QUALITY_ELIGIBLE:
        raise ValueError("inference queue inclusion policy binding is invalid")
    source_selection = _read_jsonl(resolved_sources["audit_selection"])
    source_prefilled = _read_jsonl(resolved_sources["prefilled_records"])
    require_prefilled_records(source_prefilled)
    if any(detect_audit_schema(row) != AUDIT_SELECTION_SCHEMA for row in source_selection):
        raise ValueError("bound audit selection uses an incompatible schema")
    selected_ids = [str(row.get("sprite_id") or "") for row in source_selection]
    if not all(selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("bound audit selection has missing or duplicate sprite IDs")
    source_by_id = {str(row.get("sprite_id") or ""): row for row in source_prefilled}
    if set(source_by_id) != set(selected_ids):
        raise ValueError("bound prefilled records do not exactly match the audit selection")
    source_events = _load_events(resolved_sources["human_truth"])
    ordered_source_records = [source_by_id[sprite_id] for sprite_id in selected_ids]
    source_resolved = resolve_quality_decisions(ordered_source_records, source_events)
    if any(not decision.complete for decision in source_resolved.values()):
        raise ValueError("bound human truth does not complete the final inference queue")
    projected_base, projected_excluded = _project_inference_queue(
        source_selection, source_by_id, source_resolved, policy
    )
    projected_queue_id = _stable_hash(
        {
            "identity_version": "label_v4_source_bound_queue_id_v2",
            "source_sha256": {name: _file_hash(source_path) for name, source_path in resolved_sources.items()},
            "records": projected_base,
        }
    )
    if projected_queue_id != queue_id:
        raise ValueError("inference queue ID does not match the bound source projection")
    rows = _read_jsonl(path)
    projected_rows = [{**row, "queue_id": projected_queue_id} for row in projected_base]
    if rows != projected_rows:
        raise ValueError("inference queue rows do not match the bound source projection")
    if _read_jsonl(path.parent / "excluded_records.jsonl") != projected_excluded:
        raise ValueError("inference queue exclusions do not match the bound source projection")
    if int(manifest.get("records", -1)) != len(rows):
        raise ValueError("inference queue record-count binding mismatch")
    ids = [str(row.get("sprite_id") or "") for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("inference queue has missing or duplicate sprite IDs")
    if str(manifest.get("ordered_sprite_ids_hash") or "") != _stable_hash(ids):
        raise ValueError("inference queue ordered-ID binding mismatch")
    for row in rows:
        if str(row.get("schema_version")) != INFERENCE_QUEUE_SCHEMA or str(row.get("queue_id") or "") != queue_id:
            raise ValueError("inference queue row schema or queue-ID binding mismatch")
        if str(row.get("quality_state") or "") not in QUALITY_TERMINAL:
            raise ValueError("inference queue row has unknown or unreviewed quality state")
        event_value = row.get("quality_event")
        try:
            event = ReviewEvent.from_dict(event_value) if isinstance(event_value, Mapping) else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"inference queue quality-event binding invalid: {exc}") from None
        if (
            event is None
            or event.action not in QUALITY_ACTIONS
            or event.sprite_id != str(row.get("sprite_id"))
            or event.human_outcome != str(row.get("quality_state"))
        ):
            raise ValueError("inference queue quality-event decision binding mismatch")
    return {
        "queue_id": queue_id,
        "queue_sha256": actual_hash,
        "rows": rows,
        "manifest": manifest,
        "freeze": freeze,
        "source_bindings_verified": True,
    }


def audit_existing_events(records_path: str | Path, human_truth: str | Path) -> dict[str, Any]:
    records = _read_jsonl(Path(records_path))
    by_id = {str(row.get("sprite_id", "")): row for row in records}
    categories: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "schema_incompatible",
            "identity_mismatch",
            "proposal_hash_mismatch",
            "malformed",
            "valid_quality",
            "valid_semantic",
            "ignored_non_authoritative",
        )
    }
    diagnostics: dict[str, list[dict[str, Any]]] = {
        "unsafe_null_acceptance": [],
        "premature_completion": [],
    }
    parsed: list[ReviewEvent] = []
    raw_lines = [line for line in Path(human_truth).read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_number, raw in enumerate(raw_lines, 1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            categories["malformed"].append({"line": line_number, "reason": f"malformed_json:{exc}"})
            continue
        if not isinstance(value, Mapping):
            categories["malformed"].append({"line": line_number, "reason": "event_must_be_a_json_object"})
            continue
        schema_version = str(value.get("schema_version") or "")
        if schema_version != REVIEW_EVENT_SCHEMA_VERSION:
            reason = "missing_schema_version" if not schema_version else f"unsupported_schema:{schema_version}"
            categories["schema_incompatible"].append({"line": line_number, "reason": reason})
            continue
        try:
            event = ReviewEvent.from_dict(value)
        except (ValueError, TypeError) as exc:
            categories["malformed"].append({"line": line_number, "reason": str(exc)})
            continue
        record = by_id.get(event.sprite_id)
        summary = {
            "line": line_number,
            "event_id": event.event_id,
            "sprite_id": event.sprite_id,
            "action": event.action,
        }
        parsed.append(event)
        if event.action in QUALITY_ACTIONS:
            category, reason = _quality_event_authority(record, event)
        elif event.field_name:
            category, reason = _semantic_event_authority(record, event)
        else:
            category, reason = "ignored_non_authoritative", "record_action_is_not_two_pass_authority"
        categories[category].append({**summary, "reason": reason})
        if record is not None and event.field_name:
            validation = validate_semantic_field(record, event.field_name)
            if event.human_outcome == "correct" and not validation.valid:
                diagnostics["unsafe_null_acceptance"].append({**summary, "value_state": validation.value_state})
        if event.metadata.get("record_completed"):
            diagnostics["premature_completion"].append({**summary, "reason": "legacy_generic_completion_flag"})
    quality = resolve_quality_decisions(records, parsed)
    incomplete = [sprite_id for sprite_id, decision in quality.items() if not decision.complete]
    category_counts = {name: len(items) for name, items in categories.items()}
    return {
        "schema_version": "label_v4_existing_event_audit_v2",
        "events_total": len(raw_lines),
        "category_counts": category_counts,
        "categories_exhaustive": sum(category_counts.values()) == len(raw_lines),
        "categories": categories,
        "diagnostic_counts": {name: len(items) for name, items in diagnostics.items()},
        "diagnostics": diagnostics,
        "legacy_category_counts": {
            "valid_quality_event": category_counts["valid_quality"],
            "valid_semantic_event": category_counts["valid_semantic"],
            "unsafe_null_acceptance": len(diagnostics["unsafe_null_acceptance"]),
            "premature_completion": len(diagnostics["premature_completion"]),
        },
        "incomplete": incomplete,
        "incomplete_count": len(incomplete),
        "correction_plan": [
            "Append a superseding valid quality event for incompatible or changed quality decisions.",
            "Append explicit terminal semantic judgments; never delete unsafe historical events.",
            "Exclude unsafe null acceptances and premature completion flags from denominators.",
        ],
    }


def _load_events(path: Path) -> list[ReviewEvent]:
    result: list[ReviewEvent] = []
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            result.append(ReviewEvent.from_dict(json.loads(raw)))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
