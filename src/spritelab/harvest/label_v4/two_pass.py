"""Strict two-pass quality and semantic calibration contracts for Labeling v4."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def resolve_quality_decisions(
    records: Sequence[Mapping[str, Any]], events: Sequence[ReviewEvent]
) -> dict[str, QualityResolution]:
    """Resolve the latest schema- and proposal-compatible terminal quality event."""

    by_id = {str(record.get("sprite_id", "")): record for record in records}
    valid: dict[str, list[ReviewEvent]] = {sprite_id: [] for sprite_id in by_id}
    ignored = Counter()
    for event in events:
        record = by_id.get(event.sprite_id)
        if not _compatible_event(event) or record is None or event.action not in QUALITY_ACTIONS or event.field_name:
            ignored[event.sprite_id] += 1
            continue
        audit_id = str(record.get("audit_id", ""))
        event_audit_id = str(event.metadata.get("audit_id") or event.session_id or "")
        expected_hash = immutable_proposal_digest(record)
        if event_audit_id != audit_id or (event.proposal_hash and event.proposal_hash != expected_hash):
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


def quality_resume_index(records: Sequence[Mapping[str, Any]], events: Sequence[ReviewEvent]) -> int:
    resolved = resolve_quality_decisions(records, events)
    return next(
        (index for index, record in enumerate(records) if not resolved[str(record.get("sprite_id", ""))].complete),
        0,
    )


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
    input_hash = str(provenance.get("image_hash") or "")
    if not input_hash:
        return False, "missing_model_input_binding"
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
    expected_hash = immutable_proposal_digest(record)
    audit_id = str(record.get("audit_id", ""))
    for event in events:
        if not _compatible_event(event) or event.sprite_id != record.get("sprite_id") or not event.field_name:
            continue
        if event.proposal_hash and event.proposal_hash != expected_hash:
            continue
        event_audit_id = str(event.metadata.get("audit_id") or event.session_id or "")
        if event_audit_id and event_audit_id != audit_id:
            continue
        if event.human_outcome in SEMANTIC_TERMINAL_OUTCOMES:
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
        if validate_semantic_field(record, name).value_state == "model_abstained"
        and (name not in latest or latest[name].human_outcome != "model_abstention_accepted")
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
            reason = ""
            if quality[sprite_id].effective_state not in QUALITY_ELIGIBLE:
                reason = quality[sprite_id].effective_state
            elif not completion["complete"]:
                reason = "incomplete_semantic_record"
            elif not validation.valid or validation.value_state != "known":
                reason = validation.reason
            elif event.human_outcome == "not_applicable":
                reason = "not_applicable"
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
        "scorable": scorable,
        "excluded": excluded,
        "abstention_judgments": abstention_judgments,
    }


def freeze_inference_queue(
    audit_selection: str | Path,
    prefilled_records: str | Path,
    human_truth: str | Path,
    output_root: str | Path,
    *,
    inclusion_policy: Sequence[str] = tuple(sorted(QUALITY_ELIGIBLE)),
    allow_partial: bool = False,
) -> dict[str, Any]:
    selection_path, prefilled_path, truth_path = map(Path, (audit_selection, prefilled_records, human_truth))
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
                    "quality_risk_penalty": 0.15 if decision.effective_state == "quality_uncertain_usable" else 0.0,
                    "quality_event": decision.event.to_dict() if decision.event else None,
                    "prefill_record_hash": _stable_hash(by_id[sprite_id]),
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
    queue_id = _stable_hash(base_queue)
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
        "records": len(queue),
        "audit_selection_sha256": _file_hash(selection_path),
        "prefilled_records_sha256": _file_hash(prefilled_path),
        "human_truth_sha256": _file_hash(truth_path),
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


def verify_frozen_inference_queue(queue_path: str | Path) -> dict[str, Any]:
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
    rows = _read_jsonl(path)
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
    unhashed_rows = [{key: value for key, value in row.items() if key != "queue_id"} for row in rows]
    if _stable_hash(unhashed_rows) != queue_id:
        raise ValueError("inference queue content-to-ID binding mismatch")
    return {"queue_id": queue_id, "queue_sha256": actual_hash, "rows": rows, "manifest": manifest, "freeze": freeze}


def audit_existing_events(records_path: str | Path, human_truth: str | Path) -> dict[str, Any]:
    records = _read_jsonl(Path(records_path))
    by_id = {str(row.get("sprite_id", "")): row for row in records}
    categories: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "valid_quality_event",
            "valid_semantic_event",
            "unsafe_null_acceptance",
            "premature_completion",
            "schema_incompatible",
            "incomplete",
        )
    }
    parsed: list[ReviewEvent] = []
    raw_lines = [line for line in Path(human_truth).read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_number, raw in enumerate(raw_lines, 1):
        try:
            value = json.loads(raw)
            event = ReviewEvent.from_dict(value)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            categories["schema_incompatible"].append({"line": line_number, "reason": str(exc)})
            continue
        record = by_id.get(event.sprite_id)
        summary = {
            "line": line_number,
            "event_id": event.event_id,
            "sprite_id": event.sprite_id,
            "action": event.action,
        }
        if record is None or (event.session_id and event.session_id != str(record.get("audit_id", ""))):
            categories["schema_incompatible"].append({**summary, "reason": "record_or_manifest_mismatch"})
            continue
        parsed.append(event)
        if event.action in QUALITY_ACTIONS:
            categories["valid_quality_event"].append(summary)
        elif event.field_name:
            validation = validate_semantic_field(record, event.field_name)
            if event.human_outcome == "correct" and not validation.valid:
                categories["unsafe_null_acceptance"].append({**summary, "value_state": validation.value_state})
            else:
                categories["valid_semantic_event"].append(summary)
        if event.metadata.get("record_completed"):
            categories["premature_completion"].append({**summary, "reason": "legacy_generic_completion_flag"})
    quality = resolve_quality_decisions(records, parsed)
    incomplete = [sprite_id for sprite_id, decision in quality.items() if not decision.complete]
    categories["incomplete"] = [{"sprite_id": sprite_id, "reason": "quality_unreviewed"} for sprite_id in incomplete]
    return {
        "schema_version": "label_v4_existing_event_audit_v1",
        "events_total": len(raw_lines),
        "category_counts": {name: len(items) for name, items in categories.items()},
        "categories": categories,
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
