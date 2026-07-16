"""Fail-closed product projection of authoritative memorization evidence."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from spritelab.evaluation.candidate_bundle import load_candidate_bundle
from spritelab.evaluation.memorization import (
    HARD_EVIDENCE_CLASSES,
    REVIEW_REQUIRED_EVIDENCE_CLASSES,
    WARNING_EVIDENCE_CLASSES,
    EvidenceClass,
    parse_evidence_class,
)
from spritelab.evaluation.memorization_review import bound_event_identity, canonical_sha256, replay_review_events
from spritelab.product_features.evaluation.models import ReviewFeatureLink

PROMOTION_INTEGRITY_MESSAGE = "Promotion integrity is not currently certified."
INCOMPLETE_EVIDENCE_MESSAGE = (
    "Memorization review evidence is incomplete. Evaluation must be rerun with the current evidence contract."
)
INVALID_EVIDENCE_MESSAGE = "Review evidence is invalid or outdated. Evaluation must be rerun."
_CLEARING_OUTCOMES = frozenset({"different_sprite", "common_generic_shape", "likely_false_positive"})
_BLOCKING_OUTCOMES = frozenset({"same_sprite_or_memorized"})
_CONTROLLED_REVIEW_OUTCOMES = (
    "same_sprite_or_memorized",
    "uncertain",
    "different_sprite",
    "common_generic_shape",
    "likely_false_positive",
)


class MemorizationDisplayState(StrEnum):
    HARD_EVIDENCE = "Hard evidence"
    REVIEW_REQUIRED = "Review required"
    WARNINGS = "Warnings"
    CLEARED = "Cleared by valid review"
    INVALID_REVIEW = "Invalid review"
    NOT_COMPARABLE = "Not comparable"
    NO_MATERIAL_MATCH = "No material match"


def _empty_display(message: str, *, evidence_state: str, reasons: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "schema_version": "spritelab.product.memorization-display.v2",
        "evidence_state": evidence_state,
        "items": [],
        "counts": {state.value: 0 for state in MemorizationDisplayState},
        "review_required_count": 0,
        "review_message": message,
        "review_link": None,
        "review_contract": "spritelab.evaluation.memorization_review.signed-v2",
        "review_action_available": False,
        "action_unavailable_reason": message,
        "validation_reasons": list(reasons),
        "legacy_reviews": [],
        "writes_review_log": False,
    }


def _legacy_rows(review_log: Path | None) -> list[dict[str, Any]]:
    if review_log is None:
        return []
    replay = replay_review_events(review_log)
    return [
        {
            **dict(event),
            "display_state": "Legacy review (read-only)",
            "promotion_authority": False,
            "review_action_available": False,
        }
        for event in replay.legacy_events
    ]


def _pair_reasons(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
    reasons = [
        {
            "evidence_class": pair.get("evidence_class"),
            "evidence_metrics": pair.get("evidence_metrics"),
            "evidence_diagnostics": pair.get("evidence_diagnostics"),
        }
    ]
    additional = pair.get("evidence_reasons")
    if isinstance(additional, list):
        for reason in additional:
            if isinstance(reason, str):
                reasons.append(
                    {
                        "evidence_class": reason,
                        "evidence_metrics": pair.get("evidence_metrics"),
                        "evidence_diagnostics": pair.get("evidence_diagnostics"),
                    }
                )
            elif isinstance(reason, Mapping):
                reasons.append(dict(reason))
    return reasons


def _project_pair(
    pair: Mapping[str, Any],
    *,
    chain: Any,
    log_invalid: bool,
) -> dict[str, Any]:
    evidence_class = parse_evidence_class(pair.get("evidence_class"))
    event = chain.authoritative_event if chain is not None and chain.chain_status == "valid" else None
    outcome = str(event.get("review_outcome")) if isinstance(event, Mapping) else None
    authoritative = event is not None and not log_invalid
    if evidence_class in HARD_EVIDENCE_CLASSES:
        current_state = "blocked"
        display_state = MemorizationDisplayState.HARD_EVIDENCE
        unavailable = "Hard memorization evidence cannot be cleared by review."
    elif evidence_class in REVIEW_REQUIRED_EVIDENCE_CLASSES:
        chain_status = chain.chain_status if chain is not None else "missing"
        if chain_status == "identity_mismatch":
            current_state = "not_comparable"
            display_state = MemorizationDisplayState.NOT_COMPARABLE
            unavailable = "The signed review has the wrong immutable identity binding."
        elif log_invalid or chain_status in {"invalid", "incomplete", "contradictory"}:
            current_state = "invalid_review"
            display_state = MemorizationDisplayState.INVALID_REVIEW
            unavailable = "The signed review event chain is invalid."
        elif authoritative and outcome in _BLOCKING_OUTCOMES:
            current_state = "blocked"
            display_state = MemorizationDisplayState.HARD_EVIDENCE
            unavailable = "The authoritative signed review blocks this pair."
        elif authoritative and outcome in _CLEARING_OUTCOMES:
            current_state = "cleared_by_valid_bound_review"
            display_state = MemorizationDisplayState.CLEARED
            unavailable = ""
        else:
            current_state = "review_required"
            display_state = MemorizationDisplayState.REVIEW_REQUIRED
            unavailable = "" if chain_status in {"missing", "valid"} else "The review chain is unavailable."
    elif evidence_class in WARNING_EVIDENCE_CLASSES:
        current_state = "not_comparable" if log_invalid else "warning_only"
        display_state = MemorizationDisplayState.NOT_COMPARABLE if log_invalid else MemorizationDisplayState.WARNINGS
        unavailable = "Warning-only evidence has no review action."
    elif evidence_class is EvidenceClass.NO_MATERIAL_MATCH:
        current_state = "not_comparable" if log_invalid else "no_material_match"
        display_state = (
            MemorizationDisplayState.NOT_COMPARABLE if log_invalid else MemorizationDisplayState.NO_MATERIAL_MATCH
        )
        unavailable = "No memorization review is required."
    else:  # pragma: no cover - parse_evidence_class is exhaustive
        current_state = "not_comparable"
        display_state = MemorizationDisplayState.NOT_COMPARABLE
        unavailable = "Unsupported evidence class."
    chain_status = chain.chain_status if chain is not None else "not_applicable"
    actionable = (
        evidence_class in REVIEW_REQUIRED_EVIDENCE_CLASSES and chain_status in {"missing", "valid"} and not log_invalid
    )
    return {
        "pair_id": str(pair["pair_id"]),
        "evidence_class": evidence_class.value,
        "evidence_reasons": _pair_reasons(pair),
        "generated_image": str(pair.get("generated_png_path") or ""),
        "training_comparison_image": str(pair.get("training_image_path") or ""),
        "diagnostics": pair.get("evidence_diagnostics"),
        "metrics": pair.get("evidence_metrics"),
        "display_state": display_state.value,
        "current_review_state": current_state,
        "review_authoritative": authoritative,
        "authoritative_event_sha256": event.get("event_sha256") if authoritative else None,
        "event_chain_status": chain_status,
        "event_chain_valid": chain_status in {"missing", "valid"} and not log_invalid,
        "review_action_available": actionable,
        "controlled_review_outcomes": list(_CONTROLLED_REVIEW_OUTCOMES) if actionable else [],
        "clear_action_available": actionable and evidence_class not in HARD_EVIDENCE_CLASSES,
        "action_unavailable_reason": "" if actionable else unavailable,
        "candidate_bundle_path": str(pair.get("candidate_bundle_path") or ""),
    }


def memorization_display(
    evidence: Path | Sequence[Mapping[str, Any]] | None,
    *,
    review_log: Path | None = None,
    expected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only a current strict-v2 bundle through authoritative replay."""
    if evidence is None or not isinstance(evidence, Path) or not evidence.is_file():
        display = _empty_display(INCOMPLETE_EVIDENCE_MESSAGE, evidence_state="incomplete")
        display["legacy_reviews"] = _legacy_rows(review_log)
        return display
    validation = load_candidate_bundle(evidence, expected_context=expected_context)
    if not validation.valid:
        display = _empty_display(
            INVALID_EVIDENCE_MESSAGE,
            evidence_state=validation.state,
            reasons=validation.reasons,
        )
        display["legacy_reviews"] = _legacy_rows(review_log)
        return display
    bundle = validation.bundle
    pairs = list(validation.pairs)
    review_pairs = {
        str(pair["pair_id"]): pair
        for pair in pairs
        if parse_evidence_class(pair.get("evidence_class")) in REVIEW_REQUIRED_EVIDENCE_CLASSES
    }
    expected = {pair_id: bound_event_identity(bundle, pair) for pair_id, pair in review_pairs.items()}
    replay_path = review_log or evidence.with_name("review_events.jsonl")
    replay = replay_review_events(replay_path, expected_identities=expected)
    extra_pair_ids = replay.seen_pair_ids - set(review_pairs)
    log_invalid = bool(replay.global_invalid_events or replay.legacy_events or extra_pair_ids)
    items = [
        _project_pair(
            {**pair, "candidate_bundle_path": str(evidence.resolve())},
            chain=replay.chains.get(str(pair["pair_id"])),
            log_invalid=log_invalid,
        )
        for pair in pairs
    ]
    counts = Counter(str(item["display_state"]) for item in items)
    required = sum(item["current_review_state"] == "review_required" for item in items)
    actionable = any(bool(item["review_action_available"]) for item in items)
    link = ReviewFeatureLink()
    legacy = _legacy_rows(replay_path)
    return {
        "schema_version": "spritelab.product.memorization-display.v2",
        "evidence_state": "complete",
        "candidate_evidence_sha256": bundle.get("candidate_evidence_sha256"),
        "items": items,
        "counts": {state.value: counts[state.value] for state in MemorizationDisplayState},
        "review_required_count": required,
        "review_message": f"{required} memorization candidates require review",
        "review_link": link.to_dict() if actionable else None,
        "review_contract": "spritelab.evaluation.memorization_review.signed-v2",
        "review_action_available": actionable,
        "action_unavailable_reason": "" if actionable else "No valid current bound-v2 review candidate is actionable.",
        "validation_reasons": [
            *replay.invalid_reasons,
            *(
                [f"review events exist for non-review candidates: {', '.join(sorted(extra_pair_ids))}"]
                if extra_pair_ids
                else []
            ),
        ],
        "legacy_reviews": legacy,
        "writes_review_log": False,
    }


def _repository_root(path: Path) -> Path | None:
    for parent in (path.resolve(), *path.resolve().parents):
        if (parent / ".git").exists():
            return parent
    return None


def promotion_integrity_display(
    audit_path: Path | None,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Display authorization state only when the complete semantic code identity is current."""
    audit: dict[str, Any] = {}
    if audit_path and audit_path.is_file():
        try:
            value = json.loads(audit_path.read_text(encoding="utf-8"))
            audit = value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            audit = {}
    verdict = str(audit.get("overall_verdict") or audit.get("verdict") or audit.get("status") or "NOT_AUDITED").upper()
    root = (
        repository_root.resolve()
        if repository_root is not None
        else (_repository_root(audit_path.parent) if audit_path else None)
    )
    if root is not None:
        from spritelab.v3.status import verify_memorization_audit_applicability

        verification = verify_memorization_audit_applicability(root, audit or None)
        identity_current = verification.identity_current
        applicability = verification.status.value
        applicability_reasons = list(verification.reasons)
    else:
        identity_current = False
        applicability = "NOT_COMPARABLE"
        applicability_reasons = ["repository_root_unavailable"]
    authorized_by_audit = (
        verdict in {"PASS", "PASSED"}
        and identity_current
        and applicability == "PASS"
        and isinstance(audit.get("authorization"), Mapping)
        and audit["authorization"].get("checkpoint_promotion") is True
    )
    return {
        "schema_version": "spritelab.product.promotion-display.v2",
        "integrity_certified": authorized_by_audit,
        "promotion_authorized": False,
        "message": "Promotion integrity is certified for display; promotion is not authorized by this feature."
        if authorized_by_audit
        else PROMOTION_INTEGRITY_MESSAGE,
        "audit_verdict": verdict,
        "audit_applicability": applicability,
        "audit_applicability_reasons": applicability_reasons,
        "audit_code_identity_current": identity_current,
        "audit_identity": canonical_sha256(audit) if audit else None,
        "actions": [],
        "promotion_actions_performed": 0,
    }
