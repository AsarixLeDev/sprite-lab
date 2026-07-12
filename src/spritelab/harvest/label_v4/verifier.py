"""Strict normalization and decision bookkeeping for Labeling-v4 verification.

Provider artifacts remain raw evidence.  This module accepts only results for
known, directional claim identifiers and derives effects by exact claim or
conflict id; dispute codes are deliberately never used as resolution keys.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

VERIFIER_NORMALIZATION_SCHEMA_VERSION = "label_verifier_normalized_v1.0"

VerifierVerdict = Literal["supported", "contradicted", "unresolved"]
VerifierIndependence = Literal[
    "same_model_independent_prompt",
    "different_model_independent_prompt",
]
VerifierEffect = Literal[
    "conflict_resolved",
    "conflict_retained",
    "claim_rejected",
    "claim_abstained",
    "risk_changed",
    "no_decision_change",
]

ALLOWED_VERIFIER_VERDICTS = frozenset({"supported", "contradicted", "unresolved"})
SAME_MODEL_INDEPENDENT_PROMPT: VerifierIndependence = "same_model_independent_prompt"
DIFFERENT_MODEL_INDEPENDENT_PROMPT: VerifierIndependence = "different_model_independent_prompt"


@dataclass(frozen=True)
class NormalizedVerifierClaimResult:
    claim_id: str
    verdict: VerifierVerdict
    visible_support: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "visible_support": list(self.visible_support),
            "unsupported_fields": list(self.unsupported_fields),
        }


@dataclass(frozen=True)
class RejectedVerifierResult:
    index: int | None
    reason: str
    raw_result: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "reason": self.reason,
            "raw_result": copy.deepcopy(self.raw_result),
        }


@dataclass(frozen=True)
class ParsedVerifierResponse:
    claim_results: tuple[NormalizedVerifierClaimResult, ...] = ()
    rejected_results: tuple[RejectedVerifierResult, ...] = ()
    unanswered_claim_ids: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()
    schema_version: str = VERIFIER_NORMALIZATION_SCHEMA_VERSION

    @property
    def valid(self) -> bool:
        return not self.rejected_results and not self.unanswered_claim_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim_results": [result.to_dict() for result in self.claim_results],
            "rejected_results": [result.to_dict() for result in self.rejected_results],
            "unanswered_claim_ids": list(self.unanswered_claim_ids),
            "unsupported_fields": list(self.unsupported_fields),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class VerifierClaimEffectRecord:
    claim_id: str
    field: str
    verdict: VerifierVerdict
    conflict_id: str | None
    effects: tuple[VerifierEffect, ...]
    decision_change: Literal["decision_changed", "no_decision_change"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "field": self.field,
            "verdict": self.verdict,
            "conflict_id": self.conflict_id,
            "effects": list(self.effects),
            "decision_change": self.decision_change,
        }


@dataclass(frozen=True)
class VerifierDecisionSummary:
    claim_effects: tuple[VerifierClaimEffectRecord, ...]
    overall_effects: tuple[VerifierEffect, ...]
    decision_change: Literal["decision_changed", "no_decision_change"]
    resolved_conflict_ids: tuple[str, ...] = ()
    retained_conflict_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_effects": [effect.to_dict() for effect in self.claim_effects],
            "overall_effects": list(self.overall_effects),
            "decision_change": self.decision_change,
            "resolved_conflict_ids": list(self.resolved_conflict_ids),
            "retained_conflict_ids": list(self.retained_conflict_ids),
        }


def classify_verifier_independence(
    verifier_model_identity: str,
    upstream_model_identities: Sequence[str],
) -> VerifierIndependence:
    """Label prompt independence without overstating same-model evidence."""

    verifier_identity = str(verifier_model_identity).strip()
    upstream = tuple(str(value).strip() for value in upstream_model_identities if str(value).strip())
    if not verifier_identity:
        raise ValueError("verifier_model_identity is required")
    if not upstream:
        raise ValueError("at least one upstream model identity is required")
    if verifier_identity in upstream:
        return SAME_MODEL_INDEPENDENT_PROMPT
    return DIFFERENT_MODEL_INDEPENDENT_PROMPT


def parse_verifier_response(
    raw_output: Mapping[str, Any] | Any,
    *,
    known_claims: Sequence[Mapping[str, Any] | str],
) -> ParsedVerifierResponse:
    """Reject unknown ids, duplicate ids, and non-enum verdicts.

    Rejected provider values are copied into diagnostics.  ``raw_output`` is
    never mutated and should continue to be stored on the ProviderArtifact.
    """

    known, known_order = _known_claim_index(known_claims)
    if not isinstance(raw_output, Mapping):
        return ParsedVerifierResponse(
            rejected_results=(RejectedVerifierResult(None, "verifier_output_not_object", raw_output),),
            unanswered_claim_ids=known_order,
        )
    raw_results = raw_output.get("claim_results")
    unsupported_fields = _string_tuple(raw_output.get("unsupported_fields"))
    if not _is_sequence(raw_results):
        return ParsedVerifierResponse(
            rejected_results=(RejectedVerifierResult(None, "claim_results_not_array", copy.deepcopy(raw_results)),),
            unanswered_claim_ids=known_order,
            unsupported_fields=unsupported_fields,
        )

    ids = [str(value.get("claim_id", "")).strip() for value in raw_results if isinstance(value, Mapping)]
    duplicate_ids = {claim_id for claim_id, count in Counter(ids).items() if claim_id and count > 1}
    accepted: list[NormalizedVerifierClaimResult] = []
    rejected: list[RejectedVerifierResult] = []
    accepted_ids: set[str] = set()
    for index, raw_result in enumerate(raw_results):
        if not isinstance(raw_result, Mapping):
            rejected.append(RejectedVerifierResult(index, "claim_result_not_object", copy.deepcopy(raw_result)))
            continue
        claim_id = str(raw_result.get("claim_id", "")).strip()
        if not claim_id:
            rejected.append(RejectedVerifierResult(index, "claim_id_missing", copy.deepcopy(raw_result)))
            continue
        if claim_id in duplicate_ids:
            rejected.append(RejectedVerifierResult(index, "duplicate_claim_id", copy.deepcopy(raw_result)))
            continue
        if claim_id not in known:
            rejected.append(RejectedVerifierResult(index, "unknown_claim_id", copy.deepcopy(raw_result)))
            continue
        verdict = str(raw_result.get("verdict", ""))
        if verdict not in ALLOWED_VERIFIER_VERDICTS:
            rejected.append(RejectedVerifierResult(index, "invalid_verdict", copy.deepcopy(raw_result)))
            continue
        accepted.append(
            NormalizedVerifierClaimResult(
                claim_id=claim_id,
                verdict=verdict,  # type: ignore[arg-type]
                visible_support=_string_tuple(raw_result.get("visible_support")),
                unsupported_fields=_string_tuple(raw_result.get("unsupported_fields")),
            )
        )
        accepted_ids.add(claim_id)
    return ParsedVerifierResponse(
        claim_results=tuple(accepted),
        rejected_results=tuple(rejected),
        unanswered_claim_ids=tuple(claim_id for claim_id in known_order if claim_id not in accepted_ids),
        unsupported_fields=unsupported_fields,
    )


def derive_verifier_effects(
    parsed: ParsedVerifierResponse,
    *,
    known_claims: Sequence[Mapping[str, Any] | str],
    risk_changed_claim_ids: Sequence[str] = (),
) -> VerifierDecisionSummary:
    """Derive auditable effects using exact claim/conflict identifiers only.

    A supported result resolves a conflict only when its known claim explicitly
    opts in with ``resolve_on_supported`` and supplies a unique ``conflict_id``.
    A ``dispute_code`` is intentionally ignored.
    """

    known, _known_order = _known_claim_index(known_claims)
    risk_changed = {str(value) for value in risk_changed_claim_ids}
    unknown_risk_ids = risk_changed - set(known)
    if unknown_risk_ids:
        raise ValueError(f"unknown risk-changed claim ids: {sorted(unknown_risk_ids)}")

    records: list[VerifierClaimEffectRecord] = []
    resolved: list[str] = []
    retained: list[str] = []
    for result in parsed.claim_results:
        claim = known[result.claim_id]
        field = str(claim.get("field", ""))
        conflict_id = str(claim.get("conflict_id", "")).strip() or None
        resolve_on_supported = bool(claim.get("resolve_on_supported", False))
        if resolve_on_supported and conflict_id is None:
            raise ValueError(f"claim {result.claim_id!r} resolves without a conflict_id")
        unsupported = field and (field in parsed.unsupported_fields or field in result.unsupported_fields)
        effects: list[VerifierEffect] = []
        changed = False
        if unsupported or result.verdict == "unresolved":
            effects.append("claim_abstained")
            if conflict_id:
                effects.append("conflict_retained")
                retained.append(conflict_id)
        elif result.verdict == "contradicted":
            effects.append("claim_rejected")
            changed = True
            if conflict_id:
                effects.append("conflict_retained")
                retained.append(conflict_id)
        elif resolve_on_supported:
            effects.append("conflict_resolved")
            changed = True
            resolved.append(str(conflict_id))
        elif conflict_id:
            effects.append("conflict_retained")
            retained.append(conflict_id)
        if result.claim_id in risk_changed:
            effects.append("risk_changed")
            changed = True
        if not effects:
            effects.append("no_decision_change")
        records.append(
            VerifierClaimEffectRecord(
                claim_id=result.claim_id,
                field=field,
                verdict=result.verdict,
                conflict_id=conflict_id,
                effects=tuple(dict.fromkeys(effects)),
                decision_change="decision_changed" if changed else "no_decision_change",
            )
        )

    overall_effects = tuple(dict.fromkeys(effect for record in records for effect in record.effects))
    decision_changed = any(record.decision_change == "decision_changed" for record in records)
    if not decision_changed and "no_decision_change" not in overall_effects:
        overall_effects = (*overall_effects, "no_decision_change")
    if not overall_effects:
        overall_effects = ("no_decision_change",)
    return VerifierDecisionSummary(
        claim_effects=tuple(records),
        overall_effects=overall_effects,
        decision_change="decision_changed" if decision_changed else "no_decision_change",
        resolved_conflict_ids=tuple(dict.fromkeys(resolved)),
        retained_conflict_ids=tuple(value for value in dict.fromkeys(retained) if value not in resolved),
    )


def _known_claim_index(
    known_claims: Sequence[Mapping[str, Any] | str],
) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
    index: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    for raw_claim in known_claims:
        claim = {"claim_id": raw_claim} if isinstance(raw_claim, str) else raw_claim
        if not isinstance(claim, Mapping):
            raise TypeError("known claims must be mappings or claim-id strings")
        claim_id = str(claim.get("claim_id", "")).strip()
        if not claim_id:
            raise ValueError("known claim_id is required")
        if claim_id in index:
            raise ValueError(f"duplicate known claim_id: {claim_id}")
        index[claim_id] = dict(claim)
        order.append(claim_id)
    return index, tuple(order)


def _string_tuple(value: Any) -> tuple[str, ...]:
    values = value if _is_sequence(value) else (value,) if value is not None else ()
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
