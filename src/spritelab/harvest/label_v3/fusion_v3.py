"""Auto-Labeling v3: calibrated per-field fusion and abstention.

Operates per field, with explicit evidence reliability, dependency tracking,
calibration support, and contradiction handling. No max(), no naive averaging,
no double-counting of correlated sources.

Accepts:
  - per-field accepted, abstained, quarantined, rejected decisions
  - broader hierarchy fallback when exact identity is unsafe
  - structured contradiction resolution
  - versioned calibration artifacts
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.harvest.label_v3.evidence import (
    EvidenceItem,
)
from spritelab.harvest.label_v3.field_decisions import (
    DecisionReason,
    FieldDecision,
    FieldState,
)
from spritelab.harvest.label_v3.impossible_combinations import validate_impossible_combinations
from spritelab.harvest.label_v3.reason_codes import (
    ContradictionSeverity,
    contradiction_severity,
)
from spritelab.harvest.label_v3.taxonomy_v3 import (
    broader_hierarchy_node,
    deepest_supported_node,
    taxonomy_relation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldFusionInput:
    field: str
    evidence_items: tuple[EvidenceItem, ...]
    calibration_stratum: str = "global"
    precision_target: float = 0.99
    confidence_level: float = 0.95
    min_evidence_count: int = 1
    auto_accept_enabled: bool = True
    policy_hash: str = ""


@dataclass(frozen=True)
class FieldFusionResult:
    field: str
    decision: FieldDecision
    evidence_used: tuple[str, ...]
    evidence_excluded: tuple[str, ...]
    exclusion_reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.decision.state == "accepted"


def _dependency_group(evidence: EvidenceItem) -> str:
    return evidence.dependency_group or evidence.evidence_family


def _is_independent(evidence_a: EvidenceItem, evidence_b: EvidenceItem) -> bool:
    """Two evidence items are independent if they do not share a dependency group."""
    if evidence_a.evidence_id == evidence_b.evidence_id:
        return False
    group_a = _dependency_group(evidence_a)
    group_b = _dependency_group(evidence_b)
    if group_a and group_b and group_a == group_b:
        return False
    if evidence_a.source_hints_exposed and evidence_b.source_hints_exposed:
        return False
    return True


def _independent_count(evidence: EvidenceItem, all_evidence: Sequence[EvidenceItem]) -> int:
    """Count how many other evidence items are independent of this one."""
    count = 0
    for other in all_evidence:
        if _is_independent(evidence, other):
            count += 1
    return count


def fuse_field(
    field_input: FieldFusionInput,
    *,
    calibration_support: dict[str, Any] | None = None,
) -> FieldFusionResult:
    """Fuse evidence for a single field, producing an accepted/abstained/rejected decision."""

    evidence_items = field_input.evidence_items
    field = field_input.field
    policy_hash = field_input.policy_hash

    used_ids: list[str] = []
    excluded_ids: list[str] = []
    exclusion_reasons: list[str] = []
    warnings: list[str] = []

    # Filter: keep only deterministic or non-degenerate evidence
    valid_evidence: list[EvidenceItem] = []
    for item in evidence_items:
        if "degenerate" in str(item.warnings).lower():
            excluded_ids.append(item.evidence_id)
            exclusion_reasons.append("degenerate_evidence")
            continue
        valid_evidence.append(item)

    if not valid_evidence:
        return FieldFusionResult(
            field=field,
            decision=FieldDecision(
                sprite_id=evidence_items[0].sprite_id if evidence_items else "",
                field_name=field,
                state="unlabeled",
                decision_reason="insufficient_evidence",
                policy_hash=policy_hash,
            ),
            evidence_used=(),
            evidence_excluded=tuple(excluded_ids),
            exclusion_reasons=tuple(exclusion_reasons),
            warnings=("no_valid_evidence",),
        )

    # Independent evidence count for reliability estimation
    independent_sets: dict[str, set[str]] = {}
    for item in valid_evidence:
        group = _dependency_group(item)
        independent_sets.setdefault(group, set()).add(item.evidence_id)

    independent_group_count = len(independent_sets)
    # Classify field-level conflict (hierarchy-aware): compatible broad/specific
    # agreement is NOT a conflict; genuine cross-subtree disagreement is.
    conflict = _field_conflict(valid_evidence, field=field)

    # If we have at least the minimum number of independent evidence
    has_enough = independent_group_count >= field_input.min_evidence_count

    # Try to derive a consensus value
    consensus_value = _consensus_value(valid_evidence, field=field)

    # Build candidates and alternatives
    candidates = _extract_candidates(valid_evidence, field=field)
    n_best = _extract_n_best(valid_evidence, field=field)

    # Estimate calibrated probability if calibration support available
    calibrated_estimate: float | None = None
    confidence_interval: tuple[float, float] | None = None

    if calibration_support:
        calibrated_estimate = calibration_support.get("calibrated_probability")
        ci_low = calibration_support.get("ci_lower")
        ci_high = calibration_support.get("ci_upper")
        if ci_low is not None and ci_high is not None:
            confidence_interval = (float(ci_low), float(ci_high))

    # Decision logic
    if not has_enough:
        decision = FieldDecision(
            sprite_id=valid_evidence[0].sprite_id,
            field_name=field,
            state="abstained",
            candidates=tuple(candidates) if candidates else (),
            n_best_alternatives=tuple(n_best)[:5] if n_best else (),
            evidence_refs=tuple(used_ids),
            excluded_evidence_refs=tuple(excluded_ids),
            exclusion_reasons=tuple(exclusion_reasons),
            decision_reason="insufficient_evidence",
            calibrated_estimate=calibrated_estimate,
            confidence_interval=confidence_interval,
            calibration_support=dict(calibration_support or {}),
            policy_hash=policy_hash,
        )
        return FieldFusionResult(
            field=field,
            decision=decision,
            evidence_used=tuple(used_ids),
            evidence_excluded=tuple(excluded_ids),
            exclusion_reasons=tuple(exclusion_reasons),
            warnings=("insufficient_independent_evidence",),
        )

    if conflict in ("contradiction", "ambiguous"):
        severity = _max_contradiction_severity(valid_evidence)
        if conflict == "contradiction":
            # Genuine cross-subtree conflict (or explicit code): quarantine when
            # high/fatal, else abstain.
            state = "quarantined" if severity.value in ("high", "fatal") else "abstained"
            reason: DecisionReason = "conflicting_evidence"
            warn = "contradictory_evidence"
        else:
            # Sibling ambiguity: never accept, but not a hard contradiction.
            state = "ambiguous"
            reason = "ambiguous_identity"
            warn = "ambiguous_evidence"
        decision = FieldDecision(
            sprite_id=valid_evidence[0].sprite_id,
            field_name=field,
            accepted_value=None,
            state=state,
            candidates=tuple(candidates) if candidates else (),
            n_best_alternatives=tuple(n_best)[:5] if n_best else (),
            evidence_refs=tuple(used_ids),
            excluded_evidence_refs=tuple(excluded_ids),
            exclusion_reasons=tuple(exclusion_reasons),
            contradiction_codes=tuple({code for item in valid_evidence for code in item.contradiction_codes}),
            decision_reason=reason,
            calibrated_estimate=calibrated_estimate,
            confidence_interval=confidence_interval,
            calibration_support=dict(calibration_support or {}),
            policy_hash=policy_hash,
        )
        return FieldFusionResult(
            field=field,
            decision=decision,
            evidence_used=tuple(used_ids),
            evidence_excluded=tuple(excluded_ids),
            exclusion_reasons=tuple(exclusion_reasons),
            warnings=(warn,),
        )

    if calibrated_estimate is not None and confidence_interval is not None:
        ci_lower = confidence_interval[0]
        if ci_lower >= field_input.precision_target and field_input.auto_accept_enabled:
            decision_state: FieldState = "accepted"
            decision_reason: DecisionReason = "strong_evidence_consensus"
        else:
            decision_state = "abstained"
            decision_reason = "calibration_insufficient"
            warnings.append(f"ci_lower={ci_lower:.4f} < target={field_input.precision_target:.4f}")
    else:
        decision_state = "abstained"
        decision_reason = "calibration_insufficient"
        warnings.append("no_calibration_support")

    # Never accept without a concrete consensus value, even when calibration
    # would otherwise permit it — an accepted field must carry a value.
    if decision_state == "accepted" and consensus_value is None:
        decision_state = "abstained"
        decision_reason = "insufficient_evidence"
        warnings.append("no_consensus_value")

    for item in valid_evidence:
        used_ids.append(item.evidence_id)

    decision = FieldDecision(
        sprite_id=valid_evidence[0].sprite_id,
        field_name=field,
        accepted_value=consensus_value if decision_state == "accepted" else None,
        state=decision_state,
        candidates=tuple(candidates) if candidates else (),
        n_best_alternatives=tuple(n_best)[:5] if n_best else (),
        evidence_refs=tuple(used_ids),
        excluded_evidence_refs=tuple(excluded_ids),
        exclusion_reasons=tuple(exclusion_reasons),
        calibrated_estimate=calibrated_estimate,
        confidence_interval=confidence_interval,
        calibration_support=dict(calibration_support or {}),
        decision_reason=decision_reason,
        policy_hash=policy_hash,
    )

    return FieldFusionResult(
        field=field,
        decision=decision,
        evidence_used=tuple(used_ids),
        evidence_excluded=tuple(excluded_ids),
        exclusion_reasons=tuple(exclusion_reasons),
        warnings=tuple(warnings),
    )


def fuse_hierarchical_object(
    all_evidence: Sequence[EvidenceItem],
    *,
    sprite_id: str = "",
    policy_hash: str = "",
    precision_target: float = 0.99,
    auto_accept_enabled: bool = True,
    calibration_support: dict[str, Any] | None = None,
) -> FieldFusionResult:
    """Fuse object identity with hierarchy backoff.

    Tries to accept the deepest safely-supported hierarchy node.
    Falls back to broader nodes when exact identity is unsafe.

    ``auto_accept_enabled`` is the caller's authorization to accept — the
    pipeline sets it only when a calibration lower bound meets the object
    precision target. When acceptance happens, ``calibration_support`` (if
    provided) is attached to the decision so every accepted object carries a
    calibration record.
    """

    candidates = _extract_candidates(all_evidence, field="canonical_object")
    used_ids = [item.evidence_id for item in all_evidence]

    cal_estimate = calibration_support.get("calibrated_probability") if calibration_support else None
    _ci_lo = calibration_support.get("ci_lower") if calibration_support else None
    _ci_hi = calibration_support.get("ci_upper") if calibration_support else None
    cal_ci = (float(_ci_lo), float(_ci_hi)) if _ci_lo is not None and _ci_hi is not None else None

    # Hierarchy-aware conflict on object identity. Compatible broad/specific
    # evidence (sword + bladed_weapon) is NOT a conflict; genuine cross-subtree
    # disagreement (sword + shield) quarantines; sibling ambiguity (sword +
    # dagger) abstains rather than accepting an arbitrary one.
    conflict = _field_conflict(all_evidence, field="canonical_object")
    if conflict in ("contradiction", "ambiguous"):
        return FieldFusionResult(
            field="canonical_object",
            decision=FieldDecision(
                sprite_id=sprite_id,
                field_name="canonical_object",
                state="quarantined" if conflict == "contradiction" else "ambiguous",
                candidates=tuple(candidates),
                n_best_alternatives=tuple((c, 0.0) for c in candidates[:5]),
                evidence_refs=tuple(used_ids),
                decision_reason="conflicting_evidence" if conflict == "contradiction" else "ambiguous_identity",
                policy_hash=policy_hash,
            ),
            evidence_used=tuple(used_ids),
            evidence_excluded=(),
            exclusion_reasons=(),
            warnings=(f"object_{conflict}",),
        )

    if not candidates:
        return FieldFusionResult(
            field="canonical_object",
            decision=FieldDecision(
                sprite_id=sprite_id,
                field_name="canonical_object",
                state="unknown",
                decision_reason="insufficient_evidence",
                policy_hash=policy_hash,
            ),
            evidence_used=(),
            evidence_excluded=(),
            exclusion_reasons=(),
            warnings=("no_object_candidates",),
        )

    best_candidate = candidates[0]
    node = deepest_supported_node(best_candidate)

    if node is None:
        return FieldFusionResult(
            field="canonical_object",
            decision=FieldDecision(
                sprite_id=sprite_id,
                field_name="canonical_object",
                state="novel",
                candidates=tuple(candidates),
                decision_reason="novel_class",
                evidence_refs=tuple(used_ids),
                policy_hash=policy_hash,
            ),
            evidence_used=tuple(used_ids),
            evidence_excluded=(),
            exclusion_reasons=(),
            warnings=("object_not_in_taxonomy",),
        )

    if node.open_set_allowed:
        return FieldFusionResult(
            field="canonical_object",
            decision=FieldDecision(
                sprite_id=sprite_id,
                field_name="canonical_object",
                accepted_value=node.name if auto_accept_enabled else None,
                hierarchy_node=node.name,
                state="accepted" if auto_accept_enabled else "abstained",
                candidates=tuple(candidates),
                evidence_refs=tuple(used_ids),
                calibrated_estimate=cal_estimate,
                confidence_interval=cal_ci,
                calibration_support=dict(calibration_support or {}),
                decision_reason="strong_evidence_consensus" if auto_accept_enabled else "calibration_insufficient",
                policy_hash=policy_hash,
            ),
            evidence_used=tuple(used_ids),
            evidence_excluded=(),
            exclusion_reasons=(),
            warnings=(),
        )

    broader = broader_hierarchy_node(node)
    fallback_value = broader.name if broader and broader.name != "object" else node.name

    return FieldFusionResult(
        field="canonical_object",
        decision=FieldDecision(
            sprite_id=sprite_id,
            field_name="canonical_object",
            accepted_value=None,
            hierarchy_node=fallback_value,
            state="abstained",
            candidates=tuple(candidates),
            n_best_alternatives=tuple((c, 0.0) for c in candidates[:5]),
            evidence_refs=tuple(used_ids),
            decision_reason="hierarchy_fallback",
            policy_hash=policy_hash,
        ),
        evidence_used=tuple(used_ids),
        evidence_excluded=(),
        exclusion_reasons=(),
        warnings=(f"hierarchy_fallback:{best_candidate}->{fallback_value}",),
    )


def validate_combinations(
    category: str,
    object_name: str,
    material: str = "",
    shape: str = "",
    tags: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return validate_impossible_combinations(
        category=category,
        canonical_object=object_name,
        material=material,
        shape=shape,
        tags=tags,
    )


# Reliability thresholds for contradiction detection. A weak signal must not be
# able to veto (contradict) a much stronger, independent one — the spec requires
# reliability-weighted fusion, not blanket disagreement vetoes.
_RELIABILITY_FLOOR = 0.6
_PEER_MARGIN = 0.25


# Fields whose values live in the v3 taxonomy and therefore support hierarchy
# compatibility (broad vs specific agreement). Other fields are flat: distinct
# values are simply disagreement.
_HIERARCHICAL_FIELDS = frozenset({"canonical_object", "category"})


def _peer_values(evidence_items: Sequence[EvidenceItem], field: str) -> list[str]:
    """The distinct values proposed by independent, comparably-reliable groups.

    A weak low-score signal (far below the top score) is not a peer and cannot
    create a conflict against a much stronger one.
    """
    scored = _group_field_values_scored(evidence_items, field)
    if len(scored) < 2:
        return list({v for v, _ in scored.values()})
    top = max(score for _, score in scored.values())
    floor = max(_RELIABILITY_FLOOR, top - _PEER_MARGIN)
    peers: list[str] = []
    for value, score in scored.values():
        if score >= floor and value not in peers:
            peers.append(value)
    return peers


def _field_conflict(evidence_items: Sequence[EvidenceItem], *, field: str) -> str:
    """Classify the field conflict as ``none`` / ``ambiguous`` / ``contradiction``.

    Hierarchy-aware for taxonomy fields: ``sword`` + ``bladed_weapon`` is
    *compatible* (``none``), ``sword`` + ``dagger`` is ``ambiguous`` (siblings),
    ``sword`` + ``shield`` is a ``contradiction`` (different subtrees). Flat
    fields treat any two distinct peer values as a contradiction. An explicit
    contradiction code always yields ``contradiction``.
    """
    for item in evidence_items:
        if item.contradiction_codes:
            return "contradiction"

    peers = _peer_values(evidence_items, field)
    if len(peers) < 2:
        return "none"

    if field not in _HIERARCHICAL_FIELDS:
        # Flat field: distinct comparably-reliable values genuinely disagree.
        return "contradiction"

    worst = "none"
    rank = {"none": 0, "ambiguous": 1, "contradiction": 2}
    for i in range(len(peers)):
        for j in range(i + 1, len(peers)):
            relation = taxonomy_relation(peers[i], peers[j])
            if relation in ("agree", "compatible", "unknown"):
                level = "none"
            elif relation == "sibling":
                level = "ambiguous"
            else:  # "contradict"
                level = "contradiction"
            if rank[level] > rank[worst]:
                worst = level
    return worst


def _detect_contradictions(evidence_items: Sequence[EvidenceItem], *, field: str) -> bool:
    """Backward-compatible boolean: True only for a genuine contradiction.

    Hierarchy-compatible broad/specific agreement is not a contradiction.
    """
    return _field_conflict(evidence_items, field=field) == "contradiction"


def _field_value_of(item: EvidenceItem, field: str) -> str | None:
    """Extract the value this evidence item proposes for ``field`` (or None)."""
    value = item.proposed_value
    if isinstance(value, dict):
        if field == "canonical_object":
            candidate = value.get("canonical_object") or value.get("object_name") or value.get(field)
        else:
            candidate = value.get(field)
        if isinstance(candidate, str) and candidate and candidate != "unknown":
            return candidate
        return None
    if isinstance(value, str) and value and value != "unknown":
        return value
    return None


def _group_field_values_scored(evidence_items: Sequence[EvidenceItem], field: str) -> dict[str, tuple[str, float]]:
    """Map each independent dependency group to its representative (value, score).

    Evidence sharing a dependency group (e.g. filename tokens + filename-derived
    values, or a VLM that saw the filename) contributes a *single* representative
    — the highest-scoring value in the group — so correlated agreement is never
    double-counted.
    """
    per_group: dict[str, tuple[str, float]] = {}
    for item in evidence_items:
        val = _field_value_of(item, field)
        if val is None:
            continue
        score = float(item.raw_score or 0.0)
        group = _dependency_group(item)
        current = per_group.get(group)
        if current is None or score > current[1]:
            per_group[group] = (val, score)
    return per_group


_SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "fatal": 3}


def _max_contradiction_severity(evidence_items: Sequence[EvidenceItem]) -> ContradictionSeverity:
    max_sev = ContradictionSeverity.LOW
    max_order = 0
    for item in evidence_items:
        for code in item.contradiction_codes:
            sev = contradiction_severity(code)
            order = _SEVERITY_ORDER.get(sev.value if hasattr(sev, "value") else str(sev), 0)
            if order > max_order:
                max_order = order
                max_sev = sev
    return max_sev


def _consensus_value(evidence_items: Sequence[EvidenceItem], *, field: str) -> Any:
    """Reliability-weighted consensus, **one representative per dependency group**.

    This deliberately does not use a raw ``max()`` or a naive vote count over
    evidence items: correlated items (a variant group, or a VLM that saw the
    filename) collapse to one representative, and each value's weight is the sum
    of the reliabilities of the independent groups supporting it. A weak filename
    echo (0.55) therefore cannot out-tiebreak an authoritative sheet mapping
    (0.96). Ties break deterministically by value name for reproducibility.
    """
    scored = _group_field_values_scored(evidence_items, field)
    if not scored:
        return None
    weight: dict[str, float] = {}
    for value, score in scored.values():
        weight[value] = weight.get(value, 0.0) + score
    return sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _extract_candidates(evidence_items: Sequence[EvidenceItem], *, field: str) -> list[str]:
    """Return candidate values for ``field``, ranked by descending reliability.

    Field-strict: object candidates never pull in category values. Explicit
    ``candidate_object_names`` lists are honoured for the object field.
    """
    scores: dict[str, float] = {}
    for item in evidence_items:
        score = float(item.raw_score or 0.0)
        values: list[str] = []
        primary = _field_value_of(item, field)
        if primary:
            values.append(primary)
        proposed = item.proposed_value
        if field == "canonical_object" and isinstance(proposed, dict):
            extra = proposed.get("candidate_object_names")
            if isinstance(extra, (list, tuple)):
                values.extend(str(v) for v in extra)
        for value in values:
            if value and value != "unknown":
                scores[value] = max(scores.get(value, 0.0), score)
    return [value for value, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _extract_n_best(evidence_items: Sequence[EvidenceItem], *, field: str) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for item in evidence_items:
        raw_score = item.raw_score or 0.0
        value = item.proposed_value
        if isinstance(value, dict):
            value = value.get(field) or value.get("object_name")
        if isinstance(value, str) and value and value != "unknown":
            scores[value] = max(scores.get(value, 0.0), raw_score)
    return sorted(scores.items(), key=lambda x: -x[1])


def build_field_fusion_input(
    field: str,
    evidence_batch: Any,
    *,
    calibration_stratum: str = "global",
    precision_target: float = 0.99,
    policy_hash: str = "",
) -> FieldFusionInput:
    """Build FieldFusionInput from a DeterministicEvidenceBatch or evidence list."""

    if hasattr(evidence_batch, "all_evidence"):
        all_ev = evidence_batch.all_evidence()
    elif isinstance(evidence_batch, (list, tuple)):
        all_ev = tuple(evidence_batch)
    else:
        all_ev = ()

    field_evidence = tuple(item for item in all_ev if field in item.target_fields or not item.target_fields)

    return FieldFusionInput(
        field=field,
        evidence_items=field_evidence,
        calibration_stratum=calibration_stratum,
        precision_target=precision_target,
        policy_hash=policy_hash,
    )
