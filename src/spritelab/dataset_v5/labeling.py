"""Per-field reconciliation and supervision for blind Sol labels."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from spritelab.dataset_v5.taint import detect_filename_taint, reconcile_metadata_taint

RECONCILIATION_VERSION = "raw_v5_sol_reconciliation_v1"
SUPERVISION_POLICY_VERSION = "raw_v5_supervision_v1"
SUPERVISION_CLASSES = ("supervised_strong", "supervised_weak", "auxiliary_only", "unlabeled")
SEMANTIC_FIELDS = (
    "category",
    "canonical_object",
    "domain",
    "role",
    "visual_form",
    "material_applicability",
    "explicit_material",
    "color_roles",
    "description",
)
CRITICAL_FIELDS = frozenset({"category", "canonical_object", "domain", "role"})


def reconcile_sol_passes(
    adjudication: Mapping[str, Any],
    consistency: Mapping[str, Any],
    *,
    deterministic_facts: Mapping[str, Any],
    local_proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconcile same-model consistency evidence without overstating it."""

    fields: dict[str, dict[str, Any]] = {}
    critical_conflicts: list[str] = []
    for field in SEMANTIC_FIELDS:
        first = copy.deepcopy(adjudication.get(field))
        second = copy.deepcopy(consistency.get(field))
        first_abstained = field in (adjudication.get("abstentions") or {})
        second_abstained = field in (consistency.get("abstentions") or {})
        if first_abstained or second_abstained:
            state = "abstained"
            value = None
            supervision = "unlabeled"
            agreement = first_abstained and second_abstained and first == second
            reason = "sol_abstention_preserved"
        elif first != second:
            state = "conflicted"
            value = None
            supervision = "auxiliary_only"
            agreement = False
            reason = "same_model_consistency_disagreement"
            if field in CRITICAL_FIELDS:
                critical_conflicts.append(field)
        else:
            value = first
            agreement = True
            state = _value_state(field, value)
            supervision = "supervised_weak" if state == "known" else "unlabeled"
            reason = "sol_same_model_consistency_agreement"
        if field == "explicit_material" and value is not None:
            supported = _material_visually_supported(adjudication, consistency)
            if not supported:
                value = None
                state = "unsupported_removed"
                supervision = "auxiliary_only"
                agreement = False
                reason = "unsupported_exact_material_removed"
        fields[field] = {
            "agreement": agreement,
            "adjudication_value": first,
            "consistency_value": second,
            "negative_target": False,
            "reason": reason,
            "state": state,
            "supervision_class": supervision,
            "target_mask": 1 if supervision in {"supervised_strong", "supervised_weak"} else 0,
            "value": value,
        }
    deterministic = {
        field: {
            "negative_target": False,
            "state": "known",
            "supervision_class": "supervised_strong",
            "target_mask": 1,
            "value": copy.deepcopy(value),
        }
        for field, value in sorted(deterministic_facts.items())
    }
    local_agreement = _local_agreement(local_proposal, fields)
    return {
        "critical_conflicts": critical_conflicts,
        "deterministic_fields": deterministic,
        "fields": fields,
        "inclusion_decision": "quarantine" if critical_conflicts else "candidate",
        "local_proposal": copy.deepcopy(local_proposal),
        "local_sol_agreement": local_agreement,
        "local_sol_agreement_is_independent_verification": False,
        "reconciliation_version": RECONCILIATION_VERSION,
        "same_model_two_pass_is_independent_verification": False,
        "supervision_policy_version": SUPERVISION_POLICY_VERSION,
    }


def reconcile_source_metadata(
    blind_reconciliation: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    *,
    blind_labels_frozen: bool,
) -> dict[str, Any]:
    """Load source evidence only after blind artifacts have been frozen."""

    if not blind_labels_frozen:
        raise ValueError("source metadata cannot be introduced before blind labeling is frozen")
    original_filename = str(
        source_metadata.get("original_source_filename") or source_metadata.get("original_archive_member") or ""
    )
    taint = detect_filename_taint(original_filename)
    blind_values = {
        field: details.get("value")
        for field, details in blind_reconciliation.get("fields", {}).items()
        if isinstance(details, Mapping)
    }
    taint_result = reconcile_metadata_taint(taint, blind_values)
    declared = source_metadata.get("declared_semantics")
    conflicts: list[dict[str, Any]] = []
    if isinstance(declared, Mapping):
        for field, source_value in declared.items():
            blind_value = blind_values.get(str(field))
            if blind_value not in (None, "unknown", "oov") and source_value != blind_value:
                conflicts.append(
                    {
                        "blind_value": copy.deepcopy(blind_value),
                        "field": str(field),
                        "source_value": copy.deepcopy(source_value),
                    }
                )
    return {
        "blind_label_unchanged": True,
        "filename_taint": taint,
        "metadata_conflicts": conflicts,
        "source_metadata_role": "post_blind_provenance_and_conflict_evidence",
        **taint_result,
    }


def supervision_policy() -> dict[str, Any]:
    return {
        "classes": list(SUPERVISION_CLASSES),
        "missing": {"negative_target": False, "supervision_class": "unlabeled"},
        "not_applicable": {"field_excluded": True, "supervision_class": "unlabeled"},
        "oov": {"distinct_state": True, "negative_target": False, "supervision_class": "unlabeled"},
        "policy_version": SUPERVISION_POLICY_VERSION,
        "rules": {
            "conflicted_field": "auxiliary_only",
            "deterministic_exact_field": "supervised_strong",
            "local_sol_agreement": "never_supervised_strong",
            "sol_semantic_uncalibrated": "supervised_weak",
            "sol_two_pass_agreement": "consistency_only_never_supervised_strong",
        },
        "unknown": {"distinct_state": True, "negative_target": False, "supervision_class": "unlabeled"},
    }


def _value_state(field: str, value: Any) -> str:
    if field == "material_applicability" and value == "not_applicable":
        return "not_applicable"
    if isinstance(value, str) and value.casefold() == "unknown":
        return "unknown"
    if isinstance(value, str) and value.casefold() == "oov":
        return "oov"
    if value in (None, "", [], {}):
        return "missing"
    return "known"


def _material_visually_supported(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if first.get("material_applicability") != "applicable" or second.get("material_applicability") != "applicable":
        return False
    required = "visually_justified_exact_material"
    first_signals = set((first.get("field_risk_signals") or {}).get("explicit_material") or ())
    second_signals = set((second.get("field_risk_signals") or {}).get("explicit_material") or ())
    first_rationale = str((first.get("field_rationales") or {}).get("explicit_material") or "").strip()
    second_rationale = str((second.get("field_rationales") or {}).get("explicit_material") or "").strip()
    return required in first_signals and required in second_signals and bool(first_rationale and second_rationale)


def _local_agreement(
    local: Mapping[str, Any] | None,
    reconciled: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool] | None:
    if local is None:
        return None
    return {field: local.get(field) == details.get("value") for field, details in reconciled.items() if field in local}
