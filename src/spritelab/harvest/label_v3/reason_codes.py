"""Auto-Labeling v3: structured reason and contradiction codes.

Every non-accepted decision carries a machine-readable reason code.
Contradiction codes capture disagreements between evidence sources.
"""

from __future__ import annotations

from enum import Enum


class ContradictionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    FATAL = "fatal"


class ContradictionAction(str, Enum):
    MASK_FIELD = "mask_field"
    ABSTAIN_FIELD = "abstain_field"
    QUARANTINE_RECORD = "quarantine_record"
    HARD_REJECT_RECORD = "hard_reject_record"


CONTRADICTION_CODES: dict[str, tuple[str, str, ContradictionSeverity, ContradictionAction]] = {
    "cat_source_vs_vlm": (
        "Category mismatch between source profile and VLM",
        "source_profile",
        ContradictionSeverity.HIGH,
        ContradictionAction.ABSTAIN_FIELD,
    ),
    "obj_filename_vs_vlm": (
        "Object name mismatch between filename rules and VLM",
        "filename_rules",
        ContradictionSeverity.MEDIUM,
        ContradictionAction.ABSTAIN_FIELD,
    ),
    "obj_neighbor_vs_vlm": (
        "Object mismatch between nearest-neighbor evidence and VLM",
        "nearest_neighbor",
        ContradictionSeverity.MEDIUM,
        ContradictionAction.ABSTAIN_FIELD,
    ),
    "impossible_combination": (
        "Proposed combination violates taxonomy impossible-combination rules",
        "taxonomy",
        ContradictionSeverity.FATAL,
        ContradictionAction.MASK_FIELD,
    ),
    "source_vs_visual": (
        "Source-hinted label contradicts visual facts",
        "visual_facts",
        ContradictionSeverity.HIGH,
        ContradictionAction.QUARANTINE_RECORD,
    ),
    "sheet_vs_coordinate": (
        "Sheet mapping contradicts coordinate-derived evidence",
        "sheet_mapping",
        ContradictionSeverity.HIGH,
        ContradictionAction.QUARANTINE_RECORD,
    ),
    "pack_outlier": (
        "Sprite-level evidence significantly differs from pack-consistency evidence",
        "pack_consistency",
        ContradictionSeverity.MEDIUM,
        ContradictionAction.ABSTAIN_FIELD,
    ),
    "variant_inconsistent": (
        "Variant group member has inconsistent decisions",
        "variant_group",
        ContradictionSeverity.MEDIUM,
        ContradictionAction.ABSTAIN_FIELD,
    ),
    "color_material_mismatch": (
        "Color and material evidence suggest incompatible attributes",
        "color_palette",
        ContradictionSeverity.LOW,
        ContradictionAction.ABSTAIN_FIELD,
    ),
    "open_set_forced": (
        "Evidence suggests open-set but classification forced an in-distribution label",
        "open_set",
        ContradictionSeverity.HIGH,
        ContradictionAction.QUARANTINE_RECORD,
    ),
    "vlm_hallucination_detected": (
        "VLM output matches known hallucination patterns",
        "vlm",
        ContradictionSeverity.HIGH,
        ContradictionAction.ABSTAIN_FIELD,
    ),
    "correlated_agreement_undiscounted": (
        "Multiple evidence sources show agreement but share dependency",
        "fusion",
        ContradictionSeverity.MEDIUM,
        ContradictionAction.ABSTAIN_FIELD,
    ),
}


REASON_CODES: dict[str, str] = {
    "insufficient_evidence": "Not enough evidence sources with adequate calibration support",
    "conflicting_evidence": "Evidence sources disagree beyond acceptable threshold",
    "open_set_unknown": "Object is likely outside the known taxonomy",
    "novel_class": "Object appears to be a novel class not in taxonomy",
    "ambiguous_identity": "Multiple plausible identities with similar support",
    "impossible_combination": "Proposed combination of category, object, and attributes violates rules",
    "provenance_failure": "Missing or incomplete provenance information",
    "image_integrity_failure": "Image failed integrity checks (alpha, resize, corruption)",
    "missing_required_field": "Required field has no evidence",
    "calibration_insufficient": "Calibration data for this field/stratum is insufficient",
    "below_minimum_information": "Sprite content is below the minimum-information threshold",
    "blank_or_empty": "Sprite has no opaque pixels",
    "tiny_fragment": "Sprite content is too small for reliable labeling",
    "environment_tile_excluded": "Environment tile outside selected dataset scope",
    "malformed_alpha": "Alpha channel is malformed or unsupported",
    "destructive_resize": "Image has been resized with non-nearest-neighbor interpolation",
    "unsupported_domain": "Domain is not in the supported set",
    "uncertain_sheet_mapping": "Sheet coordinate mapping is inconsistent or uncertain",
    "inconsistent_variant_group": "Variant group has inconsistent base identity",
    "irreconcilable_contradiction": "Deterministic evidence sources are irreconcilable",
    "human_flag": "Human operator flagged this decision",
    "policy_rejection": "Decision rejected by explicit policy rule",
    "field_not_applicable": "Field does not apply to this sprite",
}


def contradiction_severity(code: str) -> ContradictionSeverity:
    entry = CONTRADICTION_CODES.get(code)
    return entry[2] if entry else ContradictionSeverity.MEDIUM


def contradiction_action(code: str) -> ContradictionAction:
    entry = CONTRADICTION_CODES.get(code)
    return entry[3] if entry else ContradictionAction.ABSTAIN_FIELD


def reason_description(code: str) -> str:
    return REASON_CODES.get(code, code)
