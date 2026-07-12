"""Auto-Labeling v3: backward-compatible adapter.

Exposes accepted v3 field decisions through a legacy-shaped interface without
changing original v2 records. Masked/abstained fields are never claimed as
accepted.
"""

from __future__ import annotations

from typing import Any

from spritelab.harvest.label_v3.record_decisions import RecordDecision


def v3_field_to_legacy_suggestion(
    record: RecordDecision,
    *,
    accepted_only: bool = True,
) -> dict[str, Any]:
    """Build a LabelSuggestion-shaped dict from accepted v3 fields."""

    category = (
        record.category.accepted_value
        if record.category.state == "accepted" and isinstance(record.category.accepted_value, str)
        else "unknown"
    )
    object_name = (
        record.canonical_object.accepted_value
        if record.canonical_object.state == "accepted" and isinstance(record.canonical_object.accepted_value, str)
        else ""
    )
    tags = tuple(str(v) for v in (record.tags.accepted_tags if accepted_only else record.tags.all_tags))
    materials = ()
    if record.material.state == "accepted":
        if isinstance(record.material.accepted_value, str):
            materials = (record.material.accepted_value,)
        elif isinstance(record.material.accepted_value, (tuple, list)):
            materials = tuple(str(v) for v in record.material.accepted_value)

    dominant_colors = ()
    if record.color.state == "accepted":
        if isinstance(record.color.accepted_value, str):
            dominant_colors = (record.color.accepted_value,)
        elif isinstance(record.color.accepted_value, (tuple, list)):
            dominant_colors = tuple(str(v) for v in record.color.accepted_value)

    confidence = record.category.calibrated_estimate or 0.0

    return {
        "category": category,
        "object_name": object_name,
        "tags": list(tags),
        "short_description": "",
        "confidence": confidence,
        "confidence_reason": record.category.decision_reason,
        "source": "label_v3_adapter",
        "materials": list(materials),
        "mood": (),
        "dominant_colors": list(dominant_colors),
        "warnings": list(record.reason_details) if record.record_state != "auto_accept" else [],
        "evidence": list(record.category.evidence_refs),
        "source_consistency": "",
        "alternative_object_names": [],
        "evidence_for_source": [],
        "evidence_against_source": [],
        "candidate_object_names": list(record.canonical_object.n_best_alternatives),
    }


def build_legacy_safe_fused_label(
    v3_record: RecordDecision,
    *,
    include_partial: bool = True,
) -> dict[str, Any]:
    """Build a SafeFusedLabel-shaped dict from a v3 RecordDecision.

    Only exposes accepted or partially-accepted records. Records in quarantine,
    hard_reject, or unknown state are returned with appropriate bucket and
    needs_review flags.
    """

    suggestion = v3_field_to_legacy_suggestion(v3_record, accepted_only=not include_partial)

    if v3_record.record_state == "auto_accept":
        bucket = "auto_v3_trusted"
        needs_review = False
        review_priority = 0.05
    elif v3_record.record_state == "partial_accept":
        bucket = "auto_v3_partial"
        needs_review = False
        review_priority = 0.15
    elif v3_record.record_state == "quarantine":
        bucket = "needs_review_v3_quarantine"
        needs_review = True
        review_priority = 0.8
    elif v3_record.record_state == "hard_reject":
        bucket = "needs_review_v3_hard_reject"
        needs_review = True
        review_priority = 1.0
    else:
        bucket = "needs_review_v3_unknown"
        needs_review = True
        review_priority = 0.9

    return {
        "safe_prefill": suggestion,
        "filename_suggestion": None,
        "vlm_suggestion": None,
        "fused_suggestion": suggestion,
        "bucket": bucket,
        "label_confidence_tier": "T2",
        "needs_review": needs_review,
        "flags": [v3_record.record_state, *v3_record.reason_codes],
        "conflict_reasons": list(v3_record.reason_details),
        "provenance": {
            "object_name": "label_v3",
            "category": "label_v3",
            "tags": ["label_v3"],
            "description": "label_v3_adapter",
        },
        "review_priority": review_priority,
        "label_quality": {
            "bucket": bucket,
            "needs_review": needs_review,
            "flags": [v3_record.record_state],
            "conflict_reasons": list(v3_record.reason_details),
            "provenance": {"source": "label_v3_adapter"},
            "review_priority": review_priority,
        },
    }


def expose_accepted_v3_fields(
    v3_record: RecordDecision,
    *,
    field_mask: set[str] | None = None,
) -> dict[str, Any]:
    """Return only the explicitly accepted fields and their values.

    The ``field_mask`` limits which fields are exposed. If None, all accepted
    fields are exposed. Fields not in the mask or not accepted are omitted.
    """
    result: dict[str, Any] = {
        "sprite_id": v3_record.sprite_id,
        "record_state": v3_record.record_state,
    }

    allowed = field_mask or {
        "domain",
        "category",
        "canonical_object",
        "surface_alias",
        "color",
        "material",
        "shape",
        "role",
        "tags",
        "description",
    }

    field_map = {
        "domain": v3_record.domain,
        "category": v3_record.category,
        "canonical_object": v3_record.canonical_object,
        "surface_alias": v3_record.surface_alias,
        "color": v3_record.color,
        "material": v3_record.material,
        "shape": v3_record.shape,
        "role": v3_record.role,
        "description": v3_record.description,
    }

    for field_name, decision in field_map.items():
        if field_name not in allowed:
            continue
        if decision.state != "accepted":
            continue
        result[field_name] = decision.accepted_value
        result[f"{field_name}_calibrated_estimate"] = decision.calibrated_estimate
        result[f"{field_name}_evidence_refs"] = list(decision.evidence_refs)

    if "tags" in allowed and v3_record.tags.accepted_tags:
        result["tags"] = list(v3_record.tags.accepted_tags)

    return result
