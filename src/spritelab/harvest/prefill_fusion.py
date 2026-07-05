"""Fuse filename-rule and Qwen metadata suggestions deterministically."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.dataset_maker.model import normalize_category, normalize_tag
from spritelab.harvest.filename_rules import (
    FilenameMetadataSuggestion,
    filename_suggestion_to_dict,
    metadata_suggestions_differ,
)

QUALITY_REQUEST_FAILURE = "request_failure"
QUALITY_INVALID_JSON = "invalid_json"
QUALITY_DEGENERATE = "degenerate"
QUALITY_WARNING_ONLY = "warning_only"
QUALITY_LOW_CONFIDENCE = "low_confidence"
QUALITY_LOW_VOTE_AGREEMENT = "low_vote_agreement"
QUALITY_FILENAME_QWEN_CONFLICT = "filename_qwen_conflict"
QUALITY_FUSED_AUTOMATICALLY = "fused_automatically"
QUALITY_NEEDS_REVIEW = "needs_review"

_AMBIGUOUS_OBJECTS = {
    "",
    "ambiguous",
    "ambiguous_object",
    "ambiguous_shape",
    "unknown",
    "unknown_object",
    "unidentified",
    "unidentified_object",
}
_AMBIGUOUS_TAGS = {"ambiguous", "blurry", "indistinct", "unknown", "unidentified"}


@dataclass(frozen=True)
class FusedPrefillResult:
    fused_suggestion: dict[str, Any]
    prefill_quality: dict[str, Any]


def fuse_prefill_suggestions(
    filename_suggestion: FilenameMetadataSuggestion | Mapping[str, Any],
    qwen_suggestion: Mapping[str, Any] | None,
    *,
    min_qwen_confidence: float = 0.55,
    fusion_policy: str = "weighted",
    adjudication: Mapping[str, Any] | None = None,
) -> FusedPrefillResult:
    """Return deterministic fused metadata plus quality diagnostics.

    ``adjudication`` is the forced-choice verdict from a second model call
    (candidate A = blind Qwen suggestion, candidate B = filename rules). It
    replaces the old model-self-reported ``filename_agreement`` field.
    """

    filename_data = _filename_dict(filename_suggestion)
    qwen_data = dict(qwen_suggestion or {})
    adjudication_data = dict(adjudication or {})
    adjudication_choice = normalize_tag(str(adjudication_data.get("choice", ""))) if adjudication_data else ""
    qwen_warnings = tuple(str(warning) for warning in qwen_data.get("warnings", ()) if str(warning).strip())
    conflict_reasons = metadata_suggestions_differ(_filename_object(filename_data), qwen_data)

    filename_confidence = _float_or_none(filename_data.get("confidence")) or 0.0
    qwen_confidence = _float_or_none(qwen_data.get("confidence"))
    filename_strong = filename_confidence >= 0.9 and bool(filename_data.get("object_name"))
    filename_weak = filename_confidence < 0.8
    qwen_has_content = _suggestion_has_content(qwen_data)
    qwen_degenerate = _has_warning(qwen_warnings, "degenerate_response")
    # A degenerate answer (background/boilerplate) is never trustworthy content.
    qwen_unknown = _qwen_is_unknown_or_ambiguous(qwen_data) or qwen_degenerate
    qwen_low = qwen_confidence is not None and qwen_confidence < min_qwen_confidence
    qwen_warning_only = bool(qwen_warnings) and not qwen_has_content

    flags: list[str] = []
    if qwen_warning_only:
        flags.append(QUALITY_WARNING_ONLY)
    if _has_warning(qwen_warnings, "invalid json"):
        flags.append(QUALITY_INVALID_JSON)
    if _has_warning(qwen_warnings, "degenerate_response"):
        flags.append(QUALITY_DEGENERATE)
    if _has_warning(qwen_warnings, "could not connect") or _has_warning(qwen_warnings, "timed out"):
        flags.append(QUALITY_REQUEST_FAILURE)
    if qwen_low:
        flags.append(QUALITY_LOW_CONFIDENCE)
    vote_stats = qwen_data.get("vote_stats") if isinstance(qwen_data.get("vote_stats"), Mapping) else None
    if vote_stats is not None:
        vote_agreement = _float_or_none(vote_stats.get("category_agreement"))
        if vote_agreement is not None and vote_agreement < 2 / 3:
            flags.append(QUALITY_LOW_VOTE_AGREEMENT)
    if conflict_reasons and qwen_has_content:
        flags.append(QUALITY_FILENAME_QWEN_CONFLICT)

    agreement = _agreement(qwen_data, conflict_reasons, adjudication_choice)
    source = "filename_rules"
    review_required = False

    if not qwen_has_content:
        fused = _base_suggestion(filename_data)
        review_required = not filename_strong
    elif adjudication_choice == "a":
        fused = _merge_suggestions(filename_data, qwen_data, preferred="qwen")
        source = "qwen_adjudicated"
    elif adjudication_choice == "b":
        fused = _merge_suggestions(filename_data, qwen_data, preferred="filename")
        source = "filename_adjudicated"
    elif adjudication_choice in {"both_wrong", "cannot_tell"}:
        fused = _merge_suggestions(filename_data, qwen_data, preferred="filename" if filename_confidence >= (qwen_confidence or 0.0) else "qwen")
        source = "mixed"
        review_required = True
        corrected = _adjudication_correction(adjudication_data)
        if corrected:
            fused.update(corrected)
            source = "adjudicator_corrected"
    elif filename_strong and (qwen_unknown or qwen_warning_only or qwen_low):
        fused = _merge_suggestions(filename_data, qwen_data, preferred="filename")
    elif qwen_has_content and filename_weak:
        fused = _merge_suggestions(filename_data, qwen_data, preferred="qwen")
        source = "qwen"
        review_required = bool(conflict_reasons)
    elif conflict_reasons:
        fused = _merge_suggestions(filename_data, qwen_data, preferred="filename" if filename_confidence >= (qwen_confidence or 0.0) else "qwen")
        source = "mixed"
        review_required = True
    else:
        fused = _merge_suggestions(filename_data, qwen_data, preferred="qwen")
        source = "qwen"

    bucket = _primary_bucket(flags, review_required=review_required)
    if bucket == QUALITY_FUSED_AUTOMATICALLY and review_required:
        bucket = QUALITY_NEEDS_REVIEW

    fused["source"] = source
    fused["fusion_policy"] = fusion_policy
    quality = {
        "bucket": bucket,
        "flags": sorted(set(flags)),
        "agreement": agreement,
        "conflict_reasons": list(conflict_reasons),
        "needs_review": bucket == QUALITY_NEEDS_REVIEW,
        "filename_confidence": filename_confidence,
        "qwen_confidence": qwen_confidence,
        "filename_confidence_reason": filename_data.get("confidence_reason", ""),
        "qwen_warnings": list(qwen_warnings),
        "fusion_policy": fusion_policy,
    }
    if adjudication_data:
        quality["adjudication"] = adjudication_data
    return FusedPrefillResult(fused_suggestion=fused, prefill_quality=quality)


def summarize_prefill_quality(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        quality = record.get("prefill_quality")
        if isinstance(quality, Mapping):
            counts[str(quality.get("bucket") or "unknown")] += 1
            for flag in quality.get("flags") or ():
                counts[str(flag)] += 1
    return dict(sorted(counts.items()))


def _filename_dict(suggestion: FilenameMetadataSuggestion | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(suggestion, FilenameMetadataSuggestion):
        return filename_suggestion_to_dict(suggestion)
    return dict(suggestion)


def _filename_object(data: Mapping[str, Any]) -> FilenameMetadataSuggestion:
    return FilenameMetadataSuggestion(
        category=str(data.get("category", "unknown")),
        object_name=str(data.get("object_name", "")),
        tags=tuple(str(tag) for tag in data.get("tags") or ()),
        materials=tuple(str(value) for value in data.get("materials") or ()),
        mood=tuple(str(value) for value in data.get("mood") or ()),
        short_description=str(data.get("short_description", "")),
        confidence=_float_or_none(data.get("confidence")) or 0.0,
        confidence_reason=str(data.get("confidence_reason", "")),
    )


def _base_suggestion(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": normalize_category(str(data.get("category", "unknown"))),
        "object_name": normalize_tag(str(data.get("object_name", ""))),
        "tags": _merge_tags(data.get("tags") or ()),
        "materials": _merge_tags(data.get("materials") or ()),
        "mood": _merge_tags(data.get("mood") or ()),
        "dominant_colors": _merge_tags(data.get("dominant_colors") or ()),
        "short_description": str(data.get("short_description", "")).strip(),
        "suggested_sprite_id": str(data.get("suggested_sprite_id", "")).strip(),
        "confidence": _float_or_none(data.get("confidence")),
    }


def _merge_suggestions(filename_data: Mapping[str, Any], qwen_data: Mapping[str, Any], *, preferred: str) -> dict[str, Any]:
    primary = qwen_data if preferred == "qwen" else filename_data
    secondary = filename_data if preferred == "qwen" else qwen_data
    base = _base_suggestion(primary)
    fallback = _base_suggestion(secondary)
    category = base["category"] if base["category"] != "unknown" else fallback["category"]
    object_name = base["object_name"] or fallback["object_name"]
    tags = _merge_tags(
        (object_name,),
        base.get("tags") or (),
        fallback.get("tags") or (),
        base.get("materials") or (),
        fallback.get("materials") or (),
        base.get("mood") or (),
        fallback.get("mood") or (),
        base.get("dominant_colors") or (),
        fallback.get("dominant_colors") or (),
    )
    return {
        "category": category,
        "object_name": object_name,
        "tags": tags,
        "materials": _merge_tags(base.get("materials") or (), fallback.get("materials") or ()),
        "mood": _merge_tags(base.get("mood") or (), fallback.get("mood") or ()),
        "dominant_colors": _merge_tags(base.get("dominant_colors") or (), fallback.get("dominant_colors") or ()),
        "short_description": base["short_description"] or fallback["short_description"],
        "suggested_sprite_id": base["suggested_sprite_id"] or fallback["suggested_sprite_id"],
        "confidence": max(base["confidence"] or 0.0, fallback["confidence"] or 0.0),
    }


def _suggestion_has_content(data: Mapping[str, Any]) -> bool:
    return any(
        [
            normalize_category(str(data.get("category", "unknown"))) != "unknown",
            bool(normalize_tag(str(data.get("object_name", "")))),
            bool(data.get("tags")),
            bool(data.get("materials")),
            bool(data.get("mood")),
            bool(data.get("dominant_colors")),
            bool(str(data.get("short_description", "")).strip()),
            data.get("confidence") is not None,
        ]
    )


def _qwen_is_unknown_or_ambiguous(data: Mapping[str, Any]) -> bool:
    object_name = normalize_tag(str(data.get("object_name", "")))
    tags = set(_merge_tags(data.get("tags") or ()))
    category = normalize_category(str(data.get("category", "unknown")))
    warnings = " ".join(str(warning).lower() for warning in data.get("warnings") or ())
    return (
        category == "unknown"
        or object_name in _AMBIGUOUS_OBJECTS
        or bool(tags & _AMBIGUOUS_TAGS)
        or "ambiguous" in warnings
        or "no clear" in warnings
    )


def _adjudication_correction(adjudication_data: Mapping[str, Any]) -> dict[str, Any]:
    """Corrected fields the adjudicator supplied alongside ``both_wrong``."""

    corrected: dict[str, Any] = {}
    category = normalize_category(str(adjudication_data.get("corrected_category", ""))) if str(adjudication_data.get("corrected_category", "")).strip() else ""
    object_name = normalize_tag(str(adjudication_data.get("corrected_object_name", "")))
    if category and category != "unknown":
        corrected["category"] = category
    if object_name:
        corrected["object_name"] = object_name
    return corrected


def _agreement(qwen_data: Mapping[str, Any], conflict_reasons: Sequence[str], adjudication_choice: str) -> str:
    """Agreement is computed in code, never self-reported by the model."""

    if not qwen_data:
        return "missing_qwen"
    if adjudication_choice == "a":
        return "adjudicated_qwen"
    if adjudication_choice == "b":
        return "adjudicated_filename"
    if adjudication_choice in {"both_wrong", "cannot_tell"}:
        return f"adjudicated_{adjudication_choice}"
    return "conflict" if conflict_reasons else "agree"


def _primary_bucket(flags: Sequence[str], *, review_required: bool) -> str:
    if QUALITY_REQUEST_FAILURE in flags:
        return QUALITY_REQUEST_FAILURE
    if QUALITY_INVALID_JSON in flags:
        return QUALITY_INVALID_JSON
    if QUALITY_DEGENERATE in flags:
        return QUALITY_DEGENERATE
    if QUALITY_WARNING_ONLY in flags:
        return QUALITY_WARNING_ONLY
    if review_required:
        return QUALITY_NEEDS_REVIEW
    if QUALITY_LOW_CONFIDENCE in flags:
        return QUALITY_LOW_CONFIDENCE
    if QUALITY_FILENAME_QWEN_CONFLICT in flags:
        return QUALITY_FILENAME_QWEN_CONFLICT
    return QUALITY_FUSED_AUTOMATICALLY


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _has_warning(warnings: Sequence[str], needle: str) -> bool:
    return any(needle in warning.lower() for warning in warnings)


def _merge_tags(*groups: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        if group is None:
            continue
        values = (group,) if isinstance(group, str) else group
        for value in values:
            token = normalize_tag(str(value))
            if token and token not in seen:
                seen.add(token)
                result.append(token)
    return result
