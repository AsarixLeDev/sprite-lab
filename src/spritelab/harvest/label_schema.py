"""JSON-safe dataclasses for safe harvest label suggestions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from spritelab.harvest.label_taxonomy import normalize_category, normalize_object_name, normalize_tag, normalize_tags


@dataclass(frozen=True)
class LabelSuggestion:
    category: str
    object_name: str
    tags: tuple[str, ...] = ()
    short_description: str = ""
    confidence: float = 0.0
    confidence_reason: str = ""
    source: str = ""
    materials: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    dominant_colors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    source_consistency: str = ""
    alternative_object_names: tuple[str, ...] = ()
    evidence_for_source: tuple[str, ...] = ()
    evidence_against_source: tuple[str, ...] = ()
    candidate_object_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_category(self.category))
        object.__setattr__(self, "object_name", normalize_object_name(self.object_name))
        object.__setattr__(self, "tags", normalize_tags(self.tags))
        object.__setattr__(self, "short_description", str(self.short_description).strip())
        object.__setattr__(self, "confidence", _clamp_confidence(self.confidence))
        object.__setattr__(self, "confidence_reason", str(self.confidence_reason).strip())
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "materials", normalize_tags(self.materials))
        object.__setattr__(self, "mood", normalize_tags(self.mood))
        object.__setattr__(self, "dominant_colors", normalize_tags(self.dominant_colors))
        object.__setattr__(self, "warnings", _clean_strings(self.warnings))
        object.__setattr__(self, "evidence", _clean_strings(self.evidence))
        object.__setattr__(self, "source_consistency", _normalize_source_consistency(self.source_consistency))
        object.__setattr__(
            self, "alternative_object_names", _normalize_objects(self.alternative_object_names, max_items=3)
        )
        object.__setattr__(self, "evidence_for_source", _clean_strings(self.evidence_for_source)[:5])
        object.__setattr__(self, "evidence_against_source", _clean_strings(self.evidence_against_source)[:5])
        object.__setattr__(
            self, "candidate_object_names", _normalize_objects(self.candidate_object_names, max_items=40)
        )


@dataclass(frozen=True)
class SafeFusedLabel:
    safe_prefill: LabelSuggestion
    filename_suggestion: LabelSuggestion | None
    vlm_suggestion: LabelSuggestion | None
    fused_suggestion: LabelSuggestion
    bucket: str
    needs_review: bool
    flags: tuple[str, ...]
    conflict_reasons: tuple[str, ...]
    provenance: dict[str, str | list[str]]
    review_priority: float
    label_confidence_tier: Literal["T0", "T1", "T2", "T3", "T4"] | str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket", str(self.bucket).strip() or "needs_review")
        object.__setattr__(self, "needs_review", bool(self.needs_review))
        object.__setattr__(self, "flags", normalize_tags(self.flags))
        object.__setattr__(self, "conflict_reasons", _clean_strings(self.conflict_reasons))
        object.__setattr__(self, "provenance", _clean_provenance(self.provenance))
        object.__setattr__(self, "review_priority", max(0.0, min(1.0, float(self.review_priority))))
        tier = str(self.label_confidence_tier).strip().upper()
        object.__setattr__(
            self,
            "label_confidence_tier",
            tier if tier in _LABEL_CONFIDENCE_TIERS else confidence_tier_for_bucket(self.bucket),
        )


_LABEL_CONFIDENCE_TIERS = frozenset({"T0", "T1", "T2", "T3", "T4"})


def confidence_tier_for_bucket(bucket: str) -> Literal["T0", "T1", "T2", "T3", "T4"]:
    """Derive additive label trust tiers from existing fusion buckets."""

    normalized = str(bucket).strip()
    if normalized == "auto_filename_trusted":
        return "T0"
    if normalized == "auto_rpg_496_specialized":
        return "T1"
    if normalized in {"auto_prefix_family_trusted", "fused_automatically", "auto_filename_with_vlm_conflict"}:
        return "T2"
    if normalized in {"auto_vlm_candidate_ranked", "auto_vlm_when_filename_weak"}:
        return "T3"
    return "T4"


def label_suggestion_to_json(suggestion: LabelSuggestion | None) -> dict[str, Any] | None:
    if suggestion is None:
        return None
    return {
        "category": suggestion.category,
        "object_name": suggestion.object_name,
        "tags": list(suggestion.tags),
        "short_description": suggestion.short_description,
        "confidence": suggestion.confidence,
        "confidence_reason": suggestion.confidence_reason,
        "source": suggestion.source,
        "materials": list(suggestion.materials),
        "mood": list(suggestion.mood),
        "dominant_colors": list(suggestion.dominant_colors),
        "warnings": list(suggestion.warnings),
        "evidence": list(suggestion.evidence),
        "source_consistency": suggestion.source_consistency,
        "alternative_object_names": list(suggestion.alternative_object_names),
        "evidence_for_source": list(suggestion.evidence_for_source),
        "evidence_against_source": list(suggestion.evidence_against_source),
        "candidate_object_names": list(suggestion.candidate_object_names),
    }


def label_suggestion_from_json(data: Mapping[str, Any] | None) -> LabelSuggestion | None:
    if not isinstance(data, Mapping):
        return None
    description = data.get("short_description") or data.get("visual_description") or data.get("description") or ""
    object_name = data.get("object_name") or data.get("possible_object_name") or data.get("suggested_object_name") or ""
    category = data.get("category") or data.get("possible_category") or "unknown"
    tags = data.get("tags")
    if tags is None:
        tags = data.get("visual_tags")
    evidence = data.get("evidence")
    if evidence is None:
        evidence = data.get("visual_evidence")
    confidence = data.get("confidence")
    if confidence is None:
        confidence = _uncertainty_confidence(str(data.get("uncertainty", "")))
    source_consistency = data.get("source_consistency")
    if source_consistency is None:
        source_consistency = data.get("agrees_with_source", "")
    return LabelSuggestion(
        category=str(category),
        object_name=str(object_name),
        tags=_as_str_tuple(tags),
        short_description=str(description),
        confidence=_clamp_confidence(confidence),
        confidence_reason=str(data.get("confidence_reason", "")),
        source=str(data.get("source", "")),
        materials=_as_str_tuple(data.get("materials")),
        mood=_as_str_tuple(data.get("mood")),
        dominant_colors=_as_str_tuple(data.get("dominant_colors")),
        warnings=_as_str_tuple(data.get("warnings")),
        evidence=_as_str_tuple(evidence),
        source_consistency=str(source_consistency),
        alternative_object_names=_as_str_tuple(data.get("alternative_object_names"))[:3],
        evidence_for_source=_as_str_tuple(data.get("evidence_for_source")),
        evidence_against_source=_as_str_tuple(data.get("evidence_against_source")),
        candidate_object_names=_as_str_tuple(data.get("candidate_object_names")),
    )


def safe_fused_label_to_json(label: SafeFusedLabel) -> dict[str, Any]:
    provenance_fields = _label_provenance_fields(label)
    quality = {
        "bucket": label.bucket,
        "label_confidence_tier": label.label_confidence_tier,
        "needs_review": label.needs_review,
        "flags": list(label.flags),
        "conflict_reasons": list(label.conflict_reasons),
        "provenance": label.provenance,
        "review_priority": label.review_priority,
        **provenance_fields,
    }
    return {
        "safe_prefill": label_suggestion_to_json(label.safe_prefill),
        "filename_suggestion": label_suggestion_to_json(label.filename_suggestion),
        "vlm_suggestion": label_suggestion_to_json(label.vlm_suggestion),
        "fused_suggestion": label_suggestion_to_json(label.fused_suggestion),
        "bucket": label.bucket,
        "label_confidence_tier": label.label_confidence_tier,
        "needs_review": label.needs_review,
        "flags": list(label.flags),
        "conflict_reasons": list(label.conflict_reasons),
        "provenance": label.provenance,
        "review_priority": label.review_priority,
        **provenance_fields,
        "label_quality": quality,
    }


def safe_fused_label_from_json(data: Mapping[str, Any]) -> SafeFusedLabel:
    quality = data.get("label_quality") if isinstance(data.get("label_quality"), Mapping) else {}
    safe_prefill = label_suggestion_from_json(data.get("safe_prefill"))
    fused = label_suggestion_from_json(data.get("fused_suggestion"))
    if safe_prefill is None:
        safe_prefill = fused or LabelSuggestion("unknown", "", source="missing")
    if fused is None:
        fused = safe_prefill
    filename = label_suggestion_from_json(data.get("filename_suggestion"))
    vlm = label_suggestion_from_json(data.get("vlm_suggestion"))
    if vlm is None:
        vlm = label_suggestion_from_json(data.get("vlm_descriptor"))
    if vlm is None:
        vlm = label_suggestion_from_json(data.get("qwen_suggestion"))
    return SafeFusedLabel(
        safe_prefill=safe_prefill,
        filename_suggestion=filename,
        vlm_suggestion=vlm,
        fused_suggestion=fused,
        bucket=str(data.get("bucket") or quality.get("bucket") or "needs_review"),
        needs_review=bool(data.get("needs_review", quality.get("needs_review", True))),
        flags=_as_str_tuple(data.get("flags", quality.get("flags"))),
        conflict_reasons=_as_str_tuple(data.get("conflict_reasons", quality.get("conflict_reasons"))),
        provenance=_mapping_dict(data.get("provenance", quality.get("provenance"))),
        review_priority=_clamp_confidence(data.get("review_priority", quality.get("review_priority", 1.0))),
        label_confidence_tier=str(data.get("label_confidence_tier") or quality.get("label_confidence_tier") or ""),
    )


def _uncertainty_confidence(value: str) -> float:
    normalized = value.strip().lower()
    return {
        "confident": 0.85,
        "likely": 0.65,
        "unsure": 0.4,
        "cannot_tell": 0.2,
    }.get(normalized, 0.0)


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _clean_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _normalize_objects(values: Sequence[str], *, max_items: int) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        object_name = normalize_object_name(str(value))
        if not object_name or object_name in seen:
            continue
        seen.add(object_name)
        result.append(object_name)
        if len(result) >= max_items:
            break
    return tuple(result)


def _normalize_source_consistency(value: Any) -> str:
    normalized = normalize_tag(str(value))
    mapping = {
        "yes": "consistent",
        "true": "consistent",
        "consistent": "consistent",
        "no": "contradicted",
        "false": "contradicted",
        "contradicted": "contradicted",
        "conflict": "contradicted",
        "unclear": "unclear",
        "maybe": "unclear",
        "unknown": "unclear",
        "": "",
        "no_source": "no_source",
        "none": "no_source",
    }
    return mapping.get(normalized, "unclear")


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clean_provenance(value: Mapping[str, Any]) -> dict[str, str | list[str]]:
    result: dict[str, str | list[str]] = {}
    for key, raw in dict(value or {}).items():
        clean_key = str(key).strip()
        if not clean_key:
            continue
        if isinstance(raw, str):
            result[clean_key] = raw
        elif isinstance(raw, Sequence):
            result[clean_key] = [str(item) for item in raw]
        else:
            result[clean_key] = str(raw)
    return result


def _label_provenance_fields(label: SafeFusedLabel) -> dict[str, Any]:
    sources: list[str] = []
    for value in label.provenance.values():
        values = [value] if isinstance(value, str) else value
        for source in values:
            clean = str(source).strip()
            if clean and clean not in sources:
                sources.append(clean)
    conflict_codes = [flag for flag in label.flags if "conflict" in flag or "hallucination" in flag]
    return {
        "label_tier_reason": f"fusion_bucket:{label.bucket}",
        "fusion_bucket": label.bucket,
        "label_sources": sources,
        "label_conflict_codes": conflict_codes,
    }
