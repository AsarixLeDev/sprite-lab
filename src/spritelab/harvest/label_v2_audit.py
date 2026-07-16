"""Read-only upstream audit summaries for Labeling v2 runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.label_schema import confidence_tier_for_bucket


def summarize_label_v2_upstream(
    run_dir: str | Path,
    *,
    prediction_file: str = "label_v2_suggestions.jsonl",
) -> dict[str, Any]:
    """Summarize all imported statuses without mutating a harvest run.

    The compact record list makes accepted, quarantine, review, and rejected
    rows auditable without exporting raw visual facts or training artifacts.
    """

    run_path = Path(run_dir)
    imported = read_jsonl(run_path / "imported.jsonl")
    predictions = read_jsonl(run_path / prediction_file)
    predictions_by_id = {str(row.get("sprite_id", "")): row for row in predictions if row.get("sprite_id")}
    statuses: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    audit_codes: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for imported_row in imported:
        sprite_id = str(imported_row.get("sprite_id", ""))
        prediction = predictions_by_id.get(sprite_id, {})
        quality = prediction.get("label_quality") if isinstance(prediction.get("label_quality"), Mapping) else {}
        profile = prediction.get("source_profile") if isinstance(prediction.get("source_profile"), Mapping) else {}
        bucket = str(quality.get("bucket") or prediction.get("bucket") or "")
        tier = str(quality.get("label_confidence_tier") or prediction.get("label_confidence_tier") or "")
        if not tier and bucket:
            tier = confidence_tier_for_bucket(bucket)
        status = _audit_status(imported_row, quality)
        reason = _review_or_rejection_reason(imported_row, quality)
        codes = [str(code) for code in quality.get("audit_codes") or () if str(code)]
        statuses[status] += 1
        profiles[str(profile.get("name") or "unknown")] += 1
        tiers[tier or "unknown"] += 1
        buckets[bucket or "missing"] += 1
        if reason:
            reasons[reason] += 1
        audit_codes.update(codes)
        records.append(
            {
                "sprite_id": sprite_id,
                "status": status,
                "source_profile": str(profile.get("name") or "unknown"),
                "label_confidence_tier": tier or "unknown",
                "fusion_bucket": bucket or "missing",
                "review_or_rejection_reason": reason,
                "audit_codes": codes,
            }
        )

    return {
        "run_dir": str(run_path),
        "prediction_file": str(prediction_file),
        "record_count": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "source_profile_counts": dict(sorted(profiles.items())),
        "confidence_tier_counts": dict(sorted(tiers.items())),
        "fusion_bucket_counts": dict(sorted(buckets.items())),
        "review_or_rejection_reason_counts": dict(sorted(reasons.items())),
        "audit_code_histogram": dict(sorted(audit_codes.items())),
        "records": records,
    }


def _audit_status(record: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    status = str(record.get("status", "")).strip().lower()
    if status == "rejected":
        return "rejected"
    if status in {"quarantine", "needs_fix"}:
        return "quarantine"
    if status in {"review", "needs_review"} or bool(quality.get("needs_review")):
        return "review"
    if status == "accepted":
        return "accepted"
    return status or "unknown"


def _review_or_rejection_reason(record: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    reasons = [str(value) for value in quality.get("conflict_reasons") or () if str(value)]
    if reasons:
        return reasons[0]
    errors = record.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        for value in errors:
            if str(value):
                return str(value)
    return str(record.get("rejection_reason") or record.get("review_reason") or "")
