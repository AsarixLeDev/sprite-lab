"""Apply label-v2 prediction records back to harvest run metadata."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.dataset_maker.model import normalize_category, normalize_tag
from spritelab.harvest.catalog import append_harvest_event, read_jsonl, write_jsonl

SAFE_AUTO_BUCKETS = {
    "auto_filename_trusted",
    "auto_prefix_family_trusted",
    "auto_vlm_candidate_ranked",
    "auto_rpg_496_specialized",
}

REVIEW_BUCKETS = {
    "needs_review",
    "needs_review_candidate_conflict",
}

LABEL_V2_REVIEW_QUEUE = "label_v2_review_queue.jsonl"
LABEL_V2_APPLY_REPORT_JSON = "label_v2_apply_report.json"
LABEL_V2_APPLY_REPORT_MD = "label_v2_apply_report.md"


def apply_label_v2_predictions(
    run_dir: str | Path,
    *,
    prediction_file: str | Path = "label_v2_suggestions.jsonl",
    mode: str = "auto-only",
    accept_auto: bool = False,
    out_imported: str | Path | None = None,
    out_review: str | Path | None = None,
    dry_run: bool = False,
    overwrite_human_labels: bool = False,
    build_id: str = "",
    require_semantic_v3_for_auto: bool = False,
) -> dict[str, Any]:
    """Apply label-v2 suggestions to ``imported.jsonl`` and write audit outputs."""

    run_path = Path(run_dir)
    imported_path = run_path / "imported.jsonl"
    prediction_path = _resolve_run_path(run_path, prediction_file)
    output_imported_path = _resolve_run_path(run_path, out_imported) if out_imported is not None else imported_path
    output_review_path = _resolve_run_path(run_path, out_review) if out_review is not None else run_path / LABEL_V2_REVIEW_QUEUE

    if mode not in {"auto-only", "all", "review-only"}:
        raise ValueError("mode must be one of: auto-only, all, review-only")
    if not run_path.is_dir():
        raise FileNotFoundError(f"harvest run directory not found: {run_path}")
    if not imported_path.exists():
        raise FileNotFoundError(f"imported.jsonl not found: {imported_path}")
    if not prediction_path.exists():
        raise FileNotFoundError(f"prediction file not found: {prediction_path}")

    imported = read_jsonl(imported_path)
    predictions = read_jsonl(prediction_path)
    predictions_by_id, duplicate_prediction_ids = _predictions_by_sprite_id(predictions)
    imported_ids = {str(record.get("sprite_id", "")) for record in imported if str(record.get("sprite_id", ""))}
    prediction_ids = {str(record.get("sprite_id", "")) for record in predictions if str(record.get("sprite_id", ""))}

    updated: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    auto_skip_reasons: Counter[str] = Counter()
    auto_validation_counts: Counter[str] = Counter()
    matched_imported_ids: set[str] = set()
    applied_ids: list[str] = []
    accepted_ids: list[str] = []
    human_preserved_ids: list[str] = []
    raw_auto_rows_seen = sum(1 for prediction in predictions if is_raw_auto_prediction(prediction))
    for prediction in predictions:
        if not is_raw_auto_prediction(prediction):
            continue
        for reason in _auto_prediction_validation_errors(prediction, require_semantic_v3=True):
            auto_validation_counts[reason] += 1

    for prediction in predictions:
        bucket = prediction_bucket(prediction)
        buckets[bucket] += 1
        safe = _mapping(prediction.get("safe_prefill"))
        categories[str(safe.get("category", "unknown") or "unknown")] += 1
        if is_review_prediction(prediction):
            review_queue.append(build_review_queue_record(prediction, imported_by_id=None))

    imported_by_id = {str(record.get("sprite_id", "")): record for record in imported if str(record.get("sprite_id", ""))}
    if review_queue:
        review_queue = [
            build_review_queue_record(prediction, imported_by_id=imported_by_id)
            for prediction in predictions
            if is_review_prediction(prediction)
        ]

    for record in imported:
        sprite_id = str(record.get("sprite_id", ""))
        prediction = predictions_by_id.get(sprite_id)
        if prediction is None:
            updated.append(dict(record))
            skipped_reasons["missing_prediction"] += 1
            continue
        matched_imported_ids.add(sprite_id)
        bucket = prediction_bucket(prediction)
        selected = should_apply_prediction(prediction, mode=mode)
        if not selected:
            skipped_record = dict(record)
            if is_review_prediction(prediction):
                if accept_auto:
                    review_status = _review_status(record)
                    if review_status != str(record.get("status", "")):
                        counts["review_labels_quarantined"] += 1
                    skipped_record["status"] = review_status
                counts["skipped_review_labels"] += 1
                skipped_reasons["review_bucket"] += 1
            else:
                skipped_reasons["bucket_not_selected"] += 1
                if is_raw_auto_prediction(prediction):
                    auto_skip_reasons["bucket_not_selected"] += 1
            updated.append(skipped_record)
            continue
        if _has_human_label(record) and not overwrite_human_labels:
            updated.append(dict(record))
            counts["human_labels_preserved"] += 1
            human_preserved_ids.append(sprite_id)
            skipped_reasons["human_label_preserved"] += 1
            if is_raw_auto_prediction(prediction):
                auto_skip_reasons["human_label_preserved"] += 1
            continue
        auto_validation_errors = _auto_prediction_validation_errors(
            prediction,
            require_semantic_v3=require_semantic_v3_for_auto,
        )
        if is_safe_auto_prediction(prediction) and auto_validation_errors:
            updated.append(dict(record))
            for reason in auto_validation_errors:
                skipped_reasons[reason] += 1
                auto_skip_reasons[reason] += 1
            continue
        applied = apply_prediction_to_imported_record(
            record,
            prediction,
            prediction_file=prediction_path.name,
            accept_auto=accept_auto and is_safe_auto_prediction(prediction),
            build_id=build_id,
        )
        if is_review_prediction(prediction):
            review_status = _review_status(record)
            if review_status != str(record.get("status", "")):
                counts["review_labels_quarantined"] += 1
            applied["status"] = review_status
        updated.append(applied)
        applied_ids.append(sprite_id)
        if is_safe_auto_prediction(prediction):
            counts["applied_auto_labels"] += 1
            if accept_auto:
                counts["accepted_auto_labels"] += 1
                accepted_ids.append(sprite_id)
        elif is_review_prediction(prediction):
            counts["applied_review_labels"] += 1
        else:
            counts["applied_other_labels"] += 1

    missing_predictions = sorted(imported_ids - prediction_ids)
    missing_imported = sorted(prediction_ids - imported_ids)
    for prediction in predictions:
        if not is_raw_auto_prediction(prediction):
            continue
        if str(prediction.get("sprite_id", "")) in imported_ids:
            continue
        auto_skip_reasons["sprite_id_mismatch"] += 1
        skipped_reasons["sprite_id_mismatch"] += 1
    report: dict[str, Any] = {
        "run_dir": str(run_path),
        "prediction_file": str(prediction_path),
        "mode": mode,
        "dry_run": bool(dry_run),
        "accept_auto": bool(accept_auto),
        "overwrite_human_labels": bool(overwrite_human_labels),
        "build_id": str(build_id),
        "require_semantic_v3_for_auto": bool(require_semantic_v3_for_auto),
        "predictions": len(predictions),
        "imported_sprites": len(imported),
        "raw_auto_rows_seen": int(raw_auto_rows_seen),
        "matched_imported_sprites": len(matched_imported_ids),
        "applied_auto_labels": int(counts["applied_auto_labels"]),
        "applied_review_labels": int(counts["applied_review_labels"]),
        "applied_other_labels": int(counts["applied_other_labels"]),
        "applied_total": len(applied_ids),
        "skipped_review_labels": int(counts["skipped_review_labels"]),
        "missing_predictions": len(missing_predictions),
        "missing_imported_sprites": len(missing_imported),
        "accepted_auto_labels": int(counts["accepted_auto_labels"]),
        "review_labels_quarantined": int(counts["review_labels_quarantined"]),
        "human_labels_preserved": int(counts["human_labels_preserved"]),
        "review_queue_size": len(review_queue),
        "counts_by_bucket": dict(sorted(buckets.items())),
        "counts_by_category": dict(sorted(categories.items())),
        "auto_rows_skipped": max(0, int(raw_auto_rows_seen) - int(counts["applied_auto_labels"])),
        "auto_validation_counts": dict(sorted(auto_validation_counts.items())),
        "auto_skip_reasons": dict(sorted(auto_skip_reasons.items())),
        "skipped_by_reason": dict(sorted(skipped_reasons.items())),
        "missing_prediction_sprite_ids": missing_predictions,
        "missing_imported_sprite_ids": missing_imported,
        "duplicate_prediction_sprite_ids": duplicate_prediction_ids,
        "applied_sprite_ids": applied_ids,
        "accepted_sprite_ids": accepted_ids,
        "human_preserved_sprite_ids": human_preserved_ids,
        "out_imported": str(output_imported_path),
        "out_review": str(output_review_path),
        "out_report_json": str(run_path / LABEL_V2_APPLY_REPORT_JSON),
        "out_report_md": str(run_path / LABEL_V2_APPLY_REPORT_MD),
        "wrote_imported": None,
        "wrote_review_queue": None,
        "wrote_report_json": None,
        "wrote_report_md": None,
    }

    if not dry_run:
        write_jsonl(output_imported_path, updated)
        write_jsonl(output_review_path, review_queue)
        report["wrote_imported"] = str(output_imported_path)
        report["wrote_review_queue"] = str(output_review_path)
        json_path = run_path / LABEL_V2_APPLY_REPORT_JSON
        md_path = run_path / LABEL_V2_APPLY_REPORT_MD
        report["wrote_report_json"] = str(json_path)
        report["wrote_report_md"] = str(md_path)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(format_apply_report_markdown(report), encoding="utf-8")
        append_harvest_event(
            run_path,
            "apply_label_v2",
            {
                "prediction_file": prediction_path.name,
                "mode": mode,
                "applied_total": len(applied_ids),
                "review_queue_size": len(review_queue),
                "accept_auto": bool(accept_auto),
                "build_id": str(build_id),
            },
        )
    return report


def apply_prediction_to_imported_record(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    prediction_file: str,
    accept_auto: bool,
    build_id: str = "",
) -> dict[str, Any]:
    """Return an imported record with one prediction applied and audited."""

    updated = dict(record)
    safe = _mapping(prediction.get("safe_prefill"))
    label_quality = _mapping(prediction.get("label_quality"))
    vlm_descriptor = _mapping(prediction.get("vlm_descriptor") or prediction.get("vlm_suggestion"))
    candidate_object_names = [str(value) for value in prediction.get("candidate_object_names") or () if str(value)]
    flags = [str(value) for value in label_quality.get("flags") or prediction.get("flags") or () if str(value)]
    bucket = prediction_bucket(prediction)

    category = normalize_category(str(safe.get("category", "unknown")))
    object_name = normalize_tag(str(safe.get("object_name", "")))
    tags = _normalized_list(safe.get("tags") or ())
    materials = _normalized_list(safe.get("materials") or ())
    mood = _normalized_list(safe.get("mood") or ())
    short_description = str(safe.get("short_description", "")).strip()

    if category:
        updated["category"] = category
    updated["object_name"] = object_name
    updated["tags"] = tags
    if short_description:
        updated["notes"] = short_description
        updated["short_description"] = short_description
    else:
        updated.setdefault("short_description", "")
    updated["materials"] = materials
    updated["mood"] = mood
    if accept_auto:
        updated["status"] = "accepted"

    auto_metadata = dict(updated.get("auto_metadata") or {}) if isinstance(updated.get("auto_metadata"), Mapping) else {}
    label_v2_metadata = {
        "applied": True,
        "prediction_file": prediction_file,
        "bucket": bucket,
        "flags": flags,
        "safe_prefill": dict(safe),
        "vlm_descriptor": dict(vlm_descriptor),
        "candidate_object_names": candidate_object_names,
    }
    if build_id:
        label_v2_metadata["applied_at_build_id"] = str(build_id)
    auto_metadata.update(
        {
            "label_v2_applied": True,
            "label_v2_prediction_file": prediction_file,
            "label_v2_bucket": bucket,
            "label_v2_flags": flags,
            "label_v2_safe_prefill": dict(safe),
            "label_v2_vlm_descriptor": dict(vlm_descriptor),
            "label_v2_candidate_object_names": candidate_object_names,
            "label_v2": label_v2_metadata,
        }
    )
    if build_id:
        auto_metadata["label_v2_applied_at_build_id"] = str(build_id)
    if label_quality:
        auto_metadata["label_v2_label_quality"] = dict(label_quality)
        label_v2_metadata["label_quality"] = dict(label_quality)
    conflict_reasons = [str(value) for value in label_quality.get("conflict_reasons") or prediction.get("conflict_reasons") or () if str(value)]
    if conflict_reasons:
        auto_metadata["label_v2_conflict_reasons"] = conflict_reasons
    semantic_v3 = _mapping(prediction.get("semantic_v3"))
    if semantic_v3:
        auto_metadata["semantic_v3"] = semantic_v3
    updated["auto_metadata"] = auto_metadata
    return updated


def should_apply_prediction(prediction: Mapping[str, Any], *, mode: str) -> bool:
    if mode == "auto-only":
        return is_safe_auto_prediction(prediction)
    if mode == "review-only":
        return is_review_prediction(prediction)
    if mode == "all":
        return True
    raise ValueError("mode must be one of: auto-only, all, review-only")


def is_safe_auto_prediction(prediction: Mapping[str, Any]) -> bool:
    return prediction_bucket(prediction) in SAFE_AUTO_BUCKETS and not is_review_prediction(prediction)


def is_review_prediction(prediction: Mapping[str, Any]) -> bool:
    bucket = prediction_bucket(prediction)
    quality = _mapping(prediction.get("label_quality"))
    return bool(prediction.get("needs_review", quality.get("needs_review", False))) or bucket in REVIEW_BUCKETS or bucket.startswith("needs_review")


def is_raw_auto_prediction(prediction: Mapping[str, Any]) -> bool:
    return not is_review_prediction(prediction)


def prediction_bucket(prediction: Mapping[str, Any]) -> str:
    quality = _mapping(prediction.get("label_quality"))
    return str(prediction.get("bucket") or quality.get("bucket") or "missing")


def _auto_prediction_validation_errors(
    prediction: Mapping[str, Any],
    *,
    require_semantic_v3: bool,
) -> list[str]:
    errors: list[str] = []
    safe = prediction.get("safe_prefill")
    if not isinstance(safe, Mapping) or not safe:
        errors.append("invalid_safe_prefill")
        safe = {}
    object_name = normalize_tag(str(safe.get("object_name", "")))
    category = normalize_category(str(safe.get("category", "")))
    if not object_name:
        errors.append("missing_object_name")
    if not category or category == "unknown":
        errors.append("missing_category")
    if require_semantic_v3 and not _mapping(prediction.get("semantic_v3")):
        errors.append("missing_semantic_v3")
    return errors


def build_review_queue_record(
    prediction: Mapping[str, Any],
    *,
    imported_by_id: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    sprite_id = str(prediction.get("sprite_id", ""))
    imported = imported_by_id.get(sprite_id, {}) if imported_by_id is not None else {}
    quality = _mapping(prediction.get("label_quality"))
    vlm_descriptor = _mapping(prediction.get("vlm_descriptor") or prediction.get("vlm_suggestion"))
    conflict_reasons = [
        str(value)
        for value in quality.get("conflict_reasons") or prediction.get("conflict_reasons") or ()
        if str(value)
    ]
    flags = [str(value) for value in quality.get("flags") or prediction.get("flags") or () if str(value)]
    reason_parts = [prediction_bucket(prediction), *conflict_reasons]
    return {
        "sprite_id": sprite_id,
        "relative_path": str(imported.get("relative_path") or prediction.get("relative_path") or ""),
        "final_png_path": str(imported.get("final_png_path") or prediction.get("final_png_path") or ""),
        "safe_prefill": dict(_mapping(prediction.get("safe_prefill"))),
        "bucket": prediction_bucket(prediction),
        "flags": flags,
        "label_quality": dict(quality),
        "vlm_descriptor": dict(vlm_descriptor),
        "candidate_object_names": [str(value) for value in prediction.get("candidate_object_names") or () if str(value)],
        "conflict_reasons": conflict_reasons,
        "reason": "; ".join(part for part in reason_parts if part),
        "missing_imported": bool(imported_by_id is not None and sprite_id not in imported_by_id),
    }


def format_apply_summary(report: Mapping[str, Any]) -> str:
    lines = [
        f"Predictions: {int(report.get('predictions', 0))}",
        f"Raw auto rows seen: {int(report.get('raw_auto_rows_seen', 0))}",
        f"Matched imported sprites: {int(report.get('matched_imported_sprites', 0))}",
        f"Applied auto labels: {int(report.get('applied_auto_labels', 0))}",
        f"Auto rows skipped: {int(report.get('auto_rows_skipped', 0))}",
        f"Skipped review labels: {int(report.get('skipped_review_labels', 0))}",
        f"Missing predictions: {int(report.get('missing_predictions', 0))}",
        f"Missing imported sprites: {int(report.get('missing_imported_sprites', 0))}",
        f"Accepted auto labels: {int(report.get('accepted_auto_labels', 0))}",
        f"Human labels preserved: {int(report.get('human_labels_preserved', 0))}",
        f"Wrote imported: {report.get('wrote_imported') or '(dry-run) ' + str(report.get('out_imported', ''))}",
        f"Wrote review queue: {report.get('wrote_review_queue') or '(dry-run) ' + str(report.get('out_review', ''))}",
    ]
    for reason, count in dict(report.get("auto_skip_reasons") or {}).items():
        lines.append(f"Auto skip {reason}: {count}")
    for reason, count in dict(report.get("auto_validation_counts") or {}).items():
        lines.append(f"Auto validation {reason}: {count}")
    if report.get("wrote_report_json"):
        lines.append(f"Wrote report json: {report['wrote_report_json']}")
    if report.get("wrote_report_md"):
        lines.append(f"Wrote report md: {report['wrote_report_md']}")
    return "\n".join(lines) + "\n"


def format_apply_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Label v2 Apply Report",
        "",
        f"Run: `{report.get('run_dir', '')}`",
        f"Prediction file: `{report.get('prediction_file', '')}`",
        f"Mode: `{report.get('mode', '')}`",
        f"Dry run: `{bool(report.get('dry_run', False))}`",
        f"Accept auto: `{bool(report.get('accept_auto', False))}`",
        f"Overwrite human labels: `{bool(report.get('overwrite_human_labels', False))}`",
        f"Require semantic-v3 for auto: `{bool(report.get('require_semantic_v3_for_auto', False))}`",
        "",
        "## Summary",
        "",
        f"- Predictions: {int(report.get('predictions', 0))}",
        f"- Imported sprites: {int(report.get('imported_sprites', 0))}",
        f"- Raw auto rows seen: {int(report.get('raw_auto_rows_seen', 0))}",
        f"- Matched imported sprites: {int(report.get('matched_imported_sprites', 0))}",
        f"- Applied auto labels: {int(report.get('applied_auto_labels', 0))}",
        f"- Auto rows skipped: {int(report.get('auto_rows_skipped', 0))}",
        f"- Applied review labels: {int(report.get('applied_review_labels', 0))}",
        f"- Applied other labels: {int(report.get('applied_other_labels', 0))}",
        f"- Skipped review labels: {int(report.get('skipped_review_labels', 0))}",
        f"- Missing predictions: {int(report.get('missing_predictions', 0))}",
        f"- Missing imported sprites: {int(report.get('missing_imported_sprites', 0))}",
        f"- Accepted auto labels: {int(report.get('accepted_auto_labels', 0))}",
        f"- Review labels quarantined: {int(report.get('review_labels_quarantined', 0))}",
        f"- Human labels preserved: {int(report.get('human_labels_preserved', 0))}",
        f"- Review queue size: {int(report.get('review_queue_size', 0))}",
        "",
        "## Buckets",
    ]
    for bucket, count in dict(report.get("counts_by_bucket") or {}).items():
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "## Categories"])
    for category, count in dict(report.get("counts_by_category") or {}).items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Auto Skips"])
    for reason, count in dict(report.get("auto_skip_reasons") or {}).items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Auto Validation Counts"])
    for reason, count in dict(report.get("auto_validation_counts") or {}).items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Skipped"])
    for reason, count in dict(report.get("skipped_by_reason") or {}).items():
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Missing Predictions"])
    missing_predictions = list(report.get("missing_prediction_sprite_ids") or ())
    lines.append(f"Count: {len(missing_predictions)}")
    for sprite_id in missing_predictions[:50]:
        lines.append(f"- {sprite_id}")
    lines.extend(["", "## Missing Imported Sprites"])
    missing_imported = list(report.get("missing_imported_sprite_ids") or ())
    lines.append(f"Count: {len(missing_imported)}")
    for sprite_id in missing_imported[:50]:
        lines.append(f"- {sprite_id}")
    return "\n".join(lines) + "\n"


def _predictions_by_sprite_id(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for record in records:
        sprite_id = str(record.get("sprite_id", ""))
        if not sprite_id:
            continue
        by_id[sprite_id] = dict(record)
        counts[sprite_id] += 1
    duplicates = sorted(sprite_id for sprite_id, count in counts.items() if count > 1)
    return by_id, duplicates


def _resolve_run_path(run_dir: Path, value: str | Path | None) -> Path:
    if value is None:
        return run_dir
    path = Path(value)
    if path.is_absolute():
        return path
    return run_dir / path


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalized_list(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or ():
        token = normalize_tag(str(value))
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _has_human_label(record: Mapping[str, Any]) -> bool:
    if bool(record.get("human_label") or record.get("human_labeled") or record.get("manually_labeled")):
        return True
    if str(record.get("label_source", "")).strip().lower() in {"human", "manual", "golden", "review_gui"}:
        return True
    if str(record.get("labeler", "")).strip() or str(record.get("labeled_at", "")).strip():
        return True
    auto_metadata = record.get("auto_metadata") if isinstance(record.get("auto_metadata"), Mapping) else {}
    if bool(auto_metadata.get("human_label") or auto_metadata.get("human_labeled") or auto_metadata.get("manually_labeled")):
        return True
    if str(auto_metadata.get("label_source", "")).strip().lower() in {"human", "manual", "golden", "review_gui"}:
        return True
    if str(auto_metadata.get("labeler", "")).strip() or str(auto_metadata.get("labeled_at", "")).strip():
        return True
    return False


def _review_status(record: Mapping[str, Any]) -> str:
    current = str(record.get("status", "")).strip().lower()
    if current and current != "accepted":
        return current
    return "quarantine"
