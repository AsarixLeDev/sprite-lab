"""Evaluation and threshold sweep helpers for label-v2 suggestions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.golden import GoldenLabel
from spritelab.harvest.label_taxonomy import (
    normalize_category,
    normalize_object_name,
    normalize_tags,
    object_name_token_f1,
    split_object_tokens,
)

AUTO_BUCKETS = {
    "auto_filename_trusted",
    "auto_filename_with_vlm_conflict",
    "auto_prefix_family_trusted",
    "fused_automatically",
    "auto_vlm_when_filename_weak",
    "auto_vlm_candidate_ranked",
}


def load_label_v2_predictions(
    runs: Sequence[str | Path],
    *,
    prediction_file: str = "label_v2_suggestions.jsonl",
) -> list[dict[str, Any]]:
    """Load label-v2 prediction records from one or more run directories."""

    records: list[dict[str, Any]] = []
    for run in runs:
        path = Path(run) / prediction_file
        for record in read_jsonl(path):
            copied = dict(record)
            copied.setdefault("run_dir", str(run))
            records.append(copied)
    return records


def evaluate_label_v2(
    golden: Mapping[str, GoldenLabel],
    prediction_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score safe_prefill predictions against golden labels by sprite_id."""

    by_id = {str(record.get("sprite_id", "")): record for record in prediction_records if record.get("sprite_id")}
    matched_ids = sorted(set(golden) & set(by_id))
    missing_ids = sorted(set(golden) - set(by_id))

    pairs = [(golden[sprite_id], _suggestion(by_id[sprite_id])) for sprite_id in matched_ids]
    auto_pairs = [
        (golden[sprite_id], _suggestion(by_id[sprite_id]))
        for sprite_id in matched_ids
        if _is_auto_record(by_id[sprite_id])
    ]

    buckets: Counter[str] = Counter()
    bucket_pairs: dict[str, list[tuple[GoldenLabel, Mapping[str, Any]]]] = defaultdict(list)
    per_source_pairs: dict[str, list[tuple[GoldenLabel, Mapping[str, Any]]]] = defaultdict(list)
    per_category_pairs: dict[str, list[tuple[GoldenLabel, Mapping[str, Any]]]] = defaultdict(list)
    conflict_pairs: Counter[str] = Counter()
    object_mismatch_patterns: Counter[str] = Counter()
    near_miss_compound_errors: Counter[str] = Counter()
    broad_to_specific_error_patterns: Counter[str] = Counter()
    over_specific_error_patterns: Counter[str] = Counter()
    category_mismatch_pairs: Counter[str] = Counter()
    error_classes: Counter[str] = Counter()
    review_bucket_object_correct_count = 0
    review_bucket_count = 0
    for sprite_id in matched_ids:
        record = by_id[sprite_id]
        label = golden[sprite_id]
        suggestion = _suggestion(record)
        bucket = _bucket(record)
        buckets[bucket] += 1
        bucket_pairs[bucket].append((label, suggestion))
        per_source_pairs[str(record.get("source_id") or record.get("source_name") or "unknown")].append(
            (label, suggestion)
        )
        per_category_pairs[label.category].append((label, suggestion))
        pred_category = normalize_category(str(suggestion.get("category", "unknown")))
        pred_object = normalize_object_name(str(suggestion.get("object_name", "")))
        gold_object = normalize_object_name(label.object_name)
        if label.category != pred_category:
            category_mismatch_pairs[f"{label.category}->{pred_category}"] += 1
        pred_tags = normalize_tags(suggestion.get("tags") or ())
        _, _, tag_f1 = _set_prf(set(label.tags), set(pred_tags))
        object_exact = bool(gold_object and gold_object == pred_object)
        category_match = label.category == pred_category
        if bucket.startswith("needs_review"):
            review_bucket_count += 1
            if category_match and object_exact:
                review_bucket_object_correct_count += 1
        if not category_match or not object_exact or tag_f1 < 0.999:
            error_class = _error_class(
                gold_category=label.category,
                pred_category=pred_category,
                gold_object=gold_object,
                pred_object=pred_object,
                tag_f1=tag_f1,
                bucket=bucket,
            )
            error_classes[error_class] += 1
        if gold_object and gold_object != pred_object:
            pattern = f"{pred_object}->{gold_object}"
            object_mismatch_patterns[pattern] += 1
            pred_tokens = set(split_object_tokens(pred_object))
            gold_tokens = set(split_object_tokens(gold_object))
            token_f1 = object_name_token_f1(gold_object, pred_object)
            broad_to_specific = _is_broad_to_specific(pred_object, gold_object)
            if broad_to_specific:
                broad_to_specific_error_patterns[pattern] += 1
            if _is_over_specific_prediction(pred_object, gold_object):
                over_specific_error_patterns[pattern] += 1
            if broad_to_specific or (token_f1 > 0.0 and (len(pred_tokens) > 1 or len(gold_tokens) > 1)):
                near_miss_compound_errors[pattern] += 1
        filename = _object_from(record.get("filename_suggestion"))
        vlm = _object_from(
            record.get("vlm_suggestion") or record.get("vlm_descriptor") or record.get("qwen_suggestion")
        )
        if filename and vlm and filename != vlm:
            conflict_pairs[f"{filename}->{vlm}"] += 1

    matched = len(matched_ids)
    field_metrics = _field_metrics(pairs)
    auto_metrics = _field_metrics(auto_pairs)
    auto_count = len(auto_pairs)
    auto_precision = auto_metrics["category_accuracy"]
    bucket_metrics = {key: _field_metrics(value) for key, value in sorted(bucket_pairs.items())}
    review_count = sum(
        1
        for sprite_id in matched_ids
        if bool(_quality(by_id[sprite_id]).get("needs_review", False))
        or _bucket(by_id[sprite_id]).startswith("needs_review")
    )
    tag_only_errors = (
        error_classes.get("tag_only_mismatch", 0)
        + error_classes.get("exact_match_tag_gap", 0)
        + error_classes.get("review_bucket_match", 0)
    )
    category_errors = error_classes.get("category_mismatch", 0)
    broad_to_specific_errors = error_classes.get("broad_to_specific_miss", 0)
    over_specific_errors = error_classes.get("over_specific_prediction", 0)
    hard_object_errors = (
        category_errors + error_classes.get("object_mismatch", 0) + broad_to_specific_errors + over_specific_errors
    )
    return {
        "golden_count": len(golden),
        "prediction_count": len(by_id),
        "matched_count": matched,
        "missing_prediction_count": len(missing_ids),
        "missing_prediction_ids": missing_ids,
        "category_accuracy": field_metrics["category_accuracy"],
        "object_exact_accuracy": field_metrics["object_exact_accuracy"],
        "object_token_f1": field_metrics["object_token_f1"],
        "tag_precision": field_metrics["tag_precision"],
        "tag_recall": field_metrics["tag_recall"],
        "tag_f1": field_metrics["tag_f1"],
        "auto_coverage": auto_count / len(golden) if golden else 0.0,
        "auto_precision": auto_precision,
        "auto_object_token_f1": auto_metrics["object_token_f1"],
        "auto_count": auto_count,
        "review_rate": review_count / matched if matched else 0.0,
        "buckets": dict(sorted(buckets.items())),
        "confusion_matrix": field_metrics["confusion_matrix"],
        "top_conflict_pairs": dict(conflict_pairs.most_common(20)),
        "top_object_mismatch_patterns": dict(object_mismatch_patterns.most_common(20)),
        "near_miss_compound_errors": dict(near_miss_compound_errors.most_common(20)),
        "broad_to_specific_errors": broad_to_specific_errors,
        "broad_to_specific_error_patterns": dict(broad_to_specific_error_patterns.most_common(20)),
        "over_specific_errors": over_specific_errors,
        "over_specific_error_patterns": dict(over_specific_error_patterns.most_common(20)),
        "error_classes": dict(error_classes.most_common()),
        "hard_object_errors": hard_object_errors,
        "tag_only_errors": tag_only_errors,
        "category_errors": category_errors,
        "review_bucket_object_correct_count": review_bucket_object_correct_count,
        "review_bucket_object_correct_rate": review_bucket_object_correct_count / review_bucket_count
        if review_bucket_count
        else 0.0,
        "category_mismatch_pairs": dict(category_mismatch_pairs.most_common(20)),
        "specificity_gap_rate": broad_to_specific_errors / matched if matched else 0.0,
        "per_source": {key: _field_metrics(value) for key, value in sorted(per_source_pairs.items())},
        "per_category": {key: _field_metrics(value) for key, value in sorted(per_category_pairs.items())},
        "bucket_metrics": bucket_metrics,
        "object_exact_accuracy_by_bucket": {
            key: value["object_exact_accuracy"] for key, value in bucket_metrics.items()
        },
        "category_accuracy_by_bucket": {key: value["category_accuracy"] for key, value in bucket_metrics.items()},
        "headline": {
            "auto_coverage_at_auto_precision_0_95": auto_count / len(golden)
            if golden and auto_precision >= 0.95
            else 0.0,
            "auto_precision_target": 0.95,
        },
    }


def label_v2_error_records(
    golden: Mapping[str, GoldenLabel],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    errors_mode: str = "all",
) -> list[dict[str, Any]]:
    """Return per-record mismatch diagnostics for fast label-v2 iteration."""

    by_id = {str(record.get("sprite_id", "")): record for record in prediction_records if record.get("sprite_id")}
    errors: list[dict[str, Any]] = []
    for sprite_id in sorted(golden):
        label = golden[sprite_id]
        record = by_id.get(sprite_id)
        if record is None:
            error_class = "category_mismatch"
            if _include_error_class(error_class, errors_mode):
                errors.append(
                    {
                        "sprite_id": sprite_id,
                        "source_id": "",
                        "golden": _golden_json(label),
                        "predicted": {},
                        "bucket": "missing",
                        "flags": [],
                        "object_exact_match": False,
                        "object_token_f1": 0.0,
                        "tag_f1": 0.0,
                        "category_mismatch": True,
                        "error_class": error_class,
                        "reason": "missing_prediction",
                        "vlm_possible_object": "",
                        "vlm_alternative_object_names": [],
                    }
                )
            continue
        suggestion = _suggestion(record)
        pred_category = normalize_category(str(suggestion.get("category", "unknown")))
        pred_object = normalize_object_name(str(suggestion.get("object_name", "")))
        pred_tags = normalize_tags(suggestion.get("tags") or ())
        object_f1 = object_name_token_f1(label.object_name, pred_object)
        object_exact = bool(label.object_name and normalize_object_name(label.object_name) == pred_object)
        _, _, tag_f1 = _set_prf(set(label.tags), set(pred_tags))
        category_mismatch = label.category != pred_category
        if not category_mismatch and object_exact and tag_f1 >= 0.999:
            continue
        vlm = _vlm_mapping(record)
        error_class = _error_class(
            gold_category=label.category,
            pred_category=pred_category,
            gold_object=normalize_object_name(label.object_name),
            pred_object=pred_object,
            tag_f1=tag_f1,
            bucket=_bucket(record),
        )
        if not _include_error_class(error_class, errors_mode):
            continue
        reasons: list[str] = []
        if category_mismatch:
            reasons.append(f"category:{label.category}->{pred_category}")
        if not object_exact:
            reasons.append(f"object:{label.object_name}->{pred_object}")
        if tag_f1 < 0.999:
            reasons.append(f"tag_f1:{tag_f1:.3f}")
        errors.append(
            {
                "sprite_id": sprite_id,
                "source_id": str(record.get("source_id", "")),
                "golden": _golden_json(label),
                "predicted": {"category": pred_category, "object_name": pred_object, "tags": list(pred_tags)},
                "bucket": _bucket(record),
                "flags": _flags(record),
                "object_exact_match": object_exact,
                "object_token_f1": object_f1,
                "tag_f1": tag_f1,
                "category_mismatch": category_mismatch,
                "error_class": error_class,
                "reason": "; ".join(reasons) or "mismatch",
                "vlm_possible_object": _object_from(vlm),
                "vlm_alternative_object_names": list(normalize_tags(vlm.get("alternative_object_names") or ()))
                if isinstance(vlm, Mapping)
                else [],
            }
        )
    return errors


def sweep_label_v2_operating_points(
    golden: Mapping[str, GoldenLabel],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    trusted_filename_thresholds: Sequence[float] = (0.75, 0.8, 0.85, 0.9, 0.95),
    vlm_thresholds: Sequence[float] = (0.65, 0.75, 0.85),
    conflict_policies: Sequence[str] = ("auto_trusted_filename_conflicts", "review_conflicts"),
    precision_target: float = 0.95,
) -> dict[str, Any]:
    """Evaluate simple threshold operating points over existing v2 records."""

    points = []
    for trusted_threshold in trusted_filename_thresholds:
        for vlm_threshold in vlm_thresholds:
            for conflict_policy in conflict_policies:
                filtered = [
                    _record_with_threshold_bucket(
                        record,
                        trusted_filename_threshold=trusted_threshold,
                        vlm_threshold=vlm_threshold,
                        conflict_policy=conflict_policy,
                    )
                    for record in prediction_records
                ]
                result = evaluate_label_v2(golden, filtered)
                point = {
                    "trusted_filename_threshold": trusted_threshold,
                    "vlm_threshold": vlm_threshold,
                    "conflict_policy": conflict_policy,
                    "auto_coverage": result["auto_coverage"],
                    "auto_precision": result["auto_precision"],
                    "category_accuracy": result["category_accuracy"],
                    "object_token_f1": result["object_token_f1"],
                    "review_rate": result["review_rate"],
                }
                points.append(point)

    satisfying = [
        point
        for point in points
        if point["auto_precision"] >= precision_target and point["category_accuracy"] >= precision_target
    ]
    best = (
        max(
            satisfying or points,
            key=lambda point: (
                point["auto_coverage"],
                point["object_token_f1"],
                point["auto_precision"],
                -point["review_rate"],
            ),
        )
        if points
        else None
    )
    return {
        "precision_target": precision_target,
        "points": points,
        "best": best,
    }


def format_label_v2_report(result: Mapping[str, Any]) -> str:
    lines = [
        f"Golden labels: {result.get('golden_count', 0)}",
        f"Predictions: {result.get('prediction_count', 0)}",
        f"Matched: {result.get('matched_count', 0)}",
        f"Missing predictions: {result.get('missing_prediction_count', 0)}",
        "",
        f"Category accuracy: {float(result.get('category_accuracy', 0.0)):.3f}",
        f"Object exact accuracy: {float(result.get('object_exact_accuracy', 0.0)):.3f}",
        f"Object token-F1: {float(result.get('object_token_f1', 0.0)):.3f}",
        f"Tag F1: {float(result.get('tag_f1', 0.0)):.3f}",
        f"Auto coverage: {float(result.get('auto_coverage', 0.0)):.3f}",
        f"Auto precision: {float(result.get('auto_precision', 0.0)):.3f}",
        f"Review rate: {float(result.get('review_rate', 0.0)):.3f}",
        "",
        "Buckets:",
    ]
    for bucket, count in dict(result.get("buckets") or {}).items():
        lines.append(f"- {bucket}: {count}")
    classes = dict(result.get("error_classes") or {})
    if classes:
        lines.extend(
            [
                "",
                "Error classes:",
                f"- hard_object_errors: {int(result.get('hard_object_errors', 0))}",
                f"- tag_only_errors: {int(result.get('tag_only_errors', 0))}",
                f"- category_errors: {int(result.get('category_errors', 0))}",
                f"- broad_to_specific_errors: {int(result.get('broad_to_specific_errors', 0))}",
                f"- over_specific_errors: {int(result.get('over_specific_errors', 0))}",
            ]
        )
        for error_class, count in classes.items():
            lines.append(f"- {error_class}: {count}")
    if result.get("review_bucket_object_correct_count") is not None:
        lines.extend(
            [
                "",
                "Review bucket:",
                f"- object_correct_count: {int(result.get('review_bucket_object_correct_count', 0))}",
                f"- object_correct_rate: {float(result.get('review_bucket_object_correct_rate', 0.0)):.3f}",
            ]
        )
    conflicts = dict(result.get("top_conflict_pairs") or {})
    if conflicts:
        lines.extend(["", "Top conflict pairs:"])
        for pair, count in conflicts.items():
            lines.append(f"- {pair}: {count}")
    broad = dict(result.get("broad_to_specific_error_patterns") or {})
    if broad:
        lines.extend(["", "Broad-to-specific errors:"])
        for pair, count in broad.items():
            lines.append(f"- {pair}: {count}")
    over_specific = dict(result.get("over_specific_error_patterns") or {})
    if over_specific:
        lines.extend(["", "Over-specific predictions:"])
        for pair, count in over_specific.items():
            lines.append(f"- {pair}: {count}")
    mismatches = dict(result.get("top_object_mismatch_patterns") or {})
    if mismatches:
        lines.extend(["", "Top object mismatch patterns:"])
        for pair, count in mismatches.items():
            lines.append(f"- {pair}: {count}")
    category_pairs = dict(result.get("category_mismatch_pairs") or {})
    if category_pairs:
        lines.extend(["", "Category mismatch pairs:"])
        for pair, count in category_pairs.items():
            lines.append(f"- {pair}: {count}")
    return "\n".join(lines)


def _record_with_threshold_bucket(
    record: Mapping[str, Any],
    *,
    trusted_filename_threshold: float,
    vlm_threshold: float,
    conflict_policy: str,
) -> dict[str, Any]:
    copied = dict(record)
    quality = dict(_quality(record))
    bucket = _bucket(record)
    safe = _suggestion(record)
    filename = record.get("filename_suggestion") if isinstance(record.get("filename_suggestion"), Mapping) else {}
    vlm = record.get("vlm_suggestion") or record.get("vlm_descriptor") or record.get("qwen_suggestion") or {}
    filename_conf = _confidence(filename)
    vlm_conf = _confidence(vlm)
    flags = {str(flag) for flag in quality.get("flags") or record.get("flags") or ()}
    if "vlm_conflicts_with_filename" in flags and conflict_policy == "review_conflicts":
        bucket = "needs_review"
        quality["needs_review"] = True
    elif bucket.startswith("auto_filename") and filename_conf < trusted_filename_threshold:
        bucket = "needs_review"
        quality["needs_review"] = True
    elif bucket == "auto_vlm_when_filename_weak" and vlm_conf < vlm_threshold:
        bucket = "needs_review"
        quality["needs_review"] = True
    elif safe.get("confidence") is not None and _confidence(safe) < min(trusted_filename_threshold, vlm_threshold):
        bucket = "needs_review"
        quality["needs_review"] = True
    quality["bucket"] = bucket
    copied["label_quality"] = quality
    copied["bucket"] = bucket
    copied["needs_review"] = bool(quality.get("needs_review", bucket == "needs_review"))
    return copied


def _error_class(
    *,
    gold_category: str,
    pred_category: str,
    gold_object: str,
    pred_object: str,
    tag_f1: float,
    bucket: str,
) -> str:
    if gold_category != pred_category:
        return "category_mismatch"
    if gold_object != pred_object:
        if _is_broad_to_specific(pred_object, gold_object):
            return "broad_to_specific_miss"
        if _is_over_specific_prediction(pred_object, gold_object):
            return "over_specific_prediction"
        return "object_mismatch"
    if bucket.startswith("needs_review"):
        return "review_bucket_match"
    if tag_f1 < 0.999:
        return "tag_only_mismatch"
    return "exact_match_tag_gap"


def _include_error_class(error_class: str, errors_mode: str) -> bool:
    mode = str(errors_mode or "all").strip().lower()
    if mode == "all":
        return True
    tag_classes = {"tag_only_mismatch", "exact_match_tag_gap", "review_bucket_match"}
    object_classes = {"object_mismatch", "broad_to_specific_miss", "over_specific_prediction"}
    if mode == "hard":
        return error_class not in tag_classes
    if mode == "object":
        return error_class in object_classes
    if mode == "tag":
        return error_class in tag_classes
    return True


def _is_broad_to_specific(pred_object: str, gold_object: str) -> bool:
    pred_tokens = set(split_object_tokens(pred_object))
    gold_tokens = set(split_object_tokens(gold_object))
    if pred_tokens and pred_tokens < gold_tokens:
        return True
    if len(pred_tokens) != 1 or not gold_tokens:
        return False
    broad = next(iter(pred_tokens))
    family_specifics = {
        "armor": {"chestplate", "breastplate", "cuirass"},
        "clothing": {"tunic", "robe", "shirt"},
        "ring": {"ring"},
        "bones": {"bone", "bones", "rib", "chest"},
        "bone": {"bone", "bones", "rib", "chest"},
        "metal": {"metal", "coin", "shield", "ore", "ingot"},
        "wood": {"wood", "wooden", "shield"},
        "fish": {"fish", "skewer"},
        "pie": {"pie", "slice"},
        "meat": {"meat", "raw"},
        "watermelon": {"watermelon", "slice"},
        "medicine": {"medicine", "bottle", "vial"},
        "bow": {"arrow", "arrows", "bow"},
        "dagger": {"dagger"},
        "sword": {"sword"},
        "fire": {"fire", "flame", "spiral", "fireball"},
        "water": {"water", "wave"},
        "potion": {"potion", "vial", "bottle"},
    }
    if broad in {"red", "blue", "green", "yellow", "pink", "white", "orange", "purple"}:
        return bool(gold_tokens & {"potion", "vial", "bottle"})
    expected = family_specifics.get(broad)
    return bool(expected and gold_tokens & expected)


def _is_over_specific_prediction(pred_object: str, gold_object: str) -> bool:
    pred_tokens = set(split_object_tokens(pred_object))
    gold_tokens = set(split_object_tokens(gold_object))
    if gold_tokens and gold_tokens < pred_tokens:
        return True
    if len(gold_tokens) != 1 or not pred_tokens:
        return False
    broad = next(iter(gold_tokens))
    family_specifics = {
        "armor": {"chestplate", "breastplate", "cuirass"},
        "chestplate": {"chestplate", "golden", "leather"},
        "hat": {"hat", "wizard", "cap", "hood"},
        "bow": {"bow", "arrow", "arrows", "explosive", "electric", "ice", "poison", "silver"},
        "potion": {"potion", "vial", "bottle"},
        "fire": {"fire", "flame", "spiral", "fireball"},
        "water": {"water", "wave"},
        "buff": {"buff", "speed", "strength"},
    }
    expected = family_specifics.get(broad)
    return bool(expected and pred_tokens & expected)


def _field_metrics(pairs: Sequence[tuple[GoldenLabel, Mapping[str, Any]]]) -> dict[str, Any]:
    if not pairs:
        return {
            "count": 0,
            "category_accuracy": 0.0,
            "object_exact_accuracy": 0.0,
            "object_token_f1": 0.0,
            "tag_precision": 0.0,
            "tag_recall": 0.0,
            "tag_f1": 0.0,
            "confusion_matrix": {},
        }

    category_hits = 0
    object_hits = 0
    token_f1_total = 0.0
    tag_precision_total = 0.0
    tag_recall_total = 0.0
    tag_f1_total = 0.0
    confusion: Counter[str] = Counter()
    for label, suggestion in pairs:
        gold_category = label.category
        pred_category = normalize_category(str(suggestion.get("category", "unknown")))
        if gold_category == pred_category:
            category_hits += 1
        else:
            confusion[f"{gold_category}->{pred_category}"] += 1

        gold_object = normalize_object_name(label.object_name)
        pred_object = normalize_object_name(str(suggestion.get("object_name", "")))
        if gold_object and gold_object == pred_object:
            object_hits += 1
        token_f1_total += object_name_token_f1(gold_object, pred_object)

        precision, recall, f1 = _set_prf(set(label.tags), set(normalize_tags(suggestion.get("tags") or ())))
        tag_precision_total += precision
        tag_recall_total += recall
        tag_f1_total += f1

    count = len(pairs)
    return {
        "count": count,
        "category_accuracy": category_hits / count,
        "object_exact_accuracy": object_hits / count,
        "object_token_f1": token_f1_total / count,
        "tag_precision": tag_precision_total / count,
        "tag_recall": tag_recall_total / count,
        "tag_f1": tag_f1_total / count,
        "confusion_matrix": dict(confusion.most_common()),
    }


def _auto_precision(pairs: Sequence[tuple[GoldenLabel, Mapping[str, Any]]]) -> float:
    if not pairs:
        return 0.0
    hits = 0
    for label, suggestion in pairs:
        category_hit = label.category == normalize_category(str(suggestion.get("category", "unknown")))
        object_hit = object_name_token_f1(label.object_name, str(suggestion.get("object_name", ""))) >= 0.8
        if category_hit and object_hit:
            hits += 1
    return hits / len(pairs)


def _suggestion(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("safe_prefill") or record.get("fused_suggestion") or {}
    return value if isinstance(value, Mapping) else {}


def _quality(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("label_quality") or {}
    return value if isinstance(value, Mapping) else {}


def _bucket(record: Mapping[str, Any]) -> str:
    quality = _quality(record)
    return str(record.get("bucket") or quality.get("bucket") or "missing")


def _is_auto_record(record: Mapping[str, Any]) -> bool:
    quality = _quality(record)
    bucket = _bucket(record)
    if bool(record.get("needs_review", quality.get("needs_review", False))):
        return False
    return bucket in AUTO_BUCKETS or bucket.startswith("auto_")


def _object_from(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    return normalize_object_name(str(value.get("object_name") or value.get("possible_object_name") or ""))


def _vlm_mapping(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("vlm_suggestion") or record.get("vlm_descriptor") or record.get("qwen_suggestion") or {}
    return value if isinstance(value, Mapping) else {}


def _flags(record: Mapping[str, Any]) -> list[str]:
    quality = _quality(record)
    return [str(flag) for flag in (record.get("flags") or quality.get("flags") or ())]


def _golden_json(label: GoldenLabel) -> dict[str, Any]:
    return {
        "category": label.category,
        "object_name": normalize_object_name(label.object_name),
        "tags": list(label.tags),
    }


def _confidence(value: Any) -> float:
    if not isinstance(value, Mapping):
        return 0.0
    try:
        return max(0.0, min(1.0, float(value.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _set_prf(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    overlap = len(gold & pred)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
