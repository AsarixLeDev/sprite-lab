"""Prefill accuracy evaluation against a human-labeled golden set."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from spritelab.dataset_maker.model import normalize_category, normalize_tag
from spritelab.harvest.golden import GoldenLabel

CALIBRATION_BIN_EDGES = (0.0, 0.2, 0.4, 0.55, 0.7, 0.85, 1.0)


def evaluate_prefill(
    golden: Mapping[str, GoldenLabel],
    fused_records: Sequence[Mapping[str, Any]],
    *,
    degenerate_check: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Score qwen and fused suggestions against golden labels.

    ``fused_records`` are ``fused_suggestions.jsonl`` rows (sprite_id,
    qwen_suggestion, fused_suggestion, prefill_quality). Only sprites present
    in both the golden set and the records are scored.
    """

    records_by_id = {
        str(record.get("sprite_id", "")): record
        for record in fused_records
        if record.get("sprite_id")
    }
    matched_ids = sorted(set(golden) & set(records_by_id))

    qwen_pairs = []
    fused_pairs = []
    buckets: Counter[str] = Counter()
    auto_bucket_pairs = []
    degenerate_count = 0
    for sprite_id in matched_ids:
        record = records_by_id[sprite_id]
        label = golden[sprite_id]
        qwen = _as_dict(record.get("qwen_suggestion"))
        fused = _as_dict(record.get("fused_suggestion"))
        quality = _as_dict(record.get("prefill_quality"))
        bucket = str(quality.get("bucket") or "missing")
        buckets[bucket] += 1
        qwen_pairs.append((label, qwen))
        fused_pairs.append((label, fused))
        if bucket == "fused_automatically":
            auto_bucket_pairs.append((label, fused))
        if _is_degenerate(qwen, degenerate_check):
            degenerate_count += 1

    matched = len(matched_ids)
    auto_metrics = _field_metrics(auto_bucket_pairs)
    return {
        "golden_count": len(golden),
        "record_count": len(records_by_id),
        "matched_count": matched,
        "qwen": _field_metrics(qwen_pairs),
        "fused": _field_metrics(fused_pairs),
        "buckets": dict(sorted(buckets.items())),
        "auto_coverage": buckets.get("fused_automatically", 0) / matched if matched else 0.0,
        "auto_precision": auto_metrics["category_accuracy"],
        "auto_count": len(auto_bucket_pairs),
        "review_rate": buckets.get("needs_review", 0) / matched if matched else 0.0,
        "degenerate_rate": degenerate_count / matched if matched else 0.0,
        "calibration": _calibration(qwen_pairs),
    }


def _field_metrics(pairs: Sequence[tuple[GoldenLabel, Mapping[str, Any]]]) -> dict[str, Any]:
    if not pairs:
        return {
            "count": 0,
            "category_accuracy": 0.0,
            "category_macro_f1": 0.0,
            "category_confusion": {},
            "object_name_exact": 0.0,
            "object_name_token_f1": 0.0,
            "tags_precision": 0.0,
            "tags_recall": 0.0,
            "tags_f1": 0.0,
        }

    category_hits = 0
    confusion: Counter[str] = Counter()
    per_category_tp: Counter[str] = Counter()
    per_category_gold: Counter[str] = Counter()
    per_category_pred: Counter[str] = Counter()
    object_exact = 0
    object_token_f1_total = 0.0
    tags_precision_total = 0.0
    tags_recall_total = 0.0
    tags_f1_total = 0.0

    for label, suggestion in pairs:
        gold_category = label.category
        pred_category = normalize_category(str(suggestion.get("category", "unknown")))
        per_category_gold[gold_category] += 1
        per_category_pred[pred_category] += 1
        if gold_category == pred_category:
            category_hits += 1
            per_category_tp[gold_category] += 1
        else:
            confusion[f"{gold_category}->{pred_category}"] += 1

        gold_object = label.object_name
        pred_object = normalize_tag(str(suggestion.get("object_name", "")))
        if gold_object and gold_object == pred_object:
            object_exact += 1
        object_token_f1_total += _token_f1(gold_object, pred_object)

        gold_tags = set(label.tags)
        pred_tags = {normalize_tag(str(tag)) for tag in suggestion.get("tags") or ()}
        pred_tags.discard("")
        precision, recall, f1 = _set_prf(gold_tags, pred_tags)
        tags_precision_total += precision
        tags_recall_total += recall
        tags_f1_total += f1

    count = len(pairs)
    categories = sorted(set(per_category_gold) | set(per_category_pred))
    f1_values = []
    for category in categories:
        tp = per_category_tp[category]
        precision = tp / per_category_pred[category] if per_category_pred[category] else 0.0
        recall = tp / per_category_gold[category] if per_category_gold[category] else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    return {
        "count": count,
        "category_accuracy": category_hits / count,
        "category_macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "category_confusion": dict(sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0]))),
        "object_name_exact": object_exact / count,
        "object_name_token_f1": object_token_f1_total / count,
        "tags_precision": tags_precision_total / count,
        "tags_recall": tags_recall_total / count,
        "tags_f1": tags_f1_total / count,
    }


def _calibration(pairs: Sequence[tuple[GoldenLabel, Mapping[str, Any]]]) -> dict[str, Any]:
    """Bin qwen confidence against empirical category accuracy; report ECE."""

    bins: list[dict[str, Any]] = []
    edges = CALIBRATION_BIN_EDGES
    counted = 0
    ece_total = 0.0
    for low, high in zip(edges, edges[1:]):
        members = []
        for label, suggestion in pairs:
            confidence = _confidence(suggestion)
            if confidence is None:
                continue
            if low <= confidence < high or (high == edges[-1] and confidence == high):
                members.append((label, suggestion, confidence))
        if not members:
            bins.append({"bin": f"{low:.2f}-{high:.2f}", "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        hits = sum(
            1
            for label, suggestion, _ in members
            if label.category == normalize_category(str(suggestion.get("category", "unknown")))
        )
        mean_confidence = sum(confidence for _, _, confidence in members) / len(members)
        accuracy = hits / len(members)
        counted += len(members)
        ece_total += len(members) * abs(mean_confidence - accuracy)
        bins.append(
            {
                "bin": f"{low:.2f}-{high:.2f}",
                "count": len(members),
                "mean_confidence": round(mean_confidence, 4),
                "accuracy": round(accuracy, 4),
            }
        )
    return {
        "bins": bins,
        "ece": round(ece_total / counted, 4) if counted else None,
        "scored_count": counted,
    }


def format_eval_report(result: Mapping[str, Any]) -> str:
    """Return a short human-readable evaluation summary."""

    lines = [
        f"Golden labels: {result['golden_count']}",
        f"Matched sprites: {result['matched_count']}",
        "",
        "field                      qwen     fused",
    ]
    qwen = result["qwen"]
    fused = result["fused"]
    for label, key in (
        ("category accuracy", "category_accuracy"),
        ("category macro-F1", "category_macro_f1"),
        ("object_name exact", "object_name_exact"),
        ("object_name token-F1", "object_name_token_f1"),
        ("tags precision", "tags_precision"),
        ("tags recall", "tags_recall"),
        ("tags F1", "tags_f1"),
    ):
        lines.append(f"{label:<25} {qwen[key]:>7.3f}  {fused[key]:>7.3f}")
    lines += [
        "",
        f"auto coverage: {result['auto_coverage']:.3f} ({result['auto_count']} sprites)",
        f"auto precision (category): {result['auto_precision']:.3f}",
        f"review rate: {result['review_rate']:.3f}",
        f"degenerate rate: {result['degenerate_rate']:.3f}",
    ]
    ece = result.get("calibration", {}).get("ece")
    if ece is not None:
        lines.append(f"confidence ECE: {ece:.4f}")
    lines += ["", "buckets:"]
    for bucket, count in result.get("buckets", {}).items():
        lines.append(f"- {bucket}: {count}")
    return "\n".join(lines)


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = {token for token in gold.split("_") if token}
    pred_tokens = {token for token in pred.split("_") if token}
    _, _, f1 = _set_prf(gold_tokens, pred_tokens)
    return f1


def _set_prf(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    overlap = len(gold & pred)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _confidence(suggestion: Mapping[str, Any]) -> float | None:
    value = suggestion.get("confidence")
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _is_degenerate(
    suggestion: Mapping[str, Any],
    degenerate_check: Callable[[Mapping[str, Any]], bool] | None,
) -> bool:
    if degenerate_check is not None:
        return bool(degenerate_check(suggestion))
    warnings = suggestion.get("warnings") or ()
    return any("degenerate" in str(warning).lower() for warning in warnings)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
