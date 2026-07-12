"""Auto-Labeling v3: evaluation and promotion-gate checker.

Per-field precision, coverage, calibration metrics, and confidence intervals.
Operates on frozen golden suites — never a tuning set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spritelab.harvest.catalog import read_jsonl
from spritelab.harvest.golden import GoldenLabel
from spritelab.harvest.label_taxonomy import normalize_object_name, object_name_token_f1
from spritelab.harvest.label_v3.calibration import (
    compute_ece,
    compute_lower_confidence_bound,
)
from spritelab.harvest.label_v3.record_decisions import RecordDecision, record_decision_from_json

FIELDS: tuple[str, ...] = (
    "domain",
    "category",
    "canonical_object",
    "color",
    "material",
    "shape",
    "role",
)

# Fields the golden set (GoldenLabel: category, object_name, tags) can actually
# score. Everything else has no ground truth and must not be scored.
GOLDEN_EVALUABLE_FIELDS: frozenset[str] = frozenset({"category", "canonical_object"})

GOLDEN_TO_V3_FIELD_MAP: dict[str, str] = {
    "category": "category",
    "object_name": "canonical_object",
    "tags": "tags",
    "color": "color",
    "material": "material",
}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return normalize_object_name(str(value))


def _safe_ratio(numerator: int, denominator: int) -> float:
    """A ratio that is always in [0, 1] (0 when the denominator is 0)."""
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


class CalibrationEvalOverlapError(ValueError):
    """Raised when calibration and evaluation sample IDs overlap."""


def assert_calibration_eval_disjoint(
    calibration_ids: Iterable[str],
    evaluation_ids: Iterable[str],
) -> None:
    """Hard QA gate: calibration and evaluation samples must never overlap.

    Reused calibration examples in a promotion suite make the evaluation
    self-referential — a hard stop condition. Raises with the overlapping ids.
    """
    overlap = sorted({str(x) for x in calibration_ids} & {str(x) for x in evaluation_ids})
    if overlap:
        raise CalibrationEvalOverlapError(
            f"{len(overlap)} sprite id(s) appear in BOTH calibration and evaluation: "
            f"{overlap[:10]}{' …' if len(overlap) > 10 else ''}"
        )


@dataclass
class V3EvalResult:
    suite_name: str = "unnamed"
    total_golden: int = 0
    matched: int = 0
    per_field: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall: dict[str, Any] = field(default_factory=dict)
    acceptance_band: dict[str, Any] = field(default_factory=dict)
    hierarchy_accuracy: dict[str, Any] = field(default_factory=dict)
    open_set_metrics: dict[str, Any] = field(default_factory=dict)
    hard_reject_fpr: float = 0.0
    provenance_completeness: float = 0.0
    masked_field_leakage: float = 0.0
    variant_consistency: float | None = None
    per_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_domain: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_profile: dict[str, dict[str, Any]] = field(default_factory=dict)
    promotion_gates: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def evaluate_v3_against_golden(
    golden: Mapping[str, GoldenLabel],
    v3_records: Mapping[str, RecordDecision],
    *,
    suite_name: str = "unnamed",
    precision_target_category: float = 0.99,
    precision_target_canonical_object: float = 0.99,
    precision_target_color: float = 0.95,
    precision_target_material: float = 0.95,
    precision_target_shape: float = 0.90,
    confidence_level: float = 0.95,
    source_info: Mapping[str, dict[str, str]] | None = None,
) -> V3EvalResult:
    """Evaluate v3 record decisions against golden labels."""

    result = V3EvalResult(suite_name=suite_name)
    common_ids = sorted(set(golden) & set(v3_records))
    result.total_golden = len(golden)
    result.matched = len(common_ids)

    if not common_ids:
        result.warnings.append("no_matching_sprite_ids")
        return result

    # Per-field stats
    field_stats: dict[str, dict[str, list]] = {}
    for field_name in FIELDS:
        field_stats[field_name] = {
            "accepted_correct": [],
            "accepted_total": [],
            "accepted_probs": [],
            "accepted_labels": [],
            "all_correct": [],
            "all_total": [],
        }

    # Hierarchy tracking
    hierarchy_correct: list[bool] = []
    hierarchy_depth_deltas: list[int] = []

    # Per-source tracking
    per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sprite_id in common_ids:
        golden_label = golden[sprite_id]
        v3_record = v3_records[sprite_id]
        src = str(source_info.get(sprite_id, {}).get("source", "unknown")) if source_info else "unknown"
        domain = str(source_info.get(sprite_id, {}).get("domain", "unknown")) if source_info else "unknown"
        profile = str(source_info.get(sprite_id, {}).get("profile", "unknown")) if source_info else "unknown"

        row = _eval_one(golden_label, v3_record)
        per_source[src].append(row)
        row["domain"] = domain
        row["profile"] = profile

        for field_name in FIELDS:
            # Only score fields for which the golden set actually carries a
            # ground truth. GoldenLabel provides category + object_name (+tags);
            # domain/color/material/shape/role have no golden truth and must NOT
            # be scored against an unrelated attribute (e.g. tags).
            if field_name not in GOLDEN_EVALUABLE_FIELDS:
                continue
            stats = field_stats[field_name]
            key = "object_name" if field_name == "canonical_object" else field_name
            golden_val = getattr(golden_label, key, None)

            v3_val = None
            v3_fd = getattr(v3_record, field_name, None)
            if v3_fd and v3_fd.state == "accepted":
                v3_val = v3_fd.accepted_value
                is_correct = _field_match(field_name, golden_val, v3_val)
                stats["accepted_correct"].append(1 if is_correct else 0)
                stats["accepted_total"].append(1)
                prob = v3_fd.calibrated_estimate or 0.0
                stats["accepted_probs"].append(prob)
                stats["accepted_labels"].append(1 if is_correct else 0)

            if v3_fd:
                stats["all_correct"].append(1 if v3_fd.state != "rejected" else 0)
                stats["all_total"].append(1)

        # Hierarchy depth
        if hasattr(golden_label, "object_name") and golden_label.object_name:
            hn = v3_record.canonical_object.hierarchy_node
            gn = _safe_str(golden_label.object_name)
            exact_match = hn == gn
            hierarchy_correct.append(exact_match or _hierarchy_ancestor_match(hn, gn))
            hierarchy_depth_deltas.append(0 if exact_match else 1)

    # Compute per-field metrics
    field_targets = {
        "domain": precision_target_category,
        "category": precision_target_category,
        "canonical_object": precision_target_canonical_object,
        "color": precision_target_color,
        "material": precision_target_material,
        "shape": precision_target_shape,
    }

    for field_name in FIELDS:
        # Fields with no golden ground truth are explicitly not_scored rather
        # than counted as right or wrong.
        if field_name not in GOLDEN_EVALUABLE_FIELDS:
            result.per_field[field_name] = {
                "scored": False,
                "reason": "no_golden_truth_for_field",
                "accepted_total": 0,
                "accepted_correct": 0,
                "meets_target": False,
            }
            continue

        stats = field_stats[field_name]
        accepted_correct = sum(stats["accepted_correct"])
        accepted_total = len(stats["accepted_correct"])
        total_correct = sum(stats["all_correct"])
        total = len(stats["all_total"])

        # Ratios with explicit numerator/denominator, all bounded to [0, 1].
        precision = _safe_ratio(accepted_correct, accepted_total)
        coverage = _safe_ratio(accepted_total, total)
        selective_risk = 1.0 - precision if accepted_total > 0 else 0.0
        abstained = total - accepted_total
        ci_lower = compute_lower_confidence_bound(accepted_correct, accepted_total, confidence_level)
        ece = compute_ece(stats["accepted_probs"], stats["accepted_labels"]) if stats["accepted_probs"] else 0.0

        target = field_targets.get(field_name, 0.95)
        meets_target = ci_lower >= target if accepted_total > 0 else False

        result.per_field[field_name] = {
            "scored": True,
            "accepted_total": accepted_total,
            "accepted_correct": accepted_correct,
            "eligible": total,
            "abstained": abstained,
            "precision": round(precision, 4),
            "precision_numerator": accepted_correct,
            "precision_denominator": accepted_total,
            "coverage": round(coverage, 4),
            "coverage_numerator": accepted_total,
            "coverage_denominator": total,
            "selective_risk": round(selective_risk, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(compute_upper_bound(precision, accepted_total, confidence_level), 4),
            "ece": round(ece, 4),
            "precision_target": target,
            "meets_target": meets_target,
            "total_evaluated": total,
            "total_correct": total_correct,
        }

    # Overall
    all_precisions = [m["precision"] for m in result.per_field.values() if m["accepted_total"] > 0]
    result.overall = {
        "matched": result.matched,
        "total_golden": result.total_golden,
        "average_precision": sum(all_precisions) / max(1, len(all_precisions)),
    }

    # Acceptance band
    acc_correct = sum(sum(stats["accepted_correct"]) for stats in field_stats.values())
    acc_total = sum(len(stats["accepted_correct"]) for stats in field_stats.values())
    result.acceptance_band = {
        "accepted_total_decisions": acc_total,
        "accepted_correct_decisions": acc_correct,
        "band_precision": acc_correct / max(1, acc_total),
        "band_ci_lower": compute_lower_confidence_bound(acc_correct, acc_total, confidence_level),
    }

    # Hierarchy
    result.hierarchy_accuracy = {
        "exact_or_ancestor_correct": sum(hierarchy_correct) / max(1, len(hierarchy_correct)),
        "mean_depth_delta": sum(hierarchy_depth_deltas) / max(1, len(hierarchy_depth_deltas)),
        "total_evaluated": len(hierarchy_correct),
    }

    # Hard-reject FPR
    rejected_ids = [sid for sid in common_ids if v3_records[sid].record_state == "hard_reject"]
    result.hard_reject_fpr = len(rejected_ids) / max(1, len(common_ids))

    # Provenance completeness: of every *accepted* field decision across all
    # records, what fraction carries evidence references + a policy hash. This
    # must be 1.0 for promotion (every accepted field is reproducible).
    field_map_names = ("domain", "category", "canonical_object", "color", "material", "shape", "role", "description")
    accepted_field_count = 0
    accepted_fields_with_provenance = 0
    for record in v3_records.values():
        for name in field_map_names:
            fd = getattr(record, name, None)
            if fd is None or fd.state != "accepted":
                continue
            accepted_field_count += 1
            if fd.evidence_refs and fd.policy_hash:
                accepted_fields_with_provenance += 1
    result.provenance_completeness = (
        accepted_fields_with_provenance / accepted_field_count if accepted_field_count > 0 else 1.0
    )

    # Promotion gates
    result.promotion_gates = {
        "category_meets_target": result.per_field.get("category", {}).get("meets_target", False),
        "canonical_object_meets_target": result.per_field.get("canonical_object", {}).get("meets_target", False),
        "color_meets_target": result.per_field.get("color", {}).get("meets_target", False)
        if result.per_field.get("color", {}).get("accepted_total", 0) > 0
        else True,
        "material_meets_target": result.per_field.get("material", {}).get("meets_target", False)
        if result.per_field.get("material", {}).get("accepted_total", 0) > 0
        else True,
        "hard_reject_zero_fpr": result.hard_reject_fpr == 0.0,
        "provenance_complete": result.provenance_completeness >= 0.99,
        "ece_acceptable": all(
            result.per_field.get(f, {}).get("ece", 1.0) <= 0.05 for f in ("category", "canonical_object")
        ),
    }

    return result


def selective_risk_by_stratum(
    golden: Mapping[str, GoldenLabel],
    v3_records: Mapping[str, RecordDecision],
    *,
    field: str = "category",
    stratify_by: str = "source",
    source_info: Mapping[str, dict[str, str]] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Per-stratum selective-risk breakdown for one field.

    ``stratify_by`` ∈ {source, profile, domain, category, in_domain}. Each
    stratum reports accepted / eligible / correct counts, precision, coverage,
    selective risk, abstention count, and a one-sided lower confidence bound —
    every ratio with explicit numerator/denominator and bounded to [0, 1].
    """
    if field not in GOLDEN_EVALUABLE_FIELDS:
        return {"_not_scored": {"reason": "no_golden_truth_for_field", "field": field}}

    golden_key = "object_name" if field == "canonical_object" else field
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible": 0, "accepted": 0, "correct": 0})

    for sprite_id in sorted(set(golden) & set(v3_records)):
        info = (source_info or {}).get(sprite_id, {})
        if stratify_by == "category":
            stratum = str(getattr(golden[sprite_id], "category", "unknown") or "unknown")
        else:
            stratum = str(info.get(stratify_by, "unknown"))
        fd = getattr(v3_records[sprite_id], field, None)
        if fd is None:
            continue
        b = buckets[stratum]
        b["eligible"] += 1
        if fd.state == "accepted":
            b["accepted"] += 1
            if _field_match(field, getattr(golden[sprite_id], golden_key, None), fd.accepted_value):
                b["correct"] += 1

    out: dict[str, dict[str, Any]] = {}
    for stratum, b in buckets.items():
        precision = _safe_ratio(b["correct"], b["accepted"])
        coverage = _safe_ratio(b["accepted"], b["eligible"])
        out[stratum] = {
            "eligible": b["eligible"],
            "accepted": b["accepted"],
            "abstained": b["eligible"] - b["accepted"],
            "correct": b["correct"],
            "precision": round(precision, 4),
            "precision_numerator": b["correct"],
            "precision_denominator": b["accepted"],
            "coverage": round(coverage, 4),
            "coverage_numerator": b["accepted"],
            "coverage_denominator": b["eligible"],
            "selective_risk": round(1.0 - precision, 4) if b["accepted"] > 0 else 0.0,
            "ci_lower": round(compute_lower_confidence_bound(b["correct"], b["accepted"], confidence_level), 4),
        }
    return out


def _eval_one(golden: GoldenLabel, v3: RecordDecision) -> dict[str, Any]:
    return {
        "sprite_id": v3.sprite_id,
        "golden_category": golden.category,
        "golden_object": _safe_str(golden.object_name),
        "v3_category": _safe_str(v3.category.accepted_value),
        "v3_object": _safe_str(v3.canonical_object.accepted_value),
        "v3_object_state": v3.canonical_object.state,
        "v3_record_state": v3.record_state,
        "category_match": _safe_str(golden.category) == _safe_str(v3.category.accepted_value),
        "object_match": _safe_str(golden.object_name) == _safe_str(v3.canonical_object.accepted_value),
    }


def _field_match(field: str, golden_val: Any, v3_val: Any) -> bool:
    g = _safe_str(golden_val)
    v = _safe_str(v3_val)
    if g == v:
        return True
    if field == "canonical_object":
        return object_name_token_f1(g, v) >= 0.8
    if field == "category":
        return _safe_str(golden_val) == _safe_str(v3_val)
    return False


def _hierarchy_ancestor_match(v3_hierarchy_node: str, golden_object: str) -> bool:
    from spritelab.harvest.label_v3.taxonomy_v3 import get_hierarchy_node

    v3_node = get_hierarchy_node(v3_hierarchy_node)
    gold_node = get_hierarchy_node(_safe_str(golden_object))
    if v3_node is None or gold_node is None:
        return False
    return v3_node.is_ancestor_of(gold_node) or gold_node.is_ancestor_of(v3_node) or v3_node.name == gold_node.name


def compute_upper_bound(precision: float, n: int, confidence_level: float = 0.95) -> float:
    """One-sided upper confidence bound using Wilson score interval."""
    import math

    if n == 0:
        return 1.0
    z = 1.6449
    denominator = 1 + z * z / n
    center = (precision + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((precision * (1 - precision) / n) + (z * z / (4 * n * n))) / denominator
    return min(1.0, center + margin)


def load_v3_records_from_run(
    run_dir: str | Path,
    *,
    v3_file: str = "v3_records.jsonl",
) -> dict[str, RecordDecision]:
    p = Path(run_dir) / v3_file
    if not p.is_file():
        return {}
    result: dict[str, RecordDecision] = {}
    for record_data in read_jsonl(p):
        sid = str(record_data.get("sprite_id", ""))
        if sid:
            try:
                result[sid] = record_decision_from_json(record_data)
            except Exception:
                continue
    return result


def format_v3_evaluation_report(result: V3EvalResult) -> str:
    lines = [
        f"# V3 Evaluation Report — {result.suite_name}",
        "",
        f"Golden labels: {result.total_golden}",
        f"Matched with v3: {result.matched}",
        "",
        "## Per-Field Metrics",
        "| Field | Accepted | Correct | Precision | CI Lower | Coverage | ECE | Meets Target |",
        "|-------|----------|---------|-----------|----------|----------|-----|-------------|",
    ]
    for field_name in FIELDS:
        if field_name not in result.per_field:
            continue
        m = result.per_field[field_name]
        lines.append(
            f"| {field_name} | {m['accepted_total']} | {m['accepted_correct']} | "
            f"{m['precision']:.4f} | {m['ci_lower']:.4f} | {m['coverage']:.4f} | "
            f"{m['ece']:.4f} | {m['meets_target']} |"
        )

    lines.extend(
        [
            "",
            "## Promotion Gates",
        ]
    )
    for gate, status in result.promotion_gates.items():
        icon = "PASS" if status else "FAIL"
        lines.append(f"- {icon} {gate}")

    lines.extend(
        [
            "",
            "## Acceptance Band",
            f"- Precision: {result.acceptance_band.get('band_precision', 0):.4f}",
            f"- CI Lower: {result.acceptance_band.get('band_ci_lower', 0):.4f}",
            "## Hierarchy Accuracy",
            f"- Exact/ancestor correct: {result.hierarchy_accuracy.get('exact_or_ancestor_correct', 0):.4f}",
            f"- Mean depth delta: {result.hierarchy_accuracy.get('mean_depth_delta', 0):.2f}",
            "## Quality",
            f"- Hard-reject FPR: {result.hard_reject_fpr:.4f}",
            f"- Provenance completeness: {result.provenance_completeness:.4f}",
        ]
    )

    if result.warnings:
        lines.extend(["", "## Warnings"])
        for w in result.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines) + "\n"


def promotion_recommendation(result: V3EvalResult) -> str:
    """Recommend a rollout stage based on evaluation gate status."""

    gates = result.promotion_gates
    core_gates = [
        "category_meets_target",
        "canonical_object_meets_target",
        "hard_reject_zero_fpr",
        "ece_acceptable",
    ]

    all_core_pass = all(gates.get(g, False) for g in core_gates)
    all_pass = all(gates.values())

    if not all_core_pass:
        return "blocked"
    if result.matched < 30:
        return "shadow_only"
    if all_pass:
        return "eligible_for_large_batch"
    return "limited_opt_in"
