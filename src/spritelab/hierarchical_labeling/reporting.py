"""Truth-aware offline reporting for hierarchical labeling experiments."""

from __future__ import annotations

import html
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.hierarchical_labeling.contracts import HumanReferenceLabel, SyntheticOracleLabel
from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    content_identity,
    strict_json_value,
)
from spritelab.hierarchical_labeling.taxonomy import TaxonomyGraph
from spritelab.v3.run_state import atomic_write_json

REPORT_SCHEMA = "spritelab.labeling.offline-report.v1"
VERIFIED_HUMAN_TRUTH_SOURCE = "verified_append_only_human_review"
APPEND_ONLY_HUMAN_TRUTH = "append_only_human_review"
SYNTHETIC_ORACLE_TRUTH = "synthetic_oracle"


def build_report_data(
    records: Sequence[Mapping[str, Any]],
    graph: TaxonomyGraph,
    *,
    truth_rows: Sequence[Mapping[str, Any]] = (),
    synthetic_oracle_rows: Sequence[Mapping[str, Any]] = (),
    calibration_state: str = "not_ready",
    calibration_truth_scope: str | None = None,
    operational: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metrics without inventing precision when reviewed truth is absent."""

    normalized_records = [_record(row, graph) for row in records]
    identities = [row["record_identity"] for row in normalized_records]
    if len(identities) != len(set(identities)):
        raise ContractValidationError("report record identities cannot repeat")
    accepted = [row for row in normalized_records if row["accepted_path"]]
    node_counts: Counter[str] = Counter(node for row in accepted for node in row["accepted_path"])
    depth_counts: Counter[str] = Counter(str(len(row["accepted_path"]) - 1) for row in accepted)
    abstention_depths: Counter[str] = Counter(
        str(max(0, len(row["accepted_path"]))) for row in normalized_records if row["abstained"]
    )
    truth = [
        _truth(
            row,
            graph,
            expected_source=APPEND_ONLY_HUMAN_TRUTH,
            metric_source=VERIFIED_HUMAN_TRUTH_SOURCE,
        )
        for row in truth_rows
    ]
    synthetic_oracle = [
        _truth(
            row,
            graph,
            expected_source=SYNTHETIC_ORACLE_TRUTH,
            metric_source="synthetic_oracle_fixture",
        )
        for row in synthetic_oracle_rows
    ]
    truth_ids = [row["record_identity"] for row in truth]
    if len(truth_ids) != len(set(truth_ids)):
        raise ContractValidationError("report truth record identities cannot repeat")
    oracle_ids = [row["record_identity"] for row in synthetic_oracle]
    if len(oracle_ids) != len(set(oracle_ids)):
        raise ContractValidationError("report synthetic-oracle record identities cannot repeat")
    if set(truth_ids) & set(oracle_ids):
        raise ContractValidationError("a report row cannot be both verified human truth and synthetic oracle")
    record_identities = set(identities)
    if (set(truth_ids) | set(oracle_ids)) - record_identities:
        raise ContractValidationError("report truth rows must bind records present in the report")
    _one_projection(truth, name="verified human truth")
    _one_projection(synthetic_oracle, name="synthetic oracle")
    truth_metrics = _truth_metrics(
        truth,
        state="measured_from_verified_human_truth",
        truth_source=VERIFIED_HUMAN_TRUTH_SOURCE,
        empty_message="Precision is not shown because no verified human truth rows are available.",
    )
    synthetic_oracle_metrics = (
        _truth_metrics(
            synthetic_oracle,
            state="measured_from_synthetic_oracle_fixture",
            truth_source="synthetic_oracle_fixture",
            empty_message="No synthetic-oracle rows are available.",
        )
        if synthetic_oracle
        else None
    )
    effective_calibration_state = calibration_state
    if calibration_truth_scope == SYNTHETIC_ORACLE_TRUTH and calibration_state == "validated_for_scope":
        effective_calibration_state = "ready_for_experiment"
    progress = {
        "images_processed": len(normalized_records),
        "descriptions_complete": sum(row["description_complete"] for row in normalized_records),
        "embeddings_complete": sum(row["embedding_complete"] for row in normalized_records),
        "retrieval_complete": sum(row["retrieval_complete"] for row in normalized_records),
        "decisions_complete": sum(row["decision_complete"] for row in normalized_records),
        "reviews_pending": sum(row["review_pending"] for row in normalized_records),
        "calibration_state": effective_calibration_state,
    }
    operations = _operational_defaults(operational)
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "taxonomy": {
            "version": graph.version,
            "identity": graph.identity,
            "tree": _taxonomy_tree(graph),
        },
        "summary": {
            "records": {"numerator": len(normalized_records), "denominator": len(normalized_records)},
            "accepted": {"numerator": len(accepted), "denominator": len(normalized_records)},
            "abstained": {"numerator": len(normalized_records) - len(accepted), "denominator": len(normalized_records)},
            "accepted_coverage": len(accepted) / len(normalized_records) if normalized_records else None,
            "truth_rows": {"numerator": len(truth), "denominator": len(normalized_records)},
            "truth_source": "verified append-only human review" if truth else None,
            "synthetic_oracle_rows": {
                "numerator": len(synthetic_oracle),
                "denominator": len(normalized_records),
            },
            "calibration_state": effective_calibration_state,
        },
        "records_per_accepted_node": dict(sorted(node_counts.items())),
        "accepted_depth_distribution": dict(sorted(depth_counts.items())),
        "abstention_by_depth": dict(sorted(abstention_depths.items())),
        "abstention_by_source": _group_count(normalized_records, "source_identity", "abstained"),
        "abstention_by_cluster": _group_count(normalized_records, "cluster_identity", "abstained"),
        "visual_metadata_conflicts": {
            "numerator": sum(row["visual_metadata_conflict"] for row in normalized_records),
            "denominator": len(normalized_records),
        },
        "novelty_distribution": _histogram([row["novelty"] for row in normalized_records]),
        "truth_metrics": truth_metrics,
        "synthetic_oracle_metrics": synthetic_oracle_metrics,
        "truth_provenance": _truth_provenance(truth),
        "synthetic_oracle_provenance": _truth_provenance(synthetic_oracle),
        "operational": operations,
        "progress": progress,
        "galleries": {
            "retrieval_neighbor_examples": operations["retrieval_neighbor_examples"],
            "cluster_medoids": operations["cluster_medoids"],
            "samples": operations["sample_gallery"],
        },
        "claims": {
            "precision_graph_available": bool(truth),
            "precision_suppressed_without_truth": True,
            "synthetic_oracle_is_verified_human_truth": False,
            "synthetic_oracle_can_validate_production": False,
            "synthetic_or_experimental_only": True,
            "production_authorization": False,
        },
    }
    payload["report_identity"] = content_identity(REPORT_SCHEMA, payload)
    strict_json_value(payload)
    return payload


def _record(value: Mapping[str, Any], graph: TaxonomyGraph) -> dict[str, Any]:
    record_identity = str(value.get("record_identity", "")).strip()
    if not record_identity:
        raise ContractValidationError("report record identity is required")
    raw_path = value.get("accepted_path", ())
    if not isinstance(raw_path, Sequence) or isinstance(raw_path, (str, bytes)):
        raise ContractValidationError("report accepted path must be an array")
    path = tuple(str(item) for item in raw_path)
    if path and graph.path(path[-1]) != path:
        raise ContractValidationError("report accepted path is not a valid taxonomy path")
    probability = value.get("calibrated_probability")
    if probability is not None and (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0 <= float(probability) <= 1
    ):
        raise ContractValidationError("report calibrated probability must be finite from zero through one")
    return {
        "record_identity": record_identity,
        "accepted_path": path,
        "abstained": bool(value.get("abstained", not path)),
        "source_identity": str(value.get("source_identity") or "unknown"),
        "cluster_identity": str(value.get("cluster_identity") or "unknown"),
        "novelty": _probability(value.get("novelty", 0.0), "report novelty"),
        "visual_metadata_conflict": bool(value.get("visual_metadata_conflict", False)),
        "description_complete": bool(value.get("description_complete", False)),
        "embedding_complete": bool(value.get("embedding_complete", False)),
        "retrieval_complete": bool(value.get("retrieval_complete", False)),
        "decision_complete": bool(value.get("decision_complete", False)),
        "review_pending": bool(value.get("review_pending", False)),
    }


def _truth(
    value: Mapping[str, Any],
    graph: TaxonomyGraph,
    *,
    expected_source: str,
    metric_source: str,
) -> dict[str, Any]:
    if "truth_path" in value or "truth_source" in value:
        raise ContractValidationError(
            "raw truth mappings cannot enable correctness metrics; provide a projected human_reference"
        )
    reference = value.get("human_reference")
    if expected_source == APPEND_ONLY_HUMAN_TRUTH:
        if not isinstance(reference, HumanReferenceLabel) or not reference.verified_append_only:
            raise ContractValidationError("report correctness rows require an identity-bound truth projection")
        assert reference.verification is not None
        verification = reference.verification
        verification.__post_init__()
        reference.__post_init__()
        if verification.source != expected_source:
            raise ContractValidationError(f"report truth projection must have source {expected_source}")
        projected_record_identity = reference.record_identity
        projected_evidence_identity = verification.evidence_bundle_identity
        projected_source_identity = verification.source_identity
        projected_cluster_identity = verification.cluster_identity
        projected_leakage_group_identity = verification.leakage_group_identity
        projected_taxonomy_identity = verification.taxonomy_identity
        actual = reference.taxonomy_path
        partition = reference.partition
        projection_identity = verification.identity
        review_log_identity = verification.review_log_identity
        chain_tip_identity = verification.chain_tip_identity
        cohort_identity = verification.cohort_identity
    else:
        if not isinstance(reference, SyntheticOracleLabel) or reference.truth_source != expected_source:
            raise ContractValidationError("synthetic oracle metrics require a SyntheticOracleLabel")
        reference.__post_init__()
        projected_record_identity = reference.record_identity
        projected_evidence_identity = reference.evidence_bundle_identity
        projected_source_identity = reference.source_identity
        projected_cluster_identity = reference.cluster_identity
        projected_leakage_group_identity = reference.leakage_group_identity
        projected_taxonomy_identity = reference.taxonomy_identity
        actual = reference.taxonomy_path
        partition = reference.partition
        projection_identity = reference.identity
        review_log_identity = reference.oracle_set_identity
        chain_tip_identity = reference.oracle_set_identity
        cohort_identity = reference.cohort_identity
    if partition != "holdout":
        raise ContractValidationError("report precision rows require the immutable holdout partition")
    record_identity = str(value.get("record_identity") or "")
    if not record_identity or record_identity != projected_record_identity:
        raise ContractValidationError("report truth row does not bind the projected record identity")
    evidence_bundle_identity = str(value.get("evidence_bundle_identity") or "")
    if not evidence_bundle_identity or evidence_bundle_identity != projected_evidence_identity:
        raise ContractValidationError("report truth row does not bind the projected evidence bundle")
    source_identity = str(value.get("source_identity") or "")
    cluster_identity = str(value.get("cluster_identity") or "")
    if source_identity != projected_source_identity or cluster_identity != projected_cluster_identity:
        raise ContractValidationError("report truth row does not bind projected source/cluster identities")
    leakage_group_identity = str(value.get("leakage_group_identity") or "")
    if leakage_group_identity != projected_leakage_group_identity:
        raise ContractValidationError("report truth row does not bind the projected leakage group")
    if projected_taxonomy_identity != graph.identity:
        raise ContractValidationError("report truth projection taxonomy does not bind the selected graph")
    predicted = tuple(str(item) for item in value.get("predicted_path", ()))
    if predicted and graph.path(predicted[-1]) != predicted:
        raise ContractValidationError("report predicted path is not a valid taxonomy path")
    if actual and graph.path(actual[-1]) != actual:
        raise ContractValidationError("report truth path is not a valid taxonomy path")
    if not actual:
        raise ContractValidationError("human truth row must provide a reviewed taxonomy path")
    probability = _probability(value.get("calibrated_probability", 0.0), "truth calibrated probability")
    return {
        "record_identity": record_identity,
        "predicted_path": predicted,
        "truth_path": actual,
        "calibrated_probability": probability,
        "source_identity": source_identity,
        "cluster_identity": cluster_identity,
        "leakage_group_identity": leakage_group_identity,
        "evidence_bundle_identity": evidence_bundle_identity,
        "truth_source": metric_source,
        "truth_projection_identity": projection_identity,
        "review_log_identity": review_log_identity,
        "chain_tip_identity": chain_tip_identity,
        "cohort_identity": cohort_identity,
    }


def _one_projection(rows: Sequence[Mapping[str, Any]], *, name: str) -> None:
    for key in ("review_log_identity", "chain_tip_identity", "cohort_identity"):
        values = {str(row[key]) for row in rows}
        if len(values) > 1:
            raise ContractValidationError(f"report {name} rows span multiple {key.replace('_', ' ')} values")


def _truth_provenance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return {
        "truth_source": rows[0]["truth_source"],
        "review_log_identity": rows[0]["review_log_identity"],
        "chain_tip_identity": rows[0]["chain_tip_identity"],
        "cohort_identity": rows[0]["cohort_identity"],
        "leakage_group_identities": sorted({str(row["leakage_group_identity"]) for row in rows}),
        "truth_projection_identities": sorted(str(row["truth_projection_identity"]) for row in rows),
    }


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ContractValidationError(f"{name} must be finite from zero through one")
    return result


def _correct(row: Mapping[str, Any]) -> bool:
    predicted = row["predicted_path"]
    return bool(predicted) and predicted[-1] in row["truth_path"]


def _truth_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    state: str,
    truth_source: str,
    empty_message: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "state": "no_human_truth",
            "precision_coverage_curve": None,
            "confusion_matrix": None,
            "class_precision": None,
            "source_precision": None,
            "message": empty_message,
        }
    thresholds = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)
    curve = []
    for threshold in thresholds:
        accepted = [row for row in rows if row["predicted_path"] and row["calibrated_probability"] >= threshold]
        correct = sum(_correct(row) for row in accepted)
        curve.append(
            {
                "threshold": threshold,
                "precision_numerator": correct,
                "precision_denominator": len(accepted),
                "precision": correct / len(accepted) if accepted else None,
                "coverage_numerator": len(accepted),
                "coverage_denominator": len(rows),
                "coverage": len(accepted) / len(rows),
                "risk": (len(accepted) - correct) / len(accepted) if accepted else None,
                "truth_source": truth_source,
            }
        )
    confusion: Counter[tuple[str, str]] = Counter()
    class_rows: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    source_rows: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        predicted = row["predicted_path"][-1] if row["predicted_path"] else "abstained"
        actual = row["truth_path"][-1]
        confusion[(actual, predicted)] += 1
        class_rows[predicted].append(row)
        source_rows[row["source_identity"]].append(row)
    return {
        "state": state,
        "sample_size": len(rows),
        "truth_source": truth_source,
        "precision_coverage_curve": curve,
        "confusion_matrix": [
            {"truth": truth, "predicted": predicted, "count": count, "denominator": len(rows)}
            for (truth, predicted), count in sorted(confusion.items())
        ],
        "class_precision": _precision_groups(class_rows, truth_source=truth_source),
        "source_precision": _precision_groups(source_rows, truth_source=truth_source),
    }


def _precision_groups(groups: Mapping[str, Sequence[Mapping[str, Any]]], *, truth_source: str) -> dict[str, Any]:
    return {
        name: {
            "numerator": sum(_correct(row) for row in rows if row["predicted_path"]),
            "denominator": sum(bool(row["predicted_path"]) for row in rows),
            "sample_size": len(rows),
            "truth_source": truth_source,
        }
        for name, rows in sorted(groups.items())
    }


def _taxonomy_tree(graph: TaxonomyGraph) -> list[dict[str, Any]]:
    return [
        {
            "node_id": node.node_id,
            "display_name": node.display_name,
            "parent_id": node.parent_id,
            "depth": graph.depth(node.node_id),
        }
        for node in graph.nodes
    ]


def _group_count(records: Sequence[Mapping[str, Any]], key: str, flag: str) -> dict[str, Any]:
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {
        name: {"numerator": sum(bool(row[flag]) for row in rows), "denominator": len(rows)}
        for name, rows in sorted(groups.items())
    }


def _histogram(values: Sequence[float]) -> list[dict[str, Any]]:
    return [
        {
            "minimum": lower,
            "maximum": upper,
            "numerator": sum(lower <= value < upper or (upper == 1.0 and value == 1.0) for value in values),
            "denominator": len(values),
        }
        for lower, upper in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
    ]


def _operational_defaults(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    defaults: dict[str, Any] = {
        "provider_usage": {},
        "provider_failures": {},
        "cache": {"hits": 0, "lookups": 0, "rate": None},
        "hosted_call_count": 0,
        "cost_estimate": "unknown",
        "human_review": {"completed": 0, "pending": 0, "throughput_per_hour": None},
        "taxonomy_gaps": [],
        "retrieval_neighbor_examples": [],
        "cluster_medoids": [],
        "sample_gallery": [],
    }
    defaults.update(raw)
    strict_json_value(defaults)
    return defaults


def write_offline_report(data: Mapping[str, Any], output_directory: str | Path) -> tuple[Path, Path]:
    strict_json_value(dict(data))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "labeling_report.json"
    html_path = output / "labeling_report.html"
    atomic_write_json(json_path, dict(data))
    html_path.write_text(_render_html(data), encoding="utf-8", newline="\n")
    return json_path, html_path


def _render_html(data: Mapping[str, Any]) -> str:
    summary = data["summary"]
    truth = data["truth_metrics"]
    curve = truth.get("precision_coverage_curve")
    curve_markup = (
        _curve_svg(curve)
        + "<table><thead><tr><th>Threshold</th><th>Precision</th><th>Coverage</th><th>Sample</th></tr></thead><tbody>"
        + "".join(
            f"<tr><td>{row['threshold']:.2f}</td><td>{_metric(row['precision'])}</td>"
            f"<td>{_metric(row['coverage'])}</td><td>{row['precision_numerator']}/{row['precision_denominator']}"
            f" accepted; truth n={row['coverage_denominator']}</td></tr>"
            for row in curve
        )
        + "</tbody></table>"
        if curve
        else f"<p class='empty'>{html.escape(str(truth['message']))}</p>"
    )
    embedded = json.dumps(data, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sprite Lab hierarchical labeling report</title><style>
body{{font:15px system-ui,sans-serif;margin:0;background:#10141c;color:#eef2f7}}main{{max-width:1100px;margin:auto;padding:28px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}section,.card{{background:#19212d;border:1px solid #334155;border-radius:10px;padding:16px;margin:14px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #334155}}.empty{{color:#fbbf24}}svg{{width:100%;max-width:720px;height:220px;background:#0f172a}}code{{overflow-wrap:anywhere}}
</style></head><body><main><h1>Hierarchical labeling report</h1>
<div class="cards"><div class="card"><strong>Records</strong><p>{summary["records"]["numerator"]}</p></div>
<div class="card"><strong>Accepted coverage</strong><p>{_metric(summary["accepted_coverage"])}</p></div>
<div class="card"><strong>Human truth</strong><p>{summary["truth_rows"]["numerator"]} reviewed rows</p></div>
<div class="card"><strong>Calibration</strong><p>{html.escape(str(summary["calibration_state"]))}</p></div></div>
<section><h2>Precision and coverage</h2>{curve_markup}</section>
<section><h2>Accepted depth</h2><pre>{html.escape(json.dumps(data["accepted_depth_distribution"], indent=2))}</pre></section>
<section><h2>Taxonomy</h2><p>{len(data["taxonomy"]["tree"])} controlled nodes; identity <code>{html.escape(data["taxonomy"]["identity"])}</code>.</p></section>
<section><h2>Operations and galleries</h2><details><summary>Embedded report data</summary><pre id="raw"></pre></details></section>
<script type="application/json" id="report-data">{embedded}</script><script>const d=JSON.parse(document.getElementById('report-data').textContent);document.getElementById('raw').textContent=JSON.stringify(d.operational,null,2);</script>
</main></body></html>"""


def _metric(value: float | None) -> str:
    return "No data" if value is None else f"{100 * value:.1f}%"


def _curve_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    points = [
        (30 + 650 * float(row["coverage"]), 190 - 160 * float(row["precision"]))
        for row in rows
        if row.get("precision") is not None
    ]
    if not points:
        return ""
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        '<svg role="img" aria-label="Precision versus coverage"><line x1="30" y1="190" x2="680" y2="190" '
        'stroke="#64748b"/><line x1="30" y1="190" x2="30" y2="30" stroke="#64748b"/>'
        f'<polyline points="{coordinates}" fill="none" stroke="#38bdf8" stroke-width="3"/></svg>'
    )
