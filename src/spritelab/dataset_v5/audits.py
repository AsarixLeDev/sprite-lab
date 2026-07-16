"""Continuous fail-closed label-faithfulness audits for raw Dataset-v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from spritelab.dataset_v5.blind import BlindPayloadLeakageError, audit_blind_payload
from spritelab.dataset_v5.identity import canonical_json_bytes, is_opaque_record_id
from spritelab.dataset_v5.labeling import (
    RECONCILIATION_VERSION,
    SEMANTIC_FIELDS,
    SUPERVISION_CLASSES,
    SUPERVISION_POLICY_VERSION,
)

AUDIT_POLICY_VERSION = "raw_v5_label_audit_v1"
DEFAULT_STOP_GATES: dict[str, float] = {
    "filename_leakage": 0.0,
    "cache_identity_collisions": 0.0,
    "silent_repairs": 0.0,
    "critical_field_contradiction_rate": 0.02,
    "unsupported_exact_material_rate": 0.01,
    "taxonomy_invalidity": 0.0,
    "missing_provenance": 0.0,
    "hard_relation_leakage": 0.0,
    "schema_invalidity": 0.0,
}

_ALLOWED_FIELD_STATES = frozenset(
    {
        "abstained",
        "conflicted",
        "known",
        "missing",
        "not_applicable",
        "oov",
        "unknown",
        "unsupported_removed",
    }
)
_REQUIRED_MEASURED_METRICS = frozenset(
    {
        "abstention_rate",
        "cache_identity_collisions",
        "critical_field_agreement",
        "critical_field_contradiction_rate",
        "description_contradictions",
        "duplicate_request_rate",
        "field_disagreement_rate",
        "filename_taint_cases",
        "filename_leakage",
        "hard_relation_leakage",
        "invalid_json",
        "invalid_json_rate",
        "missing_provenance",
        "new_taxonomy_value_count",
        "repair_rate",
        "schema_invalidity",
        "silent_repairs",
        "source_visual_conflicts",
        "taxonomy_invalidity",
        "unsupported_exact_material_rate",
    }
)


def audit_label_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_id: str,
    forbidden_metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    relation_manifest: Sequence[Mapping[str, Any]] = (),
    split_by_record: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit one deterministic batch and return a blocking status."""

    if not records:
        raise ValueError("label audit batch must not be empty")
    if len(records) > 50:
        raise ValueError("label batches must contain at most 50 records")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("label audit batch_id must be non-empty")
    forbidden_metadata_by_id = forbidden_metadata_by_id or {}
    critical_total = 0
    critical_conflicts = 0
    unsupported_material = 0
    exact_material_claims = 0
    invalid_json = 0
    silent_repairs = 0
    taxonomy_invalidity = 0
    missing_provenance = 0
    filename_leakage = 0
    duplicate_requests = 0
    request_hashes: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selected_full: set[str] = set()
    remaining: list[str] = []
    cache_rows: list[Mapping[str, Any]] = []
    field_total = 0
    field_conflicts = 0
    abstentions = 0
    artifact_total = 0
    source_visual_conflicts = 0
    filename_taint_cases = 0
    new_taxonomy_value_count = 0
    schema_invalidity = 0
    schema_findings: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    record_ids: list[str] = []
    pack_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    creator_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record_index, record in enumerate(records):
        record_id = str(record.get("record_id") or "")
        record_ids.append(record_id)
        record_issues = _label_record_schema_issues(record, seen_record_ids=seen_record_ids)
        if record_issues:
            schema_invalidity += len(record_issues)
            schema_findings.append(
                {
                    "issues": record_issues,
                    "record_id": record_id,
                    "record_index": record_index,
                }
            )
            selected_full.add(record_id)
        if record_id:
            seen_record_ids.add(record_id)
        reconciliation = record.get("reconciliation") if isinstance(record.get("reconciliation"), Mapping) else {}
        fields = reconciliation.get("fields") if isinstance(reconciliation.get("fields"), Mapping) else {}
        for field in ("category", "canonical_object", "domain", "role"):
            details = fields.get(field) if isinstance(fields.get(field), Mapping) else {}
            critical_total += 1
            if not details or details.get("state") == "conflicted" or _unreconciled_disagreement(details):
                critical_conflicts += 1
                selected_full.add(record_id)
        material = fields.get("explicit_material") if isinstance(fields.get("explicit_material"), Mapping) else {}
        if material.get("adjudication_value") is not None:
            exact_material_claims += 1
            if material.get("state") == "unsupported_removed":
                unsupported_material += 1
                selected_full.add(record_id)
        for field_name, details in fields.items():
            if not isinstance(details, Mapping):
                continue
            field_total += 1
            if details.get("state") == "conflicted":
                field_conflicts += 1
            if details.get("state") == "abstained":
                abstentions += 1
                selected_full.add(record_id)
            value = details.get("value")
            if value is not None:
                label_counts[str(field_name)][_json_stable_value(value)] += 1
        if record.get("source_visual_conflict"):
            source_visual_conflicts += 1
            selected_full.add(record_id)
        if record.get("filename_taint_status") == "tainted_metadata":
            filename_taint_cases += 1
            selected_full.add(record_id)
        if record.get("new_taxonomy_values"):
            values = record.get("new_taxonomy_values")
            new_values = len(values) if isinstance(values, Sequence) and not isinstance(values, str) else 1
            new_taxonomy_value_count += new_values
            taxonomy_invalidity += new_values
            selected_full.add(record_id)
        category = _field_value(fields, "category")
        if category is not None:
            pack = str(record.get("source_pack") or "").strip()
            creator = str(record.get("source_creator") or record.get("creator_lineage") or "").strip()
            if pack:
                pack_category_counts[pack][str(category)] += 1
            if creator:
                creator_category_counts[creator][str(category)] += 1
        artifacts = [
            value
            for value in (record.get("adjudication_artifact"), record.get("consistency_artifact"))
            if isinstance(value, Mapping)
        ]
        for artifact in artifacts:
            artifact_total += 1
            cache_rows.append(artifact)
            if artifact.get("status") == "invalid":
                invalid_json += 1
                selected_full.add(record_id)
            if (
                artifact.get("repair_used")
                or artifact.get("silent_repair")
                or str(artifact.get("stage_status") or "").startswith("success_after_")
            ):
                silent_repairs += 1
                selected_full.add(record_id)
            request_hash = str(artifact.get("request_sha256") or "")
            if request_hash:
                request_hashes[request_hash] += 1
        reported_taxonomy_invalidity = record.get("taxonomy_invalidity_count", 0)
        if (
            isinstance(reported_taxonomy_invalidity, int)
            and not isinstance(reported_taxonomy_invalidity, bool)
            and reported_taxonomy_invalidity >= 0
        ):
            taxonomy_invalidity += reported_taxonomy_invalidity
        if not record.get("source_binding_valid", False):
            missing_provenance += 1
        payload = record.get("blind_request_payload")
        if isinstance(payload, Mapping):
            try:
                audit_blind_payload(payload, forbidden_metadata=forbidden_metadata_by_id.get(record_id))
            except BlindPayloadLeakageError:
                filename_leakage += 1
                selected_full.add(record_id)
        if record_id not in selected_full:
            remaining.append(record_id)
    duplicate_requests = sum(count - 1 for count in request_hashes.values() if count > 1)
    cache_collisions = cache_identity_collisions(cache_rows)
    hard_leakage = hard_relation_leakage(relation_manifest, split_by_record or {})
    critical_rate = _rate(critical_conflicts, critical_total)
    material_rate = _rate(unsupported_material, exact_material_claims)
    metrics = {
        "abstention_rate": _rate(abstentions, field_total),
        "cache_identity_collisions": cache_collisions,
        "critical_field_agreement": 1.0 - critical_rate,
        "critical_field_contradiction_rate": critical_rate,
        "description_contradictions": sum(int(row.get("description_contradiction", False)) for row in records),
        "duplicate_request_rate": _rate(duplicate_requests, max(1, len(request_hashes))),
        "field_disagreement_rate": _rate(field_conflicts, field_total),
        "filename_taint_cases": filename_taint_cases,
        "filename_leakage": filename_leakage,
        "hard_relation_leakage": hard_leakage,
        "invalid_json": invalid_json,
        "invalid_json_rate": _rate(invalid_json, artifact_total),
        "missing_provenance": missing_provenance,
        "new_taxonomy_value_count": new_taxonomy_value_count,
        "repair_rate": _rate(silent_repairs, artifact_total),
        "schema_invalidity": schema_invalidity,
        "silent_repairs": silent_repairs,
        "source_visual_conflicts": source_visual_conflicts,
        "taxonomy_invalidity": taxonomy_invalidity,
        "unsupported_exact_material_rate": material_rate,
    }
    failures = _gate_failures(metrics)
    sample = sorted(selected_full) + _deterministic_sample(remaining, fraction=0.20, salt=batch_id)
    return {
        "audit_policy_version": AUDIT_POLICY_VERSION,
        "authoritative": not failures,
        "batch_id": batch_id,
        "blocking_failures": failures,
        "label_distributions": {field: dict(sorted(counts.items())) for field, counts in sorted(label_counts.items())},
        "label_distributions_by_creator": _nested_counter_dict(creator_category_counts),
        "label_distributions_by_pack": _nested_counter_dict(pack_category_counts),
        "metrics": metrics,
        "record_count": len(records),
        "record_ids": record_ids,
        "record_ids_sha256": hashlib.sha256(canonical_json_bytes(record_ids)).hexdigest(),
        "review_sample_record_ids": list(dict.fromkeys(sample)),
        "schema_findings": schema_findings,
        "status": "pass" if not failures else "blocked_non_authoritative",
        "stop_gates": DEFAULT_STOP_GATES,
        "wilson_intervals_95": {
            "critical_field_contradiction_rate": wilson_interval(critical_conflicts, critical_total),
            "unsupported_exact_material_rate": wilson_interval(unsupported_material, exact_material_claims),
        },
    }


def audit_label_health(batch_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch_reports:
        return {
            "audit_policy_version": AUDIT_POLICY_VERSION,
            "batch_count": 0,
            "blocked_batches": [],
            "blocking_reason": "no_label_batch_audits",
            "coverage": {
                "overlapping_record_ids": [],
                "total_record_count": 0,
                "unique_record_count": 0,
            },
            "invalid_reports": [],
            "ok": False,
            "status": "blocked_non_authoritative",
        }
    blocked: list[str] = []
    invalid_reports: list[dict[str, Any]] = []
    seen_batch_ids: set[str] = set()
    seen_record_ids: set[str] = set()
    overlapping_record_ids: set[str] = set()
    partial_batches: list[str] = []
    total_record_count = 0
    for report_index, report in enumerate(batch_reports):
        batch_id = str(report.get("batch_id") or "")
        display_id = batch_id or f"<missing:{report_index}>"
        issues = _batch_report_schema_issues(report)
        if batch_id in seen_batch_ids:
            issues.append("duplicate_batch_id")
        if batch_id:
            seen_batch_ids.add(batch_id)
        record_ids = report.get("record_ids")
        if isinstance(record_ids, list):
            total_record_count += len(record_ids)
            for record_id in record_ids:
                if isinstance(record_id, str) and record_id in seen_record_ids:
                    overlapping_record_ids.add(record_id)
                elif isinstance(record_id, str):
                    seen_record_ids.add(record_id)
        record_count = report.get("record_count")
        if isinstance(record_count, int) and not isinstance(record_count, bool) and 0 < record_count < 50:
            partial_batches.append(display_id)
        if report.get("status") != "pass":
            blocked.append(display_id)
        if issues:
            invalid_reports.append({"batch_id": display_id, "issues": sorted(set(issues))})
    if len(partial_batches) > 1:
        invalid_reports.append(
            {
                "batch_id": "<coverage>",
                "issues": ["multiple_partial_batches:" + ",".join(sorted(partial_batches))],
            }
        )
    if overlapping_record_ids:
        invalid_reports.append({"batch_id": "<coverage>", "issues": ["overlapping_record_ids"]})
    ok = not blocked and not invalid_reports
    return {
        "audit_policy_version": AUDIT_POLICY_VERSION,
        "batch_count": len(batch_reports),
        "blocked_batches": blocked,
        "coverage": {
            "overlapping_record_ids": sorted(overlapping_record_ids),
            "partial_batches": sorted(partial_batches),
            "total_record_count": total_record_count,
            "unique_record_count": len(seen_record_ids),
        },
        "invalid_reports": invalid_reports,
        "ok": ok,
        "status": "healthy" if ok else "blocked_non_authoritative",
    }


def verify_no_name_leakage(
    requests: Sequence[Mapping[str, Any]],
    forbidden_metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    findings = []
    for row in requests:
        record_id = str(row.get("record_id") or row.get("metadata", {}).get("record_id") or "")
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else row
        try:
            audit_blind_payload(payload, forbidden_metadata=forbidden_metadata_by_id.get(record_id))
        except BlindPayloadLeakageError as exc:
            findings.append({"record_id": record_id, "details": str(exc)})
    return {
        "audit_policy_version": AUDIT_POLICY_VERSION,
        "filename_leakage": len(findings),
        "findings": findings,
        "ok": not findings,
    }


def verify_label_drift(
    current: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current_index = _index_records_for_drift(current, side="current")
    baseline_index = _index_records_for_drift(baseline, side="baseline")
    changed = []
    for record_id in sorted(set(current_index) & set(baseline_index)):
        left = _semantic_projection(baseline_index[record_id])
        right = _semantic_projection(current_index[record_id])
        if left != right:
            changed.append({"record_id": record_id, "before": left, "after": right})
    added = sorted(set(current_index) - set(baseline_index))
    removed = sorted(set(baseline_index) - set(current_index))
    current_distributions = _label_distributions(current_index.values())
    baseline_distributions = _label_distributions(baseline_index.values())
    distribution_drift = _mapping_differences(baseline_distributions, current_distributions)
    current_pack = _grouped_category_distributions(current_index.values(), "source_pack")
    baseline_pack = _grouped_category_distributions(baseline_index.values(), "source_pack")
    current_creator = _grouped_category_distributions(current_index.values(), "source_creator", "creator_lineage")
    baseline_creator = _grouped_category_distributions(baseline_index.values(), "source_creator", "creator_lineage")
    pack_drift = _mapping_differences(baseline_pack, current_pack)
    creator_drift = _mapping_differences(baseline_creator, current_creator)
    return {
        "added_record_ids": added,
        "audit_policy_version": AUDIT_POLICY_VERSION,
        "category_distribution_drift": distribution_drift.get("category", []),
        "changed": changed,
        "creator_drift": creator_drift,
        "drift_count": len(changed) + len(added) + len(removed),
        "label_distribution_drift": distribution_drift,
        "ok": not changed
        and not added
        and not removed
        and not distribution_drift
        and not pack_drift
        and not creator_drift,
        "pack_drift": pack_drift,
        "removed_record_ids": removed,
    }


def cache_identity_collisions(rows: Sequence[Mapping[str, Any]]) -> int:
    requests_by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cache_key = str(row.get("cache_key") or "")
        request_hash = str(row.get("request_sha256") or "")
        if cache_key and request_hash:
            requests_by_key[cache_key].add(request_hash)
    return sum(1 for requests in requests_by_key.values() if len(requests) > 1)


def hard_relation_leakage(
    relations: Sequence[Mapping[str, Any]],
    split_by_record: Mapping[str, str],
) -> int:
    leaks = 0
    for relation in relations:
        if not relation.get("hard_split_constraint", True):
            continue
        splits = {
            split_by_record.get(str(member))
            for member in relation.get("members", [])
            if split_by_record.get(str(member)) is not None
        }
        if len(splits) > 1:
            leaks += 1
    return leaks


def audit_random_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
    salt: str = "raw-v5-audit",
) -> list[Mapping[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    ordered = sorted(
        records,
        key=lambda row: hashlib.sha256(f"{salt}\0{row.get('record_id', '')}".encode()).hexdigest(),
    )
    return ordered[:count]


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> dict[str, float | int]:
    if trials <= 0:
        return {"lower": 0.0, "observed": 0.0, "successes": successes, "trials": trials, "upper": 1.0}
    observed = successes / trials
    denominator = 1 + z * z / trials
    center = (observed + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt((observed * (1 - observed) + z * z / (4 * trials)) / trials) / denominator
    return {
        "lower": max(0.0, center - radius),
        "observed": observed,
        "successes": successes,
        "trials": trials,
        "upper": min(1.0, center + radius),
    }


def _label_record_schema_issues(
    record: Mapping[str, Any],
    *,
    seen_record_ids: set[str],
) -> list[str]:
    issues: list[str] = []
    raw_record_id = record.get("record_id")
    if not isinstance(raw_record_id, str) or not is_opaque_record_id(raw_record_id):
        issues.append("record_id_not_opaque")
    elif raw_record_id in seen_record_ids:
        issues.append("duplicate_record_id")
    if not isinstance(record.get("source_binding_valid"), bool):
        issues.append("source_binding_valid_not_boolean")
    taxonomy_count = record.get("taxonomy_invalidity_count", 0)
    if isinstance(taxonomy_count, bool) or not isinstance(taxonomy_count, int) or taxonomy_count < 0:
        issues.append("taxonomy_invalidity_count_invalid")
    new_values = record.get("new_taxonomy_values")
    if new_values is not None and (
        not isinstance(new_values, Sequence)
        or isinstance(new_values, str)
        or not all(isinstance(value, str) and value for value in new_values)
    ):
        issues.append("new_taxonomy_values_invalid")

    reconciliation = record.get("reconciliation")
    if not isinstance(reconciliation, Mapping):
        issues.append("reconciliation_missing")
        fields: Mapping[str, Any] = {}
    else:
        if reconciliation.get("reconciliation_version") != RECONCILIATION_VERSION:
            issues.append("reconciliation_version_invalid")
        if reconciliation.get("supervision_policy_version") != SUPERVISION_POLICY_VERSION:
            issues.append("supervision_policy_version_invalid")
        if reconciliation.get("inclusion_decision") not in {"candidate", "quarantine"}:
            issues.append("inclusion_decision_invalid")
        critical_conflicts = reconciliation.get("critical_conflicts")
        if not isinstance(critical_conflicts, list) or not all(isinstance(value, str) for value in critical_conflicts):
            issues.append("critical_conflicts_invalid")
        if not isinstance(reconciliation.get("deterministic_fields"), Mapping):
            issues.append("deterministic_fields_missing")
        raw_fields = reconciliation.get("fields")
        if not isinstance(raw_fields, Mapping):
            issues.append("semantic_fields_missing")
            fields = {}
        else:
            fields = raw_fields
    missing_fields = sorted(set(SEMANTIC_FIELDS) - set(fields))
    unexpected_fields = sorted(set(fields) - set(SEMANTIC_FIELDS))
    if missing_fields:
        issues.append("missing_semantic_fields:" + ",".join(missing_fields))
    if unexpected_fields:
        issues.append("unexpected_semantic_fields:" + ",".join(unexpected_fields))
    required_detail_keys = {
        "adjudication_value",
        "agreement",
        "consistency_value",
        "negative_target",
        "state",
        "supervision_class",
        "target_mask",
        "value",
    }
    for field in SEMANTIC_FIELDS:
        details = fields.get(field)
        if not isinstance(details, Mapping):
            continue
        missing_keys = sorted(required_detail_keys - set(details))
        if missing_keys:
            issues.append(f"{field}:missing_detail_keys:" + ",".join(missing_keys))
        state = details.get("state")
        supervision = details.get("supervision_class")
        target_mask = details.get("target_mask")
        if state not in _ALLOWED_FIELD_STATES:
            issues.append(f"{field}:state_invalid")
        if supervision not in SUPERVISION_CLASSES:
            issues.append(f"{field}:supervision_class_invalid")
        if not isinstance(details.get("agreement"), bool):
            issues.append(f"{field}:agreement_not_boolean")
        if details.get("negative_target") is not False:
            issues.append(f"{field}:negative_target_invalid")
        if isinstance(target_mask, bool) or target_mask not in {0, 1}:
            issues.append(f"{field}:target_mask_invalid")
        expected_mask = 1 if supervision in {"supervised_strong", "supervised_weak"} else 0
        if target_mask in {0, 1} and not isinstance(target_mask, bool) and target_mask != expected_mask:
            issues.append(f"{field}:target_mask_supervision_mismatch")
        if supervision == "supervised_strong":
            issues.append(f"{field}:uncalibrated_semantic_marked_strong")
        if state == "known" and supervision != "supervised_weak":
            issues.append(f"{field}:known_semantic_not_weak")
        if state in {"abstained", "missing", "not_applicable", "oov", "unknown"} and supervision != "unlabeled":
            issues.append(f"{field}:non_target_state_not_unlabeled")
        if state in {"conflicted", "unsupported_removed"} and supervision != "auxiliary_only":
            issues.append(f"{field}:conflict_state_not_auxiliary")
        if state in {"abstained", "conflicted", "missing", "unsupported_removed"} and details.get("value") is not None:
            issues.append(f"{field}:non_target_state_has_value")

    for pass_name in ("adjudication", "consistency"):
        artifact = record.get(f"{pass_name}_artifact")
        if not isinstance(artifact, Mapping):
            issues.append(f"{pass_name}_artifact_missing")
            continue
        required_artifact_keys = {
            "authoritative",
            "backend",
            "cache_key",
            "endpoint_identity",
            "model_identifier",
            "model_version",
            "pass_kind",
            "prompt_version",
            "provider",
            "provider_schema_version",
            "request_schema_version",
            "request_sha256",
            "response_schema_version",
            "status",
        }
        missing_artifact_keys = sorted(required_artifact_keys - set(artifact))
        if missing_artifact_keys:
            issues.append(f"{pass_name}_artifact_missing_keys:" + ",".join(missing_artifact_keys))
        status = artifact.get("status")
        authoritative = artifact.get("authoritative")
        if status not in {"invalid", "success"}:
            issues.append(f"{pass_name}_artifact_status_invalid")
        if not isinstance(authoritative, bool):
            issues.append(f"{pass_name}_artifact_authoritative_not_boolean")
        if status == "success" and authoritative is not True:
            issues.append(f"{pass_name}_artifact_success_not_authoritative")
        if status == "invalid" and authoritative is not False:
            issues.append(f"{pass_name}_artifact_invalid_marked_authoritative")
        if artifact.get("pass_kind") != pass_name:
            issues.append(f"{pass_name}_artifact_pass_kind_mismatch")
    payload = record.get("blind_request_payload")
    if not isinstance(payload, Mapping) or not payload:
        issues.append("blind_request_payload_missing")
    return issues


def _batch_report_schema_issues(report: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("audit_policy_version") != AUDIT_POLICY_VERSION:
        issues.append("audit_policy_version_invalid")
    batch_id = report.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        issues.append("batch_id_missing")
    status = report.get("status")
    if status not in {"blocked_non_authoritative", "pass"}:
        issues.append("status_invalid")
    authoritative = report.get("authoritative")
    if not isinstance(authoritative, bool) or authoritative is not (status == "pass"):
        issues.append("authoritative_status_mismatch")
    record_count = report.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or not 1 <= record_count <= 50:
        issues.append("record_count_invalid")
    record_ids = report.get("record_ids")
    if not isinstance(record_ids, list):
        issues.append("record_ids_missing")
    else:
        if record_count != len(record_ids):
            issues.append("record_count_mismatch")
        if any(not isinstance(record_id, str) or not is_opaque_record_id(record_id) for record_id in record_ids):
            issues.append("record_ids_not_opaque")
        if len(set(record_ids)) != len(record_ids):
            issues.append("duplicate_record_ids_within_batch")
        expected_digest = hashlib.sha256(canonical_json_bytes(record_ids)).hexdigest()
        if report.get("record_ids_sha256") != expected_digest:
            issues.append("record_ids_sha256_mismatch")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        issues.append("metrics_missing")
    else:
        missing_metrics = sorted(_REQUIRED_MEASURED_METRICS - set(metrics))
        if missing_metrics:
            issues.append("metrics_missing_keys:" + ",".join(missing_metrics))
        for metric in _REQUIRED_MEASURED_METRICS & set(metrics):
            value = metrics[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                issues.append(f"metric_not_finite_numeric:{metric}")
        if not any(issue.startswith("metric_not_finite_numeric:") for issue in issues):
            expected_failures = _gate_failures(metrics)
            if report.get("blocking_failures") != expected_failures:
                issues.append("blocking_failures_do_not_match_metrics")
    if report.get("stop_gates") != DEFAULT_STOP_GATES:
        issues.append("stop_gates_invalid")
    if not isinstance(report.get("blocking_failures"), list):
        issues.append("blocking_failures_missing")
    if not isinstance(report.get("review_sample_record_ids"), list):
        issues.append("review_sample_missing")
    if not isinstance(report.get("schema_findings"), list):
        issues.append("schema_findings_missing")
    if not isinstance(report.get("wilson_intervals_95"), Mapping):
        issues.append("wilson_intervals_missing")
    return issues


def _unreconciled_disagreement(details: Mapping[str, Any]) -> bool:
    state = details.get("state")
    if state in {"abstained", "missing", "not_applicable", "oov", "unknown", "unsupported_removed"}:
        return False
    return details.get("adjudication_value") != details.get("consistency_value") or details.get("agreement") is not True


def _gate_failures(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for metric, threshold in DEFAULT_STOP_GATES.items():
        observed = float(metrics.get(metric, 0) or 0)
        if observed > threshold:
            failures.append({"metric": metric, "observed": observed, "threshold": threshold})
    return failures


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _deterministic_sample(values: Sequence[str], *, fraction: float, salt: str) -> list[str]:
    count = math.ceil(len(values) * fraction)
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest(),
    )[:count]


def _semantic_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    reconciliation = record.get("reconciliation") if isinstance(record.get("reconciliation"), Mapping) else {}
    fields = reconciliation.get("fields") if isinstance(reconciliation.get("fields"), Mapping) else {}
    raw_critical_conflicts = reconciliation.get("critical_conflicts")
    critical_conflicts = (
        sorted(str(value) for value in raw_critical_conflicts)
        if isinstance(raw_critical_conflicts, Sequence) and not isinstance(raw_critical_conflicts, str)
        else []
    )
    return {
        "critical_conflicts": critical_conflicts,
        "deterministic_fields": _field_projection(reconciliation.get("deterministic_fields")),
        "fields": _field_projection(fields),
        "inclusion_decision": reconciliation.get("inclusion_decision"),
        "passes": {
            "adjudication": _artifact_identity_projection(record.get("adjudication_artifact")),
            "consistency": _artifact_identity_projection(record.get("consistency_artifact")),
        },
        "reconciliation_version": reconciliation.get("reconciliation_version"),
        "supervision_policy_version": reconciliation.get("supervision_policy_version"),
    }


def _index_records_for_drift(
    records: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for record_index, row in enumerate(records):
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not is_opaque_record_id(record_id):
            raise ValueError(f"{side} drift record {record_index} has a missing or non-opaque record_id")
        if record_id in index:
            raise ValueError(f"{side} drift input contains duplicate record_id: {record_id}")
        index[record_id] = row
    return index


def _field_projection(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "adjudication_value",
        "agreement",
        "consistency_value",
        "negative_target",
        "reason",
        "state",
        "supervision_class",
        "target_mask",
        "value",
    )
    return {
        str(field): {key: details.get(key) for key in keys}
        for field, details in sorted(value.items())
        if isinstance(details, Mapping)
    }


def _artifact_identity_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "backend",
        "cache_key",
        "endpoint_identity",
        "model_family",
        "model_identifier",
        "model_version",
        "pass_kind",
        "prompt_version",
        "provider",
        "provider_schema_version",
        "request_schema_version",
        "response_schema_version",
    )
    return {key: value.get(key) for key in keys}


def _field_value(fields: Mapping[str, Any], field: str) -> Any:
    details = fields.get(field)
    return details.get("value") if isinstance(details, Mapping) else None


def _nested_counter_dict(values: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(sorted(counter.items())) for key, counter in sorted(values.items())}


def _label_distributions(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        projected_fields = _semantic_projection(record).get("fields", {})
        for field, details in projected_fields.items():
            value = details.get("value") if isinstance(details, Mapping) else None
            if value is not None:
                counts[field][_json_stable_value(value)] += 1
    return _nested_counter_dict(counts)


def _grouped_category_distributions(
    records: Iterable[Mapping[str, Any]], *group_fields: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        group = next((str(record.get(field) or "").strip() for field in group_fields if record.get(field)), "")
        category_details = _semantic_projection(record).get("fields", {}).get("category", {})
        category = category_details.get("value") if isinstance(category_details, Mapping) else None
        if group and category is not None:
            counts[group][_json_stable_value(category)] += 1
    return _nested_counter_dict(counts)


def _mapping_differences(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


def _json_stable_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
