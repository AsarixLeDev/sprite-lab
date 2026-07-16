"""Provider-safe preparation contract for Labeling-v4 calibration review."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.pipeline import LabelV4PipelineConfig, label_record_v4

AUDIT_SELECTION_SCHEMA = "label_v4_audit_selection_v1"
PREFILLED_AUDIT_SCHEMA = "label_v4_prefilled_audit_record_v1"
HUMAN_TRUTH_SCHEMA = "label_v4_human_truth_v1"
LEGACY_SELECTION_SCHEMAS = frozenset({"label_v4_calibration_wave1_v1", AUDIT_SELECTION_SCHEMA})
INFERENCE_POLICIES = frozenset({"deterministic-only", "cached-only", "semantic-minimal"})
QUEUE_IDENTITY_FIELDS = (
    "queue_id",
    "audit_order",
    "prefill_record_hash",
    "audit_id",
    "sprite_id",
    "exported_rgba_hash",
    "normalized_alpha_hash",
    "alpha_mask_hash",
    "downloaded_file_hash",
    "quality_state",
    "quality_event",
)
VALUE_STATES = frozenset(
    {
        "known",
        "model_abstained",
        "not_applicable",
        "not_scorable",
        "missing_prediction",
        "provider_failed",
        "unsupported",
    }
)
PREFILL_FIELDS = (
    "canonical_object",
    "category",
    "domain",
    "role",
    "explicit_material",
    "surface_alias",
    "filename_color_hints",
    "palette_colors",
    "primary_colors",
    "secondary_colors",
    "outline_colors",
    "highlight_colors",
    "shadow_colors",
    "size_hint",
    "condition",
    "shape",
    "visual_form",
    "parts",
    "description",
)
CRITICAL_FIELDS = (
    "canonical_object",
    "category",
    "domain",
    "role",
    "explicit_material",
    "primary_colors",
    "description",
)
MODEL_REQUIRED_CRITICAL_FIELDS = ("canonical_object", "category", "domain", "role")


class AuditSchemaError(ValueError):
    """Raised when a review/preparation boundary receives the wrong artifact."""


def detect_audit_schema(row: Mapping[str, Any]) -> str:
    schema = str(row.get("schema_version", ""))
    if schema in LEGACY_SELECTION_SCHEMAS:
        return AUDIT_SELECTION_SCHEMA
    if schema == PREFILLED_AUDIT_SCHEMA:
        return PREFILLED_AUDIT_SCHEMA
    if schema in {HUMAN_TRUTH_SCHEMA, "label_review_event_v4.1"}:
        return HUMAN_TRUTH_SCHEMA
    if schema.startswith("label_record_v4"):
        return "label_v4_prediction_record"
    return schema or "unknown"


def require_prefilled_records(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AuditSchemaError("assisted-v4 input has no records")
    detected = detect_audit_schema(rows[0])
    if detected != PREFILLED_AUDIT_SCHEMA:
        raise AuditSchemaError(
            f"Input schema is {detected}.\n"
            f"The assisted review GUI requires {PREFILLED_AUDIT_SCHEMA}.\n"
            "Run label-v4-prepare-audit first."
        )
    for index, row in enumerate(rows, 1):
        if detect_audit_schema(row) != PREFILLED_AUDIT_SCHEMA:
            raise AuditSchemaError(f"mixed input schema at record {index}")
        fields = row.get("fields")
        if not isinstance(fields, Mapping) or any(name not in fields for name in PREFILL_FIELDS):
            raise AuditSchemaError(f"prefilled record {index} does not contain every required field")
        for name in PREFILL_FIELDS:
            field = fields[name]
            if not isinstance(field, Mapping) or field.get("value_state") not in VALUE_STATES:
                raise AuditSchemaError(f"{row.get('sprite_id')}:{name} has no valid value_state")
            if field.get("value") is None and not str(field.get("reason", "")).strip():
                raise AuditSchemaError(f"{row.get('sprite_id')}:{name} null has no reason")


def prepare_audit(
    selection_path: str | Path,
    output_root: str | Path,
    *,
    inference_policy: str = "cached-only",
    allow_provider_calls: bool = False,
    artifact_roots: Sequence[str | Path] = (),
    vlm_provider: Any | None = None,
    text_provider: Any | None = None,
    bound_audit_selection: str | Path | None = None,
    bound_prefilled_records: str | Path | None = None,
    bound_human_truth: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare immutable-review projections; never constructs a provider implicitly."""

    if inference_policy not in INFERENCE_POLICIES:
        raise ValueError(f"invalid inference policy: {inference_policy}")
    if allow_provider_calls and inference_policy != "semantic-minimal":
        raise ValueError("--allow-provider-calls is valid only with semantic-minimal")
    if allow_provider_calls and (vlm_provider is None or text_provider is None):
        raise ValueError("semantic-minimal provider calls require explicitly configured VLM and text providers")
    source = Path(selection_path).resolve()
    before = _sha256_file(source)
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"semantic preparation output already exists: {output}")
    queue_verification: dict[str, Any] | None = None
    if source.name == "inference_queue.jsonl":
        from spritelab.harvest.label_v4.two_pass import verify_frozen_inference_queue

        queue_verification = verify_frozen_inference_queue(
            source,
            audit_selection=bound_audit_selection,
            prefilled_records=bound_prefilled_records,
            human_truth=bound_human_truth,
        )
        rows = list(queue_verification["rows"])
    else:
        rows = _read_jsonl(source)
    if not rows:
        raise AuditSchemaError("audit selection is empty")
    input_schema = detect_audit_schema(rows[0])
    queue_schema = "label_v4_inference_queue_v1"
    is_queue = str(rows[0].get("schema_version")) == queue_schema
    if is_queue and queue_verification is None:
        from spritelab.harvest.label_v4.two_pass import verify_frozen_inference_queue

        queue_verification = verify_frozen_inference_queue(
            source,
            audit_selection=bound_audit_selection,
            prefilled_records=bound_prefilled_records,
            human_truth=bound_human_truth,
        )
        rows = list(queue_verification["rows"])
    if input_schema != AUDIT_SELECTION_SCHEMA and not is_queue:
        raise AuditSchemaError(f"prepare input must use {AUDIT_SELECTION_SCHEMA} or {queue_schema}")
    if any(str(row.get("schema_version")) not in LEGACY_SELECTION_SCHEMAS | {queue_schema} for row in rows):
        raise AuditSchemaError("prepare input contains mixed or incompatible schemas")

    roots = tuple(Path(root).resolve() for root in artifact_roots)
    excluded_quality = [
        {**copy.deepcopy(row), "semantic_preparation_reason": f"ineligible_quality:{row.get('quality_state')}"}
        for row in rows
        if is_queue and str(row.get("quality_state")) not in {"quality_suitable", "quality_uncertain_usable"}
    ]
    eligible_rows = (
        [row for row in rows if row not in excluded_quality]
        if not is_queue
        else [row for row in rows if str(row.get("quality_state")) in {"quality_suitable", "quality_uncertain_usable"}]
    )
    cached = _compatible_rich_records(eligible_rows, roots) if inference_policy != "deterministic-only" else {}
    prepared: list[dict[str, Any]] = []
    for index, selection in enumerate(eligible_rows):
        stage_a = label_record_v4(selection, config=LabelV4PipelineConfig(mode="A", use_cache=False))
        candidate = cached.get(str(selection.get("sprite_id", "")))
        if candidate is not None and candidate.get("image_hash") == stage_a.get("image_hash"):
            prediction = candidate
            prediction_origin = "compatible_cached_rich_vlm"
        elif allow_provider_calls:
            prediction = label_record_v4(
                selection,
                config=LabelV4PipelineConfig(mode="B", use_cache=False),
                vlm_provider=vlm_provider,
                text_provider=text_provider,
            )
            prediction_origin = "semantic_minimal_provider"
        else:
            prediction = stage_a
            prediction_origin = "deterministic_stage_a"
        prepared.append(_prefill_record(selection, prediction, stage_a, index=index, origin=prediction_origin))

    if prepared:
        require_prefilled_records(prepared)
    elif not excluded_quality:
        raise AuditSchemaError("assisted-v4 input has no records")
    output.mkdir(parents=True)
    records_path = output / "audit_prefilled_records.jsonl"
    _write_jsonl(records_path, prepared)
    from spritelab.harvest.label_v4.two_pass import semantic_readiness

    semantic_ready = [row for row in prepared if semantic_readiness(row)[0]]
    semantic_failed = [
        row
        for row in prepared
        if not semantic_readiness(row)[0]
        and any((field or {}).get("value_state") == "provider_failed" for field in row.get("fields", {}).values())
    ]
    failed_ids = {str(row.get("sprite_id")) for row in semantic_failed}
    semantic_pending = [
        row for row in prepared if not semantic_readiness(row)[0] and str(row.get("sprite_id")) not in failed_ids
    ]
    semantic_excluded = excluded_quality
    partition_integrity = _assert_partition_integrity(
        rows, semantic_ready, semantic_pending, semantic_failed, semantic_excluded
    )
    _write_jsonl(output / "semantic_ready_records.jsonl", semantic_ready)
    _write_jsonl(output / "semantic_pending_inference.jsonl", semantic_pending)
    _write_jsonl(output / "semantic_failed_records.jsonl", semantic_failed)
    _write_jsonl(output / "semantic_excluded_quality.jsonl", semantic_excluded)
    coverage = summarize_prefill_coverage(prepared)
    manifest = {
        "schema_version": "label_v4_audit_prefill_manifest_v1",
        "input_schema": input_schema if input_schema != "unknown" else str(rows[0].get("schema_version")),
        "output_schema": PREFILLED_AUDIT_SCHEMA,
        "input_path": str(source),
        "input_sha256": before,
        "records": len(prepared),
        "semantic_ready_records": len(semantic_ready),
        "semantic_pending_inference": len(semantic_pending),
        "semantic_failed_records": len(semantic_failed),
        "semantic_excluded_quality": len(semantic_excluded),
        "inference_policy": inference_policy,
        "provider_calls_allowed": bool(allow_provider_calls),
        "provider_calls_made": sum(int(row.get("provider_calls_made", 0)) for row in prepared),
        "artifact_roots": [str(root) for root in roots],
        "coverage": coverage,
        "diagnostics": _diagnostics(prepared),
        "output_sha256": _sha256_file(records_path),
        "frozen_input_unchanged": before == _sha256_file(source),
        "queue_id": queue_verification["queue_id"] if queue_verification else None,
        "queue_file_sha256": queue_verification["queue_sha256"] if queue_verification else None,
        "queue_source_bindings_verified": bool(
            queue_verification and queue_verification.get("source_bindings_verified") is True
        ),
        "partition_integrity": partition_integrity,
    }
    _write_json(output / "audit_prefill_manifest.json", manifest)
    (output / "audit_prefill_report.md").write_text(_report_markdown(manifest), encoding="utf-8", newline="\n")
    if before != _sha256_file(source):
        raise RuntimeError("frozen audit selection changed during prefill")
    return manifest


def summarize_prefill_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "records_total": len(rows),
        "records_with_complete_deterministic_critical_semantics": sum(
            row.get("stage_a_critical_semantics_complete") is True for row in rows
        ),
        "records_with_compatible_cached_rich_vlm_predictions": sum(
            row.get("prediction_origin") == "compatible_cached_rich_vlm" for row in rows
        ),
        "records_requiring_missing_b_c_inference": sum(
            row.get("prediction_state") == "missing_required_model_stage" for row in rows
        ),
        "records_with_genuine_model_abstentions": sum(
            any(field.get("value_state") == "model_abstained" for field in row.get("fields", {}).values())
            for row in rows
        ),
        "records_with_quality_quarantine": sum(
            str(row.get("source_suitability", {}).get("status", "")) == "quarantine" for row in rows
        ),
    }


def _prefill_record(
    selection: Mapping[str, Any], prediction: Mapping[str, Any], stage_a: Mapping[str, Any], *, index: int, origin: str
) -> dict[str, Any]:
    semantics = dict(prediction.get("semantics") or {})
    reconciliation = dict(prediction.get("reconciliation") or {})
    proposals = dict(reconciliation.get("field_proposals") or {})
    risks = dict(prediction.get("field_risks") or {})
    stage_names = {str(item.get("stage")) for item in prediction.get("stage_ledger") or ()}
    has_rich = {"B_blind_vlm_proposal", "C_text_reconciliation"}.issubset(stage_names)
    stage_a_complete = bool(stage_a.get("stage_a_field_coverage", {}).get("critical_semantics_complete"))
    missing = [] if has_rich or stage_a_complete else ["B_blind_vlm_proposal", "C_text_reconciliation"]
    fields: dict[str, dict[str, Any]] = {}
    for name in PREFILL_FIELDS:
        value = _semantic_value(semantics, prediction, name)
        proposal = proposals.get(name) if isinstance(proposals.get(name), Mapping) else {}
        risk = risks.get(name) if isinstance(risks.get(name), Mapping) else {}
        state, reason = _value_state(name, value, has_rich=has_rich, missing=bool(missing), prediction=prediction)
        fields[name] = {
            "field": name,
            "value": copy.deepcopy(value),
            "value_state": state,
            "reason": reason,
            "alternatives": _alternatives(proposal, semantics, name),
            "evidence": _evidence(proposal, prediction, name, value),
            "uncertainty_1_20": risk.get("uncertainty_1_20"),
            "risk_band": risk.get("uncertainty_band", "not_scorable"),
            "conflict_disposition": _conflict_disposition(prediction, name),
            "training_consequence": risk.get("training_consequence")
            or risk.get("training_state")
            or "pending_human_audit",
        }
    record_risk = dict(prediction.get("record_risk") or {})
    field_proposals = {
        name: {
            "value": copy.deepcopy(field["value"]),
            "value_state": field["value_state"],
            "reason": field["reason"],
            "alternatives": copy.deepcopy(field["alternatives"]),
            "support": copy.deepcopy(field["evidence"]),
            "conflicts": [] if field["conflict_disposition"] == "none" else [field["conflict_disposition"]],
        }
        for name, field in fields.items()
    }
    quality_fields = {
        name: {
            "uncertainty_1_20": field["uncertainty_1_20"],
            "uncertainty_band": field["risk_band"],
            "training_consequence": field["training_consequence"],
            "calibration_state": "uncalibrated",
        }
        for name, field in fields.items()
    }
    record = {
        "schema_version": PREFILLED_AUDIT_SCHEMA,
        "sprite_id": str(selection.get("sprite_id", "")),
        "audit_id": str(selection.get("audit_id", "")),
        "image_path": str(prediction.get("image_path") or selection.get("image_path") or ""),
        "native_dimensions": copy.deepcopy(
            selection.get("native_dimensions")
            or {
                "width": prediction.get("deterministic_evidence", {}).get("pixels", {}).get("width"),
                "height": prediction.get("deterministic_evidence", {}).get("pixels", {}).get("height"),
            }
        ),
        "native_dimensions_meaning": "source_native_pre_transform",
        "decoded_exported_dimensions": {
            "width": prediction.get("deterministic_evidence", {}).get("pixels", {}).get("width"),
            "height": prediction.get("deterministic_evidence", {}).get("pixels", {}).get("height"),
        },
        "resize_policy": str(selection.get("resize_policy") or ""),
        "source_metadata": {
            key: copy.deepcopy(selection.get(key))
            for key in (
                "source_id",
                "pack_id",
                "pack_name",
                "source_sheet",
                "source_image",
                "archive_member",
                "author",
                "sub_artist",
            )
        },
        "source_suitability": {
            "status": selection.get("suitability_status", "unknown"),
            "reason_codes": list(selection.get("suitability_reason_codes") or ()),
            "score": selection.get("suitability_score"),
        },
        "suitability_decision": "pending",
        "suitability_reason_codes": list(selection.get("suitability_reason_codes") or ()),
        "review_mode": "blind" if index % 5 == 0 else "assisted",
        "proposal_visible_before_judgment": False if index % 5 == 0 else True,
        "prediction_state": "complete"
        if has_rich
        else "complete_deterministic"
        if stage_a_complete
        else "missing_required_model_stage",
        "missing_stages": missing,
        "provider_calls_allowed": bool(origin == "semantic_minimal_provider"),
        "provider_calls_made": int(prediction.get("new_provider_calls", 0) or 0)
        if origin == "semantic_minimal_provider"
        else 0,
        "prediction_origin": origin,
        "quality_risk_penalty": selection.get("quality_risk_penalty", 0.0),
        **{name: copy.deepcopy(selection.get(name)) for name in QUEUE_IDENTITY_FIELDS},
        "stage_a_critical_semantics_complete": stage_a_complete,
        "fields": fields,
        "field_proposals": field_proposals,
        "label_quality": {
            "record_uncertainty_1_20": record_risk.get("uncertainty_1_20"),
            "critical_field_max_uncertainty": max(
                (field["uncertainty_1_20"] for field in fields.values() if field["uncertainty_1_20"] is not None),
                default=None,
            ),
            "fields": quality_fields,
        },
        "record_risk": record_risk,
        "record_summary": {name: copy.deepcopy(fields[name]["value"]) for name in CRITICAL_FIELDS},
        "model_provenance": copy.deepcopy(dict(prediction)),
    }
    bind_model_stage_proof(record)
    return record


def bind_model_stage_proof(record: dict[str, Any]) -> None:
    """Bind successful model-stage evidence to this exact review projection."""

    provenance = record.get("model_provenance")
    if not isinstance(provenance, dict):
        return
    outcomes = {
        str(item.get("stage")): item for item in provenance.get("stage_outcomes") or () if isinstance(item, Mapping)
    }
    ledger = {
        str(item.get("stage")): item for item in provenance.get("stage_ledger") or () if isinstance(item, Mapping)
    }
    stage_artifacts: dict[str, dict[str, str]] = {}
    for stage in ("B_blind_vlm_proposal", "C_text_reconciliation"):
        artifact = ledger.get(stage)
        outcome = outcomes.get(stage)
        if artifact is None or outcome is None:
            continue
        stage_artifacts[stage] = {
            "artifact_sha256": _canonical_hash(artifact),
            "stage_status": str(outcome.get("stage_status") or ""),
            "proposal_input_hash": str(provenance.get("image_hash") or ""),
        }
    provenance["review_record_binding"] = {
        "audit_record_id": str(record.get("audit_id") or ""),
        "sprite_id": str(record.get("sprite_id") or ""),
        "exported_rgba_hash": str(record.get("exported_rgba_hash") or ""),
        "proposal_hash": _canonical_hash(record.get("field_proposals") or {}),
        "proposal_input_hash": str(provenance.get("image_hash") or ""),
        "queue_id": str(record.get("queue_id") or ""),
        "prefill_record_hash": str(record.get("prefill_record_hash") or ""),
        "stage_artifacts": stage_artifacts,
    }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _semantic_value(semantics: Mapping[str, Any], prediction: Mapping[str, Any], name: str) -> Any:
    colors = semantics.get("colors") if isinstance(semantics.get("colors"), Mapping) else {}
    shape = semantics.get("shape") if isinstance(semantics.get("shape"), Mapping) else {}
    if name in {
        "filename_color_hints",
        "palette_colors",
        "primary_colors",
        "secondary_colors",
        "outline_colors",
        "highlight_colors",
        "shadow_colors",
    }:
        if name == "filename_color_hints":
            return list(prediction.get("deterministic_evidence", {}).get("filename", {}).get(name) or ())
        return list(colors.get(name) or ())
    if name == "shape":
        return copy.deepcopy(shape)
    if name == "parts":
        return list(shape.get("parts") or ())
    value = semantics.get(name, prediction.get(name))
    return None if value == "unknown" else copy.deepcopy(value)


def _value_state(
    name: str, value: Any, *, has_rich: bool, missing: bool, prediction: Mapping[str, Any]
) -> tuple[str, str]:
    if value not in (None, "", [], {}):
        return "known", "normalized_available_evidence"
    outcomes = {
        str(item.get("stage")): str(item.get("stage_status")) for item in prediction.get("stage_outcomes") or ()
    }
    failed = any(
        item.get("failure_diagnostics")
        and outcomes.get(str(item.get("stage"))) not in {"success_after_json_repair", "deterministic_fallback"}
        for item in prediction.get("stage_ledger") or ()
    )
    if failed:
        return "provider_failed", "required_provider_stage_failed"
    if missing and name in (
        *MODEL_REQUIRED_CRITICAL_FIELDS,
        "surface_alias",
        "description",
        "condition",
        "size_hint",
        "visual_form",
    ):
        return "missing_prediction", "rich_vlm_stage_not_executed"
    if has_rich and name in (
        *MODEL_REQUIRED_CRITICAL_FIELDS,
        "surface_alias",
        "description",
        "condition",
        "size_hint",
        "visual_form",
    ):
        return "model_abstained", "model_stage_completed_without_promoted_value"
    if name in {
        "filename_color_hints",
        "secondary_colors",
        "outline_colors",
        "highlight_colors",
        "shadow_colors",
        "parts",
    }:
        return "not_applicable", "no_supported_value_for_optional_field"
    if name == "explicit_material":
        return "unsupported", "material_applicability_not_established"
    return "unsupported", "available_pipeline_stages_do_not_produce_field"


def _alternatives(proposal: Mapping[str, Any], semantics: Mapping[str, Any], name: str) -> list[Any]:
    result = [
        copy.deepcopy(item.get("value") if isinstance(item, Mapping) and "value" in item else item)
        for item in proposal.get("alternatives") or ()
    ]
    if name == "canonical_object":
        result.extend(semantics.get("canonical_object_alternatives") or ())
    return _dedupe(result)


def _dedupe(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _evidence(proposal: Mapping[str, Any], prediction: Mapping[str, Any], name: str, value: Any) -> list[str]:
    result = [str(item) for item in proposal.get("support") or proposal.get("evidence_refs") or ()]
    if name in {"filename_color_hints", "palette_colors", "shape", "parts"}:
        result.append("deterministic_pixels" if name != "filename_color_hints" else "deterministic_filename")
    if value not in (None, "", [], {}) and not result:
        result.append("normalized_pipeline_evidence")
    return list(dict.fromkeys(result))


def _conflict_disposition(prediction: Mapping[str, Any], name: str) -> str:
    conflicts = prediction.get("unresolved_conflicts") or ()
    relevant = [item for item in conflicts if not isinstance(item, Mapping) or item.get("field") in {None, "", name}]
    return "none" if not relevant else "unresolved_requires_human_review"


def _compatible_rich_records(
    selection: Sequence[Mapping[str, Any]], roots: Sequence[Path]
) -> dict[str, dict[str, Any]]:
    wanted = {str(row.get("sprite_id", "")) for row in selection}
    result: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            try:
                rows = _read_jsonl(path)
            except (OSError, json.JSONDecodeError):
                continue
            for row in rows:
                sprite_id = str(row.get("sprite_id", ""))
                if sprite_id not in wanted or not str(row.get("schema_version", "")).startswith("label_record_v4"):
                    continue
                stages = {str(item.get("stage")): item for item in row.get("stage_ledger") or ()}
                b, c = stages.get("B_blind_vlm_proposal"), stages.get("C_text_reconciliation")
                outcomes = {str(item.get("stage")): item for item in row.get("stage_outcomes") or ()}
                repaired_c = (
                    outcomes.get("C_text_reconciliation", {}).get("stage_status") == "success_after_json_repair"
                )
                if not b or not c or b.get("failure_diagnostics") or (c.get("failure_diagnostics") and not repaired_c):
                    continue
                vlm = row.get("vlm_proposal")
                artifact = row.get("reconciliation_provider_artifact")
                identities = " ".join(
                    str(value.get("model_identity", "")) for value in (vlm, artifact) if isinstance(value, Mapping)
                ).lower()
                if "mock" in identities or not isinstance(vlm, Mapping) or not isinstance(artifact, Mapping):
                    continue
                result.setdefault(sprite_id, copy.deepcopy(row))
    return result


def _report_markdown(manifest: Mapping[str, Any]) -> str:
    coverage = manifest["coverage"]
    lines = [
        "# Labeling-v4 calibration audit prefill",
        "",
        "The GUI consumes `label_v4_prefilled_audit_record_v1`; the frozen audit manifest is selection metadata, not a prediction record.",
        "",
        f"Inference policy: `{manifest['inference_policy']}`. Provider calls allowed: `{str(manifest['provider_calls_allowed']).lower()}`. Provider calls made: `{manifest['provider_calls_made']}`.",
        "",
        "## Current coverage",
        "",
    ]
    labels = {
        "records_total": "Records total",
        "records_with_complete_deterministic_critical_semantics": "Records with complete deterministic critical semantics",
        "records_with_compatible_cached_rich_vlm_predictions": "Records with compatible cached rich-VLM predictions",
        "records_requiring_missing_b_c_inference": "Records requiring missing B/C inference",
        "records_with_genuine_model_abstentions": "Records with genuine model abstentions",
        "records_with_quality_quarantine": "Records with quality quarantine",
    }
    lines.extend(f"- {labels[key]}: {coverage[key]}" for key in labels)
    mineral = dict(manifest.get("diagnostics", {}).get("acq_craftpix_minerals_icon29") or {})
    lines.extend(
        [
            "",
            "## Reported mineral record",
            "",
            f"`acq_craftpix_minerals_icon29` resolved to `{mineral.get('image_path', '')}`. "
            f"Its prediction state is `{mineral.get('prediction_state', 'unknown')}` and its canonical-object state is "
            f"`{mineral.get('canonical_object_value_state', 'unknown')}` because `{mineral.get('canonical_object_reason', 'unknown')}`. "
            "The original GUI null was a raw-selection/prediction schema mismatch plus a missing prefill and missing B/C stages, "
            "not a genuine abstention and not a resolver failure.",
            "",
            "Missing model stages are represented as `missing_prediction`, never as semantic abstention.",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        sprite_id = str(row.get("sprite_id", ""))
        if sprite_id != "acq_craftpix_minerals_icon29":
            continue
        canonical = row.get("fields", {}).get("canonical_object", {})
        result[sprite_id] = {
            "image_path": row.get("image_path"),
            "resolver_succeeded": bool(row.get("image_path")),
            "prediction_state": row.get("prediction_state"),
            "missing_stages": row.get("missing_stages", []),
            "canonical_object_value_state": canonical.get("value_state"),
            "canonical_object_reason": canonical.get("reason"),
            "diagnosis": "schema_mismatch_missing_prefill_and_missing_model_stage",
        }
    return result


def _id_set_hash(values: set[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_partition_integrity(
    cohort: Sequence[Mapping[str, Any]],
    ready: Sequence[Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
    failed: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless all preparation partitions are disjoint and exhaustive."""

    cohort_ids = [str(row.get("sprite_id") or "") for row in cohort]
    if not all(cohort_ids) or len(set(cohort_ids)) != len(cohort_ids):
        raise AuditSchemaError("semantic preparation cohort has missing or duplicate sprite IDs")
    source = {str(row["sprite_id"]): row for row in cohort}
    partitions = {"ready": ready, "pending": pending, "failed": failed, "excluded": excluded}
    sets: dict[str, set[str]] = {}
    for name, rows in partitions.items():
        ids = [str(row.get("sprite_id") or "") for row in rows]
        if not all(ids) or len(set(ids)) != len(ids):
            raise AuditSchemaError(f"semantic {name} partition has missing or duplicate sprite IDs")
        sets[name] = set(ids)
        for row in rows:
            original = source.get(str(row["sprite_id"]))
            if original is None:
                raise AuditSchemaError(f"semantic {name} partition contains a foreign record")
            for field in QUEUE_IDENTITY_FIELDS:
                if field in original and row.get(field) != original.get(field):
                    raise AuditSchemaError(f"semantic {name} partition lost identity field {field}")
    names = tuple(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AuditSchemaError(f"semantic partitions {left}/{right} overlap: {sorted(overlap)}")
    union = set().union(*sets.values())
    if union != set(cohort_ids):
        raise AuditSchemaError("semantic preparation partitions do not exhaust the validated cohort")
    return {
        "cohort_count": len(cohort_ids),
        "cohort_ids_sha256": _id_set_hash(set(cohort_ids)),
        "partition_counts": {name: len(values) for name, values in sets.items()},
        "partition_id_set_sha256": {name: _id_set_hash(values) for name, values in sets.items()},
        "disjoint": True,
        "exhaustive": True,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
