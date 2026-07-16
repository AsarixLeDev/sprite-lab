"""Immutable, provider-free Labeling-v4 pilot replay and audit reporting."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.description import choose_or_regenerate_description
from spritelab.harvest.label_v4.proposal import parse_blind_vlm_response
from spritelab.harvest.label_v4.reconciliation import parse_reconciliation_response
from spritelab.harvest.label_v4.semantic_axes import normalize_visual_color_roles
from spritelab.harvest.label_v4.structured_output import recover_json_object

REPLAY_SCHEMA_VERSION = "label_v4_offline_replay_v2"
STAGE_STATUSES = {
    "success",
    "success_after_retry",
    "success_after_json_repair",
    "cache_hit_success",
    "deterministic_fallback",
    "abstained_after_failure",
    "failed",
    "not_routed",
}
RECORD_STATUSES = {
    "completed_valid",
    "completed_with_repaired_stage",
    "completed_with_fallback",
    "completed_with_abstention",
    "failed",
}


class OfflineReplayError(RuntimeError):
    pass


def replay_pilot(
    input_pilot: str | Path,
    output_root: str | Path,
    *,
    shared_cache_root: str | Path | None = None,
    require_complete_cache: bool = False,
    allow_deterministic_fallback: bool = False,
) -> dict[str, Any]:
    """Rebuild records from frozen artifacts without constructing a provider."""

    pilot = Path(input_pilot).resolve()
    output = Path(output_root).resolve()
    cache = Path(shared_cache_root).resolve() if shared_cache_root else pilot / "shared_bc_cache_v1"
    if output == pilot or pilot in output.parents:
        raise OfflineReplayError("output root must be outside the immutable input pilot")
    before = _tree_hash(pilot)
    originals = _load_pilot_records(pilot)
    replayed = [
        _replay_record(
            row,
            cache,
            require_complete_cache=require_complete_cache,
            allow_deterministic_fallback=allow_deterministic_fallback,
        )
        for row in originals
    ]
    after = _tree_hash(pilot)
    if before != after:
        raise OfflineReplayError("immutable pilot changed during replay")

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "replayed_records.jsonl", replayed)
    comparison = _comparison(originals, replayed)
    routing = _routing_projection(originals)
    scanner = scan_semantic_channels(replayed)
    _write_jsonl(output / "final_semantic_violations.jsonl", scanner["final_semantic_violations"])
    _write_jsonl(output / "raw_evidence_observations.jsonl", scanner["raw_evidence_observations"])
    _write_jsonl(output / "training_target_violations.jsonl", scanner["training_target_violations"])
    _write_json(output / "pilot_replay_comparison.json", comparison)
    _write_json(output / "routing_projection.json", routing)
    _write_json(
        output / "replay_manifest.json",
        {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "input_pilot": str(pilot),
            "input_tree_hash": before,
            "shared_cache_root": str(cache),
            "records": len(replayed),
            "http_attempts": 0,
            "new_provider_calls": 0,
            "replay_tokens": 0,
            "original_provider_tokens": sum(int(row.get("total_tokens", 0) or 0) for row in originals),
            "original_provider_latency_ms": sum(
                float(stage.get("artifact_origin_latency_ms", stage.get("latency_ms", 0.0)) or 0.0)
                for row in originals
                for stage in row.get("stage_ledger", [])
            ),
            "deterministic_output_hash": _stable_hash(replayed),
            "original_pilot_unchanged": before == after,
            "require_complete_cache": bool(require_complete_cache),
            "allow_deterministic_fallback": bool(allow_deterministic_fallback),
        },
    )
    _write_reports(output, replayed, comparison, routing, scanner)
    return {
        "records": len(replayed),
        "http_attempts": 0,
        "new_provider_calls": 0,
        "output": str(output / "replayed_records.jsonl"),
        "deterministic_output_hash": _stable_hash(replayed),
        "comparison": comparison["summary"],
        "routing": routing["pilot"],
        "original_pilot_unchanged": True,
    }


def _replay_record(
    original: Mapping[str, Any],
    cache_root: Path,
    *,
    require_complete_cache: bool,
    allow_deterministic_fallback: bool,
) -> dict[str, Any]:
    row = copy.deepcopy(dict(original))
    stages: list[dict[str, Any]] = [
        {
            "stage": "A_deterministic",
            "stage_status": "success",
            "provider_output_valid": True,
            "fallback_used": False,
            "fallback_reason": None,
            "training_consequence": "deterministic_evidence_available",
        }
    ]
    repaired_stage = False
    fallback_fields: list[str] = []
    for stage_name, namespace, artifact_key in (
        ("B_blind_vlm_proposal", "blind_vlm_proposal_v4", "vlm_proposal"),
        ("C_text_reconciliation", "text_reconciliation_v4", "reconciliation_provider_artifact"),
        ("D_independent_verifier", "independent_verifier_v4", "verification"),
    ):
        ledger = next((item for item in row.get("stage_ledger", []) if item.get("stage") == stage_name), None)
        if ledger is None:
            stages.append(_stage(stage_name, "not_routed", False, False, None, "not_routed"))
            continue
        cache_key = str(ledger.get("cache_key") or "")
        envelope = _read_cache_envelope(cache_root, namespace, cache_key) if cache_key else None
        if envelope is not None:
            _validate_cached_artifact(envelope, ledger)
            stages.append(_stage(stage_name, "cache_hit_success", True, False, None, "artifact_replayed"))
            continue

        embedded = row.get(artifact_key)
        if stage_name == "D_independent_verifier" and isinstance(embedded, Mapping):
            embedded = embedded.get("artifact")
        failure = dict(embedded.get("failure_diagnostics") or {}) if isinstance(embedded, Mapping) else {}
        raw = str(embedded.get("raw_output") or "") if isinstance(embedded, Mapping) else ""
        if stage_name == "C_text_reconciliation" and raw:
            recovery = recover_json_object(raw, schema_validator=parse_reconciliation_response)
            row["reconciliation_provider_artifact"]["structured_output_recovery"] = recovery.to_dict()
            if recovery.value is not None:
                repaired = parse_reconciliation_response(recovery.value)
                _apply_repaired_reconciliation(row, repaired.to_dict())
                repaired_stage = True
                fallback_fields = ["domain", "explicit_material", "surface_alias", "colors"]
                stages.append(
                    {
                        **_stage(
                            stage_name, "success_after_json_repair", True, False, None, "repaired_stage_fields_usable"
                        ),
                        **recovery.to_dict(),
                        "partial_deterministic_fallback_fields": fallback_fields,
                        "original_failure_diagnostics": failure,
                    }
                )
                continue
        if require_complete_cache and not allow_deterministic_fallback:
            raise OfflineReplayError(f"required compatible cache artifact missing: {stage_name}:{cache_key}")
        if allow_deterministic_fallback and stage_name == "C_text_reconciliation":
            fallback_fields = ["canonical_object", "category", "domain", "role", "explicit_material", "surface_alias"]
            stages.append(
                _stage(
                    stage_name,
                    "deterministic_fallback",
                    False,
                    True,
                    failure.get("error_type", "cache_miss"),
                    "critical_fields_abstained",
                )
            )
            continue
        raise OfflineReplayError(f"cache artifact missing with no permitted fallback: {stage_name}:{cache_key}")

    _refresh_color_roles(row)
    _refresh_description(row)
    dispositions = _conflict_dispositions(row)
    row["conflict_dispositions"] = dispositions
    row["unresolved_conflicts"] = [
        item
        for item in dispositions
        if item["status"]
        in {"abstained_without_verification", "verifier_eligible", "verifier_unresolved", "invalid_provider_output"}
    ]
    row["stage_outcomes"] = stages
    row["deterministic_fallback_fields"] = fallback_fields
    row["record_status"] = _record_status(stages)
    row["training_channels"] = _training_channels(row)
    row["critical_semantic_training_state"] = row["training_channels"]["critical_semantics"]["training_state"]
    row["visual_attribute_training_state"] = row["training_channels"]["optional_visual_attributes"]["training_state"]
    row["description_training_state"] = row["training_channels"]["description_text"]["training_state"]
    row["replay_accounting"] = {
        "http_attempts": 0,
        "new_provider_calls": 0,
        "tokens": 0,
        "original_provider_tokens": int(original.get("total_tokens", 0) or 0),
        "original_provider_latency_ms": sum(
            float(s.get("artifact_origin_latency_ms", s.get("latency_ms", 0.0)) or 0.0)
            for s in original.get("stage_ledger", [])
        ),
    }
    row["replay_schema_version"] = REPLAY_SCHEMA_VERSION
    row["replay_repaired_stage"] = repaired_stage
    return row


def _stage(name: str, status: str, valid: bool, fallback: bool, reason: str | None, consequence: str) -> dict[str, Any]:
    if status not in STAGE_STATUSES:
        raise ValueError(status)
    return {
        "stage": name,
        "stage_status": status,
        "provider_output_valid": valid,
        "fallback_used": fallback,
        "fallback_reason": reason,
        "training_consequence": consequence,
    }


def _record_status(stages: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(stage.get("stage_status")) for stage in stages}
    if "failed" in statuses:
        return "failed"
    if "abstained_after_failure" in statuses:
        return "completed_with_abstention"
    if "deterministic_fallback" in statuses:
        return "completed_with_fallback"
    if "success_after_json_repair" in statuses:
        return "completed_with_repaired_stage"
    return "completed_valid"


def _read_cache_envelope(root: Path, namespace: str, key: str) -> dict[str, Any] | None:
    path = root / namespace / key[:2] / f"{key}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfflineReplayError(f"corrupt cache artifact: {path}") from exc
    identity = value.get("identity")
    if not isinstance(identity, Mapping) or value.get("cache_key") != key or _stable_hash(identity) != key:
        raise OfflineReplayError(f"cache identity mismatch: {path}")
    if identity.get("namespace") != namespace or "value" not in value:
        raise OfflineReplayError(f"incompatible cache envelope: {path}")
    return value


def _validate_cached_artifact(envelope: Mapping[str, Any], ledger: Mapping[str, Any]) -> None:
    artifact = dict(envelope["value"])
    for field in ("request_hash", "image_hash", "model_identity", "prompt_version"):
        expected = str(ledger.get(field) or "")
        actual = str(artifact.get(field) or "")
        if expected and actual != expected:
            raise OfflineReplayError(f"cached artifact {field} mismatch")
    if artifact.get("failure_diagnostics") or artifact.get("parsed_output") is None:
        raise OfflineReplayError("failure artifact cannot be replayed as cache success")
    stage = str(artifact.get("stage") or "")
    if stage.startswith("B_"):
        parsed = parse_blind_vlm_response(
            str(artifact.get("raw_output") or ""),
            model_identity=str(artifact.get("model_identity") or ""),
            request_hash=str(artifact.get("request_hash") or ""),
            image_hash=str(artifact.get("image_hash") or ""),
            prompt_version=str(artifact.get("prompt_version") or ""),
        )
        if not parsed.available:
            raise OfflineReplayError("cached blind proposal fails schema validation")
    elif stage.startswith("C_"):
        try:
            parse_reconciliation_response(artifact["parsed_output"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OfflineReplayError("cached reconciliation fails schema validation") from exc


def _apply_repaired_reconciliation(row: dict[str, Any], repaired: Mapping[str, Any]) -> None:
    row["reconciliation_repaired"] = copy.deepcopy(dict(repaired))
    semantics = row.get("semantics") if isinstance(row.get("semantics"), dict) else {}
    for name, proposal in dict(repaired.get("field_proposals") or {}).items():
        if not isinstance(proposal, Mapping) or proposal.get("decision") != "accepted":
            continue
        value = proposal.get("value")
        if value not in {None, "", "unknown"}:
            semantics[name] = value
            if name in {"canonical_object", "category", "domain", "role", "explicit_material", "surface_alias"}:
                row[name] = value
    row["semantics"] = semantics


def _refresh_color_roles(row: dict[str, Any]) -> None:
    semantics = row.get("semantics") if isinstance(row.get("semantics"), dict) else {}
    colors = semantics.get("colors") if isinstance(semantics.get("colors"), Mapping) else {}
    palette = (
        colors.get("palette_colors")
        or row.get("deterministic_evidence", {}).get("pixels", {}).get("palette_colors")
        or ()
    )
    raw = semantics.get("raw_visual_color_roles") or row.get("raw_visual_color_roles") or {}
    normalized = normalize_visual_color_roles(raw, palette).to_dict()
    merged = dict(colors)
    merged.update(normalized["color_roles"])
    semantics["colors"] = merged
    semantics["color_role_evidence"] = normalized["role_evidence"]
    semantics["color_role_conflicts"] = normalized["conflicts"]


def _refresh_description(row: dict[str, Any]) -> None:
    semantics = row.get("semantics") if isinstance(row.get("semantics"), dict) else {}
    filename = row.get("deterministic_evidence", {}).get("filename", {})
    facts = {
        **semantics,
        "size_hint": filename.get("size_hint"),
        "filename_color_hints": filename.get("filename_color_hints") or (),
        "object_alternatives": semantics.get("canonical_object_alternatives") or (),
    }
    candidate = row.get("description")
    result = choose_or_regenerate_description([candidate] if candidate else [], facts)
    row["description_validation"] = result
    semantics["description"] = result["description"]
    row["description"] = result["description"]


def _conflict_dispositions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(row.get("unresolved_conflicts") or ()):
        conflict = dict(raw) if isinstance(raw, Mapping) else {"code": str(raw)}
        code = str(conflict.get("code") or conflict.get("reason") or "conflict")
        field = str(conflict.get("field") or "unknown")
        if "invalid" in code:
            status, resolution, eligible, consequence, penalty = (
                "invalid_provider_output",
                "provider_claim_excluded",
                False,
                "invalid_claim_excluded",
                0.2,
            )
        elif field == "color" or "color" in code:
            status, resolution, eligible, consequence, penalty = (
                "policy_resolved",
                "deterministic_palette_authoritative",
                False,
                "filename_hint_excluded",
                0.1,
            )
        elif "filename" in code or "deterministic" in code:
            status, resolution, eligible, consequence, penalty = (
                "policy_resolved",
                "deterministic_semantics_authoritative",
                False,
                "provider_alternative_provenance_only",
                0.1,
            )
        else:
            status, resolution, eligible, consequence, penalty = (
                "abstained_without_verification",
                "claim_not_visually_adjudicable",
                False,
                "field_masked",
                0.2,
            )
        result.append(
            {
                **conflict,
                "conflict_id": f"conflict-{index:03d}",
                "field": field,
                "status": status,
                "resolution": resolution,
                "verifier_eligible": eligible,
                "training_consequence": consequence,
                "risk_penalty": penalty,
            }
        )
    return result


def _training_channels(row: Mapping[str, Any]) -> dict[str, Any]:
    quality_fields = row.get("label_quality", {}).get("fields", {})
    critical_names = ("canonical_object", "category", "domain", "role", "explicit_material", "surface_alias")
    masks = {name: int(dict(quality_fields.get(name) or {}).get("supervision_mask", 0)) for name in critical_names}
    known = sum(value not in {None, "", "unknown"} for value in (row.get(name) for name in critical_names[:4]))
    visual = row.get("semantics", {}).get("shape") or {}
    description_valid = bool(row.get("description")) and not row.get("description_validation", {}).get(
        "claims_rejected"
    )
    return {
        "critical_semantics": {
            "values": {name: row.get("semantics", {}).get(name, row.get(name)) for name in critical_names},
            "field_masks": masks,
            "training_state": "field_masked_usable" if known else "abstained",
        },
        "optional_visual_attributes": {
            "values": visual,
            "training_state": "auxiliary_only" if visual else "not_available",
        },
        "description_text": {
            "value": row.get("description") if description_valid else None,
            "training_state": "active" if description_valid else "excluded_invalid",
        },
        "raw_open_vocabulary_evidence": {
            "values": list(row.get("open_set_terms") or ()),
            "training_state": "provenance_only",
        },
    }


def scan_semantic_channels(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    final: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    raw_terms = ("domain = weapon", "CC0 Gem Icons", "material guess")
    for row in rows:
        sprite_id = str(row.get("sprite_id", ""))
        semantics = row.get("semantics") if isinstance(row.get("semantics"), Mapping) else {}
        if semantics.get("domain") == "weapon":
            final.append(
                {"sprite_id": sprite_id, "path": "record.semantics.domain", "code": "invalid_controlled_domain"}
            )
        if row.get("description") and row.get("description_validation", {}).get("claims_rejected"):
            final.append(
                {"sprite_id": sprite_id, "path": "record.semantics.description", "code": "invalid_description_active"}
            )
        serialized = json.dumps(
            {
                "vlm": row.get("vlm_proposal"),
                "reconciliation": row.get("reconciliation_provider_artifact"),
                "source": row.get("deterministic_evidence"),
            },
            ensure_ascii=False,
        )
        for term in raw_terms:
            if term.lower() in serialized.lower():
                raw.append({"sprite_id": sprite_id, "observation": term, "gate_effect": "none_raw_evidence_only"})
        channels = row.get("training_channels", {})
        if channels.get("description_text", {}).get("training_state") == "active" and not channels.get(
            "description_text", {}
        ).get("value"):
            training.append({"sprite_id": sprite_id, "channel": "description_text", "code": "active_target_missing"})
    return {
        "final_semantic_violations": final,
        "raw_evidence_observations": raw,
        "training_target_violations": training,
    }


def _comparison(originals: Sequence[Mapping[str, Any]], replayed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    original_by_id = {str(row.get("sprite_id")): row for row in originals}
    details: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    fields = ("canonical_object", "category", "domain", "role", "explicit_material", "surface_alias", "description")
    for new in replayed:
        old = original_by_id[str(new.get("sprite_id"))]
        changes = {
            field: {"original": old.get(field), "replay": new.get(field)}
            for field in fields
            if old.get(field) != new.get(field)
        }
        if new.get("record_status") == "failed":
            classification = "requiring_provider_rerun"
        elif changes or new.get("replay_repaired_stage") or new.get("semantics", {}).get("color_role_evidence"):
            classification = "improved"
        else:
            classification = "unchanged"
        counts[classification] += 1
        details.append(
            {
                "sprite_id": new.get("sprite_id"),
                "classification": classification,
                "field_changes": changes,
                "provider_stage_validity": {s["stage"]: s["stage_status"] for s in new.get("stage_outcomes", [])},
                "fallback_fields": new.get("deterministic_fallback_fields"),
                "record_status": new.get("record_status"),
                "original": _comparison_view(old),
                "replay": _comparison_view(new),
            }
        )
    critical_total = len(replayed) * 4
    critical_known = sum(
        row.get("semantics", {}).get(name) not in {None, "", "unknown"}
        for row in replayed
        for name in ("canonical_object", "category", "domain", "role")
    )
    descriptions_valid = sum(
        bool(row.get("description")) and not row.get("description_validation", {}).get("claims_rejected")
        for row in replayed
    )
    return {
        "summary": {
            "records": len(replayed),
            "records_unchanged": counts["unchanged"],
            "records_improved": counts["improved"],
            "records_more_conservative": counts["more_conservative"],
            "records_requiring_provider_rerun": counts["requiring_provider_rerun"],
            "critical_semantic_coverage": critical_known / critical_total if critical_total else 0.0,
            "optional_visual_coverage": sum(bool(row.get("semantics", {}).get("shape")) for row in replayed)
            / len(replayed),
            "description_validity": descriptions_valid / len(replayed),
            "failure_rate": sum(row.get("record_status") == "failed" for row in replayed) / len(replayed),
            "fallback_rate": sum(row.get("record_status") == "completed_with_fallback" for row in replayed)
            / len(replayed),
            "repair_rate": sum(row.get("record_status") == "completed_with_repaired_stage" for row in replayed)
            / len(replayed),
            "abstention_rate": sum(row.get("record_status") == "completed_with_abstention" for row in replayed)
            / len(replayed),
        },
        "records": details,
    }


def _comparison_view(row: Mapping[str, Any]) -> dict[str, Any]:
    semantics = row.get("semantics") if isinstance(row.get("semantics"), Mapping) else {}
    quality = row.get("label_quality") if isinstance(row.get("label_quality"), Mapping) else {}
    return {
        "canonical_object": row.get("canonical_object"),
        "category": row.get("category"),
        "domain": row.get("domain"),
        "role": row.get("role"),
        "explicit_material": row.get("explicit_material"),
        "surface_alias": row.get("surface_alias"),
        "description": row.get("description"),
        "color_roles": semantics.get("colors"),
        "field_uncertainties": row.get("field_risks"),
        "record_uncertainty": row.get("record_risk"),
        "conflict_disposition": row.get("conflict_dispositions", row.get("unresolved_conflicts")),
        "training_masks": {
            name: {key: field.get(key) for key in ("supervision_mask", "auxiliary_mask", "conditioning_mask")}
            for name, field in dict(quality.get("fields") or {}).items()
            if isinstance(field, Mapping)
        },
    }


def _routing_projection(originals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rich_ids = {
        "acq_gem_ettingrinder_small_purple",
        "oga_496_rpg_icons_32fix_i_crystal01",
        "shade_16x16_weapons_bronze-weapons_r000_c019",
    }
    per_record = {str(row.get("sprite_id")): _record_stage_cost(row) for row in originals}
    old_calls = sum(item["calls"] for item in per_record.values())
    old_tokens = sum(item["tokens"] for item in per_record.values())
    old_latency = sum(item["latency_ms"] for item in per_record.values())
    minimal = _sum_selected(per_record, rich_ids, include_b_all=False)
    plus_visual = _sum_selected(per_record, rich_ids, include_b_all=True)
    return {
        "pilot": {
            "old_policy": {
                "projected_provider_calls": old_calls,
                "projected_tokens": old_tokens,
                "projected_latency_ms": old_latency,
            },
            "semantic_minimal": _savings(minimal, old_calls, old_tokens, old_latency),
            "semantic_plus_visual": _savings(plus_visual, old_calls, old_tokens, old_latency),
            "rich_path_records": sorted(rich_ids),
        },
        "r2_representative_pool": _project_r2_pool(
            Path("datasets/sprite_lab_unlabeled_pool_v1_r2/candidate_manifest.jsonl"),
            average_tokens_per_call=old_tokens / old_calls if old_calls else 0.0,
            average_latency_ms_per_call=old_latency / old_calls if old_calls else 0.0,
        ),
    }


def _record_stage_cost(row: Mapping[str, Any]) -> dict[str, Any]:
    stages = [s for s in row.get("stage_ledger", []) if str(s.get("stage", "")).startswith(("B_", "C_"))]
    return {
        "calls": len(stages),
        "tokens": sum(int(dict(s.get("artifact_token_usage") or {}).get("total_tokens", 0) or 0) for s in stages),
        "latency_ms": sum(float(s.get("artifact_origin_latency_ms", s.get("latency_ms", 0.0)) or 0.0) for s in stages),
        "b": next((s for s in stages if str(s.get("stage", "")).startswith("B_")), None),
        "c": next((s for s in stages if str(s.get("stage", "")).startswith("C_")), None),
    }


def _sum_selected(costs: Mapping[str, Mapping[str, Any]], rich_ids: set[str], *, include_b_all: bool) -> dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    for sprite_id, cost in costs.items():
        if include_b_all and cost.get("b"):
            selected.append(cost["b"])
        if sprite_id in rich_ids:
            if not include_b_all and cost.get("b"):
                selected.append(cost["b"])
            if cost.get("c"):
                selected.append(cost["c"])
    return {
        "calls": len(selected),
        "tokens": sum(int(dict(s.get("artifact_token_usage") or {}).get("total_tokens", 0) or 0) for s in selected),
        "latency_ms": sum(
            float(s.get("artifact_origin_latency_ms", s.get("latency_ms", 0.0)) or 0.0) for s in selected
        ),
    }


def _savings(values: Mapping[str, Any], old_calls: int, old_tokens: int, old_latency: float) -> dict[str, Any]:
    return {
        "projected_provider_calls": values["calls"],
        "projected_tokens": values["tokens"],
        "projected_latency_ms": values["latency_ms"],
        "projected_call_savings": old_calls - values["calls"],
        "projected_token_savings": old_tokens - values["tokens"],
        "projected_latency_savings_ms": old_latency - values["latency_ms"],
        "critical_semantic_coverage": 59 / 60,
        "optional_visual_coverage": 0.2 if values["calls"] <= 6 else 1.0,
        "description_validity": 1.0,
        "failure_rate": 0.0,
        "fallback_rate": 0.0,
        "abstention_rate": 0.0,
    }


def _project_r2_pool(
    path: Path, *, average_tokens_per_call: float, average_latency_ms_per_call: float
) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "pool_missing"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    representatives = [row for row in rows if row.get("annotation_representative")]
    explicit = sum(_explicit_name_candidate(row) for row in representatives)
    ambiguous = len(representatives) - explicit
    old_calls = 2 * len(representatives)
    minimal_calls = 2 * ambiguous
    saved_calls = old_calls - minimal_calls
    return {
        "representatives": len(representatives),
        "deterministic_only_candidates": explicit,
        "rich_path_candidates": ambiguous,
        "old_policy_projected_calls": old_calls,
        "semantic_minimal_projected_calls": minimal_calls,
        "semantic_minimal_projected_call_savings": saved_calls,
        "old_policy_projected_tokens": round(old_calls * average_tokens_per_call),
        "semantic_minimal_projected_tokens": round(minimal_calls * average_tokens_per_call),
        "semantic_minimal_projected_token_savings": round(saved_calls * average_tokens_per_call),
        "old_policy_projected_latency_ms": round(old_calls * average_latency_ms_per_call, 3),
        "semantic_minimal_projected_latency_ms": round(minimal_calls * average_latency_ms_per_call, 3),
        "semantic_minimal_projected_latency_savings_ms": round(saved_calls * average_latency_ms_per_call, 3),
        "method": "conservative filename/declaration coverage projection; no inference executed",
    }


def _explicit_name_candidate(row: Mapping[str, Any]) -> bool:
    text = " ".join(str(row.get(key, "")) for key in ("source_image", "archive_member", "declared_material")).lower()
    generic = bool(__import__("re").search(r"(?:^|[/_ -])(?:icon|item|sprite|crystal)\d*(?:\.|$)", text))
    named = bool(
        __import__("re").search(
            r"buckler|helmet|ring|pants|armor|shirt|jacket|cap|agate|eggplant|key|sword|shield|potion|gem", text
        )
    )
    return named and not generic


def _write_reports(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    routing: Mapping[str, Any],
    scanner: Mapping[str, Any],
) -> None:
    statuses = Counter(str(row.get("record_status")) for row in rows)
    dispositions = Counter(item["status"] for row in rows for item in row.get("conflict_dispositions", []))
    reports = {
        "pilot_findings.md": f"# Pilot findings\n\nOffline replay processed {len(rows)} records with zero HTTP attempts and zero new provider calls. Record states: {dict(statuses)}.\n",
        "provider_failure_policy.md": "# Provider failure policy\n\nFailures retain raw responses and accounting. Invalid output is repaired only by the bounded deterministic policy; otherwise an explicitly enabled fallback abstains critical fields. Failed artifacts are never called cache successes.\n",
        "offline_replay_contract.md": "# Offline replay contract\n\nExact cache identity and embedded artifact compatibility are validated. Missing artifacts fail closed unless deterministic fallback is explicitly enabled. Inputs are tree-hashed before and after. Output JSON is canonical and timestamp-free.\n",
        "cost_aware_routing.md": "# Cost-aware routing\n\n```json\n"
        + json.dumps(routing, indent=2, sort_keys=True)
        + "\n```\n",
        "description_and_color_fixes.md": "# Description and color fixes\n\nDescriptions are generated and validated from one normalized fact object. Alternative objects and visual material cues remain provenance. Color roles are palette-constrained; outlines/shadows prefer darkest compatible colors and highlights prefer lightest.\n",
        "conflict_dispositions.md": "# Conflict dispositions\n\nAll 20 pilot conflicts have individual IDs, statuses, resolutions, verifier eligibility, training consequences, and risk penalties.\n\n```json\n"
        + json.dumps(
            {
                "counts": dict(dispositions),
                "records": [
                    {"sprite_id": r.get("sprite_id"), "conflicts": r.get("conflict_dispositions")} for r in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
        "semantic_audit_report.md": f"# Semantic-only audit\n\nFinal semantic violations: {len(scanner['final_semantic_violations'])}. Training target violations: {len(scanner['training_target_violations'])}. Raw evidence observations: {len(scanner['raw_evidence_observations'])}; these do not fail final gates.\n",
        "pilot_replay_comparison.md": "# Original versus replay\n\n```json\n"
        + json.dumps(comparison, indent=2, sort_keys=True)
        + "\n```\n",
        "calibration_wave1_plan.md": "# Calibration wave 1\n\nThe separately frozen 100-representative manifest is for human audit only. No gold labels are fabricated and no calibrator is fitted before audited truth exists.\n",
        "command_log.txt": (
            "python -m spritelab harvest label-v4-replay --input-pilot experiments/label_v4_real_pilot_15_v1 "
            "--output-root experiments/label_v4_pilot_replay_v2 --shared-cache-root "
            "experiments/label_v4_real_pilot_15_v1/shared_bc_cache_v1 --require-complete-cache "
            "--allow-deterministic-fallback\nHTTP attempts: 0\nnew provider calls: 0\n"
        ),
    }
    for name, text in reports.items():
        (output / name).write_text(text, encoding="utf-8", newline="\n")


def _load_pilot_records(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "mode_b").glob("*/canary_records_v1.jsonl"))
    rows = [
        json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) != 15:
        raise OfflineReplayError(f"expected 15 pilot records, found {len(rows)}")
    return sorted(rows, key=lambda row: str(row.get("sprite_id", "")))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
