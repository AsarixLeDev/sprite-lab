"""Fixed 25-record no-cost A/B/C regression experiment for Labeling v4."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spritelab.harvest.label_v4.description import validate_description
from spritelab.harvest.label_v4.family_audit import audit_families
from spritelab.harvest.label_v4.pipeline import LabelV4PipelineConfig, label_record_v4
from spritelab.harvest.label_v4.pixel_evidence import analyze_pixels
from spritelab.harvest.label_v4.providers import MockJSONProvider
from spritelab.harvest.label_v4.risk import SEMANTIC_FIELDS
from spritelab.harvest.label_v4.semantic_axes import CATEGORY_VALUES, DOMAIN_VALUES, ROLE_VALUES

COHORT_EXPERIMENT_VERSION = "label_v4_same_cohort_v1.0"
MOCK_FIXTURE_VERSION = "cohort_blind_mock_v1.0"
DEFAULT_COHORT = Path("out/r2_annotation_batch_0001_semantic_accept_only_25.jsonl")
DEFAULT_RESOLVED = Path("out/r2_assisted_v3_batch_0001/scheduler_resolved_candidates.jsonl")
DEFAULT_OUTPUT = Path("experiments/label_v4_same_cohort_comparison")
TARGETED_SMOKE_IDS: tuple[str, ...] = (
    "acq_idylwild_armory_iron_buckler",
    "acq_idylwild_armory_platemail_helmet",
    "acq_idylwild_armory_iron_ring",
    "acq_idylwild_armory_copper_ring",
    "acq_idylwild_armory_cloth_pants",
    "acq_idylwild_armory_quilted_armor",
    "acq_idylwild_armory_tattered_shirt",
    "acq_idylwild_armory_leather_cap",
    "acq_idylwild_armory_chainmail_jacket",
    "acq_gem_ettingrinder_small_purple",
    "oga_496_rpg_icons_32fix_i_crystal01",
    "oga_cc0_gem_7soul1_agate",
    "shade_16x16_weapons_bronze-weapons_r000_c019",
    "oga_cc0_food_ocal_eggplant",
    "oga_cc0_key_rcorre_key_01",
)


def run_same_cohort_comparison(
    *,
    cohort_path: str | Path = DEFAULT_COHORT,
    resolved_path: str | Path = DEFAULT_RESOLVED,
    output_dir: str | Path = DEFAULT_OUTPUT,
    max_records: int | None = None,
    sprite_ids: Sequence[str] | None = None,
    record_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic A and deterministic-mocked B/C without paid inference."""

    cohort_path = Path(cohort_path)
    resolved_path = Path(resolved_path)
    output_dir = Path(output_dir)
    cohort_rows = select_cohort_rows(
        _read_jsonl(cohort_path),
        sprite_ids=sprite_ids,
        record_manifest=record_manifest,
        max_records=max_records,
    )
    ordered_ids = [str(row.get("sprite_id", "")) for row in cohort_rows]
    resolved_index = {str(row.get("sprite_id", "")): row for row in _read_jsonl(resolved_path)}
    missing = [sprite_id for sprite_id in ordered_ids if sprite_id not in resolved_index]
    if missing:
        raise ValueError(f"cohort records missing from resolved inputs: {missing}")
    records = [resolved_index[sprite_id] for sprite_id in ordered_ids]

    fixture_rows: list[dict[str, Any]] = []
    responses: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        image_path = Path(str(record.get("final_png_path", "")))
        pixels = analyze_pixels(image_path)
        proposal = cohort_mock_proposal(str(record.get("sprite_id", "")), pixels)
        image_hash = str(pixels["image_hash"])
        responses[("B_blind_vlm_proposal", image_hash)] = proposal
        fixture_rows.append(
            {
                "fixture_version": MOCK_FIXTURE_VERSION,
                "sprite_id": str(record.get("sprite_id", "")),
                "image_hash": image_hash,
                "proposal": proposal,
                "blind_runtime_key": ["B_blind_vlm_proposal", image_hash],
                "scheduler_or_filename_passed_at_runtime": False,
                "fixture_is_human_reviewed_truth": False,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_context = tempfile.TemporaryDirectory(prefix=".cohort_cache_", dir=output_dir)
    cache_dir = Path(cache_context.name)
    vlm_provider = MockJSONProvider(
        responses,
        model_identity="mock-cohort-blind-vlm-v1",
        namespace="blind_vlm_proposal_v4",
    )
    outputs: dict[str, list[dict[str, Any]]] = {}
    try:
        for variant in ("A", "B", "C"):
            config = LabelV4PipelineConfig(
                mode=variant,
                cache_dir=cache_dir,
                use_cache=True,
                shared_cache=True,
                force_vlm_for_comparison=variant in {"B", "C"},
            )
            rows: list[dict[str, Any]] = []
            for record in records:
                rows.append(
                    label_record_v4(
                        record,
                        config=config,
                        vlm_provider=vlm_provider if variant in {"B", "C"} else None,
                    )
                )
            outputs[variant] = rows
    finally:
        cache_context.cleanup()

    metrics = {variant: compare_metrics(rows) for variant, rows in outputs.items()}
    comparison = {
        "schema_version": COHORT_EXPERIMENT_VERSION,
        "cohort_path": str(cohort_path).replace("\\", "/"),
        "cohort_sha256": _file_hash(cohort_path),
        "resolved_path": str(resolved_path).replace("\\", "/"),
        "record_count": len(records),
        "ordered_sprite_ids": ordered_ids,
        "selection": {
            "max_records": int(max_records) if max_records is not None else None,
            "sprite_ids": [str(value) for value in (sprite_ids or ())],
            "record_manifest": str(record_manifest).replace("\\", "/") if record_manifest is not None else None,
            "record_manifest_sha256": _file_hash(Path(record_manifest)) if record_manifest is not None else None,
        },
        "providers": {
            "A": "deterministic_only",
            "B": "deterministic_mock_blind_vlm+mock_text_reconciliation",
            "C": "B+adaptive_mock_independent_verifier",
        },
        "paid_provider_calls": 0,
        "legacy_cache_reuse": False,
        "metrics": metrics,
        "deltas_vs_A": {variant: _metric_deltas(metrics["A"], metrics[variant]) for variant in ("B", "C")},
        "acceptance_checks": same_cohort_acceptance_checks(outputs),
        "caveat": (
            "B/C are deterministic mocked pipeline regressions, not measurements of real-provider accuracy and not "
            "verified truth. Uncalibrated fields retain conservative uncertainty."
        ),
    }

    _atomic_write_json(output_dir / "mock_provider_fixtures_v1.json", {"fixtures": fixture_rows})
    for variant, rows in outputs.items():
        _atomic_write_jsonl(output_dir / f"cohort_{variant}_v1.jsonl", rows)
        _atomic_write_json(output_dir / f"family_audit_{variant}_v1.json", audit_families(rows))
    _atomic_write_json(output_dir / "comparison_metrics_v1.json", comparison)
    targeted_results = _targeted_mock_results(outputs, comparison)
    _atomic_write_json(output_dir / "targeted_mock_results.json", targeted_results)
    _atomic_write_bytes(
        output_dir / "targeted_mock_results.md",
        _targeted_mock_markdown(targeted_results).encode("utf-8"),
    )
    manifest = _artifact_manifest(output_dir)
    _atomic_write_json(output_dir / "artifact_manifest_v1.json", manifest)
    return comparison


def _targeted_mock_results(
    outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    variants: dict[str, list[dict[str, Any]]] = {}
    for variant, rows in outputs.items():
        variants[variant] = [
            {
                "sprite_id": str(row.get("sprite_id", "")),
                "domain": row.get("domain"),
                "category": row.get("category"),
                "canonical_object": row.get("canonical_object"),
                "canonical_object_alternatives": _nested(
                    row, "reconciliation", "field_proposals", "canonical_object", "alternatives"
                )
                or [],
                "visual_form": _nested(row, "semantics", "visual_form") or [],
                "surface_alias": row.get("surface_alias"),
                "role": row.get("role"),
                "explicit_material": row.get("explicit_material"),
                "size_hint": _nested(row, "semantics", "size_hint"),
                "filename_color_hints": _nested(row, "semantics", "colors", "filename_color_hints") or [],
                "description": row.get("description"),
                "uncertainty_1_20": _nested(row, "record_risk", "record_uncertainty_1_20"),
                "legacy_evidence_used": bool(row.get("legacy_evidence_used")),
                "new_provider_calls": int(row.get("new_provider_calls", row.get("provider_call_count", 0)) or 0),
                "actual_http_attempts": int(row.get("actual_http_attempts", 0) or 0),
            }
            for row in rows
        ]
    return {
        "schema_version": "label_v4_targeted_mock_results_v1.0",
        "record_count": int(comparison.get("record_count", 0) or 0),
        "ordered_sprite_ids": list(comparison.get("ordered_sprite_ids") or ()),
        "paid_provider_calls": 0,
        "actual_http_attempts": 0,
        "acceptance_checks": dict(comparison.get("acceptance_checks") or {}),
        "variants": variants,
    }


def _targeted_mock_markdown(results: Mapping[str, Any]) -> str:
    lines = [
        "# Labeling v4 targeted mock results",
        "",
        f"Records: {int(results.get('record_count', 0) or 0)}. Paid provider calls: 0. Actual HTTP attempts: 0.",
        "",
        "| Mode | Sprite | Canonical object | Category | Material | Role | Alias | Uncertainty |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for variant, rows in dict(results.get("variants") or {}).items():
        for row in rows:
            alias = str(row.get("surface_alias") or "—").replace("|", "\\|")
            canonical = str(row.get("canonical_object") or "abstained").replace("|", "\\|")
            lines.append(
                "| {variant} | `{sprite}` | {canonical} | {category} | {material} | {role} | {alias} | {risk} |".format(
                    variant=variant,
                    sprite=str(row.get("sprite_id", "")),
                    canonical=canonical,
                    category=row.get("category") or "unknown",
                    material=row.get("explicit_material") or "—",
                    role=row.get("role") or "unknown",
                    alias=alias,
                    risk=row.get("uncertainty_1_20") if row.get("uncertainty_1_20") is not None else "—",
                )
            )
    return "\n".join(lines) + "\n"


def select_cohort_rows(
    cohort_rows: Sequence[Mapping[str, Any]],
    *,
    sprite_ids: Sequence[str] | None = None,
    record_manifest: str | Path | None = None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic ordered cohort without treating absent rows as evaluated.

    Repeated ``sprite_ids`` retain command-line order.  A JSONL record manifest
    contributes its rows in file order after the explicit IDs.  Each manifest
    row must be an object with a non-empty ``sprite_id``.  Duplicate or unknown
    targets are rejected rather than silently changing the executed cohort.
    """

    rows = [dict(row) for row in cohort_rows]
    index: dict[str, dict[str, Any]] = {}
    duplicate_cohort_ids: list[str] = []
    for row in rows:
        sprite_id = str(row.get("sprite_id", "")).strip()
        if not sprite_id:
            raise ValueError("cohort row has no sprite_id")
        if sprite_id in index:
            duplicate_cohort_ids.append(sprite_id)
        else:
            index[sprite_id] = row
    if duplicate_cohort_ids:
        raise ValueError(f"duplicate sprite ids in cohort: {_dedupe_ordered(duplicate_cohort_ids)}")

    requested_ids = [_validated_sprite_id(value, source="--sprite-id") for value in (sprite_ids or ())]
    if record_manifest is not None:
        requested_ids.extend(_read_record_manifest_ids(Path(record_manifest)))
    duplicate_targets = _duplicates(requested_ids)
    if duplicate_targets:
        raise ValueError(f"duplicate targeted sprite ids: {duplicate_targets}")

    if requested_ids:
        missing = [sprite_id for sprite_id in requested_ids if sprite_id not in index]
        if missing:
            raise ValueError(f"targeted records missing from cohort: {missing}")
        selected = [index[sprite_id] for sprite_id in requested_ids]
    else:
        selected = rows
    if max_records is not None:
        selected = selected[: max(0, int(max_records))]
    return selected


def cohort_mock_proposal(sprite_id: str, pixels: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canned *proposal*, explicitly not a reviewed ground-truth label."""

    spec = _fixture_spec(sprite_id)
    colors = list(pixels.get("palette_colors") or ())
    primary = list(spec.get("primary") or colors[:1])
    secondary = list(spec.get("secondary") or colors[1:2])
    alternatives = list(spec.get("alternatives") or ())
    uncertainties = list(spec.get("uncertainties") or ())
    if alternatives and not uncertainties:
        uncertainties = ["identity has a plausible alternative"]
    return {
        "schema_version": "vlm_proposal_v4.1",
        "canonical_object_candidates": [
            {"value": spec["object"], "visual_support": list(spec.get("visual_support") or ())},
            *[
                {"value": value, "visual_support": ["plausible alternative silhouette interpretation"]}
                for value in alternatives
            ],
        ],
        "category_candidates": [spec["category"]],
        "surface_alias_candidates": [spec["alias"]],
        "role_candidates": [spec["role"]],
        "shape": dict(spec["shape"]),
        "color_roles": {
            "primary": primary,
            "secondary": secondary,
            "outline": ["black"] if "black" in colors else [],
            "shadow": [value for value in colors if value.startswith("dark_")][:1],
            "highlight": [value for value in colors if value.startswith("light_")][:1],
        },
        "material_visual_cues": list(spec.get("material_cues") or ()),
        "description_candidates": [spec["description"]],
        "uncertainties": uncertainties,
        "alternative_interpretations": alternatives,
        "unsupported_fields": ["explicit_material"],
    }


def compare_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverage_fields = SEMANTIC_FIELDS
    total_slots = len(rows) * len(coverage_fields)
    covered = sum(_present(_semantic_leaf(row, field)) for row in rows for field in coverage_fields)
    contradictions = sum(bool(row.get("unresolved_conflicts")) for row in rows)
    taxonomy_valid = sum(
        row.get("domain") in DOMAIN_VALUES and row.get("category") in CATEGORY_VALUES and row.get("role") in ROLE_VALUES
        for row in rows
    )
    unsupported_material = 0
    valid_descriptions = 0
    open_set_expected = 0
    open_set_preserved = 0
    uncertainty = Counter()
    edit_count = 0
    logical_calls = 0
    actual_calls = 0
    cache_hits = 0
    latency = 0.0
    for row in rows:
        filename = _nested(row, "deterministic_evidence", "filename") or {}
        explicit = str(row.get("explicit_material") or "")
        explicit_candidates = set(filename.get("explicit_material_candidates") or ())
        if explicit and explicit not in explicit_candidates:
            unsupported_material += 1
        semantics = row.get("semantics") if isinstance(row.get("semantics"), Mapping) else {}
        valid, _unsupported = validate_description(str(row.get("description") or ""), semantics)
        valid_descriptions += int(valid)
        expected_terms = set(filename.get("open_set_tokens") or ())
        open_set_expected += len(expected_terms)
        open_set_preserved += len(expected_terms & set(row.get("open_set_terms") or ()))
        score = _nested(row, "record_risk", "record_uncertainty_1_20")
        if score is not None:
            uncertainty[int(score)] += 1
        # Estimate edits from missing semantic leaves and surfaced disputes.
        # Conservative uncalibrated scores alone do not imply a human must edit
        # every field; they control training eligibility until audit support.
        edit_count += sum(not _present(_semantic_leaf(row, field)) for field in coverage_fields)
        edit_count += len(row.get("unresolved_conflicts") or ())
        edit_count += int(not valid)
        stage_rows = [value for value in row.get("stage_ledger") or () if isinstance(value, Mapping)]
        logical_calls += sum(value.get("stage") != "A_deterministic" for value in stage_rows)
        actual_calls += sum(bool(value.get("provider_call")) for value in stage_rows)
        cache_hits += sum(bool(value.get("cache_hit")) for value in stage_rows)
        latency += sum(float(value.get("latency_ms") or 0.0) for value in stage_rows)
    count = len(rows)
    return {
        "records": count,
        "field_coverage": covered / total_slots if total_slots else 0.0,
        "covered_field_slots": covered,
        "total_field_slots": total_slots,
        "contradiction_rate": contradictions / count if count else 0.0,
        "taxonomy_validity": taxonomy_valid / count if count else 0.0,
        "unsupported_material_rate": unsupported_material / count if count else 0.0,
        "description_validity": valid_descriptions / count if count else 0.0,
        "open_set_preservation": open_set_preserved / open_set_expected if open_set_expected else 1.0,
        "uncertainty_histogram_1_20": {str(score): uncertainty.get(score, 0) for score in range(1, 21)},
        "logical_model_stage_count": logical_calls,
        "actual_mock_provider_call_count": actual_calls,
        "paid_provider_call_count": 0,
        "latency_ms": latency,
        "cache_hits": cache_hits,
        "estimated_human_field_edits": edit_count,
    }


def same_cohort_acceptance_checks(outputs: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for variant, rows in outputs.items():
        indexed = {str(row.get("sprite_id", "")): row for row in rows}
        expected = {
            "acq_idylwild_armory_iron_buckler": ("buckler", "armor", "iron", "defensive_equipment"),
            "acq_idylwild_armory_platemail_helmet": ("helmet", "armor", "plate_metal", "wearable_equipment"),
            "acq_idylwild_armory_iron_ring": ("ring", "jewelry", "iron", "wearable_equipment"),
            "acq_idylwild_armory_copper_ring": ("ring", "jewelry", "copper", "wearable_equipment"),
            "acq_idylwild_armory_cloth_pants": ("pants", "clothing", "cloth", "wearable_equipment"),
            "acq_idylwild_armory_quilted_armor": ("armor", "armor", None, "wearable_equipment"),
            "acq_idylwild_armory_tattered_shirt": ("shirt", "clothing", None, "wearable_equipment"),
            "acq_idylwild_armory_leather_cap": ("cap", "clothing", "leather", "wearable_equipment"),
            "acq_idylwild_armory_chainmail_jacket": ("jacket", "armor", "chainmail", "wearable_equipment"),
        }
        named_checks = {}
        for sprite_id, wanted in expected.items():
            row = indexed.get(sprite_id)
            if row is None:
                named_checks[sprite_id] = {
                    "status": "not_evaluated",
                    "pass": None,
                    "observed": None,
                    "expected": wanted,
                }
                continue
            observed = (
                row.get("canonical_object"),
                row.get("category"),
                row.get("explicit_material"),
                row.get("role"),
            )
            passed = observed == wanted
            named_checks[sprite_id] = {
                "status": "passed" if passed else "failed",
                "pass": passed,
                "observed": observed,
                "expected": wanted,
            }
        small = indexed.get("acq_gem_ettingrinder_small_purple")
        agate = indexed.get("oga_cc0_gem_7soul1_agate")
        crystal = indexed.get("oga_496_rpg_icons_32fix_i_crystal01")
        shade = indexed.get("shade_16x16_weapons_bronze-weapons_r000_c019")
        small_colors = _nested(small or {}, "semantics", "colors", "filename_color_hints") or []
        small_size = _nested(small or {}, "semantics", "size_hint")
        small_passed = bool(small and small_size == "small" and "purple" in small_colors)
        crystal_object = crystal.get("canonical_object") if crystal else None
        crystal_passed = bool(crystal and crystal_object and "cluster" in str(crystal_object))
        shade_description = str(shade.get("description") or "") if shade else ""
        shade_alias = str(shade.get("surface_alias") or "") if shade else ""
        shade_core = bool(
            shade
            and shade.get("category") == "weapon"
            and shade.get("role") == "weapon"
            and shade.get("explicit_material") == "bronze"
            and shade.get("canonical_object") is None
            and not any(
                re.search(rf"\b{re.escape(term)}\b", shade_description.lower())
                for term in ("pencil", "pen", "matchstick")
            )
            and "16x16 weapons" not in shade_alias.lower()
        )
        agate_passed = bool(
            agate
            and agate.get("legacy_evidence_used") is False
            and agate.get("explicit_material") != "glass"
            and agate.get("domain") in DOMAIN_VALUES
        )
        checks[variant] = {
            "named_deterministic_recovery": named_checks,
            "small_purple": {
                "status": _acceptance_status(small is not None, small_passed),
                "canonical_object": small.get("canonical_object") if small else None,
                "size_hint": small_size,
                "filename_color_hints": small_colors,
                "conflicts": list(small.get("unresolved_conflicts") or ()) if small else [],
            },
            "crystal_cluster": {
                "status": _acceptance_status(crystal is not None, crystal_passed),
                "canonical_object": crystal_object,
            },
            "shade_weapon": {
                "status": _acceptance_status(shade is not None, shade_core),
                "canonical_object": shade.get("canonical_object") if shade else None,
                "visual_form": _nested(shade or {}, "semantics", "visual_form") or [],
                "surface_alias": shade.get("surface_alias") if shade else None,
                "description": shade_description,
                "alternatives": _nested(
                    shade or {}, "reconciliation", "field_proposals", "canonical_object", "alternatives"
                )
                or [],
            },
            "agate_legacy_isolation": {
                "status": _acceptance_status(agate is not None, agate_passed),
                "legacy_evidence_used": agate.get("legacy_evidence_used") if agate else None,
                "explicit_material": agate.get("explicit_material") if agate else None,
                "domain": agate.get("domain") if agate else None,
            },
        }
    return checks


def _acceptance_status(evaluated: bool, passed: bool) -> str:
    if not evaluated:
        return "not_evaluated"
    return "passed" if passed else "failed"


def _fixture_spec(sprite_id: str) -> dict[str, Any]:
    common_shapes = {
        "ring": {
            "silhouette": ["round"],
            "aspect": ["compact"],
            "orientation": ["front_facing"],
            "structure": ["ring_shaped"],
            "edge_profile": ["rounded"],
            "parts": ["band", "setting"],
        },
        "gem": {
            "silhouette": ["oval"],
            "aspect": ["compact"],
            "orientation": ["front_facing"],
            "structure": ["solid"],
            "edge_profile": ["beveled"],
            "parts": ["facets"],
        },
        "clothing": {
            "silhouette": ["irregular"],
            "aspect": ["wide"],
            "orientation": ["front_facing"],
            "structure": ["layered"],
            "edge_profile": ["rounded"],
            "parts": ["fabric_panels"],
        },
    }
    exact: dict[str, dict[str, Any]] = {
        "shade_16x16_weapons_bronze-weapons_r000_c019": {
            "object": "cylinder",
            "alternatives": ["rod", "stick", "pencil", "pen", "matchstick"],
            "category": "tool",
            "alias": "bar",
            "role": "tool",
            "visual_support": ["elongated rounded form", "consistent width", "dark outline"],
            "shape": {
                "silhouette": ["elongated"],
                "aspect": ["elongated"],
                "orientation": ["diagonal"],
                "structure": ["solid"],
                "edge_profile": ["rounded"],
                "parts": [],
            },
            "material_cues": ["smooth_surface"],
            "uncertainties": ["functional identity is unresolved from isolated pixels"],
            "description": "A simple elongated object resembling a stick or pencil.",
        },
        "oga_cc0_food_ocal_eggplant": {
            "object": "eggplant",
            "category": "food",
            "alias": "purple eggplant",
            "role": "consumable",
            "visual_support": ["purple oval body", "green stem"],
            "shape": {
                "silhouette": ["oval"],
                "aspect": ["tall"],
                "orientation": ["vertical"],
                "structure": ["solid"],
                "edge_profile": ["rounded"],
                "parts": ["body", "stem"],
            },
            "material_cues": ["organic"],
            "description": "A purple eggplant with a rounded body and visible stem.",
        },
        "oga_cc0_key_rcorre_key_01": {
            "object": "key",
            "category": "key",
            "alias": "ornate key",
            "role": "quest_item",
            "visual_support": ["toothed shaft", "looped bow"],
            "shape": {
                "silhouette": ["elongated"],
                "aspect": ["wide"],
                "orientation": ["horizontal"],
                "structure": ["ring_shaped"],
                "edge_profile": ["jagged"],
                "parts": ["bow", "shaft", "teeth"],
            },
            "material_cues": ["metallic"],
            "description": "An ornate key with a looped bow, narrow shaft, and visible teeth.",
        },
        "acq_idylwild_armory_iron_buckler": {
            "object": "buckler",
            "category": "armor",
            "alias": "round shield",
            "role": "defensive_equipment",
            "visual_support": ["round shield", "central boss", "dark rim"],
            "shape": {
                "silhouette": ["round"],
                "aspect": ["compact"],
                "orientation": ["front_facing"],
                "structure": ["rimmed", "bossed"],
                "edge_profile": ["rounded"],
                "parts": ["rim", "central_boss"],
            },
            "material_cues": ["metallic"],
            "description": "A round buckler with a dark rim and raised central boss.",
        },
        "acq_idylwild_armory_platemail_helmet": {
            "object": "helmet",
            "category": "armor",
            "alias": "plate helmet",
            "role": "wearable_equipment",
            "visual_support": ["head-shaped shell", "face opening", "metal highlights"],
            "shape": {
                "silhouette": ["round"],
                "aspect": ["compact"],
                "orientation": ["front_facing"],
                "structure": ["layered"],
                "edge_profile": ["rounded"],
                "parts": ["crown", "face_opening"],
            },
            "material_cues": ["metallic"],
            "description": "A compact helmet with a rounded crown and visible face opening.",
        },
        "oga_cc0_gem_7soul1_agate": {
            "object": "agate",
            "alternatives": ["oval_gem"],
            "category": "gem",
            "alias": "pink oval agate",
            "role": "resource",
            "visual_support": ["polished oval form", "pink highlight bands"],
            "shape": common_shapes["gem"],
            "material_cues": ["polished"],
            "description": "A polished-looking pink oval agate with a bright curved highlight.",
        },
        "acq_gem_thekingphoenix_diamond": {
            "object": "diamond",
            "category": "gem",
            "alias": "faceted diamond",
            "role": "resource",
            "visual_support": ["symmetrical faceted gem", "pointed lower half"],
            "shape": {
                **common_shapes["gem"],
                "silhouette": ["diamond"],
                "edge_profile": ["pointed", "beveled"],
            },
            "material_cues": ["crystalline"],
            "description": "A faceted diamond-shaped gem with a pointed base and bright highlights.",
        },
    }
    if sprite_id in exact:
        return exact[sprite_id]
    if "crystal0" in sprite_id:
        return {
            "object": "crystal_cluster",
            "alternatives": ["crystal_shards"],
            "category": "gem",
            "alias": "jagged crystal cluster",
            "role": "resource",
            "visual_support": ["multiple pointed crystals", "shared clustered base"],
            "shape": {
                "silhouette": ["multipart"],
                "aspect": ["wide"],
                "orientation": ["front_facing"],
                "structure": ["clustered", "multipart"],
                "edge_profile": ["jagged", "pointed"],
                "parts": ["crystal_points", "cluster_base"],
            },
            "material_cues": ["crystalline"],
            "description": "A clustered group of jagged crystal points rising from a shared base.",
        }
    if "ring" in sprite_id or "jewelry_buch" in sprite_id:
        return {
            "object": "ring",
            "alternatives": ["ornamental_band"],
            "category": "jewelry",
            "alias": "ornamental gemstone ring",
            "role": "wearable_equipment",
            "visual_support": ["circular band", "raised colored setting"],
            "shape": common_shapes["ring"],
            "material_cues": ["metallic"],
            "description": "An ornamental ring with a circular band and raised colored setting.",
        }
    if any(token in sprite_id for token in ("jewel_", "gem_orange", "small_purple", "small_white")):
        primary = ["blue"] if "small_white" in sprite_id else []
        return {
            "object": "faceted_gem",
            "alternatives": ["gem"],
            "category": "gem",
            "alias": "small faceted gemstone",
            "role": "resource",
            "visual_support": ["compact gem silhouette", "faceted highlights"],
            "shape": common_shapes["gem"],
            "material_cues": ["crystalline"],
            "description": "A small faceted gemstone with a dark outline and bright highlights.",
            "primary": primary,
        }
    clothing: dict[str, tuple[str, str, str, list[str]]] = {
        "cloth_pants": ("pants", "clothing", "cloth pants", ["legs", "waistband"]),
        "quilted_armor": ("armor", "armor", "quilted armor", ["quilted_panels", "collar"]),
        "tattered_shirt": ("shirt", "clothing", "tattered shirt", ["torso", "sleeves", "torn_edges"]),
        "leather_cap": ("cap", "clothing", "leather cap", ["crown", "brim"]),
        "chainmail_jacket": ("jacket", "armor", "chainmail jacket", ["torso", "sleeves", "mail_links"]),
    }
    for token, (object_name, category, alias, parts) in clothing.items():
        if token in sprite_id:
            return {
                "object": object_name,
                "category": category,
                "alias": alias,
                "role": "wearable_equipment",
                "visual_support": ["wearable garment silhouette", *parts],
                "shape": {**common_shapes["clothing"], "parts": parts},
                "material_cues": ["textile"],
                "description": f"A front-facing {alias} with visible {_join_parts(parts)}.",
            }
    raise KeyError(f"no fixed mock proposal for cohort sprite {sprite_id}")


def _metric_deltas(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    names = (
        "field_coverage",
        "contradiction_rate",
        "taxonomy_validity",
        "unsupported_material_rate",
        "description_validity",
        "open_set_preservation",
        "estimated_human_field_edits",
    )
    return {name: float(b.get(name, 0.0)) - float(a.get(name, 0.0)) for name in names}


def _join_parts(parts: Sequence[str]) -> str:
    values = [value.replace("_", " ") for value in parts]
    return ", ".join(values[:-1]) + (" and " if len(values) > 1 else "") + values[-1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_record_manifest_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"record manifest not found: {path}")
    result: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in record manifest {path}:{line_number}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"record manifest row must be an object: {path}:{line_number}")
        result.append(_validated_sprite_id(row.get("sprite_id"), source=f"{path}:{line_number}"))
    if not result:
        raise ValueError(f"record manifest contains no sprite ids: {path}")
    return result


def _validated_sprite_id(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sprite_id must be a non-empty string ({source})")
    return value.strip()


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _dedupe_ordered(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, default=str, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest_v1.json":
            files[path.name] = {"sha256": _file_hash(path), "bytes": path.stat().st_size}
    return {
        "schema_version": COHORT_EXPERIMENT_VERSION,
        "files": files,
        "historical_artifacts_modified": False,
        "paid_provider_calls": 0,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _semantic_leaf(row: Mapping[str, Any], field_name: str) -> Any:
    semantics = row.get("semantics") if isinstance(row.get("semantics"), Mapping) else {}
    shape = semantics.get("shape") if isinstance(semantics.get("shape"), Mapping) else {}
    colors = semantics.get("colors") if isinstance(semantics.get("colors"), Mapping) else {}
    if field_name in {"silhouette", "aspect", "orientation", "structure", "edge_profile", "parts"}:
        return shape.get(field_name)
    if field_name in {
        "palette_colors",
        "primary_colors",
        "secondary_colors",
        "outline_colors",
        "shadow_colors",
        "highlight_colors",
        "filename_color_hints",
    }:
        return colors.get(field_name)
    return semantics.get(field_name, row.get(field_name))


def _present(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return int(bool(value.strip()) and value.strip() != "unknown")
    return int(bool(value))
