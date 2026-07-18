"""Conservative, hierarchical semantic labeling contracts for Dataset-v5.

This module is deliberately provider neutral.  It validates and reconciles
already-produced proposals, builds field-specific health reports, and prepares
review/calibration records.  It never performs network I/O or promotes model
agreement to human truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

CONTRACT_VERSION = "sprite_lab_hierarchical_label_contract_v3"
TAXONOMY_VERSION = "sprite_lab_visual_taxonomy_hierarchy_v2"
FIELD_STATE_VERSION = "sprite_lab_semantic_field_state_v3"
COMPARISON_VERSION = "sprite_lab_field_comparison_v3"
HEALTH_GATE_VERSION = "sprite_lab_field_health_gate_v3"
RECONCILIATION_VERSION = "sprite_lab_conservative_reconciliation_v3"
REVIEW_QUEUE_VERSION = "sprite_lab_review_by_exception_v1"
REVIEW_EVENT_VERSION = "sprite_lab_review_event_v1"
CALIBRATION_INPUT_VERSION = "sprite_lab_calibration_input_v2"
PROMPT_VERSION = "sprite_lab_conservative_prompt_v3"
OUTPUT_SCHEMA_VERSION = "sprite_lab_conservative_label_output_schema_v3"
UNKNOWN_SENTINEL_POLICY_VERSION = "sprite_lab_conditional_unknown_sentinel_policy_v1"
PROVIDER_NORMALIZATION_VERSION = "sprite_lab_provider_conditional_normalization_v1"
CONDITIONAL_INVARIANT_VERSION = "sprite_lab_conditional_identity_invariant_v1"
HISTORICAL_READ_ADAPTER_VERSION = "sprite_lab_historical_conditional_read_adapter_v1"
SALVAGE_SUMMARY_VERSION = "sprite_lab_conditional_salvage_summary_v1"

CORE_FIELDS = ("domain", "broad_category")
CONDITIONAL_FIELDS = ("canonical_object", "role")
AUXILIARY_FIELDS = ("visual_form", "visual_material_cue", "colors", "description")
SEMANTIC_FIELDS = (*CORE_FIELDS, *CONDITIONAL_FIELDS, *AUXILIARY_FIELDS)
CALIBRATION_FIELDS = (*CORE_FIELDS, *CONDITIONAL_FIELDS)

FIELD_STATES = (
    "labeled",
    "model_abstained",
    "not_applicable",
    "invalid_output",
    "conflict",
    "pending_review",
)
ABSTENTION_REASONS = (
    "visually_ambiguous",
    "insufficient_resolution",
    "multiple_plausible_objects",
    "role_not_visually_demonstrated",
    "taxonomy_has_no_safe_match",
    "contact_sheet_insufficient",
    "image_unidentifiable",
    "provider_returned_unknown",
    "legacy_unknown_normalized",
    "reconciliation_has_no_defensible_exact_identity",
)
COMPARISON_CLASSIFICATIONS = (
    "exact_agreement",
    "compatible_hierarchy",
    "one_side_abstention",
    "both_abstained",
    "taxonomy_granularity_difference",
    "true_contradiction",
    "invalid_comparison",
    "not_applicable",
)
CALIBRATION_STATES = (
    "human_verified",
    "human_abstained",
    "model_agreement_candidate",
    "model_conflict",
    "unreviewed",
)
REVIEW_ACTIONS = (
    "accept_broad_label",
    "choose_safer_parent",
    "abstain",
    "exclude_semantic_supervision",
)

_FIELD_ALIASES = {"category": "broad_category"}
_LEGACY_STATE_MAP = {
    "known": "labeled",
    "model_abstained": "model_abstained",
    "not_applicable": "not_applicable",
    "unsupported": "invalid_output",
}
_CONDITIONAL_UNKNOWN_SENTINEL = "unknown"
# These are the only historical schemas in repository evidence that use the
# controlled ``unknown`` enum.  The value match remains exact; no arbitrary
# placeholder text is converted to abstention.
_DOCUMENTED_LEGACY_UNKNOWN_SCHEMAS = frozenset(
    {
        "sprite_lab_codex_blind_label_v1",
        "sprite_lab_labeling_health_blind_pass_v2",
    }
)
_MATERIAL_CUE_ALIASES = {"metallic": "metal-like", "wooden": "wood-like", "organic": "organic-like"}
_MATERIAL_CUES = frozenset(
    {"metal-like", "wood-like", "stone-like", "crystalline", "fabric-like", "organic-like", "liquid-like", "unknown"}
)
_EXACT_MATERIAL_TERMS = frozenset(
    {"bronze", "copper", "diamond", "emerald", "gold", "iron", "ruby", "sapphire", "silver", "steel"}
)
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "archive_member_path",
        "creator",
        "filename",
        "local_path",
        "member_path",
        "original_filename",
        "pack",
        "path",
        "source",
        "source_pack",
        "source_path",
        "source_url",
    }
)

# Values are explicit edges, not similarity hints.  Broad-field leaf nodes are
# included so a mis-granular but declared value such as weapon/sword can be
# compared without accepting arbitrary strings.
_HIERARCHIES: dict[str, dict[str, tuple[str, ...]]] = {
    "domain": {
        "visual_icon": ("item_icon",),
        "item_icon": (
            "inventory_icon",
            "equipment_icon",
            "resource_icon",
            "organic_icon",
            "spell_icon",
        ),
        "organic_icon": ("food_icon", "plant_icon"),
        "unknown": (),
    },
    "broad_category": {
        "item": (
            "equipment_item",
            "access_item",
            "resource_item",
            "organic_item",
            "vessel_item",
            "symbolic_item",
            "misc_item",
        ),
        "equipment_item": ("weapon", "armor", "tool", "jewelry", "clothing"),
        "access_item": ("key",),
        "resource_item": ("gem", "material", "mineral"),
        "organic_item": ("plant", "food"),
        "vessel_item": ("container", "potion"),
        "symbolic_item": ("spell",),
        "weapon": ("sword", "dagger", "bow", "axe", "spear", "staff", "wand"),
        "armor": ("helmet", "shield", "boots", "gloves", "chest armor"),
        "tool": ("hammer", "hoe", "pickaxe", "shovel", "sickle"),
        "unknown": (),
    },
    "canonical_object": {
        "object": (
            "amulet",
            "apple",
            "arrow",
            "axe",
            "bag",
            "belt",
            "berry",
            "bone",
            "book",
            "boots",
            "bottle",
            "bow",
            "bracelet",
            "bread",
            "bucket",
            "carrot",
            "cheese",
            "chest",
            "chest armor",
            "cloak",
            "club",
            "coin",
            "crystal",
            "cut gemstone",
            "dagger",
            "egg",
            "feather",
            "fish",
            "flame",
            "flower",
            "gloves",
            "hammer",
            "hat",
            "helmet",
            "herb",
            "hoe",
            "ingot",
            "key",
            "lantern",
            "leaf",
            "lightning bolt",
            "magic orb",
            "map",
            "meat",
            "mushroom",
            "necklace",
            "ore chunk",
            "pants",
            "pickaxe",
            "plant sprig",
            "potion bottle",
            "pouch",
            "ring",
            "robe",
            "rope",
            "rune",
            "scroll",
            "seed",
            "shell",
            "shield",
            "shirt",
            "shoes",
            "shovel",
            "sickle",
            "spear",
            "spell book",
            "staff",
            "sword",
            "torch",
            "tree branch",
            "wand",
        ),
        "botanical_form": ("flower", "herb", "leaf", "plant sprig", "tree branch"),
        "gem_form": ("crystal", "cut gemstone", "ore chunk"),
        "unknown": (),
    },
    "role": {
        "semantic_role": ("combat", "decoration", "resource_role", "wearable_role", "utility_role"),
        "combat": ("combat_weapon", "protective_equipment"),
        "decoration": ("decorative_item",),
        "resource_role": ("crafting_resource",),
        "wearable_role": ("wearable_accessory", "wearable_clothing"),
        "utility_role": (
            "access_token",
            "consumable_food",
            "consumable_potion",
            "crafting_tool",
            "cutting_tool",
            "mining_tool",
            "spell_effect",
            "storage_container",
            "misc_item",
        ),
        "unknown": (),
    },
}

# Only these sibling groups have a visually safe parent.  A generic common
# root never makes two values compatible (helmet/mineral remains a conflict).
_SAFE_SIBLING_PARENTS = {
    "domain": frozenset({"organic_icon"}),
    "broad_category": frozenset({"organic_item", "vessel_item"}),
    "canonical_object": frozenset({"botanical_form", "gem_form"}),
    "role": frozenset(),
}


class ProviderAdapter(Protocol):
    """Adapter boundary for provider-specific serialization only.

    Implementations may translate the neutral request, but this backend never
    sends it and contains no provider-specific HTTP behavior.
    """

    def format_request(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FieldValidation:
    valid: bool
    state: str
    value: Any
    reason: str | None = None
    taxonomy_valid: bool = True


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": (numerator / denominator) if denominator else None,
    }


def _json_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _field_name(name: str) -> str:
    return _FIELD_ALIASES.get(str(name), str(name))


def is_controlled_conditional_unknown(field_name: str, value: Any) -> bool:
    """Return true only for the exact controlled conditional unknown enum.

    Deliberately do not strip, case-fold, tokenize, or substring-match.  Invalid
    text remains invalid output instead of being silently converted to an
    abstention.
    """

    return _field_name(field_name) in CONDITIONAL_FIELDS and value == _CONDITIONAL_UNKNOWN_SENTINEL


def unknown_sentinel_policy() -> dict[str, Any]:
    return {
        "schema_version": UNKNOWN_SENTINEL_POLICY_VERSION,
        "applies_to": list(CONDITIONAL_FIELDS),
        "canonical_unknown_enum": _CONDITIONAL_UNKNOWN_SENTINEL,
        "matching": "exact_case_sensitive_full_value",
        "substring_matching": False,
        "fuzzy_matching": False,
        "whitespace_or_case_normalization": False,
        "documented_legacy_schemas": sorted(_DOCUMENTED_LEGACY_UNKNOWN_SCHEMAS),
        "documented_legacy_sentinels": [_CONDITIONAL_UNKNOWN_SENTINEL],
        "unrecognized_placeholders_are_invalid_output": [
            "unspecified",
            "not sure",
            "ambiguous",
            "none",
        ],
        "labeled_unknown_normalizes_to": {
            "state": "model_abstained",
            "value": None,
            "reason": "provider_returned_unknown",
        },
    }


def conditional_identity_contract() -> dict[str, Any]:
    return {
        "schema_version": CONDITIONAL_INVARIANT_VERSION,
        "fields": list(CONDITIONAL_FIELDS),
        "invariants": {
            "labeled": {
                "value": "concrete_controlled_taxonomy_value",
                "unknown_sentinel_permitted": False,
                "null_permitted": False,
            },
            "model_abstained": {"value": None, "controlled_reason_required": True},
            "not_applicable": {"value": None},
            "invalid_output": {"supervision_value_permitted": False},
        },
        "role_requires_direct_visual_demonstration": True,
        "canonical_object_requires_unmistakable_visual_identity": True,
        "sentinel_policy_version": UNKNOWN_SENTINEL_POLICY_VERSION,
        "provider_normalization_version": PROVIDER_NORMALIZATION_VERSION,
        "historical_read_adapter_version": HISTORICAL_READ_ADAPTER_VERSION,
    }


def taxonomy_hierarchy() -> dict[str, Any]:
    return {
        "schema_version": TAXONOMY_VERSION,
        "compatibility_is_explicit_only": True,
        "string_similarity_used": False,
        "hierarchies": {
            field: {parent: list(children) for parent, children in edges.items()}
            for field, edges in _HIERARCHIES.items()
        },
        "safe_sibling_parents": {field: sorted(parents) for field, parents in _SAFE_SIBLING_PARENTS.items()},
    }


def hierarchical_label_contract() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "levels": {
            "core_visual": {
                "fields": list(CORE_FIELDS),
                "purpose": "broad_visually_supportable_identity_only",
                "candidate_requirements": [
                    "visual_evidence_supports_value",
                    "taxonomy_output_valid",
                    "independent_exact_or_explicit_parent_child_agreement",
                    "no_true_core_contradiction",
                    "new_independent_health_gate_passes",
                ],
            },
            "conditional_identity": {
                "fields": list(CONDITIONAL_FIELDS),
                "default": "model_abstained",
                "rule": "label_only_when_visually_unmistakable_or_directly_demonstrated",
                "unknown_is_identity": False,
                "labeled_unknown_forbidden": True,
                "does_not_invalidate_core": True,
            },
            "auxiliary_visual": {
                "fields": list(AUXILIARY_FIELDS),
                "purposes": ["search", "review", "weak_metadata"],
                "determines_core_training_eligibility": False,
            },
        },
        "forbidden_inferences": [
            "gameplay_function",
            "object_use",
            "context_only_exact_identity",
            "filename_semantics",
            "source_pack_identity",
            "exact_material_from_appearance",
        ],
        "material_cues": sorted(_MATERIAL_CUES),
        "historical_category_alias": {"category": "broad_category"},
        "conditional_identity_invariant": conditional_identity_contract(),
    }


def field_state_contract() -> dict[str, Any]:
    return {
        "schema_version": FIELD_STATE_VERSION,
        "states": list(FIELD_STATES),
        "abstention_state": "model_abstained",
        "abstention_requires_null_value": True,
        "abstention_requires_reason": True,
        "abstention_reasons": list(ABSTENTION_REASONS),
        "empty_successful_label_forbidden": True,
        "conditional_identity_invariant_version": CONDITIONAL_INVARIANT_VERSION,
        "conditional_labeled_unknown_forbidden": True,
        "invalid_output_has_no_supervision_value": True,
    }


def disagreement_classification_contract() -> dict[str, Any]:
    return {
        "schema_version": COMPARISON_VERSION,
        "classifications": list(COMPARISON_CLASSIFICATIONS),
        "field_aware": True,
        "taxonomy_version": TAXONOMY_VERSION,
        "string_similarity_used": False,
        "true_contradiction_examples": [
            {"field": "broad_category", "left": "helmet", "right": "mineral"},
            {"field": "role", "left": "combat", "right": "decoration"},
        ],
        "compatible_example": {"field": "broad_category", "left": "weapon", "right": "sword"},
    }


def field_health_gate_contract() -> dict[str, Any]:
    return {
        "schema_version": HEALTH_GATE_VERSION,
        "required_rates": [
            "domain_disagreement",
            "broad_category_disagreement",
            "canonical_object_disagreement",
            "role_disagreement",
            "combined_core_disagreement",
            "conditional_field_abstention",
            "conditional_labeled_unknown_count",
            "conditional_labeled_unknown_rate",
            "reconciled_labeled_unknown_count",
            "conditional_abstention_count",
            "conditional_abstention_rate",
            "conditional_concrete_label_count",
            "conditional_invalid_state_count",
            "invalid_output",
            "taxonomy_invalidity",
            "filename_leakage",
            "hash_mismatch",
            "missing_record",
            "duplicate_record",
        ],
        "core_success_gates": {
            "domain_true_contradiction_rate_max": 0.05,
            "broad_category_true_contradiction_rate_max": 0.05,
            "combined_core_true_contradiction_rate_max": 0.05,
            "taxonomy_invalidity": 0,
            "filename_leakage": 0,
            "hash_mismatch": 0,
            "missing_record_rate": 0,
            "duplicate_record_rate": 0,
            "conditional_labeled_unknown_count": 0,
            "reconciled_labeled_unknown_count": 0,
            "conditional_invalid_state_count": 0,
        },
        "compatible_hierarchy_counts_as_true_contradiction": False,
        "one_side_abstention_reported_separately": True,
        "all_rates_include_numerator_and_denominator": True,
        "conditional_pass_field_denominator": "records_times_passes_times_two_conditional_fields",
        "conditional_reconciled_field_denominator": "records_times_two_conditional_fields",
    }


def health_metric_contract() -> dict[str, Any]:
    return {
        "schema_version": "sprite_lab_conditional_health_metric_contract_v1",
        "pass_field_denominator": "provider_records_times_two_conditional_fields",
        "reconciled_field_denominator": "reconciled_records_times_two_conditional_fields",
        "metrics": {
            "conditional_labeled_unknown_count": "labeled exact controlled unknown values in normalized pass fields",
            "conditional_labeled_unknown_rate": "conditional_labeled_unknown_count / pass_field_denominator",
            "reconciled_labeled_unknown_count": "labeled exact controlled unknown values after reconciliation",
            "conditional_abstention_count": "valid model_abstained normalized pass fields",
            "conditional_abstention_rate": "conditional_abstention_count / pass_field_denominator",
            "conditional_concrete_label_count": "valid labeled non-unknown controlled pass fields",
            "conditional_invalid_state_count": "invalid or internally routed pass field states",
        },
        "required_zero_gates": [
            "conditional_labeled_unknown_count",
            "reconciled_labeled_unknown_count",
            "conditional_invalid_state_count",
        ],
        "numerators_and_denominators_required": True,
        "exact_unknown_comparison_classification_after_normalization": "both_abstained",
    }


def reconciliation_contract() -> dict[str, Any]:
    return {
        "schema_version": RECONCILIATION_VERSION,
        "core": {
            "exact_agreement": "candidate_pending_new_health_gate",
            "compatible_hierarchy": "safest_shared_parent_candidate_pending_new_health_gate",
            "one_side_abstention": "model_abstained",
            "both_abstained": "model_abstained",
            "taxonomy_granularity_difference": "model_abstained_or_pending_taxonomy_review",
            "true_contradiction": "pending_review",
            "invalid_comparison": "invalid_output",
        },
        "conditional": {
            "exact_agreement": "requires_independent_unmistakable_visual_evidence",
            "unknown_plus_unknown": "model_abstained_null",
            "unknown_plus_model_abstained": "model_abstained_null",
            "model_abstained_plus_unknown": "model_abstained_null",
            "unknown_plus_concrete": "model_abstained_null",
            "different_concrete_values": "model_abstained_null",
            "same_concrete_with_weak_evidence": "model_abstained_null",
            "shared_unknown_is_exact_identity_agreement": False,
            "any_disagreement": "model_abstained",
            "one_side_abstention": "model_abstained",
            "ambiguous": "model_abstained",
        },
        "auxiliary": {
            "preserve_both_proposals": True,
            "creates_strong_supervision": False,
            "blocks_valid_core": False,
        },
        "select_more_specific_for_information_gain": False,
    }


def review_queue_contract() -> dict[str, Any]:
    return {
        "schema_version": REVIEW_QUEUE_VERSION,
        "queue_reasons": [
            "true_core_contradiction",
            "unresolved_taxonomy_mapping",
            "uncertain_broad_category_for_conditioned_view",
            "required_label_invalid_output",
            "explicit_calibration_sample",
        ],
        "not_queued": [
            "conditional_field_abstention",
            "conditional_one_side_abstention",
            "conditional_disagreement",
            "conditional_both_side_abstention",
            "auxiliary_disagreement",
            "safe_image_only_record",
            "already_excluded_unsuitable_record",
        ],
        "required_prefill": [
            "image_identity",
            "current_conservative_result",
            "pass_a_proposal",
            "health_check_or_pass_b_proposal",
            "field_level_disagreement",
            "recommended_safe_action",
            "allowed_actions",
            "review_reason",
        ],
        "allowed_actions": list(REVIEW_ACTIONS),
        "event_contract": "append_only",
        "review_may_change_provenance_license_or_image_identity": False,
    }


def calibration_input_contract() -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_INPUT_VERSION,
        "fields": list(CALIBRATION_FIELDS),
        "states": list(CALIBRATION_STATES),
        "truth_states": ["human_verified", "human_abstained"],
        "model_agreement_is_truth": False,
        "conditional_unknown_acceptance_contribution": 0,
        "conditional_unknown_abstention_contribution": 1,
        "conditional_unknown_correctness_rows": 0,
        "conditional_unknown_human_truth_rows": 0,
        "metrics": ["accuracy", "coverage", "abstention_rate", "risk_at_accepted_coverage", "confidence_bins"],
        "readiness_states": ["not_ready", "insufficient_truth", "ready"],
    }


def historical_compatibility_contract() -> dict[str, Any]:
    return {
        "schema_version": HISTORICAL_READ_ADAPTER_VERSION,
        "source_artifacts_mutated": False,
        "documented_source_schemas": sorted(_DOCUMENTED_LEGACY_UNKNOWN_SCHEMAS),
        "conditional_labeled_unknown_exposed_as": {
            "state": "model_abstained",
            "value": None,
            "reason": "legacy_unknown_normalized",
        },
        "diagnostic_metadata_retained": [
            "original_schema_version",
            "original_raw_state",
            "original_raw_value",
            "source_artifact_identity",
            "normalization_marker",
        ],
        "human_review_inferred": False,
        "supervision_strength_added": False,
    }


def conservative_prompt_v3() -> str:
    return """# Sprite Lab conservative blind visual-label prompt v3

Inspect pixels and opaque image identity only. Prefer a broad correct class over
a specific guess. Abstain when multiple objects are plausible. For
`canonical_object` and `role`, never emit `unknown` as a labeled value. When a
conditional identity is unknown, emit `model_abstained`, use a null value, and
include one controlled abstention reason. Exact agreement on `unknown` is still
abstention, never exact identity agreement. `canonical_object` may be labeled
only when the exact object is unmistakable. `role` may be labeled only when the
role is directly demonstrated visually. Insufficient resolution, several
plausible identities, or no safe exact taxonomy value require abstention.

Do not infer gameplay use, object use, filename meaning, source or pack
identity, or exact material. Material cues must remain visual (for example
metal-like, wood-like, or stone-like).

Describe pairs, sets, bundles, and clusters explicitly. Use the shared parent
when fine core identity is uncertain and the versioned taxonomy declares that
parent. Report low-resolution ambiguity and abstain with a structured reason.
Provider conditional fields use exactly one of `labeled`, `model_abstained`, or
`not_applicable`; invalid output is diagnosed at the boundary, while conflict
and pending review are reconciliation states. An abstention has a null value and
one allowed structured abstention reason.
Never use filename, path, creator, source, pack, prior label, or provenance
metadata. These are model proposals, not human truth.
"""


def conservative_prompt_v2() -> str:
    """Compatibility entry point returning the current conservative prompt."""

    return conservative_prompt_v3()


def _conditional_provider_field_schema(field_name: str) -> dict[str, Any]:
    evidence_flag = "visually_demonstrated" if field_name == "role" else "visually_unmistakable"
    common_properties: dict[str, Any] = {
        "state": {},
        "value": {},
        "reason": {"type": ["string", "null"]},
        "evidence": {"type": ["string", "null"]},
        "visually_unmistakable": {"type": "boolean"},
        "visually_demonstrated": {"type": "boolean"},
    }
    required = ["state", "value", "reason", "evidence", evidence_flag]
    return {
        "additionalProperties": False,
        "properties": common_properties,
        "required": required,
        "type": "object",
        "oneOf": [
            {
                "properties": {
                    "state": {"const": "labeled"},
                    "value": {"enum": sorted(taxonomy_values(field_name) - {_CONDITIONAL_UNKNOWN_SENTINEL})},
                    "reason": {"type": "null"},
                    "evidence": {"minLength": 1, "type": "string"},
                    evidence_flag: {"const": True},
                },
                "required": required,
            },
            {
                "properties": {
                    "state": {"const": "model_abstained"},
                    "value": {"type": "null"},
                    "reason": {"enum": list(ABSTENTION_REASONS)},
                },
                "required": required,
            },
            {
                "properties": {
                    "state": {"const": "not_applicable"},
                    "value": {"type": "null"},
                    "reason": {"type": ["string", "null"]},
                },
                "required": required,
            },
        ],
    }


def conservative_output_schema() -> dict[str, Any]:
    field_schema = {
        "additionalProperties": False,
        "properties": {
            "state": {"enum": list(FIELD_STATES)},
            "value": {},
            "reason": {"type": ["string", "null"]},
            "evidence": {"type": ["string", "null"]},
            "visually_unmistakable": {"type": "boolean"},
            "visually_demonstrated": {"type": "boolean"},
        },
        "required": ["state", "value", "reason", "evidence"],
        "type": "object",
        "allOf": [
            {
                "if": {"properties": {"state": {"const": "model_abstained"}}},
                "then": {
                    "properties": {
                        "value": {"type": "null"},
                        "reason": {"enum": list(ABSTENTION_REASONS)},
                    }
                },
            }
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "additionalProperties": False,
        "properties": {
            "record_id": {"pattern": r"^rec_[0-9a-f]{64}$", "type": "string"},
            "image_sha256": {"pattern": r"^[0-9a-f]{64}$", "type": "string"},
            "fields": {
                "additionalProperties": False,
                "properties": {
                    field: (
                        _conditional_provider_field_schema(field)
                        if field in CONDITIONAL_FIELDS
                        else _json_copy(field_schema)
                    )
                    for field in SEMANTIC_FIELDS
                },
                "required": list(SEMANTIC_FIELDS),
                "type": "object",
            },
            "prompt_version": {"const": PROMPT_VERSION},
        },
        "required": ["record_id", "image_sha256", "fields", "prompt_version"],
        "type": "object",
    }


def validate_field(field_name: str, proposal: Mapping[str, Any] | None) -> FieldValidation:
    name = _field_name(field_name)
    if not isinstance(proposal, Mapping):
        return FieldValidation(False, "invalid_output", None, "field_output_missing", False)
    state = str(proposal.get("state") or "")
    value = proposal.get("value")
    reason = str(proposal.get("reason") or "") or None
    if state not in FIELD_STATES:
        return FieldValidation(False, "invalid_output", None, "uncontrolled_field_state", False)
    if state == "labeled":
        if value is None or value == "" or value == [] or value == {}:
            return FieldValidation(False, "invalid_output", None, "empty_successful_label", False)
        if name in CONDITIONAL_FIELDS and is_controlled_conditional_unknown(name, value):
            return FieldValidation(
                False,
                "invalid_output",
                None,
                "conditional_identity_unknown_must_abstain",
                False,
            )
        if name in _HIERARCHIES and str(value) not in taxonomy_values(name):
            return FieldValidation(False, "invalid_output", None, "unknown_taxonomy_value", False)
        if name == "visual_material_cue" and str(value) not in _MATERIAL_CUES:
            material_reason = (
                "exact_material_overclaim"
                if str(value).casefold() in _EXACT_MATERIAL_TERMS
                else "invalid_visual_material_cue"
            )
            return FieldValidation(False, "invalid_output", None, material_reason, False)
    elif state == "model_abstained":
        if value is not None:
            return FieldValidation(False, "invalid_output", None, "abstention_value_must_be_null", True)
        if reason not in ABSTENTION_REASONS:
            return FieldValidation(False, "invalid_output", None, "invalid_or_missing_abstention_reason", True)
    elif state in {"not_applicable", "invalid_output", "conflict", "pending_review"} and value is not None:
        return FieldValidation(False, "invalid_output", None, f"{state}_value_must_be_null", True)
    return FieldValidation(True, state, _json_copy(value), reason, True)


def taxonomy_values(field_name: str) -> frozenset[str]:
    edges = _HIERARCHIES.get(_field_name(field_name), {})
    values = set(edges)
    for children in edges.values():
        values.update(children)
    return frozenset(values)


def provider_normalization_contract() -> dict[str, Any]:
    return {
        "schema_version": PROVIDER_NORMALIZATION_VERSION,
        "applies_to": list(CONDITIONAL_FIELDS),
        "sentinel_policy_version": UNKNOWN_SENTINEL_POLICY_VERSION,
        "matrix": [
            {
                "input": {"state": "labeled", "value": "unknown"},
                "output": {
                    "state": "model_abstained",
                    "value": None,
                    "reason": "provider_returned_unknown",
                },
            },
            {
                "input": {"state": "labeled", "value": None},
                "output": {"state": "invalid_output", "value": None, "diagnostic_preserved": True},
            },
            {
                "input": {"state": "model_abstained", "value": "non_null"},
                "output": {
                    "state": "invalid_output",
                    "value": None,
                    "discarded_value_diagnostic_preserved": True,
                },
            },
            {
                "input": {"state": "labeled", "value": "arbitrary_invalid_text"},
                "output": {
                    "state": "invalid_output",
                    "value": None,
                    "converted_to_abstention": False,
                    "diagnostic_preserved": True,
                },
            },
        ],
    }


def normalize_provider_field(field_name: str, proposal: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize one provider conditional field before semantic validation.

    The function is fail-closed.  Only the exact controlled unknown sentinel is
    converted to abstention; every other invalid output remains diagnosable as
    ``invalid_output`` and carries no supervision value.
    """

    name = _field_name(field_name)
    if name not in CONDITIONAL_FIELDS:
        return _json_copy(proposal) if isinstance(proposal, Mapping) else {}
    if not isinstance(proposal, Mapping):
        return {
            "state": "invalid_output",
            "value": None,
            "reason": "field_output_missing",
            "normalization": {
                "schema_version": PROVIDER_NORMALIZATION_VERSION,
                "marker": "invalid_provider_output_preserved",
                "raw_state": None,
                "raw_value": None,
                "source_was_mapping": False,
            },
            "taxonomy_valid": False,
        }

    normalized = _json_copy(proposal)
    state = str(proposal.get("state") or "")
    value = _json_copy(proposal.get("value"))
    diagnostic = {
        "schema_version": PROVIDER_NORMALIZATION_VERSION,
        "raw_state": state,
        "raw_value": value,
        "source_was_mapping": True,
    }
    if state == "labeled" and is_controlled_conditional_unknown(name, value):
        normalized.update(
            {
                "state": "model_abstained",
                "value": None,
                "reason": "provider_returned_unknown",
                "normalization": {**diagnostic, "marker": "controlled_unknown_normalized"},
                "taxonomy_valid": True,
            }
        )
        return normalized

    validation = validate_field(name, normalized)
    if validation.valid:
        normalized["state"] = validation.state
        normalized["value"] = _json_copy(validation.value)
        if validation.reason is not None:
            normalized["reason"] = validation.reason
        normalized["taxonomy_valid"] = validation.taxonomy_valid
        return normalized

    normalized.update(
        {
            "state": "invalid_output",
            "value": None,
            "reason": validation.reason or "invalid_provider_output",
            "normalization": {
                **diagnostic,
                "marker": "invalid_provider_output_preserved",
                "discarded_non_null_value": value is not None,
            },
            "taxonomy_valid": validation.taxonomy_valid,
        }
    )
    return normalized


def normalize_provider_output(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized semantic view without mutating provider bytes/data."""

    normalized = _json_copy(record)
    source_fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else record
    normalized_fields = normalized.get("fields") if isinstance(normalized.get("fields"), Mapping) else normalized
    for field_name in CONDITIONAL_FIELDS:
        raw = source_fields.get(field_name)
        normalized_fields[field_name] = normalize_provider_field(field_name, raw)
    normalized["provider_normalization"] = {
        "schema_version": PROVIDER_NORMALIZATION_VERSION,
        "source_schema_version": str(record.get("schema_version") or "unknown"),
        "source_record_rewritten": False,
        "human_reviewed": False,
    }
    return normalized


def _parents(field_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for parent, children in _HIERARCHIES.get(_field_name(field_name), {}).items():
        for child in children:
            # A more specific declared grouping wins over the generic object root.
            if child not in result or result[child] == "object":
                result[child] = parent
    return result


def _ancestors(field_name: str, value: str) -> list[str]:
    parents = _parents(field_name)
    result: list[str] = []
    current = value
    seen: set[str] = set()
    while current in parents and current not in seen:
        seen.add(current)
        current = parents[current]
        result.append(current)
    return result


def safest_shared_parent(field_name: str, left: str, right: str) -> str | None:
    name = _field_name(field_name)
    if left == right:
        return left
    left_ancestors = _ancestors(name, left)
    right_ancestors = _ancestors(name, right)
    if left in right_ancestors:
        return left
    if right in left_ancestors:
        return right
    common = [value for value in left_ancestors if value in right_ancestors]
    if not common:
        return None
    parent = common[0]
    return parent if parent in _SAFE_SIBLING_PARENTS.get(name, frozenset()) else None


def compare_field(
    field_name: str,
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, Any]:
    name = _field_name(field_name)
    normalized_left = normalize_provider_field(name, left) if name in CONDITIONAL_FIELDS else left
    normalized_right = normalize_provider_field(name, right) if name in CONDITIONAL_FIELDS else right
    left_valid = validate_field(name, normalized_left)
    right_valid = validate_field(name, normalized_right)
    left_taxonomy_valid = left_valid.taxonomy_valid and bool(
        normalized_left.get("taxonomy_valid", True) if isinstance(normalized_left, Mapping) else True
    )
    right_taxonomy_valid = right_valid.taxonomy_valid and bool(
        normalized_right.get("taxonomy_valid", True) if isinstance(normalized_right, Mapping) else True
    )
    base = {
        "schema_version": COMPARISON_VERSION,
        "field": name,
        "left": {"state": left_valid.state, "value": _json_copy(left_valid.value)},
        "right": {"state": right_valid.state, "value": _json_copy(right_valid.value)},
        "taxonomy_version": TAXONOMY_VERSION,
        "safe_parent": None,
        "normalization": {
            "left": _json_copy(normalized_left.get("normalization"))
            if isinstance(normalized_left, Mapping) and normalized_left.get("normalization")
            else None,
            "right": _json_copy(normalized_right.get("normalization"))
            if isinstance(normalized_right, Mapping) and normalized_right.get("normalization")
            else None,
        },
    }
    if not left_valid.valid or not right_valid.valid:
        return {
            **base,
            "classification": "invalid_comparison",
            "reasons": [reason for reason in (left_valid.reason, right_valid.reason) if reason],
            "taxonomy_valid": left_taxonomy_valid and right_taxonomy_valid,
        }
    if "not_applicable" in {left_valid.state, right_valid.state}:
        return {**base, "classification": "not_applicable", "taxonomy_valid": True}
    if left_valid.state == right_valid.state == "model_abstained":
        return {**base, "classification": "both_abstained", "taxonomy_valid": True}
    if "model_abstained" in {left_valid.state, right_valid.state}:
        return {**base, "classification": "one_side_abstention", "taxonomy_valid": True}
    if left_valid.state != "labeled" or right_valid.state != "labeled":
        reasons = []
        for proposal, validation in ((normalized_left, left_valid), (normalized_right, right_valid)):
            proposal_reason = proposal.get("reason") if isinstance(proposal, Mapping) else None
            if validation.state == "invalid_output" and proposal_reason:
                reasons.append(str(proposal_reason))
        return {
            **base,
            "classification": "invalid_comparison",
            "reasons": reasons,
            "taxonomy_valid": left_taxonomy_valid and right_taxonomy_valid,
        }
    left_value = str(left_valid.value)
    right_value = str(right_valid.value)
    if left_value == right_value:
        return {**base, "classification": "exact_agreement", "taxonomy_valid": True}
    if _CONDITIONAL_UNKNOWN_SENTINEL in {left_value, right_value}:
        return {
            **base,
            "classification": "taxonomy_granularity_difference",
            "taxonomy_valid": True,
        }
    safe_parent = safest_shared_parent(name, left_value, right_value)
    if safe_parent is not None:
        direct_hierarchy = left_value in _ancestors(name, right_value) or right_value in _ancestors(name, left_value)
        classification = "compatible_hierarchy" if direct_hierarchy else "taxonomy_granularity_difference"
        # Explicitly safe siblings are reconcilable to a shared parent even
        # though the diagnostic name distinguishes them from parent/child.
        return {
            **base,
            "classification": classification,
            "safe_parent": safe_parent,
            "taxonomy_valid": True,
        }
    return {**base, "classification": "true_contradiction", "taxonomy_valid": True}


def _abstention(reason: str) -> dict[str, Any]:
    return {"state": "model_abstained", "value": None, "reason": reason, "training_eligible": False}


def _conditional_is_unmistakable(proposal: Mapping[str, Any] | None, field_name: str) -> bool:
    if not isinstance(proposal, Mapping):
        return False
    evidence = str(proposal.get("evidence") or proposal.get("visual_evidence") or "").strip()
    if field_name == "role":
        return bool(proposal.get("visually_demonstrated") and evidence)
    return bool(proposal.get("visually_unmistakable") and evidence)


def reconcile_proposals(
    pass_a: Mapping[str, Any],
    pass_b: Mapping[str, Any],
    *,
    image_only_eligible: bool = True,
    semantic_label_required: bool = False,
) -> dict[str, Any]:
    normalized_pass_a = normalize_provider_output(pass_a)
    normalized_pass_b = normalize_provider_output(pass_b)
    left_fields = (
        normalized_pass_a.get("fields") if isinstance(normalized_pass_a.get("fields"), Mapping) else normalized_pass_a
    )
    right_fields = (
        normalized_pass_b.get("fields") if isinstance(normalized_pass_b.get("fields"), Mapping) else normalized_pass_b
    )
    fields: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    mandatory_review_reasons: list[str] = []
    for field_name in SEMANTIC_FIELDS:
        legacy_name = "category" if field_name == "broad_category" else field_name
        left = left_fields.get(field_name, left_fields.get(legacy_name))
        right = right_fields.get(field_name, right_fields.get(legacy_name))
        comparison = compare_field(field_name, left, right)
        comparisons[field_name] = comparison
        classification = comparison["classification"]
        if field_name in CORE_FIELDS:
            if classification == "exact_agreement":
                result = {
                    "state": "labeled",
                    "value": _json_copy(comparison["left"]["value"]),
                    "reason": "exact_independent_model_agreement",
                    "candidate_status": "pending_new_independent_health_gate",
                    "training_eligible": False,
                }
            elif classification in {"compatible_hierarchy", "taxonomy_granularity_difference"} and comparison.get(
                "safe_parent"
            ):
                result = {
                    "state": "labeled",
                    "value": comparison["safe_parent"],
                    "reason": "explicit_safest_shared_parent",
                    "candidate_status": "pending_new_independent_health_gate",
                    "training_eligible": False,
                }
            elif classification == "true_contradiction":
                result = {
                    "state": "pending_review",
                    "value": None,
                    "reason": "true_core_contradiction",
                    "training_eligible": False,
                }
                mandatory_review_reasons.append(f"{field_name}:true_core_contradiction")
            elif classification == "invalid_comparison":
                result = {
                    "state": "invalid_output",
                    "value": None,
                    "reason": "invalid_core_output",
                    "training_eligible": False,
                }
                if semantic_label_required:
                    mandatory_review_reasons.append(f"{field_name}:required_label_invalid_output")
            elif classification == "not_applicable":
                result = {
                    "state": "not_applicable",
                    "value": None,
                    "reason": "field_not_applicable",
                    "training_eligible": False,
                }
            elif classification == "taxonomy_granularity_difference":
                result = _abstention("taxonomy_has_no_safe_match")
                mandatory_review_reasons.append(f"{field_name}:unresolved_taxonomy_mapping")
            else:
                result = _abstention("visually_ambiguous")
        elif field_name in CONDITIONAL_FIELDS:
            if (
                classification == "exact_agreement"
                and _conditional_is_unmistakable(left, field_name)
                and _conditional_is_unmistakable(right, field_name)
            ):
                result = {
                    "state": "labeled",
                    "value": _json_copy(comparison["left"]["value"]),
                    "reason": "exact_agreement_with_direct_visual_support",
                    "candidate_status": "conditional_model_candidate_not_truth",
                    "training_eligible": False,
                }
            elif classification == "not_applicable":
                result = {
                    "state": "not_applicable",
                    "value": None,
                    "reason": "field_not_applicable",
                    "training_eligible": False,
                }
            elif classification == "invalid_comparison":
                result = {
                    "state": "invalid_output",
                    "value": None,
                    "reason": "invalid_conditional_output",
                    "diagnostics": {
                        "comparison_reasons": _json_copy(comparison.get("reasons") or []),
                        "normalization": _json_copy(comparison.get("normalization")),
                    },
                    "training_eligible": False,
                }
            else:
                reason = "role_not_visually_demonstrated" if field_name == "role" else "multiple_plausible_objects"
                result = _abstention(reason)
        else:
            if classification == "exact_agreement":
                result = {
                    "state": "labeled",
                    "value": _json_copy(comparison["left"]["value"]),
                    "reason": "auxiliary_exact_agreement",
                    "training_eligible": False,
                }
            elif classification == "both_abstained":
                result = _abstention("visually_ambiguous")
            elif classification == "not_applicable":
                result = {
                    "state": "not_applicable",
                    "value": None,
                    "reason": "field_not_applicable",
                    "training_eligible": False,
                }
            elif classification == "invalid_comparison":
                result = {
                    "state": "invalid_output",
                    "value": None,
                    "reason": "invalid_auxiliary_output",
                    "training_eligible": False,
                }
            else:
                result = {
                    "state": "conflict",
                    "value": None,
                    "reason": "auxiliary_proposals_preserved_without_promotion",
                    "proposals": [_json_copy(left), _json_copy(right)],
                    "training_eligible": False,
                }
        fields[field_name] = result
    return {
        "schema_version": RECONCILIATION_VERSION,
        "record_id": str(pass_a.get("record_id") or pass_b.get("record_id") or ""),
        "image_sha256": str(pass_a.get("image_sha256") or pass_b.get("image_sha256") or ""),
        "fields": fields,
        "comparisons": comparisons,
        "mandatory_review": bool(mandatory_review_reasons),
        "mandatory_review_reasons": mandatory_review_reasons,
        "image_only_eligible": bool(image_only_eligible),
        "semantic_abstention_changes_image_only_eligibility": False,
        "semantic_label_required": semantic_label_required,
        "model_agreement_is_human_truth": False,
        "provider_normalization_version": PROVIDER_NORMALIZATION_VERSION,
    }


def _proposal_field(record: Mapping[str, Any], field_name: str) -> Mapping[str, Any] | None:
    fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else record
    legacy_name = "category" if field_name == "broad_category" else field_name
    value = fields.get(field_name, fields.get(legacy_name))
    return value if isinstance(value, Mapping) else None


def conditional_identity_health(
    records: Sequence[Mapping[str, Any]],
    reconciled_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Measure conditional semantics already exposed to current logic.

    Callers that read provider or historical records normalize them first.  A
    raw labeled-unknown record therefore fails this health gate, while a record
    passed through the versioned boundary contributes one abstention.
    """

    pass_denominator = len(records) * len(CONDITIONAL_FIELDS)
    reconciled_denominator = len(reconciled_records) * len(CONDITIONAL_FIELDS)
    labeled_unknown = 0
    abstentions = 0
    concrete_labels = 0
    invalid_states = 0
    normalized_unknowns = 0
    for record in records:
        for field_name in CONDITIONAL_FIELDS:
            proposal = _proposal_field(record, field_name)
            state = str(proposal.get("state") or "") if isinstance(proposal, Mapping) else ""
            value = proposal.get("value") if isinstance(proposal, Mapping) else None
            if state == "labeled" and is_controlled_conditional_unknown(field_name, value):
                labeled_unknown += 1
            validation = validate_field(field_name, proposal)
            if validation.valid and validation.state == "model_abstained":
                abstentions += 1
            elif validation.valid and validation.state == "labeled":
                concrete_labels += 1
            elif not validation.valid or validation.state in {"invalid_output", "conflict", "pending_review"}:
                invalid_states += 1
            normalization = proposal.get("normalization") if isinstance(proposal, Mapping) else None
            if isinstance(normalization, Mapping) and normalization.get("marker") == "controlled_unknown_normalized":
                normalized_unknowns += 1

    reconciled_labeled_unknown = 0
    for record in reconciled_records:
        for field_name in CONDITIONAL_FIELDS:
            proposal = _proposal_field(record, field_name)
            if (
                isinstance(proposal, Mapping)
                and proposal.get("state") == "labeled"
                and is_controlled_conditional_unknown(field_name, proposal.get("value"))
            ):
                reconciled_labeled_unknown += 1

    metric_details = {
        "conditional_labeled_unknown_count": _rate(labeled_unknown, pass_denominator),
        "conditional_abstention_count": _rate(abstentions, pass_denominator),
        "conditional_concrete_label_count": _rate(concrete_labels, pass_denominator),
        "conditional_invalid_state_count": _rate(invalid_states, pass_denominator),
        "reconciled_labeled_unknown_count": _rate(reconciled_labeled_unknown, reconciled_denominator),
    }
    gates = {
        "conditional_labeled_unknown_count": labeled_unknown == 0,
        "reconciled_labeled_unknown_count": reconciled_labeled_unknown == 0,
        "conditional_invalid_state_count": invalid_states == 0,
    }
    return {
        "schema_version": "sprite_lab_conditional_identity_health_v1",
        "pass_field_denominator": pass_denominator,
        "reconciled_field_denominator": reconciled_denominator,
        "conditional_labeled_unknown_count": labeled_unknown,
        "conditional_labeled_unknown_rate": _rate(labeled_unknown, pass_denominator),
        "reconciled_labeled_unknown_count": reconciled_labeled_unknown,
        "conditional_abstention_count": abstentions,
        "conditional_abstention_rate": _rate(abstentions, pass_denominator),
        "conditional_concrete_label_count": concrete_labels,
        "conditional_invalid_state_count": invalid_states,
        "conditional_unknown_normalization_count": normalized_unknowns,
        "metric_details": metric_details,
        "gates": gates,
        "passed": all(gates.values()),
    }


def field_health_report(
    pairs: Sequence[Mapping[str, Any]],
    *,
    expected_record_ids: Sequence[str] | None = None,
    filename_leakage_count: int = 0,
    hash_mismatch_count: int = 0,
) -> dict[str, Any]:
    expected = (
        list(expected_record_ids)
        if expected_record_ids is not None
        else [str(row.get("record_id") or "") for row in pairs]
    )
    observed = [str(row.get("record_id") or "") for row in pairs]
    counts = Counter(observed)
    duplicate_count = sum(value - 1 for value in counts.values() if value > 1)
    missing_count = len(set(expected) - set(observed))
    denominator = len(expected)
    field_counts: dict[str, Counter[str]] = {field: Counter() for field in CALIBRATION_FIELDS}
    invalid_output = 0
    taxonomy_invalidity = 0
    conditional_abstentions = 0
    normalized_provider_records: list[dict[str, Any]] = []
    reconciled_records: list[dict[str, Any]] = []
    for pair in pairs:
        raw_left = pair.get("pass_a") if isinstance(pair.get("pass_a"), Mapping) else {}
        raw_right = pair.get("pass_b") if isinstance(pair.get("pass_b"), Mapping) else {}
        left = normalize_provider_output(raw_left)
        right = normalize_provider_output(raw_right)
        normalized_provider_records.extend((left, right))
        reconciled_records.append(reconcile_proposals(left, right))
        for field_name in CALIBRATION_FIELDS:
            comparison = compare_field(
                field_name, _proposal_field(left, field_name), _proposal_field(right, field_name)
            )
            field_counts[field_name][comparison["classification"]] += 1
            if comparison["classification"] == "invalid_comparison":
                invalid_output += 1
                if not comparison.get("taxonomy_valid", True):
                    taxonomy_invalidity += 1
            if field_name in CONDITIONAL_FIELDS and comparison["classification"] in {
                "one_side_abstention",
                "both_abstained",
                "true_contradiction",
                "taxonomy_granularity_difference",
            }:
                conditional_abstentions += 1
    fields: dict[str, Any] = {}
    disagreement_classes = {
        "compatible_hierarchy",
        "one_side_abstention",
        "taxonomy_granularity_difference",
        "true_contradiction",
    }
    for field_name, classifications in field_counts.items():
        raw_disagreement = sum(classifications[name] for name in disagreement_classes)
        fields[field_name] = {
            "classification_counts": {name: classifications[name] for name in COMPARISON_CLASSIFICATIONS},
            "raw_disagreement": _rate(raw_disagreement, denominator),
            "true_contradiction": _rate(classifications["true_contradiction"], denominator),
            "compatible_hierarchy": _rate(classifications["compatible_hierarchy"], denominator),
            "one_side_abstention": _rate(classifications["one_side_abstention"], denominator),
            "both_abstained": _rate(classifications["both_abstained"], denominator),
        }
    combined_core_contradictions = sum(field_counts[field]["true_contradiction"] for field in CORE_FIELDS)
    combined_core_disagreements = sum(
        sum(field_counts[field][name] for name in disagreement_classes) for field in CORE_FIELDS
    )
    core_denominator = denominator * len(CORE_FIELDS)
    all_field_denominator = denominator * len(CALIBRATION_FIELDS)
    conditional_health = conditional_identity_health(normalized_provider_records, reconciled_records)
    metrics = {
        "domain_disagreement": fields["domain"]["raw_disagreement"],
        "broad_category_disagreement": fields["broad_category"]["raw_disagreement"],
        "canonical_object_disagreement": fields["canonical_object"]["raw_disagreement"],
        "role_disagreement": fields["role"]["raw_disagreement"],
        "combined_core_disagreement": _rate(combined_core_disagreements, core_denominator),
        "combined_core_true_contradiction": _rate(combined_core_contradictions, core_denominator),
        "conditional_field_abstention": _rate(conditional_abstentions, denominator * len(CONDITIONAL_FIELDS)),
        "invalid_output": _rate(invalid_output, all_field_denominator),
        "taxonomy_invalidity": _rate(taxonomy_invalidity, all_field_denominator),
        "filename_leakage": _rate(filename_leakage_count, denominator),
        "hash_mismatch": _rate(hash_mismatch_count, denominator),
        "missing_record": _rate(missing_count, denominator),
        "duplicate_record": _rate(duplicate_count, denominator),
        "conditional_labeled_unknown_count": conditional_health["conditional_labeled_unknown_count"],
        "conditional_labeled_unknown_rate": conditional_health["conditional_labeled_unknown_rate"],
        "reconciled_labeled_unknown_count": conditional_health["reconciled_labeled_unknown_count"],
        "conditional_abstention_count": conditional_health["conditional_abstention_count"],
        "conditional_abstention_rate": conditional_health["conditional_abstention_rate"],
        "conditional_concrete_label_count": conditional_health["conditional_concrete_label_count"],
        "conditional_invalid_state_count": conditional_health["conditional_invalid_state_count"],
    }
    gates = {
        "domain_true_contradiction_rate": (fields["domain"]["true_contradiction"]["rate"] or 0.0) <= 0.05,
        "broad_category_true_contradiction_rate": (fields["broad_category"]["true_contradiction"]["rate"] or 0.0)
        <= 0.05,
        "combined_core_true_contradiction_rate": (metrics["combined_core_true_contradiction"]["rate"] or 0.0) <= 0.05,
        "taxonomy_invalidity": taxonomy_invalidity == 0,
        "filename_leakage": filename_leakage_count == 0,
        "hash_mismatch": hash_mismatch_count == 0,
        "missing_record_rate": missing_count == 0,
        "duplicate_record_rate": duplicate_count == 0,
        **conditional_health["gates"],
    }
    return {
        "schema_version": HEALTH_GATE_VERSION,
        "record_denominator": denominator,
        "fields": fields,
        "metrics": metrics,
        "gates": gates,
        "conditional_identity": conditional_health,
        "passed": all(gates.values()),
    }


def filename_leakage_findings(value: Any) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def walk(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold() in _FORBIDDEN_METADATA_KEYS:
                    findings.append({"location": f"{location}.{key}", "reason": "forbidden_metadata_key"})
                walk(child, f"{location}.{key}")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                walk(child, f"{location}[{index}]")

    walk(value, "$")
    return findings


def provider_neutral_request(
    *,
    record_id: str,
    image_sha256: str,
    image_reference: str,
    adapter: ProviderAdapter | None = None,
) -> Mapping[str, Any]:
    request = {
        "schema_version": "sprite_lab_provider_neutral_label_request_v2",
        "record_id": record_id,
        "image_sha256": image_sha256,
        "image_reference": image_reference,
        "prompt_version": PROMPT_VERSION,
        "prompt": conservative_prompt_v3(),
        "contracts": {
            "field_states": field_state_contract(),
            "hierarchy": taxonomy_hierarchy(),
            "conditional_identity": conditional_identity_contract(),
            "unknown_sentinel_policy": unknown_sentinel_policy(),
            "provider_normalization": provider_normalization_contract(),
            "response_schema": conservative_output_schema(),
        },
    }
    findings = filename_leakage_findings(request)
    if findings:
        raise ValueError(f"blind request contains forbidden metadata: {findings}")
    return adapter.format_request(request) if adapter is not None else request


def adapt_historical_label(
    record: Mapping[str, Any],
    *,
    source_artifact_identity: str | None = None,
) -> dict[str, Any]:
    """Read v1 labels conservatively without rewriting or strengthening them."""

    fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
    adapted: dict[str, Any] = {}
    warnings: list[str] = []
    source_schema_version = str(record.get("schema_version") or "unknown")
    artifact_identity = str(
        source_artifact_identity
        or record.get("source_artifact_identity")
        or record.get("artifact_identity")
        or record.get("record_id")
        or "unknown_historical_artifact"
    )
    for field_name in (*CORE_FIELDS, *CONDITIONAL_FIELDS, "visual_form", "visual_material_cue", "description"):
        source_name = "category" if field_name == "broad_category" else field_name
        source = fields.get(source_name)
        if not isinstance(source, Mapping):
            adapted[field_name] = {
                "state": "invalid_output",
                "value": None,
                "reason": "historical_field_state_missing",
            }
            warnings.append(f"{field_name}:historical_field_state_missing")
            continue
        legacy_state = str(source.get("state") or "")
        value = _json_copy(source.get("value"))
        raw_value = _json_copy(value)
        field_diagnostic: dict[str, Any] | None = None
        if legacy_state not in _LEGACY_STATE_MAP:
            adapted[field_name] = {
                "state": "invalid_output",
                "value": None,
                "reason": "historical_field_state_unknown",
            }
            warnings.append(f"{field_name}:historical_field_state_unknown")
            continue
        state = _LEGACY_STATE_MAP[legacy_state]
        reason: str | None = None
        if state == "labeled" and is_controlled_conditional_unknown(field_name, value):
            state = "model_abstained"
            value = None
            reason = "legacy_unknown_normalized"
            warnings.append(f"{field_name}:historical_unknown_conservatively_abstained")
            field_diagnostic = {
                "schema_version": HISTORICAL_READ_ADAPTER_VERSION,
                "normalization_marker": "legacy_unknown_normalized",
                "original_schema_version": source_schema_version,
                "original_raw_state": legacy_state,
                "original_raw_value": raw_value,
                "source_artifact_identity": artifact_identity,
                "source_artifact_rewritten": False,
                "human_reviewed": False,
            }
        elif state == "labeled" and value == _CONDITIONAL_UNKNOWN_SENTINEL:
            # Preserve the pre-existing conservative read behavior for
            # non-conditional taxonomy unknowns without changing core logic.
            state = "model_abstained"
            value = None
            reason = "taxonomy_has_no_safe_match"
            warnings.append(f"{field_name}:historical_unknown_conservatively_abstained")
        elif state == "labeled" and (value is None or value == "" or value == [] or value == {}):
            state = "invalid_output"
            value = None
            reason = "historical_empty_successful_label"
            warnings.append(f"{field_name}:historical_empty_successful_label")
            field_diagnostic = {
                "schema_version": HISTORICAL_READ_ADAPTER_VERSION,
                "normalization_marker": "legacy_invalid_output_preserved",
                "original_schema_version": source_schema_version,
                "original_raw_state": legacy_state,
                "original_raw_value": raw_value,
                "source_artifact_identity": artifact_identity,
                "source_artifact_rewritten": False,
                "human_reviewed": False,
            }
        elif state == "model_abstained":
            value = None
            reason = "visually_ambiguous"
            if raw_value is not None:
                field_diagnostic = {
                    "schema_version": HISTORICAL_READ_ADAPTER_VERSION,
                    "normalization_marker": "legacy_abstention_value_discarded",
                    "original_schema_version": source_schema_version,
                    "original_raw_state": legacy_state,
                    "original_raw_value": raw_value,
                    "source_artifact_identity": artifact_identity,
                    "source_artifact_rewritten": False,
                    "human_reviewed": False,
                }
        elif state == "not_applicable":
            value = None
        elif state == "invalid_output":
            value = None
            reason = "historical_unsupported_not_strengthened"
        if field_name == "visual_material_cue" and isinstance(value, str) and value in _MATERIAL_CUE_ALIASES:
            warnings.append(f"{field_name}:explicit_legacy_visual_cue_alias")
            value = _MATERIAL_CUE_ALIASES[value]
        adapted_field = {
            "state": state,
            "value": value,
            "reason": reason,
            "historical_confidence": source.get("confidence"),
            "confidence_was_inferred": False,
            "evidence": source.get("visual_evidence"),
            "visually_unmistakable": False,
            "visually_demonstrated": False,
        }
        if field_diagnostic is not None:
            adapted_field["diagnostic_metadata"] = field_diagnostic
        validation = validate_field(field_name, adapted_field)
        if not validation.valid:
            adapted_field["diagnostic_metadata"] = field_diagnostic or {
                "schema_version": HISTORICAL_READ_ADAPTER_VERSION,
                "normalization_marker": "legacy_invalid_output_preserved",
                "original_schema_version": source_schema_version,
                "original_raw_state": legacy_state,
                "original_raw_value": raw_value,
                "source_artifact_identity": artifact_identity,
                "source_artifact_rewritten": False,
                "human_reviewed": False,
            }
            adapted_field.update(
                {
                    "state": "invalid_output",
                    "value": None,
                    "reason": validation.reason or "historical_invalid_output",
                }
            )
            warnings.append(f"{field_name}:{adapted_field['reason']}")
        adapted[field_name] = adapted_field
    colors = {
        name: _json_copy((fields.get(name) or {}).get("value"))
        for name in ("primary_colors", "secondary_colors", "outline_colors")
        if isinstance(fields.get(name), Mapping)
    }
    adapted["colors"] = {
        "state": "labeled" if colors else "invalid_output",
        "value": colors or None,
        "reason": None if colors else "historical_field_state_missing",
    }
    explicit_material = fields.get("explicit_material")
    if isinstance(explicit_material, Mapping) and explicit_material.get("value") is not None:
        warnings.append("explicit_material:not_carried_into_visual_cue")
    return {
        "schema_version": CONTRACT_VERSION,
        "record_id": str(record.get("record_id") or ""),
        "image_sha256": str(record.get("image_sha256") or ""),
        "fields": adapted,
        "migration": {
            "source_schema_version": source_schema_version,
            "source_artifact_identity": artifact_identity,
            "adapter": HISTORICAL_READ_ADAPTER_VERSION,
            "warnings": warnings,
            "confidence_or_supervision_strength_added": False,
            "source_record_rewritten": False,
            "human_reviewed": False,
        },
    }


def build_review_queue(
    reconciled_records: Sequence[Mapping[str, Any]],
    *,
    explicit_calibration_record_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    calibration_ids = set(explicit_calibration_record_ids)
    queue: list[dict[str, Any]] = []
    for record in reconciled_records:
        if record.get("excluded_unsuitable"):
            continue
        record_id = str(record.get("record_id") or "")
        comparisons = record.get("comparisons") if isinstance(record.get("comparisons"), Mapping) else {}
        fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
        reasons: list[str] = []
        for field_name in CORE_FIELDS:
            comparison = comparisons.get(field_name) if isinstance(comparisons.get(field_name), Mapping) else {}
            field_result = fields.get(field_name) if isinstance(fields.get(field_name), Mapping) else {}
            if comparison.get("classification") == "true_contradiction":
                reasons.append(f"{field_name}:true_core_contradiction")
            elif comparison.get("classification") == "taxonomy_granularity_difference" and not comparison.get(
                "safe_parent"
            ):
                reasons.append(f"{field_name}:unresolved_taxonomy_mapping")
            elif field_result.get("state") == "invalid_output" and record.get("semantic_label_required", False):
                reasons.append(f"{field_name}:required_label_invalid_output")
        if record.get("uncertain_broad_category_for_conditioned_view"):
            reasons.append("broad_category:uncertain_broad_category_for_conditioned_view")
        if record_id in calibration_ids:
            reasons.append("record:explicit_calibration_sample")
        if not reasons:
            continue
        queue.append(
            {
                "schema_version": REVIEW_QUEUE_VERSION,
                "review_item_id": hashlib.sha256(f"{REVIEW_QUEUE_VERSION}\0{record_id}".encode()).hexdigest(),
                "image_identity": {
                    "record_id": record_id,
                    "image_sha256": str(record.get("image_sha256") or ""),
                },
                "current_conservative_result": _json_copy(fields),
                "pass_a_proposal": _json_copy(record.get("pass_a_proposal")),
                "health_check_or_pass_b_proposal": _json_copy(record.get("pass_b_proposal")),
                "field_level_disagreement": _json_copy(comparisons),
                "recommended_safe_action": "choose_safer_parent"
                if any(isinstance(value, Mapping) and value.get("safe_parent") for value in comparisons.values())
                else "abstain",
                "allowed_actions": list(REVIEW_ACTIONS),
                "review_reason": reasons,
            }
        )
    return queue


def build_conservative_salvage_summary(reconciled_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a derived, non-supervisory salvage view from reconciled records."""

    core_counts = dict.fromkeys(CORE_FIELDS, 0)
    conditional_counts = dict.fromkeys(CONDITIONAL_FIELDS, 0)
    conditional_abstentions = dict.fromkeys(CONDITIONAL_FIELDS, 0)
    conditional_candidates: list[dict[str, Any]] = []
    for record in reconciled_records:
        fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
        comparisons = record.get("comparisons") if isinstance(record.get("comparisons"), Mapping) else {}
        for field_name in CORE_FIELDS:
            result = fields.get(field_name) if isinstance(fields.get(field_name), Mapping) else {}
            comparison = comparisons.get(field_name) if isinstance(comparisons.get(field_name), Mapping) else {}
            if result.get("state") == "labeled" and comparison.get("classification") in {
                "exact_agreement",
                "compatible_hierarchy",
                "taxonomy_granularity_difference",
            }:
                core_counts[field_name] += 1
        for field_name in CONDITIONAL_FIELDS:
            result = fields.get(field_name) if isinstance(fields.get(field_name), Mapping) else {}
            validation = validate_field(field_name, result)
            if validation.valid and validation.state == "labeled":
                conditional_counts[field_name] += 1
                conditional_candidates.append(
                    {
                        "record_id": str(record.get("record_id") or ""),
                        "field": field_name,
                        "value": _json_copy(validation.value),
                        "candidate_only": True,
                        "human_truth": False,
                        "strong_supervision": False,
                    }
                )
            elif validation.valid and validation.state == "model_abstained":
                conditional_abstentions[field_name] += 1
    return {
        "schema_version": SALVAGE_SUMMARY_VERSION,
        "record_count": len(reconciled_records),
        "core_agreement_candidate_counts": {
            **core_counts,
            "total": sum(core_counts.values()),
        },
        "conditional_concrete_identity_counts": {
            **conditional_counts,
            "total": sum(conditional_counts.values()),
        },
        "conditional_abstention_counts": {
            **conditional_abstentions,
            "total": sum(conditional_abstentions.values()),
        },
        "conditional_identity_candidates": conditional_candidates,
        "unknown_identity_candidate_count": sum(
            is_controlled_conditional_unknown(row["field"], row["value"]) for row in conditional_candidates
        ),
        "review_required_records": sum(bool(record.get("mandatory_review")) for record in reconciled_records),
        "image_only_eligibility_count": sum(bool(record.get("image_only_eligible")) for record in reconciled_records),
        "strong_supervision_count": 0,
        "calibration_truth_count": 0,
        "human_truth_count": 0,
        "original_salvage_evidence_modified": False,
    }


def append_review_event(
    path: str | Path,
    review_item: Mapping[str, Any],
    *,
    action: str,
    field_name: str,
    selected_value: Any = None,
    reviewer_id: str,
) -> dict[str, Any]:
    allowed = set(review_item.get("allowed_actions") or ())
    if action not in REVIEW_ACTIONS or action not in allowed:
        raise ValueError(f"review action is not allowed: {action}")
    if field_name not in CALIBRATION_FIELDS:
        raise ValueError(f"review event requires one calibration field: {field_name}")
    if action in {"accept_broad_label", "choose_safer_parent"} and selected_value is None:
        raise ValueError(f"{action} requires selected_value")
    identity = review_item.get("image_identity") if isinstance(review_item.get("image_identity"), Mapping) else {}
    event = {
        "schema_version": REVIEW_EVENT_VERSION,
        "event_id": uuid.uuid4().hex,
        "timestamp": _utc_now(),
        "review_item_id": str(review_item.get("review_item_id") or ""),
        "record_id": str(identity.get("record_id") or ""),
        "image_sha256": str(identity.get("image_sha256") or ""),
        "field": field_name,
        "action": action,
        "selected_value": _json_copy(selected_value),
        "reviewer_id": str(reviewer_id),
        "changes_provenance_license_or_image_identity": False,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(event) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return event


def build_calibration_inputs(
    reconciled_records: Sequence[Mapping[str, Any]],
    review_events: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for event in review_events:
        if event.get("schema_version") == REVIEW_EVENT_VERSION:
            latest[(str(event.get("record_id") or ""), str(event.get("field") or ""))] = event
    rows: list[dict[str, Any]] = []
    for record in reconciled_records:
        record_id = str(record.get("record_id") or "")
        fields = record.get("fields") if isinstance(record.get("fields"), Mapping) else {}
        comparisons = record.get("comparisons") if isinstance(record.get("comparisons"), Mapping) else {}
        for field_name in CALIBRATION_FIELDS:
            result = fields.get(field_name) if isinstance(fields.get(field_name), Mapping) else {}
            raw_comparison = comparisons.get(field_name)
            comparison: dict[str, Any] = dict(raw_comparison) if isinstance(raw_comparison, Mapping) else {}
            if (
                field_name in CONDITIONAL_FIELDS
                and result.get("state") == "labeled"
                and is_controlled_conditional_unknown(field_name, result.get("value"))
            ):
                result = normalize_provider_field(field_name, result)
                comparison = {**comparison, "classification": "both_abstained"}
            event = latest.get((record_id, field_name))
            truth_value: Any = None
            if event and event.get("action") in {"accept_broad_label", "choose_safer_parent"}:
                calibration_state = "human_verified"
                truth_value = _json_copy(event.get("selected_value"))
            elif event and event.get("action") == "abstain":
                calibration_state = "human_abstained"
            elif comparison.get("classification") in {"exact_agreement", "compatible_hierarchy"}:
                calibration_state = "model_agreement_candidate"
            elif comparison.get("classification") == "true_contradiction":
                calibration_state = "model_conflict"
            else:
                calibration_state = "unreviewed"
            rows.append(
                {
                    "schema_version": CALIBRATION_INPUT_VERSION,
                    "record_id": record_id,
                    "field": field_name,
                    "calibration_state": calibration_state,
                    "prediction_state": result.get("state"),
                    "predicted_value": _json_copy(result.get("value")),
                    "confidence": result.get("confidence"),
                    "truth_value": truth_value,
                    "truth_source": "explicit_human_review" if calibration_state.startswith("human_") else None,
                    "model_agreement_treated_as_truth": False,
                    "accepted_label_contribution": 1 if result.get("state") == "labeled" else 0,
                    "abstention_contribution": 1 if result.get("state") == "model_abstained" else 0,
                    "correctness_eligible": bool(
                        calibration_state.startswith("human_") and result.get("state") == "labeled"
                    ),
                    "human_truth_created_by_normalization": False,
                }
            )
    return rows


def calibration_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field_name in CALIBRATION_FIELDS:
        field_rows = [row for row in rows if row.get("field") == field_name]
        truth_rows = [
            row for row in field_rows if row.get("calibration_state") in {"human_verified", "human_abstained"}
        ]
        accepted_truth = [row for row in truth_rows if row.get("prediction_state") == "labeled"]
        correct = sum(
            row.get("calibration_state") == "human_verified" and row.get("predicted_value") == row.get("truth_value")
            for row in accepted_truth
        )
        abstained = sum(row.get("prediction_state") != "labeled" for row in field_rows)
        bins: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in truth_rows:
            confidence = row.get("confidence")
            if confidence is not None:
                bins[str(confidence)].append(row)
        confidence_bins = []
        for name, bin_rows in sorted(bins.items()):
            accepted = [row for row in bin_rows if row.get("prediction_state") == "labeled"]
            bin_correct = sum(
                row.get("calibration_state") == "human_verified"
                and row.get("predicted_value") == row.get("truth_value")
                for row in accepted
            )
            confidence_bins.append(
                {
                    "bin": name,
                    "truth_count": len(bin_rows),
                    "accepted_count": len(accepted),
                    "accuracy": (bin_correct / len(accepted)) if accepted else None,
                }
            )
        fields[field_name] = {
            "accuracy": _rate(correct, len(accepted_truth)),
            "coverage": _rate(len(accepted_truth), len(truth_rows)),
            "abstention_rate": _rate(abstained, len(field_rows)),
            "risk_at_accepted_coverage": _rate(len(accepted_truth) - correct, len(accepted_truth)),
            "confidence_bins": confidence_bins,
            "truth_count": len(truth_rows),
            "model_only_rows_excluded_from_truth": len(field_rows) - len(truth_rows),
        }
    return {"schema_version": "sprite_lab_calibration_metrics_v1", "fields": fields}


def calibration_readiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_truth_per_field: int = 20,
    min_nonempty_confidence_bins: int = 2,
) -> dict[str, Any]:
    truth_counts = {
        field: sum(
            row.get("field") == field and row.get("calibration_state") in {"human_verified", "human_abstained"}
            for row in rows
        )
        for field in CALIBRATION_FIELDS
    }
    model_agreement_rows = sum(row.get("calibration_state") == "model_agreement_candidate" for row in rows)
    if sum(truth_counts.values()) == 0:
        status = "not_ready"
        reasons = ["no_explicit_human_truth", "model_model_agreement_is_not_calibration_truth"]
    else:
        sparse = [field for field, count in truth_counts.items() if count < min_truth_per_field]
        bin_counts = {
            field: len(
                {
                    str(row.get("confidence"))
                    for row in rows
                    if row.get("field") == field
                    and row.get("calibration_state") in {"human_verified", "human_abstained"}
                    and row.get("confidence") is not None
                }
            )
            for field in CALIBRATION_FIELDS
        }
        sparse_bins = [field for field, count in bin_counts.items() if count < min_nonempty_confidence_bins]
        if sparse or sparse_bins:
            status = "insufficient_truth"
            reasons = [
                *(f"{field}:truth_below_{min_truth_per_field}" for field in sparse),
                *(f"{field}:confidence_bins_below_{min_nonempty_confidence_bins}" for field in sparse_bins),
            ]
        else:
            status = "ready"
            reasons = []
    return {
        "schema_version": "sprite_lab_calibration_readiness_v1",
        "status": status,
        "reasons": reasons,
        "truth_counts": truth_counts,
        "model_agreement_candidate_rows": model_agreement_rows,
        "model_agreement_candidate_rows_counted_as_truth": 0,
        "fit_calibration_model": status == "ready",
        "metrics": calibration_metrics(rows),
    }


def contract_documents() -> dict[str, Any]:
    return {
        "hierarchical_label_contract.json": hierarchical_label_contract(),
        "conditional_identity_contract.json": conditional_identity_contract(),
        "unknown_sentinel_policy.json": unknown_sentinel_policy(),
        "provider_normalization_contract.json": provider_normalization_contract(),
        "field_state_contract.json": field_state_contract(),
        "disagreement_classification_contract.json": disagreement_classification_contract(),
        "field_health_gate_contract.json": field_health_gate_contract(),
        "health_metric_contract.json": health_metric_contract(),
        "reconciliation_contract.json": reconciliation_contract(),
        "review_queue_contract.json": review_queue_contract(),
        "calibration_input_contract.json": calibration_input_contract(),
        "historical_compatibility_contract.json": historical_compatibility_contract(),
    }


def stage_independent_audit(
    raw_experiment: str | Path,
    output: str | Path,
    *,
    model_display: str,
    session_id: str,
) -> dict[str, Any]:
    """Stage a fresh v2 campaign without labeling or calling a provider."""

    from spritelab.dataset_v5.codex_blind import stage_campaign

    progress = stage_campaign(
        raw_experiment,
        output,
        model_display=model_display,
        session_id=session_id,
    )
    target = Path(output).resolve()
    (target / "blind_prompt.md").write_text(conservative_prompt_v3(), encoding="utf-8", newline="\n")
    (target / "output_schema.json").write_text(
        json.dumps(conservative_output_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, document in contract_documents().items():
        (target / name).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    policy_path = target / "forbidden_metadata_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["prompt_version"] = PROMPT_VERSION
    policy["schema_version"] = "sprite_lab_codex_forbidden_metadata_policy_v2"
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    progress["campaign_version"] = "sprite_lab_codex_blind_campaign_v2"
    progress["labeler"]["prompt_version"] = PROMPT_VERSION
    progress["new_independent_health_gate_status"] = "not_established"
    progress["historical_campaign_resumed"] = False
    progress["provider_calls"] = 0
    progress["new_production_labels"] = 0
    progress["pass_b_completed"] = 0
    (target / "progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    audit_plan = {
        "schema_version": "sprite_lab_independent_blind_audit_plan_v2",
        "prompt_version": PROMPT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "health_gate_version": HEALTH_GATE_VERSION,
        "status": "staged_unlabeled",
        "provider_calls": 0,
        "production_labels_created": 0,
        "historical_pass_b_continued": False,
        "instructions": [
            "use_a_fresh_independent_session",
            "label_pass_a_blindly_with_prompt_v2",
            "run_a_new_independent_health_sample",
            "evaluate_field_specific_health_before_any_pass_b",
        ],
    }
    (target / "independent_audit_plan.json").write_text(
        json.dumps(audit_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"audit_plan": audit_plan, "output": str(target), "progress": progress}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m spritelab.dataset_v5.conservative_labeling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser(
        "stage-independent-audit",
        help="Stage a fresh provider-neutral v2 audit without producing labels.",
    )
    stage.add_argument("--raw-experiment", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--model-display", required=True)
    stage.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    result = stage_independent_audit(
        args.raw_experiment,
        args.output,
        model_display=args.model_display,
        session_id=args.session_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
