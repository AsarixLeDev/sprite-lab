from __future__ import annotations

import json

import pytest

from spritelab.harvest.label_v4.description import choose_or_regenerate_description, validate_description
from spritelab.harvest.label_v4.filename_parser import parse_filename_semantics
from spritelab.harvest.label_v4.proposal import (
    BlindVLMProposal,
    ProposalValidationError,
    build_blind_vlm_prompt,
    parse_blind_vlm_response,
)
from spritelab.harvest.label_v4.reconciliation import (
    ReconciliationResult,
    ReconciliationValidationError,
    build_reconciliation_prompt,
    normalize_reconciled_value,
    parse_reconciliation_response,
    reconcile_evidence,
)
from spritelab.harvest.label_v4.risk import estimate_field_risk
from spritelab.harvest.label_v4.routing import (
    INDEPENDENT_VERIFIER_CACHE_NAMESPACE,
    AdaptiveRoutingSignals,
    decide_adaptive_routing,
)
from spritelab.harvest.label_v4.semantic_axes import (
    CATEGORY_VALUES,
    DOMAIN_VALUES,
    ROLE_VALUES,
    AxisValidationError,
    ColorAttributes,
    MaterialEvidence,
    ShapeAttributes,
    normalize_visual_color_roles,
    validate_axis_value,
)


def _rich_proposal(**updates):
    payload = {
        "object_candidates": [{"value": "buckler", "visual_support": ["round shield", "central boss", "metal rim"]}],
        "category_candidates": ["armor", "shield"],
        "surface_alias_candidates": ["round metal shield"],
        "role_candidates": ["defensive_equipment"],
        "shape": {
            "silhouette": ["round"],
            "aspect": ["compact"],
            "orientation": ["front_facing"],
            "structure": ["rimmed", "bossed"],
            "edge_profile": ["rounded"],
            "parts": ["rim", "central_boss"],
        },
        "color_roles": {
            "primary": ["gray"],
            "secondary": [],
            "outline": ["black"],
            "shadow": ["dark_gray"],
            "highlight": ["light_gray"],
        },
        "material_visual_cues": ["metallic"],
        "description_candidates": ["A round metal buckler with a dark rim and central boss."],
        "uncertainties": [],
        "alternative_interpretations": [],
        "unsupported_fields": [],
    }
    payload.update(updates)
    return payload


def test_semantic_axes_are_strict_and_pixel_art_is_never_domain():
    assert DOMAIN_VALUES == (
        "inventory_icon",
        "equipment_icon",
        "resource_icon",
        "food_icon",
        "plant_icon",
        "spell_icon",
        "unknown",
    )
    assert "gem" in CATEGORY_VALUES and "weapon" in CATEGORY_VALUES
    assert "defensive_equipment" in ROLE_VALUES
    assert "pixel_art" not in DOMAIN_VALUES
    assert validate_axis_value("style", "pixel-art") == "pixel_art"
    with pytest.raises(AxisValidationError):
        validate_axis_value("domain", "pixel_art")
    with pytest.raises(AxisValidationError):
        validate_axis_value("domain", "weapon")


def test_material_shape_and_color_contracts_preserve_separate_evidence():
    material = MaterialEvidence(
        explicit_material="iron", visual_material_cue=("metallic",), explicit_support=("filename",)
    )
    assert material.explicit_material == "iron"
    assert material.visual_material_cue == ("metallic",)

    shape = ShapeAttributes(
        silhouette=("round",),
        aspect=("wide",),
        orientation=("front_facing",),
        structure=("ring_shaped", "bossed"),
        edge_profile=("rounded",),
        parts=("rim", "central boss"),
    )
    assert shape.to_dict()["parts"] == ["rim", "central_boss"]

    colors = ColorAttributes(palette_colors=("gray", "black"), primary_colors=("gray",), highlight_colors=("white",))
    assert colors.role_membership_conflicts() == ("highlight_colors:white:not_in_palette",)


@pytest.mark.parametrize(
    ("name", "canonical", "category", "role", "material"),
    [
        ("iron_buckler.png", "buckler", "armor", "defensive_equipment", "iron"),
        ("platemail_helmet.png", "helmet", "armor", "wearable_equipment", "plate_metal"),
        ("iron_ring.png", "ring", "jewelry", "wearable_equipment", "iron"),
        ("copper_ring.png", "ring", "jewelry", "wearable_equipment", "copper"),
        ("cloth_pants.png", "pants", "clothing", "wearable_equipment", "cloth"),
        ("quilted_armor.png", "armor", "armor", "wearable_equipment", None),
        ("tattered_shirt.png", "shirt", "clothing", "wearable_equipment", None),
        ("leather_cap.png", "cap", "clothing", "wearable_equipment", "leather"),
        ("chainmail_jacket.png", "jacket", "armor", "wearable_equipment", "chainmail"),
    ],
)
def test_compositional_filename_named_examples(name, canonical, category, role, material):
    parsed = parse_filename_semantics(name)
    assert parsed.canonical_object == canonical
    assert parsed.category == category
    assert parsed.role == role
    assert parsed.explicit_material == material
    assert parsed.surface_alias == name.removesuffix(".png").replace("_", " ")
    assert parsed.generic is False
    assert parsed.token_provenance


def test_filename_parser_recovers_color_size_condition_variants_sequences_and_context():
    small = parse_filename_semantics("small_purple.png", pack_context={"category": "gem"})
    assert small.size_hint == "small"
    assert small.filename_color_hints == ("purple",)
    assert small.canonical_object is None
    assert small.object_source == ""
    assert small.surface_alias is None
    assert small.category == "gem"
    assert small.role == "unknown"  # category context alone does not synthesize function

    rich = parse_filename_semantics("ancient_moon_relic_alt_v2_003.png")
    assert rich.condition_hints == ("ancient",)
    assert rich.variant_suffixes == ("alt", "v2")
    assert rich.sequence_numbers == ("003",)
    assert {"moon", "relic"} <= set(rich.open_set_tokens)
    assert rich.generic is False
    # Every lexeme survives classification even though the object is open-set.
    assert {event.classification for event in rich.token_provenance} >= {
        "condition",
        "open_set",
        "variant",
        "sequence",
    }
    assert any("preserve_open_set_token" in event.transformation for event in rich.token_provenance)


def test_filename_parser_composes_member_sheet_pack_mapping_and_source_metadata():
    parsed = parse_filename_semantics(
        "sprite_02.png",
        member_path="icons/equipment/front/steel_helm.png",
        sheet_name="armor sheet",
        pack_name="fantasy equipment",
        declarative_mapping={"canonical_object": "helmet", "category": "armor"},
        source_metadata={"declared_material": "steel"},
    )
    assert parsed.canonical_object == "helmet"
    assert parsed.category == "armor"
    assert parsed.explicit_material == "steel"
    assert parsed.generic is False  # the explicit member basename is top-tier identity evidence
    sources = {event.source for event in parsed.token_provenance}
    assert {"filename", "member_path", "sheet_name", "pack_name"} <= sources
    assert "source_metadata.declared_material" in sources


def test_rich_blind_proposal_roundtrip_has_alternatives_shape_roles_and_audit_metadata():
    payload = _rich_proposal(
        uncertainties=["identity is ambiguous at this scale"],
        alternative_interpretations=["ornamental round plate"],
    )
    artifact = parse_blind_vlm_response(
        json.dumps(payload),
        model_identity="mock-vlm:1",
        request_hash="request-sha256",
        image_hash="rgba-sha256",
        latency_ms=12.5,
        token_usage={"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
    )
    assert artifact.available
    assert artifact.model_identity == "mock-vlm:1"
    assert artifact.request_hash == "request-sha256"
    assert artifact.image_hash == "rgba-sha256"
    assert artifact.latency_ms == 12.5
    assert artifact.token_usage.total_tokens == 160
    assert artifact.proposal is not None
    assert artifact.proposal.object_candidates[0].visual_support[-1] == "metal rim"
    assert artifact.proposal.shape.structure == ("rimmed", "bossed")
    assert artifact.proposal.color_roles.outline_colors == ("black",)
    assert artifact.proposal.alternative_interpretations == ("ornamental round plate",)
    assert "confidence" not in json.dumps(artifact.to_dict())


def test_blind_proposal_rejects_self_confidence_and_ambiguous_without_alternative():
    with pytest.raises(ProposalValidationError):
        BlindVLMProposal.from_dict(_rich_proposal(confidence=0.7))

    invalid = _rich_proposal(uncertainties=["ambiguous identity"], alternative_interpretations=[])
    failed = parse_blind_vlm_response(
        invalid,
        model_identity="mock",
        request_hash="request",
        image_hash="image",
    )
    assert not failed.available
    assert failed.failure is not None
    assert failed.failure.failure_type == "schema_validation_failure"
    assert failed.parsed_output is None


def test_qwen_candidate_mapping_and_scalar_visual_axes_are_boundedly_normalized():
    payload = {
        "object_candidates": {
            "value": ["cylinder", "rod", "stick", "pencil", "pen", "matchstick"],
            "visual_support": [
                "elongated, rounded form with consistent width",
                "dark outline suggesting a defined edge",
            ],
        },
        "category_candidates": ["tool", "writing instrument"],
        "surface_alias_candidates": ["bar", "shaft"],
        "role_candidates": ["handle", "component"],
        "shape": {
            "silhouette": "elongated oval or rounded rectangle",
            "aspect": "long and narrow (high length-to-width ratio)",
            "orientation": "diagonal, from top-left to bottom-right",
            "structure": "uniform cross-section along length",
            "edge_profile": "rounded edges, smooth contour",
            "parts": [],
        },
        "color_roles": {
            "primary": "brownish-orange",
            "secondary": "darker brown",
            "outline": "black",
            "shadow": "darker brown along edges",
            "highlight": "lighter orange in center",
        },
        "material_visual_cues": ["no visible metallic sheen"],
        "description_candidates": ["A diagonally oriented rod."],
        "uncertainties": ["Exact material cannot be determined."],
        "alternative_interpretations": ["Could be a matchstick."],
        "unsupported_fields": [],
    }
    artifact = parse_blind_vlm_response(
        json.dumps(payload),
        model_identity="qwen-mock",
        request_hash="request",
        image_hash="image",
    )
    assert artifact.available
    assert artifact.proposal is not None
    assert [candidate.value for candidate in artifact.proposal.object_candidates] == [
        "cylinder",
        "rod",
        "stick",
        "pencil",
        "pen",
        "matchstick",
    ]
    assert all(
        candidate.visual_support
        == (
            "elongated, rounded form with consistent width",
            "dark outline suggesting a defined edge",
        )
        for candidate in artifact.proposal.object_candidates
    )
    assert artifact.proposal.shape.silhouette == ("elongated oval or rounded rectangle",)
    assert artifact.proposal.shape.orientation == ("diagonal, from top-left to bottom-right",)
    assert artifact.proposal.color_roles.primary_colors == ("brownish_orange",)
    assert artifact.proposal.color_roles.shadow_colors == ("darker_brown_along_edges",)
    assert artifact.proposal.raw_visual_color_roles["primary_colors"] == ("brownish-orange",)
    assert artifact.proposal.raw_visual_color_roles["shadow_colors"] == ("darker brown along edges",)
    # Structural normalization must not rewrite the audited provider payload.
    assert artifact.parsed_output is not None
    assert isinstance(artifact.parsed_output["object_candidates"], dict)
    assert artifact.parsed_output["shape"]["silhouette"] == "elongated oval or rounded rectangle"
    assert artifact.parsed_output["color_roles"]["primary"] == "brownish-orange"


def test_qwen_structural_normalizer_still_rejects_forbidden_confidence():
    payload = {
        "object_candidates": {"value": ["rod"], "visual_support": ["elongated"]},
        "shape": {"silhouette": "elongated"},
        "color_roles": {"primary": "brown"},
        "confidence": 0.7,
    }
    artifact = parse_blind_vlm_response(
        payload,
        model_identity="qwen-mock",
        request_hash="request",
        image_hash="image",
    )
    assert not artifact.available
    assert artifact.failure is not None
    assert artifact.failure.failure_type == "schema_validation_failure"
    assert artifact.parsed_output is None


def test_truncated_qwen_json_remains_a_parse_failure_not_a_partial_proposal():
    raw = '{"object_candidates":{"value":["rod"],"visual_support":["elongated"]},"shape":'
    artifact = parse_blind_vlm_response(
        raw,
        model_identity="qwen-mock",
        request_hash="request",
        image_hash="image",
    )
    assert not artifact.available
    assert artifact.proposal is None
    assert artifact.raw_output == raw
    assert artifact.parsed_output is None
    assert artifact.failure is not None
    assert artifact.failure.failure_type == "json_parse_failure"


def test_blind_prompt_is_context_free_and_disallows_material_and_score_shortcuts():
    prompt = build_blind_vlm_prompt().lower()
    assert "iron_buckler" not in prompt
    assert "scheduler" not in prompt
    assert "candidate vocabulary" not in prompt
    assert "never infer an exact material" in prompt
    assert "do not emit confidence" in prompt


def test_reconciliation_preserves_raw_normalized_alternatives_and_conflicts():
    deterministic = parse_filename_semantics("iron_buckler.png")
    proposal = BlindVLMProposal.from_dict(
        _rich_proposal(
            object_candidates=[
                {"value": "shield", "visual_support": ["round defensive object"]},
                {"value": "buckler", "visual_support": ["central boss"]},
            ]
        )
    )
    result = reconcile_evidence(deterministic, proposal)
    obj = result.field_proposals["canonical_object"]
    assert obj.raw_open_vocabulary_value == "buckler"
    assert obj.normalized_controlled_value == "buckler"
    assert obj.alternatives[0] == "shield"
    assert "filename_vlm_object_disagreement" in obj.conflicts
    assert result.field_proposals["explicit_material"].value == "iron"
    assert result.field_proposals["visual_material_cue"].value == ["metallic"]
    assert any(conflict["field"] == "canonical_object" for conflict in result.unresolved_conflicts)


def test_filename_visual_color_conflict_is_surfaced_without_overwrite():
    deterministic = parse_filename_semantics("small_purple.png", pack_context={"category": "gem"})
    proposal = BlindVLMProposal.from_dict(
        _rich_proposal(
            object_candidates=[{"value": "gem", "visual_support": ["faceted object"]}],
            color_roles={
                "primary": ["blue"],
                "secondary": [],
                "outline": ["black"],
                "shadow": ["dark_blue"],
                "highlight": ["light_blue"],
            },
        )
    )
    result = reconcile_evidence(deterministic, proposal)
    assert result.field_proposals["filename_color_hints"].value == ["purple"]
    assert result.field_proposals["color_roles"].value["primary_colors"] == ["blue"]
    assert "filename_visual_color_conflict" in result.field_proposals["color_roles"].conflicts
    assert any(conflict["code"] == "filename_visual_color_conflict" for conflict in result.unresolved_conflicts)


def test_reconciliation_preserves_open_set_and_rejects_axis_confusion():
    deterministic = parse_filename_semantics("asset_001.png")
    proposal = BlindVLMProposal.from_dict(
        _rich_proposal(
            object_candidates=[{"value": "ceremonial sun token", "visual_support": ["radiating disk"]}],
            category_candidates=["shield"],
        )
    )
    result = reconcile_evidence(deterministic, proposal)
    assert result.field_proposals["canonical_object"].raw_open_vocabulary_value == "ceremonial_sun_token"
    assert "ceremonial_sun_token" in result.open_set_terms
    assert result.field_proposals["category"].value == "armor"
    assert any(
        action.get("raw") == "shield" and action.get("normalized") == "armor"
        for action in result.taxonomy_mapping_actions
    )
    normalized, action = normalize_reconciled_value("domain", "pixel_art")
    assert normalized is None and action["action"] == "invalid_axis_value"


def test_text_reconciliation_parser_keeps_raw_and_normalized_values():
    parsed = parse_reconciliation_response(
        {
            "field_proposals": {
                "canonical_object": {
                    "value": "platemail helmet",
                    "alternatives": ["helmet"],
                    "support": ["filename", "vlm_visual"],
                    "conflicts": [],
                }
            },
            "open_set_terms": ["novel_crest"],
        }
    )
    field = parsed.field_proposals["canonical_object"]
    assert field.raw_open_vocabulary_value == "platemail helmet"
    assert field.normalized_controlled_value == "platemail_helmet"
    assert field.value == "platemail_helmet"
    assert parsed.open_set_terms == ("novel_crest",)


def test_filename_precedence_and_sheet_cell_alias_policy_are_explicit():
    filename_wins = parse_filename_semantics(
        "iron_buckler.png",
        declarative_mapping={
            "canonical_object": "helmet",
            "surface_alias": "mapped helmet",
            "category": "armor",
        },
    )
    assert filename_wins.canonical_object == "buckler"
    assert filename_wins.object_source == "sprite_filename"
    assert filename_wins.surface_alias == "iron buckler"
    assert filename_wins.surface_alias_source == "sprite_filename"

    shade = parse_filename_semantics(
        {
            "relative_path": "16x16 Weapons RPG Icons/bronze-weapons.png",
            "archive_member": "16x16 Weapons RPG Icons/bronze-weapons.png",
            "source_sheet": "16x16 Weapons RPG Icons/bronze-weapons.png",
            "pack_name": "16x16 Weapon RPG Icons",
            "declared_material": "bronze",
            "auto_metadata": {
                "sheet_mapping": {
                    "category": "weapon",
                    "role": "weapon",
                    "material": "bronze",
                    "sheet_coordinate": "r000_c019",
                }
            },
        },
        pack_context={"canonical_object": "spear", "category": "weapon", "role": "weapon"},
    )
    assert shade.canonical_object is None
    assert shade.surface_alias is None
    assert shade.category == "weapon"
    assert shade.role == "weapon"
    assert shade.explicit_material == "bronze"
    assert shade.field_sources["category"] == "explicit_cell_mapping"
    assert shade.field_sources["explicit_material"] == "explicit_cell_mapping"


def test_generic_visual_form_is_preserved_while_canonical_object_abstains():
    proposal = BlindVLMProposal.from_dict(
        _rich_proposal(
            object_candidates=[
                {"value": "cylinder", "visual_support": ["elongated rounded form"]},
                {"value": "rod", "visual_support": ["uniform narrow width"]},
                {"value": "stick", "visual_support": ["simple elongated silhouette"]},
                {"value": "pencil", "visual_support": ["one possible interpretation"]},
            ],
            category_candidates=["tool"],
        )
    )
    result = reconcile_evidence(parse_filename_semantics("asset_001.png"), proposal)
    canonical = result.field_proposals["canonical_object"]
    assert canonical.value is None
    assert canonical.decision == "abstained"
    assert {"rod", "stick"} <= set(canonical.alternatives)
    assert {"cylinder", "rod", "stick"} <= set(result.field_proposals["visual_form"].value)
    assert any(action["action"] == "canonical_object_promotion_abstained" for action in result.taxonomy_mapping_actions)


def test_source_record_object_identity_cannot_bypass_promotion_gate():
    deterministic = parse_filename_semantics(
        "opaque_asset.png",
        source_metadata={
            "declared_canonical_object": "sword",
            "declared_category": "weapon",
            "declared_role": "weapon",
        },
    )
    assert deterministic.object_source == "source_record_metadata"

    result = reconcile_evidence(deterministic, None)
    canonical = result.field_proposals["canonical_object"]
    assert canonical.value is None
    assert canonical.decision == "abstained"
    assert "sword" in canonical.alternatives
    assert result.field_proposals["category"].value == "weapon"
    assert result.field_proposals["role"].value == "weapon"


def test_invalid_provider_domain_is_raw_but_never_normalized_or_supported():
    parsed = parse_reconciliation_response(
        {
            "field_proposals": {
                "domain": {
                    "raw_open_vocabulary_value": "weapon",
                    "normalized_controlled_value": "weapon",
                    "alternatives": [],
                    "support": ["taxonomy: weapon is a valid domain"],
                    "conflicts": [],
                }
            },
            "claims_accepted": [],
            "claims_rejected": [],
            "claims_unresolved": [],
        }
    )
    domain = parsed.field_proposals["domain"]
    assert domain.raw_open_vocabulary_value == "weapon"
    assert domain.normalized_controlled_value is None
    assert domain.value is None
    assert domain.decision == "rejected"
    assert domain.support == ()
    assert any(action.get("action") == "invalid_axis_value" for action in parsed.taxonomy_mapping_actions)
    assert any(conflict.get("code") == "invalid_taxonomy_provider_output" for conflict in parsed.unresolved_conflicts)
    assert "valid domain" not in json.dumps(parsed.to_dict())


def test_provider_object_aliases_are_boundary_only_and_output_is_canonical():
    parsed = parse_reconciliation_response(
        {
            "field_proposals": [
                {
                    "field": "object_name",
                    "raw_open_vocabulary_value": "buckler",
                    "normalized_controlled_value": "buckler",
                    "alternatives": [],
                    "support": ["filename"],
                    "conflicts": [],
                },
                {
                    "field": "canonical_object",
                    "raw_open_vocabulary_value": "shield",
                    "normalized_controlled_value": "shield",
                    "alternatives": [],
                    "support": ["vlm_visual"],
                    "conflicts": [],
                },
            ]
        }
    )
    assert set(parsed.field_proposals) == {"canonical_object"}
    assert parsed.field_proposals["canonical_object"].value == "buckler"
    assert any(
        action.get("action") == "legacy_provider_field_alias_canonicalized"
        for action in parsed.taxonomy_mapping_actions
    )
    assert any(
        conflict.get("code") == "duplicate_reconciliation_provider_field" for conflict in parsed.unresolved_conflicts
    )
    prompt = build_reconciliation_prompt(parse_filename_semantics("asset.png"), None)
    assert "use canonical_object, never object or object_name" in prompt


def test_palette_normalization_controls_roles_and_surfaces_outside_colors():
    normalized = normalize_visual_color_roles(
        {
            "primary": ["brownish-orange", "blue"],
            "outline": "black",
            "shadow": "darker brown along edges",
            "highlight": "lighter orange in center",
        },
        ["black", "brown", "orange", "red"],
    )
    assert normalized.raw_visual_color_roles["primary_colors"] == ("brownish-orange", "blue")
    assert normalized.color_roles.primary_colors == ("brown",)
    assert normalized.color_roles.outline_colors == ("black",)
    assert normalized.color_roles.shadow_colors == ("brown",)
    assert normalized.color_roles.highlight_colors == ("orange",)
    assert any(
        conflict["code"] == "color_role_outside_palette" and conflict["raw_visual_color"] == "blue"
        for conflict in normalized.conflicts
    )


def test_terminal_claim_collections_are_disjoint_and_legacy_inversion_is_repaired():
    claim = {"field": "explicit_material", "value": "bronze", "reason": "explicit source"}
    with pytest.raises(ReconciliationValidationError):
        ReconciliationResult(claims_accepted=(claim,), claims_rejected=(claim,))

    repaired = parse_reconciliation_response(
        {
            "field_proposals": {},
            "claims_rejected": [
                {
                    "claim": "explicit_material: bronze",
                    "reason": "Material is explicit. Claim is accepted, not rejected.",
                }
            ],
        }
    )
    assert len(repaired.claims_accepted) == 1
    assert repaired.claims_rejected == ()


def test_description_is_fact_generated_and_rejects_alternative_object_nouns():
    facts = {
        "canonical_object": None,
        "category": "weapon",
        "role": "weapon",
        "explicit_material": "bronze",
        "visual_form": ["rod", "elongated_form"],
        "object_alternatives": ["pencil", "pen", "matchstick"],
        "colors": {
            "palette_colors": ["black", "brown", "orange"],
            "primary_colors": ["brown"],
            "outline_colors": ["black"],
        },
        "shape": {"aspect": ["elongated"]},
    }
    valid, unsupported = validate_description("A bronze pencil or matchstick.", facts)
    assert not valid
    assert {"unsupported_object:pencil", "unsupported_object:matchstick"} <= set(unsupported)

    result = choose_or_regenerate_description(
        ["A bronze pencil or matchstick.", "A rod that could be a pen."],
        facts,
    )
    lowered = result["description"].lower()
    assert result["source"] == "regenerated_from_target_facts"
    assert result["claims_rejected"] == []
    assert result["candidate_claims_rejected"]
    assert "bronze" in lowered and "weapon icon" in lowered
    assert not {"pencil", "pen", "matchstick"} & set(lowered.replace(".", "").split())


def test_explicit_abstention_and_unsafe_promotions_have_forced_risk_floors():
    abstained = estimate_field_risk(
        "canonical_object",
        value_present=False,
        risk_features={"explicit_abstention": True},
    )
    unsafe_description = estimate_field_risk(
        "description",
        value_present=True,
        risk_features={"description_uses_alternative_interpretation": True},
    )
    assert abstained.uncertainty_1_20 is not None and abstained.uncertainty_1_20 >= 17
    assert unsafe_description.uncertainty_1_20 is not None and unsafe_description.uncertainty_1_20 >= 13


def test_adaptive_routing_runs_only_needed_stages_and_keeps_verifier_independent():
    cheap = decide_adaptive_routing(AdaptiveRoutingSignals())
    assert cheap.run_stage_a is True
    assert cheap.run_stage_b is cheap.run_stage_c is cheap.run_stage_d is False

    proposed = decide_adaptive_routing(AdaptiveRoutingSignals(canonical_object_missing=True, description_missing=True))
    assert proposed.run_stage_b and proposed.run_stage_c
    assert not proposed.run_stage_d
    assert "canonical_object_missing" in proposed.stage_b_reasons

    disputed = decide_adaptive_routing(
        AdaptiveRoutingSignals(
            vlm_deterministic_disagreement=True,
            filename_visual_color_conflict=True,
            material_visual_only=True,
            critical_field_risk_upper=0.55,
        ),
        verifier_risk_threshold=0.40,
    )
    assert disputed.run_stage_c and disputed.run_stage_d
    assert "prepare_disputed_claims_for_verifier" in disputed.stage_c_reasons
    assert "material_inferred_not_explicit" in disputed.stage_d_reasons
    assert disputed.verifier_cache_namespace == INDEPENDENT_VERIFIER_CACHE_NAMESPACE
    assert disputed.verifier_must_be_independent is True
