from __future__ import annotations

from spritelab.harvest.label_candidates import object_is_generic
from spritelab.harvest.label_fusion_v2 import fuse_label_v2
from spritelab.harvest.label_schema import LabelSuggestion, confidence_tier_for_bucket, safe_fused_label_to_json
from spritelab.harvest.label_taxonomy import CATEGORY_VALUES
from spritelab.harvest.source_profiles import (
    detect_source_profile,
    hardcoded_source_profiles,
    loaded_source_profiles,
    source_profile_to_json,
)


def test_packaged_source_profiles_match_hardcoded_fallback() -> None:
    hardcoded = hardcoded_source_profiles()
    loaded = loaded_source_profiles()
    assert {name: source_profile_to_json(profile) for name, profile in loaded.items()} == {
        name: source_profile_to_json(profile) for name, profile in hardcoded.items()
    }


def test_packaged_taxonomy_preserves_existing_category_order() -> None:
    assert CATEGORY_VALUES == (
        "unknown",
        "item_icon",
        "block",
        "plant",
        "ui_icon",
        "entity",
        "character",
        "weapon",
        "tool",
        "armor",
        "material",
        "effect_icon",
        "environment_prop",
    )


def test_config_backed_denylist_blocks_hallucination_malformed_and_generic() -> None:
    profile = detect_source_profile({"source_id": "some_unknown_pack"})
    filename = LabelSuggestion("unknown", "mystery", confidence=0.1, source="filename_rules_v2")
    hallucination = LabelSuggestion("material", "gold_bar", confidence=0.9, source="vlm_descriptor")
    fused = fuse_label_v2(filename, hallucination, None, profile=profile)
    assert fused.bucket == "needs_review"
    assert "vlm_known_hallucination" in fused.flags
    malformed = LabelSuggestion("item_icon", "sho", confidence=0.9, source="filename_rules_v2")
    assert "malformed_filename_object" in fuse_label_v2(malformed, None, None, profile=profile).flags
    assert object_is_generic("object")


def test_bucket_tiers_and_provenance_are_additive() -> None:
    assert confidence_tier_for_bucket("auto_filename_trusted") == "T0"
    assert confidence_tier_for_bucket("auto_rpg_496_specialized") == "T1"
    assert confidence_tier_for_bucket("fused_automatically") == "T2"
    assert confidence_tier_for_bucket("auto_vlm_when_filename_weak") == "T3"
    assert confidence_tier_for_bucket("needs_review") == "T4"
    profile = detect_source_profile({"source_id": "oga_cc0_food_ocal"})
    fused = fuse_label_v2(LabelSuggestion("item_icon", "apple", confidence=0.95), None, None, profile=profile)
    payload = safe_fused_label_to_json(fused)
    assert payload["label_confidence_tier"] == "T0"
    assert payload["fusion_bucket"] == fused.bucket
    assert payload["label_tier_reason"] == f"fusion_bucket:{fused.bucket}"
