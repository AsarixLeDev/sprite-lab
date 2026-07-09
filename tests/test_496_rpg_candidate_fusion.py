from spritelab.harvest.label_fusion_v2 import fuse_label_v2
from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.source_profiles import detect_source_profile


def _profile():
    return detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"})


def test_shoes_primary_candidate_can_auto_rank() -> None:
    candidates = ("shoes", "boots", "boot", "footwear")
    filename = LabelSuggestion(
        "armor", "shoes", confidence=0.75, source="filename_rules_v2", candidate_object_names=candidates
    )
    vlm = LabelSuggestion(
        "armor",
        "boot",
        confidence=0.86,
        source="vlm_descriptor",
        source_consistency="consistent",
        candidate_object_names=candidates,
    )

    fused = fuse_label_v2(filename, vlm, None, profile=_profile())

    assert fused.bucket == "auto_vlm_candidate_ranked"
    assert fused.safe_prefill.object_name in {"boot", "boots"}
    assert "vlm_candidate_match" in fused.flags


def test_necklace_primary_candidate_can_auto_rank() -> None:
    candidates = ("necklace", "amulet", "pendant", "medallion", "charm")
    filename = LabelSuggestion(
        "item_icon", "necklace", confidence=0.75, source="filename_rules_v2", candidate_object_names=candidates
    )
    vlm = LabelSuggestion(
        "item_icon",
        "pendant",
        confidence=0.86,
        source="vlm_descriptor",
        source_consistency="consistent",
        candidate_object_names=candidates,
    )

    fused = fuse_label_v2(filename, vlm, None, profile=_profile())

    assert fused.bucket == "auto_vlm_candidate_ranked"
    assert fused.safe_prefill.object_name == "pendant"


def test_gold_coin_candidate_can_auto_rank_without_hallucination_block() -> None:
    candidates = ("gold", "gold_coin", "coin", "gold_ingot", "gold_nugget", "currency")
    filename = LabelSuggestion(
        "material", "gold", confidence=0.75, source="filename_rules_v2", candidate_object_names=candidates
    )
    vlm = LabelSuggestion(
        "material",
        "coin",
        confidence=0.86,
        source="vlm_descriptor",
        source_consistency="consistent",
        candidate_object_names=candidates,
    )

    fused = fuse_label_v2(filename, vlm, None, profile=_profile())

    assert fused.bucket == "auto_vlm_candidate_ranked"
    assert fused.safe_prefill.object_name in {"coin", "gold_coin"}
    assert "vlm_known_hallucination" not in fused.flags


def test_helmet_contradicted_mushroom_stays_review_and_never_uses_mushroom() -> None:
    candidates = ("helmet", "helm", "headgear")
    filename = LabelSuggestion(
        "armor", "helmet", confidence=0.75, source="filename_rules_v2", candidate_object_names=candidates
    )
    vlm = LabelSuggestion(
        "plant",
        "mushroom",
        confidence=0.9,
        source="vlm_descriptor",
        source_consistency="contradicted",
        evidence_against_source=("cap and stem silhouette",),
        candidate_object_names=candidates,
    )

    fused = fuse_label_v2(filename, vlm, None, profile=_profile())

    assert fused.needs_review
    assert fused.bucket == "needs_review_candidate_conflict"
    assert fused.safe_prefill.object_name != "mushroom"
    assert "vlm_outside_candidate_family" in fused.flags


def test_armor_alternatives_are_seen_without_forcing_unrelated_label() -> None:
    candidates = ("armor", "chestplate", "breastplate", "cuirass", "armor_piece", "leather_armor")
    filename = LabelSuggestion(
        "armor", "armor", confidence=0.75, source="filename_rules_v2", candidate_object_names=candidates
    )
    vlm = LabelSuggestion(
        "armor",
        "armor",
        confidence=0.86,
        source="vlm_descriptor",
        source_consistency="consistent",
        alternative_object_names=("chestplate", "breastplate", "leather_armor"),
        candidate_object_names=candidates,
    )

    fused = fuse_label_v2(filename, vlm, None, profile=_profile())

    assert fused.bucket in {"auto_prefix_family_trusted", "auto_vlm_candidate_ranked"}
    assert fused.safe_prefill.object_name in set(candidates)
    assert "vlm_alternative_candidate_match" in fused.flags


def test_generic_vlm_with_candidates_reviews_unless_alternative_is_clear() -> None:
    candidates = ("necklace", "amulet", "pendant", "medallion", "charm")
    filename = LabelSuggestion(
        "item_icon", "necklace", confidence=0.75, source="filename_rules_v2", candidate_object_names=candidates
    )
    generic = LabelSuggestion(
        "item_icon",
        "item",
        confidence=0.7,
        source="vlm_descriptor",
        source_consistency="unclear",
        candidate_object_names=candidates,
    )

    fused_generic = fuse_label_v2(filename, generic, None, profile=_profile())

    assert fused_generic.needs_review
    assert fused_generic.safe_prefill.object_name == "necklace"
    assert "vlm_generic_with_candidates" in fused_generic.flags

    generic_with_alternative = LabelSuggestion(
        "item_icon",
        "item",
        confidence=0.7,
        source="vlm_descriptor",
        source_consistency="unclear",
        alternative_object_names=("pendant",),
        candidate_object_names=candidates,
    )

    fused_alternative = fuse_label_v2(filename, generic_with_alternative, None, profile=_profile())

    assert fused_alternative.bucket == "auto_vlm_candidate_ranked"
    assert fused_alternative.safe_prefill.object_name == "pendant"
    assert "vlm_alternative_candidate_match" in fused_alternative.flags
