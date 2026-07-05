from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2
from spritelab.harvest.label_fusion_v2 import fuse_label_v2
from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.source_profiles import detect_source_profile


def _record(filename: str, source_id: str = "oga_496_rpg_icons_32fix") -> dict[str, str]:
    return {
        "sprite_id": filename.removesuffix(".png").lower(),
        "filename": filename,
        "relative_path": filename,
        "source_id": source_id,
        "source_name": source_id,
    }


def test_prefix_family_echo_of_malformed_filename_object_needs_review() -> None:
    profile = detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"})
    filename = LabelSuggestion("armor", "sho", confidence=0.75, source="filename_rules_v2")
    vlm = LabelSuggestion("armor", "sho", confidence=0.85, source="vlm_descriptor", source_consistency="consistent")

    fused = fuse_label_v2(filename, vlm, None, profile=profile)

    assert fused.bucket != "fused_automatically"
    assert fused.needs_review
    assert "malformed_filename_object" in fused.flags
    assert fused.safe_prefill.object_name != "sho"


def test_496_shoes_armour_and_elm_filename_contexts_are_safe() -> None:
    shoes = suggest_from_filename_v2(_record("A_Shoes01.png"))
    assert shoes.suggestion.object_name == "shoes"
    assert shoes.suggestion.object_name != "sho"
    fused_shoes = fuse_label_v2(
        shoes.suggestion,
        LabelSuggestion("armor", "shoes", confidence=0.85, source="vlm_descriptor", source_consistency="consistent"),
        None,
        profile=shoes.profile,
    )
    assert fused_shoes.safe_prefill.object_name == "shoes"
    assert fused_shoes.bucket == "auto_prefix_family_trusted"

    armour = suggest_from_filename_v2(_record("A_Armour01.png"))
    assert armour.suggestion.object_name == "armor"
    fused_armour = fuse_label_v2(
        armour.suggestion,
        LabelSuggestion("armor", "armour", confidence=0.85, source="vlm_descriptor", source_consistency="consistent"),
        None,
        profile=armour.profile,
    )
    assert fused_armour.safe_prefill.object_name == "armor"
    assert fused_armour.safe_prefill.object_name != "armour"

    elm = suggest_from_filename_v2(_record("C_Elm01.png"))
    assert elm.suggestion.object_name == "helmet"
    assert "headgear" in elm.suggestion.tags
    fused_elm = fuse_label_v2(
        elm.suggestion,
        LabelSuggestion("armor", "elm", confidence=0.85, source="vlm_descriptor", source_consistency="unclear"),
        None,
        profile=elm.profile,
    )
    assert fused_elm.safe_prefill.object_name != "elm"


def test_prefix_family_alternatives_can_win_candidate_ranking() -> None:
    profile = detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"})
    filename = LabelSuggestion(
        "armor",
        "armor",
        confidence=0.75,
        source="filename_rules_v2",
        candidate_object_names=("chestplate", "breastplate", "leather_armor"),
    )
    vlm = LabelSuggestion(
        "armor",
        "armor",
        confidence=0.86,
        source="vlm_descriptor",
        source_consistency="consistent",
        alternative_object_names=("chestplate", "breastplate"),
        candidate_object_names=("chestplate", "breastplate", "leather_armor"),
    )

    fused = fuse_label_v2(filename, vlm, None, profile=profile)

    assert fused.bucket == "auto_vlm_candidate_ranked"
    assert fused.safe_prefill.object_name == "chestplate"
    assert "vlm_alternative_candidate_match" in fused.flags


def test_candidate_conflict_reviews_outside_primary_even_with_matching_alternative() -> None:
    profile = detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"})
    filename = LabelSuggestion(
        "item_icon",
        "necklace",
        confidence=0.75,
        source="filename_rules_v2",
        candidate_object_names=("necklace", "pendant"),
    )
    vlm = LabelSuggestion(
        "item_icon",
        "chicken",
        confidence=0.88,
        source="vlm_descriptor",
        source_consistency="unclear",
        alternative_object_names=("pendant",),
        candidate_object_names=("necklace", "pendant"),
    )

    fused = fuse_label_v2(filename, vlm, None, profile=profile)

    assert fused.bucket == "needs_review_candidate_conflict"
    assert fused.needs_review
    assert fused.safe_prefill.object_name == "necklace"
    assert "vlm_outside_candidate_family" in fused.flags


def test_primary_candidate_can_auto_but_generic_alternative_case_reviews() -> None:
    profile = detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"})
    shoes = LabelSuggestion(
        "armor",
        "shoes",
        confidence=0.75,
        source="filename_rules_v2",
        candidate_object_names=("boots", "shoes"),
    )
    boot_vlm = LabelSuggestion(
        "armor",
        "boots",
        confidence=0.86,
        source="vlm_descriptor",
        source_consistency="consistent",
        candidate_object_names=("boots", "shoes"),
    )

    fused_boots = fuse_label_v2(shoes, boot_vlm, None, profile=profile)

    assert fused_boots.bucket == "auto_vlm_candidate_ranked"
    assert fused_boots.safe_prefill.object_name == "boots"
    assert "vlm_candidate_match" in fused_boots.flags

    gem_profile = detect_source_profile({"source_id": "oga_cc0_gem_7soul1"})
    weak_gem = LabelSuggestion(
        "material",
        "gem",
        confidence=0.25,
        source="filename_rules_v2",
        candidate_object_names=("round_gem", "triangle_gem"),
    )
    generic_vlm = LabelSuggestion(
        "material",
        "gem",
        confidence=0.9,
        source="vlm_descriptor",
        alternative_object_names=("round_gem",),
        candidate_object_names=("round_gem", "triangle_gem"),
    )

    fused_generic = fuse_label_v2(weak_gem, generic_vlm, None, profile=gem_profile)

    assert fused_generic.needs_review
    assert fused_generic.bucket == "needs_review_candidate_conflict"
    assert "vlm_generic_with_candidates" in fused_generic.flags


def test_exact_trusted_food_behavior_remains_strong() -> None:
    food = suggest_from_filename_v2(_record("butter.png", "oga_cc0_food_ocal"))
    vlm = LabelSuggestion("material", "gold_bar", tags=("gold",), confidence=0.85, source="vlm_descriptor")

    fused = fuse_label_v2(food.suggestion, vlm, None, profile=food.profile)

    assert fused.safe_prefill.object_name == "butter"
    assert fused.bucket == "auto_filename_with_vlm_conflict"
    assert not fused.needs_review
