from spritelab.harvest.label_fusion_v2 import FusionThresholds, fuse_label_v2
from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.source_profiles import SourceProfile, detect_source_profile


def test_detect_clean_source_profiles() -> None:
    assert detect_source_profile({"source_id": "oga_cc0_food_ocal"}).name == "cc0_food"
    assert detect_source_profile({"source_id": "oga_cc0_food_ocal"}).trusted_filename
    assert detect_source_profile({"source_id": "oga_cc0_tool_ocal"}).name == "cc0_tool"
    assert detect_source_profile({"source_id": "oga_cc0_gem_7soul1"}).name == "cc0_gem"


def test_detect_rpg_and_unknown_profiles() -> None:
    rpg_profile = detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"})
    assert rpg_profile.name == "oga_496_rpg_icons"
    assert rpg_profile.sheet_specialization == "rpg_496"
    assert rpg_profile.fusion_threshold_override == 0.65
    assert detect_source_profile({"source_id": "some_unknown_pack"}).name == "generic_unknown"


def test_profile_fusion_threshold_override_controls_filename_strength() -> None:
    filename = LabelSuggestion("item_icon", "lantern", confidence=0.7, source="filename_rules_v2")
    thresholds = FusionThresholds(filename_confidence_threshold=0.8)
    generic_profile = SourceProfile("generic", "unknown", "none", ("unknown",), ())
    override_profile = SourceProfile("override", "unknown", "none", ("unknown",), (), fusion_threshold_override=0.65)

    generic = fuse_label_v2(filename, None, None, profile=generic_profile, thresholds=thresholds)
    overridden = fuse_label_v2(filename, None, None, profile=override_profile, thresholds=thresholds)

    assert "filename_weak" in generic.flags
    assert "filename_weak" not in overridden.flags
