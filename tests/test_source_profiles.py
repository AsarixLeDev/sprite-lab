from spritelab.harvest.source_profiles import detect_source_profile


def test_detect_clean_source_profiles() -> None:
    assert detect_source_profile({"source_id": "oga_cc0_food_ocal"}).name == "cc0_food"
    assert detect_source_profile({"source_id": "oga_cc0_food_ocal"}).trusted_filename
    assert detect_source_profile({"source_id": "oga_cc0_tool_ocal"}).name == "cc0_tool"
    assert detect_source_profile({"source_id": "oga_cc0_gem_7soul1"}).name == "cc0_gem"


def test_detect_rpg_and_unknown_profiles() -> None:
    assert detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"}).name == "oga_496_rpg_icons"
    assert detect_source_profile({"source_id": "some_unknown_pack"}).name == "generic_unknown"
