from spritelab.harvest.source_profiles import (
    detect_source_profile,
    is_exact_filename_trusted,
    is_prefix_family_trusted,
    source_profile_to_json,
)


def test_496_profile_is_prefix_family_not_exact_filename_trusted() -> None:
    profile = detect_source_profile({"source_id": "oga_496_rpg_icons_32fix"})

    assert profile.name == "oga_496_rpg_icons"
    assert profile.filename_trust == "prefix_family"
    assert not profile.trusted_filename
    assert not is_exact_filename_trusted(profile)
    assert is_prefix_family_trusted(profile)
    assert source_profile_to_json(profile)["filename_trust"] == "prefix_family"


def test_clean_food_tool_gem_profiles_remain_exact_trusted() -> None:
    for source_id in ("oga_cc0_food_ocal", "oga_cc0_tool_ocal", "oga_cc0_gem_7soul1"):
        profile = detect_source_profile({"source_id": source_id})
        assert profile.filename_trust == "exact"
        assert profile.trusted_filename
        assert is_exact_filename_trusted(profile)


def test_unknown_profile_has_no_filename_trust() -> None:
    profile = detect_source_profile({"source_id": "some_unknown_pack"})

    assert profile.filename_trust == "none"
    assert not profile.trusted_filename
    assert not is_prefix_family_trusted(profile)
