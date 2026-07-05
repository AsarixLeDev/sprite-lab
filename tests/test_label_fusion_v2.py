from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2
from spritelab.harvest.label_fusion_v2 import fuse_label_v2
from spritelab.harvest.label_schema import LabelSuggestion


def _filename(name: str):
    return suggest_from_filename_v2({"sprite_id": name, "relative_path": f"{name}.png", "source_id": "oga_cc0_food_ocal"})


def test_trusted_food_filename_wins_over_hallucinations() -> None:
    cases = [
        ("butter", "gold_bar"),
        ("cheese_wedge", "gold_bar"),
        ("orange", "coin"),
        ("kiwi", "coin_stack"),
    ]
    for filename_object, vlm_object in cases:
        result = _filename(filename_object)
        vlm = LabelSuggestion(category="material", object_name=vlm_object, tags=("gold", "currency"), confidence=0.85, source="vlm_descriptor")
        fused = fuse_label_v2(result.suggestion, vlm, None, profile=result.profile)
        assert fused.safe_prefill.object_name == filename_object
        assert not fused.needs_review
        assert "vlm_food_as_currency_hallucination" in fused.flags


def test_agreement_weak_filename_and_degenerate_rules() -> None:
    apple = _filename("apple")
    vlm_apple = LabelSuggestion(category="item_icon", object_name="apple", tags=("fruit",), confidence=0.85, source="vlm_descriptor")
    fused = fuse_label_v2(apple.suggestion, vlm_apple, None, profile=apple.profile)
    assert fused.bucket in {"auto_filename_trusted", "fused_automatically"}

    weak = LabelSuggestion(category="unknown", object_name="mystery", confidence=0.55, source="filename_rules_v2")
    vlm = LabelSuggestion(category="item_icon", object_name="key", tags=("key",), confidence=0.85, source="vlm_descriptor")
    weak_profile = suggest_from_filename_v2({"sprite_id": "tile_001", "relative_path": "tile_001.png", "source_id": "unknown"}).profile
    fused = fuse_label_v2(weak, vlm, None, profile=weak_profile)
    assert fused.safe_prefill.object_name == "key"
    assert fused.bucket == "auto_vlm_when_filename_weak"

    degenerate = LabelSuggestion(category="unknown", object_name="unknown", confidence=0.95, warnings=("degenerate_response",), source="vlm_descriptor")
    fused = fuse_label_v2(weak, degenerate, None, profile=weak_profile)
    assert fused.needs_review
    assert "vlm_degenerate" in fused.flags


def test_profile_category_correction_and_conflict_prefill() -> None:
    tool = suggest_from_filename_v2({"sprite_id": "compass", "relative_path": "compass.png", "source_id": "oga_cc0_tool_ocal"})
    vlm = LabelSuggestion(category="item_icon", object_name="compass", tags=("navigation",), confidence=0.85, source="vlm_descriptor")
    fused = fuse_label_v2(tool.suggestion, vlm, None, profile=tool.profile)
    assert fused.safe_prefill.category == "tool"

    gem = suggest_from_filename_v2({"sprite_id": "ruby", "relative_path": "ruby.png", "source_id": "oga_cc0_gem_7soul1"})
    fused = fuse_label_v2(gem.suggestion, vlm, None, profile=gem.profile)
    assert fused.safe_prefill.category == "material"

    cheese = _filename("cheese_wedge")
    bad_vlm = LabelSuggestion(category="material", object_name="gold_bar", confidence=0.85, source="vlm_descriptor")
    fused = fuse_label_v2(cheese.suggestion, bad_vlm, None, profile=cheese.profile)
    assert fused.bucket == "auto_filename_with_vlm_conflict"
    assert fused.safe_prefill.object_name == "cheese_wedge"
