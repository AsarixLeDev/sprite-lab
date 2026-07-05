import json

from PIL import Image

from spritelab.dataset_maker.prefill import (
    build_vlm_descriptor_prompt,
    compute_image_cache_key,
    parse_descriptor_suggestion,
    prepare_vlm_image_view,
)
from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2
from spritelab.harvest.golden import GoldenLabel
from spritelab.harvest.golden_lint import lint_golden_file, write_jsonl
from spritelab.harvest.label_candidates import candidate_objects_for_record, exact_sheet_object_for_record
from spritelab.harvest.label_fusion_v2 import fuse_label_v2
from spritelab.harvest.label_schema import LabelSuggestion, label_suggestion_to_json
from spritelab.harvest.label_v2_eval import evaluate_label_v2, label_v2_error_records
from spritelab.harvest.source_profiles import detect_source_profile


def _record(sprite_id: str, source_id: str) -> dict[str, str]:
    return {
        "sprite_id": sprite_id,
        "filename": f"{sprite_id}.png",
        "relative_path": f"{sprite_id}.png",
        "source_id": source_id,
        "source_name": source_id,
    }


def test_descriptor_prompt_strong_source_schema_traps_and_views() -> None:
    prompt = build_vlm_descriptor_prompt(
        {"object_name": "butter", "category": "item_icon", "confidence": 0.96},
        image_facts={"content_width": 12, "content_height": 8, "dominant_colors": ["yellow"]},
        candidate_object_names=("butter", "cheese_wedge"),
        image_view_mode="both",
    )

    assert "'butter'" in prompt
    assert "possible_object_name MUST equal" in prompt
    assert "Known tiny-sprite traps" in prompt
    assert "source_consistency" in prompt
    assert "evidence_for_source" in prompt
    assert "evidence_against_source" in prompt
    assert "alternative_object_names" in prompt
    assert "- full canvas view: preserves 32x32 placement" in prompt
    assert "- cropped close-up view: makes small details easier to inspect" in prompt
    assert "magenta background" in prompt and "not sprite content" in prompt
    assert "You are not the final label authority" in prompt


def test_descriptor_prompt_weak_source_contains_candidates() -> None:
    prompt = build_vlm_descriptor_prompt(
        {"object_name": "gem", "confidence": 0.25},
        candidate_object_names=("round_gem", "triangle_gem"),
    )

    assert "No trusted object name is available" in prompt
    assert "Candidate object names from this source/profile" in prompt
    assert "- round_gem" in prompt
    assert "- triangle_gem" in prompt


def test_sheet_candidate_maps_cover_gem_tool_and_food_cells() -> None:
    expected_gems = {
        "gem-7soul1_r000_c000": "round_gem",
        "gem-7soul1_r000_c001": "triangle_gem",
        "gem-7soul1_r000_c002": "diamond_gem",
        "gem-7soul1_r000_c003": "oval_gem",
        "gem-7soul1_r001_c000": "mixed_gem",
        "gem-7soul1_r001_c001": "ruby_gem",
        "gem-7soul1_r001_c002": "sapphire_gem",
        "gem-7soul1_r001_c003": "dark_blue_gem",
        "gem-7soul1_r002_c000": "red_gem",
        "gem-7soul1_r002_c001": "gray_gem",
    }
    for sprite_id, object_name in expected_gems.items():
        record = _record(sprite_id, "oga_cc0_gem_7soul1")
        profile = detect_source_profile(record)
        assert exact_sheet_object_for_record(record, profile) == object_name
        assert suggest_from_filename_v2(record).suggestion.object_name == object_name

    expected_tools = {
        "tool-ocal_r000_c001": "compass",
        "tool-ocal_r000_c002": "compass",
        "tool-ocal_r000_c004": "compass_geometric",
        "tool-ocal_r000_c005": "compass_geometric",
        "tool-ocal_r001_c000": "ruler",
        "tool-ocal_r001_c001": "ruler_triangle",
        "tool-ocal_r001_c003": "meter",
        "tool-ocal_r001_c004": "meter",
        "tool-ocal_r002_c001": "tool_case",
        "tool-ocal_r002_c002": "tool_case",
        "tool-ocal_r002_c003": "secateur",
    }
    for sprite_id, object_name in expected_tools.items():
        record = _record(sprite_id, "oga_cc0_tool_ocal")
        assert suggest_from_filename_v2(record).suggestion.object_name == object_name

    food = _record("food-ocal_r006_c006", "oga_cc0_food_ocal")
    assert suggest_from_filename_v2(food).suggestion.object_name == "orange_juice"
    generic_tool = _record("tool-ocal_r009_c009", "oga_cc0_tool_ocal")
    candidates = candidate_objects_for_record(
        generic_tool,
        detect_source_profile(generic_tool),
        {"object_name": "tool", "confidence": 0.25},
    )
    assert "compass" in candidates and "tool_case" in candidates


def test_descriptor_image_views_sizes_edges_background_and_cache_key() -> None:
    full = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    full.putpixel((0, 0), (255, 0, 0, 255))
    full.putpixel((1, 0), (0, 0, 255, 255))

    full_view = prepare_vlm_image_view(full, view="full")
    assert full_view.size == (512, 512)
    assert full_view.getpixel((15, 0)) == (255, 0, 0)
    assert full_view.getpixel((16, 0)) == (0, 0, 255)
    assert full_view.getpixel((511, 511)) == (255, 0, 255)

    crop_default = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(5, 25):
        for x in range(5, 25):
            crop_default.putpixel((x, y), (10, 20, 30, 255))
    assert prepare_vlm_image_view(crop_default, view="crop").size == (512, 512)

    small = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(14, 18):
        for x in range(14, 18):
            small.putpixel((x, y), (10, 20, 30, 255))
    small_crop = prepare_vlm_image_view(small, view="crop")
    assert small_crop.size == (768, 768)
    assert small_crop.getpixel((0, 0)) == (255, 0, 255)

    assert compute_image_cache_key(full, model="m", prompt_version="p", image_view="full") != compute_image_cache_key(
        full, model="m", prompt_version="p", image_view="crop"
    )
    assert compute_image_cache_key(full, model="m", prompt_version="p1") != compute_image_cache_key(full, model="m", prompt_version="p2")


def test_descriptor_parser_accepts_new_old_and_bad_enums() -> None:
    parsed = parse_descriptor_suggestion(
        json.dumps(
            {
                "visual_description": "Yellow rectangular food.",
                "visual_tags": ["Yellow", "rectangular"],
                "source_consistency": "consistent",
                "evidence_for_source": ["yellow block"],
                "evidence_against_source": [],
                "possible_object_name": "Butter",
                "alternative_object_names": ["cheese_wedge", "gold_bar", "corn", "lemon"],
                "possible_category": "item_icon",
                "uncertainty": "confident",
                "warnings": [],
            }
        )
    )
    assert parsed.source_consistency == "consistent"
    assert parsed.object_name == "butter"
    assert parsed.alternative_object_names == ("cheese_wedge", "gold_bar", "corn")
    assert parsed.evidence_for_source == ("yellow block",)
    assert parsed.evidence_against_source == ()
    assert parsed.tags == ("yellow", "rectangular")

    old = parse_descriptor_suggestion(
        json.dumps(
            {
                "visual_description": "A round yellow object.",
                "visual_tags": ["Roundish"],
                "possible_object_name": "coin",
                "possible_category": "material",
                "agrees_with_source": "no",
                "contradiction_reason": "metal coin visible",
                "uncertainty": "maybe",
                "warnings": [],
            }
        )
    )
    assert old.source_consistency == "contradicted"
    assert old.uncertainty == "unsure"
    assert old.evidence_against_source == ("metal coin visible",)


def test_fusion_keeps_trusted_food_over_known_vlm_traps_and_preserves_alternatives() -> None:
    for source_object, vlm_object in (
        ("butter", "gold_bar"),
        ("cheese_wedge", "gold_bar"),
        ("orange", "coin"),
        ("milk_carton", "red_potion_bottle"),
    ):
        filename = suggest_from_filename_v2(_record(source_object, "oga_cc0_food_ocal"))
        vlm = LabelSuggestion(
            category="material",
            object_name=vlm_object,
            tags=("gold", "currency"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="contradicted",
            evidence_against_source=("looks metallic",),
            alternative_object_names=("butter", "cheese_wedge"),
        )
        fused = fuse_label_v2(filename.suggestion, vlm, None, profile=filename.profile)
        assert fused.safe_prefill.object_name == source_object
        assert not fused.needs_review
        assert "vlm_known_hallucination" in fused.flags
        assert label_suggestion_to_json(fused.vlm_suggestion)["alternative_object_names"] == ["butter", "cheese_wedge"]


def test_fusion_candidate_ranked_weak_gem_and_tool_cases() -> None:
    gem_profile = detect_source_profile(_record("gem-7soul1_r000_c001", "oga_cc0_gem_7soul1"))
    weak_gem = LabelSuggestion("material", "gem", confidence=0.25, source="filename_rules_v2", candidate_object_names=("triangle_gem", "round_gem"))
    vlm_gem = LabelSuggestion("material", "triangle_gem", tags=("triangle",), confidence=0.85, source="vlm_descriptor", candidate_object_names=("triangle_gem", "round_gem"))
    fused = fuse_label_v2(weak_gem, vlm_gem, None, profile=gem_profile)
    assert fused.safe_prefill.object_name == "triangle_gem"
    assert fused.bucket == "auto_vlm_candidate_ranked"

    generic_vlm = LabelSuggestion("material", "gem", confidence=0.9, source="vlm_descriptor", candidate_object_names=("triangle_gem", "round_gem"))
    fused_generic = fuse_label_v2(weak_gem, generic_vlm, None, profile=gem_profile)
    assert fused_generic.needs_review

    tool_profile = detect_source_profile(_record("tool-ocal_r000_c001", "oga_cc0_tool_ocal"))
    weak_tool = LabelSuggestion("tool", "tool", confidence=0.25, source="filename_rules_v2", candidate_object_names=("compass", "ruler"))
    vlm_tool = LabelSuggestion("tool", "compass", tags=("navigation",), confidence=0.85, source="vlm_descriptor", candidate_object_names=("compass", "ruler"))
    fused_tool = fuse_label_v2(weak_tool, vlm_tool, None, profile=tool_profile)
    assert fused_tool.safe_prefill.object_name == "compass"
    assert fused_tool.bucket == "auto_vlm_candidate_ranked"


def test_deterministic_canonicalization_polish() -> None:
    assert suggest_from_filename_v2(_record("scissor", "oga_cc0_tool_ocal")).suggestion.object_name == "scissors"
    assert suggest_from_filename_v2(_record("wiresnip_blue", "oga_cc0_tool_ocal")).suggestion.object_name == "wiresnips_blue"
    assert suggest_from_filename_v2(_record("case", "oga_cc0_tool_ocal")).suggestion.object_name == "tool_case"
    assert suggest_from_filename_v2(_record("amethist", "oga_cc0_gem_7soul1")).suggestion.object_name == "amethyst"
    assert suggest_from_filename_v2(_record("saphire", "oga_cc0_gem_7soul1")).suggestion.object_name == "sapphire"
    assert suggest_from_filename_v2(_record("ovale_gem", "oga_cc0_gem_7soul1")).suggestion.object_name == "oval_gem"
    assert suggest_from_filename_v2(_record("tomatoes_cherry", "oga_cc0_food_ocal")).suggestion.object_name == "cherry_tomatoes"
    assert suggest_from_filename_v2(_record("cherry_tomatoes", "oga_cc0_food_ocal")).suggestion.object_name == "cherry_tomatoes"
    assert suggest_from_filename_v2(_record("juice_orange", "oga_cc0_food_ocal")).suggestion.object_name == "orange_juice"
    ice = suggest_from_filename_v2(_record("ice_cream_sandwich", "oga_cc0_food_ocal")).suggestion
    assert ice.category == "item_icon"


def test_golden_lint_flags_and_writes_non_destructive_fixes(tmp_path) -> None:
    golden = tmp_path / "golden.jsonl"
    rows = [
        {"sprite_id": "a", "category": "effect_icon", "object_name": "ice_cream_sandwich", "tags": ["ice_cream_sandwich"]},
        {"sprite_id": "b", "category": "material", "object_name": "saphire", "tags": ["saphire"]},
        {"sprite_id": "c", "category": "material", "object_name": "gem", "tags": ["gem"]},
        {"sprite_id": "dup", "category": "item_icon", "object_name": "apple", "tags": ["apple"]},
        {"sprite_id": "dup", "category": "tool", "object_name": "compass", "tags": ["tool"]},
    ]
    write_jsonl(golden, rows)

    issues, suggestions = lint_golden_file(golden, fix=True)
    codes = {issue["code"] for issue in issues}
    assert "ice_cream_sandwich_effect_icon" in codes or "category_object_mismatch" in codes
    assert "typo_object_name" in codes
    assert "sparse_tags" in codes
    assert "duplicate_conflicting_labels" in codes
    out = tmp_path / "fixed_suggestions.jsonl"
    write_jsonl(out, suggestions)
    assert "suggestion_only" in out.read_text(encoding="utf-8")


def test_eval_errors_out_records_mismatches_missing_and_keeps_bucket_metrics() -> None:
    golden = {
        "a": GoldenLabel("a", "item_icon", "apple", ("fruit",)),
        "b": GoldenLabel("b", "tool", "compass", ("tool",)),
        "missing": GoldenLabel("missing", "material", "round_gem", ("gem",)),
    }
    records = [
        {
            "sprite_id": "a",
            "source_id": "food",
            "safe_prefill": {"category": "item_icon", "object_name": "orange", "tags": ["fruit"]},
            "vlm_descriptor": {"object_name": "orange", "alternative_object_names": ["apple"]},
            "label_quality": {"bucket": "auto_filename_trusted", "needs_review": False, "flags": ["auto_filename_trusted"]},
        },
        {
            "sprite_id": "b",
            "source_id": "tool",
            "safe_prefill": {"category": "tool", "object_name": "compass", "tags": ["tool"]},
            "label_quality": {"bucket": "auto_vlm_candidate_ranked", "needs_review": False, "flags": ["auto_vlm_candidate_ranked"]},
        },
    ]

    result = evaluate_label_v2(golden, records)
    errors = label_v2_error_records(golden, records)
    assert result["buckets"]["auto_vlm_candidate_ranked"] == 1
    assert any(error["sprite_id"] == "missing" and error["reason"] == "missing_prediction" for error in errors)
    apple_error = next(error for error in errors if error["sprite_id"] == "a")
    assert apple_error["golden"]["object_name"] == "apple"
    assert apple_error["predicted"]["object_name"] == "orange"
    assert apple_error["bucket"] == "auto_filename_trusted"
    assert apple_error["flags"] == ["auto_filename_trusted"]
    assert apple_error["object_exact_match"] is False
    assert apple_error["vlm_possible_object"] == "orange"
    assert apple_error["vlm_alternative_object_names"] == ["apple"]
