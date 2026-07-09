from __future__ import annotations

import pytest

from spritelab.harvest.creative_concepts import parse_creative_concept
from spritelab.harvest.semantic_extractors import extract_attribute_tokens, extract_base_object
from spritelab.harvest.semantic_v3 import (
    DEFAULT_NEGATIVE_TAGS,
    SCHEMA_VERSION,
    attach_semantic_v3,
    build_semantic_v3_record,
    semantic_v3_from_json,
    semantic_v3_to_json,
)


def _prediction(
    object_name: str,
    category: str,
    *,
    tags: list[str] | None = None,
    materials: list[str] | None = None,
    mood: list[str] | None = None,
    dominant_colors: list[str] | None = None,
    shape_hints: list[str] | None = None,
    candidates: list[str] | None = None,
    vlm_alternatives: list[str] | None = None,
    bucket: str = "auto_filename_trusted",
) -> dict:
    return {
        "sprite_id": f"test_{object_name}",
        "candidate_object_names": list(candidates or []),
        "safe_prefill": {
            "category": category,
            "object_name": object_name,
            "tags": list(tags or []),
            "short_description": "",
            "materials": list(materials or []),
            "mood": list(mood or []),
        },
        "visual_facts": {
            "dominant_colors": list(dominant_colors or []),
            "shape_hints": list(shape_hints or []),
        },
        "vlm_descriptor": {
            "object_name": "",
            "alternative_object_names": list(vlm_alternatives or []),
        },
        "source_profile": {"name": "oga_496_rpg_icons", "domain": "rpg_icons"},
        "bucket": bucket,
        "label_quality": {"bucket": bucket},
    }


def test_converts_simple_record_to_semantic_v3() -> None:
    output = attach_semantic_v3(_prediction("ruby_gem", "material", dominant_colors=["red", "black"]))
    semantic = output["semantic_v3"]
    assert semantic["schema_version"] == SCHEMA_VERSION
    assert semantic["category"] == "material"
    assert semantic["object_name"] == "ruby_gem"
    assert semantic["base_object"] == "gem"
    assert semantic["open_name"] == "ruby gem"
    assert "red" in semantic["attributes"]["colors"]
    assert "crystal" in semantic["attributes"]["materials"]
    assert "crafting_material" in semantic["attributes"]["function"]
    assert semantic["captions"]
    assert semantic["negative_tags"] == list(DEFAULT_NEGATIVE_TAGS)


def test_conversion_does_not_change_category_or_object_name() -> None:
    prediction = _prediction("golden_chestplate", "armor")
    output = attach_semantic_v3(prediction)
    assert output["safe_prefill"]["category"] == "armor"
    assert output["safe_prefill"]["object_name"] == "golden_chestplate"
    assert output["semantic_v3"]["category"] == "armor"
    assert output["semantic_v3"]["object_name"] == "golden_chestplate"
    # every original prediction field is preserved untouched
    for key, value in prediction.items():
        assert output[key] == value


@pytest.mark.parametrize(
    ("object_name", "expected_base"),
    [
        ("golden_chestplate", "chestplate"),
        ("red_potion", "potion"),
        ("electric_arrow", "arrow"),
        ("ruby_gem", "gem"),
        ("wooden_shield", "shield"),
        ("leather_chestplate", "chestplate"),
        ("yellow_vial", "vial"),
        ("fire_feather", "feather"),
        ("square_gem", "gem"),
    ],
)
def test_base_object_extraction(object_name: str, expected_base: str) -> None:
    base, warnings = extract_base_object(object_name)
    assert base == expected_base
    assert not warnings


def test_effect_icon_identity_becomes_its_own_base_object() -> None:
    base, warnings = extract_base_object("shadow", category="effect_icon")
    assert base == "shadow"
    assert not warnings


def test_unknown_compound_falls_back_with_warning() -> None:
    base, warnings = extract_base_object("frobnicated_zorblet")
    assert base == "frobnicated_zorblet"
    assert "base_object_fallback_full_name" in warnings


def test_colors_extracted_from_object_name_and_tags() -> None:
    record = build_semantic_v3_record(
        _prediction("golden_chestplate", "armor", tags=["armor", "yellow"], dominant_colors=["black"])
    )
    assert "gold" in record.attributes.colors
    assert "yellow" in record.attributes.colors
    assert "black" in record.attributes.colors


def test_materials_extracted_from_object_name_and_tags() -> None:
    record = build_semantic_v3_record(_prediction("wooden_shield", "armor", tags=["metal"]))
    assert "wood" in record.attributes.materials
    assert "metal" in record.attributes.materials
    record = build_semantic_v3_record(_prediction("red_potion", "item_icon"))
    assert "glass" in record.attributes.materials
    assert "liquid" in record.attributes.materials


@pytest.mark.parametrize(
    ("object_name", "expected_effect"),
    [
        ("electric_arrow", "electric"),
        ("fire_sword", "fire"),
        ("ice_arrow", "ice"),
        ("poison_dagger", "poison"),
        ("calming_spores", "calming"),
    ],
)
def test_effects_extracted_from_object_name(object_name: str, expected_effect: str) -> None:
    record = build_semantic_v3_record(_prediction(object_name, "item_icon"))
    assert expected_effect in record.attributes.effects


def test_charged_token_maps_to_electric_and_charged() -> None:
    extracted = extract_attribute_tokens(("charged",))
    assert "electric" in extracted.effects
    assert "charged" in extracted.effects


def test_generates_multiple_caption_styles() -> None:
    record = build_semantic_v3_record(_prediction("red_potion", "item_icon", dominant_colors=["red", "black"]))
    assert len(record.captions) >= 4
    assert "red potion" in record.captions
    assert any(caption.startswith("32x32 pixel art") for caption in record.captions)
    assert any("transparent background" in caption for caption in record.captions)
    # attribute dropout keeps the bare base object available
    assert "potion" in record.captions


def test_captions_include_32x32_pixel_art_style_phrase() -> None:
    record = build_semantic_v3_record(_prediction("golden_chestplate", "armor"))
    assert any("32x32 pixel art" in caption for caption in record.captions)
    assert record.prompt_phrases
    assert record.prompt_phrases[0].startswith("32x32 pixel art")


def test_captions_do_not_invent_ungrounded_words() -> None:
    record = build_semantic_v3_record(_prediction("red_potion", "item_icon", dominant_colors=["red", "black", "white"]))
    allowed = {
        "32x32",
        "pixel",
        "art",
        "fantasy",
        "rpg",
        "icon",
        "centered",
        "made",
        "of",
        "outline",
        "transparent",
        "background",
        "item",
    }
    allowed.update(record.open_name.split())
    allowed.update(record.base_object.split("_"))
    attributes = record.attributes
    for group in (
        attributes.colors,
        attributes.materials,
        attributes.shapes,
        attributes.effects,
        attributes.state,
        attributes.function,
        attributes.mood,
        attributes.parts,
    ):
        for value in group:
            allowed.update(value.split("_"))
    for caption in record.captions:
        for word in caption.replace(",", " ").lower().split():
            assert word in allowed, f"ungrounded word {word!r} in caption {caption!r}"


def test_captions_never_contain_forbidden_content() -> None:
    for name, category in (("ruby_gem", "material"), ("shadow", "effect_icon"), ("sword", "weapon")):
        record = build_semantic_v3_record(_prediction(name, category))
        for caption in record.captions:
            lowered = caption.lower()
            assert "photorealistic" not in lowered
            assert "watermark" not in lowered


def test_aliases_come_from_vetted_vlm_alternatives_and_name_tokens() -> None:
    record = build_semantic_v3_record(
        _prediction(
            "chestplate",
            "armor",
            candidates=["chestplate", "breastplate", "cuirass"],
            vlm_alternatives=["breastplate", "cuirass", "spaceship"],
        )
    )
    assert "breastplate" in record.aliases
    assert "cuirass" in record.aliases
    # unvetted VLM inventions never become aliases
    assert "spaceship" not in record.aliases
    ruby = build_semantic_v3_record(_prediction("ruby_gem", "material"))
    assert "ruby" in ruby.aliases


def test_expected_color_warning_for_colorless_gem() -> None:
    record = build_semantic_v3_record(_prediction("gem", "material"))
    assert "no_color_information" in record.warnings


def test_json_round_trip_preserves_record() -> None:
    record = build_semantic_v3_record(_prediction("electric_arrow", "effect_icon", dominant_colors=["yellow"]))
    data = semantic_v3_to_json(record)
    parsed = semantic_v3_from_json(data)
    assert parsed is not None
    assert parsed.base_object == record.base_object
    assert parsed.open_name == record.open_name
    assert parsed.attributes == record.attributes
    assert parsed.captions == record.captions
    assert parsed.negative_tags == record.negative_tags


def test_creative_concept_charged_sinew() -> None:
    concept = parse_creative_concept("charged_sinew")
    assert concept.recognized
    assert concept.base_object == "sinew"
    assert concept.modifiers == ("charged",)
    assert "electric" in concept.attributes.effects
    assert "charged" in concept.attributes.effects
    assert "organic" in concept.attributes.materials
    assert "fibrous" in concept.attributes.shapes


def test_creative_concept_calming_spores() -> None:
    concept = parse_creative_concept("calming_spores")
    assert concept.recognized
    assert concept.base_object == "spores"
    assert "calming" in concept.attributes.effects
    assert "soft" in concept.attributes.mood
    assert "small_dots" in concept.attributes.shapes


def test_creative_concept_square_gem_composes_shape_and_base() -> None:
    concept = parse_creative_concept("square_gem")
    assert concept.base_object == "gem"
    assert "square" in concept.attributes.shapes
    assert "crystal" in concept.attributes.materials
