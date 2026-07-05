from pathlib import Path

from spritelab.harvest.assisted_golden import AssistedGoldenCandidate, AssistedGoldenLabel
from spritelab.harvest.assisted_golden_gui import candidate_gui_model


def test_gui_candidate_model_exposes_editable_fields_and_label_v2_context() -> None:
    candidate = AssistedGoldenCandidate(
        sprite_id="armor_05",
        final_png_path=Path("A_Armor05.png"),
        suggested_category="armor",
        suggested_object_name="armor",
        suggested_tags=("armor", "wearable"),
        suggested_description="A gray armor icon.",
        prefill_source="label_v2",
        prefill_category="armor",
        prefill_object_name="armor",
        prefill_tags=("armor", "wearable"),
        prefill_short_description="A gray armor icon.",
        prefill_materials=("metal",),
        prefill_mood=("defensive",),
        candidate_object_names=("armor", "chestplate", "breastplate"),
        alternative_object_names=("chestplate", "breastplate"),
        vlm_object_name="armor",
        vlm_short_description="Looks like a chestplate.",
        vlm_source_consistency="consistent",
        visual_facts={"dominant_colors": ["gray"], "content_width": 31},
        gold_category="armor",
        gold_object_name="armor",
        gold_tags=("armor", "wearable"),
        gold_short_description="A gray armor icon.",
        gold_materials=("metal",),
        gold_mood=("defensive",),
    )

    model = candidate_gui_model(candidate)

    assert model["editable"]["category"] == "armor"
    assert model["editable"]["object_name"] == "armor"
    assert model["editable"]["tags"] == ["armor", "wearable"]
    assert model["editable"]["short_description"] == "A gray armor icon."
    assert model["editable"]["materials"] == ["metal"]
    assert model["editable"]["mood"] == ["defensive"]
    assert "chestplate" in model["object_name_choices"]
    assert model["reference"]["candidate_object_names"] == ["armor", "chestplate", "breastplate"]
    assert model["reference"]["alternative_object_names"] == ["chestplate", "breastplate"]
    assert model["reference"]["vlm_short_description"] == "Looks like a chestplate."
    assert model["reference"]["visual_facts"]["dominant_colors"] == ["gray"]


def test_gui_candidate_model_prefers_existing_human_label_for_editing() -> None:
    candidate = AssistedGoldenCandidate(
        sprite_id="armor_05",
        final_png_path=Path("A_Armor05.png"),
        suggested_category="armor",
        suggested_object_name="armor",
        suggested_tags=("armor",),
        prefill_source="label_v2",
        prefill_category="armor",
        prefill_object_name="armor",
        prefill_tags=("armor",),
        candidate_object_names=("armor", "chestplate"),
    )
    label = AssistedGoldenLabel(
        sprite_id="armor_05",
        category="armor",
        object_name="chestplate",
        tags=("armor", "metal"),
        short_description="Reviewed chestplate.",
        materials=("metal",),
    )

    model = candidate_gui_model(candidate, label)

    assert model["editable"]["object_name"] == "chestplate"
    assert model["editable"]["short_description"] == "Reviewed chestplate."
    assert model["reference"]["prefill"]["object_name"] == "armor"
