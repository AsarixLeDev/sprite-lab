from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2


def _record(name: str, source_id: str = "oga_cc0_food_ocal") -> dict[str, str]:
    return {"sprite_id": name.removesuffix(".png"), "relative_path": name, "source_id": source_id, "source_name": source_id}


def test_food_bad_examples_are_high_confidence_filename_labels() -> None:
    expected = {
        "butter.png": "butter",
        "cheese_wedge.png": "cheese_wedge",
        "cheese_wheel.png": "cheese_wheel",
        "milk_carton.png": "milk_carton",
        "orange.png": "orange",
        "kiwi.png": "kiwi",
    }
    for filename, object_name in expected.items():
        result = suggest_from_filename_v2(_record(filename))
        assert result.suggestion.object_name == object_name
        assert result.suggestion.confidence >= 0.9
        assert result.suggestion.category == "item_icon"


def test_food_specific_tags_and_containers() -> None:
    cheese = suggest_from_filename_v2(_record("cheese_wedge.png")).suggestion
    assert {"cheese", "dairy"} <= set(cheese.tags)
    milk = suggest_from_filename_v2(_record("milk_carton.png")).suggestion
    assert {"milk", "dairy", "container"} <= set(milk.tags)
    orange = suggest_from_filename_v2(_record("orange.png")).suggestion
    assert {"fruit", "citrus"} <= set(orange.tags)
    lemon = suggest_from_filename_v2(_record("lemon.png")).suggestion
    assert {"fruit", "citrus"} <= set(lemon.tags)
    soda = suggest_from_filename_v2(_record("soda_can_apple.png")).suggestion
    assert soda.object_name in {"soda_can_apple", "apple_soda_can"}
    assert soda.object_name != "apple"
    juice = suggest_from_filename_v2(_record("juice_orange.png")).suggestion
    assert juice.object_name == "orange_juice"


def test_tool_gem_and_rpg_profiles() -> None:
    compass = suggest_from_filename_v2(_record("compass_02.png", "oga_cc0_tool_ocal")).suggestion
    assert compass.category == "tool"
    ruler = suggest_from_filename_v2(_record("ruler_triangle_01.png", "oga_cc0_tool_ocal")).suggestion
    assert ruler.category == "tool"
    assert ruler.object_name == "ruler_triangle"
    ruby = suggest_from_filename_v2(_record("ruby.png", "oga_cc0_gem_7soul1")).suggestion
    assert ruby.category == "material"
    assert ruby.object_name == "ruby"
    sapphire = suggest_from_filename_v2(_record("saphire_gem.png", "oga_cc0_gem_7soul1")).suggestion
    assert sapphire.object_name == "sapphire_gem"
    axe = suggest_from_filename_v2(_record("w_axe014.png", "oga_496_rpg_icons_32fix")).suggestion
    assert axe.category == "weapon"
    assert "axe" in axe.object_name
    poison = suggest_from_filename_v2(_record("s_poison01.png", "oga_496_rpg_icons_32fix")).suggestion
    assert poison.category == "effect_icon"
    assert "poison" in poison.object_name
