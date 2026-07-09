from _harvest_testdata import make_sprite_png
from spritelab.harvest.filename_rules_v2 import suggest_from_filename_v2
from spritelab.harvest.label_candidates import specialize_496_rpg_object
from spritelab.harvest.label_fusion_v2 import FusionThresholds
from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.label_v2_pipeline import build_label_v2_record


def _record(filename: str, source_id: str = "oga_496_rpg_icons_32fix") -> dict[str, str]:
    return {
        "sprite_id": filename.removesuffix(".png").lower(),
        "filename": filename,
        "relative_path": filename,
        "source_id": source_id,
        "source_name": source_id,
    }


def _row(tmp_path, filename: str, vlm: LabelSuggestion) -> dict:
    run = tmp_path / "run"
    png = make_sprite_png(run / filename, empty=True)
    return build_label_v2_record(
        _record(filename) | {"final_png_path": str(png)},
        run_dir=run,
        vlm=vlm,
        thresholds=FusionThresholds(),
    )


def _safe(row: dict) -> tuple[str, str]:
    safe = row["safe_prefill"]
    return safe["category"], safe["object_name"]


def test_496_category_overrides_for_materials_and_mushroom(tmp_path) -> None:
    gem = _row(
        tmp_path,
        "I_Agate.png",
        LabelSuggestion(
            "material",
            "agate",
            tags=("gem",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    fabric = _row(
        tmp_path,
        "I_Fabric.png",
        LabelSuggestion(
            "material",
            "fabric",
            tags=("cloth",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    mushroom = _row(
        tmp_path,
        "I_C_Mushroom.png",
        LabelSuggestion(
            "plant",
            "mushroom",
            tags=("mushroom",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(gem)[0] == "material"
    assert _safe(gem)[1] in {"agate", "agate_gem"}
    assert _safe(fabric) == ("material", "fabric")
    assert _safe(mushroom) == ("item_icon", "mushroom")


def test_496_shield_shape_overrides_material_family(tmp_path) -> None:
    wood = _row(
        tmp_path,
        "E_Wood03.png",
        LabelSuggestion(
            "material",
            "wood",
            tags=("bordered", "wood_grain", "shield_shape"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    metal = _row(
        tmp_path,
        "E_Metal09.png",
        LabelSuggestion(
            "material",
            "metal",
            tags=("shield_shape", "metallic"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(wood) == ("armor", "wooden_shield")
    assert _safe(metal)[0] == "armor"
    assert _safe(metal)[1] in {"shield", "metal_shield"}


def test_496_armor_and_clothing_specificity_uses_vlm_evidence(tmp_path) -> None:
    chestplate = _row(
        tmp_path,
        "A_Armor05.png",
        LabelSuggestion(
            "armor",
            "armor",
            tags=("chest_shape", "shoulder_guards"),
            alternative_object_names=("chestplate", "breastplate"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    golden = _row(
        tmp_path,
        "A_Armour03.png",
        LabelSuggestion(
            "armor",
            "armor",
            tags=("gold", "yellow", "torso_shape"),
            alternative_object_names=("chestplate", "breastplate"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    leather = _row(
        tmp_path,
        "A_Armour01.png",
        LabelSuggestion(
            "armor",
            "leather_armor",
            tags=("brown", "tan", "torso_covering"),
            alternative_object_names=("chestplate", "breastplate"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    tunic = _row(
        tmp_path,
        "A_Clothing02.png",
        LabelSuggestion(
            "armor",
            "clothing",
            tags=("v_neck", "short_sleeve"),
            alternative_object_names=("tunic",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    broad = _row(
        tmp_path,
        "A_Armor05.png",
        LabelSuggestion("armor", "armor", confidence=0.85, source="vlm_descriptor", source_consistency="consistent"),
    )

    assert _safe(chestplate) == ("armor", "chestplate")
    assert _safe(golden) == ("armor", "golden_chestplate")
    assert _safe(leather) == ("armor", "leather_chestplate")
    assert _safe(tunic) == ("armor", "tunic")
    assert _safe(broad) == ("armor", "armor")


def test_496_potion_color_and_container_specificity(tmp_path) -> None:
    red = _row(
        tmp_path,
        "P_Red03.png",
        LabelSuggestion(
            "item_icon",
            "potion",
            tags=("bottle", "cork", "red_liquid"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    blue = _row(
        tmp_path,
        "P_Blue05.png",
        LabelSuggestion(
            "item_icon",
            "potion",
            tags=("bottle", "liquid", "blue"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    green = _row(
        tmp_path,
        "P_Green06.png",
        LabelSuggestion(
            "item_icon",
            "potion",
            tags=("bottle", "liquid", "green"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    pink = _row(
        tmp_path,
        "P_Pink04.png",
        LabelSuggestion(
            "item_icon",
            "potion",
            tags=("vial", "diagonal", "pink", "liquid"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    medicine = _row(
        tmp_path,
        "P_Medicine06.png",
        LabelSuggestion(
            "item_icon",
            "medicine",
            tags=("bottle", "cork", "liquid"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(red) == ("item_icon", "red_potion")
    assert _safe(blue) == ("item_icon", "blue_potion")
    assert _safe(green) == ("item_icon", "green_potion")
    assert _safe(pink) == ("item_icon", "pink_vial")
    assert _safe(medicine) == ("item_icon", "medicine_bottle")
    assert _safe(red)[1] != "red"


def test_496_food_preparation_specificity(tmp_path) -> None:
    fish = _row(
        tmp_path,
        "I_C_Fish.png",
        LabelSuggestion(
            "item_icon",
            "fish",
            tags=("food",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    raw_fish = _row(
        tmp_path,
        "I_C_RawFish.png",
        LabelSuggestion(
            "item_icon",
            "raw_fish",
            tags=("fish",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    pie = _row(
        tmp_path,
        "I_C_Pie.png",
        LabelSuggestion(
            "item_icon",
            "pie",
            tags=("slice", "baked_food"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    watermelon = _row(
        tmp_path,
        "I_C_Watermellon.png",
        LabelSuggestion(
            "item_icon",
            "watermelon",
            tags=("slice", "red", "green"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    meat = _row(
        tmp_path,
        "I_C_RawMeat.png",
        LabelSuggestion(
            "item_icon",
            "meat",
            tags=("meat",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(fish) == ("item_icon", "fish_skewer")
    assert _safe(raw_fish) == ("item_icon", "raw_fish_skewer")
    assert _safe(pie) == ("item_icon", "pie_slice")
    assert _safe(watermelon) == ("item_icon", "watermelon_slice")
    assert _safe(meat) == ("item_icon", "raw_meat")


def test_496_effect_and_weapon_specificity(tmp_path) -> None:
    cases = {
        "S_Bow03.png": "electric_arrow",
        "S_Bow11.png": "multiple_arrows",
        "S_Dagger02.png": "electric_dagger",
        "S_Sword07.png": "bleeding_sword",
        "S_Water03.png": "water_wave",
        "W_Gold_Mace.png": "golden_mace",
        "W_Gold_Sword.png": "golden_sword",
        "W_Throw004.png": "throwing_star",
    }
    for filename, expected in cases.items():
        category = "weapon" if filename.startswith("W_") else "effect_icon"
        family = (
            "gold"
            if filename.startswith("W_Gold")
            else filename.split("_", 1)[1].split(".", 1)[0].rstrip("0123456789").lower()
        )
        row = _row(
            tmp_path,
            filename,
            LabelSuggestion(
                category,
                family,
                tags=("weapon_shape", "curved", "metallic"),
                confidence=0.85,
                source="vlm_descriptor",
                source_consistency="consistent",
            ),
        )
        assert row["safe_prefill"]["object_name"] == expected


def test_496_bow_explosive_arrow_requires_explicit_explosion_evidence(tmp_path) -> None:
    weak = _row(
        tmp_path,
        "S_Bow13.png",
        LabelSuggestion(
            "weapon",
            "bow",
            tags=("curved", "wooden", "string", "arrow_nock", "yellow", "orange", "weapon_shape"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    strong = _row(
        tmp_path,
        "S_Bow06.png",
        LabelSuggestion(
            "weapon",
            "bow",
            tags=("curved", "weapon_shape", "arrow", "explosion", "blast"),
            short_description="A bow firing an explosive arrow with a blast effect.",
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert weak["safe_prefill"]["object_name"] == "bow"
    assert _safe(strong) == ("effect_icon", "explosive_arrow")


def test_496_wizard_hat_promotion_requires_strong_wizard_evidence(tmp_path) -> None:
    generic = _row(
        tmp_path,
        "C_Hat01.png",
        LabelSuggestion(
            "armor",
            "hat",
            tags=("conical", "wide_brim", "headgear"),
            alternative_object_names=("wizard_hat",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    wizard = _row(
        tmp_path,
        "C_Hat01.png",
        LabelSuggestion(
            "armor",
            "hat",
            tags=("pointed", "magic", "headgear"),
            alternative_object_names=("wizard_hat",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(generic) == ("armor", "hat")
    assert _safe(wizard) == ("armor", "wizard_hat")


def test_496_golden_chestplate_requires_gold_not_red_orange_highlights(tmp_path) -> None:
    red_orange = _row(
        tmp_path,
        "A_Armor05.png",
        LabelSuggestion(
            "armor",
            "armor",
            tags=("chest_shape", "shoulder_guards", "red", "orange", "gold"),
            alternative_object_names=("chestplate", "breastplate"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    yellow_gold = _row(
        tmp_path,
        "A_Armor05.png",
        LabelSuggestion(
            "armor",
            "armor",
            tags=("chest_shape", "shoulder_guards", "yellow", "gold"),
            alternative_object_names=("chestplate", "breastplate"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(red_orange) == ("armor", "chestplate")
    assert _safe(yellow_gold) == ("armor", "golden_chestplate")


def test_496_potion_shape_and_visual_color_cleanup(tmp_path) -> None:
    compact = _row(
        tmp_path,
        "P_Pink04.png",
        LabelSuggestion(
            "item_icon",
            "potion",
            tags=("compact", "diagonal", "pink", "liquid"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    tall = _row(
        tmp_path,
        "P_Yellow07.png",
        LabelSuggestion(
            "item_icon",
            "potion",
            tags=("tall", "vertical", "yellow", "liquid", "potion"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    purple = _row(
        tmp_path,
        "P_Pink08.png",
        LabelSuggestion(
            "item_icon",
            "potion",
            tags=("narrow", "purple", "liquid", "bottle"),
            dominant_colors=("purple",),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(compact) == ("item_icon", "pink_vial")
    assert _safe(tall) == ("item_icon", "yellow_potion")
    assert _safe(purple) == ("item_icon", "purple_vial")


def test_496_safe_broad_to_specific_compounds(tmp_path) -> None:
    fire_cannon = _row(
        tmp_path,
        "I_Cannon02.png",
        LabelSuggestion(
            "item_icon",
            "cannon",
            tags=("orange", "blast", "barrel_shaped"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    yellow_cannon = _row(
        tmp_path,
        "I_Cannon04.png",
        LabelSuggestion(
            "item_icon",
            "cannon",
            tags=("yellow", "barrel_shaped"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    fire_feather = _row(
        tmp_path,
        "I_Feather02.png",
        LabelSuggestion(
            "item_icon",
            "feather",
            tags=("orange", "flame", "feather"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )
    yellow_ink = _row(
        tmp_path,
        "I_Ink.png",
        LabelSuggestion(
            "item_icon",
            "ink",
            tags=("yellow", "container_like", "liquid"),
            confidence=0.85,
            source="vlm_descriptor",
            source_consistency="consistent",
        ),
    )

    assert _safe(fire_cannon) == ("item_icon", "fire_cannon")
    assert _safe(yellow_cannon) == ("item_icon", "yellow_cannon")
    assert _safe(fire_feather) == ("item_icon", "fire_feather")
    assert _safe(yellow_ink) == ("item_icon", "yellow_ink_bucket")


def test_496_specialization_is_profile_gated() -> None:
    record = _record("P_Red03.png", source_id="generic_source")
    filename = suggest_from_filename_v2(record)
    result = specialize_496_rpg_object(
        record,
        filename.profile,
        filename.suggestion,
        parsed_tokens=filename.parsed_tokens,
        candidate_object_names=("red_potion",),
        vlm=LabelSuggestion("item_icon", "potion", tags=("bottle",), confidence=0.85, source="vlm_descriptor"),
    )

    assert result.object_name == ""
    assert result.category == ""
    assert result.flags == ()

    cannon_record = _record("I_Cannon02.png", source_id="generic_source")
    cannon_filename = suggest_from_filename_v2(cannon_record)
    cannon_result = specialize_496_rpg_object(
        cannon_record,
        cannon_filename.profile,
        cannon_filename.suggestion,
        parsed_tokens=cannon_filename.parsed_tokens,
        candidate_object_names=("fire_cannon",),
        vlm=LabelSuggestion("item_icon", "cannon", tags=("orange", "blast"), confidence=0.85, source="vlm_descriptor"),
    )

    assert cannon_result.object_name == ""
    assert cannon_result.category == ""
    assert cannon_result.flags == ()
