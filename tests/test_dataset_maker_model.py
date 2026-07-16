from __future__ import annotations

from pathlib import Path

from spritelab.dataset_maker.model import (
    DatasetMakerItem,
    normalize_category,
    normalize_sprite_id,
    normalize_tag,
    validate_dataset_maker_item,
)


def test_tag_normalization_works() -> None:
    assert normalize_tag(" Copper Vial ") == "copper_vial"
    assert normalize_tag("Magic/Fire") == "magic_fire"


def test_category_normalization_works() -> None:
    assert normalize_category("Item Icon") == "item_icon"
    assert normalize_category("") == "unknown"


def test_sprite_id_normalization_works() -> None:
    assert normalize_sprite_id(" Copper Vial 001!.png ") == "copper_vial_001_.png"


def test_invalid_empty_sprite_id_is_rejected() -> None:
    item = DatasetMakerItem(sprite_id="!!!", source_path=Path("x.png"), status="accepted")

    assert any("sprite_id" in error for error in validate_dataset_maker_item(item))


def test_invalid_status_is_rejected() -> None:
    item = DatasetMakerItem(sprite_id="sprite", source_path=Path("x.png"), status="ready")

    assert any("status" in error for error in validate_dataset_maker_item(item))


def test_invalid_split_is_rejected() -> None:
    item = DatasetMakerItem(sprite_id="sprite", source_path=Path("x.png"), status="accepted", split="dev")

    assert any("split" in error for error in validate_dataset_maker_item(item))


def test_tags_deduplicate_while_preserving_order() -> None:
    item = DatasetMakerItem(
        sprite_id="sprite",
        source_path=Path("x.png"),
        status="accepted",
        tags=("Copper", "Vial", "copper", "Blue Gem", "blue_gem"),
    )

    assert item.tags == ("copper", "vial", "blue_gem")
