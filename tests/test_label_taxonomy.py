from spritelab.harvest.label_taxonomy import (
    canonicalize_object_name,
    normalize_category,
    normalize_tags,
    object_name_token_f1,
)


def test_typos_and_plurals_canonicalize() -> None:
    assert canonicalize_object_name("saphire") == "sapphire"
    assert canonicalize_object_name("ovale") == "oval"
    assert canonicalize_object_name("cherries") == "cherry"
    assert canonicalize_object_name("blueberries") == "blueberry"
    assert canonicalize_object_name("grapes") == "grape"


def test_invalid_category_and_tag_dedupe() -> None:
    assert normalize_category("not-a-category") == "unknown"
    assert normalize_tags(["Dark Blue", "dark_blue", " watermellon "]) == ("dark_blue", "watermelon")


def test_object_name_token_f1() -> None:
    assert object_name_token_f1("cheese_wedge", "cheese") > 0.0
    assert object_name_token_f1("orange", "coin") == 0.0
