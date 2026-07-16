from __future__ import annotations

from spritelab.training.tokenization import SpriteTextTokenizer, semantic_strings_from_record, tokenize_text


def test_tokenizes_snake_case_and_prose() -> None:
    assert tokenize_text("Blue_crystal-hammer, 32x32 RPG icon!") == [
        "blue",
        "crystal",
        "hammer",
        "32x32",
        "rpg",
        "icon",
    ]


def test_builds_deterministic_vocabulary() -> None:
    texts = ["red potion", "blue_potion", "red sword"]
    a = SpriteTextTokenizer.build(texts)
    b = SpriteTextTokenizer.build(reversed(texts))
    assert a.token_to_id == b.token_to_id
    assert a.token_to_id["red"] < a.token_to_id["blue"]


def test_encodes_decodes_special_tokens_and_unknowns() -> None:
    tokenizer = SpriteTextTokenizer.build(["red potion"], max_length=6)
    encoded = tokenizer.encode("red moonlit")
    assert encoded[0] == tokenizer.bos_id
    assert encoded[1] == tokenizer.token_to_id["red"]
    assert encoded[2] == tokenizer.unk_id
    assert encoded[3] == tokenizer.eos_id
    assert encoded[-1] == tokenizer.pad_id
    assert tokenizer.decode(encoded) == "red"


def test_saves_and_loads_vocabulary(tmp_path) -> None:
    tokenizer = SpriteTextTokenizer.build(["golden sword", "wooden shield"], max_length=8)
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    loaded = SpriteTextTokenizer.load(path)
    assert loaded.token_to_id == tokenizer.token_to_id
    assert loaded.max_length == 8


def test_semantic_strings_include_nested_conditioning() -> None:
    record = {
        "category": "weapon",
        "object_name": "golden_sword",
        "base_object": "sword",
        "caption_type": "attribute",
        "conditioning": {
            "semantic_v3": {
                "open_name": "golden sword",
                "attributes": {
                    "colors": ["gold"],
                    "materials": ["metal"],
                    "shapes": [],
                    "effects": [],
                    "state": [],
                    "function": ["attack"],
                },
            },
            "kept_attributes": {"colors": ["gold"]},
            "dropped_attributes": {"materials": ["metal"]},
            "dropout_ops": ["drop_material"],
        },
    }
    text = " ".join(semantic_strings_from_record(record))
    assert "golden_sword" in text
    assert "drop_material" in text
    assert "kept_attributes" in text
