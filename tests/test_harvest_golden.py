"""Tests for golden-set sampling and label storage."""

from __future__ import annotations

from spritelab.harvest.golden import (
    GoldenLabel,
    append_golden_label,
    golden_label_from_dict,
    golden_label_to_dict,
    load_golden_labels,
    sample_golden_candidates,
)


def _records(counts: dict[str, int]) -> list[dict]:
    records = []
    for source, count in counts.items():
        for index in range(count):
            records.append(
                {
                    "sprite_id": f"{source}_sprite_{index:03d}",
                    "source_name": source,
                    "relative_path": f"{source}/tile_{index:03d}.png",
                    "final_png_path": f"harvest_runs/{source}/final/tile_{index:03d}.png",
                }
            )
    return records


def test_sample_is_deterministic() -> None:
    records = _records({"pack_a": 50, "pack_b": 30})
    first = sample_golden_candidates(records, 20, seed=7)
    second = sample_golden_candidates(records, 20, seed=7)
    assert first == second
    assert len(first) == 20


def test_sample_changes_with_seed() -> None:
    records = _records({"pack_a": 200})
    assert sample_golden_candidates(records, 20, seed=1) != sample_golden_candidates(records, 20, seed=2)


def test_sample_stratifies_proportionally() -> None:
    records = _records({"pack_a": 90, "pack_b": 10})
    sample = sample_golden_candidates(records, 10, seed=0)
    by_source = {"pack_a": 0, "pack_b": 0}
    for record in sample:
        by_source[record["strata"][0]] += 1
    assert by_source["pack_a"] == 9
    assert by_source["pack_b"] == 1


def test_sample_represents_small_strata() -> None:
    records = _records({"pack_a": 500, "pack_b": 2})
    sample = sample_golden_candidates(records, 10, seed=0)
    sources = {record["strata"][0] for record in sample}
    assert "pack_b" in sources


def test_sample_clamps_to_population() -> None:
    records = _records({"pack_a": 5})
    assert len(sample_golden_candidates(records, 100, seed=0)) == 5
    assert sample_golden_candidates(records, 0, seed=0) == []
    assert sample_golden_candidates([], 10, seed=0) == []


def test_golden_label_normalization() -> None:
    label = GoldenLabel(
        sprite_id="Pack_A/Sprite 1",
        category="Item_Icon",
        object_name="Red Potion",
        tags=("Potion", "potion", "LIQUID", ""),
    )
    assert label.sprite_id == "pack_a_sprite_1"
    assert label.category == "item_icon"
    assert label.object_name == "red_potion"
    assert label.tags == ("potion", "liquid")


def test_append_and_load_last_write_wins(tmp_path) -> None:
    path = tmp_path / "golden_labels.jsonl"
    append_golden_label(path, GoldenLabel(sprite_id="a", category="plant"))
    append_golden_label(path, GoldenLabel(sprite_id="b", category="weapon"))
    append_golden_label(path, GoldenLabel(sprite_id="a", category="item_icon", object_name="apple"))

    labels = load_golden_labels(path)
    assert set(labels) == {"a", "b"}
    assert labels["a"].category == "item_icon"
    assert labels["a"].object_name == "apple"
    assert labels["a"].labeled_at  # stamped on append
    assert labels["b"].category == "weapon"


def test_load_missing_and_corrupt_lines(tmp_path) -> None:
    assert load_golden_labels(tmp_path / "missing.jsonl") == {}
    path = tmp_path / "golden_labels.jsonl"
    path.write_text('not json\n{"sprite_id": "ok", "category": "plant"}\n[1,2]\n', encoding="utf-8")
    labels = load_golden_labels(path)
    assert set(labels) == {"ok"}


def test_label_dict_roundtrip() -> None:
    label = GoldenLabel(sprite_id="a", category="plant", object_name="fern", tags=("fern",), labeler="m")
    assert golden_label_from_dict(golden_label_to_dict(label)) == label
