from __future__ import annotations

import pytest

from spritelab.training.splits import make_group_aware_split


def test_every_sprite_appears_exactly_once() -> None:
    sprite_ids = [f"sprite_{index}" for index in range(12)]

    split = make_group_aware_split(sprite_ids, seed=1)

    assigned = [*split.train, *split.val, *split.test]
    assert sorted(assigned) == sorted(sprite_ids)
    assert len(assigned) == len(set(assigned))


def test_split_is_deterministic_for_fixed_seed() -> None:
    sprite_ids = [f"sprite_{index}" for index in range(20)]

    assert make_group_aware_split(sprite_ids, seed=123) == make_group_aware_split(sprite_ids, seed=123)


def test_different_seed_can_change_assignment() -> None:
    sprite_ids = [f"sprite_{index}" for index in range(30)]

    left = make_group_aware_split(sprite_ids, seed=1)
    right = make_group_aware_split(sprite_ids, seed=2)

    assert (left.train, left.val, left.test) != (right.train, right.val, right.test)


def test_sprites_in_same_group_stay_in_same_split() -> None:
    sprite_ids = ["a", "b", "c", "d", "e", "f"]
    groups = {"a": "dupe_1", "b": "dupe_1"}

    split = make_group_aware_split(sprite_ids, group_by_sprite_id=groups, seed=5)
    lookup = dict.fromkeys(split.train, "train")
    lookup.update(dict.fromkeys(split.val, "val"))
    lookup.update(dict.fromkeys(split.test, "test"))

    assert lookup["a"] == lookup["b"]


def test_fractions_approximately_work_for_normal_datasets() -> None:
    sprite_ids = [f"sprite_{index:02d}" for index in range(50)]

    split = make_group_aware_split(sprite_ids, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1)

    assert 35 <= len(split.train) <= 45
    assert 3 <= len(split.val) <= 7
    assert 3 <= len(split.test) <= 7


def test_fewer_than_three_groups_is_handled() -> None:
    split = make_group_aware_split(["a", "b"], seed=1)

    assert sorted([*split.train, *split.val, *split.test]) == ["a", "b"]


def test_invalid_fractions_raise() -> None:
    with pytest.raises(ValueError, match="must equal 1"):
        make_group_aware_split(["a"], train_fraction=0.8, val_fraction=0.2, test_fraction=0.2)
