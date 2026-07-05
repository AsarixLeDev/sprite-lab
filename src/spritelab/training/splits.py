"""Deterministic group-aware train/val/test splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DEFAULT_SPLIT_SEED = 1337


@dataclass(frozen=True)
class SplitAssignment:
    """Sprite IDs assigned to train/val/test with duplicate group metadata."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    group_by_sprite_id: dict[str, str]


def make_group_aware_split(
    sprite_ids: Sequence[str],
    group_by_sprite_id: Mapping[str, str] | None = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = DEFAULT_SPLIT_SEED,
) -> SplitAssignment:
    """Assign sprites to splits while keeping duplicate groups together."""

    _validate_fractions(train_fraction, val_fraction, test_fraction)
    ordered_sprite_ids = tuple(sorted(dict.fromkeys(str(sprite_id) for sprite_id in sprite_ids)))
    if not ordered_sprite_ids:
        return SplitAssignment(train=(), val=(), test=(), group_by_sprite_id={})

    group_by_id = {
        sprite_id: str(group_by_sprite_id.get(sprite_id, sprite_id)) if group_by_sprite_id else sprite_id
        for sprite_id in ordered_sprite_ids
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for sprite_id in ordered_sprite_ids:
        groups[group_by_id[sprite_id]].append(sprite_id)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    train_groups, val_groups, test_groups = _assign_groups(
        group_ids,
        groups,
        total_count=len(ordered_sprite_ids),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    def flatten(group_keys: list[str]) -> tuple[str, ...]:
        values: list[str] = []
        for group_key in group_keys:
            values.extend(groups[group_key])
        return tuple(sorted(values))

    return SplitAssignment(
        train=flatten(train_groups),
        val=flatten(val_groups),
        test=flatten(test_groups),
        group_by_sprite_id=group_by_id,
    )


def _validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(value < 0.0 for value in fractions):
        raise ValueError("split fractions must be non-negative.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.")


def _assign_groups(
    group_ids: list[str],
    groups: Mapping[str, Sequence[str]],
    *,
    total_count: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    if len(group_ids) == 1:
        return [group_ids[0]], [], []
    if len(group_ids) == 2:
        return [group_ids[0]], [group_ids[1]], []

    target_train = int(round(total_count * train_fraction))
    target_val = int(round(total_count * val_fraction))
    target_test = total_count - target_train - target_val

    target_train = min(max(1, target_train), total_count - 2)
    target_val = min(max(1, target_val), total_count - target_train - 1)
    target_test = max(1, total_count - target_train - target_val)

    targets = {"train": target_train, "val": target_val, "test": target_test}
    split_groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for group_id in group_ids:
        group_size = len(groups[group_id])
        split_name = _best_split_for_group(split_counts, targets, group_size)
        split_groups[split_name].append(group_id)
        split_counts[split_name] += group_size

    return split_groups["train"], split_groups["val"], split_groups["test"]


def _best_split_for_group(
    split_counts: Mapping[str, int],
    targets: Mapping[str, int],
    group_size: int,
) -> str:
    order = ("train", "val", "test")

    def key(split_name: str) -> tuple[float, int]:
        target = max(1, targets[split_name])
        projected = split_counts[split_name] + group_size
        deficit = (target - split_counts[split_name]) / target
        overshoot = max(0, projected - target)
        return (deficit, -overshoot)

    return max(order, key=key)
