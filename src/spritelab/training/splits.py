"""Deterministic group-aware train/val/test splitting."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
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


@dataclass(frozen=True)
class GroupSplitRecord:
    """Metadata required for deterministic, leakage-safe group assignment."""

    sprite_id: str
    group_id: str
    category: str = "unknown"
    object_name: str = ""
    source_pack: str = ""
    split_override: str | None = None


@dataclass(frozen=True)
class BalancedSplitAssignment:
    split_by_sprite_id: dict[str, str]
    group_by_sprite_id: dict[str, str]
    group_overrides: dict[str, str]


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


def make_balanced_group_aware_split(
    records: Sequence[GroupSplitRecord],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = DEFAULT_SPLIT_SEED,
) -> BalancedSplitAssignment:
    """Assign indivisible variant groups with deterministic soft balancing.

    This deliberately keeps group integrity hard while treating row, category,
    object, and source-pack targets as soft objectives. One/two-example
    objects are anchored to train unless a compatible manual override reserves
    their group.
    """

    _validate_fractions(train_fraction, val_fraction, test_fraction)
    ordered = sorted(records, key=lambda record: record.sprite_id)
    groups: dict[str, list[GroupSplitRecord]] = defaultdict(list)
    for record in ordered:
        groups[str(record.group_id or record.sprite_id)].append(record)
    overrides: dict[str, str] = {}
    for group_id, members in groups.items():
        values = {str(member.split_override) for member in members if member.split_override is not None}
        invalid = values - {"train", "val", "test"}
        if invalid:
            raise ValueError(f"{group_id}: invalid manual split override(s) {sorted(invalid)}")
        if len(values) > 1:
            raise ValueError(f"manual split overrides divide leakage group {group_id}: {sorted(values)}")
        if values:
            overrides[group_id] = next(iter(values))

    total = len(ordered)
    if not total:
        return BalancedSplitAssignment({}, {}, {})
    targets = {"train": total * train_fraction, "val": total * val_fraction, "test": total * test_fraction}
    object_counts = Counter(record.object_name for record in ordered if record.object_name)
    split_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_split: dict[str, str] = {}

    def assign(group_id: str, split: str) -> None:
        group_split[group_id] = split
        for member in groups[group_id]:
            split_counts[split] += 1
            category_counts[member.category][split] += 1
            source_counts[member.source_pack][split] += 1

    for group_id in sorted(overrides):
        assign(group_id, overrides[group_id])

    rng = random.Random(seed)
    remaining = [group_id for group_id in sorted(groups) if group_id not in group_split]
    rng.shuffle(remaining)
    # Large groups first avoids end-stage target overshoot; seeded shuffle
    # supplies deterministic tie-breaking independent of manifest order.
    remaining.sort(key=lambda group_id: -len(groups[group_id]))
    for group_id in remaining:
        members = groups[group_id]
        rare_objects = {
            member.object_name for member in members if member.object_name and object_counts[member.object_name] <= 2
        }
        if rare_objects:
            assign(group_id, "train")
            continue
        split = _best_balanced_split(members, split_counts, category_counts, source_counts, targets)
        assign(group_id, split)

    split_by_sprite_id = {
        member.sprite_id: group_split[group_id] for group_id, members in groups.items() for member in members
    }
    group_by_sprite_id = {member.sprite_id: group_id for group_id, members in groups.items() for member in members}
    return BalancedSplitAssignment(split_by_sprite_id, group_by_sprite_id, overrides)


def _best_balanced_split(
    members: Sequence[GroupSplitRecord],
    split_counts: Mapping[str, int],
    category_counts: Mapping[str, Mapping[str, int]],
    source_counts: Mapping[str, Mapping[str, int]],
    targets: Mapping[str, float],
) -> str:
    names = ("train", "val", "test")
    group_size = len(members)

    def score(name: str) -> tuple[float, float, float, int]:
        row_deficit = (targets[name] - split_counts.get(name, 0)) / max(1.0, targets[name])
        category_deficit = sum(
            (targets[name] / max(1.0, sum(targets.values())))
            - (
                category_counts.get(member.category, {}).get(name, 0)
                / max(1, sum(category_counts.get(member.category, {}).values()))
            )
            for member in members
        )
        source_deficit = sum(
            (targets[name] / max(1.0, sum(targets.values())))
            - (
                source_counts.get(member.source_pack, {}).get(name, 0)
                / max(1, sum(source_counts.get(member.source_pack, {}).values()))
            )
            for member in members
            if member.source_pack
        )
        overshoot = max(0.0, split_counts.get(name, 0) + group_size - targets[name])
        return (
            row_deficit + 0.25 * category_deficit + 0.10 * source_deficit,
            -overshoot,
            row_deficit,
            -names.index(name),
        )

    return max(names, key=score)


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

    target_train = round(total_count * train_fraction)
    target_val = round(total_count * val_fraction)
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
