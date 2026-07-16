from __future__ import annotations

import pytest

from spritelab.dataset_maker.group_aware_split import (
    GroupAwareSplitConfig,
    SplitPlanRecord,
    plan_group_aware_split,
)


def _record(
    sprite_id: str,
    *,
    alpha: str = "alpha",
    rgba: str = "rgba",
    category: str = "item_icon",
    object_name: str = "item",
    source_pack: str = "pack_a",
    current_split: str = "train",
    explicit: tuple[str, ...] = (),
    override: str | None = None,
) -> SplitPlanRecord:
    manifest = {
        "sprite_id": sprite_id,
        "category": category,
        "object_name": object_name,
        "source_pack": source_pack,
    }
    if override:
        manifest["split_override"] = override
    return SplitPlanRecord(manifest, current_split, alpha, rgba, explicit, f"{source_pack}:sheet")


def test_exact_duplicates_and_recolors_cannot_cross_splits() -> None:
    records = [
        _record("exact_a", alpha="a1", rgba="r1"),
        _record("exact_b", alpha="a1", rgba="r1", current_split="test"),
        _record("recolor_a", alpha="a2", rgba="r2"),
        _record("recolor_b", alpha="a2", rgba="r3", current_split="val"),
        *[
            _record(f"other_{index}", alpha=f"a{index + 10}", rgba=f"r{index + 10}", object_name=f"object_{index}")
            for index in range(12)
        ],
    ]
    plan = plan_group_aware_split(records, GroupAwareSplitConfig(seed=9))
    assert plan["assignments"]["exact_a"] == plan["assignments"]["exact_b"]
    assert plan["assignments"]["recolor_a"] == plan["assignments"]["recolor_b"]
    assert plan["leakage_after"]["gates_pass"]


def test_manifest_order_and_seed_are_deterministic() -> None:
    records = [
        _record(f"s{index}", alpha=f"a{index}", rgba=f"r{index}", category=f"cat_{index % 3}") for index in range(30)
    ]
    first = plan_group_aware_split(records, GroupAwareSplitConfig(seed=123))
    second = plan_group_aware_split(list(reversed(records)), GroupAwareSplitConfig(seed=123))
    assert first["assignments"] == second["assignments"]


def test_conflicting_manual_overrides_fail_clearly() -> None:
    records = [
        _record("a", alpha="shared", rgba="r1", override="train"),
        _record("b", alpha="shared", rgba="r2", override="test"),
    ]
    with pytest.raises(ValueError, match="divide leakage group"):
        plan_group_aware_split(records, GroupAwareSplitConfig())


def test_explicit_variant_groups_and_source_ood_holdout_are_complete() -> None:
    records = [
        _record("v1", alpha="a1", rgba="r1", explicit=("family:potion_1",), source_pack="pack_a"),
        _record("v2", alpha="a2", rgba="r2", explicit=("family:potion_1",), source_pack="pack_a"),
        *[
            _record(f"b{index}", alpha=f"ba{index}", rgba=f"br{index}", source_pack="pack_b", category="tool")
            for index in range(12)
        ],
        *[
            _record(f"c{index}", alpha=f"ca{index}", rgba=f"cr{index}", source_pack="pack_c", category="tool")
            for index in range(12)
        ],
    ]
    config = GroupAwareSplitConfig(source_ood_holdout_packs=("pack_b",))
    plan = plan_group_aware_split(records, config)
    assert plan["assignments"]["v1"] == plan["assignments"]["v2"]
    assert {split for sprite_id, split in plan["source_ood"]["assignments"].items() if sprite_id.startswith("b")} == {
        "eval"
    }
    assert {split for sprite_id, split in plan["source_ood"]["assignments"].items() if sprite_id.startswith("c")} == {
        "train"
    }

    authored = [
        _record("author_a", alpha="author_a", rgba="author_a", source_pack="author_pack"),
        _record("author_b", alpha="author_b", rgba="author_b", source_pack="other_pack"),
    ]
    authored[0].manifest["author"] = "held_out_artist"
    author_plan = plan_group_aware_split(
        authored, GroupAwareSplitConfig(source_ood_holdout_authors=("held_out_artist",))
    )
    assert author_plan["source_ood"]["assignments"]["author_a"] == "eval"


def test_group_aware_split_is_approximately_balanced_and_legacy_split_still_loads() -> None:
    from spritelab.training.splits import make_group_aware_split

    records = [
        _record(
            f"s{index}",
            alpha=f"a{index}",
            rgba=f"r{index}",
            category=f"cat_{index % 3}",
            object_name=f"object_{index % 6}",
        )
        for index in range(60)
    ]
    plan = plan_group_aware_split(records, GroupAwareSplitConfig(seed=7))
    fractions = plan["main"]["fractions"]
    assert abs(fractions["train"] - 0.8) < 0.12
    assert abs(fractions["val"] - 0.1) < 0.10
    assert abs(fractions["test"] - 0.1) < 0.10
    assert make_group_aware_split(["a", "b"], {"a": "g", "b": "g"}).train == ("a", "b")
