from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.codec.roles import ROLE_HIGHLIGHT, ROLE_MIDTONE, ROLE_OUTLINE
from spritelab.training.palette_swap import (
    DEFAULT_SWAP_FAMILIES,
    PaletteSwapConfig,
    apply_palette_swap,
    estimate_applied,
    nearest_family,
    parse_families,
    sample_seed,
)

# --- shared tiny sprite: dark outline ring + midtone fill + highlight ---------


def _sprite() -> dict:
    index = np.zeros((32, 32), dtype=np.int64)
    role = np.zeros((32, 32), dtype=np.int64)
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1.0
    index[8:24, 8:24] = 2  # fill
    index[9:12, 9:12] = 3  # highlight
    index[8, 8:24] = 1  # outline row
    role[8:24, 8:24] = ROLE_MIDTONE
    role[9:12, 9:12] = ROLE_HIGHLIGHT
    role[8, 8:24] = ROLE_OUTLINE
    palette = np.zeros((5, 3), dtype=np.float32)
    palette[1] = [0.08, 0.08, 0.10]  # dark outline
    palette[2] = [0.20, 0.58, 0.26]  # green midtone fill
    palette[3] = [0.55, 0.90, 0.60]  # light green highlight
    mask = np.zeros((5,), dtype=bool)
    mask[1] = mask[2] = mask[3] = True
    return {"index": index, "role": role, "alpha": alpha, "palette": palette, "mask": mask}


def _record() -> dict:
    return {
        "sprite_id": "t_green_sword",
        "caption": "green sword",
        "colors": ["green"],
        "materials": ["metal"],
        "category": "weapon",
        "base_object": "sword",
        "conditioning": {"semantic_v3": {"attributes": {"colors": ["green"], "materials": ["metal"]}}},
    }


def _apply(config: PaletteSwapConfig, sprite=None, record=None):
    sprite = sprite or _sprite()
    record = record or _record()
    return apply_palette_swap(
        index_map=sprite["index"],
        alpha=sprite["alpha"],
        role_map=sprite["role"],
        palette_rgb=sprite["palette"],
        palette_mask=sprite["mask"],
        record=record,
        caption=str(record.get("caption", "")),
        sprite_id=str(record.get("sprite_id", "")),
        config=config,
    )


# --- Part C: color family remapping ------------------------------------------


def test_parse_families_filters_unknown_and_defaults() -> None:
    assert parse_families("red,blue,not_a_color") == ("red", "blue")
    assert parse_families("") == DEFAULT_SWAP_FAMILIES
    assert parse_families(["Green", "GOLD"]) == ("green", "gold")


def test_remap_preserves_luminance_order_and_produces_ramp() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), seed=1)
    result = _apply(config)
    assert result.applied
    fill = result.palette_rgb[2]
    highlight = result.palette_rgb[3]
    # Highlight was lighter than fill in the source; order preserved after recolor.
    assert float(np.mean(highlight)) > float(np.mean(fill))
    # Recolored fill is now blue-dominant.
    assert nearest_family(fill) == "blue"


def test_gray_family_desaturates() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("gray",), seed=1)
    result = _apply(config)
    fill = result.palette_rgb[2]
    assert abs(float(fill[0]) - float(fill[1])) < 0.05
    assert abs(float(fill[1]) - float(fill[2])) < 0.05


# --- Part B: role-preserving recolor -----------------------------------------


def test_outline_preserved_and_fill_recolored() -> None:
    sprite = _sprite()
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("red",), seed=2)
    result = _apply(config, sprite=sprite)
    assert result.applied
    assert np.allclose(result.palette_rgb[1], sprite["palette"][1])  # outline untouched
    assert not np.allclose(result.palette_rgb[2], sprite["palette"][2])  # fill recolored
    assert "midtone" in result.roles_recolored


def test_luminance_fallback_preserves_near_black_when_roles_missing() -> None:
    sprite = _sprite()
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("red",), seed=2)
    result = apply_palette_swap(
        index_map=sprite["index"],
        alpha=sprite["alpha"],
        role_map=None,  # force luminance fallback
        palette_rgb=sprite["palette"],
        palette_mask=sprite["mask"],
        record=_record(),
        caption="green sword",
        sprite_id="t_green_sword",
        config=config,
    )
    assert result.applied
    assert np.allclose(result.palette_rgb[1], sprite["palette"][1])  # near-black preserved
    assert not np.allclose(result.palette_rgb[2], sprite["palette"][2])


# --- Part D: prompt/semantic update ------------------------------------------


def test_prompt_and_semantic_colors_updated() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), seed=3)
    result = _apply(config)
    assert result.target_color_family == "blue"
    assert result.caption == "blue sword"
    assert result.record["colors"] == ["blue"]
    assert result.record["primary_color"] == "blue"
    assert result.record["conditioning"]["semantic_v3"]["attributes"]["colors"] == ["blue"]


def test_prompt_prepends_color_when_absent() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("red",), seed=3)
    record = _record()
    record["caption"] = "a sword"
    result = _apply(config, record=record)
    assert result.caption == "red sword" or result.caption.startswith("red ")


def test_material_conflict_removed() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), seed=3)
    record = _record()
    record["object_name"] = "gold_coin"
    record["materials"] = ["gold"]
    record["conditioning"]["semantic_v3"]["attributes"]["materials"] = ["gold"]
    result = _apply(config, record=record)
    assert "gold" not in result.record["materials"]
    assert "gold" not in result.record["conditioning"]["semantic_v3"]["attributes"]["materials"]


def test_update_prompts_disabled_keeps_record() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), update_prompts=False, seed=3)
    result = _apply(config)
    assert result.applied
    assert result.caption == "green sword"
    assert result.record["colors"] == ["green"]


def test_require_explicit_caption_color_skips_colorless_caption() -> None:
    record = _record()
    record["caption"] = "a sword"
    config = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("blue",),
        require_explicit_caption_color=True,
        seed=3,
    )
    result = _apply(config, record=record)
    assert result.applied is False
    assert result.ineligibility_reason == "no_explicit_caption_color"


def test_require_explicit_semantic_color_skips_missing_semantic_colors() -> None:
    record = _record()
    record.pop("colors", None)
    record.pop("color", None)
    record["conditioning"]["semantic_v3"]["attributes"]["colors"] = []
    config = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("blue",),
        require_explicit_semantic_color=True,
        seed=3,
    )
    result = _apply(config, record=record)
    assert result.applied is False
    assert result.ineligibility_reason == "no_explicit_semantic_color"


def test_requiring_both_explicit_color_gates() -> None:
    config = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("blue",),
        require_explicit_caption_color=True,
        require_explicit_semantic_color=True,
        seed=3,
    )
    assert _apply(config).applied is True

    record = _record()
    record["caption"] = "a sword"
    result = _apply(config, record=record)
    assert result.applied is False
    assert result.ineligibility_reason == "no_explicit_caption_color"


def test_legacy_require_explicit_color_alias_requires_both_gates() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), require_explicit_color=True, seed=3)
    report = config.report_dict()
    assert report["palette_swap_require_explicit_color_maps_to"] == "caption_and_semantic"
    assert report["palette_swap_require_explicit_caption_color"] is True
    assert report["palette_swap_require_explicit_semantic_color"] is True


def test_no_caption_prepend_leaves_colorless_caption_unchanged_when_semantic_gate_allows() -> None:
    record = _record()
    record["caption"] = "a sword"
    config = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("blue",),
        require_explicit_semantic_color=True,
        no_caption_prepend=True,
        seed=3,
    )
    result = _apply(config, record=record)
    assert result.applied is True
    assert result.caption == "a sword"
    assert result.caption_change_kind == "none"


def test_no_caption_prepend_skips_colorless_caption_without_semantic_gate() -> None:
    record = _record()
    record["caption"] = "a sword"
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), no_caption_prepend=True, seed=3)
    result = _apply(config, record=record)
    assert result.applied is False
    assert result.ineligibility_reason == "no_caption_color_no_prepend"


def test_same_family_target_is_skipped_when_no_different_target_exists() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("green",), seed=3)
    result = _apply(config)
    assert result.applied is False
    assert result.same_family_skip is True
    assert result.ineligibility_reason == "same_family_target_unavailable"


def test_same_family_target_is_resampled_to_different_family() -> None:
    result = None
    for seed in range(100):
        config = PaletteSwapConfig(enabled=True, prob=1.0, families=("green", "blue"), seed=seed)
        candidate = _apply(config)
        if candidate.target_resampled_from_same_family:
            result = candidate
            break
    assert result is not None
    assert result.applied is True
    assert result.source_color_family == "green"
    assert result.target_color_family == "blue"


# --- Part F/I: determinism ----------------------------------------------------


def test_same_seed_produces_identical_augmentation() -> None:
    config = PaletteSwapConfig(enabled=True, prob=0.5, families=("red", "blue", "gold"), seed=99)
    first = _apply(config)
    second = _apply(config)
    assert first.applied == second.applied
    assert first.target_color_family == second.target_color_family
    assert np.array_equal(first.palette_rgb, second.palette_rgb)


def test_deterministic_mode_ignores_draw_index() -> None:
    config = PaletteSwapConfig(enabled=True, prob=1.0, families=("red", "blue", "gold"), seed=99)
    first = _apply(config)
    second = apply_palette_swap(
        index_map=_sprite()["index"],
        alpha=_sprite()["alpha"],
        role_map=_sprite()["role"],
        palette_rgb=_sprite()["palette"],
        palette_mask=_sprite()["mask"],
        record=_record(),
        caption="green sword",
        sprite_id="t_green_sword",
        config=config,
        draw_index=999,
    )
    assert first.applied == second.applied
    assert first.target_color_family == second.target_color_family
    assert np.array_equal(first.palette_rgb, second.palette_rgb)


def test_stochastic_mode_varies_targets_for_same_sprite_across_draws() -> None:
    config = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("red", "blue", "yellow", "purple"),
        stochastic=True,
        seed=99,
    )
    targets = {
        _apply(config, record={**_record(), "sprite_id": "same_sprite"}).target_color_family
        for _ in range(1)
    }
    targets |= {
        apply_palette_swap(
            index_map=_sprite()["index"],
            alpha=_sprite()["alpha"],
            role_map=_sprite()["role"],
            palette_rgb=_sprite()["palette"],
            palette_mask=_sprite()["mask"],
            record={**_record(), "sprite_id": "same_sprite"},
            caption="green sword",
            sprite_id="same_sprite",
            config=config,
            draw_index=draw_index,
        ).target_color_family
        for draw_index in range(20)
    }
    assert len(targets) > 1


def test_stochastic_keep_original_probability_keeps_prompt_and_palette() -> None:
    sprite = _sprite()
    config = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("blue",),
        stochastic=True,
        keep_original_prob=1.0,
        seed=3,
    )
    result = _apply(config, sprite=sprite)
    assert result.eligible is True
    assert result.kept_original is True
    assert result.applied is False
    assert result.caption == "green sword"
    assert result.record["colors"] == ["green"]
    assert np.array_equal(result.palette_rgb, sprite["palette"])
    meta = result.metadata()
    assert meta["palette_swap_stochastic"] is True
    assert meta["palette_swap_draw_index"] == 0


def test_colorless_caption_structured_only_mode_does_not_prepend() -> None:
    record = _record()
    record["caption"] = "a sword"
    config = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("blue",),
        require_explicit_caption_color=True,
        require_explicit_semantic_color=True,
        allow_colorless_caption_if_semantic_color=True,
        no_caption_prepend=True,
        seed=3,
    )
    result = _apply(config, record=record)
    assert result.applied is True
    assert result.caption == "a sword"
    assert result.caption_change_kind == "none"
    assert result.record["colors"] == ["blue"]


def test_sample_seed_is_worker_order_independent() -> None:
    # Same (base_seed, sprite_id) -> identical seed regardless of call order.
    seeds_a = [sample_seed(7, sid) for sid in ("a", "b", "c")]
    seeds_b = [sample_seed(7, sid) for sid in ("c", "a", "b")]
    assert seeds_a[0] == seeds_b[1]  # "a"
    assert seeds_a[2] == seeds_b[0]  # "c"


def test_estimate_applied_matches_prob_direction() -> None:
    records = [{"sprite_id": f"s_{i}"} for i in range(200)]
    disabled = PaletteSwapConfig(enabled=False, prob=0.0)
    assert estimate_applied(records, disabled)["applied_count"] == 0
    enabled = PaletteSwapConfig(enabled=True, prob=0.5, families=("red", "blue"), seed=5)
    stats = estimate_applied(records, enabled)
    assert 0.3 < stats["applied_rate"] < 0.7


def test_inactive_config_is_noop() -> None:
    sprite = _sprite()
    config = PaletteSwapConfig(enabled=True, prob=0.0)
    result = _apply(config, sprite=sprite)
    assert result.applied is False
    assert np.array_equal(result.palette_rgb, sprite["palette"])


# --- Dataset / DataLoader integration ----------------------------------------

torch = pytest.importorskip("torch", exc_type=ImportError)

from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch  # noqa: E402
from spritelab.training.structured_conditioning import (  # noqa: E402
    build_structured_conditioning_vocab,
)


def _build_dataset_dir(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    count = 4
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    alpha[:, 8:24, 8:24] = 255
    index = np.zeros((count, 32, 32), dtype=np.int16)
    index[:, 8:24, 8:24] = 2
    index[:, 8, 8:24] = 1  # outline
    role = np.zeros((count, 32, 32), dtype=np.uint8)
    role[:, 8:24, 8:24] = ROLE_MIDTONE
    role[:, 8, 8:24] = ROLE_OUTLINE
    palette = np.zeros((count, 5, 3), dtype=np.uint8)
    palette[:, 1] = [20, 20, 26]  # outline
    palette[:, 2] = [50, 150, 66]  # green fill
    palette_mask = np.zeros((count, 5), dtype=bool)
    palette_mask[:, 0] = True  # background / transparent row
    palette_mask[:, 1] = True
    palette_mask[:, 2] = True
    sprite_ids = [f"s_{i}" for i in range(count)]
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index,
        role_map=role,
        palette=palette,
        palette_mask=palette_mask,
        category_id=np.ones((count,), dtype=np.int64),
        sprite_id=np.array(sprite_ids, dtype=np.str_),
    )
    rows = [
        {
            "sprite_id": sid,
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": i,
            "object_name": "green_sword",
            "base_object": "sword",
            "category": "weapon",
            "caption": "green sword",
            "colors": ["green"],
        }
        for i, sid in enumerate(sprite_ids)
    ]
    manifest = dataset / "training_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return dataset, manifest


def test_dataset_augmentation_preserves_structure_and_updates_colors(tmp_path: Path) -> None:
    dataset_dir, manifest = _build_dataset_dir(tmp_path)
    swap = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), seed=123)
    plain = SpriteTrainingDataset(dataset_dir, manifest, split="train", cache_samples=False)
    aug = SpriteTrainingDataset(dataset_dir, manifest, split="train", palette_swap=swap, cache_samples=False)

    base = plain[0]
    swapped = aug[0]
    # alpha + index geometry unchanged
    assert torch.equal(base["alpha"], swapped["alpha"])
    assert torch.equal(base["index_map"], swapped["index_map"])
    assert torch.equal(base["palette_mask"], swapped["palette_mask"])
    # transparent pixels remain transparent
    assert float(swapped["alpha"].sum()) == float(base["alpha"].sum())
    # visible RGB changed
    assert not torch.allclose(base["rgb"], swapped["rgb"])
    # no fully transparent corruption: still has visible pixels
    assert float((swapped["alpha"] > 0).sum()) > 0
    # caption updated deterministically
    assert swapped["caption"] == "blue sword"
    assert swapped["palette_swap"]["palette_swap_applied"] is True
    assert swapped["palette_swap"]["target_color_family"] == "blue"


def test_dataset_augmentation_is_deterministic_across_instances(tmp_path: Path) -> None:
    dataset_dir, manifest = _build_dataset_dir(tmp_path)
    swap = PaletteSwapConfig(enabled=True, prob=0.5, families=("red", "blue", "gold"), seed=123)
    a = SpriteTrainingDataset(dataset_dir, manifest, split="train", palette_swap=swap, cache_samples=False)
    b = SpriteTrainingDataset(dataset_dir, manifest, split="train", palette_swap=swap, cache_samples=False)
    for i in range(len(a)):
        assert torch.equal(a[i]["rgba"], b[i]["rgba"])
        assert a[i]["caption"] == b[i]["caption"]
        assert a[i]["palette_swap"] == b[i]["palette_swap"]


def test_dataset_stochastic_repeated_access_varies_same_sprite(tmp_path: Path) -> None:
    dataset_dir, manifest = _build_dataset_dir(tmp_path)
    swap = PaletteSwapConfig(
        enabled=True,
        prob=1.0,
        families=("red", "blue", "yellow", "purple"),
        stochastic=True,
        seed=123,
    )
    dataset = SpriteTrainingDataset(dataset_dir, manifest, split="train", palette_swap=swap, cache_samples=True)
    samples = [dataset[0] for _ in range(20)]
    targets = {sample["palette_swap"]["target_color_family"] for sample in samples if sample["palette_swap"]["palette_swap_applied"]}
    draw_indices = [sample["palette_swap"]["palette_swap_draw_index"] for sample in samples]
    assert len(targets) > 1
    assert draw_indices == list(range(20))


def test_batch_carries_updated_structured_colors(tmp_path: Path) -> None:
    dataset_dir, manifest = _build_dataset_dir(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    vocab = build_structured_conditioning_vocab(
        [*rows, {"colors": ["blue"], "category": "weapon", "base_object": "sword"}]
    )
    swap = PaletteSwapConfig(enabled=True, prob=1.0, families=("blue",), seed=123)
    dataset = SpriteTrainingDataset(
        dataset_dir, manifest, split="train", palette_swap=swap, structured_vocab=vocab, cache_samples=False
    )
    batch = collate_sprite_batch([dataset[0], dataset[1]])
    assert "palette_swap" in batch
    assert all(item["target_color_family"] == "blue" for item in batch["palette_swap"])
    blue_id = list(vocab.colors).index("blue")
    # updated primary color id points at blue for augmented samples
    assert int(batch["structured_primary_color_id"][0]) == blue_id


def test_tiny_cpu_training_smoke_with_palette_swap(tmp_path: Path) -> None:
    from spritelab.training.generator_challenger import (
        ChallengerTrainConfig,
        run_challenger_training,
    )

    dataset_dir, manifest = _build_dataset_dir(tmp_path)
    out = tmp_path / "run"
    report = run_challenger_training(
        ChallengerTrainConfig(
            dataset_dir=dataset_dir,
            training_manifest=manifest,
            out_dir=out,
            batch_size=2,
            max_steps=2,
            device="cpu",
            seed=123,
            conditioning_mode="caption_semantic_structured",
            base_channels=8,
            channel_mults="1,2",
            embed_dim=8,
            sample_every=0,
            save_every=0,
            validation_mode="none",
            palette_swap_augmentation=True,
            palette_swap_prob=1.0,
            palette_swap_families="blue",
        )
    )
    assert report["steps_completed"] == 2
    swap = report["palette_swap"]
    assert swap["palette_swap_augmentation"] is True
    assert swap["palette_swap_prob"] == 1.0
    assert swap["applied_rate"] == 1.0
    config_json = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config_json["palette_swap"]["palette_swap_families"] == ["blue"]
