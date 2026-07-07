from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.codec.roles import ROLE_HIGHLIGHT, ROLE_MIDTONE, ROLE_OUTLINE
from spritelab.training.palette_swap import PaletteSwapConfig, apply_palette_swap
from spritelab.training.palette_swap_review import (
    PaletteSwapReviewConfig,
    aggregate_metrics,
    compute_red_flags,
    review_palette_swap,
    summarize_dataset_palette_swap,
    swap_config_from_review,
)


def _build_dataset(
    tmp_path: Path,
    *,
    count: int = 8,
    explicit_color: bool = True,
    semantic_color: bool | None = None,
    with_roles: bool = True,
) -> tuple[Path, Path]:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    alpha = np.zeros((count, 32, 32), dtype=np.uint8)
    alpha[:, 8:24, 8:24] = 255
    index = np.zeros((count, 32, 32), dtype=np.int16)
    index[:, 8:24, 8:24] = 2
    index[:, 9:12, 9:12] = 3
    index[:, 8, 8:24] = 1
    role = np.zeros((count, 32, 32), dtype=np.uint8)
    if with_roles:
        role[:, 8:24, 8:24] = ROLE_MIDTONE
        role[:, 9:12, 9:12] = ROLE_HIGHLIGHT
        role[:, 8, 8:24] = ROLE_OUTLINE
    palette = np.zeros((count, 5, 3), dtype=np.uint8)
    palette[:, 1] = [20, 20, 26]
    palette[:, 2] = [50, 150, 66]  # green fill
    palette[:, 3] = [140, 230, 150]  # green highlight
    mask = np.zeros((count, 5), dtype=bool)
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = True
    mask[:, 3] = True
    ids = [f"s_{i}" for i in range(count)]
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index,
        role_map=role,
        palette=palette,
        palette_mask=mask,
        category_id=np.ones((count,), dtype=np.int64),
        sprite_id=np.array(ids, dtype=np.str_),
    )
    rows = []
    if semantic_color is None:
        semantic_color = explicit_color
    for i, sid in enumerate(ids):
        row = {
            "sprite_id": sid,
            "split": "train",
            "npz_file": "train.npz",
            "npz_row": i,
            "object_name": "green_sword",
            "base_object": "sword",
            "category": "weapon",
            "caption": "green sword" if explicit_color else "a sword",
        }
        if semantic_color:
            row["colors"] = ["green"]
        rows.append(row)
    manifest = dataset / "training_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return dataset, manifest


def _review_config(dataset: Path, manifest: Path, out: Path, **overrides) -> PaletteSwapReviewConfig:
    params = dict(
        dataset_dir=dataset,
        training_manifest=manifest,
        out_dir=out,
        seed=20260706,
        max_samples=8,
        palette_swap_prob=1.0,
        palette_swap_families="red,blue,green,yellow,purple",
    )
    params.update(overrides)
    return PaletteSwapReviewConfig(**params)


# --- Part F: review CLI writes JSON/Markdown/contact sheets -------------------


def test_review_writes_reports_and_contact_sheets(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path)
    result = review_palette_swap(_review_config(dataset, manifest, tmp_path / "review"))
    assert result.json_path.is_file()
    assert result.markdown_path.is_file()
    assert result.jsonl_path.is_file()
    assert result.contact_sheets  # at least the pairs sheet
    assert any(name == "pairs" for name in result.contact_sheets)
    assert any(name.startswith("target_family:") for name in result.contact_sheets)
    assert any(name.startswith("source_to_target:") for name in result.contact_sheets)

    rows = [json.loads(line) for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    row = rows[0]
    for key in (
        "sprite_id",
        "category",
        "object_name",
        "base_object",
        "original_caption",
        "augmented_caption",
        "original_semantic_colors",
        "augmented_semantic_colors",
        "original_materials",
        "augmented_materials",
        "source_color_family",
        "target_color_family",
        "roles_recolored",
        "recolored_palette_indices",
        "role_map_trusted",
        "fallback_heuristic_used",
        "visible_color_count_before",
        "visible_color_count_after",
        "palette_family_histogram_before",
        "palette_family_histogram_after",
    ):
        assert key in row


def test_review_json_has_stable_top_level_schema_fields(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path)
    result = review_palette_swap(_review_config(dataset, manifest, tmp_path / "review"))
    report = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == 2
    for key in (
        "sample_count",
        "applied_count",
        "applied_rate",
        "eligible_count",
        "ineligible_count",
        "applied_rate_eligible",
        "fallback_heuristic_rate",
        "material_conflict_drop_count",
        "caption_color_replaced_count",
        "caption_color_prepended_count",
        "caption_no_color_count",
        "mean_rgb_delta_visible",
        "mean_palette_entries_changed",
        "mean_visible_color_count_before",
        "mean_visible_color_count_after",
        "target_family_counts",
        "source_family_counts",
        "source_to_target_matrix",
        "ineligibility_reason_counts",
        "same_family_skip_count",
        "red_flags",
    ):
        assert key in report


def test_review_cli_runs(tmp_path: Path) -> None:
    from spritelab.training.cli import main as train_cli

    dataset, manifest = _build_dataset(tmp_path)
    out = tmp_path / "cli_review"
    train_cli(
        [
            "dataset-palette-swap-review",
            "--dataset",
            str(dataset),
            "--training-manifest",
            str(manifest),
            "--out-dir",
            str(out),
            "--seed",
            "20260706",
            "--palette-swap-prob",
            "1.0",
            "--palette-swap-families",
            "red,blue,green,yellow,purple",
            "--palette-swap-require-explicit-caption-color",
            "--palette-swap-require-explicit-semantic-color",
            "--palette-swap-no-caption-prepend",
            "--max-samples",
            "8",
        ]
    )
    assert (out / "palette_swap_review.json").is_file()
    assert (out / "palette_swap_review.md").is_file()
    assert (out / "palette_swap_samples.jsonl").is_file()
    report = json.loads((out / "palette_swap_review.json").read_text(encoding="utf-8"))
    assert report["require_explicit_caption_color"] is True
    assert report["require_explicit_semantic_color"] is True
    assert report["no_caption_prepend"] is True
    assert report["config"]["palette_swap_no_caption_prepend"] is True


def test_generator_challenger_cli_accepts_conservative_palette_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spritelab.training import generator_challenger
    from spritelab.training.cli import main as train_cli

    captured = []

    def fake_train(config):
        captured.append(config)
        return {"initial_train_loss": 1.0, "final_train_loss": 0.5, "val_loss": None}

    monkeypatch.setattr(generator_challenger, "run_challenger_training", fake_train)
    train_cli(
        [
            "generator-challenger",
            "--dataset",
            str(tmp_path / "dataset"),
            "--training-manifest",
            str(tmp_path / "training_manifest.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--palette-swap-augmentation",
            "--palette-swap-prob",
            "0.3",
            "--palette-swap-target-families",
            "red,blue,green,yellow,purple",
            "--palette-swap-require-explicit-caption-color",
            "--palette-swap-require-explicit-semantic-color",
            "--palette-swap-no-caption-prepend",
            "--palette-swap-allow-material-colors",
            "false",
        ]
    )

    assert captured
    assert captured[0].palette_swap_require_explicit_caption_color is True
    assert captured[0].palette_swap_require_explicit_semantic_color is True
    assert captured[0].palette_swap_no_caption_prepend is True
    assert captured[0].palette_swap_allow_material_colors is False


# --- Part F: deterministic source->target matrix -----------------------------


def test_source_to_target_matrix_is_deterministic(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path)
    first = review_palette_swap(_review_config(dataset, manifest, tmp_path / "r1"))
    second = review_palette_swap(_review_config(dataset, manifest, tmp_path / "r2"))
    assert first.report["metrics"]["source_to_target_matrix"] == second.report["metrics"]["source_to_target_matrix"]
    assert first.report["metrics"]["target_family_counts"] == second.report["metrics"]["target_family_counts"]


# --- Part F: conservative mode skips samples without explicit color -----------


def test_conservative_requires_explicit_color(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path, explicit_color=False)
    result = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "review", palette_swap_require_explicit_color=True)
    )
    metrics = result.report["metrics"]
    assert metrics["applied_count"] == 0
    assert metrics["ineligibility_reason_counts"].get("no_explicit_caption_color", 0) == metrics["sample_count"]


def test_require_explicit_caption_color_skips_colorless_captions(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path, explicit_color=False, semantic_color=True)
    result = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "review", palette_swap_require_explicit_caption_color=True)
    )
    metrics = result.report["metrics"]
    assert metrics["applied_count"] == 0
    assert metrics["caption_color_prepended_count"] == 0
    assert metrics["ineligibility_reason_counts"].get("no_explicit_caption_color", 0) == metrics["sample_count"]


def test_require_explicit_semantic_color_skips_missing_semantic_colors(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path, explicit_color=True, semantic_color=False)
    result = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "review", palette_swap_require_explicit_semantic_color=True)
    )
    metrics = result.report["metrics"]
    assert metrics["applied_count"] == 0
    assert metrics["ineligibility_reason_counts"].get("no_explicit_semantic_color", 0) == metrics["sample_count"]


def test_requiring_both_explicit_gates_works(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path, explicit_color=True, semantic_color=True)
    result = review_palette_swap(
        _review_config(
            dataset,
            manifest,
            tmp_path / "review",
            palette_swap_require_explicit_caption_color=True,
            palette_swap_require_explicit_semantic_color=True,
        )
    )
    assert result.report["metrics"]["applied_count"] == 8
    assert result.report["require_explicit_caption_color"] is True
    assert result.report["require_explicit_semantic_color"] is True


def test_no_caption_prepend_plus_caption_requirement_yields_zero_prepends(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path, explicit_color=False, semantic_color=True)
    result = review_palette_swap(
        _review_config(
            dataset,
            manifest,
            tmp_path / "review",
            palette_swap_require_explicit_caption_color=True,
            palette_swap_no_caption_prepend=True,
        )
    )
    metrics = result.report["metrics"]
    assert metrics["applied_count"] == 0
    assert metrics["caption_color_prepended_count"] == 0
    names = {flag["name"] for flag in result.report["red_flags"]}
    assert "frequent_caption_prepending" not in names


# --- Part F: conservative mode skips samples without reliable role map --------


def test_conservative_requires_role_map(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path, with_roles=False)
    result = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "review", palette_swap_require_role_map=True)
    )
    metrics = result.report["metrics"]
    assert metrics["applied_count"] == 0
    assert metrics["ineligibility_reason_counts"].get("role_map_unreliable", 0) > 0


# --- Part F: material colors excluded when allow_material_colors=false --------


def test_material_colors_excluded(tmp_path: Path) -> None:
    # Gold-fill sprite: eligible when material colors allowed, ineligible when not.
    dataset = tmp_path / "ds"
    dataset.mkdir()
    alpha = np.zeros((2, 32, 32), dtype=np.uint8)
    alpha[:, 8:24, 8:24] = 255
    index = np.zeros((2, 32, 32), dtype=np.int16)
    index[:, 8:24, 8:24] = 2
    index[:, 8, 8:24] = 1
    role = np.zeros((2, 32, 32), dtype=np.uint8)
    role[:, 8:24, 8:24] = ROLE_MIDTONE
    role[:, 8, 8:24] = ROLE_OUTLINE
    palette = np.zeros((2, 5, 3), dtype=np.uint8)
    palette[:, 1] = [20, 20, 26]
    palette[:, 2] = [200, 158, 52]  # gold fill
    mask = np.zeros((2, 5), dtype=bool)
    mask[:, 0] = mask[:, 1] = mask[:, 2] = True
    ids = ["g_0", "g_1"]
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index,
        role_map=role,
        palette=palette,
        palette_mask=mask,
        category_id=np.ones((2,), dtype=np.int64),
        sprite_id=np.array(ids, dtype=np.str_),
    )
    rows = [
        {"sprite_id": s, "split": "train", "npz_file": "train.npz", "npz_row": i,
         "object_name": "gold_coin", "base_object": "coin", "category": "item_icon",
         "caption": "gold coin", "colors": ["gold"], "materials": ["gold"]}
        for i, s in enumerate(ids)
    ]
    manifest = dataset / "training_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    allowed = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "allow", max_samples=2,
                       palette_swap_families="red,blue,gold", palette_swap_source_families="red,blue,green,gold")
    )
    assert allowed.report["metrics"]["applied_count"] > 0

    blocked = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "block", max_samples=2,
                       palette_swap_families="red,blue", palette_swap_allow_material_colors=False)
    )
    metrics = blocked.report["metrics"]
    assert metrics["applied_count"] == 0
    assert metrics["ineligibility_reason_counts"].get("material_color_source", 0) > 0


# --- Part F: target family distribution respects target families -------------


def test_target_families_respected(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path, count=16)
    result = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "review", max_samples=16,
                       palette_swap_target_families="red,blue")
    )
    targets = set(result.report["metrics"]["target_family_counts"])
    assert targets
    assert targets <= {"red", "blue"}


# --- Part F: color confidence threshold works --------------------------------


def test_color_confidence_threshold(tmp_path: Path) -> None:
    # Mixed sprite: fill split across two families -> lower confidence.
    dataset = tmp_path / "ds"
    dataset.mkdir()
    index = np.zeros((1, 32, 32), dtype=np.int16)
    index[0, 8:16, 8:24] = 2  # half green
    index[0, 16:24, 8:24] = 3  # half red
    index[0, 8, 8:24] = 1
    alpha = np.zeros((1, 32, 32), dtype=np.uint8)
    alpha[0, 8:24, 8:24] = 255
    role = np.zeros((1, 32, 32), dtype=np.uint8)
    role[0, 8:24, 8:24] = ROLE_MIDTONE
    role[0, 8, 8:24] = ROLE_OUTLINE
    palette = np.zeros((1, 5, 3), dtype=np.uint8)
    palette[0, 1] = [20, 20, 26]
    palette[0, 2] = [50, 150, 66]  # green
    palette[0, 3] = [200, 50, 50]  # red
    mask = np.zeros((1, 5), dtype=bool)
    mask[0, 0] = mask[0, 1] = mask[0, 2] = mask[0, 3] = True
    np.savez_compressed(
        dataset / "train.npz",
        alpha=alpha,
        index_map=index,
        role_map=role,
        palette=palette,
        palette_mask=mask,
        category_id=np.ones((1,), dtype=np.int64),
        sprite_id=np.array(["mixed_0"], dtype=np.str_),
    )
    rows = [{"sprite_id": "mixed_0", "split": "train", "npz_file": "train.npz", "npz_row": 0,
             "category": "weapon", "caption": "green sword", "colors": ["green"]}]
    manifest = dataset / "training_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    lenient = review_palette_swap(_review_config(dataset, manifest, tmp_path / "lenient", max_samples=1))
    assert lenient.report["metrics"]["applied_count"] == 1
    row = json.loads(lenient.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert 0.0 < row["source_color_confidence"] < 1.0

    strict = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "strict", max_samples=1,
                       palette_swap_min_color_confidence=0.99)
    )
    metrics = strict.report["metrics"]
    assert metrics["applied_count"] == 0
    assert metrics["ineligibility_reason_counts"].get("low_color_confidence", 0) == 1


# --- Part F: current behavior remains available (no conservative filters) -----


def test_permissive_default_still_augments(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path)
    result = review_palette_swap(_review_config(dataset, manifest, tmp_path / "review"))
    assert result.report["metrics"]["applied_count"] == 8


# --- Part F: training report includes eligibility/apply stats ----------------


def test_summarize_dataset_palette_swap_stats(tmp_path: Path) -> None:
    dataset, manifest = _build_dataset(tmp_path)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    config = swap_config_from_review(_review_config(dataset, manifest, tmp_path / "unused"))
    stats = summarize_dataset_palette_swap(dataset, records, config)
    for key in (
        "eligible_count",
        "ineligible_count",
        "applied_count",
        "applied_rate_total",
        "applied_rate_eligible",
        "ineligibility_reason_counts",
        "target_family_counts",
        "source_to_target_matrix",
        "fallback_heuristic_rate",
        "material_conflict_drop_count",
    ):
        assert key in stats
    assert stats["applied_count"] == 8
    assert stats["applied_rate_eligible"] == 1.0


def test_red_flags_material_like_targets(tmp_path: Path) -> None:
    # Force many gray/gold/brown targets -> material-like target red flag.
    dataset, manifest = _build_dataset(tmp_path, count=12)
    result = review_palette_swap(
        _review_config(dataset, manifest, tmp_path / "review", max_samples=12,
                       palette_swap_families="gray,gold,brown")
    )
    names = {flag["name"] for flag in result.report["red_flags"]}
    assert "many_material_like_targets" in names


def test_red_flag_when_caption_prepending_occurs_while_disabled() -> None:
    rows = [
        {
            "applied": True,
            "eligible_for_palette_swap": True,
            "source_color_family": "green",
            "target_color_family": "blue",
            "category": "weapon",
            "roles_recolored": ["midtone"],
            "fallback_heuristic_used": False,
            "caption_change_kind": "prepended",
            "original_caption": "sword",
            "original_caption_color_families": [],
            "material_conflict_drop_count": 0,
            "mean_rgb_delta_visible": 0.1,
            "palette_entries_changed": 2,
            "visible_color_count_before": 3,
            "visible_color_count_after": 3,
        }
    ]
    metrics = aggregate_metrics(rows)
    names = {
        flag["name"]
        for flag in compute_red_flags(metrics, rows, {"palette_swap_no_caption_prepend": True})
    }
    assert "caption_prepending_nonzero_when_disabled" in names


def test_aggregate_metrics_empty() -> None:
    metrics = aggregate_metrics([])
    assert metrics["applied_count"] == 0
    assert metrics["applied_rate"] == 0.0
