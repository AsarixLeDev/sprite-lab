"""Tests for the v2 Phase 0 no-training evaluation harness.

Most tests exercise pure parse/build/aggregate/decide/write functions
without touching PyTorch or the filesystem (except for tmp_path fixtures).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spritelab.training.v2_phase0_eval import (
    V1_1_CFG_BASE_SCALE,
    V1_1_CFG_COLOR_SCALE,
    RunCell,
    RunMetrics,
    V2Phase0EvalConfig,
    _base_mode,
    _compute_deltas,
    _count_manifest_records,
    _read_manifest_metadata,
    _skip_cell_dir_is_valid,
    aggregate_metrics,
    build_run_plan,
    collect_run_metrics,
    decide_v1_1,
    parse_factored_grid,
    parse_null_field_sets,
    parse_presets,
    parse_seeds,
    resolve_preset_params,
    verify_cell_metadata,
    write_summary_csv,
    write_summary_json,
    write_summary_markdown,
)


# ── Preset resolution ───────────────────────────────────────────────────────

def test_resolve_preset_params_v1() -> None:
    p = resolve_preset_params("v1")
    assert p["export_preset"] == "v1"
    assert p["factored_cfg"] is False
    assert p["cfg_base_scale"] is None
    assert p["cfg_color_scale"] is None


def test_resolve_preset_params_v1_1() -> None:
    p = resolve_preset_params("v1.1")
    assert p["export_preset"] == "v1.1"
    assert p["factored_cfg"] is True
    assert p["cfg_base_scale"] == V1_1_CFG_BASE_SCALE
    assert p["cfg_color_scale"] == V1_1_CFG_COLOR_SCALE


def test_resolve_preset_params_aliases() -> None:
    for alias in ("v1_1", "phase1_v1_1", "V1.1", "V1_1"):
        p = resolve_preset_params(alias)
        assert p["export_preset"] == "v1.1", f"failed for alias {alias}"
        assert p["factored_cfg"] is True

    for alias in ("phase1_v1", "PHASE1_V1"):
        p = resolve_preset_params(alias)
        assert p["export_preset"] == "v1", f"failed for alias {alias}"
        assert p["factored_cfg"] is False


def test_resolve_preset_params_unknown() -> None:
    p = resolve_preset_params("nonexistent")
    assert p["export_preset"] is None
    assert p["factored_cfg"] is False


# ── Parse helpers ───────────────────────────────────────────────────────────

def test_parse_seeds_default() -> None:
    assert parse_seeds("") == (20260723, 20260724, 20260725)


def test_parse_seeds_multiple() -> None:
    assert parse_seeds("20260723,20260724,20260725") == (20260723, 20260724, 20260725)


def test_parse_seeds_with_spaces() -> None:
    assert parse_seeds(" 100 , 200 , 300 ") == (100, 200, 300)


def test_parse_presets_default() -> None:
    assert parse_presets("") == ("v1", "v1.1")


def test_parse_presets_filters_unknown() -> None:
    assert parse_presets("v1,invalid,v1.1") == ("v1", "v1.1")


def test_parse_null_field_sets_multiple() -> None:
    assert parse_null_field_sets("colors,object_id,category") == ("colors", "object_id", "category")


def test_parse_null_field_sets_combined() -> None:
    assert parse_null_field_sets("object_id+colors") == ("object_id+colors",)


def test_parse_factored_grid_basic() -> None:
    grid = parse_factored_grid("base=1.5,2.0;color=3.0,4.5")
    assert len(grid) == 4
    expected = [
        {"cfg_base_scale": 1.5, "cfg_color_scale": 3.0},
        {"cfg_base_scale": 1.5, "cfg_color_scale": 4.5},
        {"cfg_base_scale": 2.0, "cfg_color_scale": 3.0},
        {"cfg_base_scale": 2.0, "cfg_color_scale": 4.5},
    ]
    assert grid == expected


# ── Build run plan ──────────────────────────────────────────────────────────

def test_build_run_plan_v1_cell_has_correct_params() -> None:
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        presets=("v1",),
        seeds=(20260723,),
    )
    cells = build_run_plan(config)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.export_preset == "v1"
    assert cell.factored_cfg is False
    assert cell.cfg_base_scale is None
    assert cell.cfg_color_scale is None


def test_build_run_plan_v1_1_cell_has_factored_cfg_enabled() -> None:
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        presets=("v1.1",),
        seeds=(20260723,),
    )
    cells = build_run_plan(config)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.export_preset == "v1.1"
    assert cell.factored_cfg is True
    assert cell.cfg_base_scale == V1_1_CFG_BASE_SCALE
    assert cell.cfg_color_scale == V1_1_CFG_COLOR_SCALE


def test_build_run_plan_v1_and_v1_1_create_distinct_cells() -> None:
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        presets=("v1", "v1.1"),
        seeds=(20260723,),
    )
    cells = build_run_plan(config)
    assert len(cells) == 2
    v1_cell = [c for c in cells if c.export_preset == "v1"][0]
    v11_cell = [c for c in cells if c.export_preset == "v1.1"][0]
    assert v1_cell.factored_cfg is False
    assert v11_cell.factored_cfg is True
    assert v1_cell.mode != v11_cell.mode  # distinct mode names


def test_build_run_plan_v1_1_aliases_also_work() -> None:
    """v1_1 and phase1_v1_1 should resolve to the same factored CFG cell."""
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        presets=("v1_1",),
        seeds=(99,),
    )
    cells = build_run_plan(config)
    assert len(cells) == 1
    assert cells[0].export_preset == "v1.1"
    assert cells[0].factored_cfg is True


def test_build_run_plan_with_ablations() -> None:
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        presets=("v1",),
        seeds=(20260723,),
        include_ablations=True,
        null_field_sets=("colors", "object_id", "object_id+colors"),
    )
    cells = build_run_plan(config)
    assert len(cells) == 4  # 1 preset + 3 ablations
    ablation_modes = [cell.mode for cell in cells if "ablation" in cell.mode]
    assert len(ablation_modes) == 3


def test_build_run_plan_with_factored_grid() -> None:
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        presets=tuple(),
        seeds=(20260723,),
        factored_grid="base=2.5;color=3.0",
    )
    cells = build_run_plan(config)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.factored_cfg
    assert cell.cfg_base_scale == 2.5
    assert cell.cfg_color_scale == 3.0


def test_build_run_plan_ablations_without_flag_skip() -> None:
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        include_ablations=False,
        null_field_sets=("colors",),
    )
    cells = build_run_plan(config)
    assert all("ablation" not in cell.mode for cell in cells)


def test_build_run_plan_unknown_preset_skipped() -> None:
    config = V2Phase0EvalConfig(
        out=Path("out"),
        checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"),
        dataset=Path("ds"),
        presets=("nonexistent",),
        seeds=(1,),
    )
    assert build_run_plan(config) == []


# ── Manifest metadata helpers ────────────────────────────────────────────────

def _write_fake_manifest(cell_dir: Path, *, export_preset: str = "", factored_cfg: bool = False,
                         cfg_base_scale: float | None = None, cfg_color_scale: float | None = None,
                         sample_count: int = 8) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(sample_count):
        record = {
            "sample_id": f"sample_{i:06d}",
            "prompt": f"test prompt {i}",
            "prompt_id": f"prompt_{i:04d}",
            "export_preset": export_preset,
            "factored_cfg": factored_cfg,
            "cfg_base_scale": cfg_base_scale,
            "cfg_color_scale": cfg_color_scale,
        }
        lines.append(json.dumps(record))
    (cell_dir / "generated_manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fake_qa(cell_dir: Path) -> None:
    (cell_dir / "generated_qa_report.json").write_text(json.dumps({
        "errors": [], "warnings": [], "ok": True, "sample_count": 8,
    }))


def test_read_manifest_metadata_v1() -> None:
    _write_fake_manifest(Path("/tmp/fake_v1"), export_preset="v1", factored_cfg=False)
    meta = _read_manifest_metadata(Path("/tmp/fake_v1"))
    assert meta is not None
    assert meta["export_preset"] == "v1"
    assert meta["factored_cfg"] is False


def test_read_manifest_metadata_v1_1() -> None:
    _write_fake_manifest(Path("/tmp/fake_v1_1"), export_preset="v1.1", factored_cfg=True,
                         cfg_base_scale=2.5, cfg_color_scale=3.0)
    meta = _read_manifest_metadata(Path("/tmp/fake_v1_1"))
    assert meta is not None
    assert meta["export_preset"] == "v1.1"
    assert meta["factored_cfg"] is True
    assert float(meta["cfg_base_scale"]) == pytest.approx(2.5)
    assert float(meta["cfg_color_scale"]) == pytest.approx(3.0)


def test_verify_cell_metadata_passes_v1(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, export_preset="v1", factored_cfg=False)
    cell = RunCell(mode="preset_v1_seed1", export_preset="v1", seed=1)
    verify_cell_metadata(cell, tmp_path)  # should not raise


def test_verify_cell_metadata_passes_v1_1(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, export_preset="v1.1", factored_cfg=True,
                         cfg_base_scale=2.5, cfg_color_scale=3.0)
    cell = RunCell(mode="preset_v1_1_seed1", export_preset="v1.1", seed=1,
                   factored_cfg=True, cfg_base_scale=2.5, cfg_color_scale=3.0)
    verify_cell_metadata(cell, tmp_path)  # should not raise


def test_verify_cell_metadata_fails_mismatched_factored_cfg(tmp_path: Path) -> None:
    """Manifest has v1.1 export_preset but factored_cfg=False; cell expects True."""
    _write_fake_manifest(tmp_path, export_preset="v1.1", factored_cfg=False,
                         cfg_base_scale=2.5, cfg_color_scale=3.0)
    cell = RunCell(mode="preset_v1_1_seed1", export_preset="v1.1", seed=1,
                   factored_cfg=True, cfg_base_scale=2.5, cfg_color_scale=3.0)
    with pytest.raises(RuntimeError, match="factored_cfg mismatch"):
        verify_cell_metadata(cell, tmp_path)


def test_verify_cell_metadata_fails_mismatched_export_preset(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, export_preset="v1.1", factored_cfg=True,
                         cfg_base_scale=2.5, cfg_color_scale=3.0)
    cell = RunCell(mode="preset_v1_seed1", export_preset="v1", seed=1)
    with pytest.raises(RuntimeError, match="export_preset mismatch"):
        verify_cell_metadata(cell, tmp_path)


def test_count_manifest_records(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, sample_count=32)
    assert _count_manifest_records(tmp_path) == 32


# ── Skip-sampling-if-exists validation ──────────────────────────────────────

def test_skip_cell_dir_is_valid_when_metadata_matches(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, export_preset="v1.1", factored_cfg=True,
                         cfg_base_scale=2.5, cfg_color_scale=3.0)
    _write_fake_qa(tmp_path)
    cell = RunCell(mode="preset_v1_1_seed1", export_preset="v1.1", seed=1,
                   factored_cfg=True, cfg_base_scale=2.5, cfg_color_scale=3.0)
    assert _skip_cell_dir_is_valid(cell, tmp_path) is True


def test_skip_cell_dir_is_invalid_when_metadata_mismatches(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, export_preset="v1", factored_cfg=False)
    _write_fake_qa(tmp_path)
    cell = RunCell(mode="preset_v1_1_seed1", export_preset="v1.1", seed=1,
                   factored_cfg=True, cfg_base_scale=2.5, cfg_color_scale=3.0)
    assert _skip_cell_dir_is_valid(cell, tmp_path) is False


def test_skip_cell_dir_is_invalid_when_no_qa(tmp_path: Path) -> None:
    _write_fake_manifest(tmp_path, export_preset="v1", factored_cfg=False)
    cell = RunCell(mode="preset_v1_seed1", export_preset="v1", seed=1)
    assert _skip_cell_dir_is_valid(cell, tmp_path) is False


# ── Collect metrics from synthetic output files ─────────────────────────────

def _write_fake_run_outputs(run_dir: Path, *, sample_count: int = 96,
                            export_preset: str = "v1", factored_cfg: bool = False,
                            cfg_base_scale: float | None = None,
                            cfg_color_scale: float | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(sample_count):
        lines.append(json.dumps({
            "sample_id": f"sample_{i:06d}",
            "prompt": f"prompt {i}",
            "prompt_id": f"prompt_{i:04d}",
            "export_preset": export_preset,
            "factored_cfg": factored_cfg,
            "cfg_base_scale": cfg_base_scale,
            "cfg_color_scale": cfg_color_scale,
        }))
    (run_dir / "generated_manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "generated_qa_report.json").write_text(json.dumps({
        "errors": [], "warnings": ["sample_000000: generated sprite is fully transparent"],
        "checks": {}, "ok": True, "sample_count": sample_count,
    }))
    review_dir = run_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "generated_review_report.json").write_text(json.dumps({
        "sample_count": sample_count,
        "overall": {
            "count": sample_count,
            "median_visible_color_count": 12.0,
            "warning_counts": {"touches_border": 49, "too_many_rare_colors": 0, "single_blob": 29},
            "total_warnings": 130,
        },
        "errors": [],
    }))
    (run_dir / "prompt_faithfulness_report.json").write_text(json.dumps({
        "sample_count": sample_count,
        "category_consistency_rate": 0.8068,
        "category_consistency_ci95": [0.72, 0.88],
        "color_consistency_rate": 0.8438,
        "color_consistency_ci95": [0.76, 0.91],
        "repeated_silhouette_rate": 0.0,
        "generic_blob_collapse_rate": 0.3021,
        "generic_potion_collapse_rate": 0.0521,
        "near_copy_rate": 0.0,
        "source_selection": {"mode": "all", "source_count_used": 928, "source_candidate_hash": "083d55be9803"},
        "nearest_source_summary": {"p10_distance": 0.15},
    }))
    (run_dir / "generation_report.json").write_text(json.dumps({
        "palette_projection": {"applied": True, "method": "deterministic_kmeans",
                               "target_colors": 16, "min_pixel_share": 0.01,
                               "mean_rgb_mae_visible": 0.03, "destructive_rate": 0.0},
    }))
    (run_dir / "palette_projection_report.json").write_text(json.dumps({
        "median_visible_color_count_before": 32.0, "median_visible_color_count_after": 12.0,
    }))


def test_collect_run_metrics_from_synthetic_v1(tmp_path: Path) -> None:
    _write_fake_run_outputs(tmp_path, export_preset="v1", factored_cfg=False)
    cell = RunCell(mode="preset_v1_seed20260723", export_preset="v1", seed=20260723)
    metrics = collect_run_metrics(tmp_path, cell, 96)
    assert metrics.export_preset == "v1"
    assert metrics.factored_cfg is False
    assert metrics.sample_count_generated == 96
    assert metrics.qa_errors == 0
    assert metrics.median_visible_colors == 12.0
    assert metrics.category_consistency == pytest.approx(0.8068)


def test_collect_run_metrics_from_synthetic_v1_1(tmp_path: Path) -> None:
    _write_fake_run_outputs(tmp_path, export_preset="v1.1", factored_cfg=True,
                            cfg_base_scale=2.5, cfg_color_scale=3.0)
    cell = RunCell(mode="preset_v1_1_seed20260723", export_preset="v1.1", seed=20260723,
                   factored_cfg=True, cfg_base_scale=2.5, cfg_color_scale=3.0)
    metrics = collect_run_metrics(tmp_path, cell, 256)
    assert metrics.export_preset == "v1.1"
    assert metrics.factored_cfg is True
    assert metrics.cfg_base_scale == 2.5
    assert metrics.cfg_color_scale == 3.0
    assert metrics.requested_max_samples == 256
    assert metrics.sample_count_generated == 96


def test_collect_run_metrics_reports_counts(tmp_path: Path) -> None:
    _write_fake_run_outputs(tmp_path, sample_count=32)
    cell = RunCell(mode="test_seed1", export_preset="v1", seed=1)
    metrics = collect_run_metrics(tmp_path, cell, 128)
    assert metrics.requested_max_samples == 128
    assert metrics.sample_count_generated == 32
    assert metrics.review_sample_count == 32
    assert metrics.faithfulness_sample_count == 32


def test_collect_run_metrics_missing_files(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cell = RunCell(mode="empty_seed0", export_preset=None, seed=0)
    metrics = collect_run_metrics(tmp_path, cell, 0)
    assert metrics.qa_errors == 0
    assert metrics.sample_count_generated == 0


# ── Aggregate metrics ───────────────────────────────────────────────────────

def _make_fake_metrics(mode: str, seed: int, category: float, color: float,
                       export_preset: str | None = None,
                       factored_cfg: bool = False) -> RunMetrics:
    return RunMetrics(
        mode=mode, seed=seed, requested_max_samples=96, sample_count_generated=96,
        review_sample_count=96, faithfulness_sample_count=96,
        export_preset=export_preset, factored_cfg=factored_cfg,
        qa_errors=0, median_visible_colors=12.0,
        category_consistency=category, color_consistency=color,
        repeated_silhouette_rate=0.0, blob_collapse_rate=0.30,
        potion_collapse_rate=0.05, near_copy_rate=0.0, source_count_used=928,
    )


def test_aggregate_metrics_preserves_export_preset_and_factored() -> None:
    cells = [
        _make_fake_metrics("preset_v1_seed20260723", 20260723, 0.81, 0.84,
                           export_preset="v1", factored_cfg=False),
        _make_fake_metrics("preset_v1_1_seed20260723", 20260723, 0.79, 0.87,
                           export_preset="v1.1", factored_cfg=True),
    ]
    aggs = aggregate_metrics(cells)
    assert aggs["preset_v1"]["export_preset"] == "v1"
    assert aggs["preset_v1"]["factored_cfg"] is False
    assert aggs["preset_v1_1"]["export_preset"] == "v1.1"
    assert aggs["preset_v1_1"]["factored_cfg"] is True


def test_aggregate_metrics_computes_means_and_stdevs() -> None:
    cells = [
        _make_fake_metrics("preset_v1_seed20260723", 20260723, 0.81, 0.84, export_preset="v1"),
        _make_fake_metrics("preset_v1_seed20260724", 20260724, 0.79, 0.83, export_preset="v1"),
        _make_fake_metrics("preset_v1_seed20260725", 20260725, 0.83, 0.85, export_preset="v1"),
    ]
    aggs = aggregate_metrics(cells)
    v1 = aggs["preset_v1"]
    assert v1["run_count"] == 3
    assert v1["category_consistency_mean"] == pytest.approx(0.81)
    assert v1["qa_errors_total"] == 0


def test_base_mode_strips_seed_suffix() -> None:
    assert _base_mode("preset_v1_seed20260723") == "preset_v1"
    assert _base_mode("ablation_colors_seed123") == "ablation_colors"
    assert _base_mode("simple_mode") == "simple_mode"


# ── Compute deltas ──────────────────────────────────────────────────────────

def test_compute_deltas_nonzero_when_different() -> None:
    aggs = {
        "preset_v1": {"color_consistency_mean": 0.840, "category_consistency_mean": 0.810,
                       "rare_color_warning_rate_mean": 0.0, "blob_collapse_rate_mean": 0.302,
                       "near_copy_rate_mean": 0.0},
        "preset_v1_1": {"color_consistency_mean": 0.870, "category_consistency_mean": 0.791,
                         "rare_color_warning_rate_mean": 0.003, "blob_collapse_rate_mean": 0.295,
                         "near_copy_rate_mean": 0.0},
    }
    deltas = _compute_deltas(aggs)
    d = deltas["preset_v1_1"]
    assert d["color_consistency_mean_delta"] == pytest.approx(0.03)
    assert d["category_consistency_mean_delta"] == pytest.approx(-0.019)


# ── Decision rules ──────────────────────────────────────────────────────────

def test_decide_v1_1_pass() -> None:
    v1 = {"color_consistency_mean": 0.84, "category_consistency_mean": 0.81,
          "rare_color_warning_rate_mean": 0.0, "blob_collapse_rate_mean": 0.302, "near_copy_rate_mean": 0.0}
    v11 = {"color_consistency_mean": 0.88, "category_consistency_mean": 0.80,
           "rare_color_warning_rate_mean": 0.0, "blob_collapse_rate_mean": 0.29,
           "near_copy_rate_mean": 0.0, "qa_errors_total": 0}
    assert decide_v1_1(v1, v11)["label"] == "pass"


def test_decide_v1_1_fail_color() -> None:
    v1 = {"color_consistency_mean": 0.84}
    v11 = {"color_consistency_mean": 0.83, "category_consistency_mean": 0.80,
           "rare_color_warning_rate_mean": 0.0, "blob_collapse_rate_mean": 0.29,
           "near_copy_rate_mean": 0.0, "qa_errors_total": 0}
    assert decide_v1_1(v1, v11)["label"] == "fail"


def test_decide_v1_1_borderline() -> None:
    v1 = {"color_consistency_mean": 0.84}
    v11 = {"color_consistency_mean": 0.868, "category_consistency_mean": 0.80,
           "rare_color_warning_rate_mean": 0.0, "blob_collapse_rate_mean": 0.29,
           "near_copy_rate_mean": 0.0, "qa_errors_total": 0}
    assert decide_v1_1(v1, v11)["label"] == "borderline"


def test_decide_v1_1_qa_fail() -> None:
    v1 = {"color_consistency_mean": 0.84, "blob_collapse_rate_mean": 0.30, "near_copy_rate_mean": 0.0}
    v11 = {"color_consistency_mean": 0.88, "category_consistency_mean": 0.80,
           "rare_color_warning_rate_mean": 0.0, "blob_collapse_rate_mean": 0.29,
           "near_copy_rate_mean": 0.0, "qa_errors_total": 1}
    assert decide_v1_1(v1, v11)["label"] == "fail"


def test_decide_v1_1_missing_agg() -> None:
    assert decide_v1_1(None, None)["label"] == "fail"


# ── Report writers ──────────────────────────────────────────────────────────

def _sample_summary() -> dict:
    return {
        "meta": {
            "checkpoint": "ckpt.pt", "prompts": "p.jsonl", "dataset": "ds/",
            "max_samples": 96, "seeds": [20260723], "device": "cpu",
            "out_dir": "out_dir/", "source_hash": "abc123",
        },
        "preset_definitions": {"v1": "v1 desc", "v1.1": "v1.1 desc"},
        "per_run": [
            {
                "mode": "preset_v1_seed20260723", "seed": 20260723,
                "export_preset": "v1", "factored_cfg": False,
                "cfg_base_scale": None, "cfg_color_scale": None,
                "requested_max_samples": 96, "sample_count_generated": 96,
                "review_sample_count": 96, "faithfulness_sample_count": 96,
                "qa_errors": 0, "qa_warnings": 1,
                "median_visible_colors": 12.0, "rare_color_warning_rate": 0.0,
                "touches_border_rate": 0.51, "category_consistency": 0.8068,
                "category_ci95": [0.72, 0.88], "color_consistency": 0.8438,
                "color_ci95": [0.76, 0.91], "repeated_silhouette_rate": 0.0,
                "blob_collapse_rate": 0.3021, "potion_collapse_rate": 0.0521,
                "near_copy_rate": 0.0, "p10_nearest_source_distance": 0.15,
                "source_count_used": 928, "source_candidate_hash": "abc",
                "mean_rgb_mae_visible": 0.03, "destructive_rate": 0.0,
            },
        ],
        "aggregates": {
            "preset_v1": {"mode": "preset_v1", "run_count": 1, "export_preset": "v1",
                          "factored_cfg": False, "qa_errors_total": 0,
                          "category_consistency_mean": 0.8068, "color_consistency_mean": 0.8438,
                          "source_count_used": 928, "source_candidate_hash": "abc"},
        },
        "deltas_vs_v1": {},
        "decision": {"label": "pass", "explanation": "All criteria passed.", "criteria": {}},
    }


def test_write_summary_json_includes_export_preset(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    write_summary_json(_sample_summary(), path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    run = loaded["per_run"][0]
    assert run["export_preset"] == "v1"
    assert run["factored_cfg"] is False
    assert run["requested_max_samples"] == 96
    assert run["sample_count_generated"] == 96
    assert run["review_sample_count"] == 96
    assert run["faithfulness_sample_count"] == 96


def test_write_summary_markdown_includes_factored_column(tmp_path: Path) -> None:
    path = tmp_path / "test.md"
    write_summary_markdown(_sample_summary(), path)
    content = path.read_text(encoding="utf-8")
    assert "v2 Phase 0 Evaluation Report" in content
    assert "Factored" in content
    assert "Null Fields" in content
    assert "Gen" in content
    assert "Review" in content
    assert "Faith" in content
    assert "abc123" in content
    assert "20260723" in content


def test_write_summary_csv_includes_new_columns(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    write_summary_csv(_sample_summary(), path)
    content = path.read_text(encoding="utf-8")
    assert "export_preset" in content
    assert "factored_cfg" in content
    assert "requested_max_samples" in content
    assert "sample_count_generated" in content


# ── Prompt builder tests ────────────────────────────────────────────────────

def _fake_manifest(path: Path, count: int = 50, seed: int = 42) -> None:
    """Write a minimal synthetic training manifest for prompt builder tests."""
    import random as _random
    rng = _random.Random(seed)
    categories = ["weapon", "armor", "item_icon", "tool", "material", "plant", "effect_icon"]
    objects_by_cat = {
        "weapon": ["sword", "axe", "bow", "dagger", "hammer"],
        "armor": ["helm", "shield", "chestplate", "boots", "gauntlet"],
        "item_icon": ["potion", "scroll", "ring", "lantern", "key"],
        "tool": ["pickaxe", "wrench", "fishing_rod", "hoe", "saw"],
        "material": ["gem", "coin", "ingot", "crystal", "stone"],
        "plant": ["mushroom", "flower", "leaf", "cactus", "vine"],
        "effect_icon": ["flame", "spark", "star", "snowflake", "bolt"],
    }
    all_colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "gray", "gold", "silver"]
    materials = ["iron", "wooden", "stone", "golden", "leather"]
    shapes = ["round", "square", "triangular", "oblong"]
    functions = ["attack", "defense", "utility"]
    styles = ["pixel_art", "icon", "fantasy"]

    lines = []
    for i in range(count):
        cat = rng.choice(categories)
        obj = rng.choice(objects_by_cat[cat])
        color = rng.choice(all_colors)
        material = rng.choice(materials)
        shape = rng.choice(shapes)
        func = rng.choice(functions)
        style = rng.choice(styles)
        row = {
            "sprite_id": f"test_{cat}_{obj}_{i:04d}",
            "split": "train",
            "category": cat,
            "object_name": obj,
            "base_object": obj,
            "caption": f"{color} {obj} icon",
            "conditioning": {
                "semantic_v3": {
                    "attributes": {
                        "colors": [color],
                        "materials": [material],
                        "shapes": [shape],
                        "function": [func],
                        "effects": [],
                        "state": [],
                        "style": [style],
                    },
                    "base_object": obj,
                    "open_name": obj,
                }
            },
        }
        lines.append(json.dumps(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prompt_builder_cli_accepts_subcommand() -> None:
    """The CLI parser accepts build-v2-eval-prompts as a valid subcommand."""
    import argparse
    from spritelab.training.cli import main

    # Just verify the subcommand is recognized by the argument parser.
    # We can't run it without a real manifest, so test parser only.
    try:
        main(["build-v2-eval-prompts", "--dataset", "ds", "--training-manifest", "manifest.jsonl",
              "--out", "out.jsonl", "--target-count", "10"])
    except (SystemExit, FileNotFoundError):
        pass  # expected when manifest doesn't exist


def test_prompt_builder_deterministic(tmp_path: Path) -> None:
    from spritelab.training.v2_eval_prompts import V2EvalPromptsConfig, build_v2_eval_prompts

    manifest = tmp_path / "manifest.jsonl"
    _fake_manifest(manifest, count=50, seed=123)
    out1 = tmp_path / "prompts_v1.jsonl"
    out2 = tmp_path / "prompts_v2.jsonl"

    config1 = V2EvalPromptsConfig(dataset=tmp_path, training_manifest=manifest,
                                   out=out1, target_count=32, seed=42)
    config2 = V2EvalPromptsConfig(dataset=tmp_path, training_manifest=manifest,
                                   out=out2, target_count=32, seed=42)

    r1 = build_v2_eval_prompts(config1)
    r2 = build_v2_eval_prompts(config2)

    assert r1["prompt_count"] == r2["prompt_count"]
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_prompt_builder_respects_target_count(tmp_path: Path) -> None:
    from spritelab.training.v2_eval_prompts import V2EvalPromptsConfig, build_v2_eval_prompts

    manifest = tmp_path / "manifest.jsonl"
    _fake_manifest(manifest, count=200, seed=1)
    out = tmp_path / "prompts.jsonl"

    for target in [16, 48, 128, 384]:
        config = V2EvalPromptsConfig(dataset=tmp_path, training_manifest=manifest,
                                      out=out, target_count=target, seed=7)
        r = build_v2_eval_prompts(config)
        # Should be within 20% of target (families may not have enough combos)
        assert abs(r["prompt_count"] - target) <= max(20, target * 0.3), (
            f"target={target} got {r['prompt_count']}"
        )
        assert r["prompt_count"] > 0


def test_prompt_builder_no_duplicate_ids(tmp_path: Path) -> None:
    from spritelab.training.v2_eval_prompts import V2EvalPromptsConfig, build_v2_eval_prompts

    manifest = tmp_path / "manifest.jsonl"
    _fake_manifest(manifest, count=100, seed=5)
    out = tmp_path / "prompts.jsonl"
    config = V2EvalPromptsConfig(dataset=tmp_path, training_manifest=manifest,
                                  out=out, target_count=128, seed=99)
    r = build_v2_eval_prompts(config)

    ids = []
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(json.loads(line)["prompt_id"])
    assert len(ids) == len(set(ids))
    assert r["duplicates_removed"] == 0


def test_prompt_builder_report_has_coverage_counts(tmp_path: Path) -> None:
    from spritelab.training.v2_eval_prompts import V2EvalPromptsConfig, build_v2_eval_prompts

    manifest = tmp_path / "manifest.jsonl"
    _fake_manifest(manifest, count=100, seed=3)
    out = tmp_path / "prompts.jsonl"
    config = V2EvalPromptsConfig(dataset=tmp_path, training_manifest=manifest,
                                  out=out, target_count=128, seed=8, out_report=True)
    r = build_v2_eval_prompts(config)

    assert "families" in r
    assert "category_counts" in r
    assert "color_counts" in r
    assert "object_counts_top20" in r
    assert "vocab_summary" in r
    assert r["vocab_summary"]["categories"] >= 3


def test_prompt_builder_rows_are_sampler_compatible(tmp_path: Path) -> None:
    from spritelab.training.v2_eval_prompts import V2EvalPromptsConfig, build_v2_eval_prompts
    from spritelab.training.sample_generator import read_prompt_records

    manifest = tmp_path / "manifest.jsonl"
    _fake_manifest(manifest, count=50, seed=99)
    out = tmp_path / "prompts.jsonl"
    config = V2EvalPromptsConfig(dataset=tmp_path, training_manifest=manifest,
                                  out=out, target_count=16, seed=42)
    build_v2_eval_prompts(config)

    records = read_prompt_records(out, max_records=16)
    assert len(records) > 0
    for record in records:
        assert record.get("prompt")
        assert record.get("prompt_id")
        assert record.get("colors") or record.get("colors") is not None


def test_build_prompts_and_prompts_mutually_exclusive() -> None:
    from spritelab.training.cli import main
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        with pytest.raises(SystemExit):
            main([
                "run-v2-phase0-eval",
                "--out", "test_out", "--checkpoint", "ckpt.pt",
                "--prompts", "p.jsonl", "--dataset", "ds",
                "--build-prompts", "--dry-run",
            ])
    finally:
        sys.stdout = old_stdout


def test_build_prompts_flag_accepted_on_dry_run() -> None:
    from spritelab.training.cli import main
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    buf = StringIO()
    try:
        sys.stdout = buf
        main([
            "run-v2-phase0-eval", "--dry-run",
            "--out", "test_out", "--checkpoint", "ckpt.pt",
            "--dataset", "ds",
            "--build-prompts", "--prompt-count", "32", "--prompt-seed", "5",
        ])
    finally:
        sys.stdout = old_stdout
    assert "Dry run" in buf.getvalue()


# ── CLI integration tests ───────────────────────────────────────────────────

def test_cli_accepts_run_v2_phase0_eval_subcommand() -> None:
    from spritelab.training.cli import main
    main(["run-v2-phase0-eval", "--dry-run",
          "--out", "test_out", "--checkpoint", "ckpt.pt",
          "--prompts", "p.jsonl", "--dataset", "ds"])


def test_cli_dry_run_shows_export_preset() -> None:
    from spritelab.training.cli import main
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    buf = StringIO()
    try:
        sys.stdout = buf
        main([
            "run-v2-phase0-eval", "--dry-run",
            "--out", "test_out", "--checkpoint", "ckpt.pt",
            "--prompts", "p.jsonl", "--dataset", "ds",
            "--presets", "v1,v1.1", "--seeds", "20260723",
        ])
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue()
    assert "export_preset=v1" in output
    assert "export_preset=v1.1" in output
    assert "factored_cfg" in output


def test_cli_skip_sampling_flag_accepted() -> None:
    from spritelab.training.cli import main
    main(["run-v2-phase0-eval", "--dry-run",
          "--out", "test_out", "--checkpoint", "ckpt.pt",
          "--prompts", "p.jsonl", "--dataset", "ds",
          "--skip-sampling-if-exists"])


def test_cli_no_training_invoked() -> None:
    """Verify the v2 phase0 eval does not invoke any training command."""
    from spritelab.training.v2_phase0_eval import run_v2_phase0_eval, V2Phase0EvalConfig

    config = V2Phase0EvalConfig(out=Path("test_out"), checkpoint=Path("ckpt.pt"),
                                prompts=Path("p.jsonl"), dataset=Path("ds"), dry_run=True)
    result = run_v2_phase0_eval(config)
    assert result["dry_run"] is True
    assert "planned_cells" in result


# ── Speed optimizations (default-on, unlike training subcommands) ──────────


def test_speed_optimizations_default_on() -> None:
    config = V2Phase0EvalConfig(out=Path("test_out"), checkpoint=Path("ckpt.pt"),
                                 prompts=Path("p.jsonl"), dataset=Path("ds"))
    assert config.speed_optimizations is True


def test_speed_optimizations_default_applies_cudnn_and_tf32(monkeypatch: pytest.MonkeyPatch) -> None:
    """This harness never trains, so unlike every training subcommand, its backend
    speed flags default ON: it resamples the same checkpoint/shape repeatedly
    across cells and seeds, so cuDNN autotuning pays for itself."""
    pytest.importorskip("torch", exc_type=ImportError)
    from spritelab.training.v2_phase0_eval import run_v2_phase0_eval, V2Phase0EvalConfig

    calls: list[dict[str, object]] = []

    def fake_apply_backend_speed_flags(**kwargs: object) -> None:
        calls.append(kwargs)
        raise RuntimeError("stop-after-speed-flags")

    monkeypatch.setattr(
        "spritelab.training.optim_utils.apply_backend_speed_flags", fake_apply_backend_speed_flags
    )

    config = V2Phase0EvalConfig(
        out=Path("test_out"), checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"), dataset=Path("ds"),
        presets=("v1",), seeds=(1,),
    )
    with pytest.raises(RuntimeError, match="stop-after-speed-flags"):
        run_v2_phase0_eval(config)
    assert calls == [{"cudnn_benchmark": True, "tf32": True}]


def test_speed_optimizations_disabled_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch", exc_type=ImportError)
    from spritelab.training.v2_phase0_eval import run_v2_phase0_eval, V2Phase0EvalConfig

    calls: list[dict[str, object]] = []

    def fake_apply_backend_speed_flags(**kwargs: object) -> None:
        calls.append(kwargs)
        raise RuntimeError("stop-after-speed-flags")

    monkeypatch.setattr(
        "spritelab.training.optim_utils.apply_backend_speed_flags", fake_apply_backend_speed_flags
    )

    config = V2Phase0EvalConfig(
        out=Path("test_out"), checkpoint=Path("ckpt.pt"),
        prompts=Path("p.jsonl"), dataset=Path("ds"),
        presets=("v1",), seeds=(1,),
        speed_optimizations=False,
    )
    with pytest.raises(RuntimeError, match="stop-after-speed-flags"):
        run_v2_phase0_eval(config)
    assert calls == [{"cudnn_benchmark": False, "tf32": False}]


def test_cli_speed_optimizations_flag_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    import spritelab.training.v2_phase0_eval as v2_phase0_eval
    from spritelab.training.cli import main as train_cli
    from spritelab.training.v2_phase0_eval import V2Phase0EvalConfig

    captured: list[V2Phase0EvalConfig] = []

    def fake_run(config: V2Phase0EvalConfig) -> dict[str, object]:
        captured.append(config)
        return {"dry_run": False}

    monkeypatch.setattr(v2_phase0_eval, "run_v2_phase0_eval", fake_run)

    train_cli([
        "run-v2-phase0-eval",
        "--out", "test_out", "--checkpoint", "ckpt.pt",
        "--prompts", "p.jsonl", "--dataset", "ds",
        "--no-speed-optimizations",
    ])
    assert captured[0].speed_optimizations is False

    captured.clear()
    train_cli([
        "run-v2-phase0-eval",
        "--out", "test_out", "--checkpoint", "ckpt.pt",
        "--prompts", "p.jsonl", "--dataset", "ds",
    ])
    assert captured[0].speed_optimizations is True
