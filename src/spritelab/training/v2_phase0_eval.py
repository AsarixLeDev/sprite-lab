"""Reproducible v2 Phase 0 no-training evaluation harness.

Invoke via:
    python -m spritelab train run-v2-phase0-eval [...args]

This module orchestrates sampling + QA + review + prompt-faithfulness for
preset and ablation comparison cells, collects metrics from JSON output
files, aggregates across seeds, and writes summary reports. Never trains.

All pure functions (parse, build, aggregate, decide, write) are importable
without PyTorch so they can be unit-tested on CPU-only / headless CI.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# v1.1 factored-CFG constants. Must match generator_challenger.V1_1_CFG_BASE_SCALE
# and V1_1_CFG_COLOR_SCALE so the harness gives identical results to the CLI
# --export-preset v1.1 path.
V1_1_CFG_BASE_SCALE = 2.5
V1_1_CFG_COLOR_SCALE = 3.0

V1_PRESET_NAMES: tuple[str, ...] = ("v1", "phase1_v1")
V1_1_PRESET_NAMES: tuple[str, ...] = ("v1.1", "v1_1", "phase1_v1_1")

# Must match generator_challenger.NULL_FIELD_CHOICES
NULL_FIELD_CHOICES: tuple[str, ...] = (
    "caption",
    "semantic",
    "category",
    "object_id",
    "base_object",
    "colors",
    "materials",
    "shapes",
    "function",
    "style",
    "structured",
)

# ── Eval profiles ───────────────────────────────────────────────────────────

OOD_CORE_FAMILIES: tuple[str, ...] = (
    "object_color_pairs",
    "rare_combos",
    "style_stress",
)

OOD_PLUS_GRID_FAMILIES: tuple[str, ...] = (
    "category_color_grid",
    "object_color_pairs",
    "rare_combos",
    "style_stress",
)

ALL_PROFILE_FAMILIES: tuple[str, ...] = ()

EVAL_PROFILES: dict[str, tuple[str, ...]] = {
    "all": ALL_PROFILE_FAMILIES,
    "ood_core": OOD_CORE_FAMILIES,
    "ood_plus_grid": OOD_PLUS_GRID_FAMILIES,
}

EXCLUDED_FAMILY_ANCHORS: str = "in_distribution_anchors"


# ── Config dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class V2Phase0EvalConfig:
    out: Path
    checkpoint: Path
    prompts: Path | None  # optional when build_prompts=True
    dataset: Path
    presets: tuple[str, ...] = ("v1", "v1.1")
    seeds: tuple[int, ...] = (20260723, 20260724, 20260725)
    max_samples: int = 96
    device: str = "cpu"
    batch_size: int = 16
    include_ablations: bool = False
    null_field_sets: tuple[str, ...] = ()
    factored_grid: str = ""
    skip_sampling_if_exists: bool = False
    faithfulness_max_sources: int = 0
    no_contact_sheets: bool = False
    dry_run: bool = False
    build_prompts: bool = False
    prompt_count: int = 384
    prompt_seed: int = 20260706
    report_only: bool = False
    allow_partial_report: bool = False
    # This harness never trains (no optimizer/EMA), so only the backend conv/matmul
    # flags from optim_utils.apply_backend_speed_flags apply; unlike every training
    # subcommand, they default ON here since sampling repeats the same fixed
    # checkpoint/shape across many cells and seeds. --no-speed-optimizations restores
    # the plain (numerically stricter) path.
    speed_optimizations: bool = True
    eval_profile: str = "all"
    profile_weighting: str = "family"


@dataclass(frozen=True)
class RunCell:
    """One reproducible evaluation cell (preset / ablation / factored-grid)."""

    mode: str
    export_preset: str | None  # "v1" or "v1.1"; None for raw/ablation cells
    seed: int
    null_fields: str = ""  # comma-separated for ChallengerSampleConfig
    null_field_set: str = ""  # original --null-field-sets group token (e.g. "object_id+colors")
    factored_cfg: bool = False
    cfg_base_scale: float | None = None
    cfg_color_scale: float | None = None


@dataclass
class RunMetrics:
    mode: str = ""
    seed: int = 0
    requested_max_samples: int = 0
    sample_count_generated: int = 0
    review_sample_count: int = 0
    faithfulness_sample_count: int = 0
    out_dir: Path = field(default_factory=Path)
    # Sampling metadata
    export_preset: str | None = None
    factored_cfg: bool = False
    cfg_base_scale: float | None = None
    cfg_color_scale: float | None = None
    null_fields: str = ""
    null_field_set: str = ""
    # QA
    qa_errors: int = 0
    qa_warnings: int = 0
    # Review
    median_visible_colors: float | None = None
    rare_color_warning_rate: float | None = None
    touches_border_rate: float | None = None
    # Faithfulness
    category_consistency: float | None = None
    category_ci95: list[float] | None = None
    color_consistency: float | None = None
    color_ci95: list[float] | None = None
    repeated_silhouette_rate: float | None = None
    blob_collapse_rate: float | None = None
    potion_collapse_rate: float | None = None
    near_copy_rate: float | None = None
    p10_nearest_source_distance: float | None = None
    source_count_used: int = 0
    source_candidate_hash: str = ""
    # Projection
    median_colors_before: float | None = None
    median_colors_after: float | None = None
    mean_rgb_mae_visible: float | None = None
    destructive_rate: float | None = None


# ── Preset resolution (must match CLI _apply_export_preset_defaults) ────────


def resolve_preset_params(preset: str) -> dict[str, Any]:
    """Return the canonical export_preset + factored-CFG settings for a preset name."""
    normalized = str(preset).strip().lower()
    if normalized in V1_PRESET_NAMES:
        return {
            "export_preset": "v1",
            "factored_cfg": False,
            "cfg_base_scale": None,
            "cfg_color_scale": None,
        }
    if normalized in V1_1_PRESET_NAMES:
        return {
            "export_preset": "v1.1",
            "factored_cfg": True,
            "cfg_base_scale": V1_1_CFG_BASE_SCALE,
            "cfg_color_scale": V1_1_CFG_COLOR_SCALE,
        }
    return {
        "export_preset": None,
        "factored_cfg": False,
        "cfg_base_scale": None,
        "cfg_color_scale": None,
    }


# ── Null-field helpers (pure, testable) ─────────────────────────────────────


def parse_null_field_set_group(group: str) -> list[str]:
    """Split a '+' delimited group into individual field names.

    >>> parse_null_field_set_group("object_id+colors")
    ["object_id", "colors"]
    """
    parts = [token.strip() for token in str(group).split("+") if token.strip()]
    if not parts:
        return []
    # Validate each field
    unknown = [p for p in parts if p not in NULL_FIELD_CHOICES]
    if unknown:
        raise ValueError(f"Unknown null-field value(s): {unknown!r}; expected only: {NULL_FIELD_CHOICES}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for field in parts:
        if field not in seen:
            seen.add(field)
            result.append(field)
    return result


def parse_null_field_sets(null_field_sets: str) -> tuple[str, ...]:
    """Parse comma-separated group tokens (each token may be '+' delimited).

    Returns the raw group tokens as strings for use in config.  The
    expansion into individual field lists happens in ``build_run_plan``
    via ``parse_null_field_set_group``.

    >>> parse_null_field_sets("colors,object_id+colors,caption+semantic")
    ("colors", "object_id+colors", "caption+semantic")
    """
    if not null_field_sets or not str(null_field_sets).strip():
        return tuple()
    return tuple(token.strip() for token in str(null_field_sets).split(",") if token.strip())


def null_field_group_slug(fields: list[str]) -> str:
    """Build a deterministic, filesystem-safe slug for a null-field group.

    >>> null_field_group_slug(["object_id", "colors"])
    "object_id_plus_colors"
    """
    return "_plus_".join(fields)


def _sanitize_dirname(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


# ── Other parse helpers ─────────────────────────────────────────────────────


def parse_seeds(seeds: str) -> tuple[int, ...]:
    parts = [token.strip() for token in str(seeds).split(",") if token.strip()]
    if not parts:
        return (20260723, 20260724, 20260725)
    return tuple(int(part) for part in parts)


def parse_presets(presets: str) -> tuple[str, ...]:
    parts = [token.strip().lower() for token in str(presets).split(",") if token.strip()]
    if not parts:
        return ("v1", "v1.1")
    all_known = {*V1_PRESET_NAMES, *V1_1_PRESET_NAMES}
    return tuple(part for part in parts if part in all_known)


def parse_factored_grid(grid: str) -> list[dict[str, float]]:
    if not grid or not str(grid).strip():
        return []
    base_scales: list[float] = []
    color_scales: list[float] = []
    for segment in str(grid).split(";"):
        segment = segment.strip()
        if segment.startswith("base="):
            base_scales = [float(v) for v in segment[len("base=") :].split(",") if v.strip()]
        elif segment.startswith("color="):
            color_scales = [float(v) for v in segment[len("color=") :].split(",") if v.strip()]
    if not base_scales or not color_scales:
        return []
    result: list[dict[str, float]] = []
    for b in base_scales:
        for c in color_scales:
            result.append({"cfg_base_scale": b, "cfg_color_scale": c})
    return result


# ── Build run plan (pure, testable) ─────────────────────────────────────────


def build_run_plan(config: V2Phase0EvalConfig) -> list[RunCell]:
    cells: list[RunCell] = []

    for preset in config.presets:
        params = resolve_preset_params(preset)
        ep = params["export_preset"]
        if ep is None:
            continue
        for seed in config.seeds:
            short = ep.replace(".", "_").replace("-", "_")
            cells.append(
                RunCell(
                    mode=f"preset_{short}_seed{seed}",
                    export_preset=ep,
                    seed=seed,
                    factored_cfg=bool(params["factored_cfg"]),
                    cfg_base_scale=params["cfg_base_scale"],
                    cfg_color_scale=params["cfg_color_scale"],
                )
            )

    if config.include_ablations and config.null_field_sets:
        for nfs_token in config.null_field_sets:
            fields = parse_null_field_set_group(nfs_token)
            slug = null_field_group_slug(fields)
            comma_fields = ",".join(fields)
            for seed in config.seeds:
                cells.append(
                    RunCell(
                        mode=f"ablation_{slug}_seed{seed}",
                        export_preset=None,
                        seed=seed,
                        null_fields=comma_fields,
                        null_field_set=nfs_token,
                    )
                )

    if config.factored_grid:
        grid_entries = parse_factored_grid(config.factored_grid)
        for entry in grid_entries:
            b = entry["cfg_base_scale"]
            c = entry["cfg_color_scale"]
            for seed in config.seeds:
                b_str = str(b).replace(".", "_")
                c_str = str(c).replace(".", "_")
                cells.append(
                    RunCell(
                        mode=f"factored_base{b_str}_color{c_str}_seed{seed}",
                        export_preset=None,
                        seed=seed,
                        factored_cfg=True,
                        cfg_base_scale=b,
                        cfg_color_scale=c,
                    )
                )

    return cells


# ── Manifest metadata verification ──────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_manifest_metadata(cell_dir: Path) -> dict[str, Any] | None:
    """Return the sampling metadata from the first manifest record, or None."""
    manifest_path = cell_dir / "generated_manifest.jsonl"
    if not manifest_path.is_file():
        return None
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        return {
            "export_preset": record.get("export_preset") or "",
            "factored_cfg": bool(record.get("factored_cfg")),
            "cfg_base_scale": record.get("cfg_base_scale"),
            "cfg_color_scale": record.get("cfg_color_scale"),
            "null_fields": record.get("null_fields") or [],
            "sample_count": 1,
        }
    return None


def _count_manifest_records(cell_dir: Path) -> int:
    manifest_path = cell_dir / "generated_manifest.jsonl"
    if not manifest_path.is_file():
        return 0
    count = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def _normalize_null_set(value: Any) -> set[str]:
    """Normalize null_fields from manifest (list or string) to a set."""
    if isinstance(value, list):
        return set(str(v) for v in value if v)
    if isinstance(value, str):
        return set(v.strip() for v in value.split(",") if v.strip())
    return set()


def verify_cell_metadata(cell: RunCell, cell_dir: Path) -> None:
    """Verify sampled outputs match the expected cell configuration.

    Raises RuntimeError if the manifest metadata contradicts the cell definition.
    """
    meta = _read_manifest_metadata(cell_dir)
    if meta is None:
        return

    expected_ep = cell.export_preset or ""
    actual_ep = str(meta.get("export_preset") or "")

    if expected_ep and actual_ep and actual_ep != expected_ep:
        raise RuntimeError(
            f"Manifest export_preset mismatch in {cell_dir}: expected {expected_ep!r}, got {actual_ep!r}"
        )

    expected_fc = bool(cell.factored_cfg)
    actual_fc = bool(meta.get("factored_cfg"))
    if expected_fc != actual_fc:
        raise RuntimeError(f"Manifest factored_cfg mismatch in {cell_dir}: expected {expected_fc}, got {actual_fc}")

    if expected_fc:
        expected_base = cell.cfg_base_scale
        actual_base = meta.get("cfg_base_scale")
        if expected_base is not None and actual_base is not None:
            if abs(float(expected_base) - float(actual_base)) > 1e-6:
                raise RuntimeError(
                    f"Manifest cfg_base_scale mismatch in {cell_dir}: expected {expected_base}, got {actual_base}"
                )

        expected_color = cell.cfg_color_scale
        actual_color = meta.get("cfg_color_scale")
        if expected_color is not None and actual_color is not None:
            if abs(float(expected_color) - float(actual_color)) > 1e-6:
                raise RuntimeError(
                    f"Manifest cfg_color_scale mismatch in {cell_dir}: expected {expected_color}, got {actual_color}"
                )

    # Verify null-fields match
    if cell.null_fields:
        expected_set = _normalize_null_set(cell.null_fields)
        actual_set = _normalize_null_set(meta.get("null_fields"))
        if expected_set and actual_set and expected_set != actual_set:
            raise RuntimeError(
                f"Manifest null_fields mismatch in {cell_dir}: "
                f"expected {sorted(expected_set)}, got {sorted(actual_set)}"
            )


def _skip_cell_dir_is_valid(cell: RunCell, cell_dir: Path) -> bool:
    """Return True if the cell_dir contains outputs that match *cell*."""
    meta = _read_manifest_metadata(cell_dir)
    if meta is None:
        return False

    expected_ep = cell.export_preset or ""
    actual_ep = str(meta.get("export_preset") or "")
    if expected_ep and actual_ep != expected_ep:
        return False

    if bool(cell.factored_cfg) != bool(meta.get("factored_cfg")):
        return False

    if cell.factored_cfg:
        if cell.cfg_base_scale is not None:
            actual_base = meta.get("cfg_base_scale")
            if actual_base is not None and abs(float(cell.cfg_base_scale) - float(actual_base)) > 1e-6:
                return False
        if cell.cfg_color_scale is not None:
            actual_color = meta.get("cfg_color_scale")
            if actual_color is not None and abs(float(cell.cfg_color_scale) - float(actual_color)) > 1e-6:
                return False

    # Verify null-fields match
    if cell.null_fields:
        expected_set = _normalize_null_set(cell.null_fields)
        actual_set = _normalize_null_set(meta.get("null_fields"))
        if expected_set and actual_set and expected_set != actual_set:
            return False

    # Must have QA report to be considered complete
    qa_path = cell_dir / "generated_qa_report.json"
    if not qa_path.is_file():
        return False

    return True


def _cell_outputs_exist(cell_dir: Path) -> bool:
    """Check if a cell directory has the minimum outputs for report harvesting."""
    return (
        (cell_dir / "generated_manifest.jsonl").is_file()
        and (cell_dir / "generated_qa_report.json").is_file()
        and (cell_dir / "prompt_faithfulness_report.json").is_file()
    )


# ── Collect metrics from run output files ────────────────────────────────────


def collect_run_metrics(
    run_dir: Path,
    cell: RunCell,
    requested_max_samples: int,
) -> RunMetrics:
    metrics = RunMetrics(
        mode=cell.mode,
        seed=cell.seed,
        requested_max_samples=requested_max_samples,
        out_dir=run_dir,
        export_preset=cell.export_preset,
        factored_cfg=cell.factored_cfg,
        cfg_base_scale=cell.cfg_base_scale,
        cfg_color_scale=cell.cfg_color_scale,
        null_fields=cell.null_fields,
        null_field_set=cell.null_field_set,
    )

    metrics.sample_count_generated = _count_manifest_records(run_dir)

    qa_path = run_dir / "generated_qa_report.json"
    qa = _read_json(qa_path)
    if qa:
        metrics.qa_errors = len(qa.get("errors", []))
        metrics.qa_warnings = len(qa.get("warnings", []))

    review_path = run_dir / "review" / "generated_review_report.json"
    review = _read_json(review_path)
    if not review:
        review_path = run_dir / "generated_review_report.json"
        review = _read_json(review_path)
    if review:
        overall = review.get("overall") if isinstance(review.get("overall"), dict) else {}
        warning_counts = overall.get("warning_counts") if isinstance(overall.get("warning_counts"), dict) else {}
        sample_count = max(1, int(review.get("sample_count") or 0))
        metrics.review_sample_count = int(review.get("sample_count") or 0)
        metrics.median_visible_colors = overall.get("median_visible_color_count")
        metrics.rare_color_warning_rate = warning_counts.get("too_many_rare_colors", 0) / sample_count
        metrics.touches_border_rate = warning_counts.get("touches_border", 0) / sample_count

    faith_path = run_dir / "prompt_faithfulness_report.json"
    faith = _read_json(faith_path)
    if faith:
        metrics.faithfulness_sample_count = int(faith.get("sample_count") or 0)
        metrics.category_consistency = faith.get("category_consistency_rate")
        metrics.category_ci95 = faith.get("category_consistency_ci95")
        metrics.color_consistency = faith.get("color_consistency_rate")
        metrics.color_ci95 = faith.get("color_consistency_ci95")
        metrics.repeated_silhouette_rate = faith.get("repeated_silhouette_rate")
        metrics.blob_collapse_rate = faith.get("generic_blob_collapse_rate")
        metrics.potion_collapse_rate = faith.get("generic_potion_collapse_rate")
        metrics.near_copy_rate = faith.get("near_copy_rate")
        nearest = faith.get("nearest_source_summary")
        if isinstance(nearest, dict):
            metrics.p10_nearest_source_distance = nearest.get("p10_distance")
        selection = faith.get("source_selection")
        if isinstance(selection, dict):
            metrics.source_count_used = int(selection.get("source_count_used", 0))
            metrics.source_candidate_hash = str(selection.get("source_candidate_hash", ""))

    gen_path = run_dir / "generation_report.json"
    gen = _read_json(gen_path)
    if gen:
        proj = gen.get("palette_projection")
        if isinstance(proj, dict):
            metrics.mean_rgb_mae_visible = proj.get("mean_rgb_mae_visible")
            metrics.destructive_rate = proj.get("destructive_rate")
        proj_detail = _read_json(run_dir / "palette_projection_report.json")
        if proj_detail:
            metrics.median_colors_before = proj_detail.get("median_visible_color_count_before")
            metrics.median_colors_after = proj_detail.get("median_visible_color_count_after")

    return metrics


# ── Aggregate metrics across seeds ──────────────────────────────────────────


def _safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(statistics.fmean(vals)) if vals else None


def _safe_stdev(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(statistics.pstdev(vals)) if len(vals) >= 2 else 0.0


def _safe_min(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(min(vals)) if vals else None


def _safe_max(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(max(vals)) if vals else None


def _aggregate_cells(name: str, cells: list[RunMetrics]) -> dict[str, Any]:
    count = len(cells)
    first = cells[0] if cells else None
    return {
        "mode": name,
        "seeds": [cell.seed for cell in cells],
        "run_count": count,
        "export_preset": first.export_preset if first else None,
        "factored_cfg": first.factored_cfg if first else False,
        "cfg_base_scale": first.cfg_base_scale if first else None,
        "cfg_color_scale": first.cfg_color_scale if first else None,
        "null_fields": first.null_fields if first else "",
        "requested_max_samples": first.requested_max_samples if first else 0,
        "sample_count_generated": _safe_mean([float(c.sample_count_generated) for c in cells]) if cells else None,
        "qa_errors_mean": _safe_mean([float(c.qa_errors) for c in cells]),
        "qa_errors_total": sum(c.qa_errors for c in cells),
        "qa_warnings_mean": _safe_mean([float(c.qa_warnings) for c in cells]),
        "median_visible_colors_mean": _safe_mean(
            [c.median_visible_colors for c in cells if c.median_visible_colors is not None]
        ),
        "median_visible_colors_std": _safe_stdev(
            [c.median_visible_colors for c in cells if c.median_visible_colors is not None]
        ),
        "median_visible_colors_min": _safe_min(
            [c.median_visible_colors for c in cells if c.median_visible_colors is not None]
        ),
        "median_visible_colors_max": _safe_max(
            [c.median_visible_colors for c in cells if c.median_visible_colors is not None]
        ),
        "rare_color_warning_rate_mean": _safe_mean(
            [c.rare_color_warning_rate for c in cells if c.rare_color_warning_rate is not None]
        ),
        "rare_color_warning_rate_std": _safe_stdev(
            [c.rare_color_warning_rate for c in cells if c.rare_color_warning_rate is not None]
        ),
        "touches_border_rate_mean": _safe_mean(
            [c.touches_border_rate for c in cells if c.touches_border_rate is not None]
        ),
        "touches_border_rate_std": _safe_stdev(
            [c.touches_border_rate for c in cells if c.touches_border_rate is not None]
        ),
        "category_consistency_mean": _safe_mean(
            [c.category_consistency for c in cells if c.category_consistency is not None]
        ),
        "category_consistency_std": _safe_stdev(
            [c.category_consistency for c in cells if c.category_consistency is not None]
        ),
        "category_consistency_min": _safe_min(
            [c.category_consistency for c in cells if c.category_consistency is not None]
        ),
        "category_consistency_max": _safe_max(
            [c.category_consistency for c in cells if c.category_consistency is not None]
        ),
        "color_consistency_mean": _safe_mean([c.color_consistency for c in cells if c.color_consistency is not None]),
        "color_consistency_std": _safe_stdev([c.color_consistency for c in cells if c.color_consistency is not None]),
        "color_consistency_min": _safe_min([c.color_consistency for c in cells if c.color_consistency is not None]),
        "color_consistency_max": _safe_max([c.color_consistency for c in cells if c.color_consistency is not None]),
        "repeated_silhouette_rate_mean": _safe_mean(
            [c.repeated_silhouette_rate for c in cells if c.repeated_silhouette_rate is not None]
        ),
        "repeated_silhouette_rate_std": _safe_stdev(
            [c.repeated_silhouette_rate for c in cells if c.repeated_silhouette_rate is not None]
        ),
        "blob_collapse_rate_mean": _safe_mean(
            [c.blob_collapse_rate for c in cells if c.blob_collapse_rate is not None]
        ),
        "blob_collapse_rate_std": _safe_stdev(
            [c.blob_collapse_rate for c in cells if c.blob_collapse_rate is not None]
        ),
        "potion_collapse_rate_mean": _safe_mean(
            [c.potion_collapse_rate for c in cells if c.potion_collapse_rate is not None]
        ),
        "potion_collapse_rate_std": _safe_stdev(
            [c.potion_collapse_rate for c in cells if c.potion_collapse_rate is not None]
        ),
        "near_copy_rate_mean": _safe_mean([c.near_copy_rate for c in cells if c.near_copy_rate is not None]),
        "near_copy_rate_std": _safe_stdev([c.near_copy_rate for c in cells if c.near_copy_rate is not None]),
        "p10_nearest_source_distance_mean": _safe_mean(
            [c.p10_nearest_source_distance for c in cells if c.p10_nearest_source_distance is not None]
        ),
        "source_count_used": cells[0].source_count_used if cells else 0,
        "source_candidate_hash": cells[0].source_candidate_hash if cells else "",
        "mean_rgb_mae_visible_mean": _safe_mean(
            [c.mean_rgb_mae_visible for c in cells if c.mean_rgb_mae_visible is not None]
        ),
        "destructive_rate_mean": _safe_mean([c.destructive_rate for c in cells if c.destructive_rate is not None]),
    }


def aggregate_metrics(all_cells: list[RunMetrics]) -> dict[str, Any]:
    by_mode: dict[str, list[RunMetrics]] = {}
    for cell in all_cells:
        base_mode = _base_mode(cell.mode)
        by_mode.setdefault(base_mode, []).append(cell)

    aggregates: dict[str, Any] = {}
    for mode, cells in sorted(by_mode.items()):
        aggregates[mode] = _aggregate_cells(mode, cells)

    return aggregates


def _base_mode(mode: str) -> str:
    parts = mode.rsplit("_seed", 1)
    return parts[0] if len(parts) == 2 else mode


# ── Decision rules ──────────────────────────────────────────────────────────


def decide_v1_1(v1_agg: dict[str, Any] | None, v1_1_agg: dict[str, Any] | None) -> dict[str, Any]:
    if v1_agg is None or v1_1_agg is None:
        return {
            "label": "fail",
            "explanation": "Missing v1 or v1.1 aggregate; cannot decide.",
            "criteria": {},
        }

    v1_color = v1_agg.get("color_consistency_mean")
    v1_cat = v1_agg.get("category_consistency_mean")
    v1_rare = v1_agg.get("rare_color_warning_rate_mean")
    v1_blob = v1_agg.get("blob_collapse_rate_mean")
    v1_near = v1_agg.get("near_copy_rate_mean")

    v11_color = v1_1_agg.get("color_consistency_mean")
    v11_cat = v1_1_agg.get("category_consistency_mean")
    v11_rare = v1_1_agg.get("rare_color_warning_rate_mean")
    v11_blob = v1_1_agg.get("blob_collapse_rate_mean")
    v11_near = v1_1_agg.get("near_copy_rate_mean")
    v11_qa = v1_1_agg.get("qa_errors_total", -1)

    def _ok(value: Any) -> bool:
        return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))

    criteria: dict[str, dict[str, Any]] = {}
    passed = 0
    failed = 0
    borderline = 0

    def _check(name: str, actual: Any, target: Any, *, comparison: str) -> str:
        nonlocal passed, failed, borderline
        if not _ok(actual) or not _ok(target):
            criteria[name] = {"actual": actual, "target": target, "status": "unknown"}
            return "unknown"
        actual_v = float(actual)
        target_v = float(target)
        criteria[name] = {"actual": actual_v, "target": target_v, "status": "checking"}
        if comparison == ">=":
            ok = actual_v >= target_v
        elif comparison == "<=":
            ok = actual_v <= target_v
        elif comparison == "==":
            ok = actual_v == target_v
        else:
            ok = False
        margin = abs(actual_v - target_v)
        if not ok:
            if margin < 0.025:
                borderline += 1
                criteria[name]["status"] = "borderline"
                return "borderline"
            failed += 1
            criteria[name]["status"] = "fail"
            return "fail"
        passed += 1
        criteria[name]["status"] = "pass"
        return "pass"

    if _ok(v1_color) and _ok(v11_color):
        _check("color_delta", v11_color, float(v1_color) + 0.03, comparison=">=")
    if _ok(v1_cat) and _ok(v11_cat):
        _check("category_delta", v11_cat, float(v1_cat) - 0.02, comparison=">=")
    _check("rare_rate", v11_rare, 0.01, comparison="<=")
    if _ok(v1_blob) and _ok(v11_blob):
        _check("blob_delta", v11_blob, v1_blob, comparison="<=")
    if _ok(v1_near) and _ok(v11_near):
        _check("near_copy_delta", v11_near, float(v1_near) + 0.01, comparison="<=")
    if isinstance(v11_qa, (int, float)):
        _check("qa_errors", v11_qa, 0, comparison="==")

    label = "pass"
    explanation = ""
    if failed > 0:
        label = "fail"
        explanation = f"{failed} criterion(s) failed."
    elif borderline > 0 and passed > 0:
        label = "borderline"
        explanation = f"{borderline} criterion(s) borderline within CI/noise margin."
    elif passed >= len(criteria):
        explanation = "All criteria passed."

    return {"label": label, "explanation": explanation, "criteria": criteria}


def _profile_family_filter(families: tuple[str, ...]) -> tuple[str, ...]:
    """Return the effective family filter for an eval profile."""
    if not families:
        return tuple()
    return families


# ── Profile aggregate helpers ───────────────────────────────────────────────


def compute_profile_aggregates(
    breakdowns: dict[str, Any],
    profile: str,
    weighting: str,
    all_prompt_aggregates: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-profile aggregates from breakdown data.

    Uses family tables to compute profile-filtered metrics.
    Returns a dict of {mode_name: {metric_name: value}}.
    """
    families = EVAL_PROFILES.get(profile, ALL_PROFILE_FAMILIES)
    family_tbl = breakdowns.get("by_family") or []
    if not family_tbl or not families:
        return {}

    # Filter to selected families
    filtered = [r for r in family_tbl if r.get("prompt_family") in families]

    result: dict[str, dict[str, Any]] = {}
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        m = str(row.get("_mode", ""))
        by_mode.setdefault(m, []).append(row)

    for mode, rows in sorted(by_mode.items()):
        if weighting == "family":
            # Family-weighted: average per-family means equally
            cat_vals: list[float] = []
            col_vals: list[float] = []
            blob_vals: list[float] = []
            for r in rows:
                v = r.get("category_consistency")
                if v is not None:
                    cat_vals.append(float(v))
                v = r.get("color_consistency")
                if v is not None:
                    col_vals.append(float(v))
                blob_vals.append(float(r.get("blob_collapse_rate", 0.0)))
            result[mode] = {
                "category_consistency_mean_weighted": _safe_mean(cat_vals) if cat_vals else None,
                "color_consistency_mean_weighted": _safe_mean(col_vals) if col_vals else None,
                "blob_collapse_rate_mean_weighted": _safe_mean(blob_vals) if blob_vals else None,
                "families_used": list({str(r.get("prompt_family")) for r in rows}),
                "sample_count_profiled": sum(int(r.get("sample_count", 0)) for r in rows),
            }
        else:
            # Sample-weighted
            total_n = sum(int(r.get("sample_count", 0)) for r in rows)
            if total_n == 0:
                result[mode] = {
                    "category_consistency_mean_weighted": None,
                    "color_consistency_mean_weighted": None,
                    "blob_collapse_rate_mean_weighted": None,
                    "families_used": [],
                    "sample_count_profiled": 0,
                }
                continue
            w_cat = 0.0
            w_col = 0.0
            w_blob = 0.0
            for r in rows:
                n = int(r.get("sample_count", 0))
                v = r.get("category_consistency")
                if v is not None:
                    w_cat += float(v) * n
                v = r.get("color_consistency")
                if v is not None:
                    w_col += float(v) * n
                w_blob += float(r.get("blob_collapse_rate", 0.0)) * n
            result[mode] = {
                "category_consistency_mean_weighted": w_cat / total_n if total_n else None,
                "color_consistency_mean_weighted": w_col / total_n if total_n else None,
                "blob_collapse_rate_mean_weighted": w_blob / total_n if total_n else None,
                "families_used": list({str(r.get("prompt_family")) for r in rows}),
                "sample_count_profiled": total_n,
            }

    return result


def decide_ood_core(
    baseline_agg: dict[str, Any] | None,
    candidate_agg: dict[str, Any] | None,
    all_prompt_agg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """OOD-core profile decision for candidate vs baseline.

    Labels: pass, borderline, fail, color_tradeoff, not_applicable.
    """
    if baseline_agg is None or candidate_agg is None or not baseline_agg or not candidate_agg:
        return {"label": "not_applicable", "explanation": "Missing profile data.", "criteria": {}}

    b_cat = baseline_agg.get("category_consistency_mean_weighted")
    b_col = baseline_agg.get("color_consistency_mean_weighted")
    b_blob = baseline_agg.get("blob_collapse_rate_mean_weighted")

    c_cat = candidate_agg.get("category_consistency_mean_weighted")
    c_col = candidate_agg.get("color_consistency_mean_weighted")
    c_blob = candidate_agg.get("blob_collapse_rate_mean_weighted")

    def _ok(v: Any) -> bool:
        return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))

    criteria: dict[str, dict[str, Any]] = {}
    passed = 0
    failed = 0
    borderline = 0
    checks = 0

    def _check(name: str, actual: Any, target: Any, *, comparison: str) -> str:
        nonlocal passed, failed, borderline, checks
        checks += 1
        if not _ok(actual) or not _ok(target):
            criteria[name] = {"actual": actual, "target": target, "status": "unknown"}
            return "unknown"
        av = float(actual)
        tv = float(target)
        criteria[name] = {"actual": av, "target": tv, "status": "checking"}
        if comparison == ">=":
            ok = av >= tv
        elif comparison == "<=":
            ok = av <= tv
        elif comparison == "==":
            ok = av == tv
        else:
            ok = False
        margin = abs(av - tv)
        if not ok:
            if margin < 0.025:
                borderline += 1
                criteria[name]["status"] = "borderline"
                return "borderline"
            failed += 1
            criteria[name]["status"] = "fail"
            return "fail"
        passed += 1
        criteria[name]["status"] = "pass"
        return "pass"

    if _ok(b_cat) and _ok(c_cat):
        _check("ood_category", c_cat, float(b_cat) + 0.03, comparison=">=")
    if _ok(b_col) and _ok(c_col):
        _check("ood_color", c_col, float(b_col) + 0.03, comparison=">=")
    if _ok(b_blob) and _ok(c_blob):
        _check("ood_blob", c_blob, b_blob, comparison="<=")

    # Detect color_tradeoff: color improves >= 0.03 but category drops > 0.02
    if _ok(b_cat) and _ok(c_cat) and _ok(b_col) and _ok(c_col):
        cat_drop = float(b_cat) - float(c_cat)
        col_gain = float(c_col) - float(b_col)
        if col_gain >= 0.03 and cat_drop > 0.02 and failed == 0:
            criteria["color_tradeoff"] = {
                "color_gain": col_gain,
                "category_drop": cat_drop,
                "status": "info",
            }
            return {
                "label": "color_tradeoff",
                "explanation": (
                    f"Color improved by {col_gain:.4f} but category dropped by {cat_drop:.4f}. "
                    f"This is a color-control tradeoff, not a categorical improvement."
                ),
                "criteria": criteria,
            }

    if failed > 0:
        label = "fail"
        expl = f"{failed} criterion(s) failed."
    elif borderline > 0 and passed > 0:
        label = "borderline"
        expl = f"{borderline} criterion(s) borderline within CI/noise margin."
    elif passed >= checks:
        label = "pass"
        expl = "All OOD-core criteria passed."
    else:
        label = "not_applicable"
        expl = f"Checks: {checks}, passed: {passed}"

    return {"label": label, "explanation": expl, "criteria": criteria}


def _load_prompt_lookup(prompts_path: Path) -> dict[str, dict[str, Any]]:
    """Return a dict mapping prompt_id to {family, category, color, object, base_object}."""
    lookup: dict[str, dict[str, Any]] = {}
    if not prompts_path.is_file():
        return lookup
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        pid = str(row.get("prompt_id") or "").strip()
        if not pid:
            continue
        colors = row.get("colors") or []
        lookup[pid] = {
            "prompt_family": str(row.get("prompt_family") or ""),
            "category": str(row.get("category") or ""),
            "color": str(colors[0]) if colors else "",
            "object_name": str(row.get("object_name") or ""),
            "base_object": str(row.get("base_object") or ""),
        }
    return lookup


def _category_color_key(category: str, color: str) -> str:
    return f"{category}_{color}" if category and color else (category or color or "unknown")


def _compute_breakdowns(
    cells: list[RunMetrics],
    prompts_path: Path,
    aggregates: dict[str, Any],
    summaries_dir: Path,
) -> dict[str, Any]:
    """Compute per-group breakdown metrics and write CSV/JSON files.

    Returns a dict with keys: by_family, by_category, by_color,
    by_category_color, by_object, by_base_object.
    """
    prompt_lookup = _load_prompt_lookup(prompts_path)
    if not prompt_lookup:
        print("  Note: no prompt metadata available for breakdowns")
        return {}

    # Collect per-run sample-level data
    run_samples: list[dict[str, Any]] = []

    for cell in cells:
        run_dir = cell.out_dir
        faith_path = run_dir / "prompt_faithfulness_report.json"
        faith = _read_json(faith_path)
        if not faith:
            continue

        faith_samples = faith.get("samples") if isinstance(faith.get("samples"), list) else []
        faith_idx: dict[str, dict[str, Any]] = {}
        for s in faith_samples:
            pid = str(s.get("prompt_id") or "")
            if pid:
                faith_idx[pid] = dict(s)

        # Read generated manifest for sample_id matching
        manifest_path = run_dir / "generated_manifest.jsonl"
        manifest_by_pid: dict[str, dict[str, Any]] = {}
        gen_count = 0
        if manifest_path.is_file():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                pid = str(rec.get("prompt_id") or "").strip()
                if pid:
                    gen_count += 1
                    manifest_by_pid[pid] = dict(rec)

        for pid, prompt_meta in prompt_lookup.items():
            faith_sample = faith_idx.get(pid, {})
            row = {
                "run_mode": cell.mode,
                "seed": cell.seed,
                "prompt_family": prompt_meta["prompt_family"],
                "category": prompt_meta["category"],
                "color": prompt_meta["color"],
                "category_color": _category_color_key(prompt_meta["category"], prompt_meta["color"]),
                "object_name": prompt_meta["object_name"],
                "base_object": prompt_meta["base_object"],
                "category_consistent": faith_sample.get("category_consistent"),
                "color_consistent": faith_sample.get("color_consistent"),
                "generic_blob_like": faith_sample.get("generic_blob_like"),
                "sample_id": faith_sample.get("sample_id") or manifest_by_pid.get(pid, {}).get("sample_id", ""),
            }
            run_samples.append(row)

    if not run_samples:
        print("  Note: no per-sample data available for breakdowns")
        return {}

    # Group keys to compute breakdowns
    group_keys = ["prompt_family", "category", "color", "category_color", "object_name", "base_object"]
    group_names = {
        "prompt_family": "by_family",
        "category": "by_category",
        "color": "by_color",
        "category_color": "by_category_color",
        "object_name": "by_object",
        "base_object": "by_base_object",
    }

    aggregate_scalars = {k: v for k, v in aggregates.items()}

    result: dict[str, Any] = {}
    for gk in group_keys:
        tbl = _build_breakdown_table(run_samples, gk, aggregate_scalars)
        if tbl:
            out_key = group_names[gk]
            result[out_key] = tbl
            csv_path = summaries_dir / f"phase0_eval_breakdown_{out_key}.csv"
            _write_breakdown_csv(tbl, csv_path, gk, list_tbl_mode_keys(tbl))

    json_path = summaries_dir / "phase0_eval_breakdowns.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")

    return result


def list_tbl_mode_keys(tbl: list[dict[str, Any]]) -> list[str]:
    """Return the sorted mode names present in a breakdown table."""
    modes: set[str] = set()
    for row in tbl:
        m = row.get("_mode", "")
        if m:
            modes.add(m)
    return sorted(modes)


def _build_breakdown_table(
    samples: list[dict[str, Any]],
    group_key: str,
    aggregates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one breakdown table (list of rows) grouped by *group_key*.

    Each row is keyed by (base_mode, group_value).  Rows include sample_count
    and metric means per group.
    """
    # Group samples by (base_mode, group_value)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in samples:
        base = _base_mode(str(s.get("run_mode", "")))
        gv = str(s.get(group_key) or "").strip()
        if not gv:
            continue
        groups.setdefault((base, gv), []).append(s)

    rows: list[dict[str, Any]] = []
    for (mode, gv), group_samples in sorted(groups.items()):
        n = len(group_samples)
        row: dict[str, Any] = {
            "_mode": mode,
            group_key: gv,
            "sample_count": n,
            "category_consistency": _safe_mean(
                [1.0 for s in group_samples if s.get("category_consistent") is True]
                + [0.0 for s in group_samples if s.get("category_consistent") is False]
            ),
            "color_consistency": _safe_mean(
                [1.0 for s in group_samples if s.get("color_consistent") is True]
                + [0.0 for s in group_samples if s.get("color_consistent") is False]
            ),
            "blob_collapse_rate": sum(1 for s in group_samples if s.get("generic_blob_like")) / float(n) if n else 0.0,
        }
        # Delta vs preset_v1 baseline
        v1_agg = aggregates.get("preset_v1")
        if v1_agg and mode != "preset_v1":
            v1_cat = v1_agg.get("category_consistency_mean")
            v1_col = v1_agg.get("color_consistency_mean")
            v1_blob = v1_agg.get("blob_collapse_rate_mean")
            if row["category_consistency"] is not None and v1_cat is not None:
                row["delta_category"] = float(row["category_consistency"]) - float(v1_cat)
            else:
                row["delta_category"] = None
            if row["color_consistency"] is not None and v1_col is not None:
                row["delta_color"] = float(row["color_consistency"]) - float(v1_col)
            else:
                row["delta_color"] = None
            if row["blob_collapse_rate"] is not None and v1_blob is not None:
                row["delta_blob"] = float(row["blob_collapse_rate"]) - float(v1_blob)
            else:
                row["delta_blob"] = None
        else:
            row["delta_category"] = None
            row["delta_color"] = None
            row["delta_blob"] = None

        rows.append(row)

    return rows


def _write_breakdown_csv(
    rows: list[dict[str, Any]],
    path: Path,
    group_key: str,
    mode_order: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "_mode",
        group_key,
        "sample_count",
        "category_consistency",
        "color_consistency",
        "blob_collapse_rate",
        "delta_category",
        "delta_color",
        "delta_blob",
    ]
    header = ",".join(columns)
    lines = [header]
    for row in rows:
        values = [str(row.get(col, "")).replace(",", ";").replace("\n", " ") for col in columns]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Report writers ──────────────────────────────────────────────────────────


def _fmt(val: Any) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_run = summary.get("per_run") or []
    aggregates = summary.get("aggregates") or {}
    columns = [
        "mode",
        "seed",
        "export_preset",
        "factored_cfg",
        "cfg_base_scale",
        "cfg_color_scale",
        "null_fields",
        "null_field_set",
        "requested_max_samples",
        "sample_count_generated",
        "review_sample_count",
        "faithfulness_sample_count",
        "qa_errors",
        "qa_warnings",
        "median_visible_colors",
        "rare_color_warning_rate",
        "touches_border_rate",
        "category_consistency",
        "color_consistency",
        "repeated_silhouette_rate",
        "blob_collapse_rate",
        "potion_collapse_rate",
        "near_copy_rate",
        "p10_nearest_source_distance",
        "source_count_used",
        "source_candidate_hash",
        "mean_rgb_mae_visible",
        "destructive_rate",
    ]
    header = ",".join(columns)
    lines = [header]
    for run in per_run:
        values = [str(run.get(col, "")).replace(",", ";").replace("\n", " ") for col in columns]
        lines.append(",".join(values))
    for mode_name, agg in aggregates.items():
        agg_row = {**agg, "mode": f"{mode_name}_aggregate", "seed": ""}
        values = [str(agg_row.get(col, "")).replace(",", ";").replace("\n", " ") for col in columns]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = summary.get("meta") or {}
    per_run = summary.get("per_run") or []
    aggregates = summary.get("aggregates") or {}
    decision = summary.get("decision") or {}
    deltas = summary.get("deltas_vs_v1") or {}
    preset_info = summary.get("preset_definitions") or {}

    lines = [
        "# v2 Phase 0 Evaluation Report",
        "",
        "## Metadata",
        "",
        f"- Checkpoint: `{meta.get('checkpoint', '')}`",
        f"- Prompts: `{meta.get('prompts', '')}`",
        f"- Dataset: `{meta.get('dataset', '')}`",
        f"- Source hash: `{meta.get('source_hash', '')}`",
        f"- Requested max samples: {meta.get('max_samples', '')}",
        f"- Seeds: {', '.join(str(s) for s in (meta.get('seeds') or []))}",
        f"- Device: `{meta.get('device', '')}`",
        "",
    ]
    if meta.get("prompts_built"):
        lines.append(
            f"- Prompts built: {meta.get('prompt_count', '')} (target: {meta.get('prompt_count_requested', meta.get('prompt_count', ''))})"
        )
        lines.append("")
    else:
        lines.append("")

    if preset_info:
        lines.extend(
            [
                "## Preset Definitions",
                "",
                "| Preset | Description |",
                "|---|---|",
            ]
        )
        for name, desc in preset_info.items():
            lines.append(f"| `{name}` | {desc} |")
        lines.append("")

    if decision:
        lines.extend(
            [
                "## Decision Summary",
                "",
                f"**v1.1 promotion/confirmation: `{decision.get('label', '')}`**",
                f"{decision.get('explanation', '')}",
                "",
            ]
        )
        criteria = decision.get("criteria") or {}
        if criteria:
            lines.extend(
                [
                    "| Criterion | Actual | Target | Status |",
                    "|---|---|---|---|",
                ]
            )
            for name, info in sorted(criteria.items()):
                lines.append(
                    f"| {name} | {_fmt(info.get('actual'))} | {_fmt(info.get('target'))} | `{info.get('status', '')}` |"
                )
                lines.append("")

    # Profile decision summary
    profile_decs = summary.get("profile_decisions") or {}
    if profile_decs:
        for profile_name, pdec in profile_decs.items():
            lines.extend(
                [
                    f"### OOD-Core Decision (`{profile_name}`)",
                    "",
                    f"**Label: `{pdec.get('label', '')}`**",
                    f"{pdec.get('explanation', '')}",
                    "",
                ]
            )
            pc = pdec.get("criteria") or {}
            if pc:
                lines.extend(
                    [
                        "| Criterion | Actual | Target | Status |",
                        "|---|---|---|---|",
                    ]
                )
                for name, info in sorted(pc.items()):
                    lines.append(
                        f"| {name} | {_fmt(info.get('actual'))} | {_fmt(info.get('target'))} | "
                        f"`{info.get('status', '')}` |"
                    )
                lines.append("")

    lines.extend(
        [
            "## Per-Run Results",
            "",
            "| Mode | Seed | Preset | Factored | Null Fields | Gen | Review | Faith | QA Err | Colors | Rare % | Touch % | Category | Color | Repeated % | Blob % | Potion % | NearCopy % | p10 |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in per_run:
        fc_str = "yes" if run.get("factored_cfg") else "no"
        nf_str = str(run.get("null_fields") or run.get("null_field_set") or "")
        gen_str = str(run.get("sample_count_generated", ""))
        rev_str = str(run.get("review_sample_count", ""))
        faith_str = str(run.get("faithfulness_sample_count", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_escape(str(run.get('mode', '')))}`",
                    str(run.get("seed", "")),
                    str(run.get("export_preset") or ""),
                    fc_str,
                    _md_escape(nf_str),
                    gen_str,
                    rev_str,
                    faith_str,
                    str(run.get("qa_errors", "")),
                    _fmt(run.get("median_visible_colors")),
                    _fmt(_pct(run.get("rare_color_warning_rate"))),
                    _fmt(_pct(run.get("touches_border_rate"))),
                    _fmt(run.get("category_consistency")),
                    _fmt(run.get("color_consistency")),
                    _fmt(_pct(run.get("repeated_silhouette_rate"))),
                    _fmt(_pct(run.get("blob_collapse_rate"))),
                    _fmt(_pct(run.get("potion_collapse_rate"))),
                    _fmt(_pct(run.get("near_copy_rate"))),
                    _fmt(run.get("p10_nearest_source_distance")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Mode | Runs | Colors (mean +/- std) | Category (mean +/- std) | Color (mean +/- std) | Repeated % | Blob % | Potion % | NearCopy % | QA Err Total |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode_name, agg in sorted(aggregates.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md_escape(str(mode_name))}`",
                    str(agg.get("run_count", "")),
                    _fmt_mean_std(agg.get("median_visible_colors_mean"), agg.get("median_visible_colors_std")),
                    _fmt_mean_std(agg.get("category_consistency_mean"), agg.get("category_consistency_std")),
                    _fmt_mean_std(agg.get("color_consistency_mean"), agg.get("color_consistency_std")),
                    _fmt_mean_std(
                        _pct(agg.get("repeated_silhouette_rate_mean")), _pct(agg.get("repeated_silhouette_rate_std"))
                    ),
                    _fmt_mean_std(_pct(agg.get("blob_collapse_rate_mean")), _pct(agg.get("blob_collapse_rate_std"))),
                    _fmt_mean_std(
                        _pct(agg.get("potion_collapse_rate_mean")), _pct(agg.get("potion_collapse_rate_std"))
                    ),
                    _fmt_mean_std(_pct(agg.get("near_copy_rate_mean")), _pct(agg.get("near_copy_rate_std"))),
                    str(agg.get("qa_errors_total", "")),
                ]
            )
            + " |"
        )

    # Family breakdown table
    breakdowns_dict = summary.get("breakdowns") or {}
    if breakdowns_dict:
        family_tbl = breakdowns_dict.get("by_family") or []
        if family_tbl:
            mode_order = list_tbl_mode_keys(family_tbl)
            lines.extend(
                [
                    "",
                    "## Breakdown by Prompt Family",
                    "",
                    "| Family | "
                    + " | ".join(f"`{m}` Cat | `{m}` Color | `{m}` Blob | D Cat | D Color |" for m in mode_order)
                    + " |",
                    "|---:|" + "---:|".join([""] * (len(mode_order) * 6)) + " |",
                ]
            )
            # Group rows by family
            by_family: dict[str, list[dict[str, Any]]] = {}
            for row in family_tbl:
                f = str(row.get("prompt_family") or "")
                by_family.setdefault(f, []).append(row)
            for family, rows in sorted(by_family.items()):
                family_cols = [f"`{_md_escape(family)}`"]
                for m in mode_order:
                    mode_row = next((r for r in rows if r.get("_mode") == m), None)
                    if mode_row:
                        family_cols.extend(
                            [
                                _fmt(mode_row.get("category_consistency")),
                                _fmt(mode_row.get("color_consistency")),
                                _fmt(_pct(mode_row.get("blob_collapse_rate"))),
                                _fmt(mode_row.get("delta_category"))
                                if mode_row.get("delta_category") is not None
                                else "",
                                _fmt(mode_row.get("delta_color")) if mode_row.get("delta_color") is not None else "",
                            ]
                        )
                    else:
                        family_cols.extend(["", "", "", "", ""])
                lines.append(" | ".join(family_cols) + " |")

    # Profile aggregate tables
    profile_aggs = summary.get("profile_aggregates") or {}
    if profile_aggs:
        for profile_name, pa in profile_aggs.items():
            lines.extend(
                [
                    "",
                    f"### Profile Aggregate (`{profile_name}`, weighting=`{summary.get('profile_weighting', 'family')}`)",
                    "",
                    "| Mode | Families | Profiled Samples | Category (weighted) | Color (weighted) | Blob (weighted) |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for mode, agg in sorted(pa.items()):
                fams = ", ".join(str(f) for f in agg.get("families_used", []))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{_md_escape(str(mode))}`",
                            _md_escape(fams),
                            str(agg.get("sample_count_profiled", "")),
                            _fmt(agg.get("category_consistency_mean_weighted")),
                            _fmt(agg.get("color_consistency_mean_weighted")),
                            _fmt(agg.get("blob_collapse_rate_mean_weighted")),
                        ]
                    )
                    + " |"
                )

    if deltas:
        lines.extend(
            [
                "",
                "## Deltas vs `v1` Baseline",
                "",
                "| Mode | Color Delta | Category Delta | Rare Delta | Blob Delta | NearCopy Delta |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for mode_name, delta in sorted(deltas.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_md_escape(str(mode_name))}`",
                        _fmt(delta.get("color_consistency_mean_delta")),
                        _fmt(delta.get("category_consistency_mean_delta")),
                        _fmt(delta.get("rare_color_warning_rate_mean_delta")),
                        _fmt(delta.get("blob_collapse_rate_mean_delta")),
                        _fmt(delta.get("near_copy_rate_mean_delta")),
                    ]
                )
                + " |"
            )

    report_mode = "report-only harvest" if summary.get("meta", {}).get("report_only") else ""
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Metrics are computed from small sample sizes; confidence intervals reflect binomial uncertainty at that n.",
            "- No training was run; this is a no-training presets-and-conditioning-ablation evaluation.",
            "- Results should not be interpreted as statistical proof of superiority without larger-n confirmations.",
            "- Requested `max_samples` may exceed available prompts; actual generated count is reported in the Gen/Review/Faith columns.",
            f"- Report mode: {report_mode}" if report_mode else "",
            "",
            f"Generated contact sheets and reports are in `{meta.get('out_dir', '')}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(val: Any) -> Any:
    if val is None or not isinstance(val, (int, float)):
        return None
    return float(val) * 100.0


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def _fmt_mean_std(mean_val: Any, std_val: Any) -> str:
    if mean_val is None:
        return "n/a"
    s = _fmt(mean_val)
    if std_val is not None:
        try:
            s += f" +/- {float(std_val):.4f}"
        except (TypeError, ValueError):
            pass
    return s


def _compute_deltas(aggregates: dict[str, Any]) -> dict[str, Any]:
    v1_agg = aggregates.get("preset_v1")
    if v1_agg is None:
        return {}
    deltas: dict[str, Any] = {}
    for mode_name, agg in aggregates.items():
        if mode_name == "preset_v1":
            continue
        delta: dict[str, float | None] = {}
        for key in (
            "color_consistency_mean",
            "category_consistency_mean",
            "rare_color_warning_rate_mean",
            "blob_collapse_rate_mean",
            "near_copy_rate_mean",
        ):
            v1_val = v1_agg.get(key)
            other_val = agg.get(key)
            if (
                v1_val is not None
                and other_val is not None
                and isinstance(v1_val, (int, float))
                and isinstance(other_val, (int, float))
            ):
                delta[f"{key}_delta"] = float(other_val) - float(v1_val)
            else:
                delta[f"{key}_delta"] = None
        deltas[mode_name] = delta
    return deltas


# ── Main orchestration ──────────────────────────────────────────────────────


def run_v2_phase0_eval(config: V2Phase0EvalConfig) -> dict[str, Any]:
    cells = build_run_plan(config)
    if config.dry_run:
        print(f"Dry run: {len(cells)} planned cells")
        for cell in cells:
            preset_info = f"export_preset={cell.export_preset}" if cell.export_preset else ""
            diag_info = ""
            if cell.null_fields:
                diag_info = f"null_fields={cell.null_fields}"
            if cell.factored_cfg:
                diag_info = f"factored_cfg base={cell.cfg_base_scale} color={cell.cfg_color_scale}"
            extra = " ".join(p for p in [preset_info, diag_info] if p)
            print(f"  {cell.mode} seed={cell.seed} {extra}")
        return {"dry_run": True, "planned_cells": len(cells)}

    if not config.report_only:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError("PyTorch is required for spritelab v2 Phase 0 eval sampling.")

        from spritelab.training.generated_qa import qa_generated_sprites
        from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites
        from spritelab.training.generator_challenger import (
            ChallengerSampleConfig,
            run_sample_generator_challenger,
        )
        from spritelab.training.optim_utils import apply_backend_speed_flags
        from spritelab.training.prompt_faithfulness import PromptFaithfulnessConfig, run_prompt_faithfulness

        # Every cell/seed resamples the same checkpoint at the same fixed shape, so
        # cuDNN algorithm search pays for itself across the whole run; a no-op on CPU.
        apply_backend_speed_flags(
            cudnn_benchmark=config.speed_optimizations,
            tf32=config.speed_optimizations,
        )

    out_dir = Path(config.out)
    runs_dir = out_dir / "runs"
    summaries_dir = out_dir / "summaries"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Resolve prompts path (build if requested, otherwise use provided)
    prompts_path: Path
    prompt_build_report: dict[str, Any] | None = None
    if config.build_prompts:
        prompts_dir = out_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompts_path = prompts_dir / f"v2_eval_prompts_{config.prompt_count}.jsonl"

        # Check if prompts already exist with matching config
        if prompts_path.is_file() and (prompts_path.with_suffix(".report.json")).is_file():
            print(f"Reusing existing prompt file: {prompts_path}")
            report_json = _read_json(prompts_path.with_suffix(".report.json"))
            if report_json:
                existing_seed = report_json.get("seed")
                existing_target = report_json.get("target_count")
                if existing_seed == config.prompt_seed and existing_target == config.prompt_count:
                    prompt_build_report = report_json
                else:
                    raise RuntimeError(
                        f"Existing prompt file {prompts_path} has seed={existing_seed}, "
                        f"target={existing_target} but requested seed={config.prompt_seed}, "
                        f"target={config.prompt_count}. Remove the file or use matching args."
                    )
        else:
            from spritelab.training.v2_eval_prompts import V2EvalPromptsConfig, build_v2_eval_prompts

            manifest = config.dataset / "training_manifest.jsonl"
            prompt_build_report = build_v2_eval_prompts(
                V2EvalPromptsConfig(
                    dataset=config.dataset,
                    training_manifest=manifest,
                    out=prompts_path,
                    target_count=config.prompt_count,
                    seed=config.prompt_seed,
                    out_report=True,
                )
            )
            print(f"Built {prompt_build_report['prompt_count']} prompts (target: {config.prompt_count})")
            print(f"Prompt families: {prompt_build_report['families']}")
    elif config.prompts is not None:
        prompts_path = Path(config.prompts)
    else:
        raise RuntimeError("Either --prompts or --build-prompts must be provided.")

    all_metrics: list[RunMetrics] = []

    # Report-only mode: harvest existing outputs
    if config.report_only:
        missing: list[str] = []
        for cell in cells:
            cell_dir = runs_dir / cell.mode
            if _cell_outputs_exist(cell_dir):
                if not _skip_cell_dir_is_valid(cell, cell_dir):
                    raise RuntimeError(
                        f"Existing outputs in {cell_dir} do not match cell configuration "
                        f"(export_preset={cell.export_preset}, null_fields={cell.null_fields}). "
                        f"Remove or re-sample the directory, or adjust your cell parameters."
                    )
                metrics = collect_run_metrics(cell_dir, cell, config.max_samples)
                all_metrics.append(metrics)
                print(f"  Harvested: {cell.mode}")
            else:
                missing.append(cell.mode)

        if missing:
            msg = f"Missing {len(missing)} run(s): {missing}"
            if config.allow_partial_report:
                print(f"Warning: {msg}. Proceeding with partial report.")
            else:
                raise RuntimeError(
                    f"{msg}. Use --allow-partial-report to proceed, or --skip-sampling-if-exists to generate them."
                )
    else:
        for cell in cells:
            cell_dir = runs_dir / cell.mode
            print(f"\n--- {cell.mode} (seed={cell.seed}) ---")

            skip_ok = False
            if config.skip_sampling_if_exists and cell_dir.is_dir():
                if _skip_cell_dir_is_valid(cell, cell_dir):
                    print(f"  Skipping: valid outputs already exist in {cell_dir}")
                    skip_ok = True
                else:
                    raise RuntimeError(
                        f"Existing outputs in {cell_dir} do not match cell configuration "
                        f"(export_preset={cell.export_preset}, factored_cfg={cell.factored_cfg}, "
                        f"null_fields={cell.null_fields}). "
                        f"Remove the directory or run without --skip-sampling-if-exists to re-sample."
                    )

            if not skip_ok:
                sample_config = ChallengerSampleConfig(
                    checkpoint=config.checkpoint,
                    prompts=prompts_path,
                    out_dir=cell_dir,
                    export_preset=cell.export_preset,
                    max_samples=config.max_samples,
                    steps=30,
                    cfg_scale=3.0,
                    max_colors=32,
                    alpha_threshold=0.5,
                    device=config.device,
                    seed=cell.seed,
                    batch_size=config.batch_size,
                    write_raw_rgba=True,
                    write_hard_rgba=True,
                    contact_sheet_labels="prompt",
                    project_palette=True,
                    project_palette_target_colors=16,
                    project_palette_min_pixel_share=0.01,
                    project_palette_method="deterministic_kmeans",
                    factored_cfg=cell.factored_cfg,
                    cfg_base_scale=cell.cfg_base_scale,
                    cfg_color_scale=cell.cfg_color_scale,
                    null_fields=cell.null_fields,
                )
                _ = run_sample_generator_challenger(sample_config)
                print(f"  Sampled to {cell_dir}")

                verify_cell_metadata(cell, cell_dir)
                print(
                    f"  Metadata verification: OK (export_preset={cell.export_preset}, factored_cfg={cell.factored_cfg}, null_fields={cell.null_fields})"
                )

            # QA
            print("  Running QA...")
            qa_result = qa_generated_sprites(cell_dir)
            qa_errors = len(qa_result.errors)
            qa_warnings = len(qa_result.warnings)
            print(f"  QA: errors={qa_errors} warnings={qa_warnings}")

            # Review
            print("  Running review...")
            review_result = review_generated_sprites(
                GeneratedReviewConfig(
                    generated_dir=cell_dir,
                    out_dir=cell_dir / "review",
                    group_by="none",
                    max_samples_per_sheet=64,
                )
            )

            # Prompt faithfulness
            print("  Running prompt faithfulness...")
            faith_report = run_prompt_faithfulness(
                PromptFaithfulnessConfig(
                    generated=cell_dir,
                    prompts=prompts_path,
                    dataset=config.dataset,
                    out=cell_dir / "prompt_faithfulness_report.md",
                    out_json=cell_dir / "prompt_faithfulness_report.json",
                    max_sources=config.faithfulness_max_sources,
                    source_selection="auto",
                )
            )

            # Collect metrics
            metrics = collect_run_metrics(cell_dir, cell, config.max_samples)

            gen_count = metrics.sample_count_generated
            if gen_count < config.max_samples:
                print(
                    f"  Note: requested {config.max_samples} samples, "
                    f"available prompts produced {gen_count} generated sprites"
                )

            all_metrics.append(metrics)

    # Aggregate
    aggregates = aggregate_metrics(all_metrics)
    deltas = _compute_deltas(aggregates)

    # Breakdowns by prompt family / category / color / object / base_object
    breakdowns = _compute_breakdowns(all_metrics, prompts_path, aggregates, summaries_dir)

    # Decision (need v1_agg/v11_agg for profile decision too)
    v1_cells = [m for m in all_metrics if _base_mode(m.mode) == "preset_v1"]
    v11_cells = [m for m in all_metrics if _base_mode(m.mode) == "preset_v1_1"]
    v1_agg = aggregates.get("preset_v1")
    v11_agg = aggregates.get("preset_v1_1")

    # Profile-specific aggregates (ood_core / ood_plus_grid)
    profile_aggregates: dict[str, Any] = {}
    profile_decisions: dict[str, Any] = {}
    if config.eval_profile != "all" and breakdowns:
        pa = compute_profile_aggregates(breakdowns, config.eval_profile, config.profile_weighting, aggregates)
        if pa:
            profile_aggregates[config.eval_profile] = pa
            v1_prof = pa.get("preset_v1")
            v11_prof = pa.get("preset_v1_1")
            profile_decisions[config.eval_profile] = decide_ood_core(v1_prof, v11_prof, v1_agg)

    decision = decide_v1_1(v1_agg, v11_agg)

    per_run = [
        {
            "mode": m.mode,
            "seed": m.seed,
            "export_preset": m.export_preset,
            "factored_cfg": m.factored_cfg,
            "cfg_base_scale": m.cfg_base_scale,
            "cfg_color_scale": m.cfg_color_scale,
            "null_fields": m.null_fields,
            "null_field_set": m.null_field_set,
            "requested_max_samples": m.requested_max_samples,
            "sample_count_generated": m.sample_count_generated,
            "review_sample_count": m.review_sample_count,
            "faithfulness_sample_count": m.faithfulness_sample_count,
            "qa_errors": m.qa_errors,
            "qa_warnings": m.qa_warnings,
            "median_visible_colors": m.median_visible_colors,
            "rare_color_warning_rate": m.rare_color_warning_rate,
            "touches_border_rate": m.touches_border_rate,
            "category_consistency": m.category_consistency,
            "category_ci95": m.category_ci95,
            "color_consistency": m.color_consistency,
            "color_ci95": m.color_ci95,
            "repeated_silhouette_rate": m.repeated_silhouette_rate,
            "blob_collapse_rate": m.blob_collapse_rate,
            "potion_collapse_rate": m.potion_collapse_rate,
            "near_copy_rate": m.near_copy_rate,
            "p10_nearest_source_distance": m.p10_nearest_source_distance,
            "source_count_used": m.source_count_used,
            "source_candidate_hash": m.source_candidate_hash,
            "median_colors_before": m.median_colors_before,
            "median_colors_after": m.median_colors_after,
            "mean_rgb_mae_visible": m.mean_rgb_mae_visible,
            "destructive_rate": m.destructive_rate,
        }
        for m in all_metrics
    ]

    source_hash = ""
    if v1_cells:
        source_hash = v1_cells[0].source_candidate_hash

    summary = {
        "meta": {
            "checkpoint": str(config.checkpoint),
            "prompts": str(prompts_path),
            "dataset": str(config.dataset),
            "max_samples": config.max_samples,
            "seeds": list(config.seeds),
            "device": config.device,
            "out_dir": str(out_dir),
            "source_hash": source_hash,
            "prompts_built": config.build_prompts,
            "prompt_count": prompt_build_report["prompt_count"] if prompt_build_report else None,
            "prompt_count_requested": config.prompt_count if config.build_prompts else None,
            "prompt_seed": config.prompt_seed if config.build_prompts else None,
            "prompt_jsonl_path": str(prompts_path) if config.build_prompts else None,
            "prompt_report_path": str(prompts_path.with_suffix(".report.json")) if config.build_prompts else None,
            "prompt_build_report": prompt_build_report,
            "report_only": config.report_only,
            "allow_partial_report": config.allow_partial_report,
        },
        "preset_definitions": {
            "v1": "Phase 1 EMA checkpoint, CFG 3.0, 30 steps, k16 deterministic palette projection.",
            "v1.1": (
                "v1 base settings plus factored CFG: base_scale=2.5, color_scale=3.0. Optional color-strong preset."
            ),
        },
        "per_run": per_run,
        "aggregates": aggregates,
        "deltas_vs_v1": deltas,
        "decision": decision,
        "breakdowns": {k: v for k, v in breakdowns.items() if v} if breakdowns else {},
        "profile_aggregates": profile_aggregates,
        "profile_decisions": profile_decisions,
        "eval_profile": config.eval_profile,
        "profile_weighting": config.profile_weighting,
    }

    json_path = summaries_dir / "phase0_eval_summary.json"
    csv_path = summaries_dir / "phase0_eval_summary.csv"
    md_path = summaries_dir / "phase0_eval_report.md"
    write_summary_json(summary, json_path)
    write_summary_csv(summary, csv_path)
    write_summary_markdown(summary, md_path)

    print("\n--- Summary ---")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print(f"Decision: {decision['label']} - {decision['explanation']}")

    return summary
