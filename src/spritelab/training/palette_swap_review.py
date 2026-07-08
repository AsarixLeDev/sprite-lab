"""Offline review + quality metrics for palette-swap augmentation.

Samples augmented training examples *without training* so aggressive vs
conservative palette-swap settings can be inspected before committing GPU time.
Produces JSON + Markdown reports, a JSONL of per-sample diagnostics, and contact
sheets (original vs augmented) grouped by source category, target family, and
source->target family.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from spritelab.training.data import read_jsonl
from spritelab.training.framing_metrics import SPRITE_SIZE, checkerboard_rgba, jsonable, rgba_array_to_image
from spritelab.training.palette_swap import (
    ACHROMATIC_FAMILIES,
    DEFAULT_SWAP_FAMILIES_TEXT,
    MATERIAL_COLOR_FAMILIES,
    PaletteSwapConfig,
    apply_palette_swap,
    _explicit_caption_colors,
    _explicit_semantic_colors,
    nearest_family,
)
from spritelab.training.rgba import npz_row_to_rgba
from spritelab.training.structured_conditioning import extract_structured_fields

SCHEMA_VERSION = 2

# Red-flag thresholds.
FALLBACK_RATE_WARN = 0.25
MATERIAL_LIKE_TARGET_WARN = 0.35
MATERIAL_CONFLICT_WARN = 0.30
CAPTION_PREPEND_WARN = 0.50
ROLE_RECOLOR_LOW_WARN = 1.0
ROLE_RECOLOR_HIGH_WARN = 10.0
CATEGORY_MATERIAL_TARGET_WARN = 0.30
CONSERVATIVE_APPLIED_RATE_MIN = 0.05
CONSERVATIVE_APPLIED_RATE_MAX = 0.15
EXPANDED_EFFECTIVE_ELIGIBLE_RATE_MIN = 0.30
EXPANDED_EFFECTIVE_ELIGIBLE_RATE_MAX = 0.50
SAME_FAMILY_RECOLOR_WARN = 0.0
TARGET_IMBALANCE_SHARE_WARN = 0.60
TARGET_IMBALANCE_MIN_FAMILIES = 3
_MATERIAL_LIKE_TARGETS = frozenset(MATERIAL_COLOR_FAMILIES | ACHROMATIC_FAMILIES)


@dataclass(frozen=True)
class PaletteSwapReviewConfig:
    dataset_dir: Path
    training_manifest: Path
    out_dir: Path
    seed: int = 20260706
    max_samples: int = 256
    draws_per_sprite: int = 1
    review_selection: str = "first"
    palette_swap_prob: float = 0.5
    palette_swap_families: str = DEFAULT_SWAP_FAMILIES_TEXT
    palette_swap_target_families: str | None = None
    palette_swap_source_families: str | None = None
    palette_swap_category_filter: str | None = None
    palette_swap_min_color_confidence: float = 0.0
    palette_swap_stochastic: bool = False
    palette_swap_keep_original_prob: float = 0.0
    palette_swap_require_role_map: bool = False
    palette_swap_require_explicit_color: bool = False
    palette_swap_require_explicit_caption_color: bool = False
    palette_swap_require_explicit_semantic_color: bool = False
    palette_swap_allow_colorless_caption_if_semantic_color: bool = False
    palette_swap_no_caption_prepend: bool = False
    palette_swap_allow_material_colors: bool = True
    palette_swap_preserve_outline: bool = True
    palette_swap_update_prompts: bool = True


@dataclass
class PaletteSwapReviewResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    jsonl_path: Path
    contact_sheets: dict[str, Path] = field(default_factory=dict)


def swap_config_from_review(config: PaletteSwapReviewConfig) -> PaletteSwapConfig:
    namespace = SimpleNamespace(
        palette_swap_augmentation=True,
        palette_swap_prob=config.palette_swap_prob,
        palette_swap_families=config.palette_swap_families,
        palette_swap_target_families=config.palette_swap_target_families,
        palette_swap_source_families=config.palette_swap_source_families,
        palette_swap_category_filter=config.palette_swap_category_filter,
        palette_swap_min_color_confidence=config.palette_swap_min_color_confidence,
        palette_swap_stochastic=config.palette_swap_stochastic,
        palette_swap_keep_original_prob=config.palette_swap_keep_original_prob,
        palette_swap_require_role_map=config.palette_swap_require_role_map,
        palette_swap_require_explicit_color=config.palette_swap_require_explicit_color,
        palette_swap_require_explicit_caption_color=config.palette_swap_require_explicit_caption_color,
        palette_swap_require_explicit_semantic_color=config.palette_swap_require_explicit_semantic_color,
        palette_swap_allow_colorless_caption_if_semantic_color=(
            config.palette_swap_allow_colorless_caption_if_semantic_color
        ),
        palette_swap_no_caption_prepend=config.palette_swap_no_caption_prepend,
        palette_swap_allow_material_colors=config.palette_swap_allow_material_colors,
        palette_swap_preserve_outline=config.palette_swap_preserve_outline,
        palette_swap_update_prompts=config.palette_swap_update_prompts,
        seed=config.seed,
    )
    return PaletteSwapConfig.from_training_config(namespace)


@dataclass
class _SampleEvaluation:
    row: dict[str, Any]
    before_rgba: np.ndarray
    after_rgba: np.ndarray
    applied: bool


def review_palette_swap(config: PaletteSwapReviewConfig) -> PaletteSwapReviewResult:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    swap_config = swap_config_from_review(config)
    records = read_jsonl(config.training_manifest)
    selected = _select_review_records(
        records,
        selection=str(config.review_selection),
        max_samples=int(config.max_samples),
        seed=int(config.seed),
    )
    review_category_counts = _category_counts(selected)
    available_category_counts = _category_counts(records)

    evaluations = list(
        _evaluate_samples(
            Path(config.dataset_dir),
            selected,
            swap_config,
            draws_per_sprite=max(1, int(config.draws_per_sprite)),
        )
    )
    rows = [item.row for item in evaluations]
    metrics = aggregate_metrics(rows)
    red_flags = compute_red_flags(metrics, rows, swap_config)

    contact_sheets = _write_contact_sheets(out_dir, evaluations)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(config.dataset_dir),
        "training_manifest": str(config.training_manifest),
        "config": swap_config.report_dict(),
        "seed": int(config.seed),
        "requested_max_samples": int(config.max_samples),
        "draws_per_sprite": max(1, int(config.draws_per_sprite)),
        "review_selection": str(config.review_selection),
        "review_selected_record_count": len(selected),
        "review_available_record_count": len(records),
        "review_category_counts": review_category_counts,
        "review_available_category_counts": available_category_counts,
        "metrics": metrics,
        "red_flags": red_flags,
        "contact_sheets": {key: str(path) for key, path in sorted(contact_sheets.items())},
    }
    report.update(_stable_top_level_fields(metrics, swap_config))

    json_path = out_dir / "palette_swap_review.json"
    markdown_path = out_dir / "palette_swap_review.md"
    jsonl_path = out_dir / "palette_swap_samples.jsonl"
    json_path.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(format_palette_swap_review_markdown(report), encoding="utf-8")
    jsonl_path.write_text("".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    return PaletteSwapReviewResult(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
        jsonl_path=jsonl_path,
        contact_sheets=contact_sheets,
    )


def summarize_dataset_palette_swap(
    dataset_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    config: PaletteSwapConfig,
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Eligibility/apply statistics over dataset records for the training report."""

    selected = list(records) if max_samples is None else list(records)[: max(0, int(max_samples))]
    rows = [item.row for item in _evaluate_samples(Path(dataset_dir), selected, config)]
    metrics = aggregate_metrics(rows)
    eligible = metrics["eligible_count"]
    applied = metrics["applied_count"]
    return {
        "sample_count": metrics["sample_count"],
        "eligible_count": eligible,
        "ineligible_count": metrics["ineligible_count"],
        "applied_count": applied,
        "swapped_count": metrics["swapped_count"],
        "kept_original_count": metrics["kept_original_count"],
        "effective_eligible_count_before_keep_original": metrics["effective_eligible_count_before_keep_original"],
        "unchanged_ineligible_count": metrics["unchanged_ineligible_count"],
        "unchanged_not_triggered_count": metrics["unchanged_not_triggered_count"],
        "applied_rate": metrics["applied_rate"],
        "applied_rate_total": metrics["applied_rate"],
        "applied_rate_eligible": metrics["applied_rate_eligible"],
        "effective_swapped_rate_total": metrics["effective_swapped_rate_total"],
        "effective_kept_original_rate_total": metrics["effective_kept_original_rate_total"],
        "effective_eligible_rate_total_before_keep_original": metrics[
            "effective_eligible_rate_total_before_keep_original"
        ],
        "effective_swapped_rate_eligible": metrics["effective_swapped_rate_eligible"],
        "effective_kept_original_rate_eligible": metrics["effective_kept_original_rate_eligible"],
        "ineligibility_reason_counts": metrics["ineligibility_reason_counts"],
        "source_family_counts": metrics["source_family_counts"],
        "target_family_counts": metrics["target_family_counts"],
        "source_to_target_matrix": metrics["source_to_target_matrix"],
        "fallback_heuristic_rate": metrics["fallback_heuristic_rate"],
        "material_conflict_drop_count": metrics["material_conflict_drop_count"],
        "caption_color_replaced_count": metrics["caption_color_replaced_count"],
        "caption_color_prepended_count": metrics["caption_color_prepended_count"],
        "caption_no_color_count": metrics["caption_no_color_count"],
        "same_family_skip_count": metrics["same_family_skip_count"],
        "same_family_recolor_count": metrics["same_family_recolor_count"],
        "same_family_recolor_rate": metrics["same_family_recolor_rate"],
        "colorless_caption_structured_only_count": metrics["colorless_caption_structured_only_count"],
    }


def _evaluate_samples(
    dataset_dir: Path,
    records: Sequence[Mapping[str, Any]],
    swap_config: PaletteSwapConfig,
    *,
    draws_per_sprite: int = 1,
):
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    draw_count = max(1, int(draws_per_sprite))
    global_draw_index = 0
    for record in records:
        npz_file = str(record.get("npz_file") or f"{record.get('split', '')}.npz")
        npz_row = int(record.get("npz_row", -1))
        arrays = npz_cache.get(npz_file)
        if arrays is None:
            path = dataset_dir / npz_file
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            npz_cache[npz_file] = arrays
        if npz_row < 0 or npz_row >= int(np.asarray(arrays["alpha"]).shape[0]):
            continue
        for _draw in range(draw_count):
            yield _evaluate_one(record, arrays, npz_row, swap_config, draw_index=global_draw_index)
            global_draw_index += 1


def _select_review_records(
    records: Sequence[Mapping[str, Any]],
    *,
    selection: str,
    max_samples: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    mode = str(selection or "first").strip().lower()
    if mode not in {"first", "random", "balanced", "all"}:
        raise ValueError("--review-selection must be one of: first, random, balanced, all")
    rows = list(records)
    if mode == "all" or int(max_samples) <= 0:
        return rows
    limit = max(0, min(int(max_samples), len(rows)))
    if mode == "first":
        return rows[:limit]
    rng = np.random.default_rng(int(seed))
    if mode == "random":
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        return [rows[index] for index in indices[:limit]]

    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(_row_category(row), []).append(row)
    categories = sorted(by_category)
    if not categories:
        return []
    per_category_cap = int(np.ceil(limit / float(len(categories))))
    selected: list[Mapping[str, Any]] = []
    for category in categories:
        category_rows = list(by_category[category])
        indices = list(range(len(category_rows)))
        rng.shuffle(indices)
        for index in indices[:per_category_cap]:
            if len(selected) >= limit:
                break
            selected.append(category_rows[index])
    return selected


def _category_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_row_category(record) for record in records).items()))


def _row_category(record: Mapping[str, Any]) -> str:
    value = str(record.get("category") or record.get("prompt_category") or "").strip().lower()
    if value:
        return value
    fields = extract_structured_fields(record)
    return str(fields.get("category") or "unknown").strip().lower() or "unknown"


def _evaluate_one(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    npz_row: int,
    swap_config: PaletteSwapConfig,
    *,
    draw_index: int | None = None,
) -> _SampleEvaluation:
    index_np = np.asarray(arrays["index_map"][npz_row], dtype=np.int64)
    alpha_np = np.asarray(arrays["alpha"][npz_row], dtype=np.float32)
    role_np = np.asarray(arrays["role_map"][npz_row], dtype=np.int64) if "role_map" in arrays else None
    palette_np = _palette_rgb_float(np.asarray(arrays["palette"][npz_row]))
    palette_mask_np = np.asarray(arrays["palette_mask"][npz_row], dtype=bool)
    sprite_id = str(record.get("sprite_id") or "")
    caption = str(record.get("caption", ""))

    result = apply_palette_swap(
        index_map=index_np,
        alpha=alpha_np,
        role_map=role_np,
        palette_rgb=palette_np,
        palette_mask=palette_mask_np,
        record=record,
        caption=caption,
        sprite_id=sprite_id,
        config=swap_config,
        draw_index=draw_index,
    )

    before_rgba = npz_row_to_rgba(index_map=index_np, alpha=alpha_np, palette=palette_np, palette_mask=palette_mask_np)
    after_rgba = (
        npz_row_to_rgba(index_map=index_np, alpha=alpha_np, palette=result.palette_rgb, palette_mask=palette_mask_np)
        if result.applied
        else before_rgba
    )
    visible = np.asarray(before_rgba[3]) > 0.0
    original_fields = extract_structured_fields(record)
    augmented_fields = extract_structured_fields(result.record)
    original_caption_colors = sorted(_explicit_caption_colors(record, caption))
    original_semantic_colors = sorted(_explicit_semantic_colors(record))

    metadata = result.metadata()
    material_conflict_tokens = sorted({str(token) for token in result.materials_dropped if str(token)})
    material_conflict_dropped = bool(material_conflict_tokens)
    colorless_caption_structured_only = bool(
        not original_caption_colors
        and original_semantic_colors
        and result.eligible
        and swap_config.no_caption_prepend
        and (
            swap_config.allow_colorless_caption_if_semantic_color
            or swap_config.require_semantic_color()
        )
    )
    row = {
        "sprite_id": sprite_id,
        "palette_swap_draw_index": metadata.get("palette_swap_draw_index"),
        "palette_swap_seed": metadata.get("palette_swap_seed"),
        "palette_swap_stochastic": metadata.get("palette_swap_stochastic"),
        "category": str(record.get("category") or original_fields.get("category") or ""),
        "object_name": str(record.get("object_name") or original_fields.get("object_name") or ""),
        "base_object": str(record.get("base_object") or original_fields.get("base_object") or ""),
        "original_caption": caption,
        "augmented_caption": result.caption if result.applied else caption,
        "original_semantic_colors": list(original_fields.get("colors") or []),
        "augmented_semantic_colors": list(augmented_fields.get("colors") or []),
        "original_caption_color_families": original_caption_colors,
        "original_semantic_color_families": original_semantic_colors,
        "original_materials": list(original_fields.get("materials") or []),
        "augmented_materials": list(augmented_fields.get("materials") or []),
        "material_conflict_dropped": material_conflict_dropped,
        "material_conflict_tokens_dropped": material_conflict_tokens,
        "material_conflict_reason": "material_color_token_conflicted_with_target_family"
        if material_conflict_dropped
        else "",
        "source_color_family": result.source_color_family,
        "target_color_family": result.target_color_family,
        "source_color_confidence": round(float(result.source_color_confidence), 4),
        "eligible_for_palette_swap": bool(result.eligible),
        "ineligibility_reason": result.ineligibility_reason,
        "triggered": bool(result.triggered),
        "applied": bool(result.applied),
        "kept_original": bool(result.kept_original),
        "roles_recolored": list(result.roles_recolored),
        "recolored_palette_indices": list(result.recolored_palette_indices),
        "role_map_trusted": bool(result.role_map_trusted),
        "fallback_heuristic_used": bool(result.fallback_heuristic_used),
        "caption_change_kind": result.caption_change_kind,
        "material_conflict_drop_count": 1 if material_conflict_dropped else 0,
        "target_resampled_from_same_family": bool(result.target_resampled_from_same_family),
        "same_family_skip": bool(result.same_family_skip),
        "visible_color_count_before": _visible_color_count(before_rgba, visible),
        "visible_color_count_after": _visible_color_count(after_rgba, visible),
        "palette_family_histogram_before": _family_histogram(index_np, visible, palette_np),
        "palette_family_histogram_after": _family_histogram(index_np, visible, result.palette_rgb),
        "mean_rgb_delta_visible": _mean_rgb_delta(before_rgba, after_rgba, visible),
        "palette_entries_changed": len(result.recolored_palette_indices),
        "colorless_caption_structured_only": colorless_caption_structured_only,
    }
    return _SampleEvaluation(row=row, before_rgba=before_rgba, after_rgba=after_rgba, applied=bool(result.applied))


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    applied_rows = [row for row in rows if row.get("applied")]
    kept_rows = [row for row in rows if row.get("kept_original")]
    eligible_rows = [row for row in rows if row.get("eligible_for_palette_swap")]
    applied_count = len(applied_rows)
    kept_original_count = len(kept_rows)
    eligible_count = len(eligible_rows)
    material_conflict_drop_count = int(sum(1 for row in rows if row.get("material_conflict_dropped")))

    source_to_target: dict[str, Counter] = {}
    category_to_target: dict[str, Counter] = {}
    for row in applied_rows:
        source = str(row.get("source_color_family") or "unknown")
        target = str(row.get("target_color_family") or "unknown")
        category = str(row.get("category") or "unknown")
        source_to_target.setdefault(source, Counter())[target] += 1
        category_to_target.setdefault(category, Counter())[target] += 1

    caption_kinds = Counter()
    for row in applied_rows:
        kind = str(row.get("caption_change_kind") or "none")
        if kind == "replaced":
            caption_kinds["replaced"] += 1
        elif kind == "prepended" and str(row.get("original_caption") or "").strip():
            caption_kinds["prepended"] += 1
        else:
            caption_kinds["no_color"] += 1

    ineligibility_reason_counts = Counter(
        str(row.get("ineligibility_reason")) for row in rows if not row.get("eligible_for_palette_swap")
    )
    role_counts: Counter = Counter()
    for row in applied_rows:
        role_counts.update(str(role) for role in row.get("roles_recolored") or [])
    same_family_recolor_count = sum(
        1
        for row in applied_rows
        if str(row.get("source_color_family") or "") and row.get("source_color_family") == row.get("target_color_family")
    )
    per_sprite_target_counters: dict[str, Counter] = {}
    per_sprite_draw_counts: Counter = Counter()
    per_sprite_kept_counts: Counter = Counter()
    per_sprite_swapped_counts: Counter = Counter()
    for row in rows:
        sprite_id = str(row.get("sprite_id") or "unknown")
        per_sprite_draw_counts[sprite_id] += 1
        if row.get("kept_original"):
            per_sprite_kept_counts[sprite_id] += 1
        if row.get("applied"):
            target = str(row.get("target_color_family") or "")
            if target:
                per_sprite_target_counters.setdefault(sprite_id, Counter())[target] += 1
                per_sprite_swapped_counts[sprite_id] += 1
    per_sprite_target_diversity = {
        sprite_id: {
            "draw_count": int(per_sprite_draw_counts[sprite_id]),
            "swapped_count": int(per_sprite_swapped_counts[sprite_id]),
            "kept_original_count": int(per_sprite_kept_counts[sprite_id]),
            "distinct_target_family_count": len(per_sprite_target_counters.get(sprite_id, {})),
            "target_family_counts": dict(sorted(per_sprite_target_counters.get(sprite_id, {}).items())),
        }
        for sprite_id in sorted(per_sprite_draw_counts)
    }
    target_diversity_histogram = Counter(
        int(stats["distinct_target_family_count"]) for stats in per_sprite_target_diversity.values()
    )
    category_coverage = _category_coverage(rows)

    metrics = {
        "sample_count": total,
        "applied_count": applied_count,
        "swapped_count": applied_count,
        "kept_original_count": kept_original_count,
        "effective_eligible_count_before_keep_original": applied_count + kept_original_count,
        "unchanged_ineligible_count": total - eligible_count,
        "unchanged_not_triggered_count": int(
            sum(1 for row in eligible_rows if not row.get("triggered") and not row.get("applied") and not row.get("kept_original"))
        ),
        "eligible_count": eligible_count,
        "eligible_rate": (eligible_count / float(total)) if total else 0.0,
        "ineligible_count": total - eligible_count,
        "applied_rate": (applied_count / float(total)) if total else 0.0,
        "applied_rate_eligible": (applied_count / float(eligible_count)) if eligible_count else 0.0,
        "effective_swapped_rate_total": (applied_count / float(total)) if total else 0.0,
        "effective_kept_original_rate_total": (kept_original_count / float(total)) if total else 0.0,
        "effective_eligible_rate_total_before_keep_original": (
            (applied_count + kept_original_count) / float(total)
        )
        if total
        else 0.0,
        "effective_eligible_rate_total": ((applied_count + kept_original_count) / float(total)) if total else 0.0,
        "effective_swapped_rate_eligible": (applied_count / float(eligible_count)) if eligible_count else 0.0,
        "effective_kept_original_rate_eligible": (
            kept_original_count / float(eligible_count)
        )
        if eligible_count
        else 0.0,
        "kept_original_rate_eligible": (kept_original_count / float(eligible_count)) if eligible_count else 0.0,
        "source_family_counts": dict(Counter(str(row.get("source_color_family") or "unknown") for row in applied_rows)),
        "target_family_counts": dict(Counter(str(row.get("target_color_family") or "unknown") for row in applied_rows)),
        "source_to_target_matrix": {src: dict(counter) for src, counter in sorted(source_to_target.items())},
        "category_to_target_matrix": {cat: dict(counter) for cat, counter in sorted(category_to_target.items())},
        "role_recolored_counts": dict(role_counts),
        "fallback_heuristic_rate": _rate(applied_rows, "fallback_heuristic_used"),
        "material_conflict_drop_count": material_conflict_drop_count,
        "caption_color_replaced_count": int(caption_kinds["replaced"]),
        "caption_color_prepended_count": int(caption_kinds["prepended"]),
        "caption_no_color_count": int(caption_kinds["no_color"]),
        "mean_rgb_delta_visible": _mean(applied_rows, "mean_rgb_delta_visible"),
        "mean_palette_entries_changed": _mean(applied_rows, "palette_entries_changed"),
        "mean_visible_color_count_before": _mean(applied_rows, "visible_color_count_before"),
        "mean_visible_color_count_after": _mean(applied_rows, "visible_color_count_after"),
        "ineligibility_reason_counts": dict(sorted(ineligibility_reason_counts.items())),
        "same_family_skip_count": int(sum(1 for row in rows if row.get("same_family_skip"))),
        "same_family_resample_count": int(sum(1 for row in applied_rows if row.get("target_resampled_from_same_family"))),
        "same_family_recolor_count": int(same_family_recolor_count),
        "same_family_recolor_rate": (same_family_recolor_count / float(applied_count)) if applied_count else 0.0,
        "colorless_caption_structured_only_count": int(
            sum(1 for row in rows if row.get("colorless_caption_structured_only"))
        ),
        "per_sprite_target_diversity": per_sprite_target_diversity,
        "target_family_diversity_histogram_per_sprite": dict(sorted(target_diversity_histogram.items())),
        "category_coverage": category_coverage,
    }
    return metrics


def _category_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category") or "unknown"), []).append(row)
    coverage: dict[str, dict[str, Any]] = {}
    for category, category_rows in sorted(by_category.items()):
        total = len(category_rows)
        eligible = sum(1 for row in category_rows if row.get("eligible_for_palette_swap"))
        swapped = sum(1 for row in category_rows if row.get("applied"))
        kept_original = sum(1 for row in category_rows if row.get("kept_original"))
        coverage[category] = {
            "total": total,
            "eligible": eligible,
            "swapped": swapped,
            "kept_original": kept_original,
            "eligible_rate": (eligible / float(total)) if total else 0.0,
            "swapped_rate": (swapped / float(total)) if total else 0.0,
            "kept_original_rate": (kept_original / float(total)) if total else 0.0,
        }
    return coverage


def compute_red_flags(
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    config: PaletteSwapConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    applied = int(metrics.get("applied_count") or 0)
    total = int(metrics.get("sample_count") or len(rows) or 0)
    applied_rate = float(metrics.get("applied_rate") or 0.0)
    conservative_rate_context = _conservative_red_flag_context(config)
    if conservative_rate_context and total > 0:
        expanded_context = _config_bool(config, "palette_swap_allow_colorless_caption_if_semantic_color")
        rate_for_band = (
            float(metrics.get("effective_eligible_rate_total_before_keep_original") or 0.0)
            if expanded_context
            else applied_rate
        )
        min_rate = EXPANDED_EFFECTIVE_ELIGIBLE_RATE_MIN if expanded_context else CONSERVATIVE_APPLIED_RATE_MIN
        max_rate = EXPANDED_EFFECTIVE_ELIGIBLE_RATE_MAX if expanded_context else CONSERVATIVE_APPLIED_RATE_MAX
        if rate_for_band < min_rate:
            flags.append(
                _flag(
                    "applied_rate_too_low",
                    rate_for_band,
                    min_rate,
                    "Palette-swap effective coverage is below the configured target band.",
                )
            )
        elif rate_for_band > max_rate:
            flags.append(
                _flag(
                    "applied_rate_too_high",
                    rate_for_band,
                    max_rate,
                    "Palette-swap effective coverage is above the configured target band.",
                )
            )
    if applied <= 0:
        return flags

    fallback_rate = float(metrics.get("fallback_heuristic_rate") or 0.0)
    fallback_warn = 0.0 if conservative_rate_context else FALLBACK_RATE_WARN
    if fallback_rate > fallback_warn:
        flags.append(_flag("high_fallback_heuristic_rate", fallback_rate, fallback_warn,
                           "Many recolors relied on the luminance fallback instead of a trusted role map."))

    target_counts = metrics.get("target_family_counts") if isinstance(metrics.get("target_family_counts"), Mapping) else {}
    material_like = sum(int(count) for family, count in target_counts.items() if family in _MATERIAL_LIKE_TARGETS)
    material_like_rate = material_like / float(applied)
    if material_like_rate > MATERIAL_LIKE_TARGET_WARN:
        flags.append(_flag("many_material_like_targets", material_like_rate, MATERIAL_LIKE_TARGET_WARN,
                           "Many samples target gray/gold/brown families that are material-like or ambiguous."))

    material_conflict_rate = int(metrics.get("material_conflict_drop_count") or 0) / float(applied)
    if material_conflict_rate > MATERIAL_CONFLICT_WARN:
        flags.append(_flag("frequent_material_conflicts", material_conflict_rate, MATERIAL_CONFLICT_WARN,
                           "Recolors frequently dropped conflicting material color tokens (noisy relabeling)."))

    prepended = int(metrics.get("caption_color_prepended_count") or 0)
    prepend_rate = prepended / float(applied)
    if _config_bool(config, "palette_swap_no_caption_prepend") and prepended != 0:
        flags.append(_flag("caption_prepending_nonzero_when_disabled", prepended, 0.0,
                           "Caption prepending is disabled but applied rows still prepended a color token."))
    if _config_bool(config, "palette_swap_require_explicit_caption_color"):
        missing_caption = sum(1 for row in rows if row.get("applied") and not row.get("original_caption_color_families"))
        if missing_caption:
            flags.append(_flag("explicit_caption_required_but_missing", missing_caption / float(applied), 0.0,
                               "Caption-color gate is enabled but applied rows lacked a caption color token."))
    if prepend_rate > CAPTION_PREPEND_WARN:
        flags.append(_flag("frequent_caption_prepending", prepend_rate, CAPTION_PREPEND_WARN,
                           "Most captions had no explicit color, so a color was invented and prepended."))

    same_family_rate = float(metrics.get("same_family_recolor_rate") or 0.0)
    if same_family_rate > SAME_FAMILY_RECOLOR_WARN:
        flags.append(_flag("same_family_recolor_rate_high", same_family_rate, SAME_FAMILY_RECOLOR_WARN,
                           "Applied palette swaps include source and target colors from the same family."))

    mean_entries = float(metrics.get("mean_palette_entries_changed") or 0.0)
    if mean_entries < ROLE_RECOLOR_LOW_WARN:
        flags.append(_flag("role_recolor_rate_too_low", mean_entries, ROLE_RECOLOR_LOW_WARN,
                           "Very few palette entries changed per sample; augmentation is barely visible."))
    elif mean_entries > ROLE_RECOLOR_HIGH_WARN:
        flags.append(_flag("role_recolor_rate_too_high", mean_entries, ROLE_RECOLOR_HIGH_WARN,
                           "Many palette entries changed per sample; recolor may bleed past fill regions."))

    target_family_count = len(target_counts)
    if target_counts:
        expected_min_families = min(
            TARGET_IMBALANCE_MIN_FAMILIES,
            max(1, len(_config_target_families(config)) or TARGET_IMBALANCE_MIN_FAMILIES),
        )
        max_share = max(int(count) for count in target_counts.values()) / float(applied)
        if applied >= expected_min_families and target_family_count < expected_min_families:
            flags.append(_flag("target_family_imbalance", target_family_count, expected_min_families,
                               "Applied swaps cover too few target families."))
        elif applied >= expected_min_families and max_share > TARGET_IMBALANCE_SHARE_WARN:
            flags.append(_flag("target_family_imbalance", max_share, TARGET_IMBALANCE_SHARE_WARN,
                               "Applied swaps are concentrated in one target family."))
    if target_family_count >= 2 and not _config_bool(config, "palette_swap_stochastic"):
        shares = [int(count) / float(applied) for count in target_counts.values()]
        if max(shares) < (1.5 / float(target_family_count)):
            flags.append(_flag("target_distribution_too_uniform", max(shares), 1.5 / float(target_family_count),
                               "Target family distribution is nearly uniform across all objects/materials."))

    category_matrix = metrics.get("category_to_target_matrix") if isinstance(metrics.get("category_to_target_matrix"), Mapping) else {}
    weird: list[str] = []
    for category, targets in category_matrix.items():
        category_total = sum(int(v) for v in targets.values())
        if category_total < 8:
            continue
        material_like_here = sum(int(count) for fam, count in targets.items() if fam in _MATERIAL_LIKE_TARGETS)
        if material_like_here / float(category_total) > CATEGORY_MATERIAL_TARGET_WARN:
            weird.append(str(category))
    if weird:
        flags.append(
            {
                "name": "categories_receiving_material_like_colors",
                "categories": sorted(weird),
                "threshold": CATEGORY_MATERIAL_TARGET_WARN,
                "detail": "Some object categories are frequently recolored into material-like/ambiguous families.",
            }
        )
    return flags


def format_palette_swap_review_markdown(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), Mapping) else {}
    config = report.get("config") if isinstance(report.get("config"), Mapping) else {}
    red_flags = report.get("red_flags") if isinstance(report.get("red_flags"), list) else []
    lines = [
        "# Palette-Swap Review",
        "",
        f"Dataset: `{report.get('dataset_dir', '')}`",
        f"Seed: {report.get('seed')}",
        f"Selection: {report.get('review_selection', 'first')}",
        "",
        "## Settings",
        "",
        f"- Probability: {config.get('palette_swap_prob')}",
        f"- Target families: {config.get('palette_swap_target_families')}",
        f"- Source families: {config.get('palette_swap_source_families')}",
        f"- Category filter: {config.get('palette_swap_category_filter')}",
        f"- Min color confidence: {config.get('palette_swap_min_color_confidence')}",
        f"- Stochastic: {config.get('palette_swap_stochastic')}",
        f"- Keep original probability: {config.get('palette_swap_keep_original_prob')}",
        f"- Require role map: {config.get('palette_swap_require_role_map')}",
        f"- Require explicit color: {config.get('palette_swap_require_explicit_color')}",
        f"- Require explicit caption color: {config.get('palette_swap_require_explicit_caption_color')}",
        f"- Require explicit semantic color: {config.get('palette_swap_require_explicit_semantic_color')}",
        f"- Allow colorless caption if semantic color: {config.get('palette_swap_allow_colorless_caption_if_semantic_color')}",
        f"- No caption prepend: {config.get('palette_swap_no_caption_prepend')}",
        f"- Allow material colors: {config.get('palette_swap_allow_material_colors')}",
        "",
        "## Metrics",
        "",
        f"- Samples evaluated: {metrics.get('sample_count')}",
        f"- Eligible: {metrics.get('eligible_count')} / ineligible: {metrics.get('ineligible_count')}",
        f"- Applied: {metrics.get('applied_count')} (rate {_fmt(metrics.get('applied_rate'))})",
        f"- Swapped / kept original / unchanged ineligible: {metrics.get('swapped_count')}"
        f" / {metrics.get('kept_original_count')} / {metrics.get('unchanged_ineligible_count')}",
        f"- Effective eligible coverage before keep-original: "
        f"{_fmt(metrics.get('effective_eligible_rate_total_before_keep_original'))}",
        f"- Effective swapped rate total/eligible: {_fmt(metrics.get('effective_swapped_rate_total'))}"
        f" / {_fmt(metrics.get('effective_swapped_rate_eligible'))}",
        f"- Effective kept-original rate total/eligible: {_fmt(metrics.get('effective_kept_original_rate_total'))}"
        f" / {_fmt(metrics.get('effective_kept_original_rate_eligible'))}",
        f"- Fallback-heuristic rate (applied): {_fmt(metrics.get('fallback_heuristic_rate'))}",
        f"- Material-conflict drops: {metrics.get('material_conflict_drop_count')}",
        f"- Caption replaced/prepended/no-color: {metrics.get('caption_color_replaced_count')}"
        f" / {metrics.get('caption_color_prepended_count')} / {metrics.get('caption_no_color_count')}",
        f"- Colorless-caption structured-only: {metrics.get('colorless_caption_structured_only_count')}",
        f"- Same-family skips/resamples/recolors: {metrics.get('same_family_skip_count')}"
        f" / {metrics.get('same_family_resample_count')} / {metrics.get('same_family_recolor_count')}",
        f"- Mean RGB delta (visible): {_fmt(metrics.get('mean_rgb_delta_visible'))}",
        f"- Mean palette entries changed: {_fmt(metrics.get('mean_palette_entries_changed'))}",
        f"- Mean visible colors before/after: {_fmt(metrics.get('mean_visible_color_count_before'))}"
        f" / {_fmt(metrics.get('mean_visible_color_count_after'))}",
        "",
        "## Red Flags",
        "",
    ]
    if red_flags:
        for flag in red_flags:
            if isinstance(flag, Mapping):
                lines.append(f"- **{flag.get('name')}**: {flag.get('detail')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Ineligibility Reasons", ""])
    reasons = metrics.get("ineligibility_reason_counts") if isinstance(metrics.get("ineligibility_reason_counts"), Mapping) else {}
    if reasons:
        for reason, count in sorted(reasons.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Target Family Counts (applied)", ""])
    for family, count in sorted((metrics.get("target_family_counts") or {}).items(), key=lambda item: (-int(item[1]), str(item[0]))):
        lines.append(f"- {family}: {count}")

    lines.extend(["", "## Target Diversity Per Sprite", ""])
    diversity_hist = metrics.get("target_family_diversity_histogram_per_sprite")
    if isinstance(diversity_hist, Mapping) and diversity_hist:
        for distinct_count, sprite_count in sorted(diversity_hist.items(), key=lambda item: int(item[0])):
            lines.append(f"- {distinct_count} target families: {sprite_count} sprites")
    else:
        lines.append("- none")

    lines.extend(["", "## Category Coverage", ""])
    coverage = report.get("category_coverage") if isinstance(report.get("category_coverage"), Mapping) else {}
    if coverage:
        for category, stats in sorted(coverage.items()):
            if not isinstance(stats, Mapping):
                continue
            lines.append(
                f"- {category}: total={stats.get('total')}, eligible={stats.get('eligible')}, "
                f"swapped={stats.get('swapped')}, kept_original={stats.get('kept_original')}, "
                f"eligible_rate={_fmt(stats.get('eligible_rate'))}, swapped_rate={_fmt(stats.get('swapped_rate'))}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Source -> Target Matrix (applied)", ""])
    matrix = metrics.get("source_to_target_matrix") if isinstance(metrics.get("source_to_target_matrix"), Mapping) else {}
    for source, targets in sorted(matrix.items()):
        rendered = ", ".join(f"{tgt}:{cnt}" for tgt, cnt in sorted(targets.items(), key=lambda item: (-int(item[1]), str(item[0]))))
        lines.append(f"- {source} -> {rendered}")

    contact_sheets = report.get("contact_sheets") if isinstance(report.get("contact_sheets"), Mapping) else {}
    lines.extend(["", "## Contact Sheets", ""])
    if contact_sheets:
        for name, path in sorted(contact_sheets.items()):
            lines.append(f"- {name}: `{path}`")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


# --- stable schema helpers ---------------------------------------------------


_TOP_LEVEL_METRIC_ALIASES: tuple[str, ...] = (
    "sample_count",
    "applied_count",
    "swapped_count",
    "kept_original_count",
    "effective_eligible_count_before_keep_original",
    "unchanged_ineligible_count",
    "unchanged_not_triggered_count",
    "applied_rate",
    "eligible_count",
    "eligible_rate",
    "ineligible_count",
    "applied_rate_eligible",
    "effective_swapped_rate_total",
    "effective_kept_original_rate_total",
    "effective_eligible_rate_total_before_keep_original",
    "effective_eligible_rate_total",
    "effective_swapped_rate_eligible",
    "effective_kept_original_rate_eligible",
    "kept_original_rate_eligible",
    "fallback_heuristic_rate",
    "material_conflict_drop_count",
    "caption_color_replaced_count",
    "caption_color_prepended_count",
    "caption_no_color_count",
    "colorless_caption_structured_only_count",
    "mean_rgb_delta_visible",
    "mean_palette_entries_changed",
    "mean_visible_color_count_before",
    "mean_visible_color_count_after",
    "target_family_counts",
    "source_family_counts",
    "source_to_target_matrix",
    "ineligibility_reason_counts",
    "same_family_skip_count",
    "per_sprite_target_diversity",
    "target_family_diversity_histogram_per_sprite",
    "category_coverage",
)


def _stable_top_level_fields(metrics: Mapping[str, Any], config: PaletteSwapConfig) -> dict[str, Any]:
    fields = {key: metrics.get(key) for key in _TOP_LEVEL_METRIC_ALIASES}
    fields.update(
        {
            "require_explicit_caption_color": config.require_caption_color(),
            "require_explicit_semantic_color": config.require_semantic_color(),
            "allow_colorless_caption_if_semantic_color": bool(
                config.allow_colorless_caption_if_semantic_color
            ),
            "no_caption_prepend": bool(config.no_caption_prepend),
        }
    )
    return fields


def _config_bool(config: PaletteSwapConfig | Mapping[str, Any] | None, key: str) -> bool:
    if isinstance(config, PaletteSwapConfig):
        if key == "palette_swap_require_explicit_caption_color":
            return config.require_caption_color()
        if key == "palette_swap_require_explicit_semantic_color":
            return config.require_semantic_color()
        if key == "palette_swap_allow_colorless_caption_if_semantic_color":
            return bool(config.allow_colorless_caption_if_semantic_color)
        if key == "palette_swap_no_caption_prepend":
            return bool(config.no_caption_prepend)
        return bool(getattr(config, key.removeprefix("palette_swap_"), False))
    if isinstance(config, Mapping):
        if key == "palette_swap_require_explicit_caption_color":
            return bool(config.get(key) or config.get("require_explicit_caption_color"))
        if key == "palette_swap_require_explicit_semantic_color":
            return bool(config.get(key) or config.get("require_explicit_semantic_color"))
        if key == "palette_swap_allow_colorless_caption_if_semantic_color":
            return bool(config.get(key) or config.get("allow_colorless_caption_if_semantic_color"))
        if key == "palette_swap_no_caption_prepend":
            return bool(config.get(key) or config.get("no_caption_prepend"))
        return bool(config.get(key))
    return False


def _config_target_families(config: PaletteSwapConfig | Mapping[str, Any] | None) -> tuple[str, ...]:
    if isinstance(config, PaletteSwapConfig):
        return tuple(config.families)
    if isinstance(config, Mapping):
        value = config.get("palette_swap_target_families") or config.get("palette_swap_families") or ()
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        if isinstance(value, Sequence):
            return tuple(str(part).strip() for part in value if str(part).strip())
    return ()


def _conservative_red_flag_context(config: PaletteSwapConfig | Mapping[str, Any] | None) -> bool:
    if isinstance(config, PaletteSwapConfig):
        return bool(
            config.require_role_map
            or config.require_caption_color()
            or config.require_semantic_color()
            or config.allow_colorless_caption_if_semantic_color
            or config.no_caption_prepend
            or not config.allow_material_colors
            or config.source_families is not None
            or config.category_filter is not None
            or float(config.min_color_confidence) > 0.0
        )
    if isinstance(config, Mapping):
        return bool(
            config.get("palette_swap_require_role_map")
            or config.get("palette_swap_require_explicit_color")
            or config.get("palette_swap_require_explicit_caption_color")
            or config.get("palette_swap_require_explicit_semantic_color")
            or config.get("palette_swap_allow_colorless_caption_if_semantic_color")
            or config.get("palette_swap_no_caption_prepend")
            or config.get("require_explicit_caption_color")
            or config.get("require_explicit_semantic_color")
            or config.get("no_caption_prepend")
            or config.get("palette_swap_source_families")
            or config.get("palette_swap_category_filter")
            or float(config.get("palette_swap_min_color_confidence") or 0.0) > 0.0
            or config.get("palette_swap_allow_material_colors") is False
        )
    return False


# --- contact sheets ----------------------------------------------------------


def _write_contact_sheets(out_dir: Path, evaluations: Sequence[_SampleEvaluation]) -> dict[str, Path]:
    applied = [item for item in evaluations if item.applied]
    contact_sheets: dict[str, Path] = {}
    if not applied:
        return contact_sheets

    overall = out_dir / "palette_swap_pairs.png"
    if _build_pair_sheet(applied, overall):
        contact_sheets["pairs"] = overall

    _write_grouped_sheets(out_dir / "by_source_category", applied, key="category", label="source_category", into=contact_sheets)
    _write_grouped_sheets(out_dir / "by_target_family", applied, key="target_color_family", label="target_family", into=contact_sheets)
    _write_grouped_sheets(
        out_dir / "by_source_to_target",
        applied,
        key=lambda row: f"{row.get('source_color_family') or 'unknown'}_to_{row.get('target_color_family') or 'unknown'}",
        label="source_to_target",
        into=contact_sheets,
    )
    return contact_sheets


def _write_grouped_sheets(
    group_dir: Path,
    applied: Sequence[_SampleEvaluation],
    *,
    key,
    label: str,
    into: dict[str, Path],
    max_groups: int = 64,
) -> None:
    groups: dict[str, list[_SampleEvaluation]] = {}
    for item in applied:
        group_key = key(item.row) if callable(key) else str(item.row.get(key) or "unknown")
        groups.setdefault(str(group_key or "unknown"), []).append(item)
    for group_key in sorted(groups)[:max_groups]:
        path = group_dir / f"{_safe_name(group_key)}.png"
        if _build_pair_sheet(groups[group_key], path):
            into[f"{label}:{group_key}"] = path


def _build_pair_sheet(items: Sequence[_SampleEvaluation], out_path: Path, *, scale: int = 4, columns: int = 4) -> bool:
    from PIL import Image

    rows = list(items)
    if not rows:
        return False
    cell = SPRITE_SIZE * int(scale)
    padding = int(scale)
    pair_width = cell * 2 + padding  # before | after
    grid_cols = max(1, min(int(columns), len(rows)))
    grid_rows = (len(rows) + grid_cols - 1) // grid_cols
    sheet_width = grid_cols * pair_width + (grid_cols + 1) * padding
    sheet_height = grid_rows * cell + (grid_rows + 1) * padding
    sheet = Image.new("RGBA", (sheet_width, sheet_height), (38, 38, 42, 255))
    for index, item in enumerate(rows):
        before = checkerboard_rgba(rgba_array_to_image(np.asarray(item.before_rgba))).resize((cell, cell), Image.Resampling.NEAREST)
        after = checkerboard_rgba(rgba_array_to_image(np.asarray(item.after_rgba))).resize((cell, cell), Image.Resampling.NEAREST)
        col = index % grid_cols
        row = index // grid_cols
        left = padding + col * (pair_width + padding)
        top = padding + row * (cell + padding)
        sheet.alpha_composite(before, (left, top))
        sheet.alpha_composite(after, (left + cell + padding, top))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


# --- small helpers -----------------------------------------------------------


def _palette_rgb_float(palette: np.ndarray) -> np.ndarray:
    value = np.asarray(palette)
    rgb = value[:, :3].astype(np.float32, copy=False)
    if value.dtype.kind in "ui" or (rgb.size and float(np.nanmax(rgb)) > 1.0):
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def _visible_color_count(rgba: np.ndarray, visible: np.ndarray) -> int:
    if not bool(np.any(visible)):
        return 0
    rgb = np.moveaxis(np.asarray(rgba[:3]), 0, -1)[visible]
    quantized = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return int(np.unique(quantized, axis=0).shape[0])


def _family_histogram(index: np.ndarray, visible: np.ndarray, palette: np.ndarray) -> dict[str, int]:
    used, counts = np.unique(np.asarray(index)[visible].astype(np.int64), return_counts=True)
    histogram: Counter = Counter()
    for palette_index, count in zip(used, counts, strict=False):
        histogram[nearest_family(np.asarray(palette)[int(palette_index), :3])] += int(count)
    return dict(sorted(histogram.items(), key=lambda item: (-int(item[1]), str(item[0]))))


def _mean_rgb_delta(before_rgba: np.ndarray, after_rgba: np.ndarray, visible: np.ndarray) -> float:
    if not bool(np.any(visible)):
        return 0.0
    delta = np.abs(np.asarray(after_rgba[:3], dtype=np.float32) - np.asarray(before_rgba[:3], dtype=np.float32))
    return round(float(delta[:, visible].mean()), 6)


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if row.get(key)) / len(rows))


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return float(sum(values) / len(values)) if values else 0.0


def _flag(name: str, value: float, threshold: float, detail: str) -> dict[str, Any]:
    return {"name": name, "value": round(float(value), 4), "threshold": round(float(threshold), 4), "detail": detail}


def _safe_name(text: str) -> str:
    return "".join(char if (char.isalnum() or char in "_-") else "_" for char in str(text)) or "unknown"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Review palette-swap augmentation without training.")
    parser.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--draws-per-sprite", type=int, default=1)
    parser.add_argument("--review-selection", choices=["first", "random", "balanced", "all"], default="first")
    parser.add_argument("--palette-swap-prob", type=float, default=0.5)
    parser.add_argument("--palette-swap-families", default=DEFAULT_SWAP_FAMILIES_TEXT)
    parser.add_argument("--palette-swap-target-families")
    parser.add_argument("--palette-swap-source-families")
    parser.add_argument("--palette-swap-category-filter")
    parser.add_argument("--palette-swap-min-color-confidence", type=float, default=0.0)
    parser.add_argument("--palette-swap-stochastic", action="store_true", default=False)
    parser.add_argument("--palette-swap-keep-original-prob", type=float, default=0.0)
    parser.add_argument("--palette-swap-require-role-map", action="store_true", default=False)
    parser.add_argument("--palette-swap-require-explicit-color", action="store_true", default=False)
    parser.add_argument("--palette-swap-require-explicit-caption-color", action="store_true", default=False)
    parser.add_argument("--palette-swap-require-explicit-semantic-color", action="store_true", default=False)
    parser.add_argument("--palette-swap-allow-colorless-caption-if-semantic-color", action="store_true", default=False)
    parser.add_argument("--palette-swap-no-caption-prepend", action="store_true", default=False)
    parser.add_argument("--palette-swap-allow-material-colors", type=_bool_arg, default=True)
    parsed = parser.parse_args(argv)
    result = review_palette_swap(PaletteSwapReviewConfig(**vars(parsed)))
    print(f"Evaluated samples: {result.report['metrics']['sample_count']}")
    print(f"Applied: {result.report['metrics']['applied_count']} (rate {result.report['metrics']['applied_rate']:.4f})")
    print(f"Markdown report: {result.markdown_path}")
    print(f"JSON report: {result.json_path}")
    print(f"Contact sheets: {len(result.contact_sheets)}")


def _bool_arg(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
