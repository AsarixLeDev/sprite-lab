"""Controlled prompt/noise sensitivity sampling and deterministic metrics."""

from __future__ import annotations

import itertools
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]

from spritelab.training.conditioning import (
    apply_conditioning_mode,
    checkpoint_conditioning_mode,
    checkpoint_semantic_max_length,
)
from spritelab.training.eval_baseline import resolve_device
from spritelab.training.eval_generator import _load_checkpoint, _tokenizer_from_checkpoint
from spritelab.training.framing_metrics import image_to_rgba_array
from spritelab.training.generated_canonicalizer import (
    build_generation_contact_sheet,
    canonicalize_generated_rgba,
    write_generated_sprite_artifacts,
    write_generation_reports,
)
from spritelab.training.generator_models import TinyCaptionSpriteGenerator
from spritelab.training.sample_generator import read_prompt_records

SPRITE_SIZE = 32
SCHEMA_VERSION = "prompt_sensitivity_v1.0"

THRESHOLDS: dict[str, float] = {
    "near_duplicate_alpha_iou_min": 0.98,
    "near_duplicate_rgb_mae_max": 0.015,
    "near_duplicate_histogram_max": 0.05,
    "prompt_insensitive_mean_difference_max": 0.03,
    "noise_insensitive_diversity_max": 0.03,
    "unstable_shape_mean_alpha_iou_max": 0.45,
    "color_prompt_weak_histogram_max": 0.08,
}

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gold",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow",
}
SHAPE_WORDS = {"round", "square", "tall", "wide", "thin", "curved", "pointed"}
PREFERRED_PROMPT_PAIRS: tuple[tuple[str, str], ...] = (
    ("red potion", "blue potion"),
    ("gold sword", "iron sword"),
    ("square gem", "round gem"),
    ("charged sinew", "calming spores"),
)


@dataclass(frozen=True)
class PromptSensitivityConfig:
    checkpoint: Path
    prompts: Path
    out_dir: Path
    device: str = "cpu"
    seed: int = 123
    max_prompts: int = 32
    noise_samples: int = 16
    max_pairs: int = 8
    max_colors: int = 32
    alpha_threshold: float = 0.5
    batch_size: int = 16


def run_prompt_sensitivity(config: PromptSensitivityConfig) -> dict[str, Any]:
    th = _require_torch()
    started = time.perf_counter()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    ckpt = _load_checkpoint(config.checkpoint)
    tokenizer = _tokenizer_from_checkpoint(ckpt)
    conditioning_mode = checkpoint_conditioning_mode(ckpt)
    semantic_max_length = checkpoint_semantic_max_length(ckpt)
    model = TinyCaptionSpriteGenerator(**dict(ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompts = read_prompt_records(config.prompts, max_records=config.max_prompts)
    if not prompts:
        raise ValueError("prompt sensitivity requires at least one prompt")

    same_noise_records = _generate_prompt_set(
        model=model,
        tokenizer=tokenizer,
        prompt_records=prompts,
        out_dir=out_dir / "same_noise_different_prompts",
        checkpoint=config.checkpoint,
        conditioning_mode=conditioning_mode,
        semantic_max_length=semantic_max_length,
        device=device,
        batch_size=config.batch_size,
        max_colors=config.max_colors,
        alpha_threshold=config.alpha_threshold,
        sample_set="same_noise_different_prompts",
        fixed_noise_seed=int(config.seed),
        seed=int(config.seed),
    )
    same_noise_metrics = summarize_pairwise_differences(
        _pairwise_image_metrics_for_records(out_dir / "same_noise_different_prompts", same_noise_records)
    )
    same_noise_metrics["warnings"] = _same_noise_warnings(same_noise_metrics)

    repeated_prompt = prompts[0]
    noise_prompt_records = [dict(repeated_prompt, noise_index=index) for index in range(max(1, int(config.noise_samples)))]
    same_prompt_records = _generate_prompt_set(
        model=model,
        tokenizer=tokenizer,
        prompt_records=noise_prompt_records,
        out_dir=out_dir / "same_prompt_different_noise",
        checkpoint=config.checkpoint,
        conditioning_mode=conditioning_mode,
        semantic_max_length=semantic_max_length,
        device=device,
        batch_size=config.batch_size,
        max_colors=config.max_colors,
        alpha_threshold=config.alpha_threshold,
        sample_set="same_prompt_different_noise",
        noise_seed_start=int(config.seed) * 100000,
        seed=int(config.seed),
    )
    same_prompt_pair_metrics = _pairwise_image_metrics_for_records(
        out_dir / "same_prompt_different_noise",
        same_prompt_records,
    )
    same_prompt_metrics = summarize_noise_diversity(same_prompt_pair_metrics)
    same_prompt_metrics["prompt_id"] = str(repeated_prompt.get("prompt_id", ""))
    same_prompt_metrics["prompt"] = str(repeated_prompt.get("prompt", ""))
    same_prompt_metrics["warnings"] = _same_prompt_warnings(same_prompt_metrics)

    prompt_pairs = discover_prompt_pairs(prompts, max_pairs=config.max_pairs)
    pair_records, pair_metrics = _generate_prompt_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=prompt_pairs,
        out_dir=out_dir / "prompt_pairs",
        checkpoint=config.checkpoint,
        conditioning_mode=conditioning_mode,
        semantic_max_length=semantic_max_length,
        device=device,
        max_colors=config.max_colors,
        alpha_threshold=config.alpha_threshold,
        seed=int(config.seed),
    )
    prompt_pair_summary = {
        "pair_count": len(pair_metrics),
        "pairs": pair_metrics,
        "near_duplicate_rate": _near_duplicate_rate([pair["metrics"] for pair in pair_metrics]),
        "warnings": sorted({warning for pair in pair_metrics for warning in pair.get("warnings", [])}),
    }

    overview_contact_sheet = _write_overview_contact_sheet(out_dir)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(config.checkpoint),
        "prompts": str(config.prompts),
        "conditioning_mode": conditioning_mode,
        "semantic_max_length": int(semantic_max_length),
        "seed": int(config.seed),
        "thresholds": dict(THRESHOLDS),
        "sets": {
            "same_noise_different_prompts": {
                "sample_count": len(same_noise_records),
                "folder": "same_noise_different_prompts",
                "metrics": same_noise_metrics,
            },
            "same_prompt_different_noise": {
                "sample_count": len(same_prompt_records),
                "folder": "same_prompt_different_noise",
                "metrics": same_prompt_metrics,
            },
            "prompt_pairs": {
                "sample_count": len(pair_records),
                "folder": "prompt_pairs",
                "metrics": prompt_pair_summary,
            },
        },
        "warnings": sorted(
            set(same_noise_metrics["warnings"])
            | set(same_prompt_metrics["warnings"])
            | set(prompt_pair_summary["warnings"])
        ),
        "contact_sheet": None if overview_contact_sheet is None else overview_contact_sheet.name,
        "elapsed_seconds": time.perf_counter() - started,
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
    }
    (out_dir / "prompt_sensitivity_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "prompt_sensitivity_report.md").write_text(format_prompt_sensitivity_markdown(report), encoding="utf-8")
    return report


def pairwise_image_metrics(image_a: Image.Image | np.ndarray, image_b: Image.Image | np.ndarray) -> dict[str, Any]:
    """Compute deterministic pairwise image metrics for canonical 32x32 RGBA sprites."""

    a = image_to_rgba_array(image_a)
    b = image_to_rgba_array(image_b)
    alpha_a = a[..., 3] > 0
    alpha_b = b[..., 3] > 0
    union = alpha_a | alpha_b
    intersection = alpha_a & alpha_b
    union_count = int(np.count_nonzero(union))
    intersection_count = int(np.count_nonzero(intersection))
    alpha_iou = 1.0 if union_count == 0 else intersection_count / float(union_count)
    alpha_mae = float(np.mean(np.abs(a[..., 3].astype(np.float32) - b[..., 3].astype(np.float32)) / 255.0))

    if union_count:
        rgb_mae = float(
            np.mean(np.abs(a[..., :3].astype(np.float32) - b[..., :3].astype(np.float32))[union] / 255.0)
        )
    else:
        rgb_mae = 0.0

    bbox_a = _bbox(alpha_a)
    bbox_b = _bbox(alpha_b)
    center_distance = _bbox_center_distance(bbox_a, bbox_b)
    area_difference = abs(_bbox_area(bbox_a) - _bbox_area(bbox_b)) / float(SPRITE_SIZE * SPRITE_SIZE)
    edge_iou = _mask_iou(_edge_map(alpha_a), _edge_map(alpha_b))
    color_count_difference = abs(_visible_color_count(a) - _visible_color_count(b))
    histogram_distance = _rgb_histogram_distance(a, b)

    metrics = {
        "alpha_iou": float(alpha_iou),
        "alpha_mae": float(alpha_mae),
        "rgb_mae_visible_union": float(rgb_mae),
        "rgb_histogram_distance": float(histogram_distance),
        "color_histogram_distance": float(histogram_distance),
        "bbox_center_distance": center_distance,
        "bbox_area_difference": float(area_difference),
        "color_count_difference": int(color_count_difference),
        "edge_map_iou": float(edge_iou),
    }
    metrics["combined_difference_score"] = combined_difference_score(metrics)
    metrics["near_duplicate"] = is_near_duplicate(metrics)
    return metrics


def combined_difference_score(metrics: Mapping[str, Any]) -> float:
    """Aggregate scale-compatible difference metrics into a conservative score."""

    values = [
        1.0 - float(metrics.get("alpha_iou") or 0.0),
        float(metrics.get("alpha_mae") or 0.0),
        float(metrics.get("rgb_mae_visible_union") or 0.0),
        float(metrics.get("rgb_histogram_distance") or 0.0),
        float(metrics.get("bbox_area_difference") or 0.0),
    ]
    center = metrics.get("bbox_center_distance")
    if isinstance(center, (int, float)) and math.isfinite(float(center)):
        values.append(min(1.0, float(center) / float(SPRITE_SIZE)))
    return float(statistics.fmean(values))


def is_near_duplicate(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics.get("alpha_iou") or 0.0) >= THRESHOLDS["near_duplicate_alpha_iou_min"]
        and float(metrics.get("rgb_mae_visible_union") or 0.0) <= THRESHOLDS["near_duplicate_rgb_mae_max"]
        and float(metrics.get("rgb_histogram_distance") or 0.0) <= THRESHOLDS["near_duplicate_histogram_max"]
    )


def summarize_pairwise_differences(pair_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    differences = [float(metric.get("combined_difference_score") or 0.0) for metric in pair_metrics]
    return {
        "pair_count": len(pair_metrics),
        "mean_pairwise_difference": _mean(differences),
        "median_pairwise_difference": _median(differences),
        "near_duplicate_rate": _near_duplicate_rate(pair_metrics),
        "mean_alpha_iou": _mean(float(metric.get("alpha_iou") or 0.0) for metric in pair_metrics),
        "mean_rgb_distance": _mean(float(metric.get("rgb_mae_visible_union") or 0.0) for metric in pair_metrics),
    }


def summarize_noise_diversity(pair_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = summarize_pairwise_differences(pair_metrics)
    summary["diversity_score"] = summary["mean_pairwise_difference"]
    return summary


def discover_prompt_pairs(records: Sequence[Mapping[str, Any]], *, max_pairs: int = 8) -> list[dict[str, Any]]:
    """Find or synthesize controlled prompt pairs with small text changes."""

    limit = max(0, int(max_pairs))
    if limit == 0:
        return []
    by_prompt: dict[str, Mapping[str, Any]] = {_normalize_prompt(record.get("prompt", "")): record for record in records}
    pairs: list[dict[str, Any]] = []

    def add_pair(a: Mapping[str, Any], b: Mapping[str, Any], pair_id: str, source: str) -> None:
        if len(pairs) >= limit:
            return
        key = tuple(sorted((str(a.get("prompt", "")), str(b.get("prompt", "")))))
        if any(pair.get("_key") == key for pair in pairs):
            return
        pairs.append({"pair_id": pair_id, "a": dict(a), "b": dict(b), "source": source, "_key": key})

    for left, right in PREFERRED_PROMPT_PAIRS:
        a = by_prompt.get(_normalize_prompt(left))
        b = by_prompt.get(_normalize_prompt(right))
        if a is not None and b is not None:
            add_pair(a, b, _pair_id(left, right), "exact_eval_prompt")

    for group_name, token_set in (("color", COLOR_WORDS), ("shape", SHAPE_WORDS)):
        groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
        for record in records:
            tokens = _prompt_tokens(record.get("prompt", ""))
            if not any(token in token_set for token in tokens):
                continue
            skeleton = tuple(token for token in tokens if token not in token_set)
            if skeleton:
                groups.setdefault(skeleton, []).append(record)
        for skeleton, group in sorted(groups.items(), key=lambda item: (item[0], len(item[1]))):
            ordered = sorted(group, key=lambda record: str(record.get("prompt", "")))
            for a, b in itertools.combinations(ordered, 2):
                a_tokens = _prompt_tokens(a.get("prompt", ""))
                b_tokens = _prompt_tokens(b.get("prompt", ""))
                if _single_token_family_change(a_tokens, b_tokens, token_set):
                    add_pair(a, b, _pair_id(str(a.get("prompt", "")), str(b.get("prompt", ""))), f"{group_name}_heuristic")
                    break
            if len(pairs) >= limit:
                break
        if len(pairs) >= limit:
            break

    for left, right in PREFERRED_PROMPT_PAIRS:
        if len(pairs) >= limit:
            break
        a = _synthetic_prompt_record(left, records)
        b = _synthetic_prompt_record(right, records)
        add_pair(a, b, _pair_id(left, right), "synthetic_control")

    for pair in pairs:
        pair.pop("_key", None)
    return pairs


def format_prompt_sensitivity_markdown(report: Mapping[str, Any]) -> str:
    sets = report.get("sets") if isinstance(report.get("sets"), Mapping) else {}
    same_noise = _metrics_for_set(sets, "same_noise_different_prompts")
    same_prompt = _metrics_for_set(sets, "same_prompt_different_noise")
    prompt_pairs = _metrics_for_set(sets, "prompt_pairs")
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []

    lines = [
        "# Prompt Sensitivity Report",
        "",
        f"Checkpoint: `{report.get('checkpoint', '')}`",
        f"Conditioning mode: `{report.get('conditioning_mode', '')}`",
        "",
        "## Summary",
        "",
        f"- Same-noise prompt difference: {_fmt(same_noise.get('mean_pairwise_difference'))}",
        f"- Same-noise near-duplicate rate: {_fmt(same_noise.get('near_duplicate_rate'))}",
        f"- Same-prompt diversity score: {_fmt(same_prompt.get('diversity_score'))}",
        f"- Same-prompt mean alpha IoU: {_fmt(same_prompt.get('mean_alpha_iou'))}",
        f"- Prompt pairs: {int(prompt_pairs.get('pair_count') or 0)}",
        f"- Warnings: {', '.join(str(w) for w in warnings) if warnings else '(none)'}",
        "",
        "These deterministic metrics measure output sensitivity to prompt and noise controls; they do not prove semantic correctness.",
        "",
        "## Prompt Pairs",
        "",
        "| Pair | Alpha IoU | RGB MAE | Histogram | Warnings |",
        "|---|---:|---:|---:|---|",
    ]
    for pair in prompt_pairs.get("pairs") or []:
        if not isinstance(pair, Mapping):
            continue
        metrics = pair.get("metrics") if isinstance(pair.get("metrics"), Mapping) else {}
        pair_warnings = ", ".join(str(w) for w in pair.get("warnings", [])) or "(none)"
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(str(pair.get("pair_id", ""))),
                    _fmt(metrics.get("alpha_iou")),
                    _fmt(metrics.get("rgb_mae_visible_union")),
                    _fmt(metrics.get("rgb_histogram_distance")),
                    _md_escape(pair_warnings),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _generate_prompt_set(
    *,
    model: TinyCaptionSpriteGenerator,
    tokenizer: Any,
    prompt_records: Sequence[Mapping[str, Any]],
    out_dir: Path,
    checkpoint: Path,
    conditioning_mode: str,
    semantic_max_length: int,
    device: Any,
    batch_size: int,
    max_colors: int,
    alpha_threshold: float,
    sample_set: str,
    seed: int,
    fixed_noise_seed: int | None = None,
    noise_seed_start: int | None = None,
) -> list[dict[str, Any]]:
    th = _require_torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    prompt_rows = [dict(record) for record in prompt_records]

    fixed_noise = None
    if fixed_noise_seed is not None:
        fixed_noise = model.sample_noise(1, device=device, seed=int(fixed_noise_seed))

    for batch_start in range(0, len(prompt_rows), max(1, int(batch_size))):
        batch = prompt_rows[batch_start : batch_start + max(1, int(batch_size))]
        caption_tokens = th.as_tensor(
            [tokenizer.encode(str(record["prompt"]), max_length=tokenizer.max_length) for record in batch],
            dtype=th.long,
            device=device,
        )
        semantic_tokens = th.as_tensor(
            [tokenizer.encode_record_semantics(record, max_length=semantic_max_length) for record in batch],
            dtype=th.long,
            device=device,
        )
        if fixed_noise is not None:
            noise = fixed_noise.repeat(len(batch), 1)
            noise_seeds = [int(fixed_noise_seed)] * len(batch)
        else:
            start = int(noise_seed_start if noise_seed_start is not None else seed * 100000)
            noise_seeds = [start + batch_start + index for index in range(len(batch))]
            noise = th.cat([model.sample_noise(1, device=device, seed=noise_seed) for noise_seed in noise_seeds], dim=0)
        inputs = apply_conditioning_mode(
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            mode=conditioning_mode,
            pad_token_id=tokenizer.pad_id,
        )
        with th.no_grad():
            outputs = model(**inputs, noise=noise)
        rgba_batch = _outputs_to_rgba(outputs)
        for index, prompt_record in enumerate(batch):
            sample_index = batch_start + index
            sample_id = f"{sample_set}_{sample_index:06d}"
            sprite = canonicalize_generated_rgba(
                rgba_batch[index],
                max_colors=max_colors,
                alpha_threshold=alpha_threshold,
            )
            metadata = {
                **prompt_record,
                "checkpoint": str(checkpoint),
                "conditioning_mode": conditioning_mode,
                "sample_set": sample_set,
                "seed": int(seed),
                "noise_seed": int(noise_seeds[index]),
                "same_noise": fixed_noise is not None,
                "alpha_threshold": float(alpha_threshold),
                "max_colors": int(max_colors),
            }
            records.append(write_generated_sprite_artifacts(sprite, out_dir, sample_id, metadata))

    contact_sheet = build_generation_contact_sheet(out_dir, records, out_dir / "generation_contact_sheet.png")
    write_generation_reports(
        out_dir=out_dir,
        records=records,
        config={
            "checkpoint": str(checkpoint),
            "conditioning_mode": conditioning_mode,
            "sample_set": sample_set,
            "seed": int(seed),
            "fixed_noise_seed": fixed_noise_seed,
            "noise_seed_start": noise_seed_start,
            "max_colors": int(max_colors),
            "alpha_threshold": float(alpha_threshold),
        },
        contact_sheet=None if contact_sheet is None else contact_sheet.name,
    )
    return records


def _generate_prompt_pairs(
    *,
    model: TinyCaptionSpriteGenerator,
    tokenizer: Any,
    pairs: Sequence[Mapping[str, Any]],
    out_dir: Path,
    checkpoint: Path,
    conditioning_mode: str,
    semantic_max_length: int,
    device: Any,
    max_colors: int,
    alpha_threshold: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generated_records: list[dict[str, Any]] = []
    pair_reports: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        pair_id = str(pair.get("pair_id") or f"pair_{index:04d}")
        pair_seed = int(seed) * 200000 + index
        records = [
            dict(pair["a"], pair_id=pair_id, pair_side="a"),
            dict(pair["b"], pair_id=pair_id, pair_side="b"),
        ]
        pair_generated = _generate_prompt_set(
            model=model,
            tokenizer=tokenizer,
            prompt_records=records,
            out_dir=out_dir,
            checkpoint=checkpoint,
            conditioning_mode=conditioning_mode,
            semantic_max_length=semantic_max_length,
            device=device,
            batch_size=2,
            max_colors=max_colors,
            alpha_threshold=alpha_threshold,
            sample_set=f"prompt_pair_{index:04d}",
            seed=seed,
            fixed_noise_seed=pair_seed,
        )
        generated_records.extend(pair_generated)
        image_a = _record_image(out_dir, pair_generated[0])
        image_b = _record_image(out_dir, pair_generated[1])
        metrics = pairwise_image_metrics(image_a, image_b)
        warnings = _prompt_pair_warnings(pair, metrics)
        pair_reports.append(
            {
                "pair_id": pair_id,
                "prompt_a": str(records[0].get("prompt", "")),
                "prompt_b": str(records[1].get("prompt", "")),
                "prompt_id_a": str(records[0].get("prompt_id", "")),
                "prompt_id_b": str(records[1].get("prompt_id", "")),
                "source": str(pair.get("source", "")),
                "same_noise": True,
                "noise_seed": pair_seed,
                "sample_id_a": str(pair_generated[0].get("sample_id", "")),
                "sample_id_b": str(pair_generated[1].get("sample_id", "")),
                "metrics": metrics,
                "warnings": warnings,
            }
        )

    contact_sheet = build_generation_contact_sheet(out_dir, generated_records, out_dir / "generation_contact_sheet.png")
    write_generation_reports(
        out_dir=out_dir,
        records=generated_records,
        config={
            "checkpoint": str(checkpoint),
            "conditioning_mode": conditioning_mode,
            "sample_set": "prompt_pairs",
            "seed": int(seed),
            "max_colors": int(max_colors),
            "alpha_threshold": float(alpha_threshold),
        },
        contact_sheet=None if contact_sheet is None else contact_sheet.name,
    )
    return generated_records, pair_reports


def _outputs_to_rgba(outputs: Mapping[str, Any]) -> np.ndarray:
    rgb_logits = outputs["rgb_logits"].detach().cpu().numpy().astype(np.float32)
    alpha_logits = outputs["alpha_logits"].detach().cpu().numpy().astype(np.float32)
    rgb = _sigmoid(rgb_logits)
    alpha = _sigmoid(alpha_logits)
    rgba_chw = np.concatenate([rgb, alpha], axis=1)
    return np.moveaxis(rgba_chw, 1, -1).astype(np.float32, copy=False)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


def _pairwise_image_metrics_for_records(generated_dir: Path, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    images = [_record_image(generated_dir, record) for record in records]
    for (index_a, image_a), (index_b, image_b) in itertools.combinations(enumerate(images), 2):
        item = pairwise_image_metrics(image_a, image_b)
        item["sample_id_a"] = str(records[index_a].get("sample_id", ""))
        item["sample_id_b"] = str(records[index_b].get("sample_id", ""))
        item["prompt_id_a"] = str(records[index_a].get("prompt_id", ""))
        item["prompt_id_b"] = str(records[index_b].get("prompt_id", ""))
        metrics.append(item)
    return metrics


def _record_image(generated_dir: Path, record: Mapping[str, Any]) -> Image.Image:
    paths = record.get("paths") if isinstance(record.get("paths"), Mapping) else {}
    rel = paths.get("indexed_png") or paths.get("hard_rgba") or paths.get("raw_rgba")
    if not rel:
        raise ValueError(f"{record.get('sample_id', '')}: generated record has no image path")
    image = Image.open(Path(generated_dir) / str(rel))
    image.load()
    return image.copy().convert("RGBA")


def _same_noise_warnings(metrics: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if (
        int(metrics.get("pair_count") or 0) > 0
        and float(metrics.get("mean_pairwise_difference") or 0.0)
        < THRESHOLDS["prompt_insensitive_mean_difference_max"]
    ):
        warnings.append("prompt_insensitive")
    return warnings


def _same_prompt_warnings(metrics: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if (
        int(metrics.get("pair_count") or 0) > 0
        and float(metrics.get("diversity_score") or 0.0) < THRESHOLDS["noise_insensitive_diversity_max"]
    ):
        warnings.append("noise_insensitive")
    if (
        int(metrics.get("pair_count") or 0) > 0
        and float(metrics.get("mean_alpha_iou") or 1.0) < THRESHOLDS["unstable_shape_mean_alpha_iou_max"]
    ):
        warnings.append("unstable_shape")
    return warnings


def _prompt_pair_warnings(pair: Mapping[str, Any], metrics: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if is_near_duplicate(metrics):
        warnings.append("near_duplicate_pair")
    prompt_a = str(pair.get("a", {}).get("prompt", "") if isinstance(pair.get("a"), Mapping) else "")
    prompt_b = str(pair.get("b", {}).get("prompt", "") if isinstance(pair.get("b"), Mapping) else "")
    if _color_token_changed(prompt_a, prompt_b) and float(metrics.get("rgb_histogram_distance") or 0.0) < THRESHOLDS[
        "color_prompt_weak_histogram_max"
    ]:
        warnings.append("color_prompt_weak")
    return warnings


def _near_duplicate_rate(metrics: Sequence[Mapping[str, Any]]) -> float:
    if not metrics:
        return 0.0
    return float(sum(1 for metric in metrics if is_near_duplicate(metric)) / float(len(metrics)))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    x_min, y_min, x_max, y_max = bbox
    return int((x_max - x_min + 1) * (y_max - y_min + 1))


def _bbox_center_distance(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float | None:
    if a is None or b is None:
        return None
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return float(math.hypot(ax - bx, ay - by))


def _edge_map(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    neighbors_full = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~neighbors_full


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool)
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return 1.0
    intersection = np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)
    return float(np.count_nonzero(intersection) / float(union_count))


def _visible_color_count(rgba: np.ndarray) -> int:
    visible = rgba[..., 3] > 0
    if not bool(np.any(visible)):
        return 0
    return int(np.unique(rgba[..., :3][visible], axis=0).shape[0])


def _rgb_histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(_rgb_histogram(a) - _rgb_histogram(b))))


def _rgb_histogram(rgba: np.ndarray) -> np.ndarray:
    visible = rgba[..., 3] > 0
    if not bool(np.any(visible)):
        return np.zeros((8, 8, 8), dtype=np.float64)
    rgb = rgba[..., :3][visible].astype(np.float64)
    hist, _edges = np.histogramdd(rgb, bins=(8, 8, 8), range=((0, 256), (0, 256), (0, 256)))
    total = float(hist.sum())
    return hist / total if total else hist


def _single_token_family_change(left: Sequence[str], right: Sequence[str], family: set[str]) -> bool:
    if len(left) != len(right):
        return False
    diffs = [(a, b) for a, b in zip(left, right) if a != b]
    return len(diffs) == 1 and diffs[0][0] in family and diffs[0][1] in family


def _color_token_changed(prompt_a: str, prompt_b: str) -> bool:
    left = set(_prompt_tokens(prompt_a)) & COLOR_WORDS
    right = set(_prompt_tokens(prompt_b)) & COLOR_WORDS
    return bool(left or right) and left != right


def _synthetic_prompt_record(prompt: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokens = _prompt_tokens(prompt)
    colors = [token for token in tokens if token in COLOR_WORDS]
    base_tokens = [token for token in tokens if token not in COLOR_WORDS and token not in SHAPE_WORDS]
    base_object = "_".join(base_tokens) if base_tokens else tokens[-1] if tokens else "object"
    return {
        "prompt_id": "synthetic_" + _safe_name(prompt),
        "prompt": prompt,
        "category": "synthetic_control",
        "target_semantics": {
            "base_object": base_object,
            "attributes": {"colors": colors},
        },
    }


def _normalize_prompt(value: Any) -> str:
    return " ".join(_prompt_tokens(value))


def _prompt_tokens(value: Any) -> list[str]:
    text = str(value).replace("_", " ").replace("-", " ").lower()
    return [part for part in text.split() if part]


def _pair_id(left: str, right: str) -> str:
    return f"{_safe_name(left)}__{_safe_name(right)}"


def _safe_name(value: str) -> str:
    cleaned = "_".join(_prompt_tokens(value))
    return cleaned or "prompt"


def _write_overview_contact_sheet(out_dir: Path) -> Path | None:
    sheets: list[Image.Image] = []
    for folder in ("same_noise_different_prompts", "same_prompt_different_noise", "prompt_pairs"):
        path = out_dir / folder / "generation_contact_sheet.png"
        if not path.is_file():
            continue
        image = Image.open(path).convert("RGBA")
        image.load()
        sheets.append(image.copy())
    if not sheets:
        return None
    width = max(image.width for image in sheets)
    height = sum(image.height for image in sheets)
    sheet = Image.new("RGBA", (width, height), (36, 36, 40, 255))
    top = 0
    for image in sheets:
        sheet.alpha_composite(image, (0, top))
        top += image.height
    out_path = out_dir / "prompt_sensitivity_contact_sheet.png"
    sheet.save(out_path)
    return out_path


def _metrics_for_set(sets: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = sets.get(key)
    if not isinstance(value, Mapping):
        return {}
    metrics = value.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _mean(values: Any) -> float:
    data = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return float(statistics.fmean(data)) if data else 0.0


def _median(values: Sequence[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(data)) if data else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for spritelab prompt sensitivity sampling.")
    return torch


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate and measure prompt/noise sensitivity sets.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--noise-samples", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--max-colors", type=int, default=32)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parsed = parser.parse_args(argv)
    report = run_prompt_sensitivity(PromptSensitivityConfig(**vars(parsed)))
    same_noise = report["sets"]["same_noise_different_prompts"]["metrics"]
    same_prompt = report["sets"]["same_prompt_different_noise"]["metrics"]
    print(f"Same-noise mean difference: {same_noise['mean_pairwise_difference']:.6f}")
    print(f"Same-prompt diversity: {same_prompt['diversity_score']:.6f}")
    print(f"Outputs written to {parsed.out_dir}")


if __name__ == "__main__":
    main()
