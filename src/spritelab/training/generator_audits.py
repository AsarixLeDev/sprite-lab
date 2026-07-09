"""Experiment orchestration for generator failure audits."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.training.data import read_jsonl
from spritelab.training.framing_metrics import compute_sprite_framing_metrics
from spritelab.training.generated_qa import qa_generated_sprites
from spritelab.training.generated_review import GeneratedReviewConfig, review_generated_sprites
from spritelab.training.generator_challenger import (
    ChallengerSampleConfig,
    ChallengerTrainConfig,
    run_challenger_training,
    run_sample_generator_challenger,
)
from spritelab.training.ood_prompts import OodCompositionalPromptConfig, build_ood_compositional_prompts
from spritelab.training.overfit_subset import OverfitSubsetSelection, select_overfit_subset
from spritelab.training.palette_swap import DEFAULT_SWAP_FAMILIES_TEXT as PALETTE_SWAP_DEFAULT_FAMILIES
from spritelab.training.prompt_faithfulness import PromptFaithfulnessConfig, run_prompt_faithfulness
from spritelab.training.prompt_records import read_prompt_records
from spritelab.training.prompt_sensitivity import PromptSensitivityConfig, run_prompt_sensitivity
from spritelab.training.rgba import npz_row_to_rgba
from spritelab.training.sample_generator import SampleGeneratorConfig, run_sample_generator
from spritelab.training.source_match_review import SourceMatchReviewConfig, run_source_match_review
from spritelab.training.structured_conditioning import structured_prompt_summary
from spritelab.training.train_generator import GeneratorTrainConfig, run_generator_training


@dataclass(frozen=True)
class RegressionGeneratorAuditConfig:
    dataset: Path
    training_manifest: Path
    out_dir: Path
    device: str = "cpu"
    seed: int = 20260706


@dataclass(frozen=True)
class ChallengerGeneratorAuditConfig:
    dataset: Path
    training_manifest: Path
    out_dir: Path
    architecture: str = "rectified_flow"
    device: str = "cpu"
    seed: int = 20260706


@dataclass(frozen=True)
class FullV4ChallengerAuditConfig:
    dataset: Path
    training_manifest: Path
    out_dir: Path
    export_preset: str | None = None
    architecture: str = "rectified_flow"
    device: str = "cpu"
    seed: int = 20260706
    max_steps: int = 25000
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 0.0002
    conditioning_mode: str = "caption_semantic"
    cfg_dropout: float = 0.1
    structured_field_dropout: float = 0.0
    ema_decay: float = 0.999
    sample_ema: bool = False
    foreground_rgb_loss_weight: float = 1.0
    background_rgb_loss_weight: float = 1.0
    palette_loss_weight: float = 0.0
    palette_loss_temperature: float = 0.05
    palette_swap_augmentation: bool = False
    palette_swap_prob: float = 0.0
    palette_swap_families: str = PALETTE_SWAP_DEFAULT_FAMILIES
    palette_swap_stochastic: bool = False
    palette_swap_keep_original_prob: float = 0.0
    palette_swap_preserve_outline: bool = True
    palette_swap_update_prompts: bool = True
    palette_swap_target_families: str | None = None
    palette_swap_source_families: str | None = None
    palette_swap_category_filter: str | None = None
    palette_swap_min_color_confidence: float = 0.0
    palette_swap_require_role_map: bool = False
    palette_swap_require_explicit_color: bool = False
    palette_swap_require_explicit_caption_color: bool = False
    palette_swap_require_explicit_semantic_color: bool = False
    palette_swap_allow_colorless_caption_if_semantic_color: bool = False
    palette_swap_no_caption_prepend: bool = False
    palette_swap_allow_material_colors: bool = True
    sample_steps: int = 30
    cfg_scale: float = 2.0
    max_colors: int = 32
    alpha_threshold: float = 0.5
    project_palette: bool = False
    project_palette_target_colors: int = 16
    project_palette_min_pixel_share: float = 0.01
    project_palette_method: str = "deterministic_kmeans"
    max_eval_prompts: int = 128
    max_sensitivity_prompts: int = 32
    faithfulness_max_sources: int = 0
    noise_samples: int = 2
    sample_batch_size: int = 16
    eval_prompts: Path | None = None
    reuse_existing_prompts: bool = False
    run_ood_compositional: bool = False
    ood_prompts: Path | None = None
    eval_checkpoints: bool = False
    eval_checkpoint_every: int = 5000
    eval_checkpoint_steps: str | None = None
    checkpoint_eval_max_samples: int | None = None
    amp: bool = True
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 500
    # Opt-in training-loop speed knobs (see docs/training_speed_notes.md); every
    # default here reproduces today's behaviour.
    metrics_every: int = 1
    fused_adamw: bool = False
    cudnn_benchmark: bool = False
    tf32: bool = False
    eval_max_batches: int = 0
    # v2 Phase 1 conditioning architecture (default-off)
    film_conditioning: bool = False
    bottleneck_attention: bool = False
    structured_field_dropout_rates: dict[str, float] | None = None
    # v2 Phase 2 palette/index auxiliary heads (default-off)
    index_head_loss_weight: float = 0.0
    palette_head_loss_weight: float = 0.0
    palette_presence_loss_weight: float = 0.0
    index_head_warmup_steps: int = 0
    palette_head_use_gt_palette_prob: float = 1.0


MICRO_OVERFIT_STEPS_PER_SPRITE = 188
MICRO_OVERFIT_MINIMUM_STEPS = 3000
MICRO_OVERFIT_STEP_ROUNDING = 100

FULL_V4_AUDIT_THRESHOLDS: dict[str, float] = {
    "border_touch_source_delta_warn": 0.10,
    "visible_color_source_delta_warn": 8.0,
    "repeated_silhouette_rate_warn": 0.20,
    "generic_blob_collapse_rate_warn": 0.20,
    "category_consistency_min": 0.65,
    "color_consistency_min": 0.75,
    "prompt_pair_near_duplicate_rate_warn": 0.25,
    "generated_qa_errors_warn": 0.0,
}

OOD_SELECTION_METRIC = "ood_control_score_v1"
OOD_SELECTION_GUARDRAILS: dict[str, float] = {
    "category_min": 0.75,
    "color_min": 0.80,
    "rare_color_max": 0.35,
    "blob_max": 0.35,
}
GROUNDED_SELECTION_GUARDRAILS: dict[str, float] = {
    "category_min": 0.92,
    "color_min": 0.72,
}


def default_micro_overfit_steps(
    sprite_count: int,
    *,
    steps_per_sprite: int = MICRO_OVERFIT_STEPS_PER_SPRITE,
    minimum_steps: int = MICRO_OVERFIT_MINIMUM_STEPS,
    round_to: int = MICRO_OVERFIT_STEP_ROUNDING,
) -> int:
    """Return the default micro-overfit training budget for unique sprites."""
    count = int(sprite_count)
    if count <= 0:
        raise ValueError("sprite_count must be positive")
    raw_steps = max(int(minimum_steps), round(count * int(steps_per_sprite)))
    rounding = int(round_to)
    if rounding > 1:
        raw_steps = int(round(raw_steps / rounding) * rounding)
    return max(int(minimum_steps), raw_steps)


REGRESSION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "overfit_16_sprites_no_margin",
        "count": 16,
        "steps": 3000,
        "margin": False,
        "batch_size": 8,
        "step_count_source": "explicit_regression_specific",
        "budget_note": "Regression-specific explicit budget retained for continuity.",
    },
    {
        "name": "overfit_32_sprites_no_margin",
        "count": 32,
        "steps": 3000,
        "margin": False,
        "batch_size": 8,
        "step_count_source": "explicit_regression_specific",
        "budget_note": "Regression-specific explicit budget retained for continuity.",
    },
    {
        "name": "overfit_32_sprites_with_margin",
        "count": 32,
        "steps": 3000,
        "margin": True,
        "batch_size": 8,
        "step_count_source": "explicit_regression_specific",
        "budget_note": "Regression-specific explicit budget retained for continuity.",
    },
    {
        "name": "overfit_64_sprites_with_margin",
        "count": 64,
        "steps": 5000,
        "margin": True,
        "batch_size": 16,
        "step_count_source": "explicit_regression_specific",
        "budget_note": "Regression-specific explicit budget retained for continuity.",
    },
)

CHALLENGER_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "overfit_16_sprites", "count": 16, "batch_size": 8},
    {"name": "overfit_32_sprites", "count": 32, "batch_size": 8},
    {"name": "overfit_64_sprites", "count": 64, "batch_size": 8},
    {"name": "full_v4_smoke_2k", "count": None, "steps": 2000, "batch_size": 32, "step_count_source": "explicit"},
)


def _resolve_micro_overfit_budget(spec: Mapping[str, Any]) -> dict[str, Any]:
    count = spec.get("count")
    has_explicit_steps = spec.get("steps") is not None
    if has_explicit_steps:
        steps = int(spec["steps"])
        if steps <= 0:
            raise ValueError(f"{spec.get('name', 'audit run')}: steps must be positive")
        source = str(spec.get("step_count_source") or "explicit")
        defaulted = False
    else:
        if count is None:
            raise ValueError(f"{spec.get('name', 'audit run')}: steps must be explicit when count is not set")
        steps = default_micro_overfit_steps(int(count))
        source = "default_micro_overfit_budget"
        defaulted = True
    return {
        "steps": steps,
        "step_count_source": source,
        "step_count_defaulted_from_helper": defaulted,
        "default_steps_per_sprite": MICRO_OVERFIT_STEPS_PER_SPRITE if count is not None else None,
        "default_minimum_steps": MICRO_OVERFIT_MINIMUM_STEPS if count is not None else None,
        "default_step_rounding": MICRO_OVERFIT_STEP_ROUNDING if count is not None else None,
        "budget_note": spec.get("budget_note"),
    }


def run_regression_generator_audit(config: RegressionGeneratorAuditConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(config.training_manifest)
    runs = []
    for spec in REGRESSION_SPECS:
        budget = _resolve_micro_overfit_budget(spec)
        selection = select_overfit_subset(
            rows,
            count=int(spec["count"]),
            split="train",
            seed=int(config.seed),
            stratify="category",
        )
        prompt_file = out_dir / "prompts" / f"{spec['name']}.jsonl"
        _write_subset_prompts(prompt_file, selection)
        run_dir = out_dir / "runs" / str(spec["name"])
        sprite_id_file = _write_overfit_sprite_id_file(
            run_dir / "overfit_sprite_ids.json",
            selection,
            audit_type="regression_generator",
            run_name=str(spec["name"]),
            training_manifest=config.training_manifest,
        )
        generated_dir = out_dir / "generated" / str(spec["name"])
        margin = bool(spec["margin"])
        train_report = run_generator_training(
            GeneratorTrainConfig(
                dataset_dir=config.dataset,
                training_manifest=config.training_manifest,
                out_dir=run_dir,
                batch_size=int(spec["batch_size"]),
                max_steps=int(budget["steps"]),
                device=config.device,
                seed=int(config.seed),
                sample_every=0,
                save_every=0,
                conditioning_mode="caption_semantic",
                max_train_sprites=None,
                sprite_id_list=sprite_id_file,
                overfit_split="train",
                validation_mode="same",
                border_alpha_weight=0.5 if margin else 0.0,
                center_weight=0.05 if margin else 0.0,
                margin_band_weight=0.1 if margin else 0.0,
                margin_band_size=1,
            )
        )
        prompt_sprite_ids = _prompt_sprite_ids(prompt_file)
        subset_check = _assert_overfit_subset_matches(
            train_report,
            prompt_sprite_ids,
            run_name=str(spec["name"]),
        )
        sample_report = run_sample_generator(
            SampleGeneratorConfig(
                checkpoint=run_dir / "checkpoint_last.pt",
                prompts=prompt_file,
                out_dir=generated_dir,
                max_samples=int(spec["count"]),
                device=config.device,
                seed=int(config.seed),
                contact_sheet_labels="prompt_and_seed",
            )
        )
        generated_subset_check = _assert_generated_targets_match_prompts(
            generated_dir,
            prompt_sprite_ids,
            run_name=str(spec["name"]),
        )
        qa = qa_generated_sprites(generated_dir).to_json_dict()
        review = review_generated_sprites(
            GeneratedReviewConfig(
                generated_dir=generated_dir,
                out=generated_dir / "generated_review_report.md",
                out_json=generated_dir / "generated_review_report.json",
                out_dir=generated_dir / "review",
                group_by="category",
                compare_raw_indexed=True,
            )
        ).report
        source_match = run_source_match_review(
            SourceMatchReviewConfig(
                generated=generated_dir,
                dataset=config.dataset,
                training_manifest=config.training_manifest,
                out=generated_dir / "source_match",
            )
        )
        source_match_subset_check = _assert_source_match_targets_match_prompts(
            source_match,
            prompt_sprite_ids,
            run_name=str(spec["name"]),
        )
        runs.append(
            _audit_run_summary(
                spec,
                budget=budget,
                train_report=train_report,
                sample_report=sample_report,
                qa=qa,
                review=review,
                source_match=source_match,
                overfit_subset=subset_check,
                generated_target_subset=generated_subset_check,
                source_match_target_subset=source_match_subset_check,
                overfit_sprite_id_file=sprite_id_file,
            )
        )
    report = {
        "audit_type": "regression_generator",
        "dataset": str(config.dataset),
        "training_manifest": str(config.training_manifest),
        "seed": int(config.seed),
        "runs": runs,
        "decision": _regression_decision(runs),
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
    }
    _write_audit_report(out_dir, report, filename="audit_report")
    return report


def run_challenger_generator_audit(config: ChallengerGeneratorAuditConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(config.training_manifest)
    eval_prompts = Path(config.dataset) / "eval_prompts.jsonl"
    runs = []
    for spec in CHALLENGER_SPECS:
        budget = _resolve_micro_overfit_budget(spec)
        count = spec.get("count")
        selection = (
            select_overfit_subset(rows, count=int(count), split="train", seed=int(config.seed), stratify="category")
            if count is not None
            else None
        )
        prompt_file = out_dir / "prompts" / f"{spec['name']}.jsonl"
        if selection is not None:
            _write_subset_prompts(prompt_file, selection)
            max_samples = int(count)
        else:
            prompt_file = eval_prompts
            max_samples = 64
        run_dir = out_dir / "runs" / str(spec["name"])
        sprite_id_file = (
            _write_overfit_sprite_id_file(
                run_dir / "overfit_sprite_ids.json",
                selection,
                audit_type="challenger_generator",
                run_name=str(spec["name"]),
                training_manifest=config.training_manifest,
            )
            if selection is not None
            else None
        )
        generated_dir = out_dir / "generated" / str(spec["name"])
        train_report = run_challenger_training(
            ChallengerTrainConfig(
                dataset_dir=config.dataset,
                training_manifest=config.training_manifest,
                out_dir=run_dir,
                architecture=config.architecture,
                batch_size=int(spec["batch_size"]),
                max_steps=int(budget["steps"]),
                device=config.device,
                seed=int(config.seed),
                conditioning_mode="caption_semantic",
                cfg_dropout=0.1,
                max_train_sprites=None,
                sprite_id_list=sprite_id_file,
                overfit_split="train" if count is not None else None,
                validation_mode="same" if count is not None else "val",
                sample_every=0,
                save_every=0,
            )
        )
        prompt_sprite_ids = _prompt_sprite_ids(prompt_file) if selection is not None else None
        subset_check = (
            _assert_overfit_subset_matches(
                train_report,
                prompt_sprite_ids,
                run_name=str(spec["name"]),
            )
            if prompt_sprite_ids is not None
            else None
        )
        sample_report = run_sample_generator_challenger(
            ChallengerSampleConfig(
                checkpoint=run_dir / "checkpoint_last.pt",
                prompts=prompt_file,
                out_dir=generated_dir,
                max_samples=max_samples,
                steps=30,
                cfg_scale=2.0,
                device=config.device,
                seed=int(config.seed),
                contact_sheet_labels="prompt_and_seed",
            )
        )
        generated_subset_check = (
            _assert_generated_targets_match_prompts(
                generated_dir,
                prompt_sprite_ids,
                run_name=str(spec["name"]),
            )
            if prompt_sprite_ids is not None
            else None
        )
        qa = qa_generated_sprites(generated_dir).to_json_dict()
        review = review_generated_sprites(
            GeneratedReviewConfig(
                generated_dir=generated_dir,
                out=generated_dir / "generated_review_report.md",
                out_json=generated_dir / "generated_review_report.json",
                out_dir=generated_dir / "review",
                group_by="category",
                compare_raw_indexed=True,
            )
        ).report
        source_match = (
            run_source_match_review(
                SourceMatchReviewConfig(
                    generated=generated_dir,
                    dataset=config.dataset,
                    training_manifest=config.training_manifest,
                    out=generated_dir / "source_match",
                )
            )
            if selection is not None
            else None
        )
        source_match_subset_check = (
            _assert_source_match_targets_match_prompts(
                source_match,
                prompt_sprite_ids,
                run_name=str(spec["name"]),
            )
            if source_match is not None and prompt_sprite_ids is not None
            else None
        )
        sensitivity = run_prompt_sensitivity(
            PromptSensitivityConfig(
                checkpoint=run_dir / "checkpoint_last.pt",
                prompts=prompt_file,
                out_dir=generated_dir / "prompt_sensitivity",
                device=config.device,
                seed=int(config.seed),
                max_prompts=min(max_samples, 32),
                noise_samples=8,
                max_pairs=8,
            )
        )
        faithfulness = run_prompt_faithfulness(
            PromptFaithfulnessConfig(
                generated=generated_dir,
                prompts=prompt_file,
                dataset=config.dataset,
                out=generated_dir / "prompt_faithfulness_report.md",
                out_json=generated_dir / "prompt_faithfulness_report.json",
            )
        )
        runs.append(
            _audit_run_summary(
                spec,
                budget=budget,
                train_report=train_report,
                sample_report=sample_report,
                qa=qa,
                review=review,
                source_match=source_match,
                prompt_sensitivity=sensitivity,
                prompt_faithfulness=faithfulness,
                overfit_subset=subset_check,
                generated_target_subset=generated_subset_check,
                source_match_target_subset=source_match_subset_check,
                overfit_sprite_id_file=sprite_id_file,
            )
        )
    report = {
        "audit_type": "challenger_generator",
        "dataset": str(config.dataset),
        "training_manifest": str(config.training_manifest),
        "seed": int(config.seed),
        "architecture": config.architecture,
        "runs": runs,
        "decision": _challenger_decision(runs),
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
    }
    _write_audit_report(out_dir, report, filename="audit_report")
    return report


def build_balanced_eval_prompts(
    *,
    training_manifest: str | Path,
    out: str | Path,
    max_prompts: int = 128,
) -> dict[str, Any]:
    rows = read_jsonl(training_manifest)
    first_by_sprite: dict[str, dict[str, Any]] = {}
    for row in rows:
        sprite_id = str(row.get("sprite_id") or "").strip()
        if sprite_id and sprite_id not in first_by_sprite:
            first_by_sprite[sprite_id] = dict(row)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in first_by_sprite.values():
        by_category[_category(row)].append(row)

    categories = sorted(by_category)
    selected: list[dict[str, Any]] = []
    limit = max(0, int(max_prompts))
    while len(selected) < limit and any(by_category.values()):
        for category in categories:
            if len(selected) >= limit:
                break
            if by_category[category]:
                selected.append(by_category[category].pop(0))

    prompt_rows = [_eval_prompt_row(row, index) for index, row in enumerate(selected)]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in prompt_rows),
        encoding="utf-8",
    )
    return _prompt_set_summary(prompt_rows, prompt_file=out_path, reused_existing=False)


def run_full_v4_challenger_audit(config: FullV4ChallengerAuditConfig) -> dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = _resolve_full_v4_prompt_file(config)
    if config.reuse_existing_prompts and prompt_file.is_file():
        prompt_summary = _read_prompt_set_summary(prompt_file, max_prompts=config.max_eval_prompts)
    else:
        prompt_summary = build_balanced_eval_prompts(
            training_manifest=config.training_manifest,
            out=prompt_file,
            max_prompts=config.max_eval_prompts,
        )
    checkpoint_steps = _resolve_eval_checkpoint_steps(config)

    train_dir = out_dir / _full_v4_train_dir_name(config.max_steps)
    generated_dir = out_dir / "generated_eval"
    train_report = run_challenger_training(
        ChallengerTrainConfig(
            dataset_dir=config.dataset,
            training_manifest=config.training_manifest,
            out_dir=train_dir,
            architecture=config.architecture,
            batch_size=int(config.batch_size),
            max_steps=int(config.max_steps),
            learning_rate=float(config.learning_rate),
            device=config.device,
            seed=int(config.seed),
            num_workers=int(config.num_workers),
            conditioning_mode=config.conditioning_mode,
            cfg_dropout=float(config.cfg_dropout),
            structured_field_dropout=float(config.structured_field_dropout),
            ema_decay=float(config.ema_decay),
            foreground_rgb_loss_weight=float(config.foreground_rgb_loss_weight),
            background_rgb_loss_weight=float(config.background_rgb_loss_weight),
            palette_loss_weight=float(config.palette_loss_weight),
            palette_loss_temperature=float(config.palette_loss_temperature),
            palette_swap_augmentation=bool(config.palette_swap_augmentation),
            palette_swap_prob=float(config.palette_swap_prob),
            palette_swap_families=str(config.palette_swap_families),
            palette_swap_stochastic=bool(config.palette_swap_stochastic),
            palette_swap_keep_original_prob=float(config.palette_swap_keep_original_prob),
            palette_swap_preserve_outline=bool(config.palette_swap_preserve_outline),
            palette_swap_update_prompts=bool(config.palette_swap_update_prompts),
            palette_swap_target_families=config.palette_swap_target_families,
            palette_swap_source_families=config.palette_swap_source_families,
            palette_swap_category_filter=config.palette_swap_category_filter,
            palette_swap_min_color_confidence=float(config.palette_swap_min_color_confidence),
            palette_swap_require_role_map=bool(config.palette_swap_require_role_map),
            palette_swap_require_explicit_color=bool(config.palette_swap_require_explicit_color),
            palette_swap_require_explicit_caption_color=bool(config.palette_swap_require_explicit_caption_color),
            palette_swap_require_explicit_semantic_color=bool(config.palette_swap_require_explicit_semantic_color),
            palette_swap_allow_colorless_caption_if_semantic_color=bool(
                config.palette_swap_allow_colorless_caption_if_semantic_color
            ),
            palette_swap_no_caption_prepend=bool(config.palette_swap_no_caption_prepend),
            palette_swap_allow_material_colors=bool(config.palette_swap_allow_material_colors),
            sample_every=0,
            save_every=0,
            checkpoint_steps=checkpoint_steps,
            validation_mode="val",
            amp=bool(config.amp),
            lr_schedule=str(config.lr_schedule),
            lr_warmup_steps=int(config.lr_warmup_steps),
            metrics_every=int(config.metrics_every),
            fused_adamw=bool(config.fused_adamw),
            cudnn_benchmark=bool(config.cudnn_benchmark),
            tf32=bool(config.tf32),
            eval_max_batches=int(config.eval_max_batches),
            film_conditioning=bool(config.film_conditioning),
            bottleneck_attention=bool(config.bottleneck_attention),
            structured_field_dropout_rates=config.structured_field_dropout_rates,
            index_head_loss_weight=float(config.index_head_loss_weight),
            palette_head_loss_weight=float(config.palette_head_loss_weight),
            palette_presence_loss_weight=float(config.palette_presence_loss_weight),
            index_head_warmup_steps=int(config.index_head_warmup_steps),
            palette_head_use_gt_palette_prob=float(config.palette_head_use_gt_palette_prob),
        )
    )
    final_sample_checkpoint = train_dir / ("checkpoint_last_ema.pt" if config.sample_ema else "checkpoint_last.pt")
    if config.sample_ema and not final_sample_checkpoint.is_file():
        if str(config.export_preset or "").lower() in {"v1", "phase1_v1"}:
            final_sample_checkpoint = train_dir / "checkpoint_last.pt"
        else:
            raise FileNotFoundError(f"--sample-ema requested but EMA checkpoint is missing: {final_sample_checkpoint}")
    sample_checkpoint = final_sample_checkpoint
    checkpoint_evaluation: dict[str, Any] | None = None
    ood_prompt_file: Path | None = None
    ood_prompt_summary: dict[str, Any] | None = None
    if config.eval_checkpoints:
        ood_prompt_file, ood_prompt_summary = _prepare_full_v4_ood_prompts(config)
        checkpoint_evaluation = _evaluate_full_v4_checkpoint_candidates(
            config,
            train_dir=train_dir,
            checkpoint_steps=checkpoint_steps,
            grounded_prompt_file=prompt_file,
            grounded_prompt_summary=prompt_summary,
            ood_prompt_file=ood_prompt_file,
            ood_prompt_summary=ood_prompt_summary,
        )
        selected_checkpoint = checkpoint_evaluation.get("selected_checkpoint")
        if selected_checkpoint:
            sample_checkpoint = Path(str(selected_checkpoint))
    sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=sample_checkpoint,
            prompts=prompt_file,
            out_dir=generated_dir,
            export_preset=config.export_preset,
            max_samples=int(prompt_summary["prompt_count"]),
            steps=int(config.sample_steps),
            cfg_scale=float(config.cfg_scale),
            max_colors=int(config.max_colors),
            alpha_threshold=float(config.alpha_threshold),
            device=config.device,
            seed=int(config.seed),
            batch_size=int(config.sample_batch_size),
            dither=False,
            write_raw_rgba=True,
            write_hard_rgba=True,
            contact_sheet_labels="prompt_and_seed",
            project_palette=bool(config.project_palette),
            project_palette_target_colors=int(config.project_palette_target_colors),
            project_palette_min_pixel_share=float(config.project_palette_min_pixel_share),
            project_palette_method=str(config.project_palette_method),
        )
    )
    qa = qa_generated_sprites(generated_dir).to_json_dict()
    review_result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=generated_dir,
            out=generated_dir / "generated_review_report.md",
            out_json=generated_dir / "generated_review_report.json",
            out_dir=generated_dir / "review",
            group_by="category",
            max_samples_per_sheet=int(config.max_eval_prompts),
            compare_raw_indexed=True,
        )
    )
    faithfulness_json = generated_dir / "prompt_faithfulness_report.json"
    faithfulness = run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=generated_dir,
            prompts=prompt_file,
            dataset=config.dataset,
            out=generated_dir / "prompt_faithfulness_report.md",
            out_json=faithfulness_json,
            max_sources=int(config.faithfulness_max_sources),
            source_selection="auto",
        )
    )
    _verify_faithfulness_matches_disk(faithfulness, faithfulness_json)
    sensitivity = run_prompt_sensitivity(
        PromptSensitivityConfig(
            checkpoint=sample_checkpoint,
            prompts=prompt_file,
            out_dir=generated_dir / "prompt_sensitivity",
            device=config.device,
            seed=int(config.seed),
            max_prompts=int(config.max_sensitivity_prompts),
            noise_samples=int(config.noise_samples),
            max_pairs=8,
            max_colors=int(config.max_colors),
            alpha_threshold=float(config.alpha_threshold),
            batch_size=int(config.sample_batch_size),
        )
    )

    sections = {
        "training": _full_v4_training_summary(train_report),
        "palette_swap": _full_v4_palette_swap_summary(train_report, config),
        "prompt_set": prompt_summary,
        "generation": _full_v4_generation_summary(sample_report, generated_dir=generated_dir),
        "palette_projection": _full_v4_palette_projection_summary(generated_dir, sample_report),
        "generated_qa": _full_v4_qa_summary(qa),
        "generated_review": _full_v4_generated_review_summary(review_result.report, max_colors=int(config.max_colors)),
        "prompt_faithfulness": _full_v4_faithfulness_summary(faithfulness),
        "prompt_sensitivity": _full_v4_sensitivity_summary(sensitivity),
    }
    source_distribution = _source_distribution_report(
        dataset=config.dataset,
        training_manifest=config.training_manifest,
        grounded_category_counts=prompt_summary.get("category_counts", {}),
    )
    sections["source_distribution"] = source_distribution
    sections["generated_vs_source"] = {
        "grounded_eval": _generated_vs_source_delta(
            sections["generated_review"],
            _section(source_distribution, "grounded_eval_category_weighted"),
        )
    }
    ood_artifacts: dict[str, Any] = {}
    if config.run_ood_compositional or config.eval_checkpoints:
        if ood_prompt_file is None or ood_prompt_summary is None:
            ood_prompt_file, ood_prompt_summary = _prepare_full_v4_ood_prompts(config)
        ood_generated_dir = out_dir / "generated_ood_compositional"
        ood_section, ood_artifacts = _run_full_v4_ood_evaluation(
            config,
            checkpoint=sample_checkpoint,
            ood_prompt_file=ood_prompt_file,
            ood_prompt_summary=ood_prompt_summary,
            generated_dir=ood_generated_dir,
            seed=int(config.seed) + 17,
            include_sensitivity=True,
            faithfulness_max_sources=int(config.faithfulness_max_sources),
        )
        sections["ood_compositional"] = ood_section
        source_distribution["ood_eval_category_weighted"] = _category_weighted_source_baseline(
            _section(source_distribution, "by_category"),
            ood_prompt_summary.get("category_counts", {}),
        )
        sections["generated_vs_source"]["ood_eval"] = _generated_vs_source_delta(
            sections["ood_compositional"]["generated_review"],
            _section(source_distribution, "ood_eval_category_weighted"),
        )
    decision = decide_full_v4_challenger_audit(sections)
    report = {
        "schema_version": "full_v4_challenger_audit_v1.0",
        "audit_type": "challenger_full_v4",
        "dataset": str(config.dataset),
        "training_manifest": str(config.training_manifest),
        "seed": int(config.seed),
        "architecture": config.architecture,
        "conditioning_mode": config.conditioning_mode,
        "config": {key: _jsonable(value) for key, value in asdict(config).items()},
        "artifacts": {
            "train_dir": str(train_dir),
            "checkpoint": str(train_dir / "checkpoint_last.pt"),
            "final_step_sample_checkpoint": str(final_sample_checkpoint),
            "sample_checkpoint": str(sample_checkpoint),
            "sampled_ema": str(sample_checkpoint).endswith("_ema.pt"),
            "checkpoint_selection_enabled": bool(config.eval_checkpoints),
            "eval_prompts": str(prompt_file),
            "generated_dir": str(generated_dir),
            "generated_contact_sheet": None
            if sample_report.get("contact_sheet") is None
            else str(generated_dir / str(sample_report.get("contact_sheet"))),
            "palette_projection_report": str(generated_dir / "palette_projection_report.json")
            if (generated_dir / "palette_projection_report.json").is_file()
            else None,
            "palette_projection_contact_sheet": str(generated_dir / "contact_sheet_projected.png")
            if (generated_dir / "contact_sheet_projected.png").is_file()
            else None,
            "review_contact_sheets": review_result.report.get("contact_sheets", {}),
            **ood_artifacts,
        },
        **sections,
        "decision": decision,
        "thresholds": dict(FULL_V4_AUDIT_THRESHOLDS),
    }
    if checkpoint_evaluation is not None:
        report["checkpoint_evaluation"] = checkpoint_evaluation
        report["final_step_metrics"] = checkpoint_evaluation.get("final_step_metrics")
        report["selected_checkpoint_metrics"] = checkpoint_evaluation.get("selected_metrics")
    _write_full_v4_audit_report(out_dir, report)
    return report


def decide_full_v4_challenger_audit(sections: Mapping[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    _section(sections, "training")
    qa = _section(sections, "generated_qa")
    review = _section(sections, "generated_review")
    faithfulness = _section(sections, "prompt_faithfulness")
    sensitivity = _section(sections, "prompt_sensitivity")
    ood = _section(sections, "ood_compositional")
    generated_vs_source = _section(sections, "generated_vs_source")

    _warn_if_gt(warnings, "generated_qa_errors", qa.get("errors"), FULL_V4_AUDIT_THRESHOLDS["generated_qa_errors_warn"])
    _warn_source_delta(
        warnings,
        "border_touch_source_delta",
        _section(_section(generated_vs_source, "grounded_eval"), "deltas").get("border_touch_rate"),
        FULL_V4_AUDIT_THRESHOLDS["border_touch_source_delta_warn"],
    )
    _warn_source_delta(
        warnings,
        "visible_color_count_source_delta",
        _section(_section(generated_vs_source, "grounded_eval"), "deltas").get("mean_visible_color_count"),
        FULL_V4_AUDIT_THRESHOLDS["visible_color_source_delta_warn"],
    )
    if bool(review.get("median_visible_colors_pinned")):
        warnings.append(
            {
                "name": "median_visible_colors_pinned",
                "value": review.get("median_visible_color_count"),
                "threshold": review.get("max_colors"),
                "direction": "pinned",
            }
        )
    _warn_if_gt(
        warnings,
        "repeated_silhouette_rate",
        faithfulness.get("repeated_silhouette_rate"),
        FULL_V4_AUDIT_THRESHOLDS["repeated_silhouette_rate_warn"],
    )
    _warn_if_gt(
        warnings,
        "generic_blob_collapse_rate",
        faithfulness.get("generic_blob_collapse_rate"),
        FULL_V4_AUDIT_THRESHOLDS["generic_blob_collapse_rate_warn"],
    )
    _warn_if_lt(
        warnings,
        "category_consistency_rate",
        faithfulness.get("category_consistency_rate"),
        FULL_V4_AUDIT_THRESHOLDS["category_consistency_min"],
    )
    _warn_if_lt(
        warnings,
        "color_consistency_rate",
        faithfulness.get("color_consistency_rate"),
        FULL_V4_AUDIT_THRESHOLDS["color_consistency_min"],
    )
    _warn_if_gt(
        warnings,
        "prompt_pair_near_duplicate_rate",
        sensitivity.get("prompt_pair_near_duplicate_rate"),
        FULL_V4_AUDIT_THRESHOLDS["prompt_pair_near_duplicate_rate_warn"],
    )
    if ood:
        ood_qa = _section(ood, "generated_qa")
        ood_review = _section(ood, "generated_review")
        ood_faithfulness = _section(ood, "prompt_faithfulness")
        ood_sensitivity = _section(ood, "prompt_sensitivity")
        _warn_if_gt(
            warnings,
            "ood_generated_qa_errors",
            ood_qa.get("errors"),
            FULL_V4_AUDIT_THRESHOLDS["generated_qa_errors_warn"],
        )
        _warn_source_delta(
            warnings,
            "ood_border_touch_source_delta",
            _section(_section(generated_vs_source, "ood_eval"), "deltas").get("border_touch_rate"),
            FULL_V4_AUDIT_THRESHOLDS["border_touch_source_delta_warn"],
        )
        _warn_source_delta(
            warnings,
            "ood_visible_color_count_source_delta",
            _section(_section(generated_vs_source, "ood_eval"), "deltas").get("mean_visible_color_count"),
            FULL_V4_AUDIT_THRESHOLDS["visible_color_source_delta_warn"],
        )
        if bool(ood_review.get("median_visible_colors_pinned")):
            warnings.append(
                {
                    "name": "ood_median_visible_colors_pinned",
                    "value": ood_review.get("median_visible_color_count"),
                    "threshold": ood_review.get("max_colors"),
                    "direction": "pinned",
                }
            )
        _warn_if_lt(
            warnings,
            "ood_category_consistency_rate",
            ood_faithfulness.get("category_consistency_rate"),
            FULL_V4_AUDIT_THRESHOLDS["category_consistency_min"],
        )
        _warn_if_lt(
            warnings,
            "ood_color_consistency_rate",
            ood_faithfulness.get("color_consistency_rate"),
            FULL_V4_AUDIT_THRESHOLDS["color_consistency_min"],
        )
        _warn_if_gt(
            warnings,
            "ood_generic_blob_collapse_rate",
            ood_faithfulness.get("generic_blob_collapse_rate"),
            FULL_V4_AUDIT_THRESHOLDS["generic_blob_collapse_rate_warn"],
        )
        _warn_if_gt(
            warnings,
            "ood_prompt_pair_near_duplicate_rate",
            ood_sensitivity.get("prompt_pair_near_duplicate_rate"),
            FULL_V4_AUDIT_THRESHOLDS["prompt_pair_near_duplicate_rate_warn"],
        )

    names = {str(item.get("name")) for item in warnings}
    if names & {"generated_qa_errors", "ood_generated_qa_errors"}:
        code = "B"
        label = "Needs evaluation/prompt-set fixes before training conclusions."
    elif names & {
        "category_consistency_rate",
        "color_consistency_rate",
        "repeated_silhouette_rate",
        "generic_blob_collapse_rate",
        "prompt_pair_near_duplicate_rate",
        "ood_category_consistency_rate",
        "ood_color_consistency_rate",
        "ood_generic_blob_collapse_rate",
        "ood_prompt_pair_near_duplicate_rate",
    }:
        code = "D"
        label = "Needs conditioning/control changes, but keep architecture/loss unchanged until the harness is stable."
    elif names & {
        "border_touch_source_delta",
        "visible_color_count_source_delta",
        "median_visible_colors_pinned",
        "ood_border_touch_source_delta",
        "ood_visible_color_count_source_delta",
        "ood_median_visible_colors_pinned",
    }:
        code = "C"
        label = "Needs source-relative framing or palette-noise fixes before broader model changes."
    elif names:
        code = "F"
        label = "Unknown."
    else:
        code = "A"
        label = "Healthy enough for longer full-v4 training."
    return {"code": code, "label": label, "warnings": warnings}


def compare_challenger_conditioning_audits(
    *,
    baseline: str | Path | Mapping[str, Any],
    structured: str | Path | Mapping[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    baseline_report = _report_with_computed_source_deltas(_load_full_v4_report(baseline))
    structured_report = _report_with_computed_source_deltas(_load_full_v4_report(structured))
    source_selection_comparison = _source_selection_comparison(baseline_report, structured_report)
    checkpoint_selection_comparison = _checkpoint_selection_comparison(baseline_report, structured_report)
    projection_comparison = _palette_projection_comparison(baseline_report, structured_report)
    warnings = [
        *source_selection_comparison["warnings"],
        *checkpoint_selection_comparison["warnings"],
        *projection_comparison["warnings"],
        *_cfg_scale_comparison_warnings(baseline_report, structured_report),
    ]
    grounded_baseline_metrics = _comparison_metrics_for_eval(baseline_report, eval_key="grounded")
    grounded_structured_metrics = _comparison_metrics_for_eval(structured_report, eval_key="grounded")
    ood_baseline_metrics = _comparison_metrics_for_eval(baseline_report, eval_key="ood")
    ood_structured_metrics = _comparison_metrics_for_eval(structured_report, eval_key="ood")
    grounded_comparisons = _comparison_table(grounded_baseline_metrics, grounded_structured_metrics)
    ood_comparisons = _comparison_table(ood_baseline_metrics, ood_structured_metrics)
    rare_color_comparison = _rare_color_comparison(
        baseline_report,
        structured_report,
        fallback=ood_comparisons.get("too_many_rare_colors_rate")
        or grounded_comparisons.get("too_many_rare_colors_rate"),
    )
    answers = {
        "dataset_grounded_category": _metric_status(grounded_comparisons.get("category_consistency")),
        "ood_category": _metric_status(ood_comparisons.get("category_consistency")),
        "dataset_grounded_color": _metric_status(grounded_comparisons.get("color_consistency")),
        "ood_color": _metric_status(ood_comparisons.get("color_consistency")),
        "ood_blob_collapse": _metric_status(ood_comparisons.get("generic_blob_collapse")),
        "rare_color_rate": _metric_status(rare_color_comparison),
        "ood_category_consistency_improved": _improved(ood_comparisons.get("category_consistency")),
        "ood_color_consistency_improved": _improved(ood_comparisons.get("color_consistency")),
        "generic_blob_collapse_reduced": _improved(ood_comparisons.get("generic_blob_collapse")),
        "generic_potion_collapse_reduced": _improved(ood_comparisons.get("generic_potion_collapse")),
        "micro_overfit_preserved": "not_assessed_by_full_v4_comparison",
    }
    report = {
        "schema_version": "challenger_conditioning_comparison_v1.4",
        "warnings": warnings,
        "source_selection_comparison": source_selection_comparison,
        "checkpoint_selection_comparison": checkpoint_selection_comparison,
        "projection_comparison": projection_comparison,
        "baseline": {
            "conditioning_mode": baseline_report.get("conditioning_mode"),
            "source_selection": source_selection_comparison["baseline"],
            "checkpoint_selection": checkpoint_selection_comparison["baseline"],
            "palette_projection": projection_comparison["baseline"],
            "selected_checkpoint": checkpoint_selection_comparison["baseline"].get("selected_checkpoint"),
            "selected_step": checkpoint_selection_comparison["baseline"].get("selected_step"),
            "selection_score": checkpoint_selection_comparison["baseline"].get("selection_score"),
            "dataset_grounded_metrics": grounded_baseline_metrics,
            "ood_metrics": ood_baseline_metrics,
        },
        "structured": {
            "conditioning_mode": structured_report.get("conditioning_mode"),
            "source_selection": source_selection_comparison["structured"],
            "checkpoint_selection": checkpoint_selection_comparison["structured"],
            "palette_projection": projection_comparison["structured"],
            "selected_checkpoint": checkpoint_selection_comparison["structured"].get("selected_checkpoint"),
            "selected_step": checkpoint_selection_comparison["structured"].get("selected_step"),
            "selection_score": checkpoint_selection_comparison["structured"].get("selection_score"),
            "dataset_grounded_metrics": grounded_structured_metrics,
            "ood_metrics": ood_structured_metrics,
        },
        "dataset_grounded": {
            "comparisons": grounded_comparisons,
        },
        "ood_compositional": {
            "comparisons": ood_comparisons,
        },
        "source_distribution_deltas": {
            "baseline": _source_delta_summary(baseline_report),
            "structured": _source_delta_summary(structured_report),
            "baseline_source_baselines": _source_distribution_summary(baseline_report),
            "structured_source_baselines": _source_distribution_summary(structured_report),
            "rare_color_comparison": rare_color_comparison,
        },
        "comparisons": ood_comparisons,
        "answers": answers,
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "challenger_conditioning_comparison.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "challenger_conditioning_comparison.md").write_text(
        _format_challenger_conditioning_comparison_markdown(report),
        encoding="utf-8",
    )
    return report


def _resolve_eval_checkpoint_steps(config: FullV4ChallengerAuditConfig) -> tuple[int, ...]:
    if not bool(config.eval_checkpoints):
        return ()
    max_steps = int(config.max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive for checkpoint evaluation")
    if config.eval_checkpoint_steps:
        steps = list(_parse_eval_checkpoint_steps(config.eval_checkpoint_steps))
    else:
        every = int(config.eval_checkpoint_every)
        if every <= 0:
            raise ValueError("--eval-checkpoint-every must be positive when --eval-checkpoints is enabled")
        steps = list(range(every, max_steps + 1, every))
    steps.append(max_steps)
    normalized = sorted({int(step) for step in steps})
    for step in normalized:
        if step <= 0:
            raise ValueError("checkpoint evaluation steps must be positive")
        if step > max_steps:
            raise ValueError(f"checkpoint evaluation step {step} exceeds max_steps={max_steps}")
    return tuple(normalized)


def _parse_eval_checkpoint_steps(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = [str(part).strip() for part in value]
    if not raw_values:
        raise ValueError("--eval-checkpoint-steps must contain at least one step")
    return tuple(int(part) for part in raw_values)


def _prepare_full_v4_ood_prompts(config: FullV4ChallengerAuditConfig) -> tuple[Path, dict[str, Any]]:
    ood_prompt_file = _resolve_full_v4_ood_prompt_file(config)
    if config.ood_prompts is not None and ood_prompt_file.is_file():
        ood_prompt_summary = _read_prompt_set_summary(ood_prompt_file, max_prompts=10_000)
    else:
        build_ood_compositional_prompts(OodCompositionalPromptConfig(out=ood_prompt_file))
        ood_prompt_summary = _read_prompt_set_summary(ood_prompt_file, max_prompts=10_000)
    return ood_prompt_file, ood_prompt_summary


def _checkpoint_eval_sample_count(
    config: FullV4ChallengerAuditConfig,
    prompt_summary: Mapping[str, Any],
) -> int:
    prompt_count = int(prompt_summary.get("prompt_count") or 0)
    if config.checkpoint_eval_max_samples is None:
        return prompt_count
    return max(0, min(prompt_count, int(config.checkpoint_eval_max_samples)))


def _run_full_v4_grounded_evaluation(
    config: FullV4ChallengerAuditConfig,
    *,
    checkpoint: Path,
    prompt_file: Path,
    prompt_summary: Mapping[str, Any],
    generated_dir: Path,
    seed: int,
    faithfulness_max_sources: int,
    max_samples: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_count = int(prompt_summary.get("prompt_count") or 0) if max_samples is None else int(max_samples)
    sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=checkpoint,
            prompts=prompt_file,
            out_dir=generated_dir,
            export_preset=config.export_preset,
            max_samples=sample_count,
            steps=int(config.sample_steps),
            cfg_scale=float(config.cfg_scale),
            max_colors=int(config.max_colors),
            alpha_threshold=float(config.alpha_threshold),
            device=config.device,
            seed=int(seed),
            batch_size=int(config.sample_batch_size),
            dither=False,
            write_raw_rgba=True,
            write_hard_rgba=True,
            contact_sheet_labels="prompt_and_seed",
            project_palette=bool(config.project_palette),
            project_palette_target_colors=int(config.project_palette_target_colors),
            project_palette_min_pixel_share=float(config.project_palette_min_pixel_share),
            project_palette_method=str(config.project_palette_method),
        )
    )
    qa = qa_generated_sprites(generated_dir).to_json_dict()
    review_result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=generated_dir,
            out=generated_dir / "generated_review_report.md",
            out_json=generated_dir / "generated_review_report.json",
            out_dir=generated_dir / "review",
            group_by="category",
            max_samples_per_sheet=sample_count,
            compare_raw_indexed=True,
        )
    )
    faithfulness_json = generated_dir / "prompt_faithfulness_report.json"
    faithfulness = run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=generated_dir,
            prompts=prompt_file,
            dataset=config.dataset,
            out=generated_dir / "prompt_faithfulness_report.md",
            out_json=faithfulness_json,
            max_sources=int(faithfulness_max_sources),
            source_selection="auto",
        )
    )
    _verify_faithfulness_matches_disk(faithfulness, faithfulness_json)
    section = {
        "prompt_set": {**dict(prompt_summary), "evaluated_prompt_count": sample_count},
        "generation": _full_v4_generation_summary(sample_report, generated_dir=generated_dir),
        "palette_projection": _full_v4_palette_projection_summary(generated_dir, sample_report),
        "generated_qa": _full_v4_qa_summary(qa),
        "generated_review": _full_v4_generated_review_summary(review_result.report, max_colors=int(config.max_colors)),
        "prompt_faithfulness": _full_v4_faithfulness_summary(faithfulness),
    }
    artifacts = {
        "grounded_prompts": str(prompt_file),
        "grounded_generated_dir": str(generated_dir),
        "grounded_generated_contact_sheet": None
        if sample_report.get("contact_sheet") is None
        else str(generated_dir / str(sample_report.get("contact_sheet"))),
        "grounded_review_contact_sheets": review_result.report.get("contact_sheets", {}),
        "grounded_palette_projection_report": str(generated_dir / "palette_projection_report.json")
        if (generated_dir / "palette_projection_report.json").is_file()
        else None,
        "grounded_palette_projection_contact_sheet": str(generated_dir / "contact_sheet_projected.png")
        if (generated_dir / "contact_sheet_projected.png").is_file()
        else None,
    }
    return section, artifacts


def _run_full_v4_ood_evaluation(
    config: FullV4ChallengerAuditConfig,
    *,
    checkpoint: Path,
    ood_prompt_file: Path,
    ood_prompt_summary: Mapping[str, Any],
    generated_dir: Path,
    seed: int,
    include_sensitivity: bool,
    faithfulness_max_sources: int,
    max_samples: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_count = int(ood_prompt_summary["prompt_count"]) if max_samples is None else int(max_samples)
    ood_sample_report = run_sample_generator_challenger(
        ChallengerSampleConfig(
            checkpoint=checkpoint,
            prompts=ood_prompt_file,
            out_dir=generated_dir,
            export_preset=config.export_preset,
            max_samples=sample_count,
            steps=int(config.sample_steps),
            cfg_scale=float(config.cfg_scale),
            max_colors=int(config.max_colors),
            alpha_threshold=float(config.alpha_threshold),
            device=config.device,
            seed=int(seed),
            batch_size=int(config.sample_batch_size),
            dither=False,
            write_raw_rgba=True,
            write_hard_rgba=True,
            contact_sheet_labels="prompt_and_seed",
            project_palette=bool(config.project_palette),
            project_palette_target_colors=int(config.project_palette_target_colors),
            project_palette_min_pixel_share=float(config.project_palette_min_pixel_share),
            project_palette_method=str(config.project_palette_method),
        )
    )
    ood_qa = qa_generated_sprites(generated_dir).to_json_dict()
    ood_review_result = review_generated_sprites(
        GeneratedReviewConfig(
            generated_dir=generated_dir,
            out=generated_dir / "generated_review_report.md",
            out_json=generated_dir / "generated_review_report.json",
            out_dir=generated_dir / "review",
            group_by="category",
            max_samples_per_sheet=sample_count,
            compare_raw_indexed=True,
        )
    )
    ood_faithfulness_json = generated_dir / "prompt_faithfulness_report.json"
    ood_faithfulness = run_prompt_faithfulness(
        PromptFaithfulnessConfig(
            generated=generated_dir,
            prompts=ood_prompt_file,
            dataset=config.dataset,
            out=generated_dir / "prompt_faithfulness_report.md",
            out_json=ood_faithfulness_json,
            max_sources=int(faithfulness_max_sources),
            source_selection="auto",
        )
    )
    _verify_faithfulness_matches_disk(ood_faithfulness, ood_faithfulness_json)
    section = {
        "prompt_set": {**dict(ood_prompt_summary), "evaluated_prompt_count": sample_count},
        "generation": _full_v4_generation_summary(ood_sample_report, generated_dir=generated_dir),
        "palette_projection": _full_v4_palette_projection_summary(generated_dir, ood_sample_report),
        "generated_qa": _full_v4_qa_summary(ood_qa),
        "generated_review": _full_v4_generated_review_summary(
            ood_review_result.report, max_colors=int(config.max_colors)
        ),
        "prompt_faithfulness": _full_v4_faithfulness_summary(ood_faithfulness),
    }
    if include_sensitivity:
        ood_sensitivity = run_prompt_sensitivity(
            PromptSensitivityConfig(
                checkpoint=checkpoint,
                prompts=ood_prompt_file,
                out_dir=generated_dir / "prompt_sensitivity",
                device=config.device,
                seed=int(seed),
                max_prompts=int(min(config.max_sensitivity_prompts, sample_count)),
                noise_samples=int(config.noise_samples),
                max_pairs=8,
                max_colors=int(config.max_colors),
                alpha_threshold=float(config.alpha_threshold),
                batch_size=int(config.sample_batch_size),
            )
        )
        section["prompt_sensitivity"] = _full_v4_sensitivity_summary(ood_sensitivity)
    artifacts = {
        "ood_prompts": str(ood_prompt_file),
        "ood_generated_dir": str(generated_dir),
        "ood_generated_contact_sheet": None
        if ood_sample_report.get("contact_sheet") is None
        else str(generated_dir / str(ood_sample_report.get("contact_sheet"))),
        "ood_review_contact_sheets": ood_review_result.report.get("contact_sheets", {}),
        "ood_palette_projection_report": str(generated_dir / "palette_projection_report.json")
        if (generated_dir / "palette_projection_report.json").is_file()
        else None,
        "ood_palette_projection_contact_sheet": str(generated_dir / "contact_sheet_projected.png")
        if (generated_dir / "contact_sheet_projected.png").is_file()
        else None,
    }
    return section, artifacts


def _evaluate_full_v4_checkpoint_candidates(
    config: FullV4ChallengerAuditConfig,
    *,
    train_dir: Path,
    checkpoint_steps: Sequence[int],
    grounded_prompt_file: Path,
    grounded_prompt_summary: Mapping[str, Any],
    ood_prompt_file: Path,
    ood_prompt_summary: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = _full_v4_checkpoint_candidates(
        train_dir,
        checkpoint_steps=checkpoint_steps,
        prefer_ema=bool(config.sample_ema),
    )
    leaderboard: list[dict[str, Any]] = []
    grounded_eval_samples = _checkpoint_eval_sample_count(config, grounded_prompt_summary)
    ood_eval_samples = _checkpoint_eval_sample_count(config, ood_prompt_summary)
    for candidate in candidates:
        grounded_dir = Path(config.out_dir) / "checkpoint_grounded_eval" / _checkpoint_eval_dir_name(candidate)
        grounded_section, _grounded_artifacts = _run_full_v4_grounded_evaluation(
            config,
            checkpoint=Path(str(candidate["checkpoint_path"])),
            prompt_file=grounded_prompt_file,
            prompt_summary=grounded_prompt_summary,
            generated_dir=grounded_dir,
            seed=int(config.seed),
            faithfulness_max_sources=0,
            max_samples=grounded_eval_samples,
        )
        generated_dir = Path(config.out_dir) / "checkpoint_ood_eval" / _checkpoint_eval_dir_name(candidate)
        ood_section, _artifacts = _run_full_v4_ood_evaluation(
            config,
            checkpoint=Path(str(candidate["checkpoint_path"])),
            ood_prompt_file=ood_prompt_file,
            ood_prompt_summary=ood_prompt_summary,
            generated_dir=generated_dir,
            seed=int(config.seed) + 17,
            include_sensitivity=False,
            faithfulness_max_sources=0,
            max_samples=ood_eval_samples,
        )
        leaderboard.append(_checkpoint_ood_leaderboard_entry(candidate, ood_section, grounded_section=grounded_section))
    if not leaderboard:
        raise ValueError("--eval-checkpoints produced no checkpoint candidates")
    selected = _select_ood_checkpoint_entry(leaderboard)
    selected_path = str(selected["checkpoint_path"])
    for entry in leaderboard:
        entry["selected"] = str(entry["checkpoint_path"]) == selected_path
    ranked = sorted(leaderboard, key=_checkpoint_selection_rank_key, reverse=True)
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
    final_step = int(config.max_steps)
    final_entry = next((entry for entry in leaderboard if int(entry.get("step") or -1) == final_step), None)
    return {
        "enabled": True,
        "selection_metric": OOD_SELECTION_METRIC,
        "selected_checkpoint": selected_path,
        "selected_step": int(selected["step"]),
        "selected_ema": bool(selected.get("ema")),
        "selected_score": selected.get("ood_score"),
        "selected_deployable": bool(selected.get("guardrails_passed")),
        "selected_guardrail_failures": list(selected.get("guardrail_failures") or []),
        "final_step": final_step,
        "final_step_checkpoint": None if final_entry is None else str(final_entry.get("checkpoint_path")),
        "final_step_metrics": None if final_entry is None else _checkpoint_metric_snapshot(final_entry),
        "selected_metrics": _checkpoint_metric_snapshot(selected),
        "leaderboard": ranked,
        "guardrails": {
            "grounded": {"qa_errors": 0, **dict(GROUNDED_SELECTION_GUARDRAILS)},
            "ood": {"qa_errors": 0, **dict(OOD_SELECTION_GUARDRAILS)},
        },
        "checkpoint_eval_max_samples": None
        if config.checkpoint_eval_max_samples is None
        else int(config.checkpoint_eval_max_samples),
        "grounded_eval_samples": int(grounded_eval_samples),
        "ood_eval_samples": int(ood_eval_samples),
    }


def _full_v4_checkpoint_candidates(
    train_dir: Path,
    *,
    checkpoint_steps: Sequence[int],
    prefer_ema: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    suffix = "_ema" if prefer_ema else ""
    for step in checkpoint_steps:
        path = Path(train_dir) / f"checkpoint_step_{int(step):06d}{suffix}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint evaluation candidate is missing: {path}")
        candidates.append(
            {
                "step": int(step),
                "checkpoint_path": str(path),
                "checkpoint": str(path),
                "ema": bool(prefer_ema),
            }
        )
    return candidates


def _checkpoint_eval_dir_name(candidate: Mapping[str, Any]) -> str:
    variant = "ema" if bool(candidate.get("ema")) else "raw"
    return f"step_{int(candidate.get('step') or 0):06d}_{variant}"


def _checkpoint_ood_leaderboard_entry(
    candidate: Mapping[str, Any],
    ood_section: Mapping[str, Any],
    *,
    grounded_section: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _checkpoint_ood_metrics(ood_section)
    grounded_metrics = _checkpoint_grounded_metrics(grounded_section or {})
    score = _ood_control_score_v1(metrics)
    grounded_failures = _grounded_guardrail_failures(grounded_metrics)
    ood_failures = _ood_guardrail_failures(metrics)
    failures = [*grounded_failures, *ood_failures]
    guardrails_passed = not failures
    return {
        **dict(candidate),
        "score": score,
        "ood_score": score,
        "grounded_qa_errors": int(grounded_metrics.get("qa_errors") or 0),
        "grounded_qa_error_rate": grounded_metrics.get("qa_error_rate"),
        "grounded_category": grounded_metrics.get("category"),
        "grounded_color": grounded_metrics.get("color"),
        "grounded_guardrails_passed": not grounded_failures,
        "grounded_guardrail_failures": grounded_failures,
        "ood_qa_errors": int(metrics.get("qa_errors") or 0),
        "ood_qa_error_rate": metrics.get("qa_error_rate"),
        "ood_category": metrics.get("category"),
        "ood_color": metrics.get("color"),
        "ood_rare_color_rate": metrics.get("rare_color_rate"),
        "ood_blob_collapse_rate": metrics.get("blob_collapse_rate"),
        "ood_guardrails_passed": not ood_failures,
        "ood_guardrail_failures": ood_failures,
        "qa_errors": int(metrics.get("qa_errors") or 0),
        "qa_error_rate": metrics.get("qa_error_rate"),
        "category": metrics.get("category"),
        "color": metrics.get("color"),
        "rare_color_rate": metrics.get("rare_color_rate"),
        "repeated_silhouette_rate": metrics.get("repeated_silhouette_rate"),
        "blob_rate": metrics.get("blob_collapse_rate"),
        "blob_collapse_rate": metrics.get("blob_collapse_rate"),
        "potion_rate": metrics.get("potion_collapse_rate"),
        "potion_collapse_rate": metrics.get("potion_collapse_rate"),
        "touches_border_rate": metrics.get("touches_border_rate"),
        "passes_guardrails": guardrails_passed,
        "guardrails_passed": guardrails_passed,
        "guardrail_failures": failures,
        "deployable": guardrails_passed,
        "selected": False,
    }


def _checkpoint_ood_metrics(ood_section: Mapping[str, Any]) -> dict[str, Any]:
    qa = _section(ood_section, "generated_qa")
    review = _section(ood_section, "generated_review")
    faithfulness = _section(ood_section, "prompt_faithfulness")
    prompts = _section(ood_section, "prompt_set")
    qa_errors = int(qa.get("errors") or 0)
    sample_count = _optional_int(qa.get("sample_count")) or _optional_int(prompts.get("prompt_count")) or 0
    return {
        "qa_errors": qa_errors,
        "qa_error_rate": float(qa_errors) / float(max(1, sample_count)),
        "category": _first_float(
            faithfulness.get("category_consistency_rate"),
            faithfulness.get("nearest_source_category_consistency_rate"),
        ),
        "color": _optional_float(faithfulness.get("color_consistency_rate")),
        "rare_color_rate": _optional_float(review.get("too_many_rare_colors_rate")),
        "repeated_silhouette_rate": _optional_float(faithfulness.get("repeated_silhouette_rate")),
        "blob_collapse_rate": _optional_float(faithfulness.get("generic_blob_collapse_rate")),
        "potion_collapse_rate": _optional_float(faithfulness.get("generic_potion_collapse_rate")),
        "touches_border_rate": _optional_float(review.get("touches_border_rate")),
    }


def _checkpoint_grounded_metrics(grounded_section: Mapping[str, Any]) -> dict[str, Any]:
    qa = _section(grounded_section, "generated_qa")
    faithfulness = _section(grounded_section, "prompt_faithfulness")
    prompts = _section(grounded_section, "prompt_set")
    qa_errors = int(qa.get("errors") or 0)
    sample_count = _optional_int(qa.get("sample_count")) or _optional_int(prompts.get("evaluated_prompt_count")) or 0
    return {
        "qa_errors": qa_errors,
        "qa_error_rate": float(qa_errors) / float(max(1, sample_count)),
        "category": _first_float(
            faithfulness.get("category_consistency_rate"),
            faithfulness.get("nearest_source_category_consistency_rate"),
        ),
        "color": _optional_float(faithfulness.get("color_consistency_rate")),
    }


def _ood_control_score_v1(metrics: Mapping[str, Any]) -> float:
    category = _metric_or_default(metrics, "category", "ood_category", default=0.0)
    color = _metric_or_default(metrics, "color", "ood_color", default=0.0)
    rare = _metric_or_default(metrics, "rare_color_rate", "ood_rare_color_rate", default=1.0)
    blob = _metric_or_default(metrics, "blob_collapse_rate", "ood_blob_rate", default=1.0)
    repeated = _metric_or_default(metrics, "repeated_silhouette_rate", "ood_repeated_silhouette_rate", default=1.0)
    potion = _metric_or_default(metrics, "potion_collapse_rate", "ood_potion_rate", default=1.0)
    qa_error_rate = _metric_or_default(metrics, "qa_error_rate", default=1.0)
    return 2.0 * category + 2.0 * color - 1.5 * rare - 1.5 * blob - 1.0 * repeated - 0.5 * potion - 1.0 * qa_error_rate


def _metric_or_default(metrics: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = _optional_float(metrics.get(key))
        if value is not None:
            return value
    return float(default)


def _ood_guardrail_failures(metrics: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if int(metrics.get("qa_errors") or 0) != 0:
        failures.append("qa_errors")
    category = _optional_float(metrics.get("category"))
    if category is None or category < OOD_SELECTION_GUARDRAILS["category_min"]:
        failures.append("ood_category")
    color = _optional_float(metrics.get("color"))
    if color is None or color < OOD_SELECTION_GUARDRAILS["color_min"]:
        failures.append("ood_color")
    rare = _optional_float(metrics.get("rare_color_rate"))
    if rare is None or rare > OOD_SELECTION_GUARDRAILS["rare_color_max"]:
        failures.append("ood_rare_color_rate")
    blob = _optional_float(metrics.get("blob_collapse_rate"))
    if blob is None or blob > OOD_SELECTION_GUARDRAILS["blob_max"]:
        failures.append("ood_blob_rate")
    return failures


def _grounded_guardrail_failures(metrics: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if int(metrics.get("qa_errors") or 0) != 0:
        failures.append("grounded_qa_errors")
    category = _optional_float(metrics.get("category"))
    if category is None or category < GROUNDED_SELECTION_GUARDRAILS["category_min"]:
        failures.append("grounded_category")
    color = _optional_float(metrics.get("color"))
    if color is None or color < GROUNDED_SELECTION_GUARDRAILS["color_min"]:
        failures.append("grounded_color")
    return failures


def _checkpoint_selection_rank_key(entry: Mapping[str, Any]) -> tuple[int, float, int]:
    passed = 1 if bool(entry.get("guardrails_passed")) else 0
    score = _optional_float(entry.get("ood_score"))
    step = int(entry.get("step") or 0)
    return passed, float("-inf") if score is None else score, -step


def _select_ood_checkpoint_entry(leaderboard: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(max(leaderboard, key=_checkpoint_selection_rank_key))


def _checkpoint_metric_snapshot(entry: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "step",
        "checkpoint_path",
        "ema",
        "score",
        "ood_score",
        "grounded_qa_errors",
        "grounded_qa_error_rate",
        "grounded_category",
        "grounded_color",
        "grounded_guardrails_passed",
        "grounded_guardrail_failures",
        "ood_qa_errors",
        "ood_qa_error_rate",
        "ood_category",
        "ood_color",
        "ood_rare_color_rate",
        "ood_blob_collapse_rate",
        "ood_guardrails_passed",
        "ood_guardrail_failures",
        "qa_errors",
        "qa_error_rate",
        "category",
        "color",
        "rare_color_rate",
        "repeated_silhouette_rate",
        "blob_rate",
        "blob_collapse_rate",
        "potion_rate",
        "potion_collapse_rate",
        "touches_border_rate",
        "passes_guardrails",
        "guardrails_passed",
        "guardrail_failures",
        "deployable",
    )
    return {key: entry.get(key) for key in keys}


def _eval_prompt_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    sprite_id = str(row.get("sprite_id") or "").strip()
    prompt = str(row.get("caption") or row.get("prompt") or row.get("object_name") or sprite_id)
    return {
        **dict(row),
        "prompt": prompt,
        "prompt_id": sprite_id or f"eval_prompt_{index:04d}",
        "target_sprite_id": sprite_id,
        "source_sprite_id": sprite_id,
        "eval_prompt_index": int(index),
    }


def _read_prompt_set_summary(path: Path, *, max_prompts: int) -> dict[str, Any]:
    rows = read_prompt_records(path, max_records=max_prompts)
    return _prompt_set_summary(rows, prompt_file=path, reused_existing=True)


def _prompt_set_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_file: Path,
    reused_existing: bool,
) -> dict[str, Any]:
    target_ids = [
        str(
            row.get("target_sprite_id")
            or row.get("source_sprite_id")
            or row.get("sprite_id")
            or row.get("prompt_id")
            or ""
        )
        for row in rows
    ]
    categories = Counter(_category(row) for row in rows)
    return {
        "prompt_file": str(prompt_file),
        "prompt_count": len(rows),
        "category_counts": dict(sorted(categories.items())),
        "unique_target_sprite_count": len({sprite_id for sprite_id in target_ids if sprite_id}),
        "target_ids_unique": len([sprite_id for sprite_id in target_ids if sprite_id])
        == len({sprite_id for sprite_id in target_ids if sprite_id}),
        "reused_existing_prompts": bool(reused_existing),
        **structured_prompt_summary(rows),
    }


def _full_v4_palette_swap_summary(
    train_report: Mapping[str, Any],
    config: FullV4ChallengerAuditConfig,
) -> dict[str, Any]:
    reported = train_report.get("palette_swap") if isinstance(train_report.get("palette_swap"), Mapping) else {}
    summary = {
        "palette_swap_augmentation": bool(config.palette_swap_augmentation),
        "palette_swap_prob": float(config.palette_swap_prob),
        "palette_swap_families": str(config.palette_swap_families),
        "palette_swap_stochastic": bool(config.palette_swap_stochastic),
        "palette_swap_keep_original_prob": float(config.palette_swap_keep_original_prob),
        "palette_swap_target_families": config.palette_swap_target_families,
        "palette_swap_source_families": config.palette_swap_source_families,
        "palette_swap_category_filter": config.palette_swap_category_filter,
        "palette_swap_min_color_confidence": float(config.palette_swap_min_color_confidence),
        "palette_swap_require_role_map": bool(config.palette_swap_require_role_map),
        "palette_swap_require_explicit_color": bool(config.palette_swap_require_explicit_color),
        "palette_swap_require_explicit_caption_color": bool(config.palette_swap_require_explicit_caption_color),
        "palette_swap_require_explicit_semantic_color": bool(config.palette_swap_require_explicit_semantic_color),
        "palette_swap_allow_colorless_caption_if_semantic_color": bool(
            config.palette_swap_allow_colorless_caption_if_semantic_color
        ),
        "palette_swap_no_caption_prepend": bool(config.palette_swap_no_caption_prepend),
        "palette_swap_allow_material_colors": bool(config.palette_swap_allow_material_colors),
        "palette_swap_preserve_outline": bool(config.palette_swap_preserve_outline),
        "palette_swap_update_prompts": bool(config.palette_swap_update_prompts),
    }
    for key in (
        "applied_count",
        "swapped_count",
        "kept_original_count",
        "effective_eligible_count_before_keep_original",
        "unchanged_ineligible_count",
        "unchanged_not_triggered_count",
        "applied_rate_total",
        "applied_rate_eligible",
        "effective_swapped_rate_total",
        "effective_kept_original_rate_total",
        "effective_eligible_rate_total_before_keep_original",
        "effective_swapped_rate_eligible",
        "effective_kept_original_rate_eligible",
        "eligible_count",
        "ineligible_count",
        "sample_count",
        "ineligibility_reason_counts",
        "target_family_counts",
        "source_to_target_matrix",
        "fallback_heuristic_rate",
        "material_conflict_drop_count",
        "colorless_caption_structured_only_count",
        "applied_rate",
    ):
        if key in reported:
            summary[key] = reported[key]
    return summary


def _full_v4_training_summary(train_report: Mapping[str, Any]) -> dict[str, Any]:
    final_loss = _optional_float(train_report.get("final_train_loss"))
    val_loss = _optional_float(train_report.get("val_loss"))
    return {
        "train_records": _optional_int(train_report.get("train_records")),
        "val_records": _optional_int(train_report.get("val_records")),
        "conditioning_mode": train_report.get("conditioning_mode"),
        "cfg_dropout": _optional_float(train_report.get("cfg_dropout")),
        "structured_field_dropout": _optional_float(train_report.get("structured_field_dropout")),
        "structured_fields_enabled": bool(train_report.get("structured_fields_enabled")),
        "structured_vocab_sizes": train_report.get("structured_vocab_sizes"),
        "ema_enabled": bool(train_report.get("ema_enabled")),
        "ema_decay": _optional_float(train_report.get("ema_decay")),
        "foreground_rgb_loss_weight": _optional_float(train_report.get("foreground_rgb_loss_weight")),
        "background_rgb_loss_weight": _optional_float(train_report.get("background_rgb_loss_weight")),
        "palette_loss_weight": _optional_float(train_report.get("palette_loss_weight")),
        "palette_loss_temperature": _optional_float(train_report.get("palette_loss_temperature")),
        "initial_train_loss": _optional_float(train_report.get("initial_train_loss")),
        "final_train_loss": final_loss,
        "final_train_loss_components": train_report.get("final_train_loss_components", {}),
        "last_step_loss": _optional_float(train_report.get("last_step_loss")),
        "last_step_loss_components": train_report.get("last_step_loss_components", {}),
        "val_loss": val_loss,
        "val_loss_components": train_report.get("val_loss_components", {}),
        "val_train_loss_gap": None if val_loss is None or final_loss is None else val_loss - final_loss,
        "val_train_loss_ratio": None
        if val_loss is None or final_loss is None or final_loss <= 0.0
        else val_loss / final_loss,
        "steps": _optional_int(
            train_report.get("steps_completed")
            if train_report.get("steps_completed") is not None
            else train_report.get("max_steps")
        ),
        "batch_size": _optional_int(train_report.get("batch_size")),
        "model_config": train_report.get("model_config", {}),
        "elapsed_seconds": _optional_float(train_report.get("elapsed_seconds")),
    }


def _full_v4_generation_summary(sample_report: Mapping[str, Any], *, generated_dir: Path) -> dict[str, Any]:
    contact_sheet = sample_report.get("contact_sheet")
    return {
        "sample_count": _optional_int(sample_report.get("sample_count")),
        "warnings": _optional_int(sample_report.get("warnings")),
        "fully_transparent_count": _optional_int(sample_report.get("fully_transparent_count")),
        "max_visible_color_count": _optional_int(sample_report.get("max_visible_color_count")),
        "contact_sheet": None if contact_sheet is None else str(generated_dir / str(contact_sheet)),
    }


def _full_v4_palette_projection_summary(generated_dir: Path, sample_report: Mapping[str, Any]) -> dict[str, Any]:
    report_path = Path(generated_dir) / "palette_projection_report.json"
    projection = (
        sample_report.get("palette_projection") if isinstance(sample_report.get("palette_projection"), Mapping) else {}
    )
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                report = loaded
        except json.JSONDecodeError:
            report = {}
    applied = bool(report) or bool(projection.get("applied"))
    contact_sheet = Path(generated_dir) / "contact_sheet_projected.png"
    summary = {
        "applied": bool(applied),
        "method": report.get("method") or projection.get("method"),
        "target_colors": _optional_int(report.get("target_colors") or projection.get("target_colors")),
        "min_pixel_share": _optional_float(report.get("min_pixel_share") or projection.get("min_pixel_share")),
        "median_visible_color_count_before": _optional_float(report.get("median_visible_color_count_before")),
        "median_visible_color_count_after": _optional_float(report.get("median_visible_color_count_after")),
        "mean_visible_color_count_before": _optional_float(report.get("mean_visible_color_count_before")),
        "mean_visible_color_count_after": _optional_float(report.get("mean_visible_color_count_after")),
        "rare_color_rate_before": _optional_float(report.get("rare_color_rate_before")),
        "rare_color_rate_after": _optional_float(report.get("rare_color_rate_after")),
        "mean_rgb_mae_visible": _optional_float(
            report.get("mean_rgb_mae_visible") or projection.get("mean_rgb_mae_visible")
        ),
        "destructive_rate": _optional_float(report.get("destructive_rate") or projection.get("destructive_rate")),
        "safe_count": _optional_int(report.get("safe_count")),
        "moderate_count": _optional_int(report.get("moderate_count")),
        "destructive_count": _optional_int(report.get("destructive_count")),
        "report": str(report_path) if report_path.is_file() else projection.get("report"),
        "contact_sheet": str(contact_sheet) if contact_sheet.is_file() else projection.get("contact_sheet"),
    }
    if report.get("sample_count") is not None:
        summary["sample_count"] = _optional_int(report.get("sample_count"))
    return summary


def _full_v4_qa_summary(qa: Mapping[str, Any]) -> dict[str, Any]:
    warnings = [str(item) for item in qa.get("warnings") or []]
    return {
        "sample_count": _optional_int(qa.get("sample_count")),
        "errors": len(qa.get("errors") or []),
        "warnings": len(warnings),
        "fully_transparent_count": sum(1 for warning in warnings if "fully transparent" in warning),
        "ok": bool(qa.get("ok")),
        "checks": qa.get("checks", {}),
    }


def _full_v4_generated_review_summary(review: Mapping[str, Any], *, max_colors: int | None = None) -> dict[str, Any]:
    overall = review.get("overall") if isinstance(review.get("overall"), Mapping) else {}
    warning_counts = overall.get("warning_counts") if isinstance(overall.get("warning_counts"), Mapping) else {}
    samples = review.get("samples") if isinstance(review.get("samples"), list) else []
    sample_metrics = [
        sample.get("metrics")
        for sample in samples
        if isinstance(sample, Mapping) and isinstance(sample.get("metrics"), Mapping)
    ]
    sample_warnings = [
        str(warning) for sample in samples if isinstance(sample, Mapping) for warning in (sample.get("warnings") or [])
    ]
    if not warning_counts and sample_warnings:
        warning_counts = Counter(sample_warnings)
    sample_count = max(1, int(review.get("sample_count") or overall.get("count") or len(samples) or 0))
    touches_border_rate = _review_rate(
        direct=review.get("touches_border_rate") or overall.get("touches_border_rate"),
        warning_counts=warning_counts,
        warning_name="touches_border",
        sample_count=sample_count,
        metrics=sample_metrics,
        metric_name="touches_border",
    )
    too_many_rare_colors_rate = _review_rate(
        direct=review.get("too_many_rare_colors_rate") or overall.get("too_many_rare_colors_rate"),
        warning_counts=warning_counts,
        warning_name="too_many_rare_colors",
        sample_count=sample_count,
        metrics=sample_metrics,
        metric_name=None,
    )
    return {
        "sample_count": _optional_int(review.get("sample_count")),
        "mean_alpha_coverage": _first_float(
            overall.get("mean_alpha_coverage"), _mean_metric(sample_metrics, "alpha_coverage")
        ),
        "mean_bbox_width": _first_float(overall.get("mean_bbox_width"), _mean_metric(sample_metrics, "bbox_width")),
        "mean_bbox_height": _first_float(overall.get("mean_bbox_height"), _mean_metric(sample_metrics, "bbox_height")),
        "mean_center_offset": _first_float(
            overall.get("mean_center_offset"),
            overall.get("mean_center_offset_from_image_center"),
            _mean_metric(sample_metrics, "center_offset_from_image_center"),
        ),
        "mean_visible_color_count": _first_float(
            overall.get("mean_visible_color_count"),
            review.get("mean_visible_color_count"),
            _group_weighted_metric(review.get("groups"), "mean_visible_color_count"),
            _mean_metric(sample_metrics, "visible_color_count"),
        ),
        "median_visible_color_count": _first_float(
            overall.get("median_visible_color_count"), _median_metric(sample_metrics, "visible_color_count")
        ),
        "warning_counts": dict(sorted((str(key), int(value)) for key, value in warning_counts.items())),
        "touches_border_rate": touches_border_rate,
        "too_many_rare_colors_rate": too_many_rare_colors_rate,
        "quantization_destructive_warnings": int(warning_counts.get("quantization_destructive", 0)),
        "mean_raw_indexed_rgb_mae_visible": _optional_float(overall.get("mean_raw_indexed_rgb_mae_visible")),
        "max_colors": _optional_int(max_colors),
        "median_visible_colors_pinned": bool(
            max_colors is not None
            and _first_float(
                overall.get("median_visible_color_count"), _median_metric(sample_metrics, "visible_color_count")
            )
            == float(max_colors)
        ),
        "groups": review.get("groups", {}),
        "contact_sheets": review.get("contact_sheets", {}),
    }


_FAITHFULNESS_CONSISTENCY_KEYS: tuple[str, ...] = (
    "sample_count",
    "category_consistency_rate",
    "nearest_source_category_consistency_rate",
    "color_consistency_rate",
    "repeated_silhouette_rate",
    "generic_blob_collapse_rate",
    "generic_potion_collapse_rate",
    "nearest_neighbor_duplicate_rate",
)


def _verify_faithfulness_matches_disk(faithfulness: Mapping[str, Any], out_json: Path) -> None:
    """Assert the freshly-computed metrics match the JSON written to disk.

    Guards against stale/partial report reuse: the full-v4 audit embeds exactly the
    numbers on disk for the current run, or fails loudly.
    """

    if not Path(out_json).is_file():
        raise FileNotFoundError(f"prompt faithfulness report was not written: {out_json}")
    on_disk = json.loads(Path(out_json).read_text(encoding="utf-8"))
    for key in _FAITHFULNESS_CONSISTENCY_KEYS:
        embedded = faithfulness.get(key)
        written = on_disk.get(key)
        if embedded != written:
            raise ValueError(
                f"prompt faithfulness mismatch for {key!r}: in-memory {embedded!r} != on-disk {written!r} ({out_json})"
            )


def _full_v4_faithfulness_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    nearest = report.get("nearest_source_summary") if isinstance(report.get("nearest_source_summary"), Mapping) else {}
    source_selection = report.get("source_selection") if isinstance(report.get("source_selection"), Mapping) else {}
    return {
        "sample_count": _optional_int(report.get("sample_count")),
        "source_selection": dict(source_selection),
        "mean_nearest_source_distance": _first_float(
            nearest.get("mean_distance"),
            nearest.get("mean_dist"),
            report.get("mean_nearest_source_distance"),
            report.get("mean_dist"),
        ),
        "nearest_source_category_consistency_rate": _optional_float(
            report.get("nearest_source_category_consistency_rate", report.get("category_consistency_rate"))
        ),
        "category_consistency_rate": _optional_float(report.get("category_consistency_rate")),
        "color_consistency_rate": _optional_float(report.get("color_consistency_rate")),
        "shape_bbox_consistency_rate": _optional_float(report.get("shape_bbox_consistency_rate")),
        "repeated_silhouette_rate": _optional_float(report.get("repeated_silhouette_rate")),
        "nearest_neighbor_duplicate_rate": _optional_float(report.get("nearest_neighbor_duplicate_rate")),
        "generic_potion_collapse_rate": _optional_float(report.get("generic_potion_collapse_rate")),
        "generic_flame_collapse_rate": _optional_float(report.get("generic_flame_collapse_rate")),
        "generic_blob_collapse_rate": _optional_float(report.get("generic_blob_collapse_rate")),
        "worst_object_families": report.get("object_families_worst_faithfulness", [])[:10],
        "failed_color_prompts": report.get("color_prompts_failed", [])[:20],
    }


def _full_v4_sensitivity_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    sets = report.get("sets") if isinstance(report.get("sets"), Mapping) else {}
    same_noise = _nested_metrics(sets, "same_noise_different_prompts")
    same_prompt = _nested_metrics(sets, "same_prompt_different_noise")
    prompt_pairs = _nested_metrics(sets, "prompt_pairs")
    pairs = prompt_pairs.get("pairs") if isinstance(prompt_pairs.get("pairs"), list) else []
    synthetic_failures = [
        pair
        for pair in pairs
        if isinstance(pair, Mapping)
        and str(pair.get("source")) == "synthetic_control"
        and (pair.get("warnings") or pair.get("metrics", {}).get("near_duplicate"))
    ]
    return {
        "same_noise_mean_difference": _optional_float(same_noise.get("mean_pairwise_difference")),
        "same_noise_near_duplicate_rate": _optional_float(same_noise.get("near_duplicate_rate")),
        "same_prompt_diversity": _optional_float(same_prompt.get("diversity_score")),
        "prompt_pair_near_duplicate_rate": _optional_float(prompt_pairs.get("near_duplicate_rate")),
        "warnings": list(report.get("warnings") or []),
        "synthetic_control_failures": synthetic_failures,
    }


def _nested_metrics(sets: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = sets.get(key)
    if not isinstance(item, Mapping):
        return {}
    metrics = item.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _optional_float(value)
        if numeric is not None:
            return numeric
    return None


def _review_rate(
    *,
    direct: Any,
    warning_counts: Mapping[str, Any],
    warning_name: str,
    sample_count: int,
    metrics: Sequence[Mapping[str, Any]],
    metric_name: str | None,
) -> float | None:
    direct_value = _optional_float(direct)
    if direct_value is not None:
        return direct_value
    if warning_name in warning_counts:
        return float(int(warning_counts.get(warning_name) or 0) / float(max(1, sample_count)))
    if metric_name:
        metric_values = [bool(item.get(metric_name)) for item in metrics if metric_name in item]
        if metric_values:
            return float(sum(1 for value in metric_values if value) / float(len(metric_values)))
    return None


def _group_weighted_metric(groups: Any, key: str) -> float | None:
    if not isinstance(groups, Mapping):
        return None
    numerator = 0.0
    denominator = 0
    for summary in groups.values():
        if not isinstance(summary, Mapping):
            continue
        value = _optional_float(summary.get(key))
        count = _optional_int(summary.get("count"))
        if value is None or count is None or count <= 0:
            continue
        numerator += value * count
        denominator += count
    return None if denominator <= 0 else numerator / float(denominator)


def _mean_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _numeric_values(metrics, key)
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _median_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = sorted(_numeric_values(metrics, key))
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return float(values[midpoint])
    return float((values[midpoint - 1] + values[midpoint]) / 2.0)


def _numeric_values(metrics: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in metrics:
        value = item.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _source_distribution_report(
    *,
    dataset: Path,
    training_manifest: Path,
    grounded_category_counts: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        samples = _load_source_framing_samples(Path(dataset), Path(training_manifest))
    except Exception as exc:
        empty = _empty_source_baseline(error=str(exc))
        return {
            "full_source": empty,
            "by_category": {},
            "grounded_eval_category_weighted": empty,
            "errors": [str(exc)],
        }
    by_category_samples: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_category_samples[str(sample.get("category") or "unknown")].append(sample)
    by_category = {
        category: _source_baseline_from_samples(category_samples)
        for category, category_samples in sorted(by_category_samples.items())
    }
    return {
        "full_source": _source_baseline_from_samples(samples),
        "by_category": by_category,
        "grounded_eval_category_weighted": _category_weighted_source_baseline(by_category, grounded_category_counts),
        "errors": [],
    }


def _load_source_framing_samples(dataset: Path, training_manifest: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(training_manifest)
    npz_cache: dict[str, dict[str, Any]] = {}
    seen_sprite_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    for record in rows:
        sprite_id = str(record.get("sprite_id") or "").strip()
        if not sprite_id or sprite_id in seen_sprite_ids:
            continue
        npz_file = str(record.get("npz_file") or f"{record.get('split', '')}.npz")
        npz_row = int(record.get("npz_row", -1))
        arrays = _load_npz_arrays(dataset / npz_file, npz_cache)
        if npz_row < 0 or npz_row >= int(np.asarray(arrays["alpha"]).shape[0]):
            continue
        rgba = npz_row_to_rgba(
            index_map=np.asarray(arrays["index_map"][npz_row]),
            alpha=np.asarray(arrays["alpha"][npz_row]),
            palette=np.asarray(arrays["palette"][npz_row]),
            palette_mask=np.asarray(arrays["palette_mask"][npz_row], dtype=bool),
        )
        samples.append(
            {
                "sprite_id": sprite_id,
                "category": _category(record),
                "metrics": compute_sprite_framing_metrics(rgba),
            }
        )
        seen_sprite_ids.add(sprite_id)
    return samples


def _load_npz_arrays(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    cache[key] = arrays
    return arrays


def _source_baseline_from_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        sample.get("metrics")
        for sample in samples
        if isinstance(sample, Mapping) and isinstance(sample.get("metrics"), Mapping)
    ]
    categories = Counter(str(sample.get("category") or "unknown") for sample in samples if isinstance(sample, Mapping))
    return {
        "count": len(samples),
        "category_counts": dict(sorted(categories.items())),
        "metrics": _source_metric_summary(metrics),
    }


def _empty_source_baseline(*, error: str | None = None) -> dict[str, Any]:
    payload = {
        "count": 0,
        "category_counts": {},
        "metrics": _source_metric_summary([]),
    }
    if error:
        payload["error"] = error
    return payload


def _source_metric_summary(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "border_touch_rate": _border_touch_rate(metrics),
        "mean_alpha_coverage": _mean_metric(metrics, "alpha_coverage"),
        "mean_bbox_width": _mean_metric(metrics, "bbox_width"),
        "mean_bbox_height": _mean_metric(metrics, "bbox_height"),
        "mean_center_offset": _mean_metric(metrics, "center_offset_from_image_center"),
        "mean_visible_color_count": _mean_metric(metrics, "visible_color_count"),
    }


def _border_touch_rate(metrics: Sequence[Mapping[str, Any]]) -> float | None:
    values = [bool(metric.get("touches_border")) for metric in metrics if isinstance(metric, Mapping)]
    if not values:
        return None
    return float(sum(1 for value in values if value) / float(len(values)))


def _category_weighted_source_baseline(
    by_category: Mapping[str, Any],
    category_counts: Mapping[str, Any],
) -> dict[str, Any]:
    counts = {str(key): int(value) for key, value in category_counts.items() if int(value) > 0}
    total_weight = sum(counts.values())
    metrics: dict[str, float | None] = {}
    missing: list[str] = []
    metric_keys = (
        "border_touch_rate",
        "mean_alpha_coverage",
        "mean_bbox_width",
        "mean_bbox_height",
        "mean_center_offset",
        "mean_visible_color_count",
    )
    for key in metric_keys:
        numerator = 0.0
        denominator = 0
        for category, count in counts.items():
            source_category = by_category.get(category)
            if not isinstance(source_category, Mapping):
                if category not in missing:
                    missing.append(category)
                continue
            source_metrics = (
                source_category.get("metrics") if isinstance(source_category.get("metrics"), Mapping) else {}
            )
            value = _optional_float(source_metrics.get(key))
            if value is None:
                if category not in missing:
                    missing.append(category)
                continue
            numerator += value * count
            denominator += count
        metrics[key] = None if denominator <= 0 else numerator / float(denominator)
    return {
        "prompt_count": int(total_weight),
        "category_counts": dict(sorted(counts.items())),
        "missing_categories": sorted(missing),
        "metrics": metrics,
    }


def _generated_vs_source_delta(
    generated_review: Mapping[str, Any],
    source_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    source_metrics = source_baseline.get("metrics") if isinstance(source_baseline.get("metrics"), Mapping) else {}
    generated_metrics = {
        "border_touch_rate": _optional_float(generated_review.get("touches_border_rate")),
        "mean_alpha_coverage": _optional_float(generated_review.get("mean_alpha_coverage")),
        "mean_bbox_width": _optional_float(generated_review.get("mean_bbox_width")),
        "mean_bbox_height": _optional_float(generated_review.get("mean_bbox_height")),
        "mean_center_offset": _optional_float(generated_review.get("mean_center_offset")),
        "mean_visible_color_count": _optional_float(generated_review.get("mean_visible_color_count")),
    }
    deltas = {
        key: None
        if generated_metrics.get(key) is None or _optional_float(source_metrics.get(key)) is None
        else float(generated_metrics[key]) - float(source_metrics[key])
        for key in generated_metrics
    }
    return {
        "generated": generated_metrics,
        "source": {key: _optional_float(source_metrics.get(key)) for key in generated_metrics},
        "deltas": deltas,
    }


def _resolve_full_v4_prompt_file(config: FullV4ChallengerAuditConfig) -> Path:
    if config.eval_prompts is not None:
        return Path(config.eval_prompts)
    return Path(config.out_dir) / "prompts" / "balanced_eval_prompts.jsonl"


def _resolve_full_v4_ood_prompt_file(config: FullV4ChallengerAuditConfig) -> Path:
    if config.ood_prompts is not None:
        return Path(config.ood_prompts)
    return Path(config.out_dir) / "prompts" / "ood_compositional_prompts.jsonl"


def _full_v4_train_dir_name(max_steps: int) -> str:
    steps = int(max_steps)
    if steps >= 1000 and steps % 1000 == 0:
        return f"train_{steps // 1000}k"
    return f"train_{steps}"


def _write_full_v4_audit_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full_v4_challenger_audit.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "full_v4_challenger_audit.md").write_text(_format_full_v4_markdown(report), encoding="utf-8")


def _format_full_v4_markdown(report: Mapping[str, Any]) -> str:
    training = _section(report, "training")
    prompts = _section(report, "prompt_set")
    generation = _section(report, "generation")
    palette_projection = _section(report, "palette_projection")
    qa = _section(report, "generated_qa")
    review = _section(report, "generated_review")
    faithfulness = _section(report, "prompt_faithfulness")
    sensitivity = _section(report, "prompt_sensitivity")
    decision = _section(report, "decision")
    artifacts = _section(report, "artifacts")
    ood = _section(report, "ood_compositional")
    checkpoint_evaluation = _section(report, "checkpoint_evaluation")
    source_distribution = _section(report, "source_distribution")
    generated_vs_source = _section(report, "generated_vs_source")
    lines = [
        "# Full-v4 Challenger Audit",
        "",
        f"Dataset: `{report.get('dataset', '')}`",
        f"Seed: `{report.get('seed', '')}`",
        f"Conditioning mode: `{report.get('conditioning_mode') or training.get('conditioning_mode') or ''}`",
        f"CFG scale: `{_section(report, 'config').get('cfg_scale', '')}`",
        f"Sampled EMA checkpoint: {bool(artifacts.get('sampled_ema'))}",
        f"Sample checkpoint: `{artifacts.get('sample_checkpoint', '')}`",
        f"Decision: **{decision.get('code', 'F')}. {decision.get('label', 'Unknown.')}**",
        "",
        "Current v1 default: Phase 1 EMA, CFG 3.0, k16 deterministic palette projection.",
        "Palette-swap branches are experimental and not adopted.",
        "",
        "## Training",
        "",
        f"- Train records: {_fmt_int(training.get('train_records'))}",
        f"- Val records: {_fmt_int(training.get('val_records'))}",
        f"- Steps: {_fmt_int(training.get('steps'))}",
        f"- Batch size: {_fmt_int(training.get('batch_size'))}",
        f"- Initial train loss: {_fmt(training.get('initial_train_loss'))}",
        f"- Final train loss: {_fmt(training.get('final_train_loss'))}",
        f"- Last step loss: {_fmt(training.get('last_step_loss'))}",
        f"- Val loss: {_fmt(training.get('val_loss'))}",
        f"- Val/train loss ratio: {_fmt(training.get('val_train_loss_ratio'))}",
        f"- Val/train loss gap: {_fmt(training.get('val_train_loss_gap'))}",
        f"- CFG dropout: {_fmt(training.get('cfg_dropout'))}",
        f"- Structured field dropout: {_fmt(training.get('structured_field_dropout'))}",
        f"- Structured fields enabled: {bool(training.get('structured_fields_enabled'))}",
        f"- EMA enabled: {bool(training.get('ema_enabled'))}",
        f"- EMA decay: {_fmt(training.get('ema_decay'))}",
        f"- Foreground RGB loss weight: {_fmt(training.get('foreground_rgb_loss_weight'))}",
        f"- Background RGB loss weight: {_fmt(training.get('background_rgb_loss_weight'))}",
        f"- Palette loss weight: {_fmt(training.get('palette_loss_weight'))}",
        f"- Palette loss temperature: {_fmt(training.get('palette_loss_temperature'))}",
        f"- Elapsed seconds: {_fmt(training.get('elapsed_seconds'))}",
        f"- Structured vocab sizes: `{json.dumps(_jsonable(training.get('structured_vocab_sizes')), sort_keys=True)}`",
        f"- Model config: `{json.dumps(_jsonable(training.get('model_config', {})), sort_keys=True)}`",
        "",
        "## Prompt Set",
        "",
        f"- Prompt file: `{prompts.get('prompt_file', '')}`",
        f"- Eval prompt count: {_fmt_int(prompts.get('prompt_count'))}",
        f"- Unique target sprite count: {_fmt_int(prompts.get('unique_target_sprite_count'))}",
        f"- Prompt target IDs unique: {bool(prompts.get('target_ids_unique'))}",
        f"- Category counts: `{json.dumps(_jsonable(prompts.get('category_counts', {})), sort_keys=True)}`",
        f"- Structured fields present: {bool(prompts.get('structured_fields_present'))}",
        f"- Structured present count: {_fmt_int(prompts.get('structured_present_count'))}",
        f"- Structured field counts: `{json.dumps(_jsonable(prompts.get('structured_field_counts', {})), sort_keys=True)}`",
        "",
        "## Generation",
        "",
        f"- Sample count: {_fmt_int(generation.get('sample_count'))}",
        f"- Max visible colors: {_fmt_int(generation.get('max_visible_color_count'))}",
        f"- Fully transparent count: {_fmt_int(generation.get('fully_transparent_count'))}",
        f"- Contact sheet: `{generation.get('contact_sheet') or ''}`",
        "",
        "## Palette Projection",
        "",
        *_format_palette_projection_summary_lines(palette_projection),
        "",
        "## Generated QA",
        "",
        f"- Sample count: {_fmt_int(qa.get('sample_count'))}",
        f"- Errors: {_fmt_int(qa.get('errors'))}",
        f"- Warnings: {_fmt_int(qa.get('warnings'))}",
        f"- Fully transparent count: {_fmt_int(qa.get('fully_transparent_count'))}",
        "",
        "## Generated Review",
        "",
        f"- Mean alpha coverage: {_fmt(review.get('mean_alpha_coverage'))}",
        f"- Median visible colors: {_fmt(review.get('median_visible_color_count'))}",
        f"- Warning counts: `{json.dumps(_jsonable(review.get('warning_counts', {})), sort_keys=True)}`",
        f"- Touches-border rate: {_fmt(review.get('touches_border_rate'))}",
        f"- Too-many-rare-colors rate: {_fmt(review.get('too_many_rare_colors_rate'))}",
        f"- Quantization destructive warnings: {_fmt_int(review.get('quantization_destructive_warnings'))}",
        f"- Mean raw/indexed RGB MAE visible: {_fmt(review.get('mean_raw_indexed_rgb_mae_visible'))}",
        f"- Group summary by category: `{json.dumps(_jsonable(review.get('groups', {})), sort_keys=True)}`",
        f"- Review contact sheets: `{json.dumps(_jsonable(review.get('contact_sheets', {})), sort_keys=True)}`",
        "",
        "## Source Baselines",
        "",
        *_format_source_baseline_lines(source_distribution, generated_vs_source, "grounded_eval"),
        "",
        "## Prompt Faithfulness",
        "",
        f"- Source selection: `{json.dumps(_jsonable(faithfulness.get('source_selection', {})), sort_keys=True)}`",
        f"- Mean nearest-source distance: {_fmt(faithfulness.get('mean_nearest_source_distance'))}",
        f"- Nearest-source category consistency: {_fmt(faithfulness.get('nearest_source_category_consistency_rate', faithfulness.get('category_consistency_rate')))}",
        f"- Color consistency heuristic: {_fmt(faithfulness.get('color_consistency_rate'))}",
        f"- Shape/bbox consistency heuristic: {_fmt(faithfulness.get('shape_bbox_consistency_rate'))}",
        f"- Repeated silhouette rate: {_fmt(faithfulness.get('repeated_silhouette_rate'))}",
        f"- Nearest-neighbor duplicate rate: {_fmt(faithfulness.get('nearest_neighbor_duplicate_rate'))}",
        f"- Generic potion collapse rate: {_fmt(faithfulness.get('generic_potion_collapse_rate'))}",
        f"- Generic flame collapse rate: {_fmt(faithfulness.get('generic_flame_collapse_rate'))}",
        f"- Generic blob collapse rate: {_fmt(faithfulness.get('generic_blob_collapse_rate'))}",
        f"- Worst object families: `{json.dumps(_jsonable(faithfulness.get('worst_object_families', [])), sort_keys=True)}`",
        f"- Failed color prompts: `{json.dumps(_jsonable(faithfulness.get('failed_color_prompts', [])), sort_keys=True)}`",
        "",
        "## Prompt Sensitivity",
        "",
        f"- Same-noise mean difference: {_fmt(sensitivity.get('same_noise_mean_difference'))}",
        f"- Same-noise near-duplicate rate: {_fmt(sensitivity.get('same_noise_near_duplicate_rate'))}",
        f"- Same-prompt diversity: {_fmt(sensitivity.get('same_prompt_diversity'))}",
        f"- Prompt-pair near-duplicate rate: {_fmt(sensitivity.get('prompt_pair_near_duplicate_rate'))}",
        f"- Warnings: {', '.join(str(item) for item in sensitivity.get('warnings', [])) or '(none)'}",
        f"- Synthetic control failures: `{json.dumps(_jsonable(sensitivity.get('synthetic_control_failures', [])), sort_keys=True)}`",
        "",
    ]
    if ood:
        lines.extend(
            _format_full_v4_ood_markdown(
                ood, source_distribution=source_distribution, generated_vs_source=generated_vs_source
            )
        )
    if bool(checkpoint_evaluation.get("enabled")):
        lines.extend(_format_checkpoint_ood_leaderboard_markdown(checkpoint_evaluation))
    lines.extend(
        [
            "## Final Decision",
            "",
            f"Decision code: **{decision.get('code', 'F')}**",
            "",
            decision.get("label", "Unknown."),
            "",
            "Warnings:",
        ]
    )
    warnings = decision.get("warnings") if isinstance(decision.get("warnings"), list) else []
    if warnings:
        for warning in warnings:
            lines.append(
                f"- {warning.get('name')}: value={_fmt(warning.get('value'))}, threshold={_fmt(warning.get('threshold'))}"
            )
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Train dir: `{artifacts.get('train_dir', '')}`",
            f"- Checkpoint: `{artifacts.get('checkpoint', '')}`",
            f"- Final-step sample checkpoint: `{artifacts.get('final_step_sample_checkpoint', '')}`",
            f"- Sample checkpoint: `{artifacts.get('sample_checkpoint', '')}`",
            f"- Generated dir: `{artifacts.get('generated_dir', '')}`",
            f"- Generated contact sheet: `{artifacts.get('generated_contact_sheet') or ''}`",
            "",
        ]
    )
    return "\n".join(lines)


def _format_palette_projection_summary_lines(summary: Mapping[str, Any]) -> list[str]:
    if not bool(summary.get("applied")):
        return ["- Applied: false"]
    return [
        "- Applied: true",
        f"- Method: `{summary.get('method') or ''}`",
        f"- Target colors: {_fmt_int(summary.get('target_colors'))}",
        f"- Min pixel share: {_fmt(summary.get('min_pixel_share'))}",
        f"- Median visible colors: {_fmt(summary.get('median_visible_color_count_before'))} -> {_fmt(summary.get('median_visible_color_count_after'))}",
        f"- Mean visible colors: {_fmt(summary.get('mean_visible_color_count_before'))} -> {_fmt(summary.get('mean_visible_color_count_after'))}",
        f"- Rare-color rate: {_fmt(summary.get('rare_color_rate_before'))} -> {_fmt(summary.get('rare_color_rate_after'))}",
        f"- Mean RGB MAE visible: {_fmt(summary.get('mean_rgb_mae_visible'))}",
        f"- Destructive rate: {_fmt(summary.get('destructive_rate'))}",
        f"- Safe / moderate / destructive: {_fmt_int(summary.get('safe_count'))} / {_fmt_int(summary.get('moderate_count'))} / {_fmt_int(summary.get('destructive_count'))}",
        f"- Report: `{summary.get('report') or ''}`",
        f"- Before/after contact sheet: `{summary.get('contact_sheet') or ''}`",
    ]


def _format_source_baseline_lines(
    source_distribution: Mapping[str, Any],
    generated_vs_source: Mapping[str, Any],
    eval_key: str,
) -> list[str]:
    weighted_key = "ood_eval_category_weighted" if eval_key == "ood_eval" else "grounded_eval_category_weighted"
    full_metrics = _section(_section(source_distribution, "full_source"), "metrics")
    weighted = _section(source_distribution, weighted_key)
    weighted_metrics = _section(weighted, "metrics")
    deltas = _section(_section(generated_vs_source, eval_key), "deltas")
    return [
        f"- Full-source baseline: `{json.dumps(_jsonable(full_metrics), sort_keys=True)}`",
        f"- Category-weighted source baseline: `{json.dumps(_jsonable(weighted_metrics), sort_keys=True)}`",
        f"- Weighted baseline category counts: `{json.dumps(_jsonable(weighted.get('category_counts', {})), sort_keys=True)}`",
        f"- Missing source categories: `{json.dumps(_jsonable(weighted.get('missing_categories', [])), sort_keys=True)}`",
        f"- Generated-vs-source deltas: `{json.dumps(_jsonable(deltas), sort_keys=True)}`",
    ]


def _format_full_v4_ood_markdown(
    ood: Mapping[str, Any],
    *,
    source_distribution: Mapping[str, Any],
    generated_vs_source: Mapping[str, Any],
) -> list[str]:
    prompts = _section(ood, "prompt_set")
    qa = _section(ood, "generated_qa")
    review = _section(ood, "generated_review")
    palette_projection = _section(ood, "palette_projection")
    faithfulness = _section(ood, "prompt_faithfulness")
    sensitivity = _section(ood, "prompt_sensitivity")
    return [
        "## OOD Compositional",
        "",
        f"- Prompt file: `{prompts.get('prompt_file', '')}`",
        f"- Prompt count: {_fmt_int(prompts.get('prompt_count'))}",
        f"- Category counts: `{json.dumps(_jsonable(prompts.get('category_counts', {})), sort_keys=True)}`",
        f"- Structured fields present: {bool(prompts.get('structured_fields_present'))}",
        f"- QA errors: {_fmt_int(qa.get('errors'))}",
        f"- Projection applied: {bool(palette_projection.get('applied'))}",
        f"- Projection method/target: `{palette_projection.get('method') or ''}` / {_fmt_int(palette_projection.get('target_colors'))}",
        f"- Projection visible colors: {_fmt(palette_projection.get('median_visible_color_count_before'))} -> {_fmt(palette_projection.get('median_visible_color_count_after'))}",
        f"- Projection mean RGB MAE visible: {_fmt(palette_projection.get('mean_rgb_mae_visible'))}",
        f"- Projection destructive rate: {_fmt(palette_projection.get('destructive_rate'))}",
        f"- Touches-border rate: {_fmt(review.get('touches_border_rate'))}",
        f"- Too-many-rare-colors rate: {_fmt(review.get('too_many_rare_colors_rate'))}",
        f"- Source selection: `{json.dumps(_jsonable(faithfulness.get('source_selection', {})), sort_keys=True)}`",
        f"- Nearest-source category consistency: {_fmt(faithfulness.get('nearest_source_category_consistency_rate', faithfulness.get('category_consistency_rate')))}",
        f"- Color consistency: {_fmt(faithfulness.get('color_consistency_rate'))}",
        f"- Repeated silhouette rate: {_fmt(faithfulness.get('repeated_silhouette_rate'))}",
        f"- Generic potion collapse rate: {_fmt(faithfulness.get('generic_potion_collapse_rate'))}",
        f"- Generic blob collapse rate: {_fmt(faithfulness.get('generic_blob_collapse_rate'))}",
        f"- Same-noise mean difference: {_fmt(sensitivity.get('same_noise_mean_difference'))}",
        f"- Same-prompt diversity: {_fmt(sensitivity.get('same_prompt_diversity'))}",
        f"- Prompt-pair near-duplicate rate: {_fmt(sensitivity.get('prompt_pair_near_duplicate_rate'))}",
        "",
        "### OOD Source Baseline",
        "",
        *_format_source_baseline_lines(source_distribution, generated_vs_source, "ood_eval"),
        "",
    ]


def _format_checkpoint_ood_leaderboard_markdown(checkpoint_evaluation: Mapping[str, Any]) -> list[str]:
    leaderboard = (
        checkpoint_evaluation.get("leaderboard") if isinstance(checkpoint_evaluation.get("leaderboard"), list) else []
    )
    lines = [
        "## Checkpoint OOD Leaderboard",
        "",
        f"- Selection metric: `{checkpoint_evaluation.get('selection_metric', OOD_SELECTION_METRIC)}`",
        f"- Selected checkpoint: `{checkpoint_evaluation.get('selected_checkpoint', '')}`",
        f"- Selected step: {_fmt_int(checkpoint_evaluation.get('selected_step'))}",
        f"- Selected deployable: {bool(checkpoint_evaluation.get('selected_deployable'))}",
        "",
        "| Rank | Step | Checkpoint | EMA | OOD score | Grounded pass | Grounded category | Grounded color | OOD pass | OOD QA errors | OOD category | OOD color | Rare-color rate | Repeated silhouette | Blob collapse | Potion collapse | Touches-border | Deployable | Selected |",
        "|---:|---:|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for entry in leaderboard:
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt_int(entry.get("rank")),
                    _fmt_int(entry.get("step")),
                    f"`{entry.get('checkpoint_path') or entry.get('checkpoint') or ''}`",
                    str(bool(entry.get("ema"))),
                    _fmt(entry.get("ood_score")),
                    str(bool(entry.get("grounded_guardrails_passed"))),
                    _fmt(entry.get("grounded_category")),
                    _fmt(entry.get("grounded_color")),
                    str(bool(entry.get("ood_guardrails_passed"))),
                    _fmt_int(entry.get("qa_errors")),
                    _fmt(entry.get("category")),
                    _fmt(entry.get("color")),
                    _fmt(entry.get("rare_color_rate")),
                    _fmt(entry.get("repeated_silhouette_rate")),
                    _fmt(entry.get("blob_collapse_rate")),
                    _fmt(entry.get("potion_collapse_rate")),
                    _fmt(entry.get("touches_border_rate")),
                    str(bool(entry.get("deployable"))),
                    str(bool(entry.get("selected"))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _load_full_v4_report(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    if path.is_dir():
        path = path / "full_v4_challenger_audit.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _faithfulness_source_selection(report: Mapping[str, Any], *, eval_key: str) -> dict[str, Any]:
    section = _section(report, "ood_compositional") if eval_key == "ood" else report
    faithfulness = _section(section, "prompt_faithfulness")
    selection = faithfulness.get("source_selection")
    return dict(selection) if isinstance(selection, Mapping) else {}


_SOURCE_SELECTION_WARNING = (
    "Prompt-faithfulness metrics are not directly comparable because source candidate hashes differ."
)


def _source_selection_comparison(
    baseline_report: Mapping[str, Any],
    structured_report: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = {
        "grounded": _faithfulness_source_selection(baseline_report, eval_key="grounded"),
        "ood": _faithfulness_source_selection(baseline_report, eval_key="ood"),
    }
    structured = {
        "grounded": _faithfulness_source_selection(structured_report, eval_key="grounded"),
        "ood": _faithfulness_source_selection(structured_report, eval_key="ood"),
    }
    warnings: list[str] = []
    for eval_key in ("grounded", "ood"):
        base_hash = baseline[eval_key].get("source_candidate_hash")
        struct_hash = structured[eval_key].get("source_candidate_hash")
        if base_hash and struct_hash and base_hash != struct_hash:
            warnings.append(f"{eval_key}: {_SOURCE_SELECTION_WARNING}")
    return {"baseline": baseline, "structured": structured, "warnings": warnings}


def _checkpoint_selection_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_evaluation = _section(report, "checkpoint_evaluation")
    artifacts = _section(report, "artifacts")
    enabled = bool(checkpoint_evaluation.get("enabled"))
    selected_checkpoint = checkpoint_evaluation.get("selected_checkpoint")
    selected_step = checkpoint_evaluation.get("selected_step")
    selection_score = _optional_float(checkpoint_evaluation.get("selected_score"))
    if selection_score is None:
        selected_metrics = _section(checkpoint_evaluation, "selected_metrics")
        selection_score = _optional_float(selected_metrics.get("ood_score"))
    selected_metrics = _section(checkpoint_evaluation, "selected_metrics")
    sample_checkpoint = artifacts.get("sample_checkpoint")
    final_step_checkpoint = (
        checkpoint_evaluation.get("final_step_checkpoint")
        or artifacts.get("final_step_sample_checkpoint")
        or artifacts.get("checkpoint")
    )
    return {
        "enabled": enabled,
        "selection_metric": checkpoint_evaluation.get("selection_metric") if enabled else None,
        "selected_checkpoint": selected_checkpoint,
        "selected_step": _optional_int(selected_step),
        "selection_score": selection_score,
        "selected_deployable": checkpoint_evaluation.get("selected_deployable") if enabled else None,
        "guardrails_passed": selected_metrics.get("guardrails_passed") if enabled else None,
        "grounded_guardrails_passed": selected_metrics.get("grounded_guardrails_passed") if enabled else None,
        "ood_guardrails_passed": selected_metrics.get("ood_guardrails_passed") if enabled else None,
        "grounded_guardrail_failures": list(selected_metrics.get("grounded_guardrail_failures") or [])
        if enabled
        else [],
        "ood_guardrail_failures": list(selected_metrics.get("ood_guardrail_failures") or []) if enabled else [],
        "guardrail_failures": list(selected_metrics.get("guardrail_failures") or []) if enabled else [],
        "sample_checkpoint": sample_checkpoint,
        "final_step_checkpoint": final_step_checkpoint,
        "sampled_selected_checkpoint": bool(
            enabled and selected_checkpoint and str(sample_checkpoint) == str(selected_checkpoint)
        ),
        "sampled_final_step": bool(final_step_checkpoint and str(sample_checkpoint) == str(final_step_checkpoint)),
    }


def _checkpoint_selection_comparison(
    baseline_report: Mapping[str, Any],
    structured_report: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _checkpoint_selection_summary(baseline_report)
    structured = _checkpoint_selection_summary(structured_report)
    warnings: list[str] = []
    if bool(baseline["enabled"]) != bool(structured["enabled"]):
        warnings.append("comparing final-step run vs selected-checkpoint run: checkpoint selection differs")
    elif bool(baseline["enabled"]) and bool(structured["enabled"]):
        if bool(baseline["sampled_final_step"]) != bool(structured["sampled_final_step"]):
            warnings.append(
                "one selected-checkpoint report sampled the final step while the other selected an earlier checkpoint"
            )
    return {
        "baseline": baseline,
        "structured": structured,
        "both_used_checkpoint_selection": bool(baseline["enabled"] and structured["enabled"]),
        "warnings": warnings,
    }


def _palette_projection_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    projection = _section(report, "palette_projection")
    config = _section(report, "config")
    applied = bool(projection.get("applied"))
    if not projection and config.get("project_palette") is not None:
        applied = bool(config.get("project_palette"))
    return {
        "applied": bool(applied),
        "method": projection.get("method") or config.get("project_palette_method"),
        "target_colors": _optional_int(projection.get("target_colors") or config.get("project_palette_target_colors")),
        "min_pixel_share": _optional_float(
            projection.get("min_pixel_share") or config.get("project_palette_min_pixel_share")
        ),
        "median_visible_color_count_before": _optional_float(projection.get("median_visible_color_count_before")),
        "median_visible_color_count_after": _optional_float(projection.get("median_visible_color_count_after")),
        "mean_rgb_mae_visible": _optional_float(projection.get("mean_rgb_mae_visible")),
        "destructive_rate": _optional_float(projection.get("destructive_rate")),
    }


def _palette_projection_comparison(
    baseline_report: Mapping[str, Any],
    structured_report: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = _palette_projection_summary(baseline_report)
    structured = _palette_projection_summary(structured_report)
    warnings: list[str] = []
    if bool(baseline["applied"]) != bool(structured["applied"]):
        warnings.append("projection differs: comparing projected vs non-projected runs")
    if baseline.get("method") and structured.get("method") and baseline.get("method") != structured.get("method"):
        warnings.append(
            f"projection methods differ: baseline={baseline.get('method')}, structured={structured.get('method')}"
        )
    if (
        baseline.get("target_colors") is not None
        and structured.get("target_colors") is not None
        and baseline.get("target_colors") != structured.get("target_colors")
    ):
        warnings.append(
            f"projection target colors differ: baseline={baseline.get('target_colors')}, structured={structured.get('target_colors')}"
        )
    if (
        baseline.get("min_pixel_share") is not None
        and structured.get("min_pixel_share") is not None
        and baseline.get("min_pixel_share") != structured.get("min_pixel_share")
    ):
        warnings.append(
            "projection min pixel share differs: "
            f"baseline={_fmt(baseline.get('min_pixel_share'))}, structured={_fmt(structured.get('min_pixel_share'))}"
        )
    return {
        "baseline": baseline,
        "structured": structured,
        "warnings": warnings,
    }


def _cfg_scale_comparison_warnings(
    baseline_report: Mapping[str, Any],
    structured_report: Mapping[str, Any],
) -> list[str]:
    baseline_cfg = _optional_float(_section(baseline_report, "config").get("cfg_scale"))
    structured_cfg = _optional_float(_section(structured_report, "config").get("cfg_scale"))
    if baseline_cfg is None or structured_cfg is None or baseline_cfg == structured_cfg:
        return []
    return [f"CFG scales differ: baseline={baseline_cfg:g}, structured={structured_cfg:g}"]


def _comparison_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    return _comparison_metrics_for_eval(report, eval_key="ood" if _section(report, "ood_compositional") else "grounded")


def _comparison_metrics_for_eval(report: Mapping[str, Any], *, eval_key: str) -> dict[str, Any]:
    section = _section(report, "ood_compositional") if eval_key == "ood" else report
    training = _section(report, "training")
    qa = _section(section, "generated_qa")
    config = _section(report, "config")
    review = _full_v4_generated_review_summary(
        _section(section, "generated_review"),
        max_colors=_optional_int(config.get("max_colors")),
    )
    faithfulness = _section(section, "prompt_faithfulness")
    sensitivity = _section(section, "prompt_sensitivity")
    return {
        "generated_qa_errors": qa.get("errors"),
        "category_consistency": faithfulness.get("category_consistency_rate"),
        "color_consistency": faithfulness.get("color_consistency_rate"),
        "repeated_silhouette_rate": faithfulness.get("repeated_silhouette_rate"),
        "generic_potion_collapse": faithfulness.get("generic_potion_collapse_rate"),
        "generic_blob_collapse": faithfulness.get("generic_blob_collapse_rate"),
        "prompt_pair_near_duplicate_rate": sensitivity.get("prompt_pair_near_duplicate_rate"),
        "same_noise_mean_difference": sensitivity.get("same_noise_mean_difference"),
        "same_prompt_diversity": sensitivity.get("same_prompt_diversity"),
        "touches_border_rate": review.get("touches_border_rate"),
        "too_many_rare_colors_rate": review.get("too_many_rare_colors_rate"),
        "mean_visible_color_count": review.get("mean_visible_color_count"),
        "median_visible_color_count": review.get("median_visible_color_count"),
        "train_loss": training.get("final_train_loss"),
        "val_loss": training.get("val_loss"),
        "val_train_loss_ratio": training.get("val_train_loss_ratio"),
    }


def _comparison_table(
    baseline_metrics: Mapping[str, Any],
    structured_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: _metric_comparison(
            baseline_metrics.get(key),
            structured_metrics.get(key),
            higher_is_better=key
            in {
                "category_consistency",
                "color_consistency",
                "same_noise_mean_difference",
                "same_prompt_diversity",
            },
        )
        for key in sorted(set(baseline_metrics) | set(structured_metrics))
    }


def _metric_comparison(baseline: Any, structured: Any, *, higher_is_better: bool) -> dict[str, Any]:
    before = _optional_float(baseline)
    after = _optional_float(structured)
    delta = None if before is None or after is None else after - before
    if delta is None:
        improved = None
    elif higher_is_better:
        improved = delta > 0.0
    else:
        improved = delta < 0.0
    return {
        "baseline": before,
        "structured": after,
        "delta": delta,
        "higher_is_better": bool(higher_is_better),
        "improved": improved,
    }


def _metric_status(comparison: Mapping[str, Any] | None, *, tolerance: float = 1.0e-6) -> str:
    if not isinstance(comparison, Mapping):
        return "unavailable"
    delta = _optional_float(comparison.get("delta"))
    if delta is None:
        return "unavailable"
    if abs(delta) <= float(tolerance):
        baseline = _optional_float(comparison.get("baseline"))
        structured = _optional_float(comparison.get("structured"))
        if baseline is not None and structured is not None and baseline >= 0.95 and structured >= 0.95:
            return "unchanged/saturated"
        return "unchanged"
    improved = comparison.get("improved")
    return "improved" if bool(improved) else "degraded"


def _source_delta_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    generated_vs_source = _section(report, "generated_vs_source")
    return {
        "grounded_eval": _section(_section(generated_vs_source, "grounded_eval"), "deltas"),
        "ood_eval": _section(_section(generated_vs_source, "ood_eval"), "deltas"),
    }


def _source_distribution_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    source_distribution = _section(report, "source_distribution")
    return {
        "full_source": _section(_section(source_distribution, "full_source"), "metrics"),
        "grounded_eval_category_weighted": _section(
            _section(source_distribution, "grounded_eval_category_weighted"),
            "metrics",
        ),
        "ood_eval_category_weighted": _section(
            _section(source_distribution, "ood_eval_category_weighted"),
            "metrics",
        ),
    }


def _report_with_computed_source_deltas(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if _section(report, "source_distribution") and _section(report, "generated_vs_source"):
        return report
    dataset = _report_path_value(report, "dataset")
    training_manifest = _report_path_value(report, "training_manifest")
    if dataset is None or training_manifest is None:
        return report

    result: dict[str, Any] = dict(report)
    prompt_set = _section(result, "prompt_set")
    source_distribution = _source_distribution_report(
        dataset=dataset,
        training_manifest=training_manifest,
        grounded_category_counts=prompt_set.get("category_counts", {}),
    )
    generated_vs_source = {
        "grounded_eval": _generated_vs_source_delta(
            _full_v4_generated_review_summary(
                _section(result, "generated_review"),
                max_colors=_optional_int(_section(result, "config").get("max_colors")),
            ),
            _section(source_distribution, "grounded_eval_category_weighted"),
        )
    }
    ood = _section(result, "ood_compositional")
    if ood:
        ood_prompt_set = _section(ood, "prompt_set")
        source_distribution["ood_eval_category_weighted"] = _category_weighted_source_baseline(
            _section(source_distribution, "by_category"),
            ood_prompt_set.get("category_counts", {}),
        )
        generated_vs_source["ood_eval"] = _generated_vs_source_delta(
            _full_v4_generated_review_summary(
                _section(ood, "generated_review"),
                max_colors=_optional_int(_section(result, "config").get("max_colors")),
            ),
            _section(source_distribution, "ood_eval_category_weighted"),
        )
    result["source_distribution"] = source_distribution
    result["generated_vs_source"] = generated_vs_source
    return result


def _report_path_value(report: Mapping[str, Any], key: str) -> Path | None:
    value = report.get(key)
    if value is None:
        value = _section(report, "config").get(key)
    if value is None:
        return None
    return Path(str(value))


def _rare_color_comparison(
    baseline_report: Mapping[str, Any],
    structured_report: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any] | None,
) -> dict[str, Any]:
    baseline_delta = _rare_color_source_delta(baseline_report)
    structured_delta = _rare_color_source_delta(structured_report)
    if baseline_delta is None or structured_delta is None:
        return (
            dict(fallback) if isinstance(fallback, Mapping) else _metric_comparison(None, None, higher_is_better=False)
        )
    return _metric_comparison(baseline_delta, structured_delta, higher_is_better=False)


def _rare_color_source_delta(report: Mapping[str, Any]) -> float | None:
    generated_vs_source = _section(report, "generated_vs_source")
    for key in ("ood_eval", "grounded_eval"):
        delta = _optional_float(_section(_section(generated_vs_source, key), "deltas").get("mean_visible_color_count"))
        if delta is not None:
            return delta
    return None


def _improved(comparison: Mapping[str, Any] | None) -> bool | None:
    if comparison is None:
        return None
    improved = comparison.get("improved")
    return bool(improved) if improved is not None else None


def _format_challenger_conditioning_comparison_markdown(report: Mapping[str, Any]) -> str:
    baseline = _section(report, "baseline")
    structured = _section(report, "structured")
    grounded = _section(_section(report, "dataset_grounded"), "comparisons")
    ood = _section(_section(report, "ood_compositional"), "comparisons")
    source_deltas = _section(report, "source_distribution_deltas")
    answers = _section(report, "answers")
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    lines = [
        "# Challenger Conditioning Comparison",
        "",
        f"Baseline mode: `{baseline.get('conditioning_mode', '')}`",
        f"Structured mode: `{structured.get('conditioning_mode', '')}`",
        f"Both used checkpoint selection: {bool(_section(report, 'checkpoint_selection_comparison').get('both_used_checkpoint_selection'))}",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines += [
        "## Checkpoint selection",
        "",
        "| Run | Enabled | Selected step | Selection score | Guardrails | Grounded | OOD | Selected checkpoint |",
        "|---|---|---:|---:|---|---|---|---|",
        (
            "| Baseline | "
            + " | ".join(
                [
                    str(bool(_section(baseline, "checkpoint_selection").get("enabled"))),
                    _fmt_int(baseline.get("selected_step")),
                    _fmt(baseline.get("selection_score")),
                    str(_section(baseline, "checkpoint_selection").get("guardrails_passed")),
                    str(_section(baseline, "checkpoint_selection").get("grounded_guardrails_passed")),
                    str(_section(baseline, "checkpoint_selection").get("ood_guardrails_passed")),
                    f"`{baseline.get('selected_checkpoint') or ''}`",
                ]
            )
            + " |"
        ),
        (
            "| Structured | "
            + " | ".join(
                [
                    str(bool(_section(structured, "checkpoint_selection").get("enabled"))),
                    _fmt_int(structured.get("selected_step")),
                    _fmt(structured.get("selection_score")),
                    str(_section(structured, "checkpoint_selection").get("guardrails_passed")),
                    str(_section(structured, "checkpoint_selection").get("grounded_guardrails_passed")),
                    str(_section(structured, "checkpoint_selection").get("ood_guardrails_passed")),
                    f"`{structured.get('selected_checkpoint') or ''}`",
                ]
            )
            + " |"
        ),
        "",
        "## Palette projection",
        "",
        "| Run | Applied | Method | Target colors | Min pixel share | Median colors | Mean RGB MAE | Destructive rate |",
        "|---|---|---|---:|---:|---|---:|---:|",
        _format_projection_comparison_row("Baseline", _section(baseline, "palette_projection")),
        _format_projection_comparison_row("Structured", _section(structured, "palette_projection")),
        "",
        "## Answers",
        "",
        f"- Dataset-grounded category: {answers.get('dataset_grounded_category')}",
        f"- OOD category: {answers.get('ood_category')}",
        f"- Dataset-grounded color: {answers.get('dataset_grounded_color')}",
        f"- OOD color: {answers.get('ood_color')}",
        f"- OOD blob collapse: {answers.get('ood_blob_collapse')}",
        f"- Rare-color rate: {answers.get('rare_color_rate')}",
        f"- Micro-overfit preserved: {answers.get('micro_overfit_preserved')}",
        "",
        "## Dataset-grounded eval comparison",
        "",
        "| Metric | Baseline | Structured | Delta | Improved |",
        "|---|---:|---:|---:|---|",
    ]
    for key, value in grounded.items():
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key),
                    _fmt(value.get("baseline")),
                    _fmt(value.get("structured")),
                    _fmt(value.get("delta")),
                    str(value.get("improved")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## OOD compositional eval comparison",
            "",
            "| Metric | Baseline | Structured | Delta | Improved |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for key, value in ood.items():
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key),
                    _fmt(value.get("baseline")),
                    _fmt(value.get("structured")),
                    _fmt(value.get("delta")),
                    str(value.get("improved")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Source-distribution deltas",
            "",
            f"- Baseline source baselines: `{json.dumps(_jsonable(source_deltas.get('baseline_source_baselines', {})), sort_keys=True)}`",
            f"- Structured source baselines: `{json.dumps(_jsonable(source_deltas.get('structured_source_baselines', {})), sort_keys=True)}`",
            f"- Baseline generated-vs-source deltas: `{json.dumps(_jsonable(source_deltas.get('baseline', {})), sort_keys=True)}`",
            f"- Structured generated-vs-source deltas: `{json.dumps(_jsonable(source_deltas.get('structured', {})), sort_keys=True)}`",
            f"- Rare-color source-delta comparison: `{json.dumps(_jsonable(source_deltas.get('rare_color_comparison', {})), sort_keys=True)}`",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _format_projection_comparison_row(label: str, summary: Mapping[str, Any]) -> str:
    before = _fmt(summary.get("median_visible_color_count_before"))
    after = _fmt(summary.get("median_visible_color_count_after"))
    colors = f"{before} -> {after}" if before != "NA" or after != "NA" else "NA"
    return (
        f"| {label} | "
        + " | ".join(
            [
                str(bool(summary.get("applied"))),
                f"`{summary.get('method') or ''}`",
                _fmt_int(summary.get("target_colors")),
                _fmt(summary.get("min_pixel_share")),
                colors,
                _fmt(summary.get("mean_rgb_mae_visible")),
                _fmt(summary.get("destructive_rate")),
            ]
        )
        + " |"
    )


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _warn_if_gt(warnings: list[dict[str, Any]], name: str, value: Any, threshold: float) -> None:
    numeric = _optional_float(value)
    if numeric is not None and numeric > float(threshold):
        warnings.append({"name": name, "value": numeric, "threshold": float(threshold), "direction": ">"})


def _warn_if_lt(warnings: list[dict[str, Any]], name: str, value: Any, threshold: float) -> None:
    numeric = _optional_float(value)
    if numeric is not None and numeric < float(threshold):
        warnings.append({"name": name, "value": numeric, "threshold": float(threshold), "direction": "<"})


def _warn_source_delta(warnings: list[dict[str, Any]], name: str, value: Any, threshold: float) -> None:
    numeric = _optional_float(value)
    if numeric is not None and numeric > float(threshold):
        warnings.append({"name": name, "value": numeric, "threshold": float(threshold), "direction": "source_delta>"})


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _category(row: Mapping[str, Any]) -> str:
    value = str(row.get("category") or row.get("prompt_category") or "unknown").strip()
    return value or "unknown"


def _write_overfit_sprite_id_file(
    path: Path,
    selection: OverfitSubsetSelection,
    *,
    audit_type: str,
    run_name: str,
    training_manifest: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "overfit_sprite_ids_v1",
        "audit_type": audit_type,
        "run_name": run_name,
        "training_manifest": str(training_manifest),
        **selection.to_report(),
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_subset_prompts(path: Path, selection: OverfitSubsetSelection) -> None:
    first_by_id: dict[str, Mapping[str, Any]] = {}
    for row in selection.rows:
        sprite_id = str(row.get("sprite_id", ""))
        if sprite_id and sprite_id not in first_by_id:
            first_by_id[sprite_id] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sprite_id in enumerate(selection.sprite_ids):
        row = first_by_id[sprite_id]
        rows.append(
            {
                **dict(row),
                "prompt": str(row.get("caption") or row.get("object_name") or sprite_id),
                "prompt_id": sprite_id,
                "target_sprite_id": sprite_id,
                "source_sprite_id": sprite_id,
                "subset_index": index,
            }
        )
    path.write_text("".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prompt_sprite_ids(path: Path) -> tuple[str, ...]:
    return tuple(_record_sprite_id(record) for record in read_prompt_records(path))


def _generated_target_sprite_ids(generated_dir: Path) -> tuple[str, ...]:
    manifest_path = Path(generated_dir) / "generated_manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"{generated_dir}: generated_manifest.jsonl is missing before source-match")
    rows: list[str] = []
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{manifest_path}:{line_no}: expected JSON object")
        rows.append(_record_sprite_id(value))
    return tuple(rows)


def _record_sprite_id(record: Mapping[str, Any]) -> str:
    for key in ("target_sprite_id", "source_sprite_id", "sprite_id", "prompt_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"record has no sprite ID field: {record}")


def _assert_overfit_subset_matches(
    train_report: Mapping[str, Any],
    prompt_sprite_ids: Sequence[str],
    *,
    run_name: str,
) -> dict[str, Any]:
    subset = train_report.get("overfit_subset")
    if not isinstance(subset, Mapping):
        raise ValueError(f"{run_name}: train_report.overfit_subset is missing for micro-overfit audit")
    train_sprite_ids = tuple(str(sprite_id) for sprite_id in subset.get("sprite_ids") or [])
    return _assert_sprite_id_sets_match(
        train_sprite_ids,
        prompt_sprite_ids,
        run_name=run_name,
        left_label="train_report.overfit_subset.sprite_ids",
        right_label="prompt_sprite_ids",
    )


def _assert_generated_targets_match_prompts(
    generated_dir: Path,
    prompt_sprite_ids: Sequence[str],
    *,
    run_name: str,
) -> dict[str, Any]:
    generated_sprite_ids = _generated_target_sprite_ids(generated_dir)
    return _assert_sprite_id_sets_match(
        generated_sprite_ids,
        prompt_sprite_ids,
        run_name=run_name,
        left_label="generated_manifest.target_sprite_ids",
        right_label="prompt_sprite_ids",
    )


def _assert_source_match_targets_match_prompts(
    source_match: Mapping[str, Any],
    prompt_sprite_ids: Sequence[str],
    *,
    run_name: str,
) -> dict[str, Any]:
    samples = source_match.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError(f"{run_name}: source_match.samples is missing for micro-overfit audit")
    source_match_sprite_ids = tuple(_record_sprite_id(sample) for sample in samples if isinstance(sample, Mapping))
    return _assert_sprite_id_sets_match(
        source_match_sprite_ids,
        prompt_sprite_ids,
        run_name=run_name,
        left_label="source_match.samples.target_sprite_ids",
        right_label="prompt_sprite_ids",
    )


def _assert_sprite_id_sets_match(
    left_sprite_ids: Sequence[str],
    right_sprite_ids: Sequence[str],
    *,
    run_name: str,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left = tuple(str(sprite_id) for sprite_id in left_sprite_ids)
    right = tuple(str(sprite_id) for sprite_id in right_sprite_ids)
    left_set = set(left)
    right_set = set(right)
    missing_from_left = sorted(right_set - left_set)
    extra_in_left = sorted(left_set - right_set)
    if left_set != right_set:
        raise ValueError(
            f"{run_name}: overfit subset mismatch between {left_label} and {right_label}; "
            f"{left_label} count={len(left_set)}, {right_label} count={len(right_set)}, "
            f"missing_from_{left_label}={missing_from_left}, extra_in_{left_label}={extra_in_left}"
        )
    return {
        "sets_equal": True,
        "ordered_equal": left == right,
        "left_label": left_label,
        "right_label": right_label,
        _left_sprite_id_key(left_label): list(left),
        "prompt_sprite_ids": list(right),
        "missing_from_left": missing_from_left,
        "extra_in_left": extra_in_left,
    }


def _left_sprite_id_key(left_label: str) -> str:
    if left_label.startswith("train_report"):
        return "train_sprite_ids"
    if left_label.startswith("source_match"):
        return "source_match_sprite_ids"
    return "generated_sprite_ids"


def _audit_budget_metadata(
    spec: Mapping[str, Any],
    budget: Mapping[str, Any],
    train_report: Mapping[str, Any],
) -> dict[str, Any]:
    sprite_count = _optional_int(spec.get("count"))
    train_row_count = _optional_int(
        train_report.get("effective_train_records")
        if train_report.get("effective_train_records") is not None
        else train_report.get("train_records")
    )
    steps = int(budget.get("steps") or train_report.get("steps_completed") or train_report.get("max_steps") or 0)
    batch_size = int(spec.get("batch_size") or train_report.get("batch_size") or 0)
    steps_per_sprite = None if sprite_count is None else float(steps) / float(sprite_count)
    total_sample_slots = steps * batch_size if steps > 0 and batch_size > 0 else None
    exposure = {
        "optimizer_steps_per_sprite": steps_per_sprite,
        "sample_slots_total": total_sample_slots,
        "sample_slots_per_sprite": None
        if total_sample_slots is None or sprite_count is None
        else float(total_sample_slots) / float(sprite_count),
        "sample_slots_per_train_row": None
        if total_sample_slots is None or train_row_count is None
        else float(total_sample_slots) / float(train_row_count),
    }
    return {
        "sprite_count": sprite_count,
        "train_row_count": train_row_count,
        "steps": steps,
        "steps_per_sprite": steps_per_sprite,
        "batch_size": batch_size,
        "approx_update_exposure": exposure,
        "step_count_source": str(budget.get("step_count_source") or "explicit"),
        "step_count_defaulted_from_helper": bool(budget.get("step_count_defaulted_from_helper")),
        "micro_overfit_budget_policy": {
            "default_steps_per_sprite": budget.get("default_steps_per_sprite"),
            "default_minimum_steps": budget.get("default_minimum_steps"),
            "default_step_rounding": budget.get("default_step_rounding"),
        },
        "budget_note": budget.get("budget_note"),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _audit_run_summary(
    spec: Mapping[str, Any],
    *,
    budget: Mapping[str, Any],
    train_report: Mapping[str, Any],
    sample_report: Mapping[str, Any],
    qa: Mapping[str, Any],
    review: Mapping[str, Any],
    source_match: Mapping[str, Any] | None = None,
    prompt_sensitivity: Mapping[str, Any] | None = None,
    prompt_faithfulness: Mapping[str, Any] | None = None,
    overfit_subset: Mapping[str, Any] | None = None,
    generated_target_subset: Mapping[str, Any] | None = None,
    source_match_target_subset: Mapping[str, Any] | None = None,
    overfit_sprite_id_file: Path | None = None,
) -> dict[str, Any]:
    overall = review.get("overall") if isinstance(review.get("overall"), Mapping) else {}
    budget_metadata = _audit_budget_metadata(spec, budget, train_report)
    return {
        "name": str(spec.get("name") or ""),
        **budget_metadata,
        "final_train_loss": train_report.get("final_train_loss"),
        "loss_decrease": train_report.get("loss_decrease"),
        "sample_count": sample_report.get("sample_count"),
        "qa_ok": bool(qa.get("ok")),
        "qa_errors": len(qa.get("errors") or []),
        "review_total_warnings": overall.get("total_warnings"),
        "review_mean_alpha_coverage": overall.get("mean_alpha_coverage"),
        "source_match_mean_visible_rgb_mae": None if source_match is None else source_match.get("mean_visible_rgb_mae"),
        "source_match_mean_alpha_iou": None if source_match is None else source_match.get("mean_alpha_iou"),
        "source_match_near_match_rate": None if source_match is None else source_match.get("near_match_rate"),
        "prompt_sensitivity_warnings": [] if prompt_sensitivity is None else prompt_sensitivity.get("warnings", []),
        "prompt_faithfulness_repeated_silhouette_rate": None
        if prompt_faithfulness is None
        else prompt_faithfulness.get("repeated_silhouette_rate"),
        "prompt_faithfulness_color_consistency_rate": None
        if prompt_faithfulness is None
        else prompt_faithfulness.get("color_consistency_rate"),
        "subset_equality": None if overfit_subset is None else bool(overfit_subset.get("sets_equal")),
        "overfit_sprite_id_file": None if overfit_sprite_id_file is None else str(overfit_sprite_id_file),
        "overfit_subset": None if overfit_subset is None else dict(overfit_subset),
        "generated_target_subset": None if generated_target_subset is None else dict(generated_target_subset),
        "source_match_target_subset": None if source_match_target_subset is None else dict(source_match_target_subset),
        "status": _run_status(qa, source_match, sprite_count=budget_metadata["sprite_count"]),
    }


def _run_status(
    qa: Mapping[str, Any],
    source_match: Mapping[str, Any] | None,
    *,
    sprite_count: int | None = None,
) -> str:
    if len(qa.get("errors") or []) > 0:
        return "fail"
    if source_match is None:
        return "pass"
    near = source_match.get("near_match_rate")
    alpha = source_match.get("mean_alpha_iou")
    if sprite_count != 64:
        if isinstance(near, (int, float)) and float(near) >= 0.50:
            return "pass"
        if isinstance(alpha, (int, float)) and float(alpha) >= 0.75:
            return "warn"
        return "fail"

    rgb_mae = source_match.get("mean_visible_rgb_mae")
    if not (isinstance(alpha, (int, float)) and isinstance(rgb_mae, (int, float)) and isinstance(near, (int, float))):
        return "fail"
    if float(alpha) >= 0.90 and float(rgb_mae) <= 0.10 and float(near) >= 0.65:
        return "pass"
    if float(alpha) >= 0.85 and float(rgb_mae) <= 0.15 and float(near) >= 0.45:
        return "warn"
    return "fail"


def _regression_decision(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    small = [run for run in runs if run.get("sprite_count") in (16, 32)]
    can_overfit = any(str(run.get("status")) == "pass" for run in small)
    return {
        "can_micro_overfit": can_overfit,
        "recommendation": (
            "full-v4 failure is likely data/regularization/capacity/conditioning limited"
            if can_overfit
            else "regression architecture or objective is too weak for crisp memorization"
        ),
    }


def _challenger_decision(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overfit = [run for run in runs if str(run.get("name", "")).startswith("overfit_")]
    statuses = [str(run.get("status")) for run in overfit]
    if statuses and all(status == "pass" for status in statuses):
        gate = "pass"
    elif any(status == "fail" for status in statuses):
        gate = "fail"
    else:
        gate = "warn"
    return {
        "micro_overfit_gate": gate,
        "recommendation": (
            "continue challenger architecture"
            if gate == "pass"
            else "inspect challenger micro-overfit scaling before broadening runs"
        ),
    }


def _write_audit_report(out_dir: Path, report: Mapping[str, Any], *, filename: str) -> None:
    (out_dir / f"{filename}.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / f"{filename}.md").write_text(_format_audit_markdown(report), encoding="utf-8")


def _format_audit_markdown(report: Mapping[str, Any]) -> str:
    decision = report.get("decision") if isinstance(report.get("decision"), Mapping) else {}
    lines = [
        "# Generator Audit Report",
        "",
        f"Audit type: `{report.get('audit_type', '')}`",
        f"Seed: `{report.get('seed', '')}`",
        f"Recommendation: **{decision.get('recommendation', '')}**",
        "",
        "## Budget Policy",
        "",
        (
            "Default micro-overfit runs normalize training budget by unique sprite count: "
            f"`steps_per_sprite={MICRO_OVERFIT_STEPS_PER_SPRITE}`, "
            f"`minimum_steps={MICRO_OVERFIT_MINIMUM_STEPS}`, and "
            f"`round_to={MICRO_OVERFIT_STEP_ROUNDING}`. This gives 16 sprites -> 3000 steps, "
            "32 sprites -> about 6000 steps, and 64 sprites -> about 12000 steps."
        ),
        "",
        (
            "Old fixed-step comparisons are not directly fair across sprite counts. Compare 16- and 32-sprite "
            "micro-overfit runs by `steps_per_sprite` and exposure, not total steps alone."
        ),
        "",
        (
            "Runs marked `explicit` or `explicit_regression_specific` use an audit-defined step count instead of "
            "the default helper; regression-specific budgets are retained for continuity and reported explicitly."
        ),
        "",
        (
            "The 64-sprite source-match gate is pass at alpha IoU >= 0.90, visible RGB MAE <= 0.10, "
            "and near-match >= 0.65; warn at alpha IoU >= 0.85, visible RGB MAE <= 0.15, "
            "and near-match >= 0.45."
        ),
        "",
        "## Runs",
        "",
        "| Run | Status | Sprites | Train rows | Steps | Steps/sprite | Batch | Budget | Subset match | QA errors | Warnings | Source RGB MAE | Source alpha IoU | Near-match | Loss |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.get("runs") or []:
        if not isinstance(run, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("name", "")),
                    str(run.get("status", "")),
                    _fmt_int(run.get("sprite_count")),
                    _fmt_int(run.get("train_row_count")),
                    _fmt_int(run.get("steps")),
                    _fmt(run.get("steps_per_sprite")),
                    _fmt_int(run.get("batch_size")),
                    str(run.get("step_count_source", "")),
                    _subset_match_label(run.get("subset_equality")),
                    str(int(run.get("qa_errors") or 0)),
                    str(int(run.get("review_total_warnings") or 0)),
                    _fmt(run.get("source_match_mean_visible_rgb_mae")),
                    _fmt(run.get("source_match_mean_alpha_iou")),
                    _fmt(run.get("source_match_near_match_rate")),
                    _fmt(run.get("final_train_loss")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _subset_match_label(value: Any) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
