"""Small conditional rectified-flow challenger for 32x32 RGBA sprites."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import platform
import random
import re
import stat
import time
import warnings
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from spritelab.harvest.label_v4.training_quality import (
    dataset_uncertainty_report,
    evaluation_uncertainty_report,
    training_uncertainty_report,
    uncertainty_correlation_report,
)
from spritelab.product_core import strict_json_loads
from spritelab.training.checkpoint_io import load_checkpoint as _load_checkpoint
from spritelab.training.checkpoint_io import tokenizer_from_checkpoint as _tokenizer_from_checkpoint
from spritelab.training.conditioning import (
    CONDITIONING_MODES,
    DEFAULT_CONDITIONING_MODE,
    apply_conditioning_mode,
    checkpoint_semantic_max_length,
    uses_structured_conditioning,
    validate_conditioning_mode,
)
from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch, read_jsonl
from spritelab.training.device import move_batch_to_device, resolve_device
from spritelab.training.generated_canonicalizer import (
    build_generation_contact_sheet,
    canonicalize_generated_rgba,
    write_generated_sprite_artifacts,
    write_generation_reports,
)
from spritelab.training.optim_utils import (
    amp_autocast,
    apply_backend_speed_flags,
    build_adamw,
    build_lr_scheduler,
    clip_gradients,
    dataloader_perf_kwargs,
    device_type,
)
from spritelab.training.overfit_subset import (
    OverfitSubsetSelection,
    read_sprite_id_list,
    select_overfit_subset,
)
from spritelab.training.palette_conditioning import (
    PALETTE_CONDITION_CHANNELS,
    PALETTE_CONDITION_SLOTS,
    PaletteConditionLibrary,
    build_palette_condition_library,
)
from spritelab.training.palette_swap import (
    DEFAULT_SWAP_FAMILIES_TEXT,
    PaletteSwapConfig,
    estimate_applied,
)
from spritelab.training.progress import StepProgressBar
from spritelab.training.prompt_records import read_prompt_records
from spritelab.training.prompt_sensitivity import COLOR_WORDS
from spritelab.training.rgba import save_rgba_contact_sheet
from spritelab.training.role_ramp_transplant import RoleRampTransplantConfig
from spritelab.training.sampler_resume import (
    StatefulPermutationSampler,
    UnsupportedExactResumeError,
    inspect_sampler_resume_state,
    validate_worker_mode,
)
from spritelab.training.structured_conditioning import (
    MULTI_HOT_FIELDS,
    STATUS_FIELDS,
    STRUCTURED_BATCH_KEYS,
    StructuredConditioningVocab,
    build_structured_conditioning_vocab,
    encode_structured_conditioning,
    save_structured_conditioning_vocab,
    structured_vocab_from_checkpoint,
)
from spritelab.training.timestep_validation import (
    DEFAULT_TIMESTEP_BOUNDARIES,
    TimestepBucketAccumulator,
    validate_timestep_boundaries,
)
from spritelab.training.tokenization import SpriteTextTokenizer
from spritelab.utils.safe_fs import AnchoredDirectory, OwnedFileIdentity, UnsafeFilesystemOperation

SPRITE_SIZE = 32
AUXILIARY_HEAD_PREFIXES = ("palette_head_rgb.", "palette_head_presence.", "index_head.")
FORWARD_OUTPUT_SCHEMA_VERSION = "spritelab_generator_forward_v2"
CAMPAIGN_RUN_CONTRACT_SCHEMA_VERSION = "spritelab_generator_campaign_run_contract_v1"
CAMPAIGN_EVALUATION_RECORD_SCHEMA_VERSION = "spritelab_generator_campaign_evaluation_v1"


class AuxiliaryHeadsMode(str, Enum):
    """Physical palette/index head construction; independent of loss weights."""

    ABSENT = "absent"
    PALETTE_INDEX = "palette_index"


def normalize_auxiliary_heads_mode(value: AuxiliaryHeadsMode | str | None) -> tuple[AuxiliaryHeadsMode, bool]:
    """Return physical mode and whether the historical adapter selected it.

    Historical configs omitted this field.  They always constructed the heads,
    including when every auxiliary loss weight was zero, so omission maps to a
    marked legacy palette/index architecture and never to a headless model.
    """
    if value is None:
        return AuxiliaryHeadsMode.PALETTE_INDEX, True
    if isinstance(value, AuxiliaryHeadsMode):
        return value, False
    try:
        return AuxiliaryHeadsMode(str(value)), False
    except ValueError as exc:
        raise ValueError("auxiliary_heads_mode must be 'absent' or 'palette_index'") from exc


def resolve_auxiliary_heads_mode(
    value: AuxiliaryHeadsMode | str | None, experiment_manifest: Mapping[str, Any] | None
) -> tuple[AuxiliaryHeadsMode, bool]:
    """Resolve a direct option or the same explicit option bound in a manifest."""
    if value is None and isinstance(experiment_manifest, Mapping):
        architecture = experiment_manifest.get("model_architecture")
        if isinstance(architecture, Mapping) and architecture.get("identity_kind") == "explicit":
            value = architecture.get("auxiliary_heads_mode")
    return normalize_auxiliary_heads_mode(value)


# Palette range cache: hoists per-step GPU→CPU sync out of the hot loop.
# Training palettes are consistently either 0-1 or 0-255 for the entire
# dataset; we check once on first encounter and cache the answer.
_palette_scale_needs_normalize: bool | None = None


def _ensure_palette_in_01(palette: Any) -> None:
    """Normalize palette [0,255] to [0,1] in-place if needed (cached check)."""
    global _palette_scale_needs_normalize
    if _palette_scale_needs_normalize is None:
        if palette.numel() and float(palette.detach().max().cpu()) > 1.0:
            palette.div_(255.0)
            _palette_scale_needs_normalize = True
        else:
            _palette_scale_needs_normalize = False
    elif _palette_scale_needs_normalize:
        palette.div_(255.0)


STRUCTURED_DROPOUT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("category", ("category_id", "category_status_id")),
    ("object_id", ("object_id", "object_status_id")),
    ("base_object_id", ("base_object_id", "base_object_status_id")),
    ("colors", ("primary_color_id", "color_multi_hot", "primary_color_status_id", "color_status_id")),
    ("materials", ("material_multi_hot", "material_status_id")),
    ("shapes", ("shape_multi_hot", "shape_status_id")),
    ("function", ("function_multi_hot", "function_status_id")),
    ("style", ("style_multi_hot", "style_status_id")),
)

# v2 Phase 0 diagnostics: sampling-time field ablation choices (see docs/v2_phase0_diagnostics.md).
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
_NULL_FIELD_STRUCTURED_KEYS: dict[str, tuple[str, ...]] = {
    "category": ("category_id", "category_status_id"),
    "object_id": ("object_id", "object_status_id"),
    "base_object": ("base_object_id", "base_object_status_id"),
    "colors": ("primary_color_id", "color_multi_hot", "primary_color_status_id", "color_status_id"),
    "materials": ("material_multi_hot", "material_status_id"),
    "shapes": ("shape_multi_hot", "shape_status_id"),
    "function": ("function_multi_hot", "function_status_id"),
    "style": ("style_multi_hot", "style_status_id"),
}
_COLOR_STRUCTURED_KEYS: tuple[str, ...] = (
    "primary_color_id",
    "color_multi_hot",
    "primary_color_status_id",
    "color_status_id",
)

_ModuleBase = nn.Module if nn is not None else object


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for spritelab challenger generator.")
    return torch, nn


@dataclass(frozen=True)
class ChallengerTrainConfig:
    dataset_dir: Path
    training_manifest: Path
    out_dir: Path
    architecture: str = "rectified_flow"
    split: str = "train"
    batch_size: int = 32
    max_steps: int = 5000
    learning_rate: float = 2e-4
    device: str = "cpu"
    seed: int = 123
    num_workers: int = 0
    conditioning_mode: str = DEFAULT_CONDITIONING_MODE
    cfg_dropout: float = 0.1
    structured_field_dropout: float = 0.0
    ema_decay: float = 0.999
    foreground_rgb_loss_weight: float = 1.0
    background_rgb_loss_weight: float = 1.0
    palette_loss_weight: float = 0.0
    palette_loss_temperature: float = 0.05
    palette_swap_augmentation: bool = False
    palette_swap_prob: float = 0.0
    palette_swap_families: str = DEFAULT_SWAP_FAMILIES_TEXT
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
    role_ramp_transplant_prob: float = 0.0
    role_ramp_transplant_keep_original_prob: float = 0.5
    role_ramp_transplant_exclude_families: str = "gold,brown"
    role_ramp_transplant_require_trusted_role_map: bool = True
    role_ramp_transplant_debug_samples: int = 0
    role_ramp_transplant_max_resample_attempts: int = 8
    role_ramp_transplant_require_fill_target_match: bool = True
    role_ramp_transplant_min_primary_fill_coverage: float = 0.03
    base_channels: int = 64
    channel_mults: str = "1,2,4"
    res_blocks_per_level: int = 2
    embed_dim: int = 64
    sample_every: int = 250
    save_every: int = 1000
    checkpoint_steps: tuple[int, ...] = ()
    caption_policy_filter: str | None = None
    caption_max_length: int = 32
    semantic_max_length: int = 48
    max_records: int | None = None
    max_train_sprites: int | None = None
    sprite_id_list: Path | None = None
    overfit_split: str | None = None
    validation_mode: str = "auto"
    # Opt-in speed/quality knobs; defaults keep training numerically identical.
    amp: bool = False
    grad_clip: float = 0.0
    lr_schedule: str = "none"
    lr_warmup_steps: int = 0
    # Opt-in training-loop speed knobs (see docs/training_speed_notes.md). Every
    # field here is a no-op at its default: metrics_every=1 syncs every step
    # (unchanged), fused_adamw/cudnn_benchmark/tf32 default off, and
    # eval_max_batches=0 evaluates the full loader as before.
    metrics_every: int = 1
    fused_adamw: bool = False
    cudnn_benchmark: bool = False
    tf32: bool = False
    eval_max_batches: int = 0
    # v2 Phase 1 conditioning architecture (default-off; see docs/v2_phase1_conditioning.md)
    film_conditioning: bool = False
    bottleneck_attention: bool = False
    structured_field_dropout_rates: dict[str, float] | None = None
    # v2 Phase 2 palette/index auxiliary heads (default-off; see docs/v2_phase2_palette_index_heads.md)
    index_head_loss_weight: float = 0.0
    palette_head_loss_weight: float = 0.0
    palette_presence_loss_weight: float = 0.0
    index_head_warmup_steps: int = 0
    palette_head_use_gt_palette_prob: float = 1.0
    auxiliary_heads_mode: AuxiliaryHeadsMode | str | None = None
    # v3 Phase 0 explicit canonical palette input (default-off).
    palette_conditioning: bool = False
    palette_conditioning_dropout: float = 0.0
    palette_conditioning_dim: int = 64
    palette_conditioning_inject: str = "decoder"
    experiment_manifest: dict[str, Any] | None = None
    resume_from: Path | None = None
    retained_output_root: Path | None = None
    retained_training_manifest_records: Sequence[Mapping[str, Any]] | None = None
    retained_dataset_descriptors: Mapping[str, int] | None = None
    retained_dataset_content_sha256: Mapping[str, str] | None = None
    resume_descriptor: int | None = None
    expected_resume_sha256: str | None = None
    campaign_run_contract: dict[str, Any] | None = None
    unsafe_resume: bool = False
    unsafe_resume_reason: str | None = None
    determinism: str = "off"
    gradient_accumulation_steps: int = 1
    timestep_validation_boundaries: tuple[float, ...] = DEFAULT_TIMESTEP_BOUNDARIES
    stop_after_step: int | None = None


@dataclass(frozen=True)
class ChallengerSampleConfig:
    checkpoint: Path
    prompts: Path
    out_dir: Path
    expected_checkpoint_sha256: str | None = None
    expected_checkpoint_step: int | None = None
    expected_checkpoint_variant: str | None = None
    export_preset: str | None = None
    max_samples: int = 64
    steps: int = 30
    cfg_scale: float = 2.0
    max_colors: int = 32
    alpha_threshold: float = 0.5
    device: str = "cpu"
    seed: int = 123
    noise_seed: int | None = None
    batch_size: int = 16
    dither: bool = False
    write_raw_rgba: bool = True
    write_hard_rgba: bool = True
    contact_sheet_labels: str = "prompt"
    project_palette: bool = False
    project_palette_target_colors: int = 16
    project_palette_min_pixel_share: float = 0.01
    project_palette_method: str = "deterministic_kmeans"
    # v2 Phase 0 diagnostics (no-training sampling-time knobs; off by default, see
    # docs/v2_phase0_diagnostics.md). factored_cfg replaces the scalar cfg_scale
    # guidance term with independent base/color guidance axes; null_fields zeroes
    # selected conditioning fields at sample time to probe which fields the model
    # actually uses.
    factored_cfg: bool = False
    cfg_base_scale: float | None = None
    cfg_color_scale: float | None = None
    null_fields: str = ""
    # v2 Phase 2 Exp A: sampling-only guidance surgery (default-off, see docs/...)
    color_guidance_rgb_only: bool = False
    color_guidance_start_t: float = 0.0
    color_guidance_ramp_t: float = 0.0
    object_id_scale: float = 1.0
    # v3 Phase 0. A trained palette-conditioned checkpoint can be sampled with
    # a real source palette, a retrieved palette, or an explicit null condition.
    palette_conditioning_source: str = "none"
    palette_conditioning_dataset: Path | None = None
    palette_conditioning_training_manifest: Path | None = None
    palette_conditioning_exclude_exact_prompt_target: bool = False
    allow_legacy_conditioning_v1: bool = True


class RectifiedFlowUNet(_ModuleBase):
    """Compact U-Net that predicts rectified-flow velocity for RGBA sprites."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int = 64,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        res_blocks_per_level: int = 2,
        pad_token_id: int = 0,
        structured_vocab_sizes: Mapping[str, int] | None = None,
        film_conditioning: bool = False,
        bottleneck_attention: bool = False,
        palette_conditioning: bool = False,
        palette_conditioning_dropout: float = 0.0,
        palette_conditioning_dim: int = 64,
        palette_conditioning_inject: str = "decoder",
        auxiliary_heads_mode: AuxiliaryHeadsMode | str | None = None,
    ) -> None:
        th, nn_mod = _require_torch()
        del th
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(value) for value in channel_mults)
        self.res_blocks_per_level = int(res_blocks_per_level)
        self.pad_token_id = int(pad_token_id)
        self.structured_vocab_sizes = _normalize_structured_vocab_sizes(structured_vocab_sizes)
        self.film_conditioning = bool(film_conditioning)
        self.bottleneck_attention = bool(bottleneck_attention)
        self.palette_conditioning = bool(palette_conditioning)
        self.palette_conditioning_dropout = float(palette_conditioning_dropout)
        self.palette_conditioning_dim = int(palette_conditioning_dim)
        self.palette_conditioning_inject = str(palette_conditioning_inject).strip().lower()
        self.auxiliary_heads_mode, self.legacy_auxiliary_heads_adapter = normalize_auxiliary_heads_mode(
            auxiliary_heads_mode
        )
        if self.palette_conditioning_inject not in {"decoder", "all", "bottleneck"}:
            raise ValueError("palette_conditioning_inject must be one of: decoder, all, bottleneck")

        channels = [max(8, self.base_channels * value) for value in self.channel_mults]
        emb_dim = max(self.embed_dim, self.base_channels * 4)
        self.token_embedding = nn_mod.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_token_id)
        self.time_mlp = nn_mod.Sequential(
            nn_mod.Linear(emb_dim, emb_dim),
            nn_mod.SiLU(),
            nn_mod.Linear(emb_dim, emb_dim),
        )
        structured_dim = self._init_structured_conditioning_modules(nn_mod)
        self.cond_mlp = nn_mod.Sequential(
            nn_mod.Linear(self.embed_dim * 2 + structured_dim, emb_dim),
            nn_mod.SiLU(),
            nn_mod.Linear(emb_dim, emb_dim),
        )
        if self.palette_conditioning:
            self.palette_condition_encoder = nn_mod.Sequential(
                nn_mod.Linear(PALETTE_CONDITION_CHANNELS, self.palette_conditioning_dim),
                nn_mod.SiLU(),
                nn_mod.Linear(self.palette_conditioning_dim, self.palette_conditioning_dim),
                nn_mod.SiLU(),
            )
            self.palette_condition_projection = nn_mod.Linear(self.palette_conditioning_dim, emb_dim)
        else:
            self.palette_condition_encoder = None
            self.palette_condition_projection = None
        self.input = nn_mod.Conv2d(4, channels[0], kernel_size=3, padding=1)
        self.downs = nn_mod.ModuleList()
        current = channels[0]
        for level, channel in enumerate(channels):
            blocks = nn_mod.ModuleList()
            for block_index in range(max(1, self.res_blocks_per_level)):
                blocks.append(
                    _ResidualBlock(
                        current if block_index == 0 else channel, channel, emb_dim, film=self.film_conditioning
                    )
                )
            down = (
                nn_mod.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
                if level < len(channels) - 1
                else None
            )
            self.downs.append(nn_mod.ModuleDict({"blocks": blocks, "down": down or nn_mod.Identity()}))
            current = channel
        self.mid = nn_mod.ModuleList(
            [
                _ResidualBlock(current, current, emb_dim, film=self.film_conditioning),
                _ResidualBlock(current, current, emb_dim, film=self.film_conditioning),
            ]
        )
        # v2 Phase 1: optional lightweight bottleneck self-attention (default-off)
        # Placed here so *current* still holds the bottleneck channel count.
        if self.bottleneck_attention:
            self.bottleneck_attn = _SelfAttentionBlock(current, num_heads=4)
        else:
            self.bottleneck_attn = None
        self.ups = nn_mod.ModuleList()
        for level in reversed(range(len(channels) - 1)):
            skip_channel = channels[level]
            blocks = nn_mod.ModuleList()
            blocks.append(_ResidualBlock(current + skip_channel, skip_channel, emb_dim, film=self.film_conditioning))
            for _ in range(max(0, self.res_blocks_per_level - 1)):
                blocks.append(_ResidualBlock(skip_channel, skip_channel, emb_dim, film=self.film_conditioning))
            self.ups.append(
                nn_mod.ModuleDict(
                    {
                        "up": nn_mod.Sequential(
                            nn_mod.Upsample(scale_factor=2, mode="nearest"),
                            nn_mod.Conv2d(current, current, kernel_size=3, padding=1),
                        ),
                        "blocks": blocks,
                    }
                )
            )
            current = skip_channel
        self.output = nn_mod.Sequential(
            _group_norm(current),
            nn_mod.SiLU(),
            nn_mod.Conv2d(current, 4, kernel_size=3, padding=1),
        )
        self._time_embedding_dim = emb_dim

        # Physical construction is selected only by auxiliary_heads_mode.
        # No placeholder modules or attributes are registered in absent mode.
        if self.auxiliary_heads_mode is AuxiliaryHeadsMode.PALETTE_INDEX:
            K = 16
            bottleneck_channels = channels[-1]
            self._bottleneck_pool = nn_mod.AdaptiveAvgPool2d(1)
            self.palette_head_rgb = nn_mod.Sequential(
                nn_mod.Linear(bottleneck_channels, 128),
                nn_mod.SiLU(),
                nn_mod.Linear(128, K * 3),
            )
            self.palette_head_presence = nn_mod.Sequential(
                nn_mod.Linear(bottleneck_channels, 128),
                nn_mod.SiLU(),
                nn_mod.Linear(128, K),
            )
            self.index_head = nn_mod.Conv2d(current, K, kernel_size=1)

    def config(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "base_channels": self.base_channels,
            "channel_mults": list(self.channel_mults),
            "res_blocks_per_level": self.res_blocks_per_level,
            "pad_token_id": self.pad_token_id,
            "structured_vocab_sizes": dict(self.structured_vocab_sizes) if self.structured_vocab_sizes else None,
            "film_conditioning": self.film_conditioning,
            "bottleneck_attention": self.bottleneck_attention,
            "palette_conditioning": self.palette_conditioning,
            "palette_conditioning_dropout": self.palette_conditioning_dropout,
            "palette_conditioning_dim": self.palette_conditioning_dim,
            "palette_conditioning_inject": self.palette_conditioning_inject,
            "auxiliary_heads_mode": self.auxiliary_heads_mode.value,
        }

    def forward(
        self,
        x: Any,
        t: Any,
        *,
        caption_tokens: Any,
        semantic_tokens: Any | None = None,
        structured_conditioning: Mapping[str, Any] | None = None,
        return_aux: bool = False,
        object_id_scale: float = 1.0,
        palette_condition: Any | None = None,
    ) -> Any:
        th, _nn_mod = _require_torch()
        emb = self._conditioning_embedding(
            t,
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            structured_conditioning=structured_conditioning,
            object_id_scale=object_id_scale,
        )
        h = self.input(x)
        palette_emb = self._palette_condition_embedding(palette_condition, batch=int(x.shape[0]), device=x.device)
        down_emb = emb if palette_emb is None or self.palette_conditioning_inject != "all" else emb + palette_emb
        mid_emb = (
            emb
            if palette_emb is None or self.palette_conditioning_inject not in {"all", "bottleneck"}
            else emb + palette_emb
        )
        decoder_emb = (
            emb
            if palette_emb is None or self.palette_conditioning_inject not in {"all", "decoder"}
            else emb + palette_emb
        )
        skips: list[Any] = []
        for level, down in enumerate(self.downs):
            for block in down["blocks"]:
                h = block(h, down_emb)
            if level < len(self.downs) - 1:
                skips.append(h)
                h = down["down"](h)
        # Bottleneck: save for palette head
        bottleneck_h = h
        for block in self.mid:
            h = block(h, mid_emb)
        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)
        for up in self.ups:
            h = up["up"](h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = th.nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = th.cat([h, skip], dim=1)
            for block in up["blocks"]:
                h = block(h, decoder_emb)
        # h is now the final feature map before output
        output_rgba = self.output(h)

        if not return_aux:
            return output_rgba

        result = {
            "schema_version": FORWARD_OUTPUT_SCHEMA_VERSION,
            "velocity": output_rgba,
            "auxiliary_heads_mode": self.auxiliary_heads_mode.value,
            "auxiliary_heads_available": self.auxiliary_heads_mode is AuxiliaryHeadsMode.PALETTE_INDEX,
            "palette_rgb": None,
            "palette_presence_logits": None,
            "index_logits": None,
        }
        if self.auxiliary_heads_mode is AuxiliaryHeadsMode.ABSENT:
            return result

        b_feat = self._bottleneck_pool(bottleneck_h).flatten(1)
        result["palette_rgb"] = self.palette_head_rgb(b_feat).view(-1, 16, 3)
        result["palette_presence_logits"] = self.palette_head_presence(b_feat)
        result["index_logits"] = self.index_head(h)
        return result

    def _palette_condition_embedding(self, palette_condition: Any | None, *, batch: int, device: Any) -> Any | None:
        """Encode slot features and apply per-example training dropout."""
        th, _nn_mod = _require_torch()
        if not self.palette_conditioning or self.palette_condition_encoder is None:
            return None
        if palette_condition is None:
            condition = th.zeros(batch, PALETTE_CONDITION_SLOTS, PALETTE_CONDITION_CHANNELS, device=device)
        else:
            condition = palette_condition.to(device=device, dtype=next(self.parameters()).dtype)
            if (
                condition.ndim != 3
                or condition.shape[0] != batch
                or condition.shape[1:] != (PALETTE_CONDITION_SLOTS, PALETTE_CONDITION_CHANNELS)
            ):
                raise ValueError(
                    "palette_condition must have shape "
                    f"[B, {PALETTE_CONDITION_SLOTS}, {PALETTE_CONDITION_CHANNELS}], got {tuple(condition.shape)}"
                )
        slot = self.palette_condition_encoder(condition)
        # Coverage is a stable pooling weight; present but tiny entries retain a
        # small contribution so an accent cannot disappear completely.
        weights = (condition[..., 4] + condition[..., 3] * 0.01).unsqueeze(-1)
        pooled = (slot * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0e-6)
        embedding = self.palette_condition_projection(pooled)
        if self.training and self.palette_conditioning_dropout > 0.0:
            mask = th.rand(batch, device=device) < min(1.0, max(0.0, self.palette_conditioning_dropout))
            embedding = embedding.masked_fill(mask[:, None], 0.0)
        return embedding

    def _conditioning_embedding(
        self,
        t: Any,
        *,
        caption_tokens: Any,
        semantic_tokens: Any | None,
        structured_conditioning: Mapping[str, Any] | None,
        object_id_scale: float = 1.0,
    ) -> Any:
        th, _nn_mod = _require_torch()
        batch = int(caption_tokens.shape[0])
        if semantic_tokens is None:
            semantic_cond = th.zeros(batch, self.embed_dim, device=caption_tokens.device)
        else:
            semantic_cond = self._mean_pool_tokens(semantic_tokens)
        caption_cond = self._mean_pool_tokens(caption_tokens)
        pieces = [caption_cond, semantic_cond]
        structured_cond = self._structured_embedding(
            structured_conditioning,
            batch=batch,
            device=caption_tokens.device,
            object_id_scale=object_id_scale,
        )
        if structured_cond is not None:
            pieces.append(structured_cond)
        cond = self.cond_mlp(th.cat(pieces, dim=1))
        time_emb = _sinusoidal_embedding(t.reshape(batch), self._time_embedding_dim).to(cond.device, cond.dtype)
        return self.time_mlp(time_emb) + cond

    def _mean_pool_tokens(self, tokens: Any) -> Any:
        _th, _nn_mod = _require_torch()
        token_ids = tokens.long().clamp(min=0, max=self.vocab_size - 1)
        embedded = self.token_embedding(token_ids)
        mask = token_ids.ne(self.pad_token_id).float().unsqueeze(-1)
        summed = (embedded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _init_structured_conditioning_modules(self, nn_mod: Any) -> int:
        if not self.structured_vocab_sizes:
            self.structured_id_embeddings = nn_mod.ModuleDict()
            self.structured_multi_hot_projections = nn_mod.ModuleDict()
            self.structured_status_embeddings = nn_mod.ModuleDict()
            return 0
        id_specs = {
            "category_id": "category_vocab_size",
            "object_id": "object_vocab_size",
            "base_object_id": "base_object_vocab_size",
            "primary_color_id": "color_vocab_size",
        }
        multi_specs = {
            "color_multi_hot": "color_vocab_size",
            "material_multi_hot": "material_vocab_size",
            "shape_multi_hot": "shape_vocab_size",
            "function_multi_hot": "function_vocab_size",
            "style_multi_hot": "style_vocab_size",
        }
        self.structured_id_embeddings = nn_mod.ModuleDict(
            {
                field: nn_mod.Embedding(
                    max(1, int(self.structured_vocab_sizes[size_key])),
                    self.embed_dim,
                    padding_idx=0,
                )
                for field, size_key in id_specs.items()
            }
        )
        self.structured_multi_hot_projections = nn_mod.ModuleDict(
            {
                field: nn_mod.Linear(max(1, int(self.structured_vocab_sizes[size_key])), self.embed_dim, bias=False)
                for field, size_key in multi_specs.items()
            }
        )
        status_size = int(self.structured_vocab_sizes.get("field_status_vocab_size") or 0)
        self.structured_status_embeddings = nn_mod.ModuleDict(
            {field: nn_mod.Embedding(status_size, self.embed_dim, padding_idx=0) for field in STATUS_FIELDS}
            if status_size > 1
            else {}
        )
        return self.embed_dim * (len(id_specs) + len(multi_specs) + len(self.structured_status_embeddings))

    def _structured_embedding(
        self,
        structured_conditioning: Mapping[str, Any] | None,
        *,
        batch: int,
        device: Any,
        object_id_scale: float = 1.0,
    ) -> Any | None:
        if not self.structured_vocab_sizes:
            return None
        th, _nn_mod = _require_torch()
        pieces: list[Any] = []
        for field, embedding in self.structured_id_embeddings.items():
            value = None if structured_conditioning is None else structured_conditioning.get(field)
            if value is None:
                ids = th.zeros(batch, dtype=th.long, device=device)
            else:
                ids = value.to(device=device, dtype=th.long).reshape(batch)
            emb = embedding(ids)
            if field == "object_id" and object_id_scale != 1.0:
                emb = emb * float(object_id_scale)
            pieces.append(emb)
        for field, projection in self.structured_multi_hot_projections.items():
            value = None if structured_conditioning is None else structured_conditioning.get(field)
            width = int(projection.in_features)
            if value is None:
                multi_hot = th.zeros(batch, width, dtype=next(projection.parameters()).dtype, device=device)
            else:
                multi_hot = value.to(device=device, dtype=next(projection.parameters()).dtype).reshape(batch, width)
            pieces.append(projection(multi_hot))
        for field, embedding in self.structured_status_embeddings.items():
            value = None if structured_conditioning is None else structured_conditioning.get(field)
            if value is None:
                ids = th.zeros(batch, dtype=th.long, device=device)
            else:
                ids = value.to(device=device, dtype=th.long).reshape(batch)
            pieces.append(embedding(ids))
        return th.cat(pieces, dim=1)


class _SelfAttentionBlock(_ModuleBase):
    """Lightweight multi-head self-attention for the U-Net bottleneck (v2 Phase 1)."""

    def __init__(self, channels: int, *, num_heads: int = 4) -> None:
        _th, nn_mod = _require_torch()
        super().__init__()
        self.norm = nn_mod.LayerNorm(channels)
        self.attn = nn_mod.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: Any) -> Any:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        residual = x
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return residual + x


class _ResidualBlock(_ModuleBase):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, *, film: bool = False) -> None:
        _th, nn_mod = _require_torch()
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn_mod.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb = nn_mod.Linear(emb_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn_mod.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn_mod.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn_mod.Identity()
        )
        self.act = nn_mod.SiLU()
        self._film = bool(film)
        if self._film:
            self.emb_scale1 = nn_mod.Linear(emb_dim, out_channels)
            self.emb_shift1 = nn_mod.Linear(emb_dim, out_channels)
            self.emb_scale2 = nn_mod.Linear(emb_dim, out_channels)
            self.emb_shift2 = nn_mod.Linear(emb_dim, out_channels)
            # Initialize scale near zero, shift comparable to additive path
            nn_mod.init.zeros_(self.emb_scale1.weight)
            nn_mod.init.zeros_(self.emb_scale2.weight)

    def forward(self, x: Any, emb: Any) -> Any:
        h = self.conv1(self.act(self.norm1(x)))
        if self._film:
            h = h * (1.0 + self.emb_scale1(emb).unsqueeze(-1).unsqueeze(-1)) + self.emb_shift1(emb).unsqueeze(
                -1
            ).unsqueeze(-1)
        else:
            h = h + self.emb(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        if self._film:
            h = h * (1.0 + self.emb_scale2(emb).unsqueeze(-1).unsqueeze(-1)) + self.emb_shift2(emb).unsqueeze(
                -1
            ).unsqueeze(-1)
        return h + self.skip(x)


class ChallengerPromptAdapter:
    """Adapter exposing challenger sampling through the regression diagnostic API."""

    def __init__(
        self,
        model: RectifiedFlowUNet,
        *,
        steps: int = 30,
        cfg_scale: float = 1.0,
        pad_token_id: int = 0,
        structured_vocab: StructuredConditioningVocab | None = None,
    ) -> None:
        self.model = model
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.pad_token_id = int(pad_token_id)
        self.structured_vocab = structured_vocab

    def eval(self) -> ChallengerPromptAdapter:
        self.model.eval()
        return self

    def sample_noise(self, batch_size: int, *, device: Any | None = None, seed: int | None = None) -> Any:
        th, _nn_mod = _require_torch()
        if device is None:
            device = next(self.model.parameters()).device
        generator = None
        if seed is not None:
            try:
                generator = th.Generator(device=device)
            except TypeError:  # pragma: no cover - older torch fallback.
                generator = th.Generator()
            generator.manual_seed(int(seed))
        return th.randn(int(batch_size), 4, SPRITE_SIZE, SPRITE_SIZE, device=device, generator=generator)

    def __call__(
        self,
        *,
        caption_tokens: Any,
        semantic_tokens: Any | None = None,
        structured_conditioning: Mapping[str, Any] | None = None,
        noise: Any | None = None,
    ) -> dict[str, Any]:
        if noise is None:
            noise = self.sample_noise(int(caption_tokens.shape[0]), device=caption_tokens.device)
        rgba = integrate_rectified_flow(
            self.model,
            noise,
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            structured_conditioning=structured_conditioning,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            pad_token_id=self.pad_token_id,
        )
        return _rgba_to_logit_outputs(rgba)


class _CampaignRunWriter(AbstractContextManager["_CampaignRunWriter"]):
    """Hold one exact run root and publish single-link final-name artifacts."""

    def __init__(self, logical_root: Path, physical_root: Path) -> None:
        self.logical_root = logical_root
        self.physical_root = physical_root
        self._anchor: AnchoredDirectory | None = None

    def __enter__(self) -> _CampaignRunWriter:
        self.physical_root.mkdir(parents=True, exist_ok=True)
        anchor = AnchoredDirectory(self.logical_root, self.logical_root)
        anchor.__enter__()
        try:
            physical = self.physical_root.stat()
            held = anchor.directory_metadata()
            if not stat.S_ISDIR(physical.st_mode) or not OwnedFileIdentity.from_stat(held).matches(physical):
                raise UnsafeFilesystemOperation("retained campaign output root identity changed")
            for name in anchor.names():
                metadata = anchor.lstat(name)
                reparse = bool(int(getattr(metadata, "st_file_attributes", 0) or 0) & 0x400)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                    or reparse
                    or metadata.st_nlink != 1
                ):
                    raise UnsafeFilesystemOperation(f"campaign output contains an unsafe entry: {name}")
        except BaseException as exc:
            anchor.__exit__(type(exc), exc, exc.__traceback__)
            raise
        self._anchor = anchor
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._anchor is not None:
            self._anchor.__exit__(exc_type, exc_value, traceback)
            self._anchor = None

    @property
    def anchor(self) -> AnchoredDirectory:
        if self._anchor is None:
            raise UnsafeFilesystemOperation("campaign output writer is not open")
        return self._anchor

    def names(self) -> tuple[str, ...]:
        return self.anchor.names()

    def lexists(self, name: str) -> bool:
        return self.anchor.lexists(name)

    def read_bytes(self, name: str, *, max_bytes: int = 128 * 1024 * 1024) -> bytes:
        descriptor = self._open_read(name)
        try:
            before = os.fstat(descriptor)
            if before.st_size > max_bytes:
                raise UnsafeFilesystemOperation(f"campaign artifact is oversized: {name}")
            content = bytearray()
            while len(content) <= max_bytes:
                chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(content)))
                if not chunk:
                    break
                content.extend(chunk)
            after = os.fstat(descriptor)
            identity = OwnedFileIdentity.from_stat(before)
            if (
                len(content) > max_bytes
                or not identity.matches(after)
                or after.st_nlink != 1
                or not identity.matches(self.anchor.lstat(name))
            ):
                raise UnsafeFilesystemOperation(f"campaign artifact changed while read: {name}")
            return bytes(content)
        finally:
            os.close(descriptor)

    def file_sha256(self, name: str) -> str:
        descriptor = self._open_read(name)
        try:
            before = os.fstat(descriptor)
            identity = OwnedFileIdentity.from_stat(before)
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            if not identity.matches(after) or after.st_nlink != 1 or not identity.matches(self.anchor.lstat(name)):
                raise UnsafeFilesystemOperation(f"campaign artifact changed while hashed: {name}")
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    def write_json_idempotent(self, name: str, value: Mapping[str, Any]) -> bytes:
        content = _canonical_pretty_json_bytes(value)
        try:
            self.write_bytes_exclusive(name, content)
        except FileExistsError as exc:
            if self.read_bytes(name, max_bytes=max(1024, len(content))) != content:
                raise UnsafeFilesystemOperation(
                    f"immutable campaign artifact conflicts with retained bytes: {name}"
                ) from exc
        return content

    def write_bytes_exclusive(self, name: str, content: bytes) -> str:
        descriptor, identity = self._open_exclusive(name)
        try:
            _write_descriptor_all(descriptor, content)
            os.fsync(descriptor)
            self._verify_written(name, descriptor, identity, content)
            self._sync_directory()
            self._verify_written(name, descriptor, identity, content)
            return hashlib.sha256(content).hexdigest()
        finally:
            os.close(descriptor)

    def write_torch_checkpoint(self, name: str, checkpoint: Mapping[str, Any], torch_module: Any) -> str:
        descriptor, identity = self._open_exclusive(name)
        try:
            with os.fdopen(os.dup(descriptor), "wb") as handle:
                torch_module.save(dict(checkpoint), handle)
                handle.flush()
                os.fsync(handle.fileno())
            content_sha256 = _descriptor_sha256(descriptor)
            self._verify_descriptor_identity(name, descriptor, identity)
            self._sync_directory()
            self._verify_descriptor_identity(name, descriptor, identity)
            if not hmac.compare_digest(content_sha256, _descriptor_sha256(descriptor)):
                raise UnsafeFilesystemOperation(f"campaign checkpoint changed after publication: {name}")
            return content_sha256
        finally:
            os.close(descriptor)

    def _open_exclusive(self, name: str) -> tuple[int, OwnedFileIdentity]:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0))
        descriptor = self.anchor.open_file_immovable(name, flags, 0o600)
        metadata = os.fstat(descriptor)
        identity = OwnedFileIdentity.from_stat(metadata)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size != 0:
            os.close(descriptor)
            raise UnsafeFilesystemOperation(f"new campaign artifact is unsafe: {name}")
        return descriptor, identity

    def _open_read(self, name: str) -> int:
        flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
        descriptor = self.anchor.open_file_immovable(name, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise UnsafeFilesystemOperation(f"campaign artifact is not a single-link regular file: {name}")
        return descriptor

    def _verify_written(
        self,
        name: str,
        descriptor: int,
        identity: OwnedFileIdentity,
        expected: bytes,
    ) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        actual = bytearray()
        while len(actual) <= len(expected):
            chunk = os.read(descriptor, min(1024 * 1024, len(expected) + 1 - len(actual)))
            if not chunk:
                break
            actual.extend(chunk)
        if bytes(actual) != expected:
            raise UnsafeFilesystemOperation(f"campaign artifact content changed during publication: {name}")
        self._verify_descriptor_identity(name, descriptor, identity)

    def _verify_descriptor_identity(
        self,
        name: str,
        descriptor: int,
        identity: OwnedFileIdentity,
    ) -> None:
        metadata = os.fstat(descriptor)
        if not identity.matches(metadata) or metadata.st_nlink != 1 or not identity.matches(self.anchor.lstat(name)):
            raise UnsafeFilesystemOperation(f"campaign artifact identity changed during publication: {name}")

    def _sync_directory(self) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(
            self.anchor.fixed_directory_path(),
            os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _canonical_pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(_jsonable(dict(value)), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_descriptor_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("campaign artifact write made no progress")
        offset += written


def _descriptor_sha256(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


_CAMPAIGN_RUN_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_identity",
        "run_id",
        "run_identity",
        "seed",
        "output_root",
        "resolved_config_sha256",
        "execution_contract_sha256",
        "expected_checkpoint_steps",
        "expected_evaluation_steps",
        "max_optimizer_steps",
        "schedule_name",
        "evaluation_ema_policy",
        "training_code_identity_sha256",
        "resolved_config",
    }
)
_CONCRETE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_campaign_run_contract(config: ChallengerTrainConfig) -> dict[str, Any] | None:
    value = config.campaign_run_contract
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _CAMPAIGN_RUN_CONTRACT_KEYS:
        raise ValueError("campaign run contract has an unsupported shape")
    contract = deepcopy(dict(value))
    if contract.get("schema_version") != CAMPAIGN_RUN_CONTRACT_SCHEMA_VERSION:
        raise ValueError("campaign run contract schema is unsupported")
    for field in (
        "campaign_identity",
        "run_identity",
        "resolved_config_sha256",
        "execution_contract_sha256",
        "training_code_identity_sha256",
    ):
        if not isinstance(contract.get(field), str) or _CONCRETE_SHA256.fullmatch(contract[field]) is None:
            raise ValueError(f"campaign run contract {field} is not a concrete SHA-256")
    for field in ("campaign_id", "run_id", "output_root", "schedule_name", "evaluation_ema_policy"):
        item = contract.get(field)
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(f"campaign run contract {field} is invalid")
    if type(contract.get("seed")) is not int or int(contract["seed"]) != int(config.seed):
        raise ValueError("campaign run contract seed does not match the trainer")
    if type(contract.get("max_optimizer_steps")) is not int or contract["max_optimizer_steps"] != int(config.max_steps):
        raise ValueError("campaign run contract max optimizer steps do not match the trainer")
    if Path(contract["output_root"]) != Path(config.out_dir):
        raise ValueError("campaign run contract logical output root does not match the trainer")
    if config.unsafe_resume:
        raise ValueError("campaign run contracts never permit unsafe resume")
    resolved_config = contract.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("campaign run contract resolved config is invalid")
    from spritelab.training.experiment_system import stable_hash

    if stable_hash(resolved_config) != contract["resolved_config_sha256"]:
        raise ValueError("campaign run contract resolved config identity changed")
    for field in ("expected_checkpoint_steps", "expected_evaluation_steps"):
        raw_steps = contract.get(field)
        if not isinstance(raw_steps, list) or any(type(step) is not int or step <= 0 for step in raw_steps):
            raise ValueError(f"campaign run contract {field} is invalid")
        if raw_steps != sorted(set(raw_steps)) or not raw_steps or raw_steps[-1] != int(config.max_steps):
            raise ValueError(f"campaign run contract {field} does not end at the fixed optimizer step")
    if contract["evaluation_ema_policy"] not in {"live", "ema", "both"}:
        raise ValueError("campaign run contract evaluation EMA policy is invalid")
    return contract


def _initialize_campaign_run(
    writer: _CampaignRunWriter,
    contract: Mapping[str, Any],
) -> None:
    from spritelab.product_web.events import (
        EVENT_FILENAME,
        EVENT_HISTORY_ORIGIN_FILENAME,
        _build_event_history_origin,
        _event_history_origin_bindings,
        verify_event_migration,
    )

    identity_name = "run_identity.json"
    if writer.lexists(identity_name):
        identity = strict_json_loads(writer.read_bytes(identity_name, max_bytes=1024 * 1024))
        if not isinstance(identity, Mapping):
            raise UnsafeFilesystemOperation("retained campaign run identity is malformed")
        for field in (
            "campaign_id",
            "campaign_identity",
            "run_id",
            "run_identity",
            "seed",
            "output_root",
            "resolved_config_sha256",
            "execution_contract_sha256",
        ):
            if identity.get(field) != contract.get(field):
                raise UnsafeFilesystemOperation(f"retained campaign run identity changed: {field}")
        migration = verify_event_migration(
            str(contract["run_id"]),
            writer.physical_root,
            origin_required=True,
        )
        if not migration.resume_compatible:
            raise UnsafeFilesystemOperation("retained campaign event history is not resume compatible")
        return
    if writer.names():
        raise UnsafeFilesystemOperation("nonempty campaign output root has no immutable run identity")
    writer.write_bytes_exclusive(EVENT_FILENAME, b"")
    origin = _build_event_history_origin(
        str(contract["run_id"]),
        writer.physical_root,
        migration_record=None,
    )
    origin_content = writer.write_json_idempotent(EVENT_HISTORY_ORIGIN_FILENAME, origin)
    bindings = _event_history_origin_bindings(
        origin,
        hashlib.sha256(origin_content).hexdigest(),
        hashlib.sha256(b"").hexdigest(),
    )
    identity = {
        "campaign_id": contract["campaign_id"],
        "campaign_identity": contract["campaign_identity"],
        "run_id": contract["run_id"],
        "run_identity": contract["run_identity"],
        "seed": contract["seed"],
        "output_root": contract["output_root"],
        "resolved_config_sha256": contract["resolved_config_sha256"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        **bindings,
    }
    writer.write_json_idempotent(identity_name, identity)
    migration = verify_event_migration(
        str(contract["run_id"]),
        writer.physical_root,
        origin_required=True,
    )
    if not migration.resume_compatible:
        raise UnsafeFilesystemOperation("new campaign event history is not resume compatible")


def run_challenger_training(config: ChallengerTrainConfig) -> dict[str, Any]:
    """Train one challenger, including ``create_unsafe_resume_revocation`` handling."""

    contract = _validate_campaign_run_contract(config)
    if contract is None:
        return _run_challenger_training_impl(config, campaign_writer=None, campaign_contract=None)
    if config.resume_from is not None and (
        not isinstance(config.expected_resume_sha256, str)
        or _CONCRETE_SHA256.fullmatch(config.expected_resume_sha256) is None
    ):
        raise ValueError("campaign resume requires the exact retained checkpoint SHA-256")
    logical_root = Path(config.out_dir)
    physical_root = Path(config.retained_output_root) if config.retained_output_root is not None else logical_root
    with _CampaignRunWriter(logical_root, physical_root) as writer:
        return _run_challenger_training_impl(config, campaign_writer=writer, campaign_contract=contract)


def _run_challenger_training_impl(
    config: ChallengerTrainConfig,
    *,
    campaign_writer: _CampaignRunWriter | None,
    campaign_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config.resume_from is None and (
        config.resume_descriptor is not None or config.expected_resume_sha256 is not None
    ):
        raise ValueError("retained resume inputs require an exact --resume checkpoint")
    if config.resume_descriptor is not None and (
        type(config.resume_descriptor) is not int
        or config.resume_descriptor < 0
        or not isinstance(config.expected_resume_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", config.expected_resume_sha256) is None
    ):
        raise ValueError("retained resume descriptor requires an exact checkpoint SHA-256")
    if config.unsafe_resume:
        if config.resume_from is None:
            raise ValueError("--unsafe-resume requires --resume")
        if not isinstance(config.unsafe_resume_reason, str) or not config.unsafe_resume_reason.strip():
            raise ValueError("--unsafe-resume requires a nonempty explicit --unsafe-resume-reason")
    th, _nn_mod = _require_torch()
    auxiliary_mode, legacy_auxiliary_adapter = resolve_auxiliary_heads_mode(
        config.auxiliary_heads_mode, config.experiment_manifest
    )
    auxiliary_weights = {
        "index_head_loss_weight": float(config.index_head_loss_weight),
        "palette_head_loss_weight": float(config.palette_head_loss_weight),
        "palette_presence_loss_weight": float(config.palette_presence_loss_weight),
    }
    nonzero_auxiliary_weights = [name for name, value in auxiliary_weights.items() if value != 0.0]
    if auxiliary_mode is AuxiliaryHeadsMode.ABSENT and nonzero_auxiliary_weights:
        raise ValueError(
            "auxiliary_heads_mode='absent' is incompatible with nonzero auxiliary losses: "
            + ", ".join(nonzero_auxiliary_weights)
        )
    if str(config.architecture).lower() != "rectified_flow":
        raise ValueError("only --architecture rectified_flow is supported")
    metrics_every = int(config.metrics_every)
    if metrics_every < 1:
        raise ValueError("metrics_every must be >= 1")
    if not 0.0 <= float(config.palette_conditioning_dropout) <= 1.0:
        raise ValueError("palette_conditioning_dropout must be in [0, 1]")
    if int(config.palette_conditioning_dim) < 1:
        raise ValueError("palette_conditioning_dim must be positive")
    timestep_boundaries = validate_timestep_boundaries(config.timestep_validation_boundaries)
    started = time.perf_counter()
    _set_seed(config.seed)
    global _palette_scale_needs_normalize
    _palette_scale_needs_normalize = None  # reset per-run palette scale cache
    device = resolve_device(config.device)
    from spritelab.training.determinism import configure_determinism

    determinism_report = configure_determinism(config.determinism, device=device, torch_module=th)
    if determinism_report["mode"] != "off" and (config.cudnn_benchmark or config.tf32):
        raise ValueError("determinism warn/strict is incompatible with cudnn_benchmark or tf32")
    if int(config.gradient_accumulation_steps) != 1:
        raise ValueError(
            "gradient_accumulation_steps other than 1 are not yet supported for exact resume; "
            "gradient tensors would need checkpointing"
        )
    logical_out_dir = Path(config.out_dir)
    out_dir = Path(config.retained_output_root) if config.retained_output_root is not None else logical_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if campaign_writer is not None and campaign_contract is not None:
        _initialize_campaign_run(campaign_writer, campaign_contract)
    conditioning_mode = validate_conditioning_mode(config.conditioning_mode)

    retained_inputs = (
        config.retained_training_manifest_records,
        config.retained_dataset_descriptors,
        config.retained_dataset_content_sha256,
    )
    if all(value is None for value in retained_inputs):
        manifest_rows = read_jsonl(config.training_manifest)
    elif any(value is None for value in retained_inputs):
        raise ValueError("retained training manifest records, dataset descriptors, and hashes are inseparable")
    else:
        retained_records = config.retained_training_manifest_records
        retained_descriptors = config.retained_dataset_descriptors
        retained_hashes = config.retained_dataset_content_sha256
        assert retained_records is not None
        assert retained_descriptors is not None
        assert retained_hashes is not None
        if (
            isinstance(retained_records, (str, bytes))
            or not isinstance(retained_records, Sequence)
            or not retained_records
            or any(not isinstance(record, Mapping) for record in retained_records)
        ):
            raise ValueError("retained training manifest records are malformed")
        if (
            not isinstance(retained_descriptors, Mapping)
            or not isinstance(retained_hashes, Mapping)
            or not retained_descriptors
            or set(retained_descriptors) != set(retained_hashes)
        ):
            raise ValueError("retained dataset descriptor and hash inventories differ")
        manifest_rows = [dict(record) for record in retained_records]
    token_rows = [row for row in manifest_rows if _matches_caption_policy(row, config.caption_policy_filter)]
    effective_split = str(config.overfit_split or config.split)
    subset_selection = _select_training_subset(token_rows, config, split=effective_split)
    selected_sprite_ids = None if subset_selection is None else subset_selection.sprite_ids
    train_rows = (
        list(subset_selection.rows)
        if subset_selection is not None
        else [row for row in token_rows if row.get("split") == effective_split]
    )
    tokenizer = SpriteTextTokenizer.build_from_records(
        train_rows or token_rows or manifest_rows, max_length=config.caption_max_length
    )
    tokenizer.save(out_dir / "vocab.json")
    preloaded_resume = (
        _load_checkpoint(
            config.resume_from,
            expected_sha256=config.expected_resume_sha256,
            retained_descriptor=config.resume_descriptor,
        )
        if config.resume_from is not None
        else None
    )
    structured_vocab = None
    if uses_structured_conditioning(conditioning_mode):
        if config.unsafe_resume and isinstance(preloaded_resume, Mapping):
            structured_vocab = structured_vocab_from_checkpoint(preloaded_resume, allow_schema_v1_adapter=True)
        if structured_vocab is None:
            structured_vocab = build_structured_conditioning_vocab(
                [row for row in token_rows if row.get("split") == effective_split] or train_rows or token_rows
            )
    if structured_vocab is not None:
        save_structured_conditioning_vocab(structured_vocab, out_dir / "structured_conditioning_vocab.json")

    palette_swap = PaletteSwapConfig.from_training_config(config)
    role_ramp_transplant = RoleRampTransplantConfig.from_training_config(config)
    shared_npz_cache: dict[str, Any] = {}
    train_dataset = SpriteTrainingDataset(
        config.dataset_dir,
        config.training_manifest,
        split=effective_split,
        max_records=config.max_records,
        tokenizer=tokenizer,
        caption_max_length=config.caption_max_length,
        semantic_max_length=config.semantic_max_length,
        caption_policy_filter=config.caption_policy_filter,
        sprite_ids=selected_sprite_ids,
        structured_vocab=structured_vocab,
        npz_cache=shared_npz_cache,
        palette_swap=palette_swap,
        role_ramp_transplant=role_ramp_transplant,
        palette_conditioning=bool(config.palette_conditioning),
        retained_records=config.retained_training_manifest_records,
        retained_npz_descriptors=config.retained_dataset_descriptors,
        retained_npz_sha256=config.retained_dataset_content_sha256,
    )
    palette_swap_summary = {**palette_swap.report_dict()}
    if palette_swap.active():
        from spritelab.training.palette_swap_review import summarize_dataset_palette_swap

        palette_swap_summary.update(
            summarize_dataset_palette_swap(
                config.dataset_dir,
                [dict(record) for record in train_dataset.records],
                palette_swap,
                npz_cache=shared_npz_cache,
            )
        )
    else:
        palette_swap_summary.update(estimate_applied([dict(record) for record in train_dataset.records], palette_swap))
    role_ramp_summary = {**role_ramp_transplant.report_dict()}
    if train_dataset.role_ramp_library is not None:
        role_ramp_summary.update(train_dataset.role_ramp_library.report_dict())
    if len(train_dataset) == 0:
        raise ValueError(f"training manifest has no records for split {effective_split!r}")

    validation_mode = _resolve_validation_mode(config.validation_mode, subset_selection)
    if validation_mode == "same":
        val_source: Any = train_dataset
    elif validation_mode == "none":
        val_source = []
    else:
        val_source = SpriteTrainingDataset(
            config.dataset_dir,
            config.training_manifest,
            split="val",
            tokenizer=tokenizer,
            caption_max_length=config.caption_max_length,
            semantic_max_length=config.semantic_max_length,
            caption_policy_filter=config.caption_policy_filter,
            structured_vocab=structured_vocab,
            npz_cache=shared_npz_cache,
            palette_conditioning=bool(config.palette_conditioning),
            retained_records=config.retained_training_manifest_records,
            retained_npz_descriptors=config.retained_dataset_descriptors,
            retained_npz_sha256=config.retained_dataset_content_sha256,
        )

    dataset_label_quality = dataset_uncertainty_report(token_rows)
    train_label_quality = training_uncertainty_report(train_dataset.records)
    val_label_quality = evaluation_uncertainty_report(getattr(val_source, "records", []))
    label_quality_correlations = uncertainty_correlation_report(token_rows)

    sampler_generator = th.Generator().manual_seed(config.seed)
    loader_generator = th.Generator().manual_seed(config.seed + 1)
    train_sampler = StatefulPermutationSampler(train_dataset, generator=sampler_generator)
    loader_perf = dataloader_perf_kwargs(device, num_workers=config.num_workers)
    train_loader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        generator=loader_generator,
        collate_fn=collate_sprite_batch,
        **loader_perf,
    )
    eval_train_loader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_sprite_batch,
        **loader_perf,
    )
    val_loader = th.utils.data.DataLoader(
        val_source,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_sprite_batch,
        **loader_perf,
    )

    model_config = {
        "vocab_size": len(tokenizer),
        "embed_dim": int(config.embed_dim),
        "base_channels": int(config.base_channels),
        "channel_mults": _parse_channel_mults(config.channel_mults),
        "res_blocks_per_level": int(config.res_blocks_per_level),
        "pad_token_id": tokenizer.pad_id,
        "structured_vocab_sizes": None if structured_vocab is None else structured_vocab.sizes(),
        "palette_conditioning": bool(config.palette_conditioning),
        "palette_conditioning_dropout": float(config.palette_conditioning_dropout),
        "palette_conditioning_dim": int(config.palette_conditioning_dim),
        "palette_conditioning_inject": str(config.palette_conditioning_inject),
    }
    model = RectifiedFlowUNet(
        **model_config,
        film_conditioning=config.film_conditioning,
        bottleneck_attention=config.bottleneck_attention,
        auxiliary_heads_mode=None if legacy_auxiliary_adapter else auxiliary_mode,
    ).to(device)
    model_config = model.config()
    optimizer = build_adamw(model.parameters(), lr=float(config.learning_rate), fused=config.fused_adamw)
    ema_enabled = float(config.ema_decay) > 0.0
    ema_state = _init_ema_state(model) if ema_enabled else None
    ema_fast_cache = _init_ema_fast_state(ema_state, model) if ema_enabled else None
    from spritelab.training.experiment_system import (
        EXPERIMENT_MANIFEST_VERSION,
        RESUME_HARD_FIELDS,
        conditioning_schema,
        file_sha256,
        measured_architecture_identity,
        stable_hash,
    )

    effective_experiment_manifest = (
        None if config.experiment_manifest is None else deepcopy(dict(config.experiment_manifest))
    )
    if effective_experiment_manifest is not None:
        optimizer_identity = {
            "name": "adamw",
            "learning_rate": float(config.learning_rate),
            "schedule": str(config.lr_schedule),
            "warmup_steps": int(config.lr_warmup_steps),
        }
        loss_identity = {
            "velocity": "masked_mse",
            "palette_head_loss_weight": float(config.palette_head_loss_weight),
            "palette_presence_loss_weight": float(config.palette_presence_loss_weight),
            "index_head_loss_weight": float(config.index_head_loss_weight),
        }
        sampling_identity = {"name": "stateful_permutation_v1", "shuffle": True}
        actual_conditioning = conditioning_schema(
            mode=conditioning_mode,
            tokenizer=tokenizer.to_json_dict(),
            structured_vocab=None if structured_vocab is None else structured_vocab.to_json_dict(),
        )
        manifest_path = Path(config.training_manifest)
        if config.retained_training_manifest_records is None:
            manifest_content_sha256 = file_sha256(manifest_path)
        else:
            retained_manifest_hashes = {
                str(effective_experiment_manifest.get(field) or "")
                for field in ("dataset_manifest_hash", "split_manifest_hash")
            }
            if len(retained_manifest_hashes) != 1:
                raise ValueError("retained training manifest identities are missing or inconsistent")
            manifest_content_sha256 = retained_manifest_hashes.pop()
            if _CONCRETE_SHA256.fullmatch(manifest_content_sha256) is None:
                raise ValueError("retained training manifest identity is not a concrete SHA-256")
        effective_experiment_manifest.update(
            {
                "manifest_version": EXPERIMENT_MANIFEST_VERSION,
                "dataset_manifest": str(manifest_path),
                "dataset_manifest_hash": manifest_content_sha256,
                "split_manifest": str(manifest_path),
                "split_manifest_hash": manifest_content_sha256,
                "conditioning_schema": actual_conditioning,
                "conditioning_schema_hash": actual_conditioning["hash"],
                "optimizer_configuration": optimizer_identity,
                "optimizer_identity_hash": stable_hash(optimizer_identity),
                "schedule_identity_hash": stable_hash(
                    {"schedule": str(config.lr_schedule), "warmup_steps": int(config.lr_warmup_steps)}
                ),
                "loss_configuration": loss_identity,
                "loss_configuration_hash": stable_hash(loss_identity),
                "micro_batch_size": int(config.batch_size),
                "global_batch_size": int(config.batch_size),
                "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
                "effective_batch_size": int(config.batch_size) * int(config.gradient_accumulation_steps),
                "precision_policy": "amp" if config.amp else "fp32",
                "autocast_policy": {"enabled": bool(config.amp)},
                "ema": {"enabled": ema_enabled, "decay": float(config.ema_decay)},
                "ema_enabled": ema_enabled,
                "ema_decay": float(config.ema_decay),
                "ema_identity_hash": stable_hash({"enabled": ema_enabled, "decay": float(config.ema_decay)}),
                "random_seeds": {"training": int(config.seed)},
                "seed_identity_hash": stable_hash({"training": int(config.seed)}),
                "sampler_policy": sampling_identity,
                "sampler_policy_hash": stable_hash(sampling_identity),
                "determinism_policy": str(config.determinism),
                "determinism_policy_hash": stable_hash(str(config.determinism)),
                "evaluation_cadence": int(config.sample_every),
                "checkpoint_cadence": int(config.save_every),
                "max_optimizer_steps": int(config.max_steps),
                "lineage_parent_identity": "root",
            }
        )
    if effective_experiment_manifest is not None:
        measured_identity = measured_architecture_identity(model)
        effective_experiment_manifest["model_architecture"] = measured_identity
        effective_experiment_manifest["model_architecture_hash"] = measured_identity["hash"]
        effective_experiment_manifest["auxiliary_heads_mode"] = auxiliary_mode.value
        effective_experiment_manifest["legacy_architecture"] = legacy_auxiliary_adapter
        effective_experiment_manifest["fair_architecture_comparison_eligible"] = not legacy_auxiliary_adapter
        effective_experiment_manifest["checkpoint_promotion_eligible"] = not legacy_auxiliary_adapter
        effective_experiment_manifest["experiment_configuration_hash"] = stable_hash(
            {
                field: effective_experiment_manifest.get(field)
                for field in RESUME_HARD_FIELDS
                if field != "experiment_configuration_hash"
            }
        )
        effective_experiment_manifest["experiment_hash"] = stable_hash(
            {key: value for key, value in effective_experiment_manifest.items() if key != "experiment_hash"}
        )
    apply_backend_speed_flags(cudnn_benchmark=config.cudnn_benchmark, tf32=config.tf32)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=config.lr_schedule,
        max_steps=config.max_steps,
        warmup_steps=config.lr_warmup_steps,
    )
    non_blocking = device_type(device) == "cuda"
    unsafe_resume_record: dict[str, Any] = {}
    persisted_config = asdict(config)
    persisted_config.pop("retained_output_root", None)
    persisted_config.pop("retained_training_manifest_records", None)
    persisted_config.pop("retained_dataset_descriptors", None)
    persisted_config.pop("retained_dataset_content_sha256", None)
    persisted_config.pop("resume_descriptor", None)
    persisted_config.pop("campaign_run_contract", None)
    config_json = {
        **{key: _jsonable(value) for key, value in persisted_config.items()},
        "architecture": "rectified_flow",
        "model_type": "generator_challenger",
        "conditioning_mode": conditioning_mode,
        "model_config": model_config,
        "auxiliary_heads_mode": auxiliary_mode.value,
        "legacy_auxiliary_heads_adapter": legacy_auxiliary_adapter,
        "experiment_manifest": effective_experiment_manifest,
        "unsafe_resume_record": unsafe_resume_record,
        "train_records": len(train_dataset),
        "val_records": len(val_source),
        "validation_mode": validation_mode,
        "overfit_subset": None if subset_selection is None else subset_selection.to_report(),
        "structured_conditioning_vocab": None if structured_vocab is None else structured_vocab.to_json_dict(),
        "structured_vocab_sizes": None if structured_vocab is None else structured_vocab.sizes(),
        "structured_fields_enabled": structured_vocab is not None,
        "structured_field_dropout": float(config.structured_field_dropout),
        "structured_field_dropout_rates": _jsonable(config.structured_field_dropout_rates),
        "film_conditioning": bool(config.film_conditioning),
        "bottleneck_attention": bool(config.bottleneck_attention),
        "index_head_loss_weight": float(config.index_head_loss_weight),
        "palette_head_loss_weight": float(config.palette_head_loss_weight),
        "palette_presence_loss_weight": float(config.palette_presence_loss_weight),
        "index_head_warmup_steps": int(config.index_head_warmup_steps),
        "palette_head_use_gt_palette_prob": float(config.palette_head_use_gt_palette_prob),
        "ema_enabled": ema_enabled,
        "ema_decay": float(config.ema_decay),
        "foreground_rgb_loss_weight": float(config.foreground_rgb_loss_weight),
        "background_rgb_loss_weight": float(config.background_rgb_loss_weight),
        "palette_loss_weight": float(config.palette_loss_weight),
        "palette_loss_temperature": float(config.palette_loss_temperature),
        "palette_swap": palette_swap_summary,
        "role_ramp_transplant": role_ramp_summary,
        "conditioning_schema_version": (
            "spritelab_conditioning_v1"
            if structured_vocab is not None and structured_vocab.schema_version.endswith("_v1")
            else "spritelab_conditioning_v2"
        ),
        "determinism": determinism_report,
        "label_quality": {
            "dataset": dataset_label_quality,
            "training": train_label_quality,
            "evaluation": val_label_quality,
            "correlation_analysis": label_quality_correlations,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    initial_train_losses = _evaluate_challenger_losses(
        model,
        eval_train_loader,
        device=device,
        conditioning_mode=conditioning_mode,
        pad_token_id=tokenizer.pad_id,
        seed=config.seed,
        foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
        background_rgb_loss_weight=config.background_rgb_loss_weight,
        palette_loss_weight=config.palette_loss_weight,
        palette_loss_temperature=config.palette_loss_temperature,
        max_batches=config.eval_max_batches,
        timestep_boundaries=timestep_boundaries,
    )
    metrics_path = out_dir / "train_metrics.jsonl"
    if config.resume_from is None:
        metrics_path.write_text("", encoding="utf-8")
    preview_batch_cpu = next(iter(eval_train_loader))
    preview_batch = move_batch_to_device(preview_batch_cpu, device)

    step = 0
    last_loss = float(initial_train_losses["loss"])
    last_loss_components: dict[str, float] = {}
    checkpoint_steps = (
        tuple(int(step) for step in campaign_contract["expected_checkpoint_steps"])
        if campaign_contract is not None
        else _normalize_checkpoint_steps(config.checkpoint_steps, max_steps=config.max_steps)
    )
    run_limit = min(config.max_steps, int(config.stop_after_step or config.max_steps))
    checkpoint_step_paths: list[str] = []
    checkpoint_step_ema_paths: list[str] = []
    resume_mismatches: list[str] = []
    run_warnings: list[str] = []
    invalid_batch_count = 0
    skipped_batch_reasons: dict[str, int] = {}
    resumed_sampler_state: Mapping[str, Any] | None = None
    if config.num_workers > 0:
        message = "num_workers > 0 uses DataLoader prefetch; saved sample cursors are not exact-resume qualified"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        run_warnings.append(message)
    if config.resume_from is not None:
        from spritelab.training.experiment_system import (
            create_unsafe_resume_revocation,
            file_sha256,
            restore_rng_state,
            validate_resume_compatibility,
            write_unsafe_resume_revocation,
        )

        resume_checkpoint = preloaded_resume
        assert isinstance(resume_checkpoint, Mapping)
        saved_manifest = resume_checkpoint.get("experiment_manifest")
        checkpoint_state_mismatches: list[str] = []
        for field in ("model_state_dict", "optimizer_state_dict"):
            if not isinstance(resume_checkpoint.get(field), Mapping):
                checkpoint_state_mismatches.append(f"checkpoint_state.{field}.missing")
        if scheduler is not None and not isinstance(resume_checkpoint.get("scheduler_state_dict"), Mapping):
            checkpoint_state_mismatches.append("checkpoint_state.scheduler_state_dict.missing")
        if ema_state is not None and not isinstance(resume_checkpoint.get("ema_state_dict"), Mapping):
            checkpoint_state_mismatches.append("checkpoint_state.ema_state_dict.missing")
        if not isinstance(resume_checkpoint.get("rng_states"), Mapping):
            checkpoint_state_mismatches.append("checkpoint_state.rng_states.missing")
        saved_step = resume_checkpoint.get("global_step", resume_checkpoint.get("step"))
        if type(saved_step) is not int or not 0 <= saved_step <= int(config.max_steps):
            checkpoint_state_mismatches.append("checkpoint_state.optimizer_step.invalid")
        candidate_sampler_state = resume_checkpoint.get("sampler_state")
        sampler_mismatches = inspect_sampler_resume_state(
            candidate_sampler_state,
            dataset_size=len(train_dataset),
            batch_size=int(config.batch_size),
            num_workers=int(config.num_workers),
        )
        checkpoint_state_mismatches.extend(sampler_mismatches)
        if not isinstance(effective_experiment_manifest, Mapping):
            raise RuntimeError("resume requires a target runtime experiment manifest")
        if not isinstance(saved_manifest, Mapping):
            if not config.unsafe_resume:
                raise RuntimeError("safe resume requires experiment manifests in both config and checkpoint")
            resume_mismatches = ["experiment_manifest", *checkpoint_state_mismatches]
            unsafe_resume_record.update(
                create_unsafe_resume_revocation(
                    reason=config.unsafe_resume_reason,
                    mismatches=resume_mismatches,
                    source_checkpoint_identity=file_sha256(config.resume_from),
                    target_runtime_identity=effective_experiment_manifest.get("experiment_hash"),
                )
            )
        else:
            resume_mismatches = validate_resume_compatibility(
                effective_experiment_manifest,
                saved_manifest,
                unsafe=config.unsafe_resume,
                unsafe_reason=config.unsafe_resume_reason,
                unsafe_record=unsafe_resume_record if config.unsafe_resume else None,
                additional_mismatches=checkpoint_state_mismatches,
            )
        if config.unsafe_resume:
            write_unsafe_resume_revocation(out_dir, unsafe_resume_record)
        unloadable = {
            "checkpoint_state.model_state_dict.missing",
            "checkpoint_state.optimizer_state_dict.missing",
        }.intersection(checkpoint_state_mismatches)
        if unloadable:
            raise UnsupportedExactResumeError(
                "checkpoint cannot be loaded even as unsafe resume: " + ", ".join(sorted(unloadable))
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if ema_state is not None and isinstance(resume_checkpoint.get("ema_state_dict"), Mapping):
            ema_state.clear()
            ema_state.update({key: value.clone() for key, value in resume_checkpoint["ema_state_dict"].items()})
            ema_fast_cache = _init_ema_fast_state(ema_state, model)
        step = int(resume_checkpoint.get("global_step", resume_checkpoint.get("step", 0)))
        resumed_sampler_state = candidate_sampler_state if not sampler_mismatches else None
        if not isinstance(resumed_sampler_state, Mapping):
            message = "checkpoint sampler state is not exact-resume compatible; data order will restart"
            if not config.unsafe_resume:  # pragma: no cover - compatibility validation rejects first.
                raise UnsupportedExactResumeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            run_warnings.append(message)
        else:
            validate_worker_mode(
                num_workers=config.num_workers,
                exact_resume=True,
                unsafe=config.unsafe_resume,
            )
            train_sampler.load_state_dict(
                resumed_sampler_state,
                batch_size=config.batch_size,
                loader_generator=loader_generator,
                num_workers=config.num_workers,
                unsafe=config.unsafe_resume,
            )
            invalid_batch_count = int(resumed_sampler_state.get("invalid_batch_count", 0))
            skipped_batch_reasons = {
                str(key): int(value)
                for key, value in dict(resumed_sampler_state.get("skipped_batch_reasons", {})).items()
            }
        if isinstance(resume_checkpoint.get("rng_states"), Mapping):
            restore_rng_state(resume_checkpoint["rng_states"])

    def current_sampler_state() -> dict[str, Any]:
        return train_sampler.state_dict(
            batch_size=config.batch_size,
            loader_generator=loader_generator,
            accumulation_position=0,
            invalid_batch_count=invalid_batch_count,
            skipped_batch_reasons=skipped_batch_reasons,
            num_workers=config.num_workers,
            worker_seed_base=config.seed + 1,
        )

    campaign_evaluations = (
        _load_campaign_evaluations(campaign_writer, campaign_contract)
        if campaign_writer is not None and campaign_contract is not None
        else {}
    )

    def evaluate_campaign_step(optimizer_step: int) -> None:
        if campaign_writer is None or campaign_contract is None or optimizer_step in campaign_evaluations:
            return
        if len(val_source) == 0:
            raise ValueError("campaign completion requires a nonempty validation split")
        policy = str(campaign_contract["evaluation_ema_policy"])
        weights: dict[str, Any] = {}
        if policy in {"live", "both"}:
            weights["live"] = _evaluate_challenger_losses(
                model,
                val_loader,
                device=device,
                conditioning_mode=conditioning_mode,
                pad_token_id=tokenizer.pad_id,
                seed=config.seed + 2 + optimizer_step,
                foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
                background_rgb_loss_weight=config.background_rgb_loss_weight,
                palette_loss_weight=config.palette_loss_weight,
                palette_loss_temperature=config.palette_loss_temperature,
                max_batches=config.eval_max_batches,
                timestep_boundaries=timestep_boundaries,
            )
        if policy in {"ema", "both"}:
            if ema_state is None:
                raise ValueError("campaign completion requires retained EMA state")
            live_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            model.load_state_dict(ema_state)
            try:
                weights["ema"] = _evaluate_challenger_losses(
                    model,
                    val_loader,
                    device=device,
                    conditioning_mode=conditioning_mode,
                    pad_token_id=tokenizer.pad_id,
                    seed=config.seed + 2 + optimizer_step,
                    foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
                    background_rgb_loss_weight=config.background_rgb_loss_weight,
                    palette_loss_weight=config.palette_loss_weight,
                    palette_loss_temperature=config.palette_loss_temperature,
                    max_batches=config.eval_max_batches,
                    timestep_boundaries=timestep_boundaries,
                )
            finally:
                model.load_state_dict(live_state)
        evaluation = _campaign_evaluation_record(
            campaign_contract,
            optimizer_step=optimizer_step,
            metrics_by_weight=weights,
        )
        campaign_writer.write_json_idempotent(
            f"evaluation_step_{optimizer_step:06d}.json",
            evaluation,
        )
        campaign_evaluations[optimizer_step] = evaluation

    model.train()
    progress = StepProgressBar(config.max_steps, desc=f"challenger:{out_dir.name}")
    # Keep the metrics file open for the whole run instead of reopening it every
    # step; the line content and order are unchanged, so the file is identical.
    metrics_handle = metrics_path.open("a", encoding="utf-8")
    try:
        train_iterator = iter(train_loader)
        if isinstance(resumed_sampler_state, Mapping) and int(resumed_sampler_state["sample_cursor"]) < int(
            resumed_sampler_state["dataset_size"]
        ):
            # Iterator construction consumes a DataLoader base seed.  A resumed
            # mid-epoch iterator is not a new epoch, so restore the persisted
            # post-construction state to keep loader-generator state exact.
            loader_generator.set_state(resumed_sampler_state["dataloader_generator_state"])
        while step < run_limit:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            if step >= run_limit:
                break
            batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
            with amp_autocast(device, config.amp):
                losses = rectified_flow_loss(
                    model,
                    batch,
                    conditioning_mode=conditioning_mode,
                    cfg_dropout=config.cfg_dropout,
                    structured_field_dropout=config.structured_field_dropout,
                    structured_field_dropout_rates=config.structured_field_dropout_rates,
                    pad_token_id=tokenizer.pad_id,
                    foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
                    background_rgb_loss_weight=config.background_rgb_loss_weight,
                    palette_loss_weight=config.palette_loss_weight,
                    palette_loss_temperature=config.palette_loss_temperature,
                    index_head_loss_weight=config.index_head_loss_weight,
                    palette_head_loss_weight=config.palette_head_loss_weight,
                    palette_presence_loss_weight=config.palette_presence_loss_weight,
                    global_step=step,
                    index_head_warmup_steps=config.index_head_warmup_steps,
                )
                loss = losses["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_gradients(model, config.grad_clip)
            optimizer.step()
            if ema_fast_cache is not None:
                _update_ema_state_fast(ema_fast_cache, decay=float(config.ema_decay))
            if scheduler is not None:
                scheduler.step()
            step += 1
            if step % metrics_every == 0 or step >= run_limit:
                last_loss = float(loss.detach().cpu())
                loss_metrics = _loss_metrics(losses)
                last_loss_components = dict(loss_metrics)
                progress.update(step, last_loss, lr=float(optimizer.param_groups[0]["lr"]))
                _write_jsonl_line(
                    metrics_handle,
                    {
                        "step": step,
                        **loss_metrics,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_norm": gradient_norm,
                        "condition_dropout_fraction": float(losses.get("condition_dropout_fraction", 0.0)),
                        "timestep_mean": float(losses.get("timestep_mean", 0.0)),
                        "invalid_batch_count": invalid_batch_count,
                        "skipped_batch_reasons": skipped_batch_reasons,
                        "sample_cursor": train_sampler.sample_cursor,
                        "epoch": train_sampler.epoch,
                        "gpu_memory_allocated_bytes": int(th.cuda.memory_allocated(device))
                        if device_type(device) == "cuda"
                        else 0,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
            if config.sample_every > 0 and step % int(config.sample_every) == 0:
                _write_challenger_sample_sheet(
                    model,
                    preview_batch,
                    out_dir / f"samples_step_{step:06d}.png",
                    conditioning_mode=conditioning_mode,
                    pad_token_id=tokenizer.pad_id,
                    steps=min(16, max(2, int(config.sample_every))),
                )
            should_save_checkpoint = (
                step in set(checkpoint_steps)
                if campaign_writer is not None
                else _should_save_checkpoint_step(
                    step,
                    save_every=config.save_every,
                    checkpoint_steps=checkpoint_steps,
                )
            )
            if should_save_checkpoint:
                metrics_handle.flush()
                step_checkpoint = out_dir / f"checkpoint_step_{step:06d}.pt"
                sampler_snapshot = current_sampler_state()
                checkpoint_sha256 = _save_checkpoint(
                    step_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    config_json=config_json,
                    step=step,
                    ema_decay=float(config.ema_decay),
                    checkpoint_variant="step",
                    scheduler=scheduler,
                    ema_state=ema_state,
                    metrics_summary=last_loss_components,
                    sampler_state=sampler_snapshot,
                    immutable_writer=campaign_writer,
                )
                checkpoint_step_paths.append(str(logical_out_dir / step_checkpoint.name))
                if campaign_writer is not None and campaign_contract is not None:
                    if not isinstance(checkpoint_sha256, str):
                        raise UnsafeFilesystemOperation("campaign checkpoint publication did not retain a content hash")
                    campaign_writer.write_json_idempotent(
                        f"checkpoint_step_{step:06d}.json",
                        _campaign_checkpoint_sidecar(
                            campaign_contract,
                            effective_experiment_manifest,
                            step=step,
                            checkpoint_name=step_checkpoint.name,
                            checkpoint_sha256=checkpoint_sha256,
                            scheduler_present=scheduler is not None,
                            ema_present=ema_state is not None,
                            sampler_state=sampler_snapshot,
                        ),
                    )
                if ema_state is not None and campaign_writer is None:
                    step_ema_checkpoint = out_dir / f"checkpoint_step_{step:06d}_ema.pt"
                    _save_checkpoint(
                        step_ema_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        config_json=config_json,
                        step=step,
                        model_state_dict=ema_state,
                        ema_decay=float(config.ema_decay),
                        checkpoint_variant="step_ema",
                        ema_weights=True,
                        scheduler=scheduler,
                        ema_state=ema_state,
                        metrics_summary=last_loss_components,
                        sampler_state=current_sampler_state(),
                    )
                    checkpoint_step_ema_paths.append(str(logical_out_dir / step_ema_checkpoint.name))
            if campaign_contract is not None and step in set(campaign_contract["expected_evaluation_steps"]):
                evaluate_campaign_step(step)
    finally:
        metrics_handle.close()
        progress.close(final_loss=last_loss)

    final_train_losses = _evaluate_challenger_losses(
        model,
        eval_train_loader,
        device=device,
        conditioning_mode=conditioning_mode,
        pad_token_id=tokenizer.pad_id,
        seed=config.seed + 1,
        foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
        background_rgb_loss_weight=config.background_rgb_loss_weight,
        palette_loss_weight=config.palette_loss_weight,
        palette_loss_temperature=config.palette_loss_temperature,
        max_batches=config.eval_max_batches,
        timestep_boundaries=timestep_boundaries,
    )
    val_losses = (
        _evaluate_challenger_losses(
            model,
            val_loader,
            device=device,
            conditioning_mode=conditioning_mode,
            pad_token_id=tokenizer.pad_id,
            seed=config.seed + 2,
            foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
            background_rgb_loss_weight=config.background_rgb_loss_weight,
            palette_loss_weight=config.palette_loss_weight,
            palette_loss_temperature=config.palette_loss_temperature,
            max_batches=config.eval_max_batches,
            timestep_boundaries=timestep_boundaries,
        )
        if len(val_source)
        else None
    )
    ema_val_losses = None
    if ema_state is not None and len(val_source):
        live_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema_state)
        try:
            ema_val_losses = _evaluate_challenger_losses(
                model,
                val_loader,
                device=device,
                conditioning_mode=conditioning_mode,
                pad_token_id=tokenizer.pad_id,
                seed=config.seed + 2,
                foreground_rgb_loss_weight=config.foreground_rgb_loss_weight,
                background_rgb_loss_weight=config.background_rgb_loss_weight,
                palette_loss_weight=config.palette_loss_weight,
                palette_loss_temperature=config.palette_loss_temperature,
                max_batches=config.eval_max_batches,
                timestep_boundaries=timestep_boundaries,
            )
        finally:
            model.load_state_dict(live_state)
    if campaign_writer is None:
        _save_checkpoint(
            out_dir / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            config_json=config_json,
            step=step,
            ema_decay=float(config.ema_decay),
            checkpoint_variant="last",
            scheduler=scheduler,
            ema_state=ema_state,
            metrics_summary=last_loss_components,
            sampler_state=current_sampler_state(),
        )
        if ema_state is not None:
            _save_checkpoint(
                out_dir / "checkpoint_last_ema.pt",
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                config_json=config_json,
                step=step,
                model_state_dict=ema_state,
                ema_decay=float(config.ema_decay),
                checkpoint_variant="last_ema",
                ema_weights=True,
                scheduler=scheduler,
                ema_state=ema_state,
                metrics_summary=last_loss_components,
                sampler_state=current_sampler_state(),
            )
        _save_checkpoint(
            out_dir / "checkpoint_best.pt",
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            config_json=config_json,
            step=step,
            ema_decay=float(config.ema_decay),
            checkpoint_variant="best",
            scheduler=scheduler,
            ema_state=ema_state,
            metrics_summary=last_loss_components,
            sampler_state=current_sampler_state(),
        )
        if ema_state is not None:
            _save_checkpoint(
                out_dir / "checkpoint_best_ema.pt",
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                config_json=config_json,
                step=step,
                model_state_dict=ema_state,
                ema_decay=float(config.ema_decay),
                checkpoint_variant="best_ema",
                ema_weights=True,
                scheduler=scheduler,
                ema_state=ema_state,
                metrics_summary=last_loss_components,
                sampler_state=current_sampler_state(),
            )
    _write_challenger_sample_sheet(
        model,
        preview_batch,
        out_dir / "samples_final.png",
        conditioning_mode=conditioning_mode,
        pad_token_id=tokenizer.pad_id,
        steps=16,
    )
    report = {
        "model_type": "generator_challenger",
        "architecture": "rectified_flow",
        "dataset": str(config.dataset_dir),
        "training_manifest": str(config.training_manifest),
        "conditioning_mode": conditioning_mode,
        "cfg_dropout": float(config.cfg_dropout),
        "structured_field_dropout": float(config.structured_field_dropout),
        "ema_enabled": ema_enabled,
        "ema_decay": float(config.ema_decay),
        "foreground_rgb_loss_weight": float(config.foreground_rgb_loss_weight),
        "background_rgb_loss_weight": float(config.background_rgb_loss_weight),
        "palette_loss_weight": float(config.palette_loss_weight),
        "palette_loss_temperature": float(config.palette_loss_temperature),
        "palette_swap": palette_swap_summary,
        "role_ramp_transplant": role_ramp_summary,
        "palette_conditioning": bool(config.palette_conditioning),
        "seed": int(config.seed),
        "batch_size": int(config.batch_size),
        "max_steps": int(config.max_steps),
        "steps_completed": int(step),
        "device": str(device),
        "split": effective_split,
        "train_records": len(train_dataset),
        "val_records": len(val_source),
        "validation_mode": validation_mode,
        "overfit_subset": None if subset_selection is None else subset_selection.to_report(),
        "initial_train_loss": float(initial_train_losses["loss"]),
        "initial_train_loss_components": initial_train_losses,
        "final_train_loss": float(final_train_losses["loss"]),
        "final_train_loss_components": final_train_losses,
        "loss_decrease": float(initial_train_losses["loss"]) - float(final_train_losses["loss"]),
        "last_step_loss": last_loss,
        "last_step_loss_components": last_loss_components,
        "val_loss": None if val_losses is None else float(val_losses["loss"]),
        "val_loss_components": None if val_losses is None else val_losses,
        "ema_val_loss": None if ema_val_losses is None else float(ema_val_losses["loss"]),
        "ema_val_loss_components": ema_val_losses,
        "timestep_validation": {
            "boundaries": list(timestep_boundaries),
            "non_ema": None if val_losses is None else val_losses.get("timestep_buckets"),
            "ema": None if ema_val_losses is None else ema_val_losses.get("timestep_buckets"),
        },
        "loss_decreased": float(final_train_losses["loss"]) < float(initial_train_losses["loss"]),
        "elapsed_seconds": time.perf_counter() - started,
        "data_throughput_sprites_per_second": (step * int(config.batch_size))
        / max(1e-9, time.perf_counter() - started),
        "model_config": model_config,
        "structured_vocab_sizes": None if structured_vocab is None else structured_vocab.sizes(),
        "structured_fields_enabled": structured_vocab is not None,
        "checkpoint_last": str(
            logical_out_dir
            / (f"checkpoint_step_{step:06d}.pt" if campaign_writer is not None else "checkpoint_last.pt")
        ),
        "checkpoint_best": str(
            logical_out_dir
            / (f"checkpoint_step_{step:06d}.pt" if campaign_writer is not None else "checkpoint_best.pt")
        ),
        "checkpoint_last_ema": (
            None
            if ema_state is None or campaign_writer is not None
            else str(logical_out_dir / "checkpoint_last_ema.pt")
        ),
        "checkpoint_best_ema": (
            None
            if ema_state is None or campaign_writer is not None
            else str(logical_out_dir / "checkpoint_best_ema.pt")
        ),
        "checkpoint_steps": list(checkpoint_steps),
        "checkpoint_step_paths": checkpoint_step_paths,
        "checkpoint_step_ema_paths": checkpoint_step_ema_paths,
        "warnings": run_warnings + list(determinism_report.get("issues", [])),
        "determinism": determinism_report,
        "sampler_state_schema": current_sampler_state()["schema_version"],
        "resume_from": None if config.resume_from is None else str(config.resume_from),
        "resume_checkpoint_sha256": config.expected_resume_sha256,
        "resume_compatibility_mismatches": resume_mismatches,
        "unsafe_resume_record": unsafe_resume_record,
        "auxiliary_heads_mode": auxiliary_mode.value,
        "legacy_auxiliary_heads_adapter": legacy_auxiliary_adapter,
        "fair_architecture_comparison_eligible": not legacy_auxiliary_adapter and not bool(unsafe_resume_record),
        "checkpoint_promotion_eligible": not legacy_auxiliary_adapter and not bool(unsafe_resume_record),
        "label_quality": {
            "dataset": dataset_label_quality,
            "training": train_label_quality,
            "evaluation": val_label_quality,
            "correlation_analysis": label_quality_correlations,
        },
    }
    (out_dir / "train_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if campaign_writer is not None and campaign_contract is not None:
        if step == int(campaign_contract["max_optimizer_steps"]):
            _publish_campaign_completion(
                campaign_writer,
                campaign_contract,
                experiment_manifest=effective_experiment_manifest,
                report=report,
                final_train_losses=final_train_losses,
                validation_losses=val_losses,
                ema_validation_losses=ema_val_losses,
                evaluations=campaign_evaluations,
                resume_mismatches=resume_mismatches,
                determinism_report=determinism_report,
            )
        elif step > int(campaign_contract["max_optimizer_steps"]):
            raise UnsafeFilesystemOperation("campaign trainer exceeded its fixed optimizer-step budget")
    return report


# Optional v1.1 export preset (see docs/v1_1_factored_cfg.md): v1 base settings (Phase 1
# EMA checkpoint, CFG 3.0, 30 steps, k16 projection) plus factored CFG at the base/color
# scales validated by the 3-seed Phase 0 factored-CFG confirmation. v1 remains the default;
# v1.1 must be requested explicitly via --export-preset v1.1 (or the v1_1 / phase1_v1_1
# aliases) and never changes v1/phase1_v1 behavior.
V1_PRESET_ALIASES: tuple[str, ...] = ("v1", "phase1_v1")
V1_1_PRESET_ALIASES: tuple[str, ...] = ("v1.1", "v1_1", "phase1_v1_1")
V1_1_CFG_BASE_SCALE = 2.5
V1_1_CFG_COLOR_SCALE = 3.0


def normalize_export_preset(export_preset: str | None) -> str | None:
    """Return the canonical preset name ("v1" or "v1.1") for a preset alias, or None."""

    normalized = str(export_preset or "").strip().lower()
    if normalized in V1_PRESET_ALIASES:
        return "v1"
    if normalized in V1_1_PRESET_ALIASES:
        return "v1.1"
    return None


def _resolve_sample_export_checkpoint(checkpoint: Path, export_preset: str | None) -> Path:
    if normalize_export_preset(export_preset) is None:
        return Path(checkpoint)

    path = Path(checkpoint)
    if path.is_dir():
        for name in (
            "checkpoint_last_ema.pt",
            "checkpoint_best_ema.pt",
            "checkpoint_last.pt",
            "checkpoint_best.pt",
        ):
            candidate = path / name
            if candidate.is_file():
                return candidate
        return path

    candidates: list[Path] = []
    if path.name == "checkpoint_last.pt":
        candidates.append(path.with_name("checkpoint_last_ema.pt"))
    if path.name == "checkpoint_best.pt":
        candidates.append(path.with_name("checkpoint_best_ema.pt"))
    if path.suffix == ".pt" and not path.stem.endswith("_ema"):
        candidates.append(path.with_name(f"{path.stem}_ema{path.suffix}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def run_sample_generator_challenger(config: ChallengerSampleConfig) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    started = time.perf_counter()
    _set_seed(config.seed)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    checkpoint = _resolve_sample_export_checkpoint(Path(config.checkpoint), config.export_preset)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Requested: {config.checkpoint} (export-preset={config.export_preset or 'none'})\n"
            "The official v1 checkpoint path is "
            "experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt "
            "(see docs/v1_default.md). Pass --checkpoint to point at a different file, "
            "or train the Phase 1 challenger first."
        )
    ckpt = _load_checkpoint(checkpoint, expected_sha256=config.expected_checkpoint_sha256)
    if config.expected_checkpoint_step is not None:
        expected_step = config.expected_checkpoint_step
        step = ckpt.get("step")
        global_step = ckpt.get("global_step")
        if (
            type(expected_step) is not int
            or expected_step < 0
            or type(step) is not int
            or type(global_step) is not int
            or step != global_step
            or step != expected_step
        ):
            raise ValueError("checkpoint step/global-step metadata does not match the expected artifact")
    if config.expected_checkpoint_variant is not None:
        expected_variant = config.expected_checkpoint_variant
        if expected_variant not in {"live", "ema"} or ckpt.get("ema_weights") is not (expected_variant == "ema"):
            raise ValueError("checkpoint live/EMA metadata does not match the expected artifact")
    if ckpt.get("model_type") != "generator_challenger":
        raise ValueError("checkpoint model type does not match generator_challenger")
    model, tokenizer, conditioning_mode, semantic_max_length = load_challenger_from_checkpoint(
        ckpt, device=device, legacy_evaluation_import=True
    )
    structured_vocab = structured_vocab_from_checkpoint(
        ckpt, allow_schema_v1_adapter=config.allow_legacy_conditioning_v1
    )
    prompts = read_prompt_records(config.prompts, max_records=config.max_samples)
    palette_source_mode = str(config.palette_conditioning_source or "none").strip().lower()
    if palette_source_mode not in {"none", "source", "retrieved"}:
        raise ValueError("palette_conditioning_source must be one of: none, source, retrieved")
    if palette_source_mode != "none" and not model.palette_conditioning:
        raise ValueError("palette conditioning source was requested for a checkpoint without --palette-conditioning")
    palette_library: PaletteConditionLibrary | None = None
    if model.palette_conditioning and palette_source_mode != "none":
        dataset_dir = config.palette_conditioning_dataset or Path(str(ckpt.get("dataset") or ""))
        manifest_path = config.palette_conditioning_training_manifest or Path(str(ckpt.get("training_manifest") or ""))
        if not str(dataset_dir) or not str(manifest_path) or not Path(manifest_path).is_file():
            raise ValueError(
                "--palette-conditioning-source source|retrieved requires --palette-conditioning-dataset "
                "and --palette-conditioning-training-manifest (or those paths embedded in the checkpoint)"
            )
        palette_library = build_palette_condition_library(Path(dataset_dir), read_jsonl(Path(manifest_path)))
        if not palette_library.entries:
            raise ValueError("palette conditioning library has no usable real palettes")
    manifest_records: list[dict[str, Any]] = []
    base_noise_seed = int(config.noise_seed) if config.noise_seed is not None else int(config.seed) * 100000
    null_fields = _parse_null_fields(config.null_fields)
    color_token_ids = color_token_ids_for_tokenizer(tokenizer) if config.factored_cfg else ()
    for batch_start in range(0, len(prompts), max(1, int(config.batch_size))):
        batch_records = prompts[batch_start : batch_start + max(1, int(config.batch_size))]
        if not batch_records:
            continue
        noise_seeds = [
            base_noise_seed
            + int(
                record.get("benchmark_noise_offset")
                if record.get("benchmark_noise_offset") is not None
                else batch_start + index
            )
            for index, record in enumerate(batch_records)
        ]
        caption_tokens = th.as_tensor(
            [tokenizer.encode(str(record["prompt"]), max_length=tokenizer.max_length) for record in batch_records],
            dtype=th.long,
            device=device,
        )
        semantic_tokens = th.as_tensor(
            [tokenizer.encode_record_semantics(record, max_length=semantic_max_length) for record in batch_records],
            dtype=th.long,
            device=device,
        )
        structured_conditioning = _structured_conditioning_for_records(
            batch_records,
            structured_vocab=structured_vocab,
            device=device,
        )
        inputs = apply_conditioning_mode(
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            mode=conditioning_mode,
            pad_token_id=tokenizer.pad_id,
            structured_conditioning=structured_conditioning,
        )
        if null_fields:
            inputs = apply_conditioning_field_ablations(
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
                fields=null_fields,
                pad_token_id=tokenizer.pad_id,
            )
        palette_entries = _select_palette_conditions(
            batch_records,
            library=palette_library,
            mode=palette_source_mode,
            exclude_exact_prompt_target=bool(config.palette_conditioning_exclude_exact_prompt_target),
        )
        palette_condition = (
            None
            if not model.palette_conditioning
            else th.as_tensor(
                np.stack(
                    [
                        np.zeros((PALETTE_CONDITION_SLOTS, PALETTE_CONDITION_CHANNELS), dtype=np.float32)
                        if entry is None
                        else entry.condition
                        for entry in palette_entries
                    ]
                ),
                dtype=th.float32,
                device=device,
            )
        )
        initial = th.cat(
            [_sample_initial_noise(1, device=device, seed=noise_seed) for noise_seed in noise_seeds],
            dim=0,
        )
        with th.no_grad():
            rgba_batch = integrate_rectified_flow(
                model,
                initial,
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
                steps=config.steps,
                cfg_scale=config.cfg_scale,
                pad_token_id=tokenizer.pad_id,
                factored_cfg=config.factored_cfg,
                cfg_base_scale=config.cfg_base_scale,
                cfg_color_scale=config.cfg_color_scale,
                color_token_ids=color_token_ids,
                color_guidance_rgb_only=config.color_guidance_rgb_only,
                color_guidance_start_t=config.color_guidance_start_t,
                color_guidance_ramp_t=config.color_guidance_ramp_t,
                object_id_scale=config.object_id_scale,
                palette_condition=palette_condition,
            )
        rgba_np = np.moveaxis(rgba_batch.detach().cpu().numpy().astype(np.float32), 1, -1)
        for item_index, prompt_record in enumerate(batch_records):
            sample_index = batch_start + item_index
            sample_id = f"sample_{sample_index:06d}"
            sprite = canonicalize_generated_rgba(
                rgba_np[item_index],
                max_colors=config.max_colors,
                alpha_threshold=config.alpha_threshold,
                dither=config.dither,
            )
            metadata = {
                **prompt_record,
                "checkpoint": str(checkpoint),
                "requested_checkpoint": str(config.checkpoint),
                "export_preset": str(config.export_preset or ""),
                "seed": int(config.seed),
                "noise_seed": int(noise_seeds[item_index]),
                "model_output_finite": bool(np.isfinite(rgba_np[item_index]).all()),
                "conditioning_mode": conditioning_mode,
                "model_type": "generator_challenger",
                "architecture": "rectified_flow",
                "cfg_scale": float(config.cfg_scale),
                "steps": int(config.steps),
                "alpha_threshold": float(config.alpha_threshold),
                "max_colors": int(config.max_colors),
                "dither": bool(config.dither),
                "factored_cfg": bool(config.factored_cfg),
                "cfg_base_scale": None if config.cfg_base_scale is None else float(config.cfg_base_scale),
                "cfg_color_scale": None if config.cfg_color_scale is None else float(config.cfg_color_scale),
                "null_fields": list(null_fields),
                "color_guidance_rgb_only": bool(config.color_guidance_rgb_only),
                "color_guidance_start_t": float(config.color_guidance_start_t),
                "color_guidance_ramp_t": float(config.color_guidance_ramp_t),
                "object_id_scale": float(config.object_id_scale),
                "palette_conditioning": bool(model.palette_conditioning),
                "palette_conditioning_source": palette_source_mode,
                "palette_conditioning_palette_source_id": None
                if palette_entries[item_index] is None
                else palette_entries[item_index].sprite_id,
                "palette_conditioning_palette_family": None
                if palette_entries[item_index] is None
                else palette_entries[item_index].family,
            }
            record = write_generated_sprite_artifacts(
                sprite,
                out_dir,
                sample_id,
                metadata,
                write_raw_rgba=config.write_raw_rgba,
                write_hard_rgba=config.write_hard_rgba,
            )
            if config.project_palette:
                from spritelab.training.palette_projection import project_generated_sprite_record

                record = project_generated_sprite_record(
                    sprite,
                    out_dir,
                    record,
                    target_colors=config.project_palette_target_colors,
                    min_pixel_share=config.project_palette_min_pixel_share,
                    alpha_threshold=config.alpha_threshold,
                    method=config.project_palette_method,
                )
            manifest_records.append(record)
    contact_sheet_path = build_generation_contact_sheet(
        out_dir,
        manifest_records,
        out_dir / "generation_contact_sheet.png",
        include_raw=config.write_raw_rgba,
    )
    _write_contact_sheet_label_mapping(out_dir, manifest_records, label_mode=config.contact_sheet_labels)
    projection_report = None
    if config.project_palette:
        from spritelab.training.palette_projection import write_runtime_projection_report

        projection_report = write_runtime_projection_report(
            out_dir,
            manifest_records,
            target_colors=config.project_palette_target_colors,
            min_pixel_share=config.project_palette_min_pixel_share,
            alpha_threshold=config.alpha_threshold,
            method=config.project_palette_method,
        )
    config_json = {key: _jsonable(value) for key, value in asdict(config).items()}
    if not config.project_palette:
        for key in (
            "project_palette",
            "project_palette_target_colors",
            "project_palette_min_pixel_share",
            "project_palette_method",
        ):
            config_json.pop(key, None)
    config_json["checkpoint_resolved"] = str(checkpoint)
    report = write_generation_reports(
        out_dir=out_dir,
        records=manifest_records,
        config={
            **config_json,
            "device_resolved": str(device),
            "conditioning_mode": conditioning_mode,
            "structured_vocab_sizes": None if structured_vocab is None else structured_vocab.sizes(),
            "semantic_max_length": semantic_max_length,
            "model_type": "generator_challenger",
            "architecture": "rectified_flow",
            "elapsed_seconds": time.perf_counter() - started,
        },
        contact_sheet=None if contact_sheet_path is None else contact_sheet_path.name,
    )
    if projection_report is not None:
        report["palette_projection"] = {
            "applied": True,
            "report": "palette_projection_report.json",
            "contact_sheet": "contact_sheet_projected.png"
            if (out_dir / "contact_sheet_projected.png").is_file()
            else None,
            "method": str(config.project_palette_method),
            "target_colors": int(config.project_palette_target_colors),
            "min_pixel_share": float(config.project_palette_min_pixel_share),
            "mean_rgb_mae_visible": projection_report.get("mean_rgb_mae_visible"),
            "destructive_rate": projection_report.get("destructive_rate"),
        }
        (out_dir / "generation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def rectified_flow_loss(
    model: RectifiedFlowUNet,
    batch: dict[str, Any],
    *,
    conditioning_mode: str,
    cfg_dropout: float,
    structured_field_dropout: float = 0.0,
    structured_field_dropout_rates: Mapping[str, float] | None = None,
    pad_token_id: int,
    foreground_rgb_loss_weight: float = 1.0,
    background_rgb_loss_weight: float = 1.0,
    palette_loss_weight: float = 0.0,
    palette_loss_temperature: float = 0.05,
    # v2 Phase 2
    index_head_loss_weight: float = 0.0,
    palette_head_loss_weight: float = 0.0,
    palette_presence_loss_weight: float = 0.0,
    global_step: int = 0,
    index_head_warmup_steps: int = 0,
) -> Any:
    th, _nn_mod = _require_torch()
    auxiliary_weights = (
        float(index_head_loss_weight),
        float(palette_head_loss_weight),
        float(palette_presence_loss_weight),
    )
    if model.auxiliary_heads_mode is AuxiliaryHeadsMode.ABSENT and any(weight != 0.0 for weight in auxiliary_weights):
        raise ValueError("nonzero auxiliary loss requires auxiliary_heads_mode='palette_index'")
    target_rgba = batch["rgba"]
    target = target_rgba * 2.0 - 1.0
    x0 = th.randn_like(target)
    t = th.rand(int(target.shape[0]), device=target.device)
    view_t = t.view(-1, 1, 1, 1)
    xt = (1.0 - view_t) * x0 + view_t * target
    velocity = target - x0
    inputs = apply_conditioning_mode(
        caption_tokens=batch["caption_tokens"],
        semantic_tokens=batch["semantic_tokens"],
        mode=conditioning_mode,
        pad_token_id=pad_token_id,
        structured_conditioning=_structured_conditioning_from_batch(batch),
    )
    inputs = _apply_cfg_dropout(inputs, dropout=cfg_dropout, pad_token_id=pad_token_id)
    inputs = _apply_structured_field_dropout(
        inputs,
        dropout=structured_field_dropout,
        training=bool(model.training),
        dropout_rates=structured_field_dropout_rates,
    )

    # Determine if aux heads should run
    aux_active = (palette_head_loss_weight > 0.0 or palette_presence_loss_weight > 0.0) or (
        index_head_loss_weight > 0.0 and global_step >= index_head_warmup_steps
    )

    if aux_active:
        aux = model(
            xt,
            t,
            caption_tokens=inputs["caption_tokens"],
            semantic_tokens=inputs["semantic_tokens"],
            structured_conditioning=inputs.get("structured_conditioning"),
            return_aux=True,
            palette_condition=batch.get("palette_condition"),
        )
        pred = aux["velocity"]
    else:
        pred = model(
            xt,
            t,
            caption_tokens=inputs["caption_tokens"],
            semantic_tokens=inputs["semantic_tokens"],
            structured_conditioning=inputs.get("structured_conditioning"),
            palette_condition=batch.get("palette_condition"),
        )

    losses = _velocity_loss_components(
        pred,
        velocity,
        target_rgba=target_rgba,
        foreground_rgb_loss_weight=foreground_rgb_loss_weight,
        background_rgb_loss_weight=background_rgb_loss_weight,
        sample_weights=batch.get("record_loss_weight"),
    )
    per_example_velocity_loss = (pred - velocity).square().mean(dim=(1, 2, 3))
    for lower, upper, label in (
        (0.0, 0.25, "0_00_0_25"),
        (0.25, 0.5, "0_25_0_50"),
        (0.5, 0.75, "0_50_0_75"),
        (0.75, 1.01, "0_75_1_00"),
    ):
        weights = ((t >= lower) & (t < upper)).to(per_example_velocity_loss.dtype)
        losses[f"loss_timestep_{label}"] = (per_example_velocity_loss * weights).sum() / weights.sum().clamp(min=1.0)
    # Palette soft-min auxiliary loss: skip when weight is zero to avoid the
    # per-step GPU→CPU sync inside palette_soft_min_auxiliary_loss.
    palette_weight = float(palette_loss_weight)
    if palette_weight > 0.0:
        palette_aux = palette_soft_min_auxiliary_loss(
            x1_hat=xt + (1.0 - view_t) * pred,
            target_rgba=target_rgba,
            palette=batch.get("palette"),
            palette_mask=batch.get("palette_mask"),
            temperature=palette_loss_temperature,
            sample_weights=batch.get("record_loss_weight"),
        )
    else:
        palette_aux = pred.sum() * 0.0
    losses["loss_palette_aux"] = palette_aux

    # v2 Phase 2: palette and index head losses
    zero = pred.sum() * 0.0
    losses["loss_palette_head"] = zero
    losses["loss_palette_presence"] = zero
    losses["loss_index_head"] = zero
    losses["index_head_active"] = False
    losses["auxiliary_heads_mode"] = model.auxiliary_heads_mode.value
    losses["auxiliary_heads_available"] = model.auxiliary_heads_mode is AuxiliaryHeadsMode.PALETTE_INDEX
    losses["auxiliary_loss_enabled"] = aux_active

    if aux_active:
        if not aux["auxiliary_heads_available"] or any(
            aux[key] is None for key in ("palette_rgb", "palette_presence_logits", "index_logits")
        ):
            raise RuntimeError("auxiliary output contract is unavailable for an enabled auxiliary loss")
        head_losses = _palette_head_loss(aux["palette_rgb"], aux["palette_presence_logits"], batch)
        losses["loss_palette_head"] = head_losses["loss_palette_head"]
        losses["loss_palette_presence"] = head_losses["loss_palette_presence"]

        eff_index_weight = float(index_head_loss_weight) if global_step >= index_head_warmup_steps else 0.0
        if eff_index_weight > 0.0:
            losses["loss_index_head"] = _index_head_loss(aux["index_logits"], batch)
            losses["index_head_active"] = True

    total = losses["loss_velocity"] + (palette_aux * palette_weight if palette_weight > 0.0 else zero)
    total = total + float(palette_head_loss_weight) * losses["loss_palette_head"]
    total = total + float(palette_presence_loss_weight) * losses["loss_palette_presence"]
    if global_step >= index_head_warmup_steps:
        total = total + float(index_head_loss_weight) * losses["loss_index_head"]

    losses["loss"] = total
    losses["condition_dropout_fraction"] = float(inputs.get("cfg_dropout_fraction", 0.0))
    losses["timestep_mean"] = float(t.detach().mean().cpu())
    return losses


def evaluate_challenger_loss(
    model: RectifiedFlowUNet,
    loader: Any,
    *,
    device: Any,
    conditioning_mode: str,
    pad_token_id: int,
    seed: int,
    foreground_rgb_loss_weight: float = 1.0,
    background_rgb_loss_weight: float = 1.0,
    palette_loss_weight: float = 0.0,
    palette_loss_temperature: float = 0.05,
) -> float:
    global _palette_scale_needs_normalize
    _palette_scale_needs_normalize = None  # reset per-run palette scale cache
    return float(
        _evaluate_challenger_losses(
            model,
            loader,
            device=device,
            conditioning_mode=conditioning_mode,
            pad_token_id=pad_token_id,
            seed=seed,
            foreground_rgb_loss_weight=foreground_rgb_loss_weight,
            background_rgb_loss_weight=background_rgb_loss_weight,
            palette_loss_weight=palette_loss_weight,
            palette_loss_temperature=palette_loss_temperature,
        )["loss"]
    )


def _evaluate_challenger_losses(
    model: RectifiedFlowUNet,
    loader: Any,
    *,
    device: Any,
    conditioning_mode: str,
    pad_token_id: int,
    seed: int,
    foreground_rgb_loss_weight: float = 1.0,
    background_rgb_loss_weight: float = 1.0,
    palette_loss_weight: float = 0.0,
    palette_loss_temperature: float = 0.05,
    max_batches: int = 0,
    timestep_boundaries: Sequence[float] = DEFAULT_TIMESTEP_BOUNDARIES,
) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    generator = th.Generator(device=device)
    generator.manual_seed(int(seed))
    totals: dict[str, float] = {}
    count = 0.0
    batches_seen = 0
    batch_cap = int(max_batches)
    bucket_accumulator = TimestepBucketAccumulator(timestep_boundaries)
    was_training = bool(model.training)
    model.eval()
    with th.no_grad():
        for batch in loader:
            if batch_cap > 0 and batches_seen >= batch_cap:
                break
            batches_seen += 1
            batch = move_batch_to_device(batch, device)
            target_rgba = batch["rgba"]
            target = target_rgba * 2.0 - 1.0
            x0 = th.randn(target.shape, device=device, generator=generator)
            t = th.rand(int(target.shape[0]), device=device, generator=generator)
            view_t = t.view(-1, 1, 1, 1)
            xt = (1.0 - view_t) * x0 + view_t * target
            velocity = target - x0
            inputs = apply_conditioning_mode(
                caption_tokens=batch["caption_tokens"],
                semantic_tokens=batch["semantic_tokens"],
                mode=conditioning_mode,
                pad_token_id=pad_token_id,
                structured_conditioning=_structured_conditioning_from_batch(batch),
            )
            aux = model(
                xt,
                t,
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
                palette_condition=batch.get("palette_condition"),
                return_aux=True,
            )
            pred = aux["velocity"]
            losses = _velocity_loss_components(
                pred,
                velocity,
                target_rgba=target_rgba,
                foreground_rgb_loss_weight=foreground_rgb_loss_weight,
                background_rgb_loss_weight=background_rgb_loss_weight,
                sample_weights=batch.get("record_loss_weight"),
            )
            palette_aux = palette_soft_min_auxiliary_loss(
                x1_hat=xt + (1.0 - view_t) * pred,
                target_rgba=target_rgba,
                palette=batch.get("palette"),
                palette_mask=batch.get("palette_mask"),
                temperature=palette_loss_temperature,
                sample_weights=batch.get("record_loss_weight"),
            )
            palette_weight = float(palette_loss_weight)
            losses["loss_palette_aux"] = palette_aux
            losses["loss"] = losses["loss_velocity"] + (
                palette_aux * palette_weight if palette_weight > 0.0 else palette_aux * 0.0
            )
            batch_size = int(target.shape[0])
            sample_weights = batch.get("record_loss_weight")
            batch_mass = float(sample_weights.detach().sum().cpu()) if sample_weights is not None else float(batch_size)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_mass
            count += batch_mass
            per_sample = _validation_per_sample_losses(
                pred=pred,
                velocity=velocity,
                aux=aux,
                xt=xt,
                view_t=view_t,
                batch=batch,
                foreground_rgb_loss_weight=foreground_rgb_loss_weight,
                background_rgb_loss_weight=background_rgb_loss_weight,
                palette_loss_temperature=palette_loss_temperature,
            )
            for row, timestep in zip(per_sample, t.detach().cpu().tolist(), strict=True):
                bucket_accumulator.add(float(timestep), row)
    if was_training:
        model.train()
    result: dict[str, Any] = {key: value / count for key, value in sorted(totals.items())} if count else {"loss": 0.0}
    result["timestep_buckets"] = bucket_accumulator.report()
    return result


def _validation_per_sample_losses(
    *,
    pred: Any,
    velocity: Any,
    aux: Mapping[str, Any],
    xt: Any,
    view_t: Any,
    batch: Mapping[str, Any],
    foreground_rgb_loss_weight: float,
    background_rgb_loss_weight: float,
    palette_loss_temperature: float,
) -> list[dict[str, float]]:
    th, _nn_mod = _require_torch()
    squared = (pred - velocity).square()
    visible = batch["rgba"][:, 3:4].to(device=pred.device) > 0.5
    rgb_weight = th.where(
        visible,
        th.as_tensor(float(foreground_rgb_loss_weight), device=pred.device, dtype=pred.dtype),
        th.as_tensor(float(background_rgb_loss_weight), device=pred.device, dtype=pred.dtype),
    )
    flow = (squared[:, :3] * rgb_weight).sum(dim=(1, 2, 3)) + squared[:, 3:4].sum(dim=(1, 2, 3))
    flow = flow / float(squared[0].numel())
    palette_per_sample = _palette_auxiliary_per_sample(
        x1_hat=xt + (1.0 - view_t) * pred,
        target_rgba=batch["rgba"],
        palette=batch.get("palette"),
        palette_mask=batch.get("palette_mask"),
        temperature=palette_loss_temperature,
    )
    rows: list[dict[str, float]] = []
    for index in range(int(pred.shape[0])):
        sliced = {
            key: value[index : index + 1] if isinstance(value, th.Tensor) and value.ndim else value
            for key, value in batch.items()
        }
        index_logits = aux.get("index_logits")
        index_loss = (
            pred[index : index + 1].sum() * 0.0
            if index_logits is None
            else _index_head_loss(index_logits[index : index + 1], sliced)
        )
        rows.append(
            {
                "loss_velocity": float(flow[index].detach().cpu()),
                "loss_palette_aux": float(palette_per_sample[index].detach().cpu()),
                "loss_index_head": float(index_loss.detach().cpu()),
            }
        )
    return rows


def _palette_auxiliary_per_sample(
    *, x1_hat: Any, target_rgba: Any, palette: Any, palette_mask: Any, temperature: float
) -> Any:
    th, _nn_mod = _require_torch()
    batch_size = int(x1_hat.shape[0])
    result = th.zeros(batch_size, device=x1_hat.device, dtype=x1_hat.dtype)
    if palette is None or palette_mask is None or batch_size == 0:
        return result
    palette_rgb = palette.to(device=x1_hat.device, dtype=x1_hat.dtype)[..., :3]
    if palette_rgb.numel() and float(palette_rgb.detach().max().cpu()) > 1.0:
        palette_rgb = palette_rgb / 255.0
    valid = palette_mask.to(device=x1_hat.device, dtype=th.bool).clone()
    if valid.shape[1] > 0:
        valid[:, 0] = False
    pred_rgb = ((x1_hat[:, :3] + 1.0) * 0.5).clamp(0.0, 1.0).permute(0, 2, 3, 1)
    distances = ((pred_rgb[:, :, :, None, :] - palette_rgb[:, None, None, :, :]) ** 2).sum(dim=-1)
    distances = distances.masked_fill(~valid[:, None, None, :], 1.0e6)
    weights = th.softmax(-distances / max(float(temperature), 1.0e-6), dim=-1)
    soft_min = (weights * distances).sum(dim=-1)
    visible = target_rgba[:, 3] > 0.5
    for index in range(batch_size):
        mask = visible[index] & valid[index].any()
        if bool(mask.any()):
            result[index] = soft_min[index][mask].mean()
    return result


def _velocity_loss_components(
    pred: Any,
    velocity: Any,
    *,
    target_rgba: Any,
    foreground_rgb_loss_weight: float = 1.0,
    background_rgb_loss_weight: float = 1.0,
    sample_weights: Any | None = None,
) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    squared = (pred - velocity) ** 2
    rgb_squared = squared[:, :3]
    alpha_squared = squared[:, 3:4]
    visible = target_rgba[:, 3:4].to(device=pred.device) > 0.5
    rgb_weight = th.where(
        visible,
        th.as_tensor(float(foreground_rgb_loss_weight), dtype=rgb_squared.dtype, device=rgb_squared.device),
        th.as_tensor(float(background_rgb_loss_weight), dtype=rgb_squared.dtype, device=rgb_squared.device),
    )
    weighted_rgb_squared = rgb_squared * rgb_weight
    per_example_denom = float(squared[0].numel())
    per_example_rgb = weighted_rgb_squared.sum(dim=(1, 2, 3)) / per_example_denom
    per_example_alpha = alpha_squared.sum(dim=(1, 2, 3)) / per_example_denom
    loss_rgb = _normalized_weighted_mean(per_example_rgb, sample_weights)
    loss_alpha = _normalized_weighted_mean(per_example_alpha, sample_weights)
    return {
        "loss_velocity": loss_rgb + loss_alpha,
        "loss_rgb": loss_rgb,
        "loss_alpha": loss_alpha,
    }


def _normalized_weighted_mean(values: Any, sample_weights: Any | None) -> Any:
    """Return ``sum(w*x)/sum(w)`` and preserve the ordinary mean for legacy rows."""

    th, _nn_mod = _require_torch()
    flattened = values.reshape(-1)
    if sample_weights is None:
        return flattened.mean()
    weights = sample_weights.to(device=flattened.device, dtype=flattened.dtype).reshape(-1)
    if weights.shape != flattened.shape:
        raise ValueError(f"sample_weights shape {tuple(weights.shape)} does not match values {tuple(flattened.shape)}")
    if not bool(th.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("sample_weights must be finite and non-negative")
    denominator = weights.sum()
    if not bool(denominator > 0):
        return flattened.sum() * 0.0
    return (flattened * weights).sum() / denominator


# v2 Phase 2: palette/index auxiliary losses ─────────────────────────────────


def _palette_head_loss(
    pred_rgb: Any,
    pred_presence_logits: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """Direct palette prediction: slot-aligned MSE + presence BCE.

    Handles variable-sized ground-truth palettes by truncating or
    zero-padding predictions to match the GT slot count.
    """
    th, _nn_mod = _require_torch()
    zero = pred_rgb.sum() * 0.0
    palette = batch.get("palette")
    palette_mask = batch.get("palette_mask")
    if palette is None or palette_mask is None:
        return {"loss_palette_head": zero, "loss_palette_presence": zero}

    gt_rgb = palette.to(dtype=pred_rgb.dtype, device=pred_rgb.device)
    _ensure_palette_in_01(gt_rgb)
    gt_rgb = gt_rgb.clamp(0.0, 1.0)
    gt_mask = palette_mask.to(device=pred_rgb.device, dtype=th.bool)

    K_gt = gt_rgb.shape[1]
    K_pred = pred_rgb.shape[1]

    if K_pred != K_gt:
        # Truncate or zero-pad predictions to match GT slot count
        if K_pred > K_gt:
            pred_rgb = pred_rgb[:, :K_gt]
            pred_presence_logits = pred_presence_logits[:, :K_gt]
        else:
            # Pad with zeros (RGB) and large negative logits (presence)
            pad_k = K_gt - K_pred
            pred_rgb = th.cat(
                [
                    pred_rgb,
                    th.zeros(pred_rgb.shape[0], pad_k, 3, device=pred_rgb.device, dtype=pred_rgb.dtype),
                ],
                dim=1,
            )
            pred_presence_logits = th.cat(
                [
                    pred_presence_logits,
                    th.full(
                        (pred_presence_logits.shape[0], pad_k),
                        -10.0,
                        device=pred_presence_logits.device,
                        dtype=pred_presence_logits.dtype,
                    ),
                ],
                dim=1,
            )

    mse = ((pred_rgb - gt_rgb) ** 2).mean(dim=-1)  # (B, K_gt)
    mse = mse.masked_select(gt_mask)
    loss_rgb = mse.mean() if mse.numel() > 0 else zero

    if pred_presence_logits.shape[-1] == gt_mask.shape[-1]:
        bce = th.nn.functional.binary_cross_entropy_with_logits(pred_presence_logits, gt_mask.float())
    else:
        bce = zero

    return {"loss_palette_head": loss_rgb, "loss_palette_presence": bce}


def _index_head_loss(
    pred_logits: Any,
    batch: dict[str, Any],
) -> Any:
    """Cross-entropy on visible pixel index map.

    Clamps out-of-range GT indices to the maximum predicted class,
    ignoring transparent/background pixels.
    """
    th, _nn_mod = _require_torch()
    zero = pred_logits.sum() * 0.0
    index_map = batch.get("index_map")
    if index_map is None:
        return zero

    gt = index_map.to(device=pred_logits.device, dtype=th.long)
    max_valid = pred_logits.shape[1] - 1

    # Visible pixels only (index > 0 removes background/transparent)
    visible = gt > 0
    if not bool(visible.any()):
        return zero

    gt_clamped = gt.clamp(min=0, max=max_valid)
    ce = th.nn.functional.cross_entropy(pred_logits, gt_clamped, reduction="none")
    ce_masked = ce.masked_select(visible)
    return ce_masked.mean() if ce_masked.numel() > 0 else zero


def palette_soft_min_auxiliary_loss(
    *,
    x1_hat: Any,
    target_rgba: Any,
    palette: Any,
    palette_mask: Any,
    temperature: float = 0.05,
    sample_weights: Any | None = None,
) -> Any:
    th, _nn_mod = _require_torch()
    zero = x1_hat.sum() * 0.0
    if palette is None or palette_mask is None:
        return zero
    if target_rgba.shape[0] == 0:
        return zero
    visible = target_rgba[:, 3] > 0.5
    if not bool(visible.any()):
        return zero

    palette_rgb = palette.to(device=x1_hat.device, dtype=x1_hat.dtype)
    if palette_rgb.ndim != 3 or palette_rgb.shape[-1] < 3:
        return zero
    palette_rgb = palette_rgb[..., :3]
    _ensure_palette_in_01(palette_rgb)
    palette_rgb = palette_rgb.clamp(0.0, 1.0)
    valid = palette_mask.to(device=x1_hat.device, dtype=th.bool)
    if valid.ndim != 2 or valid.shape[:2] != palette_rgb.shape[:2]:
        return zero
    if valid.shape[1] > 0:
        valid = valid.clone()
        valid[:, 0] = False
    valid_per_sample = valid.any(dim=1)
    if not bool(valid_per_sample.any()):
        return zero

    pred_rgb = ((x1_hat[:, :3] + 1.0) * 0.5).clamp(0.0, 1.0).permute(0, 2, 3, 1)
    distances = ((pred_rgb[:, :, :, None, :] - palette_rgb[:, None, None, :, :]) ** 2).sum(dim=-1)
    distances = distances.masked_fill(~valid[:, None, None, :], 1.0e6)
    temp = max(float(temperature), 1.0e-6)
    weights = th.softmax(-distances / temp, dim=-1)
    soft_min = (weights * distances).sum(dim=-1)
    pixel_mask = visible & valid_per_sample[:, None, None]
    if not bool(pixel_mask.any()):
        return zero
    if sample_weights is None:
        return soft_min[pixel_mask].mean()
    record_weights = sample_weights.to(device=x1_hat.device, dtype=x1_hat.dtype).reshape(-1)
    if record_weights.shape[0] != soft_min.shape[0]:
        raise ValueError("sample_weights batch size does not match palette loss batch")
    if not bool(th.isfinite(record_weights).all()) or bool((record_weights < 0).any()):
        raise ValueError("sample_weights must be finite and non-negative")
    pixel_weights = record_weights[:, None, None].expand_as(soft_min)[pixel_mask]
    denominator = pixel_weights.sum()
    if not bool(denominator > 0):
        return zero
    return (soft_min[pixel_mask] * pixel_weights).sum() / denominator


def integrate_rectified_flow(
    model: RectifiedFlowUNet,
    initial: Any,
    *,
    caption_tokens: Any,
    semantic_tokens: Any | None,
    structured_conditioning: Mapping[str, Any] | None = None,
    steps: int,
    cfg_scale: float,
    pad_token_id: int,
    factored_cfg: bool = False,
    cfg_base_scale: float | None = None,
    cfg_color_scale: float | None = None,
    color_token_ids: Sequence[int] | None = None,
    color_guidance_rgb_only: bool = False,
    color_guidance_start_t: float = 0.0,
    color_guidance_ramp_t: float = 0.0,
    object_id_scale: float = 1.0,
    palette_condition: Any | None = None,
) -> Any:
    """Integrate the rectified-flow ODE, optionally with factored (base/color) CFG.

    ``factored_cfg`` defaults to False and, when off, this reproduces the original
    all-or-nothing CFG path (``v_uncond + cfg_scale * (v_cond - v_uncond)``) exactly.
    When ``factored_cfg`` is True, guidance is split into a base term (uncond -> color-
    stripped conditioning) and a color term (color-stripped -> full conditioning), each
    with its own scale; see docs/v2_phase0_diagnostics.md.

    v2 Phase 2 Exp A — sampling-only guidance surgery (all defaults-off):

    * ``color_guidance_rgb_only``: zero the alpha channel of the color-axis
      guidance term ``v_cond - v_no_color``, so color guidance affects RGB
      only and does not disturb alpha/silhouette.
    * ``color_guidance_start_t`` / ``color_guidance_ramp_t``: apply color
      guidance only after a flow-time threshold (late-window).  Flow-time *t*
      runs from ~0.017 (near clean) to ~0.983 (near noise) for 30 steps;
      colour guidance is active when ``t >= color_guidance_start_t``, with
      an optional linear ramp of width ``color_guidance_ramp_t``.
    """
    th, _nn_mod = _require_torch()
    model.eval()
    x = initial
    total_steps = max(1, int(steps))
    dt = 1.0 / float(total_steps)
    uncond_caption = caption_tokens.new_full(caption_tokens.shape, int(pad_token_id))
    uncond_semantic = (
        None if semantic_tokens is None else semantic_tokens.new_full(semantic_tokens.shape, int(pad_token_id))
    )
    uncond_structured = _null_structured_conditioning(structured_conditioning)

    no_color_caption = no_color_semantic = no_color_structured = None
    base_scale = color_scale = 0.0
    start_t = float(color_guidance_start_t)
    ramp_t = max(0.0, float(color_guidance_ramp_t))
    rgb_only = bool(color_guidance_rgb_only)
    obj_scale = float(object_id_scale)
    if factored_cfg:
        base_scale = float(cfg_scale if cfg_base_scale is None else cfg_base_scale)
        color_scale = float(cfg_scale if cfg_color_scale is None else cfg_color_scale)
        no_color = strip_color_conditioning(
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            structured_conditioning=structured_conditioning,
            color_token_ids=color_token_ids or (),
            pad_token_id=pad_token_id,
        )
        no_color_caption = no_color["caption_tokens"]
        no_color_semantic = no_color["semantic_tokens"]
        no_color_structured = no_color.get("structured_conditioning")
        use_cfg = True
    else:
        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6

    for index in range(total_steps):
        t_value = (index + 0.5) / float(total_steps)
        t = th.full((int(x.shape[0]),), float(t_value), device=x.device, dtype=x.dtype)
        v_cond = model(
            x,
            t,
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            structured_conditioning=structured_conditioning,
            object_id_scale=obj_scale,
            palette_condition=palette_condition,
        )
        if factored_cfg:
            v_uncond = model(
                x,
                t,
                caption_tokens=uncond_caption,
                semantic_tokens=uncond_semantic,
                structured_conditioning=uncond_structured,
                object_id_scale=obj_scale,
                palette_condition=palette_condition,
            )
            v_no_color = model(
                x,
                t,
                caption_tokens=no_color_caption,
                semantic_tokens=no_color_semantic,
                structured_conditioning=no_color_structured,
                object_id_scale=obj_scale,
                palette_condition=palette_condition,
            )
            # ── guidance surgery ──
            color_axis = v_cond - v_no_color  # (B, 4, 32, 32)
            if rgb_only:
                color_axis = color_axis.clone()
                color_axis[:, 3:4] = 0.0  # zero alpha channel only

            effective_color_scale = color_scale
            if start_t > 0.0:
                # Late-window: active when t <= start_t (closer to clean image).
                # Flow-time t goes from ~0.017 (near clean) to ~0.983 (near noise).
                # Ramp window extends toward cleaner t: [start_t - ramp_t, start_t].
                ramp_start = max(0.0, start_t - ramp_t)
                if t_value > start_t:
                    effective_color_scale = 0.0
                elif ramp_t > 0.0 and t_value >= ramp_start:
                    effective_color_scale = color_scale * ((start_t - t_value) / ramp_t)

            velocity = v_uncond + base_scale * (v_no_color - v_uncond) + effective_color_scale * color_axis
        elif use_cfg:
            v_uncond = model(
                x,
                t,
                caption_tokens=uncond_caption,
                semantic_tokens=uncond_semantic,
                structured_conditioning=uncond_structured,
                palette_condition=palette_condition,
            )
            velocity = v_uncond + float(cfg_scale) * (v_cond - v_uncond)
        else:
            velocity = v_cond
        x = x + dt * velocity
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def color_token_ids_for_tokenizer(tokenizer: SpriteTextTokenizer) -> tuple[int, ...]:
    """Token ids whose text form is a known color word (see COLOR_WORDS)."""

    return tuple(sorted(tokenizer.token_to_id[word] for word in COLOR_WORDS if word in tokenizer.token_to_id))


def strip_color_conditioning(
    *,
    caption_tokens: Any,
    semantic_tokens: Any | None,
    structured_conditioning: Mapping[str, Any] | None,
    color_token_ids: Sequence[int],
    pad_token_id: int,
) -> dict[str, Any]:
    """Return conditioning inputs with color signal removed, without mutating inputs.

    Removes color from every color channel the model consumes: caption/semantic color
    tokens (matched against ``color_token_ids``, typically from
    :func:`color_token_ids_for_tokenizer`) and the structured ``primary_color_id`` /
    ``color_multi_hot`` fields. Category, object, base_object, material, shape,
    function, and style signal are left untouched.
    """

    th, _nn_mod = _require_torch()
    ids = {int(value) for value in color_token_ids}

    def _strip_tokens(tokens: Any) -> Any:
        if tokens is None or not ids:
            return tokens
        cloned = tokens.clone()
        mask = th.zeros_like(cloned, dtype=th.bool)
        for token_id in ids:
            mask = mask | cloned.eq(token_id)
        cloned[mask] = int(pad_token_id)
        return cloned

    stripped_structured: dict[str, Any] | None = None
    if isinstance(structured_conditioning, Mapping):
        stripped_structured = dict(structured_conditioning)
        for key in _COLOR_STRUCTURED_KEYS:
            value = stripped_structured.get(key)
            if isinstance(value, th.Tensor):
                stripped_structured[key] = th.zeros_like(value)

    result: dict[str, Any] = {
        "caption_tokens": _strip_tokens(caption_tokens),
        "semantic_tokens": None if semantic_tokens is None else _strip_tokens(semantic_tokens),
    }
    if stripped_structured is not None:
        result["structured_conditioning"] = stripped_structured
    return result


def apply_conditioning_field_ablations(
    *,
    caption_tokens: Any,
    semantic_tokens: Any | None,
    structured_conditioning: Mapping[str, Any] | None,
    fields: Sequence[str],
    pad_token_id: int,
) -> dict[str, Any]:
    """Null selected conditioning fields at sample time, without mutating inputs.

    ``fields`` is a subset of :data:`NULL_FIELD_CHOICES`. ``caption``/``semantic``
    null the whole caption/semantic token stream; ``structured`` nulls every
    structured field; the remaining names null one structured field group each
    (see :data:`_NULL_FIELD_STRUCTURED_KEYS`). No-op when ``fields`` is empty.
    """

    th, _nn_mod = _require_torch()
    normalized = {str(field).strip().lower() for field in fields if str(field).strip()}
    if not normalized:
        return {
            "caption_tokens": caption_tokens,
            "semantic_tokens": semantic_tokens,
            **({"structured_conditioning": structured_conditioning} if structured_conditioning is not None else {}),
        }
    unknown = normalized - set(NULL_FIELD_CHOICES)
    if unknown:
        raise ValueError(f"Unknown --null-fields values: {sorted(unknown)}; expected one of {NULL_FIELD_CHOICES}")

    result_caption = caption_tokens
    if "caption" in normalized and caption_tokens is not None:
        result_caption = caption_tokens.new_full(caption_tokens.shape, int(pad_token_id))

    result_semantic = semantic_tokens
    if "semantic" in normalized and semantic_tokens is not None:
        result_semantic = semantic_tokens.new_full(semantic_tokens.shape, int(pad_token_id))

    result_structured = structured_conditioning
    if isinstance(structured_conditioning, Mapping):
        if "structured" in normalized:
            result_structured = _null_structured_conditioning(structured_conditioning)
        else:
            keys_to_zero: set[str] = set()
            for field in normalized:
                keys_to_zero.update(_NULL_FIELD_STRUCTURED_KEYS.get(field, ()))
            if keys_to_zero:
                cloned = dict(structured_conditioning)
                for key in keys_to_zero:
                    value = cloned.get(key)
                    if isinstance(value, th.Tensor):
                        cloned[key] = th.zeros_like(value)
                result_structured = cloned

    result: dict[str, Any] = {"caption_tokens": result_caption, "semantic_tokens": result_semantic}
    if result_structured is not None:
        result["structured_conditioning"] = result_structured
    return result


def load_challenger_from_checkpoint(
    ckpt: dict[str, Any],
    *,
    device: Any,
    auxiliary_heads_mode: AuxiliaryHeadsMode | str | None = None,
    legacy_evaluation_import: bool = False,
) -> tuple[RectifiedFlowUNet, SpriteTextTokenizer, str, int]:
    if str(ckpt.get("model_type") or "") != "generator_challenger":
        raise ValueError("checkpoint is not a generator_challenger checkpoint")
    tokenizer = _tokenizer_from_checkpoint(ckpt)
    checkpoint_config = dict(ckpt["model_config"])
    checkpoint_mode, checkpoint_legacy = normalize_auxiliary_heads_mode(checkpoint_config.get("auxiliary_heads_mode"))
    requested_mode = (
        checkpoint_mode if auxiliary_heads_mode is None else normalize_auxiliary_heads_mode(auxiliary_heads_mode)[0]
    )
    if requested_mode is not checkpoint_mode:
        raise RuntimeError(
            "checkpoint architecture is incompatible: "
            f"checkpoint_auxiliary_heads_mode={checkpoint_mode.value!r}, "
            f"requested_auxiliary_heads_mode={requested_mode.value!r}; safe resume/import is blocked"
        )
    manifest = ckpt.get("experiment_manifest")
    if isinstance(manifest, Mapping):
        claimed_mode = manifest.get("auxiliary_heads_mode")
        if claimed_mode is not None and str(claimed_mode) != checkpoint_mode.value:
            raise RuntimeError(
                "checkpoint architecture identity is inconsistent: "
                f"manifest={claimed_mode!r}, model_config={checkpoint_mode.value!r}"
            )
    checkpoint_config["auxiliary_heads_mode"] = None if checkpoint_legacy else checkpoint_mode
    model = RectifiedFlowUNet(**checkpoint_config).to(device)
    expected_keys = set(model.state_dict())
    supplied_keys = set(ckpt["model_state_dict"])
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)

    def key_class(key: str) -> str:
        return "auxiliary" if key.startswith(AUXILIARY_HEAD_PREFIXES) else "base_model"

    legacy_missing_auxiliary = (
        checkpoint_legacy and bool(missing) and not unexpected and all(key_class(key) == "auxiliary" for key in missing)
    )
    if missing or unexpected:
        if legacy_evaluation_import and legacy_missing_auxiliary:
            # Preserve every historical tensor and initialize only the heads
            # absent from that old format, then finish with a strict load.  The
            # resulting model is explicitly evaluation-only and non-promotable.
            merged_state = model.state_dict()
            merged_state.update(ckpt["model_state_dict"])
            model.load_state_dict(merged_state, strict=True)
            model.checkpoint_import_mode = "legacy_missing_auxiliary_evaluation_only"
            model.safe_resume_eligible = False
            model.fair_architecture_comparison_eligible = False
            model.checkpoint_promotion_eligible = False
        else:
            raise RuntimeError(
                "checkpoint model state is incompatible: "
                f"missing={missing}, unexpected={unexpected}, "
                f"missing_key_classes={sorted({key_class(key) for key in missing})}, "
                f"unexpected_key_classes={sorted({key_class(key) for key in unexpected})}"
            )
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    conditioning_mode = validate_conditioning_mode(str(ckpt.get("conditioning_mode") or DEFAULT_CONDITIONING_MODE))
    semantic_max_length = checkpoint_semantic_max_length(ckpt)
    return model, tokenizer, conditioning_mode, semantic_max_length


def load_challenger_prompt_adapter(
    ckpt: dict[str, Any],
    *,
    device: Any,
    steps: int = 30,
    cfg_scale: float = 1.0,
    allow_legacy_conditioning_v1: bool = True,
) -> tuple[ChallengerPromptAdapter, SpriteTextTokenizer, str, int]:
    model, tokenizer, conditioning_mode, semantic_max_length = load_challenger_from_checkpoint(
        ckpt, device=device, legacy_evaluation_import=True
    )
    structured_vocab = structured_vocab_from_checkpoint(ckpt, allow_schema_v1_adapter=allow_legacy_conditioning_v1)
    return (
        ChallengerPromptAdapter(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            pad_token_id=tokenizer.pad_id,
            structured_vocab=structured_vocab,
        ),
        tokenizer,
        conditioning_mode,
        semantic_max_length,
    )


def _write_challenger_sample_sheet(
    model: RectifiedFlowUNet,
    batch: dict[str, Any],
    path: Path,
    *,
    conditioning_mode: str,
    pad_token_id: int,
    steps: int,
) -> None:
    th, _nn_mod = _require_torch()
    model.eval()
    with th.no_grad():
        initial = th.zeros_like(batch["rgba"])
        inputs = apply_conditioning_mode(
            caption_tokens=batch["caption_tokens"],
            semantic_tokens=batch["semantic_tokens"],
            mode=conditioning_mode,
            pad_token_id=pad_token_id,
            structured_conditioning=_structured_conditioning_from_batch(batch),
        )
        rgba = integrate_rectified_flow(
            model,
            initial,
            caption_tokens=inputs["caption_tokens"],
            semantic_tokens=inputs["semantic_tokens"],
            structured_conditioning=inputs.get("structured_conditioning"),
            palette_condition=batch.get("palette_condition"),
            steps=steps,
            cfg_scale=1.0,
            pad_token_id=pad_token_id,
        )
    save_rgba_contact_sheet(outputs=_rgba_to_logit_outputs(rgba), batch=batch, path=path)
    model.train()


def _init_ema_state(model: RectifiedFlowUNet) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _update_ema_state(ema_state: dict[str, Any], model: RectifiedFlowUNet, *, decay: float) -> None:
    """Reference (one-tensor-at-a-time) EMA update.

    Kept for external callers and as the ground truth ``_update_ema_state_fast``
    is checked against in tests/test_training_speed_options.py. The training
    hot loop uses the cached foreach path below instead.
    """
    th, _nn_mod = _require_torch()
    clipped_decay = min(1.0, max(0.0, float(decay)))
    with th.no_grad():
        for key, value in model.state_dict().items():
            current = value.detach()
            if key not in ema_state:
                ema_state[key] = current.clone()
                continue
            target = ema_state[key]
            if target.dtype.is_floating_point:
                target.mul_(clipped_decay).add_(
                    current.to(device=target.device, dtype=target.dtype), alpha=1.0 - clipped_decay
                )
            else:
                target.copy_(current.to(device=target.device, dtype=target.dtype))


def _init_ema_fast_state(ema_state: dict[str, Any], model: RectifiedFlowUNet) -> dict[str, Any]:
    """Cache tensor-pair references for ``_update_ema_state_fast``.

    ``model.state_dict()`` values share storage with the model's own parameters
    and buffers, and the training loop never reassigns those tensors after
    ``model.to(device)`` (the optimizer mutates their storage in place), so
    caching the references once here stays valid for the rest of the run. Do
    not call this again mid-run unless the model's parameters/buffers are
    replaced wholesale.
    """
    source = model.state_dict()
    float_keys = [key for key, value in ema_state.items() if value.dtype.is_floating_point]
    nonfloat_keys = [key for key in ema_state if key not in float_keys]
    return {
        "ema_float": [ema_state[key] for key in float_keys],
        "src_float": [source[key] for key in float_keys],
        "ema_nonfloat": [ema_state[key] for key in nonfloat_keys],
        "src_nonfloat": [source[key] for key in nonfloat_keys],
    }


def _update_ema_state_fast(cache: dict[str, Any], *, decay: float) -> None:
    """Foreach-batched EMA update; numerically identical to ``_update_ema_state``.

    Two ``torch._foreach_*`` calls update every floating-point tensor in one
    shot instead of a Python-level loop with ~2 kernel launches per tensor;
    non-floating-point tensors (none currently, but handled for parity with
    ``_update_ema_state``) are copied directly.
    """
    th, _nn_mod = _require_torch()
    clipped_decay = min(1.0, max(0.0, float(decay)))
    with th.no_grad():
        if cache["ema_float"]:
            th._foreach_mul_(cache["ema_float"], clipped_decay)
            th._foreach_add_(cache["ema_float"], cache["src_float"], alpha=1.0 - clipped_decay)
        for target, source in zip(cache["ema_nonfloat"], cache["src_nonfloat"], strict=False):
            target.copy_(source)


def _loss_metrics(losses: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in losses.items():
        if not str(key).startswith("loss"):
            continue
        if hasattr(value, "detach"):
            metrics[str(key)] = float(value.detach().cpu())
    return metrics


def _save_checkpoint(
    path: Path,
    *,
    model: RectifiedFlowUNet,
    optimizer: Any,
    tokenizer: SpriteTextTokenizer,
    config_json: dict[str, Any],
    step: int,
    model_state_dict: Mapping[str, Any] | None = None,
    ema_decay: float = 0.0,
    checkpoint_variant: str = "last",
    ema_weights: bool = False,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    ema_state: Mapping[str, Any] | None = None,
    metrics_summary: Mapping[str, Any] | None = None,
    data_position: Mapping[str, Any] | None = None,
    sampler_state: Mapping[str, Any] | None = None,
    immutable_writer: _CampaignRunWriter | None = None,
) -> str | None:
    th, _nn_mod = _require_torch()
    from spritelab.training.experiment_system import capture_rng_state

    checkpoint = {
        "model_type": "generator_challenger",
        "architecture": "rectified_flow",
        "model_state_dict": model.state_dict() if model_state_dict is None else dict(model_state_dict),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config(),
        "auxiliary_heads_mode": model.auxiliary_heads_mode.value,
        "legacy_auxiliary_heads_adapter": bool(model.legacy_auxiliary_heads_adapter),
        "vocab": tokenizer.to_json_dict(),
        "train_config": config_json,
        "structured_conditioning_vocab": config_json.get("structured_conditioning_vocab"),
        "structured_vocab_sizes": config_json.get("structured_vocab_sizes"),
        "conditioning_mode": str(config_json.get("conditioning_mode", DEFAULT_CONDITIONING_MODE)),
        "cfg_dropout": float(config_json.get("cfg_dropout", 0.0)),
        "structured_field_dropout": float(config_json.get("structured_field_dropout", 0.0)),
        "structured_fields_enabled": bool(config_json.get("structured_fields_enabled", False)),
        "ema_decay": float(ema_decay),
        "ema_weights": bool(ema_weights),
        "checkpoint_variant": str(checkpoint_variant),
        "foreground_rgb_loss_weight": float(config_json.get("foreground_rgb_loss_weight", 1.0)),
        "background_rgb_loss_weight": float(config_json.get("background_rgb_loss_weight", 1.0)),
        "palette_loss_weight": float(config_json.get("palette_loss_weight", 0.0)),
        "palette_loss_temperature": float(config_json.get("palette_loss_temperature", 0.05)),
        "dataset": str(config_json.get("dataset_dir") or config_json.get("dataset") or ""),
        "training_manifest": str(config_json.get("training_manifest") or ""),
        "seed": int(config_json.get("seed") or 0),
        "step": int(step),
        "global_step": int(step),
        "epoch": int((sampler_state or {}).get("epoch", config_json.get("epoch", 0))),
        "data_position": dict(
            data_position
            or {
                "batch_index": (sampler_state or {}).get("batch_index"),
                "sample_cursor": (sampler_state or {}).get("sample_cursor"),
            }
        ),
        "sampler_state": None if sampler_state is None else dict(sampler_state),
        "ema_state_dict": None if ema_state is None else dict(ema_state),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "rng_states": capture_rng_state(),
        "experiment_manifest": config_json.get("experiment_manifest"),
        "dataset_hashes": {
            "dataset_manifest_hash": (config_json.get("experiment_manifest") or {}).get("dataset_manifest_hash"),
            "split_manifest_hash": (config_json.get("experiment_manifest") or {}).get("split_manifest_hash"),
        },
        "conditioning_vocabulary": {
            "text": tokenizer.to_json_dict(),
            "structured": config_json.get("structured_conditioning_vocab"),
        },
        "conditioning_schema_version": config_json.get("conditioning_schema_version")
        or ((config_json.get("experiment_manifest") or {}).get("conditioning_schema") or {}).get("version"),
        "code_version": (config_json.get("experiment_manifest") or {}).get("software_version"),
        "training_metrics_summary": dict(metrics_summary or {}),
        "checkpoint_lineage": (config_json.get("experiment_manifest") or {}).get("checkpoint_lineage", []),
        "checkpoint_type": "generator_challenger_rectified_flow_v1",
    }
    if immutable_writer is not None:
        if path.parent != immutable_writer.physical_root or path.name != Path(path.name).name:
            raise UnsafeFilesystemOperation("campaign checkpoint path is outside the retained output root")
        return immutable_writer.write_torch_checkpoint(path.name, checkpoint, th)
    path.parent.mkdir(parents=True, exist_ok=True)
    th.save(checkpoint, path)
    return None


def _campaign_checkpoint_sidecar(
    contract: Mapping[str, Any],
    experiment_manifest: Mapping[str, Any] | None,
    *,
    step: int,
    checkpoint_name: str,
    checkpoint_sha256: str,
    scheduler_present: bool,
    ema_present: bool,
    sampler_state: Mapping[str, Any],
) -> dict[str, Any]:
    from spritelab.training.campaign import RESUME_CHECKPOINT_SCHEMA_VERSION
    from spritelab.training.experiment_system import stable_hash

    manifest_identity = None if experiment_manifest is None else experiment_manifest.get("experiment_hash")
    if not isinstance(manifest_identity, str) or _CONCRETE_SHA256.fullmatch(manifest_identity) is None:
        if experiment_manifest is None:
            raise ValueError("campaign checkpoint requires a concrete experiment manifest")
        manifest_identity = stable_hash(experiment_manifest)
    if sampler_state.get("dataloader_generator_state") is None:
        raise ValueError("campaign checkpoint has no exact data-loader generator state")
    state_presence = {
        "model_state_dict": True,
        "optimizer_state_dict": True,
        "scheduler_state_dict": scheduler_present,
        "ema_state_dict": ema_present,
        "rng_states": True,
        "sampler_state": True,
        "dataloader_generator_state": True,
    }
    schedule_required = str(contract["schedule_name"]).strip().lower() != "none"
    ema_required = contract["evaluation_ema_policy"] in {"ema", "both"}
    if schedule_required and not scheduler_present:
        raise ValueError("campaign checkpoint is missing required scheduler state")
    if ema_required and not ema_present:
        raise ValueError("campaign checkpoint is missing required EMA state")
    return {
        "optimizer_step": int(step),
        "campaign_identity": contract["campaign_identity"],
        "run_identity": contract["run_identity"],
        "resumability_metadata": {
            "schema_version": RESUME_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_relative_path": checkpoint_name,
            "checkpoint_content_sha256": checkpoint_sha256,
            "source_checkpoint_identity": checkpoint_sha256,
            "target_runtime_identity": contract["run_identity"],
            "experiment_manifest_identity": manifest_identity,
            "exact_replay_eligible": True,
            "unsafe_resume": False,
            "max_optimizer_steps": contract["max_optimizer_steps"],
            "gradient_accumulation_position": 0,
            "state_presence": state_presence,
        },
    }


def _campaign_evaluation_record(
    contract: Mapping[str, Any],
    *,
    optimizer_step: int,
    metrics_by_weight: Mapping[str, Any],
) -> dict[str, Any]:
    from spritelab.training.experiment_system import stable_hash

    base = {
        "schema_version": CAMPAIGN_EVALUATION_RECORD_SCHEMA_VERSION,
        "campaign_identity": contract["campaign_identity"],
        "run_identity": contract["run_identity"],
        "seed": contract["seed"],
        "optimizer_step": int(optimizer_step),
        "evaluated_weights": sorted(str(key) for key in metrics_by_weight),
        "metrics_by_weight": _jsonable(dict(metrics_by_weight)),
    }
    return {**base, "record_identity": stable_hash(base)}


def _load_campaign_evaluations(
    writer: _CampaignRunWriter,
    contract: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    from spritelab.training.experiment_system import stable_hash

    expected = {int(step) for step in contract["expected_evaluation_steps"]}
    found: dict[int, dict[str, Any]] = {}
    for name in writer.names():
        match = re.fullmatch(r"evaluation_step_(\d+)\.json", name)
        if match is None:
            continue
        step = int(match.group(1))
        if step not in expected or step in found:
            raise UnsafeFilesystemOperation(f"campaign output has an off-schedule evaluation: {name}")
        content = writer.read_bytes(name, max_bytes=16 * 1024 * 1024)
        value = strict_json_loads(content)
        if not isinstance(value, Mapping):
            raise UnsafeFilesystemOperation(f"campaign evaluation is malformed: {name}")
        record = dict(value)
        identity = record.pop("record_identity", None)
        required_weights = (
            {"live", "ema"} if contract["evaluation_ema_policy"] == "both" else {str(contract["evaluation_ema_policy"])}
        )
        if (
            value.get("schema_version") != CAMPAIGN_EVALUATION_RECORD_SCHEMA_VERSION
            or value.get("campaign_identity") != contract["campaign_identity"]
            or value.get("run_identity") != contract["run_identity"]
            or value.get("seed") != contract["seed"]
            or value.get("optimizer_step") != step
            or set(value.get("evaluated_weights") or ()) != required_weights
            or not isinstance(value.get("metrics_by_weight"), Mapping)
            or set(value["metrics_by_weight"]) != required_weights
            or identity != stable_hash(record)
            or content != _canonical_pretty_json_bytes(value)
        ):
            raise UnsafeFilesystemOperation(f"campaign evaluation identity changed: {name}")
        found[step] = dict(value)
    return found


def _publish_campaign_completion(
    writer: _CampaignRunWriter,
    contract: Mapping[str, Any],
    *,
    experiment_manifest: Mapping[str, Any] | None,
    report: Mapping[str, Any],
    final_train_losses: Mapping[str, Any],
    validation_losses: Mapping[str, Any] | None,
    ema_validation_losses: Mapping[str, Any] | None,
    evaluations: Mapping[int, Mapping[str, Any]],
    resume_mismatches: Sequence[str],
    determinism_report: Mapping[str, Any],
) -> None:
    from spritelab.training.campaign import ARTIFACT_MANIFEST_SCHEMA_VERSION, PER_RUN_ARTIFACTS
    from spritelab.training.experiment_system import stable_hash

    expected_checkpoints = [int(step) for step in contract["expected_checkpoint_steps"]]
    expected_evaluations = [int(step) for step in contract["expected_evaluation_steps"]]
    if sorted(evaluations) != expected_evaluations:
        raise UnsafeFilesystemOperation("campaign evaluation evidence does not exactly match the schedule")
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_names: set[str] = set()
    for step in expected_checkpoints:
        checkpoint_name = f"checkpoint_step_{step:06d}.pt"
        sidecar_name = f"checkpoint_step_{step:06d}.json"
        checkpoint_names.add(checkpoint_name)
        checkpoint_sha256 = writer.file_sha256(checkpoint_name)
        sidecar_value = strict_json_loads(writer.read_bytes(sidecar_name, max_bytes=4 * 1024 * 1024))
        if not isinstance(sidecar_value, Mapping):
            raise UnsafeFilesystemOperation(f"campaign checkpoint sidecar is malformed: {sidecar_name}")
        metadata = sidecar_value.get("resumability_metadata")
        if (
            sidecar_value.get("optimizer_step") != step
            or sidecar_value.get("campaign_identity") != contract["campaign_identity"]
            or sidecar_value.get("run_identity") != contract["run_identity"]
            or not isinstance(metadata, Mapping)
            or metadata.get("checkpoint_relative_path") != checkpoint_name
            or metadata.get("checkpoint_content_sha256") != checkpoint_sha256
            or metadata.get("source_checkpoint_identity") != checkpoint_sha256
            or metadata.get("target_runtime_identity") != contract["run_identity"]
            or metadata.get("exact_replay_eligible") is not True
            or metadata.get("unsafe_resume") is not False
        ):
            raise UnsafeFilesystemOperation(f"campaign checkpoint sidecar identity changed: {sidecar_name}")
        checkpoint_rows.append(
            {
                "optimizer_step": step,
                "relative_path": checkpoint_name,
                "content_sha256": checkpoint_sha256,
                "resumability_metadata_identity": stable_hash(metadata),
            }
        )
    on_disk_checkpoints = {
        name for name in writer.names() if re.fullmatch(r"checkpoint(?:_step_\d+)?(?:_ema)?\.(?:bin|pt|pth|ckpt)", name)
    }
    if on_disk_checkpoints != checkpoint_names:
        raise UnsafeFilesystemOperation("campaign checkpoint set contains an off-schedule or missing file")

    common = {
        "campaign_identity": contract["campaign_identity"],
        "run_identity": contract["run_identity"],
        "seed": contract["seed"],
    }
    evaluated_weights = sorted(
        {str(weight) for evaluation in evaluations.values() for weight in evaluation.get("evaluated_weights", ())}
    )
    metric_definitions = {
        "loss": {"unit": "mean rectified-flow objective", "split": "validation"},
        "loss_velocity": {"unit": "mean velocity MSE", "split": "validation"},
    }
    experiment_identity = None if experiment_manifest is None else experiment_manifest.get("experiment_hash")
    if not isinstance(experiment_identity, str) or _CONCRETE_SHA256.fullmatch(experiment_identity) is None:
        if experiment_manifest is None:
            raise UnsafeFilesystemOperation("campaign completion has no experiment manifest")
        experiment_identity = stable_hash(experiment_manifest)
    training_metrics_definition = {"unit": "mean loss", "split": "train", "aggregation": "optimizer_step"}
    validation_metrics_definition = {
        "unit": "mean loss",
        "split": "validation",
        "aggregation": "fixed_schedule",
    }
    ema_metrics_definition = {"weights": "ema", "split": "validation", "aggregation": "fixed_schedule"}
    live_metrics_definition = {"weights": "live", "split": "validation", "aggregation": "fixed_schedule"}
    raw_metrics_sha256 = writer.file_sha256("train_metrics.jsonl")
    train_report_sha256 = writer.file_sha256("train_report.json")
    values: dict[str, dict[str, Any]] = {
        "experiment_manifest": {
            **common,
            "experiment_manifest_identity": experiment_identity,
            "experiment_manifest": deepcopy(dict(experiment_manifest or {})),
        },
        "resolved_config": {
            **common,
            "resolved_config_sha256": contract["resolved_config_sha256"],
            "execution_contract_sha256": contract["execution_contract_sha256"],
            "resolved_config": deepcopy(dict(contract["resolved_config"])),
        },
        "checkpoint_series": {
            **common,
            "checkpoint_steps": expected_checkpoints,
            "checkpoints": checkpoint_rows,
        },
        "training_metrics": {
            **common,
            "definition": training_metrics_definition,
            "steps_completed": int(report["steps_completed"]),
            "final": _jsonable(dict(final_train_losses)),
            "raw_metrics_sha256": raw_metrics_sha256,
            "train_report_sha256": train_report_sha256,
        },
        "validation_metrics": {
            **common,
            "definition": validation_metrics_definition,
            "final": None if validation_losses is None else _jsonable(dict(validation_losses)),
        },
        "ema_metrics": {
            **common,
            "definition": ema_metrics_definition,
            "evaluations": [
                {"optimizer_step": step, "metrics": evaluation["metrics_by_weight"].get("ema")}
                for step, evaluation in sorted(evaluations.items())
            ],
        },
        "live_metrics": {
            **common,
            "definition": live_metrics_definition,
            "evaluations": [
                {"optimizer_step": step, "metrics": evaluation["metrics_by_weight"].get("live")}
                for step, evaluation in sorted(evaluations.items())
            ],
        },
        "evaluation_reports": {
            **common,
            "evaluation_steps": expected_evaluations,
            "evaluated_weights": evaluated_weights,
            "metric_definitions": metric_definitions,
            "evaluations": [deepcopy(dict(evaluations[step])) for step in expected_evaluations],
        },
        "effective_pass_report": {
            **common,
            "max_optimizer_steps": contract["max_optimizer_steps"],
            "micro_batch_size": int(report["batch_size"]),
            "effective_batch_size": int(report["batch_size"]),
            "train_records": int(report["train_records"]),
            "optimizer_step_sprite_exposures": int(report["steps_completed"]) * int(report["batch_size"]),
        },
        "resume_report": {
            **common,
            "resume_used": report.get("resume_from") is not None,
            "source_checkpoint_sha256": report.get("resume_checkpoint_sha256"),
            "compatibility_mismatches": list(resume_mismatches),
            "unsafe_resume": False,
            "exact_replay_eligible": not resume_mismatches,
        },
        "environment_identity": {
            **common,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "torch_version": None if torch is None else str(torch.__version__),
            "device": str(report["device"]),
            "determinism": _jsonable(dict(determinism_report)),
        },
        "code_identity": {
            **common,
            "training_code_identity_sha256": contract["training_code_identity_sha256"],
        },
        "run_completion_marker": {
            **common,
            "complete": True,
            "failed": False,
            "partial": False,
            "final_optimizer_step": expected_checkpoints[-1],
            "checkpoint_set_identity": stable_hash(checkpoint_rows),
            "evaluation_set_identity": stable_hash([dict(evaluations[step]) for step in expected_evaluations]),
            "resolved_config_sha256": contract["resolved_config_sha256"],
            "execution_contract_sha256": contract["execution_contract_sha256"],
            "experiment_manifest_identity": experiment_identity,
            "training_code_identity_sha256": contract["training_code_identity_sha256"],
        },
    }
    expected_value_names = set(PER_RUN_ARTIFACTS) - {"run_identity", "artifact_manifest"}
    if set(values) != expected_value_names:
        raise UnsafeFilesystemOperation("campaign completion artifact implementation is incomplete")
    serialized = {name: _canonical_pretty_json_bytes(value) for name, value in values.items()}
    artifact_entries: list[dict[str, Any]] = []
    run_identity_content = writer.read_bytes("run_identity.json", max_bytes=1024 * 1024)
    serialized_with_identity = {**serialized, "run_identity": run_identity_content}
    for name in sorted(serialized_with_identity):
        entry = {
            "artifact_type": name,
            "relative_path": f"{name}.json",
            "content_sha256": hashlib.sha256(serialized_with_identity[name]).hexdigest(),
            "producing_run_identity": contract["run_identity"],
            "seed": contract["seed"],
            "final_role": "required_run_artifact",
        }
        if name in {"training_metrics", "validation_metrics", "ema_metrics", "live_metrics"}:
            entry["metric_definition_identity"] = stable_hash(values[name]["definition"])
        elif name == "evaluation_reports":
            entry["metric_definition_identity"] = stable_hash(values[name]["metric_definitions"])
        artifact_entries.append(entry)
    for row in checkpoint_rows:
        artifact_entries.append(
            {
                "artifact_type": "checkpoint",
                "relative_path": row["relative_path"],
                "content_sha256": row["content_sha256"],
                "producing_run_identity": contract["run_identity"],
                "seed": contract["seed"],
                "scheduled_step": row["optimizer_step"],
            }
        )
    artifact_manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        **common,
        "artifacts": artifact_entries,
    }
    for name in sorted(values):
        if name == "run_completion_marker":
            continue
        writer.write_json_idempotent(f"{name}.json", values[name])
    writer.write_json_idempotent("artifact_manifest.json", artifact_manifest)
    for entry in artifact_entries:
        relative = str(entry["relative_path"])
        if relative == "run_completion_marker.json":
            continue
        if writer.file_sha256(relative) != entry["content_sha256"]:
            raise UnsafeFilesystemOperation(f"campaign completion artifact changed before commit: {relative}")
    writer.write_json_idempotent("run_completion_marker.json", values["run_completion_marker"])
    marker_entry = next(entry for entry in artifact_entries if entry["relative_path"] == "run_completion_marker.json")
    if writer.file_sha256("run_completion_marker.json") != marker_entry["content_sha256"]:
        raise UnsafeFilesystemOperation("campaign completion marker changed during commit")


def _apply_cfg_dropout(inputs: dict[str, Any], *, dropout: float, pad_token_id: int) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    probability = float(dropout)
    if probability <= 0.0:
        return inputs
    caption = inputs["caption_tokens"]
    mask = th.rand((int(caption.shape[0]),), device=caption.device) < min(1.0, probability)
    if not bool(mask.any()):
        return inputs
    caption = caption.clone()
    caption[mask] = int(pad_token_id)
    semantic = inputs.get("semantic_tokens")
    if semantic is not None:
        semantic = semantic.clone()
        semantic[mask] = int(pad_token_id)
    structured = inputs.get("structured_conditioning")
    if isinstance(structured, Mapping):
        structured = _masked_structured_conditioning(structured, mask)
    result = {"caption_tokens": caption, "semantic_tokens": semantic}
    result["cfg_dropout_fraction"] = float(mask.float().mean().detach().cpu())
    if structured is not None:
        result["structured_conditioning"] = structured
    return result


def _apply_structured_field_dropout(
    inputs: dict[str, Any],
    *,
    dropout: float,
    training: bool,
    dropout_rates: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Apply per-group structured field dropout during training.

    If ``dropout_rates`` is provided, it maps group names to per-group rates;
    groups not listed fall back to the scalar ``dropout`` value.  Otherwise
    every group uses the same scalar ``dropout`` (existing behaviour).
    """
    th, _nn_mod = _require_torch()
    structured = inputs.get("structured_conditioning")
    if not training or not isinstance(structured, Mapping):
        return inputs

    batch_size, device = _structured_batch_shape(structured)
    if batch_size <= 0 or device is None:
        return inputs

    per_group: dict[str, float] = {}
    if dropout_rates:
        per_group = {str(k): float(v) for k, v in dropout_rates.items()}

    dropped: dict[str, Any] = {
        str(key): value.clone() if isinstance(value, th.Tensor) else value for key, value in structured.items()
    }
    any_masked = False
    for group_name, fields in STRUCTURED_DROPOUT_GROUPS:
        present_fields = [field for field in fields if isinstance(dropped.get(field), th.Tensor)]
        if not present_fields:
            continue
        group_rate = per_group.get(group_name, dropout)
        probability = min(1.0, max(0.0, float(group_rate)))
        if probability <= 0.0:
            continue
        mask = th.rand((batch_size,), device=device) < probability
        if not bool(mask.any()):
            continue
        any_masked = True
        for field in present_fields:
            tensor = dropped[field]
            tensor[mask] = 0
            if field in MULTI_HOT_FIELDS and tensor.ndim >= 2 and tensor.shape[1] > 0:
                tensor[mask, 0] = 1
    if not any_masked:
        return inputs
    result = dict(inputs)
    result["structured_conditioning"] = dropped
    return result


def _structured_batch_shape(structured: Mapping[str, Any]) -> tuple[int, Any | None]:
    th, _nn_mod = _require_torch()
    for value in structured.values():
        if isinstance(value, th.Tensor) and value.ndim >= 1:
            return int(value.shape[0]), value.device
    return 0, None


def _select_training_subset(
    token_rows: list[dict[str, Any]],
    config: ChallengerTrainConfig,
    *,
    split: str,
) -> OverfitSubsetSelection | None:
    sprite_ids = read_sprite_id_list(config.sprite_id_list) if config.sprite_id_list is not None else None
    if config.max_train_sprites is None and sprite_ids is None:
        return None
    return select_overfit_subset(
        token_rows,
        count=config.max_train_sprites,
        sprite_ids=sprite_ids,
        split=split,
        seed=config.seed,
    )


def _resolve_validation_mode(mode: str, subset_selection: OverfitSubsetSelection | None) -> str:
    normalized = str(mode or "auto").strip().lower().replace("-", "_")
    if normalized == "auto":
        return "same" if subset_selection is not None else "val"
    if normalized not in {"val", "same", "none"}:
        raise ValueError("validation_mode must be one of: auto, val, same, none")
    return normalized


def _normalize_structured_vocab_sizes(value: Mapping[str, int] | None) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    keys = (
        "category_vocab_size",
        "object_vocab_size",
        "base_object_vocab_size",
        "color_vocab_size",
        "material_vocab_size",
        "shape_vocab_size",
        "function_vocab_size",
        "style_vocab_size",
    )
    result = {key: max(1, int(value.get(key) or 0)) for key in keys}
    if "field_status_vocab_size" in value:
        result["field_status_vocab_size"] = max(1, int(value.get("field_status_vocab_size") or 0))
    return result


def _structured_conditioning_from_batch(batch: Mapping[str, Any]) -> dict[str, Any] | None:
    if not all(key in batch for key in STRUCTURED_BATCH_KEYS):
        return None
    return {key.removeprefix("structured_"): batch[key] for key in STRUCTURED_BATCH_KEYS}


def _structured_conditioning_for_records(
    records: Sequence[Mapping[str, Any]],
    *,
    structured_vocab: StructuredConditioningVocab | None,
    device: Any,
) -> dict[str, Any] | None:
    if structured_vocab is None:
        return None
    th, _nn_mod = _require_torch()
    encoded = [encode_structured_conditioning(record, structured_vocab) for record in records]
    result: dict[str, Any] = {}
    for field in ("category_id", "object_id", "base_object_id", "primary_color_id"):
        result[field] = th.as_tensor([int(row[field]) for row in encoded], dtype=th.long, device=device)
    for field in MULTI_HOT_FIELDS:
        result[field] = th.as_tensor([row[field] for row in encoded], dtype=th.float32, device=device)
    for field in STATUS_FIELDS:
        result[field] = th.as_tensor([int(row[field]) for row in encoded], dtype=th.long, device=device)
    return result


def _select_palette_conditions(
    records: Sequence[Mapping[str, Any]],
    *,
    library: PaletteConditionLibrary | None,
    mode: str,
    exclude_exact_prompt_target: bool,
) -> list[Any | None]:
    """Resolve source/retrieved palette rows in prompt order for sampling."""
    if mode == "none" or library is None:
        return [None] * len(records)
    by_sprite = {entry.sprite_id: entry for entry in library.entries if entry.sprite_id}
    selected: list[Any | None] = []
    for record in records:
        if mode == "source":
            target = str(record.get("sprite_id") or record.get("prompt_id") or "")
            entry = by_sprite.get(target)
            if entry is None:
                entry = library.retrieve(record, exclude_sprite_id=exclude_exact_prompt_target)
        else:
            entry = library.retrieve(record, exclude_sprite_id=exclude_exact_prompt_target)
        selected.append(entry)
    return selected


def _null_structured_conditioning(structured: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(structured, Mapping):
        return None
    th, _nn_mod = _require_torch()
    result = {
        str(key): th.zeros_like(value) if isinstance(value, th.Tensor) else value for key, value in structured.items()
    }
    for key in MULTI_HOT_FIELDS:
        value = result.get(key)
        if isinstance(value, th.Tensor) and value.ndim >= 2 and value.shape[1] > 0:
            value[:, 0] = 1
    return result


def _masked_structured_conditioning(structured: Mapping[str, Any], mask: Any) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    result: dict[str, Any] = {}
    for key, value in structured.items():
        if isinstance(value, th.Tensor):
            cloned = value.clone()
            cloned[mask] = 0
            if str(key) in MULTI_HOT_FIELDS and cloned.ndim >= 2 and cloned.shape[1] > 0:
                cloned[mask, 0] = 1
            result[str(key)] = cloned
        else:
            result[str(key)] = value
    return result


def _parse_null_fields(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    else:
        parts = [str(part).strip().lower() for part in value if str(part).strip()]
    unknown = sorted(set(parts) - set(NULL_FIELD_CHOICES))
    if unknown:
        raise ValueError(f"Unknown --null-fields values: {unknown}; expected one of {NULL_FIELD_CHOICES}")
    return tuple(parts)


def _parse_channel_mults(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return (1, 2, 4)
        return tuple(max(1, int(part)) for part in parts)
    return tuple(max(1, int(part)) for part in value)


def _sample_initial_noise(batch_size: int, *, device: Any, seed: int | None = None) -> Any:
    th, _nn_mod = _require_torch()
    generator = None
    if seed is not None:
        try:
            generator = th.Generator(device=device)
        except TypeError:  # pragma: no cover - older torch fallback.
            generator = th.Generator()
        generator.manual_seed(int(seed))
    return th.randn(int(batch_size), 4, SPRITE_SIZE, SPRITE_SIZE, device=device, generator=generator)


def _sinusoidal_embedding(t: Any, dim: int) -> Any:
    th, _nn_mod = _require_torch()
    half = max(1, int(dim) // 2)
    freqs = th.exp(-math.log(10000.0) * th.arange(half, device=t.device, dtype=t.dtype) / max(1, half - 1))
    args = t[:, None] * freqs[None, :]
    emb = th.cat([th.sin(args), th.cos(args)], dim=1)
    if emb.shape[1] < int(dim):
        emb = th.nn.functional.pad(emb, (0, int(dim) - int(emb.shape[1])))
    return emb[:, : int(dim)]


def _group_norm(channels: int) -> Any:
    _th, nn_mod = _require_torch()
    groups = min(8, int(channels))
    while int(channels) % groups != 0 and groups > 1:
        groups -= 1
    return nn_mod.GroupNorm(groups, int(channels))


def _rgba_to_logit_outputs(rgba: Any) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    value = rgba.clamp(1e-4, 1.0 - 1e-4)
    logits = th.log(value / (1.0 - value))
    return {"rgb_logits": logits[:, :3], "alpha_logits": logits[:, 3:4]}


def _write_contact_sheet_label_mapping(out_dir: Path, records: list[dict[str, Any]], *, label_mode: str) -> None:
    rows = []
    for record in records:
        paths = record.get("paths") if isinstance(record.get("paths"), dict) else {}
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "sample_filename": paths.get("indexed_png") or paths.get("hard_rgba") or paths.get("raw_rgba"),
                "prompt": record.get("prompt"),
                "prompt_id": record.get("prompt_id"),
                "seed": record.get("seed"),
                "noise_seed": record.get("noise_seed"),
                "conditioning": record.get("conditioning_mode"),
                "label_mode": label_mode,
                "nearest_source_object": record.get("nearest_source_object"),
                "nearest_source_category": record.get("nearest_source_category"),
            }
        )
    (out_dir / "contact_sheet_labels.json").write_text(
        json.dumps(_jsonable(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Contact Sheet Labels", ""]
    for row in rows:
        lines.append(
            f"- `{row.get('sample_id')}` `{row.get('prompt_id')}` seed={row.get('noise_seed')}: {row.get('prompt')}"
        )
    lines.append("")
    (out_dir / "contact_sheet_labels.md").write_text("\n".join(lines), encoding="utf-8")


def _matches_caption_policy(record: dict[str, Any], caption_policy_filter: str | None) -> bool:
    if not caption_policy_filter:
        return True
    audit = record.get("audit") if isinstance(record.get("audit"), dict) else {}
    return str(audit.get("caption_policy", "")) == str(caption_policy_filter)


def _write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def _set_seed(seed: int) -> None:
    th, _nn_mod = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def _normalize_checkpoint_steps(steps: Sequence[int] | None, *, max_steps: int) -> tuple[int, ...]:
    if not steps:
        return ()
    normalized: set[int] = set()
    limit = int(max_steps)
    for raw_step in steps:
        step = int(raw_step)
        if step <= 0:
            raise ValueError("checkpoint_steps must contain positive step numbers")
        if step > limit:
            raise ValueError(f"checkpoint step {step} exceeds max_steps={limit}")
        normalized.add(step)
    return tuple(sorted(normalized))


def _should_save_checkpoint_step(step: int, *, save_every: int, checkpoint_steps: Sequence[int]) -> bool:
    return (int(save_every) > 0 and int(step) % int(save_every) == 0) or int(step) in set(checkpoint_steps)


from spritelab.training.report_utils import jsonable as _jsonable


def main_train(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a conditional generator challenger.")
    parser.add_argument("--dataset", required=True, type=Path, dest="dataset_dir")
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--architecture", default="rectified_flow")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", "--learning-rate", type=float, default=2e-4, dest="learning_rate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--conditioning-mode", choices=CONDITIONING_MODES, default=DEFAULT_CONDITIONING_MODE)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--structured-field-dropout", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--foreground-rgb-loss-weight", type=float, default=1.0)
    parser.add_argument("--background-rgb-loss-weight", type=float, default=1.0)
    parser.add_argument("--palette-loss-weight", type=float, default=0.0)
    parser.add_argument("--palette-loss-temperature", type=float, default=0.05)
    parsed = parser.parse_args(argv)
    report = run_challenger_training(ChallengerTrainConfig(**vars(parsed)))
    print(f"Initial train loss: {report['initial_train_loss']:.6f}")
    print(f"Final train loss: {report['final_train_loss']:.6f}")
    print(f"Outputs written to {parsed.out_dir}")


def main_sample(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sample and canonicalize a generator challenger.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, dest="out_dir")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--max-colors", type=int, default=32)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--project-palette", action="store_true", default=False)
    parser.add_argument("--project-palette-target-colors", type=int, default=16)
    parser.add_argument("--project-palette-min-pixel-share", type=float, default=0.01)
    parser.add_argument("--project-palette-method", choices=["deterministic_kmeans"], default="deterministic_kmeans")
    parsed = parser.parse_args(argv)
    report = run_sample_generator_challenger(ChallengerSampleConfig(**vars(parsed)))
    print(f"Generated samples: {report['sample_count']}")
    print(f"Outputs written to {parsed.out_dir}")
