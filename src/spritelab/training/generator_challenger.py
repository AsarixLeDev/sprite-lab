"""Small conditional rectified-flow challenger for 32x32 RGBA sprites."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised when torch is absent or broken.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from spritelab.training.conditioning import (
    CONDITIONING_MODES,
    DEFAULT_CONDITIONING_MODE,
    apply_conditioning_mode,
    checkpoint_semantic_max_length,
    uses_structured_conditioning,
    validate_conditioning_mode,
)
from spritelab.training.data import SpriteTrainingDataset, collate_sprite_batch, read_jsonl
from spritelab.training.eval_baseline import move_batch_to_device, resolve_device
from spritelab.training.eval_generator import _load_checkpoint, _tokenizer_from_checkpoint
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
from spritelab.training.palette_swap import (
    DEFAULT_SWAP_FAMILIES_TEXT,
    PaletteSwapConfig,
    estimate_applied,
)
from spritelab.training.progress import StepProgressBar
from spritelab.training.prompt_sensitivity import COLOR_WORDS
from spritelab.training.rgba import save_rgba_contact_sheet
from spritelab.training.sample_generator import read_prompt_records
from spritelab.training.structured_conditioning import (
    MULTI_HOT_FIELDS,
    STRUCTURED_BATCH_KEYS,
    StructuredConditioningVocab,
    build_structured_conditioning_vocab,
    encode_structured_conditioning,
    save_structured_conditioning_vocab,
    structured_vocab_from_checkpoint,
)
from spritelab.training.tokenization import SpriteTextTokenizer

SPRITE_SIZE = 32
STRUCTURED_DROPOUT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("category", ("category_id",)),
    ("object_id", ("object_id",)),
    ("base_object_id", ("base_object_id",)),
    ("colors", ("primary_color_id", "color_multi_hot")),
    ("materials", ("material_multi_hot",)),
    ("shapes", ("shape_multi_hot",)),
    ("function", ("function_multi_hot",)),
    ("style", ("style_multi_hot",)),
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
    "category": ("category_id",),
    "object_id": ("object_id",),
    "base_object": ("base_object_id",),
    "colors": ("primary_color_id", "color_multi_hot"),
    "materials": ("material_multi_hot",),
    "shapes": ("shape_multi_hot",),
    "function": ("function_multi_hot",),
    "style": ("style_multi_hot",),
}
_COLOR_STRUCTURED_KEYS: tuple[str, ...] = ("primary_color_id", "color_multi_hot")

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


@dataclass(frozen=True)
class ChallengerSampleConfig:
    checkpoint: Path
    prompts: Path
    out_dir: Path
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

        # v2 Phase 2: palette/index auxiliary heads (always present, losses default-off)
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
    ) -> Any:
        th, _nn_mod = _require_torch()
        emb = self._conditioning_embedding(
            t,
            caption_tokens=caption_tokens,
            semantic_tokens=semantic_tokens,
            structured_conditioning=structured_conditioning,
        )
        h = self.input(x)
        skips: list[Any] = []
        for level, down in enumerate(self.downs):
            for block in down["blocks"]:
                h = block(h, emb)
            if level < len(self.downs) - 1:
                skips.append(h)
                h = down["down"](h)
        # Bottleneck: save for palette head
        bottleneck_h = h
        for block in self.mid:
            h = block(h, emb)
        if self.bottleneck_attn is not None:
            h = self.bottleneck_attn(h)
        for up in self.ups:
            h = up["up"](h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = th.nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = th.cat([h, skip], dim=1)
            for block in up["blocks"]:
                h = block(h, emb)
        # h is now the final feature map before output
        output_rgba = self.output(h)

        if not return_aux:
            return output_rgba

        # v2 Phase 2 auxiliary outputs
        b_feat = self._bottleneck_pool(bottleneck_h).flatten(1)  # (B, C_bottleneck)
        palette_rgb = self.palette_head_rgb(b_feat).view(-1, 16, 3)  # (B, 16, 3)
        palette_presence = self.palette_head_presence(b_feat)  # (B, 16) logits
        index_logits = self.index_head(h)  # (B, 16, H, W)

        return {
            "velocity": output_rgba,
            "palette_rgb": palette_rgb,
            "palette_presence_logits": palette_presence,
            "index_logits": index_logits,
        }

    def _conditioning_embedding(
        self,
        t: Any,
        *,
        caption_tokens: Any,
        semantic_tokens: Any | None,
        structured_conditioning: Mapping[str, Any] | None,
    ) -> Any:
        th, _nn_mod = _require_torch()
        batch = int(caption_tokens.shape[0])
        if semantic_tokens is None:
            semantic_cond = th.zeros(batch, self.embed_dim, device=caption_tokens.device)
        else:
            semantic_cond = self._mean_pool_tokens(semantic_tokens)
        caption_cond = self._mean_pool_tokens(caption_tokens)
        pieces = [caption_cond, semantic_cond]
        structured_cond = self._structured_embedding(structured_conditioning, batch=batch, device=caption_tokens.device)
        if structured_cond is not None:
            pieces.append(structured_cond)
        cond = self.cond_mlp(th.cat(pieces, dim=1))
        time_emb = _sinusoidal_embedding(t.reshape(batch), self._time_embedding_dim).to(cond.device, cond.dtype)
        return self.time_mlp(time_emb) + cond

    def _mean_pool_tokens(self, tokens: Any) -> Any:
        th, _nn_mod = _require_torch()
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
        return self.embed_dim * (len(id_specs) + len(multi_specs))

    def _structured_embedding(
        self,
        structured_conditioning: Mapping[str, Any] | None,
        *,
        batch: int,
        device: Any,
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
            pieces.append(embedding(ids))
        for field, projection in self.structured_multi_hot_projections.items():
            value = None if structured_conditioning is None else structured_conditioning.get(field)
            width = int(projection.in_features)
            if value is None:
                multi_hot = th.zeros(batch, width, dtype=next(projection.parameters()).dtype, device=device)
            else:
                multi_hot = value.to(device=device, dtype=next(projection.parameters()).dtype).reshape(batch, width)
            pieces.append(projection(multi_hot))
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


def run_challenger_training(config: ChallengerTrainConfig) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    if str(config.architecture).lower() != "rectified_flow":
        raise ValueError("only --architecture rectified_flow is supported")
    metrics_every = int(config.metrics_every)
    if metrics_every < 1:
        raise ValueError("metrics_every must be >= 1")
    started = time.perf_counter()
    _set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditioning_mode = validate_conditioning_mode(config.conditioning_mode)

    manifest_rows = read_jsonl(config.training_manifest)
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
    structured_vocab = (
        build_structured_conditioning_vocab(
            [row for row in token_rows if row.get("split") == effective_split] or train_rows or token_rows
        )
        if uses_structured_conditioning(conditioning_mode)
        else None
    )
    if structured_vocab is not None:
        save_structured_conditioning_vocab(structured_vocab, out_dir / "structured_conditioning_vocab.json")

    palette_swap = PaletteSwapConfig.from_training_config(config)
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
    )
    palette_swap_summary = {**palette_swap.report_dict()}
    if palette_swap.active():
        from spritelab.training.palette_swap_review import summarize_dataset_palette_swap

        palette_swap_summary.update(
            summarize_dataset_palette_swap(
                config.dataset_dir,
                [dict(record) for record in train_dataset.records],
                palette_swap,
            )
        )
    else:
        palette_swap_summary.update(estimate_applied([dict(record) for record in train_dataset.records], palette_swap))
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
        )

    loader_generator = th.Generator().manual_seed(config.seed)
    loader_perf = dataloader_perf_kwargs(device, num_workers=config.num_workers)
    train_loader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
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
    }
    model = RectifiedFlowUNet(
        **model_config,
        film_conditioning=config.film_conditioning,
        bottleneck_attention=config.bottleneck_attention,
    ).to(device)
    optimizer = build_adamw(model.parameters(), lr=float(config.learning_rate), fused=config.fused_adamw)
    ema_enabled = float(config.ema_decay) > 0.0
    ema_state = _init_ema_state(model) if ema_enabled else None
    ema_fast_cache = _init_ema_fast_state(ema_state, model) if ema_enabled else None
    apply_backend_speed_flags(cudnn_benchmark=config.cudnn_benchmark, tf32=config.tf32)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=config.lr_schedule,
        max_steps=config.max_steps,
        warmup_steps=config.lr_warmup_steps,
    )
    non_blocking = device_type(device) == "cuda"
    config_json = {
        **{key: _jsonable(value) for key, value in asdict(config).items()},
        "architecture": "rectified_flow",
        "model_type": "generator_challenger",
        "conditioning_mode": conditioning_mode,
        "model_config": model_config,
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
    )
    metrics_path = out_dir / "train_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    preview_batch_cpu = next(iter(eval_train_loader))
    preview_batch = move_batch_to_device(preview_batch_cpu, device)

    step = 0
    last_loss = float(initial_train_losses["loss"])
    last_loss_components: dict[str, float] = {}
    checkpoint_steps = _normalize_checkpoint_steps(config.checkpoint_steps, max_steps=config.max_steps)
    checkpoint_step_paths: list[str] = []
    checkpoint_step_ema_paths: list[str] = []
    model.train()
    progress = StepProgressBar(config.max_steps, desc=f"challenger:{out_dir.name}")
    # Keep the metrics file open for the whole run instead of reopening it every
    # step; the line content and order are unchanged, so the file is identical.
    metrics_handle = metrics_path.open("a", encoding="utf-8")
    try:
        while step < config.max_steps:
            for batch in train_loader:
                if step >= config.max_steps:
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
                clip_gradients(model, config.grad_clip)
                optimizer.step()
                if ema_fast_cache is not None:
                    _update_ema_state_fast(ema_fast_cache, decay=float(config.ema_decay))
                if scheduler is not None:
                    scheduler.step()
                step += 1
                if step % metrics_every == 0 or step >= config.max_steps:
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
                if _should_save_checkpoint_step(step, save_every=config.save_every, checkpoint_steps=checkpoint_steps):
                    metrics_handle.flush()
                    step_checkpoint = out_dir / f"checkpoint_step_{step:06d}.pt"
                    _save_checkpoint(
                        step_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        config_json=config_json,
                        step=step,
                        ema_decay=float(config.ema_decay),
                        checkpoint_variant="step",
                    )
                    checkpoint_step_paths.append(str(step_checkpoint))
                    if ema_state is not None:
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
                        )
                        checkpoint_step_ema_paths.append(str(step_ema_checkpoint))
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
        )
        if len(val_source)
        else None
    )
    _save_checkpoint(
        out_dir / "checkpoint_last.pt",
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config_json=config_json,
        step=step,
        ema_decay=float(config.ema_decay),
        checkpoint_variant="last",
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
        "loss_decreased": float(final_train_losses["loss"]) < float(initial_train_losses["loss"]),
        "elapsed_seconds": time.perf_counter() - started,
        "model_config": model_config,
        "structured_vocab_sizes": None if structured_vocab is None else structured_vocab.sizes(),
        "structured_fields_enabled": structured_vocab is not None,
        "checkpoint_last": str(out_dir / "checkpoint_last.pt"),
        "checkpoint_best": str(out_dir / "checkpoint_best.pt"),
        "checkpoint_last_ema": None if ema_state is None else str(out_dir / "checkpoint_last_ema.pt"),
        "checkpoint_best_ema": None if ema_state is None else str(out_dir / "checkpoint_best_ema.pt"),
        "checkpoint_steps": list(checkpoint_steps),
        "checkpoint_step_paths": checkpoint_step_paths,
        "checkpoint_step_ema_paths": checkpoint_step_ema_paths,
        "warnings": [],
    }
    (out_dir / "train_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    ckpt = _load_checkpoint(checkpoint)
    model, tokenizer, conditioning_mode, semantic_max_length = load_challenger_from_checkpoint(ckpt, device=device)
    structured_vocab = structured_vocab_from_checkpoint(ckpt)
    prompts = read_prompt_records(config.prompts, max_records=config.max_samples)
    manifest_records: list[dict[str, Any]] = []
    base_noise_seed = int(config.noise_seed) if config.noise_seed is not None else int(config.seed) * 100000
    null_fields = _parse_null_fields(config.null_fields)
    color_token_ids = color_token_ids_for_tokenizer(tokenizer) if config.factored_cfg else ()
    for batch_start in range(0, len(prompts), max(1, int(config.batch_size))):
        batch_records = prompts[batch_start : batch_start + max(1, int(config.batch_size))]
        if not batch_records:
            continue
        noise_seeds = [base_noise_seed + batch_start + index for index in range(len(batch_records))]
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
        )
        pred = aux["velocity"]
    else:
        pred = model(
            xt,
            t,
            caption_tokens=inputs["caption_tokens"],
            semantic_tokens=inputs["semantic_tokens"],
            structured_conditioning=inputs.get("structured_conditioning"),
        )

    losses = _velocity_loss_components(
        pred,
        velocity,
        target_rgba=target_rgba,
        foreground_rgb_loss_weight=foreground_rgb_loss_weight,
        background_rgb_loss_weight=background_rgb_loss_weight,
    )
    palette_aux = palette_soft_min_auxiliary_loss(
        x1_hat=xt + (1.0 - view_t) * pred,
        target_rgba=target_rgba,
        palette=batch.get("palette"),
        palette_mask=batch.get("palette_mask"),
        temperature=palette_loss_temperature,
    )
    palette_weight = float(palette_loss_weight)
    losses["loss_palette_aux"] = palette_aux

    # v2 Phase 2: palette and index head losses
    zero = pred.sum() * 0.0
    losses["loss_palette_head"] = zero
    losses["loss_palette_presence"] = zero
    losses["loss_index_head"] = zero
    losses["index_head_active"] = False

    if aux_active:
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
) -> dict[str, float]:
    th, _nn_mod = _require_torch()
    generator = th.Generator(device=device)
    generator.manual_seed(int(seed))
    totals: dict[str, float] = {}
    count = 0
    batches_seen = 0
    batch_cap = int(max_batches)
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
            pred = model(
                xt,
                t,
                caption_tokens=inputs["caption_tokens"],
                semantic_tokens=inputs["semantic_tokens"],
                structured_conditioning=inputs.get("structured_conditioning"),
            )
            losses = _velocity_loss_components(
                pred,
                velocity,
                target_rgba=target_rgba,
                foreground_rgb_loss_weight=foreground_rgb_loss_weight,
                background_rgb_loss_weight=background_rgb_loss_weight,
            )
            palette_aux = palette_soft_min_auxiliary_loss(
                x1_hat=xt + (1.0 - view_t) * pred,
                target_rgba=target_rgba,
                palette=batch.get("palette"),
                palette_mask=batch.get("palette_mask"),
                temperature=palette_loss_temperature,
            )
            palette_weight = float(palette_loss_weight)
            losses["loss_palette_aux"] = palette_aux
            losses["loss"] = losses["loss_velocity"] + (
                palette_aux * palette_weight if palette_weight > 0.0 else palette_aux * 0.0
            )
            batch_size = int(target.shape[0])
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            count += batch_size
    if was_training:
        model.train()
    return {key: value / float(count) for key, value in sorted(totals.items())} if count else {"loss": 0.0}


def _velocity_loss_components(
    pred: Any,
    velocity: Any,
    *,
    target_rgba: Any,
    foreground_rgb_loss_weight: float = 1.0,
    background_rgb_loss_weight: float = 1.0,
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
    denom = float(squared.numel())
    loss_rgb = weighted_rgb_squared.sum() / denom
    loss_alpha = alpha_squared.sum() / denom
    return {
        "loss_velocity": loss_rgb + loss_alpha,
        "loss_rgb": loss_rgb,
        "loss_alpha": loss_alpha,
    }


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
    if gt_rgb.numel() and float(gt_rgb.detach().max().cpu()) > 1.0:
        gt_rgb = gt_rgb / 255.0
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
    if palette_rgb.numel() and float(palette_rgb.detach().max().cpu()) > 1.0:
        palette_rgb = palette_rgb / 255.0
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
    return soft_min[pixel_mask].mean()


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
) -> Any:
    """Integrate the rectified-flow ODE, optionally with factored (base/color) CFG.

    ``factored_cfg`` defaults to False and, when off, this reproduces the original
    all-or-nothing CFG path (``v_uncond + cfg_scale * (v_cond - v_uncond)``) exactly.
    When ``factored_cfg`` is True, guidance is split into a base term (uncond -> color-
    stripped conditioning) and a color term (color-stripped -> full conditioning), each
    with its own scale; see docs/v2_phase0_diagnostics.md.
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
        )
        if factored_cfg:
            v_uncond = model(
                x,
                t,
                caption_tokens=uncond_caption,
                semantic_tokens=uncond_semantic,
                structured_conditioning=uncond_structured,
            )
            v_no_color = model(
                x,
                t,
                caption_tokens=no_color_caption,
                semantic_tokens=no_color_semantic,
                structured_conditioning=no_color_structured,
            )
            velocity = v_uncond + base_scale * (v_no_color - v_uncond) + color_scale * (v_cond - v_no_color)
        elif use_cfg:
            v_uncond = model(
                x,
                t,
                caption_tokens=uncond_caption,
                semantic_tokens=uncond_semantic,
                structured_conditioning=uncond_structured,
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
    ckpt: dict[str, Any], *, device: Any
) -> tuple[RectifiedFlowUNet, SpriteTextTokenizer, str, int]:
    if str(ckpt.get("model_type") or "") != "generator_challenger":
        raise ValueError("checkpoint is not a generator_challenger checkpoint")
    tokenizer = _tokenizer_from_checkpoint(ckpt)
    model = RectifiedFlowUNet(**dict(ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
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
) -> tuple[ChallengerPromptAdapter, SpriteTextTokenizer, str, int]:
    model, tokenizer, conditioning_mode, semantic_max_length = load_challenger_from_checkpoint(ckpt, device=device)
    structured_vocab = structured_vocab_from_checkpoint(ckpt)
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
        for target, source in zip(cache["ema_nonfloat"], cache["src_nonfloat"]):
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
) -> None:
    th, _nn_mod = _require_torch()
    checkpoint = {
        "model_type": "generator_challenger",
        "architecture": "rectified_flow",
        "model_state_dict": model.state_dict() if model_state_dict is None else dict(model_state_dict),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config(),
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
        "checkpoint_type": "generator_challenger_rectified_flow_v0",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    th.save(checkpoint, path)


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
    return result if any(size > 1 for size in result.values()) else result


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
    return result


def _null_structured_conditioning(structured: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(structured, Mapping):
        return None
    th, _nn_mod = _require_torch()
    return {
        str(key): th.zeros_like(value) if isinstance(value, th.Tensor) else value for key, value in structured.items()
    }


def _masked_structured_conditioning(structured: Mapping[str, Any], mask: Any) -> dict[str, Any]:
    th, _nn_mod = _require_torch()
    result: dict[str, Any] = {}
    for key, value in structured.items():
        if isinstance(value, th.Tensor):
            cloned = value.clone()
            cloned[mask] = 0
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
