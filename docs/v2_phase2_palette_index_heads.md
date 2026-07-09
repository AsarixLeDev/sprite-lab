# Sprite Lab v2 Phase 2 — Palette / Index Supervision

Discrete sprite-structure auxiliary heads and losses.

## Why Phase 2?

Phase 1 tried FiLM, bottleneck attention, and per-group dropout. All variants
improved color at the expense of category on hard OOD-core prompts. More
conditioning-only training is unlikely to solve object/color disentanglement.

Every training batch already carries discrete sprite supervision:

* `palette` — ground-truth K=16 RGB palette
* `palette_mask` — which slots are actually used
* `index_map` — per-pixel palette index
* `role_map` — per-pixel semantic role

These are exact, not approximate. Phase 2 adds lightweight auxiliary heads
that predict this discrete structure, so the model can learn it as part of
the training signal rather than relying entirely on continuous RGBA +
post-hoc k16 projection.

## Flags (all default-off)

| Flag | Default | Description |
|------|---------|-------------|
| `--index-head-loss-weight` | `0.0` | Cross-entropy on index map prediction |
| `--palette-head-loss-weight` | `0.0` | Slot-aligned MSE on palette RGB prediction |
| `--palette-presence-loss-weight` | `0.0` | BCE on palette slot presence |
| `--index-head-warmup-steps` | `0` | Index head inactive before this global step |
| `--palette-head-use-gt-palette-prob` | `1.0` | Future hook for predicted vs GT palette (use 1.0 for now) |

## Heads

### Palette Head

Input: bottleneck features (after U-Net mid blocks), globally pooled.
Outputs:

* `palette_rgb` — `[B, 16, 3]` float RGB, slot-aligned MSE against GT palette
* `palette_presence_logits` — `[B, 16]` logits, BCE against `palette_mask`

Slots correspond directly to GT palette order. No Hungarian matching needed.

### Index Head

Input: final feature map (before output convolution).
Output: `[B, 16, H, W]` logits, CE against `index_map` on visible pixels
(alpha > 0) only.

Warmup: index loss weight is zero for `global_step < index_head_warmup_steps`.
This lets velocity converge to reasonable shapes before the index map
forces discrete structure.

## Losses

```text
total_loss = velocity_loss
  + palette_loss_weight * soft_min_palette_aux
  + palette_head_loss_weight * loss_palette_head
  + palette_presence_loss_weight * loss_palette_presence
  + effective_index_weight * loss_index_head
```

where `effective_index_weight = 0 if step < warmup else index_head_loss_weight`.

New loss components reported: `loss_palette_head`, `loss_palette_presence`,
`loss_index_head`, `index_head_active`.

## Sampling / Export

Unchanged. The palette/index heads are training-only auxiliary losses.
They do not replace k16 projection or change generated outputs.

## First Smoke Command (1k steps)

```powershell
cd C:\Users\Mathieu\Documents\sprite-lab
$env:PYTHONPATH = "src"
$py = "C:\Users\Mathieu\anaconda3\python.exe"

& $py -m spritelab train audit-challenger-full-v4 `
  --dataset datasets\sprite_lab_multisource_v4 `
  --training-manifest datasets\sprite_lab_multisource_v4\training_manifest.jsonl `
  --out experiments\challenger_full_v4_v2_phase2_palette_index_smoke_1k `
  --architecture rectified_flow `
  --device cuda `
  --seed 20260706 `
  --max-steps 1000 `
  --batch-size 32 `
  --num-workers 4 `
  --lr 0.0002 `
  --conditioning-mode caption_semantic_structured `
  --cfg-dropout 0.1 `
  --structured-field-dropout 0.1 `
  --ema-decay 0.999 `
  --sample-ema `
  --foreground-rgb-loss-weight 2.0 `
  --background-rgb-loss-weight 0.25 `
  --palette-loss-weight 0.1 `
  --film-conditioning `
  --bottleneck-attention `
  --index-head-loss-weight 0.25 `
  --palette-head-loss-weight 0.10 `
  --palette-presence-loss-weight 0.05 `
  --index-head-warmup-steps 0 `
  --sample-steps 30 `
  --cfg-scale 3.0 `
  --export-preset v1 `
  --max-colors 32 `
  --alpha-threshold 0.5 `
  --max-eval-prompts 32 `
  --max-sensitivity-prompts 8 `
  --noise-samples 1 `
  --sample-batch-size 32 `
  --run-ood-compositional
```

## First Full Training Command (25k steps)

```powershell
& $py -m spritelab train audit-challenger-full-v4 `
  --dataset datasets\sprite_lab_multisource_v4 `
  --training-manifest datasets\sprite_lab_multisource_v4\training_manifest.jsonl `
  --out experiments\challenger_full_v4_v2_phase2_palette_index `
  --architecture rectified_flow `
  --device cuda `
  --seed 20260706 `
  --max-steps 25000 `
  --batch-size 32 `
  --num-workers 4 `
  --lr 0.0002 `
  --conditioning-mode caption_semantic_structured `
  --cfg-dropout 0.1 `
  --structured-field-dropout 0.1 `
  --ema-decay 0.999 `
  --sample-ema `
  --foreground-rgb-loss-weight 2.0 `
  --background-rgb-loss-weight 0.25 `
  --palette-loss-weight 0.1 `
  --film-conditioning `
  --bottleneck-attention `
  --index-head-loss-weight 0.25 `
  --palette-head-loss-weight 0.10 `
  --palette-presence-loss-weight 0.05 `
  --index-head-warmup-steps 2000 `
  --sample-steps 30 `
  --cfg-scale 3.0 `
  --export-preset v1 `
  --max-colors 32 `
  --alpha-threshold 0.5 `
  --max-eval-prompts 128 `
  --max-sensitivity-prompts 32 `
  --noise-samples 2 `
  --sample-batch-size 32 `
  --run-ood-compositional
```

## Evaluation After Training

```powershell
& $py -m spritelab train run-v2-phase0-eval `
  --out experiments\v2_phase2_palette_index_eval `
  --checkpoint experiments\challenger_full_v4_v2_phase2_palette_index\train_25k\checkpoint_last_ema.pt `
  --dataset datasets\sprite_lab_multisource_v4 `
  --build-prompts `
  --prompt-count 384 `
  --prompt-seed 20260706 `
  --presets v1 `
  --seeds 20260723,20260724,20260725 `
  --max-samples 384 `
  --eval-profile ood_core `
  --profile-weighting family `
  --device cuda `
  --batch-size 32
```

## Go / No-Go Criteria

Same as Phase 1, but OOD-core family-weighted with the added requirement
that the model does not regress on blob or near-copy:

```text
category >= baseline_ood_core_category + 0.03
color    >= baseline_ood_core_color + 0.03
blob     <= baseline_ood_core_blob
potion   <= baseline_ood_core_potion
nearcopy <= baseline_ood_core_nearcopy + 0.01
rare     <= 0.01 after projection
QA errors = 0
```

## Caveats

* Heads are always present in the model graph (small parameter overhead ~20K)
  but losses default to 0.0 weight, so they train through gradient but don't
  influence the loss unless explicitly enabled.
* Index head warmup should be set long enough for velocity to produce
  recognizable shapes (500-2000 steps).
* Palette prediction uses slot-aligned MSE; if dataset palettes are not
  consistently ordered, this may need Hungarian matching in a future phase.
* Sampling and export (k16 projection) are unchanged; the heads are
  training-only.
