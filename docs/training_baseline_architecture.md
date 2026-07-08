# Training Baseline Architecture

This document defines the first model-training foundation for `sprite-lab`.
The goal is a deterministic, CPU-safe diagnostic baseline that proves the
semantic training manifest can be loaded, conditioned, reconstructed, trained,
checkpointed, and evaluated without changing the dataset contract.

## Current Dataset Contract

The exported semantic-v3 dataset is a directory containing split rasters and
JSONL metadata:

```text
datasets/<dataset_name>/
  train.npz
  val.npz
  test.npz
  manifest_train.jsonl
  manifest_val.jsonl
  manifest_test.jsonl
  training_manifest.jsonl
  eval_prompts.jsonl
  dataset_qa_report.json
  training_manifest_qa_report.json
```

Each split `.npz` contains fixed arrays:

```text
alpha        [N, 32, 32] uint8, values 0/1
index_map    [N, 32, 32] int, palette row indices; row 0 is transparent
role_map     [N, 32, 32] uint8 role ids
palette      [N, K, 3] uint8 RGB palette rows
palette_mask [N, K] bool, valid palette rows for each sprite
category_id  [N] int64
sprite_id    [N] string
```

`training_manifest.jsonl` expands each sprite into multiple conditioning rows.
Every row points back to a raster with `split`, `npz_file`, `npz_row`, and
`sprite_id`, and carries `caption`, `caption_type`, `category`, `object_name`,
`base_object`, `conditioning.semantic_v3`, dropout accounting, and audit
metadata.

The loader must treat the `.npz` files as the source of pixels. It must not
assume per-sprite PNG files exist.

## What This Baseline Is

The Phase A baseline is a conditional autoencoder/reconstructor. It receives
the target sprite representation as input along with caption and semantic
conditioning, then predicts the same sprite structure.

It is intended to validate:

- manifest-to-raster lookup;
- caption and semantic token plumbing;
- reconstruction target shapes and losses;
- deterministic one-batch overfit behavior;
- checkpoint and evaluation wiring.

## What This Baseline Is Not

This is not a production generator. It is not diffusion, not a large
transformer, not a reference-image model, and not expected to produce novel
sprites. It sees the target sprite as input, so good reconstruction only proves
training plumbing and representation correctness.

## Input Representation

`alpha` is loaded as a float tensor `[1, 32, 32]` with values 0/1. It is both an
input channel and the target for alpha reconstruction.

`index_map` is loaded as a long tensor `[32, 32]`. Palette row `0` is the
transparent/background row. The model embeds index ids and predicts logits over
the fixed split palette row count `K`.

`palette` is loaded as `[K, 3]` float RGB in `[0, 1]`. The baseline does not
predict palettes; it uses the source palette for reconstruction previews and
valid-target masking.

`palette_mask` is loaded as `[K]` bool. Index reconstruction loss ignores target
pixels whose target palette row is invalid for that sample.

`role_map` is loaded as a long tensor `[32, 32]`. The model can use role ids as
input and can optionally predict role logits as an auxiliary diagnostic head.

`caption` is the sampled training-manifest caption. It is tokenized with a
small deterministic tokenizer.

Semantic fields come from the manifest row, especially `category`,
`object_name`, `base_object`, `caption_type`, `conditioning.semantic_v3`,
`kept_attributes`, `dropped_attributes`, and `dropout_ops`. These are flattened
into semantic tokens with the same tokenizer.

## Target Representation

The required targets are:

- `alpha` reconstruction through `alpha_logits [B, 1, 32, 32]`;
- palette-index reconstruction through `index_logits [B, K, 32, 32]`;
- optional `role_map` reconstruction through `role_logits [B, R, 32, 32]`.

Palette RGB rows are not predicted in Phase A. Predicting palettes would add a
second output space and extra ambiguity before the core raster/conditioning
contract is proven.

## Conditioning Strategy

Caption tokens are produced by lowercasing, splitting snake_case, removing
punctuation, adding `<bos>/<eos>`, truncating to a fixed length, and padding
with `<pad>`.

Semantic tokens are produced by flattening grounded manifest fields and nested
semantic-v3 attributes. Prefix-like marker tokens such as `category`,
`base_object`, `kept_colors`, and `dropout_ops` keep the conditioning source
visible to the model without external tokenizer dependencies.

Category/base-object information is present in the semantic token stream. The
numeric `category_id` is also embedded separately as a compact conditioning
signal.

The baseline mean-pools token embeddings with padding masks, combines caption,
semantic, and category embeddings, and fuses the result into convolutional
features with FiLM-style scale/shift at the bottleneck.

## Minimal Baseline Architecture

`SpriteCondAutoencoder` has:

- index embedding for `index_map`;
- optional role embedding for `role_map`;
- alpha input channel;
- small convolutional encoder;
- mean-pooled caption and semantic embedding encoder;
- category embedding;
- bottleneck FiLM projection;
- small convolutional decoder;
- alpha, index, and optional role heads.

It is intentionally small enough to run on CPU in tests and smoke commands.

## Losses

`sprite_reconstruction_loss` returns:

```text
loss
loss_alpha
loss_index
loss_role
```

Alpha uses `BCEWithLogitsLoss` on every pixel.

Index reconstruction uses cross entropy only on opaque pixels whose target
palette row is valid under `palette_mask`. Transparent pixels are ignored for
index loss; alpha loss is responsible for reconstructing transparency. This
keeps the background policy consistent with the export contract where
transparent pixels have `index_map == 0`.

Role reconstruction is optional auxiliary cross entropy when `role_logits` and
`role_map` are present. Its default weight is lower than alpha/index.

## Training Loop

The baseline training command:

```powershell
python -m spritelab train baseline `
  --dataset datasets\oga_496_rpg_icons_32fix_label_v2_semantic_v3 `
  --training-manifest datasets\oga_496_rpg_icons_32fix_label_v2_semantic_v3\training_manifest.jsonl `
  --out runs\baseline_496_smoke `
  --batch-size 16 `
  --max-steps 200 `
  --device cpu `
  --seed 4962026
```

The loop sets deterministic Python, NumPy, and torch seeds. CPU is the default;
CUDA is optional when explicitly requested or selected by `--device auto`.
Metrics are written as JSONL per step. A final report records initial/final
loss, optional validation loss, elapsed time, warnings, and paths.

## Checkpoints

Run directories contain:

```text
config.json
vocab.json
train_metrics.jsonl
train_report.json
checkpoint_last.pt
checkpoint_best.pt  # when validation is available
reconstructions.png # when Pillow preview export succeeds
```

Checkpoints include model state, optimizer state, model config, vocabulary, and
training report fields needed for evaluation.

## Evaluation Scripts

The evaluation command loads a checkpoint, reconstructs a requested split, and
writes `eval_report.json` plus an optional reconstruction contact sheet:

```powershell
python -m spritelab train eval-baseline `
  --dataset <dataset_dir> `
  --training-manifest <training_manifest.jsonl> `
  --checkpoint <checkpoint_last.pt> `
  --split val `
  --out <eval_dir>
```

`eval_prompts.jsonl` is counted as metadata plumbing only. This baseline does
not generate images from prompt-only inputs.

## Overfit-One-Batch Protocol

`--overfit-batches 1` restricts training to a fixed tiny subset and repeatedly
optimizes it:

```powershell
python -m spritelab train baseline `
  --dataset datasets\oga_496_rpg_icons_32fix_label_v2_semantic_v3 `
  --training-manifest datasets\oga_496_rpg_icons_32fix_label_v2_semantic_v3\training_manifest.jsonl `
  --out runs\baseline_496_overfit `
  --batch-size 8 `
  --max-steps 100 `
  --overfit-batches 1 `
  --device cpu
```

Acceptance is that final train loss is clearly lower than initial train loss
and the command writes metrics, report, and checkpoint artifacts.

## Future Extension Path

Better tokenizer/text encoder: replace the deterministic tokenizer with a
learned BPE or compact text encoder while keeping the same manifest fields.

Discrete image tokens: add an image-token target over `index_map` or learned
palette tokens for transformer experiments.

Diffusion/flow/transformer: introduce a Phase B generator that consumes
caption/semantic tokens plus noise and predicts `alpha/index_map` without
seeing the target sprite as input.

Reference-image conditioning branch: add a 32x32 reference encoder that maps an
external image into conditioning features, then combine it with text/semantic
conditioning for prompts such as "dolphin from a reference image, converted to
32x32 pixel art."

## Test Strategy

Tests should be CPU-safe and skip cleanly if torch is unavailable. Coverage
should include tokenizer determinism, manifest-to-npz loading, split filtering,
useful errors for missing files and bad rows, forward output shapes, variable
caption lengths via padding, finite losses, transparent-mask policy, backward
gradients, tiny training steps, checkpoint/metrics writing, overfit loss
decrease, inspect-data CLI output, and real-dataset inspection when the
reference dataset is present.

Existing dataset QA, training-manifest QA, label-v2, and semantic-v3 tests must
remain independent of this torch-dependent baseline.

## Non-Goals And Safety Constraints

- No remote models or network calls.
- No GPU requirement.
- No large jobs in tests.
- No heavyweight tokenizer dependency.
- No dataset format rewrite.
- No per-sprite PNG assumption.
- No production-quality image-generation claims.

## Caption-To-RGBA Generator V0

The first true generation pass is intentionally separate from the reconstructor.
`TinyCaptionSpriteGenerator` receives only:

```text
caption_tokens
optional semantic_tokens
latent noise
```

It does not receive `alpha`, `index_map`, `role_map`, palette rows, or any
reference image as input. It mean-pools caption and semantic token embeddings,
concatenates latent noise, projects the conditioning vector into a learned 8x8
feature grid, upsamples to 16x16 and 32x32 with small convolutional blocks, and
emits:

```text
rgb_logits   [B, 3, 32, 32]
alpha_logits [B, 1, 32, 32]
```

### RGBA Target

Generator v0 does not predict palette indices. The exported datasets use
per-sprite palettes, so palette index `3` is not a universal color class across
sprites. Predicting `index_map` from captions alone would require palette
prediction or palette conditioning first.

Instead, the loader derives a universal target from each `.npz` row:

```text
rgb  = palette[index_map]
a    = alpha
rgba = concat(rgb, a)
```

Targets are float tensors in `[0, 1]`:

```text
rgba  [4, 32, 32]
rgb   [3, 32, 32]
alpha [1, 32, 32]
```

Transparent pixels keep alpha 0. Their RGB target is zeroed so the generator has
a stable transparent-background target independent of palette row 0.

### Generator Loss

`rgba_generator_loss` returns:

```text
loss
loss_alpha
loss_rgb_opaque
loss_rgb_all
```

Alpha uses binary cross entropy with logits. Opaque RGB uses L1 on
`sigmoid(rgb_logits)` masked by target alpha. A smaller all-pixel RGB L1 term
stabilizes transparent areas:

```text
loss = loss_alpha + loss_rgb_opaque + 0.25 * loss_rgb_all
```

### Commands

Training:

```powershell
python -m spritelab train generator `
  --dataset <dataset_dir> `
  --training-manifest <training_manifest.jsonl> `
  --out <run_dir> `
  --batch-size 32 `
  --max-steps 1000 `
  --device cuda `
  --seed 123
```

Evaluation:

```powershell
python -m spritelab train eval-generator `
  --dataset <dataset_dir> `
  --training-manifest <training_manifest.jsonl> `
  --checkpoint <checkpoint.pt> `
  --split val `
  --out <eval_dir>
```

Prompt-only sampling is also supported with `--checkpoint`, `--prompts`, and
`--out`.

Generator run directories contain `config.json`, `vocab.json`,
`train_metrics.jsonl`, `train_report.json`, `checkpoint_last.pt`, and sample
PNG contact sheets such as `samples_step_000020.png` and `samples_final.png`.
