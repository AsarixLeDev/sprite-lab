# v1 Default: Model, Export Path, and Demo Gallery

## Purpose

This document describes the official v1 release/demo path for Sprite Lab's
32x32 pixel-art sprite generator: which checkpoint to sample, which sampling
and decode settings to use, how to regenerate a visual QA gallery, and why the
palette-swap branches are not part of v1.

## Quickstart

```powershell
cd C:\Users\Mathieu\Documents\sprite-lab
$env:PYTHONPATH = "src"
$py = "C:\Users\Mathieu\anaconda3\python.exe"

& $py -m spritelab train build-v1-gallery `
  --out experiments\v1_gallery `
  --device cuda `
  --seed 20260723 `
  --batch-size 32
```

This is the one command a new user needs: it builds the deterministic v1
prompt set, samples it with the official v1 preset, and writes contact
sheets and a report under `experiments\v1_gallery`. `scripts\build_v1_gallery.ps1`
wraps the same command with `-OutDir`/`-Device`/`-Seed`/`-BatchSize`
parameters. See "Building a v1 demo gallery" below for the full output
layout and optional arguments, or "Local v1 gallery GUI" for an interactive
alternative.

## Official v1 path

```text
Model:      Phase 1 EMA checkpoint
Checkpoint: experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt
Sampling:   CFG 3.0, 30 steps
Decode:     k16 deterministic palette projection (deterministic_kmeans)
Projection min_pixel_share: 0.01
Palette swap: parked / experimental (not used for v1)
```

The Phase 1 structured model is the adopted v1 model. Full-v4 learnability is
solved and no further training is required for the v1 release path; this
document only covers packaging, sampling, and validation of the already
trained checkpoint.

These settings are exposed as the `v1` (alias `phase1_v1`) export preset:
`--export-preset v1` on `sample-generator-challenger` resolves the checkpoint
to its EMA sibling when available and fills in CFG scale, steps, and
projection defaults automatically (see `_apply_export_preset_defaults` in
`src/spritelab/training/cli.py`).

## Building a v1 demo gallery

`build-v1-gallery` generates a deterministic gallery end to end: it builds (or
reads) a prompt set, samples it with the v1 preset, runs QA and structural
review, builds contact sheets, and writes a Markdown/JSON summary report. It
never trains a model.

```powershell
cd C:\Users\Mathieu\Documents\sprite-lab
$env:PYTHONPATH = "src"
$py = "C:\Users\Mathieu\anaconda3\python.exe"

& $py -m spritelab train build-v1-gallery `
  --out experiments\v1_gallery `
  --device cuda `
  --seed 20260723 `
  --batch-size 32
```

If CUDA is unavailable, pass `--device cpu` instead. CPU is not the default
because the real release gallery should be sampled on the same device class
used for validation.

Outputs are written to `<out>`:

```text
<out>\samples\                  # raw/hard/indexed PNGs, projected PNGs, manifest
<out>\contact_sheets\           # overall, per-category, and per-color contact sheets
<out>\v1_gallery_prompts.jsonl  # the prompt set actually sampled
<out>\v1_gallery_report.md
<out>\v1_gallery_report.json
```

Optional arguments: `--checkpoint`, `--prompts` (use a custom prompt file
instead of the built-in set), `--num-samples` (cap the prompt/sample count),
`--categories` (comma-separated filter over the built-in categories),
`--contact-sheet-columns`, and `--include-ood` / `--include-grounded` /
`--include-stress-prompts` (toggle the built-in prompt families).

### Built-in prompt set

When `--prompts` is not given, the gallery builds a deterministic prompt set
(`src/spritelab/training/v1_gallery.py`) covering weapons, armor, item icons,
tools, materials, effect icons, and plants, with color coverage across red,
blue, green, yellow, purple, brown, gray, metallic, and gold (applied only
where semantically appropriate, e.g. no "metallic mushroom"). It also mixes in
a handful of compositional (unseen-combination) prompts, style-stress
prompts, and — when `--include-ood` is set (default on) — a trimmed slice of
the existing OOD compositional prompt set from
`src/spritelab/training/ood_prompts.py`. The prompt set size is in the 48-96
range by default.

## Local v1 gallery GUI

For interactive use, `v1-gallery-gui` launches a local Gradio GUI over the
same `build_v1_gallery_demo` code path as `build-v1-gallery`: pick an output
directory, an optional custom prompt file, device, seed, and prompt-family
toggles, click "Build v1 gallery", and preview the resulting contact sheets
in the browser. It never trains a model. Requires the `ui` extra
(`pip install -e ".[ui]"` or `pip install gradio`).

```powershell
& $py -m spritelab train v1-gallery-gui --out experiments\v1_gallery_gui
```

`--host` and `--port` control where the local server listens (defaults to
`127.0.0.1` and a Gradio-assigned port).

## Sampling custom prompts with the v1 preset

To sample your own prompt file with the same official settings used for
release validation:

```powershell
& $py -m spritelab train sample-generator-challenger `
  --export-preset v1 `
  --checkpoint experiments\challenger_full_v4_phase1\train_25k\checkpoint_last_ema.pt `
  --prompts my_prompts.jsonl `
  --out experiments\my_v1_samples `
  --device cuda `
  --seed 20260723
```

`--export-preset v1` applies CFG 3.0, 30 sampling steps, and k16 deterministic
palette projection (`--project-palette`, `--project-palette-target-colors 16`,
`--project-palette-min-pixel-share 0.01`, `--project-palette-method
deterministic_kmeans`) unless you explicitly override those flags.

## What projection does

Decode-time palette projection (`src/spritelab/training/palette_projection.py`)
reduces each generated sprite's visible RGB palette to at most k=16 colors
using a deterministic weighted k-means over the sprite's own visible pixels,
then merges any cluster covering less than `min_pixel_share` (1%) of visible
pixels into its nearest surviving cluster. It leaves alpha untouched and only
recolors the RGB channels of visible pixels. This is applied at decode time,
not baked into training, so it's a pure post-processing step over the raw
model output.

## Why palette swap is parked

Palette-swap augmentation (`src/spritelab/training/palette_swap.py` and the
`dataset-palette-swap-review` / `--palette-swap-*` training flags) is a
training-time data augmentation that recolors indexed sprites toward target
color families before they reach the model. It remains research-only:

- Static/strict palette-swap training runs were evaluated and rejected.
- Stochastic palette swap is kept as a research branch, not part of the
  validated v1 path.
- The v1 export preset always reports `palette_swap: parked / experimental`
  and does not enable palette-swap augmentation.

No further model training is planned for the v1 release path; palette swap
may be revisited in a future model iteration, but v1 packaging and validation
do not depend on it.

## Where the validation metrics came from

The numbers below are the validated v1 OOD result over 96 samples, produced
by the full-v4 challenger audit / OOD-compositional evaluation pipeline
(`src/spritelab/training/generator_audits.py`) against the Phase 1 EMA
checkpoint with the v1 preset (CFG 3.0, 30 steps, k16 projection,
`min_pixel_share=0.01`), using all 928 available source sprites for
prompt-faithfulness nearest-source retrieval (`source_hash=083d55be9803`).
They are recorded as a static reference in
`spritelab.training.v1_gallery.VALIDATED_V1_OOD_REFERENCE` and embedded in
every `v1_gallery_report.json` for context; a single gallery run does not
recompute prompt-faithfulness metrics (category/color consistency, blob,
potion, repeated-silhouette, border-touch) since those require dataset-grounded
nearest-source retrieval, not just the generated samples.

```text
QA errors: 0
median visible colors: 32 -> 12
rare-color warnings: 0.3229 -> 0.0000
category: 0.8068
color: 0.8438
repeated: 0.0000
blob: 0.3021
potion: 0.0521
touch: 0.5104
mean RGB MAE: 0.0206
destructive rate: 0.0
source_count_used: 928
source_hash: 083d55be9803
```

## Troubleshooting

If the Phase 1 checkpoint is missing (e.g. on a fresh clone, before running
training), `build-v1-gallery`, `sample-generator-challenger`, and the GUI all
fail fast with a `FileNotFoundError` naming the resolved checkpoint path and
the official v1 checkpoint path, rather than a raw PyTorch loader error. Pass
`--checkpoint` to point at wherever the checkpoint actually lives, or train
the Phase 1 challenger first (out of scope for v1 packaging).

## Official v1 default

Official v1 default: Phase 1 EMA + CFG 3.0 + k16 deterministic palette
projection. Palette-swap branches are experimental and not used for v1.
