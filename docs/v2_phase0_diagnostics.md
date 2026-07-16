# v2 Phase 0: No-Training Diagnostics

## Purpose

Phase 0 of the v2 investigation adds sampling-time and reporting-time diagnostics
on top of the existing v1 model and pipeline: no training, no architecture
changes, no checkpoint changes, and no change to default v1 behavior unless one
of the new flags below is passed explicitly. The goal is to answer, before any
retraining decision, whether the Phase 1 checkpoint's conditioning fields (in
particular color vs. category/object) are actually driving generation, and
whether the existing n=96 v1 OOD numbers (see [`docs/v1_default.md`](v1_default.md))
are precise enough to act on.

This document assumes the Phase 1 EMA checkpoint at
`experiments/challenger_full_v4_phase1/train_25k/checkpoint_last_ema.pt` and the
standard environment setup:

```powershell
# Run from the cloned repository root.
$env:PYTHONPATH = "src"
$py = "python"
```

## 1. v1 rebaseline at n=256/288

The existing `sample-generator-challenger` + `prompt-faithfulness` pipeline
already supports arbitrarily large sample counts; nothing new was added here
beyond documenting the invocation. Use `--export-preset v1` so sampling matches
the validated release settings (CFG 3.0, 30 steps, k16 projection), and use
`--max-sources 0` (or `--faithfulness-max-sources 0` in the full audit) for
prompt-faithfulness so nearest-source retrieval draws from all available
sources rather than a biased subset (category consistency is nearest-source
retrieval, so a biased source subset biases the metric).

```powershell
& $py -m spritelab train sample-generator-challenger `
  --checkpoint experiments\challenger_full_v4_phase1\train_25k\checkpoint_last_ema.pt `
  --prompts experiments\challenger_full_v4_phase1\prompts\ood_compositional_prompts.jsonl `
  --out experiments\v2_phase0\v1_ood_256 `
  --export-preset v1 `
  --max-samples 256 `
  --device cuda `
  --seed 20260723 `
  --batch-size 32

& $py -m spritelab train generated-qa --generated experiments\v2_phase0\v1_ood_256

& $py -m spritelab train generated-review --generated experiments\v2_phase0\v1_ood_256

& $py -m spritelab train prompt-faithfulness `
  --generated experiments\v2_phase0\v1_ood_256 `
  --prompts experiments\challenger_full_v4_phase1\prompts\ood_compositional_prompts.jsonl `
  --dataset <path-to-training-dataset> `
  --out experiments\v2_phase0\v1_ood_256\prompt_faithfulness_report.md `
  --out-json experiments\v2_phase0\v1_ood_256\prompt_faithfulness_report.json `
  --max-sources 0
```

For n=288, use the same commands with `--max-samples 288` and a prompt file
with at least 288 rows (`build-ood-compositional-prompts --max-prompts 288` can
extend the built-in set).

The resulting `prompt_faithfulness_report.json` now carries `_ci95` fields
(Wilson 95% confidence intervals, see section 4) next to each rate metric, so
n=256/288 runs are directly comparable to the n=96 reference numbers with
uncertainty attached, instead of bare point estimates.

## 2. Factored CFG sweep

`sample-generator-challenger` now accepts `--factored-cfg` plus
`--cfg-base-scale` / `--cfg-color-scale`. This is **off by default**; without
`--factored-cfg`, `--cfg-scale` behaves exactly as before (byte-identical
sampling path).

Conceptually, guidance is split into two additive terms instead of one:

```text
v = v_uncond + cfg_base_scale * (v_no_color - v_uncond) + cfg_color_scale * (v_full - v_no_color)
```

where `v_no_color` is the conditioned prediction with color signal stripped
from the caption tokens, semantic tokens, and structured `primary_color_id` /
`color_multi_hot` fields (via `strip_color_conditioning` in
`generator_challenger.py`), while category/object/base_object/material/shape/
function/style signal is left untouched. This targets the "biggest immediate
opportunity" from Fable's v2 verdict: Phase 1 trained with independent
structured field dropout, so color and non-color guidance can be pulled apart
at sample time without retraining.

Precedence: if `--factored-cfg` is set and `--cfg-base-scale` /
`--cfg-color-scale` are omitted, both default to `--cfg-scale`'s value (so
`--factored-cfg --cfg-scale 3.0` alone reproduces plain CFG-3.0 behavior
through the factored path — useful as a sanity check before sweeping).

Recommended sweep grid:

```text
base_scale  in {1.5, 2.0, 2.5, 3.0}
color_scale in {2.0, 3.0, 4.5, 6.0}
```

Example single run:

```powershell
& $py -m spritelab train sample-generator-challenger `
  --checkpoint experiments\challenger_full_v4_phase1\train_25k\checkpoint_last_ema.pt `
  --prompts experiments\challenger_full_v4_phase1\prompts\ood_compositional_prompts.jsonl `
  --out experiments\v2_phase0\factored_cfg\base2.0_color4.5 `
  --export-preset v1 `
  --max-samples 96 `
  --factored-cfg `
  --cfg-base-scale 2.0 `
  --cfg-color-scale 4.5 `
  --device cuda `
  --seed 20260723 `
  --batch-size 32
```

`--export-preset v1` still fills in steps/max-colors/alpha-threshold/palette
projection defaults; only CFG becomes factored. Run `generated-qa` and
`prompt-faithfulness` (as in section 1) against each sweep cell's output
directory, then compare `color_consistency_rate` / `category_consistency_rate`
/ `generic_blob_collapse_rate` (with their CIs) across the grid.

## 3. Sampling-time field ablations

`sample-generator-challenger` also accepts `--null-fields`, a comma-separated
list of conditioning fields to null at sample time (default empty / no-op).
Choices: `caption`, `semantic`, `category`, `object_id`, `base_object`,
`colors`, `materials`, `shapes`, `function`, `style`, `structured` (the last
nulls every structured field at once). This is implemented by
`apply_conditioning_field_ablations` in `generator_challenger.py` and applies
after conditioning-mode resolution but before CFG, so it composes with both
plain and factored CFG, and with the v1 preset.

Recommended ablation set:

```text
none
colors
object_id
category
caption
semantic
structured
object_id+colors
```

Example (nulling `object_id` and `colors` together):

```powershell
& $py -m spritelab train sample-generator-challenger `
  --checkpoint experiments\challenger_full_v4_phase1\train_25k\checkpoint_last_ema.pt `
  --prompts experiments\challenger_full_v4_phase1\prompts\ood_compositional_prompts.jsonl `
  --out experiments\v2_phase0\ablations\object_id_colors `
  --export-preset v1 `
  --max-samples 96 `
  --null-fields object_id,colors `
  --device cuda `
  --seed 20260723 `
  --batch-size 32
```

Run `prompt-faithfulness` on each ablation's output the same way as section 1.
If nulling `object_id` alone barely moves `category_consistency_rate` /
`color_consistency_rate` while nulling `colors` collapses
`color_consistency_rate`, that's evidence `object_id` is doing most of the
conditioning work and color is comparatively weak — i.e. the "does object_id
dominate" question from the v2 verdict.

## 4. Confidence intervals

`prompt_faithfulness_report.json` now includes Wilson 95% confidence interval
fields (`spritelab.training.stats.wilson_confidence_interval`) alongside the
existing scalar rates, without removing or renaming any existing field:

```text
category_consistency_rate / category_consistency_ci95
nearest_source_category_consistency_rate / nearest_source_category_consistency_ci95
color_consistency_rate / color_consistency_ci95
repeated_silhouette_rate / repeated_silhouette_rate_ci95
generic_potion_collapse_rate / generic_potion_collapse_rate_ci95
generic_blob_collapse_rate / generic_blob_collapse_rate_ci95
nearest_neighbor_duplicate_rate / nearest_neighbor_duplicate_rate_ci95
near_copy_rate / near_copy_rate_ci95  (alias of nearest_neighbor_duplicate_rate)
```

At n=96, Wilson intervals are wide (typically +/-8-10 points on rates near
0.8); at n=256+ they tighten substantially. **Do not treat a change in a point
estimate as real if the new value falls inside the old value's CI** — that's
noise, not signal (this is Fable's "n=96 is too noisy for decisions" point).

## 5. Near-copy / retrieval visibility

`prompt_faithfulness_report.json`'s `nearest_source_summary` now includes
`p10_distance` (10th percentile nearest-source distance) alongside the
existing `mean_distance` / `median_distance` / `top_nearest_source_objects`. A
low `p10_distance` with a high `category_consistency_rate` is a signal the
model may be scoring well on category by retrieving a near-identical source
sprite rather than generalizing.

The report also exposes `near_copy_rate` (alias of the existing
`nearest_neighbor_duplicate_rate`) plus `near_copy_criterion` and
`near_copy_distance_threshold` fields that document *how* it's computed:
duplicates are detected by exact match of `alpha_silhouette_hash` and
`color_histogram_signature` between two generated samples (not a distance
threshold against source sprites), so `near_copy_distance_threshold` is
reported as `null` rather than an invented number — read `near_copy_rate` as
"how often two generated samples are exact silhouette+color duplicates of each
other," and `nearest_source_summary` (mean/median/p10 distance) as the
separate signal for closeness to *source* sprites.

## 6. Source-relative blob/framing diagnostics

`generator_audits.py`'s `run_full_v4_challenger_audit` already computes
source-relative framing baselines (`source_distribution.full_source` /
`by_category` / `grounded_eval_category_weighted` / `ood_eval_category_weighted`,
each with `border_touch_rate`, `mean_alpha_coverage`, `mean_bbox_width/height`,
`mean_center_offset`, `mean_visible_color_count` computed from source sprites
via `compute_sprite_framing_metrics`) and reports generated-vs-source deltas
(`generated_vs_source.grounded_eval.deltas.border_touch_rate` and the `ood_eval`
equivalent). No new code was added for this Phase 0 pass — read those existing
fields rather than the raw `border_touch_rate` alone, since border-touch is
reported per icon-cropped source sprites too and un-cropped source framing
naturally touches borders more often than a trained model's output.

`generic_blob_collapse_rate` (prompt-faithfulness) does **not** yet have a
source-relative baseline: source sprites are hand-authored, not generated by
this pipeline, and computing an analogous "how often would applying the same
blob-like heuristic to a source sprite flag it as a blob" baseline would reuse
`_structural_stats`'s `generic_blob_like` heuristic from `prompt_faithfulness.py`
against source images the way `_load_source_framing_samples` already does for
framing. This is a plausible small follow-up, not implemented here to keep
Phase 0 diagnostics-only; do not blindly penalize a nonzero
`generic_blob_collapse_rate` without checking whether the underlying prompt
objects (potions, gems, etc.) are inherently blob-shaped.

## 7. Go / no-go interpretation

- If factored CFG improves `color_consistency_rate` to >= 0.88 while keeping
  `category_consistency_rate` >= 0.78 and rare-color / k16 projection
  stability holds (destructive rate stays ~0, median visible colors after
  projection stays low), that combination is a candidate for a `v1.1` export
  preset (CFG parameters only — no retraining, no checkpoint change).
- If field ablations show `object_id` dominates (`category`/`color` degrade
  little when `object_id` is intact but `colors` is nulled, while nulling
  `object_id` alone tanks category) and color remains weak even under
  factored CFG, that's evidence for prioritizing per-group conditioning
  dropout/injection changes in an actual v2 training run, not just sampling
  tweaks.
- If a metric's new point estimate sits inside the old value's `_ci95`
  interval, treat it as noise, not a result — rerun at higher n before
  drawing conclusions, don't chase single-run deltas.

**Update:** a 3-seed confirmation at `cfg_base_scale=2.5, cfg_color_scale=3.0`
passed the promotion rule above narrowly (color +0.03, category -0.019) and is
now available as the optional `v1.1` export preset (`v1` remains the default).
See [`docs/v1_1_factored_cfg.md`](v1_1_factored_cfg.md) for the full
comparison, usage, and decision writeup.
