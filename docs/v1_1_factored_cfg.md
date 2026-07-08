# v1.1: Optional Factored-CFG Color-Strong Preset

## Summary

**`v1` remains the official, safest default.** `v1.1` is an **optional**
color-strong preset built on top of `v1` with no retraining involved: it uses
the same Phase 1 EMA checkpoint and the same k16 deterministic palette
projection as `v1`, and adds factored CFG (see
[`docs/v2_phase0_diagnostics.md`](v2_phase0_diagnostics.md)) at
`cfg_base_scale = 2.5` and `cfg_color_scale = 3.0`.

In the 3-seed Phase 0 confirmation below, `v1.1` improves color consistency by
about **+0.03** over `v1`, with a small category-consistency tradeoff of about
**-0.019**. `v1.1` must be requested explicitly (`--export-preset v1.1`); it
is never applied by default.

## Why factored CFG here

Plain CFG applies a single scalar to all conditioning fields at once. Factored
CFG (added in v2 Phase 0, see
[`docs/v2_phase0_diagnostics.md`](v2_phase0_diagnostics.md)) splits guidance
into a base term (uncond -> color-stripped conditioning) and a color term
(color-stripped -> full conditioning), each with its own scale:

```text
v = v_uncond + cfg_base_scale * (v_no_color - v_uncond) + cfg_color_scale * (v_full - v_no_color)
```

A 3-seed sweep confirmation compared plain `v1` (CFG 3.0) against
`factored_cfg=true, cfg_base_scale=2.5, cfg_color_scale=3.0` (same v1 k16
export projection in both cases), and the factored setting passed the v1.1
promotion rule, narrowly.

## Aggregate metrics (3-seed Phase 0 confirmation)

Mean +/- stdev over 3 seeds, OOD compositional prompts:

```text
v1:
rare       0.0000 +/- 0.0000
category   0.8106 +/- 0.0142
color      0.8368 +/- 0.0049
repeated   0.0139 +/- 0.0098
blob       0.3021 +/- 0.0340
potion     0.0312 +/- 0.0170
near_copy  0.0000 +/- 0.0000
touch      0.4965 +/- 0.0196
median     12.0000 +/- 0.0000

factored_b2_5_c3_0 (v1.1):
rare       0.0035 +/- 0.0049
category   0.7917 +/- 0.0283
color      0.8681 +/- 0.0260
repeated   0.0208 +/- 0.0000
blob       0.2951 +/- 0.0130
potion     0.0278 +/- 0.0130
near_copy  0.0000 +/- 0.0000
touch      0.4896 +/- 0.0085
median     12.0000 +/- 0.0000

Deltas (v1.1 - v1):
color     +0.0312
category  -0.0189
rare      +0.0035
blob      -0.0069
near_copy +0.0000
```

These are also recorded programmatically as
`spritelab.training.v1_gallery.VALIDATED_V1_1_FACTORED_CFG_REFERENCE` and
embedded in `v1_gallery_report.json` when a gallery is built with
`--export-preset v1.1`. As with the `v1` 96-sample reference in
[`docs/v1_default.md`](v1_default.md), these numbers are a static recorded
result, not recomputed by a single sampling/gallery run — rerun
`prompt-faithfulness` against fresh samples (see
[`docs/v2_phase0_diagnostics.md`](v2_phase0_diagnostics.md) section 4 for
confidence intervals) before treating a new run's numbers as a repeat
confirmation.

## Decision

- This passes the v1.1 promotion rule, but narrowly.
- `v1` remains the safest default for general use.
- `v1.1` is documented as a color-strong preset with a small category
  tradeoff — reach for it when color fidelity matters more than category
  precision for a given use case (e.g. palette/style exploration), not as a
  blanket replacement for `v1`.
- If a future rebaseline at higher n (see
  [`docs/v2_phase0_diagnostics.md`](v2_phase0_diagnostics.md)) shows the
  category delta widening past a Wilson CI boundary, revisit whether `v1.1`
  should remain available as-is.

## Usage

Preset aliases: `v1.1`, `v1_1`, `phase1_v1_1` (all equivalent; `.`/`_` variants
exist because shells and filenames don't always like dots).

### Sampling

```powershell
cd C:\Users\Mathieu\Documents\sprite-lab
$env:PYTHONPATH = "src"
$py = "C:\Users\Mathieu\anaconda3\python.exe"

& $py -m spritelab train sample-generator-challenger `
  --checkpoint experiments\challenger_full_v4_phase1\train_25k\checkpoint_last_ema.pt `
  --prompts experiments\challenger_full_v4_phase1\prompts\ood_compositional_prompts.jsonl `
  --out experiments\v1_1_smoke `
  --export-preset v1.1 `
  --max-samples 16 `
  --device cuda `
  --seed 20260723 `
  --batch-size 16
```

`--export-preset v1.1` fills in the same steps/CFG-scale/max-colors/
alpha-threshold/k16-projection defaults as `--export-preset v1`, then layers
`factored_cfg=true`, `cfg_base_scale=2.5`, `cfg_color_scale=3.0` on top. Every
generated sample's manifest row (`generated_manifest.jsonl`) and the run's
`generation_report.json` config record `export_preset`, `factored_cfg`,
`cfg_base_scale`, and `cfg_color_scale` explicitly, alongside the usual
palette-projection metadata, so a v1.1 run is unambiguous after the fact.
Explicit `--cfg-base-scale` / `--cfg-color-scale` / `--factored-cfg` flags
still take precedence over the preset if you pass them yourself.

### Gallery

```powershell
& $py -m spritelab train build-v1-gallery `
  --out experiments\v1_1_gallery `
  --export-preset v1.1 `
  --device cuda `
  --seed 20260723 `
  --batch-size 32
```

`build-v1-gallery` defaults to `--export-preset v1` (unchanged); passing
`--export-preset v1.1` activates factored CFG for that gallery build only.
The generated `v1_gallery_report.md` title and `preset.name` field say `v1.1`
in that case (and `v1` otherwise), and the JSON report additionally embeds
`validated_v1_1_factored_cfg_reference` (the aggregate metrics above) when the
`v1.1` preset was used.

## No training involved

Nothing above trains or modifies the Phase 1 checkpoint. `v1.1` is a
sampling-time preset only: same weights, same k16 projection, different CFG
math at decode time.
