# Sprite Lab v2 Phase 1 — Conditioning Architecture

No-training diagnostics and infrastructure.

## Phase 0 Ablation Summary (296-prompt suite, 3 seeds)

```text
baseline v1 (caption_semantic_structured):
category  0.7218
color     0.7995
repeated  5.41%
blob      35.14%
potion    1.58%
nearcopy  0.68%

key deltas vs baseline:

null structured:  category -0.307  color -0.233  potion +12.8pp
null colors:      color   -0.293  category +0.126  blob +5.1pp
null category:    category -0.148  color +0.014
null object_id:   category -0.087  color +0.062  blob -6.2pp
null caption:     category -0.013  color +0.001
null semantic:    category -0.033  color +0.002
```

Interpretation:

* Structured fields are the dominant control path.
* Colors field is real, strong, and independent.
* Category field matters independently from object_id.
* Object_id carries shape/category identity but may reduce color flexibility
  (higher color consistency when ablated).
* Caption contributes almost nothing above structured fields.
* Semantic contributes a modest category signal.
* v2 Phase 1 should improve structured conditioning injection and
  object/color disentanglement without touching palette swap.

## Feature Flags

All default **off** to preserve v1/v1.1 behaviour exactly.

### `--film-conditioning`

FiLM (Feature-wise Linear Modulation) in every residual block.

Instead of:

```text
h = h + emb_bias
```

the block computes:

```text
h = h * (1 + gamma(emb)) + beta(emb)
```

Scale paths are initialised near zero so early training is close to
the old additive path but gains richer conditioning capacity.

### `--bottleneck-attention`

Lightweight 4-head self-attention at the U-Net bottleneck (lowest
spatial resolution, typically 8x8 or 4x4 after the encoder).

Applied in residual form:

```text
x = x + Attention(LayerNorm(x))
```

Attention is **not** applied at every level — only the bottleneck —
keeping cost minimal and targeting global shape/category control.

### `--structured-field-dropout-rates`

Per-group structured dropout rates, overriding the scalar
`--structured-field-dropout` for listed groups.

Format:

```text
--structured-field-dropout-rates "category=0.10,object_id=0.35,colors=0.15"
```

Valid group names (matching `STRUCTURED_DROPOUT_GROUPS`):

```
category, object_id, base_object, colors, materials, shapes, function, style
```

Rates must be in [0, 1]. Unlisted groups fall back to the scalar rate.
When `--structured-field-dropout-rates` is omitted, the existing scalar
`--structured-field-dropout` behaviour is unchanged.

## Proposed First Training Command

```powershell
cd C:\Users\Mathieu\Documents\sprite-lab
$env:PYTHONPATH = "src"
$py = "C:\Users\Mathieu\anaconda3\python.exe"

& $py -m spritelab train audit-challenger-full-v4 `
  --dataset datasets\sprite_lab_multisource_v4 `
  --training-manifest datasets\sprite_lab_multisource_v4\training_manifest.jsonl `
  --out experiments\challenger_full_v4_v2_phase1_conditioning `
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
  --structured-field-dropout-rates "category=0.10,object_id=0.35,base_object=0.25,colors=0.15,materials=0.10,shapes=0.10,function=0.10,style=0.10" `
  --ema-decay 0.999 `
  --sample-ema `
  --foreground-rgb-loss-weight 2.0 `
  --background-rgb-loss-weight 0.25 `
  --palette-loss-weight 0.1 `
  --film-conditioning `
  --bottleneck-attention `
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

**Rationale for rates** (from Phase 0 ablation deltas):

| Group       | Rate   | Reason |
|-------------|--------|--------|
| category    | 0.10   | Keep strong — large category drop when ablated |
| object_id   | 0.35   | Moderate dropout — carries identity but may reduce color flexibility |
| base_object | 0.25   | Similar to object_id, slightly less critical |
| colors      | 0.15   | Keep mostly present — strongest individual control signal |
| materials   | 0.10   | Present in training, keep mostly on |
| shapes      | 0.10   | Keep mostly on |
| function    | 0.10   | Keep mostly on |
| style       | 0.10   | Keep mostly on |

**Foreground/background loss weights** (from Phase 0 framing diagnostics):

* Foreground 2.0 — emphasise sprite content over background.
* Background 0.25 — still model background, but 4x less weight.

**Palette loss** (experimental):

* Weight 0.1 — lightweight palette auxiliary loss to reduce chroma noise.

## Evaluation After Training

Use the Phase 0 harness with the built 296-prompt suite:

```powershell
& $py -m spritelab train run-v2-phase0-eval `
  --out experiments\v2_phase1_conditioning_eval `
  --checkpoint experiments\challenger_full_v4_v2_phase1_conditioning\train_25k\checkpoint_last_ema.pt `
  --dataset datasets\sprite_lab_multisource_v4 `
  --build-prompts `
  --prompt-count 384 `
  --prompt-seed 20260706 `
  --presets v1,v1.1 `
  --seeds 20260723,20260724,20260725 `
  --max-samples 384 `
  --device cuda `
  --batch-size 32
```

## Go / No-Go Criteria

Compared to the **Phase 0 v1 baseline** on the same 296-prompt suite:

```text
category >= baseline + 0.03
color    >= baseline + 0.03
blob     <= baseline
potion   <= baseline
nearcopy <= baseline + 0.01
rare     <= 0.01 after projection
QA errors = 0
```

Use the actual baseline from the harness report (not hardcoded values).

If v1.1 (factored CFG on top) also passes with a margin, that is a bonus
signal but not required for Phase 1 pass.

## Caveats

* Palette swap is **not** targeted in this phase; the ablation data suggests
  the main levers are structured conditioning injection and object/color
  disentanglement, not colour remapping.
* FiLM and attention are architectural changes; they increase parameter count
  modestly (~15-20% for FiLM, ~5% for bottleneck attention at current model
  scale).
* Per-group dropout only applies during training; sampling is unaffected.
* All flags default off; existing v1/v1.1 checkpoints and configs remain
  fully compatible.
