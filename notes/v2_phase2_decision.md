# v2 Phase 2 — Decision

Date: 2026-07

## Decision

**Adopt** palette/index auxiliary losses for training.

**Reject** head decode as export path.

## Rationale

### Training improvement

Palette/index auxiliary losses improve OOD-core blob collapse from 0.435 → 0.270 and category from 0.567 → 0.651. Color improves modestly (+0.019).

### Head decode rejection

Predicted index + palette reconstruction drops color consistency from 0.719 to 0.510 with no blob improvement. The index head is accurate (91.9% visible-pixel) and the presence head is excellent (F1 0.988), but the predicted palette RGB is too imprecise (MAE 0.076) for direct export.

### v1.1 rejection

Factored CFG trades category for color (-0.073 category, +0.049 color on OOD-core). Remains a color-control option, not a default.

## Recommended export

```
--export-preset v1
--cfg-scale 3.0
--sample-steps 30
--max-colors 16
```

## Recommended future work

1. Better palette RGB supervision or color-slot alignment.
2. Color disentanglement without sacrificing category.
3. Palette presence loss tuning.
4. Do NOT train more conditioning-only variants.
