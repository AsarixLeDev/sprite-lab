# v2 Status

Last updated: 2026-07

## Current best checkpoint

```
experiments\challenger_full_v4_v2_phase2_palette_index\train_25k\checkpoint_last_ema.pt
```

Config: FiLM + bottleneck attention + palette/index auxiliary heads. 25k steps on multisource v4.

## Recommended export

```powershell
--export-preset v1
--cfg-scale 3.0
--sample-steps 30
--max-colors 16
```

Continuous k16 deterministic palette projection. Same as v1 default.

## Phase summary

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | FiLM, bottleneck attention, factored CFG | Done |
| Phase 1 | Per-group dropout, conditioning experiments | Done (no gain over Phase 0) |
| Phase 2 | Palette/index auxiliary heads | Done (adopted for training, rejected for export) |

See `docs/v2_phase2_palette_index_heads.md` for full details.

## Key metrics (OOD-core, family-weighted)

| Metric | v1 baseline | v2 best |
|--------|------------|---------|
| Category consistency | 0.567 | 0.651 |
| Color consistency | 0.714 | 0.733 |
| Blob collapse | 0.435 | 0.270 |

## Active decisions

* **Keep** v1 export preset (continuous RGBA + k16 projection).
* **Keep** palette/index heads as training-only auxiliary losses.
* **Reject** v1.1 / factored CFG for Phase 2.
* **Reject** head decode as export path.

## Next direction (TBD)

* Investigate better palette RGB supervision / color-slot alignment.
* Color disentanglement without sacrificing category.
* Palette presence loss tuning (BCE currently 1.21).
