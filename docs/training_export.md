# Training Export

## Purpose

Training export turns latest `accepted` curated `SpriteBundle` directories into
fixed-shape arrays for future ML training. It does not train a model and does
not mutate raw bundles.

## Required inputs

- Bundle root containing saved SpriteBundle directories.
- `curation.jsonl` with latest human decisions.
- Optional quality report JSON.
- Optional dedupe report JSON.

Only sprites whose latest curation status is `accepted` are exported. Rejected,
quarantine, needs-fix, and uncurated bundles are excluded.

## Outputs

An export directory contains:

- `train.npz`
- `val.npz`
- `test.npz`
- `manifest_train.jsonl`
- `manifest_val.jsonl`
- `manifest_test.jsonl`
- `vocab.json`
- `palette_semantics_report.json`
- `palette_semantics_report.md`
- `training_readiness_report.md`
- `export_config.json`

## Array contract

Each split `.npz` contains:

| Key | Dtype | Shape |
|---|---|---|
| `alpha` | `uint8` | `[N, 32, 32]` |
| `index_map` | `int16` | `[N, 32, 32]` |
| `role_map` | `uint8` | `[N, 32, 32]` |
| `palette` | `uint8` | `[N, max_palette_slots + 1, 3]` |
| `palette_mask` | `bool` | `[N, max_palette_slots + 1]` |
| `category_id` | `int64` | `[N]` |
| `sprite_id` | string | `[N]` |

Palette row `0` is always the dummy transparent slot. Palette rows beyond a
sprite's actual palette length are zero-padded and masked out. If a bundle has
no role map, export creates a deterministic fallback: transparent pixels become
`ROLE_TRANSPARENT`, opaque pixels become `ROLE_UNKNOWN`.

## Split safety

If a dedupe report is provided, duplicate and near-duplicate groups are kept in
the same split. If no dedupe report is provided, export still works but the
readiness report warns that near-duplicate leakage cannot be guaranteed.

## Readiness gate

The readiness report fails on hard errors such as:

- no accepted sprites;
- zero exported sprites;
- invalid accepted bundles;
- duplicate leakage across splits;
- palette size above `max_palette_slots`;
- invalid exported token values;
- missing required split files.

It warns on weak but usable datasets, such as tiny exports, missing dedupe or
quality reports, many unknown categories, missing role maps, high palette-slot
role entropy, or high palette-size variance.

## Example commands

Palette report:

```bash
python -m spritelab training palette-report \
  --bundles outputs/bundles \
  --curation curation.jsonl \
  --out reports/palette_semantics_report.md \
  --json reports/palette_semantics_report.json
```

Training export:

```bash
python -m spritelab training export \
  --bundles outputs/bundles \
  --curation curation.jsonl \
  --out datasets/v0 \
  --dataset-name v0 \
  --train 0.8 \
  --val 0.1 \
  --test 0.1 \
  --seed 1337 \
  --max-palette-slots 32 \
  --quality-report reports/quality_report.json \
  --dedupe-report reports/dedupe_report.json
```

Rebuild readiness report for an export:

```bash
python -m spritelab training readiness --export datasets/v0
```
