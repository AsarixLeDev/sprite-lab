# ML Dataset Format

The `spritelab.ml` package trains and evaluates on Dataset Maker exports.

## Directory layout

```text
datasets/<name>/
  train.npz
  val.npz
  test.npz
  manifest_train.jsonl
  manifest_val.jsonl
  manifest_test.jsonl
  vocab.json
  dataset_config.json
```

`manifest_*.jsonl`, `vocab.json`, and `dataset_config.json` are optional at
load time; a missing `.npz` split file raises `FileNotFoundError`.

## Required npz keys

```text
alpha         uint8/bool  [N, 32, 32]   values 0/1
index_map     int         [N, 32, 32]   0 = transparent, >= 1 = palette slot
role_map      int         [N, 32, 32]   0 = transparent, 255 = unknown opaque
palette       uint8       [N, K, 3]     row 0 = dummy transparent [0, 0, 0]
palette_mask  bool        [N, K]        True for row 0 and real rows
category_id   int64       [N]
sprite_id     str         [N]
```

Where `K = max_palette_slots + 1`.

## Token conventions

```text
0   = transparent
253 = pad (INDEX_PAD)
254 = mask (INDEX_MASK)
```

Normal output classes are `0..max_palette_slots`. Model inputs may contain
`INDEX_MASK`, but predictions never include mask/pad tokens.

## Role-map fallback

```text
0   = transparent
255 = unknown opaque
```

Other role IDs (outline, shadow, midtone, light, highlight, accent, emissive,
texture detail) are defined in `spritelab.codec.roles`.

## Commands

- `ml validate-dataset` — validates the array contract of one split.
- `ml baseline-eval` — evaluates dumb baselines under a fixed opaque mask and
  writes `baseline_metrics.json` plus preview grids.
- `ml overfit-smoke` — trains a tiny sanity-check model on a few samples and
  writes `metrics.json` plus `predictions.png`.

## 4-item overfit example

```bash
PYTHONPATH=src python -m spritelab ml validate-dataset --dataset datasets/v0 --split train
PYTHONPATH=src python -m spritelab ml baseline-eval --dataset datasets/v0 --split train --mask-fraction 0.5 --out outputs/baselines_v0_4items
PYTHONPATH=src python -m spritelab ml overfit-smoke --dataset datasets/v0 --split train --max-samples 4 --steps 500 --batch-size 4 --mask-fraction 0.5 --out outputs/overfit_v0_4items
```
