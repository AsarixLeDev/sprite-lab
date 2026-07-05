# Dataset Maker GUI

## Purpose

The Dataset Maker GUI converts user PNGs plus human-entered metadata into ML-ready dataset files for the future Phase 7-9 `SpriteBundleDataset`.

Input PNGs are inspected, validated, optionally quantized, reviewed, accepted or rejected, and exported to fixed-shape `.npz` arrays with matching JSONL manifests.

## Expected PNG input

- PNG
- 32x32
- RGBA or convertible to RGBA
- hard alpha only, with alpha values 0 or 255
- limited palette, or quantizable to `max_palette_slots`

Wrong-size PNGs are rejected by default. The GUI can optionally resize to 32x32 using nearest-neighbor sampling.

## GUI workflow

1. Import PNGs from uploads or a local directory.
2. Review sprite, alpha, role-map, palette, validation errors, and warnings.
3. Add details such as `sprite_id`, category, tags, source, license, author, notes, and split override.
4. Accept, reject, mark needs-fix, or quarantine each sprite.
5. Export the dataset.

Raw PNGs are never modified.

## Export format

The exporter writes:

```text
datasets/<dataset_name>/
  train.npz
  val.npz
  test.npz
  manifest_train.jsonl
  manifest_val.jsonl
  manifest_test.jsonl
  vocab.json
  dataset_config.json
  dataset_report.md
  rejected.jsonl
```

Rejected, needs-fix, and quarantine sprites are excluded from training arrays and recorded in `rejected.jsonl`.

## Required npz keys

Each split `.npz` contains:

```text
alpha         uint8  [N, 32, 32]             values 0 or 1
index_map     int16  [N, 32, 32]             0 = transparent
role_map      uint8  [N, 32, 32]             0 = transparent, 255 = unknown fallback
palette       uint8  [N, max_palette_slots + 1, 3]
palette_mask  bool   [N, max_palette_slots + 1]
category_id   int64  [N]
sprite_id     string [N]
```

Palette row 0 is always dummy transparent RGB `[0, 0, 0]`, and `palette_mask[:, 0]` is always `True`. Visible palette rows are copied into rows `1..K`; padded rows are `[0, 0, 0]` with mask `False`.

Hard validity rules:

- transparent pixels have `index_map == 0`
- opaque pixels have `index_map >= 1`
- `index_map` never points to a `False` palette row
- every sample is exactly 32x32
- exported `sprite_id` values are unique

## Commands

Launch the GUI:

```bash
python -m spritelab dataset-maker --output-root datasets --host 127.0.0.1 --port 7860
```

Headless import/export:

```bash
python -m spritelab dataset-maker-import-export \
  --png-dir raw_pngs \
  --dataset-name v0 \
  --output-root datasets \
  --category item_icon \
  --tags "minecraft,item_icon" \
  --max-palette-slots 32 \
  --quantize-overcolor \
  --infer-role-map
```

## Notes

- raw PNGs are never modified
- rejected sprites are excluded from training arrays
- role-map fallback is transparent outside alpha and unknown inside alpha
- this output is meant for Phase 7-9 `SpriteBundleDataset`
- optional Qwen metadata auto-fill is documented in `docs/qwen_prefill.md`
