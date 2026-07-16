# Project Brain

> **Historical milestone record.** This file preserves early codec and labeling
> decisions, but its current milestone, next tickets, and non-goals are not the
> current repository architecture. Several listed non-goals are now implemented.
> Use [`AGENTS.md`](AGENTS.md) for the current search/code map, [`README.md`](README.md)
> for the product overview, and feature documents under `docs/` for current
> workflows.

## Project Goal

Build a native 32x32 palette-index pixel-art generator with clean intermediate
representations for transparency, color palettes, index maps, semantic roles,
and metadata.

## Current Milestone

Milestone 1 created the project skeleton and the Sprite Bundle Codec:
saving, loading, validating, reconstructing, and previewing sprite bundles.

Ticket 4 added Palette Canonicalizer v1: deterministic palette slot statistics,
rough role hints, stable visible-slot ordering, index-map remapping, and
metadata recording for canonicalization reports.

The clean 32x32 RGBA to SpriteBundle encoder adds strict input conversion for
already-clean pixel art: hard alpha extraction, exact palette extraction,
index-map construction, optional palette canonicalization, and a basic role map.

Batch ingestion from a folder of clean 32x32 PNGs adds recursive local file
scanning, strict per-PNG encoding, one bundle directory per accepted sprite,
deterministic manifests, rejection reports, stable IDs, SHA256 provenance, and
optional deterministic split metadata.

The dataset preview grid CLI adds a visual audit path for existing manifests and
bundle folders. It creates deterministic contact sheets from reconstructed
32x32 sprites without modifying the dataset.

The dataset quality report CLI adds automatic diagnostics for existing
SpriteBundle datasets: per-sprite metrics, dataset-level summaries, heuristic
issue codes, JSON/Markdown reports, and issue-specific flag files.

The dataset dedupe report CLI adds deterministic duplicate diagnostics for
existing SpriteBundle datasets: source SHA256 groups, decoded RGBA duplicate
groups, bundle-content duplicate groups, simple near-duplicate groups, and
cross-split leakage reports.

OKLab palette quantization adds an opt-in lossy path for over-color 32x32
sprites: hard alpha extraction, OKLab clustering over opaque pixels, palette and
index-map construction, quantization metadata, direct quantization CLI, and
batch ingestion fallback for too-many-color strict failures.

Role-map heuristic v2 adds deterministic semantic role inference for
palette-index sprites: per-slot role features, palette-slot role assignment,
per-pixel role maps, role validation, and role preview images for debugging.

This project still intentionally does not include ML training.

## Representation Conventions

- Sprites are exactly 32x32 pixels.
- `alpha` is a 32x32 binary mask where `0` is transparent and `1` is opaque.
- `palette` is a `K x 3` RGB array.
- `palette[0]` is a dummy transparent slot, usually `[0, 0, 0]`.
- `index_map == 0` means transparent.
- Opaque pixels must use palette slots `1..K-1`.
- Optional `role_map` values use stable role constants from
  `spritelab.codec.roles`.
- Metadata must be JSON-serializable.
- The encoder is strict and exact.
- The encoder does not quantize.
- Over-color images are rejected instead of silently reduced.
- Reconstructed transparent pixels are normalized to RGBA alpha 0.
- Batch ingestion input PNGs must already be exact 32x32.
- The ingester uses the strict encoder.
- Bad files are written to `rejected.json`, not silently fixed.
- Dataset manifests are deterministic and reproducible for the same inputs.
- Preview grids read existing bundles/manifests and do not modify datasets.
- Preview grids are for visual QA before training.
- Preview sprites are displayed with nearest-neighbor scaling only.
- Quality reports inspect datasets but do not modify them.
- Quality issue codes are heuristic warnings.
- Quality metrics should remain deterministic.
- Quality reports are intended for pre-training QA and later generated-output QA.
- Dedupe reports inspect datasets but do not modify them.
- Decoded RGBA hash is the main visual exact-duplicate signal.
- Bundle content hash checks internal representation duplicates.
- Near-duplicate groups are heuristic warnings.
- Cross-split duplicate leakage is critical before ML training.
- Strict exact encoding remains the default.
- Quantization is opt-in.
- Quantization uses hard alpha and only clusters opaque pixels.
- Quantized outputs must still be valid SpriteBundles.
- Quantization metadata is stored for later QA.
- Role inference is deterministic and heuristic.
- Role inference improves future training signals but is not artist-perfect.
- Slot 0 remains transparent.
- Role previews are debug artifacts.
- Canonicalization must preserve decoded RGBA pixel-perfectly.

## Next Tickets

- Ticket - dataset browser / curation UI
- Ticket - dataset split/export utilities for training
- Ticket - masked index-map inpainting baseline
- Ticket - alpha/silhouette metrics improvements
- Ticket - candidate cleanup / manual curation workflow
- Ticket - palette library / palette retrieval baseline

## Known Non-Goals

- No ML training yet.
- No web scraping.
- No OpenGameArt ingestion yet.
- No palette quantization yet.
- No alpha extraction from arbitrary images yet.
- No Gradio UI yet.
- Palette canonicalization is heuristic only.
- Role hints are approximate and do not update `role_map`.
- OKLab quantization is not implemented yet.
- No human correction UI yet.
- No image resizing or spritesheet splitting yet.

## Label v2 Project Brain

Current golden set path: `evals/golden_v1_small/golden_labels.jsonl`.

Baseline failure evidence: VLM-first/Qwen fusion can confidently mislabel clean
food sprites, including `butter -> gold_bar`, `cheese_wedge -> gold_bar`,
`cheese_wheel -> gold_coin`, `milk_carton -> stone_bottle`, `orange -> coin`,
and `kiwi -> coin_stack`. The old `fused_suggestion` could store the wrong VLM
object even when the bucket was `needs_review`.

Chosen philosophy: filename/source-first, VLM-assisted, golden-eval-gated.
Trusted clean source filenames are authoritative for object/category when
confidence is at least `0.85`; VLM output is descriptor/verifier evidence and
must not override a trusted food/tool/gem filename by itself.

Current default operating thresholds:

- Trusted filename threshold: `0.85`
- Auto VLM threshold when filename is weak: `0.80`
- Trusted filename conflicts: auto-fill safe filename by default and flag for
  review/reporting instead of seeding the GUI with the VLM object
- Exact duplicate propagation: on by default
- Near duplicate propagation: off by default

Latest small-golden sweep result (`evals/golden_v1_small/label_v2_sweep.json`):

- Best measured trusted filename threshold: `0.75`
- Best measured VLM threshold: `0.65`
- Conflict policy: `auto_trusted_filename_conflicts`
- Auto coverage: `0.682`
- Auto precision: `0.956`
- Object token-F1: `0.724`

Command examples:

```bash
python -m spritelab harvest fuse-prefill-v2 \
  --run harvest_runs\oga_cc0_food_ocal \
  --out harvest_runs\oga_cc0_food_ocal\label_v2_suggestions.jsonl

python -m spritelab harvest label-v2-report \
  --run harvest_runs\oga_cc0_food_ocal

python -m spritelab harvest prefill-eval-v2 \
  --golden evals\golden_v1_small\golden_labels.jsonl \
  --runs harvest_runs\oga_cc0_food_ocal,harvest_runs\oga_cc0_tool_ocal,harvest_runs\oga_cc0_gem_7soul1 \
  --prediction-file label_v2_suggestions.jsonl \
  --out evals\golden_v1_small\label_v2_eval.json

python -m spritelab harvest label-v2-sweep \
  --golden evals\golden_v1_small\golden_labels.jsonl \
  --runs harvest_runs\oga_cc0_food_ocal,harvest_runs\oga_cc0_tool_ocal,harvest_runs\oga_cc0_gem_7soul1 \
  --out evals\golden_v1_small\label_v2_sweep.json
```
