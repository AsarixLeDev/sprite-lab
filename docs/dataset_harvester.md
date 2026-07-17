# Dataset Harvester

## Purpose

Turns public packs / ZIPs / directories into ML-ready SpriteLab datasets
(`train.npz` / `val.npz` / `test.npz`, the format consumed by `spritelab.ml`).

The hierarchy is strict:

```text
deterministic code = validity
source manifest = license/provenance
Qwen/local VLM = metadata suggestions only
human/bulk rules = final approval policy
exporter = strict dataset format
```

Qwen never decides license, author, source URL, copyright status, split, or
image validity - it only suggests category, object name, tags, materials, mood,
a short description, and a sprite ID. Dominant colors are computed
deterministically from the bundle palette.

## Recommended source policy

- Prefer `own_work` and `cc0`.
- Use `cc_by` / `oga_by` only with attribution metadata (author + source URL).
- Treat unknown/custom licenses as quarantine by default.
- Do not use ripped commercial sprites.

This document is not legal advice; the tool preserves metadata and warns.

## Source workflows

- **Manual ZIP import** — download the pack yourself (works even for sites
  that block automation), then `harvest import-zip`.
- **Local directory import** — point at a folder of PNGs with `harvest import-dir`.
- **Direct ZIP URL import** — `harvest download-zip` streams the file,
  computes SHA256, and extracts safely.
- **Known registry** — `data/source_registry.example.json` lists known packs
  as a convenience, not a legal guarantee.

### Smart source prefill

Paste one public pack-page URL instead of repeating its title, source ID, and
every known metadata link. `source-prefill` is network-free and recognizes
OpenGameArt, Kenney, and itch.io, with a conservative generic fallback:

```bash
python -m spritelab harvest source-prefill \
  https://kenney.nl/assets/new-platformer-pack
```

The same defaults are applied directly by every import command when
`--source-url` is present, so a Kenney ZIP needs only the local ZIP, run name,
pack page, and explicit license confirmation:

```bash
python -m spritelab harvest import-zip \
  --zip path/to/New-Platformer-Pack.zip \
  --run-name new_platformer_pack \
  --source-url https://kenney.nl/assets/new-platformer-pack \
  --user-confirmed-license
```

Explicit CLI fields always override prefills. Kenney supplies its creator,
CC0 evidence, and terms defaults. OpenGameArt and itch.io licenses remain
pack-specific and are never inferred from the host; supply `--author` and
`--license` after reviewing the page. Selecting `--license cc0` fills the
canonical CC0 evidence URL, but `--user-confirmed-license` remains explicit.
Use `--source-preset generic` to disable host recognition or an explicit
platform preset to reject a mismatched/spoofed host.

## Public source examples

- **Kenney**: manual ZIP or direct URL when available; usually CC0, prefilled
  as author "Kenney" — still record the pack page URL.
- **OpenGameArt**: check the per-pack license carefully (cc0, cc_by, cc_by_sa,
  oga_by, public_domain, wtfpl); author is required when attribution is required.
- **itch.io**: check the per-pack license carefully; "free" never means CC0.
- **Lospec**: useful for palettes, not sprite image datasets.

## Large-scale workflow (10k–100k)

1. import each source (`import-zip` / `import-dir` / `download-zip`);
2. sheets are sliced automatically (32×32 default, 16×16 or custom grids supported;
   small tiles can be center-padded to 32×32);
3. rule-based autolabel runs during import (path/pack-name keywords);
4. `label-v2` creates source-aware filename suggestions for coded sprite names;
5. `qwen-prefill` optionally batches VLM suggestions with retries and an
   on-disk cache keyed by image hash, so re-runs resume where they stopped;
6. `fuse-prefill-v2` safely combines existing Qwen output with label-v2;
7. `apply-policy` bulk-accepts/quarantines/rejects;
8. sample-review in the GUI (paginated browsing, previews on demand);
9. `export` writes the dataset.

Run state lives in JSONL files under `harvest_runs/<run_name>/` so every step
is resumable.

## Commands

```bash
# GUI
python -m spritelab harvest gui --output-root datasets --run-root harvest_runs --port 7861

# Manual ZIP
python -m spritelab harvest import-zip \
  --zip path\to\pack.zip \
  --run-name kenney_generic_items \
  --source-id kenney_generic_items \
  --source-name "Kenney Generic Items" \
  --source-url "<asset page url>" \
  --license cc0 --author Kenney --user-confirmed-license \
  --max-palette-slots 32 --slice-sheets --tile-size 32

# Local directory
python -m spritelab harvest import-dir \
  --dir raw_candidates\v1_micro \
  --run-name own_v1_micro \
  --source-id own_v1_micro \
  --source-name "Own v1 micro candidates" \
  --license own_work --author "Monseigneur" --user-confirmed-license

# Direct ZIP URL
python -m spritelab harvest download-zip \
  --url "<direct zip url>" \
  --run-name pack_run --source-id pack_id --source-name "Pack Name" \
  --license cc0 --author "Author" --user-confirmed-license

# Qwen prefill
python -m spritelab harvest qwen-prefill \
  --run harvest_runs\kenney_generic_items \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --cache-dir .prefill_cache --max-items 1000 --workers 4

# Qwen prefill through native Ollama on Windows
python -m spritelab harvest qwen_prefill \
  --run harvest_runs\kenney_generic_items \
  --backend ollama \
  --model qwen2.5vl:7b \
  --base-url http://127.0.0.1:11434 \
  --cache-dir .prefill_cache --max-items 1000 --workers 2

# Qwen prefill through a RunPod OpenAI-compatible endpoint
python -m spritelab harvest qwen_prefill \
  --run harvest_runs\kenney_generic_items \
  --backend openai_compatible \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url https://<your-runpod-endpoint>/v1 \
  --runpod-token %RUNPOD_API_KEY% \
  --cache-dir .prefill_cache --max-items 1000 --workers 8

# In PowerShell, use:
#   --runpod-token $env:RUNPOD_API_KEY

# Worker count controls concurrent prefill requests. Keep it at 1 if your
# local Ollama model runs out of VRAM; raise it for RunPod or servers that can
# handle parallel requests.

# Reliability options:
#   --retry-attempts 2
#   --min-qwen-confidence 0.55
#   --fusion-policy weighted
#   --filename-hint             # off by default; blind-first is safer
#   --no-propagate-dups         # disable exact duplicate propagation
#   --propagate-near-dups       # opt in to perceptual near-duplicate propagation
#   --near-dup-threshold 2

# Safely combine existing Qwen suggestions with source-aware label-v2 rules.
python -m spritelab harvest fuse-prefill-v2 \
  --run harvest_runs\kenney_generic_items \
  --out harvest_runs\kenney_generic_items\label_v2_suggestions.jsonl

# For deterministic source-aware labels without a VLM request:
python -m spritelab harvest label-v2 \
  --run harvest_runs\kenney_generic_items \
  --no-vlm

# Assisted golden correction GUI
python -m spritelab harvest assisted-golden \
  --run harvest_runs\kenney_generic_items \
  --n 250 --seed 1337 \
  --port 7862

# Bulk policy
python -m spritelab harvest apply-policy \
  --run harvest_runs\kenney_generic_items \
  --auto-accept-valid-cc0 --quarantine-unknown-license --reject-invalid

# Export
python -m spritelab harvest export \
  --run harvest_runs\kenney_generic_items \
  --dataset-name v1_micro_64 --output-root datasets \
  --train 0.8 --val 0.1 --test 0.1 --seed 1337 --overwrite
```

## Output

```text
harvest_runs/<run_name>/
  sources.jsonl
  candidates.jsonl
  imported.jsonl
  rejected.jsonl
  qwen_suggestions.jsonl
  fused_suggestions.jsonl
  golden_candidates.jsonl
  golden_labels.jsonl
  golden_assisted_state.json
  events.jsonl
  harvest_report.md
  harvest_report.json
  extracted/  sliced/  padded/  downloads/

datasets/<dataset_name>/   (standard Dataset Maker export)
```

## Safety

- Raw assets are never modified; slicing/padding writes new files.
- Licenses and provenance are preserved in manifests.
- Export is blocked for `unknown` / `noncommercial` / `no_derivatives` /
  `all_rights_reserved` / `custom_unreviewed` licenses unless
  `--allow-unknown-license` is passed — and even then samples are marked
  with a `license_override` tag/quality issue.
- Qwen suggestions are not legal facts.
- Prefill quality buckets distinguish request failures, invalid JSON,
  warning-only responses, low confidence, filename/Qwen conflicts,
  automatic fusion, and samples needing review.
- ZIP extraction ignores absolute paths and `..` traversal entries.

## Assisted golden labeling

Purpose: use existing auto-labels as prefill so the human corrects instead of
labeling from scratch. The GUI loads filename-rule, cached Qwen, fused, and
existing run metadata; labels autosave immediately to `golden_labels.jsonl`.

```bash
python -m spritelab harvest assisted-golden \
  --run harvest_runs\oga_496_rpg_icons_32fix \
  --n 250 \
  --seed 1337 \
  --port 7862
```

You can prebuild the candidate file without launching the GUI:

```bash
python -m spritelab harvest assisted-golden-sample \
  --run harvest_runs\oga_496_rpg_icons_32fix \
  --n 250 \
  --seed 1337
```

Workflow:

1. Run Qwen/rules/fusion prefill.
2. Launch assisted golden GUI.
3. Accept obvious labels.
4. Correct wrong labels.
5. Add notes for ambiguous or bad crops.
6. Stop any time; labels autosave.
7. Resume later with the same command.
8. Run `prefill-eval-v2`.

Fields: `category`, `object_name`, `tags`, and `notes`.

Speed controls are visible buttons: `Accept as-is`, `Save + next`, `Skip`,
`Mark unknown`, and note buttons for `ambiguous`, `bad_crop`, and
`tile_not_item`. Keyboard shortcuts can be added later if the active Gradio
version supports them cleanly.

Tips:

- Use `unknown` only when the object cannot be identified.
- Useful notes: `ambiguous`, `bad_crop`, `tile_not_item`, `duplicate`, `mismatch`.
- Keep tags short and snake_case.

## Label v2: filename/source-first safe prefill

`label-v2` is the preferred scalable labeling path for clean asset packs. It
uses source/profile-aware filename parsing as the primary authority, adds
deterministic visual facts, treats VLM output as a descriptor/verifier, and
writes additive v2 files without changing old `qwen_suggestions.jsonl` or
`fused_suggestions.jsonl`.

Why this exists: free-form VLM labels can confidently misidentify simple food
sprites, for example `butter -> gold_bar`, `cheese_wedge -> gold_bar`,
`milk_carton -> stone_bottle`, `orange -> coin`, and `kiwi -> coin_stack`.
For trusted clean packs, the filename/source profile wins; the VLM answer is
retained as descriptor/conflict evidence only.

```bash
# Safe fusion from existing Qwen suggestions; no remote VLM call.
python -m spritelab harvest fuse-prefill-v2 \
  --run harvest_runs\oga_cc0_food_ocal \
  --out harvest_runs\oga_cc0_food_ocal\label_v2_suggestions.jsonl

# Label-v2 run without VLM calls.
python -m spritelab harvest label-v2 \
  --run harvest_runs\oga_cc0_food_ocal \
  --no-vlm

# Descriptor-mode VLM, optionally skipped for high-confidence filenames.
python -m spritelab harvest label-v2 \
  --run harvest_runs\oga_cc0_food_ocal \
  --backend openai_compatible \
  --model "qwen/qwen3-vl-32b-instruct" \
  --base-url "$base" \
  --api-key "$RUNPOD_API_KEY" \
  --cache-dir .prefill_cache_runpod_label_v2 \
  --vlm-role descriptor \
  --vlm-only-when-needed

# Report current v2 coverage, conflicts, hallucination flags, and duplicates.
python -m spritelab harvest label-v2-report \
  --run harvest_runs\oga_cc0_food_ocal
```

Outputs:

```text
label_v2_suggestions.jsonl
label_v2_summary.json
label_v2_report.md
```

Each v2 row contains `safe_prefill`, `filename_suggestion`, `vlm_descriptor`,
`visual_facts`, `label_quality`, provenance, flags, and the compatibility
`fused_suggestion` field. Review and assisted-golden flows should use
`safe_prefill` for editable fields.

Evaluation and threshold sweep:

```bash
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

Operating thresholds should be chosen from the sweep report, with the headline
goal of maximum auto coverage while keeping auto precision at or above 0.95 on
the golden set.
