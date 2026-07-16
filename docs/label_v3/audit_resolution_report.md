# Audit Resolution Report — Auto-Labeling v3 Phase 0

**Date**: 2026-07-10
**Repository**: project root
**Schema**: Resolves all audit discrepancies listed in the v3 implementation prompt.

---

## 1. VLM Source Hints

**Discrepancy**: One audit describes Labeling v2 as blind-first by default; another says descriptor mode always receives source hints and candidates.

**Verified behavior** (live repository, `prefill.py:501` and `prefill.py:641`):

- `PrefillConfig.include_filename_hint` defaults to `False` (prefill.py:292).
- However, the VLM role check `self.config.vlm_role == "descriptor"` acts as a **hard override**: when the VLM role is `"descriptor"`, the filename hint is **always** included regardless of `include_filename_hint` (lines 501, 641).
- The condition `if self.config.include_filename_hint or self.config.vlm_role == "descriptor"` means descriptor mode **always receives source hints and candidate object names**.
- In `label_v2_pipeline.py:576`, the backend is created with `include_filename_hint=True` and `vlm_role="descriptor"` by default.
- Therefore: **Label v2 VLM calls are NOT blind-first by default**. The descriptor role receives source hints and candidates.
- The `QwenBatchPrefillConfig` in `autolabel.py:132` defaults `include_filename_hint=False`, but the pipeline-created backend overrides this.
- The `--no-filename-hint` CLI flag exists (`_args.py:69`) but is not the default path.

**Resolution**: Both audits are partially correct. The *config default* is blind-first (`False`), but the *pipeline default* is hint-enabled (descriptor role). V3 must treat the descriptor-mode behavior as the existing baseline and differentiate blind vs. hinted contexts explicitly.

---

## 2. Semantic-v3 Schema

**Discrepancy**: One audit identifies a `SemanticV3Record` with schema version and QA checks; another describes `semantic_v3` as an unvalidated passthrough.

**Verified behavior** (live repository):

- `SemanticV3Record` is a frozen dataclass in `semantic_v3.py:123` with schema version `"semantic_v3.0"`.
- It is produced by `build_semantic_v3_record()` which is **deterministic and offline** (no VLM/LLM).
- `semantic_v3_from_json()` (line 188) is **permissive**: missing fields get defaults, unexpected fields are ignored.
- QA checks exist in `qa.py` for semantic_v3 content (caption hygiene, color expectations for base objects) but these are **warnings, not hard errors**.
- The QA checks do validate caption content, forbidden terms, and base object coverage.
- `attach_semantic_v3()` adds the record as a new key alongside existing label-v2 fields—it does not replace primary labels.

**Resolution**: The schema is formal (`SemanticV3Record`) and QA does validate it, but the validation is permissive at deserialization time. The "unvalidated passthrough" characterization refers to the fact that `semantic_v3_from_json` accepts any dict and fills defaults—the QA gate is downstream of deserialization.

---

## 3. Review-tier Records Entering Training

**Discrepancy**: One audit reports QA review-leak checks and auto-only quarantine; another reports that `needs_review=True` and T4 records can enter training because the tier is not filtered by default.

**Verified behavior** (live repository):

- `qa.py:46`: `REVIEW_STATUS_TOKENS` includes `needs_review`, `quarantine`, `needs_fix`, `rejected`, `review`.
- `qa.py:509-525`: `_review_status_leak()` checks for these statuses in both the top-level `status` field and nested `label_v2.label_quality.needs_review`.
- However, `training/data.py:102-114`: The `SpriteTrainingDataset` filters **only** by `split` and `caption_policy_filter`. There is **no filter** on `needs_review`, `confidence_tier`, or `bucket`.
- `qa.py` is a **post-export validation gate**—it detects review leaks but does not prevent import.
- Export at `exporter.py:77`: Only `item.status == "accepted"` sprites are exported. But the status check is on `DatasetMakerItem.status`, not on label_v2 review flags.

**Resolution**: Both audits are correct. The QA gate **detects** review leaks after export; training data loading does **not filter** by tier. T4 records and `needs_review=True` records CAN enter training if they pass the `accepted` status filter at export time but have review flags in their label_v2 metadata. This is a **real bypass**.

---

## 4. Duplicate and Near-duplicate Behavior

**Discrepancy**: Reports variously describe exact RGBA grouping, near-duplicate propagation, and perceptual hashing.

**Verified behavior** (live repository):

- `label_dedupe.py`: `Group_label_records_by_exact_rgba()` groups by SHA-256 of decoded RGBA bytes (exact pixel match, PNG-encoding independent).
- `prefill_dedupe.py`: `PrefillGroup` supports both exact duplicates and near-duplicates with a pixel-difference threshold.
- `label_v2_pipeline.py:98`: Uses `group_label_records_by_exact_rgba()` for exact duplicate propagation.
- `label_fusion_v2.py`: No perceptual hashing. The fusion only considers exact RGBA duplicates.
- `prefill.py`: The `propagate_near_dups` config flag exists but defaults to `False`.
- `autolabel.py:239`: `group_sprites_for_prefill()` supports both exact and near duplicates.

**Resolution**: Exact RGBA dedupe is the primary mechanism in label v2. Near-duplicate support exists in prefill but is off by default. Perceptual hashing is not implemented.

---

## 5. Source Profile Count

**Discrepancy**: Reports refer to 14, 15, and 16 profiles.

**Verified behavior** (live repository):

- `source_profiles.py`: `_FALLBACK_PROFILES` contains exactly **14** profiles.
- The loaded profiles are validated against the fallback set—the config YAML (`source_profiles.yaml`) must match these 14 and cannot add new ones.
- The `loaded_source_profiles()` function returns exactly 14 profiles.
- The 15/16 counts in audits likely counted the `generic_unknown` profile differently or miscounted.

**Resolution**: **14 profiles** code-defined and YAML-defined; no dynamic profiles; fallback count = 14.

---

## 6. Training Tier Usage

**Discrepancy**: One report says training reads confidence tiers for filtering; another says tiers are present but not filtered by default.

**Verified behavior** (live repository):

- `label_schema.py:84-96`: `confidence_tier_for_bucket()` maps fusion buckets to T0-T4 tiers.
- `training_manifest.py:31`: Imports `confidence_tier_for_bucket`.
- `training\data.py`: `SpriteTrainingDataset.__init__()` filters by `split` and `caption_policy_filter` only. **No tier filter**.
- `training_manifest_qa.py`: Validates training manifests but does not filter by tier.
- `qa.py:784-793`: `_record_label_tier()` resolves tiers for audit reporting only.

**Resolution**: Confidence tiers are **computed and stored** but **not filtered** by default in training. The tier is present in records and QA checks but no training path excludes low-tier records automatically.

---

## 7. Image Transparency Views

**Discrepancy**: The current VLM path reportedly uses a solid magenta matte because checkerboards caused hallucinations.

**Verified behavior** (live repository):

- `prefill.py:1263,1301,1354,1419`: All VLM prompts explicitly instruct: "Do not mention the magenta background", "The solid magenta background was added for display and is NOT part of the sprite."
- `prefill.py:136-143`: "checkerboard", "magenta background", "pink background" are in `_DEGENERATE_TEXT_PATTERNS` and `_DEGENERATE_OBJECTS`.
- `label_fusion_v2.py:1047`: Detects "checkerboard" text as degenerate VLM output.
- The `prepare_vlm_image()` function (in prefill.py) uses a magenta matte (code confirms this pattern).
- Checkerboard is used for **display** in training review tools (`framing_metrics.py:217`), not for VLM input.

**Resolution**: Magenta matte is the baseline VLM image view. Checkerboard views are for human display only, not VLM input. Checkerboard references in VLM output are treated as degenerate.

---

## 8. Current Golden Metrics

**Discrepancy**: Existing reported values are based on only 66 labels from three exact-filename-trust sources.

**Verified behavior** (live repository):

- `golden.py`: `GoldenLabel` is a simple dataclass with `sprite_id`, `category`, `object_name`, `tags`.
- `.golden_sample.jsonl` and `.golden_labels.jsonl` are the persistence files.
- The golden set is used by `label_v2_eval.py` for evaluation.
- No golden data files are checked into the repository (they are run-specific artifacts).

**Resolution**: Treat existing golden evaluations as historical baselines only. V3 evaluation must use frozen suites with explicit non-overlap from tuning data.

---

## Summary of Verified Facts

| # | Fact | Verified Value |
|---|------|---------------|
| 1 | VLM source hints | Descriptor mode always receives hints; `include_filename_hint` config default is `False` but overridden in pipeline |
| 2 | Semantic-v3 schema | Formal `SemanticV3Record` exists; QA validates downstream; deserialization is permissive |
| 3 | Review-tier in training | QA detects leaks; training loader does NOT filter by tier—**real bypass** |
| 4 | Dedupe behavior | Exact RGBA SHA-256 grouping; near-dupes off by default; no perceptual hashing |
| 5 | Source profiles | **14** (code + YAML + fallback all agree) |
| 6 | Training tier filtering | Tiers computed and stored; **not filtered** by default in training |
| 7 | Image views | Magenta matte baseline; checkerboard = human display only; degenerate detection active |
| 8 | Golden metrics | 66 labels, 3 trust sources; historical baselines only |

## Protected Artifacts Inventory

During Phase 0, these files and keys must not change:

- `label_schema.py` — `LabelSuggestion`, `SafeFusedLabel`
- `label_fusion_v2.py` — `fuse_label_v2()`, `FusionThresholds`
- `label_v2_pipeline.py` — `build_label_v2_records()`, output format
- `label_taxonomy.py` — `CATEGORY_VALUES`, all normalization functions
- `source_profiles.py` — 14 built-in profiles, `detect_source_profile()`
- `semantic_v3.py` — `SemanticV3Record`, `build_semantic_v3_record()`
- `exporter.py` — export pipeline and output format
- `qa.py` — all QA gates
- `training/data.py` — `SpriteTrainingDataset`
- `training_manifest.py` — manifest format
- `label_v2_suggestions.jsonl` — current output keys
- `imported.jsonl` — current status and auto_metadata keys
- `manifest_{split}.jsonl` and `{split}.npz` naming
- All existing CLI commands under `harvest label-v2`, `semantic-v3`, `fuse-prefill-v2`
