# Semantic Labeling v3 Architecture

Status: implemented (foundation pass, 2026-07).
Owner modules: `spritelab.harvest.semantic_v3`, `spritelab.harvest.semantic_extractors`,
`spritelab.harvest.creative_concepts`.

This document describes the architectural pivot from *exact object-name
classification* to *compositional semantic generation metadata*, and the
concrete foundation implemented in this pass.

---

## 1. Current pipeline summary

The pipeline that produced the 496 RPG icon reference pack:

```text
harvest import (import-zip / import-dir)
  -> imported.jsonl (one record per accepted 32x32 sprite)
  -> label-v2 (filename rules + source profiles + optional VLM descriptor + fusion)
       -> label_v2_suggestions*.jsonl  (safe_prefill / bucket / flags / review routing)
  -> apply-label-v2 (safe auto buckets applied, review bucket quarantined)
       -> imported.jsonl updated, label_v2_review_queue.jsonl, audit in auto_metadata
  -> export (npz rasters per split + manifest_{train,val,test}.jsonl + vocab/config)
  -> dataset-qa (raster contract, manifest integrity, label-v2 audit, split checks)
```

Authority model that must not change:

* `label-v2` owns primary `category` and `object_name` safety
  (`safe_prefill` is the single applied label source).
* `apply-label-v2` owns review/quarantine routing and the audit trail
  (`auto_metadata.label_v2_*`).
* `dataset-qa` owns exported-dataset integrity and never mutates datasets.

Reference checkpoint (496 pack): 494 predictions, 476 applied/exported,
18 quarantined, 0 QA errors, 588 tests passing.

## 2. Why exact object-name optimization is insufficient

The 496 work optimized "label this exact pack correctly". That produces a
**closed taxonomy**: `golden_chestplate`, `red_potion`, `fire_cannon` are
opaque strings. A generator trained on those strings can only reproduce seen
names. It cannot answer:

* easy unseen composition — `square_gem`, `blue_crystal_hammer`, `mossy_key`;
* creative concepts — `charged_sinew`, `calming_spores`, `moonlit_resin`;
* reference-guided concepts — "this dolphin photo, as a 32x32 icon".

Composition requires the dataset to expose the *factors* each sprite is made
of: a base object carrying visual identity, plus attributes (color, material,
shape, effect, state, function, mood, style) and grounded natural-language
captions that recombine those factors. `golden_chestplate` is one data point;
`chestplate + {gold, metal, protection, torso_shaped}` teaches the model what
`silver_chestplate` or `crystal_chestplate` would mean.

Repeating the 496 approach per pack (hand-tuned exact object maps) also
overfits the labeling pipeline itself: every new source would need its own
trial-and-error loop. Source adaptation must instead extract *reusable
semantic components*.

## 3. Target semantic schema (`semantic_v3.0`)

Implemented in `src/spritelab/harvest/semantic_v3.py` as frozen dataclasses,
JSON-safe, tuple-based like the rest of the project:

```python
@dataclass(frozen=True)
class SemanticAttributes:
    colors: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    shapes: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    function: tuple[str, ...] = ()
    mood: tuple[str, ...] = ()
    style: tuple[str, ...] = ()
    parts: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()

@dataclass(frozen=True)
class SemanticV3Record:
    schema_version: str          # "semantic_v3.0"
    category: str                # copied from safe_prefill, never changed
    object_name: str             # copied from safe_prefill, never changed
    base_object: str             # visual-identity carrier ("chestplate")
    open_name: str               # human-readable open-vocab name ("golden chestplate")
    attributes: SemanticAttributes
    aliases: tuple[str, ...]     # grounded synonyms (VLM alternatives, candidates)
    captions: tuple[str, ...]    # multiple grounded caption styles
    prompt_phrases: tuple[str, ...]
    negative_tags: tuple[str, ...]
    source_evidence: dict        # which inputs produced which fields
    warnings: tuple[str, ...]
```

JSON shape in prediction files and exported manifests:

```json
{
  "semantic_v3": {
    "schema_version": "semantic_v3.0",
    "category": "material",
    "object_name": "ruby_gem",
    "base_object": "gem",
    "open_name": "ruby gem",
    "attributes": {"colors": ["red"], "materials": ["crystal", "mineral"], ...},
    "aliases": ["ruby", "red gemstone"],
    "captions": ["ruby gem", "red gem made of crystal", "32x32 pixel art ..."],
    "prompt_phrases": ["32x32 pixel art ruby gem", "..."],
    "negative_tags": ["photorealistic", "large_scene", "text", "watermark"],
    "source_evidence": {...},
    "warnings": []
  }
}
```

Stability rules:

* `schema_version` is bumped on any breaking field change.
* `label_v2` metadata is preserved separately and untouched.
* All list fields are order-stable and deduplicated.

## 4. Migration strategy: label-v2 records -> semantic-v3 records

Semantic-v3 is a **pure, deterministic, offline enrichment layer**. It reads a
label-v2 prediction JSONL and writes a new JSONL where each record gains a
`semantic_v3` key. It never edits `safe_prefill`, buckets, flags, or review
routing. No VLM/LLM calls, no network, no GPU.

```text
label_v2_suggestions_X.jsonl
  --(harvest semantic-v3)--> label_v2_suggestions_X_semantic_v3.jsonl
  --(harvest apply-label-v2)--> imported.jsonl (auto_metadata.semantic_v3 added)
  --(harvest export)--> manifest_*.jsonl (semantic_v3 field per record)
  --(harvest dataset-qa --require-semantic-v3)--> validated
```

Inputs used per prediction record, in priority order:

1. `safe_prefill.category` / `object_name` — copied verbatim (authority).
2. `safe_prefill` tokens: `tags`, `materials`, `mood`, `dominant_colors`,
   `short_description`.
3. `visual_facts.dominant_colors` and `visual_facts.shape_hints`
   (deterministic raster evidence — strongest grounding for colors/shapes).
4. `object_name` tokens (via the shared `label_taxonomy` tokenizer).
5. `vlm_descriptor.alternative_object_names` and `candidate_object_names`
   (aliases only — filtered against the accepted object).
6. `source_profile` (family hints and style tags).

Old prediction files remain valid: every consumer treats `semantic_v3` as
optional. Existing exported datasets are never mutated; producing a semantic
dataset requires explicitly re-running apply + export.

## 5. Source-family semantic extractor strategy

`src/spritelab/harvest/semantic_extractors.py` replaces "per-pack exact object
maps" with **shared vocabularies + family profiles**:

* Token vocabularies (pack-independent):
  * `COLOR_TOKENS` — direct colors plus derived ones
    (`golden -> gold, yellow`; `ruby -> red`; `emerald -> green`).
  * `MATERIAL_TOKENS` — `golden -> metal, gold`; `wooden -> wood`;
    gemstone names -> `crystal, mineral`; etc.
  * `EFFECT_TOKENS` — `electric/thunder/charged -> electric`; `fire`, `ice`,
    `poison`, `holy`, `shadow`, `bleeding`, `calming`, `glowing`, ...
  * `STATE_TOKENS` — `raw`, `cooked`, `polished`, `broken`, `sliced`, ...
* `BASE_OBJECT_FAMILIES` — one entry per semantic family (gems, containers,
  food, weapons, armor, tools, jewelry/keys, organic parts, currency,
  effect icons). Each family lists its base objects and contributes default
  `function` / `materials` / `shapes` / `parts` attributes.
* Per-base-object extras where a single family default is too coarse
  (`chestplate -> torso_shaped`, `arrow -> projectile`, ...).

Base-object extraction is conservative:

1. exact-name overrides for the few compounds that *are* the identity
   (`throwing_star`, `fish_skewer` policy: the prepared form wins);
2. otherwise scan object-name tokens right-to-left, first known base object
   wins (`golden_chestplate -> chestplate`, `ruby_gem -> gem`);
3. effect-icon names whose identity *is* the effect (`shadow`, `buff`,
   `holy`) become their own base object with `function = status_effect`;
4. fallback: full object name, with a `base_object_fallback_full_name`
   warning when the name is compound.

Adding a new source pack should mean: check its tokens against the shared
vocabularies, add missing *generic* tokens (a new color, a new base object),
and only then consider a small family profile. Never add
`filename -> exact_object` maps here.

## 6. Caption generation strategy

`build_captions(record, *, max_captions=8)` in `semantic_v3.py` generates
several caption styles from the semantic record only — every word must be
traceable to an extracted attribute or the style constants:

1. minimal: `ruby gem`
2. decomposed: `red polished gem made of crystal`
3. style-aware: `32x32 pixel art fantasy RPG red gem icon`
4. prompt-like: `centered 32x32 pixel art red crystal gem, black outline,
   transparent background` (the "black outline" clause is only emitted when
   black is actually among the sprite's dominant colors)
5. attribute dropout: `gem`, `red gem`, `armor icon`, ... (teaches the model
   that partial descriptions are valid)

Grounding rules:

* no invented details: adjectives come only from extracted
  colors/effects/state; "made of X" only from extracted materials;
* no lore, no non-visual flavor text;
* clean prose, not tag dumps; deduplicated; length-capped.

`prompt_phrases` are the 2–3 most training-prompt-shaped variants;
`negative_tags` default to `photorealistic, large_scene, text, watermark`.

## 7. Dataset export changes

`spritelab.dataset_maker.exporter._manifest_records` now includes a
`semantic_v3` object in each manifest record **when present** in the sprite's
`auto_metadata`. Everything else (npz layout, vocab, config, rejected.jsonl,
label_v2 audit block) is unchanged. Datasets without semantic metadata export
exactly as before — the field is simply absent.

`apply-label-v2` copies `prediction["semantic_v3"]` into
`auto_metadata["semantic_v3"]` alongside the existing `label_v2_*` audit keys.
`safe_prefill` remains the only source of applied labels.

## 8. Dataset QA changes

`spritelab.dataset_maker.qa.qa_dataset` gains `require_semantic_v3` (CLI:
`--require-semantic-v3`) and a `semantic_v3_checks` section.

Errors (for records that have `semantic_v3`, or all records in require mode):

* missing `semantic_v3` (require mode only);
* missing/empty `schema_version`, `base_object`, `open_name`;
* missing `attributes` object;
* empty `captions`, non-string captions, captions above the length cap;
* `semantic_v3.category` != manifest `category`;
* `semantic_v3.object_name` != manifest `object_name`;
* forbidden caption content (`photorealistic`, `watermark`, `text overlay`).

Warnings (aggregate, non-blocking):

* records with fewer than 3 captions;
* missing/empty `negative_tags`;
* high share of compound object names whose `base_object` fell back to the
  full name;
* records with no extracted attributes at all;
* gem/potion-family records with no color information.

Old datasets without semantic metadata pass untouched unless
`--require-semantic-v3` is given.

## 9. Evaluation strategy

* **Seen objects** — existing golden-label evaluation (`prefill-eval-v2`)
  stays authoritative for `category`/`object_name`. Semantic-v3 adds the
  `semantic-v3-report` command: base-object coverage, attribute coverage by
  family, caption statistics, warning counts. Regressions show up as coverage
  drops, not accuracy drops.
* **Unseen compositions** (`square_gem`, `mossy_key`) — evaluate the
  *decomposition round-trip*: parse the concept name with the same extractor
  vocabularies and verify the produced attributes are expressible in the
  schema and covered by dataset attributes (each factor seen somewhere, even
  if the combination is not). This is a dataset-coverage metric, not a model
  metric.
* **Creative concepts** (`charged_sinew`, `calming_spores`) —
  `creative_concepts.parse_creative_concept` maps modifier+substance names to
  grounded visual attributes. Evaluation: the parser must decompose the
  concept list into non-empty, schema-valid attributes. Future training-time
  evaluation will compare generated sprites against these attribute targets.
* **Reference-image conditioning** (future) — out of scope here; the schema
  is already the right target representation: a reference-image encoder
  should emit `SemanticAttributes` + captions, which then feed the same
  conditioning path as text prompts. No dataset changes are required now
  because captions/attributes are image-groundable by construction.

## 10. Implementation phases and test plan

Phase 1 (this pass — done):

* schema + converter (`semantic_v3.py`), extractor vocabularies
  (`semantic_extractors.py`), creative-concept parser (`creative_concepts.py`);
* CLI: `harvest semantic-v3`, `harvest semantic-v3-report`;
* apply/export/QA integration behind presence checks;
* tests: `test_semantic_v3.py`, `test_semantic_v3_cli.py`,
  `test_semantic_export.py`, `test_semantic_dataset_qa.py`;
* 496 end-to-end validation: semantic-v3 -> apply -> export -> QA with
  `--require-semantic-v3`, expecting 476 records / 0 errors.

Phase 2 (future):

* extend vocabularies as new source families arrive (data-driven, no
  per-pack object maps);
* optional rule-based caption randomization metadata for training-time
  augmentation;
* golden semantic labels for a small sample to measure attribute precision.

Phase 3 (future):

* small-LLM enrichment as a *copywriter only*: suggests aliases/caption
  smoothing, a deterministic validator rejects anything that adds unseen
  objects/colors/materials, never touches `category`/`object_name`/
  `base_object`. Offline/batched; never required.

Phase 4 (future): prompt parser for generation (creative concepts ->
`SemanticAttributes`), reference-image encoder to the same representation.

Test plan per phase: unit tests on extraction and captions (grounding,
no-invention), CLI round-trip tests on tmp runs, export/QA integration tests,
and the 496 pack as the frozen regression reference.

## 11. Non-goals

* No model training, image generation, or reference-image conditioning code.
* No giant closed taxonomy; vocabularies stay small and generic.
* No fresh VLM/LLM calls anywhere in this layer; no GPU; no internet.
* No mutation of existing exported datasets without an explicit re-export.
* No per-pack exact object maps disguised as semantic extractors.
* No changes to label-v2 fusion, buckets, or review routing.
* `object_name` accuracy work stays in label-v2; semantic-v3 never overrides
  it.
