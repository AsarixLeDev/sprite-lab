# Architecture Decision Record — Auto-Labeling v3

**Date**: 2026-07-10
**Status**: Implemented (Phases 0-4, 6 complete; Phases 5, 7 require GUI/infra integration)

## Decision: Versioned, additive v3 architecture alongside v2

**Context**: Sprite Lab needs a precision-first labeling system for 50,000-100,000+ sprites. The existing Labeling v2 system has measured precision gaps, relies on trusted-source heuristics, and offers limited hierarchy/open-set support.

**Decision**: Implement v3 as a versioned, parallel system rather than modifying v2 modules. All v3 code lives in `src/spritelab/harvest/label_v3/`.

**Rationale**:
- v2 behavior must not change (proven compatibility constraints)
- v2 modules are extensively tested and validated
- Separate versioning allows independent evolution
- Risk isolation: v3 bugs cannot affect v2 production pipelines

## Module Architecture

```
src/spritelab/harvest/label_v3/
  __init__.py                  — Public API re-exports
  evidence.py                  — EvidenceItem: versioned evidence contract (Phase 1)
  field_decisions.py           — FieldDecision, TagDecision, AcceptedTagSet (Phase 1)
  record_decisions.py          — RecordDecision, derive_record_state (Phase 1)
  reason_codes.py              — CONTRADICTION_CODES, REASON_CODES, severity/action registry (Phase 1)
  taxonomy_v3.py               — Hierarchical taxonomy with parent/child/synonym support (Phase 1)
  impossible_combinations.py   — Declarative impossible-combination validation rules (Phase 1)
  sha256_utils.py              — Content-addressing and hashing utilities (Phase 1)
  config_v3.py                 — V3LabelingPolicy, V3PipelineConfig (Phase 1)
  adapter.py                   — Legacy-shaped adapters for v2 consumers (Phase 1)
  deterministic_evidence.py    — 10 deterministic evidence producers (Phase 2)
  vlm_orchestration.py         — 5-stage VLM cascade with context tracking (Phase 3)
  fusion_v3.py                 — Per-field calibrated fusion with hierarchy backoff (Phase 4)
  calibration.py               — Versioned calibration artifacts with confidence bounds (Phase 4)
  pipeline_v3.py               — Pipeline orchestration with dry-run, stage isolation (Phase 6)
```

## Key Design Decisions

### 1. Evidence-first, field-level fusion

Evidence is collected independently per producer stage. No evidence is collapsed into a single suggestion before final fusion. Each evidence item carries:
- Schema version, stable evidence ID, sprite ID
- Evidence family and producer stage
- Target fields, proposed values, raw scores
- Calibration stratum, dependency group
- Exposure tracking (source hints, candidate hints)
- Stage/config hashes for cache identity

### 2. Field states are explicit and distinct

Nine distinguishable states: `accepted`, `abstained`, `quarantined`, `rejected`, `unknown`, `novel`, `ambiguous`, `unlabeled`, `not_applicable`. No collapsing to empty strings or `<unk>` tokens.

### 3. Hierarchy backoff instead of forced classification

When exact object identity is unsafe, the system falls back to the deepest safely-supported hierarchy node (e.g., `bladed_weapon` instead of `sword`). Novel objects are not forced into the nearest exact class.

### 4. Calibration-driven promotion

Auto-accept requires per-field calibration support with a one-sided 95% lower confidence bound meeting the configured precision target. Without calibration data, decisions default to abstained. Sparse strata inherit from broader strata or remain unpromoted.

### 5. Correlation-aware fusion

Dependency groups track correlated evidence sources. Filename+filename-hinted-VLM are recognized as correlated. Multiple propagated variants are counted as one independent source. Raw scores are never combined with naive averaging.

### 6. Staged VLM cascade

Five-stage VLM pipeline (blind descriptor → morphology → constrained classification → open-set verification → consistency verification) with strict context exposure tracking. Source hints only appear after the blind stage. Every response records exposed context.

### 7. Backward compatibility

All existing v2 schemas, tests, manifests, and training defaults remain unchanged. V3 writes versioned sidecar artifacts by default. A legacy adapter exposes accepted v3 fields to legacy-shaped consumers.

## Implementation Status

| Phase | Status | Key Deliverables |
|-------|--------|-----------------|
| 0 | Complete | Audit resolution report, compatibility baseline, existing tests verified |
| 1 | Complete | Evidence/decision/taxonomy schemas, reason codes, adapter, config |
| 2 | Complete | 10 deterministic evidence producers, coverage of all required facts |
| 3 | Complete | 5-stage VLM cascade architecture, context tracking, exposure control |
| 4 | Complete | Calibrated per-field fusion, hierarchy backoff, contradiction handling |
| 5 | Deferred | Needs integration with existing Gradio assisted GUI |
| 6 | Complete | Pipeline orchestration, dry-run, stage isolation, summary reports |
| 7 | Deferred | Needs golden dataset, evaluation suites, promotion policy |

## Test Coverage

- 79 tests in `tests/test_label_v3_phase1.py`
- Covers: evidence schema, field decisions, record decisions, reason codes, taxonomy, impossible combinations, hashing, adapter, calibration, fusion, deterministic evidence, VLM orchestration, pipeline
- All existing v2 tests pass unchanged (24 passed + 4 Windows cleanup errors)
- Legacy v2 compatibility verified: `LabelSuggestion`, `SafeFusedLabel`, `CATEGORY_VALUES`, source profiles, semantic_v3

## Non-Negotiable Boundaries (Verified)

1. No model training or fine-tuning
2. No production-scale VLM runs
3. No overwrite of production datasets or historical runs
4. No change to Labeling v2 semantics
5. No deletion of rejected/quarantined records
6. All v3 operations default to dry-run or sidecar output
7. No forced coverage targets at cost of precision
