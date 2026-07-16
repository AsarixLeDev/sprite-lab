# Auto-Labeling v3 — Implementation Summary & Final Recommendation

**Date:** 2026-07-11
**Status of this document:** supersedes the "gaps" sections of
[review_and_fixes.md](review_and_fixes.md). It records the work done to take the
DeepSeek scaffold from *non-functional* to a *complete, deterministic, resumable
shadow-mode system* with an explicit, safe apply path.

This is the completion report requested by the prompt's "Required reports"
section. The Phase-0 audit resolution and architecture decision record already
exist ([audit_resolution_report.md](audit_resolution_report.md),
[architecture_decision_record.md](architecture_decision_record.md)); the initial
correctness review is in [review_and_fixes.md](review_and_fixes.md).

---

## 1. What changed since the initial review

The initial review fixed the correctness bugs that made the system silently wrong
(false hard-rejects, dead calibration path, dead hierarchy backoff, forbidden
consensus). This second pass **implemented the phases DeepSeek had only
scaffolded**:

| Phase | Before (DeepSeek) | Now |
|-------|-------------------|-----|
| **2 — deterministic evidence** | filenames tokenized only; no values proposed → `--no-vlm` produced ~0 candidates | `filename_rules_v2` wired as a value-proposing producer (`_build_filename_value_evidence`); exact-trust profiles cannot inherit unsafe trust from unmapped filenames; sheet-mapping evidence now reaches category fusion |
| **4 — fusion** | `max()`/count consensus; disagreement = blanket veto | **reliability-weighted** consensus (one representative per dependency group, weighted by score) and **reliability-aware** contradiction detection (a weak signal can't veto a strong one) |
| **6 — scale/resume** | single in-memory compute-all; `shard`/ledger/failures were docstring-only | real sharded, resumable runner: deterministic `sprite_id`-hash sharding, fsync'd per-record persistence, completion ledger, failure queue, deterministic shard merge, streaming reports |
| **7 — eval/promote** | `provenance_completeness` mathematically broken; domain/material scored against `tags`; `label-v3-promote` always returned `shadow_only` (`__wrapped__` bug) | metrics fixed; only golden-evaluable fields scored; promotion recommendation computed correctly; frozen-suite manifests + leakage checks added |
| **5/rollout** | no apply/export path | `label-v3-apply` writes a **new-output-only**, dry-run-default sidecar + migration report; never touches historical artifacts |

## 2. New / changed modules

- `harvest/label_v3/deterministic_evidence.py` — filename value evidence producer.
- `harvest/label_v3/fusion_v3.py` — reliability-weighted consensus + contradiction.
- `harvest/label_v3/pipeline_v3.py` — reusable `compute_record_decision`, pack/profile lineage.
- `harvest/label_v3/pipeline_stages_v3.py` *(new)* — sharding, resume, ledger, failure queue, merge, streaming reports.
- `harvest/label_v3/frozen_suites_v3.py` *(new)* — frozen-suite manifests + leakage control.
- `harvest/label_v3/apply_v3.py` *(new)* — safe apply/export + migration report.
- `harvest/label_v3/label_v3_eval.py` — fixed provenance/field-match metrics.
- `harvest/label_v3/label_v3_cli.py` — new commands: `label-v3-shard`, `label-v3-merge`, `label-v3-retry`, `label-v3-apply`; streaming `label-v3-report`; fixed `label-v3-promote`.

## 3. CLI surface (all under `spritelab harvest`)

```
label-v3          run the in-memory pipeline (dry-run by default)
label-v3-shard    run one deterministic, resumable shard   (--shard I/N, --no-resume)
label-v3-merge    deterministically merge shards -> canonical v3_records.jsonl
label-v3-retry    retry only the retryable failures for a shard
label-v3-report   streaming per-pack + global report (bounded memory)
label-v3-eval     evaluate against golden labels
label-v3-promote  compute promotion recommendation from eval JSONs
label-v3-apply    apply accepted fields to a NEW output (dry-run default) + migration report
calibrate-v3      build an empirical calibration artifact from corrections
```

## 4. Operational guarantees (all test-backed)

- **Deterministic**: `--no-vlm` runs are byte-identical on repeat.
- **Shard-invariant**: a 1-shard and an N-shard run merge to a byte-identical
  canonical file (`test_shard_count_does_not_change_merged_output`).
- **Resumable**: kill-and-resume reproduces the full-run output; a crash re-does
  at most the one in-flight record (fsync per record, ledger after record).
- **Config-safe**: an output root refuses to mix results from a different config.
- **Merge-safe**: conflicting duplicate ids are rejected; identical ones collapse.
- **Failure isolation**: retry processes only retryable, not-yet-completed items.
- **Streaming**: reports never materialize the full record set (100k-safe).
- **Apply-safe**: dry-run default, new filenames only, refuses to write into a
  directory holding `imported.jsonl` / `label_v2_suggestions.jsonl`; rollback =
  delete the new output.

## 5. Precision-first behavior (unchanged intent, now actually enforced)

- No field is accepted without a calibration lower bound ≥ its precision target.
- Category can be accepted while object abstains (`partial_accept`); object
  identity backs off to a broad hierarchy node when the exact child is unsafe.
- Impossible combinations hard-reject **only** genuinely conflicting *accepted*
  fields (verified: `food`+`sword` → reject; `weapon`+`sword` → keep).
- With no calibration data, the system abstains on everything — by design.

## 6. Tests

New/updated v3 tests: **142 passing** across
`test_label_v3_phase1/phase6/phase7/phase57.py` (plus 36 semantic-v3 tests
unchanged). Project-wide collection is clean; no change to any v2/shared module,
historical artifact, or training default.

## 7. Final promotion recommendation

**`shadow_only` — eligible to advance to `limited_opt_in` once real calibration
and frozen suites exist.**

The machinery for `limited_opt_in` is now all present and tested (calibration
building, calibrated acceptance, frozen-suite leakage checks, safe apply). What
remains is **data, not code**:

1. Run the Phase-5 GUI calibration workflow on a real harvest run to mint
   per-field calibration artifacts (the pipeline consumes them today).
2. Freeze Phase-7 suites (`in_domain` / `unseen_pack` / `source_ood`) with the
   leakage checker, and confirm the category + canonical-object one-sided 95%
   lower-bound precision meets target on each.
3. Only then apply to a **new** output on packs not used for threshold fitting.

No core gate is met by *measured* data yet (there is none), so large-batch use
remains **blocked**. The system is, however, ready to generate v3 sidecars in
shadow mode and to be calibrated.

### VLM note

The multi-stage VLM cascade remains an interface scaffold (no live backend), per
the "no production VLM run" boundary. All acceptance today comes from
deterministic evidence (filename rules + sheet mappings + visual facts) gated by
calibration. When an approved VLM backend is wired into `vlm_orchestration`, its
evidence flows through the same reliability-weighted, calibration-gated fusion —
no acceptance path bypasses calibration.
