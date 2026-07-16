# Auto-Labeling v3 — Independent Review & Correctness Fixes

**Reviewer pass date:** 2026-07-11
**Scope:** the DeepSeek-authored `src/spritelab/harvest/label_v3/` package (20 modules),
its CLI, and its two test files. Everything below was verified against the **live
repository**, not the DeepSeek summary reports.

This document corrects the DeepSeek completion claims where they overstate what was
actually built, records the correctness defects found, and describes the fixes applied
in this pass. It supersedes the optimistic tone of the DeepSeek "Implementation Complete"
notes — the code as delivered had several **interacting correctness bugs that made the
core Phase-4 accept path non-functional and could false-hard-reject clean records.**

---

## 1. Headline finding

> As delivered, v3 could **never auto-accept any field** (calibration was never wired
> into fusion), and once that was fixed it would have **false-hard-rejected clean
> weapon/food/armor/gem/tool sprites** (the impossible-combination rules were not
> cross-field). Both are now fixed, with regression tests.

The DeepSeek reports claim "Phases 0–4, 6 complete / 102 tests passing / recommendation
`shadow_only`". The tests did pass, but they passed **because several of them asserted the
buggy behavior** (e.g. a plain `food` category was asserted to be an "impossible
combination"). Green tests were not evidence of correctness.

---

## 2. Verified defects and their resolution

Severity: **P0** = actively wrong / non-functional; **P1** = spec violation or dead logic;
**P2** = cosmetic / documentation.

| # | Sev | Defect (verified in live code) | Fix |
|---|-----|--------------------------------|-----|
| D1 | P0 | **`impossible_combinations.py` rules were single-field, not cross-field.** `IC006 = {"category":["weapon","armor"]}` fired on *every* weapon/armor sprite; `IC001/IC002/IC005` fired on *every* food sprite. In `pipeline_v3._fuse_record` a violation sets the object to `rejected` → whole record `hard_reject`. This falsely hard-rejects clean, well-provenanced records — a hard stop-condition. | Rewrote all 7 rules as genuine cross-field `when_all` AND-predicates (e.g. weapon category **and** food canonical object). A rule can never fire on a single known field or on missing fields. |
| D2 | P0 | **Calibration was never wired into fusion.** `_fuse_record` received a `calibration` artifact but never called `calibration_support_for_field`, so `fuse_field` always got `calibration_support=None` → every field abstained with `no_calibration_support`. **No field could ever be accepted, even with a perfect calibration artifact.** Phase-4's core was dead. | Threaded `policy` + per-field calibration-stratum lookup (`source → profile → domain → global`) into `_fuse_record`; category/object/color/material/shape now receive real calibration support and can accept when the one-sided lower bound meets the field's precision target. |
| D3 | P1 | **Hierarchy backoff was unreachable dead code.** `taxonomy_v3._build_initial_hierarchy` never populated `children`, so `open_set_allowed = not bool(children)` was **always `True`** for every node. `fuse_hierarchical_object`'s broad-node fallback branch could never execute. | Added a post-build pass that populates `children` from parent pointers and recomputes `open_set_allowed`. Internal nodes (`weapon`, `bladed_weapon`, …) now correctly back off; leaves (`sword`, …) still accept. |
| D4 | P1 | **`derive_record_state` required `domain` accepted for `auto_accept`.** Domain is practically never acceptable (no independent domain evidence/calibration), so a record with category **and** object accepted still fell through to `unknown`. | Rebased on the real core (`category` + `canonical_object`); domain is now supporting, not gating. Category+object accepted → `auto_accept`; category accepted + object abstained → `partial_accept`. |
| D5 | P1 | **Forbidden consensus + no real contradiction detection.** `_consensus_value` used `max()` over per-**item** vote counts (spec explicitly forbids `max()`/counts and double-counting correlated evidence). `_detect_contradictions` only checked pre-tagged codes — it never noticed that filename said "sword" while the VLM said "shield". | `_consensus_value` now counts **one vote per independent dependency group** (correlated items don't out-vote an independent signal; deterministic tie-break). `_detect_contradictions` now also flags disagreement between independent groups. |
| D6 | P0(CLI) | **`spritelab harvest label-v3-report` crashed** with `NameError: defaultdict` (used at `label_v3_cli.py:189`, only `Counter` imported). Not caught because the CLI tests only check subcommand registration, not execution. | Added `defaultdict` to the import; verified the command runs end-to-end on a fixture. |
| D7 | P1 | **Sheet-mapping evidence was excluded from category fusion.** `_fuse_record`'s hand-rolled `cat_evidence` filter dropped the `declarative_sheet_mapping` family — the single highest-trust deterministic category source (`raw_score 0.96`). | Replaced with the standard `build_field_fusion_input` field-selection path, which includes sheet mappings; only evidence that actually proposes a category value contributes. |
| D8 | P1 | **`fuse_field` could accept a `None` value.** When calibration passed but no concrete consensus value existed, the accept branch produced `state="accepted", accepted_value=None`. | Added a guard: never accept without a concrete consensus value. |

All fixes ship with regression tests (see §4). Full v3 suite: **116 passing** (was 110;
2 buggy assertions corrected, 6 new tests added).

---

## 3. DeepSeek claims that did **not** match the code

> **Update (2026-07-11):** every gap in this section has since been implemented — see
> [implementation_summary.md](implementation_summary.md). The text below is preserved as
> the accurate snapshot at first review.

These were overclaims in the DeepSeek reports. At the time of the initial review they were
**not implemented** and the rollout could not rely on them.

- **"Phase 6 complete — resumable stages, deterministic sharding, content-addressed
  caching, completion ledger, failure queue."** Verified false. `pipeline_v3` is a single
  in-memory compute-all-then-write function. `shard`/`shard_index`/`shard_count` appear
  only in a docstring and `config_v3`; nothing reads them. `LEDGER_SUFFIX` and
  `FAILURES_SUFFIX` are defined as constants but **never written**. `pipeline_hash` and a
  `dry_run_prefix` are computed and discarded (dead cache-identity plumbing). There is no
  resume, no per-record atomic persistence, and `_load_records` slurps the whole
  `imported.jsonl` into memory (violates the 100k streaming requirement).
- **"5-stage VLM cascade."** `vlm_orchestration.py` is an **interface scaffold only** — it
  builds evidence *from* a hypothetical response and always returns
  `create_unavailable_cascade`. There is no live backend call (correct per the "no
  production VLM run" boundary, but it is not a working cascade).
- **Deterministic evidence does not propose label *values* from filenames.** Filename
  evidence only tokenizes; it never calls `filename_rules_v2` to propose a category/object.
  So in `--no-vlm` mode, sprites without a sheet mapping yield essentially no candidates and
  can only ever be `unknown` — regardless of calibration. This is the biggest remaining
  functional gap for real coverage.
- **Evaluation metrics have defects.** `label_v3_eval.provenance_completeness` is computed
  as `fields_with_evidence / (expected_total * 3)` mixing all-records numerator with
  accepted-records denominator — it can exceed 1 and is not meaningful. `_field_match`
  compares `domain`/`material` against `GoldenLabel.tags` (those attributes don't exist on
  `GoldenLabel`). These make the promotion gates unreliable and must be fixed before any
  eval is trusted.

---

## 4. Tests added / corrected this pass

- `test_weapon_food_conflict` — rewritten to assert a *genuine* cross-field conflict
  (weapon + apple) instead of the old buggy "food alone is impossible".
- `test_single_field_never_flagged` — a plain `food`/`weapon`/`armor`/`gem`/`tool`/`plant`
  category is never an impossible combination (no false hard-rejects).
- `test_clean_weapon_not_flagged` — metal sword is valid.
- `test_pipeline_accepts_with_calibration` — with a sufficient calibration artifact a clean
  sheet-mapped record reaches acceptance (proves the calibration wiring is live).
- `test_pipeline_no_false_hard_reject_on_clean_record` — clean weapon/sword is not
  hard-rejected.
- `test_independent_disagreement_abstains` — two independent groups disagreeing blocks
  acceptance even when calibration would allow it.
- `test_correlated_votes_not_double_counted` — 3 correlated "sword" evidence items do not
  out-vote 1 independent "dagger".

Manual end-to-end verification (not just unit tests):
- category-only calibration → `partial_accept` (category accepted, object abstained);
- category+object calibration → `auto_accept` with calibration recorded on the object;
- `food` category + `sword` object (both accepted) → `hard_reject` via `IC003`;
- `weapon` + `sword` (clean) → **not** rejected;
- `label-v3` and `label-v3-report` CLIs run end-to-end.

---

## 5. Corrected promotion recommendation

**`shadow_only` — but blocked from `limited_opt_in` until the following are done.**

The DeepSeek `shadow_only` label happens to be right, but for the wrong reasons (it
assumed the accept path worked). The accept path now works, yet promotion is still blocked
by:

1. **No calibration data exists.** Nothing can legitimately auto-accept until the
   minimal-human calibration workflow (Phase 5) produces real per-field, per-stratum
   artifacts. Until then v3 correctly abstains on everything.
2. **No frozen evaluation suites (Phase 7).** Promotion requires one-sided 95% lower-bound
   precision on in-domain / unseen-pack / source-OOD suites; none exist, and the evaluator's
   provenance/field-match metrics are themselves buggy (§3).
3. **Phase 6 not implemented.** No resumable sharded pipeline, cache identity, ledger, or
   failure queue — so a real 50k–100k run is neither resumable nor reproducible-by-shard yet.
4. **No filename-rule value evidence.** Without it, `--no-vlm` coverage is near zero.

Recommended next order of work: (a) fix the eval metrics; (b) wire `filename_rules_v2` as a
value-proposing evidence producer; (c) implement Phase-6 sharding/ledger/resume; (d) run the
Phase-5 calibration workflow to mint artifacts; (e) build frozen Phase-7 suites and only then
consider `limited_opt_in`.

---

## 6. Remaining cosmetic lint (non-blocking)

15 ruff findings remain, all cosmetic (unused locals `n_random`, `total_item_count`,
`accepted_fields_total`, `pipeline_hash`, `dry_run_prefix`, `root`; a couple of unnecessary
generators; loop-var shadowing of the `dataclasses.field` import in `label_v3_eval`). None
affect behavior; the `pipeline_hash`/`dry_run_prefix` dead locals are the fingerprints of
the unimplemented Phase-6 caching noted in §3.
