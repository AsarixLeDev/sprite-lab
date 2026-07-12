# Memorization integration audit v1

## Result

**FAIL — the combined detector/review/promotion boundary is not certified.** The isolated detector-v2 rules and the isolated fail-closed promotion outcomes pass their focused tests, but four boundary contradictions prevent promotion authorization. No production file, test, historical artifact, checkpoint, or review log was modified.

## Critical findings

1. `suite.evaluate_gates` makes every review-required candidate fail the machine gate. `decide_promotion` makes every machine-gate failure an irreversible hard block. A conclusive, identity-bound v2 clearing review therefore cannot make a real review-required detector result eligible.
2. The detector emits `exact_rgba_nontrivial`; promotion recognizes comparable hard evidence as `exact_decoded_rgba`. Detector output reaches the decision layer as blocked/not-comparable rather than as the intended comparable hard block. Low-evidence exact-RGBA names also disagree (`exact_rgba_low_evidence_collision` versus the decision layer's `blank_collision`/`near_blank_collision`).
3. Detector-policy identity is incomplete. The detector and suite emit policy SHA-256 `8d6288ab...`, but promotion neither validates that SHA nor restricts the policy version to `memorization_detector_v2`. A caller-matched `memorization_detector_v999` probe was eligible, and changing the declared policy SHA to zeroes remained eligible.
4. The v2 review field named `candidate_evidence_sha256` is computed from one pair object, not the candidate-evidence document. A top-level candidate-file change remained eligible with the review hash unchanged.

The decision layer also trusts `promotion.pass` without recomputing or cross-checking its memorization counts. This is why the synthetic bundle can be structurally accepted with `pass=true` and `review_required_count=1`, a state the production suite would not emit.

## Detector-v2 verification

Focused CPU tests reconfirmed all requested cases: blank/blank is low evidence; blank/2×2 is not near-pixel evidence; nontrivial exact RGBA is machine-hard; exact-alpha and translation relations require review; generic sparse collisions are warnings; unresolved review evidence requires human action; and unsupported versions fail closed in the detector and suite gate.

## Promotion-decision verification

Focused tests reconfirmed that `same_sprite_or_memorized` blocks; uncertain and missing reviews remain pending when other inputs are comparable; malformed, stale, legacy-v1, or identity-mismatched reviews block as not comparable; human review cannot clear a machine failure; and an internally valid, machine-passing, conclusively cleared synthetic input returns eligible deterministically.

That synthetic `eligible` result is only a compatibility probe. It does not authorize a checkpoint, production evidence, or promotion action. The synthetic checkpoint is plain text, no real checkpoint was presented as eligible, and the audit performed no promotion operation.

## Historical result

The production decision command re-evaluated the pinned historical inputs and returned `blocked` / `not_comparable`. The machine gate still fails, only 8 of 54 frozen-suite samples exist, and all nine historical v1 review rows are identity-unbound and have no promotion authority. The decision SHA-256 remains `a05efed301338b84f3d4be923718651c3b76af3891a0aa8cc20fd18605e61fdc`.

## Validation and safety

- Focused tests: **40 passed**, 0 failed, 0 skipped, no warnings.
- Ruff: all five audited production files passed.
- Format check: all five audited production files already formatted.
- `git diff --check`: passed.
- Frozen/historical pins: all 12 verified unchanged.
- Training runs: 0; generation runs: 0; provider calls: 0; CUDA initialization: false; checkpoint promotions: 0.

## Required remediation before certification

Unify evidence-class vocabulary; make reviewed machine-gate evidence resolvable without weakening independent failures; validate the exact detector policy version and SHA; bind reviews to the whole canonical candidate-evidence document; require direct training-image binding; and cross-check machine report semantics. Then repeat this audit independently on a fresh evidence directory.

**Checkpoint promotion remains blocked and is not authorized.**
