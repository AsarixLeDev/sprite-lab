# Recovery step 3 parallel dispatch

These prompts prepare recovery order step 3 after the conditioned Dataset-v5
focused implementation tests passed. A later independent audit still returned
FAIL for strict receipt-v2 validation and identity cross-binding, so recovery
step 2 remains open. Each prompt is standalone and must be given to a separate
agent only when its prerequisites are satisfied.

Every agent must first read `AGENTS.md`,
`AUTONOMOUS_OVERNIGHT_COMPLETION_PROMPT.md`, `AUTONOMOUS_RUN_REPORT.md`, and
`TO_DELETE.md` completely. Preserve the dirty worktree and retained evidence.
Do not normalize the index, clean test roots, acquire data, call providers,
change configuration, launch training, push, or author PASS/certificate
evidence. Use `apply_patch`, a unique `.pytest_tmp_step3_<lane>_<nonce>`, and
report every new residue.

## Current lane state

- `_WorkerTerminal` wiring is implemented; it needs independent verification
  and later protocol-schema integration. Its first independent review returned
  FAIL for a terminal race and pre-validation descriptor leak.
- Windows inherited-token handling now fails closed before native derivation;
  never add an unrestricted-token fallback.
- Exact early Playground project-source loading is implemented in
  `playground_worker.py`; it needs merged verification.
- Playground operation-control and smoke execution-state CAS lanes are active.
- Conditioned step 2 must close the strict independent-auditor receipt and
  identity-binding findings before production use or any PASS claim.
- The current isolated smoke command bootstrap exceeds the Windows command-line
  limit. The runtime-contract lane must replace that oversized delivery
  mechanism without weakening source identity.

## Collision and serialization rules

| File | Required order |
|---|---|
| `training/smoke_bundle.py` | source/bootstrap design -> runtime control -> strict artifact schemas |
| `training/smoke_runner.py` | state CAS -> worker protocol consumer integration |
| `training/smoke_worker.py` | WorkerTerminal -> state-identity/protocol strictness |
| `evaluation/playground_worker.py` | exact source finder -> held-bootstrap execution |
| `evaluation/local_generator.py` | operation control -> lease CAS/bootstrap |
| `evaluation/playground.py` | one exclusive owner for terminal CAS/reconstruction |
| `utils/write_confinement.py` | freeze current inherited-token change pending review |
| `tests/test_product_exploratory_smoke.py` | freeze; use new test modules |

Never allow two agents to edit the same file, even when their intended symbols
are different.

## Dispatch order

1. Finish active source-finder, Playground-control, and smoke-state-CAS lanes.
2. In parallel, run independent WorkerTerminal/Windows reviews and the exclusive
   Playground terminal-CAS lane.
3. Run the single smoke runtime/bootstrap/schema lane after `smoke_bundle.py` is
   source-stable.
4. Run Playground lease/bootstrap after both `local_generator.py` and
   `playground_worker.py` are source-stable.
5. Run worker protocol strictness after runner CAS is frozen.
6. Run the hostile matrix only after all production writers are quiescent.
7. Commission a fresh read-only runtime/Playground/smoke audit. Implementation
   evidence is not an independent PASS.
