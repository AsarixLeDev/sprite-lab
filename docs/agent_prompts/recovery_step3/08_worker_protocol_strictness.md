# Strictly bind smoke worker protocols

Prerequisites: WorkerTerminal and runner state-CAS interfaces are frozen.
Follow the common safety preamble in `README.md`.

Exclusive production file: `training/smoke_worker.py`. Exclusive new test file:
`tests/test_smoke_worker_protocol_strictness.py`. Do not edit runner, bundle, or
the existing exploratory-smoke test module.

Enforce exact heartbeat, outcome, and execution-state key sets/types. The
worker must authenticate the runner's current state identity and transition
sequence. Bind outcomes to the current immutable heartbeat, plan, launch,
worker process, receipt, and terminal state; shape-only 64-hex checks are not
enough. Enforce timestamp ordering and exact status/exit mappings. Reject
unknown/missing fields, bool-as-int, forged/stale heartbeat hashes, cross-launch
outcomes, and terminal mismatches.

Preserve all WorkerTerminal cancellation/deadline behavior. Check operation
control at the final safe visibility boundary, keep publication exclusive and
byte-identically idempotent only, close descriptors/processes, and expose no
private path or raw exception text.

