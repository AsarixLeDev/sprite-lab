# Implement exact smoke execution-state CAS

Follow the common safety preamble in `README.md`.

Exclusive production file: `training/smoke_runner.py`. Exclusive new test file:
`tests/test_smoke_execution_state_cas.py`. Do not edit smoke bundle/worker,
Playground, conditioned-v5, confinement, or existing large tests.

Define the exact state key set and exact primitive types. Reject extras,
missing fields, bool-as-int values, malformed timestamps, and status-dependent
field inconsistencies. Add a canonical state identity and require both the
expected transition sequence and exact prior identity under the held lock.
Recompute before atomic publication, reread the postimage, and reject
same-sequence/different-body ABA. Terminal states are immutable; idempotence is
allowed only for an exactly compatible terminal value. Bind COMPLETE to the
exact receipt and exit code, CANCELLED to 130, and TIMED_OUT to 124.

Cover unknown/missing keys, ABA, stale rollback, restart reconstruction, and
COMPLETE-vs-CANCELLED/TIMED_OUT races. Report the new state field to the worker
protocol owner; runner-only work is incomplete until the worker authenticates
it.

