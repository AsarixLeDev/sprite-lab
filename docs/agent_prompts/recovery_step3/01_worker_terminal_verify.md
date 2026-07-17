# Independently verify WorkerTerminal

This is read-only verification. Follow the common safety preamble in
`README.md`. Do not edit production code.

Review `training/smoke_worker.py`, the state contract in
`training/smoke_runner.py`, and the WorkerTerminal additions in
`tests/test_product_exploratory_smoke.py`.

Verify that cancellation or timeout during every guard scan maps exactly to
`CANCELLED/130` or `TIMED_OUT/124`; finalization cannot publish COMPLETE after a
terminal race; `_finish` rejects status/exit mismatches; operation exceptions
are not collapsed into FAILED; publication is exclusive and pathless; and all
process/descriptor containment is released safely.

Use a unique basetemp. Return exact file hashes, commands, results, and any
counterexample. Do not certify the subsystem.

