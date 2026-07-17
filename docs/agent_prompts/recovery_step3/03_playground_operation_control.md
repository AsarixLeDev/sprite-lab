# Finish Playground operation control

Follow the common safety preamble in `README.md`.

Exclusive production file:
`product_features/evaluation/local_generator.py`. Exclusive new test file:
`tests/test_playground_operation_control_step3.py`. Do not edit Playground
service/worker, smoke files, confinement, or conditioned-v5.

Propagate the durable cancellation/deadline callback through every source,
code, runtime, checkpoint, prompt, result, asset, and report scan. Check before
and after each potentially long read/hash loop, while waiting for locks/process
activation, after the worker result is read, immediately before every visible
publication, and immediately before return. Preserve exact cancellation and
timeout error types and close process/descriptor resources on every early exit.

Add deterministic interruption tests at early inventory, mid-hash, post-worker,
and final-publication boundaries. Run a unique-basetemp focused suite plus Ruff,
format-check, and diff-check.

