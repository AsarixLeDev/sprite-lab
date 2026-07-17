# Finish Playground terminal CAS and reconstruction

Prerequisite: the Playground operation-control lane is frozen. Follow the
common safety preamble in `README.md`.

Exclusive production file: `product_features/evaluation/playground.py`.
Exclusive new test file: `tests/test_playground_terminal_cas.py`. Do not edit
`local_generator.py`, `playground_worker.py`, EventRepository core, or smoke
files.

Require `explicit_action is True` and exact booleans for cost/adapter gates.
Install operation control before any adapter/code identity scan and clear it in
an outer `finally`. Serialize lifecycle mutations under the exact run lock and
compare a prior state/event identity.

Make events authoritative and atomic state a verified cache. A crash between
report, image event, terminal event, and state must be recoverable or fail
closed—never appear COMPLETE by inference. Reconstruction must unconditionally
match event-derived and state-derived results, require one legal terminal
event, reject duplicate/out-of-order/post-terminal events, and bind request,
checkpoint, adapter, runtime, report, artifact inventory, command, and terminal
identities. Recheck cancellation/deadline after lock acquisition and at the
last visibility boundary.

Inject faults at every publication boundary and race two service instances
across COMPLETE/CANCELLED/TIMED_OUT. Preserve passive no-Torch status behavior.

