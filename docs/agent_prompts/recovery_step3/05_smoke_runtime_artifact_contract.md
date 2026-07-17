# Finish the neutral smoke runtime, bootstrap, and artifact contract

Prerequisite: all source-finder writers are frozen. Follow the common safety
preamble in `README.md`.

Exclusive production files: `training/smoke_bundle.py` and
`utils/runtime_closure.py`. Exclusive new tests:
`tests/test_runtime_closure_operation_control_step3.py`,
`tests/test_smoke_artifact_schema_strictness.py`, and a uniquely named bootstrap
test. Do not edit runner/worker, Playground, conditioned-v5, or existing large
test modules.

Replace the oversized Windows `-c` bootstrap with an identity-bound delivery
that stays within platform command limits and executes only exact verified
bytes. Install the project-source policy before any project import. Execute the
real bootstrap in tests; a wrong digest, oversized input, descriptor
substitution, or preloaded project module must execute no marker.

Operation checks must occur before/during/after all source, runtime, extension,
resource, and hash walks, before bound module execution, and at the final
closure rehash/return. Enforce exact nested key sets and primitive types for
plans, interpreter/orchestration/runtime rows, device receipts, evidence, and
Playground registration. Reject noncanonical ordering, duplicates, bool-as-int,
self-rehashed unknown fields, empty verification, and incomplete semantic
cross-bindings.

Keep `runtime_closure.py` product-neutral, passive, pathless, and stable for
conditioned code. Preserve the honest documented native/DSO/resource residuals;
do not claim FD pinning that is not implemented. Use bounded synthetic fixtures
until all source writers are quiescent, then run one real closure scan.

