# Finish Playground lease CAS and held-worker bootstrap

Prerequisites: `local_generator.py` operation control and
`playground_worker.py` exact source loading are frozen. Follow the common safety
preamble in `README.md`.

Exclusive production files: `product_features/evaluation/local_generator.py`
and `product_features/evaluation/playground_worker.py`. Exclusive new tests:
`tests/test_playground_lease_cas.py` and
`tests/test_playground_worker_bootstrap.py`. Do not edit Playground service,
smoke files, runtime closure, or confinement.

Give the lease an exact schema, canonical identity, transition sequence,
predecessor identity, and exact owner process identity rather than PID alone.
Protect immutable fields; allow only legal ACTIVE to terminal transitions;
reject stale heartbeat, PID reuse, rollback, and foreign-owner updates. Lock
waits must be bounded and operation-aware.

Make the `-I -B -S -c` held-worker bootstrap a named identity-bound constant.
Read only a held descriptor with a byte limit and exact digest, install the
source policy before project imports, reject preloaded modules, execute the
real bootstrap in a subprocess test, and preserve post-import source
revalidation. Check deadlines through confinement, checkpoint/prompt reads,
sampling finalization, report publication, and return. Preserve process-group/
Job termination and pathless stderr.

