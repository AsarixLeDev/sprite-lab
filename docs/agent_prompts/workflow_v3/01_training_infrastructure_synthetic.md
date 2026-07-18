# Parallel lane: dataset-independent training infrastructure

You are working in the Sprite Lab repository root while production
dataset acquisition, review, conditioning, publication, freeze, campaign,
smoke, audit, and activation are owned by other agents. Improve reusable
training infrastructure without advancing or bypassing any production gate.

## Objective

Audit, test, and only when a focused synthetic regression proves a defect,
harden dataset-independent training behavior:

- trainer math and finite-loss behavior;
- exact repeat/resume state where the contract requires bit identity;
- checkpoint schema, corruption refusal, and safe loading;
- campaign-plan validation without publication or launch;
- device-policy and deterministic-environment reporting;
- readiness, Start, and Resume refusal while prerequisites are absent or stale;
- public privacy, pathlessness, concurrency, and filesystem confinement.

Your result is implementation evidence only. It is not Phase H smoke evidence,
a Phase-I audit, or permission to train.

## Mandatory startup and ownership

1. Read `AGENTS.md` completely. Read the relevant portions of
   `TRAINING_READINESS_REPORT.md`, `STOP_HANDOFF_REPORT_2026-07-18.md`,
   `docs/product_training.md`, `docs/training_baseline_architecture.md`, and
   `docs/training_speed_notes.md`.
2. Run `git status --short`, `git branch --show-current`, and
   `git rev-parse HEAD`. Preserve every existing change and artifact.
3. Record the SHA-256 of `spritelab.yaml` and the existence of configured
   activation, campaign, audit, and full-run destinations. Never create a
   missing destination merely to inspect it.
4. Trace focused tests before source. Likely starting points include
   `src/spritelab/training/`, `src/spritelab/product_features/training/`, and
   matching `tests/test_training_*.py` / `tests/test_product_training_*.py`.
5. Announce a small exclusive file set. Recheck each file hash before editing.
   If it is dirty, changes during your lane, or belongs to another agent, stop
   and report the collision.
6. If Phase A/B or another source/runtime audit is active, remain read-only.

## Absolute prohibitions

- Do not read, copy, hash, relabel, or modify production/private dataset trees.
  Use only tiny synthetic fixtures in a unique
  `.pytest_tmp_workflow_v3_training_<nonce>` root.
- Do not download assets, contact a provider, use credentials, spend money, or
  make any network call.
- Do not run `spritelab train`, `spritelab v3 train`, `execute_campaign`, a
  backend adapter, remote/SSH/cloud compute, `POST /training/api/start`, or
  Resume. Install fakes that fail if a launch boundary is reached.
- Do not change `spritelab.yaml`, set either production authorization flag,
  publish a freeze/campaign, write activation markers, or enable Start.
- Do not create/edit production smoke bundles, audit reports, hashes, receipts,
  action records, certificates, checkpoints, run roots, or PASS evidence.
- Do not initialize Torch/CUDA, spawn subprocesses, access credentials, or
  mutate state from passive status/readiness paths.
- Do not stage, commit, push, clean, reset, delete, or rewrite another lane's
  files. Integration belongs to the coordinator.

A tiny synthetic forward/backward or one/two optimizer-step check is allowed
only inside the current test process, with no campaign launcher, persistent run
root, or production-evidence claim.

## Work program

### 1. Inventory the contracts

Map current focused coverage for:

- optimizer, scheduler, scaler when applicable, EMA, sampler/RNG, and step
  counters across exact repeat/resume;
- validation/evaluation mode, no-gradient behavior, finite aggregation, and
  train/validation separation;
- checkpoint exact keysets, hashes, atomicity, truncation/corruption refusal,
  foreign identity refusal, and safe weights-only loading;
- campaign/config semantic agreement, schedule projection, seed uniqueness,
  output confinement, and refusal before side effects;
- CPU/CUDA policy reporting and deterministic-environment qualification;
- Start/Resume blockers for missing/stale freeze, campaign, applicable audit,
  authorization, code/config binding, checkpoint, and continuation receipt.

Start with focused files such as:

- `tests/test_training_correctness_v2.py`
- `tests/test_training_campaign.py`
- `tests/test_training_resume_binding.py`
- `tests/test_training_migration_resume.py`
- `tests/test_training_local_device_readiness.py`
- `tests/test_training_plan_path_confinement.py`
- `tests/test_training_web_start_controls.py`
- `tests/test_training_web_readiness_hardening.py`
- `tests/test_product_training_feature.py`

Select one bounded gap. Do not rewrite already complete areas.

### 2. Prove the defect or missing contract

Use synthetic inputs and explicit sentinels. For Start/Resume tests, inject a
backend/process factory that raises immediately if called. For deterministic
state, compare exact tensors/state dictionaries and canonical semantic state
when exact identity is required; do not replace an exact contract with a loose
tolerance.

Include relevant adversarial cases:

- missing/stale/mismatched/truncated/duplicate-key/wrong-type/non-finite input;
- traversal, absolute/UNC/device path, symlink/reparse, and hard-link aliases;
- configuration, code, checkpoint, or receipt drift between validation/use;
- repeated/concurrent requests and crash/restart boundaries;
- passive state with launch, subprocess, provider, Torch/CUDA-init, credential,
  and write sentinels;
- public errors/events containing no private path, credential assignment,
  bearer/basic token, URL credential, command line, or provider secret.

### 3. Fix narrowly

Only after the focused regression fails for the expected reason, apply the
smallest compatible fix. Preserve lazy imports, deterministic ordering, stable
schemas, SHA-256 identities, exact authorization booleans, path confinement,
atomic publication, and cross-platform behavior. Never weaken a gate to make a
fixture pass. If stricter production behavior is correct, fix the fixture.

If the defect belongs to an active Dataset-v5, Harvest, conditioned,
smoke/runtime, activation, or filesystem owner, deliver the reproducer and do
not edit that file.

### 4. Test efficiently

Do not begin with a broad suite. Inspect prior durations or run one
representative focused node with `--durations=20`; profile it if historically
slow. Remove only repeated immutable work, keeping all live security tests live.

For every worker set:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:VECLIB_MAXIMUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
python -m pytest <exact nodes/files> -q --durations=20 `
  --basetemp=.pytest_tmp_workflow_v3_training_<nonce> -p no:cacheprovider
```

Inventory exact nodes before broadening. Partition into deterministic,
disjoint, exhaustive shards; use unique process `TEMP`, `TMP`, and `--basetemp`;
keep every shard below 10 minutes; report each shard immediately. On timeout,
stop only the verified owned process, preserve completed evidence, and split
only the uncovered inventory. Never rerun the same monolith with a larger
timeout.

Run Ruff check/format on changed paths, targeted mypy for type-sensitive source
changes, `git diff --check`, and inspect the actual diff/status. Do not change
the shared Python environment without coordinator approval.

## Completion and handoff

Return:

1. contract inventory and threat model;
2. reproduced defects and counterexamples;
3. exact changed files and why;
4. commands, node/shard inventories, timings, counts, and reviewed skips;
5. timing before/after any optimization and why live checks remain live;
6. start/end hashes, concurrent drift, and all created residue;
7. residual risks and production prerequisites still required;
8. a verdict limited to `synthetic training infrastructure checks passed`,
   `checks failed`, or `incomplete`.

At handoff prove `spritelab.yaml` is byte-identical, the real project remains
blocked, Start remains unavailable, and no new production artifact exists.
State explicitly: **no data acquired, no provider called, no configuration
activated, no audit PASS authored, and no training process launched**.

Stop rather than improvise if real data/checkpoints, production publication,
activation, audit issuance, provider/network access, CUDA from a passive path,
another agent's file, a gate weakening, destructive cleanup, or a shard over 10
minutes would be required.

Prioritize agent spawning for parallel tasks for faster work
