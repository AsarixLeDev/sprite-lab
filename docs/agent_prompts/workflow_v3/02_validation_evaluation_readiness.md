# Parallel lane: synthetic validation and evaluation readiness

You are working in the Sprite Lab repository root while other agents
own production Harvest and dataset work. Improve and independently verify the
validation, evaluation, memorization, audit-applicability, and readiness-display
infrastructure without consuming or fabricating production evidence.

## Objective

Make the infrastructure ready to consume a future exact conditioned Dataset-v5
freeze and future checkpoints. Using synthetic fixtures only, verify that it:

- rejects missing, malformed, foreign, stale, or identity-mismatched inputs;
- reports `BLOCKED`, `NOT AUDITED`, `STALE`, or `FAIL` honestly;
- binds future decisions to exact dataset/view, freeze, campaign, checkpoint,
  benchmark, metric-definition, policy, code, auditor, and review identities;
- keeps passive status/discovery/rendering/dry-run paths free of generation,
  providers, Torch/CUDA, subprocesses, credentials, and mutation;
- keeps Playground outputs ineligible for production benchmarks, Dataset-v5,
  audit, promotion, resume, and campaign evidence;
- remains deterministic, bounded, pathless, portable, and fail closed.

Synthetic test success is not a dataset-validation PASS, Phase F, Phase H,
Phase I, promotion decision, or production readiness verdict.

## Mandatory startup and ownership

1. Read `AGENTS.md`, relevant sections of `TRAINING_READINESS_REPORT.md` and
   `STOP_HANDOFF_REPORT_2026-07-18.md`, `docs/product_evaluation.md`, and
   `docs/product_playground.md`.
2. Run `git status --short`, `git branch --show-current`, and
   `git rev-parse HEAD`. Preserve the dirty worktree and every artifact.
3. Record hashes for files you may edit. Announce an exclusive file set and
   recheck it before every edit and handoff. Stop on concurrent drift.
4. Stay read-only while Phase A/B or another exact source/runtime audit is
   active. Do not collide with dataset, conditioned-publication, training-start,
   smoke-runtime, or filesystem-confinement owners.
5. Prefer additive, uniquely named tests over a shared test file owned by
   another lane.

Primary areas may include `src/spritelab/evaluation/`,
`src/spritelab/product_features/evaluation/`, and their focused tests. Touch
training audit/readiness projection only for a demonstrated applicability or
public-projection defect and only with exclusive ownership. Never touch launch,
activation, production campaign execution, or smoke execution in this lane.

## Absolute prohibitions

- Do not download, harvest, import, label, review, condition, freeze, or modify
  production data. Do not write production `artifacts/`, `datasets/`,
  `harvest_runs/`, `runs/`, or `outputs/`.
- Do not issue a production audit/certificate/PASS, validation report, freeze,
  campaign, checkpoint, memorization clearance, promotion, or activation.
- Do not edit `spritelab.yaml`, set authorization flags, enable Start, launch or
  resume training/evaluation, or POST production actions.
- Do not call providers/network, spend money, load a real model, initialize
  CUDA, or generate with a real checkpoint. Use deterministic fake adapters.
- Do not let exploratory outputs become benchmark, promotion, audit, resume,
  Dataset-v5, or production evidence.
- Do not stage, commit, push, clean, reset, delete, or rewrite other work.
- Do not weaken exact identities/booleans, bless legacy or stale evidence, hide
  a real defect with a skip, or cache live drift/tamper/reload state.

## Work program

### 1. Build a contract and threat matrix

Trace every selected artifact from schema/version through writer, loader,
validator, identity construction, applicability decision, public projection,
and tests. Record:

- exact identities and authority it must bind;
- authoritative writer/readers;
- missing/stale/tamper behavior;
- passive versus explicit-action boundary;
- public fields versus private diagnostics;
- existing tests and uncovered risks.

Search narrowly for `audit`, `PASS`, `STALE`, `checkpoint`,
`dataset_identity`, `training_view_identity`, `benchmark`,
`metric_definition`, `review`, `promotion`, `dry_run`, `provider`, `torch`,
`subprocess`, `replace`, `move`, `unlink`, `resolve`, and `relative_to`.

### 2. Exercise a synthetic fail-closed matrix

Use tiny fixtures only in
`.pytest_tmp_workflow_v3_validation_<nonce>`. Verify relevant cases:

- missing/malformed reports never become PASS;
- a plausible PASS bound to old code, dataset/view, freeze, campaign,
  checkpoint, benchmark, policy, metric definition, audit subject, auditor, or
  review log is stale/inapplicable;
- duplicate keys, non-exact booleans, unknown schema, wrong types,
  traversal/absolute/UNC/device paths, Unicode/case collisions,
  symlink/reparse/hard-link aliases, partial/replaced files, and post-scan drift
  fail closed;
- incomplete/replayed/conflicting memorization reviews cannot clear evidence;
- checkpoint discovery excludes incomplete, foreign, stale-dataset,
  unsafe-resume, missing, unverified, or hash-mismatched checkpoints;
- metric comparisons reject definition mismatches before aggregation;
- public API/dashboard/status output exposes no credentials, private paths,
  raw exceptions, command lines, environment values, or provider diagnostics;
- exploratory outputs remain explicitly ineligible for frozen benchmarks,
  promotion, Dataset-v5, audits, resume, and production use;
- missing freeze/campaign/audit/authorization leaves production actions blocked.

Use call/mutation sentinels to prove passive operations make zero generator,
provider, subprocess, Torch/CUDA, credential, config, or artifact calls. Inspect
`sys.modules` before/after instead of importing optional heavy dependencies.

### 3. Test passive and dry-run behavior

Exercise only passive services/endpoints and documented validation or dry-run
entry points in a synthetic project. Confirm page open, status, discovery,
report rendering, preset loading, checkpoint listing, and plan validation do
not mutate inputs or create production artifacts.

Use an in-process test client. Do not start a persistent server unless the
coordinator assigns browser QA. Never point it at real production state or POST
Start/Resume/evaluation/promotion/provider actions.

### 4. Fix only demonstrated defects

Save the minimal failing regression, reconfirm ownership/hash, and apply the
smallest compatible fix. Preserve lazy imports, privacy, stable schemas,
deterministic ordering, identities, exact booleans, path confinement, and live
security tests. Report schema/user-visible changes to the coordinator.

If the fix belongs to another active owner, send the reproducer and stop. Never
silently redefine production evidence or change a fixture to manufacture PASS.

### 5. Test efficiently

Inventory exact nodes first. Use prior durations or benchmark one
representative slow test; profile only that representative. Remove repeated
immutable fixture/hashing/process work before broadening, while keeping live
drift/tamper/confinement/cancellation/reload tests live.

For every parallel worker:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:VECLIB_MAXIMUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
python -m pytest <exact nodes/files> -q --durations=20 `
  --basetemp=.pytest_tmp_workflow_v3_validation_<nonce> -p no:cacheprovider
```

Partition broader coverage deterministically with no overlap/gap, isolate slow
Playground/local-generator cases, use unique process `TEMP`/`TMP`/`--basetemp`,
and keep every shard below 10 minutes. Record results immediately. On timeout,
terminate only the verified owned process and resume only uncovered inventory.

Run Ruff check/format on changed paths, targeted mypy for type-sensitive source,
`git diff --check`, and inspect the diff/status. Do not change the environment
without approval.

## Completion and handoff

Return:

1. contract/threat matrix;
2. exact files changed and justifications;
3. defects, counterexamples, and delegated findings;
4. exact commands, node/shard inventories, timings, counts, and skip reasons;
5. optimization timing and proof live checks remain live;
6. start/end hashes, drift, and created residue;
7. residual risks and exact production prerequisites;
8. a verdict limited to `synthetic validation infrastructure checks passed`,
   `checks failed`, or `incomplete`.

State explicitly: **no data acquired, no provider/network/model/CUDA action,
no production artifact or PASS issued, no configuration activated, and no
training/evaluation execution launched**.

Stop instead of improvising if another agent owns the file, real data/model or
production state is required, a gate would be weakened, a real action/provider
would be called, a shard approaches 10 minutes, destructive cleanup is needed,
or production prerequisites appear during the lane. Do not consume newly
available production state without a newly scoped assignment.

Prioritize agent spawning for parallel tasks for faster work
