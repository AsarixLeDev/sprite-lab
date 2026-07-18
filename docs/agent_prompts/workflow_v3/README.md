# Workflow v3 parallel agent prompts

These standalone prompts describe work that can proceed while the production
dataset is being acquired, reviewed, conditioned, audited, and frozen. They are
intentionally dataset-independent. Their evidence is provisional and must not
be represented as Phase F, H, I, or J evidence.

## Current boundary

- Phase A has a local certified commit. Recompute `git rev-parse HEAD`; never
  trust a copied hash without checking it.
- Phase B has independently issued Harvest capability evidence. Passive reload
  must still pass before any acquisition.
- Phases C through G own production data, labels, the conditioned candidate,
  independent dataset audits, the freeze, and the exact campaign.
- Phase H smoke evidence and the Phase-I training audit must be reproduced
  later against the exact final freeze and campaign.
- Phase J activation and **Start training** remain forbidden until every prior
  phase passes. These prompts never click Start or launch a campaign.

## Common safety and coordination rules

Give each numbered prompt to a separate agent. Every agent must:

1. Read `AGENTS.md`, `TRAINING_READINESS_REPORT.md`, and the current
   `STOP_HANDOFF_REPORT_2026-07-18.md` before acting.
2. Run `git status --short`, `git branch --show-current`, and record hashes of
   every file it may edit. Existing changes and untracked artifacts belong to
   the user or another agent.
3. Announce an exclusive file set before editing. Never edit a file owned by
   an active Harvest, dataset, conditioned-publication, smoke, audit, or
   training-start lane.
4. Stay read-only while an exact source/runtime certification is in progress.
   A tracked edit can stale Phase A/B evidence. If the coordinator says the
   tree is frozen, report findings and proposed patches instead of editing.
5. Use only synthetic fixtures inside unique repository-local temp roots. Do
   not acquire, import, label, freeze, activate, promote, resume, or train.
6. Make no provider/network/credential/billable call. Passive UI/status paths
   must not initialize Torch/CUDA, spawn subprocesses, or mutate state.
7. Never issue or claim a production PASS, audit, certificate, freeze,
   campaign, checkpoint, memorization clearance, or activation artifact.
8. Optimize tests before running a broad or historically slow lane. Inventory
   exact nodes, keep shards below 10 minutes, use unique `TEMP`, `TMP`, and
   `--basetemp`, cap native threads at one, and record every shard immediately.
9. Preserve live drift, tamper, confinement, cancellation, and reload tests.
   Cache only inputs that a scenario explicitly declares immutable.
10. Return exact commands, timings, counts, skip reasons, hashes, changed files,
    residues, counterexamples, and an explicit statement that no production
    action occurred.

## Prompt set and collision boundaries

| Prompt | Primary ownership | Must not own |
|---|---|---|
| `01_training_infrastructure_synthetic.md` | Dataset-independent trainer, checkpoint, resume, device, plan, and readiness contracts | Dataset-v5, Harvest, production smoke/audit artifacts, activation, provider or backend launch |
| `02_validation_evaluation_readiness.md` | Synthetic evaluation, checkpoint discovery, metrics, memorization, audit applicability, public projections | Production dataset validation/audit evidence, model generation, promotion, activation, Start/Resume |

When both lanes need the same file, neither edits it until the coordinator
assigns one exclusive owner. Prefer a new narrowly named test module over a
shared collision-prone test file.

## Dispatch and handoff

The two lanes may run concurrently during Phases C and D. First run their
read-only inventories and threat models; then coordinate exclusive fixes. After
both finish, commission a separate read-only review. Do not merge their results
into a production readiness verdict. Any final dataset, freeze, campaign,
checkpoint, code, runtime, driver, or policy change requires the applicable
Phase H/I evidence to be rerun.
