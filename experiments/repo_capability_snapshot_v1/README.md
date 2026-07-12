# Sprite Lab capability snapshot

This directory is the conservative operator entry point for the current repository milestone. It describes what can be inspected safely; it is not an authorization to review real records, call a provider, generate a benchmark, train a model, freeze production Dataset-v5, or promote a checkpoint.

## What currently works

- The frozen unlabeled r2 pool verifies read-only: 3,233 retained records and 1,288 geometry families.
- The immutable legacy Dataset-v5 preview and the current audited policy-v2 core-plus-weighted preview pass their read-only verifiers. They remain previews, not production Dataset-v5.
- The Dataset-v5 named-view implementation builds, verifies, and freezes synthetic fixtures deterministically for all seven named views. No production view was built or frozen.
- Labeling-v4 two-pass remediation and its 100-record CPU-only synthetic rehearsal pass locally. The rehearsal produced 80 queue members, 20 exclusions, zero provider calls, and byte-identical repeated freezes.
- Headless model construction and three-seed campaign mechanisms are implemented and locally tested. Memorization detector and decision components also have isolated passing tests.

## Implemented but not certified

- Labeling-v4 still needs a fresh independent audit of the final remediation before real review or a real queue freeze.
- Dataset-v5 named-view tooling still needs a fresh, commit-bound independent builder audit and resolution of the open production policies.
- The training infrastructure audit fails closed because final Dataset-v5, evaluation, and related resume/campaign identities are not bound. Safe resume, campaign fairness, and execution are not certified; training is blocked.
- The combined memorization integration audit fails even though isolated detector and decision cases pass. Detector/decision exact-match classes disagree, the detector-policy SHA is not enforced, candidate evidence binds only the pair, the suite's `review_required` machine gate cannot be cleared by a valid review, and inconsistent synthetic evidence was accepted. Promotion is blocked.

## What remains blocked

The real quality GUI, real inference queue freeze, provider-backed semantic inference, production Dataset-v5 freeze, 54-output benchmark generation, real campaign execution, and checkpoint promotion remain blocked. See [blocked_commands.md](blocked_commands.md) for the gate behind each restriction.

## Next human action

From the repository root, run the safe read-only snapshot without tests using the tested form that works even when direct PowerShell-script execution is blocked:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\repo_capability_snapshot_v1\safe_commands.ps1
```

If local execution policy already permits scripts, direct invocation is equivalent. Then commission a fresh independent review of the final Labeling-v4 remediation. The next implementation and authorization steps are ordered in [next_milestones.md](next_milestones.md). Do not start real review while that audit is outstanding.
