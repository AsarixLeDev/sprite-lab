# Sprite Lab V5 next-milestone report

## 1. Executive result

The milestone is locally complete on `codex/v5-next-milestone`: Labeling-v4 remediation and Dataset-v5 named-view tooling were implemented and committed, the requested read-only infrastructure audits and operator snapshot were created, and the full repository suite passes. No provider inference, GPU training, benchmark generation, production Dataset-v5 freeze, real review session, or checkpoint promotion was performed.

This work deliberately does **not** authorize production progression. The audits found fail-closed defects in training resume/campaign binding and incompatible memorization integration semantics. Those findings are preserved as blockers rather than hidden behind passing isolated tests.

## 2. Labeling-v4 result

All nine requested two-pass safety defects and their adversarial tests were addressed. The deterministic synthetic rehearsal produced 100 quality decisions, an 80-record source-bound semantic queue, and 20 excluded records; repeated outputs were byte-identical and provider calls were zero. Native-dimension diagnostics are nonblocking. Wave-1 selection and human-truth hashes remained unchanged.

Local implementation and rehearsal checks pass, but this session is not independent certification: **fresh independent Labeling-v4 audit still required**. No real quality GUI session, real inference queue, or provider-backed semantic preparation is authorized here.

## 3. Dataset-v5 builder result

The repository now has deterministic named-view contract validation, build, verification, freeze, and frozen-view verification for all seven required views. Synthetic builds cover the four exact supervision classes, source/policy replay, relation closure, identity binding, tamper rejection, and byte-identical rebuilds. The approved contract and unlabeled-pool-r2 pins match exactly. No production view or production freeze was created.

A fresh independent builder audit, final policy decisions, and real production inputs are still required before a production Dataset-v5 freeze.

## 4. Training audit result

Headless architecture and architecture identity pass. CPU construction recomputed 7,929,284 parameters for `absent` and 8,004,372 for `palette_index`, an independently verified difference of 75,088. The absent model owns no auxiliary modules, parameters, state, optimizer members, or EMA keys; enabled heads remain physically present even at zero auxiliary loss. Architecture hashes are stable, cross-mode distinct, and independent of loss-only changes.

Safe resume, campaign fairness, and execution safety fail certification. Jointly missing resume-hard fields are accepted; most manifest claims are not rebound to the actual runtime; an unsafe low-level path can omit its revocation record. Campaign identity is optional, commands and resolved configs are not bound, launch authorization is not an execution gate, identity hashes are syntactic, protected fields can be waived too broadly, and incomplete aggregation can report complete. Training remains blocked.

## 5. Memorization integration result

The isolated detector-v2 and promotion-decision contracts pass their focused tests, and a fully synthetic bundle is structurally accepted by the real decision CLI. That `eligible` result is a compatibility probe only and does not authorize any real checkpoint. The historical checkpoint remains `blocked / not_comparable`.

Combined integration fails certification. Review-required detector evidence makes the suite machine gate fail, while the decision layer makes any machine failure irreversible, so a valid clearing review cannot unlock a real report. Detector and decision evidence-class names disagree; the production detector-policy hash is not enforced; `candidate_evidence_sha256` hashes only a pair; machine-pass semantics are trusted rather than recomputed; and direct training-image verification is optional. Checkpoint promotion remains blocked.

## 6. Human-visible capability snapshot

The operator snapshot at `experiments/repo_capability_snapshot_v1/` distinguishes implemented, tested, independently audited, and safe-to-run capabilities. Its PowerShell script runs only help, read-only verification, contract validation, report discovery, and optional focused CPU tests. It never launches a GUI, provider call, generation, training, production freeze, or promotion.

## 7. Tests and warnings

Phase-6 targeted selections all pass: Labeling-v4 64, Dataset-v5 91, memorization/promotion 47, and training infrastructure 51 (253 targeted passes total). The full repository suite passes with 1,885 tests. No required test failed or skipped. The only warning is the already-known PyTorch warning in `tests/test_ml_overfit_smoke.py::test_masked_loss_scalar` about converting a `requires_grad` tensor to a scalar; no new warning appeared. A direct `.ps1` launch was blocked by this host's Windows execution policy; the tested `powershell.exe -ExecutionPolicy Bypass -File` form passes and is used below.

Scoped Ruff and formatting checks pass for every changed Python file and every audited Python scope. `git diff --check` passes.

## 8. Commits created

1. `9795eb6` - Fix remaining Labeling-v4 two-pass safety defects
2. `861b9f1` - Implement Dataset-v5 named view and freeze tooling
3. `SELF` - Add infrastructure audits and capability snapshot

The third SHA cannot be embedded inside the commit that defines it. Resolve it after the commit with `git rev-parse HEAD`; the exact SHA is also returned in the final response.

## 9. Anything blocked or incomplete

Production remains blocked in dependency order by: fresh independent Labeling-v4 audit; real human quality completion and source-bound semantic preparation; fresh independent Dataset-v5 builder audit and final policy decisions; final frozen Dataset-v5 and model-config hashes; training resume/campaign/execution remediation; evaluation-contract and memorization-integration implementation/certification; a ready three-seed campaign; real training; the 54-output benchmark plus bound v2 reviews; and only then a promotion decision. No checkpoint promotion is authorized.

## 10. Exact next human action

From the repository root, inspect the safe state and report index with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\repo_capability_snapshot_v1\safe_commands.ps1
```

Then assign a fresh independent Labeling-v4 audit before starting any real Wave-1 review.
