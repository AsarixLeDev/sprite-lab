# Blocked commands and operations

These restrictions are fail-closed. A command being present in `--help` does not make it authorized. Do not remove an item until a fresh audit explicitly closes its gate and binds the exact inputs, code identity, and evidence.

| Blocked operation | Representative command class (do not run) | Why it remains blocked |
|---|---|---|
| Real quality GUI | `python -m spritelab harvest assisted-v4 ... --mode quality-only` on Wave-1 | Final Labeling-v4 remediation has local tests and rehearsal only; a fresh independent audit is still required. |
| Real inference queue freeze | `python -m spritelab harvest label-v4-freeze-inference-queue ...` on real truth | Real human truth does not yet exist and the final two-pass binding behavior is not independently certified. |
| Provider-backed semantic inference | `python -m spritelab harvest label-v4-prepare-audit ... --allow-provider-calls` | Network/provider calls and credit use are prohibited; the real quality gate and queue are not ready. |
| Production-v5 freeze | `python -m spritelab.dataset_v5.cli freeze-view ...` on a production view | Named-view tooling lacks the fresh builder audit, approved production policy bundle, calibrated truth, complete evaluation identities, and production authorization record. |
| 54-output benchmark generation | `python -m spritelab eval generate-suite ...` for the frozen 54-output suite | No eligible three-seed campaign/checkpoint exists; generation is prohibited in this milestone. |
| Real campaign execution | `python -m spritelab train campaign-run ... --execute --confirm-execute` | Final Dataset-v5 and evaluation identities are unbound, and the infrastructure audit does not certify safe resume, fairness, or execution. |
| Checkpoint promotion | Any checkpoint copy, registry update, deployment, or release action | Combined memorization integration fails: class semantics, policy-hash enforcement, evidence binding, and review-clearing behavior are not safe. No checkpoint is eligible. |

`promotion-decision` writes a diagnostic decision report; it does not itself copy or promote a checkpoint. Until the integration defects are remediated and re-audited, it is restricted to synthetic or historical fail-closed diagnostics and cannot be treated as authorization.
