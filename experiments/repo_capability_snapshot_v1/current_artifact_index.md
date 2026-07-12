# Current artifact index

Paths are relative to the repository root. Start with the Markdown report; use adjacent JSON only when exact machine fields or hashes are needed.

| Area | Human-readable report | Current interpretation |
|---|---|---|
| Labeling-v4 final remediation | `experiments/label_v4_calibration_wave1/two_pass_remediation_v2/remediation_report.md` | Local remediation passes; fresh independent audit required. |
| Labeling-v4 prior audit | `experiments/label_v4_calibration_wave1/two_pass_implementation_audit_v2/audit_report.md` | Historical audit context; it does not certify the later final remediation. |
| Labeling-v4 100-record rehearsal | `experiments/label_v4_calibration_wave1/two_pass_remediation_v2/synthetic_rehearsal_report.json` | Synthetic only: 100 quality events, 80 queue members, 20 exclusions, zero providers. |
| Dataset-v5 readiness | `experiments/v5_readiness_audit_v1/audit_report.md` | Production freeze blocked; report predates the named-view implementation. |
| Dataset-v5 named-view implementation | `experiments/v5_named_view_builder_v1/implementation_report.md` | Seven synthetic views pass; production remains blocked pending independent builder audit. |
| Dataset-v5 contract | `experiments/v5_view_contract_v1/contract_report.md` | Contract source for named views; unresolved production decisions remain adjacent. |
| Weighting audit | `experiments/v5_weighting_policy_audit_v1/audit_report.md` | Policy analysis; not a production approval. |
| Raking 0.20 candidate | `experiments/v5_weighting_candidate_raking020_v1/candidate_report.md` | Preferred audited candidate, explicitly candidate-only and promotion-forbidden. |
| Evaluation design | `experiments/v5_evaluation_design_v1/design_report.md` | Candidate design; evaluation identities and calibration are not final. |
| Membership diversity audit | `experiments/v5_membership_diversity_audit_v1/audit_report.md` | Documents membership gaps and candidate policies. |
| Training readiness | `experiments/training_readiness_audit_v1/audit_report.md` | Historical readiness/blocker baseline; does not authorize training. |
| Training/evaluation gate contract | `experiments/training_evaluation_gate_contract_v1/contract_report.md` | Gate design only; final identities remain unavailable. |
| Headless architecture remediation | `experiments/training_headless_architecture_remediation_v1/remediation_report.md` | Implemented and locally validated; independent certification still required. |
| Campaign orchestration | `experiments/training_campaign_orchestration_v1/remediation_report.md` | Three-seed mechanism implemented; real campaign remains blocked. |
| Memorization detector | `experiments/memorization_detector_hardening_v1/remediation_report.md` | Isolated detector hardening; combined integration still fails. |
| Memorization promotion decision | `experiments/memorization_promotion_remediation_v1/remediation_report.md` | Fail-closed decision layer; combined integration defects block reliance on eligibility. |
| Historical promotion result | `experiments/memorization_promotion_remediation_v1/historical_decision.md` | Historical Phase-1 checkpoint is not eligible. |
| Memorization full-suite readiness | `experiments/memorization_full_suite_readiness_audit_v1/audit_report.md` | Documents 54-output and review requirements; no benchmark generation is authorized. |

The current Phase-3 and Phase-4 integration audit evidence is assembled elsewhere in this milestone. Its operative verdicts are already conservative here: training and promotion remain blocked. Consult the final milestone report for their finalized evidence paths and hashes.
