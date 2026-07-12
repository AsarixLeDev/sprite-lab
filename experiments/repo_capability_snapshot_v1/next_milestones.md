# Next milestones and dependency order

No step below authorizes the next one merely by being completed. Each transition requires exact hashes, a clean/current code identity, and an explicit pass from the named audit or human gate.

| Order | Milestone | Required result before advancing |
|---:|---|---|
| 1 | Independently audit final Labeling-v4 two-pass remediation | Fresh audit binds the final commit and evidence, reruns adversarial resolution/queue tests, and passes without relying on the remediation's self-report. |
| 2 | Remediate and re-audit memorization integration | Align exact-match classifications, enforce detector-policy SHA, bind complete candidate evidence, make valid review clearing coherent with machine gates, and reject internally inconsistent bundles. |
| 3 | Resolve production Dataset-v5 policy decisions | Explicitly approve or reject raking `0.20`, quality/calibration rules, open-set taxonomy, source-OOD scope, mushroom policy, and evaluation coverage. Candidate reports alone are insufficient. |
| 4 | Independently audit the Dataset-v5 named-view builder | Bind the final code, six contract files, frozen-r2 identity, source manifests, relation detector, policies, and synthetic repeat/tamper evidence. Production authorization must remain absent during the audit. |
| 5 | Conduct real Labeling-v4 review and calibration | Only after step 1: run quality review, freeze the bound inference queue, perform approved semantic work, and create calibrated human truth without provider use unless separately authorized. |
| 6 | Build, verify, approve, and freeze final Dataset-v5/evaluation views | Requires steps 3–5, complete provenance/license/leakage checks, calibrated supervision, exact evaluation identities, and a fresh production authorization record. |
| 7 | Repeat training infrastructure certification with concrete identities | Re-audit safe resume, all resume-hard fields, three-seed fairness, campaign artifact completeness, and zero-launch safety against the final Dataset-v5 and evaluation hashes. |
| 8 | Explicitly authorize and run the real three-seed campaign | Requires a passing step 7 plus separate human/GPU authorization. Fixed steps, seeds, schedules, checkpoints, EMA/live policy, and output roots must match the certified manifest. |
| 9 | Generate and review the frozen 54-output benchmark | Requires complete eligible campaign artifacts; bind every output, detector report, policy hash, candidate pair, and v2 review event. |
| 10 | Make a new promotion decision, then separately authorize promotion | Requires steps 2 and 9, all machine gates, complete current reviews, exact identities, and an independently audited eligible decision. The decision report is not the promotion action. |

The safe immediate command is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\experiments\repo_capability_snapshot_v1\safe_commands.ps1
```

The first substantive human gate is step 1: commission a fresh independent Labeling-v4 audit. Steps 2–4 can be prepared in parallel, but real human review begins only after step 1 passes, and training remains downstream of step 7.
