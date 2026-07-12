# Training Infrastructure Audit v2

## Result

**Do not authorize training.** The physical headless architecture and its measured architecture identity pass this read-only CPU audit. Safe resume, campaign fairness, and execution safety fail because the current contracts have fail-closed gaps that the focused passing tests do not cover.

| Area | Verdict | Evidence |
|---|---|---|
| Headless architecture | PASS | `absent` has zero auxiliary modules, parameters, state keys, optimizer members, and EMA keys. |
| Architecture identity | PASS | Repeated construction is stable, the two physical modes differ, and loss-only settings do not define the physical identity. |
| Safe resume | FAIL | Jointly absent hard fields compare equal; low-level unsafe resume can bypass without a durable unsafe record. |
| Campaign fairness | FAIL | Commands/config bytes and physical cell configs are not bound; protected fields can be declared away; several required settings are optional. |
| Execution safety | FAIL | Flag and blocked-plan gates work, but an otherwise ready plan can carry an arbitrary unbound command and may be executable while launch authorization is false. |

## Recomputed architecture facts

The production constructor was run on CPU with `vocab_size=64`, `embed_dim=64`, `base_channels=64`, `channel_mults=(1,2,4)`, and two residual blocks per level.

- `absent`: 7,929,284 parameters, 151 parameter/state/EMA keys, 0 auxiliary parameters.
- `palette_index`: 8,004,372 parameters, 161 parameter/state/EMA keys, 75,088 auxiliary parameters.
- Measured difference: **75,088**, exactly equal to physically owned auxiliary parameters.
- Repeated architecture hashes were stable and differed across modes.

## Resume finding

All 21 populated resume-hard fields reject a changed value, and a recorded unsafe override captures all mismatches while revoking exact replay, fair comparison, and promotion eligibility. That does not make the API fail closed: `{}` safely resumes to `{}`, and `unsafe=True` without `unsafe_record` bypasses mismatches without recording the reason or revocations. The safe-resume verdict is therefore FAIL.

## Campaign finding

The three unique seed rule, placeholder blocking, fixed-step schedule helpers, foreign-root rejection, unsafe-resume rejection, missing-artifact detection, and deterministic three-seed aggregation work in their tested forms. The campaign is still not a trustworthy execution identity. `campaign_identity` is optional; `experiment_command` and `resolved_config_path` are excluded from run identity; resolved config bytes are never hashed; identity hashes are only syntax checked; protected fields can be waived by declaring them experimental; CFG and sampling steps are optional; arbitrary epoch fields are accepted; physical architecture configs are not bound to cell labels; artifact and resumability checks are shallow; and an entire cell can be omitted from aggregation.

## Execution safety

No real launch occurred during this audit: training runs started = 0, campaign subprocess launches = 0, CUDA initialized = false, provider calls = 0. Plan/validate/status and missing-flag/blocked-plan paths do not launch in the covered tests. The overall verdict remains FAIL because a ready manifest can bind neither its command nor resolved config bytes and can be executable despite `baseline_launch_authorized=false`.

## Validation

Focused suites: 51 passed (18 architecture/resume and 33 campaign), 0 failed, 0 skipped. Scoped Ruff check and format check passed on all six audited files. Production code and tests were not modified.

Training remains blocked until these defects are remediated and independently re-audited, final frozen Dataset-v5/model/evaluation identities exist, memorization integration is certified, and a ready three-seed campaign is bound end to end.
