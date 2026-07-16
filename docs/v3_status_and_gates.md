# Sprite Lab v3 status and gates

The unified state uses these controlled values: `NOT_STARTED`, `READY`, `RUNNING`, `PAUSED`, `NEEDS_REVIEW`, `BLOCKED`, `FAILED`, `COMPLETE`, `INCONCLUSIVE`, and `STALE`.

Every stage includes a human explanation, blockers, warnings, evidence paths and SHA-256 hashes, source identity, next action, exact command, and safe-resume availability. A directory existing is never enough to mark a stage complete.

Tracked stages are:

1. Raw-source provenance
2. Extraction
3. Suitability
4. Semantic labeling
5. Semantic calibration
6. Dataset-v5 view construction
7. Dataset freeze
8. Training-infrastructure audit
9. Training campaign
10. Evaluation generation
11. Evaluation metrics
12. Memorization review
13. Promotion decision

## Three different questions

`status` deliberately separates:

- implementation readiness: whether the repository has the machinery;
- independent audit: `PASS`, `FAIL`, `INCONCLUSIVE`, `STALE`, or `NOT AUDITED`;
- production authorization: whether every bound identity, safety gate, and explicit policy permits an action.

An implementation may exist while its audit fails and production remains blocked. A stale pass never authorizes an action. A stale failure remains historical evidence but is not represented as a current certification.

Training audit applicability is checked by hashing every file bound in `frozen_hash_verification.json`. Memorization applicability is checked against the recorded commit over the evaluation subsystem. If those identities change, status reports `STALE AUDIT`/`STALE` and returns exit code 6 when a state-changing command depends on it.

The integrated evidence currently establishes a complete raw-source disposition, a stopped blind-label health gate, failed training-infrastructure audit gates, and a failed memorization/review-integrity audit. Therefore Dataset-v5 production freezing, training, generation, and promotion remain unauthorized.
