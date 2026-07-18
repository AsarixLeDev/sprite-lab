# Product evaluation

The evaluation product feature presents verified Sprite Lab checkpoints, runs the Standard Sprite Lab benchmark through the existing evaluation backend, and projects its real report data into a user-facing dashboard.

## Registration

The feature is a `ProductPlugin` and deliberately does not edit the central registry or web shell:

```python
from spritelab.product_features.evaluation import build_plugin

plugin = build_plugin()
```

An integration layer supplies `plugin` to `spritelab.v3.cli.main(..., plugins=[plugin])` and/or `create_product_app(..., plugins=[plugin])`. The plugin registration function is `build_plugin()`. Its CLI registration callback replaces the reserved `v3 eval` command through the existing feature-owned registry contract, making the composed command:

```text
python -m spritelab v3 eval
```

## Checkpoint selection

Candidates come only from canonical `spritelab.v3.run-state.v1` training run state. The normal selector includes complete, local, verified checkpoints whose dataset identity matches the active project identity. It displays:

- a friendly run name and date;
- training profile and completion state;
- a redacted dataset identity summary;
- checkpoint step and live/EMA variant;
- verification state.

Incomplete, invalid, foreign, unsafe-resume, stale-dataset, unverified, and missing checkpoints are excluded from the normal selector. The ordinary checkpoint API is always pathless; a `technical_details` query cannot upgrade it. The advanced technical endpoint preserves the controlled reason and reveals identities/paths only after explicit acknowledgement. The default is the newest complete checkpoint, preferring EMA for an otherwise identical run and step.

Checkpoint discovery requires both active dataset and training-view identities. A complete checkpoint is not production-eligible when either active identity is missing, empty, padded, or malformed; callers cannot obtain an unbound fallback catalog.

## Evaluation execution

Opening `/evaluation` performs no generation. `Start evaluation` is an explicit action. `Validate plan` is a dry run and records zero generation runs and zero promotion actions.

The stage contract is:

1. Checkpoint validation
2. Benchmark validation
3. Generation
4. Structural metrics
5. Conditional metrics
6. Diversity
7. Palette analysis
8. Memorization detector
9. Review completeness
10. Promotion decision report

The product service invokes `spritelab.evaluation.suite.score_suite` for evaluation metrics after a typed generation adapter has produced benchmark samples. A failed stage becomes visible and blocks downstream stages without discarding completed evidence. Remote or billable generation requires explicit confirmation.

## Dashboard

The dashboard contains stage progress, metric cards, distributions, category results, permitted source aggregates, a filterable sample gallery, a checkpoint comparison projection, memorization and review state, the final gate summary, and downloadable JSON report data. Charts use per-image report rows; when rows or a metric are missing they show an explicit no-data state.

Source aggregates are opt-in and use public source identities only. Gallery records never return training/source filesystem paths. Checkpoint paths are reduced to a public artifact name outside technical details.

Evaluation action flags accept exact JSON booleans only. Strings such as `"false"`, numbers, arrays, and objects are rejected before generation, so they cannot satisfy explicit-action, billable-confirmation, dry-run, or source-result gates. Generator and evaluator exception details remain private; durable stages and public API errors use fixed pathless, credential-free messages.

Comparisons require identical metric-definition identities. Reports with different explicit definitions, schemas, thresholds, detector policies, comparison methods, or parameter identities are rejected before any average or delta is calculated.

The same compatibility gate applies to the standalone paired-report comparison. An incompatible comparison is refused before its output directory is created.

## Memorization and promotion safety

The display vocabulary includes `Hard evidence`, `Review required`, `Warnings`, `Cleared by valid review`, and `Not comparable`. Hard evidence never exposes a clear action. A clearing review is displayed as authoritative only when it is a valid bound review event with an explicit matching event hash and the complete log replays without integrity errors. Legacy, unsigned, malformed, incomplete, or competing reviews cannot clear a product display state.

Review work links through the common review contract at `/review?queue=memorization`; this feature does not create a new review log or write the existing one.

The failed memorization-integration audit is not bypassed. The page shows `Promotion integrity is not currently certified.` and the promotion display always has an empty action list. No method in this plugin promotes, authorizes promotion, or mutates a checkpoint registry.

A memorization audit candidate must use the strict independent-audit v1 envelope. The loader is bounded and rejects duplicate keys, non-finite values, a non-object root, and malformed schemas consistently across v3, developer, and Product status. The envelope self-consistency checks cover the declared dataset/view, freeze, campaign, checkpoint and weights, benchmark, metric definition, detector/promotion policy, candidate evidence, review log, semantic code identity, auditor identities, operation, and embedded receipt.

The embedded receipt is not trusted merely because it declares itself server-managed or hashes consistently with the report. Until a separate service-owned immutable receipt/action record and a resolver for every current audit-subject binding exist, every otherwise well-formed v1 report is `NOT_COMPARABLE` with `trusted_audit_receipt_unavailable`; it cannot produce current certification, a PASS authority result, or a promotion action. Legacy or minimal code-identity-only documents remain stale.

Passive reconstruction reports stale evaluation artifacts in memory without rewriting durable run state. Reconciliation and durable transitions remain explicit-action responsibilities.
