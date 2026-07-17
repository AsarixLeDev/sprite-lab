# Product training

The Training feature is a feature-owned `ProductPlugin`. Integration code registers it by calling:

```python
from spritelab.product_features.training import build_plugin

plugin = build_plugin()
```

The feature does not register itself globally and does not modify the product shell. Its navigation item points to `/training`, and the existing command remains:

```text
python -m spritelab v3 train
```

## Routine experience

The normal page asks for a profile and compute target, not campaign manifests, hashes, random seeds, output paths, or checkpoint paths. Advanced configuration is a collapsed `<details>` section. The four profile names are `recommended`, `fast_preview`, `quality`, and `custom`.

A profile is only a selector. It selects an existing campaign specification from `training.campaign_config` under `product_profiles.<profile>.campaign` or `campaign_path`. The feature passes that specification to the existing `plan_campaign`, `validate_campaign`, `audit_resume`, and `execute_campaign` functions. It does not copy optimizer, schedule, loss, determinism, checkpoint, or resume semantics.

The recommended profile may also select a campaign document directly when the configured document has no `product_profiles` mapping. A custom request must exactly match the configured `product_profiles.custom` campaign; it is not an unbound launch-time override.

## Preparing an accepted dataset

The Training page can publish an optional immutable image-only baseline after one explicit confirmation. This path verifies source hashes, encodes canonical 32x32 arrays, runs dataset and training-manifest QA, checks the trainer loader when available, freezes a deterministic generic vocabulary, and validates a non-launchable three-seed campaign draft.

The baseline is never a production freeze. Its manifest records `artifact_kind: image_only_baseline`, `production_authorized: false`, `training_eligible: false`, and `activation_forbidden: true`. Its campaign draft records both `executable: false` and `launch_authorized: false`. Preparation does not write `dataset.view_manifest`, `dataset.freeze_manifest`, `training.dataset_freeze`, `training.campaign_config`, `execution.allow_dataset_production_freeze`, or `execution.allow_training` and therefore cannot make **Start training** available.

Preparation writes content-addressed artifacts under `.spritelab/training-preparation/baseline-*`, binds their identity to stable item IDs, verified source bytes, and exact preparation-recipe source hashes, and records a whole-publication hash and byte-size inventory. Semantic and hierarchical proposals are intentionally not consumed by this baseline. Persisted artifacts and API responses do not expose absolute local paths. Reuse reconstructs the expected publication independently and refuses changed, re-signed, or semantically promoted content.

Background state is durable under `.spritelab/training-preparation/jobs`. It binds source, config, code, input, result, worker, and job identities. Cross-process locking refuses concurrent starts, a live worker is not interrupted by a second server process, and a dead worker is reconstructed as safely retryable. The privacy-safe event history is atomically rewritten and capped at 200 records. Linked, reparse-point, or hard-linked state and history entries fail closed.

## Conditioned Dataset-v5 activation contract

The web start action fails closed until the exact selected profile satisfies `spritelab.training.conditioned-dataset-contract.v2`. `dataset.freeze_manifest` and `training.dataset_freeze` must contain the same canonical project-relative path; absolute paths, traversal, links, reparse points, and hard links are rejected.

The activation manifest uses `spritelab.dataset.freeze.conditioned.v5`, declares Dataset version 5, `dataset_kind: conditioned`, semantic-label dependence, complete production authorization, and 2,000 through 3,000 images. Its exact artifact set is the view manifest, split manifest, conditioning vocabulary, benchmark manifest, independent labeling audit, and validation report. Every binding records a canonical publication-relative path, SHA-256, and byte count. The full publication inventory records the same data for every regular file other than the activation manifest, plus an exact file count, total bytes, and canonical inventory identity. Unexpected, missing, linked, changed, or re-signed entries invalidate activation.

The selected recommended, quality, or custom campaign must be executable and launch-authorized with exactly three standard seeds and 5,000 optimizer steps. It binds the activation bytes plus the exact view, split, vocabulary, and benchmark identities. The campaign config lives outside the frozen publication, avoiding a campaign/freeze hash cycle. `build_conditioned_three_seed_campaign` is the shared no-write builder for the Dataset-v5 publisher.

This structural contract does not replace the independent training-infrastructure audit. Applicable reports use `spritelab.training.infrastructure-audit.v2` and `spritelab.training.infrastructure-audit-hashes.v2`, bind the exact activation manifest, campaign config, campaign identity, and training-code identity, and contain exactly these 18 `PASS` gates: tracked code inventory; no untracked production Python; dataset/view/freeze/campaign/vocabulary identity; dataset and training-manifest QA; production-loader coverage; campaign/experiment compatibility; CPU/CUDA smoke evidence; CUDA/driver/Torch/device compatibility; determinism environment qualification; launch-receipt/execution binding; backend command safety; idempotency/concurrency refusal; output-root/resume safety; event-history/migration identity; publication/config atomicity and restart; filesystem containment and link defenses; API/UI privacy; and curated/full test results.

Audit applicability re-hashes every recorded file and dynamically scans all training-bound production roots. Adding even one untracked production `.py` file makes a prior `PASS` stale. Start and every safe resume re-load the selected profile, activation, campaign, audit, and code bindings before any backend operation. Baseline preparation cannot manufacture or activate this contract.

## Pre-activation exploratory smoke registration

The Evaluation Playground exposes a separate, web-operated infrastructure-smoke lane for a completed conditioned publication before configuration activation. The server derives all six publication bindings from the selected conditioned job and prepares immutable plan/config/manifest artifacts under `artifacts/training/smokes/<smoke-id>`. Its CPU and CUDA outputs are fixed under `runs/v3/training-smokes/<smoke-id>/<device>`; every full 5,000-step campaign root remains an absent sentinel.

Explicit CSRF-protected POST actions run CPU and then CUDA, never concurrently. Execution uses a fixed argument array with `shell=False`; CPU binds `CUDA_VISIBLE_DEVICES=-1` and `SPRITELAB_PROGRESS=0`, while CUDA binds `CUDA_VISIBLE_DEVICES=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `SPRITELAB_PROGRESS=0` before Python starts. Mutable execution state is atomically published beside the immutable plan, retains bounded path-scrubbed log tails, and reconstructs a final receipt after restart. A dead owner without a receipt is `INTERRUPTED` and the bundle is permanently nonresumable.

Each child receives the exact plan-bound minimal environment: provider credentials, `PYTHONPATH`, user-site startup, and unrelated host variables are excluded, while temporary and cache paths point inside the bundle. Worker and trainer start with `-I -B`; a stdlib-only preflight verifies the immutable plan, environment, exact interpreter bytes, orchestration sources, and the complete production-Python inventory before any Sprite Lab import. A bound source loader then rechecks every imported `spritelab` source file, so a source change between preflight and import fails closed.

The exact interpreter target remains open across both process creations. On Windows each child is created suspended, assigned to a kill-on-close Job, verified against the held executable, and only then resumed. On Linux each child executes the held `/proc/self/fd` target with a parent-death signal and post-`prctl` parent-race check. The outer worker publishes launch-bound heartbeats and a terminal outcome; a receipt alone never marks a still-running worker complete, and any failed image, containment, heartbeat, outcome, or receipt check requires a fresh bundle.

Registration revalidates the current publication, freeze, campaign, full training-code identity, real configuration hash, absent campaign roots, device environment, reports, finite step-2 metrics, strict CUDA qualification, and every checkpoint through weights-only loading. It reads both immutable receipts server-side and snapshots the CUDA live/EMA pair beneath `runs/v3/playground/exploratory-checkpoints/<content-id>`. The exploratory catalog is never merged into the production Evaluation catalog and records production, evaluation, resume, campaign-execution, and promotion eligibility as false. Before activation it uses prospective activation validation without an audit; after activation it remains available only when the exact activated config has a current applicable `PASS` infrastructure audit.

The command argv is displayed only in a transparency disclosure. Manual CLI execution, browser-supplied paths, hash transcription, receipt pasting, resume, and promotion are not part of the workflow. Passive page/catalog/status reads launch no subprocess, import no Torch, initialize no CUDA, and create no directories.

## Mandatory launch checks

Before any backend `prepare`, `upload`, or `launch` call, the feature verifies:

- the exact conditioned Dataset-v5 activation contract and freeze-to-campaign hash binding;
- dataset and split identity bindings through campaign validation;
- applicability and `PASS` status of the independent training-infrastructure audit;
- campaign identity and resolved configuration identity;
- project and campaign launch authorization;
- fresh, owned, complete, or safely resumable output roots;
- safe-resume identity and checkpoint schedule rules;
- disk requirements when the campaign/backend supplies an estimate;
- backend device/environment capability without importing Torch or initializing CUDA;
- the complete campaign artifact and completion-marker contract.

The existing repository is intentionally blocked: its Dataset-v5 production freeze is absent, the training-infrastructure audit contains failed gates, `execution.allow_training` is false, and no campaign config is selected. The Training page reports these blockers and launches nothing. This feature does not fix or bypass that audit.

Cloud backends require a fresh `confirm_cloud: true` on the start request after the plan has passed. Confirmation is checked before preparation. Page load only returns status data; it does not run a connection test or allocate a resource. Connection tests use their own explicit endpoint.

## Dashboard and events

Local and hosted jobs use the same `spritelab.product.event.v1` `ProductEvent` schema. The dashboard aggregates:

- campaign and per-seed progress;
- optimizer step and total steps;
- training and validation loss curves;
- learning rate and optional gradient norm;
- optional GPU utilization and VRAM;
- estimated completion;
- checkpoint schedule and timeline;
- logs, warnings, pause state, and safe-resume availability;
- remote resource uncertainty, possible continuing cost, and shutdown guidance.

Remote checkpoints are not safe-resume points until the backend identity is verified and the artifact has been downloaded and hash-verified. Unsafe resume is never exposed.

## Intermediate previews

`PreviewScheduler` runs only at the intersection of the configured checkpoint schedule and preview interval. Prompts and generation seeds are fixed in `PreviewConfiguration`; each event records the checkpoint, training seed, prompt, generation seed, parameters, and output path. Outputs live under the run's `previews/checkpoint_<step>/seed_<seed>/` directory.

Every preview is marked `exploratory: true`, `benchmark_evidence: false`, and `promotion_evidence: false`. Preview generation can be disabled. A preview exception emits a warning event and never changes the training status.

## Feature routes

- `GET /training/api/preparation` - passive background-preparation state and privacy-safe logs.
- `POST /training/api/preparation` - explicitly authorized immutable image-only baseline preparation using `authorize_baseline: true`; rejects freeze or training authorization and starts no training process.

- `GET /training` — page shell; no backend probe or launch.
- `GET /training/api/state` — current plan, exact-profile conditioned activation contract, and blockers; device check deferred.
- `GET /training/api/settings` — redacted compute settings.
- `POST /training/api/connection-test` — explicit backend probe.
- `POST /training/api/start` — revalidates the selected profile, custom input, activation, audit, and campaign before backend work.
- `GET /training/api/runs/{run_id}` — refreshes events and dashboard data.
- `POST /training/api/runs/{run_id}/pause` — graceful interruption request.
- `POST /training/api/runs/{run_id}/resume` — revalidates the retained activation and campaign, then performs safe resume only.
