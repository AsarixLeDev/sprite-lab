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

The recommended profile may also select a campaign document directly when the configured document has no `product_profiles` mapping. Custom configuration is intentionally an advanced API input and is still validated as a normal backend campaign.

## Preparing an accepted dataset

The Training page can prepare the active accepted product dataset after two separate, explicit choices:

- freeze authorization publishes an immutable image-only view after source-hash verification, canonical 32x32 encoding, dataset QA, training-manifest QA, and a production-loader check;
- training authorization sets `execution.allow_training: true`, but does not launch anything and does not satisfy the independent audit gate.

Preparation writes content-addressed artifacts under the project, binds their identity to stable item IDs, verified source bytes, and the exact preparation-recipe source hashes, records a whole-publication hash inventory, freezes a deterministic generic image-only conditioning vocabulary, validates the recommended campaign through `plan_campaign` and `validate_campaign`, and updates `spritelab.yaml` atomically. Semantic and hierarchical proposals are not consumed by this image-only path. Persisted preparation artifacts and API responses do not expose absolute local paths. Reuse refuses a publication when any inventoried artifact changed. A concurrent preparation request is refused while the active background job continues.

Campaign resolved configurations are complete `spritelab_experiment_config_v1` documents. They are validated through the same experiment-manifest loader used by `train experiment run`; they are not campaign-only documents passed to an incompatible command.

## Mandatory launch checks

Before any backend `prepare`, `upload`, or `launch` call, the feature verifies:

- production dataset freeze and its authorization;
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
- `POST /training/api/preparation` - explicitly authorized image-only preparation; starts no training process.

- `GET /training` — page shell; no backend probe or launch.
- `GET /training/api/state` — current plan and blockers; device check deferred.
- `GET /training/api/settings` — redacted compute settings.
- `POST /training/api/connection-test` — explicit backend probe.
- `POST /training/api/start` — revalidates every gate and starts only an authorized campaign.
- `GET /training/api/runs/{run_id}` — refreshes events and dashboard data.
- `POST /training/api/runs/{run_id}/pause` — graceful interruption request.
- `POST /training/api/runs/{run_id}/resume` — safe resume only.
